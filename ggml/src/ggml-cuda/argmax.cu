#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    // 0, not -1: NaN comparisons never update the running max, so an all-NaN row
    // would otherwise return -1 - which downstream gathers use as a row index
    int   argmax = 0;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

static constexpr int TP_TOP1_THREADS = 256;

static __global__ void tp_top1_stats_f32(
        const float * __restrict__ logits,
        float * __restrict__ stats,
        const int32_t ncols,
        const int32_t rank,
        const int32_t nranks,
        const int32_t global_offset,
        const bool need_probability) {
    __shared__ float shared_values[TP_TOP1_THREADS];
    __shared__ int32_t shared_ids[TP_TOP1_THREADS];
    __shared__ int32_t shared_finite[TP_TOP1_THREADS];

    for (int32_t i = threadIdx.x; i < 4*nranks; i += blockDim.x) {
        stats[i] = 0.0f;
    }

    float max_logit = -FLT_MAX;
    int32_t max_id = global_offset;
    bool all_finite = true;
    for (int32_t i = threadIdx.x; i < ncols; i += blockDim.x) {
        const float value = logits[i];
        if (!isfinite(value)) {
            all_finite = false;
            continue;
        }
        const int32_t id = global_offset + i;
        if (value > max_logit || (value == max_logit && id < max_id)) {
            max_logit = value;
            max_id = id;
        }
    }

    shared_values[threadIdx.x] = max_logit;
    shared_ids[threadIdx.x] = max_id;
    shared_finite[threadIdx.x] = all_finite;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int32_t i = 1; i < blockDim.x; ++i) {
            const float value = shared_values[i];
            const int32_t id = shared_ids[i];
            if (value > max_logit || (value == max_logit && id < max_id)) {
                max_logit = value;
                max_id = id;
            }
            all_finite = all_finite && shared_finite[i];
        }
        shared_values[0] = max_logit;
        shared_ids[0] = max_id;
        shared_finite[0] = all_finite;
    }
    __syncthreads();

    float local_sumexp = 0.0f;
    if (need_probability && shared_finite[0]) {
        for (int32_t i = threadIdx.x; i < ncols; i += blockDim.x) {
            local_sumexp += expf(logits[i] - shared_values[0]);
        }
        shared_values[threadIdx.x] = local_sumexp;
        __syncthreads();
        if (threadIdx.x == 0) {
            local_sumexp = 0.0f;
            for (int32_t i = 0; i < blockDim.x; ++i) {
                local_sumexp += shared_values[i];
            }
            shared_values[0] = local_sumexp;
        }
        __syncthreads();
        local_sumexp = shared_values[0];
    }

    if (threadIdx.x == 0) {
        stats[              rank] = max_logit;
        stats[  nranks +    rank] = need_probability ? local_sumexp : 0.0f;
        stats[2*nranks +    rank] = shared_ids[0];
        stats[3*nranks +    rank] = shared_finite[0] ? 1.0f : 0.0f;
    }
}

static __global__ void tp_top1_select_f32(
        const float * __restrict__ stats,
        float * __restrict__ result,
        const int32_t nranks,
        const bool need_probability) {
    if (threadIdx.x != 0) {
        return;
    }

    bool valid = true;
    float max_logit = -FLT_MAX;
    int32_t max_id = 0;
    for (int32_t rank = 0; rank < nranks; ++rank) {
        const float rank_max = stats[rank];
        const float rank_id_f = stats[2*nranks + rank];
        const float rank_finite = stats[3*nranks + rank];
        valid = valid && isfinite(rank_max) && isfinite(rank_id_f) && rank_finite == 1.0f;
        const int32_t rank_id = (int32_t) rank_id_f;
        if (rank_max > max_logit || (rank_max == max_logit && rank_id < max_id)) {
            max_logit = rank_max;
            max_id = rank_id;
        }
    }

    float value = 1.0f;
    if (valid && need_probability) {
        float denominator = 0.0f;
        for (int32_t rank = 0; rank < nranks; ++rank) {
            denominator += stats[nranks + rank]*expf(stats[rank] - max_logit);
        }
        valid = isfinite(denominator) && denominator > 0.0f;
        value = valid ? 1.0f/denominator : nanf("");
    }

    result[0] = max_id;
    result[1] = valid ? value : nanf("");
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}

void ggml_cuda_tp_top1_stats(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[0] <= INT32_MAX);

    const int32_t rank = ggml_get_op_params_i32(dst, 0);
    const int32_t nranks = ggml_get_op_params_i32(dst, 1);
    const int32_t global_offset = ggml_get_op_params_i32(dst, 2);
    const bool need_probability = ggml_get_op_params_i32(dst, 3);

    tp_top1_stats_f32<<<1, TP_TOP1_THREADS, 0, ctx.stream()>>>(
        (const float *) src0->data, (float *) dst->data, (int32_t) src0->ne[0],
        rank, nranks, global_offset, need_probability);
}

void ggml_cuda_tp_top1_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t nranks = ggml_get_op_params_i32(dst, 0);
    const bool need_probability = ggml_get_op_params_i32(dst, 1);

    tp_top1_select_f32<<<1, 1, 0, ctx.stream()>>>(
        (const float *) src0->data, (float *) dst->data, nranks, need_probability);
}
