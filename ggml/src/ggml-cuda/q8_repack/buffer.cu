// Repack buffer type: on upload, supported Q8_0 weights are converted to the
// gfx906 two-plane layout; everything else is copied through unchanged. Enabled
// only on gfx906 (the arch the kernels compile for); the Q8_0 path is gated by
// GGML_CUDA_REPACK_Q8_0 in ggml_cuda_repack_tensor_supported().
#include "repack.cuh"
#include "repack-common.cuh"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct ggml_backend_cuda_repack_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft) {
    bool is_repack = buft->iface.get_name == ggml_backend_cuda_repack_buffer_type_get_name;
    return is_repack;
}

// Repack one tensor into the GCN layout (per-expert stride repack_gcn_nbytes) and
// upload it; the source may be the host upload buffer or an accumulated staging copy.
static void repack_and_upload(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const uint8_t * src) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    ggml_cuda_set_device(ctx->device);

    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];

    const size_t src_stride = ggml_nbytes(tensor) / ne2;
    const size_t dst_stride = repack_gcn_nbytes(tensor->type, ne0, ne1);
    std::vector<uint8_t> repacked(dst_stride * ne2);
    for (int64_t e = 0; e < ne2; e++) {
        repack_q8_0_host((const block_q8_0 *) (src + e * src_stride),
            repacked.data() + e * dst_stride, ne0, ne1);
    }

    CUDA_CHECK(cudaMemcpyAsync(tensor->data, repacked.data(), repacked.size(),
        cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// ---- async upload path ------------------------------------------------------
// The loader's async path delivers CANONICAL bytes in file-order chunks whose
// boundaries ignore the 34-byte block structure, so chunks stage into a
// per-device scratch at their canonical offsets and the repack kernel runs on
// the SAME stream once the tensor is complete. Stream ordering makes one
// scratch per device safe: the next tensor's H2D writes are queued after this
// tensor's repack kernel has read the bytes.

// Device-side repack: one thread per destination sub-block. Reads canonical
// block_q8_0 (34 B: half d, then 32 int8) and writes the two planes. Padding
// sub-blocks (blk >= n_blocks) are zeroed, matching repack_q8_0_host().
static __global__ void repack_q8_0_kernel(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_blocks, const int64_t nsp,
        const int64_t qs_len, const int64_t src_stride, const int64_t dst_stride,
        const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t blk = i % nsp;
        const int64_t row = (i / nsp) % ne1;
        const int64_t e   = i / (nsp * ne1);

        uint8_t * d_qs = dst + e * dst_stride + (row * nsp + blk) * 32;
        uint8_t * d_d  = dst + e * dst_stride + qs_len + (row * nsp + blk) * 2;

        if (blk < n_blocks) {
            const uint8_t * sb = src + e * src_stride + (row * n_blocks + blk) * 34;
#pragma unroll
            for (int k = 0; k < 32; k++) {
                d_qs[k] = sb[2 + k];
            }
            d_d[0] = sb[0];
            d_d[1] = sb[1];
        } else {
#pragma unroll
            for (int k = 0; k < 32; k++) {
                d_qs[k] = 0;
            }
            d_d[0] = 0;
            d_d[1] = 0;
        }
    }
}

struct repack_async_state {
    uint8_t *           scratch  = nullptr;
    size_t              cap      = 0;
    const ggml_tensor * cur      = nullptr;
    size_t              received = 0;
};
static repack_async_state s_async[GGML_CUDA_MAX_DEVICES];
static std::mutex s_async_mutex;

void ggml_cuda_repack_set_tensor_async(int device, cudaStream_t stream,
        ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(s_async_mutex);
    repack_async_state & st = s_async[device];
    const size_t total = ggml_nbytes(tensor);

    ggml_cuda_set_device(device);
    if (st.cap < total) {
        // cudaFree synchronizes, but growth is monotonic across a load
        if (st.scratch != nullptr) {
            CUDA_CHECK(cudaFree(st.scratch));
        }
        CUDA_CHECK(cudaMalloc(&st.scratch, total));
        st.cap = total;
    }
    if (st.cur != tensor) {
        // the loader finishes one tensor before starting the next - a switch
        // with bytes outstanding means interleaved chunks and silent corruption
        GGML_ASSERT(st.cur == nullptr && "repack async upload: previous tensor incomplete");
        st.cur      = tensor;
        st.received = 0;
    }
    GGML_ASSERT(offset + size <= total);
    CUDA_CHECK(cudaMemcpyAsync(st.scratch + offset, data, size, cudaMemcpyHostToDevice, stream));
    st.received += size;

    if (st.received == total) {
        const int64_t ne0      = tensor->ne[0];
        const int64_t ne1      = tensor->ne[1];
        const int64_t ne2      = tensor->ne[2];
        const int64_t n_blocks = ne0 / 32;
        const int64_t nsp      = repack_nsp(ne0);
        const size_t  src_str  = total / ne2;
        const size_t  dst_str  = repack_gcn_nbytes(tensor->type, ne0, ne1);
        const int64_t n_out    = ne2 * ne1 * nsp;
        const int     block    = 256;
        const int     grid     = (int) std::min<int64_t>((n_out + block - 1) / block, 65535);
        switch (tensor->type) {
            case GGML_TYPE_Q8_0:
                repack_q8_0_kernel<<<grid, block, 0, stream>>>(
                    st.scratch, (uint8_t *) tensor->data, ne1, n_blocks, nsp,
                    ne1 * nsp * 32, (int64_t) src_str, (int64_t) dst_str, n_out);
                break;
            default:
                GGML_ABORT("unsupported repack type for async upload");
        }
        CUDA_CHECK(cudaGetLastError());
        st.cur      = nullptr;
        st.received = 0;
    }
}

void ggml_cuda_repack_async_release(int device) {
    if (s_async[device].scratch == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(s_async_mutex);
    repack_async_state & st = s_async[device];
    if (st.scratch != nullptr) {
        GGML_ASSERT(st.cur == nullptr && "repack async upload: tensor incomplete at release");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(st.scratch));
        st.scratch = nullptr;
        st.cap     = 0;
    }
}

// Set-tensor override: repack supported Q8_0 weights into the GCN layout on upload;
// everything else is copied through unchanged.
static void ggml_backend_cuda_repack_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {

    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    ggml_cuda_set_device(ctx->device);

    if (!ggml_cuda_repack_tensor_supported(tensor)) {
        CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size,
            cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return;
    }

    const size_t t_nbytes = ggml_nbytes(tensor);

    if (offset != 0 || size != t_nbytes) {
        // Partial write: accumulate into a host staging buffer and repack once complete.
        static std::mutex staging_mutex;

        struct staging_entry { std::vector<uint8_t> data; size_t filled = 0; };
        static std::map<void*, staging_entry> staging;
        void * key = tensor->data;
        staging_entry full;
        {
            std::lock_guard<std::mutex> lock(staging_mutex);
            auto & staged = staging[key];
            if (staged.data.empty()) {
                staged.data.resize(t_nbytes, 0);
            }
            GGML_ASSERT(offset + size <= t_nbytes);
            memcpy(staged.data.data() + offset, data, size);
            staged.filled += size;
            if (staged.filled < t_nbytes) {
                return;
            }
            full = std::move(staged);
            staging.erase(key);
        }
        repack_and_upload(buffer, tensor, full.data.data());
        return;
    }

    repack_and_upload(buffer, tensor, (const uint8_t *) data);
}

static ggml_backend_buffer_t ggml_backend_cuda_repack_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;

    ggml_backend_buffer_t buffer =
        ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(ctx->device), size);
    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.set_tensor  = ggml_backend_cuda_repack_buffer_set_tensor;
    // Weights are write-once; the inherited CUDA get_tensor would misread the
    // two-plane layout as canonical, so it must be explicitly nulled
    // (ggml_backend_tensor_get has no null guard).
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    buffer->iface.set_tensor_2d = nullptr;
    buffer->iface.get_tensor_2d = nullptr;
    return buffer;
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    if (ggml_cuda_repack_tensor_supported(tensor)) {
        return repack_gcn_nbytes(tensor->type, tensor->ne[0], tensor->ne[1]) * tensor->ne[2];
    }
    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_repack_buffer_type_interface = {
    ggml_backend_cuda_repack_buffer_type_get_name,
    ggml_backend_cuda_repack_buffer_type_alloc_buffer,
    ggml_backend_cuda_repack_buffer_type_get_alignment,
    nullptr,
    ggml_backend_cuda_repack_buffer_type_get_alloc_size,
    nullptr,
};

// Repacked buffer type: only enabled on gfx906 with GGML_CUDA_REPACK_Q8_0, else
// returns nullptr so the caller falls back to the normal CUDA buffer type.
// Gating discovery here makes "repack off" fully native: no repack buft is
// offered to the loader, so no structural repack code (meta supports_op,
// set_tensor bypass, fusion suppression) is ever active.
ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }
    // Kernels only compile for gfx906 (__gfx906__ guard); a wider GCN gate would
    // hand two-plane weights to archs whose repack kernels are NO_DEVICE_CODE.
    if (ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_VEGA20) {
        return nullptr;
    }
    if (!ggml_cuda_repack_enabled()) {
        return nullptr;
    }

    static ggml_backend_buffer_type buft_storage[GGML_CUDA_MAX_DEVICES];
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {};

    if (!initialized[device]) {
        buft_storage[device] = {
            ggml_backend_cuda_repack_buffer_type_interface,
            ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
            new ggml_backend_cuda_repack_buffer_type_context{
                                 device, GGML_CUDA_NAME + std::to_string(device) + "_Repacked"},
        };
        initialized[device] = true;
    }
    return &buft_storage[device];
}
