// Host-side support for the Q8_0 repacked-weight path: tensor support check, host
// repack routine, and the persistent per-view cache shared by dense and MoE paths.
#include "repack.cuh"
#include "repack-common.cuh"

#include <cstring>
#include <map>
#include <mutex>

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t) {
    // Views are never in repacked layout (the scale-plane offset needs the FULL ne1).
    if (t->view_src != nullptr) return false;
    if ((ggml_n_dims(t) != 2 && ggml_n_dims(t) != 3) || !ggml_is_contiguous(t)) {
        return false;
    }
    switch (t->type) {
        case GGML_TYPE_Q8_0: {
            return t->ne[0] % 32 == 0;
        }
        default:             return false;
    }
}

bool ggml_cuda_repack_mul_mat_should_fire(const ggml_tensor * src0) {
    if (src0->buffer == nullptr || !ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_get_type(src0->buffer))) {
        return false;
    }
    if (ggml_cuda_repack_tensor_supported(src0)) {
        return true;              // full tensor in repacked layout
    }
    return src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src);
}

// Host repack of one Q8_0 matrix into the row-interleaved layout: each row is
// [qs 32*n_blocks][f16 scales 2*n_blocks] at repack_row_stride() bytes.
void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    // a non-multiple ne0 would silently drop the row tail below, so fail loudly instead
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t rs       = repack_row_stride(ne0);
    const int64_t qoff     = n_blocks * 32;

    memset(dst, 0, (size_t) ne1 * rs);

    for (int64_t row = 0; row < ne1; row++) {
        uint8_t * r = dst + row * rs;
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(r + blk * 32, b->qs, 32);
            memcpy(r + qoff + blk * 2, &b->d, 2);
        }
    }
}

// Persistent cache for re-packed view buffers. Pool allocs are invalid inside CUDA
// graph capture, so buffers are cudaMalloc'd once per unique view and never freed.
struct RepackViewCacheKey {
    const void * view_data;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    bool operator<(const RepackViewCacheKey & o) const {
        if (view_data != o.view_data) return view_data < o.view_data;
        if (ne0 != o.ne0) return ne0 < o.ne0;
        if (ne1 != o.ne1) return ne1 < o.ne1;
        return ne2 < o.ne2;
    }
};

struct RepackViewCacheEntry {
    uint8_t * d_ptr;
    size_t    size;
};

static std::map<RepackViewCacheKey, RepackViewCacheEntry> s_view_cache;
static std::mutex s_view_cache_mutex;

// Get or create a persistent device buffer for a re-packed view (source may be a
// graph-captured stream; src/dst addresses are stable because the entry persists).
const uint8_t * repack_q8_0_view_get_cached(
        const ggml_tensor * view, const ggml_tensor * base,
        cudaStream_t stream) {
    const int64_t ne0_v = view->ne[0];
    const int64_t ne1_v = view->ne[1];
    const int64_t ne2_v = view->ne[2];
    const int64_t ne0_b = base->ne[0];
    const int64_t ne1_b = base->ne[1];
    const int64_t ne2_b = base->ne[2];
    GGML_ASSERT(ne0_v == ne0_b);
    GGML_ASSERT(base->view_src == nullptr);

    const int64_t rs = repack_row_stride(ne0_v);
    const uint8_t * base_ptr = (const uint8_t *) base->data;
    const uint8_t * view_ptr = (const uint8_t *) view->data;

    GGML_ASSERT((view_ptr - base_ptr) % rs == 0);

    // Rows are self-contained in the row-interleaved layout, so the whole view
    // (ne1_v x ne2_v rows) is ONE contiguous range.
    const size_t total = (size_t) ne1_v * ne2_v * rs;

    RepackViewCacheKey key{ view->data, ne0_v, ne1_v, ne2_v };

    std::lock_guard<std::mutex> lock(s_view_cache_mutex);
    auto it = s_view_cache.find(key);
    if (it != s_view_cache.end()) {
        return it->second.d_ptr;
    }

    uint8_t * d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, total));

    CUDA_CHECK(cudaMemcpyAsync(d_ptr, view_ptr, total,
        cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    s_view_cache[key] = { d_ptr, total };
    return d_ptr;
}
