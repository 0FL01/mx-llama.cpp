// Repack buffer type: on upload, supported Q8_0 weights are converted to the
// gfx906 two-plane layout; everything else is copied through unchanged. Enabled
// only on gfx906 (the arch the kernels compile for); the Q8_0 path is gated by
// GGML_CUDA_REPACK_Q8_0 in ggml_cuda_repack_tensor_supported().
#include "repack.cuh"
#include "repack-common.cuh"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"

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
    bool is_repack = buft->iface.get_name == ggml_backend_cuda_repack_buffer_type_get_name ||
           ggml_backend_meta_buft_is_repack(buft);
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
