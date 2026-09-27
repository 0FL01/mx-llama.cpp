#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

struct context_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer * buffer) const { ggml_backend_buffer_free(buffer); }
};

using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr  = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

static ggml_type weight_type = GGML_TYPE_Q4_0;

static size_t planar_size(int64_t ne0, int64_t rows) {
    const size_t payload = weight_type == GGML_TYPE_Q8_0 ? ne0 : ne0 / 2;
    const size_t stride = payload + (payload % 128 == 0 ? 16 : 0);
    const size_t scale_bytes = weight_type == GGML_TYPE_MXFP4 ? 1 : 2;
    return (stride + (size_t) ne0 / 32 * scale_bytes) * rows;
}

static void upload_weights(ggml_backend_t backend, ggml_tensor * w, const std::vector<uint8_t> & weights) {
    if (std::getenv("Q4_REPACK_TEST_ASYNC") != nullptr) {
        ggml_backend_tensor_set_async(backend, w, weights.data(), 0, weights.size());
        ggml_backend_synchronize(backend);
    } else {
        ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    }
}

static bool run_dense_case(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        ggml_backend_buffer_type_t compute_buft,
        const std::vector<uint8_t> & weights,
        const std::vector<float> & input,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t n_tokens,
        std::vector<float> & output) {
    const ggml_init_params params = {
        /* .mem_size = */ 1024 * 1024,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    context_ptr ctx_weights(ggml_init(params));
    context_ptr ctx_compute(ggml_init(params));
    if (!ctx_weights || !ctx_compute) {
        return false;
    }

    const int64_t base_rows = weights.size() / ggml_row_size(weight_type, ne0);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_weights.get(), weight_type, ne0, base_rows);
    buffer_ptr wbuf(ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights.get(), weight_buft));
    if (!wbuf) {
        return false;
    }
    ggml_backend_buffer_set_usage(wbuf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t planar_bytes = planar_size(ne0, base_rows);
    if (weight_buft != compute_buft && ggml_backend_buft_get_alloc_size(weight_buft, w) != planar_bytes) {
        std::fprintf(stderr, "Q4_0 repack buffer reports incorrect allocation size\n");
        return false;
    }
    upload_weights(backend, w, weights);

    if (base_rows != ne1) {
        w = ggml_view_2d(ctx_compute.get(), w, ne0, ne1, w->nb[1], w->nb[1]);
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx_compute.get(), GGML_TYPE_F32, ne0, n_tokens);
    ggml_tensor * y = ggml_mul_mat(ctx_compute.get(), w, x);
    buffer_ptr compute_buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx_compute.get(), compute_buft));
    if (!compute_buf) {
        return false;
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx_compute.get());
    ggml_build_forward_expand(graph, y);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "Q4_0 MUL_MAT graph compute failed (tokens=%lld)\n", (long long) n_tokens);
        return false;
    }
    ggml_backend_synchronize(backend);

    output.resize((size_t) ne1 * (size_t) n_tokens);
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
    return true;
}

static bool run_moe_case(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        ggml_backend_buffer_type_t compute_buft,
        const std::vector<uint8_t> & weights,
        const std::vector<float> & input,
        const std::vector<int32_t> & ids,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t n_experts,
        const int64_t n_tokens,
        std::vector<float> & output) {
    const ggml_init_params params = {
        /* .mem_size = */ 1024 * 1024,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    context_ptr ctx_weights(ggml_init(params));
    context_ptr ctx_compute(ggml_init(params));
    if (!ctx_weights || !ctx_compute) {
        return false;
    }

    const int64_t base_experts = weights.size() / (ggml_row_size(weight_type, ne0) * ne1);
    ggml_tensor * w = ggml_new_tensor_3d(ctx_weights.get(), weight_type, ne0, ne1, base_experts);
    buffer_ptr wbuf(ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights.get(), weight_buft));
    if (!wbuf) {
        return false;
    }
    ggml_backend_buffer_set_usage(wbuf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t planar_bytes = planar_size(ne0, ne1 * base_experts);
    if (weight_buft != compute_buft && ggml_backend_buft_get_alloc_size(weight_buft, w) != planar_bytes) {
        std::fprintf(stderr, "Q4_0 MoE repack buffer reports incorrect allocation size\n");
        return false;
    }
    upload_weights(backend, w, weights);

    if (base_experts != n_experts) {
        w = ggml_view_3d(ctx_compute.get(), w, ne0, ne1, n_experts, w->nb[1], w->nb[2], w->nb[2]);
    }
    const int64_t n_used = ids.size() / n_tokens;
    const int64_t n_channels = input.size() / (ne0 * n_tokens);
    ggml_tensor * x = ggml_new_tensor_3d(ctx_compute.get(), GGML_TYPE_F32, ne0, n_channels, n_tokens);
    ggml_tensor * expert_ids = ggml_new_tensor_2d(ctx_compute.get(), GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * y = ggml_mul_mat_id(ctx_compute.get(), w, x, expert_ids);
    buffer_ptr compute_buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx_compute.get(), compute_buft));
    if (!compute_buf) {
        return false;
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(expert_ids, ids.data(), 0, ids.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(ctx_compute.get());
    ggml_build_forward_expand(graph, y);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "Q4_0 MUL_MAT_ID graph compute failed (tokens=%lld)\n", (long long) n_tokens);
        return false;
    }
    ggml_backend_synchronize(backend);

    output.resize((size_t) ne1 * (size_t) n_tokens * (size_t) n_used);
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
    return true;
}

static ggml_backend_buffer_type_t find_repack_buft(ggml_backend_t & backend) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto get_extra_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
        if (get_extra_bufts == nullptr) {
            continue;
        }
        ggml_backend_buffer_type_t * bufts = get_extra_bufts(dev);
        if (bufts == nullptr || bufts[0] == nullptr) {
            continue;
        }

        backend = ggml_backend_dev_init(dev, nullptr);
        if (backend != nullptr) {
            std::fprintf(stderr, "Testing Q4_0 repack on %s (%s)\n",
                         ggml_backend_dev_name(dev), ggml_backend_reg_name(reg));
            return bufts[0];
        }
    }
    return nullptr;
}

static std::vector<uint8_t> make_q4_0_weights(const int64_t ne0, const int64_t ne1) {
    if (weight_type != GGML_TYPE_Q4_0) {
        std::vector<float> values((size_t) ne0 * ne1);
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin((float) i * 0.13f);
        std::vector<uint8_t> weights(ggml_row_size(weight_type, ne0) * ne1);
        ggml_quantize_chunk(weight_type, values.data(), weights.data(), 0, ne1, ne0, nullptr);
        return weights;
    }
    constexpr size_t block_bytes = 18;
    const int64_t n_blocks = ne0 / 32;
    std::vector<uint8_t> weights((size_t) ne1 * (size_t) n_blocks * block_bytes);
    std::mt19937 codes(0x4a906);

    for (int64_t row = 0; row < ne1; ++row) {
        for (int64_t block = 0; block < n_blocks; ++block) {
            uint8_t * dst = weights.data() + ((size_t) row * (size_t) n_blocks + (size_t) block) * block_bytes;
            // Varied positive/negative FP16 scales and nonzero activation mean
            // expose missing Q4_0 zero-point correction and scale addressing.
            const uint16_t scale = (uint16_t) ((0x2800 + ((row * 19 + block * 7) % 2048)) |
                                               ((row + block) % 3 == 0 ? 0x8000 : 0));
            std::memcpy(dst, &scale, sizeof(scale));
            for (int i = 0; i < 16; ++i) {
                const uint8_t qlo = (uint8_t) (codes() & 0x0f);
                const uint8_t qhi = (uint8_t) (codes() & 0x0f);
                dst[2 + i] = (uint8_t) (qlo | (qhi << 4));
            }
        }
    }
    return weights;
}

int main(int argc, char ** argv) {
    const char * repack_env = std::getenv("GGML_CUDA_REPACK_Q4_0");
    if (repack_env == nullptr || std::strcmp(repack_env, "0") == 0) {
        std::fprintf(stderr, "Set GGML_CUDA_REPACK_Q4_0=1 to run this test\n");
        return 1;
    }

    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t repack_buft = find_repack_buft(backend);
    if (backend == nullptr || repack_buft == nullptr) {
        std::fprintf(stderr, "No gfx906 repack backend found; skipping\n");
        return 77;
    }

    ggml_backend_buffer_type_t compute_buft = ggml_backend_get_default_buffer_type(backend);
    const int64_t ne0 = argc > 1 ? std::strtoll(argv[1], nullptr, 10) : 512;
    const int64_t ne1 = argc > 2 ? std::strtoll(argv[2], nullptr, 10) : 128;
    const bool views = argc > 3 && std::strcmp(argv[3], "views") == 0;
    if (argc > 4) {
        if (std::strcmp(argv[4], "q4_0") == 0) weight_type = GGML_TYPE_Q4_0;
        else if (std::strcmp(argv[4], "q8_0") == 0) weight_type = GGML_TYPE_Q8_0;
        else if (std::strcmp(argv[4], "mxfp4") == 0) weight_type = GGML_TYPE_MXFP4;
        else return 1;
    }
    std::fprintf(stderr, "Parity weight type: %s\n", ggml_type_name(weight_type));
    if (ne0 <= 0 || ne0 % 32 != 0 || ne1 <= 0) return 1;
    // Optional canonical weight/F32 activation snapshots isolate a real-model
    // projection on exactly the same input, without autoregressive divergence.
    const char * weight_file = std::getenv("Q4_REPACK_TEST_WEIGHT_FILE");
    const char * input_file = std::getenv("Q4_REPACK_TEST_INPUT_FILE");
    if ((weight_file == nullptr) != (input_file == nullptr)) return 1;
    std::vector<uint8_t> fixture_weights;
    std::vector<float> fixture_input;
    std::vector<int64_t> widths = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 32, 64, 128, 512};
    if (weight_file) {
        if (views) return 1;
        auto read_bytes = [](const char * path) {
            std::ifstream file(path, std::ios::binary);
            if (!file) return std::vector<uint8_t>();
            return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), {});
        };
        fixture_weights = read_bytes(weight_file);
        const auto bytes = read_bytes(input_file);
        if (fixture_weights.size() != ggml_row_size(weight_type, ne0) * ne1 ||
            bytes.empty() || bytes.size() % (ne0 * sizeof(float)) != 0) return 1;
        fixture_input.resize(bytes.size() / sizeof(float));
        std::memcpy(fixture_input.data(), bytes.data(), bytes.size());
        widths = {(int64_t) fixture_input.size() / ne0};
    }
    std::mt19937 rng(0x9064);
    std::uniform_real_distribution<float> distribution(-0.7f, 1.2f);

    for (const int64_t n_tokens : widths) {
        const std::vector<uint8_t> weights = weight_file ? fixture_weights : make_q4_0_weights(ne0, ne1 + (views ? 2 : 0));
        std::vector<float> input = weight_file ? fixture_input : std::vector<float>((size_t) ne0 * (size_t) n_tokens);
        if (!weight_file) for (float & value : input) {
            value = distribution(rng) + 0.2f;
        }

        std::vector<float> canonical;
        std::vector<float> repacked;
        if (!run_dense_case(backend, compute_buft, compute_buft, weights, input,
                            ne0, ne1, n_tokens, canonical) ||
            !run_dense_case(backend, repack_buft, compute_buft, weights, input,
                            ne0, ne1, n_tokens, repacked)) {
            ggml_backend_free(backend);
            return 1;
        }

        double err2 = 0.0;
        double ref2 = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < canonical.size(); ++i) {
            const float diff = canonical[i] - repacked[i];
            err2 += (double) diff * diff;
            ref2 += (double) canonical[i] * canonical[i];
            max_abs = std::max(max_abs, std::abs(diff));
        }
        const double nmse = ref2 > 0.0 ? err2 / ref2 : err2;
        std::printf("%s dense tokens=%lld NMSE=%.8g max_abs=%.8g\n",
                    ggml_type_name(weight_type), (long long) n_tokens, nmse, max_abs);
        if (!std::isfinite(nmse) || nmse > 5e-4) {
            std::fprintf(stderr, "Q4_0 dense parity failed at tokens=%lld\n", (long long) n_tokens);
            ggml_backend_free(backend);
            return 1;
        }
    }

    if (weight_file) {
        ggml_backend_free(backend);
        return 0;
    }
    constexpr int64_t n_experts = 16;
    const std::vector<uint8_t> moe_weights = make_q4_0_weights(ne0, ne1 * (n_experts + (views ? 1 : 0)));
    for (const int64_t n_used : {1, 10}) {
    for (const int64_t n_channels : (n_used == 1 ? std::vector<int64_t>{1} : std::vector<int64_t>{1, n_used})) {
    for (const int64_t n_tokens : {1, 2, 3, 4, 5, 6, 7, 8, 12, 32, 64, 1024}) {
        std::vector<float> input((size_t) ne0 * (size_t) n_tokens * (size_t) n_channels);
        std::vector<int32_t> ids((size_t) n_tokens * (size_t) n_used);
        for (float & value : input) {
            value = distribution(rng) + 0.2f;
        }
        for (int64_t i = 0; i < n_tokens; ++i) {
            for (int64_t j = 0; j < n_used; ++j) {
                ids[(size_t) i * n_used + j] = (int32_t) ((i * 3 + i / 4 + j) % n_experts);
            }
        }

        std::vector<float> canonical;
        std::vector<float> repacked;
        if (!run_moe_case(backend, compute_buft, compute_buft, moe_weights, input, ids,
                          ne0, ne1, n_experts, n_tokens, canonical) ||
            !run_moe_case(backend, repack_buft, compute_buft, moe_weights, input, ids,
                          ne0, ne1, n_experts, n_tokens, repacked)) {
            ggml_backend_free(backend);
            return 1;
        }

        double err2 = 0.0;
        double ref2 = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < canonical.size(); ++i) {
            const float diff = canonical[i] - repacked[i];
            err2 += (double) diff * diff;
            ref2 += (double) canonical[i] * canonical[i];
            max_abs = std::max(max_abs, std::abs(diff));
        }
        const double nmse = ref2 > 0.0 ? err2 / ref2 : err2;
        std::printf("%s MoE tokens=%lld used=%lld channels=%lld NMSE=%.8g max_abs=%.8g\n",
                    ggml_type_name(weight_type), (long long) n_tokens, (long long) n_used,
                    (long long) n_channels, nmse, max_abs);
        if (!std::isfinite(nmse) || nmse > 5e-4) {
            std::fprintf(stderr, "Q4_0 MoE parity failed at tokens=%lld\n", (long long) n_tokens);
            ggml_backend_free(backend);
            return 1;
        }
    }

    }
    }
    ggml_backend_free(backend);
    return 0;
}
