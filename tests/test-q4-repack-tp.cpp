#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

struct context_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};
struct buffer_deleter {
    void operator()(ggml_backend_buffer * buf) const { ggml_backend_buffer_free(buf); }
};
struct allocator_deleter {
    void operator()(ggml_gallocr_t alloc) const { ggml_gallocr_free(alloc); }
};
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

struct split_config {
    ggml_backend_meta_split_axis axis;
    bool uneven;
    bool segmented;
};

// Follow llama-model's static-tensor callback convention. Column parallelism
// mirrors activations; reduction parallelism splits both operands on dim 0.
static ggml_backend_meta_split_state split_state(const ggml_tensor * tensor, void * userdata) {
    const auto & config = *static_cast<const split_config *>(userdata);
    ggml_backend_meta_split_state state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    const bool weight = std::strcmp(tensor->name, "output.weight") == 0;
    if (!weight && config.axis != GGML_BACKEND_SPLIT_AXIS_0) {
        return state;
    }
    state.axis = config.axis;
    state.n_segments = config.segmented ? 2 : 1;
    const int64_t segment_width = tensor->ne[state.axis] / state.n_segments;
    for (uint32_t s = 0; s < state.n_segments; ++s) {
        state.nr[s] = 1;
        state.ne[2*s] = segment_width / (config.uneven ? 4 : 2);
        state.ne[2*s + 1] = segment_width - state.ne[2*s];
    }
    return state;
}

static std::vector<uint8_t> make_weights(int64_t width, int64_t rows) {
    std::vector<uint8_t> result(ggml_row_size(GGML_TYPE_Q4_0, width) * rows);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t block = 0; block < width / 32; ++block) {
            uint8_t * dst = result.data() + (row * (width / 32) + block) * 18;
            const uint16_t scale = uint16_t((0x2800 + ((row * 19 + block * 7) % 2048)) |
                                          ((row + block) % 3 == 0 ? 0x8000 : 0));
            std::memcpy(dst, &scale, sizeof(scale));
            for (int i = 0; i < 16; ++i) {
                dst[2+i] = uint8_t(((i + block + row * 3) & 15) |
                                  (((15 - i + block * 3 + row) & 15) << 4));
            }
        }
    }
    return result;
}

static bool run_case(ggml_backend_t backend, ggml_backend_buffer_type_t weight_buft,
                     ggml_backend_buffer_type_t compute_buft,
                     const std::vector<uint8_t> & weights, const std::vector<float> & input,
                     int64_t width, int64_t rows, int64_t tokens, std::vector<float> & output) {
    const ggml_init_params params = {1024 * 1024, nullptr, true};
    context_ptr weight_ctx(ggml_init(params));
    context_ptr input_ctx(ggml_init(params));
    context_ptr compute_ctx(ggml_init(params));
    if (!weight_ctx || !input_ctx || !compute_ctx) return false;
    ggml_tensor * w = ggml_new_tensor_2d(weight_ctx.get(), GGML_TYPE_Q4_0, width, rows);
    ggml_set_name(w, "output.weight");
    buffer_ptr weight_buf(ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx.get(), weight_buft));
    if (!weight_buf) return false;
    ggml_backend_buffer_set_usage(weight_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());

    // Separate static input buffer allows the callback to assign its lane split.
    // Never put activations or graph outputs in the planar weight buffer.
    ggml_tensor * x = ggml_new_tensor_2d(input_ctx.get(), GGML_TYPE_F32, width, tokens);
    ggml_set_name(x, "input");
    buffer_ptr input_buf(ggml_backend_alloc_ctx_tensors_from_buft(input_ctx.get(), compute_buft));
    if (!input_buf) return false;
    ggml_backend_buffer_set_usage(input_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_tensor * y = ggml_mul_mat(compute_ctx.get(), w, x);
    // Identity consumer closes a reduction-parallel PARTIAL boundary through
    // meta's normal AllReduce; direct PARTIAL readback is not supported.
    y = ggml_scale(compute_ctx.get(), y, 1.0f);
    ggml_set_name(y, "tp2.output");
    ggml_cgraph * graph = ggml_new_graph(compute_ctx.get());
    ggml_build_forward_expand(graph, y);
    // The graph allocator marks buffers COMPUTE before deriving simple tensors;
    // static ctx allocation would apply the weight callback to graph outputs.
    std::unique_ptr<ggml_gallocr, allocator_deleter> alloc(ggml_gallocr_new(compute_buft));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc.get(), graph)) return false;
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_synchronize(backend);
    output.resize(rows * tokens);
    // Only F32 outputs are read back; planar Q4 bytes never reach a canonical consumer.
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
    return true;
}

int main() {
    const char * env = std::getenv("GGML_CUDA_REPACK_Q4_0");
    if (!env || std::strcmp(env, "1") != 0) {
        std::fprintf(stderr, "Set GGML_CUDA_REPACK_Q4_0=1\n");
        return 1;
    }
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) devices.push_back(dev);
    }
    if (devices.size() < 2) {
        std::fprintf(stderr, "TP2 requires two GPU devices; skipping\n");
        return 77;
    }
    std::fprintf(stderr, "TP2 lanes: %s, %s\n", ggml_backend_dev_name(devices[0]), ggml_backend_dev_name(devices[1]));
    split_config config = {GGML_BACKEND_SPLIT_AXIS_1, false, false};
    auto meta = ggml_backend_meta_device(devices.data(), 2, 2, split_state, &config);
    auto backend = ggml_backend_dev_init(meta, nullptr);
    if (!backend) return 1;
    auto compute_buft = ggml_backend_dev_buffer_type(meta);
    auto reg = ggml_backend_dev_backend_reg(meta);
    auto extra_fn = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
    auto extras = extra_fn ? extra_fn(meta) : nullptr;
    if (!extras || !extras[0]) {
        std::fprintf(stderr, "Meta TP2 does not expose all-lane repack buffer type\n");
        ggml_backend_free(backend);
        return 1;
    }
    std::fprintf(stderr, "TP2 canonical=%s planar=%s\n",
                 ggml_backend_buft_name(compute_buft), ggml_backend_buft_name(extras[0]));
    constexpr int64_t width = 512, rows = 128;
    const auto weights = make_weights(width, rows);
    std::mt19937 rng(0x9064);
    std::uniform_real_distribution<float> distribution(-0.5f, 1.4f);
    int cases = 0;
    for (auto axis : {GGML_BACKEND_SPLIT_AXIS_1, GGML_BACKEND_SPLIT_AXIS_0}) {
        for (bool uneven : {false, true}) {
            for (bool segmented : {false, true}) {
                config = {axis, uneven, segmented};
                for (int64_t tokens : {1, 2, 3, 4, 8, 32, 128}) {
                    std::vector<float> input(width * tokens);
                    for (float & value : input) value = distribution(rng);
                    std::vector<float> canonical, planar;
                    if (!run_case(backend, compute_buft, compute_buft, weights, input, width, rows, tokens, canonical) ||
                        !run_case(backend, extras[0], compute_buft, weights, input, width, rows, tokens, planar)) {
                        std::fprintf(stderr, "TP2 compute failed: axis=%d uneven=%d segmented=%d tokens=%lld\n",
                                     int(axis), uneven, segmented, (long long) tokens);
                        ggml_backend_free(backend);
                        return 1;
                    }
                    double err2 = 0.0, ref2 = 0.0;
                    float max_abs = 0.0f;
                    for (size_t i = 0; i < canonical.size(); ++i) {
                        if (!std::isfinite(canonical[i]) || !std::isfinite(planar[i])) {
                            std::fprintf(stderr, "Nonfinite TP2 output\n");
                            ggml_backend_free(backend);
                            return 1;
                        }
                        const float diff = canonical[i] - planar[i];
                        err2 += double(diff) * diff;
                        ref2 += double(canonical[i]) * canonical[i];
                        max_abs = std::max(max_abs, std::abs(diff));
                    }
                    const double nmse = ref2 > 0 ? err2 / ref2 : err2;
                    std::printf("TP2 Q4_0 MUL_MAT axis=%d uneven=%d segments=%d tokens=%lld NMSE=%.8g max_abs=%.8g\n",
                                int(axis), uneven, segmented ? 2 : 1, (long long) tokens, nmse, max_abs);
                    std::fflush(stdout);
                    if (!std::isfinite(nmse) || nmse > 5e-4) {
                        std::fprintf(stderr, "TP2 canonical/planar parity failed\n");
                        ggml_backend_free(backend);
                        return 1;
                    }
                    ++cases;
                }
            }
        }
    }
    std::printf("PASS: %d TP2 canonical/planar cases\n", cases);
    ggml_backend_free(backend);
    return 0;
}
