#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static void test_dummy(ggml_backend_t backend, ggml_type type, int nt, int channels, bool strided,
                       int dummy, int fusion) {
    constexpr int width = 256;
    constexpr int rows = 17; // Tail in the two-row multi-token kernel.
    constexpr int experts = 4;
    constexpr int used = 4;
    const int id_stride = strided ? used + 3 : used;
    ggml_init_params ip = {64*ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(ip));
    GGML_ASSERT(ctx);
    auto * c = ctx.get();
    auto * weights = ggml_new_tensor_3d(c, type, width, rows, experts);
    auto * gate_weights = fusion == 2 ? ggml_dup_tensor(c, weights) : nullptr;
    auto * bias = fusion ? ggml_new_tensor_2d(c, GGML_TYPE_F32, rows, experts) : nullptr;
    auto * input = ggml_new_tensor_3d(c, GGML_TYPE_F32, width, channels, nt);
    auto * ranking = ggml_new_tensor_2d(c, GGML_TYPE_I32, id_stride, nt);
    auto * ids = strided ? ggml_view_2d(c, ranking, used, nt, ranking->nb[1], sizeof(int32_t)) : ranking;
    ggml_set_input(input);
    ggml_set_input(ranking);

    const auto chain = [&](bool marked) {
        auto * up = ggml_mul_mat_id(c, weights, input, ids);
        GGML_ASSERT(ggml_mul_mat_id_get_cache_dummy(up) == -1);
        if (marked) {
            ggml_mul_mat_id_set_cache_dummy(up, dummy);
            GGML_ASSERT(ggml_mul_mat_id_get_cache_dummy(up) == dummy && up->op_params[0] == 0);
            ggml_mul_mat_id_set_cache_dummy(up, -1);
            GGML_ASSERT(ggml_mul_mat_id_get_cache_dummy(up) == -1);
            ggml_mul_mat_id_set_cache_dummy(up, dummy);
        }
        if (bias) { up = ggml_add_id(c, up, bias, ids); }
        if (gate_weights) {
            // This gate has no zero expert guarantee, even at the marked up expert.
            auto * gate = ggml_mul_mat_id(c, gate_weights, input, ids);
            gate = ggml_add_id(c, gate, bias, ids);
            up = ggml_swiglu_split(c, gate, up);
        }
        ggml_set_output(up);
        return up;
    };
    auto * reference = chain(false);
    auto * marked = chain(true);
    auto * graph = ggml_new_graph_custom(c, 64, false);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, marked);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        GGML_ASSERT(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)));
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(c, backend));
    GGML_ASSERT(buffer);
    ggml_backend_buffer_clear(buffer.get(), 0);

    const auto fill_weights = [&](ggml_tensor * w, int zero_expert) {
        std::vector<float> values(ggml_nelements(w));
        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.01f*(1 + i % 13); }
        if (zero_expert >= 0) {
            std::fill_n(values.begin() + zero_expert*width*rows, width*rows, 0.0f);
        }
        std::vector<char> packed(ggml_nbytes(w));
        std::vector<float> imatrix(width, 1.0f);
        GGML_ASSERT(ggml_quantize_chunk(type, values.data(), packed.data(), 0, rows*experts, width,
                                       imatrix.data()) == packed.size());
        ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    };
    // Establish the zero expert before execution; never infer it from an ID position or device readback.
    fill_weights(weights, dummy);
    if (gate_weights) { fill_weights(gate_weights, -1); }
    std::vector<float> biases(rows*experts);
    for (size_t i = 0; i < biases.size(); ++i) { biases[i] = 0.25f + 0.001f*i; }
    if (bias) { ggml_backend_tensor_set(bias, biases.data(), 0, ggml_nbytes(bias)); }
    std::vector<float> x(ggml_nelements(input));
    for (size_t i = 0; i < x.size(); ++i) { x[i] = 0.02f*(1 + i % 17); }
    ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
    std::vector<int32_t> route(id_stride*nt, 0);
    std::vector<float> a(ggml_nelements(reference)), b(a.size());
    const std::vector<float> sentinel(a.size(), std::numeric_limits<float>::quiet_NaN());

    // Reuse the graph across all-hit, all-miss, mixed and all-hit routes. IDs include duplicates.
    for (int round = 0; round < 4; ++round) {
        for (int t = 0; t < nt; ++t) {
            for (int e = 0; e < used; ++e) {
                const int hit = dummy == experts - 1 ? 2 : experts - 1;
                const int miss = dummy >= 0 ? dummy : experts - 1;
                const int pos = (e + t) % used;
                route[t*id_stride + (strided ? 1 : 0) + e] =
                    round == 1 ? miss : round == 2 ? (pos < 2 ? miss : hit) : (pos % 2 ? hit : 0);
            }
        }
        ggml_backend_tensor_set(ranking, route.data(), 0, ggml_nbytes(ranking));
        ggml_backend_tensor_set(reference, sentinel.data(), 0, ggml_nbytes(reference));
        ggml_backend_tensor_set(marked, sentinel.data(), 0, ggml_nbytes(marked));
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(reference, a.data(), 0, ggml_nbytes(reference));
        ggml_backend_tensor_get(marked, b.data(), 0, ggml_nbytes(marked));
        for (int t = 0; t < nt; ++t) {
            for (int e = 0; e < used; ++e) {
                const int id = route[t*id_stride + (strided ? 1 : 0) + e];
                for (int r = 0; r < rows; ++r) {
                    const size_t i = (t*used + e)*rows + r;
                    const bool close = std::isfinite(a[i]) && std::isfinite(b[i]) &&
                                       std::fabs(a[i] - b[i]) <= 1e-6f + 1e-5f*std::fabs(a[i]);
                    if (!close) {
                        std::fprintf(stderr, "%s nt=%d channels=%d strided=%d dummy=%d fusion=%d round=%d t=%d e=%d r=%d: %g != %g\n",
                                     ggml_type_name(type), nt, channels, strided, dummy, fusion, round, t, e, r, a[i], b[i]);
                        GGML_ABORT("marked/unmarked mismatch");
                    }
                    if (id == dummy && fusion == 0) {
                        GGML_ASSERT(a[i] == 0.0f && b[i] == 0.0f);
                    } else if (id == dummy && fusion == 1) {
                        GGML_ASSERT(std::fabs(b[i] - biases[id*rows + r]) < 1e-6f);
                    } else {
                        // Includes ordinary nonzero last experts and biased dummy + nonzero gate fallback.
                        if (!(a[i] > 0.0f && b[i] > 0.0f)) {
                            std::vector<char> readback(ggml_nbytes(weights));
                            ggml_backend_tensor_get(weights, readback.data(), 0, readback.size());
                            float row[width];
                            ggml_get_type_traits(type)->to_float(readback.data(), row, width);
                            float sum = 0; for (int j = 0; j < width; ++j) { sum += row[j]*x[j]; }
                            std::fprintf(stderr, "weight[0]=%g weight[1]=%g cpu-dot=%g input[0]=%g\n", row[0], row[1], sum, x[0]);
                            std::fprintf(stderr, "positivity: type=%s nt=%d channels=%d strided=%d dummy=%d fusion=%d round=%d id=%d row=%d a=%g b=%g\n",
                                ggml_type_name(type), nt, channels, strided, dummy, fusion, round, id, r, a[i], b[i]);
                        }
                        GGML_ASSERT(a[i] > 0.0f && b[i] > 0.0f);
                    }
                }
            }
        }
    }
}

int main(int argc, char ** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [backend-name]\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    ggml_backend_ptr backend(argc == 2 ? ggml_backend_init_by_name(argv[1], nullptr) :
                                        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    GGML_ASSERT(backend);
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) { set_threads(backend.get(), 2); }
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
                     GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS}) {
        for (int nt = 1; nt <= 3; ++nt) {
            for (int channels : {1, 4}) {
                for (bool strided : {false, true}) {
                    for (int dummy : {-1, 1, 3}) {
                        test_dummy(backend.get(), type, nt, channels, strided, dummy, 0);
                    }
                }
            }
        }
        // Single-token fusion candidates: nonzero bias, then nonzero bias and unmarked gate weights.
        for (int fusion : {1, 2}) { test_dummy(backend.get(), type, 1, 1, true, 1, fusion); }
    }
    std::printf("dummy MMVQ (%s): marked/unmarked, batches 1..3, route reuse, strides, tails and fusion fallback passed\n",
                ggml_backend_name(backend.get()));
}
