#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../src/llama-mtp-vocab.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static void test_ranges() {
    require(llama_mtp_vocab_ranges("", 512).empty(), "empty ranges must preserve baseline");
    require(llama_mtp_vocab_ranges("0:128,480:512", 512) ==
            std::vector<llama_mtp_vocab_range>{{0, 128}, {480, 512}}, "valid range parse failed");
    for (const char * value : {"0", ":128", "0:", "-1:128", "0:0", "0:513", "0:128,",
                              "480:512,0:128", "0:256,128:512", "0:128x", "0:128:129", "0:999999999999999999999"}) {
        bool rejected = false;
        try {
            llama_mtp_vocab_ranges(value, 512);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "invalid vocabulary range accepted");
    }
}

static void test_backend(ggml_backend_t backend, int columns, const std::vector<llama_mtp_vocab_range> & ranges) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({4*1024*1024, nullptr, true}), ggml_free);
    require(bool(ctx), "metadata allocation failed");
    auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q6_K, 256, 512);
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, columns);
    auto * baseline = ggml_mul_mat(ctx.get(), weight, input);
    auto * restricted = llama_mtp_vocab_logits(ctx.get(), weight, input, ranges);
    ggml_set_output(baseline);
    ggml_set_output(restricted);
    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, baseline);
    ggml_build_forward_expand(graph, restricted);
    require(ggml_backend_supports_op(backend, baseline), "backend lacks Q6_K matmul");
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "tensor allocation failed");
    std::vector<float> weights(ggml_nelements(weight)), values(ggml_nelements(input));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = std::sin(float(i)*0.031f) + float(i%11)*0.01f;
    }
    std::vector<uint8_t> raw(ggml_nbytes(weight));
    require(ggml_quantize_chunk(GGML_TYPE_Q6_K, weights.data(), raw.data(), 0, 512, 256, nullptr) == raw.size(),
            "quantization failed");
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::cos(float(i)*0.017f) + 0.17f;
    }
    ggml_backend_tensor_set(weight, raw.data(), 0, raw.size());
    ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
    for (int repeat = 0; repeat < 3; ++repeat) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "head graph failed");
        ggml_backend_synchronize(backend);
        std::vector<float> expected(ggml_nelements(baseline)), actual(ggml_nelements(restricted));
        ggml_backend_tensor_get(baseline, expected.data(), 0, expected.size()*sizeof(float));
        ggml_backend_tensor_get(restricted, actual.data(), 0, actual.size()*sizeof(float));
        double error = 0, norm = 0, maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const int64_t row = i % weight->ne[1];
            const bool kept = std::any_of(ranges.begin(), ranges.end(),
                    [row](const llama_mtp_vocab_range & range) { return range.first <= row && row < range.second; });
            if (!kept) {
                require(std::isinf(actual[i]) && std::signbit(actual[i]), "omitted ID is not negative infinity");
            } else {
                require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "kept ID is not finite");
                const double delta = double(actual[i]) - expected[i];
                error += delta*delta;
                norm += double(expected[i])*expected[i];
                maximum = std::max(maximum, std::abs(delta));
            }
        }
        require(error/std::max(norm, 1e-30) <= 5e-4, "kept-row projection differs");
        std::printf("mtp-vocab-test: backend=%s columns=%d ranges=%zu repeat=%d nmse=%.9g max_abs=%.9g\n",
                    ggml_backend_name(backend), columns, ranges.size(), repeat, error/std::max(norm, 1e-30), maximum);
    }
}

int main() {
    try {
        test_ranges();
        ggml_backend_load_all();
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(cpu != nullptr, "CPU backend unavailable");
        std::vector<ggml_backend_dev_t> devices = {cpu};
        if (auto * gpu = ggml_backend_dev_by_name("ROCm0")) {
            devices.push_back(gpu);
        }
        for (auto * device : devices) {
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
                    ggml_backend_dev_init(device, nullptr), ggml_backend_free);
            require(bool(backend), "backend initialization failed");
            for (int columns : {1, 2, 3, 4, 8}) {
                test_backend(backend.get(), columns, {{0, 128}, {480, 512}});
                test_backend(backend.get(), columns, {{17, 113}, {477, 511}});
                test_backend(backend.get(), columns, {{0, 512}});
            }
        }
        std::printf("PASS: strict ranges, Q6_K full-K row views, original IDs and masked gaps\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
