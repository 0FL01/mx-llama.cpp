#include "arg.h"
#include "common.h"
#include "json.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    params.warmup = false;
    const char * fixtures_path = std::getenv("LLAMA_MOE_PREFILL_FIXTURES");
    const char * output_prefix = std::getenv("LLAMA_MOE_PREFILL_OUTPUT_PREFIX");
    const bool has_speculation = std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type type) { return type != COMMON_SPECULATIVE_TYPE_NONE; });
    if (!fixtures_path || !output_prefix || params.n_batch < 64 || params.n_ubatch < 64 ||
            params.n_parallel != 1 || params.n_predict < 1 || has_speculation) {
        std::fprintf(stderr, "Expected target-only parameters and explicit fixture/output paths\n");
        return 1;
    }
    try {
        std::ifstream file(fixtures_path);
        if (!file) {
            throw std::runtime_error("cannot open fixed token fixtures");
        }
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        const auto fixtures = common_json::parse(text);
        llama_backend_init();
        llama_numa_init(params.numa);
        // Direct context initialization avoids the server's rollback-capability probe.
        auto init = common_init_from_params(params, true);
        auto * model = init->model();
        if (!model) {
            throw std::runtime_error("target model initialization failed");
        }
        const auto * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        for (const char * name : { "pp_512", "pp_4096", "pp_16384" }) {
            const auto cparams = common_context_params_to_llama(params);
            llama_context_ptr context(llama_init_from_model(model, cparams));
            auto * ctx = context.get();
            if (!ctx || !llama_get_memory(ctx)) {
                throw std::runtime_error("fresh target context initialization failed");
            }
            auto tokens = fixtures[name]["tokens"].get<std::vector<llama_token>>();
            const size_t remainder = tokens.size() % std::min(params.n_batch, params.n_ubatch);
            if (tokens.empty() || tokens.size() > llama_n_ctx(ctx) || (remainder > 0 && remainder < 64)) {
                throw std::runtime_error("fixture must contain only wide prefill ubatches");
            }
            for (const auto token : tokens) {
                if (token < 0 || token >= n_vocab) {
                    throw std::runtime_error("invalid fixture token ID");
                }
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_perf_context_reset(ctx);
            llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
            const int64_t begin_us = ggml_time_us();
            for (size_t offset = 0; offset < tokens.size(); offset += params.n_batch) {
                common_batch_clear(batch);
                const size_t count = std::min((size_t) params.n_batch, tokens.size() - offset);
                for (size_t i = 0; i < count; ++i) {
                    common_batch_add(batch, tokens[offset + i], offset + i, { 0 }, i + 1 == count);
                }
                if (llama_decode(ctx, batch) != 0) {
                    llama_batch_free(batch);
                    throw std::runtime_error("prefill decode failed");
                }
            }
            const float * logits = llama_get_logits_ith(ctx, -1);
            const int64_t end_us = ggml_time_us();
            llama_batch_free(batch);
            const auto perf = llama_perf_context(ctx);
            if (!logits || perf.n_p_eval != (int32_t) tokens.size() ||
                    !std::isfinite(perf.t_p_eval_ms) || perf.t_p_eval_ms <= 0.0) {
                throw std::runtime_error("invalid prefill timing/output");
            }
            for (int32_t i = 0; i < n_vocab; ++i) {
                if (!std::isfinite(logits[i])) {
                    throw std::runtime_error("nonfinite target logit");
                }
            }
            const std::string path = std::string(output_prefix) + "-" + name + ".f32";
            FILE * output = std::fopen(path.c_str(), "wb");
            if (!output) {
                throw std::runtime_error("cannot open output snapshot");
            }
            const bool written = std::fwrite(logits, sizeof(float), n_vocab, output) == (size_t) n_vocab;
            const int closed = std::fclose(output);
            if (!written || closed != 0) {
                throw std::runtime_error("output snapshot write failed");
            }
            std::vector<llama_token> generated;
            llama_perf_context_reset(ctx);
            const int64_t generation_begin_us = ggml_time_us();
            for (int32_t step = 0; step < params.n_predict; ++step) {
                llama_token token = LLAMA_TOKEN_NULL;
                float maximum = -INFINITY;
                for (int32_t i = 0; i < n_vocab; ++i) {
                    if (!llama_vocab_is_eog(vocab, i) && logits[i] > maximum) {
                        token = i;
                        maximum = logits[i];
                    }
                }
                if (token == LLAMA_TOKEN_NULL) {
                    throw std::runtime_error("no valid greedy token");
                }
                generated.push_back(token);
                if (step + 1 < params.n_predict) {
                    llama_token input = token;
                    if (llama_decode(ctx, llama_batch_get_one(&input, 1)) != 0) {
                        throw std::runtime_error("post-prefill generation failed");
                    }
                    logits = llama_get_logits_ith(ctx, -1);
                    for (int32_t i = 0; i < n_vocab; ++i) {
                        if (!std::isfinite(logits[i])) {
                            throw std::runtime_error("nonfinite generation logit");
                        }
                    }
                }
            }
            const int64_t generation_end_us = ggml_time_us();
            const auto generation_perf = llama_perf_context(ctx);
            if (generation_perf.n_eval != std::max(1, params.n_predict - 1) ||
                    !std::isfinite(generation_perf.t_eval_ms) ||
                    (params.n_predict == 1 ? generation_perf.t_eval_ms != 0.0 : generation_perf.t_eval_ms <= 0.0)) {
                throw std::runtime_error("invalid generation counters");
            }
            std::printf("{\"case\":\"%s\",\"prompt_n\":%zu,\"prompt_ms\":%.9g,"
                        "\"prompt_per_second\":%.9g,\"wall_ms\":%.9g,\"begin_us\":%lld,"
                        "\"end_us\":%lld,\"argmax\":%d,\"n_vocab\":%d,\"predicted_n\":%d,"
                        "\"predicted_ms\":%.9g,\"generation_wall_ms\":%.9g,\"tokens\":[",
                        name, tokens.size(), perf.t_p_eval_ms, tokens.size() * 1000.0 / perf.t_p_eval_ms,
                        (end_us - begin_us) / 1000.0, (long long) begin_us, (long long) end_us,
                        generated.front(), n_vocab, params.n_predict, generation_perf.t_eval_ms,
                        (generation_end_us - generation_begin_us) / 1000.0);
            for (size_t i = 0; i < generated.size(); ++i) {
                std::printf("%s%d", i ? "," : "", generated[i]);
            }
            std::printf("]}\n");
            std::fflush(stdout);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Cold prefill validation failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
