// Manual real-model routing/logit oracle. Run with identical common CLI flags
// and GGML_CUDA_REPACK_Q4_0=0/1; stdout is structured JSONL, logs use stderr.
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "json.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <vector>

static bool routing_callback(ggml_tensor * t, bool ask, void *) {
    const bool routing = t->type == GGML_TYPE_I32 &&
        std::strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const char * trace_dir = std::getenv("Q4_REPACK_TEST_TRACE_DIR");
    // A JSON allowlist selects readback-supported outputs; partial-reduction
    // tensors are not readable until their AllReduce boundary has closed.
    static const std::set<std::string> trace_names = [] {
        const char * names = std::getenv("Q4_REPACK_TEST_TRACE_NAMES");
        const auto values = common_json::parse(names ? names : "[]").get<std::vector<std::string>>();
        return std::set<std::string>(values.begin(), values.end());
    }();
    const bool trace = trace_dir != nullptr && t->type == GGML_TYPE_F32 &&
                       trace_names.count(t->name) > 0 && ggml_is_contiguous(t) &&
                       ggml_nelements(t) > 0 && t->ne[2] == 1 && t->ne[3] == 1;
    const bool selected = routing || trace;
    if (ask && std::getenv("Q4_REPACK_TEST_TRACE_GRAPH") && t->op == GGML_OP_MUL_MAT) {
        static std::set<std::string> graph_seen;
        if (graph_seen.insert(t->name).second) {
            std::printf("{\"matmul\":\"%s\",\"weight\":\"%s\",\"input\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld]}\n",
                t->name, t->src[0]->name, t->src[1]->name, (long long)t->ne[0],
                (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
        }
    }
    if (ask) return selected;
    if (!selected) return true;
    if (trace) {
        static std::map<std::string, size_t> seen;
        const size_t occurrence = seen[t->name]++;
        if (occurrence < 2) {
            std::string name(t->name);
            std::replace(name.begin(), name.end(), '/', '_');
            if (occurrence != 0) name += "." + std::to_string(occurrence);
            const std::string path = std::string(trace_dir) + "/" + name + ".bin";
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
            FILE * file = std::fopen(path.c_str(), "wb");
            if (file == nullptr) return false;
            const bool written = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
            const int closed = std::fclose(file);
            if (!written || closed != 0) return false;
            std::printf("{\"trace\":\"%s\",\"op\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"nb\":[%zu,%zu,%zu,%zu]}\n",
                t->name, ggml_op_name(t->op), (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], (long long) t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
        }
        return true;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    std::printf("{\"routing\":\"%s\",\"ids\":[", t->name);
    bool first = true;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        int32_t id;
        std::memcpy(&id, bytes.data() + i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0], sizeof(id));
        std::printf("%s%d", first ? "" : ",", id);
        first = false;
    }
    std::printf("]}\n");
    return true;
}

int main(int argc, char ** argv) {
    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    params.cb_eval = routing_callback;
    params.warmup = false;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    auto * model = init->model();
    if (!ctx || !model || params.n_predict <= 0) return 1;
    const auto * vocab = llama_model_get_vocab(model);
    auto tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (const char * path = std::getenv("Q4_REPACK_TEST_TOKENS_PATH")) {
        try {
            std::ifstream file(path);
            if (!file) throw std::runtime_error("cannot open token fixture");
            const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            tokens = common_json::parse(text).get<std::vector<llama_token>>();
        } catch (const std::exception & error) {
            std::fprintf(stderr, "Invalid token fixture: %s\n", error.what());
            return 1;
        }
        for (const auto token : tokens) {
            if (token < 0 || token >= llama_vocab_n_tokens(vocab)) return 1;
        }
    }
    if (tokens.empty()) return 1;
    if (std::getenv("Q4_REPACK_TEST_PPL")) {
        // Teacher forcing keeps inputs identical even when greedy tokens differ.
        llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
        double nll = 0.0;
        size_t scored = 0;
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (size_t offset = 0; offset < tokens.size(); offset += params.n_batch) {
            const size_t count = std::min((size_t)params.n_batch, tokens.size() - offset);
            common_batch_clear(batch);
            for (size_t i = 0; i < count; ++i) {
                common_batch_add(batch, tokens[offset+i], offset+i, {0}, true);
            }
            if (llama_decode(ctx, batch)) { llama_batch_free(batch); return 1; }
            for (size_t i = 0; i < count && offset+i+1 < tokens.size(); ++i) {
                const float * logits = llama_get_logits_ith(ctx, i);
                float maximum = -INFINITY;
                for (int j = 0; j < n_vocab; ++j) {
                    if (!std::isfinite(logits[j])) { llama_batch_free(batch); return 1; }
                    maximum = std::max(maximum, logits[j]);
                }
                double sum = 0.0;
                for (int j = 0; j < n_vocab; ++j) sum += std::exp((double)logits[j] - maximum);
                const double loss = maximum + std::log(sum) - logits[tokens[offset+i+1]];
                nll += loss;
                ++scored;
                std::printf("{\"ppl_position\":%zu,\"nll\":%.12g}\n", offset+i, loss);
            }
        }
        llama_batch_free(batch);
        if (scored == 0 || !std::isfinite(nll)) return 1;
        std::printf("{\"ppl_tokens\":%zu,\"mean_nll\":%.12g,\"perplexity\":%.12g}\n", scored, nll/scored, std::exp(nll/scored));
        return 0;
    }
    for (size_t offset = 0; offset < tokens.size(); offset += params.n_batch) {
        const size_t count = std::min((size_t) params.n_batch, tokens.size() - offset);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + offset, count))) return 1;
    }
    for (int step = 0; step < params.n_predict; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_vocab; ++i) if (!std::isfinite(logits[i])) return 1;
        // Optional full-vocabulary snapshot for a same-input numerical oracle.
        // Later autoregressive steps are not comparable after an argmax diverges.
        if (step == 0) {
            const char * path = std::getenv("Q4_REPACK_TEST_LOGITS_PATH");
            if (path != nullptr) {
                FILE * file = std::fopen(path, "wb");
                if (file == nullptr) return 1;
                const bool written = std::fwrite(logits, sizeof(float), n_vocab, file) == (size_t) n_vocab;
                const int closed = std::fclose(file);
                if (!written || closed != 0) return 1;
            }
        }
        llama_token token = std::max_element(logits, logits + n_vocab) - logits;
        std::printf("{\"step\":%d,\"argmax\":%d,\"logit\":%.9g}\n", step, token, logits[token]);
        if (step + 1 < params.n_predict && llama_decode(ctx, llama_batch_get_one(&token, 1))) return 1;
    }
    std::fflush(stdout);
    return 0;
}
