#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <unordered_map>

// No weights, kernels or backend allocation: exercise the real KV metadata and
// get_prev_tokens(), not a second implementation of the candidate lookup.
struct metadata_model : llama_model {
    metadata_model() : llama_model(llama_model_default_params()) { hparams = {}; }
    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return nullptr; }
};

struct query {
    std::vector<llama_pos> pos;
    std::vector<llama_token> tok;
    std::vector<llama_seq_id *> seq;
    std::vector<int32_t> counts;
    llama_seq_id id;
    llama_ubatch batch = {};
    query(std::vector<llama_pos> p, llama_seq_id s, bool embd = false) : pos(std::move(p)), tok(pos.size(), 17),
        seq(pos.size()), counts(pos.size(), 1), id(s) {
        for (auto & x : seq) { x = &id; }
        batch.n_tokens = batch.n_seq_tokens = pos.size();
        batch.n_seqs = batch.n_seqs_unq = 1;
        batch.n_pos = 1;
        batch.pos = pos.data();
        batch.token = embd ? nullptr : tok.data();
        batch.seq_id = seq.data();
        batch.seq_id_unq = &id;
        batch.n_seq_id = counts.data();
    }
};

// The original scan/window algorithm. Keep its distinct duplicate tie rules.
static void reference(const llama_kv_cache & kv, const llama_ubatch & b, uint32_t n, std::vector<llama_token> & out) {
    out.assign(b.n_tokens*n, LLAMA_TOKEN_NULL);
    if (!n) { return; }
    const auto bounds = std::minmax_element(b.pos, b.pos + b.n_tokens);
    const llama_pos w0 = *bounds.first - n, pmax = *bounds.second;
    std::bitset<LLAMA_MAX_SEQ> seqs;
    for (uint32_t s = 0; s < b.n_seqs_unq; ++s) { seqs.set(b.seq_id_unq[s]); }
    std::unordered_map<uint64_t, llama_token> hist;
    const auto key = [](llama_seq_id s, llama_pos p) { return (uint64_t(s) << 32) | uint32_t(p); };
    std::array<std::pair<llama_pos, llama_token>, LLAMA_MAX_SEQ> below;
    below.fill({-1, LLAMA_TOKEN_NULL});
    for (uint32_t s = 0; s < kv.get_n_stream(); ++s) {
        kv.get_cells(s).for_each_token_in(seqs, 0, pmax + 1, [&](llama_seq_id id, llama_pos p, llama_token t) {
            if (p >= w0) { hist[key(id, p)] = t; }
            else if (p > below[id].first) { below[id] = {p, t}; }
        });
    }
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> ord;
    for (uint32_t i = 0; i < b.n_tokens; ++i) {
        const auto id = b.seq_id[i][0];
        auto & prior = ord[id];
        for (uint32_t j = 0; j < n; ++j) {
            const int d = n-j;
            llama_pos p = b.pos[i]-d;
            if (!b.token) {
                const int64_t k = int64_t(prior.size())-d;
                const auto first = prior.empty() ? i : prior.front();
                p = k >= 0 ? b.pos[prior[k]] : b.pos[first]+k;
            }
            if (p < 0) { continue; }
            llama_token t = below[id].second;
            for (llama_pos q = p; q >= w0; --q) {
                const auto it = hist.find(key(id, q));
                if (it != hist.end()) { t = it->second; break; }
            }
            out[i*n+j] = t;
        }
        prior.push_back(i);
    }
}

static std::vector<int64_t> hash_inputs(const query & q, const std::vector<llama_token> & prev, uint32_t n) {
    // Same missing/EOS cut as qwen4exp's PLE hash consumer. Multipliers and
    // head-vocab modular arithmetic are unchanged by the production patch.
    std::vector<int64_t> out;
    constexpr llama_token eos = 9;
    for (uint32_t i = 0; i < q.batch.n_tokens; ++i) {
        out.push_back(q.tok[i]);
        bool cut = false;
        for (uint32_t s = 1; s <= n; ++s) {
            const auto t = cut ? LLAMA_TOKEN_NULL : prev[i*n+n-s];
            cut = cut || t < 0 || t == eos;
            out.push_back(cut ? eos : t);
        }
    }
    return out;
}

static void check(const llama_kv_cache & kv, query & q) {
    for (uint32_t n : {0, 1, 3, 7}) {
        std::vector<llama_token> a, b;
        reference(kv, q.batch, n, a);
        kv.get_prev_tokens(q.batch, n, b);
        GGML_ASSERT(a == b);
        GGML_ASSERT(hash_inputs(q, a, n) == hash_inputs(q, b, n));
    }
}

static void put(llama_kv_cache & kv, uint32_t cell, llama_pos pos, llama_token tok, llama_seq_id seq) {
    // The object is mutable; only the inspection accessor is const. Avoid a
    // new production mutation API just for constructing adversarial layouts.
    auto & cells = const_cast<llama_kv_cells &>(kv.get_cells(seq));
    if (!cells.is_empty(cell)) { cells.rm(cell); }
    cells.pos_set(cell, pos);
    llama_kv_cell_ext ext; ext.tok = tok;
    cells.ext_set(cell, ext);
    cells.seq_add(cell, seq);
}

int main(int argc, char **) {
    metadata_model model;
    llama_kv_cache kv(model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32, false, false, true,
                      65536, 4, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr);
    auto check_all = [&] {
        for (llama_seq_id s = 0; s < 4; ++s) {
            for (bool embd : {false, true}) {
                for (int p = 0; p < 42; ++p) {
                    query q({p, p, p+2}, s, embd); check(kv, q);
                }
            }
        }
    };
    check_all(); // sequence start and no predecessors
    for (int i = 0; i < 24; ++i) { put(kv, (i*7)%29, i/2, i%5 ? i+10 : LLAMA_TOKEN_NULL, i%2); }
    check_all(); // duplicates with reversed cell order, gaps and missing token IDs
    kv.seq_cp(0, 2, 0, -1); check_all();
    kv.seq_rm(2, 2, 4); check_all();
    kv.seq_add(0, 0, -1, 5); check_all();
    kv.seq_div(0, 0, -1, 3); check_all();
    auto & cells = const_cast<llama_kv_cells &>(kv.get_cells(0));
    cells.reset_shift();
    auto saved = cells.cp(0, cells.size());
    auto scattered = cells.cp(std::vector<uint32_t>{0, 7, 14});
    kv.seq_keep(2); check_all();
    cells.set(0, saved); check_all();
    cells.set(std::vector<uint32_t>{1, 8, 15}, scattered); check_all();
    for (int accepted = 0; accepted <= 2; ++accepted) {
        kv.clear(false);
        for (int i = 0; i < 12; ++i) { put(kv, i, i, 20+i, 0); }
        auto prefix = cells.cp(0, cells.size());
        for (int i = 0; i < 3; ++i) { put(kv, 12+i, 12+i, 40+i, 0); }
        kv.seq_rm(0, 12+accepted, -1); check_all();
        cells.set(0, prefix); check_all();
        // Four re-evaluated prefix tokens after state restore.
        for (int i = 8; i < 12; ++i) { put(kv, i, i, 20+i, 0); }
        check_all();
    }
    std::mt19937 rng(4242);
    for (int iteration = 0; iteration < 120; ++iteration) {
        const auto cell = rng()%32;
        put(kv, cell, rng()%32, rng()%21 == 0 ? LLAMA_TOKEN_NULL : rng()%100, rng()%4);
        if (iteration%10 == 0) { check_all(); }
    }
    puts("PLE history: exact predecessor/hash-input equivalence, duplicate ties, gaps, mutations and rollback passed");
    if (argc > 1) {
        using clock = std::chrono::steady_clock;
        kv.clear(false);
        const auto start = clock::now();
        for (int i = 0; i < 65536; ++i) { put(kv, i, i, i%50000, 0); }
        const auto built = clock::now();
        std::vector<llama_token> result;
        for (int nt = 1; nt <= 3; ++nt) {
            std::vector<llama_pos> pos;
            for (int i = 0; i < nt; ++i) { pos.push_back(65530+i); }
            query q(pos, 0);
            for (bool ref : {true, false}) {
                const auto t = clock::now();
                for (int rep = 0; rep < 32; ++rep) {
                    if (ref) { reference(kv, q.batch, 3, result); }
                    else { kv.get_prev_tokens(q.batch, 3, result); }
                }
                printf("nt=%d reference=%d us/call=%.3f\n", nt, ref,
                    std::chrono::duration<double, std::micro>(clock::now()-t).count()/32);
            }
        }
        printf("populate65536_ms=%.3f\n", std::chrono::duration<double, std::milli>(built-start).count());
    }
}
