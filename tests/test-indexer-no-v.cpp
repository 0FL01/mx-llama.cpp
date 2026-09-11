#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-io.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Weightless CPU allocations and the production memory serializers; no graph execution.
struct metadata_model : llama_model {
    metadata_model(bool draft = false) : llama_model(llama_model_default_params()) {
        hparams = {};
        hparams.n_layer_all = 2;
        hparams.n_layer_nextn = draft ? 1 : 0;
        hparams.n_embd_head_k_full = 256;
        hparams.n_embd_head_v_full = hparams.indexer_head_size = 128;
        hparams.n_head_arr.fill(2);
        hparams.n_head_kv_arr.fill(2);
    }
    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return nullptr; }
};

struct tensor_record {
    std::string name;
    size_t at;
};

struct buffer_writer : llama_io_write_i {
    std::vector<uint8_t> data;
    std::vector<tensor_record> tensors;
    void write(const void * src, size_t size) override {
        const auto * p = static_cast<const uint8_t *>(src);
        data.insert(data.end(), p, p + size);
    }
    void write_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        tensors.push_back({t->view_src ? t->view_src->name : t->name, data.size()});
        const size_t at = data.size();
        data.resize(at + size);
        ggml_backend_tensor_get(t, data.data() + at, offset, size);
    }
    size_t n_bytes() override { return data.size(); }
};

struct buffer_reader : llama_io_read_i {
    const std::vector<uint8_t> & data;
    size_t at = 0;
    std::vector<std::string> tensors;
    explicit buffer_reader(const std::vector<uint8_t> & data) : data(data) {}
    void check(size_t size) const {
        if (size > data.size() - at) { throw std::runtime_error("test buffer exhausted"); }
    }
    void read(void * dst, size_t size) override {
        check(size);
        std::memcpy(dst, data.data() + at, size);
        at += size;
    }
    void read_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        check(size);
        tensors.emplace_back(t->name);
        ggml_backend_tensor_set(t, data.data() + at, offset, size);
        at += size;
    }
    size_t n_bytes() override { return at; }
};

static void set_switch(const char * value) {
#ifdef _WIN32
    GGML_ASSERT(_putenv_s("LLAMA_INDEXER_NO_V", value ? value : "") == 0);
#else
    GGML_ASSERT(value ? setenv("LLAMA_INDEXER_NO_V", value, 1) == 0 : unsetenv("LLAMA_INDEXER_NO_V") == 0);
#endif
}

static std::unique_ptr<llama_memory_hybrid_idx> make(metadata_model & model, const char * value) {
    set_switch(value);
    const uint32_t il = model.hparams.n_layer_nextn ? 1 : 0;
    return std::make_unique<llama_memory_hybrid_idx>(model,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, 64, 16, 0, LLAMA_SWA_TYPE_NONE,
        GGML_TYPE_F32, GGML_TYPE_F32, 8, 2, 2, false, true,
        [il](uint32_t i) { return i == il; }, [](uint32_t) { return false; },
        [il](uint32_t i) { return i == il; });
}

static buffer_writer save(const llama_memory_i & memory, llama_seq_id seq = -1, llama_state_seq_flags flags = 0) {
    buffer_writer out;
    memory.state_write(out, seq, flags);
    return out;
}

static ggml_tensor * v_storage(llama_kv_cache & cache, ggml_context * ctx, int il) {
    llama_kv_cache::slot_info slot = {};
    slot.s0 = slot.s1 = 0;
    auto * view = cache.get_v(ctx, il, cache.get_size(), slot);
    GGML_ASSERT(view->view_src);
    return view->view_src;
}

static std::vector<uint8_t> bytes(ggml_tensor * t) {
    std::vector<uint8_t> result(ggml_nbytes(t));
    ggml_backend_tensor_get(t, result.data(), 0, result.size());
    return result;
}

static void fill_tensor(ggml_tensor * t, uint8_t seed) {
    std::vector<uint8_t> data(ggml_nbytes(t));
    for (size_t i = 0; i < data.size(); ++i) { data[i] = uint8_t(seed + i*13 + i/31); }
    ggml_backend_tensor_set(t, data.data(), 0, data.size());
}

static void seed_cache(llama_kv_cache & cache, bool has_v, uint8_t seed) {
    cache.clear(true);
    auto & cells = const_cast<llama_kv_cells &>(cache.get_cells(0));
    for (uint32_t i = 0; i < 4; ++i) {
        cells.pos_set(2*i, 10 + i);
        cells.seq_add(2*i, 0);
    }
    ggml_context_ptr ctx(ggml_init({16*ggml_tensor_overhead(), nullptr, true}));
    for (auto il : cache.get_layer_ids()) {
        fill_tensor(cache.get_k_storage(il), seed);
        if (has_v) { fill_tensor(v_storage(cache, ctx.get(), il), seed + 7); }
    }
}

static void seed_memory(llama_memory_hybrid_idx & memory, bool no_v) {
    seed_cache(*memory.get_mem_attn(), true, 17);
    seed_cache(*memory.get_mem_idx(), !no_v, 39);
}

static void fragmented_destination(llama_kv_cache & cache) {
    cache.clear(false);
    auto & cells = const_cast<llama_kv_cells &>(cache.get_cells(1));
    for (uint32_t i = 0; i < cache.get_size(); i += 2) {
        cells.pos_set(i, 100 + i);
        cells.seq_add(i, 1);
    }
}

static size_t header_offset(const buffer_writer & out, const char * name) {
    for (const auto & record : out.tensors) {
        if (record.name == name) {
            // First K payload follows layout, layer count, type and row size.
            GGML_ASSERT(record.at >= 3*sizeof(uint32_t) + sizeof(uint64_t));
            return record.at - 3*sizeof(uint32_t) - sizeof(uint64_t);
        }
    }
    GGML_ABORT("missing K payload");
}

static uint32_t word(const std::vector<uint8_t> & data, size_t offset) {
    GGML_ASSERT(offset + sizeof(uint32_t) <= data.size());
    uint32_t value;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

static void same_shape(ggml_tensor * a, ggml_tensor * b) {
    GGML_ASSERT(a->type == b->type && ggml_nbytes(a) == ggml_nbytes(b));
    for (int d = 0; d < GGML_MAX_DIMS; ++d) { GGML_ASSERT(a->ne[d] == b->ne[d] && a->nb[d] == b->nb[d]); }
}

static void allocation_test(metadata_model & model) {
    auto off = make(model, nullptr);
    auto on = make(model, "1");
    auto zero = make(model, "0");
    auto other = make(model, "true");
    const auto il = off->get_mem_idx()->get_layer_ids().at(0);
    auto * ka = off->get_mem_idx()->get_k_storage(il);
    auto * kb = on->get_mem_idx()->get_k_storage(il);
    same_shape(ka, kb);
    GGML_ASSERT(ka->ne[0] == 128 && ka->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(ka->data != kb->data && ka->buffer != kb->buffer);
    GGML_ASSERT(kb->buffer != on->get_mem_attn()->get_k_storage(il)->buffer);
    GGML_ASSERT(&on->get_mem_idx()->get_cells(0) != &on->get_mem_attn()->get_cells(0));
    GGML_ASSERT(!model.hparams.is_mla() && model.hparams.n_head_kv(il) == 2);

    ggml_context_ptr ctx(ggml_init({16*ggml_tensor_overhead(), nullptr, true}));
    auto * va = v_storage(*off->get_mem_idx(), ctx.get(), il);
    const auto buft = ggml_backend_buffer_get_type(ka->buffer);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t v_alloc = ggml_backend_buft_get_alloc_size(buft, va);
    const size_t removed = (v_alloc + alignment - 1)/alignment*alignment;
    GGML_ASSERT(removed > 0);
    GGML_ASSERT(ggml_backend_buffer_get_size(ka->buffer) - ggml_backend_buffer_get_size(kb->buffer) == removed);
    GGML_ASSERT(off->memory_breakdown().at(buft) - on->memory_breakdown().at(buft) == removed);
    GGML_ASSERT(off->get_mem_attn()->memory_breakdown() == on->get_mem_attn()->memory_breakdown());
    GGML_ASSERT(off->memory_breakdown() == zero->memory_breakdown());
    GGML_ASSERT(off->memory_breakdown() == other->memory_breakdown());
    same_shape(off->get_mem_attn()->get_k_storage(il), on->get_mem_attn()->get_k_storage(il));
    same_shape(v_storage(*off->get_mem_attn(), ctx.get(), il), v_storage(*on->get_mem_attn(), ctx.get(), il));

    seed_memory(*off, false);
    seed_memory(*on, true);
    seed_memory(*zero, false);
    seed_memory(*other, false);
    GGML_ASSERT(save(*off).data == save(*zero).data && save(*off).data == save(*other).data);
    GGML_ASSERT(save(*off->get_mem_attn()).data == save(*on->get_mem_attn()).data);
    GGML_ASSERT(bytes(ka) == bytes(kb));
    const auto legacy = save(*off->get_mem_idx());
    const auto modern = save(*on->get_mem_idx());
    const size_t h = header_offset(legacy, ka->name);
    GGML_ASSERT(h == header_offset(modern, kb->name));
    GGML_ASSERT(word(legacy.data, h) == 0 && word(legacy.data, h + 4) == 1);
    GGML_ASSERT(word(modern.data, h) == 4 && word(modern.data, h + 4) == ((1u << 31) | 1u));
    GGML_ASSERT(legacy.tensors.size() == 8 && modern.tensors.size() == 4);
    GGML_ASSERT(legacy.data.size() - modern.data.size() == 12 + 4*ggml_row_size(va->type, va->ne[0]));

    // These are the exact baseline header predicates, including its bool cast.
    for (bool transposed : {false, true}) {
        const bool legacy_accepts = word(modern.data, h + 4) == 1 && transposed == (bool) word(modern.data, h);
        GGML_ASSERT(!legacy_accepts);
    }
    std::printf("%s CPU indexer allocation delta: %zu bytes\n", model.hparams.n_layer_nextn ? "draft" : "target", removed);
}

static void roundtrip_test(metadata_model & model, bool no_v, llama_seq_id seq) {
    auto src = make(model, no_v ? "1" : "0");
    auto dst = make(model, no_v ? "1" : "0");
    seed_memory(*src, no_v);
    seed_memory(*dst, no_v);
    if (seq >= 0) {
        fragmented_destination(*dst->get_mem_attn());
        fragmented_destination(*dst->get_mem_idx());
    }
    const auto untouched_attn = save(*dst->get_mem_attn(), 1);
    const auto untouched_idx = save(*dst->get_mem_idx(), 1);
    auto out = save(*src, seq);
    const size_t end = out.n_bytes();
    src->get_mem_attn()->state_write(out, seq); // A real following section must stay aligned.
    buffer_reader in(out.data);
    dst->state_read(in, seq);
    GGML_ASSERT(in.n_bytes() == end);
    GGML_ASSERT(save(*src, seq).data == save(*dst, seq).data);
    dst->get_mem_attn()->state_read(in, seq);
    GGML_ASSERT(in.n_bytes() == out.n_bytes());
    if (seq >= 0) {
        GGML_ASSERT(save(*dst->get_mem_attn(), 1).data == untouched_attn.data);
        GGML_ASSERT(save(*dst->get_mem_idx(), 1).data == untouched_idx.data);
        const auto & a = dst->get_mem_attn()->get_cells(0);
        const auto & b = dst->get_mem_idx()->get_cells(0);
        for (uint32_t i = 0; i < a.size(); ++i) {
            GGML_ASSERT(a.is_empty(i) == b.is_empty(i));
            GGML_ASSERT(a.seq_has(i, 0) == b.seq_has(i, 0));
            if (!a.is_empty(i)) { GGML_ASSERT(a.pos_get(i) == b.pos_get(i)); }
        }
    }

    // PARTIAL_ONLY is recurrent-only; indexer bytes are intentionally absent in both modes.
    const auto before = save(*dst->get_mem_idx());
    auto partial = save(*src, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    buffer_reader pin(partial.data);
    dst->state_read(pin, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    GGML_ASSERT(pin.n_bytes() == partial.n_bytes() && pin.tensors.empty());
    GGML_ASSERT(save(*dst->get_mem_idx()).data == before.data);
}

static void rejection_test(metadata_model & model, bool source_no_v, llama_seq_id seq, bool hybrid) {
    auto src = make(model, source_no_v ? "1" : "0");
    auto dst = make(model, source_no_v ? "0" : "1");
    seed_memory(*src, source_no_v);
    seed_memory(*dst, !source_no_v);
    const auto il = src->get_mem_idx()->get_layer_ids().at(0);
    const auto before = bytes(dst->get_mem_idx()->get_k_storage(il));
    auto out = hybrid ? save(*src, seq) : save(*src->get_mem_idx(), seq);
    const size_t h = header_offset(out, src->get_mem_idx()->get_k_storage(il)->name);
    const size_t section_end = out.n_bytes();
    src->get_mem_attn()->state_write(out, seq);
    buffer_reader in(out.data);
    bool rejected = false;
    try {
        if (hybrid) { dst->state_read(in, seq); }
        else { dst->get_mem_idx()->state_read(in, seq); }
    } catch (const std::runtime_error & e) {
        rejected = std::string(e.what()) == "failed to restore kv cache";
    }
    GGML_ASSERT(rejected && in.n_bytes() == h + 8 && in.n_bytes() < section_end);
    for (const auto & name : in.tensors) { GGML_ASSERT(name.find("cache_idx_") != 0); }
    if (!hybrid) { GGML_ASSERT(in.tensors.empty()); }
    if (seq >= 0) { GGML_ASSERT(bytes(dst->get_mem_idx()->get_k_storage(il)) == before); }
    GGML_ASSERT(dst->get_mem_idx()->seq_pos_max(0) == -1);
    if (hybrid) { GGML_ASSERT(dst->get_mem_attn()->seq_pos_max(0) == -1); }
}

static void true_mla_test() {
    metadata_model model;
    model.hparams.n_embd_head_k_mla_impl = model.hparams.n_embd_head_v_mla_impl = 128;
    auto off = make(model, "0");
    auto on = make(model, "1");
    seed_cache(*off->get_mem_idx(), false, 19);
    seed_cache(*on->get_mem_idx(), false, 19);
    const auto a = save(*off->get_mem_idx());
    const auto b = save(*on->get_mem_idx());
    GGML_ASSERT(a.data == b.data && off->memory_breakdown() == on->memory_breakdown());
    const size_t h = header_offset(a, off->get_mem_idx()->get_k_storage(0)->name);
    GGML_ASSERT(word(a.data, h) == 0 && word(a.data, h + 4) == 1);
    buffer_reader in(a.data);
    on->get_mem_idx()->state_read(in);
    GGML_ASSERT(in.n_bytes() == a.data.size() && save(*on->get_mem_idx()).data == a.data);
}

static void layout_test(bool mla, bool transposed) {
    metadata_model model;
    if (mla) {
        model.hparams.n_embd_head_k_mla_impl = model.hparams.n_embd_head_v_mla_impl = 128;
    }
    auto make_cache = [&] {
        return std::make_unique<llama_kv_cache>(model, model.hparams,
            GGML_TYPE_F16, GGML_TYPE_F16, transposed, false, true, 64, 2, 16, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, [](uint32_t il) { return il == 0; }, nullptr, nullptr);
    };
    auto src = make_cache();
    auto dst = make_cache();
    seed_cache(*src, !mla, 23);
    auto out = save(*src);
    const size_t h = header_offset(out, src->get_k_storage(0)->name);
    GGML_ASSERT(word(out.data, h) == (transposed ? 1u : 0u) && word(out.data, h + 4) == 1);
    const size_t end = out.n_bytes();
    src->state_write(out);
    buffer_reader in(out.data);
    dst->state_read(in);
    GGML_ASSERT(in.n_bytes() == end && save(*dst).data == save(*src).data);
    dst->state_read(in);
    GGML_ASSERT(in.n_bytes() == out.n_bytes());
}

static void empty_test(metadata_model & model) {
    auto off = make(model, "0");
    auto on = make(model, "1");
    auto a = save(*off);
    auto b = save(*on);
    GGML_ASSERT(a.data == b.data && a.tensors.empty());
    buffer_reader ain(a.data), bin(b.data);
    on->state_read(ain);
    off->state_read(bin);
    GGML_ASSERT(ain.n_bytes() == a.n_bytes() && bin.n_bytes() == b.n_bytes());
}

int main() {
    const char * env = std::getenv("LLAMA_INDEXER_NO_V");
    const bool had_env = env != nullptr;
    const std::string saved_env = env ? env : "";
    metadata_model target;
    metadata_model draft(true);
    for (auto * model : {&target, &draft}) {
        allocation_test(*model);
        empty_test(*model);
        for (bool on : {false, true}) {
            for (llama_seq_id seq : {-1, 0}) {
                roundtrip_test(*model, on, seq);
                for (bool hybrid : {false, true}) { rejection_test(*model, on, seq, hybrid); }
            }
        }
    }
    auto t = make(target, "1");
    auto d = make(draft, "1");
    auto * kt = t->get_mem_idx()->get_k_storage(0);
    auto * kd = d->get_mem_idx()->get_k_storage(1);
    GGML_ASSERT(kt->data != kd->data && kt->buffer != kd->buffer);
    GGML_ASSERT(&t->get_mem_idx()->get_cells(0) != &d->get_mem_idx()->get_cells(0));
    seed_cache(*t->get_mem_idx(), false, 29);
    const auto saved_target = save(*t->get_mem_idx());
    seed_cache(*d->get_mem_idx(), false, 97);
    GGML_ASSERT(save(*t->get_mem_idx()).data == saved_target.data);
    true_mla_test();
    for (bool mla : {false, true}) {
        for (bool transposed : {false, true}) { layout_test(mla, transposed); }
    }
    set_switch(had_env ? saved_env.c_str() : nullptr);
    std::puts("indexer no-V allocation, ownership and serializer checks passed");
}
