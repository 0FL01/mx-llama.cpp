#include "llama-moecache.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

std::unique_ptr<llama_moe_cache> llama_moe_cache::create(const llama_model & model, int32_t slots, int32_t inserts) {
    if (slots <= 0 || model.hparams.no_alloc) {
        return nullptr;
    }
    if (uint32_t(slots) > model.hparams.n_expert || inserts <= 0) {
        throw std::runtime_error("invalid MoE cache slots/inserts");
    }
    auto cache = std::make_unique<llama_moe_cache>();
    cache->max_inserts = inserts;
    const auto canonical_host = [](const ggml_tensor * t, ggml_backend_buffer_type_t host_buft) {
        return t && t->data && t->buffer && ggml_backend_buffer_is_host(t->buffer) &&
            ggml_is_contiguous(t) && t->ne[3] == 1 &&
            // Only canonical CPU or device-pinned host storage, never CPU_REPACK.
            (ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cpu_buffer_type() ||
             ggml_backend_buffer_get_type(t->buffer) == host_buft);
    };
    size_t bytes = 0;
    for (const auto & l : model.layers) {
        if (!l.ffn_gate_inp || !l.ffn_gate_inp->buffer ||
            ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
            continue;
        }
        auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer));
        if (!device) { continue; }
        auto * host_buft = ggml_backend_dev_host_buffer_type(device);
        if (!canonical_host(l.ffn_up_exps, host_buft) || !canonical_host(l.ffn_gate_exps, host_buft) ||
            !canonical_host(l.ffn_down_exps, host_buft)) { continue; }
        const int64_t ne = l.ffn_up_exps->ne[2];
        if (ne != l.ffn_gate_exps->ne[2] || ne != l.ffn_down_exps->ne[2] || slots > ne) {
            continue;
        }
        llama_moe_cache_layer c = {};
        c.n_slots = slots;
        c.up_src = l.ffn_up_exps;
        c.gate_src = l.ffn_gate_exps;
        c.down_src = l.ffn_down_exps;
        const auto new_context = [&]() {
            ggml_init_params ip = { 8*ggml_tensor_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) { throw std::runtime_error("MoE cache metadata allocation failed"); }
            cache->contexts.emplace_back(ctx);
            return ctx;
        };
        const auto alloc = [&](ggml_context * ctx, ggml_backend_buffer_type_t buft) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) { throw std::runtime_error("MoE cache buffer allocation failed"); }
            cache->buffers.emplace_back(buf);
            ggml_backend_buffer_clear(buf, 0);
            return buf;
        };
        auto * host = new_context();
        c.host_table = ggml_new_tensor_2d(host, GGML_TYPE_I32, 1, ne);
        c.observations = ggml_new_tensor_1d(host, GGML_TYPE_I64, ne + 1);
        alloc(host, ggml_backend_cpu_buffer_type());
        auto * dev = new_context();
        const auto companion = [&](const ggml_tensor * src) {
            auto * t = ggml_new_tensor_3d(dev, src->type, src->ne[0], src->ne[1], slots + 1);
            ggml_format_name(t, "moe_cache.%s", src->name);
            bytes += ggml_nbytes(t);
            return t;
        };
        c.up_c = companion(c.up_src);
        c.gate_c = companion(c.gate_src);
        c.down_c = companion(c.down_src);
        c.dev_table = ggml_new_tensor_2d(dev, GGML_TYPE_I32, 1, ne);
        auto * dev_buffer = alloc(dev, ggml_backend_dev_buffer_type(device));
        // Keep each cache chain on its owning layer's device in a layer split.
        ggml_backend_buffer_set_usage(dev_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<int32_t> table(ne, slots);
        ggml_backend_tensor_set(c.host_table, table.data(), 0, ne*sizeof(int32_t));
        ggml_backend_tensor_set(c.dev_table, table.data(), 0, ne*sizeof(int32_t));
        c.slot_expert.assign(slots, -1);
        cache->layers.push_back(std::move(c));
    }
    if (cache->layers.empty()) {
        LLAMA_LOG_WARN("moe-cache: no canonical host expert layers with device routers; disabled\n");
        return nullptr;
    }
    LLAMA_LOG_INFO("moe-cache: %zu layers, %d slots/layer, %d inserts/step, %.1f MiB\n",
        cache->layers.size(), slots, inserts, bytes/1048576.0);
    return cache;
}

const llama_moe_cache_layer * llama_moe_cache::lookup(const ggml_tensor * up) const {
    for (const auto & c : layers) {
        if (c.up_src == up) { return &c; }
    }
    return nullptr;
}

void llama_moe_cache::step() {
    for (auto & c : layers) {
        auto * table = static_cast<int32_t *>(c.host_table->data);
        auto * observed = static_cast<int64_t *>(c.observations->data);
        if (observed[c.up_src->ne[2]] == c.processed_clock) { continue; }
        std::vector<int32_t> pending;
        for (int32_t id = 0; id < c.up_src->ne[2]; ++id) {
            if (observed[id] > c.processed_clock && table[id] == c.n_slots) { pending.push_back(id); }
        }
        std::sort(pending.begin(), pending.end(), [&](int32_t a, int32_t b) { return observed[a] > observed[b]; });
        // Each expert occurs once in pending; synchronous uploads have no in-flight state.
        int budget = max_inserts;
        for (int32_t id : pending) {
            if (budget-- <= 0) { break; }
            int slot = 0;
            for (int s = 0; s < c.n_slots; ++s) {
                if (c.slot_expert[s] < 0) { slot = s; break; }
                if (observed[c.slot_expert[s]] < observed[c.slot_expert[slot]]) { slot = s; }
            }
            const int victim = c.slot_expert[slot];
            // Do not replace a more recently used expert, including earlier inserts.
            if (victim >= 0 && observed[victim] >= observed[id]) { continue; }
            if (victim >= 0) { table[victim] = c.n_slots; }
            const auto upload = [&](ggml_tensor * dst, const ggml_tensor * src) {
                GGML_ASSERT(dst->nb[2] == src->nb[2]);
                ggml_backend_tensor_set(dst, static_cast<const char *>(src->data) + id*src->nb[2],
                    slot*dst->nb[2], src->nb[2]);
            };
            upload(c.up_c, c.up_src);
            upload(c.gate_c, c.gate_src);
            upload(c.down_c, c.down_src);
            c.slot_expert[slot] = id;
            table[id] = slot; // publish only after all three blocking uploads finish
        }
        ggml_backend_tensor_set(c.dev_table, table, 0, ggml_nbytes(c.host_table));
        c.processed_clock = observed[c.up_src->ne[2]];
    }
}
