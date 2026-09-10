#pragma once

#include "ggml-cpp.h"
#include <memory>
#include <vector>

struct llama_model;

// Context-owned, bounded cache. All mutations/uploads happen after scheduler
// synchronization. Canonical host weights remain authoritative.
struct llama_moe_cache_layer {
    int32_t n_slots;
    ggml_backend_t backend; // borrowed from the owning context
    ggml_tensor * up_src;
    ggml_tensor * gate_src;
    ggml_tensor * down_src;
    ggml_tensor * up_c;
    ggml_tensor * gate_c;
    ggml_tensor * down_c;
    ggml_tensor * host_table;
    ggml_tensor * dev_table;
    // CPU gate op writes timestamps; final two elements count routes and hits.
    ggml_tensor * observations;
    std::vector<int32_t> slot_expert;
    int64_t processed_clock = 0;
    // Per-call presence, not token counts: observations collapse repeated IDs.
    std::vector<uint32_t> frequency;
    std::vector<int64_t> admitted_at;
    std::vector<bool> used;
    int64_t calls = 0;
    int64_t admissions = 0;
    int64_t evictions = 0;
    int64_t unused_evictions = 0;
    int64_t first_hits = 0;
    int64_t first_hit_calls = 0;
};

struct llama_moe_cache {
    std::vector<ggml_context_ptr> contexts;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::vector<llama_moe_cache_layer> layers;
    int32_t max_inserts;
    int64_t steps = 0;
    int64_t update_us = 0;
    size_t upload_bytes = 0;
    bool frequency_gated = false; // opt-in diagnostic policy; ranked LRU is default
    bool report_each_step = false;

    static std::unique_ptr<llama_moe_cache> create(const llama_model & model, int32_t slots, int32_t inserts,
                                                const std::vector<ggml_backend_ptr> & backends);
    const llama_moe_cache_layer * lookup(const ggml_tensor * up) const;
    void step(); // caller must synchronize graph compute first
};
