#pragma once

#include "ggml-cpp.h"
#include <memory>
#include <vector>

struct llama_model;

// Context-owned, bounded cache. All mutations/uploads happen after scheduler
// synchronization. Canonical host weights remain authoritative.
struct llama_moe_cache_layer {
    int32_t n_slots;
    ggml_tensor * up_src;
    ggml_tensor * gate_src;
    ggml_tensor * down_src;
    ggml_tensor * up_c;
    ggml_tensor * gate_c;
    ggml_tensor * down_c;
    ggml_tensor * host_table;
    ggml_tensor * dev_table;
    // CPU gate op writes last-observed timestamps; element n_expert is clock.
    ggml_tensor * observations;
    std::vector<int32_t> slot_expert;
    int64_t processed_clock = 0;
};

struct llama_moe_cache {
    std::vector<ggml_context_ptr> contexts;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::vector<llama_moe_cache_layer> layers;
    int32_t max_inserts;

    static std::unique_ptr<llama_moe_cache> create(const llama_model & model, int32_t slots, int32_t inserts);
    const llama_moe_cache_layer * lookup(const ggml_tensor * up) const;
    void step(); // caller must synchronize graph compute first
};
