#pragma once

// Private scheduler controls shared by ggml and in-tree consumers.

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Force a host drain before overwriting non-graph inputs. Used by contexts
// whose tiny ubatches can outrun cross-device event ordering.
GGML_API void ggml_backend_sched_set_sync_non_graph_inputs(ggml_backend_sched_t sched, bool enabled);

// Read-only canonical expert residency.
// The owner keeps tensors and the table alive and stable until graph completion.
struct ggml_backend_sched_moe_cache_source {
    ggml_backend_t backend;
    const struct ggml_tensor * tensor;
    const int32_t * table;
    int32_t n_slots;
};

typedef bool (*ggml_backend_sched_moe_cache_lookup)(
        const struct ggml_tensor * weight,
        struct ggml_backend_sched_moe_cache_source * source,
        void * user_data);

GGML_API void ggml_backend_sched_set_moe_cache_lookup(
        ggml_backend_sched_t sched, ggml_backend_sched_moe_cache_lookup lookup, void * user_data);

#ifdef __cplusplus
}
#endif
