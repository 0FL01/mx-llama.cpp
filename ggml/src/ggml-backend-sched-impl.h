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

// Private dispatch scope: a null graph checks eligibility. An accepted graph
// consumes every host source before returning; no descriptors escape.
typedef bool (*ggml_backend_moe_prefill_stream_t)(
        ggml_backend_t backend, struct ggml_tensor * op, const struct ggml_tensor * source,
        const struct ggml_backend_sched_moe_cache_source * cache, const uint32_t * selected,
        struct ggml_cgraph * graph, enum ggml_status * status);

#ifdef __cplusplus
}
#endif
