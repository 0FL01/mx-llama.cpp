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

// Private registry getter: "ggml_backend_moe_cache_down_v1". No vendor types cross this interface.
struct ggml_backend_moe_cache_down;

struct ggml_backend_moe_cache_down_v1 {
    uint32_t version;
    size_t struct_size;

    // dst has n_slots canonical slices followed by one read-only dummy slice.
    // NULL means use synchronous uploads. The caller owns the returned handle.
    struct ggml_backend_moe_cache_down * (*create)(
            ggml_backend_t backend, const struct ggml_tensor * dst, size_t slot_size, int32_t n_slots);

    // Serialize these calls with backend dispatch. The first write drains previous readers.
    // Source bytes stay alive and unchanged until drain; seal precedes any consumer dispatch.
    void (*enqueue)(struct ggml_backend_moe_cache_down * down, size_t offset, const void * source, size_t size);
    void (*seal)(struct ggml_backend_moe_cache_down * down);
    void (*drain)(struct ggml_backend_moe_cache_down * down);

    // Buffer/backend teardown drains and detaches the handle; destroy still releases it.
    void (*destroy)(struct ggml_backend_moe_cache_down * down);
    bool (*query)(struct ggml_backend_moe_cache_down * down); // latest sealed epoch is ready
};

typedef const struct ggml_backend_moe_cache_down_v1 * (*ggml_backend_moe_cache_down_v1_get_t)(void);

// TEST-private registry getter: "ggml_backend_moe_cache_down_test_v1". Test builds only.
struct ggml_backend_moe_cache_down_test_stats {
    uint64_t epoch;
    uint64_t captures;
    uint64_t bindings;
    uint64_t rebinds;
    uint64_t launches;
    uint64_t direct_waits;
    uint64_t event_records[2];
    uint64_t event_rebinds[2];
    bool registered;
    bool delay_pending;
    bool delay_timed_out;
    bool teardown_entered;
};

struct ggml_backend_moe_cache_down_test_v1 {
    uint32_t version;
    size_t struct_size;

    // Hold the side stream until release, with a 1..10000 ms watchdog. No GPU work in the callback.
    void (*hold)(struct ggml_backend_moe_cache_down * down, uint32_t timeout_ms);
    void (*release)(struct ggml_backend_moe_cache_down * down);
    // Only this call may run concurrently with teardown; join its caller before releasing sources.
    bool (*release_on_detach)(struct ggml_backend_moe_cache_down * down);
    void (*snapshot)(struct ggml_backend_moe_cache_down * down, struct ggml_backend_moe_cache_down_test_stats * stats);
};

typedef const struct ggml_backend_moe_cache_down_test_v1 * (*ggml_backend_moe_cache_down_test_v1_get_t)(void);

#ifdef __cplusplus
}
#endif
