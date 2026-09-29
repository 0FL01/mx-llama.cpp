#include "moe-cache-down.cuh"
#include "ggml-backend-impl.h"

#include <limits>

#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
#include <chrono>
#include <condition_variable>
#endif

struct ggml_backend_moe_cache_down {
    ggml_backend_cuda_context * owner = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    void * data = nullptr;
    size_t writable_size = 0;
    size_t total_size = 0;
    cudaEvent_t ready[2] = { nullptr, nullptr };
    uint64_t epoch = 0;
    bool open = false;
#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
    ggml_backend_moe_cache_down_test_stats test_stats = {};
    std::mutex test_mutex;
    std::condition_variable test_cv;
    std::chrono::steady_clock::time_point test_deadline;
    bool test_released = true;
    bool test_pending = false;
    bool test_timed_out = false;
    bool test_detaching = false;
#endif
};

// Handles belong to the caller. These lists only track live event owners.
static std::mutex ggml_cuda_moe_down_mutex;
static std::vector<ggml_backend_moe_cache_down *> ggml_cuda_moe_down_live;

static void ggml_cuda_moe_down_sync_readers(ggml_backend_cuda_context * ctx) {
    GGML_ASSERT(!ctx->token_capture_active);
#ifdef USE_CUDA_GRAPH
    GGML_ASSERT(ctx->moe_down_capture == nullptr);
#endif
    for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
        for (int stream = 0; stream < GGML_CUDA_MAX_STREAMS; ++stream) {
            if (ctx->streams[device][stream] != nullptr) {
                ggml_cuda_set_device(device);
                CUDA_CHECK(cudaStreamSynchronize(ctx->streams[device][stream]));
            }
        }
    }
    ggml_cuda_set_device(ctx->device);
}

static void ggml_cuda_moe_down_invalidate(ggml_backend_cuda_context * ctx) {
#ifdef USE_CUDA_GRAPH
    // Registration changes alter waits and fusion, including graphs with no old bindings.
    ctx->cuda_graphs.clear();
#else
    GGML_UNUSED(ctx);
#endif
}

static ggml_backend_buffer_t ggml_cuda_moe_down_buffer(const ggml_tensor * tensor) {
    while (tensor->buffer == nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor->buffer;
}

static bool ggml_cuda_moe_down_matches(const ggml_backend_moe_cache_down * down, const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr || ggml_cuda_moe_down_buffer(tensor) != down->buffer) {
        return false;
    }
    const uintptr_t data = (uintptr_t) tensor->data;
    const uintptr_t base = (uintptr_t) down->data;
    // Copied scheduler descriptors match by buffer/data; views also cover overlapping slices.
    return data >= base ? data - base < down->total_size : base - data < ggml_nbytes(tensor);
}

static bool ggml_cuda_moe_down_uses(const ggml_backend_moe_cache_down * down, const ggml_tensor * node) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (ggml_cuda_moe_down_matches(down, node->src[i])) {
            return true;
        }
    }
    return false;
}

static ggml_backend_moe_cache_down * ggml_cuda_moe_down_create(
        ggml_backend_t backend, const ggml_tensor * dst, size_t slot_size, int32_t n_slots) {
#ifdef GGML_CUDA_MOE_CACHE_DOWN
    if (!ggml_backend_is_cuda(backend) || dst == nullptr || dst->data == nullptr ||
            slot_size == 0 || n_slots <= 0 ||
            slot_size > std::numeric_limits<size_t>::max() / ((size_t) n_slots + 1)) {
        GGML_LOG_WARN("%s: invalid DOWN registration, using synchronous uploads\n", __func__);
        return nullptr;
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buffer = ggml_cuda_moe_down_buffer(dst);
    const size_t total_size = slot_size * ((size_t) n_slots + 1);
    if (ctx->copy_only || ctx->token_capture_active || ctx->token_graph_count != 0 ||
            buffer == nullptr || buffer->buft != ggml_backend_cuda_buffer_type(ctx->device) ||
            !ggml_is_contiguous(dst) || dst->ne[2] != (int64_t) n_slots + 1 || dst->ne[3] != 1 ||
            dst->nb[2] != slot_size || ggml_nbytes(dst) != total_size) {
        GGML_LOG_WARN("%s: unsupported DOWN destination or live raw token graph, using synchronous uploads\n", __func__);
        return nullptr;
    }
    const uintptr_t data = (uintptr_t) dst->data;
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(buffer);
    if (data < base || data - base > buffer->size || total_size > buffer->size - (data - base) ||
            total_size > std::numeric_limits<uintptr_t>::max() - data) {
        GGML_LOG_WARN("%s: DOWN destination exceeds its buffer, using synchronous uploads\n", __func__);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(ggml_cuda_moe_down_mutex);
    for (const auto * other : ggml_cuda_moe_down_live) {
        const uintptr_t other_data = (uintptr_t) other->data;
        if (other->buffer == buffer && data < other_data + other->total_size && other_data < data + total_size) {
            GGML_LOG_WARN("%s: overlapping DOWN registration, using synchronous uploads\n", __func__);
            return nullptr;
        }
    }
    ggml_cuda_moe_down_sync_readers(ctx);
    ggml_cuda_moe_down_invalidate(ctx);
    if (ctx->moe_down_stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->moe_down_stream, cudaStreamNonBlocking));
    }
    auto down = std::make_unique<ggml_backend_moe_cache_down>();
    down->owner = ctx;
    down->buffer = buffer;
    down->data = dst->data;
    down->writable_size = slot_size * (size_t) n_slots;
    down->total_size = total_size;
    for (auto & event : down->ready) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(event, ctx->moe_down_stream));
    }
#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
    down->test_stats.event_records[0] = 1;
    down->test_stats.event_records[1] = 1;
#endif
    ctx->moe_down.push_back(down.get());
    ggml_cuda_moe_down_live.push_back(down.get());
    return down.release();
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(dst);
    GGML_UNUSED(slot_size);
    GGML_UNUSED(n_slots);
    GGML_LOG_WARN("%s: graph event-wait APIs unavailable, using synchronous uploads\n", __func__);
    return nullptr;
#endif
}

static void ggml_cuda_moe_down_enqueue(ggml_backend_moe_cache_down * down, size_t offset, const void * source, size_t size) {
    GGML_ASSERT(down != nullptr && down->owner != nullptr);
    GGML_ASSERT(offset < down->writable_size && size <= down->writable_size - offset);
    if (size == 0) {
        return;
    }
    GGML_ASSERT(source != nullptr);
    if (!down->open) {
        ggml_cuda_moe_down_sync_readers(down->owner);
        down->open = true;
    }
    ggml_cuda_set_device(down->owner->device);
    // All slot writes, including replacement and earlier pending epochs, share this stream.
    CUDA_CHECK(cudaMemcpyAsync((char *) down->data + offset, source, size, cudaMemcpyHostToDevice,
                              down->owner->moe_down_stream));
}

static void ggml_cuda_moe_down_seal(ggml_backend_moe_cache_down * down) {
    GGML_ASSERT(down != nullptr && down->owner != nullptr);
    if (!down->open) {
        ggml_cuda_moe_down_sync_readers(down->owner);
    }
    ggml_cuda_set_device(down->owner->device);
    ++down->epoch;
    CUDA_CHECK(cudaEventRecord(down->ready[down->epoch & 1], down->owner->moe_down_stream));
#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
    ++down->test_stats.event_records[down->epoch & 1];
#endif
    down->open = false;
}

static void ggml_cuda_moe_down_drain(ggml_backend_moe_cache_down * down) {
    if (down == nullptr || down->owner == nullptr) {
        return;
    }
    if (down->open) {
        ggml_cuda_moe_down_seal(down);
    }
    ggml_cuda_set_device(down->owner->device);
    CUDA_CHECK(cudaStreamSynchronize(down->owner->moe_down_stream));
}

static void ggml_cuda_moe_down_detach(ggml_backend_moe_cache_down * down) {
    ggml_backend_cuda_context * ctx = down->owner;
#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
    {
        std::lock_guard<std::mutex> lock(down->test_mutex);
        down->test_detaching = true;
        down->test_cv.notify_all();
    }
#endif
    ggml_cuda_moe_down_sync_readers(ctx);
    ggml_cuda_moe_down_drain(down);
    // No executable may retain an event after its registration is detached.
    ggml_cuda_moe_down_invalidate(ctx);
    for (auto & event : down->ready) {
        CUDA_CHECK(cudaEventDestroy(event));
        event = nullptr;
    }
    ctx->moe_down.erase(std::remove(ctx->moe_down.begin(), ctx->moe_down.end(), down), ctx->moe_down.end());
    ggml_cuda_moe_down_live.erase(std::remove(ggml_cuda_moe_down_live.begin(), ggml_cuda_moe_down_live.end(), down),
                                ggml_cuda_moe_down_live.end());
    down->owner = nullptr;
    down->buffer = nullptr;
    down->data = nullptr;
}

static void ggml_cuda_moe_down_destroy(ggml_backend_moe_cache_down * down) {
    if (down == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(ggml_cuda_moe_down_mutex);
    if (down->owner != nullptr) {
        ggml_cuda_moe_down_detach(down);
    }
    delete down;
}

static bool ggml_cuda_moe_down_query(ggml_backend_moe_cache_down * down) {
    if (down == nullptr || down->owner == nullptr) {
        return true;
    }
    GGML_ASSERT(!down->open);
    ggml_cuda_set_device(down->owner->device);
    const cudaError_t status = cudaEventQuery(down->ready[down->epoch & 1]);
    if (status == cudaErrorNotReady) {
        return false;
    }
    CUDA_CHECK(status);
    return true;
}

const ggml_backend_moe_cache_down_v1 * ggml_cuda_moe_cache_down_get_v1() {
    static const ggml_backend_moe_cache_down_v1 api = {
        1, sizeof(ggml_backend_moe_cache_down_v1),
        ggml_cuda_moe_down_create, ggml_cuda_moe_down_enqueue, ggml_cuda_moe_down_seal,
        ggml_cuda_moe_down_drain, ggml_cuda_moe_down_destroy, ggml_cuda_moe_down_query,
    };
    return &api;
}

#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
#if defined(GGML_CUDA_MOE_CACHE_DOWN) && defined(USE_CUDA_GRAPH)
static void ggml_cuda_moe_down_test_hold(ggml_backend_moe_cache_down * down, uint32_t timeout_ms) {
    GGML_ASSERT(down != nullptr && down->owner != nullptr && !down->open);
    GGML_ASSERT(timeout_ms > 0 && timeout_ms <= 10000);
    ggml_cuda_moe_down_sync_readers(down->owner);
    {
        std::lock_guard<std::mutex> lock(down->test_mutex);
        GGML_ASSERT(!down->test_pending && !down->test_detaching);
        down->test_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        down->test_released = false;
        down->test_pending = true;
        down->test_timed_out = false;
    }
    down->open = true;
    const auto wait_fn = [](void * user_data) {
        auto * state = static_cast<ggml_backend_moe_cache_down *>(user_data);
        std::unique_lock<std::mutex> lock(state->test_mutex);
        if (!state->test_cv.wait_until(lock, state->test_deadline, [state] { return state->test_released; })) {
            state->test_timed_out = true;
            state->test_released = true;
        }
        state->test_pending = false;
        state->test_cv.notify_all();
    };
    CUDA_CHECK(cudaLaunchHostFunc(down->owner->moe_down_stream, wait_fn, down));
}

static void ggml_cuda_moe_down_test_release(ggml_backend_moe_cache_down * down) {
    GGML_ASSERT(down != nullptr);
    std::lock_guard<std::mutex> lock(down->test_mutex);
    if (down->test_pending && !down->test_released && std::chrono::steady_clock::now() >= down->test_deadline) {
        down->test_timed_out = true;
    }
    down->test_released = true;
    down->test_cv.notify_all();
}

static bool ggml_cuda_moe_down_test_release_on_detach(ggml_backend_moe_cache_down * down) {
    GGML_ASSERT(down != nullptr);
    std::unique_lock<std::mutex> lock(down->test_mutex);
    const bool entered = down->test_cv.wait_until(lock, down->test_deadline, [down] { return down->test_detaching; });
    if (!entered || std::chrono::steady_clock::now() >= down->test_deadline) {
        down->test_timed_out = true;
    }
    const bool ok = entered && !down->test_timed_out;
    down->test_released = true;
    down->test_cv.notify_all();
    return ok;
}

static void ggml_cuda_moe_down_test_snapshot(
        ggml_backend_moe_cache_down * down, ggml_backend_moe_cache_down_test_stats * stats) {
    GGML_ASSERT(down != nullptr && stats != nullptr);
    std::lock_guard<std::mutex> lock(down->test_mutex);
    *stats = down->test_stats;
    stats->epoch = down->epoch;
    stats->registered = down->owner != nullptr;
    stats->delay_pending = down->test_pending;
    stats->delay_timed_out = down->test_timed_out;
    stats->teardown_entered = down->test_detaching;
}
#endif

const ggml_backend_moe_cache_down_test_v1 * ggml_cuda_moe_cache_down_test_get_v1() {
#if defined(GGML_CUDA_MOE_CACHE_DOWN) && defined(USE_CUDA_GRAPH)
    static const ggml_backend_moe_cache_down_test_v1 api = {
        1, sizeof(ggml_backend_moe_cache_down_test_v1),
        ggml_cuda_moe_down_test_hold, ggml_cuda_moe_down_test_release,
        ggml_cuda_moe_down_test_release_on_detach, ggml_cuda_moe_down_test_snapshot,
    };
    return &api;
#else
    return nullptr;
#endif
}
#endif

void ggml_cuda_moe_cache_down_free_backend(ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(ggml_cuda_moe_down_mutex);
    while (!ctx->moe_down.empty()) {
        ggml_cuda_moe_down_detach(ctx->moe_down.back());
    }
    if (ctx->moe_down_stream != nullptr) {
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaStreamSynchronize(ctx->moe_down_stream));
        CUDA_CHECK(cudaStreamDestroy(ctx->moe_down_stream));
        ctx->moe_down_stream = nullptr;
    }
}

void ggml_cuda_moe_cache_down_free_buffer(ggml_backend_buffer_t buffer) {
    std::lock_guard<std::mutex> lock(ggml_cuda_moe_down_mutex);
    for (size_t i = 0; i < ggml_cuda_moe_down_live.size();) {
        auto * down = ggml_cuda_moe_down_live[i];
        if (down->buffer == buffer) {
            ggml_cuda_moe_down_detach(down);
        } else {
            ++i;
        }
    }
}

void ggml_cuda_moe_cache_down_wait(ggml_backend_cuda_context * ctx, const ggml_tensor * node) {
#ifdef GGML_CUDA_MOE_CACHE_DOWN
    for (auto * down : ctx->moe_down) {
        if (!ggml_cuda_moe_down_uses(down, node)) {
            continue;
        }
        GGML_ASSERT(!down->open);
        const cudaEvent_t event = down->ready[down->epoch & 1];
        cudaStream_t stream = ctx->stream();
        cudaStreamCaptureStatus status;
        unsigned long long capture_id = 0;
        cudaGraph_t capture_graph = nullptr;
        const cudaGraphNode_t * frontier = nullptr;
        size_t n_frontier = 0;
        CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream, &status, &capture_id, &capture_graph, &frontier, &n_frontier));
        if (status == cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
            ++down->test_stats.direct_waits;
#endif
            continue;
        }
        GGML_ASSERT(status == cudaStreamCaptureStatusActive);
#ifdef USE_CUDA_GRAPH
        GGML_ASSERT(ctx->moe_down_capture != nullptr && capture_graph != nullptr);
        cudaGraphNode_t wait_node = nullptr;
        CUDA_CHECK(cudaGraphAddEventWaitNode(&wait_node, capture_graph, frontier, n_frontier, event));
        CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream, &wait_node, 1, cudaStreamSetCaptureDependencies));
        ctx->moe_down_capture->moe_down_waits.push_back({ wait_node, down });
#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
        ++down->test_stats.bindings;
#endif
#else
        GGML_ABORT("DOWN readiness cannot bind an untracked graph capture");
#endif
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(node);
#endif
}

bool ggml_cuda_moe_cache_down_fusion_ok(
        ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, int node_idx, int node_count) {
    if (ctx->moe_down.empty()) {
        return true;
    }
    if (node_idx < 0 || node_count <= 0 || node_idx > cgraph->n_nodes - node_count) {
        return false;
    }
    for (auto * down : ctx->moe_down) {
        if (ggml_cuda_moe_down_uses(down, cgraph->nodes[node_idx])) {
            continue; // the leading consumer already waits for this registration
        }
        for (int i = node_idx + 1; i < node_idx + node_count; ++i) {
            if (!ggml_cuda_is_view_or_noop(cgraph->nodes[i]) && ggml_cuda_moe_down_uses(down, cgraph->nodes[i])) {
                return false;
            }
        }
    }
    return true;
}

#ifdef USE_CUDA_GRAPH
void ggml_cuda_moe_cache_down_rebind(ggml_cuda_graph * graph) {
#ifdef GGML_CUDA_MOE_CACHE_DOWN
    for (const auto & wait : graph->moe_down_waits) {
        GGML_ASSERT(wait.down->owner != nullptr && !wait.down->open);
        const cudaEvent_t event = wait.down->ready[wait.down->epoch & 1];
        CUDA_CHECK(cudaGraphEventWaitNodeSetEvent(wait.node, event));
        CUDA_CHECK(cudaGraphExecEventWaitNodeSetEvent(graph->instance, wait.node, event));
#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
        ++wait.down->test_stats.rebinds;
        ++wait.down->test_stats.event_rebinds[wait.down->epoch & 1];
#endif
    }
#else
    GGML_ASSERT(graph->moe_down_waits.empty());
#endif
}

#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
static void ggml_cuda_moe_down_test_graph_count(ggml_cuda_graph * graph, bool captured) {
    std::vector<ggml_backend_moe_cache_down *> seen;
    for (const auto & wait : graph->moe_down_waits) {
        if (std::find(seen.begin(), seen.end(), wait.down) != seen.end()) {
            continue;
        }
        seen.push_back(wait.down);
        if (captured) {
            ++wait.down->test_stats.captures;
        } else {
            ++wait.down->test_stats.launches;
        }
    }
}

void ggml_cuda_moe_cache_down_test_captured(ggml_cuda_graph * graph) {
    ggml_cuda_moe_down_test_graph_count(graph, true);
}

void ggml_cuda_moe_cache_down_test_launched(ggml_cuda_graph * graph) {
    ggml_cuda_moe_down_test_graph_count(graph, false);
}
#endif
#endif
