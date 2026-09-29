#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-sched-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using sched_ptr = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
using copy_fn = bool (*)(ggml_backend_t, ggml_backend_t, const ggml_tensor *, ggml_tensor *);

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

struct copy_state {
    const std::vector<uint8_t> * weights;
    ggml_tensor * cache;
    ggml_backend_t owner;
    std::vector<int32_t> table;
    std::set<int32_t> used;
    size_t expert_size;
    size_t copies = 0;
    size_t bytes = 0;
    bool verified = false;
};

static copy_fn original_copy;
static copy_state * active_copy;

// Count real async cache-source copies, not lookups or ordinary input transfers.
static bool tracked_copy(ggml_backend_t src_backend, ggml_backend_t dst_backend,
                         const ggml_tensor * src, ggml_tensor * dst) {
    const bool copied = original_copy(src_backend, dst_backend, src, dst);
    if (copied && active_copy && src->buffer == active_copy->cache->buffer) {
        require(ggml_nbytes(dst) == active_copy->expert_size, "D2D range has wrong size");
        ++active_copy->copies;
        active_copy->bytes += ggml_nbytes(dst);
    }
    return copied;
}

static bool lookup_cache(const ggml_tensor *, ggml_backend_sched_moe_cache_source * source, void * user_data) {
    const auto & state = *static_cast<const copy_state *>(user_data);
    *source = { state.owner, state.cache, state.table.data(), 3 };
    return true;
}

static bool check_staged_bytes(ggml_tensor * tensor, bool ask, void * user_data) {
    if (tensor->op != GGML_OP_MUL_MAT_ID) { return false; }
    if (ask) { return true; }
    auto & state = *static_cast<copy_state *>(user_data);
    const auto * staged = tensor->src[0];
    require(staged->buffer && !ggml_backend_buffer_is_host(staged->buffer), "weights did not reach GPU");
    require(ggml_nbytes(staged) == state.weights->size(), "temporary layout changed");
    std::vector<uint8_t> bytes(ggml_nbytes(staged));
    ggml_backend_tensor_get(staged, bytes.data(), 0, bytes.size());
    for (const auto expert : state.used) {
        const size_t offset = expert * state.expert_size;
        require(std::memcmp(bytes.data() + offset, state.weights->data() + offset, state.expert_size) == 0,
                "selected expert bytes differ");
        if (!state.used.count(expert + 1) && expert + 1 < staged->ne[2]) {
            const size_t tail = std::min<size_t>(state.expert_size, 512);
            require(std::memcmp(bytes.data() + offset + state.expert_size,
                                state.weights->data() + offset + state.expert_size, tail) == 0,
                    "MMQ tail is not the canonical neighbour");
        }
    }
    state.verified = true;
    return true;
}

static std::vector<float> run_case(ggml_backend_t cpu, ggml_backend_t gpu, ggml_tensor * weights,
                                  ggml_tensor * cache, const std::vector<uint8_t> & raw,
                                  const std::vector<int32_t> & ids, int tokens, int channels, int mode,
                                  ggml_backend_t peer = nullptr) {
    context_ptr inputs(ggml_init({1024*1024, nullptr, true}), ggml_free);
    context_ptr compute(ggml_init({1024*1024, nullptr, true}), ggml_free);
    require(inputs && compute, "metadata allocation failed");
    auto * x = ggml_new_tensor_3d(inputs.get(), GGML_TYPE_F32, weights->ne[0], channels, tokens);
    auto * selected = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I32, 2, tokens);
    for (int token = 0; token < tokens; ++token) {
        require(ids[2*token] != ids[2*token + 1], "router fixture must have distinct top-k IDs per token");
    }
    ggml_set_input(x);
    ggml_set_input(selected);
    buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), cpu), ggml_backend_buffer_free);
    require(bool(input_buffer), "input allocation failed");
    std::vector<float> values(ggml_nelements(x));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.35f + std::sin(float(i) * 0.017f) + 0.1f * std::cos(float(i) * 0.031f);
    }
    ggml_backend_tensor_set(x, values.data(), 0, values.size()*sizeof(float));
    ggml_backend_tensor_set(selected, ids.data(), 0, ids.size()*sizeof(int32_t));
    auto * out = ggml_mul_mat_id(compute.get(), weights, x, selected);
    ggml_set_output(out);
    auto * graph = ggml_new_graph_custom(compute.get(), 128, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backends[] = {gpu, peer ? peer : cpu, cpu};
    sched_ptr sched(ggml_backend_sched_new(backends, nullptr, peer ? 3 : 2, 128, true, true), ggml_backend_sched_free);
    require(bool(sched), "scheduler allocation failed");
    if (!peer) { ggml_backend_sched_set_tensor_backend(sched.get(), out, gpu); }
    ggml_backend_sched_set_sync_non_graph_inputs(sched.get(), false);
    copy_state state = { &raw, cache, mode == 3 ? cpu : peer ? peer : gpu, std::vector<int32_t>(8, 3),
                         std::set<int32_t>(ids.begin(), ids.end()), weights->nb[2] };
    if (mode != 1) { // cold hook: all entries point to the zero dummy
        state.table[5] = 0;
        state.table[1] = 1;
        state.table[7] = 2;
    }
    if (mode != 0) {
        ggml_backend_sched_set_moe_cache_lookup(sched.get(), lookup_cache, &state);
    }
    ggml_backend_sched_set_eval_callback(sched.get(), check_staged_bytes, &state);
    require(ggml_backend_sched_alloc_graph(sched.get(), graph), "graph allocation failed");
    const char * owner_env = std::getenv("GGML_SCHED_MOE_PREFILL_OWNER");
    const bool prefer_owner = owner_env && std::atoi(owner_env) == 1 && tokens >= 64;
    if (peer) {
        require(ggml_backend_sched_get_tensor_backend(sched.get(), out) == (prefer_owner ? peer : gpu),
                "expert offload did not select the expected backend");
    }
    active_copy = &state;
    require(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "compute failed");
    ggml_backend_sched_synchronize(sched.get());
    active_copy = nullptr;
    require(state.verified, "staging check did not run");
    size_t expected_copies = 0;
    if (mode == 2 && (!peer || prefer_owner)) {
        for (const auto id : state.used) { expected_copies += state.table[id] < 3; }
    }
    require(state.copies == expected_copies && state.bytes == expected_copies * state.expert_size,
            "async D2D hit count differs");
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
    for (const auto value : result) { require(std::isfinite(value), "nonfinite result"); }
    std::printf("type=%s tokens=%d channels=%d mode=%d d2d=%zu bytes=%zu\n",
                ggml_type_name(weights->type), tokens, channels, mode, state.copies, state.bytes);
    return result;
}

int main() {
    ggml_backend_load_all();
    auto * gpu_device = ggml_backend_dev_by_name("ROCm0");
    auto * cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!gpu_device || !cpu_device) { std::fprintf(stderr, "SKIP: ROCm0/CPU unavailable\n"); return 77; }
    backend_ptr gpu(ggml_backend_dev_init(gpu_device, nullptr), ggml_backend_free);
    backend_ptr cpu(ggml_backend_dev_init(cpu_device, nullptr), ggml_backend_free);
    auto * peer_device = ggml_backend_dev_by_name("ROCm1");
    backend_ptr peer(peer_device ? ggml_backend_dev_init(peer_device, nullptr) : nullptr, ggml_backend_free);
    require(gpu && cpu && gpu->iface.cpy_tensor_async, "backend initialization failed");
    original_copy = gpu->iface.cpy_tensor_async;
    gpu->iface.cpy_tensor_async = tracked_copy;
    if (peer) {
        require(peer->iface.cpy_tensor_async == original_copy, "different peer copy implementation");
        peer->iface.cpy_tensor_async = tracked_copy;
    }
    try {
        int cases = 0;
        int bit_exact = 0;
        double max_nmse = 0, max_abs = 0;
        for (const auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1}) {
            context_ptr host(ggml_init({1024*1024, nullptr, true}), ggml_free);
            context_ptr device(ggml_init({1024*1024, nullptr, true}), ggml_free);
            require(host && device, "weight metadata allocation failed");
            auto * weights = ggml_new_tensor_3d(host.get(), type, 256, 32, 8);
            auto * cache = ggml_new_tensor_3d(device.get(), type, 256, 32, 4);
            buffer_ptr host_buffer(ggml_backend_alloc_ctx_tensors(host.get(), cpu.get()), ggml_backend_buffer_free);
            buffer_ptr device_buffer(ggml_backend_alloc_ctx_tensors(device.get(), gpu.get()), ggml_backend_buffer_free);
            require(host_buffer && device_buffer, "weight allocation failed");
            ggml_backend_buffer_set_usage(host_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_set_usage(device_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<float> values(ggml_nelements(weights));
            for (size_t i = 0; i < values.size(); ++i) { values[i] = std::sin(float(i)*0.031f) + float(i%7)*0.03f; }
            std::vector<uint8_t> raw(ggml_nbytes(weights));
            require(ggml_quantize_chunk(type, values.data(), raw.data(), 0, 32*8, 256, nullptr) == raw.size(),
                    "quantization size mismatch");
            ggml_backend_tensor_set(weights, raw.data(), 0, raw.size());
            std::vector<uint8_t> cached(ggml_nbytes(cache), 0);
            const int residents[] = {5, 1, 7}; // slot order deliberately differs from expert order
            for (int slot = 0; slot < 3; ++slot) {
                std::memcpy(cached.data() + slot*cache->nb[2], raw.data() + residents[slot]*weights->nb[2], weights->nb[2]);
            }
            ggml_backend_tensor_set(cache, cached.data(), 0, cached.size());
            for (const int tokens : {1, 4, 64}) {
                for (const int channels : {1, 2}) {
                    for (const auto & pattern : {std::vector<int32_t>{1,5,5,1}, {1,2,5,0}, {5,7,1,7}, {5,1,5,1}}) {
                        std::vector<int32_t> ids(2*tokens);
                        for (size_t i = 0; i < ids.size(); ++i) { ids[i] = pattern[i%pattern.size()]; }
                        const auto baseline = run_case(cpu.get(), gpu.get(), weights, cache, raw, ids, tokens, channels, 0);
                        for (const int mode : {1, 2, 3}) {
                            const auto actual = run_case(cpu.get(), gpu.get(), weights, cache, raw, ids, tokens, channels, mode);
                            double error = 0, norm = 0, max_error = 0;
                            for (size_t i = 0; i < actual.size(); ++i) {
                                error += std::pow(double(actual[i]) - baseline[i], 2);
                                norm += double(baseline[i])*baseline[i];
                                max_error = std::max(max_error, std::abs(double(actual[i]) - baseline[i]));
                            }
                            if (error/std::max(norm, 1e-30) > 5e-4) {
                                const auto repeat = run_case(cpu.get(), gpu.get(), weights, cache, raw, ids, tokens, channels, 0);
                                double repeat_error = 0;
                                for (size_t i = 0; i < repeat.size(); ++i) {
                                    repeat_error += std::pow(double(repeat[i]) - baseline[i], 2);
                                }
                                std::fprintf(stderr, "parity type=%s tokens=%d channels=%d mode=%d ids=%d,%d nmse=%.9g max=%.9g pure_H2D_repeat_nmse=%.9g\n",
                                        ggml_type_name(type), tokens, channels, mode, ids[0], ids[1],
                                        error/std::max(norm, 1e-30), max_error, repeat_error/std::max(norm, 1e-30));
                            }
                            require(error/std::max(norm, 1e-30) <= 5e-4, "output parity failed");
                            max_nmse = std::max(max_nmse, error/std::max(norm, 1e-30));
                            max_abs = std::max(max_abs, max_error);
                            bit_exact += std::memcmp(actual.data(), baseline.data(), actual.size()*sizeof(float)) == 0;
                            ++cases;
                        }
                        std::vector<uint8_t> after(cached.size());
                        ggml_backend_tensor_get(cache, after.data(), 0, after.size());
                        require(after == cached, "cache payload or zero dummy changed");
                    }
                }
            }
            if (type == GGML_TYPE_Q4_0 && peer) {
                context_ptr peer_context(ggml_init({1024*1024, nullptr, true}), ggml_free);
                require(bool(peer_context), "peer metadata allocation failed");
                auto * peer_cache = ggml_dup_tensor(peer_context.get(), cache);
                buffer_ptr peer_buffer(ggml_backend_alloc_ctx_tensors(peer_context.get(), peer.get()), ggml_backend_buffer_free);
                require(bool(peer_buffer), "peer cache allocation failed");
                ggml_backend_buffer_set_usage(peer_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                ggml_backend_tensor_set(peer_cache, cached.data(), 0, cached.size());
                for (int tokens : {42, 64}) {
                    std::vector<int32_t> ids(2*tokens);
                    for (size_t i = 0; i < ids.size(); ++i) { ids[i] = i%2 ? 5 : 1; }
                    const auto baseline = run_case(cpu.get(), gpu.get(), weights, cache, raw, ids, tokens, 1, 0);
                    const auto actual = run_case(cpu.get(), gpu.get(), weights, peer_cache, raw, ids, tokens, 1, 2, peer.get());
                    require(std::memcmp(actual.data(), baseline.data(), actual.size()*sizeof(float)) == 0,
                            "owner placement output changed");
                }
                std::vector<uint8_t> after(cached.size());
                ggml_backend_tensor_get(peer_cache, after.data(), 0, after.size());
                require(after == cached, "peer cache payload or dummy changed");
                std::printf("PASS: two-device owner placement and canonical bytes\n");
            }
        }
        gpu->iface.cpy_tensor_async = original_copy;
        if (peer) { peer->iface.cpy_tensor_async = original_copy; }
        std::printf("PASS: %d canonical-byte/output/cache/async-D2D cases\n", cases);
        std::printf("moe-cache-sched-result: {\"comparisons\":%d,\"bit_exact\":%d,\"max_nmse\":%.9g,\"max_abs\":%.9g}\n",
                    cases, bit_exact, max_nmse, max_abs);
        return 0;
    } catch (const std::exception & error) {
        active_copy = nullptr;
        gpu->iface.cpy_tensor_async = original_copy;
        if (peer) { peer->iface.cpy_tensor_async = original_copy; }
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
