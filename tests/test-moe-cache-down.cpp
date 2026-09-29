#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-sched-impl.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using down_stats = ggml_backend_moe_cache_down_test_stats;
using residents = std::array<int32_t, 2>;
using table_values = std::array<int32_t, 4>;
using weight_bytes = std::array<std::vector<uint8_t>, 3>;

static constexpr int width = 256;
static constexpr int experts = 4;
static constexpr int n_slots = 2;
static constexpr double oracle_tolerance = 1e-4; // Same CPU-chain tolerance as test-moe-cache.

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

struct unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

static std::vector<uint8_t> download(const ggml_tensor * tensor) {
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    return bytes;
}

static table_values make_table(const residents & ids) {
    table_values table = {n_slots, n_slots, n_slots, n_slots};
    require(ids[0] != ids[1], "resident experts must differ");
    for (int slot = 0; slot < n_slots; ++slot) { table[ids[slot]] = slot; }
    return table;
}

static std::vector<int32_t> make_routes(int nt, const residents & ids, int pattern) {
    std::vector<int32_t> misses;
    for (int id = 0; id < experts; ++id) {
        if (id != ids[0] && id != ids[1]) { misses.push_back(id); }
    }
    std::vector<int32_t> routes(2*nt);
    for (int token = 0; token < nt; ++token) {
        const int i = token % 2;
        routes[2*token] = pattern == 2 ? misses[i] : ids[i];
        routes[2*token + 1] = pattern == 0 ? ids[1-i] : misses[pattern == 2 ? 1-i : i];
        require(routes[2*token] != routes[2*token + 1], "real route IDs must differ per token");
    }
    return routes;
}

struct chain {
    ggml_backend_t backend;
    bool cached;
    int nt;
    context_ptr weights{nullptr, ggml_free};
    context_ptr compute{nullptr, ggml_free};
    buffer_ptr weight_buffer{nullptr, ggml_backend_buffer_free};
    buffer_ptr compute_buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * up;
    ggml_tensor * gate;
    ggml_tensor * down;
    ggml_tensor * table;
    ggml_tensor * input;
    ggml_tensor * routes;
    ggml_tensor * output;
    ggml_cgraph * graph;

    chain(ggml_backend_t owner, int tokens, bool cache) : backend(owner), cached(cache), nt(tokens) {
        weights.reset(ggml_init({1024*1024, nullptr, true}));
        compute.reset(ggml_init({1024*1024, nullptr, true}));
        require(weights && compute, "metadata allocation failed");
        const int count = cached ? n_slots + 1 : experts;
        up = ggml_new_tensor_3d(weights.get(), GGML_TYPE_Q4_0, width, width, count);
        gate = ggml_dup_tensor(weights.get(), up);
        down = ggml_new_tensor_3d(weights.get(), GGML_TYPE_Q4_1, width, width, count);
        table = ggml_new_tensor_2d(weights.get(), GGML_TYPE_I32, 1, experts);
        ggml_set_name(down, "registered-down");
        weight_buffer.reset(ggml_backend_alloc_ctx_tensors(weights.get(), backend));
        require(bool(weight_buffer), "weight allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(weight_buffer.get(), 0);

        auto * ctx = compute.get();
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, nt);
        routes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, nt);
        ggml_set_input(input);
        ggml_set_input(routes);
        auto * ids = routes;
        auto * down_src = down;
        if (cached) {
            auto * flat = ggml_reshape_1d(ctx, routes, 2*nt);
            ids = ggml_reshape_2d(ctx, ggml_get_rows(ctx, table, flat), 2, nt);
            // Register the original; consume a descriptor with a different address and name.
            down_src = ggml_dup_tensor(ctx, down);
            down_src->buffer = down->buffer;
            down_src->data = down->data;
            ggml_set_name(down_src, "copied-down");
        }
        auto * u = ggml_mul_mat_id(ctx, up, input, ids);
        auto * g = ggml_mul_mat_id(ctx, gate, input, ids);
        output = ggml_mul_mat_id(ctx, down_src, ggml_swiglu_split(ctx, g, u), ids);
        if (cached) {
            for (auto * node : {u, g, output}) { ggml_mul_mat_id_set_cache_dummy(node, n_slots); }
        }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
                throw unsupported("backend does not support the quantized cache chain");
            }
        }
        compute_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx, backend));
        require(bool(compute_buffer), "compute allocation failed");
        ggml_backend_buffer_clear(compute_buffer.get(), 0);
        if (cached) {
            require(down_src != down && down_src->buffer == down->buffer && down_src->data == down->data,
                    "copied DOWN descriptor identity changed");
        }
    }

    std::array<ggml_tensor *, 3> tensors() const { return {up, gate, down}; }

    void set_inputs(const std::vector<float> & x, const std::vector<int32_t> & ids, const table_values & map) {
        ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(routes, ids.data(), 0, ggml_nbytes(routes));
        ggml_backend_tensor_set(table, map.data(), 0, ggml_nbytes(table));
        const std::vector<float> sentinel(ggml_nelements(output), std::numeric_limits<float>::quiet_NaN());
        ggml_backend_tensor_set(output, sentinel.data(), 0, ggml_nbytes(output));
    }

    void launch() { require(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS, "compute failed"); }

    std::vector<float> result() const {
        std::vector<float> values(ggml_nelements(output));
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
        return values;
    }

    std::vector<float> run() {
        launch();
        ggml_backend_synchronize(backend);
        return result();
    }

    void check_table(const table_values & map) const {
        table_values actual;
        ggml_backend_tensor_get(table, actual.data(), 0, sizeof(actual));
        require(actual == map, "table changed during compute/upload");
    }
};

struct saved_tensor {
    ggml_tensor * tensor;
    ggml_tensor value;

    void check() const {
        require(tensor->data == value.data && tensor->buffer == value.buffer &&
                tensor->type == value.type && tensor->op == value.op && tensor->flags == value.flags &&
                tensor->view_src == value.view_src && tensor->view_offs == value.view_offs &&
                std::memcmp(tensor->ne, value.ne, sizeof(value.ne)) == 0 &&
                std::memcmp(tensor->nb, value.nb, sizeof(value.nb)) == 0 &&
                std::memcmp(tensor->src, value.src, sizeof(value.src)) == 0 &&
                std::memcmp(tensor->op_params, value.op_params, sizeof(value.op_params)) == 0,
                "persistent tensor metadata/address changed");
    }
};

struct saved_chain {
    const chain & owner;
    std::vector<saved_tensor> tensors;
    std::vector<ggml_tensor *> nodes;

    explicit saved_chain(const chain & c) : owner(c) {
        for (auto * ctx : {c.weights.get(), c.compute.get()}) {
            for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
                tensors.push_back({tensor, *tensor});
            }
        }
        for (int i = 0; i < ggml_graph_n_nodes(c.graph); ++i) { nodes.push_back(ggml_graph_node(c.graph, i)); }
    }

    void check() const {
        require(ggml_graph_n_nodes(owner.graph) == int(nodes.size()), "persistent graph shape changed");
        for (size_t i = 0; i < nodes.size(); ++i) {
            require(ggml_graph_node(owner.graph, int(i)) == nodes[i], "persistent graph node changed");
        }
        for (const auto & tensor : tensors) { tensor.check(); }
    }
};

static weight_bytes canonical_weights(chain & cpu) {
    weight_bytes raw;
    const auto tensors = cpu.tensors();
    for (int kind = 0; kind < 3; ++kind) {
        auto * tensor = tensors[kind];
        std::vector<float> values(ggml_nelements(tensor));
        for (size_t i = 0; i < values.size(); ++i) {
            const int expert = int(i / (width*width));
            values[i] = (int((i + 3*kind) % 19) - 9)*0.01f*(1.0f + 0.07f*expert);
        }
        raw[kind].resize(ggml_nbytes(tensor));
        require(ggml_quantize_chunk(tensor->type, values.data(), raw[kind].data(), 0,
                                   width*experts, width, nullptr) == raw[kind].size(), "quantization size mismatch");
        ggml_backend_tensor_set(tensor, raw[kind].data(), 0, raw[kind].size());
    }
    return raw;
}

static weight_bytes cached_weights(const weight_bytes & raw, const chain & c, const residents & ids) {
    weight_bytes bytes;
    const auto tensors = c.tensors();
    for (int kind = 0; kind < 3; ++kind) {
        const size_t size = tensors[kind]->nb[2];
        bytes[kind].assign(ggml_nbytes(tensors[kind]), 0);
        for (int slot = 0; slot < n_slots; ++slot) {
            std::memcpy(bytes[kind].data() + slot*size, raw[kind].data() + ids[slot]*size, size);
        }
    }
    return bytes;
}

static void check_payload(const chain & c, const weight_bytes & bytes) {
    const auto tensors = c.tensors();
    for (int kind = 0; kind < 3; ++kind) {
        require(download(tensors[kind]) == bytes[kind], "canonical payload or zero dummy changed");
    }
}

static double compare(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    require(actual.size() == expected.size(), "result size mismatch");
    double error = 0, norm = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite result");
        error += double(actual[i] - expected[i])*(actual[i] - expected[i]);
        norm += double(expected[i])*expected[i];
    }
    const double nmse = error / (norm + 1e-30);
    if (nmse >= oracle_tolerance) { std::fprintf(stderr, "%s nmse=%.9g\n", label, nmse); }
    require(nmse < oracle_tolerance, label);
    return nmse;
}

static std::vector<float> cached_oracle(
        std::vector<float> full, const std::vector<int32_t> & routes, const table_values & table) {
    // Only cached lanes belong to this backend atom. Miss lanes must use the zero dummy.
    for (size_t lane = 0; lane < routes.size(); ++lane) {
        if (table[routes[lane]] == n_slots) { std::fill_n(full.begin() + lane*width, width, 0.0f); }
    }
    return full;
}

static void check_dummy_result(
        const std::vector<float> & values, const std::vector<int32_t> & routes, const table_values & table) {
    for (size_t lane = 0; lane < routes.size(); ++lane) {
        if (table[routes[lane]] != n_slots) { continue; }
        for (int row = 0; row < width; ++row) { require(values[lane*width + row] == 0.0f, "dummy lane is not zero"); }
    }
}

struct registration {
    const ggml_backend_moe_cache_down_v1 * api;
    const ggml_backend_moe_cache_down_test_v1 * test;
    ggml_backend_moe_cache_down * down;

    ~registration() {
        if (down) {
            test->release(down);
            api->destroy(down);
        }
    }

    void destroy() {
        auto * handle = down;
        down = nullptr;
        api->destroy(handle);
    }

    down_stats stats() const {
        down_stats value;
        test->snapshot(down, &value);
        return value;
    }
};

struct hold_guard {
    registration & reg;
    bool held = true;
    explicit hold_guard(registration & r) : reg(r) { reg.test->hold(reg.down, 10000); }
    ~hold_guard() { if (held) { reg.test->release(reg.down); } }
    void release() { reg.test->release(reg.down); held = false; }
    void dismiss() { held = false; }
};

static void check_replay(const down_stats & before, const down_stats & after) {
    require(after.captures == before.captures && after.bindings == before.bindings,
            "epoch/input change recaptured the graph");
    require(after.direct_waits == before.direct_waits, "replay fell back to direct dispatch");
    require(after.launches == before.launches + 1 && after.rebinds > before.rebinds &&
            after.event_rebinds[after.epoch & 1] > before.event_rebinds[after.epoch & 1],
            "cached executable was not rebound and launched");
    require(!after.delay_timed_out, "side-stream gate watchdog expired");
}

static void check_pending(const registration & reg) {
    const auto stats = reg.stats();
    require(stats.registered && stats.delay_pending && !stats.delay_timed_out, "test gate is not holding the side stream");
    require(!reg.api->query(reg.down), "DOWN epoch is ready before delayed replay");
}

static void run_case(ggml_backend_t cpu, ggml_backend_dev_t device, int nt) {
    const char * name = ggml_backend_dev_name(device);
    std::printf("case device=%s width=%d\n", name, nt);
    const auto backend_reg = ggml_backend_dev_backend_reg(device);
    const auto get_api = (ggml_backend_moe_cache_down_v1_get_t)
        ggml_backend_reg_get_proc_address(backend_reg, "ggml_backend_moe_cache_down_v1");
    const auto get_test = (ggml_backend_moe_cache_down_test_v1_get_t)
        ggml_backend_reg_get_proc_address(backend_reg, "ggml_backend_moe_cache_down_test_v1");
    if (!get_api || !get_test) { throw unsupported("DOWN/test-private registry getters unavailable"); }
    const auto * api = get_api();
    const auto * test = get_test();
    if (!test) { throw unsupported("DOWN external-wait capture support was not compiled"); }
    require(api && api->version == 1 && api->struct_size >= sizeof(*api) &&
            test->version == 1 && test->struct_size >= sizeof(*test), "private DOWN ABI mismatch");
    require(api->create && api->enqueue && api->seal && api->drain && api->destroy && api->query &&
            test->hold && test->release && test->release_on_detach && test->snapshot, "private DOWN ABI is incomplete");
    backend_ptr gpu(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    backend_ptr synchronous(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    require(gpu && synchronous, "GPU backend initialization failed");
    chain reference(cpu, nt, false);
    chain baseline(synchronous.get(), nt, true);
    chain async(gpu.get(), nt, true);
    const auto raw = canonical_weights(reference);
    auto current = cached_weights(raw, async, {0, 1});
    for (auto * c : {&baseline, &async}) {
        const auto tensors = c->tensors();
        for (int kind = 0; kind < 3; ++kind) {
            ggml_backend_tensor_set(tensors[kind], current[kind].data(), 0, current[kind].size());
        }
    }
    // Sources outlive the registration, including exception cleanup and pending teardown.
    buffer_ptr pinned(nullptr, ggml_backend_buffer_free);
    registration reg{api, test, api->create(gpu.get(), async.down, async.down->nb[2], n_slots)};
    require(reg.down != nullptr, "DOWN create rejected a canonical device buffer");
    api->drain(reg.down);
    require(api->query(reg.down) && reg.stats().epoch == 0, "initial epoch is not ready");
    const size_t slot_size = async.down->nb[2];
    const saved_chain ref_saved(reference), base_saved(baseline), async_saved(async);
    table_values table;
    std::vector<int32_t> routes;
    std::vector<float> oracle, expected;
    double max_nmse = 0;

    const auto prepare = [&](residents ids, int pattern, int phase) {
        current = cached_weights(raw, async, ids);
        table = make_table(ids);
        routes = make_routes(nt, ids, pattern);
        std::vector<float> x(width*nt);
        for (size_t i = 0; i < x.size(); ++i) { x[i] = (int((i + 3*phase) % 11) - 5)*0.1f; }
        for (auto * c : {&reference, &baseline, &async}) { c->set_inputs(x, routes, table); }
        for (auto * c : {&baseline, &async}) {
            const auto tensors = c->tensors();
            for (int kind = 0; kind < 2; ++kind) {
                ggml_backend_tensor_set(tensors[kind], current[kind].data(), 0, current[kind].size());
            }
        }
        ggml_backend_tensor_set(baseline.down, current[2].data(), 0, current[2].size());
        oracle = cached_oracle(reference.run(), routes, table);
        expected = baseline.run();
        max_nmse = std::max(max_nmse, compare(expected, oracle, "synchronous GPU/CPU-chain parity failed"));
        check_dummy_result(expected, routes, table);
        check_payload(baseline, current);
    };

    const auto verify = [&] {
        const auto actual = async.result();
        max_nmse = std::max(max_nmse, compare(actual, expected, "async/synchronous SAME GPU parity failed"));
        max_nmse = std::max(max_nmse, compare(actual, oracle, "async GPU/CPU-chain parity failed"));
        check_dummy_result(actual, routes, table);
        check_payload(reference, raw);
        check_payload(baseline, current);
        check_payload(async, current);
        for (const auto * c : {&reference, &baseline, &async}) { c->check_table(table); }
        ref_saved.check();
        base_saved.check();
        async_saved.check();
    };

    prepare({0, 1}, 0, 0);
    for (int warm = 0; warm < 8; ++warm) {
        async.run();
        verify();
        const auto stats = reg.stats();
        if (stats.captures > 0 && stats.bindings > 0 && stats.launches >= 2) { break; }
    }
    const auto captured = reg.stats();
    require(captured.epoch == 0 && captured.captures > 0 && captured.bindings > 0 &&
            captured.rebinds > 0 && captured.launches >= 2 && captured.direct_waits > 0,
            "ready epoch0 did not warm, capture and replay a bound executable");

    // Pageable uploads have no pending assertion; readiness here is an observation only.
    prepare({2, 3}, 0, 1);
    auto before = reg.stats();
    const auto start = std::chrono::steady_clock::now();
    for (int slot = 0; slot < n_slots; ++slot) {
        api->enqueue(reg.down, slot*slot_size, current[2].data() + slot*slot_size, slot_size);
    }
    api->seal(reg.down);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    const bool pageable_pending = !api->query(reg.down);
    std::printf("pageable device=%s width=%d pending=%d enqueue_seal_us=%lld\n", name, nt, int(pageable_pending), (long long) us);
    async.launch();
    check_replay(before, reg.stats());
    api->drain(reg.down);
    ggml_backend_synchronize(gpu.get());
    require(reg.stats().epoch == before.epoch + 1 && api->query(reg.down), "pageable epoch/drain failed");
    verify();

    auto * host_type = ggml_backend_dev_host_buffer_type(device);
    if (!host_type || !ggml_backend_buft_is_host(host_type)) { throw unsupported("device-host buffer type unavailable"); }
    pinned.reset(ggml_backend_buft_alloc_buffer(host_type, 3*slot_size));
    if (!pinned || ggml_backend_buffer_get_type(pinned.get()) != host_type) {
        throw unsupported("device-host allocation fell back to pageable memory");
    }
    auto * source = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned.get()));
    require(source != nullptr, "pinned source has no base address");

    const residents replacements[] = {{1, 3}, {0, 2}, {3, 0}, {2, 1}, {0, 1}, {2, 3}};
    for (int round = 0; round < 6; ++round) {
        prepare(replacements[round], round % 3, round + 2);
        std::memcpy(source, current[2].data(), 2*slot_size);
        const int stale = (replacements[round][0] + 1) % experts;
        std::memcpy(source + 2*slot_size, raw[2].data() + stale*slot_size, slot_size);
        before = reg.stats();
        {
            hold_guard hold(reg);
            api->enqueue(reg.down, 0, source + 2*slot_size, slot_size);
            if (round == 0) {
                api->seal(reg.down); // The next seal also covers this earlier pending replacement.
                check_pending(reg);
            }
            api->enqueue(reg.down, 0, source, slot_size);
            api->enqueue(reg.down, slot_size, source + slot_size, slot_size);
            api->seal(reg.down);
            require(reg.stats().epoch == before.epoch + (round == 0 ? 2 : 1), "admission epoch count differs");
            check_pending(reg); // Last operation before unchanged executable replay.
            async.launch();
            check_replay(before, reg.stats());
            check_pending(reg);
            hold.release();
        }
        api->drain(reg.down);
        ggml_backend_synchronize(gpu.get());
        require(api->query(reg.down) && !reg.stats().delay_pending && !reg.stats().delay_timed_out, "pinned drain failed");
        verify();

        prepare(replacements[round], (round + 1) % 3, round + 20);
        before = reg.stats();
        require(api->query(reg.down), "no-admission replay began with a pending epoch");
        async.launch();
        check_replay(before, reg.stats());
        ggml_backend_synchronize(gpu.get());
        const auto after = reg.stats();
        require(after.epoch == before.epoch && after.event_records[0] == before.event_records[0] &&
                after.event_records[1] == before.event_records[1], "no-admission replay changed readiness");
        verify();
    }
    const auto alternated = reg.stats();
    require(alternated.captures == captured.captures && alternated.bindings == captured.bindings &&
            alternated.event_records[0] > 2 && alternated.event_records[1] > 2 &&
            alternated.event_rebinds[0] > 2 && alternated.event_rebinds[1] > 2,
            "both persistent event handles were not alternated and rebound");

    prepare({3, 1}, 0, 40);
    std::memcpy(source, current[2].data(), 2*slot_size);
    before = reg.stats();
    hold_guard hold(reg);
    for (int slot = 0; slot < n_slots; ++slot) { api->enqueue(reg.down, slot*slot_size, source + slot*slot_size, slot_size); }
    api->seal(reg.down);
    check_pending(reg);
    async.launch();
    const auto final_replay = reg.stats();
    check_replay(before, final_replay);
    check_pending(reg);
    bool release_ok = false;
    auto * pending_handle = reg.down;
    std::thread releaser([test, pending_handle, &release_ok] { release_ok = test->release_on_detach(pending_handle); });
    hold.dismiss();
    // Each width uses a different teardown owner with a graph reader still queued.
    if (nt == 1) {
        reg.destroy();
    } else if (nt == 2) {
        async.weight_buffer.reset();
    } else {
        gpu.reset();
    }
    releaser.join();
    require(release_ok, "teardown did not enter while the side stream was held");
    if (reg.down) {
        const auto detached = reg.stats();
        require(!detached.registered && detached.teardown_entered && !detached.delay_pending &&
                !detached.delay_timed_out && api->query(reg.down), "teardown left a live/pending registration");
    }
    if (nt != 2) { verify(); } // The buffer-free case has no payload left to download.
    else {
        check_payload(reference, raw);
        check_payload(baseline, current);
        reference.check_table(table);
        baseline.check_table(table);
        ref_saved.check();
        base_saved.check();
        async_saved.check();
    }
    std::printf("case passed device=%s width=%d epochs=%llu captures=%llu bindings=%llu rebinds=%llu launches=%llu direct_waits=%llu max_nmse=%.9g teardown=%s\n",
                name, nt, (unsigned long long) final_replay.epoch, (unsigned long long) final_replay.captures,
                (unsigned long long) final_replay.bindings, (unsigned long long) final_replay.rebinds,
                (unsigned long long) final_replay.launches, (unsigned long long) final_replay.direct_waits,
                max_nmse, nt == 1 ? "handle" : nt == 2 ? "buffer" : "backend");
}

int main() {
    try {
        ggml_backend_load_all();
        auto * cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_device) { throw unsupported("CPU backend unavailable"); }
        backend_ptr cpu(ggml_backend_dev_init(cpu_device, nullptr), ggml_backend_free);
        require(bool(cpu), "CPU backend initialization failed");
        int cases = 0;
        for (const char * name : {"ROCm0", "ROCm1"}) {
            auto * device = ggml_backend_dev_by_name(name);
            if (!device) { std::printf("SKIP: %s unavailable\n", name); continue; }
            for (int nt = 1; nt <= 3; ++nt) { run_case(cpu.get(), device, nt); ++cases; }
        }
        if (!cases) { throw unsupported("ROCm0/ROCm1 unavailable"); }
        std::printf("PASS: %d model-free DOWN capture/replay/readiness cases\n", cases);
        return 0;
    } catch (const unsupported & error) {
        std::fprintf(stderr, "SKIP: unsupported capability: %s\n", error.what());
        return 77;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
