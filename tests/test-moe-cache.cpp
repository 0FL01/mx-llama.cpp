#include "../src/llama-moecache.h"

#include <cmath>
#include <cstdio>
#include <vector>

// Optional device argument checks the same split on actual accelerator kernels.
static void test_batch(int nt, ggml_type type, ggml_backend_t device) {
    constexpr int width = 256;
    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    GGML_ASSERT(backend);
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) { set_threads(backend.get(), 2); }
    llama_moe_cache cache;
    cache.max_inserts = 2;
    ggml_init_params ip = { 32*ggml_tensor_overhead(), nullptr, true };
    cache.contexts.emplace_back(ggml_init(ip));
    auto * ctx = cache.contexts.back().get();
    llama_moe_cache_layer c = {};
    c.n_slots = 2;
    c.backend = device ? device : backend.get();
    c.slot_expert.assign(2, -1);
    c.up_src = ggml_new_tensor_3d(ctx, type, width, width, 4);
    c.gate_src = ggml_dup_tensor(ctx, c.up_src);
    c.down_src = ggml_dup_tensor(ctx, c.up_src);
    c.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 4);
    c.observations = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 6);
    cache.buffers.emplace_back(ggml_backend_alloc_ctx_tensors(ctx, backend.get()));
    ggml_backend_buffer_clear(cache.buffers.back().get(), 0);
    cache.contexts.emplace_back(ggml_init(ip));
    auto * dctx = cache.contexts.back().get();
    c.up_c = ggml_new_tensor_3d(dctx, type, width, width, 3);
    c.gate_c = ggml_dup_tensor(dctx, c.up_c);
    c.down_c = ggml_dup_tensor(dctx, c.up_c);
    c.dev_table = ggml_dup_tensor(dctx, c.host_table);
    cache.buffers.emplace_back(ggml_backend_alloc_ctx_tensors(dctx, device ? device : backend.get()));
    ggml_backend_buffer_clear(cache.buffers.back().get(), 0);
    ggml_backend_buffer_set_usage(cache.buffers.back().get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto * w : {c.up_src, c.gate_src, c.down_src}) {
        std::vector<float> data(ggml_nelements(w));
        for (size_t i = 0; i < data.size(); ++i) { data[i] = (int(i % 19) - 9)*0.01f; }
        std::vector<char> packed(ggml_nbytes(w));
        std::vector<float> imatrix(width, 1.0f);
        GGML_ASSERT(ggml_quantize_chunk(type, data.data(), packed.data(), 0, width*4, width, imatrix.data()) == packed.size());
        ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    }
    int32_t initial[] = {2, 2, 2, 2};
    ggml_backend_tensor_set(c.host_table, initial, 0, sizeof(initial));
    ggml_backend_tensor_set(c.dev_table, initial, 0, sizeof(initial));
    cache.layers.push_back(c);

    // Cold misses (duplicate dummy IDs), mixed hits/misses, eviction, all hits.
    for (int round = 0; round < 4; ++round) {
        ggml_init_params gp = { 128*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true };
        ggml_context_ptr graph_ctx(ggml_init(gp));
        auto * gctx = graph_ctx.get();
        ggml_backend_t backends[] = {device ? device : backend.get(), backend.get()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, device ? 2 : 1, 128, false, true));
        auto * input = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, width, 1, nt);
        ggml_set_input(input);
        const int nk = round == 3 ? 2 : 3;
        auto * ranking = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, 4, nt);
        ggml_set_input(ranking);
        if (device) {
            ggml_backend_sched_set_tensor_backend(sched.get(), input, device);
            ggml_backend_sched_set_tensor_backend(sched.get(), ranking, device);
        }
        auto * ids = ggml_view_2d(gctx, ranking, nk, nt, ranking->nb[1], 0);
        auto * flat = ggml_reshape_1d(gctx, ggml_cont(gctx, ids), nk*nt);
        auto * slots = ggml_reshape_2d(gctx, ggml_get_rows(gctx, c.dev_table, flat), nk, nt);
        auto * cpu_input = ggml_dup(gctx, input);
        auto * cpu_ids = ggml_dup(gctx, ids);
        ggml_backend_sched_set_tensor_backend(sched.get(), cpu_input, backend.get());
        ggml_backend_sched_set_tensor_backend(sched.get(), cpu_ids, backend.get());
        const auto chain = [&](ggml_tensor * u, ggml_tensor * g, ggml_tensor * d, ggml_tensor * inp, ggml_tensor * route, bool skip) {
            auto * up = ggml_mul_mat_id(gctx, u, inp, route);
            auto * gate = ggml_mul_mat_id(gctx, g, inp, route);
            auto * down = ggml_mul_mat_id(gctx, d, ggml_swiglu_split(gctx, gate, up), route);
            auto * owner = device && u == c.up_c ? device : backend.get();
            for (auto * op : {up, gate, down}) { ggml_backend_sched_set_tensor_backend(sched.get(), op, owner); }
            if (skip) {
                for (auto * op : {up, gate, down}) {
                    op->src[3] = c.host_table;
                    op->op_params[0] = c.n_slots;
                }
                gate->src[4] = c.observations;
            }
            return down;
        };
        auto * ref = chain(c.up_src, c.gate_src, c.down_src, input, ids, false);
        auto * cpu = chain(c.up_src, c.gate_src, c.down_src, cpu_input, cpu_ids, true);
        auto * hit = chain(c.up_c, c.gate_c, c.down_c, input, slots, false);
        auto * sum = ggml_add(gctx, cpu, hit);
        auto * graph = ggml_new_graph_custom(gctx, 128, false);
        ggml_build_forward_expand(graph, ref);
        ggml_build_forward_expand(graph, cpu_input);
        ggml_build_forward_expand(graph, cpu_ids);
        ggml_build_forward_expand(graph, hit);
        ggml_build_forward_expand(graph, sum);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
        std::vector<float> x(width*nt);
        for (size_t i = 0; i < x.size(); ++i) { x[i] = (int(i % 11) - 5)*0.1f; }
        std::vector<int32_t> route(4*nt);
        for (int t = 0; t < nt; ++t) {
            for (int i = 0; i < 4; ++i) { route[t*4+i] = (i + (round >= 2 ? 2 : 0)) % 4; }
        }
        ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(ranking, route.data(), 0, ggml_nbytes(ranking));
        const auto * observed = static_cast<const int64_t *>(c.observations->data);
        int64_t expected_hits = observed[5];
        const int64_t expected_routes = observed[4] + nk*nt;
        for (int t = 0; t < nt; ++t) {
            for (int i = 0; i < nk; ++i) {
                expected_hits += static_cast<const int32_t *>(c.host_table->data)[route[4*t+i]] < c.n_slots;
            }
        }
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        GGML_ASSERT(observed[4] == expected_routes && observed[5] == expected_hits);
        std::vector<float> a(ggml_nelements(ref)), b(a.size());
        ggml_backend_tensor_get(ref, a.data(), 0, ggml_nbytes(ref));
        ggml_backend_tensor_get(sum, b.data(), 0, ggml_nbytes(sum));
        double error = 0, norm = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            GGML_ASSERT(std::isfinite(a[i]) && std::isfinite(b[i]));
            error += double(a[i] - b[i])*(a[i] - b[i]);
            norm += double(a[i])*a[i];
        }
        GGML_ASSERT(error / (norm + 1e-30) < (device ? 1e-4 : 1e-12));
        cache.step();
        const auto * table = static_cast<const int32_t *>(c.host_table->data);
        GGML_ASSERT(table[round < 2 ? 0 : 2] < c.n_slots);
        GGML_ASSERT(table[round < 2 ? 1 : 3] < c.n_slots);
        GGML_ASSERT(table[round < 2 ? 2 : 0] == c.n_slots);
        const auto old_slots = cache.layers[0].slot_expert;
        cache.step(); // no new observations: no churn
        GGML_ASSERT(old_slots == cache.layers[0].slot_expert);
        for (auto * w : {c.up_c, c.gate_c, c.down_c}) {
            std::vector<char> dummy(w->nb[2]);
            ggml_backend_tensor_get(w, dummy.data(), c.n_slots*w->nb[2], w->nb[2]);
            for (char value : dummy) { GGML_ASSERT(value == 0); }
        }
    }
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_ptr device;
    if (argc == 2) {
        device.reset(ggml_backend_init_by_name(argv[1], nullptr));
        GGML_ASSERT(device);
    }
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_Q8_0}) {
        for (int nt = 1; nt <= 3; ++nt) { test_batch(nt, type, device.get()); }
    }
    puts("MoE cache: batches 1..3, cold/mixed/evicted splits and dummy slots passed");
}
