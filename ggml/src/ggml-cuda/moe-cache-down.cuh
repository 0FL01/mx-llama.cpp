#pragma once

#include "common.cuh"
#include "ggml-backend-sched-impl.h"

const ggml_backend_moe_cache_down_v1 * ggml_cuda_moe_cache_down_get_v1();
void ggml_cuda_moe_cache_down_free_backend(ggml_backend_cuda_context * ctx);
void ggml_cuda_moe_cache_down_free_buffer(ggml_backend_buffer_t buffer);
void ggml_cuda_moe_cache_down_wait(ggml_backend_cuda_context * ctx, const ggml_tensor * node);
bool ggml_cuda_moe_cache_down_fusion_ok(
        ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, int node_idx, int node_count);
#ifdef USE_CUDA_GRAPH
void ggml_cuda_moe_cache_down_rebind(ggml_cuda_graph * graph);
#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
void ggml_cuda_moe_cache_down_test_captured(ggml_cuda_graph * graph);
void ggml_cuda_moe_cache_down_test_launched(ggml_cuda_graph * graph);
#endif
#endif

#ifdef GGML_CUDA_MOE_CACHE_DOWN_TEST
const ggml_backend_moe_cache_down_test_v1 * ggml_cuda_moe_cache_down_test_get_v1();
#endif
