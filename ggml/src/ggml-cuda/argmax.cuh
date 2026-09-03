#include "common.cuh"

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_tp_top1_stats(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_tp_top1_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
