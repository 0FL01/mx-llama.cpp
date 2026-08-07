#include "debug.h"

#include "common.h"
#include "log.h"

#include <cmath>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

struct common_debug_cb_user_data::impl {
    std::vector<uint8_t>    data;
    std::vector<std::regex> tensor_filters;
    bool                    abort_on_nan{false};
};

common_debug_cb_user_data::common_debug_cb_user_data() : pimpl(std::make_unique<impl>()) {}
common_debug_cb_user_data::~common_debug_cb_user_data() = default;

common_debug_cb_user_data::common_debug_cb_user_data(common_params & params, const std::vector<std::string> & filter_patterns, bool abort_on_nan)
    : pimpl(std::make_unique<impl>())
{
    for (const auto & pattern : filter_patterns) {
        try {
            std::string anchored_pattern = "^" + pattern;
            pimpl->tensor_filters.emplace_back(anchored_pattern, std::regex::optimize);
        } catch (const std::regex_error & e) {
            throw std::runtime_error("Invalid regex pattern '" + pattern + "': " + e.what());
        }
    }
    pimpl->abort_on_nan = abort_on_nan;

    params.cb_eval           = common_debug_cb_eval;
    params.cb_eval_user_data = this;
}

static std::string common_ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static float common_ggml_get_float_value(const uint8_t * data,
                           ggml_type       type,
                           const size_t *  nb,
                           size_t          i0,
                           size_t          i1,
                           size_t          i2,
                           size_t          i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float  v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
}

#define INDENT "    "

static void common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n, bool abort_on_nan) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG(INDENT "[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                LOG(INDENT INDENT "..., \n");
                i2 = ne[2] - n;
            }
            LOG(INDENT INDENT "[\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    LOG(INDENT INDENT INDENT "..., \n");
                    i1 = ne[1] - n;
                }
                LOG(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        LOG("   ..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        LOG(", ");
                    }
                }
                LOG("  ],\n");
            }
            LOG(INDENT INDENT "],\n");
        }
        LOG(INDENT "]\n");
        LOG(INDENT "sum = %f\n", sum);
    }

    if (abort_on_nan) {
        if (std::isnan(sum)) {
            LOG("encountered NaN - aborting\n");
            exit(0);
        }
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (common_debug_cb_user_data *) user_data;
    auto * pimpl = cb_data->pimpl.get();

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        // GGML_DEBUG_SKIP_PASSES=N / GGML_DEBUG_MAX_PASSES=M: only observe graph
        // passes in [N, M]. A pass is one full walk over the graph, detected by
        // re-seeing the first tensor ever asked about (graph reuse keeps node
        // pointers stable across passes). Lets a long-context bisect dump only
        // the generation steps around a corruption boundary.
        static const long skip_passes = []() {
            const char * env = getenv("GGML_DEBUG_SKIP_PASSES");
            return env ? atol(env) : 0L;
        }();
        static const long max_passes = []() {
            const char * env = getenv("GGML_DEBUG_MAX_PASSES");
            return env ? atol(env) : 0L;
        }();
        // GGML_DEBUG_ASK_TRACE=N: log the first N asked node names to find a
        // stable per-pass sentinel name.
        static long ask_trace = []() {
            const char * env = getenv("GGML_DEBUG_ASK_TRACE");
            return env ? atol(env) : 0L;
        }();
        if (ask_trace > 0) {
            ask_trace--;
            LOG("asktrace %s op=%s ne=[%lld,%lld]\n", t->name, ggml_op_desc(t),
                (long long) t->ne[0], (long long) t->ne[1]);
        }
        // Match the sentinel by NAME: a graph rebuild (prefill -> decode shape
        // change) allocates new tensor structs, so pointers do not survive, but
        // node names are builder-assigned and stable. GGML_DEBUG_PASS_SENTINEL
        // overrides the default (first asked name) when that name does not recur
        // in later graphs.
        static const char * env_sentinel = getenv("GGML_DEBUG_PASS_SENTINEL");
        static std::string pass_sentinel;
        static long pass_count = 0;
        if (skip_passes > 0 || max_passes > 0) {
            if (pass_sentinel.empty()) {
                pass_sentinel = env_sentinel ? env_sentinel : t->name;
                pass_count    = 0;
                LOG("common_debug_cb_eval: pass window [%ld, %ld], sentinel %s\n",
                    skip_passes, max_passes > 0 ? max_passes : -1, pass_sentinel.c_str());
            }
            if (pass_sentinel == t->name) {
                pass_count++;
                if (pass_count >= skip_passes && (max_passes <= 0 || pass_count <= max_passes)) {
                    LOG("common_debug_cb_eval: === pass %ld ===\n", pass_count);
                }
            }
            if (pass_count < skip_passes) {
                return false;
            }
            if (max_passes > 0 && pass_count > max_passes) {
                return false;
            }
        }
        // GGML_DEBUG_ONLY_NAME=substr: observe only nodes whose name contains
        // the substring. Turns a full-graph dump into a single-tensor probe.
        static const char * only_name = getenv("GGML_DEBUG_ONLY_NAME");
        if (only_name != nullptr && strstr(t->name, only_name) == nullptr) {
            return false;
        }
        // GGML_DEBUG_SKIP_NE1=N: skip tensors with ne[1] > N entirely (no
        // readback, no print). Lets a debug run dump per-token decode graphs
        // while skipping the prompt-sized prefill tensors whose per-node
        // readback would otherwise dominate the run.
        static const int64_t skip_ne1 = []() {
            const char * env = getenv("GGML_DEBUG_SKIP_NE1");
            return env ? atoll(env) : (int64_t) 0;
        }();
        if (skip_ne1 > 0 && t->ne[1] > skip_ne1) {
            return false;
        }
        // GGML_DEBUG_SKIP_NONCONT=1: skip non-contiguous tensors. The meta
        // backend cannot gather non-contiguous axis-split tensors (asserts),
        // and their contiguous successors carry the same information.
        static const bool skip_noncont = []() {
            const char * env = getenv("GGML_DEBUG_SKIP_NONCONT");
            return env && atoi(env) != 0;
        }();
        if (skip_noncont && !ggml_is_contiguous(t)) {
            return false;
        }
        return true;  // Always retrieve data
    }

    bool matches_filter = pimpl->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : pimpl->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    char src1_str[128] = { 0 };
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, common_ggml_ne_string(src1).c_str());
    }

    if (matches_filter) {
        LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__, t->name, ggml_type_name(t->type),
            ggml_op_desc(t), src0->name, common_ggml_ne_string(src0).c_str(), src1 ? src1_str : "",
            common_ggml_ne_string(t).c_str());
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        pimpl->data.resize(n_bytes);
        ggml_backend_tensor_get(t, pimpl->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type) && matches_filter) {
        uint8_t * data = is_host ? (uint8_t *) t->data : pimpl->data.data();
        // GGML_DEBUG_SUM_ONLY=1: one line per node (sum + finite-count) instead of
        // the value excerpt. Cuts a per-node dump by 10-50x so a bisect can afford
        // to observe many generation passes.
        static const bool sum_only = []() {
            const char * env = getenv("GGML_DEBUG_SUM_ONLY");
            return env && atoi(env) != 0;
        }();
        if (sum_only) {
            double sum = 0.0;
            double asum = 0.0;
            int64_t n_nonfinite = 0;
            for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
                for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                    for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                        for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                            const float v = common_ggml_get_float_value(data, t->type, t->nb, i0, i1, i2, i3);
                            if (!std::isfinite(v)) {
                                n_nonfinite++;
                                continue;
                            }
                            sum  += v;
                            asum += std::fabs(v);
                        }
                    }
                }
            }
            LOG("sumline %s sum=%.6f asum=%.6f nonfinite=%lld\n",
                t->name, sum, asum, (long long) n_nonfinite);
        } else {
            common_debug_print_tensor(data, t->type, t->ne, t->nb, 3, pimpl->abort_on_nan);
        }
    }

    return true;
}
