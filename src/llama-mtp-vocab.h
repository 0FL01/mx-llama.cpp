#pragma once

#include "ggml.h"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using llama_mtp_vocab_range = std::pair<int64_t, int64_t>;

inline std::vector<llama_mtp_vocab_range> llama_mtp_vocab_ranges(const std::string & value, int64_t n_vocab) {
    std::vector<llama_mtp_vocab_range> ranges;
    if (value.empty()) {
        return ranges;
    }
    size_t pos = 0;
    while (pos < value.size()) {
        const size_t comma = value.find(',', pos);
        const size_t end = comma == std::string::npos ? value.size() : comma;
        const size_t colon = value.find(':', pos);
        int64_t first = -1, last = -1;
        if (colon == std::string::npos || colon >= end) {
            throw std::runtime_error("MTP vocabulary ranges must use start:end pairs");
        }
        const auto a = std::from_chars(value.data() + pos, value.data() + colon, first);
        const auto b = std::from_chars(value.data() + colon + 1, value.data() + end, last);
        if (a.ec != std::errc() || a.ptr != value.data() + colon ||
                b.ec != std::errc() || b.ptr != value.data() + end ||
                first < 0 || first >= last || last > n_vocab ||
                (!ranges.empty() && first < ranges.back().second) ||
                (comma != std::string::npos && comma + 1 == value.size())) {
            throw std::runtime_error("MTP vocabulary ranges must be ordered, non-overlapping and within the vocabulary");
        }
        ranges.emplace_back(first, last);
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return ranges;
}

inline ggml_tensor * llama_mtp_vocab_logits(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * input,
                                          const std::vector<llama_mtp_vocab_range> & ranges) {
    GGML_ASSERT(!ranges.empty() && ggml_is_contiguous(weight) && ggml_n_dims(weight) == 2);
    GGML_ASSERT(weight->ne[0] == input->ne[0] && ggml_n_dims(input) <= 2);
    ggml_tensor * full = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, weight->ne[1], input->ne[1]), -INFINITY);
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto & range = ranges[i];
        GGML_ASSERT(range.first >= 0 && range.first < range.second && range.second <= weight->ne[1]);
        ggml_tensor * rows = ggml_view_2d(ctx, weight, weight->ne[0], range.second - range.first,
                                        weight->nb[1], range.first * weight->nb[1]);
        ggml_tensor * projection = ggml_mul_mat(ctx, rows, input);
        ggml_format_name(projection, "mtp_head_projection.%zu", i);
        full = ggml_set_2d_inplace(ctx, full, projection, full->nb[1], range.first * sizeof(float));
    }
    return full;
}
