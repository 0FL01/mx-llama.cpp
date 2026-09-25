#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Host-only Q4_0 planar layout primitives. Kept independent of CUDA/HIP so the
// byte-preserving serialization can be tested by the regular host test suite.
static inline size_t repack_q4_0_planar_qs_row_stride(const int64_t ne0) {
    assert(ne0 >= 0 && ne0 % 32 == 0);
    size_t rs = (size_t) ne0 / 2;
    if (rs % 128 == 0) {
        rs += 16;
    }
    return rs;
}

static inline size_t repack_q4_0_planar_nbytes(const int64_t ne0, const int64_t ne1) {
    assert(ne1 >= 0);
    return (size_t) ne1 * (repack_q4_0_planar_qs_row_stride(ne0) + (size_t) (ne0 / 32) * 2);
}

static inline void repack_q4_0_planar_host(
        const uint8_t * src, uint8_t * dst, const int64_t ne0, const int64_t ne1,
        const size_t qs_str) {
    assert(src != nullptr && dst != nullptr);
    assert(ne0 >= 0 && ne0 % 32 == 0);
    assert(ne1 >= 0);
    assert(qs_str == repack_q4_0_planar_qs_row_stride(ne0));

    const int64_t n_blocks = ne0 / 32;
    const size_t qs_len = (size_t) ne1 * qs_str;

    std::memset(dst, 0, qs_len + (size_t) ne1 * (size_t) n_blocks * 2);

    for (int64_t row = 0; row < ne1; ++row) {
        for (int64_t blk = 0; blk < n_blocks; ++blk) {
            const uint8_t * src_block = src + (size_t) (row * n_blocks + blk) * 18;
            uint8_t * dst_qs = dst + (size_t) row * qs_str + (size_t) blk * 16;
            uint8_t * dst_d = dst + qs_len + (size_t) (row * n_blocks + blk) * 2;

            // Canonical block_q4_0 is [FP16 d][16 packed qs bytes]. Preserve both
            // bit patterns verbatim; only move them into separate planes.
            std::memcpy(dst_qs, src_block + 2, 16);
            std::memcpy(dst_d, src_block, 2);
        }
    }
}
