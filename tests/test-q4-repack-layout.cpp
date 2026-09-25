#include "../ggml/src/ggml-cuda/q8_repack/repack-q4-0-host.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(const bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-q4-repack-layout: %s\n", message);
        std::exit(1);
    }
}

static void test_shape(const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const size_t src_size = (size_t) ne1 * (size_t) n_blocks * 18;
    const size_t qs_str = repack_q4_0_planar_qs_row_stride(ne0);
    const size_t qs_size = (size_t) ne1 * qs_str;
    const size_t dst_size = repack_q4_0_planar_nbytes(ne0, ne1);

    check(dst_size == qs_size + (size_t) ne1 * (size_t) n_blocks * 2,
          "planar byte count does not equal packed payload plus FP16 scales");

    std::vector<uint8_t> src(src_size);
    std::vector<uint8_t> dst(dst_size, 0xff);
    for (size_t i = 0; i < src.size(); ++i) {
        // Deterministic mixed payload/scale bytes, including arbitrary FP16 bit
        // patterns: the layout operation must not reinterpret either plane.
        src[i] = (uint8_t) ((i * 73 + i / 7 + 19) & 0xff);
    }

    repack_q4_0_planar_host(src.data(), dst.data(), ne0, ne1, qs_str);

    for (int64_t row = 0; row < ne1; ++row) {
        for (int64_t blk = 0; blk < n_blocks; ++blk) {
            const size_t canonical = (size_t) (row * n_blocks + blk) * 18;
            const size_t planar_qs = (size_t) row * qs_str + (size_t) blk * 16;
            const size_t planar_d = qs_size + (size_t) (row * n_blocks + blk) * 2;
            for (int i = 0; i < 16; ++i) {
                check(dst[planar_qs + (size_t) i] == src[canonical + 2 + (size_t) i],
                      "packed nibble byte changed during repack");
            }
            check(dst[planar_d] == src[canonical] && dst[planar_d + 1] == src[canonical + 1],
                  "FP16 scale bits changed during repack");
        }
        for (size_t pad = (size_t) n_blocks * 16; pad < qs_str; ++pad) {
            check(dst[(size_t) row * qs_str + pad] == 0,
                  "de-alias padding was not zeroed");
        }
    }
}

static void test_nibble_order() {
    uint8_t canonical[18] = {};
    canonical[0] = 0x34;
    canonical[1] = 0x12;
    for (int i = 0; i < 16; ++i) {
        const uint8_t q = (uint8_t) (i & 0x0f);
        canonical[2 + i] = (uint8_t) (q | (q << 4));
    }

    uint8_t planar[18] = {};
    repack_q4_0_planar_host(canonical, planar, 32, 1, 16);
    for (int i = 0; i < 16; ++i) {
        const uint8_t packed = planar[i];
        check((packed & 0x0f) == (i & 0x0f), "low nibble ordering is incorrect");
        check(((packed >> 4) & 0x0f) == (i & 0x0f), "high nibble ordering is incorrect");
    }
    check(planar[16] == canonical[0] && planar[17] == canonical[1],
          "scale plane did not preserve the original FP16 bits");
}

int main() {
    test_shape(32, 3);       // no de-alias padding
    test_shape(256, 2);      // packed row is 128 B and needs the 16 B bump
    test_shape(5120, 3);     // representative Qwen row geometry (2560 + 16 B)
    test_nibble_order();
    return 0;
}
