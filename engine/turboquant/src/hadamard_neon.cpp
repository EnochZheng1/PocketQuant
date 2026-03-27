/**
 * ARM NEON optimized Fast Walsh-Hadamard Transform.
 *
 * CRITICAL CONSTRAINT: The FWHT butterfly at stage h operates on pairs
 * (data[j], data[j+h]). When h < 4, a naive vld1q_f32 would load
 * OVERLAPPING memory regions, corrupting the transform. Each stage
 * requires a different NEON strategy:
 *
 *   h=1: Adjacent pairs (0,1),(2,3),(4,5)... Use vtrn to deinterleave
 *   h=2: Pairs separated by 2: (0,2),(1,3),(4,6)... Use vzip on halves
 *   h>=4: Pairs separated by >=4: safe for direct vld1q_f32/vst1q_f32
 *
 * Performance target: ~100ns per d=128 vector on Cortex-X4 @ 3.3GHz.
 */

#include "hadamard_neon.h"
#include "hadamard.h"
#include "turboquant.h"
#include <math.h>

#if defined(TQ_HAVE_NEON)
#include <arm_neon.h>

void tq_fwht_neon_impl(float *data, int d) {
    for (int h = 1; h < d; h <<= 1) {
        if (h == 1) {
            // Stage h=1: butterfly on adjacent pairs (0,1), (2,3), (4,5), ...
            // Load 4 consecutive floats [a, b, c, d] and produce:
            //   [a+b, a-b, c+d, c-d]
            // Use vtrn to separate even/odd elements within each pair.
            for (int i = 0; i < d; i += 4) {
                float32x4_t v = vld1q_f32(&data[i]);
                // Deinterleave into even/odd: evens=[a,c], odds=[b,d]
                float32x2_t lo = vget_low_f32(v);   // [a, b]
                float32x2_t hi = vget_high_f32(v);   // [c, d]
                float32x2x2_t t0 = vtrn_f32(lo, hi); // .val[0]=[a,c] .val[1]=[b,d]
                float32x2_t sums = vadd_f32(t0.val[0], t0.val[1]); // [a+b, c+d]
                float32x2_t difs = vsub_f32(t0.val[0], t0.val[1]); // [a-b, c-d]
                // Re-interleave: [a+b, a-b, c+d, c-d]
                float32x2x2_t t1 = vtrn_f32(sums, difs);
                vst1q_f32(&data[i], vcombine_f32(t1.val[0], t1.val[1]));
            }
        } else if (h == 2) {
            // Stage h=2: butterfly on pairs separated by 2: (0,2),(1,3), (4,6),(5,7)
            // Load 4 floats [a, b, c, d] and produce:
            //   [a+c, b+d, a-c, b-d]
            // The lower half (a,b) pairs with the upper half (c,d).
            for (int i = 0; i < d; i += 4) {
                float32x4_t v = vld1q_f32(&data[i]);
                float32x2_t lo = vget_low_f32(v);    // [a, b]
                float32x2_t hi = vget_high_f32(v);    // [c, d]
                float32x2_t sums = vadd_f32(lo, hi);  // [a+c, b+d]
                float32x2_t difs = vsub_f32(lo, hi);  // [a-c, b-d]
                vst1q_f32(&data[i], vcombine_f32(sums, difs));
            }
        } else {
            // Stage h>=4: pairs separated by h >= 4 elements.
            // Safe to load non-overlapping 4-float vectors directly.
            for (int i = 0; i < d; i += h << 1) {
                for (int j = i; j < i + h; j += 4) {
                    float32x4_t a = vld1q_f32(&data[j]);
                    float32x4_t b = vld1q_f32(&data[j + h]);
                    vst1q_f32(&data[j],     vaddq_f32(a, b));
                    vst1q_f32(&data[j + h], vsubq_f32(a, b));
                }
            }
        }
    }

    // Normalize by 1/sqrt(d) — makes FWHT self-inverse and norm-preserving
    float s = 1.0f / sqrtf((float)d);
    float32x4_t vscale = vdupq_n_f32(s);
    for (int i = 0; i < d; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        vst1q_f32(&data[i], vmulq_f32(v, vscale));
    }
}

void tq_sign_flip_neon_impl(float *data, const float *signs, int d) {
    for (int i = 0; i < d; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        float32x4_t s = vld1q_f32(&signs[i]);
        vst1q_f32(&data[i], vmulq_f32(v, s));
    }
}

// Override public API to use NEON
void tq_fwht_neon(float *data, int d) {
    tq_fwht_neon_impl(data, d);
}

void tq_sign_flip_neon(float *data, const float *signs, int d) {
    tq_sign_flip_neon_impl(data, signs, d);
}

// ---------------------------------------------------------------------------
// NEON 1-bit quantization — fully vectorized sign extraction
// ---------------------------------------------------------------------------

void tq_quantize_1bit_neon(const float *input, int d, uint32_t *out_bits, float *out_scale) {
    // Compute L2 norm using NEON FMA
    float32x4_t sum_sq = vdupq_n_f32(0.0f);
    for (int i = 0; i < d; i += 4) {
        float32x4_t v = vld1q_f32(&input[i]);
        sum_sq = vfmaq_f32(sum_sq, v, v);
    }
    float norm = sqrtf(vaddvq_f32(sum_sq));
    *out_scale = norm / sqrtf((float)d);

    // Pack sign bits using NEON comparison + vectorized bit collection.
    // Process 32 floats (one uint32 word) at a time.
    // For each group of 4 floats, vcgeq produces a mask where each lane
    // is 0xFFFFFFFF (>=0) or 0x00000000 (<0). We shift the MSB of each
    // lane into position using vshrn to narrow, then collect all 32 bits.
    float32x4_t zero = vdupq_n_f32(0.0f);
    int n_words = d / 32;

    for (int w = 0; w < n_words; w++) {
        uint32_t word = 0;
        const float *base = &input[w * 32];

        for (int g = 0; g < 8; g++) {
            // Compare 4 floats against zero
            uint32x4_t mask = vcgeq_f32(vld1q_f32(&base[g * 4]), zero);
            // Narrow: take bit 31 of each 32-bit lane, pack into 16-bit lanes
            uint16x4_t narrow = vshrn_n_u32(mask, 16);
            // Extract the 4 sign bits from the narrowed result
            // Each lane is 0xFFFF or 0x0000 — we need bit 0 of each
            alignas(8) uint16_t lanes[4];
            vst1_u16(lanes, narrow);
            word |= ((lanes[0] & 1) << (g * 4 + 0));
            word |= ((lanes[1] & 1) << (g * 4 + 1));
            word |= ((lanes[2] & 1) << (g * 4 + 2));
            word |= ((lanes[3] & 1) << (g * 4 + 3));
        }
        out_bits[w] = word;
    }
}

// ---------------------------------------------------------------------------
// NEON 1-bit dequantization — vectorized reconstruction
// ---------------------------------------------------------------------------

void tq_dequantize_1bit_neon(const uint32_t *bits, int d, float scale, float *output) {
    float32x4_t pos_val = vdupq_n_f32(scale);
    float32x4_t neg_val = vdupq_n_f32(-scale);

    int n_words = d / 32;
    for (int w = 0; w < n_words; w++) {
        uint32_t word = bits[w];
        for (int g = 0; g < 8; g++) {
            // Build 4 floats from 4 bits using aligned temp array.
            // Direct NEON vector initialization {a,b,c,d} is a Clang extension
            // that fails on strict NDK/GCC — use vld1q_f32 from aligned memory.
            alignas(16) float temp[4] = {
                (word >> (g * 4 + 0)) & 1 ? scale : -scale,
                (word >> (g * 4 + 1)) & 1 ? scale : -scale,
                (word >> (g * 4 + 2)) & 1 ? scale : -scale,
                (word >> (g * 4 + 3)) & 1 ? scale : -scale
            };
            float32x4_t v = vld1q_f32(temp);
            vst1q_f32(&output[w * 32 + g * 4], v);
        }
    }
}

#else // !TQ_HAVE_NEON

// Fallback to scalar on non-ARM platforms
void tq_fwht_neon_impl(float *data, int d) {
    tq_fwht_scalar(data, d);
}

void tq_sign_flip_neon_impl(float *data, const float *signs, int d) {
    tq_sign_flip_scalar(data, signs, d);
}

void tq_fwht_neon(float *data, int d) {
    tq_fwht_scalar(data, d);
}

#endif // TQ_HAVE_NEON
