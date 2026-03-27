/**
 * ARM NEON optimized Fast Walsh-Hadamard Transform.
 *
 * Processes 4 floats per NEON lane using vaddq_f32/vsubq_f32 for butterfly ops.
 * For d=128: 7 butterfly stages, each processing 64 pairs.
 * Each stage operates on non-overlapping pairs, so NEON processes 4 pairs at once.
 *
 * Expected performance: ~100-200ns per vector on Cortex-X4 @ 3.3GHz.
 */

#include "hadamard_neon.h"
#include "hadamard.h"
#include "turboquant.h"
#include <math.h>

#if defined(TQ_HAVE_NEON)
#include <arm_neon.h>

void tq_fwht_neon_impl(float *data, int d) {
    // NEON butterfly FWHT
    // Each stage h: for pairs (i, i+h), compute (a+b, a-b)
    // NEON processes 4 consecutive pairs at a time when h >= 4
    for (int h = 1; h < d; h <<= 1) {
        if (h >= 4) {
            // NEON path: process 4 butterflies at a time
            for (int i = 0; i < d; i += h << 1) {
                for (int j = i; j < i + h; j += 4) {
                    float32x4_t a = vld1q_f32(&data[j]);
                    float32x4_t b = vld1q_f32(&data[j + h]);
                    vst1q_f32(&data[j],     vaddq_f32(a, b));
                    vst1q_f32(&data[j + h], vsubq_f32(a, b));
                }
            }
        } else {
            // Scalar path for h=1, h=2 (stride too small for NEON)
            for (int i = 0; i < d; i += h << 1) {
                for (int j = i; j < i + h; j++) {
                    float x = data[j];
                    float y = data[j + h];
                    data[j]     = x + y;
                    data[j + h] = x - y;
                }
            }
        }
    }

    // Normalize by 1/sqrt(d)
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

// NEON 1-bit quantization
void tq_quantize_1bit_neon(const float *input, int d, uint32_t *out_bits, float *out_scale) {
    // Compute L2 norm using NEON
    float32x4_t sum_sq = vdupq_n_f32(0.0f);
    for (int i = 0; i < d; i += 4) {
        float32x4_t v = vld1q_f32(&input[i]);
        sum_sq = vfmaq_f32(sum_sq, v, v);  // fused multiply-add: sum_sq += v*v
    }
    // Horizontal sum of 4 lanes
    float norm_sq = vaddvq_f32(sum_sq);
    float norm = sqrtf(norm_sq);
    *out_scale = norm / sqrtf((float)d);

    // Pack sign bits using NEON comparison
    float32x4_t zero = vdupq_n_f32(0.0f);
    int n_words = d / 32;
    for (int w = 0; w < n_words; w++) {
        uint32_t word = 0;
        // Process 32 floats in groups of 4
        for (int g = 0; g < 8; g++) {
            float32x4_t v = vld1q_f32(&input[w * 32 + g * 4]);
            // vcgeq: compare >= 0, result is 0xFFFFFFFF or 0x00000000
            uint32x4_t mask = vcgeq_f32(v, zero);
            // Extract each lane's sign bit
            word |= ((vgetq_lane_u32(mask, 0) & 1) << (g * 4 + 0));
            word |= ((vgetq_lane_u32(mask, 1) & 1) << (g * 4 + 1));
            word |= ((vgetq_lane_u32(mask, 2) & 1) << (g * 4 + 2));
            word |= ((vgetq_lane_u32(mask, 3) & 1) << (g * 4 + 3));
        }
        out_bits[w] = word;
    }
}

void tq_dequantize_1bit_neon(const uint32_t *bits, int d, float scale, float *output) {
    float32x4_t pos = vdupq_n_f32(scale);
    float32x4_t neg = vdupq_n_f32(-scale);

    int n_words = d / 32;
    for (int w = 0; w < n_words; w++) {
        uint32_t word = bits[w];
        for (int g = 0; g < 8; g++) {
            // Extract 4 bits
            float vals[4];
            for (int b = 0; b < 4; b++) {
                vals[b] = (word & (1u << (g * 4 + b))) ? scale : -scale;
            }
            vst1q_f32(&output[w * 32 + g * 4], vld1q_f32(vals));
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
