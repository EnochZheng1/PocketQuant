/**
 * 1-bit quantization and dequantization.
 *
 * After Hadamard rotation, values are approximately Gaussian with equal variance
 * across all dimensions. We quantize to just the sign bit (1 bit per value),
 * with a per-head FP32 scale factor (L2 norm / sqrt(d)).
 *
 * Storage: d/32 uint32_t packed words + 1 float scale = (d/8 + 4) bytes per head.
 * For d=128: 16 + 4 = 20 bytes vs 256 bytes at FP16 = 12.8x compression.
 */

#include "quantize.h"
#include "turboquant.h"
#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Scalar 1-bit quantization
// ---------------------------------------------------------------------------

void tq_quantize_1bit_scalar(const float *input, int d, uint32_t *out_bits, float *out_scale) {
    // Compute L2 norm for scale factor
    float norm_sq = 0.0f;
    for (int i = 0; i < d; i++) {
        norm_sq += input[i] * input[i];
    }
    float norm = sqrtf(norm_sq);

    // Scale = expected magnitude after reconstruction
    // For a d-dimensional vector quantized to signs, reconstruction has
    // magnitude norm/sqrt(d) per component
    *out_scale = norm / sqrtf((float)d);

    // Pack sign bits: bit=1 if value >= 0, bit=0 if value < 0
    int n_words = d / 32;
    for (int w = 0; w < n_words; w++) {
        uint32_t word = 0;
        for (int b = 0; b < 32; b++) {
            if (input[w * 32 + b] >= 0.0f) {
                word |= (1u << b);
            }
        }
        out_bits[w] = word;
    }
}

// ---------------------------------------------------------------------------
// Scalar 1-bit dequantization
// ---------------------------------------------------------------------------

void tq_dequantize_1bit_scalar(const uint32_t *bits, int d, float scale, float *output) {
    int n_words = d / 32;
    for (int w = 0; w < n_words; w++) {
        uint32_t word = bits[w];
        for (int b = 0; b < 32; b++) {
            // bit=1 → +scale, bit=0 → -scale
            output[w * 32 + b] = (word & (1u << b)) ? scale : -scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void tq_quantize_1bit(const float *input, int d, uint32_t *out_bits, float *out_scale) {
    tq_quantize_1bit_scalar(input, d, out_bits, out_scale);
}

void tq_dequantize_1bit(const uint32_t *bits, int d, float scale, float *output) {
    tq_dequantize_1bit_scalar(bits, d, scale, output);
}
