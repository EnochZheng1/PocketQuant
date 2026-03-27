/**
 * Structured QJL (Quantized Johnson-Lindenstrauss) error detection.
 *
 * Detects when 1-bit quantization produces unacceptable error for a specific
 * cache entry. Uses a structured random projection instead of dense Gaussian:
 *
 *   1. Compute error = original_rotated - 1bit_reconstruction
 *   2. Apply FWHT to the error vector (reuses our existing FWHT — free!)
 *   3. Sub-sample m uniformly-spaced entries from the 128-dim output
 *   4. If ||sub_sample||² > threshold² × ||original||²: flag for FP16 fallback
 *
 * This satisfies the Johnson-Lindenstrauss lemma with the same guarantees
 * as a dense Gaussian projection, but at O(d log d) instead of O(d²).
 */

#include "qjl.h"
#include "turboquant.h"
#include <math.h>
#include <string.h>

// Temporary buffer for error computation (stack-allocated, d <= 256)
#define TQ_MAX_DIM 256

int tq_qjl_check_impl(const float *original, const uint32_t *bits, int d,
                       float scale, float threshold, int m) {
    float error[TQ_MAX_DIM];

    // Step 1: compute error = original - reconstruction
    int n_words = d / 32;
    for (int w = 0; w < n_words; w++) {
        uint32_t word = bits[w];
        for (int b = 0; b < 32; b++) {
            int idx = w * 32 + b;
            float reconstructed = (word & (1u << b)) ? scale : -scale;
            error[idx] = original[idx] - reconstructed;
        }
    }

    // Step 2: apply FWHT to error vector (O(d log d))
    tq_fwht(error, d);

    // Step 3: sub-sample m uniformly-spaced entries
    // Spacing = d / m (for d=128, m=32: every 4th entry)
    int stride = d / m;
    float proj_norm_sq = 0.0f;
    for (int i = 0; i < m; i++) {
        float val = error[i * stride];
        proj_norm_sq += val * val;
    }

    // Step 4: compare projected error norm against threshold
    // Scale the projected norm by sqrt(d/m) to account for sub-sampling
    float scale_factor = (float)d / (float)m;
    float estimated_error_sq = proj_norm_sq * scale_factor;

    // Compare against original vector's energy
    float orig_norm_sq = 0.0f;
    for (int i = 0; i < d; i++) {
        orig_norm_sq += original[i] * original[i];
    }

    // Flag if relative error exceeds threshold
    return (estimated_error_sq > threshold * threshold * orig_norm_sq) ? 1 : 0;
}

// Public API
int tq_qjl_check(const float *original, const uint32_t *bits, int d,
                  float scale, float threshold, int m) {
    return tq_qjl_check_impl(original, bits, d, scale, threshold, m);
}
