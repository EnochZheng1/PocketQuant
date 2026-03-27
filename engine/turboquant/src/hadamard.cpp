/**
 * Scalar reference implementation of the Fast Walsh-Hadamard Transform.
 *
 * The FWHT applies the normalized Hadamard matrix H_d / sqrt(d) in O(d log d).
 * Key properties:
 *   - Self-inverse: FWHT(FWHT(x)) = x
 *   - Norm-preserving: ||FWHT(x)|| = ||x||
 *   - Spreads information across all dimensions (prerequisite for 1-bit quantization)
 *
 * The randomized version multiplies by a sign vector s (s_i in {-1,+1}) before
 * and after the transform, ensuring the rotation is uniformly random.
 */

#include "hadamard.h"
#include "turboquant.h"
#include <math.h>

// ---------------------------------------------------------------------------
// Scalar FWHT — in-place butterfly network
// ---------------------------------------------------------------------------

void tq_fwht_scalar(float *data, int d) {
    // Butterfly stages: log2(d) stages, each with d/2 butterfly operations
    for (int h = 1; h < d; h <<= 1) {
        for (int i = 0; i < d; i += h << 1) {
            for (int j = i; j < i + h; j++) {
                float x = data[j];
                float y = data[j + h];
                data[j]     = x + y;
                data[j + h] = x - y;
            }
        }
    }
    // Normalize by 1/sqrt(d) to make it self-inverse and norm-preserving
    float scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) {
        data[i] *= scale;
    }
}

// ---------------------------------------------------------------------------
// Sign vector generation (deterministic from seed)
// ---------------------------------------------------------------------------

void tq_generate_signs_impl(float *signs, int d, uint64_t seed) {
    // Simple xoshiro256** PRNG for deterministic sign generation
    uint64_t s0 = seed ^ 0x9E3779B97F4A7C15ULL;
    uint64_t s1 = s0 * 0xBF58476D1CE4E5B9ULL;
    uint64_t s2 = s1 ^ (s1 >> 31);
    uint64_t s3 = s2 * 0x94D049BB133111EBULL;

    for (int i = 0; i < d; i++) {
        // Mix state
        s0 += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s0;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        // Use lowest bit to determine sign
        signs[i] = (z & 1) ? 1.0f : -1.0f;
    }
}

// ---------------------------------------------------------------------------
// Element-wise sign flip
// ---------------------------------------------------------------------------

void tq_sign_flip_scalar(float *data, const float *signs, int d) {
    for (int i = 0; i < d; i++) {
        data[i] *= signs[i];
    }
}

// ---------------------------------------------------------------------------
// Public API (delegates to scalar or NEON)
// ---------------------------------------------------------------------------

void tq_fwht(float *data, int d) {
    tq_fwht_scalar(data, d);
}

void tq_sign_flip(float *data, const float *signs, int d) {
    tq_sign_flip_scalar(data, signs, d);
}

void tq_generate_signs(float *signs, int d, uint64_t seed) {
    tq_generate_signs_impl(signs, d, seed);
}
