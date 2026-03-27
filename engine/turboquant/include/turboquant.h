/**
 * TurboQuant — KV cache compression via Hadamard rotation + 1-bit quantization
 *
 * Public API for the PocketQuant edge inference engine.
 * Provides FWHT rotation, 1-bit quantize/dequantize, QJL error detection,
 * and a compressed cache data structure.
 *
 * All functions operate on float32 arrays of size `d` (head dimension, must be power of 2).
 * Primary implementation uses ARM NEON intrinsics; scalar fallback available.
 */

#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Hadamard Rotation (Fast Walsh-Hadamard Transform)
// ---------------------------------------------------------------------------

/**
 * In-place Fast Walsh-Hadamard Transform (scalar reference).
 * Applies H_d / sqrt(d) where H_d is the Hadamard matrix.
 * Self-inverse: tq_fwht(tq_fwht(x)) = x.
 * Norm-preserving: ||tq_fwht(x)|| = ||x||.
 *
 * @param data   Float array of length `d` (modified in-place)
 * @param d      Dimension (must be power of 2, typically 128)
 */
void tq_fwht(float *data, int d);

/**
 * NEON-optimized FWHT for d=128.
 * Falls back to tq_fwht on non-ARM platforms.
 */
void tq_fwht_neon(float *data, int d);

/**
 * Apply randomized sign flips before/after FWHT.
 * signs[i] must be +1.0f or -1.0f.
 */
void tq_sign_flip(float *data, const float *signs, int d);

#if defined(TQ_HAVE_NEON)
void tq_sign_flip_neon(float *data, const float *signs, int d);
#endif

/**
 * Generate a deterministic sign vector from a seed.
 * Output: signs[i] in {-1.0f, +1.0f} for i in [0, d).
 */
void tq_generate_signs(float *signs, int d, uint64_t seed);

// ---------------------------------------------------------------------------
// 1-Bit Quantization
// ---------------------------------------------------------------------------

/**
 * Quantize float vector to 1-bit (sign bit) with scale.
 *
 * @param input      Float array of length `d` (rotated values)
 * @param d          Dimension (must be multiple of 32)
 * @param out_bits   Output packed bits: d/32 uint32_t values
 * @param out_scale  Output FP32 scale (L2 norm / sqrt(d))
 */
void tq_quantize_1bit(const float *input, int d, uint32_t *out_bits, float *out_scale);

/**
 * Dequantize 1-bit packed values back to float.
 *
 * @param bits    Packed bits: d/32 uint32_t values
 * @param d       Dimension
 * @param scale   Scale factor from quantization
 * @param output  Output float array of length `d`
 */
void tq_dequantize_1bit(const uint32_t *bits, int d, float scale, float *output);

#if defined(TQ_HAVE_NEON)
void tq_quantize_1bit_neon(const float *input, int d, uint32_t *out_bits, float *out_scale);
void tq_dequantize_1bit_neon(const uint32_t *bits, int d, float scale, float *output);
#endif

// ---------------------------------------------------------------------------
// QJL Error Detection (Structured Random — sub-sampled Hadamard)
// ---------------------------------------------------------------------------

/**
 * Check if 1-bit quantization error exceeds threshold.
 * Uses structured random projection: FWHT(error) then sub-sample m dims.
 *
 * @param original   Original rotated float vector (length d)
 * @param bits       1-bit quantized packed bits
 * @param d          Dimension
 * @param scale      Scale from quantization
 * @param threshold  Error threshold (typically 0.1 - 0.3)
 * @param m          Number of sub-sampled dimensions (typically 32)
 * @return           1 if error exceeds threshold (use fallback), 0 if OK
 */
int tq_qjl_check(const float *original, const uint32_t *bits, int d,
                  float scale, float threshold, int m);

// ---------------------------------------------------------------------------
// Compressed Cache Data Structure
// ---------------------------------------------------------------------------

typedef struct tq_cache tq_cache;

/**
 * Create a compressed KV cache.
 *
 * @param n_layers    Number of transformer layers
 * @param n_heads     Number of KV heads per layer
 * @param head_dim    Head dimension (must be power of 2)
 * @param max_seq     Maximum sequence length (KV cache capacity)
 * @param seed        Random seed for sign vectors
 * @param qjl_threshold  QJL error threshold
 * @return            Opaque cache handle (must be freed with tq_cache_free)
 */
tq_cache *tq_cache_create(int n_layers, int n_heads, int head_dim,
                           int max_seq, uint64_t seed, float qjl_threshold);

void tq_cache_free(tq_cache *cache);

/**
 * Store a K or V vector into the compressed cache.
 * Applies: sign-flip → FWHT → 1-bit quantize → QJL check → store.
 *
 * @param cache     Cache handle
 * @param layer     Layer index
 * @param head      Head index
 * @param pos       Sequence position
 * @param vec       Float vector of length head_dim (post-RoPE for K)
 * @param is_value  0 for K cache, 1 for V cache
 */
void tq_cache_store(tq_cache *cache, int layer, int head, int pos,
                    const float *vec, int is_value);

/**
 * Retrieve a decompressed K or V vector from the cache.
 * Applies: dequantize → inverse FWHT → inverse sign-flip.
 * For QJL-flagged entries, returns the FP16 fallback (promoted to FP32).
 *
 * @param cache     Cache handle
 * @param layer     Layer index
 * @param head      Head index
 * @param pos       Sequence position
 * @param out_vec   Output float vector of length head_dim
 * @param is_value  0 for K cache, 1 for V cache
 */
void tq_cache_load(const tq_cache *cache, int layer, int head, int pos,
                   float *out_vec, int is_value);

/**
 * Get cache statistics.
 */
typedef struct {
    size_t total_entries;
    size_t fallback_entries;      // Entries using FP16 fallback (QJL-flagged)
    float  fallback_rate;         // fallback_entries / total_entries
    size_t compressed_bytes;      // Total compressed storage used
    size_t uncompressed_bytes;    // What FP16 would have used
    float  compression_ratio;     // uncompressed / compressed
} tq_cache_stats;

void tq_cache_get_stats(const tq_cache *cache, tq_cache_stats *stats);

/** Clear all entries (keep allocated memory). */
void tq_cache_clear(tq_cache *cache);

#ifdef __cplusplus
}
#endif

#endif // TURBOQUANT_H
