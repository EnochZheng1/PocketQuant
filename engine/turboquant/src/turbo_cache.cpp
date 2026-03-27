/**
 * Compressed KV cache data structure.
 *
 * Stores K and V vectors as 1-bit (sign) quantized values after Hadamard rotation.
 * Each entry consists of:
 *   - d/32 uint32_t packed sign bits
 *   - 1 float scale factor (L2 norm / sqrt(d))
 *   - 1 uint16_t fallback index (0 = no fallback, >0 = slot+1 in sparse fallback array)
 *
 * QJL-flagged entries store original FP32 values in a SPARSE fallback buffer
 * sized to ~10% of max_seq. This avoids the OOM crash from pre-allocating
 * a dense fallback array (which would be 896 MB for Qwen2.5 7B at 8K context).
 */

#include "turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Fallback budget: 10% of max_seq per head. If QJL flags more than this,
// excess entries silently skip fallback (use 1-bit approximation instead).
#define FALLBACK_BUDGET_PCT 10

// Internal: per-layer compressed storage for one cache type (K or V)
struct tq_layer_cache {
    uint32_t *bits;           // [n_heads * max_seq * words_per_head]
    float    *scales;         // [n_heads * max_seq]
    uint16_t *fallback_idx;   // [n_heads * max_seq] — 0=no fallback, >0=slot+1 in fallback[]
    float    *fallback;       // SPARSE: [n_heads * fallback_cap * head_dim]
    size_t    fallback_count; // current number of used fallback slots
    size_t    fallback_cap;   // max fallback slots per layer cache
};

struct tq_cache {
    int n_layers;
    int n_heads;
    int head_dim;
    int max_seq;
    int words_per_head;   // head_dim / 32

    float qjl_threshold;
    int   qjl_m;          // sub-sample dimension (default 32)

    float *signs;         // [head_dim]

    tq_layer_cache *k_cache;  // [n_layers]
    tq_layer_cache *v_cache;  // [n_layers]

    // Statistics (atomic for thread safety)
    size_t total_stores;
    size_t fallback_stores;
};

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

static void init_layer_cache(tq_layer_cache *lc, int n_heads, int max_seq,
                             int head_dim, int words_per_head) {
    size_t n_entries = (size_t)n_heads * max_seq;
    lc->bits         = (uint32_t *)calloc(n_entries * words_per_head, sizeof(uint32_t));
    lc->scales       = (float *)calloc(n_entries, sizeof(float));
    lc->fallback_idx = (uint16_t *)calloc(n_entries, sizeof(uint16_t));

    // Sparse fallback: budget is 10% of total entries
    lc->fallback_cap   = (n_entries * FALLBACK_BUDGET_PCT) / 100;
    if (lc->fallback_cap < 64) lc->fallback_cap = 64;  // minimum 64 slots
    lc->fallback       = (float *)calloc(lc->fallback_cap * head_dim, sizeof(float));
    lc->fallback_count = 0;
}

static void free_layer_cache(tq_layer_cache *lc) {
    free(lc->bits);
    free(lc->scales);
    free(lc->fallback_idx);
    free(lc->fallback);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

tq_cache *tq_cache_create(int n_layers, int n_heads, int head_dim,
                           int max_seq, uint64_t seed, float qjl_threshold) {
    tq_cache *cache = (tq_cache *)calloc(1, sizeof(tq_cache));
    cache->n_layers      = n_layers;
    cache->n_heads       = n_heads;
    cache->head_dim      = head_dim;
    cache->max_seq       = max_seq;
    cache->words_per_head = head_dim / 32;
    cache->qjl_threshold = qjl_threshold;
    cache->qjl_m         = 32;

    cache->signs = (float *)malloc(head_dim * sizeof(float));
    tq_generate_signs(cache->signs, head_dim, seed);

    cache->k_cache = (tq_layer_cache *)calloc(n_layers, sizeof(tq_layer_cache));
    cache->v_cache = (tq_layer_cache *)calloc(n_layers, sizeof(tq_layer_cache));

    for (int l = 0; l < n_layers; l++) {
        init_layer_cache(&cache->k_cache[l], n_heads, max_seq, head_dim, cache->words_per_head);
        init_layer_cache(&cache->v_cache[l], n_heads, max_seq, head_dim, cache->words_per_head);
    }

    return cache;
}

void tq_cache_free(tq_cache *cache) {
    if (!cache) return;
    for (int l = 0; l < cache->n_layers; l++) {
        free_layer_cache(&cache->k_cache[l]);
        free_layer_cache(&cache->v_cache[l]);
    }
    free(cache->k_cache);
    free(cache->v_cache);
    free(cache->signs);
    free(cache);
}

void tq_cache_store(tq_cache *cache, int layer, int head, int pos,
                    const float *vec, int is_value) {
    int d = cache->head_dim;
    int wph = cache->words_per_head;
    tq_layer_cache *lc = is_value ? &cache->v_cache[layer] : &cache->k_cache[layer];
    size_t entry_idx = (size_t)head * cache->max_seq + pos;

    // Stack-allocated, 16-byte aligned for NEON. Zero heap overhead.
    alignas(16) float rotated[256];
    memcpy(rotated, vec, d * sizeof(float));

    // Step 1: sign-flip
    tq_sign_flip(rotated, cache->signs, d);

    // Step 2: FWHT rotation
#if defined(TQ_HAVE_NEON)
    tq_fwht_neon(rotated, d);
#else
    tq_fwht(rotated, d);
#endif

    // Step 3: 1-bit quantize
    alignas(16) uint32_t bits[8];
    float scale;
#if defined(TQ_HAVE_NEON)
    tq_quantize_1bit_neon(rotated, d, bits, &scale);
#else
    tq_quantize_1bit(rotated, d, bits, &scale);
#endif

    // Step 4: QJL error check
    int needs_fallback = tq_qjl_check(rotated, bits, d, scale,
                                       cache->qjl_threshold, cache->qjl_m);

    // Step 5: store compressed data
    memcpy(&lc->bits[entry_idx * wph], bits, wph * sizeof(uint32_t));
    lc->scales[entry_idx] = scale;

    if (needs_fallback) {
        // Atomically grab a fallback slot from the sparse pool
        size_t fb_slot = __atomic_fetch_add(&lc->fallback_count, 1, __ATOMIC_RELAXED);
        if (fb_slot < lc->fallback_cap) {
            lc->fallback_idx[entry_idx] = (uint16_t)(fb_slot + 1);  // +1 so 0 means "no fallback"
            memcpy(&lc->fallback[fb_slot * d], vec, d * sizeof(float));
        } else {
            // Budget exhausted — fall back to 1-bit approximation (no FP32 backup)
            lc->fallback_idx[entry_idx] = 0;
        }
        __atomic_fetch_add(&cache->fallback_stores, 1, __ATOMIC_RELAXED);
    } else {
        lc->fallback_idx[entry_idx] = 0;
    }

    __atomic_fetch_add(&cache->total_stores, 1, __ATOMIC_RELAXED);
}

void tq_cache_load(const tq_cache *cache, int layer, int head, int pos,
                   float *out_vec, int is_value) {
    int d = cache->head_dim;
    int wph = cache->words_per_head;
    const tq_layer_cache *lc = is_value ? &cache->v_cache[layer] : &cache->k_cache[layer];
    size_t entry_idx = (size_t)head * cache->max_seq + pos;

    // Check if this entry has an FP32 fallback
    uint16_t fb_idx = lc->fallback_idx[entry_idx];
    if (fb_idx > 0) {
        size_t fb_slot = (size_t)(fb_idx - 1);
        memcpy(out_vec, &lc->fallback[fb_slot * d], d * sizeof(float));
        return;
    }

    // Dequantize from 1-bit
    float scale = lc->scales[entry_idx];
#if defined(TQ_HAVE_NEON)
    tq_dequantize_1bit_neon(&lc->bits[entry_idx * wph], d, scale, out_vec);
#else
    tq_dequantize_1bit(&lc->bits[entry_idx * wph], d, scale, out_vec);
#endif

    // Inverse FWHT (self-inverse)
#if defined(TQ_HAVE_NEON)
    tq_fwht_neon(out_vec, d);
#else
    tq_fwht(out_vec, d);
#endif

    // Inverse sign-flip
    tq_sign_flip(out_vec, cache->signs, d);
}

void tq_cache_get_stats(const tq_cache *cache, tq_cache_stats *stats) {
    stats->total_entries = cache->total_stores;
    stats->fallback_entries = cache->fallback_stores;
    stats->fallback_rate = cache->total_stores > 0
        ? (float)cache->fallback_stores / (float)cache->total_stores
        : 0.0f;

    // Calculate actual memory usage (compressed + sparse fallback)
    size_t compressed_total = 0;
    size_t fallback_total = 0;
    for (int l = 0; l < cache->n_layers; l++) {
        size_t n_entries = (size_t)cache->n_heads * cache->max_seq;
        compressed_total += n_entries * cache->words_per_head * sizeof(uint32_t);  // bits
        compressed_total += n_entries * sizeof(float);    // scales
        compressed_total += n_entries * sizeof(uint16_t); // fallback_idx
        fallback_total += cache->k_cache[l].fallback_count * cache->head_dim * sizeof(float);
        fallback_total += cache->v_cache[l].fallback_count * cache->head_dim * sizeof(float);
    }

    stats->compressed_bytes = (compressed_total * 2) + fallback_total;  // *2 for K+V
    stats->uncompressed_bytes = (size_t)cache->n_layers * 2 *
        (size_t)cache->n_heads * cache->max_seq * cache->head_dim * sizeof(float);

    stats->compression_ratio = stats->compressed_bytes > 0
        ? (float)stats->uncompressed_bytes / (float)stats->compressed_bytes
        : 0.0f;
}

void tq_cache_clear(tq_cache *cache) {
    for (int l = 0; l < cache->n_layers; l++) {
        size_t n_entries = (size_t)cache->n_heads * cache->max_seq;
        memset(cache->k_cache[l].fallback_idx, 0, n_entries * sizeof(uint16_t));
        memset(cache->v_cache[l].fallback_idx, 0, n_entries * sizeof(uint16_t));
        cache->k_cache[l].fallback_count = 0;
        cache->v_cache[l].fallback_count = 0;
    }
    cache->total_stores = 0;
    cache->fallback_stores = 0;
}

/**
 * Shift cache entries, preserving [0, system_pos).
 * Also compacts the sparse fallback array.
 */
static void shift_layer_cache(tq_layer_cache *lc, int n_heads, int max_seq,
                               int head_dim, int words_per_head,
                               int n_discard, int system_pos) {
    for (int h = 0; h < n_heads; h++) {
        size_t base = (size_t)h * max_seq;
        size_t src  = base + system_pos + n_discard;
        size_t dst  = base + system_pos;
        int n_move  = max_seq - system_pos - n_discard;
        if (n_move <= 0) continue;

        // Shift bits
        memmove(&lc->bits[dst * words_per_head],
                &lc->bits[src * words_per_head],
                (size_t)n_move * words_per_head * sizeof(uint32_t));

        // Shift scales
        memmove(&lc->scales[dst],
                &lc->scales[src],
                (size_t)n_move * sizeof(float));

        // Shift fallback_idx
        memmove(&lc->fallback_idx[dst],
                &lc->fallback_idx[src],
                (size_t)n_move * sizeof(uint16_t));

        // Zero vacated tail
        size_t tail = base + max_seq - n_discard;
        memset(&lc->fallback_idx[tail], 0, (size_t)n_discard * sizeof(uint16_t));
    }

    // Rebuild sparse fallback array: compact by scanning all valid entries
    // This is O(n_entries) but only runs during context shifts (rare)
    size_t n_entries = (size_t)n_heads * max_seq;
    size_t new_count = 0;
    float *new_fb = (float *)malloc(lc->fallback_cap * head_dim * sizeof(float));

    for (size_t i = 0; i < n_entries; i++) {
        if (lc->fallback_idx[i] > 0) {
            size_t old_slot = (size_t)(lc->fallback_idx[i] - 1);
            if (old_slot < lc->fallback_cap && new_count < lc->fallback_cap) {
                memcpy(&new_fb[new_count * head_dim],
                       &lc->fallback[old_slot * head_dim],
                       head_dim * sizeof(float));
                lc->fallback_idx[i] = (uint16_t)(new_count + 1);
                new_count++;
            } else {
                lc->fallback_idx[i] = 0;  // lost due to capacity
            }
        }
    }

    memcpy(lc->fallback, new_fb, new_count * head_dim * sizeof(float));
    free(new_fb);
    lc->fallback_count = new_count;
}

void tq_cache_shift(tq_cache *cache, int n_discard, int system_pos) {
    if (n_discard <= 0) return;

    for (int l = 0; l < cache->n_layers; l++) {
        shift_layer_cache(&cache->k_cache[l], cache->n_heads, cache->max_seq,
                          cache->head_dim, cache->words_per_head, n_discard, system_pos);
        shift_layer_cache(&cache->v_cache[l], cache->n_heads, cache->max_seq,
                          cache->head_dim, cache->words_per_head, n_discard, system_pos);
    }
}
