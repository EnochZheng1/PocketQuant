/**
 * Compressed KV cache data structure.
 *
 * Stores K and V vectors as 1-bit (sign) quantized values after Hadamard rotation.
 * Each entry consists of:
 *   - d/32 uint32_t packed sign bits (e.g., 4 words for d=128)
 *   - 1 float scale factor (L2 norm / sqrt(d))
 *   - 1 uint8 precision flag (0 = 1-bit, 1 = FP16 fallback)
 *
 * QJL-flagged entries also store the original FP16 values in a separate fallback buffer.
 *
 * Memory layout per layer per cache (K or V):
 *   bits:      [n_heads][max_seq][d/32] uint32_t
 *   scales:    [n_heads][max_seq] float
 *   flags:     [n_heads][max_seq] uint8_t
 *   fallback:  dynamically allocated FP16 entries
 */

#include "turboquant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal: per-layer compressed storage for one cache type (K or V)
struct tq_layer_cache {
    uint32_t *bits;       // [n_heads * max_seq * words_per_head]
    float    *scales;     // [n_heads * max_seq]
    uint8_t  *flags;      // [n_heads * max_seq] — 0=1bit, 1=fallback
    float    *fallback;   // [n_heads * max_seq * head_dim] — only flagged entries are populated
};

struct tq_cache {
    int n_layers;
    int n_heads;
    int head_dim;
    int max_seq;
    int words_per_head;   // head_dim / 32

    float qjl_threshold;
    int   qjl_m;          // sub-sample dimension (default 32)

    // Per-layer sign vectors for randomized Hadamard
    float *signs;         // [head_dim]

    // K and V caches per layer
    tq_layer_cache *k_cache;  // [n_layers]
    tq_layer_cache *v_cache;  // [n_layers]

    // Statistics
    size_t total_stores;
    size_t fallback_stores;
};

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

static void init_layer_cache(tq_layer_cache *lc, int n_heads, int max_seq,
                             int head_dim, int words_per_head) {
    size_t n_entries = (size_t)n_heads * max_seq;
    lc->bits     = (uint32_t *)calloc(n_entries * words_per_head, sizeof(uint32_t));
    lc->scales   = (float *)calloc(n_entries, sizeof(float));
    lc->flags    = (uint8_t *)calloc(n_entries, sizeof(uint8_t));
    lc->fallback = (float *)calloc(n_entries * head_dim, sizeof(float));
}

static void free_layer_cache(tq_layer_cache *lc) {
    free(lc->bits);
    free(lc->scales);
    free(lc->flags);
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
    cache->qjl_m         = 32;  // sub-sample 32 dims from FWHT output

    // Generate sign vector
    cache->signs = (float *)malloc(head_dim * sizeof(float));
    tq_generate_signs(cache->signs, head_dim, seed);

    // Allocate per-layer caches
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
    // This runs 1000+ times per token (every head × every layer), so
    // heap allocation here would destroy TPS via fragmentation.
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
    alignas(16) uint32_t bits[8]; // max d=256 → 8 words
    float scale;
#if defined(TQ_HAVE_NEON)
    tq_quantize_1bit_neon(rotated, d, bits, &scale);
#else
    tq_quantize_1bit(rotated, d, bits, &scale);
#endif

    // Step 4: QJL error check
    int needs_fallback = tq_qjl_check(rotated, bits, d, scale,
                                       cache->qjl_threshold, cache->qjl_m);

    // Step 5: store
    memcpy(&lc->bits[entry_idx * wph], bits, wph * sizeof(uint32_t));
    lc->scales[entry_idx] = scale;
    lc->flags[entry_idx] = needs_fallback ? 1 : 0;

    if (needs_fallback) {
        // Store original (pre-rotation) vector as FP32 fallback
        memcpy(&lc->fallback[entry_idx * d], vec, d * sizeof(float));
        cache->fallback_stores++;
    }

    cache->total_stores++;
}

void tq_cache_load(const tq_cache *cache, int layer, int head, int pos,
                   float *out_vec, int is_value) {
    int d = cache->head_dim;
    int wph = cache->words_per_head;
    const tq_layer_cache *lc = is_value ? &cache->v_cache[layer] : &cache->k_cache[layer];
    size_t entry_idx = (size_t)head * cache->max_seq + pos;

    // Check if this entry uses FP16 fallback
    if (lc->flags[entry_idx]) {
        memcpy(out_vec, &lc->fallback[entry_idx * d], d * sizeof(float));
        return;
    }

    // Dequantize from 1-bit
    float scale = lc->scales[entry_idx];
#if defined(TQ_HAVE_NEON)
    tq_dequantize_1bit_neon(&lc->bits[entry_idx * wph], d, scale, out_vec);
#else
    tq_dequantize_1bit(&lc->bits[entry_idx * wph], d, scale, out_vec);
#endif

    // Inverse FWHT (self-inverse, so just apply FWHT again)
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

    // Calculate memory usage
    size_t entries_per_layer = (size_t)cache->n_heads * cache->max_seq;
    size_t compressed_per_layer =
        entries_per_layer * cache->words_per_head * sizeof(uint32_t) +  // bits
        entries_per_layer * sizeof(float) +                              // scales
        entries_per_layer * sizeof(uint8_t);                             // flags

    size_t fallback_per_layer =
        entries_per_layer * cache->head_dim * sizeof(float);  // worst case

    stats->compressed_bytes = (size_t)cache->n_layers * 2 * compressed_per_layer;  // K + V
    stats->uncompressed_bytes = (size_t)cache->n_layers * 2 *
        entries_per_layer * cache->head_dim * sizeof(float);  // FP32 baseline

    stats->compression_ratio = stats->compressed_bytes > 0
        ? (float)stats->uncompressed_bytes / (float)stats->compressed_bytes
        : 0.0f;
}

void tq_cache_clear(tq_cache *cache) {
    size_t entries_per_layer = (size_t)cache->n_heads * cache->max_seq;
    for (int l = 0; l < cache->n_layers; l++) {
        memset(cache->k_cache[l].flags, 0, entries_per_layer);
        memset(cache->v_cache[l].flags, 0, entries_per_layer);
    }
    cache->total_stores = 0;
    cache->fallback_stores = 0;
}

/**
 * Shift cache entries to mirror llama.cpp's shift_context().
 * Discards the first n_discard positions per head, slides the rest down.
 * This keeps the compressed cache in sync with the ggml KV buffer positions.
 */
/**
 * Shift one layer cache, preserving positions [0, system_pos).
 * Mirrors llama.cpp's seq_rm(system_pos, system_pos+n_discard) +
 * seq_add(system_pos+n_discard, cur_pos, -n_discard).
 *
 * Before: [SYSTEM...][DISCARD...][KEEP...]
 * After:  [SYSTEM...][KEEP...][ZEROED...]
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

        // Shift flags
        memmove(&lc->flags[dst],
                &lc->flags[src],
                (size_t)n_move * sizeof(uint8_t));

        // Shift fallback
        memmove(&lc->fallback[dst * head_dim],
                &lc->fallback[src * head_dim],
                (size_t)n_move * head_dim * sizeof(float));

        // Zero vacated tail
        size_t tail = base + max_seq - n_discard;
        memset(&lc->flags[tail], 0, (size_t)n_discard * sizeof(uint8_t));
    }
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
