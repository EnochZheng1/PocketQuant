/**
 * Full pipeline integration test.
 * Tests the complete flow: store → load through the tq_cache API.
 * Measures reconstruction MSE and compression ratio.
 */

#include "turboquant.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define D 128
#define N_LAYERS 2
#define N_HEADS 4
#define MAX_SEQ 64

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); tests_failed++; } \
    else { tests_passed++; } \
} while(0)

static float randf() { return ((float)rand() / RAND_MAX) * 2.0f - 1.0f; }

void test_cache_store_load_round_trip() {
    printf("Test: cache store→load round-trip\n");
    tq_cache *cache = tq_cache_create(N_LAYERS, N_HEADS, D, MAX_SEQ, 12345, 0.3f);

    srand(42);
    float original[D], recovered[D];
    for (int i = 0; i < D; i++) original[i] = randf();

    // Store K vector at layer 0, head 0, position 0
    tq_cache_store(cache, 0, 0, 0, original, 0);

    // Load it back
    tq_cache_load(cache, 0, 0, 0, recovered, 0);

    // Measure reconstruction error
    float mse = 0, energy = 0;
    for (int i = 0; i < D; i++) {
        float diff = original[i] - recovered[i];
        mse += diff * diff;
        energy += original[i] * original[i];
    }
    mse /= D;
    float relative_error = sqrtf(mse / (energy / D));

    printf("  MSE: %.6f, relative error: %.1f%%\n", mse, relative_error * 100);
    ASSERT(relative_error < 0.5f, "Relative error should be < 50%");

    tq_cache_free(cache);
}

void test_cache_multiple_positions() {
    printf("Test: cache with multiple positions and heads\n");
    tq_cache *cache = tq_cache_create(N_LAYERS, N_HEADS, D, MAX_SEQ, 54321, 0.3f);

    srand(77);
    float vectors[8][D];
    float recovered[D];
    int n_vecs = 8;

    // Store 8 vectors across different heads and positions
    for (int v = 0; v < n_vecs; v++) {
        for (int i = 0; i < D; i++) vectors[v][i] = randf();
        int head = v % N_HEADS;
        int pos = v / N_HEADS;
        tq_cache_store(cache, 0, head, pos, vectors[v], 0);
    }

    // Verify each can be loaded independently
    int all_ok = 1;
    for (int v = 0; v < n_vecs; v++) {
        int head = v % N_HEADS;
        int pos = v / N_HEADS;
        tq_cache_load(cache, 0, head, pos, recovered, 0);

        float mse = 0, energy = 0;
        for (int i = 0; i < D; i++) {
            float diff = vectors[v][i] - recovered[i];
            mse += diff * diff;
            energy += vectors[v][i] * vectors[v][i];
        }
        float rel_err = sqrtf((mse / D) / (energy / D));
        if (rel_err > 0.5f) all_ok = 0;
    }
    ASSERT(all_ok, "All vectors should reconstruct with < 50% relative error");

    tq_cache_free(cache);
}

void test_cache_stats() {
    printf("Test: cache statistics\n");
    tq_cache *cache = tq_cache_create(N_LAYERS, N_HEADS, D, MAX_SEQ, 99999, 0.2f);

    srand(100);
    // Store many vectors to get stats
    for (int l = 0; l < N_LAYERS; l++) {
        for (int h = 0; h < N_HEADS; h++) {
            for (int p = 0; p < 16; p++) {
                float vec[D];
                for (int i = 0; i < D; i++) vec[i] = randf();
                tq_cache_store(cache, l, h, p, vec, 0);  // K
                tq_cache_store(cache, l, h, p, vec, 1);  // V
            }
        }
    }

    tq_cache_stats stats;
    tq_cache_get_stats(cache, &stats);

    printf("  Total entries: %zu\n", stats.total_entries);
    printf("  Fallback entries: %zu (%.1f%%)\n",
           stats.fallback_entries, stats.fallback_rate * 100);
    printf("  Compression ratio: %.1fx\n", stats.compression_ratio);

    ASSERT(stats.total_entries == (size_t)N_LAYERS * N_HEADS * 16 * 2,
           "Total entries should match stored count");
    ASSERT(stats.fallback_rate < 0.5f, "Fallback rate should be reasonable");
    ASSERT(stats.compression_ratio > 1.0f, "Should achieve compression");

    tq_cache_free(cache);
}

void test_cache_clear() {
    printf("Test: cache clear resets stats\n");
    tq_cache *cache = tq_cache_create(1, 1, D, MAX_SEQ, 42, 0.3f);

    float vec[D];
    srand(50);
    for (int i = 0; i < D; i++) vec[i] = randf();
    tq_cache_store(cache, 0, 0, 0, vec, 0);

    tq_cache_stats stats;
    tq_cache_get_stats(cache, &stats);
    ASSERT(stats.total_entries == 1, "Should have 1 entry before clear");

    tq_cache_clear(cache);
    tq_cache_get_stats(cache, &stats);
    ASSERT(stats.total_entries == 0, "Should have 0 entries after clear");

    tq_cache_free(cache);
}

void test_kv_independence() {
    printf("Test: K and V caches are independent\n");
    tq_cache *cache = tq_cache_create(1, 1, D, MAX_SEQ, 42, 0.3f);

    float k_vec[D], v_vec[D];
    srand(200);
    for (int i = 0; i < D; i++) {
        k_vec[i] = randf();
        v_vec[i] = randf() * 5.0f;  // different magnitude
    }

    tq_cache_store(cache, 0, 0, 0, k_vec, 0);  // K
    tq_cache_store(cache, 0, 0, 0, v_vec, 1);  // V

    float k_out[D], v_out[D];
    tq_cache_load(cache, 0, 0, 0, k_out, 0);
    tq_cache_load(cache, 0, 0, 0, v_out, 1);

    // K and V should reconstruct differently
    float k_v_diff = 0;
    for (int i = 0; i < D; i++) {
        k_v_diff += fabsf(k_out[i] - v_out[i]);
    }
    printf("  K-V difference: %.2f\n", k_v_diff);
    ASSERT(k_v_diff > 1.0f, "K and V should be different");

    tq_cache_free(cache);
}

int main() {
    printf("=== Full Pipeline Tests ===\n\n");
    test_cache_store_load_round_trip();
    test_cache_multiple_positions();
    test_cache_stats();
    test_cache_clear();
    test_kv_independence();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
