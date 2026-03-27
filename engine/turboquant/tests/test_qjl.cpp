/**
 * Unit tests for QJL error detection.
 */

#include "turboquant.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define D 128

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); tests_failed++; } \
    else { tests_passed++; } \
} while(0)

static float randf() { return ((float)rand() / RAND_MAX) * 2.0f - 1.0f; }

void test_low_error_passes() {
    printf("Test: QJL passes for well-quantized vectors\n");
    srand(42);
    int flagged = 0;
    int n_tests = 100;

    for (int t = 0; t < n_tests; t++) {
        float input[D];
        for (int i = 0; i < D; i++) input[i] = randf();

        // Rotate
        tq_fwht(input, D);

        // Quantize
        uint32_t bits[D / 32];
        float scale;
        tq_quantize_1bit(input, D, bits, &scale);

        // Check with reasonable threshold
        int flag = tq_qjl_check(input, bits, D, scale, 0.3f, 32);
        if (flag) flagged++;
    }

    float rate = (float)flagged / n_tests;
    printf("  Flagged: %d/%d (%.1f%%)\n", flagged, n_tests, rate * 100);
    ASSERT(rate < 0.3f, "Fallback rate should be < 30% for random vectors");
}

void test_high_error_detected() {
    printf("Test: QJL detects high-error vectors\n");
    srand(77);
    int detected = 0;
    int n_tests = 50;

    for (int t = 0; t < n_tests; t++) {
        // Create a vector with energy concentrated in one dimension
        // (bad for 1-bit quantization since scale is based on total norm)
        float input[D];
        memset(input, 0, sizeof(input));
        input[0] = 10.0f;  // all energy in one dim
        input[1] = 0.001f;
        // Rest are near-zero but get scale based on the large value

        uint32_t bits[D / 32];
        float scale;
        tq_quantize_1bit(input, D, bits, &scale);

        // This should be flagged — the reconstruction error is huge
        // because scale = 10/sqrt(128) ≈ 0.88, but most dims should be ~0
        int flag = tq_qjl_check(input, bits, D, scale, 0.15f, 32);
        if (flag) detected++;
    }

    float rate = (float)detected / n_tests;
    printf("  Detected: %d/%d (%.1f%%)\n", detected, n_tests, rate * 100);
    ASSERT(rate > 0.5f, "Should detect >50% of high-error vectors");
}

void test_threshold_sensitivity() {
    printf("Test: lower threshold → more fallbacks\n");
    srand(99);
    float input[D];
    for (int i = 0; i < D; i++) input[i] = randf();
    tq_fwht(input, D);

    uint32_t bits[D / 32];
    float scale;
    tq_quantize_1bit(input, D, bits, &scale);

    int flag_tight = tq_qjl_check(input, bits, D, scale, 0.05f, 32);
    int flag_loose = tq_qjl_check(input, bits, D, scale, 0.99f, 32);

    printf("  Tight threshold (0.05): %s\n", flag_tight ? "FLAGGED" : "OK");
    printf("  Loose threshold (0.99): %s\n", flag_loose ? "FLAGGED" : "OK");
    ASSERT(flag_tight >= flag_loose, "Tighter threshold should flag at least as often");
}

int main() {
    printf("=== QJL Error Detection Tests ===\n\n");
    test_low_error_passes();
    test_high_error_detected();
    test_threshold_sensitivity();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
