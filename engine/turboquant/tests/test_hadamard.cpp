/**
 * Unit tests for Hadamard rotation (FWHT).
 * Tests: self-inverse property, norm preservation, NEON vs scalar equivalence.
 */

#include "turboquant.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define D 128
#define EPSILON 1e-5f

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

static float randf() {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

static float norm(const float *v, int d) {
    float s = 0;
    for (int i = 0; i < d; i++) s += v[i] * v[i];
    return sqrtf(s);
}

static float max_diff(const float *a, const float *b, int d) {
    float mx = 0;
    for (int i = 0; i < d; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > mx) mx = diff;
    }
    return mx;
}

void test_self_inverse() {
    printf("Test: FWHT self-inverse property\n");
    float data[D], original[D];
    srand(42);
    for (int i = 0; i < D; i++) {
        data[i] = original[i] = randf();
    }

    tq_fwht(data, D);  // forward
    tq_fwht(data, D);  // inverse (self-inverse)

    float diff = max_diff(data, original, D);
    printf("  Max diff after round-trip: %e\n", diff);
    ASSERT(diff < EPSILON, "FWHT(FWHT(x)) should equal x");
}

void test_norm_preservation() {
    printf("Test: FWHT norm preservation\n");
    float data[D];
    srand(123);
    for (int i = 0; i < D; i++) data[i] = randf();

    float norm_before = norm(data, D);
    tq_fwht(data, D);
    float norm_after = norm(data, D);

    float ratio = norm_after / norm_before;
    printf("  Norm before: %.6f, after: %.6f, ratio: %.6f\n",
           norm_before, norm_after, ratio);
    ASSERT(fabsf(ratio - 1.0f) < EPSILON, "||FWHT(x)|| should equal ||x||");
}

void test_sign_flip_inverse() {
    printf("Test: sign flip is self-inverse\n");
    float data[D], original[D], signs[D];
    srand(77);
    for (int i = 0; i < D; i++) data[i] = original[i] = randf();
    tq_generate_signs(signs, D, 12345);

    tq_sign_flip(data, signs, D);
    tq_sign_flip(data, signs, D);

    float diff = max_diff(data, original, D);
    ASSERT(diff < EPSILON, "Double sign-flip should restore original");
}

void test_randomized_fwht_round_trip() {
    printf("Test: randomized FWHT full round-trip\n");
    float data[D], original[D], signs[D];
    srand(99);
    for (int i = 0; i < D; i++) data[i] = original[i] = randf();
    tq_generate_signs(signs, D, 54321);

    // Forward: sign-flip → FWHT
    tq_sign_flip(data, signs, D);
    tq_fwht(data, D);

    // Inverse: FWHT → sign-flip (both are self-inverse)
    tq_fwht(data, D);
    tq_sign_flip(data, signs, D);

    float diff = max_diff(data, original, D);
    printf("  Max diff: %e\n", diff);
    ASSERT(diff < EPSILON, "Full randomized round-trip should restore original");
}

#if defined(TQ_HAVE_NEON)
void test_neon_matches_scalar() {
    printf("Test: NEON FWHT matches scalar\n");
    float data_scalar[D], data_neon[D];
    srand(200);
    for (int i = 0; i < D; i++) {
        data_scalar[i] = data_neon[i] = randf();
    }

    tq_fwht(data_scalar, D);
    tq_fwht_neon(data_neon, D);

    float diff = max_diff(data_scalar, data_neon, D);
    printf("  Max diff scalar vs NEON: %e\n", diff);
    ASSERT(diff < EPSILON, "NEON and scalar should produce same result");
}
#endif

int main() {
    printf("=== Hadamard (FWHT) Tests ===\n\n");

    test_self_inverse();
    test_norm_preservation();
    test_sign_flip_inverse();
    test_randomized_fwht_round_trip();

#if defined(TQ_HAVE_NEON)
    test_neon_matches_scalar();
#else
    printf("(NEON tests skipped — not on ARM)\n");
#endif

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
