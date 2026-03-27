/**
 * Unit tests for 1-bit quantization/dequantization.
 */

#include "turboquant.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define D 128

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); tests_failed++; } \
    else { tests_passed++; } \
} while(0)

static float randf() { return ((float)rand() / RAND_MAX) * 2.0f - 1.0f; }

void test_sign_preservation() {
    printf("Test: quantize preserves sign bits\n");
    float input[D];
    uint32_t bits[D / 32];
    float scale;

    srand(42);
    for (int i = 0; i < D; i++) input[i] = randf();

    tq_quantize_1bit(input, D, bits, &scale);

    int correct = 0;
    for (int w = 0; w < D / 32; w++) {
        for (int b = 0; b < 32; b++) {
            int idx = w * 32 + b;
            int bit = (bits[w] >> b) & 1;
            int expected = (input[idx] >= 0.0f) ? 1 : 0;
            if (bit == expected) correct++;
        }
    }
    printf("  Correct sign bits: %d/%d\n", correct, D);
    ASSERT(correct == D, "All sign bits should match");
}

void test_round_trip_shape() {
    printf("Test: quantize→dequantize produces correct signs\n");
    float input[D], output[D];
    uint32_t bits[D / 32];
    float scale;

    srand(55);
    for (int i = 0; i < D; i++) input[i] = randf();

    tq_quantize_1bit(input, D, bits, &scale);
    tq_dequantize_1bit(bits, D, scale, output);

    int sign_match = 0;
    for (int i = 0; i < D; i++) {
        if ((input[i] >= 0) == (output[i] >= 0)) sign_match++;
    }
    printf("  Sign matches: %d/%d, scale: %.6f\n", sign_match, D, scale);
    ASSERT(sign_match == D, "Dequantized signs should match input signs");
}

void test_scale_magnitude() {
    printf("Test: scale reflects input magnitude\n");
    float input[D];
    uint32_t bits[D / 32];
    float scale;

    // Unit vector — norm = 1.0, scale should be 1/sqrt(128)
    for (int i = 0; i < D; i++) input[i] = 0.0f;
    input[0] = 1.0f;

    tq_quantize_1bit(input, D, bits, &scale);
    float expected_scale = 1.0f / sqrtf((float)D);
    printf("  Scale: %.6f, expected: %.6f\n", scale, expected_scale);
    ASSERT(fabsf(scale - expected_scale) < 0.001f, "Scale should be norm/sqrt(d)");
}

void test_mse_is_bounded() {
    printf("Test: reconstruction MSE is bounded\n");
    float input[D], output[D];
    uint32_t bits[D / 32];
    float scale;

    srand(88);
    // Gaussian-like input (what rotated vectors look like)
    for (int i = 0; i < D; i++) input[i] = randf();

    tq_quantize_1bit(input, D, bits, &scale);
    tq_dequantize_1bit(bits, D, scale, output);

    float mse = 0;
    float input_energy = 0;
    for (int i = 0; i < D; i++) {
        float diff = input[i] - output[i];
        mse += diff * diff;
        input_energy += input[i] * input[i];
    }
    mse /= D;
    float relative_error = sqrtf(mse / (input_energy / D));
    printf("  MSE: %.6f, relative error: %.2f%%\n", mse, relative_error * 100);
    ASSERT(relative_error < 0.5f, "Relative error should be < 50% for random input");
}

int main() {
    printf("=== 1-Bit Quantization Tests ===\n\n");
    test_sign_preservation();
    test_round_trip_shape();
    test_scale_magnitude();
    test_mse_is_bounded();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
