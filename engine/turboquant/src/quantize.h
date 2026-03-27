#ifndef TQ_QUANTIZE_H
#define TQ_QUANTIZE_H

#include <stdint.h>

// Scalar reference: quantize float vector to 1-bit (sign bits) + FP32 scale
void tq_quantize_1bit_scalar(const float *input, int d, uint32_t *out_bits, float *out_scale);

// Scalar reference: dequantize 1-bit back to float
void tq_dequantize_1bit_scalar(const uint32_t *bits, int d, float scale, float *output);

#endif
