#ifndef TQ_HADAMARD_H
#define TQ_HADAMARD_H

#include <stdint.h>

// Scalar reference FWHT (works on any platform)
void tq_fwht_scalar(float *data, int d);

// Generate deterministic sign vector from seed
void tq_generate_signs_impl(float *signs, int d, uint64_t seed);

// Element-wise sign flip (scalar)
void tq_sign_flip_scalar(float *data, const float *signs, int d);

#endif
