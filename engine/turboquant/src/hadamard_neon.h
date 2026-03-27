#ifndef TQ_HADAMARD_NEON_H
#define TQ_HADAMARD_NEON_H

// ARM NEON optimized FWHT for d=128
// Falls back to scalar if NEON not available
void tq_fwht_neon_impl(float *data, int d);
void tq_sign_flip_neon_impl(float *data, const float *signs, int d);

#endif
