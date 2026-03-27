#ifndef TQ_QJL_H
#define TQ_QJL_H

#include <stdint.h>

/**
 * Structured QJL error detection.
 * Instead of dense Gaussian projection (O(d²)), uses sub-sampled Hadamard (O(d log d)).
 * Returns 1 if error exceeds threshold.
 */
int tq_qjl_check_impl(const float *original, const uint32_t *bits, int d,
                       float scale, float threshold, int m);

#endif
