#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include "weight.h"

// Multiplies dst to src element-wise, and stores the values in dst.
void whadamard(weight_t *restrict dst, const weight_t *restrict src, size_t size);

// Propagates the values from a layer to the next one via matrix multiplication.
void wforwardprop(weight_t *restrict dst, const weight_t *restrict src,
    const weight_t *restrict weights, size_t dstSize, size_t srcSize);

// Backpropagates the error from a layer to the previous one via matrix multiplication.
// (In this case, src is the layer L and dst is the layer (L-1).)
void wbackprop(weight_t *restrict dst, const weight_t *restrict src,
    const weight_t *restrict weights, size_t dstSize, size_t srcSize);

// Updates the gradient values from the error and the layer output values.
void wgradupdate(weight_t *restrict gradient, const weight_t *restrict error,
    const weight_t *src, size_t inputSize, size_t outputSize);

// Increments the first layer pre-activation values.
void wincrement(weight_t *restrict accumulator, const weight_t *restrict weights,
    size_t accSize);

// Decrements the first layer pre-activation values.
void wdecrement(weight_t *restrict accumulator, const weight_t *restrict weights,
    size_t accSize);

#endif