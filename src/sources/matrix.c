#include <string.h>
#include "matrix.h"

// Note: there's some precision loss on matrix multiplications for now, still
// thinking about a nice solution to fix that without impacting performance
// much.

void whadamard(weight_t *restrict dst, const weight_t *restrict src, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        dst[i] = ((int64_t)dst[i] * src[i]) >> WG_PREC;
}

void wforwardprop(weight_t *restrict dst, const weight_t *restrict src,
    const weight_t *restrict weights, size_t dstSize, size_t srcSize)
{
    // Start by setting dst to the biases to avoid
    // having to set the buffer twice.
    memcpy(dst, weights + srcSize * dstSize, sizeof(weight_t) * dstSize);

    // Then perform all matrix multiplications.
    for (size_t i = 0; i < srcSize; ++i)
    {
        weight_t v = src[i];

        if (v == 0)
            continue ;

        else if (v == WG_ONE)
            for (size_t k = 0; k < dstSize; ++k)
                dst[k] += weights[i * dstSize + k];

        else
            for (size_t k = 0; k < dstSize; ++k)
                dst[k] += ((int64_t)v * weights[i * dstSize + k]) >> WG_PREC;
    }
}

void wbackprop(weight_t *restrict dst, const weight_t *restrict src,
    const weight_t *restrict weights, size_t dstSize, size_t srcSize)
{
    for (size_t i = 0; i < dstSize; ++i)
        dst[i] = 0;

    for (size_t i = 0; i < dstSize; ++i)
        for (size_t k = 0; k < srcSize; ++k)
            dst[i] += ((int64_t)src[k] * weights[i * srcSize + k]) >> WG_PREC;
}

void wgradupdate(weight_t *restrict gradient, const weight_t *restrict error,
    const weight_t *src, size_t inputSize, size_t outputSize)
{
    for (size_t i = 0; i < inputSize; ++i)
        for (size_t o = 0; o < outputSize; ++o)
            gradient[i * outputSize + o] += ((int64_t)error[o] * src[i]) >> WG_PREC;

    for (size_t o = 0; o < outputSize; ++o)
        gradient[inputSize * outputSize + o] += error[o];
}

void wincrement(weight_t *restrict accumulator, const weight_t *restrict weights,
    size_t accSize)
{
    for (size_t i = 0; i < accSize; ++i)
        accumulator[i] += weights[i];
}

void wdecrement(weight_t *restrict accumulator, const weight_t *restrict weights,
    size_t accSize)
{
    for (size_t i = 0; i < accSize; ++i)
        accumulator[i] -= weights[i];
}
