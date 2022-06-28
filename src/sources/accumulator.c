#include <string.h>
#include "accumulator.h"
#include "matrix.h"

void acc_reset(const Network *restrict nn, weight_t *acc)
{
    memcpy(acc, nn->weights + nn->layerSizes[0] * nn->layerSizes[1],
        sizeof(weight_t) * nn->layerSizes[1]);
}

void acc_increment(const Network *restrict nn, weight_t *restrict acc, size_t index)
{
    wincrement(acc, nn->weights + index * nn->layerSizes[1], nn->layerSizes[1]);
}

void acc_decrement(const Network *restrict nn, weight_t *restrict acc, size_t index)
{
    wdecrement(acc, nn->weights + index * nn->layerSizes[1], nn->layerSizes[1]);
}

void nn_acc_compute(const Network *restrict nn, weight_t *restrict acc,
    weight_t *restrict outputBuffer)
{
    nn->activations[0](acc, outputBuffer, nn->layerSizes[1]);

    for (size_t l = 1; l < nn->layers; ++l)
    {
        // Preload some constant values to simplify further calculations

        const size_t inputSize = nn->layerSizes[l];
        const size_t outputSize = nn->layerSizes[l + 1];
        weight_t *const weights = nn->weights + nn->layerOffsets[l];

        wforwardprop(acc, outputBuffer, weights, outputSize, inputSize);

        nn->activations[l](acc, outputBuffer, outputSize);
    }
}
