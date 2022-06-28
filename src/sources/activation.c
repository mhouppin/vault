#include <math.h>
#include <string.h>
#include "activation.h"

const ActivationPair ActivationList[ACTIVATION_COUNT] = {
    {&identity_a, &identity_d},
    {&sigmoid_a, &sigmoid_d},
    {NULL, NULL},
    {&relu_a, &relu_d},
    {&clipped_relu_a, &clipped_relu_d},
    {NULL, NULL},
    {NULL, NULL},
    {NULL, NULL},
    {NULL, NULL},
    {NULL, NULL},
    {NULL, NULL},
    {NULL, NULL},
};

void identity_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size)
{
    memcpy(outputs, inputs, size * sizeof(weight_t));
}

void identity_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size)
{
    (void)inputs;

    for (size_t i = 0; i < size; ++i)
        outputs[i] = WG_ONE;
}

void sigmoid_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size)
{
    // TODO: this is only a temporary solution. I need to research a better way
    // of computing the sigmoid on fixed point integers.

    for (size_t i = 0; i < size; ++i)
    {
        double v = (double)inputs[i] / (double)WG_ONE;
        v = 1.0 / (1.0 + exp(-v));
        outputs[i] = v * (double)WG_ONE + 0.5;
    }
}

void sigmoid_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size)
{
    sigmoid_a(inputs, outputs, size);

    for (size_t i = 0; i < size; ++i)
        outputs[i] = ((int64_t)outputs[i] * (WG_ONE - outputs[i])) >> WG_PREC;
}

void relu_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        outputs[i] = inputs[i] < 0 ? 0 : inputs[i];
}

void relu_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        outputs[i] = inputs[i] > 0 ? WG_ONE : 0;
}

void clipped_relu_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        outputs[i] = inputs[i] < 0 ? 0 : inputs[i] > WG_ONE ? WG_ONE : inputs[i];
}

void clipped_relu_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        outputs[i] = (inputs[i] > 0 && inputs[i] < WG_ONE) ? WG_ONE : 0;
}