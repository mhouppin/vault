#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stddef.h>
#include "weight.h"

// Typedef for activation functions/derivatives.
typedef void (*Activation)(const weight_t *restrict, weight_t *restrict, size_t);

// Struct for activation functions.
typedef struct _ActivationPair
{
    Activation function;
    Activation derivative;
}
ActivationPair;

enum
{
    Identity, Sigmoid, Tanh, ReLU, ClippedReLU, GELU, Softplus, ELU,
    LeakyReLU, SiLU, Mish, Gaussian, ACTIVATION_COUNT
};

extern const ActivationPair ActivationList[ACTIVATION_COUNT];

void identity_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void identity_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void sigmoid_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void sigmoid_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void tanh_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void tanh_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void relu_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void relu_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void clipped_relu_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void clipped_relu_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void gelu_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void gelu_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void softplus_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void softplus_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void elu_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void elu_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void leaky_relu_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void leaky_relu_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void silu_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void silu_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void mish_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void mish_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void gaussian_a(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);
void gaussian_d(const weight_t *restrict inputs, weight_t *restrict outputs, size_t size);

#endif
