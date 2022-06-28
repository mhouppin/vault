#ifndef NETWORK_H
#define NETWORK_H

#include <stdio.h>
#include "activation.h"
#include "weight.h"

// API for loading a single u32 from a file.

int integer_load(FILE *fp, uint32_t *u);
int integer_save(FILE *fp, uint32_t u);

// General struct for ANNs.
//
// Note: unless initialization fails during nn_load() or nn_create(),
// nn_destroy() is never called in the API. If you have previously allocated
// resources on a network, please call nn_destroy() before using these
// functions to avoid memory leaks.
typedef struct _Network
{
    // Number of layers (including the output layer).
    size_t layers;

    // Array denoting the number of neurons per layer, not including biases.
    size_t *layerSizes;

    // Array holding all the weights of the network. We store them contiguously
    // to avoid having to do multiple allocations at initialization time.
    weight_t *weights;

    // Array of pre-computed offsets for accessing the weights of a layer.
    // Intended as both a speedup and a simplification of the
    // inference/backprop code.
    size_t *layerOffsets;

    // List of activation IDs for each layer.
    int *activationIds;

    // List of activation functions for each layer.
    Activation *activations;

    // List of activation derivatives for each layer.
    Activation *derivatives;

    // Two arrays capable of holding all inputs/outputs of a layer, used for
    // inference and backprop computations.
    weight_t *cpuInput;
    weight_t *cpuOutput;
}
Network;

// Creates a new network with the given parameters, with all weights and biases
// zeroed. Returns 0 if sucessful, a non-zero integer otherwise.
int nn_create(Network *nn, size_t layers, const size_t layerSizes[], int activationIds[]);

// Loads a network from a file. Returns 0 if sucessful, a non-zero integer
// otherwise.
int nn_load(Network *nn, const char *filename);

// Saves a network from a file. Returns 0 if sucessful, a non-zero integer
// otherwise.
int nn_save(Network *nn, const char *filename);

// Changes the activation function of a specific layer.
void nn_set_layer_activation(Network *nn, size_t layer, int activationId);

// Computes the outputs for the given inputs and returns them in the user-given
// buffer.
void nn_compute(Network *restrict nn, const weight_t *restrict inputs,
    weight_t *restrict outputs);

// Computes the outputs for the given inputs in ioBuffer and returns them in the
// same buffer. This function requires that both cpuBuffer and ioBuffer are able
// to hold the largest layer of the network in memory.
void nn_const_compute(const Network *restrict nn, weight_t *restrict ioBuffer,
    weight_t *restrict cpuBuffer);

// Randomize all weights of the network given the value range and the initial
// seed. Note that this yields the same results as making a loop of
// nn_init_layer_weights() with the same parameters.
void nn_init_all_weights(Network *nn, weight_t minValue, weight_t maxValue, int seed);

// Randomize all weights of the specified layer given the value range and the
// initial seed.
void nn_init_layer_weights(Network *nn, weight_t minValue, weight_t maxValue, int seed, size_t layer);

// Frees all memory allocated by the network.
void nn_destroy(Network *nn);

#endif
