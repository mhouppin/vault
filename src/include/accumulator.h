#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

#include <network.h>

// Reset the accumulator state by setting it to the network biases.
void acc_reset(const Network *restrict nn, weight_t *acc);

// Update the accumulator state from an input increment.
void acc_increment(const Network *restrict nn, weight_t *restrict acc, size_t index);

// Update the accumulator state from an input decrement.
void acc_decrement(const Network *restrict nn, weight_t *restrict acc, size_t index);

// Compute the network output from the accumulator state. Works the same as in the
// nn_const_compute() function, except that this time the accumulator and the output
// buffer can be smaller than the network input size.
void nn_acc_compute(const Network *restrict nn, weight_t *restrict acc,
    weight_t *restrict outputBuffer);

#endif
