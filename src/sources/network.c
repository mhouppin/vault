/*
**    Stash, a UCI chess playing engine developed from scratch
**    Copyright (C) 2019-2025 Morgan Houppin
**
**    Stash is free software: you can redistribute it and/or modify
**    it under the terms of the GNU General Public License as published by
**    the Free Software Foundation, either version 3 of the License, or
**    (at your option) any later version.
**
**    Stash is distributed in the hope that it will be useful,
**    but WITHOUT ANY WARRANTY; without even the implied warranty of
**    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**    GNU General Public License for more details.
**
**    You should have received a copy of the GNU General Public License
**    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "network.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

// Clipped relu.
static i32 crelu(i16 x) {
    return i16_clamp(x, 0, QA);
}

void acc_init_zero(Accumulator *acc) {
    for (usize i = 0; i < HIDDEN_SIZE; ++i) {
        acc->values[i] = 0;
    }
}

void acc_init(Accumulator *restrict acc, const Network *restrict network) {
    for (usize i = 0; i < HIDDEN_SIZE; ++i) {
        acc->values[i] = network->feature_bias.values[i];
    }
}

void acc_add(Accumulator *restrict acc, usize idx, const Network *restrict network) {
    assert(idx < INPUT_SIZE);

    for (usize i = 0; i < HIDDEN_SIZE; ++i) {
        acc->values[i] += network->feature_weights[idx].values[i];
    }
}

void acc_sub(Accumulator *restrict acc, usize idx, const Network *restrict network) {
    assert(idx < INPUT_SIZE);

    for (usize i = 0; i < HIDDEN_SIZE; ++i) {
        acc->values[i] -= network->feature_weights[idx].values[i];
    }
}

void network_init(Network *network) {
    for (usize i = 0; i < INPUT_SIZE; ++i) {
        acc_init_zero(&network->feature_weights[i]);
    }

    acc_init_zero(&network->feature_bias);
    acc_init_zero(&network->stm_output_weights);
    acc_init_zero(&network->nstm_output_weights);
    network->output_bias = 0;
}

static bool fread_acc(Accumulator *acc, FILE *f) {
    return fread(&acc->values, 2, HIDDEN_SIZE, f) == HIDDEN_SIZE;
}

void network_load_from_file(Network *network, const char *filename) {
    FILE *f = fopen(filename, "rb");

    if (f == NULL) {
        printf("info string unable to load network file: %s\n", strerror(errno));
        network_init(network);
        return;
    }

    for (usize i = 0; i < INPUT_SIZE; ++i) {
        if (!fread_acc(&network->feature_weights[i], f)) {
            goto load_fail;
        }
    }

    if (!fread_acc(&network->feature_bias, f)) {
        goto load_fail;
    }

    if (!fread_acc(&network->stm_output_weights, f)) {
        goto load_fail;
    }

    if (!fread_acc(&network->nstm_output_weights, f)) {
        goto load_fail;
    }

    if (fread(&network->output_bias, 2, 1, f) != 1) {
        goto load_fail;
    }

    fclose(f);
    return;

load_fail:
    printf("info string unable to load network file: File too small?\n");
    fclose(f);
    network_init(network);
}

i16 network_evaluate(const Network *restrict network, Accumulator *restrict us, Accumulator *restrict them) {
    i32 output = 0;

    // STM acc -> output
    for (usize i = 0; i < HIDDEN_SIZE; ++i) {
        output += crelu(us->values[i]) * (i32)network->stm_output_weights.values[i];
    }

    // NSTM acc -> output
    for (usize i = 0; i < HIDDEN_SIZE; ++i) {
        output += crelu(them->values[i]) * (i32)network->nstm_output_weights.values[i];
    }

    // Add output bias.
    output += network->output_bias;

    // Apply scaling.
    output *= SCALE;

    // Remove quantisation.
    output /= (i32)QA * (i32)QB;

    // Clamp to non-guaranteed-win scores.
    return (i16)i32_clamp(output, -9999, 9999);
}
