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

#ifndef NETWORK_H
#define NETWORK_H

#include "chess_types.h"
#include "core.h"

enum {
    INPUT_SIZE = 768,
    HIDDEN_SIZE = 64,

    QA = 255,
    QB = 64,
    SCALE = 192,
};

typedef struct __attribute__((aligned(64))) {
    i16 values[HIDDEN_SIZE];
} Accumulator;

typedef struct {
    Accumulator pov[COLOR_NB];
} AccumulatorPair;

typedef struct {
    Accumulator feature_weights[INPUT_SIZE];
    Accumulator feature_bias;
    Accumulator stm_output_weights;
    Accumulator nstm_output_weights;
    i16 output_bias;
} Network;

INLINED u16 feature_index(Piece piece, Square square) {
    return piece_color(piece) * 384 + (piece_type(piece) - PAWN) * 64 + square;
}

// Zero-initializes the accumulator.
void acc_init_zero(Accumulator *acc);

// Initializes the accumulator with the network bias.
void acc_init(Accumulator *restrict acc, const Network *restrict network);

void acc_add(Accumulator *restrict acc, usize idx, const Network *restrict network);

void acc_sub(Accumulator *restrict acc, usize idx, const Network *restrict network);

INLINED void acc_pair_add(AccumulatorPair *restrict acc_pair, usize idx, const Network *restrict network) {
    acc_add(&acc_pair->pov[WHITE], idx, network);
    acc_add(&acc_pair->pov[BLACK], ((idx >= 384) ? idx - 384 : idx + 384) ^ 56, network);
}

INLINED void acc_pair_sub(AccumulatorPair *restrict acc_pair, usize idx, const Network *restrict network) {
    acc_sub(&acc_pair->pov[WHITE], idx, network);
    acc_sub(&acc_pair->pov[BLACK], ((idx >= 384) ? idx - 384 : idx + 384) ^ 56, network);
}

// Zero-initializes the network.
void network_init(Network *network);

// Loads the network from the given file.
void network_load_from_file(Network *network, const char *filename);

// Calculates the output of the network.
i16 network_evaluate(const Network *restrict network, Accumulator *restrict us, Accumulator *restrict them);

#endif
