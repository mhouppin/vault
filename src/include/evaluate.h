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

#ifndef EVALUATE_H
#define EVALUATE_H

#include "board.h"
#include "network.h"

Score evaluate(const Board *restrict board, AccumulatorPair *restrict acc_pair);

Score evaluate_noacc(const Board *board);

Score nn_evaluate(const Board *restrict board, const Network *restrict network, AccumulatorPair *restrict acc_pair);

Score nn_evaluate_noacc(const Board *board, const Network *network);

#endif
