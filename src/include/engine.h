/*
**    Vault, a UCI-compliant chess engine derivating from Stash
**    Copyright (C) 2019-2022 Morgan Houppin
**
**    Vault is free software: you can redistribute it and/or modify
**    it under the terms of the GNU General Public License as published by
**    the Free Software Foundation, either version 3 of the License, or
**    (at your option) any later version.
**
**    Vault is distributed in the hope that it will be useful,
**    but WITHOUT ANY WARRANTY; without even the implied warranty of
**    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**    GNU General Public License for more details.
**
**    You should have received a copy of the GNU General Public License
**    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_H
#define ENGINE_H

#include <time.h>
#include "board.h"
#include "history.h"
#include "movepick.h"
#include "worker.h"

enum { MAX_PLIES = 240 };

extern int Reductions[64][64];
extern int Pruning[2][7];

void sort_root_moves(root_move_t *begin, root_move_t *end);
root_move_t *find_root_move(root_move_t *begin, root_move_t *end, move_t move);

void *engine_go(void *ptr);
score_t qsearch(board_t *board, score_t alpha, score_t beta, searchstack_t *ss, bool pvNode);
score_t search(board_t *board, int depth, score_t alpha, score_t beta,
    searchstack_t *ss, bool pvNode);

void update_quiet_history(const board_t *board, int depth,
    move_t bestmove, const move_t quiets[64], int qcount, searchstack_t *ss);
void update_capture_history(const board_t *board, int depth,
    move_t bestmove, const move_t captures[64], int ccount, searchstack_t *ss);

score_t evaluate(const board_t *board);

void init_reduction_table(void);

void *engine_thread(void *nothing);

#endif
