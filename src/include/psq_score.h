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

#ifndef PSQ_SCORE_H
#define PSQ_SCORE_H

#include "types.h"

enum
{
    PAWN_MG_SCORE = 75,
    KNIGHT_MG_SCORE = 348,
    BISHOP_MG_SCORE = 366,
    ROOK_MG_SCORE = 490,
    QUEEN_MG_SCORE = 1078,

    PAWN_EG_SCORE = 162,
    KNIGHT_EG_SCORE = 577,
    BISHOP_EG_SCORE = 614,
    ROOK_EG_SCORE = 974,
    QUEEN_EG_SCORE = 1823
};

extern const score_t PieceScores[PHASE_NB][PIECE_NB];
extern scorepair_t PsqScore[PIECE_NB][SQUARE_NB];

void psq_score_init(void);

#endif // PSQ_SCORE_H
