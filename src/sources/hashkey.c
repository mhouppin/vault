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

# include <math.h>
# include "bitboard.h"
# include "hashkey.h"
# include "random.h"

hashkey_t ZobristPsq[PIECE_NB][SQUARE_NB];
hashkey_t ZobristEnPassant[FILE_NB];
hashkey_t ZobristCastling[CASTLING_NB];
hashkey_t ZobristBlackToMove;

void zobrist_init(void)
{
    uint64_t seed = 0x7F6E5D4C3B2A1908ull;

    for (piece_t piece = WHITE_PAWN; piece <= BLACK_KING; ++piece)
        for (square_t square = SQ_A1; square <= SQ_H8; ++square)
            ZobristPsq[piece][square] = qrandom(&seed);

    for (file_t file = FILE_A; file <= FILE_H; ++file)
        ZobristEnPassant[file] = qrandom(&seed);

    for (int cr = 0; cr < CASTLING_NB; ++cr)
    {
        ZobristCastling[cr] = 0;
        bitboard_t b = cr;
        while (b)
        {
            hashkey_t k = ZobristCastling[1ull << bb_pop_first_sq(&b)];
            ZobristCastling[cr] ^= k ? k : qrandom(&seed);
        }
    }

    ZobristBlackToMove = qrandom(&seed);
}
