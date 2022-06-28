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

#ifndef BITBOARD_H
#define BITBOARD_H

#if (defined(USE_PREFETCH) || defined(USE_POPCNT) || defined(USE_PEXT))
// Do not include the header if unspecified, because some compilers might
// not have it.
#include <immintrin.h>
#endif

#include "types.h"

typedef uint64_t bitboard_t;

#define FILE_A_BITS 0x0101010101010101ull
#define FILE_B_BITS 0x0202020202020202ull
#define FILE_C_BITS 0x0404040404040404ull
#define FILE_D_BITS 0x0808080808080808ull
#define FILE_E_BITS 0x1010101010101010ull
#define FILE_F_BITS 0x2020202020202020ull
#define FILE_G_BITS 0x4040404040404040ull
#define FILE_H_BITS 0x8080808080808080ull

#define RANK_1_BITS 0x00000000000000FFull
#define RANK_2_BITS 0x000000000000FF00ull
#define RANK_3_BITS 0x0000000000FF0000ull
#define RANK_4_BITS 0x00000000FF000000ull
#define RANK_5_BITS 0x000000FF00000000ull
#define RANK_6_BITS 0x0000FF0000000000ull
#define RANK_7_BITS 0x00FF000000000000ull
#define RANK_8_BITS 0xFF00000000000000ull

#define FULL_BITS 0xFFFFFFFFFFFFFFFFull

#define DARK_SQUARES 0xAA55AA55AA55AA55ull

#define KINGSIDE_BITS 0xF0F0F0F0F0F0F0F0ull
#define QUEENSIDE_BITS 0x0F0F0F0F0F0F0F0Full
#define CENTER_FILES_BITS 0x3C3C3C3C3C3C3C3Cull

extern bitboard_t LineBits[SQUARE_NB][SQUARE_NB];
extern bitboard_t PseudoMoves[PIECETYPE_NB][SQUARE_NB];
extern bitboard_t PawnMoves[COLOR_NB][SQUARE_NB];

typedef struct magic_s
{
    bitboard_t mask;
    bitboard_t magic;
    bitboard_t *moves;
    unsigned int shift;
}
magic_t;

INLINED unsigned int magic_index(const magic_t *magic, bitboard_t occupied)
{
#ifdef USE_PEXT
    return (_pext_u64(occupied, magic->mask));
#else
    return ((unsigned int)(((occupied & magic->mask) * magic->magic) >> magic->shift));
#endif
}

extern magic_t RookMagics[SQUARE_NB];
extern magic_t BishopMagics[SQUARE_NB];

void bitboard_init(void);

INLINED bitboard_t square_bb(square_t square)
{
    return ((bitboard_t)1 << square);
}

INLINED bitboard_t shift_up(bitboard_t b)
{
    return (b << 8);
}

INLINED bitboard_t shift_down(bitboard_t b)
{
    return (b >> 8);
}

INLINED bitboard_t shift_left(bitboard_t b)
{
    return ((b & ~FILE_A_BITS) >> 1);
}

INLINED bitboard_t shift_right(bitboard_t b)
{
    return ((b & ~FILE_H_BITS) << 1);
}

INLINED bitboard_t shift_up_left(bitboard_t b)
{
    return ((b & ~FILE_A_BITS) << 7);
}

INLINED bitboard_t shift_up_right(bitboard_t b)
{
    return ((b & ~FILE_H_BITS) << 9);
}

INLINED bitboard_t shift_down_left(bitboard_t b)
{
    return ((b & ~FILE_A_BITS) >> 9);
}

INLINED bitboard_t shift_down_right(bitboard_t b)
{
    return ((b & ~FILE_H_BITS) >> 7);
}

INLINED bitboard_t relative_shift_up(bitboard_t b, color_t c)
{
    return ((c == WHITE) ? shift_up(b) : shift_down(b));
}

INLINED bitboard_t relative_shift_down(bitboard_t b, color_t c)
{
    return ((c == WHITE) ? shift_down(b) : shift_up(b));
}

INLINED bool more_than_one(bitboard_t b)
{
    return (b & (b - 1));
}

INLINED bitboard_t file_bb(file_t file)
{
    return (FILE_A_BITS << file);
}

INLINED bitboard_t sq_file_bb(square_t square)
{
    return (file_bb(sq_file(square)));
}

INLINED bitboard_t rank_bb(rank_t rank)
{
    return (RANK_1_BITS << (8 * rank));
}

INLINED bitboard_t sq_rank_bb(square_t square)
{
    return (rank_bb(sq_rank(square)));
}

INLINED bitboard_t between_bb(square_t sq1, square_t sq2)
{
    return (LineBits[sq1][sq2] & ((FULL_BITS << (sq1 + (sq1 < sq2))) ^ (FULL_BITS << (sq2 + !(sq1 < sq2)))));
}

INLINED bool sq_aligned(square_t sq1, square_t sq2, square_t sq3)
{
    return (LineBits[sq1][sq2] & square_bb(sq3));
}

INLINED bitboard_t bishop_moves_bb(square_t square, bitboard_t occupied)
{
    const magic_t *magic = &BishopMagics[square];

    return (magic->moves[magic_index(magic, occupied)]);
}

INLINED bitboard_t rook_moves_bb(square_t square, bitboard_t occupied)
{
    const magic_t *magic = &RookMagics[square];

    return (magic->moves[magic_index(magic, occupied)]);
}

INLINED bitboard_t wpawns_attacks_bb(bitboard_t b)
{
    return (shift_up_left(b) | shift_up_right(b));
}

INLINED bitboard_t bpawns_attacks_bb(bitboard_t b)
{
    return (shift_down_left(b) | shift_down_right(b));
}

INLINED bitboard_t wpawns_2attacks_bb(bitboard_t b)
{
    return (shift_up_left(b) & shift_up_right(b));
}

INLINED bitboard_t bpawns_2attacks_bb(bitboard_t b)
{
    return (shift_down_left(b) & shift_down_right(b));
}

INLINED bitboard_t adjacent_files_bb(square_t s)
{
    bitboard_t fileBB = sq_file_bb(s);
    return (shift_left(fileBB) | shift_right(fileBB));
}

INLINED bitboard_t forward_ranks_bb(color_t c, square_t s)
{
    if (c == WHITE)
        return (~RANK_1_BITS << 8 * sq_rank(s));
    else
        return (~RANK_8_BITS >> 8 * (RANK_8 - sq_rank(s)));
}

INLINED bitboard_t forward_file_bb(color_t c, square_t s)
{
    return (forward_ranks_bb(c, s) & sq_file_bb(s));
}

INLINED bitboard_t pawn_attack_span_bb(color_t c, square_t s)
{
    return (forward_ranks_bb(c, s) & adjacent_files_bb(s));
}

INLINED bitboard_t passed_pawn_span_bb(color_t c, square_t s)
{
    return (forward_ranks_bb(c, s) & (adjacent_files_bb(s) | sq_file_bb(s)));
}

INLINED int popcount(bitboard_t b)
{
#ifndef USE_POPCNT
    const bitboard_t m1 = 0x5555555555555555ull;
    const bitboard_t m2 = 0x3333333333333333ull;
    const bitboard_t m4 = 0x0F0F0F0F0F0F0F0Full;
    const bitboard_t hx = 0x0101010101010101ull;

    b -= (b >> 1) & m1;
    b = (b & m2) + ((b >> 2) & m2);
    b = (b + (b >> 4)) & m4;
    return ((b * hx) >> 56);

#elif defined(_MSC_VER) || defined (__INTEL_COMPILER)

    return ((int)_mm_popcnt_u64(b));

#else

    return (__builtin_popcountll(b));

#endif
}

#if defined(__GNUC__)

INLINED square_t bb_first_sq(bitboard_t b)
{
    return (__builtin_ctzll(b));
}

INLINED square_t bb_last_sq(bitboard_t b)
{
    return (SQ_H8 ^ __builtin_clzll(b));
}

#elif defined(_MSC_VER)

INLINED square_t bb_first_sq(bitboard_t b)
{
    unsigned long index;
    _BitScanForward64(&index, b);
    return ((square_t)index);
}

INLINED square_t bb_last_sq(bitboard_t b)
{
    unsigned long index;
    _BitScanReverse64(&index, b);
    return ((square_t)index);
}

#else
#error "Unsupported compiler."
#endif

INLINED square_t bb_pop_first_sq(bitboard_t *b)
{
    const square_t square = bb_first_sq(*b);
    *b &= *b - 1;
    return (square);
}

INLINED square_t bb_relative_last_sq(color_t c, bitboard_t b)
{
    return (c == WHITE ? bb_last_sq(b) : bb_first_sq(b));
}

INLINED void prefetch(void *ptr __attribute__((unused)))
{
#ifdef USE_PREFETCH
    _mm_prefetch(ptr, _MM_HINT_T0);
#elif defined(__GNUC__)
    __builtin_prefetch(ptr);
#endif
}

#endif // BITBOARD_H
