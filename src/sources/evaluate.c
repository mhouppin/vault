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

#include "evaluate.h"

#include "movelist.h"
#include "psq_table.h"

// clang-format on

static bool opposite_colored_bishops(Bitboard bishops) {
    return (bishops & DSQ_BB) && (bishops & LSQ_BB);
}

static bool is_kxk_endgame(const Board *board, Color us) {
    // If the weak side has pieces or Pawns, this is not a KXK endgame.
    if (bb_more_than_one(board_color_bb(board, color_flip(us)))) {
        return false;
    }

    return board->stack->material[us] >= ROOK_MG_SCORE;
}

// Bonus for pieces close to the corner.
static Score corner_bonus(Square square) {
    const Rank rank = square_rank(square);
    const File file = square_file(square);
    const i16 rdist = i16_min(rank, rank ^ 0b111u);
    const i16 fdist = i16_min(file, file ^ 0b111u);

    return 50 - 2 * (fdist * fdist + rdist * rdist);
}

// Bonus for pieces close to each other.
static Score close_bonus(Square square1, Square square2) {
    return 70 - 10 * square_distance(square1, square2);
}

static Score eval_kbnk(const Board *board, Color us) {
    const Square our_king = board_king_square(board, us);
    Square their_king = board_king_square(board, color_flip(us));
    Score score = VICTORY + KNIGHT_MG_SCORE + BISHOP_MG_SCORE + close_bonus(their_king, our_king);

    // Don't push the King to the wrong corner.
    if (board_piecetype_bb(board, BISHOP) & DSQ_BB) {
        their_king = square_flip(their_king);
    }

    score += i8_abs((i8)square_file(their_king) - (i8)square_rank(their_king)) * 100;
    return board->side_to_move == us ? score : -score;
}

static Score eval_kxk(const Board *board, Color us) {
    // Be careful to avoid stalemating the weak King.
    if (board->side_to_move != us && !board->stack->checkers) {
        Movelist list;

        movelist_generate_legal(&list, board);

        if (movelist_size(&list) == 0) {
            return 0;
        }
    }

    // KBNK needs some special handling for mating correctly.
    if (board->stack->material[us] == KNIGHT_MG_SCORE + BISHOP_MG_SCORE) {
        return eval_kbnk(board, us);
    }

    const Square winning_ksq = board_king_square(board, us);
    const Square losing_ksq = board_king_square(board, color_flip(us));
    Score score = board->stack->material[us]
        + board_piece_count(board, create_piece(us, PAWN)) * PAWN_MG_SCORE;

    // Push the weak King to the corner.
    score += corner_bonus(losing_ksq);

    // Keep the two Kings close for mating with low material.
    score += close_bonus(winning_ksq, losing_ksq);

    // Set the score as winning if we have mating material:
    // - a major piece;
    // - a Bishop and a Knight;
    // - two opposite colored Bishops;
    // - three Knights.
    const Bitboard knights = board_piecetype_bb(board, KNIGHT);
    const Bitboard bishops = board_piecetype_bb(board, BISHOP);

    if (board_piecetypes_bb(board, QUEEN, ROOK) || (knights && bishops)
        || opposite_colored_bishops(bishops) || bb_popcount(knights) >= 3) {
        score += VICTORY;
    }

    return (board->side_to_move == us) ? score : -score;
}

Score evaluate_noacc(const Board *board) {
    extern Network GlobalNetwork;

    // Is there a KXK situation ? (lone King vs mating material)
    if (is_kxk_endgame(board, WHITE)) {
        return eval_kxk(board, WHITE);
    }

    if (is_kxk_endgame(board, BLACK)) {
        return eval_kxk(board, BLACK);
    }

    return nn_evaluate_noacc(board, &GlobalNetwork);
}

Score evaluate(const Board *board, AccumulatorPair *restrict acc_pair) {
    extern Network GlobalNetwork;

    // Is there a KXK situation ? (lone King vs mating material)
    if (is_kxk_endgame(board, WHITE)) {
        return eval_kxk(board, WHITE);
    }

    if (is_kxk_endgame(board, BLACK)) {
        return eval_kxk(board, BLACK);
    }

    return nn_evaluate(board, &GlobalNetwork, acc_pair);
}

Score nn_evaluate_noacc(const Board *board, const Network *network) {
    Accumulator white, black;

    acc_init(&white, network);
    acc_init(&black, network);

    for (Bitboard occ = board_occupancy_bb(board); occ;) {
        Square sq = bb_pop_first_square(&occ);
        Piece pc = board_piece_on(board, sq);
        u16 widx = feature_index(pc, sq);
        u16 bidx = feature_index(opposite_piece(pc), square_flip(sq));

        acc_add(&white, widx, network);
        acc_add(&black, bidx, network);
    }

    if (board->side_to_move == WHITE) {
        return network_evaluate(network, &white, &black);
    } else {
        return network_evaluate(network, &black, &white);
    }
}

Score nn_evaluate(
    const Board *restrict board,
    const Network *restrict network,
    AccumulatorPair *restrict acc_pair
) {
    const Color us = board->side_to_move;
    const Color them = color_flip(us);

    return network_evaluate(network, &acc_pair->pov[us], &acc_pair->pov[them]);
}
