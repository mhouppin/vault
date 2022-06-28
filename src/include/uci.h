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

#ifndef UCI_H
#define UCI_H

#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <time.h>
#include "worker.h"

#if (SIZE_MAX == UINT64_MAX)
#define FMT_INFO PRIu64
#define KEY_INFO PRIx64
typedef uint64_t info_t;
#define MAX_HASH 33554432
#else
#define FMT_INFO PRIu32
#define KEY_INFO PRIx32
typedef uint32_t info_t;
#define MAX_HASH 2048
#endif

typedef struct ucioptions_s
{
    long threads;
    long hash;
    long moveOverhead;
    long multiPv;
    char *networkFile;
    bool chess960;
    bool ponder;
}
ucioptions_t;

extern pthread_attr_t WorkerSettings;
extern ucioptions_t Options;
extern const char *Delimiters;

typedef struct cmdlink_s
{
    const char *commandName;
    void (*call)(const char *);
}
cmdlink_t;

char *get_next_token(char **str);

const char *move_to_str(move_t move, bool isChess960);
const char *score_to_str(score_t score);
move_t str_to_move(const board_t *board, const char *str);

void uci_bench(const char *args);
void uci_d(const char *args);
void uci_debug(const char *args);
void uci_go(const char *args);
void uci_isready(const char *args);
void uci_ponderhit(const char *args);
void uci_position(const char *args);
void uci_quit(const char *args);
void uci_setoption(const char *args);
void uci_stop(const char *args);
void uci_uci(const char *args);
void uci_ucinewgame(const char *args);
void uci_loop(int argc, char **argv);

#endif
