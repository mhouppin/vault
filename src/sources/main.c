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

#include <pthread.h>
#include <stdio.h>
#include "endgame.h"
#include "engine.h"
#include "network.h"
#include "option.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"
#include "worker.h"

board_t Board = {};
pthread_attr_t WorkerSettings;
goparams_t SearchParams;
option_list_t OptionList;
movelist_t SearchMoves;

uint64_t Seed = 1048592ul;

ucioptions_t Options = {
    1, 16, 100, 1, NULL, false, false
};

Network NN = {};

char *Selfdir = NULL;
char *Basedir = NULL;

timeman_t Timeman;

const char *Delimiters = " \r\t\n";

int main(int argc, char **argv)
{
    if (nn_create(&NN, 1, (size_t[]){736, 1}, (int[]){Identity}))
    {
        perror("Unable to create network");
        return (1);
    }

#if defined(_WIN32) || defined (_WIN64)
    char folderSep = '\\';
#else
    char folderSep = '/';
#endif

    Basedir = malloc(3);

    if (Basedir == NULL)
    {
        perror("Unable to allocate memory for folder string");
        return (1);
    }

    Basedir[0] = '.';
    Basedir[1] = folderSep;
    Basedir[2] = '\0';

    char *lastFolderSep = strrchr(*argv, folderSep);

    if (lastFolderSep == NULL)
        Selfdir = strdup(Basedir);
    else
    {
        size_t len = (size_t)(lastFolderSep - *argv + 1);

        Selfdir = malloc(len + 1);

        if (Selfdir != NULL)
        {
            memcpy(Selfdir, *argv, len);
            Selfdir[len] = '\0';
        }
    }

    if (Selfdir == NULL)
    {
        perror("Unable to allocate memory for folder string");
        return (1);
    }

    bitboard_init();
    psq_score_init();
    zobrist_init();
    cyclic_init();
    init_kpk_bitbase();
    init_endgame_table();

    tt_resize(16);
    init_reduction_table();
    pthread_attr_init(&WorkerSettings);
    pthread_attr_setstacksize(&WorkerSettings, 4ul * 1024 * 1024);
    wpool_init(&WPool, 1);

    // Wait for the engine thread to be ready

    worker_wait_search_end(wpool_main_worker(&WPool));
    uci_loop(argc, argv);
    wpool_init(&WPool, 0);

    return (0);
}
