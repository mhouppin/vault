#!/usr/bin/env bash

set -e

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"
cd ..

git lfs fetch origin main

cd src

CFLAGS="$CFLAGS -fprofile-generate" LDFLAGS="$LDFLAGS -lgcov" ARCH="$ARCH" make re

./vault "setoption name EvalFile value ../default.nn" bench

CFLAGS="$CFLAGS -fprofile-use -fno-peel-loops -fno-tracer" LDFLAGS="$LDFLAGS -lgcov" \
    ARCH="$ARCH" make re

make clean

rm src/sources/*.gcda
