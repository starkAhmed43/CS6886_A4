#!/usr/bin/env bash
# Build and run the CS6886 Assignment 4 experiments (E1-E4).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$REPO_DIR/results"
BIN_DIR="$RESULTS_DIR/bin"
CC=gcc
CFLAGS="-O2 -Wall -Werror"
SRC="$REPO_DIR/sparten.c"

mkdir -p "$BIN_DIR"

build() {
    local name="$1"; shift
    "$CC" $CFLAGS "$@" "$SRC" -o "$BIN_DIR/$name" -lm
}

run() {
    local bin="$1" outdir="$2"
    mkdir -p "$outdir"
    (cd "$outdir" && "$BIN_DIR/$bin" > output.txt)
    echo "ran $bin -> $outdir/output.txt"
}

# E1: default build.
build e1
run e1 "$RESULTS_DIR/E1"

# E2: tile shapes.
for tile in 1x16 2x8 4x4 8x2 16x1; do
    r="${tile%x*}"; c="${tile#*x}"
    name="e2_tile_${tile}"
    build "$name" -DTILE_ROWS="$r" -DTILE_COLS="$c"
    run "$name" "$RESULTS_DIR/E2/tile_${tile}"
done

# E3: memory bandwidth.
for bw in 4 8 16 32 64; do
    name="e3_bw_${bw}"
    build "$name" -DBYTES_PER_CYCLE="$bw"
    run "$name" "$RESULTS_DIR/E3/bw_${bw}"
done

# E4: trace of the first gemm_inner batch.
build e4 -DE4_TRACE -DNUM_IMAGES=1
run e4 "$RESULTS_DIR/E4"

"$REPO_DIR/analysis/breakdown.sh"
"$REPO_DIR/analysis/fc_batching.sh"
"$REPO_DIR/analysis/run_ahead.sh"
conda run -n cs6886_a4 python "$REPO_DIR/analysis/plots.py"
