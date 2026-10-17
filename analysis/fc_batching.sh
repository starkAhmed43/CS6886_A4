#!/usr/bin/env bash
# Build and run the fc1/fc2 batching simulation for report question 8:
# batch N images' GEMM rows together so the 4 x 4 tile's 16 PEs are not
# mostly idle on fc1/fc2's single-row GEMM.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$REPO_DIR/results/bin"
OUT_DIR="$REPO_DIR/results/fc_batching"
SRC="$REPO_DIR/analysis/fc_batching.c"
BIN="$BIN_DIR/fc_batching"

mkdir -p "$BIN_DIR" "$OUT_DIR"

gcc -O2 -Wall -Werror -I"$REPO_DIR" "$SRC" -o "$BIN" -lm

(cd "$OUT_DIR" && "$BIN")

echo "wrote $OUT_DIR/fc_batching.csv and $OUT_DIR/fc_batching.txt"
