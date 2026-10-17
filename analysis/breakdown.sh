#!/usr/bin/env bash
# Build the cycle-breakdown tool against a patched copy of sparten.c and run
# it for the configs used in the report (question 2).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$REPO_DIR/results/bin"
OUT_DIR="$REPO_DIR/results/breakdown"
SRC="$REPO_DIR/sparten.c"
PATCHED="$BIN_DIR/sparten_patched.c"
BREAKDOWN_C="$REPO_DIR/analysis/breakdown.c"
CSV="$OUT_DIR/breakdown.csv"
TXT="$OUT_DIR/breakdown.txt"

mkdir -p "$BIN_DIR" "$OUT_DIR"

# Patch a copy of sparten.c: rename the given end_batch() to end_batch_base(),
# then add a thin end_batch() that records per-batch data (TODO code path
# unchanged) before calling it. sparten.c itself is never modified.
awk '
{
    if ($0 == "void end_batch(void)") {
        print "static void end_batch_base(void)"
        in_func = 1
        next
    }
    print
    if (in_func && $0 == "}") {
        print ""
        print "void end_batch(void)"
        print "{"
        print "    breakdown_record_batch();"
        print "    end_batch_base();"
        print "}"
        in_func = 0
    }
}
' "$SRC" > "$PATCHED"

: > "$CSV"
: > "$TXT"

run_config() {
    local config="$1" header="$2"; shift 2
    local bin="$BIN_DIR/breakdown_${config}"
    gcc -O2 -Wall -Werror -I"$BIN_DIR" -I"$REPO_DIR" "$@" "$BREAKDOWN_C" -o "$bin" -lm
    "$bin" "$config" "$CSV" "$TXT" "$header"
}

run_config "tile4x4_bw16" 1

for tile in 1x16 2x8 8x2 16x1; do
    r="${tile%x*}"; c="${tile#*x}"
    run_config "tile${tile}_bw16" 0 -DTILE_ROWS="$r" -DTILE_COLS="$c"
done

for bw in 4 8 32 64; do
    run_config "tile4x4_bw${bw}" 0 -DBYTES_PER_CYCLE="$bw"
done

echo "wrote $CSV and $TXT"
