#!/usr/bin/env bash
# Build the run-ahead estimator against a patched copy of sparten.c and run
# it for the report's default config (question 8).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$REPO_DIR/results/bin"
OUT_DIR="$REPO_DIR/results/run_ahead"
SRC="$REPO_DIR/sparten.c"
PATCHED="$BIN_DIR/sparten_patched.c"
RUN_AHEAD_C="$REPO_DIR/analysis/run_ahead.c"
CSV="$OUT_DIR/run_ahead.csv"
TXT="$OUT_DIR/run_ahead.txt"

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
        print "    run_ahead_record_batch();"
        print "    end_batch_base();"
        print "}"
        in_func = 0
    }
}
' "$SRC" > "$PATCHED"

BIN="$BIN_DIR/run_ahead"
gcc -O2 -Wall -Werror -I"$BIN_DIR" -I"$REPO_DIR" "$RUN_AHEAD_C" -o "$BIN" -lm
"$BIN" "$CSV" "$TXT"

echo "wrote $CSV and $TXT"
