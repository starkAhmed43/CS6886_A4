/* run_ahead.c - report question 8: what the inner product would cost if PEs
 * could run ahead through the depth tiles of one output tile without a
 * barrier at every batch, barrier-synchronizing only once per output tile.
 *
 * Built by analysis/run_ahead.sh against a patched copy of sparten.c (see
 * that script) where end_batch() first calls run_ahead_record_batch() and
 * then the given end_batch_base(). sparten.c itself is never changed.
 *
 * Today, every batch (one depth tile of one output tile) ends with a
 * barrier, so a batch takes its slowest PE. In the run-ahead design, each PE
 * keeps its output for the whole output tile and runs through all of that
 * tile's depth tiles back to back; only then do the PEs barrier. So for one
 * output tile:
 *   compute = max over PEs of (that PE's cycles summed over the tile's depth
 *             tiles)
 *   memory  = ceil(sum of the tile's batch bytes / BYTES_PER_CYCLE)
 *   time    = max(compute, memory)
 * summed over a layer's output tiles.
 *
 * The depth tiles of one output tile are exactly D = ceil(depth / 16)
 * consecutive batches, where depth is the GEMM's reduction dimension (9 x
 * in_channels for a conv layer, in_channels for an FC layer - see
 * lower_conv/lower_fc). gemm_inner emits batches in output-tile-major,
 * depth-tile-minor order, so grouping every D consecutive batches of a given
 * Stats pointer into one output tile reproduces that structure exactly.
 *
 * The baseline, perfect-balance bound, memory floor and compute peak don't
 * need any new per-batch bookkeeping: they are already in Stats (cycles,
 * busy_pe_cycles, bytes, macs).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void run_ahead_record_batch(void);

#define main sparten_main
#include "sparten_patched.c"
#undef main

/* Per-(layer, order) run-ahead tracker, keyed by the Stats pointer `current`
 * in first-seen order, same as breakdown.c: conv1 inner, conv1 outer, conv2
 * inner, conv2 outer, ... Only the inner entries (even index) are used. */
typedef struct {
    const Stats *stats;
    int depth_tiles;              /* D: depth tiles per output tile          */
    int batch_in_tile;            /* depth tiles seen so far in this tile    */
    long pe_sum[NUM_PES];         /* each PE's cycles summed over the tile   */
    long bytes_sum;               /* bytes summed over the tile              */
    long long run_ahead_cycles;   /* total over all images and output tiles  */
} Track;

static Track track[NUM_LAYERS * 2];
static int track_count = 0;

static int layer_depth(int l)
{
    const Layer *L = &LAYERS[l];
    return L->type == CONV ? 9 * L->in_channels : L->in_channels;
}

void run_ahead_record_batch(void)
{
    int idx = -1;
    for (int i = 0; i < track_count; i++)
        if (track[i].stats == current) {
            idx = i;
            break;
        }
    if (idx < 0) {
        idx = track_count++;
        track[idx].stats = current;
        int depth = layer_depth(idx / 2);
        track[idx].depth_tiles = (depth + NUM_PES - 1) / NUM_PES;
        track[idx].batch_in_tile = 0;
        memset(track[idx].pe_sum, 0, sizeof(track[idx].pe_sum));
        track[idx].bytes_sum = 0;
        track[idx].run_ahead_cycles = 0;
    }
    if (idx % 2 != INNER)
        return; /* only the inner product runs ahead this way */

    for (int pe = 0; pe < NUM_PES; pe++)
        track[idx].pe_sum[pe] += pe_cycles[pe];
    track[idx].bytes_sum += batch_bytes;
    track[idx].batch_in_tile++;

    if (track[idx].batch_in_tile == track[idx].depth_tiles) {
        long compute = 0;
        for (int pe = 0; pe < NUM_PES; pe++)
            if (track[idx].pe_sum[pe] > compute)
                compute = track[idx].pe_sum[pe];
        long memory = (track[idx].bytes_sum + BYTES_PER_CYCLE - 1) / BYTES_PER_CYCLE;
        track[idx].run_ahead_cycles += compute > memory ? compute : memory;
        track[idx].batch_in_tile = 0;
        memset(track[idx].pe_sum, 0, sizeof(track[idx].pe_sum));
        track[idx].bytes_sum = 0;
    }
}

static void run_sparten_quiet(void)
{
    fflush(stdout);
    int saved_fd = dup(STDOUT_FILENO);
    FILE *null = freopen("/dev/null", "w", stdout);
    (void)null;
    sparten_main();
    fflush(stdout);
    dup2(saved_fd, STDOUT_FILENO);
    close(saved_fd);
    clearerr(stdout);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s csv_path txt_path\n", argv[0]);
        return 1;
    }
    const char *csv_path = argv[1], *txt_path = argv[2];

    run_sparten_quiet();

    FILE *csv = fopen(csv_path, "w");
    FILE *txt = fopen(txt_path, "w");
    if (!csv || !txt) {
        fprintf(stderr, "cannot open output files\n");
        return 1;
    }

    fprintf(csv, "layer,depth,depth_tiles,baseline_cycles,run_ahead_cycles,"
                 "perfect_balance_bound,memory_floor,compute_peak,speedup\n");
    fprintf(txt, "run-ahead estimate: one barrier per output tile instead of per batch\n");
    fprintf(txt, "inner product, tile %d x %d, %d bytes/cycle, %d images\n\n", TILE_ROWS,
            TILE_COLS, BYTES_PER_CYCLE, NUM_IMAGES);
    fprintf(txt, "%-6s %6s %5s | %12s %12s %12s %12s %12s %9s\n", "layer", "depth", "D",
            "baseline", "run_ahead", "perfect_bal", "mem_floor", "compute_pk", "speedup");

    double tot_baseline = 0, tot_run_ahead = 0, tot_balance = 0, tot_memory = 0, tot_compute = 0;
    for (int l = 0; l < NUM_LAYERS; l++) {
        int idx = 2 * l + INNER;
        const Stats *s = track[idx].stats;
        int depth = layer_depth(l);
        int D = track[idx].depth_tiles;

        double baseline = (double)s->cycles / NUM_IMAGES;
        double run_ahead = (double)track[idx].run_ahead_cycles / NUM_IMAGES;
        double balance = (double)s->busy_pe_cycles / 16.0 / NUM_IMAGES;
        double memory = (double)s->bytes / 16.0 / NUM_IMAGES;
        double compute = (double)s->macs / 16.0 / NUM_IMAGES;
        double speedup = run_ahead > 0 ? baseline / run_ahead : 0;

        tot_baseline += baseline;
        tot_run_ahead += run_ahead;
        tot_balance += balance;
        tot_memory += memory;
        tot_compute += compute;

        fprintf(csv, "%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", LAYERS[l].name, depth, D,
                baseline, run_ahead, balance, memory, compute, speedup);
        fprintf(txt, "%-6s %6d %5d | %12.1f %12.1f %12.1f %12.1f %12.1f %8.2fx\n",
                LAYERS[l].name, depth, D, baseline, run_ahead, balance, memory, compute, speedup);
    }
    double total_speedup = tot_run_ahead > 0 ? tot_baseline / tot_run_ahead : 0;
    fprintf(csv, "total,,,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", tot_baseline, tot_run_ahead,
            tot_balance, tot_memory, tot_compute, total_speedup);
    fprintf(txt, "%-6s %6s %5s | %12.1f %12.1f %12.1f %12.1f %12.1f %8.2fx\n", "total", "", "",
            tot_baseline, tot_run_ahead, tot_balance, tot_memory, tot_compute, total_speedup);
    fprintf(txt, "\n");

    fclose(csv);
    fclose(txt);
    printf("wrote %s and %s\n", csv_path, txt_path);
    return 0;
}
