/* breakdown.c - split the gap between ideal cycles (useful MACs / 16) and
 * measured cycles into causes, per layer and loop order.
 *
 * Built by analysis/breakdown.sh against a patched copy of sparten.c (see
 * that script) where end_batch() first calls breakdown_record_batch() and
 * then the given end_batch_base(). sparten.c itself is never changed.
 *
 *   fill_drain     pipeline fill/drain cycles each PE pays per chunk
 *   barrier_wait   busy PEs waiting for the slowest busy PE in their batch
 *   idle_no_task   PEs with no task at all in a (partial) batch
 *   accumulator    extra stall when the busiest accumulator (outer product)
 *                  is the batch's bottleneck, not the slowest PE
 *   memory_stall   cycles where moving bytes takes longer than computing
 *
 * ideal + fill_drain + barrier_wait + idle_no_task + accumulator +
 * memory_stall must equal cycles for every row (checked below).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Filled in by breakdown_record_batch(), declared before sparten_main() is
 * defined so the patched end_batch() can call it. */
void breakdown_record_batch(void);

#define main sparten_main
#include "sparten_patched.c"
#undef main

/* ======================================================================
 * Per-(layer, order) accumulators, in PE-cycles except memory_stall, which
 * is in cycles. Entries are keyed by the Stats pointer `current`, in the
 * first-seen order: conv1 inner, conv1 outer, conv2 inner, conv2 outer, ...
 * ====================================================================== */

typedef struct {
    const Stats *stats;
    long long no_task;
    long long accumulator_extra;
    long long barrier_wait;
    long long memory_stall;
} Breakdown;

static Breakdown breakdown[NUM_LAYERS * 2];
static int breakdown_count = 0;

void breakdown_record_batch(void)
{
    int idx = -1;
    for (int i = 0; i < breakdown_count; i++)
        if (breakdown[i].stats == current) {
            idx = i;
            break;
        }
    if (idx < 0) {
        idx = breakdown_count++;
        breakdown[idx].stats = current;
    }

    int tasks = 0, slowest = 0, busiest = 0;
    for (int pe = 0; pe < NUM_PES; pe++) {
        if (pe_cycles[pe] > 0)
            tasks++;
        if (pe_cycles[pe] > slowest)
            slowest = pe_cycles[pe];
        if (accumulator_load[pe] > busiest)
            busiest = accumulator_load[pe];
    }
    long compute = slowest > busiest ? slowest : busiest;
    long memory = (batch_bytes + BYTES_PER_CYCLE - 1) / BYTES_PER_CYCLE;
    long batch = compute > memory ? compute : memory;

    long long barrier_wait = 0;
    for (int pe = 0; pe < NUM_PES; pe++)
        if (pe_cycles[pe] > 0)
            barrier_wait += slowest - pe_cycles[pe];

    breakdown[idx].no_task += (long long)(NUM_PES - tasks) * compute;
    breakdown[idx].accumulator_extra += (long long)NUM_PES * (compute - slowest);
    breakdown[idx].barrier_wait += barrier_wait;
    breakdown[idx].memory_stall += batch - compute;
}

/* ======================================================================
 * Driver: run sparten_main() with its own stdout suppressed, then print the
 * breakdown as CSV and as an aligned text table.
 * ====================================================================== */

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
    if (argc != 5) {
        fprintf(stderr, "usage: %s config csv_path txt_path header(0/1)\n", argv[0]);
        return 1;
    }
    const char *config = argv[1], *csv_path = argv[2], *txt_path = argv[3];
    int header = atoi(argv[4]);

    run_sparten_quiet();

    FILE *csv = fopen(csv_path, "a");
    FILE *txt = fopen(txt_path, "a");
    if (!csv || !txt) {
        fprintf(stderr, "cannot open output files\n");
        return 1;
    }
    if (header)
        fprintf(csv, "config,layer,order,batches,macs,ideal,fill_drain,barrier_wait,"
                     "idle_no_task,accumulator,memory_stall,cycles,balanced_bound,"
                     "memory_bound_batches\n");

    fprintf(txt, "config %s (tile %d x %d, %d bytes/cycle, %d images)\n", config, TILE_ROWS,
            TILE_COLS, BYTES_PER_CYCLE, NUM_IMAGES);
    fprintf(txt, "%-6s %-5s %8s %10s %10s %10s %12s %12s %11s %12s %10s %14s %20s\n", "layer",
            "order", "batches", "macs", "ideal", "fill_drain", "barrier_wait", "idle_no_task",
            "accumulator", "memory_stall", "cycles", "balanced_bnd", "mem_bound_batches");

    int fail = 0;
    for (int i = 0; i < breakdown_count; i++) {
        int layer = i / 2, order = i % 2;
        const Stats *s = breakdown[i].stats;

        long batches = s->batches / NUM_IMAGES;
        long macs = s->macs / NUM_IMAGES;
        long memory_bound_batches = s->memory_bound_batches / NUM_IMAGES;

        double ideal = (double)s->macs / 16.0 / NUM_IMAGES;
        double fill_drain = (double)(s->busy_pe_cycles - s->macs) / 16.0 / NUM_IMAGES;
        double barrier_wait = (double)breakdown[i].barrier_wait / 16.0 / NUM_IMAGES;
        double idle_no_task = (double)breakdown[i].no_task / 16.0 / NUM_IMAGES;
        double accumulator = (double)breakdown[i].accumulator_extra / 16.0 / NUM_IMAGES;
        double memory_stall = (double)breakdown[i].memory_stall / NUM_IMAGES;
        double cycles = (double)s->cycles / NUM_IMAGES;
        double balanced_bound = (double)s->busy_pe_cycles / 16.0 / NUM_IMAGES;

        double sum = ideal + fill_drain + barrier_wait + idle_no_task + accumulator + memory_stall;
        if (fabs(sum - cycles) > 1e-6 * (cycles > 1 ? cycles : 1)) {
            fprintf(stderr, "FAIL: %s %s %s: ideal+fill_drain+barrier_wait+idle_no_task+"
                           "accumulator+memory_stall = %.6f != cycles %.6f\n",
                    config, LAYERS[layer].name, ORDER_NAME[order], sum, cycles);
            fail = 1;
        }

        fprintf(csv, "%s,%s,%s,%ld,%ld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%ld\n", config,
                LAYERS[layer].name, ORDER_NAME[order], batches, macs, ideal, fill_drain,
                barrier_wait, idle_no_task, accumulator, memory_stall, cycles, balanced_bound,
                memory_bound_batches);
        fprintf(txt,
                "%-6s %-5s %8ld %10ld %10.1f %12.1f %12.1f %11.1f %12.1f %10.1f %13.1f %14.1f"
                " %20ld\n",
                LAYERS[layer].name, ORDER_NAME[order], batches, macs, ideal, fill_drain,
                barrier_wait, idle_no_task, accumulator, memory_stall, cycles, balanced_bound,
                memory_bound_batches);
    }
    fprintf(txt, "\n");
    fclose(csv);
    fclose(txt);
    return fail;
}
