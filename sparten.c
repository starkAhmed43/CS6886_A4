/*
 * sparten.c - run a pruned INT8 MNIST network on a simulated SparTen-style
 * sparse accelerator with 16 PEs.
 *
 *   gcc -O2 -Wall sparten.c -o sparten -lm && ./sparten
 *
 * Every layer is lowered to a matrix multiplication (GEMM)
 *     C[rows][cols] = A[rows][depth] x B[depth][cols]
 * which is cut into tiles of TILE_ROWS x TILE_DEPTH x TILE_COLS. Each tile runs
 * as one batch of up to 16 PE tasks; the next batch starts only when all 16 PEs
 * are done (no load balancing).
 *
 * Each layer is run twice, once with the inner product (TODO 4) and once with
 * the outer product (TODO 5), and both results are checked. Only the inner
 * product's result is passed on to the next layer.
 *
 * Settings can be changed with -D, e.g. -DTILE_ROWS=2 -DTILE_COLS=8 -DNUM_IMAGES=10.
 * Search for "TODO" to find your work.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mnist_images.h"
#include "mnist_model.h"
#include "pe.h"

/* ======================================================================
 * Settings
 * ====================================================================== */

#define NUM_PES 16
#define TILE_DEPTH PE_VECTOR_LENGTH     /* the depth of a tile is the PE's vector length: 16 */
#ifndef TILE_ROWS
#define TILE_ROWS 4
#endif
#ifndef TILE_COLS
#define TILE_COLS 4
#endif
#ifndef NUM_IMAGES
#define NUM_IMAGES NUM_TEST_IMAGES
#endif
#define CLOCK_MHZ 800

/* Memory: the PEs read their operands from on-chip SRAM through a 128-bit
 * port, so 16 bytes per cycle by default. */
#ifndef BYTES_PER_CYCLE
#define BYTES_PER_CYCLE 16
#endif
#define PSUM_BYTES 4                        /* one partial sum (int32)                  */
#define PEAK_OPS_PER_CYCLE (2 * NUM_PES)    /* 16 MACs per cycle; a MAC is 2 operations */

/* Inner product: one PE per output of the tile. Outer product: one PE per
 * depth index of the tile. */
_Static_assert(TILE_ROWS * TILE_COLS == NUM_PES, "TILE_ROWS x TILE_COLS must equal 16");
_Static_assert(TILE_DEPTH == NUM_PES, "the outer product needs one PE per depth index");

/* ======================================================================
 * Data types
 * ====================================================================== */

/* A matrix of ints, stored row by row. */
typedef struct {
    int rows, cols;
    int *data;
} Matrix;
/* Element (r, c) of the matrix that m points to. */
#define AT(m, r, c) ((m)->data[(r) * (m)->cols + (c)])

/* A feature map, stored channel-last: data[y][x][channel]. */
typedef struct {
    int height, width, channels;
    int *data;
} Tensor;
/* The value at row y, column x, channel c of the tensor that t points to. */
#define PIXEL(t, y, x, c) ((t)->data[((y) * (t)->width + (x)) * (t)->channels + (c)])

enum { CONV, FC };
typedef struct {
    const char *name;
    int type;                    /* CONV: 3x3, stride 1, padding 1, then 2x2 max-pool */
    int in_channels, out_channels;   /* FC: the number of inputs and outputs      */
    const signed char *weights;  /* CONV: [out][3][3][in]   FC: [out][in]          */
    const int *bias;
    int mult, shift;             /* requantization; shift 0 = last layer (no ReLU) */
} Layer;

static const Layer LAYERS[] = {
    {"conv1", CONV, 1, 8, conv1_weights, conv1_bias, CONV1_MULT, CONV1_SHIFT},
    {"conv2", CONV, 8, 16, conv2_weights, conv2_bias, CONV2_MULT, CONV2_SHIFT},
    {"fc1", FC, 784, 32, fc1_weights, fc1_bias, FC1_MULT, FC1_SHIFT},
    {"fc2", FC, 32, 10, fc2_weights, fc2_bias, FC2_MULT, FC2_SHIFT},
};
#define NUM_LAYERS 4

enum { INNER, OUTER };           /* the two loop orders */
static const char *ORDER_NAME[2] = {"inner", "outer"};

/* Totals for one layer and loop order, summed over all images. A useful MAC is
 * one whose two operands are both non-zero. */
typedef struct {
    long dense_macs;           /* rows x depth x cols: the MACs without sparsity        */
    long macs;                 /* useful MACs                                           */
    long batches;
    long cycles;               /* sum over batches of max(compute, memory cycles)       */
    long compute_cycles;       /* sum over batches of the compute cycles                */
    long busy_pe_cycles;       /* sum over batches and PEs of the cycles each PE worked */
    long bytes;                /* bytes moved between SRAM and the PEs                  */
    long memory_bound_batches; /* batches whose memory cycles exceed their compute cycles */
    double energy_pj;          /* energy in picojoules (pJ)                             */
} Stats;

/* A rows x cols matrix of zeros. */
Matrix new_matrix(int rows, int cols)
{
    Matrix m = {rows, cols, calloc((size_t)rows * cols, sizeof(int))};
    return m;
}

/* ======================================================================
 * Given: batches of up to 16 PE tasks
 *
 * A loop order runs each batch as
 *     start_batch();
 *     ... pe_task_inner() or pe_task_outer() for each PE that has a task,
 *     ... count_bytes() for the bytes the batch moves (TODO 7),
 *     end_batch();
 * ====================================================================== */

static Stats *current;                 /* the Stats of the GEMM being run            */
static int pe_cycles[NUM_PES];         /* cycles each PE spends in this batch        */
static int accumulator_load[NUM_PES];  /* outer product: values each of the 16
                                          accumulators (one per output) adds        */
static long batch_bytes;               /* bytes moved in this batch                  */

/* Start a new batch. */
void start_batch(void)
{
    memset(pe_cycles, 0, sizeof(pe_cycles));
    memset(accumulator_load, 0, sizeof(accumulator_load));
    batch_bytes = 0;
}

/* Record n more bytes moved between SRAM and the PEs in the current batch.
 * The calls add up, so you can call it once per piece or once with the sum. */
void count_bytes(long n)
{
    batch_bytes += n;
}

/* Give PE `pe` an inner-product task: a row piece of the A block and a column
 * piece of the B block. Returns their dot product. */
int pe_task_inner(int pe, const SparseVector *row_piece, const SparseVector *column_piece)
{
    PeResult result = pe_inner(row_piece, column_piece);
    pe_cycles[pe] += result.cycles;
    current->macs += result.macs;
    current->energy_pj += result.energy_pj;
    return result.sum;
}

/* Give PE `pe` an outer-product task: a column piece of the A block
 * (TILE_ROWS values) and a row piece of the B block (TILE_COLS values). Their
 * products are added into the tile's partial sums. */
void pe_task_outer(int pe, const SparseVector *column_piece, const SparseVector *row_piece,
                   int partial_sums[TILE_ROWS][TILE_COLS])
{
    PeResult result = pe_outer(column_piece, row_piece, &partial_sums[0][0], TILE_COLS,
                               accumulator_load);
    pe_cycles[pe] += result.cycles;
    current->macs += result.macs;
    current->energy_pj += result.energy_pj;
}

/* Finish the batch and add its cycles and bytes to the current Stats.
 *   compute cycles: all PEs wait for the slowest one (no load balancing). In
 *                   the outer product, each of the 16 accumulators adds only
 *                   one value per cycle, so the batch also cannot finish before
 *                   its busiest accumulator.
 *   memory cycles:  the bytes moved divided by BYTES_PER_CYCLE, rounded up.
 * The operands are transferred while the PEs compute, so the two overlap and
 * the batch takes whichever is larger. */
void end_batch(void)
{
    int slowest_pe = 0, busiest_accumulator = 0;
    for (int pe = 0; pe < NUM_PES; pe++) {
        if (pe_cycles[pe] > slowest_pe)
            slowest_pe = pe_cycles[pe];
        if (accumulator_load[pe] > busiest_accumulator)
            busiest_accumulator = accumulator_load[pe];
        current->busy_pe_cycles += pe_cycles[pe];
    }
    long compute_cycles = slowest_pe > busiest_accumulator ? slowest_pe : busiest_accumulator;
    long memory_cycles = (batch_bytes + BYTES_PER_CYCLE - 1) / BYTES_PER_CYCLE;
    current->compute_cycles += compute_cycles;
    current->cycles += memory_cycles > compute_cycles ? memory_cycles : compute_cycles;
    current->memory_bound_batches += memory_cycles > compute_cycles;
    current->bytes += batch_bytes;
    current->batches++;
}

/* ======================================================================
 * TODO 1-2: lowering layers to GEMM
 * ====================================================================== */

/* TODO 1 - lower_fc: a fully-connected layer as a GEMM.
 *   - A has a single row, with one column per input value. B has one row per
 *     input value and one column per output. Element (d, o) of B is the weight
 *     that connects input d to output o.
 *   - The input tensor is stored channel-last; flatten it in that same order.
 *   - The weights are stored output by output: all the weights of output 0,
 *     then all the weights of output 1, and so on.
 *   - Checked by: A x B must equal the fully-connected layer computed directly.
 * A and B are already allocated and filled with zeros. AT(A, r, c) is element
 * (r, c) of A. */
void lower_fc(const Layer *L, const Tensor *in, Matrix *A, Matrix *B)
{
    *A = new_matrix(1, L->in_channels);
    *B = new_matrix(L->in_channels, L->out_channels);
    for (int d = 0; d < L->in_channels; d++) {
        AT(A, 0, d) = in->data[d];
        for (int o = 0; o < L->out_channels; o++)
            AT(B, d, o) = L->weights[o * L->in_channels + d];
    }
}

/* TODO 2 - lower_conv: a 3x3 convolution as a GEMM (im2col).
 *   - A has one row per output position, taken row by row (stride 1 and
 *     padding 1, so the output has the same height and width as the input),
 *     and one column per element of the 3x3 window across all input channels.
 *   - B has one row per window element and one column per filter.
 *   - Order the window elements the way the filter's weights are stored: by
 *     filter row, then filter column, and then channel.
 *   - Window elements that fall outside the image are set to 0 (padding).
 *   - Checked by: A x B must equal the convolution computed directly.
 * A and B are already allocated and filled with zeros. PIXEL(in, y, x, c) is
 * the input value at row y, column x, channel c. */
void lower_conv(const Layer *L, const Tensor *in, Matrix *A, Matrix *B)
{
    int depth = 9 * L->in_channels;
    *A = new_matrix(in->height * in->width, depth);
    *B = new_matrix(depth, L->out_channels);
    for (int d = 0; d < depth; d++)
        for (int k = 0; k < L->out_channels; k++)
            AT(B, d, k) = L->weights[k * depth + d];
    for (int y = 0; y < in->height; y++)
        for (int x = 0; x < in->width; x++) {
            int p = y * in->width + x;
            for (int filter_row = 0; filter_row < 3; filter_row++)
                for (int filter_col = 0; filter_col < 3; filter_col++) {
                    int in_y = y + filter_row - 1, in_x = x + filter_col - 1;
                    if (in_y < 0 || in_y >= in->height || in_x < 0 || in_x >= in->width)
                        continue;
                    for (int c = 0; c < L->in_channels; c++) {
                        int d = (filter_row * 3 + filter_col) * L->in_channels + c;
                        AT(A, p, d) = PIXEL(in, in_y, in_x, c);
                    }
                }
        }
}

/* ======================================================================
 * TODO 3: tiling
 * ====================================================================== */

/* TODO 3 - load_tile: cut one tile out of A and B.
 *   - Copy into a the TILE_ROWS x 16 block of A that starts at row row0,
 *     column depth0, and into b the 16 x TILE_COLS block of B that starts at
 *     row depth0, column col0.
 *   - Any part of a block that falls outside its matrix is set to 0, since the
 *     last tile in each dimension can be partial.
 *   - Checked by: the results of TODOs 4 and 5.
 * a and b have already been filled with zeros. */
void load_tile(const Matrix *A, const Matrix *B, int row0, int depth0, int col0,
               int a[TILE_ROWS][TILE_DEPTH], int b[TILE_DEPTH][TILE_COLS])
{
    memset(a, 0, sizeof(int) * TILE_ROWS * TILE_DEPTH);
    memset(b, 0, sizeof(int) * TILE_DEPTH * TILE_COLS);
    for (int r = 0; r < TILE_ROWS; r++)
        for (int d = 0; d < TILE_DEPTH; d++)
            if (row0 + r < A->rows && depth0 + d < A->cols)
                a[r][d] = AT(A, row0 + r, depth0 + d);
    for (int d = 0; d < TILE_DEPTH; d++)
        for (int c = 0; c < TILE_COLS; c++)
            if (depth0 + d < B->rows && col0 + c < B->cols)
                b[d][c] = AT(B, depth0 + d, col0 + c);
}

/* ======================================================================
 * TODO 4-5: loop orders (TODO 7: byte counting inside both)
 * ====================================================================== */

/* TODO 4 - gemm_inner: the inner-product loop order.
 *   - Every output tile of C is computed by going through its depth tiles in
 *     order. Each depth tile is one batch, between start_batch() and
 *     end_batch().
 *   - Within a batch, each of the 16 PEs computes one output of the tile: PE
 *     (r, c) gets that output's row piece of the A block and column piece of
 *     the B block (16 values each, stored with encode()) and adds the result of
 *     pe_task_inner() to that output's running sum.
 *   - Every PE whose output lies inside C gets a task, even if its pieces are
 *     all zeros. A PE whose output lies outside C gets no task.
 *   - When the last depth tile is done, write the tile's outputs into C.
 *   - Checked by: C must equal a plain matrix multiply.
 * Each PE owns one output and keeps its partial sum while the depth tiles
 * stream past: the output-stationary dataflow that SparTen uses.
 *
 * TODO 7 (once TODOs 4 and 5 give correct results): before each end_batch(),
 * count the bytes the batch moves with count_bytes().
 *   - Each row piece of A and each column piece of B is read once and
 *     broadcast to all the PEs that use it. Count each piece once, not once per
 *     PE, and count only the pieces whose row or column lies inside C. Use
 *     vector_bytes() for a stored vector (a vector with no non-zeros still
 *     costs the 2 bytes of its SparseMap).
 *   - In the last depth tile, also count PSUM_BYTES for each finished output
 *     written to C. */
void gemm_inner(const Matrix *A, const Matrix *B, Matrix *C)
{
#ifdef E4_TRACE
    static int traced = 0;
#endif
    for (int row0 = 0; row0 < A->rows; row0 += TILE_ROWS)
        for (int col0 = 0; col0 < B->cols; col0 += TILE_COLS) {
            int sums[TILE_ROWS][TILE_COLS];
            memset(sums, 0, sizeof(sums));
            for (int depth0 = 0; depth0 < A->cols; depth0 += TILE_DEPTH) {
                int a[TILE_ROWS][TILE_DEPTH], b[TILE_DEPTH][TILE_COLS];
                load_tile(A, B, row0, depth0, col0, a, b);

                SparseVector rows[TILE_ROWS], cols[TILE_COLS];
                for (int r = 0; r < TILE_ROWS; r++)
                    encode(a[r], TILE_DEPTH, &rows[r]);
                for (int c = 0; c < TILE_COLS; c++) {
                    int column[TILE_DEPTH];
                    for (int d = 0; d < TILE_DEPTH; d++)
                        column[d] = b[d][c];
                    encode(column, TILE_DEPTH, &cols[c]);
                }

#ifdef E4_TRACE
                int e4_trace = !traced && row0 == 0 && col0 == 0 && depth0 == 0;
                if (e4_trace) {
                    for (int r = 0; r < TILE_ROWS; r++)
                        printf("E4 row %d mask=%04x count=%d\n", r, rows[r].mask, rows[r].count);
                    for (int c = 0; c < TILE_COLS; c++)
                        printf("E4 col %d mask=%04x count=%d\n", c, cols[c].mask, cols[c].count);
                    for (int r = 0; r < TILE_ROWS; r++)
                        for (int c = 0; c < TILE_COLS; c++)
                            if (row0 + r < A->rows && col0 + c < B->cols)
                                printf("E4 PE %d,%d matches %d\n", r, c,
                                       __builtin_popcount(rows[r].mask & cols[c].mask));
                }
#endif
                start_batch();
                for (int r = 0; r < TILE_ROWS; r++)
                    for (int c = 0; c < TILE_COLS; c++)
                        if (row0 + r < A->rows && col0 + c < B->cols)
                            sums[r][c] += pe_task_inner(r * TILE_COLS + c, &rows[r], &cols[c]);
                long bytes = 0;
                for (int r = 0; r < TILE_ROWS; r++)
                    if (row0 + r < A->rows)
                        bytes += vector_bytes(&rows[r]);
                for (int c = 0; c < TILE_COLS; c++)
                    if (col0 + c < B->cols)
                        bytes += vector_bytes(&cols[c]);
                if (depth0 + TILE_DEPTH >= A->cols)
                    for (int r = 0; r < TILE_ROWS; r++)
                        for (int c = 0; c < TILE_COLS; c++)
                            if (row0 + r < A->rows && col0 + c < B->cols)
                                bytes += PSUM_BYTES;
                count_bytes(bytes);
                end_batch();
#ifdef E4_TRACE
                if (e4_trace) {
                    int slowest_pe = 0;
                    for (int pe = 0; pe < NUM_PES; pe++)
                        if (pe_cycles[pe] > slowest_pe)
                            slowest_pe = pe_cycles[pe];
                    long memory_cycles = (batch_bytes + BYTES_PER_CYCLE - 1) / BYTES_PER_CYCLE;
                    printf("E4 slowest_pe_cycles=%d batch_bytes=%ld memory_cycles=%ld\n",
                           slowest_pe, batch_bytes, memory_cycles);
                    traced = 1;
                }
#endif
            }
            for (int r = 0; r < TILE_ROWS; r++)
                for (int c = 0; c < TILE_COLS; c++)
                    if (row0 + r < A->rows && col0 + c < B->cols)
                        AT(C, row0 + r, col0 + c) = sums[r][c];
        }
}

/* TODO 5 - gemm_outer: the outer-product loop order.
 *   - The depth tile is the outermost loop: for each depth tile, every output
 *     tile of C is one batch, between start_batch() and end_batch().
 *   - Within a batch, each PE handles one depth index k of the tile: PE k gets
 *     column k of the A block (TILE_ROWS values) and row k of the B block
 *     (TILE_COLS values), stored with encode(), and calls pe_task_outer(),
 *     which adds its products into the tile's partial sums.
 *   - Every PE whose depth index lies inside the depth of A gets a task, even
 *     if its pieces are all zeros. A PE whose depth index lies past it gets no
 *     task.
 *   - After each batch, add the tile's partial sums into C, which starts at
 *     zero.
 *   - Checked by: C must equal a plain matrix multiply.
 * This is the Cartesian-product dataflow of SCNN: no matching is needed, but
 * the partial sums have to be merged across PEs and across depth tiles.
 *
 * TODO 7 (once TODOs 4 and 5 give correct results): before each end_batch(),
 * count the bytes the batch moves with count_bytes().
 *   - Every PE reads its own column piece and row piece, so count both for
 *     every PE that has a task. Use vector_bytes() for a stored vector.
 *   - After every depth tile, count PSUM_BYTES written for each output of the
 *     tile that lies inside C. From the second depth tile on, also count
 *     PSUM_BYTES read for each of those outputs. */
void gemm_outer(const Matrix *A, const Matrix *B, Matrix *C)
{
    for (int depth0 = 0; depth0 < A->cols; depth0 += TILE_DEPTH)
        for (int row0 = 0; row0 < A->rows; row0 += TILE_ROWS)
            for (int col0 = 0; col0 < B->cols; col0 += TILE_COLS) {
                int a[TILE_ROWS][TILE_DEPTH], b[TILE_DEPTH][TILE_COLS];
                load_tile(A, B, row0, depth0, col0, a, b);

                SparseVector a_cols[TILE_DEPTH], b_rows[TILE_DEPTH];
                for (int k = 0; k < TILE_DEPTH; k++) {
                    int column[TILE_ROWS];
                    for (int r = 0; r < TILE_ROWS; r++)
                        column[r] = a[r][k];
                    encode(column, TILE_ROWS, &a_cols[k]);
                    encode(b[k], TILE_COLS, &b_rows[k]);
                }

                int sums[TILE_ROWS][TILE_COLS];
                memset(sums, 0, sizeof(sums));

                start_batch();
                long bytes = 0;
                for (int k = 0; k < TILE_DEPTH; k++)
                    if (depth0 + k < A->cols) {
                        pe_task_outer(k, &a_cols[k], &b_rows[k], sums);
                        bytes += vector_bytes(&a_cols[k]) + vector_bytes(&b_rows[k]);
                    }
                for (int r = 0; r < TILE_ROWS; r++)
                    for (int c = 0; c < TILE_COLS; c++)
                        if (row0 + r < A->rows && col0 + c < B->cols) {
                            bytes += PSUM_BYTES;
                            if (depth0 > 0)
                                bytes += PSUM_BYTES;
                        }
                count_bytes(bytes);
                end_batch();

                for (int r = 0; r < TILE_ROWS; r++)
                    for (int c = 0; c < TILE_COLS; c++)
                        if (row0 + r < A->rows && col0 + c < B->cols)
                            AT(C, row0 + r, col0 + c) += sums[r][c];
            }
}

/* ======================================================================
 * TODO 6: profiling metrics
 * ====================================================================== */

typedef struct {
    double cycles, time_us, utilization, macs_per_cycle, energy_nj, power_mw;
} Metrics;

/* TODO 6 - layer_metrics: profiling numbers per image.
 * The Stats hold totals over num_images images, and record energy in
 * picojoules (pJ). Every metric you return is per image, and energy is
 * returned in nanojoules (1 nJ = 1000 pJ).
 *   cycles          cycles per image
 *   time_us         time per image in microseconds, at CLOCK_MHZ
 *   utilization     the fraction of all PE-cycles (16 PEs x cycles) spent on a
 *                   useful MAC
 *   macs_per_cycle  useful MACs per cycle
 *   energy_nj       energy per image in nanojoules, converted from the
 *                   picojoules in the Stats
 *   power_mw        average power in milliwatts while the layer runs
 * Return all zeros if there are no cycles.
 * Checked by: the WARNING lines of the sanity check. */
Metrics layer_metrics(const Stats *s, int num_images)
{
    Metrics metrics = {0, 0, 0, 0, 0, 0};
    if (s->cycles == 0)
        return metrics;

    double cycles = (double)s->cycles / num_images;
    double time_us = cycles / CLOCK_MHZ;
    double energy_nj = s->energy_pj / 1000.0 / num_images;

    metrics.cycles = cycles;
    metrics.time_us = time_us;
    metrics.utilization = (double)s->macs / ((double)NUM_PES * s->cycles);
    metrics.macs_per_cycle = (double)s->macs / s->cycles;
    metrics.energy_nj = energy_nj;
    metrics.power_mw = energy_nj / time_us; /* nJ/us = mW */
    return metrics;
}

/* ======================================================================
 * TODO 8: the roofline
 * ====================================================================== */

typedef struct {
    double intensity;    /* operations per byte moved             */
    double achieved;     /* operations per cycle reached          */
    double attainable;   /* the lower of the two roofs            */
    int memory_bound;    /* 1 if the memory roof is the lower one */
} RooflinePoint;

/* TODO 8 - roofline_point: place a layer on the roofline.
 * Count a MAC as two operations (a multiply and an add). The Stats hold totals
 * over all images. The roofline has two roofs: the memory roof is the
 * performance that the bandwidth allows at a given intensity,
 * BYTES_PER_CYCLE x intensity; the compute roof is the peak performance,
 * PEAK_OPS_PER_CYCLE. Return:
 *   intensity     operations per byte moved
 *   achieved      operations per cycle reached
 *   attainable    the lower of the two roofs at this intensity
 *   memory_bound  1 when the memory roof is the lower one (the bandwidth is
 *                 the limit), otherwise 0
 * Return all zeros if there are no bytes or no cycles.
 * Checked by: the WARNING lines of the sanity check. */
RooflinePoint roofline_point(const Stats *s)
{
    RooflinePoint point = {0, 0, 0, 0};
    if (s->bytes == 0 || s->cycles == 0)
        return point;

    double ops = 2.0 * s->macs;
    double intensity = ops / s->bytes;
    double memory_roof = BYTES_PER_CYCLE * intensity;

    point.intensity = intensity;
    point.achieved = ops / s->cycles;
    point.attainable = memory_roof < PEAK_OPS_PER_CYCLE ? memory_roof : PEAK_OPS_PER_CYCLE;
    point.memory_bound = memory_roof < PEAK_OPS_PER_CYCLE;
    return point;
}

/* ======================================================================
 * Given: correctness checks, post-processing, report, and the network
 * ====================================================================== */

/* C = A x B, computed directly. Used to check gemm_inner and gemm_outer. */
void dense_gemm(const Matrix *A, const Matrix *B, Matrix *C)
{
    for (int r = 0; r < A->rows; r++)
        for (int c = 0; c < B->cols; c++) {
            int sum = 0;
            for (int d = 0; d < A->cols; d++)
                sum += AT(A, r, d) * AT(B, d, c);
            AT(C, r, c) = sum;
        }
}

/* The layer computed directly from the tensor, without lowering. Used to check
 * lower_fc and lower_conv. out[position][output] is laid out like C. */
void reference_layer(const Layer *L, const Tensor *in, Matrix *out)
{
    for (int p = 0; p < out->rows; p++)
        for (int k = 0; k < out->cols; k++) {
            int sum = 0;
            if (L->type == FC) {
                for (int d = 0; d < L->in_channels; d++)
                    sum += in->data[d] * L->weights[k * L->in_channels + d];
            } else {
                int y = p / in->width, x = p % in->width;
                for (int filter_row = 0; filter_row < 3; filter_row++)
                    for (int filter_col = 0; filter_col < 3; filter_col++) {
                        int in_y = y + filter_row - 1, in_x = x + filter_col - 1;
                        if (in_y < 0 || in_y >= in->height || in_x < 0 || in_x >= in->width)
                            continue;
                        for (int c = 0; c < L->in_channels; c++)
                            sum += PIXEL(in, in_y, in_x, c) *
                                   L->weights[((k * 3 + filter_row) * 3 + filter_col) *
                                              L->in_channels + c];
                    }
            }
            AT(out, p, k) = sum;
        }
}

/* 1 if the two matrices have the same shape and the same values. */
int same_matrix(const Matrix *first, const Matrix *second)
{
    return first->rows == second->rows && first->cols == second->cols &&
           memcmp(first->data, second->data, sizeof(int) * first->rows * first->cols) == 0;
}

/* Add the bias, requantize to INT8 with ReLU, and max-pool (CONV only). Returns
 * the next layer's input. The last layer (shift 0) returns its logits as they
 * are. */
Tensor post_process(const Layer *L, const Tensor *in, const Matrix *C)
{
    Tensor out = {in->height, in->width, L->out_channels, NULL};
    if (L->type == FC)
        out.height = out.width = 1;
    int n = C->rows * C->cols;
    int *requantized = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        long long acc = (long long)C->data[i] + L->bias[i % C->cols];
        if (L->shift == 0) {
            requantized[i] = (int)acc;
            continue;
        }
        long long v = (acc * L->mult + (1LL << (L->shift - 1))) >> L->shift;
        requantized[i] = v < 0 ? 0 : v > 127 ? 127 : (int)v;     /* ReLU, then clamp to INT8 */
    }
    if (L->type == FC) {
        out.data = requantized;
        return out;
    }
    out.height /= 2;                                             /* 2x2 max-pool */
    out.width /= 2;
    out.data = malloc(sizeof(int) * out.height * out.width * out.channels);
    for (int y = 0; y < out.height; y++)
        for (int x = 0; x < out.width; x++)
            for (int c = 0; c < out.channels; c++) {
                int best = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        int v = requantized[((2 * y + dy) * in->width + 2 * x + dx) * out.channels + c];
                        if (v > best)
                            best = v;
                    }
                PIXEL(&out, y, x, c) = best;
            }
    free(requantized);
    return out;
}

/* Draw the roofline and one point per layer and loop order as an SVG file
 * (open it in a web browser). Both axes are logarithmic, from 0.1 to 100. */
void write_roofline_svg(const char *path, const RooflinePoint points[NUM_LAYERS][2])
{
    FILE *svg = fopen(path, "w");
    if (!svg)
        return;
    const double W = 720, H = 480, left = 80, right = 30, top = 50, bottom = 60;
/* Pixel position of a value on the x (operational intensity) and y (performance) axis. */
#define PLOT_X(v) (left + (log10(v) + 1) / 3 * (W - left - right))
#define PLOT_Y(v) (H - bottom - (log10(v) + 1) / 3 * (H - top - bottom))
    fprintf(svg, "<svg xmlns='http://www.w3.org/2000/svg' width='%.0f' height='%.0f' "
                 "font-family='Helvetica, Arial, sans-serif' font-size='12'>\n"
                 "<rect width='100%%' height='100%%' fill='#ffffff'/>\n", W, H);
    fprintf(svg, "<text x='%.0f' y='28' font-size='15' fill='#14171c'>Roofline: %d PEs, "
                 "tile %d x %d x %d, %d bytes/cycle</text>\n",
            left, NUM_PES, TILE_ROWS, TILE_DEPTH, TILE_COLS, BYTES_PER_CYCLE);
    for (int e = -1; e <= 2; e++) {                  /* grid lines and tick labels */
        double v = pow(10, e);
        fprintf(svg, "<line x1='%.1f' y1='%.0f' x2='%.1f' y2='%.0f' stroke='#e4e6ea'/>\n",
                PLOT_X(v), top, PLOT_X(v), H - bottom);
        fprintf(svg, "<line x1='%.0f' y1='%.1f' x2='%.0f' y2='%.1f' stroke='#e4e6ea'/>\n",
                left, PLOT_Y(v), W - right, PLOT_Y(v));
        fprintf(svg, "<text x='%.1f' y='%.0f' text-anchor='middle' fill='#555c66'>%g</text>\n",
                PLOT_X(v), H - bottom + 18, v);
        fprintf(svg, "<text x='%.0f' y='%.1f' text-anchor='end' fill='#555c66'>%g</text>\n",
                left - 8, PLOT_Y(v) + 4, v);
    }
    fprintf(svg, "<text x='%.0f' y='%.0f' text-anchor='middle' fill='#14171c'>operational "
                 "intensity (ops per byte)</text>\n", (left + W - right) / 2, H - 18);
    fprintf(svg, "<text transform='translate(22 %.0f) rotate(-90)' text-anchor='middle' "
                 "fill='#14171c'>performance (ops per cycle)</text>\n", (top + H - bottom) / 2);
    double ridge = (double)PEAK_OPS_PER_CYCLE / BYTES_PER_CYCLE;
    fprintf(svg, "<polyline fill='none' stroke='#14171c' stroke-width='2' "
                 "points='%.1f,%.1f %.1f,%.1f %.1f,%.1f'/>\n",
            PLOT_X(0.1), PLOT_Y(BYTES_PER_CYCLE * 0.1), PLOT_X(ridge), PLOT_Y(PEAK_OPS_PER_CYCLE),
            PLOT_X(100), PLOT_Y(PEAK_OPS_PER_CYCLE));
    fprintf(svg, "<text x='%.1f' y='%.1f' fill='#555c66'>peak %d ops/cycle</text>\n",
            PLOT_X(ridge) + 8, PLOT_Y(PEAK_OPS_PER_CYCLE) - 8, PEAK_OPS_PER_CYCLE);
    fprintf(svg, "<text x='%.1f' y='%.1f' fill='#555c66'>ridge %.1f ops/byte</text>\n",
            PLOT_X(ridge) + 8, PLOT_Y(PEAK_OPS_PER_CYCLE) + 16, ridge);
    const char *color[2] = {"#2a78d6", "#eb6834"};
    for (int l = 0; l < NUM_LAYERS; l++)
        for (int order = INNER; order <= OUTER; order++) {
            const RooflinePoint *point = &points[l][order];
            if (point->intensity <= 0 || point->achieved <= 0)
                continue;
            double x = PLOT_X(point->intensity), y = PLOT_Y(point->achieved);
            if (order == INNER)                      /* inner product: circle */
                fprintf(svg, "<circle cx='%.1f' cy='%.1f' r='6' fill='%s'/>\n", x, y, color[order]);
            else                                     /* outer product: square */
                fprintf(svg, "<rect x='%.1f' y='%.1f' width='11' height='11' fill='%s'/>\n",
                        x - 5.5, y - 5.5, color[order]);
            /* inner-product labels to the right; outer-product labels to the left,
             * alternately above and below so close points stay readable */
            fprintf(svg, "<text x='%.1f' y='%.1f' text-anchor='%s' fill='#14171c'>%s</text>\n",
                    order == INNER ? x + 9 : x - 9, order == INNER ? y + 4 : y + (l % 2 ? -7 : 15),
                    order == INNER ? "start" : "end", LAYERS[l].name);
        }
    fprintf(svg, "<circle cx='%.0f' cy='%.0f' r='6' fill='%s'/><text x='%.0f' y='%.0f' "
                 "fill='#14171c'>inner product</text>\n", left + 20, top + 20, color[INNER],
            left + 32, top + 24);
    fprintf(svg, "<rect x='%.0f' y='%.0f' width='11' height='11' fill='%s'/><text x='%.0f' "
                 "y='%.0f' fill='#14171c'>outer product</text>\n", left + 14.5, top + 34.5,
            color[OUTER], left + 32, top + 44);
    fprintf(svg, "</svg>\n");
#undef PLOT_X
#undef PLOT_Y
    fclose(svg);
}

/* Print a WARNING line for every result of one layer and loop order that
 * cannot be right. Passing these checks does not mean a number is correct,
 * only that it is possible. Returns the number of warnings printed. */
int sanity_check(int l, int order, const Stats *s, int rows, int cols, int num_images)
{
    int warnings = 0;
#define WARN(...)                                                                          \
    do {                                                                                   \
        printf("WARNING %s %s: ", LAYERS[l].name, ORDER_NAME[order]);                      \
        printf(__VA_ARGS__);                                                               \
        printf("\n");                                                                      \
        warnings++;                                                                        \
    } while (0)
    if (s->macs == 0)
        return 0;                       /* nothing computed yet: nothing to check */
    Metrics metrics = layer_metrics(s, num_images);
    if (metrics.utilization <= 0 || metrics.utilization > 1)
        WARN("utilization %.3f must be above 0 and at most 1 (TODO 6)", metrics.utilization);
    if (metrics.macs_per_cycle <= 0 || metrics.macs_per_cycle > NUM_PES)
        WARN("%.2f MACs per cycle is impossible with %d PEs (TODO 6)", metrics.macs_per_cycle,
             NUM_PES);
    if (metrics.time_us <= 0 || metrics.energy_nj <= 0 || metrics.power_mw <= 0)
        WARN("time, energy and power must all be positive (TODO 6)");

    /* The fewest bytes: every output written once. The most bytes one batch can
     * move: every piece full of non-zeros, plus the partial sums of all 16
     * outputs (written once in the inner product; written and read in the
     * outer product). */
    long outputs = (long)rows * cols * num_images;
    long max_bytes_per_batch = order == INNER
        ? (TILE_ROWS + TILE_COLS) * (SPARSEMAP_BYTES + TILE_DEPTH) + NUM_PES * PSUM_BYTES
        : NUM_PES * (2 * SPARSEMAP_BYTES + TILE_ROWS + TILE_COLS) + 2 * NUM_PES * PSUM_BYTES;
    if (s->bytes < outputs * PSUM_BYTES)
        WARN("%ld bytes per image is less than writing each output once (TODO 7)",
             s->bytes / num_images);
    else if (s->bytes > s->batches * max_bytes_per_batch)
        WARN("%ld bytes per image is more than the batches can possibly move (TODO 7)",
             s->bytes / num_images);

    RooflinePoint point = roofline_point(s);
    double ridge = (double)PEAK_OPS_PER_CYCLE / BYTES_PER_CYCLE;
    if (point.attainable <= 0)
        WARN("the roofline point is empty (TODO 8)");
    else {
        if (point.achieved > point.attainable * 1.0001)
            WARN("achieved %.2f ops/cycle is above the roof %.2f (TODO 7 or 8)", point.achieved,
                 point.attainable);
        if (point.attainable > PEAK_OPS_PER_CYCLE * 1.0001)
            WARN("attainable %.2f is above the peak of %d ops/cycle (TODO 8)", point.attainable,
                 PEAK_OPS_PER_CYCLE);
        if ((point.intensity < ridge * 0.999 && !point.memory_bound) ||
            (point.intensity > ridge * 1.001 && point.memory_bound))
            WARN("memory_bound does not agree with the ridge point (TODO 8)");
    }
#undef WARN
    return warnings;
}

/* Add one layer's Stats into a running total. */
void add_stats(Stats *total, const Stats *s)
{
    total->dense_macs += s->dense_macs;
    total->macs += s->macs;
    total->batches += s->batches;
    total->cycles += s->cycles;
    total->compute_cycles += s->compute_cycles;
    total->busy_pe_cycles += s->busy_pe_cycles;
    total->bytes += s->bytes;
    total->memory_bound_batches += s->memory_bound_batches;
    total->energy_pj += s->energy_pj;
}

int main(void)
{
    static Stats stats[NUM_LAYERS][2];          /* per layer and loop order, over all images */
    static int gemm_rows[NUM_LAYERS], gemm_depth[NUM_LAYERS], gemm_cols[NUM_LAYERS];
    int correct = 0, golden_match = 0, check_failures = 0;

    /* ---------------------------------------------------- run the network */
    for (int img = 0; img < NUM_IMAGES; img++) {
        Tensor activation = {28, 28, 1, malloc(sizeof(int) * 28 * 28)};   /* this layer's input */
        for (int i = 0; i < 28 * 28; i++)
            activation.data[i] = test_images[img][i];

        for (int l = 0; l < NUM_LAYERS; l++) {
            const Layer *L = &LAYERS[l];
            Matrix A, B;
            if (L->type == CONV)
                lower_conv(L, &activation, &A, &B);
            else
                lower_fc(L, &activation, &A, &B);
            int rows = L->type == CONV ? activation.height * activation.width : 1;
            gemm_rows[l] = A.rows;
            gemm_depth[l] = A.cols;
            gemm_cols[l] = B.cols;

            /* CHECK the lowering (TODOs 1-2) against the layer computed directly. */
            Matrix reference = new_matrix(rows, L->out_channels);
            Matrix lowered = new_matrix(A.rows, B.cols);
            reference_layer(L, &activation, &reference);
            dense_gemm(&A, &B, &lowered);
            if (!same_matrix(&lowered, &reference) && img == 0) {
                printf("CHECK %s: the lowered GEMM does not compute the layer (TODO 1/2)\n",
                       L->name);
                check_failures++;
            }
            /* Run both loop orders and CHECK them (TODOs 3-5) against a plain GEMM. */
            Matrix C[2];
            for (int order = INNER; order <= OUTER; order++) {
                current = &stats[l][order];
                current->dense_macs += (long)A.rows * A.cols * B.cols;
                C[order] = new_matrix(A.rows, B.cols);
                if (order == INNER)
                    gemm_inner(&A, &B, &C[order]);
                else
                    gemm_outer(&A, &B, &C[order]);
                if (!same_matrix(&C[order], &lowered) && img == 0) {
                    printf("CHECK %s: %s product gives the wrong result (TODO 3/%d)\n", L->name,
                           ORDER_NAME[order], order == INNER ? 4 : 5);
                    check_failures++;
                }
            }
            /* Only the inner product's result is passed on to the next layer. */
            Tensor next = post_process(L, &activation, &C[INNER]);
            free(activation.data);
            activation = next;
            free(A.data); free(B.data); free(reference.data); free(lowered.data);
            free(C[INNER].data); free(C[OUTER].data);
        }
        int predicted = 0;                       /* the digit with the largest logit */
        for (int digit = 1; digit < 10; digit++)
            if (activation.data[digit] > activation.data[predicted])
                predicted = digit;
        correct += predicted == test_labels[img];
        golden_match += predicted == golden_predictions[img];
        free(activation.data);
    }

    /* ---------------------------------------------------- per-layer table */
    printf("SparTen MNIST: %d PEs, tile %d rows x %d depth x %d cols, %d MHz, %d images\n",
           NUM_PES, TILE_ROWS, TILE_DEPTH, TILE_COLS, CLOCK_MHZ, NUM_IMAGES);
    printf("accuracy %d/%d, matches golden predictions %d/%d, check failures %d\n\n", correct,
           NUM_IMAGES, golden_match, NUM_IMAGES, check_failures);
    printf("per image        GEMM rows x depth x cols  dense MACs  sparse MACs |"
           "        cycles   util  energy(nJ) power(mW)\n");
    static Stats totals[2];                      /* all layers, per loop order */
    for (int order = INNER; order <= OUTER; order++) {
        for (int l = 0; l <= NUM_LAYERS; l++) {  /* l == NUM_LAYERS: the total row */
            const Stats *s = l < NUM_LAYERS ? &stats[l][order] : &totals[order];
            if (l < NUM_LAYERS)
                add_stats(&totals[order], s);
            Metrics metrics = layer_metrics(s, NUM_IMAGES);
            char shape[32] = "";
            if (l < NUM_LAYERS)
                snprintf(shape, sizeof(shape), "%d x %d x %d", gemm_rows[l], gemm_depth[l],
                         gemm_cols[l]);
            printf("%-5s %-6s %24s %11ld %12ld | %13.0f %5.1f%% %11.1f %9.1f\n",
                   l < NUM_LAYERS ? LAYERS[l].name : "total", ORDER_NAME[order], shape,
                   s->dense_macs / NUM_IMAGES, s->macs / NUM_IMAGES, metrics.cycles,
                   100 * metrics.utilization, metrics.energy_nj, metrics.power_mw);
        }
        printf("\n");
    }

    /* ---------------------------------------------------- roofline: table, CSV, SVG */
    RooflinePoint points[NUM_LAYERS][2];
    printf("roofline: peak %d ops/cycle, %d bytes/cycle, ridge at %.1f ops/byte\n",
           PEAK_OPS_PER_CYCLE, BYTES_PER_CYCLE, (double)PEAK_OPS_PER_CYCLE / BYTES_PER_CYCLE);
    printf("layer order  bytes/image  ops/byte  achieved  attainable  %% of roof  "
           "memory-bound batches  bound\n");
    FILE *csv = fopen("roofline.csv", "w");
    if (csv)
        fprintf(csv, "layer,dataflow,tile_rows,tile_cols,ops_per_image,bytes_per_image,"
                     "cycles_per_image,intensity,achieved,attainable,bound\n");
    for (int order = INNER; order <= OUTER; order++)
        for (int l = 0; l <= NUM_LAYERS; l++) {  /* l == NUM_LAYERS: the total row */
            const Stats *s = l < NUM_LAYERS ? &stats[l][order] : &totals[order];
            RooflinePoint point = roofline_point(s);
            if (l < NUM_LAYERS)
                points[l][order] = point;
            const char *name = l < NUM_LAYERS ? LAYERS[l].name : "total";
            const char *bound = point.attainable == 0 ? "-" : point.memory_bound ? "memory" : "compute";
            printf("%-5s %-6s %11ld %9.2f %9.2f %11.2f %9.1f%% %15.1f%%      %s\n", name,
                   ORDER_NAME[order], s->bytes / NUM_IMAGES, point.intensity, point.achieved,
                   point.attainable, point.attainable ? 100 * point.achieved / point.attainable : 0,
                   s->batches ? 100.0 * s->memory_bound_batches / s->batches : 0, bound);
            if (csv)
                fprintf(csv, "%s,%s,%d,%d,%ld,%ld,%ld,%.4f,%.4f,%.4f,%s\n", name,
                        ORDER_NAME[order], TILE_ROWS, TILE_COLS, 2 * s->macs / NUM_IMAGES,
                        s->bytes / NUM_IMAGES, s->cycles / NUM_IMAGES, point.intensity,
                        point.achieved, point.attainable, bound);
        }
    if (csv)
        fclose(csv);
    write_roofline_svg("roofline.svg", points);
    printf("wrote roofline.csv and roofline.svg\n");

    /* ---------------------------------------------------- sanity check */
    int warnings = 0;
    for (int order = INNER; order <= OUTER; order++)
        for (int l = 0; l < NUM_LAYERS; l++)
            warnings += sanity_check(l, order, &stats[l][order], gemm_rows[l], gemm_cols[l],
                                     NUM_IMAGES);
    printf(warnings ? "sanity check: %d impossible value(s), see WARNING lines above\n"
                    : "sanity check: no impossible values found (this does not prove the "
                      "values are right)\n", warnings);
    return check_failures ? 1 : 0;
}
