/* fc_batching.c - report question 8: batch the fc1 and fc2 GEMMs across
 * several images so a tile's 16 PEs are not mostly idle on a single-row GEMM.
 *
 * fc1 (A is 1 x 784) and fc2 (A is 1 x 32) give gemm_inner/gemm_outer only one
 * row of work, so at the default 4 x 4 tile only 4 of the 16 PEs ever get a
 * task. Stacking N images' flattened inputs into an N x depth matrix lets a
 * single tile serve up to 4 images (TILE_ROWS = 4) and keeps every PE busy.
 *
 * This reuses the real lowering and GEMM code from sparten.c: `#define main
 * sparten_main` hides its main() so we can supply our own, then the TODO
 * implementations (lower_conv, lower_fc, gemm_inner, gemm_outer, post_process)
 * are called exactly as sparten's main() calls them to reproduce the forward
 * pass over the 100 test images. sparten.c itself is never modified.
 *
 *   gcc -O2 -Wall -Werror -I.. fc_batching.c -o fc_batching -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main sparten_main
#include "../sparten.c"
#undef main

#define NUM_N 5
static const int BATCH_N[NUM_N] = {1, 2, 4, 8, 16};

/* Captured per-image inputs to fc1 and fc2, gathered by replaying the forward
 * pass with gemm_inner (the path sparten's main() passes to the next layer). */
static int fc1_input[NUM_TEST_IMAGES][784];
static int fc2_input[NUM_TEST_IMAGES][32];
static Matrix fc1_B, fc2_B;   /* lower_fc's B: the same for every image */

/* Run one layer with gemm_inner and post_process, the way sparten's main()
 * does, and free the intermediates. Returns the next layer's activation. */
static Tensor run_layer_inner(const Layer *L, Tensor activation, Matrix *out_A, Matrix *out_B)
{
    Matrix A, B;
    if (L->type == CONV)
        lower_conv(L, &activation, &A, &B);
    else
        lower_fc(L, &activation, &A, &B);

    static Stats scratch;
    current = &scratch;
    Matrix C = new_matrix(A.rows, B.cols);
    gemm_inner(&A, &B, &C);

    Tensor next = post_process(L, &activation, &C);
    free(activation.data);
    free(C.data);
    if (out_A) *out_A = A; else free(A.data);
    if (out_B) *out_B = B; else free(B.data);
    return next;
}

/* Reproduce the forward pass for every image, capturing each image's fc1 and
 * fc2 input vectors (fc1's is the flattened 7x7x16 activation after conv2;
 * fc2's is fc1's 32-value output after post_process). */
static void capture_inputs(void)
{
    for (int img = 0; img < NUM_TEST_IMAGES; img++) {
        Tensor activation = {28, 28, 1, malloc(sizeof(int) * 28 * 28)};
        for (int i = 0; i < 28 * 28; i++)
            activation.data[i] = test_images[img][i];

        activation = run_layer_inner(&LAYERS[0], activation, NULL, NULL);   /* conv1 */
        activation = run_layer_inner(&LAYERS[1], activation, NULL, NULL);   /* conv2 */

        memcpy(fc1_input[img], activation.data, sizeof(int) * 784);

        Matrix A1, B1;
        Tensor fc1_out = run_layer_inner(&LAYERS[2], activation, &A1, &B1);  /* fc1 */
        if (img == 0)
            fc1_B = B1;
        else
            free(B1.data);
        free(A1.data);

        memcpy(fc2_input[img], fc1_out.data, sizeof(int) * 32);

        Matrix A2, B2;
        Tensor fc2_out = run_layer_inner(&LAYERS[3], fc1_out, &A2, &B2);     /* fc2 */
        if (img == 0)
            fc2_B = B2;
        else
            free(B2.data);
        free(A2.data);
        free(fc2_out.data);
    }
}

/* One row of the report: a layer, loop order and batch size N. */
typedef struct {
    const char *layer;
    const char *order;
    int n;
    double cycles_per_image, bytes_per_image, energy_nj_per_image;
    long batches, memory_bound_batches;
} Row;

/* Run gemm_inner or gemm_outer on images grouped N at a time, checking every
 * batched GEMM against dense_gemm, and fill in one report row. */
static Row run_batched(const char *layer_name, const char *order_name, int order, int n,
                       const int *inputs, int depth, const Matrix *B)
{
    Stats stats = {0};
    for (int start = 0; start < NUM_TEST_IMAGES; start += n) {
        int rows = n < NUM_TEST_IMAGES - start ? n : NUM_TEST_IMAGES - start;
        Matrix A = new_matrix(rows, depth);
        for (int r = 0; r < rows; r++)
            for (int d = 0; d < depth; d++)
                AT(&A, r, d) = inputs[(start + r) * depth + d];

        Matrix C = new_matrix(rows, B->cols), ref = new_matrix(rows, B->cols);
        current = &stats;
        if (order == INNER)
            gemm_inner(&A, B, &C);
        else
            gemm_outer(&A, B, &C);
        dense_gemm(&A, B, &ref);
        if (!same_matrix(&C, &ref)) {
            fprintf(stderr, "FAIL: %s %s N=%d: batched GEMM does not match dense_gemm\n",
                    layer_name, order_name, n);
            exit(1);
        }
        free(A.data); free(C.data); free(ref.data);
    }

    Row row;
    row.layer = layer_name;
    row.order = order_name;
    row.n = n;
    row.cycles_per_image = (double)stats.cycles / NUM_TEST_IMAGES;
    row.bytes_per_image = (double)stats.bytes / NUM_TEST_IMAGES;
    row.energy_nj_per_image = stats.energy_pj / 1000.0 / NUM_TEST_IMAGES;
    row.batches = stats.batches;
    row.memory_bound_batches = stats.memory_bound_batches;
    return row;
}

int main(void)
{
    capture_inputs();

    /* conv1 + conv2 inner-product cycles per image, from results/E1/output.txt,
     * unaffected by batching fc1/fc2. */
    const double CONV_INNER_CYCLES = 3674.0 + 13986.0;
    const double BASELINE_TOTAL_CYCLES = 22796.0;   /* total inner, N = 1, from E1 */

    Row rows[2][2][NUM_N];   /* [fc1, fc2][inner, outer][N] */
    for (int n = 0; n < NUM_N; n++) {
        rows[0][INNER][n] = run_batched("fc1", "inner", INNER, BATCH_N[n],
                                        &fc1_input[0][0], 784, &fc1_B);
        rows[0][OUTER][n] = run_batched("fc1", "outer", OUTER, BATCH_N[n],
                                        &fc1_input[0][0], 784, &fc1_B);
        rows[1][INNER][n] = run_batched("fc2", "inner", INNER, BATCH_N[n],
                                        &fc2_input[0][0], 32, &fc2_B);
        rows[1][OUTER][n] = run_batched("fc2", "outer", OUTER, BATCH_N[n],
                                        &fc2_input[0][0], 32, &fc2_B);
    }

    FILE *csv = fopen("fc_batching.csv", "w");
    FILE *txt = fopen("fc_batching.txt", "w");
    if (!csv || !txt) {
        fprintf(stderr, "cannot open output files\n");
        return 1;
    }

    fprintf(csv, "layer,order,n,cycles_per_image,bytes_per_image,energy_nj_per_image,batches,"
                 "memory_bound_batches\n");
    for (int l = 0; l < 2; l++)
        for (int order = INNER; order <= OUTER; order++)
            for (int n = 0; n < NUM_N; n++) {
                const Row *r = &rows[l][order][n];
                fprintf(csv, "%s,%s,%d,%.2f,%.2f,%.4f,%ld,%ld\n", r->layer, r->order, r->n,
                        r->cycles_per_image, r->bytes_per_image, r->energy_nj_per_image,
                        r->batches, r->memory_bound_batches);
            }
    fclose(csv);

    fprintf(txt, "Report Q8: batching fc1/fc2 images into one GEMM (tile %d x %d, "
                 "%d bytes/cycle, %d images)\n\n", TILE_ROWS, TILE_COLS, BYTES_PER_CYCLE,
            NUM_TEST_IMAGES);
    fprintf(txt, "%-5s %-6s %3s %12s %12s %14s %8s %20s\n", "layer", "order", "N",
            "cycles/img", "bytes/img", "energy(nJ)/img", "batches", "mem_bound_batches");
    for (int l = 0; l < 2; l++)
        for (int order = INNER; order <= OUTER; order++)
            for (int n = 0; n < NUM_N; n++) {
                const Row *r = &rows[l][order][n];
                fprintf(txt, "%-5s %-6s %3d %12.2f %12.2f %14.4f %8ld %20ld\n", r->layer,
                        r->order, r->n, r->cycles_per_image, r->bytes_per_image,
                        r->energy_nj_per_image, r->batches, r->memory_bound_batches);
            }

    fprintf(txt, "\nWhole-network inner-product cycles per image (conv1 + conv2 from E1, "
                 "fc1 + fc2 batched):\n");
    fprintf(txt, "%3s %14s %10s\n", "N", "total_cycles", "speedup");
    for (int n = 0; n < NUM_N; n++) {
        double total = CONV_INNER_CYCLES + rows[0][INNER][n].cycles_per_image +
                      rows[1][INNER][n].cycles_per_image;
        fprintf(txt, "%3d %14.1f %9.2fx\n", BATCH_N[n], total, BASELINE_TOTAL_CYCLES / total);
    }
    fclose(txt);

    printf("wrote fc_batching.csv and fc_batching.txt\n");
    return 0;
}
