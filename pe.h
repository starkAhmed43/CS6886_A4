/* pe.h - one SparTen processing element (PE). GIVEN: do not change.
 *
 * SparseMap format
 *   A vector of PE_VECTOR_LENGTH (16) values is stored as a 16-bit SparseMap
 *   (bit i is 1 if element i is non-zero) plus its non-zero values, packed in
 *   order. encode() builds this format from a plain array.
 *
 * Inner product (pe_inner), the way a SparTen compute unit does it
 *   1. the priority encoder holds mask_a AND mask_b; every cycle it issues the
 *      lowest set position (a match) and clears that bit;
 *   2. two prefix sums count the 1s below that position in each mask, which is
 *      where each operand's value sits in its packed array;
 *   3. the accumulator reads both values, multiplies them, and accumulates the
 *      product.
 *
 * Outer product (pe_outer)
 *   Every non-zero of one vector is multiplied with every non-zero of the
 *   other, so no matching is needed.
 *
 * Timing, measured on the RTL of our own PE implementation
 *   The PE handles one match per cycle, and filling and draining its pipeline
 *   adds 8 cycles, so a chunk with m matches takes m + 8 cycles. A chunk with
 *   no matches takes 7 cycles. The outer product is assumed to use the same
 *   pipeline.
 */
#ifndef PE_H
#define PE_H

#include <stdint.h>

#define PE_VECTOR_LENGTH 16            /* values per vector = bits in a SparseMap */

/* ---------------------------------------------------------------- timing */

#define PIPELINE_FILL_DRAIN_CYCLES 8   /* added to the matches of a chunk */
#define EMPTY_CHUNK_CYCLES 7           /* a chunk with no matches          */

/* Cycles of one chunk that does `macs` useful MACs. */
static inline int chunk_cycles(int macs)
{
    return macs > 0 ? macs + PIPELINE_FILL_DRAIN_CYCLES : EMPTY_CHUNK_CYCLES;
}

/* ---------------------------------------------------------------- energy
 *
 * Table 4 of the SparTen paper gives the power of each component of one 32-PE
 * cluster (45 nm, 800 MHz, 118.3 mW in total). At full activity every PE
 * handles one match per cycle, so the cluster handles 32 x 800 million matches
 * per second, and the energy of one match is
 *
 *     energy per match (pJ) = power (mW) / (32 x 0.8 GHz)
 *
 * The paper's SparseMap is 128 bits wide and ours is 16 bits, so we scale the
 * prefix-sum and priority-encoder energy by 16/128 (our assumption). Pipeline
 * fill and drain cycles and idle PEs cost no energy.
 */
#define PJ_PER_MATCH(milliwatts) ((milliwatts) / (32 * 0.8))
#define SPARSEMAP_WIDTH_SCALE (PE_VECTOR_LENGTH / 128.0)

/* Energy of each component per match, in picojoules (pJ). */
#define ENERGY_PREFIX_SUMS (PJ_PER_MATCH(48.0) * SPARSEMAP_WIDTH_SCALE)   /* both of them */
#define ENERGY_PRIORITY_ENCODER (PJ_PER_MATCH(6.4) * SPARSEMAP_WIDTH_SCALE)
#define ENERGY_MAC PJ_PER_MATCH(13.82)
#define ENERGY_BUFFERS PJ_PER_MATCH(19.2)
#define ENERGY_PERMUTE_NETWORK PJ_PER_MATCH(10.6)
#define ENERGY_OTHER PJ_PER_MATCH(20.28)

/* One inner-product match: a priority-encoder step, two prefix sums, the MAC,
 * the operand buffers, and the rest of the PE. */
#define ENERGY_PER_INNER_MATCH (ENERGY_PRIORITY_ENCODER + ENERGY_PREFIX_SUMS + ENERGY_MAC + \
                                ENERGY_BUFFERS + ENERGY_OTHER)
/* One outer-product product: as above, but with no prefix sums (the packed
 * values are read in order), plus the permute network that routes the product
 * to its accumulator. */
#define ENERGY_PER_OUTER_PRODUCT (ENERGY_PRIORITY_ENCODER + ENERGY_MAC + ENERGY_BUFFERS + \
                                  ENERGY_OTHER + ENERGY_PERMUTE_NETWORK)
/* Starting a chunk: the priority encoder's first step, paid even with no match. */
#define ENERGY_CHUNK_START ENERGY_PRIORITY_ENCODER

/* ---------------------------------------------------------------- SparseMap */

typedef struct {
    uint16_t mask;                    /* the SparseMap: bit i is 1 if element i is non-zero */
    int count;                        /* number of non-zero values                          */
    int values[PE_VECTOR_LENGTH];     /* the non-zero values, packed in order               */
} SparseVector;

/* Bytes one stored vector takes in SRAM: its 2-byte SparseMap plus 1 byte per
 * non-zero value (INT8). A vector with no non-zeros still costs its SparseMap. */
#define SPARSEMAP_BYTES (PE_VECTOR_LENGTH / 8)
static inline int vector_bytes(const SparseVector *v)
{
    return SPARSEMAP_BYTES + v->count;
}

/* Store the first `length` values of `dense` (length <= 16) in SparseMap format. */
static inline void encode(const int *dense, int length, SparseVector *v)
{
    v->mask = 0;
    v->count = 0;
    for (int i = 0; i < length; i++)
        if (dense[i] != 0) {
            v->mask |= 1u << i;
            v->values[v->count++] = dense[i];
        }
}

/* ---------------------------------------------------------------- priority encoder */

/* Holds the matching positions that have not been issued yet. */
typedef struct {
    uint16_t remaining;
} PriorityEncoder;

static inline void encoder_start(PriorityEncoder *encoder, uint16_t mask_a, uint16_t mask_b)
{
    encoder->remaining = mask_a & mask_b;
}

/* One cycle: issue the lowest remaining position and clear it. Returns -1 when
 * no position is left. */
static inline int encoder_next(PriorityEncoder *encoder)
{
    for (int position = 0; position < PE_VECTOR_LENGTH; position++)
        if (encoder->remaining & (1u << position)) {
            encoder->remaining &= ~(1u << position);
            return position;
        }
    return -1;
}

/* ---------------------------------------------------------------- prefix sum */

/* The number of 1s in `mask` below `position`, which is the index of that
 * element's value in the packed values[]. */
static inline int prefix_sum(uint16_t mask, int position)
{
    int ones = 0;
    for (int i = 0; i < position; i++)
        ones += (mask >> i) & 1;
    return ones;
}

/* ---------------------------------------------------------------- the PE */

/* What one chunk did. */
typedef struct {
    int sum;            /* pe_inner: the dot product                       */
    int macs;           /* useful MACs (both operands non-zero)            */
    int cycles;         /* cycles the chunk took                           */
    double energy_pj;   /* energy the chunk used, in picojoules            */
} PeResult;

/* Inner product: the dot product of a and b, one match per cycle. */
static inline PeResult pe_inner(const SparseVector *a, const SparseVector *b)
{
    PeResult result = {.sum = 0, .macs = 0, .cycles = 0, .energy_pj = ENERGY_CHUNK_START};
    PriorityEncoder encoder;
    encoder_start(&encoder, a->mask, b->mask);
    int position;
    while ((position = encoder_next(&encoder)) != -1) {
        int a_value = a->values[prefix_sum(a->mask, position)];
        int b_value = b->values[prefix_sum(b->mask, position)];
        result.sum += a_value * b_value;
        result.macs++;
    }
    result.cycles = chunk_cycles(result.macs);
    result.energy_pj += result.macs * ENERGY_PER_INNER_MATCH;
    return result;
}

/* Outer product, one product per cycle. `a` is a column piece (one value per
 * row of the tile) and `b` is a row piece (one value per column of the tile).
 * The product of element r of a and element c of b is added to the tile's
 * partial sum partial_sums[r * cols + c], and that output's accumulator gets
 * one more value to add: accumulator_load[r * cols + c]++. */
static inline PeResult pe_outer(const SparseVector *a, const SparseVector *b, int *partial_sums,
                                int cols, int *accumulator_load)
{
    PeResult result = {.sum = 0, .macs = 0, .cycles = 0, .energy_pj = ENERGY_CHUNK_START};
    int a_index = 0;                               /* next packed value of a */
    for (int r = 0; r < PE_VECTOR_LENGTH; r++) {
        if (!(a->mask & (1u << r)))
            continue;
        int b_index = 0;                           /* next packed value of b */
        for (int c = 0; c < PE_VECTOR_LENGTH; c++) {
            if (!(b->mask & (1u << c)))
                continue;
            partial_sums[r * cols + c] += a->values[a_index] * b->values[b_index];
            accumulator_load[r * cols + c]++;
            result.macs++;
            b_index++;
        }
        a_index++;
    }
    result.cycles = chunk_cycles(result.macs);
    result.energy_pj += result.macs * ENERGY_PER_OUTER_PRODUCT;
    return result;
}

#endif
