# CS6886 Assignment 4: SparTen on MNIST

You will complete a simulator of a SparTen-style sparse CNN accelerator
(Gondimalla et al., MICRO 2019) with 16 PEs, and use it to run a real pruned
INT8 CNN (98.25% accuracy on MNIST). The full instructions are in the
assignment PDF; this file is a short summary.

```sh
gcc -O2 -Wall sparten.c -o sparten -lm && ./sparten
gcc -O2 -Wall -DTILE_ROWS=2 -DTILE_COLS=8 sparten.c -o sparten -lm && ./sparten   # another tile
gcc -O2 -Wall -DBYTES_PER_CYCLE=8 sparten.c -o sparten -lm && ./sparten          # less bandwidth
gcc -O2 -Wall -DNUM_IMAGES=5 sparten.c -o sparten -lm && ./sparten               # fewer images
```

Each run also writes `roofline.csv` (for your own plots) and `roofline.svg`
(open it in a web browser).

## Files

| File | What it is |
|---|---|
| `sparten.c` | the top-level simulator: you complete the 8 TODOs here |
| `pe.h` | given, do not change: the PE-level simulator (one SparTen compute unit) |
| `mnist_model.h` | the pruned INT8 network: conv1 (1 to 8 channels), conv2 (8 to 16), fc1 (784 to 32), fc2 (32 to 10) |
| `mnist_images.h` | 100 test images, their labels, and the digit the reference model predicts for each |

## The SparTen model

- **Processing element (`pe.h`).** A vector of 16 values is stored as a
  16-bit SparseMap (bit i is 1 if element i is non-zero) plus its non-zero
  values packed in order. `pe_inner` computes a dot product:
  - the priority encoder holds `mask_a AND mask_b`; every cycle it issues the
    lowest set position (a match) and clears that bit;
  - two prefix sums count the 1s below that position in each mask, which is
    where each operand's value sits in its packed array;
  - the accumulator reads both values, multiplies them, and accumulates the
    product.

  `pe_outer` multiplies every non-zero of one vector with every non-zero of
  the other (no matching is needed).
- **PE timing** (measured on the RTL of our own PE implementation). The PE
  handles one match per cycle, and filling and draining its pipeline adds 8
  cycles, so a chunk with m matches takes **m + 8 cycles**. A chunk with no
  matches takes **7 cycles**. The outer product is assumed to use the same
  pipeline. The clock frequency is 800 MHz.
- **Energy.** Table 4 of the SparTen paper: one 32-PE cluster at 45 nm and
  800 MHz consumes 118.3 mW. `pe.h` converts each component's power into an
  energy per match, in picojoules (pJ).
- **Memory.** The PEs read their operands from on-chip SRAM through a 128-bit
  port: 16 bytes per cycle. A stored vector costs 2 bytes of SparseMap plus 1
  byte per non-zero value; a partial sum costs 4 bytes.
- **Batches.** One tile is one batch of up to 16 PE tasks. The next batch
  starts only when all 16 PEs are done (no load balancing), so the *compute
  cycles* of a batch are those of its slowest PE. In the outer product, each
  of the 16 accumulators also adds only one value per cycle. The *memory
  cycles* of a batch are its bytes divided by the bandwidth, rounded up. The
  two overlap, so a batch takes whichever is larger.
- **Roofline.** Peak compute is 16 MACs = 32 operations per cycle. Peak
  bandwidth is 16 bytes per cycle. The ridge point is 32 / 16 = 2 operations
  per byte.

## What you write

| TODO | Function | What it does |
|---|---|---|
| 1 | `lower_fc` | a fully-connected layer as a GEMM `A[1 × in] × B[in × out]` |
| 2 | `lower_conv` | a 3×3 convolution as a GEMM (im2col): `A[positions × 9·channels] × B[9·channels × filters]` |
| 3 | `load_tile` | cut a `TILE_ROWS × 16` block of A and a `16 × TILE_COLS` block of B, zero-padding the edges |
| 4 | `gemm_inner` | inner product: one PE per output of the tile; each depth tile is one batch |
| 5 | `gemm_outer` | outer product: one PE per depth index of the tile; the depth tile is the outermost loop |
| 6 | `layer_metrics` | cycles, time, utilization, MACs per cycle, energy and power per image |
| 7 | inside `gemm_inner` and `gemm_outer` | count the bytes each batch moves with `count_bytes()` |
| 8 | `roofline_point` | operational intensity, achieved performance, the attainable roof, and whether the layer is memory-bound |

## Checking your work

- You can compile and run the program before completing any TODO. It prints
  CHECK lines, a low accuracy, and zeros in the tables until the TODOs are
  done. A CHECK line means that one of the correctness checks failed, and it
  names the TODO to look at:
  - TODOs 1–2 are compared against the layer computed directly;
  - TODOs 3–5 are compared against a plain matrix multiply.
- Only the inner product's result is passed on to the next layer, so the
  accuracy reaches 100/100 when TODOs 1–4 are right. The outer product is
  checked only by its own CHECK line.
- For TODOs 6–8 there is no reference output. The last line of the output is
  a sanity check that prints a WARNING line for every impossible result (for
  example, a utilization above 100%) and names the TODO to look at. The
  absence of warnings does not mean that your numbers are right, only that
  they are possible. Check them against the worked examples in TODO 7 and by
  hand in experiment E4.
