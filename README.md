# GPU QR Benchmark Suite

This directory contains CUDA implementations and benchmarks for randomized and Cholesky-based QR factorizations, with an emphasis on mixed-precision and large tall-and-skinny matrices.

## Included algorithms

- `mrcqr_fp64`, `mrcqr_fp32`, `mrcqr_fp16`
  - Mixed-precision randomized Cholesky QR variants
  - Uses a randomized sketch plus a final FP64 Cholesky QR refinement
- `cholqr2`
  - Optimized CholeskyQR2 implementation
- `svqb2`
  - SVQB2 variant included in the benchmark suite
- `mrsqr*`, `mreig*`
  - Related randomized QR and eigenvalue-based variants used in timing comparisons

## Main source files

- `mrcqr.cu` / `mrcqr.hpp` — MRCQR implementation and public API
- `cholqr2.cu` / `cholqr2.hpp` — CholeskyQR2 implementation
- `svqb2.cu` / `svqb2.hpp` — SVQB2 implementation
- `common.cuh` — common CUDA error checks and timing utilities
- `driver_accuracy.cu` — accuracy comparison across condition numbers
- `driver_perf.cpp` — timing benchmark across matrix sizes
- `driver_geqrf.cu` — cuSOLVER QR reference benchmark
- `driver_profile.cu` — profiling driver for sub-operations
- `driver_scaling.cu` — scaling experiments

## Build

This project is built with a Makefile in the same directory.

```bash
make all
```

Individual targets are also available:

```bash
make driver_accuracy
make driver_perf
make driver_geqrf
make driver_profile
make driver_scaling
```

Convenience run targets:

```bash
make run_accuracy
make run_perf
```

Cleanup:

```bash
make clean
```

## Requirements

- CUDA toolkit with `nvcc`
- `cublas`, `cusolver`, and `cufft` libraries
- A working GPU for benchmark execution

On some systems you may need to load the CUDA environment first, for example:

```bash
module load cuda
```

or set the relevant environment variables used by the Makefile before building.

## Notes

- The Makefile auto-detects the local machine and CUDA installation where possible.
- It also auto-detects the GPU compute capability when `nvidia-smi` is available.
- You can override the target GPU architecture manually with:

```bash
make GENCODE="-gencode=arch=compute_80,code=sm_80"
```

- Timing results are written into the `results/` directory by the run targets.
