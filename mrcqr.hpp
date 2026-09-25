// mrcqr.hpp
// Mixed-precision Randomized CholeskyQR (MRCQR) — public API.
//
// All three variants use:
//   - Rademacher sign kernel (int8 diagonal, fused convert + sign flip)
//   - FFT-based sketch (D2Z / R2C / cufftXt for FP64 / FP32 / FP16)
//   - In-place SIDE_RIGHT TRSM (no transpose copies)
//   - FP64 CholQR phase for all variants
//
// Parameters:
//   h_A      input  matrix A (m×n, column-major, host)
//   m, n     matrix dimensions
//   c        sketch size (number of sampled FFT frequencies); recommended c = 3*n
//   h_Q      output orthonormal factor Q (m×n, column-major, host)
//   h_R      output upper-triangular factor R (n×n, column-major, host)
//   time_s   average GPU computation time in seconds (excludes host↔device copies)
//   trials   number of timed repetitions (device buffers allocated once)
//   re_only  if false (default): sketch has 2c rows (Re+Im of each sampled frequency),
//            SketchQR on 2c×n.  If true: sketch has c rows (Re only), SketchQR on c×n
//            (lower cost; accuracy should be verified for the target condition number).
//
// A ≈ Q * R  to double precision regardless of sketch precision.
#pragma once

void mrcqr_fp64(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s,
                int trials = 1, bool re_only = true);

void mrcqr_fp32(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s,
                int trials = 1, bool re_only = true);

void mrcqr_fp16(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s,
                int trials = 1, bool re_only = true);

void mrsqr_fp64(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s,
                int trials = 1, bool re_only = true);
                
void mrsqr2_fp64(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s,
                int trials = 1, bool re_only = true);

void mrsqr2_gesvdj_fp64(const double* h_A, int m, int n, int c,
                        double* h_Q, double* h_R, double& time_s,
                        int trials = 1, bool re_only = true);

void mrsqr2_gesvdp_fp64(const double* h_A, int m, int n, int c,
                        double* h_Q, double* h_R, double& time_s,
                        int trials = 1, bool re_only = true);

void mreig_svqb(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s,
                int trials = 1, bool re_only = true);

void mreig2_svqb(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s,
                int trials = 1, bool re_only = true);
