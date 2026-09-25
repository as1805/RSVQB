// cholqr2.hpp
// Optimised CholeskyQR2 — public API.
//
// Uses in-place SIDE_RIGHT TRSM (no transpose copies, no extra m×n buffers).
//
// Parameters:
//   h_A     input  matrix A (m×n, column-major, host)
//   m, n    matrix dimensions
//   h_Q     output orthonormal factor Q (m×n, column-major, host)
//   h_R     output upper-triangular factor R (n×n, column-major, host)
//   time_s  average GPU computation time in seconds (excludes host↔device copies)
//   trials  number of timed repetitions (device buffers allocated once)
#pragma once

void cholqr2(const double* h_A, int m, int n,
             double* h_Q, double* h_R, double& time_s, int trials = 1);
