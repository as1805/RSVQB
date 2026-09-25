// svqb2.hpp
// Two-pass SVQB (Stathopoulos & Wu) reorthogonalization — public API.
//
// Pass i:  G = XᵀX,  D = sqrt(diag(G)),  Ĝ = D⁻¹GD⁻¹ = V S Vᵀ (eig),
//          M = D⁻¹ V S^(-1/2),  X := X M   (X orthonormal on exit)
// Pass 2 re-applies the same map to the Pass-1 output (Q₁) for
// numerical reorthogonalization, analogous to CholQR2's two Cholesky passes.
//
// R is recovered as R = R₂·R₁ with R₁ = Q₁ᵀA, R₂ = QᵀQ₁, so that A ≈ Q·R.
// Unlike CholQR2 (where R falls out of the timed passes for free as the
// Cholesky factors), R1/R2 here are new O(mn²) GEMMs, so — for an honest
// time comparison against CholQR2 — their cost IS included in time_s.
//
// Parameters:
//   h_A     input  matrix A (m×n, column-major, host)
//   m, n    matrix dimensions
//   h_Q     output orthonormal factor Q (m×n, column-major, host)
//   h_R     output upper-triangular-equivalent factor R (n×n, column-major, host)
//   time_s  average GPU computation time in seconds, INCLUDING R recovery
//           (excludes only host<->device copies)
//   trials  number of timed repetitions (device buffers allocated once)
#pragma once

void svqb2(const double* h_A, int m, int n,
           double* h_Q, double* h_R, double& time_s, int trials = 1);

void svqb3(const double* h_A, int m, int n,
           double* h_Q, double* h_R, double& time_s, int trials = 1);
           // Mixed-precision SVQB: first SVQB pass in FP32, promote to FP64,
// then perform the second SVQB pass in FP64.
void svqb2_mixed(const double* h_A, int m, int n,
                 double* h_Q, double* h_R, double& time_s, int trials=1);

// One FP64 SVQB pass followed by a Cholesky-QR pass.
void svqb2_cholqr(const double* h_A, int m, int n,
                  double* h_Q, double* h_R, double& time_s, int trials=1);
