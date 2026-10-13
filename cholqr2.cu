// cholqr2.cu
// Optimised CholeskyQR2 library.
// See cholqr2.hpp for API documentation.
//
// Algorithm (two passes, both in-place on the single m×n buffer d_A):
//   Pass 1:  G  = AᵀA,  R₁ = chol(G),  A := A·R₁⁻¹ = Q₀  (SIDE_RIGHT)
//   Pass 2:  G₁ = Q₀ᵀQ₀, R₂ = chol(G₁), A := Q₀·R₂⁻¹ = Q (SIDE_RIGHT)
//   R = R₂·R₁  (small n×n GEMM, not timed)

#include "cholqr2.hpp"
#include "common.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

// Upper-triangle copy for n×n matrices (Cholesky factor extraction).
struct TriNN {
    int n; const double* src; double* dst;
    __host__ __device__ void operator()(int x) const {
        int row = x/n, col = x%n;
        if (row <= col) dst[row + n*col] = src[row + n*col];
    }
};

void cholqr2(const double* h_A, int m, int n,
             double* h_Q, double* h_R, double& time_s, int trials)
{
    const double a = 1., b = 0.;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    cublasHandle_t    blas;   CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    // Upload A (NOT timed). d_A is the only m×n live buffer.
    thrust::device_vector<double> d_A(h_A, h_A + (long long)m*n);
    thrust::device_vector<double> d_Abak(d_A);

    thrust::device_vector<double> d_G(n*n), d_G1(n*n);
    thrust::device_vector<double> d_R1(n*n, 0.), d_R2(n*n, 0.), d_R(n*n);

    double* pA  = thrust::raw_pointer_cast(d_A.data());
    double* pG  = thrust::raw_pointer_cast(d_G.data());
    double* pG1 = thrust::raw_pointer_cast(d_G1.data());
    double* pR1 = thrust::raw_pointer_cast(d_R1.data());
    double* pR2 = thrust::raw_pointer_cast(d_R2.data());
    double* pR  = thrust::raw_pointer_cast(d_R.data());

    int lw = 0; double* d_work = nullptr; int* d_info = nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, uplo, n, pG, n, &lw));
    CUDA_CHECK(cudaMalloc(&d_work, lw * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    auto one = [&]() {
        pA = thrust::raw_pointer_cast(d_A.data());

        // Pass 1
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                                 &a, pA, m, pA, m, &b, pG, n));
        CUSOLVER_CHECK(cusolverDnDpotrf(solver, uplo, n, pG, n, d_work, lw, d_info));
        CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, m, n, &a, pG, n, pA, m));
        // d_A now holds Q₀

        // Pass 2
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                                 &a, pA, m, pA, m, &b, pG1, n));
        CUSOLVER_CHECK(cusolverDnDpotrf(solver, uplo, n, pG1, n, d_work, lw, d_info));
        CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, m, n, &a, pG1, n, pA, m));
        // d_A now holds Q
    };

    // Warmup (NOT timed)
    thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
    one(); CUDA_CHECK(cudaDeviceSynchronize());

    // Timed trials (host↔device copies excluded)
    cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

CUBLAS_CHECK(cublasSetStream(blas, stream));
CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));
    EV wall; wall.start(stream);
    for (int t = 0; t < trials; ++t) {
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
        one();
    }
    time_s = wall.stop(stream) / trials;

    // Compute R = R₂ · R₁ (NOT timed)
    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n*n), TriNN{n, pG,  pR1});
    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n*n), TriNN{n, pG1, pR2});
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                             &a, pR2, n, pR1, n, &b, pR, n));
    thrust::copy(d_R.begin(), d_R.end(), h_R);

    // Download Q (NOT timed)
    pA = thrust::raw_pointer_cast(d_A.data());
    thrust::copy(d_A.begin(), d_A.begin() + (long long)m*n, h_Q);

    CUDA_CHECK(cudaFree(d_work)); CUDA_CHECK(cudaFree(d_info));
    cublasDestroy(blas); cusolverDnDestroy(solver);
}
