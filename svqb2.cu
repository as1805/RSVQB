// svqb2.cu
// Two-pass SVQB reorthogonalization library.
// See svqb2.hpp for API documentation.
//
// Algorithm (two passes, ping-ponged between the two live m×n buffers d_A/d_B):
//   Pass 1:  G  = AᵀA,   D  = sqrt(diag(G)),  Ĝ  = D⁻¹GD⁻¹ = V₁S₁V₁ᵀ,
//            M₁ = D⁻¹V₁S₁^(-1/2),   B := A·M₁ = Q₁
//   Pass 2:  G₁ = Q₁ᵀQ₁, D₁ = sqrt(diag(G₁)), Ĝ₁ = D₁⁻¹G₁D₁⁻¹ = V₂S₂V₂ᵀ,
//            M₂ = D₁⁻¹V₂S₂^(-1/2), A := Q₁·M₂ = Q
//   R₁ = Q₁ᵀA₀   (TIMED — see note below)
//   R₂ = QᵀQ₁    (TIMED)
//   R  = R₂·R₁   (small n×n GEMM, negligible cost either way)
//
// Note: unlike CholQR2's in-place SIDE_RIGHT TRSM (triangular, safe to alias),
// SVQB's update matrix M is dense, so cuBLAS GEMM requires distinct
// input/output buffers — hence two live m×n buffers instead of one.
//


#include "svqb2.hpp"
#include "common.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <cmath>

// Diagonal extraction: Dinv[i] = 1/sqrt(G_ii)  (guards non-positive diag entries)
struct DiagInv {
    int n; const double* G; double* Dinv;
    __host__ __device__ void operator()(int i) const {
        double v = G[i + n * i];
        Dinv[i] = (v > 0.0) ? 1.0 / sqrt(v) : 1.0;
    }
};

// Eigenvalue regularization + inverse sqrt: rsqrt[i] = 1/sqrt(max(s_i, maxeig*1e-15))
struct RSqrtClamp {
    double maxeig;
    __host__ __device__ double operator()(double s) const {
        double floor_ = maxeig * 1e-15;
        double sc = (s < floor_) ? floor_ : s;
        return 1.0 / sqrt(sc);
    }
};

// ---------------------------------------------------------------------------
// One SVQB orthogonalization pass: pXout = pXin * M, where M orthonormalizes
// the columns of pXin via a symmetric-eigendecomposition of the (normalized)
// Gram matrix. pXin and pXout must be distinct m×n buffers.
// Scratch buffers (pG, peig, pDinv, prsqrt, pM) are n×n / n and reused
// across calls; syevd workspace is likewise pre-allocated by the caller.
// ---------------------------------------------------------------------------
static void svqb_pass(cublasHandle_t blas, cusolverDnHandle_t solver,
                       const double* pXin, double* pXout, int m, int n,
                       double* pG, double* peig, double* pDinv, double* prsqrt, double* pM,
                       double* d_wev, int lw_ev, int* d_iev)
{
    const double a = 1., b = 0.;

    // G = Xin^T Xin
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &a, pXin, m, pXin, m, &b, pG, n));

    // Dinv[i] = 1/sqrt(G_ii)
    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n), DiagInv{n, pG, pDinv});

    // Ghat = Dinv * G * Dinv  (row scale, then column scale — cheap, n×n only)
    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_LEFT,  n, n, pG, n, pDinv, 1, pG, n));
    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_RIGHT, n, n, pG, n, pDinv, 1, pG, n));

    // Eigendecompose Ghat in place: pG <- V (eigenvectors), peig <- eigenvalues (ascending)
    CUSOLVER_CHECK(cusolverDnDsyevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                    n, pG, n, peig, d_wev, lw_ev, d_iev));

    // rsqrt[i] = 1/sqrt(clamp(eig[i]));  max eigenvalue = peig[n-1] (ascending order)
    double maxeig;
    CUDA_CHECK(cudaMemcpy(&maxeig, peig + (n - 1), sizeof(double), cudaMemcpyDeviceToHost));
    thrust::device_ptr<double> deig(peig), drsqrt(prsqrt);
    thrust::transform(deig, deig + n, drsqrt, RSqrtClamp{maxeig});

    // M = Dinv * V * diag(rsqrt)
    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_RIGHT, n, n, pG, n, prsqrt, 1, pM, n)); // pM = V*S^-1/2
    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_LEFT,  n, n, pM, n, pDinv,  1, pM, n)); // pM = Dinv*(V*S^-1/2)

    // Xout = Xin * M
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &a, pXin, m, pM, n, &b, pXout, m));
}

// ---------------------------------------------------------------------------
// FP32 SVQB pass. This is the low-precision analogue of svqb_pass above.
// ---------------------------------------------------------------------------
struct DiagInvF {
    int n; const float* G; float* Dinv;
    __host__ __device__ void operator()(int i) const {
        float v = G[i + n * i];
        Dinv[i] = (v > 0.0f) ? 1.0f / sqrtf(v) : 1.0f;
    }
};

struct RSqrtClampF {
    float maxeig;
    __host__ __device__ float operator()(float s) const {
        float floor_ = maxeig * 1e-6f;
        float sc = (s < floor_) ? floor_ : s;
        return 1.0f / sqrtf(sc);
    }
};

static void svqb_pass_fp32(cublasHandle_t blas, cusolverDnHandle_t solver,
                            const float* pXin, float* pXout, int m, int n,
                            float* pG, float* peig, float* pDinv,
                            float* prsqrt, float* pM,
                            float* d_wev, int lw_ev, int* d_iev)
{
    const float a = 1.f, b = 0.f;

    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &a, pXin, m, pXin, m, &b, pG, n));

    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n),
                     DiagInvF{n, pG, pDinv});

    CUBLAS_CHECK(cublasSdgmm(blas, CUBLAS_SIDE_LEFT, n, n,
                             pG, n, pDinv, 1, pG, n));
    CUBLAS_CHECK(cublasSdgmm(blas, CUBLAS_SIDE_RIGHT, n, n,
                             pG, n, pDinv, 1, pG, n));

    CUSOLVER_CHECK(cusolverDnSsyevd(solver, CUSOLVER_EIG_MODE_VECTOR,
                                     CUBLAS_FILL_MODE_LOWER, n, pG, n, peig,
                                     d_wev, lw_ev, d_iev));

    float maxeig;
    CUDA_CHECK(cudaMemcpy(&maxeig, peig + (n - 1), sizeof(float),
                          cudaMemcpyDeviceToHost));

    thrust::device_ptr<float> deig(peig), drsqrt(prsqrt);
    thrust::transform(deig, deig + n, drsqrt, RSqrtClampF{maxeig});

    CUBLAS_CHECK(cublasSdgmm(blas, CUBLAS_SIDE_RIGHT, n, n,
                             pG, n, prsqrt, 1, pM, n));
    CUBLAS_CHECK(cublasSdgmm(blas, CUBLAS_SIDE_LEFT, n, n,
                             pM, n, pDinv, 1, pM, n));

    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &a, pXin, m, pM, n, &b, pXout, m));
}

// Cholesky-QR pass: G = X^T X = R^T R, Q = X R^{-1}.
// The returned R is the Cholesky factor, upper triangular.
static void cholqr_pass(cublasHandle_t blas, cusolverDnHandle_t solver,
                        const double* pXin, double* pXout, int m, int n,
                        double* pG, double* pR,
                        double* d_work, int lw, int* d_info)
{
    const double one = 1., zero = 0.;

    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &one, pXin, m, pXin, m,
                             &zero, pG, n));

    // Upper-triangular Cholesky: G = R^T R.
    CUSOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER,
                                     n, pG, n, d_work, lw, d_info));

    // Save R before pG is used as the triangular solve matrix.
    CUBLAS_CHECK(cublasDcopy(blas, n * n, pG, 1, pR, 1));

    // Q = X * inv(R), implemented as a right-side triangular solve.
    CUBLAS_CHECK(cublasDcopy(blas, m * n, pXin, 1, pXout, 1));
    CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                             m, n, &one, pG, n, pXout, m));
}

// ---------------------------------------------------------------------------
// Mixed precision SVQB:
//   A --(FP32 SVQB pass)--> Q1_fp32 --promote--> Q1_fp64
//     --(FP64 SVQB pass)--> Q.
// ---------------------------------------------------------------------------
void svqb2_mixed(const double* h_A, int m, int n,
                 double* h_Q, double* h_R, double& time_s, int trials)
{
    const double one = 1.0, zero = 0.0;
    const long long mn = (long long)m * n;

    cublasHandle_t blas;     CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    // FP64 live buffers: A64 becomes Q1; B64 becomes final Q.
    thrust::device_vector<double> d_A64(h_A, h_A + mn);
    thrust::device_vector<double> d_Abak(d_A64);
    thrust::device_vector<double> d_B64(mn);

    // FP32 live buffers: A32 is input A0; B32 becomes Q1_32.
    thrust::device_vector<float> d_A32(mn);
    thrust::device_vector<float> d_B32(mn);

    thrust::device_vector<double> d_G64(n * n), d_M64(n * n);
    thrust::device_vector<double> d_eig64(n), d_Dinv64(n), d_rsqrt64(n);

    thrust::device_vector<float> d_G32(n * n), d_M32(n * n);
    thrust::device_vector<float> d_eig32(n), d_Dinv32(n), d_rsqrt32(n);

    thrust::device_vector<double> d_R1(n * n), d_R2(n * n), d_R(n * n);

    double* pG64   = thrust::raw_pointer_cast(d_G64.data());
double* pEig64 = thrust::raw_pointer_cast(d_eig64.data());

float* pG32    = thrust::raw_pointer_cast(d_G32.data());
float* pEig32  = thrust::raw_pointer_cast(d_eig32.data());

int lw64 = 0, lw32 = 0;
double* d_wev64 = nullptr;
int*    d_iev64 = nullptr;
float*  d_wev32 = nullptr;
int*    d_iev32 = nullptr;

CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
    solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    n, pG64, n, pEig64, &lw64));

CUDA_CHECK(cudaMalloc((void**)&d_wev64, (size_t)lw64 * sizeof(double)));
CUDA_CHECK(cudaMalloc((void**)&d_iev64, sizeof(int)));

CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(
    solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    n, pG32, n, pEig32, &lw32));

CUDA_CHECK(cudaMalloc((void**)&d_wev32, (size_t)lw32 * sizeof(float)));
CUDA_CHECK(cudaMalloc((void**)&d_iev32, sizeof(int)));

    double* pA64 = thrust::raw_pointer_cast(d_A64.data());
    double* pB64 = thrust::raw_pointer_cast(d_B64.data());
    float*  pA32 = thrust::raw_pointer_cast(d_A32.data());
    float*  pB32 = thrust::raw_pointer_cast(d_B32.data());

    double* pAbak = thrust::raw_pointer_cast(d_Abak.data());
    double* pR1   = thrust::raw_pointer_cast(d_R1.data());
    double* pR2   = thrust::raw_pointer_cast(d_R2.data());
    double* pR    = thrust::raw_pointer_cast(d_R.data());

    //int lw64 = 0, lw32 = 0;
   // double* d_wev64 = nullptr; int* d_iev64 = nullptr;
   // float*  d_wev32 = nullptr; int* d_iev32 = nullptr;

    auto one_run = [&]() {
        pA64 = thrust::raw_pointer_cast(d_A64.data());
        pB64 = thrust::raw_pointer_cast(d_B64.data());
        pA32 = thrust::raw_pointer_cast(d_A32.data());
        pB32 = thrust::raw_pointer_cast(d_B32.data());

        // Pass 1: A0 in FP32 -> Q1 in FP32.
        svqb_pass_fp32(
            blas, solver, pA32, pB32, m, n,
            thrust::raw_pointer_cast(d_G32.data()),
            thrust::raw_pointer_cast(d_eig32.data()),
            thrust::raw_pointer_cast(d_Dinv32.data()),
            thrust::raw_pointer_cast(d_rsqrt32.data()),
            thrust::raw_pointer_cast(d_M32.data()),
            d_wev32, lw32, d_iev32);

        // Promote Q1_32 to FP64.  A64 now holds Q1.
        thrust::copy(d_B32.begin(), d_B32.end(), d_A64.begin());

        // Pass 2: Q1 in FP64 -> final Q in FP64.
        svqb_pass(
            blas, solver, pA64, pB64, m, n,
            thrust::raw_pointer_cast(d_G64.data()),
            thrust::raw_pointer_cast(d_eig64.data()),
            thrust::raw_pointer_cast(d_Dinv64.data()),
            thrust::raw_pointer_cast(d_rsqrt64.data()),
            thrust::raw_pointer_cast(d_M64.data()),
            d_wev64, lw64, d_iev64);
    };

    auto restore_input = [&]() {
        // Each pass overwrites its input, so restore before every run.
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A64.begin());
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A32.begin()); // double -> float
    };

    // Warmup — not timed.
    restore_input();
    one_run();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed trials — factorization passes only.
    EV wall;
    wall.start();

    for (int t = 0; t < trials; ++t) {
        restore_input();
        one_run();
    }

    time_s = wall.stop() / trials;

    // After the final run:
    // A64 = Q1, B64 = Q, Abak = original A0.

    // R1 = Q1^T A0
    CUBLAS_CHECK(cublasDgemm(
        blas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, n, m, &one,
        pA64, m, pAbak, m,
        &zero, pR1, n));

    // R2 = Q^T Q1
    CUBLAS_CHECK(cublasDgemm(
        blas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, n, m, &one,
        pB64, m, pA64, m,
        &zero, pR2, n));

    // R = R2 R1
    CUBLAS_CHECK(cublasDgemm(
        blas, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n, &one,
        pR2, n, pR1, n,
        &zero, pR, n));

    // Final Q is B64, not A64.
    thrust::copy(d_B64.begin(), d_B64.end(), h_Q);
    thrust::copy(d_R.begin(), d_R.end(), h_R);

    cudaFree(d_wev64);
    cudaFree(d_iev64);
    cudaFree(d_wev32);
    cudaFree(d_iev32);

    cublasDestroy(blas);
    cusolverDnDestroy(solver);
}

// ---------------------------------------------------------------------------
// SVQB followed by Cholesky QR.
//   A -- SVQB --> Q1 --Cholesky QR--> Q.
// The final R is recovered as Q^T A, so the routine returns a QR factorization
// of the original input rather than the intermediate SVQB factorization.
// ---------------------------------------------------------------------------
void svqb2_cholqr(const double* h_A, int m, int n,
                  double* h_Q, double* h_R, double& time_s, int trials)
{
    const double one = 1., zero = 0.;

    cublasHandle_t blas; CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    thrust::device_vector<double> d_A(h_A, h_A + (long long)m*n);
    thrust::device_vector<double> d_Abak(d_A);
    thrust::device_vector<double> d_B((long long)m*n);
    thrust::device_vector<double> d_G(n*n), d_M(n*n);
    thrust::device_vector<double> d_eig(n), d_Dinv(n), d_rsqrt(n);
    thrust::device_vector<double> d_Rchol(n*n), d_R(n*n);

    double* pA = thrust::raw_pointer_cast(d_A.data());
    double* pB = thrust::raw_pointer_cast(d_B.data());

    int lw_ev = 0; double* d_wev = nullptr; int* d_iev = nullptr;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
        solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n, thrust::raw_pointer_cast(d_G.data()), n,
        thrust::raw_pointer_cast(d_eig.data()), &lw_ev));
    CUDA_CHECK(cudaMalloc(&d_wev, lw_ev * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_iev, sizeof(int)));

    int lw_chol = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        solver, CUBLAS_FILL_MODE_UPPER, n,
        thrust::raw_pointer_cast(d_G.data()), n, &lw_chol));
    thrust::device_vector<double> d_work_chol(lw_chol);

    auto one_run = [&]() {
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());

        // SVQB pass 1.
                // Cholesky QR pass.
                    svqb_pass(
            blas, solver, pA, pB, m, n,
            thrust::raw_pointer_cast(d_G.data()),
            thrust::raw_pointer_cast(d_eig.data()),
            thrust::raw_pointer_cast(d_Dinv.data()),
            thrust::raw_pointer_cast(d_rsqrt.data()),
            thrust::raw_pointer_cast(d_M.data()),
            d_wev, lw_ev, d_iev);

        cholqr_pass(
            blas, solver, pB, pA, m, n,
            thrust::raw_pointer_cast(d_G.data()),
            thrust::raw_pointer_cast(d_Rchol.data()),
            thrust::raw_pointer_cast(d_work_chol.data()),
            lw_chol, thrust::raw_pointer_cast(d_iev));
    

    };

    one_run();
    CUDA_CHECK(cudaDeviceSynchronize());

    EV wall; wall.start();
    for (int t = 0; t < trials; ++t) one_run();
    CUDA_CHECK(cudaDeviceSynchronize());
    time_s = wall.stop() / trials;

    // R for the original A. This also avoids having to explicitly invert M.
    CUBLAS_CHECK(cublasDgemm(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
        &one, pA, m, thrust::raw_pointer_cast(d_Abak.data()), m,
        &zero, thrust::raw_pointer_cast(d_R.data()), n));

    thrust::copy(d_A.begin(), d_A.end(), h_Q);
    thrust::copy(d_R.begin(), d_R.end(), h_R);

    cudaFree(d_wev); cudaFree(d_iev);
    cublasDestroy(blas); cusolverDnDestroy(solver);
}

void svqb2(const double* h_A, int m, int n,
           double* h_Q, double* h_R, double& time_s, int trials)
{
    const double a = 1., b = 0.;

    cublasHandle_t     blas;   CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    // Upload A (NOT timed). d_A / d_B are the two live m×n ping-pong buffers.
    thrust::device_vector<double> d_A(h_A, h_A + (long long)m * n);
    thrust::device_vector<double> d_Abak(d_A);
    thrust::device_vector<double> d_B((long long)m * n);

    thrust::device_vector<double> d_G(n * n), d_M(n * n);
    thrust::device_vector<double> d_eig(n), d_Dinv(n), d_rsqrt(n);
    thrust::device_vector<double> d_R1(n * n), d_R2(n * n), d_R(n * n);

    double* pA    = thrust::raw_pointer_cast(d_A.data());
    double* pB    = thrust::raw_pointer_cast(d_B.data());
    double* pG    = thrust::raw_pointer_cast(d_G.data());
    double* pM    = thrust::raw_pointer_cast(d_M.data());
    double* peig  = thrust::raw_pointer_cast(d_eig.data());
    double* pDinv = thrust::raw_pointer_cast(d_Dinv.data());
    double* prsqrt= thrust::raw_pointer_cast(d_rsqrt.data());
    double* pR1   = thrust::raw_pointer_cast(d_R1.data());
    double* pR2   = thrust::raw_pointer_cast(d_R2.data());
    double* pR    = thrust::raw_pointer_cast(d_R.data());

    int lw_ev = 0; double* d_wev = nullptr; int* d_iev = nullptr;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER, n, pG, n, peig, &lw_ev));
    CUDA_CHECK(cudaMalloc(&d_wev, lw_ev * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_iev, sizeof(int)));

    double* pAbak = thrust::raw_pointer_cast(d_Abak.data());

    auto one = [&]() {
        pA = thrust::raw_pointer_cast(d_A.data());
        pB = thrust::raw_pointer_cast(d_B.data());

        // Pass 1: A -> B  (B becomes Q1)
        svqb_pass(blas, solver, pA, pB, m, n, pG, peig, pDinv, prsqrt, pM, d_wev, lw_ev, d_iev);
        // Pass 2 (reorthogonalization): B -> A  (A becomes Q)
        svqb_pass(blas, solver, pB, pA, m, n, pG, peig, pDinv, prsqrt, pM, d_wev, lw_ev, d_iev);


    };

    // Warmup (NOT timed)
    thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
    one(); CUDA_CHECK(cudaDeviceSynchronize());

    // Timed trials (host<->device copies excluded; R recovery included, see above)
    EV wall; wall.start();
    for (int t = 0; t < trials; ++t) {
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
        one();
    }
    time_s = wall.stop() / trials;

            // R recovery — TIMED (see "Timing honesty note" above): unlike CholQR2's
        // Cholesky factors, these are not free byproducts of the passes above.
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                                 &a, pB, m, pAbak, m, &b, pR1, n));   // R1 = Q1^T * A0
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                                 &a, pA, m, pB, m, &b, pR2, n));      // R2 = Q^T * Q1
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &a, pR2, n, pR1, n, &b, pR, n));     // R = R2 * R1

    // Download Q and R from the last trial (NOT timed — host<->device copy only)
    pA = thrust::raw_pointer_cast(d_A.data());
    thrust::copy(d_A.begin(), d_A.begin() + (long long)m * n, h_Q);
    thrust::copy(d_R.begin(), d_R.end(), h_R);

    CUDA_CHECK(cudaFree(d_wev)); CUDA_CHECK(cudaFree(d_iev));
    cublasDestroy(blas); cusolverDnDestroy(solver);
}

void svqb3(const double* h_A, int m, int n,
           double* h_Q, double* h_R, double& time_s, int trials)
{
    const double a = 1., b = 0.;

    cublasHandle_t     blas;   CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    // Upload A (NOT timed). d_A / d_B are the two live m×n ping-pong buffers.
    thrust::device_vector<double> d_A(h_A, h_A + (long long)m * n);
    thrust::device_vector<double> d_Abak(d_A);
    thrust::device_vector<double> d_B((long long)m * n);
     thrust::device_vector<double> d_C((long long)m * n);

    thrust::device_vector<double> d_G(n * n), d_M(n * n);
    thrust::device_vector<double> d_eig(n), d_Dinv(n), d_rsqrt(n);
    thrust::device_vector<double> d_R1(n * n), d_R2(n * n), d_R3(n * n), d_R4(n * n), d_R(n * n);

    double* pA    = thrust::raw_pointer_cast(d_A.data());
    double* pB    = thrust::raw_pointer_cast(d_B.data());
     double* pC    = thrust::raw_pointer_cast(d_C.data());
    double* pG    = thrust::raw_pointer_cast(d_G.data());
    double* pM    = thrust::raw_pointer_cast(d_M.data());
    double* peig  = thrust::raw_pointer_cast(d_eig.data());
    double* pDinv = thrust::raw_pointer_cast(d_Dinv.data());
    double* prsqrt= thrust::raw_pointer_cast(d_rsqrt.data());
    double* pR1   = thrust::raw_pointer_cast(d_R1.data());
    double* pR2   = thrust::raw_pointer_cast(d_R2.data());
    double* pR3   = thrust::raw_pointer_cast(d_R3.data());
    double* pR4   = thrust::raw_pointer_cast(d_R4.data());
    double* pR    = thrust::raw_pointer_cast(d_R.data());

    int lw_ev = 0; double* d_wev = nullptr; int* d_iev = nullptr;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER, n, pG, n, peig, &lw_ev));
    CUDA_CHECK(cudaMalloc(&d_wev, lw_ev * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_iev, sizeof(int)));

    double* pAbak = thrust::raw_pointer_cast(d_Abak.data());

    auto one = [&]() {
        pA = thrust::raw_pointer_cast(d_A.data());
        pB = thrust::raw_pointer_cast(d_B.data());
        pC = thrust::raw_pointer_cast(d_C.data());

        // Pass 1: A -> B  (B becomes Q1)
        svqb_pass(blas, solver, pA, pB, m, n, pG, peig, pDinv, prsqrt, pM, d_wev, lw_ev, d_iev);
        // Pass 2 (reorthogonalization): B -> A  (A becomes Q)
        svqb_pass(blas, solver, pB, pC, m, n, pG, peig, pDinv, prsqrt, pM, d_wev, lw_ev, d_iev);
        svqb_pass(blas, solver, pC, pA, m, n, pG, peig, pDinv, prsqrt, pM, d_wev, lw_ev, d_iev);


    };

    // Warmup (NOT timed)
    thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
    one(); CUDA_CHECK(cudaDeviceSynchronize());

    // Timed trials (host<->device copies excluded; R recovery included, see above)
    EV wall; wall.start();
    for (int t = 0; t < trials; ++t) {
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
        one();
    }
    time_s = wall.stop() / trials;


            CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                                 &a, pC, m, pAbak, m, &b, pR1, n));   // R1 = Q2^T * A0

        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                                 &a, pB, m, pC, m, &b, pR2, n));   // R2 = Q1^T * A0
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                                 &a, pA, m, pB, m, &b, pR3, n));      // R3 = Q^T * Q1
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &a, pR3, n, pR2, n, &b, pR4, n));     // R4 = R3 * R2
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &a, pR4, n, pR1, n, &b, pR, n));     // R = R4 * R1

    // Download Q and R from the last trial (NOT timed — host<->device copy only)
    pA = thrust::raw_pointer_cast(d_A.data());
    thrust::copy(d_A.begin(), d_A.begin() + (long long)m * n, h_Q);
    thrust::copy(d_R.begin(), d_R.end(), h_R);

    CUDA_CHECK(cudaFree(d_wev)); CUDA_CHECK(cudaFree(d_iev));
    cublasDestroy(blas); cusolverDnDestroy(solver);
}
