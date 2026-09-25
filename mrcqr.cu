// mrcqr.cu
// MRCQR library: FP64 / FP32 / FP16 sketch variants.
// See mrcqr.hpp for API documentation.

#include "mrcqr.hpp"
#include "common.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <cmath>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Random functors (GPU-side, seeded per element for reproducibility)
// ---------------------------------------------------------------------------
struct RandUnif {
    __host__ __device__ double operator()(unsigned i) const {
        thrust::default_random_engine r(i * 2654435761u);
        thrust::uniform_real_distribution<double> d(-1, 1);
        r.discard(i); return d(r);
    }
};
struct RandRadI8 {
    __host__ __device__ int8_t operator()(unsigned i) const {
        thrust::default_random_engine r(i * 2654435761u);
        thrust::uniform_real_distribution<float> d(0, 1);
        r.discard(i); return d(r) < .5f ? (int8_t)-1 : (int8_t)1;
    }
};
struct RandSamp {
    int N;
    __host__ __device__ int operator()(unsigned i) const {
        thrust::default_random_engine r(i * 2246822519u);
        thrust::uniform_int_distribution<int> d(0, N - 1);
        r.discard(i); return d(r);
    }
};

// ---------------------------------------------------------------------------
// Diagonal multiply + type-conversion kernels (fused sign flip + cast)
// ---------------------------------------------------------------------------
__global__ void diag_fp64(const double* A, const int8_t* D, double* FA, int m, int n) {
    for (long long i = blockIdx.x*blockDim.x+threadIdx.x; i < (long long)m*n; i += blockDim.x*gridDim.x) {
        unsigned long long b = __double_as_longlong(A[i]);
        b ^= (unsigned long long)(D[i % m] < 0) << 63;
        FA[i] = __longlong_as_double(b);
    }
}
__global__ void diag_fp32(const double* A, const int8_t* D, float* FA, int m, int n) {
    for (long long i = blockIdx.x*blockDim.x+threadIdx.x; i < (long long)m*n; i += blockDim.x*gridDim.x) {
        unsigned b = __float_as_uint((float)A[i]);
        b ^= (unsigned)(D[i % m] < 0) << 31;
        FA[i] = __uint_as_float(b);
    }
}
__global__ void diag_fp16(const double* A, const int8_t* D, __half* FA, int m, int n) {
    for (long long i = blockIdx.x*blockDim.x+threadIdx.x; i < (long long)m*n; i += blockDim.x*gridDim.x) {
        unsigned short b = __half_as_ushort(__float2half((float)A[i]));
        b ^= (unsigned short)(D[i % m] < 0) << 15;
        FA[i] = __ushort_as_half(b);
    }
}

__global__ void print_matrix_small(
    const double* A, int m, int n,
    int max_rows, int max_cols)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    int rows = min(m, max_rows);
    int cols = min(n, max_cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.16e ", A[i + j*m]);
        }
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// Row-selection functors.
//
// Re+Im (default, re_only=false): c frequencies → 2c rows.
//   Rows 0..c-1   : real parts  of fft[idx[row]]
//   Rows c..2c-1  : imaginary parts of the same c frequencies (free extra rows).
//   SketchQR runs on a 2c×n matrix.
//
// Re-only (re_only=true): c frequencies → c rows (real parts only).
//   SketchQR runs on a c×n matrix — roughly half the cost of Re+Im.
// ---------------------------------------------------------------------------
struct SelF64 {      // Re+Im
    const cufftDoubleComplex* fft; const int* idx; int c, m, rpc;
    __host__ __device__ double operator()(int i) const {
        int col=i/(2*c), lr=i%(2*c), r=idx[lr%c], ar=(r<=m/2)?r:(m-r);
        auto v = fft[ar + col*rpc]; return (lr < c) ? v.x : v.y;
    }
};
struct SelF32 {      // Re+Im
    const cufftComplex* fft; const int* idx; int c, m, rpc;
    __host__ __device__ float operator()(int i) const {
        int col=i/(2*c), lr=i%(2*c), r=idx[lr%c], ar=(r<=m/2)?r:(m-r);
        auto v = fft[ar + col*rpc]; return (lr < c) ? v.x : v.y;
    }
};
struct SelF16 {      // Re+Im
    const __half2* fft; const int* idx; int c, m, rpc;
    __host__ __device__ float operator()(int i) const {
        int col=i/(2*c), lr=i%(2*c), r=idx[lr%c], ar=(r<=m/2)?r:(m-r);
        float2 v = __half22float2(fft[ar + col*rpc]);
        return (lr < c) ? v.x : v.y;
    }
};
struct SelF64_Re {   // Re-only
    const cufftDoubleComplex* fft; const int* idx; int c, m, rpc;
    __host__ __device__ double operator()(int i) const {
        int col=i/c, r=idx[i%c], ar=(r<=m/2)?r:(m-r);
        return fft[ar + col*rpc].x;
    }
};
struct SelF32_Re {   // Re-only
    const cufftComplex* fft; const int* idx; int c, m, rpc;
    __host__ __device__ float operator()(int i) const {
        int col=i/c, r=idx[i%c], ar=(r<=m/2)?r:(m-r);
        return fft[ar + col*rpc].x;
    }
};
struct SelF16_Re {   // Re-only
    const __half2* fft; const int* idx; int c, m, rpc;
    __host__ __device__ float operator()(int i) const {
        int col=i/c, r=idx[i%c], ar=(r<=m/2)?r:(m-r);
        return __half22float2(fft[ar + col*rpc]).x;
    }
};

// ---------------------------------------------------------------------------
// Upper-triangle copy functors: extract R from sketch QR result.
//   TriD64:      double 2c×n source  → double n×n dest
//   TriF32toF64: float  2c×n source  → double n×n dest  (FP32/FP16 variants)
//   TriNN:       double n×n  source  → double n×n dest  (Cholesky factor)
// ---------------------------------------------------------------------------
struct TriD64 {
    int c, n; const double* src; double* dst;
    __host__ __device__ void operator()(int x) const {
        int row=x/n, col=x%n;
        if (row <= col && row < n) dst[row + n*col] = src[row + c*col];
    }
};
struct TriF32toF64 {
    int c, n; const float* src; double* dst;
    __host__ __device__ void operator()(int x) const {
        int row=x/n, col=x%n;
        if (row <= col && row < n) dst[row + n*col] = (double)src[row + c*col];
    }
};
struct TriNN {
    int n; const double* src; double* dst;
    __host__ __device__ void operator()(int x) const {
        int row=x/n, col=x%n;
        if (row <= col) dst[row + n*col] = src[row + n*col];
    }
};

// ---------------------------------------------------------------------------
// Shared CholQR phase (FP64, SIDE_RIGHT in-place TRSM — no transposes).
// On entry:  d_A holds the preconditioned matrix Q₀  (m×n).
// On exit:   d_A holds Q.   d_G holds Cholesky factor R_c (n×n, upper tri).
// ---------------------------------------------------------------------------
static void cholqr_phase(cublasHandle_t blas, cusolverDnHandle_t solver,
                         double* pA, double* pG,
                         double* d_wch, int lw_ch, int* d_ich,
                         int m, int n)
{
    const double a = 1., b = 0.;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    // TrsmPre: A := A · R_s⁻¹  (in-place, SIDE_RIGHT)
    CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, m, n, &a,
                             pG, n, pA, m));  // pG holds R_s here

    // GEMM: G = Q₀ᵀ Q₀
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &a, pA, m, pA, m, &b, pG, n));

    // Cholesky: G = R_cᵀ R_c
    CUSOLVER_CHECK(cusolverDnDpotrf(solver, uplo, n, pG, n, d_wch, lw_ch, d_ich));

    // TrsmFin: A := A · R_c⁻¹  (in-place, SIDE_RIGHT)
    CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, m, n, &a,
                             pG, n, pA, m));
}

static void cholqr_phase_nosolve(cublasHandle_t blas, cusolverDnHandle_t solver,
                         double* pA, double* pG,
                         double* d_wch, int lw_ch, int* d_ich,
                         int m, int n)
{
    const double a = 1., b = 0.;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    // // TrsmPre: A := A · R_s⁻¹  (in-place, SIDE_RIGHT)
    // CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_N,
    //                          CUBLAS_DIAG_NON_UNIT, m, n, &a,
    //                          pG, n, pA, m));  // pG holds R_s here

    // GEMM: G = Q₀ᵀ Q₀
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &a, pA, m, pA, m, &b, pG, n));

    // Cholesky: G = R_cᵀ R_c
    CUSOLVER_CHECK(cusolverDnDpotrf(solver, uplo, n, pG, n, d_wch, lw_ch, d_ich));

    // TrsmFin: A := A · R_c⁻¹  (in-place, SIDE_RIGHT)
    CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, m, n, &a,
                             pG, n, pA, m));
}

// ---------------------------------------------------------------------------
// Compute full R = R_c · R_s  (not timed, tiny n×n work).
// Extracts upper triangles of both factors before multiplying.
// ---------------------------------------------------------------------------
static void compute_R(cublasHandle_t blas,
                      const double* pG,    // n×n, R_c in upper tri
                      const double* pRs,   // n×n, R_s (already clean upper tri)
                      double* h_R, int n)
{
    const double a = 1., b = 0.;
    thrust::device_vector<double> d_Rc(n*n, 0.);
    thrust::device_vector<double> d_Rfull(n*n);
    double* pRc    = thrust::raw_pointer_cast(d_Rc.data());
    double* pRfull = thrust::raw_pointer_cast(d_Rfull.data());

    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n*n),
                     TriNN{n, pG, pRc});

    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                             &a, pRc, n, pRs, n, &b, pRfull, n));
    thrust::copy(d_Rfull.begin(), d_Rfull.end(), h_R);
}

static void compute_B(cublasHandle_t blas,
                      const double* pG,    // n×n, R_c in upper tri
                      const double* pVT,   // n×n, V (already clean upper tri)
                      const double * pSigma, // n×1, singular values
                      double* h_R, int n)
{
    const double a = 1., b = 0.;
    thrust::device_vector<double> d_Rc(n*n, 0.);
    thrust::device_vector<double> d_Rfull(n*n);
    double* pRc    = thrust::raw_pointer_cast(d_Rc.data());
    double* pRfull = thrust::raw_pointer_cast(d_Rfull.data());

    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n*n),
                     TriNN{n, pG, pRc});

    thrust::transform(
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(n*n),
    d_Rc.begin(),
    [pRc, pSigma, n] __device__(int k) {
        const int j = k / n;
        return pRc[k] * pSigma[j];
    });
    
    

    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                             &a, pRc, n, pVT, n, &b, pRfull, n));
    thrust::copy(d_Rfull.begin(), d_Rfull.end(), h_R);
}

// ============================================================================
// mrcqr_fp64 — FP64 sketch (cufftExecD2Z, Dgeqrf)
// ============================================================================
void mrcqr_fp64(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s, int trials, bool re_only)
{
    const int half_r = m/2 + 1;
    const int sketch_rows = re_only ? c : 2*c;  // c×n or 2c×n sketch for QR
    const double norm = std::sqrt((double)c / m);
    const int BLK = 256, GRD = (int)std::min(65535LL, ((long long)m*n + BLK - 1) / BLK);

    cublasHandle_t    blas;   CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    // Upload A (NOT timed)
    thrust::device_vector<double> d_A(h_A, h_A + (long long)m*n);
    thrust::device_vector<double> d_Abak(d_A);

    // Working buffers
    thrust::device_vector<int8_t>             d_D(m);
    thrust::device_vector<int>                d_S(c);
    thrust::device_vector<double>             d_FA((long long)m*n);
    thrust::device_vector<cufftDoubleComplex> d_fft((long long)n * half_r);
    thrust::device_vector<double>             d_sk((long long)sketch_rows*n);
    thrust::device_vector<double>             d_RsTR(n*n, 0.);
    thrust::device_vector<double>             d_G(n*n);

    double* pA   = thrust::raw_pointer_cast(d_A.data());
    int8_t* pD   = thrust::raw_pointer_cast(d_D.data());
    int*    pS   = thrust::raw_pointer_cast(d_S.data());
    double* pFA  = thrust::raw_pointer_cast(d_FA.data());
    auto*   pFft = thrust::raw_pointer_cast(d_fft.data());
    double* pSk  = thrust::raw_pointer_cast(d_sk.data());
    double* pRs  = thrust::raw_pointer_cast(d_RsTR.data());
    double* pG   = thrust::raw_pointer_cast(d_G.data());

    // FFT plan
    cufftHandle fp; int fn[1] = {m};
    CUFFT_CHECK(cufftPlanMany(&fp, 1, fn, fn, 1, m, fn, 1, half_r, CUFFT_D2Z, n));

    // Sketch QR workspace
    int lw_sk = 0; double* d_tau = nullptr; double* d_wsk = nullptr; int* d_isk = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau, n * sizeof(double)));
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver, sketch_rows, n, pSk, sketch_rows, &lw_sk));
    CUDA_CHECK(cudaMalloc(&d_wsk, lw_sk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isk, sizeof(int)));

    // CholQR workspace
    int lw_ch = 0; double* d_wch = nullptr; int* d_ich = nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                               n, pG, n, &lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch, lw_ch * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich, sizeof(int)));

    auto one = [&]() {
        pA = thrust::raw_pointer_cast(d_A.data());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m), d_D.begin(), RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c), d_S.begin(), RandSamp{m});

        diag_fp64<<<GRD, BLK>>>(pA, pD, pFA, m, n);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecD2Z(fp, pFA, pFft));

        if (re_only)
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows*n),
                              d_sk.begin(), SelF64_Re{pFft, pS, c, m, half_r});
        else
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows*n),
                              d_sk.begin(), SelF64{pFft, pS, c, m, half_r});
        thrust::transform(d_sk.begin(), d_sk.end(), d_sk.begin(),
                          [norm] __device__(double x) { return x * norm; });

        CUSOLVER_CHECK(cusolverDnDgeqrf(solver, sketch_rows, n, pSk, sketch_rows,
                                        d_tau, d_wsk, lw_sk, d_isk));

        thrust::fill(d_RsTR.begin(), d_RsTR.end(), 0.);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(sketch_rows*n),
                         TriD64{sketch_rows, n, pSk, pRs});

        // pG temporarily holds R_s for TrsmPre; cholqr_phase overwrites pG with R_c
        thrust::copy(d_RsTR.begin(), d_RsTR.end(), d_G.begin());
        cholqr_phase(blas, solver, pA, pG, d_wch, lw_ch, d_ich, m, n);
    };

    // Warmup (NOT timed)
    thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
    one(); CUDA_CHECK(cudaDeviceSynchronize());

    // Timed trials (host↔device copies excluded)
    EV wall; wall.start();
    for (int t = 0; t < trials; ++t) {
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
        one();
    }
    time_s = wall.stop() / trials;

    // Compute R = R_c · R_s (NOT timed)
    compute_R(blas, pG, pRs, h_R, n);

    // Download Q (NOT timed)
    pA = thrust::raw_pointer_cast(d_A.data());
    thrust::copy(d_A.begin(), d_A.begin() + (long long)m*n, h_Q);

    cufftDestroy(fp);
    cudaFree(d_tau); cudaFree(d_wsk); cudaFree(d_isk);
    cudaFree(d_wch); cudaFree(d_ich);
    cublasDestroy(blas); cusolverDnDestroy(solver);
}

static void svqb_pass(cublasHandle_t blas, cusolverDnHandle_t solver,
                       const double* pXin, double* pXout, int m, int n,
                       double* pG, double* peig, double* pDinv, double* prsqrt, double* pM,
                       double* d_wev, int lw_ev, int* d_iev);

void mreig_svqb(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s, int trials, bool re_only)
{
    const int half_r = m / 2 + 1;
    const int sketch_rows = re_only ? c : 2 * c;
    const double norm = std::sqrt((double)c / m);
    const int BLK = 256;
    const int GRD = (int)std::min(65535LL, ((long long)m * n + BLK - 1) / BLK);
    const double one = 1.0, zero = 0.0;

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    thrust::device_vector<double> d_A(h_A, h_A + (long long)m * n);
    thrust::device_vector<double> d_Abak(d_A);
    thrust::device_vector<int8_t> d_D(m);
    thrust::device_vector<int> d_S(c);
    thrust::device_vector<double> d_FA((long long)m * n);
    thrust::device_vector<cufftDoubleComplex> d_fft((long long)n * half_r);
    thrust::device_vector<double> d_sk((long long)sketch_rows * n);
    thrust::device_vector<double> d_skG(n * n);
    thrust::device_vector<double> d_inv_s(n);
    thrust::device_vector<double> d_pre(n * n);
    thrust::device_vector<double> d_A1((long long)m * n);
    thrust::device_vector<double> d_B((long long)m * n);
    thrust::device_vector<double> d_G(n * n), d_M(n * n);
    thrust::device_vector<double> d_eig(n), d_Dinv(n), d_rsqrt(n);

    double* pA = thrust::raw_pointer_cast(d_A.data());
    double* pAbak = thrust::raw_pointer_cast(d_Abak.data());
    int8_t* pD = thrust::raw_pointer_cast(d_D.data());
    int* pS = thrust::raw_pointer_cast(d_S.data());
    double* pFA = thrust::raw_pointer_cast(d_FA.data());
    auto* pFft = thrust::raw_pointer_cast(d_fft.data());
    double* pSk = thrust::raw_pointer_cast(d_sk.data());
    double* pSkG = thrust::raw_pointer_cast(d_skG.data());
    double* pInvS = thrust::raw_pointer_cast(d_inv_s.data());
    double* pPre = thrust::raw_pointer_cast(d_pre.data());
    double* pA1 = thrust::raw_pointer_cast(d_A1.data());
    double* pB = thrust::raw_pointer_cast(d_B.data());
    double* pG = thrust::raw_pointer_cast(d_G.data());
    double* pM = thrust::raw_pointer_cast(d_M.data());
    double* pEig = thrust::raw_pointer_cast(d_eig.data());
    double* pDinv = thrust::raw_pointer_cast(d_Dinv.data());
    double* pRsqrt = thrust::raw_pointer_cast(d_rsqrt.data());

    cufftHandle fp;
    int fn[1] = {m};
    CUFFT_CHECK(cufftPlanMany(&fp, 1, fn, fn, 1, m,
                              fn, 1, half_r, CUFFT_D2Z, n));

    int lw_ev = 0;
    double* d_wev = nullptr;
    int* d_iev = nullptr;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
        solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n, pSkG, n, pEig, &lw_ev));
    CUDA_CHECK(cudaMalloc(&d_wev, lw_ev * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_iev, sizeof(int)));

    auto one_trial = [&]() {
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
        pA = thrust::raw_pointer_cast(d_A.data());

        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m),
                          d_D.begin(), RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c),
                          d_S.begin(), RandSamp{m});

        diag_fp64<<<GRD, BLK>>>(pA, pD, pFA, m, n);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecD2Z(fp, pFA, pFft));

        if (re_only) {
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows * n),
                              d_sk.begin(), SelF64_Re{pFft, pS, c, m, half_r});
        } else {
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows * n),
                              d_sk.begin(), SelF64{pFft, pS, c, m, half_r});
        }
        thrust::transform(d_sk.begin(), d_sk.end(), d_sk.begin(),
                          [norm] __device__(double x) { return x * norm; });

        // Gs = As^T As; Gs = V Lambda V^T.
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 n, n, sketch_rows, &one,
                                 pSk, sketch_rows, pSk, sketch_rows,
                                 &zero, pSkG, n));
        CUSOLVER_CHECK(cusolverDnDsyevd(
            solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            n, pSkG, n, pEig, d_wev, lw_ev, d_iev));

        double maxeig;
        CUDA_CHECK(cudaMemcpy(&maxeig, pEig + (n - 1), sizeof(double),
                              cudaMemcpyDeviceToHost));
        thrust::transform(thrust::device_pointer_cast(pEig),
                          thrust::device_pointer_cast(pEig) + n,
                          thrust::device_pointer_cast(pInvS),
                          [maxeig] __device__(double value) {
                              double floor_ = maxeig * 1e-15;
                              return 1.0 / sqrt(value < floor_ ? floor_ : value);
                          });
        CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_RIGHT, n, n,
                                 pSkG, n, pInvS, 1, pPre, n));

        // A1 = A V Lambda^(-1/2).
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, n, &one, pA, m, pPre, n,
                                 &zero, pA1, m));

        // Final orthogonalization uses the established SVQB implementation.
        svqb_pass(blas, solver, pA1, pB, m, n, pG, pEig, pDinv,
                  pRsqrt, pM, d_wev, lw_ev, d_iev);
    };

    one_trial();
    CUDA_CHECK(cudaDeviceSynchronize());
    EV wall;
    wall.start();
    for (int trial = 0; trial < trials; ++trial)
        one_trial();
    time_s = wall.stop() / trials;

    // B recovery is excluded from timing, consistent with mrsqr2.
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                             n, n, m, &one, pB, m, pAbak, m,
                             &zero, pG, n));
    thrust::copy(d_G.begin(), d_G.end(), h_R);
    thrust::copy(d_B.begin(), d_B.end(), h_Q);

    cufftDestroy(fp);
    cudaFree(d_wev);
    cudaFree(d_iev);
    cublasDestroy(blas);
    cusolverDnDestroy(solver);
}

void mreig2_svqb(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s, int trials, bool re_only)
{
    const int half_r = m / 2 + 1;
    const int sketch_rows = re_only ? c : 2 * c;
    const double norm = std::sqrt((double)c / m);
    const int BLK = 256;
    const int GRD = (int)std::min(65535LL, ((long long)m * n + BLK - 1) / BLK);
    const double one = 1.0, zero = 0.0;

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    thrust::device_vector<double> d_A(h_A, h_A + (long long)m * n);
    thrust::device_vector<double> d_Abak(d_A);
    thrust::device_vector<int8_t> d_D(m);
    thrust::device_vector<int> d_S(c);
    thrust::device_vector<double> d_FA((long long)m * n);
    thrust::device_vector<cufftDoubleComplex> d_fft((long long)n * half_r);
    thrust::device_vector<double> d_sk((long long)sketch_rows * n);
    thrust::device_vector<double> d_C((long long)sketch_rows * n);
    thrust::device_vector<double> d_skG(n * n);
    thrust::device_vector<double> d_inv_s(n);
    thrust::device_vector<double> d_pre(n * n);
    thrust::device_vector<double> d_A1((long long)m * n);
    thrust::device_vector<double> d_B((long long)m * n);
    thrust::device_vector<double> d_G(n * n), d_M(n * n),d_M1(n * n);
    thrust::device_vector<double> d_eig(n), d_Dinv(n), d_rsqrt(n);

    double* pA = thrust::raw_pointer_cast(d_A.data());
    double* pAbak = thrust::raw_pointer_cast(d_Abak.data());
    int8_t* pD = thrust::raw_pointer_cast(d_D.data());
    int* pS = thrust::raw_pointer_cast(d_S.data());
    double* pFA = thrust::raw_pointer_cast(d_FA.data());
    auto* pFft = thrust::raw_pointer_cast(d_fft.data());
    double* pSk = thrust::raw_pointer_cast(d_sk.data());
    double* pC = thrust::raw_pointer_cast(d_C.data());
    double* pSkG = thrust::raw_pointer_cast(d_skG.data());
    double* pInvS = thrust::raw_pointer_cast(d_inv_s.data());
    double* pPre = thrust::raw_pointer_cast(d_pre.data());
    double* pA1 = thrust::raw_pointer_cast(d_A1.data());
    double* pB = thrust::raw_pointer_cast(d_B.data());
    double* pG = thrust::raw_pointer_cast(d_G.data());
    double* pM = thrust::raw_pointer_cast(d_M.data());
    double* pM1 = thrust::raw_pointer_cast(d_M1.data());
    double* pEig = thrust::raw_pointer_cast(d_eig.data());
    double* pDinv = thrust::raw_pointer_cast(d_Dinv.data());
    double* pRsqrt = thrust::raw_pointer_cast(d_rsqrt.data());

    cufftHandle fp;
    int fn[1] = {m};
    CUFFT_CHECK(cufftPlanMany(&fp, 1, fn, fn, 1, m,
                              fn, 1, half_r, CUFFT_D2Z, n));

    int lw_ev = 0;
    double* d_wev = nullptr;
    int* d_iev = nullptr;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
        solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n, pSkG, n, pEig, &lw_ev));
    CUDA_CHECK(cudaMalloc(&d_wev, lw_ev * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_iev, sizeof(int)));

    auto one_trial = [&]() {
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
        pA = thrust::raw_pointer_cast(d_A.data());

        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m),
                          d_D.begin(), RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c),
                          d_S.begin(), RandSamp{m});

        diag_fp64<<<GRD, BLK>>>(pA, pD, pFA, m, n);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecD2Z(fp, pFA, pFft));

        if (re_only) {
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows * n),
                              d_sk.begin(), SelF64_Re{pFft, pS, c, m, half_r});
        } else {
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows * n),
                              d_sk.begin(), SelF64{pFft, pS, c, m, half_r});
        }
        thrust::transform(d_sk.begin(), d_sk.end(), d_sk.begin(),
                          [norm] __device__(double x) { return x * norm; });

        svqb_pass(blas, solver, pSk, pC, sketch_rows, n, pG, pEig, pDinv,
                  pRsqrt, pM, d_wev, lw_ev, d_iev);
       
        svqb_pass(blas, solver, pC, pSk, sketch_rows, n, pG, pEig, pDinv,
                  pRsqrt, pM1, d_wev, lw_ev, d_iev);
       


        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, n, n, &one, pM1, n, pM, n,
                                 &zero, pPre, n));
                                  
          // A1 = A V Lambda^(-1/2).
        CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, n, &one, pA, m, pPre, n,
                                 &zero, pA1, m));

          printf("done\n");
        svqb_pass(blas, solver, pA1, pB, m, n, pG, pEig, pDinv,
                  pRsqrt, pM, d_wev, lw_ev, d_iev);
             printf("done\n");


        // // Gs = As^T As; Gs = V Lambda V^T.
        // CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
        //                          n, n, sketch_rows, &one,
        //                          pSk, sketch_rows, pSk, sketch_rows,
        //                          &zero, pSkG, n));
        // CUSOLVER_CHECK(cusolverDnDsyevd(
        //     solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        //     n, pSkG, n, pEig, d_wev, lw_ev, d_iev));

        // double maxeig;
        // CUDA_CHECK(cudaMemcpy(&maxeig, pEig + (n - 1), sizeof(double),
        //                       cudaMemcpyDeviceToHost));
        // thrust::transform(thrust::device_pointer_cast(pEig),
        //                   thrust::device_pointer_cast(pEig) + n,
        //                   thrust::device_pointer_cast(pInvS),
        //                   [maxeig] __device__(double value) {
        //                       double floor_ = maxeig * 1e-15;
        //                       return 1.0 / sqrt(value < floor_ ? floor_ : value);
        //                   });
        // CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_RIGHT, n, n,
        //                          pSkG, n, pInvS, 1, pPre, n));

        // // A1 = A V Lambda^(-1/2).
        // CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
        //                          m, n, n, &one, pA, m, pPre, n,
        //                          &zero, pA1, m));

        // // Final orthogonalization uses the established SVQB implementation.
        // svqb_pass(blas, solver, pA1, pB, m, n, pG, pEig, pDinv,
        //           pRsqrt, pM, d_wev, lw_ev, d_iev);
    };

    one_trial();
     printf("done1\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    EV wall;
    wall.start();
    for (int trial = 0; trial < trials; ++trial)
        one_trial();
    time_s = wall.stop() / trials;

    // B recovery is excluded from timing, consistent with mrsqr2.
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                             n, n, m, &one, pB, m, pAbak, m,
                             &zero, pG, n)); 
    thrust::copy(d_G.begin(), d_G.end(), h_R);
    thrust::copy(d_B.begin(), d_B.end(), h_Q);

    cufftDestroy(fp);
    cudaFree(d_wev);
    cudaFree(d_iev);
    cublasDestroy(blas);
    cusolverDnDestroy(solver);
}



// ============================================================================
// mrcqr_fp32 — FP32 sketch (cufftExecR2C, Sgeqrf, float→double R_s)
// ============================================================================
void mrcqr_fp32(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s, int trials, bool re_only)
{
    const int half_r = m/2 + 1;
    const int sketch_rows = re_only ? c : 2*c;
    const float normf = (float)std::sqrt((double)c / m);
    const int BLK = 256, GRD = (int)std::min(65535LL, ((long long)m*n + BLK - 1) / BLK);

    cublasHandle_t    blas;   CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    thrust::device_vector<double> d_A(h_A, h_A + (long long)m*n);
    thrust::device_vector<double> d_Abak(d_A);

    thrust::device_vector<int8_t>    d_D(m);
    thrust::device_vector<int>       d_S(c);
    thrust::device_vector<float>     d_FA((long long)m*n);
    thrust::device_vector<cufftComplex> d_fft((long long)n * half_r);
    thrust::device_vector<float>     d_sk((long long)sketch_rows*n);
    thrust::device_vector<double>    d_RsTR(n*n, 0.);
    thrust::device_vector<double>    d_G(n*n);

    double* pA  = thrust::raw_pointer_cast(d_A.data());
    int8_t* pD  = thrust::raw_pointer_cast(d_D.data());
    int*    pS  = thrust::raw_pointer_cast(d_S.data());
    float*  pFA = thrust::raw_pointer_cast(d_FA.data());
    auto*   pFft= thrust::raw_pointer_cast(d_fft.data());
    float*  pSk = thrust::raw_pointer_cast(d_sk.data());
    double* pRs = thrust::raw_pointer_cast(d_RsTR.data());
    double* pG  = thrust::raw_pointer_cast(d_G.data());

    cufftHandle fp; int fn[1] = {m};
    CUFFT_CHECK(cufftPlanMany(&fp, 1, fn, fn, 1, m, fn, 1, half_r, CUFFT_R2C, n));

    int lw_sk = 0; float* d_tau = nullptr; float* d_wsk = nullptr; int* d_isk = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau, n * sizeof(float)));
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver, sketch_rows, n, pSk, sketch_rows, &lw_sk));
    CUDA_CHECK(cudaMalloc(&d_wsk, lw_sk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_isk, sizeof(int)));

    int lw_ch = 0; double* d_wch = nullptr; int* d_ich = nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                               n, pG, n, &lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch, lw_ch * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich, sizeof(int)));

    auto one = [&]() {
        pA = thrust::raw_pointer_cast(d_A.data());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m), d_D.begin(), RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c), d_S.begin(), RandSamp{m});

        diag_fp32<<<GRD, BLK>>>(pA, pD, pFA, m, n);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecR2C(fp, pFA, pFft));

        if (re_only)
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows*n),
                              d_sk.begin(), SelF32_Re{pFft, pS, c, m, half_r});
        else
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows*n),
                              d_sk.begin(), SelF32{pFft, pS, c, m, half_r});
        thrust::transform(d_sk.begin(), d_sk.end(), d_sk.begin(),
                          [normf] __device__(float x) { return x * normf; });

        CUSOLVER_CHECK(cusolverDnSgeqrf(solver, sketch_rows, n, pSk, sketch_rows,
                                        d_tau, d_wsk, lw_sk, d_isk));

        thrust::fill(d_RsTR.begin(), d_RsTR.end(), 0.);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(sketch_rows*n),
                         TriF32toF64{sketch_rows, n, pSk, pRs});

        thrust::copy(d_RsTR.begin(), d_RsTR.end(), d_G.begin());
        cholqr_phase(blas, solver, pA, pG, d_wch, lw_ch, d_ich, m, n);
    };

    thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
    one(); CUDA_CHECK(cudaDeviceSynchronize());

    EV wall; wall.start();
    for (int t = 0; t < trials; ++t) {
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
        one();
    }
    time_s = wall.stop() / trials;

    compute_R(blas, pG, pRs, h_R, n);
    pA = thrust::raw_pointer_cast(d_A.data());
    thrust::copy(d_A.begin(), d_A.begin() + (long long)m*n, h_Q);

    cufftDestroy(fp);
    cudaFree(d_tau); cudaFree(d_wsk); cudaFree(d_isk);
    cudaFree(d_wch); cudaFree(d_ich);
    cublasDestroy(blas); cusolverDnDestroy(solver);
}

// ============================================================================
// mrcqr_fp16 — FP16 sketch (cufftXt FP16 R2C, Sgeqrf on promoted float sketch)
// ============================================================================
void mrcqr_fp16(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s, int trials, bool re_only)
{
    const int half_r = m/2 + 1;
    const int sketch_rows = re_only ? c : 2*c;
    const float normf = (float)std::sqrt((double)c / m);
    const int BLK = 256, GRD = (int)std::min(65535LL, ((long long)m*n + BLK - 1) / BLK);

    cublasHandle_t    blas;   CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    thrust::device_vector<double> d_A(h_A, h_A + (long long)m*n);
    thrust::device_vector<double> d_Abak(d_A);

    thrust::device_vector<int8_t>  d_D(m);
    thrust::device_vector<int>     d_S(c);
    thrust::device_vector<__half>  d_FA((long long)m*n);
    thrust::device_vector<__half2> d_fft((long long)n * half_r);
    thrust::device_vector<float>   d_sk((long long)sketch_rows*n); // promoted from FP16 for Sgeqrf
    thrust::device_vector<double>  d_RsTR(n*n, 0.);
    thrust::device_vector<double>  d_G(n*n);

    double* pA   = thrust::raw_pointer_cast(d_A.data());
    int8_t* pD   = thrust::raw_pointer_cast(d_D.data());
    int*    pS   = thrust::raw_pointer_cast(d_S.data());
    __half* pFA  = thrust::raw_pointer_cast(d_FA.data());
    __half2*pFft = thrust::raw_pointer_cast(d_fft.data());
    float*  pSk  = thrust::raw_pointer_cast(d_sk.data());
    double* pRs  = thrust::raw_pointer_cast(d_RsTR.data());
    double* pG   = thrust::raw_pointer_cast(d_G.data());

    // cufftXt plan: FP16 real → FP16 complex
    cufftHandle fp; CUFFT_CHECK(cufftCreate(&fp));
    long long n1[1] = {m}; size_t workSize = 0;
    CUFFT_CHECK(cufftXtMakePlanMany(fp, 1, n1,
                                    n1, 1, m, CUDA_R_16F,
                                    n1, 1, half_r, CUDA_C_16F,
                                    n, &workSize, CUDA_R_16F));

    int lw_sk = 0; float* d_tau = nullptr; float* d_wsk = nullptr; int* d_isk = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau, n * sizeof(float)));
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver, sketch_rows, n, pSk, sketch_rows, &lw_sk));
    CUDA_CHECK(cudaMalloc(&d_wsk, lw_sk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_isk, sizeof(int)));

    int lw_ch = 0; double* d_wch = nullptr; int* d_ich = nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                               n, pG, n, &lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch, lw_ch * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich, sizeof(int)));

    auto one = [&]() {
        pA = thrust::raw_pointer_cast(d_A.data());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m), d_D.begin(), RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c), d_S.begin(), RandSamp{m});

        diag_fp16<<<GRD, BLK>>>(pA, pD, pFA, m, n);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftXtExec(fp, (void*)pFA, (void*)pFft, CUFFT_FORWARD));

        // Row-select: __half2 → float (cuSolver has no FP16 geqrf)
        if (re_only)
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows*n),
                              d_sk.begin(), SelF16_Re{pFft, pS, c, m, half_r});
        else
            thrust::transform(thrust::counting_iterator<int>(0),
                              thrust::counting_iterator<int>(sketch_rows*n),
                              d_sk.begin(), SelF16{pFft, pS, c, m, half_r});
        thrust::transform(d_sk.begin(), d_sk.end(), d_sk.begin(),
                          [normf] __device__(float x) { return x * normf; });

        CUSOLVER_CHECK(cusolverDnSgeqrf(solver, sketch_rows, n, pSk, sketch_rows,
                                        d_tau, d_wsk, lw_sk, d_isk));

        // Promote float R_s → double n×n
        thrust::fill(d_RsTR.begin(), d_RsTR.end(), 0.);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(sketch_rows*n),
                         TriF32toF64{sketch_rows, n, pSk, pRs});

        thrust::copy(d_RsTR.begin(), d_RsTR.end(), d_G.begin());
        cholqr_phase(blas, solver, pA, pG, d_wch, lw_ch, d_ich, m, n);
    };

    thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
    one(); CUDA_CHECK(cudaDeviceSynchronize());

    EV wall; wall.start();
    for (int t = 0; t < trials; ++t) {
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
        one();
    }
    time_s = wall.stop() / trials;

    compute_R(blas, pG, pRs, h_R, n);
    pA = thrust::raw_pointer_cast(d_A.data());
    thrust::copy(d_A.begin(), d_A.begin() + (long long)m*n, h_Q);

    cufftDestroy(fp);
    cudaFree(d_tau); cudaFree(d_wsk); cudaFree(d_isk);
    cudaFree(d_wch); cudaFree(d_ich);
    cublasDestroy(blas); cusolverDnDestroy(solver);
}


void mrsqr_fp64(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s, int trials, bool re_only)
{
    const int half_r = m/2 + 1;
    const int sketch_rows = re_only ? c : 2*c;
    const double norm = std::sqrt((double)c / m);

    const int BLK = 256;
    const int GRD = (int)std::min(
        65535LL, ((long long)m*n + BLK - 1) / BLK);

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));

    cusolverDnHandle_t solver;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    // Upload A (NOT timed)
    thrust::device_vector<double> d_A(h_A, h_A + (long long)m*n);
    thrust::device_vector<double> d_Abak(d_A);

    // Working buffers
    thrust::device_vector<int8_t> d_D(m);
    thrust::device_vector<int> d_S(c);

    thrust::device_vector<double> d_FA((long long)m*n);
    thrust::device_vector<cufftDoubleComplex> d_fft(
        (long long)n * half_r);

    thrust::device_vector<double> d_sk(
        (long long)sketch_rows*n);

    thrust::device_vector<double> d_G(n*n);

    // A1 = A * V * Sigma^{-1}
    thrust::device_vector<double> d_A1((long long)m*n);

    // V^T from SVD
    thrust::device_vector<double> d_VT(n*n);

    // Singular values
    thrust::device_vector<double> d_sigma(n);

    thrust::device_vector<double> d_RsTR(n*n);

    double* pA = thrust::raw_pointer_cast(d_A.data());
    int8_t* pD = thrust::raw_pointer_cast(d_D.data());
    int* pS = thrust::raw_pointer_cast(d_S.data());
    double* pFA = thrust::raw_pointer_cast(d_FA.data());
    auto* pFft = thrust::raw_pointer_cast(d_fft.data());
    double* pSk = thrust::raw_pointer_cast(d_sk.data());
    double* pRs = thrust::raw_pointer_cast(d_RsTR.data());
    double* pG = thrust::raw_pointer_cast(d_G.data());
    double* pA1 = thrust::raw_pointer_cast(d_A1.data());
    double* pVT = thrust::raw_pointer_cast(d_VT.data());
    double* pSigma = thrust::raw_pointer_cast(d_sigma.data());

    // FFT plan
    cufftHandle fp;
    int fn[1] = {m};

    CUFFT_CHECK(cufftPlanMany(
        &fp, 1, fn,
        fn, 1, m,
        fn, 1, half_r,
        CUFFT_D2Z, n));

    // ------------------------------------------------------------
    // SVD workspace
    //
    // d_sk contains the sampled matrix A_s.
    // We only need V^T and the singular values.
    // ------------------------------------------------------------
    int lw_svd = 0;

    CUSOLVER_CHECK(
        cusolverDnDgesvd_bufferSize(
            solver,
            sketch_rows,
            n,
            &lw_svd));

    double* d_wsvd = nullptr;
    CUDA_CHECK(cudaMalloc(
        &d_wsvd, lw_svd * sizeof(double)));

    // ------------------------------------------------------------
    // Cholesky QR workspace
    // ------------------------------------------------------------
    int lw_ch = 0;
    double* d_wch = nullptr;
    int* d_ich = nullptr;

    CUSOLVER_CHECK(
        cusolverDnDpotrf_bufferSize(
            solver,
            CUBLAS_FILL_MODE_UPPER,
            n, pG, n,
            &lw_ch));

    CUDA_CHECK(cudaMalloc(
        &d_wch, lw_ch * sizeof(double)));

    CUDA_CHECK(cudaMalloc(
        &d_ich, sizeof(int)));

    // gesvd does not form U; gesvdj and gesvdp reuse d_A1 for U.

    double* d_U = nullptr;
    int svd_info = 0;

    auto one = [&]() {

        pA = thrust::raw_pointer_cast(d_A.data());

        thrust::transform(
            thrust::counting_iterator<unsigned>(0),
            thrust::counting_iterator<unsigned>(m),
            d_D.begin(),
            RandRadI8());

        thrust::transform(
            thrust::counting_iterator<unsigned>(0),
            thrust::counting_iterator<unsigned>(c),
            d_S.begin(),
            RandSamp{m});

        // Randomized transform
        diag_fp64<<<GRD, BLK>>>(
            pA, pD, pFA, m, n);

        CUDA_CHECK(cudaGetLastError());

        CUFFT_CHECK(
            cufftExecD2Z(fp, pFA, pFft));

        // --------------------------------------------------------
        // Form sampled matrix A_s
        // --------------------------------------------------------
        if (re_only)
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(
                    sketch_rows*n),
                d_sk.begin(),
                SelF64_Re{
                    pFft, pS, c, m, half_r});
        else
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(
                    sketch_rows*n),
                d_sk.begin(),
                SelF64{
                    pFft, pS, c, m, half_r});

        thrust::transform(
            d_sk.begin(), d_sk.end(),
            d_sk.begin(),
            [norm] __device__(double x) {
                return x * norm;
            });

        // --------------------------------------------------------
        // SVD of sampled matrix
        //
        // A_s = U Sigma V^T
        //
        // d_sk is overwritten by cusolver.
        // d_sigma contains Sigma.
        // d_VT contains V^T.
        // --------------------------------------------------------
        CUSOLVER_CHECK(
            cusolverDnDgesvd(
                solver,
                'N',       // do not compute U
                'A',       // compute full V^T
                sketch_rows,
                n,
                pSk,
                sketch_rows,
                pSigma,
                d_U,
                1,
                pVT,
                n,
                d_wsvd,
                lw_svd,
                nullptr,
                &svd_info));

        CUDA_CHECK(cudaDeviceSynchronize());

        // --------------------------------------------------------
        // Form
        //
        //     A1 = A V Sigma^{-1}
        //
        // gesvd stores V^T; gesvdj and gesvdp store V.
        //
        // First:
        //     A1 <- A V
        // --------------------------------------------------------
        const double one_d = 1.0;
        const double zero_d = 0.0;

        CUBLAS_CHECK(
            cublasDgemm(
                blas,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                m, n, n,
                &one_d,
                pA, m,
                pVT, n,
                &zero_d,
                pA1, m));

                   // print_matrix_small<<<1, 1>>>(pA1, 1, n, 1, n);
        // --------------------------------------------------------
        // Then scale columns by Sigma^{-1}.
        //
        // A1(:,j) <- A1(:,j) / sigma(j)
        //
        // This is V Sigma^{-1} = V L^{-1/2}.
        // --------------------------------------------------------
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(m*n),
            d_A1.begin(),
            [pA1,pSigma, m] __device__(int k) {

                const int j = k / m;

                return pA1[k] / pSigma[j];
            });

             

        // --------------------------------------------------------
        // Cholesky QR:
        //
        //     A1 = Q Rc
        //
        // pA1 is overwritten with Q.
        // pG contains Rc.
        // --------------------------------------------------------
        thrust::fill(
            d_G.begin(),
            d_G.end(),
            0.0);

        cholqr_phase_nosolve(
            blas,
            solver,
            pA1,
            pG,
            d_wch,
            lw_ch,
            d_ich,
            m,
            n);


               
CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Warmup (NOT timed)
    thrust::copy(
        d_Abak.begin(),
        d_Abak.end(),
        d_A.begin());

    one();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed trials (host↔device copies excluded)
    EV wall;
    wall.start();

    for (int t = 0; t < trials; ++t) {

        thrust::copy(
            d_Abak.begin(),
            d_Abak.end(),
            d_A.begin());

        one();
    }

    time_s = wall.stop() / trials;


        // Compute R = R_c · R_s (NOT timed)
    compute_B(blas, pG, pVT, pSigma,h_R, n);
   

// // Copy B into the output workspace.
// thrust::copy(
//     d_Btmp.begin(),
//     d_Btmp.end(),
//     d_RsTR.begin());

    // ------------------------------------------------------------
    // Download Q
    //
    // d_A1 now contains Q.
    // ------------------------------------------------------------
    thrust::copy(
        d_A1.begin(),
        d_A1.begin() + (long long)m*n,
        h_Q);

    // ------------------------------------------------------------
    // Download B
    //
    // h_R is now really B, not a triangular R.
    // A approximately/effectively satisfies
    //
    //     A = Q B
    //
    // up to the numerical effects of the randomized preconditioner.
    // ------------------------------------------------------------
    // thrust::copy(
    //     d_RsTR.begin(),
    //     d_RsTR.end(),
    //     h_R);

    cufftDestroy(fp);

    cudaFree(d_wsvd);
    cudaFree(d_wch);
    cudaFree(d_ich);

    cublasDestroy(blas);
    cusolverDnDestroy(solver);
}
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
static void svqb_pass(cublasHandle_t blas, cusolverDnHandle_t solver,
                       const double* pXin, double* pXout, int m, int n,
                       double* pG, double* peig, double* pDinv, double* prsqrt, double* pM,
                       double* d_wev, int lw_ev, int* d_iev)
{
    const double a = 1., b = 0.;

    // G = Xin^T Xin
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &a, pXin, m, pXin, m, &b, pG, n));

    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n), DiagInv{n, pG, pDinv});
    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_LEFT, n, n, pG, n, pDinv, 1, pG, n));
    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_RIGHT, n, n, pG, n, pDinv, 1, pG, n));

    // Eigendecompose normalized G in place: pG <- V, peig <- eigenvalues (ascending)
    CUSOLVER_CHECK(cusolverDnDsyevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                    n, pG, n, peig, d_wev, lw_ev, d_iev));

   // rsqrt[i] = 1/sqrt(clamp(eig[i]));  max eigenvalue = peig[n-1] (ascending order)
    double maxeig;
    CUDA_CHECK(cudaMemcpy(&maxeig, peig + (n - 1), sizeof(double), cudaMemcpyDeviceToHost));
    thrust::device_ptr<double> deig(peig), drsqrt(prsqrt);
    thrust::transform(deig, deig + n, drsqrt, RSqrtClamp{maxeig});

    // M = Dinv * V * diag(rsqrt)
    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_RIGHT, n, n, pG, n, prsqrt, 1, pM, n));
    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_LEFT, n, n, pM, n, pDinv, 1, pM, n));

    // Xout = Xin * M
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &a, pXin, m, pM, n, &b, pXout, m));
}

enum class MRSQR2Svd { Gesvd, Gesvdj, Gesvdp };

static void mrsqr2_fp64_impl(const double* h_A, int m, int n, int c,
                             double* h_Q, double* h_R, double& time_s,
                             int trials, bool re_only, MRSQR2Svd svd_kind)
{
    const double a = 1., b = 0.;
    const int half_r = m/2 + 1;
    const int sketch_rows = re_only ? c : 2*c;
    const double norm = std::sqrt((double)c / m);

    const int BLK = 256;
    const int GRD = (int)std::min(
        65535LL, ((long long)m*n + BLK - 1) / BLK);

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));

    cusolverDnHandle_t solver;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    // Upload A (NOT timed)
    thrust::device_vector<double> d_A(h_A, h_A + (long long)m*n);
    thrust::device_vector<double> d_Abak(d_A);

    // Working buffers
    thrust::device_vector<int8_t> d_D(m);
    thrust::device_vector<int> d_S(c);

    thrust::device_vector<double> d_FA((long long)m*n);
    thrust::device_vector<cufftDoubleComplex> d_fft(
        (long long)n * half_r);

    thrust::device_vector<double> d_sk(
        (long long)sketch_rows*n);

    // Final B = Rc * Sigma * V^T
    thrust::device_vector<double> d_RsTR(n*n, 0.0);

    // Rc from Cholesky QR
    thrust::device_vector<double> d_G(n*n);

    // A1 = A * V * Sigma^{-1}
    thrust::device_vector<double> d_A1((long long)m*n);

    // V^T from SVD
    thrust::device_vector<double> d_VT(n*n);

    // Singular values
    thrust::device_vector<double> d_sigma(n);

     thrust::device_vector<double> d_B((long long)m * n);

    thrust::device_vector<double>  d_M(n * n);
    thrust::device_vector<double> d_eig(n), d_Dinv(n), d_rsqrt(n);
    thrust::device_vector<double> d_R1(n * n), d_R2(n * n), d_R(n * n);

    double* pA = thrust::raw_pointer_cast(d_A.data());
    int8_t* pD = thrust::raw_pointer_cast(d_D.data());
    int* pS = thrust::raw_pointer_cast(d_S.data());
    double* pFA = thrust::raw_pointer_cast(d_FA.data());
    auto* pFft = thrust::raw_pointer_cast(d_fft.data());
    double* pSk = thrust::raw_pointer_cast(d_sk.data());
    double* pA1 = thrust::raw_pointer_cast(d_A1.data());
    double* pVT = thrust::raw_pointer_cast(d_VT.data());
    double* pSigma = thrust::raw_pointer_cast(d_sigma.data());

     double* pB    = thrust::raw_pointer_cast(d_B.data());
    double* pG    = thrust::raw_pointer_cast(d_G.data());
    double* pM    = thrust::raw_pointer_cast(d_M.data());
    double* peig  = thrust::raw_pointer_cast(d_eig.data());
    double* pDinv = thrust::raw_pointer_cast(d_Dinv.data());
    double* prsqrt= thrust::raw_pointer_cast(d_rsqrt.data());
        double* pR1   = thrust::raw_pointer_cast(d_R1.data());
        int lw_ev = 0; double* d_wev = nullptr; int* d_iev = nullptr;
         CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER, n, pG, n, peig, &lw_ev));
    CUDA_CHECK(cudaMalloc(&d_wev, lw_ev * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_iev, sizeof(int)));
        double* pAbak = thrust::raw_pointer_cast(d_Abak.data());
    // FFT plan
    cufftHandle fp;
    int fn[1] = {m};

    CUFFT_CHECK(cufftPlanMany(
        &fp, 1, fn,
        fn, 1, m,
        fn, 1, half_r,
        CUFFT_D2Z, n));

    // ------------------------------------------------------------
    // SVD workspace
    //
    // d_sk contains the sampled matrix A_s.
    // We only need V^T and the singular values.
    // ------------------------------------------------------------
    int lw_svd = 0;
    double* d_wsvd = nullptr;
    gesvdjInfo_t gesvdj_params = nullptr; double jacobi_tolerance=1e-15; int jacobi_max_sweeps=20;
    cusolverDnParams_t gesvdp_params = nullptr;
    size_t gesvdp_device_bytes = 0, gesvdp_host_bytes = 0;
    void* d_gesvdp_work = nullptr;
    std::vector<unsigned char> h_gesvdp_work;

    if (svd_kind == MRSQR2Svd::Gesvd) {
        CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(
            solver, sketch_rows, n, &lw_svd));
        CUDA_CHECK(cudaMalloc(&d_wsvd, lw_svd * sizeof(double)));
    } else if (svd_kind == MRSQR2Svd::Gesvdj) {
        CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
            CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params,jacobi_tolerance));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params,jacobi_max_sweeps));
        CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(
            solver, CUSOLVER_EIG_MODE_VECTOR, 1,
            sketch_rows, n, pSk, sketch_rows, pSigma,
            pA1, m, pVT, n, &lw_svd, gesvdj_params));
        CUDA_CHECK(cudaMalloc(&d_wsvd, lw_svd * sizeof(double)));
    } else {
        CUSOLVER_CHECK(cusolverDnCreateParams(&gesvdp_params));
        CUSOLVER_CHECK(cusolverDnXgesvdp_bufferSize(
            solver, gesvdp_params, CUSOLVER_EIG_MODE_VECTOR, 1,
            sketch_rows, n, CUDA_R_64F, pSk, sketch_rows,
            CUDA_R_64F, pSigma, CUDA_R_64F, pA1, m,
            CUDA_R_64F, pVT, n, CUDA_R_64F,
            &gesvdp_device_bytes, &gesvdp_host_bytes));
        CUDA_CHECK(cudaMalloc(&d_gesvdp_work, gesvdp_device_bytes));
        h_gesvdp_work.resize(gesvdp_host_bytes);
    }

    // SVD auxiliary outputs.
    // We do not need U, so d_U is not allocated.
    // jobu = 'N', jobvt = 'A'.

    double* d_U = (svd_kind == MRSQR2Svd::Gesvd) ? nullptr : pA1;
    int svd_info = 0;
    thrust::device_vector<int> d_svd_info(1);
    int* p_svd_info = thrust::raw_pointer_cast(d_svd_info.data());
    double gesvdp_err_sigma = 0.0;

    auto one = [&]() {

        pA = thrust::raw_pointer_cast(d_A.data());

        thrust::transform(
            thrust::counting_iterator<unsigned>(0),
            thrust::counting_iterator<unsigned>(m),
            d_D.begin(),
            RandRadI8());

        thrust::transform(
            thrust::counting_iterator<unsigned>(0),
            thrust::counting_iterator<unsigned>(c),
            d_S.begin(),
            RandSamp{m});

        // Randomized transform
        diag_fp64<<<GRD, BLK>>>(
            pA, pD, pFA, m, n);

        CUDA_CHECK(cudaGetLastError());

        CUFFT_CHECK(
            cufftExecD2Z(fp, pFA, pFft));

        // --------------------------------------------------------
        // Form sampled matrix A_s
        // --------------------------------------------------------
        if (re_only)
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(
                    sketch_rows*n),
                d_sk.begin(),
                SelF64_Re{
                    pFft, pS, c, m, half_r});
        else
            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(
                    sketch_rows*n),
                d_sk.begin(),
                SelF64{
                    pFft, pS, c, m, half_r});

        thrust::transform(
            d_sk.begin(), d_sk.end(),
            d_sk.begin(),
            [norm] __device__(double x) {
                return x * norm;
            });

        // --------------------------------------------------------
        // SVD of sampled matrix
        //
        // A_s = U Sigma V^T
        //
        // d_sk is overwritten by cusolver.
        // d_sigma contains Sigma.
        // d_VT contains V^T.
        // --------------------------------------------------------
        if (svd_kind == MRSQR2Svd::Gesvd) {
            CUSOLVER_CHECK(cusolverDnDgesvd(
                solver, 'N', 'A', sketch_rows, n, pSk, sketch_rows,
                pSigma, d_U, 1, pVT, n, d_wsvd, lw_svd,
                nullptr, &svd_info));
        } else if (svd_kind == MRSQR2Svd::Gesvdj) {
            CUSOLVER_CHECK(cusolverDnDgesvdj(
                solver, CUSOLVER_EIG_MODE_VECTOR, 1,
                sketch_rows, n, pSk, sketch_rows, pSigma,
                pA1, m, pVT, n, d_wsvd, lw_svd,
                p_svd_info, gesvdj_params));
        } else {
            CUSOLVER_CHECK(cusolverDnXgesvdp(
                solver, gesvdp_params, CUSOLVER_EIG_MODE_VECTOR, 1,
                sketch_rows, n, CUDA_R_64F, pSk, sketch_rows,
                CUDA_R_64F, pSigma, CUDA_R_64F, pA1, m,
                CUDA_R_64F, pVT, n, CUDA_R_64F,
                d_gesvdp_work, gesvdp_device_bytes,
                h_gesvdp_work.data(), gesvdp_host_bytes,
                p_svd_info, &gesvdp_err_sigma));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // --------------------------------------------------------
        // Form
        //
        //     A1 = A V Sigma^{-1}
        //
        // Since V^T is stored in pVT, V = (V^T)^T.
        //
        // First:
        //     A1 <- A V
        // --------------------------------------------------------
        const double one_d = 1.0;
        const double zero_d = 0.0;

        CUBLAS_CHECK(
            cublasDgemm(
                blas,
                CUBLAS_OP_N,
                svd_kind == MRSQR2Svd::Gesvd ? CUBLAS_OP_T : CUBLAS_OP_N,
                m, n, n,
                &one_d,
                pA, m,
                pVT, n,
                &zero_d,
                pA1, m));

                   // print_matrix_small<<<1, 1>>>(pA1, 1, n, 1, n);
        // --------------------------------------------------------
        // Then scale columns by Sigma^{-1}.
        //
        // A1(:,j) <- A1(:,j) / sigma(j)
        //
        // This is V Sigma^{-1} = V L^{-1/2}.
        // --------------------------------------------------------
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(m*n),
            d_A1.begin(),
            [pA1,pSigma, m] __device__(int k) {

                const int j = k / m;

                return pA1[k] / pSigma[j];
            });

             

        // Randomized preconditioning makes A1 well-conditioned. SVQB now
        // produces the final orthonormal factor without a Cholesky step.
        svqb_pass(blas, solver, pA1, pB, m, n, pG, peig, pDinv, prsqrt,
                  pM, d_wev, lw_ev, d_iev);

               
CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Warmup (NOT timed)
    thrust::copy(
        d_Abak.begin(),
        d_Abak.end(),
        d_A.begin());

    one();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed trials (host↔device copies excluded)
    EV wall;
    wall.start();

    for (int t = 0; t < trials; ++t) {

        thrust::copy(
            d_Abak.begin(),
            d_Abak.end(),
            d_A.begin());

        one();
    }

    time_s = wall.stop() / trials;

    // Recover B outside the timed region: B = Q^T A.
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &a, pB, m, pAbak, m, &b, pR1, n));
    thrust::copy(d_R1.begin(), d_R1.end(), h_R);
   

// // Copy B into the output workspace.
// thrust::copy(
//     d_Btmp.begin(),
//     d_Btmp.end(),
//     d_RsTR.begin());

    // ------------------------------------------------------------
    // Download Q
    //
    // d_B now contains Q.
    // ------------------------------------------------------------
    thrust::copy(
        d_B.begin(),
        d_B.begin() + (long long)m*n,
        h_Q);

    // ------------------------------------------------------------
    // Download B
    //
    // h_R is now really B, not a triangular R.
    // A approximately/effectively satisfies
    //
    //     A = Q B
    //
    // up to the numerical effects of the randomized preconditioner.
    // ------------------------------------------------------------
    // thrust::copy(
    //     d_RsTR.begin(),
    //     d_RsTR.end(),
    //     h_R);

    cufftDestroy(fp);

    cudaFree(d_wsvd);
    if (gesvdj_params) CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
    if (gesvdp_params) CUSOLVER_CHECK(cusolverDnDestroyParams(gesvdp_params));
    cudaFree(d_gesvdp_work);
    cudaFree(d_wev);
    cudaFree(d_iev);

    cublasDestroy(blas);
    cusolverDnDestroy(solver);
}

void mrsqr2_fp64(const double* h_A, int m, int n, int c,
                double* h_Q, double* h_R, double& time_s, int trials, bool re_only)
{
    mrsqr2_fp64_impl(h_A, m, n, c, h_Q, h_R, time_s, trials, re_only,
                     MRSQR2Svd::Gesvd);
}

void mrsqr2_gesvdj_fp64(const double* h_A, int m, int n, int c,
                        double* h_Q, double* h_R, double& time_s,
                        int trials, bool re_only)
{
    mrsqr2_fp64_impl(h_A, m, n, c, h_Q, h_R, time_s, trials, re_only,
                     MRSQR2Svd::Gesvdj);
}

void mrsqr2_gesvdp_fp64(const double* h_A, int m, int n, int c,
                        double* h_Q, double* h_R, double& time_s,
                        int trials, bool re_only)
{
    mrsqr2_fp64_impl(h_A, m, n, c, h_Q, h_R, time_s, trials, re_only,
                     MRSQR2Svd::Gesvdp);
}