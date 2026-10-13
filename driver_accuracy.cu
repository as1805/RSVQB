// driver_accuracy.cu
// Accuracy across singular-value distributions: MRCQR, MRSQR2, and references.
// No Eigen — all checks are GPU-side via cuBLAS/cuSOLVER.
//
// Fixed: m = 2^20, n = 50, sketch size c = 3n
// Sweep: kappa = 10^1, ..., 10^16 for each spectrum family.

#include "mrcqr.hpp"
#include "cholqr2.hpp"
#include "svqb2.hpp"
#include "common.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>
#include <algorithm>

// ---------------------------------------------------------------------------
// Random functor (uniform on [-1,1], deterministic per index)
// ---------------------------------------------------------------------------
struct RandUnif {
    unsigned offset;
    __host__ __device__ double operator()(unsigned i) const {
        thrust::default_random_engine r((i + offset) * 2654435761u);
        thrust::uniform_real_distribution<double> d(-1, 1);
        r.discard(i); return d(r);
    }
};

// ---------------------------------------------------------------------------
// GPU QR → explicit Q  (Dgeqrf + Dorgqr, square or tall)
// d_X is overwritten with the orthonormal Q factor.
// ---------------------------------------------------------------------------
static void gpu_ortho(cusolverDnHandle_t solver, double* d_X, int m, int n)
{
    int lw_qr = 0, lw_org = 0;
    double* d_tau  = nullptr;
    double* d_work = nullptr;
    int*    d_info = nullptr;

    CUDA_CHECK(cudaMalloc(&d_tau,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver, m, n, d_X, m, &lw_qr));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(solver, m, n, n, d_X, m, d_tau, &lw_org));
    int lw = std::max(lw_qr, lw_org);
    CUDA_CHECK(cudaMalloc(&d_work, lw * sizeof(double)));

    CUSOLVER_CHECK(cusolverDnDgeqrf(solver, m, n, d_X, m, d_tau, d_work, lw, d_info));
    CUSOLVER_CHECK(cusolverDnDorgqr(solver, m, n, n, d_X, m, d_tau, d_work, lw, d_info));

    cudaFree(d_tau); cudaFree(d_work); cudaFree(d_info);
}

// ---------------------------------------------------------------------------
// Generate A = U * diag(sigma) * V^T on GPU, then download to h_A.
// ---------------------------------------------------------------------------
static void makeSVDTestMatrix(cublasHandle_t blas, cusolverDnHandle_t solver,
                              int m, int n, const std::vector<double>& h_sigma,
                              std::vector<double>& h_A)
{
    assert((int)h_sigma.size() == n);
    assert(m >= n);
    thrust::device_vector<double> d_U(m * n), d_V(n * n);

    thrust::transform(thrust::counting_iterator<unsigned>(0),
                      thrust::counting_iterator<unsigned>(m * n),
                      d_U.begin(), RandUnif{0u});
    thrust::transform(thrust::counting_iterator<unsigned>(0),
                      thrust::counting_iterator<unsigned>(n * n),
                      d_V.begin(), RandUnif{1000000u});

    gpu_ortho(solver, thrust::raw_pointer_cast(d_U.data()), m, n);
    gpu_ortho(solver, thrust::raw_pointer_cast(d_V.data()), n, n);

    // Scale columns of U: U[:,i] *= sigma_i.
    thrust::device_vector<double> d_sigma(h_sigma);

    CUBLAS_CHECK(cublasDdgmm(blas, CUBLAS_SIDE_RIGHT, m, n,
                             thrust::raw_pointer_cast(d_U.data()), m,
                             thrust::raw_pointer_cast(d_sigma.data()), 1,
                             thrust::raw_pointer_cast(d_U.data()), m));

    // Form A = U * Sigma * V^T.
    thrust::device_vector<double> d_A(m * n);
    const double a = 1., b = 0.;
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n,
                             &a,
                             thrust::raw_pointer_cast(d_U.data()), m,
                             thrust::raw_pointer_cast(d_V.data()), n,
                             &b,
                             thrust::raw_pointer_cast(d_A.data()), m));

    h_A.resize(m * n);
    thrust::copy(d_A.begin(), d_A.end(), h_A.begin());
}

static std::vector<double> geometricSigma(int n, double kappa)
{
    std::vector<double> sigma(n);
    for (int i = 0; i < n; ++i)
        sigma[i] = std::pow(10.0, -std::log10(kappa) * i / (n - 1));
    return sigma;
}

static std::vector<double> linearSigma(int n, double kappa)
{
    std::vector<double> sigma(n);
    for (int i = 0; i < n; ++i)
        sigma[i] = 1.0 - (1.0 - 1.0 / kappa) * (double)i / (n - 1);
    return sigma;
}

static std::vector<double> clusteredSigma(int n, double kappa)
{
    std::vector<double> sigma(n);
    for (int i = 0; i < n; ++i)
        sigma[i] = (i < n / 2) ? 1.0 : 1.0 / kappa;
    return sigma;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ||Q^T Q - I||_2  (spectral norm = largest singular value of the error matrix)
// ---------------------------------------------------------------------------
static double ortho_check(cublasHandle_t blas, cusolverDnHandle_t solver,
                          const double* h_Q, int m, int n)
{
    const double a = 1., b = 0.;
    thrust::device_vector<double> d_Q(h_Q, h_Q + (long long)m * n);
    thrust::device_vector<double> d_G(n * n);

    // G = Q^T Q  (n×n)
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &a,
                             thrust::raw_pointer_cast(d_Q.data()), m,
                             thrust::raw_pointer_cast(d_Q.data()), m,
                             &b,
                             thrust::raw_pointer_cast(d_G.data()), n));

    // Subtract identity: G -= I
    thrust::host_vector<double> h_G(d_G);
    for (int i = 0; i < n; i++)
        h_G[i + n * i] -= 1.0;
    thrust::copy(h_G.begin(), h_G.end(), d_G.begin());

    // ||G - I||_2 = max |eigenvalue|  (G-I is symmetric)
    // Compute via Dsyevd and take max absolute eigenvalue.
    thrust::device_vector<double> d_eig(n);
    double* pG   = thrust::raw_pointer_cast(d_G.data());
    double* peig = thrust::raw_pointer_cast(d_eig.data());

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver,
        CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
        n, pG, n, peig, &lwork));
    thrust::device_vector<double> d_work(lwork);
    thrust::device_vector<int>    d_info(1);
    CUSOLVER_CHECK(cusolverDnDsyevd(solver,
        CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
        n, pG, n, peig,
        thrust::raw_pointer_cast(d_work.data()), lwork,
        thrust::raw_pointer_cast(d_info.data())));

    thrust::host_vector<double> h_eig(d_eig);
    double norm2 = 0.;
    for (int i = 0; i < n; i++)
        norm2 = std::max(norm2, std::abs(h_eig[i]));
    return norm2;
}

// ---------------------------------------------------------------------------
// ||A - QR||_F / ||A||_F  (Frobenius norm via cublasDnrm2 on flattened matrix)
// ---------------------------------------------------------------------------
static double residual_check(cublasHandle_t blas,
                             const double* h_A, const double* h_Q,
                             const double* h_R, int m, int n)
{
    const double a = 1., b = 0., neg1 = -1.;
    const long long mn = (long long)m * n;

    thrust::device_vector<double> d_A(h_A, h_A + mn);
    thrust::device_vector<double> d_Q(h_Q, h_Q + mn);
    thrust::device_vector<double> d_R(h_R, h_R + (long long)n * n);
    thrust::device_vector<double> d_QR(mn);

    // d_QR = Q * R
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &a,
                             thrust::raw_pointer_cast(d_Q.data()), m,
                             thrust::raw_pointer_cast(d_R.data()), n,
                             &b,
                             thrust::raw_pointer_cast(d_QR.data()), m));
    // d_QR -= d_A
    CUBLAS_CHECK(cublasDaxpy(blas, (int)mn, &neg1,
                             thrust::raw_pointer_cast(d_A.data()), 1,
                             thrust::raw_pointer_cast(d_QR.data()), 1));

    double norm_diff = 0., norm_A = 0.;
    CUBLAS_CHECK(cublasDnrm2(blas, (int)mn,
                             thrust::raw_pointer_cast(d_QR.data()), 1, &norm_diff));
    CUBLAS_CHECK(cublasDnrm2(blas, (int)mn,
                             thrust::raw_pointer_cast(d_A.data()), 1, &norm_A));
    return norm_diff / norm_A;
}

// ---------------------------------------------------------------------------
// Extract n×n upper triangle from m×n matrix (column-major, lda_src = m).
// Used to recover R from the in-place geqrf result before Dorgqr runs.
// ---------------------------------------------------------------------------
struct TriMNtoNN {
    int m_src, n; const double* src; double* dst;
    __host__ __device__ void operator()(int x) const {
        int row = x / n, col = x % n;
        if (row <= col) dst[row + n * col] = src[row + m_src * col];
    }
};

// ---------------------------------------------------------------------------
// geqrf reference: cuSOLVER Dgeqrf (Q explicit via Dorgqr, R from upper tri).
// h_Q and h_R are host output buffers of size m*n and n*n respectively.
// ---------------------------------------------------------------------------
static void run_geqrf(cusolverDnHandle_t solver, cublasHandle_t blas,
                      const double* h_A, int m, int n,
                      double* h_Q, double* h_R)
{
    thrust::device_vector<double> d_A(h_A, h_A + (long long)m * n);
    thrust::device_vector<double> d_R(n * n, 0.);

    int lw_qr = 0, lw_org = 0;
    double* d_tau  = nullptr;
    double* d_work = nullptr;
    int*    d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    double* pA = thrust::raw_pointer_cast(d_A.data());
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver, m, n, pA, m, &lw_qr));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(solver, m, n, n, pA, m, d_tau, &lw_org));
    int lw = std::max(lw_qr, lw_org);
    CUDA_CHECK(cudaMalloc(&d_work, lw * sizeof(double)));

    // Dgeqrf: upper triangle of d_A becomes R, lower = Householder vectors
    CUSOLVER_CHECK(cusolverDnDgeqrf(solver, m, n, pA, m, d_tau, d_work, lw, d_info));

    // Extract R (n×n upper triangle) before Dorgqr overwrites d_A
    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n * n),
                     TriMNtoNN{m, n, pA, thrust::raw_pointer_cast(d_R.data())});

    // Dorgqr: form explicit Q in d_A
    CUSOLVER_CHECK(cusolverDnDorgqr(solver, m, n, n, pA, m, d_tau, d_work, lw, d_info));

    thrust::copy(d_A.begin(), d_A.begin() + (long long)m * n, h_Q);
    thrust::copy(d_R.begin(), d_R.end(), h_R);

    cudaFree(d_tau); cudaFree(d_work); cudaFree(d_info);
}

// ---------------------------------------------------------------------------
int main()
{
    const int m = 1 << 20;   // 2^17
    const int n = 50;
    const int c = 3 * n;

    struct SpectrumCase {
        const char* name;
    };
    const std::vector<SpectrumCase> spectra = {
        {"geometric"},
        {"linear"},
        {"clustered"}
    };
    const int KAPPA_COUNT = 16;
    const int NK = (int)spectra.size() * KAPPA_COUNT;

    const double NV = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> dev_f64(NK,NV), dev_f32(NK,NV), dev_f16(NK,NV),
                        dev_cholqr2(NK,NV), dev_geqrf(NK,NV), dev_svqb2(NK,NV),
                        dev_mrsqr(NK,NV), dev_mrsqr2(NK,NV),
                        dev_mreig_svqb(NK,NV), dev_mrsqr2_gesvdj(NK,NV),
                        dev_mrsqr2_gesvdp(NK,NV),
                        dev_mrsqr2_fp32(NK,NV), dev_mrsqr2_gesvdj_fp32(NK,NV),
                        dev_mrsqr2_gesvdp_fp32(NK,NV);
    std::vector<double> res_f64(NK,NV), res_f32(NK,NV), res_f16(NK,NV),
                        res_cholqr2(NK,NV), res_geqrf(NK,NV), res_svqb2(NK,NV),
                        res_mrsqr(NK,NV), res_mrsqr2(NK,NV),
                        res_mreig_svqb(NK,NV), res_mrsqr2_gesvdj(NK,NV),
                        res_mrsqr2_gesvdp(NK,NV),
                        res_mrsqr2_fp32(NK,NV), res_mrsqr2_gesvdj_fp32(NK,NV),
                        res_mrsqr2_gesvdp_fp32(NK,NV);
    std::vector<double> h_err_mrsqr2_gesvdp(NK,NV);
    std::vector<double> h_err_mrsqr2_gesvdp_fp32(NK,NV);

    std::vector<double> h_Q((long long)m * n), h_R((long long)n * n);
    double time_s;

    cublasHandle_t    blas;   CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    printf("=== Accuracy across singular-value distributions: m=%d, n=%d, c=%d ===\n",
           m, n, c);
    printf("Sweeping kappa = 10^1, ..., 10^16 for geometric, linear, and clustered spectra.\n\n");

    for (int spectrum_index = 0; spectrum_index < (int)spectra.size(); ++spectrum_index) {
      for (int kappa_index = 0; kappa_index < KAPPA_COUNT; ++kappa_index) {
        const int i = spectrum_index * KAPPA_COUNT + kappa_index;
        const double kappa = std::pow(10.0, kappa_index + 1);
        std::vector<double> sigma;
        if (spectrum_index == 0)
            sigma = geometricSigma(n, kappa);
        else if (spectrum_index == 1)
            sigma = linearSigma(n, kappa);
        else
            sigma = clusteredSigma(n, kappa);

        printf("spectrum = %s, kappa = 1e%d ... ",
               spectra[spectrum_index].name, kappa_index + 1);
        fflush(stdout);

        std::vector<double> h_A;
        makeSVDTestMatrix(blas, solver, m, n, sigma, h_A);

        auto measure = [&](double& dev, double& res) {
            dev = ortho_check(blas, solver, h_Q.data(), m, n);
            res = residual_check(blas, h_A.data(), h_Q.data(), h_R.data(), m, n);
            if (!std::isfinite(dev)) dev = NV;
            if (!std::isfinite(res)) res = NV;
        };

        mrcqr_fp64(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s);
        measure(dev_f64[i], res_f64[i]);

        mrcqr_fp32(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s);
        measure(dev_f32[i], res_f32[i]);

        mrcqr_fp16(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s);
        measure(dev_f16[i], res_f16[i]);

        cholqr2(h_A.data(), m, n, h_Q.data(), h_R.data(), time_s);
        measure(dev_cholqr2[i], res_cholqr2[i]);

        svqb2(h_A.data(), m, n, h_Q.data(), h_R.data(), time_s);
        measure(dev_svqb2[i], res_svqb2[i]);

        mrsqr_fp64(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s);
        measure(dev_mrsqr[i], res_mrsqr[i]);

        mrsqr2_fp64(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s);
        measure(dev_mrsqr2[i], res_mrsqr2[i]);

        mrsqr2_gesvdj_fp64(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s);
        measure(dev_mrsqr2_gesvdj[i], res_mrsqr2_gesvdj[i]);

        mrsqr2_gesvdp_fp64(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s,
                   1, true, &h_err_mrsqr2_gesvdp[i]);
        measure(dev_mrsqr2_gesvdp[i], res_mrsqr2_gesvdp[i]);

        mrsqr2_fp32(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s);
        measure(dev_mrsqr2_fp32[i], res_mrsqr2_fp32[i]);

        mrsqr2_gesvdj_fp32(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s);
        measure(dev_mrsqr2_gesvdj_fp32[i], res_mrsqr2_gesvdj_fp32[i]);

        mrsqr2_gesvdp_fp32(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s,
                           1, true, &h_err_mrsqr2_gesvdp_fp32[i]);
        measure(dev_mrsqr2_gesvdp_fp32[i], res_mrsqr2_gesvdp_fp32[i]);

        mreig2_svqb(h_A.data(), m, n, c, h_Q.data(), h_R.data(), time_s);
        measure(dev_mreig_svqb[i], res_mreig_svqb[i]);

        run_geqrf(solver, blas, h_A.data(), m, n, h_Q.data(), h_R.data());
        measure(dev_geqrf[i], res_geqrf[i]);

      fflush(stdout);
      }
    }

    cublasDestroy(blas); cusolverDnDestroy(solver);

    // MATLAB output
    auto print_matrix = [&](const char* name, const std::vector<double>& v) {
        printf("%s = [\n", name);
        for (int row = 0; row < (int)spectra.size(); ++row) {
            printf("  ");
            for (int col = 0; col < KAPPA_COUNT; ++col) {
                if (col) printf(", ");
                const double value = v[row * KAPPA_COUNT + col];
                std::isnan(value) ? printf("NaN") : printf("%.6e", value);
            }
            printf(";\n");
        }
        printf("];\n");
    };

    printf("\n%% === MATLAB DATA (m=%d, n=%d) ===\n", m, n);
    printf("spectrum_types = {'geometric', 'linear', 'clustered'};\n");
    printf("cond_A = 10.^(1:16);\n");
    printf("%% Result matrices have one row per spectrum_types entry and one column per cond_A.\n");

    printf("%% Deviation from orthonormality ||Q^T Q - I||_2\n");
    print_matrix("dev_fp64",    dev_f64);
    print_matrix("dev_fp32",    dev_f32);
    print_matrix("dev_fp16",    dev_f16);
    print_matrix("dev_cholqr2", dev_cholqr2);
    print_matrix("dev_svqb",    dev_svqb2);
    print_matrix("dev_mrsqr",   dev_mrsqr);
    print_matrix("dev_mrsqr2",  dev_mrsqr2);
    print_matrix("dev_mrsqr2_gesvdj", dev_mrsqr2_gesvdj);
    print_matrix("dev_mrsqr2_gesvdp", dev_mrsqr2_gesvdp);
    print_matrix("dev_mrsqr2_fp32", dev_mrsqr2_fp32);
    print_matrix("dev_mrsqr2_gesvdj_fp32", dev_mrsqr2_gesvdj_fp32);
    print_matrix("dev_mrsqr2_gesvdp_fp32", dev_mrsqr2_gesvdp_fp32);
    print_matrix("h_err_mrsqr2_gesvdp", h_err_mrsqr2_gesvdp);
    print_matrix("h_err_mrsqr2_gesvdp_fp32", h_err_mrsqr2_gesvdp_fp32);
    print_matrix("dev_mreig2_svqb", dev_mreig_svqb);
    print_matrix("dev_geqrf",     dev_geqrf);

    printf("\n%% Relative residual ||A - QR||_F / ||A||_F\n");
    print_matrix("res_fp64",    res_f64);
    print_matrix("res_fp32",    res_f32);
    print_matrix("res_fp16",    res_f16);
    print_matrix("res_cholqr2", res_cholqr2);
    print_matrix("res_svqb2",   res_svqb2);
    print_matrix("res_mrsqr",   res_mrsqr);
    print_matrix("res_mrsqr2",  res_mrsqr2);
    print_matrix("res_mrsqr2_gesvdj", res_mrsqr2_gesvdj);
    print_matrix("res_mrsqr2_gesvdp", res_mrsqr2_gesvdp);
    print_matrix("res_mrsqr2_fp32", res_mrsqr2_fp32);
    print_matrix("res_mrsqr2_gesvdj_fp32", res_mrsqr2_gesvdj_fp32);
    print_matrix("res_mrsqr2_gesvdp_fp32", res_mrsqr2_gesvdp_fp32);
    print_matrix("res_mreig_svqb", res_mreig_svqb);
    print_matrix("res_geqrf",     res_geqrf);

    return 0;
}
