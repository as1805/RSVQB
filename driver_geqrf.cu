// driver_geqrf.cu
// Timing benchmark for cuSOLVER Dgeqrf + Dorgqr (Householder QR).
// Same sweep as driver_perf: m = 2^20, n = 10:10:500, 5 trials.
// Host<->device copy time is excluded from reported times.

#include "common.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Time Dgeqrf + Dorgqr on an m×n matrix.
// h_A is the host input; timing excludes upload and download.
// Returns average time in seconds over `trials` runs.
// ---------------------------------------------------------------------------
static double time_geqrf(cusolverDnHandle_t solver,
                         const double* h_A, int m, int n, int trials)
{
    // Upload (NOT timed)
    thrust::device_vector<double> d_A(h_A, h_A + (long long)m * n);
    thrust::device_vector<double> d_Abak(d_A);

    // Query workspace sizes
    double* d_tau  = nullptr;
    double* d_work = nullptr;
    int*    d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    double* pA = thrust::raw_pointer_cast(d_A.data());
    int lw_qr = 0, lw_org = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver, m, n, pA, m, &lw_qr));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(solver, m, n, n, pA, m,
                                               d_tau, &lw_org));
    int lw = std::max(lw_qr, lw_org);
    CUDA_CHECK(cudaMalloc(&d_work, lw * sizeof(double)));

    auto one = [&]() {
        pA = thrust::raw_pointer_cast(d_A.data());
        // Restore A from backup (device-to-device, inside timing window)
        thrust::copy(d_Abak.begin(), d_Abak.end(), d_A.begin());
        CUSOLVER_CHECK(cusolverDnDgeqrf(solver, m, n, pA, m,
                                        d_tau, d_work, lw, d_info));
        CUSOLVER_CHECK(cusolverDnDorgqr(solver, m, n, n, pA, m,
                                        d_tau, d_work, lw, d_info));
    };

    // Warmup (NOT timed)
    one(); CUDA_CHECK(cudaDeviceSynchronize());

    // Timed trials
    EV wall; wall.start();
    for (int t = 0; t < trials; ++t) one();
    double time_s = wall.stop() / trials;

    cudaFree(d_tau); cudaFree(d_work); cudaFree(d_info);
    return time_s;
}

// ---------------------------------------------------------------------------
int main()
{
    const int m      = 1 << 20;
    const int n_min  = 20, n_max = 200, n_step = 20;
    const int trials = 5;
    const int NN     = (n_max - n_min) / n_step + 1;

    cusolverDnHandle_t solver;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    printf("=== geqrf Timing Benchmark: m=%d, n=%d:%d:%d, trials=%d ===\n\n",
           m, n_min, n_step, n_max, trials);
    printf("Allocating host buffer ... "); fflush(stdout);

    std::vector<double> h_A((size_t)m * n_max);
    std::mt19937_64 gen(42ULL);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (auto& x : h_A) x = dis(gen);
    printf("done\n\n");

    std::vector<double> t_geqrf(NN);

    int idx = 0;
    for (int n = n_min; n <= n_max; n += n_step, ++idx) {
        t_geqrf[idx] = time_geqrf(solver, h_A.data(), m, n, trials);
        printf("n=%3d  geqrf=%.4f (s)\n", n, t_geqrf[idx]);
        fflush(stdout);
    }

    cusolverDnDestroy(solver);

    // MATLAB output
    printf("\n%% === MATLAB DATA (m=2^20, n=%d:%d:%d, trials=%d) ===\n\n",
           n_min, n_step, n_max, trials);
    printf("cols = %d:%d:%d;\n\n", n_min, n_step, n_max);
    printf("geqrf_times = [\n");
    for (int i = 0; i < NN; i++) {
        printf("  %.8f", t_geqrf[i]);
        if ((i + 1) % 10 == 0 || i == NN - 1) printf("\n");
    }
    printf("];\n");

    return 0;
}
