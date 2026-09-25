// driver_perf.cu
// Timing driver for the three SVQB variants.
// Set ENABLE_SVQB_MIXED or ENABLE_SVQB_CHOLQR to 0 to remove an add-on
// from the experiment without changing the rest of the driver.

#include "svqb2.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifndef ENABLE_SVQB_MIXED
#define ENABLE_SVQB_MIXED 1
#endif

#ifndef ENABLE_SVQB_CHOLQR
#define ENABLE_SVQB_CHOLQR 1
#endif

static void make_random_matrix(int m, int n, std::vector<double>& A)
{
    std::mt19937_64 gen(12345);
    std::normal_distribution<double> normal(0.0, 1.0);
    A.resize((long long)m * n);
    for (double& x : A) x = normal(gen);
}

int main(int argc, char** argv)
{
    // Usage: ./driver_perf [m] [n] [trials]
    const int m = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    const int n = (argc > 2) ? std::atoi(argv[2]) : 50;
    const int trials = (argc > 3) ? std::atoi(argv[3]) : 20;

    std::vector<double> h_A;
    make_random_matrix(m, n, h_A);

    std::vector<double> h_Q((long long)m * n);
    std::vector<double> h_R((long long)n * n);

    double time_svqb = 0.0;
    double time_mixed = 0.0;
    double time_cholqr = 0.0;

    printf("=== SVQB performance ===\n");
    printf("m = %d, n = %d, trials = %d\n\n", m, n, trials);

    // Original two-pass FP64 SVQB.
    svqb2(h_A.data(), m, n, h_Q.data(), h_R.data(), time_svqb, trials);
    printf("svqb2          : %.6e s\n", time_svqb);

#if ENABLE_SVQB_MIXED
    // FP32 first pass + FP64 second pass.
    svqb2_mixed(h_A.data(), m, n, h_Q.data(), h_R.data(), time_mixed, trials);
    printf("svqb2_mixed    : %.6e s\n", time_mixed);
#else
    printf("svqb2_mixed    : disabled\n");
#endif

#if ENABLE_SVQB_CHOLQR
    // FP64 SVQB pass + Cholesky QR pass.
    svqb2_cholqr(h_A.data(), m, n, h_Q.data(), h_R.data(), time_cholqr, trials);
    printf("svqb2_cholqr   : %.6e s\n", time_cholqr);
#else
    printf("svqb2_cholqr   : disabled\n");
#endif

    printf("\n%% MATLAB data\n");
    printf("time_svqb2 = %.16e;\n", time_svqb);

#if ENABLE_SVQB_MIXED
    printf("time_svqb2_mixed = %.16e;\n", time_mixed);
#endif

#if ENABLE_SVQB_CHOLQR
    printf("time_svqb2_cholqr = %.16e;\n", time_cholqr);
#endif

    return 0;
}
