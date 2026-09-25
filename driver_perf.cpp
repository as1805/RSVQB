// driver_perf.cpp
// Performance benchmark: m = 2^20, n = 10:10:500 (50 values).
// Methods: MRCQR (FP64 / FP32 / FP16) and CholQR2.
// Host<->device copy time excluded from reported times (see mrcqr.hpp).
//
// Host memory: ~8.4 GB (h_A + h_Q at n_max=500).

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <string>

#include "mrcqr.hpp"
#include "cholqr2.hpp"
#include "svqb2.hpp"

int main()
{
    const int m      = 1 << 20;   // 2^20 = 1048576
    const int n_min  = 10, n_max = 500, n_step = 10; //was 10
    const int trials = 5;
    const int NN     = (n_max - n_min) / n_step + 1;   // 50 values

    std::vector<double> time_mrcqr_fp64(NN), time_mrcqr_fp32(NN), time_mrcqr_fp16(NN),
                        time_cholqr2(NN), time_svqb2(NN), time_mrsqr(NN),
                        time_mrsqr2(NN), time_mreig_svqb(NN),
                        time_mrsqr2_gesvdj(NN), time_mrsqr2_gesvdp(NN);

    printf("=== Performance Benchmark: m=2^20, n=%d:%d:%d, trials=%d ===\n\n",
           n_min, n_step, n_max, trials);
    printf("Allocating host buffers ... "); fflush(stdout);

    std::vector<double> h_A((size_t)m * n_max);
    std::vector<double> h_Q((size_t)m * n_max);
    std::vector<double> h_R((size_t)n_max * n_max);

    std::mt19937_64 gen(42ULL);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (auto& x : h_A) x = dis(gen);
    printf("done\n\n");

        printf("%-5s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
            "n", "FP64(s)", "FP32(s)", "FP16(s)", "CQR2(s)", "SVQB2(s)",
            "SVQBCR(s)", "SVQB2M(s)", "MREIG(s)", "GESVDJ(s)", "GESVDP(s)");
    printf("%s\n", std::string(106, '-').c_str());

    int idx = 0;
    for (int n = n_min; n <= n_max; n += n_step, ++idx) {
        const int c = 3 * n;
        double ts;

        mrcqr_fp64(h_A.data(), m, n, c, h_Q.data(), h_R.data(), ts, trials);
        time_mrcqr_fp64[idx] = ts;

        mrcqr_fp32(h_A.data(), m, n, c, h_Q.data(), h_R.data(), ts, trials);
        time_mrcqr_fp32[idx] = ts;

        mrcqr_fp16(h_A.data(), m, n, c, h_Q.data(), h_R.data(), ts, trials);
        time_mrcqr_fp16[idx] = ts;

        cholqr2(h_A.data(), m, n, h_Q.data(), h_R.data(), ts, trials);
        time_cholqr2[idx] = ts;

        svqb2(h_A.data(), m, n, h_Q.data(), h_R.data(), ts, trials);
        time_svqb2[idx] = ts;

        mrsqr_fp64(h_A.data(), m, n,c, h_Q.data(), h_R.data(), ts, trials);
        time_mrsqr[idx] = ts;

       mrsqr2_fp64(h_A.data(), m, n,c, h_Q.data(), h_R.data(), ts, trials);
        time_mrsqr2[idx] = ts;

        mreig_svqb(h_A.data(), m, n,c, h_Q.data(), h_R.data(), ts, trials);
        time_mreig_svqb[idx] = ts;

         mrsqr2_gesvdj_fp64(h_A.data(), m, n, c, h_Q.data(), h_R.data(), ts, trials);
         time_mrsqr2_gesvdj[idx] = ts;

         mrsqr2_gesvdp_fp64(h_A.data(), m, n, c, h_Q.data(), h_R.data(), ts, trials);
         time_mrsqr2_gesvdp[idx] = ts;

         printf("n=%-3d  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
             n, time_mrcqr_fp64[idx], time_mrcqr_fp32[idx], time_mrcqr_fp16[idx],
             time_cholqr2[idx], time_svqb2[idx], time_mrsqr[idx], time_mrsqr2[idx],
             time_mreig_svqb[idx], time_mrsqr2_gesvdj[idx], time_mrsqr2_gesvdp[idx]);
        fflush(stdout);
    }

    // MATLAB output
    auto print_arr = [&](const char* name, const std::vector<double>& v) {
        printf("%s = [\n", name);
        for (int i = 0; i < (int)v.size(); i++) {
            printf("  %.8f", v[i]);
            if ((i + 1) % 10 == 0 || i == (int)v.size() - 1) printf("\n");
        }
        printf("];\n\n");
    };

    printf("\n%% === MATLAB DATA (m=2^20, n=%d:%d:%d, trials=%d) ===\n\n",
           n_min, n_step, n_max, trials);
    printf("cols = %d:%d:%d;\n\n", n_min, n_step, n_max);
    print_arr("mrcqr_fp64_times", time_mrcqr_fp64);
    print_arr("mrcqr_fp32_times", time_mrcqr_fp32);
    print_arr("mrcqr_fp16_times", time_mrcqr_fp16);
    print_arr("cholqr2_times", time_cholqr2);
    print_arr("svqb2_times", time_svqb2);
    print_arr("mrsqr_times", time_mrsqr);
    print_arr("mrsqr2_times", time_mrsqr2);
    print_arr("mreig_svqb_times", time_mreig_svqb);
    print_arr("mrsqr2_gesvdj_times", time_mrsqr2_gesvdj);
    print_arr("mrsqr2_gesvdp_times", time_mrsqr2_gesvdp);

    return 0;
}
