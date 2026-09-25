// driver_profile.cu
// Sub-operation profiling for MRCQR, randomized SVD QR, and CholQR2.
//
// MRCQR phases:   DiagMult | FFT | Select+Norm | SketchQR || TrsmPre | GEMM | Chol | TrsmFin
// CholQR2 phases: GEMM1 | Chol1 | TRSM1 | GEMM2 | Chol2 | TRSM2
//
// Fixed: m = 2^20.  n ∈ {50, 100, 200, 500}.  10 trials each.

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

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Random functor
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
struct RandSamp { int N;
    __host__ __device__ int operator()(unsigned i) const {
        thrust::default_random_engine r(i * 2246822519u);
        thrust::uniform_int_distribution<int> d(0, N-1);
        r.discard(i); return d(r);
    }
};

// ---------------------------------------------------------------------------
// Diagonal multiply + convert kernels
// ---------------------------------------------------------------------------
__global__ void diag_fp64(const double* A, const int8_t* D, double* FA, int m, int n) {
    for (long long i = blockIdx.x*blockDim.x+threadIdx.x; i < (long long)m*n; i += blockDim.x*gridDim.x) {
        unsigned long long b = __double_as_longlong(A[i]);
        b ^= (unsigned long long)(D[i%m]<0) << 63; FA[i] = __longlong_as_double(b); } }

__global__ void diag_fp32(const double* A, const int8_t* D, float* FA, int m, int n) {
    for (long long i = blockIdx.x*blockDim.x+threadIdx.x; i < (long long)m*n; i += blockDim.x*gridDim.x) {
        unsigned b = __float_as_uint((float)A[i]);
        b ^= (unsigned)(D[i%m]<0) << 31; FA[i] = __uint_as_float(b); } }

__global__ void diag_fp16(const double* A, const int8_t* D, __half* FA, int m, int n) {
    for (long long i = blockIdx.x*blockDim.x+threadIdx.x; i < (long long)m*n; i += blockDim.x*gridDim.x) {
        unsigned short b = __half_as_ushort(__float2half((float)A[i]));
        b ^= (unsigned short)(D[i%m]<0) << 15; FA[i] = __ushort_as_half(b); } }

// ---------------------------------------------------------------------------
// Row-selection functors — Re-only (matches re_only=true default in mrcqr.cu).
// c sampled frequencies → c rows (real parts only), SketchQR on c×n.
// ---------------------------------------------------------------------------
struct SelF64 { const cufftDoubleComplex* fft; const int* idx; int c,m,rpc;
    __host__ __device__ double operator()(int i) const {
        int col=i/c,r=idx[i%c],ar=(r<=m/2)?r:(m-r);
        return fft[ar+col*rpc].x; } };
struct SelF32 { const cufftComplex* fft; const int* idx; int c,m,rpc;
    __host__ __device__ float operator()(int i) const {
        int col=i/c,r=idx[i%c],ar=(r<=m/2)?r:(m-r);
        return fft[ar+col*rpc].x; } };
struct SelF16 { const __half2* fft; const int* idx; int c,m,rpc;
    __host__ __device__ float operator()(int i) const {
        int col=i/c,r=idx[i%c],ar=(r<=m/2)?r:(m-r);
        return __half22float2(fft[ar+col*rpc]).x; } };

// ---------------------------------------------------------------------------
// Upper-tri copy functors
// ---------------------------------------------------------------------------
struct TriD64 { int c,n; const double* src; double* dst;
    __host__ __device__ void operator()(int x) const {
        int row=x/n,col=x%n; if(row<=col&&row<n) dst[row+n*col]=src[row+c*col]; } };
struct TriF32toF64 { int c,n; const float* src; double* dst;
    __host__ __device__ void operator()(int x) const {
        int row=x/n,col=x%n; if(row<=col&&row<n) dst[row+n*col]=(double)src[row+c*col]; } };
struct TriNN { int n; const double* src; double* dst;
    __host__ __device__ void operator()(int x) const {
        int row=x/n,col=x%n; if(row<=col) dst[row+n*col]=src[row+n*col]; } };

// ---------------------------------------------------------------------------
// Timing structs (all times in ms, averaged over trials)
// ---------------------------------------------------------------------------
struct MRCQRTimes {
    double diag, fft, select, skqr;          // sketch phase
    double trsm_pre, gemm, chol, trsm_fin;   // CholQR phase
    double total;
    double sketch() const { return diag+fft+select+skqr; }
    double cholqr() const { return trsm_pre+gemm+chol+trsm_fin; }
};
struct CQR2Times {
    double gemm1, chol1, trsm1;  // pass 1
    double gemm2, chol2, trsm2;  // pass 2
    double total;
};

struct MRSQRTimes {
    double diag, fft, select, svd;
    double pre_gemm, pre_scale;
    double chol_gemm, chol, chol_trsm;
    double svqb_gram, svqb_norm, svqb_eig, svqb_apply;
    double total;
};
struct MREigTimes {
    double diag, fft, select, gram, eig, scale, pre_gemm;
    double svqb_gram, svqb_norm, svqb_eig, svqb_apply;
    double total;
};

enum class ProfileSvd { Gesvd, Gesvdj, Gesvdp };

// ---------------------------------------------------------------------------
// Shared CholQR phase profiler.
// On entry: pA = Q₀ (m×n), pG = R_s (n×n, upper tri).
// On exit:  pA = Q,  pG = R_c (Cholesky factor).
// Fills trsm_pre, gemm, chol, trsm_fin in `t` (accumulated, divide by trials after).
// ---------------------------------------------------------------------------
static void cholqr_timed(cublasHandle_t blas, cusolverDnHandle_t solver,
                         double* pA, double* pG,
                         double* d_wch, int lw_ch, int* d_ich,
                         int m, int n, MRCQRTimes& t, bool rec, EV& ev)
{
    const double a=1., b=0.;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    double dt;

    ev.start();
    CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, m, n, &a, pG, n, pA, m));
    dt=ev.stop(); if(rec) t.trsm_pre += dt*1e3;

    ev.start();
    CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &a, pA, m, pA, m, &b, pG, n));
    dt=ev.stop(); if(rec) t.gemm += dt*1e3;

    ev.start();
    CUSOLVER_CHECK(cusolverDnDpotrf(solver, uplo, n, pG, n, d_wch, lw_ch, d_ich));
    dt=ev.stop(); if(rec) t.chol += dt*1e3;

    ev.start();
    CUBLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, m, n, &a, pG, n, pA, m));
    dt=ev.stop(); if(rec) t.trsm_fin += dt*1e3;
}

// ===========================================================================
// MRCQR FP64 profiler
// ===========================================================================
static MRCQRTimes profile_fp64(cublasHandle_t blas, cusolverDnHandle_t solver,
                                thrust::device_vector<double>& d_A,
                                const thrust::device_vector<double>& d_Abak,
                                int m, int n, int trials)
{
    const int c=3*n, half_r=m/2+1;
    const double norm=std::sqrt((double)c/m);
    const int BLK=256, GRD=(int)std::min(65535LL,((long long)m*n+BLK-1)/BLK);

    thrust::device_vector<int8_t>             d_D(m);
    thrust::device_vector<int>                d_S(c);
    thrust::device_vector<double>             d_FA((long long)m*n);
    thrust::device_vector<cufftDoubleComplex> d_fft((long long)n*half_r);
    thrust::device_vector<double>             d_sk(c*n);
    thrust::device_vector<double>             d_RsTR(n*n, 0.);
    thrust::device_vector<double>             d_G(n*n);

    double *pA, *pFA=thrust::raw_pointer_cast(d_FA.data());
    int8_t *pD=thrust::raw_pointer_cast(d_D.data());
    int    *pS=thrust::raw_pointer_cast(d_S.data());
    auto   *pFft=thrust::raw_pointer_cast(d_fft.data());
    double *pSk=thrust::raw_pointer_cast(d_sk.data());
    double *pRs=thrust::raw_pointer_cast(d_RsTR.data());
    double *pG=thrust::raw_pointer_cast(d_G.data());

    cufftHandle fp; int fn[1]={m};
    CUFFT_CHECK(cufftPlanMany(&fp,1,fn,fn,1,m,fn,1,half_r,CUFFT_D2Z,n));

    int lw_sk=0; double *d_tau=nullptr,*d_wsk=nullptr; int *d_isk=nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau,n*sizeof(double)));
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver,c,n,pSk,c,&lw_sk));
    CUDA_CHECK(cudaMalloc(&d_wsk,lw_sk*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isk,sizeof(int)));

    int lw_ch=0; double *d_wch=nullptr; int *d_ich=nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_UPPER,n,pG,n,&lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch,lw_ch*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich,sizeof(int)));

    MRCQRTimes t={}; EV ev, wall;

    auto one=[&](bool rec){
        thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
        pA=thrust::raw_pointer_cast(d_A.data()); double dt;
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m),d_D.begin(),RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c),d_S.begin(),RandSamp{m});

        ev.start(); diag_fp64<<<GRD,BLK>>>(pA,pD,pFA,m,n);
        CUDA_CHECK(cudaGetLastError()); dt=ev.stop(); if(rec) t.diag+=dt*1e3;

        ev.start(); CUFFT_CHECK(cufftExecD2Z(fp,pFA,pFft));
        dt=ev.stop(); if(rec) t.fft+=dt*1e3;

        ev.start();
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(c*n),
                          d_sk.begin(),SelF64{pFft,pS,c,m,half_r});
        thrust::transform(d_sk.begin(),d_sk.end(),d_sk.begin(),
                          [norm]__device__(double x){return x*norm;});
        dt=ev.stop(); if(rec) t.select+=dt*1e3;

        ev.start();
        CUSOLVER_CHECK(cusolverDnDgeqrf(solver,c,n,pSk,c,d_tau,d_wsk,lw_sk,d_isk));
        dt=ev.stop(); if(rec) t.skqr+=dt*1e3;

        thrust::fill(d_RsTR.begin(),d_RsTR.end(),0.);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(c*n),TriD64{c,n,pSk,pRs});
        thrust::copy(d_RsTR.begin(),d_RsTR.end(),d_G.begin());
        cholqr_timed(blas,solver,pA,pG,d_wch,lw_ch,d_ich,m,n,t,rec,ev);
    };

    one(false); CUDA_CHECK(cudaDeviceSynchronize());
    wall.start(); for(int i=0;i<trials;++i) one(true);
    t.total=wall.stop()*1e3/trials;

    auto avg=[&](double& x){x/=trials;};
    avg(t.diag);avg(t.fft);avg(t.select);avg(t.skqr);
    avg(t.trsm_pre);avg(t.gemm);avg(t.chol);avg(t.trsm_fin);

    cufftDestroy(fp);
    cudaFree(d_tau);cudaFree(d_wsk);cudaFree(d_isk);
    cudaFree(d_wch);cudaFree(d_ich);
    return t;
}

// ===========================================================================
// MRCQR FP32 profiler
// ===========================================================================
static MRCQRTimes profile_fp32(cublasHandle_t blas, cusolverDnHandle_t solver,
                                thrust::device_vector<double>& d_A,
                                const thrust::device_vector<double>& d_Abak,
                                int m, int n, int trials)
{
    const int c=3*n, half_r=m/2+1;
    const float normf=(float)std::sqrt((double)c/m);
    const int BLK=256, GRD=(int)std::min(65535LL,((long long)m*n+BLK-1)/BLK);

    thrust::device_vector<int8_t>    d_D(m);
    thrust::device_vector<int>       d_S(c);
    thrust::device_vector<float>     d_FA((long long)m*n);
    thrust::device_vector<cufftComplex> d_fft((long long)n*half_r);
    thrust::device_vector<float>     d_sk(c*n);
    thrust::device_vector<double>    d_RsTR(n*n,0.);
    thrust::device_vector<double>    d_G(n*n);

    double *pA; int8_t *pD=thrust::raw_pointer_cast(d_D.data());
    int *pS=thrust::raw_pointer_cast(d_S.data());
    float *pFA=thrust::raw_pointer_cast(d_FA.data());
    auto  *pFft=thrust::raw_pointer_cast(d_fft.data());
    float *pSk=thrust::raw_pointer_cast(d_sk.data());
    double *pRs=thrust::raw_pointer_cast(d_RsTR.data());
    double *pG=thrust::raw_pointer_cast(d_G.data());

    cufftHandle fp; int fn[1]={m};
    CUFFT_CHECK(cufftPlanMany(&fp,1,fn,fn,1,m,fn,1,half_r,CUFFT_R2C,n));

    int lw_sk=0; float *d_tau=nullptr,*d_wsk=nullptr; int *d_isk=nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau,n*sizeof(float)));
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver,c,n,pSk,c,&lw_sk));
    CUDA_CHECK(cudaMalloc(&d_wsk,lw_sk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_isk,sizeof(int)));

    int lw_ch=0; double *d_wch=nullptr; int *d_ich=nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_UPPER,n,pG,n,&lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch,lw_ch*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich,sizeof(int)));

    MRCQRTimes t={}; EV ev, wall;

    auto one=[&](bool rec){
        thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
        pA=thrust::raw_pointer_cast(d_A.data()); double dt;
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m),d_D.begin(),RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c),d_S.begin(),RandSamp{m});

        ev.start(); diag_fp32<<<GRD,BLK>>>(pA,pD,pFA,m,n);
        CUDA_CHECK(cudaGetLastError()); dt=ev.stop(); if(rec) t.diag+=dt*1e3;

        ev.start(); CUFFT_CHECK(cufftExecR2C(fp,pFA,pFft));
        dt=ev.stop(); if(rec) t.fft+=dt*1e3;

        ev.start();
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(c*n),
                          d_sk.begin(),SelF32{pFft,pS,c,m,half_r});
        thrust::transform(d_sk.begin(),d_sk.end(),d_sk.begin(),
                          [normf]__device__(float x){return x*normf;});
        dt=ev.stop(); if(rec) t.select+=dt*1e3;

        ev.start();
        CUSOLVER_CHECK(cusolverDnSgeqrf(solver,c,n,pSk,c,d_tau,d_wsk,lw_sk,d_isk));
        dt=ev.stop(); if(rec) t.skqr+=dt*1e3;

        thrust::fill(d_RsTR.begin(),d_RsTR.end(),0.);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(c*n),TriF32toF64{c,n,pSk,pRs});
        thrust::copy(d_RsTR.begin(),d_RsTR.end(),d_G.begin());
        cholqr_timed(blas,solver,pA,pG,d_wch,lw_ch,d_ich,m,n,t,rec,ev);
    };

    one(false); CUDA_CHECK(cudaDeviceSynchronize());
    wall.start(); for(int i=0;i<trials;++i) one(true);
    t.total=wall.stop()*1e3/trials;

    auto avg=[&](double& x){x/=trials;};
    avg(t.diag);avg(t.fft);avg(t.select);avg(t.skqr);
    avg(t.trsm_pre);avg(t.gemm);avg(t.chol);avg(t.trsm_fin);

    cufftDestroy(fp);
    cudaFree(d_tau);cudaFree(d_wsk);cudaFree(d_isk);
    cudaFree(d_wch);cudaFree(d_ich);
    return t;
}

// ===========================================================================
// MRCQR FP16 profiler
// ===========================================================================
static MRCQRTimes profile_fp16(cublasHandle_t blas, cusolverDnHandle_t solver,
                                thrust::device_vector<double>& d_A,
                                const thrust::device_vector<double>& d_Abak,
                                int m, int n, int trials)
{
    const int c=3*n, half_r=m/2+1;
    const float normf=(float)std::sqrt((double)c/m);
    const int BLK=256, GRD=(int)std::min(65535LL,((long long)m*n+BLK-1)/BLK);

    thrust::device_vector<int8_t>  d_D(m);
    thrust::device_vector<int>     d_S(c);
    thrust::device_vector<__half>  d_FA((long long)m*n);
    thrust::device_vector<__half2> d_fft((long long)n*half_r);
    thrust::device_vector<float>   d_sk(c*n);
    thrust::device_vector<double>  d_RsTR(n*n,0.);
    thrust::device_vector<double>  d_G(n*n);

    double *pA; int8_t *pD=thrust::raw_pointer_cast(d_D.data());
    int *pS=thrust::raw_pointer_cast(d_S.data());
    __half *pFA=thrust::raw_pointer_cast(d_FA.data());
    __half2 *pFft=thrust::raw_pointer_cast(d_fft.data());
    float *pSk=thrust::raw_pointer_cast(d_sk.data());
    double *pRs=thrust::raw_pointer_cast(d_RsTR.data());
    double *pG=thrust::raw_pointer_cast(d_G.data());

    cufftHandle fp; CUFFT_CHECK(cufftCreate(&fp));
    long long n1[1]={m}; size_t ws=0;
    CUFFT_CHECK(cufftXtMakePlanMany(fp,1,n1,n1,1,m,CUDA_R_16F,
                                     n1,1,half_r,CUDA_C_16F,n,&ws,CUDA_R_16F));

    int lw_sk=0; float *d_tau=nullptr,*d_wsk=nullptr; int *d_isk=nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau,n*sizeof(float)));
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver,c,n,pSk,c,&lw_sk));
    CUDA_CHECK(cudaMalloc(&d_wsk,lw_sk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_isk,sizeof(int)));

    int lw_ch=0; double *d_wch=nullptr; int *d_ich=nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_UPPER,n,pG,n,&lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch,lw_ch*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich,sizeof(int)));

    MRCQRTimes t={}; EV ev, wall;

    auto one=[&](bool rec){
        thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
        pA=thrust::raw_pointer_cast(d_A.data()); double dt;
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m),d_D.begin(),RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c),d_S.begin(),RandSamp{m});

        ev.start(); diag_fp16<<<GRD,BLK>>>(pA,pD,pFA,m,n);
        CUDA_CHECK(cudaGetLastError()); dt=ev.stop(); if(rec) t.diag+=dt*1e3;

        ev.start();
        CUFFT_CHECK(cufftXtExec(fp,(void*)pFA,(void*)pFft,CUFFT_FORWARD));
        dt=ev.stop(); if(rec) t.fft+=dt*1e3;

        ev.start();
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(c*n),
                          d_sk.begin(),SelF16{pFft,pS,c,m,half_r});
        thrust::transform(d_sk.begin(),d_sk.end(),d_sk.begin(),
                          [normf]__device__(float x){return x*normf;});
        dt=ev.stop(); if(rec) t.select+=dt*1e3;

        ev.start();
        CUSOLVER_CHECK(cusolverDnSgeqrf(solver,c,n,pSk,c,d_tau,d_wsk,lw_sk,d_isk));
        dt=ev.stop(); if(rec) t.skqr+=dt*1e3;

        thrust::fill(d_RsTR.begin(),d_RsTR.end(),0.);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(c*n),TriF32toF64{c,n,pSk,pRs});
        thrust::copy(d_RsTR.begin(),d_RsTR.end(),d_G.begin());
        cholqr_timed(blas,solver,pA,pG,d_wch,lw_ch,d_ich,m,n,t,rec,ev);
    };

    one(false); CUDA_CHECK(cudaDeviceSynchronize());
    wall.start(); for(int i=0;i<trials;++i) one(true);
    t.total=wall.stop()*1e3/trials;

    auto avg=[&](double& x){x/=trials;};
    avg(t.diag);avg(t.fft);avg(t.select);avg(t.skqr);
    avg(t.trsm_pre);avg(t.gemm);avg(t.chol);avg(t.trsm_fin);

    cufftDestroy(fp);
    cudaFree(d_tau);cudaFree(d_wsk);cudaFree(d_isk);
    cudaFree(d_wch);cudaFree(d_ich);
    return t;
}

// ===========================================================================
// Randomized SVD preconditioner profiler
// ===========================================================================
static MRSQRTimes profile_mrsqr(cublasHandle_t blas, cusolverDnHandle_t solver,
                                thrust::device_vector<double>& d_A,
                                const thrust::device_vector<double>& d_Abak,
                                int m, int n, int trials, bool use_svqb_postprocess,
                                ProfileSvd svd_kind)
{
    const int c=3*n, half_r=m/2+1;
    const double norm=std::sqrt((double)c/m);
    const int BLK=256, GRD=(int)std::min(65535LL,((long long)m*n+BLK-1)/BLK);
    const double one=1., zero=0.;

    thrust::device_vector<int8_t> d_D(m);
    thrust::device_vector<int> d_S(c);
    thrust::device_vector<double> d_FA((long long)m*n);
    thrust::device_vector<cufftDoubleComplex> d_fft((long long)n*half_r);
    thrust::device_vector<double> d_sk((long long)c*n);
    thrust::device_vector<double> d_A1((long long)m*n);
    thrust::device_vector<double> d_VT(n*n), d_sigma(n), d_G(n*n);
    thrust::device_vector<double> d_B((long long)m*n);
    thrust::device_vector<double> d_M(n*n), d_eig(n), d_Dinv(n), d_rsqrt(n);

    double *pA=thrust::raw_pointer_cast(d_A.data());
    double *pFA=thrust::raw_pointer_cast(d_FA.data());
    int8_t *pD=thrust::raw_pointer_cast(d_D.data());
    int *pS=thrust::raw_pointer_cast(d_S.data());
    auto *pFft=thrust::raw_pointer_cast(d_fft.data());
    double *pSk=thrust::raw_pointer_cast(d_sk.data());
    double *pA1=thrust::raw_pointer_cast(d_A1.data());
    double *pVT=thrust::raw_pointer_cast(d_VT.data());
    double *pSigma=thrust::raw_pointer_cast(d_sigma.data());
    double *pG=thrust::raw_pointer_cast(d_G.data());
    double *pB=thrust::raw_pointer_cast(d_B.data());
    double *pM=thrust::raw_pointer_cast(d_M.data());
    double *pEig=thrust::raw_pointer_cast(d_eig.data());
    double *pDinv=thrust::raw_pointer_cast(d_Dinv.data());
    double *pRsqrt=thrust::raw_pointer_cast(d_rsqrt.data());

    cufftHandle fp; int fn[1]={m};
    CUFFT_CHECK(cufftPlanMany(&fp,1,fn,fn,1,m,fn,1,half_r,CUFFT_D2Z,n));

    int lw_svd=0; double *d_wsvd=nullptr; int svd_info=0; double jacobi_tolerance=1e-15; int jacobi_max_sweeps=20;
    gesvdjInfo_t gesvdj_params=nullptr;
    cusolverDnParams_t gesvdp_params=nullptr;
    size_t gesvdp_device_bytes=0, gesvdp_host_bytes=0;
    void* d_gesvdp_work=nullptr;
    std::vector<unsigned char> h_gesvdp_work;
    if (svd_kind == ProfileSvd::Gesvd) {
        CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(solver,c,n,&lw_svd));
        CUDA_CHECK(cudaMalloc(&d_wsvd,lw_svd*sizeof(double)));
    } else if (svd_kind == ProfileSvd::Gesvdj) {
          CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params,jacobi_tolerance));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params,jacobi_max_sweeps));
      
        CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(
            solver,CUSOLVER_EIG_MODE_VECTOR,1,c,n,pSk,c,pSigma,
            pA1,m,pVT,n,&lw_svd,gesvdj_params));
        CUDA_CHECK(cudaMalloc(&d_wsvd,lw_svd*sizeof(double)));

    } else {
        CUSOLVER_CHECK(cusolverDnCreateParams(&gesvdp_params));
        CUSOLVER_CHECK(cusolverDnXgesvdp_bufferSize(
            solver,gesvdp_params,CUSOLVER_EIG_MODE_VECTOR,1,c,n,
            CUDA_R_64F,pSk,c,CUDA_R_64F,pSigma,CUDA_R_64F,pA1,m,
            CUDA_R_64F,pVT,n,CUDA_R_64F,
            &gesvdp_device_bytes,&gesvdp_host_bytes));
        CUDA_CHECK(cudaMalloc(&d_gesvdp_work,gesvdp_device_bytes));
        h_gesvdp_work.resize(gesvdp_host_bytes);
    }
    thrust::device_vector<int> d_svd_info(1);
    int* p_svd_info=thrust::raw_pointer_cast(d_svd_info.data());
    double gesvdp_err_sigma=0.0;

    int lw_ch=0; double *d_wch=nullptr; int *d_ich=nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_UPPER,
                                               n,pG,n,&lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch,lw_ch*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich,sizeof(int)));

    int lw_ev=0; double *d_wev=nullptr; int *d_iev=nullptr;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver,CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER,n,pG,n,
                                               pEig,&lw_ev));
    CUDA_CHECK(cudaMalloc(&d_wev,lw_ev*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_iev,sizeof(int)));

    MRSQRTimes t={}; EV ev, wall;
    auto one_trial=[&](bool rec) {
        thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
        pA=thrust::raw_pointer_cast(d_A.data());
        double dt;

        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m),d_D.begin(),RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c),d_S.begin(),RandSamp{m});

        ev.start();
        diag_fp64<<<GRD,BLK>>>(pA,pD,pFA,m,n);
        CUDA_CHECK(cudaGetLastError());
        dt=ev.stop(); if(rec) t.diag+=dt*1e3;

        ev.start();
        CUFFT_CHECK(cufftExecD2Z(fp,pFA,pFft));
        dt=ev.stop(); if(rec) t.fft+=dt*1e3;

        ev.start();
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(c*n),d_sk.begin(),
                          SelF64{pFft,pS,c,m,half_r});
        thrust::transform(d_sk.begin(),d_sk.end(),d_sk.begin(),
                          [norm]__device__(double x){return x*norm;});
        dt=ev.stop(); if(rec) t.select+=dt*1e3;

        ev.start();
        if (svd_kind == ProfileSvd::Gesvd) {
            CUSOLVER_CHECK(cusolverDnDgesvd(solver,'N','A',c,n,pSk,c,pSigma,
                                            nullptr,1,pVT,n,d_wsvd,lw_svd,
                                            nullptr,&svd_info));
        } else if (svd_kind == ProfileSvd::Gesvdj) {
            CUSOLVER_CHECK(cusolverDnDgesvdj(
                solver,CUSOLVER_EIG_MODE_VECTOR,1,c,n,pSk,c,pSigma,
                pA1,m,pVT,n,d_wsvd,lw_svd,p_svd_info,gesvdj_params));
        } else {
            CUSOLVER_CHECK(cusolverDnXgesvdp(
                solver,gesvdp_params,CUSOLVER_EIG_MODE_VECTOR,1,c,n,
                CUDA_R_64F,pSk,c,CUDA_R_64F,pSigma,CUDA_R_64F,pA1,m,
                CUDA_R_64F,pVT,n,CUDA_R_64F,d_gesvdp_work,
                gesvdp_device_bytes,h_gesvdp_work.data(),gesvdp_host_bytes,
                p_svd_info,&gesvdp_err_sigma));
        }
        dt=ev.stop(); if(rec) t.svd+=dt*1e3;

        ev.start();
        CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_N,
                     svd_kind == ProfileSvd::Gesvd ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 m,n,n,
                                 &one,pA,m,pVT,n,&zero,pA1,m));
        dt=ev.stop(); if(rec) t.pre_gemm+=dt*1e3;

        ev.start();
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(m*n),d_A1.begin(),
                          [pA1,pSigma,m]__device__(int k){
                              return pA1[k]/pSigma[k/m];
                          });
        dt=ev.stop(); if(rec) t.pre_scale+=dt*1e3;

        if (!use_svqb_postprocess) {
            ev.start();
            CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,
                                     &one,pA1,m,pA1,m,&zero,pG,n));
            dt=ev.stop(); if(rec) t.chol_gemm+=dt*1e3;

            ev.start();
            CUSOLVER_CHECK(cusolverDnDpotrf(solver,CUBLAS_FILL_MODE_UPPER,
                                             n,pG,n,d_wch,lw_ch,d_ich));
            dt=ev.stop(); if(rec) t.chol+=dt*1e3;

            ev.start();
            CUBLAS_CHECK(cublasDtrsm(blas,CUBLAS_SIDE_RIGHT,
                                     CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT,m,n,&one,pG,n,pA1,m));
            dt=ev.stop(); if(rec) t.chol_trsm+=dt*1e3;
        } else {
            ev.start();
            CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,
                                     &one,pA1,m,pA1,m,&zero,pG,n));
            dt=ev.stop(); if(rec) t.svqb_gram+=dt*1e3;

            ev.start();
            thrust::for_each(thrust::counting_iterator<int>(0),
                             thrust::counting_iterator<int>(n),
                             [pG,pDinv,n]__device__(int i) {
                                 double v=pG[i+n*i];
                                 pDinv[i]=(v>0.)?1./sqrt(v):1.;
                             });
            CUBLAS_CHECK(cublasDdgmm(blas,CUBLAS_SIDE_LEFT,n,n,pG,n,pDinv,1,pG,n));
            CUBLAS_CHECK(cublasDdgmm(blas,CUBLAS_SIDE_RIGHT,n,n,pG,n,pDinv,1,pG,n));
            dt=ev.stop(); if(rec) t.svqb_norm+=dt*1e3;

            ev.start();
            CUSOLVER_CHECK(cusolverDnDsyevd(solver,CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_LOWER,n,pG,n,
                                             pEig,d_wev,lw_ev,d_iev));
            double maxeig;
            CUDA_CHECK(cudaMemcpy(&maxeig,pEig+n-1,sizeof(double),cudaMemcpyDeviceToHost));
            thrust::transform(thrust::device_pointer_cast(pEig),
                              thrust::device_pointer_cast(pEig)+n,
                              thrust::device_pointer_cast(pRsqrt),
                              [maxeig]__device__(double s){
                                  double floor_=maxeig*1e-15;
                                  return 1./sqrt(max(s,floor_));
                              });
            dt=ev.stop(); if(rec) t.svqb_eig+=dt*1e3;

            ev.start();
            CUBLAS_CHECK(cublasDdgmm(blas,CUBLAS_SIDE_RIGHT,n,n,pG,n,pRsqrt,1,pM,n));
            CUBLAS_CHECK(cublasDdgmm(blas,CUBLAS_SIDE_LEFT,n,n,pM,n,pDinv,1,pM,n));
            CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,m,n,n,
                                     &one,pA1,m,pM,n,&zero,pB,m));
            dt=ev.stop(); if(rec) t.svqb_apply+=dt*1e3;
        }
    };

    one_trial(false); CUDA_CHECK(cudaDeviceSynchronize());
    wall.start();
    for (int i=0;i<trials;++i) one_trial(true);
    t.total=wall.stop()*1e3/trials;

    auto avg=[&](double& x){x/=trials;};
    avg(t.diag); avg(t.fft); avg(t.select); avg(t.svd);
    avg(t.pre_gemm); avg(t.pre_scale);
    avg(t.chol_gemm); avg(t.chol); avg(t.chol_trsm);
    avg(t.svqb_gram); avg(t.svqb_norm); avg(t.svqb_eig); avg(t.svqb_apply);

    cufftDestroy(fp);
    cudaFree(d_wsvd); cudaFree(d_gesvdp_work); cudaFree(d_wch); cudaFree(d_ich);
    if (gesvdj_params) CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
    if (gesvdp_params) CUSOLVER_CHECK(cusolverDnDestroyParams(gesvdp_params));
    cudaFree(d_wev); cudaFree(d_iev);
    return t;
}

static MREigTimes profile_mreig_svqb(cublasHandle_t blas, cusolverDnHandle_t solver,
                                     thrust::device_vector<double>& d_A,
                                     const thrust::device_vector<double>& d_Abak,
                                     int m, int n, int trials)
{
    const int c=3*n, half_r=m/2+1;
    const double norm=std::sqrt((double)c/m), one=1., zero=0.;
    const int BLK=256, GRD=(int)std::min(65535LL,((long long)m*n+BLK-1)/BLK);
    thrust::device_vector<int8_t> d_D(m); thrust::device_vector<int> d_S(c);
    thrust::device_vector<double> d_FA((long long)m*n), d_sk((long long)c*n);
    thrust::device_vector<cufftDoubleComplex> d_fft((long long)n*half_r);
    thrust::device_vector<double> d_SkG(n*n), d_inv(n), d_pre(n*n);
    thrust::device_vector<double> d_A1((long long)m*n), d_B((long long)m*n);
    thrust::device_vector<double> d_G(n*n), d_M(n*n), d_eig(n), d_Dinv(n), d_rsqrt(n);
    double *pA=thrust::raw_pointer_cast(d_A.data()), *pFA=thrust::raw_pointer_cast(d_FA.data());
    int8_t *pD=thrust::raw_pointer_cast(d_D.data()); int *pS=thrust::raw_pointer_cast(d_S.data());
    auto *pFft=thrust::raw_pointer_cast(d_fft.data()); double *pSk=thrust::raw_pointer_cast(d_sk.data());
    double *pSkG=thrust::raw_pointer_cast(d_SkG.data()), *pInv=thrust::raw_pointer_cast(d_inv.data());
    double *pPre=thrust::raw_pointer_cast(d_pre.data()), *pA1=thrust::raw_pointer_cast(d_A1.data());
    double *pB=thrust::raw_pointer_cast(d_B.data()), *pG=thrust::raw_pointer_cast(d_G.data());
    double *pM=thrust::raw_pointer_cast(d_M.data()), *pEig=thrust::raw_pointer_cast(d_eig.data());
    double *pDinv=thrust::raw_pointer_cast(d_Dinv.data()), *pRsqrt=thrust::raw_pointer_cast(d_rsqrt.data());

    cufftHandle fp; int fn[1]={m};
    CUFFT_CHECK(cufftPlanMany(&fp,1,fn,fn,1,m,fn,1,half_r,CUFFT_D2Z,n));
    int lw=0; double *d_work=nullptr; int *d_info=nullptr;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver,CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER,n,pSkG,n,pEig,&lw));
    CUDA_CHECK(cudaMalloc(&d_work,lw*sizeof(double))); CUDA_CHECK(cudaMalloc(&d_info,sizeof(int)));
    int lwsv=0; double *d_wsv=nullptr; int *d_isv=nullptr;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver,CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER,n,pG,n,pEig,&lwsv));
    CUDA_CHECK(cudaMalloc(&d_wsv,lwsv*sizeof(double))); CUDA_CHECK(cudaMalloc(&d_isv,sizeof(int)));
    MREigTimes t={}; EV ev,wall;
    auto one_trial=[&](bool rec){
        thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
        double dt; pA=thrust::raw_pointer_cast(d_A.data());
        thrust::transform(thrust::counting_iterator<unsigned>(0),thrust::counting_iterator<unsigned>(m),d_D.begin(),RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),thrust::counting_iterator<unsigned>(c),d_S.begin(),RandSamp{m});
        ev.start(); diag_fp64<<<GRD,BLK>>>(pA,pD,pFA,m,n); CUDA_CHECK(cudaGetLastError()); dt=ev.stop(); if(rec)t.diag+=dt*1e3;
        ev.start(); CUFFT_CHECK(cufftExecD2Z(fp,pFA,pFft)); dt=ev.stop(); if(rec)t.fft+=dt*1e3;
        ev.start(); thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(c*n),d_sk.begin(),SelF64{pFft,pS,c,m,half_r});
        thrust::transform(d_sk.begin(),d_sk.end(),d_sk.begin(),[norm]__device__(double x){return x*norm;});
        dt=ev.stop(); if(rec)t.select+=dt*1e3;
        ev.start(); CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,c,&one,pSk,c,pSk,c,&zero,pSkG,n)); dt=ev.stop(); if(rec)t.gram+=dt*1e3;
        ev.start(); CUSOLVER_CHECK(cusolverDnDsyevd(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,pSkG,n,pEig,d_work,lw,d_info));
        double maxeig; CUDA_CHECK(cudaMemcpy(&maxeig,pEig+n-1,sizeof(double),cudaMemcpyDeviceToHost));
        thrust::transform(thrust::device_pointer_cast(pEig),thrust::device_pointer_cast(pEig)+n,thrust::device_pointer_cast(pInv),[maxeig]__device__(double s){return 1./sqrt(max(s,maxeig*1e-15));});
        dt=ev.stop(); if(rec)t.eig+=dt*1e3;
        ev.start(); CUBLAS_CHECK(cublasDdgmm(blas,CUBLAS_SIDE_RIGHT,n,n,pSkG,n,pInv,1,pPre,n)); dt=ev.stop(); if(rec)t.scale+=dt*1e3;
        ev.start(); CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,m,n,n,&one,pA,m,pPre,n,&zero,pA1,m)); dt=ev.stop(); if(rec)t.pre_gemm+=dt*1e3;
        ev.start();
        CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,&one,pA1,m,pA1,m,&zero,pG,n));
        dt=ev.stop(); if(rec)t.svqb_gram+=dt*1e3;
        ev.start(); thrust::for_each(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(n),[pG,pDinv,n]__device__(int i){double v=pG[i+n*i];pDinv[i]=(v>0.)?1./sqrt(v):1.;});
        CUBLAS_CHECK(cublasDdgmm(blas,CUBLAS_SIDE_LEFT,n,n,pG,n,pDinv,1,pG,n)); CUBLAS_CHECK(cublasDdgmm(blas,CUBLAS_SIDE_RIGHT,n,n,pG,n,pDinv,1,pG,n));
        dt=ev.stop(); if(rec)t.svqb_norm+=dt*1e3;
        ev.start(); CUSOLVER_CHECK(cusolverDnDsyevd(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,pG,n,pEig,d_wsv,lwsv,d_isv));
        CUDA_CHECK(cudaMemcpy(&maxeig,pEig+n-1,sizeof(double),cudaMemcpyDeviceToHost)); thrust::transform(thrust::device_pointer_cast(pEig),thrust::device_pointer_cast(pEig)+n,thrust::device_pointer_cast(pRsqrt),[maxeig]__device__(double s){return 1./sqrt(max(s,maxeig*1e-15));});
        dt=ev.stop(); if(rec)t.svqb_eig+=dt*1e3;
        ev.start(); CUBLAS_CHECK(cublasDdgmm(blas,CUBLAS_SIDE_RIGHT,n,n,pG,n,pRsqrt,1,pM,n)); CUBLAS_CHECK(cublasDdgmm(blas,CUBLAS_SIDE_LEFT,n,n,pM,n,pDinv,1,pM,n)); CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,m,n,n,&one,pA1,m,pM,n,&zero,pB,m)); dt=ev.stop(); if(rec)t.svqb_apply+=dt*1e3;
    };
    one_trial(false); CUDA_CHECK(cudaDeviceSynchronize()); wall.start(); for(int i=0;i<trials;++i)one_trial(true); t.total=wall.stop()*1e3/trials;
    auto avg=[&](double& x){x/=trials;}; avg(t.diag);avg(t.fft);avg(t.select);avg(t.gram);avg(t.eig);avg(t.scale);avg(t.pre_gemm);avg(t.svqb_gram);avg(t.svqb_norm);avg(t.svqb_eig);avg(t.svqb_apply);
    cufftDestroy(fp); cudaFree(d_work);cudaFree(d_info);cudaFree(d_wsv);cudaFree(d_isv); return t;
}

// ===========================================================================
// CholQR2 profiler
// ===========================================================================
static CQR2Times profile_cholqr2(cublasHandle_t blas, cusolverDnHandle_t solver,
                                  thrust::device_vector<double>& d_A,
                                  const thrust::device_vector<double>& d_Abak,
                                  int m, int n, int trials)
{
    const double a=1., b=0.;
    cublasFillMode_t uplo=CUBLAS_FILL_MODE_UPPER;

    thrust::device_vector<double> d_G(n*n), d_G1(n*n);
    double *pA, *pG=thrust::raw_pointer_cast(d_G.data());
    double *pG1=thrust::raw_pointer_cast(d_G1.data());

    int lw=0; double *d_work=nullptr; int *d_info=nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver,uplo,n,pG,n,&lw));
    CUDA_CHECK(cudaMalloc(&d_work,lw*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info,sizeof(int)));

    CQR2Times t={}; EV ev, wall;

    auto one=[&](bool rec){
        thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
        pA=thrust::raw_pointer_cast(d_A.data()); double dt;

        ev.start();
        CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,
                                 &a,pA,m,pA,m,&b,pG,n));
        dt=ev.stop(); if(rec) t.gemm1+=dt*1e3;

        ev.start();
        CUSOLVER_CHECK(cusolverDnDpotrf(solver,uplo,n,pG,n,d_work,lw,d_info));
        dt=ev.stop(); if(rec) t.chol1+=dt*1e3;

        ev.start();
        CUBLAS_CHECK(cublasDtrsm(blas,CUBLAS_SIDE_RIGHT,uplo,CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT,m,n,&a,pG,n,pA,m));
        dt=ev.stop(); if(rec) t.trsm1+=dt*1e3;

        ev.start();
        CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,
                                 &a,pA,m,pA,m,&b,pG1,n));
        dt=ev.stop(); if(rec) t.gemm2+=dt*1e3;

        ev.start();
        CUSOLVER_CHECK(cusolverDnDpotrf(solver,uplo,n,pG1,n,d_work,lw,d_info));
        dt=ev.stop(); if(rec) t.chol2+=dt*1e3;

        ev.start();
        CUBLAS_CHECK(cublasDtrsm(blas,CUBLAS_SIDE_RIGHT,uplo,CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT,m,n,&a,pG1,n,pA,m));
        dt=ev.stop(); if(rec) t.trsm2+=dt*1e3;
    };

    one(false); CUDA_CHECK(cudaDeviceSynchronize());
    wall.start(); for(int i=0;i<trials;++i) one(true);
    t.total=wall.stop()*1e3/trials;

    auto avg=[&](double& x){x/=trials;};
    avg(t.gemm1);avg(t.chol1);avg(t.trsm1);
    avg(t.gemm2);avg(t.chol2);avg(t.trsm2);

    cudaFree(d_work); cudaFree(d_info);
    return t;
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------
static void print_mrcqr(const char* name, const MRCQRTimes& t)
{
    printf("%-16s %7.3f %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f %7.3f | %7.3f  %7.3f  %7.3f\n",
           name,
           t.diag, t.fft, t.select, t.skqr,
           t.trsm_pre, t.gemm, t.chol, t.trsm_fin,
           t.sketch(), t.cholqr(), t.total);
}
static void print_cqr2(const CQR2Times& t)
{
    printf("%-16s %7.3f %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f %7.3f | %7s  %7.3f  %7.3f\n",
           "CholQR2",
           t.gemm1, t.chol1, t.trsm1, 0.,
           t.gemm2, t.chol2, t.trsm2, 0.,
           "n/a", t.gemm1+t.chol1+t.trsm1+t.gemm2+t.chol2+t.trsm2, t.total);
}
static void print_mrsqr(const char* name, const MRSQRTimes& t, bool show_svqb_stages)
{
    printf("%-16s %7.3f %7.3f %7.3f %7.3f | %7.3f %7.3f |",
           name, t.diag, t.fft, t.select, t.svd, t.pre_gemm, t.pre_scale);
    if (show_svqb_stages) {
         printf(" %7.3f %7.3f %7.3f %7.3f | %7.3f\n",
             t.svqb_gram, t.svqb_norm, t.svqb_eig, t.svqb_apply, t.total);
    } else {
        printf(" %7.3f %7.3f %7.3f %7s | %7.3f\n",
               t.chol_gemm, t.chol, t.chol_trsm, "n/a", t.total);
    }
}
static void print_mrsqr_header()
{
    printf("%-16s %7s %7s %7s %7s | %7s %7s | %7s %7s %7s %7s | %7s\n",
           "Method", "DiagMlt", "FFT", "Select", "SVD",
           "PreGEMM", "PreScale", "Stage1", "Stage2", "Stage3", "Stage4", "Total");
    //printf("  Stage1-3 = CholQR GEMM/Chol/TRSM for mrsqr; Stage1-4 = SVQB Gram/Norm/Eig/Apply for mrsqr2\n");
    printf("%s\n", std::string(126,'-').c_str());
}
static void print_mreig(const MREigTimes& t)
{
    printf("%-16s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f %7.3f\n",
        "mreig_svqb", t.diag, t.fft, t.select, t.gram, t.eig,
        t.scale, t.pre_gemm, t.svqb_gram, t.svqb_norm, t.svqb_eig,
        t.svqb_apply);
    printf("  columns: Diag FFT Select SketchGram SketchEig Scale PreGEMM SVQBGram SVQBNorm SVQBEig SVQBApply\n");
    printf("  total: %.3f ms\n", t.total);
}
static void print_header()
{
    printf("%-16s %7s %7s %7s %7s | %7s %7s %7s %7s | %7s  %7s  %7s\n",
           "Method",
           "DiagMlt","FFT","Select","SkQR",
           "TrsmPre","GEMM","Chol","TrsmFin",
           "Sketch","CholQR","Total");
    printf("%s\n", std::string(110,'-').c_str());
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    const int m = 1 << 15;
    const int TRIALS = 10;
    // n = 2^4 to 2^11. Peak device memory at n=2048, m=2^20 (FP64): ~65 GB — H100 only.
    const std::vector<int> n_sizes = {16, 32, 64, 128, 256, 512};//, 1024, 2048};

    cublasHandle_t    blas;   CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    printf("\n");
    printf("================================================================\n");
    printf("  MRCQR vs CholQR2 Sub-Operation Profile\n");
    printf("  m = %d (2^20),  %d trials.  All times in ms.\n", m, TRIALS);
    printf("  CholQR2 columns: Pass1(GEMM|Chol|TRSM) | Pass2(GEMM|Chol|TRSM)\n");
    printf("================================================================\n");

    int max_n = n_sizes.back();
    thrust::device_vector<double> d_A((long long)m*max_n), d_Abak((long long)m*max_n);
    thrust::transform(thrust::counting_iterator<unsigned>(0),
                      thrust::counting_iterator<unsigned>(m*max_n),
                      d_A.begin(), RandUnif());
    thrust::copy(d_A.begin(), d_A.end(), d_Abak.begin());

    for (int n : n_sizes) {
        printf("\n════  n = %d  (sketch c = %d)  ════\n", n, 3*n);
        print_header();

        MRCQRTimes profile_mrcqr_fp64 = profile_fp64(blas, solver, d_A, d_Abak, m, n, TRIALS);
        MRCQRTimes profile_mrcqr_fp32 = profile_fp32(blas, solver, d_A, d_Abak, m, n, TRIALS);
        MRCQRTimes profile_mrcqr_fp16 = profile_fp16(blas, solver, d_A, d_Abak, m, n, TRIALS);
        CQR2Times profile_cholqr2_result = profile_cholqr2(blas, solver, d_A, d_Abak, m, n, TRIALS);
        MRSQRTimes profile_mrsqr_result = profile_mrsqr(blas, solver, d_A, d_Abak, m, n, TRIALS, false, ProfileSvd::Gesvd);
        MRSQRTimes profile_mrsqr2_result = profile_mrsqr(blas, solver, d_A, d_Abak, m, n, TRIALS, true, ProfileSvd::Gesvd);
        MRSQRTimes profile_mrsqr2_gesvdj = profile_mrsqr(blas, solver, d_A, d_Abak, m, n, TRIALS, true, ProfileSvd::Gesvdj);
        MRSQRTimes profile_mrsqr2_gesvdp = profile_mrsqr(blas, solver, d_A, d_Abak, m, n, TRIALS, true, ProfileSvd::Gesvdp);
        MREigTimes profile_mreig_svqb_result = profile_mreig_svqb(blas, solver, d_A, d_Abak, m, n, TRIALS);

        print_mrcqr("MRCQR (FP64)", profile_mrcqr_fp64);
        print_mrcqr("MRCQR (FP32)", profile_mrcqr_fp32);
        print_mrcqr("MRCQR (FP16)", profile_mrcqr_fp16);
        print_cqr2(profile_cholqr2_result);

        printf("\n  Randomized SVD QR profiles\n");
        print_mrsqr_header();
        print_mrsqr("mrsqr", profile_mrsqr_result, false);
        print_mrsqr("mrsqr2", profile_mrsqr2_result, true);
        print_mrsqr("mrsqr2_gesvdj", profile_mrsqr2_gesvdj, true);
        print_mrsqr("mrsqr2_gesvdp", profile_mrsqr2_gesvdp, true);
        print_mreig(profile_mreig_svqb_result);

        // // Sketch overhead as % of MRCQR FP64 total
        // printf("\n  Sketch overhead (FP64): %.1f%% of total\n",
        //        100.*profile_mrcqr_fp64.sketch()/profile_mrcqr_fp64.total);
        // printf("  Sketch overhead (FP32): %.1f%% of total\n",
        //        100.*profile_mrcqr_fp32.sketch()/profile_mrcqr_fp32.total);
        // printf("  Sketch overhead (FP16): %.1f%% of total\n",
        //        100.*profile_mrcqr_fp16.sketch()/profile_mrcqr_fp16.total);
        // printf("  CholQR2 vs MRCQR FP64 speedup: %.2fx\n",
        //        profile_mrcqr_fp64.total / profile_cholqr2_result.total);
    }

    cublasDestroy(blas); cusolverDnDestroy(solver);
    return 0;
}
