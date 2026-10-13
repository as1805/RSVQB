// driver_scaling.cu
// Scaling experiment: fix n = 100, vary m = 2^17 to 2^24.
// A is generated directly on GPU (no host memory for A or Q).
// Methods: MRCQR (FP64 / FP32 / FP16) and CholQR2.
// Re-only sketch: c = 3n rows, SketchQR on c×n.
//
// Peak device memory at m=2^24, n=100 (FP64): ~54 GB.  H100 (80 GB) only.
// All times in ms, averaged over TRIALS runs.

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

// ---------------------------------------------------------------------------
// Random / diagonal / selection / upper-tri functors (Re-only sketch)
// ---------------------------------------------------------------------------
struct RandUnif {
    __host__ __device__ double operator()(unsigned i) const {
        thrust::default_random_engine r(i * 2654435761u);
        thrust::uniform_real_distribution<double> d(-1, 1);
        r.discard(i); return d(r); } };
struct RandRadI8 {
    __host__ __device__ int8_t operator()(unsigned i) const {
        thrust::default_random_engine r(i * 2654435761u);
        thrust::uniform_real_distribution<float> d(0, 1);
        r.discard(i); return d(r) < .5f ? (int8_t)-1 : (int8_t)1; } };
struct RandSamp { int N;
    __host__ __device__ int operator()(unsigned i) const {
        thrust::default_random_engine r(i * 2246822519u);
        thrust::uniform_int_distribution<int> d(0, N-1);
        r.discard(i); return d(r); } };

__global__ void diag_fp64(const double*A,const int8_t*D,double*FA,int m,int n){
    for(long long i=blockIdx.x*blockDim.x+threadIdx.x;i<(long long)m*n;i+=blockDim.x*gridDim.x){
        unsigned long long b=__double_as_longlong(A[i]);
        b^=(unsigned long long)(D[i%m]<0)<<63;FA[i]=__longlong_as_double(b);}}
__global__ void diag_fp32(const double*A,const int8_t*D,float*FA,int m,int n){
    for(long long i=blockIdx.x*blockDim.x+threadIdx.x;i<(long long)m*n;i+=blockDim.x*gridDim.x){
        unsigned b=__float_as_uint((float)A[i]);
        b^=(unsigned)(D[i%m]<0)<<31;FA[i]=__uint_as_float(b);}}
__global__ void diag_fp16(const double*A,const int8_t*D,__half*FA,int m,int n){
    for(long long i=blockIdx.x*blockDim.x+threadIdx.x;i<(long long)m*n;i+=blockDim.x*gridDim.x){
        unsigned short b=__half_as_ushort(__float2half((float)A[i]));
        b^=(unsigned short)(D[i%m]<0)<<15;FA[i]=__ushort_as_half(b);}}

// Re-only selectors
struct SelF64{const cufftDoubleComplex*fft;const int*idx;int c,m,rpc;
    __host__ __device__ double operator()(int i)const{
        int col=i/c,r=idx[i%c],ar=(r<=m/2)?r:(m-r);
        return fft[ar+col*rpc].x;}};
struct SelF32{const cufftComplex*fft;const int*idx;int c,m,rpc;
    __host__ __device__ float operator()(int i)const{
        int col=i/c,r=idx[i%c],ar=(r<=m/2)?r:(m-r);
        return fft[ar+col*rpc].x;}};
struct SelF16{const __half2*fft;const int*idx;int c,m,rpc;
    __host__ __device__ float operator()(int i)const{
        int col=i/c,r=idx[i%c],ar=(r<=m/2)?r:(m-r);
        return __half22float2(fft[ar+col*rpc]).x;}};

struct TriD64{int c,n;const double*src;double*dst;
    __host__ __device__ void operator()(int x)const{
        int row=x/n,col=x%n;
        if(row<=col&&row<n)dst[row+n*col]=src[row+c*col];}};
struct TriF32toF64{int c,n;const float*src;double*dst;
    __host__ __device__ void operator()(int x)const{
        int row=x/n,col=x%n;
        if(row<=col&&row<n)dst[row+n*col]=(double)src[row+c*col];}};

// Shared CholQR phase (in-place SIDE_RIGHT TRSM, no transposes)
static void cholqr(cublasHandle_t blas,cusolverDnHandle_t solver,
                   double*pA,double*pG,double*d_wch,int lw_ch,int*d_ich,int m,int n)
{
    const double a=1.,b=0.; cublasFillMode_t uplo=CUBLAS_FILL_MODE_UPPER;
    CUBLAS_CHECK(cublasDtrsm(blas,CUBLAS_SIDE_RIGHT,uplo,CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,m,n,&a,pG,n,pA,m));
    CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,
                             &a,pA,m,pA,m,&b,pG,n));
    CUSOLVER_CHECK(cusolverDnDpotrf(solver,uplo,n,pG,n,d_wch,lw_ch,d_ich));
    CUBLAS_CHECK(cublasDtrsm(blas,CUBLAS_SIDE_RIGHT,uplo,CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT,m,n,&a,pG,n,pA,m));
}

// ===========================================================================
// Timing functions — return average wall-clock time in ms.
// d_A is overwritten; restored from d_Abak each trial.
// ===========================================================================
static double time_fp64(cublasHandle_t blas,cusolverDnHandle_t solver,
                        thrust::device_vector<double>&d_A,
                        const thrust::device_vector<double>&d_Abak,
                        int m,int n,int trials)
{
    const int c=3*n,half_r=m/2+1;
    const double norm=std::sqrt((double)c/m);
    const int BLK=256,GRD=(int)std::min(65535LL,((long long)m*n+BLK-1)/BLK);

    thrust::device_vector<int8_t>             d_D(m);
    thrust::device_vector<int>                d_S(c);
    thrust::device_vector<double>             d_FA((long long)m*n);
    thrust::device_vector<cufftDoubleComplex> d_fft((long long)n*(half_r));
    thrust::device_vector<double>             d_sk(c*n);
    thrust::device_vector<double>             d_RsTR(n*n,0.),d_G(n*n);

    int8_t*pD=thrust::raw_pointer_cast(d_D.data());
    int*pS=thrust::raw_pointer_cast(d_S.data());
    double*pFA=thrust::raw_pointer_cast(d_FA.data());
    auto*pFft=thrust::raw_pointer_cast(d_fft.data());
    double*pSk=thrust::raw_pointer_cast(d_sk.data());
    double*pRs=thrust::raw_pointer_cast(d_RsTR.data());
    double*pG=thrust::raw_pointer_cast(d_G.data());

    cufftHandle fp; int fn[1]={m};
    CUFFT_CHECK(cufftPlanMany(&fp,1,fn,fn,1,m,fn,1,half_r,CUFFT_D2Z,n));

    int lw_sk=0; double*d_tau=nullptr,*d_wsk=nullptr; int*d_isk=nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau,n*sizeof(double)));
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver,c,n,pSk,c,&lw_sk));
    CUDA_CHECK(cudaMalloc(&d_wsk,lw_sk*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isk,sizeof(int)));

    int lw_ch=0; double*d_wch=nullptr; int*d_ich=nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_UPPER,n,pG,n,&lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch,lw_ch*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich,sizeof(int)));

    auto one=[&](){
        double*pA=thrust::raw_pointer_cast(d_A.data());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m),d_D.begin(),RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c),d_S.begin(),RandSamp{m});
        diag_fp64<<<GRD,BLK>>>(pA,pD,pFA,m,n); CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecD2Z(fp,pFA,pFft));
        thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(c*n),
                          d_sk.begin(),SelF64{pFft,pS,c,m,half_r});
        thrust::transform(d_sk.begin(),d_sk.end(),d_sk.begin(),
                          [norm]__device__(double x){return x*norm;});
        CUSOLVER_CHECK(cusolverDnDgeqrf(solver,c,n,pSk,c,d_tau,d_wsk,lw_sk,d_isk));
        thrust::fill(d_RsTR.begin(),d_RsTR.end(),0.);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(c*n),TriD64{c,n,pSk,pRs});
        thrust::copy(d_RsTR.begin(),d_RsTR.end(),d_G.begin());
        cholqr(blas,solver,pA,pG,d_wch,lw_ch,d_ich,m,n);
    };
    thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
    one(); CUDA_CHECK(cudaDeviceSynchronize());
    cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

CUBLAS_CHECK(cublasSetStream(blas, stream));
CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));
CUFFT_CHECK(cufftSetStream(fp, stream));
    EV wall; wall.start(stream);
    for(int t=0;t<trials;++t){thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());one();}
    double ms=wall.stop(stream)*1e3/trials;
    cufftDestroy(fp);cudaFree(d_tau);cudaFree(d_wsk);cudaFree(d_isk);cudaFree(d_wch);cudaFree(d_ich);
    return ms;
}

static double time_fp32(cublasHandle_t blas,cusolverDnHandle_t solver,
                        thrust::device_vector<double>&d_A,
                        const thrust::device_vector<double>&d_Abak,
                        int m,int n,int trials)
{
    const int c=3*n,half_r=m/2+1;
    const float normf=(float)std::sqrt((double)c/m);
    const int BLK=256,GRD=(int)std::min(65535LL,((long long)m*n+BLK-1)/BLK);

    thrust::device_vector<int8_t>    d_D(m);
    thrust::device_vector<int>       d_S(c);
    thrust::device_vector<float>     d_FA((long long)m*n);
    thrust::device_vector<cufftComplex> d_fft((long long)n*(half_r));
    thrust::device_vector<float>     d_sk(c*n);
    thrust::device_vector<double>    d_RsTR(n*n,0.),d_G(n*n);

    int8_t*pD=thrust::raw_pointer_cast(d_D.data());
    int*pS=thrust::raw_pointer_cast(d_S.data());
    float*pFA=thrust::raw_pointer_cast(d_FA.data());
    auto*pFft=thrust::raw_pointer_cast(d_fft.data());
    float*pSk=thrust::raw_pointer_cast(d_sk.data());
    double*pRs=thrust::raw_pointer_cast(d_RsTR.data());
    double*pG=thrust::raw_pointer_cast(d_G.data());

    cufftHandle fp; int fn[1]={m};
    CUFFT_CHECK(cufftPlanMany(&fp,1,fn,fn,1,m,fn,1,half_r,CUFFT_R2C,n));

    int lw_sk=0; float*d_tau=nullptr,*d_wsk=nullptr; int*d_isk=nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau,n*sizeof(float)));
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver,c,n,pSk,c,&lw_sk));
    CUDA_CHECK(cudaMalloc(&d_wsk,lw_sk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_isk,sizeof(int)));

    int lw_ch=0; double*d_wch=nullptr; int*d_ich=nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_UPPER,n,pG,n,&lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch,lw_ch*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich,sizeof(int)));

    auto one=[&](){
        double*pA=thrust::raw_pointer_cast(d_A.data());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m),d_D.begin(),RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c),d_S.begin(),RandSamp{m});
        diag_fp32<<<GRD,BLK>>>(pA,pD,pFA,m,n); CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecR2C(fp,pFA,pFft));
        thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(c*n),
                          d_sk.begin(),SelF32{pFft,pS,c,m,half_r});
        thrust::transform(d_sk.begin(),d_sk.end(),d_sk.begin(),
                          [normf]__device__(float x){return x*normf;});
        CUSOLVER_CHECK(cusolverDnSgeqrf(solver,c,n,pSk,c,d_tau,d_wsk,lw_sk,d_isk));
        thrust::fill(d_RsTR.begin(),d_RsTR.end(),0.);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(c*n),TriF32toF64{c,n,pSk,pRs});
        thrust::copy(d_RsTR.begin(),d_RsTR.end(),d_G.begin());
        cholqr(blas,solver,pA,pG,d_wch,lw_ch,d_ich,m,n);
    };
    thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
    cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

CUBLAS_CHECK(cublasSetStream(blas, stream));
CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));
CUFFT_CHECK(cufftSetStream(fp, stream));
    one(); CUDA_CHECK(cudaDeviceSynchronize());
    EV wall; wall.start(stream);
    for(int t=0;t<trials;++t){thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());one();}
    double ms=wall.stop(stream)*1e3/trials;
    cufftDestroy(fp);cudaFree(d_tau);cudaFree(d_wsk);cudaFree(d_isk);cudaFree(d_wch);cudaFree(d_ich);
    return ms;
}

static double time_fp16(cublasHandle_t blas,cusolverDnHandle_t solver,
                        thrust::device_vector<double>&d_A,
                        const thrust::device_vector<double>&d_Abak,
                        int m,int n,int trials)
{
    const int c=3*n,half_r=m/2+1;
    const float normf=(float)std::sqrt((double)c/m);
    const int BLK=256,GRD=(int)std::min(65535LL,((long long)m*n+BLK-1)/BLK);

    thrust::device_vector<int8_t>  d_D(m);
    thrust::device_vector<int>     d_S(c);
    thrust::device_vector<__half>  d_FA((long long)m*n);
    thrust::device_vector<__half2> d_fft((long long)n*(half_r));
    thrust::device_vector<float>   d_sk(c*n);
    thrust::device_vector<double>  d_RsTR(n*n,0.),d_G(n*n);

    int8_t*pD=thrust::raw_pointer_cast(d_D.data());
    int*pS=thrust::raw_pointer_cast(d_S.data());
    __half*pFA=thrust::raw_pointer_cast(d_FA.data());
    __half2*pFft=thrust::raw_pointer_cast(d_fft.data());
    float*pSk=thrust::raw_pointer_cast(d_sk.data());
    double*pRs=thrust::raw_pointer_cast(d_RsTR.data());
    double*pG=thrust::raw_pointer_cast(d_G.data());

    cufftHandle fp; CUFFT_CHECK(cufftCreate(&fp));
    long long n1[1]={m}; size_t ws=0;
    CUFFT_CHECK(cufftXtMakePlanMany(fp,1,n1,n1,1,m,CUDA_R_16F,
                                     n1,1,half_r,CUDA_C_16F,n,&ws,CUDA_R_16F));

    int lw_sk=0; float*d_tau=nullptr,*d_wsk=nullptr; int*d_isk=nullptr;
    CUDA_CHECK(cudaMalloc(&d_tau,n*sizeof(float)));
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver,c,n,pSk,c,&lw_sk));
    CUDA_CHECK(cudaMalloc(&d_wsk,lw_sk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_isk,sizeof(int)));

    int lw_ch=0; double*d_wch=nullptr; int*d_ich=nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_UPPER,n,pG,n,&lw_ch));
    CUDA_CHECK(cudaMalloc(&d_wch,lw_ch*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ich,sizeof(int)));

    auto one=[&](){
        double*pA=thrust::raw_pointer_cast(d_A.data());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(m),d_D.begin(),RandRadI8());
        thrust::transform(thrust::counting_iterator<unsigned>(0),
                          thrust::counting_iterator<unsigned>(c),d_S.begin(),RandSamp{m});
        diag_fp16<<<GRD,BLK>>>(pA,pD,pFA,m,n); CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftXtExec(fp,(void*)pFA,(void*)pFft,CUFFT_FORWARD));
        thrust::transform(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(c*n),
                          d_sk.begin(),SelF16{pFft,pS,c,m,half_r});
        thrust::transform(d_sk.begin(),d_sk.end(),d_sk.begin(),
                          [normf]__device__(float x){return x*normf;});
        CUSOLVER_CHECK(cusolverDnSgeqrf(solver,c,n,pSk,c,d_tau,d_wsk,lw_sk,d_isk));
        thrust::fill(d_RsTR.begin(),d_RsTR.end(),0.);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(c*n),TriF32toF64{c,n,pSk,pRs});
        thrust::copy(d_RsTR.begin(),d_RsTR.end(),d_G.begin());
        cholqr(blas,solver,pA,pG,d_wch,lw_ch,d_ich,m,n);
    };
    thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
    one(); CUDA_CHECK(cudaDeviceSynchronize());
    cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

CUBLAS_CHECK(cublasSetStream(blas, stream));
CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));
CUFFT_CHECK(cufftSetStream(fp, stream));
    EV wall; wall.start(stream);
    for(int t=0;t<trials;++t){thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());one();}
    double ms=wall.stop(stream)*1e3/trials;
    cufftDestroy(fp);cudaFree(d_tau);cudaFree(d_wsk);cudaFree(d_isk);cudaFree(d_wch);cudaFree(d_ich);
    return ms;
}

static double time_cholqr2(cublasHandle_t blas,cusolverDnHandle_t solver,
                            thrust::device_vector<double>&d_A,
                            const thrust::device_vector<double>&d_Abak,
                            int m,int n,int trials)
{
    const double a=1.,b=0.; cublasFillMode_t uplo=CUBLAS_FILL_MODE_UPPER;
    thrust::device_vector<double> d_G(n*n),d_G1(n*n);
    double*pG=thrust::raw_pointer_cast(d_G.data());
    double*pG1=thrust::raw_pointer_cast(d_G1.data());

    int lw=0; double*d_work=nullptr; int*d_info=nullptr;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver,uplo,n,pG,n,&lw));
    CUDA_CHECK(cudaMalloc(&d_work,lw*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info,sizeof(int)));

    auto one=[&](){
        double*pA=thrust::raw_pointer_cast(d_A.data());
        CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,&a,pA,m,pA,m,&b,pG,n));
        CUSOLVER_CHECK(cusolverDnDpotrf(solver,uplo,n,pG,n,d_work,lw,d_info));
        CUBLAS_CHECK(cublasDtrsm(blas,CUBLAS_SIDE_RIGHT,uplo,CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT,m,n,&a,pG,n,pA,m));
        CUBLAS_CHECK(cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,m,&a,pA,m,pA,m,&b,pG1,n));
        CUSOLVER_CHECK(cusolverDnDpotrf(solver,uplo,n,pG1,n,d_work,lw,d_info));
        CUBLAS_CHECK(cublasDtrsm(blas,CUBLAS_SIDE_RIGHT,uplo,CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT,m,n,&a,pG1,n,pA,m));
    };
    thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());
    one(); CUDA_CHECK(cudaDeviceSynchronize());
    cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

CUBLAS_CHECK(cublasSetStream(blas, stream));
CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));

    EV wall; wall.start(stream);
    for(int t=0;t<trials;++t){thrust::copy(d_Abak.begin(),d_Abak.begin()+(long long)m*n,d_A.begin());one();}
    double ms=wall.stop(stream)*1e3/trials;
    cudaFree(d_work); cudaFree(d_info);
    return ms;
}

// ===========================================================================
int main()
{
    const int n      = 100;
    const int TRIALS = 5;
    const int m_min_exp = 17, m_max_exp = 24;   // m = 2^17 to 2^24
    const int NM = m_max_exp - m_min_exp + 1;

    cublasHandle_t    blas;   CUBLAS_CHECK(cublasCreate(&blas));
    cusolverDnHandle_t solver; CUSOLVER_CHECK(cusolverDnCreate(&solver));

    const long long m_max = 1LL << m_max_exp;

    printf("================================================================\n");
    printf("  Scaling Experiment: n=%d, m = 2^%d to 2^%d, %d trials\n",
           n, m_min_exp, m_max_exp, TRIALS);
    printf("  Allocating d_A and d_Abak at m_max=%lld ... ", m_max);
    fflush(stdout);

    thrust::device_vector<double> d_A(m_max * n), d_Abak(m_max * n);
    thrust::transform(thrust::counting_iterator<unsigned>(0),
                      thrust::counting_iterator<unsigned>((unsigned)(m_max*n)),
                      d_Abak.begin(), RandUnif());
    printf("done\n");
    printf("================================================================\n\n");

    std::vector<long long> m_vals(NM);
    std::vector<double> t_f64(NM), t_f32(NM), t_f16(NM), t_cqr2(NM);

    printf("%-10s %10s %10s %10s %10s\n","m","FP64(ms)","FP32(ms)","FP16(ms)","CQR2(ms)");
    printf("%s\n", std::string(52,'-').c_str());

    for(int e = m_min_exp; e <= m_max_exp; ++e){
        long long m = 1LL << e;
        int idx = e - m_min_exp;
        m_vals[idx] = m;

        t_f64[idx]  = time_fp64  (blas,solver,d_A,d_Abak,(int)m,n,TRIALS);
        t_f32[idx]  = time_fp32  (blas,solver,d_A,d_Abak,(int)m,n,TRIALS);
        t_f16[idx]  = time_fp16  (blas,solver,d_A,d_Abak,(int)m,n,TRIALS);
        t_cqr2[idx] = time_cholqr2(blas,solver,d_A,d_Abak,(int)m,n,TRIALS);

        printf("2^%-8d %10.3f %10.3f %10.3f %10.3f\n",
               e, t_f64[idx], t_f32[idx], t_f16[idx], t_cqr2[idx]);
        fflush(stdout);
    }

    cublasDestroy(blas); cusolverDnDestroy(solver);

    // MATLAB output
    printf("\n%% === MATLAB DATA (n=%d, m=2^%d:2^%d, trials=%d) ===\n\n",
           n, m_min_exp, m_max_exp, TRIALS);
    printf("m_vals = [");
    for(int i=0;i<NM;i++){if(i)printf(", ");printf("2^%d",m_min_exp+i);}
    printf("];\n\n");

    auto print=[&](const char*name,const std::vector<double>&v){
        printf("%s = [",name);
        for(int i=0;i<NM;i++){if(i)printf(", ");printf("%.4f",v[i]);}
        printf("];\n");
    };
    print("fp64_ms",  t_f64);
    print("fp32_ms",  t_f32);
    print("fp16_ms",  t_f16);
    print("cqr2_ms",  t_cqr2);

    return 0;
}
