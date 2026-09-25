// common.cuh
// Shared utilities: error-checking macros and GPU event timer.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cufft.h>

#define CUDA_CHECK(x) \
    do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
        fprintf(stderr,"CUDA %s:%d  %s\n",__FILE__,__LINE__, \
                cudaGetErrorString(_e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) \
    do { cublasStatus_t _s=(x); if(_s!=CUBLAS_STATUS_SUCCESS){ \
        fprintf(stderr,"cuBLAS %s:%d  code=%d\n",__FILE__,__LINE__,(int)_s); \
        exit(1); } } while(0)
#define CUSOLVER_CHECK(x) \
    do { cusolverStatus_t _s=(x); if(_s!=CUSOLVER_STATUS_SUCCESS){ \
        fprintf(stderr,"cuSolver %s:%d  code=%d\n",__FILE__,__LINE__,(int)_s); \
        exit(1); } } while(0)
#define CUFFT_CHECK(x) \
    do { cufftResult _r=(x); if(_r!=CUFFT_SUCCESS){ \
        fprintf(stderr,"cuFFT %s:%d  code=%d\n",__FILE__,__LINE__,(int)_r); \
        exit(1); } } while(0)

// GPU event-based stopwatch (returns seconds).
struct EV {
    cudaEvent_t s, e;
    EV()  { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~EV() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void   start() { cudaEventRecord(s); }
    double stop()  {
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e);
        return (double)ms * 1e-3;
    }
};
