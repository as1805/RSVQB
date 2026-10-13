HOSTNAME := $(shell hostname)

comma := ,

# ---------------------------------------------------------------------
# Site-specific setup: CUDA_PATH / library dirs still depend on which
# cluster you're on, because that's about module systems and install
# locations, not GPU architecture.
# ---------------------------------------------------------------------
ifeq ($(HOSTNAME),mimir)

    CUDA_PATH   := /opt/apps/sysnet/cuda/cuda-12.8
    CUDA_LIBDIR := $(CUDA_PATH)/lib64
    MATH_LIBDIR :=
    MATH_INCDIR :=

else ifneq ($(findstring stampede3.,$(HOSTNAME)),)

ifndef TACC_CUDA_DIR
$(error TACC_CUDA_DIR is not set — run 'module load cuda' before make)
endif

ifndef TACC_NVIDIA_MATH_DIR
$(error TACC_NVIDIA_MATH_DIR is not set — run 'module load cuda' before make)
endif

    CUDA_PATH   := $(TACC_CUDA_DIR)
    CUDA_LIBDIR := $(TACC_CUDA_LIB)
    MATH_LIBDIR := $(TACC_NVIDIA_MATH_DIR)/lib64
    MATH_INCDIR := $(TACC_NVIDIA_MATH_DIR)/include

else ifeq ($(HOSTNAME),gpu18)

    CUDA_PATH   := /usr/local/apps/cuda/cuda-12.0
    CUDA_LIBDIR := $(CUDA_PATH)/lib64
    MATH_LIBDIR := $(CUDA_PATH)/lib64
    MATH_INCDIR := $(CUDA_PATH)/include

else
    # Unknown node: don't hard-fail, try to find a CUDA install on PATH.
    NVCC_ON_PATH := $(shell command -v nvcc 2>/dev/null)
    ifeq ($(strip $(NVCC_ON_PATH)),)
$(error Unrecognised machine '$(HOSTNAME)' and no nvcc found on PATH. \
Please 'module load cuda' or set CUDA_PATH/CUDA_LIBDIR/MATH_LIBDIR/MATH_INCDIR manually)
    endif
    CUDA_PATH   := $(shell dirname $(shell dirname $(NVCC_ON_PATH)))
    CUDA_LIBDIR := $(CUDA_PATH)/lib64
    MATH_LIBDIR :=
    MATH_INCDIR :=
endif

# ---------------------------------------------------------------------
# GPU architecture: auto-detect instead of hardcoding per hostname.
#
# If a GPU is visible right now (e.g. you're compiling inside a job
# that already has a GPU allocated), build only for that architecture
# — fastest compile. Otherwise (e.g. compiling on a login node), fall
# back to a multi-arch "fat" binary covering Volta through Hopper plus
# PTX, so the resulting binary runs correctly wherever it lands.
#
# Override any time with: make GENCODE="-gencode=arch=compute_80,code=sm_80"
# ---------------------------------------------------------------------
NVIDIA_SMI := $(shell command -v nvidia-smi 2>/dev/null)

ifneq ($(strip $(NVIDIA_SMI)),)
    GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.[:space:]')
endif

ifneq ($(strip $(GPU_CC)),)
    GENCODE ?= -gencode=arch=compute_$(GPU_CC),code=sm_$(GPU_CC)
else
    GENCODE ?= -gencode=arch=compute_70,code=sm_70 \
               -gencode=arch=compute_75,code=sm_75 \
               -gencode=arch=compute_80,code=sm_80 \
               -gencode=arch=compute_86,code=sm_86 \
               -gencode=arch=compute_89,code=sm_89 \
               -gencode=arch=compute_90,code=sm_90 \
               -gencode=arch=compute_90,code=compute_90
endif

NVCC      := $(CUDA_PATH)/bin/nvcc
CXX       := g++

NVCCFLAGS := $(GENCODE) -std=c++17 -O3 --extended-lambda -Xcompiler -fPIC \
$(if $(MATH_INCDIR),-I$(MATH_INCDIR),)

CXXFLAGS  := -O3 -std=c++17

LDFLAGS   := -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) \
$(if $(MATH_LIBDIR),-L$(MATH_LIBDIR) -Wl$(comma)-rpath$(comma)$(MATH_LIBDIR),) \
             -Wl,--disable-new-dtags \
             -lcublas -lcublasLt -lcudart -lcusolver -lcufft

LIBS      := mrcqr.o cholqr2.o svqb2.o

TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# -----------------------------------------------------------------------
all: driver_accuracy driver_perf driver_geqrf driver_profile driver_scaling

# Library objects
mrcqr.o: mrcqr.cu mrcqr.hpp common.cuh
	$(NVCC) -c $< $(NVCCFLAGS) -o $@

cholqr2.o: cholqr2.cu cholqr2.hpp common.cuh
	$(NVCC) -c $< $(NVCCFLAGS) -o $@

svqb2.o: svqb2.cu svqb2.hpp common.cuh
	$(NVCC) -c $< $(NVCCFLAGS) -o $@

# Accuracy benchmark
driver_accuracy.o: driver_accuracy.cu mrcqr.hpp cholqr2.hpp svqb2.hpp common.cuh
	$(NVCC) -c $< $(NVCCFLAGS) -o $@

driver_accuracy: driver_accuracy.o $(LIBS)
	$(CXX) $^ -o $@ $(LDFLAGS)

# Performance benchmark
driver_perf.o: driver_perf.cpp mrcqr.hpp cholqr2.hpp svqb2.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

driver_perf: driver_perf.o $(LIBS)
	$(CXX) $^ -o $@ $(LDFLAGS)

# geqrf benchmark
driver_geqrf.o: driver_geqrf.cu common.cuh
	$(NVCC) -c $< $(NVCCFLAGS) -o $@

driver_geqrf: driver_geqrf.o
	$(CXX) $^ -o $@ $(LDFLAGS)

# Sub-operation profiler
driver_profile.o: driver_profile.cu common.cuh
	$(NVCC) -c $< $(NVCCFLAGS) -o $@

driver_profile: driver_profile.o
	$(CXX) $^ -o $@ $(LDFLAGS)

# Run targets
run_accuracy: driver_accuracy
	@mkdir -p results
	./driver_accuracy 2>&1 | tee results/accuracy_$(TIMESTAMP).txt

run_perf: driver_perf
	@mkdir -p results
	./driver_perf 2>&1 | tee results/perf_$(TIMESTAMP).txt

# Scaling experiment
driver_scaling.o: driver_scaling.cu common.cuh
	$(NVCC) -c $< $(NVCCFLAGS) -o $@

driver_scaling: driver_scaling.o
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm -f *.o driver_accuracy driver_perf driver_geqrf driver_profile driver_scaling