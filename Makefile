CUDA_BASE   = /usr/local/cuda
BOOST_BASE  = /afs/hep.wisc.edu/home/abocci/public/boost
ALPAKA_BASE = /afs/hep.wisc.edu/home/abocci/public/alpaka

CXX  := g++
NVCC := $(CUDA_BASE)/bin/nvcc

CXXFLAGS       := -std=c++17 -O2 -g -I$(BOOST_BASE)/include -I$(ALPAKA_BASE)/include
CXX_HOST_FLAGS := -pthread
CXX_CUDA_FLAGS := --expt-relaxed-constexpr

all:
	g++ -std=c++17 project_cpu.cc -o project_cpu
	nvcc project_cuda.cu -o project_cuda
	nvcc project_cuda_managed.cu -o project_cuda_managed
	nvcc project_cuda_shared.cu -o project_cuda_shared
	$(NVCC) -x cu -ccbin $(CXX) $(CXXFLAGS) $(CXX_CUDA_FLAGS) -Xcompiler '$(CXX_HOST_FLAGS)' project_cpu_alpaka.cc -DALPAKA_ACC_GPU_CUDA_ENABLED -o project_cpu_alpaka