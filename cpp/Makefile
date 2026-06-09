# CUDA-Q MIPT trajectory generation + separate MOSEK GMN analysis build.
#
# Default build:
#   make
# produces:
#   ./mipt.exe  - simulates circuits and saves complex three-qubit RDM datasets
#   ./gmn.exe   - reads rho3 datasets and computes GMN for every stored subsystem
#
# CPU simulation:
#   make GPU=0 CUDA_RHO=0
#
# NVIDIA simulation with GPU reduced-density-matrix kernels:
#   make GPU=1 CUDA_RHO=1 FP64=1
#   make GPU=1 CUDA_RHO=1 FP64=0
#
# GMN analysis:
#   ./gmn.exe 0 rho3.bin data_real.csv       # real-only GMN on Re(rho)
#   ./gmn.exe 1 rho3.bin data_complex.csv    # full complex Hermitian GMN SDP

NVQXX   ?= nvq++
HOSTCXX ?= g++
HOSTCC  ?= gcc
NVCC    ?= nvcc

MOSEK_HOME ?= $(HOME)/mosek/11.2/tools/platform/linux64x86
MOSEK_INC  ?= $(MOSEK_HOME)/h
MOSEK_LIB  ?= $(MOSEK_HOME)/bin

GPU ?= 0
FP64 ?= 1
CUDA_RHO ?= $(GPU)
OPENMP ?= 1

NVQXXFLAGS_COMMON := -std=c++20 -O3 -DNDEBUG -march=native
CXXFLAGS_COMMON   := -std=c++20 -O3 -DNDEBUG -march=native -I$(MOSEK_INC)
CFLAGS_COMMON     := -std=c11 -O3 -DNDEBUG -march=native -I$(MOSEK_INC)
NVCCFLAGS         := -std=c++17 -O3
MOSEK_LDFLAGS     := -L$(MOSEK_LIB) -Wl,-rpath,$(MOSEK_LIB)
MOSEK_LIBS        := -lmosek64

ifeq ($(OPENMP),1)
  CXXFLAGS_COMMON += -fopenmp
  CFLAGS_COMMON   += -fopenmp
  HOST_LDFLAGS    += -fopenmp
endif

ifeq ($(GPU),1)
  CUDAQ_TARGET_FLAGS := --target nvidia
  ifeq ($(FP64),1)
    CUDAQ_TARGET_FLAGS += --target-option fp64
  else
    CUDAQ_TARGET_FLAGS += --target-option fp32
  endif
endif

ifeq ($(CUDA_RHO),1)
  NVQXXFLAGS_COMMON += -DMIPT_ENABLE_CUDA_RHO
  MIPT_CUDA_LIBS := -lcudart
endif

CONFIG := gpu$(GPU)_fp64$(FP64)_rho$(CUDA_RHO)
MIPT_OBJDIR := build/mipt_$(CONFIG)
GMN_OBJDIR  := build/gmn_capi_omp$(OPENMP)
MIPT_BINARY := $(MIPT_OBJDIR)/mipt.exe
GMN_BINARY  := $(GMN_OBJDIR)/gmn.exe

MIPT_OBJS := $(MIPT_OBJDIR)/mipt.o
ifeq ($(CUDA_RHO),1)
  MIPT_OBJS += $(MIPT_OBJDIR)/rho_reduce.o
endif

GMN_OBJS := $(GMN_OBJDIR)/gmn_main.o $(GMN_OBJDIR)/gmn_mosek_c.o

.PHONY: all clean cpu gpu print-config mipt.exe gmn.exe

all: mipt.exe gmn.exe

cpu:
	$(MAKE) all GPU=0 CUDA_RHO=0

gpu:
	$(MAKE) all GPU=1 CUDA_RHO=1 FP64=$(FP64)

print-config:
	@echo "CONFIG=$(CONFIG)"
	@echo "NVQXX=$(NVQXX)"
	@echo "CUDAQ_TARGET_FLAGS=$(CUDAQ_TARGET_FLAGS)"
	@echo "MIPT_OBJDIR=$(MIPT_OBJDIR)"
	@echo "GMN_OBJDIR=$(GMN_OBJDIR)"
	@echo "MIPT_BINARY=$(MIPT_BINARY)"
	@echo "GMN_BINARY=$(GMN_BINARY)"
	@echo "OPENMP=$(OPENMP)"

mipt.exe: $(MIPT_BINARY)
	cp $(MIPT_BINARY) $@

gmn.exe: $(GMN_BINARY)
	cp $(GMN_BINARY) $@

$(MIPT_BINARY): $(MIPT_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) -o $@

$(GMN_BINARY): $(GMN_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) -o $@

$(MIPT_OBJDIR):
	mkdir -p $@

$(GMN_OBJDIR):
	mkdir -p $@

$(MIPT_OBJDIR)/mipt.o: mipt.cpp density_io.hpp rho_reduce.hpp | $(MIPT_OBJDIR)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@

$(MIPT_OBJDIR)/rho_reduce.o: rho_reduce.cu rho_reduce.hpp | $(MIPT_OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GMN_OBJDIR)/gmn_main.o: gmn_main.cpp density_io.hpp gmn.hpp | $(GMN_OBJDIR)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@

$(GMN_OBJDIR)/gmn_mosek_c.o: gmn_mosek_c.c gmn.hpp | $(GMN_OBJDIR)
	$(HOSTCC) $(CFLAGS_COMMON) -c $< -o $@

clean:
	rm -rf build mipt.exe gmn.exe *.tmp
