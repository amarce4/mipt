# CUDA-Q + CUDA reduction + MOSEK Optimizer C API build.
#
# The MOSEK Fusion C++ implementation is deliberately not linked: CUDA-Q's
# nvq++ toolchain on this installation links libc++, while Linux Fusion is
# built through GCC/libstdc++.  The C Optimizer API avoids that ABI conflict.
#
# Usage:
#   make mipt.exe GPU=1 FP64=1 MOSEK_HOME=$HOME/mosek/11.2/tools/platform/linux64x86
#   make mipt.exe GPU=1 FP64=0 MOSEK_HOME=$HOME/mosek/11.2/tools/platform/linux64x86
#   make mipt.exe GPU=0          MOSEK_HOME=$HOME/mosek/11.2/tools/platform/linux64x86

NVQXX ?= nvq++
HOSTCC ?= gcc
NVCC   ?= nvcc

EIGEN_INC  ?= /usr/include/eigen3
MOSEK_HOME ?= $(HOME)/mosek/11.2/tools/platform/linux64x86
MOSEK_INC  ?= $(MOSEK_HOME)/h
MOSEK_LIB  ?= $(MOSEK_HOME)/bin

GPU ?= 0
FP64 ?= 1
CUDA_RHO ?= $(GPU)

NVQXXFLAGS_COMMON := -std=c++20 -O3 -DNDEBUG -march=native -I$(EIGEN_INC) -I$(MOSEK_INC)
CFLAGS_COMMON     := -std=c11 -O3 -DNDEBUG -march=native -I$(MOSEK_INC)
NVCCFLAGS         := -std=c++17 -O3
LDFLAGS_COMMON    := -L$(MOSEK_LIB) -Wl,-rpath,$(MOSEK_LIB)
# Use only the ABI-stable Optimizer C API library, not -lfusion64.
LDLIBS_COMMON     := -lmosek64

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
  LDLIBS_COMMON += -lcudart
endif

CONFIG := gpu$(GPU)_fp64$(FP64)_rho$(CUDA_RHO)_capi
OBJDIR := build/$(CONFIG)
BINARY := $(OBJDIR)/mipt.exe
OBJS   := $(OBJDIR)/mipt.o $(OBJDIR)/gmn_mosek_c.o
ifeq ($(CUDA_RHO),1)
  OBJS += $(OBJDIR)/rho_reduce.o
endif

.PHONY: all clean gpu cpu print-config mipt.exe

all: mipt.exe

cpu:
	$(MAKE) mipt.exe GPU=0 CUDA_RHO=0

gpu:
	$(MAKE) mipt.exe GPU=1 CUDA_RHO=1 FP64=$(FP64)

print-config:
	@echo "CONFIG=$(CONFIG)"
	@echo "NVQXX=$(NVQXX)"
	@echo "CUDAQ_TARGET_FLAGS=$(CUDAQ_TARGET_FLAGS)"
	@echo "OBJDIR=$(OBJDIR)"
	@echo "BINARY=$(BINARY)"

mipt.exe: $(BINARY)
	cp $(BINARY) $@

$(BINARY): $(OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(LDFLAGS_COMMON) $(LDLIBS_COMMON) -o $@

$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/mipt.o: mipt.cpp gmn.hpp rho_reduce.hpp | $(OBJDIR)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@

$(OBJDIR)/gmn_mosek_c.o: gmn_mosek_c.c gmn.hpp | $(OBJDIR)
	$(HOSTCC) $(CFLAGS_COMMON) -c $< -o $@

$(OBJDIR)/rho_reduce.o: rho_reduce.cu rho_reduce.hpp | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf build mipt.exe *.tmp
