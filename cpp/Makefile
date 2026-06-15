# CUDA-Q MIPT trajectory generation + separate MOSEK GMN analysis build.
#
# Default build:
#   make
# produces:
#   ./mipt.exe  - simulates circuits and saves complex three-qubit RDM datasets
#   ./gmn.exe   - reads rho3 datasets and computes GMN for every stored subsystem
#   ./ring_inflation.exe - selects max-GMN subsystem per record and solves level-2 ring inflation with SCS
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
#   ./ring_inflation.exe rho3.bin 2000 1e-3 300 [limit_per_p] [p_min p_max [record_start record_stop]]

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

# ---- SCS C library configuration for ring_inflation.exe ----
# SCS source/CMake installs commonly provide versioned direct/indirect solver
# libraries such as libscsdir.so.3.2.11 without an unversioned libscsdir.so
# symlink.  Linking the discovered file path directly is therefore more robust
# than using -lscs or -lscsdir.
#
# Override examples:
#   make all SCS_HOME=$HOME/opt/scs
#   make all SCS_LIBFILE=$HOME/mipt/scs/build/lib/libscsdir.so.3.2.11
#   make all SCS_CFLAGS="-I$HOME/scs/include" SCS_LDFLAGS="$HOME/scs/out/libscsdir.a"
SCS_HOME ?= $(HOME)/opt/scs
SCS_INC  ?= $(SCS_HOME)/include
SCS_LIB  ?= $(SCS_HOME)/lib

SCS_CFLAGS ?= -I$(SCS_INC) -I$(SCS_INC)/scs

SCS_LIBFILE ?= $(firstword   $(wildcard $(SCS_LIB)/libscsdir.so)   $(wildcard $(SCS_LIB)/libscsdir.so.*)   $(wildcard $(SCS_LIB)/libscsdir.a)   $(wildcard $(SCS_LIB)/libscsindir.so)   $(wildcard $(SCS_LIB)/libscsindir.so.*)   $(wildcard $(SCS_LIB)/libscsindir.a))

ifeq ($(strip $(SCS_LDFLAGS)),)
  ifneq ($(strip $(SCS_LIBFILE)),)
    SCS_LDFLAGS := $(SCS_LIBFILE) -Wl,-rpath,$(SCS_LIB)
  else
    # Last-resort fallback. If this fails, run:
    #   find "$HOME" -type f \( -name 'libscsdir.*' -o -name 'libscsindir.*' \) -print 2>/dev/null
    # and pass SCS_LIBFILE=/path/to/libscsdir.so.VERSION.
    SCS_LDFLAGS := -L$(SCS_LIB) -Wl,-rpath,$(SCS_LIB) -lscsdir
  endif
endif

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
RING_OBJDIR := build/ring_scs_omp$(OPENMP)
MIPT_BINARY := $(MIPT_OBJDIR)/mipt.exe
GMN_BINARY  := $(GMN_OBJDIR)/gmn.exe
RING_BINARY := $(RING_OBJDIR)/ring_inflation.exe

MIPT_OBJS := $(MIPT_OBJDIR)/mipt.o
ifeq ($(CUDA_RHO),1)
  MIPT_OBJS += $(MIPT_OBJDIR)/rho_reduce.o
endif

GMN_OBJS := $(GMN_OBJDIR)/gmn_main.o $(GMN_OBJDIR)/gmn_mosek_c.o
RING_OBJS := $(RING_OBJDIR)/ring_inflation.o $(RING_OBJDIR)/gmn_mosek_c.o

.PHONY: all clean cpu gpu print-config mipt.exe gmn.exe ring_inflation.exe

all: mipt.exe gmn.exe ring_inflation.exe

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
	@echo "RING_OBJDIR=$(RING_OBJDIR)"
	@echo "MIPT_BINARY=$(MIPT_BINARY)"
	@echo "GMN_BINARY=$(GMN_BINARY)"
	@echo "RING_BINARY=$(RING_BINARY)"
	@echo "SCS_CFLAGS=$(SCS_CFLAGS)"
	@echo "SCS_LIBFILE=$(SCS_LIBFILE)"
	@echo "SCS_LDFLAGS=$(SCS_LDFLAGS)"
	@echo "OPENMP=$(OPENMP)"

mipt.exe: $(MIPT_BINARY)
	cp $(MIPT_BINARY) $@

gmn.exe: $(GMN_BINARY)
	cp $(GMN_BINARY) $@

ring_inflation.exe: $(RING_BINARY)
	cp $(RING_BINARY) $@

$(MIPT_BINARY): $(MIPT_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) -o $@

$(GMN_BINARY): $(GMN_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) -o $@

$(RING_BINARY): $(RING_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) $(SCS_LDFLAGS) -o $@

$(MIPT_OBJDIR):
	mkdir -p $@

$(GMN_OBJDIR):
	mkdir -p $@

$(RING_OBJDIR):
	mkdir -p $@

$(MIPT_OBJDIR)/mipt.o: mipt.cpp density_io.hpp rho_reduce.hpp | $(MIPT_OBJDIR)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@

$(MIPT_OBJDIR)/rho_reduce.o: rho_reduce.cu rho_reduce.hpp | $(MIPT_OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GMN_OBJDIR)/gmn_main.o: gmn_main.cpp density_io.hpp gmn.hpp | $(GMN_OBJDIR)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@

$(GMN_OBJDIR)/gmn_mosek_c.o: gmn_mosek_c.c gmn.hpp | $(GMN_OBJDIR)
	$(HOSTCC) $(CFLAGS_COMMON) -c $< -o $@

$(RING_OBJDIR)/ring_inflation.o: ring_inflation.cpp density_io.hpp gmn.hpp | $(RING_OBJDIR)
	$(HOSTCXX) $(CXXFLAGS_COMMON) $(SCS_CFLAGS) -c $< -o $@

$(RING_OBJDIR)/gmn_mosek_c.o: gmn_mosek_c.c gmn.hpp | $(RING_OBJDIR)
	$(HOSTCC) $(CFLAGS_COMMON) -c $< -o $@

clean:
	rm -rf build mipt.exe gmn.exe ring_inflation.exe *.tmp
