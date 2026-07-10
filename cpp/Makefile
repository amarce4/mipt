# CUDA-Q MIPT trajectory generation + separate MOSEK GMN analysis build.
#
# Default build:
#   make
# produces:
#   ./mipt.exe  - simulates MMS/Haar/fermionic/RFGS circuits and saves complex three-qubit RDM datasets
#   ./gmn.exe   - reads rho3 datasets and computes GMN for every stored subsystem
#   ./fgmn.exe  - reads rho3 datasets and computes fermionic GMN for every stored subsystem
#   ./ring_inflation.exe - selects max-GMN subsystem per record and solves level-2 ring inflation with SCS
#   ./tmi.exe  - reads rho_tmi.bin and computes TMI to p,tmi CSV
#   ./sim_tmi.exe - simulates circuits and computes TMI online to p,tmi CSV
#                 accepts fermion=0 (MMS/qubit), 1 (Haar fermion), or 2 (FRGS fermion)
#                 accepts all_cycles=0/1 to emit one or N/4 cyclic TMI values per circuit
#   ./entropy.exe - simulates circuits and emits translation-averaged second-Renyi entropy scaling data
#
# sim_tmi.exe dynamically loads LAPACK/BLAS symbols at runtime.  Installing
# OpenBLAS/LAPACK enables the fast exact Schmidt-SVD path; otherwise it falls
# back to the older reduced-density-matrix path.
#
# Default build follows the GPU/CUDA_RHO defaults below.  Override GPU=0
# CUDA_RHO=0 for CPU-side CUDA-Q simulation when that is faster for small jobs.
#
# Examples:
#   make mipt.exe GPU=0 CUDA_RHO=0
#   make mipt.exe GPU=1 CUDA_RHO=1 FP64=0
#   make sim_tmi.exe GPU=1 CUDA_RHO=1 FP64=0
#   make sim_tmi.exe GPU=1 CUDA_RHO=1 FP64=1
#
# CUDA-Q tensor-network / MPS targets are opt-in.  When a tensor-network target
# does not expose a dense rank-1 state tensor, mipt.exe falls back to backend-
# native amplitude queries for 3-site RDM extraction.  For N above
# MIPT_MPS_EXACT_OBSERVABLE_MAX_QUBITS this uses an approximate uniform-
# environment Monte Carlo estimator controlled by MIPT_MPS_RDM_MC_SAMPLES.
# sim_tmi.exe can use the same amplitude-query fallback by reconstructing a
# dense vector for small N, controlled by SIM_TMI_MPS_DENSE_MAX_QUBITS.
#
# CUDA .cu compilation uses nvcc when it is available, otherwise falls back
# to nvq++. This keeps CUDA-Q-only installs usable on machines where nvcc is
# not installed or not on PATH. Override with, e.g.:
#   make sim_tmi.exe CUDA_CU_COMPILER=/usr/local/cuda/bin/nvcc
#
# GMN analysis:
#   ./gmn.exe 0 rho3.bin data_real.csv       # real-only GMN on Re(rho)
#   ./gmn.exe 1 rho3.bin data_complex.csv    # full complex Hermitian GMN SDP
#   ./fgmn.exe rho3.bin data_fgmn.csv         # fermionic GMN SDP
#   ./ring_inflation.exe rho3.bin 2000 1e-3 300 [limit_per_p] [p_min p_max [record_start record_stop]]

NVQXX   ?= nvq++
HOSTCXX ?= g++
HOSTCC  ?= gcc
NVCC    ?= nvcc

MOSEK_HOME ?= $(HOME)/mosek/11.2/tools/platform/linux64x86
MOSEK_INC  ?= $(MOSEK_HOME)/h
MOSEK_LIB  ?= $(MOSEK_HOME)/bin

GPU ?= 1
FP64 ?= 0
CUDAQ_BACKEND ?= $(if $(MIPT_TENSOR_BACKEND),$(MIPT_TENSOR_BACKEND),nvidia)
CUDA_RHO ?= $(if $(filter nvidia,$(CUDAQ_BACKEND)),$(GPU),0)
OPENMP ?= 1
NVQXXFLAGS_COMMON := -std=c++20 -O3 -DNDEBUG -march=native -DMIPT_CUDAQ_BACKEND_NAME=\"$(CUDAQ_BACKEND)\"
CXXFLAGS_COMMON   := -std=c++20 -O3 -DNDEBUG -march=native -I$(MOSEK_INC)
CFLAGS_COMMON     := -std=c11 -O3 -DNDEBUG -march=native -I$(MOSEK_INC)
NVCCFLAGS         := -std=c++14 -O3
MOSEK_LDFLAGS     := -L$(MOSEK_LIB) -Wl,-rpath,$(MOSEK_LIB)
MOSEK_LIBS        := -lmosek64 -lfusion64
DL_LIBS           ?= -ldl

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
  ifeq ($(CUDAQ_BACKEND),nvidia)
    CUDAQ_TARGET_FLAGS := --target nvidia
    NVQXXFLAGS_COMMON += -DSIM_TMI_GPU_BUILD=1
    ifeq ($(FP64),1)
      CUDAQ_TARGET_FLAGS += --target-option fp64
    else
      CUDAQ_TARGET_FLAGS += --target-option fp32
    endif
  else ifeq ($(CUDAQ_BACKEND),tensornet)
    CUDAQ_TARGET_FLAGS := --target tensornet
    NVQXXFLAGS_COMMON += -DSIM_TMI_GPU_BUILD=1 -DMIPT_TENSOR_NETWORK_BUILD=1
  else ifeq ($(CUDAQ_BACKEND),tensornet-mps)
    CUDAQ_TARGET_FLAGS := --target tensornet-mps
    NVQXXFLAGS_COMMON += -DSIM_TMI_GPU_BUILD=1 -DMIPT_TENSOR_NETWORK_BUILD=1 -DMIPT_APPROX_MPS_BUILD=1
  else
    $(error Unsupported CUDAQ_BACKEND=$(CUDAQ_BACKEND). Use nvidia, tensornet, or tensornet-mps.)
  endif
endif

ifeq ($(CUDA_RHO),1)
  NVQXXFLAGS_COMMON += -DMIPT_ENABLE_CUDA_RHO
  MIPT_CUDA_LIBS := -lcudart
  SIM_TMI_CUDA_LIBS := -lcudart -lcusolver -lcublas
endif

# ---- CUDA .cu compiler selection ----
# Some CUDA-Q installs provide nvq++ but not nvcc.  Use nvcc when present;
# otherwise compile the CUDA helper translation units with nvq++ so GPU builds
# still work on systems such as CUDA-Q-only RTX 4080 Super setups.
NVCC_FOUND := $(shell command -v $(NVCC) 2>/dev/null)
ifeq ($(strip $(NVCC_FOUND)),)
  CUDA_CU_COMPILER ?= $(NVQXX)
  CUDA_CU_FLAGS    ?= $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON)
  CUDA_CU_BACKEND  := nvq++ fallback (nvcc not found)
else
  CUDA_CU_COMPILER ?= $(NVCC)
  CUDA_CU_FLAGS    ?= $(NVCCFLAGS)
  CUDA_CU_BACKEND  := nvcc
endif

CONFIG := gpu$(GPU)_fp64$(FP64)_rho$(CUDA_RHO)
MIPT_OBJDIR := build/mipt_$(CONFIG)
GMN_OBJDIR  := build/gmn_capi_omp$(OPENMP)
FGMN_OBJDIR := build/fgmn_fusion_omp$(OPENMP)
RING_OBJDIR := build/ring_scs_omp$(OPENMP)
TMI_OBJDIR  := build/tmi_omp$(OPENMP)
MIPT_BINARY := $(MIPT_OBJDIR)/mipt.exe
SIM_TMI_BINARY := $(MIPT_OBJDIR)/sim_tmi.exe
ENTROPY_BINARY := $(MIPT_OBJDIR)/entropy.exe
GMN_BINARY  := $(GMN_OBJDIR)/gmn.exe
FGMN_BINARY := $(FGMN_OBJDIR)/fgmn.exe
RING_BINARY := $(RING_OBJDIR)/ring_inflation.exe
TMI_BINARY  := $(TMI_OBJDIR)/tmi.exe

MIPT_OBJS := $(MIPT_OBJDIR)/mipt.o
SIM_TMI_OBJS := $(MIPT_OBJDIR)/sim_tmi.o
ENTROPY_OBJS := $(MIPT_OBJDIR)/entropy.o
ifeq ($(CUDA_RHO),1)
  MIPT_OBJS += $(MIPT_OBJDIR)/rho_reduce.o
  SIM_TMI_OBJS += $(MIPT_OBJDIR)/rho_reduce.o $(MIPT_OBJDIR)/tmi_cuda_svd.o $(MIPT_OBJDIR)/tmi_cuda_rdm.o
  ENTROPY_OBJS += $(MIPT_OBJDIR)/rho_reduce.o $(MIPT_OBJDIR)/entropy_cuda.o
  ENTROPY_CUDA_LIBS := -lcublas
endif

GMN_OBJS := $(GMN_OBJDIR)/gmn_main.o $(GMN_OBJDIR)/gmn_mosek_c.o
FGMN_OBJS := $(FGMN_OBJDIR)/fgmn_main.o $(FGMN_OBJDIR)/fgmn.o
RING_OBJS := $(RING_OBJDIR)/ring_inflation.o $(RING_OBJDIR)/gmn_mosek_c.o
TMI_OBJS := $(TMI_OBJDIR)/tmi.o

.PHONY: all clean cpu gpu print-config mipt.exe sim_tmi.exe entropy.exe gmn.exe fgmn.exe ring_inflation.exe tmi.exe

all: mipt.exe sim_tmi.exe entropy.exe gmn.exe fgmn.exe ring_inflation.exe tmi.exe

cpu:
	$(MAKE) all GPU=0 CUDA_RHO=0

gpu:
	$(MAKE) all GPU=1 CUDA_RHO=1 FP64=$(FP64)

print-config:
	@echo "CONFIG=$(CONFIG)"
	@echo "NVQXX=$(NVQXX)"
	@echo "NVCC=$(NVCC)"
	@echo "NVCC_FOUND=$(NVCC_FOUND)"
	@echo "CUDA_CU_COMPILER=$(CUDA_CU_COMPILER)"
	@echo "CUDA_CU_FLAGS=$(CUDA_CU_FLAGS)"
	@echo "CUDA_CU_BACKEND=$(CUDA_CU_BACKEND)"
	@echo "CUDAQ_BACKEND=$(CUDAQ_BACKEND)"
	@echo "CUDAQ_TARGET_FLAGS=$(CUDAQ_TARGET_FLAGS)"
	@echo "MIPT_OBJDIR=$(MIPT_OBJDIR)"
	@echo "GMN_OBJDIR=$(GMN_OBJDIR)"
	@echo "FGMN_OBJDIR=$(FGMN_OBJDIR)"
	@echo "RING_OBJDIR=$(RING_OBJDIR)"
	@echo "MIPT_BINARY=$(MIPT_BINARY)"
	@echo "SIM_TMI_BINARY=$(SIM_TMI_BINARY)"
	@echo "ENTROPY_BINARY=$(ENTROPY_BINARY)"
	@echo "GMN_BINARY=$(GMN_BINARY)"
	@echo "FGMN_BINARY=$(FGMN_BINARY)"
	@echo "RING_BINARY=$(RING_BINARY)"
	@echo "TMI_BINARY=$(TMI_BINARY)"
	@echo "SCS_CFLAGS=$(SCS_CFLAGS)"
	@echo "SCS_LIBFILE=$(SCS_LIBFILE)"
	@echo "SCS_LDFLAGS=$(SCS_LDFLAGS)"
	@echo "OPENMP=$(OPENMP)"

mipt.exe: $(MIPT_BINARY)
	cp $(MIPT_BINARY) $@

sim_tmi.exe: $(SIM_TMI_BINARY)
	cp $(SIM_TMI_BINARY) $@

entropy.exe: $(ENTROPY_BINARY)
	cp $(ENTROPY_BINARY) $@

gmn.exe: $(GMN_BINARY)
	cp $(GMN_BINARY) $@

fgmn.exe: $(FGMN_BINARY)
	cp $(FGMN_BINARY) $@

ring_inflation.exe: $(RING_BINARY)
	cp $(RING_BINARY) $@

tmi.exe: $(TMI_BINARY)
	cp $(TMI_BINARY) $@

$(MIPT_BINARY): $(MIPT_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(DL_LIBS) -o $@

$(SIM_TMI_BINARY): $(SIM_TMI_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(SIM_TMI_CUDA_LIBS) $(DL_LIBS) -o $@

$(ENTROPY_BINARY): $(ENTROPY_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(ENTROPY_CUDA_LIBS) $(DL_LIBS) -o $@

$(TMI_BINARY): $(TMI_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(DL_LIBS) -o $@

$(GMN_BINARY): $(GMN_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) -o $@

$(FGMN_BINARY): $(FGMN_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) -o $@

$(RING_BINARY): $(RING_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) $(SCS_LDFLAGS) -o $@

$(MIPT_OBJDIR):
	mkdir -p $@

$(GMN_OBJDIR):
	mkdir -p $@

$(FGMN_OBJDIR):
	mkdir -p $@

$(RING_OBJDIR):
	mkdir -p $@

$(TMI_OBJDIR):
	mkdir -p $@

$(MIPT_OBJDIR)/mipt.o: mipt.cpp mipt_fermion.cpp density_io.hpp rho_reduce.hpp mipt_backend.hpp mipt_native_mps.hpp | $(MIPT_OBJDIR)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@

$(MIPT_OBJDIR)/sim_tmi.o: sim_tmi.cpp mipt.cpp mipt_fermion.cpp density_io.hpp rho_reduce.hpp tmi_cuda_svd.hpp tmi_cuda_rdm.hpp mipt_backend.hpp mipt_native_mps.hpp | $(MIPT_OBJDIR)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@

$(MIPT_OBJDIR)/entropy.o: entropy.cpp entropy_cuda.hpp mipt.cpp mipt_fermion.cpp density_io.hpp rho_reduce.hpp mipt_backend.hpp mipt_native_mps.hpp | $(MIPT_OBJDIR)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@

$(MIPT_OBJDIR)/entropy_cuda.o: entropy_cuda.cu entropy_cuda.hpp | $(MIPT_OBJDIR)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@

$(MIPT_OBJDIR)/rho_reduce.o: rho_reduce.cu rho_reduce.hpp | $(MIPT_OBJDIR)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@

$(MIPT_OBJDIR)/tmi_cuda_svd.o: tmi_cuda_svd.cu tmi_cuda_svd.hpp | $(MIPT_OBJDIR)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@

$(MIPT_OBJDIR)/tmi_cuda_rdm.o: tmi_cuda_rdm.cu tmi_cuda_rdm.hpp | $(MIPT_OBJDIR)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@

$(GMN_OBJDIR)/gmn_main.o: gmn_main.cpp density_io.hpp gmn.hpp | $(GMN_OBJDIR)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@

$(GMN_OBJDIR)/gmn_mosek_c.o: gmn_mosek_c.c gmn.hpp | $(GMN_OBJDIR)
	$(HOSTCC) $(CFLAGS_COMMON) -c $< -o $@

$(FGMN_OBJDIR)/fgmn_main.o: fgmn_main.cpp density_io.hpp fgmn.hpp | $(FGMN_OBJDIR)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@

$(FGMN_OBJDIR)/fgmn.o: fgmn.cpp fgmn.hpp | $(FGMN_OBJDIR)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@

$(RING_OBJDIR)/ring_inflation.o: ring_inflation.cpp density_io.hpp gmn.hpp | $(RING_OBJDIR)
	$(HOSTCXX) $(CXXFLAGS_COMMON) $(SCS_CFLAGS) -c $< -o $@

$(RING_OBJDIR)/gmn_mosek_c.o: gmn_mosek_c.c gmn.hpp | $(RING_OBJDIR)
	$(HOSTCC) $(CFLAGS_COMMON) -c $< -o $@

$(TMI_OBJDIR)/tmi.o: tmi.cpp | $(TMI_OBJDIR)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@

clean:
	rm -rf build mipt.exe sim_tmi.exe entropy.exe gmn.exe fgmn.exe ring_inflation.exe tmi.exe *.tmp
