# MIPT simulation and entanglement-analysis build.
#
# Common builds:
#   make                                  # GPU CUDA-Q + CUDA reductions
#   make GPU=0 CUDA_RHO=0                 # CPU CUDA-Q
#   make sim_tmi.exe GPU=1 FP64=0
#   make print-config

NVQXX   ?= nvq++
HOSTCXX ?= g++
HOSTCC  ?= gcc
# NVCC is resolved further down, after the toolkit search.

GPU ?= 1
FP64 ?= 0
OPENMP ?= 1
CUDAQ_BACKEND ?= $(if $(MIPT_TENSOR_BACKEND),$(MIPT_TENSOR_BACKEND),nvidia)
COMPILED_BACKEND := $(if $(filter 1,$(GPU)),$(CUDAQ_BACKEND),cpu)
CUDA_RHO ?= $(if $(filter nvidia,$(CUDAQ_BACKEND)),$(GPU),0)

MOSEK_HOME ?= $(HOME)/mosek/11.2/tools/platform/linux64x86
MOSEK_INC  ?= $(MOSEK_HOME)/h
MOSEK_LIB  ?= $(MOSEK_HOME)/bin

SCS_HOME ?= $(HOME)/opt/scs
SCS_INC  ?= $(SCS_HOME)/include
SCS_LIB  ?= $(SCS_HOME)/lib
SCS_CFLAGS ?= -I$(SCS_INC) -I$(SCS_INC)/scs
SCS_LIBFILE ?= $(firstword \
  $(wildcard $(SCS_LIB)/libscsdir.so) \
  $(wildcard $(SCS_LIB)/libscsdir.so.*) \
  $(wildcard $(SCS_LIB)/libscsdir.a) \
  $(wildcard $(SCS_LIB)/libscsindir.so) \
  $(wildcard $(SCS_LIB)/libscsindir.so.*) \
  $(wildcard $(SCS_LIB)/libscsindir.a))

ifeq ($(strip $(SCS_LDFLAGS)),)
  ifneq ($(strip $(SCS_LIBFILE)),)
    SCS_LDFLAGS := $(SCS_LIBFILE) -Wl,-rpath,$(SCS_LIB)
  else
    SCS_LDFLAGS := -L$(SCS_LIB) -Wl,-rpath,$(SCS_LIB) -lscsdir
  endif
endif

CPPFLAGS_COMMON := -Iinclude
NVQXXFLAGS_COMMON := $(CPPFLAGS_COMMON) -std=c++20 -O3 -DNDEBUG -march=native \
  -DMIPT_CUDAQ_BACKEND_NAME=\"$(COMPILED_BACKEND)\" \
  -DMIPT_CUDAQ_PRECISION=$(if $(filter 1,$(FP64)),64,32) -MMD -MP
CXXFLAGS_COMMON := $(CPPFLAGS_COMMON) -std=c++20 -O3 -DNDEBUG -march=native \
  -MMD -MP
CFLAGS_COMMON := $(CPPFLAGS_COMMON) -std=c11 -O3 -DNDEBUG -march=native \
  -MMD -MP
MOSEK_CPPFLAGS := -I$(MOSEK_INC)
# The distro nvcc on PATH may be far older than the installed toolkit (11.5 on
# one of our hosts, which predates Ada and has no -arch for it), so prefer a
# /usr/local toolkit. /usr/local/cuda comes first because it is the symlink the
# system considers current; the version-sorted wildcard is the fallback and its
# last entry is the highest version present. Without an explicit -arch, nvcc
# defaults to sm_52 and every hand-written kernel is JIT-recompiled from PTX at
# load with no architecture-specific code generation.
NVCC ?= $(firstword $(wildcard /usr/local/cuda/bin/nvcc) \
  $(lastword $(wildcard /usr/local/cuda-*/bin/nvcc)) nvcc)
# sm_75 = Turing (GTX 16xx / RTX 20xx), sm_89 = Ada (RTX 40xx). The trailing
# compute_89 keeps PTX in the binary so it still runs on newer architectures.
CUDA_ARCHS ?= -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=compute_89
NVCCFLAGS := $(CPPFLAGS_COMMON) -std=c++17 -O3 $(CUDA_ARCHS)
DL_LIBS ?= -ldl
MOSEK_LDFLAGS := -L$(MOSEK_LIB) -Wl,-rpath,$(MOSEK_LIB)
MOSEK_LIBS := -lmosek64 -lfusion64

ifeq ($(OPENMP),1)
  CXXFLAGS_COMMON += -fopenmp
  CFLAGS_COMMON += -fopenmp
  HOST_LDFLAGS += -fopenmp
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
    NVQXXFLAGS_COMMON += -DSIM_TMI_GPU_BUILD=1 -DMIPT_TENSOR_NETWORK_BUILD=1 \
      -DMIPT_APPROX_MPS_BUILD=1
  else
    $(error Unsupported CUDAQ_BACKEND=$(CUDAQ_BACKEND); use nvidia, tensornet, or tensornet-mps)
  endif
endif

# Locate one CUDA toolkit and use it for both headers and libraries.
#
# Two separate hazards make this explicit rather than implicit:
#
#  * Headers. custatevec.h does `#include <library_types.h>`, which lives in the
#    CUDA toolkit, not in ~/.cudaq/include. On a host that also has the distro
#    nvidia-cuda-dev package the header happens to be in /usr/include (a default
#    clang search path) and the build succeeds by accident -- with a CUDA 11
#    header. On a host without that package the compile fails outright with
#    "'library_types.h' file not found". Pass -I explicitly so neither happens.
#
#  * Libraries. A bare -lcublas resolves to the distro CUDA 11 libcublas.so.11
#    while cuSOLVER drags in CUDA 12's libcublas.so.12, so a GPU executable can
#    load two runtimes at once. That is tolerated until libcustatevec (built
#    against CUDA 12) becomes a direct dependency, at which point it initialises
#    against the CUDA 11 runtime and every device query fails with "no
#    CUDA-capable device is detected".
#
# Override CUDA_HOME (or CUDA_INC_DIR / CUDA_LIB_DIR individually) if the
# toolkit is somewhere unusual.
NVCC_PATH := $(shell command -v $(NVCC) 2>/dev/null)
CUDA_HOME ?= $(firstword \
  $(wildcard $(CUDA_PATH)) \
  $(if $(NVCC_PATH),$(patsubst %/bin/,%,$(dir $(realpath $(NVCC_PATH))))) \
  $(wildcard /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-12))
CUDA_INC_DIR ?= $(firstword $(wildcard \
  $(CUDA_HOME)/include \
  $(CUDA_HOME)/targets/x86_64-linux/include))
CUDA_LIB_DIR ?= $(firstword $(wildcard \
  $(CUDA_HOME)/targets/x86_64-linux/lib \
  $(CUDA_HOME)/lib64))
ifneq ($(strip $(CUDA_LIB_DIR)),)
  CUDA_LDFLAGS := -L$(CUDA_LIB_DIR) -Wl,-rpath,$(CUDA_LIB_DIR)
endif

ifeq ($(CUDA_RHO),1)
  NVQXXFLAGS_COMMON += -DMIPT_ENABLE_CUDA_RHO
  MIPT_CUDA_LIBS := $(CUDA_LDFLAGS) -lcudart
  SIM_TMI_CUDA_LIBS := -lcusolver -lcublas
  # cuSOLVER is for the von Neumann (S1) spectrum; the Renyi-2 path needs only
  # the cuBLAS batched GEMM.
  ENTROPY_CUDA_LIBS := -lcublas -lcusolver
endif

# Direct cuStateVec circuit engine (mipt/cusv). Replaces CUDA-Q's gate-by-gate
# dispatch with fused bond matrices, a single diagonal pass for the
# Jordan-Wigner string, a joint measurement layer, and an in-place state vector.
# Requires the GPU statevector target; cuStateVec ships with CUDA-Q itself, so
# there is no extra dependency to install.
CUSV ?= $(if $(filter 1,$(GPU)),$(if $(filter nvidia,$(CUDAQ_BACKEND)),1,0),0)
ifeq ($(CUSV),1)
  ifeq ($(strip $(CUDA_INC_DIR)),)
    $(error CUSV=1 needs the CUDA toolkit headers, because custatevec.h includes \
      <library_types.h>. No toolkit include directory was found (looked at CUDA_PATH, \
      the directory containing $(NVCC), and /usr/local/cuda*). Set CUDA_HOME=/path/to/cuda \
      or CUDA_INC_DIR=/path/to/cuda/include, or build with CUSV=0 to use the CUDA-Q path)
  endif
  NVQXXFLAGS_COMMON += -DMIPT_ENABLE_CUSV -I$(CUDA_INC_DIR)
  CUSV_LIBS := -lcustatevec
endif

ifeq ($(strip $(NVCC_PATH)),)
  CUDA_CU_COMPILER ?= $(NVQXX)
  CUDA_CU_FLAGS ?= $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON)
  CUDA_CU_BACKEND := nvq++ fallback
else
  CUDA_CU_COMPILER ?= $(NVCC)
  CUDA_CU_FLAGS ?= $(NVCCFLAGS)
  CUDA_CU_BACKEND := nvcc
endif

PRECISION := $(if $(filter 1,$(FP64)),64,32)
BACKEND_TAG := $(subst -,_,$(COMPILED_BACKEND))
CONFIG := $(BACKEND_TAG)_fp$(PRECISION)_rho$(CUDA_RHO)
BUILD := build/$(CONFIG)
HOST_BUILD := build/host_omp$(OPENMP)

MIPT_BIN := $(BUILD)/mipt.exe
MIPT_PROBED_BIN := $(BUILD)/mipt_probed.exe
FREE_ENERGY_BIN := $(BUILD)/free_energy.exe
SIM_TMI_BIN := $(BUILD)/sim_tmi.exe
DIST_SCALING_BIN := $(BUILD)/dist_scaling.exe
ENTROPY_BIN := $(BUILD)/entropy.exe
GMN_BIN := $(HOST_BUILD)/gmn.exe
FGMN_BIN := $(HOST_BUILD)/fgmn.exe
RING_BIN := $(HOST_BUILD)/ring_inflation.exe
EXP_VALS_BIN := $(HOST_BUILD)/exp_vals.exe
CORE_TEST_BIN := $(HOST_BUILD)/core_tests
DIST_TEST_BIN := $(HOST_BUILD)/dist_metrics_tests
SPECTRAL_TEST_BIN := $(HOST_BUILD)/spectral_tests
FRONT_TEST_BIN := $(HOST_BUILD)/front_metrics_tests
GMN_SOLVER_TEST_BIN := $(HOST_BUILD)/gmn_solver_tests

MIPT_OBJS := $(BUILD)/mipt.o
MIPT_PROBED_OBJS := $(BUILD)/mipt_probed.o
MIPT_PROBED_FGMN_PIC_OBJ := $(HOST_BUILD)/fgmn_pic.o
MIPT_PROBED_FGMN_LIB := $(BUILD)/libmipt_fgmn.so
FREE_ENERGY_OBJS := $(BUILD)/free_energy.o
SIM_TMI_OBJS := $(BUILD)/sim_tmi.o
DIST_SCALING_OBJS := $(BUILD)/dist_scaling.o
ENTROPY_OBJS := $(BUILD)/entropy.o
GMN_OBJS := $(HOST_BUILD)/gmn.o $(HOST_BUILD)/gmn_mosek.o $(HOST_BUILD)/fgmn.o
FGMN_OBJS := $(HOST_BUILD)/fgmn_main.o $(HOST_BUILD)/fgmn.o
RING_OBJS := $(HOST_BUILD)/ring_inflation_main.o $(HOST_BUILD)/ring_inflation.o $(HOST_BUILD)/gmn_mosek.o
EXP_VALS_OBJS := $(HOST_BUILD)/exp_vals_main.o $(HOST_BUILD)/exp_vals_analysis.o
CORE_TEST_OBJS := $(HOST_BUILD)/core_tests.o
DIST_TEST_OBJS := $(HOST_BUILD)/dist_metrics_tests.o
SPECTRAL_TEST_OBJS := $(HOST_BUILD)/spectral_tests.o
FRONT_TEST_OBJS := $(HOST_BUILD)/front_metrics_tests.o
# Exercises both GMN back ends, so it links the C API and the Fusion object.
GMN_SOLVER_TEST_OBJS := $(HOST_BUILD)/gmn_solver_tests.o $(HOST_BUILD)/gmn_mosek.o \
  $(HOST_BUILD)/fgmn.o

# Every executable that builds circuit states now goes through mipt/cusv.
ifeq ($(CUSV),1)
  MIPT_OBJS += $(BUILD)/cusv_diag.o
  MIPT_PROBED_OBJS += $(BUILD)/cusv_diag.o
  FREE_ENERGY_OBJS += $(BUILD)/cusv_diag.o
  SIM_TMI_OBJS += $(BUILD)/cusv_diag.o
  DIST_SCALING_OBJS += $(BUILD)/cusv_diag.o
  ENTROPY_OBJS += $(BUILD)/cusv_diag.o
endif

ifeq ($(CUDA_RHO),1)
  MIPT_OBJS += $(BUILD)/rho_reduce.o
  MIPT_PROBED_OBJS += $(BUILD)/ancilla_rdm.o
  FREE_ENERGY_OBJS += $(BUILD)/free_energy_measure.o $(BUILD)/tmi_cuda_svd.o
  SIM_TMI_OBJS += $(BUILD)/rho_reduce.o $(BUILD)/tmi_cuda_svd.o $(BUILD)/tmi_cuda_rdm.o
  # k=2 uses the batched two-site reduction, k=3 the batched three-site one.
  DIST_SCALING_OBJS += $(BUILD)/rho2_reduce.o $(BUILD)/rho_reduce.o
  ENTROPY_OBJS += $(BUILD)/rho_reduce.o $(BUILD)/entropy_cuda.o
endif

ALL_OBJS := $(MIPT_OBJS) $(MIPT_PROBED_OBJS) $(FREE_ENERGY_OBJS) \
  $(SIM_TMI_OBJS) $(DIST_SCALING_OBJS) $(ENTROPY_OBJS) $(GMN_OBJS) $(FGMN_OBJS) \
  $(RING_OBJS) $(EXP_VALS_OBJS) $(CORE_TEST_OBJS) $(DIST_TEST_OBJS) \
  $(SPECTRAL_TEST_OBJS) $(FRONT_TEST_OBJS) $(GMN_SOLVER_TEST_OBJS) \
  $(MIPT_PROBED_FGMN_PIC_OBJ)
DEPS := $(ALL_OBJS:.o=.d)

# nvq++ accepts -MMD/-MP but does not emit dependency files, so the CUDA-Q
# objects below would never notice an edited header and would silently link a
# stale binary. Depend them on every header instead: coarse, but a header
# change must never produce a stale simulation executable.
MIPT_HEADERS := $(shell find include -name '*.hpp' 2>/dev/null)
# The .cu objects are built by nvcc without -MMD/-MP, so nothing noticed an
# edited .cuh and the link picked up a stale object. Depend them on every CUDA
# header explicitly: coarse, but a header change must never silently survive
# into a binary. (Same reasoning as MIPT_HEADERS above, which covers nvq++.)
MIPT_CUDA_HEADERS := $(shell find include -name '*.cuh' 2>/dev/null) $(MIPT_HEADERS)

.PHONY: all simulation analysis clean cpu gpu print-config format test-core test-dist test-front \
  test-spectral test-gmn test-rdm-gram test-rho-reduce test-cusv test-cusv-gates \
  test-mode3 bench-cusv \
  mipt.exe mipt_probed.exe free_energy.exe sim_tmi.exe dist_scaling.exe \
  entropy.exe gmn.exe fgmn.exe ring_inflation.exe exp_vals.exe

all: simulation analysis
simulation: mipt.exe mipt_probed.exe free_energy.exe sim_tmi.exe dist_scaling.exe entropy.exe
analysis: gmn.exe fgmn.exe ring_inflation.exe exp_vals.exe
cpu:
	$(MAKE) all GPU=0 CUDA_RHO=0
gpu:
	$(MAKE) all GPU=1 CUDA_RHO=1 FP64=$(FP64)

mipt.exe: $(MIPT_BIN)
	cp $< $@
mipt_probed.exe: $(MIPT_PROBED_BIN) $(MIPT_PROBED_FGMN_LIB)
	cp $(MIPT_PROBED_BIN) $@
	cp $(MIPT_PROBED_FGMN_LIB) ./libmipt_fgmn.so
free_energy.exe: $(FREE_ENERGY_BIN)
	cp $< $@
sim_tmi.exe: $(SIM_TMI_BIN)
	cp $< $@
dist_scaling.exe: $(DIST_SCALING_BIN) $(MIPT_PROBED_FGMN_LIB)
	cp $(DIST_SCALING_BIN) $@
	cp $(MIPT_PROBED_FGMN_LIB) ./libmipt_fgmn.so
entropy.exe: $(ENTROPY_BIN)
	cp $< $@
gmn.exe: $(GMN_BIN)
	cp $< $@
fgmn.exe: $(FGMN_BIN)
	cp $< $@
ring_inflation.exe: $(RING_BIN)
	cp $< $@
exp_vals.exe: $(EXP_VALS_BIN)
	cp $< $@

$(MIPT_BIN): $(MIPT_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(CUSV_LIBS) $(DL_LIBS) -o $@
$(MIPT_PROBED_BIN): $(MIPT_PROBED_OBJS) $(MIPT_PROBED_FGMN_LIB)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(HOST_LDFLAGS) $(MIPT_PROBED_OBJS) \
	  $(MIPT_CUDA_LIBS) $(CUSV_LIBS) -L$(BUILD) -Wl,-rpath,'$$ORIGIN' -lmipt_fgmn \
	  $(DL_LIBS) -o $@
$(MIPT_PROBED_FGMN_LIB): $(MIPT_PROBED_FGMN_PIC_OBJ) | $(BUILD)
	$(HOSTCXX) $(HOST_LDFLAGS) -shared $< $(MOSEK_LDFLAGS) $(MOSEK_LIBS) \
	  $(DL_LIBS) -Wl,-soname,libmipt_fgmn.so -o $@
$(FREE_ENERGY_BIN): $(FREE_ENERGY_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(SIM_TMI_CUDA_LIBS) $(CUSV_LIBS) $(DL_LIBS) -o $@
$(SIM_TMI_BIN): $(SIM_TMI_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(SIM_TMI_CUDA_LIBS) $(CUSV_LIBS) $(DL_LIBS) -o $@
# k=3 solves GMN/fGMN, so this links the same MOSEK shim mipt_probed uses.
$(DIST_SCALING_BIN): $(DIST_SCALING_OBJS) $(MIPT_PROBED_FGMN_LIB)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(HOST_LDFLAGS) $(DIST_SCALING_OBJS) \
	  $(MIPT_CUDA_LIBS) $(CUSV_LIBS) -L$(BUILD) -Wl,-rpath,'$$ORIGIN' -lmipt_fgmn \
	  $(DL_LIBS) -o $@
$(ENTROPY_BIN): $(ENTROPY_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(ENTROPY_CUDA_LIBS) $(CUSV_LIBS) $(DL_LIBS) -o $@
$(GMN_BIN): $(GMN_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) -o $@
$(FGMN_BIN): $(FGMN_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) -o $@
$(RING_BIN): $(RING_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) $(SCS_LDFLAGS) -o $@
$(EXP_VALS_BIN): $(EXP_VALS_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ -o $@

$(CORE_TEST_BIN): $(CORE_TEST_OBJS)
	$(HOSTCXX) $^ -o $@
$(DIST_TEST_BIN): $(DIST_TEST_OBJS)
	$(HOSTCXX) $^ -o $@
$(SPECTRAL_TEST_BIN): $(SPECTRAL_TEST_OBJS)
	$(HOSTCXX) $^ -o $@
$(FRONT_TEST_BIN): $(FRONT_TEST_OBJS)
	$(HOSTCXX) $^ -o $@
$(GMN_SOLVER_TEST_BIN): $(GMN_SOLVER_TEST_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) -o $@

test-core: $(CORE_TEST_BIN)
	$(CORE_TEST_BIN)
test-dist: $(DIST_TEST_BIN)
	$(DIST_TEST_BIN)
test-spectral: $(SPECTRAL_TEST_BIN)
	$(SPECTRAL_TEST_BIN)
test-front: $(FRONT_TEST_BIN)
	$(FRONT_TEST_BIN)
# Needs MOSEK, no GPU. Pins both GMN back ends and fGMN to closed forms; see the
# header comment in tests/gmn_solver_tests.cpp for why |W> is the load-bearing
# case. Single-threaded so the check is deterministic.
test-gmn: $(GMN_SOLVER_TEST_BIN)
	GMN_MOSEK_NUM_THREADS=1 $(GMN_SOLVER_TEST_BIN)

# --- reduction-kernel equivalence -------------------------------------------
#
# Checks the gather + cuBLAS-herk RDM/entropy path against a host fp64
# reference. Needs a GPU. Built with nvcc directly since it drives the CUDA
# reduction objects without any CUDA-Q involvement.
RDM_GRAM_TEST_BIN := $(BUILD)/rdm_gram_tests
$(RDM_GRAM_TEST_BIN): tests/rdm_gram_tests.cu $(BUILD)/tmi_cuda_rdm.o $(BUILD)/tmi_cuda_svd.o | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ $(if $(CUDA_LIB_DIR),-L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)) -lcublas -lcusolver -o $@
test-rdm-gram: $(RDM_GRAM_TEST_BIN)
	$(RDM_GRAM_TEST_BIN)

# Fused two- and three-site reduction sweeps against a from-first-principles
# host reference (permutation parity, not the kernels' own sign masks), for
# both traces, both precisions and unsorted retained sets. Needs a GPU.
RHO_REDUCE_TEST_BIN := $(BUILD)/rho_reduce_tests
$(RHO_REDUCE_TEST_BIN): tests/rho_reduce_tests.cu $(BUILD)/rho2_reduce.o $(BUILD)/rho_reduce.o | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ $(if $(CUDA_LIB_DIR),-L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)) -o $@
test-rho-reduce: $(RHO_REDUCE_TEST_BIN)
	$(RHO_REDUCE_TEST_BIN)

# --- cuStateVec engine tests ------------------------------------------------
#
# dump_bond extracts each bond's 4x4 matrix from the *CUDA-Q runtime* by
# applying the historical gate decomposition to every basis state. It is always
# built in fp64 so the comparison tolerance can be tight regardless of FP64.
# cusv_gate_tests then checks the fused matrices against that dump, and
# cusv_equiv runs whole circuits both ways and compares state vectors.
CUDAQ_INC ?= $(HOME)/.cudaq/include
CUSV_GATE_TEST_BIN := $(HOST_BUILD)/cusv_gate_tests
DUMP_BOND_BIN := $(BUILD)/dump_bond
CUSV_EQUIV_BIN := $(BUILD)/cusv_equiv
MODE3_EQUIV_BIN := $(BUILD)/mode3_equiv
CUSV_BENCH_BIN := $(BUILD)/cusv_bench
CUSV_GROUND_TRUTH := $(BUILD)/cusv_ground_truth.txt
CUSV_TEST_SEED ?= 12345

$(CUSV_GATE_TEST_BIN): tests/cusv_gate_tests.cpp $(MIPT_HEADERS) | $(HOST_BUILD)
	$(HOSTCXX) $(CPPFLAGS_COMMON) -I$(CUDAQ_INC) -std=c++20 -O2 -Wno-attributes $< -o $@

$(DUMP_BOND_BIN): tests/dump_bond.cpp $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) --target nvidia --target-option fp64 $(CPPFLAGS_COMMON) -std=c++20 -O2 \
	  -DNDEBUG -DMIPT_CUDAQ_BACKEND_NAME=\"nvidia\" -DMIPT_CUDAQ_PRECISION=64 $< -o $@

$(CUSV_EQUIV_BIN): tests/cusv_equiv.cpp $(BUILD)/cusv_diag.o $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) $< $(BUILD)/cusv_diag.o \
	  $(MIPT_CUDA_LIBS) $(CUSV_LIBS) $(DL_LIBS) -o $@

$(MODE3_EQUIV_BIN): tests/mode3_equiv.cpp $(BUILD)/cusv_diag.o $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) $< $(BUILD)/cusv_diag.o \
	  $(MIPT_CUDA_LIBS) $(CUSV_LIBS) $(DL_LIBS) -o $@

$(CUSV_BENCH_BIN): tests/cusv_bench.cpp $(BUILD)/cusv_diag.o $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) $< $(BUILD)/cusv_diag.o \
	  $(MIPT_CUDA_LIBS) $(CUSV_LIBS) $(DL_LIBS) -o $@

$(CUSV_GROUND_TRUTH): $(DUMP_BOND_BIN)
	$(DUMP_BOND_BIN) $(CUSV_TEST_SEED) > $@

# Host-only: no GPU, no cuStateVec. Needs the ground-truth dump, which does.
test-cusv-gates: $(CUSV_GATE_TEST_BIN) $(CUSV_GROUND_TRUTH)
	$(CUSV_GATE_TEST_BIN) $(CUSV_GROUND_TRUTH) $(CUSV_TEST_SEED)

test-cusv: test-cusv-gates $(CUSV_EQUIV_BIN)
	$(CUSV_EQUIV_BIN) 12 6 2024
	$(CUSV_EQUIV_BIN) 14 8 31337

# The mode-3 pattern cusv_equiv cannot reach: a register two qubits wider than
# the circuit, with references attached mid-trajectory.
test-mode3: $(MODE3_EQUIV_BIN)
	$(MODE3_EQUIV_BIN) 10 2024
	$(MODE3_EQUIV_BIN) 12 31337

bench-cusv: $(CUSV_BENCH_BIN)
	$(CUSV_BENCH_BIN) 20 16 0.15 2

$(BUILD) $(HOST_BUILD):
	mkdir -p $@

$(BUILD)/mipt.o: apps/mipt.cpp $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@
$(BUILD)/mipt_probed.o: apps/mipt_probed.cpp $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@
$(BUILD)/free_energy.o: apps/free_energy.cpp $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@
$(BUILD)/sim_tmi.o: apps/sim_tmi.cpp $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@
$(BUILD)/dist_scaling.o: apps/dist_scaling.cpp $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@
$(BUILD)/entropy.o: apps/entropy.cpp $(MIPT_HEADERS) | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@

$(BUILD)/rho_reduce.o: src/cuda/rho_reduce.cu $(MIPT_CUDA_HEADERS) | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/rho2_reduce.o: src/cuda/rho2_reduce.cu $(MIPT_CUDA_HEADERS) | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/ancilla_rdm.o: src/cuda/ancilla_rdm.cu $(MIPT_CUDA_HEADERS) | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/free_energy_measure.o: src/cuda/free_energy_measure.cu $(MIPT_CUDA_HEADERS) | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/tmi_cuda_svd.o: src/cuda/tmi_cuda_svd.cu $(MIPT_CUDA_HEADERS) | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/tmi_cuda_rdm.o: src/cuda/tmi_cuda_rdm.cu $(MIPT_CUDA_HEADERS) | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/entropy_cuda.o: src/cuda/entropy_cuda.cu $(MIPT_CUDA_HEADERS) | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/cusv_diag.o: src/cuda/cusv_diag.cu $(MIPT_CUDA_HEADERS) | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@

$(HOST_BUILD)/gmn.o: apps/gmn.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@
$(HOST_BUILD)/fgmn_main.o: apps/fgmn.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@
$(HOST_BUILD)/fgmn.o: src/analysis/fgmn.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) $(MOSEK_CPPFLAGS) -c $< -o $@
$(MIPT_PROBED_FGMN_PIC_OBJ): src/analysis/fgmn.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) $(MOSEK_CPPFLAGS) -fPIC -c $< -o $@
$(HOST_BUILD)/ring_inflation_main.o: apps/ring_inflation.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@
$(HOST_BUILD)/ring_inflation.o: src/analysis/ring_inflation.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) $(SCS_CFLAGS) -c $< -o $@
$(HOST_BUILD)/exp_vals_main.o: apps/exp_vals.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@
$(HOST_BUILD)/exp_vals_analysis.o: src/analysis/exp_vals.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@
$(HOST_BUILD)/gmn_mosek.o: src/analysis/gmn_mosek.c | $(HOST_BUILD)
	$(HOSTCC) $(CFLAGS_COMMON) $(MOSEK_CPPFLAGS) -c $< -o $@
$(HOST_BUILD)/core_tests.o: tests/core_tests.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CPPFLAGS_COMMON) -std=c++20 -O2 -MMD -MP -c $< -o $@
$(HOST_BUILD)/dist_metrics_tests.o: tests/dist_metrics_tests.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CPPFLAGS_COMMON) -std=c++20 -O2 -MMD -MP -c $< -o $@
$(HOST_BUILD)/spectral_tests.o: tests/spectral_tests.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CPPFLAGS_COMMON) -std=c++20 -O2 -MMD -MP -c $< -o $@
$(HOST_BUILD)/front_metrics_tests.o: tests/front_metrics_tests.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CPPFLAGS_COMMON) -std=c++20 -O2 -MMD -MP -c $< -o $@
$(HOST_BUILD)/gmn_solver_tests.o: tests/gmn_solver_tests.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CPPFLAGS_COMMON) $(MOSEK_CPPFLAGS) -std=c++20 -O2 -MMD -MP -c $< -o $@

print-config:
	@echo "CONFIG=$(CONFIG)"
	@echo "PRECISION=$(PRECISION)"
	@echo "CUDAQ_BACKEND=$(CUDAQ_BACKEND)"
	@echo "COMPILED_BACKEND=$(COMPILED_BACKEND)"
	@echo "CUDAQ_TARGET_FLAGS=$(CUDAQ_TARGET_FLAGS)"
	@echo "CUDA_RHO=$(CUDA_RHO)"
	@echo "CUSV=$(CUSV)"
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "CUDA_INC_DIR=$(CUDA_INC_DIR)"
	@echo "CUDA_LIB_DIR=$(CUDA_LIB_DIR)"
	@echo "CUDA_CU_COMPILER=$(CUDA_CU_COMPILER)"
	@echo "CUDA_CU_BACKEND=$(CUDA_CU_BACKEND)"
	@echo "CUDA_ARCHS=$(CUDA_ARCHS)"
	@echo "BUILD=$(BUILD)"
	@echo "SCS_LDFLAGS=$(SCS_LDFLAGS)"
	@echo "OPENMP=$(OPENMP)"

format:
	@command -v clang-format >/dev/null || { echo "clang-format is not installed."; exit 1; }
	clang-format -i $$(find apps include src tests -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.c' \))

clean:
	rm -rf build *.exe *.tmp libmipt_fgmn.so

-include $(DEPS)
