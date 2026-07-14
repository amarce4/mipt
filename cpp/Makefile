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
NVCC    ?= nvcc

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
  -DMIPT_CUDAQ_BACKEND_NAME=\"$(COMPILED_BACKEND)\" -MMD -MP
CXXFLAGS_COMMON := $(CPPFLAGS_COMMON) -std=c++20 -O3 -DNDEBUG -march=native \
  -MMD -MP
CFLAGS_COMMON := $(CPPFLAGS_COMMON) -std=c11 -O3 -DNDEBUG -march=native \
  -MMD -MP
MOSEK_CPPFLAGS := -I$(MOSEK_INC)
NVCCFLAGS := $(CPPFLAGS_COMMON) -std=c++14 -O3
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

ifeq ($(CUDA_RHO),1)
  NVQXXFLAGS_COMMON += -DMIPT_ENABLE_CUDA_RHO
  MIPT_CUDA_LIBS := -lcudart
  SIM_TMI_CUDA_LIBS := -lcusolver -lcublas
  ENTROPY_CUDA_LIBS := -lcublas
endif

NVCC_FOUND := $(shell command -v $(NVCC) 2>/dev/null)
ifeq ($(strip $(NVCC_FOUND)),)
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
SIM_TMI_BIN := $(BUILD)/sim_tmi.exe
ENTROPY_BIN := $(BUILD)/entropy.exe
GMN_BIN := $(HOST_BUILD)/gmn.exe
FGMN_BIN := $(HOST_BUILD)/fgmn.exe
RING_BIN := $(HOST_BUILD)/ring_inflation.exe
CORE_TEST_BIN := $(HOST_BUILD)/core_tests

MIPT_OBJS := $(BUILD)/mipt.o
SIM_TMI_OBJS := $(BUILD)/sim_tmi.o
ENTROPY_OBJS := $(BUILD)/entropy.o
GMN_OBJS := $(HOST_BUILD)/gmn.o $(HOST_BUILD)/gmn_mosek.o $(HOST_BUILD)/fgmn.o
FGMN_OBJS := $(HOST_BUILD)/fgmn_main.o $(HOST_BUILD)/fgmn.o
RING_OBJS := $(HOST_BUILD)/ring_inflation_main.o $(HOST_BUILD)/ring_inflation.o $(HOST_BUILD)/gmn_mosek.o
CORE_TEST_OBJS := $(HOST_BUILD)/core_tests.o

ifeq ($(CUDA_RHO),1)
  MIPT_OBJS += $(BUILD)/rho_reduce.o
  SIM_TMI_OBJS += $(BUILD)/rho_reduce.o $(BUILD)/tmi_cuda_svd.o $(BUILD)/tmi_cuda_rdm.o
  ENTROPY_OBJS += $(BUILD)/rho_reduce.o $(BUILD)/entropy_cuda.o
endif

ALL_OBJS := $(MIPT_OBJS) $(SIM_TMI_OBJS) $(ENTROPY_OBJS) \
  $(GMN_OBJS) $(FGMN_OBJS) $(RING_OBJS) $(CORE_TEST_OBJS)
DEPS := $(ALL_OBJS:.o=.d)

.PHONY: all simulation analysis clean cpu gpu print-config format test-core \
  mipt.exe sim_tmi.exe entropy.exe gmn.exe fgmn.exe ring_inflation.exe

all: simulation analysis
simulation: mipt.exe sim_tmi.exe entropy.exe
analysis: gmn.exe fgmn.exe ring_inflation.exe
cpu:
	$(MAKE) all GPU=0 CUDA_RHO=0
gpu:
	$(MAKE) all GPU=1 CUDA_RHO=1 FP64=$(FP64)

mipt.exe: $(MIPT_BIN)
	cp $< $@
sim_tmi.exe: $(SIM_TMI_BIN)
	cp $< $@
entropy.exe: $(ENTROPY_BIN)
	cp $< $@
gmn.exe: $(GMN_BIN)
	cp $< $@
fgmn.exe: $(FGMN_BIN)
	cp $< $@
ring_inflation.exe: $(RING_BIN)
	cp $< $@

$(MIPT_BIN): $(MIPT_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(DL_LIBS) -o $@
$(SIM_TMI_BIN): $(SIM_TMI_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(SIM_TMI_CUDA_LIBS) $(DL_LIBS) -o $@
$(ENTROPY_BIN): $(ENTROPY_OBJS)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $^ $(MIPT_CUDA_LIBS) $(ENTROPY_CUDA_LIBS) $(DL_LIBS) -o $@
$(GMN_BIN): $(GMN_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) -o $@
$(FGMN_BIN): $(FGMN_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) -o $@
$(RING_BIN): $(RING_OBJS)
	$(HOSTCXX) $(HOST_LDFLAGS) $^ $(MOSEK_LDFLAGS) $(MOSEK_LIBS) $(SCS_LDFLAGS) -o $@

$(CORE_TEST_BIN): $(CORE_TEST_OBJS)
	$(HOSTCXX) $^ -o $@

test-core: $(CORE_TEST_BIN)
	$(CORE_TEST_BIN)

$(BUILD) $(HOST_BUILD):
	mkdir -p $@

$(BUILD)/mipt.o: apps/mipt.cpp | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@
$(BUILD)/sim_tmi.o: apps/sim_tmi.cpp | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@
$(BUILD)/entropy.o: apps/entropy.cpp | $(BUILD)
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) $(NVQXXFLAGS_COMMON) -c $< -o $@

$(BUILD)/rho_reduce.o: src/cuda/rho_reduce.cu | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/tmi_cuda_svd.o: src/cuda/tmi_cuda_svd.cu | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/tmi_cuda_rdm.o: src/cuda/tmi_cuda_rdm.cu | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@
$(BUILD)/entropy_cuda.o: src/cuda/entropy_cuda.cu | $(BUILD)
	$(CUDA_CU_COMPILER) $(CUDA_CU_FLAGS) -c $< -o $@

$(HOST_BUILD)/gmn.o: apps/gmn.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@
$(HOST_BUILD)/fgmn_main.o: apps/fgmn.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@
$(HOST_BUILD)/fgmn.o: src/analysis/fgmn.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) $(MOSEK_CPPFLAGS) -c $< -o $@
$(HOST_BUILD)/ring_inflation_main.o: apps/ring_inflation.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) -c $< -o $@
$(HOST_BUILD)/ring_inflation.o: src/analysis/ring_inflation.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CXXFLAGS_COMMON) $(SCS_CFLAGS) -c $< -o $@
$(HOST_BUILD)/gmn_mosek.o: src/analysis/gmn_mosek.c | $(HOST_BUILD)
	$(HOSTCC) $(CFLAGS_COMMON) $(MOSEK_CPPFLAGS) -c $< -o $@
$(HOST_BUILD)/core_tests.o: tests/core_tests.cpp | $(HOST_BUILD)
	$(HOSTCXX) $(CPPFLAGS_COMMON) -std=c++20 -O2 -MMD -MP -c $< -o $@

print-config:
	@echo "CONFIG=$(CONFIG)"
	@echo "PRECISION=$(PRECISION)"
	@echo "CUDAQ_BACKEND=$(CUDAQ_BACKEND)"
	@echo "COMPILED_BACKEND=$(COMPILED_BACKEND)"
	@echo "CUDAQ_TARGET_FLAGS=$(CUDAQ_TARGET_FLAGS)"
	@echo "CUDA_RHO=$(CUDA_RHO)"
	@echo "CUDA_CU_COMPILER=$(CUDA_CU_COMPILER)"
	@echo "CUDA_CU_BACKEND=$(CUDA_CU_BACKEND)"
	@echo "BUILD=$(BUILD)"
	@echo "SCS_LDFLAGS=$(SCS_LDFLAGS)"
	@echo "OPENMP=$(OPENMP)"

format:
	@command -v clang-format >/dev/null || { echo "clang-format is not installed."; exit 1; }
	clang-format -i $$(find apps include src tests -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.c' \))

clean:
	rm -rf build *.exe *.tmp

-include $(DEPS)
