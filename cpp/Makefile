# CUDA-Q + MOSEK Fusion build
#
# Usage:
#   make mipt.exe          # CPU/default CUDA-Q target
#   make mipt.exe GPU=1    # GPU target: adds --target nvidia
#   make gpu mipt.exe      # same as GPU=1
#   make clean
#
# Required environment variables:
#   MSKINC = path to MOSEK headers
#   MSKLIB = path to MOSEK libraries/bin

NVQXX ?= nvq++
CXX   ?= g++

# Allow: make gpu mipt.exe
ifneq ($(filter gpu,$(MAKECMDGOALS)),)
GPU := 1
endif

ifeq ($(GPU),1)
CUDAQ_TARGET_FLAGS := --target nvidia
else
CUDAQ_TARGET_FLAGS :=
endif

CXXSTD   := -std=c++20
OPTFLAGS := -O2

EIGEN_INC ?= /usr/include/eigen3

CUDAQ_CXXFLAGS := $(CUDAQ_TARGET_FLAGS) $(CXXSTD) -I$(EIGEN_INC)
GMN_CXXFLAGS   := $(CXXSTD) $(OPTFLAGS) -fPIC -I$(MSKINC)

.PHONY: clean check-env gpu

gpu:
	@:

check-env:
	@test -n "$(MSKINC)" || (echo "ERROR: MSKINC is not set"; exit 1)
	@test -n "$(MSKLIB)" || (echo "ERROR: MSKLIB is not set"; exit 1)

gmn.o: gmn.cpp gmn.hpp | check-env
	$(CXX) $(GMN_CXXFLAGS) -c gmn.cpp -o gmn.o

libgmn.so: gmn.o | check-env
	$(CXX) -shared gmn.o \
		-L$(MSKLIB) \
		-Wl,-rpath,$(MSKLIB) \
		-lfusion64 -lmosek64 \
		-o libgmn.so

mipt_1d.o: mipt_1d.cpp gmn.hpp
	$(NVQXX) $(CUDAQ_CXXFLAGS) -c mipt_1d.cpp -o mipt_1d.o

%.exe: mipt_1d.o libgmn.so
	$(NVQXX) $(CUDAQ_TARGET_FLAGS) mipt_1d.o \
		-L. \
		-Wl,-rpath,'$$ORIGIN' \
		-Wl,-rpath,$(MSKLIB) \
		-lgmn \
		-o $@

clean:
	rm -f *.o *.so *.exe
