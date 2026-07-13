# Validation record

The supplied environment does not contain CUDA-Q, CUDA compiler tooling, MOSEK
Fusion, or SCS. Validation therefore separates dependency-free behavioral checks
from interface/build checks that use minimal local API stubs.

## Passed checks

### Scientific sampler equivalence

With `n=8`, `periods=3`, `p=0.37`, and a fixed `std::mt19937` seed, the original
and refactored samplers produced byte-identical serialized layer data for:

- MMS;
- Haar U(4);
- fermionic RPPU with FSWAP boundary handling;
- fermionic RPPU with Jordan-Wigner boundary handling;
- qRPPU;
- RFGS.

This covers gate parameters, measurement flags, single-site choices, bond order,
and periodic-boundary construction. It is the strongest dependency-free check
that circuit sampling was preserved exactly.

### Core regression tests

`make test-core` passes and covers:

- circuit metadata and invalid circuit-type rejection;
- probability-grid generation;
- range-checked environment parsing;
- binary density-file write/read round trips.

### Translation-unit and header checks

- Every public header compiles as the sole include in an otherwise empty C++20
  translation unit.
- A two-translation-unit linkage test passes for the shared circuit, density,
  entropy, native-MPS, and TMI headers. This checks that header-defined functions
  are ODR-safe.
- `mipt.cpp`, `sim_tmi.cpp`, and `entropy.cpp` pass warning-enabled C++20 syntax
  checks in both CPU-like and `MIPT_ENABLE_CUDA_RHO` configurations against a
  minimal CUDA-Q API stub.
- GMN, fGMN, and ring-inflation entry points pass warning-enabled host syntax
  checks. The ring-inflation implementation also passes against a minimal SCS API
  stub.
- No source file textually includes another `.cpp` file.
- Circuit selection has one dispatch switch, in `make_circuit_1d`.

### Build graph checks

Make dry-runs pass for:

- CPU CUDA-Q simulation;
- NVIDIA CUDA-Q simulation with CUDA reduced-density helpers;
- `tensornet-mps` simulation;
- all analysis executables.

The generated paths distinguish backend, scalar precision, and CUDA-RDM mode, so
incompatible object files cannot be silently reused.

## External validation still required

On a configured simulation machine, run at minimum:

```bash
make clean
make simulation GPU=1 CUDA_RHO=1 FP64=0
make test-core
./mipt.exe --help
./sim_tmi.exe --help
./entropy.exe --help
```

On a solver machine, additionally run:

```bash
make analysis
./gmn.exe --help
./fgmn.exe --help
./ring_inflation.exe --help
```

A real CUDA-Q build is necessary to validate compiler-specific `__qpu__` lowering
and CUDA linkage. Real MOSEK/Fusion and SCS builds are necessary to validate their
SDK-specific APIs and runtime linkage. Those claims are intentionally not inferred
from stub-based checks.
