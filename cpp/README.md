# MIPT codebase

This tree separates reusable simulation infrastructure from executable entry points.
No executable includes another `.cpp` file.

## Layout

- `include/mipt/circuit.hpp` — abstract 1D circuit interface, concrete samplers, and factory.
- `include/mipt/circuits/` — MMS, Haar, RPPU, RFGS, and qRPPU kernels/frontends.
- `include/mipt/rdm.hpp` — qubit and fermionic reduced-density-matrix utilities.
- `include/mipt/native_mps.hpp` — native CPU TEBD/MPS backend.
- `include/mipt/backend.hpp` — runtime backend controls.
- `include/mipt/env.hpp` — shared, range-checked environment parsing.
- `include/mipt/density.hpp` — binary density-matrix file format.
- `include/mipt/cuda/`, `src/cuda/` — CUDA reduction and entropy helpers.
- `include/mipt/analysis/`, `src/analysis/` — GMN, fGMN, and ring-inflation interfaces and implementations.
- `apps/` — one source file per executable.
- `tests/` — dependency-free regression tests for shared infrastructure.
- `docs/` — architecture and migration notes.

## Circuit types

| `circ_type` | Circuit | Trace used for observables |
|---:|---|---|
| 0 | MMS | qubit |
| 1 | Haar U(4) | qubit |
| 2 | fermionic RPPU | fermionic |
| 3 | RFGS | fermionic |
| 4 | qRPPU | qubit |

All 1D executables obtain circuit states through `mipt::Circuit1D`. The factory in
`include/mipt/circuit.hpp` is the sole dispatch point, so kernels and frontend logic
cannot drift between `mipt.exe`, `sim_tmi.exe`, and `entropy.exe`.

## Build

```bash
make simulation                   # mipt, sim_tmi, and entropy
make analysis                     # GMN, fGMN, and ring inflation
make                              # both groups
make GPU=0 CUDA_RHO=0             # CPU CUDA-Q build
make sim_tmi.exe GPU=1 FP64=0
make test-core                    # no CUDA-Q/MOSEK/SCS required
make print-config
```

The CUDA helper compiler falls back from `nvcc` to `nvq++` when `nvcc` is not on
`PATH`. MOSEK is required for GMN/fGMN and SCS is required for ring inflation.

Build outputs are isolated in names such as `build/nvidia_fp32_rho1`, preventing
objects from incompatible configurations from being mixed.

The old Makefile referenced `tmi.cpp`, but that source was absent from the supplied
archive. The stale `tmi.exe` target was therefore removed; `sim_tmi.exe` remains the
maintained online TMI path.

## Refactor invariants

- Circuit gate sequences, measurement sampling, boundary handling, and trace choice
  are unchanged.
- RFGS identifiers were normalized from the historical `FRGS` typo.
- qRPPU is now exposed consistently by the shared circuit factory, including
  `entropy.exe`.
- Generated dependency files (`-MMD -MP`) replace hand-maintained header lists in
  the Makefile.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Migration map](docs/MIGRATION.md)
- [Validation record](docs/VALIDATION.md)

## Validation performed during the refactor

- dependency-free core tests pass;
- deterministic circuit sampling is byte-identical to the supplied implementation
  for MMS, Haar, both fermionic RPPU boundary modes, qRPPU, and RFGS;
- every public header passes standalone compilation and a multi-translation-unit
  ODR/linkage check;
- the CUDA-Q application entry points pass warning-enabled C++20 syntax checks in
  CPU and CUDA-reduction configurations against a local API stub;
- host entry points and the ring-inflation implementation pass warning-enabled
  syntax checks with dependency stubs where required;
- CPU, NVIDIA, `tensornet-mps`, and analysis Makefile command graphs pass dry-runs.

See [the validation record](docs/VALIDATION.md) for the exact scope and limits.
A real CUDA-Q/MOSEK/SCS build remains required on a configured machine.
