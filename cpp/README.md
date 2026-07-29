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
- `include/mipt/cusv/` — direct cuStateVec circuit engine (see below).
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
make simulation                   # mipt, probes, free energy, TMI, and entropy
make analysis                     # GMN, fGMN, and ring inflation
make                              # both groups
make GPU=0 CUDA_RHO=0             # CPU CUDA-Q build
make sim_tmi.exe GPU=1 FP64=0
make mipt_probed.exe GPU=1 FP64=0
make free_energy.exe GPU=1 CUDA_RHO=1 FP64=0
make test-core                    # no CUDA-Q/MOSEK/SCS required
make test-cusv                    # cuStateVec engine equivalence (needs a GPU)
make bench-cusv                   # CUDA-Q vs cuStateVec timing
make print-config
```

The CUDA helper compiler falls back from `nvcc` to `nvq++` when `nvcc` is not on
`PATH`. MOSEK is required for GMN/fGMN and SCS is required for ring inflation.

Build outputs are isolated in names such as `build/nvidia_fp32_rho1`, preventing
objects from incompatible configurations from being mixed.

## Measurement-record free energy

`free_energy.exe` implements the leading-Lyapunov/free-energy protocol of
Zabalo *et al.* (arXiv:2107.03393):

```bash
./free_energy.exe N P REALIZATIONS CIRC_TYPE INIT_STATE
```

`INIT_STATE=0` draws a random product state and `INIT_STATE=1` a random Haar
state. For parity-preserving circuit types 2, 3, and 4, each draw lies in one
randomly selected fixed-parity sector. The first `4N` timesteps equilibrate the
state; the next `24N` timesteps form the production record. Within each
measurement layer, Born probabilities are evaluated and projectors are applied
sequentially in site order, with the state renormalized after every outcome.

`REALIZATIONS` is the number of Monte Carlo trajectories. It is included in
both default filenames as `_reals_<count>` (for example, `_reals_10k`). The
executable writes a per-trajectory `*_samples.csv` for `c_eff` bootstrapping
and an aggregate `*_timeseries.csv` containing the cumulative record entropy
from `t=0` for the equilibration diagnostic. Half-chain von Neumann entropy is
recorded through `2N` by default; adjust
`MIPT_FREE_ENERGY_ENTROPY_MAX_T` or disable it with
`MIPT_FREE_ENERGY_HALF_ENTROPY=0`. GPU builds route half-chain cuts of seven
or more kept sites (starting at `N=14`) to cuSOLVER by default; override this
free-energy-specific threshold with
`MIPT_FREE_ENERGY_CUDA_ENTROPY_MIN_KEPT`.

The sample column `free_energy_density_tilde` is the production-window slope
divided by `N`, in natural-log units, before the spacetime anisotropy is
applied. In the extracted root-level `data_analysis.py`, use
`free_energy_ceff(..., alpha=...)` for the Fig. 1(b) double fit and
`free_energy_equilibration(...)` for Supplementary Fig. S1(a,b).

## Unified probe simulator

All one-, two-, and four-reference protocols are provided by one executable:

```bash
./mipt_probed.exe PROBES N REALIZATIONS CIRC_TYPE MODE ARGS...
```

| Mode | Arguments | Protocol |
|---:|---|---|
| 0 | `p t_min t_max` | Legacy time trajectory. One reference is attached at `t=0`; two/four are attached after `t0=2N`. |
| 1 | `p_min p_max p_res` | Fixed-`t=4N` critical scan. One probe uses the Gullans--Huse no-measurement encoding quench; two/four output `I2` or `I2,I3,I4`. |
| 2 | `t_min t_max t_res p_min p_max p_res` | Rectangular `p,t` scan of the joint reference entropy `S_Q`. |

The output filename is always generated from the arguments. Use the supplied
`data_analysis.py` functions
`probe_pc_collapse(...)` and `probe_entropy_map(...)` for the mode-1 and mode-2
outputs. Run `./mipt_probed.exe --help` for the probe geometry and environment
controls.

Modes 1 and 2 isolate each `p` point in a fresh subprocess by default. This
keeps a long scan from accumulating CUDA-Q/cuStateVec runtime state and writes
an atomic checkpoint after every completed point. Re-running the identical
command resumes an existing output; set `MIPT_PROBED_RESUME=0` to replace it or
`MIPT_PROBED_ISOLATE_P=0` to restore the single-process behavior. A failed point
is retried once in a fresh process with CPU prefetch disabled; control this with
`MIPT_PROBED_RETRIES`.

### One-probe mode-1 performance

For parity-preserving circuits (`circ_type` 2, 3, or 4), one-probe mode 1 uses
an exact symmetry encoding by default. The physical state has `N+1` qubits but
is confined to the even total-parity sector: the reference bit is equal to the
system parity. The runner therefore evolves only `N` dense qubits, initializes
the encoded Bell pair with `H` on the probe site, and obtains `S_Q` from the
even/odd system-parity weights. System gates, measurement locations and
outcomes, the `p=0` encoding interval, and the `t=4N` readout are unchanged.
Set `MIPT_PROBED_PARITY_ENCODING=0` for an explicit-reference A/B run.

Build the NVIDIA target in FP32 (the banner should say
`statevector_precision=fp32`):

```bash
make mipt_probed.exe GPU=1 CUDA_RHO=1 FP64=0
```

> **Correction.** Earlier revisions of this file recommended tuning
> `CUDAQ_FUSION_MAX_QUBITS` and `CUDAQ_FUSION_NUM_HOST_THREADS` with a
> `benchmark_probed_gpu.sh` sweep. CUDA-Q 0.14.2 has **no gate fusion at all** —
> neither string appears in the installed runtime, and there is no fusion pass in
> `cudaq-opt` — so those variables are no-ops. The script is gone as well. The
> cuStateVec engine below replaces that tuning path; benchmark it with
> `make bench-cusv`.

## cuStateVec circuit engine

`include/mipt/cusv/` drives cuStateVec directly instead of going through
CUDA-Q's gate-by-gate dispatch. cuStateVec ships inside CUDA-Q, so nothing extra
needs installing. It is on by default for `GPU=1` with the `nvidia` backend
(`CUSV=1`) and covers `mipt_probed.exe` and `free_energy.exe`.

What it changes, relative to one full state-vector pass per gate:

1. each two-site bond is applied as the single 4×4 matrix its 8-, 13-, or
   20-gate decomposition multiplies out to;
2. the periodic Jordan–Wigner CZ chain becomes one diagonal sign pass;
3. a measurement layer is sampled from its joint distribution in one pass and
   collapsed in one more, instead of ~2 passes per measured site;
4. the state vector is mutated in place, so no state is copied per timestep and
   only one 2^N buffer is live;
5. consecutive bonds with disjoint supports are fused into one larger matrix.

Points 1 and 5 do not change the circuit: a fused matrix is the product of the
gates it replaces, so its support is unchanged, and a block is a Kronecker
product of disjoint bond gates, so it cannot entangle sites the circuit did not.
Blocking only merges consecutive ops, so nothing is reordered past a
non-commuting neighbour. Point 3 is exact by the chain rule — sequential Born
sampling with renormalisation samples the joint distribution, and Z-basis
projectors on distinct sites commute.

```bash
make test-cusv        # fused matrices vs CUDA-Q ground truth, then whole circuits
make bench-cusv       # CUDA-Q vs cuStateVec on one trajectory
MIPT_CUSV=0 ./mipt_probed.exe ...   # A/B against the CUDA-Q path
```

| Variable | Default | Meaning |
|---|---:|---|
| `MIPT_CUSV` | 1 | 0 restores the CUDA-Q gate-by-gate path |
| `MIPT_CUSV_BLOCK_TARGETS` | 4 | qubits per fused block (4 = two bonds per pass) |
| `MIPT_CUSV_MEASURE_BITS` | 16 | max sites sampled in one joint draw before chunking |

`MIPT_CUSV_BLOCK_TARGETS` is hardware-dependent. A 2^k apply moves the same bytes
as a 2-target apply but costs 2^(k-1) flop/byte, so the best k tracks the GPU's
compute-to-bandwidth balance: ~23 flop/byte on a GTX 1650 favours k=4, ~68 on an
RTX 4080 Super should favour k=6. Re-run `make bench-cusv` on new hardware.

RFGS (`circ_type 3`) is not implemented in the engine and falls back to CUDA-Q.

### CUDA toolkit discovery

`custatevec.h` includes `<library_types.h>`, which comes from the CUDA toolkit
rather than from `~/.cudaq/include`, so `CUSV=1` compiles with an explicit
`-I$(CUDA_INC_DIR)`. The toolkit root is discovered from `CUDA_PATH`, then from
the real path of the selected `nvcc`, then from `/usr/local/cuda*`:

```bash
make print-config | grep CUDA_       # shows CUDA_HOME / CUDA_INC_DIR / CUDA_LIB_DIR
make CUDA_HOME=/opt/cuda-12.6 ...    # override when the toolkit is elsewhere
make CUSV=0 ...                      # build without the engine at all
```

If no toolkit is found, `CUSV=1` fails fast with an explanatory error rather
than a `'library_types.h' file not found` deep inside `cudaq-quake`.

Do not rely on the header being picked up implicitly: hosts with the distro
`nvidia-cuda-dev` package have a **CUDA 11** copy in `/usr/include`, which is on
clang's default search path, so a build can succeed there against the wrong
headers and fail on a host without that package.

The old Makefile referenced `tmi.cpp`, but that source was absent from the supplied
archive. The stale `tmi.exe` target was therefore removed; `sim_tmi.exe` remains the
maintained online TMI path.

## Host pause control

Every executable honours the same cooperative pause. Creating `PAUSE_MIPT` in
the process working directory stops the run at its next safe checkpoint —
between trajectories, batches, or records, with output already flushed — and
prints:

```text
PAUSED by host sentinel PAUSE_MIPT; remove it to resume.
```

Removing the file resumes it and prints the time lost:

```text
RESUMED after 37m 28s.
```

Paused time is excluded from progress rates and ETAs. `MIPT_PAUSE_FILE=path`
selects a different sentinel for any executable and `MIPT_PAUSE_FILE=0`
disables it; `MIPT_DIST_PAUSE_FILE` and `SIM_TMI_PAUSE_FILE` remain as
per-executable overrides for `dist_scaling.exe` and `sim_tmi.exe`. The shared
implementation is `include/mipt/util/pause.hpp`.

## Refactor invariants

- Circuit gate sequences, measurement sampling, boundary handling, and trace choice
  are unchanged.
- RFGS identifiers were normalized from the historical `FRGS` typo.
- qRPPU is now exposed consistently by the shared circuit factory, including
  `entropy.exe`.
- Generated dependency files (`-MMD -MP`) replace hand-maintained header lists in
  the Makefile, except for the CUDA-Q objects: nvq++ accepts those flags but
  writes no `.d` files, so those objects depend on every header explicitly.
- The cuStateVec engine reproduces the circuits exactly; see the equivalence
  tests in `tests/cusv_gate_tests.cpp` and `tests/cusv_equiv.cpp`.

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
