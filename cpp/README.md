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

### Resuming an interrupted run

`free_energy.exe`, `entropy.exe` and `dist_scaling.exe` all resume; the shared
CSV-reading and Welford-reconstruction machinery is in
`include/mipt/util/resume_csv.hpp`, with one small per-application header on
top. In every case the run's own output file *is* the checkpoint — there is no
side-car state — which is what lets a run interrupted by a binary that predates
the feature be picked up by one that has it. See "Resuming entropy.exe and
dist_scaling.exe" below for those two.

Re-issuing the identical command continues an interrupted run instead of
overwriting it. The two CSVs are themselves the checkpoint: the samples file
supplies the finished trajectories and, through the invertibility of the
splitmix64 seed derivation, the master seed, while the time series supplies the
per-timestep Welford accumulators (`m2` is recovered from the stored sample
standard deviation). A resumed run therefore continues the same seed stream and
reproduces what an uninterrupted run would have written, to a few ulp on the
standard-deviation columns. Runs interrupted by a binary that predates this
support resume fine, since nothing beyond the two CSVs is consulted.

A resume refuses rather than guesses when the files on disk describe a different
run: a mismatched `N`, `p`, `CIRC_TYPE`, `INIT_STATE` or `REALIZATIONS`, a
different CSV schema, a different half-chain entropy schedule, non-contiguous
trajectory indices, or only one of the two files present. An unterminated final
samples row — a kill during the flush — is dropped; every complete row is kept.
Because the samples row is appended before the time series is rewritten, an
interrupt between the two leaves the samples file one row ahead; that row is
kept and `completed_realizations` reports the aggregate's own count, so the two
files stay individually truthful. `MIPT_FREE_ENERGY_RESUME=0` discards the
existing files and starts over.

The sample column `free_energy_density_tilde` is the production-window slope
divided by `N`, in natural-log units, before the spacetime anisotropy is
applied. In the extracted root-level `data_analysis.py`, use
`free_energy_ceff(..., alpha=...)` for the Fig. 1(b) double fit and
`free_energy_equilibration(...)` for Supplementary Fig. S1(a,b).

## Resuming entropy.exe and dist_scaling.exe

Both write a single aggregated CSV and rewrite it as the run proceeds, so that
file is the whole checkpoint. Re-issuing the identical command — same arguments,
same output path — continues the averages rather than restarting them.
`MIPT_ENTROPY_RESUME=0` and `MIPT_DIST_RESUME=0` restore the old
overwrite-and-restart behaviour.

Neither executable seeds its circuits (`CircuitWorkspace1D::seed` is never
called, so trajectories come from `std::random_device`), so unlike
`free_energy.exe` there is no seed stream to continue and none is needed: the
trajectories are i.i.d. draws and a resumed run simply takes more of them. The
consequence is that a resumed run is statistically identical to an uninterrupted
one but not comparable trajectory by trajectory.

**`entropy.exe`.** The completed count is the `completed_realizations` column;
the accumulators come back through `m2 = stddev² · (count − 1)`. A block whose
von Neumann entropy was skipped is written as NaN and must return as an *empty*
accumulator rather than a zero sample, so the S1 schedule is read off the file's
own NaNs and then held against the current settings — resuming under a different
`MIPT_ENTROPY_S1`/`MIPT_ENTROPY_S1_MAX_LA` is refused, since it would leave the
S1 columns averaged over a different number of trajectories than the S2 ones.
Pre-2026-08-05 S2-only files resume as S2-only scans (`MIPT_ENTROPY_S1=0`) and
are refused otherwise. Because `stderr = stddev / sqrt(count)`, the two spread
columns cross-check the count for free; a file whose columns disagree is refused
as edited or concatenated.

**`dist_scaling.exe`.** Every bin field is a column or implied by one, so the
CSV is an invertible projection of the aggregation bins; `make test-dist` pins
that. The completed count is a bin's sample count divided by the records one
trajectory contributes to it — `embedding_count` for k=2, and
`min(MIPT_DIST_EMBEDDINGS_PER_GEOMETRY, embeddings)` for k=3 — and every bin
must agree. For k=3 the SDP schedule also has to resume in position:
`SdpSettings::selects` is a pure function of `bin.records` and
`bin.gmn.requested`, and both come back (the former as `tmi_samples`, since
every record feeds the TMI; the latter as `gmn_requested_count`).

> **GMN solves in flight at the interrupt are lost, and their loss is not
> neutral.** A record is counted in `gmn_requested_count` when it is submitted
> but contributes its value only when its batch is collected, so a kill leaves
> `requested > value.count + solver_failures`. The resume detects that from the
> queue's own invariant and rolls `requested` back, which returns those schedule
> slots so everything sampled from here on is unbiased. It cannot recover the
> lost values — the density matrices are long gone — and those records are
> exactly the ones the zero prefilter did *not* settle, so the surviving average
> over-weights `GMN = 0`. A resume prints the reclaimed count and warns when any
> geometry lost more than 2% of its requested records. This skew is a property
> of the interrupted file itself, not of resuming it: it is equally present in
> the partial CSV a killed run has always left behind. Folding the prefiltered
> zeros through the same queue as the solves would remove it at the source.

## Unified probe simulator

All one-, two-, and four-reference protocols are provided by one executable:

```bash
./mipt_probed.exe PROBES N REALIZATIONS CIRC_TYPE MODE ARGS...
```

| Mode | Arguments | Protocol |
|---:|---|---|
| 0 | `p t_min t_max` | Time trajectory. One reference is attached at absolute `t=0` and the Gullans--Huse encoder runs for `t_encode=2N` measurement-free timesteps; `t_min`/`t_max` are elapsed times after it, so the circuit is `2N+t_max` deep. Two/four references are attached after `t0=2N` of monitored evolution and keep absolute times. |
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
(`CUSV=1`) and covers **all six simulation executables**, through two entry
points:

| Entry point | Used by |
|---|---|
| `Circuit1D::try_cusv` (`circuit.hpp`) | `mipt.exe`, `sim_tmi.exe`, `entropy.exe`, `dist_scaling.exe` |
| `probed::CircuitWorkspace1D` (`probed.hpp`) | `mipt_probed.exe`, `free_energy.exe` |

Both start from a CUDA-Q `|0...0>` and mutate that buffer in place, so what they
return is an ordinary `cudaq::state` and no RDM, entropy, or TMI consumer had to
change.

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

## `sim_tmi.exe` entropy order — `SIM_TMI_ENTROPY_ORDER`

`I_3 = S_A + S_B + S_C + S_D - S_AB - S_AC - S_BC` can be built from either
entropy. `SIM_TMI_ENTROPY_ORDER=1` (**the default**) uses the von Neumann
entropy and is exactly the historical behaviour. `SIM_TMI_ENTROPY_ORDER=2` uses
the Rényi-2 entropy `S_2 = -log2 Tr(rho^2)`.

**Order 2 exists because it removes the half-cut eigensolve.** For a pure
state the purity is the squared Frobenius norm of `rho = M M^H`, so the device
path stops at the cuBLAS `herk` that the von Neumann path only uses as a
prologue. Measured on a GTX 1650, fp32, at the half-cut sizes TMI actually
uses:

| half cut | gather + herk | `Tr(rho^2)` | `Cheevd` | `Cgesvd` (the >512 path) |
|---|---|---|---|---|
| 1024 (N=20) | 2.3 ms | 0.25 ms | 51.9 ms | 230.6 ms |
| 4096 (N=24) | 110.8 ms | 2.4 ms | 1497.5 ms | 4636.6 ms |
| 8192 (N=26) | 887.4 ms | 8.3 ms | 10626.7 ms | — |

Instrumented on real trajectories at N=24 (Haar, `p=0.2`, `all_cycles=0`), the
three half cuts are **98.7%** of a TMI evaluation — 9.30 s against 0.12 s for
all four quarter RDMs. End to end, per realization:

| | order 1 | order 2 | |
|---|---|---|---|
| N=20 | 1.047 s | 0.072 s | **14.5×** |
| N=24 | 9.275 s | 1.010 s | **9.2×** |

The whole-app factor is smaller than the per-half-cut one because once the
eigensolve is gone the circuit and the Schmidt gather dominate again.

**At N=28 the memory matters more than the time.** The half cut is
16384×16384, and cuSOLVER's workspace *alone* is 6.16 GB for `Cheevd`,
`Xsyevd`, and `Cgesvd`, or 10.26 GB for `Xgesvdp`. With the state (2.15 GB),
the Schmidt matrix (2.15 GB) and the Gram (2.15 GB) on top, order 1 needs
~12.6 GB before CUDA-Q's own allocations. `retry_entropy_in_f64` would want a
4.3 GB fp64 Gram plus ~12 GB of fp64 workspace, so at that size the middle rung
of the fallback ladder cannot run at all. Order 2 allocates no solver
workspace: ~6.45 GB total.

Precision is *better*, not worse. The fp32 Gram plus an fp64 purity reduction
reproduces an fp64 `Zherk` reference to ≤5e-7 bits at dim 4096, on both flat
(Page) and steeply decaying spectra, against the ~6e-5 bits the von Neumann
path already accepts.

**The physics caveat is the real cost.** `I_3` at criticality is a universal
number that depends on the Rényi index, so order-2 output must never be pooled
with order-1 output. `p_c` and `nu` do not depend on it, so a crossing/collapse
analysis is unaffected. To keep the two archives from ever landing in one glob,
order 2 appends `_s2` to the output tag — it lands in both the directory and
the file stem, and order-1 paths are byte-identical to before:

```text
csv/tmi/haar/tmi_haar_n_24_real_...csv        # order 1 (unchanged)
csv/tmi/haar_s2/tmi_haar_s2_n_24_real_...csv  # order 2
```

The CSV schema stays `p,tmi` — these files reach millions of rows, the same
reason `gmn.exe` keeps its schema minimal — so the order is carried by the path
and by the banner's `entropy_order=` field. Unlike the other `MIPT_*`/`SIM_TMI_*`
knobs, an unparseable value **refuses** instead of falling back to the default:
silently spending a multi-hour run on the other observable is a worse failure
than an early exit.

Validation, since trajectories are not reproducible: a deep `p=0` Haar circuit
at N=16 reproduces the Page-state closed forms for both orders — order 1 gives
`-5.84743 ± 0.00028` against `-5.847228`, order 2 gives `-5.02235 ± 0.00038`
against `-5.022476` (300 realizations each). All four dispatch combinations
(device/host quarters × device/host half cuts) agree with the order-2 value
within 1.5σ. `make test-rdm-gram` pins the device purity reduction against a
purity taken straight from an fp64 reference RDM under both traces, and
`make test-core` pins the host primitives against closed forms.

### What was measured and rejected

- **Raising `MIPT_TMI_HEEVD_MAX_ROWS` above 512.** On physical N=24 states,
  `Cgesvd` gives 9.27 / 9.08 / 9.62 s per evaluation while `Cheevd` gives
  4.99 / 19.5 / 9.14 s — intermittently faster, intermittently 2× slower as the
  documented non-convergence fires and drags in the fp64 retry. The 512 cap is
  correct.
- **The 64-bit API.** `cusolverDnXsyevd` matches legacy `Cheevd` to within noise
  at every size tried and has byte-identical workspace.
- **`cusolverDnXgesvdp`.** 1.66× more workspace.

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
