# Architecture

## Dependency direction

```text
apps/
  -> mipt/circuit.hpp, mipt/entropy.hpp, mipt/tmi/runner.hpp
  -> mipt/analysis/*.hpp

mipt/circuit.hpp
  -> mipt/circuits/*.hpp
  -> mipt/backend.hpp, mipt/types.hpp

mipt/entropy.hpp, mipt/tmi/*.hpp, mipt/rdm.hpp
  -> mipt/circuit.hpp
  -> mipt/cuda/*.hpp (only when CUDA reductions are enabled)

src/cuda/, src/analysis/
  -> public headers under include/mipt/
```

Executable entry points are thin and do not own circuit kernels or include another
implementation file. Simulation CLIs contain argument parsing and orchestration;
external solver implementations remain under `src/analysis/`.

## Circuit model

`mipt::Circuit1D` is the abstract simulation interface. A concrete circuit class
owns its random-number generator and reusable layer storage, builds one sampled
trajectory, and invokes the corresponding CUDA-Q kernel. `CircuitWorkspace1D`
retains the selected concrete object between trajectories to avoid repeated
allocation.

The factory `make_circuit_1d(CircuitType)` is the only circuit-dispatch switch.
Circuit metadata—display name, output tag, trace convention, and even-site
constraint—is defined once in `mipt/types.hpp`.

To add a 1D circuit:

1. Add its layer representation, sampler, and CUDA-Q kernel under
   `include/mipt/circuits/`.
2. Add one `CircuitType`/`CircuitInfo` entry.
3. Implement a `Circuit1D` subclass and register it in `make_circuit_1d`.

The simulation, TMI, and entropy executables then receive it through the same
interface. Observable code should never duplicate a circuit sampler.

## Observable and storage modules

- `mipt/rdm.hpp`: reduced density matrices and CUDA-Q state extraction.
- `mipt/entropy.hpp`: second-Renyi entropy workflow.
- `mipt/tmi/`: TMI linear algebra, entropy/RDM computation, and run control.
- `mipt/density.hpp`: versioned binary RDM format with validation.
- `mipt/native_mps.hpp`: optional native CPU TEBD/MPS implementation.
- `mipt/env.hpp`: common environment-variable parsing.

## External solver boundary

GMN/fGMN and ring inflation remain separate analysis programs because they have
independent MOSEK/Fusion and SCS dependencies. Their public solver entry points
live under `include/mipt/analysis/`; implementation files live under
`src/analysis/`.

## Scientific invariants

The refactor does not intentionally alter sampled gates, measurement Bernoulli
trials, periodic-boundary constructions, reduced-density conventions, or output
record ordering. The only functional consistency correction is that qRPPU is now
available to `entropy.exe` through the same factory used by the other 1D tools.
