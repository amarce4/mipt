# Migration from the supplied flat tree

## File mapping

| Previous file | New location |
|---|---|
| `mipt.cpp` executable code | `apps/mipt.cpp` |
| reusable MMS/Haar code formerly in `mipt.cpp` | `include/mipt/circuits/mms.hpp`, `haar.hpp` |
| `mipt_fermion.cpp` | `include/mipt/circuits/fermion.hpp` |
| duplicated circuit dispatch in executables | `include/mipt/circuit.hpp` |
| `mipt_backend.hpp` | `include/mipt/backend.hpp` |
| `mipt_native_mps.hpp` | `include/mipt/native_mps.hpp` |
| `density_io.hpp` | `include/mipt/density.hpp` |
| monolithic `sim_tmi.cpp` | `apps/sim_tmi.cpp`, `include/mipt/tmi/*.hpp` |
| monolithic `entropy.cpp` | `apps/entropy.cpp`, `include/mipt/entropy.hpp` |
| CUDA headers/sources | `include/mipt/cuda/`, `src/cuda/` |
| GMN/fGMN headers/sources | `include/mipt/analysis/`, `src/analysis/`, `apps/` |
| `ring_inflation.cpp` | `apps/ring_inflation.cpp`, `include/mipt/analysis/ring_inflation.hpp`, `src/analysis/ring_inflation.cpp` |

## Renamed symbols

- Historical `FRGS` spellings are now `RFGS`/`Rfgs`.
- `MiptCircuitWorkStats` is now `CircuitWorkStats`.
- Backend helpers are under `mipt::backend`.
- Density-file types are under `mipt::io`.
- TMI implementation types are under `mipt::tmi`.

## Build changes

- Header search root is `include/`.
- Object files are isolated by backend, precision, and CUDA-RDM setting, for
  example `build/nvidia_fp32_rho1/`.
- Generated dependency files replace manually maintained dependency lists.
- `make simulation`, `make analysis`, and `make test-core` are available.
- The stale `tmi.exe` target was removed because the archive contained no
  `tmi.cpp`; `sim_tmi.exe` is the maintained online TMI executable.

Command-line executable names and the binary density-file format are retained.
