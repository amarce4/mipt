#pragma once

// Diagonal sign passes for the cuStateVec circuit engine.
//
// mipt_cuda_jw_string_* implements optimization #2: the periodic-boundary
// Jordan-Wigner chain prod_k CZ(k, control) is diagonal, so the whole chain --
// n-2 separate CZ gates in the CUDA-Q kernel, each a full state-vector pass --
// collapses to a single pass that flips the sign of every amplitude whose
// `control` bit is set and whose masked bit count is odd.

#include <cstdint>

extern "C" {

int mipt_cuda_jw_string_f32(void *device_state_vector, int n_qubits, int control_qubit,
                            std::uint64_t string_mask);

int mipt_cuda_jw_string_f64(void *device_state_vector, int n_qubits, int control_qubit,
                            std::uint64_t string_mask);

const char *mipt_cuda_cusv_diag_last_error();

} // extern "C"
