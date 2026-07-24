#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Sequentially sample one computational-basis measurement from a dense CUDA
// device statevector and apply the normalized projector in place. The branch
// probability is divided by the actual input norm, so collapse also removes
// small normalization drift from finite-precision unitary evolution.
// uniform_01 must lie in [0,1).  realized_probability is the conditional Born
// probability of the sampled outcome before collapse.
int mipt_cuda_measure_collapse_f32(
    void *device_state_vector,
    int n_qubits,
    int measured_qubit,
    double uniform_01,
    int *outcome,
    double *realized_probability);

int mipt_cuda_measure_collapse_f64(
    void *device_state_vector,
    int n_qubits,
    int measured_qubit,
    double uniform_01,
    int *outcome,
    double *realized_probability);

const char *mipt_cuda_measure_collapse_last_error();

#ifdef __cplusplus
}
#endif
