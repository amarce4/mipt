#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Reduces a contiguous CUDA-Q state-vector tensor to the real 8x8 reduced
// density matrix of the last three qubits. The input pointer must be a device
// pointer when these functions are called.
//
// Return value: 0 on success; nonzero CUDA error code on failure.
int mipt_cuda_last3_reduced_density_real_f64(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_real_row_major);

int mipt_cuda_last3_reduced_density_real_f32(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_real_row_major);

#ifdef __cplusplus
}
#endif
