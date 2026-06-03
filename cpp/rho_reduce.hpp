#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Computes complete complex 8x8 reduced density matrices for arbitrary
 * three-qubit subsystems of a contiguous CUDA-Q device statevector.
 *
 * host_subsystems is flattened as:
 *   {q[0][0], q[0][1], q[0][2], q[1][0], ...}
 * For each subsystem s, reduced-basis bit k corresponds to host_subsystems[3*s+k].
 *
 * Output layout for each subsystem s:
 *   base = s * (2 * 8 * 8)
 *   host_rho_ri[base + 2 * (row * 8 + col) + 0] = Re(rho_s[row,col])
 *   host_rho_ri[base + 2 * (row * 8 + col) + 1] = Im(rho_s[row,col])
 *
 * Return value: 0 on success; nonzero CUDA error code on failure.
 */
int mipt_cuda_rho3_subsystems_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major);

int mipt_cuda_rho3_subsystems_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major);

#ifdef __cplusplus
}
#endif
