#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Computes complete complex 4x4 reduced density matrices for arbitrary
 * two-qubit/two-mode subsystems of a contiguous CUDA-Q device statevector.
 *
 * host_subsystems is flattened as:
 *   {q[0][0], q[0][1], q[1][0], q[1][1], ...}
 * Reduced-basis bit k corresponds to host_subsystems[2*s+k].
 *
 * Output layout for each subsystem s:
 *   base = s * (2 * 4 * 4)
 *   host_rho_ri[base + 2 * (row * 4 + col) + 0] = Re(rho_s[row,col])
 *   host_rho_ri[base + 2 * (row * 4 + col) + 1] = Im(rho_s[row,col])
 *
 * Return value: 0 on success; nonzero CUDA error code on failure.
 */
int mipt_cuda_rho2_subsystems_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major);

int mipt_cuda_rho2_subsystems_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major);

/* Same contract, with the fermionic partial-trace sign convention. */
int mipt_cuda_rho2_subsystems_complex_fermion_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major);

int mipt_cuda_rho2_subsystems_complex_fermion_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major);

/*
 * Computes the ordinary and fermionic partial traces in one statevector pass.
 * The two output buffers use the same layout described above.
 */
int mipt_cuda_rho2_subsystems_complex_both_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major,
    double *host_fermion_rho_ri_row_major);

int mipt_cuda_rho2_subsystems_complex_both_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major,
    double *host_fermion_rho_ri_row_major);

#ifdef __cplusplus
}
#endif
