#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Computes the complete complex 16x16 reduced density matrix of the final
 * four qubits/modes of a contiguous CUDA-Q device statevector. The local
 * basis uses the final four qubits in ascending index order, with the first
 * retained qubit as the least-significant local bit.
 *
 * Output is a full row-major matrix with interleaved real/imaginary values:
 *   host_rho_ri[2 * (16 * row + col) + 0] = Re(rho[row,col])
 *   host_rho_ri[2 * (16 * row + col) + 1] = Im(rho[row,col])
 *
 * Return value: 0 on success; a nonzero CUDA/cuBLAS status on failure.
 */
int mipt_cuda_rho4_tail_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_ri);

int mipt_cuda_rho4_tail_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_ri);

/*
 * Fermionic partial trace for the same four final modes. Moving the retained
 * modes ahead of the environment contributes the Jordan-Wigner reordering
 * sign (-1)^(N_retained N_environment) to each amplitude.
 */
int mipt_cuda_rho4_tail_complex_fermion_f64(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_ri);

int mipt_cuda_rho4_tail_complex_fermion_f32(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_ri);

#ifdef __cplusplus
}
#endif
