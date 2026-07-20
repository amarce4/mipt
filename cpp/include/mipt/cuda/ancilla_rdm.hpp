#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Computes the complete complex 2x2 reduced density matrix of one retained
 * qubit directly from a contiguous CUDA-Q device statevector.
 *
 * Output layout:
 *   host_rho_ri[0] = rho00
 *   host_rho_ri[1] = rho11
 *   host_rho_ri[2] = Re(rho01)
 *   host_rho_ri[3] = Im(rho01)
 *
 * Return value: 0 on success; nonzero CUDA error code on failure.
 */
int mipt_cuda_rho1_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit,
    double *host_rho_ri);

int mipt_cuda_rho1_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit,
    double *host_rho_ri);

/*
 * Fermionic partial-trace convention. The diagonal entries are unchanged;
 * the off-diagonal entry includes the Jordan-Wigner reordering sign incurred
 * by moving the retained mode ahead of lower-index environment modes.
 */
int mipt_cuda_rho1_complex_fermion_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit,
    double *host_rho_ri);

int mipt_cuda_rho1_complex_fermion_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit,
    double *host_rho_ri);

/*
 * Computes the complete complex 4x4 reduced density matrix of two retained
 * qubits/modes. The local basis order is |q0 q1> with q0 as the least
 * significant local bit: |00>, |10>, |01>, |11>.
 *
 * Output layout is a full row-major matrix with interleaved real/imaginary
 * values:
 *   host_rho_ri[2 * (4 * row + col) + 0] = Re(rho[row,col])
 *   host_rho_ri[2 * (4 * row + col) + 1] = Im(rho[row,col])
 */
int mipt_cuda_rho2_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *host_rho_ri);

int mipt_cuda_rho2_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *host_rho_ri);

int mipt_cuda_rho2_complex_fermion_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *host_rho_ri);

int mipt_cuda_rho2_complex_fermion_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *host_rho_ri);

/*
 * Computes the complete complex 16x16 reduced density matrix of four retained
 * qubits/modes. Local bit i corresponds to retained_qubit_i. Output is a full
 * row-major matrix with interleaved real/imaginary values (512 doubles).
 */
int mipt_cuda_rho4_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    int retained_qubit2,
    int retained_qubit3,
    double *host_rho_ri);

int mipt_cuda_rho4_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    int retained_qubit2,
    int retained_qubit3,
    double *host_rho_ri);

int mipt_cuda_rho4_complex_fermion_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    int retained_qubit2,
    int retained_qubit3,
    double *host_rho_ri);

int mipt_cuda_rho4_complex_fermion_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    int retained_qubit2,
    int retained_qubit3,
    double *host_rho_ri);

#ifdef __cplusplus
}
#endif
