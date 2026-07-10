#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Computes a complete reduced density matrix for one arbitrary retained-mode
 * subsystem of a contiguous CUDA device statevector.  This is intended for the
 * small A/B/C/D blocks in sim_tmi.exe so the full statevector does not have to
 * be copied back to the host merely to form those small RDMs.
 *
 * host_kept_modes gives the retained modes in little-endian reduced-basis order.
 * Output is row-major real/imag pairs:
 *   rho[2 * (row * D + col) + 0] = Re rho[row,col]
 *   rho[2 * (row * D + col) + 1] = Im rho[row,col]
 *
 * Return value: 0 on success; nonzero CUDA error/status on failure.
 */
int mipt_cuda_rdm_subsystem_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *host_rho_ri_row_major);

int mipt_cuda_rdm_subsystem_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *host_rho_ri_row_major);

#ifdef __cplusplus
}
#endif
