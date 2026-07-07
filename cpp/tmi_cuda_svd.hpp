#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Computes the von Neumann entropy of a subsystem of a pure CUDA device
 * statevector by building the Schmidt coefficient matrix on the GPU and
 * computing singular values with cuSOLVER.
 *
 * host_kept_modes gives the subsystem modes in the same little-endian order
 * used by sim_tmi.cpp.  If fermion_trace is nonzero, the same fermionic sign
 * convention used by sim_tmi.cpp's host path is applied when constructing the
 * coefficient matrix.
 *
 * Return value: 0 on success; nonzero CUDA/cuSOLVER-style error code on
 * failure.  The caller should fall back to the host LAPACK SVD path on failure.
 */
int mipt_cuda_entropy_svd_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *entropy_out);

int mipt_cuda_entropy_svd_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *entropy_out);

#ifdef __cplusplus
}
#endif
