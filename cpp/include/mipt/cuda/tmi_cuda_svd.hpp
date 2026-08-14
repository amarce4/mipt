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

/*
 * Renyi-2 entropy S_2 = -log2 Tr(rho^2) of the same subsystem, from the same
 * Schmidt gather.
 *
 * The purity is the squared Frobenius norm of rho = M M^H, so this stops after
 * the cuBLAS herk that mipt_cuda_entropy_svd_* only uses as a prologue: no
 * eigensolver, and none of its workspace.  At a 16384-wide half cut (N=28) that
 * workspace alone is 6.2 GB, so this is the difference between fitting on a
 * 16 GB card and not.  Same arguments, same return convention.
 */
int mipt_cuda_renyi2_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *entropy_out);

int mipt_cuda_renyi2_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *entropy_out);

#ifdef __cplusplus
}
#endif
