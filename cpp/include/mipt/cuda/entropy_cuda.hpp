#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Compute translation-averaged entanglement entropies for every contiguous
// cyclic subsystem size L_A=min_kept,...,max_kept directly from a dense CUDA
// statevector. The max_kept RDM is formed once per cyclic translation; smaller
// RDMs are obtained recursively by tracing out the trailing retained mode.
//
// Both host_s1_means and host_s2_means must have max_kept-min_kept+1 elements
// and hold natural-log entropies on return.
//
//   host_s2_means  second Renyi, S2 = -ln Tr(rho^2), from the Frobenius norm.
//   host_s1_means  von Neumann, S1 = -Tr(rho ln rho), from the full spectrum.
//                  Pass NULL to skip it; the eigensolve is the dominant cost of
//                  this routine and grows as (2^L_A)^3.
//
// max_s1_kept caps the subsystem size the spectrum is computed for: sizes above
// it get NaN in host_s1_means rather than an eigensolve. Pass max_kept for no
// cap. It is ignored when host_s1_means is NULL.
//
// Returns 0 on success; nonzero values indicate invalid input, CUDA runtime,
// cuBLAS, cuSOLVER, allocation, or numerical failures.
int mipt_cuda_cyclic_entropies_f32(const void *device_state,
                                   int n,
                                   int min_kept,
                                   int max_kept,
                                   int fermionic_trace,
                                   int max_s1_kept,
                                   double *host_s1_means,
                                   double *host_s2_means);

int mipt_cuda_cyclic_entropies_f64(const void *device_state,
                                   int n,
                                   int min_kept,
                                   int max_kept,
                                   int fermionic_trace,
                                   int max_s1_kept,
                                   double *host_s1_means,
                                   double *host_s2_means);

// Human-readable description of the most recent failure on the calling thread.
const char *mipt_cuda_cyclic_entropies_last_error(void);

#ifdef __cplusplus
}
#endif
