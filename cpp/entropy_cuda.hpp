#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Compute translation-averaged second-Renyi entropies for every contiguous
// cyclic subsystem size L_A=min_kept,...,max_kept directly from a dense CUDA
// statevector. The max_kept RDM is formed once per cyclic translation; smaller
// RDMs are obtained recursively by tracing out the trailing retained mode.
// host_means must have max_kept-min_kept+1 elements.
//
// Returns 0 on success; nonzero values indicate invalid input, CUDA runtime,
// cuBLAS, allocation, or numerical failures.
int mipt_cuda_cyclic_s2_f32(const void *device_state,
                            int n,
                            int min_kept,
                            int max_kept,
                            int fermionic_trace,
                            double *host_means);

int mipt_cuda_cyclic_s2_f64(const void *device_state,
                            int n,
                            int min_kept,
                            int max_kept,
                            int fermionic_trace,
                            double *host_means);

// Human-readable description of the most recent failure on the calling thread.
const char *mipt_cuda_cyclic_s2_last_error(void);

#ifdef __cplusplus
}
#endif
