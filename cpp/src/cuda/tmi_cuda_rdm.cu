#include "mipt/cuda/tmi_cuda_rdm.hpp"
#include "mipt/cuda/schmidt_gram.cuh"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Reduced density matrix of an arbitrary retained-mode subset, exported to the
// host as row-major interleaved fp64.
//
// The previous kernel launched one block per output element (kept_dim^2 blocks)
// and had each block stream the entire environment, so it read
//     2 * kept_dim^2 * env_dim = 2 * 2^(n + kept)
// amplitudes where the problem needs 2^n.  For a kept=6 quarter block at N=24
// that is 2^31 loads -- 128x redundant, about 17 GB of traffic per call, which
// is why SIM_TMI_DEVICE_SMALL_RDM defaulted to off and sim_tmi.exe copied the
// whole state vector to the host instead.
//
// Now the state is gathered once into the Schmidt matrix and cuBLAS herk forms
// rho = M M^H, which is the same arithmetic but tiled through shared memory and
// read from global memory exactly once.
//
// The herk accumulates at the precision of the state vector rather than in
// double as the old kernel did.  tests/rdm_gram_tests checks the resulting RDM
// against a double-accumulated reference; the measured difference is far below
// the fp32 representation error of the state itself.

namespace
{
using namespace mipt_gram;

enum
{
    MAX_DEVICE_RDM_KEPT = 12,
    CONVERT_THREADS = 256
};

int validate_modes(int n, const int *modes, int kept_count)
{
    if (n <= 0 || n >= 63 || modes == NULL || kept_count <= 0 || kept_count > n ||
        kept_count > MAX_DEVICE_RDM_KEPT)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    unsigned char seen[64] = {0};
    for (int i = 0; i < kept_count; ++i)
    {
        const int q = modes[i];
        if (q < 0 || q >= n || seen[q] != 0)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        seen[q] = 1;
    }
    return 0;
}

template <typename Real, typename ComplexT>
int launch_rdm_subsystem(const void *device_state_vector,
                         int n_qubits,
                         const int *host_kept_modes,
                         int kept_count,
                         int fermion_trace,
                         double *host_rho_ri)
{
    if (device_state_vector == NULL || host_rho_ri == NULL)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    int status = validate_modes(n_qubits, host_kept_modes, kept_count);
    if (status != 0)
    {
        return status;
    }
    if (kept_count >= n_qubits)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    static thread_local GramWorkspace<ComplexT> ws;
    static thread_local double *d_rho_ri = NULL;
    static thread_local size_t rho_ri_capacity = 0;

    size_t rows = 0;
    size_t cols = 0;
    status = build_gram<Real, ComplexT>(device_state_vector, n_qubits,
                                        host_kept_modes, kept_count,
                                        fermion_trace, ws, rows, cols);
    if (status != 0)
    {
        return status;
    }

    const size_t matrix_elements = rows * rows;
    cudaError_t cuda_status = cudaSuccess;
    if (thread_buffer(2u * matrix_elements, d_rho_ri, rho_ri_capacity,
                      cuda_status) == NULL)
    {
        return static_cast<int>(cuda_status);
    }

    const size_t wanted = (matrix_elements + CONVERT_THREADS - 1) / CONVERT_THREADS;
    const int blocks = static_cast<int>(wanted < 65535 ? wanted : size_t{65535});
    gram_to_row_major_ri_kernel<ComplexT><<<blocks, CONVERT_THREADS>>>(
        ws.gram, static_cast<int>(rows), d_rho_ri);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return static_cast<int>(cuda_status);
    }

    cuda_status = cudaMemcpy(host_rho_ri, d_rho_ri,
                             2u * matrix_elements * sizeof(double),
                             cudaMemcpyDeviceToHost);
    return static_cast<int>(cuda_status);
}
} // namespace

extern "C" int mipt_cuda_rdm_subsystem_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *host_rho_ri_row_major)
{
    return launch_rdm_subsystem<double, cuDoubleComplex>(
        device_state_vector, n_qubits, host_kept_modes, kept_count,
        fermion_trace, host_rho_ri_row_major);
}

extern "C" int mipt_cuda_rdm_subsystem_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *host_rho_ri_row_major)
{
    return launch_rdm_subsystem<float, cuComplex>(
        device_state_vector, n_qubits, host_kept_modes, kept_count,
        fermion_trace, host_rho_ri_row_major);
}
