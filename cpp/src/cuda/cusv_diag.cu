#include "mipt/cuda/cusv_diag.hpp"

#include <cstdio>
#include <cuda_runtime.h>

namespace
{
constexpr int THREADS = 256;

char last_error[256] = {0};

void set_cuda_error(const char *operation, cudaError_t status)
{
    std::snprintf(last_error, sizeof(last_error), "%s failed: %s", operation,
                  cudaGetErrorString(status));
}

template <typename Real>
struct Cx
{
    Real re;
    Real im;
};

__device__ __forceinline__ bool odd_popcount64(std::uint64_t x)
{
    return (__popcll(static_cast<unsigned long long>(x)) & 1) != 0;
}

// amplitude *= -1 where bit(control) is set and popcount(index & mask) is odd.
template <typename Real>
__global__ void jw_string_kernel(Cx<Real> *__restrict__ state, std::uint64_t dimension,
                                 std::uint64_t control_mask, std::uint64_t string_mask)
{
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; index < dimension; index += stride)
    {
        if ((index & control_mask) != 0 && odd_popcount64(index & string_mask))
        {
            state[index].re = -state[index].re;
            state[index].im = -state[index].im;
        }
    }
}

template <typename Real>
int jw_string(void *device_state_vector, int n_qubits, int control_qubit, std::uint64_t string_mask)
{
    last_error[0] = '\0';
    if (device_state_vector == nullptr || n_qubits < 1 || n_qubits >= 63 || control_qubit < 0 ||
        control_qubit >= n_qubits)
    {
        std::snprintf(last_error, sizeof(last_error), "Invalid JW-string argument.");
        return -1;
    }
    if (string_mask == 0)
    {
        return 0;
    }

    const std::uint64_t dimension = std::uint64_t{1} << n_qubits;
    const std::uint64_t required = (dimension + THREADS - 1u) / THREADS;
    const int blocks = static_cast<int>(required < 65535u ? required : 65535u);

    jw_string_kernel<Real><<<blocks, THREADS>>>(reinterpret_cast<Cx<Real> *>(device_state_vector),
                                                dimension, std::uint64_t{1} << control_qubit,
                                                string_mask);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        set_cuda_error("JW-string kernel launch", status);
        return static_cast<int>(status);
    }
    return 0;
}

} // namespace

extern "C" int mipt_cuda_jw_string_f32(void *device_state_vector, int n_qubits, int control_qubit,
                                       std::uint64_t string_mask)
{
    return jw_string<float>(device_state_vector, n_qubits, control_qubit, string_mask);
}

extern "C" int mipt_cuda_jw_string_f64(void *device_state_vector, int n_qubits, int control_qubit,
                                       std::uint64_t string_mask)
{
    return jw_string<double>(device_state_vector, n_qubits, control_qubit, string_mask);
}

extern "C" const char *mipt_cuda_cusv_diag_last_error()
{
    return last_error;
}
