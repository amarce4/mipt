#include "mipt/cuda/free_energy_measure.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace
{
constexpr int THREADS = 256;
thread_local char last_error[512] = "";

template <typename Real>
struct Cx
{
    Real re;
    Real im;
};

__device__ __forceinline__ double atomic_add_f64(double *address, double value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto *address_as_ull = reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    }
    while (assumed != old);
    return __longlong_as_double(old);
#endif
}

template <typename Real>
__global__ void branch_norms_kernel(
    const Cx<Real> *__restrict__ state,
    std::uint64_t dimension,
    std::uint64_t measured_mask,
    double *__restrict__ branch_norms)
{
    double local_total = 0.0;
    double local_one = 0.0;
    const std::uint64_t first =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t index = first; index < dimension; index += stride)
    {
        const Cx<Real> value = state[index];
        const double re = static_cast<double>(value.re);
        const double im = static_cast<double>(value.im);
        const double weight = re * re + im * im;
        local_total += weight;
        if ((index & measured_mask) != 0)
            local_one += weight;
    }

    __shared__ double total_scratch[THREADS];
    __shared__ double one_scratch[THREADS];
    total_scratch[threadIdx.x] = local_total;
    one_scratch[threadIdx.x] = local_one;
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            total_scratch[threadIdx.x] +=
                total_scratch[threadIdx.x + offset];
            one_scratch[threadIdx.x] += one_scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomic_add_f64(branch_norms, total_scratch[0]);
        atomic_add_f64(branch_norms + 1, one_scratch[0]);
    }
}

template <typename Real>
__global__ void collapse_kernel(
    Cx<Real> *__restrict__ state,
    std::uint64_t dimension,
    std::uint64_t measured_mask,
    int outcome,
    double inverse_norm)
{
    const std::uint64_t first =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    const Real scale = static_cast<Real>(inverse_norm);
    for (std::uint64_t index = first; index < dimension; index += stride)
    {
        const int bit = (index & measured_mask) != 0 ? 1 : 0;
        if (bit == outcome)
        {
            state[index].re *= scale;
            state[index].im *= scale;
        }
        else
        {
            state[index].re = Real{0};
            state[index].im = Real{0};
        }
    }
}

void set_cuda_error(const char *operation, cudaError_t status)
{
    std::snprintf(
        last_error,
        sizeof(last_error),
        "%s failed: %s",
        operation,
        cudaGetErrorString(status));
}

template <typename Real>
int measure_collapse(
    void *device_state_vector,
    int n_qubits,
    int measured_qubit,
    double uniform_01,
    int *outcome,
    double *realized_probability)
{
    last_error[0] = '\0';
    if (device_state_vector == nullptr || outcome == nullptr ||
        realized_probability == nullptr || n_qubits < 1 || n_qubits >= 63 ||
        measured_qubit < 0 || measured_qubit >= n_qubits ||
        !(uniform_01 >= 0.0 && uniform_01 < 1.0))
    {
        std::snprintf(last_error, sizeof(last_error),
                      "Invalid sequential-measurement argument.");
        return -1;
    }

    thread_local double *device_branch_norms = nullptr;
    if (device_branch_norms == nullptr)
    {
        const cudaError_t status =
            cudaMalloc(&device_branch_norms, 2 * sizeof(double));
        if (status != cudaSuccess)
        {
            set_cuda_error("cudaMalloc(branch norms)", status);
            return static_cast<int>(status);
        }
    }

    cudaError_t status =
        cudaMemset(device_branch_norms, 0, 2 * sizeof(double));
    if (status != cudaSuccess)
    {
        set_cuda_error("cudaMemset(branch norms)", status);
        return static_cast<int>(status);
    }

    const std::uint64_t dimension = std::uint64_t{1} << n_qubits;
    const std::uint64_t measured_mask = std::uint64_t{1} << measured_qubit;
    const std::uint64_t required_blocks =
        (dimension + static_cast<std::uint64_t>(THREADS) - 1u) /
        static_cast<std::uint64_t>(THREADS);
    const int blocks = static_cast<int>(
        std::min<std::uint64_t>(required_blocks, 65535u));

    branch_norms_kernel<Real><<<blocks, THREADS>>>(
        reinterpret_cast<const Cx<Real> *>(device_state_vector),
        dimension,
        measured_mask,
        device_branch_norms);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        set_cuda_error("branch-norm kernel launch", status);
        return static_cast<int>(status);
    }

    double branch_norms[2] = {0.0, 0.0};
    status = cudaMemcpy(
        branch_norms,
        device_branch_norms,
        2 * sizeof(double),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        set_cuda_error("cudaMemcpy(branch norms)", status);
        return static_cast<int>(status);
    }

    const double total_norm = branch_norms[0];
    const double branch_one_norm = branch_norms[1];
    if (!(total_norm > 0.0) || !std::isfinite(total_norm) ||
        !std::isfinite(branch_one_norm) || branch_one_norm < 0.0 ||
        branch_one_norm > total_norm * (1.0 + 1.0e-10))
    {
        std::snprintf(
            last_error,
            sizeof(last_error),
            "Invalid branch norms: total=%.17g, one=%.17g",
            total_norm,
            branch_one_norm);
        return -2;
    }
    const double probability_one = std::max(
        0.0, std::min(1.0, branch_one_norm / total_norm));
    *outcome = uniform_01 < probability_one ? 1 : 0;
    *realized_probability = *outcome ? probability_one : 1.0 - probability_one;
    const double realized_branch_norm =
        *outcome ? branch_one_norm : total_norm - branch_one_norm;
    if (!(*realized_probability > 0.0) ||
        !std::isfinite(*realized_probability))
    {
        std::snprintf(
            last_error,
            sizeof(last_error),
            "Sampled an outcome with invalid conditional probability %.17g",
            *realized_probability);
        return -3;
    }
    if (!(realized_branch_norm > 0.0) ||
        !std::isfinite(realized_branch_norm))
    {
        std::snprintf(
            last_error,
            sizeof(last_error),
            "Sampled an outcome with invalid branch norm %.17g",
            realized_branch_norm);
        return -4;
    }

    collapse_kernel<Real><<<blocks, THREADS>>>(
        reinterpret_cast<Cx<Real> *>(device_state_vector),
        dimension,
        measured_mask,
        *outcome,
        1.0 / std::sqrt(realized_branch_norm));
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        set_cuda_error("collapse kernel launch", status);
        return static_cast<int>(status);
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        set_cuda_error("collapse kernel synchronization", status);
        return static_cast<int>(status);
    }
    return 0;
}
} // namespace

extern "C" int mipt_cuda_measure_collapse_f32(
    void *device_state_vector,
    int n_qubits,
    int measured_qubit,
    double uniform_01,
    int *outcome,
    double *realized_probability)
{
    return measure_collapse<float>(
        device_state_vector,
        n_qubits,
        measured_qubit,
        uniform_01,
        outcome,
        realized_probability);
}

extern "C" int mipt_cuda_measure_collapse_f64(
    void *device_state_vector,
    int n_qubits,
    int measured_qubit,
    double uniform_01,
    int *outcome,
    double *realized_probability)
{
    return measure_collapse<double>(
        device_state_vector,
        n_qubits,
        measured_qubit,
        uniform_01,
        outcome,
        realized_probability);
}

extern "C" const char *mipt_cuda_measure_collapse_last_error()
{
    return last_error;
}
