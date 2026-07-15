#include "mipt/cuda/ancilla_rdm.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace
{
constexpr int THREADS = 256;
constexpr int OUTPUT_VALUES = 4;

template <typename Real>
struct Cx
{
    Real re;
    Real im;
};

// Native atomicAdd(double*) requires compute capability >= 6.0. nvcc may
// otherwise compile this translation unit for an older default architecture,
// even when the runtime GPU is newer. Use the standard atomicCAS fallback so
// both the FP32 and FP64 entry points compile for every supported target.
__device__ __forceinline__ double atomic_add_f64(
    double *address,
    double value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto *address_as_ull =
        reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed = 0;
    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(
                value + __longlong_as_double(assumed)));
    }
    while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__device__ __forceinline__ bool odd_popcount64(std::uint64_t value)
{
    return (__popcll(static_cast<unsigned long long>(value)) & 1) != 0;
}

__device__ __forceinline__ std::uint64_t embed_environment_bits(
    std::uint64_t environment,
    int retained_qubit)
{
    const std::uint64_t lower_mask = retained_qubit == 0
        ? std::uint64_t{0}
        : ((std::uint64_t{1} << retained_qubit) - 1u);
    const std::uint64_t lower = environment & lower_mask;
    const std::uint64_t upper = environment >> retained_qubit;
    return lower | (upper << (retained_qubit + 1));
}

template <typename Real>
__global__ void rho1_complex_kernel(
    const Cx<Real> *__restrict__ psi,
    int n_qubits,
    int retained_qubit,
    double *__restrict__ rho,
    int fermion_trace)
{
    const std::uint64_t environment_dim =
        std::uint64_t{1} << (n_qubits - 1);
    const std::uint64_t retained_mask =
        std::uint64_t{1} << retained_qubit;
    const std::uint64_t lower_environment_mask = retained_qubit == 0
        ? std::uint64_t{0}
        : ((std::uint64_t{1} << retained_qubit) - 1u);

    double local00 = 0.0;
    double local11 = 0.0;
    double local01_re = 0.0;
    double local01_im = 0.0;

    const std::uint64_t first =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t environment = first;
         environment < environment_dim;
         environment += stride)
    {
        const std::uint64_t index0 =
            embed_environment_bits(environment, retained_qubit);
        const std::uint64_t index1 = index0 | retained_mask;
        const Cx<Real> a = psi[index0];
        const Cx<Real> b = psi[index1];

        const double ar = static_cast<double>(a.re);
        const double ai = static_cast<double>(a.im);
        const double br = static_cast<double>(b.re);
        const double bi = static_cast<double>(b.im);

        local00 += ar * ar + ai * ai;
        local11 += br * br + bi * bi;

        double sign = 1.0;
        if (fermion_trace &&
            odd_popcount64(environment & lower_environment_mask))
        {
            sign = -1.0;
        }
        local01_re += sign * (ar * br + ai * bi);
        local01_im += sign * (ai * br - ar * bi);
    }

    __shared__ double scratch00[THREADS];
    __shared__ double scratch11[THREADS];
    __shared__ double scratch01_re[THREADS];
    __shared__ double scratch01_im[THREADS];

    scratch00[threadIdx.x] = local00;
    scratch11[threadIdx.x] = local11;
    scratch01_re[threadIdx.x] = local01_re;
    scratch01_im[threadIdx.x] = local01_im;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            scratch00[threadIdx.x] += scratch00[threadIdx.x + stride];
            scratch11[threadIdx.x] += scratch11[threadIdx.x + stride];
            scratch01_re[threadIdx.x] += scratch01_re[threadIdx.x + stride];
            scratch01_im[threadIdx.x] += scratch01_im[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomic_add_f64(&rho[0], scratch00[0]);
        atomic_add_f64(&rho[1], scratch11[0]);
        atomic_add_f64(&rho[2], scratch01_re[0]);
        atomic_add_f64(&rho[3], scratch01_im[0]);
    }
}

template <typename Real>
int launch_rho1_complex(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit,
    double *host_rho,
    int fermion_trace)
{
    if (device_state_vector == nullptr || host_rho == nullptr ||
        n_qubits < 1 || n_qubits >= 63 ||
        retained_qubit < 0 || retained_qubit >= n_qubits)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    static thread_local double *device_rho = nullptr;
    if (device_rho == nullptr)
    {
        const cudaError_t allocation =
            cudaMalloc(&device_rho, OUTPUT_VALUES * sizeof(double));
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
    }

    cudaError_t status = cudaMemset(
        device_rho, 0, OUTPUT_VALUES * sizeof(double));
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    const std::uint64_t environment_dim =
        std::uint64_t{1} << (n_qubits - 1);
    const std::uint64_t blocks_needed =
        (environment_dim + THREADS - 1) / THREADS;
    const int blocks = static_cast<int>(
        blocks_needed < 1024u ? blocks_needed : 1024u);

    rho1_complex_kernel<Real><<<blocks, THREADS>>>(
        reinterpret_cast<const Cx<Real> *>(device_state_vector),
        n_qubits,
        retained_qubit,
        device_rho,
        fermion_trace ? 1 : 0);

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    status = cudaMemcpy(
        host_rho,
        device_rho,
        OUTPUT_VALUES * sizeof(double),
        cudaMemcpyDeviceToHost);
    return static_cast<int>(status);
}
} // namespace

extern "C" int mipt_cuda_rho1_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit,
    double *host_rho_ri)
{
    return launch_rho1_complex<double>(
        device_state_vector,
        n_qubits,
        retained_qubit,
        host_rho_ri,
        0);
}

extern "C" int mipt_cuda_rho1_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit,
    double *host_rho_ri)
{
    return launch_rho1_complex<float>(
        device_state_vector,
        n_qubits,
        retained_qubit,
        host_rho_ri,
        0);
}

extern "C" int mipt_cuda_rho1_complex_fermion_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit,
    double *host_rho_ri)
{
    return launch_rho1_complex<double>(
        device_state_vector,
        n_qubits,
        retained_qubit,
        host_rho_ri,
        1);
}

extern "C" int mipt_cuda_rho1_complex_fermion_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit,
    double *host_rho_ri)
{
    return launch_rho1_complex<float>(
        device_state_vector,
        n_qubits,
        retained_qubit,
        host_rho_ri,
        1);
}
