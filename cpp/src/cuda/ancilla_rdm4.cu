#include "mipt/cuda/ancilla_rdm4.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace
{
constexpr int THREADS = 256;
constexpr int LOCAL_DIMENSION = 16;
constexpr int OUTPUT_COMPLEX_VALUES = LOCAL_DIMENSION * LOCAL_DIMENSION;
constexpr int OUTPUT_REAL_VALUES = 2 * OUTPUT_COMPLEX_VALUES;
constexpr int CUBLAS_STATUS_BASE = 1000;

template <typename Real>
struct Cx
{
    Real re;
    Real im;
};

__device__ __forceinline__ bool odd_popcount64(std::uint64_t value)
{
    return (__popcll(static_cast<unsigned long long>(value)) & 1) != 0;
}

template <typename Real>
__global__ void gather_tail_four_kernel(
    const Cx<Real> *__restrict__ psi,
    int system_qubits,
    std::uint64_t environment_dimension,
    Cx<Real> *__restrict__ amplitudes,
    int fermion_trace)
{
    const std::uint64_t total =
        environment_dimension * static_cast<std::uint64_t>(LOCAL_DIMENSION);
    const std::uint64_t first =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

    for (std::uint64_t flat = first; flat < total; flat += stride)
    {
        const int local = static_cast<int>(flat & 15u);
        const std::uint64_t environment = flat >> 4;
        const std::uint64_t source = environment |
            (static_cast<std::uint64_t>(local) << system_qubits);

        Cx<Real> value = psi[source];
        if (fermion_trace && odd_popcount64(environment) &&
            ((__popc(static_cast<unsigned int>(local)) & 1) != 0))
        {
            value.re = -value.re;
            value.im = -value.im;
        }
        // Column-major 16 x environment_dimension matrix: local is the row.
        amplitudes[flat] = value;
    }
}

inline int cublas_error(cublasStatus_t status)
{
    return CUBLAS_STATUS_BASE + static_cast<int>(status);
}

inline cublasHandle_t &thread_cublas_handle()
{
    static thread_local cublasHandle_t handle = nullptr;
    return handle;
}

template <typename Real>
struct Gemm;

template <>
struct Gemm<double>
{
    static cublasStatus_t gathered(
        cublasHandle_t handle,
        const void *amplitudes,
        void *rho,
        int environment_dimension)
    {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        return cublasZgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_C,
            LOCAL_DIMENSION, LOCAL_DIMENSION, environment_dimension,
            &alpha,
            reinterpret_cast<const cuDoubleComplex *>(amplitudes),
            LOCAL_DIMENSION,
            reinterpret_cast<const cuDoubleComplex *>(amplitudes),
            LOCAL_DIMENSION,
            &beta,
            reinterpret_cast<cuDoubleComplex *>(rho),
            LOCAL_DIMENSION);
    }

    static cublasStatus_t direct(
        cublasHandle_t handle,
        const void *state_vector,
        void *rho,
        int environment_dimension)
    {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        return cublasZgemm(
            handle, CUBLAS_OP_C, CUBLAS_OP_N,
            LOCAL_DIMENSION, LOCAL_DIMENSION, environment_dimension,
            &alpha,
            reinterpret_cast<const cuDoubleComplex *>(state_vector),
            environment_dimension,
            reinterpret_cast<const cuDoubleComplex *>(state_vector),
            environment_dimension,
            &beta,
            reinterpret_cast<cuDoubleComplex *>(rho),
            LOCAL_DIMENSION);
    }
};

template <>
struct Gemm<float>
{
    static cublasStatus_t gathered(
        cublasHandle_t handle,
        const void *amplitudes,
        void *rho,
        int environment_dimension)
    {
        const cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
        const cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
        return cublasCgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_C,
            LOCAL_DIMENSION, LOCAL_DIMENSION, environment_dimension,
            &alpha,
            reinterpret_cast<const cuFloatComplex *>(amplitudes),
            LOCAL_DIMENSION,
            reinterpret_cast<const cuFloatComplex *>(amplitudes),
            LOCAL_DIMENSION,
            &beta,
            reinterpret_cast<cuFloatComplex *>(rho),
            LOCAL_DIMENSION);
    }

    static cublasStatus_t direct(
        cublasHandle_t handle,
        const void *state_vector,
        void *rho,
        int environment_dimension)
    {
        const cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
        const cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
        return cublasCgemm(
            handle, CUBLAS_OP_C, CUBLAS_OP_N,
            LOCAL_DIMENSION, LOCAL_DIMENSION, environment_dimension,
            &alpha,
            reinterpret_cast<const cuFloatComplex *>(state_vector),
            environment_dimension,
            reinterpret_cast<const cuFloatComplex *>(state_vector),
            environment_dimension,
            &beta,
            reinterpret_cast<cuFloatComplex *>(rho),
            LOCAL_DIMENSION);
    }
};

template <typename Real>
int launch_rho4_tail_complex(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho,
    int fermion_trace)
{
    if (device_state_vector == nullptr || host_rho == nullptr ||
        n_qubits < 4 || n_qubits >= 63)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int system_qubits = n_qubits - 4;
    const std::uint64_t environment_dimension_u64 =
        std::uint64_t{1} << system_qubits;
    if (environment_dimension_u64 >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int environment_dimension =
        static_cast<int>(environment_dimension_u64);
    const std::uint64_t amplitude_count_u64 =
        environment_dimension_u64 * LOCAL_DIMENSION;
    if (amplitude_count_u64 >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                   sizeof(Cx<Real>)))
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const std::size_t amplitude_bytes =
        static_cast<std::size_t>(amplitude_count_u64) * sizeof(Cx<Real>);

    static thread_local void *device_amplitudes = nullptr;
    static thread_local std::size_t amplitude_capacity = 0;
    static thread_local void *device_rho = nullptr;

    if (fermion_trace && amplitude_capacity < amplitude_bytes)
    {
        if (device_amplitudes != nullptr)
        {
            const cudaError_t release = cudaFree(device_amplitudes);
            if (release != cudaSuccess)
            {
                return static_cast<int>(release);
            }
            device_amplitudes = nullptr;
            amplitude_capacity = 0;
        }
        const cudaError_t allocation = cudaMalloc(
            &device_amplitudes, amplitude_bytes);
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
        amplitude_capacity = amplitude_bytes;
    }

    if (device_rho == nullptr)
    {
        const cudaError_t allocation = cudaMalloc(
            &device_rho, OUTPUT_COMPLEX_VALUES * sizeof(Cx<Real>));
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
    }

    cudaError_t cuda_status = cudaSuccess;
    if (fermion_trace)
    {
        const std::uint64_t blocks_needed =
            (amplitude_count_u64 + THREADS - 1) / THREADS;
        const int blocks = static_cast<int>(
            std::min<std::uint64_t>(blocks_needed, 1024u));
        gather_tail_four_kernel<Real><<<blocks, THREADS>>>(
            reinterpret_cast<const Cx<Real> *>(device_state_vector),
            system_qubits,
            environment_dimension_u64,
            reinterpret_cast<Cx<Real> *>(device_amplitudes),
            1);

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess)
        {
            return static_cast<int>(cuda_status);
        }
    }

    cublasHandle_t &handle = thread_cublas_handle();
    if (handle == nullptr)
    {
        const cublasStatus_t create_status = cublasCreate(&handle);
        if (create_status != CUBLAS_STATUS_SUCCESS)
        {
            handle = nullptr;
            return cublas_error(create_status);
        }
    }

    const cublasStatus_t blas_status = fermion_trace
        ? Gemm<Real>::gathered(
              handle, device_amplitudes, device_rho, environment_dimension)
        : Gemm<Real>::direct(
              handle, device_state_vector, device_rho, environment_dimension);
    if (blas_status != CUBLAS_STATUS_SUCCESS)
    {
        return cublas_error(blas_status);
    }

    Cx<Real> host_matrix[OUTPUT_COMPLEX_VALUES]{};
    cuda_status = cudaMemcpy(
        host_matrix,
        device_rho,
        sizeof(host_matrix),
        cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess)
    {
        return static_cast<int>(cuda_status);
    }

    std::fill(host_rho, host_rho + OUTPUT_REAL_VALUES, 0.0);
    for (int row = 0; row < LOCAL_DIMENSION; ++row)
    {
        for (int col = 0; col < LOCAL_DIMENSION; ++col)
        {
            // cuBLAS stores C in column-major order. The direct tail-state
            // product forms B^H B, whose matrix elements are the transpose of
            // rho; the gathered fermionic path forms A A^H directly.
            const int source = fermion_trace
                ? row + LOCAL_DIMENSION * col
                : col + LOCAL_DIMENSION * row;
            const Cx<Real> value = host_matrix[source];
            const int output = 2 * (LOCAL_DIMENSION * row + col);
            host_rho[output] = static_cast<double>(value.re);
            host_rho[output + 1] = static_cast<double>(value.im);
        }
    }
    return 0;
}
} // namespace

extern "C" int mipt_cuda_rho4_tail_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_ri)
{
    return launch_rho4_tail_complex<double>(
        device_state_vector, n_qubits, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho4_tail_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_ri)
{
    return launch_rho4_tail_complex<float>(
        device_state_vector, n_qubits, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho4_tail_complex_fermion_f64(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_ri)
{
    return launch_rho4_tail_complex<double>(
        device_state_vector, n_qubits, host_rho_ri, 1);
}

extern "C" int mipt_cuda_rho4_tail_complex_fermion_f32(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_ri)
{
    return launch_rho4_tail_complex<float>(
        device_state_vector, n_qubits, host_rho_ri, 1);
}
