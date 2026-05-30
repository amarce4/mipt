#include "rho_reduce.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace
{
    constexpr int KEEP_DIM = 8;
    constexpr int RHO_SIZE = KEEP_DIM * KEEP_DIM;
    constexpr int THREADS = 256;

    template <typename Real>
    struct Cx
    {
        Real re;
        Real im;
    };

    template <typename Real>
    __global__ void last3_rho_real_kernel(
        const Cx<Real> *__restrict__ psi,
        std::uint64_t env_dim,
        int env_qubits,
        double *__restrict__ rho)
    {
        const int k = blockIdx.x;
        const int i = k / KEEP_DIM;
        const int j = k - i * KEEP_DIM;

        const std::uint64_t i_offset = static_cast<std::uint64_t>(i) << env_qubits;
        const std::uint64_t j_offset = static_cast<std::uint64_t>(j) << env_qubits;

        double sum = 0.0;

        for (std::uint64_t env = static_cast<std::uint64_t>(threadIdx.x);
             env < env_dim;
             env += static_cast<std::uint64_t>(blockDim.x))
        {
            const Cx<Real> a = psi[i_offset | env];
            const Cx<Real> b = psi[j_offset | env];

            // real(a * conj(b))
            sum += static_cast<double>(a.re) * static_cast<double>(b.re) +
                   static_cast<double>(a.im) * static_cast<double>(b.im);
        }

        __shared__ double scratch[THREADS];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            rho[k] = scratch[0];
        }
    }

    inline int checked_last_cuda_error()
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return static_cast<int>(err);
        }

        err = cudaDeviceSynchronize();
        return static_cast<int>(err);
    }

    inline double *rho_device_buffer(cudaError_t &status)
    {
        static thread_local double *buffer = nullptr;

        status = cudaSuccess;

        if (buffer == nullptr)
        {
            status = cudaMalloc(&buffer, RHO_SIZE * sizeof(double));
            if (status != cudaSuccess)
            {
                return nullptr;
            }
        }

        return buffer;
    }

    template <typename Real>
    int launch_last3_reduced_density_real(
        const void *device_state_vector,
        int n_qubits,
        double *host_rho_real_row_major)
    {
        if (device_state_vector == nullptr || host_rho_real_row_major == nullptr)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }

        if (n_qubits < 3 || n_qubits >= 63)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }

        cudaError_t alloc_status = cudaSuccess;
        double *d_rho = rho_device_buffer(alloc_status);
        if (d_rho == nullptr)
        {
            return static_cast<int>(alloc_status);
        }

        const int env_qubits = n_qubits - 3;
        const std::uint64_t env_dim = std::uint64_t{1} << env_qubits;

        last3_rho_real_kernel<Real><<<RHO_SIZE, THREADS>>>(
            reinterpret_cast<const Cx<Real> *>(device_state_vector),
            env_dim,
            env_qubits,
            d_rho);

        int status = checked_last_cuda_error();
        if (status != 0)
        {
            return status;
        }

        const cudaError_t copy_status = cudaMemcpy(
            host_rho_real_row_major,
            d_rho,
            RHO_SIZE * sizeof(double),
            cudaMemcpyDeviceToHost);

        return static_cast<int>(copy_status);
    }
}

extern "C" int mipt_cuda_last3_reduced_density_real_f64(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_real_row_major)
{
    return launch_last3_reduced_density_real<double>(
        device_state_vector,
        n_qubits,
        host_rho_real_row_major);
}

extern "C" int mipt_cuda_last3_reduced_density_real_f32(
    const void *device_state_vector,
    int n_qubits,
    double *host_rho_real_row_major)
{
    return launch_last3_reduced_density_real<float>(
        device_state_vector,
        n_qubits,
        host_rho_real_row_major);
}
