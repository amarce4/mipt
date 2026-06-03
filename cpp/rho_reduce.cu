#include "rho_reduce.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace
{
    constexpr int RETAINED_QUBITS = 3;
    constexpr int RHO_DIM = 1 << RETAINED_QUBITS; // 8
    constexpr int RHO_ELEMENTS = RHO_DIM * RHO_DIM; // 64
    constexpr int RHO_VALUES = 2 * RHO_ELEMENTS; // real/imag
    constexpr int THREADS = 256;

    template <typename Real>
    struct Cx
    {
        Real re;
        Real im;
    };

    __device__ __forceinline__ std::uint64_t embed_environment_bits(
        std::uint64_t env,
        int n_qubits,
        int q0,
        int q1,
        int q2)
    {
        std::uint64_t index = 0;
        int env_bit = 0;
        for (int q = 0; q < n_qubits; ++q)
        {
            if (q != q0 && q != q1 && q != q2)
            {
                index |= ((env >> env_bit) & std::uint64_t{1}) << q;
                ++env_bit;
            }
        }
        return index;
    }

    __device__ __forceinline__ std::uint64_t embed_retained_bits(
        int local_basis,
        int q0,
        int q1,
        int q2)
    {
        return (static_cast<std::uint64_t>((local_basis >> 0) & 1) << q0) |
               (static_cast<std::uint64_t>((local_basis >> 1) & 1) << q1) |
               (static_cast<std::uint64_t>((local_basis >> 2) & 1) << q2);
    }

    template <typename Real>
    __global__ void rho3_subsystems_complex_kernel(
        const Cx<Real> *__restrict__ psi,
        int n_qubits,
        std::uint64_t env_dim,
        const int *__restrict__ subsystems,
        double *__restrict__ rho_ri)
    {
        const int subsystem = static_cast<int>(blockIdx.x) / RHO_ELEMENTS;
        const int element = static_cast<int>(blockIdx.x) % RHO_ELEMENTS;
        const int row = element / RHO_DIM;
        const int col = element - row * RHO_DIM;
        const int q0 = subsystems[3 * subsystem + 0];
        const int q1 = subsystems[3 * subsystem + 1];
        const int q2 = subsystems[3 * subsystem + 2];
        const std::uint64_t row_offset = embed_retained_bits(row, q0, q1, q2);
        const std::uint64_t col_offset = embed_retained_bits(col, q0, q1, q2);

        double sum_re = 0.0;
        double sum_im = 0.0;
        for (std::uint64_t env = static_cast<std::uint64_t>(threadIdx.x);
             env < env_dim;
             env += static_cast<std::uint64_t>(blockDim.x))
        {
            const std::uint64_t base =
                embed_environment_bits(env, n_qubits, q0, q1, q2);
            const Cx<Real> a = psi[base | row_offset];
            const Cx<Real> b = psi[base | col_offset];
            sum_re += static_cast<double>(a.re) * static_cast<double>(b.re) +
                      static_cast<double>(a.im) * static_cast<double>(b.im);
            sum_im += static_cast<double>(a.im) * static_cast<double>(b.re) -
                      static_cast<double>(a.re) * static_cast<double>(b.im);
        }

        __shared__ double scratch_re[THREADS];
        __shared__ double scratch_im[THREADS];
        scratch_re[threadIdx.x] = sum_re;
        scratch_im[threadIdx.x] = sum_im;
        __syncthreads();

        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                scratch_re[threadIdx.x] += scratch_re[threadIdx.x + stride];
                scratch_im[threadIdx.x] += scratch_im[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            const int output = subsystem * RHO_VALUES + 2 * element;
            rho_ri[output + 0] = scratch_re[0];
            rho_ri[output + 1] = scratch_im[0];
        }
    }

    template <typename T>
    T *thread_buffer(std::size_t required_elements,
                     T *&buffer,
                     std::size_t &capacity,
                     cudaError_t &status)
    {
        status = cudaSuccess;
        if (capacity < required_elements)
        {
            if (buffer != nullptr)
            {
                status = cudaFree(buffer);
                if (status != cudaSuccess)
                {
                    return nullptr;
                }
                buffer = nullptr;
                capacity = 0;
            }

            status = cudaMalloc(&buffer, required_elements * sizeof(T));
            if (status != cudaSuccess)
            {
                return nullptr;
            }
            capacity = required_elements;
        }
        return buffer;
    }

    template <typename Real>
    int launch_rho3_subsystems_complex(
        const void *device_state_vector,
        int n_qubits,
        const int *host_subsystems,
        int subsystem_count,
        double *host_rho_ri_row_major)
    {
        if (device_state_vector == nullptr || host_subsystems == nullptr ||
            host_rho_ri_row_major == nullptr || subsystem_count <= 0 ||
            n_qubits < RETAINED_QUBITS || n_qubits >= 63)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }

        for (int s = 0; s < subsystem_count; ++s)
        {
            const int q0 = host_subsystems[3 * s + 0];
            const int q1 = host_subsystems[3 * s + 1];
            const int q2 = host_subsystems[3 * s + 2];
            if (q0 < 0 || q0 >= n_qubits || q1 < 0 || q1 >= n_qubits ||
                q2 < 0 || q2 >= n_qubits || q0 == q1 || q0 == q2 || q1 == q2)
            {
                return static_cast<int>(cudaErrorInvalidValue);
            }
        }

        static thread_local int *d_subsystems = nullptr;
        static thread_local std::size_t subsystem_capacity = 0;
        static thread_local double *d_rho = nullptr;
        static thread_local std::size_t rho_capacity = 0;

        cudaError_t status = cudaSuccess;
        const std::size_t subsystem_values =
            3u * static_cast<std::size_t>(subsystem_count);
        const std::size_t output_values =
            static_cast<std::size_t>(RHO_VALUES) *
            static_cast<std::size_t>(subsystem_count);

        if (thread_buffer(subsystem_values, d_subsystems,
                          subsystem_capacity, status) == nullptr)
        {
            return static_cast<int>(status);
        }
        if (thread_buffer(output_values, d_rho, rho_capacity, status) == nullptr)
        {
            return static_cast<int>(status);
        }

        status = cudaMemcpy(d_subsystems,
                            host_subsystems,
                            subsystem_values * sizeof(int),
                            cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
        {
            return static_cast<int>(status);
        }

        const std::uint64_t env_dim =
            std::uint64_t{1} << (n_qubits - RETAINED_QUBITS);
        const int blocks = subsystem_count * RHO_ELEMENTS;
        rho3_subsystems_complex_kernel<Real><<<blocks, THREADS>>>(
            reinterpret_cast<const Cx<Real> *>(device_state_vector),
            n_qubits,
            env_dim,
            d_subsystems,
            d_rho);

        status = cudaGetLastError();
        if (status != cudaSuccess)
        {
            return static_cast<int>(status);
        }

        status = cudaMemcpy(host_rho_ri_row_major,
                            d_rho,
                            output_values * sizeof(double),
                            cudaMemcpyDeviceToHost);
        return static_cast<int>(status);
    }
}

extern "C" int mipt_cuda_rho3_subsystems_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major)
{
    return launch_rho3_subsystems_complex<double>(
        device_state_vector,
        n_qubits,
        host_subsystems,
        subsystem_count,
        host_rho_ri_row_major);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major)
{
    return launch_rho3_subsystems_complex<float>(
        device_state_vector,
        n_qubits,
        host_subsystems,
        subsystem_count,
        host_rho_ri_row_major);
}
