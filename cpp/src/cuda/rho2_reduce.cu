#include "mipt/cuda/rho2_reduce.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace
{
constexpr int RETAINED_QUBITS = 2;
constexpr int RHO_DIM = 1 << RETAINED_QUBITS; // 4
constexpr int RHO_ELEMENTS = RHO_DIM * RHO_DIM; // 16
constexpr int RHO_VALUES = 2 * RHO_ELEMENTS;
constexpr int THREADS = 256;

template <typename Real>
struct Cx
{
    Real re;
    Real im;
};

__host__ __device__ __forceinline__ bool odd_popcount64(std::uint64_t x)
{
#if defined(__CUDA_ARCH__)
    return (__popcll(static_cast<unsigned long long>(x)) & 1) != 0;
#elif defined(__GNUG__) || defined(__clang__)
    return (__builtin_popcountll(static_cast<unsigned long long>(x)) & 1) != 0;
#else
    bool parity = false;
    while (x)
    {
        parity = !parity;
        x &= x - 1;
    }
    return parity;
#endif
}

__device__ __forceinline__ bool is_retained_mode(int q, int q0, int q1)
{
    return q == q0 || q == q1;
}

__device__ __forceinline__ int retained_mode_for_local_bit(int local_bit,
                                                            int q0,
                                                            int q1)
{
    return local_bit == 0 ? q0 : q1;
}

__device__ __forceinline__ bool fermion_kept_internal_odd(int local_basis,
                                                           int q0,
                                                           int q1)
{
    return ((local_basis & 1) != 0) && ((local_basis & 2) != 0) && q0 > q1;
}

__device__ __forceinline__ std::uint64_t fermion_env_cross_mask(
    int local_basis,
    int n_qubits,
    int q0,
    int q1)
{
    std::uint64_t mask = 0;
    int env_bit = 0;
    for (int q = 0; q < n_qubits; ++q)
    {
        if (is_retained_mode(q, q0, q1))
        {
            continue;
        }

        bool crosses = false;
        for (int k = 0; k < RETAINED_QUBITS; ++k)
        {
            const int kept_q = retained_mode_for_local_bit(k, q0, q1);
            if (((local_basis >> k) & 1) && q < kept_q)
            {
                crosses = !crosses;
            }
        }
        if (crosses)
        {
            mask |= std::uint64_t{1} << env_bit;
        }
        ++env_bit;
    }
    return mask;
}

__device__ __forceinline__ std::uint64_t embed_environment_bits(
    std::uint64_t env,
    int n_qubits,
    int q0,
    int q1)
{
    std::uint64_t index = 0;
    int env_bit = 0;
    for (int q = 0; q < n_qubits; ++q)
    {
        if (q != q0 && q != q1)
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
    int q1)
{
    return (static_cast<std::uint64_t>((local_basis >> 0) & 1) << q0) |
           (static_cast<std::uint64_t>((local_basis >> 1) & 1) << q1);
}

template <typename Real>
__global__ void rho2_subsystems_complex_kernel(
    const Cx<Real> *__restrict__ psi,
    int n_qubits,
    std::uint64_t env_dim,
    const int *__restrict__ subsystems,
    double *__restrict__ rho_ri,
    int fermion_trace)
{
    const int subsystem = static_cast<int>(blockIdx.x) / RHO_ELEMENTS;
    const int element = static_cast<int>(blockIdx.x) % RHO_ELEMENTS;
    const int row = element / RHO_DIM;
    const int col = element - row * RHO_DIM;
    const int q0 = subsystems[2 * subsystem + 0];
    const int q1 = subsystems[2 * subsystem + 1];

    const std::uint64_t row_offset = embed_retained_bits(row, q0, q1);
    const std::uint64_t col_offset = embed_retained_bits(col, q0, q1);
    const std::uint64_t row_cross_mask = fermion_trace
                                                ? fermion_env_cross_mask(row, n_qubits, q0, q1)
                                                : 0;
    const std::uint64_t col_cross_mask = fermion_trace
                                                ? fermion_env_cross_mask(col, n_qubits, q0, q1)
                                                : 0;
    const bool row_internal_odd = fermion_trace
                                      ? fermion_kept_internal_odd(row, q0, q1)
                                      : false;
    const bool col_internal_odd = fermion_trace
                                      ? fermion_kept_internal_odd(col, q0, q1)
                                      : false;

    double sum_re = 0.0;
    double sum_im = 0.0;
    for (std::uint64_t env = static_cast<std::uint64_t>(threadIdx.x);
         env < env_dim;
         env += static_cast<std::uint64_t>(blockDim.x))
    {
        const std::uint64_t base = embed_environment_bits(env, n_qubits, q0, q1);
        const Cx<Real> a = psi[base | row_offset];
        const Cx<Real> b = psi[base | col_offset];
        const bool row_odd = row_internal_odd != odd_popcount64(env & row_cross_mask);
        const bool col_odd = col_internal_odd != odd_popcount64(env & col_cross_mask);
        const double sign = (fermion_trace && (row_odd != col_odd)) ? -1.0 : 1.0;

        sum_re += sign *
                  (static_cast<double>(a.re) * static_cast<double>(b.re) +
                   static_cast<double>(a.im) * static_cast<double>(b.im));
        sum_im += sign *
                  (static_cast<double>(a.im) * static_cast<double>(b.re) -
                   static_cast<double>(a.re) * static_cast<double>(b.im));
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
    if (capacity >= required_elements)
    {
        return buffer;
    }
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
    return buffer;
}

template <typename Real>
int launch_rho2_subsystems_complex(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major,
    int fermion_trace)
{
    if (device_state_vector == nullptr || host_subsystems == nullptr ||
        host_rho_ri_row_major == nullptr || subsystem_count <= 0 ||
        n_qubits < RETAINED_QUBITS || n_qubits >= 63)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    for (int s = 0; s < subsystem_count; ++s)
    {
        const int q0 = host_subsystems[2 * s + 0];
        const int q1 = host_subsystems[2 * s + 1];
        if (q0 < 0 || q0 >= n_qubits || q1 < 0 || q1 >= n_qubits || q0 == q1)
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
        2u * static_cast<std::size_t>(subsystem_count);
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
    rho2_subsystems_complex_kernel<Real><<<blocks, THREADS>>>(
        reinterpret_cast<const Cx<Real> *>(device_state_vector),
        n_qubits,
        env_dim,
        d_subsystems,
        d_rho,
        fermion_trace ? 1 : 0);

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
} // namespace

extern "C" int mipt_cuda_rho2_subsystems_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major)
{
    return launch_rho2_subsystems_complex<double>(
        device_state_vector, n_qubits, host_subsystems,
        subsystem_count, host_rho_ri_row_major, 0);
}

extern "C" int mipt_cuda_rho2_subsystems_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major)
{
    return launch_rho2_subsystems_complex<float>(
        device_state_vector, n_qubits, host_subsystems,
        subsystem_count, host_rho_ri_row_major, 0);
}

extern "C" int mipt_cuda_rho2_subsystems_complex_fermion_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major)
{
    return launch_rho2_subsystems_complex<double>(
        device_state_vector, n_qubits, host_subsystems,
        subsystem_count, host_rho_ri_row_major, 1);
}

extern "C" int mipt_cuda_rho2_subsystems_complex_fermion_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_subsystems,
    int subsystem_count,
    double *host_rho_ri_row_major)
{
    return launch_rho2_subsystems_complex<float>(
        device_state_vector, n_qubits, host_subsystems,
        subsystem_count, host_rho_ri_row_major, 1);
}
