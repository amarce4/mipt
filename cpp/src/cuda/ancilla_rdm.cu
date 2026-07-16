#include "mipt/cuda/ancilla_rdm.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

namespace
{
constexpr int THREADS = 256;
constexpr int RHO1_OUTPUT_VALUES = 4;
constexpr int RHO2_PACKED_VALUES = 16;
constexpr int RHO2_OUTPUT_VALUES = 32;

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

__device__ __forceinline__ std::uint64_t embed_environment_bits_two(
    std::uint64_t environment,
    int retained_qubit0,
    int retained_qubit1)
{
    const int lower = retained_qubit0 < retained_qubit1
        ? retained_qubit0
        : retained_qubit1;
    const int upper = retained_qubit0 < retained_qubit1
        ? retained_qubit1
        : retained_qubit0;
    return embed_environment_bits(
        embed_environment_bits(environment, lower), upper);
}

__device__ __forceinline__ int two_mode_reordered_position(
    int mode,
    int retained_qubit0,
    int retained_qubit1)
{
    if (mode == retained_qubit0)
    {
        return 0;
    }
    if (mode == retained_qubit1)
    {
        return 1;
    }
    return 2 + mode - (retained_qubit0 < mode ? 1 : 0) -
           (retained_qubit1 < mode ? 1 : 0);
}

__device__ __forceinline__ double two_mode_fermion_reorder_sign(
    std::uint64_t occupation_mask,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1)
{
    std::uint64_t seen_new_positions = 0;
    int parity = 0;
    for (int mode = 0; mode < n_qubits; ++mode)
    {
        if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
        {
            continue;
        }
        const int position = two_mode_reordered_position(
            mode, retained_qubit0, retained_qubit1);
        const std::uint64_t lower_or_equal =
            (std::uint64_t{1} << (position + 1)) - std::uint64_t{1};
        parity ^= odd_popcount64(seen_new_positions & ~lower_or_equal) ? 1 : 0;
        seen_new_positions |= std::uint64_t{1} << position;
    }
    return parity ? -1.0 : 1.0;
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

    for (int reduction_stride = THREADS / 2;
         reduction_stride > 0;
         reduction_stride >>= 1)
    {
        if (threadIdx.x < reduction_stride)
        {
            scratch00[threadIdx.x] +=
                scratch00[threadIdx.x + reduction_stride];
            scratch11[threadIdx.x] +=
                scratch11[threadIdx.x + reduction_stride];
            scratch01_re[threadIdx.x] +=
                scratch01_re[threadIdx.x + reduction_stride];
            scratch01_im[threadIdx.x] +=
                scratch01_im[threadIdx.x + reduction_stride];
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

__host__ __device__ __forceinline__ int packed_pair_offset(int row, int col)
{
    // row < col, ordered as 01, 02, 03, 12, 13, 23.
    if (row == 0)
    {
        return 4 + 2 * (col - 1);
    }
    if (row == 1)
    {
        return 10 + 2 * (col - 2);
    }
    return 14;
}

template <typename Real>
__global__ void rho2_complex_kernel(
    const Cx<Real> *__restrict__ psi,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *__restrict__ packed_rho,
    int fermion_trace)
{
    const std::uint64_t environment_dim =
        std::uint64_t{1} << (n_qubits - 2);
    const std::uint64_t mask0 = std::uint64_t{1} << retained_qubit0;
    const std::uint64_t mask1 = std::uint64_t{1} << retained_qubit1;

    double local[RHO2_PACKED_VALUES]{};
    const std::uint64_t first =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t grid_stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

    for (std::uint64_t environment = first;
         environment < environment_dim;
         environment += grid_stride)
    {
        const std::uint64_t base = embed_environment_bits_two(
            environment, retained_qubit0, retained_qubit1);
        double ar[4]{};
        double ai[4]{};
        for (int local_index = 0; local_index < 4; ++local_index)
        {
            const std::uint64_t index = base |
                ((local_index & 1) ? mask0 : std::uint64_t{0}) |
                ((local_index & 2) ? mask1 : std::uint64_t{0});
            const Cx<Real> amplitude = psi[index];
            const double sign = fermion_trace
                ? two_mode_fermion_reorder_sign(
                      index,
                      n_qubits,
                      retained_qubit0,
                      retained_qubit1)
                : 1.0;
            ar[local_index] = sign * static_cast<double>(amplitude.re);
            ai[local_index] = sign * static_cast<double>(amplitude.im);
        }

        for (int row = 0; row < 4; ++row)
        {
            local[row] += ar[row] * ar[row] + ai[row] * ai[row];
            for (int col = row + 1; col < 4; ++col)
            {
                const int offset = packed_pair_offset(row, col);
                local[offset] += ar[row] * ar[col] + ai[row] * ai[col];
                local[offset + 1] += ai[row] * ar[col] - ar[row] * ai[col];
            }
        }
    }

    __shared__ double scratch[RHO2_PACKED_VALUES][THREADS];
    for (int component = 0; component < RHO2_PACKED_VALUES; ++component)
    {
        scratch[component][threadIdx.x] = local[component];
    }
    __syncthreads();

    for (int reduction_stride = THREADS / 2;
         reduction_stride > 0;
         reduction_stride >>= 1)
    {
        if (threadIdx.x < reduction_stride)
        {
            for (int component = 0;
                 component < RHO2_PACKED_VALUES;
                 ++component)
            {
                scratch[component][threadIdx.x] +=
                    scratch[component][threadIdx.x + reduction_stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        for (int component = 0; component < RHO2_PACKED_VALUES; ++component)
        {
            atomic_add_f64(
                &packed_rho[component], scratch[component][0]);
        }
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
            cudaMalloc(&device_rho, RHO1_OUTPUT_VALUES * sizeof(double));
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
    }

    cudaError_t status = cudaMemset(
        device_rho, 0, RHO1_OUTPUT_VALUES * sizeof(double));
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
        RHO1_OUTPUT_VALUES * sizeof(double),
        cudaMemcpyDeviceToHost);
    return static_cast<int>(status);
}

template <typename Real>
int launch_rho2_complex(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *host_rho,
    int fermion_trace)
{
    if (device_state_vector == nullptr || host_rho == nullptr ||
        n_qubits < 2 || n_qubits >= 63 ||
        retained_qubit0 < 0 || retained_qubit0 >= n_qubits ||
        retained_qubit1 < 0 || retained_qubit1 >= n_qubits ||
        retained_qubit0 == retained_qubit1)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    static thread_local double *device_packed_rho = nullptr;
    if (device_packed_rho == nullptr)
    {
        const cudaError_t allocation = cudaMalloc(
            &device_packed_rho, RHO2_PACKED_VALUES * sizeof(double));
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
    }

    cudaError_t status = cudaMemset(
        device_packed_rho, 0, RHO2_PACKED_VALUES * sizeof(double));
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    const std::uint64_t environment_dim =
        std::uint64_t{1} << (n_qubits - 2);
    const std::uint64_t blocks_needed =
        (environment_dim + THREADS - 1) / THREADS;
    const int blocks = static_cast<int>(
        blocks_needed < 1024u ? blocks_needed : 1024u);

    rho2_complex_kernel<Real><<<blocks, THREADS>>>(
        reinterpret_cast<const Cx<Real> *>(device_state_vector),
        n_qubits,
        retained_qubit0,
        retained_qubit1,
        device_packed_rho,
        fermion_trace ? 1 : 0);

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    double packed[RHO2_PACKED_VALUES]{};
    status = cudaMemcpy(
        packed,
        device_packed_rho,
        RHO2_PACKED_VALUES * sizeof(double),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    std::fill(host_rho, host_rho + RHO2_OUTPUT_VALUES, 0.0);
    for (int diagonal = 0; diagonal < 4; ++diagonal)
    {
        host_rho[2 * (4 * diagonal + diagonal)] = packed[diagonal];
    }
    for (int row = 0; row < 4; ++row)
    {
        for (int col = row + 1; col < 4; ++col)
        {
            const int packed_offset = packed_pair_offset(row, col);
            const double real = packed[packed_offset];
            const double imag = packed[packed_offset + 1];
            const int upper = 2 * (4 * row + col);
            const int lower = 2 * (4 * col + row);
            host_rho[upper] = real;
            host_rho[upper + 1] = imag;
            host_rho[lower] = real;
            host_rho[lower + 1] = -imag;
        }
    }
    return 0;
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

extern "C" int mipt_cuda_rho2_complex_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *host_rho_ri)
{
    return launch_rho2_complex<double>(
        device_state_vector,
        n_qubits,
        retained_qubit0,
        retained_qubit1,
        host_rho_ri,
        0);
}

extern "C" int mipt_cuda_rho2_complex_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *host_rho_ri)
{
    return launch_rho2_complex<float>(
        device_state_vector,
        n_qubits,
        retained_qubit0,
        retained_qubit1,
        host_rho_ri,
        0);
}

extern "C" int mipt_cuda_rho2_complex_fermion_f64(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *host_rho_ri)
{
    return launch_rho2_complex<double>(
        device_state_vector,
        n_qubits,
        retained_qubit0,
        retained_qubit1,
        host_rho_ri,
        1);
}

extern "C" int mipt_cuda_rho2_complex_fermion_f32(
    const void *device_state_vector,
    int n_qubits,
    int retained_qubit0,
    int retained_qubit1,
    double *host_rho_ri)
{
    return launch_rho2_complex<float>(
        device_state_vector,
        n_qubits,
        retained_qubit0,
        retained_qubit1,
        host_rho_ri,
        1);
}
