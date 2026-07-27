#include "mipt/cuda/ancilla_rdm.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

namespace
{
constexpr int THREADS = 256;
constexpr int RHO1_OUTPUT_VALUES = 4;
constexpr int PARITY_OUTPUT_VALUES = 2;
constexpr int RHO2_PACKED_VALUES = 16;
constexpr int RHO2_OUTPUT_VALUES = 32;
constexpr int RHO3_THREADS = 128;
constexpr int RHO3_DIMENSION = 8;
constexpr int RHO3_TILE = 4;
constexpr int RHO3_COMPONENTS_PER_TILE = 2 * RHO3_TILE * RHO3_TILE;
constexpr int RHO3_OUTPUT_VALUES = 2 * RHO3_DIMENSION * RHO3_DIMENSION;
constexpr int RHO4_THREADS = 128;
constexpr int RHO4_DIMENSION = 16;
constexpr int RHO4_TILE = 4;
constexpr int RHO4_COMPONENTS_PER_TILE = 2 * RHO4_TILE * RHO4_TILE;
constexpr int RHO4_OUTPUT_VALUES = 2 * RHO4_DIMENSION * RHO4_DIMENSION;

template <typename Real> struct Cx
{
    Real re;
    Real im;
};

// Native atomicAdd(double*) requires compute capability >= 6.0. nvcc may
// otherwise compile this translation unit for an older default architecture,
// even when the runtime GPU is newer. Use the standard atomicCAS fallback so
// both the FP32 and FP64 entry points compile for every supported target.
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
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__device__ __forceinline__ bool odd_popcount64(std::uint64_t value)
{
    return (__popcll(static_cast<unsigned long long>(value)) & 1) != 0;
}

__device__ __forceinline__ std::uint64_t embed_environment_bits(std::uint64_t environment, int retained_qubit)
{
    const std::uint64_t lower_mask =
        retained_qubit == 0 ? std::uint64_t{0} : ((std::uint64_t{1} << retained_qubit) - 1u);
    const std::uint64_t lower = environment & lower_mask;
    const std::uint64_t upper = environment >> retained_qubit;
    return lower | (upper << (retained_qubit + 1));
}

__device__ __forceinline__ std::uint64_t embed_environment_bits_two(std::uint64_t environment, int retained_qubit0,
    int retained_qubit1)
{
    const int lower = retained_qubit0 < retained_qubit1 ? retained_qubit0 : retained_qubit1;
    const int upper = retained_qubit0 < retained_qubit1 ? retained_qubit1 : retained_qubit0;
    return embed_environment_bits(embed_environment_bits(environment, lower), upper);
}

__device__ __forceinline__ std::uint64_t embed_environment_bits_three(std::uint64_t environment, int q0, int q1, int q2)
{
    int retained[3] = {q0, q1, q2};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = i + 1; j < 3; ++j)
        {
            if (retained[j] < retained[i])
            {
                const int tmp = retained[i];
                retained[i] = retained[j];
                retained[j] = tmp;
            }
        }
    }
    std::uint64_t value = environment;
    for (int i = 0; i < 3; ++i)
    {
        value = embed_environment_bits(value, retained[i]);
    }
    return value;
}

__device__ __forceinline__ std::uint64_t embed_environment_bits_four(std::uint64_t environment, int q0, int q1, int q2,
    int q3)
{
    int retained[4] = {q0, q1, q2, q3};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
        {
            if (retained[j] < retained[i])
            {
                const int tmp = retained[i];
                retained[i] = retained[j];
                retained[j] = tmp;
            }
        }
    }
    std::uint64_t value = environment;
    for (int i = 0; i < 4; ++i)
    {
        value = embed_environment_bits(value, retained[i]);
    }
    return value;
}

__device__ __forceinline__ int two_mode_reordered_position(int mode, int retained_qubit0, int retained_qubit1)
{
    if (mode == retained_qubit0)
    {
        return 0;
    }
    if (mode == retained_qubit1)
    {
        return 1;
    }
    return 2 + mode - (retained_qubit0 < mode ? 1 : 0) - (retained_qubit1 < mode ? 1 : 0);
}

__device__ __forceinline__ double two_mode_fermion_reorder_sign(std::uint64_t occupation_mask, int n_qubits,
                                                                int retained_qubit0, int retained_qubit1)
{
    std::uint64_t seen_new_positions = 0;
    int parity = 0;
    for (int mode = 0; mode < n_qubits; ++mode)
    {
        if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
        {
            continue;
        }
        const int position = two_mode_reordered_position(mode, retained_qubit0, retained_qubit1);
        const std::uint64_t lower_or_equal = (std::uint64_t{1} << (position + 1)) - std::uint64_t{1};
        parity ^= odd_popcount64(seen_new_positions & ~lower_or_equal) ? 1 : 0;
        seen_new_positions |= std::uint64_t{1} << position;
    }
    return parity ? -1.0 : 1.0;
}

__device__ __forceinline__ int three_mode_reordered_position(int mode, int q0, int q1, int q2)
{
    const int retained[3] = {q0, q1, q2};
    for (int local = 0; local < 3; ++local)
    {
        if (mode == retained[local])
        {
            return local;
        }
    }
    int before = 0;
    for (int local = 0; local < 3; ++local)
    {
        before += retained[local] < mode ? 1 : 0;
    }
    return 3 + mode - before;
}

__device__ __forceinline__ double three_mode_fermion_reorder_sign(std::uint64_t occupation_mask, int n_qubits, int q0,
                                                                  int q1, int q2)
{
    std::uint64_t seen_new_positions = 0;
    int parity = 0;
    for (int mode = 0; mode < n_qubits; ++mode)
    {
        if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
        {
            continue;
        }
        const int position = three_mode_reordered_position(mode, q0, q1, q2);
        const std::uint64_t lower_or_equal = (std::uint64_t{1} << (position + 1)) - std::uint64_t{1};
        parity ^= odd_popcount64(seen_new_positions & ~lower_or_equal) ? 1 : 0;
        seen_new_positions |= std::uint64_t{1} << position;
    }
    return parity ? -1.0 : 1.0;
}

__device__ __forceinline__ int four_mode_reordered_position(int mode, int q0, int q1, int q2, int q3)
{
    const int retained[4] = {q0, q1, q2, q3};
    for (int local = 0; local < 4; ++local)
    {
        if (mode == retained[local])
        {
            return local;
        }
    }
    int before = 0;
    for (int local = 0; local < 4; ++local)
    {
        before += retained[local] < mode ? 1 : 0;
    }
    return 4 + mode - before;
}

__device__ __forceinline__ double four_mode_fermion_reorder_sign(std::uint64_t occupation_mask, int n_qubits, int q0,
                                                                 int q1, int q2, int q3)
{
    std::uint64_t seen_new_positions = 0;
    int parity = 0;
    for (int mode = 0; mode < n_qubits; ++mode)
    {
        if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
        {
            continue;
        }
        const int position = four_mode_reordered_position(mode, q0, q1, q2, q3);
        const std::uint64_t lower_or_equal = (std::uint64_t{1} << (position + 1)) - std::uint64_t{1};
        parity ^= odd_popcount64(seen_new_positions & ~lower_or_equal) ? 1 : 0;
        seen_new_positions |= std::uint64_t{1} << position;
    }
    return parity ? -1.0 : 1.0;
}

template <typename Real>
__global__ void rho1_complex_kernel(const Cx<Real> *__restrict__ psi, int n_qubits, int retained_qubit,
                                    double *__restrict__ rho, int fermion_trace)
{
    const std::uint64_t environment_dim = std::uint64_t{1} << (n_qubits - 1);
    const std::uint64_t retained_mask = std::uint64_t{1} << retained_qubit;
    const std::uint64_t lower_environment_mask =
        retained_qubit == 0 ? std::uint64_t{0} : ((std::uint64_t{1} << retained_qubit) - 1u);

    double local00 = 0.0;
    double local11 = 0.0;
    double local01_re = 0.0;
    double local01_im = 0.0;

    const std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t environment = first; environment < environment_dim; environment += stride)
    {
        const std::uint64_t index0 = embed_environment_bits(environment, retained_qubit);
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
        if (fermion_trace && odd_popcount64(environment & lower_environment_mask))
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

    for (int reduction_stride = THREADS / 2; reduction_stride > 0; reduction_stride >>= 1)
    {
        if (threadIdx.x < reduction_stride)
        {
            scratch00[threadIdx.x] += scratch00[threadIdx.x + reduction_stride];
            scratch11[threadIdx.x] += scratch11[threadIdx.x + reduction_stride];
            scratch01_re[threadIdx.x] += scratch01_re[threadIdx.x + reduction_stride];
            scratch01_im[threadIdx.x] += scratch01_im[threadIdx.x + reduction_stride];
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
__global__ void parity_weights_kernel(const Cx<Real> *__restrict__ psi, std::uint64_t dimension,
    double *__restrict__ weights)
{
    double local_even = 0.0;
    double local_odd = 0.0;
    const std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t index = first; index < dimension; index += stride)
    {
        const Cx<Real> amplitude = psi[index];
        const double real = static_cast<double>(amplitude.re);
        const double imag = static_cast<double>(amplitude.im);
        const double probability = real * real + imag * imag;
        if (odd_popcount64(index))
        {
            local_odd += probability;
        }
        else
        {
            local_even += probability;
        }
    }

    __shared__ double scratch_even[THREADS];
    __shared__ double scratch_odd[THREADS];
    scratch_even[threadIdx.x] = local_even;
    scratch_odd[threadIdx.x] = local_odd;
    __syncthreads();

    for (int reduction_stride = THREADS / 2; reduction_stride > 0; reduction_stride >>= 1)
    {
        if (threadIdx.x < reduction_stride)
        {
            scratch_even[threadIdx.x] += scratch_even[threadIdx.x + reduction_stride];
            scratch_odd[threadIdx.x] += scratch_odd[threadIdx.x + reduction_stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomic_add_f64(&weights[0], scratch_even[0]);
        atomic_add_f64(&weights[1], scratch_odd[0]);
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
__global__ void rho2_complex_kernel(const Cx<Real> *__restrict__ psi, int n_qubits, int retained_qubit0,
                                    int retained_qubit1, double *__restrict__ packed_rho, int fermion_trace)
{
    const std::uint64_t environment_dim = std::uint64_t{1} << (n_qubits - 2);
    const std::uint64_t mask0 = std::uint64_t{1} << retained_qubit0;
    const std::uint64_t mask1 = std::uint64_t{1} << retained_qubit1;

    double local[RHO2_PACKED_VALUES]{};
    const std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t grid_stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

    for (std::uint64_t environment = first; environment < environment_dim; environment += grid_stride)
    {
        const std::uint64_t base = embed_environment_bits_two(environment, retained_qubit0, retained_qubit1);
        double ar[4]{};
        double ai[4]{};
        for (int local_index = 0; local_index < 4; ++local_index)
        {
            const std::uint64_t index =
                base | ((local_index & 1) ? mask0 : std::uint64_t{0}) | ((local_index & 2) ? mask1 : std::uint64_t{0});
            const Cx<Real> amplitude = psi[index];
            const double sign =
                fermion_trace ? two_mode_fermion_reorder_sign(index, n_qubits, retained_qubit0, retained_qubit1) : 1.0;
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

    for (int reduction_stride = THREADS / 2; reduction_stride > 0; reduction_stride >>= 1)
    {
        if (threadIdx.x < reduction_stride)
        {
            for (int component = 0; component < RHO2_PACKED_VALUES; ++component)
            {
                scratch[component][threadIdx.x] += scratch[component][threadIdx.x + reduction_stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        for (int component = 0; component < RHO2_PACKED_VALUES; ++component)
        {
            atomic_add_f64(&packed_rho[component], scratch[component][0]);
        }
    }
}

template <typename Real>
__global__ void rho3_complex_kernel(const Cx<Real> *__restrict__ psi, int n_qubits, int q0, int q1, int q2,
                                    double *__restrict__ rho, int fermion_trace)
{
    const int tile = blockIdx.x;
    const int row0 = (tile / 2) * RHO3_TILE;
    const int col0 = (tile % 2) * RHO3_TILE;
    const std::uint64_t masks[3] = {std::uint64_t{1} << q0, std::uint64_t{1} << q1, std::uint64_t{1} << q2};
    const std::uint64_t environment_dim = std::uint64_t{1} << (n_qubits - 3);
    double local[RHO3_COMPONENTS_PER_TILE]{};

    for (std::uint64_t environment = threadIdx.x; environment < environment_dim; environment += blockDim.x)
    {
        const std::uint64_t base = embed_environment_bits_three(environment, q0, q1, q2);
        double row_re[RHO3_TILE]{};
        double row_im[RHO3_TILE]{};
        double col_re[RHO3_TILE]{};
        double col_im[RHO3_TILE]{};

        for (int item = 0; item < RHO3_TILE; ++item)
        {
            const int row_local = row0 + item;
            const int col_local = col0 + item;
            std::uint64_t row_index = base;
            std::uint64_t col_index = base;
            for (int bit = 0; bit < 3; ++bit)
            {
                row_index |= (row_local & (1 << bit)) ? masks[bit] : std::uint64_t{0};
                col_index |= (col_local & (1 << bit)) ? masks[bit] : std::uint64_t{0};
            }
            const Cx<Real> row_amplitude = psi[row_index];
            const Cx<Real> col_amplitude = psi[col_index];
            const double row_sign =
                fermion_trace ? three_mode_fermion_reorder_sign(row_index, n_qubits, q0, q1, q2) : 1.0;
            const double col_sign =
                fermion_trace ? three_mode_fermion_reorder_sign(col_index, n_qubits, q0, q1, q2) : 1.0;
            row_re[item] = row_sign * static_cast<double>(row_amplitude.re);
            row_im[item] = row_sign * static_cast<double>(row_amplitude.im);
            col_re[item] = col_sign * static_cast<double>(col_amplitude.re);
            col_im[item] = col_sign * static_cast<double>(col_amplitude.im);
        }

        for (int row = 0; row < RHO3_TILE; ++row)
        {
            for (int col = 0; col < RHO3_TILE; ++col)
            {
                const int component = 2 * (RHO3_TILE * row + col);
                local[component] += row_re[row] * col_re[col] + row_im[row] * col_im[col];
                local[component + 1] += row_im[row] * col_re[col] - row_re[row] * col_im[col];
            }
        }
    }

    __shared__ double scratch[RHO3_COMPONENTS_PER_TILE][RHO3_THREADS];
    for (int component = 0; component < RHO3_COMPONENTS_PER_TILE; ++component)
    {
        scratch[component][threadIdx.x] = local[component];
    }
    __syncthreads();

    for (int stride = RHO3_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            for (int component = 0; component < RHO3_COMPONENTS_PER_TILE; ++component)
            {
                scratch[component][threadIdx.x] += scratch[component][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        for (int row = 0; row < RHO3_TILE; ++row)
        {
            for (int col = 0; col < RHO3_TILE; ++col)
            {
                const int component = 2 * (RHO3_TILE * row + col);
                const int output = 2 * (RHO3_DIMENSION * (row0 + row) + col0 + col);
                rho[output] = scratch[component][0];
                rho[output + 1] = scratch[component + 1][0];
            }
        }
    }
}

template <typename Real>
__global__ void rho4_complex_kernel(const Cx<Real> *__restrict__ psi, int n_qubits, int q0, int q1, int q2, int q3,
                                    double *__restrict__ rho, int fermion_trace)
{
    const int tile = blockIdx.x;
    const int row0 = (tile / 4) * RHO4_TILE;
    const int col0 = (tile % 4) * RHO4_TILE;
    const std::uint64_t masks[4] = {std::uint64_t{1} << q0, std::uint64_t{1} << q1, std::uint64_t{1} << q2,
        std::uint64_t{1} << q3};
    const std::uint64_t environment_dim = std::uint64_t{1} << (n_qubits - 4);
    double local[RHO4_COMPONENTS_PER_TILE]{};

    for (std::uint64_t environment = threadIdx.x; environment < environment_dim; environment += blockDim.x)
    {
        const std::uint64_t base = embed_environment_bits_four(environment, q0, q1, q2, q3);
        double row_re[RHO4_TILE]{};
        double row_im[RHO4_TILE]{};
        double col_re[RHO4_TILE]{};
        double col_im[RHO4_TILE]{};

        for (int item = 0; item < RHO4_TILE; ++item)
        {
            const int row_local = row0 + item;
            const int col_local = col0 + item;
            std::uint64_t row_index = base;
            std::uint64_t col_index = base;
            for (int bit = 0; bit < 4; ++bit)
            {
                row_index |= (row_local & (1 << bit)) ? masks[bit] : std::uint64_t{0};
                col_index |= (col_local & (1 << bit)) ? masks[bit] : std::uint64_t{0};
            }
            const Cx<Real> row_amplitude = psi[row_index];
            const Cx<Real> col_amplitude = psi[col_index];
            const double row_sign =
                fermion_trace ? four_mode_fermion_reorder_sign(row_index, n_qubits, q0, q1, q2, q3) : 1.0;
            const double col_sign =
                fermion_trace ? four_mode_fermion_reorder_sign(col_index, n_qubits, q0, q1, q2, q3) : 1.0;
            row_re[item] = row_sign * static_cast<double>(row_amplitude.re);
            row_im[item] = row_sign * static_cast<double>(row_amplitude.im);
            col_re[item] = col_sign * static_cast<double>(col_amplitude.re);
            col_im[item] = col_sign * static_cast<double>(col_amplitude.im);
        }

        for (int row = 0; row < RHO4_TILE; ++row)
        {
            for (int col = 0; col < RHO4_TILE; ++col)
            {
                const int component = 2 * (RHO4_TILE * row + col);
                local[component] += row_re[row] * col_re[col] + row_im[row] * col_im[col];
                local[component + 1] += row_im[row] * col_re[col] - row_re[row] * col_im[col];
            }
        }
    }

    __shared__ double scratch[RHO4_COMPONENTS_PER_TILE][RHO4_THREADS];
    for (int component = 0; component < RHO4_COMPONENTS_PER_TILE; ++component)
    {
        scratch[component][threadIdx.x] = local[component];
    }
    __syncthreads();

    for (int stride = RHO4_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            for (int component = 0; component < RHO4_COMPONENTS_PER_TILE; ++component)
            {
                scratch[component][threadIdx.x] += scratch[component][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        for (int row = 0; row < RHO4_TILE; ++row)
        {
            for (int col = 0; col < RHO4_TILE; ++col)
            {
                const int component = 2 * (RHO4_TILE * row + col);
                const int output = 2 * (RHO4_DIMENSION * (row0 + row) + col0 + col);
                rho[output] = scratch[component][0];
                rho[output + 1] = scratch[component + 1][0];
            }
        }
    }
}

template <typename Real>
int launch_rho1_complex(const void *device_state_vector, int n_qubits, int retained_qubit, double *host_rho,
    int fermion_trace)
{
    if (device_state_vector == nullptr || host_rho == nullptr || n_qubits < 1 || n_qubits >= 63 || retained_qubit < 0 ||
        retained_qubit >= n_qubits)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    static thread_local double *device_rho = nullptr;
    if (device_rho == nullptr)
    {
        const cudaError_t allocation = cudaMalloc(&device_rho, RHO1_OUTPUT_VALUES * sizeof(double));
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
    }

    cudaError_t status = cudaMemset(device_rho, 0, RHO1_OUTPUT_VALUES * sizeof(double));
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    const std::uint64_t environment_dim = std::uint64_t{1} << (n_qubits - 1);
    const std::uint64_t blocks_needed = (environment_dim + THREADS - 1) / THREADS;
    const int blocks = static_cast<int>(blocks_needed < 1024u ? blocks_needed : 1024u);

    rho1_complex_kernel<Real><<<blocks, THREADS>>>(reinterpret_cast<const Cx<Real> *>(device_state_vector), n_qubits,
                                                   retained_qubit, device_rho, fermion_trace ? 1 : 0);

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    status = cudaMemcpy(host_rho, device_rho, RHO1_OUTPUT_VALUES * sizeof(double), cudaMemcpyDeviceToHost);
    return static_cast<int>(status);
}

template <typename Real> int launch_parity_weights(const void *device_state_vector, int n_qubits, double *host_weights)
{
    if (device_state_vector == nullptr || host_weights == nullptr || n_qubits < 1 || n_qubits >= 63)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    static thread_local double *device_weights = nullptr;
    if (device_weights == nullptr)
    {
        const cudaError_t allocation = cudaMalloc(&device_weights, PARITY_OUTPUT_VALUES * sizeof(double));
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
    }

    cudaError_t status = cudaMemset(device_weights, 0, PARITY_OUTPUT_VALUES * sizeof(double));
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    const std::uint64_t dimension = std::uint64_t{1} << n_qubits;
    const std::uint64_t blocks_needed = (dimension + THREADS - 1) / THREADS;
    const int blocks = static_cast<int>(blocks_needed < 1024u ? blocks_needed : 1024u);
    parity_weights_kernel<Real>
        <<<blocks, THREADS>>>(reinterpret_cast<const Cx<Real> *>(device_state_vector), dimension, device_weights);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    status = cudaMemcpy(host_weights, device_weights, PARITY_OUTPUT_VALUES * sizeof(double), cudaMemcpyDeviceToHost);
    return static_cast<int>(status);
}

template <typename Real>
int launch_rho2_complex(const void *device_state_vector, int n_qubits, int retained_qubit0, int retained_qubit1,
                        double *host_rho, int fermion_trace)
{
    if (device_state_vector == nullptr || host_rho == nullptr || n_qubits < 2 || n_qubits >= 63 ||
        retained_qubit0 < 0 || retained_qubit0 >= n_qubits || retained_qubit1 < 0 || retained_qubit1 >= n_qubits ||
        retained_qubit0 == retained_qubit1)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    static thread_local double *device_packed_rho = nullptr;
    if (device_packed_rho == nullptr)
    {
        const cudaError_t allocation = cudaMalloc(&device_packed_rho, RHO2_PACKED_VALUES * sizeof(double));
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
    }

    cudaError_t status = cudaMemset(device_packed_rho, 0, RHO2_PACKED_VALUES * sizeof(double));
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    const std::uint64_t environment_dim = std::uint64_t{1} << (n_qubits - 2);
    const std::uint64_t blocks_needed = (environment_dim + THREADS - 1) / THREADS;
    const int blocks = static_cast<int>(blocks_needed < 1024u ? blocks_needed : 1024u);

    rho2_complex_kernel<Real><<<blocks, THREADS>>>(reinterpret_cast<const Cx<Real> *>(device_state_vector), n_qubits,
                                                   retained_qubit0, retained_qubit1, device_packed_rho,
        fermion_trace ? 1 : 0);

    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    double packed[RHO2_PACKED_VALUES]{};
    status = cudaMemcpy(packed, device_packed_rho, RHO2_PACKED_VALUES * sizeof(double), cudaMemcpyDeviceToHost);
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

template <typename Real>
int launch_rho3_complex(const void *device_state_vector, int n_qubits, int q0, int q1, int q2, double *host_rho,
                        int fermion_trace)
{
    const int retained[3] = {q0, q1, q2};
    if (device_state_vector == nullptr || host_rho == nullptr || n_qubits < 3 || n_qubits >= 63)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    for (int i = 0; i < 3; ++i)
    {
        if (retained[i] < 0 || retained[i] >= n_qubits)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        for (int j = i + 1; j < 3; ++j)
        {
            if (retained[i] == retained[j])
            {
                return static_cast<int>(cudaErrorInvalidValue);
            }
        }
    }

    static thread_local double *device_rho = nullptr;
    if (device_rho == nullptr)
    {
        const cudaError_t allocation = cudaMalloc(&device_rho, RHO3_OUTPUT_VALUES * sizeof(double));
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
    }

    rho3_complex_kernel<Real><<<4, RHO3_THREADS>>>(reinterpret_cast<const Cx<Real> *>(device_state_vector), n_qubits,
                                                   q0, q1, q2, device_rho, fermion_trace ? 1 : 0);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    status = cudaMemcpy(host_rho, device_rho, RHO3_OUTPUT_VALUES * sizeof(double), cudaMemcpyDeviceToHost);
    return static_cast<int>(status);
}

template <typename Real>
int launch_rho4_complex(const void *device_state_vector, int n_qubits, int q0, int q1, int q2, int q3, double *host_rho,
    int fermion_trace)
{
    const int retained[4] = {q0, q1, q2, q3};
    if (device_state_vector == nullptr || host_rho == nullptr || n_qubits < 4 || n_qubits >= 63)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    for (int i = 0; i < 4; ++i)
    {
        if (retained[i] < 0 || retained[i] >= n_qubits)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        for (int j = i + 1; j < 4; ++j)
        {
            if (retained[i] == retained[j])
            {
                return static_cast<int>(cudaErrorInvalidValue);
            }
        }
    }

    static thread_local double *device_rho = nullptr;
    if (device_rho == nullptr)
    {
        const cudaError_t allocation = cudaMalloc(&device_rho, RHO4_OUTPUT_VALUES * sizeof(double));
        if (allocation != cudaSuccess)
        {
            return static_cast<int>(allocation);
        }
    }

    rho4_complex_kernel<Real><<<16, RHO4_THREADS>>>(reinterpret_cast<const Cx<Real> *>(device_state_vector), n_qubits,
                                                    q0, q1, q2, q3, device_rho, fermion_trace ? 1 : 0);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    status = cudaMemcpy(host_rho, device_rho, RHO4_OUTPUT_VALUES * sizeof(double), cudaMemcpyDeviceToHost);
    return static_cast<int>(status);
}
} // namespace

extern "C" int mipt_cuda_rho1_complex_f64(const void *device_state_vector, int n_qubits, int retained_qubit,
    double *host_rho_ri)
{
    return launch_rho1_complex<double>(device_state_vector, n_qubits, retained_qubit, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho1_complex_f32(const void *device_state_vector, int n_qubits, int retained_qubit,
    double *host_rho_ri)
{
    return launch_rho1_complex<float>(device_state_vector, n_qubits, retained_qubit, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho1_complex_fermion_f64(const void *device_state_vector, int n_qubits, int retained_qubit,
    double *host_rho_ri)
{
    return launch_rho1_complex<double>(device_state_vector, n_qubits, retained_qubit, host_rho_ri, 1);
}

extern "C" int mipt_cuda_rho1_complex_fermion_f32(const void *device_state_vector, int n_qubits, int retained_qubit,
    double *host_rho_ri)
{
    return launch_rho1_complex<float>(device_state_vector, n_qubits, retained_qubit, host_rho_ri, 1);
}

extern "C" int mipt_cuda_parity_weights_f64(const void *device_state_vector, int n_qubits, double *host_weights)
{
    return launch_parity_weights<double>(device_state_vector, n_qubits, host_weights);
}

extern "C" int mipt_cuda_parity_weights_f32(const void *device_state_vector, int n_qubits, double *host_weights)
{
    return launch_parity_weights<float>(device_state_vector, n_qubits, host_weights);
}

extern "C" int mipt_cuda_rho2_complex_f64(const void *device_state_vector, int n_qubits, int retained_qubit0,
                                          int retained_qubit1, double *host_rho_ri)
{
    return launch_rho2_complex<double>(device_state_vector, n_qubits, retained_qubit0, retained_qubit1, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho2_complex_f32(const void *device_state_vector, int n_qubits, int retained_qubit0,
                                          int retained_qubit1, double *host_rho_ri)
{
    return launch_rho2_complex<float>(device_state_vector, n_qubits, retained_qubit0, retained_qubit1, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho2_complex_fermion_f64(const void *device_state_vector, int n_qubits, int retained_qubit0,
                                                  int retained_qubit1, double *host_rho_ri)
{
    return launch_rho2_complex<double>(device_state_vector, n_qubits, retained_qubit0, retained_qubit1, host_rho_ri, 1);
}

extern "C" int mipt_cuda_rho2_complex_fermion_f32(const void *device_state_vector, int n_qubits, int retained_qubit0,
                                                  int retained_qubit1, double *host_rho_ri)
{
    return launch_rho2_complex<float>(device_state_vector, n_qubits, retained_qubit0, retained_qubit1, host_rho_ri, 1);
}

extern "C" int mipt_cuda_rho3_complex_f64(const void *device_state_vector, int n_qubits, int q0, int q1, int q2,
    double *host_rho_ri)
{
    return launch_rho3_complex<double>(device_state_vector, n_qubits, q0, q1, q2, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho3_complex_f32(const void *device_state_vector, int n_qubits, int q0, int q1, int q2,
    double *host_rho_ri)
{
    return launch_rho3_complex<float>(device_state_vector, n_qubits, q0, q1, q2, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho3_complex_fermion_f64(const void *device_state_vector, int n_qubits, int q0, int q1, int q2,
    double *host_rho_ri)
{
    return launch_rho3_complex<double>(device_state_vector, n_qubits, q0, q1, q2, host_rho_ri, 1);
}

extern "C" int mipt_cuda_rho3_complex_fermion_f32(const void *device_state_vector, int n_qubits, int q0, int q1, int q2,
    double *host_rho_ri)
{
    return launch_rho3_complex<float>(device_state_vector, n_qubits, q0, q1, q2, host_rho_ri, 1);
}

extern "C" int mipt_cuda_rho4_complex_f64(const void *device_state_vector, int n_qubits, int q0, int q1, int q2, int q3,
    double *host_rho_ri)
{
    return launch_rho4_complex<double>(device_state_vector, n_qubits, q0, q1, q2, q3, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho4_complex_f32(const void *device_state_vector, int n_qubits, int q0, int q1, int q2, int q3,
    double *host_rho_ri)
{
    return launch_rho4_complex<float>(device_state_vector, n_qubits, q0, q1, q2, q3, host_rho_ri, 0);
}

extern "C" int mipt_cuda_rho4_complex_fermion_f64(const void *device_state_vector, int n_qubits, int q0, int q1, int q2,
                                                  int q3, double *host_rho_ri)
{
    return launch_rho4_complex<double>(device_state_vector, n_qubits, q0, q1, q2, q3, host_rho_ri, 1);
}

extern "C" int mipt_cuda_rho4_complex_fermion_f32(const void *device_state_vector, int n_qubits, int q0, int q1, int q2,
                                                  int q3, double *host_rho_ri)
{
    return launch_rho4_complex<float>(device_state_vector, n_qubits, q0, q1, q2, q3, host_rho_ri, 1);
}
