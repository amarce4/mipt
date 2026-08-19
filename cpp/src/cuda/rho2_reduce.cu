#include "mipt/cuda/rho2_reduce.hpp"
#include "mipt/cuda/reduce_sweep.cuh"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// A *named* namespace, not an anonymous one: reduce_sweep.cuh already
// contributes an anonymous namespace containing a __global__, and nvcc emits
// one _NV_ANON_NAMESPACE alias per translation unit -- a second one breaks the
// build inside cudafe1.stub.c with no reference to any line you wrote. Same
// trap tmi_cuda_svd.cu records.
namespace mipt_rho2_sweep
{
using namespace mipt_reduce;

constexpr int RETAINED_QUBITS = 2;
constexpr int RHO_DIM = 1 << RETAINED_QUBITS;   // 4
constexpr int RHO_ELEMENTS = RHO_DIM * RHO_DIM; // 16
constexpr int RHO_VALUES = 2 * RHO_ELEMENTS;
// Only the lower triangle is swept; the upper is its conjugate.
constexpr int PACKED = (RHO_DIM * (RHO_DIM + 1)) / 2; // 10

__device__ __forceinline__ bool is_retained_mode(int q, int q0, int q1)
{
    return q == q0 || q == q1;
}

__device__ __forceinline__ int retained_mode_for_local_bit(int local_bit, int q0, int q1)
{
    return local_bit == 0 ? q0 : q1;
}

__device__ __forceinline__ bool fermion_kept_internal_odd(int local_basis, int q0, int q1)
{
    return ((local_basis & 1) != 0) && ((local_basis & 2) != 0) && q0 > q1;
}

__device__ __forceinline__ std::uint64_t fermion_env_cross_mask(int local_basis, int n_qubits,
                                                                int q0, int q1)
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

__device__ __forceinline__ std::uint64_t embed_retained_bits(int local_basis, int q0, int q1)
{
    return (static_cast<std::uint64_t>((local_basis >> 0) & 1) << q0) |
           (static_cast<std::uint64_t>((local_basis >> 1) & 1) << q1);
}

// One block sweeps one environment slice of one pair, keeping the whole lower
// triangle of the 4x4 in registers. See reduce_sweep.cuh for why the running
// sums are fp32 and why the fermionic trace costs no extra loads.
template <typename Real, int MODE>
__global__ void rho2_sweep_kernel(const Cx<Real> *__restrict__ psi, int n_qubits,
                                  std::uint64_t env_dim, const int *__restrict__ subsystems,
                                  double *__restrict__ partials, int split)
{
    constexpr int TRACES = (MODE == SWEEP_BOTH) ? 2 : 1;
    constexpr bool NEEDS_SIGNS = (MODE != SWEEP_ORDINARY);

    const int subsystem = static_cast<int>(blockIdx.x);
    const int slice = static_cast<int>(blockIdx.y);
    const int q0 = subsystems[2 * subsystem + 0];
    const int q1 = subsystems[2 * subsystem + 1];

    // Retained bit k of the RDM is mode `q[k]`, which callers do not sort --
    // mode 5 passes descending sets on purpose. Only the environment
    // embedding needs ascending order.
    int sorted[RETAINED_QUBITS] = {q0, q1};
    if (sorted[1] < sorted[0])
    {
        const int swap = sorted[0];
        sorted[0] = sorted[1];
        sorted[1] = swap;
    }
    const EnvEmbedding embed = make_env_embedding(sorted, RETAINED_QUBITS);

    std::uint64_t row_offset[RHO_DIM];
    std::uint64_t cross_mask[RHO_DIM];
    int internal_odd = 0;
#pragma unroll
    for (int r = 0; r < RHO_DIM; ++r)
    {
        row_offset[r] = embed_retained_bits(r, q0, q1);
        cross_mask[r] = NEEDS_SIGNS ? fermion_env_cross_mask(r, n_qubits, q0, q1) : 0;
        if (NEEDS_SIGNS && fermion_kept_internal_odd(r, q0, q1))
        {
            internal_odd |= 1 << r;
        }
    }

    using Traits = SweepTraits<Real>;
    using Run = typename Traits::Run;
    double acc_re[TRACES][PACKED];
    double acc_im[TRACES][PACKED];
    Run run_re[TRACES][PACKED];
    Run run_im[TRACES][PACKED];
#pragma unroll
    for (int t = 0; t < TRACES; ++t)
    {
#pragma unroll
        for (int e = 0; e < PACKED; ++e)
        {
            acc_re[t][e] = 0.0;
            acc_im[t][e] = 0.0;
            run_re[t][e] = Run(0);
            run_im[t][e] = Run(0);
        }
    }

    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * static_cast<std::uint64_t>(split);
    std::uint64_t step = 0;
    for (std::uint64_t env = static_cast<std::uint64_t>(threadIdx.x) +
                             static_cast<std::uint64_t>(slice) * blockDim.x;
         env < env_dim; env += stride, ++step)
    {
        const std::uint64_t base = embed(env);
        Run xr[TRACES][RHO_DIM];
        Run xi[TRACES][RHO_DIM];
#pragma unroll
        for (int r = 0; r < RHO_DIM; ++r)
        {
            const Cx<Real> amplitude = psi[base | row_offset[r]];
            const Run re = static_cast<Run>(amplitude.re);
            const Run im = static_cast<Run>(amplitude.im);
            if (MODE != SWEEP_FERMIONIC)
            {
                xr[0][r] = re;
                xi[0][r] = im;
            }
            if (NEEDS_SIGNS)
            {
                // eps_r = +/-1, and sign(row, col) = eps_row * eps_col, so
                // folding it into the amplitude gives the fermionic RDM as an
                // ordinary outer product.
                const bool odd =
                    (((internal_odd >> r) & 1) != 0) != odd_popcount64(env & cross_mask[r]);
                const Run eps = odd ? Run(-1) : Run(1);
                const int t = (MODE == SWEEP_BOTH) ? 1 : 0;
                xr[t][r] = eps * re;
                xi[t][r] = eps * im;
            }
        }

#pragma unroll
        for (int t = 0; t < TRACES; ++t)
        {
#pragma unroll
            for (int row = 0; row < RHO_DIM; ++row)
            {
#pragma unroll
                for (int col = 0; col <= row; ++col)
                {
                    const int e = tri_index(row, col);
                    run_re[t][e] += xr[t][row] * xr[t][col] + xi[t][row] * xi[t][col];
                    if (col < row)
                    {
                        run_im[t][e] += xi[t][row] * xr[t][col] - xr[t][row] * xi[t][col];
                    }
                }
            }
        }

        if constexpr (Traits::mixed)
        {
            if ((step & (PROMOTE_EVERY - 1)) == PROMOTE_EVERY - 1)
            {
#pragma unroll
                for (int t = 0; t < TRACES; ++t)
                {
#pragma unroll
                    for (int e = 0; e < PACKED; ++e)
                    {
                        acc_re[t][e] += run_re[t][e];
                        acc_im[t][e] += run_im[t][e];
                        run_re[t][e] = Run(0);
                        run_im[t][e] = Run(0);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TRACES; ++t)
    {
#pragma unroll
        for (int e = 0; e < PACKED; ++e)
        {
            acc_re[t][e] += run_re[t][e];
            acc_im[t][e] += run_im[t][e];
        }
    }

    __shared__ double scratch[SWEEP_THREADS];
    const std::size_t out_base =
        ((static_cast<std::size_t>(subsystem) * static_cast<std::size_t>(split) +
          static_cast<std::size_t>(slice)) *
         static_cast<std::size_t>(TRACES)) *
        static_cast<std::size_t>(2 * PACKED);
    for (int t = 0; t < TRACES; ++t)
    {
        for (int e = 0; e < PACKED; ++e)
        {
            for (int half = 0; half < 2; ++half)
            {
                scratch[threadIdx.x] = (half == 0) ? acc_re[t][e] : acc_im[t][e];
                __syncthreads();
                for (int reach = SWEEP_THREADS / 2; reach > 0; reach >>= 1)
                {
                    if (static_cast<int>(threadIdx.x) < reach)
                    {
                        scratch[threadIdx.x] += scratch[threadIdx.x + reach];
                    }
                    __syncthreads();
                }
                if (threadIdx.x == 0)
                {
                    partials[out_base + static_cast<std::size_t>(t) * (2 * PACKED) + 2 * e + half] =
                        scratch[0];
                }
            }
        }
    }
}

template <typename Real, int MODE>
int launch_rho2_sweep(const void *device_state_vector, int n_qubits, const int *host_subsystems,
                      int subsystem_count, double *host_rho_ri_row_major,
                      double *host_fermion_rho_ri_row_major)
{
    constexpr int TRACES = (MODE == SWEEP_BOTH) ? 2 : 1;

    if (device_state_vector == nullptr || host_subsystems == nullptr ||
        host_rho_ri_row_major == nullptr || subsystem_count <= 0 ||
        n_qubits < RETAINED_QUBITS || n_qubits >= 63)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (MODE == SWEEP_BOTH && host_fermion_rho_ri_row_major == nullptr)
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
    static thread_local double *d_partials = nullptr;
    static thread_local std::size_t partial_capacity = 0;
    static thread_local double *d_rho = nullptr;
    static thread_local std::size_t rho_capacity = 0;
    static thread_local double *d_fermion_rho = nullptr;
    static thread_local std::size_t fermion_rho_capacity = 0;

    cudaError_t status = cudaSuccess;
    const std::size_t subsystem_values = 2u * static_cast<std::size_t>(subsystem_count);
    const std::size_t output_values =
        static_cast<std::size_t>(RHO_VALUES) * static_cast<std::size_t>(subsystem_count);
    const std::uint64_t env_dim = std::uint64_t{1} << (n_qubits - RETAINED_QUBITS);
    const int split = choose_split(env_dim, subsystem_count);
    const std::size_t partial_values = static_cast<std::size_t>(subsystem_count) *
                                       static_cast<std::size_t>(split) *
                                       static_cast<std::size_t>(TRACES * 2 * PACKED);

    if (thread_buffer(subsystem_values, d_subsystems, subsystem_capacity, status) == nullptr ||
        thread_buffer(partial_values, d_partials, partial_capacity, status) == nullptr ||
        thread_buffer(output_values, d_rho, rho_capacity, status) == nullptr)
    {
        return static_cast<int>(status);
    }
    if (TRACES == 2 &&
        thread_buffer(output_values, d_fermion_rho, fermion_rho_capacity, status) == nullptr)
    {
        return static_cast<int>(status);
    }

    status = cudaMemcpy(d_subsystems, host_subsystems, subsystem_values * sizeof(int),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    rho2_sweep_kernel<Real, MODE><<<dim3(subsystem_count, split), SWEEP_THREADS>>>(
        reinterpret_cast<const Cx<Real> *>(device_state_vector), n_qubits, env_dim, d_subsystems,
        d_partials, split);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    finalize_kernel<RHO_DIM, PACKED><<<subsystem_count, RHO_ELEMENTS>>>(
        d_partials, split, TRACES, d_rho, (TRACES == 2) ? d_fermion_rho : nullptr);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    status = cudaMemcpy(host_rho_ri_row_major, d_rho, output_values * sizeof(double),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess || TRACES == 1)
    {
        return static_cast<int>(status);
    }
    status = cudaMemcpy(host_fermion_rho_ri_row_major, d_fermion_rho,
                        output_values * sizeof(double), cudaMemcpyDeviceToHost);
    return static_cast<int>(status);
}
} // namespace mipt_rho2_sweep

using namespace mipt_rho2_sweep;

extern "C" int mipt_cuda_rho2_subsystems_complex_f64(const void *device_state_vector, int n_qubits,
                                                     const int *host_subsystems,
                                                     int subsystem_count,
                                                     double *host_rho_ri_row_major)
{
    return launch_rho2_sweep<double, SWEEP_ORDINARY>(device_state_vector, n_qubits, host_subsystems,
                                                     subsystem_count, host_rho_ri_row_major,
                                                     nullptr);
}

extern "C" int mipt_cuda_rho2_subsystems_complex_f32(const void *device_state_vector, int n_qubits,
                                                     const int *host_subsystems,
                                                     int subsystem_count,
                                                     double *host_rho_ri_row_major)
{
    return launch_rho2_sweep<float, SWEEP_ORDINARY>(device_state_vector, n_qubits, host_subsystems,
                                                    subsystem_count, host_rho_ri_row_major,
                                                    nullptr);
}

extern "C" int mipt_cuda_rho2_subsystems_complex_fermion_f64(const void *device_state_vector,
                                                             int n_qubits,
                                                             const int *host_subsystems,
                                                             int subsystem_count,
                                                             double *host_rho_ri_row_major)
{
    return launch_rho2_sweep<double, SWEEP_FERMIONIC>(device_state_vector, n_qubits,
                                                      host_subsystems, subsystem_count,
                                                      host_rho_ri_row_major, nullptr);
}

extern "C" int mipt_cuda_rho2_subsystems_complex_fermion_f32(const void *device_state_vector,
                                                             int n_qubits,
                                                             const int *host_subsystems,
                                                             int subsystem_count,
                                                             double *host_rho_ri_row_major)
{
    return launch_rho2_sweep<float, SWEEP_FERMIONIC>(device_state_vector, n_qubits, host_subsystems,
                                                     subsystem_count, host_rho_ri_row_major,
                                                     nullptr);
}

extern "C" int mipt_cuda_rho2_subsystems_complex_both_f64(const void *device_state_vector,
                                                          int n_qubits,
                                                          const int *host_subsystems,
                                                          int subsystem_count,
                                                          double *host_rho_ri_row_major,
                                                          double *host_fermion_rho_ri_row_major)
{
    return launch_rho2_sweep<double, SWEEP_BOTH>(device_state_vector, n_qubits, host_subsystems,
                                                 subsystem_count, host_rho_ri_row_major,
                                                 host_fermion_rho_ri_row_major);
}

extern "C" int mipt_cuda_rho2_subsystems_complex_both_f32(const void *device_state_vector,
                                                          int n_qubits,
                                                          const int *host_subsystems,
                                                          int subsystem_count,
                                                          double *host_rho_ri_row_major,
                                                          double *host_fermion_rho_ri_row_major)
{
    return launch_rho2_sweep<float, SWEEP_BOTH>(device_state_vector, n_qubits, host_subsystems,
                                                subsystem_count, host_rho_ri_row_major,
                                                host_fermion_rho_ri_row_major);
}
