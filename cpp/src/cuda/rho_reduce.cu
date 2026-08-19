#include "mipt/cuda/rho_reduce.hpp"
#include "mipt/cuda/reduce_sweep.cuh"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// A *named* namespace, not an anonymous one: reduce_sweep.cuh already
// contributes an anonymous namespace containing a __global__, and nvcc emits
// one _NV_ANON_NAMESPACE alias per translation unit -- a second one breaks the
// build inside cudafe1.stub.c with no reference to any line you wrote. Same
// trap tmi_cuda_svd.cu records.
namespace mipt_rho3_sweep
{
using namespace mipt_reduce;

constexpr int RETAINED_QUBITS = 3;
constexpr int RHO_DIM = 1 << RETAINED_QUBITS;   // 8
constexpr int RHO_ELEMENTS = RHO_DIM * RHO_DIM; // 64
constexpr int RHO_VALUES = 2 * RHO_ELEMENTS;
constexpr int PACKED = (RHO_DIM * (RHO_DIM + 1)) / 2; // 36 lower-triangle entries

// The 8x8 lower triangle does not fit in registers alongside its fp32 running
// sums for two traces, so it is covered by four tiles, each holding at most
// ten complex accumulators:
//
//   tile 0   rows 0-3 x cols 0-3, lower triangle   10 entries, 4 loads
//   tile 1   rows 4-7 x cols 4-7, lower triangle   10 entries, 4 loads
//   tile 2   rows 4-7 x cols 0-1                    8 entries, 6 loads
//   tile 3   rows 4-7 x cols 2-3                    8 entries, 6 loads
//
// which is 36 entries -- an exact cover, so the partial buffer needs no
// zeroing -- at 20 amplitude loads per environment index against the 8 a
// single-tile sweep would need. That is 2.5x the minimum traffic for both
// traces, against 32x for the old one-element-per-block kernels.
constexpr int TILES = 4;
constexpr int TILE_ROWS = 4;
constexpr int TILE_MAX_ENTRIES = 10;

__device__ __forceinline__ bool is_retained_mode(int q, int q0, int q1, int q2)
{
    return q == q0 || q == q1 || q == q2;
}

__device__ __forceinline__ int retained_mode_for_local_bit(int local_bit, int q0, int q1, int q2)
{
    return local_bit == 0 ? q0 : (local_bit == 1 ? q1 : q2);
}

__device__ __forceinline__ bool fermion_kept_internal_odd(int local_basis, int q0, int q1, int q2)
{
    const int q[RETAINED_QUBITS] = {q0, q1, q2};
    bool parity = false;
    for (int i = 0; i < RETAINED_QUBITS; ++i)
    {
        if (((local_basis >> i) & 1) == 0)
        {
            continue;
        }
        for (int j = i + 1; j < RETAINED_QUBITS; ++j)
        {
            if (((local_basis >> j) & 1) && q[i] > q[j])
            {
                parity = !parity;
            }
        }
    }
    return parity;
}

__device__ __forceinline__ std::uint64_t fermion_env_cross_mask(int local_basis, int n_qubits,
                                                                int q0, int q1, int q2)
{
    // The host fermionic trace first orders modes as
    //   kept_modes(q0,q1,q2) + environment_modes(ascending).
    // Moving the occupied kept modes ahead of occupied environment modes
    // contributes one sign flip for each occupied env mode below an occupied
    // kept mode. This mask is expressed in environment-basis bit coordinates.
    std::uint64_t mask = 0;
    int env_bit = 0;
    for (int q = 0; q < n_qubits; ++q)
    {
        if (is_retained_mode(q, q0, q1, q2))
        {
            continue;
        }

        bool crosses = false;
        for (int k = 0; k < RETAINED_QUBITS; ++k)
        {
            const int kept_q = retained_mode_for_local_bit(k, q0, q1, q2);
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

__device__ __forceinline__ std::uint64_t embed_retained_bits(int local_basis, int q0, int q1,
                                                             int q2)
{
    return (static_cast<std::uint64_t>((local_basis >> 0) & 1) << q0) |
           (static_cast<std::uint64_t>((local_basis >> 1) & 1) << q1) |
           (static_cast<std::uint64_t>((local_basis >> 2) & 1) << q2);
}

// One block sweeps one environment slice of one tile of one triangle. See
// reduce_sweep.cuh for the mixed-precision accumulation and the fermionic
// sign factorization.
template <typename Real, int MODE>
__global__ void rho3_sweep_kernel(const Cx<Real> *__restrict__ psi, int n_qubits,
                                  std::uint64_t env_dim, const int *__restrict__ subsystems,
                                  double *__restrict__ partials, int split)
{
    constexpr int TRACES = (MODE == SWEEP_BOTH) ? 2 : 1;
    constexpr bool NEEDS_SIGNS = (MODE != SWEEP_ORDINARY);

    const int subsystem = static_cast<int>(blockIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    const int slice = static_cast<int>(blockIdx.z);

    // Tile 0 is the only one whose rows start at 0; the rest read rows 4-7.
    const int row_base = (tile == 0) ? 0 : TILE_ROWS;
    const int col_base = (tile == 0) ? 0 : (tile == 1 ? TILE_ROWS : (tile == 2 ? 0 : 2));
    const bool diagonal_tile = (tile <= 1);
    const int col_count = diagonal_tile ? TILE_ROWS : 2;

    const int q0 = subsystems[3 * subsystem + 0];
    const int q1 = subsystems[3 * subsystem + 1];
    const int q2 = subsystems[3 * subsystem + 2];

    // Retained bit k is mode q[k]; callers do not sort those. Only the
    // environment embedding needs ascending order.
    int sorted[RETAINED_QUBITS] = {q0, q1, q2};
#pragma unroll
    for (int i = 0; i < RETAINED_QUBITS; ++i)
    {
#pragma unroll
        for (int j = i + 1; j < RETAINED_QUBITS; ++j)
        {
            if (sorted[j] < sorted[i])
            {
                const int swap = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = swap;
            }
        }
    }
    const EnvEmbedding embed = make_env_embedding(sorted, RETAINED_QUBITS);

    // Basis states this tile touches: its rows, then its columns when they are
    // a different set.
    constexpr int MAX_STATES = TILE_ROWS + 2;
    int state_of[MAX_STATES];
    std::uint64_t state_offset[MAX_STATES];
    std::uint64_t cross_mask[MAX_STATES];
    int internal_odd = 0;
    const int state_count = diagonal_tile ? TILE_ROWS : (TILE_ROWS + col_count);
#pragma unroll
    for (int i = 0; i < MAX_STATES; ++i)
    {
        const int basis = (i < TILE_ROWS) ? (row_base + i) : (col_base + (i - TILE_ROWS));
        state_of[i] = basis;
        state_offset[i] = embed_retained_bits(basis, q0, q1, q2);
        cross_mask[i] = NEEDS_SIGNS ? fermion_env_cross_mask(basis, n_qubits, q0, q1, q2) : 0;
        if (NEEDS_SIGNS && fermion_kept_internal_odd(basis, q0, q1, q2))
        {
            internal_odd |= 1 << i;
        }
    }

    using Traits = SweepTraits<Real>;
    using Run = typename Traits::Run;
    double acc_re[TRACES][TILE_MAX_ENTRIES];
    double acc_im[TRACES][TILE_MAX_ENTRIES];
    Run run_re[TRACES][TILE_MAX_ENTRIES];
    Run run_im[TRACES][TILE_MAX_ENTRIES];
#pragma unroll
    for (int t = 0; t < TRACES; ++t)
    {
#pragma unroll
        for (int e = 0; e < TILE_MAX_ENTRIES; ++e)
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
        Run xr[TRACES][MAX_STATES];
        Run xi[TRACES][MAX_STATES];
#pragma unroll
        for (int i = 0; i < MAX_STATES; ++i)
        {
            if (i >= state_count)
            {
                continue;
            }
            const Cx<Real> amplitude = psi[base | state_offset[i]];
            const Run re = static_cast<Run>(amplitude.re);
            const Run im = static_cast<Run>(amplitude.im);
            if (MODE != SWEEP_FERMIONIC)
            {
                xr[0][i] = re;
                xi[0][i] = im;
            }
            if (NEEDS_SIGNS)
            {
                const bool odd =
                    (((internal_odd >> i) & 1) != 0) != odd_popcount64(env & cross_mask[i]);
                const Run eps = odd ? Run(-1) : Run(1);
                const int t = (MODE == SWEEP_BOTH) ? 1 : 0;
                xr[t][i] = eps * re;
                xi[t][i] = eps * im;
            }
        }

#pragma unroll
        for (int t = 0; t < TRACES; ++t)
        {
            int entry = 0;
#pragma unroll
            for (int r = 0; r < TILE_ROWS; ++r)
            {
#pragma unroll
                for (int c = 0; c < TILE_ROWS; ++c)
                {
                    if (c >= col_count || (diagonal_tile && c > r))
                    {
                        continue;
                    }
                    const int ci = diagonal_tile ? c : (TILE_ROWS + c);
                    run_re[t][entry] += xr[t][r] * xr[t][ci] + xi[t][r] * xi[t][ci];
                    run_im[t][entry] += xi[t][r] * xr[t][ci] - xr[t][r] * xi[t][ci];
                    ++entry;
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
                    for (int e = 0; e < TILE_MAX_ENTRIES; ++e)
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
        for (int e = 0; e < TILE_MAX_ENTRIES; ++e)
        {
            acc_re[t][e] += run_re[t][e];
            acc_im[t][e] += run_im[t][e];
        }
    }

    // Packed index of this tile's entries in the full 8x8 lower triangle,
    // recomputed in the same order the accumulation used.
    int packed_of_entry[TILE_MAX_ENTRIES];
    int entries = 0;
    for (int r = 0; r < TILE_ROWS; ++r)
    {
        for (int c = 0; c < TILE_ROWS; ++c)
        {
            if (c >= col_count || (diagonal_tile && c > r))
            {
                continue;
            }
            const int ci = diagonal_tile ? c : (TILE_ROWS + c);
            packed_of_entry[entries] = tri_index(state_of[r], state_of[ci]);
            ++entries;
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
        for (int e = 0; e < entries; ++e)
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
                    partials[out_base + static_cast<std::size_t>(t) * (2 * PACKED) +
                             2 * packed_of_entry[e] + half] = scratch[0];
                }
            }
        }
    }
}

template <typename Real, int MODE>
int launch_rho3_sweep(const void *device_state_vector, int n_qubits, const int *host_subsystems,
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
        const int q0 = host_subsystems[3 * s + 0];
        const int q1 = host_subsystems[3 * s + 1];
        const int q2 = host_subsystems[3 * s + 2];
        if (q0 < 0 || q0 >= n_qubits || q1 < 0 || q1 >= n_qubits || q2 < 0 || q2 >= n_qubits ||
            q0 == q1 || q0 == q2 || q1 == q2)
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
    const std::size_t subsystem_values = 3u * static_cast<std::size_t>(subsystem_count);
    const std::size_t output_values =
        static_cast<std::size_t>(RHO_VALUES) * static_cast<std::size_t>(subsystem_count);
    const std::uint64_t env_dim = std::uint64_t{1} << (n_qubits - RETAINED_QUBITS);
    const int split = choose_split(env_dim, subsystem_count * TILES);
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

    rho3_sweep_kernel<Real, MODE><<<dim3(subsystem_count, TILES, split), SWEEP_THREADS>>>(
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

template <typename Real, int MODE>
int launch_rho3_sweep_from_host(const void *host_state_vector, int n_qubits,
                                const int *host_subsystems, int subsystem_count,
                                double *host_rho_ri_row_major,
                                double *host_fermion_rho_ri_row_major)
{
    if (host_state_vector == nullptr || n_qubits < RETAINED_QUBITS || n_qubits >= 63)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    static thread_local Cx<Real> *d_state = nullptr;
    static thread_local std::size_t state_capacity = 0;

    cudaError_t status = cudaSuccess;
    const std::size_t state_values = static_cast<std::size_t>(std::uint64_t{1} << n_qubits);
    if (thread_buffer(state_values, d_state, state_capacity, status) == nullptr)
    {
        return static_cast<int>(status);
    }

    status = cudaMemcpy(d_state, host_state_vector, state_values * sizeof(Cx<Real>),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }

    return launch_rho3_sweep<Real, MODE>(d_state, n_qubits, host_subsystems, subsystem_count,
                                         host_rho_ri_row_major, host_fermion_rho_ri_row_major);
}
} // namespace mipt_rho3_sweep

using namespace mipt_rho3_sweep;

extern "C" int mipt_cuda_rho3_subsystems_complex_f64(const void *device_state_vector, int n_qubits,
                                                     const int *host_subsystems,
                                                     int subsystem_count,
                                                     double *host_rho_ri_row_major)
{
    return launch_rho3_sweep<double, SWEEP_ORDINARY>(device_state_vector, n_qubits, host_subsystems,
                                                     subsystem_count, host_rho_ri_row_major,
                                                     nullptr);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_f32(const void *device_state_vector, int n_qubits,
                                                     const int *host_subsystems,
                                                     int subsystem_count,
                                                     double *host_rho_ri_row_major)
{
    return launch_rho3_sweep<float, SWEEP_ORDINARY>(device_state_vector, n_qubits, host_subsystems,
                                                    subsystem_count, host_rho_ri_row_major,
                                                    nullptr);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_fermion_f64(const void *device_state_vector,
                                                             int n_qubits,
                                                             const int *host_subsystems,
                                                             int subsystem_count,
                                                             double *host_rho_ri_row_major)
{
    return launch_rho3_sweep<double, SWEEP_FERMIONIC>(device_state_vector, n_qubits,
                                                      host_subsystems, subsystem_count,
                                                      host_rho_ri_row_major, nullptr);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_fermion_f32(const void *device_state_vector,
                                                             int n_qubits,
                                                             const int *host_subsystems,
                                                             int subsystem_count,
                                                             double *host_rho_ri_row_major)
{
    return launch_rho3_sweep<float, SWEEP_FERMIONIC>(device_state_vector, n_qubits, host_subsystems,
                                                     subsystem_count, host_rho_ri_row_major,
                                                     nullptr);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_both_f64(const void *device_state_vector,
                                                          int n_qubits, const int *host_subsystems,
                                                          int subsystem_count,
                                                          double *host_rho_ri_row_major,
                                                          double *host_fermion_rho_ri_row_major)
{
    return launch_rho3_sweep<double, SWEEP_BOTH>(device_state_vector, n_qubits, host_subsystems,
                                                 subsystem_count, host_rho_ri_row_major,
                                                 host_fermion_rho_ri_row_major);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_both_f32(const void *device_state_vector,
                                                          int n_qubits, const int *host_subsystems,
                                                          int subsystem_count,
                                                          double *host_rho_ri_row_major,
                                                          double *host_fermion_rho_ri_row_major)
{
    return launch_rho3_sweep<float, SWEEP_BOTH>(device_state_vector, n_qubits, host_subsystems,
                                                subsystem_count, host_rho_ri_row_major,
                                                host_fermion_rho_ri_row_major);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_host_f64(const void *host_state_vector,
                                                          int n_qubits, const int *host_subsystems,
                                                          int subsystem_count,
                                                          double *host_rho_ri_row_major)
{
    return launch_rho3_sweep_from_host<double, SWEEP_ORDINARY>(
        host_state_vector, n_qubits, host_subsystems, subsystem_count, host_rho_ri_row_major,
        nullptr);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_host_f32(const void *host_state_vector,
                                                          int n_qubits, const int *host_subsystems,
                                                          int subsystem_count,
                                                          double *host_rho_ri_row_major)
{
    return launch_rho3_sweep_from_host<float, SWEEP_ORDINARY>(host_state_vector, n_qubits,
                                                              host_subsystems, subsystem_count,
                                                              host_rho_ri_row_major, nullptr);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_host_fermion_f64(const void *host_state_vector,
                                                                  int n_qubits,
                                                                  const int *host_subsystems,
                                                                  int subsystem_count,
                                                                  double *host_rho_ri_row_major)
{
    return launch_rho3_sweep_from_host<double, SWEEP_FERMIONIC>(
        host_state_vector, n_qubits, host_subsystems, subsystem_count, host_rho_ri_row_major,
        nullptr);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_host_fermion_f32(const void *host_state_vector,
                                                                  int n_qubits,
                                                                  const int *host_subsystems,
                                                                  int subsystem_count,
                                                                  double *host_rho_ri_row_major)
{
    return launch_rho3_sweep_from_host<float, SWEEP_FERMIONIC>(host_state_vector, n_qubits,
                                                               host_subsystems, subsystem_count,
                                                               host_rho_ri_row_major, nullptr);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_host_both_f64(
    const void *host_state_vector, int n_qubits, const int *host_subsystems, int subsystem_count,
    double *host_rho_ri_row_major, double *host_fermion_rho_ri_row_major)
{
    return launch_rho3_sweep_from_host<double, SWEEP_BOTH>(
        host_state_vector, n_qubits, host_subsystems, subsystem_count, host_rho_ri_row_major,
        host_fermion_rho_ri_row_major);
}

extern "C" int mipt_cuda_rho3_subsystems_complex_host_both_f32(
    const void *host_state_vector, int n_qubits, const int *host_subsystems, int subsystem_count,
    double *host_rho_ri_row_major, double *host_fermion_rho_ri_row_major)
{
    return launch_rho3_sweep_from_host<float, SWEEP_BOTH>(
        host_state_vector, n_qubits, host_subsystems, subsystem_count, host_rho_ri_row_major,
        host_fermion_rho_ri_row_major);
}
