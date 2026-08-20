#include "mipt/cuda/cusv_parity.hpp"

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <cuda_runtime.h>

namespace
{
// Everything here lives in one anonymous namespace.  A second one containing a
// __global__ would break the build inside cudafe1.stub.c with no reference to
// any line written here -- see the note in cuda/reduce_sweep.cuh.

constexpr int THREADS = 256;
// Grid cap for the reduction, whose finalize kernel walks the partials
// serially, and the wider cap the elementwise passes use.
constexpr int MAX_BLOCKS = 1024;
constexpr int WIDE_BLOCKS = 65535;

// How many amplitudes accumulate in fp32 before folding into the fp64
// accumulator, for fp32 states.  Same trade-off as cuda/reduce_sweep.cuh: it
// keeps the deviation far below the fp32 state vector's own noise while
// leaving the reduction bandwidth-bound rather than fp64-throughput-bound.
constexpr int PROMOTE_EVERY = 64;

char last_error[256] = {0};

void set_cuda_error(const char *operation, cudaError_t status)
{
    std::snprintf(last_error, sizeof(last_error), "%s failed: %s", operation,
                  cudaGetErrorString(status));
}

template <typename Real>
struct Cx
{
    Real re;
    Real im;
};

// An fp32 state accumulates in fp32 and promotes periodically; an fp64 state
// accumulates in fp64 throughout, because the mixed path would throw away the
// precision an FP64=1 build exists for.
template <typename Real>
struct AccumTraits
{
    static constexpr bool mixed = !std::is_same<Real, double>::value;
    using Run = typename std::conditional<mixed, float, double>::type;
};

__device__ __forceinline__ bool odd_popcount64(std::uint64_t x)
{
    return (__popcll(static_cast<unsigned long long>(x)) & 1) != 0;
}

// The two 2x2 blocks of a parity-preserving gate on (target, n-1), in the
// encoded picture.  Layout: v[s * 8 + (2 * row + col) * 2 + (0 = re, 1 = im)].
template <typename Real>
struct ParityGateMatrices
{
    Real v[16];
};

template <typename Real>
__global__ void parity_gate_kernel(Cx<Real> *__restrict__ state, std::uint64_t pairs, int target,
                                   ParityGateMatrices<Real> gate)
{
    // Both blocks are hoisted into registers and selected per thread. Indexing
    // the by-value parameter directly instead costs 2.5x: `gate` lives in the
    // constant bank, which broadcasts only when a warp reads one address, and
    // s varies thread by thread. Measured at 2^23: 0.79 ms this way against
    // 1.99 ms with a divergent `gate.v + 8 * s`.
    const Real e0 = gate.v[0], e1 = gate.v[1], e2 = gate.v[2], e3 = gate.v[3];
    const Real e4 = gate.v[4], e5 = gate.v[5], e6 = gate.v[6], e7 = gate.v[7];
    const Real o0 = gate.v[8], o1 = gate.v[9], o2 = gate.v[10], o3 = gate.v[11];
    const Real o4 = gate.v[12], o5 = gate.v[13], o6 = gate.v[14], o7 = gate.v[15];

    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    const std::uint64_t low_mask = (std::uint64_t{1} << target) - 1u;
    const std::uint64_t target_bit = std::uint64_t{1} << target;
    for (std::uint64_t j = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < pairs; j += stride)
    {
        // Expand j into the index whose `target` bit is clear.
        const std::uint64_t i0 = ((j >> target) << (target + 1)) | (j & low_mask);
        const std::uint64_t i1 = i0 | target_bit;
        // Both members of the pair share s: clearing bit `target` from either
        // one gives i0, whose popcount parity is s.
        const bool odd = odd_popcount64(i0);
        const Real m0 = odd ? o0 : e0, m1 = odd ? o1 : e1;
        const Real m2 = odd ? o2 : e2, m3 = odd ? o3 : e3;
        const Real m4 = odd ? o4 : e4, m5 = odd ? o5 : e5;
        const Real m6 = odd ? o6 : e6, m7 = odd ? o7 : e7;

        const Cx<Real> a = state[i0];
        const Cx<Real> b = state[i1];
        state[i0].re = m0 * a.re - m1 * a.im + m2 * b.re - m3 * b.im;
        state[i0].im = m0 * a.im + m1 * a.re + m2 * b.im + m3 * b.re;
        state[i1].re = m4 * a.re - m5 * a.im + m6 * b.re - m7 * b.im;
        state[i1].im = m4 * a.im + m5 * a.re + m6 * b.im + m7 * b.re;
    }
}

template <typename Real>
__global__ void parity_abs2_kernel(const Cx<Real> *__restrict__ state, std::uint64_t dimension,
                                   double *__restrict__ partials)
{
    using Run = typename AccumTraits<Real>::Run;
    // Two scalars rather than a two-element array. nvcc scalarizes the array
    // form too and the two measure the same; this way the absence of a
    // dynamically indexed local array does not depend on an optimizer pass.
    Run run_even = Run(0), run_odd = Run(0);
    double total_even = 0.0, total_odd = 0.0;

    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    int step = 0;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < dimension; i += stride)
    {
        const Cx<Real> a = state[i];
        const Run w = static_cast<Run>(a.re) * static_cast<Run>(a.re) +
                      static_cast<Run>(a.im) * static_cast<Run>(a.im);
        if (odd_popcount64(i)) { run_odd += w; }
        else { run_even += w; }
        if (AccumTraits<Real>::mixed && ++step == PROMOTE_EVERY)
        {
            step = 0;
            total_even += static_cast<double>(run_even);
            total_odd += static_cast<double>(run_odd);
            run_even = Run(0);
            run_odd = Run(0);
        }
    }

    __shared__ double shared[2 * THREADS];
    shared[threadIdx.x] = total_even + static_cast<double>(run_even);
    shared[THREADS + threadIdx.x] = total_odd + static_cast<double>(run_odd);
    __syncthreads();
    // Fixed-order tree reduction; no atomics, so the block partial is
    // independent of warp scheduling.
    for (int half = THREADS / 2; half > 0; half >>= 1)
    {
        if (static_cast<int>(threadIdx.x) < half)
        {
            shared[threadIdx.x] += shared[threadIdx.x + half];
            shared[THREADS + threadIdx.x] += shared[THREADS + threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        partials[2 * blockIdx.x + 0] = shared[0];
        partials[2 * blockIdx.x + 1] = shared[THREADS];
    }
}

// One block, THREADS threads. The strided partial sums and the tree below both
// run in a fixed order, so the result does not depend on scheduling.
//
// On where this reduction's time actually goes, since two plausible-sounding
// answers were measured and were both wrong: it is not the accumulators and
// not this finalize (a single-thread version measured the same). The sweep
// kernel runs at the read-bandwidth floor -- 0.40 ms for 64 MB, 169 GB/s,
// against a 0.39 ms plain-read floor at the same grid -- and the ~1.4 ms the
// entry point costs is dominated by the synchronizing device-to-host copy of
// the two weights. Sampling needs them on the host, so that sync is inherent
// rather than an artefact; cuStateVec's own Abs2SumArray pays it too. Note
// also that the grid must stay near 1024 blocks: at 8192 this sweep falls to
// 59 GB/s.
__global__ void parity_abs2_finalize(const double *__restrict__ partials, int blocks,
                                     double *__restrict__ out)
{
    __shared__ double shared[2 * THREADS];
    double even = 0.0;
    double odd = 0.0;
    for (int b = static_cast<int>(threadIdx.x); b < blocks; b += THREADS)
    {
        even += partials[2 * b + 0];
        odd += partials[2 * b + 1];
    }
    shared[threadIdx.x] = even;
    shared[THREADS + threadIdx.x] = odd;
    __syncthreads();
    for (int half = THREADS / 2; half > 0; half >>= 1)
    {
        if (static_cast<int>(threadIdx.x) < half)
        {
            shared[threadIdx.x] += shared[threadIdx.x + half];
            shared[THREADS + threadIdx.x] += shared[THREADS + threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        out[0] = shared[0];
        out[1] = shared[THREADS];
    }
}

template <typename Real>
__global__ void parity_collapse_kernel(Cx<Real> *__restrict__ state, std::uint64_t dimension,
                                       int keep_odd, Real scale)
{
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < dimension; i += stride)
    {
        if ((odd_popcount64(i) ? 1 : 0) == keep_odd)
        {
            state[i].re *= scale;
            state[i].im *= scale;
        }
        else
        {
            state[i].re = Real(0);
            state[i].im = Real(0);
        }
    }
}

template <typename Real>
__global__ void parity_expand_kernel(Cx<Real> *__restrict__ state, std::uint64_t half)
{
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t e = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         e < half; e += stride)
    {
        const Cx<Real> v = state[e];
        const Cx<Real> zero = {Real(0), Real(0)};
        const bool odd = odd_popcount64(e);
        state[e] = odd ? zero : v;
        state[e | half] = odd ? v : zero;
    }
}

int grid_for(std::uint64_t elements, int cap)
{
    const std::uint64_t required = (elements + THREADS - 1u) / THREADS;
    const std::uint64_t capped = required < static_cast<std::uint64_t>(cap)
                                     ? required
                                     : static_cast<std::uint64_t>(cap);
    return static_cast<int>(capped < 1u ? 1u : capped);
}

bool bad_state(const void *state, int qubits, int max_qubits)
{
    if (state == nullptr || qubits < 1 || qubits > max_qubits)
    {
        std::snprintf(last_error, sizeof(last_error), "Invalid parity-sector argument.");
        return true;
    }
    return false;
}

int launch_status(const char *what)
{
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        set_cuda_error(what, status);
        return static_cast<int>(status);
    }
    return 0;
}

template <typename Real>
int parity_gate(void *state, int encoded_qubits, int target, const double *even_ri,
                const double *odd_ri)
{
    last_error[0] = '\0';
    if (bad_state(state, encoded_qubits, 62) || even_ri == nullptr || odd_ri == nullptr ||
        target < 0 || target >= encoded_qubits)
    {
        std::snprintf(last_error, sizeof(last_error), "Invalid parity-gate argument.");
        return -1;
    }
    ParityGateMatrices<Real> gate;
    for (int k = 0; k < 8; ++k)
    {
        gate.v[k] = static_cast<Real>(even_ri[k]);
        gate.v[8 + k] = static_cast<Real>(odd_ri[k]);
    }
    const std::uint64_t pairs = std::uint64_t{1} << (encoded_qubits - 1);
    parity_gate_kernel<Real><<<grid_for(pairs, MAX_BLOCKS), THREADS>>>(reinterpret_cast<Cx<Real> *>(state),
                                                           pairs, target, gate);
    return launch_status("parity-gate kernel launch");
}

// One scratch buffer per thread; the circuit driver is single-threaded per run
// (the SDP workers are what run concurrently), so this never contends.
template <typename Real>
double *partial_buffer(cudaError_t &status)
{
    static thread_local double *buffer = nullptr;
    status = cudaSuccess;
    if (buffer == nullptr)
    {
        // 2 bins per block, plus 2 for the finalized result.
        status = cudaMalloc(&buffer, (2u * MAX_BLOCKS + 2u) * sizeof(double));
        if (status != cudaSuccess)
        {
            buffer = nullptr;
            return nullptr;
        }
    }
    return buffer;
}

template <typename Real>
int parity_abs2(const void *state, int encoded_qubits, double *host_weights)
{
    last_error[0] = '\0';
    if (bad_state(state, encoded_qubits, 62) || host_weights == nullptr)
    {
        return -1;
    }
    cudaError_t status = cudaSuccess;
    double *partials = partial_buffer<Real>(status);
    if (partials == nullptr)
    {
        set_cuda_error("cudaMalloc(parity partials)", status);
        return static_cast<int>(status);
    }
    const std::uint64_t dimension = std::uint64_t{1} << encoded_qubits;
    const int blocks = grid_for(dimension, MAX_BLOCKS);
    parity_abs2_kernel<Real><<<blocks, THREADS>>>(reinterpret_cast<const Cx<Real> *>(state),
                                                  dimension, partials);
    if (const int rc = launch_status("parity-abs2 kernel launch"))
    {
        return rc;
    }
    parity_abs2_finalize<<<1, THREADS>>>(partials, blocks, partials + 2 * MAX_BLOCKS);
    if (const int rc = launch_status("parity-abs2 finalize launch"))
    {
        return rc;
    }
    status = cudaMemcpy(host_weights, partials + 2 * MAX_BLOCKS, 2 * sizeof(double),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        set_cuda_error("cudaMemcpy(parity weights)", status);
        return static_cast<int>(status);
    }
    return 0;
}

template <typename Real>
int parity_collapse(void *state, int encoded_qubits, int outcome, double weight)
{
    last_error[0] = '\0';
    if (bad_state(state, encoded_qubits, 62) || (outcome != 0 && outcome != 1) || !(weight > 0.0))
    {
        std::snprintf(last_error, sizeof(last_error), "Invalid parity-collapse argument.");
        return -1;
    }
    const std::uint64_t dimension = std::uint64_t{1} << encoded_qubits;
    const Real scale = static_cast<Real>(1.0 / std::sqrt(weight));
    parity_collapse_kernel<Real><<<grid_for(dimension, WIDE_BLOCKS), THREADS>>>(
        reinterpret_cast<Cx<Real> *>(state), dimension, outcome, scale);
    return launch_status("parity-collapse kernel launch");
}

template <typename Real>
int parity_expand(void *state, int full_qubits)
{
    last_error[0] = '\0';
    if (bad_state(state, full_qubits, 62) || full_qubits < 2)
    {
        std::snprintf(last_error, sizeof(last_error), "Invalid parity-expand argument.");
        return -1;
    }
    const std::uint64_t half = std::uint64_t{1} << (full_qubits - 1);
    parity_expand_kernel<Real><<<grid_for(half, WIDE_BLOCKS), THREADS>>>(reinterpret_cast<Cx<Real> *>(state),
                                                            half);
    return launch_status("parity-expand kernel launch");
}

} // anonymous namespace

extern "C" int mipt_cuda_parity_gate_f32(void *state, int encoded_qubits, int target,
                                         const double *even_ri, const double *odd_ri)
{
    return parity_gate<float>(state, encoded_qubits, target, even_ri, odd_ri);
}

extern "C" int mipt_cuda_parity_gate_f64(void *state, int encoded_qubits, int target,
                                         const double *even_ri, const double *odd_ri)
{
    return parity_gate<double>(state, encoded_qubits, target, even_ri, odd_ri);
}

extern "C" int mipt_cuda_parity_abs2_f32(const void *state, int encoded_qubits,
                                         double *host_weights)
{
    return parity_abs2<float>(state, encoded_qubits, host_weights);
}

extern "C" int mipt_cuda_parity_abs2_f64(const void *state, int encoded_qubits,
                                         double *host_weights)
{
    return parity_abs2<double>(state, encoded_qubits, host_weights);
}

extern "C" int mipt_cuda_parity_collapse_f32(void *state, int encoded_qubits, int outcome,
                                             double weight)
{
    return parity_collapse<float>(state, encoded_qubits, outcome, weight);
}

extern "C" int mipt_cuda_parity_collapse_f64(void *state, int encoded_qubits, int outcome,
                                             double weight)
{
    return parity_collapse<double>(state, encoded_qubits, outcome, weight);
}

extern "C" int mipt_cuda_parity_expand_f32(void *state, int full_qubits)
{
    return parity_expand<float>(state, full_qubits);
}

extern "C" int mipt_cuda_parity_expand_f64(void *state, int full_qubits)
{
    return parity_expand<double>(state, full_qubits);
}

extern "C" const char *mipt_cuda_cusv_parity_last_error()
{
    return last_error;
}
