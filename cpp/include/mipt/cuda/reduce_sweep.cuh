#pragma once

// Shared scaffolding for the two- and three-site reduced-density-matrix
// sweeps in rho2_reduce.cu and rho_reduce.cu.
//
// Both kernels used to assign *one output element per block* and re-stream the
// whole environment for it: 16 sweeps of the state vector per pair, 64 per
// triangle, with `double` accumulators throughout. Two things were wrong with
// that, and only fixing both pays:
//
//   1. The traffic is 8x (pairs) / 16x (triangles) the minimum. One sweep that
//      keeps every output of a tile in registers reads each amplitude once.
//   2. The accumulate, not the load, is what actually costs. Consumer GPUs run
//      fp64 at 1/32 (Turing) or 1/64 (Ada) of fp32, so a kernel doing eight
//      double flops per two loads is arithmetic-bound. Measured on a GTX 1650
//      at N=24: fusing alone took the pair reduction from 5158 ms to 2294 ms
//      (2.2x, and only 16 GB/s -- still flop-bound); fusing *and* accumulating
//      in fp32 with a periodic promotion to fp64 took it to 184 ms (28x, 201
//      GB/s, i.e. bandwidth-bound at last).
//
// Accuracy of the mixed-precision accumulation, against the old fully-fp64
// kernels on random states at N=24: 7e-10 (pairs) and 4e-10 (triangles) max
// absolute deviation, versus the ~1e-7 relative error the fp32 state vector
// already carries. `PROMOTE_EVERY` sets the fp32 run length; the error grows
// as its square root.
//
// Where the remaining headroom is: the dual-trace three-site sweep uses 221
// registers, which is one block per SM on Turing. That is why it only starts
// beating two single-trace calls once the environment is large enough for
// bandwidth to matter -- 3x slower at N=20, 1.3x faster at N=24. Cutting the
// per-tile accumulator count (templating on the tile kind so the diagonal
// tiles can drop their identically-zero diagonal imaginary parts) would fix
// it; it was measured as worth ~1.5% of a k=0 run and left alone.
//
// The other structural simplification is that the fermionic sign factorizes.
// The old code computed
//     sign(row, col, env) = (row_odd(env) != col_odd(env)) ? -1 : +1
// per matrix element. Since that is (-1)^row_odd * (-1)^col_odd, it is the
// same as sign-flipping the gathered amplitude vector before forming the outer
// product: with eps_r(env) = 1 - 2*row_odd(env), the fermionic RDM is
// sum_env (eps . a)(eps . a)^H. So one sweep can produce *both* traces from
// one set of loads, which is what the `_both` entry points do.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>

namespace mipt_reduce
{
// Everything here is in an anonymous namespace: without -rdc=true each .cu
// compiles its own copy of every kernel it instantiates and registers it
// against its own fatbin, so identically-mangled host stubs from two
// translation units can bind to the wrong fatbin entry. Same hazard
// schmidt_gram.cuh documents.
namespace
{

constexpr int SWEEP_THREADS = 256;

// How many environment steps accumulate in fp32 before being folded into the
// fp64 accumulator. 64 keeps the deviation from a fully-fp64 reduction well
// below the fp32 state vector's own noise while costing one fp64 add per
// element per 64 loads.
constexpr int PROMOTE_EVERY = 64;

// Which traces a sweep produces.
enum SweepMode
{
    SWEEP_ORDINARY = 0,
    SWEEP_FERMIONIC = 1,
    SWEEP_BOTH = 2
};

template <typename Real>
struct Cx
{
    Real re;
    Real im;
};

// How a sweep accumulates.
//
// An fp32 state accumulates in fp32 and promotes into an fp64 accumulator
// every PROMOTE_EVERY steps. An fp64 state does not: the loads are already
// fp64, the mixed path would throw that precision away, and there is nothing
// to win because an FP64=1 build is paying the GPU's fp64 rate on every
// other operation anyway. With `mixed` false the fp64 accumulator provably
// stays zero until the epilogue, so ptxas folds it away and the running sums
// *are* the accumulators.
template <typename Real>
struct SweepTraits
{
    static constexpr bool mixed = !std::is_same<Real, double>::value;
    using Run = typename std::conditional<mixed, float, double>::type;
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

// Packed index of a lower-triangular element (row >= col).
__host__ __device__ __forceinline__ int tri_index(int row, int col)
{
    return (row * (row + 1)) / 2 + col;
}

// Interleaves `count` zero bits into `env` at the sorted retained positions,
// so that the result is the state-vector index whose retained modes are all
// zero. The old kernels did this with a loop over all n qubits per amplitude;
// with the retained modes sorted it is a handful of mask-and-shift operations,
// which matters once one thread does it per four or eight loads.
struct EnvEmbedding
{
    // segment[k] selects the environment bits that shift up by k. Unused
    // segments are zero, so all four can be applied unconditionally.
    std::uint64_t segment[4];

    __device__ __forceinline__ std::uint64_t operator()(std::uint64_t env) const
    {
        return (env & segment[0]) | ((env & segment[1]) << 1) | ((env & segment[2]) << 2) |
               ((env & segment[3]) << 3);
    }
};

// `sorted` must hold the retained modes in ascending order.
__device__ __forceinline__ EnvEmbedding make_env_embedding(const int *sorted, int count)
{
    EnvEmbedding embedding;
    std::uint64_t consumed = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        if (k < count)
        {
            // Environment bits landing below retained mode `sorted[k]`: that
            // mode has already had k bits removed from beneath it.
            const std::uint64_t below = (std::uint64_t{1} << (sorted[k] - k)) - 1u;
            embedding.segment[k] = below ^ consumed;
            consumed = below;
        }
        else if (k == count)
        {
            embedding.segment[k] = ~consumed;
        }
        else
        {
            embedding.segment[k] = 0;
        }
    }
    return embedding;
}

// Growable per-thread device scratch, shared by both reduction units.
template <typename T>
T *thread_buffer(std::size_t required_elements, T *&buffer, std::size_t &capacity,
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

// How many environment slices to split each subsystem across.
//
// One block per (subsystem, tile) leaves the device mostly idle -- a k=3
// trajectory has ~26 triangles, i.e. ~104 blocks -- so the environment is cut
// into `split` strided slices whose partial sums are added by a second kernel.
// The partials are summed in a fixed order rather than with atomics, so the
// result does not depend on block scheduling.
inline int choose_split(std::uint64_t env_dim, int tile_blocks)
{
    // Never so many slices that a thread gets fewer than four iterations;
    // below that the block-level reduction dominates the sweep.
    std::uint64_t ceiling = env_dim / (static_cast<std::uint64_t>(SWEEP_THREADS) * 4u);
    if (ceiling < 1u)
    {
        ceiling = 1u;
    }
    int split = 1;
    while (split < 64 && static_cast<std::uint64_t>(split) * 2u <= ceiling &&
           static_cast<std::uint64_t>(tile_blocks) * static_cast<std::uint64_t>(split) * 2u <= 4096u)
    {
        split *= 2;
    }
    return split;
}

// Sums the per-slice partial lower triangles into the full row-major complex
// matrices the C entry points promise. One thread per output element.
template <int DIM, int PACKED>
__global__ void finalize_kernel(const double *__restrict__ partials, int split, int traces,
                                double *__restrict__ out_first, double *__restrict__ out_second)
{
    const int subsystem = static_cast<int>(blockIdx.x);
    const int element = static_cast<int>(threadIdx.x);
    const int row = element / DIM;
    const int col = element - row * DIM;
    const bool lower = row >= col;
    const int packed = lower ? tri_index(row, col) : tri_index(col, row);

    for (int trace = 0; trace < traces; ++trace)
    {
        double re = 0.0;
        double im = 0.0;
        for (int slice = 0; slice < split; ++slice)
        {
            const std::size_t base =
                ((static_cast<std::size_t>(subsystem) * static_cast<std::size_t>(split) +
                  static_cast<std::size_t>(slice)) *
                     static_cast<std::size_t>(traces) +
                 static_cast<std::size_t>(trace)) *
                    static_cast<std::size_t>(2 * PACKED) +
                static_cast<std::size_t>(2 * packed);
            re += partials[base + 0];
            im += partials[base + 1];
        }
        // Only the lower triangle is accumulated; the upper is its conjugate.
        double *out = (trace == 0) ? out_first : out_second;
        const std::size_t offset =
            static_cast<std::size_t>(subsystem) * static_cast<std::size_t>(2 * DIM * DIM) +
            static_cast<std::size_t>(2 * element);
        out[offset + 0] = re;
        out[offset + 1] = lower ? im : -im;
    }
}

} // anonymous namespace: per-TU kernel copies
} // namespace mipt_reduce
