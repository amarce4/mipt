#pragma once

// Shared Schmidt-matrix gather and Gram-matrix formation for the TMI reduction
// kernels.  Included by src/cuda/tmi_cuda_svd.cu and src/cuda/tmi_cuda_rdm.cu.
//
// Both of those used to stream the whole state vector once per output element:
// tmi_cuda_rdm.cu launched kept_dim^2 blocks that each looped over the entire
// environment, so a kept=6 quarter block at N=24 moved 2 * 2^(N+kept) = 2^31
// amplitudes -- 128 times the 2^N the problem actually requires.  Here the
// state is gathered once into the rows x cols Schmidt matrix and the reduction
// is handed to cuBLAS, which tiles it through shared memory.
//
// Layout contract: the Schmidt matrix is column-major, rows = 2^kept indexed by
// the retained modes in little-endian `kept_modes` order, cols = 2^(n-kept)
// indexed by the complement in ascending mode order.  Then
//     rho = M M^H          (rows x rows, Hermitian)
// is exactly the reduced density matrix of the retained modes, so the same
// gather serves both the entropy path and the RDM-export path.
//
// This translation unit deliberately avoids the C++ standard library: nvcc
// compiles it against libstdc++ while CUDA-Q's nvq++ links libc++, and mixing
// the two produced unresolved std::__cxx11 symbols at link time.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

namespace mipt_gram
{
// Everything below lives in an anonymous namespace, so each translation unit
// that includes this header gets its own internal-linkage copy.
//
// This is the same hazard docs/ARCHITECTURE.md records for nvq++ and the
// __qpu__ kernels, and it bites nvcc's __global__ template instantiations too.
// Without -rdc=true each TU compiles a complete, self-contained copy of every
// kernel it instantiates and registers it against its own fatbin at load time.
// When two TUs instantiate the same template the mangled host stubs collide,
// the linker keeps one, and the surviving stub can end up bound to the other
// TU's fatbin entry.
//
// The failure is quiet and confusing: with both tmi_cuda_svd.cu and
// tmi_cuda_rdm.cu including this header, cusolverDn?heevd started returning
// info != 0 ("did not converge") for half cuts, sim_tmi.exe read that as a
// broken device solver, permanently disabled the device half-cut path, and fell
// back to host LAPACK -- so the run stayed correct but got *slower* than the
// code being replaced. Neither TU misbehaves on its own.
namespace
{
enum
{
    GATHER_THREADS = 256
};

template <typename Real>
struct Cx
{
    Real re;
    Real im;
};

__host__ __device__ inline bool odd_popcount64(uint64_t x)
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

template <typename T>
inline T *thread_buffer(size_t required_elements,
                        T *&buffer,
                        size_t &capacity,
                        cudaError_t &status)
{
    status = cudaSuccess;
    if (required_elements == 0)
    {
        required_elements = 1;
    }
    if (capacity < required_elements)
    {
        if (buffer != NULL)
        {
            status = cudaFree(buffer);
            if (status != cudaSuccess)
            {
                return NULL;
            }
            buffer = NULL;
            capacity = 0;
        }
        status = cudaMalloc(reinterpret_cast<void **>(&buffer),
                            required_elements * sizeof(T));
        if (status != cudaSuccess)
        {
            return NULL;
        }
        capacity = required_elements;
    }
    return buffer;
}

// --- host-side index tables -------------------------------------------------

inline uint64_t *basis_offsets_host(const int *modes, int count)
{
    if (modes == NULL || count < 0 || count >= 63)
    {
        return NULL;
    }
    const uint64_t dim = uint64_t{1} << count;
    uint64_t *offsets =
        static_cast<uint64_t *>(malloc(static_cast<size_t>(dim) * sizeof(uint64_t)));
    if (offsets == NULL)
    {
        return NULL;
    }
    for (uint64_t basis = 0; basis < dim; ++basis)
    {
        uint64_t offset = 0;
        for (int k = 0; k < count; ++k)
        {
            offset |= ((basis >> k) & uint64_t{1}) << modes[k];
        }
        offsets[static_cast<size_t>(basis)] = offset;
    }
    return offsets;
}

inline int complement_modes_host(int n,
                                 const int *kept_modes,
                                 int kept_count,
                                 int **env_modes_out)
{
    if (env_modes_out == NULL)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *env_modes_out = NULL;
    if (n <= 0 || kept_modes == NULL || kept_count < 0 || kept_count > n)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    unsigned char *kept = static_cast<unsigned char *>(
        calloc(static_cast<size_t>(n), sizeof(unsigned char)));
    if (kept == NULL)
    {
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    for (int k = 0; k < kept_count; ++k)
    {
        const int q = kept_modes[k];
        if (q < 0 || q >= n || kept[q] != 0)
        {
            free(kept);
            return static_cast<int>(cudaErrorInvalidValue);
        }
        kept[q] = 1;
    }

    const int env_count = n - kept_count;
    int *env_modes =
        static_cast<int *>(malloc(static_cast<size_t>(env_count) * sizeof(int)));
    if (env_count > 0 && env_modes == NULL)
    {
        free(kept);
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    int out = 0;
    for (int q = 0; q < n; ++q)
    {
        if (!kept[q])
        {
            env_modes[out++] = q;
        }
    }
    free(kept);
    *env_modes_out = env_modes;
    return 0;
}

// Jordan-Wigner reordering sign for the fermionic trace, tabulated per Schmidt
// row: moving retained mode k past the environment modes below it flips the
// sign once for each occupied lower environment mode.
inline uint64_t *fermion_cross_masks_by_row_host(const int *kept_modes,
                                                 int kept_count,
                                                 const int *env_modes,
                                                 int env_count,
                                                 size_t rows)
{
    if (kept_modes == NULL || env_modes == NULL || kept_count < 0 || env_count < 0)
    {
        return NULL;
    }

    uint64_t *masks = static_cast<uint64_t *>(calloc(rows, sizeof(uint64_t)));
    if (masks == NULL)
    {
        return NULL;
    }
    uint64_t *lower_env_mask = static_cast<uint64_t *>(
        calloc(static_cast<size_t>(kept_count > 0 ? kept_count : 1), sizeof(uint64_t)));
    if (lower_env_mask == NULL)
    {
        free(masks);
        return NULL;
    }

    for (int k = 0; k < kept_count; ++k)
    {
        uint64_t mask = 0;
        for (int e = 0; e < env_count; ++e)
        {
            if (env_modes[e] < kept_modes[k])
            {
                mask |= uint64_t{1} << e;
            }
        }
        lower_env_mask[k] = mask;
    }
    for (size_t row = 0; row < rows; ++row)
    {
        uint64_t mask = 0;
        for (int k = 0; k < kept_count; ++k)
        {
            if ((static_cast<uint64_t>(row) >> k) & uint64_t{1})
            {
                mask ^= lower_env_mask[k];
            }
        }
        masks[row] = mask;
    }
    free(lower_env_mask);
    return masks;
}

// --- device kernels ---------------------------------------------------------

// Two layouts, because coalescing depends on which mode set owns the low bits
// of the state index.
//
// The state index is env_offsets[col] | row_offsets[row].  Whichever of the two
// offset tables contains mode 0 advances by 1 as its index increments; the
// other advances by 2^(position of its lowest mode).  So the index that varies
// fastest across a warp has to be the one whose table is unit-stride, or every
// lane in the warp lands on a different cache line.
//
// Getting this wrong is expensive and easy to miss: with the retained index
// always fastest, TMI half-cuts (which retain modes 0..N/2-1) stayed coalesced
// while the B/C/D quarter blocks did not, and sim_tmi at N=20 ran 2.6x *slower*
// than the kernel this replaced even though it moved 30x fewer bytes.
//
// KEPT_FASTEST=true  -> M  is rows x cols, column-major, herk uses OP_N.
// KEPT_FASTEST=false -> M^T is cols x rows, column-major, herk uses OP_C;
//                       A^H A is the same Gram matrix.
template <typename Real, typename ComplexT, bool KEPT_FASTEST>
__global__ void build_schmidt_matrix_kernel(const Cx<Real> *__restrict__ psi,
                                            ComplexT *__restrict__ matrix,
                                            size_t elements,
                                            size_t rows,
                                            size_t cols,
                                            const uint64_t *__restrict__ row_offsets,
                                            const uint64_t *__restrict__ env_offsets,
                                            const uint64_t *__restrict__ row_cross_masks,
                                            int fermion_trace)
{
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        size_t row;
        size_t col;
        if (KEPT_FASTEST)
        {
            row = index % rows;
            col = index / rows;
        }
        else
        {
            col = index % cols;
            row = index / cols;
        }
        const uint64_t src = env_offsets[col] | row_offsets[row];
        const Cx<Real> value = psi[src];
        const Real sign =
            (fermion_trace &&
             odd_popcount64(static_cast<uint64_t>(col) & row_cross_masks[row]))
                ? Real(-1)
                : Real(1);
        matrix[index].x = sign * value.re;
        // The transposed layout stores M^H, not M^T.
        //
        // herk OP_C computes A^H A.  Feeding it a plain M^T would give
        // (M^T)^H M^T = conj(M) M^T = conj(rho) -- Hermitian, correctly traced,
        // and with every off-diagonal conjugated, which is invisible to the
        // entropy (conj(rho) is similar to rho) but wrong for anything that
        // exports the matrix. Storing conj(M[row][col]) makes A = M^H, so
        // A^H A = M M^H = rho exactly.
        matrix[index].y = KEPT_FASTEST ? (sign * value.im) : (-sign * value.im);
    }
}

// cuBLAS herk writes only the triangle it is asked for, so the other one keeps
// whatever the workspace held before -- and since cusolverDn?heevd overwrites
// its input, that is the tridiagonalized debris of the previous call.  Passing
// such a matrix to heevd makes it fail to converge (info = n-1), which
// sim_tmi.exe treats as "device SVD is broken", disables the device half-cut
// path for the rest of the run, and silently falls back to host LAPACK.
//
// Mirroring the lower triangle into the upper one makes the buffer genuinely
// Hermitian, which costs one pass over rows^2 and removes the dependence on
// exactly which triangle the solver decides to read.
template <typename ComplexT>
__global__ void symmetrize_lower_to_upper_kernel(ComplexT *__restrict__ gram, int dim)
{
    const long long total = static_cast<long long>(dim) * dim;
    for (long long index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        const int col = static_cast<int>(index / dim);
        const int row = static_cast<int>(index - static_cast<long long>(col) * dim);
        if (row < col)
        {
            // Column-major: (row, col) in the strict upper triangle is the
            // conjugate of (col, row), which herk populated.
            const ComplexT source = gram[static_cast<size_t>(row) * dim + col];
            ComplexT value;
            value.x = source.x;
            value.y = -source.y;
            gram[index] = value;
        }
        else if (row == col)
        {
            gram[index].y = 0;
        }
    }
}

// Mirror the lower triangle into the
// upper one and emit the row-major interleaved fp64 layout the host consumers
// expect (rho_ri[2*(r*dim+c)] = Re, +1 = Im).
//
// A column-major Hermitian matrix reinterpreted row-major is its own conjugate
// transpose, which for a Hermitian matrix is itself -- so the column-major
// lower triangle supplies the row-major upper triangle directly.
template <typename ComplexT>
__global__ void gram_to_row_major_ri_kernel(const ComplexT *__restrict__ gram,
                                            int dim,
                                            double *__restrict__ rho_ri)
{
    const int total = dim * dim;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += gridDim.x * blockDim.x)
    {
        const int row = index / dim;
        const int col = index - row * dim;

        // Read from the populated (lower, column-major) triangle: element
        // (i, j) with i >= j lives at gram[j * dim + i].
        double re;
        double im;
        if (row >= col)
        {
            const ComplexT value = gram[static_cast<size_t>(col) * dim + row];
            re = static_cast<double>(value.x);
            im = static_cast<double>(value.y);
            // gram[j*dim+i] is column-major (i,j) = rho(i,j) with i >= j.
        }
        else
        {
            const ComplexT value = gram[static_cast<size_t>(row) * dim + col];
            re = static_cast<double>(value.x);
            im = -static_cast<double>(value.y);
        }
        if (row == col)
        {
            im = 0.0;
        }
        rho_ri[2 * static_cast<size_t>(index) + 0] = re;
        rho_ri[2 * static_cast<size_t>(index) + 1] = im;
    }
}

// Promote a Gram matrix to fp64 for the retry eigensolve.
template <typename ComplexT>
__global__ void promote_gram_to_f64_kernel(const ComplexT *__restrict__ src,
                                           cuDoubleComplex *__restrict__ dst,
                                           long long elements)
{
    for (long long index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        dst[index].x = static_cast<double>(src[index].x);
        dst[index].y = static_cast<double>(src[index].y);
    }
}

// --- cuBLAS wrappers --------------------------------------------------------

inline cublasStatus_t herk(cublasHandle_t handle,
                           cublasOperation_t op,
                           int n,
                           int k,
                           const cuComplex *a,
                           int lda,
                           cuComplex *c,
                           int ldc)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasCherk(handle, CUBLAS_FILL_MODE_LOWER, op,
                       n, k, &alpha, a, lda, &beta, c, ldc);
}

inline cublasStatus_t herk(cublasHandle_t handle,
                           cublasOperation_t op,
                           int n,
                           int k,
                           const cuDoubleComplex *a,
                           int lda,
                           cuDoubleComplex *c,
                           int ldc)
{
    const double alpha = 1.0;
    const double beta = 0.0;
    return cublasZherk(handle, CUBLAS_FILL_MODE_LOWER, op,
                       n, k, &alpha, a, lda, &beta, c, ldc);
}

inline int thread_cublas_handle(cublasHandle_t &handle)
{
    static thread_local cublasHandle_t tls_handle = NULL;
    if (tls_handle == NULL)
    {
        const cublasStatus_t status = cublasCreate(&tls_handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            return 200000 + static_cast<int>(status);
        }
    }
    handle = tls_handle;
    return 0;
}

// --- combined gather + Gram -------------------------------------------------

// The Schmidt index tables depend only on (n, kept_modes, fermion_trace), and
// callers cycle through a handful of fixed retained sets -- sim_tmi.exe repeats
// AB / AC / BC / D once per sample.  Rebuilding and re-uploading three tables
// per call cost three H2D copies each time, which dominates at small N where
// the actual reduction is microseconds.  Cache them per retained set.
enum
{
    GRAM_CACHE_SLOTS = 8,
    GRAM_CACHE_MAX_MODES = 32
};

template <typename ComplexT>
struct GramWorkspace
{
    // Scratch shared by every retained set; overwritten on each call.
    ComplexT *matrix = NULL;
    size_t matrix_capacity = 0;
    ComplexT *gram = NULL;
    size_t gram_capacity = 0;

    struct Slot
    {
        bool valid = false;
        int n = 0;
        int kept_count = 0;
        int fermion_trace = 0;
        int modes[GRAM_CACHE_MAX_MODES] = {0};
        uint64_t *row_offsets = NULL;
        size_t row_capacity = 0;
        uint64_t *env_offsets = NULL;
        size_t env_capacity = 0;
        uint64_t *row_cross_masks = NULL;
        size_t mask_capacity = 0;
    };
    Slot slots[GRAM_CACHE_SLOTS];
    int next_slot = 0;
};

template <typename ComplexT>
inline typename GramWorkspace<ComplexT>::Slot *find_cached_slot(
    GramWorkspace<ComplexT> &ws, int n, const int *kept_modes, int kept_count,
    int fermion_trace)
{
    if (kept_count > GRAM_CACHE_MAX_MODES)
    {
        return NULL;
    }
    for (int i = 0; i < GRAM_CACHE_SLOTS; ++i)
    {
        typename GramWorkspace<ComplexT>::Slot &slot = ws.slots[i];
        if (!slot.valid || slot.n != n || slot.kept_count != kept_count ||
            slot.fermion_trace != fermion_trace)
        {
            continue;
        }
        bool same = true;
        for (int k = 0; k < kept_count; ++k)
        {
            if (slot.modes[k] != kept_modes[k])
            {
                same = false;
                break;
            }
        }
        if (same)
        {
            return &slot;
        }
    }
    return NULL;
}

// Gathers the Schmidt matrix and forms rho = M M^H into ws.gram (lower triangle,
// column-major, leading dimension `rows`).  Returns 0 on success.
template <typename Real, typename ComplexT>
inline int build_gram(const void *device_state_vector,
                      int n_qubits,
                      const int *host_kept_modes,
                      int kept_count,
                      int fermion_trace,
                      GramWorkspace<ComplexT> &ws,
                      size_t &rows_out,
                      size_t &cols_out)
{
    if (device_state_vector == NULL || host_kept_modes == NULL ||
        n_qubits <= 0 || n_qubits >= 63 || kept_count <= 0 ||
        kept_count >= n_qubits || kept_count >= 31)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const size_t rows = size_t{1} << kept_count;
    const size_t cols = size_t{1} << (n_qubits - kept_count);
    rows_out = rows;
    cols_out = cols;

    cudaError_t cuda_status = cudaSuccess;
    if (thread_buffer(rows * cols, ws.matrix, ws.matrix_capacity, cuda_status) == NULL ||
        thread_buffer(rows * rows, ws.gram, ws.gram_capacity, cuda_status) == NULL)
    {
        return static_cast<int>(cuda_status);
    }

    typename GramWorkspace<ComplexT>::Slot *slot =
        find_cached_slot(ws, n_qubits, host_kept_modes, kept_count, fermion_trace);
    if (slot == NULL)
    {
        // Round-robin eviction: callers cycle through a fixed set, so anything
        // more elaborate would never pay for itself.
        slot = &ws.slots[ws.next_slot];
        ws.next_slot = (ws.next_slot + 1) % GRAM_CACHE_SLOTS;
        slot->valid = false;

        int *env_modes = NULL;
        int status =
            complement_modes_host(n_qubits, host_kept_modes, kept_count, &env_modes);
        if (status != 0)
        {
            return status;
        }
        const int env_count = n_qubits - kept_count;

        uint64_t *row_offsets = basis_offsets_host(host_kept_modes, kept_count);
        uint64_t *env_offsets = basis_offsets_host(env_modes, env_count);
        uint64_t *row_cross_masks =
            fermion_trace
                ? fermion_cross_masks_by_row_host(host_kept_modes, kept_count,
                                                  env_modes, env_count, rows)
                : static_cast<uint64_t *>(calloc(rows, sizeof(uint64_t)));

        if (row_offsets == NULL || env_offsets == NULL || row_cross_masks == NULL)
        {
            status = static_cast<int>(cudaErrorMemoryAllocation);
        }
        if (status == 0 &&
            (thread_buffer(rows, slot->row_offsets, slot->row_capacity, cuda_status) == NULL ||
             thread_buffer(cols, slot->env_offsets, slot->env_capacity, cuda_status) == NULL ||
             thread_buffer(rows, slot->row_cross_masks, slot->mask_capacity, cuda_status) == NULL))
        {
            status = static_cast<int>(cuda_status);
        }
        if (status == 0)
        {
            cuda_status = cudaMemcpy(slot->row_offsets, row_offsets,
                                     rows * sizeof(uint64_t), cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess) { status = static_cast<int>(cuda_status); }
        }
        if (status == 0)
        {
            cuda_status = cudaMemcpy(slot->env_offsets, env_offsets,
                                     cols * sizeof(uint64_t), cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess) { status = static_cast<int>(cuda_status); }
        }
        if (status == 0)
        {
            cuda_status = cudaMemcpy(slot->row_cross_masks, row_cross_masks,
                                     rows * sizeof(uint64_t), cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess) { status = static_cast<int>(cuda_status); }
        }

        free(env_modes);
        free(row_offsets);
        free(env_offsets);
        free(row_cross_masks);
        if (status != 0)
        {
            return status;
        }

        slot->n = n_qubits;
        slot->kept_count = kept_count;
        slot->fermion_trace = fermion_trace;
        for (int k = 0; k < kept_count; ++k)
        {
            slot->modes[k] = host_kept_modes[k];
        }
        slot->valid = true;
    }

    // Put the unit-stride mode set on the fastest-varying thread index.  Mode 0
    // belongs to exactly one of the two sets, and that set's offset table is the
    // one that advances by 1.
    bool kept_owns_mode_zero = false;
    for (int k = 0; k < kept_count; ++k)
    {
        if (host_kept_modes[k] == 0)
        {
            kept_owns_mode_zero = true;
            break;
        }
    }

    const size_t elements = rows * cols;
    const size_t wanted = (elements + GATHER_THREADS - 1) / GATHER_THREADS;
    const int blocks = static_cast<int>(wanted < 65535 ? wanted : size_t{65535});
    if (kept_owns_mode_zero)
    {
        build_schmidt_matrix_kernel<Real, ComplexT, true><<<blocks, GATHER_THREADS>>>(
            reinterpret_cast<const Cx<Real> *>(device_state_vector),
            ws.matrix, elements, rows, cols,
            slot->row_offsets, slot->env_offsets, slot->row_cross_masks, fermion_trace);
    }
    else
    {
        build_schmidt_matrix_kernel<Real, ComplexT, false><<<blocks, GATHER_THREADS>>>(
            reinterpret_cast<const Cx<Real> *>(device_state_vector),
            ws.matrix, elements, rows, cols,
            slot->row_offsets, slot->env_offsets, slot->row_cross_masks, fermion_trace);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return static_cast<int>(cuda_status);
    }

    cublasHandle_t handle = NULL;
    const int handle_status = thread_cublas_handle(handle);
    if (handle_status != 0)
    {
        return handle_status;
    }

    // OP_N consumes M (rows x cols, lda = rows); OP_C consumes M^T
    // (cols x rows, lda = cols).  Both produce rho = M M^H.
    const cublasStatus_t blas_status =
        kept_owns_mode_zero
            ? herk(handle, CUBLAS_OP_N, static_cast<int>(rows), static_cast<int>(cols),
                   ws.matrix, static_cast<int>(rows), ws.gram, static_cast<int>(rows))
            : herk(handle, CUBLAS_OP_C, static_cast<int>(rows), static_cast<int>(cols),
                   ws.matrix, static_cast<int>(cols), ws.gram, static_cast<int>(rows));
    if (blas_status != CUBLAS_STATUS_SUCCESS)
    {
        return 200000 + static_cast<int>(blas_status);
    }

    const size_t gram_elements = rows * rows;
    const size_t gram_wanted = (gram_elements + GATHER_THREADS - 1) / GATHER_THREADS;
    const int gram_blocks =
        static_cast<int>(gram_wanted < 65535 ? gram_wanted : size_t{65535});
    symmetrize_lower_to_upper_kernel<ComplexT><<<gram_blocks, GATHER_THREADS>>>(
        ws.gram, static_cast<int>(rows));
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return static_cast<int>(cuda_status);
    }

    // Required, and not merely defensive.  cusolverDn?heevd was observed
    // reading this buffer before the symmetrize kernel above had filled its
    // upper triangle, picking up the uninitialized half, and returning
    // "did not converge".  sim_tmi.exe reads that as a broken device solver and
    // permanently falls back to host LAPACK, so the symptom was a silent 2x
    // slowdown rather than a wrong number.  One sync per call is nothing next
    // to the eigensolve it precedes.
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
    {
        return static_cast<int>(cuda_status);
    }
    return 0;
}
} // anonymous namespace: per-TU kernel copies
} // namespace mipt_gram
