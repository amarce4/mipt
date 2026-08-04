#include "mipt/cuda/entropy_cuda.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Minimal cuSOLVER declarations, for the von Neumann (S1) path only. Do not
// include <cusolverDn.h> here: some CUDA 11.x + GCC 11 setups fail while
// parsing that header through nvcc, and tmi_cuda_svd.cu already declares the
// same entry points by hand for the same reason. The C ABI is stable and the
// enum arguments are int-sized.
extern "C" {
struct cusolverDnContext;
typedef cusolverDnContext *cusolverDnHandle_t;
typedef int cusolverStatus_t;

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);

cusolverStatus_t cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle, int jobz,
                                             int uplo, int n, const cuComplex *A,
                                             int lda, const float *W, int *lwork);
cusolverStatus_t cusolverDnCheevd(cusolverDnHandle_t handle, int jobz, int uplo,
                                  int n, cuComplex *A, int lda, float *W,
                                  cuComplex *work, int lwork, int *devInfo);
cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle, int jobz,
                                             int uplo, int n,
                                             const cuDoubleComplex *A, int lda,
                                             const double *W, int *lwork);
cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t handle, int jobz, int uplo,
                                  int n, cuDoubleComplex *A, int lda, double *W,
                                  cuDoubleComplex *work, int lwork, int *devInfo);

cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *lwork);
cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu,
                                  signed char jobvt, int m, int n, cuComplex *A,
                                  int lda, float *S, cuComplex *U, int ldu,
                                  cuComplex *VT, int ldvt, cuComplex *work,
                                  int lwork, float *rwork, int *devInfo);
cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, int *lwork);
cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu,
                                  signed char jobvt, int m, int n,
                                  cuDoubleComplex *A, int lda, double *S,
                                  cuDoubleComplex *U, int ldu,
                                  cuDoubleComplex *VT, int ldvt,
                                  cuDoubleComplex *work, int lwork,
                                  double *rwork, int *devInfo);
}

#ifndef CUSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_SUCCESS 0
#endif

namespace
{
enum
{
    // cusolverEigMode_t / cublasFillMode_t, which are int-sized enums.
    EIG_MODE_NOVECTOR = 0,
    FILL_MODE_LOWER = 0
};

    // This CUDA translation unit deliberately avoids C++ standard-library
    // containers, strings, and streams. nvcc normally compiles those against
    // libstdc++, while CUDA-Q's nvq++ driver links against libc++; mixing the
    // two ABIs caused unresolved std::__cxx11 and iostream symbols at link time.
    thread_local char g_last_error[512] = {0};

    void clear_error()
    {
        g_last_error[0] = '\0';
    }

    void set_error(const char *message)
    {
        if (message == nullptr)
        {
            message = "Unknown CUDA entropy error.";
        }
        std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
    }

    void set_cuda_error(const char *what, cudaError_t status)
    {
        std::snprintf(g_last_error, sizeof(g_last_error),
                      "%s failed: %s (status=%d)", what,
                      cudaGetErrorString(status), static_cast<int>(status));
    }

    void set_cublas_error(const char *what, cublasStatus_t status)
    {
        std::snprintf(g_last_error, sizeof(g_last_error),
                      "%s failed with cuBLAS status=%d", what,
                      static_cast<int>(status));
    }

    std::size_t min_size(std::size_t a, std::size_t b)
    {
        return a < b ? a : b;
    }

    std::size_t max_size(std::size_t a, std::size_t b)
    {
        return a > b ? a : b;
    }

    int min_int(int a, int b)
    {
        return a < b ? a : b;
    }

    int max_int(int a, int b)
    {
        return a > b ? a : b;
    }

    std::size_t max_workspace_bytes()
    {
        constexpr std::size_t default_mb = 512;
        const char *env = std::getenv("MIPT_ENTROPY_CUDA_MAX_MB");
        if (env == nullptr || *env == '\0')
        {
            return default_mb * std::size_t{1024} * std::size_t{1024};
        }
        char *end = nullptr;
        const unsigned long long mb = std::strtoull(env, &end, 10);
        if (end == env || mb == 0)
        {
            return default_mb * std::size_t{1024} * std::size_t{1024};
        }
        const std::size_t mib = std::size_t{1024} * std::size_t{1024};
        const unsigned long long max_mb =
            static_cast<unsigned long long>(static_cast<std::size_t>(-1) / mib);
        const unsigned long long clamped = mb < max_mb ? mb : max_mb;
        return static_cast<std::size_t>(clamped) * mib;
    }

    // Largest RDM dimension the divide-and-conquer eigensolver is trusted with.
    //
    // Same caveat tmi_cuda_svd.cu records: cusolverDn?heevd stops converging on
    // physical trajectory RDMs once the matrix gets large (it returns info != 0
    // at 1024 on inputs verified exactly Hermitian, trace-1 and NaN-free, in
    // both fp32 and fp64), while gesvd handles the same matrices. Cap the fast
    // path and keep gesvd for the rest.
    int heevd_max_rows()
    {
        static int cap = -1;
        if (cap < 0)
        {
            const char *env = std::getenv("MIPT_ENTROPY_HEEVD_MAX_ROWS");
            if (env == nullptr || *env == '\0')
            {
                cap = 512;
            }
            else
            {
                const long value = std::strtol(env, nullptr, 10);
                cap = (value < 0) ? 0 : static_cast<int>(value);
            }
        }
        return cap;
    }

    // Dimensions where heevd has already failed once, per precision. Retrying it
    // at every subsystem size of every trajectory would cost a wasted
    // eigensolve each, and heevd fails deterministically on a given size rather
    // than sporadically.
    //
    // These are not hypothetical. On a GTX 1650 with CUDA 12.4 the fp32 Cheevd
    // stops converging on physical trajectory RDMs at dim=256 -- roughly where
    // sim_tmi sees it, allowing for the different matrices -- while fp64 Zheevd
    // handles the same states at 8 ms against gesvd's 55 ms. Hence the ordering
    // below: native precision, then fp64, then gesvd.
    bool heevd_known_bad(int dim, bool double_precision, bool record)
    {
        static thread_local int bad[2][32] = {{0}};
        static thread_local int count[2] = {0, 0};
        const int slot = double_precision ? 1 : 0;
        for (int i = 0; i < count[slot]; ++i)
        {
            if (bad[slot][i] == dim) { return true; }
        }
        if (record && count[slot] < 32)
        {
            bad[slot][count[slot]++] = dim;
            // A silent demotion here reads as an unexplained slowdown, which is
            // exactly the trap sim_tmi's "half-cut SVD was disabled" warning
            // exists to avoid. Say so once per (dimension, precision).
            std::fprintf(stderr,
                         "\nentropy: cuSOLVER %sheevd did not converge on the %dx%d "
                         "von Neumann RDM; %s for this size from now on.\n",
                         double_precision ? "Z" : "C", dim, dim,
                         double_precision ? "falling back to gesvd"
                                          : "retrying in fp64");
        }
        return false;
    }

    template <typename T>
    struct SolverTraits;

    template <>
    struct SolverTraits<cuFloatComplex>
    {
        using Real = float;
        static constexpr bool is_single = true;
        // Eigenvalues at or below the fp32 Gram noise floor are dropped: the
        // logarithm of a negative roundoff eigenvalue would poison the sum.
        static double eigenvalue_floor() { return 1.0e-12; }
    };

    template <>
    struct SolverTraits<cuDoubleComplex>
    {
        using Real = double;
        static constexpr bool is_single = false;
        static double eigenvalue_floor() { return 1.0e-15; }
    };

    __global__ void promote_to_f64_kernel(const cuFloatComplex *input,
                                          cuDoubleComplex *output,
                                          std::uint64_t count)
    {
        const std::uint64_t index =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= count)
        {
            return;
        }
        const cuFloatComplex value = input[index];
        output[index] = make_cuDoubleComplex(static_cast<double>(value.x),
                                             static_cast<double>(value.y));
    }

    cusolverStatus_t heevd_buffer_size(cusolverDnHandle_t handle, int n,
                                       const cuFloatComplex *a, const float *w,
                                       int *lwork)
    {
        return cusolverDnCheevd_bufferSize(handle, EIG_MODE_NOVECTOR,
                                           FILL_MODE_LOWER, n, a, n, w, lwork);
    }

    cusolverStatus_t heevd_buffer_size(cusolverDnHandle_t handle, int n,
                                       const cuDoubleComplex *a, const double *w,
                                       int *lwork)
    {
        return cusolverDnZheevd_bufferSize(handle, EIG_MODE_NOVECTOR,
                                           FILL_MODE_LOWER, n, a, n, w, lwork);
    }

    cusolverStatus_t heevd(cusolverDnHandle_t handle, int n, cuFloatComplex *a,
                           float *w, cuFloatComplex *work, int lwork, int *info)
    {
        return cusolverDnCheevd(handle, EIG_MODE_NOVECTOR, FILL_MODE_LOWER, n, a,
                                n, w, work, lwork, info);
    }

    cusolverStatus_t heevd(cusolverDnHandle_t handle, int n, cuDoubleComplex *a,
                           double *w, cuDoubleComplex *work, int lwork, int *info)
    {
        return cusolverDnZheevd(handle, EIG_MODE_NOVECTOR, FILL_MODE_LOWER, n, a,
                                n, w, work, lwork, info);
    }

    cusolverStatus_t gesvd_buffer_size(cusolverDnHandle_t handle, int n,
                                       const cuFloatComplex *, int *lwork)
    {
        return cusolverDnCgesvd_bufferSize(handle, n, n, lwork);
    }

    cusolverStatus_t gesvd_buffer_size(cusolverDnHandle_t handle, int n,
                                       const cuDoubleComplex *, int *lwork)
    {
        return cusolverDnZgesvd_bufferSize(handle, n, n, lwork);
    }

    // rho is Hermitian positive semidefinite, so its singular values *are* its
    // eigenvalues and gesvd is a drop-in replacement for heevd on the same
    // buffer. Both destroy their input.
    cusolverStatus_t gesvd_novectors(cusolverDnHandle_t handle, int n,
                                     cuFloatComplex *a, float *s,
                                     cuFloatComplex *work, int lwork,
                                     float *rwork, int *info)
    {
        return cusolverDnCgesvd(handle, 'N', 'N', n, n, a, n, s, nullptr, 1,
                                nullptr, 1, work, lwork, rwork, info);
    }

    cusolverStatus_t gesvd_novectors(cusolverDnHandle_t handle, int n,
                                     cuDoubleComplex *a, double *s,
                                     cuDoubleComplex *work, int lwork,
                                     double *rwork, int *info)
    {
        return cusolverDnZgesvd(handle, 'N', 'N', n, n, a, n, s, nullptr, 1,
                                nullptr, 1, work, lwork, rwork, info);
    }

    template <typename T>
    struct CudaComplexTraits;

    template <>
    struct CudaComplexTraits<cuFloatComplex>
    {
        static __device__ cuFloatComplex signed_value(cuFloatComplex value, int sign)
        {
            if (sign < 0)
            {
                value.x = -value.x;
                value.y = -value.y;
            }
            return value;
        }

        static __device__ cuFloatComplex add(cuFloatComplex a, cuFloatComplex b)
        {
            return make_cuFloatComplex(a.x + b.x, a.y + b.y);
        }

        static __device__ double real(cuFloatComplex value)
        {
            return static_cast<double>(value.x);
        }

        static __device__ double norm(cuFloatComplex value)
        {
            const double x = static_cast<double>(value.x);
            const double y = static_cast<double>(value.y);
            return x * x + y * y;
        }

        static __device__ double purity_tolerance()
        {
            return 1e-4;
        }
    };

    template <>
    struct CudaComplexTraits<cuDoubleComplex>
    {
        static __device__ cuDoubleComplex signed_value(cuDoubleComplex value, int sign)
        {
            if (sign < 0)
            {
                value.x = -value.x;
                value.y = -value.y;
            }
            return value;
        }

        static __device__ cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b)
        {
            return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
        }

        static __device__ double real(cuDoubleComplex value)
        {
            return value.x;
        }

        static __device__ double norm(cuDoubleComplex value)
        {
            return value.x * value.x + value.y * value.y;
        }

        static __device__ double purity_tolerance()
        {
            return 1e-10;
        }
    };

    __device__ int fermionic_sign_from_index(std::uint64_t occupation_mask,
                                              const int *position_in_new_order,
                                              int n)
    {
        std::uint64_t seen_positions = 0;
        int parity = 0;
        for (int mode = 0; mode < n; ++mode)
        {
            if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
            {
                continue;
            }
            const int pos = position_in_new_order[mode];
            const std::uint64_t lower_or_equal =
                (std::uint64_t{1} << (pos + 1)) - std::uint64_t{1};
            parity ^= (__popcll(seen_positions & ~lower_or_equal) & 1);
            seen_positions |= std::uint64_t{1} << pos;
        }
        return parity ? -1 : 1;
    }

    template <typename ComplexT>
    __global__ void gather_cyclic_matrices_kernel(const ComplexT *state,
                                                   int n,
                                                   int kept,
                                                   std::uint64_t total_dim,
                                                   int translation_count,
                                                   const int *starts,
                                                   const int *positions,
                                                   int fermionic_trace,
                                                   ComplexT *matrices)
    {
        const std::uint64_t linear =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::uint64_t total =
            total_dim * static_cast<std::uint64_t>(translation_count);
        if (linear >= total)
        {
            return;
        }

        const int batch = static_cast<int>(linear / total_dim);
        const std::uint64_t old_index = linear - static_cast<std::uint64_t>(batch) * total_dim;
        const int start = starts[batch];

        std::uint64_t row = 0;
        for (int local = 0; local < kept; ++local)
        {
            const int mode = (start + local) % n;
            row |= ((old_index >> mode) & std::uint64_t{1}) << local;
        }

        std::uint64_t col = 0;
        int env_bit = 0;
        for (int mode = 0; mode < n; ++mode)
        {
            const int distance = (mode - start + n) % n;
            if (distance < kept)
            {
                continue;
            }
            col |= ((old_index >> mode) & std::uint64_t{1}) << env_bit;
            ++env_bit;
        }

        int sign = 1;
        if (fermionic_trace)
        {
            sign = fermionic_sign_from_index(
                old_index,
                positions + static_cast<std::size_t>(batch) * static_cast<std::size_t>(n),
                n);
        }

        const std::uint64_t retained_dim = std::uint64_t{1} << kept;
        const std::uint64_t matrix_offset =
            static_cast<std::uint64_t>(batch) * total_dim + col * retained_dim + row;
        matrices[matrix_offset] =
            CudaComplexTraits<ComplexT>::signed_value(state[old_index], sign);
    }

    template <typename ComplexT>
    __global__ void partial_trace_trailing_mode_kernel(const ComplexT *input,
                                                        int input_dim,
                                                        int translation_count,
                                                        ComplexT *output)
    {
        const int output_dim = input_dim >> 1;
        const std::uint64_t input_elements =
            static_cast<std::uint64_t>(input_dim) * static_cast<std::uint64_t>(input_dim);
        const std::uint64_t output_elements =
            static_cast<std::uint64_t>(output_dim) * static_cast<std::uint64_t>(output_dim);
        const std::uint64_t linear =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::uint64_t total =
            output_elements * static_cast<std::uint64_t>(translation_count);
        if (linear >= total)
        {
            return;
        }

        const int batch = static_cast<int>(linear / output_elements);
        const std::uint64_t local =
            linear - static_cast<std::uint64_t>(batch) * output_elements;
        const int row = static_cast<int>(local % static_cast<std::uint64_t>(output_dim));
        const int col = static_cast<int>(local / static_cast<std::uint64_t>(output_dim));

        const std::uint64_t input_base =
            static_cast<std::uint64_t>(batch) * input_elements;
        const std::uint64_t output_base =
            static_cast<std::uint64_t>(batch) * output_elements;
        const std::uint64_t index_zero =
            input_base + static_cast<std::uint64_t>(col) * input_dim + row;
        const std::uint64_t index_one =
            input_base + static_cast<std::uint64_t>(col + output_dim) * input_dim +
            static_cast<std::uint64_t>(row + output_dim);

        output[output_base + local] =
            CudaComplexTraits<ComplexT>::add(input[index_zero], input[index_one]);
    }

    template <typename ComplexT>
    __global__ void purity_kernel(const ComplexT *rho,
                                  std::uint64_t rho_elements_per_matrix,
                                  int matrix_dim,
                                  int translation_count,
                                  int result_offset,
                                  double *results,
                                  int *numeric_error)
    {
        const int batch = static_cast<int>(blockIdx.x);
        if (batch >= translation_count)
        {
            return;
        }

        double local_trace = 0.0;
        double local_frobenius = 0.0;
        const ComplexT *matrix =
            rho + static_cast<std::uint64_t>(batch) * rho_elements_per_matrix;

        for (std::uint64_t index = threadIdx.x;
             index < rho_elements_per_matrix;
             index += blockDim.x)
        {
            const ComplexT value = matrix[index];
            local_frobenius += CudaComplexTraits<ComplexT>::norm(value);
            const int row = static_cast<int>(index % static_cast<std::uint64_t>(matrix_dim));
            const int col = static_cast<int>(index / static_cast<std::uint64_t>(matrix_dim));
            if (row == col)
            {
                local_trace += CudaComplexTraits<ComplexT>::real(value);
            }
        }

        __shared__ double trace_shared[256];
        __shared__ double frobenius_shared[256];
        trace_shared[threadIdx.x] = local_trace;
        frobenius_shared[threadIdx.x] = local_frobenius;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                trace_shared[threadIdx.x] += trace_shared[threadIdx.x + stride];
                frobenius_shared[threadIdx.x] += frobenius_shared[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            const double trace = trace_shared[0];
            double purity = frobenius_shared[0] / (trace * trace);
            if (!(trace > 0.0) || !isfinite(trace) || !(purity > 0.0) || !isfinite(purity) ||
                purity > 1.0 + CudaComplexTraits<ComplexT>::purity_tolerance())
            {
                atomicExch(numeric_error, 1);
                // The numeric_error flag is authoritative; avoid CUDART_NAN,
                // which is not defined by all CUDA toolkit/header combinations.
                results[result_offset + batch] = 0.0;
                return;
            }
            purity = fmin(1.0, purity);
            results[result_offset + batch] = -log(purity);
        }
    }

    struct Workspace
    {
        cublasHandle_t handle = nullptr;
        void *matrices = nullptr;
        void *rho = nullptr;
        int *starts = nullptr;
        int *positions = nullptr;
        double *results = nullptr;
        int *numeric_error = nullptr;
        std::size_t matrices_bytes = 0;
        std::size_t rho_bytes = 0;
        std::size_t starts_bytes = 0;
        std::size_t positions_bytes = 0;
        std::size_t results_bytes = 0;

        // S1 only. The eigensolvers destroy their input, so the RDM batch is
        // copied into `solver` and the RDMs themselves stay intact for the
        // recursive partial trace that produces the next subsystem size.
        cusolverDnHandle_t solver_handle = nullptr;
        void *solver = nullptr;
        void *eigenvalues = nullptr;
        void *solver_work = nullptr;
        void *solver_rwork = nullptr;
        int *solver_info = nullptr;
        std::size_t solver_bytes = 0;
        std::size_t eigenvalues_bytes = 0;
        std::size_t solver_work_bytes = 0;
        std::size_t solver_rwork_bytes = 0;
        std::size_t solver_info_bytes = 0;
        // Allocated only if an fp32 eigensolve actually fails. Bounded by
        // MIPT_ENTROPY_HEEVD_MAX_ROWS, which is why it stays outside the
        // MIPT_ENTROPY_CUDA_MAX_MB batching budget.
        void *solver64 = nullptr;
        void *eigenvalues64 = nullptr;
        std::size_t solver64_bytes = 0;
        std::size_t eigenvalues64_bytes = 0;

        ~Workspace()
        {
            if (matrices != nullptr) cudaFree(matrices);
            if (rho != nullptr) cudaFree(rho);
            if (starts != nullptr) cudaFree(starts);
            if (positions != nullptr) cudaFree(positions);
            if (results != nullptr) cudaFree(results);
            if (numeric_error != nullptr) cudaFree(numeric_error);
            if (solver != nullptr) cudaFree(solver);
            if (eigenvalues != nullptr) cudaFree(eigenvalues);
            if (solver64 != nullptr) cudaFree(solver64);
            if (eigenvalues64 != nullptr) cudaFree(eigenvalues64);
            if (solver_work != nullptr) cudaFree(solver_work);
            if (solver_rwork != nullptr) cudaFree(solver_rwork);
            if (solver_info != nullptr) cudaFree(solver_info);
            if (solver_handle != nullptr) cusolverDnDestroy(solver_handle);
            if (handle != nullptr) cublasDestroy(handle);
        }
    };

    Workspace &workspace()
    {
        static Workspace instance;
        return instance;
    }

    bool ensure_buffer(void **ptr, std::size_t &capacity, std::size_t required,
                       const char *label)
    {
        if (required <= capacity)
        {
            return true;
        }
        if (*ptr != nullptr)
        {
            const cudaError_t free_status = cudaFree(*ptr);
            if (free_status != cudaSuccess)
            {
                set_cuda_error("cudaFree", free_status);
                return false;
            }
            *ptr = nullptr;
            capacity = 0;
        }
        const cudaError_t status = cudaMalloc(ptr, required);
        if (status != cudaSuccess)
        {
            set_cuda_error(label, status);
            return false;
        }
        capacity = required;
        return true;
    }

    bool ensure_workspace(Workspace &ws,
                          std::size_t matrices_bytes,
                          std::size_t rho_bytes,
                          std::size_t starts_bytes,
                          std::size_t positions_bytes,
                          std::size_t results_bytes)
    {
        if (ws.handle == nullptr)
        {
            const cublasStatus_t status = cublasCreate(&ws.handle);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                set_cublas_error("cublasCreate", status);
                return false;
            }
        }
        if (!ensure_buffer(&ws.matrices, ws.matrices_bytes, matrices_bytes,
                           "cudaMalloc(entropy matrices)")) return false;
        if (!ensure_buffer(&ws.rho, ws.rho_bytes, rho_bytes,
                           "cudaMalloc(entropy RDMs)")) return false;
        if (!ensure_buffer(reinterpret_cast<void **>(&ws.starts), ws.starts_bytes,
                           starts_bytes, "cudaMalloc(entropy starts)")) return false;
        if (!ensure_buffer(reinterpret_cast<void **>(&ws.positions), ws.positions_bytes,
                           positions_bytes, "cudaMalloc(entropy positions)")) return false;
        if (!ensure_buffer(reinterpret_cast<void **>(&ws.results), ws.results_bytes,
                           results_bytes, "cudaMalloc(entropy results)")) return false;
        if (ws.numeric_error == nullptr)
        {
            const cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&ws.numeric_error), sizeof(int));
            if (status != cudaSuccess)
            {
                set_cuda_error("cudaMalloc(entropy numeric status)", status);
                return false;
            }
        }
        return true;
    }

    void build_positions(int n,
                         int kept,
                         const int *starts,
                         int start_count,
                         int *positions)
    {
        for (int batch = 0; batch < start_count; ++batch)
        {
            const int start = starts[batch];
            int position = 0;
            for (int local = 0; local < kept; ++local)
            {
                const int mode = (start + local) % n;
                positions[batch * n + mode] = position++;
            }
            for (int mode = 0; mode < n; ++mode)
            {
                const int distance = (mode - start + n) % n;
                if (distance >= kept)
                {
                    positions[batch * n + mode] = position++;
                }
            }
        }
    }

    // --- von Neumann entropy ------------------------------------------------
    //
    // S2 needs only Tr(rho^2), which the purity kernel reads straight off the
    // Frobenius norm. S1 = -Tr(rho ln rho) needs the whole spectrum, so every
    // (subsystem size, cyclic translation) pair costs one dense eigensolve of a
    // 2^L_A matrix. That is the dominant cost of this routine at large L_A and
    // is why the caller can cap it or turn it off.

    bool ensure_solver_handle(Workspace &ws)
    {
        if (ws.solver_handle != nullptr)
        {
            return true;
        }
        const cusolverStatus_t status = cusolverDnCreate(&ws.solver_handle);
        if (status != CUSOLVER_STATUS_SUCCESS)
        {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "cusolverDnCreate failed with status=%d",
                          static_cast<int>(status));
            ws.solver_handle = nullptr;
            return false;
        }
        return true;
    }

    // Host landing buffer for one level's eigenvalues. Grown, never shrunk; a
    // plain malloc keeps this translation unit free of libstdc++ containers.
    void *host_eigenvalue_buffer(std::size_t bytes)
    {
        static thread_local void *buffer = nullptr;
        static thread_local std::size_t capacity = 0;
        if (bytes > capacity)
        {
            std::free(buffer);
            buffer = std::malloc(bytes);
            capacity = (buffer != nullptr) ? bytes : 0;
        }
        return buffer;
    }

    // Eigenvalues -> S1 in nats, for one matrix. The eigenvalues are rescaled by
    // their sum rather than assumed to be normalized, which removes the same
    // trajectory-normalization drift the purity path divides out with trace^2.
    //
    // `floor_value` follows the precision of the *state vector*, not of the
    // eigensolver: an fp64 retry on an fp32 RDM does not make the roundoff-level
    // end of the spectrum meaningful.
    template <typename Real>
    bool von_neumann_nats(const Real *values,
                          int count,
                          double floor_value,
                          double &entropy_out)
    {
        double trace = 0.0;
        for (int i = 0; i < count; ++i)
        {
            trace += static_cast<double>(values[i]);
        }
        if (!(trace > 0.0) || !isfinite(trace))
        {
            return false;
        }

        double entropy = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const double lambda = static_cast<double>(values[i]) / trace;
            if (lambda > floor_value)
            {
                entropy -= lambda * log(lambda);
            }
        }
        if (!isfinite(entropy))
        {
            return false;
        }
        entropy_out = (fabs(entropy) < 1.0e-12) ? 0.0 : entropy;
        return true;
    }

    // Enqueue one heevd per matrix and read the convergence flags once.
    //
    // Sharing a single workspace buffer across the batch is safe because every
    // solve is issued on the default stream and so runs on its own. Reading the
    // info flags per level rather than per solve keeps the host synchronizations
    // down to one, which matters when a level is 30 solves of a 4x4.
    //
    // Returns 0 on success. `converged` distinguishes an ordinary cuSOLVER
    // non-convergence, which the caller answers by demoting to the next solver,
    // from a hard failure, which it propagates.
    template <typename SolveT>
    int run_heevd_batch(Workspace &ws,
                        SolveT *matrices,
                        typename SolverTraits<SolveT>::Real *eigenvalues,
                        int dim,
                        int batch_count,
                        int *host_info,
                        bool &converged)
    {
        converged = false;

        int lwork = 0;
        cusolverStatus_t solver_status =
            heevd_buffer_size(ws.solver_handle, dim, matrices, eigenvalues, &lwork);
        if (solver_status != CUSOLVER_STATUS_SUCCESS)
        {
            return 0;
        }
        if (!ensure_buffer(&ws.solver_work, ws.solver_work_bytes,
                           static_cast<std::size_t>(max_int(1, lwork)) * sizeof(SolveT),
                           "cudaMalloc(entropy eigensolver workspace)"))
        {
            return 24;
        }

        const std::size_t elements =
            static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim);
        for (int b = 0; b < batch_count; ++b)
        {
            solver_status = heevd(ws.solver_handle, dim,
                                  matrices + static_cast<std::size_t>(b) * elements,
                                  eigenvalues + static_cast<std::size_t>(b) * dim,
                                  static_cast<SolveT *>(ws.solver_work), lwork,
                                  ws.solver_info + b);
            if (solver_status != CUSOLVER_STATUS_SUCCESS)
            {
                return 0;
            }
        }

        const cudaError_t cuda_status =
            cudaMemcpy(host_info, ws.solver_info,
                       static_cast<std::size_t>(batch_count) * sizeof(int),
                       cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess)
        {
            set_cuda_error("cudaMemcpy(entropy eigensolver status)", cuda_status);
            return 25;
        }

        converged = true;
        for (int b = 0; b < batch_count; ++b)
        {
            if (host_info[b] != 0) { converged = false; break; }
        }
        return 0;
    }

    // One eigensolve per translation for a single subsystem size. `rho_batch`
    // holds `batch_count` matrices of dimension `dim`, packed with stride
    // dim*dim, and is left untouched -- the caller still needs it for the
    // partial trace down to the next size, and every solver here destroys its
    // input, which is why each stage refills its own copy.
    template <typename ComplexT>
    int compute_level_s1(Workspace &ws,
                         const ComplexT *rho_batch,
                         int dim,
                         int batch_count,
                         double *host_out)
    {
        using Real = typename SolverTraits<ComplexT>::Real;

        if (!ensure_solver_handle(ws))
        {
            return 20;
        }

        const std::size_t elements =
            static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim);
        const std::size_t batch = static_cast<std::size_t>(batch_count);
        const std::size_t solver_bytes = batch * elements * sizeof(ComplexT);
        const std::size_t eigenvalues_bytes =
            batch * static_cast<std::size_t>(dim) * sizeof(Real);
        const std::size_t info_bytes = batch * sizeof(int);

        if (!ensure_buffer(&ws.solver, ws.solver_bytes, solver_bytes,
                           "cudaMalloc(entropy eigensolver input)") ||
            !ensure_buffer(&ws.eigenvalues, ws.eigenvalues_bytes, eigenvalues_bytes,
                           "cudaMalloc(entropy eigenvalues)") ||
            !ensure_buffer(reinterpret_cast<void **>(&ws.solver_info),
                           ws.solver_info_bytes, info_bytes,
                           "cudaMalloc(entropy eigensolver status)"))
        {
            return 21;
        }

        ComplexT *solver = static_cast<ComplexT *>(ws.solver);
        Real *eigenvalues = static_cast<Real *>(ws.eigenvalues);

        int *host_info = static_cast<int *>(std::malloc(info_bytes));
        if (host_info == nullptr)
        {
            set_error("Could not allocate the host eigensolver status buffer.");
            return 22;
        }

        const bool heevd_allowed = dim <= heevd_max_rows();
        bool solved = false;
        bool solved_in_f64 = false;
        cudaError_t cuda_status = cudaSuccess;
        int status = 0;

        if (heevd_allowed && !heevd_known_bad(dim, false, false))
        {
            cuda_status = cudaMemcpy(solver, rho_batch, solver_bytes,
                                     cudaMemcpyDeviceToDevice);
            if (cuda_status != cudaSuccess)
            {
                set_cuda_error("cudaMemcpy(entropy eigensolver input)", cuda_status);
                std::free(host_info);
                return 23;
            }
            status = run_heevd_batch<ComplexT>(ws, solver, eigenvalues, dim,
                                               batch_count, host_info, solved);
            if (status != 0) { std::free(host_info); return status; }
            if (!solved) { heevd_known_bad(dim, false, true); }
        }

        // fp32 heevd gives up on physical RDMs well before fp64 does, and fp64
        // is still an order of magnitude faster than gesvd, so promote before
        // demoting. The promotion reads the untouched originals.
        if (!solved && heevd_allowed && SolverTraits<ComplexT>::is_single &&
            !heevd_known_bad(dim, true, false))
        {
            if (!ensure_buffer(&ws.solver64, ws.solver64_bytes,
                               batch * elements * sizeof(cuDoubleComplex),
                               "cudaMalloc(entropy fp64 eigensolver input)") ||
                !ensure_buffer(&ws.eigenvalues64, ws.eigenvalues64_bytes,
                               batch * static_cast<std::size_t>(dim) * sizeof(double),
                               "cudaMalloc(entropy fp64 eigenvalues)"))
            {
                std::free(host_info);
                return 26;
            }

            const std::uint64_t promote_count =
                static_cast<std::uint64_t>(batch) * elements;
            constexpr int threads = 256;
            const int blocks = static_cast<int>((promote_count + threads - 1) / threads);
            promote_to_f64_kernel<<<blocks, threads>>>(
                reinterpret_cast<const cuFloatComplex *>(rho_batch),
                static_cast<cuDoubleComplex *>(ws.solver64), promote_count);
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess)
            {
                set_cuda_error("promote_to_f64_kernel", cuda_status);
                std::free(host_info);
                return 27;
            }

            status = run_heevd_batch<cuDoubleComplex>(
                ws, static_cast<cuDoubleComplex *>(ws.solver64),
                static_cast<double *>(ws.eigenvalues64), dim, batch_count,
                host_info, solved);
            if (status != 0) { std::free(host_info); return status; }
            if (solved) { solved_in_f64 = true; }
            else { heevd_known_bad(dim, true, true); }
        }

        if (!solved)
        {
            cuda_status = cudaMemcpy(solver, rho_batch, solver_bytes,
                                     cudaMemcpyDeviceToDevice);
            if (cuda_status != cudaSuccess)
            {
                set_cuda_error("cudaMemcpy(entropy eigensolver input)", cuda_status);
                std::free(host_info);
                return 28;
            }

            int lwork = 0;
            cusolverStatus_t solver_status =
                gesvd_buffer_size(ws.solver_handle, dim, solver, &lwork);
            if (solver_status != CUSOLVER_STATUS_SUCCESS)
            {
                std::snprintf(g_last_error, sizeof(g_last_error),
                              "cusolverDn?gesvd_bufferSize failed with status=%d",
                              static_cast<int>(solver_status));
                std::free(host_info);
                return 29;
            }
            if (!ensure_buffer(&ws.solver_work, ws.solver_work_bytes,
                               static_cast<std::size_t>(max_int(1, lwork)) * sizeof(ComplexT),
                               "cudaMalloc(entropy eigensolver workspace)") ||
                !ensure_buffer(&ws.solver_rwork, ws.solver_rwork_bytes,
                               static_cast<std::size_t>(max_int(1, 5 * dim)) * sizeof(Real),
                               "cudaMalloc(entropy eigensolver real workspace)"))
            {
                std::free(host_info);
                return 30;
            }

            for (int b = 0; b < batch_count; ++b)
            {
                solver_status = gesvd_novectors(
                    ws.solver_handle, dim,
                    solver + static_cast<std::size_t>(b) * elements,
                    eigenvalues + static_cast<std::size_t>(b) * dim,
                    static_cast<ComplexT *>(ws.solver_work), lwork,
                    static_cast<Real *>(ws.solver_rwork), ws.solver_info + b);
                if (solver_status != CUSOLVER_STATUS_SUCCESS)
                {
                    std::snprintf(g_last_error, sizeof(g_last_error),
                                  "cusolverDn?gesvd failed with status=%d at dim=%d",
                                  static_cast<int>(solver_status), dim);
                    std::free(host_info);
                    return 31;
                }
            }

            cuda_status = cudaMemcpy(host_info, ws.solver_info, info_bytes,
                                     cudaMemcpyDeviceToHost);
            if (cuda_status != cudaSuccess)
            {
                set_cuda_error("cudaMemcpy(entropy eigensolver status)", cuda_status);
                std::free(host_info);
                return 32;
            }
            for (int b = 0; b < batch_count; ++b)
            {
                if (host_info[b] != 0)
                {
                    std::snprintf(g_last_error, sizeof(g_last_error),
                                  "cusolverDn?gesvd did not converge (info=%d) at dim=%d",
                                  host_info[b], dim);
                    std::free(host_info);
                    return 33;
                }
            }
        }
        std::free(host_info);

        const std::size_t landing_bytes =
            solved_in_f64 ? batch * static_cast<std::size_t>(dim) * sizeof(double)
                          : eigenvalues_bytes;
        void *host_eigenvalues = host_eigenvalue_buffer(landing_bytes);
        if (host_eigenvalues == nullptr)
        {
            set_error("Could not allocate the host eigenvalue buffer.");
            return 34;
        }
        cuda_status = cudaMemcpy(host_eigenvalues,
                                 solved_in_f64 ? ws.eigenvalues64 : ws.eigenvalues,
                                 landing_bytes, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess)
        {
            set_cuda_error("cudaMemcpy(entropy eigenvalues)", cuda_status);
            return 35;
        }

        const double floor_value = SolverTraits<ComplexT>::eigenvalue_floor();
        for (int b = 0; b < batch_count; ++b)
        {
            const std::size_t offset = static_cast<std::size_t>(b) * dim;
            const bool ok =
                solved_in_f64
                    ? von_neumann_nats(static_cast<const double *>(host_eigenvalues) + offset,
                                       dim, floor_value, host_out[b])
                    : von_neumann_nats(static_cast<const Real *>(host_eigenvalues) + offset,
                                       dim, floor_value, host_out[b]);
            if (!ok)
            {
                set_error("GPU von Neumann entropy produced a non-positive trace "
                          "or a non-finite value.");
                return 36;
            }
        }
        return 0;
    }

    template <typename ComplexT>
    cublasStatus_t batched_rho(cublasHandle_t handle,
                               const ComplexT *matrices,
                               ComplexT *rho,
                               int retained_dim,
                               int environment_dim,
                               long long matrix_stride,
                               long long rho_stride,
                               int batch_count);

    template <>
    cublasStatus_t batched_rho<cuFloatComplex>(cublasHandle_t handle,
                                               const cuFloatComplex *matrices,
                                               cuFloatComplex *rho,
                                               int retained_dim,
                                               int environment_dim,
                                               long long matrix_stride,
                                               long long rho_stride,
                                               int batch_count)
    {
        const cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
        const cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
        return cublasCgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_C,
            retained_dim,
            retained_dim,
            environment_dim,
            &alpha,
            matrices,
            retained_dim,
            matrix_stride,
            matrices,
            retained_dim,
            matrix_stride,
            &beta,
            rho,
            retained_dim,
            rho_stride,
            batch_count);
    }

    template <>
    cublasStatus_t batched_rho<cuDoubleComplex>(cublasHandle_t handle,
                                                const cuDoubleComplex *matrices,
                                                cuDoubleComplex *rho,
                                                int retained_dim,
                                                int environment_dim,
                                                long long matrix_stride,
                                                long long rho_stride,
                                                int batch_count)
    {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        return cublasZgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_C,
            retained_dim,
            retained_dim,
            environment_dim,
            &alpha,
            matrices,
            retained_dim,
            matrix_stride,
            matrices,
            retained_dim,
            matrix_stride,
            &beta,
            rho,
            retained_dim,
            rho_stride,
            batch_count);
    }

    template <typename ComplexT>
    int compute_cyclic_entropies(const void *device_state,
                                 int n,
                                 int min_kept,
                                 int max_kept,
                                 int fermionic_trace,
                                 int max_s1_kept,
                                 double *host_s1_means,
                                 double *host_s2_means)
    {
        clear_error();
        if (device_state == nullptr || host_s2_means == nullptr)
        {
            set_error("Null device statevector or output pointer.");
            return 1;
        }
        if (n < 2 || n >= 31 || min_kept < 1 || max_kept < min_kept || max_kept > n / 2)
        {
            set_error("Invalid N or retained-subsystem range for CUDA cyclic entropy.");
            return 2;
        }

        const int s1_kept_limit = min_int(max_kept, max_s1_kept);
        const bool want_s1 = (host_s1_means != nullptr) && (s1_kept_limit >= min_kept);

        // Form only the largest requested RDM. Smaller contiguous intervals are
        // prefixes in the same cyclic local-mode order, so recursively tracing
        // the trailing retained mode gives exactly the independently computed
        // RDMs. Fermionic permutation signs are therefore needed only once,
        // during the initial statevector gather.
        const std::uint64_t total_dim = std::uint64_t{1} << n;
        const int retained_dim_max = 1 << max_kept;
        const int environment_dim_max = 1 << (n - max_kept);
        const std::uint64_t rho_elements_max =
            static_cast<std::uint64_t>(retained_dim_max) *
            static_cast<std::uint64_t>(retained_dim_max);
        const int size_count = max_kept - min_kept + 1;
        const std::size_t result_count =
            static_cast<std::size_t>(size_count) * static_cast<std::size_t>(n);
        const std::size_t workspace_limit = max_workspace_bytes();

        // The eigensolver works on its own copy of the RDM batch, capped at the
        // largest size S1 is actually asked for.
        const std::uint64_t s1_elements_max =
            want_s1 ? (std::uint64_t{1} << (2 * s1_kept_limit)) : std::uint64_t{0};

        const std::size_t bytes_per_translation =
            static_cast<std::size_t>(total_dim) * sizeof(ComplexT) +
            static_cast<std::size_t>(rho_elements_max) * sizeof(ComplexT) +
            static_cast<std::size_t>(s1_elements_max) * sizeof(ComplexT) +
            static_cast<std::size_t>(n + 1) * sizeof(int);
        const std::size_t translations_by_memory =
            workspace_limit / max_size(std::size_t{1}, bytes_per_translation);
        const std::size_t translation_batch_size = min_size(
            static_cast<std::size_t>(n),
            max_size(std::size_t{1}, translations_by_memory));
        const int translation_batch = max_int(1, static_cast<int>(translation_batch_size));

        const std::size_t matrices_bytes =
            static_cast<std::size_t>(translation_batch) *
            static_cast<std::size_t>(total_dim) * sizeof(ComplexT);
        const std::size_t rho_bytes =
            static_cast<std::size_t>(translation_batch) *
            static_cast<std::size_t>(rho_elements_max) * sizeof(ComplexT);
        const std::size_t starts_bytes =
            static_cast<std::size_t>(translation_batch) * sizeof(int);
        const std::size_t positions_bytes =
            static_cast<std::size_t>(translation_batch) *
            static_cast<std::size_t>(n) * sizeof(int);
        const std::size_t results_bytes = result_count * sizeof(double);

        Workspace &ws = workspace();
        if (!ensure_workspace(ws, matrices_bytes, rho_bytes, starts_bytes,
                              positions_bytes, results_bytes))
        {
            return 3;
        }

        // Fixed host scratch avoids libstdc++ allocations in this nvcc object.
        // The input guard n<31 makes these bounds sufficient.
        double all_results[31 * 31] = {0.0};
        double all_s1[31 * 31] = {0.0};
        double level_s1[31] = {0.0};
        int starts_host[31] = {0};
        int positions_host[31 * 31] = {0};

        for (int first_start = 0; first_start < n; first_start += translation_batch)
        {
            const int current_batch = min_int(translation_batch, n - first_start);
            for (int batch = 0; batch < current_batch; ++batch)
            {
                starts_host[batch] = first_start + batch;
            }
            build_positions(n, max_kept, starts_host, current_batch, positions_host);

            cudaError_t cuda_status = cudaMemcpy(
                ws.starts, starts_host,
                static_cast<std::size_t>(current_batch) * sizeof(int),
                cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess)
            {
                set_cuda_error("cudaMemcpy(entropy starts)", cuda_status);
                return 4;
            }
            cuda_status = cudaMemcpy(
                ws.positions, positions_host,
                static_cast<std::size_t>(current_batch) *
                    static_cast<std::size_t>(n) * sizeof(int),
                cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess)
            {
                set_cuda_error("cudaMemcpy(entropy positions)", cuda_status);
                return 5;
            }

            const std::uint64_t gather_work =
                total_dim * static_cast<std::uint64_t>(current_batch);
            constexpr int threads = 256;
            const int gather_blocks =
                static_cast<int>((gather_work + threads - 1) / threads);
            gather_cyclic_matrices_kernel<ComplexT><<<gather_blocks, threads>>>(
                static_cast<const ComplexT *>(device_state), n, max_kept,
                total_dim, current_batch, ws.starts, ws.positions,
                fermionic_trace, static_cast<ComplexT *>(ws.matrices));
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess)
            {
                set_cuda_error("gather_cyclic_matrices_kernel", cuda_status);
                return 6;
            }

            const cublasStatus_t blas_status = batched_rho<ComplexT>(
                ws.handle, static_cast<const ComplexT *>(ws.matrices),
                static_cast<ComplexT *>(ws.rho), retained_dim_max,
                environment_dim_max, static_cast<long long>(total_dim),
                static_cast<long long>(rho_elements_max), current_batch);
            if (blas_status != CUBLAS_STATUS_SUCCESS)
            {
                set_cublas_error(
                    "cublasGemmStridedBatched(entropy half-chain RDM)", blas_status);
                return 7;
            }

            cuda_status = cudaMemset(ws.numeric_error, 0, sizeof(int));
            if (cuda_status != cudaSuccess)
            {
                set_cuda_error("cudaMemset(entropy numeric status)", cuda_status);
                return 8;
            }

            const ComplexT *current_rho = static_cast<const ComplexT *>(ws.rho);
            ComplexT *next_rho = static_cast<ComplexT *>(ws.matrices);
            int current_dim = retained_dim_max;

            for (int kept = max_kept; kept >= min_kept; --kept)
            {
                const std::uint64_t current_elements =
                    static_cast<std::uint64_t>(current_dim) *
                    static_cast<std::uint64_t>(current_dim);
                const int result_offset = (kept - min_kept) * n + first_start;
                purity_kernel<ComplexT><<<current_batch, 256>>>(
                    current_rho, current_elements, current_dim, current_batch,
                    result_offset, ws.results, ws.numeric_error);
                cuda_status = cudaGetLastError();
                if (cuda_status != cudaSuccess)
                {
                    set_cuda_error("purity_kernel", cuda_status);
                    return 9;
                }

                if (want_s1 && kept <= s1_kept_limit)
                {
                    const int s1_status = compute_level_s1<ComplexT>(
                        ws, current_rho, current_dim, current_batch, level_s1);
                    if (s1_status != 0)
                    {
                        return s1_status;
                    }
                    for (int batch = 0; batch < current_batch; ++batch)
                    {
                        all_s1[result_offset + batch] = level_s1[batch];
                    }
                }

                if (kept == min_kept)
                {
                    break;
                }

                const int next_dim = current_dim >> 1;
                const std::uint64_t next_elements =
                    static_cast<std::uint64_t>(next_dim) *
                    static_cast<std::uint64_t>(next_dim);
                const std::uint64_t trace_work =
                    next_elements * static_cast<std::uint64_t>(current_batch);
                const int trace_blocks =
                    static_cast<int>((trace_work + threads - 1) / threads);
                partial_trace_trailing_mode_kernel<ComplexT><<<trace_blocks, threads>>>(
                    current_rho, current_dim, current_batch, next_rho);
                cuda_status = cudaGetLastError();
                if (cuda_status != cudaSuccess)
                {
                    set_cuda_error(
                        "partial_trace_trailing_mode_kernel", cuda_status);
                    return 10;
                }

                current_dim = next_dim;
                const ComplexT *new_current = next_rho;
                next_rho = const_cast<ComplexT *>(current_rho);
                current_rho = new_current;
            }

            int numeric_error = 0;
            cuda_status = cudaMemcpy(&numeric_error, ws.numeric_error, sizeof(int),
                                     cudaMemcpyDeviceToHost);
            if (cuda_status != cudaSuccess)
            {
                set_cuda_error("cudaMemcpy(entropy numeric status)", cuda_status);
                return 11;
            }
            if (numeric_error != 0)
            {
                set_error("GPU entropy purity calculation produced an invalid trace or purity.");
                return 12;
            }
        }

        const cudaError_t result_status = cudaMemcpy(
            all_results, ws.results, result_count * sizeof(double),
            cudaMemcpyDeviceToHost);
        if (result_status != cudaSuccess)
        {
            set_cuda_error("cudaMemcpy(entropy results)", result_status);
            return 13;
        }

        for (int size_index = 0; size_index < size_count; ++size_index)
        {
            double sum = 0.0;
            double s1_sum = 0.0;
            const bool size_has_s1 = want_s1 && (min_kept + size_index) <= s1_kept_limit;
            for (int start_index = 0; start_index < n; ++start_index)
            {
                const std::size_t index =
                    static_cast<std::size_t>(size_index) * static_cast<std::size_t>(n) +
                    static_cast<std::size_t>(start_index);
                const double value = all_results[index];
                if (!isfinite(value))
                {
                    set_error("GPU entropy result contained a non-finite value.");
                    return 14;
                }
                sum += value;
                if (size_has_s1)
                {
                    const double s1_value = all_s1[index];
                    if (!isfinite(s1_value))
                    {
                        set_error("GPU von Neumann entropy result contained a "
                                  "non-finite value.");
                        return 15;
                    }
                    s1_sum += s1_value;
                }
            }
            host_s2_means[size_index] = sum / static_cast<double>(n);
            if (host_s1_means != nullptr)
            {
                // Sizes the caller excluded report NaN, never a zero that would
                // be indistinguishable from a product state.
                host_s1_means[size_index] =
                    size_has_s1 ? (s1_sum / static_cast<double>(n)) : nan("");
            }
        }
        return 0;
    }

}

extern "C" int mipt_cuda_cyclic_entropies_f32(const void *device_state,
                                              int n,
                                              int min_kept,
                                              int max_kept,
                                              int fermionic_trace,
                                              int max_s1_kept,
                                              double *host_s1_means,
                                              double *host_s2_means)
{
    return compute_cyclic_entropies<cuFloatComplex>(
        device_state, n, min_kept, max_kept, fermionic_trace, max_s1_kept,
        host_s1_means, host_s2_means);
}

extern "C" int mipt_cuda_cyclic_entropies_f64(const void *device_state,
                                              int n,
                                              int min_kept,
                                              int max_kept,
                                              int fermionic_trace,
                                              int max_s1_kept,
                                              double *host_s1_means,
                                              double *host_s2_means)
{
    return compute_cyclic_entropies<cuDoubleComplex>(
        device_state, n, min_kept, max_kept, fermionic_trace, max_s1_kept,
        host_s1_means, host_s2_means);
}

extern "C" const char *mipt_cuda_cyclic_entropies_last_error(void)
{
    return g_last_error;
}
