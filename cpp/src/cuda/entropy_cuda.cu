#include "mipt/cuda/entropy_cuda.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace
{
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

        ~Workspace()
        {
            if (matrices != nullptr) cudaFree(matrices);
            if (rho != nullptr) cudaFree(rho);
            if (starts != nullptr) cudaFree(starts);
            if (positions != nullptr) cudaFree(positions);
            if (results != nullptr) cudaFree(results);
            if (numeric_error != nullptr) cudaFree(numeric_error);
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
    int compute_cyclic_s2(const void *device_state,
                          int n,
                          int min_kept,
                          int max_kept,
                          int fermionic_trace,
                          double *host_means)
    {
        clear_error();
        if (device_state == nullptr || host_means == nullptr)
        {
            set_error("Null device statevector or output pointer.");
            return 1;
        }
        if (n < 2 || n >= 31 || min_kept < 1 || max_kept < min_kept || max_kept > n / 2)
        {
            set_error("Invalid N or retained-subsystem range for CUDA cyclic entropy.");
            return 2;
        }

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

        const std::size_t bytes_per_translation =
            static_cast<std::size_t>(total_dim) * sizeof(ComplexT) +
            static_cast<std::size_t>(rho_elements_max) * sizeof(ComplexT) +
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
            for (int start_index = 0; start_index < n; ++start_index)
            {
                const double value = all_results[
                    static_cast<std::size_t>(size_index) * static_cast<std::size_t>(n) +
                    static_cast<std::size_t>(start_index)];
                if (!isfinite(value))
                {
                    set_error("GPU entropy result contained a non-finite value.");
                    return 14;
                }
                sum += value;
            }
            host_means[size_index] = sum / static_cast<double>(n);
        }
        return 0;
    }

}

extern "C" int mipt_cuda_cyclic_s2_f32(const void *device_state,
                                         int n,
                                         int min_kept,
                                         int max_kept,
                                         int fermionic_trace,
                                         double *host_means)
{
    return compute_cyclic_s2<cuFloatComplex>(device_state, n, min_kept, max_kept,
                                             fermionic_trace, host_means);
}

extern "C" int mipt_cuda_cyclic_s2_f64(const void *device_state,
                                         int n,
                                         int min_kept,
                                         int max_kept,
                                         int fermionic_trace,
                                         double *host_means)
{
    return compute_cyclic_s2<cuDoubleComplex>(device_state, n, min_kept, max_kept,
                                              fermionic_trace, host_means);
}

extern "C" const char *mipt_cuda_cyclic_s2_last_error(void)
{
    return g_last_error;
}
