#include "tmi_cuda_svd.hpp"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include <cuda_runtime.h>
#include <cuComplex.h>

// Minimal cuSOLVER dense-SVD declarations.
// Do not include <cusolverDn.h> here: some CUDA 11.x + GCC 11 setups fail
// while parsing that header through nvcc.  These legacy gesvd entry points have
// a stable C ABI, and the object exposes only extern "C" functions to the
// CUDA-Q/clang++ side.  This file intentionally avoids STL types so that the
// nvcc/g++ object does not pull libstdc++ symbols into a CUDA-Q libc++ link.
extern "C" {
struct cusolverDnContext;
typedef cusolverDnContext *cusolverDnHandle_t;
typedef int cusolverStatus_t;

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle);
cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork);
cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m,
                                             int n,
                                             int *lwork);
cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  float *S,
                                  cuComplex *U,
                                  int ldu,
                                  cuComplex *VT,
                                  int ldvt,
                                  cuComplex *work,
                                  int lwork,
                                  float *rwork,
                                  int *devInfo);
cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t handle,
                                  signed char jobu,
                                  signed char jobvt,
                                  int m,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  double *S,
                                  cuDoubleComplex *U,
                                  int ldu,
                                  cuDoubleComplex *VT,
                                  int ldvt,
                                  cuDoubleComplex *work,
                                  int lwork,
                                  double *rwork,
                                  int *devInfo);
}

#ifndef CUSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_SUCCESS 0
#endif

namespace
{
    enum
    {
        THREADS = 256,
        CUSOLVER_ERROR_BASE = 100000
    };

    template <typename Real>
    struct Cx
    {
        Real re;
        Real im;
    };

    static inline size_t min_size_t(size_t a, size_t b)
    {
        return (a < b) ? a : b;
    }

    static inline int max_int(int a, int b)
    {
        return (a > b) ? a : b;
    }

    template <typename T>
    T *thread_buffer(size_t required_elements,
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

            status = cudaMalloc(reinterpret_cast<void **>(&buffer), required_elements * sizeof(T));
            if (status != cudaSuccess)
            {
                return NULL;
            }
            capacity = required_elements;
        }
        return buffer;
    }

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

    static uint64_t *basis_offsets_host(const int *modes, int count)
    {
        if (modes == NULL || count < 0 || count >= 63)
        {
            return NULL;
        }
        const uint64_t dim = uint64_t{1} << count;
        uint64_t *offsets = static_cast<uint64_t *>(malloc(static_cast<size_t>(dim) * sizeof(uint64_t)));
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

    static int complement_modes_host(int n,
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

        unsigned char *kept = static_cast<unsigned char *>(calloc(static_cast<size_t>(n), sizeof(unsigned char)));
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
        int *env_modes = static_cast<int *>(malloc(static_cast<size_t>(env_count) * sizeof(int)));
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

    static uint64_t *fermion_cross_masks_by_row_host(const int *kept_modes,
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

        uint64_t *lower_env_mask = static_cast<uint64_t *>(calloc(static_cast<size_t>(kept_count), sizeof(uint64_t)));
        if (kept_count > 0 && lower_env_mask == NULL)
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

    template <typename Real, typename ComplexT>
    __global__ void build_schmidt_matrix_kernel(const Cx<Real> *__restrict__ psi,
                                                ComplexT *__restrict__ matrix_col_major,
                                                size_t elements,
                                                size_t rows,
                                                const uint64_t *__restrict__ row_offsets,
                                                const uint64_t *__restrict__ env_offsets,
                                                const uint64_t *__restrict__ row_cross_masks,
                                                int fermion_trace)
    {
        for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
             index < elements;
             index += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const size_t row = index % rows;
            const size_t col = index / rows;
            const uint64_t src = env_offsets[col] | row_offsets[row];
            const Cx<Real> value = psi[src];
            const Real sign = (fermion_trace && odd_popcount64(static_cast<uint64_t>(col) & row_cross_masks[row]))
                                  ? Real(-1)
                                  : Real(1);
            matrix_col_major[index].x = sign * value.re;
            matrix_col_major[index].y = sign * value.im;
        }
    }

    template <typename Real>
    double entropy_from_singular_values(const Real *values, int count)
    {
        double entropy = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const double sigma = static_cast<double>(values[i]);
            const double lambda = sigma * sigma;
            if (lambda > 1.0e-15)
            {
                entropy -= lambda * (log(lambda) / 0.693147180559945309417232121458176568);
            }
        }
        if (fabs(entropy) < 1.0e-12)
        {
            return 0.0;
        }
        return entropy;
    }

    template <typename ComplexT>
    int prepare_common_buffers(int n_qubits,
                               const int *host_kept_modes,
                               int kept_count,
                               int fermion_trace,
                               size_t rows,
                               size_t cols,
                               ComplexT *&d_matrix,
                               size_t &matrix_capacity,
                               uint64_t *&d_row_offsets,
                               size_t &row_capacity,
                               uint64_t *&d_env_offsets,
                               size_t &env_capacity,
                               uint64_t *&d_row_cross_masks,
                               size_t &mask_capacity)
    {
        if (n_qubits <= 0 || n_qubits >= 63 || kept_count <= 0 || kept_count >= n_qubits ||
            kept_count >= 31 || host_kept_modes == NULL)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }

        int *env_modes = NULL;
        int status = complement_modes_host(n_qubits, host_kept_modes, kept_count, &env_modes);
        if (status != 0)
        {
            return status;
        }
        const int env_count = n_qubits - kept_count;

        uint64_t *row_offsets = basis_offsets_host(host_kept_modes, kept_count);
        uint64_t *env_offsets = basis_offsets_host(env_modes, env_count);
        uint64_t *row_cross_masks = NULL;
        if (fermion_trace)
        {
            row_cross_masks = fermion_cross_masks_by_row_host(host_kept_modes, kept_count, env_modes, env_count, rows);
        }
        else
        {
            row_cross_masks = static_cast<uint64_t *>(calloc(rows, sizeof(uint64_t)));
        }

        if (row_offsets == NULL || env_offsets == NULL || row_cross_masks == NULL)
        {
            free(env_modes);
            free(row_offsets);
            free(env_offsets);
            free(row_cross_masks);
            return static_cast<int>(cudaErrorMemoryAllocation);
        }

        cudaError_t cuda_status = cudaSuccess;
        if (thread_buffer(rows * cols, d_matrix, matrix_capacity, cuda_status) == NULL)
        {
            status = static_cast<int>(cuda_status);
            goto cleanup;
        }
        if (thread_buffer(rows, d_row_offsets, row_capacity, cuda_status) == NULL)
        {
            status = static_cast<int>(cuda_status);
            goto cleanup;
        }
        if (thread_buffer(cols, d_env_offsets, env_capacity, cuda_status) == NULL)
        {
            status = static_cast<int>(cuda_status);
            goto cleanup;
        }
        if (thread_buffer(rows, d_row_cross_masks, mask_capacity, cuda_status) == NULL)
        {
            status = static_cast<int>(cuda_status);
            goto cleanup;
        }

        cuda_status = cudaMemcpy(d_row_offsets, row_offsets, rows * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
        {
            status = static_cast<int>(cuda_status);
            goto cleanup;
        }
        cuda_status = cudaMemcpy(d_env_offsets, env_offsets, cols * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
        {
            status = static_cast<int>(cuda_status);
            goto cleanup;
        }
        cuda_status = cudaMemcpy(d_row_cross_masks, row_cross_masks, rows * sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
        {
            status = static_cast<int>(cuda_status);
            goto cleanup;
        }

        status = 0;

    cleanup:
        free(env_modes);
        free(row_offsets);
        free(env_offsets);
        free(row_cross_masks);
        return status;
    }

    int thread_cusolver_handle(cusolverDnHandle_t &handle)
    {
        static thread_local cusolverDnHandle_t tls_handle = NULL;
        if (tls_handle == NULL)
        {
            const cusolverStatus_t status = cusolverDnCreate(&tls_handle);
            if (status != CUSOLVER_STATUS_SUCCESS)
            {
                return CUSOLVER_ERROR_BASE + static_cast<int>(status);
            }
        }
        handle = tls_handle;
        return 0;
    }

    int launch_entropy_svd_f32(const void *device_state_vector,
                               int n_qubits,
                               const int *host_kept_modes,
                               int kept_count,
                               int fermion_trace,
                               double *entropy_out)
    {
        if (device_state_vector == NULL || entropy_out == NULL ||
            n_qubits <= 0 || n_qubits >= 63 || kept_count <= 0 || kept_count >= n_qubits ||
            kept_count >= 31 || host_kept_modes == NULL)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }

        const size_t rows = size_t{1} << kept_count;
        const size_t cols = size_t{1} << (n_qubits - kept_count);
        const int min_dim = static_cast<int>(min_size_t(rows, cols));
        if (rows == 1 || cols == 1)
        {
            *entropy_out = 0.0;
            return 0;
        }
        if (rows > static_cast<size_t>(INT_MAX) || cols > static_cast<size_t>(INT_MAX))
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }

        static thread_local cuComplex *d_matrix = NULL;
        static thread_local size_t matrix_capacity = 0;
        static thread_local uint64_t *d_row_offsets = NULL;
        static thread_local size_t row_capacity = 0;
        static thread_local uint64_t *d_env_offsets = NULL;
        static thread_local size_t env_capacity = 0;
        static thread_local uint64_t *d_row_cross_masks = NULL;
        static thread_local size_t mask_capacity = 0;
        static thread_local float *d_singular_values = NULL;
        static thread_local size_t singular_capacity = 0;
        static thread_local cuComplex *d_work = NULL;
        static thread_local size_t work_capacity = 0;
        static thread_local cuComplex *d_dummy_u = NULL;
        static thread_local size_t dummy_u_capacity = 0;
        static thread_local cuComplex *d_dummy_vt = NULL;
        static thread_local size_t dummy_vt_capacity = 0;
        static thread_local float *d_rwork = NULL;
        static thread_local size_t rwork_capacity = 0;
        static thread_local int *d_info = NULL;
        static thread_local size_t info_capacity = 0;

        int status = prepare_common_buffers<cuComplex>(n_qubits,
                                                       host_kept_modes,
                                                       kept_count,
                                                       fermion_trace,
                                                       rows,
                                                       cols,
                                                       d_matrix,
                                                       matrix_capacity,
                                                       d_row_offsets,
                                                       row_capacity,
                                                       d_env_offsets,
                                                       env_capacity,
                                                       d_row_cross_masks,
                                                       mask_capacity);
        if (status != 0)
        {
            return status;
        }

        const size_t elements = rows * cols;
        const int blocks = static_cast<int>(min_size_t(size_t{65535}, (elements + THREADS - 1) / THREADS));
        build_schmidt_matrix_kernel<float, cuComplex><<<blocks, THREADS>>>(
            reinterpret_cast<const Cx<float> *>(device_state_vector),
            d_matrix,
            elements,
            rows,
            d_row_offsets,
            d_env_offsets,
            d_row_cross_masks,
            fermion_trace);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess)
        {
            return static_cast<int>(cuda_status);
        }

        cusolverDnHandle_t handle = NULL;
        status = thread_cusolver_handle(handle);
        if (status != 0)
        {
            return status;
        }

        int lwork = 0;
        cusolverStatus_t solver_status = cusolverDnCgesvd_bufferSize(
            handle,
            static_cast<int>(rows),
            static_cast<int>(cols),
            &lwork);
        if (solver_status != CUSOLVER_STATUS_SUCCESS)
        {
            return CUSOLVER_ERROR_BASE + static_cast<int>(solver_status);
        }

        if (thread_buffer(static_cast<size_t>(min_dim), d_singular_values, singular_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(static_cast<size_t>(max_int(1, lwork)), d_work, work_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(size_t{1}, d_dummy_u, dummy_u_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(size_t{1}, d_dummy_vt, dummy_vt_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(static_cast<size_t>(max_int(1, 5 * min_dim)), d_rwork, rwork_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(size_t{1}, d_info, info_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }

        signed char jobu = 'N';
        signed char jobvt = 'N';
        solver_status = cusolverDnCgesvd(handle,
                                         jobu,
                                         jobvt,
                                         static_cast<int>(rows),
                                         static_cast<int>(cols),
                                         d_matrix,
                                         static_cast<int>(rows),
                                         d_singular_values,
                                         d_dummy_u,
                                         1,
                                         d_dummy_vt,
                                         1,
                                         d_work,
                                         lwork,
                                         d_rwork,
                                         d_info);
        if (solver_status != CUSOLVER_STATUS_SUCCESS)
        {
            return CUSOLVER_ERROR_BASE + static_cast<int>(solver_status);
        }

        int info = 0;
        cuda_status = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess)
        {
            return static_cast<int>(cuda_status);
        }
        if (info != 0)
        {
            return CUSOLVER_ERROR_BASE + 1000 + info;
        }

        float *host_singular_values = static_cast<float *>(malloc(static_cast<size_t>(min_dim) * sizeof(float)));
        if (host_singular_values == NULL)
        {
            return static_cast<int>(cudaErrorMemoryAllocation);
        }
        cuda_status = cudaMemcpy(host_singular_values,
                                 d_singular_values,
                                 static_cast<size_t>(min_dim) * sizeof(float),
                                 cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess)
        {
            free(host_singular_values);
            return static_cast<int>(cuda_status);
        }
        *entropy_out = entropy_from_singular_values(host_singular_values, min_dim);
        free(host_singular_values);
        return 0;
    }

    int launch_entropy_svd_f64(const void *device_state_vector,
                               int n_qubits,
                               const int *host_kept_modes,
                               int kept_count,
                               int fermion_trace,
                               double *entropy_out)
    {
        if (device_state_vector == NULL || entropy_out == NULL ||
            n_qubits <= 0 || n_qubits >= 63 || kept_count <= 0 || kept_count >= n_qubits ||
            kept_count >= 31 || host_kept_modes == NULL)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }

        const size_t rows = size_t{1} << kept_count;
        const size_t cols = size_t{1} << (n_qubits - kept_count);
        const int min_dim = static_cast<int>(min_size_t(rows, cols));
        if (rows == 1 || cols == 1)
        {
            *entropy_out = 0.0;
            return 0;
        }
        if (rows > static_cast<size_t>(INT_MAX) || cols > static_cast<size_t>(INT_MAX))
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }

        static thread_local cuDoubleComplex *d_matrix = NULL;
        static thread_local size_t matrix_capacity = 0;
        static thread_local uint64_t *d_row_offsets = NULL;
        static thread_local size_t row_capacity = 0;
        static thread_local uint64_t *d_env_offsets = NULL;
        static thread_local size_t env_capacity = 0;
        static thread_local uint64_t *d_row_cross_masks = NULL;
        static thread_local size_t mask_capacity = 0;
        static thread_local double *d_singular_values = NULL;
        static thread_local size_t singular_capacity = 0;
        static thread_local cuDoubleComplex *d_work = NULL;
        static thread_local size_t work_capacity = 0;
        static thread_local cuDoubleComplex *d_dummy_u = NULL;
        static thread_local size_t dummy_u_capacity = 0;
        static thread_local cuDoubleComplex *d_dummy_vt = NULL;
        static thread_local size_t dummy_vt_capacity = 0;
        static thread_local double *d_rwork = NULL;
        static thread_local size_t rwork_capacity = 0;
        static thread_local int *d_info = NULL;
        static thread_local size_t info_capacity = 0;

        int status = prepare_common_buffers<cuDoubleComplex>(n_qubits,
                                                             host_kept_modes,
                                                             kept_count,
                                                             fermion_trace,
                                                             rows,
                                                             cols,
                                                             d_matrix,
                                                             matrix_capacity,
                                                             d_row_offsets,
                                                             row_capacity,
                                                             d_env_offsets,
                                                             env_capacity,
                                                             d_row_cross_masks,
                                                             mask_capacity);
        if (status != 0)
        {
            return status;
        }

        const size_t elements = rows * cols;
        const int blocks = static_cast<int>(min_size_t(size_t{65535}, (elements + THREADS - 1) / THREADS));
        build_schmidt_matrix_kernel<double, cuDoubleComplex><<<blocks, THREADS>>>(
            reinterpret_cast<const Cx<double> *>(device_state_vector),
            d_matrix,
            elements,
            rows,
            d_row_offsets,
            d_env_offsets,
            d_row_cross_masks,
            fermion_trace);
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess)
        {
            return static_cast<int>(cuda_status);
        }

        cusolverDnHandle_t handle = NULL;
        status = thread_cusolver_handle(handle);
        if (status != 0)
        {
            return status;
        }

        int lwork = 0;
        cusolverStatus_t solver_status = cusolverDnZgesvd_bufferSize(
            handle,
            static_cast<int>(rows),
            static_cast<int>(cols),
            &lwork);
        if (solver_status != CUSOLVER_STATUS_SUCCESS)
        {
            return CUSOLVER_ERROR_BASE + static_cast<int>(solver_status);
        }

        if (thread_buffer(static_cast<size_t>(min_dim), d_singular_values, singular_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(static_cast<size_t>(max_int(1, lwork)), d_work, work_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(size_t{1}, d_dummy_u, dummy_u_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(size_t{1}, d_dummy_vt, dummy_vt_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(static_cast<size_t>(max_int(1, 5 * min_dim)), d_rwork, rwork_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }
        if (thread_buffer(size_t{1}, d_info, info_capacity, cuda_status) == NULL)
        {
            return static_cast<int>(cuda_status);
        }

        signed char jobu = 'N';
        signed char jobvt = 'N';
        solver_status = cusolverDnZgesvd(handle,
                                         jobu,
                                         jobvt,
                                         static_cast<int>(rows),
                                         static_cast<int>(cols),
                                         d_matrix,
                                         static_cast<int>(rows),
                                         d_singular_values,
                                         d_dummy_u,
                                         1,
                                         d_dummy_vt,
                                         1,
                                         d_work,
                                         lwork,
                                         d_rwork,
                                         d_info);
        if (solver_status != CUSOLVER_STATUS_SUCCESS)
        {
            return CUSOLVER_ERROR_BASE + static_cast<int>(solver_status);
        }

        int info = 0;
        cuda_status = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess)
        {
            return static_cast<int>(cuda_status);
        }
        if (info != 0)
        {
            return CUSOLVER_ERROR_BASE + 1000 + info;
        }

        double *host_singular_values = static_cast<double *>(malloc(static_cast<size_t>(min_dim) * sizeof(double)));
        if (host_singular_values == NULL)
        {
            return static_cast<int>(cudaErrorMemoryAllocation);
        }
        cuda_status = cudaMemcpy(host_singular_values,
                                 d_singular_values,
                                 static_cast<size_t>(min_dim) * sizeof(double),
                                 cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess)
        {
            free(host_singular_values);
            return static_cast<int>(cuda_status);
        }
        *entropy_out = entropy_from_singular_values(host_singular_values, min_dim);
        free(host_singular_values);
        return 0;
    }
}

extern "C" int mipt_cuda_entropy_svd_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *entropy_out)
{
    return launch_entropy_svd_f32(device_state_vector,
                                  n_qubits,
                                  host_kept_modes,
                                  kept_count,
                                  fermion_trace,
                                  entropy_out);
}

extern "C" int mipt_cuda_entropy_svd_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *entropy_out)
{
    return launch_entropy_svd_f64(device_state_vector,
                                  n_qubits,
                                  host_kept_modes,
                                  kept_count,
                                  fermion_trace,
                                  entropy_out);
}
