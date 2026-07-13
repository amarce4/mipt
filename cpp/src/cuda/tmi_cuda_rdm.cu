#include "mipt/cuda/tmi_cuda_rdm.hpp"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

namespace
{
    enum
    {
        THREADS = 256,
        MAX_DEVICE_RDM_KEPT = 12
    };

    template <typename Real>
    struct Cx
    {
        Real re;
        Real im;
    };

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

    static int validate_modes(int n, const int *modes, int kept_count)
    {
        if (n <= 0 || n >= 63 || modes == NULL || kept_count <= 0 || kept_count > n ||
            kept_count > MAX_DEVICE_RDM_KEPT)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        unsigned char seen[64] = {0};
        for (int i = 0; i < kept_count; ++i)
        {
            const int q = modes[i];
            if (q < 0 || q >= n || seen[q] != 0)
            {
                return static_cast<int>(cudaErrorInvalidValue);
            }
            seen[q] = 1;
        }
        return 0;
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

    static int *environment_modes_host(int n, const int *kept_modes, int kept_count)
    {
        unsigned char kept[64] = {0};
        for (int k = 0; k < kept_count; ++k)
        {
            kept[kept_modes[k]] = 1;
        }
        const int env_count = n - kept_count;
        int *env_modes = static_cast<int *>(malloc(static_cast<size_t>(env_count) * sizeof(int)));
        if (env_count > 0 && env_modes == NULL)
        {
            return NULL;
        }
        int cursor = 0;
        for (int q = 0; q < n; ++q)
        {
            if (!kept[q])
            {
                env_modes[cursor++] = q;
            }
        }
        return env_modes;
    }

    static uint64_t *fermion_cross_masks_host(const int *kept_modes,
                                              int kept_count,
                                              const int *env_modes,
                                              int env_count)
    {
        const uint64_t kept_dim = uint64_t{1} << kept_count;
        uint64_t *masks = static_cast<uint64_t *>(malloc(static_cast<size_t>(kept_dim) * sizeof(uint64_t)));
        if (masks == NULL)
        {
            return NULL;
        }
        uint64_t lower_env_mask[MAX_DEVICE_RDM_KEPT] = {0};
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
        for (uint64_t row = 0; row < kept_dim; ++row)
        {
            uint64_t mask = 0;
            for (int k = 0; k < kept_count; ++k)
            {
                if ((row >> k) & uint64_t{1})
                {
                    mask ^= lower_env_mask[k];
                }
            }
            masks[static_cast<size_t>(row)] = mask;
        }
        return masks;
    }

    __device__ inline uint64_t env_offset_from_basis(uint64_t env, const int *env_modes, int env_count)
    {
        uint64_t offset = 0;
        for (int e = 0; e < env_count; ++e)
        {
            offset |= ((env >> e) & uint64_t{1}) << env_modes[e];
        }
        return offset;
    }

    template <typename Real>
    __global__ void rdm_subsystem_kernel(const Cx<Real> *state,
                                         int env_count,
                                         uint64_t env_dim,
                                         const int *env_modes,
                                         const uint64_t *row_offsets,
                                         const uint64_t *cross_masks,
                                         int fermion_trace,
                                         int kept_dim,
                                         double *rho_ri)
    {
        const int element = blockIdx.x;
        const int row = element / kept_dim;
        const int col = element - row * kept_dim;
        const uint64_t row_offset = row_offsets[row];
        const uint64_t col_offset = row_offsets[col];
        const uint64_t row_cross = cross_masks[row];
        const uint64_t col_cross = cross_masks[col];

        double re = 0.0;
        double im = 0.0;
        for (uint64_t env = static_cast<uint64_t>(threadIdx.x); env < env_dim; env += blockDim.x)
        {
            const uint64_t base = env_offset_from_basis(env, env_modes, env_count);
            const Cx<Real> a0 = state[base | row_offset];
            const Cx<Real> b0 = state[base | col_offset];
            double ar = static_cast<double>(a0.re);
            double ai = static_cast<double>(a0.im);
            double br = static_cast<double>(b0.re);
            double bi = static_cast<double>(b0.im);
            if (fermion_trace && odd_popcount64(env & (row_cross ^ col_cross)))
            {
                ar = -ar;
                ai = -ai;
            }
            re += ar * br + ai * bi;
            im += ai * br - ar * bi;
        }

        __shared__ double s_re[THREADS];
        __shared__ double s_im[THREADS];
        s_re[threadIdx.x] = re;
        s_im[threadIdx.x] = im;
        __syncthreads();

        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                s_re[threadIdx.x] += s_re[threadIdx.x + stride];
                s_im[threadIdx.x] += s_im[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            const size_t out = 2u * static_cast<size_t>(element);
            rho_ri[out + 0] = s_re[0];
            rho_ri[out + 1] = s_im[0];
        }
    }

    template <typename Real>
    int launch_rdm_subsystem(const void *device_state_vector,
                             int n,
                             const int *host_kept_modes,
                             int kept_count,
                             int fermion_trace,
                             double *host_rho_ri)
    {
        if (device_state_vector == NULL || host_rho_ri == NULL)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        int status = validate_modes(n, host_kept_modes, kept_count);
        if (status != 0)
        {
            return status;
        }

        const int kept_dim = 1 << kept_count;
        const int env_count = n - kept_count;
        const uint64_t env_dim = uint64_t{1} << env_count;
        const size_t matrix_elements = static_cast<size_t>(kept_dim) * static_cast<size_t>(kept_dim);

        uint64_t *host_row_offsets = basis_offsets_host(host_kept_modes, kept_count);
        int *host_env_modes = environment_modes_host(n, host_kept_modes, kept_count);
        if (host_row_offsets == NULL || (env_count > 0 && host_env_modes == NULL))
        {
            free(host_row_offsets);
            free(host_env_modes);
            return static_cast<int>(cudaErrorMemoryAllocation);
        }

        uint64_t *host_cross_masks = NULL;
        if (fermion_trace)
        {
            host_cross_masks = fermion_cross_masks_host(host_kept_modes, kept_count, host_env_modes, env_count);
        }
        else
        {
            host_cross_masks = static_cast<uint64_t *>(calloc(static_cast<size_t>(kept_dim), sizeof(uint64_t)));
        }
        if (host_cross_masks == NULL)
        {
            free(host_row_offsets);
            free(host_env_modes);
            return static_cast<int>(cudaErrorMemoryAllocation);
        }

        static thread_local uint64_t *device_row_offsets = NULL;
        static thread_local size_t row_offsets_capacity = 0;
        static thread_local int *device_env_modes = NULL;
        static thread_local size_t env_modes_capacity = 0;
        static thread_local uint64_t *device_cross_masks = NULL;
        static thread_local size_t cross_masks_capacity = 0;
        static thread_local double *device_rho = NULL;
        static thread_local size_t rho_capacity = 0;

        cudaError_t cuda_status = cudaSuccess;
        device_row_offsets = thread_buffer(static_cast<size_t>(kept_dim), device_row_offsets, row_offsets_capacity, cuda_status);
        if (cuda_status != cudaSuccess)
        {
            status = static_cast<int>(cuda_status);
        }

        if (status == 0)
        {
            device_env_modes = thread_buffer(static_cast<size_t>(env_count), device_env_modes, env_modes_capacity, cuda_status);
            if (cuda_status != cudaSuccess)
            {
                status = static_cast<int>(cuda_status);
            }
        }

        if (status == 0)
        {
            device_cross_masks = thread_buffer(static_cast<size_t>(kept_dim), device_cross_masks, cross_masks_capacity, cuda_status);
            if (cuda_status != cudaSuccess)
            {
                status = static_cast<int>(cuda_status);
            }
        }

        if (status == 0)
        {
            device_rho = thread_buffer(2u * matrix_elements, device_rho, rho_capacity, cuda_status);
            if (cuda_status != cudaSuccess)
            {
                status = static_cast<int>(cuda_status);
            }
        }

        if (status == 0)
        {
            cuda_status = cudaMemcpy(device_row_offsets, host_row_offsets,
                                     static_cast<size_t>(kept_dim) * sizeof(uint64_t),
                                     cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess)
            {
                status = static_cast<int>(cuda_status);
            }
        }

        if (status == 0 && env_count > 0)
        {
            cuda_status = cudaMemcpy(device_env_modes, host_env_modes,
                                     static_cast<size_t>(env_count) * sizeof(int),
                                     cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess)
            {
                status = static_cast<int>(cuda_status);
            }
        }

        if (status == 0)
        {
            cuda_status = cudaMemcpy(device_cross_masks, host_cross_masks,
                                     static_cast<size_t>(kept_dim) * sizeof(uint64_t),
                                     cudaMemcpyHostToDevice);
            if (cuda_status != cudaSuccess)
            {
                status = static_cast<int>(cuda_status);
            }
        }

        if (status == 0)
        {
            rdm_subsystem_kernel<Real><<<static_cast<unsigned int>(matrix_elements), THREADS>>>(
                reinterpret_cast<const Cx<Real> *>(device_state_vector),
                env_count,
                env_dim,
                device_env_modes,
                device_row_offsets,
                device_cross_masks,
                fermion_trace ? 1 : 0,
                kept_dim,
                device_rho);
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess)
            {
                status = static_cast<int>(cuda_status);
            }
        }

        if (status == 0)
        {
            cuda_status = cudaDeviceSynchronize();
            if (cuda_status != cudaSuccess)
            {
                status = static_cast<int>(cuda_status);
            }
        }

        if (status == 0)
        {
            cuda_status = cudaMemcpy(host_rho_ri, device_rho,
                                     2u * matrix_elements * sizeof(double),
                                     cudaMemcpyDeviceToHost);
            status = static_cast<int>(cuda_status);
        }

        free(host_row_offsets);
        free(host_env_modes);
        free(host_cross_masks);
        return status;
    }
}

extern "C" int mipt_cuda_rdm_subsystem_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *host_rho_ri_row_major)
{
    return launch_rdm_subsystem<double>(device_state_vector,
                                        n_qubits,
                                        host_kept_modes,
                                        kept_count,
                                        fermion_trace,
                                        host_rho_ri_row_major);
}

extern "C" int mipt_cuda_rdm_subsystem_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *host_rho_ri_row_major)
{
    return launch_rdm_subsystem<float>(device_state_vector,
                                       n_qubits,
                                       host_kept_modes,
                                       kept_count,
                                       fermion_trace,
                                       host_rho_ri_row_major);
}
