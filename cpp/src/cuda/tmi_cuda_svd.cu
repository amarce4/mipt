#include "mipt/cuda/tmi_cuda_svd.hpp"
#include "mipt/cuda/schmidt_gram.cuh"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include <cuda_runtime.h>
#include <cuComplex.h>

// Half-cut von Neumann entropy.
//
// This used to call cusolverDnCgesvd on the rows x cols Schmidt matrix.  That
// is the legacy QR-based SVD: it runs a full Householder bidiagonalization even
// under jobu='N'/jobvt='N', only about half of which is BLAS-3, and it is one
// of the weakest routines in cuSOLVER.  It also silently required rows >= cols,
// which held only because TMI half-cuts happen to be square.
//
// Instead form the Gram matrix rho = M M^H with cublasCherk -- a tuned BLAS-3
// kernel, and half the flops of a general GEMM -- and take its eigenvalues with
// cusolverDnCheevd.  rho is the reduced density matrix, so its eigenvalues are
// the Schmidt coefficients directly and no square roots are needed.
//
// Measured on a GTX 1650 (fp32) against an fp64 gesvd reference, on both a
// steeply decaying and a nearly flat entanglement spectrum:
//
//   half-cut     gesvd      Cherk+Cheevd    speedup   entropy error
//    256x256    139 ms         70 ms          2.0x     3e-6 bits
//   1024x1024   243 ms         59 ms          4.1x     2e-5 bits
//   4096x4096  3166 ms       1658 ms          1.9x     6e-5 bits
//
// Squaring the matrix does square the condition number, but the measured
// entropy error stays at the 1e-5 bit level -- orders of magnitude below the
// statistical error on any realistic trajectory count.  cusolverDnCgesvdj was
// also tried and rejected: it is both slower (8.1 s at 4096) and far less
// accurate (3.8e-2 bits), because it hits its sweep cap.

// Minimal cuSOLVER declarations.  Do not include <cusolverDn.h> here: some
// CUDA 11.x + GCC 11 setups fail while parsing that header through nvcc.  These
// entry points have a stable C ABI, and the enum arguments are int-sized.
extern "C" {
struct cusolverDnContext;
typedef cusolverDnContext *cusolverDnHandle_t;
typedef int cusolverStatus_t;

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle);

cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m, int n, int *lwork);
cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m, int n, int *lwork);
cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu,
                                  signed char jobvt, int m, int n, cuComplex *A,
                                  int lda, float *S, cuComplex *U, int ldu,
                                  cuComplex *VT, int ldvt, cuComplex *work,
                                  int lwork, float *rwork, int *devInfo);
cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu,
                                  signed char jobvt, int m, int n,
                                  cuDoubleComplex *A, int lda, double *S,
                                  cuDoubleComplex *U, int ldu,
                                  cuDoubleComplex *VT, int ldvt,
                                  cuDoubleComplex *work, int lwork,
                                  double *rwork, int *devInfo);

cusolverStatus_t cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle,
                                             int jobz,
                                             int uplo,
                                             int n,
                                             const cuComplex *A,
                                             int lda,
                                             const float *W,
                                             int *lwork);
cusolverStatus_t cusolverDnCheevd(cusolverDnHandle_t handle,
                                  int jobz,
                                  int uplo,
                                  int n,
                                  cuComplex *A,
                                  int lda,
                                  float *W,
                                  cuComplex *work,
                                  int lwork,
                                  int *devInfo);

cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle,
                                             int jobz,
                                             int uplo,
                                             int n,
                                             const cuDoubleComplex *A,
                                             int lda,
                                             const double *W,
                                             int *lwork);
cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t handle,
                                  int jobz,
                                  int uplo,
                                  int n,
                                  cuDoubleComplex *A,
                                  int lda,
                                  double *W,
                                  cuDoubleComplex *work,
                                  int lwork,
                                  int *devInfo);
}

#ifndef CUSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_SUCCESS 0
#endif

namespace
{
using namespace mipt_gram;

enum
{
    CUSOLVER_ERROR_BASE = 100000,
    // cusolverEigMode_t / cublasFillMode_t, which are int-sized enums.
    EIG_MODE_NOVECTOR = 0,
    FILL_MODE_LOWER = 0
};

inline int max_int(int a, int b)
{
    return (a > b) ? a : b;
}

// Largest Gram dimension the divide-and-conquer eigensolver is trusted with.
//
// cusolverDn?heevd is materially faster than gesvd (2x at rows=256), but on
// real trajectory RDMs it stops converging as the cut grows: at rows=1024 both
// Cheevd and Zheevd return info != 0 on inputs verified to be exactly
// Hermitian, exactly trace-1 and NaN-free, while gesvd on the same states is
// fine.  Synthetic PSD matrices of the same size and rank do *not* reproduce
// it, so this is a property of the solver on physical spectra rather than
// something callers can preprocess away.
//
// Cap the fast path at a size where it is reliable and keep gesvd for the rest.
// Raise or lower with MIPT_TMI_HEEVD_MAX_ROWS if a newer cuSOLVER behaves
// differently -- and re-run tests/rdm_gram_tests plus a sim_tmi run at the
// sizes you care about, watching for the "half-cut SVD was disabled" warning.
inline int heevd_max_rows()
{
    static const int cap = []() {
        const char *env = getenv("MIPT_TMI_HEEVD_MAX_ROWS");
        if (env == NULL || *env == '\0') { return 512; }
        const long value = strtol(env, NULL, 10);
        return (value < 0) ? 0 : static_cast<int>(value);
    }();
    return cap;
}

// Cut sizes where heevd has already failed once in this thread. Retrying it
// every sample would cost an extra eigensolve per call for nothing.
inline bool heevd_known_bad(size_t rows, bool record)
{
    static thread_local size_t bad[32] = {0};
    static thread_local int count = 0;
    for (int i = 0; i < count; ++i)
    {
        if (bad[i] == rows) { return true; }
    }
    if (record && count < 32) { bad[count++] = rows; }
    return false;
}

// rho is a normalized density matrix, so its eigenvalues are the Schmidt
// weights and the entropy is a direct -sum lambda log2 lambda.  Eigenvalues at
// or below the fp32 Gram noise floor are dropped; log2 of a negative roundoff
// eigenvalue would otherwise poison the sum.
inline double entropy_from_eigenvalues(const float *values, int count)
{
    double entropy = 0.0;
    for (int i = 0; i < count; ++i)
    {
        const double lambda = static_cast<double>(values[i]);
        if (lambda > 1.0e-12)
        {
            entropy -= lambda * (log(lambda) / 0.693147180559945309417232121458176568);
        }
    }
    return (fabs(entropy) < 1.0e-12) ? 0.0 : entropy;
}

inline double entropy_from_eigenvalues(const double *values, int count)
{
    double entropy = 0.0;
    for (int i = 0; i < count; ++i)
    {
        const double lambda = values[i];
        if (lambda > 1.0e-15)
        {
            entropy -= lambda * (log(lambda) / 0.693147180559945309417232121458176568);
        }
    }
    return (fabs(entropy) < 1.0e-12) ? 0.0 : entropy;
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

cusolverStatus_t heevd_buffer_size(cusolverDnHandle_t handle,
                                   int n,
                                   const cuComplex *a,
                                   const float *w,
                                   int *lwork)
{
    return cusolverDnCheevd_bufferSize(handle, EIG_MODE_NOVECTOR, FILL_MODE_LOWER,
                                       n, a, n, w, lwork);
}

cusolverStatus_t heevd_buffer_size(cusolverDnHandle_t handle,
                                   int n,
                                   const cuDoubleComplex *a,
                                   const double *w,
                                   int *lwork)
{
    return cusolverDnZheevd_bufferSize(handle, EIG_MODE_NOVECTOR, FILL_MODE_LOWER,
                                       n, a, n, w, lwork);
}

cusolverStatus_t heevd(cusolverDnHandle_t handle,
                       int n,
                       cuComplex *a,
                       float *w,
                       cuComplex *work,
                       int lwork,
                       int *info)
{
    return cusolverDnCheevd(handle, EIG_MODE_NOVECTOR, FILL_MODE_LOWER,
                            n, a, n, w, work, lwork, info);
}

cusolverStatus_t heevd(cusolverDnHandle_t handle,
                       int n,
                       cuDoubleComplex *a,
                       double *w,
                       cuDoubleComplex *work,
                       int lwork,
                       int *info)
{
    return cusolverDnZheevd(handle, EIG_MODE_NOVECTOR, FILL_MODE_LOWER,
                            n, a, n, w, work, lwork, info);
}


// --- gesvd path -------------------------------------------------------------
//
// The original algorithm, kept as the reliable fallback: singular values of the
// Schmidt matrix, no Gram matrix and so no squared condition number.  Slower
// than herk + heevd but it converges on everything we have thrown at it.

cusolverStatus_t gesvd_buffer_size(cusolverDnHandle_t handle, int m, int n,
                                   const cuComplex *, int *lwork)
{
    return cusolverDnCgesvd_bufferSize(handle, m, n, lwork);
}
cusolverStatus_t gesvd_buffer_size(cusolverDnHandle_t handle, int m, int n,
                                   const cuDoubleComplex *, int *lwork)
{
    return cusolverDnZgesvd_bufferSize(handle, m, n, lwork);
}

cusolverStatus_t gesvd_novectors(cusolverDnHandle_t handle, int m, int n,
                                 cuComplex *a, float *s, cuComplex *work,
                                 int lwork, float *rwork, int *info)
{
    return cusolverDnCgesvd(handle, 'N', 'N', m, n, a, m, s, NULL, 1, NULL, 1,
                            work, lwork, rwork, info);
}
cusolverStatus_t gesvd_novectors(cusolverDnHandle_t handle, int m, int n,
                                 cuDoubleComplex *a, double *s,
                                 cuDoubleComplex *work, int lwork,
                                 double *rwork, int *info)
{
    return cusolverDnZgesvd(handle, 'N', 'N', m, n, a, m, s, NULL, 1, NULL, 1,
                            work, lwork, rwork, info);
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
    return (fabs(entropy) < 1.0e-12) ? 0.0 : entropy;
}

template <typename Real, typename ComplexT, typename EigReal>
int entropy_via_gesvd(const void *device_state_vector,
                      int n_qubits,
                      const int *host_kept_modes,
                      int kept_count,
                      int fermion_trace,
                      GramWorkspace<ComplexT> &ws,
                      double *entropy_out)
{
    static thread_local EigReal *d_sigma = NULL;
    static thread_local size_t sigma_capacity = 0;
    static thread_local ComplexT *d_work = NULL;
    static thread_local size_t work_capacity = 0;
    static thread_local EigReal *d_rwork = NULL;
    static thread_local size_t rwork_capacity = 0;
    static thread_local int *d_info = NULL;
    static thread_local size_t info_capacity = 0;

    // build_gram also leaves the gathered Schmidt matrix in ws.matrix, which is
    // what gesvd consumes; the herk it performs is wasted here but is a few ms
    // against a multi-second solve, and sharing one gather keeps the fermionic
    // sign handling in a single place.
    size_t rows = 0;
    size_t cols = 0;
    int status = build_gram<Real, ComplexT>(device_state_vector, n_qubits,
                                            host_kept_modes, kept_count,
                                            fermion_trace, ws, rows, cols);
    if (status != 0)
    {
        return status;
    }
    // gesvd requires m >= n and consumes M as rows x cols with lda = rows, so
    // it can only be used on the layout that stores M itself.
    bool kept_owns_mode_zero = false;
    for (int k = 0; k < kept_count; ++k)
    {
        if (host_kept_modes[k] == 0) { kept_owns_mode_zero = true; break; }
    }
    const int m = static_cast<int>(kept_owns_mode_zero ? rows : cols);
    const int n = static_cast<int>(kept_owns_mode_zero ? cols : rows);
    if (m < n)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int min_dim = (m < n) ? m : n;

    cusolverDnHandle_t handle = NULL;
    status = thread_cusolver_handle(handle);
    if (status != 0)
    {
        return status;
    }

    int lwork = 0;
    cusolverStatus_t solver_status =
        gesvd_buffer_size(handle, m, n, ws.matrix, &lwork);
    if (solver_status != CUSOLVER_STATUS_SUCCESS)
    {
        return CUSOLVER_ERROR_BASE + static_cast<int>(solver_status);
    }

    cudaError_t cuda_status = cudaSuccess;
    if (thread_buffer(static_cast<size_t>(min_dim), d_sigma, sigma_capacity, cuda_status) == NULL ||
        thread_buffer(static_cast<size_t>(max_int(1, lwork)), d_work, work_capacity, cuda_status) == NULL ||
        thread_buffer(static_cast<size_t>(max_int(1, 5 * min_dim)), d_rwork, rwork_capacity, cuda_status) == NULL ||
        thread_buffer(size_t{1}, d_info, info_capacity, cuda_status) == NULL)
    {
        return static_cast<int>(cuda_status);
    }

    solver_status = gesvd_novectors(handle, m, n, ws.matrix, d_sigma, d_work,
                                    lwork, d_rwork, d_info);
    if (solver_status != CUSOLVER_STATUS_SUCCESS)
    {
        return CUSOLVER_ERROR_BASE + static_cast<int>(solver_status);
    }

    int info = 0;
    cuda_status = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) { return static_cast<int>(cuda_status); }
    if (info != 0) { return CUSOLVER_ERROR_BASE + 3000 + info; }

    EigReal *host_sigma = static_cast<EigReal *>(malloc(min_dim * sizeof(EigReal)));
    if (host_sigma == NULL) { return static_cast<int>(cudaErrorMemoryAllocation); }
    cuda_status = cudaMemcpy(host_sigma, d_sigma, min_dim * sizeof(EigReal),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) { free(host_sigma); return static_cast<int>(cuda_status); }
    *entropy_out = entropy_from_singular_values(host_sigma, min_dim);
    free(host_sigma);
    return 0;
}


// Rebuild the Gram matrix and take its eigenvalues in fp64.
//
// Used only when the fp32 eigensolve reports non-convergence.  Rebuilding is
// cheap next to the solve itself (herk is ~3 ms at rows=1024 against ~60 ms for
// the eigensolve), so this costs about one extra eigensolve on the rare samples
// that need it, instead of demoting every later sample in the run to host
// LAPACK.  heevd destroys its input, which is why the Gram has to be rebuilt.
template <typename Real, typename ComplexT>
int retry_entropy_in_f64(const void *device_state_vector,
                         int n_qubits,
                         const int *host_kept_modes,
                         int kept_count,
                         int fermion_trace,
                         GramWorkspace<ComplexT> &ws,
                         double *entropy_out)
{
    static thread_local cuDoubleComplex *d_gram64 = NULL;
    static thread_local size_t gram64_capacity = 0;
    static thread_local double *d_eig64 = NULL;
    static thread_local size_t eig64_capacity = 0;
    static thread_local cuDoubleComplex *d_work64 = NULL;
    static thread_local size_t work64_capacity = 0;
    static thread_local int *d_info64 = NULL;
    static thread_local size_t info64_capacity = 0;

    size_t rows = 0;
    size_t cols = 0;
    int status = build_gram<Real, ComplexT>(device_state_vector, n_qubits,
                                            host_kept_modes, kept_count,
                                            fermion_trace, ws, rows, cols);
    if (status != 0)
    {
        return status;
    }

    const size_t elements = rows * rows;
    cudaError_t cuda_status = cudaSuccess;
    if (thread_buffer(elements, d_gram64, gram64_capacity, cuda_status) == NULL ||
        thread_buffer(rows, d_eig64, eig64_capacity, cuda_status) == NULL ||
        thread_buffer(size_t{1}, d_info64, info64_capacity, cuda_status) == NULL)
    {
        return static_cast<int>(cuda_status);
    }

    const size_t wanted = (elements + GATHER_THREADS - 1) / GATHER_THREADS;
    const int blocks = static_cast<int>(wanted < 65535 ? wanted : size_t{65535});
    promote_gram_to_f64_kernel<ComplexT><<<blocks, GATHER_THREADS>>>(
        ws.gram, d_gram64, static_cast<long long>(elements));
    cuda_status = cudaDeviceSynchronize();
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
    cusolverStatus_t solver_status = cusolverDnZheevd_bufferSize(
        handle, EIG_MODE_NOVECTOR, FILL_MODE_LOWER, static_cast<int>(rows),
        d_gram64, static_cast<int>(rows), d_eig64, &lwork);
    if (solver_status != CUSOLVER_STATUS_SUCCESS)
    {
        return CUSOLVER_ERROR_BASE + static_cast<int>(solver_status);
    }
    if (thread_buffer(static_cast<size_t>(max_int(1, lwork)), d_work64,
                      work64_capacity, cuda_status) == NULL)
    {
        return static_cast<int>(cuda_status);
    }

    solver_status = cusolverDnZheevd(handle, EIG_MODE_NOVECTOR, FILL_MODE_LOWER,
                                     static_cast<int>(rows), d_gram64,
                                     static_cast<int>(rows), d_eig64,
                                     d_work64, lwork, d_info64);
    if (solver_status != CUSOLVER_STATUS_SUCCESS)
    {
        return CUSOLVER_ERROR_BASE + static_cast<int>(solver_status);
    }

    int info = 0;
    cuda_status = cudaMemcpy(&info, d_info64, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess)
    {
        return static_cast<int>(cuda_status);
    }
    if (info != 0)
    {
        return CUSOLVER_ERROR_BASE + 2000 + info;
    }

    double *host_eigenvalues = static_cast<double *>(malloc(rows * sizeof(double)));
    if (host_eigenvalues == NULL)
    {
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    cuda_status = cudaMemcpy(host_eigenvalues, d_eig64, rows * sizeof(double),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess)
    {
        free(host_eigenvalues);
        return static_cast<int>(cuda_status);
    }
    *entropy_out = entropy_from_eigenvalues(host_eigenvalues, static_cast<int>(rows));
    free(host_eigenvalues);
    return 0;
}

template <typename Real, typename ComplexT, typename EigReal>
int launch_subsystem_entropy(const void *device_state_vector,
                             int n_qubits,
                             const int *host_kept_modes,
                             int kept_count,
                             int fermion_trace,
                             double *entropy_out)
{
    if (device_state_vector == NULL || entropy_out == NULL ||
        host_kept_modes == NULL || n_qubits <= 0 || n_qubits >= 63 ||
        kept_count <= 0 || kept_count >= n_qubits || kept_count >= 31)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const size_t rows = size_t{1} << kept_count;
    const size_t cols = size_t{1} << (n_qubits - kept_count);
    if (rows == 1 || cols == 1)
    {
        *entropy_out = 0.0;
        return 0;
    }
    if (rows > static_cast<size_t>(INT_MAX))
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    static thread_local GramWorkspace<ComplexT> ws;
    if (rows > static_cast<size_t>(heevd_max_rows()) || heevd_known_bad(rows, false))
    {
        return entropy_via_gesvd<Real, ComplexT, EigReal>(
            device_state_vector, n_qubits, host_kept_modes, kept_count,
            fermion_trace, ws, entropy_out);
    }

    static thread_local EigReal *d_eigenvalues = NULL;
    static thread_local size_t eigenvalue_capacity = 0;
    static thread_local ComplexT *d_work = NULL;
    static thread_local size_t work_capacity = 0;
    static thread_local int *d_info = NULL;
    static thread_local size_t info_capacity = 0;

    size_t built_rows = 0;
    size_t built_cols = 0;
    int status = build_gram<Real, ComplexT>(device_state_vector, n_qubits,
                                            host_kept_modes, kept_count,
                                            fermion_trace, ws,
                                            built_rows, built_cols);
    if (status != 0)
    {
        return status;
    }

    cusolverDnHandle_t handle = NULL;
    status = thread_cusolver_handle(handle);
    if (status != 0)
    {
        return status;
    }

    cudaError_t cuda_status = cudaSuccess;
    if (thread_buffer(rows, d_eigenvalues, eigenvalue_capacity, cuda_status) == NULL ||
        thread_buffer(size_t{1}, d_info, info_capacity, cuda_status) == NULL)
    {
        return static_cast<int>(cuda_status);
    }

    int lwork = 0;
    cusolverStatus_t solver_status = heevd_buffer_size(
        handle, static_cast<int>(rows), ws.gram, d_eigenvalues, &lwork);
    if (solver_status != CUSOLVER_STATUS_SUCCESS)
    {
        return CUSOLVER_ERROR_BASE + static_cast<int>(solver_status);
    }
    if (thread_buffer(static_cast<size_t>(max_int(1, lwork)), d_work,
                      work_capacity, cuda_status) == NULL)
    {
        return static_cast<int>(cuda_status);
    }

    // heevd overwrites its input, which is fine: ws.gram is scratch.
    solver_status = heevd(handle, static_cast<int>(rows), ws.gram,
                          d_eigenvalues, d_work, lwork, d_info);
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
        // fp32 heevd does not always converge on real trajectory RDMs.  The
        // input was verified clean when this was investigated (trace exactly 1,
        // exactly Hermitian, no NaN), so this is the solver, not the matrix.
        // Redo the eigensolve in fp64 rather than giving up: sim_tmi.exe
        // responds to a failure here by permanently disabling the device
        // half-cut path for the rest of the run and silently falling back to
        // host LAPACK, which costs far more than one retry.
        const int retry = retry_entropy_in_f64<Real, ComplexT>(
            device_state_vector, n_qubits, host_kept_modes, kept_count,
            fermion_trace, ws, entropy_out);
        if (retry == 0)
        {
            return 0;
        }
        // Neither precision converged: stop trying heevd at this cut size and
        // use gesvd, which is slower but has never failed here.
        heevd_known_bad(rows, true);
        return entropy_via_gesvd<Real, ComplexT, EigReal>(
            device_state_vector, n_qubits, host_kept_modes, kept_count,
            fermion_trace, ws, entropy_out);
    }


    EigReal *host_eigenvalues =
        static_cast<EigReal *>(malloc(rows * sizeof(EigReal)));
    if (host_eigenvalues == NULL)
    {
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    cuda_status = cudaMemcpy(host_eigenvalues, d_eigenvalues,
                             rows * sizeof(EigReal), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess)
    {
        free(host_eigenvalues);
        return static_cast<int>(cuda_status);
    }
    *entropy_out = entropy_from_eigenvalues(host_eigenvalues, static_cast<int>(rows));
    free(host_eigenvalues);
    return 0;
}
} // namespace

extern "C" int mipt_cuda_entropy_svd_f32(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *entropy_out)
{
    return launch_subsystem_entropy<float, cuComplex, float>(
        device_state_vector, n_qubits, host_kept_modes, kept_count,
        fermion_trace, entropy_out);
}

extern "C" int mipt_cuda_entropy_svd_f64(
    const void *device_state_vector,
    int n_qubits,
    const int *host_kept_modes,
    int kept_count,
    int fermion_trace,
    double *entropy_out)
{
    return launch_subsystem_entropy<double, cuDoubleComplex, double>(
        device_state_vector, n_qubits, host_kept_modes, kept_count,
        fermion_trace, entropy_out);
}
