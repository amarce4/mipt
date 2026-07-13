#pragma once

#include "mipt/backend.hpp"
#include "mipt/env.hpp"
#include "mipt/types.hpp"

#ifdef __has_include
#if __has_include(<dlfcn.h>)
#include <dlfcn.h>
#define MIPT_TMI_HAS_DLFCN 1
#endif
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace mipt::tmi
{
using C64 = std::complex<double>;
using C32 = std::complex<float>;
constexpr double LOG2_EPS = 1.0e-15;

enum class TraceMode
{
    Qubit,
    Fermion
};

inline std::string sim_tmi_pause_file_path()
{
    const char *value = std::getenv("SIM_TMI_PAUSE_FILE");
    if (env::disables_path(value))
    {
        return {};
    }
    if (value != nullptr && *value != '\0')
    {
        return std::string(value);
    }
    return "PAUSE_MIPT";
}

inline bool file_exists_for_pause(const std::string &path)
{
    if (path.empty())
    {
        return false;
    }
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

inline bool is_absolute_path(const std::string &path)
{
    return !path.empty() && path.front() == '/';
}

inline std::string pause_file_display_path(const std::string &path)
{
    if (path.empty() || is_absolute_path(path))
    {
        return path;
    }

    const char *pwd = std::getenv("PWD");
    if (pwd != nullptr && *pwd != '\0')
    {
        return std::string(pwd) + "/" + path;
    }

    char cwd[4096];
    if (::getcwd(cwd, sizeof(cwd)) != nullptr)
    {
        return std::string(cwd) + "/" + path;
    }
    return path;
}

inline int cuda_svd_min_kept_modes()
{
    // Default policy:
    //   N=8,12  -> keep half-cut SVDs on host LAPACK/OpenBLAS.
    //   N>=16   -> allow the CUDA/cuSOLVER half-cut path.
    //
    // The kept-mode threshold is 8 because the N=16 half-cuts have
    // 2 * block_qubits = 8 kept modes.  N=8 and N=12 are still protected
    // by the explicit small-N guard below, so leaving
    // SIM_TMI_CUDA_SVD_MIN_KEPT=2 in your shell will not accidentally put
    // those sizes back onto cuSOLVER.
    constexpr long default_min_kept_modes = 8;
    constexpr long max_kept_modes = 30;
    static const int threshold = static_cast<int>(
        env::integer("SIM_TMI_CUDA_SVD_MIN_KEPT", default_min_kept_modes, 0,
                     max_kept_modes));
    return threshold;
}

inline bool force_cuda_svd_for_small_n()
{
    static const bool force = env::boolean("SIM_TMI_FORCE_CUDA_SVD_SMALL_N", false);
    return force;
}

inline bool cuda_svd_disabled_for_small_n(int n)
{
    // N=8 and N=12 were faster before the GPU-SVD implementation.  Keep
    // their TMI half-cut SVDs on the CPU by default while retaining GPU
    // circuit simulation when the executable is built with GPU=1.
    return !force_cuda_svd_for_small_n() && (n == 8 || n == 12);
}

inline bool should_try_cuda_svd_for_problem(int n, std::size_t kept_mode_count)
{
    if (cuda_svd_disabled_for_small_n(n))
    {
        return false;
    }
    return kept_mode_count >= static_cast<std::size_t>(cuda_svd_min_kept_modes());
}

inline std::uint64_t csv_flush_row_interval(int n, int cycle_count)
{
    constexpr long max_rows = 1'000'000;
    const long env_rows = env::integer("SIM_TMI_CSV_FLUSH_ROWS", 0, 0, max_rows);
    if (env_rows > 0)
    {
        return static_cast<std::uint64_t>(env_rows);
    }

    // Small-N jobs can produce huge numbers of rows; batch them.  For N>=16
    // flush once per circuit by default, so interrupted runs lose at most one
    // circuit while avoiding one flush per all_cycles row.
    if (n == 8 || n == 12)
    {
        return 1000;
    }
    return static_cast<std::uint64_t>(std::max(1, cycle_count));
}

struct TraceSource
{
    std::size_t source_element = 0;
    int sign = 1;
};

struct TracePlan
{
    int parent_modes = 0;
    int kept_modes = 0;
    std::size_t parent_dim = 0;
    std::size_t kept_dim = 0;
    std::size_t env_terms = 0;
    std::vector<TraceSource> sources;
};

inline double entropy_from_eigenvalue(double lambda)
{
    if (lambda <= LOG2_EPS)
    {
        return 0.0;
    }
    return -lambda * std::log2(lambda);
}

inline double finish_entropy(double entropy)
{
    if (std::abs(entropy) < 1.0e-12)
    {
        return 0.0;
    }
    return entropy;
}

inline void append_double(std::string &s, double value)
{
    char buf[64];
    const auto result = std::to_chars(buf, buf + sizeof(buf), value,
                                      std::chars_format::general, 17);
    if (result.ec == std::errc{})
    {
        s.append(buf, result.ptr);
        return;
    }
    s += std::to_string(value);
}

inline void hermitize_in_place(std::vector<C64> &a, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        a[i * n + i] = C64(a[i * n + i].real(), 0.0);
        for (std::size_t j = i + 1; j < n; ++j)
        {
            const C64 h = 0.5 * (a[i * n + j] + std::conj(a[j * n + i]));
            a[i * n + j] = h;
            a[j * n + i] = std::conj(h);
        }
    }
}

inline double entropy_hermitian_2x2(const C64 *m)
{
    const double a = m[0].real();
    const double d = m[3].real();
    const C64 b = 0.5 * (m[1] + std::conj(m[2]));
    const double half_trace = 0.5 * (a + d);
    const double half_diff = 0.5 * (a - d);
    const double radius = std::sqrt(std::max(0.0, half_diff * half_diff + std::norm(b)));
    return finish_entropy(entropy_from_eigenvalue(half_trace - radius) +
                          entropy_from_eigenvalue(half_trace + radius));
}

inline double entropy_hermitian_jacobi(std::vector<C64> a, std::size_t n)
{
    hermitize_in_place(a, n);

    double scale = 1.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        scale = std::max(scale, std::abs(a[i * n + i].real()));
        for (std::size_t j = i + 1; j < n; ++j)
        {
            scale = std::max(scale, std::abs(a[i * n + j]));
        }
    }

    const double tol = 1.0e-14 * scale;
    const std::size_t max_rotations = std::max<std::size_t>(64u, 128u * n * n);

    for (std::size_t iter = 0; iter < max_rotations; ++iter)
    {
        std::size_t p = 0;
        std::size_t q = 1;
        double max_offdiag = 0.0;
        for (std::size_t i = 0; i + 1 < n; ++i)
        {
            for (std::size_t j = i + 1; j < n; ++j)
            {
                const double v = std::abs(a[i * n + j]);
                if (v > max_offdiag)
                {
                    max_offdiag = v;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_offdiag <= tol)
        {
            break;
        }

        const std::size_t pp = p * n + p;
        const std::size_t qq = q * n + q;
        const std::size_t pq = p * n + q;
        const double app = a[pp].real();
        const double aqq = a[qq].real();
        const C64 apq = a[pq];
        const double b_abs = std::abs(apq);
        if (b_abs <= 0.0)
        {
            break;
        }

        const double tau = (aqq - app) / (2.0 * b_abs);
        const double tau_abs = std::abs(tau);
        const double t = std::copysign(1.0 / (tau_abs + std::sqrt(1.0 + tau_abs * tau_abs)), tau);
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        const C64 phase = apq / b_abs;

        for (std::size_t k = 0; k < n; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }
            const std::size_t kp = k * n + p;
            const std::size_t kq = k * n + q;
            const C64 old_kp = a[kp];
            const C64 old_kq = a[kq];
            const C64 new_kp = c * old_kp * phase - s * old_kq;
            const C64 new_kq = s * old_kp * phase + c * old_kq;
            a[kp] = new_kp;
            a[p * n + k] = std::conj(new_kp);
            a[kq] = new_kq;
            a[q * n + k] = std::conj(new_kq);
        }

        a[pp] = C64(app - t * b_abs, 0.0);
        a[qq] = C64(aqq + t * b_abs, 0.0);
        a[pq] = C64(0.0, 0.0);
        a[q * n + p] = C64(0.0, 0.0);
    }

    double entropy = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        entropy += entropy_from_eigenvalue(a[i * n + i].real());
    }
    return finish_entropy(entropy);
}

namespace lapack_runtime
{
    using zheevd_fn = void (*)(char *, char *, int *, C64 *, int *, double *, C64 *, int *, double *, int *, int *, int *, int *);
    using zgesvd_fn = void (*)(char *, char *, int *, int *, C64 *, int *, double *, C64 *, int *, C64 *, int *, C64 *, int *, double *, int *);
    using cgesvd_fn = void (*)(char *, char *, int *, int *, C32 *, int *, float *, C32 *, int *, C32 *, int *, C32 *, int *, float *, int *);
    using zherk_fn = void (*)(char *, char *, int *, int *, double *, C64 *, int *, double *, C64 *, int *);

    struct Handle
    {
        std::once_flag init_once;
        std::vector<void *> libraries;
        zheevd_fn zheevd = nullptr;
        zgesvd_fn zgesvd = nullptr;
        cgesvd_fn cgesvd = nullptr;
        zherk_fn zherk = nullptr;
        std::string status;
    };

    inline Handle &handle()
    {
        static Handle h;
        return h;
    }

    inline void initialize()
    {
        auto &h = handle();
#ifdef MIPT_TMI_HAS_DLFCN
        const char *library_names[] = {
            "libopenblas.so.0",
            "libopenblas.so",
            "liblapack.so.3",
            "liblapack.so",
            "libblas.so.3",
            "libblas.so"};

        for (const char *name : library_names)
        {
            void *lib = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
            if (!lib)
            {
                continue;
            }
            h.libraries.push_back(lib);
            if (!h.zheevd)
            {
                h.zheevd = reinterpret_cast<zheevd_fn>(dlsym(lib, "zheevd_"));
            }
            if (!h.zgesvd)
            {
                h.zgesvd = reinterpret_cast<zgesvd_fn>(dlsym(lib, "zgesvd_"));
            }
            if (!h.cgesvd)
            {
                h.cgesvd = reinterpret_cast<cgesvd_fn>(dlsym(lib, "cgesvd_"));
            }
            if (!h.zherk)
            {
                h.zherk = reinterpret_cast<zherk_fn>(dlsym(lib, "zherk_"));
            }
        }

        h.status = "LAPACK/BLAS:";
        h.status += h.zheevd ? " zheevd" : " no-zheevd";
        h.status += h.zgesvd ? " zgesvd" : " no-zgesvd";
        h.status += h.cgesvd ? " cgesvd" : " no-cgesvd";
        h.status += h.zherk ? " zherk" : " no-zherk";
        if (!h.zheevd || !h.zgesvd)
        {
            h.status += "; missing symbols force slower fallbacks";
        }
#else
        h.status = "<dlfcn.h> is unavailable; using slower Jacobi/reduction fallbacks";
#endif
    }

    inline zheevd_fn zheevd()
    {
        auto &h = handle();
        std::call_once(h.init_once, initialize);
        return h.zheevd;
    }

    inline zgesvd_fn zgesvd()
    {
        auto &h = handle();
        std::call_once(h.init_once, initialize);
        return h.zgesvd;
    }

    inline cgesvd_fn cgesvd()
    {
        auto &h = handle();
        std::call_once(h.init_once, initialize);
        return h.cgesvd;
    }

    inline zherk_fn zherk()
    {
        auto &h = handle();
        std::call_once(h.init_once, initialize);
        return h.zherk;
    }

    inline const std::string &status()
    {
        auto &h = handle();
        std::call_once(h.init_once, initialize);
        return h.status;
    }
} // namespace lapack_runtime

struct EntropyWorkspace
{
    std::size_t lapack_dim = 0;
    std::vector<double> eigenvalues;
    std::vector<C64> work;
    std::vector<double> rwork;
    std::vector<int> iwork;
};

inline void prepare_lapack_workspace(std::size_t n, EntropyWorkspace &ws)
{
    if (ws.lapack_dim == n)
    {
        return;
    }
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("Matrix dimension exceeds LAPACK int range.");
    }

    auto zheevd = lapack_runtime::zheevd();
    if (!zheevd)
    {
        return;
    }

    int ni = static_cast<int>(n);
    int lda = static_cast<int>(n);
    int info = 0;
    int lwork = -1;
    int lrwork = -1;
    int liwork = -1;
    char jobz = 'N';
    char uplo = 'U';
    C64 work_query = C64(0.0, 0.0);
    double rwork_query = 0.0;
    int iwork_query = 0;
    std::vector<C64> query_matrix(n * n, C64(0.0, 0.0));
    std::vector<double> query_eigs(n, 0.0);

    zheevd(&jobz, &uplo, &ni, query_matrix.data(), &lda, query_eigs.data(),
           &work_query, &lwork, &rwork_query, &lrwork, &iwork_query, &liwork, &info);
    if (info != 0)
    {
        throw std::runtime_error("LAPACK zheevd workspace query failed.");
    }

    const int lwork_opt = std::max(1, static_cast<int>(std::ceil(work_query.real())));
    const int lrwork_opt = std::max(1, static_cast<int>(std::ceil(rwork_query)));
    const int liwork_opt = std::max(1, iwork_query);

    ws.lapack_dim = n;
    ws.eigenvalues.assign(n, 0.0);
    ws.work.assign(static_cast<std::size_t>(lwork_opt), C64(0.0, 0.0));
    ws.rwork.assign(static_cast<std::size_t>(lrwork_opt), 0.0);
    ws.iwork.assign(static_cast<std::size_t>(liwork_opt), 0);
}

inline double entropy_hermitian_inplace(std::vector<C64> &a, std::size_t n, EntropyWorkspace &ws)
{
    if (n == 0 || a.size() != n * n)
    {
        throw std::invalid_argument("Invalid matrix size passed to entropy_hermitian_inplace.");
    }
    if (n == 1)
    {
        return 0.0;
    }
    if (n == 2)
    {
        return entropy_hermitian_2x2(a.data());
    }

    hermitize_in_place(a, n);

    auto zheevd = lapack_runtime::zheevd();
    if (!zheevd)
    {
        return entropy_hermitian_jacobi(std::move(a), n);
    }

    prepare_lapack_workspace(n, ws);

    int ni = static_cast<int>(n);
    int lda = static_cast<int>(n);
    int lwork = static_cast<int>(ws.work.size());
    int lrwork = static_cast<int>(ws.rwork.size());
    int liwork = static_cast<int>(ws.iwork.size());
    int info = 0;
    char jobz = 'N';
    char uplo = 'U';

    zheevd(&jobz, &uplo, &ni, a.data(), &lda, ws.eigenvalues.data(),
           ws.work.data(), &lwork, ws.rwork.data(), &lrwork, ws.iwork.data(), &liwork, &info);
    if (info != 0)
    {
        throw std::runtime_error("LAPACK zheevd failed while computing entropy.");
    }

    double entropy = 0.0;
    for (double lambda : ws.eigenvalues)
    {
        entropy += entropy_from_eigenvalue(lambda);
    }
    return finish_entropy(entropy);
}

inline void fill_complex_from_ri_ptr(const double *rho_ri, std::size_t dim, std::vector<C64> &out)
{
    out.resize(dim * dim);
    for (std::size_t i = 0; i < dim * dim; ++i)
    {
        out[i] = C64(rho_ri[2u * i + 0], rho_ri[2u * i + 1]);
    }
}

inline double entropy_ri_ptr(const double *rho_ri,
                      std::size_t dim,
                      std::vector<C64> &matrix_buffer,
                      EntropyWorkspace &entropy_ws)
{
    fill_complex_from_ri_ptr(rho_ri, dim, matrix_buffer);
    return entropy_hermitian_inplace(matrix_buffer, dim, entropy_ws);
}

struct SvdWorkspace
{
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> singular_values;
    std::vector<C64> work;
    std::vector<double> rwork;
};

struct SvdWorkspaceF32
{
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> singular_values;
    std::vector<C32> work;
    std::vector<float> rwork;
};

template <typename Real>
inline double entropy_from_singular_values_raw(const std::vector<Real> &singular_values)
{
    double entropy = 0.0;
    for (Real sigma_value : singular_values)
    {
        const double sigma = static_cast<double>(sigma_value);
        const double lambda = sigma * sigma;
        entropy += entropy_from_eigenvalue(lambda);
    }
    return finish_entropy(entropy);
}

inline double entropy_from_singular_values(const std::vector<double> &singular_values)
{
    return entropy_from_singular_values_raw(singular_values);
}

inline double entropy_from_singular_values(const std::vector<float> &singular_values)
{
    return entropy_from_singular_values_raw(singular_values);
}

inline void prepare_zgesvd_workspace(std::size_t rows, std::size_t cols, SvdWorkspace &ws)
{
    if (ws.rows == rows && ws.cols == cols)
    {
        return;
    }
    if (rows == 0 || cols == 0 ||
        rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("Invalid matrix dimensions for LAPACK zgesvd.");
    }

    auto zgesvd = lapack_runtime::zgesvd();
    if (!zgesvd)
    {
        throw std::runtime_error("LAPACK zgesvd_ is unavailable.");
    }

    int m = static_cast<int>(rows);
    int n = static_cast<int>(cols);
    int lda = static_cast<int>(rows);
    int min_dim = std::min(m, n);
    int ldu = 1;
    int ldvt = 1;
    int lwork = -1;
    int info = 0;
    char jobu = 'N';
    char jobvt = 'N';
    C64 dummy_a(0.0, 0.0);
    C64 dummy_u(0.0, 0.0);
    C64 dummy_vt(0.0, 0.0);
    C64 work_query(0.0, 0.0);
    std::vector<double> query_s(static_cast<std::size_t>(min_dim), 0.0);
    std::vector<double> query_rwork(static_cast<std::size_t>(std::max(1, 5 * min_dim)), 0.0);

    zgesvd(&jobu, &jobvt, &m, &n, &dummy_a, &lda, query_s.data(),
           &dummy_u, &ldu, &dummy_vt, &ldvt, &work_query, &lwork,
           query_rwork.data(), &info);
    if (info != 0)
    {
        throw std::runtime_error("LAPACK zgesvd workspace query failed.");
    }

    const int lwork_opt = std::max(1, static_cast<int>(std::ceil(work_query.real())));
    ws.rows = rows;
    ws.cols = cols;
    ws.singular_values.assign(static_cast<std::size_t>(min_dim), 0.0);
    ws.work.assign(static_cast<std::size_t>(lwork_opt), C64(0.0, 0.0));
    ws.rwork.assign(static_cast<std::size_t>(std::max(1, 5 * min_dim)), 0.0);
}

inline double entropy_svd_inplace(std::vector<C64> &matrix_col_major,
                           std::size_t rows,
                           std::size_t cols,
                           SvdWorkspace &ws)
{
    if (matrix_col_major.size() != rows * cols)
    {
        throw std::invalid_argument("Invalid matrix size passed to entropy_svd_inplace.");
    }
    if (rows == 1 || cols == 1)
    {
        // A normalized pure-state coefficient vector has exactly one
        // non-zero Schmidt coefficient.
        return 0.0;
    }

    prepare_zgesvd_workspace(rows, cols, ws);
    auto zgesvd = lapack_runtime::zgesvd();
    int m = static_cast<int>(rows);
    int n = static_cast<int>(cols);
    int lda = static_cast<int>(rows);
    int ldu = 1;
    int ldvt = 1;
    int lwork = static_cast<int>(ws.work.size());
    int info = 0;
    char jobu = 'N';
    char jobvt = 'N';
    C64 dummy_u(0.0, 0.0);
    C64 dummy_vt(0.0, 0.0);

    zgesvd(&jobu, &jobvt, &m, &n, matrix_col_major.data(), &lda,
           ws.singular_values.data(), &dummy_u, &ldu, &dummy_vt, &ldvt,
           ws.work.data(), &lwork, ws.rwork.data(), &info);
    if (info != 0)
    {
        throw std::runtime_error("LAPACK zgesvd failed while computing a Schmidt spectrum.");
    }
    return entropy_from_singular_values(ws.singular_values);
}


inline void prepare_cgesvd_workspace(std::size_t rows, std::size_t cols, SvdWorkspaceF32 &ws)
{
    if (ws.rows == rows && ws.cols == cols)
    {
        return;
    }
    if (rows == 0 || cols == 0 ||
        rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("Invalid matrix dimensions for LAPACK cgesvd.");
    }

    auto cgesvd = lapack_runtime::cgesvd();
    if (!cgesvd)
    {
        throw std::runtime_error("LAPACK cgesvd_ is unavailable.");
    }

    int m = static_cast<int>(rows);
    int n = static_cast<int>(cols);
    int lda = static_cast<int>(rows);
    int min_dim = std::min(m, n);
    int ldu = 1;
    int ldvt = 1;
    int lwork = -1;
    int info = 0;
    char jobu = 'N';
    char jobvt = 'N';
    C32 dummy_a(0.0f, 0.0f);
    C32 dummy_u(0.0f, 0.0f);
    C32 dummy_vt(0.0f, 0.0f);
    C32 work_query(0.0f, 0.0f);
    std::vector<float> query_s(static_cast<std::size_t>(min_dim), 0.0f);
    std::vector<float> query_rwork(static_cast<std::size_t>(std::max(1, 5 * min_dim)), 0.0f);

    cgesvd(&jobu, &jobvt, &m, &n, &dummy_a, &lda, query_s.data(),
           &dummy_u, &ldu, &dummy_vt, &ldvt, &work_query, &lwork,
           query_rwork.data(), &info);
    if (info != 0)
    {
        throw std::runtime_error("LAPACK cgesvd workspace query failed.");
    }

    const int lwork_opt = std::max(1, static_cast<int>(std::ceil(work_query.real())));
    ws.rows = rows;
    ws.cols = cols;
    ws.singular_values.assign(static_cast<std::size_t>(min_dim), 0.0f);
    ws.work.assign(static_cast<std::size_t>(lwork_opt), C32(0.0f, 0.0f));
    ws.rwork.assign(static_cast<std::size_t>(std::max(1, 5 * min_dim)), 0.0f);
}

inline double entropy_svd_inplace(std::vector<C32> &matrix_col_major,
                           std::size_t rows,
                           std::size_t cols,
                           SvdWorkspaceF32 &ws)
{
    if (matrix_col_major.size() != rows * cols)
    {
        throw std::invalid_argument("Invalid matrix size passed to entropy_svd_inplace.");
    }
    if (rows == 1 || cols == 1)
    {
        return 0.0;
    }

    prepare_cgesvd_workspace(rows, cols, ws);
    auto cgesvd = lapack_runtime::cgesvd();
    int m = static_cast<int>(rows);
    int n = static_cast<int>(cols);
    int lda = static_cast<int>(rows);
    int ldu = 1;
    int ldvt = 1;
    int lwork = static_cast<int>(ws.work.size());
    int info = 0;
    char jobu = 'N';
    char jobvt = 'N';
    C32 dummy_u(0.0f, 0.0f);
    C32 dummy_vt(0.0f, 0.0f);

    cgesvd(&jobu, &jobvt, &m, &n, matrix_col_major.data(), &lda,
           ws.singular_values.data(), &dummy_u, &ldu, &dummy_vt, &ldvt,
           ws.work.data(), &lwork, ws.rwork.data(), &info);
    if (info != 0)
    {
        throw std::runtime_error("LAPACK cgesvd failed while computing a single-precision Schmidt spectrum.");
    }
    return entropy_from_singular_values(ws.singular_values);
}

inline std::vector<int> complement_modes(int n, const std::vector<int> &modes)
{
    std::vector<unsigned char> kept(static_cast<std::size_t>(n), 0);
    for (int q : modes)
    {
        if (q < 0 || q >= n)
        {
            throw std::invalid_argument("Subsystem mode index is out of range.");
        }
        if (kept[static_cast<std::size_t>(q)])
        {
            throw std::invalid_argument("Subsystem mode list contains duplicates.");
        }
        kept[static_cast<std::size_t>(q)] = 1;
    }

    std::vector<int> env;
    env.reserve(static_cast<std::size_t>(n - static_cast<int>(modes.size())));
    for (int q = 0; q < n; ++q)
    {
        if (!kept[static_cast<std::size_t>(q)])
        {
            env.push_back(q);
        }
    }
    return env;
}

inline std::vector<std::uint64_t> basis_offsets(const std::vector<int> &modes)
{
    if (modes.size() >= 63)
    {
        throw std::invalid_argument("Mode count is too large for uint64_t basis indexing.");
    }
    const std::uint64_t dim = std::uint64_t{1} << modes.size();
    std::vector<std::uint64_t> offsets(static_cast<std::size_t>(dim), 0);
    for (std::uint64_t basis = 0; basis < dim; ++basis)
    {
        std::uint64_t offset = 0;
        for (std::size_t k = 0; k < modes.size(); ++k)
        {
            offset |= ((basis >> k) & std::uint64_t{1}) << modes[k];
        }
        offsets[static_cast<std::size_t>(basis)] = offset;
    }
    return offsets;
}

inline std::vector<std::uint64_t> fermion_cross_masks_by_row(const std::vector<int> &kept_modes,
                                                       const std::vector<int> &env_modes)
{
    if (env_modes.size() >= 63)
    {
        throw std::invalid_argument("Environment mode count is too large for uint64_t sign masks.");
    }
    const std::uint64_t kept_dim = std::uint64_t{1} << kept_modes.size();
    std::vector<std::uint64_t> masks(static_cast<std::size_t>(kept_dim), 0);

    std::vector<std::uint64_t> lower_env_mask(kept_modes.size(), 0);
    for (std::size_t k = 0; k < kept_modes.size(); ++k)
    {
        std::uint64_t mask = 0;
        for (std::size_t e = 0; e < env_modes.size(); ++e)
        {
            if (env_modes[e] < kept_modes[k])
            {
                mask |= std::uint64_t{1} << e;
            }
        }
        lower_env_mask[k] = mask;
    }

    for (std::uint64_t row = 0; row < kept_dim; ++row)
    {
        std::uint64_t mask = 0;
        for (std::size_t k = 0; k < kept_modes.size(); ++k)
        {
            if ((row >> k) & std::uint64_t{1})
            {
                mask ^= lower_env_mask[k];
            }
        }
        masks[static_cast<std::size_t>(row)] = mask;
    }
    return masks;
}

inline bool odd_popcount(std::uint64_t x)
{
#if defined(__GNUG__) || defined(__clang__)
    return (__builtin_popcountll(x) & 1) != 0;
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

template <typename Real>
inline C64 state_value_as_c64(const std::complex<Real> &z)
{
    return C64(static_cast<double>(std::real(z)),
               static_cast<double>(std::imag(z)));
}

inline C32 state_value_as_c32(const std::complex<float> &z)
{
    return C32(std::real(z), std::imag(z));
}

struct StateEntropyWorkspace
{
    SvdWorkspace svd;
    SvdWorkspaceF32 svd_f32;
    EntropyWorkspace entropy;
    std::vector<C64> matrix_col_major;
    std::vector<C32> matrix_col_major_f32;
    std::vector<C64> coeff_chunk_col_major;
    std::vector<C64> rdm_col_major;
    std::vector<C64> rdm_row_major;
    std::vector<C64> row_values;
    std::vector<std::uint64_t> row_offsets;
    std::vector<std::uint64_t> env_offsets;
    std::vector<std::uint64_t> row_cross_masks;
};

template <typename Real>
inline void build_coefficient_matrix_col_major(const std::complex<Real> *psi,
                                        int n,
                                        const std::vector<int> &kept_modes,
                                        bool fermion_trace,
                                        StateEntropyWorkspace &ws)
{
    const std::vector<int> env_modes = complement_modes(n, kept_modes);
    const std::size_t rows = std::size_t{1} << kept_modes.size();
    const std::size_t cols = std::size_t{1} << env_modes.size();
    ws.row_offsets = basis_offsets(kept_modes);
    ws.env_offsets = basis_offsets(env_modes);
    if (fermion_trace)
    {
        ws.row_cross_masks = fermion_cross_masks_by_row(kept_modes, env_modes);
    }
    else
    {
        ws.row_cross_masks.assign(rows, 0);
    }

    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols)
    {
        throw std::bad_alloc();
    }
    ws.matrix_col_major.resize(rows * cols);
    for (std::size_t col = 0; col < cols; ++col)
    {
        const std::uint64_t env_offset = ws.env_offsets[col];
        for (std::size_t row = 0; row < rows; ++row)
        {
            C64 value = state_value_as_c64(psi[env_offset | ws.row_offsets[row]]);
            if (fermion_trace && odd_popcount(static_cast<std::uint64_t>(col) & ws.row_cross_masks[row]))
            {
                value = -value;
            }
            ws.matrix_col_major[row + col * rows] = value;
        }
    }
}

inline void build_coefficient_matrix_col_major_f32(const std::complex<float> *psi,
                                            int n,
                                            const std::vector<int> &kept_modes,
                                            bool fermion_trace,
                                            StateEntropyWorkspace &ws)
{
    const std::vector<int> env_modes = complement_modes(n, kept_modes);
    const std::size_t rows = std::size_t{1} << kept_modes.size();
    const std::size_t cols = std::size_t{1} << env_modes.size();
    ws.row_offsets = basis_offsets(kept_modes);
    ws.env_offsets = basis_offsets(env_modes);
    if (fermion_trace)
    {
        ws.row_cross_masks = fermion_cross_masks_by_row(kept_modes, env_modes);
    }
    else
    {
        ws.row_cross_masks.assign(rows, 0);
    }

    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols)
    {
        throw std::bad_alloc();
    }
    ws.matrix_col_major_f32.resize(rows * cols);
    for (std::size_t col = 0; col < cols; ++col)
    {
        const std::uint64_t env_offset = ws.env_offsets[col];
        for (std::size_t row = 0; row < rows; ++row)
        {
            C32 value = state_value_as_c32(psi[env_offset | ws.row_offsets[row]]);
            if (fermion_trace && odd_popcount(static_cast<std::uint64_t>(col) & ws.row_cross_masks[row]))
            {
                value = -value;
            }
            ws.matrix_col_major_f32[row + col * rows] = value;
        }
    }
}

template <typename Real>
inline double entropy_subsystem_svd_from_state(const std::complex<Real> *psi,
                                        int n,
                                        const std::vector<int> &kept_modes,
                                        bool fermion_trace,
                                        StateEntropyWorkspace &ws)
{
    const std::vector<int> env_modes = complement_modes(n, kept_modes);
    const std::size_t rows = std::size_t{1} << kept_modes.size();
    const std::size_t cols = std::size_t{1} << env_modes.size();

    if constexpr (std::is_same_v<Real, float>)
    {
        if (lapack_runtime::cgesvd())
        {
            build_coefficient_matrix_col_major_f32(psi, n, kept_modes, fermion_trace, ws);
            return entropy_svd_inplace(ws.matrix_col_major_f32, rows, cols, ws.svd_f32);
        }
    }

    if (!lapack_runtime::zgesvd())
    {
        throw std::runtime_error("Fast state-vector TMI needs LAPACK zgesvd_ or cgesvd_.");
    }
    build_coefficient_matrix_col_major(psi, n, kept_modes, fermion_trace, ws);
    return entropy_svd_inplace(ws.matrix_col_major, rows, cols, ws.svd);
}

template <typename Real>
inline void reduced_density_manual_from_state(const std::complex<Real> *psi,
                                       int n,
                                       const std::vector<int> &kept_modes,
                                       bool fermion_trace,
                                       StateEntropyWorkspace &ws,
                                       std::vector<C64> &out_row_major)
{
    const std::vector<int> env_modes = complement_modes(n, kept_modes);
    const std::size_t rows = std::size_t{1} << kept_modes.size();
    const std::size_t cols = std::size_t{1} << env_modes.size();
    ws.row_offsets = basis_offsets(kept_modes);
    ws.env_offsets = basis_offsets(env_modes);
    ws.row_values.assign(rows, C64(0.0, 0.0));
    if (fermion_trace)
    {
        ws.row_cross_masks = fermion_cross_masks_by_row(kept_modes, env_modes);
    }
    else
    {
        ws.row_cross_masks.assign(rows, 0);
    }

    out_row_major.assign(rows * rows, C64(0.0, 0.0));
    for (std::size_t col = 0; col < cols; ++col)
    {
        const std::uint64_t env_offset = ws.env_offsets[col];
        for (std::size_t row = 0; row < rows; ++row)
        {
            C64 value = state_value_as_c64(psi[env_offset | ws.row_offsets[row]]);
            if (fermion_trace && odd_popcount(static_cast<std::uint64_t>(col) & ws.row_cross_masks[row]))
            {
                value = -value;
            }
            ws.row_values[row] = value;
        }
        for (std::size_t row = 0; row < rows; ++row)
        {
            const C64 a = ws.row_values[row];
            for (std::size_t crow = 0; crow < rows; ++crow)
            {
                out_row_major[row * rows + crow] += a * std::conj(ws.row_values[crow]);
            }
        }
    }
}

template <typename Real>
inline bool reduced_density_zherk_from_state(const std::complex<Real> *psi,
                                      int n,
                                      const std::vector<int> &kept_modes,
                                      bool fermion_trace,
                                      StateEntropyWorkspace &ws,
                                      std::vector<C64> &out_row_major)
{
    auto zherk = lapack_runtime::zherk();
    if (!zherk)
    {
        return false;
    }

    const std::vector<int> env_modes = complement_modes(n, kept_modes);
    const std::size_t rows = std::size_t{1} << kept_modes.size();
    const std::size_t cols = std::size_t{1} << env_modes.size();
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    ws.row_offsets = basis_offsets(kept_modes);
    ws.env_offsets = basis_offsets(env_modes);
    if (fermion_trace)
    {
        ws.row_cross_masks = fermion_cross_masks_by_row(kept_modes, env_modes);
    }
    else
    {
        ws.row_cross_masks.assign(rows, 0);
    }

    ws.rdm_col_major.assign(rows * rows, C64(0.0, 0.0));
    const std::size_t target_chunk_bytes = 64u << 20;
    const std::size_t min_chunk_cols = 1;
    const std::size_t max_chunk_cols = std::max<std::size_t>(
        min_chunk_cols,
        target_chunk_bytes / std::max<std::size_t>(1, rows * sizeof(C64)));

    char uplo = 'U';
    char trans = 'N';
    int n_blas = static_cast<int>(rows);
    int lda = static_cast<int>(rows);
    int ldc = static_cast<int>(rows);
    double alpha = 1.0;
    bool first = true;

    for (std::size_t col0 = 0; col0 < cols; col0 += max_chunk_cols)
    {
        const std::size_t chunk_cols = std::min(max_chunk_cols, cols - col0);
        ws.coeff_chunk_col_major.resize(rows * chunk_cols);
        for (std::size_t local_col = 0; local_col < chunk_cols; ++local_col)
        {
            const std::size_t col = col0 + local_col;
            const std::uint64_t env_offset = ws.env_offsets[col];
            for (std::size_t row = 0; row < rows; ++row)
            {
                C64 value = state_value_as_c64(psi[env_offset | ws.row_offsets[row]]);
                if (fermion_trace && odd_popcount(static_cast<std::uint64_t>(col) & ws.row_cross_masks[row]))
                {
                    value = -value;
                }
                ws.coeff_chunk_col_major[row + local_col * rows] = value;
            }
        }

        int k_blas = static_cast<int>(chunk_cols);
        double beta = first ? 0.0 : 1.0;
        zherk(&uplo, &trans, &n_blas, &k_blas, &alpha,
              ws.coeff_chunk_col_major.data(), &lda, &beta,
              ws.rdm_col_major.data(), &ldc);
        first = false;
    }

    out_row_major.assign(rows * rows, C64(0.0, 0.0));
    for (std::size_t j = 0; j < rows; ++j)
    {
        for (std::size_t i = 0; i <= j; ++i)
        {
            C64 value = ws.rdm_col_major[i + j * rows];
            if (i == j)
            {
                value = C64(value.real(), 0.0);
            }
            out_row_major[i * rows + j] = value;
            out_row_major[j * rows + i] = std::conj(value);
        }
    }
    return true;
}

template <typename Real>
inline double entropy_subsystem_rdm_from_state(const std::complex<Real> *psi,
                                        int n,
                                        const std::vector<int> &kept_modes,
                                        bool fermion_trace,
                                        StateEntropyWorkspace &ws)
{
    const std::size_t rows = std::size_t{1} << kept_modes.size();
    if (rows == 1)
    {
        return 0.0;
    }

    if (!reduced_density_zherk_from_state(psi, n, kept_modes, fermion_trace, ws, ws.rdm_row_major))
    {
        reduced_density_manual_from_state(psi, n, kept_modes, fermion_trace, ws, ws.rdm_row_major);
    }
    return entropy_hermitian_inplace(ws.rdm_row_major, rows, ws.entropy);
}

inline int index_from_new_mode_order(int new_index, const std::vector<int> &new_mode_order)
{
    int old_index = 0;
    for (int slot = 0; slot < static_cast<int>(new_mode_order.size()); ++slot)
    {
        old_index |= ((new_index >> slot) & 1) << new_mode_order[static_cast<std::size_t>(slot)];
    }
    return old_index;
}

inline int fermionic_reorder_sign(int new_index, const std::vector<int> &new_mode_order)
{
    const int n = static_cast<int>(new_mode_order.size());
    std::vector<int> pos_new(static_cast<std::size_t>(n), 0);
    for (int slot = 0; slot < n; ++slot)
    {
        pos_new[static_cast<std::size_t>(new_mode_order[static_cast<std::size_t>(slot)])] = slot;
    }

    int inversions = 0;
    for (int mode_i = 0; mode_i < n; ++mode_i)
    {
        const int slot_i = pos_new[static_cast<std::size_t>(mode_i)];
        const int occ_i = (new_index >> slot_i) & 1;
        if (!occ_i)
        {
            continue;
        }
        for (int mode_j = mode_i + 1; mode_j < n; ++mode_j)
        {
            const int slot_j = pos_new[static_cast<std::size_t>(mode_j)];
            const int occ_j = (new_index >> slot_j) & 1;
            if (occ_j && slot_i > slot_j)
            {
                inversions ^= 1;
            }
        }
    }
    return inversions ? -1 : 1;
}

inline TracePlan make_trace_plan(int parent_modes,
                          const std::vector<int> &keep_positions,
                          TraceMode mode)
{
    if (parent_modes <= 0 || parent_modes >= 31)
    {
        throw std::invalid_argument("Invalid parent mode count in TMI partial trace.");
    }
    if (keep_positions.empty())
    {
        throw std::invalid_argument("TMI partial trace cannot retain zero modes.");
    }

    std::vector<unsigned char> keep_mask(static_cast<std::size_t>(parent_modes), 0);
    for (int q : keep_positions)
    {
        if (q < 0 || q >= parent_modes)
        {
            throw std::invalid_argument("TMI partial-trace keep position is out of range.");
        }
        if (keep_mask[static_cast<std::size_t>(q)])
        {
            throw std::invalid_argument("TMI partial-trace keep list contains duplicates.");
        }
        keep_mask[static_cast<std::size_t>(q)] = 1;
    }

    TracePlan plan;
    plan.parent_modes = parent_modes;
    plan.kept_modes = static_cast<int>(keep_positions.size());
    plan.parent_dim = std::size_t{1} << parent_modes;
    plan.kept_dim = std::size_t{1} << plan.kept_modes;
    const int traced_modes = parent_modes - plan.kept_modes;
    plan.env_terms = std::size_t{1} << traced_modes;

    std::vector<int> traced_positions;
    traced_positions.reserve(static_cast<std::size_t>(traced_modes));
    for (int q = 0; q < parent_modes; ++q)
    {
        if (!keep_mask[static_cast<std::size_t>(q)])
        {
            traced_positions.push_back(q);
        }
    }

    std::vector<int> new_mode_order;
    new_mode_order.reserve(static_cast<std::size_t>(parent_modes));
    new_mode_order.insert(new_mode_order.end(), keep_positions.begin(), keep_positions.end());
    new_mode_order.insert(new_mode_order.end(), traced_positions.begin(), traced_positions.end());

    plan.sources.reserve(plan.kept_dim * plan.kept_dim * plan.env_terms);
    for (std::size_t row_k = 0; row_k < plan.kept_dim; ++row_k)
    {
        for (std::size_t col_k = 0; col_k < plan.kept_dim; ++col_k)
        {
            for (std::size_t env = 0; env < plan.env_terms; ++env)
            {
                const int row_new = static_cast<int>(row_k + plan.kept_dim * env);
                const int col_new = static_cast<int>(col_k + plan.kept_dim * env);
                const int row = index_from_new_mode_order(row_new, new_mode_order);
                const int col = index_from_new_mode_order(col_new, new_mode_order);

                int sign = 1;
                if (mode == TraceMode::Fermion)
                {
                    sign = fermionic_reorder_sign(row_new, new_mode_order) *
                           fermionic_reorder_sign(col_new, new_mode_order);
                }
                plan.sources.push_back({static_cast<std::size_t>(row) * plan.parent_dim +
                                            static_cast<std::size_t>(col),
                                        sign});
            }
        }
    }
    return plan;
}

inline void partial_trace_ri_ptr_to(const double *rho_ri,
                             const TracePlan &plan,
                             std::vector<C64> &out)
{
    out.assign(plan.kept_dim * plan.kept_dim, C64(0.0, 0.0));
    std::size_t cursor = 0;
    for (std::size_t element = 0; element < out.size(); ++element)
    {
        double re = 0.0;
        double im = 0.0;
        for (std::size_t t = 0; t < plan.env_terms; ++t)
        {
            const auto src = plan.sources[cursor++];
            re += static_cast<double>(src.sign) * rho_ri[2u * src.source_element + 0];
            im += static_cast<double>(src.sign) * rho_ri[2u * src.source_element + 1];
        }
        out[element] = C64(re, im);
    }
}

} // namespace mipt::tmi
