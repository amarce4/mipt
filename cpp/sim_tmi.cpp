// sim_tmi.cpp
//
// Online TMI simulator for the MIPT codebase.  This executable reuses the
// circuit-generation machinery from mipt.cpp, but it does not write rho_tmi.bin.
// Instead, it immediately computes
//
//   I(A:B:C) = S(A) + S(B) + S(C) + S(D) - S(AB) - S(AC) - S(BC)
//
// and writes only:
//
//   p,tmi
//
// to CSV, one TMI value per line.  With all_cycles=1 it writes N/4 cyclic
// TMI values per circuit, still as separate p,tmi rows.  The fast path avoids
// materializing the half-system reduced density
// matrices: S(AB), S(AC), and S(BC) are computed from Schmidt singular values,
// while the small one-block entropies use BLAS Hermitian rank-k updates when
// BLAS is available at runtime.

#ifdef __has_include
#if __has_include(<dlfcn.h>)
#include <dlfcn.h>
#define SIM_TMI_HAS_DLFCN 1
#endif
#endif

#include <chrono>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#include <string_view>
#include <system_error>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>

// Reuse the existing CUDA-Q kernels, random circuit frontends, fermionic gates,
// and TMI density-matrix reducers.  Rename mipt.cpp's CLI entry point so this
// translation unit can provide its own main().
#define main mipt_original_main_for_sim_tmi
#include "mipt.cpp"
#undef main

#ifdef MIPT_ENABLE_CUDA_RHO
#include "tmi_cuda_svd.hpp"
#include "tmi_cuda_rdm.hpp"
#endif


// FRGS data structures and kernels are provided by mipt.cpp, which is included above.

namespace sim_tmi
{
    using C64 = std::complex<double>;
    using C32 = std::complex<float>;
    constexpr double LOG2_EPS = 1.0e-15;

    enum class TraceMode
    {
        Qubit,
        Fermion
    };

    long parse_nonnegative_env_long(const char *name, long fallback, long max_value)
    {
        const char *value = std::getenv(name);
        if (value == nullptr || *value == '\0')
        {
            return fallback;
        }

        char *end = nullptr;
        errno = 0;
        const long parsed = std::strtol(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > max_value)
        {
            return fallback;
        }
        return parsed;
    }

    bool parse_bool01_env(const char *name, bool fallback)
    {
        const char *value = std::getenv(name);
        if (value == nullptr || *value == '\0')
        {
            return fallback;
        }
        if (std::strcmp(value, "1") == 0 ||
            std::strcmp(value, "true") == 0 ||
            std::strcmp(value, "TRUE") == 0 ||
            std::strcmp(value, "yes") == 0 ||
            std::strcmp(value, "YES") == 0)
        {
            return true;
        }
        if (std::strcmp(value, "0") == 0 ||
            std::strcmp(value, "false") == 0 ||
            std::strcmp(value, "FALSE") == 0 ||
            std::strcmp(value, "no") == 0 ||
            std::strcmp(value, "NO") == 0)
        {
            return false;
        }
        return fallback;
    }

    bool env_value_disables_path(const char *value)
    {
        return value != nullptr &&
               (std::strcmp(value, "0") == 0 ||
                std::strcmp(value, "off") == 0 ||
                std::strcmp(value, "OFF") == 0 ||
                std::strcmp(value, "none") == 0 ||
                std::strcmp(value, "NONE") == 0 ||
                std::strcmp(value, "disabled") == 0 ||
                std::strcmp(value, "DISABLED") == 0);
    }

    std::string sim_tmi_pause_file_path()
    {
        const char *value = std::getenv("SIM_TMI_PAUSE_FILE");
        if (env_value_disables_path(value))
        {
            return {};
        }
        if (value != nullptr && *value != '\0')
        {
            return std::string(value);
        }
        return "PAUSE_MIPT";
    }

    bool file_exists_for_pause(const std::string &path)
    {
        if (path.empty())
        {
            return false;
        }
        struct stat st;
        return ::stat(path.c_str(), &st) == 0;
    }

    bool is_absolute_path(const std::string &path)
    {
        return !path.empty() && path.front() == '/';
    }

    std::string pause_file_display_path(const std::string &path)
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

    int cuda_svd_min_kept_modes()
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
            parse_nonnegative_env_long("SIM_TMI_CUDA_SVD_MIN_KEPT",
                                       default_min_kept_modes,
                                       max_kept_modes));
        return threshold;
    }

    bool force_cuda_svd_for_small_n()
    {
        static const bool force = parse_bool01_env("SIM_TMI_FORCE_CUDA_SVD_SMALL_N", false);
        return force;
    }

    bool cuda_svd_disabled_for_small_n(int n)
    {
        // N=8 and N=12 were faster before the GPU-SVD implementation.  Keep
        // their TMI half-cut SVDs on the CPU by default while retaining GPU
        // circuit simulation when the executable is built with GPU=1.
        return !force_cuda_svd_for_small_n() && (n == 8 || n == 12);
    }

    bool should_try_cuda_svd_for_problem(int n, std::size_t kept_mode_count)
    {
        if (cuda_svd_disabled_for_small_n(n))
        {
            return false;
        }
        return kept_mode_count >= static_cast<std::size_t>(cuda_svd_min_kept_modes());
    }

    std::uint64_t csv_flush_row_interval(int n, int cycle_count)
    {
        constexpr long max_rows = 1'000'000;
        const long env_rows = parse_nonnegative_env_long("SIM_TMI_CSV_FLUSH_ROWS", 0, max_rows);
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

    double entropy_from_eigenvalue(double lambda)
    {
        if (lambda <= LOG2_EPS)
        {
            return 0.0;
        }
        return -lambda * std::log2(lambda);
    }

    double finish_entropy(double entropy)
    {
        if (std::abs(entropy) < 1.0e-12)
        {
            return 0.0;
        }
        return entropy;
    }

    void append_double(std::string &s, double value)
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

    void hermitize_in_place(std::vector<C64> &a, std::size_t n)
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

    double entropy_hermitian_2x2(const C64 *m)
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

    double entropy_hermitian_jacobi(std::vector<C64> a, std::size_t n)
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

        Handle &handle()
        {
            static Handle h;
            return h;
        }

        void initialize()
        {
            auto &h = handle();
#ifdef SIM_TMI_HAS_DLFCN
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

        zheevd_fn zheevd()
        {
            auto &h = handle();
            std::call_once(h.init_once, initialize);
            return h.zheevd;
        }

        zgesvd_fn zgesvd()
        {
            auto &h = handle();
            std::call_once(h.init_once, initialize);
            return h.zgesvd;
        }

        cgesvd_fn cgesvd()
        {
            auto &h = handle();
            std::call_once(h.init_once, initialize);
            return h.cgesvd;
        }

        zherk_fn zherk()
        {
            auto &h = handle();
            std::call_once(h.init_once, initialize);
            return h.zherk;
        }

        const std::string &status()
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

    void prepare_lapack_workspace(std::size_t n, EntropyWorkspace &ws)
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

    double entropy_hermitian_inplace(std::vector<C64> &a, std::size_t n, EntropyWorkspace &ws)
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

    void fill_complex_from_ri_ptr(const double *rho_ri, std::size_t dim, std::vector<C64> &out)
    {
        out.resize(dim * dim);
        for (std::size_t i = 0; i < dim * dim; ++i)
        {
            out[i] = C64(rho_ri[2u * i + 0], rho_ri[2u * i + 1]);
        }
    }

    double entropy_ri_ptr(const double *rho_ri,
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
    double entropy_from_singular_values_raw(const std::vector<Real> &singular_values)
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

    double entropy_from_singular_values(const std::vector<double> &singular_values)
    {
        return entropy_from_singular_values_raw(singular_values);
    }

    double entropy_from_singular_values(const std::vector<float> &singular_values)
    {
        return entropy_from_singular_values_raw(singular_values);
    }

    void prepare_zgesvd_workspace(std::size_t rows, std::size_t cols, SvdWorkspace &ws)
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

    double entropy_svd_inplace(std::vector<C64> &matrix_col_major,
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


    void prepare_cgesvd_workspace(std::size_t rows, std::size_t cols, SvdWorkspaceF32 &ws)
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

    double entropy_svd_inplace(std::vector<C32> &matrix_col_major,
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

    std::vector<int> complement_modes(int n, const std::vector<int> &modes)
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

    std::vector<std::uint64_t> basis_offsets(const std::vector<int> &modes)
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

    std::vector<std::uint64_t> fermion_cross_masks_by_row(const std::vector<int> &kept_modes,
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
    void build_coefficient_matrix_col_major(const std::complex<Real> *psi,
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

    void build_coefficient_matrix_col_major_f32(const std::complex<float> *psi,
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
    double entropy_subsystem_svd_from_state(const std::complex<Real> *psi,
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
    void reduced_density_manual_from_state(const std::complex<Real> *psi,
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
    bool reduced_density_zherk_from_state(const std::complex<Real> *psi,
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
    double entropy_subsystem_rdm_from_state(const std::complex<Real> *psi,
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

    int index_from_new_mode_order(int new_index, const std::vector<int> &new_mode_order)
    {
        int old_index = 0;
        for (int slot = 0; slot < static_cast<int>(new_mode_order.size()); ++slot)
        {
            old_index |= ((new_index >> slot) & 1) << new_mode_order[static_cast<std::size_t>(slot)];
        }
        return old_index;
    }

    int fermionic_reorder_sign(int new_index, const std::vector<int> &new_mode_order)
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

    TracePlan make_trace_plan(int parent_modes,
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

    void partial_trace_ri_ptr_to(const double *rho_ri,
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

    std::vector<int> lower_block_positions(std::uint32_t block_qubits)
    {
        std::vector<int> positions;
        positions.reserve(block_qubits);
        for (std::uint32_t i = 0; i < block_qubits; ++i)
        {
            positions.push_back(static_cast<int>(i));
        }
        return positions;
    }

    std::vector<int> upper_block_positions(std::uint32_t block_qubits)
    {
        std::vector<int> positions;
        positions.reserve(block_qubits);
        for (std::uint32_t i = 0; i < block_qubits; ++i)
        {
            positions.push_back(static_cast<int>(block_qubits + i));
        }
        return positions;
    }

    struct TmiPlans
    {
        TracePlan ab_to_a;
        TracePlan ab_to_b;
        TracePlan ac_to_c;
    };

    TmiPlans make_tmi_plans(std::uint32_t block_qubits, TraceMode mode)
    {
        const int pair_modes = static_cast<int>(2u * block_qubits);
        return {
            make_trace_plan(pair_modes, lower_block_positions(block_qubits), mode),
            make_trace_plan(pair_modes, upper_block_positions(block_qubits), mode),
            make_trace_plan(pair_modes, upper_block_positions(block_qubits), mode),
        };
    }

    std::vector<int> cyclic_mode_range(int begin, int count, int n)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("cyclic_mode_range requires positive n.");
        }
        if (count < 0 || count > n)
        {
            throw std::invalid_argument("Invalid cyclic subsystem block size.");
        }

        std::vector<int> modes;
        modes.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i)
        {
            int q = (begin + i) % n;
            if (q < 0)
            {
                q += n;
            }
            modes.push_back(q);
        }
        return modes;
    }

    std::vector<TmiMatrixTerm> tmi_terms_1d_cycle(int n, int cycle_offset)
    {
        if (n < 4 || (n % 4) != 0)
        {
            throw std::invalid_argument("TMI output requires N=L >= 4 and divisible by 4.");
        }

        const int block = n / 4;
        if (cycle_offset < 0 || cycle_offset >= block)
        {
            throw std::invalid_argument("TMI cycle offset is out of range.");
        }

        const std::vector<int> A = cyclic_mode_range(cycle_offset + 0 * block, block, n);
        const std::vector<int> B = cyclic_mode_range(cycle_offset + 1 * block, block, n);
        const std::vector<int> C = cyclic_mode_range(cycle_offset + 2 * block, block, n);
        const std::vector<int> D = cyclic_mode_range(cycle_offset + 3 * block, block, n);

        return {
            {"AB", concatenate_modes(A, B)},
            {"AC", concatenate_modes(A, C)},
            {"BC", concatenate_modes(B, C)},
            {"D", D},
        };
    }

    int tmi_cycle_count(int n, bool all_cycles)
    {
        return all_cycles ? (n / 4) : 1;
    }

    struct TmiWorkspace
    {
        EntropyWorkspace entropy;
        std::vector<C64> rho_a;
        std::vector<C64> rho_b;
        std::vector<C64> rho_c;
        std::vector<C64> direct_matrix;

        StateEntropyWorkspace entropy_a;
        StateEntropyWorkspace entropy_b;
        StateEntropyWorkspace entropy_c;
        StateEntropyWorkspace entropy_d;
        // Reused for AB, AC, and BC half-cut Schmidt SVDs.  For N=24 this
        // avoids retaining three separate 4096x4096 complex work matrices.
        StateEntropyWorkspace entropy_half;
        bool cuda_svd_disabled = false;
        int cuda_svd_failure_status = 0;
        bool cuda_small_rdm_disabled = false;
        int cuda_small_rdm_failure_status = 0;
        std::vector<double> device_rdm_ri;
        std::vector<std::complex<double>> host_state_f64;
        std::vector<std::complex<float>> host_state_f32;
    };

    double tmi_from_payload(const std::vector<double> &payload,
                            std::uint32_t block_qubits,
                            const std::vector<TmiMatrixTerm> &terms,
                            const TmiPlans &plans,
                            TmiWorkspace &workspace)
    {
        if (terms.size() != 4)
        {
            throw std::invalid_argument("Online TMI expects four terms: AB, AC, BC, D.");
        }

        std::array<const double *, 4> term_ptrs{};
        std::size_t offset = 0;
        for (std::size_t i = 0; i < terms.size(); ++i)
        {
            term_ptrs[i] = payload.data() + offset;
            offset += term_value_count(terms[i]);
        }
        if (offset != payload.size())
        {
            throw std::runtime_error("Internal TMI payload-size mismatch.");
        }

        const std::size_t block_dim = std::size_t{1} << block_qubits;
        const std::size_t pair_dim = std::size_t{1} << (2u * block_qubits);

        partial_trace_ri_ptr_to(term_ptrs[0], plans.ab_to_a, workspace.rho_a);
        partial_trace_ri_ptr_to(term_ptrs[0], plans.ab_to_b, workspace.rho_b);
        partial_trace_ri_ptr_to(term_ptrs[1], plans.ac_to_c, workspace.rho_c);

        const double s_a = entropy_hermitian_inplace(workspace.rho_a, block_dim, workspace.entropy);
        const double s_b = entropy_hermitian_inplace(workspace.rho_b, block_dim, workspace.entropy);
        const double s_c = entropy_hermitian_inplace(workspace.rho_c, block_dim, workspace.entropy);
        const double s_d = entropy_ri_ptr(term_ptrs[3], block_dim, workspace.direct_matrix, workspace.entropy);
        const double s_ab = entropy_ri_ptr(term_ptrs[0], pair_dim, workspace.direct_matrix, workspace.entropy);
        const double s_ac = entropy_ri_ptr(term_ptrs[1], pair_dim, workspace.direct_matrix, workspace.entropy);
        const double s_bc = entropy_ri_ptr(term_ptrs[2], pair_dim, workspace.direct_matrix, workspace.entropy);

        return s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
    }

    template <typename Real>
    double tmi_from_statevector(const std::complex<Real> *psi,
                                int n,
                                std::uint32_t block_qubits,
                                int cycle_offset,
                                bool fermion_trace,
                                TmiWorkspace &workspace)
    {
        const int block = static_cast<int>(block_qubits);
        if (cycle_offset < 0 || cycle_offset >= block)
        {
            throw std::invalid_argument("TMI cycle offset is out of range.");
        }

        const std::vector<int> A = cyclic_mode_range(cycle_offset + 0 * block, block, n);
        const std::vector<int> B = cyclic_mode_range(cycle_offset + 1 * block, block, n);
        const std::vector<int> C = cyclic_mode_range(cycle_offset + 2 * block, block, n);
        const std::vector<int> D = cyclic_mode_range(cycle_offset + 3 * block, block, n);
        const std::vector<int> AB = concatenate_modes(A, B);
        const std::vector<int> AC = concatenate_modes(A, C);
        const std::vector<int> BC = concatenate_modes(B, C);

        const double s_a = entropy_subsystem_rdm_from_state(
            psi, n, A, fermion_trace, workspace.entropy_a);
        const double s_b = entropy_subsystem_rdm_from_state(
            psi, n, B, fermion_trace, workspace.entropy_b);
        const double s_c = entropy_subsystem_rdm_from_state(
            psi, n, C, fermion_trace, workspace.entropy_c);
        const double s_d = entropy_subsystem_rdm_from_state(
            psi, n, D, fermion_trace, workspace.entropy_d);

        const double s_ab = entropy_subsystem_svd_from_state(
            psi, n, AB, fermion_trace, workspace.entropy_half);
        const double s_ac = entropy_subsystem_svd_from_state(
            psi, n, AC, fermion_trace, workspace.entropy_half);
        const double s_bc = entropy_subsystem_svd_from_state(
            psi, n, BC, fermion_trace, workspace.entropy_half);

        return s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
    }


    bool entropy_subsystem_svd_from_device_state(const void *device_state_vector,
                                                 bool device_state_is_fp64,
                                                 int n,
                                                 const std::vector<int> &kept_modes,
                                                 bool fermion_trace,
                                                 double &entropy_out,
                                                 int &status_out)
    {
        status_out = 0;
#ifdef MIPT_ENABLE_CUDA_RHO
        if (device_state_vector == nullptr || kept_modes.empty() ||
            kept_modes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            status_out = -1;
            return false;
        }

        const int status = device_state_is_fp64
                               ? mipt_cuda_entropy_svd_f64(device_state_vector,
                                                           n,
                                                           kept_modes.data(),
                                                           static_cast<int>(kept_modes.size()),
                                                           fermion_trace ? 1 : 0,
                                                           &entropy_out)
                               : mipt_cuda_entropy_svd_f32(device_state_vector,
                                                           n,
                                                           kept_modes.data(),
                                                           static_cast<int>(kept_modes.size()),
                                                           fermion_trace ? 1 : 0,
                                                           &entropy_out);
        status_out = status;
        return status == 0 && std::isfinite(entropy_out);
#else
        (void)device_state_vector;
        (void)device_state_is_fp64;
        (void)n;
        (void)kept_modes;
        (void)fermion_trace;
        (void)entropy_out;
        status_out = -1;
        return false;
#endif
    }

    bool device_small_rdm_enabled()
    {
        // Off by default because the optimal threshold is hardware dependent.
        // Enable with SIM_TMI_DEVICE_SMALL_RDM=1 to avoid full GPU->host
        // statevector copies for the A/B/C/D small-block entropies.
        static const bool enabled = mipt_backend::env_bool("SIM_TMI_DEVICE_SMALL_RDM", false);
        return enabled;
    }

    bool entropy_subsystem_rdm_from_device_state(const void *device_state_vector,
                                                 bool device_state_is_fp64,
                                                 int n,
                                                 const std::vector<int> &kept_modes,
                                                 bool fermion_trace,
                                                 double &entropy_out,
                                                 int &status_out,
                                                 TmiWorkspace &workspace)
    {
        status_out = 0;
#ifdef MIPT_ENABLE_CUDA_RHO
        if (device_state_vector == nullptr || kept_modes.empty() ||
            kept_modes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            status_out = -1;
            return false;
        }

        const std::size_t dim = std::size_t{1} << kept_modes.size();
        if (dim == 0 || dim > (std::numeric_limits<std::size_t>::max() / dim / 2u))
        {
            status_out = -1;
            return false;
        }
        workspace.device_rdm_ri.assign(2u * dim * dim, 0.0);

        const int status = device_state_is_fp64
                               ? mipt_cuda_rdm_subsystem_f64(device_state_vector,
                                                             n,
                                                             kept_modes.data(),
                                                             static_cast<int>(kept_modes.size()),
                                                             fermion_trace ? 1 : 0,
                                                             workspace.device_rdm_ri.data())
                               : mipt_cuda_rdm_subsystem_f32(device_state_vector,
                                                             n,
                                                             kept_modes.data(),
                                                             static_cast<int>(kept_modes.size()),
                                                             fermion_trace ? 1 : 0,
                                                             workspace.device_rdm_ri.data());
        status_out = status;
        if (status != 0)
        {
            return false;
        }
        entropy_out = entropy_ri_ptr(workspace.device_rdm_ri.data(),
                                     dim,
                                     workspace.direct_matrix,
                                     workspace.entropy);
        return std::isfinite(entropy_out);
#else
        (void)device_state_vector;
        (void)device_state_is_fp64;
        (void)n;
        (void)kept_modes;
        (void)fermion_trace;
        (void)entropy_out;
        (void)workspace;
        status_out = -1;
        return false;
#endif
    }

    bool tmi_from_device_state_hybrid_gpu(const void *device_state_vector,
                                          bool device_state_is_fp64,
                                          int n,
                                          std::uint32_t block_qubits,
                                          int cycle_offset,
                                          bool fermion_trace,
                                          TmiWorkspace &workspace,
                                          double &tmi_out)
    {
        if (!device_small_rdm_enabled() || workspace.cuda_small_rdm_disabled ||
            device_state_vector == nullptr)
        {
            return false;
        }

        const int block = static_cast<int>(block_qubits);
        if (cycle_offset < 0 || cycle_offset >= block)
        {
            throw std::invalid_argument("TMI cycle offset is out of range.");
        }

        const std::vector<int> A = cyclic_mode_range(cycle_offset + 0 * block, block, n);
        const std::vector<int> B = cyclic_mode_range(cycle_offset + 1 * block, block, n);
        const std::vector<int> C = cyclic_mode_range(cycle_offset + 2 * block, block, n);
        const std::vector<int> D = cyclic_mode_range(cycle_offset + 3 * block, block, n);
        const std::vector<int> AB = concatenate_modes(A, B);
        const std::vector<int> AC = concatenate_modes(A, C);
        const std::vector<int> BC = concatenate_modes(B, C);

        auto small_entropy = [&](const std::vector<int> &modes, double &entropy) -> bool {
            int status = 0;
            if (entropy_subsystem_rdm_from_device_state(device_state_vector,
                                                        device_state_is_fp64,
                                                        n,
                                                        modes,
                                                        fermion_trace,
                                                        entropy,
                                                        status,
                                                        workspace))
            {
                return true;
            }
            workspace.cuda_small_rdm_disabled = true;
            workspace.cuda_small_rdm_failure_status = status;
            return false;
        };

        auto half_entropy = [&](const std::vector<int> &modes, double &entropy) -> bool {
            int status = 0;
            if (!workspace.cuda_svd_disabled &&
                should_try_cuda_svd_for_problem(n, modes.size()) &&
                entropy_subsystem_svd_from_device_state(device_state_vector,
                                                        device_state_is_fp64,
                                                        n,
                                                        modes,
                                                        fermion_trace,
                                                        entropy,
                                                        status))
            {
                return true;
            }
            if (!workspace.cuda_svd_disabled && status != 0)
            {
                workspace.cuda_svd_disabled = true;
                workspace.cuda_svd_failure_status = status;
            }
            return false;
        };

        double s_a = 0.0;
        double s_b = 0.0;
        double s_c = 0.0;
        double s_d = 0.0;
        double s_ab = 0.0;
        double s_ac = 0.0;
        double s_bc = 0.0;
        if (!small_entropy(A, s_a) || !small_entropy(B, s_b) ||
            !small_entropy(C, s_c) || !small_entropy(D, s_d) ||
            !half_entropy(AB, s_ab) || !half_entropy(AC, s_ac) ||
            !half_entropy(BC, s_bc))
        {
            return false;
        }

        tmi_out = s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
        return true;
    }

    bool tmi_values_from_device_state_hybrid_gpu(const void *device_state_vector,
                                                 bool device_state_is_fp64,
                                                 int n,
                                                 std::uint32_t block_qubits,
                                                 int cycle_count,
                                                 bool fermion_trace,
                                                 TmiWorkspace &workspace,
                                                 std::vector<double> &values)
    {
        values.clear();
        values.reserve(static_cast<std::size_t>(cycle_count));
        for (int cycle_offset = 0; cycle_offset < cycle_count; ++cycle_offset)
        {
            double tmi = 0.0;
            if (!tmi_from_device_state_hybrid_gpu(device_state_vector,
                                                  device_state_is_fp64,
                                                  n,
                                                  block_qubits,
                                                  cycle_offset,
                                                  fermion_trace,
                                                  workspace,
                                                  tmi))
            {
                values.clear();
                return false;
            }
            values.push_back(tmi);
        }
        return true;
    }

    template <typename Real>
    double tmi_from_statevector_hybrid_gpu_half_svd(const std::complex<Real> *host_psi,
                                                    const void *device_state_vector,
                                                    bool device_state_is_fp64,
                                                    int n,
                                                    std::uint32_t block_qubits,
                                                    int cycle_offset,
                                                    bool fermion_trace,
                                                    TmiWorkspace &workspace)
    {
        const int block = static_cast<int>(block_qubits);
        if (cycle_offset < 0 || cycle_offset >= block)
        {
            throw std::invalid_argument("TMI cycle offset is out of range.");
        }

        const std::vector<int> A = cyclic_mode_range(cycle_offset + 0 * block, block, n);
        const std::vector<int> B = cyclic_mode_range(cycle_offset + 1 * block, block, n);
        const std::vector<int> C = cyclic_mode_range(cycle_offset + 2 * block, block, n);
        const std::vector<int> D = cyclic_mode_range(cycle_offset + 3 * block, block, n);
        const std::vector<int> AB = concatenate_modes(A, B);
        const std::vector<int> AC = concatenate_modes(A, C);
        const std::vector<int> BC = concatenate_modes(B, C);

        const double s_a = entropy_subsystem_rdm_from_state(
            host_psi, n, A, fermion_trace, workspace.entropy_a);
        const double s_b = entropy_subsystem_rdm_from_state(
            host_psi, n, B, fermion_trace, workspace.entropy_b);
        const double s_c = entropy_subsystem_rdm_from_state(
            host_psi, n, C, fermion_trace, workspace.entropy_c);
        const double s_d = entropy_subsystem_rdm_from_state(
            host_psi, n, D, fermion_trace, workspace.entropy_d);

        auto half_entropy = [&](const std::vector<int> &modes) -> double {
            double entropy = 0.0;
            int status = 0;
            if (!workspace.cuda_svd_disabled &&
                should_try_cuda_svd_for_problem(n, modes.size()) &&
                entropy_subsystem_svd_from_device_state(device_state_vector,
                                                        device_state_is_fp64,
                                                        n,
                                                        modes,
                                                        fermion_trace,
                                                        entropy,
                                                        status))
            {
                return entropy;
            }

            if (!workspace.cuda_svd_disabled && status != 0)
            {
                workspace.cuda_svd_disabled = true;
                workspace.cuda_svd_failure_status = status;
            }
            return entropy_subsystem_svd_from_state(
                host_psi, n, modes, fermion_trace, workspace.entropy_half);
        };

        const double s_ab = half_entropy(AB);
        const double s_ac = half_entropy(AC);
        const double s_bc = half_entropy(BC);

        return s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
    }

    template <typename Real>
    std::vector<double> tmi_values_from_statevector_hybrid_gpu_half_svd(const std::complex<Real> *host_psi,
                                                                        const void *device_state_vector,
                                                                        bool device_state_is_fp64,
                                                                        int n,
                                                                        std::uint32_t block_qubits,
                                                                        int cycle_count,
                                                                        bool fermion_trace,
                                                                        TmiWorkspace &workspace)
    {
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(cycle_count));
        for (int cycle_offset = 0; cycle_offset < cycle_count; ++cycle_offset)
        {
            values.push_back(tmi_from_statevector_hybrid_gpu_half_svd(host_psi,
                                                                      device_state_vector,
                                                                      device_state_is_fp64,
                                                                      n,
                                                                      block_qubits,
                                                                      cycle_offset,
                                                                      fermion_trace,
                                                                      workspace));
        }
        return values;
    }

    template <typename Real>
    std::vector<double> tmi_values_from_statevector_fast(const std::complex<Real> *psi,
                                                         int n,
                                                         std::uint32_t block_qubits,
                                                         int cycle_count,
                                                         bool fermion_trace,
                                                         TmiWorkspace &workspace)
    {
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(cycle_count));
        for (int cycle_offset = 0; cycle_offset < cycle_count; ++cycle_offset)
        {
            values.push_back(tmi_from_statevector(psi,
                                                  n,
                                                  block_qubits,
                                                  cycle_offset,
                                                  fermion_trace,
                                                  workspace));
        }
        return values;
    }

    template <typename Real>
    std::vector<double> tmi_values_from_statevector_payload(const std::complex<Real> *psi,
                                                            int n,
                                                            std::uint32_t block_qubits,
                                                            int cycle_count,
                                                            bool fermion_trace,
                                                            const TmiPlans &plans,
                                                            TmiWorkspace &workspace)
    {
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(cycle_count));
        std::vector<double> payload;

        for (int cycle_offset = 0; cycle_offset < cycle_count; ++cycle_offset)
        {
            const std::vector<TmiMatrixTerm> terms = tmi_terms_1d_cycle(n, cycle_offset);
            tmi_density_matrices_from_host_statevector(psi, n, terms, payload, fermion_trace);
            values.push_back(tmi_from_payload(payload, block_qubits, terms, plans, workspace));
        }
        return values;
    }


    void dense_state_from_cudaq_amplitudes(cudaq::state &state,
                                           int n,
                                           std::vector<std::complex<double>> &out)
    {
        if (n > static_cast<int>(mipt_backend::mps_tmi_dense_max_qubits()))
        {
            throw std::runtime_error(
                "CUDA-Q tensor/MPS target did not expose a dense state tensor for TMI. "
                "The fallback can reconstruct a dense state from backend-native amplitudes only up to "
                "SIM_TMI_MPS_DENSE_MAX_QUBITS. Increase that variable for small validation jobs, "
                "or use the default nvidia statevector target for exact high-N TMI.");
        }
        const std::uint64_t dim_u64 = checked_pow2_u64(n);
        if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::invalid_argument("State-vector dimension does not fit in size_t.");
        }
        const std::size_t dim = static_cast<std::size_t>(dim_u64);
        out.resize(dim);
        double norm2 = 0.0;
        const std::uint64_t batch_size = static_cast<std::uint64_t>(mipt_backend::mps_amplitude_batch_size());
        std::vector<std::uint64_t> basis_indices;
        std::vector<std::complex<double>> amplitudes;
        if (mipt_backend::verbose_enabled())
        {
            mipt_backend::verbose_log(
                "TMI dense reconstruction: n=" + std::to_string(n) +
                " dim=" + std::to_string(dim_u64) +
                " amplitude_batch=" + std::to_string(batch_size));
        }
        for (std::uint64_t idx0 = 0; idx0 < dim_u64; idx0 += batch_size)
        {
            const std::uint64_t chunk = std::min<std::uint64_t>(batch_size, dim_u64 - idx0);
            basis_indices.clear();
            basis_indices.reserve(static_cast<std::size_t>(chunk));
            for (std::uint64_t k = 0; k < chunk; ++k)
            {
                basis_indices.push_back(idx0 + k);
            }
            cudaq_state_amplitudes_by_indices(state, n, basis_indices, amplitudes);
            for (std::uint64_t k = 0; k < chunk; ++k)
            {
                const auto amp = amplitudes[static_cast<std::size_t>(k)];
                out[static_cast<std::size_t>(idx0 + k)] = amp;
                norm2 += std::norm(amp);
            }
        }
        if (norm2 > 0.0 && std::isfinite(norm2))
        {
            const double inv_norm = 1.0 / std::sqrt(norm2);
            for (auto &amp : out)
            {
                amp *= inv_norm;
            }
        }
        mipt_backend::warn_mps_amplitude_path_once("TMI dense-state reconstruction", n, false, 0);
    }

    std::vector<double> tmi_values_from_cudaq_state_amplitude_dense(cudaq::state &state,
                                                                    int n,
                                                                    std::uint32_t block_qubits,
                                                                    int cycle_count,
                                                                    bool fermion_trace,
                                                                    TmiWorkspace &workspace)
    {
        dense_state_from_cudaq_amplitudes(state, n, workspace.host_state_f64);
        return tmi_values_from_statevector_fast(workspace.host_state_f64.data(),
                                                n,
                                                block_qubits,
                                                cycle_count,
                                                fermion_trace,
                                                workspace);
    }

    std::vector<double> tmi_values_from_cudaq_state_payload_amplitude_dense(cudaq::state &state,
                                                                            int n,
                                                                            std::uint32_t block_qubits,
                                                                            int cycle_count,
                                                                            bool fermion_trace,
                                                                            const TmiPlans &plans,
                                                                            TmiWorkspace &workspace)
    {
        dense_state_from_cudaq_amplitudes(state, n, workspace.host_state_f64);
        return tmi_values_from_statevector_payload(workspace.host_state_f64.data(),
                                                   n,
                                                   block_qubits,
                                                   cycle_count,
                                                   fermion_trace,
                                                   plans,
                                                   workspace);
    }

    std::vector<double> tmi_values_from_cudaq_state_fast(cudaq::state &state,
                                                         int n,
                                                         std::uint32_t block_qubits,
                                                         int cycle_count,
                                                         bool fermion_trace,
                                                         TmiWorkspace &workspace)
    {
        const std::uint64_t dim_u64 = checked_pow2_u64(n);
        if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::invalid_argument("State-vector dimension does not fit in size_t.");
        }
        const std::size_t dim = static_cast<std::size_t>(dim_u64);
        const auto precision = state.get_precision();
        const auto tensor = state.get_tensor();

        if (mipt_backend::verbose_enabled())
        {
            mipt_backend::verbose_log(
                "tmi_values_from_cudaq_state_fast: n=" + std::to_string(n) +
                " num_tensors=" + std::to_string(state.get_num_tensors()) +
                " tensor0_rank=" + std::to_string(tensor.get_rank()) +
                " tensor0_elements=" + std::to_string(tensor.get_num_elements()) +
                " tensor0_data=" + std::to_string(tensor.data != nullptr ? 1 : 0) +
                " state_on_gpu=" + std::to_string(state.is_on_gpu() ? 1 : 0));
        }

        const bool has_dense_rank1_tensor =
            tensor.get_rank() == 1 && tensor.get_num_elements() >= dim && tensor.data != nullptr;
        if (!has_dense_rank1_tensor)
        {
            return tmi_values_from_cudaq_state_amplitude_dense(
                state, n, block_qubits, cycle_count, fermion_trace, workspace);
        }

        if (precision == cudaq::SimulationState::precision::fp64)
        {
            if (state.is_on_gpu())
            {
                if (tensor.data != nullptr)
                {
                    std::vector<double> device_values;
                    if (tmi_values_from_device_state_hybrid_gpu(tensor.data,
                                                                true,
                                                                n,
                                                                block_qubits,
                                                                cycle_count,
                                                                fermion_trace,
                                                                workspace,
                                                                device_values))
                    {
                        return device_values;
                    }
                }
                workspace.host_state_f64.resize(dim);
                state.to_host(workspace.host_state_f64.data(), workspace.host_state_f64.size());
                if (tensor.data != nullptr)
                {
                    return tmi_values_from_statevector_hybrid_gpu_half_svd(workspace.host_state_f64.data(),
                                                                           tensor.data,
                                                                           true,
                                                                           n,
                                                                           block_qubits,
                                                                           cycle_count,
                                                                           fermion_trace,
                                                                           workspace);
                }
                return tmi_values_from_statevector_fast(workspace.host_state_f64.data(),
                                                        n,
                                                        block_qubits,
                                                        cycle_count,
                                                        fermion_trace,
                                                        workspace);
            }
            if (tensor.data == nullptr)
            {
                throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
            }
            return tmi_values_from_statevector_fast(
                reinterpret_cast<const std::complex<double> *>(tensor.data),
                n,
                block_qubits,
                cycle_count,
                fermion_trace,
                workspace);
        }

        if (state.is_on_gpu())
        {
            if (tensor.data != nullptr)
            {
                std::vector<double> device_values;
                if (tmi_values_from_device_state_hybrid_gpu(tensor.data,
                                                            false,
                                                            n,
                                                            block_qubits,
                                                            cycle_count,
                                                            fermion_trace,
                                                            workspace,
                                                            device_values))
                {
                    return device_values;
                }
            }
            workspace.host_state_f32.resize(dim);
            state.to_host(workspace.host_state_f32.data(), workspace.host_state_f32.size());
            if (tensor.data != nullptr)
            {
                return tmi_values_from_statevector_hybrid_gpu_half_svd(workspace.host_state_f32.data(),
                                                                       tensor.data,
                                                                       false,
                                                                       n,
                                                                       block_qubits,
                                                                       cycle_count,
                                                                       fermion_trace,
                                                                       workspace);
            }
            return tmi_values_from_statevector_fast(workspace.host_state_f32.data(),
                                                    n,
                                                    block_qubits,
                                                    cycle_count,
                                                    fermion_trace,
                                                    workspace);
        }
        if (tensor.data == nullptr)
        {
            throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
        }
        return tmi_values_from_statevector_fast(
            reinterpret_cast<const std::complex<float> *>(tensor.data),
            n,
            block_qubits,
            cycle_count,
            fermion_trace,
            workspace);
    }

    std::vector<double> tmi_values_from_cudaq_state_payload(cudaq::state &state,
                                                            int n,
                                                            std::uint32_t block_qubits,
                                                            int cycle_count,
                                                            bool fermion_trace,
                                                            const TmiPlans &plans,
                                                            TmiWorkspace &workspace)
    {
        const std::uint64_t dim_u64 = checked_pow2_u64(n);
        if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::invalid_argument("State-vector dimension does not fit in size_t.");
        }
        const std::size_t dim = static_cast<std::size_t>(dim_u64);
        const auto precision = state.get_precision();
        const auto tensor = state.get_tensor();

        if (mipt_backend::verbose_enabled())
        {
            mipt_backend::verbose_log(
                "tmi_values_from_cudaq_state_fast: n=" + std::to_string(n) +
                " num_tensors=" + std::to_string(state.get_num_tensors()) +
                " tensor0_rank=" + std::to_string(tensor.get_rank()) +
                " tensor0_elements=" + std::to_string(tensor.get_num_elements()) +
                " tensor0_data=" + std::to_string(tensor.data != nullptr ? 1 : 0) +
                " state_on_gpu=" + std::to_string(state.is_on_gpu() ? 1 : 0));
        }

        const bool has_dense_rank1_tensor =
            tensor.get_rank() == 1 && tensor.get_num_elements() >= dim && tensor.data != nullptr;
        if (!has_dense_rank1_tensor)
        {
            return tmi_values_from_cudaq_state_payload_amplitude_dense(
                state, n, block_qubits, cycle_count, fermion_trace, plans, workspace);
        }

        if (precision == cudaq::SimulationState::precision::fp64)
        {
            if (state.is_on_gpu())
            {
                workspace.host_state_f64.resize(dim);
                state.to_host(workspace.host_state_f64.data(), workspace.host_state_f64.size());
                return tmi_values_from_statevector_payload(workspace.host_state_f64.data(),
                                                           n,
                                                           block_qubits,
                                                           cycle_count,
                                                           fermion_trace,
                                                           plans,
                                                           workspace);
            }
            if (tensor.data == nullptr)
            {
                throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
            }
            return tmi_values_from_statevector_payload(
                reinterpret_cast<const std::complex<double> *>(tensor.data),
                n,
                block_qubits,
                cycle_count,
                fermion_trace,
                plans,
                workspace);
        }

        if (state.is_on_gpu())
        {
            workspace.host_state_f32.resize(dim);
            state.to_host(workspace.host_state_f32.data(), workspace.host_state_f32.size());
            return tmi_values_from_statevector_payload(workspace.host_state_f32.data(),
                                                       n,
                                                       block_qubits,
                                                       cycle_count,
                                                       fermion_trace,
                                                       plans,
                                                       workspace);
        }
        if (tensor.data == nullptr)
        {
            throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
        }
        return tmi_values_from_statevector_payload(
            reinterpret_cast<const std::complex<float> *>(tensor.data),
            n,
            block_qubits,
            cycle_count,
            fermion_trace,
            plans,
            workspace);
    }

    bool cudaq_state_has_device_tensor(cudaq::state &state)
    {
#ifdef MIPT_ENABLE_CUDA_RHO
        if (!state.is_on_gpu())
        {
            return false;
        }
        const auto tensor = state.get_tensor();
        return tensor.data != nullptr && tensor.get_rank() == 1;
#else
        (void)state;
        return false;
#endif
    }

    bool host_svd_available_for_state(cudaq::state &state)
    {
        const auto precision = state.get_precision();
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            return lapack_runtime::zgesvd() != nullptr;
        }
        return lapack_runtime::cgesvd() != nullptr || lapack_runtime::zgesvd() != nullptr;
    }

    bool fast_state_tmi_available(cudaq::state &state, int n, std::uint32_t block_qubits)
    {
        // The fast state-vector path always needs zheevd_ for the small A/B/C/D
        // RDM entropies.  Half-cut entropies can use either host SVD or, when the
        // state is still on GPU and CUDA_RHO is enabled, the cuSOLVER device SVD.
        // The N=8/N=12 small-N CPU override must be included here; otherwise a
        // GPU state with no host SVD could incorrectly enter the fast path and
        // fail after the CUDA half-SVD is intentionally skipped.
        if (lapack_runtime::zheevd() == nullptr)
        {
            return false;
        }
        if (host_svd_available_for_state(state))
        {
            return true;
        }
        const std::size_t half_cut_kept_modes = 2u * static_cast<std::size_t>(block_qubits);
        return cudaq_state_has_device_tensor(state) &&
               should_try_cuda_svd_for_problem(n, half_cut_kept_modes) &&
               device_small_rdm_enabled();
    }

    const char *compile_target_name()
    {
        return mipt_backend::compiled_cudaq_backend();
    }

    const char *cuda_rho_build_name()
    {
#ifdef MIPT_ENABLE_CUDA_RHO
        return "enabled";
#else
        return "disabled";
#endif
    }

    const char *state_precision_name(cudaq::SimulationState::precision precision)
    {
        return precision == cudaq::SimulationState::precision::fp64 ? "fp64" : "fp32";
    }

    std::string format_duration(double seconds)
    {
        if (!std::isfinite(seconds) || seconds < 0.0)
        {
            return "--";
        }
        const auto total = static_cast<std::uint64_t>(std::llround(seconds));
        const std::uint64_t h = total / 3600;
        const std::uint64_t m = (total % 3600) / 60;
        const std::uint64_t s = total % 60;

        std::string out;
        if (h > 0)
        {
            out += std::to_string(h);
            out += "h ";
        }
        if (h > 0 || m > 0)
        {
            out += std::to_string(m);
            out += "m ";
        }
        out += std::to_string(s);
        out += "s";
        return out;
    }

    void validate_sim_args(int n, int periods, int realizations, int res, double p_min, double p_max)
    {
        if (n < 4 || (n % 4) != 0)
        {
            throw std::invalid_argument("Online TMI requires n >= 4 and n divisible by 4.");
        }
        if (periods <= 0)
        {
            throw std::invalid_argument("periods must be positive.");
        }
        if (realizations <= 0)
        {
            throw std::invalid_argument("realizations must be positive.");
        }
        if (res <= 0)
        {
            throw std::invalid_argument("resolution must be positive.");
        }
        if (!(0.0 <= p_min && p_min <= 1.0 && 0.0 <= p_max && p_max <= 1.0))
        {
            throw std::invalid_argument("p_min and p_max must lie in [0, 1].");
        }
    }

    void ensure_output_parent_directory(const std::string &output_path);

    void run_1d_sim_tmi(int n,
                        int periods,
                        int realizations,
                        int res,
                        double p_min,
                        double p_max,
                        CircuitType circuit_mode,
                        bool all_cycles,
                        const std::string &output_path)
    {
        validate_sim_args(n, periods, realizations, res, p_min, p_max);
        if (circuit_mode != CircuitType::MMS && (n % 2) != 0)
        {
            throw std::invalid_argument(
                "circ_type 1, 2, 3, and 4 require even n in the periodic 1D brickwork geometry.");
        }

        const std::uint32_t block_qubits = static_cast<std::uint32_t>(n / 4);
        const int cycle_count = tmi_cycle_count(n, all_cycles);
        const bool fermion_trace = uses_fermionic_trace(circuit_mode);
        const TmiPlans plans = make_tmi_plans(block_qubits,
                                             fermion_trace ? TraceMode::Fermion : TraceMode::Qubit);
        const std::vector<double> ps = linspace(p_min, p_max, res);
        const std::uint64_t total = static_cast<std::uint64_t>(res) *
                                    static_cast<std::uint64_t>(realizations) *
                                    static_cast<std::uint64_t>(cycle_count);

        ensure_output_parent_directory(output_path);

        std::ofstream csv(output_path, std::ios::binary | std::ios::trunc);
        if (!csv)
        {
            throw std::runtime_error("Could not create output CSV: " + output_path);
        }
        csv << "p,tmi\n";
        csv.flush();
        if (!csv)
        {
            throw std::runtime_error("Failed while writing output CSV header.");
        }

        const std::uint64_t flush_rows = csv_flush_row_interval(n, cycle_count);
        const std::string pause_file = sim_tmi_pause_file_path();
        mipt_backend::print_backend_banner_once("sim_tmi.exe");
        const std::string pause_file_display = pause_file_display_path(pause_file);

        std::cerr << "Online TMI simulation\n"
                  << "n=" << n
                  << ", periods=" << periods
                  << ", realizations=" << realizations
                  << ", resolution=" << res
                  << ", p_min=" << p_min
                  << ", p_max=" << p_max
                  << ", block_qubits=" << block_qubits
                  << ", all_cycles=" << (all_cycles ? 1 : 0)
                  << ", tmi_cycles_per_circuit=" << cycle_count
                  << ", mode=" << circuit_type_name(circuit_mode)
                  << ", output=" << output_path << '\n'
                  << "build_target=" << compile_target_name()
                  << ", cuda_rho=" << cuda_rho_build_name()
                  << ", device_small_rdm=" << (device_small_rdm_enabled() ? 1 : 0) << '\n'
                  << "entropy_backend=" << lapack_runtime::status() << '\n'
                  << "tmi_method=selected after first state; fast path requires zheevd plus either host SVD or enabled CUDA device half-SVD" << '\n'
                  << "cuda_svd_min_kept_modes=" << cuda_svd_min_kept_modes()
                  << ", small_n_cuda_svd_policy="
                  << (force_cuda_svd_for_small_n() ? "forced-on" : "cpu-for-n8-n12")
                  << ", csv_flush_rows=" << flush_rows
                  << '\n';
        if (pause_file.empty())
        {
            std::cerr << "host_pause_control=disabled by SIM_TMI_PAUSE_FILE\n";
        }
        else
        {
            std::cerr << "host_pause_file=" << pause_file_display
                      << " ; create this file to pause at the next safe checkpoint, remove it to resume\n";
        }

        TmiWorkspace workspace;
        std::string line;
        line.reserve(128);
        std::string csv_buffer;
        csv_buffer.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(flush_rows, 4096)) * 48u);
        std::uint64_t buffered_rows = 0;
        auto flush_csv_buffer = [&]() {
            if (csv_buffer.empty())
            {
                return;
            }
            csv.write(csv_buffer.data(), static_cast<std::streamsize>(csv_buffer.size()));
            csv_buffer.clear();
            buffered_rows = 0;
            csv.flush();
            if (!csv)
            {
                throw std::runtime_error("Failed while writing output CSV.");
            }
        };

        CircuitWorkspace1D circuit_workspace;
        circuit_workspace.reserve(periods);

        const auto start = std::chrono::steady_clock::now();
        auto last_report = start;
        double paused_seconds = 0.0;
        std::uint64_t processed = 0;
        std::uint64_t last_processed = 0;
        bool printed_state_backend = false;

        auto wait_if_host_paused = [&]() {
            if (pause_file.empty() || !file_exists_for_pause(pause_file))
            {
                return;
            }

            flush_csv_buffer();
            const auto pause_start = std::chrono::steady_clock::now();
            auto last_pause_notice = pause_start;
            std::cerr << "\nPAUSED by host: found pause file \"" << pause_file_display << "\". "
                      << "CSV output is flushed. Remove the file to resume.\n"
                      << std::flush;

            while (file_exists_for_pause(pause_file))
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - last_pause_notice).count() >= 30.0)
                {
                    std::cerr << "Still paused by host for "
                              << format_duration(std::chrono::duration<double>(now - pause_start).count())
                              << "; remove \"" << pause_file_display << "\" to resume.\n"
                              << std::flush;
                    last_pause_notice = now;
                }
            }

            const auto pause_end = std::chrono::steady_clock::now();
            const auto pause_duration = pause_end - pause_start;
            const double this_pause_seconds = std::chrono::duration<double>(pause_duration).count();
            paused_seconds += this_pause_seconds;
            last_report += std::chrono::duration_cast<std::chrono::steady_clock::duration>(pause_duration);

            std::cerr << "RESUMED after host pause of " << format_duration(this_pause_seconds)
                      << "; progress rates and ETA exclude paused time.\n"
                      << std::flush;
        };

        for (std::size_t p_index = 0; p_index < ps.size(); ++p_index)
        {
            const double p = ps[p_index];
            for (int r = 0; r < realizations; ++r)
            {
                wait_if_host_paused();

                auto state = build_1d_circuit_state(
                    n, periods, p, circuit_mode, circuit_workspace, "sim_tmi");

                const bool use_fast_state_tmi = fast_state_tmi_available(state, n, block_qubits);
                if (!printed_state_backend)
                {
                    const bool device_tensor = cudaq_state_has_device_tensor(state);
                    const bool host_svd_available = host_svd_available_for_state(state);
                    const std::size_t half_cut_kept_modes = 2u * static_cast<std::size_t>(block_qubits);
                    const bool small_n_cuda_svd_disabled = cuda_svd_disabled_for_small_n(n);
                    const bool cuda_half_svd_enabled = device_tensor &&
                                                       should_try_cuda_svd_for_problem(n, half_cut_kept_modes);
                    const char *half_cut_backend = cuda_half_svd_enabled
                                                       ? "CUDA/cuSOLVER"
                                                       : (small_n_cuda_svd_disabled
                                                              ? "host LAPACK (N=8/12 CPU override)"
                                                              : (host_svd_available ? "host LAPACK" : "legacy RDM fallback"));
                    std::cerr << "state_backend=" << (state.is_on_gpu() ? "gpu" : "host")
                              << ", precision=" << state_precision_name(state.get_precision())
                              << ", device_tensor=" << (device_tensor ? 1 : 0)
                              << ", host_svd_available=" << (host_svd_available ? 1 : 0)
                              << ", small_n_cuda_svd_disabled=" << (small_n_cuda_svd_disabled ? 1 : 0)
                              << ", cuda_half_svd_enabled=" << (cuda_half_svd_enabled ? 1 : 0)
                              << ", device_small_rdm=" << (device_small_rdm_enabled() ? 1 : 0)
                              << ", half_cut_backend=" << half_cut_backend
                              << ", fast_state_tmi=" << (use_fast_state_tmi ? 1 : 0)
                              << '\n';
                    printed_state_backend = true;
                }

                std::vector<double> tmi_values;
                if (use_fast_state_tmi)
                {
                    try
                    {
                        tmi_values = tmi_values_from_cudaq_state_fast(state,
                                                                      n,
                                                                      block_qubits,
                                                                      cycle_count,
                                                                      fermion_trace,
                                                                      workspace);
                    }
                    catch (const std::bad_alloc &)
                    {
                        throw std::runtime_error(
                            "Out of host memory while building a Schmidt matrix for fast TMI. "
                            "For N=24 and especially N=28, the exact half-cut SVD still requires a large "
                            "host matrix; ensure enough RAM/swap or reduce concurrent memory pressure.");
                    }
                }
                else
                {
                    // Compatibility path for systems without a usable fast state-vector path.
                    // This is exact but much slower because it materializes pair RDMs.
                    tmi_values = tmi_values_from_cudaq_state_payload(state,
                                                                     n,
                                                                     block_qubits,
                                                                     cycle_count,
                                                                     fermion_trace,
                                                                     plans,
                                                                     workspace);
                }

                for (double tmi : tmi_values)
                {
                    line.clear();
                    append_double(line, p);
                    line.push_back(',');
                    append_double(line, tmi);
                    line.push_back('\n');

                    csv_buffer.append(line);
                    ++buffered_rows;

                    ++processed;
                    const auto now = std::chrono::steady_clock::now();
                    const double since_last = std::chrono::duration<double>(now - last_report).count();
                    if (since_last >= 1.0 || processed == total)
                    {
                        const double wall_elapsed = std::chrono::duration<double>(now - start).count();
                        const double active_elapsed = std::max(1.0e-12, wall_elapsed - paused_seconds);
                        const double avg = static_cast<double>(processed) / active_elapsed;
                        const double recent = static_cast<double>(processed - last_processed) / std::max(1.0e-12, since_last);
                        const double eta = avg > 0.0 ? static_cast<double>(total - processed) / avg : 0.0;
                        std::cerr << '\r'
                                  << "processed TMI " << processed << '/' << total
                                  << " | avg " << std::fixed << std::setprecision(2) << avg << "/s"
                                  << " | recent " << std::fixed << std::setprecision(2) << recent << "/s"
                                  << " | ETA " << format_duration(eta)
                                  << "          " << std::flush;
                        last_report = now;
                        last_processed = processed;
                    }
                }

                if (buffered_rows >= flush_rows || processed == total)
                {
                    flush_csv_buffer();
                }
            }
        }

        flush_csv_buffer();
        csv.flush();
        if (!csv)
        {
            throw std::runtime_error("Failed while finalizing output CSV.");
        }
        const auto finish = std::chrono::steady_clock::now();
        const double wall_total = std::chrono::duration<double>(finish - start).count();
        const double active_total = std::max(1.0e-12, wall_total - paused_seconds);
        std::cerr << '\n'
                  << "Completed " << processed << " TMI row(s) in "
                  << format_duration(active_total) << " active time";
        if (paused_seconds > 0.0)
        {
            std::cerr << " plus " << format_duration(paused_seconds) << " paused by host";
        }
        std::cerr << ". Overall active throughput "
                  << std::fixed << std::setprecision(2)
                  << (static_cast<double>(processed) / active_total) << "/s.\n";
        if (workspace.cuda_svd_disabled)
        {
            std::cerr << "Warning: cuSOLVER half-cut SVD was disabled after status "
                      << workspace.cuda_svd_failure_status
                      << "; using host LAPACK SVD for half cuts.\n";
        }
    }

    bool is_bool01_arg(std::string_view value)
    {
        return value == "0" || value == "1";
    }

    bool parse_bool01_arg(std::string_view value, const char *name)
    {
        if (value == "0")
        {
            return false;
        }
        if (value == "1")
        {
            return true;
        }
        throw std::invalid_argument(std::string(name) + " must be 0 or 1.");
    }

    std::string format_number_for_filename(double value)
    {
        if (std::abs(value) < 1.0e-15)
        {
            value = 0.0;
        }

        std::ostringstream out;
        out << std::fixed << std::setprecision(12) << value;
        std::string text = out.str();

        const std::size_t dot = text.find('.');
        if (dot != std::string::npos)
        {
            while (!text.empty() && text.back() == '0')
            {
                text.pop_back();
            }
            if (!text.empty() && text.back() == '.')
            {
                text.pop_back();
            }
        }

        if (text == "-0")
        {
            text = "0";
        }
        return text;
    }

    std::string tmi_output_tag(CircuitType circuit_mode, bool all_cycles)
    {
        std::string tag;
        switch (circuit_mode)
        {
        case CircuitType::MMS:
            tag = "mms";
            break;
        case CircuitType::Haar:
            tag = "haar";
            break;
        case CircuitType::FermionRPPU:
            tag = "rppu";
            break;
        case CircuitType::RFGS:
            tag = "rfgs";
            break;
        case CircuitType::QubitRPPU:
            tag = "qrppu";
            break;
        }

        if (all_cycles)
        {
            tag += "c";
        }
        return tag;
    }

    std::string default_tmi_output_path(int n,
                                        int realizations,
                                        int res,
                                        double p_min,
                                        double p_max,
                                        CircuitType circuit_mode,
                                        bool all_cycles)
    {
        const std::string tag = tmi_output_tag(circuit_mode, all_cycles);

        std::ostringstream filename;
        filename << "tmi_" << tag
                 << "_n_" << n
                 << "_real_" << realizations
                 << '_' << format_number_for_filename(p_min)
                 << '_' << format_number_for_filename(p_max)
                 << "_res_" << res
                 << ".csv";

        return std::string("csv/tmi/") + tag + "/" + filename.str();
    }

    void make_directory_if_needed(const std::string &path)
    {
        if (path.empty())
        {
            return;
        }

        if (::mkdir(path.c_str(), 0755) == 0)
        {
            return;
        }

        if (errno == EEXIST)
        {
            return;
        }

        throw std::runtime_error("Could not create output directory: " + path + " (" + std::strerror(errno) + ")");
    }

    void ensure_output_parent_directory(const std::string &output_path)
    {
        const std::size_t slash = output_path.find_last_of("/\\");
        if (slash == std::string::npos)
        {
            return;
        }

        const std::string parent = output_path.substr(0, slash);
        if (parent.empty())
        {
            return;
        }

        std::string current;
        std::size_t pos = 0;
        if (parent[0] == '/')
        {
            current = "/";
            pos = 1;
        }

        while (pos < parent.size())
        {
            const std::size_t next = parent.find_first_of("/\\", pos);
            const std::string part = parent.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
            if (!part.empty())
            {
                if (!current.empty() && current.back() != '/')
                {
                    current += '/';
                }
                current += part;
                make_directory_if_needed(current);
            }
            if (next == std::string::npos)
            {
                break;
            }
            pos = next + 1;
        }
    }

    void print_usage(const char *argv0)
    {
        std::cerr
            << "Usage:\n"
            << "  " << argv0 << " [n = 10] [periods = 10] [realizations = 10] [resolution = 5] [p_min = 0.0] [p_max = 1.0] [circ_type = 0] [all_cycles = 0] [output.csv = auto]\n\n"
            << "Arguments:\n"
            << "  circ_type = 0 uses the MMS gate set.\n"
            << "  circ_type = 1 uses random Haar U(4) brickwork gates.\n"
            << "  circ_type = 2 uses fermionic RPPU gates and fermionic partial tracing.\n"
            << "  circ_type = 3 uses the fermionic reduced gate set (RFGS).\n"
            << "  circ_type = 4 uses qRPPU: the RPPU gates as ordinary qubit gates, with no JW/CZ chains and qubit partial tracing.\n"
            << "  all_cycles = 0 computes the original offset only.\n"
            << "  all_cycles = 1 computes N/4 cyclic offsets; e.g. N=8 gives [0,1]|[2,3]|[4,5]|[6,7] and [1,2]|[3,4]|[5,6]|[7,0].\n"
            << "  For backward compatibility, output.csv may also be supplied before all_cycles.\n"
            << "  If output.csv is omitted, the default path is:\n"
            << "    csv/tmi/<tag>/tmi_<tag>_n_<n>_real_<realizations>_<p_min>_<p_max>_res_<resolution>.csv\n"
            << "  Tags are mms, haar, rppu, rfgs, and qrppu; all_cycles=1 appends c.\n\n"
            << "Output CSV columns:\n"
            << "  p,tmi\n\n"
            << "Host pause control:\n"
            << "  By default, creating PAUSE_MIPT in the process working directory pauses at the next safe checkpoint.\n"
            << "  Removing PAUSE_MIPT resumes the run. Set SIM_TMI_PAUSE_FILE=/path/to/file to use another sentinel,\n"
            << "  or SIM_TMI_PAUSE_FILE=0 to disable this mechanism. Progress rates exclude paused time.\n";
    }
} // namespace sim_tmi

int main(int argc, char **argv)
{
    try
    {
        if (argc > 1 &&
            (std::strcmp(argv[1], "--help") == 0 ||
             std::strcmp(argv[1], "-h") == 0))
        {
            sim_tmi::print_usage(argv[0]);
            return 0;
        }
        if (argc > 10)
        {
            std::cerr << "Too many arguments. Use --help for usage.\n";
            return 2;
        }

        const int n = (argc > 1) ? std::stoi(argv[1]) : 10;
        const int periods = (argc > 2) ? std::stoi(argv[2]) : 10;
        const int realizations = (argc > 3) ? std::stoi(argv[3]) : 10;
        const int res = (argc > 4) ? std::stoi(argv[4]) : 5;
        const double p_min = (argc > 5) ? std::stod(argv[5]) : 0.0;
        const double p_max = (argc > 6) ? std::stod(argv[6]) : 1.0;
        const int circ_type_value = (argc > 7) ? std::stoi(argv[7]) : 0;

        bool all_cycles = false;
        std::string output_path;
        bool output_path_explicit = false;
        if (argc > 8)
        {
            const std::string_view arg8(argv[8]);
            if (sim_tmi::is_bool01_arg(arg8))
            {
                all_cycles = sim_tmi::parse_bool01_arg(arg8, "all_cycles");
                if (argc > 9)
                {
                    output_path = argv[9];
                    output_path_explicit = true;
                }
            }
            else
            {
                output_path = argv[8];
                output_path_explicit = true;
                if (argc > 9)
                {
                    all_cycles = sim_tmi::parse_bool01_arg(argv[9], "all_cycles");
                }
            }
        }

        const CircuitType circuit_mode = parse_circuit_type(circ_type_value);
        if (!output_path_explicit)
        {
            output_path = sim_tmi::default_tmi_output_path(n,
                                                          realizations,
                                                          res,
                                                          p_min,
                                                          p_max,
                                                          circuit_mode,
                                                          all_cycles);
        }

        sim_tmi::run_1d_sim_tmi(n,
                                periods,
                                realizations,
                                res,
                                p_min,
                                p_max,
                                circuit_mode,
                                all_cycles,
                                output_path);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nerror: " << e.what() << '\n';
        return 1;
    }
}
