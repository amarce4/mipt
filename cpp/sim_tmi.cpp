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
// to CSV.  The fast path avoids materializing the half-system reduced density
// matrices: S(AB), S(AC), and S(BC) are computed from Schmidt singular values,
// while the small one-block entropies use BLAS Hermitian rank-k updates when
// BLAS is available at runtime.

#ifdef __has_include
#if __has_include(<dlfcn.h>)
#include <dlfcn.h>
#define SIM_TMI_HAS_DLFCN 1
#endif
#endif

#include <charconv>
#include <mutex>
#include <string_view>
#include <system_error>

// Reuse the existing CUDA-Q kernels, random circuit frontends, fermionic gates,
// and TMI density-matrix reducers.  Rename mipt.cpp's CLI entry point so this
// translation unit can provide its own main().
#define main mipt_original_main_for_sim_tmi
#include "mipt.cpp"
#undef main


// Fermionic reduced-gate-set architecture imported from mipt_fermion.cpp for
// sim_tmi.exe fermion=2.  This intentionally leaves mipt.cpp's existing
// fermion=1 Haar parity-preserving architecture unchanged.
struct FRGSLayerData
{
    int start = 0;

    std::vector<int> measure_flags;

    // This flag can be 1 simultaneously with the two-site gate selector.
    std::vector<int> rot_z_flags;

    // Exactly one of these should be 1 for each candidate bond.  As in
    // mipt_fermion.cpp, only the first qubit of each bond selects the gate.
    std::vector<int> rot_xx_flags;
    std::vector<int> rot_zz_flags;
};

struct MIPTKernel_1D_FRGS
{
    void operator()(int n,
                    const std::vector<FRGSLayerData> &flayers,
                    bool closed) __qpu__
    {
        cudaq::qvector q(n);

        for (std::size_t layer = 0; layer < flayers.size(); ++layer)
        {
            int start = flayers[layer].start;

            for (int i = start; i < n - 1; i += 2)
            {
                int j = (i + 1) % n;

                if (flayers[layer].rot_z_flags[i])
                {
                    rz(0.392699081698724154808, q[i]);
                }
                if (flayers[layer].rot_z_flags[j])
                {
                    rz(0.392699081698724154808, q[j]);
                }

                if (flayers[layer].rot_xx_flags[i])
                {
                    if (j < i)
                    {
                        for (int k = j + 1; k < i; ++k)
                        {
                            z<cudaq::ctrl>(q[k], q[i]);
                        }
                        rz(-1.57079632679489661923, q[i]);
                        rz(-1.57079632679489661923, q[j]);
                    }

                    cx(q[i], q[j]);
                    rx(0.78539816339744830962, q[i]);
                    cx(q[i], q[j]);

                    if (j < i)
                    {
                        s(q[i]);
                        s(q[j]);
                        for (int k = j + 1; k < i; ++k)
                        {
                            z<cudaq::ctrl>(q[k], q[i]);
                        }
                    }
                }

                if (flayers[layer].rot_zz_flags[i])
                {
                    cx(q[i], q[j]);
                    rz(0.78539816339744830962, q[j]);
                    cx(q[i], q[j]);
                }
            }

            for (int i = 0; i < n; ++i)
            {
                if (flayers[layer].measure_flags[i])
                {
                    mz(q[i]);
                }
            }
        }
    }
};

FRGSLayerData make_frgs_mipt_layer(int n,
                                   int start,
                                   double p,
                                   std::mt19937 &rng)
{
    std::bernoulli_distribution coin_flip(0.5);
    std::bernoulli_distribution measure_dist(p);

    FRGSLayerData layer;
    layer.start = start;

    layer.measure_flags.resize(n);
    layer.rot_z_flags.resize(n);
    layer.rot_xx_flags.resize(n);
    layer.rot_zz_flags.resize(n);

    for (int q = 0; q < n; ++q)
    {
        layer.rot_z_flags[q] = coin_flip(rng) ? 1 : 0;
        layer.rot_xx_flags[q] = coin_flip(rng) ? 1 : 0;
        layer.rot_zz_flags[q] = 1 - layer.rot_xx_flags[q];
        layer.measure_flags[q] = measure_dist(rng) ? 1 : 0;
    }

    return layer;
}

std::vector<FRGSLayerData> frgs_mipt_frontend(int n,
                                              int periods,
                                              double p)
{
    if ((n % 2) != 0)
    {
        throw std::invalid_argument("FRGS fermionic 1D simulation requires even n.");
    }
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("Measurement probability p must be in [0,1].");
    }

    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution extra_even_layer(0.5);

    std::vector<FRGSLayerData> layers;
    layers.reserve(2 * periods + 1);

    for (int period = 0; period < periods; ++period)
    {
        layers.push_back(make_frgs_mipt_layer(n, 0, p, rng));
        layers.push_back(make_frgs_mipt_layer(n, 1, p, rng));
    }

    if (extra_even_layer(rng))
    {
        layers.push_back(make_frgs_mipt_layer(n, 0, p, rng));
    }

    return layers;
}

namespace sim_tmi
{
    using C64 = std::complex<double>;
    constexpr double LOG2_EPS = 1.0e-15;

    enum class TraceMode
    {
        Qubit,
        Fermion
    };

    enum class CircuitMode
    {
        QubitMms = 0,
        FermionHaar = 1,
        FermionReducedGateSet = 2
    };

    CircuitMode circuit_mode_from_int(int value)
    {
        switch (value)
        {
        case 0:
            return CircuitMode::QubitMms;
        case 1:
            return CircuitMode::FermionHaar;
        case 2:
            return CircuitMode::FermionReducedGateSet;
        default:
            throw std::invalid_argument("fermion must be 0, 1, or 2.");
        }
    }

    bool uses_fermionic_trace(CircuitMode mode)
    {
        return mode != CircuitMode::QubitMms;
    }

    const char *circuit_mode_name(CircuitMode mode)
    {
        switch (mode)
        {
        case CircuitMode::QubitMms:
            return "qubit/MMS";
        case CircuitMode::FermionHaar:
            return "fermion/Haar parity-preserving + JW FSWAP";
        case CircuitMode::FermionReducedGateSet:
            return "fermion/reduced gate set from mipt_fermion.cpp";
        }
        return "unknown";
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
        using zherk_fn = void (*)(char *, char *, int *, int *, double *, C64 *, int *, double *, C64 *, int *);

        struct Handle
        {
            std::once_flag init_once;
            std::vector<void *> libraries;
            zheevd_fn zheevd = nullptr;
            zgesvd_fn zgesvd = nullptr;
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
                if (!h.zherk)
                {
                    h.zherk = reinterpret_cast<zherk_fn>(dlsym(lib, "zherk_"));
                }
            }

            h.status = "LAPACK/BLAS:";
            h.status += h.zheevd ? " zheevd" : " no-zheevd";
            h.status += h.zgesvd ? " zgesvd" : " no-zgesvd";
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

    double entropy_from_singular_values(const std::vector<double> &singular_values)
    {
        double entropy = 0.0;
        for (double sigma : singular_values)
        {
            const double lambda = sigma * sigma;
            entropy += entropy_from_eigenvalue(lambda);
        }
        return finish_entropy(entropy);
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

    struct StateEntropyWorkspace
    {
        SvdWorkspace svd;
        EntropyWorkspace entropy;
        std::vector<C64> matrix_col_major;
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

    template <typename Real>
    double entropy_subsystem_svd_from_state(const std::complex<Real> *psi,
                                            int n,
                                            const std::vector<int> &kept_modes,
                                            bool fermion_trace,
                                            StateEntropyWorkspace &ws)
    {
        if (!lapack_runtime::zgesvd())
        {
            throw std::runtime_error("Fast state-vector TMI needs LAPACK zgesvd_.");
        }
        build_coefficient_matrix_col_major(psi, n, kept_modes, fermion_trace, ws);
        const std::vector<int> env_modes = complement_modes(n, kept_modes);
        const std::size_t rows = std::size_t{1} << kept_modes.size();
        const std::size_t cols = std::size_t{1} << env_modes.size();
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
        StateEntropyWorkspace entropy_ab;
        StateEntropyWorkspace entropy_ac;
        StateEntropyWorkspace entropy_bc;
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
                                bool fermion_trace,
                                TmiWorkspace &workspace)
    {
        const int block = static_cast<int>(block_qubits);
        const std::vector<int> A = mode_range(0 * block, block);
        const std::vector<int> B = mode_range(1 * block, block);
        const std::vector<int> C = mode_range(2 * block, block);
        const std::vector<int> D = mode_range(3 * block, block);
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
            psi, n, AB, fermion_trace, workspace.entropy_ab);
        const double s_ac = entropy_subsystem_svd_from_state(
            psi, n, AC, fermion_trace, workspace.entropy_ac);
        const double s_bc = entropy_subsystem_svd_from_state(
            psi, n, BC, fermion_trace, workspace.entropy_bc);

        return s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
    }

    double tmi_from_cudaq_state_fast(cudaq::state &state,
                                     int n,
                                     std::uint32_t block_qubits,
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

        if (tensor.get_rank() != 1 || tensor.get_num_elements() < dim)
        {
            throw std::runtime_error("Expected a contiguous rank-1 CUDA-Q statevector tensor.");
        }

        if (precision == cudaq::SimulationState::precision::fp64)
        {
            if (state.is_on_gpu())
            {
                workspace.host_state_f64.resize(dim);
                state.to_host(workspace.host_state_f64.data(), workspace.host_state_f64.size());
                return tmi_from_statevector(workspace.host_state_f64.data(), n, block_qubits,
                                            fermion_trace, workspace);
            }
            if (tensor.data == nullptr)
            {
                throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
            }
            return tmi_from_statevector(
                reinterpret_cast<const std::complex<double> *>(tensor.data),
                n, block_qubits, fermion_trace, workspace);
        }

        if (state.is_on_gpu())
        {
            workspace.host_state_f32.resize(dim);
            state.to_host(workspace.host_state_f32.data(), workspace.host_state_f32.size());
            return tmi_from_statevector(workspace.host_state_f32.data(), n, block_qubits,
                                        fermion_trace, workspace);
        }
        if (tensor.data == nullptr)
        {
            throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
        }
        return tmi_from_statevector(
            reinterpret_cast<const std::complex<float> *>(tensor.data),
            n, block_qubits, fermion_trace, workspace);
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

    void run_1d_sim_tmi(int n,
                        int periods,
                        int realizations,
                        int res,
                        double p_min,
                        double p_max,
                        CircuitMode circuit_mode,
                        const std::string &output_path)
    {
        validate_sim_args(n, periods, realizations, res, p_min, p_max);

        const std::uint32_t block_qubits = static_cast<std::uint32_t>(n / 4);
        const std::vector<TmiMatrixTerm> terms = tmi_terms_1d(n);
        const bool fermion_trace = uses_fermionic_trace(circuit_mode);
        const TmiPlans plans = make_tmi_plans(block_qubits,
                                             fermion_trace ? TraceMode::Fermion : TraceMode::Qubit);
        const std::vector<double> ps = linspace(p_min, p_max, res);
        const std::uint64_t total = static_cast<std::uint64_t>(res) *
                                    static_cast<std::uint64_t>(realizations);

        std::ofstream csv(output_path, std::ios::binary | std::ios::trunc);
        if (!csv)
        {
            throw std::runtime_error("Could not create output CSV: " + output_path);
        }
        csv << "p,tmi\n";

        std::cerr << "Online TMI simulation\n"
                  << "n=" << n
                  << ", periods=" << periods
                  << ", realizations=" << realizations
                  << ", resolution=" << res
                  << ", p_min=" << p_min
                  << ", p_max=" << p_max
                  << ", block_qubits=" << block_qubits
                  << ", mode=" << circuit_mode_name(circuit_mode)
                  << ", output=" << output_path << '\n'
                  << "entropy_backend=" << lapack_runtime::status() << '\n'
                  << "tmi_method="
                  << (lapack_runtime::zgesvd()
                          ? "state-vector Schmidt/SVD for half cuts + BLAS HERK small-cut path when available"
                          : "legacy reduced-density-matrix fallback")
                  << '\n';

        TmiWorkspace workspace;
        std::string line_buffer;
        line_buffer.reserve(1u << 20);

        const auto start = std::chrono::steady_clock::now();
        auto last_report = start;
        std::uint64_t processed = 0;
        std::uint64_t last_processed = 0;

        for (std::size_t p_index = 0; p_index < ps.size(); ++p_index)
        {
            const double p = ps[p_index];
            for (int r = 0; r < realizations; ++r)
            {
                auto state = [&]() {
                    if (circuit_mode == CircuitMode::FermionReducedGateSet)
                    {
                        auto layers = frgs_mipt_frontend(n, periods, p);
                        return cudaq::get_state(MIPTKernel_1D_FRGS{}, n, layers, true);
                    }
                    if (circuit_mode == CircuitMode::FermionHaar)
                    {
                        auto layers = mipt_fermion_frontend(n, periods, p, true);
                        return cudaq::get_state(MIPTFermionKernel_1D{}, n, layers);
                    }

                    auto layers = mipt_frontend(n, periods, p);
                    return cudaq::get_state(MIPTKernel_1D{}, n, layers, true);
                }();

                double tmi = 0.0;
                if (lapack_runtime::zgesvd())
                {
                    try
                    {
                        tmi = tmi_from_cudaq_state_fast(state, n, block_qubits, fermion_trace, workspace);
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
                    // Compatibility path for systems without a usable LAPACK SVD.
                    // This is exact but much slower because it materializes pair RDMs.
                    std::vector<double> payload;
                    cudaq_tmi_density_matrices(state, n, terms, payload, fermion_trace);
                    tmi = tmi_from_payload(payload, block_qubits, terms, plans, workspace);
                }

                append_double(line_buffer, p);
                line_buffer.push_back(',');
                append_double(line_buffer, tmi);
                line_buffer.push_back('\n');

                if (line_buffer.size() >= (1u << 20))
                {
                    csv.write(line_buffer.data(), static_cast<std::streamsize>(line_buffer.size()));
                    if (!csv)
                    {
                        throw std::runtime_error("Failed while writing output CSV.");
                    }
                    line_buffer.clear();
                }

                ++processed;
                const auto now = std::chrono::steady_clock::now();
                const double since_last = std::chrono::duration<double>(now - last_report).count();
                if (since_last >= 1.0 || processed == total)
                {
                    const double elapsed = std::max(1.0e-12, std::chrono::duration<double>(now - start).count());
                    const double avg = static_cast<double>(processed) / elapsed;
                    const double recent = static_cast<double>(processed - last_processed) / std::max(1.0e-12, since_last);
                    const double eta = avg > 0.0 ? static_cast<double>(total - processed) / avg : 0.0;
                    std::cerr << '\r'
                              << "processed " << processed << '/' << total
                              << " | avg " << std::fixed << std::setprecision(2) << avg << "/s"
                              << " | recent " << std::fixed << std::setprecision(2) << recent << "/s"
                              << " | ETA " << format_duration(eta)
                              << "          " << std::flush;
                    last_report = now;
                    last_processed = processed;
                }
            }
        }

        if (!line_buffer.empty())
        {
            csv.write(line_buffer.data(), static_cast<std::streamsize>(line_buffer.size()));
            line_buffer.clear();
        }
        csv.flush();
        if (!csv)
        {
            throw std::runtime_error("Failed while finalizing output CSV.");
        }
        std::cerr << '\n';
    }

    void print_usage(const char *argv0)
    {
        std::cerr
            << "Usage:\n"
            << "  " << argv0 << " [n = 10] [periods = 10] [realizations = 10] [resolution = 5] [p_min = 0.0] [p_max = 1.0] [fermion = 0] [output.csv = tmi.csv]\n\n"
            << "Arguments:\n"
            << "  fermion = 0 uses the existing MMS/qubit circuit.\n"
            << "  fermion = 1 uses parity-preserving fermionic gates with JW FSWAP boundaries.\n"
            << "  fermion = 2 uses the reduced-gate-set fermionic architecture from mipt_fermion.cpp.\n\n"
            << "Output CSV columns:\n"
            << "  p,tmi\n";
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
        if (argc > 9)
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
        const int fermion = (argc > 7) ? std::stoi(argv[7]) : 0;
        const std::string output_path = (argc > 8) ? argv[8] : "tmi.csv";

        const sim_tmi::CircuitMode circuit_mode = sim_tmi::circuit_mode_from_int(fermion);

        sim_tmi::run_1d_sim_tmi(n,
                                periods,
                                realizations,
                                res,
                                p_min,
                                p_max,
                                circuit_mode,
                                output_path);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nerror: " << e.what() << '\n';
        return 1;
    }
}
