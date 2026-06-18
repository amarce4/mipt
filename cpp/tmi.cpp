#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef __has_include
#if __has_include(<dlfcn.h>)
#include <dlfcn.h>
#define MIPT_TMI_HAS_DLFCN 1
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    using Clock = std::chrono::steady_clock;
    using C64 = std::complex<double>;

    constexpr std::uint32_t TMI_FILE_VERSION = 1;
    constexpr std::uint32_t TMI_ENDIAN_MARKER = 0x01020304u;
    constexpr char TMI_FILE_MAGIC[8] = {'M', 'I', 'P', 'T', 'T', 'M', 'I', '1'};
    constexpr double LOG2_EPS = 1.0e-15;

    enum class TraceMode
    {
        Qubit,
        Fermion
    };

    struct TmiTermMetadata
    {
        std::string name;
        std::uint32_t kept_qubits = 0;
        std::uint32_t matrix_dimension = 0;
        std::vector<std::uint32_t> modes;
    };

    struct TmiFileMetadata
    {
        std::uint32_t spatial_dimension = 0;
        std::uint32_t total_qubits = 0;
        std::uint32_t block_qubits = 0;
        std::uint32_t periods = 0;
        std::uint32_t realizations = 0;
        std::uint32_t resolution = 0;
        std::uint32_t grid_x = 0;
        std::uint32_t grid_y = 0;
        std::uint32_t term_count = 0;
        std::uint64_t record_count = 0;
        double p_min = 0.0;
        double p_max = 0.0;
        std::vector<TmiTermMetadata> terms;
    };

    struct TmiRecord
    {
        std::uint32_t p_index = 0;
        std::uint32_t realization = 0;
        double p = 0.0;
        std::vector<std::vector<double>> term_rho_ri;
    };

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

    template <typename T>
    void read_scalar(std::istream &stream, T &value)
    {
        stream.read(reinterpret_cast<char *>(&value), sizeof(T));
        if (!stream)
        {
            throw std::runtime_error("Failed while reading TMI file.");
        }
    }

    std::size_t checked_dim_from_qubits(std::uint32_t qubits)
    {
        if (qubits >= 31)
        {
            throw std::invalid_argument("Matrix qubit count is too large for this dense TMI implementation.");
        }
        return std::size_t{1} << qubits;
    }

    std::size_t term_value_count(const TmiTermMetadata &term)
    {
        const std::size_t d = term.matrix_dimension;
        if (d != 0 && d > std::numeric_limits<std::size_t>::max() / d / 2u)
        {
            throw std::invalid_argument("TMI term payload is too large for this host.");
        }
        return 2u * d * d;
    }

    std::string clean_name(const char name[8])
    {
        std::size_t n = 0;
        while (n < 8 && name[n] != '\0')
        {
            ++n;
        }
        return std::string(name, name + n);
    }

    void validate_metadata(const TmiFileMetadata &meta)
    {
        if (meta.spatial_dimension != 1)
        {
            throw std::runtime_error("tmi.exe currently expects a 1D MIPTTMI1 file.");
        }
        if (meta.total_qubits < 4 || (meta.total_qubits % 4) != 0)
        {
            throw std::runtime_error("TMI file total_qubits must be >= 4 and divisible by 4.");
        }
        if (meta.block_qubits != meta.total_qubits / 4)
        {
            throw std::runtime_error("TMI file block_qubits is inconsistent with total_qubits/4.");
        }
        if (meta.term_count != 4 || meta.terms.size() != 4)
        {
            throw std::runtime_error("TMI file must contain exactly four stored matrices: AB, AC, BC, D.");
        }

        const std::array<std::string, 4> expected = {"AB", "AC", "BC", "D"};
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const auto &term = meta.terms[i];
            if (term.name != expected[i])
            {
                throw std::runtime_error("Unexpected TMI term order; expected AB, AC, BC, D.");
            }
            if (term.kept_qubits != term.modes.size())
            {
                throw std::runtime_error("TMI term kept_qubits does not match its mode-list length.");
            }
            if (term.matrix_dimension != checked_dim_from_qubits(term.kept_qubits))
            {
                throw std::runtime_error("TMI term matrix dimension does not match kept_qubits.");
            }
            for (std::uint32_t q : term.modes)
            {
                if (q >= meta.total_qubits)
                {
                    throw std::runtime_error("TMI term mode index is out of range.");
                }
            }
            for (std::size_t a = 0; a < term.modes.size(); ++a)
            {
                for (std::size_t b = 0; b < a; ++b)
                {
                    if (term.modes[a] == term.modes[b])
                    {
                        throw std::runtime_error("TMI term contains duplicate modes.");
                    }
                }
            }
        }

        const std::uint32_t block = meta.block_qubits;
        if (meta.terms[0].kept_qubits != 2u * block ||
            meta.terms[1].kept_qubits != 2u * block ||
            meta.terms[2].kept_qubits != 2u * block ||
            meta.terms[3].kept_qubits != block)
        {
            throw std::runtime_error("TMI file term sizes must be AB/AC/BC = L/2 and D = L/4.");
        }
    }

    class TmiMatrixReader
    {
      public:
        explicit TmiMatrixReader(const std::string &path)
            : stream_(path, std::ios::binary)
        {
            if (!stream_)
            {
                throw std::runtime_error("Could not open TMI file: " + path);
            }

            char magic[sizeof(TMI_FILE_MAGIC)]{};
            stream_.read(magic, sizeof(magic));
            if (!stream_)
            {
                throw std::runtime_error("Could not read TMI file header.");
            }
            for (std::size_t i = 0; i < sizeof(TMI_FILE_MAGIC); ++i)
            {
                if (magic[i] != TMI_FILE_MAGIC[i])
                {
                    throw std::runtime_error("Not a MIPTTMI1 TMI density-matrix file.");
                }
            }

            std::uint32_t endian_marker = 0;
            std::uint32_t version = 0;
            read_scalar(stream_, endian_marker);
            read_scalar(stream_, version);
            if (endian_marker != TMI_ENDIAN_MARKER)
            {
                throw std::runtime_error("TMI file endianness is not supported on this host.");
            }
            if (version != TMI_FILE_VERSION)
            {
                throw std::runtime_error("Unsupported TMI file version.");
            }

            read_scalar(stream_, metadata_.spatial_dimension);
            read_scalar(stream_, metadata_.total_qubits);
            read_scalar(stream_, metadata_.block_qubits);
            read_scalar(stream_, metadata_.periods);
            read_scalar(stream_, metadata_.realizations);
            read_scalar(stream_, metadata_.resolution);
            read_scalar(stream_, metadata_.grid_x);
            read_scalar(stream_, metadata_.grid_y);
            read_scalar(stream_, metadata_.term_count);
            read_scalar(stream_, metadata_.record_count);
            read_scalar(stream_, metadata_.p_min);
            read_scalar(stream_, metadata_.p_max);

            metadata_.terms.resize(metadata_.term_count);
            for (auto &term : metadata_.terms)
            {
                char name[8]{};
                stream_.read(name, sizeof(name));
                if (!stream_)
                {
                    throw std::runtime_error("TMI term metadata is incomplete.");
                }
                term.name = clean_name(name);
                read_scalar(stream_, term.kept_qubits);
                read_scalar(stream_, term.matrix_dimension);
                term.modes.resize(term.kept_qubits);
                for (std::uint32_t &mode : term.modes)
                {
                    read_scalar(stream_, mode);
                }
            }

            validate_metadata(metadata_);
            expected_values_per_record_ = 0;
            for (const auto &term : metadata_.terms)
            {
                expected_values_per_record_ += term_value_count(term);
            }
        }

        const TmiFileMetadata &metadata() const
        {
            return metadata_;
        }

        bool read_record(TmiRecord &record)
        {
            if (records_read_ >= metadata_.record_count)
            {
                return false;
            }

            read_scalar(stream_, record.p_index);
            read_scalar(stream_, record.realization);
            read_scalar(stream_, record.p);

            record.term_rho_ri.resize(metadata_.terms.size());
            std::size_t values_read = 0;
            for (std::size_t i = 0; i < metadata_.terms.size(); ++i)
            {
                const std::size_t count = term_value_count(metadata_.terms[i]);
                record.term_rho_ri[i].resize(count);
                stream_.read(reinterpret_cast<char *>(record.term_rho_ri[i].data()),
                             static_cast<std::streamsize>(count * sizeof(double)));
                if (!stream_)
                {
                    throw std::runtime_error("TMI record payload is incomplete.");
                }
                values_read += count;
            }
            if (values_read != expected_values_per_record_)
            {
                throw std::runtime_error("Internal TMI reader payload-size mismatch.");
            }

            ++records_read_;
            return true;
        }

      private:
        TmiFileMetadata metadata_;
        std::size_t expected_values_per_record_ = 0;
        std::uint64_t records_read_ = 0;
        std::ifstream stream_;
    };

    void append_uint(std::string &s, std::uint64_t value)
    {
        char buf[32];
        const auto result = std::to_chars(buf, buf + sizeof(buf), value);
        if (result.ec != std::errc{})
        {
            throw std::runtime_error("Integer formatting failed.");
        }
        s.append(buf, result.ptr);
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

        struct Handle
        {
            std::once_flag init_once;
            void *library = nullptr;
            zheevd_fn zheevd = nullptr;
            bool available = false;
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
#ifdef MIPT_TMI_HAS_DLFCN
            const char *library_names[] = {
                "liblapack.so.3",
                "liblapack.so",
                "libopenblas.so.0",
                "libopenblas.so"};

            for (const char *name : library_names)
            {
                h.library = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
                if (!h.library)
                {
                    continue;
                }
                h.zheevd = reinterpret_cast<zheevd_fn>(dlsym(h.library, "zheevd_"));
                if (h.zheevd)
                {
                    h.available = true;
                    h.status = std::string("using LAPACK zheevd_ from ") + name;
                    return;
                }
                dlclose(h.library);
                h.library = nullptr;
            }
            h.status = "LAPACK zheevd_ could not be loaded; using slower Jacobi fallback";
#else
            h.status = "<dlfcn.h> is unavailable; using slower Jacobi fallback";
#endif
        }

        zheevd_fn zheevd()
        {
            auto &h = handle();
            std::call_once(h.init_once, initialize);
            return h.zheevd;
        }

        const std::string &status()
        {
            auto &h = handle();
            std::call_once(h.init_once, initialize);
            return h.status;
        }

        bool available()
        {
            return zheevd() != nullptr;
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

    void fill_complex_from_ri(const std::vector<double> &rho_ri, std::size_t dim, std::vector<C64> &out)
    {
        if (rho_ri.size() != 2u * dim * dim)
        {
            throw std::invalid_argument("Density-matrix payload has the wrong size.");
        }
        out.resize(dim * dim);
        for (std::size_t i = 0; i < dim * dim; ++i)
        {
            out[i] = C64(rho_ri[2u * i + 0], rho_ri[2u * i + 1]);
        }
    }

    double entropy_ri(const std::vector<double> &rho_ri,
                      std::size_t dim,
                      std::vector<C64> &matrix_buffer,
                      EntropyWorkspace &entropy_ws)
    {
        fill_complex_from_ri(rho_ri, dim, matrix_buffer);
        return entropy_hermitian_inplace(matrix_buffer, dim, entropy_ws);
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

    void partial_trace_ri_to(const std::vector<double> &rho_ri,
                             const TracePlan &plan,
                             std::vector<C64> &out)
    {
        if (rho_ri.size() != 2u * plan.parent_dim * plan.parent_dim)
        {
            throw std::invalid_argument("Parent density matrix has wrong size for TMI partial trace.");
        }
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

    struct TmiComputationWorkspace
    {
        EntropyWorkspace entropy;
        std::vector<C64> rho_a;
        std::vector<C64> rho_b;
        std::vector<C64> rho_c;
        std::vector<C64> direct_matrix;
    };

    double tmi_from_record(const TmiRecord &record,
                           const TmiFileMetadata &meta,
                           const TmiPlans &plans,
                           TmiComputationWorkspace &workspace)
    {
        const std::size_t block_dim = std::size_t{1} << meta.block_qubits;
        const std::size_t pair_dim = std::size_t{1} << (2u * meta.block_qubits);

        partial_trace_ri_to(record.term_rho_ri[0], plans.ab_to_a, workspace.rho_a);
        partial_trace_ri_to(record.term_rho_ri[0], plans.ab_to_b, workspace.rho_b);
        partial_trace_ri_to(record.term_rho_ri[1], plans.ac_to_c, workspace.rho_c);

        const double s_a = entropy_hermitian_inplace(workspace.rho_a, block_dim, workspace.entropy);
        const double s_b = entropy_hermitian_inplace(workspace.rho_b, block_dim, workspace.entropy);
        const double s_c = entropy_hermitian_inplace(workspace.rho_c, block_dim, workspace.entropy);
        const double s_d = entropy_ri(record.term_rho_ri[3], block_dim, workspace.direct_matrix, workspace.entropy);
        const double s_ab = entropy_ri(record.term_rho_ri[0], pair_dim, workspace.direct_matrix, workspace.entropy);
        const double s_ac = entropy_ri(record.term_rho_ri[1], pair_dim, workspace.direct_matrix, workspace.entropy);
        const double s_bc = entropy_ri(record.term_rho_ri[2], pair_dim, workspace.direct_matrix, workspace.entropy);

        return s_a + s_b + s_c + s_d - s_ab - s_ac - s_bc;
    }

    std::uint64_t parse_u64(std::string_view text, const char *name)
    {
        std::uint64_t value = 0;
        const char *begin = text.data();
        const char *end = text.data() + text.size();
        const auto result = std::from_chars(begin, end, value);
        if (result.ec != std::errc{} || result.ptr != end)
        {
            throw std::invalid_argument(std::string("Invalid ") + name + ": " + std::string(text));
        }
        return value;
    }

    TraceMode parse_trace_mode(std::string_view text)
    {
        if (text == "0")
        {
            return TraceMode::Qubit;
        }
        if (text == "1")
        {
            return TraceMode::Fermion;
        }
        throw std::invalid_argument("trace_mode must be 0 for qubit or 1 for fermion.");
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

    std::size_t parse_chunk_size_env(std::size_t default_value)
    {
        const char *env = std::getenv("TMI_CHUNK_RECORDS");
        if (!env || *env == '\0')
        {
            return default_value;
        }
        std::uint64_t parsed = parse_u64(env, "TMI_CHUNK_RECORDS");
        if (parsed == 0)
        {
            throw std::invalid_argument("TMI_CHUNK_RECORDS must be positive when set.");
        }
        return static_cast<std::size_t>(parsed);
    }

    std::size_t default_chunk_size_for_file(const TmiFileMetadata &meta)
    {
        std::uint64_t bytes_per_record = sizeof(std::uint32_t) * 2u + sizeof(double);
        for (const auto &term : meta.terms)
        {
            bytes_per_record += static_cast<std::uint64_t>(term_value_count(term)) * sizeof(double);
        }
        constexpr std::uint64_t target_chunk_bytes = 64ull * 1024ull * 1024ull;
        std::size_t records = static_cast<std::size_t>(std::max<std::uint64_t>(1, target_chunk_bytes / std::max<std::uint64_t>(1, bytes_per_record)));
        records = std::min<std::size_t>(records, 256);
        return records;
    }

    struct PendingRecord
    {
        std::uint64_t record_linear = 0;
        TmiRecord record;
    };

    struct TmiCsvRow
    {
        std::uint64_t record_linear = 0;
        std::uint32_t p_index = 0;
        std::uint32_t realization = 0;
        double p = 0.0;
        double tmi = 0.0;
    };

    class ProgressReporter
    {
      public:
        explicit ProgressReporter(std::uint64_t total_records)
            : total_(total_records), start_(Clock::now()), last_(start_)
        {
        }

        void set_processed(std::uint64_t processed)
        {
            processed_ = processed;
        }

        void maybe_report(bool force = false)
        {
            const auto now = Clock::now();
            const double since_last = std::chrono::duration<double>(now - last_).count();
            if (!force && since_last < 1.0)
            {
                return;
            }

            const double elapsed = std::max(1.0e-12, std::chrono::duration<double>(now - start_).count());
            const double avg = static_cast<double>(processed_) / elapsed;
            const double recent = since_last > 0.0
                                      ? static_cast<double>(processed_ - last_processed_) / since_last
                                      : 0.0;
            const double eta = (avg > 0.0 && processed_ < total_)
                                   ? static_cast<double>(total_ - processed_) / avg
                                   : 0.0;

            std::cerr << '\r'
                      << "processed " << processed_ << '/' << total_ << " records"
                      << " | avg " << std::fixed << std::setprecision(1) << avg << "/s"
                      << " | recent " << std::fixed << std::setprecision(1) << recent << "/s"
                      << " | ETA " << format_duration(eta) << "         "
                      << std::flush;

            last_ = now;
            last_processed_ = processed_;
            if (force)
            {
                std::cerr << '\n';
            }
        }

      private:
        std::uint64_t total_ = 0;
        std::uint64_t processed_ = 0;
        std::uint64_t last_processed_ = 0;
        Clock::time_point start_;
        Clock::time_point last_;
    };

    void print_usage(const char *argv0)
    {
        std::cerr
            << "Usage:\n"
            << "  " << argv0 << " <trace_mode: 0|1> <rho_tmi.bin> <output.csv> [limit_per_p]\n\n"
            << "Arguments:\n"
            << "  trace_mode = 0 uses ordinary qubit partial traces inside AB/AC.\n"
            << "  trace_mode = 1 uses fermionic signed partial traces inside AB/AC.\n\n"
            << "Input:\n"
            << "  MIPTTMI1 file produced by mipt.exe ... 1. The stored terms are AB, AC, BC, D.\n\n"
            << "Output columns:\n"
            << "  record,p_index,p,realization,tmi\n";
    }
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc < 4 || argc > 5)
        {
            print_usage(argv[0]);
            return 2;
        }

        const TraceMode trace_mode = parse_trace_mode(argv[1]);
        const std::string input_path = argv[2];
        const std::string output_path = argv[3];
        std::uint64_t limit_per_p = std::numeric_limits<std::uint64_t>::max();
        if (argc == 5)
        {
            const std::uint64_t parsed = parse_u64(argv[4], "limit_per_p");
            if (parsed != 0)
            {
                limit_per_p = parsed;
            }
        }

        TmiMatrixReader reader(input_path);
        const auto &meta = reader.metadata();
        const TmiPlans plans = make_tmi_plans(meta.block_qubits, trace_mode);

        std::uint64_t planned_records = meta.record_count;
        if (limit_per_p != std::numeric_limits<std::uint64_t>::max())
        {
            if (meta.resolution > 0 && meta.realizations > 0)
            {
                planned_records = static_cast<std::uint64_t>(meta.resolution) *
                                  std::min<std::uint64_t>(limit_per_p, meta.realizations);
                planned_records = std::min<std::uint64_t>(planned_records, meta.record_count);
            }
        }

        std::ofstream output_file;
        std::ostream *out = nullptr;
        if (output_path == "-")
        {
            out = &std::cout;
        }
        else
        {
            output_file.open(output_path, std::ios::binary | std::ios::trunc);
            if (!output_file)
            {
                throw std::runtime_error("Could not create output CSV: " + output_path);
            }
            out = &output_file;
        }

        *out << "record,p_index,p,realization,tmi\n";

        std::cerr << "Input: " << input_path << '\n'
                  << "records=" << meta.record_count
                  << ", p_values=" << meta.resolution
                  << ", realizations=" << meta.realizations
                  << ", block_qubits=" << meta.block_qubits
                  << ", trace_mode=" << (trace_mode == TraceMode::Fermion ? "fermion" : "qubit")
                  << ", planned_records=" << planned_records << '\n';

        const std::size_t chunk_size = parse_chunk_size_env(default_chunk_size_for_file(meta));
        std::cerr << "entropy_backend=" << lapack_runtime::status() << '\n'
                  << "chunk_records=" << chunk_size << '\n';
#ifdef _OPENMP
        std::cerr << "OpenMP threads=" << omp_get_max_threads() << '\n';
#endif

        std::vector<std::uint64_t> records_done_per_p(std::max<std::uint32_t>(meta.resolution, 1), 0);
        ProgressReporter progress(planned_records);
        std::uint64_t records_read = 0;
        std::uint64_t records_processed = 0;

        std::vector<PendingRecord> chunk;
        std::vector<TmiCsvRow> rows;
        chunk.reserve(chunk_size);
        rows.reserve(chunk_size);

        auto process_chunk = [&]() {
            if (chunk.empty())
            {
                return;
            }

            rows.resize(chunk.size());

#pragma omp parallel
            {
                TmiComputationWorkspace workspace;
#pragma omp for schedule(dynamic, 1)
                for (std::int64_t i = 0; i < static_cast<std::int64_t>(chunk.size()); ++i)
                {
                    const auto &pending = chunk[static_cast<std::size_t>(i)];
                    const auto &rec = pending.record;
                    TmiCsvRow row;
                    row.record_linear = pending.record_linear;
                    row.p_index = rec.p_index;
                    row.realization = rec.realization;
                    row.p = rec.p;
                    row.tmi = tmi_from_record(rec, meta, plans, workspace);
                    rows[static_cast<std::size_t>(i)] = row;
                }
            }

            std::string lines;
            lines.reserve(rows.size() * 128u);
            for (const auto &row : rows)
            {
                append_uint(lines, row.record_linear);
                lines.push_back(',');
                append_uint(lines, row.p_index);
                lines.push_back(',');
                append_double(lines, row.p);
                lines.push_back(',');
                append_uint(lines, row.realization);
                lines.push_back(',');
                append_double(lines, row.tmi);
                lines.push_back('\n');
            }
            out->write(lines.data(), static_cast<std::streamsize>(lines.size()));
            if (!*out)
            {
                throw std::runtime_error("Failed while writing output CSV.");
            }

            records_processed += static_cast<std::uint64_t>(chunk.size());
            progress.set_processed(records_processed);
            progress.maybe_report(false);

            chunk.clear();
            rows.clear();
        };

        TmiRecord record;
        while (reader.read_record(record))
        {
            const std::uint64_t record_linear = records_read++;
            if (record.p_index >= records_done_per_p.size())
            {
                records_done_per_p.resize(static_cast<std::size_t>(record.p_index) + 1, 0);
            }
            if (limit_per_p != std::numeric_limits<std::uint64_t>::max() &&
                records_done_per_p[record.p_index] >= limit_per_p)
            {
                record = TmiRecord{};
                continue;
            }
            ++records_done_per_p[record.p_index];

            PendingRecord pending;
            pending.record_linear = record_linear;
            pending.record = std::move(record);
            chunk.push_back(std::move(pending));
            record = TmiRecord{};

            if (chunk.size() >= chunk_size)
            {
                process_chunk();
            }
        }
        process_chunk();

        if (output_file)
        {
            output_file.flush();
        }
        progress.set_processed(records_processed);
        progress.maybe_report(true);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nerror: " << e.what() << '\n';
        return 1;
    }
}
