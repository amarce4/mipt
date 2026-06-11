#include "density_io.hpp"
#include "gmn.hpp"

#if __has_include(<scs.h>)
#include <scs.h>
#elif __has_include(<scs/scs.h>)
#include <scs/scs.h>
#else
#error "Could not find SCS public header. Try: make ring_inflation.exe SCS_CFLAGS="-I$HOME/opt/scs/include -I$HOME/opt/scs/include/scs""
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    constexpr int QUBITS = 6;
    constexpr int FULL_DIM = 64;
    constexpr int RHO_DIM = 8;
    constexpr int RHO_RI_VALUES = 2 * RHO_DIM * RHO_DIM;
    constexpr double INV_SQRT2 = 0.70710678118654752440084436210484903928483593768847;
    constexpr double SQRT2 = 1.4142135623730950488016887242096980785696718753769;
    constexpr double COEFF_DROP_TOL = 1.0e-14;

    using Complex = std::complex<double>;

    struct Options
    {
        std::string input_file;
        int max_iters = 2000;
        double tol = 1.0e-3;
        double time_limit_secs = 0.0;
        std::string output_csv = "ring_inflation.csv";
        bool scs_verbose = false;
    };

    struct CscMatrixStorage
    {
        std::vector<scs_float> x;
        std::vector<scs_int> i;
        std::vector<scs_int> p;
        ScsMatrix matrix{};

        void bind(scs_int rows, scs_int cols)
        {
            matrix.x = x.empty() ? nullptr : x.data();
            matrix.i = i.empty() ? nullptr : i.data();
            matrix.p = p.data();
            matrix.m = rows;
            matrix.n = cols;
        }
    };

    struct ConeProgram
    {
        scs_int rows = 0;
        scs_int cols = 0;
        std::vector<scs_float> b;
        std::vector<scs_float> c;
        std::vector<scs_int> complex_psd_dims;
        CscMatrixStorage A_storage;
        ScsData data{};
        ScsCone cone{};
    };

    struct SCSResult
    {
        double score = std::numeric_limits<double>::quiet_NaN();
        std::string status;
        scs_int status_val = 0;
        scs_int iter = 0;
        double solve_time_ms = 0.0;
    };

    struct Candidate
    {
        std::uint32_t subsystem = 0;
        std::array<double, RHO_RI_VALUES> rho_ri{};
        double gmn = -std::numeric_limits<double>::infinity();
    };

    void bind_cone_program(ConeProgram &program)
    {
        program.A_storage.bind(program.rows, program.cols);
        program.data.m = program.rows;
        program.data.n = program.cols;
        program.data.A = &program.A_storage.matrix;
        program.data.P = nullptr;
        program.data.b = program.b.data();
        program.data.c = program.c.data();

        program.cone.cs = program.complex_psd_dims.data();
        program.cone.cssize = static_cast<scs_int>(program.complex_psd_dims.size());
    }

    int parse_positive_int(const char *s, const char *name)
    {
        char *end = nullptr;
        errno = 0;
        const long value = std::strtol(s, &end, 10);
        if (errno != 0 || end == s || *end != '\0' || value <= 0 ||
            value > std::numeric_limits<int>::max())
        {
            throw std::invalid_argument(std::string(name) + " must be a positive integer.");
        }
        return static_cast<int>(value);
    }

    double parse_positive_double(const char *s, const char *name, bool allow_zero)
    {
        char *end = nullptr;
        errno = 0;
        const double value = std::strtod(s, &end);
        if (errno != 0 || end == s || *end != '\0' || !std::isfinite(value) ||
            (allow_zero ? value < 0.0 : value <= 0.0))
        {
            throw std::invalid_argument(std::string(name) + " must be a " +
                                        (allow_zero ? "nonnegative" : "positive") +
                                        " finite number.");
        }
        return value;
    }

    bool env_flag(const char *name, bool default_value = false)
    {
        const char *value = std::getenv(name);
        if (value == nullptr || *value == '\0')
        {
            return default_value;
        }
        const std::string s(value);
        return !(s == "0" || s == "false" || s == "False" || s == "FALSE" || s == "no");
    }

    void set_env_if_missing(const char *name, const char *value)
    {
        if (std::getenv(name) != nullptr)
        {
            return;
        }
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 0);
#endif
    }

    void print_usage(const char *program)
    {
        std::cout
            << "Usage: " << program << " [file] [max_iters] [tol] [time_limit_secs]\n\n"
            << "Pipeline:\n"
            << "  1. Reads a version-2 MIPT 3-qubit RDM .bin file.\n"
            << "  2. Runs full complex MOSEK GMN on every stored distance-1 subsystem.\n"
            << "  3. Selects the subsystem with highest GMN for each trajectory.\n"
            << "  4. Runs the level-2 ring-inflation SDP with SCS on that subsystem.\n"
            << "  5. Writes ring_inflation.csv with columns: p,gmn,inflation_score.\n\n"
            << "Example:\n"
            << "  " << program << " rho3.bin 2000 1e-3 300\n\n"
            << "Environment variables:\n"
            << "  RING_OUTPUT_CSV=path       Output CSV path; default ring_inflation.csv.\n"
            << "  RING_SCS_VERBOSE=1         Enable SCS iteration logging.\n"
            << "  OMP_NUM_THREADS=N          Parallel GMN workers.\n"
            << "  GMN_MOSEK_NUM_THREADS=N    MOSEK threads per GMN solve; default forced to 1 when OpenMP > 1.\n";
    }

    Options parse_options(int argc, char **argv)
    {
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (argc != 5)
        {
            throw std::invalid_argument(
                "Expected exactly four arguments: [file] [max_iters] [tol] [time_limit_secs].");
        }

        Options opts;
        opts.input_file = argv[1];
        opts.max_iters = parse_positive_int(argv[2], "max_iters");
        opts.tol = parse_positive_double(argv[3], "tol", false);
        opts.time_limit_secs = parse_positive_double(argv[4], "time_limit_secs", true);
        opts.scs_verbose = env_flag("RING_SCS_VERBOSE", false);

        if (const char *out = std::getenv("RING_OUTPUT_CSV"); out != nullptr && *out != '\0')
        {
            opts.output_csv = out;
        }
        return opts;
    }

    int cvec_len(int dim)
    {
        return dim * dim;
    }

    int cvec_index(int dim, int row, int col, bool imag_part)
    {
        if (row < col)
        {
            throw std::logic_error("cvec_index requires row >= col.");
        }
        int index = 0;
        for (int c = 0; c < dim; ++c)
        {
            for (int r = c; r < dim; ++r)
            {
                if (r == c)
                {
                    if (r == row && c == col)
                    {
                        if (imag_part)
                        {
                            throw std::logic_error("Diagonal cvec entries have no imaginary coordinate.");
                        }
                        return index;
                    }
                    ++index;
                }
                else
                {
                    if (r == row && c == col)
                    {
                        return index + (imag_part ? 1 : 0);
                    }
                    index += 2;
                }
            }
        }
        throw std::logic_error("cvec index out of range.");
    }

    std::pair<int, int> offdiag_cvec_indices(int dim, int row, int col)
    {
        if (row <= col)
        {
            throw std::logic_error("offdiag_cvec_indices requires row > col.");
        }
        return {cvec_index(dim, row, col, false), cvec_index(dim, row, col, true)};
    }

    std::array<int, QUBITS> unravel6(int flat)
    {
        std::array<int, QUBITS> out{};
        for (int q = QUBITS - 1; q >= 0; --q)
        {
            out[static_cast<std::size_t>(q)] = flat & 1;
            flat >>= 1;
        }
        return out;
    }

    int ravel_bits(const std::array<int, QUBITS> &bits)
    {
        int out = 0;
        for (int q = 0; q < QUBITS; ++q)
        {
            out = (out << 1) | bits[static_cast<std::size_t>(q)];
        }
        return out;
    }

    std::vector<int> output_dims_for_keep(const std::vector<int> &keep)
    {
        return std::vector<int>(keep.size(), 2);
    }

    std::vector<int> unravel_local(int flat, int nbits)
    {
        std::vector<int> out(static_cast<std::size_t>(nbits), 0);
        for (int q = nbits - 1; q >= 0; --q)
        {
            out[static_cast<std::size_t>(q)] = flat & 1;
            flat >>= 1;
        }
        return out;
    }

    void trace_value_loop(int trace_count,
                          const std::function<void(const std::vector<int> &)> &visitor)
    {
        const int total = 1 << trace_count;
        for (int flat = 0; flat < total; ++flat)
        {
            visitor(unravel_local(flat, trace_count));
        }
    }

    class RowBuilder
    {
      public:
        int add_row(double rhs = 0.0)
        {
            rows_.emplace_back();
            b_.push_back(rhs);
            return static_cast<int>(rows_.size()) - 1;
        }

        void add_coeff(int row, int col, double value)
        {
            if (std::abs(value) <= COEFF_DROP_TOL)
            {
                return;
            }
            rows_.at(static_cast<std::size_t>(row))[col] += value;
        }

        bool row_is_empty(int row) const
        {
            return rows_.at(static_cast<std::size_t>(row)).empty();
        }

        void drop_last_empty_zero_row()
        {
            if (!rows_.empty() && rows_.back().empty() && std::abs(b_.back()) <= COEFF_DROP_TOL)
            {
                rows_.pop_back();
                b_.pop_back();
            }
        }

        int row_count() const
        {
            return static_cast<int>(rows_.size());
        }

        const std::vector<double> &rhs() const
        {
            return b_;
        }

        CscMatrixStorage to_csc(int cols) const
        {
            CscMatrixStorage storage;
            storage.p.assign(static_cast<std::size_t>(cols) + 1u, 0);

            std::vector<std::vector<std::pair<int, double>>> by_col(static_cast<std::size_t>(cols));
            for (int r = 0; r < static_cast<int>(rows_.size()); ++r)
            {
                for (const auto &kv : rows_[static_cast<std::size_t>(r)])
                {
                    if (kv.first < 0 || kv.first >= cols)
                    {
                        throw std::logic_error("Column index out of range while building CSC matrix.");
                    }
                    if (std::abs(kv.second) > COEFF_DROP_TOL)
                    {
                        by_col[static_cast<std::size_t>(kv.first)].push_back({r, kv.second});
                    }
                }
            }

            for (int c = 0; c < cols; ++c)
            {
                auto &col = by_col[static_cast<std::size_t>(c)];
                std::sort(col.begin(), col.end(), [](const auto &a, const auto &b) {
                    return a.first < b.first;
                });
                storage.p[static_cast<std::size_t>(c)] = static_cast<scs_int>(storage.x.size());
                for (const auto &entry : col)
                {
                    storage.i.push_back(static_cast<scs_int>(entry.first));
                    storage.x.push_back(static_cast<scs_float>(entry.second));
                }
            }
            storage.p[static_cast<std::size_t>(cols)] = static_cast<scs_int>(storage.x.size());
            storage.bind(static_cast<scs_int>(rows_.size()), static_cast<scs_int>(cols));
            return storage;
        }

      private:
        std::vector<std::unordered_map<int, double>> rows_;
        std::vector<double> b_;
    };

    struct HermitianVariable
    {
        int dim = 0;
        int offset = 0;

        int coord_diag(int i) const
        {
            return offset + cvec_index(dim, i, i, false);
        }

        std::pair<int, int> coord_offdiag(int row, int col) const
        {
            auto p = offdiag_cvec_indices(dim, row, col);
            return {offset + p.first, offset + p.second};
        }
    };

    void add_complex_entry_to_scalar_row(RowBuilder &builder,
                                         int row,
                                         const HermitianVariable &var,
                                         int source_i,
                                         int source_j,
                                         Complex multiplier,
                                         double scalar_real_factor,
                                         double scalar_imag_factor)
    {
        if (source_i == source_j)
        {
            const int col = var.coord_diag(source_i);
            builder.add_coeff(row, col,
                              scalar_real_factor * multiplier.real() +
                                  scalar_imag_factor * multiplier.imag());
            return;
        }

        int lower_r = source_i;
        int lower_c = source_j;
        double imag_sign = 1.0;
        if (source_i < source_j)
        {
            lower_r = source_j;
            lower_c = source_i;
            imag_sign = -1.0;
        }

        const auto coords = var.coord_offdiag(lower_r, lower_c);
        const Complex re_factor = multiplier * INV_SQRT2;
        const Complex im_factor = multiplier * Complex(0.0, imag_sign * INV_SQRT2);
        builder.add_coeff(row, coords.first,
                          scalar_real_factor * re_factor.real() +
                              scalar_imag_factor * re_factor.imag());
        builder.add_coeff(row, coords.second,
                          scalar_real_factor * im_factor.real() +
                              scalar_imag_factor * im_factor.imag());
    }

    void add_entry_to_output_cvec(RowBuilder &builder,
                                  int row_base,
                                  int output_dim,
                                  int output_i,
                                  int output_j,
                                  const HermitianVariable &var,
                                  int source_i,
                                  int source_j,
                                  double multiplier)
    {
        if (output_i < output_j)
        {
            throw std::logic_error("Output cvec mapping requires output_i >= output_j.");
        }

        if (output_i == output_j)
        {
            const int out_row = row_base + cvec_index(output_dim, output_i, output_j, false);
            add_complex_entry_to_scalar_row(builder, out_row, var, source_i, source_j,
                                            Complex(multiplier, 0.0), 1.0, 0.0);
        }
        else
        {
            const int out_re = row_base + cvec_index(output_dim, output_i, output_j, false);
            const int out_im = row_base + cvec_index(output_dim, output_i, output_j, true);
            add_complex_entry_to_scalar_row(builder, out_re, var, source_i, source_j,
                                            Complex(multiplier, 0.0), SQRT2, 0.0);
            add_complex_entry_to_scalar_row(builder, out_im, var, source_i, source_j,
                                            Complex(multiplier, 0.0), 0.0, SQRT2);
        }
    }

    std::vector<int> complement_keep(const std::vector<int> &keep)
    {
        std::vector<int> trace;
        for (int q = 0; q < QUBITS; ++q)
        {
            if (std::find(keep.begin(), keep.end(), q) == keep.end())
            {
                trace.push_back(q);
            }
        }
        return trace;
    }

    void add_ordered_marginal_to_rows(RowBuilder &builder,
                                      int row_base,
                                      const HermitianVariable &var,
                                      const std::vector<int> &keep,
                                      double multiplier)
    {
        const int out_dim = 1 << static_cast<int>(keep.size());
        const std::vector<int> trace = complement_keep(keep);

        for (int a = 0; a < out_dim; ++a)
        {
            const auto a_multi = unravel_local(a, static_cast<int>(keep.size()));
            for (int b = 0; b <= a; ++b)
            {
                const auto b_multi = unravel_local(b, static_cast<int>(keep.size()));
                trace_value_loop(static_cast<int>(trace.size()), [&](const std::vector<int> &traced) {
                    std::array<int, QUBITS> ket{};
                    std::array<int, QUBITS> bra{};
                    for (std::size_t pos = 0; pos < keep.size(); ++pos)
                    {
                        ket[static_cast<std::size_t>(keep[pos])] = a_multi[pos];
                        bra[static_cast<std::size_t>(keep[pos])] = b_multi[pos];
                    }
                    for (std::size_t pos = 0; pos < trace.size(); ++pos)
                    {
                        ket[static_cast<std::size_t>(trace[pos])] = traced[pos];
                        bra[static_cast<std::size_t>(trace[pos])] = traced[pos];
                    }
                    add_entry_to_output_cvec(builder, row_base, out_dim, a, b, var,
                                             ravel_bits(ket), ravel_bits(bra), multiplier);
                });
            }
        }
    }

    void add_partial_transpose_to_rows(RowBuilder &builder,
                                       int row_base,
                                       const HermitianVariable &var,
                                       const std::vector<int> &pt_subsystems,
                                       double multiplier)
    {
        for (int a = 0; a < FULL_DIM; ++a)
        {
            const auto out_ket = unravel6(a);
            for (int b = 0; b <= a; ++b)
            {
                const auto out_bra = unravel6(b);
                auto src_ket = out_ket;
                auto src_bra = out_bra;
                for (int q : pt_subsystems)
                {
                    std::swap(src_ket[static_cast<std::size_t>(q)],
                              src_bra[static_cast<std::size_t>(q)]);
                }
                add_entry_to_output_cvec(builder, row_base, FULL_DIM, a, b, var,
                                         ravel_bits(src_ket), ravel_bits(src_bra), multiplier);
            }
        }
    }

    void add_partial_transposed_marginal_to_rows(RowBuilder &builder,
                                                 int row_base,
                                                 const HermitianVariable &var,
                                                 const std::vector<int> &keep,
                                                 const std::vector<int> &pt_positions,
                                                 double multiplier)
    {
        const int out_dim = 1 << static_cast<int>(keep.size());
        const std::vector<int> trace = complement_keep(keep);

        for (int a = 0; a < out_dim; ++a)
        {
            const auto out_ket = unravel_local(a, static_cast<int>(keep.size()));
            for (int b = 0; b <= a; ++b)
            {
                const auto out_bra = unravel_local(b, static_cast<int>(keep.size()));
                auto src_local_ket = out_ket;
                auto src_local_bra = out_bra;
                for (int pos : pt_positions)
                {
                    std::swap(src_local_ket[static_cast<std::size_t>(pos)],
                              src_local_bra[static_cast<std::size_t>(pos)]);
                }

                trace_value_loop(static_cast<int>(trace.size()), [&](const std::vector<int> &traced) {
                    std::array<int, QUBITS> ket{};
                    std::array<int, QUBITS> bra{};
                    for (std::size_t pos = 0; pos < keep.size(); ++pos)
                    {
                        ket[static_cast<std::size_t>(keep[pos])] = src_local_ket[pos];
                        bra[static_cast<std::size_t>(keep[pos])] = src_local_bra[pos];
                    }
                    for (std::size_t pos = 0; pos < trace.size(); ++pos)
                    {
                        ket[static_cast<std::size_t>(trace[pos])] = traced[pos];
                        bra[static_cast<std::size_t>(trace[pos])] = traced[pos];
                    }
                    add_entry_to_output_cvec(builder, row_base, out_dim, a, b, var,
                                             ravel_bits(ket), ravel_bits(bra), multiplier);
                });
            }
        }
    }

    std::array<int, FULL_DIM> basis_permutation_for_keep(const std::array<int, QUBITS> &keep)
    {
        std::array<int, FULL_DIM> perm{};
        for (int a = 0; a < FULL_DIM; ++a)
        {
            const auto multi = unravel6(a);
            std::array<int, QUBITS> ket{};
            for (int pos = 0; pos < QUBITS; ++pos)
            {
                ket[static_cast<std::size_t>(keep[static_cast<std::size_t>(pos)])] =
                    multi[static_cast<std::size_t>(pos)];
            }
            perm[static_cast<std::size_t>(a)] = ravel_bits(ket);
        }
        return perm;
    }

    void add_cvec_scalar_from_entry(RowBuilder &builder,
                                    int scalar_row,
                                    bool want_imag,
                                    const HermitianVariable &var,
                                    int source_i,
                                    int source_j,
                                    double multiplier,
                                    bool offdiag_scaled)
    {
        const double factor = offdiag_scaled ? SQRT2 : 1.0;
        add_complex_entry_to_scalar_row(builder, scalar_row, var, source_i, source_j,
                                        Complex(multiplier, 0.0),
                                        want_imag ? 0.0 : factor,
                                        want_imag ? factor : 0.0);
    }

    void add_copy_swap_equalities_fixed(RowBuilder &builder,
                                        const HermitianVariable &var,
                                        const std::array<int, FULL_DIM> &perm)
    {
        for (int a = 0; a < FULL_DIM; ++a)
        {
            for (int b = 0; b <= a; ++b)
            {
                if (a == b)
                {
                    const int row = builder.add_row(0.0);
                    add_cvec_scalar_from_entry(builder, row, false, var,
                                               perm[static_cast<std::size_t>(a)],
                                               perm[static_cast<std::size_t>(b)], 1.0, false);
                    add_cvec_scalar_from_entry(builder, row, false, var, a, b, -1.0, false);
                    builder.drop_last_empty_zero_row();
                }
                else
                {
                    int row = builder.add_row(0.0);
                    add_cvec_scalar_from_entry(builder, row, false, var,
                                               perm[static_cast<std::size_t>(a)],
                                               perm[static_cast<std::size_t>(b)], 1.0, true);
                    add_cvec_scalar_from_entry(builder, row, false, var, a, b, -1.0, true);
                    builder.drop_last_empty_zero_row();

                    row = builder.add_row(0.0);
                    add_cvec_scalar_from_entry(builder, row, true, var,
                                               perm[static_cast<std::size_t>(a)],
                                               perm[static_cast<std::size_t>(b)], 1.0, true);
                    add_cvec_scalar_from_entry(builder, row, true, var, a, b, -1.0, true);
                    builder.drop_last_empty_zero_row();
                }
            }
        }
    }

    std::array<Complex, RHO_DIM * RHO_DIM> normalize_hermitian_rho(const double *rho_ri)
    {
        std::array<Complex, RHO_DIM * RHO_DIM> rho{};
        for (int r = 0; r < RHO_DIM; ++r)
        {
            for (int c = 0; c < RHO_DIM; ++c)
            {
                const int idx = r * RHO_DIM + c;
                rho[static_cast<std::size_t>(idx)] =
                    Complex(rho_ri[2 * idx + 0], rho_ri[2 * idx + 1]);
            }
        }

        const auto raw = rho;
        for (int r = 0; r < RHO_DIM; ++r)
        {
            for (int c = 0; c < RHO_DIM; ++c)
            {
                rho[static_cast<std::size_t>(r * RHO_DIM + c)] =
                    0.5 * (raw[static_cast<std::size_t>(r * RHO_DIM + c)] +
                           std::conj(raw[static_cast<std::size_t>(c * RHO_DIM + r)]));
            }
        }

        double trace = 0.0;
        for (int i = 0; i < RHO_DIM; ++i)
        {
            trace += rho[static_cast<std::size_t>(i * RHO_DIM + i)].real();
        }
        if (std::abs(trace) <= 1.0e-15)
        {
            throw std::runtime_error("Selected RDM has zero trace.");
        }
        for (auto &z : rho)
        {
            z /= trace;
        }
        return rho;
    }

    std::vector<double> cvec_from_rho(const double *rho_ri)
    {
        const auto rho = normalize_hermitian_rho(rho_ri);
        std::vector<double> out(static_cast<std::size_t>(cvec_len(RHO_DIM)), 0.0);
        for (int c = 0; c < RHO_DIM; ++c)
        {
            for (int r = c; r < RHO_DIM; ++r)
            {
                if (r == c)
                {
                    out[static_cast<std::size_t>(cvec_index(RHO_DIM, r, c, false))] =
                        rho[static_cast<std::size_t>(r * RHO_DIM + c)].real();
                }
                else
                {
                    const Complex z = rho[static_cast<std::size_t>(r * RHO_DIM + c)];
                    out[static_cast<std::size_t>(cvec_index(RHO_DIM, r, c, false))] = SQRT2 * z.real();
                    out[static_cast<std::size_t>(cvec_index(RHO_DIM, r, c, true))] = SQRT2 * z.imag();
                }
            }
        }
        return out;
    }

    std::vector<double> white_cvec_3q()
    {
        std::vector<double> out(static_cast<std::size_t>(cvec_len(RHO_DIM)), 0.0);
        for (int i = 0; i < RHO_DIM; ++i)
        {
            out[static_cast<std::size_t>(cvec_index(RHO_DIM, i, i, false))] = 1.0 / RHO_DIM;
        }
        return out;
    }

    int add_zero_rows(RowBuilder &builder, int count, const std::vector<double> *rhs = nullptr)
    {
        const int base = builder.row_count();
        for (int i = 0; i < count; ++i)
        {
            builder.add_row(rhs == nullptr ? 0.0 : rhs->at(static_cast<std::size_t>(i)));
        }
        return base;
    }

    ConeProgram build_ring_inflation_program(const double *rho_ri)
    {
        const int t_col = 0;
        const HermitianVariable tau{FULL_DIM, 1};
        const HermitianVariable gamma{FULL_DIM, 1 + cvec_len(FULL_DIM)};
        const int ncols = 1 + 2 * cvec_len(FULL_DIM);

        const std::vector<double> rho_cvec = cvec_from_rho(rho_ri);
        const std::vector<double> white = white_cvec_3q();
        std::vector<double> delta(rho_cvec.size(), 0.0);
        for (std::size_t i = 0; i < rho_cvec.size(); ++i)
        {
            delta[i] = rho_cvec[i] - white[i];
        }

        RowBuilder builder;

        // tau_012 = I/8 + t*(rho - I/8)
        {
            const int base = add_zero_rows(builder, cvec_len(RHO_DIM), &white);
            add_ordered_marginal_to_rows(builder, base, tau, {0, 1, 2}, 1.0);
            for (int r = 0; r < cvec_len(RHO_DIM); ++r)
            {
                builder.add_coeff(base + r, t_col, -delta[static_cast<std::size_t>(r)]);
            }
        }

        // Ring-inflation marginal equalities.
        {
            int base = add_zero_rows(builder, cvec_len(16));
            add_ordered_marginal_to_rows(builder, base, gamma, {0, 1, 3, 4}, 1.0);
            add_ordered_marginal_to_rows(builder, base, tau, {0, 1, 3, 4}, -1.0);

            base = add_zero_rows(builder, cvec_len(16));
            add_ordered_marginal_to_rows(builder, base, gamma, {1, 2, 4, 5}, 1.0);
            add_ordered_marginal_to_rows(builder, base, tau, {1, 2, 4, 5}, -1.0);

            base = add_zero_rows(builder, cvec_len(16));
            add_ordered_marginal_to_rows(builder, base, gamma, {2, 3, 5, 0}, 1.0);
            add_ordered_marginal_to_rows(builder, base, tau, {2, 0, 5, 3}, -1.0);
        }

        // Exact copy-swap symmetry: copy order [3,4,5,0,1,2].
        const auto copy_perm = basis_permutation_for_keep({3, 4, 5, 0, 1, 2});
        add_copy_swap_equalities_fixed(builder, tau, copy_perm);
        add_copy_swap_equalities_fixed(builder, gamma, copy_perm);

        const int zero_rows = builder.row_count();

        // Positive cone rows for 0 <= t <= 1.
        {
            int row = builder.add_row(0.0);
            builder.add_coeff(row, t_col, -1.0); // slack = t
            row = builder.add_row(1.0);
            builder.add_coeff(row, t_col, 1.0);  // slack = 1 - t
        }

        const int linear_positive_rows = 2;

        auto add_psd_identity_block = [&](const HermitianVariable &var) {
            const int base = add_zero_rows(builder, cvec_len(var.dim));
            for (int r = 0; r < cvec_len(var.dim); ++r)
            {
                builder.add_coeff(base + r, var.offset + r, -1.0); // slack = variable matrix cvec
            }
        };

        auto add_psd_expression_block = [&](int dim,
                                            const std::function<void(int)> &fill_expression_rows) {
            const int base = add_zero_rows(builder, cvec_len(dim));
            fill_expression_rows(base);
        };

        // tau >= 0 and gamma >= 0.
        add_psd_identity_block(tau);
        add_psd_identity_block(gamma);

        // tau^{T_012} >= 0.
        add_psd_expression_block(FULL_DIM, [&](int base) {
            add_partial_transpose_to_rows(builder, base, tau, {0, 1, 2}, -1.0);
        });

        // PPT constraints on gamma marginals. Slack = expression, so A = -expression.
        add_psd_expression_block(16, [&](int base) {
            add_partial_transposed_marginal_to_rows(builder, base, gamma, {0, 1, 2, 4}, {3}, -1.0);
        });
        add_psd_expression_block(16, [&](int base) {
            add_partial_transposed_marginal_to_rows(builder, base, gamma, {1, 2, 3, 5}, {3}, -1.0);
        });
        add_psd_expression_block(16, [&](int base) {
            add_partial_transposed_marginal_to_rows(builder, base, gamma, {2, 3, 4, 0}, {3}, -1.0);
        });

        ConeProgram program;
        program.rows = static_cast<scs_int>(builder.row_count());
        program.cols = static_cast<scs_int>(ncols);
        program.b.resize(static_cast<std::size_t>(program.rows));
        for (std::size_t i = 0; i < program.b.size(); ++i)
        {
            program.b[i] = static_cast<scs_float>(builder.rhs()[i]);
        }
        program.c.assign(static_cast<std::size_t>(program.cols), 0.0);
        program.c[static_cast<std::size_t>(t_col)] = -1.0; // maximize t

        program.complex_psd_dims = {FULL_DIM, FULL_DIM, FULL_DIM, 16, 16, 16};
        program.A_storage = builder.to_csc(ncols);

        program.data.m = program.rows;
        program.data.n = program.cols;
        program.data.A = &program.A_storage.matrix;
        program.data.P = nullptr;
        program.data.b = program.b.data();
        program.data.c = program.c.data();

        program.cone.z = static_cast<scs_int>(zero_rows);
        program.cone.l = static_cast<scs_int>(linear_positive_rows);
        program.cone.bsize = 0;
        program.cone.bl = nullptr;
        program.cone.bu = nullptr;
        program.cone.q = nullptr;
        program.cone.qsize = 0;
        program.cone.s = nullptr;
        program.cone.ssize = 0;
        program.cone.cs = program.complex_psd_dims.data();
        program.cone.cssize = static_cast<scs_int>(program.complex_psd_dims.size());
        program.cone.ep = 0;
        program.cone.ed = 0;
        program.cone.p = nullptr;
        program.cone.psize = 0;

        return program;
    }

    SCSResult solve_ring_inflation_scs(const double *rho_ri,
                                       int max_iters,
                                       double tol,
                                       double time_limit_secs,
                                       bool verbose)
    {
        ConeProgram program = build_ring_inflation_program(rho_ri);
        bind_cone_program(program);

        ScsSettings settings{};
        scs_set_default_settings(&settings);
        settings.max_iters = static_cast<scs_int>(max_iters);
        settings.eps_abs = static_cast<scs_float>(tol);
        settings.eps_rel = static_cast<scs_float>(tol);
        settings.eps_infeas = static_cast<scs_float>(tol);
        settings.time_limit_secs = static_cast<scs_float>(time_limit_secs);
        settings.verbose = verbose ? 1 : 0;
        settings.normalize = 1;
        settings.acceleration_lookback = 0;
        settings.warm_start = 0;

        std::vector<scs_float> x(static_cast<std::size_t>(program.cols), 0.0);
        std::vector<scs_float> y(static_cast<std::size_t>(program.rows), 0.0);
        std::vector<scs_float> s(static_cast<std::size_t>(program.rows), 0.0);
        ScsSolution sol{};
        sol.x = x.data();
        sol.y = y.data();
        sol.s = s.data();

        ScsInfo info{};
        const scs_int flag = scs(&program.data, &program.cone, &settings, &sol, &info);

        double raw = static_cast<double>(x[0]);
        if (std::isfinite(raw))
        {
            raw = std::min(1.0, std::max(0.0, raw));
        }

        SCSResult result;
        result.score = raw;
        result.status = info.status;
        result.status_val = (info.status_val != 0) ? info.status_val : flag;
        result.iter = info.iter;
        result.solve_time_ms = static_cast<double>(info.solve_time);
        return result;
    }

    Candidate select_highest_gmn_subsystem(const mipt_io::DensityFileMetadata &metadata,
                                           const mipt_io::DensityRecord &record)
    {
        if (metadata.kept_qubits != 3 || metadata.matrix_dimension != RHO_DIM)
        {
            throw std::runtime_error("Input file must contain three-qubit 8x8 reduced density matrices.");
        }
        if (record.rho_ri.size() !=
            static_cast<std::size_t>(metadata.subsystem_count) * RHO_RI_VALUES)
        {
            throw std::runtime_error("Density record payload size is inconsistent with metadata.");
        }

        std::vector<double> gmns(static_cast<std::size_t>(metadata.subsystem_count),
                                 -std::numeric_limits<double>::infinity());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (std::int64_t s = 0; s < static_cast<std::int64_t>(metadata.subsystem_count); ++s)
        {
            const double *rho = record.rho_ri.data() +
                static_cast<std::size_t>(s) * RHO_RI_VALUES;
            gmns[static_cast<std::size_t>(s)] = compute_gmn_mosek_complex_8x8(rho);
        }

        std::uint32_t best_index = 0;
        double best_gmn = -std::numeric_limits<double>::infinity();
        for (std::uint32_t s = 0; s < metadata.subsystem_count; ++s)
        {
            const double value = gmns[static_cast<std::size_t>(s)];
            if (std::isfinite(value) && value > best_gmn)
            {
                best_gmn = value;
                best_index = s;
            }
        }

        if (!std::isfinite(best_gmn))
        {
            throw std::runtime_error("All GMN solves failed or returned non-finite values for a record.");
        }

        Candidate candidate;
        candidate.subsystem = best_index;
        candidate.gmn = best_gmn;
        const double *rho = record.rho_ri.data() +
            static_cast<std::size_t>(best_index) * RHO_RI_VALUES;
        std::copy(rho, rho + RHO_RI_VALUES, candidate.rho_ri.begin());
        return candidate;
    }
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options opts = parse_options(argc, argv);

#ifdef _OPENMP
        if (omp_get_max_threads() > 1)
        {
            set_env_if_missing("GMN_MOSEK_NUM_THREADS", "1");
        }
#endif

        mipt_io::DensityMatrixReader reader(opts.input_file);
        const auto &metadata = reader.metadata();
        if (metadata.kept_qubits != 3 || metadata.matrix_dimension != RHO_DIM)
        {
            throw std::runtime_error("Input file must contain three-qubit 8x8 RDMs.");
        }

        std::ofstream out(opts.output_csv, std::ios::trunc);
        if (!out)
        {
            throw std::runtime_error("Could not create output CSV: " + opts.output_csv);
        }
        out << "p,gmn,inflation_score\n";
        out << std::setprecision(17);

        std::cout << "Running GMN-selected level-2 ring inflation from " << opts.input_file << "\n"
                  << "Records: " << metadata.record_count
                  << "; subsystems per record: " << metadata.subsystem_count
                  << "; SCS max_iters=" << opts.max_iters
                  << "; tol=" << opts.tol
                  << "; time_limit_secs=" << opts.time_limit_secs << "\n"
                  << "Output CSV: " << opts.output_csv << "\n";
#ifdef _OPENMP
        std::cout << "OpenMP GMN workers: " << omp_get_max_threads()
                  << "; GMN_MOSEK_NUM_THREADS="
                  << (std::getenv("GMN_MOSEK_NUM_THREADS")
                          ? std::getenv("GMN_MOSEK_NUM_THREADS")
                          : "MOSEK default")
                  << "\n";
#endif

        mipt_io::DensityRecord record;
        std::uint64_t processed = 0;
        const auto start = std::chrono::steady_clock::now();

        while (reader.read_record(record))
        {
            Candidate best = select_highest_gmn_subsystem(metadata, record);
            const SCSResult inflation = solve_ring_inflation_scs(
                best.rho_ri.data(), opts.max_iters, opts.tol, opts.time_limit_secs,
                opts.scs_verbose);

            out << record.p << ',' << best.gmn << ',' << inflation.score << '\n';
            ++processed;

            const auto now = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = now - start;
            const double rate = elapsed.count() > 0.0 ? processed / elapsed.count() : 0.0;
            const double eta = rate > 0.0 ? (metadata.record_count - processed) / rate : 0.0;

            std::cout << "\rProcessed " << processed << '/' << metadata.record_count
                      << "; p=" << std::setprecision(6) << record.p
                      << "; best_gmn=" << std::setprecision(6) << best.gmn
                      << "; inflation=" << std::setprecision(6) << inflation.score
                      << "; SCS=" << inflation.status
                      << " (" << inflation.iter << " iters)"
                      << "; ETA " << std::fixed << std::setprecision(1) << eta << " s       "
                      << std::flush;
        }

        std::cout << "\nWrote " << opts.output_csv << ".\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "ring_inflation.exe error: " << error.what() << "\n";
        return 1;
    }
}
