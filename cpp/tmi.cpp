#include "density_io.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    using Clock = std::chrono::steady_clock;
    using C64 = std::complex<double>;

    constexpr int RHO3_QUBITS = 3;
    constexpr int RHO3_DIM = 1 << RHO3_QUBITS; // 8
    constexpr int RHO3_VALUES = 2 * RHO3_DIM * RHO3_DIM;
    constexpr double LOG2_EPS = 1.0e-15;

    enum class TraceMode
    {
        Qubit,
        Fermion
    };

    struct TracePlan
    {
        int keep_mask = 0;
        int dim = 0;
        int env_terms = 0;
        int output_elements = 0;
        std::array<int, 16 * 4> source_elements{}; // max 4x4 output, 4 traced terms
        std::array<int, 16 * 4> source_signs{};    // +1 for qubit trace; +/-1 for fermionic trace
    };

    int popcount3(int x)
    {
        return ((x >> 0) & 1) + ((x >> 1) & 1) + ((x >> 2) & 1);
    }

    int expand_bits(int compact, const std::array<int, 3> &positions, int count)
    {
        int expanded = 0;
        for (int i = 0; i < count; ++i)
        {
            expanded |= ((compact >> i) & 1) << positions[static_cast<std::size_t>(i)];
        }
        return expanded;
    }

    int index_from_new_mode_order_little(int new_index, const std::array<int, 3> &new_mode_order)
    {
        int old_index = 0;
        for (int slot = 0; slot < RHO3_QUBITS; ++slot)
        {
            const int mode = new_mode_order[static_cast<std::size_t>(slot)];
            old_index |= ((new_index >> slot) & 1) << mode;
        }
        return old_index;
    }

    int fermionic_reorder_sign_little(int new_index, const std::array<int, 3> &new_mode_order)
    {
        // Canonical basis: |n0 n1 n2> = (c0^dag)^n0 (c1^dag)^n1 (c2^dag)^n2 |vac>.
        // This returns s in |old/canonical> = s |new_mode_order>, where new_index is
        // indexed in the new order.  It is (-1)^(number of occupied-mode inversions).
        std::array<int, 3> pos_new{};
        for (int slot = 0; slot < RHO3_QUBITS; ++slot)
        {
            pos_new[static_cast<std::size_t>(new_mode_order[static_cast<std::size_t>(slot)])] = slot;
        }

        int inversions = 0;
        for (int mode_i = 0; mode_i < RHO3_QUBITS; ++mode_i)
        {
            const int slot_i = pos_new[static_cast<std::size_t>(mode_i)];
            const int occ_i = (new_index >> slot_i) & 1;
            if (!occ_i)
            {
                continue;
            }
            for (int mode_j = mode_i + 1; mode_j < RHO3_QUBITS; ++mode_j)
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

    TracePlan make_trace_plan(int keep_mask, TraceMode mode)
    {
        TracePlan plan;
        plan.keep_mask = keep_mask;
        const int kept_count = popcount3(keep_mask);
        const int traced_count = RHO3_QUBITS - kept_count;
        plan.dim = 1 << kept_count;
        plan.env_terms = 1 << traced_count;
        plan.output_elements = plan.dim * plan.dim;

        std::array<int, 3> kept_pos{};
        std::array<int, 3> traced_pos{};
        int nk = 0;
        int nt = 0;
        for (int q = 0; q < RHO3_QUBITS; ++q)
        {
            if ((keep_mask >> q) & 1)
            {
                kept_pos[static_cast<std::size_t>(nk++)] = q;
            }
            else
            {
                traced_pos[static_cast<std::size_t>(nt++)] = q;
            }
        }

        std::array<int, 3> new_mode_order{};
        for (int i = 0; i < kept_count; ++i)
        {
            new_mode_order[static_cast<std::size_t>(i)] = kept_pos[static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < traced_count; ++i)
        {
            new_mode_order[static_cast<std::size_t>(kept_count + i)] = traced_pos[static_cast<std::size_t>(i)];
        }

        int cursor = 0;
        for (int row_k = 0; row_k < plan.dim; ++row_k)
        {
            for (int col_k = 0; col_k < plan.dim; ++col_k)
            {
                for (int e = 0; e < plan.env_terms; ++e)
                {
                    // In little-endian compact indexing, the kept subsystem occupies
                    // the low bits and the traced suffix occupies the high bits.
                    const int row_new = row_k + plan.dim * e;
                    const int col_new = col_k + plan.dim * e;
                    const int row = index_from_new_mode_order_little(row_new, new_mode_order);
                    const int col = index_from_new_mode_order_little(col_new, new_mode_order);

                    int sign = 1;
                    if (mode == TraceMode::Fermion)
                    {
                        sign = fermionic_reorder_sign_little(row_new, new_mode_order) *
                               fermionic_reorder_sign_little(col_new, new_mode_order);
                    }

                    plan.source_elements[static_cast<std::size_t>(cursor)] = row * RHO3_DIM + col;
                    plan.source_signs[static_cast<std::size_t>(cursor)] = sign;
                    ++cursor;
                }
            }
        }
        return plan;
    }

    const std::array<TracePlan, 6> &trace_plans(TraceMode mode)
    {
        static const std::array<TracePlan, 6> qubit_plans = {
            make_trace_plan(0b001, TraceMode::Qubit), // A
            make_trace_plan(0b010, TraceMode::Qubit), // B
            make_trace_plan(0b100, TraceMode::Qubit), // C
            make_trace_plan(0b011, TraceMode::Qubit), // AB
            make_trace_plan(0b101, TraceMode::Qubit), // AC
            make_trace_plan(0b110, TraceMode::Qubit)  // BC
        };
        static const std::array<TracePlan, 6> fermion_plans = {
            make_trace_plan(0b001, TraceMode::Fermion), // A
            make_trace_plan(0b010, TraceMode::Fermion), // B
            make_trace_plan(0b100, TraceMode::Fermion), // C
            make_trace_plan(0b011, TraceMode::Fermion), // AB
            make_trace_plan(0b101, TraceMode::Fermion), // AC
            make_trace_plan(0b110, TraceMode::Fermion)  // BC
        };
        return mode == TraceMode::Fermion ? fermion_plans : qubit_plans;
    }

    void partial_trace_from_rho3(const double *rho_ri, const TracePlan &plan, C64 *out)
    {
        int cursor = 0;
        for (int element = 0; element < plan.output_elements; ++element)
        {
            double re = 0.0;
            double im = 0.0;
            for (int t = 0; t < plan.env_terms; ++t)
            {
                const int src = plan.source_elements[static_cast<std::size_t>(cursor)];
                const int sign = plan.source_signs[static_cast<std::size_t>(cursor)];
                re += static_cast<double>(sign) * rho_ri[2 * src + 0];
                im += static_cast<double>(sign) * rho_ri[2 * src + 1];
                ++cursor;
            }
            out[element] = C64(re, im);
        }
    }

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

    void append_int(std::string &s, int value)
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

        // Conservative fallback for older libstdc++ double-format edge cases.
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

    double finish_entropy(double s)
    {
        if (s < 0.0 && s > -1.0e-12)
        {
            return 0.0;
        }
        return s;
    }

    double entropy_hermitian_2x2(const C64 *m)
    {
        const double a = m[0].real();
        const double d = m[3].real();
        const C64 b = 0.5 * (m[1] + std::conj(m[2]));
        const double half_trace = 0.5 * (a + d);
        const double half_diff = 0.5 * (a - d);
        const double radius = std::sqrt(std::max(0.0, half_diff * half_diff + std::norm(b)));
        const double l0 = half_trace - radius;
        const double l1 = half_trace + radius;
        return finish_entropy(entropy_from_eigenvalue(l0) + entropy_from_eigenvalue(l1));
    }

    template <int N>
    void hermitize_in_place(std::array<C64, N * N> &a)
    {
        for (int i = 0; i < N; ++i)
        {
            a[static_cast<std::size_t>(i * N + i)] = C64(a[static_cast<std::size_t>(i * N + i)].real(), 0.0);
            for (int j = i + 1; j < N; ++j)
            {
                const C64 h = 0.5 * (a[static_cast<std::size_t>(i * N + j)] +
                                     std::conj(a[static_cast<std::size_t>(j * N + i)]));
                a[static_cast<std::size_t>(i * N + j)] = h;
                a[static_cast<std::size_t>(j * N + i)] = std::conj(h);
            }
        }
    }

    template <int N>
    double entropy_hermitian_jacobi(std::array<C64, N * N> a)
    {
        hermitize_in_place<N>(a);

        double scale = 1.0;
        for (int i = 0; i < N; ++i)
        {
            scale = std::max(scale, std::abs(a[static_cast<std::size_t>(i * N + i)].real()));
            for (int j = i + 1; j < N; ++j)
            {
                scale = std::max(scale, std::abs(a[static_cast<std::size_t>(i * N + j)]));
            }
        }
        const double tol = 1.0e-14 * scale;
        const int max_rotations = 128 * N * N;

        for (int iter = 0; iter < max_rotations; ++iter)
        {
            int p = 0;
            int q = 1;
            double max_offdiag = 0.0;
            for (int i = 0; i < N - 1; ++i)
            {
                for (int j = i + 1; j < N; ++j)
                {
                    const double v = std::abs(a[static_cast<std::size_t>(i * N + j)]);
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

            const std::size_t pp = static_cast<std::size_t>(p * N + p);
            const std::size_t qq = static_cast<std::size_t>(q * N + q);
            const std::size_t pq = static_cast<std::size_t>(p * N + q);
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

            for (int k = 0; k < N; ++k)
            {
                if (k == p || k == q)
                {
                    continue;
                }
                const std::size_t kp = static_cast<std::size_t>(k * N + p);
                const std::size_t kq = static_cast<std::size_t>(k * N + q);
                const C64 old_kp = a[kp];
                const C64 old_kq = a[kq];
                const C64 new_kp = c * old_kp * phase - s * old_kq;
                const C64 new_kq = s * old_kp * phase + c * old_kq;
                a[kp] = new_kp;
                a[static_cast<std::size_t>(p * N + k)] = std::conj(new_kp);
                a[kq] = new_kq;
                a[static_cast<std::size_t>(q * N + k)] = std::conj(new_kq);
            }

            a[pp] = C64(app - t * b_abs, 0.0);
            a[qq] = C64(aqq + t * b_abs, 0.0);
            a[pq] = C64(0.0, 0.0);
            a[static_cast<std::size_t>(q * N + p)] = C64(0.0, 0.0);
        }

        double entropy = 0.0;
        for (int i = 0; i < N; ++i)
        {
            entropy += entropy_from_eigenvalue(a[static_cast<std::size_t>(i * N + i)].real());
        }
        return finish_entropy(entropy);
    }

    double entropy_hermitian_4x4(const C64 *m)
    {
        std::array<C64, 16> a{};
        for (int i = 0; i < 16; ++i)
        {
            a[static_cast<std::size_t>(i)] = m[i];
        }
        return entropy_hermitian_jacobi<4>(a);
    }

    double entropy_rho3_8x8(const double *rho_ri)
    {
        std::array<C64, 64> a{};
        for (int i = 0; i < 64; ++i)
        {
            a[static_cast<std::size_t>(i)] = C64(rho_ri[2 * i + 0], rho_ri[2 * i + 1]);
        }
        return entropy_hermitian_jacobi<8>(a);
    }

    double tmi_rho3(const double *rho_ri, TraceMode mode)
    {
        const auto &plans = trace_plans(mode);
        std::array<C64, 4> rho_1q{};
        std::array<C64, 16> rho_2q{};

        partial_trace_from_rho3(rho_ri, plans[0], rho_1q.data());
        const double s_a = entropy_hermitian_2x2(rho_1q.data());

        partial_trace_from_rho3(rho_ri, plans[1], rho_1q.data());
        const double s_b = entropy_hermitian_2x2(rho_1q.data());

        partial_trace_from_rho3(rho_ri, plans[2], rho_1q.data());
        const double s_c = entropy_hermitian_2x2(rho_1q.data());

        partial_trace_from_rho3(rho_ri, plans[3], rho_2q.data());
        const double s_ab = entropy_hermitian_4x4(rho_2q.data());

        partial_trace_from_rho3(rho_ri, plans[4], rho_2q.data());
        const double s_ac = entropy_hermitian_4x4(rho_2q.data());

        partial_trace_from_rho3(rho_ri, plans[5], rho_2q.data());
        const double s_bc = entropy_hermitian_4x4(rho_2q.data());

        const double s_abc = entropy_rho3_8x8(rho_ri);
        return s_a + s_b + s_c - s_ab - s_ac - s_bc + s_abc;
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
        throw std::invalid_argument("Invalid trace_mode: expected 0 for qubit or 1 for fermion, got " +
                                    std::string(text));
    }

    const char *trace_mode_name(TraceMode mode)
    {
        return mode == TraceMode::Fermion ? "fermion" : "qubit";
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

    class ProgressReporter
    {
      public:
        explicit ProgressReporter(std::uint64_t total_matrices)
            : total_(total_matrices), start_(Clock::now()), last_(start_)
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
                      << "processed " << processed_ << '/' << total_ << " matrices"
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
            << "  " << argv0 << " <trace_mode> <rho_file.bin> <output.csv> [limit_per_p]\n\n"
            << "Arguments:\n"
            << "  trace_mode = 0  ordinary qubit partial traces inside each 8x8 rho_ABC\n"
            << "  trace_mode = 1  fermionic partial traces inside each 8x8 rho_ABC\n\n"
            << "Output columns:\n"
            << "  p,tmi\n\n"
            << "Notes:\n"
            << "  * This expects the MIPTRHO2 .bin format with kept_qubits=3 and D=8.\n"
            << "  * Fermion mode assumes little-endian occupation/Jordan-Wigner basis.\n"
            << "  * limit_per_p limits processed trajectory records per p value; 0 means unlimited.\n";
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

        mipt_io::DensityMatrixReader reader(input_path);
        const auto &meta = reader.metadata();
        if (meta.kept_qubits != 3 || meta.matrix_dimension != RHO3_DIM)
        {
            throw std::runtime_error("tmi.exe expects three-qubit reduced density matrices (kept_qubits=3, D=8).");
        }

        const std::uint64_t subsystem_count = meta.subsystem_count;
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
        const std::uint64_t total_matrices = planned_records * subsystem_count;

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

        *out << "p,tmi\n";

        std::cerr << "Input: " << input_path << '\n'
                  << "trace_mode=" << trace_mode_name(trace_mode) << '\n'
                  << "records=" << meta.record_count
                  << ", p_values=" << meta.resolution
                  << ", realizations=" << meta.realizations
                  << ", subsystems/record=" << meta.subsystem_count
                  << ", planned_matrices=" << total_matrices << '\n';

#ifdef _OPENMP
        std::cerr << "OpenMP threads=" << omp_get_max_threads() << '\n';
#endif

        std::vector<std::uint64_t> records_done_per_p(std::max<std::uint32_t>(meta.resolution, 1), 0);
        std::vector<double> tmi_values(static_cast<std::size_t>(subsystem_count), 0.0);
        mipt_io::DensityRecord record;
        ProgressReporter progress(total_matrices);
        std::uint64_t records_read = 0;
        std::uint64_t matrices_processed = 0;

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
                continue;
            }
            ++records_done_per_p[record.p_index];

            if (record.rho_ri.size() != mipt_io::payload_value_count(meta))
            {
                throw std::runtime_error("Density record has unexpected payload size.");
            }

#pragma omp parallel for schedule(static) if (subsystem_count >= 4)
            for (std::int64_t s = 0; s < static_cast<std::int64_t>(subsystem_count); ++s)
            {
                const double *rho = record.rho_ri.data() +
                                    static_cast<std::size_t>(s) * RHO3_VALUES;
                tmi_values[static_cast<std::size_t>(s)] = tmi_rho3(rho, trace_mode);
            }

            std::string lines;
            lines.reserve(static_cast<std::size_t>(subsystem_count) * 128u);
            for (std::uint64_t s = 0; s < subsystem_count; ++s)
            {
                const std::size_t qs = static_cast<std::size_t>(s) * meta.kept_qubits;
                append_double(lines, record.p);
                lines.push_back(',');
                append_double(lines, tmi_values[static_cast<std::size_t>(s)]);
                lines.push_back('\n');
            }
            out->write(lines.data(), static_cast<std::streamsize>(lines.size()));
            if (!*out)
            {
                throw std::runtime_error("Failed while writing output CSV.");
            }

            matrices_processed += subsystem_count;
            progress.set_processed(matrices_processed);
            progress.maybe_report(false);
        }

        if (output_file)
        {
            output_file.flush();
        }
        progress.set_processed(matrices_processed);
        progress.maybe_report(true);

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nerror: " << e.what() << '\n';
        return 1;
    }
}
