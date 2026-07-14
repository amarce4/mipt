#include "mipt/density.hpp"
#include "mipt/env.hpp"
#include "mipt/analysis/gmn.hpp"
#include "mipt/analysis/fgmn.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    constexpr std::size_t RHO3_VALUES = 2u * 8u * 8u;
    constexpr double GMN_ZERO_PREFILTER_TOL = 1.0e-10;

    enum class EvaluationMode
    {
        RealOnly = 0,
        Complex = 1
    };

    struct GmnJob
    {
        double p = 0.0;
        std::uint32_t p_index = 0;
        std::uint32_t realization = 0;
        std::uint32_t subsystem = 0;
        std::uint32_t q0 = 0;
        std::uint32_t q1 = 0;
        std::uint32_t q2 = 0;
        double imag_norm = 0.0;
        std::array<double, 64> rho_real{};
        std::array<double, RHO3_VALUES> rho_ri{};
    };

    const char *mode_name(EvaluationMode mode)
    {
        return (mode == EvaluationMode::Complex) ? "complex" : "real-only";
    }

    void print_usage(const char *program)
    {
        std::cout
            << "Usage: " << program
            << " [0|1] [input_file] [output_csv] [limit_per_p]\n\n"
            << "Modes:\n"
            << "  0    Real-only SDP using Re(rho). Fast legacy path.\n"
            << "  1    Complex Hermitian SDP using the full complex rho.\n\n"
            << "Defaults:\n"
            << "  " << program << "              same as: " << program << " 0 rho3.bin data.csv\n"
            << "  " << program << " 0            reads rho3.bin and writes data.csv\n"
            << "  " << program << " 1            reads rho3.bin and writes data.csv\n\n"
            << "Examples:\n"
            << "  " << program << " 0 rho3.bin data_real.csv\n"
            << "  " << program << " 1 rho3.bin data_complex.csv\n"
            << "  " << program << " 1 rho3_2d.bin data2_complex.csv\n"
            << "  " << program << " 1 rho3.bin data_limited.csv 1000\n\n"
            << "limit_per_p:\n"
            << "  Maximum number of full records/realizations evaluated per p-value.\n"
            << "  Use 0, or omit it, for no per-p limit.\n\n"
            << "Environment variables:\n"
            << "  OMP_NUM_THREADS=N          Number of parallel GMN worker threads.\n"
            << "  GMN_MOSEK_NUM_THREADS=N    MOSEK threads per GMN solve; use 1 when OMP > 1.\n"
            << "  GMN_BATCH_RECORDS=N        Records per parallel batch; default 256.\n"
            << "  GMN_LIMIT_PER_P=N          Fallback per-p record limit; 0 or unset means no limit.\n"
            << "  GMN_MOSEK_RECYCLE=N        Rebuild cached task every N solves/thread; default 10000, 0 disables.\n\n"
            << "The former leading argument '3' has been removed. Use mode 0 or 1.\n"
            << "Six-qubit retained-subsystem analysis is not implemented; input must contain\n"
            << "three-qubit 8x8 reduced density matrices.\n";
    }

    double imaginary_frobenius_norm(const double *rho_ri)
    {
        double norm_sq = 0.0;
        for (std::size_t i = 1; i < RHO3_VALUES; i += 2)
        {
            norm_sq += rho_ri[i] * rho_ri[i];
        }
        return std::sqrt(norm_sq);
    }

    std::array<double, 64> real_matrix(const double *rho_ri)
    {
        std::array<double, 64> real{};
        for (std::size_t i = 0; i < real.size(); ++i)
        {
            real[i] = rho_ri[2u * i];
        }
        return real;
    }

    std::array<double, RHO3_VALUES> complex_matrix(const double *rho_ri)
    {
        std::array<double, RHO3_VALUES> out{};
        std::copy(rho_ri, rho_ri + RHO3_VALUES, out.begin());
        return out;
    }

    struct MatrixSanitizeResult
    {
        std::uint64_t nonfinite_values = 0;
        bool changed = false;
    };

    bool finite_complex_entry(double re, double im)
    {
        return std::isfinite(re) && std::isfinite(im);
    }

    MatrixSanitizeResult sanitize_hermitian_complex_matrix(
        std::array<double, RHO3_VALUES> &rho_ri)
    {
        MatrixSanitizeResult result;
        constexpr std::size_t dim = 8;

        for (std::size_t row = 0; row < dim; ++row)
        {
            const std::size_t k = 2u * (row * dim + row);
            if (!std::isfinite(rho_ri[k]))
            {
                rho_ri[k] = 0.0;
                ++result.nonfinite_values;
                result.changed = true;
            }
            if (!std::isfinite(rho_ri[k + 1]))
            {
                ++result.nonfinite_values;
                result.changed = true;
            }
            if (rho_ri[k + 1] != 0.0)
            {
                rho_ri[k + 1] = 0.0;
                result.changed = true;
            }
        }

        for (std::size_t row = 0; row < dim; ++row)
        {
            for (std::size_t col = row + 1; col < dim; ++col)
            {
                const std::size_t a = 2u * (row * dim + col);
                const std::size_t b = 2u * (col * dim + row);

                const double are = rho_ri[a + 0];
                const double aim = rho_ri[a + 1];
                const double bre = rho_ri[b + 0];
                const double bim = rho_ri[b + 1];

                result.nonfinite_values += std::isfinite(are) ? 0u : 1u;
                result.nonfinite_values += std::isfinite(aim) ? 0u : 1u;
                result.nonfinite_values += std::isfinite(bre) ? 0u : 1u;
                result.nonfinite_values += std::isfinite(bim) ? 0u : 1u;

                const bool a_finite = finite_complex_entry(are, aim);
                const bool b_finite = finite_complex_entry(bre, bim);

                double re = 0.0;
                double im = 0.0;
                if (a_finite && b_finite)
                {
                    /* Hermitize by averaging rho[row,col] with conj(rho[col,row]). */
                    re = 0.5 * (are + bre);
                    im = 0.5 * (aim - bim);
                }
                else if (a_finite)
                {
                    re = are;
                    im = aim;
                }
                else if (b_finite)
                {
                    re = bre;
                    im = -bim;
                }

                if (rho_ri[a + 0] != re || rho_ri[a + 1] != im ||
                    rho_ri[b + 0] != re || rho_ri[b + 1] != -im)
                {
                    result.changed = true;
                }

                rho_ri[a + 0] = re;
                rho_ri[a + 1] = im;
                rho_ri[b + 0] = re;
                rho_ri[b + 1] = -im;
            }
        }

        return result;
    }

    double csv_safe_gmn(double value)
    {
        /* Keep the CSV numeric, but do not skip valid solves before this point.
         * Non-finite density-matrix entries are sanitized matrix-entry-wise
         * before MOSEK is called; this fallback only handles actual solver
         * failures or non-finite objectives.
         */
        if (!std::isfinite(value))
        {
            return 0.0;
        }
        return (value < 0.0 && value > -1.0e-10) ? 0.0 : value;
    }

    std::uint64_t parse_nonnegative_u64(const char *value, const char *name)
    {
        if (value == nullptr || *value == '\0' || value[0] == '-')
        {
            throw std::invalid_argument(
                std::string(name) + " must be a non-negative integer.");
        }

        char *end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0')
        {
            throw std::invalid_argument(
                std::string(name) + " must be a non-negative integer.");
        }

        return static_cast<std::uint64_t>(parsed);
    }

    std::uint64_t parse_limit_per_p(int argc, char *argv[], int argument_index)
    {
        if (argc > argument_index)
        {
            return parse_nonnegative_u64(
                argv[argument_index],
                "limit_per_p");
        }

        const char *value = std::getenv("GMN_LIMIT_PER_P");
        if (value != nullptr && *value != '\0')
        {
            return parse_nonnegative_u64(value, "GMN_LIMIT_PER_P");
        }

        return 0;
    }

    EvaluationMode parse_mode(const char *arg)
    {
        const std::string value(arg);
        if (value == "0")
        {
            return EvaluationMode::RealOnly;
        }
        if (value == "1")
        {
            return EvaluationMode::Complex;
        }
        if (value == "3")
        {
            throw std::invalid_argument(
                "The former leading argument '3' has been removed. Use '0' for real-only GMN or '1' for complex GMN.");
        }
        if (value == "6")
        {
            throw std::invalid_argument(
                "Six-qubit retained-subsystem analysis has been removed; use rho3 data only.");
        }
        throw std::invalid_argument("First argument must be evaluation mode 0 or 1.");
    }

    void append_jobs_from_record(const mipt::io::DensityFileMetadata &metadata,
                                 const mipt::io::DensityRecord &record,
                                 std::vector<GmnJob> &jobs,
                                 double &max_imag_norm,
                                 std::uint64_t &nonfinite_input_value_count,
                                 std::uint64_t &sanitized_matrix_count)
    {
        for (std::uint32_t subsystem = 0;
             subsystem < metadata.subsystem_count;
             ++subsystem)
        {
            const double *rho_ri = record.rho_ri.data() +
                static_cast<std::size_t>(subsystem) * RHO3_VALUES;
            std::array<double, RHO3_VALUES> rho_complex =
                complex_matrix(rho_ri);
            const MatrixSanitizeResult sanitize_result =
                sanitize_hermitian_complex_matrix(rho_complex);
            nonfinite_input_value_count += sanitize_result.nonfinite_values;
            if (sanitize_result.changed)
            {
                ++sanitized_matrix_count;
            }

            const double imag_norm = imaginary_frobenius_norm(rho_complex.data());
            max_imag_norm = std::max(max_imag_norm, imag_norm);

            const std::size_t q_offset =
                static_cast<std::size_t>(subsystem) * metadata.kept_qubits;

            GmnJob job;
            job.p = record.p;
            job.p_index = record.p_index;
            job.realization = record.realization;
            job.subsystem = subsystem;
            job.q0 = metadata.subsystem_qubits[q_offset + 0];
            job.q1 = metadata.subsystem_qubits[q_offset + 1];
            job.q2 = metadata.subsystem_qubits[q_offset + 2];
            job.imag_norm = imag_norm;
            job.rho_ri = rho_complex;
            job.rho_real = real_matrix(job.rho_ri.data());
            jobs.push_back(job);
        }
    }
}

int main(int argc, char *argv[])
{
    try
    {
        if (argc > 1 &&
            (std::string(argv[1]) == "--help" ||
             std::string(argv[1]) == "-h"))
        {
            print_usage(argv[0]);
            return 0;
        }

        EvaluationMode mode = EvaluationMode::RealOnly;
        int next_argument = 1;
        if (argc > 1)
        {
            mode = parse_mode(argv[1]);
            next_argument = 2;
        }

        const std::string input_path =
            (argc > next_argument) ? argv[next_argument] : "rho3.bin";
        const std::string output_path =
            (argc > next_argument + 1) ? argv[next_argument + 1] : "data.csv";
        const std::uint64_t limit_per_p =
            parse_limit_per_p(argc, argv, next_argument + 2);
        if (argc > next_argument + 3)
        {
            throw std::invalid_argument("Too many command-line arguments.");
        }

#ifdef _OPENMP
        /* Avoid nested oversubscription by default: parallelism is at the
         * independent-GMN-solve level, not inside each tiny MOSEK solve.
         */
        if (omp_get_max_threads() > 1)
        {
            mipt::env::set_if_unset("GMN_MOSEK_NUM_THREADS", "1");
        }
#endif

        gmn_mosek_reset_diagnostics();

        mipt::io::DensityMatrixReader reader(input_path);
        const auto &metadata = reader.metadata();

        if (metadata.kept_qubits != 3 || metadata.matrix_dimension != 8)
        {
            throw std::runtime_error(
                "Input file must contain three-qubit 8x8 reduced density matrices.");
        }

        std::ofstream outfile(output_path);
        if (!outfile)
        {
            throw std::runtime_error("Could not create output CSV: " + output_path);
        }

        outfile << "p,gmn,bipneg\n";
        outfile << std::setprecision(17);

        const std::uint64_t max_records_evaluated = [&]() -> std::uint64_t {
            if (limit_per_p == 0)
            {
                return metadata.record_count;
            }

            const std::uint64_t resolution =
                static_cast<std::uint64_t>(metadata.resolution);
            const std::uint64_t maximum_limited_records =
                (resolution != 0 &&
                 limit_per_p >
                     std::numeric_limits<std::uint64_t>::max() / resolution)
                    ? std::numeric_limits<std::uint64_t>::max()
                    : limit_per_p * resolution;

            return std::min<std::uint64_t>(
                metadata.record_count,
                maximum_limited_records);
        }();

        const std::uint64_t total_solves =
            max_records_evaluated *
            static_cast<std::uint64_t>(metadata.subsystem_count);

        std::cout << "Calculating three-qubit " << mode_name(mode)
                  << " GMN values from " << input_path
                  << " (" << metadata.record_count << " input records, "
                  << metadata.subsystem_count << " retained subsystems each, "
                  << "up to " << total_solves << " total solves";
        if (limit_per_p != 0)
        {
            std::cout << "; limit_per_p=" << limit_per_p
                      << " records/p-value";
        }
        std::cout << ").\n";

#ifdef _OPENMP
        std::cout << "OpenMP GMN workers: " << omp_get_max_threads()
                  << "; GMN_MOSEK_NUM_THREADS="
                  << (std::getenv("GMN_MOSEK_NUM_THREADS")
                          ? std::getenv("GMN_MOSEK_NUM_THREADS")
                          : "MOSEK default")
                  << ".\n";
#else
        std::cout << "OpenMP is not enabled in this build; GMN solves are serial.\n";
#endif

        const int batch_records = static_cast<int>(mipt::env::integer(
            "GMN_BATCH_RECORDS", 256, 1, std::numeric_limits<int>::max()));
        std::vector<GmnJob> jobs;
        jobs.reserve(static_cast<std::size_t>(batch_records) * metadata.subsystem_count);
        std::vector<double> gmn_values;
        std::vector<double> bipneg_values;
        std::vector<unsigned char> prefilter_skipped_flags;

        mipt::io::DensityRecord record;
        std::vector<std::uint64_t> accepted_records_per_p(
            metadata.resolution,
            0);
        std::uint64_t records_scanned = 0;
        std::uint64_t records_evaluated = 0;
        std::uint64_t records_skipped_by_limit = 0;
        bool reader_exhausted = false;
        std::uint64_t processed = 0;
        double max_imag_norm = 0.0;
        std::uint64_t nonfinite_input_value_count = 0;
        std::uint64_t sanitized_matrix_count = 0;
        std::uint64_t nonfinite_gmn_count = 0;
        std::uint64_t clipped_negative_gmn_count = 0;
        std::uint64_t zero_prefilter_skipped = 0;
        const auto start = std::chrono::steady_clock::now();
        auto interval_start = start;
        std::uint64_t interval_processed = 0;

        while (processed < total_solves && !reader_exhausted)
        {
            jobs.clear();

            int records_in_batch = 0;
            while (records_in_batch < batch_records)
            {
                if (!reader.read_record(record))
                {
                    reader_exhausted = true;
                    break;
                }

                ++records_scanned;
                ++records_in_batch;

                if (record.p_index >= accepted_records_per_p.size())
                {
                    throw std::runtime_error(
                        "Input record has p_index outside metadata.resolution.");
                }

                if (limit_per_p != 0 &&
                    accepted_records_per_p[record.p_index] >= limit_per_p)
                {
                    ++records_skipped_by_limit;
                    continue;
                }

                ++accepted_records_per_p[record.p_index];
                ++records_evaluated;
                append_jobs_from_record(metadata, record, jobs, max_imag_norm,
                                        nonfinite_input_value_count,
                                        sanitized_matrix_count);
            }

            if (jobs.empty())
            {
                if (!reader_exhausted)
                {
                    std::cout
                        << "\rScanned " << records_scanned
                        << " records; skipped " << records_skipped_by_limit
                        << " by limit_per_p; waiting for the next accepted "
                           "p-record...        "
                        << std::flush;
                    continue;
                }
                break;
            }

            gmn_values.assign(jobs.size(), std::numeric_limits<double>::quiet_NaN());
            bipneg_values.assign(jobs.size(), std::numeric_limits<double>::quiet_NaN());
            prefilter_skipped_flags.assign(jobs.size(), 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(jobs.size()); ++i)
            {
                const std::size_t idx = static_cast<std::size_t>(i);
                const GmnJob &job = jobs[idx];

                std::array<double, RHO3_VALUES> bipneg_rho = job.rho_ri;
                if (mode == EvaluationMode::RealOnly)
                {
                    for (std::size_t k = 1; k < RHO3_VALUES; k += 2)
                    {
                        bipneg_rho[k] = 0.0;
                    }
                }

                const double min_bipneg =
                    compute_min_bipartite_negativity_8x8_cpp(bipneg_rho.data());
                bipneg_values[idx] = min_bipneg;
                if (std::isfinite(min_bipneg) &&
                    min_bipneg <= GMN_ZERO_PREFILTER_TOL)
                {
                    gmn_values[idx] = 0.0;
                    prefilter_skipped_flags[idx] = 1;
                    continue;
                }

                if (mode == EvaluationMode::Complex)
                {
                    gmn_values[idx] =
                        compute_gmn_mosek_complex_8x8(job.rho_ri.data());
                }
                else
                {
                    gmn_values[idx] =
                        compute_gmn_mosek_real_8x8(job.rho_real.data());
                }
            }

            for (std::size_t i = 0; i < jobs.size(); ++i)
            {
                const GmnJob &job = jobs[i];
                const double raw_gmn = gmn_values[i];
                const double safe_gmn = csv_safe_gmn(raw_gmn);
                if (prefilter_skipped_flags[i] != 0)
                {
                    ++zero_prefilter_skipped;
                }
                if (!std::isfinite(raw_gmn))
                {
                    ++nonfinite_gmn_count;
                }
                else if (safe_gmn != raw_gmn)
                {
                    ++clipped_negative_gmn_count;
                }
                outfile << job.p << ',' << safe_gmn << ',' << bipneg_values[i] << '\n';
            }
            outfile.flush();

            processed += jobs.size();
            interval_processed += jobs.size();

            const auto now = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = now - start;
            const std::chrono::duration<double> interval_elapsed = now - interval_start;
            const double average_rate =
                (elapsed.count() > 0.0) ? processed / elapsed.count() : 0.0;
            const double recent_rate =
                (interval_elapsed.count() > 0.0)
                    ? interval_processed / interval_elapsed.count()
                    : 0.0;
            const double eta_seconds =
                (average_rate > 0.0)
                    ? (total_solves - processed) / average_rate
                    : 0.0;

            std::cout << "\rProcessed " << processed << "/"
                      << total_solves << " matrices from "
                      << records_evaluated << " records";
            if (records_skipped_by_limit != 0)
            {
                std::cout << " (skipped " << records_skipped_by_limit
                          << " by limit)";
            }
            std::cout << "; avg "
                      << std::fixed << std::setprecision(2)
                      << average_rate << "/s, recent "
                      << recent_rate << "/s, ETA "
                      << eta_seconds << " s        "
                      << std::flush;

            if (interval_elapsed.count() >= 5.0)
            {
                interval_start = now;
                interval_processed = 0;
            }
        }

        std::cout << "\nWrote " << output_path << ".\n"
                  << "Records scanned: " << records_scanned
                  << "; records evaluated: " << records_evaluated
                  << "; records skipped by limit_per_p: "
                  << records_skipped_by_limit << "\n"
                  << "Matrices skipped by bipartite-negativity zero prefilter: "
                  << zero_prefilter_skipped << "\n"
                  << "Maximum imaginary Frobenius norm in input records: "
                  << std::scientific << std::setprecision(6)
                  << max_imag_norm << "\n";

        if (nonfinite_input_value_count > 0)
        {
            std::cerr
                << "Warning: replaced " << nonfinite_input_value_count
                << " non-finite density-matrix value(s) with 0 before solving; "
                << sanitized_matrix_count << " matrix/matrices were sanitized.\n";
        }

        if (nonfinite_gmn_count > 0)
        {
            std::cerr
                << "Warning: solver returned " << nonfinite_gmn_count
                << " non-finite GMN value(s); those CSV entries were written as 0.\n";
        }
        if (clipped_negative_gmn_count > 0)
        {
            std::cerr
                << "Warning: clipped " << clipped_negative_gmn_count
                << " tiny negative GMN value(s) to 0.\n";
        }

        const GmnMosekDiagnostics mosek_diag = gmn_mosek_get_diagnostics();
        if (mosek_diag.workspace_failures > 0)
        {
            std::cerr
                << "Warning: MOSEK workspace creation failed "
                << mosek_diag.workspace_failures << " time(s). Check MOSEK_HOME, "
                << "license availability, and runtime library path.\n";
        }
        if (mosek_diag.optimize_errors > 0)
        {
            std::cerr
                << "Warning: MOSEK optimize/update returned an error "
                << mosek_diag.optimize_errors << " time(s).\n";
        }
        if (mosek_diag.nonfinite_objectives > 0)
        {
            std::cerr
                << "Warning: MOSEK returned/exposed a non-finite objective "
                << mosek_diag.nonfinite_objectives << " time(s).\n";
        }
        if (mosek_diag.accepted_nonoptimal_solutions > 0)
        {
            std::cerr
                << "Warning: accepted "
                << mosek_diag.accepted_nonoptimal_solutions
                << " finite MOSEK primal objective(s) with non-OPTIMAL "
                << "solution status. This avoids false zeroing of complex-mode "
                << "results; rerun a small batch with OMP_NUM_THREADS=1 "
                << "if you want to isolate individual MOSEK behavior.\n";
        }

        if (mode == EvaluationMode::RealOnly && max_imag_norm > 1.0e-10)
        {
            std::cerr
                << "Warning: real-only mode evaluates Re(rho). For full complex "
                << "Hermitian GMN evaluation, rerun as: "
                << argv[0] << " 1 " << input_path << ' ' << output_path << "\n";
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "gmn.exe error: " << error.what() << "\n";
        return 1;
    }
}
