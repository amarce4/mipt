#include "mipt/density.hpp"
#include "mipt/env.hpp"
#include "mipt/analysis/fgmn.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    constexpr std::size_t RHO3_VALUES = 2u * 8u * 8u;
    constexpr std::size_t RHO3_QUBITS = 3u;

    struct FgmnJob
    {
        double p = 0.0;
        std::uint32_t p_index = 0;
        std::uint32_t realization = 0;
        std::uint32_t subsystem = 0;
        std::uint32_t q0 = 0;
        std::uint32_t q1 = 0;
        std::uint32_t q2 = 0;
        double imag_norm = 0.0;
        std::array<double, RHO3_VALUES> rho_ri{};
    };

    void print_usage(const char *program)
    {
        std::cout
            << "Usage: " << program << " [input_file] [output_csv] [limit_per_p]\n\n"
            << "Defaults:\n"
            << "  " << program << "              reads rho3.bin and writes fgmn.csv with no per-p limit\n"
            << "  " << program << " rho3.bin     reads rho3.bin and writes fgmn.csv with no per-p limit\n\n"
            << "Examples:\n"
            << "  " << program << " rho3.bin fgmn.csv\n"
            << "  " << program << " rho3_2d.bin fgmn_2d.csv 100\n\n"
            << "Input format:\n"
            << "  A mipt.exe density-matrix .bin file containing three-qubit 8x8 RDMs.\n"
            << "  One record is one full-circuit realization at one p-value and contains all\n"
            << "  stored three-qubit RDMs, usually N subsystems for an N-site circuit.\n\n"
            << "limit_per_p:\n"
            << "  Maximum number of full records/realizations evaluated per p-value.\n"
            << "  Use 0, or omit it, for no per-p limit.\n\n"
            << "Output columns:\n"
            << "  p,fgmn\n\n"
            << "Environment variables:\n"
            << "  OMP_NUM_THREADS=N          Number of parallel fGMN worker threads.\n"
            << "  GMN_MOSEK_NUM_THREADS=N    MOSEK threads per fGMN solve; use 1 when OMP > 1.\n"
            << "  FGMN_BATCH_RECORDS=N       Records per parallel batch; default 256.\n"
            << "  GMN_BATCH_RECORDS=N        Fallback batch-size variable.\n"
            << "  FGMN_LIMIT_PER_P=N         Fallback per-p record limit; 0 or unset means no limit.\n"
            << "  FGMN_MOSEK_TOL=X           Optional MOSEK feasibility/gap tolerance, e.g. 1e-7.\n"
            << "  GMN_MOSEK_TOL=X            Fallback MOSEK tolerance variable.\n"
            << "  FGMN_MOSEK_OPTIMIZER=conic Optional benchmark knob; unset for MOSEK default.\n"
            << "  FGMN_MOSEK_PRESOLVE=off    Optional benchmark knob; unset for MOSEK default.\n"
            << "  FGMN_ZERO_PREFILTER=1      Skip SDP when min bipartite f-negativity is zero.\n"
            << "  FGMN_ZERO_PREFILTER_TOL=X  Prefilter zero tolerance; default 1e-10.\n"
            << "  FGMN_MAX_CONCURRENT_MOSEK=N Limit concurrent Fusion solves; 0 disables.\n"
            << "                              If unset and OMP workers >= 4, fgmn.exe sets this to 2.\n"
            << "  FGMN_DISABLE_CRASH_HANDLER=1 Disable SIGSEGV/SIGABRT diagnostic handler.\n";
    }

    int parse_batch_records()
    {
        const int from_fgmn = static_cast<int>(mipt::env::integer(
            "FGMN_BATCH_RECORDS", -1, 1, std::numeric_limits<int>::max()));
        if (from_fgmn > 0)
        {
            return from_fgmn;
        }
        return static_cast<int>(mipt::env::integer(
            "GMN_BATCH_RECORDS", 256, 1, std::numeric_limits<int>::max()));
    }

    std::uint64_t parse_nonnegative_u64(const char *text, const char *name)
    {
        if (text == nullptr || *text == '\0' || text[0] == '-')
        {
            throw std::invalid_argument(std::string(name) + " must be a non-negative integer.");
        }

        char *end = nullptr;
        errno = 0;
        const unsigned long long value = std::strtoull(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' ||
            value > std::numeric_limits<std::uint64_t>::max())
        {
            throw std::invalid_argument(std::string(name) + " must be a non-negative integer.");
        }

        return static_cast<std::uint64_t>(value);
    }

    std::uint64_t parse_limit_per_p(int argc, char *argv[])
    {
        if (argc > 3)
        {
            return parse_nonnegative_u64(argv[3], "limit_per_p");
        }

        const char *env = std::getenv("FGMN_LIMIT_PER_P");
        if (env != nullptr && *env != '\0')
        {
            return parse_nonnegative_u64(env, "FGMN_LIMIT_PER_P");
        }

        return 0;
    }

    bool parse_bool_env(const char *name, bool default_value)
    {
        const char *s = std::getenv(name);
        if (s == nullptr || *s == '\0')
        {
            return default_value;
        }

        if (mipt::env::truthy(s))
        {
            return true;
        }
        if (mipt::env::falsey(s))
        {
            return false;
        }

        std::cerr << "Warning: ignoring invalid " << name << "='" << s
                  << "'. Expected 0/1, true/false, yes/no, or on/off.\n";
        return default_value;
    }

    double parse_nonnegative_double_env(const char *name, double default_value)
    {
        const char *s = std::getenv(name);
        if (s == nullptr || *s == '\0')
        {
            return default_value;
        }

        char *end = nullptr;
        errno = 0;
        const double value = std::strtod(s, &end);
        if (errno != 0 || end == s || *end != '\0' ||
            value < 0.0 || !std::isfinite(value))
        {
            std::cerr << "Warning: ignoring invalid " << name << "='" << s
                      << "'. Expected a non-negative finite number.\n";
            return default_value;
        }
        return value;
    }

    struct ZeroPrefilterSettings
    {
        bool enabled = false;
        double tolerance = 1.0e-10;
    };

    ZeroPrefilterSettings parse_zero_prefilter_settings()
    {
        ZeroPrefilterSettings settings;
        settings.enabled = parse_bool_env("FGMN_ZERO_PREFILTER", false);
        settings.tolerance = parse_nonnegative_double_env(
            "FGMN_ZERO_PREFILTER_TOL",
            1.0e-10);
        return settings;
    }


#if !defined(_WIN32)
    void crash_signal_handler(int signal_number)
    {
        const char msg[] =
            "\nfgmn.exe fatal native crash: received SIGSEGV/SIGABRT/SIGBUS/SIGILL/SIGFPE. "
            "This is usually a MOSEK/Fusion native crash, not a C++ exception. "
            "Retry with OMP_NUM_THREADS=2, FGMN_MAX_CONCURRENT_MOSEK=1, "
            "and unset FGMN_MOSEK_PRESOLVE.\n";
        (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
        _Exit(128 + signal_number);
    }
#endif

    void install_crash_signal_handlers()
    {
        if (parse_bool_env("FGMN_DISABLE_CRASH_HANDLER", false))
        {
            return;
        }
#if !defined(_WIN32)
        std::signal(SIGSEGV, crash_signal_handler);
        std::signal(SIGABRT, crash_signal_handler);
        std::signal(SIGBUS, crash_signal_handler);
        std::signal(SIGILL, crash_signal_handler);
        std::signal(SIGFPE, crash_signal_handler);
#endif
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

    std::array<double, RHO3_VALUES> copy_complex_matrix(const double *rho_ri)
    {
        std::array<double, RHO3_VALUES> out{};
        std::copy(rho_ri, rho_ri + RHO3_VALUES, out.begin());
        return out;
    }

    void append_jobs_from_record(const mipt::io::DensityFileMetadata &metadata,
                                 const mipt::io::DensityRecord &record,
                                 std::vector<FgmnJob> &jobs,
                                 double &max_imag_norm)
    {
        if (metadata.subsystem_qubits.size() <
            static_cast<std::size_t>(metadata.subsystem_count) * RHO3_QUBITS)
        {
            throw std::runtime_error("Input metadata does not contain enough subsystem qubit indices.");
        }

        for (std::uint32_t subsystem = 0;
             subsystem < metadata.subsystem_count;
             ++subsystem)
        {
            const double *rho_ri = record.rho_ri.data() +
                static_cast<std::size_t>(subsystem) * RHO3_VALUES;
            const double imag_norm = imaginary_frobenius_norm(rho_ri);
            max_imag_norm = std::max(max_imag_norm, imag_norm);

            const std::size_t q_offset =
                static_cast<std::size_t>(subsystem) * metadata.kept_qubits;

            FgmnJob job;
            job.p = record.p;
            job.p_index = record.p_index;
            job.realization = record.realization;
            job.subsystem = subsystem;
            job.q0 = metadata.subsystem_qubits[q_offset + 0];
            job.q1 = metadata.subsystem_qubits[q_offset + 1];
            job.q2 = metadata.subsystem_qubits[q_offset + 2];
            job.imag_norm = imag_norm;
            job.rho_ri = copy_complex_matrix(rho_ri);
            jobs.push_back(job);
        }
    }
}

int main(int argc, char *argv[])
{
    try
    {
        install_crash_signal_handlers();

        if (argc > 1 &&
            (std::string(argv[1]) == "--help" ||
             std::string(argv[1]) == "-h"))
        {
            print_usage(argv[0]);
            return 0;
        }

        const std::string input_path = (argc > 1) ? argv[1] : "rho3.bin";
        const std::string output_path = (argc > 2) ? argv[2] : "fgmn.csv";
        if (argc > 4)
        {
            throw std::invalid_argument("Too many command-line arguments. Use --help for usage.");
        }
        const std::uint64_t limit_per_p = parse_limit_per_p(argc, argv);
        const ZeroPrefilterSettings zero_prefilter = parse_zero_prefilter_settings();

#ifdef _OPENMP
        // Parallelism is at the independent-SDP level.  Avoid nested thread
        // oversubscription inside each small MOSEK solve unless the user
        // explicitly requested a different MOSEK thread count.
        if (omp_get_max_threads() > 1)
        {
            mipt::env::set_if_unset("GMN_MOSEK_NUM_THREADS", "1");
        }
        // Empirically, MOSEK Fusion 11.2 can segfault under some 4+ OpenMP
        // concurrent in-process workloads even when numThreads=1 per solve.
        // Cap concurrent Fusion calls by default at 2 for OMP>=4.  Set
        // FGMN_MAX_CONCURRENT_MOSEK=0 to explicitly restore the old behavior,
        // or =1 for maximum stability.
        if (omp_get_max_threads() >= 4 &&
            !mipt::env::has_value("FGMN_MAX_CONCURRENT_MOSEK") &&
            !mipt::env::has_value("FGMN_MAX_CONCURRENT_FUSION"))
        {
            mipt::env::set_if_unset("FGMN_MAX_CONCURRENT_MOSEK", "2");
        }
#endif

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
        outfile << "p,fgmn\n";
        outfile << std::setprecision(17);

        const std::uint64_t max_records_evaluated = [&]() -> std::uint64_t {
            if (limit_per_p == 0)
            {
                return metadata.record_count;
            }
            const std::uint64_t resolution =
                static_cast<std::uint64_t>(metadata.resolution);
            const std::uint64_t limit_total =
                (resolution != 0 &&
                 limit_per_p > std::numeric_limits<std::uint64_t>::max() / resolution)
                    ? std::numeric_limits<std::uint64_t>::max()
                    : limit_per_p * resolution;
            return std::min<std::uint64_t>(metadata.record_count, limit_total);
        }();
        const std::uint64_t total_solves =
            max_records_evaluated * static_cast<std::uint64_t>(metadata.subsystem_count);

        std::cout << "Calculating three-mode fermionic GMN values from "
                  << input_path << " (" << metadata.record_count
                  << " input records, " << metadata.subsystem_count
                  << " retained subsystems each, up to " << total_solves
                  << " total solves";
        if (limit_per_p != 0)
        {
            std::cout << "; limit_per_p=" << limit_per_p
                      << " records/p-value";
        }
        std::cout << ").\n";

#ifdef _OPENMP
        std::cout << "OpenMP fGMN workers: " << omp_get_max_threads()
                  << "; GMN_MOSEK_NUM_THREADS="
                  << (std::getenv("GMN_MOSEK_NUM_THREADS")
                          ? std::getenv("GMN_MOSEK_NUM_THREADS")
                          : "MOSEK default")
                  << ".\n";
#else
        std::cout << "OpenMP is not enabled in this build; fGMN solves are serial.\n";
#endif

        const int fusion_limit = fgmn::configured_fusion_concurrency_limit();
        std::cout << "MOSEK tolerance override: "
                  << (std::getenv("FGMN_MOSEK_TOL")
                          ? std::getenv("FGMN_MOSEK_TOL")
                          : (std::getenv("GMN_MOSEK_TOL")
                                 ? std::getenv("GMN_MOSEK_TOL")
                                 : "MOSEK default"))
                  << "; Fusion concurrency cap: "
                  << (fusion_limit > 0 ? std::to_string(fusion_limit) : std::string("unlimited"))
                  << "; zero prefilter: "
                  << (zero_prefilter.enabled ? "on" : "off")
                  << " (tol=" << std::scientific << std::setprecision(1)
                  << zero_prefilter.tolerance << std::defaultfloat << ").\n";

        const int batch_records = parse_batch_records();
        std::vector<FgmnJob> jobs;
        jobs.reserve(static_cast<std::size_t>(batch_records) * metadata.subsystem_count);
        std::vector<double> fgmn_values;
        std::vector<unsigned char> prefilter_skipped_flags;

        mipt::io::DensityRecord record;
        std::vector<std::uint64_t> accepted_records_per_p(metadata.resolution, 0);
        std::uint64_t records_scanned = 0;
        std::uint64_t records_evaluated = 0;
        std::uint64_t records_skipped_by_limit = 0;
        bool reader_exhausted = false;
        std::uint64_t processed = 0;
        std::uint64_t nonfinite_count = 0;
        std::uint64_t zero_prefilter_skipped = 0;
        double max_imag_norm = 0.0;
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
                    throw std::runtime_error("Input record has p_index outside metadata.resolution.");
                }

                if (limit_per_p != 0 &&
                    accepted_records_per_p[record.p_index] >= limit_per_p)
                {
                    ++records_skipped_by_limit;
                    continue;
                }

                ++accepted_records_per_p[record.p_index];
                ++records_evaluated;
                append_jobs_from_record(metadata, record, jobs, max_imag_norm);
            }

            if (jobs.empty())
            {
                if (!reader_exhausted)
                {
                    std::cout << "\rScanned " << records_scanned
                              << " records; skipped " << records_skipped_by_limit
                              << " by limit_per_p; waiting for next accepted p-record...        "
                              << std::flush;
                    continue;
                }
                break;
            }

            fgmn_values.assign(jobs.size(), std::numeric_limits<double>::quiet_NaN());
            prefilter_skipped_flags.assign(jobs.size(), 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(jobs.size()); ++i)
            {
                const std::size_t idx = static_cast<std::size_t>(i);
                const double *rho = jobs[idx].rho_ri.data();

                if (zero_prefilter.enabled)
                {
                    const double min_bipartite_fneg =
                        compute_min_bipartite_fermionic_negativity_8x8_cpp(rho);
                    if (std::isfinite(min_bipartite_fneg) &&
                        min_bipartite_fneg <= zero_prefilter.tolerance)
                    {
                        fgmn_values[idx] = 0.0;
                        prefilter_skipped_flags[idx] = 1;
                        continue;
                    }
                }

                fgmn_values[idx] = compute_fgmn_mosek_8x8_cpp(rho);
            }

            for (std::size_t i = 0; i < jobs.size(); ++i)
            {
                const FgmnJob &job = jobs[i];
                const double value = fgmn_values[i];
                if (prefilter_skipped_flags[i] != 0)
                {
                    ++zero_prefilter_skipped;
                }
                if (!std::isfinite(value))
                {
                    ++nonfinite_count;
                }
                outfile << job.p << ',' << value << '\n';
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
            if (zero_prefilter_skipped != 0)
            {
                std::cout << "; zero-prefilter " << zero_prefilter_skipped;
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
                  << "Maximum imaginary Frobenius norm in evaluated records: "
                  << std::scientific << std::setprecision(6)
                  << max_imag_norm << "\n";
        if (nonfinite_count != 0)
        {
            std::cerr << "Warning: solver returned " << nonfinite_count
                      << " non-finite fGMN value(s).\n";
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "fgmn.exe error: " << error.what() << "\n";
        return 1;
    }
}
