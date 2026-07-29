#include "mipt/density.hpp"
#include "mipt/env.hpp"
#include "mipt/analysis/fgmn.hpp"
#include "mipt/analysis/rdm_batch_cli.hpp"

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
using namespace mipt;

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
            << "  p,fgmn,bipneg\n\n"
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

        analysis::limit_nested_solver_threads();
#ifdef _OPENMP
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
        analysis::require_three_qubit_input(metadata);

        std::ofstream outfile(output_path);
        if (!outfile)
        {
            throw std::runtime_error("Could not create output CSV: " + output_path);
        }
        outfile << "p,fgmn,bipneg\n";
        outfile << std::setprecision(17);
        analysis::write_metadata_sidecar(output_path, metadata, input_path, "fgmn.exe");

        analysis::BatchOptions options;
        options.limit_per_p = limit_per_p;
        options.batch_records = parse_batch_records();

        std::cout << "Calculating three-mode fermionic GMN values from "
                  << input_path << " (" << metadata.record_count
                  << " input records, " << metadata.subsystem_count
                  << " retained subsystems each, up to "
                  << analysis::solves_to_perform(metadata, limit_per_p)
                  << " total solves";
        if (limit_per_p != 0)
        {
            std::cout << "; limit_per_p=" << limit_per_p
                      << " records/p-value";
        }
        std::cout << ").\n";
        analysis::print_worker_configuration("fGMN");

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

        double max_imag_norm = 0.0;

        const auto totals = analysis::run_rdm_batches<FgmnJob>(
            reader, outfile, options,
            [&](const mipt::io::DensityFileMetadata &file_metadata,
                const mipt::io::DensityRecord &record, std::vector<FgmnJob> &jobs) {
                append_jobs_from_record(file_metadata, record, jobs, max_imag_norm);
            },
            [&](const FgmnJob &job) {
                analysis::SolveResult result;
                const double *rho = job.rho_ri.data();

                result.bipneg = compute_min_bipartite_fermionic_negativity_8x8_cpp(rho);
                if (zero_prefilter.enabled && std::isfinite(result.bipneg) &&
                    result.bipneg <= zero_prefilter.tolerance)
                {
                    // A vanishing minimum bipartite negativity certifies fGMN=0.
                    result.value = 0.0;
                    result.prefiltered = true;
                    return result;
                }

                result.value = compute_fgmn_mosek_8x8_cpp(rho);
                return result;
            },
            [](std::ostream &out, const FgmnJob &job, const analysis::SolveResult &result) {
                out << job.p << ',' << result.value << ',' << result.bipneg << '\n';
            },
            /*show_prefilter_progress=*/true);

        analysis::report_batch_summary(output_path, totals, max_imag_norm,
                                       "Maximum imaginary Frobenius norm in evaluated records");
        if (totals.nonfinite_values != 0)
        {
            std::cerr << "Warning: solver returned " << totals.nonfinite_values
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
