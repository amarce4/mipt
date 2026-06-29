#include "density_io.hpp"
#include "gmn.hpp"

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
            << "Usage: " << program << " [0|1] [input_file] [output_csv]\n\n"
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
            << "  " << program << " 1 rho3_2d.bin data2_complex.csv\n\n"
            << "Environment variables:\n"
            << "  OMP_NUM_THREADS=N          Number of parallel GMN worker threads.\n"
            << "  GMN_MOSEK_NUM_THREADS=N    MOSEK threads per GMN solve; use 1 when OMP > 1.\n"
            << "  GMN_BATCH_RECORDS=N        Records per parallel batch; default 256.\n"
            << "  GMN_MOSEK_RECYCLE=N        Rebuild cached task every N solves/thread; default 10000, 0 disables.\n\n"
            << "The former leading argument '3' has been removed. Use mode 0 or 1.\n"
            << "Six-qubit retained-subsystem analysis is not implemented; input must contain\n"
            << "three-qubit 8x8 reduced density matrices.\n";
    }

    int parse_positive_env(const char *name, int default_value)
    {
        const char *s = std::getenv(name);
        if (s == nullptr || *s == '\0')
        {
            return default_value;
        }

        char *end = nullptr;
        errno = 0;
        const long value = std::strtol(s, &end, 10);
        if (errno != 0 || end == s || *end != '\0' || value <= 0 ||
            value > std::numeric_limits<int>::max())
        {
            return default_value;
        }
        return static_cast<int>(value);
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

    template <std::size_t N>
    bool all_finite(const std::array<double, N> &values)
    {
        return std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        });
    }

    double csv_safe_gmn(double value)
    {
        /* MOSEK can return no usable interior-point solution on numerically
         * difficult records, especially in the complex realified SDP. The CSV
         * should remain numeric and compact, so failed/non-finite solves are
         * emitted as 0 rather than NaN/Inf.
         */
        return std::isfinite(value) ? value : 0.0;
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

    void append_jobs_from_record(const mipt_io::DensityFileMetadata &metadata,
                                 const mipt_io::DensityRecord &record,
                                 std::vector<GmnJob> &jobs,
                                 double &max_imag_norm)
    {
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

            GmnJob job;
            job.p = record.p;
            job.p_index = record.p_index;
            job.realization = record.realization;
            job.subsystem = subsystem;
            job.q0 = metadata.subsystem_qubits[q_offset + 0];
            job.q1 = metadata.subsystem_qubits[q_offset + 1];
            job.q2 = metadata.subsystem_qubits[q_offset + 2];
            job.imag_norm = imag_norm;
            job.rho_real = real_matrix(rho_ri);
            job.rho_ri = complex_matrix(rho_ri);
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
        if (argc > next_argument + 2)
        {
            throw std::invalid_argument("Too many command-line arguments.");
        }

#ifdef _OPENMP
        /* Avoid nested oversubscription by default: parallelism is at the
         * independent-GMN-solve level, not inside each tiny MOSEK solve.
         */
        if (omp_get_max_threads() > 1)
        {
            set_env_if_missing("GMN_MOSEK_NUM_THREADS", "1");
        }
#endif

        mipt_io::DensityMatrixReader reader(input_path);
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

        outfile << "p,gmn\n";
        outfile << std::setprecision(17);

        const std::uint64_t total_solves =
            metadata.record_count * static_cast<std::uint64_t>(metadata.subsystem_count);
        std::cout << "Calculating three-qubit " << mode_name(mode)
                  << " GMN values from " << input_path
                  << " (" << metadata.record_count << " trajectories, "
                  << metadata.subsystem_count << " retained subsystems each, "
                  << total_solves << " total solves).\n";

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

        const int batch_records = parse_positive_env("GMN_BATCH_RECORDS", 256);
        std::vector<GmnJob> jobs;
        jobs.reserve(static_cast<std::size_t>(batch_records) * metadata.subsystem_count);
        std::vector<double> gmn_values;

        mipt_io::DensityRecord record;
        std::uint64_t processed = 0;
        double max_imag_norm = 0.0;
        std::uint64_t nonfinite_gmn_count = 0;
        const auto start = std::chrono::steady_clock::now();
        auto interval_start = start;
        std::uint64_t interval_processed = 0;

        while (processed < total_solves)
        {
            jobs.clear();

            int records_in_batch = 0;
            while (records_in_batch < batch_records && reader.read_record(record))
            {
                append_jobs_from_record(metadata, record, jobs, max_imag_norm);
                ++records_in_batch;
            }

            if (jobs.empty())
            {
                break;
            }

            gmn_values.assign(jobs.size(), std::numeric_limits<double>::quiet_NaN());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(jobs.size()); ++i)
            {
                const GmnJob &job = jobs[static_cast<std::size_t>(i)];
                if (mode == EvaluationMode::Complex)
                {
                    if (all_finite(job.rho_ri))
                    {
                        gmn_values[static_cast<std::size_t>(i)] =
                            compute_gmn_mosek_complex_8x8(job.rho_ri.data());
                    }
                }
                else
                {
                    if (all_finite(job.rho_real))
                    {
                        gmn_values[static_cast<std::size_t>(i)] =
                            compute_gmn_mosek_real_8x8(job.rho_real.data());
                    }
                }
            }

            for (std::size_t i = 0; i < jobs.size(); ++i)
            {
                const GmnJob &job = jobs[i];
                const double safe_gmn = csv_safe_gmn(gmn_values[i]);
                if (safe_gmn != gmn_values[i])
                {
                    ++nonfinite_gmn_count;
                }
                outfile << job.p << ',' << safe_gmn << '\n';
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
                      << total_solves << " matrices; avg "
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
                  << "Maximum imaginary Frobenius norm in input records: "
                  << std::scientific << std::setprecision(6)
                  << max_imag_norm << "\n";

        if (nonfinite_gmn_count > 0)
        {
            std::cerr
                << "Warning: replaced " << nonfinite_gmn_count
                << " non-finite GMN value(s) with 0 in the CSV.\n";
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
