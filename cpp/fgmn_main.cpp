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
    constexpr std::size_t RHO3_QUBITS = 3u;
    constexpr std::size_t RHO3_DIM = 1u << RHO3_QUBITS;
    using Subsystem = std::array<int, RHO3_QUBITS>;

    enum class EvaluationMode
    {
        RealOnly = 0,
        Complex = 1,
        Fermionic = 2
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
        if(mode == EvaluationMode::RealOnly)
        {
            return "real-only";
        }
        if(mode == EvaluationMode::Complex)
        {
            return "complex";
        }
        return "fermionic";
    }

    void print_usage(const char *program)
    {
        std::cout
            << "Usage: " << program << " [0|1] [input_file] [output_csv]\n\n"
            << "Modes:\n"
            << "  0    Real-only SDP using Re(rho). Fast legacy path.\n"
            << "  1    Complex Hermitian SDP using the full complex rho.\n"
            << "  2    Fermionic SDP using the full complex rho.\n\n"
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
            << "The former leading argument '3' has been removed. Use mode 0, 1, or 2.\n"
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
        if (value == "2")
        {
            return EvaluationMode::Fermionic;
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
        throw std::invalid_argument("First argument must be evaluation mode 0, 1, or 2.");
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

    template <typename T>
    std::vector<double> linspace(T start_in, T end_in, int num_in)
    {

        std::vector<double> linspaced;

        double start = static_cast<double>(start_in);
        double end = static_cast<double>(end_in);
        double num = static_cast<double>(num_in);

        if (num == 0)
        {
            return linspaced;
        }
        if (num == 1)
        {
            linspaced.push_back(start);
            return linspaced;
        }

        double delta = (end - start) / (num - 1);

        for (int i = 0; i < num - 1; ++i)
        {
            linspaced.push_back(start + delta * i);
        }
        linspaced.push_back(end); // I want to ensure that start and end
                                // are exactly the same as the input
        return linspaced;
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

        if(input_path == "default.bin"){//Construct default RDMs and save to file
            //std::vector<Subsystem> subsystems = {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
            std::vector<Subsystem> subsystems = {{0, 1, 2}};
            mipt_io::DensityFileMetadata metadata;
            metadata.spatial_dimension = 1u;
            metadata.total_qubits = RHO3_QUBITS;
            metadata.kept_qubits = RHO3_QUBITS;
            metadata.matrix_dimension = RHO3_DIM;
            metadata.periods = 4; // Not used in this context, should be equal to the depth of the circuit that generated the state
            metadata.realizations = 1; // Not used in this context, should be equal to the number of circuit realizations.
            metadata.resolution = 101; // Should be equal to the number of p values simulated.
            metadata.grid_x = RHO3_QUBITS;
            metadata.grid_y = 1;
            metadata.subsystem_count = static_cast<std::uint32_t>(subsystems.size());
            metadata.record_count = metadata.realizations * metadata.resolution;
            metadata.p_min = 0.0;
            metadata.p_max = 1.0; //p acts as white noise mixing in the default case
            metadata.subsystem_qubits.reserve(subsystems.size() * RHO3_QUBITS);
            for (const auto &subsystem : subsystems)
            {
                for (int q : subsystem)
                {
                    metadata.subsystem_qubits.push_back(static_cast<std::uint32_t>(q));
                }
            }
            std::vector<double> pvalues = linspace(metadata.p_min, metadata.p_max, metadata.resolution);
            mipt_io::DensityMatrixWriter writer("default.bin", metadata);
            for(std::size_t p_index = 0; p_index < metadata.resolution; ++p_index)
            {
                // Save a single (X basis) GHZ state with white noise
                double white_noise = pvalues[p_index];
                int ms_factor = 0; //If zero, make the X basis GHZ. If 1, add Molmer-Sorensen factors of -i on |011>,|110> and -1 on |101>
                std::array<double, RHO3_VALUES> rho_ri{};//= {0.24724,0.,0.00000,0.,-0.00000,0.,-0.24597,0.,-0.00000,0.,-0.24597,0.,-0.24597,0.,0.00000,0.,0.00000,0.,0.39274,0.,0.01781,0.,0.00000,0.,0.01781,0.,0.00000,0.,-0.00000,0.,0.01781,0.,-0.00000,0.,0.01781,0.,0.39274,0.,0.00000,0.,0.01781,0.,-0.00000,0.,0.00000,0.,0.01781,0.,-0.24597,0.,0.00000,0.,0.00000,0.,0.24724,0.,0.00000,0.,-0.24597,0.,-0.24597,0.,-0.00000,0.,-0.00000,0.,0.01781,0.,0.01781,0.,0.00000,0.,0.39274,0.,-0.00000,0.,-0.00000,0.,0.01781,0.,-0.24597,0.,0.00000,0.,-0.00000,0.,-0.24597,0.,-0.00000,0.,0.24724,0.,-0.24597,0.,0.00000,0.,-0.24597,0.,-0.00000,0.,0.00000,0.,-0.24597,0.,-0.00000,0.,-0.24597,0.,0.24724,0.,0.00000,0.,0.00000,0.,0.01781,0.,0.01781,0.,-0.00000,0.,0.01781,0.,0.00000,0.,0.00000,0.,0.39274,0.};
                
                for(std::size_t i = 0; i < RHO3_DIM; ++i)
                {
                    size_t index_ii = i*RHO3_DIM+i;
                    rho_ri[2u * index_ii] = white_noise/RHO3_DIM; 
                    std::size_t num_up = 0;
                    for(std::size_t b = 0; b < RHO3_QUBITS; ++b) num_up += (i >> b) & 1u;
                    if(num_up % 2 == 0){ //Means i corresponds to a |000>, |011>, |101>, or |110> state, which belong to the X-basis GHZ state. 
                        rho_ri[2u * index_ii] += (1-white_noise)/(RHO3_DIM/2);
                        for(std::size_t j = 0; j < i; j++){
                            std::size_t num_up_j = 0;
                            for(std::size_t b = 0; b < RHO3_QUBITS; ++b) num_up_j += (j >> b) & 1u;
                            if(num_up_j % 2 == 0){
                                double phase = 1;
                                if(ms_factor && i != 0) phase *= -1; //Ket has minus signs on 011,110 and 101
                                if(ms_factor && j == 5) phase *= -1; //Bra only has minus signs on 101
                                if(ms_factor && (i == 3 || i == 6) && (j == 3 || j == 6)) phase *= -1; //From i*i=-1
                                int complex = 0;
                                if(ms_factor) complex = (i == 3 || i == 6) ^ (j == 3 || j == 6); //Imaginary component if one bra/ket is |011> or |110> and the other isn't
                                size_t index_ij = i*RHO3_DIM+j;
                                rho_ri[2u * index_ij+complex] = phase*((1-white_noise)/(RHO3_DIM/2));
                                if(complex) phase *= -1; //Switch phase if entry is complex conjugated
                                size_t index_ji = j*RHO3_DIM+i;
                                rho_ri[2u * index_ji+complex] = phase*((1-white_noise)/(RHO3_DIM/2));
                            }
                        }
                    }
                }
                writer.write_record(p_index, 0, white_noise, rho_ri.data(), rho_ri.size());
            }
            writer.close();
            std::cout << "Saved default density-matrix data to default.bin.\n";
        }

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

        outfile << "p,gmn,p_index,realization,subsystem_index,q0,q1,q2\n";
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
                if (mode == EvaluationMode::Fermionic)
                {
                    gmn_values[static_cast<std::size_t>(i)] =
                        compute_fgmn_mosek_8x8_cpp(
                            jobs[static_cast<std::size_t>(i)].rho_ri.data());
                    std::printf("GMN value #%d is %2.5f\n", static_cast<int>(i), gmn_values[static_cast<std::size_t>(i)]);
                }
                else if (mode == EvaluationMode::Complex)
                {
                    gmn_values[static_cast<std::size_t>(i)] =
                        compute_gmn_mosek_complex_8x8_cpp(
                            jobs[static_cast<std::size_t>(i)].rho_ri.data());
                    std::printf("GMN value #%d is %2.5f\n", static_cast<int>(i), gmn_values[static_cast<std::size_t>(i)]);
                }
                else
                {
                    gmn_values[static_cast<std::size_t>(i)] =
                        compute_gmn_mosek_real_8x8_cpp(
                            jobs[static_cast<std::size_t>(i)].rho_real.data());
                }
            }

            std::printf("\nGMNs: [");
            for (std::size_t i = 0; i < jobs.size(); ++i)
            {
                const GmnJob &job = jobs[i];
                outfile << job.p << ',' << gmn_values[i] << ','
                        << job.p_index << ',' << job.realization << ','
                        << job.subsystem << ','
                        << job.q0 << ',' << job.q1 << ',' << job.q2 << '\n';
                std::printf("%.5f ",gmn_values[i]);
            }
            std::printf("]\n");

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

        if (mode == EvaluationMode::RealOnly && max_imag_norm > 1.0e-10)
        {
            std::cerr
                << "Warning: real-only mode evaluates Re(rho). For full complex "
                << "Hermitian GMN evaluation, rerun as: "
                << argv[0] << " 1 " << input_path << ' ' << output_path << "\n";
        }

        outfile.close();

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "gmn.exe error: " << error.what() << "\n";
        return 1;
    }
}
