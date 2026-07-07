#include "density_io.hpp"
#include "fgmn.hpp"

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

    struct FgmnJob
    {
        double fgmn = 0.0; //Predicted FGMN
        double gmn = 0.0; //Predicted GMN
        std::uint32_t realization = 0;
        std::array<double, RHO3_VALUES> rho_ri{};
    };
    

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

    void make_jobs_from_input_file(const std::string &input_path, std::vector<FgmnJob> &jobs){
        std::ifstream input_stream(input_path);
        if(!input_stream)
        {
            throw std::runtime_error("Failed to open input file: " + input_path);
        }
        std::string line;
        std::size_t job_number = 0;
        while(std::getline(input_stream, line)){
            std::istringstream line_stream(line);
            std::string line_part;
            if(!std::getline(line_stream, line_part, ',')){
                continue; // Skip empty or malformed lines
            }
            FgmnJob job;
            job.fgmn = std::stod(line_part);
            std::getline(line_stream, line_part, ',');
            job.gmn = std::stod(line_part);
            for(int i = 0; i < RHO3_VALUES; ++i){
                std::getline(line_stream, line_part, ',');
                job.rho_ri[i] = std::stod(line_part);
            }
            job.realization = static_cast<std::uint32_t>(job_number);
            ++job_number;
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

        int next_argument = 1;

        const std::string input_path =
            (argc > next_argument) ? argv[next_argument] : "../runs/random_rdm_fgmn_gmn_N_3.txt";
        const std::string output_path =
            (argc > next_argument + 1) ? argv[next_argument + 1] : "data.csv";
        if (argc > next_argument + 2)
        {
            throw std::invalid_argument("Too many command-line arguments.");
        }

        std::ofstream outfile(output_path);
        if (!outfile)
        {
            throw std::runtime_error("Could not create output CSV: " + output_path);
        }
        outfile << "realization,fgmn,fgmn_predicted,gmn,gmn_predicted\n";

#ifdef _OPENMP
        /* Avoid nested oversubscription by default: parallelism is at the
         * independent-GMN-solve level, not inside each tiny MOSEK solve.
         */
        if (omp_get_max_threads() > 1)
        {
            set_env_if_missing("GMN_MOSEK_NUM_THREADS", "1");
        }
#endif

        std::vector<FgmnJob> jobs;
        make_jobs_from_input_file(input_path, jobs);
        std::printf("Loaded %zu jobs from input file %s\n", jobs.size(), input_path.c_str());
        std::vector<double> fgmn_values(jobs.size(),0);
        std::vector<double> gmn_values(jobs.size(),0);
        
        int processed = 0;
        double max_fgmn_difference = 0.0;
        double max_gmn_difference = 0.0;
        const auto start = std::chrono::steady_clock::now();
        //auto interval_start = start;
        //std::uint64_t interval_processed = 0;

        // fgmn::GmnWorkspace fgmn_workspace(true, true);
        // fgmn::GmnWorkspace gmn_workspace(true, false);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (std::size_t i = 0; i < static_cast<std::size_t>(jobs.size()); ++i)
        {
            fgmn_values[i] =
                compute_fgmn_mosek_8x8_cpp(
                    jobs[i].rho_ri.data());
            gmn_values[i] =
                compute_gmn_mosek_complex_8x8_cpp(
                    jobs[i].rho_ri.data());
            // fgmn_values[i] = fgmn_workspace.solve(jobs[i].rho_ri.data());
            // gmn_values[i] = gmn_workspace.solve(jobs[i].rho_ri.data());
            double fgmn_diff = std::abs(fgmn_values[i]-jobs[i].fgmn);
            double gmn_diff = std::abs(gmn_values[i]-jobs[i].gmn);
            #pragma omp critical
            {
                processed++;
                max_fgmn_difference = std::max(max_fgmn_difference, fgmn_diff);
                max_gmn_difference = std::max(max_gmn_difference, gmn_diff);
                std::printf("Job %d: ",processed);
                std::printf("Found FGMN: %.6f, Expected FGMN: %.6f, Difference: %.6e | ", fgmn_values[i], jobs[i].fgmn, fgmn_diff);
                std::printf("Found GMN:  %.6f, Expected GMN:  %.6f, Difference: %.6e\n", gmn_values[i], jobs[i].gmn, gmn_diff);
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = now - start;
        //const std::chrono::duration<double> interval_elapsed = now - interval_start;
        const double average_rate =
            (elapsed.count() > 0.0) ? processed / elapsed.count() : 0.0;
        // const double recent_rate =
        //     (interval_elapsed.count() > 0.0)
        //         ? interval_processed / interval_elapsed.count()
        //         : 0.0;
        // const double eta_seconds =
        //     (average_rate > 0.0)
        //         ? (total_solves - processed) / average_rate
        //         : 0.0;

        std::printf("Maximum FGMN difference: %.6e, maximum GMN difference: %.6e\n", max_fgmn_difference, max_gmn_difference);
        std::cout << "Processed " << processed << " matrices; avg "
                    << std::fixed << std::setprecision(2)
                    << average_rate << "/s        "
                    << std::flush;


        for (std::size_t i = 0; i < jobs.size(); ++i)
        {
            const FgmnJob &job = jobs[i];
            outfile << i << ',' << fgmn_values[i] << ','
                    << job.fgmn << ',' << gmn_values[i] << ','
                    << job.gmn << '\n';
        }
        // if (interval_elapsed.count() >= 5.0)
        // {
        //     interval_start = now;
        //     interval_processed = 0;
        // }

        std::cout << "\nWrote " << output_path << ".\n" << std::flush;

        // outfile.close();
 
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "gmn.exe error: " << error.what() << "\n";
        return 1;
    }
}
