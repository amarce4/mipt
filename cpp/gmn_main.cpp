#include "density_io.hpp"
#include "gmn.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    constexpr std::size_t RHO3_VALUES = 2u * 8u * 8u;

    void print_usage(const char *program)
    {
        std::cout
            << "Usage: " << program << " [3] [input_file] [output_csv]\n\n"
            << "Defaults:\n"
            << "  " << program << "            reads rho3.bin and writes data.csv\n"
            << "  " << program << " 3          same as above; retained for compatibility\n\n"
            << "Examples:\n"
            << "  " << program << " 3 rho3.bin data.csv\n"
            << "  " << program << " 3 rho3_2d.bin data2.csv\n\n"
            << "For a periodic 1D rho3.bin file, one GMN solve is performed for\n"
            << "each stored three-site window in each circuit trajectory.\n"
            << "Six-qubit retained-subsystem analysis is not implemented.\n\n"
            << "The saved matrices are complex. This executable presently evaluates\n"
            << "the existing real-valued GMN SDP on Re(rho), and prints the maximum\n"
            << "discarded imaginary Frobenius norm as a diagnostic.\n";
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
        if (argc > 1 && std::string(argv[1]) == "3")
        {
            next_argument = 2;
        }
        else if (argc > 1 && std::string(argv[1]) == "6")
        {
            throw std::invalid_argument(
                "Six-qubit retained-subsystem analysis has been removed; use rho3 data only.");
        }

        const std::string input_path =
            (argc > next_argument) ? argv[next_argument] : "rho3.bin";
        const std::string output_path =
            (argc > next_argument + 1) ? argv[next_argument + 1] : "data.csv";
        if (argc > next_argument + 2)
        {
            throw std::invalid_argument("Too many command-line arguments.");
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
        std::cout << "Calculating three-qubit GMN values from " << input_path
                  << " (" << metadata.record_count << " trajectories, "
                  << metadata.subsystem_count << " retained subsystems each, "
                  << total_solves << " total solves).\n";

        mipt_io::DensityRecord record;
        std::uint64_t processed = 0;
        double max_imag_norm = 0.0;
        const auto start = std::chrono::steady_clock::now();

        while (reader.read_record(record))
        {
            for (std::uint32_t subsystem = 0;
                 subsystem < metadata.subsystem_count;
                 ++subsystem)
            {
                const double *rho_ri = record.rho_ri.data() +
                    static_cast<std::size_t>(subsystem) * RHO3_VALUES;
                max_imag_norm = std::max(
                    max_imag_norm, imaginary_frobenius_norm(rho_ri));

                const std::array<double, 64> rho_real = real_matrix(rho_ri);
                const double gmn = compute_gmn_mosek_real_8x8(rho_real.data());
                const std::size_t q_offset =
                    static_cast<std::size_t>(subsystem) * metadata.kept_qubits;

                outfile << record.p << ',' << gmn << ','
                        << record.p_index << ',' << record.realization << ','
                        << subsystem << ','
                        << metadata.subsystem_qubits[q_offset + 0] << ','
                        << metadata.subsystem_qubits[q_offset + 1] << ','
                        << metadata.subsystem_qubits[q_offset + 2] << '\n';
                ++processed;
            }

            const auto now = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = now - start;
            const double rate =
                (elapsed.count() > 0.0) ? processed / elapsed.count() : 0.0;
            std::cout << "\rProcessed " << processed << "/"
                      << total_solves << " matrices; "
                      << std::fixed << std::setprecision(2)
                      << rate << " GMN solves/second, " 
                      << "ETA: " << (elapsed.count() > 0.0 ? (total_solves - processed) / rate : 0.0) << " s        "
                      << std::flush;
        }

        std::cout << "\nWrote " << output_path << ".\n"
                  << "Maximum discarded imaginary Frobenius norm: "
                  << std::scientific << std::setprecision(6)
                  << max_imag_norm << "\n";

        if (max_imag_norm > 1.0e-10)
        {
            std::cerr
                << "Warning: saved density matrices have non-negligible imaginary "
                << "components. The current GMN SDP evaluates Re(rho) only.\n";
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "gmn.exe error: " << error.what() << "\n";
        return 1;
    }
}
