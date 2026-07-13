// Translation-averaged second-Renyi entropy scaling for 1D MIPT trajectories.

#include "mipt/entropy.hpp"

#include <cstring>
#include <exception>
#include <iostream>
#include <string>

using namespace mipt;
using namespace mipt::entropy;

int main(int argc, char *argv[])
{
    try
    {
        mipt::backend::print_backend_banner_once("entropy.exe");

        if (argc > 1 &&
            (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0))
        {
            print_usage(argv[0]);
            return 0;
        }
        if (argc != 7)
        {
            print_usage(argv[0]);
            return 2;
        }

        const int n = std::stoi(argv[1]);
        const int periods = std::stoi(argv[2]);
        const double p = std::stod(argv[3]);
        const int realizations = std::stoi(argv[4]);
        const CircuitType circ_type = parse_circuit_type(std::stoi(argv[5]));
        const std::string output_csv = argv[6];

        if (n < 4)
        {
            throw std::invalid_argument("N must be at least 4 for the requested cyclic entropy scan.");
        }
        if (n >= 63)
        {
            throw std::invalid_argument("N must be less than 63 for exact basis indexing.");
        }
        if (periods < 0)
        {
            throw std::invalid_argument("periods must be non-negative.");
        }
        if (p < 0.0 || p > 1.0)
        {
            throw std::invalid_argument("p must be in [0,1].");
        }
        if (realizations <= 0)
        {
            throw std::invalid_argument("realizations must be positive.");
        }
        validate_circuit_site_count(circ_type, n, "N");

        if (mipt::native_mps::enabled())
        {
            std::cerr
                << "Warning: MIPT_NATIVE_MPS=1 is not used by entropy.exe because arbitrary-block "
                   "second-Renyi entropy requires an explicit trajectory state. The selected CUDA-Q "
                   "get_state backend is used instead.\n";
        }

        const bool fermion_trace = uses_fermionic_trace(circ_type);
        const int max_small_block = n / 2;
        std::vector<RunningStats> stats(static_cast<std::size_t>(n + 1));
        CircuitWorkspace1D workspace;
        workspace.reserve(periods);

        std::cout << "Running second-Renyi chord-length scan: N=" << n
                  << ", periods=" << periods
                  << ", p=" << p
                  << ", realizations=" << realizations
                  << ", circ_type=" << static_cast<int>(circ_type)
                  << " (" << circuit_type_name(circ_type) << ')'
                  << ", fermionic_trace=" << (fermion_trace ? 1 : 0)
                  << "\n";
        std::cout << "Averaging each L_A over " << n
                  << " contiguous cyclic translations per traj.\n";

        double seconds_per_trajectory = 0.0;
        constexpr std::size_t recent_rate_window = 10;
        std::deque<double> recent_trajectory_seconds;
        double recent_seconds_sum = 0.0;
        const auto total_start = std::chrono::steady_clock::now();

        for (int r = 0; r < realizations; ++r)
        {
            const auto trajectory_start = std::chrono::steady_clock::now();
            const auto circuit_start = trajectory_start;
            auto state = workspace.simulate(n, periods, p, circ_type, "entropy");
            const auto circuit_end = std::chrono::steady_clock::now();

            constexpr int min_kept = 2;
            std::vector<double> trajectory_means;
            const bool used_cuda_entropy = cuda_hierarchical_translation_means(
                state, n, min_kept, max_small_block, fermion_trace,
                trajectory_means);
            if (!used_cuda_entropy)
            {
                const std::vector<Complex> psi =
                    cudaq_state_to_host_complex128(state, n);
                trajectory_means = host_hierarchical_translation_means(
                    psi, n, min_kept, max_small_block, fermion_trace);
            }
            const auto entropy_end = std::chrono::steady_clock::now();

            for (int kept = min_kept; kept <= max_small_block; ++kept)
            {
                stats[static_cast<std::size_t>(kept)].add(
                    trajectory_means[static_cast<std::size_t>(kept - min_kept)]);
            }

            if (r == 0)
            {
                const double circuit_seconds =
                    std::chrono::duration<double>(circuit_end - circuit_start).count();
                const double entropy_seconds =
                    std::chrono::duration<double>(entropy_end - circuit_end).count();
                std::cout << "Entropy backend: "
                          << (used_cuda_entropy
                                  ? "CUDA half-chain RDM + recursive GPU partial traces"
                                  : "host half-chain RDM + recursive host partial traces")
                          << "\n1st traj. timing: circuit=" << circuit_seconds
                          << "s, entropy=" << entropy_seconds << "s\n";
            }

            // Keep a valid checkpoint after every completed trajectory. The CSV
            // is small (O(N) rows), so this is negligible beside state evolution
            // and reduced-density construction.
            write_results(output_csv, n, periods, p, realizations, circ_type, stats);

            const auto trajectory_end = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(trajectory_end - trajectory_start).count();
            seconds_per_trajectory =
                (r == 0)
                    ? elapsed
                    : (seconds_per_trajectory * static_cast<double>(r) + elapsed) /
                          static_cast<double>(r + 1);

            recent_trajectory_seconds.push_back(elapsed);
            recent_seconds_sum += elapsed;
            if (recent_trajectory_seconds.size() > recent_rate_window)
            {
                recent_seconds_sum -= recent_trajectory_seconds.front();
                recent_trajectory_seconds.pop_front();
            }

            const int remaining = realizations - r - 1;
            const double avg_rate =
                (seconds_per_trajectory > 0.0) ? 1.0 / seconds_per_trajectory : 0.0;
            const double recent_rate =
                (recent_seconds_sum > 0.0)
                    ? static_cast<double>(recent_trajectory_seconds.size()) / recent_seconds_sum
                    : 0.0;
            const double recent_seconds_per_trajectory =
                recent_trajectory_seconds.empty()
                    ? seconds_per_trajectory
                    : recent_seconds_sum /
                          static_cast<double>(recent_trajectory_seconds.size());

            std::cout << "\r" << (r + 1) << '/' << realizations
                      << " traj. | recent(" << recent_trajectory_seconds.size() << ") "
                      << std::round(recent_rate * 100.0) / 100.0
                      << " traj/s"
                      << " | avg. "
                      << std::round(avg_rate * 100.0) / 100.0
                      << " traj/s | ETA "
                      << static_cast<int>(remaining * recent_seconds_per_trajectory)
                      << "s          " << std::flush;
        }

        const auto total_end = std::chrono::steady_clock::now();
        const double total_seconds =
            std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "\nSaved second-Renyi scaling data to " << output_csv
                  << ". Elapsed: " << total_seconds << "s\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "entropy.exe error: " << e.what() << '\n';
        return 1;
    }
}
