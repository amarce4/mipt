// Translation-averaged second-Renyi entropy scaling for 1D MIPT trajectories.

#include "mipt/entropy.hpp"
#include "mipt/util/pause.hpp"

#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

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
                   "entropies require an explicit trajectory state. The selected CUDA-Q "
                   "get_state backend is used instead.\n";
        }

        const bool fermion_trace = uses_fermionic_trace(circ_type);
        const int max_small_block = n / 2;
        constexpr int min_kept = 2;
        // 0 disables the von Neumann path entirely; otherwise it is the largest
        // L_A whose spectrum is computed.
        const int max_s1_block =
            von_neumann_enabled() ? von_neumann_max_block(max_small_block) : 0;

        // --- resume ---------------------------------------------------------
        //
        // The output CSV is the checkpoint. Read it before anything is opened
        // for writing, so a refusal cannot have destroyed what it refused.
        resume::RunIdentity identity;
        identity.n = n;
        identity.periods = periods;
        identity.p = p;
        identity.requested_realizations = realizations;
        identity.circ_type = static_cast<int>(circ_type);

        resume::Checkpoint checkpoint;
        const bool resume_enabled = mipt::env::boolean("MIPT_ENTROPY_RESUME", true);
        if (resume_enabled && std::filesystem::exists(output_csv))
        {
            checkpoint = resume::load(output_csv, identity, max_s1_block);
        }
        const int start_realization = static_cast<int>(checkpoint.completed);
        std::vector<RunningStats> s1_stats =
            checkpoint.completed > 0
                ? std::move(checkpoint.s1)
                : std::vector<RunningStats>(static_cast<std::size_t>(n + 1));
        std::vector<RunningStats> s2_stats =
            checkpoint.completed > 0
                ? std::move(checkpoint.s2)
                : std::vector<RunningStats>(static_cast<std::size_t>(n + 1));
        if (start_realization > 0)
        {
            std::cout << "Resuming from " << output_csv << ": "
                      << start_realization << '/' << realizations
                      << " trajectories already averaged in.\n";
            if (start_realization >= realizations)
            {
                std::cout << output_csv << " is already complete. Nothing to do.\n";
                return 0;
            }
        }

        CircuitWorkspace1D workspace;
        workspace.reserve(periods);

        std::cout << "Running entanglement-entropy chord-length scan: N=" << n
                  << ", periods=" << periods
                  << ", p=" << p
                  << ", realizations=" << realizations
                  << ", circ_type=" << static_cast<int>(circ_type)
                  << " (" << circuit_type_name(circ_type) << ')'
                  << ", fermionic_trace=" << (fermion_trace ? 1 : 0)
                  << "\n";
        std::cout << "Averaging each L_A over " << n
                  << " contiguous cyclic translations per traj.\n";
        std::cout << "Entropies: S2 for L_A=" << min_kept << ".." << max_small_block
                  << "; S1 ";
        if (max_s1_block < min_kept)
        {
            std::cout << "disabled (MIPT_ENTROPY_S1=0 or MIPT_ENTROPY_S1_MAX_LA below "
                      << min_kept << ")\n";
        }
        else
        {
            std::cout << "for L_A=" << min_kept << ".." << max_s1_block;
            if (max_s1_block < max_small_block)
            {
                std::cout << " (larger blocks report NaN)";
            }
            std::cout << '\n';
        }

        double seconds_per_trajectory = 0.0;
        constexpr std::size_t recent_rate_window = 10;
        std::deque<double> recent_trajectory_seconds;
        double recent_seconds_sum = 0.0;
        const auto total_start = std::chrono::steady_clock::now();

        mipt::util::PauseSentinel pause_sentinel;

        for (int r = start_realization; r < realizations; ++r)
        {
            // Safe checkpoint: the CSV is rewritten after every trajectory.
            pause_sentinel.wait();
            const auto trajectory_start = std::chrono::steady_clock::now();
            const auto circuit_start = trajectory_start;
            auto state = workspace.simulate(n, periods, p, circ_type, "entropy");
            const auto circuit_end = std::chrono::steady_clock::now();

            std::vector<double> s1_means;
            std::vector<double> s2_means;
            const bool used_cuda_entropy = cuda_hierarchical_translation_entropies(
                state, n, min_kept, max_small_block, fermion_trace, max_s1_block,
                s1_means, s2_means);
            if (!used_cuda_entropy)
            {
                const std::vector<Complex> psi =
                    cudaq_state_to_host_complex128(state, n);
                host_hierarchical_translation_entropies(
                    psi, n, min_kept, max_small_block, fermion_trace, max_s1_block,
                    s1_means, s2_means);
            }
            const auto entropy_end = std::chrono::steady_clock::now();

            for (int kept = min_kept; kept <= max_small_block; ++kept)
            {
                const std::size_t index = static_cast<std::size_t>(kept - min_kept);
                s2_stats[static_cast<std::size_t>(kept)].add(s2_means[index]);
                if (std::isfinite(s1_means[index]))
                {
                    s1_stats[static_cast<std::size_t>(kept)].add(s1_means[index]);
                }
            }

            if (r == start_realization)
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
            write_results(output_csv, n, periods, p, realizations, circ_type,
                          s1_stats, s2_stats);

            const auto trajectory_end = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(trajectory_end - trajectory_start).count();
            // Rates and the ETA describe this session, so they average over the
            // trajectories run since the resume point rather than since zero.
            const int done_here = r - start_realization;
            seconds_per_trajectory =
                (done_here == 0)
                    ? elapsed
                    : (seconds_per_trajectory * static_cast<double>(done_here) + elapsed) /
                          static_cast<double>(done_here + 1);

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
        std::cout << "\nSaved entropy scaling data to " << output_csv
                  << ". Elapsed: " << total_seconds << "s\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "entropy.exe error: " << e.what() << '\n';
        return 1;
    }
}
