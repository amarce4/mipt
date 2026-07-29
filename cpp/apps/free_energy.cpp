#include "mipt/backend.hpp"
#include "mipt/env.hpp"
#include "mipt/free_energy.hpp"
#include "mipt/probed.hpp"
#include "mipt/types.hpp"
#include "mipt/util/pause.hpp"
#include "mipt/util/text.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using mipt::util::compact_count;
using mipt::util::compact_decimal;
using mipt::util::csv_quote;

std::string format_duration(double seconds)
{
    return mipt::util::format_duration(seconds, mipt::util::DurationStyle::Padded);
}

using namespace mipt;
using mipt::free_energy::RunningStats;

struct LinearAccumulator
{
    void add(double x, double y)
    {
        ++count;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    double slope() const
    {
        const double denominator =
            static_cast<double>(count) * sum_xx - sum_x * sum_x;
        if (count < 2 || !(denominator > 0.0))
        {
            throw std::runtime_error(
                "Not enough production points for a free-energy slope.");
        }
        return (static_cast<double>(count) * sum_xy - sum_x * sum_y) /
               denominator;
    }

    std::uint64_t count = 0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
};

struct TimeStats
{
    RunningStats cumulative_f;
    RunningStats f_over_tl;
    RunningStats half_entropy;
};

std::uint64_t splitmix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::uint64_t default_seed()
{
    std::random_device random;
    return (static_cast<std::uint64_t>(random()) << 32) ^
           static_cast<std::uint64_t>(random());
}

double uniform_01(std::mt19937_64 &rng)
{
    return std::generate_canonical<double, 53>(rng);
}

void write_stat(std::ostream &out, const RunningStats &stats)
{
    if (stats.count == 0)
    {
        out << "nan,nan,nan";
        return;
    }
    out << stats.mean << ',' << stats.sample_stddev() << ',' << stats.stderr();
}

void write_timeseries(
    const std::string &path,
    int n,
    double p,
    CircuitType type,
    int init_state,
    int requested_realizations,
    int completed_realizations,
    int equilibration_timesteps,
    const std::vector<TimeStats> &stats)
{
    const std::string temporary = path + ".tmp";
    std::ofstream csv(temporary, std::ios::out | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not open time-series CSV: " + temporary);
    }
    csv << std::setprecision(17);
    csv << "N,p,circ_type,circuit_name,init_state,requested_realizations,"
           "completed_realizations,t,t_over_L,phase,F_mean,F_stddev,F_stderr,"
           "F_over_tL_mean,F_over_tL_stddev,F_over_tL_stderr,S_half_mean,"
           "S_half_stddev,S_half_stderr,S_half_count\n";

    for (std::size_t index = 0; index < stats.size(); ++index)
    {
        const int t = static_cast<int>(index);
        const char *phase = t == 0
                                ? "initial"
                                : (t <= equilibration_timesteps
                                       ? "equilibration"
                                       : "production");
        csv << n << ','
            << p << ','
            << static_cast<int>(type) << ','
            << csv_quote(circuit_type_name(type)) << ','
            << init_state << ','
            << requested_realizations << ','
            << completed_realizations << ','
            << t << ','
            << static_cast<double>(t) / static_cast<double>(n) << ','
            << phase << ',';
        write_stat(csv, stats[index].cumulative_f);
        csv << ',';
        write_stat(csv, stats[index].f_over_tl);
        csv << ',';
        write_stat(csv, stats[index].half_entropy);
        csv << ',' << stats[index].half_entropy.count << '\n';
    }
    csv.flush();
    if (!csv)
    {
        throw std::runtime_error("Failed while writing time-series CSV: " + path);
    }
    csv.close();
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
    {
        throw std::runtime_error("Could not replace time-series CSV: " + path);
    }
}

void print_usage(const char *argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [N] [p] [realizations] [circ_type] [init_state]\n\n"
        << "  N          number of sites in the periodic 1D ring\n"
        << "  p          measurement rate in [0,1]\n"
        << "  realizations number of Monte Carlo trajectories\n"
        << "  circ_type  same convention as the other MIPT simulators:\n";
    print_circuit_type_help(std::cout, "               ");
    std::cout
        << "  init_state 0 = random product state, 1 = random Haar state\n\n"
        << "The first 4L timesteps equilibrate the trajectory; the following 24L\n"
        << "timesteps determine the free-energy density. One timestep is one even\n"
        << "or odd unitary brickwork layer followed by its measurement layer.\n"
        << "For circ_type 2, 3, or 4, every initial state has a definite randomly\n"
        << "selected global parity.\n\n"
        << "Environment controls:\n"
        << "  MIPT_FREE_ENERGY_SEED=<integer>\n"
        << "  MIPT_FREE_ENERGY_PROGRESS_MS=1000\n"
        << "  MIPT_FREE_ENERGY_HALF_ENTROPY=1\n"
        << "  MIPT_FREE_ENERGY_ENTROPY_MAX_T=<absolute timestep> (default 2L)\n"
        << "  MIPT_FREE_ENERGY_ENTROPY_STRIDE=1\n"
        << "  MIPT_FREE_ENERGY_CUDA_ENTROPY=1\n"
        << "  MIPT_FREE_ENERGY_CUDA_ENTROPY_MIN_KEPT=7\n";
}
} // namespace

int main(int argc, char *argv[])
{
    try
    {
        mipt::backend::print_backend_banner_once("free_energy.exe");
        if (argc > 1 &&
            (std::strcmp(argv[1], "--help") == 0 ||
             std::strcmp(argv[1], "-h") == 0))
        {
            print_usage(argv[0]);
            return 0;
        }
        if (argc != 6)
        {
            print_usage(argv[0]);
            return 2;
        }

        const int n = std::stoi(argv[1]);
        const double p = std::stod(argv[2]);
        const long parsed_realizations = std::stol(argv[3]);
        if (parsed_realizations < 1 || parsed_realizations > 1000000000L)
        {
            throw std::invalid_argument(
                "realizations must lie in [1,1000000000].");
        }
        const int realizations = static_cast<int>(parsed_realizations);
        const CircuitType type = parse_circuit_type(std::stoi(argv[4]));
        const int init_state = std::stoi(argv[5]);
        if (n < 2 || n >= 31)
        {
            throw std::invalid_argument(
                "N must lie in [2,30] for the dense exact-state protocol.");
        }
        validate_circuit_site_count(type, n, "N");
        if (!(p >= 0.0 && p <= 1.0))
        {
            throw std::invalid_argument("p must lie in [0,1].");
        }
        if (init_state != 0 && init_state != 1)
        {
            throw std::invalid_argument("init_state must be 0 or 1.");
        }

        const int equilibration_timesteps = 4 * n;
        const int production_timesteps = 24 * n;
        const int total_timesteps = equilibration_timesteps + production_timesteps;
        const int progress_ms = static_cast<int>(env::integer(
            "MIPT_FREE_ENERGY_PROGRESS_MS", 1000, 50, 600000L));
        const bool record_half_entropy =
            env::boolean("MIPT_FREE_ENERGY_HALF_ENTROPY", true);
        const int entropy_max_t = record_half_entropy
                                      ? static_cast<int>(env::integer(
                                            "MIPT_FREE_ENERGY_ENTROPY_MAX_T",
                                            2 * n,
                                            0,
                                            total_timesteps))
                                      : -1;
        const int entropy_stride = static_cast<int>(env::integer(
            "MIPT_FREE_ENERGY_ENTROPY_STRIDE", 1, 1, total_timesteps));
        const long configured_seed = env::integer(
            "MIPT_FREE_ENERGY_SEED", -1, -1, std::numeric_limits<long>::max());
        const std::uint64_t master_seed = configured_seed >= 0
                                              ? static_cast<std::uint64_t>(configured_seed)
                                              : default_seed();

        const std::string prefix =
            "free_energy_" + std::string(circuit_type_tag(type)) +
            "_n_" + std::to_string(n) +
            "_p_" + compact_decimal(p) +
            "_init_" + std::to_string(init_state) +
            "_reals_" +
            compact_count(static_cast<std::uint64_t>(realizations));
        const std::string samples_path = prefix + "_samples.csv";
        const std::string timeseries_path = prefix + "_timeseries.csv";

        std::ofstream samples(samples_path, std::ios::out | std::ios::trunc);
        if (!samples)
        {
            throw std::runtime_error("Could not open sample CSV: " + samples_path);
        }
        samples << std::setprecision(17);
        samples
            << "N,p,circ_type,circuit_name,init_state,trajectory,parity_sector,"
               "trajectory_seed,equil_timesteps,production_timesteps,total_timesteps,"
               "measurements_total,measurements_production,F_total,F_production,"
               "production_rate_endpoint,production_slope,free_energy_density_tilde\n";

        std::cout
            << "Measurement-record free energy: N=" << n
            << ", p=" << p
            << ", circ_type=" << static_cast<int>(type)
            << " (" << circuit_type_name(type) << ')'
            << ", init_state=" << init_state
            << (init_state == 0 ? " (product)" : " (Haar)")
            << ", realizations=" << realizations
            << ", seed=" << master_seed << '\n'
            << "t_eq=" << equilibration_timesteps
            << ", t_production=" << production_timesteps
            << ", t_total=" << total_timesteps << '\n';
        if (record_half_entropy)
        {
            std::cout << "Half-chain S1 recorded through t=" << entropy_max_t
                      << " with stride=" << entropy_stride << ".\n";
        }

        probed::CircuitWorkspace1D workspace;
        workspace.reserve(total_timesteps);
        free_energy::HalfEntropyWorkspace entropy_workspace;
        const bool fermion_trace = uses_fermionic_trace(type);
        std::vector<TimeStats> time_stats(
            static_cast<std::size_t>(total_timesteps + 1));

        const auto run_start = std::chrono::steady_clock::now();
        auto last_progress = run_start;
        double last_trajectory_rate = 0.0;

        mipt::util::PauseSentinel pause_sentinel;

        for (int trajectory = 0; trajectory < realizations; ++trajectory)
        {
            // Safe checkpoint: per-trajectory samples are already flushed.
            pause_sentinel.wait([&]() { samples.flush(); });
            const auto trajectory_start = std::chrono::steady_clock::now();
            const std::uint64_t trajectory_seed = splitmix64(
                master_seed ^ static_cast<std::uint64_t>(trajectory));
            workspace.seed(static_cast<std::uint32_t>(
                splitmix64(trajectory_seed ^ 0x4349524355495455ULL)));
            workspace.prepare(n, total_timesteps, p, type);

            std::mt19937_64 initial_rng(
                splitmix64(trajectory_seed ^ 0x494e495449414c53ULL));
            std::mt19937_64 outcome_rng(
                splitmix64(trajectory_seed ^ 0x4d45415355524553ULL));
            auto initial = free_energy::make_initial_state(
                n, init_state, type, initial_rng);
            auto state = std::move(initial.state);

            double cumulative_f = 0.0;
            double production_f = 0.0;
            std::uint64_t measurements_total = 0;
            std::uint64_t measurements_production = 0;
            LinearAccumulator production_fit;

            time_stats[0].cumulative_f.add(0.0);
            if (record_half_entropy && entropy_max_t >= 0)
            {
                time_stats[0].half_entropy.add(
                    free_energy::half_chain_entropy_bits(
                        state, n, fermion_trace, entropy_workspace));
            }

            for (int timestep = 0; timestep < total_timesteps; ++timestep)
            {
                const std::vector<int> measured_sites =
                    workspace.measurement_sites(timestep);
                workspace.advance_unitary(state, timestep, "free-energy");

                double layer_surprisal = 0.0;
#ifdef MIPT_ENABLE_CUSV
                // One joint draw for the whole layer. The returned value is
                // -log P(m_1..m_k), which the chain rule makes identical to the
                // sum of the sequential conditional surprisals below.
                if (!workspace.measure_layer_in_place(state, timestep, layer_surprisal))
#endif
                {
                    layer_surprisal = 0.0;
                    for (int site : measured_sites)
                    {
                        const auto measurement = free_energy::measure_and_collapse(
                            state, n, site, uniform_01(outcome_rng));
                        layer_surprisal += measurement.surprisal;
                    }
                }

                const int t = timestep + 1;
                cumulative_f += layer_surprisal;
                measurements_total += measured_sites.size();
                time_stats[static_cast<std::size_t>(t)].cumulative_f.add(
                    cumulative_f);
                time_stats[static_cast<std::size_t>(t)].f_over_tl.add(
                    cumulative_f /
                    (static_cast<double>(t) * static_cast<double>(n)));

                if (record_half_entropy && t <= entropy_max_t &&
                    (t % entropy_stride == 0 || t == entropy_max_t))
                {
                    time_stats[static_cast<std::size_t>(t)].half_entropy.add(
                        free_energy::half_chain_entropy_bits(
                            state, n, fermion_trace, entropy_workspace));
                }

                if (t > equilibration_timesteps)
                {
                    const int production_t = t - equilibration_timesteps;
                    production_f += layer_surprisal;
                    measurements_production += measured_sites.size();
                    production_fit.add(
                        static_cast<double>(production_t), production_f);
                }
            }

            const double production_slope = production_fit.slope();
            const double endpoint_rate =
                production_f / static_cast<double>(production_timesteps);
            const double density_tilde =
                production_slope / static_cast<double>(n);
            samples << n << ','
                    << p << ','
                    << static_cast<int>(type) << ','
                    << csv_quote(circuit_type_name(type)) << ','
                    << init_state << ','
                    << trajectory << ','
                    << initial.parity_sector << ','
                    << trajectory_seed << ','
                    << equilibration_timesteps << ','
                    << production_timesteps << ','
                    << total_timesteps << ','
                    << measurements_total << ','
                    << measurements_production << ','
                    << cumulative_f << ','
                    << production_f << ','
                    << endpoint_rate << ','
                    << production_slope << ','
                    << density_tilde << '\n';
            samples.flush();
            if (!samples)
            {
                throw std::runtime_error(
                    "Failed while writing sample CSV: " + samples_path);
            }

            write_timeseries(
                timeseries_path,
                n,
                p,
                type,
                init_state,
                realizations,
                trajectory + 1,
                equilibration_timesteps,
                time_stats);

            const auto now = std::chrono::steady_clock::now();
            const double trajectory_seconds =
                std::chrono::duration<double>(now - trajectory_start).count();
            last_trajectory_rate = trajectory_seconds > 0.0
                                       ? 1.0 / trajectory_seconds
                                       : 0.0;
            const bool report = trajectory + 1 == realizations ||
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_progress).count() >= progress_ms;
            if (report)
            {
                const double elapsed =
                    std::chrono::duration<double>(now - run_start).count();
                const double average_rate =
                    elapsed > 0.0 ? static_cast<double>(trajectory + 1) / elapsed : 0.0;
                const double remaining = average_rate > 0.0
                                             ? static_cast<double>(
                                                   realizations - trajectory - 1) /
                                                   average_rate
                                             : std::numeric_limits<double>::infinity();
                std::cout << '\r'
                          << "realizations: " << trajectory + 1 << '/'
                          << realizations
                          << " | avg=" << std::fixed << std::setprecision(3)
                          << average_rate << " traj/s"
                          << " | current=" << last_trajectory_rate << " traj/s"
                          << " | ETA=" << format_duration(remaining)
                          << std::flush;
                last_progress = now;
            }
        }
        std::cout << "\nSaved " << samples_path
                  << " and " << timeseries_path << ".\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "free_energy.exe: " << error.what() << '\n';
        return 1;
    }
}
