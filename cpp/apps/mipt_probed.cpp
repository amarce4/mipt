#include "mipt/ancilla_rdm.hpp"
#include "mipt/backend.hpp"
#include "mipt/env.hpp"
#include "mipt/probed.hpp"
#include "mipt/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using namespace mipt;

struct RunningStats
{
    void add(double value)
    {
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = value - mean;
        m2 += delta * delta2;
    }

    double stderr() const
    {
        if (count < 2)
        {
            return 0.0;
        }
        const double sample_variance = m2 / static_cast<double>(count - 1);
        return std::sqrt(std::max(0.0, sample_variance) /
                         static_cast<double>(count));
    }

    std::uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
};

std::string compact_decimal(double value)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(12) << value;
    std::string text = out.str();
    while (!text.empty() && text.back() == '0')
    {
        text.pop_back();
    }
    if (!text.empty() && text.back() == '.')
    {
        text.pop_back();
    }
    if (text == "-0")
    {
        text = "0";
    }
    return text;
}

std::string compact_count(std::uint64_t count)
{
    if (count >= 1000000 && count % 1000000 == 0)
    {
        return std::to_string(count / 1000000) + "m";
    }
    if (count >= 1000 && count % 1000 == 0)
    {
        return std::to_string(count / 1000) + "k";
    }
    return std::to_string(count);
}

std::string default_output_name(int n,
                                double p,
                                int distance,
                                int realizations,
                                int t_min,
                                int t_max,
                                CircuitType type)
{
    return "probed_" + std::string(circuit_type_tag(type)) +
           "_n_" + std::to_string(n) +
           "_p_" + compact_decimal(p) +
           "_d_" + std::to_string(distance) +
           "_reals_" + compact_count(static_cast<std::uint64_t>(realizations)) +
           "_t_" + std::to_string(t_min) + "_" + std::to_string(t_max) +
           ".csv";
}

double entropy_from_ancilla_rdm(double rho00,
                                double rho11,
                                std::complex<double> rho01)
{
    const double trace = rho00 + rho11;
    if (!(trace > 0.0) || !std::isfinite(trace))
    {
        throw std::runtime_error("Ancilla reduced density matrix has invalid trace.");
    }

    rho00 /= trace;
    rho11 /= trace;
    rho01 /= trace;

    const double discriminant = std::clamp(
        std::sqrt(std::max(0.0,
            (rho00 - rho11) * (rho00 - rho11) +
                4.0 * std::norm(rho01))),
        0.0, 1.0);
    const double lambda0 = 0.5 * (1.0 + discriminant);
    const double lambda1 = 0.5 * (1.0 - discriminant);

    auto entropy_term = [](double lambda)
    {
        return lambda > 0.0 ? -lambda * std::log(lambda) : 0.0;
    };
    return entropy_term(lambda0) + entropy_term(lambda1);
}

double ancilla_entropy(cudaq::state &state,
                        int system_qubits,
                        CircuitType type)
{
    const auto rho = ancilla::reduced_density_matrix(
        state,
        system_qubits + 1,
        system_qubits,
        uses_fermionic_trace(type));
    return entropy_from_ancilla_rdm(rho.rho00, rho.rho11, rho.rho01);
}

class ProgressReporter
{
  public:
    ProgressReporter(std::uint64_t total_layers,
                     std::chrono::steady_clock::time_point start)
        : total_layers_(total_layers),
          start_(start),
          enabled_(env::boolean("MIPT_PROBED_PROGRESS", true)),
          interval_(env::integer(
              "MIPT_PROBED_PROGRESS_MS", 500, 0, 60000))
    {
        last_print_ = start_ - interval_;
    }

    long interval_ms() const
    {
        return interval_.count();
    }

    void update(int realization,
                int realizations,
                int timestep,
                std::uint64_t completed_layers,
                bool force = false)
    {
        if (!enabled_)
        {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (!force && now - last_print_ < interval_)
        {
            return;
        }
        last_print_ = now;
        printed_ = true;

        const double elapsed =
            std::chrono::duration<double>(now - start_).count();
        const double rate = elapsed > 0.0
            ? static_cast<double>(completed_layers) / elapsed
            : 0.0;
        const std::uint64_t remaining = completed_layers < total_layers_
            ? total_layers_ - completed_layers
            : 0;
        const double eta = rate > 0.0
            ? static_cast<double>(remaining) / rate
            : 0.0;

        std::cout << "\rrealization=" << realization + 1 << '/' << realizations
                  << ", t=" << timestep
                  << ", layers/s=" << std::fixed << std::setprecision(2) << rate
                  << ", ETA=" << static_cast<std::uint64_t>(eta)
                  << "s                    " << std::flush;
    }

    void finish() const
    {
        if (enabled_ && printed_)
        {
            std::cout << '\n';
        }
    }

  private:
    std::uint64_t total_layers_ = 0;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_print_;
    bool enabled_ = true;
    bool printed_ = false;
    std::chrono::milliseconds interval_{500};
};

void print_help(const char *program)
{
    std::cout << "Usage: " << program
              << " [N] [p] [d] [realizations] [t_min] [t_max] [circ_type] [output.csv]\n\n"
              << "Arguments:\n"
              << "  N            Number of MIPT system sites, excluding the ancilla (default 16).\n"
              << "  p            Measurement probability per system site and timestep (default 0.33).\n"
              << "  d            Probe depth from the appended ancilla: probe_site = N-d (default 1).\n"
              << "  realizations Independent trajectories sampled at every requested t (default 10000).\n"
              << "  t_min        First integer timestep written, inclusive (default 0).\n"
              << "  t_max        Last integer timestep written, inclusive (default 50).\n"
              << "  circ_type    Circuit architecture:\n";
    print_circuit_type_help(std::cout, "                 ");
    std::cout << "  output.csv   Optional output path. The default name encodes all parameters.\n\n"
              << "One timestep is one brickwork unitary layer followed by its measurement layer.\n"
              << "Each realization is evolved once through t_max; S is sampled from the same\n"
              << "post-measurement trajectory at every t in [t_min,t_max]. get_state state\n"
              << "continuation does not add a quantum measurement.\n"
              << "The system occupies q_0,...,q_{N-1}; the untouched ancilla is q_N.\n"
              << "The initial Bell pair is (|00>+|11>)/sqrt(2) on q_{N-d},q_N.\n"
              << "S is the von Neumann entropy in nats, so S(t=0)=ln(2).\n\n"
              << "Runtime controls:\n"
              << "  MIPT_PROBED_PREFETCH=0       Disable CPU preparation overlap (default on).\n"
              << "  MIPT_PROBED_PROGRESS=0       Disable progress output.\n"
              << "  MIPT_PROBED_PROGRESS_MS=500  Minimum progress-update interval.\n";
}

int run_cli(int argc, char *argv[])
{
    backend::print_backend_banner_once("mipt_probed.exe");

    if (argc > 1 &&
        (std::strcmp(argv[1], "--help") == 0 ||
         std::strcmp(argv[1], "-h") == 0))
    {
        print_help(argv[0]);
        return 0;
    }
    if (argc > 9)
    {
        throw std::invalid_argument("Too many arguments. Use --help for usage.");
    }

    const int n = argc > 1 ? std::stoi(argv[1]) : 16;
    const double p = argc > 2 ? std::stod(argv[2]) : 0.33;
    const int distance = argc > 3 ? std::stoi(argv[3]) : 1;
    const int realizations = argc > 4 ? std::stoi(argv[4]) : 10000;
    const int t_min = argc > 5 ? std::stoi(argv[5]) : 0;
    const int t_max = argc > 6 ? std::stoi(argv[6]) : 50;
    const CircuitType type = parse_circuit_type(argc > 7 ? std::stoi(argv[7]) : 2);
    const std::string output = argc > 8
        ? std::string(argv[8])
        : default_output_name(n, p, distance, realizations, t_min, t_max, type);

    if (n <= 0)
    {
        throw std::invalid_argument("N must be positive.");
    }
    validate_circuit_site_count(type, n, "N");
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("p must lie in [0,1].");
    }
    if (distance < 1 || distance > n)
    {
        throw std::invalid_argument("d must satisfy 1 <= d <= N.");
    }
    if (realizations <= 0)
    {
        throw std::invalid_argument("realizations must be positive.");
    }
    if (t_min < 0 || t_max < t_min)
    {
        throw std::invalid_argument("Require 0 <= t_min <= t_max.");
    }
    if (output.empty())
    {
        throw std::invalid_argument("output.csv must not be empty.");
    }
    if (env::boolean("MIPT_NATIVE_MPS", false) ||
        env::boolean("MIPT_NATIVE_TEBD", false))
    {
        throw std::invalid_argument(
            "MIPT_NATIVE_MPS=1 and MIPT_NATIVE_TEBD=1 are not supported by mipt_probed.exe; unset them so the explicit ancilla is evolved by CUDA-Q.");
    }

    const int probe_site = n - distance;
    const int row_count = t_max - t_min + 1;
    std::vector<RunningStats> stats(static_cast<std::size_t>(row_count));

    const bool prefetch_enabled =
        env::boolean("MIPT_PROBED_PREFETCH", true) &&
        realizations > 1 && t_max > 0;

    const std::uint64_t total_layers =
        static_cast<std::uint64_t>(realizations) *
        static_cast<std::uint64_t>(t_max);
    std::uint64_t completed_layers = 0;
    const auto overall_start = std::chrono::steady_clock::now();
    ProgressReporter progress(total_layers, overall_start);

    std::cout << "Probed MIPT simulation\n"
              << "system_N=" << n
              << ", total_qubits=" << n + 1
              << ", p=" << p
              << ", d=" << distance
              << ", probe_site=q_" << probe_site
              << ", ancilla=q_" << n
              << ", realizations=" << realizations
              << ", t_range=[" << t_min << ',' << t_max << ']'
              << ", circ_type=" << static_cast<int>(type)
              << " (" << circuit_type_name(type) << ")"
              << ", fermionic_trace="
              << (uses_fermionic_trace(type) ? "on" : "off")
              << ", cpu_prefetch=" << (prefetch_enabled ? "on" : "off")
              << ", progress_ms=" << progress.interval_ms()
              << ", output=" << output << '\n';

    std::array<probed::CircuitWorkspace1D, 2> workspaces;
    for (auto &workspace : workspaces)
    {
        workspace.reserve(t_max);
    }
    std::array<std::future<void>, 2> preparation_jobs;

    if (t_max > 0)
    {
        workspaces[0].prepare(n, t_max, p, type);
    }

    for (int realization = 0; realization < realizations; ++realization)
    {
        if (t_min == 0)
        {
            stats[0].add(std::log(2.0));
        }
        if (t_max == 0)
        {
            continue;
        }

        const std::size_t current = static_cast<std::size_t>(realization % 2);
        const std::size_t next = static_cast<std::size_t>((realization + 1) % 2);

        if (realization > 0)
        {
            if (prefetch_enabled)
            {
                preparation_jobs[current].get();
            }
            else
            {
                workspaces[current].prepare(n, t_max, p, type);
            }
        }

        // The inactive workspace is prepared on a CPU thread while CUDA-Q
        // evolves the current realization on the main thread/GPU.
        if (prefetch_enabled && realization + 1 < realizations)
        {
            preparation_jobs[next] = std::async(
                std::launch::async,
                [&workspace = workspaces[next], n, t_max, p, type]()
                {
                    workspace.prepare(n, t_max, p, type);
                });
        }

        auto &workspace = workspaces[current];
        std::string context;
        if (backend::verbose_enabled())
        {
            context = "realization=" + std::to_string(realization);
        }
        auto state = workspace.initialize(n, probe_site, context);

        int current_t = 0;
        if (t_min > 0)
        {
            state = workspace.advance(state, 0, t_min, context);
            completed_layers += static_cast<std::uint64_t>(t_min);
            current_t = t_min;
            stats[0].add(ancilla_entropy(state, n, type));
            progress.update(
                realization, realizations, current_t, completed_layers,
                completed_layers == total_layers);
        }

        const int first_incremental_t = std::max(1, t_min + 1);
        for (int t = first_incremental_t; t <= t_max; ++t)
        {
            state = workspace.advance(state, t - 1, 1, context);
            ++completed_layers;
            current_t = t;
            stats[static_cast<std::size_t>(t - t_min)].add(
                ancilla_entropy(state, n, type));
            progress.update(
                realization, realizations, current_t, completed_layers,
                completed_layers == total_layers);
        }
    }
    progress.finish();

    std::ofstream csv(output, std::ios::out | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not open output CSV: " + output);
    }
    csv << "t,S_mean,S_stderr\n";
    csv << std::setprecision(17);
    for (int t = t_min; t <= t_max; ++t)
    {
        const RunningStats &s = stats[static_cast<std::size_t>(t - t_min)];
        if (s.count != static_cast<std::uint64_t>(realizations))
        {
            throw std::runtime_error(
                "Internal error: incomplete statistics at t=" + std::to_string(t));
        }
        csv << t << ',' << s.mean << ',' << s.stderr() << '\n';
    }
    csv.flush();
    if (!csv)
    {
        throw std::runtime_error("Failed while writing output CSV: " + output);
    }

    const double total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - overall_start).count();
    std::cout << "Saved " << row_count << " rows to " << output
              << ". Elapsed time: " << total_seconds << "s\n";
    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        return run_cli(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "mipt_probed.exe error: " << error.what() << '\n';
        return 1;
    }
}
