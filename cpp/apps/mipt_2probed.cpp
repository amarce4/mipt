#include "mipt/ancilla_mi.hpp"
#include "mipt/backend.hpp"
#include "mipt/env.hpp"
#include "mipt/probed.hpp"
#include "mipt/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using namespace mipt;

struct InitializeTwoProbeKernel
{
    void operator()(int n) __qpu__
    {
        cudaq::qvector q(n + 2);
    }
};

struct AttachTwoProbeKernel
{
    void operator()(cudaq::state *state,
                    int n,
                    int probe_site_a,
                    int probe_site_b) __qpu__
    {
        cudaq::qvector q(state);

        // Replace the selected system sites with fresh local Bell pairs at t0.
        cudaq::reset(q[probe_site_a]);
        cudaq::reset(q[probe_site_b]);
        h(q[probe_site_a]);
        h(q[probe_site_b]);
        cx(q[probe_site_a], q[n]);
        cx(q[probe_site_b], q[n + 1]);
    }
};

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

struct ProbeStats
{
    void add(const ancilla::TwoProbeObservables &values)
    {
        mutual_information.add(values.mutual_information);
        negativity.add(values.negativity);
        log_negativity.add(values.log_negativity);
    }

    std::uint64_t count() const
    {
        if (mutual_information.count != negativity.count ||
            mutual_information.count != log_negativity.count)
        {
            throw std::runtime_error("Internal error: probe statistic counts disagree.");
        }
        return mutual_information.count;
    }

    RunningStats mutual_information;
    RunningStats negativity;
    RunningStats log_negativity;
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
                                int realizations,
                                int t_min,
                                int t_max,
                                CircuitType type)
{
    return "2probed_" + std::string(circuit_type_tag(type)) +
           "_n_" + std::to_string(n) +
           "_p_" + compact_decimal(p) +
           "_reals_" + compact_count(static_cast<std::uint64_t>(realizations)) +
           "_t_" + std::to_string(t_min) + "_" + std::to_string(t_max) +
           ".csv";
}

cudaq::state initialize_two_probe_state(int n, std::string_view context)
{
    if (backend::verbose_enabled())
    {
        backend::verbose_log(
            std::string(context) +
            " get_state start: two-probe product initialization system_n=" +
            std::to_string(n) +
            " total_qubits=" + std::to_string(n + 2));
    }
    const auto start = std::chrono::steady_clock::now();
    auto state = cudaq::get_state(InitializeTwoProbeKernel{}, n);
    if (backend::verbose_enabled())
    {
        backend::verbose_log(
            std::string(context) +
            " get_state done: two-probe product initialization elapsed_s=" +
            std::to_string(backend::seconds_since(start)));
    }
    return state;
}

cudaq::state attach_two_probes(cudaq::state &state,
                               int n,
                               int probe_site_a,
                               int probe_site_b,
                               std::string_view context)
{
    if (backend::verbose_enabled())
    {
        backend::verbose_log(
            std::string(context) +
            " get_state start: attach two probes system_n=" +
            std::to_string(n) +
            " total_qubits=" + std::to_string(n + 2));
    }
    const auto start = std::chrono::steady_clock::now();
    auto attached = cudaq::get_state(
        AttachTwoProbeKernel{},
        &state,
        n,
        probe_site_a,
        probe_site_b);
    if (backend::verbose_enabled())
    {
        backend::verbose_log(
            std::string(context) +
            " get_state done: attach two probes elapsed_s=" +
            std::to_string(backend::seconds_since(start)));
    }
    return attached;
}

class ProgressReporter
{
  public:
    ProgressReporter(std::uint64_t total_layers,
                     std::chrono::steady_clock::time_point start)
        : total_layers_(total_layers),
          start_(start),
          enabled_(env::boolean(
              "MIPT_2PROBED_PROGRESS",
              env::boolean("MIPT_PROBED_PROGRESS", true))),
          interval_(env::integer(
              "MIPT_2PROBED_PROGRESS_MS",
              env::integer("MIPT_PROBED_PROGRESS_MS", 500, 0, 60000),
              0,
              60000))
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
        const double eta_seconds = rate > 0.0
            ? static_cast<double>(remaining) / rate
            : 0.0;

        std::cout << "\rrealization=" << realization + 1 << '/' << realizations
                  << ", t=" << timestep
                  << ", layers/s=" << std::fixed << std::setprecision(2) << rate
                  << ", ETA=" << static_cast<std::uint64_t>(eta_seconds)
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
              << " [N] [p] [realizations] [t_min] [t_max] [circ_type] [output.csv]\n\n"
              << "Arguments:\n"
              << "  N            Number of MIPT system sites, excluding two ancillas (default 16).\n"
              << "  p            Measurement probability per system site and timestep (default 0.33).\n"
              << "  realizations Independent trajectories evolved through t_max (default 10000).\n"
              << "  t_min        First absolute circuit time written, inclusive (default 2N).\n"
              << "  t_max        Last absolute circuit time written, inclusive (default 10N).\n"
              << "  circ_type    Circuit architecture:\n";
    print_circuit_type_help(std::cout, "                 ");
    std::cout << "  output.csv   Optional output path. The default name encodes all parameters.\n\n"
              << "Protocol for the bulk order-parameter correlation:\n"
              << "  1. Start the N-site system and both references in a product state.\n"
              << "  2. Evolve only the system through t0=2N using periodic circuit layers.\n"
              << "  3. At t0, reset system sites q_(N-1)/2 and q_(N-1), then prepare\n"
              << "     exact Bell pairs (q_(N-1)/2,q_N) and (q_(N-1),q_(N+1)).\n"
              << "  4. Continue the same trajectory and compute I(A:B), negativity, and\n"
              << "     logarithmic negativity after every sampled post-measurement timestep.\n"
              << "     Require t_min >= t0.\n\n"
              << "The CSV columns are t, t_minus_t0, scaled_time=(t-t0)/N, I_mean,\n"
              << "I_stderr, negativity_mean, negativity_stderr, log_negativity_mean,\n"
              << "and log_negativity_stderr. Entropies and logarithmic negativity use\n"
              << "natural logarithms, so I and log-negativity are in nats.\n"
              << "For fermionic circuit types, the existing fermionic partial-trace and\n"
              << "fermionic partial-transpose conventions are used.\n\n"
              << "Runtime controls:\n"
              << "  MIPT_2PROBED_PREFETCH=0       Disable CPU preparation overlap.\n"
              << "  MIPT_2PROBED_PROGRESS=0       Disable progress output.\n"
              << "  MIPT_2PROBED_PROGRESS_MS=500  Minimum progress-update interval.\n"
              << "The corresponding MIPT_PROBED_* variables are used as fallbacks.\n";
}

int run_cli(int argc, char *argv[])
{
    backend::print_backend_banner_once("mipt_2probed.exe");

    if (argc > 1 &&
        (std::strcmp(argv[1], "--help") == 0 ||
         std::strcmp(argv[1], "-h") == 0))
    {
        print_help(argv[0]);
        return 0;
    }
    if (argc > 8)
    {
        throw std::invalid_argument("Too many arguments. Use --help for usage.");
    }

    const int n = argc > 1 ? std::stoi(argv[1]) : 16;
    const double p = argc > 2 ? std::stod(argv[2]) : 0.33;
    const int realizations = argc > 3 ? std::stoi(argv[3]) : 10000;
    const int t0 = 2 * n;
    const int t_min = argc > 4 ? std::stoi(argv[4]) : t0;
    const int t_max = argc > 5 ? std::stoi(argv[5]) : 10 * n;
    const CircuitType type = parse_circuit_type(argc > 6 ? std::stoi(argv[6]) : 2);
    const std::string output = argc > 7
        ? std::string(argv[7])
        : default_output_name(n, p, realizations, t_min, t_max, type);

    if (n <= 1)
    {
        throw std::invalid_argument("N must exceed one.");
    }
    validate_circuit_site_count(type, n, "N");
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("p must lie in [0,1].");
    }
    if (realizations <= 0)
    {
        throw std::invalid_argument("realizations must be positive.");
    }
    if (t_min < t0 || t_max < t_min)
    {
        throw std::invalid_argument(
            "Require t0=2N <= t_min <= t_max; t_min and t_max are absolute times.");
    }
    if (output.empty())
    {
        throw std::invalid_argument("output.csv must not be empty.");
    }
    if (n > 60)
    {
        throw std::invalid_argument(
            "N+2 must be below 63 for the ancilla reduced-density-matrix indexing.");
    }
    if (env::boolean("MIPT_NATIVE_MPS", false) ||
        env::boolean("MIPT_NATIVE_TEBD", false))
    {
        throw std::invalid_argument(
            "MIPT_NATIVE_MPS=1 and MIPT_NATIVE_TEBD=1 are not supported by mipt_2probed.exe; unset them so the explicit ancillas are evolved by CUDA-Q.");
    }

    const int probe_site_a = (n - 1) / 2;
    const int probe_site_b = n - 1;
    const int ancilla_a = n;
    const int ancilla_b = n + 1;
    const int total_qubits = n + 2;
    const int row_count = t_max - t_min + 1;
    std::vector<ProbeStats> stats(static_cast<std::size_t>(row_count));

    const bool prefetch_enabled = env::boolean(
        "MIPT_2PROBED_PREFETCH",
        env::boolean("MIPT_PROBED_PREFETCH", true)) &&
        realizations > 1 && t_max > 0;

    const std::uint64_t total_layers =
        static_cast<std::uint64_t>(realizations) *
        static_cast<std::uint64_t>(t_max);
    std::uint64_t completed_layers = 0;
    const auto overall_start = std::chrono::steady_clock::now();
    ProgressReporter progress(total_layers, overall_start);

    std::cout << "Two-probe bulk-correlation MIPT simulation\n"
              << "system_N=" << n
              << ", total_qubits=" << total_qubits
              << ", p=" << p
              << ", t0=" << t0
              << ", probe_A=(system q_" << probe_site_a
              << ", ancilla q_" << ancilla_a << ')'
              << ", probe_B=(system q_" << probe_site_b
              << ", ancilla q_" << ancilla_b << ')'
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
    workspaces[0].prepare(n, t_max, p, type);

    for (int realization = 0; realization < realizations; ++realization)
    {
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

        auto state = initialize_two_probe_state(n, context);
        state = workspace.advance(state, 0, t0, context);
        completed_layers += static_cast<std::uint64_t>(t0);
        state = attach_two_probes(
            state, n, probe_site_a, probe_site_b, context);

        if (t_min > t0)
        {
            state = workspace.advance(state, t0, t_min - t0, context);
            completed_layers += static_cast<std::uint64_t>(t_min - t0);
        }

        auto record_observables = [&](std::size_t row)
        {
            stats[row].add(ancilla::two_probe_observables(
                state,
                total_qubits,
                ancilla_a,
                ancilla_b,
                uses_fermionic_trace(type)));
        };

        record_observables(0);
        progress.update(
            realization,
            realizations,
            t_min,
            completed_layers,
            completed_layers == total_layers);

        for (int t = t_min + 1; t <= t_max; ++t)
        {
            state = workspace.advance(state, t - 1, 1, context);
            ++completed_layers;
            record_observables(static_cast<std::size_t>(t - t_min));
            progress.update(
                realization,
                realizations,
                t,
                completed_layers,
                completed_layers == total_layers);
        }
    }
    progress.finish();

    std::ofstream csv(output, std::ios::out | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not open output CSV: " + output);
    }
    csv << "t,t_minus_t0,scaled_time,I_mean,I_stderr,"
           "negativity_mean,negativity_stderr,"
           "log_negativity_mean,log_negativity_stderr\n";
    csv << std::setprecision(17);
    for (int t = t_min; t <= t_max; ++t)
    {
        const ProbeStats &sample =
            stats[static_cast<std::size_t>(t - t_min)];
        if (sample.count() != static_cast<std::uint64_t>(realizations))
        {
            throw std::runtime_error(
                "Internal error: incomplete statistics at t=" +
                std::to_string(t));
        }
        csv << t << ','
            << t - t0 << ','
            << static_cast<double>(t - t0) / static_cast<double>(n) << ','
            << sample.mutual_information.mean << ','
            << sample.mutual_information.stderr() << ','
            << sample.negativity.mean << ','
            << sample.negativity.stderr() << ','
            << sample.log_negativity.mean << ','
            << sample.log_negativity.stderr() << '\n';
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
        std::cerr << "mipt_2probed.exe error: " << error.what() << '\n';
        return 1;
    }
}
