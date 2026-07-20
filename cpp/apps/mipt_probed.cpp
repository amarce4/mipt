#include "mipt/ancilla_info.hpp"
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using namespace mipt;

struct InitializeReferencesKernel
{
    void operator()(int n, const std::vector<int> &sites) __qpu__
    {
        cudaq::qvector q(n + static_cast<int>(sites.size()));
        for (std::size_t i = 0; i < sites.size(); ++i)
        {
            h(q[sites[i]]);
            cx(q[sites[i]], q[n + static_cast<int>(i)]);
        }
    }
};

struct InitializeProductKernel
{
    void operator()(int total_qubits) __qpu__
    {
        cudaq::qvector q(total_qubits);
    }
};

struct AttachReferencesKernel
{
    void operator()(cudaq::state *state,
                    int n,
                    const std::vector<int> &sites) __qpu__
    {
        cudaq::qvector q(state);
        for (std::size_t i = 0; i < sites.size(); ++i)
        {
            reset(q[sites[i]]);
            h(q[sites[i]]);
            cx(q[sites[i]], q[n + static_cast<int>(i)]);
        }
    }
};

struct RunningStats
{
    void add(double value)
    {
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (value - mean);
    }

    double stderr() const
    {
        if (count < 2)
        {
            return 0.0;
        }
        return std::sqrt(std::max(0.0, m2 / static_cast<double>(count - 1)) /
                         static_cast<double>(count));
    }

    std::uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
};

struct Sample
{
    double sq = 0.0;
    double i2 = 0.0;
    double i3 = 0.0;
    double i4 = 0.0;
    double negativity = 0.0;
    double log_negativity = 0.0;
};

struct Aggregate
{
    void add(const Sample &sample)
    {
        sq.add(sample.sq);
        i2.add(sample.i2);
        i3.add(sample.i3);
        i4.add(sample.i4);
        negativity.add(sample.negativity);
        log_negativity.add(sample.log_negativity);
    }

    RunningStats sq;
    RunningStats i2;
    RunningStats i3;
    RunningStats i4;
    RunningStats negativity;
    RunningStats log_negativity;
};

struct OutputRow
{
    double p = 0.0;
    int t = 0;
    int t0 = 0;
    Aggregate stats;
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
    return text == "-0" ? "0" : text;
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

std::vector<double> linspace(double minimum, double maximum, int count)
{
    if (count <= 0)
    {
        throw std::invalid_argument("Grid resolution must be positive.");
    }
    if (minimum > maximum)
    {
        throw std::invalid_argument("Grid minimum must not exceed its maximum.");
    }
    if (count == 1)
    {
        return {minimum};
    }
    std::vector<double> values(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        values[static_cast<std::size_t>(i)] = minimum +
            (maximum - minimum) * static_cast<double>(i) /
                static_cast<double>(count - 1);
    }
    values.front() = minimum;
    values.back() = maximum;
    return values;
}

std::vector<int> integer_linspace(int minimum, int maximum, int count)
{
    const auto floating = linspace(
        static_cast<double>(minimum), static_cast<double>(maximum), count);
    std::vector<int> values;
    values.reserve(floating.size());
    for (double value : floating)
    {
        const int rounded = static_cast<int>(std::llround(value));
        if (values.empty() || values.back() != rounded)
        {
            values.push_back(rounded);
        }
    }
    if (static_cast<int>(values.size()) != count)
    {
        throw std::invalid_argument(
            "t_res requests duplicate integer times; require t_res <= t_max-t_min+1.");
    }
    return values;
}

std::vector<int> probe_sites(int probes, int n, int mode)
{
    if (probes == 1)
    {
        const int distance = env::integer("MIPT_PROBED_DISTANCE", 1, 1, n);
        return {mode == 0 ? n - distance : n / 2};
    }
    if (probes == 2)
    {
        return {(n - 1) / 2, n - 1};
    }
    if (n % 4 != 0)
    {
        throw std::invalid_argument("Four-probe protocols require N divisible by four.");
    }
    return {n / 4 - 1, n / 2 - 1, 3 * n / 4 - 1, n - 1};
}

cudaq::state initialize_state(int n,
                              const std::vector<int> &sites,
                              bool references_attached)
{
    return references_attached
        ? cudaq::get_state(InitializeReferencesKernel{}, n, sites)
        : cudaq::get_state(
              InitializeProductKernel{}, n + static_cast<int>(sites.size()));
}

cudaq::state attach_references(cudaq::state &state,
                               int n,
                               const std::vector<int> &sites)
{
    return cudaq::get_state(AttachReferencesKernel{}, &state, n, sites);
}

Sample measure_sample(cudaq::state &state,
                      int probes,
                      int n,
                      CircuitType type)
{
    const bool fermionic = uses_fermionic_trace(type);
    const int total_qubits = n + probes;
    if (probes == 1)
    {
        const auto rho = ancilla::reduced_density_matrix(
            state, total_qubits, n, fermionic);
        return {ancilla::entropy_from_rdm(rho), 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    if (probes == 2)
    {
        const auto values = ancilla::two_probe_observables(
            state, total_qubits, n, n + 1, fermionic);
        return {
            values.joint_entropy,
            values.mutual_information,
            0.0,
            0.0,
            values.negativity,
            values.log_negativity};
    }
    const auto values = ancilla::four_probe_information(
        state,
        total_qubits,
        {n, n + 1, n + 2, n + 3},
        fermionic);
    return {values.entropy, values.i2, values.i3, values.i4, 0.0, 0.0};
}

class ProgressReporter
{
  public:
    explicit ProgressReporter(std::uint64_t total)
        : total_(total),
          start_(std::chrono::steady_clock::now()),
          enabled_(env::boolean("MIPT_PROBED_PROGRESS", true)),
          interval_(env::integer(
              "MIPT_PROBED_PROGRESS_MS", 500, 0, 60000))
    {
        last_ = start_ - interval_;
    }

    void update(std::uint64_t completed, double p, int t, bool force = false)
    {
        if (!enabled_)
        {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - last_ < interval_)
        {
            return;
        }
        last_ = now;
        printed_ = true;
        const double elapsed =
            std::chrono::duration<double>(now - start_).count();
        const double rate = elapsed > 0.0
            ? static_cast<double>(completed) / elapsed
            : 0.0;
        const double eta = rate > 0.0
            ? static_cast<double>(total_ - std::min(total_, completed)) / rate
            : 0.0;
        std::cout << "\rtrajectory=" << completed << '/' << total_
                  << ", p=" << std::setprecision(5) << p
                  << ", t=" << t
                  << ", trajectories/s=" << std::fixed << std::setprecision(3)
                  << rate
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
    std::uint64_t total_ = 0;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_;
    bool enabled_ = true;
    bool printed_ = false;
    std::chrono::milliseconds interval_{500};
};

std::string default_output_name(int probes,
                                int n,
                                int realizations,
                                CircuitType type,
                                int mode,
                                const std::vector<double> &p_values,
                                const std::vector<int> &times)
{
    std::string name = "probed_" + std::to_string(probes) + "_" +
        std::string(circuit_type_tag(type)) + "_n_" + std::to_string(n) +
        "_reals_" + compact_count(static_cast<std::uint64_t>(realizations)) +
        "_mode_" + std::to_string(mode);
    if (p_values.size() == 1)
    {
        name += "_p_" + compact_decimal(p_values.front());
    }
    else
    {
        name += "_p_" + compact_decimal(p_values.front()) + "_" +
            compact_decimal(p_values.back()) + "_res_" +
            std::to_string(p_values.size());
    }
    if (!times.empty())
    {
        name += "_t_" + std::to_string(times.front()) + "_" +
            std::to_string(times.back());
        if (mode == 2)
        {
            name += "_res_" + std::to_string(times.size());
        }
    }
    return name + ".csv";
}

void print_help(const char *program)
{
    std::cout
        << "Usage: " << program
        << " [probes] [N] [realizations] [circ_type] [probe_mode] [args]\n\n"
        << "probes     1, 2, or 4 reference qubits.\n"
        << "N          Number of monitored system sites (ancillas excluded).\n"
        << "circ_type  Circuit architecture:\n";
    print_circuit_type_help(std::cout, "               ");
    std::cout
        << "probe_mode and args:\n"
        << "  0  Legacy time trajectory: [p] [t_min] [t_max]\n"
        << "     One probe is attached at t=0. Two/four probes are attached at t0=2N;\n"
        << "     consequently their t_min must be at least 2N. One-probe depth defaults\n"
        << "     to d=1 and can be set with MIPT_PROBED_DISTANCE.\n"
        << "  1  Critical-point scan: [p_min] [p_max] [p_res]\n"
        << "     One probe follows Gullans--Huse: attach at t=0, evolve without\n"
        << "     measurements to 2N, then at p to 4N. Two/four probes evolve at p to\n"
        << "     t0=2N, are attached locally, and are sampled at t=4N.\n"
        << "  2  Entropy grid: [t_min] [t_max] [t_res] [p_min] [p_max] [p_res]\n"
        << "     References are attached at t=0 and the joint entropy S_Q=S(rho_R) is\n"
        << "     recorded on the inclusive integer-time and p linspaces.\n\n"
        << "All entropies use natural logarithms. Four-probe I2/I3 are averages over\n"
        << "the six pairs/four triplets; I4 is inclusion--exclusion information.\n"
        << "Output paths are generated from all CLI arguments; output.csv is not accepted.\n\n"
        << "Runtime controls:\n"
        << "  MIPT_PROBED_DISTANCE=1       Legacy one-probe depth d.\n"
        << "  MIPT_PROBED_PREFETCH=0       Disable CPU layer prefetch.\n"
        << "  MIPT_PROBED_PROGRESS=0       Disable progress output.\n"
        << "  MIPT_PROBED_PROGRESS_MS=500  Progress-update interval.\n";
}

void write_csv(const std::string &path,
               int probes,
               int mode,
               int n,
               int realizations,
               const std::vector<OutputRow> &rows)
{
    std::ofstream csv(path, std::ios::out | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not open output CSV: " + path);
    }
    csv << std::setprecision(17);

    if (mode == 0 && probes == 1)
    {
        csv << "t,S_mean,S_stderr\n";
        for (const auto &row : rows)
        {
            csv << row.t << ',' << row.stats.sq.mean << ','
                << row.stats.sq.stderr() << '\n';
        }
    }
    else if (mode == 0 && probes == 2)
    {
        csv << "t,t_minus_t0,scaled_time,I_mean,I_stderr,"
               "negativity_mean,negativity_stderr,"
               "log_negativity_mean,log_negativity_stderr\n";
        for (const auto &row : rows)
        {
            csv << row.t << ',' << row.t - row.t0 << ','
                << static_cast<double>(row.t - row.t0) / n << ','
                << row.stats.i2.mean << ',' << row.stats.i2.stderr() << ','
                << row.stats.negativity.mean << ','
                << row.stats.negativity.stderr() << ','
                << row.stats.log_negativity.mean << ','
                << row.stats.log_negativity.stderr() << '\n';
        }
    }
    else if (mode == 0 && probes == 4)
    {
        csv << "t,t_minus_t0,scaled_time,I2_mean,I2_stderr,"
               "I3_mean,I3_stderr,I4_mean,I4_stderr\n";
        for (const auto &row : rows)
        {
            csv << row.t << ',' << row.t - row.t0 << ','
                << static_cast<double>(row.t - row.t0) / n << ','
                << row.stats.i2.mean << ',' << row.stats.i2.stderr() << ','
                << row.stats.i3.mean << ',' << row.stats.i3.stderr() << ','
                << row.stats.i4.mean << ',' << row.stats.i4.stderr() << '\n';
        }
    }
    else if (mode == 2)
    {
        csv << "p,t,S_Q_mean,S_Q_stderr\n";
        for (const auto &row : rows)
        {
            csv << row.p << ',' << row.t << ',' << row.stats.sq.mean << ','
                << row.stats.sq.stderr() << '\n';
        }
    }
    else
    {
        csv << "p,t,S_Q_mean,S_Q_stderr";
        if (probes >= 2)
        {
            csv << ",I2_mean,I2_stderr";
        }
        if (probes == 4)
        {
            csv << ",I3_mean,I3_stderr,I4_mean,I4_stderr";
        }
        csv << '\n';
        for (const auto &row : rows)
        {
            csv << row.p << ',' << row.t << ',' << row.stats.sq.mean << ','
                << row.stats.sq.stderr();
            if (probes >= 2)
            {
                csv << ',' << row.stats.i2.mean << ',' << row.stats.i2.stderr();
            }
            if (probes == 4)
            {
                csv << ',' << row.stats.i3.mean << ',' << row.stats.i3.stderr()
                    << ',' << row.stats.i4.mean << ',' << row.stats.i4.stderr();
            }
            csv << '\n';
        }
    }
    csv.flush();
    if (!csv)
    {
        throw std::runtime_error("Failed while writing output CSV: " + path);
    }

    for (const auto &row : rows)
    {
        if (row.stats.sq.count != static_cast<std::uint64_t>(realizations))
        {
            throw std::runtime_error("Internal error: incomplete probe statistics.");
        }
    }
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
    if (argc < 6)
    {
        throw std::invalid_argument("Missing arguments. Use --help for usage.");
    }

    const int probes = std::stoi(argv[1]);
    const int n = std::stoi(argv[2]);
    const int realizations = std::stoi(argv[3]);
    const CircuitType type = parse_circuit_type(std::stoi(argv[4]));
    const int mode = std::stoi(argv[5]);
    if (probes != 1 && probes != 2 && probes != 4)
    {
        throw std::invalid_argument("probes must be 1, 2, or 4.");
    }
    if (n <= 0 || realizations <= 0)
    {
        throw std::invalid_argument("N and realizations must be positive.");
    }
    validate_circuit_site_count(type, n, "N");
    if (n + probes >= 63)
    {
        throw std::invalid_argument("N+probes must be below 63.");
    }
    if (mode < 0 || mode > 2)
    {
        throw std::invalid_argument("probe_mode must be 0, 1, or 2.");
    }
    if (env::boolean("MIPT_NATIVE_MPS", false) ||
        env::boolean("MIPT_NATIVE_TEBD", false))
    {
        throw std::invalid_argument(
            "Native MPS/TEBD does not support explicit probe ancillas; unset those variables.");
    }

    std::vector<double> p_values;
    std::vector<int> times;
    int t0 = probes == 1 ? 0 : 2 * n;
    if (mode == 0)
    {
        if (argc != 9)
        {
            throw std::invalid_argument(
                "Mode 0 requires [p] [t_min] [t_max].");
        }
        const double p = std::stod(argv[6]);
        const int t_min = std::stoi(argv[7]);
        const int t_max = std::stoi(argv[8]);
        if (p < 0.0 || p > 1.0 || t_min < 0 || t_max < t_min)
        {
            throw std::invalid_argument(
                "Require p in [0,1] and 0 <= t_min <= t_max.");
        }
        if (probes > 1 && t_min < t0)
        {
            throw std::invalid_argument(
                "Two/four-probe legacy mode requires t_min >= t0=2N.");
        }
        p_values = {p};
        times.reserve(static_cast<std::size_t>(t_max - t_min + 1));
        for (int t = t_min; t <= t_max; ++t)
        {
            times.push_back(t);
        }
    }
    else if (mode == 1)
    {
        if (argc != 9)
        {
            throw std::invalid_argument(
                "Mode 1 requires [p_min] [p_max] [p_res].");
        }
        const double p_min = std::stod(argv[6]);
        const double p_max = std::stod(argv[7]);
        const int p_res = std::stoi(argv[8]);
        if (p_min < 0.0 || p_max > 1.0)
        {
            throw std::invalid_argument("p endpoints must lie in [0,1].");
        }
        p_values = linspace(p_min, p_max, p_res);
        times = {4 * n};
        t0 = 2 * n;
    }
    else
    {
        if (argc != 12)
        {
            throw std::invalid_argument(
                "Mode 2 requires [t_min] [t_max] [t_res] [p_min] [p_max] [p_res].");
        }
        const int t_min = std::stoi(argv[6]);
        const int t_max = std::stoi(argv[7]);
        const int t_res = std::stoi(argv[8]);
        const double p_min = std::stod(argv[9]);
        const double p_max = std::stod(argv[10]);
        const int p_res = std::stoi(argv[11]);
        if (t_min < 0 || t_max < t_min || p_min < 0.0 || p_max > 1.0)
        {
            throw std::invalid_argument(
                "Require nonnegative ordered times and p endpoints in [0,1].");
        }
        times = integer_linspace(t_min, t_max, t_res);
        p_values = linspace(p_min, p_max, p_res);
        t0 = 0;
    }

    const auto sites = probe_sites(probes, n, mode);
    const int t_max = times.back();
    const std::string output = default_output_name(
        probes, n, realizations, type, mode, p_values, times);
    const bool prefetch = env::boolean("MIPT_PROBED_PREFETCH", true) &&
        realizations > 1 && t_max > 0;
    const std::uint64_t total_trajectories =
        static_cast<std::uint64_t>(p_values.size()) *
        static_cast<std::uint64_t>(realizations);
    ProgressReporter progress(total_trajectories);
    std::uint64_t completed = 0;
    const auto start = std::chrono::steady_clock::now();

    std::cout << "Unified probed MIPT simulation\n"
              << "probes=" << probes
              << ", system_N=" << n
              << ", total_qubits=" << n + probes
              << ", mode=" << mode
              << ", realizations=" << realizations
              << ", p_points=" << p_values.size()
              << ", t_points=" << times.size()
              << ", circ_type=" << static_cast<int>(type)
              << " (" << circuit_type_name(type) << ')'
              << ", fermionic_trace="
              << (uses_fermionic_trace(type) ? "on" : "off")
              << ", cpu_prefetch=" << (prefetch ? "on" : "off")
              << ", output=" << output << '\n';

    std::vector<OutputRow> rows;
    rows.reserve(p_values.size() * times.size());
    for (double p : p_values)
    {
        std::vector<Aggregate> aggregates(times.size());
        std::array<probed::CircuitWorkspace1D, 2> workspaces;
        for (auto &workspace : workspaces)
        {
            workspace.reserve(t_max);
        }
        std::array<std::future<void>, 2> jobs;
        auto prepare = [&](probed::CircuitWorkspace1D &workspace)
        {
            if (mode == 1 && probes == 1)
            {
                workspace.prepare_quench(n, t_max, 2 * n, p, type);
            }
            else
            {
                workspace.prepare(n, t_max, p, type);
            }
        };
        if (t_max > 0)
        {
            prepare(workspaces[0]);
        }

        for (int realization = 0; realization < realizations; ++realization)
        {
            const std::size_t current = static_cast<std::size_t>(realization % 2);
            const std::size_t next = static_cast<std::size_t>((realization + 1) % 2);
            if (t_max > 0 && realization > 0)
            {
                if (prefetch)
                {
                    jobs[current].get();
                }
                else
                {
                    prepare(workspaces[current]);
                }
            }
            if (t_max > 0 && prefetch && realization + 1 < realizations)
            {
                jobs[next] = std::async(
                    std::launch::async,
                    [&workspace = workspaces[next], &prepare]()
                    {
                        prepare(workspace);
                    });
            }

            const bool attach_at_start = mode == 2 || probes == 1;
            auto state = initialize_state(n, sites, attach_at_start);
            auto &workspace = workspaces[current];
            int current_t = 0;
            std::size_t first_sample = 0;

            if ((mode == 0 || mode == 1) && probes > 1)
            {
                state = workspace.advance(state, 0, 2 * n);
                current_t = 2 * n;
                state = attach_references(state, n, sites);
            }

            for (std::size_t sample_index = first_sample;
                 sample_index < times.size();
                 ++sample_index)
            {
                const int target_t = times[sample_index];
                if (target_t > current_t)
                {
                    state = workspace.advance(
                        state, current_t, target_t - current_t);
                    current_t = target_t;
                }
                if (mode == 2 || mode == 1 || probes != 1 || target_t > 0)
                {
                    aggregates[sample_index].add(
                        measure_sample(state, probes, n, type));
                }
                else
                {
                    aggregates[sample_index].add(
                        {std::log(2.0), 0.0, 0.0, 0.0, 0.0, 0.0});
                }
            }
            ++completed;
            progress.update(
                completed, p, times.back(), completed == total_trajectories);
        }

        for (std::size_t i = 0; i < times.size(); ++i)
        {
            rows.push_back({p, times[i], t0, aggregates[i]});
        }
    }
    progress.finish();
    write_csv(output, probes, mode, n, realizations, rows);

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "Saved " << rows.size() << " rows to " << output
              << ". Elapsed time: " << elapsed << "s\n";
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
