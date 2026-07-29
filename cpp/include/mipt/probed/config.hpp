#pragma once

// Command-line parsing for mipt_probed.exe.
//
// Everything the simulation needs is resolved here into one ProbeRunConfig:
// the p and t grids, where the references attach, which encoding is used, and
// the generated output path. The runner then has no argv to interpret.
//
// The five protocols share one executable because they share one circuit
// evolution and one reference-attachment mechanism; only the schedule of
// "advance, attach, measure" differs between them.

#include "mipt/env.hpp"
#include "mipt/probed/geometry.hpp"
#include "mipt/probed/output.hpp"
#include "mipt/probed/sample.hpp"
#include "mipt/types.hpp"
#include "mipt/util/grid.hpp"
#include "mipt/util/text.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::probed
{
using util::checked_linspace;
using util::integer_linspace;

struct ProbeRunConfig
{
    int probes = 1;
    int n = 0;
    int realizations = 0;
    CircuitType type = CircuitType::MMS;
    int mode = 0;

    std::vector<double> p_values;
    std::vector<int> times;
    int delta_x = -1;
    int delta_t = -1;
    int t0 = 0;
    int t_eq = 0;
    double triangle_balance_cutoff = 0.0;

    // Derived once the grids are known.
    bool parity_encoding = false;
    std::vector<int> sites;
    std::vector<ProbeGeometry> geometries;
    int t_max = 0;
    std::string output;
    std::uint64_t distance_count = 1;
    bool prefetch = false;
    std::uint64_t total_trajectories = 0;
};

inline void print_help(const char *program)
{
    std::cout << "Usage: " << program << " [probes] [N] [realizations] [circ_type] [probe_mode] [args]\n\n"
              << "probes     1, 2, 3, or 4 reference qubits. Three probes are mode-4 "
                 "only.\n"
        << "N          Number of monitored system sites (ancillas excluded).\n"
        << "circ_type  Circuit architecture:\n";
    print_circuit_type_help(std::cout, "               ");
    std::cout << "probe_mode and args:\n"
        << "  0  Legacy time trajectory: [p] [t_min] [t_max]\n"
              << "     One probe is attached at t=0. Two/four probes are attached at "
                 "t0=2N;\n"
              << "     consequently their t_min must be at least 2N. One-probe depth "
                 "defaults\n"
        << "     to d=1 and can be set with MIPT_PROBED_DISTANCE.\n"
        << "  1  Critical-point scan: [p_min] [p_max] [p_res]\n"
        << "     One probe follows Gullans--Huse: attach at t=0, evolve without\n"
              << "     measurements to 2N, then at p to 4N. Two/four probes evolve at "
                 "p to\n"
        << "     t0=2N, are attached locally, and are sampled at t=4N.\n"
        << "  2  Entropy grid: [t_min] [t_max] [t_res] [p_min] [p_max] [p_res]\n"
              << "     References are attached at t=0 and the joint entropy "
                 "S_Q=S(rho_R) is\n"
        << "     recorded on the inclusive integer-time and p linspaces.\n"
        << "  3  Anisotropy correlation: [p] [delta_x] [delta_t]\n"
              << "     Requires probes=2. The first reference is attached to site 1 "
                 "at\n"
        << "     tau1=4N; the second is attached to site 1+delta_x (periodic,\n"
        << "     one-indexed) at tau2=tau1+\n"
              << "     delta_t. Use (delta_x,delta_t)=(N/2,0) for the spatial branch "
                 "or\n"
              << "     (0,delta_t>0) for the temporal branch. Mutual information is "
                 "recorded\n"
              << "     for elapsed readout time tau=0,...,8N after the second "
                 "insertion, so\n"
        << "     the absolute circuit depth is 12N+delta_t.\n\n"
              << "  4  Distance-resolved probe dynamics: [p] [tau_max] [B_min]\n"
              << "     probes=2: simulate every distance r=1,...,floor(N/2), using a "
                 "fresh\n"
              << "     random origin for each independent trajectory.\n"
              << "     probes=3: simulate every unique sorted three-distance triangle "
                 "whose\n"
              << "     CFT chord balance B=ell_min/ell_max satisfies B>=B_min, with\n"
              << "     exactly equal representation, choosing uniformly among all "
                 "lattice\n"
              << "     embeddings of that triangle. Records TMI, average pair MI, "
                 "minimum\n"
              << "     bipartite negativity, GMN/fGMN, joint purity, and mean "
                 "single-probe\n"
              << "     purity. Both protocols attach after t_eq and sample "
                 "tau=0..tau_max.\n"
              << "     B_min is optional for probes=3 and defaults to "
                 "MIPT_PROBED_TRIANGLE_BALANCE or 0.5.\n\n"
              << "All entropies use natural logarithms. Four-probe I2/I3 are averages "
                 "over\n"
              << "the six pairs/four triplets; I4 is inclusion--exclusion "
                 "information.\n"
              << "Output paths are generated from all CLI arguments; output.csv is not "
                 "accepted.\n\n"
        << "Runtime controls:\n"
        << "  MIPT_PROBED_DISTANCE=1       Legacy one-probe depth d.\n"
        << "  MIPT_PROBED_PREFETCH=0       Disable CPU layer prefetch.\n"
              << "  MIPT_PROBED_PARITY_ENCODING=0 Disable exact one-probe parity "
                 "encoding.\n"
        << "  MIPT_PROBED_ISOLATE_P=0      Disable per-p subprocess isolation.\n"
              << "  MIPT_PROBED_RESUME=0         Replace, rather than resume, a p "
                 "scan.\n"
              << "  MIPT_PROBED_RETRIES=1        Retries after a failed isolated p "
                 "point.\n"
        << "  MIPT_PROBED_PROGRESS=0       Disable progress output.\n"
        << "  MIPT_PROBED_PROGRESS_MS=500  Progress-update interval.\n"
              << "  MIPT_PROBED_T_EQ=2N          Mode-4 equilibration timesteps.\n"
              << "  MIPT_PROBED_TRIANGLE_BALANCE=0.5  Minimum three-probe CFT chord "
                 "balance.\n"
              << "  MIPT_PROBED_GMN_BATCH_RECORDS=64  RDMs per deferred SDP batch.\n"
              << "  MIPT_PROBED_GMN_PENDING_BATCHES=2 Maximum concurrent SDP batches.\n"
              << "  MIPT_PROBED_GMN_STRIDE=1     Evaluate GMN/fGMN every nth tau.\n"
              << "  MIPT_PROBED_GMN_SAMPLES_PER_BIN=0  Samples per geometry/tau; "
                 "0=all.\n"
              << "  MIPT_PROBED_GMN_ZERO_TOL=1e-10  Zero-prefilter tolerance.\n";
}

namespace detail
{

// Arguments common to every mode: probes, N, realizations, circuit, mode.
inline void parse_common_arguments(int argc, char *argv[], ProbeRunConfig &config)
{
    if (argc < 6)
    {
        throw std::invalid_argument("Missing arguments. Use --help for usage.");
    }
    config.probes = std::stoi(argv[1]);
    config.n = std::stoi(argv[2]);
    config.realizations = std::stoi(argv[3]);
    config.type = parse_circuit_type(std::stoi(argv[4]));
    config.mode = std::stoi(argv[5]);

    if (config.probes != 1 && config.probes != 2 && config.probes != 3 && config.probes != 4)
    {
        throw std::invalid_argument("probes must be 1, 2, 3, or 4.");
    }
    if (config.n <= 0 || config.realizations <= 0)
    {
        throw std::invalid_argument("N and realizations must be positive.");
    }
    validate_circuit_site_count(config.type, config.n, "N");
    if (config.n + config.probes >= 63)
    {
        throw std::invalid_argument("N+probes must be below 63.");
    }
    if (config.mode < 0 || config.mode > 4)
    {
        throw std::invalid_argument("probe_mode must be 0, 1, 2, 3, or 4.");
    }
    if (env::boolean("MIPT_NATIVE_MPS", false) || env::boolean("MIPT_NATIVE_TEBD", false))
    {
        throw std::invalid_argument("Native MPS/TEBD does not support explicit "
                                    "probe ancillas; unset those variables.");
    }
    if (config.probes == 3 && config.mode != 4)
    {
        throw std::invalid_argument("Three probes are currently supported only in mode 4.");
    }
}

// Mode 0: one p, a contiguous band of readout times.
inline void parse_mode0(int argc, char *argv[], ProbeRunConfig &config)
{
    if (argc != 9)
    {
        throw std::invalid_argument("Mode 0 requires [p] [t_min] [t_max].");
    }
    const double p = std::stod(argv[6]);
    const int t_min = std::stoi(argv[7]);
    const int t_max = std::stoi(argv[8]);
    if (p < 0.0 || p > 1.0 || t_min < 0 || t_max < t_min)
    {
        throw std::invalid_argument("Require p in [0,1] and 0 <= t_min <= t_max.");
    }
    if (config.probes > 1 && t_min < config.t0)
    {
        throw std::invalid_argument("Two/four-probe legacy mode requires t_min >= t0=2N.");
    }
    config.p_values = {p};
    config.times.reserve(static_cast<std::size_t>(t_max - t_min + 1));
    for (int t = t_min; t <= t_max; ++t)
    {
        config.times.push_back(t);
    }
}

// Mode 1: Gullans-Huse critical scan, read out at the fixed depth t=4N.
inline void parse_mode1(int argc, char *argv[], ProbeRunConfig &config)
{
    if (argc != 9)
    {
        throw std::invalid_argument("Mode 1 requires [p_min] [p_max] [p_res].");
    }
    const double p_min = std::stod(argv[6]);
    const double p_max = std::stod(argv[7]);
    const int p_res = std::stoi(argv[8]);
    if (p_min < 0.0 || p_max > 1.0)
    {
        throw std::invalid_argument("p endpoints must lie in [0,1].");
    }
    config.p_values = checked_linspace(p_min, p_max, p_res);
    config.times = {4 * config.n};
    config.t0 = 2 * config.n;
}

// Mode 2: a rectangular (p, t) grid of the joint reference entropy.
inline void parse_mode2(int argc, char *argv[], ProbeRunConfig &config)
{
    if (argc != 12)
    {
        throw std::invalid_argument("Mode 2 requires [t_min] [t_max] [t_res] [p_min] [p_max] [p_res].");
    }
    const int t_min = std::stoi(argv[6]);
    const int t_max = std::stoi(argv[7]);
    const int t_res = std::stoi(argv[8]);
    const double p_min = std::stod(argv[9]);
    const double p_max = std::stod(argv[10]);
    const int p_res = std::stoi(argv[11]);
    if (t_min < 0 || t_max < t_min || p_min < 0.0 || p_max > 1.0)
    {
        throw std::invalid_argument("Require nonnegative ordered times and p endpoints in [0,1].");
    }
    config.times = integer_linspace(t_min, t_max, t_res);
    config.p_values = checked_linspace(p_min, p_max, p_res);
    config.t0 = 0;
}

// Mode 3: two probes staggered in space or in time, for the anisotropy alpha.
inline void parse_mode3(int argc, char *argv[], ProbeRunConfig &config)
{
    if (config.probes != 2)
    {
        throw std::invalid_argument("Mode 3 requires probes=2.");
    }
    if (argc != 9)
    {
        throw std::invalid_argument("Mode 3 requires [p] [delta_x] [delta_t].");
    }
    const int n = config.n;
    const double p = std::stod(argv[6]);
    config.delta_x = std::stoi(argv[7]);
    config.delta_t = std::stoi(argv[8]);
    if (p < 0.0 || p > 1.0 || config.delta_x < 0 || config.delta_x > n / 2 || config.delta_t < 0)
    {
        throw std::invalid_argument("Mode 3 requires p in [0,1], 0 <= delta_x <= N/2, and delta_t >= 0.");
    }
    const bool spatial_branch = config.delta_t == 0;
    if (spatial_branch && (n % 2 != 0 || config.delta_x != n / 2))
    {
        throw std::invalid_argument("The mode-3 spatial branch requires even N, delta_x=N/2, delta_t=0.");
    }
    if (!spatial_branch && config.delta_x != 0)
    {
        throw std::invalid_argument("The mode-3 temporal branch requires delta_x=0 and delta_t>0.");
    }
    config.p_values = {p};
    config.times.reserve(static_cast<std::size_t>(8 * n + 1));
    for (int tau = 0; tau <= 8 * n; ++tau)
    {
        config.times.push_back(tau);
    }
    config.t0 = 4 * n + config.delta_t;
}

// Mode 4: distance-resolved pairs, or balanced triangles for three probes.
inline void parse_mode4(int argc, char *argv[], ProbeRunConfig &config)
{
    if (config.probes != 2 && config.probes != 3)
    {
        throw std::invalid_argument("Mode 4 requires probes=2 or probes=3.");
    }
    if (config.n < config.probes)
    {
        throw std::invalid_argument("Mode 4 requires N >= probes.");
    }
    if ((config.probes == 2 && argc != 8) || (config.probes == 3 && argc != 8 && argc != 9))
    {
        throw std::invalid_argument("Mode 4 requires [p] [tau_max], with optional [B_min] for probes=3.");
    }
    const double p = std::stod(argv[6]);
    const int tau_max = std::stoi(argv[7]);
    if (p < 0.0 || p > 1.0 || tau_max < 0 || tau_max > std::numeric_limits<int>::max() - 2 * config.n)
    {
        throw std::invalid_argument("Mode 4 requires p in [0,1] and a nonnegative tau_max "
                                    "that leaves room for the default t_eq=2N.");
    }
    config.t_eq = static_cast<int>(
        env::integer("MIPT_PROBED_T_EQ", 2 * config.n, 0, std::numeric_limits<int>::max() - tau_max));
    config.p_values = {p};
    config.times.reserve(static_cast<std::size_t>(tau_max + 1));
    for (int tau = 0; tau <= tau_max; ++tau)
    {
        config.times.push_back(tau);
    }
    if (config.probes == 3)
    {
        config.triangle_balance_cutoff =
            argc == 9 ? std::stod(argv[8]) : env::real("MIPT_PROBED_TRIANGLE_BALANCE", 0.5, 0.0, 1.0);
        if (!std::isfinite(config.triangle_balance_cutoff) || config.triangle_balance_cutoff < 0.0 ||
            config.triangle_balance_cutoff > 1.0)
        {
            throw std::invalid_argument("Three-probe mode 4 requires B_min in [0,1].");
        }
    }
    config.t0 = config.t_eq;
}

// Probe placement, encoding choice, output path, and workload size.
inline void resolve_derived_settings(ProbeRunConfig &config)
{
    const int n = config.n;
    config.parity_encoding = config.mode == 1 && config.probes == 1 &&
                             preserves_computational_parity(config.type) &&
                             env::boolean("MIPT_PROBED_PARITY_ENCODING", true);
    config.sites = config.mode == 4 ? std::vector<int>{} : probe_sites(config.probes, n, config.mode, config.delta_x);
    config.geometries = config.mode == 4 && config.probes == 3
                            ? enumerate_three_probe_geometries(n, config.triangle_balance_cutoff)
                            : std::vector<ProbeGeometry>{};
    config.t_max = config.mode == 3   ? 12 * n + config.delta_t
                   : config.mode == 4 ? config.t_eq + config.times.back()
                                      : config.times.back();

    const std::string generated_output =
        default_output_name(config.probes, n, config.realizations, config.type, config.mode, config.p_values,
                            config.times, config.delta_x, config.delta_t, config.t_eq,
                            config.triangle_balance_cutoff);
    config.output = env::text("MIPT_PROBED_INTERNAL_OUTPUT", generated_output.c_str());

    config.distance_count = config.mode == 4 ? config.probes == 2
                                                   ? static_cast<std::uint64_t>(n / 2)
                                                   : static_cast<std::uint64_t>(config.geometries.size())
                                             : 1u;
    config.prefetch = env::boolean("MIPT_PROBED_PREFETCH", true) &&
                      static_cast<std::uint64_t>(config.realizations) * config.distance_count > 1 && config.t_max > 0;
    config.total_trajectories = static_cast<std::uint64_t>(config.p_values.size()) *
                                static_cast<std::uint64_t>(config.realizations) * config.distance_count;
}

} // namespace detail

inline ProbeRunConfig parse_probe_arguments(int argc, char *argv[])
{
    ProbeRunConfig config;
    detail::parse_common_arguments(argc, argv, config);

    config.t0 = config.probes == 1 ? 0 : 2 * config.n;
    config.t_eq = 2 * config.n;
    switch (config.mode)
    {
    case 0:
        detail::parse_mode0(argc, argv, config);
        break;
    case 1:
        detail::parse_mode1(argc, argv, config);
        break;
    case 2:
        detail::parse_mode2(argc, argv, config);
        break;
    case 3:
        detail::parse_mode3(argc, argv, config);
        break;
    default:
        detail::parse_mode4(argc, argv, config);
        break;
    }

    detail::resolve_derived_settings(config);
    return config;
}

// The one-line run banner printed before a simulation starts.
inline std::string describe_run(const ProbeRunConfig &config)
{
    const int n = config.n;
    const int mode = config.mode;
    std::string text =
        "probes=" + std::to_string(config.probes) + ", system_N=" + std::to_string(n) +
        ", physical_total_qubits=" + std::to_string(n + config.probes) +
        ", simulated_qubits=" + std::to_string(n + config.probes - (config.parity_encoding ? 1 : 0)) +
        ", mode=" + std::to_string(mode) + ", realizations_per_p=" + std::to_string(config.realizations) +
        ", total_trajectories=" + std::to_string(config.total_trajectories) +
        ", p_points=" + std::to_string(config.p_values.size()) +
        ", t_points=" + std::to_string(config.times.size());

    if (mode == 3)
    {
        text += ", t_absolute=0.." + std::to_string(config.t_max);
    }
    else if (mode == 4)
    {
        text += ", t_eq=" + std::to_string(config.t_eq) + ", tau=0.." + std::to_string(config.times.back()) +
                (config.probes == 2 ? ", distances=1.." + std::to_string(n / 2)
                                    : ", triangle_geometries=" + std::to_string(config.geometries.size()));
    }
    else if (config.times.size() == 1)
    {
        text += ", t=" + std::to_string(config.times.front());
    }
    else
    {
        text += ", t=" + std::to_string(config.times.front()) + ".." + std::to_string(config.times.back());
    }

    if (mode == 3)
    {
        text += ", delta_x=" + std::to_string(config.delta_x) + ", delta_t=" + std::to_string(config.delta_t) +
                ", readout_tau_max=8N";
    }

    text += ", circ_type=" + std::to_string(static_cast<int>(config.type)) + " (" +
            std::string(circuit_type_name(config.type)) + ')' +
            ", fermionic_trace=" + (uses_fermionic_trace(config.type) ? "on" : "off") +
            ", probe_encoding=" + (config.parity_encoding ? "parity_symmetry" : "explicit_reference") +
            ", cpu_prefetch=" + (config.prefetch ? "on" : "off");

    if (mode == 4 && config.probes == 3)
    {
        text += ", multipartite_measure=" + std::string(uses_fermionic_trace(config.type) ? "fGMN" : "GMN") +
                ", triangle_balance_cutoff=" + util::compact_decimal(config.triangle_balance_cutoff);
    }
    return text + ", output=" + config.output;
}

} // namespace mipt::probed
