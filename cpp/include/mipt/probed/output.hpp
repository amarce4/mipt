#pragma once

// CSV output for every probe mode.
//
// Each mode has its own observable columns, but all of them are prefixed with
// the same self-describing metadata block:
//
//     N,circ_type,circuit_name,realizations,p,...
//
// so data_analysis never has to recover the run parameters by parsing the
// filename. p is part of the prefix because it is the scan variable in modes 1
// and 2 and a constant in the others; the resume logic in checkpoint.hpp finds
// it by name, not by position.

#include "mipt/env.hpp"
#include "mipt/probed/sample.hpp"
#include "mipt/types.hpp"
#include "mipt/util/geometry.hpp"
#include "mipt/util/text.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::probed
{
using util::compact_count;
using util::compact_decimal;
using util::csv_quote;

inline constexpr const char *METADATA_COLUMNS = "N,circ_type,circuit_name,realizations,p";

inline void write_metadata_fields(std::ostream &csv, int n, CircuitType type, int realizations, double p)
{
    csv << n << ',' << static_cast<int>(type) << ',' << csv_quote(circuit_type_name(type)) << ',' << realizations << ','
        << p;
}

inline std::string default_output_name(int probes, int n, int realizations, CircuitType type, int mode,
                                       const std::vector<double> &p_values, const std::vector<int> &times, int delta_x,
                                       int delta_t, int t_eq, double triangle_balance_cutoff)
{
    std::string name = "probed_" + std::to_string(probes) + "_" + std::string(circuit_type_tag(type)) + "_n_" +
                       std::to_string(n) + "_reals_" + compact_count(static_cast<std::uint64_t>(realizations)) +
                       "_mode_" + std::to_string(mode);
    if (p_values.size() == 1)
    {
        name += "_p_" + compact_decimal(p_values.front());
    }
    else
    {
        name += "_p_" + compact_decimal(p_values.front()) + "_" + compact_decimal(p_values.back()) + "_res_" +
                std::to_string(p_values.size());
    }
    if (!times.empty())
    {
        if (mode == 3)
        {
            name += "_dx_" + std::to_string(delta_x) + "_dt_" + std::to_string(delta_t) + "_tau_" +
                    std::to_string(times.front()) + "_" + std::to_string(times.back());
        }
        else if (mode == 4)
        {
            if (probes == 3)
            {
                name += "_b_" + compact_decimal(triangle_balance_cutoff);
            }
            name += "_teq_" + std::to_string(t_eq) + "_tau_0_" + std::to_string(times.back());
        }
        else
        {
            name += "_t_" + std::to_string(times.front()) + "_" + std::to_string(times.back());
            if (mode == 2)
            {
                name += "_res_" + std::to_string(times.size());
            }
        }
    }
    return name + ".csv";
}

namespace detail
{

// Mode 4, two probes: mutual information and negativity against separation.
inline void write_mode4_pairs(std::ostream &csv, int n, int realizations, CircuitType type,
                              const std::vector<OutputRow> &rows, int t_eq)
{
    csv << METADATA_COLUMNS
        << ",distance,chord_length,t_eq,tau,tau_over_chord,"
           "t_absolute,I_mean,I_stderr,negativity_mean,"
           "negativity_stderr,log_negativity_mean,"
           "log_negativity_stderr,negativity_positive_count,"
           "negativity_positive_fraction,origins_randomized\n";
    for (const auto &row : rows)
    {
        const double chord = util::chord_length(n, row.distance);
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << row.distance << ',' << chord << ',' << t_eq << ',' << row.t << ','
            << static_cast<double>(row.t) / chord << ',' << t_eq + row.t << ',' << row.stats.i2.mean << ','
            << row.stats.i2.stderr() << ',' << row.stats.negativity.mean << ',' << row.stats.negativity.stderr() << ','
            << row.stats.log_negativity.mean << ',' << row.stats.log_negativity.stderr() << ','
            << row.stats.positive_negativity << ','
            << static_cast<double>(row.stats.positive_negativity) / static_cast<double>(row.stats.negativity.count)
            << ",1\n";
    }
}

// Mode 4, three probes: multipartite metrics per triangle geometry.
inline void write_mode4_triangles(std::ostream &csv, int n, int realizations, CircuitType type,
                                  const std::vector<OutputRow> &rows, int t_eq, double triangle_balance_cutoff)
{
    const int gmn_stride =
        static_cast<int>(env::integer("MIPT_PROBED_GMN_STRIDE", 1, 1, std::numeric_limits<int>::max()));
    const int gmn_limit = static_cast<int>(env::integer("MIPT_PROBED_GMN_SAMPLES_PER_BIN", 0, 0, realizations));
    const char *measure_name = uses_fermionic_trace(type) ? "fGMN" : "GMN";
    csv << METADATA_COLUMNS
        << ",geometry_id,distance_1,distance_2,distance_3,"
           "chord_1,chord_2,chord_3,chord_geometric_mean,"
           "triangle_balance,triangle_balance_cutoff,embedding_count,"
           "t_eq,tau,tau_over_chord_geometric_mean,"
           "t_absolute,TMI_mean,TMI_stderr,average_MI_mean,"
           "average_MI_stderr,min_bipartite_negativity_mean,"
           "min_bipartite_negativity_stderr,gmn_type,GMN_mean,"
           "GMN_stderr,GMN_sample_count,GMN_requested_count,"
           "GMN_zero_prefilter_count,GMN_solver_failure_count,"
           "GMN_stride,GMN_samples_per_bin,joint_purity_mean,"
           "joint_purity_stderr,mean_single_purity_mean,"
           "mean_single_purity_stderr,realizations_per_geometry\n";
    for (const auto &row : rows)
    {
        std::array<double, 3> chords{};
        for (int i = 0; i < 3; ++i)
        {
            chords[static_cast<std::size_t>(i)] = util::chord_length(n, row.distances[static_cast<std::size_t>(i)]);
        }
        const double chord_geometric_mean = std::cbrt(chords[0] * chords[1] * chords[2]);
        const double gmn_mean = row.stats.gmn.count > 0 ? row.stats.gmn.mean : std::numeric_limits<double>::quiet_NaN();
        const double gmn_stderr =
            row.stats.gmn.count > 0 ? row.stats.gmn.stderr() : std::numeric_limits<double>::quiet_NaN();
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << row.geometry << ',' << row.distances[0] << ',' << row.distances[1] << ',' << row.distances[2]
            << ',' << chords[0] << ',' << chords[1] << ',' << chords[2] << ',' << chord_geometric_mean << ','
            << row.triangle_balance << ',' << triangle_balance_cutoff << ',' << row.embedding_count << ',' << t_eq
            << ',' << row.t << ',' << static_cast<double>(row.t) / chord_geometric_mean << ',' << t_eq + row.t << ','
            << row.stats.tmi.mean << ',' << row.stats.tmi.stderr() << ',' << row.stats.average_mi.mean << ','
            << row.stats.average_mi.stderr() << ',' << row.stats.min_bipartite_negativity.mean << ','
            << row.stats.min_bipartite_negativity.stderr() << ',' << measure_name << ',' << gmn_mean << ','
            << gmn_stderr << ',' << row.stats.gmn.count << ',' << row.stats.gmn_requested << ','
            << row.stats.gmn_zero_prefiltered << ',' << row.stats.gmn_solver_failures << ',' << gmn_stride << ','
            << gmn_limit << ',' << row.stats.joint_purity.mean << ',' << row.stats.joint_purity.stderr() << ','
            << row.stats.mean_single_purity.mean << ',' << row.stats.mean_single_purity.stderr() << ','
            << realizations << '\n';
    }
}

// Mode 3: the spacetime-anisotropy correlator between two staggered probes.
inline void write_mode3(std::ostream &csv, int n, int realizations, CircuitType type,
                        const std::vector<OutputRow> &rows, int delta_x, int delta_t)
{
    const int tau1 = 4 * n;
    const int tau2 = tau1 + delta_t;
    csv << METADATA_COLUMNS
        << ",delta_x,delta_t,tau1,tau2,tau,tau_over_L,t_absolute,"
           "I_mean,I_stderr,negativity_mean,negativity_stderr,"
           "log_negativity_mean,log_negativity_stderr\n";
    for (const auto &row : rows)
    {
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << delta_x << ',' << delta_t << ',' << tau1 << ',' << tau2 << ',' << row.t << ','
            << static_cast<double>(row.t) / n << ',' << tau2 + row.t << ',' << row.stats.i2.mean << ','
            << row.stats.i2.stderr() << ',' << row.stats.negativity.mean << ',' << row.stats.negativity.stderr() << ','
            << row.stats.log_negativity.mean << ',' << row.stats.log_negativity.stderr() << '\n';
    }
}

// Mode 0: a single time trajectory at fixed p.
inline void write_mode0(std::ostream &csv, int probes, int n, int realizations, CircuitType type,
                        const std::vector<OutputRow> &rows)
{
    if (probes == 1)
    {
        csv << METADATA_COLUMNS << ",t,S_mean,S_stderr\n";
        for (const auto &row : rows)
        {
            write_metadata_fields(csv, n, type, realizations, row.p);
            csv << ',' << row.t << ',' << row.stats.sq.mean << ',' << row.stats.sq.stderr() << '\n';
        }
        return;
    }
    if (probes == 2)
    {
        csv << METADATA_COLUMNS
            << ",t,t_minus_t0,scaled_time,I_mean,I_stderr,"
               "negativity_mean,negativity_stderr,"
               "log_negativity_mean,log_negativity_stderr\n";
        for (const auto &row : rows)
        {
            write_metadata_fields(csv, n, type, realizations, row.p);
            csv << ',' << row.t << ',' << row.t - row.t0 << ',' << static_cast<double>(row.t - row.t0) / n << ','
                << row.stats.i2.mean << ',' << row.stats.i2.stderr() << ',' << row.stats.negativity.mean << ','
                << row.stats.negativity.stderr() << ',' << row.stats.log_negativity.mean << ','
                << row.stats.log_negativity.stderr() << '\n';
        }
        return;
    }
    csv << METADATA_COLUMNS
        << ",t,t_minus_t0,scaled_time,I2_mean,I2_stderr,"
           "I3_mean,I3_stderr,I4_mean,I4_stderr\n";
    for (const auto &row : rows)
    {
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << row.t << ',' << row.t - row.t0 << ',' << static_cast<double>(row.t - row.t0) / n << ','
            << row.stats.i2.mean << ',' << row.stats.i2.stderr() << ',' << row.stats.i3.mean << ','
            << row.stats.i3.stderr() << ',' << row.stats.i4.mean << ',' << row.stats.i4.stderr() << '\n';
    }
}

// Mode 2: the (p, t) entropy grid.
inline void write_mode2(std::ostream &csv, int n, int realizations, CircuitType type,
                        const std::vector<OutputRow> &rows)
{
    csv << METADATA_COLUMNS << ",t,S_Q_mean,S_Q_stderr\n";
    for (const auto &row : rows)
    {
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << row.t << ',' << row.stats.sq.mean << ',' << row.stats.sq.stderr() << '\n';
    }
}

// Mode 1: the fixed-t critical scan, whose columns depend on the probe count.
inline void write_mode1(std::ostream &csv, int probes, int n, int realizations, CircuitType type,
                        const std::vector<OutputRow> &rows)
{
    csv << METADATA_COLUMNS << ",t,S_Q_mean,S_Q_stderr";
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
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << row.t << ',' << row.stats.sq.mean << ',' << row.stats.sq.stderr();
        if (probes >= 2)
        {
            csv << ',' << row.stats.i2.mean << ',' << row.stats.i2.stderr();
        }
        if (probes == 4)
        {
            csv << ',' << row.stats.i3.mean << ',' << row.stats.i3.stderr() << ',' << row.stats.i4.mean << ','
                << row.stats.i4.stderr();
        }
        csv << '\n';
    }
}

} // namespace detail

inline void write_csv(const std::string &path, int probes, int mode, int n, int realizations, CircuitType type,
                      const std::vector<OutputRow> &rows, int delta_x, int delta_t, int t_eq,
                      double triangle_balance_cutoff)
{
    std::ofstream csv(path, std::ios::out | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not open output CSV: " + path);
    }
    csv << std::setprecision(17);

    if (mode == 4 && probes == 2)
    {
        detail::write_mode4_pairs(csv, n, realizations, type, rows, t_eq);
    }
    else if (mode == 4)
    {
        detail::write_mode4_triangles(csv, n, realizations, type, rows, t_eq, triangle_balance_cutoff);
    }
    else if (mode == 3)
    {
        detail::write_mode3(csv, n, realizations, type, rows, delta_x, delta_t);
    }
    else if (mode == 0)
    {
        detail::write_mode0(csv, probes, n, realizations, type, rows);
    }
    else if (mode == 2)
    {
        detail::write_mode2(csv, n, realizations, type, rows);
    }
    else
    {
        detail::write_mode1(csv, probes, n, realizations, type, rows);
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

} // namespace mipt::probed
