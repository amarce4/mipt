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

#include <algorithm>
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
                                       int delta_t, int t_eq, double triangle_balance_cutoff, int x_max = -1,
                                       int r_min = -1, int r_max = -1, int delta_max = -1)
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
        else if (mode == 5 && probes == 2)
        {
            name += "_teq_" + std::to_string(t_eq) + "_r_" + std::to_string(r_min) + "_" + std::to_string(r_max) +
                    "_d_0_" + std::to_string(delta_max) + "_s_0_" + std::to_string(times.back());
        }
        else if (mode == 5)
        {
            name += "_teq_" + std::to_string(t_eq) + "_tau_0_" + std::to_string(times.back()) + "_xmax_" +
                    std::to_string(x_max);
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

// --- batch-mean sidecar -----------------------------------------------------
//
// The aggregate CSV reports one mean and one standard error per (bin, time),
// which is everything a per-point error bar needs and not enough for a fit.
// A collapse exponent is fitted to a whole curve at once, so its uncertainty
// depends on how the points covary -- and every point of a curve comes from
// the *same* trajectories, so they covary strongly. Resampling points
// independently against their own stderrs therefore understates the error on
// eta by a large factor.
//
// The fix is to keep a modest number of independent replicas of the whole
// curve. Trajectories are partitioned into `batch_count` contiguous batches
// and each batch accumulates its own copy of every bin and time, so one batch
// is one complete, statistically independent measurement of the entire grid.
// A batch bootstrap over those curves then carries the temporal and
// cross-metric covariance that a per-point resample throws away, at the cost
// of one extra file of batch_count x rows rather than one row per trajectory.
//
// 50-100 batches is the useful band: enough replicas for a percentile
// interval, few enough that each batch mean is still well determined.

inline std::string batch_output_path(const std::string &path)
{
    const std::string suffix = ".csv";
    if (path.size() >= suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
        return path.substr(0, path.size() - suffix.size()) + "_batches.csv";
    }
    return path + "_batches.csv";
}

// How many batches a run of `realizations` trajectories should use.
//
// Zero disables the sidecar. The default lands inside the 50-100 band whenever
// the run is large enough to fill it; a run with fewer trajectories than that
// gets one batch per trajectory, which is degenerate but honest -- the batch
// spread is then just the trajectory spread.
inline int batch_count_for(int realizations)
{
    const int requested = static_cast<int>(env::integer("MIPT_PROBED_BATCHES", 64, 0, 1000000));
    if (requested <= 0 || realizations <= 0)
    {
        return 0;
    }
    return std::min(requested, realizations);
}

// Which batch each realization index belongs to, and how many realizations
// each batch holds. Contiguous near-equal blocks: trajectories are i.i.d., so
// any partition is as good as any other, and contiguity keeps the mapping
// reconstructible from the CSV alone.
struct BatchPlan
{
    int count = 0;
    std::vector<int> batch_of_realization;
    std::vector<int> realizations_in_batch;

    int batch_at(std::size_t realization_index) const
    {
        return batch_of_realization[realization_index];
    }
};

inline BatchPlan make_batch_plan(int realizations, int batch_count)
{
    BatchPlan plan;
    plan.count = batch_count;
    if (batch_count <= 0)
    {
        return plan;
    }
    plan.realizations_in_batch.assign(static_cast<std::size_t>(batch_count), realizations / batch_count);
    const int remainder = realizations % batch_count;
    for (int i = 0; i < remainder; ++i)
    {
        ++plan.realizations_in_batch[static_cast<std::size_t>(i)];
    }
    plan.batch_of_realization.reserve(static_cast<std::size_t>(realizations));
    for (int batch = 0; batch < batch_count; ++batch)
    {
        for (int i = 0; i < plan.realizations_in_batch[static_cast<std::size_t>(batch)]; ++i)
        {
            plan.batch_of_realization.push_back(batch);
        }
    }
    return plan;
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

// Mode 5: the classical and entanglement fronts across the (width, x, tau) grid.
//
// One row per admissible (width, distance, elapsed time). `chord_length` is the
// periodic chord for the block's innermost site, which is the anchor the
// distance is measured from; `t_absolute` locates the row in the full circuit
// so that a truncated run can be compared with a longer one.
inline void write_mode5(std::ostream &csv, int n, int realizations, CircuitType type,
                        const std::vector<OutputRow> &rows, int t_eq)
{
    csv << METADATA_COLUMNS
        << ",t_eq,width,x,chord_length,tau,t_absolute,"
           "C_X_mean,C_X_stderr,C_Y_mean,C_Y_stderr,C_Z_mean,C_Z_stderr,"
           "C_mean_mean,C_mean_stderr,C_max_mean,C_max_stderr,"
           "I_mean,I_stderr,negativity_mean,negativity_stderr,"
           "log_negativity_mean,log_negativity_stderr,"
           "coherent_information_mean,coherent_information_stderr,"
           "S_RB_mean,S_RB_stderr,S_B_mean,S_B_stderr,"
           "negativity_positive_count,negativity_positive_fraction,"
           "samples,origins_randomized\n";
    for (const auto &row : rows)
    {
        const auto &stats = row.stats;
        // x=0 is the injection site itself, which has no chord to itself; it is
        // kept as the source amplitude that thresholds can be normalized to.
        const double chord = row.distance > 0 ? util::chord_length(n, row.distance) : 0.0;
        const double positive_fraction =
            stats.negativity.count > 0
                ? static_cast<double>(stats.positive_negativity) / static_cast<double>(stats.negativity.count)
                : std::numeric_limits<double>::quiet_NaN();
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << t_eq << ',' << row.width << ',' << row.distance << ',' << chord << ',' << row.t << ','
            << t_eq + row.t << ','
            << stats.classical_x.mean << ',' << stats.classical_x.stderr() << ',' << stats.classical_y.mean << ','
            << stats.classical_y.stderr() << ',' << stats.classical_z.mean << ',' << stats.classical_z.stderr() << ','
            << stats.classical_mean.mean << ',' << stats.classical_mean.stderr() << ',' << stats.classical_max.mean
            << ',' << stats.classical_max.stderr() << ',' << stats.i2.mean << ',' << stats.i2.stderr() << ','
            << stats.negativity.mean << ',' << stats.negativity.stderr() << ',' << stats.log_negativity.mean << ','
            << stats.log_negativity.stderr() << ',' << stats.coherent_information.mean << ','
            << stats.coherent_information.stderr() << ',' << stats.sq.mean << ',' << stats.sq.stderr() << ','
            << stats.block_entropy.mean << ',' << stats.block_entropy.stderr() << ',' << stats.positive_negativity
            << ',' << positive_fraction << ',' << stats.sq.count << ",1\n";
    }
}

// Mode 5, two probes: the staggered-insertion grid.
//
// One row per (separation r, insertion delay delta, elapsed time s). The
// readout clock starts at the *second* insertion, so `t_absolute` is
// t_eq + delta + s; `t_second_insertion` is written alongside it because the
// two probes have been in the circuit for different lengths of time and the
// survival entropies S_A and S_B are only comparable once that is accounted for.
//
// `J` and `discord` are NaN unless the run enabled the measurement
// optimization (MIPT_PROBED_DISCORD); C_max is the three-axis lower bound on J
// and is always present.
inline void write_mode5_stagger(std::ostream &csv, int n, int realizations, CircuitType type,
                                const std::vector<OutputRow> &rows, int t_eq)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    csv << METADATA_COLUMNS
        << ",t_eq,r,chord_length,delta,s,t_second_insertion,t_absolute,"
           "I_mean,I_stderr,J_mean,J_stderr,discord_mean,discord_stderr,"
           "C_X_mean,C_X_stderr,C_Y_mean,C_Y_stderr,C_Z_mean,C_Z_stderr,"
           "C_max_mean,C_max_stderr,"
           "negativity_mean,negativity_stderr,"
           "log_negativity_mean,log_negativity_stderr,"
           "negativity_positive_count,negativity_positive_fraction,"
           "conditional_negativity_mean,conditional_negativity_stderr,"
           "coherent_information_mean,coherent_information_stderr,"
           "S_AB_mean,S_AB_stderr,S_A_mean,S_A_stderr,S_B_mean,S_B_stderr,"
           "samples,origins_randomized\n";
    for (const auto &row : rows)
    {
        const auto &stats = row.stats;
        const double chord = row.distance > 0 ? util::chord_length(n, row.distance) : 0.0;
        const double positive_fraction =
            stats.negativity.count > 0
                ? static_cast<double>(stats.positive_negativity) / static_cast<double>(stats.negativity.count)
                : nan;
        // An empty accumulator means "never measured", which is a different
        // statement from "measured and found to be zero".
        const double j_mean = stats.classical_optimized.count > 0 ? stats.classical_optimized.mean : nan;
        const double j_stderr = stats.classical_optimized.count > 0 ? stats.classical_optimized.stderr() : nan;
        const double discord_mean = stats.discord.count > 0 ? stats.discord.mean : nan;
        const double discord_stderr = stats.discord.count > 0 ? stats.discord.stderr() : nan;
        const double conditional_mean = stats.conditional_negativity.count > 0 ? stats.conditional_negativity.mean : nan;
        const double conditional_stderr =
            stats.conditional_negativity.count > 0 ? stats.conditional_negativity.stderr() : nan;
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << t_eq << ',' << row.distance << ',' << chord << ',' << row.delay << ',' << row.t << ','
            << t_eq + row.delay << ',' << t_eq + row.delay + row.t << ',' << stats.i2.mean << ',' << stats.i2.stderr()
            << ',' << j_mean << ',' << j_stderr << ',' << discord_mean << ',' << discord_stderr << ','
            << stats.classical_x.mean << ',' << stats.classical_x.stderr() << ',' << stats.classical_y.mean << ','
            << stats.classical_y.stderr() << ',' << stats.classical_z.mean << ',' << stats.classical_z.stderr() << ','
            << stats.classical_max.mean << ',' << stats.classical_max.stderr() << ',' << stats.negativity.mean << ','
            << stats.negativity.stderr() << ',' << stats.log_negativity.mean << ',' << stats.log_negativity.stderr()
            << ',' << stats.positive_negativity << ',' << positive_fraction << ',' << conditional_mean << ','
            << conditional_stderr << ',' << stats.coherent_information.mean << ','
            << stats.coherent_information.stderr() << ',' << stats.sq.mean << ',' << stats.sq.stderr() << ','
            << stats.reference_entropy.mean << ',' << stats.reference_entropy.stderr() << ','
            << stats.block_entropy.mean << ',' << stats.block_entropy.stderr() << ',' << stats.sq.count << ",1\n";
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

// Mode 0, two probes: one batch-mean curve per batch id.
//
// The observable columns keep the `_mean` spelling the aggregate CSV uses so a
// reader that locates columns by name finds the same observables here; what is
// absent is the `_stderr` partner, because a batch mean's error is not the
// quantity of interest -- the spread *across* batches is.
inline void write_mode0_batches(std::ostream &csv, int n, int realizations, CircuitType type,
                                const std::vector<OutputRow> &rows)
{
    csv << METADATA_COLUMNS
        << ",batch_id,batch_realizations,t,t_minus_t0,scaled_time,"
           "I_mean,negativity_mean,log_negativity_mean,"
           "negativity_positive_fraction\n";
    for (const auto &row : rows)
    {
        const auto &stats = row.stats;
        const double positive_fraction =
            stats.negativity.count > 0
                ? static_cast<double>(stats.positive_negativity) / static_cast<double>(stats.negativity.count)
                : std::numeric_limits<double>::quiet_NaN();
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << row.batch_id << ',' << row.batch_realizations << ',' << row.t << ',' << row.t - row.t0 << ','
            << static_cast<double>(row.t - row.t0) / n << ',' << stats.i2.mean << ',' << stats.negativity.mean << ','
            << stats.log_negativity.mean << ',' << positive_fraction << '\n';
    }
}

// Mode 4, two probes: one batch-mean curve per (batch id, separation).
//
// Batches are cut on the realization index *within* a separation group, so
// batch b holds the same number of trajectories at every distance and covers
// the whole (distance, tau) grid. Different separations are different
// trajectories regardless, so a batch is a replica of the grid, not of one
// circuit.
inline void write_mode4_pair_batches(std::ostream &csv, int n, int realizations, CircuitType type,
                                     const std::vector<OutputRow> &rows, int t_eq)
{
    csv << METADATA_COLUMNS
        << ",batch_id,batch_realizations,distance,chord_length,t_eq,tau,"
           "tau_over_chord,t_absolute,I_mean,negativity_mean,"
           "log_negativity_mean,negativity_positive_fraction\n";
    for (const auto &row : rows)
    {
        const auto &stats = row.stats;
        const double chord = util::chord_length(n, row.distance);
        const double positive_fraction =
            stats.negativity.count > 0
                ? static_cast<double>(stats.positive_negativity) / static_cast<double>(stats.negativity.count)
                : std::numeric_limits<double>::quiet_NaN();
        write_metadata_fields(csv, n, type, realizations, row.p);
        csv << ',' << row.batch_id << ',' << row.batch_realizations << ',' << row.distance << ',' << chord << ','
            << t_eq << ',' << row.t << ',' << static_cast<double>(row.t) / chord << ',' << t_eq + row.t << ','
            << stats.i2.mean << ',' << stats.negativity.mean << ',' << stats.log_negativity.mean << ','
            << positive_fraction << '\n';
    }
}

} // namespace detail

// Whether this configuration writes a batch-mean sidecar at all.
//
// Only the two protocols whose curves are fitted as curves: two-probe mode 0
// (the eta collapse) and two-probe mode 4 (the distance collapse). Three-probe
// mode 4 is excluded because its GMN column is filled asynchronously from a
// solver queue that knows only the global bin, so a batch copy of it would be
// silently empty.
inline bool writes_batch_csv(int probes, int mode)
{
    return probes == 2 && (mode == 0 || mode == 4);
}

inline void write_batch_csv(const std::string &path, int probes, int mode, int n, int realizations, CircuitType type,
                            const std::vector<OutputRow> &rows, int t_eq)
{
    if (!writes_batch_csv(probes, mode))
    {
        return;
    }
    std::ofstream csv(path, std::ios::out | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not open batch CSV: " + path);
    }
    csv << std::setprecision(17);
    if (mode == 0)
    {
        detail::write_mode0_batches(csv, n, realizations, type, rows);
    }
    else
    {
        detail::write_mode4_pair_batches(csv, n, realizations, type, rows, t_eq);
    }
    csv.flush();
    if (!csv)
    {
        throw std::runtime_error("Failed while writing batch CSV: " + path);
    }
}

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

    if (mode == 5 && probes == 2)
    {
        detail::write_mode5_stagger(csv, n, realizations, type, rows, t_eq);
    }
    else if (mode == 5)
    {
        detail::write_mode5(csv, n, realizations, type, rows, t_eq);
    }
    else if (mode == 4 && probes == 2)
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
