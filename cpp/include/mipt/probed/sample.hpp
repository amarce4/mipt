#pragma once

// Per-trajectory probe measurements and their accumulators.
//
// One Sample is what a single monitored trajectory contributes at one readout
// time. Aggregate folds many Samples into means and standard errors, and
// OutputRow is one finished CSV row.

#include "mipt/util/stats.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace mipt::probed
{
using util::RunningStats;

struct Sample
{
    double sq = 0.0;
    double i2 = 0.0;
    double i3 = 0.0;
    double i4 = 0.0;
    double negativity = 0.0;
    double log_negativity = 0.0;
    double tmi = 0.0;
    double average_mi = 0.0;
    double min_bipartite_negativity = 0.0;
    double joint_purity = 0.0;
    double mean_single_purity = 0.0;

    // Mode 5 only: the Holevo information reaching the receiver block about a
    // bit encoded in each Pauli basis at the source, and the block's own
    // entropy and coherent information. See front_metrics.hpp.
    double classical_x = 0.0;
    double classical_y = 0.0;
    double classical_z = 0.0;
    double classical_mean = 0.0;
    double classical_max = 0.0;
    double coherent_information = 0.0;
    double block_entropy = 0.0;
    double reference_entropy = 0.0;
    // Optimized classical correlation and discord. Only the two-probe protocol
    // runs the measurement optimization by default, so these carry NaN
    // elsewhere and their accumulators stay empty.
    double classical_optimized = std::numeric_limits<double>::quiet_NaN();
    double discord = std::numeric_limits<double>::quiet_NaN();
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
        tmi.add(sample.tmi);
        average_mi.add(sample.average_mi);
        min_bipartite_negativity.add(sample.min_bipartite_negativity);
        joint_purity.add(sample.joint_purity);
        mean_single_purity.add(sample.mean_single_purity);
        classical_x.add(sample.classical_x);
        classical_y.add(sample.classical_y);
        classical_z.add(sample.classical_z);
        classical_mean.add(sample.classical_mean);
        classical_max.add(sample.classical_max);
        coherent_information.add(sample.coherent_information);
        block_entropy.add(sample.block_entropy);
        reference_entropy.add(sample.reference_entropy);
        // A NaN here means the caller did not ask for the measurement
        // optimization, not that the discord vanished; leaving the accumulator
        // empty is what lets the writer emit NaN rather than a fabricated zero.
        if (std::isfinite(sample.classical_optimized))
        {
            classical_optimized.add(sample.classical_optimized);
            discord.add(sample.discord);
        }
        if (sample.negativity > 1.0e-12)
        {
            ++positive_negativity;
            // E[N | N>0]. The unconditional mean collapses once entanglement
            // becomes rare, and separating the two says whether it got smaller
            // or merely less frequent.
            conditional_negativity.add(sample.negativity);
        }
    }

    RunningStats sq;
    RunningStats i2;
    RunningStats i3;
    RunningStats i4;
    RunningStats negativity;
    RunningStats log_negativity;
    RunningStats tmi;
    RunningStats average_mi;
    RunningStats min_bipartite_negativity;
    RunningStats gmn;
    RunningStats joint_purity;
    RunningStats mean_single_purity;
    RunningStats classical_x;
    RunningStats classical_y;
    RunningStats classical_z;
    RunningStats classical_mean;
    RunningStats classical_max;
    RunningStats coherent_information;
    RunningStats block_entropy;
    RunningStats reference_entropy;
    RunningStats classical_optimized;
    RunningStats discord;
    RunningStats conditional_negativity;
    std::uint64_t positive_negativity = 0;
    std::uint64_t gmn_requested = 0;
    std::uint64_t gmn_zero_prefiltered = 0;
    std::uint64_t gmn_solver_failures = 0;
};

struct ProbeGeometry
{
    std::array<int, 3> distances{};
    double balance = 0.0;
    std::vector<std::array<int, 3>> embeddings;
};

struct OutputRow
{
    double p = 0.0;
    int t = 0;
    int t0 = 0;
    int distance = -1;
    int geometry = -1;
    int width = -1; // One-probe mode-5 receiver-block width; -1 elsewhere.
    int delay = -1; // Two-probe mode-5 insertion delay; -1 elsewhere.
    std::array<int, 3> distances{};
    double triangle_balance = 0.0;
    std::size_t embedding_count = 0;
    Aggregate stats;
};

} // namespace mipt::probed
