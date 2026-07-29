#pragma once

// Per-trajectory probe measurements and their accumulators.
//
// One Sample is what a single monitored trajectory contributes at one readout
// time. Aggregate folds many Samples into means and standard errors, and
// OutputRow is one finished CSV row.

#include "mipt/util/stats.hpp"

#include <array>
#include <cstdint>
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
        if (sample.negativity > 1.0e-12)
        {
            ++positive_negativity;
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
    std::array<int, 3> distances{};
    double triangle_balance = 0.0;
    std::size_t embedding_count = 0;
    Aggregate stats;
};

} // namespace mipt::probed
