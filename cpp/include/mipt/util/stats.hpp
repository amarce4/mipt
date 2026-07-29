#pragma once

// Streaming mean/variance accumulator shared by every trajectory loop.
//
// Three byte-for-byte equivalent copies of this used to live in entropy.hpp,
// free_energy.hpp, and mipt_probed.cpp, each exposing a slightly different set
// of accessor names. All of them are provided here so no call site had to
// change its spelling.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mipt::util
{

// Welford accumulation: numerically stable for the 10^4-10^6 trajectory counts
// these simulations use, and never stores the samples.
struct RunningStats
{
    std::uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double value)
    {
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (value - mean);
    }

    double sample_stddev() const
    {
        return count > 1 ? std::sqrt(std::max(0.0, m2 / static_cast<double>(count - 1))) : 0.0;
    }

    double standard_error() const
    {
        return count > 1 ? sample_stddev() / std::sqrt(static_cast<double>(count)) : 0.0;
    }

    // Spelling used by the probe and free-energy writers.
    double stderr() const
    {
        return standard_error();
    }
};

} // namespace mipt::util
