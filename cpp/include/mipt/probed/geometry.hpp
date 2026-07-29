#pragma once

// Where the reference qubits attach to the ring.
//
// Fixed protocols (modes 0-3) use the deterministic placements in
// probe_sites. Mode 4 instead enumerates every distinct triangle of chord
// distances, so each geometry can be sampled with equal weight regardless of
// how many lattice embeddings realize it.

#include "mipt/env.hpp"
#include "mipt/probed/sample.hpp"
#include "mipt/util/geometry.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::probed
{
using util::periodic_distance;
using util::triangle_balance;

std::vector<ProbeGeometry> enumerate_three_probe_geometries(int n, double minimum_balance)
{
    if (n < 3)
    {
        throw std::invalid_argument("Three-probe mode 4 requires N >= 3.");
    }
    std::map<std::array<int, 3>, std::vector<std::array<int, 3>>> grouped;
    for (int a = 0; a < n - 2; ++a)
    {
        for (int b = a + 1; b < n - 1; ++b)
        {
            for (int c = b + 1; c < n; ++c)
            {
                std::array<int, 3> signature{periodic_distance(a, b, n), periodic_distance(a, c, n),
                                             periodic_distance(b, c, n)};
                std::sort(signature.begin(), signature.end());
                grouped[signature].push_back({a, b, c});
            }
        }
    }

    std::vector<ProbeGeometry> geometries;
    geometries.reserve(grouped.size());
    for (auto &[distances, embeddings] : grouped)
    {
        const double balance = triangle_balance(distances, n);
        if (balance + 1.0e-12 >= minimum_balance)
        {
            geometries.push_back({distances, balance, std::move(embeddings)});
        }
    }
    if (geometries.empty())
    {
        throw std::invalid_argument("No three-probe triangle geometry satisfies B >= " +
                                    std::to_string(minimum_balance) +
                                    "; lower the mode-4 B_min argument or "
                                    "MIPT_PROBED_TRIANGLE_BALANCE.");
    }
    return geometries;
}

std::vector<int> probe_sites(int probes, int n, int mode, int delta_x = 0)
{
    if (probes == 1)
    {
        const int distance = env::integer("MIPT_PROBED_DISTANCE", 1, 1, n);
        return {mode == 0 ? n - distance : n / 2};
    }
    if (probes == 2)
    {
        if (mode == 3)
        {
            // x1=1 and x2=1+delta_x in one-indexed notation.
            return {0, delta_x % n};
        }
        return {(n - 1) / 2, n - 1};
    }
    if (n % 4 != 0)
    {
        throw std::invalid_argument("Four-probe protocols require N divisible by four.");
    }
    return {n / 4 - 1, n / 2 - 1, 3 * n / 4 - 1, n - 1};
}

} // namespace mipt::probed
