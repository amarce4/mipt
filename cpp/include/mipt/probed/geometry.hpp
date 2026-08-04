#pragma once

// Where the reference qubits attach to the ring, and where mode 5 looks for
// the signal they inject.
//
// Fixed protocols (modes 0-3) use the deterministic placements in
// probe_sites. Mode 4 instead enumerates every distinct triangle of chord
// distances, so each geometry can be sampled with equal weight regardless of
// how many lattice embeddings realize it. Mode 5 attaches its single probe to a
// random site and reads out on the receiver blocks defined below.

#include "mipt/env.hpp"
#include "mipt/probed/sample.hpp"
#include "mipt/util/geometry.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <sstream>
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

// ---------------------------------------------------------------------------
// Mode 5: receiver blocks
//
// B_w(x) is the block of w consecutive sites whose *innermost* site sits at arc
// distance x from the injection site x_0, extending away from it; the mirrored
// block on the other side is measured too and the two are averaged, as the
// protocol asks.
//
// Two things follow from anchoring on the innermost site rather than on the
// block's centre. It makes the left- and right-moving blocks exact mirror
// images for every width, so the directional average is unbiased, and it makes
// B_1(x) a subset of B_2(x) a subset of B_3(x), so one RDM over the widest
// block yields every narrower one by partial trace instead of a fresh pass over
// the state vector. Physically it is also the right anchor: the front reaches a
// block through its nearest site. Against the centred convention it shifts the
// arrival time by at most (w-1)/2 sites, which moves the intercept b of
// tau(x) = b + lambda x and leaves the slope -- the velocity -- untouched.
// ---------------------------------------------------------------------------

// The widest receiver block one reference RDM can cover: three block modes plus
// the reference is a 16x16 joint matrix. See mipt::front::MAX_BLOCK_WIDTH.
inline constexpr int MAX_RECEIVER_WIDTH = 3;

// A block fits when all of its sites stay within half a ring of the origin, so
// that "distance x, moving outward" is unambiguous and the two directional
// blocks never meet on the far side.
inline bool receiver_block_fits(int n, int width, int x)
{
    return width >= 1 && width <= MAX_RECEIVER_WIDTH && x >= 0 && x + width - 1 <= n / 2;
}

// The sites of B_w(x) on one side, innermost first. `direction` is +1 or -1.
inline std::vector<int> receiver_block_sites(int n, int origin, int x, int width, int direction)
{
    std::vector<int> sites;
    sites.reserve(static_cast<std::size_t>(width));
    for (int offset = 0; offset < width; ++offset)
    {
        const int site = ((origin + direction * (x + offset)) % n + n) % n;
        sites.push_back(site);
    }
    return sites;
}

// The (width, distance) grid one mode-5 run reads out.
//
// Bins are stored width-major so that a single width is contiguous in x, which
// is the order the CSV is written and the analysis heatmaps expect. Not every
// bin is admissible -- the widest blocks run out of room first -- and the
// inadmissible ones are simply never filled.
struct ReceiverGrid
{
    std::vector<int> widths;
    int x_max = 0;

    std::size_t bin_count() const
    {
        return widths.size() * static_cast<std::size_t>(x_max + 1);
    }

    std::size_t bin(std::size_t width_index, int x) const
    {
        return width_index * static_cast<std::size_t>(x_max + 1) + static_cast<std::size_t>(x);
    }

    int width_at(std::size_t bin_index) const
    {
        return widths[bin_index / static_cast<std::size_t>(x_max + 1)];
    }

    int distance_at(std::size_t bin_index) const
    {
        return static_cast<int>(bin_index % static_cast<std::size_t>(x_max + 1));
    }
};

inline std::vector<int> parse_receiver_widths(const std::string &text)
{
    std::vector<int> widths;
    std::istringstream stream(text);
    std::string field;
    while (std::getline(stream, field, ','))
    {
        if (field.find_first_not_of(" \t") == std::string::npos)
        {
            continue;
        }
        int width = 0;
        try
        {
            width = std::stoi(field);
        }
        catch (const std::exception &)
        {
            throw std::invalid_argument("MIPT_PROBED_RECEIVER_WIDTHS must be a comma-separated list of integers.");
        }
        if (width < 1 || width > MAX_RECEIVER_WIDTH)
        {
            throw std::invalid_argument("Receiver-block widths must lie in [1, " +
                                        std::to_string(MAX_RECEIVER_WIDTH) + "].");
        }
        widths.push_back(width);
    }
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    if (widths.empty())
    {
        throw std::invalid_argument("MIPT_PROBED_RECEIVER_WIDTHS listed no widths.");
    }
    return widths;
}

// `x_max_request` of -1 means "as far as the ring allows".
inline ReceiverGrid make_receiver_grid(int n, int x_max_request)
{
    ReceiverGrid grid;
    grid.widths = parse_receiver_widths(env::text("MIPT_PROBED_RECEIVER_WIDTHS", "1,2,3"));
    const int reachable = n / 2;
    grid.x_max = x_max_request < 0 ? reachable : std::min(x_max_request, reachable);
    if (grid.x_max < 0)
    {
        throw std::invalid_argument("Mode 5 needs N >= 2 so at least one receiver distance exists.");
    }

    const bool any_admissible = std::any_of(grid.widths.begin(), grid.widths.end(), [&](int width) {
        return receiver_block_fits(n, width, 0);
    });
    if (!any_admissible)
    {
        throw std::invalid_argument("No receiver-block width fits in a ring of N=" + std::to_string(n) +
                                    "; lower MIPT_PROBED_RECEIVER_WIDTHS.");
    }
    return grid;
}

// ---------------------------------------------------------------------------
// Mode 5, two probes: the staggered (separation, insertion delay) grid
//
// Probe A goes in at x_0 after equilibration; probe B goes in `delay` timesteps
// later at x_0 + separation, and the pair is then watched for s timesteps. Both
// ancillas are isolated, so system-only unitaries cannot correlate them at all:
// whatever A--B correlation appears is generated by the intervening measurement
// record, and its arrival time as a function of (separation, delay) is what the
// protocol reads a velocity from.
//
// The grid is *not* sweepable inside one trajectory. Attaching a probe resets
// its partner site, so a different (separation, delay) is a different circuit
// history; each bin therefore needs its own trajectories and the workload
// multiplies by the bin count, exactly as three-probe mode 4 does over its
// triangle geometries.
// ---------------------------------------------------------------------------

// Bins are separation-major, so one separation is contiguous in delay -- the
// order the per-separation (delay, s) heatmaps are read back in.
struct StaggerGrid
{
    int r_min = 1;
    int r_max = 1;
    int delta_max = 0;

    std::size_t separation_count() const
    {
        return static_cast<std::size_t>(r_max - r_min + 1);
    }

    std::size_t delay_count() const
    {
        return static_cast<std::size_t>(delta_max + 1);
    }

    std::size_t bin_count() const
    {
        return separation_count() * delay_count();
    }

    std::size_t bin(int separation, int delay) const
    {
        return static_cast<std::size_t>(separation - r_min) * delay_count() + static_cast<std::size_t>(delay);
    }

    int separation_at(std::size_t bin_index) const
    {
        return r_min + static_cast<int>(bin_index / delay_count());
    }

    int delay_at(std::size_t bin_index) const
    {
        return static_cast<int>(bin_index % delay_count());
    }
};

// `r_max_request` and `delta_max_request` of -1 mean "use the default".
//
// The protocol asks for separations up to about L/4: beyond that the two probes
// are close to antipodal, both arcs of the ring carry signal, and `r` stops
// labelling a single propagation path. The delay window has to reach past the
// breakpoint Delta_c(r) = b' + lambda r, which for a light-cone velocity near
// one site per timestep means comfortably more than r_max; twice it is a safe
// default.
inline StaggerGrid make_stagger_grid(int n, int r_max_request, int delta_max_request)
{
    StaggerGrid grid;
    const int reachable = n / 2;
    if (reachable < 1)
    {
        throw std::invalid_argument("Two-probe mode 5 needs N >= 2 so a separation exists.");
    }
    grid.r_min = static_cast<int>(env::integer("MIPT_PROBED_R_MIN", 1, 1, reachable));
    grid.r_max = r_max_request < 0 ? std::max(2, n / 4) : r_max_request;
    grid.r_max = std::min(grid.r_max, reachable);
    if (grid.r_max < grid.r_min)
    {
        throw std::invalid_argument("Two-probe mode 5 requires r_max >= MIPT_PROBED_R_MIN and r_max <= N/2.");
    }
    grid.delta_max = delta_max_request < 0 ? 2 * grid.r_max : delta_max_request;
    if (grid.delta_max < 0)
    {
        throw std::invalid_argument("Two-probe mode 5 requires delta_max >= 0.");
    }
    return grid;
}

} // namespace mipt::probed
