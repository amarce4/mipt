#pragma once

// Validated parameter grids for CLI scans.
//
// Deliberately distinct from mipt::linspace in types.hpp: that one is lenient
// (an empty request yields an empty grid) and is used for internal sweeps,
// while these reject bad user input, pin both endpoints exactly, and are what
// generates the p and t values written into output filenames and CSV rows.
// Do not merge them — the endpoint pinning is what lets a resumed scan match
// the p values already present in a checkpoint.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mipt::util
{

inline std::vector<double> checked_linspace(double minimum, double maximum, int count)
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
        values[static_cast<std::size_t>(i)] =
            minimum + (maximum - minimum) * static_cast<double>(i) / static_cast<double>(count - 1);
    }
    values.front() = minimum;
    values.back() = maximum;
    return values;
}

inline std::vector<int> integer_linspace(int minimum, int maximum, int count)
{
    const auto floating = checked_linspace(static_cast<double>(minimum), static_cast<double>(maximum), count);
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
        throw std::invalid_argument("t_res requests duplicate integer times; "
                                    "require t_res <= t_max-t_min+1.");
    }
    return values;
}

} // namespace mipt::util
