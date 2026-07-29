#pragma once

// Ring geometry shared by the distance-scaling and probe workflows.
//
// Sites live on a periodic chain of N sites. The conformal chord length
//
//     l(d) = (N/pi) sin(pi d / N)
//
// is the distance that entanglement quantities actually scale with at
// criticality, so it — not the lattice separation — is what the analysis
// plots against.

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace mipt::util
{

// Shorter of the two ways round the ring.
inline int periodic_distance(int lhs, int rhs, int n)
{
    const int direct = std::abs(lhs - rhs);
    return std::min(direct, n - direct);
}

inline double chord_length(int n, int separation)
{
    if (n < 2 || separation <= 0 || separation >= n)
    {
        throw std::invalid_argument("Chord length requires 0 < separation < N.");
    }
    const double pi = std::acos(-1.0);
    return static_cast<double>(n) / pi * std::sin(pi * static_cast<double>(separation) / static_cast<double>(n));
}

// How close a probe triangle is to equilateral in chord distance: 1 is
// equilateral, and values near 0 are the degenerate collinear triangles that
// the three-probe protocol filters out.
inline double triangle_balance(const std::array<int, 3> &distances, int n)
{
    std::array<double, 3> chords{};
    std::transform(distances.begin(), distances.end(), chords.begin(),
                   [n](int distance) { return chord_length(n, distance); });
    const auto [minimum, maximum] = std::minmax_element(chords.begin(), chords.end());
    return *minimum / *maximum;
}

} // namespace mipt::util
