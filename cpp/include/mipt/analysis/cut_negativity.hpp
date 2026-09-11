#pragma once

// One-party-versus-rest negativities of a small multi-mode density matrix, in
// either trace convention.
//
// This is the single implementation behind fgmn.cpp's GMN prefilter and
// dist_scaling.exe's connected-zero triple analysis. It used to live only
// inside fgmn.cpp, which is a MOSEK translation unit, so anything that wanted a
// cut negativity without linking MOSEK had to grow its own copy -- exactly the
// situation that once left three diverging Jacobi solvers in this tree (see
// CLAUDE.md, "Bipartite negativity before 2026-07-29"). It is header-only and
// dependency-free so `make test-dist` can pin it on the host.
//
// Conventions, unchanged from fgmn.cpp:
//
//   * `rho_ri` is interleaved row-major complex, `2 * D * D` doubles.
//   * Party `s` is basis bit `s` (little-endian retained-mode order), so the
//     cut "s versus rest" transposes bit `s`.
//   * The fermionic partial transpose multiplies every entry whose row and
//     column disagree in the transposed party's occupation by i. That matrix is
//     not Hermitian, so its trace norm goes through the Hermitian dilation; the
//     ordinary partial transpose stays Hermitian and uses the absolute
//     eigenvalues.
//   * A result in (-1e-10, 0) is clamped to 0; anything more negative is
//     returned as computed, so a caller can see it.
//
// **The cut mode is moved to the front of the operator ordering first.** The
// Shapourian-Shiozaki-Ryu fermionic partial transpose is defined for a basis in
// which the transposed modes precede the rest. A basis state here is
// c_0^n0 c_1^n1 c_2^n2 |vac> in ascending mode order, so for a cut on mode s
// every occupied mode below s has to be anticommuted past it: the matrix is
// conjugated by S(x) = (-1)^{n_s(x) * (occupied modes below s)} before the
// transpose. Without it the result depends on how the sites are *labelled*.
//
// That was a real defect, found on 2026-09-11 by the first test that relabelled
// the three modes: the middle mode's cut came out wrong (0.282430 against the
// correct 0.282292 on one random state) and fGMN, which shares the convention,
// moved by up to 0.02 under a relabelling. For a parity-preserving state -- every
// RDM a definite-parity trajectory can produce -- the sign is exactly trivial for
// the first and last modes, which is why every two-mode negativity in the tree
// and the mode-5 reference cut (always the highest mode) were never affected,
// and only interior modes move. Verified against an independent implementation
// of the SSR definition: exact to 4e-16, and label-independent to 4e-16.

#include "mipt/util/spectral.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <limits>

namespace mipt::analysis
{

template <int Parties>
inline double cut_negativity(const double *rho_ri, int subsystem, bool fermionic)
{
    constexpr int D = 1 << Parties;
    if (rho_ri == nullptr || subsystem < 0 || subsystem >= Parties)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int mask = 1 << subsystem;
    const int below = mask - 1;
    // (-1)^{n_s * (occupied modes below s)}: the sign of moving mode s to the
    // front of the operator ordering. Trivial for s = 0.
    auto reorder_sign = [&](int x) {
        if ((x & mask) == 0)
        {
            return 1.0;
        }
        return (__builtin_popcount(static_cast<unsigned>(x & below)) & 1) ? -1.0 : 1.0;
    };

    std::array<std::complex<double>, static_cast<std::size_t>(D * D)> pt{};
    for (int r = 0; r < D; ++r)
    {
        for (int c = 0; c < D; ++c)
        {
            const int source_row = (r & ~mask) | (c & mask);
            const int source_col = (c & ~mask) | (r & mask);
            const std::size_t source = 2u * static_cast<std::size_t>(source_row * D + source_col);
            std::complex<double> value(rho_ri[source], rho_ri[source + 1u]);
            if (fermionic)
            {
                value *= reorder_sign(source_row) * reorder_sign(source_col);
            }
            const bool parity_violation = ((r & mask) != 0) ^ ((c & mask) != 0);
            if (fermionic && parity_violation)
            {
                value = std::complex<double>(-value.imag(), value.real());
            }
            pt[static_cast<std::size_t>(r * D + c)] = value;
        }
    }

    // The fermionic partial transpose is not Hermitian, so its trace norm goes
    // through the Hermitian dilation rather than the Gram matrix: the Gram route
    // puts a ~1e-9 floor under every cut, above the 1e-10 prefilter tolerance,
    // and an exactly-separable cut could then never be certified. See
    // util::dilation_trace_norm.
    double trace_norm = 0.0;
    const bool converged =
        fermionic ? util::dilation_trace_norm<static_cast<std::size_t>(D)>(pt, trace_norm)
                  : util::hermitian_trace_norm<static_cast<std::size_t>(D)>(pt, trace_norm);
    if (!converged)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double negativity = 0.5 * (trace_norm - 1.0);
    if (negativity < 0.0 && negativity > -1.0e-10)
    {
        negativity = 0.0;
    }
    return negativity;
}

// All `Parties` one-versus-rest negativities. Returns false if any of them
// failed to converge; the failed entries are NaN, never zero, because a zero
// here would certify a GMN of zero to every caller that prefilters on it.
template <int Parties>
inline bool cut_negativities(const double *rho_ri, bool fermionic,
                             std::array<double, static_cast<std::size_t>(Parties)> &out)
{
    bool ok = true;
    for (int subsystem = 0; subsystem < Parties; ++subsystem)
    {
        out[static_cast<std::size_t>(subsystem)] =
            cut_negativity<Parties>(rho_ri, subsystem, fermionic);
        ok = ok && out[static_cast<std::size_t>(subsystem)] ==
                       out[static_cast<std::size_t>(subsystem)];
    }
    return ok;
}

} // namespace mipt::analysis
