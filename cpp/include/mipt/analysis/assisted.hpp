#pragma once

// Assisted (localizable) fermionic negativity between two endpoints.
// -------------------------------------------------------------------
//
// **The question.** A pair can be unentangled on its own and still be the two
// ends of entanglement that a measurement elsewhere would concentrate onto it.
// If assisted fN is positive where the unconditional pair RDM is separable,
// then what the percolation graph predicts is *localizable* connectivity -- a
// resource a helper measurement can convert -- rather than unconditional
// two-site entanglement. That is a different claim from the one P(C) is usually
// read as making, and it is testable here.
//
// **Superselection is not optional.** A physical fermionic measurement must
// commute with the global parity, because no operator that changes local
// fermion parity is measurable without an external parity reference. So:
//
//   * single-mode **occupation** measurements are allowed: the projectors
//     |0><0| and |1><1| are parity-even and diagonal, and they commute with
//     every Jordan-Wigner string the other modes carry;
//   * joint two-helper measurements are allowed *within* a parity sector --
//     (|00> +- |11>)/sqrt2 in the even sector and (|01> +- |10>)/sqrt2 in the
//     odd one. Each projector is parity-even, so the same commuting argument
//     holds and the naive tensor construction is exact;
//   * single-mode X or Y measurements are **not** allowed and are deliberately
//     absent. Their projectors, (|0> +- |1>)/sqrt2, mix local fermion parity;
//     using them would silently produce a "localizable entanglement" that no
//     fermionic experiment can realize, and the number would look perfectly
//     reasonable.
//
// The two families are reported separately rather than maximized over, because
// they are different experiments with different apparatus.

#include "mipt/analysis/cut_negativity.hpp"
#include "mipt/small_rdm.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace mipt::analysis
{

// What a family of helper measurements does to the endpoints' negativity.
struct AssistedNegativity
{
    bool evaluated = false;
    // Averaged over outcomes with their Born weights: the expected endpoint
    // negativity after the measurement. This is the localizable quantity.
    double average = 0.0;
    // The single most favourable outcome, which is what a post-selecting
    // experiment would see.
    double maximum = 0.0;
    // Total Born weight of the outcomes whose conditional negativity clears the
    // threshold -- the chance the measurement actually reveals entanglement.
    double positive_probability = 0.0;
    int outcomes = 0;
    // Which outcome was the best, as an index into the family.
    int best_outcome = -1;
    // The unconditional value, carried alongside so the comparison that matters
    // -- assisted against unassisted -- needs no second source.
    double unconditional = 0.0;
};

namespace detail
{

// The two-mode fermionic negativity of a 4x4 marginal, through the generic
// path. Cutting mode 0 of two modes needs no reordering sign, so this is the
// case the SSR convention is exactly trivial for.
inline double pair_negativity(const ancilla::SmallRdm &rho)
{
    std::array<double, 32> packed{};
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            packed[2u * static_cast<std::size_t>(r * 4 + c)] = rho(r, c).real();
            packed[2u * static_cast<std::size_t>(r * 4 + c) + 1u] = rho(r, c).imag();
        }
    }
    return cut_negativity<2>(packed.data(), 0, true);
}

} // namespace detail

// One measurement outcome: a rank-1 projector on the helper modes, given as
// amplitudes over the helper subspace's 2^|helpers| basis states.
struct MeasurementOutcome
{
    std::vector<std::complex<double>> amplitudes;
    std::string label;
};

// Single-mode occupation measurement: |0><0| and |1><1|.
inline std::vector<MeasurementOutcome> occupation_family()
{
    return {{{1.0, 0.0}, "n=0"}, {{0.0, 1.0}, "n=1"}};
}

// Joint two-helper measurement inside each parity sector. Every projector is
// parity-even, so every one of them is a legitimate fermionic measurement.
inline std::vector<MeasurementOutcome> parity_sector_family()
{
    const double s = 1.0 / std::sqrt(2.0);
    // Helper basis order is |n_k n_l> with n_k the low bit.
    return {
        {{s, 0.0, 0.0, s}, "even+"},   // (|00> + |11>)/sqrt2
        {{s, 0.0, 0.0, -s}, "even-"},  // (|00> - |11>)/sqrt2
        {{0.0, s, s, 0.0}, "odd+"},    // (|01> + |10>)/sqrt2
        {{0.0, s, -s, 0.0}, "odd-"},   // (|01> - |10>)/sqrt2
    };
}

// Apply a helper measurement family to `rho` and report what it does to the
// endpoints' negativity.
//
// `modes` is the total mode count, `endpoint_mask` the two endpoint modes and
// `helper_mask` the measured ones; they must be disjoint and together need not
// cover everything -- any remaining modes are traced out first.
//
// The projector is built as |v><v| on the helper modes tensored with the
// identity elsewhere. That is exact precisely because every projector in both
// families is parity-even and therefore commutes with the Jordan-Wigner strings
// the other modes carry; it would *not* be exact for a parity-mixing family,
// which is why none is offered.
inline AssistedNegativity assisted_negativity(const ancilla::SmallRdm &rho, int modes,
                                              unsigned endpoint_mask, unsigned helper_mask,
                                              const std::vector<MeasurementOutcome> &family,
                                              double positive_tol)
{
    AssistedNegativity out;
    const int dimension = 1 << modes;
    if (rho.dimension != dimension || (endpoint_mask & helper_mask) != 0u || family.empty())
    {
        return out;
    }
    const int helper_count = __builtin_popcount(helper_mask);
    const std::size_t helper_dimension = std::size_t{1} << helper_count;
    for (const MeasurementOutcome &outcome : family)
    {
        if (outcome.amplitudes.size() != helper_dimension)
        {
            return out;
        }
    }

    // Position of each helper mode within the helper subspace index, and the
    // same for everything that is not a helper.
    std::vector<int> helper_bit(static_cast<std::size_t>(modes), -1);
    int next = 0;
    for (int m = 0; m < modes; ++m)
    {
        if ((helper_mask >> m) & 1u)
        {
            helper_bit[static_cast<std::size_t>(m)] = next++;
        }
    }
    auto helper_index = [&](int x) {
        int index = 0;
        for (int m = 0; m < modes; ++m)
        {
            if (helper_bit[static_cast<std::size_t>(m)] >= 0 && ((x >> m) & 1))
            {
                index |= 1 << helper_bit[static_cast<std::size_t>(m)];
            }
        }
        return index;
    };

    out.evaluated = true;
    {
        const ancilla::SmallRdm marginal =
            ancilla::partial_trace_fixed(rho, modes, endpoint_mask, true);
        out.unconditional = detail::pair_negativity(marginal);
    }

    double weight_total = 0.0;
    for (std::size_t o = 0; o < family.size(); ++o)
    {
        const MeasurementOutcome &outcome = family[o];
        // P rho P, with P = |v><v| on the helpers and the identity elsewhere.
        ancilla::SmallRdm projected(dimension);
        for (int r = 0; r < dimension; ++r)
        {
            const std::complex<double> vr =
                outcome.amplitudes[static_cast<std::size_t>(helper_index(r))];
            if (vr == std::complex<double>(0.0, 0.0))
            {
                continue;
            }
            for (int c = 0; c < dimension; ++c)
            {
                const std::complex<double> vc =
                    outcome.amplitudes[static_cast<std::size_t>(helper_index(c))];
                if (vc == std::complex<double>(0.0, 0.0))
                {
                    continue;
                }
                // <r| P rho P |c> = v_r conj(v_c) sum_{a,b} conj(v_a) v_b rho[a,b]
                // over the states a, b agreeing with r, c outside the helpers.
                std::complex<double> sum(0.0, 0.0);
                for (int a = 0; a < dimension; ++a)
                {
                    if ((a & ~helper_mask) != (r & ~static_cast<int>(helper_mask)))
                    {
                        continue;
                    }
                    const std::complex<double> va =
                        outcome.amplitudes[static_cast<std::size_t>(helper_index(a))];
                    if (va == std::complex<double>(0.0, 0.0))
                    {
                        continue;
                    }
                    for (int b = 0; b < dimension; ++b)
                    {
                        if ((b & ~helper_mask) != (c & ~static_cast<int>(helper_mask)))
                        {
                            continue;
                        }
                        const std::complex<double> vb =
                            outcome.amplitudes[static_cast<std::size_t>(helper_index(b))];
                        sum += std::conj(va) * vb * rho(a, b);
                    }
                }
                projected(r, c) = vr * std::conj(vc) * sum;
            }
        }
        double weight = 0.0;
        for (int d = 0; d < dimension; ++d)
        {
            weight += projected(d, d).real();
        }
        if (!(weight > 1.0e-14))
        {
            // An outcome of vanishing probability contributes nothing and its
            // conditional state is not defined. Counting it would be dividing
            // by noise.
            continue;
        }
        for (auto &value : projected.values)
        {
            value /= weight;
        }
        const ancilla::SmallRdm marginal =
            ancilla::partial_trace_fixed(projected, modes, endpoint_mask, true);
        const double negativity = detail::pair_negativity(marginal);
        if (!(negativity == negativity))
        {
            continue;
        }
        ++out.outcomes;
        weight_total += weight;
        out.average += weight * negativity;
        if (negativity > out.maximum)
        {
            out.maximum = negativity;
            out.best_outcome = static_cast<int>(o);
        }
        if (negativity > positive_tol)
        {
            out.positive_probability += weight;
        }
    }
    if (weight_total > 0.0)
    {
        // Renormalized by the weight actually accounted for, so a family that
        // lost a vanishing outcome still reports a conditional average rather
        // than one silently scaled down.
        out.average /= weight_total;
        out.positive_probability /= weight_total;
    }
    return out;
}

} // namespace mipt::analysis
