#include "mipt/dist_metrics.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>

namespace
{
using C = std::complex<double>;
using Rho = std::array<C, 16>;

std::array<double, 32> interleaved(const Rho &rho)
{
    std::array<double, 32> out{};
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        out[2 * i] = rho[i].real();
        out[2 * i + 1] = rho[i].imag();
    }
    return out;
}

Rho pure_density(const std::array<C, 4> &psi)
{
    Rho rho{};
    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            rho[i * 4 + j] = psi[i] * std::conj(psi[j]);
        }
    }
    return rho;
}

Rho fermionic_swap(const Rho &rho)
{
    constexpr std::array<int, 4> permutation{0, 2, 1, 3};
    constexpr std::array<int, 4> sign{1, 1, 1, -1};
    Rho out{};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            out[permutation[i] * 4 + permutation[j]] =
                static_cast<double>(sign[i] * sign[j]) * rho[i * 4 + j];
        }
    }
    return out;
}

void require_close(double actual, double expected, double tolerance, const char *label)
{
    if (std::abs(actual - expected) > tolerance)
    {
        std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
        std::exit(1);
    }
}

// --- three-party helpers ---------------------------------------------------

using Rho3 = std::array<C, 64>;

std::array<double, 128> interleaved3(const Rho3 &rho)
{
    std::array<double, 128> out{};
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        out[2 * i] = rho[i].real();
        out[2 * i + 1] = rho[i].imag();
    }
    return out;
}

Rho3 pure_density3(const std::array<C, 8> &psi)
{
    Rho3 rho{};
    for (std::size_t i = 0; i < 8; ++i)
    {
        for (std::size_t j = 0; j < 8; ++j)
        {
            rho[i * 8 + j] = psi[i] * std::conj(psi[j]);
        }
    }
    return rho;
}

Rho3 mix3(const Rho3 &lhs, const Rho3 &rhs, double weight)
{
    Rho3 out{};
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        out[i] = weight * lhs[i] + (1.0 - weight) * rhs[i];
    }
    return out;
}

// Exchange fermionic modes 0 and 1 of a three-mode block: swap bits 0 and 1 of
// every basis index and pick up the anticommutation sign when both are
// occupied. Every measure computed with the fermionic trace has to be blind to
// this, exactly as the two-mode version above.
Rho3 fermionic_swap3(const Rho3 &rho)
{
    std::array<int, 8> permutation{};
    std::array<int, 8> sign{};
    for (int index = 0; index < 8; ++index)
    {
        const int bit0 = index & 1;
        const int bit1 = (index >> 1) & 1;
        permutation[index] = (index & ~3) | (bit1 << 0) | (bit0 << 1);
        sign[index] = (bit0 && bit1) ? -1 : 1;
    }

    Rho3 out{};
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            out[permutation[i] * 8 + permutation[j]] =
                static_cast<double>(sign[i] * sign[j]) * rho[i * 8 + j];
        }
    }
    return out;
}
} // namespace

int main()
{
    const auto product = interleaved(pure_density({C(1.0, 0.0), {}, {}, {}}));
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::two_party_metrics(product.data(), fermionic);
        require_close(metrics.mi, 0.0, 1.0e-10, "product MI");
        require_close(metrics.mn, 0.0, 1.0e-10, "product negativity");
    }

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const Rho bell_rho = pure_density({C(inv_sqrt2, 0.0), {}, {}, C(inv_sqrt2, 0.0)});
    const auto bell = interleaved(bell_rho);
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::two_party_metrics(bell.data(), fermionic);
        require_close(metrics.mi, 2.0, 1.0e-9, "Bell MI");
        require_close(metrics.mn, 0.5, 1.0e-9, "Bell negativity");
    }

    const auto swapped = interleaved(fermionic_swap(bell_rho));
    const auto original_metrics = mipt::dist::two_party_metrics(bell.data(), true);
    const auto swapped_metrics = mipt::dist::two_party_metrics(swapped.data(), true);
    require_close(swapped_metrics.mi, original_metrics.mi, 1.0e-9, "swap-invariant fermionic MI");
    require_close(swapped_metrics.mn, original_metrics.mn, 1.0e-9, "swap-invariant fermionic negativity");

    const Rho classical = [] {
        Rho rho{};
        rho[0] = C(0.5, 0.0);
        rho[15] = C(0.5, 0.0);
        return rho;
    }();
    const auto classical_ri = interleaved(classical);
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::two_party_metrics(classical_ri.data(), fermionic);
        require_close(metrics.mi, 1.0, 1.0e-9, "classical-correlation MI");
        require_close(metrics.mn, 0.0, 1.0e-9, "classical-correlation negativity");
    }

    const Rho even = pure_density({C(std::sqrt(0.7), 0.0), {}, {},
                                   C(0.0, std::sqrt(0.3))});
    const Rho odd = pure_density({C{}, C(std::sqrt(0.2), 0.0),
                                  C(std::sqrt(0.8), 0.0), C{}});
    Rho parity_preserving_mixed{};
    for (std::size_t i = 0; i < parity_preserving_mixed.size(); ++i)
    {
        parity_preserving_mixed[i] = 0.37 * even[i] + 0.63 * odd[i];
    }
    const auto mixed_ri = interleaved(parity_preserving_mixed);
    const auto mixed_swapped_ri = interleaved(fermionic_swap(parity_preserving_mixed));
    const auto mixed_metrics = mipt::dist::two_party_metrics(mixed_ri.data(), true);
    const auto mixed_swapped_metrics =
        mipt::dist::two_party_metrics(mixed_swapped_ri.data(), true);
    require_close(mixed_swapped_metrics.mi, mixed_metrics.mi, 1.0e-9,
                  "mixed-state swap-invariant fermionic MI");
    require_close(mixed_swapped_metrics.mn, mixed_metrics.mn, 1.0e-9,
                  "mixed-state swap-invariant fermionic negativity");

    require_close(mipt::dist::two_party_geometric_chord_distance(4, 0, 1),
                  2.0 * std::sqrt(2.0) / std::acos(-1.0), 1.0e-12,
                  "N=4 nearest-neighbour chord distance");
    require_close(mipt::dist::two_party_geometric_chord_distance(4, 0, 2),
                  4.0 / std::acos(-1.0), 1.0e-12,
                  "N=4 antipodal chord distance");

    // --- three-party metrics ------------------------------------------------

    const auto product3 = interleaved3(pure_density3({C(1.0, 0.0)}));
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::three_party_metrics(product3.data(), fermionic);
        require_close(metrics.tmi, 0.0, 1.0e-9, "product TMI");
        require_close(metrics.average_mi, 0.0, 1.0e-9, "product average MI");
        require_close(metrics.joint_purity, 1.0, 1.0e-12, "product joint purity");
        require_close(metrics.mean_single_purity, 1.0, 1.0e-12, "product single purity");
    }

    // Any pure three-party state has I_3 = 0, because S_AB = S_C and so on.
    // GHZ and W therefore test the entropy bookkeeping, not the combination.
    const Rho3 ghz = pure_density3({C(inv_sqrt2, 0.0), {}, {}, {}, {}, {}, {}, C(inv_sqrt2, 0.0)});
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::three_party_metrics(interleaved3(ghz).data(), fermionic);
        require_close(metrics.tmi, 0.0, 1.0e-9, "GHZ TMI");
        require_close(metrics.average_mi, 1.0, 1.0e-9, "GHZ average MI");
        require_close(metrics.joint_purity, 1.0, 1.0e-9, "GHZ joint purity");
        require_close(metrics.mean_single_purity, 0.5, 1.0e-9, "GHZ single purity");
    }

    const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
    const Rho3 w_state =
        pure_density3({C{}, C(inv_sqrt3, 0.0), C(inv_sqrt3, 0.0), C{}, C(inv_sqrt3, 0.0), C{}, C{}, C{}});
    {
        const double single = -(1.0 / 3.0) * std::log2(1.0 / 3.0) - (2.0 / 3.0) * std::log2(2.0 / 3.0);
        const auto metrics = mipt::dist::three_party_metrics(interleaved3(w_state).data(), false);
        require_close(metrics.tmi, 0.0, 1.0e-9, "W TMI");
        require_close(metrics.average_mi, single, 1.0e-9, "W average MI");
    }

    // A classically correlated GHZ mixture is the simplest state with a
    // nonzero tripartite information: every marginal and every pair carries
    // one bit, and so does the whole block, leaving I_3 = 1 bit.
    const Rho3 classical_ghz =
        mix3(pure_density3({C(1.0, 0.0)}), pure_density3({C{}, C{}, C{}, C{}, C{}, C{}, C{}, C(1.0, 0.0)}), 0.5);
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::three_party_metrics(interleaved3(classical_ghz).data(), fermionic);
        require_close(metrics.tmi, 1.0, 1.0e-9, "classical-GHZ TMI");
        require_close(metrics.average_mi, 1.0, 1.0e-9, "classical-GHZ average MI");
        require_close(metrics.joint_purity, 0.5, 1.0e-9, "classical-GHZ joint purity");
        require_close(metrics.mean_single_purity, 0.5, 1.0e-9, "classical-GHZ single purity");
    }

    // A Bell pair on (A,B) with C idle: I_3 = 0 but the pairwise mutual
    // informations are 2, 0, 0, which is what separates `average_mi` from a
    // measure that only sees the symmetric combination.
    const Rho3 bell_ab = pure_density3({C(inv_sqrt2, 0.0), {}, {}, C(inv_sqrt2, 0.0), {}, {}, {}, {}});
    {
        const auto metrics = mipt::dist::three_party_metrics(interleaved3(bell_ab).data(), false);
        require_close(metrics.tmi, 0.0, 1.0e-9, "Bell-pair-plus-idle TMI");
        require_close(metrics.average_mi, 2.0 / 3.0, 1.0e-9, "Bell-pair-plus-idle average MI");
    }

    // Fermionic-trace metrics must be invariant under exchanging two modes
    // together with the anticommutation sign that exchange carries.
    const Rho3 parity_mixed3 = mix3(ghz, classical_ghz, 0.41);
    const auto parity_metrics = mipt::dist::three_party_metrics(interleaved3(parity_mixed3).data(), true);
    const auto parity_swapped_metrics =
        mipt::dist::three_party_metrics(interleaved3(fermionic_swap3(parity_mixed3)).data(), true);
    require_close(parity_swapped_metrics.tmi, parity_metrics.tmi, 1.0e-9,
                  "swap-invariant fermionic TMI");
    require_close(parity_swapped_metrics.average_mi, parity_metrics.average_mi, 1.0e-9,
                  "swap-invariant fermionic average MI");

    // A non-unit trace has to be divided out, not carried into the entropies.
    Rho3 unnormalized = classical_ghz;
    for (auto &value : unnormalized)
    {
        value *= 3.7;
    }
    const auto unnormalized_metrics =
        mipt::dist::three_party_metrics(interleaved3(unnormalized).data(), false);
    require_close(unnormalized_metrics.tmi, 1.0, 1.0e-9, "unnormalized-input TMI");

    // Triangle geometry: an equilateral triangle on N=6 has all three chords
    // equal, so their geometric mean is that chord.
    const double pi = std::acos(-1.0);
    require_close(mipt::dist::three_party_geometric_chord_distance(6, 0, 2, 4),
                  6.0 / pi * std::sin(2.0 * pi / 6.0), 1.0e-12,
                  "N=6 equilateral triangle chord distance");
    const auto separations = mipt::dist::triangle_separations(6, 0, 1, 3);
    if (separations != std::array<int, 3>{1, 2, 3})
    {
        std::cerr << "triangle_separations(6,0,1,3): expected {1,2,3}, got {" << separations[0] << ','
                  << separations[1] << ',' << separations[2] << "}\n";
        std::exit(1);
    }

    std::cout << "dist_metrics_tests: PASS\n";
    return 0;
}
