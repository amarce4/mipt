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

    std::cout << "dist_metrics_tests: PASS\n";
    return 0;
}
