#pragma once

// Multi-probe information measures built on the ancilla reduced density
// matrices.  The generic small-block partial trace and entropy they use live in
// small_rdm.hpp, which carries no CUDA-Q dependency.

#include "mipt/ancilla_mi.hpp"
#include "mipt/small_rdm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mipt::ancilla
{
struct ProbeInformation
{
    double entropy = 0.0;
    double i2 = 0.0;
    double i3 = 0.0;
    double i4 = 0.0;
};

struct ThreeProbeInformation
{
    double tmi = 0.0;
    double average_mi = 0.0;
    double joint_purity = 0.0;
    double mean_single_purity = 0.0;
};

inline SmallRdm partial_trace_three(const Rdm3 &rho, unsigned retained_mask, bool fermion_trace)
{
    return partial_trace_fixed(rho, 3, retained_mask, fermion_trace);
}

inline SmallRdm partial_trace_four(const Rdm4 &rho, unsigned retained_mask, bool fermion_trace)
{
    return partial_trace_fixed(rho, 4, retained_mask, fermion_trace);
}

inline ThreeProbeInformation three_probe_information(const Rdm3 &rho, bool fermion_trace)
{
    std::array<SmallRdm, 8> reduced{SmallRdm(1), SmallRdm(2), SmallRdm(2), SmallRdm(4),
                                    SmallRdm(2), SmallRdm(4), SmallRdm(4), SmallRdm(8)};
    std::array<double, 8> entropies{};
    for (unsigned mask = 1; mask < 8; ++mask)
    {
        reduced[mask] = partial_trace_three(rho, mask, fermion_trace);
        entropies[mask] = entropy_from_small_rdm(reduced[mask]);
    }

    const double i_ab = entropies[1] + entropies[2] - entropies[3];
    const double i_ac = entropies[1] + entropies[4] - entropies[5];
    const double i_bc = entropies[2] + entropies[4] - entropies[6];
    const double tmi =
        entropies[1] + entropies[2] + entropies[4] - entropies[3] - entropies[5] - entropies[6] + entropies[7];
    const double mean_single_purity =
        (purity_from_small_rdm(reduced[1]) + purity_from_small_rdm(reduced[2]) + purity_from_small_rdm(reduced[4])) /
        3.0;
    return {tmi, (i_ab + i_ac + i_bc) / 3.0, purity_from_small_rdm(reduced[7]), mean_single_purity};
}

inline ThreeProbeInformation three_probe_information(cudaq::state &state, int n_qubits,
                                                     const std::array<int, 3> &references, bool fermion_trace)
{
    return three_probe_information(reduced_density_matrix_three(state, n_qubits, references, fermion_trace),
                                   fermion_trace);
}

inline ProbeInformation four_probe_information(cudaq::state &state, int n_qubits, const std::array<int, 4> &references,
                                               bool fermion_trace)
{
    const Rdm4 rho = reduced_density_matrix_four(state, n_qubits, references, fermion_trace);
    std::array<double, 16> entropies{};
    for (unsigned mask = 1; mask < 16; ++mask)
    {
        entropies[mask] = entropy_from_small_rdm(partial_trace_four(rho, mask, fermion_trace));
    }

    double i2_sum = 0.0;
    double i3_sum = 0.0;
    int pairs = 0;
    int triplets = 0;
    double i4 = 0.0;
    for (unsigned mask = 1; mask < 16; ++mask)
    {
#if defined(__GNUG__) || defined(__clang__)
        const int count = __builtin_popcount(mask);
#else
        int count = 0;
        for (unsigned value = mask; value; value &= value - 1)
        {
            ++count;
        }
#endif
        i4 += (count % 2 == 1 ? 1.0 : -1.0) * entropies[mask];
        if (count == 2)
        {
            double pair_information = -entropies[mask];
            for (int bit = 0; bit < 4; ++bit)
            {
                if ((mask >> bit) & 1u)
                {
                    pair_information += entropies[1u << bit];
                }
            }
            i2_sum += pair_information;
            ++pairs;
        }
        else if (count == 3)
        {
            double tripartite = entropies[mask];
            for (unsigned subset = mask; subset; subset = (subset - 1) & mask)
            {
#if defined(__GNUG__) || defined(__clang__)
                const int subset_count = __builtin_popcount(subset);
#else
                int subset_count = 0;
                for (unsigned value = subset; value; value &= value - 1)
                {
                    ++subset_count;
                }
#endif
                if (subset_count == 1)
                {
                    tripartite += entropies[subset];
                }
                else if (subset_count == 2)
                {
                    tripartite -= entropies[subset];
                }
            }
            i3_sum += tripartite;
            ++triplets;
        }
    }
    return {entropies[15], i2_sum / static_cast<double>(pairs), i3_sum / static_cast<double>(triplets), i4};
}

inline ProbeInformation two_probe_information(cudaq::state &state, int n_qubits, int reference_a, int reference_b,
                                              bool fermion_trace)
{
    const auto values = two_probe_observables(state, n_qubits, reference_a, reference_b, fermion_trace);
    return {values.joint_entropy, values.mutual_information, 0.0, 0.0};
}
} // namespace mipt::ancilla
