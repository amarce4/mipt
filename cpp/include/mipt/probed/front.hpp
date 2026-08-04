#pragma once

// Mode-5 readout: how far the injected reference's signal has spread.
//
// After the probe is Bell-entangled with a random site x_0, every elapsed
// timestep is read out on the receiver blocks B_w(x) defined in geometry.hpp.
// This file is the bridge between the state vector and the correlation measures
// in front_metrics.hpp: it pulls out the joint (block, reference) density
// matrix and hands it over.
//
// The cost model is what shapes the code. Every RDM extraction is one pass over
// the 2^(N+1) amplitudes, and the grid has O(N) distances times two directions
// times the number of widths at every readout time, so the pass count is what
// this protocol pays for. Two structural savings keep it linear rather than
// cubic in the grid:
//
//   * B_1(x) nests inside B_2(x) nests inside B_3(x) (see geometry.hpp), so one
//     RDM over the widest admissible block yields all the narrower widths by a
//     16x16 partial trace instead of a fresh state-vector pass each.
//   * The widest block is chosen per distance, so distances near the antipode,
//     where only w=1 fits, extract a 4x4 rather than a 16x16.
//
// That leaves 2*(x_max+1) passes per readout time, independent of how many
// widths are requested.
//
// The two-probe variant of mode 5 shares this file because it shares the layout
// convention rather than the cost model: it extracts a single 4x4 matrix over
// the two isolated ancillas per readout time, and hands it to the same width-1
// metrics with probe A playing the reference. See measure_stagger_sample.

#include "mipt/ancilla_info.hpp"
#include "mipt/ancilla_rdm.hpp"
#include "mipt/front_metrics.hpp"
#include "mipt/probed/geometry.hpp"
#include "mipt/probed/sample.hpp"
#include "mipt/small_rdm.hpp"

#include <cudaq.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mipt::probed
{

namespace detail
{

// The joint density matrix of `sites` and the reference, with the reference as
// the highest local mode.
//
// Passing the retained modes in this order is what fixes the local basis: bit k
// of the result is `sites[k]` and the top bit is the reference, and for a
// fermionic trace the Jordan--Wigner reordering signs are computed for exactly
// that order. The block's internal ordering therefore differs between the two
// directions (it always runs outward from x_0), but a signed permutation acting
// only inside B leaves every quantity in front_metrics.hpp invariant, so the
// two directions stay comparable.
inline ancilla::SmallRdm receiver_reference_rdm(cudaq::state &state, int total_qubits,
                                                const std::vector<int> &sites, int reference, bool fermionic)
{
    const std::size_t width = sites.size();
    ancilla::SmallRdm joint(1 << (width + 1));

    if (width == 1)
    {
        const auto rho = ancilla::reduced_density_matrix_two(state, total_qubits, sites[0], reference, fermionic);
        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                joint(row, col) = rho(static_cast<std::size_t>(row), static_cast<std::size_t>(col));
            }
        }
        return joint;
    }
    if (width == 2)
    {
        const auto rho = ancilla::reduced_density_matrix_three(
            state, total_qubits, {sites[0], sites[1], reference}, fermionic);
        for (int row = 0; row < 8; ++row)
        {
            for (int col = 0; col < 8; ++col)
            {
                joint(row, col) = rho(static_cast<std::size_t>(row), static_cast<std::size_t>(col));
            }
        }
        return joint;
    }
    if (width == 3)
    {
        const auto rho = ancilla::reduced_density_matrix_four(
            state, total_qubits, {sites[0], sites[1], sites[2], reference}, fermionic);
        for (int row = 0; row < 16; ++row)
        {
            for (int col = 0; col < 16; ++col)
            {
                joint(row, col) = rho(static_cast<std::size_t>(row), static_cast<std::size_t>(col));
            }
        }
        return joint;
    }
    throw std::invalid_argument("Receiver blocks support one to three sites.");
}

// Drop the outer block modes, keeping the innermost `width` of them plus the
// reference. The retained modes keep their relative order, so the reference
// lands on bit `width` again, which is what block_metrics expects.
inline ancilla::SmallRdm narrow_receiver_rdm(const ancilla::SmallRdm &joint, int full_width, int width, bool fermionic)
{
    if (width == full_width)
    {
        return joint;
    }
    const unsigned mask = ((1u << width) - 1u) | (1u << full_width);
    return ancilla::partial_trace_fixed(joint, full_width + 1, mask, fermionic);
}

inline Sample sample_from_metrics(const front::BlockMetrics &left, const front::BlockMetrics &right)
{
    Sample sample;
    sample.sq = 0.5 * (left.joint_entropy + right.joint_entropy);
    sample.i2 = 0.5 * (left.mutual_information + right.mutual_information);
    sample.negativity = 0.5 * (left.negativity + right.negativity);
    sample.log_negativity = 0.5 * (left.log_negativity + right.log_negativity);
    sample.classical_x = 0.5 * (left.classical_x + right.classical_x);
    sample.classical_y = 0.5 * (left.classical_y + right.classical_y);
    sample.classical_z = 0.5 * (left.classical_z + right.classical_z);
    sample.classical_mean = 0.5 * (left.classical_mean + right.classical_mean);
    sample.classical_max = 0.5 * (left.classical_max + right.classical_max);
    sample.coherent_information = 0.5 * (left.coherent_information + right.coherent_information);
    sample.block_entropy = 0.5 * (left.block_entropy + right.block_entropy);
    return sample;
}

} // namespace detail

// One readout time: fill `samples[bin]` for every admissible (width, distance)
// and mark it in `filled`.
//
// Both are sized to `grid.bin_count()` by the caller and reused across readout
// times so the loop allocates nothing.
inline void measure_receiver_blocks(cudaq::state &state, int n, int origin, const ReceiverGrid &grid, bool fermionic,
                                    std::vector<Sample> &samples, std::vector<unsigned char> &filled)
{
    if (samples.size() != grid.bin_count() || filled.size() != grid.bin_count())
    {
        throw std::logic_error("Receiver-block sample buffers do not match the grid.");
    }
    const int reference = n;
    const int total_qubits = n + 1;
    std::fill(filled.begin(), filled.end(), 0u);

    // [side][width - 1]; only entries below the widest admissible block are used.
    std::array<std::array<front::BlockMetrics, MAX_RECEIVER_WIDTH>, 2> measured{};

    for (int x = 0; x <= grid.x_max; ++x)
    {
        int widest = 0;
        for (const int width : grid.widths)
        {
            if (receiver_block_fits(n, width, x))
            {
                widest = std::max(widest, width);
            }
        }
        if (widest == 0)
        {
            continue;
        }

        for (int side = 0; side < 2; ++side)
        {
            const int direction = side == 0 ? 1 : -1;
            const auto sites = receiver_block_sites(n, origin, x, widest, direction);
            const auto joint = detail::receiver_reference_rdm(state, total_qubits, sites, reference, fermionic);
            for (int width = 1; width <= widest; ++width)
            {
                measured[static_cast<std::size_t>(side)][static_cast<std::size_t>(width - 1)] = front::block_metrics(
                    detail::narrow_receiver_rdm(joint, widest, width, fermionic), width, fermionic);
            }
        }

        for (std::size_t width_index = 0; width_index < grid.widths.size(); ++width_index)
        {
            const int width = grid.widths[width_index];
            if (!receiver_block_fits(n, width, x))
            {
                continue;
            }
            const std::size_t index = grid.bin(width_index, x);
            samples[index] = detail::sample_from_metrics(measured[0][static_cast<std::size_t>(width - 1)],
                                                         measured[1][static_cast<std::size_t>(width - 1)]);
            filled[index] = 1u;
        }
    }
}

// Mode 5, two probes: every correlation measure of the two isolated ancillas.
//
// Ancilla A occupies qubit N and B occupies qubit N+1. A is the *first*
// insertion and therefore the measured party of the directed classical
// correlation J_{A->B}, so it has to land where front_metrics.hpp looks for the
// reference: the highest local mode. Retaining {B, A} in that order puts B on
// bit 0 and A on bit 1, which is exactly the width-1 layout block_metrics
// expects -- the same convention, reached by naming B the "receiver block".
inline Sample measure_stagger_sample(cudaq::state &state, int n, bool fermionic, bool optimize_classical)
{
    const auto joint = detail::receiver_reference_rdm(state, n + 2, {n + 1}, n, fermionic);
    const auto metrics = front::block_metrics(joint, 1, fermionic, optimize_classical);

    Sample sample;
    sample.sq = metrics.joint_entropy;
    sample.i2 = metrics.mutual_information;
    sample.negativity = metrics.negativity;
    sample.log_negativity = metrics.log_negativity;
    sample.classical_x = metrics.classical_x;
    sample.classical_y = metrics.classical_y;
    sample.classical_z = metrics.classical_z;
    sample.classical_mean = metrics.classical_mean;
    sample.classical_max = metrics.classical_max;
    sample.classical_optimized = metrics.classical_optimized;
    sample.discord = metrics.discord;
    sample.coherent_information = metrics.coherent_information;
    sample.block_entropy = metrics.block_entropy;        // S(B)
    sample.reference_entropy = metrics.reference_entropy; // S(A)
    return sample;
}

} // namespace mipt::probed
