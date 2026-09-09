#pragma once

// RPPU, RFGS, and qRPPU circuit definitions. qRPPU shares the parity-
// preserving gates but uses a direct qubit boundary bond and qubit trace.

#include "mipt/circuits/rppu_layer.hpp"
#include "mipt/types.hpp"

#include <cudaq.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <vector>

namespace mipt
{
__qpu__ inline void apply_rppu_layers(cudaq::qvector<> &q,
                                     int n,
                                     const std::vector<RppuLayer> &layers)
{
    for (std::size_t layer = 0; layer < layers.size(); ++layer)
    {
            for (std::size_t gi = 0; gi < layers[layer].gates.size(); ++gi)
            {
                const auto gate = layers[layer].gates[gi];
                if (gate.kind == 1)
                {
                    // FSWAP = CZ * SWAP in the |00>,|01>,|10>,|11> basis.
                    z<cudaq::ctrl>(q[gate.q0], q[gate.q1]);
                    swap(q[gate.q0], q[gate.q1]);
                    continue;
                }

                if (gate.kind == 2)
                {
                    // Direct Jordan-Wigner representation for the periodic
                    // boundary fermionic gate.  This is algebraically the
                    // same nonlocal parity-preserving gate as the historical
                    // FSWAP-around-the-ring construction, but it avoids moving
                    // modes through the MPS with a long SWAP network.
                    const int lo = gate.q0 < gate.q1 ? gate.q0 : gate.q1;
                    const int hi = gate.q0 < gate.q1 ? gate.q1 : gate.q0;
                    for (int k = lo + 1; k < hi; ++k)
                    {
                        z<cudaq::ctrl>(q[k], q[hi]);
                    }
                }

                // Let a=q0 and b=q1.  First map parity sectors with
                // CNOT(a -> b).  In the mapped basis, b=1 is the original
                // odd sector {|01>,|10>} and b=0 is the original even
                // sector {|00>,|11>}.  Controlled U3 on a then implements
                // the desired 2x2 block; r1 supplies the block-global
                // phase, which is physically relevant between parity blocks.
                cx(q[gate.q0], q[gate.q1]);

                r1(gate.odd.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.odd.theta,
                                 gate.odd.phi,
                                 gate.odd.lambda,
                                 q[gate.q1], q[gate.q0]);

                x(q[gate.q1]);
                r1(gate.even.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.even.theta,
                                 gate.even.phi,
                                 gate.even.lambda,
                                 q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);

                cx(q[gate.q0], q[gate.q1]);

                if (gate.kind == 2)
                {
                    const int lo = gate.q0 < gate.q1 ? gate.q0 : gate.q1;
                    const int hi = gate.q0 < gate.q1 ? gate.q1 : gate.q0;
                    for (int k = lo + 1; k < hi; ++k)
                    {
                        z<cudaq::ctrl>(q[k], q[hi]);
                    }
                }
            }

            for (int i = 0; i < n; ++i)
            {
                if (layers[layer].measure_flags[i])
                {
                    mz(q[i]);
                }
            }
    }
}

struct RppuKernel1D
{
    void operator()(int n,
                    const std::vector<RppuLayer> &layers) __qpu__
    {
        cudaq::qvector q(n);
        apply_rppu_layers(q, n, layers);
    }
};


struct RfgsLayer // Layer data for the reduced fermionic gate set.
{
    int start = 0;

    std::vector<int> measure_flags;

    // Per-active-bond local gate choices.  These vectors are indexed by the
    // compact brickwork-bond index, not by the absolute site index, so no RNG
    // or host->kernel payload is spent on unused q entries.
    //   value 1 -> apply T gate:            R_Z(pi/4) = exp(-i*pi/8 Z_j)
    //   value 2 -> apply single-site braid: R_Z(pi/2) = exp(-i*pi/4 Z_j)
    // Each endpoint chooses exactly one of these two gates with 50/50 odds.
    std::vector<int> local_left_mask;
    std::vector<int> local_right_mask;
};

__qpu__ inline void apply_rfgs_layers(cudaq::qvector<> &q,
                                     int n,
                                     const std::vector<RfgsLayer> &flayers,
                                     bool closed)
{
    for (std::size_t layer = 0; layer < flayers.size(); ++layer)
    {
            const int start = flayers[layer].start;
            const int bond_stop = (closed && start == 1 && n > 2) ? n : (n - 1);

            int bond_index = 0;
            for (int i = start; i < bond_stop; i += 2)
            {
                const int j = (i + 1) % n;
                const int local_i = flayers[layer].local_left_mask[bond_index];
                const int local_j = flayers[layer].local_right_mask[bond_index];
                const bool wrapping_bond = (j < i);

                if (wrapping_bond) // R_XX(pi/2) across periodic boundary
                {
                    // Wrapping fermionic bond: apply the Jordan-Wigner CZ
                    // string around the periodic boundary, then undo it after
                    // the R_XX gate.  The following R_ZZ(pi/4) is interleaved
                    // with the inverse CZ string as far as the gate
                    // dependencies allow.
                    for (int k = j + 1; k < i; ++k)
                    {
                        cz(q[k], q[i]);
                    }

                    // The wrapping XX implementation needs S^dagger on both
                    // endpoints before the XX rotation.  Fuse that diagonal
                    // phase with the 50/50 local gate choice:
                    //   local T * S^dagger -> T^dagger
                    //   local S * S^dagger -> identity
                    if (local_i == 1)
                    {
                        t<cudaq::adj>(q[i]);
                    }

                    if (local_j == 1)
                    {
                        t<cudaq::adj>(q[j]);
                    }

                    // R_XX(pi/2): exp(+i*pi/4 X_i X_j).
                    cx(q[i], q[j]);
                    rx(-1.57079632679489661923, q[i]);
                    cx(q[i], q[j]);

                    s(q[i]);
                    s(q[j]);

                    // R_ZZ(pi/4): exp(+i*pi/8 Z_i Z_j).  Put the central
                    // T^dagger leg inside the inverse CZ string so it can be
                    // scheduled in parallel with one of those CZ corrections.
                    cx(q[i], q[j]);
                    int rzz_phase_applied = 0;
                    for (int k = j + 1; k < i; ++k)
                    {
                        cz(q[k], q[i]);
                        if (rzz_phase_applied == 0)
                        {
                            t<cudaq::adj>(q[j]);
                            rzz_phase_applied = 1;
                        }
                    }
                    if (rzz_phase_applied == 0)
                    {
                        t<cudaq::adj>(q[j]);
                    }
                    cx(q[i], q[j]);
                }
                else
                {
                    // CUDA-Q T and S differ from R_Z(pi/4) and R_Z(pi/2)
                    // only by one-qubit global phases, so observables and
                    // reduced density matrices are unchanged while the
                    // simulator gets native phase gates instead of generic RZs.
                    if (local_i == 1)
                    {
                        t(q[i]);
                    }
                    else
                    {
                        s(q[i]);
                    }

                    if (local_j == 1)
                    {
                        t(q[j]);
                    }
                    else
                    {
                        s(q[j]);
                    }

                    // R_XX(pi/2): exp(+i*pi/4 X_i X_j).
                    cx(q[i], q[j]);
                    rx(-1.57079632679489661923, q[i]);
                    cx(q[i], q[j]);

                    // R_ZZ(pi/4): exp(+i*pi/8 Z_i Z_j).
                    cx(q[i], q[j]);
                    t<cudaq::adj>(q[j]);
                    cx(q[i], q[j]);
                }
                ++bond_index;
            }

            for (int i = 0; i < n; ++i)
            {
                if (flayers[layer].measure_flags[i])
                {
                    mz(q[i]); // Z measurement preserves fermion parity.
                }
            }
    }
}

struct RfgsKernel1D
{
    void operator()(int n,
                    const std::vector<RfgsLayer> &flayers,
                    bool closed) __qpu__
    {
        cudaq::qvector q(n);
        apply_rfgs_layers(q, n, flayers, closed);
    }
};

inline int rfgs_active_bond_count(int n, int start, bool closed = true)
{
    const int bond_stop = (closed && start == 1 && n > 2) ? n : (n - 1);
    if (bond_stop <= start)
    {
        return 0;
    }
    return (bond_stop - start + 1) / 2;
}

inline int rfgs_rng_bit(std::mt19937 &rng)
{
    return static_cast<int>((rng() >> 31) & 1u);
}

inline void fill_rfgs_layer(RfgsLayer &layer,
                          int n,
                          int start,
                          double p,
                          std::mt19937 &rng,
                          bool closed = true)
{
    std::bernoulli_distribution measure_dist(p);

    layer.start = start;
    layer.measure_flags.resize(n);

    const int bond_count = rfgs_active_bond_count(n, start, closed);
    layer.local_left_mask.resize(bond_count);
    layer.local_right_mask.resize(bond_count);

    for (int b = 0; b < bond_count; ++b)
    {
        // Each endpoint receives exactly one local gate: T or single-site
        // braid, selected with 50/50 odds.
        layer.local_left_mask[b] = rfgs_rng_bit(rng) ? 1 : 2;
        layer.local_right_mask[b] = rfgs_rng_bit(rng) ? 1 : 2;
    }

    if (p <= 0.0)
    {
        std::fill(layer.measure_flags.begin(), layer.measure_flags.end(), 0);
    }
    else if (p >= 1.0)
    {
        std::fill(layer.measure_flags.begin(), layer.measure_flags.end(), 1);
    }
    else
    {
        for (int qidx = 0; qidx < n; ++qidx)
        {
            layer.measure_flags[qidx] = measure_dist(rng) ? 1 : 0;
        }
    }
}

inline RfgsLayer make_rfgs_layer(int n,
                                   int start,
                                   double p,
                                   std::mt19937 &rng,
                                   bool closed = true)
{
    RfgsLayer layer;
    fill_rfgs_layer(layer, n, start, p, rng, closed);
    return layer;
}

inline void build_rfgs_layers(std::vector<RfgsLayer> &layers,
                                int n,
                                int periods,
                                double p,
                                std::mt19937 &rng,
                                bool closed = true)
{
    if ((n % 2) != 0)
    {
        throw std::invalid_argument("RFGS fermionic 1D simulation requires even n.");
    }
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("Measurement probability p must be in [0,1].");
    }

    // Optional final even layer: included with 50% probability, and if included
    // its measurements use the same measurement rate p as the bulk layers.
    std::bernoulli_distribution extra_even_layer(0.5);
    const bool add_extra_even_layer = extra_even_layer(rng);
    const std::size_t layer_count = static_cast<std::size_t>(2 * periods + (add_extra_even_layer ? 1 : 0));

    // resize() preserves the inner vector capacities for existing layers,
    // avoiding thousands of small allocations during large realization sweeps.
    layers.resize(layer_count);

    std::size_t idx = 0;
    for (int period = 0; period < periods; ++period)
    {
        fill_rfgs_layer(layers[idx++], n, 0, p, rng, closed);
        fill_rfgs_layer(layers[idx++], n, 1, p, rng, closed);
    }

    if (add_extra_even_layer)
    {
        fill_rfgs_layer(layers[idx++], n, 0, p, rng, closed);
    }
}

inline std::vector<RfgsLayer> make_rfgs_layers(int n,
                                              int periods,
                                              double p,
                                              bool closed = true)
{
    std::mt19937 rng(std::random_device{}());
    std::vector<RfgsLayer> layers;
    build_rfgs_layers(layers, n, periods, p, rng, closed);
    return layers;
}


// RFGS bonds follow the same brickwork sweep as Haar's, wrapping on the odd
// layer. The reduced gate set carries no transport operation, so there is no
// permutation to track.
inline LogicalHistory logical_history_from_rfgs(const std::vector<RfgsLayer> &layers, int n,
                                                 bool closed)
{
    LogicalHistory history;
    history.reset(n);
    history.layers.reserve(layers.size());
    for (const RfgsLayer &layer : layers)
    {
        LogicalLayer out;
        append_brickwork_bonds(out, n, layer.start, closed, history.mode_at_site);
        set_measured(out, n, layer.measure_flags, history.mode_at_site);
        history.layers.push_back(std::move(out));
    }
    history.valid = true;
    return history;
}

inline CircuitWorkStats circuit_work_stats_rfgs(const std::vector<RfgsLayer> &layers, int n, bool closed)
{
    CircuitWorkStats stats;
    stats.layers = layers.size();
    for (const auto &layer : layers)
    {
        for (int flag : layer.measure_flags)
        {
            stats.measurements += (flag != 0) ? 1u : 0u;
        }
        const int start = layer.start;
        const int bond_stop = (closed && start == 1 && n > 2) ? n : (n - 1);
        for (int i = start; i < bond_stop; i += 2)
        {
            ++stats.rfgs_bonds;
            const int j = (i + 1) % n;
            if (j < i)
            {
                ++stats.wrapping_rfgs_bonds;
                stats.estimated_cudaq_two_qubit_ops += static_cast<std::size_t>(2 * std::max(0, i - j - 1) + 4);
            }
            else
            {
                stats.estimated_cudaq_two_qubit_ops += 4u;
            }
        }
    }
    stats.logical_two_site_gates = stats.rfgs_bonds;
    return stats;
}



} // namespace mipt
