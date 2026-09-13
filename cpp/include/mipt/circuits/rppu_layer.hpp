#pragma once

// The RPPU / qRPPU layer description, its random draw, and its projection onto
// a logical spacetime layer.
//
// Split out of fermion.hpp because that header includes <cudaq.h> for the
// __qpu__ kernels, and nothing here needs it: a layer is a list of bonds and a
// list of measurement flags. Keeping the split is what lets `make test-dist`
// check, on the host, the two modelling decisions the percolation graph rests
// on -- that an FSWAP is transport rather than an entangling seed, and that a
// Jordan-Wigner boundary gate is one bond rather than a fan across the string.
// Neither is a transcription of the circuit; both are choices, and a wrong one
// silently fuses the whole ring into a single cluster.

#include "mipt/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <vector>

namespace mipt
{
struct RppuBondGate
{
    // kind = 0: adjacent parity-preserving two-mode unitary.
    // kind = 1: adjacent fermionic swap.
    // kind = 2: non-adjacent parity-preserving two-mode unitary with
    //           Jordan-Wigner CZ-string corrections generated in the kernel.
    int kind = 0;
    int q0 = 0;
    int q1 = 0;
    U2Params even;
    U2Params odd;
};

struct RppuLayer
{
    std::vector<RppuBondGate> gates;
    std::vector<int> measure_flags;
};


inline std::array<std::complex<double>, 4> haar_unitary_2(std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);

    std::array<std::complex<double>, 4> z{
        std::complex<double>(normal(rng), normal(rng)),
        std::complex<double>(normal(rng), normal(rng)),
        std::complex<double>(normal(rng), normal(rng)),
        std::complex<double>(normal(rng), normal(rng)),
    };

    // Columns of z in row-major storage.
    std::complex<double> z00 = z[0];
    std::complex<double> z10 = z[2];
    std::complex<double> z01 = z[1];
    std::complex<double> z11 = z[3];

    double n0 = std::sqrt(std::norm(z00) + std::norm(z10));
    if (n0 == 0.0)
    {
        z00 = 1.0;
        z10 = 0.0;
        n0 = 1.0;
    }

    std::complex<double> q00 = z00 / n0;
    std::complex<double> q10 = z10 / n0;

    const std::complex<double> r01 = std::conj(q00) * z01 + std::conj(q10) * z11;
    std::complex<double> u01 = z01 - q00 * r01;
    std::complex<double> u11 = z11 - q10 * r01;

    double n1 = std::sqrt(std::norm(u01) + std::norm(u11));
    if (n1 < 1e-15)
    {
        // Extremely unlikely fallback: choose a deterministic orthogonal column.
        u01 = -std::conj(q10);
        u11 = std::conj(q00);
        n1 = std::sqrt(std::norm(u01) + std::norm(u11));
    }

    std::complex<double> q01 = u01 / n1;
    std::complex<double> q11 = u11 / n1;

    return {q00, q01, q10, q11};
}

enum class RppuBoundaryMode : int
{
    FermionicFSwap = 0,
    FermionicJWString = 1,
    QubitDirect = 2,
};

inline RppuBondGate random_parity_preserving_gate(int q0, int q1, std::mt19937 &rng)
{
    RppuBondGate gate;
    gate.kind = 0;
    gate.q0 = q0;
    gate.q1 = q1;
    gate.even = decompose_u2_to_u3_phase(haar_unitary_2(rng));
    gate.odd = decompose_u2_to_u3_phase(haar_unitary_2(rng));
    return gate;
}

inline RppuBondGate random_parity_preserving_jw_string_gate(int q0, int q1, std::mt19937 &rng)
{
    RppuBondGate gate = random_parity_preserving_gate(q0, q1, rng);
    gate.kind = 2;
    return gate;
}

inline RppuBondGate fermionic_swap_gate(int q0, int q1)
{
    RppuBondGate gate;
    gate.kind = 1;
    gate.q0 = q0;
    gate.q1 = q1;
    return gate;
}

inline void fill_rppu_layer(RppuLayer &layer,
                     int n,
                     bool odd_layer,
                     double p,
                     bool closed,
                     std::mt19937 &rng,
                     RppuBoundaryMode boundary_mode)
{
    std::bernoulli_distribution measure_dist(p);

    layer.gates.clear();
    layer.measure_flags.resize(static_cast<std::size_t>(n));

    if (!odd_layer)
    {
        layer.gates.reserve(static_cast<std::size_t>(n / 2));
        for (int qubit = 0; qubit < n; qubit += 2)
        {
            layer.gates.push_back(random_parity_preserving_gate(qubit, qubit + 1, rng));
        }
    }
    else
    {
        const bool compact_boundary = boundary_mode != RppuBoundaryMode::FermionicFSwap;
        const std::size_t reserve_count = closed
            ? (compact_boundary
                   ? static_cast<std::size_t>(1 + (n - 2) / 2)
                   : static_cast<std::size_t>((n - 2) + 1 + (n - 2) + (n - 2) / 2))
            : static_cast<std::size_t>((n - 2) / 2);
        layer.gates.reserve(reserve_count);

        if (closed)
        {
            if (boundary_mode == RppuBoundaryMode::FermionicJWString)
            {
                layer.gates.push_back(random_parity_preserving_jw_string_gate(0, n - 1, rng));
            }
            else if (boundary_mode == RppuBoundaryMode::QubitDirect)
            {
                // qRPPU: this is an ordinary non-local qubit gate.  No
                // Jordan-Wigner parity string and no FSWAP transport is used.
                layer.gates.push_back(random_parity_preserving_gate(0, n - 1, rng));
            }
            else
            {
                for (int k = n - 2; k > 0; --k)
                {
                    layer.gates.push_back(fermionic_swap_gate(k, k + 1));
                }

                layer.gates.push_back(random_parity_preserving_gate(0, 1, rng));

                for (int k = 1; k < n - 1; ++k)
                {
                    layer.gates.push_back(fermionic_swap_gate(k, k + 1));
                }
            }
        }

        for (int qubit = 1; qubit < n - 1; qubit += 2)
        {
            layer.gates.push_back(random_parity_preserving_gate(qubit, qubit + 1, rng));
        }
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
        for (int qubit = 0; qubit < n; ++qubit)
        {
            layer.measure_flags[static_cast<std::size_t>(qubit)] =
                measure_dist(rng) ? 1 : 0;
        }
    }
}

inline void build_rppu_layers_with_boundary(std::vector<RppuLayer> &layers,
                           int n,
                           int periods,
                           double p,
                           bool closed,
                           std::mt19937 &rng,
                           RppuBoundaryMode boundary_mode)
{
    if ((n % 2) != 0)
    {
        throw std::invalid_argument("RPPU 1D simulation requires even n.");
    }
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("Measurement probability p must be in [0,1].");
    }

    std::bernoulli_distribution extra_even_layer(0.5);
    const bool add_extra_even_layer = extra_even_layer(rng);
    const std::size_t layer_count =
        static_cast<std::size_t>(2 * periods + (add_extra_even_layer ? 1 : 0));
    layers.resize(layer_count);

    std::size_t idx = 0;
    for (int period = 0; period < periods; ++period)
    {
        fill_rppu_layer(layers[idx++], n, false, p, closed, rng, boundary_mode);
        fill_rppu_layer(layers[idx++], n, true, p, closed, rng, boundary_mode);
    }

    if (add_extra_even_layer)
    {
        fill_rppu_layer(layers[idx++], n, false, p, closed, rng, boundary_mode);
    }
}

inline void build_rppu_layers(std::vector<RppuLayer> &layers,
                                   int n,
                                   int periods,
                                   double p,
                                   bool closed,
                                   std::mt19937 &rng,
                                   bool direct_boundary_gate = false)
{
    build_rppu_layers_with_boundary(
        layers, n, periods, p, closed, rng,
        direct_boundary_gate ? RppuBoundaryMode::FermionicJWString
                             : RppuBoundaryMode::FermionicFSwap);
}

inline void build_qrppu_layers(std::vector<RppuLayer> &layers,
                            int n,
                            int periods,
                            double p,
                            bool closed,
                            std::mt19937 &rng)
{
    build_rppu_layers_with_boundary(
        layers, n, periods, p, closed, rng, RppuBoundaryMode::QubitDirect);
}

inline std::vector<RppuLayer> make_rppu_layers(int n,
                                                    int periods,
                                                    double p,
                                                    bool closed = true)
{
    std::mt19937 rng(std::random_device{}());
    std::vector<RppuLayer> layers;
    layers.reserve(static_cast<std::size_t>(2 * periods + 1));
    build_rppu_layers(layers, n, periods, p, closed, rng, false);
    return layers;
}

inline std::vector<RppuLayer> make_qrppu_layers(int n,
                                             int periods,
                                             double p,
                                             bool closed = true)
{
    std::mt19937 rng(std::random_device{}());
    std::vector<RppuLayer> layers;
    layers.reserve(static_cast<std::size_t>(2 * periods + 1));
    build_qrppu_layers(layers, n, periods, p, closed, rng);
    return layers;
}


// ---------------------------------------------------------------------------
// Logical layer extraction
// ---------------------------------------------------------------------------

// Reduce one simulated RPPU/qRPPU layer to its entangling bonds and its
// measurement sites, updating the running position -> mode permutation.
//
// `mode_at_site` is threaded through the whole history rather than reset per
// layer. For every boundary mode in the registry it is the identity at each
// layer boundary -- the FSWAP network walks a mode down the ring and walks it
// back inside a single layer -- but a partial network would otherwise relabel
// every subsequent layer silently.
//
// kind 1 (FSWAP) is transport: it swaps two entries and emits no bond.
// kind 0 and kind 2 (adjacent gate, Jordan-Wigner boundary gate) each emit
// exactly one bond between the modes their two qubits currently hold. See the
// LogicalLayer comment in types.hpp for why the JW string contributes nothing.
inline LogicalLayer logical_layer_from_rppu(const RppuLayer &layer, int n,
                                            std::vector<int> &mode_at_site)
{
    LogicalLayer out;
    out.bonds.reserve(layer.gates.size());
    for (const RppuBondGate &gate : layer.gates)
    {
        const std::size_t q0 = static_cast<std::size_t>(gate.q0);
        const std::size_t q1 = static_cast<std::size_t>(gate.q1);
        if (gate.kind == 1)
        {
            std::swap(mode_at_site[q0], mode_at_site[q1]);
            continue;
        }
        // U3(theta, phi, lambda) has |off-diagonal| = sin(theta/2), so the
        // squared off-diagonal weight of each parity block is sin^2(theta/2)
        // exactly -- no reconstruction of the 2x2 needed.
        const double pair_amplitude = std::sin(0.5 * gate.even.theta);
        const double hop_amplitude = std::sin(0.5 * gate.odd.theta);
        out.bonds.push_back({mode_at_site[q0], mode_at_site[q1],
                             pair_amplitude * pair_amplitude,
                             hop_amplitude * hop_amplitude});
    }
    set_measured(out, n, layer.measure_flags, mode_at_site);
    return out;
}

inline LogicalHistory logical_history_from_rppu(const std::vector<RppuLayer> &layers, int n)
{
    LogicalHistory history;
    history.reset(n);
    history.layers.reserve(layers.size());
    for (const RppuLayer &layer : layers)
    {
        history.layers.push_back(logical_layer_from_rppu(layer, n, history.mode_at_site));
    }
    history.valid = true;
    return history;
}


inline CircuitWorkStats circuit_work_stats_rppu(const std::vector<RppuLayer> &layers)
{
    CircuitWorkStats stats;
    stats.layers = layers.size();
    for (const auto &layer : layers)
    {
        for (int flag : layer.measure_flags)
        {
            stats.measurements += (flag != 0) ? 1u : 0u;
        }
        for (const auto &gate : layer.gates)
        {
            if (gate.kind == 1)
            {
                ++stats.fswap_gates;
            }
            else if (gate.kind == 2)
            {
                ++stats.fermion_parity_gates;
                ++stats.fermion_jw_boundary_gates;
                const int distance = std::abs(gate.q1 - gate.q0);
                stats.fermion_jw_cz_corrections += static_cast<std::size_t>(2 * std::max(0, distance - 1));
            }
            else
            {
                ++stats.fermion_parity_gates;
            }
        }
    }
    stats.logical_two_site_gates = stats.fswap_gates + stats.fermion_parity_gates;
    // Lower-bound primitive count in the CUDA-Q kernel: FSWAP is CZ+SWAP;
    // a parity-preserving gate uses two CX and two controlled-U3 operations.
    stats.estimated_cudaq_two_qubit_ops = 2u * stats.fswap_gates +
                                          4u * stats.fermion_parity_gates +
                                          stats.fermion_jw_cz_corrections;
    return stats;
}

} // namespace mipt
