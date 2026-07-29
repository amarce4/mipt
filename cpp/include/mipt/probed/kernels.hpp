#pragma once

// Reference-qubit state preparation and readout.
//
// The monitored system occupies qubits [0, N) and the references occupy
// [N, N+probes). Attaching a reference resets its partner site, applies H,
// and entangles the pair, so the reference holds exactly the information the
// measurement record has not yet destroyed.
//
// The parity-encoded path is the exception: for parity-preserving circuits a
// single reference is redundant with the total system parity, so only N
// qubits are simulated and the probe entropy is read from the parity weights.

#include "mipt/analysis/fgmn.hpp"
#include "mipt/ancilla_info.hpp"
#include "mipt/probed/sample.hpp"
#include "mipt/types.hpp"

#include <cudaq.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mipt::probed
{

struct InitializeReferencesKernel
{
    void operator()(int n, const std::vector<int> &sites) __qpu__
    {
        cudaq::qvector q(n + static_cast<int>(sites.size()));
        for (std::size_t i = 0; i < sites.size(); ++i)
        {
            h(q[sites[i]]);
            cx(q[sites[i]], q[n + static_cast<int>(i)]);
        }
    }
};

struct InitializeProductKernel
{
    void operator()(int total_qubits) __qpu__
    {
        cudaq::qvector q(total_qubits);
    }
};

struct AttachReferencesKernel
{
    void operator()(cudaq::state *state, int n, const std::vector<int> &sites) __qpu__
    {
        cudaq::qvector q(state);
        for (std::size_t i = 0; i < sites.size(); ++i)
        {
            reset(q[sites[i]]);
            h(q[sites[i]]);
            cx(q[sites[i]], q[n + static_cast<int>(i)]);
        }
    }
};

struct AttachReferenceKernel
{
    void operator()(cudaq::state *state, int n, int site, int reference) __qpu__
    {
        cudaq::qvector q(state);
        reset(q[site]);
        h(q[site]);
        cx(q[site], q[n + reference]);
    }
};

std::array<double, 128> interleaved_rdm(const ancilla::Rdm3 &rho)
{
    std::array<double, 128> packed{};
    for (std::size_t i = 0; i < rho.values.size(); ++i)
    {
        packed[2 * i] = std::real(rho.values[i]);
        packed[2 * i + 1] = std::imag(rho.values[i]);
    }
    return packed;
}

cudaq::state initialize_state(int n, const std::vector<int> &sites, bool references_attached)
{
    return references_attached ? cudaq::get_state(InitializeReferencesKernel{}, n, sites)
                               : cudaq::get_state(InitializeProductKernel{}, n + static_cast<int>(sites.size()));
}

cudaq::state attach_references(cudaq::state &state, int n, const std::vector<int> &sites)
{
    return cudaq::get_state(AttachReferencesKernel{}, &state, n, sites);
}

cudaq::state attach_reference(cudaq::state &state, int n, int site, int reference)
{
    return cudaq::get_state(AttachReferenceKernel{}, &state, n, site, reference);
}

Sample measure_sample(cudaq::state &state, int probes, int n, CircuitType type)
{
    const bool fermionic = uses_fermionic_trace(type);
    const int total_qubits = n + probes;
    if (probes == 1)
    {
        const auto rho = ancilla::reduced_density_matrix(state, total_qubits, n, fermionic);
        return {ancilla::entropy_from_rdm(rho), 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    if (probes == 2)
    {
        const auto values = ancilla::two_probe_observables(state, total_qubits, n, n + 1, fermionic);
        return {values.joint_entropy, values.mutual_information, 0.0, 0.0, values.negativity, values.log_negativity};
    }
    const auto values = ancilla::four_probe_information(state, total_qubits, {n, n + 1, n + 2, n + 3}, fermionic);
    return {values.entropy, values.i2, values.i3, values.i4, 0.0, 0.0};
}

struct ThreeProbeMeasurement
{
    Sample sample;
    std::array<double, 128> rho{};
};

ThreeProbeMeasurement measure_three_probe_sample(cudaq::state &state, int n, CircuitType type)
{
    const bool fermionic = uses_fermionic_trace(type);
    const auto rho = ancilla::reduced_density_matrix_three(state, n + 3, {n, n + 1, n + 2}, fermionic);
    const auto information = ancilla::three_probe_information(rho, fermionic);
    auto packed = interleaved_rdm(rho);
    const double minimum_negativity = fermionic ? compute_min_bipartite_fermionic_negativity_8x8_cpp(packed.data())
                                                : compute_min_bipartite_negativity_8x8_cpp(packed.data());
    if (!std::isfinite(minimum_negativity))
    {
        throw std::runtime_error("Three-probe bipartite-negativity evaluation "
                                 "returned a non-finite value.");
    }
    Sample sample;
    sample.tmi = information.tmi;
    sample.average_mi = information.average_mi;
    sample.min_bipartite_negativity = std::max(0.0, minimum_negativity);
    sample.joint_purity = information.joint_purity;
    sample.mean_single_purity = information.mean_single_purity;
    return {sample, std::move(packed)};
}

Sample measure_parity_encoded_probe(cudaq::state &state, int n)
{
    const auto weights = ancilla::parity_sector_weights(state, n);
    const double entropy = ancilla::entropy_from_rdm({weights.even, weights.odd, {0.0, 0.0}});
    return {entropy, 0.0, 0.0, 0.0, 0.0, 0.0};
}

} // namespace mipt::probed
