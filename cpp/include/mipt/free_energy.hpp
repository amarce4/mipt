#pragma once

#include "mipt/backend.hpp"
#include "mipt/env.hpp"
#include "mipt/free_energy_measure.hpp"
#include "mipt/rdm.hpp"
#include "mipt/tmi/compute.hpp"
#include "mipt/types.hpp"
#include "mipt/util/stats.hpp"

#ifdef MIPT_ENABLE_CUDA_RHO
#include "mipt/cuda/free_energy_measure.hpp"
#endif

#include <cudaq.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::free_energy
{
using util::RunningStats;

inline MeasurementResult measure_and_collapse(
    cudaq::state &state,
    int n,
    int measured_qubit,
    double uniform_01)
{
    if (measured_qubit < 0 || measured_qubit >= n ||
        !(uniform_01 >= 0.0 && uniform_01 < 1.0))
    {
        throw std::invalid_argument("Invalid sequential-measurement request.");
    }
    const std::uint64_t dimension = checked_pow2_u64(n);
    const auto tensor = state.get_tensor();
    if (tensor.get_rank() != 1 || tensor.get_num_elements() < dimension ||
        tensor.data == nullptr)
    {
        throw std::runtime_error(
            backend::dense_state_required_message(
                "Sequential free-energy measurements"));
    }

    if (!state.is_on_gpu())
    {
        if (state.get_precision() == cudaq::SimulationState::precision::fp64)
        {
            return measure_collapse_host(
                reinterpret_cast<std::complex<double> *>(tensor.data),
                n,
                measured_qubit,
                uniform_01);
        }
        return measure_collapse_host(
            reinterpret_cast<std::complex<float> *>(tensor.data),
            n,
            measured_qubit,
            uniform_01);
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    MeasurementResult result;
    const int status =
        state.get_precision() == cudaq::SimulationState::precision::fp64
            ? mipt_cuda_measure_collapse_f64(
                  tensor.data,
                  n,
                  measured_qubit,
                  uniform_01,
                  &result.outcome,
                  &result.conditional_probability)
            : mipt_cuda_measure_collapse_f32(
                  tensor.data,
                  n,
                  measured_qubit,
                  uniform_01,
                  &result.outcome,
                  &result.conditional_probability);
    if (status != 0)
    {
        const char *detail = mipt_cuda_measure_collapse_last_error();
        throw std::runtime_error(
            "CUDA sequential measurement failed (status=" +
            std::to_string(status) + "): " +
            ((detail != nullptr && *detail != '\0') ? detail : "unknown error"));
    }
    result.surprisal = -std::log(result.conditional_probability);
    return result;
#else
    throw std::runtime_error(
        "This free_energy.exe build has a GPU statevector but was compiled "
        "without CUDA_RHO=1, so it cannot collapse measurements in place. "
        "Rebuild with GPU=1 CUDA_RHO=1 or use GPU=0 CUDA_RHO=0.");
#endif
}

struct InitializeFromStateKernel
{
    void operator()(cudaq::state *initial_state) __qpu__
    {
        cudaq::qvector q(initial_state);
    }
};

#if defined(MIPT_CUDAQ_PRECISION) && MIPT_CUDAQ_PRECISION == 32
using InitialReal = float;
#else
using InitialReal = double;
#endif

struct InitialState
{
    cudaq::state state;
    int parity_sector = -1;
};

template <typename Rng>
inline std::vector<std::complex<InitialReal>> random_product_amplitudes(
    int n,
    bool fixed_parity,
    int &parity_sector,
    Rng &rng)
{
    const std::size_t dimension =
        static_cast<std::size_t>(checked_pow2_u64(n));
    std::vector<std::complex<InitialReal>> amplitudes(
        dimension,
        std::complex<InitialReal>{0, 0});

    if (fixed_parity)
    {
        // A pure tensor product of single-site states is a global parity
        // eigenstate only when every factor is a Z/occupation eigenstate.
        // Draw such a basis product uniformly within a random parity sector.
        std::bernoulli_distribution bit(0.5);
        parity_sector = bit(rng) ? 1 : 0;
        std::uint64_t basis = 0;
        int parity = 0;
        for (int q = 0; q < n - 1; ++q)
        {
            const int value = bit(rng) ? 1 : 0;
            parity ^= value;
            basis |= static_cast<std::uint64_t>(value) << q;
        }
        basis |= static_cast<std::uint64_t>(parity ^ parity_sector) << (n - 1);
        amplitudes[static_cast<std::size_t>(basis)] =
            std::complex<InitialReal>{1, 0};
        return amplitudes;
    }

    parity_sector = -1;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    amplitudes[0] = std::complex<InitialReal>{1, 0};
    std::size_t active = 1;
    constexpr double two_pi = 6.283185307179586476925286766559;
    for (int q = 0; q < n; ++q)
    {
        const double z = 1.0 - 2.0 * unit(rng);
        const double theta = std::acos(std::clamp(z, -1.0, 1.0));
        const double phi = two_pi * unit(rng);
        const std::complex<double> zero(std::cos(0.5 * theta), 0.0);
        const std::complex<double> one =
            std::polar(std::sin(0.5 * theta), phi);
        for (std::size_t index = 0; index < active; ++index)
        {
            const std::complex<double> prior = amplitudes[index];
            amplitudes[index] = static_cast<std::complex<InitialReal>>(prior * zero);
            amplitudes[index | active] =
                static_cast<std::complex<InitialReal>>(prior * one);
        }
        active <<= 1;
    }
    return amplitudes;
}

template <typename Rng>
inline std::vector<std::complex<InitialReal>> random_haar_amplitudes(
    int n,
    bool fixed_parity,
    int &parity_sector,
    Rng &rng)
{
    const std::size_t dimension =
        static_cast<std::size_t>(checked_pow2_u64(n));
    std::vector<std::complex<InitialReal>> amplitudes(
        dimension,
        std::complex<InitialReal>{0, 0});
    std::normal_distribution<double> normal(0.0, 1.0);
    std::bernoulli_distribution sector(0.5);
    parity_sector = fixed_parity && sector(rng) ? 1 : (fixed_parity ? 0 : -1);

    long double norm2 = 0.0L;
    for (std::size_t index = 0; index < dimension; ++index)
    {
        if (fixed_parity &&
            ((__builtin_popcountll(static_cast<unsigned long long>(index)) & 1) !=
             parity_sector))
        {
            continue;
        }
        const std::complex<double> value(normal(rng), normal(rng));
        amplitudes[index] = static_cast<std::complex<InitialReal>>(value);
        norm2 += static_cast<long double>(std::norm(value));
    }
    if (!(norm2 > 0.0L))
    {
        throw std::runtime_error("Random Haar-state normalization vanished.");
    }
    const InitialReal inverse_norm =
        static_cast<InitialReal>(1.0 / std::sqrt(static_cast<double>(norm2)));
    for (auto &value : amplitudes)
    {
        value *= inverse_norm;
    }
    return amplitudes;
}

template <typename Rng>
inline InitialState make_initial_state(
    int n,
    int init_state,
    CircuitType circuit_type,
    Rng &rng)
{
    const bool fixed_parity = preserves_computational_parity(circuit_type);
    int parity_sector = -1;
    auto amplitudes = init_state == 0
                          ? random_product_amplitudes(
                                n, fixed_parity, parity_sector, rng)
                          : random_haar_amplitudes(
                                n, fixed_parity, parity_sector, rng);
    auto input_state = cudaq::state::from_data(amplitudes);
    return {
        cudaq::get_state(InitializeFromStateKernel{}, &input_state),
        parity_sector,
    };
}

struct HalfEntropyWorkspace
{
    tmi::StateEntropyWorkspace host;
    std::vector<std::complex<double>> host_f64;
    std::vector<std::complex<float>> host_f32;
    bool cuda_disabled = false;
};

inline int cuda_entropy_min_kept_modes()
{
    // A 7-mode half cut is the N=14 boundary (a 128x128 Schmidt matrix).
    // Sending that case through host LAPACK requires a device-to-host copy and
    // makes N=14 markedly slower than N=16, whose 8-mode cut already uses
    // cuSOLVER through the sim_tmi policy.  Free-energy trajectories evaluate
    // this SVD repeatedly, so use a dedicated threshold without changing
    // sim_tmi.exe's tuned small-system behavior.
    constexpr long default_min_kept_modes = 7;
    constexpr long max_kept_modes = 30;
    static const int threshold = static_cast<int>(env::integer(
        "MIPT_FREE_ENERGY_CUDA_ENTROPY_MIN_KEPT",
        default_min_kept_modes,
        0,
        max_kept_modes));
    return threshold;
}

inline double half_chain_entropy_bits(
    cudaq::state &state,
    int n,
    bool fermion_trace,
    HalfEntropyWorkspace &workspace)
{
    const int kept_count = n / 2;
    std::vector<int> kept_modes(static_cast<std::size_t>(kept_count));
    for (int q = 0; q < kept_count; ++q)
    {
        kept_modes[static_cast<std::size_t>(q)] = q;
    }

    const std::uint64_t dimension_u64 = checked_pow2_u64(n);
    const std::size_t dimension = static_cast<std::size_t>(dimension_u64);
    const auto tensor = state.get_tensor();
    if (tensor.get_rank() != 1 || tensor.get_num_elements() < dimension ||
        tensor.data == nullptr)
    {
        throw std::runtime_error(
            backend::dense_state_required_message("Half-chain entropy"));
    }

    const bool fp64 =
        state.get_precision() == cudaq::SimulationState::precision::fp64;
    if (state.is_on_gpu() && !workspace.cuda_disabled &&
        env::boolean("MIPT_FREE_ENERGY_CUDA_ENTROPY", true) &&
        kept_modes.size() >=
            static_cast<std::size_t>(cuda_entropy_min_kept_modes()))
    {
        double entropy = 0.0;
        int status = 0;
        if (tmi::entropy_subsystem_svd_from_device_state(
                tensor.data,
                fp64,
                n,
                kept_modes,
                fermion_trace,
                entropy,
                status))
        {
            return entropy;
        }
        workspace.cuda_disabled = true;
        if (backend::verbose_enabled())
        {
            backend::verbose_log(
                "Half-chain CUDA SVD failed with status=" +
                std::to_string(status) + "; using host LAPACK.");
        }
    }

    if (fp64)
    {
        workspace.host_f64.resize(dimension);
        if (state.is_on_gpu())
        {
            state.to_host(workspace.host_f64.data(), dimension);
        }
        else
        {
            const auto *source =
                reinterpret_cast<const std::complex<double> *>(tensor.data);
            std::copy(source, source + dimension, workspace.host_f64.begin());
        }
        return tmi::entropy_subsystem_svd_from_state(
            workspace.host_f64.data(),
            n,
            kept_modes,
            fermion_trace,
            workspace.host);
    }

    workspace.host_f32.resize(dimension);
    if (state.is_on_gpu())
    {
        state.to_host(workspace.host_f32.data(), dimension);
    }
    else
    {
        const auto *source =
            reinterpret_cast<const std::complex<float> *>(tensor.data);
        std::copy(source, source + dimension, workspace.host_f32.begin());
    }
    return tmi::entropy_subsystem_svd_from_state(
        workspace.host_f32.data(),
        n,
        kept_modes,
        fermion_trace,
        workspace.host);
}
} // namespace mipt::free_energy
