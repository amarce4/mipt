#pragma once

#include "mipt/backend.hpp"
#include "mipt/density.hpp"

#ifdef MIPT_ENABLE_CUDA_RHO
#include "mipt/cuda/rho_reduce.hpp"
#endif

#include <cudaq.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt
{
constexpr int RHO3_QUBITS = 3;
constexpr int RHO3_DIM = 1 << RHO3_QUBITS; // 8
constexpr std::size_t RHO3_VALUES = 2u * RHO3_DIM * RHO3_DIM;
using Subsystem = std::array<int, RHO3_QUBITS>;

inline std::uint64_t checked_pow2_u64(int exponent)
{
    if (exponent < 0 || exponent >= 63)
    {
        throw std::invalid_argument("Invalid qubit count: 2^n would overflow uint64_t.");
    }
    return std::uint64_t{1} << exponent;
}

inline int fermionic_reorder_sign_from_occupations(
    const std::vector<unsigned char> &occ_by_mode,
    const std::vector<int> &new_mode_order)
{
    // Sign for |n_0...n_{N-1}>_canonical =
    // sign * |n_{new_order[0]}...>_new_order.
    std::vector<int> pos_new(occ_by_mode.size(), 0);
    for (std::size_t pos = 0; pos < new_mode_order.size(); ++pos)
    {
        pos_new[static_cast<std::size_t>(new_mode_order[pos])] =
            static_cast<int>(pos);
    }

    int parity = 0;
    std::vector<int> occupied_positions;
    occupied_positions.reserve(occ_by_mode.size());
    for (std::size_t mode = 0; mode < occ_by_mode.size(); ++mode)
    {
        if (occ_by_mode[mode])
        {
            occupied_positions.push_back(pos_new[mode]);
        }
    }

    for (std::size_t i = 0; i < occupied_positions.size(); ++i)
    {
        for (std::size_t j = i + 1; j < occupied_positions.size(); ++j)
        {
            if (occupied_positions[i] > occupied_positions[j])
            {
                parity ^= 1;
            }
        }
    }
    return parity ? -1 : 1;
}

inline std::uint64_t index_from_occupations_little(
    const std::vector<unsigned char> &occ_by_mode)
{
    std::uint64_t idx = 0;
    for (std::size_t mode = 0; mode < occ_by_mode.size(); ++mode)
    {
        if (occ_by_mode[mode])
        {
            idx |= std::uint64_t{1} << mode;
        }
    }
    return idx;
}

inline std::vector<Subsystem> periodic_ring_three_site_subsystems(int n)
{
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument("Three-qubit RDM output requires n >= 3.");
    }

    std::vector<Subsystem> subsystems;
    subsystems.reserve(static_cast<std::size_t>(n));
    for (int start = 0; start < n; ++start)
    {
        subsystems.push_back({start, (start + 1) % n, (start + 2) % n});
    }
    return subsystems;
}

inline std::vector<Subsystem> three_site_subsystems_2d(int x, int y)
{
    std::vector<Subsystem> subsystems;
    subsystems.reserve(static_cast<std::size_t>(x*y));
    for (int i = 0; i < y; ++i) // horizontal subsystems
    {
        for (int j = 0; j < x; ++j)
        {
            subsystems.push_back({i*x + j, i*x + (j + 1)%x, i*x + (j + 2)%x});
        }
    }
    for (int i = 0; i < x; ++i) // vertical subsystems
    {
        for (int j = 0; j < y; ++j)
        {
            subsystems.push_back({j*x + i, ((j + 1)%y)*x + i, ((j + 2)%y)*x + i});
        }
    }
    return subsystems;
}

inline std::vector<Subsystem> terminal_three_site_subsystem(int n)
{
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument("Three-qubit RDM output requires at least three qubits.");
    }
    return {{n - 3, n - 2, n - 1}};
}

inline std::vector<int> flatten_subsystems(const std::vector<Subsystem> &subsystems)
{
    std::vector<int> flat;
    flat.reserve(subsystems.size() * RHO3_QUBITS);
    for (const auto &subsystem : subsystems)
    {
        flat.insert(flat.end(), subsystem.begin(), subsystem.end());
    }
    return flat;
}

template <typename Real>
inline void reduced_density_complex_from_host_statevector(
    const std::complex<Real> *psi,
    int n,
    const std::vector<Subsystem> &subsystems,
    double *rho_ri,
    bool fermion_trace = false)
{
    if (psi == nullptr || rho_ri == nullptr)
    {
        throw std::invalid_argument("Null state-vector or density-matrix pointer.");
    }
    if (n < RHO3_QUBITS || subsystems.empty())
    {
        throw std::invalid_argument("Invalid retained subsystem definition.");
    }

    const std::size_t output_values = subsystems.size() * RHO3_VALUES;
    std::fill(rho_ri, rho_ri + output_values, 0.0);
    const std::uint64_t env_dim = checked_pow2_u64(n - RHO3_QUBITS);

    // CUDA-Q qubit b maps to bit b in the linear statevector index.
    // Reduced-basis bit k maps to subsystem[k]. This preserves the
    // requested cyclic ordering for wraparound triples such as {n-1,0,1}.
    for (std::size_t s = 0; s < subsystems.size(); ++s)
    {
        const auto &subsystem = subsystems[s];
        for (int i = 0; i < RHO3_QUBITS; ++i)
        {
            if (subsystem[i] < 0 || subsystem[i] >= n)
            {
                throw std::invalid_argument("Subsystem qubit index is out of range.");
            }
            for (int j = 0; j < i; ++j)
            {
                if (subsystem[i] == subsystem[j])
                {
                    throw std::invalid_argument("Subsystem contains duplicate qubits.");
                }
            }
        }

        if (!fermion_trace)
        {
            std::array<std::uint64_t, RHO3_DIM> retained_offsets{};
            for (int local_basis = 0; local_basis < RHO3_DIM; ++local_basis)
            {
                for (int k = 0; k < RHO3_QUBITS; ++k)
                {
                    retained_offsets[local_basis] |=
                        (static_cast<std::uint64_t>((local_basis >> k) & 1)
                         << subsystem[k]);
                }
            }

            for (std::uint64_t env = 0; env < env_dim; ++env)
            {
                std::uint64_t base = 0;
                int env_bit = 0;
                for (int q = 0; q < n; ++q)
                {
                    if (q != subsystem[0] && q != subsystem[1] && q != subsystem[2])
                    {
                        base |= ((env >> env_bit) & std::uint64_t{1}) << q;
                        ++env_bit;
                    }
                }

                for (int row = 0; row < RHO3_DIM; ++row)
                {
                    const auto a = psi[base | retained_offsets[row]];
                    for (int col = 0; col < RHO3_DIM; ++col)
                    {
                        const auto b = psi[base | retained_offsets[col]];
                        const auto value = a * std::conj(b);
                        const std::size_t out =
                            s * RHO3_VALUES +
                            2u * static_cast<std::size_t>(row * RHO3_DIM + col);
                        rho_ri[out + 0] += static_cast<double>(std::real(value));
                        rho_ri[out + 1] += static_cast<double>(std::imag(value));
                    }
                }
            }
            continue;
        }

        std::vector<int> kept_modes(subsystem.begin(), subsystem.end());
        std::vector<unsigned char> is_retained(static_cast<std::size_t>(n), 0);
        for (int q : kept_modes)
        {
            is_retained[static_cast<std::size_t>(q)] = 1;
        }

        std::vector<int> environment_modes;
        environment_modes.reserve(static_cast<std::size_t>(n - RHO3_QUBITS));
        for (int q = 0; q < n; ++q)
        {
            if (!is_retained[static_cast<std::size_t>(q)])
            {
                environment_modes.push_back(q);
            }
        }

        std::vector<int> new_mode_order = kept_modes;
        new_mode_order.insert(new_mode_order.end(),
                              environment_modes.begin(),
                              environment_modes.end());

        std::vector<std::array<std::uint64_t, RHO3_DIM>> old_index_by_env(
            static_cast<std::size_t>(env_dim));
        std::vector<std::array<int, RHO3_DIM>> sign_by_env(
            static_cast<std::size_t>(env_dim));

        for (std::uint64_t env = 0; env < env_dim; ++env)
        {
            for (int local_basis = 0; local_basis < RHO3_DIM; ++local_basis)
            {
                std::vector<unsigned char> occ_by_mode(static_cast<std::size_t>(n), 0);
                for (int k = 0; k < RHO3_QUBITS; ++k)
                {
                    occ_by_mode[static_cast<std::size_t>(kept_modes[k])] =
                        static_cast<unsigned char>((local_basis >> k) & 1);
                }
                for (std::size_t e = 0; e < environment_modes.size(); ++e)
                {
                    occ_by_mode[static_cast<std::size_t>(environment_modes[e])] =
                        static_cast<unsigned char>((env >> e) & std::uint64_t{1});
                }

                old_index_by_env[static_cast<std::size_t>(env)][local_basis] =
                    index_from_occupations_little(occ_by_mode);
                sign_by_env[static_cast<std::size_t>(env)][local_basis] =
                    fermionic_reorder_sign_from_occupations(occ_by_mode, new_mode_order);
            }
        }

        for (std::uint64_t env = 0; env < env_dim; ++env)
        {
            const auto &old_index = old_index_by_env[static_cast<std::size_t>(env)];
            const auto &sign = sign_by_env[static_cast<std::size_t>(env)];
            for (int row = 0; row < RHO3_DIM; ++row)
            {
                const auto a = psi[old_index[row]] * static_cast<Real>(sign[row]);
                for (int col = 0; col < RHO3_DIM; ++col)
                {
                    const auto b = psi[old_index[col]] * static_cast<Real>(sign[col]);
                    const auto value = a * std::conj(b);
                    const std::size_t out =
                        s * RHO3_VALUES +
                        2u * static_cast<std::size_t>(row * RHO3_DIM + col);
                    rho_ri[out + 0] += static_cast<double>(std::real(value));
                    rho_ri[out + 1] += static_cast<double>(std::imag(value));
                }
            }
        }
    }
}


inline std::vector<int> cudaq_basis_bits_from_index(int n, std::uint64_t basis_index)
{
    std::vector<int> bits(static_cast<std::size_t>(n), 0);
    const bool reverse = mipt::backend::amplitude_reverse_bits() != 0;
    for (int q = 0; q < n; ++q)
    {
        const int bit = static_cast<int>((basis_index >> q) & std::uint64_t{1});
        bits[static_cast<std::size_t>(reverse ? (n - 1 - q) : q)] = bit;
    }
    return bits;
}

inline void cudaq_state_amplitudes_by_indices(cudaq::state &state,
                                              int n,
                                              const std::vector<std::uint64_t> &basis_indices,
                                              std::vector<std::complex<double>> &out)
{
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> basis_states;
    basis_states.reserve(basis_indices.size());
    for (const std::uint64_t idx : basis_indices)
    {
        basis_states.push_back(cudaq_basis_bits_from_index(n, idx));
    }
    out = state.amplitudes(basis_states);
    if (out.size() != basis_indices.size())
    {
        throw std::runtime_error("CUDA-Q state.amplitudes returned an unexpected number of amplitudes.");
    }
    if (mipt::backend::verbose_enabled())
    {
        mipt::backend::verbose_log(
            "state.amplitudes batch_size=" + std::to_string(basis_indices.size()) +
            " elapsed_s=" + std::to_string(mipt::backend::seconds_since(t0)));
    }
}

inline std::complex<double> cudaq_state_amplitude_by_index(cudaq::state &state,
                                                           int n,
                                                           std::uint64_t basis_index)
{
    std::vector<std::uint64_t> indices{basis_index};
    std::vector<std::complex<double>> amplitudes;
    cudaq_state_amplitudes_by_indices(state, n, indices, amplitudes);
    return amplitudes.front();
}

inline void normalize_rho3_subsystems(double *rho_ri, std::size_t subsystem_count)
{
    for (std::size_t s = 0; s < subsystem_count; ++s)
    {
        double trace = 0.0;
        const std::size_t base = s * RHO3_VALUES;
        for (int i = 0; i < RHO3_DIM; ++i)
        {
            trace += rho_ri[base + 2u * static_cast<std::size_t>(i * RHO3_DIM + i) + 0];
        }
        if (!(trace > 0.0) || !std::isfinite(trace))
        {
            continue;
        }
        const double inv_trace = 1.0 / trace;
        for (std::size_t k = 0; k < RHO3_VALUES; ++k)
        {
            rho_ri[base + k] *= inv_trace;
        }
    }
}

inline void reduced_density_complex_from_cudaq_amplitudes(
    cudaq::state &state,
    int n,
    const std::vector<Subsystem> &subsystems,
    double *rho_ri,
    bool fermion_trace = false)
{
    if (rho_ri == nullptr || subsystems.empty())
    {
        throw std::invalid_argument("Invalid amplitude-query RDM output definition.");
    }
    const std::uint64_t env_dim = checked_pow2_u64(n - RHO3_QUBITS);
    const std::size_t output_values = subsystems.size() * RHO3_VALUES;
    std::fill(rho_ri, rho_ri + output_values, 0.0);

    const bool exact_env_sum = n <= static_cast<int>(mipt::backend::mps_exact_observable_max_qubits());
    const long requested_samples = mipt::backend::mps_rdm_mc_samples();
    if (!exact_env_sum && requested_samples <= 0)
    {
        throw std::runtime_error(
            "Tensor/MPS amplitude-query RDM extraction would require exact summation over 2^(N-3) environment states. "
            "Set MIPT_MPS_RDM_MC_SAMPLES to a positive value for the approximate Monte Carlo estimator, "
            "or raise MIPT_MPS_EXACT_OBSERVABLE_MAX_QUBITS for exact summation.");
    }
    const std::uint64_t env_iterations = exact_env_sum
                                             ? env_dim
                                             : static_cast<std::uint64_t>(requested_samples);
    const double weight = exact_env_sum ? 1.0 : (static_cast<double>(env_dim) / static_cast<double>(env_iterations));
    mipt::backend::warn_mps_amplitude_path_once("Three-qubit RDM extraction",
                                               n,
                                               !exact_env_sum,
                                               requested_samples);

    thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<std::uint64_t> env_dist(0, env_dim - 1);

    for (std::size_t s = 0; s < subsystems.size(); ++s)
    {
        const auto &subsystem = subsystems[s];
        for (int i = 0; i < RHO3_QUBITS; ++i)
        {
            if (subsystem[i] < 0 || subsystem[i] >= n)
            {
                throw std::invalid_argument("Subsystem qubit index is out of range.");
            }
            for (int j = 0; j < i; ++j)
            {
                if (subsystem[i] == subsystem[j])
                {
                    throw std::invalid_argument("Subsystem contains duplicate qubits.");
                }
            }
        }

        std::array<std::uint64_t, RHO3_DIM> retained_offsets{};
        for (int local_basis = 0; local_basis < RHO3_DIM; ++local_basis)
        {
            for (int k = 0; k < RHO3_QUBITS; ++k)
            {
                retained_offsets[local_basis] |=
                    (static_cast<std::uint64_t>((local_basis >> k) & 1) << subsystem[k]);
            }
        }

        std::vector<int> kept_modes(subsystem.begin(), subsystem.end());
        std::vector<unsigned char> is_retained(static_cast<std::size_t>(n), 0);
        for (int q : kept_modes)
        {
            is_retained[static_cast<std::size_t>(q)] = 1;
        }

        std::vector<int> environment_modes;
        environment_modes.reserve(static_cast<std::size_t>(n - RHO3_QUBITS));
        for (int q = 0; q < n; ++q)
        {
            if (!is_retained[static_cast<std::size_t>(q)])
            {
                environment_modes.push_back(q);
            }
        }

        std::vector<int> new_mode_order = kept_modes;
        new_mode_order.insert(new_mode_order.end(),
                              environment_modes.begin(),
                              environment_modes.end());

        const std::size_t amplitude_batch = static_cast<std::size_t>(mipt::backend::mps_amplitude_batch_size());
        const std::uint64_t envs_per_batch = std::max<std::uint64_t>(
            1u, static_cast<std::uint64_t>(amplitude_batch / RHO3_DIM));
        std::vector<std::uint64_t> basis_indices;
        std::vector<int> basis_signs;
        std::vector<std::complex<double>> amplitudes;

        if (mipt::backend::verbose_enabled())
        {
            mipt::backend::verbose_log(
                "RDM subsystem=" + std::to_string(s) +
                " env_iterations=" + std::to_string(env_iterations) +
                " envs_per_batch=" + std::to_string(envs_per_batch) +
                " fermion_trace=" + std::to_string(fermion_trace ? 1 : 0));
        }

        for (std::uint64_t iter0 = 0; iter0 < env_iterations; iter0 += envs_per_batch)
        {
            const std::uint64_t chunk_envs = std::min<std::uint64_t>(envs_per_batch, env_iterations - iter0);
            basis_indices.clear();
            basis_signs.clear();
            basis_indices.reserve(static_cast<std::size_t>(chunk_envs) * RHO3_DIM);
            basis_signs.reserve(static_cast<std::size_t>(chunk_envs) * RHO3_DIM);

            for (std::uint64_t local_iter = 0; local_iter < chunk_envs; ++local_iter)
            {
                const std::uint64_t iter = iter0 + local_iter;
                const std::uint64_t env = exact_env_sum ? iter : env_dist(rng);

                if (!fermion_trace)
                {
                    std::uint64_t base = 0;
                    for (std::size_t e = 0; e < environment_modes.size(); ++e)
                    {
                        base |= ((env >> e) & std::uint64_t{1}) << environment_modes[e];
                    }
                    for (int row = 0; row < RHO3_DIM; ++row)
                    {
                        basis_indices.push_back(base | retained_offsets[row]);
                        basis_signs.push_back(1);
                    }
                }
                else
                {
                    for (int local_basis = 0; local_basis < RHO3_DIM; ++local_basis)
                    {
                        std::vector<unsigned char> occ_by_mode(static_cast<std::size_t>(n), 0);
                        for (int k = 0; k < RHO3_QUBITS; ++k)
                        {
                            occ_by_mode[static_cast<std::size_t>(kept_modes[k])] =
                                static_cast<unsigned char>((local_basis >> k) & 1);
                        }
                        for (std::size_t e = 0; e < environment_modes.size(); ++e)
                        {
                            occ_by_mode[static_cast<std::size_t>(environment_modes[e])] =
                                static_cast<unsigned char>((env >> e) & std::uint64_t{1});
                        }
                        basis_indices.push_back(index_from_occupations_little(occ_by_mode));
                        basis_signs.push_back(fermionic_reorder_sign_from_occupations(occ_by_mode, new_mode_order));
                    }
                }
            }

            cudaq_state_amplitudes_by_indices(state, n, basis_indices, amplitudes);

            for (std::uint64_t local_iter = 0; local_iter < chunk_envs; ++local_iter)
            {
                std::array<std::complex<double>, RHO3_DIM> amp{};
                const std::size_t amp_base = static_cast<std::size_t>(local_iter) * RHO3_DIM;
                for (int row = 0; row < RHO3_DIM; ++row)
                {
                    amp[row] = amplitudes[amp_base + static_cast<std::size_t>(row)] *
                               static_cast<double>(basis_signs[amp_base + static_cast<std::size_t>(row)]);
                }

                for (int row = 0; row < RHO3_DIM; ++row)
                {
                    const auto a = amp[row];
                    for (int col = 0; col < RHO3_DIM; ++col)
                    {
                        const auto value = weight * a * std::conj(amp[col]);
                        const std::size_t out =
                            s * RHO3_VALUES +
                            2u * static_cast<std::size_t>(row * RHO3_DIM + col);
                        rho_ri[out + 0] += std::real(value);
                        rho_ri[out + 1] += std::imag(value);
                    }
                }
            }
        }
    }

    if (!exact_env_sum)
    {
        normalize_rho3_subsystems(rho_ri, subsystems.size());
    }
}

inline void cudaq_three_site_density_matrices(
    cudaq::state &state,
    int n,
    const std::vector<Subsystem> &subsystems,
    double *rho_ri,
    bool fermion_trace = false)
{
    if (rho_ri == nullptr || subsystems.empty())
    {
        throw std::invalid_argument("Density-matrix output definition is empty.");
    }
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument("Three-qubit RDM output requires at least three qubits.");
    }

    const std::uint64_t dim_u64 = checked_pow2_u64(n);
    if (dim_u64 >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("State-vector dimension does not fit in size_t.");
    }
    const std::size_t dim = static_cast<std::size_t>(dim_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();

    if (mipt::backend::verbose_enabled())
    {
        mipt::backend::verbose_log(
            "cudaq_three_site_density_matrices: n=" + std::to_string(n) +
            " num_tensors=" + std::to_string(state.get_num_tensors()) +
            " tensor0_rank=" + std::to_string(tensor.get_rank()) +
            " tensor0_elements=" + std::to_string(tensor.get_num_elements()) +
            " tensor0_data=" + std::to_string(tensor.data != nullptr ? 1 : 0) +
            " state_on_gpu=" + std::to_string(state.is_on_gpu() ? 1 : 0));
    }

    const bool has_dense_rank1_tensor =
        tensor.get_rank() == 1 && tensor.get_num_elements() >= dim && tensor.data != nullptr;
    if (!has_dense_rank1_tensor)
    {
        reduced_density_complex_from_cudaq_amplitudes(
            state, n, subsystems, rho_ri, fermion_trace);
        return;
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    if (state.is_on_gpu() && tensor.data != nullptr)
    {
        const std::vector<int> flat = flatten_subsystems(subsystems);
        int status = 0;
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = fermion_trace
                         ? mipt_cuda_rho3_subsystems_complex_fermion_f64(
                               tensor.data, n, flat.data(),
                               static_cast<int>(subsystems.size()), rho_ri)
                         : mipt_cuda_rho3_subsystems_complex_f64(
                               tensor.data, n, flat.data(),
                               static_cast<int>(subsystems.size()), rho_ri);
        }
        else
        {
            status = fermion_trace
                         ? mipt_cuda_rho3_subsystems_complex_fermion_f32(
                               tensor.data, n, flat.data(),
                               static_cast<int>(subsystems.size()), rho_ri)
                         : mipt_cuda_rho3_subsystems_complex_f32(
                               tensor.data, n, flat.data(),
                               static_cast<int>(subsystems.size()), rho_ri);
        }
        if (status != 0)
        {
            throw std::runtime_error(
                std::string("CUDA three-qubit ") +
                (fermion_trace ? "fermionic " : "") +
                "reduced-density-matrix kernel failed; status=" +
                std::to_string(status));
        }
        return;
    }
#endif

    if (state.is_on_gpu())
    {
        // Without the custom reduction kernel, copy the statevector once
        // and form every requested retained-subsystem matrix on the host.
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            std::vector<std::complex<double>> host_state(dim);
            state.to_host(host_state.data(), host_state.size());
            reduced_density_complex_from_host_statevector(
                host_state.data(), n, subsystems, rho_ri, fermion_trace);
        }
        else
        {
            std::vector<std::complex<float>> host_state(dim);
            state.to_host(host_state.data(), host_state.size());
            reduced_density_complex_from_host_statevector(
                host_state.data(), n, subsystems, rho_ri, fermion_trace);
        }
        return;
    }

    if (tensor.data == nullptr)
    {
        throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        reduced_density_complex_from_host_statevector(
            reinterpret_cast<const std::complex<double> *>(tensor.data),
            n, subsystems, rho_ri, fermion_trace);
    }
    else
    {
        reduced_density_complex_from_host_statevector(
            reinterpret_cast<const std::complex<float> *>(tensor.data),
            n, subsystems, rho_ri, fermion_trace);
    }
}

// Both trace conventions from one statevector sweep.
//
// The fermionic sign factorizes into a per-basis-state factor -- see
// cuda/reduce_sweep.cuh -- so on the CUDA path this costs one set of amplitude
// loads rather than two. Everywhere else it is exactly the two-call form, and
// callers get the same numbers either way.
inline void cudaq_three_site_density_matrices_both(
    cudaq::state &state,
    int n,
    const std::vector<Subsystem> &subsystems,
    double *rho_ri,
    double *fermion_rho_ri)
{
    if (rho_ri == nullptr || fermion_rho_ri == nullptr || subsystems.empty())
    {
        throw std::invalid_argument("Density-matrix output definition is empty.");
    }
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument("Three-qubit RDM output requires at least three qubits.");
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    const std::uint64_t dim_u64 = checked_pow2_u64(n);
    if (dim_u64 <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        const std::size_t dim = static_cast<std::size_t>(dim_u64);
        const auto tensor = state.get_tensor();
        const bool has_dense_rank1_tensor =
            tensor.get_rank() == 1 && tensor.get_num_elements() >= dim && tensor.data != nullptr;
        if (has_dense_rank1_tensor && state.is_on_gpu())
        {
            const std::vector<int> flat = flatten_subsystems(subsystems);
            const int count = static_cast<int>(subsystems.size());
            const int status =
                (state.get_precision() == cudaq::SimulationState::precision::fp64)
                    ? mipt_cuda_rho3_subsystems_complex_both_f64(
                          tensor.data, n, flat.data(), count, rho_ri, fermion_rho_ri)
                    : mipt_cuda_rho3_subsystems_complex_both_f32(
                          tensor.data, n, flat.data(), count, rho_ri, fermion_rho_ri);
            if (status != 0)
            {
                throw std::runtime_error(
                    "CUDA three-qubit dual-trace reduced-density-matrix kernel failed; status=" +
                    std::to_string(status));
            }
            return;
        }
    }
#endif

    cudaq_three_site_density_matrices(state, n, subsystems, rho_ri, false);
    cudaq_three_site_density_matrices(state, n, subsystems, fermion_rho_ri, true);
}


inline mipt::io::DensityFileMetadata density_metadata(
    int dimension,
    int total_qubits,
    int periods,
    int realizations,
    int res,
    double p_min,
    double p_max,
    int grid_x,
    int grid_y,
    const std::vector<Subsystem> &subsystems)
{
    mipt::io::DensityFileMetadata metadata;
    metadata.spatial_dimension = static_cast<std::uint32_t>(dimension);
    metadata.total_qubits = static_cast<std::uint32_t>(total_qubits);
    metadata.kept_qubits = RHO3_QUBITS;
    metadata.matrix_dimension = RHO3_DIM;
    metadata.periods = static_cast<std::uint32_t>(periods);
    metadata.realizations = static_cast<std::uint32_t>(realizations);
    metadata.resolution = static_cast<std::uint32_t>(res);
    metadata.grid_x = static_cast<std::uint32_t>(grid_x);
    metadata.grid_y = static_cast<std::uint32_t>(grid_y);
    metadata.subsystem_count = static_cast<std::uint32_t>(subsystems.size());
    metadata.record_count =
        static_cast<std::uint64_t>(realizations) *
        static_cast<std::uint64_t>(res);
    metadata.p_min = p_min;
    metadata.p_max = p_max;
    metadata.subsystem_qubits.reserve(subsystems.size() * RHO3_QUBITS);
    for (const auto &subsystem : subsystems)
    {
        for (int q : subsystem)
        {
            metadata.subsystem_qubits.push_back(static_cast<std::uint32_t>(q));
        }
    }
    return metadata;
}


struct TmiMatrixTerm
{
    std::string name;
    std::vector<int> modes;
};

inline std::uint64_t term_matrix_dim_u64(const TmiMatrixTerm &term)
{
    return checked_pow2_u64(static_cast<int>(term.modes.size()));
}

inline std::size_t term_value_count(const TmiMatrixTerm &term)
{
    const std::uint64_t d64 = term_matrix_dim_u64(term);
    if (d64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("TMI matrix dimension does not fit in size_t.");
    }
    const std::size_t d = static_cast<std::size_t>(d64);
    if (d != 0 && d > std::numeric_limits<std::size_t>::max() / d / 2u)
    {
        throw std::invalid_argument("TMI matrix payload is too large for this host.");
    }
    return 2u * d * d;
}

inline std::size_t tmi_payload_value_count(const std::vector<TmiMatrixTerm> &terms)
{
    std::size_t total = 0;
    for (const auto &term : terms)
    {
        total += term_value_count(term);
    }
    return total;
}

inline std::vector<int> mode_range(int begin, int count)
{
    std::vector<int> modes;
    modes.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        modes.push_back(begin + i);
    }
    return modes;
}

inline std::vector<int> concatenate_modes(const std::vector<int> &a,
                                   const std::vector<int> &b)
{
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

template <typename Real>
inline void append_reduced_density_for_modes_from_host_statevector(
    const std::complex<Real> *psi,
    int n,
    const std::vector<int> &modes,
    double *rho_ri,
    bool fermion_trace = false)
{
    if (psi == nullptr || rho_ri == nullptr)
    {
        throw std::invalid_argument("Null state-vector or TMI density-matrix pointer.");
    }
    if (modes.empty())
    {
        throw std::invalid_argument("TMI reduced-density term cannot have zero retained modes.");
    }

    const int kept = static_cast<int>(modes.size());
    const std::uint64_t kept_dim_u64 = checked_pow2_u64(kept);
    const std::uint64_t env_dim = checked_pow2_u64(n - kept);
    if (kept_dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("TMI retained dimension does not fit in size_t.");
    }
    const std::size_t kept_dim = static_cast<std::size_t>(kept_dim_u64);

    std::vector<unsigned char> is_retained(static_cast<std::size_t>(n), 0);
    for (int q : modes)
    {
        if (q < 0 || q >= n)
        {
            throw std::invalid_argument("TMI subsystem mode index is out of range.");
        }
        if (is_retained[static_cast<std::size_t>(q)])
        {
            throw std::invalid_argument("TMI subsystem contains duplicate modes.");
        }
        is_retained[static_cast<std::size_t>(q)] = 1;
    }

    std::vector<int> environment_modes;
    environment_modes.reserve(static_cast<std::size_t>(n - kept));
    for (int q = 0; q < n; ++q)
    {
        if (!is_retained[static_cast<std::size_t>(q)])
        {
            environment_modes.push_back(q);
        }
    }

    std::fill(rho_ri, rho_ri + 2u * kept_dim * kept_dim, 0.0);

    if (!fermion_trace)
    {
        std::vector<std::uint64_t> retained_offsets(kept_dim, 0);
        for (std::uint64_t local_basis = 0; local_basis < kept_dim_u64; ++local_basis)
        {
            std::uint64_t offset = 0;
            for (int k = 0; k < kept; ++k)
            {
                offset |= ((local_basis >> k) & std::uint64_t{1})
                          << modes[static_cast<std::size_t>(k)];
            }
            retained_offsets[static_cast<std::size_t>(local_basis)] = offset;
        }

        for (std::uint64_t env = 0; env < env_dim; ++env)
        {
            std::uint64_t base = 0;
            for (std::size_t e = 0; e < environment_modes.size(); ++e)
            {
                base |= ((env >> e) & std::uint64_t{1})
                        << environment_modes[e];
            }

            for (std::uint64_t row = 0; row < kept_dim_u64; ++row)
            {
                const auto a = psi[base | retained_offsets[static_cast<std::size_t>(row)]];
                for (std::uint64_t col = 0; col < kept_dim_u64; ++col)
                {
                    const auto b = psi[base | retained_offsets[static_cast<std::size_t>(col)]];
                    const auto value = a * std::conj(b);
                    const std::size_t out = 2u * static_cast<std::size_t>(row * kept_dim_u64 + col);
                    rho_ri[out + 0] += static_cast<double>(std::real(value));
                    rho_ri[out + 1] += static_cast<double>(std::imag(value));
                }
            }
        }
        return;
    }

    std::vector<int> new_mode_order = modes;
    new_mode_order.insert(new_mode_order.end(),
                          environment_modes.begin(),
                          environment_modes.end());

    std::vector<std::uint64_t> old_index_by_new_basis(
        static_cast<std::size_t>(kept_dim_u64 * env_dim));
    std::vector<int> sign_by_new_basis(
        static_cast<std::size_t>(kept_dim_u64 * env_dim));

    for (std::uint64_t env = 0; env < env_dim; ++env)
    {
        for (std::uint64_t local_basis = 0; local_basis < kept_dim_u64; ++local_basis)
        {
            std::vector<unsigned char> occ_by_mode(static_cast<std::size_t>(n), 0);
            for (int k = 0; k < kept; ++k)
            {
                occ_by_mode[static_cast<std::size_t>(modes[static_cast<std::size_t>(k)])] =
                    static_cast<unsigned char>((local_basis >> k) & std::uint64_t{1});
            }
            for (std::size_t e = 0; e < environment_modes.size(); ++e)
            {
                occ_by_mode[static_cast<std::size_t>(environment_modes[e])] =
                    static_cast<unsigned char>((env >> e) & std::uint64_t{1});
            }

            const std::size_t idx = static_cast<std::size_t>(env * kept_dim_u64 + local_basis);
            old_index_by_new_basis[idx] = index_from_occupations_little(occ_by_mode);
            sign_by_new_basis[idx] =
                fermionic_reorder_sign_from_occupations(occ_by_mode, new_mode_order);
        }
    }

    for (std::uint64_t env = 0; env < env_dim; ++env)
    {
        const std::size_t block = static_cast<std::size_t>(env * kept_dim_u64);
        for (std::uint64_t row = 0; row < kept_dim_u64; ++row)
        {
            const std::size_t row_idx = block + static_cast<std::size_t>(row);
            const auto a = psi[old_index_by_new_basis[row_idx]] *
                           static_cast<Real>(sign_by_new_basis[row_idx]);
            for (std::uint64_t col = 0; col < kept_dim_u64; ++col)
            {
                const std::size_t col_idx = block + static_cast<std::size_t>(col);
                const auto b = psi[old_index_by_new_basis[col_idx]] *
                               static_cast<Real>(sign_by_new_basis[col_idx]);
                const auto value = a * std::conj(b);
                const std::size_t out = 2u * static_cast<std::size_t>(row * kept_dim_u64 + col);
                rho_ri[out + 0] += static_cast<double>(std::real(value));
                rho_ri[out + 1] += static_cast<double>(std::imag(value));
            }
        }
    }
}

template <typename Real>
inline void tmi_density_matrices_from_host_statevector(
    const std::complex<Real> *psi,
    int n,
    const std::vector<TmiMatrixTerm> &terms,
    std::vector<double> &payload,
    bool fermion_trace = false)
{
    payload.resize(tmi_payload_value_count(terms));
    std::size_t offset = 0;
    for (const auto &term : terms)
    {
        append_reduced_density_for_modes_from_host_statevector(
            psi, n, term.modes, payload.data() + offset, fermion_trace);
        offset += term_value_count(term);
    }
}

} // namespace mipt
