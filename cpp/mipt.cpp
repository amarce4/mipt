#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include "density_io.hpp"
#include "mipt_backend.hpp"
#include <vector>
#include <random>
#include <iostream>
#include <complex>
#include <cmath>
#include <set>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstring>
#include <sstream>

#ifdef MIPT_ENABLE_CUDA_RHO
#include "rho_reduce.hpp"
#endif

namespace
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

    std::vector<Subsystem> periodic_ring_three_site_subsystems(int n)
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

    std::vector<Subsystem> three_site_subsystems_2d(int x, int y)
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

    std::vector<Subsystem> terminal_three_site_subsystem(int n)
    {
        if (n < RHO3_QUBITS)
        {
            throw std::invalid_argument("Three-qubit RDM output requires at least three qubits.");
        }
        return {{n - 3, n - 2, n - 1}};
    }

    std::vector<int> flatten_subsystems(const std::vector<Subsystem> &subsystems)
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
    void reduced_density_complex_from_host_statevector(
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
        const bool reverse = mipt_backend::amplitude_reverse_bits() != 0;
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
        if (mipt_backend::verbose_enabled())
        {
            mipt_backend::verbose_log(
                "state.amplitudes batch_size=" + std::to_string(basis_indices.size()) +
                " elapsed_s=" + std::to_string(mipt_backend::seconds_since(t0)));
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

    void normalize_rho3_subsystems(double *rho_ri, std::size_t subsystem_count)
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

    void reduced_density_complex_from_cudaq_amplitudes(
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

        const bool exact_env_sum = n <= static_cast<int>(mipt_backend::mps_exact_observable_max_qubits());
        const long requested_samples = mipt_backend::mps_rdm_mc_samples();
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
        mipt_backend::warn_mps_amplitude_path_once("Three-qubit RDM extraction",
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

            const std::size_t amplitude_batch = static_cast<std::size_t>(mipt_backend::mps_amplitude_batch_size());
            const std::uint64_t envs_per_batch = std::max<std::uint64_t>(
                1u, static_cast<std::uint64_t>(amplitude_batch / RHO3_DIM));
            std::vector<std::uint64_t> basis_indices;
            std::vector<int> basis_signs;
            std::vector<std::complex<double>> amplitudes;

            if (mipt_backend::verbose_enabled())
            {
                mipt_backend::verbose_log(
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

    void cudaq_three_site_density_matrices(
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

        if (mipt_backend::verbose_enabled())
        {
            mipt_backend::verbose_log(
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

    mipt_io::DensityFileMetadata density_metadata(
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
        mipt_io::DensityFileMetadata metadata;
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

    std::uint64_t term_matrix_dim_u64(const TmiMatrixTerm &term)
    {
        return checked_pow2_u64(static_cast<int>(term.modes.size()));
    }

    std::size_t term_value_count(const TmiMatrixTerm &term)
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

    std::size_t tmi_payload_value_count(const std::vector<TmiMatrixTerm> &terms)
    {
        std::size_t total = 0;
        for (const auto &term : terms)
        {
            total += term_value_count(term);
        }
        return total;
    }

    std::vector<int> mode_range(int begin, int count)
    {
        std::vector<int> modes;
        modes.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i)
        {
            modes.push_back(begin + i);
        }
        return modes;
    }

    std::vector<int> concatenate_modes(const std::vector<int> &a,
                                       const std::vector<int> &b)
    {
        std::vector<int> out;
        out.reserve(a.size() + b.size());
        out.insert(out.end(), a.begin(), a.end());
        out.insert(out.end(), b.begin(), b.end());
        return out;
    }

    template <typename Real>
    void append_reduced_density_for_modes_from_host_statevector(
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
    void tmi_density_matrices_from_host_statevector(
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


}


template <typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                              // are exactly the same as the input
    return linspaced;
}

struct LayerData
{
    int start = 0;

    std::vector<int> measure_flags;

    // Exactly one of these should be 1 for each qubit.
    std::vector<int> rot_x_flags;
    std::vector<int> rot_y_flags;
    std::vector<int> rot_xy_flags;
};

struct FermionU2Params
{
    // U = exp(i global_phase) * U3(theta, phi, lambda).
    double global_phase = 0.0;
    double theta = 0.0;
    double phi = 0.0;
    double lambda = 0.0;
};

struct FermionBondGate
{
    // kind = 0: adjacent parity-preserving two-mode unitary.
    // kind = 1: adjacent fermionic swap.
    // kind = 2: non-adjacent parity-preserving two-mode unitary with
    //           Jordan-Wigner CZ-string corrections generated in the kernel.
    int kind = 0;
    int q0 = 0;
    int q1 = 0;
    FermionU2Params even;
    FermionU2Params odd;
};

struct FermionLayerData
{
    std::vector<FermionBondGate> gates;
    std::vector<int> measure_flags;
};

struct MIPTFermionKernel_1D
{
    void operator()(int n,
                    const std::vector<FermionLayerData> &layers) __qpu__
    {
        cudaq::qvector q(n);

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
};

struct MIPTKernel_1D
{
    void operator()(int n,
                    const std::vector<LayerData> &layers,
                    bool closed) __qpu__
    {
        cudaq::qvector q(n);

        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            int start = layers[layer].start;

            for (int i = start; i < n - 1; i += 2)
            {
                // Local rotation on q[i]
                if (layers[layer].rot_x_flags[i])
                {
                    rx(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_y_flags[i])
                {
                    ry(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_xy_flags[i])
                {
                    rz(-0.78539816339744830962, q[i]);
                    rx(1.57079632679489661923, q[i]);
                    rz(0.78539816339744830962, q[i]);
                }

                // Local rotation on q[i + 1]
                if (layers[layer].rot_x_flags[i + 1])
                {
                    rx(1.57079632679489661923, q[i + 1]);
                }

                if (layers[layer].rot_y_flags[i + 1])
                {
                    ry(1.57079632679489661923, q[i + 1]);
                }

                if (layers[layer].rot_xy_flags[i + 1])
                {
                    rz(-0.78539816339744830962, q[i + 1]);
                    rx(1.57079632679489661923, q[i + 1]);
                    rz(0.78539816339744830962, q[i + 1]);
                }

                // exp(-i*pi/4 X_i X_{i+1})
                h(q[i]);
                h(q[i + 1]);

                cx(q[i], q[i + 1]);
                rz(1.57079632679489661923, q[i + 1]);
                cx(q[i], q[i + 1]);

                h(q[i]);
                h(q[i + 1]);
            }

            // closed odd-layer closure: bond (n - 1, 0)
            if (closed && start == 1 && n > 2)
            {
                int i = n - 1;
                int j = 0;

                // Local rotation on q[n - 1]
                if (layers[layer].rot_x_flags[i])
                {
                    rx(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_y_flags[i])
                {
                    ry(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_xy_flags[i])
                {
                    rz(-0.78539816339744830962, q[i]);
                    rx(1.57079632679489661923, q[i]);
                    rz(0.78539816339744830962, q[i]);
                }

                // Local rotation on q[0]
                if (layers[layer].rot_x_flags[j])
                {
                    rx(1.57079632679489661923, q[j]);
                }

                if (layers[layer].rot_y_flags[j])
                {
                    ry(1.57079632679489661923, q[j]);
                }

                if (layers[layer].rot_xy_flags[j])
                {
                    rz(-0.78539816339744830962, q[j]);
                    rx(1.57079632679489661923, q[j]);
                    rz(0.78539816339744830962, q[j]);
                }

                // exp(-i*pi/4 X_{n-1} X_0)
                h(q[i]);
                h(q[j]);

                cx(q[i], q[j]);
                rz(1.57079632679489661923, q[j]);
                cx(q[i], q[j]);

                h(q[i]);
                h(q[j]);
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
};

struct MIPTKernel_2D
{
    void operator()(int x, int y,
                    const std::vector<LayerData> &layers,
                    bool closed) __qpu__
    {
        int n = x * y;
        cudaq::qvector q(n);

        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            int start = layers[layer].start;

            if (layer % 4 == 0 || layer % 4 == 1)
            {
                for (int i = 0; i < y; ++i)
                {
                    for (int j = start; j < x-1; j += 2)
                    {
                        int idx = i * x + j;
                        // Local rotation on q[idx]
                        if (layers[layer].rot_x_flags[idx])
                        {
                            rx(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_y_flags[idx])
                        {
                            ry(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_xy_flags[idx])
                        {
                            rz(-0.78539816339744830962, q[idx]);
                            rx(1.57079632679489661923, q[idx]);
                            rz(0.78539816339744830962, q[idx]);
                        }

                        // Local rotation on q[idx + 1]
                        if (layers[layer].rot_x_flags[idx + 1])
                        {
                            rx(1.57079632679489661923, q[idx + 1]);
                        }

                        if (layers[layer].rot_y_flags[idx + 1])
                        {
                            ry(1.57079632679489661923, q[idx + 1]);
                        }

                        if (layers[layer].rot_xy_flags[idx + 1])
                        {
                            rz(-0.78539816339744830962, q[idx + 1]);
                            rx(1.57079632679489661923, q[idx + 1]);
                            rz(0.78539816339744830962, q[idx + 1]);
                        }

                        // exp(-i*pi/4 X_i X_{i+1})
                        h(q[idx]);
                        h(q[idx + 1]);

                        cx(q[idx], q[idx + 1]);
                        rz(1.57079632679489661923, q[idx + 1]);
                        cx(q[idx], q[idx + 1]);

                        h(q[idx]);
                        h(q[idx + 1]);
                    }

                    // closed odd-layer closure: bond (n - 1, 0)
                    if (closed && start == 1 && x > 2)
                    {
                        int k = i*x;
                        int l = i*x + x - 1;

                        // Local rotation on q[k]
                        if (layers[layer].rot_x_flags[k])
                        {
                            rx(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_y_flags[k])
                        {
                            ry(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_xy_flags[k])
                        {
                            rz(-0.78539816339744830962, q[k]);
                            rx(1.57079632679489661923, q[k]);
                            rz(0.78539816339744830962, q[k]);
                        }

                        // Local rotation on q[0]
                        if (layers[layer].rot_x_flags[l])
                        {
                            rx(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_y_flags[l])
                        {
                            ry(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_xy_flags[l])
                        {
                            rz(-0.78539816339744830962, q[l]);
                            rx(1.57079632679489661923, q[l]);
                            rz(0.78539816339744830962, q[l]);
                        }

                        // exp(-i*pi/4 X_{n-1} X_0)
                        h(q[k]);
                        h(q[l]);

                        cx(q[k], q[l]);
                        rz(1.57079632679489661923, q[l]);
                        cx(q[k], q[l]);

                        h(q[k]);
                        h(q[l]);
                    }
                }
            }
            else
            {
                for (int i = 0; i < x; ++i)
                {
                    int x_start = (i % 2 == 0) ? 0 : 1;
                    x_start = (x_start + layers[layer].start) % 2;
                    for (int j = x_start; j < y-1; j += 2)
                    {
                        int idx = j * y + i;
                        int idx_next = (j + 1) * y + i;
                        // Local rotation on q[idx]
                        if (layers[layer].rot_x_flags[idx])
                        {
                            rx(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_y_flags[idx])
                        {
                            ry(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_xy_flags[idx])
                        {
                            rz(-0.78539816339744830962, q[idx]);
                            rx(1.57079632679489661923, q[idx]);
                            rz(0.78539816339744830962, q[idx]);
                        }

                        // Local rotation on q[idx_next]
                        if (layers[layer].rot_x_flags[idx_next])
                        {
                            rx(1.57079632679489661923, q[idx_next]);
                        }

                        if (layers[layer].rot_y_flags[idx_next])
                        {
                            ry(1.57079632679489661923, q[idx_next]);
                        }

                        if (layers[layer].rot_xy_flags[idx_next])
                        {
                            rz(-0.78539816339744830962, q[idx_next]);
                            rx(1.57079632679489661923, q[idx_next]);
                            rz(0.78539816339744830962, q[idx_next]);
                        }

                        // exp(-i*pi/4 X_i X_{i+1})
                        h(q[idx]);
                        h(q[idx_next]);

                        cx(q[idx], q[idx_next]);
                        rz(1.57079632679489661923, q[idx_next]);
                        cx(q[idx], q[idx_next]);

                        h(q[idx]);
                        h(q[idx_next]);
                    }

                    // closed odd-layer closure: bond (n - 1, 0)
                    if (closed && x_start == 1 && y > 2)
                    {
                        int k = i;
                        int l = (y-1)*y + i;

                        // Local rotation on q[k]
                        if (layers[layer].rot_x_flags[k])
                        {
                            rx(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_y_flags[k])
                        {
                            ry(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_xy_flags[k])
                        {
                            rz(-0.78539816339744830962, q[k]);
                            rx(1.57079632679489661923, q[k]);
                            rz(0.78539816339744830962, q[k]);
                        }

                        // Local rotation on q[0]
                        if (layers[layer].rot_x_flags[l])
                        {
                            rx(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_y_flags[l])
                        {
                            ry(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_xy_flags[l])
                        {
                            rz(-0.78539816339744830962, q[l]);
                            rx(1.57079632679489661923, q[l]);
                            rz(0.78539816339744830962, q[l]);
                        }

                        // exp(-i*pi/4 X_{n-1} X_0)
                        h(q[k]);
                        h(q[l]);

                        cx(q[k], q[l]);
                        rz(1.57079632679489661923, q[l]);
                        cx(q[k], q[l]);

                        h(q[k]);
                        h(q[l]);
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
};

struct FRGSLayerData // LayerData for fermions with a reduced topological gate set
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

struct MIPTKernel_1D_FRGS
{
    void operator()(int n,
                    const std::vector<FRGSLayerData> &flayers,
                    bool closed) __qpu__
    {
        cudaq::qvector q(n);

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
};

int frgs_active_bond_count(int n, int start, bool closed = true)
{
    const int bond_stop = (closed && start == 1 && n > 2) ? n : (n - 1);
    if (bond_stop <= start)
    {
        return 0;
    }
    return (bond_stop - start + 1) / 2;
}

inline int frgs_rng_bit(std::mt19937 &rng)
{
    return static_cast<int>((rng() >> 31) & 1u);
}

void fill_frgs_mipt_layer(FRGSLayerData &layer,
                          int n,
                          int start,
                          double p,
                          std::mt19937 &rng,
                          bool closed = true)
{
    std::bernoulli_distribution measure_dist(p);

    layer.start = start;
    layer.measure_flags.resize(n);

    const int bond_count = frgs_active_bond_count(n, start, closed);
    layer.local_left_mask.resize(bond_count);
    layer.local_right_mask.resize(bond_count);

    for (int b = 0; b < bond_count; ++b)
    {
        // Each endpoint receives exactly one local gate: T or single-site
        // braid, selected with 50/50 odds.
        layer.local_left_mask[b] = frgs_rng_bit(rng) ? 1 : 2;
        layer.local_right_mask[b] = frgs_rng_bit(rng) ? 1 : 2;
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

FRGSLayerData make_frgs_mipt_layer(int n,
                                   int start,
                                   double p,
                                   std::mt19937 &rng,
                                   bool closed = true)
{
    FRGSLayerData layer;
    fill_frgs_mipt_layer(layer, n, start, p, rng, closed);
    return layer;
}

void frgs_mipt_frontend_inplace(std::vector<FRGSLayerData> &layers,
                                int n,
                                int periods,
                                double p,
                                std::mt19937 &rng,
                                bool closed = true)
{
    if ((n % 2) != 0)
    {
        throw std::invalid_argument("FRGS fermionic 1D simulation requires even n.");
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
        fill_frgs_mipt_layer(layers[idx++], n, 0, p, rng, closed);
        fill_frgs_mipt_layer(layers[idx++], n, 1, p, rng, closed);
    }

    if (add_extra_even_layer)
    {
        fill_frgs_mipt_layer(layers[idx++], n, 0, p, rng, closed);
    }
}

std::vector<FRGSLayerData> frgs_mipt_frontend(int n,
                                              int periods,
                                              double p,
                                              bool closed = true)
{
    std::mt19937 rng(std::random_device{}());
    std::vector<FRGSLayerData> layers;
    frgs_mipt_frontend_inplace(layers, n, periods, p, rng, closed);
    return layers;
}


void fill_mipt_layer(LayerData &layer,
                     int n,
                     int start,
                     double p,
                     std::mt19937 &rng)
{
    std::uniform_int_distribution<int> axis_dist(0, 2);
    std::bernoulli_distribution measure_dist(p);

    layer.start = start;
    layer.measure_flags.resize(n);
    layer.rot_x_flags.resize(n);
    layer.rot_y_flags.resize(n);
    layer.rot_xy_flags.resize(n);

    for (int q = 0; q < n; ++q)
    {
        const int axis = axis_dist(rng);
        layer.rot_x_flags[q] = (axis == 0) ? 1 : 0;
        layer.rot_y_flags[q] = (axis == 1) ? 1 : 0;
        layer.rot_xy_flags[q] = (axis == 2) ? 1 : 0;

        if (p <= 0.0)
        {
            layer.measure_flags[q] = 0;
        }
        else if (p >= 1.0)
        {
            layer.measure_flags[q] = 1;
        }
        else
        {
            layer.measure_flags[q] = measure_dist(rng) ? 1 : 0;
        }
    }
}

LayerData make_mipt_layer(int n,
                          int start,
                          double p,
                          std::mt19937 &rng)
{
    LayerData layer;
    fill_mipt_layer(layer, n, start, p, rng);
    return layer;
}

void mipt_frontend_inplace(std::vector<LayerData> &layers,
                           int n,
                           int periods,
                           double p,
                           std::mt19937 &rng)
{
    std::bernoulli_distribution extra_even_layer(0.5);
    const bool add_extra_even_layer = extra_even_layer(rng);
    const std::size_t layer_count = static_cast<std::size_t>(2 * periods + (add_extra_even_layer ? 1 : 0));
    layers.resize(layer_count);

    std::size_t idx = 0;
    for (int period = 0; period < periods; ++period)
    {
        fill_mipt_layer(layers[idx++], n, 0, p, rng);
        fill_mipt_layer(layers[idx++], n, 1, p, rng);
    }
    if (add_extra_even_layer)
    {
        fill_mipt_layer(layers[idx++], n, 0, p, rng);
    }
}

std::vector<LayerData> mipt_frontend(int n,
                                     int periods,
                                     double p)
{
    std::mt19937 rng(std::random_device{}());
    std::vector<LayerData> layers;
    layers.reserve(2 * periods + 1);
    mipt_frontend_inplace(layers, n, periods, p, rng);
    return layers;
}



std::array<std::complex<double>, 4> haar_unitary_2(std::mt19937 &rng)
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

FermionU2Params decompose_u2_to_u3_phase(const std::array<std::complex<double>, 4> &u)
{
    const auto a = u[0];
    const auto b = u[1];
    const auto c = u[2];
    const auto d = u[3];

    const double ca = std::abs(a);
    const double sa = std::abs(c);
    FermionU2Params out;
    out.theta = 2.0 * std::atan2(sa, ca);

    constexpr double eps = 1e-12;
    if (ca > eps && sa > eps)
    {
        out.global_phase = std::arg(a);
        out.phi = std::arg(c) - out.global_phase;
        out.lambda = std::arg(-b) - out.global_phase;
    }
    else if (ca > eps)
    {
        out.global_phase = std::arg(a);
        out.phi = 0.0;
        out.lambda = std::arg(d) - out.global_phase;
    }
    else
    {
        out.global_phase = 0.0;
        out.phi = std::arg(c);
        out.lambda = std::arg(-b);
    }

    return out;
}

FermionBondGate random_parity_preserving_gate(int q0, int q1, std::mt19937 &rng)
{
    FermionBondGate gate;
    gate.kind = 0;
    gate.q0 = q0;
    gate.q1 = q1;
    gate.even = decompose_u2_to_u3_phase(haar_unitary_2(rng));
    gate.odd = decompose_u2_to_u3_phase(haar_unitary_2(rng));
    return gate;
}

FermionBondGate random_parity_preserving_jw_string_gate(int q0, int q1, std::mt19937 &rng)
{
    FermionBondGate gate = random_parity_preserving_gate(q0, q1, rng);
    gate.kind = 2;
    return gate;
}

FermionBondGate fermionic_swap_gate(int q0, int q1)
{
    FermionBondGate gate;
    gate.kind = 1;
    gate.q0 = q0;
    gate.q1 = q1;
    return gate;
}

void fill_fermion_layer(FermionLayerData &layer,
                        int n,
                        bool odd_layer,
                        double p,
                        bool closed,
                        std::mt19937 &rng,
                        bool direct_boundary_gate = false)
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
        const std::size_t reserve_count = closed
            ? (direct_boundary_gate
                   ? static_cast<std::size_t>(1 + (n - 2) / 2)
                   : static_cast<std::size_t>((n - 2) + 1 + (n - 2) + (n - 2) / 2))
            : static_cast<std::size_t>((n - 2) / 2);
        layer.gates.reserve(reserve_count);
        if (closed)
        {
            if (direct_boundary_gate)
            {
                layer.gates.push_back(random_parity_preserving_jw_string_gate(0, n - 1, rng));
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

    for (int qubit = 0; qubit < n; ++qubit)
    {
        if (p <= 0.0)
        {
            layer.measure_flags[static_cast<std::size_t>(qubit)] = 0;
        }
        else if (p >= 1.0)
        {
            layer.measure_flags[static_cast<std::size_t>(qubit)] = 1;
        }
        else
        {
            layer.measure_flags[static_cast<std::size_t>(qubit)] =
                measure_dist(rng) ? 1 : 0;
        }
    }
}

FermionLayerData make_fermion_layer(int n,
                                    bool odd_layer,
                                    double p,
                                    bool closed,
                                    std::mt19937 &rng,
                                    bool direct_boundary_gate = false)
{
    FermionLayerData layer;
    fill_fermion_layer(layer, n, odd_layer, p, closed, rng, direct_boundary_gate);
    return layer;
}

void mipt_fermion_frontend_inplace(std::vector<FermionLayerData> &layers,
                                   int n,
                                   int periods,
                                   double p,
                                   bool closed,
                                   std::mt19937 &rng,
                                   bool direct_boundary_gate = false)
{
    if ((n % 2) != 0)
    {
        throw std::invalid_argument("Fermionic 1D simulation requires even n.");
    }
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("Measurement probability p must be in [0,1].");
    }

    std::bernoulli_distribution extra_even_layer(0.5);
    const bool add_extra_even_layer = extra_even_layer(rng);
    const std::size_t layer_count = static_cast<std::size_t>(2 * periods + (add_extra_even_layer ? 1 : 0));
    layers.resize(layer_count);

    std::size_t idx = 0;
    for (int period = 0; period < periods; ++period)
    {
        fill_fermion_layer(layers[idx++], n, false, p, closed, rng, direct_boundary_gate);
        fill_fermion_layer(layers[idx++], n, true, p, closed, rng, direct_boundary_gate);
    }

    // Match the Python fermionic circuit: a final even layer is included with
    // 50% probability, and its measurements still use probability p.
    if (add_extra_even_layer)
    {
        fill_fermion_layer(layers[idx++], n, false, p, closed, rng, direct_boundary_gate);
    }
}

std::vector<FermionLayerData> mipt_fermion_frontend(int n,
                                                    int periods,
                                                    double p,
                                                    bool closed = true)
{
    std::mt19937 rng(std::random_device{}());
    std::vector<FermionLayerData> layers;
    layers.reserve(static_cast<std::size_t>(2 * periods + 1));
    mipt_fermion_frontend_inplace(layers, n, periods, p, closed, rng, false);
    return layers;
}



struct MiptCircuitWorkStats
{
    std::size_t layers = 0;
    std::size_t measurements = 0;
    std::size_t logical_two_site_gates = 0;
    std::size_t mms_bonds = 0;
    std::size_t fermion_parity_gates = 0;
    std::size_t fermion_jw_boundary_gates = 0;
    std::size_t fermion_jw_cz_corrections = 0;
    std::size_t fswap_gates = 0;
    std::size_t frgs_bonds = 0;
    std::size_t wrapping_frgs_bonds = 0;
    std::size_t estimated_cudaq_two_qubit_ops = 0;
};

MiptCircuitWorkStats circuit_work_stats_mms(const std::vector<LayerData> &layers, int n, bool closed)
{
    MiptCircuitWorkStats stats;
    stats.layers = layers.size();
    for (const auto &layer : layers)
    {
        for (int flag : layer.measure_flags)
        {
            stats.measurements += (flag != 0) ? 1u : 0u;
        }
        const int start = layer.start;
        for (int i = start; i < n - 1; i += 2)
        {
            ++stats.mms_bonds;
        }
        if (closed && start == 1 && n > 2)
        {
            ++stats.mms_bonds;
        }
    }
    stats.logical_two_site_gates = stats.mms_bonds;
    // Each MMS bond is implemented as two CX gates plus single-qubit basis rotations.
    stats.estimated_cudaq_two_qubit_ops = 2u * stats.mms_bonds;
    return stats;
}

MiptCircuitWorkStats circuit_work_stats_fermion(const std::vector<FermionLayerData> &layers)
{
    MiptCircuitWorkStats stats;
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

MiptCircuitWorkStats circuit_work_stats_frgs(const std::vector<FRGSLayerData> &layers, int n, bool closed)
{
    MiptCircuitWorkStats stats;
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
            ++stats.frgs_bonds;
            const int j = (i + 1) % n;
            if (j < i)
            {
                ++stats.wrapping_frgs_bonds;
                stats.estimated_cudaq_two_qubit_ops += static_cast<std::size_t>(2 * std::max(0, i - j - 1) + 4);
            }
            else
            {
                stats.estimated_cudaq_two_qubit_ops += 4u;
            }
        }
    }
    stats.logical_two_site_gates = stats.frgs_bonds;
    return stats;
}

std::string circuit_work_stats_string(const char *label,
                                      const MiptCircuitWorkStats &stats,
                                      int n,
                                      int periods,
                                      double p)
{
    std::ostringstream out;
    out << label
        << " stats: n=" << n
        << " periods=" << periods
        << " p=" << p
        << " layers=" << stats.layers
        << " measurements=" << stats.measurements
        << " logical_two_site_gates=" << stats.logical_two_site_gates
        << " estimated_cudaq_two_qubit_ops=" << stats.estimated_cudaq_two_qubit_ops;
    if (stats.mms_bonds)
    {
        out << " mms_bonds=" << stats.mms_bonds;
    }
    if (stats.fermion_parity_gates || stats.fswap_gates)
    {
        out << " parity_gates=" << stats.fermion_parity_gates
            << " fswap_gates=" << stats.fswap_gates;
        if (stats.fermion_jw_boundary_gates)
        {
            out << " jw_boundary_gates=" << stats.fermion_jw_boundary_gates
                << " jw_cz_corrections=" << stats.fermion_jw_cz_corrections;
        }
    }
    if (stats.frgs_bonds)
    {
        out << " frgs_bonds=" << stats.frgs_bonds
            << " wrapping_frgs_bonds=" << stats.wrapping_frgs_bonds;
    }
    return out.str();
}

template <typename LayerVector>
void apply_debug_prefix_layer_limit(LayerVector &layers, const char *label)
{
    const long limit = mipt_backend::debug_prefix_layers();
    if (limit <= 0)
    {
        return;
    }
    const auto old_size = layers.size();
    if (old_size > static_cast<std::size_t>(limit))
    {
        layers.resize(static_cast<std::size_t>(limit));
    }
    std::cerr << "[mipt-debug] WARNING: MIPT_DEBUG_PREFIX_LAYERS=" << limit
              << " is set; " << label << " circuit was truncated from "
              << old_size << " to " << layers.size()
              << " layers for backend profiling only. Output is not a valid production MIPT sample.\n";
}


enum class CircuitType : int
{
    MMS = 0,
    Haar = 1,
    FermionRPPU = 2,
    RFGS = 3,
};

const char *circuit_type_name(CircuitType circ_type)
{
    switch (circ_type)
    {
    case CircuitType::MMS:
        return "MMS";
    case CircuitType::Haar:
        return "Random Haar Unitary";
    case CircuitType::FermionRPPU:
        return "Fermionic Random Parity Preserving Unitary";
    case CircuitType::RFGS:
        return "RFGS";
    }
    return "Unknown";
}

CircuitType parse_circuit_type(int value)
{
    if (value < 0 || value > 3)
    {
        throw std::invalid_argument("circ_type must be one of 0=MMS, 1=Random Haar Unitary, 2=Fermionic Random Parity Preserving Unitary, or 3=RFGS.");
    }
    return static_cast<CircuitType>(value);
}

bool uses_fermionic_trace(CircuitType circ_type)
{
    return circ_type == CircuitType::FermionRPPU || circ_type == CircuitType::RFGS;
}

using ComplexD = std::complex<double>;
using Haar4 = std::array<ComplexD, 16>;
using Matrix2 = std::array<ComplexD, 4>;

struct HaarBondGate
{
    int q0 = 0; // Local least-significant basis bit in the historical Haar path.
    int q1 = 0; // Local most-significant basis bit / multiplexor control.
    FermionU2Params a0;
    FermionU2Params a1;
    FermionU2Params b0;
    FermionU2Params b1;
    double theta0 = 0.0;
    double theta1 = 0.0;
};

struct HaarLayerData
{
    std::vector<HaarBondGate> gates;
    std::vector<int> measure_flags;
};

Haar4 haar_unitary_4(std::mt19937 &rng)
{
    // Draw U(4) from the Haar measure by QR/Gram-Schmidt orthonormalizing
    // four independent complex Gaussian columns. The matrix is row-major.
    std::normal_distribution<double> normal(0.0, 1.0);
    Haar4 q{};

    for (int col = 0; col < 4; ++col)
    {
        std::array<ComplexD, 4> v{};
        for (int row = 0; row < 4; ++row)
        {
            v[row] = ComplexD(normal(rng), normal(rng));
        }

        for (int prev = 0; prev < col; ++prev)
        {
            ComplexD dot = 0.0;
            for (int row = 0; row < 4; ++row)
            {
                dot += std::conj(q[static_cast<std::size_t>(row * 4 + prev)]) * v[row];
            }
            for (int row = 0; row < 4; ++row)
            {
                v[row] -= q[static_cast<std::size_t>(row * 4 + prev)] * dot;
            }
        }

        double norm2 = 0.0;
        for (const auto &x : v)
        {
            norm2 += std::norm(x);
        }
        const double norm = std::sqrt(norm2);
        if (norm < 1e-14)
        {
            --col;
            continue;
        }

        for (int row = 0; row < 4; ++row)
        {
            q[static_cast<std::size_t>(row * 4 + col)] = v[row] / norm;
        }
    }
    return q;
}

Matrix2 matrix2_adjoint(const Matrix2 &a)
{
    return {std::conj(a[0]), std::conj(a[2]),
            std::conj(a[1]), std::conj(a[3])};
}

Matrix2 matrix2_multiply(const Matrix2 &a, const Matrix2 &b)
{
    return {
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    };
}

Matrix2 block2(const Haar4 &u, int block_row, int block_col)
{
    Matrix2 out{};
    for (int r = 0; r < 2; ++r)
    {
        for (int c = 0; c < 2; ++c)
        {
            out[static_cast<std::size_t>(2 * r + c)] =
                u[static_cast<std::size_t>((2 * block_row + r) * 4 + (2 * block_col + c))];
        }
    }
    return out;
}

void normalize_vector2(ComplexD &x, ComplexD &y)
{
    const double n = std::sqrt(std::norm(x) + std::norm(y));
    if (!(n > 0.0))
    {
        throw std::runtime_error("Degenerate vector in Haar U(4) cosine-sine decomposition.");
    }
    x /= n;
    y /= n;
}

HaarBondGate decompose_haar_u4_csd(int q0, int q1, const Haar4 &u)
{
    // Cosine-sine decomposition in the historical local basis ordering
    // {00,01,10,11} = {q1 q0}. Thus q1 selects the 2x2 blocks and q0 is
    // the target of each multiplexed one-qubit operation:
    // U = (A0 (+) A1) [ C -S; S C ] (B0 (+) B1).
    const Matrix2 x = block2(u, 0, 0);
    const Matrix2 u01 = block2(u, 0, 1);
    const Matrix2 u10 = block2(u, 1, 0);

    const Matrix2 xdag = matrix2_adjoint(x);
    const Matrix2 h = matrix2_multiply(xdag, x);
    const double h00 = std::real(h[0]);
    const double h11 = std::real(h[3]);
    const ComplexD h01 = h[1];
    const double trace = h00 + h11;
    const double disc = std::sqrt(std::max(0.0,
        (h00 - h11) * (h00 - h11) + 4.0 * std::norm(h01)));
    const double lambda0 = std::clamp(0.5 * (trace + disc), 0.0, 1.0);
    const double lambda1 = std::clamp(0.5 * (trace - disc), 0.0, 1.0);

    ComplexD v00, v10, v01, v11;
    if (std::abs(h01) > 1e-14)
    {
        v00 = h01;
        v10 = lambda0 - h00;
        normalize_vector2(v00, v10);
        // Orthogonal complement fixes the second singular vector exactly.
        v01 = -std::conj(v10);
        v11 = std::conj(v00);
    }
    else if (h00 >= h11)
    {
        v00 = 1.0; v10 = 0.0;
        v01 = 0.0; v11 = 1.0;
    }
    else
    {
        v00 = 0.0; v10 = 1.0;
        v01 = 1.0; v11 = 0.0;
    }

    const Matrix2 v{v00, v01, v10, v11};
    const double c0 = std::sqrt(lambda0);
    const double c1 = std::sqrt(lambda1);
    const double s0 = std::sqrt(std::max(0.0, 1.0 - lambda0));
    const double s1 = std::sqrt(std::max(0.0, 1.0 - lambda1));
    constexpr double eps = 1e-12;
    if (c0 < eps || c1 < eps || s0 < eps || s1 < eps)
    {
        // This has probability zero for a Haar-random matrix. Regenerate rather
        // than introducing an unstable pseudo-inverse into the exact circuit.
        throw std::runtime_error("Numerically singular Haar U(4) cosine-sine decomposition.");
    }

    // A0 = X V diag(1/c), B0 = V^dagger.
    Matrix2 a0 = matrix2_multiply(x, v);
    a0[0] /= c0; a0[2] /= c0;
    a0[1] /= c1; a0[3] /= c1;
    const Matrix2 b0 = matrix2_adjoint(v);

    // A1 = U10 B0^dagger diag(1/s) = U10 V diag(1/s).
    Matrix2 a1 = matrix2_multiply(u10, v);
    a1[0] /= s0; a1[2] /= s0;
    a1[1] /= s1; a1[3] /= s1;

    // B1 = -diag(1/s) A0^dagger U01.
    Matrix2 b1 = matrix2_multiply(matrix2_adjoint(a0), u01);
    b1[0] = -b1[0] / s0; b1[1] = -b1[1] / s0;
    b1[2] = -b1[2] / s1; b1[3] = -b1[3] / s1;

    HaarBondGate gate;
    gate.q0 = q0;
    gate.q1 = q1;
    gate.a0 = decompose_u2_to_u3_phase(a0);
    gate.a1 = decompose_u2_to_u3_phase(a1);
    gate.b0 = decompose_u2_to_u3_phase(b0);
    gate.b1 = decompose_u2_to_u3_phase(b1);
    gate.theta0 = 2.0 * std::atan2(s0, c0);
    gate.theta1 = 2.0 * std::atan2(s1, c1);
    return gate;
}

HaarBondGate random_haar_bond_gate(int q0, int q1, std::mt19937 &rng)
{
    for (;;)
    {
        try
        {
            return decompose_haar_u4_csd(q0, q1, haar_unitary_4(rng));
        }
        catch (const std::runtime_error &)
        {
            // Regenerate only for the measure-zero numerically singular case.
        }
    }
}

void fill_haar_layer(HaarLayerData &layer,
                     int n,
                     int start,
                     double p,
                     bool closed,
                     std::mt19937 &rng)
{
    const int bond_stop = (closed && start == 1 && n > 2) ? n : (n - 1);
    const int bond_count = (bond_stop > start) ? ((bond_stop - start + 1) / 2) : 0;
    layer.gates.resize(static_cast<std::size_t>(bond_count));
    layer.measure_flags.resize(static_cast<std::size_t>(n));

    int b = 0;
    for (int i = start; i < bond_stop; i += 2)
    {
        const int j = (i + 1) % n;
        layer.gates[static_cast<std::size_t>(b++)] = random_haar_bond_gate(i, j, rng);
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
        std::bernoulli_distribution measure_dist(p);
        for (int q = 0; q < n; ++q)
        {
            layer.measure_flags[static_cast<std::size_t>(q)] = measure_dist(rng) ? 1 : 0;
        }
    }
}

void haar_frontend_inplace(std::vector<HaarLayerData> &layers,
                           int n,
                           int periods,
                           double p,
                           std::mt19937 &rng,
                           bool closed = true)
{
    if ((n % 2) != 0)
    {
        throw std::invalid_argument("Random Haar 1D simulation requires even n for periodic brickwork geometry.");
    }
    layers.resize(static_cast<std::size_t>(2 * periods));
    std::size_t idx = 0;
    for (int period = 0; period < periods; ++period)
    {
        fill_haar_layer(layers[idx++], n, 0, p, closed, rng);
        fill_haar_layer(layers[idx++], n, 1, p, closed, rng);
    }
}

struct MIPTHaarKernel_1D
{
    void operator()(int n, const std::vector<HaarLayerData> &layers) __qpu__
    {
        cudaq::qvector q(n);
        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            for (std::size_t gi = 0; gi < layers[layer].gates.size(); ++gi)
            {
                const auto gate = layers[layer].gates[gi];
                // Right block-diagonal factor B0 (+) B1.
                r1(gate.b1.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.b1.theta, gate.b1.phi, gate.b1.lambda,
                                q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);
                r1(gate.b0.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.b0.theta, gate.b0.phi, gate.b0.lambda,
                                q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);

                // Cosine-sine core: Ry(theta0) for control=0 and
                // Ry(theta1) for control=1.
                ry<cudaq::ctrl>(gate.theta1, q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);
                ry<cudaq::ctrl>(gate.theta0, q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);

                // Left block-diagonal factor A0 (+) A1.
                r1(gate.a1.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.a1.theta, gate.a1.phi, gate.a1.lambda,
                                q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);
                r1(gate.a0.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.a0.theta, gate.a0.phi, gate.a0.lambda,
                                q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);
            }

            for (int i = 0; i < n; ++i)
            {
                if (layers[layer].measure_flags[static_cast<std::size_t>(i)])
                {
                    mz(q[i]);
                }
            }
        }
    }
};

#include "mipt_native_mps.hpp"

void run_1d(int n, int periods, int realizations, int res, double p_min, double p_max, CircuitType circ_type)
{
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument(
            "Saving three-qubit reduced density matrices requires n >= 3.");
    }
    if ((circ_type == CircuitType::Haar ||
         circ_type == CircuitType::FermionRPPU ||
         circ_type == CircuitType::RFGS) &&
        ((n % 2) != 0))
    {
        throw std::invalid_argument("circ_type 1, 2, and 3 require even n in the periodic 1D brickwork geometry.");
    }

    const bool fermion_trace = uses_fermionic_trace(circ_type);
    const std::vector<Subsystem> subsystems = periodic_ring_three_site_subsystems(n);
#ifdef MIPT_ENABLE_CUDA_RHO
    const std::vector<int> flat_subsystems = flatten_subsystems(subsystems);
#endif
    std::vector<double> ps = linspace(p_min, p_max, res);
    std::ofstream infofile("info.csv");

    infofile << "n,periods,realizations,resolution,p_min,p_max,circ_type,circuit_name,fermion_trace,cudaq_backend\n";
    infofile << n << "," << periods << "," << realizations << "," << res << ","
             << p_min << "," << p_max << "," << static_cast<int>(circ_type) << ","
             << circuit_type_name(circ_type) << "," << (fermion_trace ? 1 : 0) << ","
             << mipt_backend::compiled_cudaq_backend() << "\n";

    mipt_io::DensityMatrixWriter rho3_file(
        "rho3.bin",
        density_metadata(1, n, periods, realizations, res,
                         p_min, p_max, n, 1, subsystems));

    std::mt19937 haar_rng(std::random_device{}());
    std::mt19937 mms_rng(std::random_device{}());
    std::mt19937 fermion_rng(std::random_device{}());
    std::mt19937 frgs_rng(std::random_device{}());
    std::mt19937 native_mps_rng(std::random_device{}());
    std::vector<LayerData> mms_layers;
    std::vector<HaarLayerData> haar_layers;
    std::vector<FermionLayerData> fermion_layers;
    std::vector<FRGSLayerData> frgs_layers;
    mms_layers.reserve(static_cast<std::size_t>(2 * periods + 1));
    haar_layers.reserve(static_cast<std::size_t>(2 * periods));
    fermion_layers.reserve(static_cast<std::size_t>(2 * periods + 1));
    frgs_layers.reserve(static_cast<std::size_t>(2 * periods + 1));
    std::vector<double> rho3_ri(subsystems.size() * RHO3_VALUES);

    for (std::size_t p_index = 0; p_index < ps.size(); ++p_index)
    {
        const double p = ps[p_index];
        std::cout << "Simulating p = " << p << "...\n " << std::flush;

        double sec_per_circ = 0.0;

        for (int r = 0; r < realizations; ++r)
        {
            auto circ_time_start = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point start, lap1, lap2, lap3, lap4;

            if (r == 0)
            {
                start = std::chrono::steady_clock::now();
            }

            std::fill(rho3_ri.begin(), rho3_ri.end(), 0.0);

            if (mipt_native_mps::enabled())
            {
                if (r == 0)
                {
                    lap1 = std::chrono::steady_clock::now();
                    mipt_native_mps::log("MIPT_NATIVE_MPS=1: using native CPU TEBD/MPS path; CUDA-Q get_state is bypassed for this trajectory.");
                }

                mipt_native_mps::simulate_1d_rho3(
                    n, periods, p, circ_type, subsystems, rho3_ri.data(), native_mps_rng);

                if (r == 0)
                {
                    lap2 = std::chrono::steady_clock::now();
                }
            }
            else
            {
                auto state = [&]() {
                    if (circ_type == CircuitType::Haar)
                    {
                        haar_frontend_inplace(haar_layers, n, periods, p, haar_rng, true);
                        apply_debug_prefix_layer_limit(haar_layers, "Haar");
                        if (r == 0)
                        {
                            lap1 = std::chrono::steady_clock::now();
                        }
                        if (mipt_backend::verbose_enabled())
                        {
                            mipt_backend::verbose_log("get_state start: Haar n=" + std::to_string(n) +
                                                      " layers=" + std::to_string(haar_layers.size()));
                        }
                        const auto t_get_state = std::chrono::steady_clock::now();
                        auto state = cudaq::get_state(MIPTHaarKernel_1D{}, n, haar_layers);
                        if (mipt_backend::verbose_enabled())
                        {
                            mipt_backend::verbose_log("get_state done: Haar elapsed_s=" +
                                                      std::to_string(mipt_backend::seconds_since(t_get_state)));
                        }
                        return state;
                    }

                    if (circ_type == CircuitType::FermionRPPU)
                    {
                        const bool direct_boundary = mipt_backend::direct_fermion_boundary_enabled();
                        mipt_fermion_frontend_inplace(fermion_layers, n, periods, p, true, fermion_rng, direct_boundary);
                        if (direct_boundary && r == 0 && mipt_backend::verbose_enabled())
                        {
                            mipt_backend::verbose_log("FermionRPPU direct JW-CZ boundary enabled; set MIPT_DIRECT_FERMION_BOUNDARY=0 to force historical FSWAP chains.");
                        }
                        apply_debug_prefix_layer_limit(fermion_layers, "FermionRPPU");
                        if (r == 0)
                        {
                            lap1 = std::chrono::steady_clock::now();
                        }
                        if (mipt_backend::verbose_enabled())
                        {
                            const auto stats = circuit_work_stats_fermion(fermion_layers);
                            mipt_backend::verbose_log(circuit_work_stats_string("FermionRPPU", stats, n, periods, p));
                            mipt_backend::verbose_log("get_state start: FermionRPPU n=" + std::to_string(n) + " layers=" + std::to_string(fermion_layers.size()));
                        }
                        const auto t_get_state = std::chrono::steady_clock::now();
                        auto state = cudaq::get_state(MIPTFermionKernel_1D{}, n, fermion_layers);
                        if (mipt_backend::verbose_enabled())
                        {
                            mipt_backend::verbose_log("get_state done: FermionRPPU elapsed_s=" + std::to_string(mipt_backend::seconds_since(t_get_state)));
                        }
                        return state;
                    }

                    if (circ_type == CircuitType::RFGS)
                    {
                        frgs_mipt_frontend_inplace(frgs_layers, n, periods, p, frgs_rng, true);
                        apply_debug_prefix_layer_limit(frgs_layers, "RFGS");
                        if (r == 0)
                        {
                            lap1 = std::chrono::steady_clock::now();
                        }
                        if (mipt_backend::verbose_enabled())
                        {
                            const auto stats = circuit_work_stats_frgs(frgs_layers, n, true);
                            mipt_backend::verbose_log(circuit_work_stats_string("RFGS", stats, n, periods, p));
                            mipt_backend::verbose_log("get_state start: RFGS n=" + std::to_string(n) + " layers=" + std::to_string(frgs_layers.size()));
                        }
                        const auto t_get_state = std::chrono::steady_clock::now();
                        auto state = cudaq::get_state(MIPTKernel_1D_FRGS{}, n, frgs_layers, true);
                        if (mipt_backend::verbose_enabled())
                        {
                            mipt_backend::verbose_log("get_state done: RFGS elapsed_s=" + std::to_string(mipt_backend::seconds_since(t_get_state)));
                        }
                        return state;
                    }

                    mipt_frontend_inplace(mms_layers, n, periods, p, mms_rng);
                    apply_debug_prefix_layer_limit(mms_layers, "MMS");
                    if (r == 0)
                    {
                        lap1 = std::chrono::steady_clock::now();
                    }
                    if (mipt_backend::verbose_enabled())
                    {
                        const auto stats = circuit_work_stats_mms(mms_layers, n, true);
                        mipt_backend::verbose_log(circuit_work_stats_string("MMS", stats, n, periods, p));
                        mipt_backend::verbose_log("get_state start: MMS n=" + std::to_string(n) + " layers=" + std::to_string(mms_layers.size()));
                    }
                    const auto t_get_state = std::chrono::steady_clock::now();
                    auto state = cudaq::get_state(MIPTKernel_1D{}, n, mms_layers, true);
                    if (mipt_backend::verbose_enabled())
                    {
                        mipt_backend::verbose_log("get_state done: MMS elapsed_s=" + std::to_string(mipt_backend::seconds_since(t_get_state)));
                    }
                    return state;
                }();

                if (r == 0)
                {
                    lap2 = std::chrono::steady_clock::now();
                }

                cudaq_three_site_density_matrices(
                    state, n, subsystems, rho3_ri.data(), fermion_trace);
            }

            if (r == 0)
            {
                lap3 = std::chrono::steady_clock::now();
            }

            rho3_file.write_record(
                static_cast<std::uint32_t>(p_index),
                static_cast<std::uint32_t>(r),
                p,
                rho3_ri.data(),
                rho3_ri.size());

            if (r == 0)
            {
                lap4 = std::chrono::steady_clock::now();

                const std::chrono::duration<double> t_layer = lap1 - start;
                const std::chrono::duration<double> t_qc = lap2 - lap1;
                const std::chrono::duration<double> t_dm = lap3 - lap2;
                const std::chrono::duration<double> t_save = lap4 - lap3;
                std::cout << "Layer, QC, DM, Save times: "
                          << t_layer.count() << "s, "
                          << t_qc.count() << "s, "
                          << t_dm.count() << "s, "
                          << t_save.count() << "s\n";
            }

            const auto circ_time_end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> circ_time_diff =
                circ_time_end - circ_time_start;
            if (r != 0)
            {
                sec_per_circ = ((r - 1) * sec_per_circ +
                                circ_time_diff.count()) / r;
            }
            else
            {
                sec_per_circ = circ_time_diff.count();
            }

            const double time_estimate = (realizations - r) * sec_per_circ;
            std::cout << "\rAverage circuits/second: "
                      << std::round((1 / sec_per_circ) * 100.0) / 100.0
                      << ", Realisations ETA: "
                      << static_cast<int>(time_estimate)
                      << "s               " << std::flush;
        }
        std::cout << "\n";
    }

    rho3_file.close();
    std::cout << "Saved " << subsystems.size()
              << " cyclic three-qubit reduced density matrices per trajectory to rho3.bin for circ_type "
              << static_cast<int>(circ_type) << " (" << circuit_type_name(circ_type) << ")"
              << (fermion_trace ? " using fermionic partial traces.\n" : ".\n");
}




void run_2d(int x, int y, int periods, int realizations, int res, double p_min, double p_max)
{
    const int n = x * y;
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument(
            "Saving three-qubit reduced density matrices requires x*y >= 3.");
    }

    // The requested distance-one window convention is defined for the 1D ring.
    // Preserve the prior 2D output behaviour as one terminal three-qubit RDM.
    const std::vector<Subsystem> subsystems = three_site_subsystems_2d(x, y);
    std::vector<double> ps = linspace(p_min, p_max, res);
    std::ofstream infofile("info2.csv");

    // Keep the existing info2.csv format unchanged.
    infofile << "n,periods,realizations,resolution,p_min,p_max\n";
    infofile << n << "," << periods << "," << realizations << "," << res << ","
             << p_min << "," << p_max << "\n";

    mipt_io::DensityMatrixWriter rho3_file(
        "rho3_2d.bin",
        density_metadata(2, n, periods, realizations, res,
                         p_min, p_max, x, y, subsystems));

    std::mt19937 mms_rng(std::random_device{}());
    std::vector<LayerData> mms_layers;
    mms_layers.reserve(static_cast<std::size_t>(2 * periods + 1));
    std::vector<double> rho3_ri(subsystems.size() * RHO3_VALUES);

    for (std::size_t p_index = 0; p_index < ps.size(); ++p_index)
    {
        const double p = ps[p_index];
        std::cout << "Simulating p = " << p << "...\n " << std::flush;

        double sec_per_circ = 0.0;

        for (int r = 0; r < realizations; ++r)
        {
            auto circ_time_start = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point start, lap1, lap2, lap3, lap4;

            if (r == 0)
            {
                start = std::chrono::steady_clock::now();
            }

            mipt_frontend_inplace(mms_layers, n, periods, p, mms_rng);

            if (r == 0)
            {
                lap1 = std::chrono::steady_clock::now();
            }

            if (mipt_backend::verbose_enabled())
            {
                mipt_backend::verbose_log("get_state start: MMS2D n=" + std::to_string(n) + " layers=" + std::to_string(mms_layers.size()));
            }
            const auto t_get_state = std::chrono::steady_clock::now();
            auto state = cudaq::get_state(MIPTKernel_2D{}, x, y, mms_layers, true);
            if (mipt_backend::verbose_enabled())
            {
                mipt_backend::verbose_log("get_state done: MMS2D elapsed_s=" + std::to_string(mipt_backend::seconds_since(t_get_state)));
            }

            if (r == 0)
            {
                lap2 = std::chrono::steady_clock::now();
            }

            std::fill(rho3_ri.begin(), rho3_ri.end(), 0.0);
            cudaq_three_site_density_matrices(
                state, n, subsystems, rho3_ri.data());

            if (r == 0)
            {
                lap3 = std::chrono::steady_clock::now();
            }

            rho3_file.write_record(
                static_cast<std::uint32_t>(p_index),
                static_cast<std::uint32_t>(r),
                p,
                rho3_ri.data(),
                rho3_ri.size());

            if (r == 0)
            {
                lap4 = std::chrono::steady_clock::now();

                const std::chrono::duration<double> t_layer = lap1 - start;
                const std::chrono::duration<double> t_qc = lap2 - lap1;
                const std::chrono::duration<double> t_dm = lap3 - lap2;
                const std::chrono::duration<double> t_save = lap4 - lap3;
                std::cout << "Layer, QC, DM, Save times: "
                          << t_layer.count() << "s, "
                          << t_qc.count() << "s, "
                          << t_dm.count() << "s, "
                          << t_save.count() << "s\n";
            }

            const auto circ_time_end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> circ_time_diff =
                circ_time_end - circ_time_start;
            if (r != 0)
            {
                sec_per_circ = ((r - 1) * sec_per_circ +
                                circ_time_diff.count()) / r;
            }
            else
            {
                sec_per_circ = circ_time_diff.count();
            }

            const double time_estimate = (realizations - r) * sec_per_circ;
            std::cout << "\rAverage circuits/second: "
                      << std::round((1 / sec_per_circ) * 100.0) / 100.0
                      << ", Realisations ETA: "
                      << static_cast<int>(time_estimate)
                      << "s               " << std::flush;
        }
        std::cout << "\n";
    }

    rho3_file.close();
    std::cout << "Saved reduced density matrices to rho3_2d.bin.\n";
}


int main(int argc, char *argv[])
{
    mipt_backend::print_backend_banner_once("mipt.exe");
    if (argc > 1 &&
        (std::strcmp(argv[1], "--help") == 0 ||
         std::strcmp(argv[1], "-h") == 0))
    {
        std::cout << "Usage: " << argv[0]
                  << " [d = 1] [n = 10] [periods = 10] [realizations = 10] [resolution = 5] [p_min = 0.0] [p_max = 1.0] [circ_type = 0]\n";
        std::cout << "Description:\n";
        std::cout << "  d = dimensionality (1 or 2)\n";
        std::cout << "  n = number of qubits per dimension; for d=2 this means an n x n lattice\n";
        std::cout << "  periods = number of brickwork periods (1D period = even layer + odd layer)\n";
        std::cout << "  realizations = number of realizations per p value\n";
        std::cout << "  resolution = number of p grid points\n";
        std::cout << "  p_min = minimum measurement rate\n";
        std::cout << "  p_max = maximum measurement rate\n";
        std::cout << "  circ_type:\n";
        std::cout << "    0 = MMS gate set\n";
        std::cout << "    1 = Random Haar Unitary brickwork circuit\n";
        std::cout << "    2 = Fermionic Random Parity Preserving Unitary\n";
        std::cout << "    3 = RFGS from mipt_fermion.cpp\n";
        std::cout << "  For d=2, only circ_type=0 is supported.\n";
        return 0;
    }

    if (argc > 9)
    {
        std::cerr << "Too many arguments. Use --help for usage.\n";
        return 2;
    }

    int d = (argc > 1) ? std::stoi(argv[1]) : 1;
    int n = (argc > 2) ? std::stoi(argv[2]) : ((d == 1) ? 10 : 4);
    int periods = (argc > 3) ? std::stoi(argv[3]) : ((d == 1) ? 10 : 32);
    int realizations = (argc > 4) ? std::stoi(argv[4]) : 10;
    int res = (argc > 5) ? std::stoi(argv[5]) : 5;
    double p_min = (argc > 6) ? std::stod(argv[6]) : 0.0;
    double p_max = (argc > 7) ? std::stod(argv[7]) : 1.0;
    CircuitType circ_type = parse_circuit_type((argc > 8) ? std::stoi(argv[8]) : 0);

    if (d != 1 && d != 2)
    {
        throw std::invalid_argument("d must be 1 or 2.");
    }
    if (res <= 0)
    {
        throw std::invalid_argument("resolution must be positive.");
    }
    if (realizations <= 0)
    {
        throw std::invalid_argument("realizations must be positive.");
    }
    if (periods < 0)
    {
        throw std::invalid_argument("periods must be non-negative.");
    }
    if (p_min < 0.0 || p_max > 1.0 || p_min > p_max)
    {
        throw std::invalid_argument("Require 0 <= p_min <= p_max <= 1.");
    }

    auto start = std::chrono::steady_clock::now();

    if (d == 1)
    {
        std::cout << "Running 1D " << n << "-qubit circ_type "
                  << static_cast<int>(circ_type) << " (" << circuit_type_name(circ_type)
                  << ") simulation of " << realizations * res << " circuits.\n";
        run_1d(n, periods, realizations, res, p_min, p_max, circ_type);
    }
    else
    {
        if (circ_type != CircuitType::MMS)
        {
            throw std::invalid_argument("For d=2, only circ_type=0 (MMS gate set) is supported.");
        }
        std::cout << "Running 2D MMS " << n << "x" << n << " = " << n * n
                  << "-qubit simulation of " << realizations * res << " circuits.\n";
        run_2d(n, n, periods, realizations, res, p_min, p_max);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    return 0;
}
