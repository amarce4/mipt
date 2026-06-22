#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include "density_io.hpp"
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

        if (tensor.get_rank() != 1 || tensor.get_num_elements() < dim)
        {
            throw std::runtime_error("Expected a contiguous rank-1 CUDA-Q statevector tensor.");
        }

#ifdef MIPT_ENABLE_CUDA_RHO
        if (state.is_on_gpu() && !fermion_trace)
        {
            const std::vector<int> flat = flatten_subsystems(subsystems);
            int status = 0;
            if (precision == cudaq::SimulationState::precision::fp64)
            {
                status = mipt_cuda_rho3_subsystems_complex_f64(
                    tensor.data, n, flat.data(),
                    static_cast<int>(subsystems.size()), rho_ri);
            }
            else
            {
                status = mipt_cuda_rho3_subsystems_complex_f32(
                    tensor.data, n, flat.data(),
                    static_cast<int>(subsystems.size()), rho_ri);
            }
            if (status != 0)
            {
                throw std::runtime_error(
                    "CUDA three-qubit reduced-density-matrix kernel failed; status=" +
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

    constexpr std::uint32_t TMI_FILE_VERSION = 1;
    constexpr std::uint32_t TMI_ENDIAN_MARKER = 0x01020304u;
    constexpr char TMI_FILE_MAGIC[8] = {'M', 'I', 'P', 'T', 'T', 'M', 'I', '1'};

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

    std::vector<TmiMatrixTerm> tmi_terms_1d(int n)
    {
        if (n < 4)
        {
            throw std::invalid_argument("TMI output requires N=L >= 4.");
        }
        if ((n % 4) != 0)
        {
            throw std::invalid_argument("TMI output requires N=L to be divisible by 4.");
        }

        const int block = n / 4;
        const std::vector<int> A = mode_range(0 * block, block);
        const std::vector<int> B = mode_range(1 * block, block);
        const std::vector<int> C = mode_range(2 * block, block);
        const std::vector<int> D = mode_range(3 * block, block);

        // Four stored matrices are enough for the requested pure-state identity:
        // AB, AC, and BC provide the three pair entropies and can be traced down
        // to obtain A, B, and C. D is stored directly for S(D)=S(ABC).
        return {
            {"AB", concatenate_modes(A, B)},
            {"AC", concatenate_modes(A, C)},
            {"BC", concatenate_modes(B, C)},
            {"D", D},
        };
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

    void cudaq_tmi_density_matrices(
        cudaq::state &state,
        int n,
        const std::vector<TmiMatrixTerm> &terms,
        std::vector<double> &payload,
        bool fermion_trace = false)
    {
        if (terms.size() != 4)
        {
            throw std::invalid_argument("TMI output expects exactly four stored matrices: AB, AC, BC, D.");
        }

        const std::uint64_t dim_u64 = checked_pow2_u64(n);
        if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::invalid_argument("State-vector dimension does not fit in size_t.");
        }
        const std::size_t dim = static_cast<std::size_t>(dim_u64);
        const auto precision = state.get_precision();
        const auto tensor = state.get_tensor();

        if (tensor.get_rank() != 1 || tensor.get_num_elements() < dim)
        {
            throw std::runtime_error("Expected a contiguous rank-1 CUDA-Q statevector tensor.");
        }

        if (state.is_on_gpu())
        {
            if (precision == cudaq::SimulationState::precision::fp64)
            {
                std::vector<std::complex<double>> host_state(dim);
                state.to_host(host_state.data(), host_state.size());
                tmi_density_matrices_from_host_statevector(host_state.data(), n, terms, payload, fermion_trace);
            }
            else
            {
                std::vector<std::complex<float>> host_state(dim);
                state.to_host(host_state.data(), host_state.size());
                tmi_density_matrices_from_host_statevector(host_state.data(), n, terms, payload, fermion_trace);
            }
            return;
        }

        if (tensor.data == nullptr)
        {
            throw std::runtime_error("CUDA-Q state tensor has a null data pointer.");
        }

        if (precision == cudaq::SimulationState::precision::fp64)
        {
            tmi_density_matrices_from_host_statevector(
                reinterpret_cast<const std::complex<double> *>(tensor.data), n, terms, payload, fermion_trace);
        }
        else
        {
            tmi_density_matrices_from_host_statevector(
                reinterpret_cast<const std::complex<float> *>(tensor.data), n, terms, payload, fermion_trace);
        }
    }

    class TmiMatrixWriter
    {
      public:
        TmiMatrixWriter(const std::string &path,
                        int dimension,
                        int total_qubits,
                        int periods,
                        int realizations,
                        int res,
                        double p_min,
                        double p_max,
                        int grid_x,
                        int grid_y,
                        int block_qubits,
                        std::vector<TmiMatrixTerm> terms)
            : terms_(std::move(terms)),
              expected_values_(tmi_payload_value_count(terms_)),
              record_count_(static_cast<std::uint64_t>(realizations) *
                            static_cast<std::uint64_t>(res)),
              stream_(path, std::ios::binary | std::ios::trunc)
        {
            if (!stream_)
            {
                throw std::runtime_error("Could not create TMI density-matrix file: " + path);
            }
            if (terms_.size() != 4)
            {
                throw std::invalid_argument("TMI density-matrix file requires exactly four terms.");
            }

            stream_.write(TMI_FILE_MAGIC, sizeof(TMI_FILE_MAGIC));
            mipt_io::write_scalar(stream_, TMI_ENDIAN_MARKER);
            mipt_io::write_scalar(stream_, TMI_FILE_VERSION);
            mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(dimension));
            mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(total_qubits));
            mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(block_qubits));
            mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(periods));
            mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(realizations));
            mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(res));
            mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(grid_x));
            mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(grid_y));
            mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(terms_.size()));
            mipt_io::write_scalar(stream_, record_count_);
            mipt_io::write_scalar(stream_, p_min);
            mipt_io::write_scalar(stream_, p_max);

            for (const auto &term : terms_)
            {
                char name[8]{};
                const std::size_t copy_count = std::min<std::size_t>(term.name.size(), sizeof(name));
                std::memcpy(name, term.name.data(), copy_count);
                stream_.write(name, sizeof(name));
                mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(term.modes.size()));
                mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(term_matrix_dim_u64(term)));
                for (int q : term.modes)
                {
                    mipt_io::write_scalar(stream_, static_cast<std::uint32_t>(q));
                }
            }
            if (!stream_)
            {
                throw std::runtime_error("Failed while writing TMI metadata.");
            }
        }

        void write_record(std::uint32_t p_index,
                          std::uint32_t realization,
                          double p,
                          const std::vector<double> &payload)
        {
            if (payload.size() != expected_values_)
            {
                throw std::invalid_argument("Incorrect TMI density-matrix record size.");
            }
            if (records_written_ >= record_count_)
            {
                throw std::runtime_error("Attempted to write too many TMI records.");
            }

            mipt_io::write_scalar(stream_, p_index);
            mipt_io::write_scalar(stream_, realization);
            mipt_io::write_scalar(stream_, p);
            stream_.write(reinterpret_cast<const char *>(payload.data()),
                          static_cast<std::streamsize>(payload.size() * sizeof(double)));
            if (!stream_)
            {
                throw std::runtime_error("Failed while writing TMI record payload.");
            }
            ++records_written_;
        }

        void close()
        {
            if (closed_)
            {
                return;
            }
            if (records_written_ != record_count_)
            {
                throw std::runtime_error("TMI file closed before all records were written.");
            }
            stream_.flush();
            if (!stream_)
            {
                throw std::runtime_error("Failed while flushing TMI density-matrix file.");
            }
            stream_.close();
            closed_ = true;
        }

        ~TmiMatrixWriter()
        {
            if (!closed_)
            {
                stream_.close();
            }
        }

      private:
        std::vector<TmiMatrixTerm> terms_;
        std::size_t expected_values_ = 0;
        std::uint64_t record_count_ = 0;
        std::uint64_t records_written_ = 0;
        bool closed_ = false;
        std::ofstream stream_;
    };

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
    // kind = 0: parity-preserving two-mode unitary.
    // kind = 1: adjacent fermionic swap.
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
                }
                else
                {
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

struct FRGSLayerData //LayerData for fermions with a reduced gate set
{
    int start = 0;

    std::vector<int> measure_flags;

    // This flag can be 1 simultaneously with the others, but the others should not overlap with each other.
    std::vector<int> rot_z_flags;
    // Exactly one of these should be 1 for each qubit. 
    // Because these are two-layer gates, only the first qubit of each gate (last qubit if it's the wrapping gate) will determine if
    // we apply the gate or not.
    std::vector<int> rot_xx_flags;
    std::vector<int> rot_zz_flags;
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
            int start = flayers[layer].start;

            for (int i = start; i < n - 1; i += 2)
            {
                int j = (i+1) % n;
                // Local rotation on q[i]
                if (flayers[layer].rot_z_flags[i])
                {
                    rz(0.392699081698724154808, q[i]); //pi/8 Z rotation, i.e. T gate
                }
                if (flayers[layer].rot_z_flags[j])
                {
                    rz(0.392699081698724154808, q[j]);
                }
                // Local XX or ZZ rotation
                if (flayers[layer].rot_xx_flags[i]) //pi/4 XX rotation (i.e. conjugate of Molmer-Sorensen)
                {
                    if(j < i){// If we're wrapping, we need to apply a Jordan-Wigner string of conditional phase gates
                        // If q[i] occupancy changes, need to apply a Z to all sites between j and i. We can implement this by
                        // applying a CZ between q[i] and each site between j and i, and then doing it again after the XX rotation.  
                        // If the occupancy of q[i] changes, we've effectively applied the Z gates, and if it doesn't change, these gates cancel out.
                        for(int k = j+1; k < i; k++){
                            cz(3.14159265358979323846, q[k],q[i]);
                        }
                        // Also, reversing the order of the Majoranas change it from an XX rotation to a YY rotation, so we apply S gates to
                        // convert an XX to a YY since S X S^\dag = Y. 
                        rz(-1.57079632679489661923, q[i]);
                        rz(-1.57079632679489661923, q[j]);
                    }
                    // This is probably the most efficient way to implement a XX rotation using basic CUDA-Q gates?
                    //h(q[i]);h(q[j]);
                    cx(q[i], q[j]);
                    rx(0.78539816339744830962, q[i]);
                    cx(q[i], q[j]);
                    //h(q[i]);h(q[j]);
                    if(j < i){
                        s(q[i]);
                        s(q[j]);
                        for(int k = j+1; k < i; k++){
                            cz(3.14159265358979323846, q[k],q[i]);
                        }
                    }
                }
                if (flayers[layer].rot_zz_flags[i]) //pi/4 ZZ rotation (the only four-Majorana interaction)
                {
                    cx(q[i], q[j]);
                    rz(0.78539816339744830962, q[j]);
                    cx(q[i], q[j]);
                }
            }

            for (int i = 0; i < n; ++i)
            {
                if (flayers[layer].measure_flags[i])
                {
                    mz(q[i]); // Z measurement preserves fermion parity so this is still all right
                }
            }
        }
    }
};

LayerData make_mipt_layer(int n,
                          int start,
                          double p,
                          std::mt19937 &rng)
{
    std::uniform_int_distribution<int> axis_dist(0, 2);
    std::bernoulli_distribution measure_dist(p);

    LayerData layer;
    layer.start = start;

    layer.measure_flags.resize(n);

    layer.rot_x_flags.resize(n);
    layer.rot_y_flags.resize(n);
    layer.rot_xy_flags.resize(n);

    for (int q = 0; q < n; ++q)
    {
        int axis = axis_dist(rng);

        layer.rot_x_flags[q] = 0;
        layer.rot_y_flags[q] = 0;
        layer.rot_xy_flags[q] = 0;

        if (axis == 0)
        {
            layer.rot_x_flags[q] = 1;
        }
        else if (axis == 1)
        {
            layer.rot_y_flags[q] = 1;
        }
        else
        {
            layer.rot_xy_flags[q] = 1;
        }

        layer.measure_flags[q] = measure_dist(rng) ? 1 : 0;
    }

    return layer;
}

std::vector<LayerData> mipt_frontend(int n,
                                     int periods,
                                     double p)
{
    std::mt19937 rng(std::random_device{}());

    std::vector<LayerData> layers;
    layers.reserve(2 * periods + 1);

    for (int period = 0; period < periods; ++period)
    {
        layers.push_back(make_mipt_layer(n, 0, p, rng)); // even layer
        layers.push_back(make_mipt_layer(n, 1, p, rng)); // odd layer
    }

    // MIPT-style final even layer with 50% measurement probability.
    layers.push_back(make_mipt_layer(n, 0, 0.5, rng));

    return layers;
}

FRGSLayerData make_frgs_mipt_layer(int n,
                          int start,
                          double p,
                          std::mt19937 &rng)
{
    std::bernoulli_distribution coin_flip(0.5);
    std::bernoulli_distribution measure_dist(p);

    FRGSLayerData layer;
    layer.start = start;

    layer.measure_flags.resize(n);

    layer.rot_z_flags.resize(n);
    layer.rot_xx_flags.resize(n);
    layer.rot_zz_flags.resize(n);

    for (int q = 0; q < n; ++q)
    {
        layer.rot_z_flags[q] = coin_flip(rng) ? 1 : 0; // Single-site Z rotation on each site with a 50/50 chance;
        layer.rot_xx_flags[q] = coin_flip(rng) ? 1 : 0; // Apply an XX rotation with a 50/50 chance;
        layer.rot_zz_flags[q] = 1-layer.rot_xx_flags[q]; // If we're not applying an XX rotation, apply a ZZ rotation instead.

        layer.measure_flags[q] = measure_dist(rng) ? 1 : 0;
    }

    return layer;
}

std::vector<FRGSLayerData> frgs_mipt_frontend(int n,
                                     int periods,
                                     double p)
{
    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution extra_even_layer(0.5);

    std::vector<FRGSLayerData> layers;
    layers.reserve(2 * periods + 1);

    for (int period = 0; period < periods; ++period)
    {
        layers.push_back(make_frgs_mipt_layer(n, 0, p, rng)); // even layer
        layers.push_back(make_frgs_mipt_layer(n, 1, p, rng)); // odd layer
    }

    // Implement final even layer with 50% probability.
    if(extra_even_layer(rng)){
        layers.push_back(make_frgs_mipt_layer(n, 0, p, rng));
    }

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

FermionBondGate fermionic_swap_gate(int q0, int q1)
{
    FermionBondGate gate;
    gate.kind = 1;
    gate.q0 = q0;
    gate.q1 = q1;
    return gate;
}

FermionLayerData make_fermion_layer(int n,
                                    bool odd_layer,
                                    double p,
                                    bool closed,
                                    std::mt19937 &rng)
{
    std::bernoulli_distribution measure_dist(p);

    FermionLayerData layer;
    layer.measure_flags.resize(static_cast<std::size_t>(n));

    if (!odd_layer)
    {
        for (int qubit = 0; qubit < n; qubit += 2)
        {
            layer.gates.push_back(random_parity_preserving_gate(qubit, qubit + 1, rng));
        }
    }
    else
    {
        if (closed)
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

        for (int qubit = 1; qubit < n - 1; qubit += 2)
        {
            layer.gates.push_back(random_parity_preserving_gate(qubit, qubit + 1, rng));
        }
    }

    for (int qubit = 0; qubit < n; ++qubit)
    {
        layer.measure_flags[static_cast<std::size_t>(qubit)] =
            measure_dist(rng) ? 1 : 0;
    }

    return layer;
}

std::vector<FermionLayerData> mipt_fermion_frontend(int n,
                                                    int periods,
                                                    double p,
                                                    bool closed = true)
{
    if ((n % 2) != 0)
    {
        throw std::invalid_argument("Fermionic 1D simulation requires even n.");
    }
    if (p < 0.0 || p > 1.0)
    {
        throw std::invalid_argument("Measurement probability p must be in [0,1].");
    }

    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution extra_even_layer(0.5);

    std::vector<FermionLayerData> layers;
    layers.reserve(static_cast<std::size_t>(2 * periods + 1));

    for (int period = 0; period < periods; ++period)
    {
        layers.push_back(make_fermion_layer(n, false, p, closed, rng));
        layers.push_back(make_fermion_layer(n, true, p, closed, rng));
    }

    // Match the Python fermionic circuit: a final even layer is included with
    // 50% probability, and its measurements still use probability p.
    if (extra_even_layer(rng))
    {
        layers.push_back(make_fermion_layer(n, false, p, closed, rng));
    }

    return layers;
}



void run_1d(int n, int periods, int realizations, int res, double p_min, double p_max, bool fermion, bool rgs=false)
{
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument(
            "Saving three-qubit reduced density matrices requires n >= 3.");
    }

    const std::vector<Subsystem> subsystems = periodic_ring_three_site_subsystems(n);
    std::vector<double> ps = linspace(p_min, p_max, res);
    std::ofstream infofile("info.csv");

    infofile << "n,periods,realizations,resolution,p_min,p_max,fermion,rgs\n";
    infofile << n << "," << periods << "," << realizations << "," << res << ","
             << p_min << "," << p_max << "," << (fermion ? 1 : 0) << "," << (rgs ? 1 : 0) << "\n";

    mipt_io::DensityMatrixWriter rho3_file(
        "rho3.bin",
        density_metadata(1, n, periods, realizations, res,
                         p_min, p_max, n, 1, subsystems));

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

            auto state = [&]() {
                if (fermion)
                {
                    if(rgs){
                        auto layers = frgs_mipt_frontend(n, periods, p);
                        if (r == 0)
                        {
                            lap1 = std::chrono::steady_clock::now();
                        }
                        return cudaq::get_state(MIPTKernel_1D_FRGS{}, n, layers, true);
                    }
                    auto layers = mipt_fermion_frontend(n, periods, p, true);
                    if (r == 0)
                    {
                        lap1 = std::chrono::steady_clock::now();
                    }
                    return cudaq::get_state(MIPTFermionKernel_1D{}, n, layers);
                }

                auto layers = mipt_frontend(n, periods, p);
                if (r == 0)
                {
                    lap1 = std::chrono::steady_clock::now();
                }
                return cudaq::get_state(MIPTKernel_1D{}, n, layers, true);
            }();

            if (r == 0)
            {
                lap2 = std::chrono::steady_clock::now();
            }

            std::vector<double> rho3_ri(subsystems.size() * RHO3_VALUES);
            cudaq_three_site_density_matrices(
                state, n, subsystems, rho3_ri.data(), fermion);

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
              << " cyclic three-qubit reduced density matrices per trajectory to rho3.bin"
              << (fermion ? " using fermionic partial traces.\n" : ".\n");
}


void run_1d_tmi(int n, int periods, int realizations, int res, double p_min, double p_max, bool fermion, bool rgs=false)
{
    const int block_qubits = n / 4;
    const std::vector<TmiMatrixTerm> terms = tmi_terms_1d(n);
    std::vector<double> ps = linspace(p_min, p_max, res);
    std::ofstream infofile("info_tmi.csv");

    infofile << "n,periods,realizations,resolution,p_min,p_max,fermion,rgs,tmi_block_qubits,terms\n";
    infofile << n << "," << periods << "," << realizations << "," << res << ","
             << p_min << "," << p_max << "," << (fermion ? 1 : 0) << ","
             << (rgs ? 1 : 0) << "," << block_qubits << ",AB;AC;BC;D\n";

    TmiMatrixWriter tmi_file(
        "rho_tmi.bin",
        1,
        n,
        periods,
        realizations,
        res,
        p_min,
        p_max,
        n,
        1,
        block_qubits,
        terms);

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

            auto state = [&]() {
                if (fermion)
                {
                    if(rgs){
                        auto layers = frgs_mipt_frontend(n, periods, p);
                        if (r == 0)
                        {
                            lap1 = std::chrono::steady_clock::now();
                        }
                        return cudaq::get_state(MIPTKernel_1D_FRGS{}, n, layers, true);
                    }
                    auto layers = mipt_fermion_frontend(n, periods, p, true);
                    if (r == 0)
                    {
                        lap1 = std::chrono::steady_clock::now();
                    }
                    return cudaq::get_state(MIPTFermionKernel_1D{}, n, layers);
                }

                auto layers = mipt_frontend(n, periods, p);
                if (r == 0)
                {
                    lap1 = std::chrono::steady_clock::now();
                }
                return cudaq::get_state(MIPTKernel_1D{}, n, layers, true);
            }();

            if (r == 0)
            {
                lap2 = std::chrono::steady_clock::now();
            }

            std::vector<double> payload;
            cudaq_tmi_density_matrices(state, n, terms, payload, fermion);

            if (r == 0)
            {
                lap3 = std::chrono::steady_clock::now();
            }

            tmi_file.write_record(
                static_cast<std::uint32_t>(p_index),
                static_cast<std::uint32_t>(r),
                p,
                payload);

            if (r == 0)
            {
                lap4 = std::chrono::steady_clock::now();

                const std::chrono::duration<double> t_layer = lap1 - start;
                const std::chrono::duration<double> t_qc = lap2 - lap1;
                const std::chrono::duration<double> t_dm = lap3 - lap2;
                const std::chrono::duration<double> t_save = lap4 - lap3;
                std::cout << "Layer, QC, TMI-DM, Save times: "
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

    tmi_file.close();
    std::cout << "Saved TMI matrices per trajectory to rho_tmi.bin. "
              << "Stored terms: AB, AC, BC, D; block size="
              << block_qubits << " modes"
              << (fermion ? " using fermionic partial traces.\n" : ".\n");
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

            auto layers = mipt_frontend(n, periods, p);

            if (r == 0)
            {
                lap1 = std::chrono::steady_clock::now();
            }

            auto state = cudaq::get_state(MIPTKernel_2D{}, x, y, layers, true);

            if (r == 0)
            {
                lap2 = std::chrono::steady_clock::now();
            }

            std::vector<double> rho3_ri(subsystems.size() * RHO3_VALUES);
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
    if (argc > 1 &&
        (std::strcmp(argv[1], "--help") == 0 ||
         std::strcmp(argv[1], "-h") == 0))
    {
        std::cout << "Usage: " << argv[0]
                  << " [d = 1] [n = 10] [periods = 10] [realizations = 10] [resolution = 5] [p_min = 0.0] [p_max = 1.0] [fermion = 0] [tmi = 0]\n";
        std::cout << "Description:\n";
        std::cout << "  d = dimensionality (1 or 2)\n";
        std::cout << "  n = number of qubits per dimension\n";
        std::cout << "  periods = number of periods (suggested 1D: n; 2D: 2*n^2)\n";
        std::cout << "  realizations = number of realizations (simulations per point)\n";
        std::cout << "  resolution = number of points\n";
        std::cout << "  p_min = minimum measurement rate\n";
        std::cout << "  p_max = maximum measurement rate\n";
        std::cout << "  fermion = 0 uses the existing MMS/qubit circuit; fermion = 1 uses parity-preserving fermionic gates and JW FSWAP boundaries\n";
        std::cout << "  tmi = 0 writes the existing rho3.bin format; tmi = 1 writes rho_tmi.bin for four-block TMI\n";
        return 0;
    }

    if (argc > 10)
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
    int fermion = (argc > 8) ? std::stoi(argv[8]) : 0;
    int tmi_output = (argc > 9) ? std::stoi(argv[9]) : 0;
    if (fermion != 0 && fermion != 1)
    {
        throw std::invalid_argument("The optional fermion argument must be 0 or 1.");
    }
    if (tmi_output != 0 && tmi_output != 1)
    {
        throw std::invalid_argument("The optional tmi argument must be 0 or 1.");
    }

    auto start = std::chrono::steady_clock::now(); // Get start time

    if (d == 1)
    {
        std::cout << "Running 1D " << n << "-qubit "
                  << (fermion ? "fermionic" : "MMS/qubit")
                  << " simulation of " << realizations*res << " circuits.\n";
        if (tmi_output == 1)
        {
            run_1d_tmi(n, periods, realizations, res, p_min, p_max, fermion != 0);
        }
        else
        {
            run_1d(n, periods, realizations, res, p_min, p_max, fermion != 0);
        }
    }
    else if (d == 2)
    {
        if (fermion == 1)
        {
            throw std::invalid_argument("Fermionic circuit mode is currently defined only for 1D circuits.");
        }
        if (tmi_output == 1)
        {
            throw std::invalid_argument("TMI output mode is currently defined only for 1D circuits.");
        }
        std::cout << "Running 2D " << n << "x" << n << " = " << n*n << "-qubit simulation of " << realizations*res << " circuits.\n";
        run_2d(n, n, periods, realizations, res, p_min, p_max);
    }

    auto end = std::chrono::steady_clock::now(); // Get end time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    return 0;
}