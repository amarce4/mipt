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
        double *rho_ri)
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
        }
    }

    void cudaq_three_site_density_matrices(
        cudaq::state &state,
        int n,
        const std::vector<Subsystem> &subsystems,
        double *rho_ri)
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
        if (state.is_on_gpu())
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
                    host_state.data(), n, subsystems, rho_ri);
            }
            else
            {
                std::vector<std::complex<float>> host_state(dim);
                state.to_host(host_state.data(), host_state.size());
                reduced_density_complex_from_host_statevector(
                    host_state.data(), n, subsystems, rho_ri);
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
                n, subsystems, rho_ri);
        }
        else
        {
            reduced_density_complex_from_host_statevector(
                reinterpret_cast<const std::complex<float> *>(tensor.data),
                n, subsystems, rho_ri);
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
        double *rho_ri)
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

        std::fill(rho_ri, rho_ri + 2u * kept_dim * kept_dim, 0.0);
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
    }

    template <typename Real>
    void tmi_density_matrices_from_host_statevector(
        const std::complex<Real> *psi,
        int n,
        const std::vector<TmiMatrixTerm> &terms,
        std::vector<double> &payload)
    {
        payload.resize(tmi_payload_value_count(terms));
        std::size_t offset = 0;
        for (const auto &term : terms)
        {
            append_reduced_density_for_modes_from_host_statevector(
                psi, n, term.modes, payload.data() + offset);
            offset += term_value_count(term);
        }
    }

    void cudaq_tmi_density_matrices(
        cudaq::state &state,
        int n,
        const std::vector<TmiMatrixTerm> &terms,
        std::vector<double> &payload)
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
                tmi_density_matrices_from_host_statevector(host_state.data(), n, terms, payload);
            }
            else
            {
                std::vector<std::complex<float>> host_state(dim);
                state.to_host(host_state.data(), host_state.size());
                tmi_density_matrices_from_host_statevector(host_state.data(), n, terms, payload);
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
                reinterpret_cast<const std::complex<double> *>(tensor.data), n, terms, payload);
        }
        else
        {
            tmi_density_matrices_from_host_statevector(
                reinterpret_cast<const std::complex<float> *>(tensor.data), n, terms, payload);
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



void run_1d(int n, int periods, int realizations, int res, double p_min, double p_max)
{
    if (n < RHO3_QUBITS)
    {
        throw std::invalid_argument(
            "Saving three-qubit reduced density matrices requires n >= 3.");
    }

    const std::vector<Subsystem> subsystems = periodic_ring_three_site_subsystems(n);
    std::vector<double> ps = linspace(p_min, p_max, res);
    std::ofstream infofile("info.csv");

    // Keep the existing info.csv format unchanged.
    infofile << "n,periods,realizations,resolution,p_min,p_max\n";
    infofile << n << "," << periods << "," << realizations << "," << res << ","
             << p_min << "," << p_max << "\n";

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

            auto layers = mipt_frontend(n, periods, p);

            if (r == 0)
            {
                lap1 = std::chrono::steady_clock::now();
            }

            auto state = cudaq::get_state(MIPTKernel_1D{}, n, layers, true);

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
    std::cout << "Saved " << subsystems.size()
              << " cyclic three-qubit reduced density matrices per trajectory to rho3.bin.\n";
}


void run_1d_tmi(int n, int periods, int realizations, int res, double p_min, double p_max)
{
    const int block_qubits = n / 4;
    const std::vector<TmiMatrixTerm> terms = tmi_terms_1d(n);
    std::vector<double> ps = linspace(p_min, p_max, res);
    std::ofstream infofile("info_tmi.csv");

    infofile << "n,periods,realizations,resolution,p_min,p_max,tmi_block_qubits,terms\n";
    infofile << n << "," << periods << "," << realizations << "," << res << ","
             << p_min << "," << p_max << "," << block_qubits << ",AB;AC;BC;D\n";

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

            auto layers = mipt_frontend(n, periods, p);

            if (r == 0)
            {
                lap1 = std::chrono::steady_clock::now();
            }

            auto state = cudaq::get_state(MIPTKernel_1D{}, n, layers, true);

            if (r == 0)
            {
                lap2 = std::chrono::steady_clock::now();
            }

            std::vector<double> payload;
            cudaq_tmi_density_matrices(state, n, terms, payload);

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
              << block_qubits << " modes.\n";
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
                  << " [d = 1] [n = 10] [periods = 10] [realizations = 10] [resolution = 5] [p_min = 0.0] [p_max = 1.0] [tmi = 0]\n";
        std::cout << "Description:\n";
        std::cout << "  d = dimensionality (1 or 2)\n";
        std::cout << "  n = number of qubits per dimension\n";
        std::cout << "  periods = number of periods (suggested 1D: n; 2D: 2*n^2)\n";
        std::cout << "  realizations = number of realizations (simulations per point)\n";
        std::cout << "  resolution = number of points\n";
        std::cout << "  p_min = minimum measurement rate\n";
        std::cout << "  p_max = maximum measurement rate\n";
        std::cout << "  tmi = 0 writes the existing rho3.bin format; tmi = 1 writes rho_tmi.bin for four-block TMI\n";
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
    int tmi_output = (argc > 8) ? std::stoi(argv[8]) : 0;
    if (tmi_output != 0 && tmi_output != 1)
    {
        throw std::invalid_argument("The optional tmi argument must be 0 or 1.");
    }

    auto start = std::chrono::steady_clock::now(); // Get start time

    if (d == 1)
    {
        std::cout << "Running 1D " << n <<"-qubit simulation of " << realizations*res << " circuits.\n";
        if (tmi_output == 1)
        {
            run_1d_tmi(n, periods, realizations, res, p_min, p_max);
        }
        else
        {
            run_1d(n, periods, realizations, res, p_min, p_max);
        }
    }
    else if (d == 2)
    {
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