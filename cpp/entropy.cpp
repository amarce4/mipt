// entropy.cpp
// Translation-averaged second-Renyi entropy scaling for 1D MIPT trajectories.
//
// Usage:
//   ./entropy.exe [N] [periods] [p] [realizations] [circ_type] [output_csv]
//
// The output contains one row for each independent contiguous cyclic
// subsystem size L_A = 2,...,floor(N/2). Larger blocks are excluded because
// pure-state complement symmetry gives S_2(L_A)=S_2(N-L_A). The plotted
// variables corresponding to Han and Chen, arXiv:2110.10726v2, Eq. (10), are:
//
//   x = (N / pi) sin(pi L_A / N),
//   horizontal axis = ln(x),
//   vertical axis   = mean second-Renyi entropy S_2(A)
//                   = mean[-ln Tr(rho_A^2)].
//
// Natural logarithms are used throughout.  For every trajectory, S_2 is first
// averaged over all N cyclic translations of a connected block.  Error bars
// are then calculated across the independent trajectory averages, rather than
// incorrectly treating the N translated blocks from one trajectory as
// statistically independent samples.

#define main mipt_existing_main_for_entropy_build
#include "mipt.cpp"
#undef main

#ifdef MIPT_ENABLE_CUDA_RHO
#include "entropy_cuda.hpp"
#endif

#include <cstdlib>
#include <deque>
#include <iomanip>
#include <numeric>

namespace
{
    using Complex = std::complex<double>;

    struct RunningStats
    {
        std::uint64_t count = 0;
        double mean = 0.0;
        double m2 = 0.0;

        void add(double value)
        {
            ++count;
            const double delta = value - mean;
            mean += delta / static_cast<double>(count);
            const double delta2 = value - mean;
            m2 += delta * delta2;
        }

        double sample_stddev() const
        {
            return (count > 1)
                       ? std::sqrt(std::max(0.0, m2 / static_cast<double>(count - 1)))
                       : 0.0;
        }

        double standard_error() const
        {
            return (count > 0)
                       ? sample_stddev() / std::sqrt(static_cast<double>(count))
                       : 0.0;
        }
    };

    std::vector<Complex> cudaq_state_to_host_complex128(cudaq::state &state, int n)
    {
        const std::uint64_t dim_u64 = checked_pow2_u64(n);
        if (dim_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::invalid_argument("State-vector dimension does not fit in size_t.");
        }
        const std::size_t dim = static_cast<std::size_t>(dim_u64);
        const auto precision = state.get_precision();
        const auto tensor = state.get_tensor();

        std::vector<Complex> host_state(dim);
        const bool dense_rank1 =
            tensor.get_rank() == 1 && tensor.get_num_elements() >= dim && tensor.data != nullptr;

        if (dense_rank1)
        {
            if (state.is_on_gpu())
            {
                if (precision == cudaq::SimulationState::precision::fp64)
                {
                    std::vector<std::complex<double>> tmp(dim);
                    state.to_host(tmp.data(), tmp.size());
                    std::copy(tmp.begin(), tmp.end(), host_state.begin());
                }
                else
                {
                    std::vector<std::complex<float>> tmp(dim);
                    state.to_host(tmp.data(), tmp.size());
                    for (std::size_t i = 0; i < dim; ++i)
                    {
                        host_state[i] = Complex(static_cast<double>(std::real(tmp[i])),
                                                static_cast<double>(std::imag(tmp[i])));
                    }
                }
                return host_state;
            }

            if (precision == cudaq::SimulationState::precision::fp64)
            {
                const auto *ptr = reinterpret_cast<const std::complex<double> *>(tensor.data);
                std::copy(ptr, ptr + dim, host_state.begin());
            }
            else
            {
                const auto *ptr = reinterpret_cast<const std::complex<float> *>(tensor.data);
                for (std::size_t i = 0; i < dim; ++i)
                {
                    host_state[i] = Complex(static_cast<double>(std::real(ptr[i])),
                                            static_cast<double>(std::imag(ptr[i])));
                }
            }
            return host_state;
        }

        // CUDA-Q tensor-network/MPS states do not expose one contiguous rank-1
        // tensor. Query amplitudes in bounded batches using the same index and
        // bit-order handling as mipt.cpp. This remains exact, but materializing
        // all 2^N amplitudes means the entropy calculation is still exponential.
        const std::size_t requested_batch =
            static_cast<std::size_t>(std::max<long>(1, mipt_backend::mps_amplitude_batch_size()));
        std::vector<std::uint64_t> indices;
        std::vector<std::complex<double>> amplitudes;
        indices.reserve(requested_batch);

        for (std::size_t first = 0; first < dim; first += requested_batch)
        {
            const std::size_t count = std::min(requested_batch, dim - first);
            indices.clear();
            for (std::size_t j = 0; j < count; ++j)
            {
                indices.push_back(static_cast<std::uint64_t>(first + j));
            }
            cudaq_state_amplitudes_by_indices(state, n, indices, amplitudes);
            std::copy(amplitudes.begin(), amplitudes.end(), host_state.begin() + first);
        }
        return host_state;
    }

    std::vector<int> cyclic_subsystem(int n, int start, int kept)
    {
        if (kept <= 0 || kept >= n)
        {
            throw std::invalid_argument("A proper cyclic subsystem must contain between 1 and N-1 sites.");
        }
        std::vector<int> modes;
        modes.reserve(static_cast<std::size_t>(kept));
        for (int i = 0; i < kept; ++i)
        {
            modes.push_back((start + i) % n);
        }
        return modes;
    }

    void validate_modes(int n,
                        const std::vector<int> &kept_modes,
                        std::vector<unsigned char> &is_retained,
                        std::vector<int> &environment_modes)
    {
        is_retained.assign(static_cast<std::size_t>(n), 0);
        for (int q : kept_modes)
        {
            if (q < 0 || q >= n)
            {
                throw std::invalid_argument("Subsystem index is out of range.");
            }
            if (is_retained[static_cast<std::size_t>(q)])
            {
                throw std::invalid_argument("Subsystem contains duplicate sites.");
            }
            is_retained[static_cast<std::size_t>(q)] = 1;
        }

        environment_modes.clear();
        environment_modes.reserve(static_cast<std::size_t>(n - static_cast<int>(kept_modes.size())));
        for (int q = 0; q < n; ++q)
        {
            if (!is_retained[static_cast<std::size_t>(q)])
            {
                environment_modes.push_back(q);
            }
        }
    }

    std::vector<std::uint64_t> retained_basis_offsets(const std::vector<int> &kept_modes)
    {
        const int kept = static_cast<int>(kept_modes.size());
        const std::size_t retained_dim = static_cast<std::size_t>(checked_pow2_u64(kept));
        std::vector<std::uint64_t> offsets(retained_dim, 0);

        for (std::size_t local = 0; local < retained_dim; ++local)
        {
            std::uint64_t offset = 0;
            for (int bit = 0; bit < kept; ++bit)
            {
                if ((local >> bit) & std::size_t{1})
                {
                    offset |= std::uint64_t{1} << kept_modes[static_cast<std::size_t>(bit)];
                }
            }
            offsets[local] = offset;
        }
        return offsets;
    }

    std::vector<Complex> reduced_density_col_major_qubit(
        const std::vector<Complex> &psi,
        int n,
        const std::vector<int> &kept_modes)
    {
        const int kept = static_cast<int>(kept_modes.size());
        const std::uint64_t retained_dim_u64 = checked_pow2_u64(kept);
        const std::uint64_t env_dim_u64 = checked_pow2_u64(n - kept);
        const std::size_t retained_dim = static_cast<std::size_t>(retained_dim_u64);

        std::vector<unsigned char> is_retained;
        std::vector<int> environment_modes;
        validate_modes(n, kept_modes, is_retained, environment_modes);
        const std::vector<std::uint64_t> retained_offsets = retained_basis_offsets(kept_modes);

        std::vector<Complex> rho(retained_dim * retained_dim, Complex(0.0, 0.0));
        for (std::uint64_t env = 0; env < env_dim_u64; ++env)
        {
            std::uint64_t base = 0;
            for (std::size_t e = 0; e < environment_modes.size(); ++e)
            {
                if ((env >> e) & std::uint64_t{1})
                {
                    base |= std::uint64_t{1} << environment_modes[e];
                }
            }

            for (std::size_t row = 0; row < retained_dim; ++row)
            {
                const Complex a = psi[static_cast<std::size_t>(base | retained_offsets[row])];
                for (std::size_t col = 0; col < retained_dim; ++col)
                {
                    const Complex b = psi[static_cast<std::size_t>(base | retained_offsets[col])];
                    rho[col * retained_dim + row] += a * std::conj(b);
                }
            }
        }
        return rho;
    }

    int fermionic_reorder_sign_from_index(std::uint64_t occupation_mask,
                                           const std::vector<int> &position_in_new_order,
                                           int n)
    {
        // Sign of the permutation that moves the retained modes to the front
        // in kept_modes order, followed by the environment modes in canonical
        // order. This realizes the fermionic tensor-factor ordering before the
        // ordinary environment trace.
        std::uint64_t seen_new_positions = 0;
        int parity = 0;
        for (int mode = 0; mode < n; ++mode)
        {
            if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
            {
                continue;
            }
            const int pos = position_in_new_order[static_cast<std::size_t>(mode)];
            const std::uint64_t lower_or_equal =
                (std::uint64_t{1} << (pos + 1)) - std::uint64_t{1};
            parity ^= (__builtin_popcountll(seen_new_positions & ~lower_or_equal) & 1);
            seen_new_positions |= std::uint64_t{1} << pos;
        }
        return parity ? -1 : 1;
    }

    std::vector<Complex> reduced_density_col_major_fermion(
        const std::vector<Complex> &psi,
        int n,
        const std::vector<int> &kept_modes)
    {
        const int kept = static_cast<int>(kept_modes.size());
        const std::uint64_t retained_dim_u64 = checked_pow2_u64(kept);
        const std::uint64_t env_dim_u64 = checked_pow2_u64(n - kept);
        const std::size_t retained_dim = static_cast<std::size_t>(retained_dim_u64);

        std::vector<unsigned char> is_retained;
        std::vector<int> environment_modes;
        validate_modes(n, kept_modes, is_retained, environment_modes);

        std::vector<int> new_mode_order = kept_modes;
        new_mode_order.insert(new_mode_order.end(),
                              environment_modes.begin(), environment_modes.end());
        std::vector<int> position_in_new_order(static_cast<std::size_t>(n), 0);
        for (std::size_t pos = 0; pos < new_mode_order.size(); ++pos)
        {
            position_in_new_order[static_cast<std::size_t>(new_mode_order[pos])] =
                static_cast<int>(pos);
        }

        const std::vector<std::uint64_t> retained_offsets = retained_basis_offsets(kept_modes);
        std::vector<Complex> rho(retained_dim * retained_dim, Complex(0.0, 0.0));
        std::vector<Complex> amplitudes(retained_dim);

        for (std::uint64_t env = 0; env < env_dim_u64; ++env)
        {
            std::uint64_t base = 0;
            for (std::size_t e = 0; e < environment_modes.size(); ++e)
            {
                if ((env >> e) & std::uint64_t{1})
                {
                    base |= std::uint64_t{1} << environment_modes[e];
                }
            }

            for (std::size_t local = 0; local < retained_dim; ++local)
            {
                const std::uint64_t old_index = base | retained_offsets[local];
                const int sign =
                    fermionic_reorder_sign_from_index(old_index, position_in_new_order, n);
                amplitudes[local] =
                    psi[static_cast<std::size_t>(old_index)] * static_cast<double>(sign);
            }

            for (std::size_t row = 0; row < retained_dim; ++row)
            {
                const Complex a = amplitudes[row];
                for (std::size_t col = 0; col < retained_dim; ++col)
                {
                    rho[col * retained_dim + row] += a * std::conj(amplitudes[col]);
                }
            }
        }
        return rho;
    }

    double second_renyi_entropy_nats(const std::vector<Complex> &rho_col_major,
                                      std::size_t dim)
    {
        if (rho_col_major.size() != dim * dim)
        {
            throw std::invalid_argument("Reduced-density matrix payload has the wrong size.");
        }

        double trace = 0.0;
        for (std::size_t i = 0; i < dim; ++i)
        {
            trace += std::real(rho_col_major[i * dim + i]);
        }
        if (!(trace > 0.0) || !std::isfinite(trace))
        {
            throw std::runtime_error("Reduced-density matrix has a non-positive or non-finite trace.");
        }

        // For a Hermitian matrix, Tr(rho^2) = sum_ij |rho_ij|^2. Dividing by
        // trace^2 removes tiny trajectory-normalization drift without an
        // unnecessary eigenvalue decomposition.
        long double frobenius_squared = 0.0L;
        for (const Complex &value : rho_col_major)
        {
            frobenius_squared += static_cast<long double>(std::norm(value));
        }
        double purity = static_cast<double>(frobenius_squared) / (trace * trace);
        if (!std::isfinite(purity) || !(purity > 0.0))
        {
            throw std::runtime_error("Reduced-density purity is non-positive or non-finite.");
        }

        constexpr double roundoff_tol = 1e-8;
        if (purity > 1.0 + roundoff_tol)
        {
            throw std::runtime_error("Reduced-density purity exceeds one beyond numerical tolerance: " +
                                     std::to_string(purity));
        }
        purity = std::min(1.0, purity);
        return -std::log(purity);
    }

    std::vector<Complex> partial_trace_trailing_mode_col_major(
        const std::vector<Complex> &rho_col_major,
        std::size_t input_dim)
    {
        if (input_dim < 2 || (input_dim & 1u) != 0u ||
            rho_col_major.size() != input_dim * input_dim)
        {
            throw std::invalid_argument(
                "Trailing-mode partial trace received an invalid density matrix.");
        }

        const std::size_t output_dim = input_dim / 2;
        std::vector<Complex> reduced(output_dim * output_dim, Complex(0.0, 0.0));
        for (std::size_t col = 0; col < output_dim; ++col)
        {
            for (std::size_t row = 0; row < output_dim; ++row)
            {
                const Complex zero_sector = rho_col_major[col * input_dim + row];
                const Complex one_sector =
                    rho_col_major[(col + output_dim) * input_dim + (row + output_dim)];
                reduced[col * output_dim + row] = zero_sector + one_sector;
            }
        }
        return reduced;
    }

    bool entropy_cuda_enabled()
    {
        const char *value = std::getenv("MIPT_ENTROPY_CUDA");
        if (value == nullptr || *value == '\0')
        {
            return true;
        }
        return std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0 &&
               std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "OFF") != 0;
    }

    bool cuda_hierarchical_translation_means(cudaq::state &state,
                                             int n,
                                             int min_kept,
                                             int max_kept,
                                             bool fermionic_trace,
                                             std::vector<double> &means)
    {
#ifdef MIPT_ENABLE_CUDA_RHO
        if (!entropy_cuda_enabled() || !state.is_on_gpu() || n >= 31)
        {
            return false;
        }

        const std::uint64_t dim_u64 = checked_pow2_u64(n);
        const auto tensor = state.get_tensor();
        const bool dense_rank1 =
            tensor.get_rank() == 1 &&
            tensor.get_num_elements() >= static_cast<std::size_t>(dim_u64) &&
            tensor.data != nullptr;
        if (!dense_rank1)
        {
            return false;
        }

        means.assign(static_cast<std::size_t>(max_kept - min_kept + 1), 0.0);
        int status = 0;
        if (state.get_precision() == cudaq::SimulationState::precision::fp64)
        {
            status = mipt_cuda_cyclic_s2_f64(
                tensor.data, n, min_kept, max_kept,
                fermionic_trace ? 1 : 0, means.data());
        }
        else
        {
            status = mipt_cuda_cyclic_s2_f32(
                tensor.data, n, min_kept, max_kept,
                fermionic_trace ? 1 : 0, means.data());
        }
        if (status != 0)
        {
            const char *detail = mipt_cuda_cyclic_s2_last_error();
            throw std::runtime_error(
                "CUDA hierarchical entropy evaluation failed (status=" +
                std::to_string(status) + "): " +
                ((detail != nullptr && *detail != '\0') ? detail : "unknown error"));
        }
        return true;
#else
        (void)state;
        (void)n;
        (void)min_kept;
        (void)max_kept;
        (void)fermionic_trace;
        (void)means;
        return false;
#endif
    }

    std::vector<double> host_hierarchical_translation_means(
        const std::vector<Complex> &psi,
        int n,
        int min_kept,
        int max_kept,
        bool fermionic_trace)
    {
        std::vector<double> sums(
            static_cast<std::size_t>(max_kept - min_kept + 1), 0.0);

        for (int start = 0; start < n; ++start)
        {
            const std::vector<int> subsystem = cyclic_subsystem(n, start, max_kept);
            std::vector<Complex> rho =
                fermionic_trace
                    ? reduced_density_col_major_fermion(psi, n, subsystem)
                    : reduced_density_col_major_qubit(psi, n, subsystem);
            std::size_t dim = static_cast<std::size_t>(checked_pow2_u64(max_kept));

            for (int kept = max_kept; kept >= min_kept; --kept)
            {
                sums[static_cast<std::size_t>(kept - min_kept)] +=
                    second_renyi_entropy_nats(rho, dim);
                if (kept > min_kept)
                {
                    rho = partial_trace_trailing_mode_col_major(rho, dim);
                    dim /= 2;
                }
            }
        }

        for (double &value : sums)
        {
            value /= static_cast<double>(n);
        }
        return sums;
    }

    class CircuitWorkspace
    {
      public:
        explicit CircuitWorkspace(int periods)
            : mms_rng_(std::random_device{}()),
              haar_rng_(std::random_device{}()),
              fermion_rng_(std::random_device{}()),
              rfgs_rng_(std::random_device{}())
        {
            mms_layers_.reserve(static_cast<std::size_t>(2 * periods + 1));
            haar_layers_.reserve(static_cast<std::size_t>(2 * periods));
            fermion_layers_.reserve(static_cast<std::size_t>(2 * periods + 1));
            rfgs_layers_.reserve(static_cast<std::size_t>(2 * periods + 1));
        }

        cudaq::state simulate(int n, int periods, double p, CircuitType circ_type)
        {
            switch (circ_type)
            {
            case CircuitType::MMS:
                mipt_frontend_inplace(mms_layers_, n, periods, p, mms_rng_);
                apply_debug_prefix_layer_limit(mms_layers_, "MMS");
                return cudaq::get_state(MIPTKernel_1D{}, n, mms_layers_, true);

            case CircuitType::Haar:
                haar_frontend_inplace(haar_layers_, n, periods, p, haar_rng_, true);
                apply_debug_prefix_layer_limit(haar_layers_, "Haar");
                return cudaq::get_state(MIPTHaarKernel_1D{}, n, haar_layers_);

            case CircuitType::FermionRPPU:
            {
                const bool direct_boundary = mipt_backend::direct_fermion_boundary_enabled();
                mipt_fermion_frontend_inplace(fermion_layers_, n, periods, p, true,
                                              fermion_rng_, direct_boundary);
                apply_debug_prefix_layer_limit(fermion_layers_, "FermionRPPU");
                return cudaq::get_state(MIPTFermionKernel_1D{}, n, fermion_layers_);
            }

            case CircuitType::RFGS:
                frgs_mipt_frontend_inplace(rfgs_layers_, n, periods, p, rfgs_rng_, true);
                apply_debug_prefix_layer_limit(rfgs_layers_, "RFGS");
                return cudaq::get_state(MIPTKernel_1D_FRGS{}, n, rfgs_layers_, true);
            }
            throw std::logic_error("Unhandled circuit type.");
        }

      private:
        std::mt19937 mms_rng_;
        std::mt19937 haar_rng_;
        std::mt19937 fermion_rng_;
        std::mt19937 rfgs_rng_;
        std::vector<LayerData> mms_layers_;
        std::vector<HaarLayerData> haar_layers_;
        std::vector<FermionLayerData> fermion_layers_;
        std::vector<FRGSLayerData> rfgs_layers_;
    };

    std::string csv_quote(const std::string &value)
    {
        std::string escaped;
        escaped.reserve(value.size() + 2);
        escaped.push_back('"');
        for (char c : value)
        {
            if (c == '"')
            {
                escaped.push_back('"');
            }
            escaped.push_back(c);
        }
        escaped.push_back('"');
        return escaped;
    }

    void write_results(const std::string &path,
                       int n,
                       int periods,
                       double p,
                       int requested_realizations,
                       CircuitType circ_type,
                       const std::vector<RunningStats> &stats)
    {
        std::ofstream csv(path, std::ios::out | std::ios::trunc);
        if (!csv)
        {
            throw std::runtime_error("Could not open output CSV: " + path);
        }
        csv << std::setprecision(17);
        csv << "N,periods,p,requested_realizations,completed_realizations,circ_type,"
               "circuit_name,fermionic_trace,L_A,x,ln_x,S2_mean,S2_stddev,S2_stderr,"
               "subsystems_per_realization,total_subsystems\n";

        constexpr double pi = 3.141592653589793238462643383279502884;
        const bool fermionic_trace = uses_fermionic_trace(circ_type);
        const int max_kept = n / 2;
        for (int kept = 2; kept <= max_kept; ++kept)
        {
            const double x = (static_cast<double>(n) / pi) *
                             std::sin(pi * static_cast<double>(kept) /
                                      static_cast<double>(n));
            const RunningStats &s = stats[static_cast<std::size_t>(kept)];
            csv << n << ','
                << periods << ','
                << p << ','
                << requested_realizations << ','
                << s.count << ','
                << static_cast<int>(circ_type) << ','
                << csv_quote(circuit_type_name(circ_type)) << ','
                << (fermionic_trace ? 1 : 0) << ','
                << kept << ','
                << x << ','
                << std::log(x) << ','
                << s.mean << ','
                << s.sample_stddev() << ','
                << s.standard_error() << ','
                << n << ','
                << static_cast<std::uint64_t>(n) * s.count << '\n';
        }
        csv.flush();
        if (!csv)
        {
            throw std::runtime_error("Failed while writing output CSV: " + path);
        }
    }

    void print_usage(const char *argv0)
    {
        std::cout
            << "Usage: " << argv0
            << " [N] [periods] [p] [realizations] [circ_type] [output_csv]\n\n"
            << "  N            fixed number of sites/qubits/modes in the periodic 1D ring\n"
            << "  periods      number of brickwork periods (even layer + odd layer)\n"
            << "  p            measurement rate in [0,1]\n"
            << "  realizations number of independent quantum trajectories\n"
            << "  circ_type    same convention as mipt.exe:\n"
            << "                 0 = MMS gate set\n"
            << "                 1 = random Haar U(4) brickwork circuit\n"
            << "                 2 = fermionic random parity-preserving unitary (RPPU)\n"
            << "                 3 = RFGS fermionic circuit\n"
            << "  output_csv   one row per L_A=2,...,floor(N/2)\n\n"
            << "The CSV directly provides ln_x and S2_mean for the entropy-versus-ln(x) plot.\n"
            << "S2=-ln Tr(rho_A^2) uses natural logarithms. Fermionic partial tracing is\n"
            << "used automatically for circ_type 2 and 3. Smaller-block RDMs are obtained\n"
            << "by recursively tracing the largest L_A=floor(N/2) RDM.\n\n"
            << "GPU controls (GPU=1 CUDA_RHO=1 builds):\n"
            << "  MIPT_ENTROPY_CUDA=0          disable CUDA entropy evaluation\n"
            << "  MIPT_ENTROPY_CUDA_MAX_MB=512 cap temporary CUDA workspace in MiB\n";
    }
}

int main(int argc, char *argv[])
{
    try
    {
        mipt_backend::print_backend_banner_once("entropy.exe");

        if (argc > 1 &&
            (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0))
        {
            print_usage(argv[0]);
            return 0;
        }
        if (argc != 7)
        {
            print_usage(argv[0]);
            return 2;
        }

        const int n = std::stoi(argv[1]);
        const int periods = std::stoi(argv[2]);
        const double p = std::stod(argv[3]);
        const int realizations = std::stoi(argv[4]);
        const CircuitType circ_type = parse_circuit_type(std::stoi(argv[5]));
        const std::string output_csv = argv[6];

        if (n < 4)
        {
            throw std::invalid_argument("N must be at least 4 for the requested cyclic entropy scan.");
        }
        if (n >= 63)
        {
            throw std::invalid_argument("N must be less than 63 for exact basis indexing.");
        }
        if (periods < 0)
        {
            throw std::invalid_argument("periods must be non-negative.");
        }
        if (p < 0.0 || p > 1.0)
        {
            throw std::invalid_argument("p must be in [0,1].");
        }
        if (realizations <= 0)
        {
            throw std::invalid_argument("realizations must be positive.");
        }
        if ((circ_type == CircuitType::Haar ||
             circ_type == CircuitType::FermionRPPU ||
             circ_type == CircuitType::RFGS) &&
            (n % 2 != 0))
        {
            throw std::invalid_argument("circ_type 1, 2, and 3 require even N for periodic brickwork geometry.");
        }

        if (mipt_native_mps::enabled())
        {
            std::cerr
                << "Warning: MIPT_NATIVE_MPS=1 is not used by entropy.exe because arbitrary-block "
                   "second-Renyi entropy requires an explicit trajectory state. The selected CUDA-Q "
                   "get_state backend is used instead.\n";
        }

        const bool fermion_trace = uses_fermionic_trace(circ_type);
        const int max_small_block = n / 2;
        std::vector<RunningStats> stats(static_cast<std::size_t>(n + 1));
        CircuitWorkspace workspace(periods);

        std::cout << "Running second-Renyi chord-length scan: N=" << n
                  << ", periods=" << periods
                  << ", p=" << p
                  << ", realizations=" << realizations
                  << ", circ_type=" << static_cast<int>(circ_type)
                  << " (" << circuit_type_name(circ_type) << ')'
                  << ", fermionic_trace=" << (fermion_trace ? 1 : 0)
                  << "\n";
        std::cout << "Averaging each L_A over " << n
                  << " contiguous cyclic translations per traj.\n";

        double seconds_per_trajectory = 0.0;
        constexpr std::size_t recent_rate_window = 10;
        std::deque<double> recent_trajectory_seconds;
        double recent_seconds_sum = 0.0;
        const auto total_start = std::chrono::steady_clock::now();

        for (int r = 0; r < realizations; ++r)
        {
            const auto trajectory_start = std::chrono::steady_clock::now();
            const auto circuit_start = trajectory_start;
            auto state = workspace.simulate(n, periods, p, circ_type);
            const auto circuit_end = std::chrono::steady_clock::now();

            constexpr int min_kept = 2;
            std::vector<double> trajectory_means;
            const bool used_cuda_entropy = cuda_hierarchical_translation_means(
                state, n, min_kept, max_small_block, fermion_trace,
                trajectory_means);
            if (!used_cuda_entropy)
            {
                const std::vector<Complex> psi =
                    cudaq_state_to_host_complex128(state, n);
                trajectory_means = host_hierarchical_translation_means(
                    psi, n, min_kept, max_small_block, fermion_trace);
            }
            const auto entropy_end = std::chrono::steady_clock::now();

            for (int kept = min_kept; kept <= max_small_block; ++kept)
            {
                stats[static_cast<std::size_t>(kept)].add(
                    trajectory_means[static_cast<std::size_t>(kept - min_kept)]);
            }

            if (r == 0)
            {
                const double circuit_seconds =
                    std::chrono::duration<double>(circuit_end - circuit_start).count();
                const double entropy_seconds =
                    std::chrono::duration<double>(entropy_end - circuit_end).count();
                std::cout << "Entropy backend: "
                          << (used_cuda_entropy
                                  ? "CUDA half-chain RDM + recursive GPU partial traces"
                                  : "host half-chain RDM + recursive host partial traces")
                          << "\n1st traj. timing: circuit=" << circuit_seconds
                          << "s, entropy=" << entropy_seconds << "s\n";
            }

            // Keep a valid checkpoint after every completed trajectory. The CSV
            // is small (O(N) rows), so this is negligible beside state evolution
            // and reduced-density construction.
            write_results(output_csv, n, periods, p, realizations, circ_type, stats);

            const auto trajectory_end = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(trajectory_end - trajectory_start).count();
            seconds_per_trajectory =
                (r == 0)
                    ? elapsed
                    : (seconds_per_trajectory * static_cast<double>(r) + elapsed) /
                          static_cast<double>(r + 1);

            recent_trajectory_seconds.push_back(elapsed);
            recent_seconds_sum += elapsed;
            if (recent_trajectory_seconds.size() > recent_rate_window)
            {
                recent_seconds_sum -= recent_trajectory_seconds.front();
                recent_trajectory_seconds.pop_front();
            }

            const int remaining = realizations - r - 1;
            const double avg_rate =
                (seconds_per_trajectory > 0.0) ? 1.0 / seconds_per_trajectory : 0.0;
            const double recent_rate =
                (recent_seconds_sum > 0.0)
                    ? static_cast<double>(recent_trajectory_seconds.size()) / recent_seconds_sum
                    : 0.0;
            const double recent_seconds_per_trajectory =
                recent_trajectory_seconds.empty()
                    ? seconds_per_trajectory
                    : recent_seconds_sum /
                          static_cast<double>(recent_trajectory_seconds.size());

            std::cout << "\r" << (r + 1) << '/' << realizations
                      << " traj. | recent(" << recent_trajectory_seconds.size() << ") "
                      << std::round(recent_rate * 100.0) / 100.0
                      << " traj/s"
                      << " | avg. "
                      << std::round(avg_rate * 100.0) / 100.0
                      << " traj/s | ETA "
                      << static_cast<int>(remaining * recent_seconds_per_trajectory)
                      << "s          " << std::flush;
        }

        const auto total_end = std::chrono::steady_clock::now();
        const double total_seconds =
            std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "\nSaved second-Renyi scaling data to " << output_csv
                  << ". Elapsed: " << total_seconds << "s\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "entropy.exe error: " << e.what() << '\n';
        return 1;
    }
}
