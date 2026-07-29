#pragma once

#include "mipt/backend.hpp"
#include "mipt/circuit.hpp"
#include "mipt/native_mps.hpp"
#include "mipt/rdm.hpp"
#include "mipt/util/stats.hpp"
#include "mipt/util/text.hpp"

#ifdef MIPT_ENABLE_CUDA_RHO
#include "mipt/cuda/entropy_cuda.hpp"
#endif

#include <cudaq.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::entropy
{
using Complex = std::complex<double>;

using util::RunningStats;

inline std::vector<Complex> cudaq_state_to_host_complex128(cudaq::state &state, int n)
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
        static_cast<std::size_t>(std::max<long>(1, mipt::backend::mps_amplitude_batch_size()));
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

inline std::vector<int> cyclic_subsystem(int n, int start, int kept)
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

inline void validate_modes(int n,
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

inline std::vector<std::uint64_t> retained_basis_offsets(const std::vector<int> &kept_modes)
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

inline std::vector<Complex> reduced_density_col_major_qubit(
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

inline int fermionic_reorder_sign_from_index(std::uint64_t occupation_mask,
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

inline std::vector<Complex> reduced_density_col_major_fermion(
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

inline double second_renyi_entropy_nats(const std::vector<Complex> &rho_col_major,
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

inline std::vector<Complex> partial_trace_trailing_mode_col_major(
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

inline bool entropy_cuda_enabled()
{
    return mipt::env::boolean("MIPT_ENTROPY_CUDA", true);
}

inline bool cuda_hierarchical_translation_means(cudaq::state &state,
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

inline std::vector<double> host_hierarchical_translation_means(
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


using util::csv_quote;

inline void write_results(const std::string &path,
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

inline void print_usage(const char *argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [N] [periods] [p] [realizations] [circ_type] [output_csv]\n\n"
        << "  N            fixed number of sites/qubits/modes in the periodic 1D ring\n"
        << "  periods      number of brickwork periods (even layer + odd layer)\n"
        << "  p            measurement rate in [0,1]\n"
        << "  realizations number of independent quantum trajectories\n"
        << "  circ_type    same convention as mipt.exe:\n";
    print_circuit_type_help(std::cout, "                 ");
    std::cout
        << "  output_csv   one row per L_A=2,...,floor(N/2)\n\n"
        << "The CSV directly provides ln_x and S2_mean for the entropy-versus-ln(x) plot.\n"
        << "S2=-ln Tr(rho_A^2) uses natural logarithms. Fermionic partial tracing is\n"
        << "used automatically for circ_type 2 and 3; circ_type 4 uses the qubit trace. Smaller-block RDMs are obtained\n"
        << "by recursively tracing the largest L_A=floor(N/2) RDM.\n\n"
        << "GPU controls (GPU=1 CUDA_RHO=1 builds):\n"
        << "  MIPT_ENTROPY_CUDA=0          disable CUDA entropy evaluation\n"
        << "  MIPT_ENTROPY_CUDA_MAX_MB=512 cap temporary CUDA workspace in MiB\n";
}

} // namespace mipt::entropy
