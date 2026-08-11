#pragma once

#include "mipt/backend.hpp"
#include "mipt/circuit.hpp"
#include "mipt/entropy_resume.hpp"
#include "mipt/native_mps.hpp"
#include "mipt/rdm.hpp"
// The host von Neumann path reuses sim_tmi's Hermitian eigen-entropy: a
// dlopen'd LAPACK zheevd with a cyclic-Jacobi fallback. Keeping one
// implementation matters here -- the duplicated small-matrix Jacobi in the
// negativity paths is exactly what util/spectral.hpp was created to end.
#include "mipt/tmi/linalg.hpp"
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

// von Neumann entropy S1 = -Tr(rho ln rho), in nats.
//
// Unlike S2, this needs the whole spectrum, so the cost grows as dim^3 rather
// than dim^2. The matrix is taken by value because the eigensolver destroys it
// and the caller still needs rho for the recursive partial trace.
inline double von_neumann_entropy_nats(std::vector<Complex> rho_col_major,
                                       std::size_t dim,
                                       mipt::tmi::EntropyWorkspace &workspace)
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

    // Divide out the trajectory-normalization drift the same way the purity
    // path divides by trace^2, so the eigenvalues really are probabilities.
    const double inverse_trace = 1.0 / trace;
    for (Complex &value : rho_col_major)
    {
        value *= inverse_trace;
    }

    // entropy_hermitian_inplace works in bits and is indifferent to row- versus
    // column-major storage: transposing a Hermitian matrix conjugates it, which
    // leaves the (real) spectrum alone.
    constexpr double ln2 = 0.693147180559945309417232121458;
    const double entropy_bits =
        mipt::tmi::entropy_hermitian_inplace(rho_col_major, dim, workspace);
    if (!std::isfinite(entropy_bits))
    {
        throw std::runtime_error("von Neumann entropy is non-finite.");
    }
    return entropy_bits * ln2;
}

// Runtime controls for the von Neumann path. It is on by default, but it is
// also the expensive half of this executable: one dense eigensolve per
// (subsystem size, cyclic translation) against S2's single Frobenius pass.
inline bool von_neumann_enabled()
{
    return mipt::env::boolean("MIPT_ENTROPY_S1", true);
}

// Largest L_A the von Neumann entropy is computed for. 0 means "no cap".
// Larger blocks report NaN instead, which keeps a big-N scan affordable without
// pretending the missing values are zero.
inline int von_neumann_max_block(int max_kept)
{
    const long cap = mipt::env::integer("MIPT_ENTROPY_S1_MAX_LA", 0, 0, 62);
    if (cap <= 0)
    {
        return max_kept;
    }
    return std::min(max_kept, static_cast<int>(cap));
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

inline bool cuda_hierarchical_translation_entropies(cudaq::state &state,
                                            int n,
                                            int min_kept,
                                            int max_kept,
                                            bool fermionic_trace,
                                            int max_s1_kept,
                                            std::vector<double> &s1_means,
                                            std::vector<double> &s2_means)
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

    const std::size_t size_count =
        static_cast<std::size_t>(max_kept - min_kept + 1);
    s2_means.assign(size_count, 0.0);
    const bool want_s1 = max_s1_kept >= min_kept;
    s1_means.assign(size_count, std::numeric_limits<double>::quiet_NaN());

    int status = 0;
    if (state.get_precision() == cudaq::SimulationState::precision::fp64)
    {
        status = mipt_cuda_cyclic_entropies_f64(
            tensor.data, n, min_kept, max_kept,
            fermionic_trace ? 1 : 0, max_s1_kept,
            want_s1 ? s1_means.data() : nullptr, s2_means.data());
    }
    else
    {
        status = mipt_cuda_cyclic_entropies_f32(
            tensor.data, n, min_kept, max_kept,
            fermionic_trace ? 1 : 0, max_s1_kept,
            want_s1 ? s1_means.data() : nullptr, s2_means.data());
    }
    if (status != 0)
    {
        const char *detail = mipt_cuda_cyclic_entropies_last_error();
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
    (void)max_s1_kept;
    (void)s1_means;
    (void)s2_means;
    return false;
#endif
}

inline void host_hierarchical_translation_entropies(
    const std::vector<Complex> &psi,
    int n,
    int min_kept,
    int max_kept,
    bool fermionic_trace,
    int max_s1_kept,
    std::vector<double> &s1_means,
    std::vector<double> &s2_means)
{
    const std::size_t size_count =
        static_cast<std::size_t>(max_kept - min_kept + 1);
    s2_means.assign(size_count, 0.0);
    s1_means.assign(size_count, 0.0);
    std::vector<unsigned char> has_s1(size_count, 0);

    // One LAPACK workspace for the whole trajectory: it is re-queried only when
    // the matrix dimension changes, so reusing it across translations of the
    // same size costs nothing.
    static thread_local mipt::tmi::EntropyWorkspace eigen_workspace;

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
            const std::size_t index = static_cast<std::size_t>(kept - min_kept);
            s2_means[index] += second_renyi_entropy_nats(rho, dim);
            if (kept <= max_s1_kept)
            {
                s1_means[index] += von_neumann_entropy_nats(rho, dim, eigen_workspace);
                has_s1[index] = 1;
            }
            if (kept > min_kept)
            {
                rho = partial_trace_trailing_mode_col_major(rho, dim);
                dim /= 2;
            }
        }
    }

    for (std::size_t index = 0; index < size_count; ++index)
    {
        s2_means[index] /= static_cast<double>(n);
        s1_means[index] = has_s1[index]
                              ? s1_means[index] / static_cast<double>(n)
                              : std::numeric_limits<double>::quiet_NaN();
    }
}


using util::csv_quote;

inline void write_results(const std::string &path,
                   int n,
                   int periods,
                   double p,
                   int requested_realizations,
                   CircuitType circ_type,
                   const std::vector<RunningStats> &s1_stats,
                   const std::vector<RunningStats> &s2_stats)
{
    std::ofstream csv(path, std::ios::out | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not open output CSV: " + path);
    }
    csv << std::setprecision(17);
    csv << resume::kResultsHeader << '\n';

    constexpr double pi = 3.141592653589793238462643383279502884;
    const bool fermionic_trace = uses_fermionic_trace(circ_type);
    const int max_kept = n / 2;
    for (int kept = 2; kept <= max_kept; ++kept)
    {
        const double x = (static_cast<double>(n) / pi) *
                         std::sin(pi * static_cast<double>(kept) /
                                  static_cast<double>(n));
        const RunningStats &s1 = s1_stats[static_cast<std::size_t>(kept)];
        const RunningStats &s2 = s2_stats[static_cast<std::size_t>(kept)];
        // A block whose von Neumann entropy was skipped has no samples, so its
        // columns are NaN rather than the 0 an empty accumulator would report.
        const bool has_s1 = s1.count > 0;
        const double nan_value = std::numeric_limits<double>::quiet_NaN();
        csv << n << ','
            << periods << ','
            << p << ','
            << requested_realizations << ','
            << s2.count << ','
            << static_cast<int>(circ_type) << ','
            << csv_quote(circuit_type_name(circ_type)) << ','
            << (fermionic_trace ? 1 : 0) << ','
            << kept << ','
            << x << ','
            << std::log(x) << ','
            << (has_s1 ? s1.mean : nan_value) << ','
            << (has_s1 ? s1.sample_stddev() : nan_value) << ','
            << (has_s1 ? s1.standard_error() : nan_value) << ','
            << s2.mean << ','
            << s2.sample_stddev() << ','
            << s2.standard_error() << ','
            << n << ','
            << static_cast<std::uint64_t>(n) * s2.count << '\n';
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
        << "The CSV provides ln_x together with S1_mean and S2_mean for the\n"
        << "entropy-versus-ln(x) plot. Both use natural logarithms:\n"
        << "  S1 = -Tr(rho_A ln rho_A)   von Neumann, from the full spectrum\n"
        << "  S2 = -ln Tr(rho_A^2)       second Renyi, from the purity\n"
        << "Fermionic partial tracing is used automatically for circ_type 2 and 3;\n"
        << "circ_type 4 uses the qubit trace. Smaller-block RDMs are obtained by\n"
        << "recursively tracing the largest L_A=floor(N/2) RDM.\n\n"
        << "S1 costs one dense eigensolve per (L_A, translation) and so scales as\n"
        << "(2^L_A)^3, against a single O((2^L_A)^2) pass for S2. Skipped blocks\n"
        << "report NaN in the S1 columns, never 0.\n"
        << "  MIPT_ENTROPY_S1=0            report S2 only\n"
        << "  MIPT_ENTROPY_S1_MAX_LA=k     compute S1 only for L_A<=k (0 = no cap)\n\n"
        << "An interrupted run is resumed by re-issuing the identical command with\n"
        << "the same output_csv: that file is the checkpoint, so the averages carry\n"
        << "on rather than restarting. Trajectories are i.i.d. draws, so a resumed\n"
        << "run is statistically identical to an uninterrupted one.\n"
        << "  MIPT_ENTROPY_RESUME=0        discard the existing CSV and start over\n\n"
        << "GPU controls (GPU=1 CUDA_RHO=1 builds):\n"
        << "  MIPT_ENTROPY_CUDA=0          disable CUDA entropy evaluation\n"
        << "  MIPT_ENTROPY_CUDA_MAX_MB=512 cap temporary CUDA workspace in MiB\n"
        << "  MIPT_ENTROPY_HEEVD_MAX_ROWS=512\n"
        << "                               largest RDM cuSOLVER heevd is trusted with\n"
        << "                               for S1; larger blocks use the slower gesvd\n";
}

} // namespace mipt::entropy
