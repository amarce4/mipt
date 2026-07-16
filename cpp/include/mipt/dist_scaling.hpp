#pragma once

#include "mipt/backend.hpp"
#include "mipt/circuit.hpp"
#include "mipt/dist_metrics.hpp"
#include "mipt/env.hpp"
#include "mipt/rdm.hpp"
#include "mipt/types.hpp"

#ifdef MIPT_ENABLE_CUDA_RHO
#include "mipt/cuda/rho2_reduce.hpp"
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mipt::dist
{
constexpr int RHO2_DIM = 4;
constexpr std::size_t RHO2_VALUES = 2u * RHO2_DIM * RHO2_DIM;
using Pair = std::array<int, 2>;

inline std::vector<Pair> all_unordered_pairs(int n)
{
    std::vector<Pair> pairs;
    pairs.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n - 1) / 2u);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            pairs.push_back({i, j});
        }
    }
    return pairs;
}

inline std::vector<int> flatten_pairs(const std::vector<Pair> &pairs)
{
    std::vector<int> flat;
    flat.reserve(2u * pairs.size());
    for (const Pair &pair : pairs)
    {
        flat.push_back(pair[0]);
        flat.push_back(pair[1]);
    }
    return flat;
}

inline void append_double(std::string &out, double value)
{
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                      std::chars_format::general, 17);
    if (result.ec == std::errc{})
    {
        out.append(buffer, result.ptr);
        return;
    }
    out += std::to_string(value);
}

inline std::string compact_decimal(double value)
{
    if (std::abs(value) < 1.0e-15)
    {
        value = 0.0;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(12) << value;
    std::string text = out.str();
    while (!text.empty() && text.back() == '0')
    {
        text.pop_back();
    }
    if (!text.empty() && text.back() == '.')
    {
        text.pop_back();
    }
    return text == "-0" ? "0" : text;
}

inline std::string default_output_path(int n,
                                       int periods,
                                       double p,
                                       int realizations,
                                       CircuitType circuit_type)
{
    const std::string tag(circuit_type_tag(circuit_type));
    std::ostringstream filename;
    filename << "dist_scaling_" << tag
             << "_n_" << n
             << "_periods_" << periods
             << "_p_" << compact_decimal(p)
             << "_real_" << realizations
             << ".csv";
    return std::string("csv/dist_scaling/") + tag + "/" + filename.str();
}

inline std::string format_duration(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0)
    {
        return "--";
    }
    const auto total = static_cast<std::uint64_t>(std::llround(seconds));
    const std::uint64_t hours = total / 3600;
    const std::uint64_t minutes = (total % 3600) / 60;
    const std::uint64_t secs = total % 60;

    std::ostringstream out;
    if (hours > 0)
    {
        out << hours << "h ";
    }
    if (hours > 0 || minutes > 0)
    {
        out << minutes << "m ";
    }
    out << secs << 's';
    return out.str();
}

inline long dense_fallback_max_qubits()
{
    return env::integer("MIPT_DIST_MPS_DENSE_MAX_QUBITS", 16, 2, 30);
}

inline long progress_interval_ms()
{
    return env::integer("MIPT_DIST_PROGRESS_MS", 1000, 0, 60000);
}

inline std::string pause_file_path()
{
    const char *value = std::getenv("MIPT_DIST_PAUSE_FILE");
    if (env::disables_path(value))
    {
        return {};
    }
    if (value != nullptr && *value != '\0')
    {
        return std::string(value);
    }
    return "PAUSE_MIPT";
}

inline bool file_exists(const std::string &path)
{
    std::error_code error;
    return !path.empty() && std::filesystem::exists(path, error) && !error;
}

inline void ensure_output_parent_directory(const std::string &output_path)
{
    const std::filesystem::path parent = std::filesystem::path(output_path).parent_path();
    if (parent.empty())
    {
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error)
    {
        throw std::runtime_error("Could not create output directory " +
                                 parent.string() + ": " + error.message());
    }
}

inline void validate_args(int n, int periods, double p, int realizations,
                          CircuitType circuit_type)
{
    if (n < 2 || n >= 63)
    {
        throw std::invalid_argument("N must satisfy 2 <= N < 63.");
    }
    if (periods <= 0)
    {
        throw std::invalid_argument("periods must be positive.");
    }
    if (!(p >= 0.0 && p <= 1.0) || !std::isfinite(p))
    {
        throw std::invalid_argument("p must lie in [0, 1].");
    }
    if (realizations <= 0)
    {
        throw std::invalid_argument("realizations must be positive.");
    }
    validate_circuit_site_count(circuit_type, n, "N");
}

inline bool odd_popcount(std::uint64_t value)
{
#if defined(__GNUG__) || defined(__clang__)
    return (__builtin_popcountll(static_cast<unsigned long long>(value)) & 1) != 0;
#else
    bool odd = false;
    while (value)
    {
        odd = !odd;
        value &= value - 1;
    }
    return odd;
#endif
}

inline std::vector<std::uint64_t> basis_offsets_for_modes(
    const std::vector<int> &modes)
{
    const std::size_t dimension = std::size_t{1} << modes.size();
    std::vector<std::uint64_t> offsets(dimension, 0);
    std::size_t populated = 1;
    for (int mode : modes)
    {
        const std::uint64_t bit = std::uint64_t{1} << mode;
        for (std::size_t i = 0; i < populated; ++i)
        {
            offsets[populated + i] = offsets[i] | bit;
        }
        populated *= 2;
    }
    return offsets;
}

inline std::array<std::uint64_t, RHO2_DIM> fermion_cross_masks(
    const Pair &pair,
    const std::vector<int> &environment_modes)
{
    std::array<std::uint64_t, RHO2_DIM> masks{};
    for (int local_basis = 0; local_basis < RHO2_DIM; ++local_basis)
    {
        std::uint64_t mask = 0;
        for (std::size_t e = 0; e < environment_modes.size(); ++e)
        {
            bool crosses = false;
            for (int k = 0; k < 2; ++k)
            {
                if (((local_basis >> k) & 1) && environment_modes[e] < pair[k])
                {
                    crosses = !crosses;
                }
            }
            if (crosses)
            {
                mask |= std::uint64_t{1} << e;
            }
        }
        masks[static_cast<std::size_t>(local_basis)] = mask;
    }
    return masks;
}

template <typename Real>
inline void rho2_subsystems_from_host_statevector(
    const std::complex<Real> *statevector,
    int n,
    const std::vector<Pair> &pairs,
    bool fermion_trace,
    std::vector<double> &rho_ri)
{
    if (statevector == nullptr || pairs.empty())
    {
        throw std::invalid_argument("Invalid host-state two-site RDM request.");
    }
    rho_ri.assign(pairs.size() * RHO2_VALUES, 0.0);

    for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index)
    {
        const Pair &pair = pairs[pair_index];
        std::vector<int> environment_modes;
        environment_modes.reserve(static_cast<std::size_t>(n - 2));
        for (int q = 0; q < n; ++q)
        {
            if (q != pair[0] && q != pair[1])
            {
                environment_modes.push_back(q);
            }
        }
        const std::vector<std::uint64_t> environment_offsets =
            basis_offsets_for_modes(environment_modes);
        const std::array<std::uint64_t, RHO2_DIM> cross_masks =
            fermion_trace ? fermion_cross_masks(pair, environment_modes)
                          : std::array<std::uint64_t, RHO2_DIM>{};

        std::array<std::uint64_t, RHO2_DIM> retained_offsets{};
        std::array<bool, RHO2_DIM> internal_odd{};
        for (int local_basis = 0; local_basis < RHO2_DIM; ++local_basis)
        {
            retained_offsets[static_cast<std::size_t>(local_basis)] =
                (static_cast<std::uint64_t>((local_basis >> 0) & 1) << pair[0]) |
                (static_cast<std::uint64_t>((local_basis >> 1) & 1) << pair[1]);
            internal_odd[static_cast<std::size_t>(local_basis)] =
                fermion_trace && pair[0] > pair[1] && local_basis == 3;
        }

        double *out = rho_ri.data() + pair_index * RHO2_VALUES;
        for (std::size_t env = 0; env < environment_offsets.size(); ++env)
        {
            std::array<std::complex<double>, RHO2_DIM> amplitudes{};
            for (int local_basis = 0; local_basis < RHO2_DIM; ++local_basis)
            {
                const std::uint64_t state_index =
                    environment_offsets[env] +
                    retained_offsets[static_cast<std::size_t>(local_basis)];
                const auto raw = statevector[state_index];
                std::complex<double> value(static_cast<double>(raw.real()),
                                           static_cast<double>(raw.imag()));
                const bool sign_odd = internal_odd[static_cast<std::size_t>(local_basis)] !=
                    odd_popcount(static_cast<std::uint64_t>(env) &
                                 cross_masks[static_cast<std::size_t>(local_basis)]);
                if (fermion_trace && sign_odd)
                {
                    value = -value;
                }
                amplitudes[static_cast<std::size_t>(local_basis)] = value;
            }

            for (int row = 0; row < RHO2_DIM; ++row)
            {
                for (int col = 0; col < RHO2_DIM; ++col)
                {
                    const std::complex<double> value =
                        amplitudes[static_cast<std::size_t>(row)] *
                        std::conj(amplitudes[static_cast<std::size_t>(col)]);
                    const std::size_t element =
                        static_cast<std::size_t>(row * RHO2_DIM + col);
                    out[2u * element + 0u] += value.real();
                    out[2u * element + 1u] += value.imag();
                }
            }
        }
    }
}

inline void dense_state_from_amplitudes(cudaq::state &state,
                                        int n,
                                        std::vector<std::complex<double>> &out)
{
    if (n > dense_fallback_max_qubits())
    {
        throw std::runtime_error(
            "The selected CUDA-Q tensor/MPS backend did not expose a dense rank-1 state tensor. "
            "Exact all-pairs distance scaling would require reconstructing 2^N amplitudes; "
            "increase MIPT_DIST_MPS_DENSE_MAX_QUBITS only for manageable validation sizes, "
            "or use the nvidia statevector target.");
    }

    const std::uint64_t dimension_u64 = checked_pow2_u64(n);
    const std::size_t dimension = static_cast<std::size_t>(dimension_u64);
    out.resize(dimension);
    const std::uint64_t batch_size =
        static_cast<std::uint64_t>(backend::mps_amplitude_batch_size());
    std::vector<std::uint64_t> indices;
    std::vector<std::complex<double>> amplitudes;
    double norm2 = 0.0;

    for (std::uint64_t begin = 0; begin < dimension_u64; begin += batch_size)
    {
        const std::uint64_t count =
            std::min<std::uint64_t>(batch_size, dimension_u64 - begin);
        indices.clear();
        indices.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t k = 0; k < count; ++k)
        {
            indices.push_back(begin + k);
        }
        cudaq_state_amplitudes_by_indices(state, n, indices, amplitudes);
        for (std::uint64_t k = 0; k < count; ++k)
        {
            const auto value = amplitudes[static_cast<std::size_t>(k)];
            out[static_cast<std::size_t>(begin + k)] = value;
            norm2 += std::norm(value);
        }
    }

    if (!(norm2 > 0.0) || !std::isfinite(norm2))
    {
        throw std::runtime_error("Dense amplitude reconstruction produced an invalid norm.");
    }
    const double inverse_norm = 1.0 / std::sqrt(norm2);
    for (auto &value : out)
    {
        value *= inverse_norm;
    }
}

struct RdmWorkspace
{
    std::vector<double> rho_ri;
    std::vector<std::complex<double>> host_state_f64;
    std::vector<std::complex<float>> host_state_f32;
    std::vector<std::complex<double>> amplitude_state;
};

inline const char *rho2_backend_name(cudaq::state &state, int n)
{
    const auto tensor = state.get_tensor();
    const std::size_t dimension = static_cast<std::size_t>(checked_pow2_u64(n));
    const bool dense = tensor.get_rank() == 1 &&
                       tensor.get_num_elements() >= dimension &&
                       tensor.data != nullptr;
#ifdef MIPT_ENABLE_CUDA_RHO
    if (dense && state.is_on_gpu())
    {
        return "CUDA batched rho2 reduction";
    }
#endif
    if (dense && state.is_on_gpu())
    {
        return "single device-to-host state copy + host rho2 reduction";
    }
    if (dense)
    {
        return "host rho2 reduction";
    }
    return "CUDA-Q amplitude reconstruction + host rho2 reduction";
}

inline void rho2_subsystems_from_cudaq_state(cudaq::state &state,
                                              int n,
                                              const std::vector<Pair> &pairs,
                                              bool fermion_trace,
                                              RdmWorkspace &workspace)
{
    const std::uint64_t dimension_u64 = checked_pow2_u64(n);
    if (dimension_u64 > static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("State-vector dimension does not fit in size_t.");
    }
    const std::size_t dimension = static_cast<std::size_t>(dimension_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();
    const bool dense = tensor.get_rank() == 1 &&
                       tensor.get_num_elements() >= dimension &&
                       tensor.data != nullptr;

#ifdef MIPT_ENABLE_CUDA_RHO
    if (dense && state.is_on_gpu())
    {
        const std::vector<int> flat_pairs = flatten_pairs(pairs);
        workspace.rho_ri.resize(pairs.size() * RHO2_VALUES);
        int status = 0;
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = fermion_trace
                         ? mipt_cuda_rho2_subsystems_complex_fermion_f64(
                               tensor.data, n, flat_pairs.data(),
                               static_cast<int>(pairs.size()), workspace.rho_ri.data())
                         : mipt_cuda_rho2_subsystems_complex_f64(
                               tensor.data, n, flat_pairs.data(),
                               static_cast<int>(pairs.size()), workspace.rho_ri.data());
        }
        else
        {
            status = fermion_trace
                         ? mipt_cuda_rho2_subsystems_complex_fermion_f32(
                               tensor.data, n, flat_pairs.data(),
                               static_cast<int>(pairs.size()), workspace.rho_ri.data())
                         : mipt_cuda_rho2_subsystems_complex_f32(
                               tensor.data, n, flat_pairs.data(),
                               static_cast<int>(pairs.size()), workspace.rho_ri.data());
        }
        if (status != 0)
        {
            throw std::runtime_error("CUDA two-site RDM reduction failed; status=" +
                                     std::to_string(status));
        }
        return;
    }
#endif

    if (!dense)
    {
        dense_state_from_amplitudes(state, n, workspace.amplitude_state);
        rho2_subsystems_from_host_statevector(workspace.amplitude_state.data(), n,
                                              pairs, fermion_trace,
                                              workspace.rho_ri);
        return;
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        if (state.is_on_gpu())
        {
            workspace.host_state_f64.resize(dimension);
            state.to_host(workspace.host_state_f64.data(), dimension);
            rho2_subsystems_from_host_statevector(workspace.host_state_f64.data(), n,
                                                  pairs, fermion_trace,
                                                  workspace.rho_ri);
        }
        else
        {
            rho2_subsystems_from_host_statevector(
                reinterpret_cast<const std::complex<double> *>(tensor.data), n,
                pairs, fermion_trace, workspace.rho_ri);
        }
        return;
    }

    if (state.is_on_gpu())
    {
        workspace.host_state_f32.resize(dimension);
        state.to_host(workspace.host_state_f32.data(), dimension);
        rho2_subsystems_from_host_statevector(workspace.host_state_f32.data(), n,
                                              pairs, fermion_trace,
                                              workspace.rho_ri);
    }
    else
    {
        rho2_subsystems_from_host_statevector(
            reinterpret_cast<const std::complex<float> *>(tensor.data), n,
            pairs, fermion_trace, workspace.rho_ri);
    }
}

inline void run(int n,
                int periods,
                double p,
                int realizations,
                CircuitType circuit_type,
                const std::string &output_path)
{
    validate_args(n, periods, p, realizations, circuit_type);
    const bool fermion_trace = uses_fermionic_trace(circuit_type);
    const std::vector<Pair> pairs = all_unordered_pairs(n);
    const std::uint64_t pairs_per_realization =
        static_cast<std::uint64_t>(pairs.size());
    const std::uint64_t total_rows = pairs_per_realization *
                                     static_cast<std::uint64_t>(realizations);

    ensure_output_parent_directory(output_path);
    std::ofstream csv(output_path, std::ios::binary | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not create output CSV: " + output_path);
    }
    csv << "d,mi,mn\n";
    csv.flush();

    backend::print_backend_banner_once("dist_scaling.exe");
    std::cerr << "Two-site distance-scaling simulation\n"
              << "N=" << n
              << ", periods=" << periods
              << ", p=" << p
              << ", realizations=" << realizations
              << ", pairs_per_realization=" << pairs_per_realization
              << ", mode=" << circuit_type_name(circuit_type)
              << ", fermionic_trace_transpose=" << (fermion_trace ? 1 : 0)
              << ", output=" << output_path << '\n'
              << "pair_policy=unordered i<j; reversed pairs are observable-equivalent\n";

    CircuitWorkspace1D circuit_workspace;
    circuit_workspace.reserve(periods);
    RdmWorkspace rdm_workspace;
    std::vector<double> distances;
    distances.reserve(pairs.size());
    for (const Pair &pair : pairs)
    {
        distances.push_back(two_party_geometric_chord_distance(
            n, pair[0], pair[1]));
    }

    std::string csv_buffer;
    csv_buffer.reserve(std::max<std::size_t>(4096u, pairs.size() * 96u));
    std::string line;
    line.reserve(128);

    const std::string pause_file = pause_file_path();
    const auto progress_interval =
        std::chrono::milliseconds(progress_interval_ms());
    const auto start = std::chrono::steady_clock::now();
    auto last_report = start - progress_interval;
    double paused_seconds = 0.0;
    std::uint64_t processed = 0;
    std::uint64_t last_processed = 0;
    bool backend_reported = false;

    auto flush = [&]() {
        if (csv_buffer.empty())
        {
            return;
        }
        csv.write(csv_buffer.data(),
                  static_cast<std::streamsize>(csv_buffer.size()));
        csv_buffer.clear();
        csv.flush();
        if (!csv)
        {
            throw std::runtime_error("Failed while writing output CSV.");
        }
    };

    auto wait_if_paused = [&]() {
        if (pause_file.empty() || !file_exists(pause_file))
        {
            return;
        }
        flush();
        const auto pause_start = std::chrono::steady_clock::now();
        std::cerr << "\nPAUSED by host sentinel " << pause_file
                  << "; remove it to resume.\n";
        while (file_exists(pause_file))
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        const auto pause_end = std::chrono::steady_clock::now();
        const double duration =
            std::chrono::duration<double>(pause_end - pause_start).count();
        paused_seconds += duration;
        last_report += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            pause_end - pause_start);
        std::cerr << "RESUMED after " << format_duration(duration) << ".\n";
    };

    for (int realization = 0; realization < realizations; ++realization)
    {
        wait_if_paused();
        auto state = circuit_workspace.simulate(
            n, periods, p, circuit_type, "dist_scaling");
        if (!backend_reported)
        {
            std::cerr << "state_backend=" << (state.is_on_gpu() ? "gpu" : "host")
                      << ", precision="
                      << (state.get_precision() == cudaq::SimulationState::precision::fp64
                              ? "fp64"
                              : "fp32")
                      << ", rho2_backend=" << rho2_backend_name(state, n)
                      << '\n';
            backend_reported = true;
        }

        rho2_subsystems_from_cudaq_state(state, n, pairs, fermion_trace,
                                         rdm_workspace);
        if (rdm_workspace.rho_ri.size() != pairs.size() * RHO2_VALUES)
        {
            throw std::runtime_error("Internal two-site RDM payload-size mismatch.");
        }

        for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index)
        {
            const double *rho = rdm_workspace.rho_ri.data() +
                                pair_index * RHO2_VALUES;
            const PairMetrics metrics = two_party_metrics(rho, fermion_trace);

            line.clear();
            append_double(line, distances[pair_index]);
            line.push_back(',');
            append_double(line, metrics.mi);
            line.push_back(',');
            append_double(line, metrics.mn);
            line.push_back('\n');
            csv_buffer += line;
            ++processed;

            const auto now = std::chrono::steady_clock::now();
            if (progress_interval.count() == 0 ||
                now - last_report >= progress_interval ||
                processed == total_rows)
            {
                const double wall_elapsed =
                    std::chrono::duration<double>(now - start).count();
                const double active_elapsed =
                    std::max(1.0e-12, wall_elapsed - paused_seconds);
                const double average_rate =
                    static_cast<double>(processed) / active_elapsed;
                const double recent_elapsed = std::max(
                    1.0e-12,
                    std::chrono::duration<double>(now - last_report).count());
                const double recent_rate =
                    static_cast<double>(processed - last_processed) /
                    recent_elapsed;
                const double eta = average_rate > 0.0
                                       ? static_cast<double>(total_rows - processed) /
                                             average_rate
                                       : 0.0;
                std::cerr << '\r'
                          << "processed pairs " << processed << '/' << total_rows
                          << " | avg " << std::fixed << std::setprecision(2)
                          << average_rate << "/s"
                          << " | recent " << recent_rate << "/s"
                          << " | ETA " << format_duration(eta)
                          << "          " << std::flush;
                last_report = now;
                last_processed = processed;
            }
        }
        flush();
    }

    flush();
    const auto finish = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(finish - start).count();
    const double active_seconds = std::max(1.0e-12,
                                           wall_seconds - paused_seconds);
    std::cerr << "\nCompleted " << processed << " pair row(s) in "
              << format_duration(active_seconds) << " active time at "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(processed) / active_seconds
              << " pair(s)/s.\n";
}

inline void print_usage(const char *argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " [N = 10] [periods = 10] [p = 0.17] [realizations = 10]"
           " [circ_type = 0] [output.csv = auto]\n\n"
        << "Arguments:\n"
        << "  circ_type:\n";
    print_circuit_type_help(std::cerr, "    ");
    std::cerr
        << "  output.csv defaults to:\n"
        << "    csv/dist_scaling/<tag>/dist_scaling_<tag>_n_<N>_periods_<periods>_p_<p>_real_<realizations>.csv\n\n"
        << "Output CSV columns:\n"
        << "  d,mi,mn\n"
        << "where d is the two-party geometric-mean chord distance, mi is"
           " I(A:B), and mn is bipartite negativity (not logarithmic negativity).\n\n"
        << "Pair policy:\n"
        << "  Exactly one row is emitted for every unordered pair i<j. Reversing"
           " the retained-mode order does not create a distinct MI, negativity,"
           " or chord-distance observable.\n\n"
        << "Environment variables:\n"
        << "  MIPT_DIST_PROGRESS_MS=1000 controls progress updates (0 = every row).\n"
        << "  MIPT_DIST_PAUSE_FILE=path selects a pause sentinel; 0 disables it.\n"
        << "  MIPT_DIST_MPS_DENSE_MAX_QUBITS=16 bounds exact amplitude reconstruction"
           " when a tensor/MPS backend exposes no dense state tensor.\n";
}
} // namespace mipt::dist
