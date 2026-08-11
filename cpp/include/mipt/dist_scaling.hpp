#pragma once

// Online distance scaling of two- and three-party entanglement measures.
//
// One executable, two protocols selected by `k`:
//
//   k=2  every unordered site pair {i,j}: mutual information and bipartite
//        negativity, binned by ring separation.
//   k=3  every balanced site triangle {a,b,c}: the tripartite information
//        (TMI), the mean pairwise MI, the minimum one-vs-rest bipartite
//        negativity, and the genuine multipartite negativity (GMN), binned by
//        triangle geometry.
//
// Qubit circuits (circ_type 0, 1) report the ordinary-trace quantities only.
// Parity-preserving circuits (circ_type 2, 3, 4) additionally report the
// fermionic-trace versions -- fMI/fMN for k=2, fTMI/fGMN for k=3 -- so that
// the two conventions can be compared on the same trajectories.
//
// Both protocols write *aggregated* CSVs: one row per geometry carrying
// means, standard errors, and sample counts, rather than one row per
// (geometry, trajectory). A k=2 run that used to emit N(N-1)/2 x realizations
// rows now emits N/2 of them, which is what makes a 10^5-trajectory sweep a
// few kilobytes instead of a few gigabytes. `data_analysis.dist_scaling`
// reads both this format and the older row-level one.
//
// The GMN/fGMN semidefinite programs dominate the k=3 cost -- they are orders
// of magnitude slower than the trajectory that produced their density matrix
// -- so they use the same three defences as three-probe mode 4: a zero
// prefilter that skips the solve whenever a vanishing minimum bipartite
// negativity already certifies GMN=0, a schedule that evaluates only a
// content-independent subsample of records, and background batches that solve
// while the next trajectories run.

#include "mipt/analysis/fgmn.hpp"
#include "mipt/backend.hpp"
#include "mipt/circuit.hpp"
#include "mipt/dist_metrics.hpp"
#include "mipt/dist_scaling_csv.hpp"
#include "mipt/dist_scaling_resume.hpp"
#include "mipt/probed/geometry.hpp"
#include "mipt/util/pause.hpp"
#include "mipt/util/stats.hpp"
#include "mipt/util/text.hpp"
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
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace mipt::dist
{
constexpr int RHO2_DIM = 4;
constexpr std::size_t RHO2_VALUES = 2u * RHO2_DIM * RHO2_DIM;
using Pair = std::array<int, 2>;
using util::RunningStats;

// Every k=3 record is one 8x8 complex matrix; RHO3_VALUES is its interleaved
// length and the payload size of an SDP job.
using mipt::RHO3_DIM;
using mipt::RHO3_VALUES;
using mipt::Subsystem;

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

using util::compact_decimal;
using util::csv_quote;

inline std::string format_duration(double seconds)
{
    return util::format_duration(seconds, util::DurationStyle::Compact);
}

inline long dense_fallback_max_qubits()
{
    return env::integer("MIPT_DIST_MPS_DENSE_MAX_QUBITS", 16, 2, 30);
}

inline long progress_interval_ms()
{
    return env::integer("MIPT_DIST_PROGRESS_MS", 1000, 0, 60000);
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

template <typename Real>
inline void rho2_subsystems_both_from_host_statevector(
    const std::complex<Real> *statevector,
    int n,
    const std::vector<Pair> &pairs,
    std::vector<double> &rho_ri,
    std::vector<double> &fermion_rho_ri)
{
    rho2_subsystems_from_host_statevector(
        statevector, n, pairs, false, rho_ri);
    rho2_subsystems_from_host_statevector(
        statevector, n, pairs, true, fermion_rho_ri);
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
    std::vector<double> fermion_rho_ri;
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

inline const char *rho3_backend_name(cudaq::state &state, int n)
{
    const auto tensor = state.get_tensor();
    const std::size_t dimension = static_cast<std::size_t>(checked_pow2_u64(n));
    const bool dense = tensor.get_rank() == 1 &&
                       tensor.get_num_elements() >= dimension &&
                       tensor.data != nullptr;
#ifdef MIPT_ENABLE_CUDA_RHO
    if (dense && state.is_on_gpu())
    {
        return "CUDA batched rho3 reduction";
    }
#endif
    if (dense && state.is_on_gpu())
    {
        return "single device-to-host state copy + host rho3 reduction";
    }
    if (dense)
    {
        return "host rho3 reduction";
    }
    return "CUDA-Q amplitude reconstruction + host rho3 reduction";
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

inline void rho2_subsystems_both_from_cudaq_state(
    cudaq::state &state,
    int n,
    const std::vector<Pair> &pairs,
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
        workspace.fermion_rho_ri.resize(pairs.size() * RHO2_VALUES);
        int status = 0;
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = mipt_cuda_rho2_subsystems_complex_both_f64(
                tensor.data, n, flat_pairs.data(),
                static_cast<int>(pairs.size()), workspace.rho_ri.data(),
                workspace.fermion_rho_ri.data());
        }
        else
        {
            status = mipt_cuda_rho2_subsystems_complex_both_f32(
                tensor.data, n, flat_pairs.data(),
                static_cast<int>(pairs.size()), workspace.rho_ri.data(),
                workspace.fermion_rho_ri.data());
        }
        if (status != 0)
        {
            throw std::runtime_error(
                "CUDA ordinary+fermionic two-site RDM reduction failed; status=" +
                std::to_string(status));
        }
        return;
    }
#endif

    if (!dense)
    {
        dense_state_from_amplitudes(state, n, workspace.amplitude_state);
        rho2_subsystems_both_from_host_statevector(
            workspace.amplitude_state.data(), n, pairs, workspace.rho_ri,
            workspace.fermion_rho_ri);
        return;
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        if (state.is_on_gpu())
        {
            workspace.host_state_f64.resize(dimension);
            state.to_host(workspace.host_state_f64.data(), dimension);
            rho2_subsystems_both_from_host_statevector(
                workspace.host_state_f64.data(), n, pairs, workspace.rho_ri,
                workspace.fermion_rho_ri);
        }
        else
        {
            rho2_subsystems_both_from_host_statevector(
                reinterpret_cast<const std::complex<double> *>(tensor.data), n,
                pairs, workspace.rho_ri, workspace.fermion_rho_ri);
        }
        return;
    }

    if (state.is_on_gpu())
    {
        workspace.host_state_f32.resize(dimension);
        state.to_host(workspace.host_state_f32.data(), dimension);
        rho2_subsystems_both_from_host_statevector(
            workspace.host_state_f32.data(), n, pairs, workspace.rho_ri,
            workspace.fermion_rho_ri);
    }
    else
    {
        rho2_subsystems_both_from_host_statevector(
            reinterpret_cast<const std::complex<float> *>(tensor.data), n,
            pairs, workspace.rho_ri, workspace.fermion_rho_ri);
    }
}

struct SdpSettings
{
    bool enabled = true;
    int stride = 1;
    long samples_per_geometry = 0; // 0 = unlimited
    double zero_tolerance = 1.0e-10;
    std::size_t batch_size = 64;
    std::size_t pending_batches = 2;

    static SdpSettings from_environment()
    {
        SdpSettings settings;
        settings.enabled = env::boolean("MIPT_DIST_GMN", true);
        settings.stride =
            static_cast<int>(env::integer("MIPT_DIST_GMN_STRIDE", 1, 1, std::numeric_limits<int>::max()));
        settings.samples_per_geometry =
            env::integer("MIPT_DIST_GMN_SAMPLES_PER_GEOMETRY", 0, 0, std::numeric_limits<long>::max());
        settings.zero_tolerance = env::real("MIPT_DIST_GMN_ZERO_TOL", 1.0e-10, 0.0, 1.0);
        settings.batch_size =
            static_cast<std::size_t>(env::integer("MIPT_DIST_GMN_BATCH_RECORDS", 64, 1, 1000000));
        // Must not sit below the solver concurrency limit or it, rather than
        // the limiter, becomes the binding throttle.
        settings.pending_batches = static_cast<std::size_t>(env::integer(
            "MIPT_DIST_GMN_PENDING_BATCHES", 2 * analysis::default_sdp_worker_count(), 1, 64));
        return settings;
    }

    // Whether the record at index `record` of a bin gets an SDP evaluation.
    // Deliberately a pure function of the counter: selecting on the value --
    // for instance solving only the records whose bipartite negativity looks
    // large -- would bias the reported mean upwards.
    bool selects(std::uint64_t record, std::uint64_t already_requested) const
    {
        if (!enabled)
        {
            return false;
        }
        if (samples_per_geometry > 0 &&
            already_requested >= static_cast<std::uint64_t>(samples_per_geometry))
        {
            return false;
        }
        return record % static_cast<std::uint64_t>(stride) == 0;
    }
};

// Solves batches of GMN/fGMN programs on background threads while the
// simulation continues, folding each result back into the bin it came from.
//
// Results are collected on the calling thread only, so no bin is ever touched
// concurrently and the CSV writer can run between submissions.
class SdpBatchQueue
{
  public:
    SdpBatchQueue(std::vector<TripleBin> &bins, std::size_t batch_size, std::size_t max_pending_batches)
        : bins_(bins), batch_size_(std::max<std::size_t>(1, batch_size)),
          max_pending_batches_(std::max<std::size_t>(1, max_pending_batches))
    {
        batch_.reserve(batch_size_);
    }

    void submit(std::size_t bin_index, bool fermionic, const double *rho_ri)
    {
        Job job;
        job.bin = bin_index;
        job.fermionic = fermionic;
        std::copy(rho_ri, rho_ri + RHO3_VALUES, job.rho.begin());
        batch_.push_back(std::move(job));
        if (batch_.size() >= batch_size_)
        {
            dispatch();
        }
    }

    // Fold in whatever has already finished, without blocking. Called before
    // rewriting the CSV so a mid-run file is as current as it can cheaply be.
    void collect_ready()
    {
        while (!pending_.empty() &&
               pending_.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            collect_front();
        }
    }

    void finish()
    {
        if (!batch_.empty())
        {
            dispatch();
        }
        while (!pending_.empty())
        {
            collect_front();
        }
    }

    std::uint64_t solved() const
    {
        return solved_;
    }

    std::uint64_t queued() const
    {
        return queued_;
    }

  private:
    struct Job
    {
        std::size_t bin = 0;
        bool fermionic = false;
        std::array<double, RHO3_VALUES> rho{};
    };

    struct Result
    {
        std::size_t bin = 0;
        bool fermionic = false;
        double value = std::numeric_limits<double>::quiet_NaN();
    };

    void dispatch()
    {
        while (pending_.size() >= max_pending_batches_)
        {
            collect_front();
        }
        std::vector<Job> jobs;
        jobs.swap(batch_);
        batch_.reserve(batch_size_);
        queued_ += jobs.size();
        pending_.push_back(std::async(std::launch::async, [jobs = std::move(jobs)]() mutable {
            std::vector<Result> results;
            results.reserve(jobs.size());
            for (auto &job : jobs)
            {
                const double value = job.fermionic ? compute_fgmn_mosek_8x8_cpp(job.rho.data())
                                                   : compute_gmn_mosek_complex_8x8_cpp(job.rho.data());
                results.push_back({job.bin, job.fermionic, value});
            }
            return results;
        }));
    }

    void collect_front()
    {
        auto results = pending_.front().get();
        pending_.pop_front();
        for (const auto &result : results)
        {
            TripleBin &bin = bins_.at(result.bin);
            SdpStats &stats = result.fermionic ? bin.fgmn : bin.gmn;
            if (std::isfinite(result.value))
            {
                stats.value.add(std::max(0.0, result.value));
            }
            else
            {
                ++stats.solver_failures;
            }
            ++solved_;
        }
    }

    std::vector<TripleBin> &bins_;
    std::size_t batch_size_ = 1;
    std::size_t max_pending_batches_ = 1;
    std::vector<Job> batch_;
    std::deque<std::future<std::vector<Result>>> pending_;
    std::uint64_t solved_ = 0;
    std::uint64_t queued_ = 0;
};

// ---------------------------------------------------------------------------
// Progress reporting
// ---------------------------------------------------------------------------

class ProgressLine
{
  public:
    // `already_done` is what a resumed run inherited from its checkpoint. The
    // displayed counter stays absolute so the line shows progress through the
    // whole run, but rate and ETA are computed on this session's work only --
    // otherwise a resume reports a rate inflated by trajectories it never ran.
    ProgressLine(std::uint64_t total, const char *unit, std::uint64_t already_done = 0)
        : total_(total), unit_(unit), interval_(std::chrono::milliseconds(progress_interval_ms())),
          start_(std::chrono::steady_clock::now()), last_report_(start_ - interval_),
          baseline_(already_done), last_processed_(already_done)
    {
    }

    bool due(std::uint64_t processed) const
    {
        return interval_.count() == 0 || std::chrono::steady_clock::now() - last_report_ >= interval_ ||
               processed >= total_;
    }

    void report(std::uint64_t processed, double paused_seconds, const std::string &suffix)
    {
        const auto now = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(now - start_).count();
        const double active = std::max(1.0e-12, wall - paused_seconds);
        const double average_rate =
            static_cast<double>(processed - std::min(processed, baseline_)) / active;
        const double recent_elapsed =
            std::max(1.0e-12, std::chrono::duration<double>(now - last_report_).count());
        const double recent_rate = static_cast<double>(processed - last_processed_) / recent_elapsed;
        const double eta =
            average_rate > 0.0 ? static_cast<double>(total_ - std::min(processed, total_)) / average_rate : 0.0;
        std::cerr << '\r' << unit_ << ' ' << processed << '/' << total_ << " | avg " << std::fixed
                  << std::setprecision(2) << average_rate << "/s | recent " << recent_rate << "/s | ETA "
                  << format_duration(eta) << suffix << "          " << std::flush;
        last_report_ = now;
        last_processed_ = processed;
    }

    // Keep the next line from reporting a burst after a long pause.
    void absorb_pause(double seconds)
    {
        last_report_ += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(seconds));
    }

    double elapsed_seconds() const
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

  private:
    std::uint64_t total_ = 0;
    const char *unit_ = "records";
    std::chrono::milliseconds interval_{1000};
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_report_;
    std::uint64_t baseline_ = 0;
    std::uint64_t last_processed_ = 0;
};

// ---------------------------------------------------------------------------
// k=2: pairs
// ---------------------------------------------------------------------------

inline void run_pairs(const RunConfig &config)
{
    const int n = config.n;
    const bool include_fermionic = config.fermionic_outputs();
    const std::vector<Pair> pairs = all_unordered_pairs(n);
    std::vector<PairBin> bins = make_pair_bins(n);
    std::vector<std::size_t> bin_of_pair;
    bin_of_pair.reserve(pairs.size());
    for (const Pair &pair : pairs)
    {
        bin_of_pair.push_back(static_cast<std::size_t>(periodic_distance(pair[0], pair[1], n)) - 1u);
    }

    // The existing CSV is the checkpoint. Read it before anything is written,
    // so a refusal cannot have destroyed the run it refused.
    const resume::Report checkpoint =
        resume::enabled() ? resume::load_pairs(config, bins) : resume::Report{};
    const int start_realization = static_cast<int>(checkpoint.completed);

    const std::uint64_t total_records =
        static_cast<std::uint64_t>(pairs.size()) * static_cast<std::uint64_t>(config.realizations);
    std::uint64_t processed =
        static_cast<std::uint64_t>(pairs.size()) * static_cast<std::uint64_t>(start_realization);

    ensure_output_parent_directory(config.output_path);
    publish_csv(config.output_path, render_pair_csv(config, bins));

    backend::print_backend_banner_once("dist_scaling.exe");
    if (checkpoint.loaded)
    {
        std::cerr << "Resuming from " << config.output_path << ": "
                  << start_realization << '/' << config.realizations
                  << " trajectories already binned.\n";
        if (start_realization >= config.realizations)
        {
            std::cerr << config.output_path << " is already complete. Nothing to do.\n";
            return;
        }
    }
    std::cerr << "Two-site distance-scaling simulation\n"
              << "N=" << n
              << ", periods=" << config.periods
              << ", p=" << config.p
              << ", realizations=" << config.realizations
              << ", pairs_per_realization=" << pairs.size()
              << ", separation_bins=" << bins.size()
              << ", mode=" << circuit_type_name(config.type)
              << ", fermionic_outputs=" << (include_fermionic ? 1 : 0)
              << ", output=" << config.output_path << '\n'
              << "pair_policy=unordered i<j; reversed pairs are observable-equivalent\n";

    CircuitWorkspace1D circuit_workspace;
    circuit_workspace.reserve(config.periods);
    RdmWorkspace rdm_workspace;

    util::PauseSentinel pause_sentinel("MIPT_DIST_PAUSE_FILE");
    ProgressLine progress(total_records, "processed pairs", processed);
    bool backend_reported = false;

    auto flush = [&]() { publish_csv(config.output_path, render_pair_csv(config, bins)); };

    for (int realization = start_realization; realization < config.realizations; ++realization)
    {
        progress.absorb_pause(pause_sentinel.wait(flush));

        auto state = circuit_workspace.simulate(n, config.periods, config.p, config.type, "dist_scaling");
        if (!backend_reported)
        {
            std::cerr << "state_backend=" << (state.is_on_gpu() ? "gpu" : "host")
                      << ", precision="
                      << (state.get_precision() == cudaq::SimulationState::precision::fp64 ? "fp64" : "fp32")
                      << ", rho2_backend=" << rho2_backend_name(state, n) << '\n';
            backend_reported = true;
        }

        if (include_fermionic)
        {
            rho2_subsystems_both_from_cudaq_state(state, n, pairs, rdm_workspace);
        }
        else
        {
            rho2_subsystems_from_cudaq_state(state, n, pairs, false, rdm_workspace);
        }
        if (rdm_workspace.rho_ri.size() != pairs.size() * RHO2_VALUES)
        {
            throw std::runtime_error("Internal two-site RDM payload-size mismatch.");
        }
        if (include_fermionic && rdm_workspace.fermion_rho_ri.size() != pairs.size() * RHO2_VALUES)
        {
            throw std::runtime_error("Internal fermionic two-site RDM payload-size mismatch.");
        }

        for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index)
        {
            PairBin &bin = bins[bin_of_pair[pair_index]];
            const double *rho = rdm_workspace.rho_ri.data() + pair_index * RHO2_VALUES;
            const PairMetrics metrics = two_party_metrics(rho, false);
            bin.mi.add(metrics.mi);
            bin.mn.add(metrics.mn);
            if (metrics.mn > 1.0e-12)
            {
                ++bin.mn_positive;
            }
            if (include_fermionic)
            {
                const double *fermion_rho = rdm_workspace.fermion_rho_ri.data() + pair_index * RHO2_VALUES;
                const PairMetrics fermion_metrics = two_party_metrics(fermion_rho, true);
                bin.fmi.add(fermion_metrics.mi);
                bin.fmn.add(fermion_metrics.mn);
                if (fermion_metrics.mn > 1.0e-12)
                {
                    ++bin.fmn_positive;
                }
            }
            ++processed;
        }

        if (progress.due(processed))
        {
            flush();
            progress.report(processed, pause_sentinel.paused_seconds(), "");
        }
    }

    flush();
    const double active_seconds =
        std::max(1.0e-12, progress.elapsed_seconds() - pause_sentinel.paused_seconds());
    std::cerr << "\nCompleted " << processed << " pair record(s) over " << bins.size()
              << " separation bin(s) in " << format_duration(active_seconds) << " active time at "
              << std::fixed << std::setprecision(2) << static_cast<double>(processed) / active_seconds
              << " record(s)/s.\n";
}

// ---------------------------------------------------------------------------
// k=3: balanced triangles
// ---------------------------------------------------------------------------

// The triangles one trajectory measures, and the bin each belongs to.
//
// With subsampling this is redrawn per trajectory so that, over a run, every
// embedding of a geometry is visited with equal weight; without it the list is
// fixed and the caller builds it once.
inline void sample_trajectory_triangles(const std::vector<probed::ProbeGeometry> &geometries, long per_geometry,
                                        std::mt19937 &rng, std::vector<Subsystem> &subsystems,
                                        std::vector<std::size_t> &bin_of_subsystem)
{
    subsystems.clear();
    bin_of_subsystem.clear();
    for (std::size_t geometry_index = 0; geometry_index < geometries.size(); ++geometry_index)
    {
        const auto &embeddings = geometries[geometry_index].embeddings;
        if (per_geometry <= 0 || static_cast<std::size_t>(per_geometry) >= embeddings.size())
        {
            for (const auto &embedding : embeddings)
            {
                subsystems.push_back(embedding);
                bin_of_subsystem.push_back(geometry_index);
            }
            continue;
        }
        // Sampling without replacement inside one trajectory: the same
        // triangle twice would double-count one measurement, not add one.
        std::vector<std::size_t> order(embeddings.size());
        for (std::size_t i = 0; i < order.size(); ++i)
        {
            order[i] = i;
        }
        std::shuffle(order.begin(), order.end(), rng);
        for (long taken = 0; taken < per_geometry; ++taken)
        {
            subsystems.push_back(embeddings[order[static_cast<std::size_t>(taken)]]);
            bin_of_subsystem.push_back(geometry_index);
        }
    }
}

inline void run_triples(const RunConfig &config)
{
    const int n = config.n;
    const bool include_fermionic = config.fermionic_outputs();
    const auto sdp = SdpSettings::from_environment();
    const long per_geometry = embeddings_per_geometry();

    // The same enumeration and the same balance filter three-probe mode 4
    // uses, so a triangle admitted by one protocol is admitted by the other.
    const std::vector<probed::ProbeGeometry> geometries =
        probed::enumerate_three_probe_geometries(n, config.triangle_balance_cutoff);
    std::vector<TripleBin> bins = make_triple_bins(n, geometries);

    std::mt19937 rng{std::random_device{}()};
    std::vector<Subsystem> subsystems;
    std::vector<std::size_t> bin_of_subsystem;
    sample_trajectory_triangles(geometries, per_geometry, rng, subsystems, bin_of_subsystem);
    const bool resample_each_trajectory =
        per_geometry > 0 &&
        std::any_of(geometries.begin(), geometries.end(), [&](const probed::ProbeGeometry &geometry) {
            return geometry.embeddings.size() > static_cast<std::size_t>(per_geometry);
        });
    const std::size_t triangles_per_trajectory = subsystems.size();
    if (triangles_per_trajectory == 0)
    {
        throw std::runtime_error("No triangles were selected; check B_min and "
                                 "MIPT_DIST_EMBEDDINGS_PER_GEOMETRY.");
    }
    // The existing CSV is the checkpoint. Read it before anything is written,
    // so a refusal cannot have destroyed the run it refused.
    const resume::Report checkpoint =
        resume::enabled() ? resume::load_triples(config, bins, per_geometry)
                         : resume::Report{};
    const int start_realization = static_cast<int>(checkpoint.completed);

    const std::uint64_t total_records = static_cast<std::uint64_t>(triangles_per_trajectory) *
                                        static_cast<std::uint64_t>(config.realizations);
    std::uint64_t processed = static_cast<std::uint64_t>(triangles_per_trajectory) *
                              static_cast<std::uint64_t>(start_realization);

    ensure_output_parent_directory(config.output_path);
    publish_csv(config.output_path, render_triple_csv(config, bins));

    backend::print_backend_banner_once("dist_scaling.exe");
    if (checkpoint.loaded)
    {
        std::cerr << "Resuming from " << config.output_path << ": "
                  << start_realization << '/' << config.realizations
                  << " trajectories already binned.\n";
        const std::uint64_t reclaimed =
            checkpoint.reclaimed_gmn_slots + checkpoint.reclaimed_fgmn_slots;
        if (reclaimed > 0)
        {
            // Those records were submitted but never came back, so their values
            // are gone for good -- the density matrices they were computed from
            // are not stored. Returning their schedule slots makes everything
            // measured from here on correctly sampled, but it cannot undo the
            // skew already in the file: the lost records are exactly the ones
            // the zero prefilter did *not* settle, so what survives them
            // over-weights GMN=0.
            std::cerr << "  " << reclaimed
                      << " SDP solve(s) were still in flight when the run stopped. "
                         "Their schedule slots are returned, so sampling from here "
                         "on is unbiased.\n";
            const double lost = checkpoint.worst_lost_sdp_fraction;
            if (lost > 0.02)
            {
                std::cerr << "  WARNING: up to " << std::fixed << std::setprecision(1)
                          << 100.0 * lost
                          << "% of one geometry's requested GMN records were lost "
                             "with them. Those are the non-zero ones, so this "
                             "checkpoint's GMN mean is biased low until enough new "
                             "records dilute it.\n"
                          << std::defaultfloat;
            }
        }
        if (start_realization >= config.realizations)
        {
            std::cerr << config.output_path << " is already complete. Nothing to do.\n";
            return;
        }
    }
    std::cerr << "Three-site distance-scaling simulation\n"
              << "N=" << n
              << ", periods=" << config.periods
              << ", p=" << config.p
              << ", realizations=" << config.realizations
              << ", B_min=" << config.triangle_balance_cutoff
              << ", geometries=" << geometries.size()
              << ", embeddings_per_geometry=" << (per_geometry > 0 ? std::to_string(per_geometry) : "all")
              << ", triangles_per_realization=" << triangles_per_trajectory
              << ", mode=" << circuit_type_name(config.type)
              << ", fermionic_outputs=" << (include_fermionic ? 1 : 0)
              << ", output=" << config.output_path << '\n';
    if (sdp.enabled)
    {
        std::cerr << "gmn_schedule: stride=" << sdp.stride << ", samples_per_geometry="
                  << (sdp.samples_per_geometry > 0 ? std::to_string(sdp.samples_per_geometry) : "all")
                  << ", zero_tol=" << sdp.zero_tolerance << ", batch=" << sdp.batch_size
                  << ", pending_batches=" << sdp.pending_batches
                  << ", measures=" << (include_fermionic ? "GMN+fGMN" : "GMN") << '\n';
        // The SDP workers are what use the cores here, so keep MOSEK
        // single-threaded and bound how many solve at once. The bound scales
        // with the machine; see default_sdp_worker_count.
        env::set_if_unset("GMN_MOSEK_NUM_THREADS", "1");
        env::set_if_unset("FGMN_MAX_CONCURRENT_MOSEK",
                          analysis::default_sdp_worker_count_text().c_str());
        env::set_if_unset("GMN_MOSEK_TOL", analysis::DEFAULT_SDP_TOLERANCE_TEXT);
    }
    else
    {
        std::cerr << "gmn_schedule: disabled by MIPT_DIST_GMN=0; TMI and bipartite "
                     "negativity only\n";
    }

    SdpBatchQueue sdp_queue(bins, sdp.batch_size, sdp.pending_batches);
    CircuitWorkspace1D circuit_workspace;
    circuit_workspace.reserve(config.periods);

    std::vector<double> rho_ri(triangles_per_trajectory * RHO3_VALUES);
    std::vector<double> fermion_rho_ri(include_fermionic ? triangles_per_trajectory * RHO3_VALUES : 0u);

    util::PauseSentinel pause_sentinel("MIPT_DIST_PAUSE_FILE");
    ProgressLine progress(total_records, "processed triangles", processed);
    bool backend_reported = false;

    auto flush = [&]() {
        sdp_queue.collect_ready();
        publish_csv(config.output_path, render_triple_csv(config, bins));
    };

    for (int realization = start_realization; realization < config.realizations; ++realization)
    {
        progress.absorb_pause(pause_sentinel.wait(flush));

        if (resample_each_trajectory && realization > start_realization)
        {
            sample_trajectory_triangles(geometries, per_geometry, rng, subsystems, bin_of_subsystem);
        }

        auto state = circuit_workspace.simulate(n, config.periods, config.p, config.type, "dist_scaling");
        if (!backend_reported)
        {
            std::cerr << "state_backend=" << (state.is_on_gpu() ? "gpu" : "host")
                      << ", precision="
                      << (state.get_precision() == cudaq::SimulationState::precision::fp64 ? "fp64" : "fp32")
                      << ", rho3_backend=" << rho3_backend_name(state, n) << '\n';
            backend_reported = true;
        }

        cudaq_three_site_density_matrices(state, n, subsystems, rho_ri.data(), false);
        normalize_rho3_subsystems(rho_ri.data(), subsystems.size());
        if (include_fermionic)
        {
            cudaq_three_site_density_matrices(state, n, subsystems, fermion_rho_ri.data(), true);
            normalize_rho3_subsystems(fermion_rho_ri.data(), subsystems.size());
        }

        for (std::size_t index = 0; index < subsystems.size(); ++index)
        {
            TripleBin &bin = bins[bin_of_subsystem[index]];
            const double *rho = rho_ri.data() + index * RHO3_VALUES;

            const TripleMetrics metrics = three_party_metrics(rho, false);
            bin.tmi.add(metrics.tmi);
            bin.average_mi.add(metrics.average_mi);
            bin.joint_purity.add(metrics.joint_purity);
            bin.mean_single_purity.add(metrics.mean_single_purity);

            const double min_negativity = compute_min_bipartite_negativity_8x8_cpp(rho);
            if (!std::isfinite(min_negativity))
            {
                throw std::runtime_error("Three-site bipartite-negativity evaluation "
                                         "returned a non-finite value.");
            }
            bin.min_bipartite_negativity.add(std::max(0.0, min_negativity));

            const double *fermion_rho = nullptr;
            double min_fermionic_negativity = 0.0;
            if (include_fermionic)
            {
                fermion_rho = fermion_rho_ri.data() + index * RHO3_VALUES;
                const TripleMetrics fermion_metrics = three_party_metrics(fermion_rho, true);
                bin.ftmi.add(fermion_metrics.tmi);
                bin.faverage_mi.add(fermion_metrics.average_mi);
                bin.fjoint_purity.add(fermion_metrics.joint_purity);
                bin.fmean_single_purity.add(fermion_metrics.mean_single_purity);

                min_fermionic_negativity = compute_min_bipartite_fermionic_negativity_8x8_cpp(fermion_rho);
                if (!std::isfinite(min_fermionic_negativity))
                {
                    throw std::runtime_error("Three-site fermionic bipartite-negativity "
                                             "evaluation returned a non-finite value.");
                }
                min_fermionic_negativity = std::max(0.0, min_fermionic_negativity);
                bin.min_fermionic_bipartite_negativity.add(min_fermionic_negativity);
            }

            // GMN <= min_s N_s over the one-vs-rest cuts, so a vanishing
            // minimum bipartite negativity certifies GMN=0 and the solve is
            // skipped. The record still counts, which is what keeps the mean
            // over selected records unbiased.
            if (sdp.selects(bin.records, bin.gmn.requested))
            {
                ++bin.gmn.requested;
                if (min_negativity <= sdp.zero_tolerance)
                {
                    bin.gmn.value.add(0.0);
                    ++bin.gmn.zero_prefiltered;
                }
                else
                {
                    sdp_queue.submit(bin_of_subsystem[index], false, rho);
                }

                if (include_fermionic)
                {
                    ++bin.fgmn.requested;
                    if (min_fermionic_negativity <= sdp.zero_tolerance)
                    {
                        bin.fgmn.value.add(0.0);
                        ++bin.fgmn.zero_prefiltered;
                    }
                    else
                    {
                        sdp_queue.submit(bin_of_subsystem[index], true, fermion_rho);
                    }
                }
            }

            ++bin.records;
            ++processed;
        }

        if (progress.due(processed))
        {
            flush();
            std::string suffix = " | sdp " + std::to_string(sdp_queue.solved()) + "/" +
                                 std::to_string(sdp_queue.queued());
            progress.report(processed, pause_sentinel.paused_seconds(), suffix);
        }
    }

    sdp_queue.finish();
    publish_csv(config.output_path, render_triple_csv(config, bins));
    const double active_seconds =
        std::max(1.0e-12, progress.elapsed_seconds() - pause_sentinel.paused_seconds());
    std::cerr << "\nCompleted " << processed << " triangle record(s) over " << bins.size()
              << " geometry bin(s) and " << sdp_queue.solved() << " SDP solve(s) in "
              << format_duration(active_seconds) << " active time at " << std::fixed << std::setprecision(2)
              << static_cast<double>(processed) / active_seconds << " record(s)/s.\n";
}

inline void run(const RunConfig &config)
{
    validate_args(config);
    if (config.k == 3)
    {
        run_triples(config);
        return;
    }
    run_pairs(config);
}

inline void print_usage(const char *argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " [k = 2] [N = 10] [periods = 10] [p = 0.17] [realizations = 10]"
           " [circ_type = 0] [output.csv = auto] [B_min = 0.5]\n\n"
        << "Arguments:\n"
        << "  k          2 = every unordered site pair; 3 = every balanced site triangle.\n"
        << "  B_min      k=3 only: minimum CFT chord balance B = l_min/l_max a triangle\n"
        << "             must reach to be simulated. Identical filter and enumeration to\n"
        << "             mipt_probed.exe probes=3 mode=4. Ignored for k=2.\n"
        << "  circ_type:\n";
    print_circuit_type_help(std::cerr, "    ");
    std::cerr
        << "  output.csv defaults to:\n"
        << "    k=2: csv/dist_scaling/<tag>/dist_scaling_<tag>_n_<N>_periods_<periods>_p_<p>_real_<realizations>.csv\n"
        << "    k=3: csv/dist_scaling/<tag>/dist_scaling3_<tag>_n_<N>_periods_<periods>_p_<p>_real_<realizations>_b_<B_min>.csv\n\n"
        << "Reported measures:\n"
        << "  k=2  mi, mn      mutual information and bipartite negativity per pair\n"
        << "  k=3  tmi, gmn    tripartite information and genuine multipartite negativity\n"
        << "       average_mi, min_bipneg, purities as supporting columns\n"
        << "  circ_type 0 or 1 (qubit ensembles) report those only. circ_type 2, 3, or 4\n"
        << "  (parity-preserving ensembles) additionally report the fermionic-trace\n"
        << "  versions fmi/fmn and ftmi/fgmn on the same trajectories. For qRPPU\n"
        << "  (circ_type 4) the circuit carries no Jordan-Wigner strings, so its\n"
        << "  fermionic columns are a comparison observable rather than the fermionic\n"
        << "  quantity itself.\n\n"
        << "Output CSV:\n"
        << "  One aggregated row per geometry -- separation for k=2, triangle for k=3 --\n"
        << "  carrying <metric>_mean, <metric>_stderr, and <metric>_samples, preceded by\n"
        << "  the metadata block N,circ_type,circuit_name,realizations,p,periods,k,\n"
        << "  entropy_units. `d` is the effective chord distance: the chord length for\n"
        << "  k=2 and the geometric mean of the triangle's three chords for k=3. All\n"
        << "  entropies are in bits. Negativities are negativity, not logarithmic\n"
        << "  negativity. The file is rewritten as the run proceeds, so an interrupted\n"
        << "  run leaves a complete file for the trajectories it finished -- and\n"
        << "  re-issuing the identical command continues from it rather than starting\n"
        << "  over, since that file is the checkpoint. MIPT_DIST_RESUME=0 discards it.\n\n"
        << "Sampling:\n"
        << "  Every geometry is measured inside the same trajectory, so bins at one\n"
        << "  realization are correlated; the reported standard errors treat records as\n"
        << "  independent and are therefore optimistic for cross-bin comparisons.\n\n"
        << "Environment variables:\n"
        << "  MIPT_DIST_PROGRESS_MS=1000 controls progress updates and how often the CSV\n"
        << "    is republished (0 = every trajectory).\n"
        << "  MIPT_PAUSE_FILE=path or MIPT_DIST_PAUSE_FILE=path selects the pause sentinel;\n"
        << "    0 disables it. Default: PAUSE_MIPT.\n"
        << "  MIPT_DIST_MPS_DENSE_MAX_QUBITS=16 bounds exact amplitude reconstruction"
           " when a tensor/MPS backend exposes no dense state tensor.\n"
        << "  k=3 only:\n"
        << "  MIPT_DIST_EMBEDDINGS_PER_GEOMETRY=1 triangles measured per geometry per\n"
        << "    trajectory (0 = every lattice embedding).\n"
        << "  MIPT_DIST_GMN=1 set to 0 to skip every GMN/fGMN solve.\n"
        << "  MIPT_DIST_GMN_STRIDE=1 evaluate one record in this many, per geometry.\n"
        << "  MIPT_DIST_GMN_SAMPLES_PER_GEOMETRY=0 cap on evaluated records per geometry\n"
        << "    (0 = uncapped). Counts records selected, including the ones the zero\n"
        << "    prefilter settles without a solve, so the mean stays unbiased.\n"
        << "  MIPT_DIST_GMN_ZERO_TOL=1e-10 minimum bipartite negativity below which\n"
        << "    GMN is taken to be exactly zero.\n"
        << "  MIPT_DIST_GMN_BATCH_RECORDS=64 solves per background batch.\n"
        << "  MIPT_DIST_GMN_PENDING_BATCHES batches allowed in flight (default: twice\n"
        << "    the solver concurrency below).\n"
        << "  FGMN_MAX_CONCURRENT_MOSEK concurrent GMN/fGMN solves (default: cores/3,\n"
        << "    clamped to [2,8]). The SDP dominates k=3, so this is the main speed knob.\n"
        << "  GMN_MOSEK_TOL=1e-5 interior-point tolerance for the SDP. 1e-5 is 1.3x faster\n"
        << "    than MOSEK's default at the same accuracy; 1e-4 loses three digits.\n";
}
} // namespace mipt::dist
