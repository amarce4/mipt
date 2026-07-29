#pragma once

// The probe simulation loop.
//
// All five modes share the same skeleton: for each p, sample `realizations`
// monitored trajectories, and for each trajectory advance the circuit to a
// series of readout times and measure the references. They differ only in when
// the references attach and what is recorded, which is what the three
// run_*_trajectory functions below express.
//
// Layer sampling is the expensive CPU-side step, so two circuit workspaces are
// alternated: while one trajectory runs on the GPU, the next one's layers are
// prepared on a background thread.

#include "mipt/env.hpp"
#include "mipt/probed.hpp"
#include "mipt/probed/config.hpp"
#include "mipt/probed/kernels.hpp"
#include "mipt/probed/progress.hpp"
#include "mipt/probed/sample.hpp"
#include "mipt/probed/sdp.hpp"
#include "mipt/types.hpp"
#include "mipt/util/pause.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace mipt::probed
{

// Tunables for the deferred GMN/fGMN evaluation in three-probe mode 4.
struct SdpSettings
{
    int stride = 1;
    int sample_limit = 0;
    double zero_tolerance = 1.0e-10;
    std::size_t batch_size = 64;
    std::size_t pending_batches = 2;

    static SdpSettings from_environment(int realizations)
    {
        SdpSettings settings;
        settings.stride =
            static_cast<int>(env::integer("MIPT_PROBED_GMN_STRIDE", 1, 1, std::numeric_limits<int>::max()));
        settings.sample_limit =
            static_cast<int>(env::integer("MIPT_PROBED_GMN_SAMPLES_PER_BIN", 0, 0, realizations));
        settings.zero_tolerance = env::real("MIPT_PROBED_GMN_ZERO_TOL", 1.0e-10, 0.0, 1.0);
        settings.batch_size =
            static_cast<std::size_t>(env::integer("MIPT_PROBED_GMN_BATCH_RECORDS", 64, 1, 1000000));
        settings.pending_batches =
            static_cast<std::size_t>(env::integer("MIPT_PROBED_GMN_PENDING_BATCHES", 2, 1, 64));
        return settings;
    }
};

namespace detail
{

// Mode 4: equilibrate, attach the references at the sampled geometry, then
// read out along tau. Three-probe runs also queue each RDM for an SDP solve.
inline void run_mode4_trajectory(const ProbeRunConfig &config, probed::CircuitWorkspace1D &workspace,
                                 cudaq::state &state, std::vector<Aggregate> &aggregates,
                                 const std::vector<int> &active_sites, std::size_t geometry_index,
                                 std::size_t realization_index, const SdpSettings &sdp,
                                 const std::vector<std::vector<unsigned char>> &gmn_selected, SdpBatchQueue *sdp_queue)
{
    const int n = config.n;
    if (config.t_eq > 0)
    {
        workspace.advance(state, 0, config.t_eq);
    }
    int current_t = config.t_eq;
    state = attach_references(state, n, active_sites);

    const std::size_t distance_offset = geometry_index * config.times.size();
    for (std::size_t sample_index = 0; sample_index < config.times.size(); ++sample_index)
    {
        const int target_t = config.t_eq + config.times[sample_index];
        if (target_t > current_t)
        {
            workspace.advance(state, current_t, target_t - current_t);
            current_t = target_t;
        }
        const std::size_t aggregate_index = distance_offset + sample_index;
        if (config.probes == 2)
        {
            aggregates[aggregate_index].add(measure_sample(state, config.probes, n, config.type));
            continue;
        }

        auto measurement = measure_three_probe_sample(state, n, config.type);
        auto &aggregate = aggregates[aggregate_index];
        aggregate.add(measurement.sample);
        const bool selected = gmn_selected[geometry_index][realization_index] != 0;
        if (selected && config.times[sample_index] % sdp.stride == 0)
        {
            ++aggregate.gmn_requested;
            if (measurement.sample.min_bipartite_negativity <= sdp.zero_tolerance)
            {
                // A vanishing minimum bipartite negativity certifies GMN=0, so
                // the expensive SDP can be skipped for this sample.
                aggregate.gmn.add(0.0);
                ++aggregate.gmn_zero_prefiltered;
            }
            else
            {
                sdp_queue->submit({aggregate_index, std::move(measurement.rho)});
            }
        }
    }
}

// Mode 3: attach one reference at tau1=4N and the second delta_t later, then
// follow their correlation for tau=0..8N.
inline void run_mode3_trajectory(const ProbeRunConfig &config, probed::CircuitWorkspace1D &workspace,
                                 cudaq::state &state, std::vector<Aggregate> &aggregates)
{
    const int n = config.n;
    workspace.advance(state, 0, 4 * n);
    int current_t = 4 * n;
    state = attach_reference(state, n, config.sites[0], 0);
    if (config.delta_t > 0)
    {
        workspace.advance(state, current_t, config.delta_t);
        current_t += config.delta_t;
    }
    state = attach_reference(state, n, config.sites[1], 1);

    for (std::size_t sample_index = 0; sample_index < config.times.size(); ++sample_index)
    {
        const int target_t = config.t0 + config.times[sample_index];
        if (target_t > current_t)
        {
            workspace.advance(state, current_t, target_t - current_t);
            current_t = target_t;
        }
        aggregates[sample_index].add(measure_sample(state, config.probes, n, config.type));
    }
}

// Modes 0, 1, and 2: advance to each readout time and measure. Multi-probe
// runs attach their references at t0=2N first; one-probe runs are already
// entangled from t=0.
inline void run_sampling_trajectory(const ProbeRunConfig &config, probed::CircuitWorkspace1D &workspace,
                                    cudaq::state &state, std::vector<Aggregate> &aggregates, int current_t)
{
    const int n = config.n;
    if ((config.mode == 0 || config.mode == 1) && config.probes > 1)
    {
        workspace.advance(state, 0, 2 * n);
        current_t = 2 * n;
        state = attach_references(state, n, config.sites);
    }

    for (std::size_t sample_index = 0; sample_index < config.times.size(); ++sample_index)
    {
        const int target_t = config.times[sample_index];
        if (target_t > current_t)
        {
            workspace.advance(state, current_t, target_t - current_t);
            current_t = target_t;
        }
        if (config.parity_encoding)
        {
            aggregates[sample_index].add(measure_parity_encoded_probe(state, n));
        }
        else if (config.mode == 2 || config.mode == 1 || config.probes != 1 || target_t > 0)
        {
            aggregates[sample_index].add(measure_sample(state, config.probes, n, config.type));
        }
        else
        {
            // At t=0 a single reference is maximally entangled by construction.
            aggregates[sample_index].add({std::log(2.0), 0.0, 0.0, 0.0, 0.0, 0.0});
        }
    }
}

// Which sites the references attach to for this trajectory. Mode 4 re-draws
// them per trajectory so every geometry is sampled uniformly over the ring.
inline std::vector<int> sample_trajectory_sites(const ProbeRunConfig &config, std::size_t geometry_index, int distance,
                                                std::mt19937 &rng)
{
    if (config.mode != 4)
    {
        return {};
    }
    if (config.probes == 2)
    {
        std::uniform_int_distribution<int> origin_distribution(0, config.n - 1);
        const int origin = origin_distribution(rng);
        return {origin, (origin + distance) % config.n};
    }
    const auto &embeddings = config.geometries.at(geometry_index).embeddings;
    std::uniform_int_distribution<std::size_t> embedding_distribution(0, embeddings.size() - 1);
    const auto &embedding = embeddings[embedding_distribution(rng)];
    return {embedding.begin(), embedding.end()};
}

// Choose, per geometry, which realizations get an SDP evaluation when
// MIPT_PROBED_GMN_SAMPLES_PER_BIN caps the count.
inline std::vector<std::vector<unsigned char>> select_gmn_samples(const ProbeRunConfig &config,
                                                                 const SdpSettings &sdp, std::mt19937 &rng)
{
    std::vector<std::vector<unsigned char>> selected(
        static_cast<std::size_t>(config.distance_count),
        std::vector<unsigned char>(static_cast<std::size_t>(config.realizations), 1u));
    if (sdp.sample_limit > 0 && sdp.sample_limit < config.realizations)
    {
        for (auto &selection : selected)
        {
            std::fill(selection.begin(), selection.end(), 0u);
            std::fill_n(selection.begin(), static_cast<std::size_t>(sdp.sample_limit), 1u);
            std::shuffle(selection.begin(), selection.end(), rng);
        }
    }
    return selected;
}

// Fold one p point's accumulators into finished CSV rows.
inline void append_rows(const ProbeRunConfig &config, double p, const std::vector<Aggregate> &aggregates,
                        std::vector<OutputRow> &rows)
{
    const bool pairs = config.mode == 4 && config.probes == 2;
    const bool triangles = config.mode == 4 && config.probes == 3;
    for (std::size_t distance_index = 0; distance_index < static_cast<std::size_t>(config.distance_count);
         ++distance_index)
    {
        for (std::size_t i = 0; i < config.times.size(); ++i)
        {
            OutputRow row;
            row.p = p;
            row.t = config.times[i];
            row.t0 = config.t0;
            row.distance = pairs ? static_cast<int>(distance_index) + 1 : -1;
            row.geometry = triangles ? static_cast<int>(distance_index) : -1;
            if (triangles)
            {
                row.distances = config.geometries[distance_index].distances;
                row.triangle_balance = config.geometries[distance_index].balance;
                row.embedding_count = config.geometries[distance_index].embeddings.size();
            }
            row.stats = aggregates[distance_index * config.times.size() + i];
            rows.push_back(std::move(row));
        }
    }
}

} // namespace detail

// Run every p point of one configuration and return the finished rows.
inline std::vector<OutputRow> simulate(const ProbeRunConfig &config, ProgressReporter &progress)
{
    const int n = config.n;
    const int t_max = config.t_max;
    const auto sdp = SdpSettings::from_environment(config.realizations);

    std::vector<OutputRow> rows;
    rows.reserve(config.p_values.size() * config.times.size() * static_cast<std::size_t>(config.distance_count));
    std::mt19937 origin_rng{std::random_device{}()};
    std::uint64_t completed = 0;
    util::PauseSentinel pause_sentinel;

    for (std::size_t p_index = 0; p_index < config.p_values.size(); ++p_index)
    {
        progress.begin_point(completed);
        const double p = config.p_values[p_index];
        std::vector<Aggregate> aggregates(config.times.size() * static_cast<std::size_t>(config.distance_count));

        std::vector<std::vector<unsigned char>> gmn_selected;
        std::unique_ptr<SdpBatchQueue> sdp_queue;
        if (config.mode == 4 && config.probes == 3)
        {
            // The SDP workers are what use the cores here, so keep MOSEK
            // single-threaded and bound how many solve at once.
            env::set_if_unset("GMN_MOSEK_NUM_THREADS", "1");
            env::set_if_unset("FGMN_MAX_CONCURRENT_MOSEK", "2");
            gmn_selected = detail::select_gmn_samples(config, sdp, origin_rng);
            sdp_queue = std::make_unique<SdpBatchQueue>(aggregates, uses_fermionic_trace(config.type), sdp.batch_size,
                                                        sdp.pending_batches);
        }

        std::array<probed::CircuitWorkspace1D, 2> workspaces;
        for (auto &workspace : workspaces)
        {
            workspace.reserve(t_max);
        }
        std::array<std::future<void>, 2> jobs;
        auto prepare = [&](probed::CircuitWorkspace1D &workspace) {
            if (config.mode == 1 && config.probes == 1)
            {
                workspace.prepare_quench(n, t_max, 2 * n, p, config.type);
            }
            else
            {
                workspace.prepare(n, t_max, p, config.type);
            }
        };
        if (t_max > 0)
        {
            prepare(workspaces[0]);
        }

        const std::uint64_t trajectories_at_p =
            static_cast<std::uint64_t>(config.realizations) * config.distance_count;
        for (std::uint64_t trajectory = 0; trajectory < trajectories_at_p; ++trajectory)
        {
            // Safe checkpoint: between trajectories, before the next state is built.
            pause_sentinel.wait();

            const std::size_t current = static_cast<std::size_t>(trajectory % 2);
            const std::size_t next = static_cast<std::size_t>((trajectory + 1) % 2);
            if (t_max > 0 && trajectory > 0)
            {
                if (config.prefetch)
                {
                    jobs[current].get();
                }
                else
                {
                    prepare(workspaces[current]);
                }
            }
            if (t_max > 0 && config.prefetch && trajectory + 1 < trajectories_at_p)
            {
                jobs[next] =
                    std::async(std::launch::async, [&workspace = workspaces[next], &prepare]() { prepare(workspace); });
            }

            const std::size_t geometry_index =
                config.mode == 4 ? static_cast<std::size_t>(trajectory / static_cast<std::uint64_t>(config.realizations))
                                 : 0;
            const std::size_t realization_index =
                config.mode == 4 ? static_cast<std::size_t>(trajectory % static_cast<std::uint64_t>(config.realizations))
                                 : 0;
            const int distance =
                config.mode == 4 && config.probes == 2 ? static_cast<int>(geometry_index) + 1 : -1;
            const auto trajectory_sites = detail::sample_trajectory_sites(config, geometry_index, distance, origin_rng);
            const auto &active_sites = config.mode == 4 ? trajectory_sites : config.sites;

            const bool attach_at_start = config.mode == 2 || config.probes == 1;
            auto &workspace = workspaces[current];
            auto state = config.parity_encoding
                             ? workspace.simulate_parity_encoded_probe(n, active_sites.front())
                             : initialize_state(n, active_sites, attach_at_start);
            const int current_t = config.parity_encoding ? t_max : 0;

            if (config.mode == 4)
            {
                detail::run_mode4_trajectory(config, workspace, state, aggregates, active_sites, geometry_index,
                                             realization_index, sdp, gmn_selected, sdp_queue.get());
            }
            else if (config.mode == 3)
            {
                detail::run_mode3_trajectory(config, workspace, state, aggregates);
            }
            else
            {
                detail::run_sampling_trajectory(config, workspace, state, aggregates, current_t);
            }

            ++completed;
            progress.update(completed, p, p_index + 1, config.p_values.size(),
                            completed == config.total_trajectories);
        }

        if (sdp_queue)
        {
            sdp_queue->finish();
        }
        detail::append_rows(config, p, aggregates, rows);
    }
    return rows;
}

} // namespace mipt::probed
