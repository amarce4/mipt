#pragma once

// The probe simulation loop.
//
// All six modes share the same skeleton: for each p, sample `realizations`
// monitored trajectories, and for each trajectory advance the circuit to a
// series of readout times and measure the references. They differ only in when
// the references attach and what is recorded, which is what the four
// run_*_trajectory functions below express.
//
// Layer sampling is the expensive CPU-side step, so two circuit workspaces are
// alternated: while one trajectory runs on the GPU, the next one's layers are
// prepared on a background thread.

#include "mipt/env.hpp"
#include "mipt/probed.hpp"
#include "mipt/probed/config.hpp"
#include "mipt/probed/front.hpp"
#include "mipt/probed/kernels.hpp"
#include "mipt/probed/progress.hpp"
#include "mipt/probed/sample.hpp"
#include "mipt/probed/sdp.hpp"
#include "mipt/types.hpp"
#include "mipt/util/crash_report.hpp"
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
        // Must not sit below the solver concurrency limit or it, rather than
        // the limiter, becomes the binding throttle and the extra workers never
        // get a batch to work on.
        settings.pending_batches = static_cast<std::size_t>(env::integer(
            "MIPT_PROBED_GMN_PENDING_BATCHES", 2 * analysis::default_sdp_worker_count(), 1, 64));
        return settings;
    }
};

namespace detail
{

// Mode 4: equilibrate, attach the references at the sampled geometry, then
// read out along tau. Three-probe runs also queue each RDM for an SDP solve.
inline void run_mode4_trajectory(const ProbeRunConfig &config, probed::CircuitWorkspace1D &workspace,
                                 cudaq::state &state, std::vector<Aggregate> &aggregates,
                                 std::vector<Aggregate> *batch_aggregates, const std::vector<int> &active_sites,
                                 std::size_t geometry_index, std::size_t realization_index, const SdpSettings &sdp,
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
            const Sample sample = measure_sample(state, config.probes, n, config.type);
            aggregates[aggregate_index].add(sample);
            if (batch_aggregates != nullptr)
            {
                (*batch_aggregates)[aggregate_index].add(sample);
            }
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

// Mode 5: equilibrate, inject one probe at a random site, then watch the
// classical and entanglement signals spread across the receiver-block grid.
//
// Unlike mode 4 this measures every geometry inside one trajectory: the blocks
// are read out of a single evolving state, so the whole (width, distance) grid
// costs one circuit and the trajectory count is just `realizations`.
inline void run_mode5_trajectory(const ProbeRunConfig &config, probed::CircuitWorkspace1D &workspace,
                                 cudaq::state &state, std::vector<Aggregate> &aggregates, int origin,
                                 std::vector<Sample> &scratch, std::vector<unsigned char> &filled)
{
    const int n = config.n;
    if (config.t_eq > 0)
    {
        workspace.advance(state, 0, config.t_eq);
    }
    int current_t = config.t_eq;
    // Resets x0 and Bell-entangles it with the reference. The reference is
    // never touched by a later layer, so it holds exactly what the measurement
    // record has not destroyed.
    state = attach_references(state, n, {origin});

    const bool fermionic = uses_fermionic_trace(config.type);
    for (std::size_t sample_index = 0; sample_index < config.times.size(); ++sample_index)
    {
        const int target_t = config.t_eq + config.times[sample_index];
        if (target_t > current_t)
        {
            workspace.advance(state, current_t, target_t - current_t);
            current_t = target_t;
        }
        measure_receiver_blocks(state, n, origin, config.receivers, fermionic, scratch, filled);
        for (std::size_t bin = 0; bin < scratch.size(); ++bin)
        {
            if (filled[bin])
            {
                aggregates[bin * config.times.size() + sample_index].add(scratch[bin]);
            }
        }
    }
}

// Mode 5, two probes: the staggered-insertion protocol.
//
// A is Bell-entangled with x0 at t_eq, B with x0+r a further `delay` timesteps
// later, and the pair is then watched for s=0..s_max. Both ancillas sit outside
// the circuit, so a system-only unitary acts as the identity on rho_AB and
// cannot correlate them at all; everything measured here is generated by the
// intervening measurement record.
//
// This cannot be swept inside one trajectory the way the one-probe grid is:
// attaching a probe resets its partner site, so each (separation, delay) is a
// different circuit history and needs its own trajectories. `bin` is fixed for
// the whole call and the caller multiplies the workload accordingly.
inline void run_mode5_stagger_trajectory(const ProbeRunConfig &config, probed::CircuitWorkspace1D &workspace,
                                         cudaq::state &state, std::vector<Aggregate> &aggregates,
                                         const std::vector<int> &sites, int delay, std::size_t bin)
{
    const int n = config.n;
    if (config.t_eq > 0)
    {
        workspace.advance(state, 0, config.t_eq);
    }
    int current_t = config.t_eq;
    state = attach_reference(state, n, sites[0], 0);
    if (delay > 0)
    {
        workspace.advance(state, current_t, delay);
        current_t += delay;
    }
    state = attach_reference(state, n, sites[1], 1);

    const bool fermionic = uses_fermionic_trace(config.type);
    const int t_second_insertion = config.t_eq + delay;
    for (std::size_t sample_index = 0; sample_index < config.times.size(); ++sample_index)
    {
        const int target_t = t_second_insertion + config.times[sample_index];
        if (target_t > current_t)
        {
            workspace.advance(state, current_t, target_t - current_t);
            current_t = target_t;
        }
        aggregates[bin * config.times.size() + sample_index].add(
            measure_stagger_sample(state, n, fermionic, config.optimize_classical));
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
//
// A one-probe encoder needs no step of its own here: the prepared history
// carries p=0 through the first `t_encode` layers, so the advance below runs
// straight through it, and `t_readout_origin` puts a readout time of zero at
// its end.
inline void run_sampling_trajectory(const ProbeRunConfig &config, probed::CircuitWorkspace1D &workspace,
                                    cudaq::state &state, std::vector<Aggregate> &aggregates,
                                    std::vector<Aggregate> *batch_aggregates, int current_t)
{
    const int n = config.n;
    // The batch replica sees exactly the samples the global accumulator sees,
    // so the two can never disagree about what was measured.
    const auto record = [&](std::size_t index, const Sample &sample) {
        aggregates[index].add(sample);
        if (batch_aggregates != nullptr)
        {
            (*batch_aggregates)[index].add(sample);
        }
    };
    if ((config.mode == 0 || config.mode == 1) && config.probes > 1)
    {
        workspace.advance(state, 0, 2 * n);
        current_t = 2 * n;
        state = attach_references(state, n, config.sites);
    }

    for (std::size_t sample_index = 0; sample_index < config.times.size(); ++sample_index)
    {
        const int target_t = config.t_readout_origin + config.times[sample_index];
        if (target_t > current_t)
        {
            workspace.advance(state, current_t, target_t - current_t);
            current_t = target_t;
        }
        if (config.parity_encoding)
        {
            record(sample_index, measure_parity_encoded_probe(state, n));
        }
        else if (config.mode == 2 || config.mode == 1 || config.probes != 1 || target_t > 0)
        {
            record(sample_index, measure_sample(state, config.probes, n, config.type));
        }
        else
        {
            // Before any layer runs a single reference is maximally entangled by
            // construction. Only an encoder-free run (MIPT_PROBED_T_ENCODE=0
            // with t_min=0) reaches this; otherwise the encoder has already run.
            record(sample_index, {std::log(2.0), 0.0, 0.0, 0.0, 0.0, 0.0});
        }
    }
}

// Which sites the references attach to for this trajectory. Modes 4 and 5
// re-draw them per trajectory so every geometry is sampled uniformly over the
// ring; one-probe mode 5 needs only the injection site x0.
//
// A random origin also removes the need to average the two ring directions
// explicitly: on a periodic chain x0+r and x0-r are the same geometry seen from
// a different origin, and the origin is uniform.
inline std::vector<int> sample_trajectory_sites(const ProbeRunConfig &config, std::size_t geometry_index, int distance,
                                                std::mt19937 &rng)
{
    if (config.mode != 4 && config.mode != 5)
    {
        return {};
    }
    std::uniform_int_distribution<int> origin_distribution(0, config.n - 1);
    if (config.mode == 5)
    {
        const int origin = origin_distribution(rng);
        if (config.probes == 1)
        {
            return {origin};
        }
        const int separation = config.staggers.separation_at(geometry_index);
        return {origin, (origin + separation) % config.n};
    }
    if (config.probes == 2)
    {
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
//
// `batch_id`/`batch_realizations` are -1/0 for the ordinary rows and identify
// the replica for a batch-mean row; the geometry of a row is otherwise
// identical either way, which is what keeps the two files describing the same
// grid.
inline void append_rows(const ProbeRunConfig &config, double p, const std::vector<Aggregate> &aggregates,
                        std::vector<OutputRow> &rows, int batch_id = -1, int batch_realizations = 0)
{
    const bool pairs = config.mode == 4 && config.probes == 2;
    const bool triangles = config.mode == 4 && config.probes == 3;
    const bool fronts = config.mode == 5 && config.probes == 1;
    const bool stagger = config.mode == 5 && config.probes == 2;
    for (std::size_t distance_index = 0; distance_index < static_cast<std::size_t>(config.bin_count);
         ++distance_index)
    {
        for (std::size_t i = 0; i < config.times.size(); ++i)
        {
            const Aggregate &stats = aggregates[distance_index * config.times.size() + i];
            // Mode 5's grid is rectangular but not every cell is admissible:
            // the widest blocks run out of ring first. Those bins were never
            // filled, and an empty row would break the completeness check in
            // write_csv as well as confuse the analysis.
            if (fronts && stats.sq.count == 0)
            {
                continue;
            }
            OutputRow row;
            row.p = p;
            row.t = config.times[i];
            row.t0 = config.t0;
            row.batch_id = batch_id;
            row.batch_realizations = batch_realizations;
            row.width = fronts ? config.receivers.width_at(distance_index) : -1;
            row.distance = pairs     ? static_cast<int>(distance_index) + 1
                           : fronts  ? config.receivers.distance_at(distance_index)
                           : stagger ? config.staggers.separation_at(distance_index)
                                     : -1;
            row.delay = stagger ? config.staggers.delay_at(distance_index) : -1;
            row.geometry = triangles ? static_cast<int>(distance_index) : -1;
            if (triangles)
            {
                row.distances = config.geometries[distance_index].distances;
                row.triangle_balance = config.geometries[distance_index].balance;
                row.embedding_count = config.geometries[distance_index].embeddings.size();
            }
            row.stats = stats;
            rows.push_back(std::move(row));
        }
    }
}

} // namespace detail

// Everything one run produces: the aggregate rows, and -- for the two
// protocols whose curves get fitted as curves -- the independent batch-mean
// replicas of the same grid.
struct SimulationResult
{
    std::vector<OutputRow> rows;
    std::vector<OutputRow> batch_rows;
};

// Run every p point of one configuration and return the finished rows.
inline SimulationResult simulate(const ProbeRunConfig &config, ProgressReporter &progress)
{
    const int n = config.n;
    const int t_max = config.t_max;
    const auto sdp = SdpSettings::from_environment(config.realizations);
    // Only the fitted-curve protocols keep batch replicas; see writes_batch_csv.
    const BatchPlan batch_plan =
        writes_batch_csv(config.probes, config.mode)
            ? make_batch_plan(config.realizations, batch_count_for(config.realizations))
            : BatchPlan{};

    SimulationResult result;
    std::vector<OutputRow> &rows = result.rows;
    rows.reserve(config.p_values.size() * config.times.size() * static_cast<std::size_t>(config.bin_count));
    std::mt19937 origin_rng{std::random_device{}()};
    std::uint64_t completed = 0;
    util::PauseSentinel pause_sentinel;

    for (std::size_t p_index = 0; p_index < config.p_values.size(); ++p_index)
    {
        progress.begin_point(completed);
        const double p = config.p_values[p_index];
        const std::size_t aggregate_size = config.times.size() * static_cast<std::size_t>(config.bin_count);
        std::vector<Aggregate> aggregates(aggregate_size);
        // One full copy of the grid per batch. At the 64-batch default this is
        // 64 x bins x times accumulators, which is megabytes at most -- the
        // point of batching rather than storing per-trajectory samples.
        std::vector<std::vector<Aggregate>> batch_aggregates(
            static_cast<std::size_t>(std::max(batch_plan.count, 0)), std::vector<Aggregate>(aggregate_size));
        // Reused by every one-probe mode-5 readout so the inner loop allocates
        // nothing.
        std::vector<Sample> front_scratch(config.mode == 5 && config.probes == 1 ? config.receivers.bin_count() : 0);
        std::vector<unsigned char> front_filled(front_scratch.size(), 0u);

        std::vector<std::vector<unsigned char>> gmn_selected;
        std::unique_ptr<SdpBatchQueue> sdp_queue;
        if (config.mode == 4 && config.probes == 3)
        {
            // The SDP workers are what use the cores here, so keep MOSEK
            // single-threaded and bound how many solve at once. The bound
            // scales with the machine; see default_sdp_worker_count.
            env::set_if_unset("GMN_MOSEK_NUM_THREADS", "1");
            env::set_if_unset("FGMN_MAX_CONCURRENT_MOSEK",
                              analysis::default_sdp_worker_count_text().c_str());
            env::set_if_unset("GMN_MOSEK_TOL", analysis::DEFAULT_SDP_TOLERANCE_TEXT);
            gmn_selected = detail::select_gmn_samples(config, sdp, origin_rng);
            sdp_queue = std::make_unique<SdpBatchQueue>(aggregates, uses_fermionic_trace(config.type), sdp.batch_size,
                                                        sdp.pending_batches);

            // MOSEK Fusion dies on a signal rather than throwing, so record
            // what is being solved and with what concurrency before the first
            // solve; see util/crash_report.hpp.
            const char *concurrency = std::getenv("FGMN_MAX_CONCURRENT_MOSEK");
            util::crash::set_context(
                "run: mipt_probed.exe " + describe_run(config) +
                "\n  output: " + config.output +
                "\n  sdp=" + std::string(uses_fermionic_trace(config.type) ? "fGMN" : "GMN") +
                ", FGMN_MAX_CONCURRENT_MOSEK=" +
                (concurrency != nullptr ? std::string(concurrency) : std::string("unset")) +
                ", MIPT_PROBED_GMN_PENDING_BATCHES=" + std::to_string(sdp.pending_batches) +
                ", MIPT_PROBED_GMN_BATCH_RECORDS=" + std::to_string(sdp.batch_size));
            util::crash::watch("sdp_records_in_flight", sdp_queue->in_flight_counter());
            util::crash::watch("sdp_records_solved", sdp_queue->solved_counter());
        }

        std::array<probed::CircuitWorkspace1D, 2> workspaces;
        for (auto &workspace : workspaces)
        {
            workspace.reserve(t_max);
        }
        std::array<std::future<void>, 2> jobs;
        // One-probe modes 0 and 1 scramble the freshly attached ebit through a
        // measurement-free encoder first, which is a quenched history: the
        // first t_encode layers carry no measurement flags at all.
        auto prepare = [&](probed::CircuitWorkspace1D &workspace) {
            if (config.t_encode > 0)
            {
                workspace.prepare_quench(n, t_max, config.t_encode, p, config.type);
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

            // Modes that sample one bin per trajectory run `realizations`
            // consecutive trajectories per bin; the rest fill every bin from
            // every trajectory and stay on bin 0.
            const bool grouped = config.mode == 4 || (config.mode == 5 && config.probes == 2);
            const std::size_t geometry_index =
                grouped ? static_cast<std::size_t>(trajectory / static_cast<std::uint64_t>(config.realizations)) : 0;
            const std::size_t realization_index =
                grouped ? static_cast<std::size_t>(trajectory % static_cast<std::uint64_t>(config.realizations)) : 0;
            const int distance =
                config.mode == 4 && config.probes == 2 ? static_cast<int>(geometry_index) + 1 : -1;
            // Batches are cut on the realization index within a bin group, not
            // on the global trajectory counter, so every batch holds the same
            // number of trajectories at every distance and covers the whole
            // grid. Ungrouped modes run one bin, where the two coincide.
            const std::size_t batch_realization =
                grouped ? realization_index : static_cast<std::size_t>(trajectory);
            std::vector<Aggregate> *batch_target =
                batch_plan.count > 0
                    ? &batch_aggregates[static_cast<std::size_t>(batch_plan.batch_at(batch_realization))]
                    : nullptr;
            const auto trajectory_sites = detail::sample_trajectory_sites(config, geometry_index, distance, origin_rng);
            const bool random_origin = config.mode == 4 || config.mode == 5;
            const auto &active_sites = random_origin ? trajectory_sites : config.sites;

            // Mode 5 injects its probes only after equilibration, so unlike the
            // other one-probe modes it starts from a bare product state.
            const bool attach_at_start = config.mode == 2 || (config.probes == 1 && config.mode != 5);
            auto &workspace = workspaces[current];
            auto state = config.parity_encoding
                             ? workspace.simulate_parity_encoded_probe(n, active_sites.front())
                             : initialize_state(n, active_sites, attach_at_start);
            const int current_t = config.parity_encoding ? t_max : 0;

            if (config.mode == 5 && config.probes == 2)
            {
                detail::run_mode5_stagger_trajectory(config, workspace, state, aggregates, active_sites,
                                                     config.staggers.delay_at(geometry_index), geometry_index);
            }
            else if (config.mode == 5)
            {
                detail::run_mode5_trajectory(config, workspace, state, aggregates, active_sites.front(), front_scratch,
                                             front_filled);
            }
            else if (config.mode == 4)
            {
                detail::run_mode4_trajectory(config, workspace, state, aggregates, batch_target, active_sites,
                                             geometry_index, realization_index, sdp, gmn_selected, sdp_queue.get());
            }
            else if (config.mode == 3)
            {
                detail::run_mode3_trajectory(config, workspace, state, aggregates);
            }
            else
            {
                detail::run_sampling_trajectory(config, workspace, state, aggregates, batch_target, current_t);
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
        for (int batch = 0; batch < batch_plan.count; ++batch)
        {
            detail::append_rows(config, p, batch_aggregates[static_cast<std::size_t>(batch)], result.batch_rows, batch,
                                batch_plan.realizations_in_batch[static_cast<std::size_t>(batch)]);
        }
    }
    return result;
}

} // namespace mipt::probed
