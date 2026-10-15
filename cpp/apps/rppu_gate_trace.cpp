// Stage-2 gate-trace replay for connected-but-unentangled RPPU pairs.
//
//   ./rppu_gate_trace.exe <records.bin> <selection.csv> [output.csv]
//
// For every selected (realization, pair) this regenerates the trajectory
// dist_scaling.exe measured -- same master seed, same per-realization
// trajectory_seed, same layer draw, same boundary implementation, and the same
// FP64 cuStateVec engine calls with the same measurement rng, so the same
// measurement outcomes -- rebuilds the spacetime graph, finds the pair's
// shortest-maximin and widest (entangling-power) paths, and at every gate on
// those paths reports the two-mode fermionic state of that gate's own modes
// before the gate, after its unitary, and after the layer's measurements.
// gate_trace.hpp holds the reasoning and the schemas; this file is argument
// handling, refusals, and orchestration.
//
// The replay is only worth anything if it is the same trajectory, so it is
// *proven* to be, per realization: after the run, the final state goes through
// the production GPU pair reduction for every pair, and every stored record of
// that realization -- every value and every flag bit, gate-path values
// included -- has to agree within gate_trace.hpp's tolerances. One
// disagreement anywhere marks every trace row of that realization invalid
// (they are still written, with trace_valid=0, because a failed replay is
// itself the thing to look at). The anchor pair's host-reduced snapshot is also
// compared against its GPU RDM, which closes the chain from the observed
// encoded snapshot through the host parity expansion and host reduction.
//
// Exit status: 0 when every trace validated, 3 when any realization failed
// validation (outputs still complete), 2 on a usage error or a refusal before
// anything ran, 1 on any other error.

#include "mipt/dist_record_reader.hpp"
#include "mipt/dist_scaling.hpp"
#include "mipt/gate_trace.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace
{

void print_usage(const char *program)
{
    std::cout
        << "Usage: " << program << " <records.bin> <selection.csv> [output.csv]\n\n"
        << "Replays selected dist_scaling.exe k=2 RPPU trajectories (circ_type 2) and traces\n"
        << "the two-mode fermionic state at every gate on each selected pair's\n"
        << "shortest-maximin and widest-entangling-power spacetime paths.\n\n"
        << "  records.bin    a format-v2 <stem>_records.bin written by dist_scaling.exe at\n"
        << "                 fp64 with MIPT_DIST_GATE_DIAGNOSTICS=1\n"
        << "  selection.csv  data_analysis/select_gate_traces.py output, header\n"
        << "                 " << mipt::gate_trace::SELECTION_HEADER << "\n"
        << "  output.csv     default <selection stem>_gate_trace.csv; the validation table\n"
        << "                 goes beside it as <output stem>_validation.csv\n\n"
        << "Requirements (each is a refusal, not a warning):\n"
        << "  * an FP64=1 build with the cuStateVec engine (CUSV=1); the run being replayed\n"
        << "    must have statevector_precision=64. Measurement outcomes only follow the\n"
        << "    seed on the cuStateVec path, so the CUDA-Q fallback is refused.\n"
        << "  * MIPT_DIRECT_FERMION_BOUNDARY must reproduce the file's\n"
        << "    boundary_implementation (1 = jw_string, 0 = fswap_network).\n"
        << "  * MIPT_CUSV_PARITY, MIPT_CUSV_MEASURE_BITS, MIPT_CUSV_BLOCK_TARGETS must equal\n"
        << "    the file's cusv_* header fields: they change the arithmetic or the rng\n"
        << "    consumption, and with it the outcomes.\n"
        << "  * MIPT_CUSV must be on and MIPT_DEBUG_PREFIX_LAYERS unset.\n"
        << "  * every selected realization must be one the record stride kept.\n\n"
        << "Exit status: 0 all traces valid; 3 some realization failed validation (rows\n"
        << "written with trace_valid=0); 2 usage error or refusal; 1 other error.\n";
}

[[noreturn]] void refuse(const std::string &why)
{
    throw std::invalid_argument(why);
}

long header_integer(const mipt::dist::records::RecordReader &reader, const std::string &key)
{
    try
    {
        return std::stol(reader.field(key));
    }
    catch (const std::invalid_argument &)
    {
        refuse(reader.path() + ": header field `" + key + "` is not an integer.");
    }
}

#ifdef MIPT_ENABLE_CUSV

// One gate on one path of one selected pair.
struct PlannedEdge
{
    std::size_t selection = 0;
    mipt::gate_trace::PathRole role = mipt::gate_trace::PathRole::ShortestMaximin;
    int position = 0;
    std::uint32_t edge_id = 0;
    mipt::dist::GraphEdge edge;
    int mode_a = 0;
    int mode_b = 0;
};

using SnapshotKey = std::tuple<std::size_t, bool, int, int>; // layer, after, a, b

// Watches the trajectory the engine runs. The plan -- which layers to copy
// back, and which mode pairs to reduce there -- needs the layer draw, which
// only exists once simulate() has drawn it; the first callback fires after
// that, so it builds the plan there through `planner`.
class TraceObserver final : public mipt::LayerObserver
{
  public:
    using Planner = std::function<void(std::map<std::size_t, std::set<std::pair<int, int>>> &,
                                       std::map<std::size_t, std::set<std::pair<int, int>>> &)>;

    explicit TraceObserver(Planner planner) : planner_(std::move(planner)) {}

    void before_unitaries(std::size_t layer, mipt::cusv::Engine &engine, bool encoded,
                          int n) override
    {
        if (!planned_)
        {
            planner_(before_, after_);
            planned_ = true;
        }
        capture(layer, false, before_, engine, encoded, n);
    }

    void after_measurements(std::size_t layer, mipt::cusv::Engine &engine, bool encoded, int n,
                            const std::vector<int> &, const std::vector<int> &) override
    {
        capture(layer, true, after_, engine, encoded, n);
    }

    const std::map<SnapshotKey, mipt::gate_trace::Rho4> &snapshots() const { return snapshots_; }
    std::size_t host_copies() const { return host_copies_; }

  private:
    void capture(std::size_t layer, bool after,
                 const std::map<std::size_t, std::set<std::pair<int, int>>> &needs,
                 mipt::cusv::Engine &engine, bool encoded, int n)
    {
        const auto it = needs.find(layer);
        if (it == needs.end() || it->second.empty())
        {
            return;
        }
        engine.to_host(host_);
        ++host_copies_;
        const std::vector<std::complex<double>> &full =
            encoded ? (expanded_ = mipt::gate_trace::expand_parity_sector(host_, n)) : host_;
        for (const auto &[a, b] : it->second)
        {
            snapshots_[{layer, after, a, b}] = mipt::gate_trace::reduce_pair(full, n, a, b);
        }
    }

    Planner planner_;
    bool planned_ = false;
    std::map<std::size_t, std::set<std::pair<int, int>>> before_;
    std::map<std::size_t, std::set<std::pair<int, int>>> after_;
    std::map<SnapshotKey, mipt::gate_trace::Rho4> snapshots_;
    std::vector<std::complex<double>> host_;
    std::vector<std::complex<double>> expanded_;
    std::size_t host_copies_ = 0;
};

bool same_history(const mipt::LogicalHistory &x, const mipt::LogicalHistory &y)
{
    if (x.n != y.n || x.layers.size() != y.layers.size() || x.mode_at_site != y.mode_at_site)
    {
        return false;
    }
    for (std::size_t layer = 0; layer < x.layers.size(); ++layer)
    {
        const auto &a = x.layers[layer];
        const auto &b = y.layers[layer];
        if (a.measured != b.measured || a.bonds.size() != b.bonds.size())
        {
            return false;
        }
        for (std::size_t k = 0; k < a.bonds.size(); ++k)
        {
            if (a.bonds[k].a != b.bonds[k].a || a.bonds[k].b != b.bonds[k].b ||
                a.bonds[k].source_gate_index != b.bonds[k].source_gate_index)
            {
                return false;
            }
        }
    }
    return true;
}

const char *engine_path_name(mipt::EnginePath path)
{
    switch (path)
    {
    case mipt::EnginePath::CusvParity: return "cusv_parity";
    case mipt::EnginePath::Cusv: return "cusv";
    case mipt::EnginePath::CudaqFallback: return "cudaq_fallback";
    case mipt::EnginePath::None: return "none";
    }
    return "unknown";
}

int run(const std::string &records_path, const std::string &selection_path,
        const std::string &trace_path)
{
    namespace gt = mipt::gate_trace;
    namespace records = mipt::dist::records;
    using mipt::dist::GraphEdgeKind;

    // ----------------------------------------------------------- refusals
    if (MIPT_CUDAQ_PRECISION != 64)
    {
        refuse("rppu_gate_trace.exe was built at fp" + std::to_string(MIPT_CUDAQ_PRECISION) +
               ". A replay has to reproduce the measured trajectory's arithmetic, and only an "
               "fp64 trajectory is worth tracing: build with FP64=1 (make FP64=1 "
               "rppu_gate_trace.exe) and replay a dist_scaling.exe run made at fp64.");
    }
    if (!mipt::cusv::enabled())
    {
        refuse("MIPT_CUSV=0 (or a tensor-network target) disables the cuStateVec engine, and "
               "measurement outcomes only follow the seed on that engine. Unset MIPT_CUSV.");
    }
    if (mipt::backend::debug_prefix_layers() > 0)
    {
        refuse("MIPT_DEBUG_PREFIX_LAYERS truncates the circuit; production runs do not. Unset "
               "it.");
    }

    records::RecordReader reader(records_path);
    if (reader.version() != records::FORMAT_VERSION)
    {
        refuse(records_path + " is a format v" + std::to_string(reader.version()) +
               " record file; the replay needs v" + std::to_string(records::FORMAT_VERSION) +
               " (connectivity flags, embedding table, gate-path values).");
    }
    if (header_integer(reader, "k") != 2)
    {
        refuse(records_path + " is a k=" + reader.field("k") + " record file; the replay needs "
               "the k=2 pair records.");
    }
    if (header_integer(reader, "circ_type") != 2)
    {
        refuse(records_path + " is circ_type " + reader.field("circ_type") +
               "; the gate trace is for RPPU (circ_type 2) only.");
    }
    if (header_integer(reader, "statevector_precision") != 64)
    {
        refuse(records_path + " was written at statevector_precision=" +
               reader.field("statevector_precision") +
               ". An fp32 trajectory cannot be reproduced by an fp64 replay (the Born "
               "comparisons land differently), and its sub-1e-14 values are arithmetic "
               "anyway: rebuild dist_scaling.exe with FP64=1 and rerun it at fp64.");
    }
    const std::string boundary =
        mipt::backend::direct_fermion_boundary_enabled() ? "jw_string" : "fswap_network";
    if (reader.field("boundary_implementation") != boundary)
    {
        refuse(records_path + " was simulated with boundary_implementation=" +
               reader.field("boundary_implementation") + ", but this process would use " +
               boundary + ". Set MIPT_DIRECT_FERMION_BOUNDARY=" +
               (reader.field("boundary_implementation") == "jw_string" ? "1" : "0") +
               ": the two draw the same circuit but run different gate lists, so the "
               "measurement outcomes differ.");
    }
    if (reader.field("connectivity") != "1")
    {
        refuse(records_path + " was written without connectivity flags (MIPT_DIST_CONNECTIVITY=0).");
    }
    for (const char *name : records::gate_observable_names())
    {
        if (reader.observable_index(name) < 0)
        {
            refuse(records_path + " carries no `" + std::string(name) +
                   "` observable: rerun dist_scaling.exe with MIPT_DIST_GATE_DIAGNOSTICS=1.");
        }
    }
    if (reader.has("gate_path_diagnostics") && reader.field("gate_path_diagnostics") != "1")
    {
        refuse(records_path + " says gate_path_diagnostics=" + reader.field("gate_path_diagnostics") + ".");
    }
    // The engine knobs. Each changes either the floating-point order of the
    // circuit (block width) or how many uniforms a measurement layer draws
    // (chunk width, parity encoding), and so the outcomes.
    const std::vector<std::tuple<const char *, const char *, long>> knobs{
        {"cusv_parity", "MIPT_CUSV_PARITY", mipt::cusv::parity_encoding_enabled() ? 1L : 0L},
        {"cusv_measure_bits", "MIPT_CUSV_MEASURE_BITS", mipt::cusv::max_joint_measure_bits()},
        {"cusv_block_targets", "MIPT_CUSV_BLOCK_TARGETS", mipt::cusv::max_block_targets()},
    };
    for (const auto &[field, variable, current] : knobs)
    {
        if (!reader.has(field))
        {
            std::cerr << "WARNING: " << records_path << " does not record `" << field
                      << "`, so " << variable << "=" << current
                      << " cannot be checked against the run; the record comparison will "
                         "catch a mismatch.\n";
            continue;
        }
        if (header_integer(reader, field) != current)
        {
            refuse(records_path + " was run with " + field + "=" + reader.field(field) +
                   ", but this process has " + variable + "=" + std::to_string(current) +
                   ". Set " + variable + "=" + reader.field(field) + ".");
        }
    }

    const int n = static_cast<int>(header_integer(reader, "N"));
    const int periods = static_cast<int>(header_integer(reader, "periods"));
    const double p = std::stod(reader.field("p"));
    const long realizations = header_integer(reader, "realizations");
    const long stride = header_integer(reader, "record_stride");
    const std::uint64_t master_seed = std::stoull(reader.field("master_seed"));
    const double pair_zero_tol = std::stod(reader.field("pair_zero_tol"));
    const auto embedding_sites = reader.embedding_sites();
    const std::vector<std::string> &observables = reader.observables();
    const int fmn_slot = reader.observable_index("fmn");
    const int fmi_slot = reader.observable_index("fmi");
    if (fmn_slot < 0 || fmi_slot < 0)
    {
        refuse(records_path + " has no fermionic fmn/fmi observables.");
    }

    // ---------------------------------------------------------- selection
    const std::vector<gt::Selection> selection = gt::read_selection(selection_path);
    std::map<std::pair<int, int>, std::pair<int, int>> embedding_of_pair;
    for (const auto &[ids, sites] : embedding_sites)
    {
        if (sites.size() == 2)
        {
            embedding_of_pair[{std::min(sites[0], sites[1]), std::max(sites[0], sites[1])}] = ids;
        }
    }
    std::map<std::uint32_t, std::vector<std::size_t>> by_realization;
    for (std::size_t k = 0; k < selection.size(); ++k)
    {
        const gt::Selection &row = selection[k];
        const std::pair<int, int> key{std::min(row.i, row.j), std::max(row.i, row.j)};
        const auto ids = embedding_of_pair.find(key);
        if (row.i >= n || row.j >= n || ids == embedding_of_pair.end())
        {
            refuse(selection_path + ": pair (" + std::to_string(row.i) + "," +
                   std::to_string(row.j) + ") is not a pair of this N=" + std::to_string(n) +
                   " run.");
        }
        if (ids->second != std::make_pair(row.geometry_id, row.embedding_id))
        {
            refuse(selection_path + ": pair (" + std::to_string(row.i) + "," +
                   std::to_string(row.j) + ") is (geometry " + std::to_string(row.geometry_id) +
                   ", embedding " + std::to_string(row.embedding_id) + ") in the selection but (" +
                   std::to_string(ids->second.first) + ", " + std::to_string(ids->second.second) +
                   ") in " + records_path + "; the selection was made from a different file.");
        }
        if (static_cast<long>(row.realization) >= realizations)
        {
            refuse(selection_path + ": realization " + std::to_string(row.realization) +
                   " is past the run's " + std::to_string(realizations) + ".");
        }
        if (!records::stride_selects(stride, row.realization))
        {
            refuse(selection_path + ": realization " + std::to_string(row.realization) +
                   " was not kept by record_stride=" + std::to_string(stride) +
                   ", so there are no records to validate its replay against.");
        }
        by_realization[row.realization].push_back(k);
    }

    // ------------------------------------------------------------ outputs
    const std::string validation_path = gt::validation_path_for(trace_path);
    mipt::dist::ensure_output_parent_directory(trace_path);
    std::ofstream trace_out(trace_path, std::ios::trunc);
    std::ofstream validation_out(validation_path, std::ios::trunc);
    if (!trace_out || !validation_out)
    {
        throw std::runtime_error("Could not open " + trace_path + " or " + validation_path +
                                 " for writing.");
    }
    trace_out << gt::TRACE_HEADER << '\n';
    validation_out << gt::VALIDATION_HEADER << '\n';

    std::cout << "rppu_gate_trace: records=" << records_path << ", selection=" << selection_path
              << '\n'
              << "N=" << n << ", periods=" << periods << ", p=" << reader.field("p")
              << ", master_seed=" << master_seed << ", boundary=" << boundary
              << ", record_stride=" << stride << ", pair_zero_tol=" << reader.field("pair_zero_tol")
              << '\n'
              << "engine: MIPT_CUSV_PARITY=" << (mipt::cusv::parity_encoding_enabled() ? 1 : 0)
              << ", MIPT_CUSV_MEASURE_BITS=" << mipt::cusv::max_joint_measure_bits()
              << ", MIPT_CUSV_BLOCK_TARGETS=" << mipt::cusv::max_block_targets()
              << ", precision=fp" << MIPT_CUDAQ_PRECISION << '\n'
              << "selected pairs=" << selection.size() << " over " << by_realization.size()
              << " realization(s)\n"
              << "trace=" << trace_path << "\nvalidation=" << validation_path << '\n';

    const std::vector<mipt::dist::Pair> pairs = mipt::dist::all_unordered_pairs(n);
    mipt::dist::RdmWorkspace workspace;
    // Through the registry's one dispatch point, exactly as CircuitWorkspace1D
    // builds it in production. (Constructing RppuCircuit1D directly also links
    // badly: nvq++ only emits the other circuits' __qpu__ kernels when the
    // dispatch that references them is in the TU.)
    std::unique_ptr<mipt::Circuit1D> owned = mipt::make_circuit_1d(mipt::CircuitType::FermionRPPU);
    auto *rppu = dynamic_cast<mipt::RppuCircuit1D *>(owned.get());
    if (rppu == nullptr)
    {
        throw std::runtime_error("make_circuit_1d(FermionRPPU) did not return an RppuCircuit1D.");
    }
    mipt::RppuCircuit1D &circuit = *rppu;
    circuit.reserve(periods);
    circuit.require_cusv(true);

    std::size_t done = 0;
    std::size_t invalid_realizations = 0;
    std::size_t rows_written = 0;
    for (const auto &group : by_realization)
    {
        // Plain names rather than a structured binding: the planner lambda
        // below captures them.
        const std::uint32_t realization = group.first;
        const std::vector<std::size_t> &members = group.second;
        const auto stored = reader.read_realization(realization);
        if (stored.size() != pairs.size())
        {
            throw std::runtime_error(records_path + " holds " + std::to_string(stored.size()) +
                                     " record(s) for realization " +
                                     std::to_string(realization) + "; a kept trajectory has " +
                                     std::to_string(pairs.size()) + ".");
        }

        // The plan is built inside the first observer callback, from the
        // layer list simulate() has just drawn.
        mipt::LogicalHistory plan_history;
        mipt::dist::ConnectivityIndex connectivity;
        std::vector<PlannedEdge> planned;
        std::vector<std::array<std::size_t, 2>> path_edges(members.size(), {0u, 0u});
        TraceObserver observer([&](auto &before, auto &after) {
            plan_history = mipt::logical_history_from_rppu(circuit.layers(), n);
            if (!plan_history.identity_permutation())
            {
                throw std::runtime_error("The layer history ends on a non-identity mode "
                                         "permutation; the snapshot mode labels would be wrong.");
            }
            connectivity.build(plan_history, true, true);
            if (!connectivity.has_gate_metrics())
            {
                throw std::runtime_error("The regenerated graph carries no gate metrics.");
            }
            for (std::size_t m = 0; m < members.size(); ++m)
            {
                const gt::Selection &row = selection[members[m]];
                for (const gt::PathRole role : {gt::PathRole::ShortestMaximin, gt::PathRole::Widest})
                {
                    const std::vector<std::uint32_t> path =
                        role == gt::PathRole::ShortestMaximin
                            ? connectivity.shortest_maximin_path(row.i, row.j)
                            : connectivity.widest_gate_path(row.i, row.j,
                                                            mipt::dist::GateCapacity::EntanglingPower);
                    for (std::size_t position = 0; position < path.size(); ++position)
                    {
                        const mipt::dist::GraphEdge &edge = connectivity.graph_edges()[path[position]];
                        if (edge.kind != GraphEdgeKind::Unitary)
                        {
                            continue;
                        }
                        PlannedEdge entry;
                        entry.selection = m;
                        entry.role = role;
                        entry.position = static_cast<int>(position);
                        entry.edge_id = path[position];
                        entry.edge = edge;
                        entry.mode_a = static_cast<int>(edge.a % static_cast<std::uint32_t>(n));
                        entry.mode_b = static_cast<int>(edge.b % static_cast<std::uint32_t>(n));
                        const std::size_t layer = static_cast<std::size_t>(edge.layer);
                        before[layer].insert({entry.mode_a, entry.mode_b});
                        after[layer].insert({entry.mode_a, entry.mode_b});
                        ++path_edges[m][role == gt::PathRole::ShortestMaximin ? 0 : 1];
                        planned.push_back(entry);
                    }
                }
                // The anchor itself at the end, for the host-vs-GPU check.
                after[circuit.layers().size() - 1].insert(
                    {std::min(row.i, row.j), std::max(row.i, row.j)});
            }
        });

        circuit.seed(mipt::seeding::trajectory_seed(master_seed, realization));
        circuit.set_layer_observer(&observer);
        std::optional<mipt::TrajectoryResult> trajectory;
        try
        {
            trajectory.emplace(circuit.simulate(n, periods, p, "rppu_gate_trace", nullptr));
        }
        catch (...)
        {
            circuit.set_layer_observer(nullptr);
            throw;
        }
        circuit.set_layer_observer(nullptr);
        auto &state = trajectory->state;
        if (state.get_precision() != cudaq::SimulationState::precision::fp64)
        {
            refuse("The CUDA-Q target returned an fp32 state; select the fp64 target (the "
                   "FP64=1 build does this) before replaying.");
        }
        const mipt::EnginePath expected =
            (mipt::cusv::parity_encoding_enabled() && n >= 4) ? mipt::EnginePath::CusvParity
                                                              : mipt::EnginePath::Cusv;
        const std::string engine = engine_path_name(circuit.last_engine_path());
        bool realization_valid = circuit.last_engine_path() == expected;
        if (!same_history(plan_history, trajectory->history))
        {
            realization_valid = false;
            std::cerr << "WARNING: realization " << realization
                      << ": the planned history differs from the simulated one.\n";
        }

        // ---------------------------------------- the production reduction
        mipt::dist::rho2_subsystems_both_from_cudaq_state(state, n, pairs, workspace);
        std::map<std::pair<int, int>, gt::ReplayedRecord> replayed;
        std::map<std::pair<int, int>, std::size_t> pair_index;
        std::map<std::pair<int, int>, mipt::dist::PairMetrics> fermion_of;
        for (std::size_t k = 0; k < pairs.size(); ++k)
        {
            const double *rho = workspace.rho_ri.data() + k * mipt::dist::RHO2_VALUES;
            const double *frho = workspace.fermion_rho_ri.data() + k * mipt::dist::RHO2_VALUES;
            const mipt::dist::PairMetrics metrics = mipt::dist::two_party_metrics(rho, false);
            const mipt::dist::PairMetrics fermion = mipt::dist::two_party_metrics(frho, true, true);
            const int i = pairs[k][0];
            const int j = pairs[k][1];
            const mipt::dist::PairConnectivity conn = connectivity.query(i, j);
            const mipt::dist::GatePathMetrics gate = connectivity.gate_path_metrics(i, j);
            std::uint32_t flags = conn.flags();
            flags |= fermion.mn > pair_zero_tol ? mipt::dist::FLAG_FN_POSITIVE : 0u;
            flags |= metrics.mn > pair_zero_tol ? mipt::dist::FLAG_MN_POSITIVE : 0u;
            replayed[{i, j}] = {flags, gt::replayed_observables(observables, metrics, fermion, conn, gate)};
            pair_index[{i, j}] = k;
            fermion_of[{i, j}] = fermion;
        }
        const gt::RealizationCheck check =
            gt::check_realization(observables, stored, embedding_sites, replayed);
        realization_valid = realization_valid && check.valid();

        // Per anchor: stored and replayed values, and the host snapshot of the
        // final state against the production GPU RDM.
        const std::size_t last = circuit.layers().size() - 1;
        std::vector<gt::ValidationRow> validation(members.size());
        for (std::size_t m = 0; m < members.size(); ++m)
        {
            const gt::Selection &row = selection[members[m]];
            const std::pair<int, int> key{std::min(row.i, row.j), std::max(row.i, row.j)};
            gt::ValidationRow &v = validation[m];
            v.realization = realization;
            v.pair_i = row.i;
            v.pair_j = row.j;
            v.group = row.group;
            v.engine_path = engine;
            v.selection_fmn = row.fmn;
            v.selection_fmi = row.fmi;
            v.replayed_fmn = fermion_of[key].mn;
            v.replayed_fmi = fermion_of[key].mi;
            for (const auto &record : stored)
            {
                const auto &sites = embedding_sites.at({record.geometry, record.embedding});
                if (std::min(sites[0], sites[1]) == key.first && std::max(sites[0], sites[1]) == key.second)
                {
                    v.stored_fmn = record.values[static_cast<std::size_t>(fmn_slot)];
                    v.stored_fmi = record.values[static_cast<std::size_t>(fmi_slot)];
                }
            }
            const mipt::dist::PairConnectivity conn = connectivity.query(row.i, row.j);
            v.connected = conn.connected;
            v.gate_path_available = connectivity.gate_path_metrics(row.i, row.j).available;
            v.shortest_maximin_edges = path_edges[m][0];
            v.widest_edges = path_edges[m][1];
            v.check = check;
            const auto host = observer.snapshots().find({last, true, key.first, key.second});
            if (host != observer.snapshots().end())
            {
                gt::Rho4 gpu = gt::unpack(workspace.fermion_rho_ri.data() +
                                          pair_index.at(key) * mipt::dist::RHO2_VALUES);
                double trace = 0.0;
                for (int d = 0; d < 4; ++d)
                {
                    trace += gpu[static_cast<std::size_t>(d * 5)].real();
                }
                for (auto &value : gpu)
                {
                    value /= trace;
                }
                v.anchor_rdm_diff = gt::max_abs_difference(host->second, gpu);
            }
            if (!(v.anchor_rdm_diff <= gt::ANCHOR_RDM_TOL))
            {
                realization_valid = false;
            }
        }
        for (gt::ValidationRow &v : validation)
        {
            v.valid = realization_valid;
            validation_out << gt::render_validation_row(v);
        }

        // ------------------------------------------------------ trace rows
        for (const PlannedEdge &entry : planned)
        {
            const gt::Selection &row = selection[members[entry.selection]];
            const std::size_t layer = static_cast<std::size_t>(entry.edge.layer);
            const mipt::RppuBondGate &gate =
                circuit.layers()[layer].gates[static_cast<std::size_t>(entry.edge.source_gate_index)];
            const gt::Rho4 &pre = observer.snapshots().at({layer, false, entry.mode_a, entry.mode_b});
            const gt::Rho4 &post_measure =
                observer.snapshots().at({layer, true, entry.mode_a, entry.mode_b});
            const gt::Rho4 post_unitary = gt::conjugate(mipt::rppu_gate_matrix(gate), pre);

            gt::TraceRow out;
            out.realization = realization;
            out.pair_i = row.i;
            out.pair_j = row.j;
            out.group = row.group;
            out.role = entry.role;
            out.layer = entry.edge.layer;
            out.mode_a = entry.mode_a;
            out.mode_b = entry.mode_b;
            out.entangling_power = entry.edge.entangling_power;
            out.even_block_mixing = entry.edge.even_block_mixing;
            out.odd_block_mixing = entry.edge.odd_block_mixing;
            out.pre = gt::pair_state(pre);
            out.post_unitary = gt::pair_state(post_unitary);
            out.rho_change_frobenius = gt::frobenius_distance(post_unitary, pre);
            out.post_measurement = gt::pair_state(post_measure);
            const std::pair<int, int> key{std::min(row.i, row.j), std::max(row.i, row.j)};
            out.final_anchor_fmn = fermion_of[key].mn;
            out.final_anchor_fmi = fermion_of[key].mi;
            out.path_position = entry.position;
            out.edge_id = entry.edge_id;
            out.source_gate_index = entry.edge.source_gate_index;
            const auto &measured = plan_history.layers[layer].measured;
            out.measured_a = measured[static_cast<std::size_t>(entry.mode_a)] ? 1 : 0;
            out.measured_b = measured[static_cast<std::size_t>(entry.mode_b)] ? 1 : 0;
            out.trace_valid = realization_valid;
            out.validation_max_error = check.max_abs_error;
            trace_out << gt::render_trace_row(out);
            ++rows_written;
        }
        trace_out.flush();
        validation_out.flush();

        ++done;
        if (!realization_valid)
        {
            ++invalid_realizations;
        }
        std::cerr << "[" << done << "/" << by_realization.size() << "] realization "
                  << realization << ": " << members.size() << " pair(s), " << planned.size()
                  << " gate row(s), " << observer.host_copies() << " snapshot(s), engine="
                  << engine << ", records " << check.records_compared << "/" << pairs.size()
                  << ", max_err=" << check.max_abs_error << ", "
                  << (realization_valid ? "VALID" : "INVALID");
        if (!check.first_mismatch.empty())
        {
            std::cerr << " (" << check.first_mismatch << ")";
        }
        std::cerr << '\n';
    }

    std::cout << "Wrote " << rows_written << " trace row(s) and " << selection.size()
              << " validation row(s); " << invalid_realizations << " of " << by_realization.size()
              << " realization(s) failed validation.\n";
    return invalid_realizations == 0 ? 0 : 3;
}

#endif // MIPT_ENABLE_CUSV

} // namespace

int main(int argc, char **argv)
{
    if (argc > 1 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0))
    {
        print_usage(argv[0]);
        return 0;
    }
    if (argc < 3 || argc > 4)
    {
        std::cerr << "Expected <records.bin> <selection.csv> [output.csv]. Use --help.\n";
        return 2;
    }
#ifndef MIPT_ENABLE_CUSV
    std::cerr << "rppu_gate_trace.exe was built without the cuStateVec engine (CUSV=0). "
                 "Measurement outcomes only follow the seed on that engine; rebuild with "
                 "GPU=1 FP64=1 on the nvidia backend.\n";
    return 2;
#else
    try
    {
        const std::string trace_path =
            argc > 3 ? std::string(argv[3]) : mipt::gate_trace::default_trace_path(argv[2]);
        return run(argv[1], argv[2], trace_path);
    }
    catch (const std::invalid_argument &error)
    {
        std::cerr << "REFUSED: " << error.what() << '\n';
        return 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
#endif
}
