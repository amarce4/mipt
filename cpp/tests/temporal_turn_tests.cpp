// Host tests for the temporal-turn diagnostics of the spacetime graph
// (dist_connectivity.hpp, "Temporal turns" and "The doubled network";
// MIPT_DIST_TURN_DIAGNOSTICS).
//
// The graph is undirected, so a path between two final-slice endpoints may run
// backward in time, cross a gate, and run forward again. The turn diagnostics
// read such a path as a shape -- a V through a common source, or a zig-zag over
// a late *bridge* gate -- and decide whether each bridge is measured after it
// (the record conditions on it, so it can swap entanglement onto the endpoints)
// or trace-only (it cancels against its adjoint, U^dag U = I, and the path
// carries nothing). The doubled network states the same cancellation exactly:
// keep only the gates in the past light cone of a measurement or of either
// endpoint, and if the endpoints are no longer joined, rho_ij factorizes.
// Two kinds of evidence are pinned here, and they fail in different ways.
//
//   * Physics, on circuits small enough to simulate exactly. The four-mode
//     swap circuit -- G1 on (A, C1) and G2 on (C2, B) in layer 0, the bridge
//     G3 on (C1, C2) in layer 1, an idle final layer -- is the smallest in
//     which the only path between two endpoints is a zig-zag. Unmeasured, the
//     bridge cancels and (A, B) is an exact product: the graph says
//     connected, the state says nothing, and the turn diagnostics say why.
//     Measure the connectors after G3 and the record swaps entanglement onto
//     (A, B); measure C1 *before* G3 and the path is cut outright. The circuit
//     is simulated on a 16-amplitude host state and read as a LogicalHistory
//     carrying the diagnostics of its own 4x4 gates, so the two sides are one
//     circuit rather than two descriptions of it. A six-mode double zig-zag
//     then shows that one trace-only bridge is enough to cut a path whose
//     other bridge is measured, and the doubled-network theorem is checked on
//     Born-sampled random open chains: every pair the reduction disconnects
//     must come out an exact product, whatever the gates are.
//
//   * Properties of the core, on hand-built shapes whose answers are
//     countable by hand and on random histories against independent
//     restatements: a past light-cone BFS for common_source and the doubled
//     network, a walk search over (node, edge just used) states with a
//     layered closure instead of a 0-1 deque for min_turns and certified, and
//     -- on small graphs -- the enumeration of every simple path. The last is
//     what says whether "a walk exists" (the core's definition) and "a simple
//     path exists" ever disagree; disagreements are counted and printed
//     rather than failed, because the walk definition is the documented one.
//
// Every history here, like every production RPPU/qRPPU history, gives each
// mode at most one gate per layer. The walk search's non-backtracking rule
// relies on exactly that (dist_connectivity.hpp, walk_successors), and the
// random-history test asserts that the production projection satisfies it.

#include "mipt/analysis/gate_entanglement.hpp"
#include "mipt/circuits/rppu_layer.hpp"
#include "mipt/dist_connectivity.hpp"
#include "mipt/dist_metrics.hpp"
#include "mipt/gate_trace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
int failures = 0;

void expect(bool condition, const std::string &message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

using Complex = std::complex<double>;
using State = std::vector<Complex>;
using mipt::GateMatrix2;
using mipt::GateMatrix4;
using mipt::LogicalBond;
using mipt::LogicalHistory;
using mipt::LogicalLayer;
using mipt::dist::BridgeStatus;
using mipt::dist::ConnectivityIndex;
using mipt::dist::ConnectorStatus;
using mipt::dist::GraphEdge;
using mipt::dist::GraphEdgeKind;
using mipt::dist::GraphPath;
using mipt::dist::PairConnectivity;
using mipt::dist::PairTurns;
using mipt::dist::PathBridge;
using mipt::dist::PathClass;
using mipt::dist::PathShape;
namespace gt = mipt::gate_trace;

std::uint32_t node_id(int layer, int mode, int n)
{
    return static_cast<std::uint32_t>(layer * n + mode);
}

// ---------------------------------------------------------------------------
// Field-by-field comparisons
// ---------------------------------------------------------------------------

bool same_bridge(const PathBridge &x, const PathBridge &y)
{
    return x.edge == y.edge && x.layer == y.layer && x.mode_in == y.mode_in &&
           x.mode_out == y.mode_out && x.measured_layer_in == y.measured_layer_in &&
           x.measured_layer_out == y.measured_layer_out && x.status == y.status;
}

// `with_cancelled` = false for a shape read by path_shape(), which cannot know
// the endpoints' cones and leaves cancelled_gates at -1.
bool same_shape(const PathShape &x, const PathShape &y, bool with_cancelled = true)
{
    if (x.bridges.size() != y.bridges.size())
    {
        return false;
    }
    for (std::size_t k = 0; k < x.bridges.size(); ++k)
    {
        if (!same_bridge(x.bridges[k], y.bridges[k]))
        {
            return false;
        }
    }
    return x.available == y.available && x.length == y.length && x.unitary == y.unitary &&
           x.temporal_forward == y.temporal_forward && x.temporal_backward == y.temporal_backward &&
           x.turns == y.turns && x.valleys == y.valleys && x.peaks == y.peaks &&
           x.min_layer == y.min_layer && x.max_layer == y.max_layer &&
           x.gate_layers == y.gate_layers && x.bridges_measured == y.bridges_measured &&
           x.bridges_endpoint == y.bridges_endpoint &&
           x.bridges_trace_only == y.bridges_trace_only && x.path_class == y.path_class &&
           x.connector_status == y.connector_status &&
           (!with_cancelled || x.cancelled_gates == y.cancelled_gates);
}

bool same_existence(const PairTurns &x, const PairTurns &y)
{
    return x.evaluated == y.evaluated && x.min_turns == y.min_turns &&
           x.common_source == y.common_source && x.certified == y.certified &&
           x.connected_common_source == y.connected_common_source &&
           x.connected_certified == y.connected_certified &&
           x.doubled_network == y.doubled_network && x.connected_doubled == y.connected_doubled;
}

bool same_turns(const PairTurns &x, const PairTurns &y)
{
    return same_existence(x, y) && same_shape(x.shortest_maximin, y.shortest_maximin) &&
           same_shape(x.widest, y.widest);
}

bool same_value(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

bool same_query(const PairConnectivity &x, const PairConnectivity &y)
{
    return x.flags() == y.flags() && x.component_size == y.component_size &&
           x.shortest_path == y.shortest_path && x.idle_i == y.idle_i && x.idle_j == y.idle_j;
}

bool same_gate_metrics(const mipt::dist::GatePathMetrics &x, const mipt::dist::GatePathMetrics &y)
{
    return x.available == y.available && x.shortest_path_length == y.shortest_path_length &&
           x.shortest_path_unitary_count == y.shortest_path_unitary_count &&
           same_value(x.shortest_maximin_entangling_power, y.shortest_maximin_entangling_power) &&
           same_value(x.widest_entangling_power, y.widest_entangling_power) &&
           same_value(x.widest_even_block_mixing, y.widest_even_block_mixing) &&
           same_value(x.widest_odd_block_mixing, y.widest_odd_block_mixing);
}

// ---------------------------------------------------------------------------
// Host simulation: the circuit, its history, and its measurement records
// ---------------------------------------------------------------------------

struct HostGate
{
    int q0 = 0;
    int q1 = 0;
    GateMatrix4 u{};
};

struct HostLayer
{
    std::vector<HostGate> gates;
    std::vector<int> measured; // after the layer's gates, as the kernels order them
};

using HostCircuit = std::vector<HostLayer>;

// A 4x4 on qubits (q0, q1), bit 0 of the matrix index = q0 -- the engine's
// convention for every bond. On adjacent qubits the qubit and fermionic
// pictures of a parity-preserving gate coincide, which is why every circuit
// simulated here uses adjacent bonds only.
void apply_two(State &psi, int q0, int q1, const GateMatrix4 &u)
{
    const std::size_t m0 = std::size_t{1} << q0;
    const std::size_t m1 = std::size_t{1} << q1;
    for (std::size_t base = 0; base < psi.size(); ++base)
    {
        if ((base & (m0 | m1)) != 0u)
        {
            continue;
        }
        Complex in[4];
        for (int s = 0; s < 4; ++s)
        {
            in[s] = psi[base | ((s & 1) ? m0 : 0u) | ((s & 2) ? m1 : 0u)];
        }
        for (int r = 0; r < 4; ++r)
        {
            Complex sum(0.0, 0.0);
            for (int c = 0; c < 4; ++c)
            {
                sum += u[static_cast<std::size_t>(r * 4 + c)] * in[c];
            }
            psi[base | ((r & 1) ? m0 : 0u) | ((r & 2) ? m1 : 0u)] = sum;
        }
    }
}

// The history the graph is built from, derived from the very circuit that is
// simulated: one bond per gate, carrying the diagnostics of its own 4x4.
LogicalHistory history_of(int n, const HostCircuit &circuit)
{
    LogicalHistory history;
    history.reset(n);
    for (const HostLayer &host : circuit)
    {
        LogicalLayer layer;
        layer.measured.assign(static_cast<std::size_t>(n), 0u);
        int index = 0;
        for (const HostGate &gate : host.gates)
        {
            const mipt::GateEntanglement metrics = mipt::gate_entanglement(gate.u);
            // Squared off-diagonal weight of each parity block, in the
            // parity_preserving_matrix layout: even |00> -> |11> is entry
            // (3, 0), odd |10> -> |01> is entry (1, 2).
            LogicalBond bond{gate.q0, gate.q1, std::norm(gate.u[3 * 4 + 0]),
                             std::norm(gate.u[1 * 4 + 2])};
            bond.source_gate_index = index++;
            bond.entangling_power = metrics.entangling_power_normalized;
            bond.even_block_mixing = metrics.even_block_mixing;
            bond.odd_block_mixing = metrics.odd_block_mixing;
            layer.bonds.push_back(bond);
        }
        for (int mode : host.measured)
        {
            layer.measured[static_cast<std::size_t>(mode)] = 1u;
        }
        history.layers.push_back(std::move(layer));
    }
    history.valid = true;
    return history;
}

struct Branch
{
    std::string record; // one outcome digit per measurement, in circuit order
    double probability = 0.0;
    State psi; // normalized
};

// Every measurement record of nonzero probability, each as its own projected
// final state. The record is forced rather than sampled, so every branch of
// the trajectory ensemble is seen once with its Born weight.
std::vector<Branch> measurement_branches(int n, const HostCircuit &circuit)
{
    std::size_t count = 0;
    for (const HostLayer &layer : circuit)
    {
        count += layer.measured.size();
    }
    std::vector<Branch> out;
    for (std::uint32_t record = 0; record < (1u << count); ++record)
    {
        State psi(std::size_t{1} << n, Complex(0.0, 0.0));
        psi[0] = 1.0;
        Branch branch;
        int k = 0;
        for (const HostLayer &layer : circuit)
        {
            for (const HostGate &gate : layer.gates)
            {
                apply_two(psi, gate.q0, gate.q1, gate.u);
            }
            for (int mode : layer.measured)
            {
                const int outcome = static_cast<int>((record >> k++) & 1u);
                branch.record += static_cast<char>('0' + outcome);
                for (std::size_t index = 0; index < psi.size(); ++index)
                {
                    if (static_cast<int>((index >> mode) & 1u) != outcome)
                    {
                        psi[index] = 0.0;
                    }
                }
            }
        }
        double weight = 0.0;
        for (const Complex &z : psi)
        {
            weight += std::norm(z);
        }
        if (!(weight > 1.0e-20))
        {
            continue; // a record the circuit never produces
        }
        for (Complex &z : psi)
        {
            z /= std::sqrt(weight);
        }
        branch.probability = weight;
        branch.psi = std::move(psi);
        out.push_back(std::move(branch));
    }
    return out;
}

// Sequential Born sampling of one occupation, exactly the chain rule the
// engine's joint sampling realizes.
void born_measure(State &psi, int q, std::mt19937 &rng)
{
    const std::size_t bit = std::size_t{1} << q;
    double one = 0.0;
    for (std::size_t index = 0; index < psi.size(); ++index)
    {
        if ((index & bit) != 0u)
        {
            one += std::norm(psi[index]);
        }
    }
    const bool outcome = std::uniform_real_distribution<double>(0.0, 1.0)(rng) < one;
    const double scale = 1.0 / std::sqrt(outcome ? one : 1.0 - one);
    for (std::size_t index = 0; index < psi.size(); ++index)
    {
        psi[index] = (((index & bit) != 0u) == outcome) ? psi[index] * scale : Complex(0.0, 0.0);
    }
}

// What the fermionic two-mode RDM of (a, b) says, through the production
// reduction (gate_trace::reduce_pair, the Jordan-Wigner sign included) and
// the production metrics (gate_trace::pair_state, the parity-preserving
// closed form), with the generic partial-transpose route beside it.
struct PairReport
{
    double fmn = 0.0;
    double fmn_generic = 0.0;
    double fmi = 0.0;
    double g = 0.0;       // |<c_a^dag c_b>| = |rho[2][1]|
    double f = 0.0;       // |<c_a c_b>| = |rho[3][0]|
    double rho_n = 0.0;   // |p11 - n_a n_b|
    double product = 0.0; // max |rho - rho_a (x) rho_b| entrywise
};

PairReport pair_report(const State &psi, int n, int a, int b)
{
    const gt::Rho4 rho = gt::reduce_pair(psi, n, a, b);
    const gt::PairState state = gt::pair_state(rho);
    const std::array<double, 32> packed = gt::pack(rho);
    const mipt::dist::PairMetrics metrics = mipt::dist::two_party_metrics(packed.data(), true, true);
    PairReport out;
    out.fmn = state.fmn;
    out.fmn_generic = metrics.mn_generic;
    out.fmi = state.fmi;
    out.g = std::abs(rho[2 * 4 + 1]);
    out.f = std::abs(rho[3 * 4 + 0]);
    const double p10 = rho[1 * 4 + 1].real();
    const double p01 = rho[2 * 4 + 2].real();
    const double p11 = rho[3 * 4 + 3].real();
    out.rho_n = std::abs(p11 - (p10 + p11) * (p01 + p11));
    // The single-mode marginals as plain partial traces of the 4x4. A
    // single-mode coherence connects states of different parity, so for a
    // definite-parity state both marginals are diagonal and no reordering
    // sign can enter either of them.
    Complex ra[4]{};
    Complex rb[4]{};
    for (int x = 0; x < 2; ++x)
    {
        for (int y = 0; y < 2; ++y)
        {
            for (int z = 0; z < 2; ++z)
            {
                ra[x * 2 + y] += rho[static_cast<std::size_t>((x + 2 * z) * 4 + (y + 2 * z))];
                rb[x * 2 + y] += rho[static_cast<std::size_t>((z + 2 * x) * 4 + (z + 2 * y))];
            }
        }
    }
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            const Complex factor = ra[(r & 1) * 2 + (c & 1)] * rb[(r >> 1) * 2 + (c >> 1)];
            out.product =
                std::max(out.product, std::abs(rho[static_cast<std::size_t>(r * 4 + c)] - factor));
        }
    }
    return out;
}

double report_branches(const std::string &label, const std::vector<Branch> &branches, int n,
                       int a, int b, std::vector<PairReport> &reports)
{
    double largest = 0.0;
    for (const Branch &branch : branches)
    {
        const PairReport r = pair_report(branch.psi, n, a, b);
        std::printf("  %s record=%s p=%.4f fN=%.6e (generic %.6e) fMI=%.6e |rho-rho_a(x)rho_b|=%.1e\n",
                    label.c_str(), branch.record.empty() ? "-" : branch.record.c_str(),
                    branch.probability, r.fmn, r.fmn_generic, r.fmi, r.product);
        largest = std::max(largest, r.fmn);
        reports.push_back(r);
    }
    return largest;
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

// A balanced even-block mixer: |00> -> (|00> + |11>)/sqrt2, the odd block left
// alone. Two of them make the Bell pairs (A, C1) and (C2, B); a third on
// (C1, C2) is the bridge, and projecting its output onto |00> or |11>
// teleports the entanglement onto (A, B).
GateMatrix4 even_mixer()
{
    const double c = std::sqrt(0.5);
    const GateMatrix2 even{Complex(c, 0.0), Complex(-c, 0.0), Complex(c, 0.0), Complex(c, 0.0)};
    const GateMatrix2 odd{Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};
    return mipt::parity_preserving_matrix(even, odd);
}

std::array<GateMatrix4, 3> haar_gates(std::uint32_t seed)
{
    std::mt19937 rng(seed);
    std::array<GateMatrix4, 3> gates;
    for (GateMatrix4 &gate : gates)
    {
        const GateMatrix2 even = mipt::haar_unitary_2(rng);
        const GateMatrix2 odd = mipt::haar_unitary_2(rng);
        gate = mipt::parity_preserving_matrix(even, odd);
    }
    return gates;
}

// ---------------------------------------------------------------------------
// Hand-built histories
// ---------------------------------------------------------------------------

struct Bond
{
    int a;
    int b;
    double ep = 0.5;
};

using LayerSpec = std::pair<std::vector<Bond>, std::vector<int>>;

LogicalHistory make_history(int n, const std::vector<LayerSpec> &layers)
{
    LogicalHistory history;
    history.reset(n);
    for (const auto &[bonds, measured] : layers)
    {
        LogicalLayer layer;
        layer.measured.assign(static_cast<std::size_t>(n), 0u);
        int index = 0;
        for (const Bond &bond : bonds)
        {
            LogicalBond logical{bond.a, bond.b};
            logical.source_gate_index = index++;
            logical.entangling_power = bond.ep;
            logical.even_block_mixing = 0.5;
            logical.odd_block_mixing = 0.5;
            layer.bonds.push_back(logical);
        }
        for (int mode : measured)
        {
            layer.measured[static_cast<std::size_t>(mode)] = 1u;
        }
        history.layers.push_back(std::move(layer));
    }
    history.valid = true;
    return history;
}

// The GraphPath visiting `nodes` in order, with the index's own edge ids;
// empty if two consecutive nodes are not adjacent.
GraphPath path_through(const ConnectivityIndex &index, const std::vector<std::uint32_t> &nodes)
{
    GraphPath path;
    path.nodes = nodes;
    for (std::size_t k = 0; k + 1 < nodes.size(); ++k)
    {
        bool found = false;
        for (const mipt::dist::GraphArc &arc : index.arcs(nodes[k]))
        {
            if (arc.node == nodes[k + 1])
            {
                path.edges.push_back(arc.edge);
                found = true;
                break;
            }
        }
        if (!found)
        {
            return {};
        }
    }
    return path;
}

// The structural invariants of an ordered path between two final-slice
// endpoints, however it was chosen; canonical paths are also simple.
bool well_formed(const ConnectivityIndex &index, const GraphPath &path, int i, int j)
{
    if (path.edges.empty() || path.nodes.size() != path.edges.size() + 1u ||
        path.nodes.front() != index.final_node(i) || path.nodes.back() != index.final_node(j))
    {
        return false;
    }
    const std::vector<GraphEdge> &edges = index.graph_edges();
    for (std::size_t k = 0; k < path.edges.size(); ++k)
    {
        if (path.edges[k] >= edges.size())
        {
            return false;
        }
        const GraphEdge &edge = edges[path.edges[k]];
        if (!((edge.a == path.nodes[k] && edge.b == path.nodes[k + 1]) ||
              (edge.b == path.nodes[k] && edge.a == path.nodes[k + 1])))
        {
            return false;
        }
    }
    std::vector<std::uint32_t> sorted = path.nodes;
    std::sort(sorted.begin(), sorted.end());
    return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
}

// What any shape of a path between two final-slice nodes must satisfy. It
// starts backward and ends forward, so its turns alternate V, P, ..., V: an
// odd count (or none), one more valley than peaks, and one bridge per peak.
bool shape_consistent(const PathShape &s, int layers)
{
    if (!s.available)
    {
        return false;
    }
    int measured = 0;
    int endpoint = 0;
    int trace_only = 0;
    for (const PathBridge &bridge : s.bridges)
    {
        const bool is_measured = bridge.measured_layer_in >= 0 || bridge.measured_layer_out >= 0;
        if ((bridge.status == BridgeStatus::Measured) != is_measured ||
            (bridge.measured_layer_in >= 0 && bridge.measured_layer_in < bridge.layer) ||
            (bridge.measured_layer_out >= 0 && bridge.measured_layer_out < bridge.layer))
        {
            return false;
        }
        measured += bridge.status == BridgeStatus::Measured ? 1 : 0;
        endpoint += bridge.status == BridgeStatus::EndpointAnchored ? 1 : 0;
        trace_only += bridge.status == BridgeStatus::TraceOnly ? 1 : 0;
    }
    const int bridges = static_cast<int>(s.bridges.size());
    const PathClass expected_class =
        s.turns == 0   ? PathClass::TimeMonotone
        : s.turns == 1 ? PathClass::CommonSource
                       : (trace_only > 0 ? PathClass::ZigzagTraceOnly : PathClass::ZigzagCertified);
    const ConnectorStatus expected_status = bridges == 0          ? ConnectorStatus::NoBridge
                                            : measured == bridges ? ConnectorStatus::AllMeasured
                                            : measured == 0       ? ConnectorStatus::Unmeasured
                                                                  : ConnectorStatus::Mixed;
    const bool alternating =
        s.turns == 0 ? (s.valleys == 0 && s.peaks == 0) : (s.valleys == s.peaks + 1);
    return (s.turns == 0 || s.turns % 2 == 1) && alternating && bridges == s.peaks &&
           measured == s.bridges_measured && endpoint == s.bridges_endpoint &&
           trace_only == s.bridges_trace_only && s.path_class == expected_class &&
           s.connector_status == expected_status && s.temporal_forward == s.temporal_backward &&
           s.max_layer == layers - 1 && s.min_layer >= 0 &&
           s.length == s.unitary + s.temporal_forward + s.temporal_backward &&
           static_cast<int>(s.gate_layers.size()) == s.unitary &&
           (s.cancelled_gates == -1 || (s.cancelled_gates >= 0 && s.cancelled_gates <= s.unitary));
}

// ---------------------------------------------------------------------------
// Independent restatements: the graph, the cones, the doubled network, walks
// ---------------------------------------------------------------------------

struct RefEdge
{
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    bool gate = false;
    int layer = 0;
};

struct RefGraph
{
    int n = 0;
    int layers = 0;
    std::vector<RefEdge> edges;
    std::vector<std::vector<std::pair<std::uint32_t, int>>> adjacency; // (neighbour, edge)
    std::vector<int> next_measured;            // per node: first measuring layer >= t, or -1
    std::vector<std::uint8_t> gate_measured;   // per edge: a connector measured at >= its layer
    std::vector<std::uint8_t> measurement_cone; // R_M
    std::vector<std::vector<std::uint8_t>> endpoint_cones; // L_s, per site
    std::vector<int> mode_of_site;

    int layer_of(std::uint32_t node) const { return static_cast<int>(node) / n; }
    int mode_of(std::uint32_t node) const { return static_cast<int>(node) % n; }
    std::uint32_t final_node(int site) const
    {
        return node_id(layers - 1, mode_of_site[static_cast<std::size_t>(site)], n);
    }
    bool kept(std::uint32_t node, int i, int j) const
    {
        return measurement_cone[node] != 0u || endpoint_cones[static_cast<std::size_t>(i)][node] != 0u ||
               endpoint_cones[static_cast<std::size_t>(j)][node] != 0u;
    }
};

// Close a set of nodes backward in time, straight from the history: (t, m)
// reaches (t-1, m) when m is unmeasured in layer t-1, and its gate partners in
// layer t (gates act before that layer's measurements).
void close_cone(const LogicalHistory &history, std::vector<std::uint8_t> &cone,
                std::vector<std::uint32_t> stack)
{
    const int n = history.n;
    while (!stack.empty())
    {
        const std::uint32_t v = stack.back();
        stack.pop_back();
        const int t = static_cast<int>(v) / n;
        const int m = static_cast<int>(v) % n;
        auto reach = [&](std::uint32_t u) {
            if (cone[u] == 0u)
            {
                cone[u] = 1u;
                stack.push_back(u);
            }
        };
        if (t > 0 && history.layers[static_cast<std::size_t>(t - 1)].measured[static_cast<std::size_t>(m)] == 0u)
        {
            reach(node_id(t - 1, m, n));
        }
        for (const LogicalBond &bond : history.layers[static_cast<std::size_t>(t)].bonds)
        {
            if (bond.a == m && bond.b != m)
            {
                reach(node_id(t, bond.b, n));
            }
            if (bond.b == m && bond.a != m)
            {
                reach(node_id(t, bond.a, n));
            }
        }
    }
}

RefGraph reference_graph(const LogicalHistory &history)
{
    RefGraph g;
    g.n = history.n;
    g.layers = static_cast<int>(history.layers.size());
    const std::size_t nodes = static_cast<std::size_t>(g.n) * static_cast<std::size_t>(g.layers);
    g.adjacency.assign(nodes, {});
    auto add = [&](std::uint32_t a, std::uint32_t b, bool gate, int layer) {
        const int id = static_cast<int>(g.edges.size());
        g.edges.push_back({a, b, gate, layer});
        g.adjacency[a].push_back({b, id});
        g.adjacency[b].push_back({a, id});
    };
    for (int t = 0; t < g.layers; ++t)
    {
        const LogicalLayer &layer = history.layers[static_cast<std::size_t>(t)];
        for (const LogicalBond &bond : layer.bonds)
        {
            if (bond.a != bond.b)
            {
                add(node_id(t, bond.a, g.n), node_id(t, bond.b, g.n), true, t);
            }
        }
        if (t + 1 < g.layers)
        {
            for (int m = 0; m < g.n; ++m)
            {
                if (layer.measured[static_cast<std::size_t>(m)] == 0u)
                {
                    add(node_id(t, m, g.n), node_id(t + 1, m, g.n), false, t);
                }
            }
        }
    }
    // Forward scans rather than the core's single backward sweep.
    g.next_measured.assign(nodes, -1);
    for (int t = 0; t < g.layers; ++t)
    {
        for (int m = 0; m < g.n; ++m)
        {
            for (int later = t; later < g.layers; ++later)
            {
                if (history.layers[static_cast<std::size_t>(later)].measured[static_cast<std::size_t>(m)] != 0u)
                {
                    g.next_measured[node_id(t, m, g.n)] = later;
                    break;
                }
            }
        }
    }
    g.gate_measured.assign(g.edges.size(), 0u);
    for (std::size_t e = 0; e < g.edges.size(); ++e)
    {
        const RefEdge &edge = g.edges[e];
        if (edge.gate)
        {
            g.gate_measured[e] = (g.next_measured[edge.a] >= 0 || g.next_measured[edge.b] >= 0) ? 1u : 0u;
        }
    }
    g.measurement_cone.assign(nodes, 0u);
    std::vector<std::uint32_t> seeds;
    for (int t = 0; t < g.layers; ++t)
    {
        for (int m = 0; m < g.n; ++m)
        {
            if (history.layers[static_cast<std::size_t>(t)].measured[static_cast<std::size_t>(m)] != 0u)
            {
                g.measurement_cone[node_id(t, m, g.n)] = 1u;
                seeds.push_back(node_id(t, m, g.n));
            }
        }
    }
    close_cone(history, g.measurement_cone, seeds);
    g.mode_of_site = history.mode_at_site;
    g.endpoint_cones.assign(static_cast<std::size_t>(g.n), std::vector<std::uint8_t>(nodes, 0u));
    for (int site = 0; site < g.n; ++site)
    {
        auto &cone = g.endpoint_cones[static_cast<std::size_t>(site)];
        cone[g.final_node(site)] = 1u;
        close_cone(history, cone, {g.final_node(site)});
    }
    return g;
}

// The two graphs must carry one edge set: same endpoints, kind and layer.
bool same_edge_set(const RefGraph &ref, const ConnectivityIndex &index)
{
    using Key = std::tuple<std::uint32_t, std::uint32_t, bool, int>;
    std::vector<Key> a;
    std::vector<Key> b;
    for (const RefEdge &e : ref.edges)
    {
        a.emplace_back(std::min(e.a, e.b), std::max(e.a, e.b), e.gate, e.layer);
    }
    for (const GraphEdge &e : index.graph_edges())
    {
        b.emplace_back(std::min(e.a, e.b), std::max(e.a, e.b), e.kind == GraphEdgeKind::Unitary, e.layer);
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

// The doubled network: temporal edges plus the gates in R_M u L_i u L_j.
bool reference_doubled(const RefGraph &g, int i, int j)
{
    std::vector<std::uint32_t> parent(g.adjacency.size());
    std::iota(parent.begin(), parent.end(), 0u);
    std::function<std::uint32_t(std::uint32_t)> find = [&](std::uint32_t v) {
        return parent[v] == v ? v : parent[v] = find(parent[v]);
    };
    for (const RefEdge &edge : g.edges)
    {
        if (!edge.gate || g.kept(edge.a, i, j) || g.kept(edge.b, i, j))
        {
            parent[find(edge.a)] = find(edge.b);
        }
    }
    return find(g.final_node(i)) == find(g.final_node(j));
}

int reference_cancelled(const RefGraph &g, const ConnectivityIndex &index, const GraphPath &path, int i, int j)
{
    int cancelled = 0;
    for (std::uint32_t e : path.edges)
    {
        const GraphEdge &edge = index.graph_edges()[e];
        if (edge.kind == GraphEdgeKind::Unitary && !g.kept(edge.a, i, j) && !g.kept(edge.b, i, j))
        {
            ++cancelled;
        }
    }
    return cancelled;
}

// The walk search restated over (node, edge just used, last temporal
// direction): a walk may take any edge except the one it arrived by. The core
// encodes the same rule as (direction, arrived-by-gate), which is equivalent
// only because a node has at most one gate per layer -- so agreement here is
// also a check of that equivalence. Fewest turns by a layered closure (all
// turn-free moves from level k, then one turn to level k+1), not a 0-1 deque.
struct RefWalks
{
    std::vector<int> min_turns;          // per node, -1 unreachable
    std::vector<std::uint8_t> certified; // per node
};

RefWalks reference_walks(const RefGraph &g, std::uint32_t source)
{
    const std::size_t nodes = g.adjacency.size();
    std::size_t width = 1;
    for (const auto &arcs : g.adjacency)
    {
        width = std::max(width, arcs.size() + 1u);
    }
    auto encode = [&](std::uint32_t node, int in_slot, int direction) {
        return (static_cast<std::size_t>(node) * width + static_cast<std::size_t>(in_slot + 1)) * 3u +
               static_cast<std::size_t>(direction);
    };
    std::vector<std::vector<int>> back(nodes);
    for (std::size_t v = 0; v < nodes; ++v)
    {
        for (const auto &[w, e] : g.adjacency[v])
        {
            int slot = -1;
            for (std::size_t k = 0; k < g.adjacency[w].size(); ++k)
            {
                if (g.adjacency[w][k].second == e)
                {
                    slot = static_cast<int>(k);
                }
            }
            back[v].push_back(slot);
        }
    }
    // visit(next state, turns added, admissible for a certified walk)
    auto successors = [&](std::size_t state, const std::function<void(std::size_t, int, bool)> &visit) {
        const std::uint32_t v = static_cast<std::uint32_t>(state / 3u / width);
        const int in_slot = static_cast<int>((state / 3u) % width) - 1;
        const int direction = static_cast<int>(state % 3u);
        for (std::size_t k = 0; k < g.adjacency[v].size(); ++k)
        {
            if (static_cast<int>(k) == in_slot)
            {
                continue;
            }
            const auto [w, e] = g.adjacency[v][k];
            int next_direction = direction;
            int cost = 0;
            bool admissible = true;
            if (!g.edges[static_cast<std::size_t>(e)].gate)
            {
                next_direction = g.layer_of(w) > g.layer_of(v) ? 2 : 1;
                cost = direction != 0 && next_direction != direction ? 1 : 0;
                if (direction == 2 && next_direction == 1)
                {
                    // A peak, and the edge just used is its bridge.
                    const int bridge = in_slot >= 0 ? g.adjacency[v][static_cast<std::size_t>(in_slot)].second : -1;
                    admissible = bridge >= 0 && g.edges[static_cast<std::size_t>(bridge)].gate &&
                                 g.gate_measured[static_cast<std::size_t>(bridge)] != 0u;
                }
            }
            visit(encode(w, back[v][k], next_direction), cost, admissible);
        }
    };
    const std::size_t states = nodes * width * 3u;
    const std::size_t start = encode(source, -1, 0);
    std::vector<int> level(states, -1);
    level[start] = 0;
    std::vector<std::size_t> frontier{start};
    for (int turns = 0; !frontier.empty(); ++turns)
    {
        std::vector<std::size_t> members;
        std::vector<std::size_t> stack = frontier;
        while (!stack.empty())
        {
            const std::size_t s = stack.back();
            stack.pop_back();
            members.push_back(s);
            successors(s, [&](std::size_t next, int cost, bool) {
                if (cost == 0 && level[next] < 0)
                {
                    level[next] = turns;
                    stack.push_back(next);
                }
            });
        }
        std::vector<std::size_t> next_frontier;
        for (std::size_t s : members)
        {
            successors(s, [&](std::size_t next, int cost, bool) {
                if (cost == 1 && level[next] < 0)
                {
                    level[next] = turns + 1;
                    next_frontier.push_back(next);
                }
            });
        }
        frontier = std::move(next_frontier);
    }
    std::vector<std::uint8_t> seen(states, 0u);
    seen[start] = 1u;
    std::vector<std::size_t> stack{start};
    while (!stack.empty())
    {
        const std::size_t s = stack.back();
        stack.pop_back();
        successors(s, [&](std::size_t next, int, bool admissible) {
            if (admissible && seen[next] == 0u)
            {
                seen[next] = 1u;
                stack.push_back(next);
            }
        });
    }
    RefWalks out;
    out.min_turns.assign(nodes, -1);
    out.certified.assign(nodes, 0u);
    for (std::size_t s = 0; s < states; ++s)
    {
        const std::size_t v = s / 3u / width;
        if (level[s] >= 0 && (out.min_turns[v] < 0 || level[s] < out.min_turns[v]))
        {
            out.min_turns[v] = level[s];
        }
        out.certified[v] |= seen[s];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Contract constants
// ---------------------------------------------------------------------------

// The record format, the CSVs and data_analysis all read these by value, so
// they are pinned rather than assumed.
void test_contract_constants()
{
    using namespace mipt::dist;
    expect(TURN_DIAGNOSTICS_VERSION == 1, "contract: TURN_DIAGNOSTICS_VERSION is 1");
    expect(FLAG_CONNECTED_COMMON_SOURCE == (1u << 7) && FLAG_CONNECTED_CERTIFIED == (1u << 8) &&
               FLAG_CONNECTED_DOUBLED == (1u << 9),
           "contract: the turn flags are bits 7, 8 and 9");
    const std::string definition = TURN_DIAGNOSTICS_DEFINITION;
    expect(!definition.empty() && definition.find('\n') == std::string::npos &&
               definition.find('"') == std::string::npos,
           "contract: the definition is one line with no quote, so it survives a header and a quoted CSV cell");
    expect(static_cast<int>(PathClass::TimeMonotone) == 0 && static_cast<int>(PathClass::CommonSource) == 1 &&
               static_cast<int>(PathClass::ZigzagCertified) == 2 &&
               static_cast<int>(PathClass::ZigzagTraceOnly) == 3,
           "contract: the stored class codes");
    expect(std::string(path_class_name(PathClass::TimeMonotone)) == "time_monotone" &&
               std::string(path_class_name(PathClass::CommonSource)) == "common_source" &&
               std::string(path_class_name(PathClass::ZigzagCertified)) == "zigzag_certified" &&
               std::string(path_class_name(PathClass::ZigzagTraceOnly)) == "zigzag_trace_only",
           "contract: class names");
    expect(std::string(bridge_status_name(BridgeStatus::Measured)) == "measured" &&
               std::string(bridge_status_name(BridgeStatus::EndpointAnchored)) == "endpoint_anchored" &&
               std::string(bridge_status_name(BridgeStatus::TraceOnly)) == "trace_only",
           "contract: bridge status names");
    expect(std::string(connector_status_name(ConnectorStatus::NoBridge)) == "no_bridge" &&
               std::string(connector_status_name(ConnectorStatus::AllMeasured)) == "all_measured" &&
               std::string(connector_status_name(ConnectorStatus::Mixed)) == "mixed" &&
               std::string(connector_status_name(ConnectorStatus::Unmeasured)) == "unmeasured",
           "contract: connector status names");
    PairTurns all;
    all.connected_common_source = all.connected_certified = all.connected_doubled = true;
    expect(all.flags() == ((1u << 7) | (1u << 8) | (1u << 9)), "contract: PairTurns::flags carries bits 7-9");
    expect(PairTurns{}.flags() == 0u && PathShape{}.cancelled_gates == -1,
           "contract: defaults are empty flags and an uncomputed cancelled count");
}

// ---------------------------------------------------------------------------
// Part 1: the four-mode swap circuit, graph and state side by side
// ---------------------------------------------------------------------------

HostCircuit four_mode_circuit(const std::array<GateMatrix4, 3> &g, const std::vector<int> &measured_0,
                              const std::vector<int> &measured_1, const std::vector<int> &measured_2 = {})
{
    HostCircuit circuit(3);
    circuit[0].gates = {HostGate{0, 1, g[0]}, HostGate{2, 3, g[1]}};
    circuit[0].measured = measured_0;
    circuit[1].gates = {HostGate{1, 2, g[2]}};
    circuit[1].measured = measured_1;
    circuit[2].measured = measured_2; // layer 2 holds no gate: the final slice
    return circuit;
}

// The one A-B path of the four-mode circuit, counted by hand:
//
//   (2,A) B (1,A) B (0,A) -G1- (0,C1) F (1,C1) -G3- (1,C2) B (0,C2) -G2- (0,B) F (1,B) F (2,B)
//
// six temporal edges (three each way), three gates in layers (0, 1, 0), and
// the direction sequence B B F B F F: a valley at G1, a peak at G3, a valley
// at G2. G3 is the bridge, climbed on C1 and descended on C2.
void expect_four_mode_zigzag(const ConnectivityIndex &index, const PathShape &shape, PathClass cls,
                             ConnectorStatus status, BridgeStatus bridge_status, int cancelled,
                             const std::string &where)
{
    expect(shape.available, where + ": the canonical shape is available");
    expect(shape.length == 9 && shape.unitary == 3, where + ": nine edges, three of them gates");
    expect(shape.temporal_forward == 3 && shape.temporal_backward == 3,
           where + ": three temporal steps each way");
    expect(shape.turns == 3 && shape.valleys == 2 && shape.peaks == 1,
           where + ": three turns -- valley, peak, valley");
    expect(shape.min_layer == 0 && shape.max_layer == 2, where + ": it spans layers 0..2");
    expect(shape.gate_layers == std::vector<int>({0, 1, 0}), where + ": gate layers in path order (0, 1, 0)");
    expect(shape.bridges.size() == 1, where + ": one bridge");
    if (shape.bridges.size() == 1)
    {
        const PathBridge &bridge = shape.bridges[0];
        const GraphEdge &edge = index.graph_edges()[bridge.edge];
        expect(edge.kind == GraphEdgeKind::Unitary && edge.layer == 1 && edge.source_gate_index == 0,
               where + ": the bridge is G3, the only gate of layer 1");
        expect(bridge.layer == 1 && bridge.mode_in == 1 && bridge.mode_out == 2,
               where + ": climbed on C1, descended on C2");
        expect(bridge.status == bridge_status, where + ": bridge status");
    }
    expect(shape.path_class == cls, where + ": path class");
    expect(shape.connector_status == status, where + ": connector status");
    expect(shape.cancelled_gates == cancelled, where + ": cancelled gates on the path");
}

void test_four_mode_swap(const std::string &tag, const std::array<GateMatrix4, 3> &gates)
{
    constexpr int n = 4;
    const std::string where = "four-mode [" + tag + "]";
    const std::vector<std::uint32_t> zigzag = {
        node_id(2, 0, n), node_id(1, 0, n), node_id(0, 0, n), node_id(0, 1, n), node_id(1, 1, n),
        node_id(1, 2, n), node_id(0, 2, n), node_id(0, 3, n), node_id(1, 3, n), node_id(2, 3, n)};
    std::vector<PairReport> reports;

    // (a) Nothing measured. The bridge is trace-only: C1 and C2 are traced out
    // unconditioned after G3, so Tr_{C1C2}[G3 rho G3^dag] = Tr_{C1C2} rho and
    // (A, B) inherits the product of (A C1) | (C2 B) from layer 0.
    {
        const HostCircuit circuit = four_mode_circuit(gates, {}, {});
        const LogicalHistory history = history_of(n, circuit);
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const std::string what = where + " unmeasured";
        expect(index.has_gate_metrics(), what + ": bonds carry their gates' metrics");
        const PairConnectivity conn = index.query(0, 3);
        expect(conn.connected && conn.shortest_path == 9, what + ": the graph connects A and B, 9 hops");
        const PairTurns turns = index.pair_turns(0, 3);
        expect(turns.evaluated && turns.min_turns == 3, what + ": no walk needs fewer than 3 turns");
        expect(!turns.common_source && !turns.certified && !turns.doubled_network,
               what + ": neither a common source, nor certified, nor in the doubled network");
        expect(!turns.connected_common_source && !turns.connected_certified && !turns.connected_doubled &&
                   turns.flags() == 0u,
               what + ": so no refined event and no turn flag");
        expect(index.shortest_maximin_trace(0, 3).nodes == zigzag &&
                   index.widest_gate_trace(0, 3).nodes == zigzag,
               what + ": both canonical paths are the hand-counted zig-zag");
        expect_four_mode_zigzag(index, turns.shortest_maximin, PathClass::ZigzagTraceOnly,
                                ConnectorStatus::Unmeasured, BridgeStatus::TraceOnly, 1,
                                what + " shortest-maximin");
        expect_four_mode_zigzag(index, turns.widest, PathClass::ZigzagTraceOnly, ConnectorStatus::Unmeasured,
                                BridgeStatus::TraceOnly, 1, what + " widest");
        expect(!index.doubled_relevant(node_id(1, 1, n), 0, 3) && !index.doubled_relevant(node_id(1, 2, n), 0, 3) &&
                   index.doubled_relevant(node_id(0, 1, n), 0, 3) && index.doubled_relevant(node_id(0, 2, n), 0, 3),
               what + ": G3's nodes are outside every cone, G1's and G2's are inside the endpoints'");
        expect(index.next_measurement(1, 1) == -1 && index.next_measurement(0, 2) == -1,
               what + ": no connector is ever measured");
        const PairTurns mirror = index.pair_turns(3, 0);
        expect(mirror.min_turns == 3 && !mirror.certified && !mirror.doubled_network,
               what + ": the reversed pair agrees");
        std::vector<std::uint32_t> reversed(zigzag.rbegin(), zigzag.rend());
        expect(index.shortest_maximin_trace(3, 0).nodes == reversed && mirror.shortest_maximin.bridges.size() == 1 &&
                   mirror.shortest_maximin.bridges[0].mode_in == 2 &&
                   mirror.shortest_maximin.bridges[0].mode_out == 1,
               what + ": from B the same path climbs C2 and descends C1");

        const std::vector<Branch> branches = measurement_branches(n, circuit);
        expect(branches.size() == 1, what + ": one record, the empty one");
        report_branches(what + ":", branches, n, 0, 3, reports);
        const PairReport &r = reports.back();
        expect(r.fmn <= 1.0e-15 && r.fmi <= 1.0e-14 && r.g <= 1.0e-15 && r.f <= 1.0e-15 && r.rho_n <= 1.0e-15,
               what + ": fN and I(A:B) vanish to rounding");
        expect(r.product <= 1.0e-15, what + ": rho_AB = rho_A (x) rho_B entrywise");
    }

    // (b) Both connectors measured after G3. The record conditions on the
    // bridge's output, which is entanglement swapping: G3 then a projection of
    // (C1, C2) onto an even-sector state teleports the (A C1)(C2 B) pairs into
    // an (A, B) pair. The graph sees a measured bridge.
    double swapped = 0.0;
    {
        const HostCircuit circuit = four_mode_circuit(gates, {}, {1, 2});
        const LogicalHistory history = history_of(n, circuit);
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const std::string what = where + " C1,C2 measured after G3";
        const PairConnectivity conn = index.query(0, 3);
        expect(conn.connected && conn.survives_i && conn.survives_j, what + ": A and B survive and connect");
        const PairTurns turns = index.pair_turns(0, 3);
        expect(turns.min_turns == 3 && !turns.common_source, what + ": still a three-turn zig-zag");
        expect(turns.certified && turns.connected_certified, what + ": certified, and connected_certified");
        expect(turns.doubled_network && turns.connected_doubled, what + ": the doubled network keeps G3");
        expect(turns.flags() == (mipt::dist::FLAG_CONNECTED_CERTIFIED | mipt::dist::FLAG_CONNECTED_DOUBLED),
               what + ": flags are certified and doubled, not common source");
        expect(index.shortest_maximin_trace(0, 3).nodes == zigzag, what + ": the path is unchanged");
        expect_four_mode_zigzag(index, turns.shortest_maximin, PathClass::ZigzagCertified,
                                ConnectorStatus::AllMeasured, BridgeStatus::Measured, 0,
                                what + " shortest-maximin");
        if (turns.shortest_maximin.bridges.size() == 1)
        {
            expect(turns.shortest_maximin.bridges[0].measured_layer_in == 1 &&
                       turns.shortest_maximin.bridges[0].measured_layer_out == 1,
                   what + ": both connectors are measured in the bridge's own layer");
        }
        expect(index.next_measurement(0, 1) == 1 && index.next_measurement(1, 2) == 1,
               what + ": next_measurement finds layer 1 from layers 0 and 1");
        const std::vector<Branch> branches = measurement_branches(n, circuit);
        swapped = report_branches(what + ":", branches, n, 0, 3, reports);
        expect(swapped > 1.0e-3, what + ": some record leaves (A, B) entangled");
    }

    // (c) Only C1 measured after G3. The graph calls the bridge measured all
    // the same (one connector suffices for the definition); whether the record
    // of one connector suffices physically is reported, not asserted.
    std::vector<double> single;
    {
        const HostCircuit circuit = four_mode_circuit(gates, {}, {1});
        const LogicalHistory history = history_of(n, circuit);
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const std::string what = where + " C1 measured after G3";
        const PairTurns turns = index.pair_turns(0, 3);
        expect(turns.certified && turns.connected_certified && turns.doubled_network && turns.connected_doubled,
               what + ": certified and doubled on one connector");
        expect_four_mode_zigzag(index, turns.shortest_maximin, PathClass::ZigzagCertified,
                                ConnectorStatus::AllMeasured, BridgeStatus::Measured, 0,
                                what + " shortest-maximin");
        if (turns.shortest_maximin.bridges.size() == 1)
        {
            expect(turns.shortest_maximin.bridges[0].measured_layer_in == 1 &&
                       turns.shortest_maximin.bridges[0].measured_layer_out == -1,
                   what + ": C1 measured at layer 1, C2 never");
        }
        const std::vector<Branch> branches = measurement_branches(n, circuit);
        const std::size_t first = reports.size();
        report_branches(what + ":", branches, n, 0, 3, reports);
        for (std::size_t k = first; k < reports.size(); ++k)
        {
            single.push_back(reports[k].fmn);
        }
    }

    // (d) The same single measurement deferred to the idle final layer. No
    // gate touches C1 in between, so it is the same instrument: the graph must
    // still call the bridge measured (at layer 2 >= 1) and the records must
    // carry the same fN.
    {
        const HostCircuit circuit = four_mode_circuit(gates, {}, {}, {1});
        const LogicalHistory history = history_of(n, circuit);
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const std::string what = where + " C1 measured in the final layer";
        const PairTurns turns = index.pair_turns(0, 3);
        expect(turns.certified && turns.doubled_network && turns.shortest_maximin.bridges.size() == 1 &&
                   turns.shortest_maximin.bridges[0].measured_layer_in == 2 &&
                   turns.shortest_maximin.cancelled_gates == 0,
               what + ": measured at layer 2 still certifies the layer-1 bridge");
        const std::vector<Branch> branches = measurement_branches(n, circuit);
        bool same = branches.size() == single.size();
        for (std::size_t k = 0; same && k < branches.size(); ++k)
        {
            same = std::abs(pair_report(branches[k].psi, n, 0, 3).fmn - single[k]) <= 1.0e-14;
        }
        expect(same, what + ": the records carry the same fN as measuring right after G3");
    }

    // (e) C1 measured *before* G3, between G1 and G3. The measurement removes
    // the edge (0, C1) -> (1, C1), so the zig-zag no longer exists; physically
    // the record fixes A's occupation (G1 only correlates n_A with n_C1), so A
    // is pure and (A, B) is a product on every record.
    {
        const HostCircuit circuit = four_mode_circuit(gates, {1}, {});
        const LogicalHistory history = history_of(n, circuit);
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const std::string what = where + " C1 measured before G3";
        const PairConnectivity conn = index.query(0, 3);
        expect(!conn.interior_path && !conn.connected, what + ": the path is cut");
        const PairTurns turns = index.pair_turns(0, 3);
        expect(turns.evaluated && turns.min_turns == -1 && !turns.common_source && !turns.certified &&
                   !turns.doubled_network && turns.flags() == 0u,
               what + ": no walk, nothing certified, nothing doubled");
        expect(!turns.shortest_maximin.available && !turns.widest.available &&
                   turns.shortest_maximin.cancelled_gates == -1,
               what + ": and no canonical shape");
        const std::vector<Branch> branches = measurement_branches(n, circuit);
        const std::size_t first = reports.size();
        report_branches(what + ":", branches, n, 0, 3, reports);
        bool all_zero = branches.size() == 2;
        for (std::size_t k = first; k < reports.size(); ++k)
        {
            all_zero = all_zero && reports[k].fmn <= 1.0e-15 && reports[k].product <= 1.0e-15;
        }
        expect(all_zero, what + ": fN(A:B) = 0 and rho_AB a product on both records");
    }
}

// ---------------------------------------------------------------------------
// Part 2: hand-built shapes
// ---------------------------------------------------------------------------

// V shapes: one backward-to-forward turn, the source gate in both endpoints'
// past cones, no bridge. Also a V with two gates on its descending side, which
// is still one turn, and a dead endpoint, which keeps every existence flag but
// loses every refined event.
void test_v_shapes()
{
    {
        const LogicalHistory history = make_history(2, {{{{0, 1, 0.6}}, {}}, {{}, {}}, {{}, {}}});
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const PairTurns t = index.pair_turns(0, 1);
        expect(t.min_turns == 1 && t.common_source && t.certified && t.doubled_network,
               "V: one turn, a common source, certified and doubled");
        expect(t.connected_common_source && t.connected_certified && t.connected_doubled &&
                   t.flags() == ((1u << 7) | (1u << 8) | (1u << 9)),
               "V: every refined event and all three flags");
        const PathShape &s = t.shortest_maximin;
        expect(s.available && s.turns == 1 && s.valleys == 1 && s.peaks == 0 && s.bridges.empty(),
               "V: one valley, no peak, no bridge");
        expect(s.path_class == PathClass::CommonSource && s.connector_status == ConnectorStatus::NoBridge,
               "V: class common_source, no bridge");
        expect(s.length == 5 && s.temporal_forward == 2 && s.temporal_backward == 2 && s.min_layer == 0 &&
                   s.max_layer == 2 && s.gate_layers == std::vector<int>({0}) && s.cancelled_gates == 0,
               "V: two steps down, the gate, two steps up; the source gate survives the doubling");
    }
    {
        // layer 0: (1,2); layer 1: (0,1); final layer idle. (2,0) B (1,0) -G- (1,1)
        // B (0,1) -G- (0,2) F (1,2) F (2,2): directions B B F F.
        const LogicalHistory history = make_history(3, {{{{1, 2}}, {}}, {{{0, 1}}, {}}, {{}, {}}});
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const PairTurns t = index.pair_turns(0, 2);
        const PathShape &s = t.shortest_maximin;
        expect(t.min_turns == 1 && s.turns == 1 && s.path_class == PathClass::CommonSource,
               "V with two descending gates: still one turn");
        expect(s.gate_layers == std::vector<int>({1, 0}) && s.min_layer == 0 && s.cancelled_gates == 0,
               "V with two descending gates: gates in path order (1, 0)");
    }
    {
        // A dead endpoint: the walk and the doubled network exist, the refined
        // events cannot, because C itself fails on survival.
        const LogicalHistory history = make_history(2, {{{{0, 1}}, {}}, {{}, {1}}});
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const PairTurns t = index.pair_turns(0, 1);
        expect(!index.query(0, 1).connected && t.common_source && t.certified && t.doubled_network,
               "dead endpoint: existence holds while C fails");
        expect(!t.connected_common_source && !t.connected_certified && !t.connected_doubled && t.flags() == 0u,
               "dead endpoint: no refined event, no flag");
    }
}

// A single final-layer gate is the only 0-turn shape. With a weak final gate
// and a strong layer-0 gate, the two canonical paths differ in shape while
// min_turns, which is path-independent, stays 0.
void test_final_layer_gate()
{
    {
        const LogicalHistory history = make_history(2, {{{{0, 1}}, {}}});
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const PairTurns t = index.pair_turns(0, 1);
        const PathShape &s = t.shortest_maximin;
        expect(t.min_turns == 0 && t.common_source && t.certified && t.doubled_network,
               "final-layer gate: zero turns");
        expect(s.length == 1 && s.unitary == 1 && s.turns == 0 && s.temporal_forward == 0 &&
                   s.temporal_backward == 0 && s.min_layer == 0 && s.max_layer == 0 &&
                   s.path_class == PathClass::TimeMonotone && s.cancelled_gates == 0,
               "final-layer gate: one gate, time_monotone");
    }
    {
        const LogicalHistory history = make_history(2, {{{{0, 1, 0.9}}, {}}, {{{0, 1, 0.1}}, {}}});
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const PairTurns t = index.pair_turns(0, 1);
        expect(t.min_turns == 0, "weak final gate: min_turns is the direct gate's 0");
        expect(t.shortest_maximin.path_class == PathClass::TimeMonotone &&
                   t.shortest_maximin.gate_layers == std::vector<int>({1}),
               "weak final gate: the shortest path is the weak direct gate");
        expect(t.widest.path_class == PathClass::CommonSource && t.widest.gate_layers == std::vector<int>({0}) &&
                   t.widest.temporal_forward == 1 && t.widest.temporal_backward == 1,
               "weak final gate: the widest path is the V down to the strong gate");
    }
}

// Six modes in a line, three layer-0 pairs and two layer-1 bridges:
//
//   (2,0) B (1,0) B (0,0) -g- (0,1) F (1,1) =X= (1,2) B (0,2) -g- (0,3) F (1,3) =Y= (1,4) B (0,4) -g- (0,5) F (1,5) F (2,5)
//
// thirteen edges, directions B B F B F B F F: five turns, three valleys, two
// peaks. Measuring one connector of X and none of Y makes X measured and Y
// trace-only, and a single trace-only bridge is enough to cut the path --
// which the host state confirms: (A, B) comes out an exact product.
void test_double_zigzag()
{
    constexpr int n = 6;
    const GateMatrix4 g = even_mixer();
    const std::vector<std::uint32_t> expected_nodes = {
        node_id(2, 0, n), node_id(1, 0, n), node_id(0, 0, n), node_id(0, 1, n), node_id(1, 1, n),
        node_id(1, 2, n), node_id(0, 2, n), node_id(0, 3, n), node_id(1, 3, n), node_id(1, 4, n),
        node_id(0, 4, n), node_id(0, 5, n), node_id(1, 5, n), node_id(2, 5, n)};
    struct Case
    {
        std::vector<int> measured;
        PathClass cls;
        ConnectorStatus status;
        BridgeStatus x;
        BridgeStatus y;
        int cancelled;
        bool certified;
        const char *what;
    };
    const Case cases[] = {
        {{1}, PathClass::ZigzagTraceOnly, ConnectorStatus::Mixed, BridgeStatus::Measured, BridgeStatus::TraceOnly, 1, false, "C1 measured"},
        {{}, PathClass::ZigzagTraceOnly, ConnectorStatus::Unmeasured, BridgeStatus::TraceOnly, BridgeStatus::TraceOnly, 3, false, "nothing measured"},
        {{3}, PathClass::ZigzagTraceOnly, ConnectorStatus::Mixed, BridgeStatus::TraceOnly, BridgeStatus::Measured, 1, false, "C3 measured"},
        {{1, 3}, PathClass::ZigzagCertified, ConnectorStatus::AllMeasured, BridgeStatus::Measured, BridgeStatus::Measured, 0, true, "C1 and C3 measured"},
        {{1, 2, 3, 4}, PathClass::ZigzagCertified, ConnectorStatus::AllMeasured, BridgeStatus::Measured, BridgeStatus::Measured, 0, true, "all four connectors measured"},
    };
    for (const Case &c : cases)
    {
        HostCircuit circuit(3);
        circuit[0].gates = {HostGate{0, 1, g}, HostGate{2, 3, g}, HostGate{4, 5, g}};
        circuit[1].gates = {HostGate{1, 2, g}, HostGate{3, 4, g}};
        circuit[1].measured = c.measured;
        const LogicalHistory history = history_of(n, circuit);
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const std::string what = std::string("double zig-zag, ") + c.what;
        const PairTurns t = index.pair_turns(0, 5);
        const PathShape &s = t.shortest_maximin;
        expect(index.query(0, 5).connected && t.min_turns == 5, what + ": connected, five turns at least");
        expect(index.shortest_maximin_trace(0, 5).nodes == expected_nodes, what + ": the hand-counted path");
        expect(s.length == 13 && s.unitary == 5 && s.temporal_forward == 4 && s.temporal_backward == 4 &&
                   s.turns == 5 && s.valleys == 3 && s.peaks == 2,
               what + ": 13 edges, 4 steps each way, 5 turns = 3 valleys + 2 peaks");
        expect(s.gate_layers == std::vector<int>({0, 1, 0, 1, 0}), what + ": gate layers (0, 1, 0, 1, 0)");
        expect(s.bridges.size() == 2 && s.bridges[0].mode_in == 1 && s.bridges[0].mode_out == 2 &&
                   s.bridges[1].mode_in == 3 && s.bridges[1].mode_out == 4 && s.bridges[0].status == c.x &&
                   s.bridges[1].status == c.y,
               what + ": bridges X = (1,2) and Y = (3,4), in path order, with their statuses");
        const int measured = (c.x == BridgeStatus::Measured ? 1 : 0) + (c.y == BridgeStatus::Measured ? 1 : 0);
        expect(s.bridges_measured == measured && s.bridges_trace_only == 2 - measured && s.bridges_endpoint == 0,
               what + ": bridge counts");
        expect(s.path_class == c.cls && s.connector_status == c.status, what + ": class and connector status");
        expect(s.cancelled_gates == c.cancelled, what + ": gates the doubled network cancels on the path");
        expect(t.certified == c.certified && t.doubled_network == c.certified &&
                   t.connected_doubled == c.certified && !t.common_source,
               what + ": certified and doubled exactly when no bridge is trace-only");

        std::vector<PairReport> reports;
        const std::vector<Branch> branches = measurement_branches(n, circuit);
        const bool print = c.certified; // the trace-only cases are asserted below, not printed
        double largest = 0.0;
        double worst_product = 0.0;
        for (const Branch &branch : branches)
        {
            const PairReport r = pair_report(branch.psi, n, 0, 5);
            largest = std::max(largest, r.fmn);
            worst_product = std::max(worst_product, r.product);
            reports.push_back(r);
        }
        if (print)
        {
            reports.clear();
            report_branches(what + ":", branches, n, 0, 5, reports);
        }
        if (!c.certified)
        {
            std::printf("  %s: %zu records, max fN(A:B)=%.3e, max |rho-rho_a(x)rho_b|=%.3e\n", what.c_str(),
                        branches.size(), largest, worst_product);
            expect(largest <= 1.0e-15 && worst_product <= 1.0e-15,
                   what + ": a trace-only bridge leaves (A, B) an exact product on every record");
        }
        if (c.measured.size() == 4)
        {
            expect(largest > 1.0e-3, what + ": two swaps put entanglement on (A, B) for some record");
        }
    }
}

// Existence is a property of the pair, the class a property of one path.
// Route Y (late bridge, trace-only) is the shortest A-B path; route X (early
// bridge, a connector measured right after it) is longer and certified. The
// pair is certified and doubled while its shortest-maximin path is a
// trace-only zig-zag whose bridge the doubled network cancels. Raising the
// X-route gates makes the widest path take X instead.
void test_certified_but_canonical_trace_only()
{
    // modes 0=A 1=C1 2=C2 3=B 4=D1 5=D2
    for (bool strong_x : {false, true})
    {
        const double x = strong_x ? 0.9 : 0.5;
        const double y = strong_x ? 0.2 : 0.5;
        const LogicalHistory history = make_history(6, {
            {{{0, 1, x}, {2, 3, x}}, {}},
            {{{1, 2, x}}, {1}},
            {{{0, 4, y}, {5, 3, y}}, {}},
            {{{4, 5, y}}, {}},
            {{}, {}},
        });
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const std::string what = strong_x ? "two routes, strong X" : "two routes, equal gates";
        const PairTurns t = index.pair_turns(0, 3);
        expect(index.query(0, 3).connected && t.min_turns == 3 && !t.common_source,
               what + ": connected, three turns, no common source");
        expect(t.certified && t.connected_certified && t.doubled_network && t.connected_doubled,
               what + ": certified and doubled through X");
        const PathShape &s = t.shortest_maximin;
        expect(s.length == 9 && s.path_class == PathClass::ZigzagTraceOnly &&
                   s.connector_status == ConnectorStatus::Unmeasured && s.cancelled_gates == 1 &&
                   s.bridges.size() == 1 && s.bridges[0].layer == 3,
               what + ": the shortest-maximin path is the trace-only Y route, its bridge cancelled");
        if (strong_x)
        {
            expect(t.widest.length == 13 && t.widest.path_class == PathClass::ZigzagCertified &&
                       t.widest.connector_status == ConnectorStatus::AllMeasured &&
                       t.widest.cancelled_gates == 0 && t.widest.bridges.size() == 1 &&
                       t.widest.bridges[0].layer == 1,
                   what + ": the widest path is the certified X route");
        }
        else
        {
            expect(same_shape(t.widest, s), what + ": with equal gates the widest path is the shortest one");
        }
    }
}

// An endpoint-anchored bridge: a connector is one of the endpoints' own modes
// and nothing is measured after the bridge. No canonical (shortest) path can
// carry one -- the endpoint's intact worldline is always a strictly shorter
// way round -- so these paths are built by hand and read with path_shape. The
// claim in dist_connectivity.hpp is that a walk through such a bridge reroutes
// up that worldline with no peak there, so the pair is still certified (here,
// even a common source).
void test_endpoint_anchored()
{
    // (1) The bridge's descending connector is endpoint j's mode.
    //   modes 0=A(i) 1=C 2=B(j) 3=D; layer 0: (0,1) (2,3); layer 1: (1,2);
    //   layer 2: (2,3); layer 3 idle.
    {
        constexpr int n = 4;
        const LogicalHistory history = make_history(n, {{{{0, 1}, {2, 3}}, {}}, {{{1, 2}}, {}}, {{{2, 3}}, {}}, {{}, {}}});
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const GraphPath path = path_through(index, {node_id(3, 0, n), node_id(2, 0, n), node_id(1, 0, n),
                                                    node_id(0, 0, n), node_id(0, 1, n), node_id(1, 1, n),
                                                    node_id(1, 2, n), node_id(0, 2, n), node_id(0, 3, n),
                                                    node_id(1, 3, n), node_id(2, 3, n), node_id(2, 2, n),
                                                    node_id(3, 2, n)});
        expect(well_formed(index, path, 0, 2), "anchored at j: the hand-built path is a simple A-B path");
        const PathShape s = index.path_shape(path, 0, 2);
        expect(s.available && s.turns == 3 && s.peaks == 1 && s.bridges.size() == 1,
               "anchored at j: three turns, one peak");
        expect(s.bridges.size() == 1 && s.bridges[0].mode_in == 1 && s.bridges[0].mode_out == 2 &&
                   s.bridges[0].status == BridgeStatus::EndpointAnchored,
               "anchored at j: the bridge descends B's own worldline, unmeasured -> endpoint_anchored");
        expect(s.bridges_endpoint == 1 && s.bridges_measured == 0 && s.bridges_trace_only == 0 &&
                   s.path_class == PathClass::ZigzagCertified &&
                   s.connector_status == ConnectorStatus::Unmeasured && s.cancelled_gates == -1,
               "anchored at j: zigzag_certified, connectors counted unmeasured, cancelled not computed");
        expect(s.gate_layers == std::vector<int>({0, 1, 0, 2}) && s.temporal_forward == 4 &&
                   s.temporal_backward == 4 && shape_consistent(s, 4),
               "anchored at j: gate layers (0, 1, 0, 2), four steps each way");
        const PairTurns t = index.pair_turns(0, 2);
        expect(t.common_source || t.certified, "anchored at j: the walk search still finds the pair");
        expect(t.min_turns == 1 && t.common_source && t.certified && t.doubled_network,
               "anchored at j: rerouted up B's worldline it is a V");
        expect(t.shortest_maximin.bridges_endpoint == 0 && t.shortest_maximin.path_class == PathClass::CommonSource,
               "anchored at j: the canonical path is that V");

        // The same history with B measured in the final layer: the connector
        // is then measured after the bridge, so the bridge is `measured`, and
        // B is dead, so nothing refined survives.
        LogicalHistory dead = history;
        dead.layers[3].measured[2] = 1u;
        ConnectivityIndex dead_index;
        dead_index.build(dead, true, true, true);
        const PathShape d = dead_index.path_shape(path_through(dead_index, path.nodes), 0, 2);
        expect(d.bridges.size() == 1 && d.bridges[0].status == BridgeStatus::Measured &&
                   d.bridges[0].measured_layer_out == 3,
               "anchored at j, B measured last: the bridge becomes measured at layer 3");
        const PairTurns td = dead_index.pair_turns(0, 2);
        expect(td.certified && !td.connected_certified && !td.connected_doubled && !dead_index.query(0, 2).connected,
               "anchored at j, B measured last: certified but not connected");
    }
    // (2) The bridge's climbing connector is endpoint i's mode, on a
    // non-adjacent bond (the graph does not care).
    //   modes 0=A(i) 1=D 2=C 3=B(j); layer 0: (0,1) (2,3); layer 1: (0,2);
    //   layer 2: (0,1); layer 3 idle.
    {
        constexpr int n = 4;
        const LogicalHistory history = make_history(n, {{{{0, 1}, {2, 3}}, {}}, {{{0, 2}}, {}}, {{{0, 1}}, {}}, {{}, {}}});
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const GraphPath path = path_through(index, {node_id(3, 0, n), node_id(2, 0, n), node_id(2, 1, n),
                                                    node_id(1, 1, n), node_id(0, 1, n), node_id(0, 0, n),
                                                    node_id(1, 0, n), node_id(1, 2, n), node_id(0, 2, n),
                                                    node_id(0, 3, n), node_id(1, 3, n), node_id(2, 3, n),
                                                    node_id(3, 3, n)});
        expect(well_formed(index, path, 0, 3), "anchored at i: the hand-built path is a simple A-B path");
        const PathShape s = index.path_shape(path, 0, 3);
        expect(s.available && s.turns == 3 && s.bridges.size() == 1 && s.bridges[0].mode_in == 0 &&
                   s.bridges[0].mode_out == 2 && s.bridges[0].status == BridgeStatus::EndpointAnchored &&
                   s.path_class == PathClass::ZigzagCertified && shape_consistent(s, 4),
               "anchored at i: the bridge climbs A's own worldline -> endpoint_anchored, zigzag_certified");
        const PairTurns t = index.pair_turns(0, 3);
        expect(t.min_turns == 1 && t.common_source && t.certified && t.doubled_network,
               "anchored at i: the walk goes straight down A's worldline instead");
    }
}

// Malformed paths, missing metrics, separate components, self pairs and an
// unbuilt index all come back unavailable or unevaluated -- never as a zero
// that would read as a measurement.
void test_unavailable_and_inert()
{
    {
        ConnectivityIndex unbuilt;
        expect(!unbuilt.has_turns() && !unbuilt.pair_turns(0, 1).evaluated,
               "inert: an unbuilt index evaluates nothing");
    }
    {
        // make_history-style bonds without metrics: existence is a graph fact,
        // the canonical paths need gate metrics.
        LogicalHistory history;
        history.reset(2);
        LogicalLayer layer;
        layer.measured.assign(2, 0u);
        layer.bonds.push_back(LogicalBond{0, 1});
        history.layers.push_back(layer);
        history.layers.push_back(LogicalLayer{{}, {0u, 0u}});
        history.valid = true;
        for (bool precompute : {false, true})
        {
            ConnectivityIndex index;
            index.build(history, true, true, precompute);
            const PairTurns t = index.pair_turns(0, 1);
            expect(!index.has_gate_metrics() && t.evaluated && t.min_turns == 1 && t.common_source &&
                       t.doubled_network,
                   "inert: NaN gate metrics leave existence intact");
            expect(!t.shortest_maximin.available && !t.widest.available &&
                       t.shortest_maximin.cancelled_gates == -1 && t.widest.cancelled_gates == -1,
                   "inert: NaN gate metrics: no shape, no cancelled count");
        }
    }
    {
        const LogicalHistory history = make_history(4, {{{{0, 1}, {2, 3}}, {}}, {{}, {}}});
        ConnectivityIndex index;
        index.build(history, true, true, true);
        const PairTurns t = index.pair_turns(0, 2);
        expect(t.evaluated && t.min_turns == -1 && !t.common_source && !t.certified && !t.doubled_network &&
                   t.flags() == 0u && !t.shortest_maximin.available,
               "inert: separate components have no walk, no doubled network, no shape");
        expect(!index.pair_turns(1, 1).evaluated && !index.pair_turns(-1, 2).evaluated &&
                   !index.pair_turns(0, 4).evaluated,
               "inert: a self pair or an out-of-range site is not evaluated");
        expect(!index.path_shape(GraphPath{}, 0, 1).available, "inert: an empty path has no shape");
        GraphPath good = index.shortest_maximin_trace(0, 1);
        expect(index.path_shape(good, 0, 1).available, "inert: premise, the canonical 0-1 path reads");
        GraphPath short_nodes = good;
        short_nodes.nodes.pop_back();
        expect(!index.path_shape(short_nodes, 0, 1).available, "inert: nodes must be one more than edges");
        GraphPath wrong = good;
        std::swap(wrong.nodes.front(), wrong.nodes.back());
        expect(!index.path_shape(wrong, 0, 1).available || good.nodes.front() == good.nodes.back(),
               "inert: an edge that does not join its two nodes is refused");
        // Down a worldline and straight back up it: a direction change with no
        // gate between, which no path can make.
        const GraphPath bounce = path_through(index, {node_id(1, 0, 4), node_id(0, 0, 4), node_id(1, 0, 4)});
        expect(!bounce.empty() && !index.path_shape(bounce, 0, 1).available,
               "inert: a turn without a gate is refused");
        expect(!index.path_shape(good, 0, 9).available, "inert: an out-of-range endpoint is refused");
    }
}

// ---------------------------------------------------------------------------
// Part 2: random histories against the independent restatements
// ---------------------------------------------------------------------------

// A random history with at most one gate per mode per layer, on any pair of
// modes (not only neighbours), optionally with tied capacities so the
// canonical-path tie-breaking is exercised.
LogicalHistory random_matching_history(int n, int layers, double p, std::mt19937 &rng, bool ties)
{
    LogicalHistory history;
    history.reset(n);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::bernoulli_distribution measure(p);
    std::uniform_int_distribution<int> gate_count(0, n / 2);
    const double levels[] = {0.25, 0.5, 0.75, 1.0};
    for (int t = 0; t < layers; ++t)
    {
        LogicalLayer layer;
        layer.measured.assign(static_cast<std::size_t>(n), 0u);
        std::vector<int> modes(static_cast<std::size_t>(n));
        std::iota(modes.begin(), modes.end(), 0);
        std::shuffle(modes.begin(), modes.end(), rng);
        const int gates = gate_count(rng);
        for (int k = 0; k < gates; ++k)
        {
            LogicalBond bond{modes[static_cast<std::size_t>(2 * k)], modes[static_cast<std::size_t>(2 * k + 1)]};
            bond.source_gate_index = k;
            bond.entangling_power = ties ? levels[rng() % 4u] : unit(rng);
            bond.even_block_mixing = unit(rng);
            bond.odd_block_mixing = unit(rng);
            layer.bonds.push_back(bond);
        }
        for (int m = 0; m < n; ++m)
        {
            layer.measured[static_cast<std::size_t>(m)] = measure(rng) ? 1u : 0u;
        }
        history.layers.push_back(std::move(layer));
    }
    history.valid = true;
    return history;
}

bool one_gate_per_mode_per_layer(const LogicalHistory &history)
{
    for (const LogicalLayer &layer : history.layers)
    {
        std::vector<int> count(static_cast<std::size_t>(history.n), 0);
        for (const LogicalBond &bond : layer.bonds)
        {
            if (++count[static_cast<std::size_t>(bond.a)] > 1 || ++count[static_cast<std::size_t>(bond.b)] > 1)
            {
                return false;
            }
        }
    }
    return true;
}

struct PropertyTotals
{
    long histories = 0;
    long ordered_pairs = 0;
    long interior = 0;
    long connected = 0;
    long connected_common_source = 0;
    long connected_certified = 0;
    long connected_doubled = 0;
    long sm_class[4] = {0, 0, 0, 0}; // over connected pairs
    long sm_cancelled = 0;           // connected pairs whose shortest-maximin path loses a gate
    // Trace-only is a local verdict (no connector measured after the bridge);
    // the doubled network is global (a connector may still feed a later gate
    // whose partner is measured). Counted, not asserted: they can differ.
    long sm_trace_only_kept = 0;
    long certified_but_sm_trace_only = 0;
};

void check_properties(const LogicalHistory &history, const std::string &tag, PropertyTotals &totals)
{
    using mipt::dist::FLAG_CONNECTED_CERTIFIED;
    using mipt::dist::FLAG_CONNECTED_COMMON_SOURCE;
    using mipt::dist::FLAG_CONNECTED_DOUBLED;
    const int n = history.n;
    const int layers = static_cast<int>(history.layers.size());
    ConnectivityIndex plain;  // turns on demand
    ConnectivityIndex turned; // turns precomputed
    ConnectivityIndex lean;   // turns precomputed, nothing else
    plain.build(history, true, true, false);
    turned.build(history, true, true, true);
    lean.build(history, false, false, true);
    ++totals.histories;
    expect(!plain.has_turns() && turned.has_turns() && lean.has_turns(), tag + ": has_turns follows the build flag");
    const RefGraph ref = reference_graph(history);
    expect(same_edge_set(ref, turned), tag + ": the index carries exactly the history's edge set");
    for (const RefEdge &edge : ref.edges)
    {
        if (edge.gate)
        {
            expect(ref.measurement_cone[edge.a] == ref.measurement_cone[edge.b],
                   tag + ": a gate is in a cone through both of its nodes or neither");
        }
    }
    // doubled_relevant for every node of one pair, against the reference.
    if (n >= 2)
    {
        bool relevant_ok = true;
        for (std::uint32_t v = 0; v < ref.adjacency.size(); ++v)
        {
            relevant_ok = relevant_ok && turned.doubled_relevant(v, 0, n - 1) == ref.kept(v, 0, n - 1);
        }
        expect(relevant_ok, tag + ": doubled_relevant is R_M u L_i u L_j");
    }
    for (int i = 0; i < n; ++i)
    {
        const RefWalks walks = reference_walks(ref, ref.final_node(i));
        for (int j = 0; j < n; ++j)
        {
            const PairConnectivity q = turned.query(i, j);
            expect(same_query(plain.query(i, j), q), tag + ": query() is identical with and without turns");
            expect(same_gate_metrics(plain.gate_path_metrics(i, j), turned.gate_path_metrics(i, j)),
                   tag + ": gate_path_metrics is identical with and without turns");
            const PairTurns t = turned.pair_turns(i, j);
            if (i == j)
            {
                expect(!t.evaluated, tag + ": a self pair is not evaluated");
                continue;
            }
            ++totals.ordered_pairs;
            expect(t.evaluated, tag + ": every pair of a built graph is evaluated");
            expect(same_turns(t, plain.pair_turns(i, j)),
                   tag + ": precomputed pair_turns equal the on-demand ones, every field");
            expect(same_turns(t, lean.pair_turns(i, j)),
                   tag + ": and do not depend on the path or gate precomputation");

            // Existence, against the restatements.
            const std::uint32_t sink = ref.final_node(j);
            expect(t.min_turns == walks.min_turns[sink], tag + ": min_turns equals the reference walk search");
            expect(t.certified == (walks.certified[sink] != 0u), tag + ": certified equals the reference walk search");
            bool meet = false;
            for (std::size_t v = 0; v < ref.adjacency.size() && !meet; ++v)
            {
                meet = ref.endpoint_cones[static_cast<std::size_t>(i)][v] != 0u &&
                       ref.endpoint_cones[static_cast<std::size_t>(j)][v] != 0u;
            }
            expect(t.common_source == meet, tag + ": common_source iff the past light cones intersect");
            expect(t.doubled_network == reference_doubled(ref, i, j),
                   tag + ": doubled_network equals the reference reduction");
            expect(t.common_source == (t.min_turns >= 0 && t.min_turns <= 1), tag + ": common_source = min_turns <= 1");
            expect(t.min_turns < 0 || t.min_turns == 0 || t.min_turns % 2 == 1, tag + ": min_turns is 0 or odd");
            expect((t.min_turns >= 0) == q.interior_path, tag + ": a walk exists iff the endpoints share a component");

            // Nesting: C_cs <= C_cert <= C_doubled <= C, with and without survival.
            expect((!t.common_source || t.certified) && (!t.certified || t.doubled_network) &&
                       (!t.doubled_network || q.interior_path),
                   tag + ": common_source <= certified <= doubled_network <= interior_path");
            expect((!t.connected_common_source || t.connected_certified) &&
                       (!t.connected_certified || t.connected_doubled) && (!t.connected_doubled || q.connected),
                   tag + ": C_cs <= C_cert <= C_doubled <= C");
            const bool survive = q.survives_i && q.survives_j;
            expect(t.connected_common_source == (survive && t.common_source) &&
                       t.connected_certified == (survive && t.certified) &&
                       t.connected_doubled == (survive && t.doubled_network),
                   tag + ": the refined events are the existence flags gated on survival");
            const std::uint32_t mask = (t.connected_common_source ? FLAG_CONNECTED_COMMON_SOURCE : 0u) |
                                       (t.connected_certified ? FLAG_CONNECTED_CERTIFIED : 0u) |
                                       (t.connected_doubled ? FLAG_CONNECTED_DOUBLED : 0u);
            expect(t.flags() == mask && (t.flags() & q.flags()) == 0u,
                   tag + ": the turn flags are bits 7-9, disjoint from the connectivity bits");
            const PairTurns r = turned.pair_turns(j, i);
            expect(t.min_turns == r.min_turns && t.common_source == r.common_source && t.certified == r.certified &&
                       t.doubled_network == r.doubled_network && t.connected_doubled == r.connected_doubled &&
                       t.connected_certified == r.connected_certified &&
                       t.connected_common_source == r.connected_common_source,
                   tag + ": existence is symmetric in the endpoints");

            totals.interior += q.interior_path ? 1 : 0;
            totals.connected += q.connected ? 1 : 0;
            totals.connected_common_source += t.connected_common_source ? 1 : 0;
            totals.connected_certified += t.connected_certified ? 1 : 0;
            totals.connected_doubled += t.connected_doubled ? 1 : 0;

            // The canonical paths.
            if (!q.interior_path || !turned.has_gate_metrics())
            {
                expect(!t.shortest_maximin.available && !t.widest.available &&
                           t.shortest_maximin.cancelled_gates == -1 && t.widest.cancelled_gates == -1,
                       tag + ": no component or no metrics: no shape and no cancelled count");
                expect(turned.shortest_maximin_trace(i, j).empty() && turned.widest_gate_trace(i, j).empty(),
                       tag + ": and no trace");
                continue;
            }
            const GraphPath sm = turned.shortest_maximin_trace(i, j);
            const GraphPath wd = turned.widest_gate_trace(i, j);
            expect(well_formed(turned, sm, i, j) && well_formed(turned, wd, i, j),
                   tag + ": canonical traces are simple ordered paths from final_node(i) to final_node(j)");
            expect(sm.edges == turned.shortest_maximin_path(i, j) && wd.edges == turned.widest_gate_path(i, j),
                   tag + ": the traces' edges are the edge-list accessors'");
            expect(same_shape(turned.path_shape(sm, i, j), t.shortest_maximin, false) &&
                       same_shape(turned.path_shape(wd, i, j), t.widest, false),
                   tag + ": path_shape(trace) equals the stored shape field by field");
            expect(t.shortest_maximin.cancelled_gates == reference_cancelled(ref, turned, sm, i, j) &&
                       t.widest.cancelled_gates == reference_cancelled(ref, turned, wd, i, j),
                   tag + ": cancelled_gates counts the path's gates outside R_M u L_i u L_j");
            const mipt::dist::GatePathMetrics g = turned.gate_path_metrics(i, j);
            expect(t.shortest_maximin.length == g.shortest_path_length && t.shortest_maximin.length == q.shortest_path &&
                       t.shortest_maximin.unitary == g.shortest_path_unitary_count,
                   tag + ": the shortest-maximin shape realizes its gate-path metrics");
            double weakest = std::numeric_limits<double>::infinity();
            for (std::uint32_t e : wd.edges)
            {
                weakest = std::min(weakest, turned.graph_edges()[e].entangling_power);
            }
            expect(weakest == g.widest_entangling_power, tag + ": the widest trace's weakest edge is the widest value");
            for (const PathShape *shape : {&t.shortest_maximin, &t.widest})
            {
                expect(shape_consistent(*shape, layers), tag + ": a canonical shape is internally consistent");
                expect(shape->turns >= t.min_turns, tag + ": no canonical path beats min_turns");
                // A shortest path (in any subgraph that keeps every temporal
                // edge) cannot carry an endpoint-anchored bridge: the
                // endpoint's intact worldline is strictly shorter.
                expect(shape->bridges_endpoint == 0, tag + ": a canonical path has no endpoint-anchored bridge");
                if (shape->path_class != PathClass::ZigzagTraceOnly)
                {
                    // Every gate of a walk whose peaks are all measured lies in
                    // R_M u L_i u L_j, so a non-trace-only canonical path keeps
                    // all of its gates.
                    expect(shape->cancelled_gates == 0, tag + ": a path with no trace-only bridge loses no gate");
                }
            }
            if (q.connected)
            {
                ++totals.sm_class[static_cast<int>(t.shortest_maximin.path_class)];
                totals.sm_cancelled += t.shortest_maximin.cancelled_gates > 0 ? 1 : 0;
                totals.sm_trace_only_kept +=
                    (t.shortest_maximin.trace_only_bridge() && t.shortest_maximin.cancelled_gates == 0) ? 1 : 0;
                totals.certified_but_sm_trace_only +=
                    (t.connected_certified && t.shortest_maximin.trace_only_bridge()) ? 1 : 0;
            }
        }
    }
}

void test_random_properties()
{
    PropertyTotals production;
    std::mt19937 rng(20261110);
    for (int family = 0; family < 3; ++family)
    {
        for (int n : {4, 6, 8, 10, 12, 16})
        {
            for (double p : {0.0, 0.1, 0.2, 0.335, 0.5, 0.9})
            {
                for (int seed = 0; seed < 2; ++seed)
                {
                    std::vector<mipt::RppuLayer> layers;
                    if (family == 2)
                    {
                        mipt::build_qrppu_layers(layers, n, n / 2, p, true, rng);
                    }
                    else
                    {
                        mipt::build_rppu_layers(layers, n, n / 2, p, true, rng, family == 1);
                    }
                    const LogicalHistory history = mipt::logical_history_from_rppu(layers, n);
                    const std::string tag = std::string(family == 0 ? "rppu-fswap" : family == 1 ? "rppu-jw" : "qrppu") +
                                            " n=" + std::to_string(n);
                    expect(one_gate_per_mode_per_layer(history),
                           tag + ": the production projection gives each mode at most one gate per layer");
                    check_properties(history, tag, production);
                }
            }
        }
    }
    PropertyTotals matchings;
    for (int trial = 0; trial < 96; ++trial)
    {
        const int sizes[] = {5, 6, 8, 9};
        const int depths[] = {1, 2, 4, 7};
        const LogicalHistory history =
            random_matching_history(sizes[trial % 4], depths[(trial / 4) % 4], 0.05 + 0.15 * ((trial / 16) % 3), rng,
                                    trial % 2 == 0);
        check_properties(history, "matching", matchings);
    }
    for (const auto &[label, totals] : {std::pair<const char *, PropertyTotals *>{"RPPU/qRPPU", &production},
                                        std::pair<const char *, PropertyTotals *>{"random matchings", &matchings}})
    {
        std::printf("  random %s: %ld histories, %ld ordered pairs; interior %ld, C %ld, C_cs %ld, C_cert %ld, "
                    "C_doubled %ld\n",
                    label, totals->histories, totals->ordered_pairs, totals->interior, totals->connected,
                    totals->connected_common_source, totals->connected_certified, totals->connected_doubled);
        std::printf("    connected pairs by shortest-maximin class tm/cs/zc/zto = %ld/%ld/%ld/%ld; path loses a gate %ld; "
                    "trace-only yet no gate cancelled %ld; certified but path trace-only %ld\n",
                    totals->sm_class[0], totals->sm_class[1], totals->sm_class[2], totals->sm_class[3],
                    totals->sm_cancelled, totals->sm_trace_only_kept, totals->certified_but_sm_trace_only);
    }
}

// ---------------------------------------------------------------------------
// Brute force: every simple path, on small graphs
// ---------------------------------------------------------------------------

struct BruteTotals
{
    long histories = 0;
    long pairs = 0;          // unordered
    long joined = 0;         // with at least one simple path
    long paths = 0;
    long truncated = 0;
    long walk_fewer_turns = 0;          // min over walks < min over simple paths
    long walk_only_certified = 0;       // certified walk, no simple path with measured peaks only
    long walk_only_certified_anchor = 0; // ... nor with measured-or-anchored peaks only
    long path_certified = 0;
    long path_certified_anchor = 0;
    long walk_certified = 0;
    long anchored_paths = 0;
    long doubled = 0;
    std::map<int, long> min_turns;
    std::string first_disagreement;
};

// Read a simple path as a shape from first principles and compare it, field by
// field, with path_shape(); returns whether every peak is measured (and
// measured-or-anchored), and the number of cancelled gates.
struct RefPathFacts
{
    int turns = 0;
    bool all_measured = true;
    bool all_measured_or_anchored = true;
    bool anchored = false;
    int cancelled = 0;
};

RefPathFacts check_simple_path(const RefGraph &g, const ConnectivityIndex &index,
                               const std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> &edge_ids,
                               const std::vector<std::uint32_t> &nodes, const std::vector<int> &edges, int i, int j)
{
    RefPathFacts facts;
    GraphPath path;
    path.nodes = nodes;
    PathShape expected;
    expected.available = true;
    expected.length = static_cast<int>(edges.size());
    expected.min_layer = expected.max_layer = g.layer_of(nodes.front());
    for (std::uint32_t v : nodes)
    {
        expected.min_layer = std::min(expected.min_layer, g.layer_of(v));
        expected.max_layer = std::max(expected.max_layer, g.layer_of(v));
    }
    const int mode_i = g.mode_of_site[static_cast<std::size_t>(i)];
    const int mode_j = g.mode_of_site[static_cast<std::size_t>(j)];
    int last = 0;
    for (std::size_t k = 0; k < edges.size(); ++k)
    {
        const RefEdge &edge = g.edges[static_cast<std::size_t>(edges[k])];
        path.edges.push_back(edge_ids.at({std::min(edge.a, edge.b), std::max(edge.a, edge.b)}));
        if (edge.gate)
        {
            ++expected.unitary;
            expected.gate_layers.push_back(edge.layer);
            facts.cancelled += (!g.kept(edge.a, i, j) && !g.kept(edge.b, i, j)) ? 1 : 0;
            continue;
        }
        const int direction = g.layer_of(nodes[k + 1]) > g.layer_of(nodes[k]) ? 1 : -1;
        (direction > 0 ? expected.temporal_forward : expected.temporal_backward) += 1;
        if (last != 0 && direction != last)
        {
            ++expected.turns;
            if (last > 0)
            {
                ++expected.peaks;
                PathBridge bridge;
                const RefEdge &gate = g.edges[static_cast<std::size_t>(edges[k - 1])];
                bridge.edge = path.edges[k - 1];
                bridge.layer = gate.layer;
                bridge.mode_in = g.mode_of(nodes[k - 1]);
                bridge.mode_out = g.mode_of(nodes[k]);
                bridge.measured_layer_in = g.next_measured[node_id(bridge.layer, bridge.mode_in, g.n)];
                bridge.measured_layer_out = g.next_measured[node_id(bridge.layer, bridge.mode_out, g.n)];
                if (bridge.measured_layer_in >= 0 || bridge.measured_layer_out >= 0)
                {
                    bridge.status = BridgeStatus::Measured;
                    ++expected.bridges_measured;
                }
                else if (bridge.mode_in == mode_i || bridge.mode_in == mode_j || bridge.mode_out == mode_i ||
                         bridge.mode_out == mode_j)
                {
                    bridge.status = BridgeStatus::EndpointAnchored;
                    ++expected.bridges_endpoint;
                    facts.all_measured = false;
                    facts.anchored = true;
                }
                else
                {
                    bridge.status = BridgeStatus::TraceOnly;
                    ++expected.bridges_trace_only;
                    facts.all_measured = false;
                    facts.all_measured_or_anchored = false;
                }
                expected.bridges.push_back(bridge);
            }
            else
            {
                ++expected.valleys;
            }
        }
        last = direction;
    }
    const int bridges = static_cast<int>(expected.bridges.size());
    expected.path_class = expected.turns == 0   ? PathClass::TimeMonotone
                          : expected.turns == 1 ? PathClass::CommonSource
                          : expected.bridges_trace_only > 0 ? PathClass::ZigzagTraceOnly
                                                            : PathClass::ZigzagCertified;
    expected.connector_status = bridges == 0                            ? ConnectorStatus::NoBridge
                                : expected.bridges_measured == bridges ? ConnectorStatus::AllMeasured
                                : expected.bridges_measured == 0       ? ConnectorStatus::Unmeasured
                                                                       : ConnectorStatus::Mixed;
    expect(same_shape(index.path_shape(path, i, j), expected, false),
           "brute force: path_shape matches a first-principles reading of every simple path");
    // A path with no trace-only bridge keeps every gate: its measured bridges
    // are in R_M, its anchored ones on an intact endpoint worldline (L_i or
    // L_j), and every segment between lies in the cone of the turn above it.
    expect(!facts.all_measured_or_anchored || facts.cancelled == 0,
           "brute force: a simple path without a trace-only bridge loses no gate to the doubling");
    facts.turns = expected.turns;
    return facts;
}

std::string describe(const LogicalHistory &history)
{
    std::string out;
    for (std::size_t t = 0; t < history.layers.size(); ++t)
    {
        out += "    layer " + std::to_string(t) + ": bonds";
        for (const LogicalBond &bond : history.layers[t].bonds)
        {
            out += " (" + std::to_string(bond.a) + "," + std::to_string(bond.b) + ")";
        }
        out += " measured";
        for (int m = 0; m < history.n; ++m)
        {
            if (history.layers[t].measured[static_cast<std::size_t>(m)] != 0u)
            {
                out += " " + std::to_string(m);
            }
        }
        out += "\n";
    }
    return out;
}

void brute_force_history(const LogicalHistory &history, BruteTotals &totals)
{
    constexpr long budget = 400000;
    ConnectivityIndex index;
    index.build(history, true, true, true);
    const RefGraph g = reference_graph(history);
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> edge_ids;
    for (std::uint32_t e = 0; e < index.graph_edges().size(); ++e)
    {
        const GraphEdge &edge = index.graph_edges()[e];
        edge_ids[{std::min(edge.a, edge.b), std::max(edge.a, edge.b)}] = e;
    }
    ++totals.histories;
    for (int i = 0; i < history.n; ++i)
    {
        for (int j = i + 1; j < history.n; ++j)
        {
            ++totals.pairs;
            const PairTurns t = index.pair_turns(i, j);
            long paths = 0;
            bool truncated = false;
            int min_turns = -1;
            int min_length = -1;
            int min_cancelled = -1;
            bool certified = false;
            bool certified_anchor = false;
            std::vector<std::uint8_t> on(g.adjacency.size(), 0u);
            std::vector<std::uint32_t> nodes{g.final_node(i)};
            std::vector<int> edges;
            on[nodes[0]] = 1u;
            const std::uint32_t sink = g.final_node(j);
            std::function<void(std::uint32_t)> dfs = [&](std::uint32_t v) {
                if (truncated)
                {
                    return;
                }
                if (v == sink)
                {
                    if (++paths > budget)
                    {
                        truncated = true;
                        return;
                    }
                    const RefPathFacts facts = check_simple_path(g, index, edge_ids, nodes, edges, i, j);
                    min_turns = min_turns < 0 ? facts.turns : std::min(min_turns, facts.turns);
                    const int length = static_cast<int>(edges.size());
                    min_length = min_length < 0 ? length : std::min(min_length, length);
                    min_cancelled = min_cancelled < 0 ? facts.cancelled : std::min(min_cancelled, facts.cancelled);
                    certified = certified || facts.all_measured;
                    certified_anchor = certified_anchor || facts.all_measured_or_anchored;
                    totals.anchored_paths += facts.anchored ? 1 : 0;
                    return;
                }
                for (const auto &[w, e] : g.adjacency[v])
                {
                    if (on[w] != 0u)
                    {
                        continue;
                    }
                    on[w] = 1u;
                    nodes.push_back(w);
                    edges.push_back(e);
                    dfs(w);
                    nodes.pop_back();
                    edges.pop_back();
                    on[w] = 0u;
                }
            };
            dfs(nodes[0]);
            if (truncated)
            {
                ++totals.truncated;
                continue;
            }
            totals.paths += paths;
            expect((paths == 0) == (t.min_turns < 0), "brute force: a walk exists iff a simple path does");
            if (paths == 0)
            {
                expect(!t.certified && !t.doubled_network, "brute force: no path, nothing certified or doubled");
                continue;
            }
            ++totals.joined;
            ++totals.min_turns[t.min_turns];
            expect(t.min_turns <= min_turns, "brute force: min_turns over walks <= min over simple paths");
            expect(t.common_source == (min_turns <= 1), "brute force: common_source iff a simple path has <= 1 turn");
            expect(!certified || t.certified, "brute force: a simple path with measured peaks only => certified");
            expect(!certified_anchor || t.certified,
                   "brute force: a simple path with measured or endpoint-anchored peaks only => certified");
            expect(t.doubled_network == (min_cancelled == 0),
                   "brute force: doubled_network iff some simple path keeps every gate");
            if (index.has_gate_metrics())
            {
                expect(t.shortest_maximin.length == min_length,
                       "brute force: the shortest-maximin path is a shortest simple path");
                expect(t.shortest_maximin.turns >= min_turns && t.widest.turns >= min_turns,
                       "brute force: no canonical path has fewer turns than every simple path");
            }
            totals.walk_certified += t.certified ? 1 : 0;
            totals.path_certified += certified ? 1 : 0;
            totals.path_certified_anchor += certified_anchor ? 1 : 0;
            totals.doubled += t.doubled_network ? 1 : 0;
            const bool fewer = t.min_turns < min_turns;
            const bool only = t.certified && !certified;
            const bool only_anchor = t.certified && !certified_anchor;
            totals.walk_fewer_turns += fewer ? 1 : 0;
            totals.walk_only_certified += only ? 1 : 0;
            totals.walk_only_certified_anchor += only_anchor ? 1 : 0;
            if ((fewer || only_anchor) && totals.first_disagreement.empty())
            {
                totals.first_disagreement = "    pair (" + std::to_string(i) + "," + std::to_string(j) +
                                            "): walk min_turns " + std::to_string(t.min_turns) +
                                            " vs simple-path " + std::to_string(min_turns) + ", walk certified " +
                                            std::to_string(t.certified) + " vs simple-path " +
                                            std::to_string(certified_anchor) + "\n" + describe(history);
            }
        }
    }
}

void test_brute_force_simple_paths()
{
    BruteTotals totals;
    std::mt19937 rng(77001);
    for (int trial = 0; trial < 160; ++trial)
    {
        const LogicalHistory history =
            random_matching_history(6, 2 + trial % 4, 0.1 + 0.1 * ((trial / 4) % 3), rng, trial % 2 == 0);
        brute_force_history(history, totals);
    }
    for (int trial = 0; trial < 60; ++trial)
    {
        std::vector<mipt::RppuLayer> layers;
        mipt::build_rppu_layers(layers, 6, 1 + trial % 2, 0.1 + 0.1 * ((trial / 2) % 3), true, rng, trial % 4 < 2);
        brute_force_history(mipt::logical_history_from_rppu(layers, 6), totals);
    }
    expect(totals.truncated == 0, "brute force: premise, every pair's path set was enumerated in full");
    std::printf("  brute force (N=6, 2-5 layers; 160 random matchings + 60 RPPU): %ld histories, %ld pairs, %ld joined, "
                "%ld simple paths (%ld pairs truncated)\n",
                totals.histories, totals.pairs, totals.joined, totals.paths, totals.truncated);
    std::string histogram;
    for (const auto &[turns, count] : totals.min_turns)
    {
        histogram += " " + std::to_string(turns) + ":" + std::to_string(count);
    }
    std::printf("    min_turns histogram (joined pairs):%s\n", histogram.c_str());
    std::printf("    certified: walk %ld, simple path with measured peaks %ld, with measured-or-anchored peaks %ld; "
                "doubled %ld; paths carrying an anchored bridge %ld\n",
                totals.walk_certified, totals.path_certified, totals.path_certified_anchor, totals.doubled,
                totals.anchored_paths);
    std::printf("    walk vs simple path: fewer turns %ld, certified only as a walk %ld (measured peaks) / %ld "
                "(measured-or-anchored)\n",
                totals.walk_fewer_turns, totals.walk_only_certified, totals.walk_only_certified_anchor);
    if (!totals.first_disagreement.empty())
    {
        std::printf("    first disagreement:\n%s", totals.first_disagreement.c_str());
    }
}

// ---------------------------------------------------------------------------
// The doubled-network theorem, on Born-sampled random circuits
// ---------------------------------------------------------------------------

// !doubled_network(i, j) => rho_ij = rho_i (x) rho_j exactly, whatever the
// gates. Open chains, so every bond is adjacent and the qubit simulation is
// the fermionic one; the RDM is the production fermionic reduction. The
// converse is not a theorem, so the correlated fraction of the surviving
// pairs is reported rather than asserted.
void test_doubled_network_theorem()
{
    long circuits = 0;
    long pairs = 0;
    long cut = 0;
    long violations = 0;
    long cut_but_connected = 0;
    long kept = 0;
    long kept_correlated = 0;
    long connected_doubled = 0;
    long connected_doubled_correlated = 0;
    long doubled_not_certified = 0;
    long doubled_not_certified_correlated = 0;
    double worst = 0.0;
    double worst_product = 0.0;
    for (int n : {6, 8, 10})
    {
        for (double p : {0.15, 0.3, 0.45})
        {
            for (int seed = 0; seed < (n == 10 ? 12 : 40); ++seed)
            {
                std::mt19937 rng(static_cast<std::uint32_t>(1000 * seed + static_cast<int>(p * 100) + n));
                std::mt19937 born(static_cast<std::uint32_t>(seed + 7 + 100 * n));
                const int depth = n + seed % 3;
                std::vector<mipt::RppuLayer> layers(static_cast<std::size_t>(depth));
                for (int t = 0; t < depth; ++t)
                {
                    mipt::fill_rppu_layer(layers[static_cast<std::size_t>(t)], n, t % 2 == 1, p, false, rng,
                                          mipt::RppuBoundaryMode::FermionicJWString);
                }
                const LogicalHistory history = mipt::logical_history_from_rppu(layers, n);
                bool adjacent = true;
                for (const LogicalLayer &layer : history.layers)
                {
                    for (const LogicalBond &bond : layer.bonds)
                    {
                        adjacent = adjacent && std::abs(bond.a - bond.b) == 1;
                    }
                }
                expect(adjacent, "theorem: premise, an open chain has adjacent bonds only");
                State psi(std::size_t{1} << n, Complex(0.0, 0.0));
                psi[0] = 1.0;
                for (const mipt::RppuLayer &layer : layers)
                {
                    for (const mipt::RppuBondGate &gate : layer.gates)
                    {
                        apply_two(psi, gate.q0, gate.q1, mipt::rppu_gate_matrix(gate));
                    }
                    for (int q = 0; q < n; ++q)
                    {
                        if (layer.measure_flags[static_cast<std::size_t>(q)] != 0)
                        {
                            born_measure(psi, q, born);
                        }
                    }
                }
                ++circuits;
                ConnectivityIndex index;
                index.build(history, true, true, true);
                for (int i = 0; i < n; ++i)
                {
                    for (int j = i + 1; j < n; ++j)
                    {
                        ++pairs;
                        const PairTurns t = index.pair_turns(i, j);
                        const PairReport r = pair_report(psi, n, i, j);
                        const double deviation = std::max({r.g, r.f, r.rho_n});
                        if (!t.doubled_network)
                        {
                            ++cut;
                            cut_but_connected += index.query(i, j).connected ? 1 : 0;
                            worst = std::max(worst, deviation);
                            worst_product = std::max(worst_product, r.product);
                            violations += deviation > 1.0e-13 ? 1 : 0;
                            continue;
                        }
                        const bool correlated = deviation > 1.0e-10;
                        ++kept;
                        kept_correlated += correlated ? 1 : 0;
                        if (t.connected_doubled)
                        {
                            ++connected_doubled;
                            connected_doubled_correlated += correlated ? 1 : 0;
                        }
                        if (!t.certified)
                        {
                            ++doubled_not_certified;
                            doubled_not_certified_correlated += correlated ? 1 : 0;
                        }
                    }
                }
            }
        }
    }
    expect(violations == 0,
           "theorem: every pair the doubled network disconnects is an exact product (|G|, |F|, |D - n_i n_j| <= 1e-13)");
    expect(cut > 0 && cut_but_connected > 0, "theorem: premise, the test sees cut pairs that C calls connected");
    std::printf("  doubled-network theorem: %ld open-chain circuits (N=6,8,10), %ld pairs; cut %ld (of them C=1: %ld), "
                "max |G|,|F|,|D-n_i n_j| = %.2e, max |rho-rho_i(x)rho_j| = %.2e, violations %ld\n",
                circuits, pairs, cut, cut_but_connected, worst, worst_product, violations);
    std::printf("    kept %ld (correlated > 1e-10: %ld); connected_doubled %ld (correlated %ld); doubled but not "
                "certified %ld (correlated %ld)\n",
                kept, kept_correlated, connected_doubled, connected_doubled_correlated, doubled_not_certified,
                doubled_not_certified_correlated);
}

} // namespace

int main()
{
    test_contract_constants();
    test_four_mode_swap("designed", {even_mixer(), even_mixer(), even_mixer()});
    test_four_mode_swap("haar seed 20261110", haar_gates(20261110u));
    test_four_mode_swap("haar seed 4711", haar_gates(4711u));
    test_v_shapes();
    test_final_layer_gate();
    test_double_zigzag();
    test_certified_but_canonical_trace_only();
    test_endpoint_anchored();
    test_unavailable_and_inert();
    test_random_properties();
    test_brute_force_simple_paths();
    test_doubled_network_theorem();

    if (failures != 0)
    {
        std::cerr << "temporal_turn_tests: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "temporal_turn_tests: PASS\n";
    return 0;
}
