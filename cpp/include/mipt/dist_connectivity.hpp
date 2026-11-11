#pragma once

// The spacetime connectivity graph of one trajectory, and the pair queries
// dist_scaling.exe asks of it.
//
// What this is for
// ----------------
// A necessary condition for two sites to be entangled at the end of a
// monitored circuit is that a path exists through the circuit's spacetime that
// connects them without crossing a measurement. Avakian, Pereg-Barnea and
// Witczak-Krempa (arXiv:2404.16095 Sec. VI) build exactly this graph and are
// explicit that it is *necessary but not sufficient*: a spanning path can
// carry no usable correlation, and they discuss the "parasitic" graph
// connections that make the sufficiency fail.
//
// So the point of this header is not to predict entanglement from geometry. It
// is to let the two be measured *jointly*, on the same trajectory, so that the
// two conditional probabilities
//
//     kappa = P(fN > eps |  connected),      eta = P(fN > eps | not connected)
//
// can be estimated rather than assumed. A percolation picture that is
// necessary-but-incomplete predicts eta ~ 0 and kappa < 1; comparing marginals
// P(C) against P(E) cannot distinguish that from any number of alternatives,
// which is why the aggregation bins carry the full 2x2 contingency and not two
// separate counts.
//
// The graph
// ---------
// Nodes are (layer, mode) over the T layers of the trajectory. Edges are:
//
//   * a *unitary bond* joining the two modes an entangling gate acts on,
//     within the layer that applies it;
//   * a *temporal bond* joining (t, m) to (t+1, m) whenever mode m is NOT
//     measured in layer t. A measurement is the absence of an edge, not the
//     presence of one.
//
// Ordering inside a layer is gates-then-measurements, matching the circuit
// kernels: a gate at layer t is visible to both its endpoints at slice t even
// if one of them is measured immediately afterwards, and it is the measurement
// that severs the link into slice t+1.
//
// The final slice is implicit. Node (T, m) could only ever have one edge --
// down to (T-1, m), present iff m survives the last layer -- so
//
//     C  ==  A_i  &&  A_j  &&  (T-1, i) ~ (T-1, j)
//
// exactly, and the union-find only has to cover slices 0..T-1. That
// factorization is the reason the flags are reported separately: it is what
// lets P(A_i & A_j) be measured directly instead of approximated as q^2.
//
// What is deliberately not here
// -----------------------------
// The second-stage explanatory diagnostics -- edge-disjoint path counts,
// parasitic environment attachments, channel-resolved operator propagation --
// are not implemented. They explain *why* kappa < 1; they are not needed to
// establish that it is. Component size, shortest spanning path and time since
// last measurement are here because they cost one extra BFS sweep over a graph
// with a couple of thousand edges, which is nothing beside a single RDM.
//
// Nothing here depends on CUDA-Q, MOSEK, or CUDA, so `make test-dist` pins it
// against hand-constructed graphs on the host.

#include "mipt/types.hpp"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mipt::dist
{

// Bump this whenever the edge rules above change. It is written into the CSV
// and the record header so a stored contingency table can never be silently
// compared against one built under a different convention.
inline constexpr int CONNECTIVITY_GRAPH_VERSION = 1;
inline constexpr const char *CONNECTIVITY_GRAPH_DEFINITION =
    "unitary-bonds-within-layer;temporal-edge-where-unmeasured;"
    "fswap-is-transport;jw-boundary-is-one-bond";

// Compact per-record flag bits. Shared verbatim by the CSV documentation, the
// record binary, and data_analysis.records, so there is one definition of what
// bit 3 means.
enum ConnectivityFlag : std::uint32_t
{
    FLAG_EVALUATED = 1u << 0,     // connectivity was computed for this record
    FLAG_SURVIVES_I = 1u << 1,    // endpoint i is unmeasured in the final layer
    FLAG_SURVIVES_J = 1u << 2,    // endpoint j is unmeasured in the final layer
    FLAG_INTERIOR_PATH = 1u << 3, // the two endpoints share a component
    FLAG_CONNECTED = 1u << 4,     // the complete percolation event C
    FLAG_FN_POSITIVE = 1u << 5,   // fN > the configured zero tolerance
    FLAG_MN_POSITIVE = 1u << 6,   // the ordinary-trace negativity is positive
    // Temporal-turn diagnostics (MIPT_DIST_TURN_DIAGNOSTICS); written only when
    // they ran. See PairTurns below.
    FLAG_CONNECTED_COMMON_SOURCE = 1u << 7, // C, and a walk with <= 1 temporal turn exists
    FLAG_CONNECTED_CERTIFIED = 1u << 8,     // C, and a measurement-certified walk exists
    FLAG_CONNECTED_DOUBLED = 1u << 9,       // C survives the doubled-network cancellation
};

struct PairConnectivity
{
    bool evaluated = false;
    bool survives_i = false;
    bool survives_j = false;
    bool interior_path = false;
    bool connected = false;
    // Spacetime nodes in the component holding endpoint i at the final slice.
    std::uint32_t component_size = 0;
    // Hop count of the shortest spanning path, or -1 when the endpoints are in
    // different components or path lengths were not requested.
    std::int32_t shortest_path = -1;
    // Layers since each endpoint was last measured, counted back from the last
    // layer; equal to the layer count when a mode was never measured.
    std::int32_t idle_i = 0;
    std::int32_t idle_j = 0;

    std::uint32_t flags() const
    {
        std::uint32_t mask = 0;
        if (!evaluated)
        {
            return mask;
        }
        mask |= FLAG_EVALUATED;
        mask |= survives_i ? FLAG_SURVIVES_I : 0u;
        mask |= survives_j ? FLAG_SURVIVES_J : 0u;
        mask |= interior_path ? FLAG_INTERIOR_PATH : 0u;
        mask |= connected ? FLAG_CONNECTED : 0u;
        return mask;
    }
};

// ---------------------------------------------------------------------------
// Edge identities and gate-path diagnostics (MIPT_DIST_GATE_DIAGNOSTICS)
//
// The binary graph only needs to know *that* an edge exists. The gate
// diagnostics need to know *which* edge a path uses -- its kind, its layer, the
// simulated gate behind it, and how entangling that gate is -- so every
// adjacency slot also names the undirected edge it belongs to.
//
// None of this touches the union-find, endpoint survival, or `connected`:
// those are built from exactly the same edge set as before, and the tests
// assert the flags are identical with the diagnostics on and off.
// ---------------------------------------------------------------------------

enum class GraphEdgeKind
{
    Temporal,
    Unitary
};

struct GraphEdge
{
    std::uint32_t a = 0; // node ids, layer * n + mode
    std::uint32_t b = 0;
    GraphEdgeKind kind = GraphEdgeKind::Temporal;
    int layer = 0;
    // Index into the simulated layer's gate list (LogicalBond::source_gate_index);
    // -1 for a temporal edge.
    int source_gate_index = -1;
    // Capacities. A temporal edge transmits perfectly and carries 1.0 in all
    // three, so it can never lower a bottleneck; a unitary edge carries its
    // gate's normalized entangling power and parity-block mixing strengths.
    double entangling_power = 1.0;
    double even_block_mixing = 1.0;
    double odd_block_mixing = 1.0;
};

// One adjacency slot: the neighbour it reaches and the edge it travels.
struct GraphArc
{
    std::uint32_t node = 0;
    std::uint32_t edge = 0;
};

enum class GateCapacity
{
    EntanglingPower,
    EvenBlockMixing,
    OddBlockMixing
};

inline double edge_capacity(const GraphEdge &edge, GateCapacity which)
{
    switch (which)
    {
    case GateCapacity::EntanglingPower: return edge.entangling_power;
    case GateCapacity::EvenBlockMixing: return edge.even_block_mixing;
    case GateCapacity::OddBlockMixing: return edge.odd_block_mixing;
    }
    return edge.entangling_power;
}

// Gate quality along the paths joining two final-slice endpoints.
//
//   * shortest_maximin_entangling_power: over the *shortest* paths only, the
//     largest achievable weakest gate, max_{gamma shortest} min_{g in gamma}
//     e_norm(g).
//   * widest_*: the same max-min over *every* path. This is the decisive one:
//     it is near zero only when every connection between the endpoints passes
//     through a nearly non-entangling gate.
//   * shortest_path_unitary_count: the number of gates on the path the
//     shortest-maximin value is read from. Shortest paths of different gate
//     composition can coexist, so this is defined exactly as the fewest gates
//     over the shortest paths whose weakest gate attains that value -- a
//     property of the graph, not of a traversal order.
//
// `available` is false when the endpoints share no component (no path, so no
// value -- never a zero) or the graph carries no gate metrics. Everything is
// measured between the final-slice nodes, like `shortest_path`; the record
// writer additionally blanks a pair that is not `connected`, because the
// implicit last temporal step to a measured endpoint does not exist.
struct GatePathMetrics
{
    bool available = false;
    int shortest_path_length = -1;
    int shortest_path_unitary_count = 0;

    double shortest_maximin_entangling_power = std::numeric_limits<double>::quiet_NaN();
    double widest_entangling_power = std::numeric_limits<double>::quiet_NaN();
    double widest_even_block_mixing = std::numeric_limits<double>::quiet_NaN();
    double widest_odd_block_mixing = std::numeric_limits<double>::quiet_NaN();
};

// ---------------------------------------------------------------------------
// Temporal turns (MIPT_DIST_TURN_DIAGNOSTICS)
//
// The question
// ------------
// The graph is undirected, so a path between two final-slice endpoints may run
// *backward* in time, cross a gate, and run forward again. Two such shapes are
// physically different objects:
//
//   * a **common source** -- one backward-to-forward turn, a V: from i down to
//     a gate, across it, up to j. The gate at the bottom lies in the past light
//     cone of both endpoints, so it can correlate them;
//   * a **zig-zag** -- the path also turns forward-to-backward at a *peak*: it
//     climbs a connector worldline to a late gate (the **bridge**), crosses it
//     and descends another connector. A bridge acts only on connector modes. If
//     those are never measured afterwards and are traced out at the end, the
//     bridge cancels against its adjoint in the density matrix (U^dag U = I)
//     and the path carries nothing at all: a **trace-only bridge**. If a
//     connector *is* measured after the bridge, the record conditions on it and
//     the bridge can swap entanglement onto the endpoints: a
//     **measurement-certified bridge**.
//
// Conventions
// -----------
//   * A **turn** is a change of direction between two consecutive temporal
//     edges of a path (gate edges between them do not count). A path between
//     two final-slice nodes starts backward and ends forward, so its turn count
//     is odd, or zero when it is a single final-layer gate.
//   * A turn backward-to-forward is a **valley**; forward-to-backward is a
//     **peak**. In a simple path exactly one gate edge sits between the two
//     temporal edges of a turn (each node has one gate edge), and at a peak it
//     is the bridge. `mode_in` is the connector the path climbs, `mode_out` the
//     one it descends.
//   * A bridge at layer t is **measured** if either connector is measured at
//     any layer >= t (gates precede measurements inside a layer, so layer t
//     itself counts). Otherwise it is **endpoint-anchored** if a connector is
//     one of the two endpoints' modes -- an unmeasured connector then *is* that
//     endpoint's worldline, so the bridge is in its past light cone and cannot
//     cancel -- and **trace-only** if not.
//   * Path classes: 0 turns = time_monotone; 1 turn = common_source; >= 2 turns
//     = zigzag_trace_only if any bridge is trace-only (one cancelling bridge
//     cuts the path), else zigzag_certified.
//
// Pair-level existence (path-independent)
// ---------------------------------------
// The two canonical paths are one witness each; whether a *better* path exists
// is a separate question, answered over all non-backtracking walks between the
// two final-slice nodes:
//
//   * min_turns: the fewest turns any such walk needs (0-1 BFS over
//     (node, last temporal direction, arrived-by-gate) states);
//   * common_source: min_turns <= 1. Exactly "the endpoints' past light cones
//     intersect", which the tests check against an independent cone BFS;
//   * certified: a walk exists whose every peak crosses a *measured* bridge.
//     Endpoint-anchored bridges need not be admitted: a walk through one can be
//     rerouted up that endpoint's own worldline with no peak there at all.
//
// The doubled network (the definitive test)
// -----------------------------------------
// The two walk criteria above are proxies. The exact statement is about the
// density matrix rho_ij = Tr_env[K rho_0 K^dag], drawn as a doubled tensor
// network (every gate appears as U on the ket layer and U^dag on the bra layer;
// tracing a mode joins its two layers locally). Call a *sink* every measurement
// projector and the two endpoints' final nodes. A gate whose future light cone
// contains no sink acts only on modes that are traced out unconditioned, so it
// can be moved to the end and cancels against its adjoint (U^dag U = I) -- and
// so does everything in its future. What survives is exactly the gates in the
// union of the sinks' past light cones,
//
//     R_M  (past cone of every measurement event, pair-independent)
//     L_i, L_j  (past cones of the two endpoints' final nodes),
//
// and the network that remains connects modes only through those gates. So
//
//     doubled_network(i, j):  the two final nodes are joined by temporal edges
//                             and gates in R_M u L_i u L_j,
//
// and when it fails the remaining network *factorizes* between i and j:
// rho_ij = rho_i (x) rho_j exactly, whatever the gates are. That is a theorem,
// not a heuristic, and `make test-dist` checks it against simulated states.
// The converse does not hold -- a surviving connection can still carry an
// accidentally small correlation -- so this is the sharpest *necessary*
// condition the graph can state.
//
// A past cone is closed backward in time along temporal edges (where unmeasured)
// and across in-layer gate edges; a gate is in a cone exactly when either of its
// nodes is. Every gate a certified walk uses is in R_M u L_i u L_j, so
//
//     C_common_source <= C_certified <= C_doubled <= C,
//
// and none of this alters C.
// ---------------------------------------------------------------------------

inline constexpr int TURN_DIAGNOSTICS_VERSION = 1;
inline constexpr const char *TURN_DIAGNOSTICS_DEFINITION =
    "turn=direction change between consecutive temporal edges;peak=F-gate-B,bridge=its gate;"
    "bridge measured if a connector is measured at layer>=bridge layer,else endpoint-anchored "
    "if a connector is an endpoint mode,else trace-only;class:0 turns=time_monotone,"
    "1=common_source,>=2=zigzag_trace_only if any trace-only bridge else zigzag_certified;"
    "min_turns over non-backtracking walks;common_source=min_turns<=1;"
    "certified=walk whose every peak is a measured bridge;"
    "doubled_network=connected through temporal edges and gates in the past light cone of a "
    "measurement or of either endpoint (every other gate cancels as U^dag U);"
    "cancelled_gates=gates of a path outside that set";

// An ordered path: `nodes[k]` and `nodes[k+1]` are the two ends of
// `edges[k]` (graph_edges() indices). Starts at endpoint i's final-slice node.
struct GraphPath
{
    std::vector<std::uint32_t> nodes;
    std::vector<std::uint32_t> edges;
    bool empty() const { return edges.empty(); }
};

enum class PathClass : int
{
    TimeMonotone = 0,
    CommonSource = 1,
    ZigzagCertified = 2,
    ZigzagTraceOnly = 3,
};

inline const char *path_class_name(PathClass value)
{
    switch (value)
    {
    case PathClass::TimeMonotone: return "time_monotone";
    case PathClass::CommonSource: return "common_source";
    case PathClass::ZigzagCertified: return "zigzag_certified";
    case PathClass::ZigzagTraceOnly: break;
    }
    return "zigzag_trace_only";
}

enum class BridgeStatus : int
{
    Measured = 0,
    EndpointAnchored = 1,
    TraceOnly = 2,
};

inline const char *bridge_status_name(BridgeStatus value)
{
    switch (value)
    {
    case BridgeStatus::Measured: return "measured";
    case BridgeStatus::EndpointAnchored: return "endpoint_anchored";
    case BridgeStatus::TraceOnly: break;
    }
    return "trace_only";
}

// Whether a path's connectors are measured after their bridges. Endpoint-
// anchored bridges count as unmeasured here; they never make a path trace-only.
enum class ConnectorStatus : int
{
    NoBridge = 0,
    AllMeasured = 1,
    Mixed = 2,
    Unmeasured = 3,
};

inline const char *connector_status_name(ConnectorStatus value)
{
    switch (value)
    {
    case ConnectorStatus::NoBridge: return "no_bridge";
    case ConnectorStatus::AllMeasured: return "all_measured";
    case ConnectorStatus::Mixed: return "mixed";
    case ConnectorStatus::Unmeasured: break;
    }
    return "unmeasured";
}

struct PathBridge
{
    std::uint32_t edge = 0;
    int layer = 0;
    int mode_in = 0;
    int mode_out = 0;
    // First layer >= `layer` at which each connector is measured; -1 if never.
    int measured_layer_in = -1;
    int measured_layer_out = -1;
    BridgeStatus status = BridgeStatus::TraceOnly;
};

struct PathShape
{
    bool available = false;
    int length = 0; // edges
    int unitary = 0;
    int temporal_forward = 0;
    int temporal_backward = 0;
    int turns = 0;
    int valleys = 0;
    int peaks = 0;
    int min_layer = -1;
    int max_layer = -1;
    std::vector<int> gate_layers; // one per unitary edge, in path order
    std::vector<PathBridge> bridges; // one per peak, in path order
    int bridges_measured = 0;
    int bridges_endpoint = 0;
    int bridges_trace_only = 0;
    // Gates of this path that the doubled-network reduction cancels for its
    // two endpoints; > 0 means the path itself does not survive. Filled by
    // pair_turns (it needs the endpoints' cones), -1 when not computed.
    int cancelled_gates = -1;
    PathClass path_class = PathClass::TimeMonotone;
    ConnectorStatus connector_status = ConnectorStatus::NoBridge;

    bool zigzag() const { return available && turns >= 2; }
    bool trace_only_bridge() const { return available && path_class == PathClass::ZigzagTraceOnly; }
};

// Everything the turn diagnostics know about one final-slice pair.
struct PairTurns
{
    bool evaluated = false;
    // The two canonical paths of GatePathMetrics, read as shapes. Unavailable
    // when the graph carries no gate metrics or the endpoints share no
    // component.
    PathShape shortest_maximin;
    PathShape widest;
    // Over all non-backtracking walks between the two final-slice nodes; -1
    // when no walk exists.
    int min_turns = -1;
    bool common_source = false;
    bool certified = false;
    // The same with endpoint survival applied, i.e. the refined percolation
    // events. Both imply PairConnectivity::connected.
    bool connected_common_source = false;
    bool connected_certified = false;
    // The doubled-network test, without and with endpoint survival.
    bool doubled_network = false;
    bool connected_doubled = false;

    std::uint32_t flags() const
    {
        std::uint32_t mask = 0;
        mask |= connected_common_source ? FLAG_CONNECTED_COMMON_SOURCE : 0u;
        mask |= connected_certified ? FLAG_CONNECTED_CERTIFIED : 0u;
        mask |= connected_doubled ? FLAG_CONNECTED_DOUBLED : 0u;
        return mask;
    }
};

// Union-find with path halving and union by size. Small enough to keep here
// rather than grow a dependency: the graph has T*N nodes, a few thousand at
// the largest size this executable runs.
class DisjointSets
{
  public:
    void reset(std::size_t count)
    {
        parent_.resize(count);
        size_.assign(count, 1u);
        for (std::size_t i = 0; i < count; ++i)
        {
            parent_[i] = static_cast<std::uint32_t>(i);
        }
    }

    std::uint32_t find(std::uint32_t node)
    {
        while (parent_[node] != node)
        {
            parent_[node] = parent_[parent_[node]];
            node = parent_[node];
        }
        return node;
    }

    void unite(std::uint32_t a, std::uint32_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
        {
            return;
        }
        if (size_[a] < size_[b])
        {
            std::swap(a, b);
        }
        parent_[b] = a;
        size_[a] += size_[b];
    }

    std::uint32_t component_size(std::uint32_t node) { return size_[find(node)]; }

  private:
    std::vector<std::uint32_t> parent_;
    std::vector<std::uint32_t> size_;
};

// How many paths join two final-slice endpoints without sharing an edge, and
// without sharing an interior node. By Menger's theorem these are the minimum
// number of edges, and of spacetime nodes, whose removal disconnects the pair.
//
// This is the bottleneck diagnostic the shortest path cannot give. A pair joined
// by one fragile thread and a pair joined by a braid of independent channels can
// have the same shortest path; `vertex == 1` says the whole connection runs
// through a single articulation node, which is the classic place for a
// "connected" pair to carry nothing.
struct DisjointPaths
{
    int edge = 0;
    int vertex = 0;
};

// Channel-resolved connectivity: how much parity-odd (G, hopping) and
// parity-even (F, pairing) amplitude the graph's paths can actually carry.
//
// **This does not redefine the percolation event.** `connected` stays exactly
// what it was; these are diagnostics beside it. The suspicion they test is that
// binary percolation over-counts because it treats a nearly-diagonal gate --
// which transmits almost nothing in a given parity sector -- as an edge just as
// good as a maximally mixing one. If connected-zero pairs systematically have
// tiny channel weights, the gap is that counting artefact and not a physical
// incompleteness.
//
// Three numbers per channel, all from the same edge weights:
//
//   * `log_weight`: the largest achievable sum of log weights over a path,
//     i.e. the log of the most transmissive path's product. Logs rather than
//     products because a 50-layer path multiplies 50 numbers below 1 and
//     underflows; the log stays finite and additive. 0 is a perfect channel,
//     -inf is no channel at all.
//   * `bottleneck`: the largest achievable *minimum* edge weight over a path --
//     the widest-path value. A path can have a good product and one terrible
//     link; this is the terrible link.
//   * `multiplicity`: how many edge-disjoint paths survive if every edge weaker
//     than that bottleneck is deleted. 1 means the best channel is a single
//     thread, more means a braid.
struct ChannelConnectivity
{
    bool evaluated = false;
    double pair_log_weight = -std::numeric_limits<double>::infinity();
    double hop_log_weight = -std::numeric_limits<double>::infinity();
    double pair_bottleneck = 0.0;
    double hop_bottleneck = 0.0;
    int pair_multiplicity = 0;
    int hop_multiplicity = 0;
};

// How much of a pair's spacetime component actually carries the connection.
//
// Percolation asks a yes/no question, and a "yes" carried by one fragile thread
// through a large component is a very different object from a "yes" carried by
// a broad braid -- yet both are `connected = 1`. This decomposes the component
// into the part that can carry information between the endpoints and the part
// that cannot:
//
//   * the **backbone** is every spacetime vertex lying on *some* path between
//     the two endpoints. Computed exactly, from the block-cut tree: a vertex is
//     on some s-t path iff its biconnected block lies on the tree path between
//     the endpoints' blocks. (Inside a 2-connected block, any two vertices are
//     joined by a path through any third, so a block on the path contributes
//     all of its vertices.)
//   * the **articulation** count is the number of cut vertices strictly between
//     the endpoints -- vertices whose removal alone disconnects them. `0` means
//     the connection is 2-connected; `k > 0` means it passes single-file
//     through k separate places.
//   * **dangling** vertices are the rest of the component: reachable, but not on
//     any endpoint-to-endpoint path. They are where a "parasitic" component
//     shares the endpoints' information with sites that are not the partner.
//
// A large component with a small backbone and many dangling vertices is exactly
// the shape the percolation picture is accused of over-counting.
struct ComponentAnatomy
{
    bool evaluated = false;
    std::uint32_t component_nodes = 0;
    // Final-slice sites sharing the component, endpoints included. The
    // competition for the endpoints' correlations is with these, since only a
    // final-slice site is a site anything is measured on.
    std::uint32_t final_sites_in_component = 0;
    std::uint32_t backbone_nodes = 0;
    std::uint32_t articulation_nodes = 0;
    std::uint32_t dangling_nodes = 0;
    std::uint32_t branches = 0;
    double mean_branch_size = 0.0;
    std::uint32_t max_branch_size = 0;
    std::uint32_t degree_i = 0;
    std::uint32_t degree_j = 0;
    // One shortest path, split by edge kind. Which shortest path is the BFS
    // tree's, so the split is a representative rather than an invariant when
    // several shortest paths of different composition exist; the total is the
    // invariant already reported as `shortest_path`.
    std::int32_t shortest_path_gates = -1;
    std::int32_t shortest_path_temporal = -1;
    // By Menger these equal the edge- and vertex-disjoint path counts exactly,
    // so they are the same numbers under the name the cut question asks for.
    int min_edge_cut = 0;
    int min_vertex_cut = 0;
};

// Unit-capacity max flow by breadth-first augmentation. The graph has a few
// thousand arcs and every node has degree at most three (one bond, two temporal
// edges), so the flow is at most three and this is a handful of BFS sweeps.
class UnitFlowNetwork
{
  public:
    explicit UnitFlowNetwork(std::size_t nodes) : arcs_(nodes) {}

    void add(int from, int to, int capacity, int reverse_capacity)
    {
        arcs_[static_cast<std::size_t>(from)].push_back(
            {to, capacity, static_cast<int>(arcs_[static_cast<std::size_t>(to)].size())});
        arcs_[static_cast<std::size_t>(to)].push_back(
            {from, reverse_capacity,
             static_cast<int>(arcs_[static_cast<std::size_t>(from)].size()) - 1});
    }

    int max_flow(int source, int sink, int limit)
    {
        int flow = 0;
        std::vector<std::pair<int, int>> parent(arcs_.size());
        std::vector<int> queue;
        while (flow < limit)
        {
            std::fill(parent.begin(), parent.end(), std::make_pair(-1, -1));
            parent[static_cast<std::size_t>(source)] = {source, -1};
            queue.assign(1, source);
            for (std::size_t head = 0; head < queue.size() && parent[static_cast<std::size_t>(sink)].first < 0;
                 ++head)
            {
                const int node = queue[head];
                const auto &out = arcs_[static_cast<std::size_t>(node)];
                for (std::size_t slot = 0; slot < out.size(); ++slot)
                {
                    const Arc &arc = out[slot];
                    if (arc.capacity > 0 && parent[static_cast<std::size_t>(arc.to)].first < 0)
                    {
                        parent[static_cast<std::size_t>(arc.to)] = {node, static_cast<int>(slot)};
                        queue.push_back(arc.to);
                    }
                }
            }
            if (parent[static_cast<std::size_t>(sink)].first < 0)
            {
                break;
            }
            for (int node = sink; node != source;)
            {
                const auto [previous, slot] = parent[static_cast<std::size_t>(node)];
                Arc &arc = arcs_[static_cast<std::size_t>(previous)][static_cast<std::size_t>(slot)];
                arc.capacity -= 1;
                arcs_[static_cast<std::size_t>(node)][static_cast<std::size_t>(arc.reverse)].capacity += 1;
                node = previous;
            }
            ++flow;
        }
        return flow;
    }

  private:
    struct Arc
    {
        int to;
        int capacity;
        int reverse;
    };
    std::vector<std::vector<Arc>> arcs_;
};

class ConnectivityIndex
{
  public:
    // Build the graph for one trajectory. `with_paths` adds one BFS sweep per
    // site to fill the shortest-path matrix; the component labels and every
    // flag are available either way.
    //
    // `with_gate_paths` additionally precomputes GatePathMetrics for every
    // final-slice pair (one bottleneck sweep per site per capacity, plus the
    // shortest-path DAG walks), so gate_path_metrics() is a lookup. It changes
    // nothing else the index reports.
    //
    // `with_turns` additionally precomputes PairTurns for every final-slice
    // pair: two walk searches per site, plus -- when the graph carries gate
    // metrics -- both canonical paths of every pair read as shapes. Like the
    // gate paths it changes nothing else the index reports.
    void build(const LogicalHistory &history, bool with_paths, bool with_gate_paths = false,
               bool with_turns = false)
    {
        ready_ = false;
        gate_paths_.clear();
        pair_turns_.clear();
        n_ = history.n;
        layers_ = static_cast<int>(history.layers.size());
        if (!history.valid || n_ <= 0 || layers_ <= 0)
        {
            return;
        }

        const std::size_t nodes = static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
        sets_.reset(nodes);

        // Degrees first, so the adjacency can be one flat CSR array rather
        // than N*T little vectors.
        degree_.assign(nodes + 1u, 0u);
        edges_ = 0;
        auto count_edge = [&](std::uint32_t a, std::uint32_t b, const GraphEdge &, double, double) {
            ++degree_[a];
            ++degree_[b];
            ++edges_;
        };
        for_each_edge(history, count_edge);

        offset_.assign(nodes + 1u, 0u);
        for (std::size_t i = 0; i < nodes; ++i)
        {
            offset_[i + 1u] = offset_[i] + degree_[i];
        }
        adjacency_.assign(static_cast<std::size_t>(edges_) * 2u, 0u);
        edge_kind_.assign(static_cast<std::size_t>(edges_) * 2u, 0u);
        edge_from_.assign(static_cast<std::size_t>(edges_) * 2u, 0u);
        weight_pair_.assign(static_cast<std::size_t>(edges_) * 2u, 1.0);
        weight_hop_.assign(static_cast<std::size_t>(edges_) * 2u, 1.0);
        arc_edge_.assign(static_cast<std::size_t>(edges_) * 2u, 0u);
        graph_edges_.clear();
        graph_edges_.reserve(static_cast<std::size_t>(edges_));
        gate_metrics_ = true;
        cursor_ = offset_;
        auto add_edge = [&](std::uint32_t a, std::uint32_t b, const GraphEdge &meta,
                            double pair_weight, double hop_weight) {
            const bool gate = meta.kind == GraphEdgeKind::Unitary;
            const std::uint32_t edge_id = static_cast<std::uint32_t>(graph_edges_.size());
            graph_edges_.push_back(meta);
            if (gate && !(std::isfinite(meta.entangling_power) &&
                          std::isfinite(meta.even_block_mixing) &&
                          std::isfinite(meta.odd_block_mixing)))
            {
                gate_metrics_ = false;
            }
            arc_edge_[cursor_[a]] = edge_id;
            arc_edge_[cursor_[b]] = edge_id;
            edge_kind_[cursor_[a]] = gate ? 1u : 0u;
            edge_from_[cursor_[a]] = a;
            weight_pair_[cursor_[a]] = pair_weight;
            weight_hop_[cursor_[a]] = hop_weight;
            adjacency_[cursor_[a]++] = b;
            edge_kind_[cursor_[b]] = gate ? 1u : 0u;
            edge_from_[cursor_[b]] = b;
            weight_pair_[cursor_[b]] = pair_weight;
            weight_hop_[cursor_[b]] = hop_weight;
            adjacency_[cursor_[b]++] = a;
            sets_.unite(a, b);
        };
        for_each_edge(history, add_edge);

        // The turn searches rely on every node carrying at most one gate edge
        // (a mode sits in one bond per layer): a walk may then never take two
        // gate edges in a row, because the second could only be the first one
        // retraced. A history that breaks this -- two bonds on one mode in one
        // layer -- would have its turns silently miscounted, so it is recorded
        // here and the turn diagnostics refuse it. RPPU and qRPPU never do.
        single_gate_nodes_ = true;
        for (std::size_t node = 0; node < nodes && single_gate_nodes_; ++node)
        {
            int gates = 0;
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                gates += graph_edges_[arc_edge_[slot]].kind == GraphEdgeKind::Unitary ? 1 : 0;
            }
            single_gate_nodes_ = gates <= 1;
        }

        // Per-site final-slice facts. `mode_at_site` is the identity for every
        // circuit in the registry, but going through it is what keeps a future
        // transporting circuit from mislabelling both endpoints at once.
        const LogicalLayer &last = history.layers[static_cast<std::size_t>(layers_ - 1)];
        mode_of_site_ = history.mode_at_site;
        root_.assign(static_cast<std::size_t>(n_), 0u);
        component_size_.assign(static_cast<std::size_t>(n_), 0u);
        survives_.assign(static_cast<std::size_t>(n_), 0u);
        idle_.assign(static_cast<std::size_t>(n_), 0);
        for (int site = 0; site < n_; ++site)
        {
            const int mode = mode_of_site_[static_cast<std::size_t>(site)];
            const std::uint32_t node = node_index(layers_ - 1, mode);
            root_[static_cast<std::size_t>(site)] = sets_.find(node);
            component_size_[static_cast<std::size_t>(site)] = sets_.component_size(node);
            survives_[static_cast<std::size_t>(site)] =
                last.measured[static_cast<std::size_t>(mode)] ? 0u : 1u;
            idle_[static_cast<std::size_t>(site)] = layers_since_measurement(history, mode);
        }

        paths_.clear();
        if (with_paths)
        {
            fill_shortest_paths();
        }
        // One biconnected decomposition serves every pair of the trajectory:
        // the blocks do not depend on which endpoints are asked about, only the
        // tree path between them does. Doing it here rather than per anchor is
        // what keeps the anatomy affordable at 200k anchors.
        build_block_cut_tree();
        build_measurement_table(history);
        ready_ = true;
        if (with_gate_paths && gate_metrics_)
        {
            fill_gate_paths();
        }
        if (with_turns)
        {
            if (!single_gate_nodes_)
            {
                throw std::invalid_argument(
                    "Turn diagnostics need at most one gate per mode per layer; this history "
                    "puts two bonds on one mode in one layer, so its turns cannot be counted.");
            }
            fill_turns();
        }
    }

    bool has_turns() const { return ready_ && !pair_turns_.empty(); }
    // Whether the turn diagnostics are defined on this graph at all; see build().
    bool turns_supported() const { return ready_ && single_gate_nodes_; }

    // Spacetime node <-> (layer, mode).
    int node_layer(std::uint32_t node) const { return static_cast<int>(node / static_cast<std::uint32_t>(n_)); }
    int node_mode(std::uint32_t node) const { return static_cast<int>(node % static_cast<std::uint32_t>(n_)); }

    // The first layer >= `layer` at which `mode` is measured, or -1 if it never
    // is. Includes the final layer, whose measurements decide survival.
    int next_measurement(int layer, int mode) const
    {
        if (!ready_ || layer < 0 || layer >= layers_ || mode < 0 || mode >= n_)
        {
            return -1;
        }
        return next_measurement_[static_cast<std::size_t>(layer) * static_cast<std::size_t>(n_) +
                                 static_cast<std::size_t>(mode)];
    }

    // The canonical paths behind GatePathMetrics, with their ordered nodes.
    // Identical to shortest_maximin_path / widest_gate_path, which return only
    // the edges of these.
    GraphPath shortest_maximin_trace(int site_i, int site_j) const
    {
        if (!gate_query_valid(site_i, site_j))
        {
            return {};
        }
        const std::uint32_t source = final_node(site_i);
        std::vector<std::int32_t> distance;
        std::vector<std::uint32_t> order;
        bfs(source, distance, order);
        std::vector<double> best;
        shortest_bottleneck(distance, order, source, best);
        return shortest_maximin_from(source, final_node(site_j), distance, order, best);
    }

    GraphPath widest_gate_trace(int site_i, int site_j,
                                GateCapacity which = GateCapacity::EntanglingPower) const
    {
        if (!gate_query_valid(site_i, site_j))
        {
            return {};
        }
        const std::uint32_t source = final_node(site_i);
        std::vector<double> widest;
        widest_from(source, which, widest);
        return widest_from_trace(source, final_node(site_j), which, widest);
    }

    // Read a path as a turn shape. `site_i`/`site_j` are the endpoints whose
    // modes make a bridge endpoint-anchored. An empty or malformed path comes
    // back unavailable.
    PathShape path_shape(const GraphPath &path, int site_i, int site_j) const
    {
        PathShape out;
        if (!ready_ || path.edges.empty() || path.nodes.size() != path.edges.size() + 1u ||
            site_i < 0 || site_j < 0 || site_i >= n_ || site_j >= n_)
        {
            return out;
        }
        const int mode_i = mode_of_site_[static_cast<std::size_t>(site_i)];
        const int mode_j = mode_of_site_[static_cast<std::size_t>(site_j)];
        out.length = static_cast<int>(path.edges.size());
        out.min_layer = node_layer(path.nodes.front());
        out.max_layer = out.min_layer;
        for (std::uint32_t node : path.nodes)
        {
            out.min_layer = std::min(out.min_layer, node_layer(node));
            out.max_layer = std::max(out.max_layer, node_layer(node));
        }
        int last = 0; // +1 forward, -1 backward, 0 none yet
        for (std::size_t k = 0; k < path.edges.size(); ++k)
        {
            const GraphEdge &edge = graph_edges_[path.edges[k]];
            const std::uint32_t from = path.nodes[k];
            const std::uint32_t to = path.nodes[k + 1u];
            if (!((edge.a == from && edge.b == to) || (edge.a == to && edge.b == from)))
            {
                return PathShape{};
            }
            if (edge.kind == GraphEdgeKind::Unitary)
            {
                ++out.unitary;
                out.gate_layers.push_back(edge.layer);
                continue;
            }
            const int direction = node_layer(to) > node_layer(from) ? 1 : -1;
            (direction > 0 ? out.temporal_forward : out.temporal_backward) += 1;
            if (last != 0 && direction != last)
            {
                ++out.turns;
                // A turn needs the gate edge between its two temporal edges; a
                // simple path has exactly one there.
                if (k == 0 || graph_edges_[path.edges[k - 1u]].kind != GraphEdgeKind::Unitary)
                {
                    return PathShape{};
                }
                if (last > 0)
                {
                    ++out.peaks;
                    PathBridge bridge;
                    bridge.edge = path.edges[k - 1u];
                    bridge.layer = graph_edges_[bridge.edge].layer;
                    bridge.mode_in = node_mode(path.nodes[k - 1u]);
                    bridge.mode_out = node_mode(from);
                    bridge.measured_layer_in = next_measurement(bridge.layer, bridge.mode_in);
                    bridge.measured_layer_out = next_measurement(bridge.layer, bridge.mode_out);
                    if (bridge.measured_layer_in >= 0 || bridge.measured_layer_out >= 0)
                    {
                        bridge.status = BridgeStatus::Measured;
                        ++out.bridges_measured;
                    }
                    else if (bridge.mode_in == mode_i || bridge.mode_in == mode_j ||
                             bridge.mode_out == mode_i || bridge.mode_out == mode_j)
                    {
                        bridge.status = BridgeStatus::EndpointAnchored;
                        ++out.bridges_endpoint;
                    }
                    else
                    {
                        bridge.status = BridgeStatus::TraceOnly;
                        ++out.bridges_trace_only;
                    }
                    out.bridges.push_back(bridge);
                }
                else
                {
                    ++out.valleys;
                }
            }
            last = direction;
        }
        if (out.turns == 0)
        {
            out.path_class = PathClass::TimeMonotone;
        }
        else if (out.turns == 1 && out.peaks == 0)
        {
            out.path_class = PathClass::CommonSource;
        }
        else
        {
            out.path_class = out.bridges_trace_only > 0 ? PathClass::ZigzagTraceOnly
                                                        : PathClass::ZigzagCertified;
        }
        const int bridges = static_cast<int>(out.bridges.size());
        out.connector_status = bridges == 0 ? ConnectorStatus::NoBridge
                               : out.bridges_measured == bridges ? ConnectorStatus::AllMeasured
                               : out.bridges_measured == 0       ? ConnectorStatus::Unmeasured
                                                                 : ConnectorStatus::Mixed;
        out.available = true;
        return out;
    }

    // The turn diagnostics for one final-slice pair. A lookup when build()
    // precomputed them, computed on demand otherwise.
    PairTurns pair_turns(int site_i, int site_j) const
    {
        if (!ready_ || site_i == site_j || site_i < 0 || site_j < 0 || site_i >= n_ ||
            site_j >= n_)
        {
            return {};
        }
        if (!pair_turns_.empty())
        {
            return pair_turns_[static_cast<std::size_t>(site_i) * static_cast<std::size_t>(n_) +
                               static_cast<std::size_t>(site_j)];
        }
        if (!single_gate_nodes_)
        {
            return {}; // unevaluated: the turn searches are not defined here
        }
        std::vector<int> turns;
        std::vector<std::uint8_t> certified;
        const std::uint32_t source = final_node(site_i);
        walk_turns_from(source, turns);
        walk_certified_from(source, certified);
        PairTurns out;
        fill_pair_existence(site_i, site_j, turns, certified, out);
        std::vector<std::uint8_t> measurement;
        std::vector<std::uint8_t> cone_i;
        std::vector<std::uint8_t> cone_j;
        measurement_cone(measurement);
        endpoint_cone(site_i, cone_i);
        endpoint_cone(site_j, cone_j);
        fill_doubled(site_i, site_j, measurement, cone_i, cone_j, out);
        if (gate_query_valid(site_i, site_j))
        {
            out.shortest_maximin = path_shape(shortest_maximin_trace(site_i, site_j), site_i, site_j);
            out.widest = path_shape(widest_gate_trace(site_i, site_j), site_i, site_j);
            count_cancelled(shortest_maximin_trace(site_i, site_j), measurement, cone_i, cone_j,
                            out.shortest_maximin);
            count_cancelled(widest_gate_trace(site_i, site_j), measurement, cone_i, cone_j, out.widest);
        }
        return out;
    }

    // The doubled-network relevance of one node for a pair: in the past light
    // cone of a measurement or of either endpoint. Exposed for tests and the
    // offline tools; O(V) per call, so not for inner loops.
    bool doubled_relevant(std::uint32_t node, int site_i, int site_j) const
    {
        std::vector<std::uint8_t> measurement;
        std::vector<std::uint8_t> cone_i;
        std::vector<std::uint8_t> cone_j;
        measurement_cone(measurement);
        endpoint_cone(site_i, cone_i);
        endpoint_cone(site_j, cone_j);
        return measurement[node] != 0u || cone_i[node] != 0u || cone_j[node] != 0u;
    }

    // Whether every unitary edge carries finite gate metrics (RPPU/qRPPU do;
    // the other circuits leave them NaN).
    bool has_gate_metrics() const { return ready_ && gate_metrics_; }
    const std::vector<GraphEdge> &graph_edges() const { return graph_edges_; }
    std::uint32_t final_node(int site) const
    {
        return node_index(layers_ - 1, mode_of_site_[static_cast<std::size_t>(site)]);
    }
    int sites() const { return n_; }
    // The arcs leaving one node, with the edge each one travels.
    std::vector<GraphArc> arcs(std::uint32_t node) const
    {
        std::vector<GraphArc> out;
        for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
        {
            out.push_back({adjacency_[slot], arc_edge_[slot]});
        }
        return out;
    }

    // Gate quality between two final-slice endpoints. A lookup when build()
    // precomputed it, computed on demand otherwise.
    GatePathMetrics gate_path_metrics(int site_i, int site_j) const
    {
        if (!gate_query_valid(site_i, site_j))
        {
            return {};
        }
        if (!gate_paths_.empty())
        {
            return gate_paths_[static_cast<std::size_t>(site_i) * static_cast<std::size_t>(n_) +
                               static_cast<std::size_t>(site_j)];
        }
        return compute_gate_path(site_i, site_j);
    }

    // The edges (graph_edges() indices, from endpoint i to endpoint j) of the
    // canonical path behind each GatePathMetrics value: the shortest path
    // attaining the shortest-maximin value with the fewest gates, and the
    // shortest path whose every edge is at least the widest value. Ties go to
    // the lowest adjacency slot, so a regenerated circuit gives the same path.
    // Empty when the query is not available.
    std::vector<std::uint32_t> shortest_maximin_path(int site_i, int site_j) const
    {
        return shortest_maximin_trace(site_i, site_j).edges;
    }

    std::vector<std::uint32_t> widest_gate_path(int site_i, int site_j,
                                                GateCapacity which = GateCapacity::EntanglingPower) const
    {
        return widest_gate_trace(site_i, site_j, which).edges;
    }

    bool ready() const { return ready_; }
    int layers() const { return layers_; }
    bool has_paths() const { return !paths_.empty(); }

    PairConnectivity query(int site_i, int site_j) const
    {
        PairConnectivity out;
        if (!ready_ || site_i < 0 || site_j < 0 || site_i >= n_ || site_j >= n_)
        {
            return out;
        }
        const std::size_t i = static_cast<std::size_t>(site_i);
        const std::size_t j = static_cast<std::size_t>(site_j);
        out.evaluated = true;
        out.survives_i = survives_[i] != 0u;
        out.survives_j = survives_[j] != 0u;
        out.interior_path = root_[i] == root_[j];
        out.connected = out.survives_i && out.survives_j && out.interior_path;
        out.component_size = component_size_[i];
        out.idle_i = idle_[i];
        out.idle_j = idle_[j];
        if (!paths_.empty())
        {
            out.shortest_path = paths_[i * static_cast<std::size_t>(n_) + j];
        }
        return out;
    }

    // Edge- and vertex-disjoint path counts between two final-slice endpoints.
    // Computed on demand rather than for every pair, because only the pairs an
    // analysis singles out need it; returns zeros when the index is not ready
    // or the endpoints share no component.
    DisjointPaths disjoint_paths(int site_i, int site_j) const
    {
        DisjointPaths out;
        if (!ready_ || site_i == site_j || site_i < 0 || site_j < 0 || site_i >= n_ ||
            site_j >= n_)
        {
            return out;
        }
        if (root_[static_cast<std::size_t>(site_i)] != root_[static_cast<std::size_t>(site_j)])
        {
            return out;
        }
        const std::size_t nodes = static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
        const int source =
            static_cast<int>(node_index(layers_ - 1, mode_of_site_[static_cast<std::size_t>(site_i)]));
        const int sink =
            static_cast<int>(node_index(layers_ - 1, mode_of_site_[static_cast<std::size_t>(site_j)]));
        constexpr int limit = 64;

        // Edge-disjoint: every undirected edge carries one unit either way.
        UnitFlowNetwork edges(nodes);
        for (std::size_t u = 0; u < nodes; ++u)
        {
            for (std::uint32_t slot = offset_[u]; slot < offset_[u + 1u]; ++slot)
            {
                const std::uint32_t v = adjacency_[slot];
                if (u < v)
                {
                    edges.add(static_cast<int>(u), static_cast<int>(v), 1, 1);
                }
            }
        }
        out.edge = edges.max_flow(source, sink, limit);

        // Vertex-disjoint: split every node into in (2u) and out (2u+1) with a
        // unit arc between them, except the two endpoints, which may be shared.
        UnitFlowNetwork vertices(2u * nodes);
        for (std::size_t u = 0; u < nodes; ++u)
        {
            const int unbounded = limit;
            const bool endpoint = static_cast<int>(u) == source || static_cast<int>(u) == sink;
            vertices.add(static_cast<int>(2u * u), static_cast<int>(2u * u + 1u),
                         endpoint ? unbounded : 1, 0);
            for (std::uint32_t slot = offset_[u]; slot < offset_[u + 1u]; ++slot)
            {
                const std::uint32_t v = adjacency_[slot];
                vertices.add(static_cast<int>(2u * u + 1u), static_cast<int>(2u * v), 1, 0);
            }
        }
        out.vertex = vertices.max_flow(2 * source + 1, 2 * sink, limit);
        return out;
    }

    // The component anatomy for one final-slice pair. Cheap: a walk of the
    // block-cut tree plus one sweep of the component, both linear.
    ComponentAnatomy anatomy(int site_i, int site_j) const
    {
        ComponentAnatomy out;
        if (!ready_ || site_i == site_j || site_i < 0 || site_j < 0 || site_i >= n_ ||
            site_j >= n_)
        {
            return out;
        }
        const std::size_t i = static_cast<std::size_t>(site_i);
        const std::size_t j = static_cast<std::size_t>(site_j);
        if (root_[i] != root_[j])
        {
            return out;
        }
        out.evaluated = true;
        const std::uint32_t source = node_index(layers_ - 1, mode_of_site_[i]);
        const std::uint32_t sink = node_index(layers_ - 1, mode_of_site_[j]);
        out.component_nodes = component_size_[i];
        out.degree_i = offset_[source + 1u] - offset_[source];
        out.degree_j = offset_[sink + 1u] - offset_[sink];
        for (int site = 0; site < n_; ++site)
        {
            out.final_sites_in_component +=
                root_[static_cast<std::size_t>(site)] == root_[i] ? 1u : 0u;
        }

        const DisjointPaths cuts = disjoint_paths(site_i, site_j);
        out.min_edge_cut = cuts.edge;
        out.min_vertex_cut = cuts.vertex;

        split_shortest_path(source, sink, out.shortest_path_gates, out.shortest_path_temporal);

        // Mark the backbone: every vertex of every block on the block-cut-tree
        // path between the endpoints' tree nodes.
        const std::size_t nodes = static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
        std::vector<std::uint8_t> backbone(nodes, 0u);
        mark_backbone(source, sink, backbone, out.articulation_nodes);
        for (std::uint8_t flag : backbone)
        {
            out.backbone_nodes += flag != 0u ? 1u : 0u;
        }
        out.dangling_nodes = out.component_nodes > out.backbone_nodes
                                 ? out.component_nodes - out.backbone_nodes
                                 : 0u;

        // Branches: connected components of the component minus the backbone.
        std::vector<std::uint8_t> seen(nodes, 0u);
        std::vector<std::uint32_t> stack;
        std::uint64_t total = 0;
        std::vector<std::uint32_t> component;
        collect_component(source, component);
        for (std::uint32_t start : component)
        {
            if (backbone[start] != 0u || seen[start] != 0u)
            {
                continue;
            }
            std::uint32_t size = 0;
            stack.assign(1, start);
            seen[start] = 1u;
            while (!stack.empty())
            {
                const std::uint32_t node = stack.back();
                stack.pop_back();
                ++size;
                for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
                {
                    const std::uint32_t next = adjacency_[slot];
                    if (backbone[next] == 0u && seen[next] == 0u)
                    {
                        seen[next] = 1u;
                        stack.push_back(next);
                    }
                }
            }
            ++out.branches;
            total += size;
            out.max_branch_size = std::max(out.max_branch_size, size);
        }
        out.mean_branch_size =
            out.branches > 0 ? static_cast<double>(total) / static_cast<double>(out.branches) : 0.0;
        return out;
    }

    // Per final-slice site: does its worldline's last node lie on the i-j
    // backbone, is it in the same component, and how far is it from the nearer
    // endpoint. This is what the four-site helper ranking selects on, so it is
    // exposed rather than recomputed there.
    void site_roles(int site_i, int site_j, std::vector<std::uint8_t> &same_component,
                    std::vector<std::uint8_t> &on_backbone,
                    std::vector<std::int32_t> &distance) const
    {
        same_component.assign(static_cast<std::size_t>(std::max(0, n_)), 0u);
        on_backbone.assign(static_cast<std::size_t>(std::max(0, n_)), 0u);
        distance.assign(static_cast<std::size_t>(std::max(0, n_)), -1);
        if (!ready_ || site_i < 0 || site_j < 0 || site_i >= n_ || site_j >= n_)
        {
            return;
        }
        const std::size_t i = static_cast<std::size_t>(site_i);
        const std::size_t j = static_cast<std::size_t>(site_j);
        for (int site = 0; site < n_; ++site)
        {
            same_component[static_cast<std::size_t>(site)] =
                root_[static_cast<std::size_t>(site)] == root_[i] ? 1u : 0u;
            if (!paths_.empty())
            {
                const std::int32_t di =
                    paths_[i * static_cast<std::size_t>(n_) + static_cast<std::size_t>(site)];
                const std::int32_t dj =
                    paths_[j * static_cast<std::size_t>(n_) + static_cast<std::size_t>(site)];
                if (di >= 0 && dj >= 0)
                {
                    distance[static_cast<std::size_t>(site)] = std::min(di, dj);
                }
                else
                {
                    distance[static_cast<std::size_t>(site)] = std::max(di, dj);
                }
            }
        }
        if (root_[i] != root_[j])
        {
            return;
        }
        const std::size_t nodes = static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
        std::vector<std::uint8_t> backbone(nodes, 0u);
        std::uint32_t ignored = 0;
        mark_backbone(node_index(layers_ - 1, mode_of_site_[i]),
                      node_index(layers_ - 1, mode_of_site_[j]), backbone, ignored);
        for (int site = 0; site < n_; ++site)
        {
            const std::uint32_t node =
                node_index(layers_ - 1, mode_of_site_[static_cast<std::size_t>(site)]);
            on_backbone[static_cast<std::size_t>(site)] = backbone[node];
        }
    }

    // Channel-resolved path strengths between two final-slice endpoints.
    //
    // A temporal edge is weight 1 in both channels: an idle mode transmits
    // whatever it holds, perfectly, in either parity sector. Only gates
    // attenuate.
    ChannelConnectivity channels(int site_i, int site_j) const
    {
        ChannelConnectivity out;
        if (!ready_ || site_i == site_j || site_i < 0 || site_j < 0 || site_i >= n_ ||
            site_j >= n_ || weight_pair_.empty())
        {
            return out;
        }
        if (root_[static_cast<std::size_t>(site_i)] != root_[static_cast<std::size_t>(site_j)])
        {
            return out;
        }
        out.evaluated = true;
        const std::uint32_t source =
            node_index(layers_ - 1, mode_of_site_[static_cast<std::size_t>(site_i)]);
        const std::uint32_t sink =
            node_index(layers_ - 1, mode_of_site_[static_cast<std::size_t>(site_j)]);
        strongest_path(weight_pair_, source, sink, out.pair_log_weight, out.pair_bottleneck,
                       out.pair_multiplicity);
        strongest_path(weight_hop_, source, sink, out.hop_log_weight, out.hop_bottleneck,
                       out.hop_multiplicity);
        return out;
    }

  private:
    std::size_t node_count() const
    {
        return static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
    }

    bool gate_query_valid(int site_i, int site_j) const
    {
        return ready_ && gate_metrics_ && site_i != site_j && site_i >= 0 && site_j >= 0 &&
               site_i < n_ && site_j < n_ &&
               root_[static_cast<std::size_t>(site_i)] == root_[static_cast<std::size_t>(site_j)];
    }

    double capacity(std::uint32_t slot, GateCapacity which) const
    {
        return edge_capacity(graph_edges_[arc_edge_[slot]], which);
    }

    bool unitary(std::uint32_t slot) const
    {
        return graph_edges_[arc_edge_[slot]].kind == GraphEdgeKind::Unitary;
    }

    void bfs(std::uint32_t source, std::vector<std::int32_t> &distance,
             std::vector<std::uint32_t> &order) const
    {
        distance.assign(node_count(), -1);
        order.assign(1, source);
        distance[source] = 0;
        for (std::size_t head = 0; head < order.size(); ++head)
        {
            const std::uint32_t node = order[head];
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const std::uint32_t next = adjacency_[slot];
                if (distance[next] < 0)
                {
                    distance[next] = distance[node] + 1;
                    order.push_back(next);
                }
            }
        }
    }

    // Max-min entangling power over shortest paths, for every node: a DP over
    // the BFS layering, in BFS order so every predecessor is final first.
    void shortest_bottleneck(const std::vector<std::int32_t> &distance,
                             const std::vector<std::uint32_t> &order, std::uint32_t source,
                             std::vector<double> &best) const
    {
        best.assign(node_count(), -1.0);
        best[source] = std::numeric_limits<double>::infinity();
        for (std::uint32_t node : order)
        {
            if (node == source)
            {
                continue;
            }
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const std::uint32_t previous = adjacency_[slot];
                if (distance[previous] == distance[node] - 1)
                {
                    best[node] = std::max(best[node],
                                          std::min(best[previous],
                                                   capacity(slot, GateCapacity::EntanglingPower)));
                }
            }
        }
    }

    // Fewest gates over shortest paths using only edges of capacity >= floor.
    // Optionally records, per node, the in-slot of the canonical predecessor.
    void fewest_gates(const std::vector<std::int32_t> &distance,
                      const std::vector<std::uint32_t> &order, std::uint32_t source, double floor,
                      std::vector<int> &count, std::vector<std::uint32_t> *via) const
    {
        constexpr int NONE = std::numeric_limits<int>::max();
        count.assign(node_count(), NONE);
        if (via != nullptr)
        {
            via->assign(node_count(), UNSET);
        }
        count[source] = 0;
        for (std::uint32_t node : order)
        {
            if (node == source)
            {
                continue;
            }
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const std::uint32_t previous = adjacency_[slot];
                if (distance[previous] != distance[node] - 1 || count[previous] == NONE ||
                    !(capacity(slot, GateCapacity::EntanglingPower) >= floor))
                {
                    continue;
                }
                const int candidate = count[previous] + (unitary(slot) ? 1 : 0);
                if (candidate < count[node])
                {
                    count[node] = candidate;
                    if (via != nullptr)
                    {
                        // The slot is node's own arc back to `previous`; trace_back
                        // follows it.
                        (*via)[node] = slot;
                    }
                }
            }
        }
    }

    // Widest (max-min capacity) path value from `source` to every node.
    // Temporal edges carry 1.0, so they never lower a bottleneck.
    void widest_from(std::uint32_t source, GateCapacity which, std::vector<double> &widest) const
    {
        widest.assign(node_count(), -1.0);
        widest[source] = std::numeric_limits<double>::infinity();
        using Item = std::pair<double, std::uint32_t>;
        std::priority_queue<Item> heap;
        heap.push({widest[source], source});
        std::vector<std::uint8_t> done(node_count(), 0u);
        while (!heap.empty())
        {
            const auto [value, node] = heap.top();
            heap.pop();
            if (done[node] != 0u)
            {
                continue;
            }
            done[node] = 1u;
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const std::uint32_t next = adjacency_[slot];
                const double candidate = std::min(value, capacity(slot, which));
                if (done[next] == 0u && candidate > widest[next])
                {
                    widest[next] = candidate;
                    heap.push({candidate, next});
                }
            }
        }
    }

    // The canonical shortest-maximin path from per-source structures: among
    // shortest paths whose weakest gate attains the maximin, the one with the
    // fewest gates, ties to the lowest adjacency slot.
    GraphPath shortest_maximin_from(std::uint32_t source, std::uint32_t sink,
                                    const std::vector<std::int32_t> &distance,
                                    const std::vector<std::uint32_t> &order,
                                    const std::vector<double> &best) const
    {
        std::vector<int> count;
        std::vector<std::uint32_t> via;
        fewest_gates(distance, order, source, best[sink], count, &via);
        return trace_back(source, sink, via);
    }

    // The canonical widest path: the shortest path inside the subgraph of edges
    // at least as wide as the widest-path value.
    GraphPath widest_from_trace(std::uint32_t source, std::uint32_t sink, GateCapacity which,
                                const std::vector<double> &widest) const
    {
        const double floor = widest[sink];
        const std::size_t nodes = node_count();
        std::vector<std::uint32_t> via(nodes, UNSET);
        std::vector<std::uint8_t> seen(nodes, 0u);
        std::vector<std::uint32_t> queue{source};
        seen[source] = 1u;
        for (std::size_t head = 0; head < queue.size(); ++head)
        {
            const std::uint32_t node = queue[head];
            if (node == sink)
            {
                break;
            }
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const std::uint32_t next = adjacency_[slot];
                if (seen[next] == 0u && capacity(slot, which) >= floor)
                {
                    seen[next] = 1u;
                    via[next] = slot;
                    queue.push_back(next);
                }
            }
        }
        return trace_back(source, sink, via);
    }

    // Follow the recorded in-slots back from sink to source: the ordered path.
    GraphPath trace_back(std::uint32_t source, std::uint32_t sink,
                         const std::vector<std::uint32_t> &via) const
    {
        GraphPath out;
        std::uint32_t node = sink;
        out.nodes.push_back(node);
        while (node != source)
        {
            const std::uint32_t slot = via[node];
            if (slot == UNSET)
            {
                return {};
            }
            out.edges.push_back(arc_edge_[slot]);
            const GraphEdge &edge = graph_edges_[arc_edge_[slot]];
            node = edge.a == node ? edge.b : edge.a;
            out.nodes.push_back(node);
        }
        std::reverse(out.nodes.begin(), out.nodes.end());
        std::reverse(out.edges.begin(), out.edges.end());
        return out;
    }

    // next_measurement_[layer * n + mode], and whether each gate is a measured
    // bridge candidate (a connector measured at or after its layer).
    void build_measurement_table(const LogicalHistory &history)
    {
        next_measurement_.assign(node_count(), -1);
        for (int mode = 0; mode < n_; ++mode)
        {
            int next = -1;
            for (int layer = layers_ - 1; layer >= 0; --layer)
            {
                if (history.layers[static_cast<std::size_t>(layer)].measured[static_cast<std::size_t>(mode)])
                {
                    next = layer;
                }
                next_measurement_[static_cast<std::size_t>(layer) * static_cast<std::size_t>(n_) +
                                  static_cast<std::size_t>(mode)] = next;
            }
        }
        gate_measured_.assign(graph_edges_.size(), 0u);
        for (std::size_t e = 0; e < graph_edges_.size(); ++e)
        {
            const GraphEdge &edge = graph_edges_[e];
            if (edge.kind != GraphEdgeKind::Unitary)
            {
                continue;
            }
            const std::size_t base = static_cast<std::size_t>(edge.layer) * static_cast<std::size_t>(n_);
            gate_measured_[e] = (next_measurement_[base + static_cast<std::size_t>(node_mode(edge.a))] >= 0 ||
                                 next_measurement_[base + static_cast<std::size_t>(node_mode(edge.b))] >= 0)
                                    ? 1u
                                    : 0u;
        }
    }

    // Walk states: node * 6 + direction * 2 + arrived_by_gate, direction 0 =
    // none yet, 1 = backward, 2 = forward. A walk never immediately retraces an
    // edge: a node has one gate edge and at most one temporal edge each way, so
    // that rule is all "non-backtracking" needs.
    static constexpr int WALK_STATES = 6;

    template <typename Visit>
    void walk_successors(std::uint32_t state, bool require_measured_peaks, Visit visit) const
    {
        const std::uint32_t node = state / WALK_STATES;
        const int direction = static_cast<int>((state % WALK_STATES) / 2u);
        const bool by_gate = (state % 2u) != 0u;
        for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
        {
            const std::uint32_t next = adjacency_[slot];
            const GraphEdge &edge = graph_edges_[arc_edge_[slot]];
            if (edge.kind == GraphEdgeKind::Unitary)
            {
                if (by_gate)
                {
                    continue;
                }
                visit(next * WALK_STATES + static_cast<std::uint32_t>(direction) * 2u + 1u, 0);
                continue;
            }
            const int step = node_layer(next) > node_layer(node) ? 2 : 1;
            if (!by_gate && direction != 0 && step != direction)
            {
                continue; // straight back along the edge just used
            }
            const bool turn = direction != 0 && step != direction;
            if (turn && step == 1 && require_measured_peaks)
            {
                // A peak: forward, the gate this node was entered by, backward.
                bool measured = false;
                for (std::uint32_t g = offset_[node]; g < offset_[node + 1u]; ++g)
                {
                    if (graph_edges_[arc_edge_[g]].kind == GraphEdgeKind::Unitary)
                    {
                        measured = gate_measured_[arc_edge_[g]] != 0u;
                    }
                }
                if (!measured)
                {
                    continue;
                }
            }
            visit(next * WALK_STATES + static_cast<std::uint32_t>(step) * 2u, turn ? 1 : 0);
        }
    }

    // Fewest turns to every node (min over its states), -1 when unreachable.
    void walk_turns_from(std::uint32_t source, std::vector<int> &turns) const
    {
        const std::size_t states = node_count() * WALK_STATES;
        std::vector<int> best(states, -1);
        std::deque<std::uint32_t> queue;
        const std::uint32_t start = source * WALK_STATES;
        best[start] = 0;
        queue.push_back(start);
        std::vector<std::uint8_t> done(states, 0u);
        while (!queue.empty())
        {
            const std::uint32_t state = queue.front();
            queue.pop_front();
            if (done[state] != 0u)
            {
                continue;
            }
            done[state] = 1u;
            walk_successors(state, false, [&](std::uint32_t next, int cost) {
                const int candidate = best[state] + cost;
                if (best[next] < 0 || candidate < best[next])
                {
                    best[next] = candidate;
                    if (cost == 0)
                    {
                        queue.push_front(next);
                    }
                    else
                    {
                        queue.push_back(next);
                    }
                }
            });
        }
        turns.assign(node_count(), -1);
        for (std::size_t node = 0; node < node_count(); ++node)
        {
            for (int k = 0; k < WALK_STATES; ++k)
            {
                const int value = best[node * WALK_STATES + static_cast<std::size_t>(k)];
                if (value >= 0 && (turns[node] < 0 || value < turns[node]))
                {
                    turns[node] = value;
                }
            }
        }
    }

    // Which nodes a walk with only measured peaks reaches.
    void walk_certified_from(std::uint32_t source, std::vector<std::uint8_t> &reached) const
    {
        const std::size_t states = node_count() * WALK_STATES;
        std::vector<std::uint8_t> seen(states, 0u);
        std::vector<std::uint32_t> stack{source * WALK_STATES};
        seen[source * WALK_STATES] = 1u;
        while (!stack.empty())
        {
            const std::uint32_t state = stack.back();
            stack.pop_back();
            walk_successors(state, true, [&](std::uint32_t next, int) {
                if (seen[next] == 0u)
                {
                    seen[next] = 1u;
                    stack.push_back(next);
                }
            });
        }
        reached.assign(node_count(), 0u);
        for (std::size_t node = 0; node < node_count(); ++node)
        {
            for (int k = 0; k < WALK_STATES; ++k)
            {
                reached[node] |= seen[node * WALK_STATES + static_cast<std::size_t>(k)];
            }
        }
    }

    // Close `cone` backward in time from its marked nodes: down temporal edges
    // (present only where the mode was unmeasured) and across in-layer gates.
    void close_past_cone(std::vector<std::uint8_t> &cone, std::vector<std::uint32_t> stack) const
    {
        while (!stack.empty())
        {
            const std::uint32_t node = stack.back();
            stack.pop_back();
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const std::uint32_t next = adjacency_[slot];
                const bool gate = graph_edges_[arc_edge_[slot]].kind == GraphEdgeKind::Unitary;
                if ((gate || node_layer(next) < node_layer(node)) && cone[next] == 0u)
                {
                    cone[next] = 1u;
                    stack.push_back(next);
                }
            }
        }
    }

    // R_M: every node from which some measurement event is causally reachable.
    // A measurement at layer t acts after that layer's gates, so its node
    // (t, m) is itself in the cone.
    void measurement_cone(std::vector<std::uint8_t> &cone) const
    {
        cone.assign(node_count(), 0u);
        std::vector<std::uint32_t> stack;
        for (int layer = 0; layer < layers_; ++layer)
        {
            for (int mode = 0; mode < n_; ++mode)
            {
                if (next_measurement(layer, mode) == layer)
                {
                    const std::uint32_t node = node_index(layer, mode);
                    cone[node] = 1u;
                    stack.push_back(node);
                }
            }
        }
        close_past_cone(cone, std::move(stack));
    }

    void endpoint_cone(int site, std::vector<std::uint8_t> &cone) const
    {
        cone.assign(node_count(), 0u);
        const std::uint32_t node = final_node(site);
        cone[node] = 1u;
        close_past_cone(cone, {node});
    }

    // Union-find over temporal edges plus the gates the doubled network keeps.
    void fill_doubled(int site_i, int site_j, const std::vector<std::uint8_t> &measurement,
                      const std::vector<std::uint8_t> &cone_i, const std::vector<std::uint8_t> &cone_j,
                      PairTurns &out) const
    {
        DisjointSets sets;
        sets.reset(node_count());
        for (const GraphEdge &edge : graph_edges_)
        {
            if (edge.kind == GraphEdgeKind::Temporal || measurement[edge.a] != 0u ||
                cone_i[edge.a] != 0u || cone_j[edge.a] != 0u)
            {
                sets.unite(edge.a, edge.b);
            }
        }
        out.doubled_network = sets.find(final_node(site_i)) == sets.find(final_node(site_j));
        const bool survive = survives_[static_cast<std::size_t>(site_i)] != 0u &&
                             survives_[static_cast<std::size_t>(site_j)] != 0u;
        out.connected_doubled = survive && out.doubled_network;
    }

    void count_cancelled(const GraphPath &path, const std::vector<std::uint8_t> &measurement,
                         const std::vector<std::uint8_t> &cone_i, const std::vector<std::uint8_t> &cone_j,
                         PathShape &shape) const
    {
        if (!shape.available)
        {
            return;
        }
        shape.cancelled_gates = 0;
        for (std::uint32_t e : path.edges)
        {
            const GraphEdge &edge = graph_edges_[e];
            if (edge.kind == GraphEdgeKind::Unitary && measurement[edge.a] == 0u &&
                cone_i[edge.a] == 0u && cone_j[edge.a] == 0u)
            {
                ++shape.cancelled_gates;
            }
        }
    }

    void fill_pair_existence(int site_i, int site_j, const std::vector<int> &turns,
                             const std::vector<std::uint8_t> &certified, PairTurns &out) const
    {
        out.evaluated = true;
        const std::uint32_t sink = final_node(site_j);
        out.min_turns = turns[sink];
        out.common_source = out.min_turns >= 0 && out.min_turns <= 1;
        out.certified = certified[sink] != 0u;
        const bool survive = survives_[static_cast<std::size_t>(site_i)] != 0u &&
                             survives_[static_cast<std::size_t>(site_j)] != 0u;
        out.connected_common_source = survive && out.common_source;
        out.connected_certified = survive && out.certified;
    }

    // Every ordered final-slice pair: two walk searches per source, and the two
    // canonical paths of every target from per-source structures.
    void fill_turns()
    {
        pair_turns_.assign(static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_), PairTurns{});
        std::vector<int> turns;
        std::vector<std::uint8_t> certified;
        std::vector<std::int32_t> distance;
        std::vector<std::uint32_t> order;
        std::vector<double> best;
        std::vector<double> widest;
        std::vector<std::uint8_t> measurement;
        measurement_cone(measurement);
        std::vector<std::vector<std::uint8_t>> cones(static_cast<std::size_t>(n_));
        for (int site = 0; site < n_; ++site)
        {
            endpoint_cone(site, cones[static_cast<std::size_t>(site)]);
        }
        for (int i = 0; i < n_; ++i)
        {
            const std::uint32_t source = final_node(i);
            walk_turns_from(source, turns);
            walk_certified_from(source, certified);
            if (gate_metrics_)
            {
                bfs(source, distance, order);
                shortest_bottleneck(distance, order, source, best);
                widest_from(source, GateCapacity::EntanglingPower, widest);
            }
            for (int j = 0; j < n_; ++j)
            {
                if (j == i)
                {
                    continue;
                }
                PairTurns &out = pair_turns_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_) +
                                             static_cast<std::size_t>(j)];
                fill_pair_existence(i, j, turns, certified, out);
                const auto &cone_i = cones[static_cast<std::size_t>(i)];
                const auto &cone_j = cones[static_cast<std::size_t>(j)];
                if (j < i)
                {
                    // Symmetric in (i, j): reuse the mirrored pair's verdict.
                    const PairTurns &mirror =
                        pair_turns_[static_cast<std::size_t>(j) * static_cast<std::size_t>(n_) +
                                    static_cast<std::size_t>(i)];
                    out.doubled_network = mirror.doubled_network;
                    out.connected_doubled = mirror.connected_doubled;
                }
                else
                {
                    fill_doubled(i, j, measurement, cone_i, cone_j, out);
                }
                if (gate_metrics_ &&
                    root_[static_cast<std::size_t>(i)] == root_[static_cast<std::size_t>(j)])
                {
                    const std::uint32_t sink = final_node(j);
                    const GraphPath shortest = shortest_maximin_from(source, sink, distance, order, best);
                    const GraphPath wide =
                        widest_from_trace(source, sink, GateCapacity::EntanglingPower, widest);
                    out.shortest_maximin = path_shape(shortest, i, j);
                    out.widest = path_shape(wide, i, j);
                    count_cancelled(shortest, measurement, cone_i, cone_j, out.shortest_maximin);
                    count_cancelled(wide, measurement, cone_i, cone_j, out.widest);
                }
            }
        }
    }

    GatePathMetrics compute_gate_path(int site_i, int site_j) const
    {
        GatePathMetrics out;
        const std::uint32_t source = final_node(site_i);
        const std::uint32_t sink = final_node(site_j);
        std::vector<std::int32_t> distance;
        std::vector<std::uint32_t> order;
        bfs(source, distance, order);
        std::vector<double> best;
        shortest_bottleneck(distance, order, source, best);
        fill_from_source(source, sink, distance, order, best, out);
        std::vector<double> widest;
        widest_from(source, GateCapacity::EntanglingPower, widest);
        out.widest_entangling_power = widest[sink];
        widest_from(source, GateCapacity::EvenBlockMixing, widest);
        out.widest_even_block_mixing = widest[sink];
        widest_from(source, GateCapacity::OddBlockMixing, widest);
        out.widest_odd_block_mixing = widest[sink];
        return out;
    }

    void fill_from_source(std::uint32_t source, std::uint32_t sink,
                          const std::vector<std::int32_t> &distance,
                          const std::vector<std::uint32_t> &order, const std::vector<double> &best,
                          GatePathMetrics &out) const
    {
        out.available = true;
        out.shortest_path_length = distance[sink];
        out.shortest_maximin_entangling_power = best[sink];
        std::vector<int> count;
        fewest_gates(distance, order, source, best[sink], count, nullptr);
        out.shortest_path_unitary_count = count[sink];
    }

    // Every final-slice pair at once: per source, one BFS, one shortest-path
    // DP, three widest sweeps, and one fewest-gates walk per target.
    void fill_gate_paths()
    {
        gate_paths_.assign(static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_),
                           GatePathMetrics{});
        std::vector<std::int32_t> distance;
        std::vector<std::uint32_t> order;
        std::vector<double> best;
        std::vector<double> widest_ep;
        std::vector<double> widest_even;
        std::vector<double> widest_odd;
        for (int i = 0; i < n_; ++i)
        {
            const std::uint32_t source = final_node(i);
            bfs(source, distance, order);
            shortest_bottleneck(distance, order, source, best);
            widest_from(source, GateCapacity::EntanglingPower, widest_ep);
            widest_from(source, GateCapacity::EvenBlockMixing, widest_even);
            widest_from(source, GateCapacity::OddBlockMixing, widest_odd);
            for (int j = 0; j < n_; ++j)
            {
                if (j == i || root_[static_cast<std::size_t>(i)] != root_[static_cast<std::size_t>(j)])
                {
                    continue;
                }
                const std::uint32_t sink = final_node(j);
                GatePathMetrics &out =
                    gate_paths_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_) +
                                static_cast<std::size_t>(j)];
                fill_from_source(source, sink, distance, order, best, out);
                out.widest_entangling_power = widest_ep[sink];
                out.widest_even_block_mixing = widest_even[sink];
                out.widest_odd_block_mixing = widest_odd[sink];
            }
        }
    }

    // Maximum-product path (via logs), widest path, and how many edge-disjoint
    // paths survive at the widest path's bottleneck.
    void strongest_path(const std::vector<double> &weight, std::uint32_t source,
                        std::uint32_t sink, double &log_weight, double &bottleneck,
                        int &multiplicity) const
    {
        const std::size_t nodes = static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
        constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

        // Max-product: Dijkstra on the additive log weight, maximizing.
        std::vector<double> best(nodes, NEG_INF);
        std::vector<std::uint8_t> done(nodes, 0u);
        best[source] = 0.0;
        for (std::size_t step = 0; step < nodes; ++step)
        {
            std::uint32_t node = UNSET;
            double value = NEG_INF;
            for (std::size_t v = 0; v < nodes; ++v)
            {
                if (done[v] == 0u && best[v] > value)
                {
                    value = best[v];
                    node = static_cast<std::uint32_t>(v);
                }
            }
            if (node == UNSET)
            {
                break;
            }
            done[node] = 1u;
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const double w = weight[slot];
                const double next = w > 0.0 ? best[node] + std::log(w) : NEG_INF;
                if (next > best[adjacency_[slot]])
                {
                    best[adjacency_[slot]] = next;
                }
            }
        }
        log_weight = best[sink];

        // Widest path: the same sweep maximizing the minimum edge instead.
        std::vector<double> widest(nodes, 0.0);
        std::fill(done.begin(), done.end(), 0u);
        widest[source] = std::numeric_limits<double>::infinity();
        for (std::size_t step = 0; step < nodes; ++step)
        {
            std::uint32_t node = UNSET;
            double value = 0.0;
            for (std::size_t v = 0; v < nodes; ++v)
            {
                if (done[v] == 0u && widest[v] > value)
                {
                    value = widest[v];
                    node = static_cast<std::uint32_t>(v);
                }
            }
            if (node == UNSET)
            {
                break;
            }
            done[node] = 1u;
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const double next = std::min(widest[node], weight[slot]);
                if (next > widest[adjacency_[slot]])
                {
                    widest[adjacency_[slot]] = next;
                }
            }
        }
        bottleneck = widest[sink];

        multiplicity = 0;
        if (!(bottleneck > 0.0))
        {
            return;
        }
        // How many independent channels are at least this wide. Edges below the
        // bottleneck are deleted, so a braid of equally good paths counts as
        // many and a single thread counts as one.
        UnitFlowNetwork flow(nodes);
        for (std::size_t u = 0; u < nodes; ++u)
        {
            for (std::uint32_t slot = offset_[u]; slot < offset_[u + 1u]; ++slot)
            {
                const std::uint32_t v = adjacency_[slot];
                if (u < v && weight[slot] >= bottleneck)
                {
                    flow.add(static_cast<int>(u), static_cast<int>(v), 1, 1);
                }
            }
        }
        multiplicity = flow.max_flow(static_cast<int>(source), static_cast<int>(sink), 64);
    }

    std::uint32_t node_index(int layer, int mode) const
    {
        return static_cast<std::uint32_t>(layer) * static_cast<std::uint32_t>(n_) +
               static_cast<std::uint32_t>(mode);
    }

    // The single definition of the edge set, walked twice: once to size the
    // CSR arrays and once to fill them. Keeping it in one place is what stops
    // the two passes from ever disagreeing about how many edges there are.
    template <typename OnEdge>
    void for_each_edge(const LogicalHistory &history, OnEdge on_edge)
    {
        for (int layer = 0; layer < layers_; ++layer)
        {
            const LogicalLayer &current = history.layers[static_cast<std::size_t>(layer)];
            for (const LogicalBond &bond : current.bonds)
            {
                if (bond.a == bond.b)
                {
                    continue;
                }
                GraphEdge meta;
                meta.a = node_index(layer, bond.a);
                meta.b = node_index(layer, bond.b);
                meta.kind = GraphEdgeKind::Unitary;
                meta.layer = layer;
                meta.source_gate_index = bond.source_gate_index;
                meta.entangling_power = bond.entangling_power;
                meta.even_block_mixing = bond.even_block_mixing;
                meta.odd_block_mixing = bond.odd_block_mixing;
                on_edge(meta.a, meta.b, meta, bond.pair_weight, bond.hop_weight);
            }
            if (layer + 1 >= layers_)
            {
                continue;
            }
            for (int mode = 0; mode < n_; ++mode)
            {
                if (!current.measured[static_cast<std::size_t>(mode)])
                {
                    // An idle mode transmits perfectly in both parity sectors.
                    GraphEdge meta;
                    meta.a = node_index(layer, mode);
                    meta.b = node_index(layer + 1, mode);
                    meta.kind = GraphEdgeKind::Temporal;
                    meta.layer = layer;
                    on_edge(meta.a, meta.b, meta, 1.0, 1.0);
                }
            }
        }
    }

    static std::int32_t layers_since_measurement(const LogicalHistory &history, int mode)
    {
        const int layers = static_cast<int>(history.layers.size());
        for (int back = 0; back < layers; ++back)
        {
            const LogicalLayer &layer = history.layers[static_cast<std::size_t>(layers - 1 - back)];
            if (layer.measured[static_cast<std::size_t>(mode)])
            {
                return back;
            }
        }
        return layers;
    }

    void fill_shortest_paths()
    {
        const std::size_t nodes = static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
        paths_.assign(static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_), -1);
        std::vector<std::int32_t> distance(nodes, -1);
        std::vector<std::uint32_t> queue;
        queue.reserve(nodes);
        for (int site = 0; site < n_; ++site)
        {
            std::fill(distance.begin(), distance.end(), -1);
            queue.clear();
            const std::uint32_t source =
                node_index(layers_ - 1, mode_of_site_[static_cast<std::size_t>(site)]);
            distance[source] = 0;
            queue.push_back(source);
            for (std::size_t head = 0; head < queue.size(); ++head)
            {
                const std::uint32_t node = queue[head];
                for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
                {
                    const std::uint32_t next = adjacency_[slot];
                    if (distance[next] < 0)
                    {
                        distance[next] = distance[node] + 1;
                        queue.push_back(next);
                    }
                }
            }
            for (int other = 0; other < n_; ++other)
            {
                paths_[static_cast<std::size_t>(site) * static_cast<std::size_t>(n_) +
                       static_cast<std::size_t>(other)] =
                    distance[node_index(layers_ - 1,
                                        mode_of_site_[static_cast<std::size_t>(other)])];
            }
        }
    }

    // Every vertex reachable from `start`, the endpoint's component.
    void collect_component(std::uint32_t start, std::vector<std::uint32_t> &out) const
    {
        const std::size_t nodes = static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
        std::vector<std::uint8_t> seen(nodes, 0u);
        out.assign(1, start);
        seen[start] = 1u;
        for (std::size_t head = 0; head < out.size(); ++head)
        {
            const std::uint32_t node = out[head];
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const std::uint32_t next = adjacency_[slot];
                if (seen[next] == 0u)
                {
                    seen[next] = 1u;
                    out.push_back(next);
                }
            }
        }
    }

    // Walk one BFS shortest path back from sink to source, counting edges by
    // kind. `parent_slot_` records which adjacency slot reached each node, and
    // the slot's kind is the edge's kind.
    void split_shortest_path(std::uint32_t source, std::uint32_t sink, std::int32_t &gates,
                             std::int32_t &temporal) const
    {
        gates = -1;
        temporal = -1;
        const std::size_t nodes = static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
        std::vector<std::uint32_t> parent(nodes, UNSET);
        std::vector<std::uint32_t> parent_slot(nodes, UNSET);
        std::vector<std::uint32_t> queue;
        queue.reserve(nodes);
        queue.push_back(source);
        parent[source] = source;
        for (std::size_t head = 0; head < queue.size(); ++head)
        {
            const std::uint32_t node = queue[head];
            if (node == sink)
            {
                break;
            }
            for (std::uint32_t slot = offset_[node]; slot < offset_[node + 1u]; ++slot)
            {
                const std::uint32_t next = adjacency_[slot];
                if (parent[next] == UNSET)
                {
                    parent[next] = node;
                    parent_slot[next] = slot;
                    queue.push_back(next);
                }
            }
        }
        if (parent[sink] == UNSET)
        {
            return;
        }
        gates = 0;
        temporal = 0;
        for (std::uint32_t node = sink; node != source;)
        {
            const std::uint32_t slot = parent_slot[node];
            (edge_kind_[slot] != 0u ? gates : temporal) += 1;
            node = parent[node];
        }
    }

    // Tarjan's biconnected components, iteratively -- a spacetime graph is
    // T*N deep and a recursive DFS would be a stack overflow waiting for a
    // large N. Fills `block_of_edge_` (per adjacency slot) and the block-cut
    // tree in `bct_*`.
    void build_block_cut_tree()
    {
        const std::size_t nodes = static_cast<std::size_t>(layers_) * static_cast<std::size_t>(n_);
        block_of_node_.assign(nodes, UNSET);
        blocks_ = 0;
        consumed_parent_.assign(nodes, 0u);
        bct_of_node_.assign(nodes, UNSET);
        is_articulation_.assign(nodes, 0u);
        std::vector<std::uint32_t> discovery(nodes, UNSET);
        std::vector<std::uint32_t> low(nodes, 0u);
        std::vector<std::uint32_t> parent(nodes, UNSET);
        std::vector<std::uint32_t> next_slot(nodes, 0u);
        std::vector<std::uint32_t> edge_stack; // adjacency slots
        std::vector<std::uint32_t> stack;
        std::vector<std::vector<std::uint32_t>> block_vertices;
        std::uint32_t timer = 0;

        auto close_block = [&](std::uint32_t until_slot) {
            std::vector<std::uint32_t> members;
            while (!edge_stack.empty())
            {
                const std::uint32_t slot = edge_stack.back();
                edge_stack.pop_back();
                members.push_back(edge_from_[slot]);
                members.push_back(adjacency_[slot]);
                if (slot == until_slot)
                {
                    break;
                }
            }
            std::sort(members.begin(), members.end());
            members.erase(std::unique(members.begin(), members.end()), members.end());
            block_vertices.push_back(std::move(members));
            ++blocks_;
        };

        for (std::uint32_t start = 0; start < nodes; ++start)
        {
            if (discovery[start] != UNSET)
            {
                continue;
            }
            stack.assign(1, start);
            discovery[start] = low[start] = timer++;
            next_slot[start] = offset_[start];
            std::uint32_t root_children = 0;
            while (!stack.empty())
            {
                const std::uint32_t node = stack.back();
                if (next_slot[node] < offset_[node + 1u])
                {
                    const std::uint32_t slot = next_slot[node]++;
                    const std::uint32_t next = adjacency_[slot];
                    if (next == parent[node] && consumed_parent_[node] == 0u)
                    {
                        // Skip the edge back to the parent exactly once, so a
                        // genuine multi-edge would still close a cycle. This
                        // graph has none, but the guard is what makes that an
                        // assumption rather than a silent requirement.
                        consumed_parent_[node] = 1u;
                        continue;
                    }
                    if (discovery[next] == UNSET)
                    {
                        edge_stack.push_back(slot);
                        parent[next] = node;
                        discovery[next] = low[next] = timer++;
                        next_slot[next] = offset_[next];
                        consumed_parent_[next] = 0u;
                        stack.push_back(next);
                        root_children += node == start ? 1u : 0u;
                    }
                    else if (discovery[next] < discovery[node])
                    {
                        edge_stack.push_back(slot);
                        low[node] = std::min(low[node], discovery[next]);
                    }
                    continue;
                }
                stack.pop_back();
                if (stack.empty())
                {
                    continue;
                }
                const std::uint32_t up = stack.back();
                low[up] = std::min(low[up], low[node]);
                if (low[node] >= discovery[up])
                {
                    // `up` separates `node`'s subtree from the rest. The DFS
                    // root is the exception and is settled after the sweep,
                    // when its child count is final.
                    if (up != start)
                    {
                        is_articulation_[up] = 1u;
                    }
                    close_block(slot_between(up, node));
                }
            }
            // A DFS root is a cut vertex exactly when it has more than one
            // child in the DFS tree.
            if (root_children > 1u)
            {
                is_articulation_[start] = 1u;
            }
            if (!edge_stack.empty())
            {
                close_block(edge_stack.front());
            }
        }

        // The block-cut tree: one node per block, one per articulation vertex.
        bct_adjacency_.assign(blocks_ + articulation_count(), {});
        bct_block_vertices_ = std::move(block_vertices);
        std::uint32_t next_cut_node = blocks_;
        cut_node_of_.assign(nodes, UNSET);
        cut_node_vertex_.clear();
        for (std::uint32_t v = 0; v < nodes; ++v)
        {
            if (is_articulation_[v] != 0u)
            {
                cut_node_of_[v] = next_cut_node++;
                cut_node_vertex_.push_back(v);
            }
        }
        for (std::uint32_t b = 0; b < blocks_; ++b)
        {
            for (std::uint32_t v : bct_block_vertices_[b])
            {
                if (is_articulation_[v] != 0u)
                {
                    bct_adjacency_[b].push_back(cut_node_of_[v]);
                    bct_adjacency_[cut_node_of_[v]].push_back(b);
                }
                else
                {
                    block_of_node_[v] = b;
                }
            }
        }
        for (std::uint32_t v = 0; v < nodes; ++v)
        {
            bct_of_node_[v] = is_articulation_[v] != 0u ? cut_node_of_[v] : block_of_node_[v];
        }
    }

    std::uint32_t articulation_count() const
    {
        std::uint32_t count = 0;
        for (std::uint8_t flag : is_articulation_)
        {
            count += flag != 0u ? 1u : 0u;
        }
        return count;
    }

    // The adjacency slot carrying the tree edge up -> node.
    std::uint32_t slot_between(std::uint32_t up, std::uint32_t node) const
    {
        for (std::uint32_t slot = offset_[up]; slot < offset_[up + 1u]; ++slot)
        {
            if (adjacency_[slot] == node)
            {
                return slot;
            }
        }
        return UNSET;
    }

    // Mark every vertex on some source-sink path, and count the cut vertices
    // strictly between them.
    void mark_backbone(std::uint32_t source, std::uint32_t sink,
                       std::vector<std::uint8_t> &backbone,
                       std::uint32_t &articulations) const
    {
        articulations = 0;
        if (bct_of_node_.empty() || bct_of_node_[source] == UNSET ||
            bct_of_node_[sink] == UNSET)
        {
            // No decomposition (an isolated vertex, say): the endpoints are
            // their own backbone and nothing is claimed about branches.
            backbone[source] = 1u;
            backbone[sink] = 1u;
            return;
        }
        const std::uint32_t from = bct_of_node_[source];
        const std::uint32_t to = bct_of_node_[sink];
        const std::size_t tree_nodes = bct_adjacency_.size();
        std::vector<std::uint32_t> parent(tree_nodes, UNSET);
        std::vector<std::uint32_t> queue{from};
        parent[from] = from;
        for (std::size_t head = 0; head < queue.size() && parent[to] == UNSET; ++head)
        {
            for (std::uint32_t next : bct_adjacency_[queue[head]])
            {
                if (parent[next] == UNSET)
                {
                    parent[next] = queue[head];
                    queue.push_back(next);
                }
            }
        }
        if (parent[to] == UNSET)
        {
            backbone[source] = 1u;
            backbone[sink] = 1u;
            return;
        }
        for (std::uint32_t node = to;; node = parent[node])
        {
            if (node < blocks_)
            {
                for (std::uint32_t v : bct_block_vertices_[node])
                {
                    backbone[v] = 1u;
                }
            }
            else
            {
                // A cut node on the path is a vertex whose removal separates
                // the endpoints -- unless it is an endpoint itself.
                const std::uint32_t vertex = vertex_of_cut_node(node);
                backbone[vertex] = 1u;
                if (vertex != source && vertex != sink)
                {
                    ++articulations;
                }
            }
            if (node == from)
            {
                break;
            }
        }
        backbone[source] = 1u;
        backbone[sink] = 1u;
    }

    std::uint32_t vertex_of_cut_node(std::uint32_t tree_node) const
    {
        const std::size_t index = static_cast<std::size_t>(tree_node - blocks_);
        return index < cut_node_vertex_.size() ? cut_node_vertex_[index] : UNSET;
    }

    static constexpr std::uint32_t UNSET = ~std::uint32_t{0};

    bool ready_ = false;
    int n_ = 0;
    int layers_ = 0;
    std::uint64_t edges_ = 0;
    DisjointSets sets_;
    std::vector<std::uint32_t> degree_;
    std::vector<std::uint32_t> offset_;
    std::vector<std::uint32_t> cursor_;
    std::vector<std::uint32_t> adjacency_;
    // Parallel to `adjacency_`: 1 for a unitary bond, 0 for a temporal edge,
    // and which node the slot belongs to. The kind is what lets a shortest path
    // be reported as gates and waiting separately -- a path of five gates and
    // one of five idle layers are very different objects with the same length.
    std::vector<std::uint8_t> edge_kind_;
    std::vector<std::uint32_t> edge_from_;
    std::vector<double> weight_pair_;
    std::vector<double> weight_hop_;
    // Parallel to `adjacency_`: the undirected edge each slot travels, and the
    // edge table it indexes (GraphArc = {adjacency_[slot], arc_edge_[slot]}).
    std::vector<std::uint32_t> arc_edge_;
    std::vector<GraphEdge> graph_edges_;
    bool gate_metrics_ = false;
    bool single_gate_nodes_ = true;
    // n*n, filled only when build() was asked for gate paths.
    std::vector<GatePathMetrics> gate_paths_;
    // layers*n: first measurement of each mode at or after each layer.
    std::vector<int> next_measurement_;
    // Per graph edge: a gate with a connector measured at or after its layer.
    std::vector<std::uint8_t> gate_measured_;
    // n*n, filled only when build() was asked for turns.
    std::vector<PairTurns> pair_turns_;
    std::vector<int> mode_of_site_;
    std::vector<std::uint32_t> root_;
    std::vector<std::uint32_t> component_size_;
    std::vector<std::uint8_t> survives_;
    std::vector<std::int32_t> idle_;
    std::vector<std::int32_t> paths_;
    // The biconnected decomposition, built once per trajectory.
    std::uint32_t blocks_ = 0;
    std::vector<std::uint32_t> block_of_node_;
    std::vector<std::uint32_t> bct_of_node_;
    std::vector<std::uint32_t> cut_node_of_;
    std::vector<std::uint32_t> cut_node_vertex_;
    std::vector<std::uint8_t> is_articulation_;
    std::vector<std::uint8_t> consumed_parent_;
    std::vector<std::vector<std::uint32_t>> bct_adjacency_;
    std::vector<std::vector<std::uint32_t>> bct_block_vertices_;
};

} // namespace mipt::dist
