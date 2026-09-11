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
#include <deque>
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
    void build(const LogicalHistory &history, bool with_paths)
    {
        ready_ = false;
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
        auto count_edge = [&](std::uint32_t a, std::uint32_t b) {
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
        cursor_ = offset_;
        auto add_edge = [&](std::uint32_t a, std::uint32_t b) {
            adjacency_[cursor_[a]++] = b;
            adjacency_[cursor_[b]++] = a;
            sets_.unite(a, b);
        };
        for_each_edge(history, add_edge);

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
        ready_ = true;
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

  private:
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
                on_edge(node_index(layer, bond.a), node_index(layer, bond.b));
            }
            if (layer + 1 >= layers_)
            {
                continue;
            }
            for (int mode = 0; mode < n_; ++mode)
            {
                if (!current.measured[static_cast<std::size_t>(mode)])
                {
                    on_edge(node_index(layer, mode), node_index(layer + 1, mode));
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

    bool ready_ = false;
    int n_ = 0;
    int layers_ = 0;
    std::uint64_t edges_ = 0;
    DisjointSets sets_;
    std::vector<std::uint32_t> degree_;
    std::vector<std::uint32_t> offset_;
    std::vector<std::uint32_t> cursor_;
    std::vector<std::uint32_t> adjacency_;
    std::vector<int> mode_of_site_;
    std::vector<std::uint32_t> root_;
    std::vector<std::uint32_t> component_size_;
    std::vector<std::uint8_t> survives_;
    std::vector<std::int32_t> idle_;
    std::vector<std::int32_t> paths_;
};

} // namespace mipt::dist
