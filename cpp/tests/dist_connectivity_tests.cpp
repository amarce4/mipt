// Host tests for the spacetime percolation graph and the logical layer
// description it is built from.
//
// Two things are pinned here, and they fail in different ways.
//
//   * The *graph* rules. C = A_i && A_j && interior-path is an exact
//     factorization rather than an approximation, and the whole point of
//     reporting the three flags separately is that P(A_i & A_j) can then be
//     measured instead of assumed to be q^2. These tests use hand-built
//     histories where the answer is obvious by inspection.
//
//   * The *projection* from a simulated RPPU layer onto a logical one. This is
//     where the modelling decisions live: an FSWAP transports a mode rather
//     than entangling anything, and a Jordan-Wigner boundary gate is the single
//     bond (0, N-1) rather than a fan across every site its string crosses.
//     Get either wrong and the graph does not merely shift -- it fuses the ring
//     into one cluster and every pair reports "connected", which looks like a
//     result rather than a bug.

#include "mipt/circuits/rppu_layer.hpp"
#include "mipt/dist_connectivity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <string>
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

using mipt::LogicalHistory;
using mipt::LogicalLayer;
using mipt::dist::ConnectivityIndex;
using mipt::dist::PairConnectivity;

// A history from a compact description: one entry per layer, giving its bonds
// and the sites it measures.
LogicalHistory make_history(int n,
                            const std::vector<std::pair<std::vector<std::pair<int, int>>,
                                                        std::vector<int>>> &layers)
{
    LogicalHistory history;
    history.reset(n);
    for (const auto &[bonds, measured] : layers)
    {
        LogicalLayer layer;
        layer.measured.assign(static_cast<std::size_t>(n), 0u);
        for (const auto &[a, b] : bonds)
        {
            layer.bonds.push_back({a, b});
        }
        for (int site : measured)
        {
            layer.measured[static_cast<std::size_t>(site)] = 1u;
        }
        history.layers.push_back(std::move(layer));
    }
    history.valid = true;
    return history;
}

// A four-site chain with no measurements anywhere: every worldline survives and
// the bonds tie the whole ring together, so every pair is connected.
void test_unmeasured_circuit_connects_everything()
{
    const LogicalHistory history = make_history(4, {
        {{{0, 1}, {2, 3}}, {}},
        {{{1, 2}, {3, 0}}, {}},
    });
    ConnectivityIndex index;
    index.build(history, true);
    expect(index.ready(), "an index over a non-empty history is ready");
    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
        {
            const PairConnectivity conn = index.query(i, j);
            expect(conn.evaluated, "every pair of a built graph is evaluated");
            expect(conn.survives_i && conn.survives_j,
                   "no measurement in the last layer means both endpoints survive");
            expect(conn.interior_path, "an unmeasured brickwork connects every pair");
            expect(conn.connected, "and so the percolation event occurs");
            expect(conn.component_size == 8,
                   "the component is the whole spacetime: 2 layers x 4 sites");
        }
    }
}

// Measuring everything in the final layer kills every endpoint, so C=0 even
// though the interior is still fully connected. This is exactly the case where
// reporting only P(C) would hide which factor vanished.
void test_final_layer_measurement_kills_survival()
{
    const LogicalHistory history = make_history(4, {
        {{{0, 1}, {2, 3}}, {}},
        {{{1, 2}, {3, 0}}, {0, 1, 2, 3}},
    });
    ConnectivityIndex index;
    index.build(history, true);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
        {
            const PairConnectivity conn = index.query(i, j);
            expect(!conn.survives_i && !conn.survives_j, "a measured endpoint does not survive");
            expect(conn.interior_path, "the interior is still connected");
            expect(!conn.connected, "but C fails on endpoint survival alone");
        }
    }
}

// Ordering inside a layer: gates, then measurements. A bond applied in the
// last layer is therefore visible to both its endpoints even though the same
// layer measures one of them -- the measurement severs the edge into the
// *next* slice, and for the last layer that edge is exactly endpoint survival.
//
// This is the distinction that keeps C's three factors independent: the pair
// has a path and lost an endpoint, which is a different failure from having no
// path at all.
void test_measurement_cuts_forward_only()
{
    const LogicalHistory bonded_then_measured = make_history(2, {
        {{{0, 1}}, {0}},
    });
    ConnectivityIndex index;
    index.build(bonded_then_measured, true);
    const PairConnectivity conn = index.query(0, 1);
    expect(conn.interior_path,
           "the gate is visible at its own slice even though a measurement follows it");
    expect(!conn.survives_i && conn.survives_j, "the measured endpoint is the one that dies");
    expect(!conn.connected, "so C fails on survival while the path stands");

    // The same bond with no measurement percolates, which is what makes the
    // case above attributable to the measurement rather than to the geometry.
    const LogicalHistory bonded_only = make_history(2, {
        {{{0, 1}}, {}},
    });
    ConnectivityIndex clean;
    clean.build(bonded_only, true);
    expect(clean.query(0, 1).connected, "the same bond without the measurement percolates");

    // And a measurement in an *earlier* layer does cut the worldline forward:
    // here the bond is in layer 0 and site 1 is measured there, so by the final
    // slice the two are in different components.
    const LogicalHistory cut_forward = make_history(2, {
        {{{0, 1}}, {1}},
        {{}, {}},
    });
    ConnectivityIndex severed;
    severed.build(cut_forward, true);
    const PairConnectivity later = severed.query(0, 1);
    expect(!later.interior_path,
           "a measurement removes the edge into the next slice, so the path does not reach it");
    expect(later.survives_i && later.survives_j,
           "both endpoints survive the final layer, which measures nothing");
    expect(!later.connected, "and C now fails on the path instead of on survival");
}

// C factorizes exactly. Building a graph where the interior path exists but one
// endpoint dies, and one where both survive but no path exists, separates the
// two failure modes -- which is the entire reason the flags are stored apart.
void test_connectivity_factorizes()
{
    const LogicalHistory one_endpoint_dies = make_history(2, {
        {{{0, 1}}, {}},
        {{}, {1}},
    });
    ConnectivityIndex a;
    a.build(one_endpoint_dies, true);
    const PairConnectivity dies = a.query(0, 1);
    expect(dies.interior_path && dies.survives_i && !dies.survives_j && !dies.connected,
           "a path with a dead endpoint is interior_path=1, connected=0");

    const LogicalHistory no_path = make_history(2, {
        {{}, {}},
        {{}, {}},
    });
    ConnectivityIndex b;
    b.build(no_path, true);
    const PairConnectivity isolated = b.query(0, 1);
    expect(isolated.survives_i && isolated.survives_j && !isolated.interior_path &&
               !isolated.connected,
           "two surviving but never-coupled endpoints are survivors with no path");
    expect(isolated.component_size == 2,
           "an isolated worldline's component is its own two nodes");
    expect(isolated.shortest_path < 0, "and there is no path length to report");
}

// Shortest path over a graph whose answer is countable by hand.
void test_shortest_path_lengths()
{
    // Three layers, an open chain of four sites, nothing measured.
    //   layer 0: (0,1) (2,3)
    //   layer 1: (1,2)
    //   layer 2: (0,1) (2,3)
    // From (2,0) to (2,3): the two are joined directly in layer 2 -> 1 hop.
    // From (2,0) to (2,2): 0 -> (1,0) -> (1,1) -> (1,2) -> (2,2) is 4 hops, and
    // going forward through layer 2 is no shorter.
    const LogicalHistory history = make_history(4, {
        {{{0, 1}, {2, 3}}, {}},
        {{{1, 2}}, {}},
        {{{0, 1}, {2, 3}}, {}},
    });
    ConnectivityIndex index;
    index.build(history, true);
    expect(index.has_paths(), "path lengths were requested and computed");
    expect(index.query(0, 1).shortest_path == 1, "sites joined in the final layer are one hop");
    expect(index.query(2, 3).shortest_path == 1, "and so are the other final-layer partners");
    expect(index.query(0, 2).shortest_path == 4,
           "crossing the middle bond costs a step down, the bond, and a step back");
    expect(index.query(0, 0).shortest_path == 0, "a site is zero hops from itself");

    ConnectivityIndex without;
    without.build(history, false);
    expect(!without.has_paths(), "path lengths are skipped when not requested");
    expect(without.query(0, 2).shortest_path < 0, "and reported as absent rather than as zero");
    expect(without.query(0, 2).connected, "while the flags are unaffected");
}

// Menger counts. A 4-cycle joins opposite corners two independent ways; a chain
// joins its ends one way, through articulation nodes; separate components none.
void test_disjoint_paths()
{
    const LogicalHistory cycle = make_history(4, {{{{0, 1}, {1, 2}, {2, 3}, {3, 0}}, {}}});
    ConnectivityIndex ring;
    ring.build(cycle, true);
    const mipt::dist::DisjointPaths both = ring.disjoint_paths(0, 2);
    expect(both.edge == 2 && both.vertex == 2,
           "opposite corners of a cycle have two edge- and vertex-disjoint paths");

    const LogicalHistory chain = make_history(4, {{{{0, 1}, {1, 2}, {2, 3}}, {}}});
    ConnectivityIndex line;
    line.build(chain, true);
    const mipt::dist::DisjointPaths single = line.disjoint_paths(0, 3);
    expect(single.edge == 1 && single.vertex == 1,
           "the ends of a chain are joined through a single bottleneck");

    const LogicalHistory split = make_history(4, {{{{0, 1}, {2, 3}}, {}}});
    ConnectivityIndex apart;
    apart.build(split, true);
    const mipt::dist::DisjointPaths none = apart.disjoint_paths(0, 3);
    expect(none.edge == 0 && none.vertex == 0, "separate components share no path");

    // Through time: two layers where 0 and 2 meet via site 1 in layer 0 and
    // via site 3 in layer 1 -- two routes that share only the endpoints'
    // worldlines, which the vertex count may reuse only at the endpoints.
    const LogicalHistory two_routes = make_history(4, {
        {{{0, 1}, {1, 2}}, {}},
        {{{2, 3}, {3, 0}}, {}},
    });
    ConnectivityIndex routes;
    routes.build(two_routes, true);
    const mipt::dist::DisjointPaths through_time = routes.disjoint_paths(0, 2);
    expect(through_time.edge == 2, "one route in each layer gives two edge-disjoint paths");
    expect(through_time.vertex == 2, "and they share no interior node");
}

// Request 11.6: single-path, braided-path, articulation-point and
// dangling-branch graphs, against the backbone decomposition.
//
// Each shape is chosen so the answer is countable by hand, which is the only
// way to tell a correct block-cut tree from a plausible one.
void test_component_anatomy()
{
    using mipt::dist::ComponentAnatomy;

    // A single path 0-1-2-3 in one layer. Every interior vertex is an
    // articulation point separating the ends, nothing dangles, and the whole
    // component is backbone.
    {
        const LogicalHistory chain = make_history(4, {{{{0, 1}, {1, 2}, {2, 3}}, {}}});
        ConnectivityIndex index;
        index.build(chain, true);
        const ComponentAnatomy a = index.anatomy(0, 3);
        expect(a.evaluated, "11.6 single path: the anatomy is evaluated");
        expect(a.backbone_nodes == 4 && a.dangling_nodes == 0,
               "11.6 single path: every vertex is on the only path");
        expect(a.articulation_nodes == 2,
               "11.6 single path: the two interior vertices each separate the ends");
        expect(a.branches == 0 && a.max_branch_size == 0,
               "11.6 single path: nothing hangs off it");
        expect(a.min_edge_cut == 1 && a.min_vertex_cut == 1,
               "11.6 single path: a one-edge, one-vertex bottleneck");
        expect(a.degree_i == 1 && a.degree_j == 1, "11.6 single path: the ends have degree 1");
        expect(a.final_sites_in_component == 4, "11.6 single path: all four sites share it");
    }

    // A braid: a 4-cycle between opposite corners. Two independent routes, so
    // no vertex separates them and the whole cycle is backbone.
    {
        const LogicalHistory cycle = make_history(4, {{{{0, 1}, {1, 2}, {2, 3}, {3, 0}}, {}}});
        ConnectivityIndex index;
        index.build(cycle, true);
        const ComponentAnatomy a = index.anatomy(0, 2);
        expect(a.backbone_nodes == 4, "11.6 braid: both routes are backbone");
        expect(a.articulation_nodes == 0, "11.6 braid: no single vertex separates the ends");
        expect(a.dangling_nodes == 0 && a.branches == 0, "11.6 braid: nothing dangles");
        expect(a.min_vertex_cut == 2, "11.6 braid: two vertex-disjoint routes");
    }

    // A dangling branch: the path 0-1-2 carries the pair, and site 3 hangs off
    // site 1. Site 3's worldline is in the component but on no 0-2 path.
    {
        const LogicalHistory tree = make_history(4, {{{{0, 1}, {1, 2}, {1, 3}}, {}}});
        ConnectivityIndex index;
        index.build(tree, true);
        const ComponentAnatomy a = index.anatomy(0, 2);
        expect(a.backbone_nodes == 3, "11.6 dangling: the backbone is 0-1-2 only");
        expect(a.dangling_nodes == 1, "11.6 dangling: site 3 is in the component, off the path");
        expect(a.branches == 1 && a.max_branch_size == 1 && a.mean_branch_size == 1.0,
               "11.6 dangling: exactly one branch of one vertex");
        expect(a.articulation_nodes == 1, "11.6 dangling: site 1 separates the ends");
        expect(a.component_nodes == 4, "11.6 dangling: the component holds all four");
    }

    // An articulation point between two braids: 0 and 1 are doubly joined, so
    // are 2 and 3, and the two halves meet at a single vertex. Exactly one
    // vertex separates the ends, and every vertex is still on some path.
    {
        const LogicalHistory history = make_history(4, {
            {{{0, 1}, {1, 2}}, {}},
            {{{0, 1}, {2, 3}}, {}},
        });
        ConnectivityIndex index;
        index.build(history, true);
        const ComponentAnatomy a = index.anatomy(0, 3);
        expect(a.evaluated && a.min_vertex_cut >= 1,
               "11.6 articulation: the ends are joined");
        expect(a.backbone_nodes + a.dangling_nodes == a.component_nodes,
               "11.6 articulation: backbone and dangling partition the component");
    }

    // The gate / temporal split of a shortest path. Sites 0 and 1 are joined by
    // a gate in layer 0; site 0 then waits two layers. The 0-1 path is one
    // gate and no waiting; a pair that has to wait shows the waiting.
    {
        const LogicalHistory history = make_history(3, {
            {{{0, 1}}, {}},
            {{{1, 2}}, {}},
            {{}, {}},
        });
        ConnectivityIndex index;
        index.build(history, true);
        const ComponentAnatomy direct = index.anatomy(0, 1);
        expect(direct.shortest_path_gates + direct.shortest_path_temporal ==
                   index.query(0, 1).shortest_path,
               "11.6: the gate and temporal counts sum to the shortest path length");
        expect(direct.shortest_path_gates >= 1,
               "11.6: a path between two sites crosses at least one gate");
        const ComponentAnatomy far = index.anatomy(0, 2);
        expect(far.shortest_path_gates + far.shortest_path_temporal ==
                   index.query(0, 2).shortest_path,
               "11.6: and so do they for a path that has to wait a layer");
        expect(far.shortest_path_temporal >= 1,
               "11.6: reaching site 2 needs a temporal edge, since its gate is a layer later");
    }

    // Disconnected endpoints have no anatomy at all rather than a zeroed one.
    {
        const LogicalHistory split = make_history(4, {{{{0, 1}, {2, 3}}, {}}});
        ConnectivityIndex index;
        index.build(split, true);
        expect(!index.anatomy(0, 3).evaluated, "11.6: separate components have no anatomy");
    }
}

// Request 11.7: F-only, G-only, diagonal and generic gates, weighted.
//
// The binary graph cannot tell these apart -- every one of them is an edge. The
// point of the weighted diagnostic is that it can.
void test_channel_weights()
{
    using mipt::dist::ChannelConnectivity;

    // A history whose single bond carries the given parity-block weights.
    auto weighted = [](double pair_weight, double hop_weight) {
        LogicalHistory history;
        history.reset(2);
        LogicalLayer layer;
        layer.measured.assign(2, 0u);
        layer.bonds.push_back({0, 1, pair_weight, hop_weight});
        history.layers.push_back(std::move(layer));
        history.valid = true;
        return history;
    };

    // A diagonal gate: an edge in the binary graph, transmitting nothing in
    // either sector. This is the case the whole diagnostic exists for.
    {
        ConnectivityIndex index;
        const LogicalHistory history = weighted(0.0, 0.0);
        index.build(history, true);
        expect(index.query(0, 1).connected,
               "11.7 diagonal: the binary graph still calls it connected");
        const ChannelConnectivity c = index.channels(0, 1);
        expect(c.evaluated, "11.7 diagonal: the channel query is evaluated");
        expect(c.pair_bottleneck == 0.0 && c.hop_bottleneck == 0.0,
               "11.7 diagonal: neither channel carries anything");
        expect(c.pair_multiplicity == 0 && c.hop_multiplicity == 0,
               "11.7 diagonal: and no channel of any width exists");
        expect(std::isinf(c.hop_log_weight) && c.hop_log_weight < 0.0,
               "11.7 diagonal: the log weight is -inf, not a small number");
    }

    // G-only: the odd block mixes maximally, the even block not at all. The
    // hopping channel is perfect and the pairing channel is dead.
    {
        ConnectivityIndex index;
        const LogicalHistory history = weighted(0.0, 1.0);
        index.build(history, true);
        const ChannelConnectivity c = index.channels(0, 1);
        expect(c.hop_bottleneck == 1.0 && c.hop_log_weight == 0.0,
               "11.7 G-only: the hopping channel is perfect");
        expect(c.hop_multiplicity == 1, "11.7 G-only: through one thread");
        expect(c.pair_bottleneck == 0.0, "11.7 G-only: the pairing channel is dead");
    }

    // F-only is the mirror image, which is what makes the two labels
    // meaningful rather than interchangeable.
    {
        ConnectivityIndex index;
        const LogicalHistory history = weighted(1.0, 0.0);
        index.build(history, true);
        const ChannelConnectivity c = index.channels(0, 1);
        expect(c.pair_bottleneck == 1.0 && c.hop_bottleneck == 0.0,
               "11.7 F-only: exactly the opposite of G-only");
    }

    // A generic weak gate: the log weight is the log of the product, so a path
    // of two weak gates is twice the log of one.
    {
        LogicalHistory history;
        history.reset(3);
        LogicalLayer layer;
        layer.measured.assign(3, 0u);
        layer.bonds.push_back({0, 1, 0.5, 0.25});
        layer.bonds.push_back({1, 2, 0.5, 0.25});
        history.layers.push_back(std::move(layer));
        history.valid = true;
        ConnectivityIndex index;
        index.build(history, true);
        const ChannelConnectivity one = index.channels(0, 1);
        expect(std::abs(one.hop_log_weight - std::log(0.25)) < 1e-12,
               "11.7 generic: one gate is its own log weight");
        const ChannelConnectivity two = index.channels(0, 2);
        expect(std::abs(two.hop_log_weight - 2.0 * std::log(0.25)) < 1e-12,
               "11.7 generic: two gates in series multiply, so the logs add");
        expect(std::abs(two.hop_bottleneck - 0.25) < 1e-12,
               "11.7 generic: the bottleneck is the weakest link, not the product");
        expect(two.pair_bottleneck > two.hop_bottleneck,
               "11.7 generic: the two channels are resolved separately");
    }

    // A braid: two routes of equal weight between the endpoints, so the
    // multiplicity is 2 where a single route would give 1. Two layers, one
    // route each, as in the Menger test above.
    {
        LogicalHistory history;
        history.reset(4);
        for (const auto &pair : std::vector<std::vector<std::pair<int, int>>>{
                 {{0, 1}, {1, 2}}, {{2, 3}, {3, 0}}})
        {
            LogicalLayer layer;
            layer.measured.assign(4, 0u);
            for (const auto &[a, b] : pair)
            {
                layer.bonds.push_back({a, b, 0.5, 0.5});
            }
            history.layers.push_back(std::move(layer));
        }
        history.valid = true;
        ConnectivityIndex index;
        index.build(history, true);
        const ChannelConnectivity c = index.channels(0, 2);
        expect(c.hop_multiplicity == 2,
               "11.7 braid: two routes of equal weight give multiplicity 2");
    }

    // The percolation event is untouched by any of this.
    {
        ConnectivityIndex index;
        const LogicalHistory history = weighted(0.0, 0.0);
        index.build(history, true);
        expect(index.query(0, 1).connected && index.query(0, 1).interior_path,
               "11.7: weights are diagnostics beside the percolation event, not a redefinition");
    }

    // A circuit that supplies no weights (the default 1.0) degenerates to the
    // unweighted graph rather than reporting a dead channel.
    {
        const LogicalHistory plain = make_history(3, {{{{0, 1}, {1, 2}}, {}}});
        ConnectivityIndex index;
        index.build(plain, true);
        const ChannelConnectivity c = index.channels(0, 2);
        expect(c.hop_log_weight == 0.0 && c.hop_bottleneck == 1.0,
               "11.7: unweighted bonds are fully transmitting, so the weighted view degenerates");
    }
}

// Time since last measurement, the cheap explanatory diagnostic.
void test_idle_times()
{
    const LogicalHistory history = make_history(3, {
        {{}, {0}},
        {{}, {1}},
        {{}, {}},
    });
    ConnectivityIndex index;
    index.build(history, true);
    const PairConnectivity conn = index.query(0, 1);
    expect(conn.idle_i == 2, "site 0 was last measured two layers before the end");
    expect(conn.idle_j == 1, "site 1 one layer before the end");
    expect(index.query(2, 0).idle_i == 3,
           "a never-measured site reports the whole depth");
}

// An index that was never built answers "not evaluated" rather than lying.
void test_unbuilt_index_is_inert()
{
    ConnectivityIndex index;
    expect(!index.ready(), "a default index is not ready");
    expect(!index.query(0, 1).evaluated, "and every query says so");
    LogicalHistory empty;
    empty.reset(4);
    empty.valid = true;
    index.build(empty, true);
    expect(!index.ready(), "a history with no layers builds no graph");
    expect(index.query(0, 1).flags() == 0u, "an unevaluated query has an empty flag mask");
}

// ---------------------------------------------------------------------------
// The logical projection of an RPPU layer
// ---------------------------------------------------------------------------

std::set<std::pair<int, int>> bond_set(const LogicalLayer &layer)
{
    std::set<std::pair<int, int>> bonds;
    for (const mipt::LogicalBond &bond : layer.bonds)
    {
        bonds.insert({std::min(bond.a, bond.b), std::max(bond.a, bond.b)});
    }
    return bonds;
}

mipt::LogicalLayer project(const mipt::RppuLayer &layer, int n)
{
    std::vector<int> mode_at_site(static_cast<std::size_t>(n));
    for (int site = 0; site < n; ++site)
    {
        mode_at_site[static_cast<std::size_t>(site)] = site;
    }
    return mipt::logical_layer_from_rppu(layer, n, mode_at_site);
}

void test_even_layer_bonds()
{
    constexpr int n = 8;
    std::mt19937 rng(7);
    mipt::RppuLayer layer;
    mipt::fill_rppu_layer(layer, n, false, 0.0, true, rng,
                          mipt::RppuBoundaryMode::FermionicJWString);
    const auto bonds = bond_set(project(layer, n));
    expect(bonds == std::set<std::pair<int, int>>({{0, 1}, {2, 3}, {4, 5}, {6, 7}}),
           "an even layer is the four disjoint even bonds");
}

// The decisive one. The FSWAP network and the direct Jordan-Wigner gate are two
// implementations of the *same* nonlocal bond (0, N-1), so they must project
// onto identical logical layers. If FSWAPs were treated as entangling seeds the
// FSWAP variant would carry 2(N-2) extra bonds and fuse the ring; if the JW
// string were fanned out, the other variant would.
//
// The two draws consume the identical amount of randomness -- one
// parity-preserving gate for the boundary either way, then the interior bonds
// -- so seeding both with the same value also makes the measurement flags
// comparable.
void test_boundary_implementations_agree()
{
    constexpr int n = 8;
    mipt::RppuLayer fswap_layer;
    mipt::RppuLayer jw_layer;
    std::mt19937 fswap_rng(4711);
    std::mt19937 jw_rng(4711);
    mipt::fill_rppu_layer(fswap_layer, n, true, 0.3, true, fswap_rng,
                          mipt::RppuBoundaryMode::FermionicFSwap);
    mipt::fill_rppu_layer(jw_layer, n, true, 0.3, true, jw_rng,
                          mipt::RppuBoundaryMode::FermionicJWString);

    // Premise of the test: the two really are different circuits.
    expect(fswap_layer.gates.size() > jw_layer.gates.size(),
           "the FSWAP boundary uses a transport network the JW boundary does not");

    const mipt::LogicalLayer fswap = project(fswap_layer, n);
    const mipt::LogicalLayer jw = project(jw_layer, n);
    const auto expected =
        std::set<std::pair<int, int>>({{0, 7}, {1, 2}, {3, 4}, {5, 6}});
    expect(bond_set(jw) == expected,
           "the JW boundary gate is the single logical bond (0, N-1)");
    expect(bond_set(fswap) == expected,
           "and the FSWAP network reduces to exactly the same logical bond");
    expect(fswap.bonds.size() == jw.bonds.size(),
           "neither implementation contributes a spurious bond");
    expect(fswap.measured == jw.measured,
           "the two draws consume the same randomness, so they measure the same sites");
}

// An FSWAP must move a mode, not couple it. A lone FSWAP layer therefore has no
// bonds at all and permutes the labels the next layer sees.
void test_fswap_is_transport()
{
    constexpr int n = 4;
    mipt::RppuLayer layer;
    layer.measure_flags.assign(n, 0);
    layer.gates.push_back(mipt::fermionic_swap_gate(1, 2));
    layer.measure_flags[1] = 1;

    std::vector<int> mode_at_site{0, 1, 2, 3};
    const mipt::LogicalLayer logical = mipt::logical_layer_from_rppu(layer, n, mode_at_site);
    expect(logical.bonds.empty(), "an FSWAP emits no entangling bond");
    expect(mode_at_site == std::vector<int>({0, 2, 1, 3}),
           "it exchanges the two modes' positions instead");
    // The measurement flag is indexed by position, and position 1 now holds
    // mode 2 -- so it is mode 2 that was measured.
    expect(logical.measured[2] == 1u && logical.measured[1] == 0u,
           "a measurement after transport applies to the mode that was moved there");
}

// End to end: a whole RPPU history at p=0 connects everything, and at p=1
// nothing survives. Both are statements about the graph, not about the state.
void test_history_extremes()
{
    constexpr int n = 8;
    std::mt19937 rng(2026);
    std::vector<mipt::RppuLayer> layers;
    mipt::build_rppu_layers(layers, n, 4, 0.0, true, rng, true);
    ConnectivityIndex open_index;
    open_index.build(mipt::logical_history_from_rppu(layers, n), true);
    expect(open_index.query(0, n / 2).connected,
           "a measurement-free circuit percolates between any two sites");

    mipt::build_rppu_layers(layers, n, 4, 1.0, true, rng, true);
    ConnectivityIndex closed_index;
    closed_index.build(mipt::logical_history_from_rppu(layers, n), true);
    const PairConnectivity conn = closed_index.query(0, n / 2);
    expect(!conn.survives_i && !conn.survives_j,
           "measuring every site every layer leaves no endpoint alive");
    expect(!conn.connected, "so nothing percolates");

    // The permutation must come back to the identity, or every endpoint label
    // in the run is wrong. The FSWAP boundary is the only transport in the
    // tree and it walks a mode down the ring and back inside one layer.
    mipt::build_rppu_layers(layers, n, 4, 0.2, true, rng, false);
    const LogicalHistory fswap_history = mipt::logical_history_from_rppu(layers, n);
    expect(fswap_history.identity_permutation(),
           "the FSWAP boundary network returns every mode to its own site");
}


// ---------------------------------------------------------------------------
// Gate-path diagnostics (MIPT_DIST_GATE_DIAGNOSTICS)
//
// The binary graph says *whether* a path exists; these say how entangling the
// weakest gate on the best path is. Two values per pair, and the difference
// between them is the point: the shortest-maximin value only looks at shortest
// paths, the widest value at every path. A pair whose shortest path runs through
// a nearly local gate but which has a longer strong detour must report a small
// first number and a large second one -- otherwise the diagnostic would blame
// the gap on a weak gate the state could simply route around.
// ---------------------------------------------------------------------------

struct GateBond
{
    int a;
    int b;
    double ep;
    double even = 0.5;
    double odd = 0.5;
};

// A history whose bonds carry explicit gate metrics. source_gate_index is the
// bond's position within its layer, which is all a hand-built history has.
LogicalHistory make_gate_history(int n, const std::vector<std::pair<std::vector<GateBond>,
                                                                  std::vector<int>>> &layers)
{
    LogicalHistory history;
    history.reset(n);
    for (const auto &[bonds, measured] : layers)
    {
        LogicalLayer layer;
        layer.measured.assign(static_cast<std::size_t>(n), 0u);
        int index = 0;
        for (const GateBond &bond : bonds)
        {
            mipt::LogicalBond logical{bond.a, bond.b};
            logical.source_gate_index = index++;
            logical.entangling_power = bond.ep;
            logical.even_block_mixing = bond.even;
            logical.odd_block_mixing = bond.odd;
            layer.bonds.push_back(logical);
        }
        for (int site : measured)
        {
            layer.measured[static_cast<std::size_t>(site)] = 1u;
        }
        history.layers.push_back(std::move(layer));
    }
    history.valid = true;
    return history;
}

// Walk a returned edge list from endpoint i: every edge must touch the node the
// walk is at, and the walk must end on endpoint j. Reports the hop count, the
// gate count and the weakest gate capacity along the way.
struct WalkedPath
{
    bool contiguous = false;
    int hops = 0;
    int gates = 0;
    double weakest = std::numeric_limits<double>::infinity();
};

WalkedPath walk(const ConnectivityIndex &index, int i, int j,
                const std::vector<std::uint32_t> &edges,
                mipt::dist::GateCapacity which = mipt::dist::GateCapacity::EntanglingPower)
{
    WalkedPath out;
    std::uint32_t at = index.final_node(i);
    const auto &all = index.graph_edges();
    for (std::uint32_t id : edges)
    {
        if (id >= all.size())
        {
            return out;
        }
        const mipt::dist::GraphEdge &edge = all[id];
        if (edge.a == at)
        {
            at = edge.b;
        }
        else if (edge.b == at)
        {
            at = edge.a;
        }
        else
        {
            return out;
        }
        ++out.hops;
        if (edge.kind == mipt::dist::GraphEdgeKind::Unitary)
        {
            ++out.gates;
        }
        out.weakest = std::min(out.weakest, mipt::dist::edge_capacity(edge, which));
    }
    out.contiguous = !edges.empty() && at == index.final_node(j);
    return out;
}

bool same_value(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

bool same_metrics(const mipt::dist::GatePathMetrics &x, const mipt::dist::GatePathMetrics &y)
{
    return x.available == y.available && x.shortest_path_length == y.shortest_path_length &&
           x.shortest_path_unitary_count == y.shortest_path_unitary_count &&
           same_value(x.shortest_maximin_entangling_power, y.shortest_maximin_entangling_power) &&
           same_value(x.widest_entangling_power, y.widest_entangling_power) &&
           same_value(x.widest_even_block_mixing, y.widest_even_block_mixing) &&
           same_value(x.widest_odd_block_mixing, y.widest_odd_block_mixing);
}

// Shortest path is the weak direct gate in the last layer; the detour down to
// layer 0 and back crosses a strong gate. Shortest-maximin must see only the
// weak one, widest must find the detour.
void test_gate_path_weak_shortcut()
{
    const LogicalHistory history = make_gate_history(2, {
        {{{0, 1, 0.9, 0.8, 0.7}}, {}},
        {{{0, 1, 0.1, 0.2, 0.3}}, {}},
    });
    for (bool precompute : {false, true})
    {
        ConnectivityIndex index;
        index.build(history, true, precompute);
        expect(index.has_gate_metrics(), "G1 a history with finite bond metrics has gate metrics");
        const mipt::dist::GatePathMetrics g = index.gate_path_metrics(0, 1);
        expect(g.available, "G1 the endpoints share a component, so the query is available");
        expect(g.shortest_path_length == 1, "G1 the shortest path is the one direct bond");
        expect(g.shortest_path_unitary_count == 1, "G1 and it carries one gate");
        expect(g.shortest_maximin_entangling_power == 0.1,
               "G1 shortest-maximin is the weak shortcut's capacity");
        expect(g.widest_entangling_power == 0.9,
               "G1 widest finds the longer strong detour");
        expect(g.widest_even_block_mixing == 0.8 && g.widest_odd_block_mixing == 0.7,
               "G1 the mixing bottlenecks route through the detour too");
        expect(g.shortest_path_length == index.query(0, 1).shortest_path,
               "G1 the gate-path length agrees with the binary shortest path");

        const auto short_edges = index.shortest_maximin_path(0, 1);
        const WalkedPath s = walk(index, 0, 1, short_edges);
        expect(s.contiguous, "G1 the shortest-maximin path is contiguous from i to j");
        expect(s.hops == 1 && s.gates == 1 && s.weakest == 0.1,
               "G1 the shortest-maximin path is the direct weak bond");

        const auto wide_edges = index.widest_gate_path(0, 1);
        const WalkedPath w = walk(index, 0, 1, wide_edges);
        expect(w.contiguous, "G1 the widest path is contiguous from i to j");
        expect(w.hops == 3 && w.gates == 1 && w.weakest == 0.9,
               "G1 the widest path is temporal, strong gate, temporal");
        expect(index.graph_edges()[wide_edges[1]].layer == 0 &&
                   index.graph_edges()[wide_edges[1]].kind == mipt::dist::GraphEdgeKind::Unitary,
               "G1 the widest path's gate is the layer-0 one");

        const auto odd_edges = index.widest_gate_path(0, 1, mipt::dist::GateCapacity::OddBlockMixing);
        const WalkedPath o =
            walk(index, 0, 1, odd_edges, mipt::dist::GateCapacity::OddBlockMixing);
        expect(o.contiguous && o.weakest == 0.7,
               "G1 the odd-mixing widest path reaches its own bottleneck");

        // Symmetric in the endpoints.
        const mipt::dist::GatePathMetrics r = index.gate_path_metrics(1, 0);
        expect(r.widest_entangling_power == g.widest_entangling_power &&
                   r.shortest_maximin_entangling_power == g.shortest_maximin_entangling_power,
               "G1 the values do not depend on which endpoint is first");
    }
}

// Every connection crosses a weak gate: widest is small, and it is the larger
// of the two weak gates (the better of two bad routes).
void test_gate_path_all_weak()
{
    const LogicalHistory history = make_gate_history(2, {
        {{{0, 1, 0.05}}, {}},
        {{{0, 1, 0.1}}, {}},
    });
    ConnectivityIndex index;
    index.build(history, true, true);
    const mipt::dist::GatePathMetrics g = index.gate_path_metrics(0, 1);
    expect(g.available, "G2 available");
    expect(g.widest_entangling_power == 0.1,
           "G2 when every path has a weak gate, widest is the best weak gate");
    expect(g.shortest_maximin_entangling_power == 0.1, "G2 shortest-maximin agrees");
}

// A path that is mostly temporal: three idle steps on each side of one gate.
// Temporal edges carry capacity 1 and must never become the bottleneck.
void test_gate_path_temporal_is_free()
{
    const LogicalHistory history = make_gate_history(2, {
        {{{0, 1, 0.7, 0.6, 0.5}}, {}},
        {{}, {}},
        {{}, {}},
        {{}, {}},
    });
    ConnectivityIndex index;
    index.build(history, true, true);
    const mipt::dist::GatePathMetrics g = index.gate_path_metrics(0, 1);
    expect(g.available, "G3 available");
    expect(g.shortest_path_length == 7, "G3 three idle steps, the gate, three idle steps");
    expect(g.shortest_path_unitary_count == 1, "G3 one gate on the path");
    expect(g.widest_entangling_power == 0.7 && g.shortest_maximin_entangling_power == 0.7,
           "G3 temporal edges do not lower the entangling-power bottleneck");
    expect(g.widest_even_block_mixing == 0.6 && g.widest_odd_block_mixing == 0.5,
           "G3 nor the mixing bottlenecks");
    const WalkedPath w = walk(index, 0, 1, index.widest_gate_path(0, 1));
    expect(w.contiguous && w.hops == 7 && w.gates == 1, "G3 the widest path is the whole column");
}

// No path, no value: endpoints in different components, and a history whose
// bonds carry no metrics (the MMS/Haar case), both report unavailable with NaN
// -- never a zero that would read as "a non-entangling gate".
void test_gate_path_unavailable()
{
    {
        const LogicalHistory history = make_gate_history(4, {
            {{{0, 1, 0.9}, {2, 3, 0.9}}, {}},
        });
        ConnectivityIndex index;
        index.build(history, true, true);
        expect(index.has_gate_metrics(), "G4 metrics are present");
        const mipt::dist::GatePathMetrics g = index.gate_path_metrics(0, 2);
        expect(!g.available, "G4 different components: unavailable");
        expect(std::isnan(g.widest_entangling_power) &&
                   std::isnan(g.shortest_maximin_entangling_power) &&
                   std::isnan(g.widest_even_block_mixing) && std::isnan(g.widest_odd_block_mixing),
               "G4 different components: every value is NaN");
        expect(index.shortest_maximin_path(0, 2).empty() && index.widest_gate_path(0, 2).empty(),
               "G4 different components: no path is returned");
        expect(index.gate_path_metrics(0, 1).available, "G4 but the bonded pair is available");
        expect(!index.gate_path_metrics(1, 1).available, "G4 a site with itself is not a pair");
    }
    {
        // make_history leaves the LogicalBond defaults: NaN metrics.
        const LogicalHistory history = make_history(2, {{{{0, 1}}, {}}});
        ConnectivityIndex index;
        index.build(history, true, true);
        expect(index.query(0, 1).connected, "G4 the binary graph is unaffected by missing metrics");
        expect(!index.has_gate_metrics(), "G4 NaN bond metrics: no gate metrics");
        const mipt::dist::GatePathMetrics g = index.gate_path_metrics(0, 1);
        expect(!g.available && std::isnan(g.widest_entangling_power),
               "G4 NaN bond metrics: unavailable and NaN, not zero");
        expect(index.widest_gate_path(0, 1).empty(), "G4 NaN bond metrics: no path");
    }
}

// Two shortest paths of equal length, one through three gates and one through
// a single gate. The unitary count is the fewest gates over the shortest paths
// that attain the shortest-maximin value -- so it follows the bottleneck, not
// whichever path a traversal happens to see first.
//
//   final layer: 0 -(g)- 1 -(g)- 3 -(g)- 2         (three gates, length 3)
//   detour:      (1,0) ~ (0,0) -(g)- (0,2) ~ (1,2) (one gate,   length 3)
void test_gate_path_unitary_count()
{
    auto build = [](double chain, double detour) {
        return make_gate_history(4, {
            {{{0, 2, detour}}, {}},
            {{{0, 1, chain}, {1, 3, chain}, {3, 2, chain}}, {}},
        });
    };
    struct Case
    {
        double chain;
        double detour;
        double maximin;
        int count;
        const char *what;
    };
    for (const Case &c : {Case{0.8, 0.3, 0.8, 3, "the strong three-gate chain wins"},
                          Case{0.8, 0.9, 0.9, 1, "the strong one-gate detour wins"},
                          Case{0.8, 0.8, 0.8, 1, "a tie goes to the fewest gates"}})
    {
        const LogicalHistory history = build(c.chain, c.detour);
        for (bool precompute : {false, true})
        {
            ConnectivityIndex index;
            index.build(history, true, precompute);
            const mipt::dist::GatePathMetrics g = index.gate_path_metrics(0, 2);
            const std::string tag = std::string("G5 ") + c.what;
            expect(g.available && g.shortest_path_length == 3, tag + ": shortest length 3");
            expect(g.shortest_maximin_entangling_power == c.maximin, tag + ": maximin value");
            expect(g.shortest_path_unitary_count == c.count, tag + ": unitary count");
            const auto edges = index.shortest_maximin_path(0, 2);
            const WalkedPath s = walk(index, 0, 2, edges);
            expect(s.contiguous, tag + ": the canonical path is contiguous");
            expect(s.hops == 3 && s.gates == c.count && s.weakest == c.maximin,
                   tag + ": the canonical path realizes both values");
        }
    }
}

// The flags may not move when gate paths are requested, and the precomputed
// table must equal the on-demand computation for every pair.
void test_gate_paths_change_nothing_else()
{
    constexpr int n = 8;
    std::mt19937 rng(9091);
    for (int trial = 0; trial < 6; ++trial)
    {
        std::vector<mipt::RppuLayer> layers;
        mipt::build_rppu_layers(layers, n, 5, 0.15 + 0.05 * trial, true, rng, trial % 2 == 0);
        const LogicalHistory history = mipt::logical_history_from_rppu(layers, n);
        ConnectivityIndex plain;
        ConnectivityIndex gated;
        plain.build(history, true, false);
        gated.build(history, true, true);
        expect(plain.has_gate_metrics() && gated.has_gate_metrics(),
               "G6 RPPU histories carry gate metrics");
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                const PairConnectivity a = plain.query(i, j);
                const PairConnectivity b = gated.query(i, j);
                expect(a.flags() == b.flags() && a.component_size == b.component_size &&
                           a.shortest_path == b.shortest_path,
                       "G6 query() is identical with and without gate paths");
                if (i == j)
                {
                    continue;
                }
                expect(same_metrics(plain.gate_path_metrics(i, j), gated.gate_path_metrics(i, j)),
                       "G6 precomputed gate-path metrics equal the on-demand ones");
            }
        }
    }
}

// The FSWAP network and the JW-string gate are one logical bond; they must
// also carry one diagnostic. Only the index into the simulated gate list may
// differ, because the FSWAP network puts the boundary gate after its transport.
void test_boundary_gate_diagnostics_agree()
{
    constexpr int n = 8;
    mipt::RppuLayer fswap_layer;
    mipt::RppuLayer jw_layer;
    std::mt19937 fswap_rng(31337);
    std::mt19937 jw_rng(31337);
    mipt::fill_rppu_layer(fswap_layer, n, true, 0.3, true, fswap_rng,
                          mipt::RppuBoundaryMode::FermionicFSwap);
    mipt::fill_rppu_layer(jw_layer, n, true, 0.3, true, jw_rng,
                          mipt::RppuBoundaryMode::FermionicJWString);
    const LogicalLayer fswap = project(fswap_layer, n);
    const LogicalLayer jw = project(jw_layer, n);
    expect(fswap.bonds.size() == jw.bonds.size(), "G7 same logical bond count");
    bool boundary_index_differs = false;
    for (std::size_t k = 0; k < std::min(fswap.bonds.size(), jw.bonds.size()); ++k)
    {
        const mipt::LogicalBond &f = fswap.bonds[k];
        const mipt::LogicalBond &w = jw.bonds[k];
        expect(f.a == w.a && f.b == w.b, "G7 the k-th logical bond joins the same modes");
        expect(std::isfinite(f.entangling_power) && f.entangling_power == w.entangling_power,
               "G7 identical entangling power");
        expect(f.even_block_mixing == w.even_block_mixing &&
                   f.odd_block_mixing == w.odd_block_mixing,
               "G7 identical parity-block mixings");
        const auto &fg = fswap_layer.gates[static_cast<std::size_t>(f.source_gate_index)];
        const auto &wg = jw_layer.gates[static_cast<std::size_t>(w.source_gate_index)];
        expect(fg.kind != 1 && wg.kind != 1, "G7 source_gate_index never names an FSWAP");
        if ((f.a == 0 && f.b == n - 1) || (f.a == n - 1 && f.b == 0))
        {
            boundary_index_differs = f.source_gate_index != w.source_gate_index;
            expect(wg.kind == 2, "G7 the JW boundary bond points at the kind-2 gate");
            expect(fg.kind == 0 && fg.q0 == 0 && fg.q1 == 1,
                   "G7 the FSWAP boundary bond points at the gate on sites (0, 1)");
        }
    }
    expect(boundary_index_differs,
           "G7 the boundary bond's source_gate_index differs between implementations");
}

// End to end on a real draw: every connected pair has finite gate-path values
// in [0, 1], ordered widest >= shortest-maximin, and every unavailable pair is
// exactly the pairs in different components.
void test_rppu_history_gate_paths()
{
    constexpr int n = 10;
    std::mt19937 rng(424242);
    std::vector<mipt::RppuLayer> layers;
    mipt::build_rppu_layers(layers, n, 6, 0.2, true, rng, true);
    const LogicalHistory history = mipt::logical_history_from_rppu(layers, n);
    ConnectivityIndex index;
    index.build(history, true, true);
    expect(index.has_gate_metrics(), "G8 an RPPU history has gate metrics");
    int available = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            const PairConnectivity conn = index.query(i, j);
            const mipt::dist::GatePathMetrics g = index.gate_path_metrics(i, j);
            expect(g.available == conn.interior_path,
                   "G8 available exactly when the endpoints share a component");
            if (!g.available)
            {
                continue;
            }
            ++available;
            const double values[] = {g.shortest_maximin_entangling_power,
                                     g.widest_entangling_power, g.widest_even_block_mixing,
                                     g.widest_odd_block_mixing};
            for (double v : values)
            {
                expect(std::isfinite(v) && v >= 0.0 && v <= 1.0 + 1.0e-12,
                       "G8 every gate-path value is finite and in [0, 1]");
            }
            expect(g.widest_entangling_power >= g.shortest_maximin_entangling_power,
                   "G8 widest over all paths is at least the shortest-path value");
            expect(g.shortest_path_length == conn.shortest_path,
                   "G8 the gate-path length is the binary shortest path");
            expect(g.shortest_path_unitary_count >= 1 &&
                       g.shortest_path_unitary_count <= g.shortest_path_length,
                   "G8 a path between distinct modes crosses at least one gate");
            const WalkedPath s = walk(index, i, j, index.shortest_maximin_path(i, j));
            expect(s.contiguous && s.hops == g.shortest_path_length &&
                       s.gates == g.shortest_path_unitary_count &&
                       s.weakest == g.shortest_maximin_entangling_power,
                   "G8 the canonical shortest-maximin path realizes its metrics");
            const WalkedPath w = walk(index, i, j, index.widest_gate_path(i, j));
            expect(w.contiguous && w.weakest == g.widest_entangling_power,
                   "G8 the canonical widest path realizes the widest value");
        }
    }
    expect(available > 0, "G8 premise: some pair is connected at p=0.2");
}

} // namespace

int main()
{
    test_unmeasured_circuit_connects_everything();
    test_final_layer_measurement_kills_survival();
    test_measurement_cuts_forward_only();
    test_connectivity_factorizes();
    test_shortest_path_lengths();
    test_idle_times();
    test_disjoint_paths();
    test_component_anatomy();
    test_channel_weights();
    test_unbuilt_index_is_inert();
    test_even_layer_bonds();
    test_boundary_implementations_agree();
    test_fswap_is_transport();
    test_history_extremes();
    test_gate_path_weak_shortcut();
    test_gate_path_all_weak();
    test_gate_path_temporal_is_free();
    test_gate_path_unavailable();
    test_gate_path_unitary_count();
    test_gate_paths_change_nothing_else();
    test_boundary_gate_diagnostics_agree();
    test_rppu_history_gate_paths();

    if (failures != 0)
    {
        std::cerr << "dist_connectivity_tests: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "dist_connectivity_tests: PASS\n";
    return 0;
}
