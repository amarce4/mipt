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
#include <iostream>
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
    test_unbuilt_index_is_inert();
    test_even_layer_bonds();
    test_boundary_implementations_agree();
    test_fswap_is_transport();
    test_history_extremes();

    if (failures != 0)
    {
        std::cerr << "dist_connectivity_tests: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "dist_connectivity_tests: PASS\n";
    return 0;
}
