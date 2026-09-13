// Host tests for the connected-zero triple analysis (dist_pair_gap.hpp).
//
// The whole protocol is driven here with no GPU and no MOSEK: RDMs come from a
// first-principles fermionic reduction of an explicit state vector -- the same
// inversion-counting reference tests/rho_reduce_tests.cu holds the CUDA kernels
// to -- and fGMN from a deterministic fake solver that respects the
// fGMN <= min-cut bound. What is checked is the bookkeeping the request makes
// non-negotiable: every outcome recorded, nothing duplicated, nothing lost in a
// crash, and the aggregate reconciling exactly with the rows.
//
// Numbered as in the request:
//   1  exactly L-2 rows per qualifying pair
//   2  no row with k = i or k = j
//   3  shared triples computed once and fanned out
//   4  pair-class boundaries at every tolerance
//   5  pair RDMs agree with the marginals of the triple RDM
//   6  cut negativities invariant under relabelling (fGMN: make test-gmn)
//   7  known product / biseparable / genuinely tripartite cut values
//   8  the prefilter decision
//   9  solver failures preserved as missing, never zero
//   10 interruption and resume without missing or duplicated outcomes
//   11 exact reconciliation between the outcome CSV and the aggregate

#include "mipt/dist_gap_audit.hpp"
#include "mipt/dist_replay.hpp"
#include "mipt/dist_pair_gap.hpp"

#include <algorithm>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <map>
#include <set>
#include <tuple>
#include <sstream>
#include <string>
#include <vector>

namespace
{
namespace gap = mipt::dist::gap;
namespace dist = mipt::dist;
using Complex = std::complex<double>;

int failures = 0;

void expect(bool condition, const std::string &message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool refuses(const std::function<void()> &action)
{
    try
    {
        action();
    }
    catch (const std::runtime_error &)
    {
        return true;
    }
    return false;
}

std::string scratch()
{
    const auto path = std::filesystem::temp_directory_path() / "mipt_pair_gap_tests";
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path.string();
}

std::string read_file(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// ---------------------------------------------------------------------------
// First-principles fermionic reduction (see tests/rho_reduce_tests.cu)
// ---------------------------------------------------------------------------

bool jordan_wigner_odd(std::uint64_t index, const std::vector<int> &kept, int n)
{
    std::vector<int> ascending;
    for (int q = 0; q < n; ++q)
    {
        if ((index >> q) & 1u)
        {
            ascending.push_back(q);
        }
    }
    std::vector<int> target;
    for (int q : kept)
    {
        if ((index >> q) & 1u)
        {
            target.push_back(q);
        }
    }
    for (int q = 0; q < n; ++q)
    {
        if (std::find(kept.begin(), kept.end(), q) == kept.end() && ((index >> q) & 1u))
        {
            target.push_back(q);
        }
    }
    std::vector<int> order;
    for (int q : target)
    {
        order.push_back(
            static_cast<int>(std::find(ascending.begin(), ascending.end(), q) - ascending.begin()));
    }
    int inversions = 0;
    for (std::size_t a = 0; a < order.size(); ++a)
    {
        for (std::size_t b = a + 1; b < order.size(); ++b)
        {
            inversions += order[a] > order[b] ? 1 : 0;
        }
    }
    return (inversions & 1) != 0;
}

std::vector<double> reference_rdm(const std::vector<Complex> &psi, int n,
                                  const std::vector<int> &kept, bool fermionic)
{
    const int dim = 1 << static_cast<int>(kept.size());
    std::vector<Complex> rho(static_cast<std::size_t>(dim * dim));
    std::uint64_t kept_mask = 0;
    for (int q : kept)
    {
        kept_mask |= std::uint64_t{1} << q;
    }
    std::vector<Complex> column(static_cast<std::size_t>(dim));
    for (std::uint64_t env = 0; env < (std::uint64_t{1} << n); ++env)
    {
        if ((env & kept_mask) != 0)
        {
            continue;
        }
        for (int r = 0; r < dim; ++r)
        {
            std::uint64_t index = env;
            for (std::size_t k = 0; k < kept.size(); ++k)
            {
                if ((r >> k) & 1)
                {
                    index |= std::uint64_t{1} << kept[k];
                }
            }
            const double sign = (fermionic && jordan_wigner_odd(index, kept, n)) ? -1.0 : 1.0;
            column[static_cast<std::size_t>(r)] = sign * psi[index];
        }
        for (int r = 0; r < dim; ++r)
        {
            for (int c = 0; c < dim; ++c)
            {
                rho[static_cast<std::size_t>(r * dim + c)] +=
                    column[static_cast<std::size_t>(r)] * std::conj(column[static_cast<std::size_t>(c)]);
            }
        }
    }
    std::vector<double> out(2u * rho.size());
    for (std::size_t e = 0; e < rho.size(); ++e)
    {
        out[2 * e] = rho[e].real();
        out[2 * e + 1] = rho[e].imag();
    }
    return out;
}

// A random even-parity state on `n` modes.
std::vector<Complex> random_even_state(int n, std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Complex> psi(std::size_t{1} << n);
    double norm = 0.0;
    for (std::size_t x = 0; x < psi.size(); ++x)
    {
        if (__builtin_popcountll(x) % 2 == 0)
        {
            psi[x] = Complex(normal(rng), normal(rng));
            norm += std::norm(psi[x]);
        }
    }
    for (auto &a : psi)
    {
        a /= std::sqrt(norm);
    }
    return psi;
}

// psi_A (modes 0..2) (x) psi_B (modes 3..5): every pair straddling the cut is a
// product, so its fN, G, F and I_occ vanish exactly -- a supply of silent pairs
// with genuinely entangled neighbours on either side.
std::vector<Complex> split_state(std::mt19937 &rng)
{
    const std::vector<Complex> a = random_even_state(3, rng);
    const std::vector<Complex> b = random_even_state(3, rng);
    std::vector<Complex> psi(64);
    for (std::size_t x = 0; x < 64; ++x)
    {
        psi[x] = a[x & 7u] * b[x >> 3];
    }
    return psi;
}

// ---------------------------------------------------------------------------
// A synthetic trajectory: state, graph, and every pair's metrics
// ---------------------------------------------------------------------------

constexpr int N = 6;

mipt::LogicalHistory ring_history(const std::vector<int> &measured_last)
{
    mipt::LogicalHistory history;
    history.reset(N);
    mipt::LogicalLayer even;
    even.measured.assign(N, 0u);
    even.bonds = {{0, 1}, {2, 3}, {4, 5}};
    mipt::LogicalLayer odd;
    odd.measured.assign(N, 0u);
    odd.bonds = {{1, 2}, {3, 4}, {5, 0}};
    for (int site : measured_last)
    {
        odd.measured[static_cast<std::size_t>(site)] = 1u;
    }
    history.layers = {even, odd};
    history.valid = true;
    return history;
}

dist::RunConfig test_config(const std::string &directory)
{
    dist::RunConfig config;
    config.k = 2;
    config.n = N;
    config.periods = 4;
    config.p = 0.2;
    config.realizations = 8;
    config.type = mipt::CircuitType::FermionRPPU;
    config.output_path = directory + "/pairs.csv";
    config.seed = 12345;
    config.statevector_precision = 64;
    // Between the cross pairs of the perturbed split (fN <= 5.7e-4 over the
    // fixture's seeds) and every in-block pair (fN >= 1.06e-2), so the trigger
    // is exact on both kinds of trajectory.
    config.pair_zero_tol = 2.0e-3;
    config.pair_gap.enabled = true;
    config.pair_gap.output_path = directory + "/gap.csv";
    config.pair_gap.store_rho3 = true;
    config.pair_gap.controls = true;
    return config;
}

struct Trajectory
{
    std::vector<Complex> psi;
    mipt::LogicalHistory history;
    dist::ConnectivityIndex connectivity;
    std::vector<std::vector<double>> pair_rdms;
    gap::PairMeasurement measurement;
};

// Build trajectory `t` of the synthetic run, deterministically.
std::unique_ptr<Trajectory> make_trajectory(std::uint32_t t, const dist::RunConfig &config)
{
    auto out = std::make_unique<Trajectory>();
    std::mt19937 rng(1000u + t);
    out->psi = split_state(rng);
    // Odd trajectories weakly couple the two halves. Their cross pairs stay
    // under the pair threshold but are no longer exact products, so no
    // singleton cut vanishes and every anchor triple needs a real solve; the
    // even trajectories supply the exactly-prefiltered case. Both paths run in
    // every driver test.
    if (t % 2 == 1)
    {
        const std::vector<Complex> chi = random_even_state(N, rng);
        double norm = 0.0;
        for (std::size_t x = 0; x < out->psi.size(); ++x)
        {
            out->psi[x] += 0.05 * chi[x];
            norm += std::norm(out->psi[x]);
        }
        for (Complex &a : out->psi)
        {
            a /= std::sqrt(norm);
        }
    }
    // Vary which endpoints die in the last layer, so some straddling pairs
    // are disconnected and serve as controls.
    std::vector<int> measured;
    if (t % 3 == 1) measured = {3};
    if (t % 3 == 2) measured = {0, 5};
    out->history = ring_history(measured);
    out->connectivity.build(out->history, true);
    out->measurement.reset(N);
    out->measurement.realization = t;
    const int max_separation = N / 2;
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            out->pair_rdms.push_back(reference_rdm(out->psi, N, {i, j}, true));
        }
    }
    std::size_t index = 0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            gap::PairSnapshot snap;
            snap.i = i;
            snap.j = j;
            snap.separation = mipt::util::periodic_distance(i, j, N);
            snap.chord = mipt::util::chord_length(N, snap.separation);
            snap.fermion_rho_ri = out->pair_rdms[index].data();
            snap.fermion = dist::two_party_metrics(snap.fermion_rho_ri, true, true);
            snap.conn = out->connectivity.query(i, j);
            snap.entangled = snap.fermion.mn > config.pair_zero_tol;
            snap.zero_class = dist::classify_unentangled(
                snap.fermion.i_occ, snap.fermion.g2 + snap.fermion.f2, config.occupation_mi_tol,
                config.channel_floor);
            (void)max_separation;
            out->measurement.add(snap);
            ++index;
        }
    }
    return out;
}

// What PairProtocol does to the bins for the contingency and class counts --
// the only bin fields the gap aggregate reads or reconciles against.
void bin_trajectory(std::vector<dist::PairBin> &bins, const gap::PairMeasurement &measurement)
{
    for (const gap::PairSnapshot &pair : measurement.pairs)
    {
        dist::PairBin &bin = bins[static_cast<std::size_t>(pair.separation - 1)];
        ++bin.conn_records;
        if (pair.conn.connected)
        {
            ++bin.connected;
            if (pair.entangled)
            {
                ++bin.connected_ent_positive;
            }
            else
            {
                ++bin.connected_ent_zero;
                switch (pair.zero_class)
                {
                case dist::ZeroClass::OccupationCorrelated: ++bin.classical_occ_correlated; break;
                case dist::ZeroClass::CoherentSubthreshold: ++bin.classical_subthreshold; break;
                case dist::ZeroClass::Silent: ++bin.classical_silent; break;
                }
            }
        }
        else if (pair.entangled)
        {
            ++bin.disconnected_ent_positive;
        }
    }
}

gap::RdmBatch reference_batch(const Trajectory &trajectory)
{
    return [&trajectory](const std::vector<std::array<int, 3>> &triples, std::vector<double> &raw) {
        for (std::size_t t = 0; t < triples.size(); ++t)
        {
            const std::vector<double> rho = reference_rdm(
                trajectory.psi, N, {triples[t][0], triples[t][1], triples[t][2]}, true);
            std::copy(rho.begin(), rho.end(), raw.begin() + static_cast<std::ptrdiff_t>(t * 128));
        }
    };
}

// fGMN stand-in: half the minimum cut, which respects fGMN <= min_s N_s.
// SolveOutcome::exact stands in for a solver that certifies its own answer, so
// the driver tests exercise the certified path with a degenerate interval.
gap::SolveOutcome fake_solver(const double *rho)
{
    std::array<double, 3> cuts{};
    mipt::analysis::cut_negativities<3>(rho, true, cuts);
    return gap::SolveOutcome::exact(0.5 * *std::min_element(cuts.begin(), cuts.end()));
}

// ---------------------------------------------------------------------------
// 1, 2, 3: planning
// ---------------------------------------------------------------------------

void test_plan_counts_and_dedup(const std::string &directory)
{
    const dist::RunConfig config = test_config(directory);
    const auto trajectory = make_trajectory(0, config);
    const auto anchors = gap::select_anchors(trajectory->measurement, config.pair_gap, config.seed);
    expect(!anchors.empty(), "the split state supplies connected-zero anchors");
    const gap::TriplePlan plan = gap::plan_triples(trajectory->measurement, anchors);

    std::size_t first_uses = 0;
    for (std::size_t a = 0; a < anchors.size(); ++a)
    {
        const auto &pair = trajectory->measurement.pairs[anchors[a].pair_index];
        expect(plan.thirds[a].size() == static_cast<std::size_t>(N - 2),
               "test 1: exactly L-2 third sites per qualifying pair");
        std::set<int> seen;
        for (const auto &third : plan.thirds[a])
        {
            expect(third.k != pair.i && third.k != pair.j, "test 2: no third site equals i or j");
            seen.insert(third.k);
            const auto &triple = plan.triples[third.triple];
            expect(std::is_sorted(triple.begin(), triple.end()), "triples are stored sorted");
            expect(std::count(triple.begin(), triple.end(), pair.i) == 1 &&
                       std::count(triple.begin(), triple.end(), pair.j) == 1 &&
                       std::count(triple.begin(), triple.end(), third.k) == 1,
                   "test 3: each outcome references the triple of its own three sites");
            first_uses += third.first_use ? 1u : 0u;
        }
        expect(seen.size() == static_cast<std::size_t>(N - 2), "every k appears exactly once");
    }
    expect(first_uses == plan.triples.size(),
           "test 3: exactly one first use per unique triple");
    std::set<std::uint32_t> ids(plan.triple_ids.begin(), plan.triple_ids.end());
    expect(ids.size() == plan.triples.size(), "test 3: unique triples have unique ids");
    expect(plan.triples.size() < anchors.size() * static_cast<std::size_t>(N - 2),
           "test 3: anchors share triples here, so deduplication must have fired");

    // The colex rank is a bijection onto 0..C(n,3)-1.
    std::set<std::uint32_t> ranks;
    for (int c = 2; c < N; ++c)
        for (int b = 1; b < c; ++b)
            for (int a = 0; a < b; ++a)
                ranks.insert(gap::triple_rank(a, b, c));
    expect(ranks.size() == 20 && *ranks.rbegin() == 19, "triple ids are the colex ranks 0..19");
}

void test_controls(const std::string &directory)
{
    const dist::RunConfig config = test_config(directory);
    for (std::uint32_t t = 0; t < 6; ++t)
    {
        const auto trajectory = make_trajectory(t, config);
        const auto first = gap::select_anchors(trajectory->measurement, config.pair_gap, config.seed);
        const auto second = gap::select_anchors(trajectory->measurement, config.pair_gap, config.seed);
        expect(first.size() == second.size(), "control selection is deterministic");
        std::set<std::size_t> controls;
        for (std::size_t a = 0; a < first.size(); ++a)
        {
            expect(first[a].pair_index == second[a].pair_index, "the same pairs, in the same order");
            if (first[a].role != gap::AnchorRole::Control)
            {
                continue;
            }
            const auto &control = trajectory->measurement.pairs[first[a].pair_index];
            const auto &anchor =
                trajectory->measurement.pair(first[a].partner_i, first[a].partner_j);
            expect(gap::is_disconnected_zero(control), "a control is disconnected and unentangled");
            expect(control.separation == anchor.separation, "a control is distance-matched");
            expect(controls.insert(first[a].pair_index).second,
                   "controls are drawn without replacement");
        }
    }
}

// ---------------------------------------------------------------------------
// 4: class boundaries
// ---------------------------------------------------------------------------

void test_class_boundaries()
{
    using dist::ZeroClass;
    for (double occ_tol : {0.0, 1.0e-12, 1.0e-6})
    {
        for (double floor : {0.0, 1.0e-30, 1.0e-13})
        {
            const double above_occ = std::nextafter(occ_tol, 1.0);
            const double above_floor = std::nextafter(floor, 1.0);
            expect(dist::classify_unentangled(occ_tol, floor, occ_tol, floor) == ZeroClass::Silent,
                   "test 4: values exactly at both tolerances are silent");
            expect(dist::classify_unentangled(above_occ, floor, occ_tol, floor) ==
                       ZeroClass::OccupationCorrelated,
                   "test 4: one ulp above the occupation tolerance is occupation-correlated");
            expect(dist::classify_unentangled(occ_tol, above_floor, occ_tol, floor) ==
                       ZeroClass::CoherentSubthreshold,
                   "test 4: one ulp above the channel floor is coherent-subthreshold");
            expect(dist::classify_unentangled(above_occ, above_floor, occ_tol, floor) ==
                       ZeroClass::OccupationCorrelated,
                   "test 4: occupation correlation takes precedence over coherence");
        }
    }
    using dist::GapSelection;
    expect(dist::gap_selects(GapSelection::ConnectedZero, ZeroClass::OccupationCorrelated) &&
               dist::gap_selects(GapSelection::ConnectedZero, ZeroClass::Silent),
           "connected_zero selects every class");
    expect(dist::gap_selects(GapSelection::Silent, ZeroClass::Silent) &&
               !dist::gap_selects(GapSelection::Silent, ZeroClass::CoherentSubthreshold),
           "a class filter selects only its class");
    expect(dist::parse_gap_selection("silent") == GapSelection::Silent &&
               dist::parse_gap_selection("connected_zero") == GapSelection::ConnectedZero,
           "class names parse");
    bool threw = false;
    try
    {
        (void)dist::parse_gap_selection("quiet");
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    expect(threw, "an unknown class name is refused");
}

// ---------------------------------------------------------------------------
// 5: pair RDMs against the marginals of the triple RDM
// ---------------------------------------------------------------------------

void test_marginals_reproduce_pairs()
{
    std::mt19937 rng(77);
    const std::vector<Complex> psi = random_even_state(N, rng);
    double worst = 0.0;
    double ordinary_worst = 0.0;
    for (int a = 0; a < N; ++a)
    {
        for (int b = a + 1; b < N; ++b)
        {
            for (int c = b + 1; c < N; ++c)
            {
                const std::vector<double> triple = reference_rdm(psi, N, {a, b, c}, true);
                const auto rho = mipt::dist::normalized_small_rdm(triple.data(), 8);
                const std::array<std::array<int, 2>, 3> pairs{{{a, b}, {a, c}, {b, c}}};
                const std::array<unsigned, 3> masks{3u, 5u, 6u};
                for (std::size_t e = 0; e < 3; ++e)
                {
                    const std::vector<double> pair =
                        reference_rdm(psi, N, {pairs[e][0], pairs[e][1]}, true);
                    const auto want = mipt::dist::detail::normalized_hermitian_rho(pair.data());
                    worst = std::max(worst, gap::max_difference(
                                                mipt::ancilla::partial_trace_fixed(rho, 3, masks[e], true),
                                                want));
                    // The same trace done with ordinary signs, for the (a,c)
                    // marginal where b sits between the kept modes.
                    if (e == 1)
                    {
                        ordinary_worst = std::max(
                            ordinary_worst,
                            gap::max_difference(
                                mipt::ancilla::partial_trace_fixed(rho, 3, masks[e], false), want));
                    }
                }
            }
        }
    }
    expect(worst < 1.0e-13, "test 5: every fermionic triple marginal reproduces its pair RDM (" +
                                std::to_string(worst) + ")");
    expect(ordinary_worst > 1.0e-3,
           "test 5 has teeth: an ordinary trace over the middle mode does not reproduce the "
           "fermionic pair");
}

// ---------------------------------------------------------------------------
// 6, 7: cut negativities
// ---------------------------------------------------------------------------

// The Shapourian-Shiozaki-Ryu fermionic partial transpose, built independently:
// reorder so the cut mode comes first, then apply
//   |nA,nB><mA,mB| -> i^{(tA+tA') mod 2} (-1)^{(tA+tA')(tB+tB')} |mA,nB><nA,mB|.
double ssr_cut(const std::vector<Complex> &rho, int s)
{
    const std::array<int, 2> rest = s == 0 ? std::array<int, 2>{1, 2}
                                           : (s == 1 ? std::array<int, 2>{0, 2}
                                                     : std::array<int, 2>{0, 1});
    auto reorder = [&](int x, int &sign) {
        const int ns = (x >> s) & 1;
        int below = 0;
        for (int q = 0; q < s; ++q)
        {
            below += (x >> q) & 1;
        }
        sign = (ns * below) % 2 ? -1 : 1;
        return ns | (((x >> rest[0]) & 1) << 1) | (((x >> rest[1]) & 1) << 2);
    };
    std::vector<Complex> r(64);
    for (int x = 0; x < 8; ++x)
    {
        for (int z = 0; z < 8; ++z)
        {
            int sx = 1;
            int sz = 1;
            const int y = reorder(x, sx);
            const int w = reorder(z, sz);
            r[static_cast<std::size_t>(y * 8 + w)] =
                static_cast<double>(sx * sz) * rho[static_cast<std::size_t>(x * 8 + z)];
        }
    }
    std::array<Complex, 64> pt{};
    for (int row = 0; row < 8; ++row)
    {
        for (int col = 0; col < 8; ++col)
        {
            const int na = row & 1;
            const int ma = col & 1;
            const int tb = __builtin_popcount(static_cast<unsigned>(row >> 1)) +
                           __builtin_popcount(static_cast<unsigned>(col >> 1));
            const int ta = na + ma;
            Complex phase = (ta % 2) ? Complex(0.0, 1.0) : Complex(1.0, 0.0);
            if ((ta * tb) % 2)
            {
                phase = -phase;
            }
            pt[static_cast<std::size_t>(((row & ~1) | ma) * 8 + ((col & ~1) | na))] +=
                phase * r[static_cast<std::size_t>(row * 8 + col)];
        }
    }
    double norm = 0.0;
    mipt::util::gram_trace_norm<8>(pt, norm);
    return 0.5 * (norm - 1.0);
}

std::vector<Complex> random_parity_rho3(std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Complex> rho(64);
    const std::array<std::array<int, 4>, 2> sectors{{{0, 3, 5, 6}, {1, 2, 4, 7}}};
    for (int term = 0; term < 3; ++term)
    {
        std::array<Complex, 8> v{};
        for (int index : sectors[static_cast<std::size_t>(term % 2)])
        {
            v[static_cast<std::size_t>(index)] = Complex(normal(rng), normal(rng));
        }
        for (std::size_t i = 0; i < 8; ++i)
            for (std::size_t j = 0; j < 8; ++j)
                rho[i * 8 + j] += v[i] * std::conj(v[j]);
    }
    double trace = 0.0;
    for (std::size_t i = 0; i < 8; ++i)
        trace += rho[i * 8 + i].real();
    for (auto &x : rho)
        x /= trace;
    return rho;
}

std::array<double, 128> packed(const std::vector<Complex> &rho)
{
    std::array<double, 128> out{};
    for (std::size_t e = 0; e < 64; ++e)
    {
        out[2 * e] = rho[e].real();
        out[2 * e + 1] = rho[e].imag();
    }
    return out;
}

void test_cut_negativities()
{
    std::mt19937 rng(4242);
    double worst_vs_ssr = 0.0;
    double worst_relabel = 0.0;
    const std::array<std::array<int, 3>, 6> permutations{
        {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
    for (int trial = 0; trial < 20; ++trial)
    {
        const std::vector<Complex> rho = random_parity_rho3(rng);
        const auto base = packed(rho);
        std::array<double, 3> cuts{};
        mipt::analysis::cut_negativities<3>(base.data(), true, cuts);
        for (int s = 0; s < 3; ++s)
        {
            worst_vs_ssr = std::max(worst_vs_ssr, std::abs(cuts[static_cast<std::size_t>(s)] - ssr_cut(rho, s)));
        }
        for (const auto &to : permutations)
        {
            std::vector<Complex> moved(64);
            auto relabel = [&](int x, int &sign) {
                int y = 0;
                std::vector<int> targets;
                for (int s = 0; s < 3; ++s)
                {
                    if ((x >> s) & 1)
                    {
                        y |= 1 << to[static_cast<std::size_t>(s)];
                        targets.push_back(to[static_cast<std::size_t>(s)]);
                    }
                }
                int inversions = 0;
                for (std::size_t a = 0; a < targets.size(); ++a)
                    for (std::size_t b = a + 1; b < targets.size(); ++b)
                        inversions += targets[a] > targets[b] ? 1 : 0;
                sign = inversions % 2 ? -1 : 1;
                return y;
            };
            for (int r = 0; r < 8; ++r)
            {
                for (int c = 0; c < 8; ++c)
                {
                    int sr = 1;
                    int sc = 1;
                    const int pr = relabel(r, sr);
                    const int pc = relabel(c, sc);
                    moved[static_cast<std::size_t>(pr * 8 + pc)] =
                        static_cast<double>(sr * sc) * rho[static_cast<std::size_t>(r * 8 + c)];
                }
            }
            const auto moved_packed = packed(moved);
            std::array<double, 3> moved_cuts{};
            mipt::analysis::cut_negativities<3>(moved_packed.data(), true, moved_cuts);
            for (int s = 0; s < 3; ++s)
            {
                worst_relabel = std::max(
                    worst_relabel,
                    std::abs(moved_cuts[static_cast<std::size_t>(to[static_cast<std::size_t>(s)])] -
                             cuts[static_cast<std::size_t>(s)]));
            }
        }
    }
    expect(worst_vs_ssr < 1.0e-12,
           "test 6: every cut matches the Shapourian-Shiozaki-Ryu definition (" +
               std::to_string(worst_vs_ssr) + ")");
    expect(worst_relabel < 1.0e-12,
           "test 6: cut negativities follow a relabelling of the sites (" +
               std::to_string(worst_relabel) + ")");

    // 7: known states. |000> is a product; |000>+|110> is a Bell pair on modes
    // 1,2 with mode 0 uncoupled (biseparable); (|011>+|101>+|110>)/sqrt3 is a
    // genuinely tripartite, parity-definite state with every cut positive.
    auto pure = [](const std::array<Complex, 8> &v) {
        std::vector<Complex> rho(64);
        for (std::size_t i = 0; i < 8; ++i)
            for (std::size_t j = 0; j < 8; ++j)
                rho[i * 8 + j] = v[i] * std::conj(v[j]);
        return rho;
    };
    const double h = 1.0 / std::sqrt(2.0);
    const double t = 1.0 / std::sqrt(3.0);
    std::array<double, 3> cuts{};
    auto product = packed(pure({1.0, 0, 0, 0, 0, 0, 0, 0}));
    mipt::analysis::cut_negativities<3>(product.data(), true, cuts);
    expect(*std::max_element(cuts.begin(), cuts.end()) < 1.0e-12, "test 7: a product has no cut");
    auto bisep = packed(pure({h, 0, 0, 0, 0, 0, h, 0}));
    mipt::analysis::cut_negativities<3>(bisep.data(), true, cuts);
    expect(std::abs(cuts[0]) < 1.0e-12 && std::abs(cuts[1] - 0.5) < 1.0e-10 &&
               std::abs(cuts[2] - 0.5) < 1.0e-10,
           "test 7: the biseparable state's uncoupled cut vanishes and the Bell cuts are 1/2");
    auto tripartite = packed(pure({0, 0, 0, t, 0, t, t, 0}));
    mipt::analysis::cut_negativities<3>(tripartite.data(), true, cuts);
    const double w_cut = 0.5 * ((t + std::sqrt(2.0 / 3.0)) * (t + std::sqrt(2.0 / 3.0)) - 1.0);
    for (double cut : cuts)
    {
        expect(std::abs(cut - w_cut) < 1.0e-10,
               "test 7: every cut of the tripartite state is the W-state value");
    }
}

// ---------------------------------------------------------------------------
// 8: the prefilter decision
// ---------------------------------------------------------------------------

void test_prefilter_decision(const std::string &directory)
{
    const dist::RunConfig config = test_config(directory);
    const auto trajectory = make_trajectory(0, config);
    // Triple (0, 3, 4): mode 0 sits in a product with (3, 4), so the cut on it
    // vanishes and the triple is prefiltered.
    const std::vector<double> product_cut = reference_rdm(trajectory->psi, N, {0, 3, 4}, true);
    const gap::TripleResult bounded = gap::evaluate_triple(
        product_cut.data(), {0, 3, 4}, gap::triple_rank(0, 3, 4), trajectory->measurement,
        config.pair_gap);
    expect(bounded.method == gap::FgmnMethod::PrefilterBound,
           "test 8: a vanishing minimum cut is prefiltered");
    expect(bounded.fgmn == 0.0 && !bounded.positive && bounded.fgmn_raw != bounded.fgmn_raw,
           "test 8: a prefiltered triple enters averages as 0 with no raw solver value");
    expect(bounded.min_cut <= config.pair_gap.prefilter_tol,
           "test 8: the recorded bound is the minimum cut");
    expect(bounded.marginal_residual[0] < 1.0e-13 && bounded.marginal_residual[1] < 1.0e-13 &&
               bounded.marginal_residual[2] < 1.0e-13,
           "the evaluated triple's marginal residuals are at machine precision");
    expect(bounded.trace_error < 1.0e-13 && bounded.hermiticity_error < 1.0e-13,
           "a first-principles RDM has no trace or Hermiticity error");
    expect(bounded.min_eigenvalue > -1.0e-12, "and no negative eigenvalue");

    // Triple (0, 1, 2) lies inside the entangled half: every cut is positive.
    const std::vector<double> entangled = reference_rdm(trajectory->psi, N, {0, 1, 2}, true);
    const gap::TripleResult solved = gap::evaluate_triple(
        entangled.data(), {0, 1, 2}, gap::triple_rank(0, 1, 2), trajectory->measurement,
        config.pair_gap);
    expect(solved.method == gap::FgmnMethod::Mosek, "test 8: a positive minimum cut is solved");

    // The boundary itself: a minimum cut exactly at the tolerance is prefiltered.
    dist::PairGapSettings at = config.pair_gap;
    at.prefilter_tol = solved.min_cut;
    expect(gap::evaluate_triple(entangled.data(), {0, 1, 2}, 0, trajectory->measurement, at).method ==
               gap::FgmnMethod::PrefilterBound,
           "test 8: a minimum cut equal to the tolerance is prefiltered");
    at.prefilter_tol = std::nextafter(solved.min_cut, 0.0);
    expect(gap::evaluate_triple(entangled.data(), {0, 1, 2}, 0, trajectory->measurement, at).method ==
               gap::FgmnMethod::Mosek,
           "test 8: one ulp above the tolerance is solved");
}

// ---------------------------------------------------------------------------
// Certified classification (request 11.1 and 11.2)
// ---------------------------------------------------------------------------

void test_certified_classification(const std::string &directory)
{
    using gap::CertifiedClass;
    const double tol = 1.0e-6;

    // 11.1: the three ways an interval can sit relative to the threshold, plus
    // the failure that is none of them.
    expect(gap::classify_certified(2.0e-6, 3.0e-6, tol, true) == CertifiedClass::Positive,
           "11.1: an interval entirely above the threshold is positive");
    expect(gap::classify_certified(1.0e-9, 9.0e-7, tol, true) ==
               CertifiedClass::BoundedBelowThreshold,
           "11.1: an interval entirely below the threshold is bounded-small");
    expect(gap::classify_certified(1.0e-9, 3.0e-6, tol, true) == CertifiedClass::Unresolved,
           "11.1: an interval straddling the threshold is unresolved, not negative");
    expect(gap::classify_certified(2.0e-6, 3.0e-6, tol, false) == CertifiedClass::Failed,
           "11.1: a failed solve is failed whatever its numbers say");

    // The boundary is strict on both sides, so no value is both.
    expect(gap::classify_certified(tol, 2.0 * tol, tol, true) == CertifiedClass::Unresolved,
           "11.1: a lower bound exactly at the threshold does not certify positivity");
    expect(gap::classify_certified(0.0, tol, tol, true) == CertifiedClass::Unresolved,
           "11.1: an upper bound exactly at the threshold does not certify smallness");

    // A missing bound is not a bound. This is the case that matters when MOSEK
    // returns an Optimal primal with no usable dual: without a ceiling the
    // state cannot be called small, however tiny the point value is.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    expect(gap::classify_certified(1.0e-30, nan, tol, true) == CertifiedClass::Unresolved,
           "11.1: a tiny value with no upper bound is unresolved, not bounded-small");
    expect(gap::classify_certified(nan, 1.0e-30, tol, true) ==
               CertifiedClass::BoundedBelowThreshold,
           "11.1: a ceiling below the threshold certifies smallness with no lower bound");

    // 11.2: the prefilter records a bound, and is classified from that bound
    // rather than being written down as an exact zero.
    const dist::RunConfig config = test_config(directory);
    const auto trajectory = make_trajectory(0, config);
    const std::vector<double> product_cut = reference_rdm(trajectory->psi, N, {0, 3, 4}, true);
    dist::PairGapSettings settings = config.pair_gap;
    const gap::TripleResult bounded = gap::evaluate_triple(
        product_cut.data(), {0, 3, 4}, gap::triple_rank(0, 3, 4), trajectory->measurement,
        settings);
    expect(bounded.method == gap::FgmnMethod::PrefilterBound, "11.2: the fixture is prefiltered");
    expect(bounded.lower_bound == 0.0 &&
               bounded.upper_bound == bounded.min_cut + bounded.input_uncertainty,
           "11.2: a prefiltered triple is certified to [0, min_cut + the matrix's own error]");
    expect(bounded.input_uncertainty < 1e-12,
           "11.2: and a first-principles RDM carries essentially no such error");
    expect(bounded.fgmn_raw != bounded.fgmn_raw,
           "11.2: and carries no raw solver value, because none was computed");
    expect(bounded.certified == CertifiedClass::BoundedBelowThreshold,
           "11.2: at the default threshold the prefilter bound certifies smallness");
    expect(!bounded.positive, "11.2: a prefiltered triple is never positive");

    // The honest consequence: below the prefilter tolerance the same bound
    // certifies nothing, and the triple becomes unresolved rather than
    // silently negative. This is exactly what a threshold sweep has to see.
    settings.positive_tol = bounded.min_cut;
    const gap::TripleResult at_bound = gap::evaluate_triple(
        product_cut.data(), {0, 3, 4}, 0, trajectory->measurement, settings);
    expect(at_bound.certified == CertifiedClass::Unresolved,
           "11.2: a positivity threshold at the recorded bound leaves the triple unresolved");
    expect(at_bound.fgmn == 0.0,
           "11.2: it still enters averages at 0 -- the class, not the value, carries the caveat");

    // A solved triple: the min-cut bound tightens MOSEK's ceiling when it is
    // the better of the two, since fGMN <= min_s N_s holds regardless.
    const std::vector<double> entangled = reference_rdm(trajectory->psi, N, {0, 1, 2}, true);
    gap::TripleResult solved = gap::evaluate_triple(entangled.data(), {0, 1, 2}, 0,
                                                    trajectory->measurement, config.pair_gap);
    gap::SolveJob job;
    job.attempts = 1;
    // A solver that certifies its own answer: the interval collapses to the
    // point, which then sits above the threshold.
    job.outcome = gap::SolveOutcome::exact(0.5 * solved.min_cut);
    gap::apply_solve(solved, job, config.pair_gap.positive_tol);
    expect(std::abs(solved.lower_bound - 0.5 * solved.min_cut) <= solved.input_uncertainty &&
               std::abs(solved.upper_bound - 0.5 * solved.min_cut) <= solved.input_uncertainty,
           "11.2: a zero-uncertainty solve gives an interval as tight as the matrix allows");
    expect(solved.certified == CertifiedClass::Positive,
           "11.2: and a point above the threshold certifies positivity");

    // The exact cut bound caps the ceiling whenever it is the tighter one.
    gap::TripleResult capped = gap::evaluate_triple(entangled.data(), {0, 1, 2}, 0,
                                                    trajectory->measurement, config.pair_gap);
    gap::SolveJob loose;
    loose.attempts = 1;
    loose.outcome = gap::SolveOutcome::exact(0.5 * capped.min_cut);
    loose.outcome.upper_bound = 10.0 * capped.min_cut;
    gap::apply_solve(capped, loose, config.pair_gap.positive_tol);
    expect(capped.upper_bound == capped.min_cut,
           "11.2: fGMN <= min_s N_s caps the ceiling whatever the solver claims");

    // The case that actually bites on real data. At GMN_MOSEK_TOL=1e-5 MOSEK's
    // primal and dual objectives cross -- the "upper" bound lands *below* the
    // lower one, often below zero. Taking the minimum of the two ceilings there
    // would invent a tight bound out of a failure to converge and certify the
    // triple small. The crossed dual ceiling must be discarded, leaving the
    // exact cut bound, which for a solved triple is above the prefilter
    // tolerance and therefore resolves nothing.
    gap::TripleResult crossed = gap::evaluate_triple(entangled.data(), {0, 1, 2}, 0,
                                                     trajectory->measurement, config.pair_gap);
    gap::SolveJob bad;
    bad.attempts = 1;
    bad.outcome = gap::SolveOutcome::exact(0.0);
    bad.outcome.upper_bound = -1.4e-4; // the measured median on real RPPU triples
    bad.outcome.solver_gap = -1.4e-4;
    gap::apply_solve(crossed, bad, config.pair_gap.positive_tol);
    expect(crossed.solver_gap < 0.0,
           "11.2: the negative duality gap is reported rather than hidden");
    expect(crossed.lower_bound == 0.0 && std::abs(crossed.upper_bound - 1.4e-4) < 1e-12,
           "11.2: a crossed bracket becomes an uncertainty, so the interval is [0, |gap|]");
    expect(crossed.certified == CertifiedClass::Unresolved,
           "11.2: which at a 1e-10 threshold resolves nothing");
    // The same crossed bracket against a threshold wider than the crossing
    // *does* resolve, which is the case the audit's re-solve depends on: a
    // near-zero fGMN whose uncertainty is far below the threshold is certified
    // small rather than left unresolved forever.
    gap::TripleResult wide = crossed;
    gap::apply_solve(wide, bad, 1.0e-2);
    expect(wide.certified == CertifiedClass::BoundedBelowThreshold,
           "11.2: and a threshold above the crossing certifies smallness");

    // A primal value above the exact cut ceiling. fGMN <= min_s N_s is a
    // theorem, so this is proof the solver's point is not feasible enough to
    // bound anything -- measured on real near-separable control triples at
    // GMN_MOSEK_TOL=1e-10 (1.8e-8 returned against a 1.3e-8 ceiling). Believing
    // it would certify tripartite entanglement in an exact product.
    gap::TripleResult impossible = gap::evaluate_triple(entangled.data(), {0, 1, 2}, 0,
                                                        trajectory->measurement, config.pair_gap);
    gap::SolveJob over;
    over.attempts = 1;
    over.outcome = gap::SolveOutcome::exact(1.5 * impossible.min_cut);
    gap::apply_solve(impossible, over, config.pair_gap.positive_tol);
    expect(impossible.lower_bound != impossible.lower_bound,
           "11.2: a solver value above the exact cut ceiling is discarded as a bound");
    expect(impossible.certified == CertifiedClass::Unresolved,
           "11.2: and the triple is unresolved, not certified positive from an empty interval");
    expect(impossible.fgmn == 1.5 * impossible.min_cut,
           "11.2: the value is still reported -- it just stops being a bound");
}

// ---------------------------------------------------------------------------
// The driver: 9, 10, 11
// ---------------------------------------------------------------------------

struct Run
{
    dist::RunConfig config;
    std::vector<dist::PairBin> bins;
    std::unique_ptr<gap::PairGapAnalysis> analysis;
};

void process_range(Run &run, std::uint32_t begin, std::uint32_t end, bool publish_each)
{
    for (std::uint32_t t = begin; t < end; ++t)
    {
        const auto trajectory = make_trajectory(t, run.config);
        run.analysis->process(t, trajectory->measurement, trajectory->connectivity,
                              reference_batch(*trajectory));
        bin_trajectory(run.bins, trajectory->measurement);
        if (publish_each)
        {
            run.analysis->publish();
        }
    }
}

std::unique_ptr<Run> open_run(const dist::RunConfig &config, std::uint64_t completed,
                              std::vector<dist::PairBin> bins, gap::FgmnSolver solver = fake_solver)
{
    auto run = std::make_unique<Run>();
    run->config = config;
    run->bins = std::move(bins);
    run->analysis = std::make_unique<gap::PairGapAnalysis>(config, run->bins, std::move(solver), 2,
                                                           completed, true);
    return run;
}

// Every row has exactly the declared columns, and rows fan out per anchor.
void check_rows(const std::string &path, std::size_t &rows_out)
{
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    const std::size_t columns = gap::outcome_columns().size();
    std::size_t rows = 0;
    while (std::getline(file, line))
    {
        const auto fields = mipt::util::resume::split_csv_row(line);
        expect(fields.size() == columns, "every row has exactly the declared column count");
        ++rows;
    }
    rows_out = rows;
}

void test_driver_resume_and_reconciliation(const std::string &base)
{
    // Uninterrupted reference over 6 trajectories.
    const std::string ref_dir = base + "/reference";
    std::filesystem::create_directories(ref_dir);
    dist::RunConfig ref_config = test_config(ref_dir);
    auto reference = open_run(ref_config, 0, dist::make_pair_bins(N));
    process_range(*reference, 0, 6, true);
    reference->analysis->finish(std::cerr);
    const std::string reference_rows = read_file(ref_config.pair_gap.output_path);
    const std::string reference_aggregate =
        read_file(gap::aggregate_path_for(ref_config.pair_gap.output_path));
    const std::string reference_rho3 = read_file(gap::rho3_path_for(ref_config.pair_gap.output_path));

    std::size_t rows = 0;
    check_rows(ref_config.pair_gap.output_path, rows);
    expect(rows == reference->analysis->rows(), "the aggregate counts every written row");
    std::uint64_t anchors = 0;
    for (int c = 0; c < dist::ZERO_CLASS_COUNT; ++c)
        for (int s = 1; s <= N / 2; ++s)
            anchors += reference->analysis->aggregate().cell(0, c, s).qualifying_pairs +
                       reference->analysis->aggregate().cell(1, c, s).qualifying_pairs;
    expect(rows == anchors * static_cast<std::uint64_t>(N - 2),
           "test 1 end to end: rows = qualifying pairs x (L-2)");
    std::uint64_t prefiltered = 0;
    std::uint64_t mosek = 0;
    for (int r = 0; r < gap::ROLE_COUNT; ++r)
        for (int c = 0; c < dist::ZERO_CLASS_COUNT; ++c)
            for (int s = 1; s <= N / 2; ++s)
            {
                prefiltered += reference->analysis->aggregate().cell(r, c, s).outcomes_prefiltered;
                mosek += reference->analysis->aggregate().cell(r, c, s).outcomes_mosek;
            }
    expect(prefiltered > 0 && mosek > 0,
           "the driver tests exercise both the prefilter and the solver path");
    expect(reference->analysis->aggregate().cell(1, 2, 1).qualifying_pairs +
                   reference->analysis->aggregate().cell(1, 2, 2).qualifying_pairs +
                   reference->analysis->aggregate().cell(1, 2, 3).qualifying_pairs >
               0,
           "the synthetic run exercises the control group");

    // 11: an aggregate rebuilt from the outcome CSV renders identically to the
    // live one.
    {
        gap::GapAggregate rebuilt;
        rebuilt.reset(N);
        gap::scan_outcomes(ref_config.pair_gap.output_path, 1000000, rebuilt,
                           ref_config.pair_gap);
        const std::string live = gap::render_aggregate_csv(
            ref_config, reference->analysis->aggregate(), reference->bins, 6);
        const std::string replayed = gap::render_aggregate_csv(ref_config, rebuilt, reference->bins, 6);
        expect(live == replayed, "test 11: the aggregate rebuilt from the rows is identical");
        expect(live == reference_aggregate, "the published aggregate is the live one");
    }

    // 10: interrupted run. Trajectories 0..3 are processed and published; the
    // pair checkpoint only made it to 3 (a kill between the two publishes);
    // trajectory 4's rows were written but never published.
    const std::string dir = base + "/interrupted";
    std::filesystem::create_directories(dir);
    dist::RunConfig config = test_config(dir);
    std::vector<dist::PairBin> checkpoint_bins;
    {
        auto run = open_run(config, 0, dist::make_pair_bins(N));
        process_range(*run, 0, 4, true);
        const auto fourth = make_trajectory(4, config);
        run->analysis->process(4, fourth->measurement, fourth->connectivity,
                               reference_batch(*fourth));
        // The pair CSV's bins as of trajectory 3.
        checkpoint_bins = dist::make_pair_bins(N);
        for (std::uint32_t t = 0; t < 3; ++t)
        {
            bin_trajectory(checkpoint_bins, make_trajectory(t, config)->measurement);
        }
        // Killed here: no finish(), no further publish.
    }
    auto resumed = open_run(config, 3, checkpoint_bins);
    expect(resumed->analysis->restored_trimmed(), "the rows past the pair checkpoint are dropped");
    process_range(*resumed, 3, 6, true);
    resumed->analysis->finish(std::cerr);
    expect(read_file(config.pair_gap.output_path) == reference_rows,
           "test 10: a resumed run's outcome rows are byte-identical to an uninterrupted run's");
    expect(read_file(gap::aggregate_path_for(config.pair_gap.output_path)) == reference_aggregate,
           "test 10: and so is its aggregate");
    expect(read_file(gap::rho3_path_for(config.pair_gap.output_path)) == reference_rho3,
           "test 10: and so is its RDM companion");

    // Refusals. The aggregate trailing the pair checkpoint means the analysis
    // did not cover every counted trajectory.
    expect(refuses([&] { (void)open_run(config, 7, checkpoint_bins); }),
           "an aggregate behind the pair checkpoint is refused");
    // Different thresholds classify differently.
    {
        dist::RunConfig other = config;
        other.pair_gap.positive_tol = 1.0e-6;
        expect(refuses([&] { (void)open_run(other, 3, checkpoint_bins); }),
               "a different fGMN positivity tolerance is refused");
        other = config;
        other.channel_floor = 1.0e-20;
        expect(refuses([&] { (void)open_run(other, 3, checkpoint_bins); }),
               "a different channel floor is refused");
    }
    // Pair bins that disagree with the rows.
    {
        std::vector<dist::PairBin> wrong = checkpoint_bins;
        wrong[0].classical_silent += 1;
        expect(refuses([&] { (void)open_run(config, 3, wrong); }),
               "rows that do not reconcile with the pair checkpoint are refused");
    }
    // A damaged outcome file: a pair with the wrong number of rows.
    {
        const std::string damaged_dir = base + "/damaged";
        std::filesystem::create_directories(damaged_dir);
        dist::RunConfig damaged = test_config(damaged_dir);
        std::filesystem::copy_file(config.pair_gap.output_path, damaged.pair_gap.output_path);
        std::filesystem::copy_file(gap::aggregate_path_for(config.pair_gap.output_path),
                                   gap::aggregate_path_for(damaged.pair_gap.output_path));
        damaged.pair_gap.store_rho3 = false;
        std::string text = read_file(damaged.pair_gap.output_path);
        const std::size_t first_row = text.find('\n') + 1;
        const std::size_t second_row = text.find('\n', first_row) + 1;
        text.erase(first_row, second_row - first_row);
        std::ofstream(damaged.pair_gap.output_path, std::ios::binary | std::ios::trunc) << text;
        std::vector<dist::PairBin> bins = dist::make_pair_bins(N);
        for (std::uint32_t t = 0; t < 6; ++t)
        {
            bin_trajectory(bins, make_trajectory(t, damaged)->measurement);
        }
        expect(refuses([&] { (void)open_run(damaged, 6, bins); }),
               "a pair with a missing outcome row is refused");
    }
}

void test_solver_failures(const std::string &base)
{
    const std::string dir = base + "/failures";
    std::filesystem::create_directories(dir);
    dist::RunConfig config = test_config(dir);
    config.pair_gap.store_rho3 = false;
    std::atomic<int> calls{0};
    // Always fails, so every solved triple must end up missing.
    auto run = open_run(config, 0, dist::make_pair_bins(N), [&calls](const double *) {
        ++calls;
        // A plausible-looking value alongside a non-OK status: the point is
        // that it is discarded rather than trusted.
        gap::SolveOutcome out = gap::SolveOutcome::exact(0.123);
        out.status = FGMN_STATUS_NOT_OPTIMAL_BASE + 1;
        return out;
    });
    process_range(*run, 0, 3, true);
    run->analysis->finish(std::cerr);
    expect(run->analysis->failed_triples() > 0, "test 9: there are failed triples");
    expect(calls.load() ==
               static_cast<int>(run->analysis->solves()) * (1 + config.pair_gap.retries),
           "test 9: every failed solve was retried the configured number of times");

    std::ifstream file(config.pair_gap.output_path);
    std::string line;
    std::getline(file, line);
    const auto &columns = gap::outcome_columns();
    auto at = [&](const char *name) {
        return static_cast<std::size_t>(std::find(columns.begin(), columns.end(), name) -
                                        columns.begin());
    };
    std::size_t failed_rows = 0;
    while (std::getline(file, line))
    {
        const auto fields = mipt::util::resume::split_csv_row(line);
        if (fields[at("fgmn_method")] != "mosek")
        {
            continue;
        }
        ++failed_rows;
        expect(fields[at("fgmn")] == "nan" && fields[at("fgmn_raw")] == "nan",
               "test 9: a failed solve is written as NaN, never as its garbage value or zero");
        expect(fields[at("fgmn_status")] == "not_optimal_unknown",
               "test 9: the solver's status is kept");
        expect(fields[at("fgmn_positive")] == "0", "test 9: a failure is never counted positive");
        expect(fields[at("fgmn_attempts")] == std::to_string(1 + config.pair_gap.retries),
               "test 9: the attempts are recorded");
    }
    expect(failed_rows > 0, "test 9 is not vacuous");

    const std::string aggregate = read_file(gap::aggregate_path_for(config.pair_gap.output_path));
    expect(aggregate.find(",0," + std::to_string(run->analysis->failed_triples()) + ",") !=
               std::string::npos,
           "test 9: the aggregate reports run_complete=0 alongside the failure count");
    std::uint64_t undetermined = 0;
    std::uint64_t positive = 0;
    for (int c = 0; c < dist::ZERO_CLASS_COUNT; ++c)
        for (int s = 1; s <= N / 2; ++s)
        {
            undetermined += run->analysis->aggregate().cell(0, c, s).pairs_undetermined;
            positive += run->analysis->aggregate().cell(0, c, s).pairs_with_positive_third;
        }
    expect(undetermined > 0 && positive == 0,
           "test 9: pairs whose solves failed are undetermined, not negative");
}

// ---------------------------------------------------------------------------
// The per-pair summary, and the offline audit that regenerates it
// ---------------------------------------------------------------------------

void test_pair_summary(const std::string &base)
{
    const std::string dir = base + "/summary";
    std::filesystem::create_directories(dir);
    dist::RunConfig config = test_config(dir);
    config.pair_gap.controls = true;
    config.realizations = 6;
    auto run = open_run(config, 0, dist::make_pair_bins(N));
    process_range(*run, 0, 6, false);
    run->analysis->finish(std::cerr);

    const std::string summary_path = run->analysis->summary_path();
    expect(summary_path == gap::pair_summary_path_for(config.pair_gap.output_path),
           "the summary sits beside the outcome CSV under the documented name");

    // One row per qualifying pair, with exactly the declared columns.
    std::ifstream file(summary_path);
    std::string line;
    std::getline(file, line);
    expect(line + "\n" == gap::pair_summary_csv_header(), "the summary header is the column list");
    const std::size_t columns = gap::pair_summary_columns().size();
    std::size_t rows = 0;
    std::map<std::string, std::size_t> by_explanation;
    std::set<std::tuple<std::string, std::string, std::string, std::string>> identities;
    std::size_t thirds_total = 0;
    auto at = [&](const std::vector<std::string> &fields, const char *name) {
        const auto &all = gap::pair_summary_columns();
        return fields[static_cast<std::size_t>(std::find(all.begin(), all.end(), name) -
                                               all.begin())];
    };
    while (std::getline(file, line))
    {
        const auto fields = mipt::util::resume::split_csv_row(line);
        expect(fields.size() == columns, "every summary row has the declared column count");
        if (fields.size() != columns)
        {
            break;
        }
        ++rows;
        ++by_explanation[at(fields, "explanatory_class")];
        // Request 10: the identity is (realization, i, j) -- and the role, so a
        // control matched to an anchor at the same sites stays distinct.
        identities.insert({at(fields, "realization_id"), at(fields, "pair_i"),
                           at(fields, "pair_j"), at(fields, "anchor_role")});
        thirds_total += static_cast<std::size_t>(std::stoul(at(fields, "thirds")));
        expect(at(fields, "thirds") == std::to_string(N - 2),
               "every pair reports N-2 possible third sites");
        const std::size_t split =
            static_cast<std::size_t>(std::stoul(at(fields, "thirds_positive"))) +
            static_cast<std::size_t>(std::stoul(at(fields, "thirds_bounded_below_threshold"))) +
            static_cast<std::size_t>(std::stoul(at(fields, "thirds_unresolved"))) +
            static_cast<std::size_t>(std::stoul(at(fields, "thirds_failed")));
        expect(split == static_cast<std::size_t>(N - 2),
               "the four certified classes partition the thirds exactly");
        const bool robust = at(fields, "robust_fgmn") == "1";
        expect(robust == (std::stoul(at(fields, "thirds_positive")) > 0),
               "robust_fgmn is exactly 'some third is certified positive'");
        expect(robust == (at(fields, "explanatory_class") == "three_site_fgmn"),
               "and it agrees with the explanatory class");
    }
    expect(rows > 0, "the summary is not empty");
    expect(identities.size() == rows, "request 10: every (realization, i, j, role) appears once");
    expect(thirds_total == rows * static_cast<std::size_t>(N - 2),
           "the summary's thirds account for every outcome row");

    // The explanatory classes are mutually exclusive by construction; this
    // checks they also cover every pair the aggregate counted.
    std::uint64_t qualifying = 0;
    std::array<std::uint64_t, gap::EXPLANATORY_CLASS_COUNT> aggregated{};
    for (int r = 0; r < gap::ROLE_COUNT; ++r)
        for (int c = 0; c < dist::ZERO_CLASS_COUNT; ++c)
            for (int s = 1; s <= N / 2; ++s)
            {
                const gap::GapCell &cell = run->analysis->aggregate().cell(r, c, s);
                qualifying += cell.qualifying_pairs;
                for (std::size_t e = 0; e < gap::EXPLANATORY_CLASS_COUNT; ++e)
                {
                    aggregated[e] += cell.pairs_by_explanation[e];
                }
            }
    expect(qualifying == rows, "the summary has one row per pair the aggregate counted");
    for (std::size_t e = 0; e < gap::EXPLANATORY_CLASS_COUNT; ++e)
    {
        const char *name = gap::explanatory_class_name(static_cast<gap::ExplanatoryClass>(e));
        const std::size_t found = by_explanation.count(name) ? by_explanation.at(name) : 0u;
        expect(aggregated[e] == found,
               std::string("the aggregate's pairs_") + name + " matches the summary rows");
    }

    // Request 10 again, and the reason the summary is rewritten rather than
    // trimmed: after a kill and resume it must be byte-identical.
    const std::string before = read_file(summary_path);
    {
        std::vector<dist::PairBin> bins = dist::make_pair_bins(N);
        for (std::uint32_t t = 0; t < 6; ++t)
        {
            const auto trajectory = make_trajectory(t, config);
            bin_trajectory(bins, trajectory->measurement);
        }
        auto resumed = open_run(config, 6, bins);
        resumed->analysis->finish(std::cerr);
    }
    expect(read_file(summary_path) == before,
           "request 10: the summary rebuilt on resume is byte-identical");

    // The audit reads the finished run and, at the thresholds the file itself
    // records, must reproduce the live classification exactly -- otherwise a
    // swept threshold would be measuring the reader, not the physics.
    {
        namespace audit = mipt::dist::gap::audit;
        const audit::OutcomeMeta meta = audit::read_outcome_meta(config.pair_gap.output_path);
        expect(meta.n == N && meta.gap_format_version == gap::GAP_FORMAT_VERSION,
               "the audit recovers the run's identity from the outcome CSV");
        audit::AuditSettings settings;
        settings.pair_tols = {meta.pair_zero_tol};
        settings.fgmn_tols = {meta.positive_tol};
        settings.cut_tol = meta.cut_positive_tol;
        audit::SweepTable table(1, 1, meta.n);
        gap::GapAggregate rebuilt;
        rebuilt.reset(meta.n);
        std::string rebuilt_summary = gap::pair_summary_csv_header();
        dist::RunConfig audit_config = config;
        const gap::RowPrefix prefix = gap::row_prefix(audit_config);
        const audit::SweepResult swept = audit::sweep_outcomes(
            config.pair_gap.output_path, settings, meta, table, rebuilt,
            [&](const gap::PairSummaryRow &row) {
                gap::append_pair_summary_row(rebuilt_summary, prefix, meta.run_id, row);
            });
        expect(swept.pairs == rows, "the audit sees every pair the run wrote");
        expect(rebuilt_summary == before,
               "the audit regenerates the pair summary byte-identically at the recorded "
               "thresholds");
        const std::string live = gap::render_aggregate_csv(
            config, run->analysis->aggregate(), run->bins, 6);
        expect(gap::render_aggregate_csv(config, rebuilt, run->bins, 6) == live,
               "and the aggregate it rebuilds is the live one");

        // Tightening the fGMN threshold can only move pairs out of `positive`,
        // never into it: the certified lower bound is fixed and the test is
        // monotone. A sweep that violated this would be reclassifying wrongly.
        audit::AuditSettings tight = settings;
        tight.fgmn_tols = {meta.positive_tol, 1.0e-3, 1.0};
        audit::SweepTable swept_table(1, 3, meta.n);
        gap::GapAggregate ignored;
        ignored.reset(meta.n);
        audit::sweep_outcomes(config.pair_gap.output_path, tight, meta, swept_table, ignored,
                              gap::PairSummarySink());
        std::array<std::uint64_t, 3> positives{};
        for (std::size_t ft = 0; ft < 3; ++ft)
            for (int r = 0; r < gap::ROLE_COUNT; ++r)
                for (int c = 0; c < dist::ZERO_CLASS_COUNT; ++c)
                    for (int s = 1; s <= meta.n / 2; ++s)
                    {
                        positives[ft] += swept_table.cell(ft == 0 ? 0 : 0, ft, r, c, s)
                                             .pairs_with_positive_third;
                    }
        expect(positives[0] >= positives[1] && positives[1] >= positives[2],
               "raising the fGMN threshold never creates a positive pair");

        // And the audit leaves its inputs alone.
        expect(read_file(config.pair_gap.output_path).size() > 0 &&
                   read_file(summary_path) == before,
               "the audit does not modify the files it reads");
    }
}

// ---------------------------------------------------------------------------
// Request 11.3, 11.4, 11.5, 11.8: named states, named answers
// ---------------------------------------------------------------------------

std::vector<double> pack_density(const std::vector<Complex> &rho)
{
    std::vector<double> out(2u * rho.size());
    for (std::size_t e = 0; e < rho.size(); ++e)
    {
        out[2 * e] = rho[e].real();
        out[2 * e + 1] = rho[e].imag();
    }
    return out;
}

std::vector<Complex> pure_density(const std::vector<Complex> &psi)
{
    const std::size_t d = psi.size();
    std::vector<Complex> rho(d * d);
    for (std::size_t r = 0; r < d; ++r)
        for (std::size_t c = 0; c < d; ++c) rho[r * d + c] = psi[r] * std::conj(psi[c]);
    return rho;
}

mipt::ancilla::SmallRdm small_of(const std::vector<Complex> &rho, int dim)
{
    mipt::ancilla::SmallRdm out(dim);
    for (int r = 0; r < dim; ++r)
        for (int c = 0; c < dim; ++c) out(r, c) = rho[static_cast<std::size_t>(r * dim + c)];
    return out;
}

void test_named_triple_states()
{
    namespace an = mipt::analysis;
    const double s2 = 1.0 / std::sqrt(2.0);
    const double s3 = 1.0 / std::sqrt(3.0);

    auto diagnostics = [](const std::vector<Complex> &rho) {
        const std::vector<double> packed = pack_density(rho);
        const auto entropies = mipt::dist::three_party_entropies(small_of(rho, 8), true);
        return an::triple_separability(packed.data(), entropies.entropy, 1e-9, 1e-9);
    };

    // 11.3 product: no correlation across any cut.
    {
        std::vector<Complex> psi(8, Complex(0, 0));
        psi[0] = 1.0;
        const auto sep = diagnostics(pure_density(psi));
        for (int s = 0; s < 3; ++s)
        {
            expect(sep.character[static_cast<std::size_t>(s)] == an::CutCharacter::Product,
                   "11.3 product: every cut is a product cut");
            expect(std::abs(sep.realignment[static_cast<std::size_t>(s)] - 1.0) < 1e-9,
                   "11.3 product: the realignment norm of a pure product is exactly 1");
        }
    }

    // 11.3 classically correlated: (|000><000| + |110><110|)/2. Zero
    // negativity across every cut, but cuts 1 and 2 carry a full bit of
    // mutual information -- the case a negativity threshold alone cannot
    // distinguish from a product, which is why the product distance exists.
    {
        std::vector<Complex> rho(64, Complex(0, 0));
        rho[0 * 8 + 0] = 0.5;
        rho[6 * 8 + 6] = 0.5;
        const auto sep = diagnostics(rho);
        expect(sep.character[0] == an::CutCharacter::Product,
               "11.3 classical: the spectator mode is a product cut");
        for (int s : {1, 2})
        {
            expect(sep.character[static_cast<std::size_t>(s)] ==
                       an::CutCharacter::ClassicallyCorrelated,
                   "11.3 classical: the correlated cuts are classical, not entangled");
            expect(std::abs(sep.mutual_information[static_cast<std::size_t>(s)] - 1.0) < 1e-9,
                   "11.3 classical: and carry exactly one bit");
            expect(sep.qubit_cut[static_cast<std::size_t>(s)] <= 1e-12,
                   "11.3 classical: with no negativity at all");
        }
    }

    // 11.3 Bell pair with a spectator: cut 0 is a product, cuts 1 and 2 are
    // entangled. Biseparable, so no genuine tripartite entanglement.
    {
        std::vector<Complex> psi(8, Complex(0, 0));
        psi[0] = s2;
        psi[6] = s2; // |110>: a Bell pair on modes 1 and 2, mode 0 idle
        const auto sep = diagnostics(pure_density(psi));
        expect(sep.character[0] == an::CutCharacter::Product,
               "11.3 Bell+spectator: the spectator's cut is a product");
        expect(sep.character[1] == an::CutCharacter::Entangled &&
                   sep.character[2] == an::CutCharacter::Entangled,
               "11.3 Bell+spectator: the other two cuts are entangled");
        expect(std::abs(sep.qubit_cut[1] - 0.5) < 1e-9,
               "11.3 Bell+spectator: at the Bell value 1/2");
    }

    // 11.3 W and GHZ: every cut entangled, at the published values.
    {
        std::vector<Complex> w(8, Complex(0, 0));
        w[1] = s3;
        w[2] = s3;
        w[4] = s3;
        const auto sep = diagnostics(pure_density(w));
        for (int s = 0; s < 3; ++s)
        {
            expect(std::abs(sep.qubit_cut[static_cast<std::size_t>(s)] - 0.4714045) < 1e-6,
                   "11.3 W: every cut is the known 0.4714");
            expect(sep.character[static_cast<std::size_t>(s)] == an::CutCharacter::Entangled,
                   "11.3 W: and is classified entangled");
        }
    }
    {
        std::vector<Complex> ghz(8, Complex(0, 0));
        ghz[0] = s2;
        ghz[7] = s2;
        const auto sep = diagnostics(pure_density(ghz));
        for (int s = 0; s < 3; ++s)
        {
            expect(std::abs(sep.qubit_cut[static_cast<std::size_t>(s)] - 0.5) < 1e-9,
                   "11.3 GHZ: every cut is 1/2");
            expect(std::abs(sep.realignment[static_cast<std::size_t>(s)] - 2.0) < 1e-9,
                   "11.3 GHZ: and its realignment norm is 2");
        }
    }

    // 11.4 A biseparable mixture entangled across every fixed cut.
    //
    //   rho = (1/3) sum_p |Bell>_p<Bell| (x) |0>_q<0|
    //
    // over the three ways to choose which mode sits out. Every term is
    // biseparable, so the mixture is a PPT mixture and its genuine tripartite
    // negativity is exactly zero -- yet every one-vs-rest cut is entangled,
    // because two of the three terms straddle any given cut.
    //
    // This is why `three_site_all_cuts_entangled_fgmn_zero` is a *candidate*
    // class and not a conclusion: a state can populate it with no genuine
    // tripartite entanglement whatsoever.
    {
        auto bell_on = [&](int a, int b) {
            std::vector<Complex> psi(8, Complex(0, 0));
            psi[0] = s2;
            psi[(1 << a) | (1 << b)] = s2;
            return pure_density(psi);
        };
        std::vector<Complex> rho(64, Complex(0, 0));
        for (const auto &pair : std::vector<std::pair<int, int>>{{0, 1}, {0, 2}, {1, 2}})
        {
            const auto term = bell_on(pair.first, pair.second);
            for (std::size_t e = 0; e < rho.size(); ++e)
            {
                rho[e] += term[e] / 3.0;
            }
        }
        const auto sep = diagnostics(rho);
        for (int s = 0; s < 3; ++s)
        {
            expect(sep.qubit_cut[static_cast<std::size_t>(s)] > 1e-6,
                   "11.4: the biseparable mixture is entangled across every fixed cut");
            expect(sep.character[static_cast<std::size_t>(s)] == an::CutCharacter::Entangled,
                   "11.4: and every cut is classified entangled");
        }
        const std::vector<double> packed = pack_density(rho);
        std::array<double, 3> fermionic{};
        mipt::analysis::cut_negativities<3>(packed.data(), true, fermionic);
        const double smallest = *std::min_element(fermionic.begin(), fermionic.end());
        expect(smallest > 1e-6,
               "11.4: so is the fermionic minimum cut, which means the prefilter cannot "
               "settle it and the solver has to");
    }
}

// 11.5: a four-mode GHZ whose pair and three-site marginals are all separable.
void test_four_mode_ghz()
{
    const double s2 = 1.0 / std::sqrt(2.0);
    std::vector<Complex> psi(16, Complex(0, 0));
    psi[0] = s2;
    psi[15] = s2;
    const std::vector<double> rho = pack_density(pure_density(psi));
    const gap::FourResult result = gap::evaluate_four(rho.data(), {0, 1, 2, 3}, 1e-9, true, 1e-10);

    for (int s = 0; s < 4; ++s)
    {
        expect(std::abs(result.single_cuts[static_cast<std::size_t>(s)] - 0.5) < 1e-9,
               "11.5: every one-versus-three cut of a four-mode GHZ is 1/2");
    }
    for (int d = 0; d < 3; ++d)
    {
        expect(std::abs(result.double_cuts[static_cast<std::size_t>(d)] - 0.5) < 1e-9,
               "11.5: and so is every two-versus-two cut");
    }
    expect(result.max_pair_marginal_fn <= 1e-12,
           "11.5: while every pair marginal is separable");
    expect(result.max_triple_marginal_min_cut <= 1e-12,
           "11.5: and every three-mode marginal is separable too");
    expect(result.globally_entangled_locally_separable,
           "11.5: which is exactly the signature the four-site pass looks for");

    // Two Bell pairs are the discriminator: globally entangled, but the cut
    // that separates the pairs is not, and the pair marginals are.
    std::vector<Complex> bells(16, Complex(0, 0));
    bells[0b0000] = 0.5;
    bells[0b0011] = 0.5;
    bells[0b1100] = 0.5;
    bells[0b1111] = 0.5;
    const std::vector<double> paired = pack_density(pure_density(bells));
    const gap::FourResult two = gap::evaluate_four(paired.data(), {0, 1, 2, 3}, 1e-9, true, 1e-10);
    expect(!two.globally_entangled_locally_separable,
           "11.5: two Bell pairs do not carry the signature");
    expect(two.max_pair_marginal_fn > 0.4,
           "11.5: because their pair marginals are entangled");

    // 11.8: parity-respecting assisted negativity. A four-mode GHZ has no
    // endpoint entanglement at all, and no single-mode occupation measurement
    // can produce any -- but a *joint* measurement inside a parity sector
    // localizes a full Bell pair. That gap between the two families is the
    // reason they are reported separately.
    const std::size_t helpers = gap::pair_slot_of(2, 3);
    const auto &occupation = result.assisted_occupation[helpers];
    const auto &parity = result.assisted_parity[helpers];
    expect(parity.unconditional <= 1e-12,
           "11.8: the four-mode GHZ's endpoint pair is separable to begin with");
    expect(std::abs(parity.maximum - 0.5) < 1e-9,
           "11.8: a joint parity-sector measurement localizes a Bell pair");
    expect(std::abs(parity.positive_probability - 1.0) < 1e-9,
           "11.8: and does so whatever the outcome");
    expect(!occupation.evaluated,
           "11.8: the single-mode family does not apply to two helpers at once");

    // And nothing can assist a product state into entanglement.
    std::vector<Complex> product(16, Complex(0, 0));
    product[0] = 1.0;
    const auto flat = small_of(pure_density(product), 16);
    const auto assisted = mipt::analysis::assisted_negativity(
        flat, 4, 0b0011, 0b1100, mipt::analysis::parity_sector_family(), 1e-10);
    expect(assisted.maximum <= 1e-12,
           "11.8: no measurement creates entanglement in a product state");
}

// 11.9: fp64 against high precision on the same forced measurement record.
void test_high_precision_replay()
{
    namespace replay = mipt::dist::replay;
    constexpr int n = 8;
    std::vector<mipt::RppuLayer> layers;
    std::mt19937 layer_rng(20260913u);
    mipt::build_rppu_layers(layers, n, 6, 0.25, true, layer_rng, false);

    // Pass 1 samples the record; pass 2 forces it.
    replay::MeasurementRecord record;
    std::mt19937 measure_rng(99u);
    const auto fp64 = replay::run_trajectory<double>(layers, n, nullptr, record, measure_rng);
    replay::MeasurementRecord unused;
    std::mt19937 idle(1u);
    const auto high = replay::run_trajectory<long double>(layers, n, &record, unused, idle);

    expect(!record.empty(), "11.9: the circuit measured something to force");

    // Both states are normalized and confined to the even parity sector, which
    // is the invariant that says the replay simulated an RPPU trajectory
    // rather than something that merely ran.
    auto norm_and_parity = [&](const auto &state) {
        long double norm = 0.0L;
        long double odd = 0.0L;
        const auto &amps = state.amplitudes();
        for (std::size_t x = 0; x < amps.size(); ++x)
        {
            const long double weight = static_cast<long double>(std::norm(amps[x]));
            norm += weight;
            if (__builtin_popcountll(static_cast<unsigned long long>(x)) & 1)
            {
                odd += weight;
            }
        }
        return std::pair<long double, long double>(norm, odd);
    };
    const auto a = norm_and_parity(fp64);
    const auto b = norm_and_parity(high);
    expect(std::abs(a.first - 1.0L) < 1e-12L && std::abs(b.first - 1.0L) < 1e-15L,
           "11.9: both replays stay normalized");
    expect(a.second < 1e-24L && b.second < 1e-30L,
           "11.9: and both stay in the even parity sector, exactly");

    // The same forced record means the same trajectory, so the two precisions
    // must agree to the weaker of them. If they did not, the comparison would
    // be measuring a diverged trajectory rather than roundoff.
    bool nonzero_somewhere = false;
    double worst = 0.0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            const std::vector<int> pair{i, j};
            const auto low = replay::pair_observables(
                replay::reduce_modes<double>(fp64.amplitudes(), n, pair, true));
            const auto hp = replay::pair_observables(
                replay::reduce_modes<long double>(high.amplitudes(), n, pair, true));
            const long double scale = std::max(std::fabs(low.fn), std::fabs(hp.fn));
            if (scale > 1e-6L)
            {
                nonzero_somewhere = true;
                worst = std::max(worst, static_cast<double>(std::fabs(low.fn - hp.fn) / scale));
            }
            // A structural zero is zero at both precisions, not merely small.
            if (low.g2 == 0.0L)
            {
                expect(hp.g2 == 0.0L,
                       "11.9: a structurally zero channel is zero at every precision");
            }
        }
    }
    expect(nonzero_somewhere, "11.9 is not vacuous: some pair carries negativity");
    expect(worst < 1e-12,
           "11.9: on the same forced record the two precisions agree to double's own floor");

    // The verdict vocabulary, on the values it is meant to describe.
    expect(std::string(replay::replay_verdict(0.0L, 0.0L)) == "structural_zero",
           "11.9: exact zeros at both precisions are structural");
    expect(std::string(replay::replay_verdict(0.25L, 0.25L + 1e-17L)) == "converged",
           "11.9: agreement far below the double floor is convergence");
    expect(std::string(replay::replay_verdict(1e-16L, 3e-16L)) == "roundoff",
           "11.9: a floor-sized value that moves by its own size is roundoff");
    expect(std::string(replay::replay_verdict(0.25L, 0.5L)) == "diverged",
           "11.9: a large value that moves is not roundoff -- the comparison is broken");

    // A record from a different circuit is refused rather than replayed.
    replay::MeasurementRecord truncated(record.begin(), record.begin() + 1);
    bool refused = false;
    try
    {
        replay::MeasurementRecord ignored;
        std::mt19937 rng(2u);
        replay::run_trajectory<long double>(layers, n, &truncated, ignored, rng);
    }
    catch (const std::runtime_error &)
    {
        refused = true;
    }
    expect(refused, "11.9: a measurement record that does not fit the circuit is refused");
}

} // namespace

int main()
{
    const std::string directory = scratch();
    test_plan_counts_and_dedup(directory);
    test_controls(directory);
    test_class_boundaries();
    test_marginals_reproduce_pairs();
    test_cut_negativities();
    test_prefilter_decision(directory);
    test_certified_classification(directory);
    test_driver_resume_and_reconciliation(directory);
    test_solver_failures(directory);
    test_pair_summary(directory);
    test_named_triple_states();
    test_four_mode_ghz();
    test_high_precision_replay();

    if (failures != 0)
    {
        std::cerr << "dist_pair_gap_tests: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "dist_pair_gap_tests: PASS\n";
    return 0;
}
