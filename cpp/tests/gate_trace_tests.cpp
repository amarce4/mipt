// Host checks for the stage-2 gate-trace replay (gate_trace.hpp,
// dist_record_reader.hpp). No CUDA-Q runtime, no GPU, no MOSEK: everything the
// replay decides on the host is pinned here against closed forms or against a
// first-principles host simulation of the production gate list.
//
//   (a) a maximally mixed pair is a fixed point of every unitary
//   (b) a gate with odd-block mixing entangles |10>; a diagonal gate cannot
//   (c) an occupation measurement on one mode removes the gate's correlation,
//       and projecting the RDM equals measuring the full state and reducing
//   (d) reduce(U psi) == U reduce(psi) U^dag on full host states, for adjacent
//       bonds and for the periodic bond (0, n-1) in both of its
//       implementations -- the FSWAP network and the Jordan-Wigner string --
//       driven through the production layer draw and logical projection; plus
//       the closed-form reducer against dist_replay.hpp's reduce_modes
//   (e) the record reader round-trips RecordSink output, finds a realization by
//       binary search, and check_realization accepts an exact replay and
//       rejects a perturbed value or flag
//   (f) the host parity-sector expansion reproduces a directly built state
//   plus the selection-CSV reader.

#include "mipt/circuits/rppu_layer.hpp"
#include "mipt/dist_record_reader.hpp"
#include "mipt/dist_replay.hpp"
#include "mipt/gate_trace.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using mipt::gate_trace::Complex;
using mipt::gate_trace::Rho4;
namespace gt = mipt::gate_trace;
namespace records = mipt::dist::records;

int failures = 0;

void expect(bool condition, const std::string &what)
{
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

void expect_close(double a, double b, double tol, const std::string &what)
{
    if (!(std::abs(a - b) <= tol))
    {
        ++failures;
        std::cerr << "FAIL: " << what << ": " << a << " vs " << b << " (tol " << tol << ")\n";
    }
}

// ----------------------------------------------------------- host simulation

using State = std::vector<Complex>;

State random_even_state(int n, std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    State psi(std::size_t{1} << n);
    double norm = 0.0;
    for (std::size_t index = 0; index < psi.size(); ++index)
    {
        if (__builtin_popcountll(index) % 2 == 0)
        {
            psi[index] = Complex(normal(rng), normal(rng));
            norm += std::norm(psi[index]);
        }
    }
    for (Complex &z : psi)
    {
        z /= std::sqrt(norm);
    }
    return psi;
}

// A 4x4 on qubits (q0, q1), bit 0 of the matrix index = q0 -- the engine's
// convention for every bond.
void apply_two(State &psi, int q0, int q1, const mipt::GateMatrix4 &u)
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

// FSWAP = SWAP * CZ, exactly cusv::fswap_matrix() (gates.hpp), restated here
// so the reference does not share a definition with what it checks.
mipt::GateMatrix4 fswap()
{
    mipt::GateMatrix4 m{};
    m[0 * 4 + 0] = 1.0;
    m[2 * 4 + 1] = 1.0;
    m[1 * 4 + 2] = 1.0;
    m[3 * 4 + 3] = -1.0;
    return m;
}

// The Jordan-Wigner CZ string the kernel brackets a kind-2 bond with: a sign
// wherever `hi` is occupied and an odd number of the modes strictly between
// are.
void apply_jw_string(State &psi, int lo, int hi)
{
    std::size_t mask = 0;
    for (int k = lo + 1; k < hi; ++k)
    {
        mask |= std::size_t{1} << k;
    }
    for (std::size_t index = 0; index < psi.size(); ++index)
    {
        if (((index >> hi) & 1u) && (__builtin_popcountll(index & mask) & 1))
        {
            psi[index] = -psi[index];
        }
    }
}

// One production RPPU layer's unitaries, gate by gate, in list order.
void apply_layer(State &psi, const mipt::RppuLayer &layer)
{
    for (const mipt::RppuBondGate &gate : layer.gates)
    {
        if (gate.kind == 1)
        {
            apply_two(psi, gate.q0, gate.q1, fswap());
            continue;
        }
        const int lo = std::min(gate.q0, gate.q1);
        const int hi = std::max(gate.q0, gate.q1);
        if (gate.kind == 2)
        {
            apply_jw_string(psi, lo, hi);
        }
        apply_two(psi, gate.q0, gate.q1, mipt::rppu_gate_matrix(gate));
        if (gate.kind == 2)
        {
            apply_jw_string(psi, lo, hi);
        }
    }
}

// Projective Z measurement of one site with a chosen outcome, renormalized.
void measure_site(State &psi, int site, int outcome)
{
    double weight = 0.0;
    for (std::size_t index = 0; index < psi.size(); ++index)
    {
        if (static_cast<int>((index >> site) & 1u) != outcome)
        {
            psi[index] = 0.0;
        }
        else
        {
            weight += std::norm(psi[index]);
        }
    }
    for (Complex &z : psi)
    {
        z /= std::sqrt(weight);
    }
}

double rho_distance(const Rho4 &a, const Rho4 &b) { return gt::max_abs_difference(a, b); }

Rho4 pure_rho(int index)
{
    Rho4 rho{};
    rho[static_cast<std::size_t>(index * 5)] = 1.0;
    return rho;
}

mipt::RppuBondGate diagonal_gate(std::mt19937 &rng)
{
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    mipt::RppuBondGate gate;
    gate.kind = 0;
    gate.q0 = 0;
    gate.q1 = 1;
    gate.even = {angle(rng), 0.0, angle(rng), angle(rng)};
    gate.odd = {angle(rng), 0.0, angle(rng), angle(rng)};
    return gate; // entanglement left NaN, so rppu_gate_entanglement recomputes it
}

// ------------------------------------------------------------------- tests

void test_maximally_mixed(std::mt19937 &rng)
{
    Rho4 mixed{};
    for (int d = 0; d < 4; ++d)
    {
        mixed[static_cast<std::size_t>(d * 5)] = 0.25;
    }
    mipt::GateMatrix4 cnot{};
    cnot[0 * 4 + 0] = 1.0;
    cnot[3 * 4 + 1] = 1.0; // control bit 0, target bit 1: |01> <-> |11> in n_a + 2 n_b
    cnot[2 * 4 + 2] = 1.0;
    cnot[1 * 4 + 3] = 1.0;
    const mipt::RppuBondGate random = mipt::random_parity_preserving_gate(0, 1, rng);
    for (const auto &u : {cnot, mipt::rppu_gate_matrix(random)})
    {
        const Rho4 after = gt::conjugate(u, mixed);
        expect(gt::frobenius_distance(after, mixed) < 1e-15, "(a) I/4 is invariant");
        const gt::PairState state = gt::pair_state(after);
        expect(state.fmn < 1e-30 && state.fmi < 1e-12, "(a) I/4 stays uncorrelated (to rounding)");
        expect_close(state.purity, 0.25, 1e-15, "(a) purity of I/4");
        expect(state.distance_from_I4 < 1e-15, "(a) distance from I/4");
    }
}

void test_product_through_gate(std::mt19937 &rng)
{
    const Rho4 product = pure_rho(1); // |n_a = 1, n_b = 0>
    const gt::PairState before = gt::pair_state(product);
    expect(before.fmn == 0.0 && before.fmi < 1e-12, "(b) |10> is a product");
    expect_close(before.distance_from_I4, std::sqrt(0.75), 1e-15, "(b) ||P - I/4||_F");

    const mipt::RppuBondGate gate = mipt::random_parity_preserving_gate(0, 1, rng);
    const mipt::GateEntanglement metrics = mipt::rppu_gate_entanglement(gate);
    expect(metrics.odd_block_mixing > 1e-6, "(b) a drawn gate mixes the odd block");
    const Rho4 after = gt::conjugate(mipt::rppu_gate_matrix(gate), product);
    const gt::PairState entangled = gt::pair_state(after);
    expect(entangled.fmn > 1e-6, "(b) odd-block mixing entangles |10>");
    expect(entangled.fg2 > 1e-6 && entangled.ff2 < 1e-28, "(b) the coherence is hopping, not pairing");
    expect_close(entangled.purity, 1.0, 1e-13, "(b) a unitary keeps the pair pure");
    expect(gt::frobenius_distance(after, product) > 1e-6, "(b) rho_change is nonzero");

    const mipt::RppuBondGate diagonal = diagonal_gate(rng);
    const mipt::GateEntanglement diagonal_metrics = mipt::rppu_gate_entanglement(diagonal);
    expect(diagonal_metrics.odd_block_mixing < 1e-15 && diagonal_metrics.even_block_mixing < 1e-15,
           "(b) a diagonal gate has no block mixing");
    const gt::PairState untouched =
        gt::pair_state(gt::conjugate(mipt::rppu_gate_matrix(diagonal), product));
    expect(untouched.fmn == 0.0 && untouched.fg2 < 1e-30, "(b) a diagonal gate leaves fN = 0");
}

void test_measurement_removes_correlation(std::mt19937 &rng)
{
    const mipt::RppuBondGate gate = mipt::random_parity_preserving_gate(0, 1, rng);
    const Rho4 after = gt::conjugate(mipt::rppu_gate_matrix(gate), pure_rho(1));
    expect(gt::pair_state(after).fmn > 1e-6, "(c) precondition: the gate correlated the pair");
    for (int slot = 0; slot < 2; ++slot)
    {
        for (int outcome = 0; outcome < 2; ++outcome)
        {
            double weight = 0.0;
            for (int r = 0; r < 4; ++r)
            {
                weight += ((r >> slot) & 1) == outcome ? after[static_cast<std::size_t>(r * 5)].real() : 0.0;
            }
            if (weight < 1e-9)
            {
                continue;
            }
            const gt::PairState measured = gt::pair_state(gt::measure_occupation(after, slot, outcome));
            expect(measured.fmn == 0.0 && measured.fg2 < 1e-28 && measured.fmi < 1e-10,
                   "(c) measuring one mode removes the correlation");
        }
    }

    // Projecting the RDM is the same as measuring the full state and reducing.
    const int n = 6;
    State psi = random_even_state(n, rng);
    const int a = 1;
    const int b = 4;
    const Rho4 rho = gt::reduce_pair(psi, n, a, b);
    measure_site(psi, b, 1);
    expect(rho_distance(gt::reduce_pair(psi, n, a, b), gt::measure_occupation(rho, 1, 1)) < 1e-14,
           "(c) RDM projection == full-state measurement");
}

void test_covariance(std::mt19937 &rng)
{
    const int n = 8;
    // The closed-form reducer against the permutation-count reference, both
    // orders of the two modes.
    {
        const State psi = random_even_state(n, rng);
        for (const auto &[a, b] : std::vector<std::pair<int, int>>{{0, 1}, {2, 5}, {0, 7}, {6, 3}, {7, 0}})
        {
            const auto reference = mipt::dist::replay::reduce_modes<double>(psi, n, {a, b}, true);
            Rho4 expected{};
            for (std::size_t k = 0; k < 16; ++k)
            {
                expected[k] = reference[k];
            }
            expect(rho_distance(gt::reduce_pair(psi, n, a, b), expected) < 1e-15,
                   "(d) reduce_pair == replay::reduce_modes for (" + std::to_string(a) + "," +
                       std::to_string(b) + ")");
        }
    }

    using Mode = mipt::RppuBoundaryMode;
    for (const Mode mode : {Mode::FermionicFSwap, Mode::FermionicJWString})
    {
        for (const bool odd : {false, true})
        {
            mipt::RppuLayer layer;
            mipt::fill_rppu_layer(layer, n, odd, 0.0, true, rng, mode);
            std::vector<int> mode_at_site(static_cast<std::size_t>(n));
            for (int site = 0; site < n; ++site)
            {
                mode_at_site[static_cast<std::size_t>(site)] = site;
            }
            const mipt::LogicalLayer logical = mipt::logical_layer_from_rppu(layer, n, mode_at_site);
            for (int site = 0; site < n; ++site)
            {
                expect(mode_at_site[static_cast<std::size_t>(site)] == site,
                       "(d) the mode permutation is the identity at the layer boundary");
            }
            const State before = random_even_state(n, rng);
            State after = before;
            apply_layer(after, layer);
            const std::string label = std::string(mode == Mode::FermionicFSwap ? "fswap" : "jw") +
                                      (odd ? " odd" : " even");
            bool saw_boundary = false;
            for (const mipt::LogicalBond &bond : logical.bonds)
            {
                const mipt::RppuBondGate &gate =
                    layer.gates[static_cast<std::size_t>(bond.source_gate_index)];
                const Rho4 expected =
                    gt::conjugate(mipt::rppu_gate_matrix(gate), gt::reduce_pair(before, n, bond.a, bond.b));
                const double error = rho_distance(gt::reduce_pair(after, n, bond.a, bond.b), expected);
                const bool boundary = (bond.a == 0 && bond.b == n - 1) || (bond.a == n - 1 && bond.b == 0);
                saw_boundary = saw_boundary || boundary;
                expect(error < 1e-14, "(d) " + label + " bond (" + std::to_string(bond.a) + "," +
                                          std::to_string(bond.b) + ") covariance, error " +
                                          std::to_string(error));
            }
            expect(saw_boundary == odd, "(d) " + label + " has the periodic bond iff odd");
        }
    }
}

std::vector<std::array<int, 4>> pair_embeddings(int n)
{
    std::vector<std::array<int, 4>> out;
    std::vector<int> seen(static_cast<std::size_t>(n / 2), 0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            const int bin = mipt::util::periodic_distance(i, j, n) - 1;
            out.push_back({bin, seen[static_cast<std::size_t>(bin)]++, i, j});
        }
    }
    return out;
}

void test_record_reader(std::mt19937 &rng)
{
    const int n = 6;
    mipt::dist::RunConfig config;
    config.k = 2;
    config.n = n;
    config.periods = 4;
    config.p = 0.25;
    config.realizations = 5;
    config.type = mipt::CircuitType::FermionRPPU;
    config.gate_path_diagnostics = true;
    config.seed = 123456789ull;
    config.statevector_precision = 64;
    const std::vector<std::string> names = records::pair_observable_names(config);
    const auto bins = mipt::dist::make_pair_bins(n);
    const auto embeddings = pair_embeddings(n);
    const std::string header = records::pair_header_text(config, bins, embeddings);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("gate_trace_tests_" + std::to_string(rng()) + ".bin");
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    // realization -> per-embedding values and flags, as written
    std::map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::vector<double>>>> written;
    {
        records::RecordSink sink;
        sink.open(path.string(), header, names.size(), 0);
        for (const std::uint32_t realization : {0u, 1u, 3u, 4u})
        {
            for (const auto &row : embeddings)
            {
                std::vector<double> values(names.size());
                for (double &v : values)
                {
                    v = uniform(rng);
                }
                values.back() = std::numeric_limits<double>::quiet_NaN();
                const std::uint32_t flags = static_cast<std::uint32_t>(rng() & 0x7fu);
                sink.add(realization, static_cast<std::uint32_t>(row[0]),
                         static_cast<std::uint32_t>(row[1]), flags, values.data());
                written[realization].push_back({flags, values});
            }
        }
        sink.close();
    }

    records::RecordReader reader(path.string());
    expect(reader.version() == records::FORMAT_VERSION, "(e) version");
    expect(reader.observables() == names, "(e) observables round-trip");
    expect(reader.field("k") == "2" && reader.field("N") == std::to_string(n), "(e) k and N");
    expect(reader.field("master_seed") == "123456789", "(e) master seed");
    expect(std::stod(reader.field("p")) == 0.25, "(e) p round-trips exactly");
    expect(reader.field("statevector_precision") == "64", "(e) precision");
    expect(reader.field("connectivity_graph") == mipt::dist::CONNECTIVITY_GRAPH_DEFINITION,
           "(e) a value containing '=' and ',' survives");
    expect(reader.record_count() == 4u * embeddings.size(), "(e) record count");
    const auto sites = reader.embedding_sites();
    expect(sites.size() == embeddings.size(), "(e) embedding table size");
    for (const auto &row : embeddings)
    {
        const auto it = sites.find({row[0], row[1]});
        expect(it != sites.end() && it->second == std::vector<int>{row[2], row[3]},
               "(e) embedding table maps to the pair");
    }
    for (const std::uint32_t realization : {0u, 1u, 2u, 3u, 4u, 9u})
    {
        const auto stored = reader.read_realization(realization);
        const auto it = written.find(realization);
        const std::size_t expected = it == written.end() ? 0u : it->second.size();
        expect(stored.size() == expected, "(e) realization " + std::to_string(realization) + " size");
        for (std::size_t k = 0; k < std::min(stored.size(), expected); ++k)
        {
            const auto &[flags, values] = it->second[k];
            expect(stored[k].realization == realization && stored[k].flags == flags,
                   "(e) identifiers and flags");
            expect(stored[k].geometry == embeddings[k][0] && stored[k].embedding == embeddings[k][1],
                   "(e) geometry/embedding ids");
            for (std::size_t s = 0; s < values.size(); ++s)
            {
                const double a = stored[k].values[s];
                const double b = values[s];
                expect((std::isnan(a) && std::isnan(b)) || a == b, "(e) values bit for bit");
            }
        }
    }

    // check_realization: an exact replay is valid, a perturbed one is not.
    const auto stored = reader.read_realization(3);
    std::map<std::pair<int, int>, gt::ReplayedRecord> replayed;
    for (const auto &record : stored)
    {
        const auto &pair = sites.at({record.geometry, record.embedding});
        replayed[{pair[0], pair[1]}] = {record.flags, record.values};
    }
    const gt::RealizationCheck exact = gt::check_realization(names, stored, sites, replayed);
    expect(exact.valid() && exact.records_compared == embeddings.size() && exact.max_abs_error == 0.0,
           "(e) an exact replay validates");
    const int fmn_slot = reader.observable_index("fmn");
    expect(fmn_slot >= 0, "(e) fmn is an observable");
    {
        auto bad = replayed;
        bad.begin()->second.values[static_cast<std::size_t>(fmn_slot)] += 1e-9;
        const gt::RealizationCheck check = gt::check_realization(names, stored, sites, bad);
        expect(!check.valid() && check.mismatches == 1 && check.max_fmn_error > 0.0,
               "(e) an fN off by 1e-9 is a mismatch");
    }
    {
        auto tiny = replayed;
        tiny.begin()->second.values[static_cast<std::size_t>(fmn_slot)] += 1e-15;
        expect(gt::check_realization(names, stored, sites, tiny).valid(),
               "(e) an fN off by 1e-15 is within tolerance");
    }
    {
        auto bad = replayed;
        bad.begin()->second.flags ^= mipt::dist::FLAG_CONNECTED;
        expect(gt::check_realization(names, stored, sites, bad).flag_mismatches == 1,
               "(e) a flipped connectivity flag is a mismatch");
    }
    {
        auto bad = replayed;
        bad.begin()->second.values.back() = 0.0; // stored NaN, replayed zero
        expect(!gt::check_realization(names, stored, sites, bad).valid(),
               "(e) NaN does not agree with zero");
    }
    {
        auto missing = replayed;
        missing.erase(missing.begin());
        expect(!gt::check_realization(names, stored, sites, missing).valid(),
               "(e) a pair the replay did not produce is a mismatch");
    }
    std::filesystem::remove(path);
}

void test_replayed_observables()
{
    mipt::dist::RunConfig config;
    config.k = 2;
    config.n = 6;
    config.type = mipt::CircuitType::FermionRPPU;
    config.gate_path_diagnostics = true;
    config.record_detail = mipt::dist::RecordDetail::Full;
    const std::vector<std::string> names = records::pair_observable_names(config);

    mipt::dist::PairMetrics metrics;
    metrics.mi = 1.0;
    metrics.mn = 2.0;
    metrics.n_i = 3.0;
    metrics.n_j = 4.0;
    metrics.dnn = 5.0;
    metrics.g2 = 6.0;
    metrics.f2 = 7.0;
    mipt::dist::PairMetrics fermion;
    fermion.mi = 11.0;
    fermion.mn = 12.0;
    fermion.g2 = 16.0;
    fermion.f2 = 17.0;
    fermion.g = Complex(18.0, 19.0);
    fermion.f = Complex(20.0, 21.0);
    mipt::dist::PairConnectivity conn;
    conn.evaluated = true;
    conn.connected = false;
    conn.component_size = 42;
    mipt::dist::GatePathMetrics gate;
    gate.available = true;
    gate.widest_entangling_power = 0.5;
    const std::vector<double> values = gt::replayed_observables(names, metrics, fermion, conn, gate);
    expect(values.size() == names.size(), "replayed_observables covers every name");
    auto value_of = [&](const std::string &name) {
        const auto it = std::find(names.begin(), names.end(), name);
        return it == names.end() ? -1.0 : values[static_cast<std::size_t>(it - names.begin())];
    };
    expect(value_of("mi") == 1.0 && value_of("fmn") == 12.0 && value_of("ff2") == 17.0 &&
               value_of("fre_g") == 18.0 && value_of("fim_f") == 21.0,
           "replayed_observables maps detail values by name");
    expect(value_of("component_size") == 42.0, "component_size always written");
    expect(std::isnan(value_of("widest_gate_ep")), "a disconnected pair has a NaN widest path");

    bool refused = false;
    try
    {
        gt::replayed_observables({"mystery"}, metrics, fermion, conn, gate);
    }
    catch (const std::exception &)
    {
        refused = true;
    }
    expect(refused, "an unknown observable is refused");
}

void test_parity_expansion(std::mt19937 &rng)
{
    for (const int n : {4, 7, 10})
    {
        const State psi = random_even_state(n, rng);
        const std::size_t half = std::size_t{1} << (n - 1);
        State encoded(half);
        for (std::size_t e = 0; e < half; ++e)
        {
            const std::size_t derived = (__builtin_popcountll(e) & 1) ? half : 0u;
            encoded[e] = psi[e | derived];
        }
        const State full = gt::expand_parity_sector(encoded, n);
        bool exact = full.size() == psi.size();
        for (std::size_t k = 0; exact && k < psi.size(); ++k)
        {
            exact = full[k] == psi[k];
        }
        expect(exact, "(f) parity expansion reproduces the state exactly at n=" + std::to_string(n));
    }
}

void test_selection_reader()
{
    std::istringstream good(std::string(gt::SELECTION_HEADER) +
                            "\n7,0,5,silent,4,0,0,0.25,9,120,0.3\n"
                            "12,2,3,entangled_control,0,2,0.1,0.5,nan,80,\n");
    const auto rows = gt::read_selection(good, "inline");
    expect(rows.size() == 2 && rows[0].realization == 7 && rows[0].j == 5 &&
               rows[0].group == "silent" && rows[0].component_size == 120.0,
           "selection reader parses rows");
    expect(std::isnan(rows[1].shortest_path) && std::isnan(rows[1].widest_gate_ep),
           "selection reader takes nan and empty as NaN");

    std::istringstream bad(std::string(gt::SELECTION_HEADER) + "\n7,0,5,silnet,4,0,0,0,9,120,0.3\n");
    bool refused = false;
    try
    {
        gt::read_selection(bad, "inline");
    }
    catch (const std::exception &)
    {
        refused = true;
    }
    expect(refused, "selection reader refuses an unknown group");
    expect(gt::default_trace_path("a/b.c/sel.csv") == "a/b.c/sel_gate_trace.csv" &&
               gt::validation_path_for("a/b.c/sel_gate_trace.csv") ==
                   "a/b.c/sel_gate_trace_validation.csv",
           "default output paths");
}

} // namespace

int main()
{
    std::mt19937 rng(20261015u);
    test_maximally_mixed(rng);
    test_product_through_gate(rng);
    test_measurement_removes_correlation(rng);
    test_covariance(rng);
    test_record_reader(rng);
    test_replayed_observables();
    test_parity_expansion(rng);
    test_selection_reader();
    if (failures != 0)
    {
        std::cerr << failures << " gate-trace check(s) failed.\n";
        return 1;
    }
    std::cout << "gate_trace_tests: all checks passed.\n";
    return 0;
}
