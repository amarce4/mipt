#include "mipt/dist_metrics.hpp"
#include "mipt/dist_scaling_resume.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace
{
using C = std::complex<double>;
using Rho = std::array<C, 16>;

std::array<double, 32> interleaved(const Rho &rho)
{
    std::array<double, 32> out{};
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        out[2 * i] = rho[i].real();
        out[2 * i + 1] = rho[i].imag();
    }
    return out;
}

Rho pure_density(const std::array<C, 4> &psi)
{
    Rho rho{};
    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            rho[i * 4 + j] = psi[i] * std::conj(psi[j]);
        }
    }
    return rho;
}

Rho fermionic_swap(const Rho &rho)
{
    constexpr std::array<int, 4> permutation{0, 2, 1, 3};
    constexpr std::array<int, 4> sign{1, 1, 1, -1};
    Rho out{};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            out[permutation[i] * 4 + permutation[j]] =
                static_cast<double>(sign[i] * sign[j]) * rho[i * 4 + j];
        }
    }
    return out;
}

void require_close(double actual, double expected, double tolerance, const char *label)
{
    if (std::abs(actual - expected) > tolerance)
    {
        std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
        std::exit(1);
    }
}

// --- three-party helpers ---------------------------------------------------

using Rho3 = std::array<C, 64>;

std::array<double, 128> interleaved3(const Rho3 &rho)
{
    std::array<double, 128> out{};
    for (std::size_t i = 0; i < rho.size(); ++i)
    {
        out[2 * i] = rho[i].real();
        out[2 * i + 1] = rho[i].imag();
    }
    return out;
}

Rho3 pure_density3(const std::array<C, 8> &psi)
{
    Rho3 rho{};
    for (std::size_t i = 0; i < 8; ++i)
    {
        for (std::size_t j = 0; j < 8; ++j)
        {
            rho[i * 8 + j] = psi[i] * std::conj(psi[j]);
        }
    }
    return rho;
}

Rho3 mix3(const Rho3 &lhs, const Rho3 &rhs, double weight)
{
    Rho3 out{};
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        out[i] = weight * lhs[i] + (1.0 - weight) * rhs[i];
    }
    return out;
}

// Exchange fermionic modes 0 and 1 of a three-mode block: swap bits 0 and 1 of
// every basis index and pick up the anticommutation sign when both are
// occupied. Every measure computed with the fermionic trace has to be blind to
// this, exactly as the two-mode version above.
Rho3 fermionic_swap3(const Rho3 &rho)
{
    std::array<int, 8> permutation{};
    std::array<int, 8> sign{};
    for (int index = 0; index < 8; ++index)
    {
        const int bit0 = index & 1;
        const int bit1 = (index >> 1) & 1;
        permutation[index] = (index & ~3) | (bit1 << 0) | (bit0 << 1);
        sign[index] = (bit0 && bit1) ? -1 : 1;
    }

    Rho3 out{};
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            out[permutation[i] * 8 + permutation[j]] =
                static_cast<double>(sign[i] * sign[j]) * rho[i * 8 + j];
        }
    }
    return out;
}
} // namespace

// --- dist_scaling.exe resume ------------------------------------------------
//
// The resume path rests on one property: the CSV is an invertible projection of
// the aggregation bins. Everything below checks that property directly, by
// rendering bins, reading them back, and continuing the accumulation -- which
// is exactly what an interrupted run does.

namespace dist_resume_tests
{
namespace dist = mipt::dist;
namespace res = mipt::dist::resume;

void fail(const std::string &message)
{
    std::cerr << "dist resume: " << message << '\n';
    std::exit(1);
}

void expect(bool condition, const std::string &message)
{
    if (!condition)
    {
        fail(message);
    }
}

// Two accumulators must agree to within the round trip's few ulp, never to the
// statistical error -- the point is that nothing is lost, not that it is close.
void expect_stats_equal(
    const mipt::util::RunningStats &found,
    const mipt::util::RunningStats &expected,
    const std::string &what)
{
    expect(found.count == expected.count,
           what + ": count " + std::to_string(found.count) + " != " +
               std::to_string(expected.count));
    const double mean_tolerance = 1.0e-12 * std::max(1.0, std::abs(expected.mean));
    expect(std::abs(found.mean - expected.mean) <= mean_tolerance,
           what + ": mean " + std::to_string(found.mean) + " != " +
               std::to_string(expected.mean));
    const double spread_tolerance =
        1.0e-9 * std::max(1.0e-300, expected.sample_stddev());
    expect(std::abs(found.sample_stddev() - expected.sample_stddev()) <= spread_tolerance,
           what + ": stddev " + std::to_string(found.sample_stddev()) + " != " +
               std::to_string(expected.sample_stddev()));
}

std::string scratch_directory()
{
    const auto path = std::filesystem::temp_directory_path() / "mipt_dist_resume_tests";
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path.string();
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

// k=2: fill every bin with pseudo-random records, checkpoint, and continue.
void test_pair_round_trip(const std::string &directory)
{
    dist::RunConfig config;
    config.k = 2;
    config.n = 10;
    config.periods = 8;
    config.p = 0.17;
    config.realizations = 6;
    config.type = mipt::CircuitType::FermionRPPU;  // exercises the fermionic columns
    config.output_path = directory + "/pairs.csv";

    auto bins = dist::make_pair_bins(config.n);
    auto reference = bins;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    const auto add_trajectory = [&](std::vector<dist::PairBin> &target) {
        for (auto &bin : target)
        {
            for (std::uint64_t e = 0; e < bin.embedding_count; ++e)
            {
                const double mi = unit(rng);
                const double mn = unit(rng);
                const double fmi = unit(rng);
                const double fmn = unit(rng);
                bin.mi.add(mi);
                bin.mn.add(mn);
                if (mn > 0.5)
                {
                    ++bin.mn_positive;
                }
                bin.fmi.add(fmi);
                bin.fmn.add(fmn);
                if (fmn > 0.5)
                {
                    ++bin.fmn_positive;
                }
            }
        }
    };

    // Three trajectories, then a checkpoint.
    const int before_break = 3;
    for (int t = 0; t < before_break; ++t)
    {
        add_trajectory(bins);
    }
    dist::publish_csv(config.output_path, dist::render_pair_csv(config, bins));

    auto restored = dist::make_pair_bins(config.n);
    const res::Report report = res::load_pairs(config, restored);
    expect(report.completed == before_break,
           "pair resume recovered " + std::to_string(report.completed) +
               " trajectories, expected " + std::to_string(before_break));
    expect(report.loaded, "pair resume should report a loaded checkpoint");
    for (std::size_t b = 0; b < bins.size(); ++b)
    {
        expect_stats_equal(restored[b].mi, bins[b].mi, "pair mi");
        expect_stats_equal(restored[b].mn, bins[b].mn, "pair mn");
        expect_stats_equal(restored[b].fmi, bins[b].fmi, "pair fmi");
        expect_stats_equal(restored[b].fmn, bins[b].fmn, "pair fmn");
        expect(restored[b].mn_positive == bins[b].mn_positive, "pair mn_positive");
        expect(restored[b].fmn_positive == bins[b].fmn_positive, "pair fmn_positive");
    }

    // Continue both the restored and an uninterrupted copy over the same draws.
    // A resumed run must end up where an uninterrupted one would have.
    reference = bins;
    std::mt19937 saved = rng;
    for (int t = before_break; t < config.realizations; ++t)
    {
        add_trajectory(reference);
    }
    rng = saved;
    for (int t = before_break; t < config.realizations; ++t)
    {
        add_trajectory(restored);
    }
    for (std::size_t b = 0; b < bins.size(); ++b)
    {
        expect_stats_equal(restored[b].mi, reference[b].mi, "pair mi after resume");
        expect_stats_equal(restored[b].mn, reference[b].mn, "pair mn after resume");
        expect_stats_equal(restored[b].fmi, reference[b].fmi, "pair fmi after resume");
        expect_stats_equal(restored[b].fmn, reference[b].fmn, "pair fmn after resume");
        expect(restored[b].mn_positive == reference[b].mn_positive,
               "pair mn_positive after resume");
    }

    // A finished run reports the full count, which is how the app knows to stop.
    dist::publish_csv(config.output_path, dist::render_pair_csv(config, restored));
    expect(res::load_pairs(config, restored).completed == config.realizations,
           "a completed pair run must report every trajectory");

    // Refusals: a different run that happens to share the output path.
    auto other = config;
    other.p = 0.3;
    expect(refuses([&]() { auto b = dist::make_pair_bins(config.n); (void)res::load_pairs(other, b); }),
           "a different p must be refused");
    other = config;
    other.realizations = 99;
    expect(refuses([&]() { auto b = dist::make_pair_bins(config.n); (void)res::load_pairs(other, b); }),
           "a different realization count must be refused");
    other = config;
    other.type = mipt::CircuitType::Haar;  // drops the fermionic columns
    expect(refuses([&]() { auto b = dist::make_pair_bins(config.n); (void)res::load_pairs(other, b); }),
           "a schema change from a different circuit type must be refused");
    other = config;
    other.n = 12;
    expect(refuses([&]() { auto b = dist::make_pair_bins(other.n); (void)res::load_pairs(other, b); }),
           "a different N must be refused");

    // A missing file is a fresh start, not an error.
    auto absent = config;
    absent.output_path = directory + "/does_not_exist.csv";
    auto fresh = dist::make_pair_bins(config.n);
    expect(res::load_pairs(absent, fresh).completed == 0,
           "a missing checkpoint must start from zero");
}

// k=3: the same, plus the SDP bookkeeping that the schedule is a function of.
void test_triple_round_trip(const std::string &directory)
{
    dist::RunConfig config;
    config.k = 3;
    config.n = 10;
    config.periods = 8;
    config.p = 0.17;
    config.realizations = 8;
    config.type = mipt::CircuitType::FermionRPPU;
    config.triangle_balance_cutoff = 0.5;
    config.output_path = directory + "/triples.csv";

    const auto geometries =
        mipt::probed::enumerate_three_probe_geometries(config.n, config.triangle_balance_cutoff);
    expect(!geometries.empty(), "N=10 must admit at least one balanced triangle");

    const long per_geometry = 1;  // the default: one embedding per geometry
    auto bins = dist::make_triple_bins(config.n, geometries);
    std::mt19937 rng(999);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // Mirrors run_triples: every record feeds the three-party measures, and a
    // scheduled subset additionally requests an SDP solve.
    const auto add_trajectory = [&](std::vector<dist::TripleBin> &target, bool leave_in_flight) {
        for (auto &bin : target)
        {
            bin.tmi.add(unit(rng) - 0.5);
            bin.average_mi.add(unit(rng));
            bin.min_bipartite_negativity.add(unit(rng));
            bin.joint_purity.add(unit(rng));
            bin.mean_single_purity.add(unit(rng));
            bin.ftmi.add(unit(rng) - 0.5);
            bin.faverage_mi.add(unit(rng));
            bin.min_fermionic_bipartite_negativity.add(unit(rng));
            bin.fjoint_purity.add(unit(rng));
            bin.fmean_single_purity.add(unit(rng));

            ++bin.gmn.requested;
            const double roll = unit(rng);
            if (roll < 0.4)
            {
                bin.gmn.value.add(0.0);
                ++bin.gmn.zero_prefiltered;
            }
            else if (roll < 0.9)
            {
                bin.gmn.value.add(roll);
            }
            else if (!leave_in_flight)
            {
                ++bin.gmn.solver_failures;
            }
            // else: requested but neither valued nor failed -- a solve still in
            // the queue when the process died.

            ++bin.fgmn.requested;
            bin.fgmn.value.add(unit(rng));
            ++bin.records;
        }
    };

    const int before_break = 5;
    for (int t = 0; t < before_break; ++t)
    {
        add_trajectory(bins, t + 1 == before_break);
    }
    dist::publish_csv(config.output_path, dist::render_triple_csv(config, bins));

    auto restored = dist::make_triple_bins(config.n, geometries);
    const res::Report report = res::load_triples(config, restored, per_geometry);
    expect(report.completed == before_break,
           "triple resume recovered " + std::to_string(report.completed) +
               " trajectories, expected " + std::to_string(before_break));

    std::uint64_t expected_reclaimed = 0;
    for (std::size_t b = 0; b < bins.size(); ++b)
    {
        expect_stats_equal(restored[b].tmi, bins[b].tmi, "triple tmi");
        expect_stats_equal(restored[b].average_mi, bins[b].average_mi, "triple average_mi");
        expect_stats_equal(restored[b].min_bipartite_negativity,
                           bins[b].min_bipartite_negativity, "triple min_bipneg");
        expect_stats_equal(restored[b].joint_purity, bins[b].joint_purity, "triple joint_purity");
        expect_stats_equal(restored[b].mean_single_purity, bins[b].mean_single_purity,
                           "triple mean_single_purity");
        expect_stats_equal(restored[b].ftmi, bins[b].ftmi, "triple ftmi");
        expect_stats_equal(restored[b].min_fermionic_bipartite_negativity,
                           bins[b].min_fermionic_bipartite_negativity, "triple min_fbipneg");
        expect_stats_equal(restored[b].gmn.value, bins[b].gmn.value, "triple gmn value");
        expect_stats_equal(restored[b].fgmn.value, bins[b].fgmn.value, "triple fgmn value");

        // The SDP schedule is a pure function of these two counters, so both
        // must come back or the sampled fraction drifts.
        expect(restored[b].records == bins[b].records, "triple record counter");
        expect(restored[b].gmn.zero_prefiltered == bins[b].gmn.zero_prefiltered,
               "triple gmn zero_prefiltered");
        expect(restored[b].gmn.solver_failures == bins[b].gmn.solver_failures,
               "triple gmn solver_failures");

        // In-flight solves: `requested` is rolled back to what actually landed,
        // so the schedule re-offers those slots instead of retiring them.
        const std::uint64_t landed =
            bins[b].gmn.value.count + bins[b].gmn.solver_failures;
        expect(restored[b].gmn.requested == landed,
               "triple gmn requested must be rolled back to what landed");
        expected_reclaimed += bins[b].gmn.requested - landed;
    }
    expect(expected_reclaimed > 0,
           "the fixture must actually leave some solves in flight");
    expect(report.reclaimed_gmn_slots == expected_reclaimed,
           "reclaimed slot count " + std::to_string(report.reclaimed_gmn_slots) +
               " != " + std::to_string(expected_reclaimed));
    expect(report.reclaimed_fgmn_slots == 0, "fgmn had no in-flight solves");

    // Refusals specific to k=3.
    // B_min=0.6 admits 3 of the 5 geometries B_min=0.5 does, so the checkpoint
    // has more rows than the run enumerates.
    auto other = config;
    other.triangle_balance_cutoff = 0.6;
    expect(refuses([&]() {
               auto g = mipt::probed::enumerate_three_probe_geometries(
                   other.n, other.triangle_balance_cutoff);
               auto b = dist::make_triple_bins(other.n, g);
               (void)res::load_triples(other, b, per_geometry);
           }),
           "a different B_min must be refused");
    // A different embeddings-per-geometry makes the recorded sample counts a
    // non-integral number of trajectories.
    expect(geometries.front().embeddings.size() > 1,
           "the fixture needs a geometry with several embeddings");
    expect(refuses([&]() {
               auto b = dist::make_triple_bins(config.n, geometries);
               (void)res::load_triples(config, b, 0);
           }),
           "a different MIPT_DIST_EMBEDDINGS_PER_GEOMETRY must be refused");

    // records_per_trajectory must mirror sample_trajectory_triangles exactly.
    expect(res::records_per_trajectory(7, 1) == 1, "per_geometry=1 takes one embedding");
    expect(res::records_per_trajectory(7, 0) == 7, "per_geometry=0 takes them all");
    expect(res::records_per_trajectory(7, 9) == 7, "an oversized cap takes them all");
    expect(res::records_per_trajectory(7, 3) == 3, "an undersized cap takes that many");
}

void run(const std::string &directory)
{
    test_pair_round_trip(directory);
    test_triple_round_trip(directory);
    std::filesystem::remove_all(directory);
}
} // namespace dist_resume_tests

void test_dist_scaling_resume()
{
    dist_resume_tests::run(dist_resume_tests::scratch_directory());
}

int main()
{
    const auto product = interleaved(pure_density({C(1.0, 0.0), {}, {}, {}}));
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::two_party_metrics(product.data(), fermionic);
        require_close(metrics.mi, 0.0, 1.0e-10, "product MI");
        require_close(metrics.mn, 0.0, 1.0e-10, "product negativity");
    }

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const Rho bell_rho = pure_density({C(inv_sqrt2, 0.0), {}, {}, C(inv_sqrt2, 0.0)});
    const auto bell = interleaved(bell_rho);
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::two_party_metrics(bell.data(), fermionic);
        require_close(metrics.mi, 2.0, 1.0e-9, "Bell MI");
        require_close(metrics.mn, 0.5, 1.0e-9, "Bell negativity");
    }

    const auto swapped = interleaved(fermionic_swap(bell_rho));
    const auto original_metrics = mipt::dist::two_party_metrics(bell.data(), true);
    const auto swapped_metrics = mipt::dist::two_party_metrics(swapped.data(), true);
    require_close(swapped_metrics.mi, original_metrics.mi, 1.0e-9, "swap-invariant fermionic MI");
    require_close(swapped_metrics.mn, original_metrics.mn, 1.0e-9, "swap-invariant fermionic negativity");

    const Rho classical = [] {
        Rho rho{};
        rho[0] = C(0.5, 0.0);
        rho[15] = C(0.5, 0.0);
        return rho;
    }();
    const auto classical_ri = interleaved(classical);
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::two_party_metrics(classical_ri.data(), fermionic);
        require_close(metrics.mi, 1.0, 1.0e-9, "classical-correlation MI");
        require_close(metrics.mn, 0.0, 1.0e-9, "classical-correlation negativity");
    }

    const Rho even = pure_density({C(std::sqrt(0.7), 0.0), {}, {},
                                   C(0.0, std::sqrt(0.3))});
    const Rho odd = pure_density({C{}, C(std::sqrt(0.2), 0.0),
                                  C(std::sqrt(0.8), 0.0), C{}});
    Rho parity_preserving_mixed{};
    for (std::size_t i = 0; i < parity_preserving_mixed.size(); ++i)
    {
        parity_preserving_mixed[i] = 0.37 * even[i] + 0.63 * odd[i];
    }
    const auto mixed_ri = interleaved(parity_preserving_mixed);
    const auto mixed_swapped_ri = interleaved(fermionic_swap(parity_preserving_mixed));
    const auto mixed_metrics = mipt::dist::two_party_metrics(mixed_ri.data(), true);
    const auto mixed_swapped_metrics =
        mipt::dist::two_party_metrics(mixed_swapped_ri.data(), true);
    require_close(mixed_swapped_metrics.mi, mixed_metrics.mi, 1.0e-9,
                  "mixed-state swap-invariant fermionic MI");
    require_close(mixed_swapped_metrics.mn, mixed_metrics.mn, 1.0e-9,
                  "mixed-state swap-invariant fermionic negativity");

    require_close(mipt::dist::two_party_geometric_chord_distance(4, 0, 1),
                  2.0 * std::sqrt(2.0) / std::acos(-1.0), 1.0e-12,
                  "N=4 nearest-neighbour chord distance");
    require_close(mipt::dist::two_party_geometric_chord_distance(4, 0, 2),
                  4.0 / std::acos(-1.0), 1.0e-12,
                  "N=4 antipodal chord distance");

    // --- three-party metrics ------------------------------------------------

    const auto product3 = interleaved3(pure_density3({C(1.0, 0.0)}));
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::three_party_metrics(product3.data(), fermionic);
        require_close(metrics.tmi, 0.0, 1.0e-9, "product TMI");
        require_close(metrics.average_mi, 0.0, 1.0e-9, "product average MI");
        require_close(metrics.joint_purity, 1.0, 1.0e-12, "product joint purity");
        require_close(metrics.mean_single_purity, 1.0, 1.0e-12, "product single purity");
    }

    // Any pure three-party state has I_3 = 0, because S_AB = S_C and so on.
    // GHZ and W therefore test the entropy bookkeeping, not the combination.
    const Rho3 ghz = pure_density3({C(inv_sqrt2, 0.0), {}, {}, {}, {}, {}, {}, C(inv_sqrt2, 0.0)});
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::three_party_metrics(interleaved3(ghz).data(), fermionic);
        require_close(metrics.tmi, 0.0, 1.0e-9, "GHZ TMI");
        require_close(metrics.average_mi, 1.0, 1.0e-9, "GHZ average MI");
        require_close(metrics.joint_purity, 1.0, 1.0e-9, "GHZ joint purity");
        require_close(metrics.mean_single_purity, 0.5, 1.0e-9, "GHZ single purity");
    }

    const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
    const Rho3 w_state =
        pure_density3({C{}, C(inv_sqrt3, 0.0), C(inv_sqrt3, 0.0), C{}, C(inv_sqrt3, 0.0), C{}, C{}, C{}});
    {
        const double single = -(1.0 / 3.0) * std::log2(1.0 / 3.0) - (2.0 / 3.0) * std::log2(2.0 / 3.0);
        const auto metrics = mipt::dist::three_party_metrics(interleaved3(w_state).data(), false);
        require_close(metrics.tmi, 0.0, 1.0e-9, "W TMI");
        require_close(metrics.average_mi, single, 1.0e-9, "W average MI");
    }

    // A classically correlated GHZ mixture is the simplest state with a
    // nonzero tripartite information: every marginal and every pair carries
    // one bit, and so does the whole block, leaving I_3 = 1 bit.
    const Rho3 classical_ghz =
        mix3(pure_density3({C(1.0, 0.0)}), pure_density3({C{}, C{}, C{}, C{}, C{}, C{}, C{}, C(1.0, 0.0)}), 0.5);
    for (bool fermionic : {false, true})
    {
        const auto metrics = mipt::dist::three_party_metrics(interleaved3(classical_ghz).data(), fermionic);
        require_close(metrics.tmi, 1.0, 1.0e-9, "classical-GHZ TMI");
        require_close(metrics.average_mi, 1.0, 1.0e-9, "classical-GHZ average MI");
        require_close(metrics.joint_purity, 0.5, 1.0e-9, "classical-GHZ joint purity");
        require_close(metrics.mean_single_purity, 0.5, 1.0e-9, "classical-GHZ single purity");
    }

    // A Bell pair on (A,B) with C idle: I_3 = 0 but the pairwise mutual
    // informations are 2, 0, 0, which is what separates `average_mi` from a
    // measure that only sees the symmetric combination.
    const Rho3 bell_ab = pure_density3({C(inv_sqrt2, 0.0), {}, {}, C(inv_sqrt2, 0.0), {}, {}, {}, {}});
    {
        const auto metrics = mipt::dist::three_party_metrics(interleaved3(bell_ab).data(), false);
        require_close(metrics.tmi, 0.0, 1.0e-9, "Bell-pair-plus-idle TMI");
        require_close(metrics.average_mi, 2.0 / 3.0, 1.0e-9, "Bell-pair-plus-idle average MI");
    }

    // Fermionic-trace metrics must be invariant under exchanging two modes
    // together with the anticommutation sign that exchange carries.
    const Rho3 parity_mixed3 = mix3(ghz, classical_ghz, 0.41);
    const auto parity_metrics = mipt::dist::three_party_metrics(interleaved3(parity_mixed3).data(), true);
    const auto parity_swapped_metrics =
        mipt::dist::three_party_metrics(interleaved3(fermionic_swap3(parity_mixed3)).data(), true);
    require_close(parity_swapped_metrics.tmi, parity_metrics.tmi, 1.0e-9,
                  "swap-invariant fermionic TMI");
    require_close(parity_swapped_metrics.average_mi, parity_metrics.average_mi, 1.0e-9,
                  "swap-invariant fermionic average MI");

    // A non-unit trace has to be divided out, not carried into the entropies.
    Rho3 unnormalized = classical_ghz;
    for (auto &value : unnormalized)
    {
        value *= 3.7;
    }
    const auto unnormalized_metrics =
        mipt::dist::three_party_metrics(interleaved3(unnormalized).data(), false);
    require_close(unnormalized_metrics.tmi, 1.0, 1.0e-9, "unnormalized-input TMI");

    // Triangle geometry: an equilateral triangle on N=6 has all three chords
    // equal, so their geometric mean is that chord.
    const double pi = std::acos(-1.0);
    require_close(mipt::dist::three_party_geometric_chord_distance(6, 0, 2, 4),
                  6.0 / pi * std::sin(2.0 * pi / 6.0), 1.0e-12,
                  "N=6 equilateral triangle chord distance");
    const auto separations = mipt::dist::triangle_separations(6, 0, 1, 3);
    if (separations != std::array<int, 3>{1, 2, 3})
    {
        std::cerr << "triangle_separations(6,0,1,3): expected {1,2,3}, got {" << separations[0] << ','
                  << separations[1] << ',' << separations[2] << "}\n";
        std::exit(1);
    }

    test_dist_scaling_resume();

    std::cout << "dist_metrics_tests: PASS\n";
    return 0;
}
