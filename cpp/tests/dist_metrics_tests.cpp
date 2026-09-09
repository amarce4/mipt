#include "mipt/dist_metrics.hpp"
#include "mipt/dist_records.hpp"
#include "mipt/dist_scaling_resume.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <utility>
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
                bin.g2.add(unit(rng));
                bin.f2.add(unit(rng));
                if (mn > 0.5)
                {
                    ++bin.mn_positive;
                }
                bin.fmi.add(fmi);
                bin.fmn.add(fmn);
                bin.fg2.add(unit(rng));
                bin.ff2.add(unit(rng));
                if (fmn > 0.5)
                {
                    ++bin.fmn_positive;
                }
                const double residual = 1.0e-16 * unit(rng);
                bin.fmn_residual.add(residual);
                bin.fmn_residual_max = std::max(bin.fmn_residual_max, residual);

                const double n_i = unit(rng);
                const double n_j = unit(rng);
                bin.n_i.add(n_i);
                bin.n_j.add(n_j);
                const double dnn = n_i * n_j;
                bin.dnn.add(dnn);
                bin.rho_n.add(dnn - n_i * n_j);
                const double i_occ = unit(rng);
                bin.i_occ.add(i_occ);

                // Drive the contingency exactly the way PairProtocol does, so
                // that the partition invariants the resume checks are the ones
                // the run actually produces rather than ones the test invents.
                const bool survives_i = unit(rng) > 0.3;
                const bool survives_j = unit(rng) > 0.3;
                const bool interior = unit(rng) > 0.25;
                const bool connected = survives_i && survives_j && interior;
                const bool entangled = fmn > 0.5;
                ++bin.conn_records;
                bin.survive_i += survives_i ? 1u : 0u;
                bin.survive_j += survives_j ? 1u : 0u;
                bin.survive_both += (survives_i && survives_j) ? 1u : 0u;
                bin.interior_path += interior ? 1u : 0u;
                bin.component_size.add(unit(rng) * 100.0);
                if (interior)
                {
                    bin.shortest_path.add(unit(rng) * 10.0);
                }
                if (connected)
                {
                    ++bin.connected;
                    bin.ent_given_connected.add(fmn);
                    if (entangled)
                    {
                        ++bin.connected_ent_positive;
                    }
                    else
                    {
                        ++bin.connected_ent_zero;
                        bin.i_occ_connected_classical.add(i_occ);
                        if (i_occ > 0.6)
                        {
                            ++bin.classical_occ_correlated;
                        }
                        else if (i_occ > 0.3)
                        {
                            ++bin.classical_subthreshold;
                        }
                        else
                        {
                            ++bin.classical_silent;
                        }
                    }
                }
                else
                {
                    bin.ent_given_disconnected.add(fmn);
                    if (entangled)
                    {
                        ++bin.disconnected_ent_positive;
                    }
                    else
                    {
                        ++bin.disconnected_ent_zero;
                    }
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
        expect_stats_equal(restored[b].g2, bins[b].g2, "pair g2");
        expect_stats_equal(restored[b].f2, bins[b].f2, "pair f2");
        expect_stats_equal(restored[b].fg2, bins[b].fg2, "pair fg2");
        expect_stats_equal(restored[b].ff2, bins[b].ff2, "pair ff2");
        expect(restored[b].mn_positive == bins[b].mn_positive, "pair mn_positive");
        expect(restored[b].fmn_positive == bins[b].fmn_positive, "pair fmn_positive");
        expect_stats_equal(restored[b].n_i, bins[b].n_i, "pair n_i");
        expect_stats_equal(restored[b].dnn, bins[b].dnn, "pair dnn");
        expect_stats_equal(restored[b].rho_n, bins[b].rho_n, "pair rho_n");
        expect_stats_equal(restored[b].i_occ, bins[b].i_occ, "pair i_occ");
        expect_stats_equal(restored[b].ent_given_connected, bins[b].ent_given_connected,
                           "pair ent_given_connected");
        expect_stats_equal(restored[b].ent_given_disconnected, bins[b].ent_given_disconnected,
                           "pair ent_given_disconnected");
        expect_stats_equal(restored[b].component_size, bins[b].component_size,
                           "pair component_size");
        expect_stats_equal(restored[b].shortest_path, bins[b].shortest_path,
                           "pair shortest_path");
        expect_stats_equal(restored[b].fmn_residual, bins[b].fmn_residual, "pair fmn_residual");
        expect(restored[b].fmn_residual_max == bins[b].fmn_residual_max,
               "pair fmn_residual_max");
        // The whole contingency table, cell by cell. These are the counts the
        // conditional probabilities are computed from, so a silent loss here
        // would show up as a plausible-looking kappa rather than as an error.
        expect(restored[b].conn_records == bins[b].conn_records, "pair conn_records");
        expect(restored[b].survive_i == bins[b].survive_i, "pair survive_i");
        expect(restored[b].survive_j == bins[b].survive_j, "pair survive_j");
        expect(restored[b].survive_both == bins[b].survive_both, "pair survive_both");
        expect(restored[b].interior_path == bins[b].interior_path, "pair interior_path");
        expect(restored[b].connected == bins[b].connected, "pair connected");
        expect(restored[b].connected_ent_positive == bins[b].connected_ent_positive,
               "pair connected_ent_positive");
        expect(restored[b].connected_ent_zero == bins[b].connected_ent_zero,
               "pair connected_ent_zero");
        expect(restored[b].disconnected_ent_positive == bins[b].disconnected_ent_positive,
               "pair disconnected_ent_positive");
        expect(restored[b].disconnected_ent_zero == bins[b].disconnected_ent_zero,
               "pair disconnected_ent_zero");
        expect(restored[b].classical_occ_correlated == bins[b].classical_occ_correlated,
               "pair classical_occ_correlated");
        expect(restored[b].classical_subthreshold == bins[b].classical_subthreshold,
               "pair classical_subthreshold");
        expect(restored[b].classical_silent == bins[b].classical_silent,
               "pair classical_silent");
        // Non-vacuity: a partition test passes trivially on all-zero counts.
        expect(restored[b].connected > 0 && restored[b].connected_ent_positive > 0 &&
                   restored[b].connected_ent_zero > 0 &&
                   restored[b].disconnected_ent_positive > 0,
               "the contingency table must actually be populated");
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
        expect_stats_equal(restored[b].g2, reference[b].g2, "pair g2 after resume");
        expect_stats_equal(restored[b].ff2, reference[b].ff2, "pair ff2 after resume");
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
    // Pooling two classification thresholds would silently mix two different
    // definitions of "entangled" in one contingency table.
    other = config;
    other.pair_zero_tol = 1.0e-10;
    expect(refuses([&]() { auto b = dist::make_pair_bins(config.n); (void)res::load_pairs(other, b); }),
           "a different pair_zero_tol must be refused");
    // require_same(double)'s relative slack is 1e-12 * max(1, |x|), which would
    // accept zero as a match for the 1e-12 default. The text comparison must
    // not.
    other = config;
    other.pair_zero_tol = 0.0;
    expect(refuses([&]() { auto b = dist::make_pair_bins(config.n); (void)res::load_pairs(other, b); }),
           "a zero pair_zero_tol must be refused against the 1e-12 default");
    other = config;
    other.statevector_precision = config.statevector_precision == 64 ? 32 : 64;
    expect(refuses([&]() { auto b = dist::make_pair_bins(config.n); (void)res::load_pairs(other, b); }),
           "a different statevector precision must be refused");
    // A hand-edited contingency table must not read back as a valid one.
    {
        auto broken = bins;
        ++broken[1].connected_ent_positive;
        dist::publish_csv(directory + "/broken.csv", dist::render_pair_csv(config, broken));
        auto broken_config = config;
        broken_config.output_path = directory + "/broken.csv";
        expect(refuses([&]() {
                   auto b = dist::make_pair_bins(config.n);
                   (void)res::load_pairs(broken_config, b);
               }),
               "a contingency table that does not partition must be refused");
    }

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
            else if (roll < 0.8)
            {
                bin.gmn.value.add(roll);
                ++bin.gmn.positive;
            }
            else if (roll < 0.9)
            {
                // Solved, but the solver returned a zero the prefilter did not
                // catch: a value with no positive count behind it, which is
                // what the restore's positive <= solved check has to allow.
                bin.gmn.value.add(0.0);
            }
            else if (!leave_in_flight)
            {
                ++bin.gmn.solver_failures;
            }
            // else: requested but neither valued nor failed -- a solve still in
            // the queue when the process died.

            ++bin.fgmn.requested;
            const double fermionic_value = unit(rng);
            bin.fgmn.value.add(fermionic_value);
            if (fermionic_value > 0.0)
            {
                ++bin.fgmn.positive;
            }
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
        expect(restored[b].gmn.positive == bins[b].gmn.positive, "triple gmn positive");
        expect(restored[b].fgmn.positive == bins[b].fgmn.positive, "triple fgmn positive");

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
    std::uint64_t total_positive = 0;
    std::uint64_t total_solved_zero = 0;
    for (const auto &bin : bins)
    {
        total_positive += bin.gmn.positive;
        total_solved_zero +=
            bin.gmn.value.count - bin.gmn.zero_prefiltered - bin.gmn.positive;
    }
    expect(total_positive > 0, "the fixture must record some positive GMN values");
    expect(total_solved_zero > 0,
           "the fixture must record some solved-but-zero GMN values, or the "
           "positive <= solved check is never exercised");
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

// ---------------------------------------------------------------------------
// The per-record binary
//
// Everything here is checked against what was written rather than against a
// second copy of the writer's own arithmetic: a record read back is compared
// field by field with the value handed to the sink, and the aggregate the
// records imply is compared with the running means the CSV would have
// reported. The three properties that are easy to get wrong and impossible to
// notice later are deferred resolution landing in the right row, the flush
// stopping at a trajectory boundary, and the resume trim cutting exactly at
// the CSV's checkpoint.
// ---------------------------------------------------------------------------
namespace dist_record_tests
{
namespace rec = mipt::dist::records;
using dist_resume_tests::expect;
using dist_resume_tests::refuses;

std::string scratch_directory()
{
    const auto path = std::filesystem::temp_directory_path() / "mipt_dist_record_tests";
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path.string();
}

struct Row
{
    std::uint32_t realization;
    std::uint16_t geometry;
    std::uint16_t embedding;
    std::uint32_t flags;
    std::vector<double> values;
};

// Read a record file back with no help from the writer: the preamble gives
// the record size and the payload offset, and the fields are pulled out by
// memcpy at fixed offsets. If the writer and this ever disagree the test
// fails, which is the point.
std::pair<std::string, std::vector<Row>> read_back(const std::string &path,
                                                   std::size_t value_count)
{
    std::ifstream file(path, std::ios::binary);
    expect(static_cast<bool>(file), "the record file opens");
    char preamble[rec::PREAMBLE_BYTES];
    file.read(preamble, static_cast<std::streamsize>(rec::PREAMBLE_BYTES));
    expect(std::memcmp(preamble, rec::MAGIC, sizeof(rec::MAGIC)) == 0,
           "the record file starts with the magic");
    std::uint32_t version = 0;
    std::uint32_t record_bytes = 0;
    std::uint32_t header_bytes = 0;
    std::memcpy(&version, preamble + 8, 4);
    std::memcpy(&record_bytes, preamble + 12, 4);
    std::memcpy(&header_bytes, preamble + 16, 4);
    expect(version == rec::FORMAT_VERSION, "the format version is the current one");
    expect(record_bytes == rec::record_bytes(value_count),
           "the declared record size matches the observable count");
    std::string header(header_bytes, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header_bytes));

    std::vector<Row> rows;
    std::vector<char> raw(record_bytes);
    while (file.read(raw.data(), static_cast<std::streamsize>(record_bytes)))
    {
        Row row{};
        std::memcpy(&row.realization, raw.data(), 4);
        std::memcpy(&row.geometry, raw.data() + 4, 2);
        std::memcpy(&row.embedding, raw.data() + 6, 2);
        std::memcpy(&row.flags, raw.data() + 8, 4);
        std::uint32_t reserved = 1;
        std::memcpy(&reserved, raw.data() + 12, 4);
        expect(reserved == 0, "the reserved alignment word is zero");
        row.values.resize(value_count);
        std::memcpy(row.values.data(), raw.data() + rec::ID_BYTES, value_count * sizeof(double));
        rows.push_back(std::move(row));
    }
    return {header, rows};
}

// A run with no deferred values: every record is complete when it is created,
// so the file is a verbatim transcript of what was handed in.
void test_immediate_round_trip(const std::string &directory)
{
    const std::string path = directory + "/pairs_records.bin";
    std::vector<Row> expected;
    {
        rec::RecordSink sink;
        sink.open(path, "format=test\nfields=x\n", 2, 0);
        expect(sink.active(), "a freshly opened sink is active");
        expect(sink.inherited_records() == 0, "a new file inherits nothing");
        for (std::uint32_t realization = 0; realization < 5; ++realization)
        {
            for (std::uint16_t geometry = 0; geometry < 3; ++geometry)
            {
                for (std::uint16_t embedding = 0; embedding < 2; ++embedding)
                {
                    const double values[2] = {0.5 * realization + geometry,
                                              -1.0 * embedding - 0.25 * geometry};
                    // A flag mask that varies per record, so a writer that
                    // dropped or transposed the field could not pass.
                    const std::uint32_t flags =
                        mipt::dist::FLAG_EVALUATED |
                        ((realization % 2) ? mipt::dist::FLAG_SURVIVES_I : 0u) |
                        ((geometry % 2) ? mipt::dist::FLAG_CONNECTED : 0u);
                    sink.add(realization, geometry, embedding, flags, values);
                    expected.push_back(
                        {realization, geometry, embedding, flags, {values[0], values[1]}});
                }
            }
        }
        sink.close();
    }

    const auto [header, rows] = read_back(path, 2);
    expect(header == "format=test\nfields=x\n", "the header text round-trips verbatim");
    expect(rows.size() == expected.size(), "every record reached the file");
    for (std::size_t index = 0; index < rows.size(); ++index)
    {
        expect(rows[index].realization == expected[index].realization &&
                   rows[index].geometry == expected[index].geometry &&
                   rows[index].embedding == expected[index].embedding,
               "record identifiers round-trip in order");
        expect(rows[index].values[0] == expected[index].values[0] &&
                   rows[index].values[1] == expected[index].values[1],
               "record observables round-trip bit for bit");
        expect(rows[index].flags == expected[index].flags,
               "record connectivity flags round-trip");
    }
}

// A GMN arriving out of order must land in the row that asked for it, and
// nothing behind an unresolved record may be written -- otherwise the file
// stops being in trajectory order and the resume trim stops being exact.
void test_deferred_resolution(const std::string &directory)
{
    const std::string path = directory + "/triples_records.bin";
    rec::RecordSink sink;
    sink.open(path, "format=test\n", 2, 0);

    // Trajectory 0: one resolved record, then one waiting on a solve.
    const double first[2] = {1.0, 0.0};
    sink.add(0, 0, 0, 0u, first);
    const double second[2] = {2.0, std::numeric_limits<double>::quiet_NaN()};
    const std::uint64_t waiting = sink.add_pending(0, 1, 0, 0u, second, 1);
    // Trajectory 1: fully resolved, but it sits behind the pending record.
    const double third[2] = {3.0, 0.0};
    sink.add(1, 0, 0, 0u, third);

    sink.flush();
    expect(sink.written() == 0,
           "nothing is written while trajectory 0 has an unresolved record -- not even the "
           "resolved record ahead of it, since the file must hold whole trajectories");

    sink.resolve(waiting, 1, 0.75);
    sink.flush();
    expect(sink.written() == 3, "resolving the last hole releases everything behind it");
    sink.close();

    const auto [header, rows] = read_back(path, 2);
    expect(rows.size() == 3, "all three records are in the file");
    expect(rows[1].geometry == 1 && rows[1].values[0] == 2.0 && rows[1].values[1] == 0.75,
           "the deferred value landed in the row that requested it");
    expect(rows[0].values[0] == 1.0 && rows[2].values[0] == 3.0,
           "the surrounding records are untouched and still in order");
}

// A record whose solve never came back keeps its NaN, and must still be
// released -- a permanently unresolved row would hold every later trajectory
// out of the file forever.
void test_failed_solve_releases_the_row(const std::string &directory)
{
    const std::string path = directory + "/failed_records.bin";
    rec::RecordSink sink;
    sink.open(path, "format=test\n", 2, 0);
    const double values[2] = {1.5, std::numeric_limits<double>::quiet_NaN()};
    const std::uint64_t waiting = sink.add_pending(0, 0, 0, 0u, values, 1);
    sink.resolve(waiting, 1, std::numeric_limits<double>::quiet_NaN());
    sink.close();

    const auto [header, rows] = read_back(path, 2);
    expect(rows.size() == 1, "the record is written once its solve is accounted for");
    expect(rows[0].values[0] == 1.5, "the observable that did arrive is kept");
    expect(std::isnan(rows[0].values[1]),
           "a solve that failed leaves NaN, which is not the same as zero");
}

// Reopening against a CSV checkpoint discards exactly the trajectories the
// checkpoint did not count, and keeps every record of the ones it did.
void test_resume_trims_to_the_checkpoint(const std::string &directory)
{
    const std::string path = directory + "/resume_records.bin";
    const std::string header_text = "format=test\nk=2\n";
    {
        rec::RecordSink sink;
        sink.open(path, header_text, 2, 0);
        for (std::uint32_t realization = 0; realization < 6; ++realization)
        {
            for (std::uint16_t geometry = 0; geometry < 4; ++geometry)
            {
                const double values[2] = {static_cast<double>(realization), geometry + 0.5};
                sink.add(realization, geometry, 0, 0u, values);
            }
        }
        sink.close();
    }

    {
        // The CSV says 4 trajectories are binned, so trajectories 4 and 5 --
        // written but never counted -- have to go, or resuming would bin them
        // a second time.
        rec::RecordSink sink;
        sink.open(path, header_text, 2, 4);
        expect(sink.inherited_records() == 16, "the four counted trajectories are kept");
        expect(sink.truncated_on_open(), "the uncounted trajectories are reported as dropped");
        expect(sink.realizations_in_file() == 4, "the file holds exactly the counted ones");
        const double values[2] = {4.0, 9.0};
        sink.add(4, 0, 0, 0u, values);
        sink.close();
    }

    const auto [header, rows] = read_back(path, 2);
    expect(header == header_text, "the header survives the trim untouched");
    expect(rows.size() == 17, "the trimmed file plus one new record");
    for (std::size_t index = 0; index < 16; ++index)
    {
        expect(rows[index].realization == static_cast<std::uint32_t>(index / 4),
               "the kept records are the first four trajectories, in order");
    }
    expect(rows[16].realization == 4 && rows[16].values[1] == 9.0,
           "the resumed run appends after them");
}

// Reopening a file that describes a different run must refuse rather than
// interleave two ensembles in one table.
void test_mismatched_header_refuses(const std::string &directory)
{
    const std::string path = directory + "/mismatch_records.bin";
    {
        rec::RecordSink sink;
        sink.open(path, "format=test\nN=12\n", 2, 0);
        const double values[2] = {1.0, 2.0};
        sink.add(0, 0, 0, 0u, values);
        sink.close();
    }
    expect(refuses([&] {
               rec::RecordSink sink;
               sink.open(path, "format=test\nN=16\n", 2, 0);
           }),
           "a record file describing a different run is refused");
    expect(refuses([&] {
               rec::RecordSink sink;
               sink.open(path, "format=test\nN=12\n", 4, 0);
           }),
           "a record file with a different observable count is refused");
}

// The header has to carry the geometry table, because that is the only place
// `d` is stored -- a record has an id, not a distance.
void test_header_carries_the_geometry_table()
{
    mipt::dist::RunConfig config;
    config.k = 2;
    config.n = 8;
    config.type = mipt::CircuitType::FermionRPPU;
    config.record_detail = mipt::dist::RecordDetail::Basic;
    const auto bins = mipt::dist::make_pair_bins(config.n);
    // (geometry, embedding) -> sites, exactly as PairProtocol builds it.
    std::vector<std::array<int, 4>> embeddings;
    for (int i = 0; i < config.n; ++i)
    {
        for (int j = i + 1; j < config.n; ++j)
        {
            const int separation = mipt::util::periodic_distance(i, j, config.n);
            embeddings.push_back({separation - 1, 0, i, j});
        }
    }
    const std::string header = rec::pair_header_text(config, bins, embeddings);

    expect(header.find("k=2\n") != std::string::npos, "the header names the party count");
    expect(header.find("N=8\n") != std::string::npos, "the header names the system size");
    expect(header.find("observables=mi,mn,fmi,fmn\n") != std::string::npos,
           "a parity-preserving ensemble declares both trace conventions");
    expect(header.find("[geometry]\ngeometry_id,separation,d,embedding_count\n") !=
               std::string::npos,
           "the geometry table maps a geometry id to its effective distance");
    expect(header.find("[embedding]\ngeometry_id,embedding_id,site_1,site_2\n") !=
               std::string::npos,
           "the embedding table maps a (geometry, embedding) to its two sites");
    // Provenance: none of the stored flags or negativities can be interpreted
    // without these, so the header has to carry them.
    expect(header.find("pair_zero_tol=") != std::string::npos,
           "the header records the positivity threshold");
    expect(header.find("connectivity_graph_version=") != std::string::npos &&
               header.find("connectivity_graph=") != std::string::npos,
           "the header records which graph convention produced the flags");
    expect(header.find("statevector_precision=") != std::string::npos,
           "the header records the arithmetic the negativities came from");
    expect(header.find("boundary_implementation=") != std::string::npos,
           "the header records how the periodic bond was implemented");
    expect(header.find("master_seed=") != std::string::npos,
           "the header records the seed the trajectories came from");
    expect(header.find("flags:u4") != std::string::npos,
           "the field list declares the flag word");
    for (const auto &bin : bins)
    {
        std::string chord;
        mipt::dist::append_double(chord, bin.chord);
        expect(header.find(chord) != std::string::npos,
               "every separation's chord length is in the table");
    }

    config.type = mipt::CircuitType::Haar;
    const std::string qubit_header = rec::pair_header_text(config, bins, embeddings);
    expect(qubit_header.find("observables=mi,mn\n") != std::string::npos,
           "a qubit ensemble declares two observables, not four");
    expect(rec::record_bytes(2) == 32 && rec::record_bytes(4) == 48,
           "a v2 record is a 16-byte identifier block plus its float64 observables");
}

// The record payload and its declared field names must agree, at every detail
// level. They are written by two functions in two files -- fill_record_values
// in dist_scaling.hpp and pair_observable_names here -- so nothing but a test
// keeps them in step, and a transposition would silently relabel a column.
void test_record_detail_levels()
{
    mipt::dist::RunConfig config;
    config.k = 2;
    config.n = 8;
    config.type = mipt::CircuitType::FermionRPPU;

    config.record_detail = mipt::dist::RecordDetail::Basic;
    const auto basic = rec::pair_observable_names(config);
    expect(basic == std::vector<std::string>({"mi", "mn", "fmi", "fmn"}),
           "basic detail is exactly the format v1 value list");

    config.record_detail = mipt::dist::RecordDetail::Channel;
    const auto channel = rec::pair_observable_names(config);
    expect(channel == std::vector<std::string>({"mi", "mn", "fmi", "fmn", "n_i", "n_j", "dnn",
                                                "g2", "f2", "fg2", "ff2"}),
           "channel detail appends the occupation data and both traces' correlators");

    config.record_detail = mipt::dist::RecordDetail::Full;
    const auto full = rec::pair_observable_names(config);
    expect(full.size() == channel.size() + 4, "full detail adds the four channel phases");
    expect(std::equal(channel.begin(), channel.end(), full.begin()),
           "the detail levels are strictly additive, so a prefix is stable");
    expect(full[full.size() - 4] == "fre_g" && full.back() == "fim_f",
           "the phase fields name the trace they came from");

    // A qubit ensemble has no fermionic partner, so it carries neither the
    // fermionic observables nor the fermionic correlators.
    config.type = mipt::CircuitType::Haar;
    config.record_detail = mipt::dist::RecordDetail::Channel;
    const auto qubit = rec::pair_observable_names(config);
    expect(qubit == std::vector<std::string>({"mi", "mn", "n_i", "n_j", "dnn", "g2", "f2"}),
           "a qubit ensemble's channel record drops every fermionic field");
}

// The stride keeps whole trajectories: that is what makes a clustered
// bootstrap over realization_id valid on a strided file.
void test_record_stride_keeps_whole_trajectories()
{
    expect(rec::stride_selects(1, 0) && rec::stride_selects(1, 7),
           "a stride of one keeps everything");
    int kept = 0;
    for (std::uint64_t realization = 0; realization < 20; ++realization)
    {
        kept += rec::stride_selects(4, realization) ? 1 : 0;
    }
    expect(kept == 5, "a stride of four keeps one trajectory in four");
    // Decisive property: the decision depends on the trajectory alone, so every
    // record of a kept trajectory is kept and no cluster is ever partial.
    for (std::uint64_t realization = 0; realization < 20; ++realization)
    {
        const bool first = rec::stride_selects(3, realization);
        expect(first == rec::stride_selects(3, realization),
               "the stride decision is a pure function of the trajectory index");
    }
}

void run(const std::string &directory)
{
    test_immediate_round_trip(directory);
    test_deferred_resolution(directory);
    test_failed_solve_releases_the_row(directory);
    test_resume_trims_to_the_checkpoint(directory);
    test_record_detail_levels();
    test_record_stride_keeps_whole_trajectories();
    test_mismatched_header_refuses(directory);
    test_header_carries_the_geometry_table();
    std::filesystem::remove_all(directory);
}
} // namespace dist_record_tests

void test_dist_records()
{
    dist_record_tests::run(dist_record_tests::scratch_directory());
}

// ---------------------------------------------------------------------------
// The two-mode channel: the closed-form fermionic negativity, the occupation
// diagnostics, and the precision behaviour that motivates both.
// ---------------------------------------------------------------------------
namespace dist_channel_tests
{
using dist_resume_tests::expect;

// A random parity-preserving two-mode state: positive semidefinite and block
// diagonal in the local fermion parity, which is what the reduced density
// matrix of a definite-global-parity state always is. Tracing out the
// environment can only connect subsystem states of equal parity, so the
// cross-parity entries are zero exactly rather than approximately.
Rho random_parity_preserving(std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    Rho rho{};
    // {0, 3} is the even block, {1, 2} the odd one.
    const std::array<std::array<int, 2>, 2> blocks{{{0, 3}, {1, 2}}};
    for (const auto &block : blocks)
    {
        std::array<C, 4> a{};
        for (C &value : a)
        {
            value = C(normal(rng), normal(rng));
        }
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                C sum(0.0, 0.0);
                for (int k = 0; k < 2; ++k)
                {
                    sum += a[i * 2 + k] * std::conj(a[j * 2 + k]);
                }
                rho[block[i] * 4 + block[j]] = sum;
            }
        }
    }
    double trace = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        trace += rho[i * 4 + i].real();
    }
    for (C &value : rho)
    {
        value /= trace;
    }
    return rho;
}

// Round an interleaved RDM through float, so a host test can ask what an fp32
// state vector's reduction would have looked like.
std::array<double, 32> through_fp32(const std::array<double, 32> &values)
{
    std::array<double, 32> out{};
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        out[i] = static_cast<double>(static_cast<float>(values[i]));
    }
    return out;
}

// Closed forms on states whose fermionic negativity is known by hand.
void test_analytic_states()
{
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    // (|01> + |10>)/sqrt(2): one fermion hopping between the modes. All the
    // weight is in the odd parity sector and |G|^2 = 1/4, so the closed form
    // reads 2(1/4)/(sqrt(0 + 1) + 0) = 1/2.
    const auto hop = interleaved(pure_density(std::array<C, 4>{C(0.0, 0.0), C(inv_sqrt2, 0.0), C(inv_sqrt2, 0.0), C(0.0, 0.0)}));
    const auto hop_metrics = mipt::dist::two_party_metrics(hop.data(), true, true);
    require_close(hop_metrics.g2, 0.25, 1.0e-12, "hopping |G|^2");
    require_close(hop_metrics.f2, 0.0, 1.0e-12, "hopping |F|^2");
    require_close(hop_metrics.parity_even, 0.0, 1.0e-12, "hopping even-parity weight");
    require_close(hop_metrics.parity_odd, 1.0, 1.0e-12, "hopping odd-parity weight");
    require_close(hop_metrics.mn_closed, 0.5, 1.0e-12, "hopping closed-form fN");
    require_close(hop_metrics.mn, 0.5, 1.0e-12, "hopping reported fN");

    // (|00> + |11>)/sqrt(2): the pairing channel instead. |F|^2 = 1/4 and all
    // the weight is even, so the other term of the sum carries it.
    const auto pair = interleaved(pure_density(std::array<C, 4>{C(inv_sqrt2, 0.0), C(0.0, 0.0), C(0.0, 0.0), C(inv_sqrt2, 0.0)}));
    const auto pair_metrics = mipt::dist::two_party_metrics(pair.data(), true, true);
    require_close(pair_metrics.f2, 0.25, 1.0e-12, "paired |F|^2");
    require_close(pair_metrics.g2, 0.0, 1.0e-12, "paired |G|^2");
    require_close(pair_metrics.mn_closed, 0.5, 1.0e-12, "paired closed-form fN");

    // A classical mixture of the same two occupations: no coherence at all, so
    // both channels vanish and so does the negativity -- while the occupation
    // mutual information is a full bit, because the two modes are perfectly
    // correlated. This is exactly the "connected and classically correlated but
    // not entangled" cell of the contingency table.
    Rho classical{};
    classical[0] = 0.5;
    classical[3 * 4 + 3] = 0.5;
    const auto classical_ri = interleaved(classical);
    const auto classical_metrics = mipt::dist::two_party_metrics(classical_ri.data(), true, true);
    require_close(classical_metrics.mn_closed, 0.0, 1.0e-14, "classical closed-form fN");
    require_close(classical_metrics.i_occ, 1.0, 1.0e-12, "classical occupation MI");
    require_close(classical_metrics.rho_n, 0.25, 1.0e-12, "classical density correlation");
    require_close(classical_metrics.n_i, 0.5, 1.0e-12, "classical <n_i>");
    require_close(classical_metrics.dnn, 0.5, 1.0e-12, "classical D");

    // A product state: no correlation of any kind.
    Rho product{};
    for (int i = 0; i < 4; ++i)
    {
        product[i * 4 + i] = 0.25;
    }
    const auto product_ri = interleaved(product);
    const auto product_metrics = mipt::dist::two_party_metrics(product_ri.data(), true, true);
    require_close(product_metrics.i_occ, 0.0, 1.0e-12, "product occupation MI");
    require_close(product_metrics.rho_n, 0.0, 1.0e-12, "product density correlation");
    require_close(product_metrics.mn_closed, 0.0, 1.0e-14, "product closed-form fN");
}

// The closed form is not an approximation. On states where both routes are
// well conditioned they must agree to full precision, which is the only reason
// to trust the surd rearrangement over the intuition about which parity weight
// pairs with which coherence.
void test_closed_form_matches_the_generic_path()
{
    std::mt19937 rng(20260909);
    double worst = 0.0;
    double largest = 0.0;
    for (int trial = 0; trial < 400; ++trial)
    {
        const auto values = interleaved(random_parity_preserving(rng));
        const auto metrics = mipt::dist::two_party_metrics(values.data(), true, true);
        worst = std::max(worst, metrics.mn_residual);
        largest = std::max(largest, metrics.mn_closed);
        expect(metrics.parity_leakage < 1.0e-15,
               "a parity-preserving state has no cross-parity weight");
    }
    expect(largest > 0.05, "the random states must actually be entangled, or this proves nothing");
    expect(worst < 1.0e-12,
           "closed form and generic partial transpose disagree by " + std::to_string(worst));
}

// The reason the closed form exists. The generic route computes a trace norm
// that is 1 + O(N) and subtracts 1, so it loses every significant digit once
// the negativity is small: it is still right at |G|^2 = 1e-14, wrong at 1e-16,
// and returns *exactly zero* below that. The closed form is a ratio, so it
// stays accurate at any magnitude.
//
// That is what makes the positivity threshold sweepable. Against the generic
// value, P(fN > eps) would stop moving below eps ~ 1e-16 no matter what the
// physics did, because every record would have collapsed onto zero.
void test_small_negativity_precision()
{
    // A state with p_e = 0.6 and a tunable hopping coherence, whose exact
    // negativity is 2|G|^2 / (sqrt(p_e^2 + 4|G|^2) + p_e).
    const auto build = [](double g) {
        Rho rho{};
        rho[0] = 0.3;
        rho[1 * 4 + 1] = 0.2;
        rho[2 * 4 + 2] = 0.2;
        rho[3 * 4 + 3] = 0.3;
        rho[2 * 4 + 1] = C(g, 0.0);
        rho[1 * 4 + 2] = C(g, 0.0);
        return interleaved(rho);
    };

    bool saw_generic_collapse = false;
    for (double g : {1.0e-5, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10})
    {
        const auto values = build(g);
        const auto metrics = mipt::dist::two_party_metrics(values.data(), true, true);
        const double exact = 2.0 * g * g / (std::sqrt(0.36 + 4.0 * g * g) + 0.6);
        expect(std::abs(metrics.mn_closed - exact) <= 1.0e-13 * exact,
               "the closed form is accurate to full relative precision at |G| = " +
                   std::to_string(g));
        if (exact < 1.0e-16 && metrics.mn_generic == 0.0)
        {
            saw_generic_collapse = true;
        }
    }
    expect(saw_generic_collapse,
           "the generic path must be seen to collapse to exactly zero, or this test is "
           "not measuring the thing it exists for");

    // And the reported value is the closed one, so the aggregate and the
    // per-record files carry the accurate number.
    const auto values = build(1.0e-9);
    const auto metrics = mipt::dist::two_party_metrics(values.data(), true, true);
    expect(metrics.mn == metrics.mn_closed,
           "a parity-preserving fermionic trace reports the closed form");
}

// Matched fp32/fp64 around the threshold.
//
// Two separate facts, and conflating them would be a serious mistake:
//
//   1. Given the same matrix, the closed form loses nothing to fp32 rounding of
//      that matrix -- it tracks the rounded input's own exact answer.
//   2. But an fp32 *state vector* puts ~1e-7 noise into an amplitude that is
//      algebraically zero, so |G|^2 picks up a ~1e-14 floor. Below roughly that,
//      a positivity threshold is counting arithmetic noise, whatever formula is
//      used. The precision therefore has to be recorded alongside the threshold,
//      and it is: `statevector_precision` is a column in both output files.
void test_precision_floor()
{
    const auto build = [](double g) {
        Rho rho{};
        rho[0] = 0.3;
        rho[1 * 4 + 1] = 0.2;
        rho[2 * 4 + 2] = 0.2;
        rho[3 * 4 + 3] = 0.3;
        rho[2 * 4 + 1] = C(g, 0.0);
        rho[1 * 4 + 2] = C(g, 0.0);
        return interleaved(rho);
    };

    // (1) fp32 rounding of the matrix costs the closed form only the input's
    // own relative precision. The matched pair is the same state read at the
    // two precisions: the answers must track each other to ~1e-6 relative, not
    // diverge, and neither may collapse to zero.
    const double g = 3.0e-7;
    const auto exact_input = build(g);
    const auto rounded_input = through_fp32(exact_input);
    const auto exact_metrics = mipt::dist::two_party_metrics(exact_input.data(), true, true);
    const auto rounded_metrics = mipt::dist::two_party_metrics(rounded_input.data(), true, true);
    expect(exact_metrics.mn_closed > 0.0 && rounded_metrics.mn_closed > 0.0,
           "neither precision loses the negativity entirely");
    expect(std::abs(rounded_metrics.mn_closed - exact_metrics.mn_closed) <=
               1.0e-6 * exact_metrics.mn_closed,
           "fp32 and fp64 readings of one state agree to the input's own precision");
    // The same comparison against the generic route, for contrast: at this
    // magnitude it still works, which is why 1e-12 was a defensible default.
    expect(std::abs(rounded_metrics.mn_generic - exact_metrics.mn_closed) <=
               1.0e-3 * exact_metrics.mn_closed,
           "the generic path is still usable this far above its floor");

    // (2) An algebraically zero coherence carried at fp32 noise still reports a
    // negativity, and it lands near 1e-14. Anything below that is noise.
    const double noise = 1.0e-7;
    const auto noisy = build(noise);
    const auto noisy_metrics = mipt::dist::two_party_metrics(noisy.data(), true, true);
    expect(noisy_metrics.mn_closed > 1.0e-15 && noisy_metrics.mn_closed < 1.0e-13,
           "fp32-scale noise in G produces a negativity around 1e-14, not zero");
    // The historical default threshold sits two decades above that floor, which
    // is why it is a safe default and why lowering it is a decision about
    // precision rather than about physics.
    expect(noisy_metrics.mn_closed < 1.0e-12,
           "the 1e-12 default still classifies fp32 noise as unentangled");
}

// The closed form is derived for parity-preserving states only, and the caller
// says whether it applies rather than the matrix being sniffed. A state with
// cross-parity weight must fall back to the generic route and must report the
// leakage that says so.
void test_non_parity_states_fall_back()
{
    std::mt19937 rng(99);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::array<C, 4> psi{};
    for (C &value : psi)
    {
        value = C(normal(rng), normal(rng));
    }
    double norm = 0.0;
    for (const C &value : psi)
    {
        norm += std::norm(value);
    }
    for (C &value : psi)
    {
        value /= std::sqrt(norm);
    }
    const auto values = interleaved(pure_density(psi));

    const auto without = mipt::dist::two_party_metrics(values.data(), true, false);
    expect(without.parity_leakage > 1.0e-3,
           "a generic pure state has real cross-parity weight");
    expect(std::isnan(without.mn_closed), "and no closed form is offered for it");
    expect(without.mn == without.mn_generic, "so the generic value is what is reported");

    // Asked for it anyway, the closed form is simply wrong -- which is the
    // reason the decision is the caller's and is recorded per run rather than
    // being inferred from a threshold on the leakage.
    const auto with = mipt::dist::two_party_metrics(values.data(), true, true);
    expect(with.mn_residual > 1.0e-3,
           "forcing the closed form onto a parity-violating state disagrees loudly");
}

void run()
{
    test_analytic_states();
    test_closed_form_matches_the_generic_path();
    test_small_negativity_precision();
    test_precision_floor();
    test_non_parity_states_fall_back();
}
} // namespace dist_channel_tests

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

    // ------------------------------------------------------- two-point correlators
    //
    // Closed forms in the two-mode Fock basis |n_i n_j> indexed n_i + 2 n_j.
    // G = <c_i^dag c_j> moves one fermion from j to i, so it lives on the
    // {|0,1>, |1,0>} pair; F = <c_i c_j> removes both, so it lives on
    // {|1,1>, |0,0>}. rho_reduce_tests pins the same extraction against the
    // operator algebra on a random many-mode state.
    {
        // (|0,1> + |1,0>)/sqrt(2): G = 1/2 exactly, F = 0.
        const auto hop =
            interleaved(pure_density({C{}, C(inv_sqrt2, 0.0), C(inv_sqrt2, 0.0), C{}}));
        const auto metrics = mipt::dist::two_party_metrics(hop.data(), true);
        require_close(metrics.g2, 0.25, 1.0e-12, "single-fermion hopping |G|^2");
        require_close(metrics.f2, 0.0, 1.0e-12, "single-fermion hopping |F|^2");

        // (|0,0> + |1,1>)/sqrt(2), the paired state: F = 1/2 exactly, G = 0.
        const auto metrics_pair = mipt::dist::two_party_metrics(bell.data(), true);
        require_close(metrics_pair.g2, 0.0, 1.0e-12, "paired-state |G|^2");
        require_close(metrics_pair.f2, 0.25, 1.0e-12, "paired-state |F|^2");

        // A product state and a classical mixture of |0,0> and |1,1> have no
        // coherence on either pair of basis states.
        for (bool fermionic : {false, true})
        {
            const auto p = mipt::dist::two_party_metrics(product.data(), fermionic);
            require_close(p.g2, 0.0, 1.0e-12, "product |G|^2");
            require_close(p.f2, 0.0, 1.0e-12, "product |F|^2");
            const auto c = mipt::dist::two_party_metrics(classical_ri.data(), fermionic);
            require_close(c.g2, 0.0, 1.0e-12, "classical-mixture |G|^2");
            require_close(c.f2, 0.0, 1.0e-12, "classical-mixture |F|^2");
        }

        // Both are read off a trace-normalized rho, so an unnormalized input
        // must give the same answer -- the CSV means would otherwise drift
        // with whatever trace the reduction happened to accumulate.
        auto scaled = hop;
        for (double &v : scaled) { v *= 3.7; }
        const auto scaled_metrics = mipt::dist::two_party_metrics(scaled.data(), true);
        require_close(scaled_metrics.g2, 0.25, 1.0e-12, "|G|^2 is trace-normalized");
    }

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
    test_dist_records();
    dist_channel_tests::run();

    std::cout << "dist_metrics_tests: PASS\n";
    return 0;
}
