#include "mipt/dist_metrics.hpp"
#include "mipt/dist_records.hpp"
#include "mipt/dist_scaling_resume.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <sstream>
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
// Gate-path diagnostics (MIPT_DIST_GATE_DIAGNOSTICS): the record payload, the
// header provenance, the pair CSV metadata, the resume refusals, and the run
// preconditions. The graph side (capacities, widest paths) is pinned by
// dist_connectivity_tests; this pins what the writer does with the values.
// ---------------------------------------------------------------------------
namespace dist_gate_tests
{
namespace dist = mipt::dist;
namespace rec = mipt::dist::records;
namespace res = mipt::dist::resume;
using dist_record_tests::read_back;
using dist_resume_tests::expect;
using dist_resume_tests::refuses;

std::string scratch_directory()
{
    const auto path = std::filesystem::temp_directory_path() / "mipt_dist_gate_tests";
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path.string();
}

dist::RunConfig rppu_config(bool gate)
{
    dist::RunConfig config;
    config.k = 2;
    config.n = 8;
    config.periods = 4;
    config.realizations = 4;
    config.type = mipt::CircuitType::FermionRPPU;
    config.gate_path_diagnostics = gate;
    return config;
}

std::vector<std::array<int, 4>> pair_embeddings(int n)
{
    std::vector<std::array<int, 4>> embeddings;
    std::vector<int> seen(static_cast<std::size_t>(n / 2), 0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            const int bin = mipt::util::periodic_distance(i, j, n) - 1;
            embeddings.push_back({bin, seen[static_cast<std::size_t>(bin)]++, i, j});
        }
    }
    return embeddings;
}

// Remove every occurrence of `token` from `text`.
std::string erase_all(std::string text, const std::string &token)
{
    for (std::size_t at = text.find(token); at != std::string::npos; at = text.find(token))
    {
        text.erase(at, token.size());
    }
    return text;
}

bool refuses_argument(const std::function<void()> &action)
{
    try
    {
        action();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}

const std::vector<std::string> GATE_NAMES{
    "component_size", "shortest_path", "shortest_path_unitary_count",
    "shortest_maximin_gate_ep", "widest_gate_ep", "widest_even_mix", "widest_odd_mix"};

// The seven gate values go after *every* detail-level value, at every detail
// level, and nowhere when the setting is off.
void test_observable_ordering()
{
    for (const auto detail : {dist::RecordDetail::Basic, dist::RecordDetail::Channel,
                              dist::RecordDetail::Full})
    {
        for (const auto type : {mipt::CircuitType::FermionRPPU, mipt::CircuitType::QubitRPPU})
        {
            auto config = rppu_config(false);
            config.type = type;
            config.record_detail = detail;
            const auto off = rec::pair_observable_names(config);
            expect(off == rec::pair_detail_observable_names(config),
                   "with the diagnostics off the observable list is the detail list");
            for (const auto &name : off)
            {
                expect(std::find(GATE_NAMES.begin(), GATE_NAMES.end(), name) == GATE_NAMES.end(),
                       "no gate observable appears with the diagnostics off");
            }
            config.gate_path_diagnostics = true;
            const auto on = rec::pair_observable_names(config);
            expect(on.size() == off.size() + GATE_NAMES.size(),
                   "the diagnostics append exactly seven observables");
            expect(std::equal(off.begin(), off.end(), on.begin()),
                   "the detail-level observables keep their positions");
            expect(std::equal(GATE_NAMES.begin(), GATE_NAMES.end(), on.begin() + off.size()),
                   "the gate observables follow, in the contract's order");
        }
    }
    expect(rec::gate_observable_names().size() == rec::GATE_RECORD_VALUES,
           "the gate name list and the value count agree");
    for (std::size_t i = 0; i < GATE_NAMES.size(); ++i)
    {
        expect(GATE_NAMES[i] == rec::gate_observable_names()[i],
               "gate_observable_names matches the contract");
    }
}

// Off: byte-identical to the pre-feature format. On: exactly the documented
// additions and nothing else, so stripping them recovers the off header.
void test_headers()
{
    auto off = rppu_config(false);
    auto on = rppu_config(true);
    const auto bins = dist::make_pair_bins(off.n);
    const auto embeddings = pair_embeddings(off.n);

    // ---- pair CSV
    const std::string csv_off = dist::pair_csv_header(off);
    const std::string csv_on = dist::pair_csv_header(on);
    expect(csv_off.find("gate") == std::string::npos,
           "the pair CSV header carries no gate token with the diagnostics off");
    expect(csv_off.find(",connectivity_graph_version,connectivity_graph,separation,") !=
               std::string::npos,
           "with the diagnostics off the graph columns run straight into the bin columns");
    expect(csv_on.find(",connectivity_graph,gate_path_diagnostics,gate_diagnostics_version,"
                       "gate_diagnostics_definition,separation,") != std::string::npos,
           "the gate columns sit right after connectivity_graph");
    expect(erase_all(csv_on, ",gate_path_diagnostics,gate_diagnostics_version,"
                             "gate_diagnostics_definition") == csv_off,
           "the gate columns are the only difference in the pair CSV header");
    // The k=3 header never changes.
    auto triple_off = off;
    triple_off.k = 3;
    auto triple_on = triple_off;
    triple_on.gate_path_diagnostics = true;
    expect(dist::triple_csv_header(triple_off) == dist::triple_csv_header(triple_on),
           "the triangle CSV header ignores the gate setting");

    // The rendered rows carry the values.
    const std::string rendered = dist::render_pair_csv(on, bins);
    std::string definition = mipt::util::csv_quote(mipt::GATE_DIAGNOSTICS_DEFINITION);
    expect(rendered.find(",1," + std::to_string(mipt::GATE_DIAGNOSTICS_VERSION) + "," +
                         definition + ",") != std::string::npos,
           "every pair row writes 1, the version, and the quoted definition");

    // ---- record header
    const std::string rec_off = rec::pair_header_text(off, bins, embeddings);
    const std::string rec_on = rec::pair_header_text(on, bins, embeddings);
    for (const char *token : {"gate", "cusv", "component_size", "widest"})
    {
        expect(rec_off.find(token) == std::string::npos,
               std::string("the record header carries no '") + token +
                   "' with the diagnostics off");
    }
    const std::string definition_line =
        std::string("gate_diagnostics_definition=") + mipt::GATE_DIAGNOSTICS_DEFINITION + "\n";
    expect(rec_on.find("\ngate_path_diagnostics=1\n") != std::string::npos,
           "the record header announces the diagnostics");
    expect(rec_on.find("\ngate_diagnostics_version=" +
                       std::to_string(mipt::GATE_DIAGNOSTICS_VERSION) + "\n") !=
               std::string::npos,
           "the record header carries the definition version");
    expect(rec_on.find(definition_line) != std::string::npos,
           "the record header carries the definition string");
    for (const char *key : {"\ncusv_parity=", "\ncusv_measure_bits=", "\ncusv_block_targets="})
    {
        expect(rec_on.find(key) != std::string::npos,
               std::string("the record header carries ") + (key + 1));
    }
    // Strip the documented additions and the off header must come back. The
    // cuStateVec values depend on the environment, so take them from the
    // header itself.
    std::string stripped = rec_on;
    for (const char *key : {"gate_path_diagnostics=", "gate_diagnostics_version=",
                            "gate_diagnostics_definition=", "cusv_parity=",
                            "cusv_measure_bits=", "cusv_block_targets="})
    {
        const std::size_t at = stripped.find(std::string("\n") + key);
        expect(at != std::string::npos, std::string("found ") + key);
        const std::size_t end = stripped.find('\n', at + 1);
        stripped.erase(at, end - at);
    }
    std::string gate_fields;
    std::string gate_observables;
    for (const auto &name : GATE_NAMES)
    {
        gate_fields += "," + name + ":f8";
        gate_observables += "," + name;
    }
    stripped = erase_all(stripped, gate_fields);
    stripped = erase_all(stripped, gate_observables);
    expect(stripped == rec_off,
           "the gate fields and observables are the only difference in the record header");

    // k=3 record headers never change.
    auto geometries = mipt::probed::enumerate_three_probe_geometries(off.n, 0.5);
    const auto triple_bins = dist::make_triple_bins(off.n, geometries);
    expect(rec::triple_header_text(triple_off, triple_bins, 1, geometries) ==
               rec::triple_header_text(triple_on, triple_bins, 1, geometries),
           "the triangle record header ignores the gate setting");
}

// The one decision that matters: no path is NaN, never 0. component_size is
// written for every evaluated pair.
void test_record_values()
{
    mipt::dist::GatePathMetrics gate;
    gate.available = true;
    gate.shortest_path_length = 9;
    gate.shortest_path_unitary_count = 3;
    gate.shortest_maximin_entangling_power = 0.25;
    gate.widest_entangling_power = 0.5;
    gate.widest_even_block_mixing = 0.75;
    gate.widest_odd_block_mixing = 0.125;

    mipt::dist::PairConnectivity conn;
    conn.evaluated = true;
    conn.survives_i = conn.survives_j = conn.interior_path = conn.connected = true;
    conn.component_size = 42;
    conn.shortest_path = 9;

    std::array<double, rec::GATE_RECORD_VALUES> out{};
    rec::fill_gate_record_values(conn, gate, out.data());
    const std::array<double, rec::GATE_RECORD_VALUES> expected{42.0, 9.0, 3.0, 0.25,
                                                               0.5,  0.75, 0.125};
    expect(out == expected, "a connected pair writes every gate value");

    const auto blanked = [](const std::array<double, rec::GATE_RECORD_VALUES> &values) {
        for (std::size_t i = 1; i < values.size(); ++i)
        {
            if (!std::isnan(values[i]))
            {
                return false;
            }
        }
        return true;
    };

    // Shares an interior component but endpoint j was measured last: the
    // final-slice path exists, the pair is not connected.
    auto measured = conn;
    measured.survives_j = false;
    measured.connected = false;
    rec::fill_gate_record_values(measured, gate, out.data());
    expect(out[0] == 42.0, "component_size is written for a pair that is not connected");
    expect(blanked(out), "a pair that is not connected writes NaN, not 0, for every path value");

    // Different components: no path, gate metrics unavailable.
    auto disconnected = conn;
    disconnected.interior_path = disconnected.connected = false;
    disconnected.component_size = 7;
    disconnected.shortest_path = -1;
    rec::fill_gate_record_values(disconnected, mipt::dist::GatePathMetrics{}, out.data());
    expect(out[0] == 7.0, "component_size is written for a disconnected pair");
    expect(blanked(out), "a disconnected pair writes NaN for every path value");

    // Connected, but the graph had no gate metrics.
    rec::fill_gate_record_values(conn, mipt::dist::GatePathMetrics{}, out.data());
    expect(out[0] == 42.0 && blanked(out), "unavailable gate metrics are NaN, not 0");

    // An unevaluated pair has nothing at all.
    rec::fill_gate_record_values(mipt::dist::PairConnectivity{}, gate, out.data());
    expect(std::isnan(out[0]) && blanked(out), "an unevaluated pair writes NaN throughout");
}

// Write a record file under the real k=2 header with the gate fields, read it
// back with the independent reader, and check the NaNs survive bit for bit.
void test_binary_round_trip(const std::string &directory)
{
    auto config = rppu_config(true);
    const auto bins = dist::make_pair_bins(config.n);
    const auto embeddings = pair_embeddings(config.n);
    const auto names = rec::pair_observable_names(config);
    const std::size_t detail_count = rec::pair_detail_observable_names(config).size();
    const std::string header = rec::pair_header_text(config, bins, embeddings);
    const std::string path = directory + "/gate_records.bin";

    mipt::dist::GatePathMetrics gate;
    gate.available = true;
    gate.shortest_path_length = 5;
    gate.shortest_path_unitary_count = 2;
    gate.shortest_maximin_entangling_power = 0.3;
    gate.widest_entangling_power = 0.6;
    gate.widest_even_block_mixing = 0.2;
    gate.widest_odd_block_mixing = 0.9;

    std::vector<std::vector<double>> written;
    {
        rec::RecordSink sink;
        sink.open(path, header, names.size(), 0);
        for (std::uint32_t realization = 0; realization < 2; ++realization)
        {
            for (std::size_t e = 0; e < embeddings.size(); ++e)
            {
                std::vector<double> values(names.size());
                for (std::size_t slot = 0; slot < detail_count; ++slot)
                {
                    values[slot] = 0.01 * static_cast<double>(slot + e) + realization;
                }
                mipt::dist::PairConnectivity conn;
                conn.evaluated = true;
                conn.survives_i = conn.survives_j = true;
                conn.interior_path = conn.connected = (e % 3) != 0;
                conn.component_size = static_cast<std::uint32_t>(10 + e);
                rec::fill_gate_record_values(conn, conn.connected ? gate
                                                                  : mipt::dist::GatePathMetrics{},
                                             values.data() + detail_count);
                sink.add(realization, static_cast<std::uint32_t>(embeddings[e][0]),
                         static_cast<std::uint32_t>(embeddings[e][1]), conn.flags(),
                         values.data());
                written.push_back(values);
            }
        }
        sink.close();
    }

    const auto [read_header, rows] = read_back(path, names.size());
    expect(read_header == header, "the gate header round-trips verbatim");
    expect(rows.size() == written.size(), "every gate record reached the file");
    for (std::size_t index = 0; index < rows.size(); ++index)
    {
        expect(std::memcmp(rows[index].values.data(), written[index].data(),
                           names.size() * sizeof(double)) == 0,
               "gate records round-trip bit for bit, NaNs included");
        const bool connected = (rows[index].flags & mipt::dist::FLAG_CONNECTED) != 0u;
        expect(connected == !std::isnan(rows[index].values[detail_count + 4]),
               "widest_gate_ep is present exactly on connected records");
        expect(!std::isnan(rows[index].values[detail_count]),
               "component_size is present on every record");
    }

    // The record file refuses a resume across the setting, both ways: the
    // header differs and so does the record width.
    auto off = rppu_config(false);
    const auto off_names = rec::pair_observable_names(off);
    const std::string off_header = rec::pair_header_text(off, bins, embeddings);
    expect(refuses([&] {
               rec::RecordSink sink;
               sink.open(path, off_header, off_names.size(), 2);
           }),
           "a gate record file cannot be extended with the diagnostics off");
    const std::string off_path = directory + "/plain_records.bin";
    {
        rec::RecordSink sink;
        sink.open(off_path, off_header, off_names.size(), 0);
        std::vector<double> values(off_names.size(), 0.5);
        sink.add(0, 0, 0, 0u, values.data());
        sink.close();
    }
    expect(refuses([&] {
               rec::RecordSink sink;
               sink.open(off_path, header, names.size(), 1);
           }),
           "a plain record file cannot be extended with the diagnostics on");
    // ... and the same file still resumes under its own setting.
    {
        rec::RecordSink sink;
        sink.open(path, header, names.size(), 2);
        expect(sink.inherited_records() == written.size(),
               "a gate record file resumes under the setting it was written with");
        sink.close();
    }
}

// The pair CSV refuses a resume across the setting in both directions, and
// round-trips under it.
void test_csv_resume(const std::string &directory)
{
    auto on = rppu_config(true);
    on.output_path = directory + "/gate_pairs.csv";
    auto off = rppu_config(false);
    off.output_path = directory + "/plain_pairs.csv";

    auto bins = dist::make_pair_bins(on.n);
    for (auto &bin : bins)
    {
        for (std::uint64_t e = 0; e < 2 * bin.embedding_count; ++e)
        {
            for (auto *stats : {&bin.mi, &bin.mn, &bin.g2, &bin.f2, &bin.fmi, &bin.fmn,
                                &bin.fg2, &bin.ff2, &bin.fmn_residual, &bin.n_i, &bin.n_j,
                                &bin.dnn, &bin.rho_n, &bin.abs_rho_n, &bin.rho_n_sq,
                                &bin.i_occ})
            {
                stats->add(0.1 + 0.01 * static_cast<double>(e));
            }
        }
    }
    dist::publish_csv(on.output_path, dist::render_pair_csv(on, bins));
    dist::publish_csv(off.output_path, dist::render_pair_csv(off, bins));

    {
        auto restored = dist::make_pair_bins(on.n);
        const auto report = res::load_pairs(on, restored);
        expect(report.loaded && report.completed == 2,
               "a gate pair CSV resumes under the setting it was written with");
    }
    {
        auto restored = dist::make_pair_bins(off.n);
        expect(res::load_pairs(off, restored).completed == 2,
               "a plain pair CSV still resumes with the diagnostics off");
    }
    auto on_reading_off = on;
    on_reading_off.output_path = off.output_path;
    expect(refuses([&] {
               auto b = dist::make_pair_bins(on.n);
               (void)res::load_pairs(on_reading_off, b);
           }),
           "a plain pair CSV cannot be resumed with the diagnostics on");
    auto off_reading_on = off;
    off_reading_on.output_path = on.output_path;
    expect(refuses([&] {
               auto b = dist::make_pair_bins(on.n);
               (void)res::load_pairs(off_reading_on, b);
           }),
           "a gate pair CSV cannot be resumed with the diagnostics off");

    // A file claiming another definition version is refused too.
    std::ifstream in(on.output_path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string needle = ",1," + std::to_string(mipt::GATE_DIAGNOSTICS_VERSION) + ",\"";
    for (std::size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at))
    {
        text.replace(at, needle.size(), ",1,999,\"");
    }
    auto edited = on;
    edited.output_path = directory + "/edited_pairs.csv";
    dist::publish_csv(edited.output_path, text);
    expect(refuses([&] {
               auto b = dist::make_pair_bins(on.n);
               (void)res::load_pairs(edited, b);
           }),
           "a pair CSV from another gate-definition version is refused");
}

void test_validation()
{
    auto config = rppu_config(true);
    expect(!refuses_argument([&] { rec::validate_gate_diagnostics(config, true); }),
           "RPPU, k=2, connectivity on, records on is accepted");
    auto qubit = config;
    qubit.type = mipt::CircuitType::QubitRPPU;
    expect(!refuses_argument([&] { rec::validate_gate_diagnostics(qubit, true); }),
           "qRPPU is accepted");
    auto both = config;
    both.k = 0;
    expect(!refuses_argument([&] { rec::validate_gate_diagnostics(both, true); }),
           "k=0 is accepted");
    auto gap_only = config;
    gap_only.pair_gap.enabled = true;
    expect(!refuses_argument([&] { rec::validate_gate_diagnostics(gap_only, false); }),
           "the triple analysis alone is somewhere to write");

    for (const auto type : {mipt::CircuitType::Haar, mipt::CircuitType::MMS,
                            mipt::CircuitType::RFGS})
    {
        auto other = config;
        other.type = type;
        expect(refuses_argument([&] { rec::validate_gate_diagnostics(other, true); }),
               "a circuit without gate capacities is refused, not written as NaN");
    }
    auto triangles = config;
    triangles.k = 3;
    expect(refuses_argument([&] { rec::validate_gate_diagnostics(triangles, true); }),
           "k=3 is refused");
    auto no_graph = config;
    no_graph.connectivity = false;
    expect(refuses_argument([&] { rec::validate_gate_diagnostics(no_graph, true); }),
           "connectivity off is refused");
    expect(refuses_argument([&] { rec::validate_gate_diagnostics(config, false); }),
           "nowhere to write is refused");

    // Off never refuses, whatever else is set.
    auto off = config;
    off.gate_path_diagnostics = false;
    off.type = mipt::CircuitType::Haar;
    off.k = 3;
    off.connectivity = false;
    expect(!refuses_argument([&] { rec::validate_gate_diagnostics(off, false); }),
           "the diagnostics off imposes nothing");
}

void run(const std::string &directory)
{
    test_observable_ordering();
    test_headers();
    test_record_values();
    test_binary_round_trip(directory);
    test_csv_resume(directory);
    test_validation();
    std::filesystem::remove_all(directory);
}
} // namespace dist_gate_tests

void test_dist_gate_diagnostics()
{
    dist_gate_tests::run(dist_gate_tests::scratch_directory());
}

// ---------------------------------------------------------------------------
// Temporal-turn diagnostics (MIPT_DIST_TURN_DIAGNOSTICS): the twenty-one record
// values and their NaN rule, flag bits 7-9, the header provenance, the three
// refined contingency tables of the pair CSV and their exact invariants, the
// resume refusals in both directions, and the one precondition of their own.
// The shapes and the doubled network themselves are pinned by
// temporal_turn_tests; this pins what the writers do with them, on PairTurns a
// real ConnectivityIndex computed from hand-built histories.
// ---------------------------------------------------------------------------
namespace dist_turn_tests
{
namespace dist = mipt::dist;
namespace rec = mipt::dist::records;
namespace res = mipt::dist::resume;
using dist_gate_tests::erase_all;
using dist_gate_tests::GATE_NAMES;
using dist_gate_tests::pair_embeddings;
using dist_gate_tests::refuses_argument;
using dist_record_tests::read_back;
using dist_resume_tests::expect;
using dist_resume_tests::refuses;

std::string scratch_directory()
{
    const auto path = std::filesystem::temp_directory_path() / "mipt_dist_turn_tests";
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path.string();
}

// The contract's names and lines, spelled out rather than taken from the code
// under test.
const std::vector<std::string> TURN_NAMES{
    "sm_turns",        "sm_temporal_forward", "sm_temporal_backward", "sm_min_layer",
    "sm_max_layer",    "sm_class",            "sm_bridges",           "sm_bridges_measured",
    "sm_bridges_trace_only", "sm_cancelled_gates",
    "wd_turns",        "wd_temporal_forward", "wd_temporal_backward", "wd_min_layer",
    "wd_max_layer",    "wd_class",            "wd_bridges",           "wd_bridges_measured",
    "wd_bridges_trace_only", "wd_cancelled_gates",
    "min_turns"};
const std::string OLD_FLAG_BITS =
    "flag_bits=0:evaluated,1:survives_i,2:survives_j,3:interior_path,4:connected\n";
const std::string TURN_FLAG_BITS =
    "flag_bits=0:evaluated,1:survives_i,2:survives_j,3:interior_path,4:connected,"
    "5:fn_positive,6:mn_positive,7:connected_common_source,8:connected_certified,"
    "9:connected_doubled\n";
const std::string TURN_META =
    ",turn_diagnostics,turn_diagnostics_version,turn_diagnostics_definition";
const std::string TURN_COUNTERS =
    "connected_common_source_count,connected_common_source_ent_positive_count,"
    "connected_common_source_ent_zero_count,disconnected_common_source_ent_positive_count,"
    "disconnected_common_source_ent_zero_count,"
    "connected_certified_count,connected_certified_ent_positive_count,"
    "connected_certified_ent_zero_count,disconnected_certified_ent_positive_count,"
    "disconnected_certified_ent_zero_count,"
    "connected_doubled_count,connected_doubled_ent_positive_count,"
    "connected_doubled_ent_zero_count,disconnected_doubled_ent_positive_count,"
    "disconnected_doubled_ent_zero_count";
constexpr std::uint32_t TURN_BITS = mipt::dist::FLAG_CONNECTED_COMMON_SOURCE |
                                    mipt::dist::FLAG_CONNECTED_CERTIFIED |
                                    mipt::dist::FLAG_CONNECTED_DOUBLED;
const std::string NESTING = "C_common_source <= C_certified <= C_doubled <= C";

dist::RunConfig turn_config(bool turn)
{
    dist::RunConfig config = dist_gate_tests::rppu_config(true);
    config.turn_diagnostics = turn;
    return config;
}

std::size_t slot_of(const std::vector<std::string> &names, const std::string &name)
{
    return static_cast<std::size_t>(std::find(names.begin(), names.end(), name) - names.begin());
}

// A history from bonds per layer and (layer, mode) measurements. Every gate
// carries capacities, which the canonical paths need to exist at all; the
// turn values do not read them.
mipt::LogicalHistory make_history(int n, const std::vector<std::vector<std::pair<int, int>>> &bonds,
                                  const std::vector<std::pair<int, int>> &measured,
                                  bool gate_metrics = true)
{
    mipt::LogicalHistory history;
    history.reset(n);
    for (std::size_t t = 0; t < bonds.size(); ++t)
    {
        mipt::LogicalLayer layer;
        layer.measured.assign(static_cast<std::size_t>(n), 0u);
        int index = 0;
        for (const auto &[a, b] : bonds[t])
        {
            mipt::LogicalBond bond{a, b};
            bond.source_gate_index = index++;
            if (gate_metrics)
            {
                bond.entangling_power = 0.9;
                bond.even_block_mixing = 0.8;
                bond.odd_block_mixing = 0.7;
            }
            layer.bonds.push_back(bond);
        }
        for (const auto &[at, mode] : measured)
        {
            if (at == static_cast<int>(t))
            {
                layer.measured[static_cast<std::size_t>(mode)] = 1u;
            }
        }
        history.layers.push_back(std::move(layer));
    }
    history.valid = true;
    return history;
}

// The four-mode zig-zag: gates (0,1) and (2,3) in layer 0, the bridge (1,2) in
// layer 1, nothing in layer 2. The only path joining 0 and 3 climbs mode 1 to
// the bridge and descends mode 2 -- three turns, one peak -- and with nothing
// measured the bridge lies in neither endpoint's past light cone, so the
// doubled network cancels it.
mipt::LogicalHistory zigzag(const std::vector<std::pair<int, int>> &measured,
                            bool gate_metrics = true)
{
    return make_history(4, {{{0, 1}, {2, 3}}, {{1, 2}}, {}}, measured, gate_metrics);
}

// The same zig-zag plus a fifth mode, which connector 1 gates with in the last
// layer and which is measured there. The bridge is then in that measurement's
// past light cone and survives the doubled network, yet no connector is
// measured after it, so no walk is certified: C_doubled without C_certified.
mipt::LogicalHistory zigzag_with_witness()
{
    return make_history(5, {{{0, 1}, {2, 3}}, {{1, 2}}, {{1, 4}}}, {{2, 4}});
}

// The twenty-one turn values follow the seven gate values at every detail
// level, in the contract's order, and appear nowhere with the setting off.
void test_observable_ordering()
{
    expect(rec::TURN_RECORD_VALUES == TURN_NAMES.size() &&
               rec::turn_observable_names().size() == TURN_NAMES.size(),
           "twenty-one turn observables");
    for (std::size_t i = 0; i < TURN_NAMES.size(); ++i)
    {
        expect(TURN_NAMES[i] == rec::turn_observable_names()[i],
               "turn_observable_names matches the contract at " + TURN_NAMES[i]);
    }
    for (const auto detail : {dist::RecordDetail::Basic, dist::RecordDetail::Channel,
                              dist::RecordDetail::Full})
    {
        for (const auto type : {mipt::CircuitType::FermionRPPU, mipt::CircuitType::QubitRPPU})
        {
            auto config = turn_config(false);
            config.type = type;
            config.record_detail = detail;
            const auto off = rec::pair_observable_names(config);
            expect(off.size() == rec::pair_detail_observable_names(config).size() + GATE_NAMES.size(),
                   "with the turns off a record carries the gate tail alone");
            for (const auto &name : off)
            {
                expect(std::find(TURN_NAMES.begin(), TURN_NAMES.end(), name) == TURN_NAMES.end(),
                       "no turn observable appears with the setting off");
            }
            config.turn_diagnostics = true;
            const auto on = rec::pair_observable_names(config);
            expect(on.size() == off.size() + TURN_NAMES.size(),
                   "the setting appends exactly twenty-one observables");
            expect(std::equal(off.begin(), off.end(), on.begin()),
                   "every earlier observable keeps its position");
            expect(std::equal(TURN_NAMES.begin(), TURN_NAMES.end(), on.begin() + off.size()),
                   "the turn observables follow the gate ones, in the contract's order");
        }
    }
}

// The payload against the declared names, by name, for every combination of
// the two settings. Every field of the inputs is distinct, so a transposed or
// shifted slot cannot pass.
void test_diagnostic_tail_pairing()
{
    dist::PairConnectivity conn;
    conn.evaluated = conn.survives_i = conn.survives_j = conn.interior_path = conn.connected = true;
    conn.component_size = 42;
    dist::GatePathMetrics gate;
    gate.available = true;
    gate.shortest_path_length = 9;
    gate.shortest_path_unitary_count = 3;
    gate.shortest_maximin_entangling_power = 0.25;
    gate.widest_entangling_power = 0.5;
    gate.widest_even_block_mixing = 0.75;
    gate.widest_odd_block_mixing = 0.125;
    const auto shape = [](int base, dist::PathClass path_class) {
        dist::PathShape s;
        s.available = true;
        s.turns = base + 1;
        s.temporal_forward = base + 2;
        s.temporal_backward = base + 3;
        s.min_layer = base + 4;
        s.max_layer = base + 5;
        s.path_class = path_class;
        s.bridges.resize(static_cast<std::size_t>(base + 6));
        s.bridges_measured = base + 7;
        s.bridges_trace_only = base + 8;
        s.cancelled_gates = base + 9;
        return s;
    };
    dist::PairTurns turns;
    turns.evaluated = true;
    turns.shortest_maximin = shape(10, dist::PathClass::ZigzagCertified);
    turns.widest = shape(20, dist::PathClass::ZigzagTraceOnly);
    turns.min_turns = 31;
    const std::map<std::string, double> expected{
        {"component_size", 42.0},        {"shortest_path", 9.0},
        {"shortest_path_unitary_count", 3.0}, {"shortest_maximin_gate_ep", 0.25},
        {"widest_gate_ep", 0.5},         {"widest_even_mix", 0.75},
        {"widest_odd_mix", 0.125},
        {"sm_turns", 11.0},              {"sm_temporal_forward", 12.0},
        {"sm_temporal_backward", 13.0},  {"sm_min_layer", 14.0},
        {"sm_max_layer", 15.0},          {"sm_class", 2.0},
        {"sm_bridges", 16.0},            {"sm_bridges_measured", 17.0},
        {"sm_bridges_trace_only", 18.0}, {"sm_cancelled_gates", 19.0},
        {"wd_turns", 21.0},              {"wd_temporal_forward", 22.0},
        {"wd_temporal_backward", 23.0},  {"wd_min_layer", 24.0},
        {"wd_max_layer", 25.0},          {"wd_class", 3.0},
        {"wd_bridges", 26.0},            {"wd_bridges_measured", 27.0},
        {"wd_bridges_trace_only", 28.0}, {"wd_cancelled_gates", 29.0},
        {"min_turns", 31.0}};
    for (const auto detail : {dist::RecordDetail::Basic, dist::RecordDetail::Channel,
                              dist::RecordDetail::Full})
    {
        for (const bool gate_on : {false, true})
        {
            for (const bool turn_on : {false, true})
            {
                auto config = dist_gate_tests::rppu_config(gate_on);
                config.turn_diagnostics = turn_on;
                config.record_detail = detail;
                const auto names = rec::pair_observable_names(config);
                const std::size_t detail_count = rec::pair_detail_observable_names(config).size();
                std::vector<double> tail(names.size() - detail_count + 1, -7.0);
                const std::size_t filled =
                    rec::fill_pair_diagnostic_values(config, conn, gate, turns, tail.data());
                expect(filled == names.size() - detail_count,
                       "the diagnostic tail is exactly as long as the declared one");
                expect(tail.back() == -7.0, "nothing is written past the declared tail");
                for (std::size_t k = 0; k < filled; ++k)
                {
                    const std::string &name = names[detail_count + k];
                    expect(expected.count(name) == 1 && tail[k] == expected.at(name),
                           "the slot named " + name + " carries its own value");
                }
            }
        }
    }
}

// The NaN rule, on PairTurns a real index computed.
void test_record_values()
{
    using Values = std::array<double, rec::TURN_RECORD_VALUES>;
    const auto all_nan = [](const Values &values, std::size_t begin, std::size_t end) {
        for (std::size_t k = begin; k < end; ++k)
        {
            if (!std::isnan(values[k]))
            {
                return false;
            }
        }
        return true;
    };
    Values out{};

    // The connected, trace-only zig-zag: every value written, and one gate --
    // the bridge -- cancelled on each canonical path.
    dist::ConnectivityIndex index;
    index.build(zigzag({}), true, true, true);
    const dist::PairConnectivity connected = index.query(0, 3);
    const dist::PairTurns zigzag_turns = index.pair_turns(0, 3);
    expect(index.has_turns() && connected.connected && zigzag_turns.evaluated &&
               zigzag_turns.min_turns == 3 && !zigzag_turns.doubled_network,
           "the fixture is a connected zig-zag the doubled network cuts");
    rec::fill_turn_record_values(connected, zigzag_turns, out.data());
    const Values trace_only{3.0, 3.0, 3.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 1.0,
                            3.0, 3.0, 3.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 1.0, 3.0};
    expect(out == trace_only,
           "a trace-only zig-zag writes both shapes, its cancelled bridge, and min_turns");

    // The connector measured after the bridge: the same path, now certified,
    // and nothing on it cancelled.
    {
        dist::ConnectivityIndex certified;
        certified.build(zigzag({{1, 1}}), true, true, true);
        rec::fill_turn_record_values(certified.query(0, 3), certified.pair_turns(0, 3), out.data());
        const Values expected{3.0, 3.0, 3.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0,
                              3.0, 3.0, 3.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 3.0};
        expect(out == expected, "a certified zig-zag writes class 2, a measured bridge, and "
                                "no cancelled gate");
    }

    // Not connected: endpoint 3 measured in the last layer. The final-slice
    // nodes still share a component, so the index has both shapes, their
    // cancellation counts and min_turns -- the blanking is the writer's.
    {
        dist::ConnectivityIndex measured;
        measured.build(zigzag({{2, 3}}), true, true, true);
        const auto conn = measured.query(0, 3);
        const auto turns = measured.pair_turns(0, 3);
        expect(!conn.connected && conn.interior_path && turns.shortest_maximin.available &&
                   turns.shortest_maximin.cancelled_gates == 1 && turns.min_turns == 3,
               "the index has the shape of a pair whose endpoint was measured last");
        rec::fill_turn_record_values(conn, turns, out.data());
        expect(all_nan(out, 0, out.size()),
               "a pair that is not connected writes NaN, not 0, for every turn value");
    }

    // Shapes unavailable (no gate capacities): connected, min_turns known.
    {
        dist::ConnectivityIndex bare;
        bare.build(zigzag({}, false), true, true, true);
        const auto turns = bare.pair_turns(0, 3);
        expect(bare.query(0, 3).connected && !turns.shortest_maximin.available &&
                   !turns.widest.available && turns.min_turns == 3,
               "without capacities the index has no shapes but does have min_turns");
        rec::fill_turn_record_values(bare.query(0, 3), turns, out.data());
        expect(all_nan(out, 0, 20) && out[20] == 3.0,
               "unavailable shapes write NaN, not 0, while min_turns is still written");
    }

    // No walk at all: a cut below the bridge splits the components, so
    // min_turns is -1 -- written as NaN, never as -1.
    {
        dist::ConnectivityIndex cut;
        cut.build(zigzag({{0, 1}}), true, true, true);
        const auto turns = cut.pair_turns(0, 3);
        expect(!cut.query(0, 3).interior_path && turns.evaluated && turns.min_turns == -1,
               "a cut pair has no walk");
        rec::fill_turn_record_values(cut.query(0, 3), turns, out.data());
        expect(all_nan(out, 0, out.size()), "a pair with no walk writes NaN throughout");
    }

    // The two -1 rules on their own, on a connected pair (combinations the
    // index cannot produce, which is why they are built by hand): a walk search
    // that found nothing, and a cancellation count that was never computed.
    {
        dist::PairTurns turns = zigzag_turns;
        turns.min_turns = -1;
        rec::fill_turn_record_values(connected, turns, out.data());
        expect(std::isnan(out[20]) && out[0] == 3.0 && out[19] == 1.0,
               "min_turns = -1 is NaN and leaves the shapes alone");
        turns = zigzag_turns;
        turns.shortest_maximin.cancelled_gates = -1;
        rec::fill_turn_record_values(connected, turns, out.data());
        expect(std::isnan(out[9]) && out[8] == 1.0 && out[19] == 1.0 && out[20] == 3.0,
               "an uncomputed cancellation count is NaN and leaves everything else alone");
    }

    // Off, and unevaluated: nothing at all.
    rec::fill_turn_record_values(connected, dist::PairTurns{}, out.data());
    expect(all_nan(out, 0, out.size()), "an unevaluated PairTurns writes NaN throughout");
    rec::fill_turn_record_values(dist::PairConnectivity{}, zigzag_turns, out.data());
    expect(all_nan(out, 0, out.size()), "an unevaluated pair writes NaN throughout");
}

// Flag bits 7-9 through the one function the run and the replay both use, on
// every combination the hand-built histories produce.
void test_flag_bits()
{
    struct Case
    {
        mipt::LogicalHistory history;
        int i;
        int j;
        bool connected;
        bool source;
        bool certified;
        bool doubled;
        const char *what;
    };
    const std::vector<Case> cases{
        {zigzag({}), 0, 3, true, false, false, false,
         "a trace-only zig-zag is connected and nothing more"},
        {zigzag({{1, 1}}), 0, 3, true, false, true, true,
         "a connector measured after the bridge certifies the zig-zag"},
        {zigzag({{2, 2}}), 0, 3, true, false, true, true,
         "so does the other connector, measured in the last layer"},
        {zigzag({}), 0, 1, true, true, true, true, "a common source is all three"},
        {zigzag({}), 0, 2, true, true, true, true,
         "a V that crosses the bridge on its way up is a common source"},
        {zigzag_with_witness(), 0, 3, true, false, false, true,
         "a bridge feeding a later measurement survives the doubled network uncertified"},
        {zigzag({{2, 3}}), 0, 3, false, false, false, false,
         "an endpoint measured last carries no turn bit"},
        {zigzag({{0, 1}}), 0, 3, false, false, false, false,
         "a cut below the bridge carries no turn bit"},
    };
    for (const Case &c : cases)
    {
        const std::string what(c.what);
        dist::ConnectivityIndex index;
        index.build(c.history, true, true, true);
        const auto conn = index.query(c.i, c.j);
        const auto turns = index.pair_turns(c.i, c.j);
        expect(conn.connected == c.connected, what + ": connectivity");
        for (const bool fn : {false, true})
        {
            for (const bool mn : {false, true})
            {
                const std::uint32_t legacy = conn.flags() | (fn ? dist::FLAG_FN_POSITIVE : 0u) |
                                             (mn ? dist::FLAG_MN_POSITIVE : 0u);
                expect(rec::pair_record_flags(conn, fn, mn, dist::PairTurns{}) == legacy,
                       what + ": with the setting off the word is the pre-feature one");
                expect((rec::pair_record_flags(conn, fn, mn, turns) & ~TURN_BITS) == legacy,
                       what + ": the turn bits are the only difference");
            }
        }
        const std::uint32_t flags = rec::pair_record_flags(conn, false, false, turns);
        expect(((flags & dist::FLAG_CONNECTED_COMMON_SOURCE) != 0u) == c.source, what + ": bit 7");
        expect(((flags & dist::FLAG_CONNECTED_CERTIFIED) != 0u) == c.certified, what + ": bit 8");
        expect(((flags & dist::FLAG_CONNECTED_DOUBLED) != 0u) == c.doubled, what + ": bit 9");
        expect((flags & TURN_BITS) == 0u || (flags & dist::FLAG_CONNECTED) != 0u,
               what + ": a turn bit implies C");
    }

    // Bit 9 is the doubled network's answer, not the path class's: the witness
    // path is still classed trace-only, yet nothing on it cancels.
    dist::ConnectivityIndex witness;
    witness.build(zigzag_with_witness(), true, true, true);
    const auto turns = witness.pair_turns(0, 3);
    expect(turns.shortest_maximin.path_class == dist::PathClass::ZigzagTraceOnly &&
               turns.shortest_maximin.cancelled_gates == 0,
           "a trace-only class with nothing cancelled: bits 8 and 9 answer different questions");
}

// Off: byte-identical to the pre-feature format. On: exactly the documented
// additions, so stripping them recovers the off header. k=3 never changes.
void test_headers()
{
    auto off = turn_config(false);
    auto on = turn_config(true);
    const auto bins = dist::make_pair_bins(off.n);
    const auto embeddings = pair_embeddings(off.n);

    // ---- pair CSV
    const std::string csv_off = dist::pair_csv_header(off);
    const std::string csv_on = dist::pair_csv_header(on);
    for (const char *token : {"turn", "common_source", "certified", "doubled"})
    {
        expect(csv_off.find(token) == std::string::npos,
               std::string("the pair CSV header carries no '") + token + "' with the setting off");
    }
    expect(csv_on.find(",gate_diagnostics_definition" + TURN_META + ",separation,") !=
               std::string::npos,
           "the turn columns sit right after the gate columns");
    const std::string tail = "," + TURN_COUNTERS + "\n";
    expect(csv_on.size() > tail.size() &&
               csv_on.compare(csv_on.size() - tail.size(), tail.size(), tail) == 0,
           "the fifteen counters close the header, in the contract's order");
    expect(erase_all(erase_all(csv_on, TURN_META), "," + TURN_COUNTERS) == csv_off,
           "the turn columns are the only difference in the pair CSV header");
    auto triple_off = off;
    triple_off.k = 3;
    triple_off.gate_path_diagnostics = false;
    auto triple_on = triple_off;
    triple_on.gate_path_diagnostics = true;
    triple_on.turn_diagnostics = true;
    expect(dist::triple_csv_header(triple_off) == dist::triple_csv_header(triple_on),
           "the triangle CSV header ignores the turn setting");

    const std::string rendered = dist::render_pair_csv(on, bins);
    const std::string metadata = ",1," + std::to_string(dist::TURN_DIAGNOSTICS_VERSION) + "," +
                                 mipt::util::csv_quote(dist::TURN_DIAGNOSTICS_DEFINITION) + ",";
    std::size_t rows_with_metadata = 0;
    for (std::size_t at = rendered.find(metadata); at != std::string::npos;
         at = rendered.find(metadata, at + 1))
    {
        ++rows_with_metadata;
    }
    expect(rows_with_metadata == bins.size(),
           "every pair row writes 1, the version, and the quoted definition");

    // ---- record header
    const std::string rec_off = rec::pair_header_text(off, bins, embeddings);
    const std::string rec_on = rec::pair_header_text(on, bins, embeddings);
    expect(rec_off.find("\n" + OLD_FLAG_BITS) != std::string::npos,
           "with the setting off the flag_bits line is the old one");
    for (const char *token : {"turn_diagnostics", "sm_turns", "wd_turns", "min_turns",
                              "cancelled_gates", "5:fn_positive", "connected_common_source",
                              "connected_certified", "connected_doubled"})
    {
        expect(rec_off.find(token) == std::string::npos,
               std::string("the record header carries no '") + token + "' with the setting off");
    }
    const std::size_t flag_at = rec_on.find("\n" + TURN_FLAG_BITS);
    expect(flag_at != std::string::npos, "with it on the flag_bits line names bits 5 through 9");
    const std::size_t cusv = rec_on.find("\ncusv_block_targets=");
    const std::size_t line_end = cusv == std::string::npos ? cusv : rec_on.find('\n', cusv + 1);
    const std::string block = "turn_diagnostics=1\nturn_diagnostics_version=" +
                              std::to_string(dist::TURN_DIAGNOSTICS_VERSION) +
                              "\nturn_diagnostics_definition=" +
                              std::string(dist::TURN_DIAGNOSTICS_DEFINITION) + "\n";
    expect(line_end != std::string::npos && rec_on.compare(line_end + 1, block.size(), block) == 0,
           "the three turn fields follow cusv_block_targets");
    std::string stripped = rec_on;
    stripped.erase(line_end + 1, block.size());
    stripped.replace(flag_at + 1, TURN_FLAG_BITS.size(), OLD_FLAG_BITS);
    std::string fields;
    std::string observables;
    for (const auto &name : TURN_NAMES)
    {
        fields += "," + name + ":f8";
        observables += "," + name;
    }
    stripped = erase_all(erase_all(stripped, fields), observables);
    expect(stripped == rec_off,
           "the documented additions are the only difference in the record header");

    // k=3 record headers never change -- not even the flag_bits line, which a
    // k=0 run's triangle protocol sees under the turn setting too.
    const auto geometries = mipt::probed::enumerate_three_probe_geometries(off.n, 0.5);
    const auto triple_bins = dist::make_triple_bins(off.n, geometries);
    const std::string triple_plain = rec::triple_header_text(triple_off, triple_bins, 1, geometries);
    expect(triple_plain == rec::triple_header_text(triple_on, triple_bins, 1, geometries),
           "the triangle record header ignores the turn setting");
    expect(triple_plain.find("\n" + OLD_FLAG_BITS) != std::string::npos,
           "and keeps the old flag_bits line");
}

// A record file under the real header, filled through the production helpers
// from real PairTurns, read back with the independent reader.
void test_binary_round_trip(const std::string &directory)
{
    auto config = turn_config(true);
    config.n = 4;
    const auto bins = dist::make_pair_bins(config.n);
    const auto embeddings = pair_embeddings(config.n);
    const auto names = rec::pair_observable_names(config);
    const std::size_t detail_count = rec::pair_detail_observable_names(config).size();
    const std::string header = rec::pair_header_text(config, bins, embeddings);
    const std::string path = directory + "/turn_records.bin";

    // One trajectory per draw, between them every bit combination the
    // four-mode zig-zag can produce.
    const std::vector<std::vector<std::pair<int, int>>> draws{{}, {{1, 1}}, {{2, 3}}, {{0, 1}}};
    std::vector<std::vector<double>> written;
    std::vector<std::uint32_t> written_flags;
    {
        rec::RecordSink sink;
        sink.open(path, header, names.size(), 0);
        for (std::size_t r = 0; r < draws.size(); ++r)
        {
            dist::ConnectivityIndex index;
            index.build(zigzag(draws[r]), true, true, true);
            expect(index.has_turns(), "the index precomputed the turn table");
            for (const auto &row : embeddings)
            {
                std::vector<double> values(names.size(), 0.0);
                for (std::size_t slot = 0; slot < detail_count; ++slot)
                {
                    values[slot] = 0.01 * static_cast<double>(slot) + static_cast<double>(r);
                }
                const auto conn = index.query(row[2], row[3]);
                const auto turns = index.pair_turns(row[2], row[3]);
                const std::size_t filled = rec::fill_pair_diagnostic_values(
                    config, conn, index.gate_path_metrics(row[2], row[3]), turns,
                    values.data() + detail_count);
                expect(detail_count + filled == names.size(),
                       "the diagnostic tail fills the record exactly");
                const std::uint32_t flags =
                    rec::pair_record_flags(conn, (row[3] - row[2]) % 2 == 1, false, turns);
                sink.add(static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(row[0]),
                         static_cast<std::uint32_t>(row[1]), flags, values.data());
                written.push_back(values);
                written_flags.push_back(flags);
            }
        }
        sink.close();
    }

    const auto [read_header, rows] = read_back(path, names.size());
    expect(read_header == header, "the turn header round-trips verbatim");
    expect(rows.size() == written.size(), "every turn record reached the file");
    const std::size_t sm_turns = slot_of(names, "sm_turns");
    const std::size_t sm_cancelled = slot_of(names, "sm_cancelled_gates");
    const std::size_t wd_cancelled = slot_of(names, "wd_cancelled_gates");
    const std::size_t min_turns = slot_of(names, "min_turns");
    std::size_t all_three = 0;
    std::size_t certified_only = 0;
    std::size_t connected_none = 0;
    std::size_t disconnected = 0;
    for (std::size_t k = 0; k < rows.size() && k < written.size(); ++k)
    {
        expect(std::memcmp(rows[k].values.data(), written[k].data(),
                           names.size() * sizeof(double)) == 0,
               "turn records round-trip bit for bit, NaNs included");
        expect(rows[k].flags == written_flags[k], "the turn flag bits round-trip");
        const std::uint32_t flags = rows[k].flags;
        const bool connected = (flags & dist::FLAG_CONNECTED) != 0u;
        const bool source = (flags & dist::FLAG_CONNECTED_COMMON_SOURCE) != 0u;
        const bool certified = (flags & dist::FLAG_CONNECTED_CERTIFIED) != 0u;
        const bool doubled = (flags & dist::FLAG_CONNECTED_DOUBLED) != 0u;
        expect((!source || certified) && (!certified || doubled) && (!doubled || connected),
               NESTING + " holds in every stored record");
        const auto &values = rows[k].values;
        expect(connected == !std::isnan(values[sm_turns]) &&
                   connected == !std::isnan(values[min_turns]) &&
                   connected == !std::isnan(values[sm_cancelled]) &&
                   connected == !std::isnan(values[wd_cancelled]),
               "the turn values are present exactly on connected records");
        // A theorem rather than a fixture property: if every gate of a path
        // survived, the path itself would join the endpoints in the doubled
        // network.
        if (connected && !doubled)
        {
            expect(values[sm_cancelled] > 0.0 && values[wd_cancelled] > 0.0,
                   "a connected pair the doubled network cuts has a cancelled gate on each "
                   "canonical path");
        }
        all_three += (source && certified && doubled) ? 1u : 0u;
        certified_only += (!source && certified && doubled) ? 1u : 0u;
        connected_none += (connected && !doubled) ? 1u : 0u;
        disconnected += connected ? 0u : 1u;
    }
    expect(all_three > 0 && certified_only > 0 && connected_none > 0 && disconnected > 0,
           "the fixture exercises every bit combination the zig-zag produces");

    // The record file refuses a resume across the setting, both ways.
    auto off = turn_config(false);
    off.n = 4;
    const auto off_names = rec::pair_observable_names(off);
    const std::string off_header = rec::pair_header_text(off, bins, embeddings);
    expect(refuses([&] {
               rec::RecordSink sink;
               sink.open(path, off_header, off_names.size(), draws.size());
           }),
           "a turn record file cannot be extended with the setting off");
    const std::string off_path = directory + "/plain_turn_records.bin";
    {
        rec::RecordSink sink;
        sink.open(off_path, off_header, off_names.size(), 0);
        std::vector<double> values(off_names.size(), 0.5);
        sink.add(0, 0, 0, 0u, values.data());
        sink.close();
    }
    expect(refuses([&] {
               rec::RecordSink sink;
               sink.open(off_path, header, names.size(), 1);
           }),
           "a plain record file cannot be extended with the setting on");
    {
        rec::RecordSink sink;
        sink.open(path, header, names.size(), draws.size());
        expect(sink.inherited_records() == written.size(),
               "a turn record file resumes under the setting it was written with");
        sink.close();
    }
}

// Drive bins the way PairProtocol::measure does: per record (C, E) and the
// three refinements drawn nested inside C, into the plain table and all three
// refined ones from the same draw, plus every statistic load_pairs restores.
void add_trajectory(std::vector<dist::PairBin> &bins, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (auto &bin : bins)
    {
        for (std::uint64_t e = 0; e < bin.embedding_count; ++e)
        {
            for (auto *stats : {&bin.mi, &bin.mn, &bin.g2, &bin.f2, &bin.fmi, &bin.fmn, &bin.fg2,
                                &bin.ff2, &bin.fmn_residual, &bin.n_i, &bin.n_j, &bin.dnn,
                                &bin.rho_n, &bin.abs_rho_n, &bin.rho_n_sq, &bin.i_occ})
            {
                stats->add(unit(rng));
            }
            const bool survives_i = unit(rng) > 0.15;
            const bool survives_j = unit(rng) > 0.15;
            const bool interior = unit(rng) > 0.15;
            const bool connected = survives_i && survives_j && interior;
            const bool doubled = connected && unit(rng) > 0.2;
            const bool certified = doubled && unit(rng) > 0.2;
            const bool source = certified && unit(rng) > 0.3;
            const bool entangled = unit(rng) > 0.5;
            ++bin.conn_records;
            bin.survive_i += survives_i ? 1u : 0u;
            bin.survive_j += survives_j ? 1u : 0u;
            bin.survive_both += (survives_i && survives_j) ? 1u : 0u;
            bin.interior_path += interior ? 1u : 0u;
            bin.component_size.add(50.0 * unit(rng));
            if (connected)
            {
                ++bin.connected;
                bin.ent_given_connected.add(entangled ? 1.0 : 0.0);
                if (entangled)
                {
                    ++bin.connected_ent_positive;
                }
                else
                {
                    ++bin.connected_ent_zero;
                    ++bin.classical_silent;
                    bin.i_occ_connected_classical.add(0.0);
                }
            }
            else
            {
                bin.ent_given_disconnected.add(entangled ? 1.0 : 0.0);
                ++(entangled ? bin.disconnected_ent_positive : bin.disconnected_ent_zero);
            }
            bin.common_source.add(source, entangled);
            bin.certified.add(certified, entangled);
            bin.doubled.add(doubled, entangled);
        }
    }
}

bool same_cells(const dist::RefinedContingency &a, const dist::RefinedContingency &b)
{
    return a.connected == b.connected && a.connected_ent_positive == b.connected_ent_positive &&
           a.connected_ent_zero == b.connected_ent_zero &&
           a.disconnected_ent_positive == b.disconnected_ent_positive &&
           a.disconnected_ent_zero == b.disconnected_ent_zero;
}

// The refusal message, or empty when the action did not refuse.
std::string refusal(const std::function<void()> &action)
{
    try
    {
        action();
    }
    catch (const std::runtime_error &error)
    {
        return error.what();
    }
    return {};
}

// The pair CSV: the counters render in place, round-trip exactly, continue as
// an uninterrupted run would, and every exact invariant refuses a file that
// breaks it -- each edit breaking that invariant alone.
void test_csv_counters_and_resume(const std::string &directory)
{
    auto on = turn_config(true);
    on.realizations = 12;
    on.output_path = directory + "/turn_pairs.csv";
    auto off = turn_config(false);
    off.realizations = 12;
    off.output_path = directory + "/plain_pairs.csv";

    std::mt19937 rng(20261110u);
    auto bins = dist::make_pair_bins(on.n);
    const int before_break = 6;
    for (int t = 0; t < before_break; ++t)
    {
        add_trajectory(bins, rng);
    }

    // ---- every row ends with its bin's fifteen counters, in header order
    const std::string rendered = dist::render_pair_csv(on, bins);
    {
        std::istringstream lines(rendered);
        std::string line;
        std::getline(lines, line);
        for (const auto &bin : bins)
        {
            std::getline(lines, line);
            std::string tail;
            for (const auto *cells : {&bin.common_source, &bin.certified, &bin.doubled})
            {
                for (const std::uint64_t value :
                     {cells->connected, cells->connected_ent_positive, cells->connected_ent_zero,
                      cells->disconnected_ent_positive, cells->disconnected_ent_zero})
                {
                    tail += "," + std::to_string(value);
                }
            }
            expect(line.size() > tail.size() &&
                       line.compare(line.size() - tail.size(), tail.size(), tail) == 0,
                   "a pair row ends with its fifteen refined counters");
        }
    }
    dist::publish_csv(on.output_path, rendered);
    dist::publish_csv(off.output_path, dist::render_pair_csv(off, bins));

    // ---- round trip, and a continuation that ends where an uninterrupted run would
    auto restored = dist::make_pair_bins(on.n);
    const res::Report report = res::load_pairs(on, restored);
    expect(report.loaded && report.completed == before_break,
           "a turn pair CSV resumes under the setting it was written with");
    for (std::size_t b = 0; b < bins.size(); ++b)
    {
        expect(same_cells(restored[b].common_source, bins[b].common_source) &&
                   same_cells(restored[b].certified, bins[b].certified) &&
                   same_cells(restored[b].doubled, bins[b].doubled),
               "every refined counter comes back exactly");
    }
    auto reference = bins;
    const std::mt19937 saved = rng;
    for (int t = before_break; t < on.realizations; ++t)
    {
        add_trajectory(reference, rng);
    }
    rng = saved;
    for (int t = before_break; t < on.realizations; ++t)
    {
        add_trajectory(restored, rng);
    }
    for (std::size_t b = 0; b < bins.size(); ++b)
    {
        expect(same_cells(restored[b].common_source, reference[b].common_source) &&
                   same_cells(restored[b].certified, reference[b].certified) &&
                   same_cells(restored[b].doubled, reference[b].doubled),
               "a resumed run's refined tables end where an uninterrupted run's do");
    }
    {
        auto finished = on;
        finished.output_path = directory + "/finished_turn_pairs.csv";
        dist::publish_csv(finished.output_path, dist::render_pair_csv(finished, restored));
        auto again = dist::make_pair_bins(on.n);
        expect(res::load_pairs(finished, again).completed == on.realizations,
               "the continued file passes every invariant and reads back whole");
    }

    // ---- across the setting, both ways, refused by name
    {
        auto on_reading_off = on;
        on_reading_off.output_path = off.output_path;
        const std::string message = refusal([&] {
            auto fresh = dist::make_pair_bins(on.n);
            (void)res::load_pairs(on_reading_off, fresh);
        });
        expect(message.find("MIPT_DIST_TURN_DIAGNOSTICS=0") != std::string::npos &&
                   message.find("sets it to 1") != std::string::npos,
               "a plain pair CSV resumed with the setting on is refused by name");
        auto off_reading_on = off;
        off_reading_on.output_path = on.output_path;
        const std::string reverse = refusal([&] {
            auto fresh = dist::make_pair_bins(on.n);
            (void)res::load_pairs(off_reading_on, fresh);
        });
        expect(reverse.find("MIPT_DIST_TURN_DIAGNOSTICS=1") != std::string::npos &&
                   reverse.find("sets it to 0") != std::string::npos,
               "a turn pair CSV resumed with the setting off is refused by name");
        auto plain = dist::make_pair_bins(off.n);
        expect(res::load_pairs(off, plain).completed == before_break,
               "a plain pair CSV still resumes with the setting off");
    }

    // ---- provenance
    {
        std::ifstream in(on.output_path);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string quoted = mipt::util::csv_quote(dist::TURN_DIAGNOSTICS_DEFINITION);
        const std::string version = std::to_string(dist::TURN_DIAGNOSTICS_VERSION);
        const std::string metadata = ",1," + version + "," + quoted;
        const auto refused_text = [&](const std::string &replacement, const std::string &what) {
            std::string edited = text;
            std::size_t count = 0;
            for (std::size_t at = edited.find(metadata); at != std::string::npos;
                 at = edited.find(metadata, at + replacement.size()))
            {
                edited.replace(at, metadata.size(), replacement);
                ++count;
            }
            expect(count == bins.size(), what + ": every row carries the turn metadata");
            auto edited_config = on;
            edited_config.output_path = directory + "/edited_turn_pairs.csv";
            dist::publish_csv(edited_config.output_path, edited);
            expect(refuses([&] {
                       auto fresh = dist::make_pair_bins(on.n);
                       (void)res::load_pairs(edited_config, fresh);
                   }),
                   what);
        };
        refused_text(",2," + version + "," + quoted, "a turn_diagnostics value other than 1 is refused");
        refused_text(",1,999," + quoted, "a pair CSV from another turn-definition version is refused");
        refused_text(",1," + version + "," +
                         mipt::util::csv_quote(std::string(dist::TURN_DIAGNOSTICS_DEFINITION) + "x"),
                     "a pair CSV under another turn definition is refused");
    }

    // ---- every exact invariant, each broken alone
    //
    // A bin where every refinement is strict and every cell an edit below
    // moves a record out of is populated; otherwise an edit could not break
    // one identity without breaking another.
    std::size_t b = bins.size();
    for (std::size_t k = 0; k < bins.size() && b == bins.size(); ++k)
    {
        const dist::PairBin &x = bins[k];
        if (x.common_source.connected < x.certified.connected &&
            x.certified.connected < x.doubled.connected && x.doubled.connected < x.connected &&
            x.common_source.disconnected_ent_zero > 0 && x.certified.disconnected_ent_zero > 0 &&
            x.certified.disconnected_ent_positive > 0 && x.certified.connected_ent_zero > 0 &&
            x.doubled.disconnected_ent_zero > 0 && x.disconnected_ent_zero > 0)
        {
            b = k;
        }
    }
    expect(b < bins.size(), "the fixture has a bin where every refinement is strict");
    const auto refused_with = [&](const std::function<void(dist::PairBin &)> &edit,
                                  const std::string &needle, const std::string &what) {
        auto broken = bins;
        edit(broken[b]);
        auto broken_config = on;
        broken_config.output_path = directory + "/broken_turn_pairs.csv";
        dist::publish_csv(broken_config.output_path, dist::render_pair_csv(broken_config, broken));
        const std::string message = refusal([&] {
            auto fresh = dist::make_pair_bins(on.n);
            (void)res::load_pairs(broken_config, fresh);
        });
        expect(message.find(needle) != std::string::npos,
               what + " is refused by the check that names it (got: '" + message + "')");
    };
    refused_with([](dist::PairBin &x) { ++x.common_source.disconnected_ent_zero; },
                 "common_source contingency table summing to",
                 "a refined table that does not partition conn_records");
    refused_with([](dist::PairBin &x) { ++x.doubled.connected; },
                 "connected_doubled_count disagreeing",
                 "a refined connected count that is not its two cells");
    refused_with(
        [](dist::PairBin &x) {
            --x.certified.disconnected_ent_zero;
            ++x.certified.disconnected_ent_positive;
        },
        "entangled records in its certified table",
        "a refined table whose entangled margin differs from the plain one");
    // Nesting, one link at a time: copy the outer table and move one record
    // into the event, which keeps the partition, the connected count and the
    // margin exact and breaks only the link.
    const auto move_into_event = [](dist::RefinedContingency &cells) {
        --cells.disconnected_ent_zero;
        ++cells.connected_ent_zero;
        ++cells.connected;
    };
    refused_with(
        [&](dist::PairBin &x) {
            x.common_source = x.certified;
            move_into_event(x.common_source);
        },
        NESTING, "C_common_source outnumbering C_certified");
    refused_with(
        [&](dist::PairBin &x) {
            x.certified = x.doubled;
            move_into_event(x.certified);
        },
        NESTING, "C_certified outnumbering C_doubled");
    refused_with(
        [&](dist::PairBin &x) {
            x.doubled.connected = x.connected;
            x.doubled.connected_ent_positive = x.connected_ent_positive;
            x.doubled.connected_ent_zero = x.connected_ent_zero;
            x.doubled.disconnected_ent_positive = x.disconnected_ent_positive;
            x.doubled.disconnected_ent_zero = x.disconnected_ent_zero;
            move_into_event(x.doubled);
        },
        NESTING, "C_doubled outnumbering C");
    // One entangled cell alone, with every total consistent.
    refused_with(
        [](dist::PairBin &x) {
            x.common_source = x.certified;
            --x.common_source.connected_ent_zero;
            ++x.common_source.connected_ent_positive;
            --x.common_source.disconnected_ent_positive;
            ++x.common_source.disconnected_ent_zero;
        },
        NESTING, "an entangled C_common_source cell outnumbering C_certified's");
}

void test_validation()
{
    auto config = turn_config(true);
    expect(!refuses_argument([&] { rec::validate_turn_diagnostics(config); }),
           "the turn diagnostics with the gate diagnostics are accepted");
    expect(!refuses_argument([&] { rec::validate_gate_diagnostics(config, true); }),
           "and the configuration passes every gate precondition");
    auto both = config;
    both.k = 0;
    expect(!refuses_argument([&] { rec::validate_turn_diagnostics(both); }) &&
               !refuses_argument([&] { rec::validate_gate_diagnostics(both, true); }),
           "k=0 is accepted");

    auto no_gate = config;
    no_gate.gate_path_diagnostics = false;
    std::string message;
    try
    {
        rec::validate_turn_diagnostics(no_gate);
    }
    catch (const std::invalid_argument &error)
    {
        message = error.what();
    }
    expect(message.find("MIPT_DIST_TURN_DIAGNOSTICS=1") != std::string::npos &&
               message.find("MIPT_DIST_GATE_DIAGNOSTICS=1") != std::string::npos,
           "the turn diagnostics without the gate diagnostics are refused, naming both");
    for (const auto type : {mipt::CircuitType::Haar, mipt::CircuitType::MMS,
                            mipt::CircuitType::QubitRPPU})
    {
        auto other = no_gate;
        other.type = type;
        expect(refuses_argument([&] { rec::validate_turn_diagnostics(other); }),
               "the missing gate setting is refused whatever the circuit");
    }
    // The preconditions the turns inherit are the gate validator's to refuse.
    auto triangles = config;
    triangles.k = 3;
    expect(!refuses_argument([&] { rec::validate_turn_diagnostics(triangles); }) &&
               refuses_argument([&] { rec::validate_gate_diagnostics(triangles, true); }),
           "k=3 is refused through the gate preconditions, not twice");
    auto off = config;
    off.turn_diagnostics = false;
    off.gate_path_diagnostics = false;
    off.type = mipt::CircuitType::Haar;
    off.k = 3;
    off.connectivity = false;
    expect(!refuses_argument([&] { rec::validate_turn_diagnostics(off); }),
           "the setting off imposes nothing");
}

void run(const std::string &directory)
{
    test_observable_ordering();
    test_diagnostic_tail_pairing();
    test_record_values();
    test_flag_bits();
    test_headers();
    test_binary_round_trip(directory);
    test_csv_counters_and_resume(directory);
    test_validation();
    std::filesystem::remove_all(directory);
}
} // namespace dist_turn_tests

void test_dist_turn_diagnostics()
{
    dist_turn_tests::run(dist_turn_tests::scratch_directory());
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
    test_dist_gate_diagnostics();
    test_dist_turn_diagnostics();
    dist_channel_tests::run();

    std::cout << "dist_metrics_tests: PASS\n";
    return 0;
}
