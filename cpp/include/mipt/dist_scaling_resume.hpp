#pragma once

// Resuming an interrupted dist_scaling.exe run.
//
// Both protocols keep their whole state in aggregation bins and republish the
// CSV as a projection of them every progress interval. That projection is
// invertible -- every bin field is either a column or implied by one -- so the
// output file is the checkpoint, with no side-car state. See
// util/resume_csv.hpp for why that is the shape all three resumable
// executables use.
//
// Two things are recovered rather than stored:
//
//   * The completed realization count. Each trajectory contributes a fixed,
//     geometry-independent number of records to each bin -- `embedding_count`
//     pairs for k=2, and `min(MIPT_DIST_EMBEDDINGS_PER_GEOMETRY, embeddings)`
//     triangles for k=3 -- so dividing a bin's sample count by that number
//     gives the trajectories behind it. Every bin must agree, which doubles as
//     an integrity check on the file.
//
//   * The SDP schedule position. `SdpSettings::selects` is a pure function of
//     `bin.records` and `bin.gmn.requested`, and both come back: `records` is
//     the TMI sample count (every record feeds the TMI), and `requested` is an
//     explicit column.
//
// `SdpStats::positive` needs neither: it is its own column, and the solves
// lost in flight contributed no value, so nothing about it is rolled back
// alongside `requested`.
//
// The one thing that genuinely needs repair is GMN solves that were in flight
// when the run died. Those were counted in `requested` at submit time but their
// values never arrived, so a naive restore would retire schedule slots that
// produced no sample -- and because the prefiltered zeros *are* recorded while
// the in-flight non-zeros are not, that would bias the reported mean downward.
// The queue's invariant `value.count + solver_failures == requested` makes the
// loss visible, so `requested` is rolled back to what actually landed and the
// schedule re-offers those slots. Nothing is double-counted: the dropped
// records contributed no value.
//
// Trajectories are i.i.d. draws -- dist_scaling.exe never calls
// CircuitWorkspace1D::seed, so its circuits come from std::random_device -- and
// a resumed run simply draws more of them. There is no seed stream to continue
// the way free_energy.exe continues one, and none is needed.
//
// Nothing here depends on CUDA-Q, MOSEK, or a GPU, so it is exercised by
// `make test-dist`.

#include "mipt/dist_scaling_csv.hpp"
#include "mipt/util/resume_csv.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace mipt::dist::resume
{
namespace csv = util::resume;

// Re-issuing the identical command continues an interrupted run. `=0` restores
// the old behaviour of overwriting whatever the output path already holds.
inline bool enabled()
{
    return env::boolean("MIPT_DIST_RESUME", true);
}

struct Report
{
    long completed = 0;
    // GMN/fGMN schedule slots reclaimed from solves that were still in flight.
    std::uint64_t reclaimed_gmn_slots = 0;
    std::uint64_t reclaimed_fgmn_slots = 0;
    // The largest per-bin fraction of requested SDP records whose values never
    // arrived. Those records are exactly the ones the zero prefilter did *not*
    // settle -- the non-zero ones -- so the checkpoint's GMN mean is skewed
    // toward zero by this fraction. Reclaiming their schedule slots fixes the
    // sampling from here on but cannot recover the lost values: the density
    // matrices they were computed from are long gone.
    double worst_lost_sdp_fraction = 0.0;
    bool loaded = false;
};

// How many records one trajectory folds into a bin holding `embeddings`
// lattice embeddings. Mirrors sample_trajectory_triangles exactly.
inline std::uint64_t records_per_trajectory(std::uint64_t embeddings, long per_geometry)
{
    if (per_geometry <= 0 || static_cast<std::uint64_t>(per_geometry) >= embeddings)
    {
        return embeddings;
    }
    return static_cast<std::uint64_t>(per_geometry);
}

// Trajectories behind `samples` records, refusing anything that is not a whole
// number of them or that disagrees with the other bins.
inline void fold_completed(
    std::uint64_t samples,
    std::uint64_t per_trajectory,
    const std::string &path,
    std::size_t row,
    long &completed)
{
    if (per_trajectory == 0)
    {
        csv::refuse(path + " line " + std::to_string(row + 2) + " has no embeddings.");
    }
    if (samples % per_trajectory != 0)
    {
        csv::refuse(
            path + " line " + std::to_string(row + 2) + " holds " +
            std::to_string(samples) + " samples, which is not a whole number of "
            "trajectories at " + std::to_string(per_trajectory) +
            " record(s) each. The file was written with a different "
            "MIPT_DIST_EMBEDDINGS_PER_GEOMETRY, or has been edited.");
    }
    const long implied = static_cast<long>(samples / per_trajectory);
    if (completed < 0)
    {
        completed = implied;
    }
    else if (implied != completed)
    {
        csv::refuse(
            path + " line " + std::to_string(row + 2) + " implies " +
            std::to_string(implied) + " completed trajectories but earlier rows "
            "imply " + std::to_string(completed) +
            ". Every geometry is measured on every trajectory, so the counts "
            "cannot differ.");
    }
}

inline void require_schema(
    const csv::CsvTable &table, const std::string &expected_header, const std::string &path)
{
    // The header encodes the fermionic columns, so this also catches a resume
    // whose circ_type changed trace conventions without changing the filename.
    std::string expected = expected_header;
    if (!expected.empty() && expected.back() == '\n')
    {
        expected.pop_back();
    }
    if (table.header_line() != expected)
    {
        csv::refuse(
            path + " has a different column layout than this run writes, so it "
            "came from another version of dist_scaling.exe or a different "
            "circuit type. Finish it with that binary, move it aside, or set "
            "MIPT_DIST_RESUME=0 to overwrite it.");
    }
}

inline void require_metadata(
    const csv::CsvTable &table, std::size_t row, const RunConfig &config, const std::string &path)
{
    csv::require_same("N", static_cast<long>(config.n), table.integer(row, "N"), path);
    csv::require_same(
        "circ_type", static_cast<long>(config.type), table.integer(row, "circ_type"), path);
    csv::require_same(
        "realizations", static_cast<long>(config.realizations),
        table.integer(row, "realizations"), path);
    csv::require_same("p", config.p, table.real(row, "p"), path);
    csv::require_same(
        "periods", static_cast<long>(config.periods), table.integer(row, "periods"), path);
    csv::require_same("k", static_cast<long>(config.k), table.integer(row, "k"), path);
}

// --- k=2 -------------------------------------------------------------------

inline Report load_pairs(const RunConfig &config, std::vector<PairBin> &bins)
{
    Report report;
    if (!file_exists(config.output_path))
    {
        return report;
    }
    const auto table = csv::CsvTable::read(config.output_path);
    if (table.empty())
    {
        return report;
    }
    const std::string &path = config.output_path;
    require_schema(table, pair_csv_header(config), path);
    if (table.row_count() != bins.size())
    {
        csv::refuse(
            path + " holds " + std::to_string(table.row_count()) +
            " separation bins, but N=" + std::to_string(config.n) + " has " +
            std::to_string(bins.size()) + '.');
    }

    long completed = -1;
    const bool fermionic = config.fermionic_outputs();
    for (std::size_t row = 0; row < bins.size(); ++row)
    {
        require_metadata(table, row, config, path);
        PairBin &bin = bins[row];
        csv::require_same(
            "separation", static_cast<long>(bin.separation),
            table.integer(row, "separation"), path);
        csv::require_same(
            "embedding_count", static_cast<long>(bin.embedding_count),
            table.integer(row, "embedding_count"), path);

        bin.mi = table.stats_by_stderr(row, "mi");
        bin.mn = table.stats_by_stderr(row, "mn");
        bin.mn_positive = table.counter(row, "mn_positive_count");
        if (bin.mn_positive > bin.mn.count)
        {
            csv::refuse(
                path + " line " + std::to_string(row + 2) +
                " counts more positive negativities than samples.");
        }
        if (bin.mi.count != bin.mn.count)
        {
            csv::refuse(
                path + " line " + std::to_string(row + 2) +
                " has mi_samples and mn_samples that disagree; both are recorded "
                "for every pair of every trajectory.");
        }
        if (fermionic)
        {
            bin.fmi = table.stats_by_stderr(row, "fmi");
            bin.fmn = table.stats_by_stderr(row, "fmn");
            bin.fmn_positive = table.counter(row, "fmn_positive_count");
            if (bin.fmi.count != bin.mi.count || bin.fmn.count != bin.mi.count)
            {
                csv::refuse(
                    path + " line " + std::to_string(row + 2) +
                    " has fermionic sample counts that disagree with the "
                    "ordinary-trace ones; both conventions are measured on the "
                    "same records.");
            }
        }

        fold_completed(bin.mi.count, bin.embedding_count, path, row, completed);
    }

    report.completed = completed < 0 ? 0 : completed;
    report.loaded = report.completed > 0;
    return report;
}

// --- k=3 -------------------------------------------------------------------

// Restores one GMN-family measure and reclaims the schedule slots of solves
// that never came back. See the header comment for why this is not optional.
inline void restore_sdp(
    const csv::CsvTable &table,
    std::size_t row,
    std::string_view prefix,
    SdpStats &stats,
    const std::string &path,
    std::uint64_t &reclaimed,
    double &worst_lost_fraction)
{
    const std::string stem(prefix);
    stats.value = table.stats_by_stderr(row, prefix);
    stats.positive = table.counter(row, stem + "_positive_count");
    stats.requested = table.counter(row, stem + "_requested_count");
    stats.zero_prefiltered = table.counter(row, stem + "_zero_prefilter_count");
    stats.solver_failures = table.counter(row, stem + "_solver_failure_count");

    const std::uint64_t landed = stats.value.count + stats.solver_failures;
    if (landed > stats.requested)
    {
        csv::refuse(
            path + " line " + std::to_string(row + 2) + " reports " +
            std::to_string(landed) + " settled " + stem + " record(s) against " +
            std::to_string(stats.requested) +
            " requested. The counters are inconsistent; the file has been edited.");
    }
    if (stats.zero_prefiltered > stats.value.count)
    {
        csv::refuse(
            path + " line " + std::to_string(row + 2) + " prefiltered more " +
            stem + " records than it recorded values for.");
    }
    // A prefiltered record is non-positive by construction, so the positives
    // can only have come from the solved remainder.
    if (stats.positive > stats.value.count - stats.zero_prefiltered)
    {
        csv::refuse(
            path + " line " + std::to_string(row + 2) + " counts " +
            std::to_string(stats.positive) + " positive " + stem +
            " record(s) against " +
            std::to_string(stats.value.count - stats.zero_prefiltered) +
            " that were actually solved.");
    }
    const std::uint64_t lost = stats.requested - landed;
    if (lost > 0 && stats.requested > 0)
    {
        worst_lost_fraction = std::max(
            worst_lost_fraction,
            static_cast<double>(lost) / static_cast<double>(stats.requested));
    }
    reclaimed += lost;
    stats.requested = landed;
}

inline Report load_triples(
    const RunConfig &config, std::vector<TripleBin> &bins, long per_geometry)
{
    Report report;
    if (!file_exists(config.output_path))
    {
        return report;
    }
    const auto table = csv::CsvTable::read(config.output_path);
    if (table.empty())
    {
        return report;
    }
    const std::string &path = config.output_path;
    require_schema(table, triple_csv_header(config), path);
    if (table.row_count() != bins.size())
    {
        csv::refuse(
            path + " holds " + std::to_string(table.row_count()) +
            " geometry bins, but this run enumerates " + std::to_string(bins.size()) +
            ". The checkpoint was made with a different N or B_min.");
    }

    long completed = -1;
    const bool fermionic = config.fermionic_outputs();
    for (std::size_t row = 0; row < bins.size(); ++row)
    {
        require_metadata(table, row, config, path);
        TripleBin &bin = bins[row];
        csv::require_same(
            "geometry_id", static_cast<long>(row), table.integer(row, "geometry_id"), path);
        for (int leg = 0; leg < 3; ++leg)
        {
            const std::string column = "distance_" + std::to_string(leg + 1);
            csv::require_same(
                column, static_cast<long>(bin.separations[static_cast<std::size_t>(leg)]),
                table.integer(row, column), path);
        }
        csv::require_same(
            "triangle_balance_cutoff", config.triangle_balance_cutoff,
            table.real(row, "triangle_balance_cutoff"), path);
        csv::require_same(
            "embedding_count", static_cast<long>(bin.embedding_count),
            table.integer(row, "embedding_count"), path);

        bin.tmi = table.stats_by_stderr(row, "tmi");
        bin.average_mi = table.stats_by_stderr(row, "average_mi");
        bin.min_bipartite_negativity = table.stats_by_stderr(row, "min_bipneg");
        bin.joint_purity = table.stats_by_stderr(row, "joint_purity");
        bin.mean_single_purity = table.stats_by_stderr(row, "mean_single_purity");
        restore_sdp(table, row, "gmn", bin.gmn, path, report.reclaimed_gmn_slots,
                    report.worst_lost_sdp_fraction);

        if (fermionic)
        {
            bin.ftmi = table.stats_by_stderr(row, "ftmi");
            bin.faverage_mi = table.stats_by_stderr(row, "faverage_mi");
            bin.min_fermionic_bipartite_negativity =
                table.stats_by_stderr(row, "min_fbipneg");
            bin.fjoint_purity = table.stats_by_stderr(row, "fjoint_purity");
            bin.fmean_single_purity = table.stats_by_stderr(row, "fmean_single_purity");
            restore_sdp(table, row, "fgmn", bin.fgmn, path, report.reclaimed_fgmn_slots,
                        report.worst_lost_sdp_fraction);
            if (bin.ftmi.count != bin.tmi.count)
            {
                csv::refuse(
                    path + " line " + std::to_string(row + 2) +
                    " has ftmi_samples and tmi_samples that disagree; both "
                    "conventions are measured on the same records.");
            }
        }

        // Every record feeds the TMI, so its sample count *is* the record
        // counter the SDP schedule is a function of.
        bin.records = bin.tmi.count;
        if (bin.average_mi.count != bin.records ||
            bin.min_bipartite_negativity.count != bin.records)
        {
            csv::refuse(
                path + " line " + std::to_string(row + 2) +
                " has three-party sample counts that disagree; all of them are "
                "recorded for every triangle of every trajectory.");
        }

        fold_completed(
            bin.records,
            records_per_trajectory(
                static_cast<std::uint64_t>(bin.embedding_count), per_geometry),
            path,
            row,
            completed);
    }

    report.completed = completed < 0 ? 0 : completed;
    report.loaded = report.completed > 0;
    return report;
}
} // namespace mipt::dist::resume
