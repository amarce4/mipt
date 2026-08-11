#pragma once

// Resuming an interrupted entropy.exe run.
//
// entropy.exe writes a single aggregated CSV -- one row per block size L_A,
// carrying mean, sample standard deviation, standard error and the completed
// trajectory count -- and rewrites it after every trajectory. That file is the
// whole checkpoint: `m2 = stddev^2 * (count - 1)` restores the Welford triple
// the run was accumulating, so a resumed run continues the same averages
// instead of starting fresh ones. See util/resume_csv.hpp for why the CSVs are
// the checkpoint rather than a side-car file.
//
// Unlike free_energy.exe there is no seed to recover, and none is wanted:
// entropy.exe never calls CircuitWorkspace1D::seed, so its trajectories are
// drawn from std::random_device and are i.i.d. by construction. A resumed run
// simply draws more of them, which is exactly what the Monte Carlo average
// wants. The consequence is that a resumed run cannot be compared trajectory
// by trajectory against an uninterrupted one, only statistically.
//
// The S1 columns need care. A block whose von Neumann entropy was skipped has
// no samples and is written as NaN, never 0, so its accumulator must come back
// empty rather than as a zero-valued sample. The count is not a column, but it
// does not need to be: S1 is skipped exactly when `L_A > MIPT_ENTROPY_S1_MAX_LA`
// (or the whole path is off), which is a property of the settings and not of
// any trajectory, so a block either has every trajectory's sample or none.
// Which of the two holds is read off the file's own NaNs rather than off the
// current settings -- and then cross-checked against them, because resuming
// under a *different* S1 cap would otherwise leave one column with more samples
// than another with no way to tell from the file.
//
// Nothing here depends on CUDA-Q or a GPU, so it is exercised by
// `make test-core`.

#include "mipt/types.hpp"
#include "mipt/util/resume_csv.hpp"
#include "mipt/util/stats.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mipt::entropy::resume
{
using util::RunningStats;
namespace csv = util::resume;

// The exact header entropy.exe writes. Kept here so the writer, the checkpoint
// reader and the tests share one definition and a schema change cannot be made
// on one side only.
inline constexpr std::string_view kResultsHeader =
    "N,periods,p,requested_realizations,completed_realizations,circ_type,"
    "circuit_name,fermionic_trace,L_A,x,ln_x,"
    "S1_mean,S1_stddev,S1_stderr,S2_mean,S2_stddev,S2_stderr,"
    "subsystems_per_realization,total_subsystems";

// The pre-2026-08-05 schema, before S1 was added. Those files hold only the
// Renyi-2 entropy, so they can still be resumed -- their S1 accumulators just
// start empty, and the S1 columns of the finished file describe fewer
// trajectories than the S2 ones. That is refused unless the S1 path is off,
// since silently mixing the two would be worse than making the user say so.
inline constexpr std::string_view kLegacyS2OnlyHeader =
    "N,periods,p,requested_realizations,completed_realizations,circ_type,"
    "circuit_name,fermionic_trace,L_A,x,ln_x,"
    "S2_mean,S2_stddev,S2_stderr,subsystems_per_realization,total_subsystems";

struct RunIdentity
{
    int n = 0;
    int periods = 0;
    double p = 0.0;
    long requested_realizations = 0;
    int circ_type = 0;
};

struct Checkpoint
{
    long completed = 0;
    // Indexed by L_A, sized n + 1, exactly like the live accumulators.
    std::vector<RunningStats> s1;
    std::vector<RunningStats> s2;
};

// A count recovered from the two spread columns: stderr = stddev / sqrt(count).
// Used only to cross-check the count implied by the run's bookkeeping, which is
// worth doing because it is free and catches a hand-edited or concatenated file
// that the header check alone would pass.
inline bool count_matches_spread(double stddev, double standard_error, std::uint64_t count)
{
    if (!(stddev > 0.0) || !(standard_error > 0.0) || count < 2)
    {
        return true;  // no information in a zero spread
    }
    const double implied = (stddev / standard_error) * (stddev / standard_error);
    return std::abs(implied - static_cast<double>(count)) <=
           1.0e-6 * static_cast<double>(count);
}

// `max_s1_block` is the app's resolved cap: the largest L_A whose von Neumann
// entropy this run computes, or 0 when the path is disabled entirely.
inline Checkpoint load(
    const std::string &path,
    const RunIdentity &expected,
    int max_s1_block)
{
    const auto table = csv::CsvTable::read(path);
    Checkpoint checkpoint;
    checkpoint.s1.assign(static_cast<std::size_t>(expected.n + 1), RunningStats{});
    checkpoint.s2.assign(static_cast<std::size_t>(expected.n + 1), RunningStats{});
    if (table.empty())
    {
        return checkpoint;  // nothing was ever written
    }

    const std::string header = table.header_line();
    const bool legacy = header == kLegacyS2OnlyHeader;
    if (!legacy && header != kResultsHeader)
    {
        csv::refuse(
            path + " has a different column layout than this binary writes, so "
            "it came from another version of entropy.exe. Finish it with that "
            "binary, move it aside, or set MIPT_ENTROPY_RESUME=0 to overwrite "
            "it.");
    }
    if (legacy && max_s1_block >= 2)
    {
        csv::refuse(
            path + " is a pre-2026-08-05 file with no S1 columns, but this run "
            "computes the von Neumann entropy. Resuming would leave S1 averaged "
            "over fewer trajectories than S2. Re-run with MIPT_ENTROPY_S1=0 to "
            "continue it as an S2-only scan, or move it aside.");
    }

    const int max_kept = expected.n / 2;
    const std::size_t expected_rows = static_cast<std::size_t>(max_kept - 1);
    if (table.row_count() != expected_rows)
    {
        csv::refuse(
            path + " holds " + std::to_string(table.row_count()) +
            " block rows, but N=" + std::to_string(expected.n) + " has " +
            std::to_string(expected_rows) +
            " (L_A = 2..N/2). The checkpoint belongs to a different N.");
    }

    long completed = -1;
    for (std::size_t row = 0; row < table.row_count(); ++row)
    {
        csv::require_same("N", static_cast<long>(expected.n), table.integer(row, "N"), path);
        csv::require_same(
            "periods", static_cast<long>(expected.periods), table.integer(row, "periods"), path);
        csv::require_same("p", expected.p, table.real(row, "p"), path);
        csv::require_same(
            "circ_type", static_cast<long>(expected.circ_type),
            table.integer(row, "circ_type"), path);
        csv::require_same(
            "realizations", expected.requested_realizations,
            table.integer(row, "requested_realizations"), path);

        const long kept = table.integer(row, "L_A");
        if (kept != static_cast<long>(row) + 2)
        {
            csv::refuse(
                path + " line " + std::to_string(row + 2) + " has L_A=" +
                std::to_string(kept) + " where " + std::to_string(row + 2) +
                " was expected; the rows must run L_A = 2..N/2 in order.");
        }

        const long row_completed = table.integer(row, "completed_realizations");
        if (row_completed < 0)
        {
            csv::refuse(path + " reports a negative completed_realizations.");
        }
        if (completed < 0)
        {
            completed = row_completed;
        }
        else if (row_completed != completed)
        {
            csv::refuse(
                path + " line " + std::to_string(row + 2) +
                " reports completed_realizations=" + std::to_string(row_completed) +
                " but earlier rows report " + std::to_string(completed) +
                ". Every block is measured on every trajectory, so the counts "
                "cannot differ.");
        }

        const auto count = static_cast<std::uint64_t>(row_completed);
        const double s2_mean = table.real(row, "S2_mean");
        const double s2_stddev = table.real(row, "S2_stddev");
        if (count > 0 && !std::isfinite(s2_mean))
        {
            csv::refuse(
                path + " line " + std::to_string(row + 2) + " has a non-finite "
                "S2_mean over " + std::to_string(count) +
                " trajectories; the file is not a usable checkpoint.");
        }
        if (!count_matches_spread(s2_stddev, table.real(row, "S2_stderr"), count))
        {
            csv::refuse(
                path + " line " + std::to_string(row + 2) +
                " has S2_stddev and S2_stderr that do not agree with "
                "completed_realizations=" + std::to_string(count) +
                ". The file has been edited or two runs were concatenated.");
        }
        checkpoint.s2[static_cast<std::size_t>(kept)] =
            csv::stats_from_stddev(s2_mean, s2_stddev, count);

        if (legacy)
        {
            continue;
        }

        // Read the S1 schedule off the file, then hold the current settings to
        // it. A block is either sampled on every trajectory or on none.
        const double s1_mean = table.real(row, "S1_mean");
        const double s1_stddev = table.real(row, "S1_stddev");
        const bool file_has_s1 = std::isfinite(s1_mean);
        const bool run_has_s1 = kept <= static_cast<long>(max_s1_block);
        if (count > 0 && file_has_s1 != run_has_s1)
        {
            csv::refuse(
                path + " records S1 for L_A up to " +
                std::to_string(file_has_s1 ? kept : kept - 1) +
                " but this run is configured for L_A up to " +
                std::to_string(max_s1_block) +
                ". Resuming would leave the S1 columns averaged over different "
                "numbers of trajectories. Re-run with the MIPT_ENTROPY_S1 / "
                "MIPT_ENTROPY_S1_MAX_LA values the original run used.");
        }
        if (file_has_s1 &&
            !count_matches_spread(s1_stddev, table.real(row, "S1_stderr"), count))
        {
            csv::refuse(
                path + " line " + std::to_string(row + 2) +
                " has S1_stddev and S1_stderr that do not agree with "
                "completed_realizations=" + std::to_string(count) + '.');
        }
        checkpoint.s1[static_cast<std::size_t>(kept)] =
            csv::stats_from_stddev(s1_mean, s1_stddev, file_has_s1 ? count : 0);
    }

    checkpoint.completed = completed < 0 ? 0 : completed;
    return checkpoint;
}
} // namespace mipt::entropy::resume
