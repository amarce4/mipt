#pragma once

// Resuming an interrupted free_energy.exe run.
//
// A long run is checkpointed by the two CSVs it writes: `<stem>_samples.csv`
// gains one flushed row per finished trajectory, and `<stem>_timeseries.csv`
// is rewritten (temp file + rename) right after it. Between them they already
// hold everything a restart needs, so resuming reads the run's own output
// rather than a side-car checkpoint -- which is what lets a run interrupted by
// a binary that had no resume support at all be picked up by one that does.
//
// Three things make that work:
//
//   * The per-timestep accumulators are Welford triples (count, mean, m2), and
//     the time-series CSV stores mean and *sample* standard deviation at
//     17 significant digits. m2 = stddev^2 * (count - 1) recovers the third
//     member to within a couple of ulp, which is far below the 1/sqrt(count)
//     statistical error on anything derived from it. The counts are not in the
//     per-column output but are implied: `completed_realizations` for the two
//     free-energy columns, and the explicit `S_half_count` for the entropy.
//
//   * splitmix64 is a bijection, so the master seed is recoverable from any
//     single `trajectory_seed` column together with its trajectory index. A
//     resumed run therefore continues the *same* seed stream instead of
//     starting an unrelated one, and cannot re-simulate a trajectory the
//     interrupted run already recorded.
//
//   * A trajectory is appended to the samples file before the time series is
//     rewritten, so a kill between the two leaves the samples file one row
//     ahead. Resuming keeps that row (it is real data) and restarts the
//     time-series accumulator from what the checkpoint actually holds, so the
//     aggregate can end up missing at most one trajectory that the samples
//     file has. `completed_realizations` is then written from the accumulator
//     itself rather than from the loop counter, so the column stays truthful.
//
// Nothing here depends on CUDA-Q, MOSEK, or a GPU, so it is exercised by
// `make test-core`.

#include "mipt/util/stats.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mipt::free_energy::resume
{
using util::RunningStats;

// The live per-timestep accumulator, shared with the checkpoint reader so the
// two cannot drift apart.
struct TimeStats
{
    RunningStats cumulative_f;
    RunningStats f_over_tl;
    RunningStats half_entropy;
};

// The exact header line of each CSV. free_energy.exe writes these and a resume
// compares against them byte for byte, so keeping one definition is what makes
// a future schema change fail loudly instead of appending mismatched rows to a
// file written by an older binary.
constexpr std::string_view kSamplesHeader =
    "N,p,circ_type,circuit_name,init_state,trajectory,parity_sector,"
    "trajectory_seed,equil_timesteps,production_timesteps,total_timesteps,"
    "measurements_total,measurements_production,F_total,F_production,"
    "production_rate_endpoint,production_slope,free_energy_density_tilde";
constexpr std::string_view kTimeseriesHeader =
    "N,p,circ_type,circuit_name,init_state,requested_realizations,"
    "completed_realizations,t,t_over_L,phase,F_mean,F_stddev,F_stderr,"
    "F_over_tL_mean,F_over_tL_stddev,F_over_tL_stderr,S_half_mean,"
    "S_half_stddev,S_half_stderr,S_half_count";

// --- seed stream -----------------------------------------------------------

inline std::uint64_t splitmix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

// Multiplicative inverse modulo 2^64 by Newton iteration. Computed rather than
// spelled out as a magic constant so there is nothing to mistype.
inline std::uint64_t odd_inverse_mod_2_64(std::uint64_t odd)
{
    std::uint64_t inverse = 1;
    for (int step = 0; step < 6; ++step)
    {
        inverse *= 2ULL - odd * inverse;
    }
    return inverse;
}

// Inverse of `value ^= value >> shift`. Each pass fixes another `shift` bits,
// starting from the top `shift` which the xor left untouched.
inline std::uint64_t undo_xor_shift_right(std::uint64_t value, int shift)
{
    std::uint64_t result = value;
    for (int recovered = shift; recovered < 64; recovered += shift)
    {
        result = value ^ (result >> shift);
    }
    return result;
}

inline std::uint64_t splitmix64_inverse(std::uint64_t value)
{
    value = undo_xor_shift_right(value, 31);
    value *= odd_inverse_mod_2_64(0x94d049bb133111ebULL);
    value = undo_xor_shift_right(value, 27);
    value *= odd_inverse_mod_2_64(0xbf58476d1ce4e5b9ULL);
    value = undo_xor_shift_right(value, 30);
    return value - 0x9e3779b97f4a7c15ULL;
}

// free_energy.exe derives `trajectory_seed = splitmix64(master ^ trajectory)`.
inline std::uint64_t trajectory_seed_for(std::uint64_t master_seed, long trajectory)
{
    return splitmix64(master_seed ^ static_cast<std::uint64_t>(trajectory));
}

inline std::uint64_t recover_master_seed(std::uint64_t trajectory_seed, long trajectory)
{
    return splitmix64_inverse(trajectory_seed) ^
           static_cast<std::uint64_t>(trajectory);
}

// --- half-chain entropy schedule -------------------------------------------

// The single definition of "is the half-chain entropy sampled at this t". The
// trajectory loop and the checkpoint validator must agree, because a resume
// that silently mixes two entropy schedules produces a column whose per-t
// sample counts differ with no way to tell from the file.
inline bool records_half_entropy(int t, bool enabled, int entropy_max_t, int entropy_stride)
{
    if (!enabled || entropy_max_t < 0)
    {
        return false;
    }
    if (t == 0)
    {
        return true;
    }
    if (t > entropy_max_t)
    {
        return false;
    }
    return t % entropy_stride == 0 || t == entropy_max_t;
}

// --- CSV reading -----------------------------------------------------------

// Splits one CSV row, honouring the double-quoted circuit-name field written
// by util::csv_quote (doubled quotes escape a literal quote).
inline std::vector<std::string> split_csv_row(std::string_view row)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < row.size(); ++i)
    {
        const char c = row[i];
        if (quoted)
        {
            if (c != '"')
            {
                field.push_back(c);
            }
            else if (i + 1 < row.size() && row[i + 1] == '"')
            {
                field.push_back('"');
                ++i;
            }
            else
            {
                quoted = false;
            }
            continue;
        }
        if (c == '"')
        {
            quoted = true;
        }
        else if (c == ',')
        {
            fields.push_back(std::move(field));
            field.clear();
        }
        else
        {
            field.push_back(c);
        }
    }
    fields.push_back(std::move(field));
    return fields;
}

inline std::string read_whole_file(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("Could not open for resume: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

inline double parse_double_field(
    const std::string &text, const std::string &context)
{
    try
    {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size())
        {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(
            "Resume: could not parse '" + text + "' as a number in " + context);
    }
}

inline long parse_long_field(const std::string &text, const std::string &context)
{
    try
    {
        std::size_t consumed = 0;
        const long value = std::stol(text, &consumed);
        if (consumed != text.size())
        {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(
            "Resume: could not parse '" + text + "' as an integer in " + context);
    }
}

inline std::uint64_t parse_uint64_field(
    const std::string &text, const std::string &context)
{
    try
    {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed);
        if (consumed != text.size())
        {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<std::uint64_t>(value);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(
            "Resume: could not parse '" + text + "' as an unsigned integer in " +
            context);
    }
}

// --- run identity ----------------------------------------------------------

// The parameters both CSVs repeat on every row. A mismatch means the file on
// disk is a different run that happens to share a name, which is exactly the
// case a resume must refuse rather than silently pool.
struct RunIdentity
{
    int n = 0;
    double p = 0.0;
    int circ_type = 0;
    int init_state = 0;
    long requested_realizations = 0;
};

inline void require_identity_match(
    const RunIdentity &expected,
    const RunIdentity &found,
    const std::string &where)
{
    const auto fail = [&](const std::string &what,
                          const std::string &was,
                          const std::string &now) {
        throw std::runtime_error(
            "Resume refused: " + where + " was produced with " + what + '=' +
            was + ", but this run uses " + now +
            ". Move the existing files aside, or set "
            "MIPT_FREE_ENERGY_RESUME=0 to overwrite them.");
    };
    if (found.n != expected.n)
    {
        fail("N", std::to_string(found.n), std::to_string(expected.n));
    }
    // p round-trips exactly through the 17-digit output; the tolerance only
    // absorbs a shortest-round-trip writer, never a genuinely different rate.
    if (std::abs(found.p - expected.p) >
        1.0e-12 * std::max(1.0, std::abs(expected.p)))
    {
        fail("p", std::to_string(found.p), std::to_string(expected.p));
    }
    if (found.circ_type != expected.circ_type)
    {
        fail("circ_type",
             std::to_string(found.circ_type),
             std::to_string(expected.circ_type));
    }
    if (found.init_state != expected.init_state)
    {
        fail("init_state",
             std::to_string(found.init_state),
             std::to_string(expected.init_state));
    }
    if (expected.requested_realizations > 0 && found.requested_realizations > 0 &&
        found.requested_realizations != expected.requested_realizations)
    {
        fail("realizations",
             std::to_string(found.requested_realizations),
             std::to_string(expected.requested_realizations));
    }
}

// --- samples checkpoint ----------------------------------------------------

struct SamplesCheckpoint
{
    long completed = 0;             // complete, well-formed data rows
    long next_trajectory = 0;       // trajectory index the resumed loop starts at
    std::uint64_t master_seed = 0;  // recovered from the last row
    bool has_master_seed = false;
    std::uintmax_t valid_bytes = 0;   // end of the last complete row
    std::uintmax_t file_bytes = 0;    // actual file size
};

inline bool has_partial_tail(const SamplesCheckpoint &checkpoint)
{
    return checkpoint.file_bytes > checkpoint.valid_bytes;
}

// Column layout of the samples CSV.
constexpr std::size_t kSamplesColumns = 18;
constexpr std::size_t kSampleColumnN = 0;
constexpr std::size_t kSampleColumnP = 1;
constexpr std::size_t kSampleColumnCircType = 2;
constexpr std::size_t kSampleColumnInitState = 4;
constexpr std::size_t kSampleColumnTrajectory = 5;
constexpr std::size_t kSampleColumnSeed = 7;

inline SamplesCheckpoint scan_samples(
    const std::filesystem::path &path,
    std::string_view expected_header,
    const RunIdentity &expected)
{
    const std::string content = read_whole_file(path);
    SamplesCheckpoint checkpoint;
    checkpoint.file_bytes = content.size();

    std::size_t cursor = 0;
    const auto next_line = [&](std::string_view &line) -> bool {
        if (cursor >= content.size())
        {
            return false;
        }
        const std::size_t newline = content.find('\n', cursor);
        if (newline == std::string::npos)
        {
            // Unterminated tail: a kill during the flush. Not a complete row.
            return false;
        }
        line = std::string_view(content).substr(cursor, newline - cursor);
        cursor = newline + 1;
        return true;
    };

    std::string_view line;
    if (!next_line(line))
    {
        // Empty, or a header killed mid-write. Either way there are no
        // trajectories to keep; the caller restarts this file from scratch.
        checkpoint.valid_bytes = 0;
        return checkpoint;
    }
    if (line != expected_header)
    {
        throw std::runtime_error(
            "Resume refused: " + path.string() +
            " has a different column layout than this binary writes. It was "
            "made by another version of free_energy.exe; finish it with that "
            "binary, move it aside, or set MIPT_FREE_ENERGY_RESUME=0.");
    }
    checkpoint.valid_bytes = cursor;

    long line_number = 1;
    while (next_line(line))
    {
        ++line_number;
        const auto fields = split_csv_row(line);
        const std::string where =
            path.string() + " line " + std::to_string(line_number);
        if (fields.size() != kSamplesColumns)
        {
            throw std::runtime_error(
                "Resume refused: " + where + " has " +
                std::to_string(fields.size()) + " fields, expected " +
                std::to_string(kSamplesColumns) +
                ". The file is corrupt beyond a simple truncation.");
        }

        RunIdentity found;
        found.n = static_cast<int>(
            parse_long_field(fields[kSampleColumnN], where));
        found.p = parse_double_field(fields[kSampleColumnP], where);
        found.circ_type = static_cast<int>(
            parse_long_field(fields[kSampleColumnCircType], where));
        found.init_state = static_cast<int>(
            parse_long_field(fields[kSampleColumnInitState], where));
        found.requested_realizations = 0;
        require_identity_match(expected, found, path.string());

        const long trajectory =
            parse_long_field(fields[kSampleColumnTrajectory], where);
        if (trajectory != checkpoint.completed)
        {
            throw std::runtime_error(
                "Resume refused: " + where + " has trajectory=" +
                std::to_string(trajectory) + " where " +
                std::to_string(checkpoint.completed) +
                " was expected. Trajectory indices must run 0,1,2,... with no "
                "gaps; this file looks like two runs concatenated.");
        }
        checkpoint.master_seed = recover_master_seed(
            parse_uint64_field(fields[kSampleColumnSeed], where), trajectory);
        checkpoint.has_master_seed = true;
        ++checkpoint.completed;
        checkpoint.valid_bytes = cursor;
    }

    checkpoint.next_trajectory = checkpoint.completed;
    return checkpoint;
}

// Drops an unterminated trailing row (a kill mid-flush) so the file can be
// appended to. Nothing complete is ever removed.
inline void trim_partial_tail(
    const std::filesystem::path &path, const SamplesCheckpoint &checkpoint)
{
    if (!has_partial_tail(checkpoint))
    {
        return;
    }
    std::filesystem::resize_file(path, checkpoint.valid_bytes);
}

// --- time-series checkpoint ------------------------------------------------

struct TimeseriesCheckpoint
{
    long completed = 0;
    std::vector<TimeStats> stats;
};

constexpr std::size_t kTimeseriesColumns = 20;
constexpr std::size_t kTsColumnN = 0;
constexpr std::size_t kTsColumnP = 1;
constexpr std::size_t kTsColumnCircType = 2;
constexpr std::size_t kTsColumnInitState = 4;
constexpr std::size_t kTsColumnRequested = 5;
constexpr std::size_t kTsColumnCompleted = 6;
constexpr std::size_t kTsColumnT = 7;
constexpr std::size_t kTsColumnFMean = 10;
constexpr std::size_t kTsColumnFStddev = 11;
constexpr std::size_t kTsColumnFOverTlMean = 13;
constexpr std::size_t kTsColumnFOverTlStddev = 14;
constexpr std::size_t kTsColumnSHalfMean = 16;
constexpr std::size_t kTsColumnSHalfStddev = 17;
constexpr std::size_t kTsColumnSHalfCount = 19;

// Welford triple from what the writer stored. `write_stat` emits nan for an
// empty accumulator, which is how a not-sampled timestep is distinguished from
// one whose samples all happened to be zero.
inline RunningStats rebuild_stats(double mean, double stddev, std::uint64_t count)
{
    RunningStats stats;
    if (count == 0 || std::isnan(mean))
    {
        return stats;
    }
    stats.count = count;
    stats.mean = mean;
    stats.m2 = count > 1
                   ? stddev * stddev * static_cast<double>(count - 1)
                   : 0.0;
    return stats;
}

inline TimeseriesCheckpoint load_timeseries(
    const std::filesystem::path &path,
    std::string_view expected_header,
    const RunIdentity &expected,
    int total_timesteps,
    bool record_half_entropy,
    int entropy_max_t,
    int entropy_stride)
{
    const std::string content = read_whole_file(path);
    std::vector<std::string_view> lines;
    std::size_t cursor = 0;
    while (cursor < content.size())
    {
        const std::size_t newline = content.find('\n', cursor);
        if (newline == std::string::npos)
        {
            break;  // unterminated tail; the writer renames a complete file
        }
        lines.push_back(std::string_view(content).substr(cursor, newline - cursor));
        cursor = newline + 1;
    }

    if (lines.empty() || lines.front() != expected_header)
    {
        throw std::runtime_error(
            "Resume refused: " + path.string() +
            " has a different column layout than this binary writes. It was "
            "made by another version of free_energy.exe; finish it with that "
            "binary, move it aside, or set MIPT_FREE_ENERGY_RESUME=0.");
    }

    const std::size_t expected_rows =
        static_cast<std::size_t>(total_timesteps) + 1;
    if (lines.size() != expected_rows + 1)
    {
        throw std::runtime_error(
            "Resume refused: " + path.string() + " holds " +
            std::to_string(lines.size() - 1) + " timestep rows, but this run "
            "has " + std::to_string(expected_rows) +
            ". The time-series checkpoint belongs to a different N.");
    }

    TimeseriesCheckpoint checkpoint;
    checkpoint.stats.resize(expected_rows);
    long completed = -1;

    for (std::size_t row = 0; row < expected_rows; ++row)
    {
        const auto fields = split_csv_row(lines[row + 1]);
        const std::string where =
            path.string() + " line " + std::to_string(row + 2);
        if (fields.size() != kTimeseriesColumns)
        {
            throw std::runtime_error(
                "Resume refused: " + where + " has " +
                std::to_string(fields.size()) + " fields, expected " +
                std::to_string(kTimeseriesColumns) + '.');
        }

        RunIdentity found;
        found.n = static_cast<int>(parse_long_field(fields[kTsColumnN], where));
        found.p = parse_double_field(fields[kTsColumnP], where);
        found.circ_type =
            static_cast<int>(parse_long_field(fields[kTsColumnCircType], where));
        found.init_state =
            static_cast<int>(parse_long_field(fields[kTsColumnInitState], where));
        found.requested_realizations =
            parse_long_field(fields[kTsColumnRequested], where);
        require_identity_match(expected, found, path.string());

        const long t = parse_long_field(fields[kTsColumnT], where);
        if (t != static_cast<long>(row))
        {
            throw std::runtime_error(
                "Resume refused: " + where + " has t=" + std::to_string(t) +
                " on the row for t=" + std::to_string(row) + '.');
        }

        const long row_completed =
            parse_long_field(fields[kTsColumnCompleted], where);
        if (completed < 0)
        {
            completed = row_completed;
        }
        else if (row_completed != completed)
        {
            throw std::runtime_error(
                "Resume refused: " + where + " reports completed_realizations=" +
                std::to_string(row_completed) + " but earlier rows report " +
                std::to_string(completed) + '.');
        }
        if (row_completed < 0)
        {
            throw std::runtime_error(
                "Resume refused: " + where +
                " reports a negative completed_realizations.");
        }

        const auto count = static_cast<std::uint64_t>(row_completed);
        auto &stats = checkpoint.stats[row];
        stats.cumulative_f = rebuild_stats(
            parse_double_field(fields[kTsColumnFMean], where),
            parse_double_field(fields[kTsColumnFStddev], where),
            count);
        stats.f_over_tl = rebuild_stats(
            parse_double_field(fields[kTsColumnFOverTlMean], where),
            parse_double_field(fields[kTsColumnFOverTlStddev], where),
            count);

        const auto entropy_count = static_cast<std::uint64_t>(
            parse_long_field(fields[kTsColumnSHalfCount], where));
        stats.half_entropy = rebuild_stats(
            parse_double_field(fields[kTsColumnSHalfMean], where),
            parse_double_field(fields[kTsColumnSHalfStddev], where),
            entropy_count);

        // The entropy schedule is an environment setting, so a resume can
        // easily be launched with a different one. Continuing would leave a
        // column whose per-t sample counts silently disagree, so refuse.
        const bool should_record = records_half_entropy(
            static_cast<int>(row),
            record_half_entropy,
            entropy_max_t,
            entropy_stride);
        const auto wanted = should_record ? count : std::uint64_t{0};
        if (entropy_count != wanted)
        {
            throw std::runtime_error(
                "Resume refused: " + where + " has S_half_count=" +
                std::to_string(entropy_count) + " at t=" + std::to_string(row) +
                ", but the current half-chain entropy settings imply " +
                std::to_string(wanted) +
                ". Re-run with the MIPT_FREE_ENERGY_HALF_ENTROPY / "
                "_ENTROPY_MAX_T / _ENTROPY_STRIDE values the original run "
                "used.");
        }
    }

    checkpoint.completed = completed < 0 ? 0 : completed;
    return checkpoint;
}
} // namespace mipt::free_energy::resume
