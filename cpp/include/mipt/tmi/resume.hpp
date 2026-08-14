#pragma once

// Checkpoint/resume for sim_tmi.exe.
//
// Same contract as free_energy.exe, entropy.exe and dist_scaling.exe: re-issue
// the identical command and the run continues where it stopped instead of
// overwriting.  SIM_TMI_RESUME=0 restores the old truncate-on-start behaviour.
//
// What makes this app different is that its CSV is deliberately *not*
// self-describing.  A TMI scan emits res * realizations * cycles rows -- 6.3M
// for an N=12 100k-realization all-cycles scan -- so the schema stays `p,tmi`
// for the same reason gmn.exe's stays `p,gmn,bipneg`, and there is no metadata
// block to validate a resume against.  The identity is recovered from the
// generated file name instead, which carries the circuit tag (and with it
// circ_type, all_cycles and the entropy order), n, realizations, p_min, p_max
// and res.
//
// `periods` is the one run argument no name or column records, so two runs that
// differ only in depth would be conflated.  That is a deliberate, accepted gap
// rather than an oversight -- in practice a TMI scan's depth is a function of
// its size -- and it is the reason this header does not pretend to a full
// identity check the way entropy_resume.hpp does.
//
// The p column is then a much stronger cross-check than it looks.  Rows are
// written grouped by p in `linspace(p_min, p_max, res)` order, and writer and
// reader both render p through append_double at 17 significant digits, so a
// matching double round-trips to a byte-identical field.  Comparing the text
// re-derives the whole grid exactly while costing one substring compare per
// row rather than a double parse over a multi-hundred-megabyte file.

#include "mipt/env.hpp"
#include "mipt/util/resume_csv.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mipt::tmi::resume
{
namespace csv = mipt::util::resume;

inline bool enabled()
{
    return env::boolean("SIM_TMI_RESUME", true);
}

// The row of column names every sim_tmi CSV starts with.  Defined here so the
// writer and the checkpoint reader cannot drift apart.
inline constexpr const char *kResultsHeader = "p,tmi";

// --- file-name identity -----------------------------------------------------

// The fields default_tmi_output_path() encodes into a name of the form
//   tmi_<tag>_n_<n>_real_<realizations>_<p_min>_<p_max>_res_<res>.csv
// The p bounds stay as text: the name is written through
// format_number_for_filename, which is lossy, so the only meaningful
// comparison is against the same rendering of the current arguments.
struct FileIdentity
{
    std::string tag;
    long n = 0;
    long realizations = 0;
    long res = 0;
    std::string p_min_text;
    std::string p_max_text;
};

inline std::string_view path_basename(std::string_view path)
{
    const std::size_t slash = path.find_last_of("/\\");
    return (slash == std::string_view::npos) ? path : path.substr(slash + 1);
}

// Parsed right to left, which is what makes it unambiguous: the tag is
// whatever sits between the "tmi_" prefix and the last "_n_", so this needs no
// list of known tags and keeps working when a new circuit or a suffix like
// _s2 is added.
inline bool parse_output_filename(std::string_view basename, FileIdentity &out)
{
    constexpr std::string_view prefix = "tmi_";
    constexpr std::string_view suffix = ".csv";
    if (basename.size() <= prefix.size() + suffix.size() ||
        basename.substr(0, prefix.size()) != prefix ||
        basename.substr(basename.size() - suffix.size()) != suffix)
    {
        return false;
    }
    std::string_view stem =
        basename.substr(prefix.size(), basename.size() - prefix.size() - suffix.size());

    auto split_last = [](std::string_view text, std::string_view token,
                         std::string_view &head, std::string_view &tail) {
        const std::size_t at = text.rfind(token);
        if (at == std::string_view::npos)
        {
            return false;
        }
        head = text.substr(0, at);
        tail = text.substr(at + token.size());
        return true;
    };

    auto parse_long = [](std::string_view text, long &value) {
        if (text.empty())
        {
            return false;
        }
        value = 0;
        for (const char c : text)
        {
            if (c < '0' || c > '9')
            {
                return false;
            }
            value = value * 10 + (c - '0');
        }
        return true;
    };

    std::string_view head;
    std::string_view tail;
    if (!split_last(stem, "_res_", head, tail) || !parse_long(tail, out.res))
    {
        return false;
    }
    stem = head;

    if (!split_last(stem, "_real_", head, tail))
    {
        return false;
    }
    // tail is "<realizations>_<p_min>_<p_max>"
    const std::size_t first = tail.find('_');
    if (first == std::string_view::npos)
    {
        return false;
    }
    const std::size_t second = tail.find('_', first + 1);
    if (second == std::string_view::npos || tail.find('_', second + 1) != std::string_view::npos)
    {
        return false;
    }
    if (!parse_long(tail.substr(0, first), out.realizations))
    {
        return false;
    }
    out.p_min_text = std::string(tail.substr(first + 1, second - first - 1));
    out.p_max_text = std::string(tail.substr(second + 1));
    stem = head;

    if (!split_last(stem, "_n_", head, tail) || !parse_long(tail, out.n))
    {
        return false;
    }
    out.tag = std::string(head);
    return !out.tag.empty();
}

// Returns an empty string when the two agree, otherwise a message naming the
// first field that differs.  Field-by-field rather than a whole-name compare so
// a refusal says *what* is wrong.
inline std::string describe_identity_mismatch(const FileIdentity &expected,
                                              const FileIdentity &found)
{
    auto text_field = [](std::string_view field, const std::string &want,
                         const std::string &got) {
        return (want == got) ? std::string()
                             : std::string(field) + "=" + got + " (this run uses " + want + ")";
    };
    auto long_field = [](std::string_view field, long want, long got) {
        return (want == got) ? std::string()
                            : std::string(field) + "=" + std::to_string(got) +
                                  " (this run uses " + std::to_string(want) + ")";
    };

    std::string message = text_field("circuit tag", expected.tag, found.tag);
    if (message.empty()) message = long_field("n", expected.n, found.n);
    if (message.empty()) message = long_field("realizations", expected.realizations, found.realizations);
    if (message.empty()) message = text_field("p_min", expected.p_min_text, found.p_min_text);
    if (message.empty()) message = text_field("p_max", expected.p_max_text, found.p_max_text);
    if (message.empty()) message = long_field("resolution", expected.res, found.res);
    return message;
}

// --- checkpoint -------------------------------------------------------------

struct Checkpoint
{
    bool loaded = false;
    bool name_verified = false;      // the file name matched the current arguments
    std::uint64_t rows = 0;          // TMI values already recorded and kept
    std::uint64_t dropped_rows = 0;  // trailing partial circuit discarded
    std::uintmax_t kept_bytes = 0;   // file size after truncation
    std::size_t start_p_index = 0;
    int start_realization = 0;
};

// Scans an existing CSV and works out where the run stopped.
//
// `expected_name` is the base name the current arguments would generate; the
// caller builds it with default_tmi_output_path so there is one definition of
// the naming scheme.  `p_texts` is the p grid rendered exactly as the writer
// renders it.
inline Checkpoint load(const std::string &output_path,
                       const std::string &expected_name,
                       int realizations,
                       int cycle_count,
                       const std::vector<std::string> &p_texts)
{
    Checkpoint checkpoint;

    FileIdentity expected;
    FileIdentity found;
    const bool expected_ok = parse_output_filename(path_basename(expected_name), expected);
    const bool found_ok = parse_output_filename(path_basename(output_path), found);
    if (expected_ok && found_ok)
    {
        const std::string mismatch = describe_identity_mismatch(expected, found);
        if (!mismatch.empty())
        {
            csv::refuse(output_path + " was produced with " + mismatch +
                        ". Move the existing file aside, or set SIM_TMI_RESUME=0 to "
                        "overwrite it.");
        }
        checkpoint.name_verified = true;
    }

    const std::string content = csv::read_whole_file(output_path);
    std::uintmax_t valid_bytes = 0;
    const std::vector<std::string_view> lines = csv::split_complete_lines(content, valid_bytes);

    if (lines.empty())
    {
        // Empty, or a header that never made it to disk: start clean.
        return checkpoint;
    }
    if (lines.front() != kResultsHeader)
    {
        csv::refuse(output_path + " does not start with the expected header \"" +
                    std::string(kResultsHeader) + "\"; found \"" +
                    std::string(lines.front()) + "\".");
    }
    if (p_texts.empty() || realizations <= 0 || cycle_count <= 0)
    {
        csv::refuse(output_path + " cannot be resumed: the run's p grid or "
                                  "realization/cycle counts are degenerate.");
    }

    const std::uint64_t rows_per_p = static_cast<std::uint64_t>(realizations) *
                                     static_cast<std::uint64_t>(cycle_count);

    std::size_t group = 0;
    std::uint64_t rows_in_group = 0;
    std::uint64_t total_rows = 0;
    std::uintmax_t offset = lines.front().size() + 1;  // past the header
    std::uintmax_t boundary_bytes = offset;            // end of the last whole circuit

    for (std::size_t i = 1; i < lines.size(); ++i)
    {
        const std::string_view row = lines[i];
        const std::size_t comma = row.find(',');
        if (comma == std::string_view::npos)
        {
            csv::refuse(output_path + " has a malformed row at line " +
                        std::to_string(i + 1) + ": \"" + std::string(row) + "\".");
        }
        const std::string_view p_text = row.substr(0, comma);

        if (p_text != p_texts[group])
        {
            // Groups are contiguous and written in grid order, so a change of p
            // is only legal where the previous group filled up.
            if (rows_in_group != rows_per_p)
            {
                csv::refuse(output_path + " switches away from p=" + p_texts[group] +
                            " after " + std::to_string(rows_in_group) +
                            " rows, but a complete p group holds realizations*cycles = " +
                            std::to_string(rows_per_p) +
                            ". The file does not match this run's arguments.");
            }
            ++group;
            if (group >= p_texts.size() || p_text != p_texts[group])
            {
                csv::refuse(output_path + " carries p=" + std::string(p_text) +
                            " where this run's grid expects " +
                            (group < p_texts.size() ? p_texts[group]
                                                    : std::string("no further values")) +
                            ". The file was produced with different p arguments.");
            }
            rows_in_group = 0;
        }

        ++rows_in_group;
        ++total_rows;
        offset += row.size() + 1;
        if (rows_in_group % static_cast<std::uint64_t>(cycle_count) == 0)
        {
            boundary_bytes = offset;
        }
    }

    if (rows_in_group > rows_per_p)
    {
        csv::refuse(output_path + " holds " + std::to_string(rows_in_group) +
                    " rows for p=" + p_texts[group] + ", more than the " +
                    std::to_string(rows_per_p) + " a complete group can hold.");
    }

    // Whole circuits only.  A trailing partial circuit turns up whenever
    // SIM_TMI_CSV_FLUSH_ROWS is not a multiple of the cycle count -- the N=8/12
    // default of 1000 rows is exactly that -- and keeping it would both leave
    // that trajectory's cycle offsets unevenly sampled and make the restart
    // point ambiguous.  At most cycle_count-1 rows are given up.
    const std::uint64_t complete_in_group = rows_in_group / static_cast<std::uint64_t>(cycle_count);
    checkpoint.dropped_rows =
        rows_in_group - complete_in_group * static_cast<std::uint64_t>(cycle_count);
    checkpoint.rows = total_rows - checkpoint.dropped_rows;
    checkpoint.kept_bytes = boundary_bytes;

    if (complete_in_group >= static_cast<std::uint64_t>(realizations))
    {
        checkpoint.start_p_index = group + 1;
        checkpoint.start_realization = 0;
    }
    else
    {
        checkpoint.start_p_index = group;
        checkpoint.start_realization = static_cast<int>(complete_in_group);
    }
    checkpoint.loaded = true;
    return checkpoint;
}

} // namespace mipt::tmi::resume
