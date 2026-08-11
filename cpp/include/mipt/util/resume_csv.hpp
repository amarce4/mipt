#pragma once

// Reading an application's own CSV back in as a checkpoint.
//
// Three executables resume an interrupted run -- free_energy.exe, entropy.exe,
// and dist_scaling.exe -- and all three do it the same way: the CSV they write
// *is* the checkpoint, with no side-car file. That is deliberate. It is what
// lets a run interrupted by a binary that had no resume support at all be
// picked up by one that does, and it keeps the checkpoint honest, because the
// thing being restored is the same thing the analysis stack reads.
//
// What makes it work is that a Welford accumulator is three numbers and every
// writer stores two of them plus enough to imply the third:
//
//   m2 = stddev^2 * (count - 1)            (entropy.exe, free_energy.exe)
//   m2 = stderr^2 * count * (count - 1)    (dist_scaling.exe)
//
// Both are exact to within a couple of ulp, which is far below the 1/sqrt(count)
// statistical error on anything derived from them. The counts are either an
// explicit column or implied by the run's own bookkeeping.
//
// Nothing here depends on CUDA-Q, MOSEK, or a GPU, so all of it is exercised by
// `make test-core`.

#include "mipt/util/stats.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mipt::util::resume
{

// Every refusal carries this prefix so a user can tell a checkpoint rejection
// from an ordinary argument error at a glance.
[[noreturn]] inline void refuse(const std::string &reason)
{
    throw std::runtime_error("Resume refused: " + reason);
}

// --- raw text --------------------------------------------------------------

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

// Splits one CSV row, honouring the double-quoted circuit-name field written by
// util::csv_quote (doubled quotes escape a literal quote).
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

// Complete lines only. An unterminated tail is a kill during the flush; it is
// reported through `valid_bytes` rather than parsed.
inline std::vector<std::string_view> split_complete_lines(
    std::string_view content, std::uintmax_t &valid_bytes)
{
    std::vector<std::string_view> lines;
    std::size_t cursor = 0;
    while (cursor < content.size())
    {
        const std::size_t newline = content.find('\n', cursor);
        if (newline == std::string_view::npos)
        {
            break;
        }
        lines.push_back(content.substr(cursor, newline - cursor));
        cursor = newline + 1;
    }
    valid_bytes = static_cast<std::uintmax_t>(cursor);
    return lines;
}

// --- typed fields ----------------------------------------------------------

inline double parse_double_field(const std::string &text, const std::string &context)
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
        refuse("could not parse '" + text + "' as a number in " + context);
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
        refuse("could not parse '" + text + "' as an integer in " + context);
    }
}

inline std::uint64_t parse_uint64_field(const std::string &text, const std::string &context)
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
        refuse("could not parse '" + text + "' as an unsigned integer in " + context);
    }
}

// --- Welford reconstruction ------------------------------------------------

// `count == 0` and a NaN mean both mean "no samples". Writers emit nan for an
// empty accumulator precisely so an unsampled bin is distinguishable from one
// whose samples all happened to be zero, and that distinction must survive.
inline RunningStats stats_from_m2(double mean, double m2, std::uint64_t count)
{
    RunningStats stats;
    if (count == 0 || std::isnan(mean))
    {
        return stats;
    }
    stats.count = count;
    stats.mean = mean;
    stats.m2 = count > 1 ? m2 : 0.0;
    return stats;
}

inline RunningStats stats_from_stddev(double mean, double stddev, std::uint64_t count)
{
    return stats_from_m2(
        mean,
        count > 1 ? stddev * stddev * static_cast<double>(count - 1) : 0.0,
        count);
}

// dist_scaling.exe stores the standard error rather than the sample standard
// deviation: stderr = stddev / sqrt(count), so m2 = stderr^2 * count * (count-1).
inline RunningStats stats_from_stderr(double mean, double standard_error, std::uint64_t count)
{
    const double scale =
        static_cast<double>(count) * static_cast<double>(count > 0 ? count - 1 : 0);
    return stats_from_m2(mean, count > 1 ? standard_error * standard_error * scale : 0.0, count);
}

// --- a whole CSV, addressed by column name ---------------------------------

// Column *names* rather than indices, because dist_scaling.exe's schema grows
// eight (k=2) or twenty-four (k=3) columns for parity-preserving circuits, and
// an off-by-one in a hand-counted index would silently restore the wrong
// accumulator.
class CsvTable
{
  public:
    static CsvTable read(const std::filesystem::path &path)
    {
        CsvTable table;
        table.path_ = path.string();
        const std::string content = read_whole_file(path);
        table.file_bytes_ = content.size();
        const auto lines = split_complete_lines(content, table.valid_bytes_);
        if (lines.empty())
        {
            return table;
        }
        table.columns_ = split_csv_row(lines.front());
        table.rows_.reserve(lines.size() - 1);
        for (std::size_t line = 1; line < lines.size(); ++line)
        {
            auto fields = split_csv_row(lines[line]);
            if (fields.size() != table.columns_.size())
            {
                refuse(
                    table.path_ + " line " + std::to_string(line + 1) + " has " +
                    std::to_string(fields.size()) + " fields against the header's " +
                    std::to_string(table.columns_.size()) +
                    ". The file is corrupt beyond a simple truncation.");
            }
            table.rows_.push_back(std::move(fields));
        }
        return table;
    }

    bool empty() const { return columns_.empty(); }
    std::size_t row_count() const { return rows_.size(); }
    const std::string &path() const { return path_; }
    std::uintmax_t valid_bytes() const { return valid_bytes_; }
    std::uintmax_t file_bytes() const { return file_bytes_; }
    bool has_partial_tail() const { return file_bytes_ > valid_bytes_; }

    // The header line as written, so a caller can compare schemas directly.
    std::string header_line() const
    {
        std::string joined;
        for (std::size_t i = 0; i < columns_.size(); ++i)
        {
            if (i != 0)
            {
                joined += ',';
            }
            joined += columns_[i];
        }
        return joined;
    }

    bool has_column(std::string_view name) const
    {
        for (const auto &column : columns_)
        {
            if (column == name)
            {
                return true;
            }
        }
        return false;
    }

    std::size_t column(std::string_view name) const
    {
        for (std::size_t i = 0; i < columns_.size(); ++i)
        {
            if (columns_[i] == name)
            {
                return i;
            }
        }
        refuse(
            path_ + " has no '" + std::string(name) +
            "' column, so it was written by a different version of this "
            "executable. Finish it with that binary, move it aside, or disable "
            "resume to start over.");
    }

    const std::string &text(std::size_t row, std::string_view name) const
    {
        return rows_.at(row).at(column(name));
    }

    double real(std::size_t row, std::string_view name) const
    {
        return parse_double_field(text(row, name), where(row, name));
    }

    long integer(std::size_t row, std::string_view name) const
    {
        return parse_long_field(text(row, name), where(row, name));
    }

    std::uint64_t counter(std::size_t row, std::string_view name) const
    {
        return parse_uint64_field(text(row, name), where(row, name));
    }

    // mean/stderr/count triples, the shape dist_scaling.exe writes.
    RunningStats stats_by_stderr(
        std::size_t row, std::string_view prefix) const
    {
        const std::string stem(prefix);
        return stats_from_stderr(
            real(row, stem + "_mean"),
            real(row, stem + "_stderr"),
            counter(row, stem + "_samples"));
    }

    std::string where(std::size_t row, std::string_view name) const
    {
        return path_ + " line " + std::to_string(row + 2) + " column '" +
               std::string(name) + '\'';
    }

  private:
    std::string path_;
    std::vector<std::string> columns_;
    std::vector<std::vector<std::string>> rows_;
    std::uintmax_t valid_bytes_ = 0;
    std::uintmax_t file_bytes_ = 0;
};

// --- identity checks -------------------------------------------------------

// A checkpoint that describes a different run must be refused, not pooled: the
// output path is derived from the run parameters, so a file can outlive an
// argument change and still be found by name.
inline void require_same(
    std::string_view field, long expected, long found, const std::string &path)
{
    if (expected != found)
    {
        refuse(
            path + " was produced with " + std::string(field) + '=' +
            std::to_string(found) + ", but this run uses " +
            std::to_string(expected) +
            ". Move the existing file aside, or disable resume to overwrite it.");
    }
}

inline void require_same(
    std::string_view field, double expected, double found, const std::string &path)
{
    // The writers emit 17 significant digits, so a genuine match round-trips
    // exactly; the tolerance only absorbs a shorter representation.
    if (!(std::abs(expected - found) <= 1.0e-12 * std::max(1.0, std::abs(expected))))
    {
        refuse(
            path + " was produced with " + std::string(field) + '=' +
            std::to_string(found) + ", but this run uses " +
            std::to_string(expected) +
            ". Move the existing file aside, or disable resume to overwrite it.");
    }
}
} // namespace mipt::util::resume
