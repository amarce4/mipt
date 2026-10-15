#pragma once

// Reading a dist_scaling.exe record file back in C++.
//
// dist_records.hpp owns the format and the writer; data_analysis.records owns
// the Python reader. This is the C++ reader, and it exists for one consumer:
// the stage-2 gate-trace replay (rppu_gate_trace.exe), which regenerates a
// trajectory from the master seed in the header and has to prove it got the
// *same* trajectory before any trace it writes means anything. The proof is a
// comparison against every stored record of that realization, so the replay
// needs the header's provenance fields, its observable names, the embedding
// table that turns (geometry_id, embedding_id) back into two sites, and the
// records of one realization -- found by binary search, because a production
// record file does not fit in memory and nothing here ever loads one whole.
//
// Deliberately read-only and dependency-free (no CUDA-Q, no MOSEK), so
// `make test-dist` pins it against RecordSink on the host.
//
// What the reader trusts and what it checks:
//   * the preamble is checked exactly as records::inspect() checks it (it is
//     reused), so a v1 file and a v2 file are both readable;
//   * the header text is split into `key=value` lines and the bracketed
//     tables; the key=value block is everything outside a table, so a value
//     may itself contain '=' (the graph definition does) -- only the first
//     '=' on a line separates;
//   * the `fields` line is cross-checked against the preamble's record size,
//     so a header that does not describe the payload is a refusal rather than
//     a silent misread of every value.

#include "mipt/dist_records.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mipt::dist::records
{

// One stored record. `flags` is 0 in a v1 file, which had none.
struct StoredRecord
{
    std::uint32_t realization = 0;
    std::uint16_t geometry = 0;
    std::uint16_t embedding = 0;
    std::uint32_t flags = 0;
    std::vector<double> values;
};

// A header table: its column names and its rows, as text.
struct HeaderTable
{
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;

    int column(const std::string &name) const
    {
        const auto it = std::find(columns.begin(), columns.end(), name);
        return it == columns.end() ? -1 : static_cast<int>(it - columns.begin());
    }
};

namespace detail
{
inline std::vector<std::string> split_commas(const std::string &line)
{
    std::vector<std::string> out;
    std::string cell;
    std::istringstream stream(line);
    while (std::getline(stream, cell, ','))
    {
        out.push_back(cell);
    }
    if (!line.empty() && line.back() == ',')
    {
        out.emplace_back();
    }
    return out;
}
} // namespace detail

class RecordReader
{
  public:
    explicit RecordReader(std::string path) : path_(std::move(path))
    {
        info_ = inspect(path_);
        if (!info_.exists)
        {
            throw std::runtime_error("Record file " + path_ + " does not exist.");
        }
        id_bytes_ = info_.version == FORMAT_VERSION_V1 ? ID_BYTES_V1 : ID_BYTES;
        parse_header(info_.header_text);

        const auto observables = fields_.find("observables");
        if (observables == fields_.end())
        {
            throw std::runtime_error(path_ + " has no `observables` header field.");
        }
        observables_ = detail::split_commas(observables->second);
        if (info_.record_bytes != id_bytes_ + observables_.size() * sizeof(double))
        {
            throw std::runtime_error(
                path_ + " declares " + std::to_string(observables_.size()) +
                " observables, which needs " +
                std::to_string(id_bytes_ + observables_.size() * sizeof(double)) +
                "-byte records, but its preamble says " + std::to_string(info_.record_bytes) +
                ".");
        }
        file_.open(path_, std::ios::binary);
        if (!file_)
        {
            throw std::runtime_error("Could not open record file " + path_ + " for reading.");
        }
    }

    const std::string &path() const { return path_; }
    std::uint32_t version() const { return info_.version; }
    std::size_t record_bytes() const { return info_.record_bytes; }
    std::uint64_t record_count() const { return info_.record_count; }
    const std::string &header_text() const { return info_.header_text; }
    const std::vector<std::string> &observables() const { return observables_; }
    const std::map<std::string, std::string> &fields() const { return fields_; }

    bool has(const std::string &key) const { return fields_.count(key) != 0; }
    // The value of a header field; a refusal naming the field when it is absent.
    const std::string &field(const std::string &key) const
    {
        const auto it = fields_.find(key);
        if (it == fields_.end())
        {
            throw std::runtime_error(path_ + " has no `" + key + "` header field.");
        }
        return it->second;
    }

    int observable_index(const std::string &name) const
    {
        const auto it = std::find(observables_.begin(), observables_.end(), name);
        return it == observables_.end() ? -1 : static_cast<int>(it - observables_.begin());
    }

    // The bracketed tables: "geometry" and (v2) "embedding". Empty when absent.
    const HeaderTable &table(const std::string &name) const
    {
        static const HeaderTable empty;
        const auto it = tables_.find(name);
        return it == tables_.end() ? empty : it->second;
    }

    // (geometry_id, embedding_id) -> the sites that embedding names, from the
    // v2 embedding table. Empty for a v1 file, which has none.
    std::map<std::pair<int, int>, std::vector<int>> embedding_sites() const
    {
        std::map<std::pair<int, int>, std::vector<int>> out;
        const HeaderTable &embedding = table("embedding");
        const int geometry_col = embedding.column("geometry_id");
        const int embedding_col = embedding.column("embedding_id");
        if (geometry_col < 0 || embedding_col < 0)
        {
            return out;
        }
        for (const auto &row : embedding.rows)
        {
            if (row.size() != embedding.columns.size())
            {
                throw std::runtime_error(path_ + " has a malformed embedding-table row.");
            }
            std::vector<int> sites;
            for (std::size_t c = 0; c < row.size(); ++c)
            {
                if (static_cast<int>(c) != geometry_col && static_cast<int>(c) != embedding_col)
                {
                    sites.push_back(std::stoi(row[c]));
                }
            }
            out[{std::stoi(row[static_cast<std::size_t>(geometry_col)]),
                 std::stoi(row[static_cast<std::size_t>(embedding_col)])}] = std::move(sites);
        }
        return out;
    }

    StoredRecord read_at(std::uint64_t index)
    {
        if (index >= info_.record_count)
        {
            throw std::out_of_range("Record index past the end of " + path_ + ".");
        }
        buffer_.resize(info_.record_bytes);
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(info_.payload_offset +
                                               index * static_cast<std::uint64_t>(info_.record_bytes)));
        file_.read(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
        if (!file_)
        {
            throw std::runtime_error("Record file " + path_ + " is truncated where a record was "
                                     "expected.");
        }
        StoredRecord record;
        std::memcpy(&record.realization, buffer_.data(), 4);
        std::memcpy(&record.geometry, buffer_.data() + 4, 2);
        std::memcpy(&record.embedding, buffer_.data() + 6, 2);
        if (id_bytes_ == ID_BYTES)
        {
            std::memcpy(&record.flags, buffer_.data() + 8, 4);
        }
        record.values.resize(observables_.size());
        for (std::size_t slot = 0; slot < observables_.size(); ++slot)
        {
            std::memcpy(&record.values[slot], buffer_.data() + id_bytes_ + slot * sizeof(double),
                        sizeof(double));
        }
        return record;
    }

    // Every record of one realization, in file order. Records are appended in
    // nondecreasing realization order (RecordSink only flushes whole
    // trajectories), so this is a binary search for the first one and a scan
    // to the last. Empty when the file holds none -- a stride that skipped the
    // trajectory, or a run that never reached it.
    std::vector<StoredRecord> read_realization(std::uint32_t realization)
    {
        std::uint64_t low = 0;
        std::uint64_t high = info_.record_count;
        while (low < high)
        {
            const std::uint64_t mid = low + (high - low) / 2;
            if (realization_at(mid) < realization)
            {
                low = mid + 1;
            }
            else
            {
                high = mid;
            }
        }
        std::vector<StoredRecord> out;
        for (std::uint64_t index = low; index < info_.record_count; ++index)
        {
            StoredRecord record = read_at(index);
            if (record.realization != realization)
            {
                break;
            }
            out.push_back(std::move(record));
        }
        return out;
    }

  private:
    std::uint32_t realization_at(std::uint64_t index)
    {
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(info_.payload_offset +
                                               index * static_cast<std::uint64_t>(info_.record_bytes)));
        std::uint32_t value = 0;
        file_.read(reinterpret_cast<char *>(&value), 4);
        if (!file_)
        {
            throw std::runtime_error("Record file " + path_ + " is truncated where a record was "
                                     "expected.");
        }
        return value;
    }

    void parse_header(const std::string &text)
    {
        std::istringstream stream(text);
        std::string line;
        std::string open_table;
        HeaderTable *table = nullptr;
        while (std::getline(stream, line))
        {
            if (table != nullptr)
            {
                if (line == "[end]")
                {
                    table = nullptr;
                    open_table.clear();
                    continue;
                }
                if (table->columns.empty())
                {
                    table->columns = detail::split_commas(line);
                }
                else
                {
                    table->rows.push_back(detail::split_commas(line));
                }
                continue;
            }
            if (line.size() >= 2 && line.front() == '[' && line.back() == ']')
            {
                open_table = line.substr(1, line.size() - 2);
                table = &tables_[open_table];
                continue;
            }
            if (line.empty())
            {
                continue;
            }
            const std::size_t eq = line.find('=');
            if (eq == std::string::npos)
            {
                throw std::runtime_error(path_ + " has a header line that is neither "
                                                 "key=value nor a table: `" + line + "`.");
            }
            fields_[line.substr(0, eq)] = line.substr(eq + 1);
        }
        if (table != nullptr)
        {
            throw std::runtime_error(path_ + " has an unterminated [" + open_table +
                                     "] header table.");
        }
    }

    std::string path_;
    FileInfo info_;
    std::size_t id_bytes_ = ID_BYTES;
    std::map<std::string, std::string> fields_;
    std::map<std::string, HeaderTable> tables_;
    std::vector<std::string> observables_;
    std::ifstream file_;
    std::vector<char> buffer_;
};

} // namespace mipt::dist::records
