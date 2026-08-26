#pragma once

// Per-record output for dist_scaling.exe: one row per (trajectory, geometry,
// embedding) carrying that record's two observables.
//
// Why this exists
// ---------------
// The aggregated CSV reports one mean and one standard error per geometry, and
// that standard error treats records as independent. They are not: every
// geometry is measured inside the *same* trajectory, so the points of a decay
// curve covary strongly and a fit to the whole curve has a much larger error
// than propagating the per-point stderrs suggests. The fix is the same one the
// two-probe batch sidecar uses -- keep enough structure to resample whole
// trajectories -- except that here the records themselves are small enough to
// store outright, which also buys the value *distributions* an aggregate can
// never recover.
//
// Format: a flat record array with a text header
// ---------------------------------------------
// Parquet and HDF5 were both considered and both rejected, for the same two
// reasons.
//
//   * They are dependencies. Parquet means Arrow and HDF5 means libhdf5, in a
//     build that already documents real portability pain over finding one CUDA
//     toolkit. Neither buys anything here: the payload is a fixed-width table
//     of two integers and two-to-four float64s, which is precisely the case
//     where a columnar container's compression and predicate pushdown have
//     nothing to work with.
//
//   * Neither appends safely. This run is interruptible by design -- a pause
//     sentinel, a resume path, and a documented history of kills -- and a
//     Parquet file is only readable once its footer is written, so a killed
//     run leaves an unreadable file. A flat record array is append-only, and a
//     truncated tail is not merely detectable but *trivially* so: the record
//     size is fixed, so anything past the last whole record is discarded.
//
// So the layout is:
//
//     bytes  0.. 7  magic "MIPTDREC"
//     bytes  8..11  uint32 format version
//     bytes 12..15  uint32 record bytes
//     bytes 16..19  uint32 header text bytes
//     bytes 20..23  uint32 reserved (0)
//     then          the header text
//     then          record_bytes * record_count bytes of records
//
// The header text is a `key=value` block followed by the geometry lookup
// table, so the file says what it is without any accompanying CSV, and numpy
// reads the payload with one `np.fromfile(..., dtype=..., offset=...)`. The
// geometry table is what keeps `d` out of the records: a record carries a
// two-byte geometry id, and the header maps that id to the effective chord
// distance and everything else describing the geometry.
//
// Record layout (all little-endian, and every field naturally aligned, so the
// struct is 24 or 40 bytes with no padding anywhere):
//
//     uint32 realization_id
//     uint16 geometry_id
//     uint16 embedding_id
//     float64 I_k
//     float64 N_k
//     float64 fI_k     only for parity-preserving ensembles
//     float64 fN_k     only for parity-preserving ensembles
//
// with (I_k, N_k) = (MI, negativity) at k=2 and (TMI, GMN) at k=3. The
// fermionic pair is present under exactly the condition that gates the
// fermionic CSV columns, `RunConfig::fermionic_outputs()`, because the point of
// reporting both conventions is to compare them on the same records.
//
// A NaN is not a zero. A GMN the schedule did not select, and one MOSEK failed
// to solve, are both written as NaN -- the same distinction the CSV draws
// between a bin with no samples and a bin whose samples are zero. The reader
// drops them from that column and keeps the row's I_k.
//
// Deferred values and trajectory-aligned flushing
// ----------------------------------------------
// GMN comes back from a background solver thread long after the record that
// produced it, so a k=3 record cannot be written when it is created. Records
// are therefore buffered and the *resolved prefix* is flushed, which keeps the
// file in trajectory order. Two things bound the buffer: the SDP queue already
// blocks at `pending_batches` batches in flight, and a record with no pending
// solve resolves on creation. The flush additionally stops at a trajectory
// boundary, so the file only ever holds whole trajectories -- which is what
// makes truncating it against the CSV checkpoint exact.
//
// Nothing here depends on CUDA-Q, MOSEK, or CUDA, so `make test-dist` pins the
// round trip on the host.

#include "mipt/dist_scaling_csv.hpp"
#include "mipt/env.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace mipt::dist::records
{

inline constexpr char MAGIC[8] = {'M', 'I', 'P', 'T', 'D', 'R', 'E', 'C'};
inline constexpr std::uint32_t FORMAT_VERSION = 1;
inline constexpr std::size_t PREAMBLE_BYTES = 24;
inline constexpr std::size_t ID_BYTES = 8;
inline constexpr std::size_t MAX_VALUES = 4;

// Per-record output is on by default: the clustered bootstrap it enables is
// the only way to put an honest error bar on a decay exponent, and the file is
// small next to the run that produced it. `MIPT_DIST_RECORDS=0` turns it off
// for a sweep that only wants the aggregate.
inline bool enabled()
{
    return env::boolean("MIPT_DIST_RECORDS", true);
}

// How many resolved records accumulate before the buffer is written out. Only
// a memory bound -- correctness does not depend on it -- but a fast small-N run
// can produce hundreds of thousands of records between progress ticks.
inline std::size_t flush_threshold()
{
    return static_cast<std::size_t>(
        env::integer("MIPT_DIST_RECORDS_FLUSH", 8192, 1, 100000000));
}

// The record file beside a given CSV: same stem, `_records.bin`. Keeping it
// adjacent is what lets `data_analysis.dist_scaling` find it from the CSV the
// caller actually named.
inline std::string records_path_for(const std::string &csv_path)
{
    std::filesystem::path path(csv_path);
    std::filesystem::path stem = path;
    stem.replace_extension();
    return (stem.string() + "_records.bin");
}

inline std::size_t record_bytes(std::size_t value_count)
{
    return ID_BYTES + value_count * sizeof(double);
}

// ---------------------------------------------------------------------------
// Header text
// ---------------------------------------------------------------------------

inline void append_field(std::string &out, const char *key, const std::string &value)
{
    out += key;
    out += '=';
    out += value;
    out += '\n';
}

// The `key=value` block every record file carries, whatever its party count.
inline std::string common_header_fields(const RunConfig &config,
                                        const std::vector<const char *> &observables)
{
    std::string out;
    append_field(out, "format", "mipt_dist_records");
    append_field(out, "version", std::to_string(FORMAT_VERSION));
    append_field(out, "k", std::to_string(config.k));
    append_field(out, "N", std::to_string(config.n));
    append_field(out, "circ_type", std::to_string(static_cast<int>(config.type)));
    append_field(out, "circuit_name", std::string(circuit_type_name(config.type)));
    append_field(out, "periods", std::to_string(config.periods));
    std::string p_text;
    append_double(p_text, config.p);
    append_field(out, "p", p_text);
    append_field(out, "realizations", std::to_string(config.realizations));
    // Every entropy dist_scaling.exe reports is a log2 quantity, exactly as
    // the CSV's own `entropy_units` column says. The negativities are not
    // entropies and carry no unit.
    append_field(out, "entropy_units", "bits");

    std::string fields = "realization_id:u4,geometry_id:u2,embedding_id:u2";
    std::string names;
    for (std::size_t i = 0; i < observables.size(); ++i)
    {
        fields += ',';
        fields += observables[i];
        fields += ":f8";
        if (i > 0)
        {
            names += ',';
        }
        names += observables[i];
    }
    append_field(out, "fields", fields);
    append_field(out, "observables", names);
    return out;
}

inline void append_geometry_table(std::string &out, const std::string &columns,
                                  const std::string &rows, std::size_t row_count)
{
    append_field(out, "geometry_rows", std::to_string(row_count));
    out += "[geometry]\n";
    out += columns;
    out += '\n';
    out += rows;
    out += "[end]\n";
}

// k=2: the geometry is the ring separation, and `d` is its chord length.
inline std::string pair_header_text(const RunConfig &config, const std::vector<PairBin> &bins)
{
    const bool fermionic = config.fermionic_outputs();
    std::vector<const char *> observables{"mi", "mn"};
    if (fermionic)
    {
        observables.push_back("fmi");
        observables.push_back("fmn");
    }
    std::string out = common_header_fields(config, observables);

    std::string rows;
    for (std::size_t index = 0; index < bins.size(); ++index)
    {
        rows += std::to_string(index);
        rows += ',';
        rows += std::to_string(bins[index].separation);
        rows += ',';
        append_double(rows, bins[index].chord);
        rows += ',';
        append_uint(rows, bins[index].embedding_count);
        rows += '\n';
    }
    append_geometry_table(out, "geometry_id,separation,d,embedding_count", rows, bins.size());
    return out;
}

// k=3: the geometry is a triangle, and `d` is the geometric mean of its three
// chords -- the same effective distance the CSV's `d` column carries and the
// same one three-probe mode 4 writes as `chord_geometric_mean`.
inline std::string triple_header_text(const RunConfig &config, const std::vector<TripleBin> &bins,
                                      long embeddings_per_geometry)
{
    const bool fermionic = config.fermionic_outputs();
    std::vector<const char *> observables{"tmi", "gmn"};
    if (fermionic)
    {
        observables.push_back("ftmi");
        observables.push_back("fgmn");
    }
    std::string out = common_header_fields(config, observables);
    std::string balance_text;
    append_double(balance_text, config.triangle_balance_cutoff);
    append_field(out, "b_min", balance_text);
    append_field(out, "embeddings_per_geometry", std::to_string(embeddings_per_geometry));

    std::string rows;
    for (std::size_t index = 0; index < bins.size(); ++index)
    {
        const TripleBin &bin = bins[index];
        rows += std::to_string(index);
        for (int separation : bin.separations)
        {
            rows += ',';
            rows += std::to_string(separation);
        }
        rows += ',';
        append_double(rows, bin.balance);
        rows += ',';
        append_double(rows, bin.chord_geometric_mean);
        rows += ',';
        append_uint(rows, static_cast<std::uint64_t>(bin.embedding_count));
        rows += '\n';
    }
    append_geometry_table(
        out, "geometry_id,separation_1,separation_2,separation_3,triangle_balance,d,embedding_count",
        rows, bins.size());
    return out;
}

// ---------------------------------------------------------------------------
// Reading back an existing file's header, and trimming it to a checkpoint
// ---------------------------------------------------------------------------

struct FileInfo
{
    bool exists = false;
    std::string header_text;
    std::uint32_t version = 0;
    std::size_t record_bytes = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t record_count = 0;
};

inline FileInfo inspect(const std::string &path)
{
    FileInfo info;
    std::error_code error;
    if (!std::filesystem::exists(path, error) || error)
    {
        return info;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open record file " + path + " for reading.");
    }
    char preamble[PREAMBLE_BYTES];
    file.read(preamble, static_cast<std::streamsize>(PREAMBLE_BYTES));
    if (!file || std::memcmp(preamble, MAGIC, sizeof(MAGIC)) != 0)
    {
        throw std::runtime_error(path + " exists but is not a dist_scaling record file. "
                                        "Move it aside, or set MIPT_DIST_RECORDS=0.");
    }
    std::uint32_t version = 0;
    std::uint32_t bytes = 0;
    std::uint32_t header_bytes = 0;
    std::memcpy(&version, preamble + 8, 4);
    std::memcpy(&bytes, preamble + 12, 4);
    std::memcpy(&header_bytes, preamble + 16, 4);
    if (version != FORMAT_VERSION)
    {
        throw std::runtime_error(path + " was written in record format version " +
                                 std::to_string(version) + ", but this binary writes version " +
                                 std::to_string(FORMAT_VERSION) +
                                 ". Move it aside, or set MIPT_DIST_RECORDS=0.");
    }
    std::string header(header_bytes, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header_bytes));
    if (!file)
    {
        throw std::runtime_error(path + " is truncated inside its header.");
    }
    info.exists = true;
    info.header_text = std::move(header);
    info.version = version;
    info.record_bytes = bytes;
    info.payload_offset = PREAMBLE_BYTES + header_bytes;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error)
    {
        throw std::runtime_error("Could not size record file " + path + ".");
    }
    // Anything past the last whole record is what a kill left behind.
    info.record_count = bytes > 0 && size > info.payload_offset
                            ? (static_cast<std::uint64_t>(size) - info.payload_offset) / bytes
                            : 0;
    return info;
}

inline std::uint32_t read_realization_at(std::ifstream &file, const FileInfo &info,
                                         std::uint64_t index)
{
    file.seekg(static_cast<std::streamoff>(info.payload_offset +
                                           index * static_cast<std::uint64_t>(info.record_bytes)));
    char bytes[4];
    file.read(bytes, 4);
    if (!file)
    {
        throw std::runtime_error("Record file is truncated where a record was expected.");
    }
    std::uint32_t value = 0;
    std::memcpy(&value, bytes, 4);
    return value;
}

// The number of records whose realization index is below `completed`.
//
// Records are appended in nondecreasing realization order and the sink only
// ever flushes whole trajectories, so this is a binary search rather than a
// scan -- and the result is exactly the prefix that agrees with a CSV
// checkpoint reporting `completed` finished trajectories.
inline std::uint64_t records_below_realization(const std::string &path, const FileInfo &info,
                                               std::uint64_t completed)
{
    if (info.record_count == 0)
    {
        return 0;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open record file " + path + " for reading.");
    }
    std::uint64_t low = 0;
    std::uint64_t high = info.record_count;
    while (low < high)
    {
        const std::uint64_t mid = low + (high - low) / 2;
        if (static_cast<std::uint64_t>(read_realization_at(file, info, mid)) < completed)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return low;
}

// ---------------------------------------------------------------------------
// The sink
// ---------------------------------------------------------------------------

class RecordSink
{
  public:
    RecordSink() = default;
    RecordSink(const RecordSink &) = delete;
    RecordSink &operator=(const RecordSink &) = delete;
    ~RecordSink() { close(); }

    bool active() const { return active_; }
    const std::string &path() const { return path_; }
    std::size_t value_count() const { return value_count_; }
    // Whole trajectories the file already held when this run opened it.
    std::uint64_t inherited_records() const { return inherited_records_; }
    bool truncated_on_open() const { return truncated_; }
    // Trajectories present in the file, against what the CSV checkpoint said
    // was finished. The two can differ at k=3: a record cannot be written
    // until the solve that fills its GMN column comes back, so a kill leaves
    // the record file behind the CSV by up to the solver queue's depth. The
    // trajectories that are missing are missing whole, which is what a
    // trajectory-clustered bootstrap needs.
    std::uint64_t realizations_in_file() const { return realizations_in_file_; }
    std::uint64_t checkpoint_realizations() const { return checkpoint_; }

    // Open (creating or continuing) the record file for a run that has already
    // decided, from its CSV checkpoint, that `completed` trajectories are done.
    //
    // An existing file must describe the same run byte for byte -- the header
    // holds every parameter that changes what a record means -- and is trimmed
    // to the trajectories the checkpoint accepted. Records past that point
    // belong to a trajectory the CSV never counted, so keeping them would
    // double-count it once the run redoes it.
    void open(const std::string &path, const std::string &header_text, std::size_t value_count,
              std::uint64_t completed)
    {
        if (value_count == 0 || value_count > MAX_VALUES)
        {
            throw std::invalid_argument("A record carries between 1 and 4 observables.");
        }
        path_ = path;
        checkpoint_ = completed;
        value_count_ = value_count;
        record_bytes_ = record_bytes(value_count);
        flush_threshold_ = flush_threshold();

        const FileInfo info = inspect(path_);
        if (!info.exists)
        {
            ensure_output_parent_directory(path_);
            write_new_file(header_text);
        }
        else
        {
            if (info.header_text != header_text)
            {
                throw std::runtime_error(
                    path_ + " describes a different run than this one. Finish it with the "
                            "binary that wrote it, move it aside, or set MIPT_DIST_RECORDS=0 "
                            "to run without per-record output.");
            }
            if (info.record_bytes != record_bytes_)
            {
                throw std::runtime_error(path_ + " has a record size of " +
                                         std::to_string(info.record_bytes) +
                                         " bytes; this run writes " +
                                         std::to_string(record_bytes_) + ".");
            }
            const std::uint64_t keep = records_below_realization(path_, info, completed);
            inherited_records_ = keep;
            truncated_ = keep < info.record_count;
            if (keep > 0)
            {
                std::ifstream reader(path_, std::ios::binary);
                realizations_in_file_ =
                    static_cast<std::uint64_t>(read_realization_at(reader, info, keep - 1)) + 1u;
            }
            std::error_code error;
            std::filesystem::resize_file(
                path_, info.payload_offset + keep * static_cast<std::uint64_t>(record_bytes_),
                error);
            if (error)
            {
                throw std::runtime_error("Could not trim record file " + path_ + " to its "
                                         "checkpoint: " + error.message());
            }
        }

        file_.open(path_, std::ios::binary | std::ios::app);
        if (!file_)
        {
            throw std::runtime_error("Could not open record file " + path_ + " for appending.");
        }
        active_ = true;
    }

    // A record whose observables are all known now. Returns nothing: there is
    // nothing left to fill in.
    void add(std::uint32_t realization, std::uint32_t geometry, std::uint32_t embedding,
             const double *values)
    {
        if (!active_)
        {
            return;
        }
        push(realization, geometry, embedding, values, 0);
        maybe_flush();
    }

    // A record with `pending` observables still to arrive from a background
    // solver. The returned sequence number addresses it until then.
    std::uint64_t add_pending(std::uint32_t realization, std::uint32_t geometry,
                              std::uint32_t embedding, const double *values, int pending)
    {
        if (!active_)
        {
            return 0;
        }
        const std::uint64_t sequence = next_sequence_;
        push(realization, geometry, embedding, values, pending);
        return sequence;
    }

    // Fill one deferred observable. Safe to call for a sequence number that
    // was never issued (records disabled), which is what lets the SDP queue
    // stay ignorant of whether a sink exists.
    void resolve(std::uint64_t sequence, std::size_t slot, double value)
    {
        if (!active_ || sequence < first_sequence_)
        {
            return;
        }
        const std::uint64_t offset = sequence - first_sequence_;
        if (offset >= buffer_.size())
        {
            return;
        }
        Pending &record = buffer_[static_cast<std::size_t>(offset)];
        if (slot < value_count_)
        {
            record.values[slot] = value;
        }
        if (record.unresolved > 0)
        {
            --record.unresolved;
        }
    }

    // Write out every record whose observables have all arrived, stopping at
    // the trajectory boundary ahead of the first that has not. Keeping the
    // file to whole trajectories is what makes the resume trim exact.
    void flush()
    {
        if (!active_ || buffer_.empty())
        {
            return;
        }
        std::size_t resolved = 0;
        while (resolved < buffer_.size() && buffer_[resolved].unresolved == 0)
        {
            ++resolved;
        }
        if (resolved < buffer_.size())
        {
            const std::uint32_t boundary = buffer_[resolved].realization;
            while (resolved > 0 && buffer_[resolved - 1].realization == boundary)
            {
                --resolved;
            }
        }
        if (resolved == 0)
        {
            return;
        }
        bytes_.clear();
        bytes_.reserve(resolved * record_bytes_);
        for (std::size_t index = 0; index < resolved; ++index)
        {
            serialize(buffer_[index]);
        }
        file_.write(bytes_.data(), static_cast<std::streamsize>(bytes_.size()));
        if (!file_)
        {
            throw std::runtime_error("Failed while writing record file " + path_ + ".");
        }
        file_.flush();
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(resolved));
        first_sequence_ += resolved;
        written_ += resolved;
    }

    std::uint64_t written() const { return written_; }
    // Records added but never flushed. Non-zero at the end of a run means
    // solves that never came back -- the same loss the CSV documents.
    std::size_t pending() const { return buffer_.size(); }

    void close()
    {
        if (!active_)
        {
            return;
        }
        flush();
        file_.close();
        active_ = false;
    }

  private:
    struct Pending
    {
        std::uint32_t realization = 0;
        std::uint16_t geometry = 0;
        std::uint16_t embedding = 0;
        int unresolved = 0;
        std::array<double, MAX_VALUES> values{};
    };

    void write_new_file(const std::string &header_text)
    {
        std::ofstream out(path_, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            throw std::runtime_error("Could not create record file " + path_ + ".");
        }
        char preamble[PREAMBLE_BYTES] = {};
        std::memcpy(preamble, MAGIC, sizeof(MAGIC));
        const std::uint32_t version = FORMAT_VERSION;
        const std::uint32_t bytes = static_cast<std::uint32_t>(record_bytes_);
        const std::uint32_t header_bytes = static_cast<std::uint32_t>(header_text.size());
        const std::uint32_t reserved = 0;
        std::memcpy(preamble + 8, &version, 4);
        std::memcpy(preamble + 12, &bytes, 4);
        std::memcpy(preamble + 16, &header_bytes, 4);
        std::memcpy(preamble + 20, &reserved, 4);
        out.write(preamble, static_cast<std::streamsize>(PREAMBLE_BYTES));
        out.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
        out.flush();
        if (!out)
        {
            throw std::runtime_error("Failed while writing the header of " + path_ + ".");
        }
    }

    void push(std::uint32_t realization, std::uint32_t geometry, std::uint32_t embedding,
              const double *values, int pending)
    {
        if (geometry > std::numeric_limits<std::uint16_t>::max() ||
            embedding > std::numeric_limits<std::uint16_t>::max())
        {
            throw std::runtime_error("A geometry or embedding index exceeded the 16-bit record "
                                     "field. Set MIPT_DIST_RECORDS=0 to run without per-record "
                                     "output.");
        }
        Pending record;
        record.realization = realization;
        record.geometry = static_cast<std::uint16_t>(geometry);
        record.embedding = static_cast<std::uint16_t>(embedding);
        record.unresolved = pending;
        for (std::size_t index = 0; index < value_count_; ++index)
        {
            record.values[index] = values[index];
        }
        buffer_.push_back(record);
        ++next_sequence_;
    }

    void maybe_flush()
    {
        if (buffer_.size() >= flush_threshold_)
        {
            flush();
        }
    }

    void serialize(const Pending &record)
    {
        char scratch[ID_BYTES + MAX_VALUES * sizeof(double)];
        std::memcpy(scratch, &record.realization, 4);
        std::memcpy(scratch + 4, &record.geometry, 2);
        std::memcpy(scratch + 6, &record.embedding, 2);
        std::memcpy(scratch + ID_BYTES, record.values.data(), value_count_ * sizeof(double));
        bytes_.insert(bytes_.end(), scratch, scratch + record_bytes_);
    }

    bool active_ = false;
    std::string path_;
    std::size_t value_count_ = 0;
    std::size_t record_bytes_ = 0;
    std::size_t flush_threshold_ = 8192;
    std::ofstream file_;
    std::deque<Pending> buffer_;
    std::vector<char> bytes_;
    std::uint64_t first_sequence_ = 0;
    std::uint64_t next_sequence_ = 0;
    std::uint64_t written_ = 0;
    std::uint64_t inherited_records_ = 0;
    std::uint64_t realizations_in_file_ = 0;
    std::uint64_t checkpoint_ = 0;
    bool truncated_ = false;
};

// The one line every protocol prints about its record file, plus the warning
// that matters when the feature is off: a stale file from an earlier run is
// about to fall silently out of step with the CSV beside it.
inline void announce(std::ostream &out, const RecordSink &sink, const std::string &csv_path)
{
    if (!sink.active())
    {
        out << "records=off (MIPT_DIST_RECORDS=0)\n";
        const std::string path = records_path_for(csv_path);
        std::error_code error;
        if (std::filesystem::exists(path, error) && !error)
        {
            out << "  WARNING: " << path
                << " exists and will NOT be extended, so it will hold fewer "
                   "trajectories than the CSV beside it.\n";
        }
        return;
    }
    out << "records=" << sink.path() << " (" << sink.value_count() << " observables/record, "
        << record_bytes(sink.value_count()) << " bytes/record)\n";
    if (sink.inherited_records() > 0 || sink.truncated_on_open())
    {
        out << "  continuing from " << sink.inherited_records() << " record(s) over "
            << sink.realizations_in_file() << " trajectory/ies";
        if (sink.truncated_on_open())
        {
            out << "; trailing records past the CSV checkpoint were discarded";
        }
        out << ".\n";
    }
    if (sink.realizations_in_file() < sink.checkpoint_realizations())
    {
        // Only reachable at k=3, and only after a kill: a record waits for the
        // solve that fills its GMN column, so the ones still with the solver
        // -- and everything queued behind them -- never reached the file. The
        // gap is in whole trajectories, so each realization_id in the file is
        // still a complete cluster and a bootstrap over them stays valid; the
        // file simply holds fewer of them than the CSV counted.
        out << "  NOTE: the CSV checkpoint counts " << sink.checkpoint_realizations()
            << " trajectory/ies but the record file holds " << sink.realizations_in_file()
            << ". The difference was still with the SDP solver when the run stopped, so "
               "those whole trajectories are absent from the records; the aggregate has "
               "them.\n";
    }
}

} // namespace mipt::dist::records
