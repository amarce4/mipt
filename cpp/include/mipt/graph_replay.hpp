#pragma once

// dist_graph_replay.exe: regenerate a finished dist_scaling.exe run's
// spacetime graphs from its record file, on the host, to add the graph
// diagnostics the run did not compute or to print the paths behind selected
// pairs.
//
//   augment <in_records.bin> [out_records.bin]
//   paths   <records.bin> <selection.csv> [out.csv]
//
// Everything that decides anything lives here, so `make test-dist` pins it on
// the host (tests/graph_replay_tests.cpp); apps/dist_graph_replay.cpp is main().
//
// Why a replay can do this at all
// -------------------------------
// The temporal-turn and doubled-network diagnostics (PairTurns,
// MIPT_DIST_TURN_DIAGNOSTICS) are a function of the spacetime graph alone, and
// the graph is a function of the layer draw alone: which gates, and which sites
// each layer measures -- never which outcomes. Since 2026-09-09 dist_scaling.exe
// draws realization r's layers from trajectory_seed(master_seed, r) and writes
// the master seed into the record header, so every graph of a finished run can
// be recovered from the file without the state vector, the GPU or CUDA-Q. A run
// made before the diagnostics existed -- the 100k-trajectory L=20 RPPU
// gate-path run is 19M records -- therefore does not have to be rerun to get
// them: regenerate each graph in a few tens of milliseconds and append the
// values to the records it already holds.
//
// What makes an augmented file trustworthy
// ----------------------------------------
// A regenerated graph is worth something only if it is *the* graph the run
// measured, and nothing here assumes that it is.
//
//   * The run is reconstructed from the header, and the reconstruction is
//     proven exact before a single record is read: records::pair_header_text of
//     the reconstructed RunConfig has to reproduce the input header byte for
//     byte. Every double in a header is 17-digit append_double text, so parsing
//     it back is exact, and there is one renderer, so there is nothing to
//     drift. The output header is that same function with the turn diagnostics
//     on -- what dist_scaling.exe writes under MIPT_DIST_TURN_DIAGNOSTICS=1, by
//     construction rather than by imitation.
//   * Every stored record is checked against its regenerated graph: flag bits
//     0-4 against PairConnectivity::flags(), and the seven gate-path values
//     against records::fill_gate_record_values, the function production wrote
//     them with -- the three integer-valued ones bit for bit, the four
//     continuous capacities to GATE_VALUE_TOLERANCE (see there for why not bit
//     for bit), every NaN exactly where the file has one. The gate values are
//     what make the check sharp. A flag can agree by accident; a widest-path
//     entangling power is a continuous number computed from that
//     realization's own Haar draws, so a wrong seed, a wrong boundary
//     implementation or a drifted layer draw misses it by O(0.1), not by
//     1e-12. A file that already carries turn values is checked on those too,
//     exactly, turn flag bits included.
//   * One disagreement aborts the run, naming the realization, the pair, the
//     field, and both values in decimal and in hex. Output is written to
//     `<out>.partial` and renamed only after the last record, so a failed or
//     killed replay never leaves a complete-looking file behind.
//
// Three things that look like they need handling, and how they are handled
// ------------------------------------------------------------------------
//   * MIPT_DIST_CONN_PATHS is not in the header, and does not need to be. It
//     only fills PairConnectivity::shortest_path, which no record stores: the
//     record's `shortest_path` is GatePathMetrics' own breadth-first length,
//     computed whatever that setting was. The replay always builds the matrix.
//   * The renderer reads two things from the environment rather than from
//     RunConfig: the boundary implementation (MIPT_DIRECT_FERMION_BOUNDARY,
//     through RunConfig::boundary_implementation) and the three cuStateVec
//     knobs of the gate block (MIPT_CUSV_PARITY, _MEASURE_BITS, _BLOCK_TARGETS).
//     None of them is cached, so they are pinned from the header for the
//     duration of the rendering and restored afterwards -- on the calling
//     thread, before any worker exists, because setenv is not safe against a
//     concurrent getenv. The graph itself depends only on the boundary, which
//     the layer draw is handed explicitly; the cuStateVec knobs decide
//     measurement outcomes, which no graph sees.
//   * The tail. A record file holds whole trajectories -- RecordSink flushes
//     nothing else -- so the only way for one to end on a partial trajectory is
//     a kill in the middle of a write. `augment` then stops at the last whole
//     trajectory, drops the remainder and says how much it dropped, which is
//     what the dist_scaling.exe resume would do to the same file. A short
//     trajectory anywhere *before* the end is corruption and is refused.
//
// Throughput
// ----------
// One thread does all the I/O -- it reads whole trajectories, hands them to
// MIPT_GRAPH_REPLAY_THREADS workers, and writes their results strictly in file
// order -- and at most four trajectories per worker are in flight, so memory
// stays at a few MB whatever the size of the file. The graph build dominates
// completely: tens of milliseconds per N=20, T=81 trajectory with the gate
// paths and turns precomputed, against well under a millisecond of I/O.

#include "mipt/circuits/rppu_layer.hpp"
#include "mipt/dist_connectivity.hpp"
#include "mipt/dist_record_reader.hpp"
#include "mipt/dist_records.hpp"
#include "mipt/dist_scaling_csv.hpp"
#include "mipt/env.hpp"
#include "mipt/types.hpp"
#include "mipt/util/text.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace mipt::dist::graph_replay
{

// Exit statuses, shared by the CLI and its tests.
inline constexpr int STATUS_OK = 0;
inline constexpr int STATUS_ERROR = 1;      // I/O, or anything unforeseen
inline constexpr int STATUS_USAGE = 2;      // bad arguments, or an input this tool does not read
inline constexpr int STATUS_VALIDATION = 3; // the input disagrees with its own regeneration

// Arguments this tool cannot act on, or an input that is not a record file it
// knows how to replay. Nothing has been written when one is thrown.
class UsageError : public std::invalid_argument
{
  public:
    using std::invalid_argument::invalid_argument;
};

// The input disagrees with its own regeneration, or is inconsistent with its
// own header. Whatever had been written is removed before the CLI exits.
class ValidationError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

// The flag bits of a k=2 record, by who decides them. The graph bits are what a
// regeneration reproduces; the physics bits came from the state vector and are
// carried through untouched; the turn bits are what PairTurns::flags() can set.
// A bit outside all three is something no writer of this format sets.
inline constexpr std::uint32_t GRAPH_FLAG_BITS =
    FLAG_EVALUATED | FLAG_SURVIVES_I | FLAG_SURVIVES_J | FLAG_INTERIOR_PATH | FLAG_CONNECTED;
inline constexpr std::uint32_t PHYSICS_FLAG_BITS = FLAG_FN_POSITIVE | FLAG_MN_POSITIVE;
inline constexpr std::uint32_t TURN_FLAG_BITS =
    FLAG_CONNECTED_COMMON_SOURCE | FLAG_CONNECTED_CERTIFIED | FLAG_CONNECTED_DOUBLED;

// How closely a regenerated gate capacity has to reproduce the stored one.
//
// Production computes the capacities in its CUDA-Q build (clang, -O3
// -march=native, so FMA-contracted); this tool is a g++ host build. The same
// Haar draws then give the same entangling powers and block mixings to a few
// ulps, but not bit for bit. Measured on the 100k L=20 RPPU run: realization 0
// has a 1-ulp difference in widest_even_mix on 83 of its 190 records with
// every flag exact, and a g++ -O3 -march=native build misses by 2 ulps the
// other way, so no host flag set reproduces clang's contraction. The four
// continuous values (shortest_maximin_gate_ep, widest_gate_ep,
// widest_even_mix, widest_odd_mix, all in [0, 1]) are therefore compared to an
// absolute 1e-12: four decades above that rounding noise, and ten below the
// O(0.1) by which any other trajectory's gates would miss. Their NaN pattern,
// the three integer-valued gate fields, the flags, and every turn value stay
// exact, and the largest difference seen is reported so the margin is visible.
//
// What the tolerance cannot certify: the canonical paths are chosen by
// comparing capacities, so two *different* gates within rounding of each
// other at a deciding comparison could pick a different path here than
// production would have. Capacities are continuous, so that is a ~1e-15
// coincidence per comparison; the exactly-compared shortest_path_unitary_count
// comes out of the same selection and would expose many such flips.
inline constexpr double GATE_VALUE_TOLERANCE = 1.0e-12;

// How the continuous gate values agreed over some set of records.
struct GateAgreement
{
    std::uint64_t compared = 0;   // finite continuous values compared
    std::uint64_t inexact = 0;    // of those, not bitwise equal (all within tolerance)
    double max_difference = 0.0;  // the largest |stored - regenerated|

    void merge(const GateAgreement &other)
    {
        compared += other.compared;
        inexact += other.inexact;
        max_difference = std::max(max_difference, other.max_difference);
    }
};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

// Worker threads. The I/O thread is extra, so the default leaves a core for it
// and for whatever else the box is running -- typically a GPU job feeding its
// own host threads.
inline int configured_threads()
{
    const long hardware = static_cast<long>(std::thread::hardware_concurrency());
    const long fallback = std::max(1L, std::min(8L, hardware - 1L));
    return static_cast<int>(env::integer("MIPT_GRAPH_REPLAY_THREADS", fallback, 1, 256));
}

// Augment only the first N whole trajectories (0 = all). For timing runs: the
// output is then a prefix of the run, exactly like the record file of a run
// killed at that point, so it is a valid file but not the whole run.
inline std::uint64_t configured_max_realizations()
{
    return static_cast<std::uint64_t>(env::integer("MIPT_GRAPH_REPLAY_MAX_REALIZATIONS", 0, 0,
                                                   std::numeric_limits<long>::max()));
}

// `x_records.bin` -> `x_turns_records.bin`, so the augmented file still globs
// with the record files data_analysis looks for. Anything else gains the suffix.
inline std::string default_augment_output(const std::string &input)
{
    const std::string records_suffix = "_records.bin";
    const std::string bin_suffix = ".bin";
    auto ends_with = [&](const std::string &suffix) {
        return input.size() >= suffix.size() &&
               input.compare(input.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (ends_with(records_suffix))
    {
        return input.substr(0, input.size() - records_suffix.size()) + "_turns_records.bin";
    }
    if (ends_with(bin_suffix))
    {
        return input.substr(0, input.size() - bin_suffix.size()) + "_turns_records.bin";
    }
    return input + "_turns_records.bin";
}

inline std::string default_paths_output(const std::string &selection)
{
    std::filesystem::path path(selection);
    path.replace_extension();
    return path.string() + "_paths.csv";
}

namespace detail
{
// ---------------------------------------------------------------------------
// Header fields. A missing or unparseable field means the file is not one this
// tool reads, which is a usage refusal rather than a validation failure.
// ---------------------------------------------------------------------------

inline const std::string &header_text(const records::RecordReader &reader, const std::string &key)
{
    if (!reader.has(key))
    {
        throw UsageError(reader.path() + " has no `" + key +
                         "` header field, so it is not a dist_scaling.exe k=2 record file this "
                         "tool can replay.");
    }
    return reader.field(key);
}

inline long long header_integer(const records::RecordReader &reader, const std::string &key)
{
    const std::string &text = header_text(reader, key);
    errno = 0;
    char *end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (text.empty() || errno != 0 || end != text.c_str() + text.size())
    {
        throw UsageError(reader.path() + ": header field `" + key + "` is not an integer (`" +
                         text + "`).");
    }
    return value;
}

inline std::uint64_t header_unsigned(const records::RecordReader &reader, const std::string &key)
{
    const std::string &text = header_text(reader, key);
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (text.empty() || text.front() == '-' || errno != 0 || end != text.c_str() + text.size())
    {
        throw UsageError(reader.path() + ": header field `" + key +
                         "` is not an unsigned integer (`" + text + "`).");
    }
    return static_cast<std::uint64_t>(value);
}

// Exactly the double the 17-digit text names: strtod rounds correctly, and
// append_double wrote enough digits for the round trip to be the identity.
inline double header_real(const records::RecordReader &reader, const std::string &key)
{
    const std::string &text = header_text(reader, key);
    errno = 0;
    char *end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (text.empty() || errno == ERANGE || end != text.c_str() + text.size())
    {
        throw UsageError(reader.path() + ": header field `" + key + "` is not a number (`" +
                         text + "`).");
    }
    return value;
}

// ---------------------------------------------------------------------------
// Spelling out a disagreement
// ---------------------------------------------------------------------------

inline std::uint64_t bits_of(double value)
{
    std::uint64_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

inline std::string hex_bits(std::uint64_t bits)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%016llx", static_cast<unsigned long long>(bits));
    return buffer;
}

inline std::string describe_value(double value)
{
    std::string out;
    append_double(out, value);
    out += " (";
    out += hex_bits(bits_of(value));
    out += ')';
    return out;
}

inline std::string describe_flags(std::uint32_t mask)
{
    static const char *const names[] = {"evaluated",     "survives_i",  "survives_j",
                                        "interior_path", "connected",   "fn_positive",
                                        "mn_positive",   "connected_common_source",
                                        "connected_certified", "connected_doubled"};
    constexpr int named = static_cast<int>(sizeof(names) / sizeof(names[0]));
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "0x%x", mask);
    std::string out = buffer;
    out += " [";
    bool first = true;
    for (int bit = 0; bit < 32; ++bit)
    {
        if ((mask & (1u << bit)) == 0u)
        {
            continue;
        }
        if (!first)
        {
            out += ',';
        }
        first = false;
        if (bit < named)
        {
            out += names[bit];
        }
        else
        {
            out += "bit" + std::to_string(bit);
        }
    }
    out += ']';
    return out;
}

// The first line on which a stored header and its reconstruction differ.
inline std::string first_difference(const std::string &stored, const std::string &rebuilt)
{
    std::istringstream a(stored);
    std::istringstream b(rebuilt);
    std::string line_a;
    std::string line_b;
    for (std::size_t line = 1;; ++line)
    {
        const bool have_a = static_cast<bool>(std::getline(a, line_a));
        const bool have_b = static_cast<bool>(std::getline(b, line_b));
        if (!have_a && !have_b)
        {
            return "the two differ only in trailing bytes";
        }
        if (!have_a)
        {
            return "header line " + std::to_string(line) +
                   " is missing from the file; the reconstruction has `" + line_b + "`";
        }
        if (!have_b)
        {
            return "header line " + std::to_string(line) + " `" + line_a +
                   "` has no counterpart in the reconstruction";
        }
        if (line_a != line_b)
        {
            return "header line " + std::to_string(line) + " is `" + line_a +
                   "` in the file but `" + line_b + "` in the reconstruction";
        }
    }
}

// ---------------------------------------------------------------------------
// CSV cells (the selection is written by pandas; quoting is handled anyway)
// ---------------------------------------------------------------------------

inline std::vector<std::string> split_csv_line(std::string line)
{
    if (!line.empty() && line.back() == '\r')
    {
        line.pop_back();
    }
    std::vector<std::string> out;
    std::string cell;
    bool quoted = false;
    for (std::size_t k = 0; k < line.size(); ++k)
    {
        const char ch = line[k];
        if (quoted)
        {
            if (ch == '"' && k + 1 < line.size() && line[k + 1] == '"')
            {
                cell += '"';
                ++k;
            }
            else if (ch == '"')
            {
                quoted = false;
            }
            else
            {
                cell += ch;
            }
        }
        else if (ch == '"')
        {
            quoted = true;
        }
        else if (ch == ',')
        {
            out.push_back(cell);
            cell.clear();
        }
        else
        {
            cell += ch;
        }
    }
    out.push_back(cell);
    return out;
}

inline std::string csv_cell(const std::string &text)
{
    if (text.find_first_of(",\"\n\r") == std::string::npos)
    {
        return text;
    }
    return util::csv_quote(text);
}

inline long long parse_cell_integer(const std::string &text, const std::string &what)
{
    errno = 0;
    char *end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (text.empty() || errno != 0 || end != text.c_str() + text.size())
    {
        throw UsageError(what + " is not an integer (`" + text + "`).");
    }
    return value;
}

// Sets environment variables for its lifetime and restores what was there.
// Calling thread only, and only while no worker exists: setenv is not safe
// against a concurrent getenv.
class ScopedEnvironment
{
  public:
    ScopedEnvironment() = default;
    ScopedEnvironment(const ScopedEnvironment &) = delete;
    ScopedEnvironment &operator=(const ScopedEnvironment &) = delete;
    ~ScopedEnvironment()
    {
        for (auto it = saved_.rbegin(); it != saved_.rend(); ++it)
        {
            if (it->had)
            {
                ::setenv(it->name.c_str(), it->value.c_str(), 1);
            }
            else
            {
                ::unsetenv(it->name.c_str());
            }
        }
    }

    void set(const char *name, const std::string &value)
    {
        const char *old = std::getenv(name);
        saved_.push_back({name, old != nullptr, old != nullptr ? std::string(old) : std::string()});
        ::setenv(name, value.c_str(), 1);
    }

  private:
    struct Saved
    {
        std::string name;
        bool had = false;
        std::string value;
    };
    std::vector<Saved> saved_;
};

// One periodic progress line on the log stream.
class Progress
{
  public:
    Progress(std::ostream *log, const char *what, std::uint64_t total)
        : log_(log), what_(what), total_(total), start_(std::chrono::steady_clock::now()),
          last_(start_)
    {
    }

    double elapsed() const
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

    void update(std::uint64_t done)
    {
        const auto now = std::chrono::steady_clock::now();
        if (log_ == nullptr || now - last_ < std::chrono::seconds(5))
        {
            return;
        }
        last_ = now;
        const double seconds = elapsed();
        const double rate = seconds > 0.0 ? static_cast<double>(done) / seconds : 0.0;
        (*log_) << what_ << ": " << done << '/' << total_ << " realization(s)";
        if (total_ > 0)
        {
            (*log_) << " (" << std::fixed << std::setprecision(1)
                    << 100.0 * static_cast<double>(done) / static_cast<double>(total_) << "%)";
        }
        (*log_) << ", " << std::fixed << std::setprecision(1) << rate << "/s, ETA "
                << util::format_duration(rate > 0.0 && total_ >= done
                                             ? static_cast<double>(total_ - done) / rate
                                             : -1.0,
                                         util::DurationStyle::Compact)
                << std::defaultfloat << '\n';
        log_->flush();
    }

  private:
    std::ostream *log_ = nullptr;
    const char *what_ = "";
    std::uint64_t total_ = 0;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_;
};

// A file written under `<path>.partial` and renamed into place by commit().
// Destroyed uncommitted -- an exception anywhere -- it removes the partial file,
// so a failed run leaves nothing that looks like output.
class AtomicOutput
{
  public:
    explicit AtomicOutput(std::string path) : path_(std::move(path)), partial_(path_ + ".partial")
    {
        ensure_output_parent_directory(path_);
        file_ = std::fopen(partial_.c_str(), "wb");
        if (file_ == nullptr)
        {
            throw std::runtime_error("Could not create " + partial_ + ": " + std::strerror(errno));
        }
        std::setvbuf(file_, nullptr, _IOFBF, 1u << 22);
    }
    AtomicOutput(const AtomicOutput &) = delete;
    AtomicOutput &operator=(const AtomicOutput &) = delete;

    ~AtomicOutput()
    {
        if (file_ != nullptr)
        {
            std::fclose(file_);
        }
        if (!committed_)
        {
            std::error_code ignored;
            std::filesystem::remove(partial_, ignored);
        }
    }

    void write(const void *data, std::size_t bytes)
    {
        if (bytes > 0 && std::fwrite(data, 1, bytes, file_) != bytes)
        {
            throw std::runtime_error("Failed while writing " + partial_ + ": " +
                                     std::strerror(errno));
        }
    }

    void write(const std::string &text) { write(text.data(), text.size()); }

    void commit()
    {
        if (std::fflush(file_) != 0 || ::fsync(::fileno(file_)) != 0)
        {
            throw std::runtime_error("Failed while flushing " + partial_ + ": " +
                                     std::strerror(errno));
        }
        const int closed = std::fclose(file_);
        file_ = nullptr;
        if (closed != 0)
        {
            throw std::runtime_error("Failed while closing " + partial_ + ".");
        }
        std::error_code error;
        std::filesystem::rename(partial_, path_, error);
        if (error)
        {
            throw std::runtime_error("Could not rename " + partial_ + " to " + path_ + ": " +
                                     error.message());
        }
        committed_ = true;
    }

  private:
    std::string path_;
    std::string partial_;
    std::FILE *file_ = nullptr;
    bool committed_ = false;
};

// The same preamble RecordSink::write_new_file writes.
inline void write_record_preamble(AtomicOutput &out, std::size_t record_bytes,
                                  const std::string &header)
{
    char preamble[records::PREAMBLE_BYTES] = {};
    std::memcpy(preamble, records::MAGIC, sizeof(records::MAGIC));
    const std::uint32_t version = records::FORMAT_VERSION;
    const std::uint32_t bytes = static_cast<std::uint32_t>(record_bytes);
    const std::uint32_t header_bytes = static_cast<std::uint32_t>(header.size());
    const std::uint32_t reserved = 0;
    std::memcpy(preamble + 8, &version, 4);
    std::memcpy(preamble + 12, &bytes, 4);
    std::memcpy(preamble + 16, &header_bytes, 4);
    std::memcpy(preamble + 20, &reserved, 4);
    out.write(preamble, sizeof(preamble));
    out.write(header);
}

inline void refuse_overwriting_input(const std::string &input, const std::string &output)
{
    std::error_code error;
    if (std::filesystem::exists(output, error) && std::filesystem::equivalent(input, output, error))
    {
        throw UsageError("The output " + output + " is the input file itself.");
    }
}

// ---------------------------------------------------------------------------
// An order-preserving bounded parallel map
//
// `produce` (calling thread) hands out jobs until it returns false; `work` (a
// worker, with that worker's own workspace) turns one into a result; `consume`
// (calling thread) receives results strictly in production order. At most
// `window` jobs are in flight, which is what bounds memory: a finished result
// only ever waits for the consumer, never on an unbounded producer.
//
// An exception from any of the three stops everything. A worker's goes into
// its job's slot, so the consumer meets the *earliest* failing job first --
// every job before it was handed out and so completes -- which makes the
// failure reported independent of the thread count.
// ---------------------------------------------------------------------------

template <typename Job, typename Result, typename Workspace, typename Produce, typename Work,
          typename Consume>
void run_ordered(int threads, std::size_t window, Produce produce, Work work, Consume consume)
{
    struct Slot
    {
        Result result{};
        std::exception_ptr error;
    };
    std::mutex mutex;
    std::condition_variable job_ready;
    std::condition_variable result_ready;
    std::deque<std::pair<std::uint64_t, Job>> queue;
    std::map<std::uint64_t, Slot> finished;
    bool stopping = false;

    auto worker = [&]() {
        Workspace workspace;
        for (;;)
        {
            std::pair<std::uint64_t, Job> item;
            {
                std::unique_lock<std::mutex> lock(mutex);
                job_ready.wait(lock, [&] { return stopping || !queue.empty(); });
                if (stopping)
                {
                    return;
                }
                item = std::move(queue.front());
                queue.pop_front();
            }
            Slot slot;
            try
            {
                slot.result = work(std::move(item.second), workspace);
            }
            catch (...)
            {
                slot.error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                finished.emplace(item.first, std::move(slot));
            }
            result_ready.notify_all();
        }
    };

    std::vector<std::thread> pool;
    struct Joiner
    {
        std::mutex &mutex;
        std::condition_variable &job_ready;
        bool &stopping;
        std::vector<std::thread> &pool;
        ~Joiner()
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
            }
            job_ready.notify_all();
            for (std::thread &thread : pool)
            {
                thread.join();
            }
        }
    } joiner{mutex, job_ready, stopping, pool};
    for (int t = 0; t < std::max(1, threads); ++t)
    {
        pool.emplace_back(worker);
    }

    window = std::max<std::size_t>(1, window);
    std::uint64_t produced = 0;
    std::uint64_t consumed = 0;
    bool more = true;
    for (;;)
    {
        while (more && produced - consumed < window)
        {
            Job job;
            if (!produce(job))
            {
                more = false;
                break;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                queue.emplace_back(produced++, std::move(job));
            }
            job_ready.notify_one();
        }
        if (consumed == produced)
        {
            break;
        }
        Slot slot;
        {
            std::unique_lock<std::mutex> lock(mutex);
            result_ready.wait(lock, [&] { return finished.count(consumed) != 0; });
            auto node = finished.extract(consumed);
            slot = std::move(node.mapped());
        }
        if (slot.error)
        {
            std::rethrow_exception(slot.error);
        }
        consume(std::move(slot.result));
        ++consumed;
    }
}
} // namespace detail

// ---------------------------------------------------------------------------
// The run, reconstructed from the header
// ---------------------------------------------------------------------------

struct ReplayPlan
{
    std::string input_path;
    // The run as the input header describes it, and the same run with the turn
    // diagnostics on -- which is what the output header is rendered from.
    RunConfig config;
    RunConfig output_config;
    bool input_has_turns = false;
    bool jw_boundary = false;
    std::string boundary;
    std::vector<PairBin> bins;
    // {geometry_id, embedding_id, site_1, site_2} in all_unordered_pairs
    // order: exactly the rows PairProtocol's constructor writes into the
    // embedding table. (Both live in dist_scaling.hpp, which pulls in CUDA-Q,
    // so the four lines are mirrored below; the header comparison is what
    // proves the mirror right.)
    std::vector<std::array<int, 4>> embeddings;
    std::string output_header;
    std::vector<std::string> input_observables;
    std::vector<std::string> output_observables;
    std::size_t input_record_bytes = 0;
    std::size_t output_record_bytes = 0;
    std::array<std::size_t, records::GATE_RECORD_VALUES> gate_slots{};
    // Integer-valued gate fields, compared bit for bit; the rest are the
    // continuous capacities GATE_VALUE_TOLERANCE is for.
    std::array<bool, records::GATE_RECORD_VALUES> gate_exact{};
    // Output slots of the turn values -- the input's too when it has them.
    std::array<std::size_t, records::TURN_RECORD_VALUES> turn_slots{};
    int fmn_slot = -1;
    std::uint32_t known_input_flags = 0;
    std::vector<std::size_t> geometry_offset;
    std::vector<int> pair_of_flat_id;
    std::vector<int> pair_of_sites; // n*n, either orientation

    std::size_t records_per_trajectory() const { return embeddings.size(); }

    int pair_index(std::uint32_t geometry, std::uint32_t embedding) const
    {
        if (geometry >= bins.size() || embedding >= bins[geometry].embedding_count)
        {
            return -1;
        }
        return pair_of_flat_id[geometry_offset[geometry] + embedding];
    }

    int pair_of(int site_a, int site_b) const
    {
        const int n = config.n;
        if (site_a < 0 || site_b < 0 || site_a >= n || site_b >= n || site_a == site_b)
        {
            return -1;
        }
        return pair_of_sites[static_cast<std::size_t>(site_a) * static_cast<std::size_t>(n) +
                             static_cast<std::size_t>(site_b)];
    }
};

inline ReplayPlan make_plan(const records::RecordReader &reader)
{
    ReplayPlan plan;
    const std::string &path = reader.path();
    plan.input_path = path;

    // ------------------------------------------------------------ refusals
    if (reader.version() != records::FORMAT_VERSION)
    {
        throw UsageError(path + " is a format v" + std::to_string(reader.version()) +
                         " record file. The replay needs v" +
                         std::to_string(records::FORMAT_VERSION) +
                         ": connectivity flags, the embedding table, and the gate-path values a "
                         "regeneration is validated against.");
    }
    if (detail::header_integer(reader, "k") != 2)
    {
        throw UsageError(path + " is a k=" + reader.field("k") +
                         " record file; the graph diagnostics are per pair, so only k=2 files "
                         "can be replayed.");
    }
    const long long circ_type = detail::header_integer(reader, "circ_type");
    if (circ_type != static_cast<long long>(CircuitType::FermionRPPU) &&
        circ_type != static_cast<long long>(CircuitType::QubitRPPU))
    {
        throw UsageError(path + " is circ_type " + std::to_string(circ_type) +
                         "; the gate-path and turn diagnostics exist for RPPU (2) and qRPPU (4) "
                         "only.");
    }
    if (detail::header_text(reader, "connectivity") != "1")
    {
        throw UsageError(path + " was written with MIPT_DIST_CONNECTIVITY=0, so it carries no "
                                "graph flags to validate a regeneration against.");
    }
    if (!reader.has("gate_path_diagnostics") || reader.field("gate_path_diagnostics") != "1")
    {
        throw UsageError(path + " was written without MIPT_DIST_GATE_DIAGNOSTICS=1. The turn "
                                "shapes are read off the gate diagnostics' canonical paths, and "
                                "the stored gate values are what proves a regenerated graph is "
                                "the run's own.");
    }

    // ------------------------------------------------------ the RunConfig
    RunConfig &config = plan.config;
    config.k = 2;
    config.n = static_cast<int>(detail::header_integer(reader, "N"));
    config.periods = static_cast<int>(detail::header_integer(reader, "periods"));
    config.p = detail::header_real(reader, "p");
    config.realizations = static_cast<int>(detail::header_integer(reader, "realizations"));
    config.type = static_cast<CircuitType>(circ_type);
    try
    {
        config.record_detail = parse_record_detail(detail::header_text(reader, "record_detail"));
    }
    catch (const std::invalid_argument &error)
    {
        throw UsageError(path + ": " + error.what());
    }
    config.record_stride = static_cast<long>(detail::header_integer(reader, "record_stride"));
    config.pair_zero_tol = detail::header_real(reader, "pair_zero_tol");
    config.occupation_mi_tol = detail::header_real(reader, "occupation_mi_tol");
    config.channel_floor = detail::header_real(reader, "channel_floor");
    config.statevector_precision =
        static_cast<int>(detail::header_integer(reader, "statevector_precision"));
    config.seed = detail::header_unsigned(reader, "master_seed");
    config.connectivity = true;
    config.gate_path_diagnostics = true;
    if (reader.has("turn_diagnostics"))
    {
        if (reader.field("turn_diagnostics") != "1")
        {
            throw UsageError(path + " says turn_diagnostics=" + reader.field("turn_diagnostics") +
                             ".");
        }
        plan.input_has_turns = true;
    }
    config.turn_diagnostics = plan.input_has_turns;
    if (config.record_stride < 1)
    {
        throw UsageError(path + " says record_stride=" + std::to_string(config.record_stride) +
                         ".");
    }
    try
    {
        validate_args(config);
    }
    catch (const std::invalid_argument &error)
    {
        throw UsageError(path + " describes a run dist_scaling.exe would refuse: " +
                         error.what());
    }

    plan.boundary = detail::header_text(reader, "boundary_implementation");
    if (config.type == CircuitType::FermionRPPU)
    {
        if (plan.boundary == "jw_string")
        {
            plan.jw_boundary = true;
        }
        else if (plan.boundary != "fswap_network")
        {
            throw UsageError(path + " names boundary_implementation=" + plan.boundary +
                             ", which is neither jw_string nor fswap_network.");
        }
    }

    // --------------------------------------------- bins and embedding rows
    const int n = config.n;
    plan.bins = make_pair_bins(n);
    std::vector<std::uint32_t> seen(plan.bins.size(), 0u);
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            const std::size_t bin = static_cast<std::size_t>(periodic_distance(i, j, n)) - 1u;
            plan.embeddings.push_back({static_cast<int>(bin), static_cast<int>(seen[bin]++), i, j});
        }
    }

    // -------------------------------------------- the reconstruction proof
    {
        detail::ScopedEnvironment environment;
        if (config.type == CircuitType::FermionRPPU)
        {
            environment.set("MIPT_DIRECT_FERMION_BOUNDARY", plan.jw_boundary ? "1" : "0");
        }
        environment.set("MIPT_CUSV_PARITY", detail::header_text(reader, "cusv_parity"));
        environment.set("MIPT_CUSV_MEASURE_BITS", detail::header_text(reader, "cusv_measure_bits"));
        environment.set("MIPT_CUSV_BLOCK_TARGETS",
                        detail::header_text(reader, "cusv_block_targets"));
        const std::string rebuilt = records::pair_header_text(config, plan.bins, plan.embeddings);
        if (rebuilt != reader.header_text())
        {
            throw ValidationError(
                path + "'s header is not the one dist_scaling.exe writes for the run it "
                       "describes, so that run cannot be reconstructed exactly and no "
                       "regenerated graph could be trusted to be its own. First difference: " +
                detail::first_difference(reader.header_text(), rebuilt) + ".");
        }
        plan.output_config = config;
        plan.output_config.turn_diagnostics = true;
        plan.output_header =
            records::pair_header_text(plan.output_config, plan.bins, plan.embeddings);
    }

    // ------------------------------------------------------- value slots
    plan.input_observables = reader.observables();
    plan.output_observables = records::pair_observable_names(plan.output_config);
    const std::size_t stored = plan.input_observables.size();
    const std::size_t appended = plan.input_has_turns ? 0u : records::TURN_RECORD_VALUES;
    if (plan.output_observables.size() != stored + appended ||
        !std::equal(plan.input_observables.begin(), plan.input_observables.end(),
                    plan.output_observables.begin()))
    {
        throw std::logic_error(
            "records::pair_observable_names no longer keeps a record's stored observables as a "
            "prefix of the turn-diagnostic layout, so augment cannot append to the records in "
            "place. graph_replay.hpp needs updating.");
    }
    auto slot_of = [](const std::vector<std::string> &names, const char *name) {
        const auto it = std::find(names.begin(), names.end(), name);
        return it == names.end() ? -1 : static_cast<int>(it - names.begin());
    };
    const auto &gate_names = records::gate_observable_names();
    for (std::size_t k = 0; k < gate_names.size(); ++k)
    {
        const int slot = slot_of(plan.input_observables, gate_names[k]);
        if (slot < 0)
        {
            throw UsageError(path + " carries no `" + std::string(gate_names[k]) +
                             "` observable.");
        }
        plan.gate_slots[k] = static_cast<std::size_t>(slot);
        const std::string name = gate_names[k];
        plan.gate_exact[k] = name == "component_size" || name == "shortest_path" ||
                             name == "shortest_path_unitary_count";
    }
    const auto &turn_names = records::turn_observable_names();
    for (std::size_t k = 0; k < turn_names.size(); ++k)
    {
        const int slot = slot_of(plan.output_observables, turn_names[k]);
        if (slot < 0)
        {
            throw std::logic_error("records::pair_observable_names does not declare `" +
                                   std::string(turn_names[k]) + "` with the turn diagnostics on.");
        }
        plan.turn_slots[k] = static_cast<std::size_t>(slot);
    }
    plan.fmn_slot = slot_of(plan.input_observables, "fmn");
    plan.input_record_bytes = reader.record_bytes();
    plan.output_record_bytes = records::record_bytes(plan.output_observables.size());
    plan.known_input_flags =
        GRAPH_FLAG_BITS | PHYSICS_FLAG_BITS | (plan.input_has_turns ? TURN_FLAG_BITS : 0u);

    // ------------------------------------------------------ lookup tables
    plan.geometry_offset.assign(plan.bins.size(), 0u);
    std::size_t flat = 0;
    for (std::size_t g = 0; g < plan.bins.size(); ++g)
    {
        plan.geometry_offset[g] = flat;
        flat += static_cast<std::size_t>(plan.bins[g].embedding_count);
    }
    plan.pair_of_flat_id.assign(flat, -1);
    plan.pair_of_sites.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), -1);
    for (std::size_t pair = 0; pair < plan.embeddings.size(); ++pair)
    {
        const auto &row = plan.embeddings[pair];
        plan.pair_of_flat_id[plan.geometry_offset[static_cast<std::size_t>(row[0])] +
                             static_cast<std::size_t>(row[1])] = static_cast<int>(pair);
        plan.pair_of_sites[static_cast<std::size_t>(row[2]) * static_cast<std::size_t>(n) +
                           static_cast<std::size_t>(row[3])] = static_cast<int>(pair);
        plan.pair_of_sites[static_cast<std::size_t>(row[3]) * static_cast<std::size_t>(n) +
                           static_cast<std::size_t>(row[2])] = static_cast<int>(pair);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Regenerating one trajectory's graph
// ---------------------------------------------------------------------------

struct GraphWorkspace
{
    std::vector<RppuLayer> layers;
    LogicalHistory history;
    ConnectivityIndex index;
};

inline void regenerate_graph(const ReplayPlan &plan, std::uint32_t realization,
                             GraphWorkspace &workspace)
{
    const RunConfig &config = plan.config;
    // Circuit1D::seed, verbatim: the layer rng is seeded from the two 32-bit
    // halves of trajectory_seed(master, r). (The measurement rng it seeds next
    // decides outcomes, which no graph depends on.)
    const std::uint64_t value = seeding::trajectory_seed(config.seed, realization);
    std::seed_seq sequence{static_cast<std::uint32_t>(value & 0xffffffffu),
                           static_cast<std::uint32_t>(value >> 32)};
    std::mt19937 rng;
    rng.seed(sequence);
    // The draw RppuCircuit1D / QrppuCircuit1D::simulate makes first, and the
    // only one the layer rng serves.
    if (config.type == CircuitType::FermionRPPU)
    {
        build_rppu_layers(workspace.layers, config.n, config.periods, config.p, true, rng,
                          plan.jw_boundary);
    }
    else
    {
        build_qrppu_layers(workspace.layers, config.n, config.periods, config.p, true, rng);
    }
    workspace.history = logical_history_from_rppu(workspace.layers, config.n);
    workspace.index.build(workspace.history, true, true, true);
    if (!workspace.index.ready() || !workspace.index.has_gate_metrics() ||
        !workspace.index.has_turns())
    {
        throw ValidationError("The regenerated graph of realization " +
                              std::to_string(realization) +
                              " is not ready or carries no finite gate capacities, which a "
                              "regenerated RPPU/qRPPU draw always does.");
    }
}

// ---------------------------------------------------------------------------
// One stored record against its regeneration
// ---------------------------------------------------------------------------

struct PairReplay
{
    int pair = -1;
    int i = 0;
    int j = 0;
    PairConnectivity conn;
    GatePathMetrics gate;
    PairTurns turns;
    std::array<double, records::TURN_RECORD_VALUES> turn_values{};
};

// Fills `out` with the regenerated pair and returns "" when the stored record
// agrees with it, or its first disagreement, spelled out. A record that names
// no pair of the header's embedding table is malformed rather than mismatched,
// and throws.
inline std::string check_record(const ReplayPlan &plan, const ConnectivityIndex &index,
                                std::uint32_t realization, std::uint32_t geometry,
                                std::uint32_t embedding, std::uint32_t flags, const double *values,
                                PairReplay &out, GateAgreement &agreement)
{
    out.pair = plan.pair_index(geometry, embedding);
    if (out.pair < 0)
    {
        throw ValidationError(plan.input_path + ": a record of realization " +
                              std::to_string(realization) + " names (geometry_id " +
                              std::to_string(geometry) + ", embedding_id " +
                              std::to_string(embedding) +
                              "), which the header's embedding table does not have.");
    }
    const auto &row = plan.embeddings[static_cast<std::size_t>(out.pair)];
    // The embedding table's orientation, which is the one PairProtocol queried:
    // component_size and the canonical paths are read from site_1's side.
    out.i = row[2];
    out.j = row[3];
    out.conn = index.query(out.i, out.j);
    out.gate = index.gate_path_metrics(out.i, out.j);
    out.turns = index.pair_turns(out.i, out.j);
    records::fill_turn_record_values(out.conn, out.turns, out.turn_values.data());

    auto where = [&]() {
        return "realization " + std::to_string(realization) + ", pair (" + std::to_string(out.i) +
               "," + std::to_string(out.j) + ") [geometry_id " + std::to_string(geometry) +
               ", embedding_id " + std::to_string(embedding) + "]: ";
    };
    if ((flags & ~plan.known_input_flags) != 0u)
    {
        return where() + "flag bits " + detail::describe_flags(flags & ~plan.known_input_flags) +
               " are set, which no writer of a file with this header sets";
    }
    if ((flags & GRAPH_FLAG_BITS) != out.conn.flags())
    {
        return where() + "connectivity flags (bits 0-4) stored " +
               detail::describe_flags(flags & GRAPH_FLAG_BITS) +
               " but the regenerated graph gives " + detail::describe_flags(out.conn.flags());
    }
    std::array<double, records::GATE_RECORD_VALUES> gate_values{};
    records::fill_gate_record_values(out.conn, out.gate, gate_values.data());
    const auto &gate_names = records::gate_observable_names();
    GateAgreement local;
    for (std::size_t k = 0; k < gate_values.size(); ++k)
    {
        const double stored = values[plan.gate_slots[k]];
        const double rebuilt = gate_values[k];
        if (detail::bits_of(stored) == detail::bits_of(rebuilt))
        {
            local.compared += (plan.gate_exact[k] || std::isnan(stored)) ? 0u : 1u;
            continue;
        }
        const double difference = std::abs(stored - rebuilt);
        if (plan.gate_exact[k] || std::isnan(stored) || std::isnan(rebuilt) ||
            !(difference <= GATE_VALUE_TOLERANCE))
        {
            return where() + "`" + gate_names[k] + "` stored " + detail::describe_value(stored) +
                   " but the regenerated graph gives " + detail::describe_value(rebuilt) +
                   (plan.gate_exact[k] ? " (compared exactly)"
                                       : " (tolerance " + detail::describe_value(GATE_VALUE_TOLERANCE) + ")");
        }
        ++local.compared;
        ++local.inexact;
        local.max_difference = std::max(local.max_difference, difference);
    }
    if (plan.input_has_turns)
    {
        if ((flags & TURN_FLAG_BITS) != out.turns.flags())
        {
            return where() + "turn flags stored " + detail::describe_flags(flags & TURN_FLAG_BITS) +
                   " but the regenerated graph gives " + detail::describe_flags(out.turns.flags());
        }
        const auto &turn_names = records::turn_observable_names();
        for (std::size_t k = 0; k < out.turn_values.size(); ++k)
        {
            const double stored = values[plan.turn_slots[k]];
            if (detail::bits_of(stored) != detail::bits_of(out.turn_values[k]))
            {
                return where() + "`" + turn_names[k] + "` stored " +
                       detail::describe_value(stored) + " but the regenerated graph gives " +
                       detail::describe_value(out.turn_values[k]);
            }
        }
    }
    agreement.merge(local);
    return {};
}

// Raises the realization's verdict: the first disagreement, and how many of its
// records disagree in all. One wrong record usually means a wrong trajectory,
// and the count is what tells the two apart.
inline void raise_mismatch(const ReplayPlan &plan, std::uint32_t realization,
                           std::size_t mismatched, const std::string &first)
{
    throw ValidationError(
        first + ". " + std::to_string(mismatched) + " of the " +
        std::to_string(plan.records_per_trajectory()) + " records of realization " +
        std::to_string(realization) + " disagree with the graph regenerated from master_seed=" +
        std::to_string(plan.config.seed) +
        ", so the regeneration is not the run's trajectory (or the file was altered) and nothing "
        "was written.");
}

// ---------------------------------------------------------------------------
// augment
// ---------------------------------------------------------------------------

struct AugmentOptions
{
    std::string input_path;
    std::string output_path; // empty: default_augment_output(input_path)
    int threads = 0;         // 0: configured_threads()
    std::uint64_t max_realizations = 0;
    std::ostream *log = &std::cerr; // progress and notes; null for silence
};

struct AugmentSummary
{
    std::string output_path;
    std::uint64_t records = 0;
    std::uint64_t realizations = 0;
    std::uint64_t mismatches = 0;
    // An unfinished tail: whole records of an incomplete final trajectory, and
    // the bytes of a partial record after them.
    std::uint64_t dropped_records = 0;
    std::uint64_t dropped_bytes = 0;
    // Kept realizations below the last one present that the file skips.
    std::uint64_t missing_realizations = 0;
    GateAgreement agreement;
    bool limited = false; // stopped by max_realizations
    bool input_had_turns = false;
    int threads = 0;
    double seconds = 0.0;
};

namespace detail
{
struct AugmentJob
{
    std::uint32_t realization = 0;
    std::vector<unsigned char> input;
};

struct AugmentResult
{
    std::vector<unsigned char> output;
    GateAgreement agreement;
};

inline std::uint32_t realization_of(const std::vector<unsigned char> &bytes, std::size_t record,
                                    std::size_t record_bytes)
{
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + record * record_bytes, 4);
    return value;
}

inline AugmentResult augment_realization(const ReplayPlan &plan, const AugmentJob &job,
                                         GraphWorkspace &workspace)
{
    regenerate_graph(plan, job.realization, workspace);
    const std::size_t count = plan.records_per_trajectory();
    const std::size_t in_bytes = plan.input_record_bytes;
    const std::size_t out_bytes = plan.output_record_bytes;
    const std::size_t stored_bytes = plan.input_observables.size() * sizeof(double);
    AugmentResult result;
    result.output.assign(count * out_bytes, 0u);
    std::vector<std::uint8_t> seen(count, 0u);
    std::vector<double> values(plan.input_observables.size());
    std::size_t mismatched = 0;
    std::string first;
    for (std::size_t r = 0; r < count; ++r)
    {
        const unsigned char *in = job.input.data() + r * in_bytes;
        std::uint32_t realization = 0;
        std::uint16_t geometry = 0;
        std::uint16_t embedding = 0;
        std::uint32_t flags = 0;
        std::uint32_t reserved = 0;
        std::memcpy(&realization, in, 4);
        std::memcpy(&geometry, in + 4, 2);
        std::memcpy(&embedding, in + 6, 2);
        std::memcpy(&flags, in + 8, 4);
        std::memcpy(&reserved, in + 12, 4);
        if (reserved != 0u)
        {
            throw ValidationError(plan.input_path + ": a record of realization " +
                                  std::to_string(realization) +
                                  " has a nonzero reserved word, which RecordSink never writes.");
        }
        std::memcpy(values.data(), in + records::ID_BYTES, stored_bytes);
        PairReplay pair;
        const std::string why =
            check_record(plan, workspace.index, realization, geometry, embedding, flags,
                         values.data(), pair, result.agreement);
        if (seen[static_cast<std::size_t>(pair.pair)]++ != 0u)
        {
            throw ValidationError(plan.input_path + ": realization " +
                                  std::to_string(realization) + " holds pair (" +
                                  std::to_string(pair.i) + "," + std::to_string(pair.j) +
                                  ") twice, so it is not one whole trajectory.");
        }
        if (!why.empty())
        {
            if (mismatched++ == 0)
            {
                first = why;
            }
            continue;
        }
        // Same identifiers, the turn bits added (or reproduced), every stored
        // value byte for byte, and the turn values in their slots.
        unsigned char *out = result.output.data() + r * out_bytes;
        const std::uint32_t out_flags = (flags & ~TURN_FLAG_BITS) | pair.turns.flags();
        std::memcpy(out, in, 8);
        std::memcpy(out + 8, &out_flags, 4);
        std::memcpy(out + records::ID_BYTES, in + records::ID_BYTES, stored_bytes);
        for (std::size_t k = 0; k < pair.turn_values.size(); ++k)
        {
            std::memcpy(out + records::ID_BYTES + plan.turn_slots[k] * sizeof(double),
                        &pair.turn_values[k], sizeof(double));
        }
    }
    if (mismatched > 0)
    {
        raise_mismatch(plan, job.realization, mismatched, first);
    }
    return result;
}
} // namespace detail

inline AugmentSummary augment(const AugmentOptions &options)
{
    const auto started = std::chrono::steady_clock::now();
    std::error_code exists_error;
    if (!std::filesystem::exists(options.input_path, exists_error))
    {
        throw UsageError("Record file " + options.input_path + " does not exist.");
    }
    records::RecordReader reader(options.input_path);
    const ReplayPlan plan = make_plan(reader);

    AugmentSummary summary;
    summary.output_path =
        options.output_path.empty() ? default_augment_output(options.input_path) : options.output_path;
    summary.threads = options.threads > 0 ? options.threads : configured_threads();
    summary.input_had_turns = plan.input_has_turns;
    detail::refuse_overwriting_input(options.input_path, summary.output_path);

    const std::size_t per_trajectory = plan.records_per_trajectory();
    const std::uint64_t record_count = reader.record_count();
    const std::uint64_t payload_offset = records::PREAMBLE_BYTES + reader.header_text().size();
    {
        std::error_code error;
        const std::uintmax_t size = std::filesystem::file_size(options.input_path, error);
        if (!error && size > payload_offset)
        {
            summary.dropped_bytes =
                (static_cast<std::uint64_t>(size) - payload_offset) % plan.input_record_bytes;
        }
    }
    std::uint64_t whole = record_count / per_trajectory;
    if (options.max_realizations > 0)
    {
        whole = std::min<std::uint64_t>(whole, options.max_realizations);
    }

    std::ostream *log = options.log;
    if (log != nullptr)
    {
        (*log) << "dist_graph_replay augment\n"
               << "  input=" << options.input_path << " (" << record_count << " records x "
               << plan.input_record_bytes << " bytes, " << plan.input_observables.size()
               << " observables" << (plan.input_has_turns ? ", turn values present" : "")
               << ")\n"
               << "  run: N=" << plan.config.n << " periods=" << plan.config.periods
               << " p=" << reader.field("p") << " circ_type=" << static_cast<int>(plan.config.type)
               << " boundary=" << plan.boundary << " master_seed=" << plan.config.seed
               << " record_stride=" << plan.config.record_stride
               << " realizations=" << plan.config.realizations << '\n'
               << "  header reconstructed exactly (" << reader.header_text().size() << " bytes)\n"
               << "  output=" << summary.output_path << " ("
               << plan.output_observables.size() << " observables, " << plan.output_record_bytes
               << " bytes/record)\n"
               << "  threads=" << summary.threads << ", whole trajectories to replay=" << whole
               << '\n';
        log->flush();
    }

    std::ifstream input(options.input_path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("Could not open " + options.input_path + " for reading.");
    }
    input.seekg(static_cast<std::streamoff>(payload_offset));

    detail::AtomicOutput output(summary.output_path);
    detail::write_record_preamble(output, plan.output_record_bytes, plan.output_header);

    const long stride = plan.config.record_stride;
    std::uint64_t taken = 0; // records handed to jobs
    std::uint64_t jobs = 0;
    bool have_previous = false;
    std::uint32_t previous = 0;
    detail::Progress progress(log, "augment", whole);

    auto read_records = [&](std::size_t count, std::vector<unsigned char> &bytes) {
        bytes.resize(count * plan.input_record_bytes);
        input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!input)
        {
            throw std::runtime_error("Failed while reading " + options.input_path + ".");
        }
        taken += count;
    };
    auto produce = [&](detail::AugmentJob &job) -> bool {
        if (options.max_realizations > 0 && jobs >= options.max_realizations)
        {
            summary.limited = taken < record_count;
            return false;
        }
        const std::uint64_t remaining = record_count - taken;
        if (remaining == 0)
        {
            return false;
        }
        if (remaining < per_trajectory)
        {
            // The unfinished tail. Still one realization, or it is corruption.
            std::vector<unsigned char> tail;
            read_records(static_cast<std::size_t>(remaining), tail);
            const std::uint32_t first = detail::realization_of(tail, 0, plan.input_record_bytes);
            for (std::size_t k = 1; k < remaining; ++k)
            {
                if (detail::realization_of(tail, k, plan.input_record_bytes) != first)
                {
                    throw ValidationError(options.input_path + " ends in records of more than "
                                                               "one realization after its last "
                                                               "whole trajectory.");
                }
            }
            if (have_previous && first <= previous)
            {
                throw ValidationError(options.input_path + " holds more records of realization " +
                                      std::to_string(first) + " than one trajectory has.");
            }
            summary.dropped_records = remaining;
            return false;
        }
        read_records(per_trajectory, job.input);
        const std::uint32_t realization =
            detail::realization_of(job.input, 0, plan.input_record_bytes);
        for (std::size_t k = 1; k < per_trajectory; ++k)
        {
            const std::uint32_t other = detail::realization_of(job.input, k, plan.input_record_bytes);
            if (other != realization)
            {
                throw ValidationError(options.input_path + ": realization " +
                                      std::to_string(realization) + " holds " + std::to_string(k) +
                                      " record(s) before realization " + std::to_string(other) +
                                      " begins; a whole trajectory holds " +
                                      std::to_string(per_trajectory) +
                                      ". The file is not a sequence of whole trajectories.");
            }
        }
        if (have_previous && realization <= previous)
        {
            throw ValidationError(options.input_path + ": realization " +
                                  std::to_string(realization) + " follows realization " +
                                  std::to_string(previous) +
                                  "; records are written in increasing realization order.");
        }
        if (static_cast<long long>(realization) >= plan.config.realizations ||
            !records::stride_selects(stride, realization))
        {
            throw ValidationError(options.input_path + " holds realization " +
                                  std::to_string(realization) + ", which a run of " +
                                  std::to_string(plan.config.realizations) +
                                  " trajectories with record_stride=" + std::to_string(stride) +
                                  " never writes.");
        }
        const std::uint64_t expected =
            have_previous ? static_cast<std::uint64_t>(previous) + static_cast<std::uint64_t>(stride) : 0u;
        if (realization > expected)
        {
            summary.missing_realizations += (realization - expected) / static_cast<std::uint64_t>(stride);
        }
        have_previous = true;
        previous = realization;
        job.realization = realization;
        ++jobs;
        return true;
    };
    auto work = [&](detail::AugmentJob &&job, GraphWorkspace &workspace) {
        return detail::augment_realization(plan, job, workspace);
    };
    auto consume = [&](detail::AugmentResult &&result) {
        output.write(result.output.data(), result.output.size());
        summary.records += per_trajectory;
        ++summary.realizations;
        summary.agreement.merge(result.agreement);
        progress.update(summary.realizations);
    };
    detail::run_ordered<detail::AugmentJob, detail::AugmentResult, GraphWorkspace>(
        summary.threads, 4u * static_cast<std::size_t>(summary.threads), produce, work, consume);
    output.commit();
    summary.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    if (log != nullptr)
    {
        if (summary.dropped_records > 0 || summary.dropped_bytes > 0)
        {
            (*log) << "NOTE: the input ends on an unfinished trajectory -- " << summary.dropped_records
                   << " whole record(s) of it and " << summary.dropped_bytes
                   << " byte(s) of a partial record -- which a kill inside a write leaves "
                      "behind. It was dropped, as the dist_scaling.exe resume would drop it; the "
                      "output holds whole trajectories only.\n";
        }
        if (summary.missing_realizations > 0)
        {
            (*log) << "NOTE: the input skips " << summary.missing_realizations
                   << " kept realization(s) below its last one; the output skips the same.\n";
        }
        if (summary.limited)
        {
            (*log) << "NOTE: MIPT_GRAPH_REPLAY_MAX_REALIZATIONS stopped the replay after "
                   << summary.realizations
                   << " trajectories. The output is a prefix of the run, like the record file of "
                      "a run killed there, not the whole run.\n";
        }
    }
    return summary;
}

// ---------------------------------------------------------------------------
// paths
// ---------------------------------------------------------------------------

struct PathsOptions
{
    std::string records_path;
    std::string selection_path;
    std::string output_path; // empty: default_paths_output(selection_path)
    int threads = 0;
    std::ostream *log = &std::cerr;
};

struct PathsSummary
{
    std::string output_path;
    std::size_t rows = 0;
    std::size_t realizations = 0;
    std::uint64_t validated_records = 0;
    GateAgreement agreement;
    int threads = 0;
    double seconds = 0.0;
};

// The per-path column block, `sm_` then `wd_`. cancelled_gates sits after
// bridges_trace_only, where the record layout puts it.
inline const std::array<const char *, 18> &path_column_names()
{
    static const std::array<const char *, 18> names{
        "class",           "turns",          "length",           "unitary",
        "temporal_forward", "temporal_backward", "min_layer",     "max_layer",
        "bridges",         "bridges_measured", "bridges_endpoint", "bridges_trace_only",
        "cancelled_gates", "connector_status", "trace_only_bridge", "gate_layers",
        "bridge_list",     "nodes"};
    return names;
}

inline std::string paths_csv_header()
{
    std::string header =
        "realization_id,pair_i,pair_j,group,separation,connected,component_size,shortest_path,"
        "fmn,min_turns,common_source,certified,connected_common_source,connected_certified,"
        "doubled_network,connected_doubled";
    for (const char *prefix : {"sm", "wd"})
    {
        for (const char *name : path_column_names())
        {
            header += ',';
            header += prefix;
            header += '_';
            header += name;
        }
    }
    return header;
}

namespace detail
{
struct SelectionRow
{
    std::uint32_t realization = 0;
    std::uint32_t pair = 0;
    std::uint32_t group = 0;
};

struct Selection
{
    std::vector<SelectionRow> rows;
    std::vector<std::string> groups;
};

// Columns by name; `realization_id,pair_i,pair_j,group` are required and the
// rest are ignored. Every row has to name a pair and a kept realization of the
// run, or the selection was made from another file.
inline Selection read_selection(const std::string &path, const ReplayPlan &plan)
{
    std::ifstream in(path);
    if (!in)
    {
        throw UsageError("Could not open selection CSV " + path + ".");
    }
    std::string line;
    if (!std::getline(in, line))
    {
        throw UsageError(path + " is empty; expected a header naming realization_id,pair_i,"
                                "pair_j,group.");
    }
    const std::vector<std::string> header = split_csv_line(line);
    auto column = [&](const char *name) {
        const auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end())
        {
            throw UsageError(path + " has no `" + std::string(name) +
                             "` column; realization_id,pair_i,pair_j,group are required.");
        }
        return static_cast<std::size_t>(it - header.begin());
    };
    const std::size_t c_realization = column("realization_id");
    const std::size_t c_i = column("pair_i");
    const std::size_t c_j = column("pair_j");
    const std::size_t c_group = column("group");

    Selection out;
    std::map<std::string, std::uint32_t> group_ids;
    std::size_t line_number = 1;
    while (std::getline(in, line))
    {
        ++line_number;
        if (line.empty() || line == "\r")
        {
            continue;
        }
        const std::vector<std::string> cells = split_csv_line(line);
        const std::string where = path + " line " + std::to_string(line_number);
        if (cells.size() != header.size())
        {
            throw UsageError(where + " has " + std::to_string(cells.size()) +
                             " cells; the header has " + std::to_string(header.size()) + ".");
        }
        const long long realization = parse_cell_integer(cells[c_realization], where + ": realization_id");
        const long long site_i = parse_cell_integer(cells[c_i], where + ": pair_i");
        const long long site_j = parse_cell_integer(cells[c_j], where + ": pair_j");
        if (realization < 0 || realization >= plan.config.realizations)
        {
            throw UsageError(where + ": realization " + std::to_string(realization) +
                             " is not a realization of this " +
                             std::to_string(plan.config.realizations) + "-trajectory run.");
        }
        if (!records::stride_selects(plan.config.record_stride,
                                     static_cast<std::uint64_t>(realization)))
        {
            throw UsageError(where + ": realization " + std::to_string(realization) +
                             " was not kept by record_stride=" +
                             std::to_string(plan.config.record_stride) +
                             ", so there are no records to validate its graph against.");
        }
        const int pair = (site_i < 0 || site_j < 0 || site_i >= plan.config.n || site_j >= plan.config.n)
                             ? -1
                             : plan.pair_of(static_cast<int>(site_i), static_cast<int>(site_j));
        if (pair < 0)
        {
            throw UsageError(where + ": (" + std::to_string(site_i) + "," + std::to_string(site_j) +
                             ") is not a pair of this N=" + std::to_string(plan.config.n) + " run.");
        }
        const auto inserted = group_ids.emplace(cells[c_group], static_cast<std::uint32_t>(out.groups.size()));
        if (inserted.second)
        {
            out.groups.push_back(cells[c_group]);
        }
        out.rows.push_back({static_cast<std::uint32_t>(realization), static_cast<std::uint32_t>(pair),
                            inserted.first->second});
    }
    return out;
}

struct PathsJob
{
    std::uint32_t realization = 0;
    std::vector<records::StoredRecord> stored;
    std::vector<std::size_t> rows; // indices into Selection::rows
};

struct PathsResult
{
    std::string text;
    std::size_t rows = 0;
    GateAgreement agreement;
};

inline bool same_shape(const PathShape &a, const PathShape &b)
{
    if (a.available != b.available || a.length != b.length || a.unitary != b.unitary ||
        a.temporal_forward != b.temporal_forward || a.temporal_backward != b.temporal_backward ||
        a.turns != b.turns || a.valleys != b.valleys || a.peaks != b.peaks ||
        a.min_layer != b.min_layer || a.max_layer != b.max_layer || a.gate_layers != b.gate_layers ||
        a.bridges_measured != b.bridges_measured || a.bridges_endpoint != b.bridges_endpoint ||
        a.bridges_trace_only != b.bridges_trace_only || a.path_class != b.path_class ||
        a.connector_status != b.connector_status || a.bridges.size() != b.bridges.size())
    {
        return false;
    }
    for (std::size_t k = 0; k < a.bridges.size(); ++k)
    {
        const PathBridge &x = a.bridges[k];
        const PathBridge &y = b.bridges[k];
        if (x.edge != y.edge || x.layer != y.layer || x.mode_in != y.mode_in ||
            x.mode_out != y.mode_out || x.measured_layer_in != y.measured_layer_in ||
            x.measured_layer_out != y.measured_layer_out || x.status != y.status)
        {
            return false;
        }
    }
    return true;
}

inline void append_int(std::string &out, long long value)
{
    out += ',';
    out += std::to_string(value);
}

inline void append_nan(std::string &out)
{
    out += ",nan";
}

// One canonical path's column block: every field `nan` unless C holds and the
// shape could be read -- the record values' rule -- with the gate layers, the
// bridges and the nodes spelled out in path order. An available path with no
// bridge has an empty bridge list; its `bridges` count of 0 says so.
inline void append_path_columns(std::string &out, const ConnectivityIndex &index,
                                const PathShape &shape, const GraphPath &trace, bool connected)
{
    if (!(connected && shape.available))
    {
        for (std::size_t k = 0; k < path_column_names().size(); ++k)
        {
            append_nan(out);
        }
        return;
    }
    out += ',';
    out += path_class_name(shape.path_class);
    append_int(out, shape.turns);
    append_int(out, shape.length);
    append_int(out, shape.unitary);
    append_int(out, shape.temporal_forward);
    append_int(out, shape.temporal_backward);
    append_int(out, shape.min_layer);
    append_int(out, shape.max_layer);
    append_int(out, static_cast<long long>(shape.bridges.size()));
    append_int(out, shape.bridges_measured);
    append_int(out, shape.bridges_endpoint);
    append_int(out, shape.bridges_trace_only);
    if (shape.cancelled_gates >= 0)
    {
        append_int(out, shape.cancelled_gates);
    }
    else
    {
        append_nan(out);
    }
    out += ',';
    out += connector_status_name(shape.connector_status);
    append_int(out, shape.trace_only_bridge() ? 1 : 0);
    out += ',';
    for (std::size_t k = 0; k < shape.gate_layers.size(); ++k)
    {
        out += k == 0 ? "" : ";";
        out += std::to_string(shape.gate_layers[k]);
    }
    out += ',';
    for (std::size_t k = 0; k < shape.bridges.size(); ++k)
    {
        const PathBridge &bridge = shape.bridges[k];
        out += k == 0 ? "" : ";";
        out += std::to_string(bridge.layer) + ':' + std::to_string(bridge.mode_in) + '>' +
               std::to_string(bridge.mode_out) + ':' + bridge_status_name(bridge.status) + ':' +
               std::to_string(bridge.measured_layer_in) + ':' +
               std::to_string(bridge.measured_layer_out);
    }
    out += ',';
    for (std::size_t k = 0; k < trace.nodes.size(); ++k)
    {
        out += k == 0 ? "" : ";";
        out += std::to_string(index.node_layer(trace.nodes[k])) + '.' +
               std::to_string(index.node_mode(trace.nodes[k]));
    }
}

inline PathsResult paths_realization(const ReplayPlan &plan, const Selection &selection,
                                     const PathsJob &job, GraphWorkspace &workspace)
{
    regenerate_graph(plan, job.realization, workspace);
    const ConnectivityIndex &index = workspace.index;

    // Every stored record of the trajectory, not just the selected ones: a
    // selected pair cannot pass on a trajectory that diverged elsewhere.
    PathsResult result;
    std::vector<int> record_of_pair(plan.records_per_trajectory(), -1);
    std::size_t mismatched = 0;
    std::string first;
    for (std::size_t r = 0; r < job.stored.size(); ++r)
    {
        const records::StoredRecord &record = job.stored[r];
        PairReplay pair;
        const std::string why =
            check_record(plan, index, record.realization, record.geometry, record.embedding,
                         record.flags, record.values.data(), pair, result.agreement);
        if (record_of_pair[static_cast<std::size_t>(pair.pair)] >= 0)
        {
            throw ValidationError(plan.input_path + ": realization " +
                                  std::to_string(job.realization) + " holds pair (" +
                                  std::to_string(pair.i) + "," + std::to_string(pair.j) +
                                  ") twice, so it is not one whole trajectory.");
        }
        record_of_pair[static_cast<std::size_t>(pair.pair)] = static_cast<int>(r);
        if (!why.empty() && mismatched++ == 0)
        {
            first = why;
        }
    }
    if (mismatched > 0)
    {
        raise_mismatch(plan, job.realization, mismatched, first);
    }

    for (std::size_t row_index : job.rows)
    {
        const SelectionRow &row = selection.rows[row_index];
        const auto &embedding = plan.embeddings[row.pair];
        const int i = embedding[2];
        const int j = embedding[3];
        const PairConnectivity conn = index.query(i, j);
        const GatePathMetrics gate = index.gate_path_metrics(i, j);
        const PairTurns turns = index.pair_turns(i, j);
        const GraphPath sm_trace = index.shortest_maximin_trace(i, j);
        const GraphPath wd_trace = index.widest_gate_trace(i, j);
        // The shapes PairTurns reports are read off exactly these traces;
        // cancelled_gates is pair_turns' own addition and is not in path_shape.
        if (!same_shape(index.path_shape(sm_trace, i, j), turns.shortest_maximin) ||
            !same_shape(index.path_shape(wd_trace, i, j), turns.widest))
        {
            throw std::logic_error("Internal: a PairTurns shape of realization " +
                                   std::to_string(job.realization) + ", pair (" +
                                   std::to_string(i) + "," + std::to_string(j) +
                                   ") is not path_shape() of its canonical trace.");
        }
        const records::StoredRecord &stored =
            job.stored[static_cast<std::size_t>(record_of_pair[row.pair])];
        const bool connected = conn.evaluated && conn.connected;

        std::string &out = result.text;
        out += std::to_string(row.realization);
        append_int(out, i);
        append_int(out, j);
        out += ',';
        out += csv_cell(selection.groups[row.group]);
        append_int(out, plan.bins[static_cast<std::size_t>(embedding[0])].separation);
        append_int(out, connected ? 1 : 0);
        append_int(out, conn.component_size);
        if (connected && gate.available && gate.shortest_path_length >= 0)
        {
            append_int(out, gate.shortest_path_length);
        }
        else
        {
            append_nan(out);
        }
        out += ',';
        if (plan.fmn_slot >= 0)
        {
            append_double(out, stored.values[static_cast<std::size_t>(plan.fmn_slot)]);
        }
        else
        {
            out += "nan";
        }
        if (connected && turns.min_turns >= 0)
        {
            append_int(out, turns.min_turns);
        }
        else
        {
            append_nan(out);
        }
        for (const bool value : {turns.common_source, turns.certified, turns.connected_common_source,
                                 turns.connected_certified, turns.doubled_network,
                                 turns.connected_doubled})
        {
            if (turns.evaluated)
            {
                append_int(out, value ? 1 : 0);
            }
            else
            {
                append_nan(out);
            }
        }
        append_path_columns(out, index, turns.shortest_maximin, sm_trace, connected);
        append_path_columns(out, index, turns.widest, wd_trace, connected);
        out += '\n';
        ++result.rows;
    }
    return result;
}
} // namespace detail

inline PathsSummary paths(const PathsOptions &options)
{
    const auto started = std::chrono::steady_clock::now();
    std::error_code exists_error;
    if (!std::filesystem::exists(options.records_path, exists_error))
    {
        throw UsageError("Record file " + options.records_path + " does not exist.");
    }
    records::RecordReader reader(options.records_path);
    const ReplayPlan plan = make_plan(reader);
    const detail::Selection selection = detail::read_selection(options.selection_path, plan);

    PathsSummary summary;
    summary.output_path = options.output_path.empty() ? default_paths_output(options.selection_path)
                                                      : options.output_path;
    summary.threads = options.threads > 0 ? options.threads : configured_threads();
    detail::refuse_overwriting_input(options.records_path, summary.output_path);
    detail::refuse_overwriting_input(options.selection_path, summary.output_path);

    // Realization order, and selection order within a realization.
    std::vector<std::size_t> order(selection.rows.size());
    for (std::size_t k = 0; k < order.size(); ++k)
    {
        order[k] = k;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return selection.rows[a].realization < selection.rows[b].realization;
    });
    std::size_t groups = 0;
    for (std::size_t k = 0; k < order.size(); ++k)
    {
        groups += (k == 0 || selection.rows[order[k]].realization !=
                                 selection.rows[order[k - 1]].realization)
                      ? 1u
                      : 0u;
    }

    std::ostream *log = options.log;
    if (log != nullptr)
    {
        (*log) << "dist_graph_replay paths\n"
               << "  records=" << options.records_path << "\n  selection=" << options.selection_path
               << " (" << selection.rows.size() << " row(s) over " << groups
               << " realization(s))\n"
               << "  header reconstructed exactly; output=" << summary.output_path
               << ", threads=" << summary.threads << '\n';
        log->flush();
    }

    detail::AtomicOutput output(summary.output_path);
    output.write(paths_csv_header() + '\n');
    detail::Progress progress(log, "paths", groups);
    std::size_t next = 0;
    auto produce = [&](detail::PathsJob &job) -> bool {
        if (next >= order.size())
        {
            return false;
        }
        job.realization = selection.rows[order[next]].realization;
        while (next < order.size() && selection.rows[order[next]].realization == job.realization)
        {
            job.rows.push_back(order[next++]);
        }
        job.stored = reader.read_realization(job.realization);
        if (job.stored.empty())
        {
            throw UsageError(options.selection_path + " selects realization " +
                             std::to_string(job.realization) + ", which " +
                             options.records_path + " does not hold (the run never reached it).");
        }
        if (job.stored.size() != plan.records_per_trajectory())
        {
            throw ValidationError(options.records_path + " holds " +
                                  std::to_string(job.stored.size()) +
                                  " record(s) for realization " + std::to_string(job.realization) +
                                  "; a whole trajectory holds " +
                                  std::to_string(plan.records_per_trajectory()) + ".");
        }
        return true;
    };
    auto work = [&](detail::PathsJob &&job, GraphWorkspace &workspace) {
        return detail::paths_realization(plan, selection, job, workspace);
    };
    auto consume = [&](detail::PathsResult &&result) {
        output.write(result.text);
        summary.rows += result.rows;
        ++summary.realizations;
        summary.validated_records += plan.records_per_trajectory();
        summary.agreement.merge(result.agreement);
        progress.update(summary.realizations);
    };
    detail::run_ordered<detail::PathsJob, detail::PathsResult, GraphWorkspace>(
        summary.threads, 4u * static_cast<std::size_t>(summary.threads), produce, work, consume);
    output.commit();
    summary.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return summary;
}

// ---------------------------------------------------------------------------
// The command line
// ---------------------------------------------------------------------------

inline void print_usage(std::ostream &out, const std::string &program)
{
    out << "Usage:\n"
        << "  " << program << " augment <in_records.bin> [out_records.bin]\n"
        << "  " << program << " paths <records.bin> <selection.csv> [out.csv]\n\n"
        << "Regenerates the spacetime graph of every trajectory in a dist_scaling.exe k=2\n"
        << "record file from the master seed in its header -- on the host: no GPU, no CUDA-Q,\n"
        << "no state vector -- and checks each regenerated graph against the stored records\n"
        << "before using it.\n\n"
        << "  augment  writes the records again with the " << records::TURN_RECORD_VALUES
        << " temporal-turn / doubled-network\n"
        << "           observables appended and their flag bits set: byte for byte what\n"
        << "           dist_scaling.exe writes with MIPT_DIST_TURN_DIAGNOSTICS=1, given the same\n"
        << "           physics. Default output: the input with `_records.bin` replaced by\n"
        << "           `_turns_records.bin`. A file that already carries the turn values is\n"
        << "           validated on them too and reproduced unchanged.\n"
        << "  paths    one CSV row per selected pair: its connectivity, the turn and\n"
        << "           doubled-network verdicts, and both canonical paths (shortest-maximin,\n"
        << "           widest entangling power) spelled out gate by gate and node by node.\n"
        << "           selection.csv needs columns realization_id,pair_i,pair_j,group, found\n"
        << "           by name; others are ignored. Rows come out in realization order\n"
        << "           (selection order within one), oriented as the record file's embedding\n"
        << "           table orients the pair. Default output: <selection stem>_paths.csv.\n\n"
        << "Requirements (refused with exit 2): a format-v2 k=2 record file of circ_type 2\n"
        << "(RPPU) or 4 (qRPPU), written with MIPT_DIST_CONNECTIVITY=1 and\n"
        << "MIPT_DIST_GATE_DIAGNOSTICS=1.\n\n"
        << "Validation (exit 3 on any failure; nothing is left at the output path):\n"
        << "  * the header must be exactly what dist_scaling.exe renders for the run it\n"
        << "    describes, which proves the run was reconstructed exactly;\n"
        << "  * every record's flag bits 0-4 and its seven gate-path values must equal those\n"
        << "    of the regenerated graph: the integer-valued ones and every NaN exactly, the\n"
        << "    four continuous capacities to an absolute 1e-12 (production's clang build\n"
        << "    rounds them differently from this g++ one, by a few ulps) -- and its turn\n"
        << "    values and turn bits exactly too, if it has them.\n"
        << "An unfinished final trajectory (a kill inside a write) is dropped with a note.\n\n"
        << "Environment:\n"
        << "  MIPT_GRAPH_REPLAY_THREADS           worker threads (default min(8, cores-1))\n"
        << "  MIPT_GRAPH_REPLAY_MAX_REALIZATIONS  augment only the first N whole trajectories\n"
        << "                                      (timing runs; the output is then a prefix)\n\n"
        << "Exit status: 0 success; 1 I/O or other error; 2 usage error or refused input;\n"
        << "3 validation failure.\n";
}

inline std::string describe_agreement(const GateAgreement &agreement)
{
    std::string out = "continuous gate values: " + std::to_string(agreement.compared) +
                      " compared, " + std::to_string(agreement.inexact) +
                      " not bit-identical, max |stored - regenerated| = ";
    append_double(out, agreement.max_difference);
    out += " (tolerance ";
    append_double(out, GATE_VALUE_TOLERANCE);
    out += ")";
    return out;
}

inline int cli_main(const std::vector<std::string> &args, std::ostream &out, std::ostream &err)
{
    const std::string program = args.empty() ? "dist_graph_replay.exe" : args[0];
    try
    {
        for (std::size_t k = 1; k < args.size(); ++k)
        {
            if (args[k] == "--help" || args[k] == "-h")
            {
                print_usage(out, program);
                return STATUS_OK;
            }
        }
        if (args.size() < 2)
        {
            print_usage(err, program);
            return STATUS_USAGE;
        }
        const std::string &command = args[1];
        if (command == "help")
        {
            print_usage(out, program);
            return STATUS_OK;
        }
        if (command == "augment")
        {
            if (args.size() < 3 || args.size() > 4)
            {
                throw UsageError("augment takes <in_records.bin> [out_records.bin].");
            }
            AugmentOptions options;
            options.input_path = args[2];
            options.output_path = args.size() == 4 ? args[3] : std::string();
            options.threads = configured_threads();
            options.max_realizations = configured_max_realizations();
            options.log = &err;
            const AugmentSummary summary = augment(options);
            out << "augment: validated " << summary.records << " record(s) over "
                << summary.realizations << " realization(s) against their regenerated graphs; "
                << "mismatches=" << summary.mismatches << ".\n"
                << "  elapsed " << util::format_duration(summary.seconds, util::DurationStyle::Compact)
                << " (" << std::fixed << std::setprecision(3) << summary.seconds << " s), "
                << std::setprecision(2)
                << (summary.seconds > 0.0 ? static_cast<double>(summary.realizations) / summary.seconds
                                          : 0.0)
                << " realizations/s with " << summary.threads << " worker thread(s)\n"
                << std::defaultfloat << "  " << describe_agreement(summary.agreement) << '\n'
                << "  output=" << summary.output_path << '\n';
            return STATUS_OK;
        }
        if (command == "paths")
        {
            if (args.size() < 4 || args.size() > 5)
            {
                throw UsageError("paths takes <records.bin> <selection.csv> [out.csv].");
            }
            PathsOptions options;
            options.records_path = args[2];
            options.selection_path = args[3];
            options.output_path = args.size() == 5 ? args[4] : std::string();
            options.threads = configured_threads();
            options.log = &err;
            const PathsSummary summary = paths(options);
            out << "paths: wrote " << summary.rows << " row(s) over " << summary.realizations
                << " realization(s), every record of each validated against its regenerated "
                   "graph ("
                << summary.validated_records << " record(s)); mismatches=0.\n"
                << "  elapsed " << util::format_duration(summary.seconds, util::DurationStyle::Compact)
                << " with " << summary.threads << " worker thread(s)\n"
                << "  " << describe_agreement(summary.agreement) << '\n'
                << "  output=" << summary.output_path << '\n';
            return STATUS_OK;
        }
        throw UsageError("unknown command `" + command + "`; expected augment or paths.");
    }
    catch (const UsageError &error)
    {
        err << "dist_graph_replay: " << error.what() << "\nRun with --help for usage.\n";
        return STATUS_USAGE;
    }
    catch (const ValidationError &error)
    {
        err << "dist_graph_replay: VALIDATION FAILED: " << error.what() << '\n';
        return STATUS_VALIDATION;
    }
    catch (const std::exception &error)
    {
        err << "dist_graph_replay: ERROR: " << error.what() << '\n';
        return STATUS_ERROR;
    }
}

inline int cli_main(int argc, char **argv)
{
    std::vector<std::string> args(argv, argv + argc);
    return cli_main(args, std::cout, std::cerr);
}

} // namespace mipt::dist::graph_replay
