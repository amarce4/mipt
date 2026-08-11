#pragma once

// dist_scaling.exe's run configuration, aggregation bins, and CSV projection.
//
// Split out of dist_scaling.hpp so it carries no CUDA-Q, MOSEK, or CUDA
// dependency: the bins and the file they render to are pure data, and keeping
// them host-buildable is what lets `make test-dist` pin the property the resume
// path depends on -- that rendering a set of bins and reading them back is
// lossless. The same split is why small_rdm.hpp exists.
//
// The CSV is a *projection* of the bins, one row per geometry carrying means,
// standard errors and sample counts. That projection is invertible, which is
// what makes the output file its own checkpoint; see dist_scaling_resume.hpp.

#include "mipt/dist_metrics.hpp"
#include "mipt/env.hpp"
#include "mipt/probed/geometry.hpp"
#include "mipt/types.hpp"
#include "mipt/util/geometry.hpp"
#include "mipt/util/stats.hpp"
#include "mipt/util/text.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace mipt::dist
{
using util::RunningStats;
using util::compact_decimal;
using util::csv_quote;

// ---------------------------------------------------------------------------
// Number formatting
//
// 17 significant digits: enough that every double round-trips exactly, which
// the resume path relies on.
// ---------------------------------------------------------------------------

inline void append_double(std::string &out, double value)
{
    if (!std::isfinite(value))
    {
        out += std::isnan(value) ? "nan" : (value > 0.0 ? "inf" : "-inf");
        return;
    }
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                      std::chars_format::general, 17);
    if (result.ec == std::errc{})
    {
        out.append(buffer, result.ptr);
        return;
    }
    out += std::to_string(value);
}

inline void append_uint(std::string &out, std::uint64_t value)
{
    out += std::to_string(value);
}


// ---------------------------------------------------------------------------
// Run configuration
// ---------------------------------------------------------------------------

struct RunConfig
{
    int k = 2;
    int n = 10;
    int periods = 10;
    double p = 0.17;
    int realizations = 10;
    CircuitType type = CircuitType::MMS;
    std::string output_path;
    // k=3 only: the minimum CFT chord balance B = l_min/l_max a triangle must
    // reach to be simulated at all. Ignored for k=2.
    double triangle_balance_cutoff = 0.5;

    // Parity-preserving circuits report both trace conventions. This is
    // deliberately `preserves_computational_parity` and not
    // `uses_fermionic_trace`: qRPPU (circ_type 4) simulates the same
    // parity-preserving gate set in the qubit encoding, and the point of
    // running it is to compare the two conventions on it.
    bool fermionic_outputs() const
    {
        return preserves_computational_parity(type);
    }
};

inline std::string default_output_path(const RunConfig &config)
{
    const std::string tag(circuit_type_tag(config.type));
    std::ostringstream filename;
    filename << (config.k == 3 ? "dist_scaling3_" : "dist_scaling_") << tag
             << "_n_" << config.n
             << "_periods_" << config.periods
             << "_p_" << compact_decimal(config.p)
             << "_real_" << config.realizations;
    if (config.k == 3)
    {
        filename << "_b_" << compact_decimal(config.triangle_balance_cutoff);
    }
    filename << ".csv";
    return std::string("csv/dist_scaling/") + tag + "/" + filename.str();
}

inline bool file_exists(const std::string &path)
{
    std::error_code error;
    return !path.empty() && std::filesystem::exists(path, error) && !error;
}

inline void ensure_output_parent_directory(const std::string &output_path)
{
    const std::filesystem::path parent = std::filesystem::path(output_path).parent_path();
    if (parent.empty())
    {
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error)
    {
        throw std::runtime_error("Could not create output directory " +
                                 parent.string() + ": " + error.message());
    }
}

inline void validate_args(const RunConfig &config)
{
    if (config.k != 2 && config.k != 3)
    {
        throw std::invalid_argument("k must be 2 (pairs) or 3 (triangles).");
    }
    if (config.n < config.k || config.n >= 63)
    {
        throw std::invalid_argument("N must satisfy k <= N < 63.");
    }
    if (config.periods <= 0)
    {
        throw std::invalid_argument("periods must be positive.");
    }
    if (!(config.p >= 0.0 && config.p <= 1.0) || !std::isfinite(config.p))
    {
        throw std::invalid_argument("p must lie in [0, 1].");
    }
    if (config.realizations <= 0)
    {
        throw std::invalid_argument("realizations must be positive.");
    }
    if (config.k == 3 &&
        (!std::isfinite(config.triangle_balance_cutoff) || config.triangle_balance_cutoff < 0.0 ||
         config.triangle_balance_cutoff > 1.0))
    {
        throw std::invalid_argument("B_min must lie in [0, 1].");
    }
    validate_circuit_site_count(config.type, config.n, "N");
}

// ---------------------------------------------------------------------------
// Aggregation bins
//
// One bin per geometry, holding streaming means for every reported measure.
// The CSV is a projection of these, so a run that is interrupted still leaves
// a complete, correctly normalized file behind.
// ---------------------------------------------------------------------------

struct PairBin
{
    int separation = 0;
    double chord = 0.0;
    // How many lattice pairs realize this separation. On a ring that is N for
    // every separation except the antipodal one at even N, which is N/2.
    std::uint64_t embedding_count = 0;

    RunningStats mi;
    RunningStats mn;
    RunningStats fmi;
    RunningStats fmn;
    std::uint64_t mn_positive = 0;
    std::uint64_t fmn_positive = 0;
};

// A GMN-family measure: its running mean plus the bookkeeping that says how
// the mean was obtained. `requested` counts records the schedule selected,
// `zero_prefiltered` the subset a vanishing bipartite negativity settled for
// free, and `solver_failures` the subset MOSEK could not solve -- those are
// excluded from the mean rather than counted as zero.
struct SdpStats
{
    RunningStats value;
    std::uint64_t requested = 0;
    std::uint64_t zero_prefiltered = 0;
    std::uint64_t solver_failures = 0;
};

struct TripleBin
{
    std::array<int, 3> separations{};
    double balance = 0.0;
    double chord_geometric_mean = 0.0;
    std::size_t embedding_count = 0;

    RunningStats tmi;
    RunningStats average_mi;
    RunningStats min_bipartite_negativity;
    RunningStats joint_purity;
    RunningStats mean_single_purity;

    RunningStats ftmi;
    RunningStats faverage_mi;
    RunningStats min_fermionic_bipartite_negativity;
    RunningStats fjoint_purity;
    RunningStats fmean_single_purity;

    SdpStats gmn;
    SdpStats fgmn;

    // Records folded into this bin so far. The SDP schedule is a function of
    // this counter alone, which is what keeps the selection independent of the
    // values being selected.
    std::uint64_t records = 0;
};

// ---------------------------------------------------------------------------
// Deferred GMN/fGMN evaluation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CSV writers
//
// Both formats start with the self-describing metadata block every other
// application writes, so `data_analysis.loading.resolve_metadata` can read a
// file's run parameters without parsing its name.
// ---------------------------------------------------------------------------

inline constexpr const char *METADATA_COLUMNS =
    "N,circ_type,circuit_name,realizations,p,periods,k,entropy_units";

inline void append_metadata_fields(std::string &out, const RunConfig &config)
{
    out += std::to_string(config.n);
    out += ',';
    out += std::to_string(static_cast<int>(config.type));
    out += ',';
    out += csv_quote(circuit_type_name(config.type));
    out += ',';
    out += std::to_string(config.realizations);
    out += ',';
    append_double(out, config.p);
    out += ',';
    out += std::to_string(config.periods);
    out += ',';
    out += std::to_string(config.k);
    // Every entropy dist_scaling.exe reports -- MI, fMI, TMI, fTMI -- is a
    // log2 quantity. The probe and free-energy executables use nats, so the
    // unit is stated here rather than left to be remembered.
    out += ",bits";
}

inline void append_stats(std::string &out, const RunningStats &stats)
{
    out += ',';
    if (stats.count == 0)
    {
        out += "nan,nan,0";
        return;
    }
    append_double(out, stats.mean);
    out += ',';
    append_double(out, stats.stderr());
    out += ',';
    append_uint(out, stats.count);
}

inline void append_sdp_stats(std::string &out, const SdpStats &stats)
{
    append_stats(out, stats.value);
    out += ',';
    append_uint(out, stats.requested);
    out += ',';
    append_uint(out, stats.zero_prefiltered);
    out += ',';
    append_uint(out, stats.solver_failures);
}

inline std::string pair_csv_header(const RunConfig &config)
{
    std::string header(METADATA_COLUMNS);
    header +=
        ",separation,chord_length,d,embedding_count,"
        "mi_mean,mi_stderr,mi_samples,"
        "mn_mean,mn_stderr,mn_samples,mn_positive_count,mn_positive_fraction";
    if (config.fermionic_outputs())
    {
        header +=
            ",fmi_mean,fmi_stderr,fmi_samples,"
            "fmn_mean,fmn_stderr,fmn_samples,fmn_positive_count,fmn_positive_fraction";
    }
    header += '\n';
    return header;
}

inline std::string triple_csv_header(const RunConfig &config)
{
    std::string header(METADATA_COLUMNS);
    header +=
        ",geometry_id,distance_1,distance_2,distance_3,chord_1,chord_2,chord_3,d,"
        "triangle_balance,triangle_balance_cutoff,embedding_count,"
        "tmi_mean,tmi_stderr,tmi_samples,"
        "average_mi_mean,average_mi_stderr,average_mi_samples,"
        "min_bipneg_mean,min_bipneg_stderr,min_bipneg_samples,"
        "gmn_mean,gmn_stderr,gmn_samples,gmn_requested_count,"
        "gmn_zero_prefilter_count,gmn_solver_failure_count,"
        "joint_purity_mean,joint_purity_stderr,joint_purity_samples,"
        "mean_single_purity_mean,mean_single_purity_stderr,mean_single_purity_samples";
    if (config.fermionic_outputs())
    {
        header +=
            ",ftmi_mean,ftmi_stderr,ftmi_samples,"
            "faverage_mi_mean,faverage_mi_stderr,faverage_mi_samples,"
            "min_fbipneg_mean,min_fbipneg_stderr,min_fbipneg_samples,"
            "fgmn_mean,fgmn_stderr,fgmn_samples,fgmn_requested_count,"
            "fgmn_zero_prefilter_count,fgmn_solver_failure_count,"
            "fjoint_purity_mean,fjoint_purity_stderr,fjoint_purity_samples,"
            "fmean_single_purity_mean,fmean_single_purity_stderr,fmean_single_purity_samples";
    }
    header += '\n';
    return header;
}

inline double positive_fraction(std::uint64_t positive, std::uint64_t total)
{
    return total > 0 ? static_cast<double>(positive) / static_cast<double>(total)
                     : std::numeric_limits<double>::quiet_NaN();
}

inline std::string render_pair_csv(const RunConfig &config, const std::vector<PairBin> &bins)
{
    std::string out = pair_csv_header(config);
    out.reserve(out.size() + bins.size() * 192u);
    std::string line;
    for (const PairBin &bin : bins)
    {
        line.clear();
        append_metadata_fields(line, config);
        line += ',';
        line += std::to_string(bin.separation);
        line += ',';
        append_double(line, bin.chord);
        line += ',';
        append_double(line, bin.chord);
        line += ',';
        append_uint(line, bin.embedding_count);
        append_stats(line, bin.mi);
        append_stats(line, bin.mn);
        line += ',';
        append_uint(line, bin.mn_positive);
        line += ',';
        append_double(line, positive_fraction(bin.mn_positive, bin.mn.count));
        if (config.fermionic_outputs())
        {
            append_stats(line, bin.fmi);
            append_stats(line, bin.fmn);
            line += ',';
            append_uint(line, bin.fmn_positive);
            line += ',';
            append_double(line, positive_fraction(bin.fmn_positive, bin.fmn.count));
        }
        line += '\n';
        out += line;
    }
    return out;
}

inline std::string render_triple_csv(const RunConfig &config, const std::vector<TripleBin> &bins)
{
    std::string out = triple_csv_header(config);
    out.reserve(out.size() + bins.size() * 384u);
    std::string line;
    for (std::size_t index = 0; index < bins.size(); ++index)
    {
        const TripleBin &bin = bins[index];
        line.clear();
        append_metadata_fields(line, config);
        line += ',';
        line += std::to_string(index);
        for (int separation : bin.separations)
        {
            line += ',';
            line += std::to_string(separation);
        }
        for (int separation : bin.separations)
        {
            line += ',';
            append_double(line, chord_length(config.n, separation));
        }
        line += ',';
        append_double(line, bin.chord_geometric_mean);
        line += ',';
        append_double(line, bin.balance);
        line += ',';
        append_double(line, config.triangle_balance_cutoff);
        line += ',';
        append_uint(line, static_cast<std::uint64_t>(bin.embedding_count));
        append_stats(line, bin.tmi);
        append_stats(line, bin.average_mi);
        append_stats(line, bin.min_bipartite_negativity);
        append_sdp_stats(line, bin.gmn);
        append_stats(line, bin.joint_purity);
        append_stats(line, bin.mean_single_purity);
        if (config.fermionic_outputs())
        {
            append_stats(line, bin.ftmi);
            append_stats(line, bin.faverage_mi);
            append_stats(line, bin.min_fermionic_bipartite_negativity);
            append_sdp_stats(line, bin.fgmn);
            append_stats(line, bin.fjoint_purity);
            append_stats(line, bin.fmean_single_purity);
        }
        line += '\n';
        out += line;
    }
    return out;
}

// Rewrite the whole CSV from the bins.
//
// The aggregated files are at most a few hundred rows, so republishing them
// costs nothing and gives a killed run the same "partial results survive"
// behaviour the old row-streaming format had.
inline void publish_csv(const std::string &path, const std::string &contents)
{
    std::ofstream csv(path, std::ios::binary | std::ios::trunc);
    if (!csv)
    {
        throw std::runtime_error("Could not create output CSV: " + path);
    }
    csv.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    csv.flush();
    if (!csv)
    {
        throw std::runtime_error("Failed while writing output CSV.");
    }
}

inline std::vector<PairBin> make_pair_bins(int n)
{
    const int max_separation = n / 2;
    std::vector<PairBin> bins;
    bins.reserve(static_cast<std::size_t>(max_separation));
    for (int separation = 1; separation <= max_separation; ++separation)
    {
        PairBin bin;
        bin.separation = separation;
        bin.chord = chord_length(n, separation);
        // Each site starts one pair at this separation, except that at the
        // antipodal separation of an even ring the forward and backward pairs
        // coincide, so each is counted once.
        bin.embedding_count =
            (n % 2 == 0 && separation == max_separation) ? static_cast<std::uint64_t>(n / 2)
                                                         : static_cast<std::uint64_t>(n);
        bins.push_back(bin);
    }
    return bins;
}

inline long embeddings_per_geometry()
{
    // 0 means "every lattice embedding of every geometry, every trajectory".
    //
    // The default of one mirrors three-probe mode 4, which draws a single
    // random embedding per geometry per trajectory. Raising it buys more
    // samples per simulated circuit at a proportional cost in RDM reductions
    // and, more importantly, in SDP solves -- tune it together with
    // MIPT_DIST_GMN_SAMPLES_PER_GEOMETRY.
    return env::integer("MIPT_DIST_EMBEDDINGS_PER_GEOMETRY", 1, 0, 1000000);
}

inline std::vector<TripleBin> make_triple_bins(int n, const std::vector<probed::ProbeGeometry> &geometries)
{
    std::vector<TripleBin> bins;
    bins.reserve(geometries.size());
    for (const auto &geometry : geometries)
    {
        TripleBin bin;
        bin.separations = geometry.distances;
        bin.balance = geometry.balance;
        bin.chord_geometric_mean = triangle_chord_geometric_mean(n, geometry.distances);
        bin.embedding_count = geometry.embeddings.size();
        bins.push_back(bin);
    }
    return bins;
}

} // namespace mipt::dist
