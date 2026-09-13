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

#include "mipt/dist_connectivity.hpp"
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

// How much per-record detail the binary carries. Rising levels are strictly
// additive, so a `full` file can answer every question a `basic` one can.
//
//   basic   - the observables format v1 wrote, plus the connectivity flags.
//   channel - adds the occupation data and the squared two-point functions,
//             which is the set that reconstructs rho^n, the four occupation
//             probabilities, the occupation MI, the parity weights, and the
//             exact negativity. This is the default: it is the minimum that
//             answers the question the flags were added for.
//   full    - adds Re/Im of G and F, so the channel phases survive too.
enum class RecordDetail : int
{
    Basic = 0,
    Channel = 1,
    Full = 2,
};

inline const char *record_detail_name(RecordDetail detail)
{
    switch (detail)
    {
    case RecordDetail::Basic: return "basic";
    case RecordDetail::Channel: return "channel";
    case RecordDetail::Full: return "full";
    }
    return "channel";
}

inline RecordDetail parse_record_detail(const std::string &text)
{
    if (text == "basic") return RecordDetail::Basic;
    if (text == "channel") return RecordDetail::Channel;
    if (text == "full") return RecordDetail::Full;
    throw std::invalid_argument(
        "MIPT_DIST_RECORD_DETAIL must be basic, channel, or full (got \"" + text + "\").");
}

// Why a pair that carries no entanglement above threshold carries none.
//
// Exhaustive and disjoint, and the *single* definition: PairProtocol's
// aggregate split and the connected-zero triple analysis both call
// classify_unentangled, so a record counted `silent` in the pair CSV is the same
// record the triple analysis labels `silent`. Both comparisons are strict, so a
// value exactly at its tolerance falls into the lower class.
//
//   occupation_correlated  I_occ > occupation_mi_tol: classically correlated
//                          through the occupations, not entangled.
//   coherent_subthreshold  I_occ at or below tolerance, but |G|^2 + |F|^2 above
//                          the channel floor: coherence the threshold discarded.
//   silent                 neither: no detectable two-mode correlation at all.
enum class ZeroClass : int
{
    OccupationCorrelated = 0,
    CoherentSubthreshold = 1,
    Silent = 2,
};
inline constexpr int ZERO_CLASS_COUNT = 3;

inline const char *zero_class_name(ZeroClass value)
{
    switch (value)
    {
    case ZeroClass::OccupationCorrelated: return "occupation_correlated";
    case ZeroClass::CoherentSubthreshold: return "coherent_subthreshold";
    case ZeroClass::Silent: return "silent";
    }
    return "silent";
}

inline ZeroClass classify_unentangled(double i_occ, double channel_weight,
                                      double occupation_mi_tol, double channel_floor)
{
    if (i_occ > occupation_mi_tol)
    {
        return ZeroClass::OccupationCorrelated;
    }
    if (channel_weight > channel_floor)
    {
        return ZeroClass::CoherentSubthreshold;
    }
    return ZeroClass::Silent;
}

// Which graph-connected, unentangled pairs the triple analysis anchors on.
enum class GapSelection : int
{
    ConnectedZero = -1, // every connected-zero pair, all three classes
    OccupationCorrelated = 0,
    CoherentSubthreshold = 1,
    Silent = 2,
};

inline const char *gap_selection_name(GapSelection value)
{
    return value == GapSelection::ConnectedZero
               ? "connected_zero"
               : zero_class_name(static_cast<ZeroClass>(static_cast<int>(value)));
}

inline GapSelection parse_gap_selection(const std::string &text)
{
    if (text == "connected_zero") return GapSelection::ConnectedZero;
    if (text == "occupation_correlated") return GapSelection::OccupationCorrelated;
    if (text == "coherent_subthreshold") return GapSelection::CoherentSubthreshold;
    if (text == "silent") return GapSelection::Silent;
    throw std::invalid_argument(
        "MIPT_DIST_PAIR_GAP_CLASS must be connected_zero, occupation_correlated, "
        "coherent_subthreshold, or silent (got \"" + text + "\").");
}

inline bool gap_selects(GapSelection selection, ZeroClass value)
{
    return selection == GapSelection::ConnectedZero ||
           static_cast<int>(selection) == static_cast<int>(value);
}

// The connected-zero triple analysis: for every graph-connected pair (i, j)
// whose fermionic negativity is at or below pair_zero_tol, every third site k
// is added and the triple {i, j, k} is analysed in full, fGMN included. See
// dist_pair_gap.hpp for what it measures and why.
struct PairGapSettings
{
    bool enabled = false;
    GapSelection selection = GapSelection::ConnectedZero;
    // Empty means <main stem>_connected_zero_thirds.csv beside the pair CSV.
    std::string output_path;
    // Keep every unique triple's raw fermionic 8x8 in a binary companion, so a
    // future multipartite measure does not need the trajectories rerun.
    bool store_rho3 = false;
    // Also analyse one distance-matched *disconnected* zero pair per anchor,
    // drawn from the same trajectory, as a control group.
    bool controls = false;
    // fGMN <= min_s N_s, so a minimum cut at or below this certifies fGMN is
    // at most this without a solve. Defaults to MIPT_DIST_GMN_ZERO_TOL, the
    // calibrated SDP prefilter threshold -- seven decades above MOSEK's floor
    // on exactly-zero states, measured.
    double prefilter_tol = 1.0e-10;
    // fGMN above this counts as genuinely tripartite-entangled. It is compared
    // against the *certified lower bound*, so a solve whose interval straddles
    // it is unresolved rather than negative -- and at the default
    // GMN_MOSEK_TOL=1e-5 an interval is far wider than 1e-10, so a meaningful
    // sweep of this threshold wants a tighter solver tolerance too.
    double positive_tol = 1.0e-10;
    // A product distance ||rho - rho_s (x) rho_sbar||_1 above this counts as
    // correlation across the cut. It is what separates a *product* cut from a
    // classically correlated one, both of which have zero negativity.
    double correlation_tol = 1.0e-10;
    // Decompose each anchor's spacetime component into backbone, articulation
    // points and dangling branches. One linear sweep per anchor, negligible
    // beside a solve, so it is on by default -- MIPT_DIST_PAIR_GAP_ANATOMY=0.
    bool anatomy = true;
    // Channel-resolved path strengths. Two Dijkstra sweeps plus a max flow per
    // anchor, so noticeably dearer than the anatomy but still small beside a
    // solve; MIPT_DIST_PAIR_GAP_CHANNELS=0 turns it off.
    bool channels = true;
    // Four-site analysis for the residue: pairs with no positive third that are
    // *not* numerically unresolved. Off by default -- it is the one addition
    // here that costs a second RDM reduction per selected pair.
    // Assisted (localizable) endpoint negativity under parity-respecting helper
    // measurements. Cheap on a triple -- two 8x8 projections -- so on by
    // default; the four-site pass adds the joint two-helper family.
    bool assisted = true;
    // A conditional endpoint negativity above this counts as revealed.
    double assisted_tol = 1.0e-10;
    bool four_site = false;
    int four_max_helper_pairs = 12;
    bool store_rho4 = false;
    // A one-vs-rest cut negativity above this counts as an entangled cut. It
    // decides the pair summary's "all cuts entangled but fGMN bounded" class --
    // the bound-entanglement candidate -- and defaults to prefilter_tol, which
    // is the threshold that already decides whether a cut certifies fGMN.
    double cut_positive_tol = 1.0e-10;
    // Extra attempts for a failed solve, the last one serialized.
    int retries = 2;
    // GMN_MOSEK_TOL as resolved when the run started, recorded verbatim.
    std::string mosek_tol_text = "1e-5";
};

// The |G|^2 / |F|^2 floor for a given state-vector precision: one decade above
// the square of that precision's amplitude noise, so a record sitting at the
// floor is arithmetic and one above it is not.
inline double default_channel_floor(int precision)
{
    return precision >= 64 ? 1.0e-28 : 1.0e-13;
}

// The precision the state vector was built in. Compile-time, because it is a
// property of the build rather than of the run; `run()` cross-checks it
// against what the backend actually hands back and complains if they differ.
#ifndef MIPT_CUDAQ_PRECISION
#define MIPT_CUDAQ_PRECISION 32
#endif

struct RunConfig
{
    // 2 = pairs, 3 = triangles, 0 = both from the same trajectories. A k=0 run
    // fans out into two RunConfig copies carrying k=2 and k=3, so every file
    // written -- header, metadata block, checkpoint semantics -- is exactly
    // what the corresponding exclusive run writes.
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

    // --- classification thresholds ---------------------------------------
    //
    // The negativity above which a pair counts as entangled. This was a
    // hard-coded 1e-12 buried in PairProtocol::measure, which is exactly the
    // wrong place for it: it is a *classification* threshold, not a
    // mathematical zero, and every conditional probability in the output moves
    // when it moves. It is named, defaulted to the historical value, written
    // into both output files, and meant to be swept.
    double pair_zero_tol = 1.0e-12;
    // Below this, an occupation mutual information counts as zero when
    // splitting the graph-connected but unentangled records into their three
    // explanations.
    double occupation_mi_tol = 1.0e-12;
    // Below this, |G|^2 and |F|^2 count as zero for the same split.
    //
    // **It has to track the state-vector precision, and the default does.**
    // These are squared amplitudes, so the floor is the square of the state
    // vector's own noise: an fp32 trajectory carries ~1e-7 in an amplitude that
    // is algebraically zero, giving |G|^2 ~ 1e-14, while fp64 gives ~1e-30. A
    // single fixed floor therefore cannot work for both -- set it at 1e-30 and
    // every fp32 record looks like it has residual coherence, which collapses
    // the three-way split into one cell and quietly destroys the distinction it
    // exists to draw. Set by `default_channel_floor()` below; override with
    // MIPT_DIST_CHANNEL_FLOOR.
    double channel_floor = 1.0e-13;

    // --- spacetime connectivity ------------------------------------------
    bool connectivity = true;
    // Shortest spanning-path lengths cost one BFS sweep per site per
    // trajectory. Cheap, but it is a second-stage explanatory diagnostic
    // rather than part of the decisive measurement, so it can be switched off.
    bool connectivity_paths = true;

    // --- per-record output ------------------------------------------------
    RecordDetail record_detail = RecordDetail::Channel;
    // Write records only for every `record_stride`-th trajectory.
    //
    // The stride is over *trajectories*, not records, and that is the whole
    // design: it is content-independent (nothing about a record's value
    // decides whether it is kept) and it keeps every retained trajectory
    // complete, so a bootstrap clustered on realization_id stays valid. A
    // stride over records would do neither. At stride 1 the records still
    // reduce to the aggregate exactly; above 1 they no longer can, and the
    // header says so.
    long record_stride = 1;

    // --- provenance -------------------------------------------------------
    // The master seed. Every trajectory's layer draw is
    // trajectory_seed(seed, realization), so recording this makes any single
    // trajectory of a finished run replayable. Filled in at run start when the
    // caller did not pin one.
    std::uint64_t seed = 0;
    int statevector_precision = MIPT_CUDAQ_PRECISION;

    PairGapSettings pair_gap;

    // Parity-preserving circuits report both trace conventions. This is
    // deliberately `preserves_computational_parity` and not
    // `uses_fermionic_trace`: qRPPU (circ_type 4) simulates the same
    // parity-preserving gate set in the qubit encoding, and the point of
    // running it is to compare the two conventions on it.
    bool fermionic_outputs() const
    {
        return preserves_computational_parity(type);
    }

    // Which negativity the contingency counts classify on: the fermionic one
    // wherever it exists, and the ordinary one otherwise. Written into the CSV
    // so a mixed set of files is never ambiguous.
    const char *contingency_measure() const { return fermionic_outputs() ? "fmn" : "mn"; }

    // How the periodic bond is implemented. Recorded because the FSWAP network
    // and the Jordan-Wigner string are supposed to reduce to the *same* logical
    // bond (0, N-1); if a graph ever came out different between them, this is
    // the column that says which one produced it.
    const char *boundary_implementation() const
    {
        switch (type)
        {
        case CircuitType::FermionRPPU:
            return backend::direct_fermion_boundary_enabled() ? "jw_string" : "fswap_network";
        case CircuitType::QubitRPPU: return "qubit_direct";
        case CircuitType::RFGS: return "rfgs_wrapping_bond";
        case CircuitType::Haar: return "haar_wrapping_bond";
        case CircuitType::MMS: return "open_chain";
        }
        return "unknown";
    }
};

// The path an exclusive `k`-party run would write. Taking `k` explicitly rather
// than reading `config.k` is what lets a k=0 run name its two files exactly the
// way the two exclusive runs name theirs, so either can continue the other.
inline std::string default_output_path(const RunConfig &config, int k)
{
    const std::string tag(circuit_type_tag(config.type));
    std::ostringstream filename;
    filename << (k == 3 ? "dist_scaling3_" : "dist_scaling_") << tag
             << "_n_" << config.n
             << "_periods_" << config.periods
             << "_p_" << compact_decimal(config.p)
             << "_real_" << config.realizations;
    if (k == 3)
    {
        filename << "_b_" << compact_decimal(config.triangle_balance_cutoff);
    }
    filename << ".csv";
    return std::string("csv/dist_scaling/") + tag + "/" + filename.str();
}

inline std::string default_output_path(const RunConfig &config)
{
    return default_output_path(config, config.k);
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
    if (config.k != 0 && config.k != 2 && config.k != 3)
    {
        throw std::invalid_argument(
            "k must be 2 (pairs), 3 (triangles), or 0 (both, on shared trajectories).");
    }
    // k=0 measures triangles too, so it needs the three sites k does not name.
    const int minimum_sites = (config.k == 0) ? 3 : config.k;
    if (config.n < minimum_sites || config.n >= 63)
    {
        throw std::invalid_argument("N must satisfy " + std::to_string(minimum_sites) +
                                    " <= N < 63.");
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
    if (config.k != 2 &&
        (!std::isfinite(config.triangle_balance_cutoff) || config.triangle_balance_cutoff < 0.0 ||
         config.triangle_balance_cutoff > 1.0))
    {
        throw std::invalid_argument("B_min must lie in [0, 1].");
    }
    if (config.k == 0 && !config.output_path.empty())
    {
        throw std::invalid_argument(
            "k=0 writes one CSV per party count, so it cannot take a single explicit "
            "output path. Omit it to get the two default names, or run k=2 and k=3 "
            "separately.");
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
    // Mean of |G|^2 and |F|^2 over records, not |mean G|^2 -- the phases of a
    // two-point function average to zero over trajectories, so the squared
    // mean is not the same object and is not what decays with a useful
    // exponent. See dist::two_point_correlators.
    RunningStats g2;
    RunningStats f2;
    RunningStats fmi;
    RunningStats fmn;
    RunningStats fg2;
    RunningStats ff2;
    std::uint64_t mn_positive = 0;
    std::uint64_t fmn_positive = 0;

    // --- occupation-basis diagnostics -------------------------------------
    // Diagonal quantities, identical under both traces, so one copy each.
    RunningStats n_i;
    RunningStats n_j;
    RunningStats dnn;   // D = <n_i n_j>
    RunningStats rho_n; // D - <n_i><n_j>, signed
    // The signed mean above can cancel to zero across records of opposite sign
    // while every record is strongly correlated. These two cannot.
    RunningStats abs_rho_n;
    RunningStats rho_n_sq;
    RunningStats i_occ; // occupation-basis mutual information, bits

    // The disagreement between the closed-form fermionic negativity and the
    // generic partial transpose. Should be at the generic path's own noise
    // floor; a bin where it is not says the parity assumption failed there.
    RunningStats fmn_residual;
    double fmn_residual_max = 0.0;

    // --- the joint contingency table --------------------------------------
    //
    // Everything below is conditioned on connectivity having been evaluated at
    // all, which `conn_records` counts. The point of storing the full joint
    // table rather than the two marginals is that P(C) and P(E) alone cannot
    // separate "the graph is necessary but incomplete" from any number of
    // other stories; the conditionals
    //
    //     kappa = connected_ent_positive / connected
    //     eta   = disconnected_ent_positive / (conn_records - connected)
    //
    // can. `survive_i`/`survive_j`/`survive_both` are kept separately so that
    // P(A_i & A_j) is measured rather than approximated as q^2 -- the two
    // endpoints of a pair sit in the same trajectory and are not independent.
    std::uint64_t conn_records = 0;
    std::uint64_t survive_i = 0;
    std::uint64_t survive_j = 0;
    std::uint64_t survive_both = 0;
    std::uint64_t interior_path = 0;
    std::uint64_t connected = 0;
    std::uint64_t connected_ent_positive = 0;
    std::uint64_t connected_ent_zero = 0;
    std::uint64_t disconnected_ent_positive = 0;
    std::uint64_t disconnected_ent_zero = 0;

    // The three explanations for a record that is graph-connected but carries
    // no entanglement above threshold. Exhaustive and disjoint, so they sum to
    // `connected_ent_zero`.
    //
    //   1. occupation-correlated: I_occ > tol. Connected and classically
    //      correlated, but not fermionically entangled.
    //   2. subthreshold: I_occ ~ 0 but |G|^2 or |F|^2 is above the numerical
    //      floor, so there is coherence that the threshold discarded. These
    //      are the records that would move if eps moved.
    //   3. silent: I_occ ~ 0 and both channels at the floor. The spanning path
    //      exists and carries no detectable two-mode correlation at all.
    std::uint64_t classical_occ_correlated = 0;
    std::uint64_t classical_subthreshold = 0;
    std::uint64_t classical_silent = 0;

    // Conditional magnitudes. The signed rho_n above is deliberately not the
    // only density-correlation summary: cancellations between records of
    // opposite sign can make its mean small while every record is strongly
    // correlated, so the contingency counts carry the weight.
    RunningStats ent_given_connected;
    RunningStats ent_given_disconnected;
    RunningStats i_occ_connected_classical; // I_occ over the C=1, E=0 records

    // Second-stage explanatory diagnostics; see dist_connectivity.hpp.
    RunningStats component_size;
    RunningStats shortest_path; // over connected records only
};

// A GMN-family measure: its running mean plus the bookkeeping that says how
// the mean was obtained. `requested` counts records the schedule selected,
// `zero_prefiltered` the subset a vanishing bipartite negativity settled for
// free, and `solver_failures` the subset MOSEK could not solve -- those are
// excluded from the mean rather than counted as zero.
//
// `positive` counts the records whose measure came out strictly positive, the
// same quantity `PairBin::mn_positive` carries for the two-party negativities.
// It is what splits the mean into how often a triangle is genuinely entangled
// at all and how much it is when it is; without it the two are indivisible in
// the aggregate, since the CSV keeps no records.
//
// Prefiltered records need no test: the prefilter fires when the minimum
// bipartite negativity is at or below the tolerance, and GMN <= that minimum,
// so a prefiltered record is non-positive by construction. Only solved values
// are compared, against the *same* tolerance, which is what keeps one
// threshold behind both paths.
struct SdpStats
{
    RunningStats value;
    std::uint64_t requested = 0;
    std::uint64_t zero_prefiltered = 0;
    std::uint64_t solver_failures = 0;
    std::uint64_t positive = 0;
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

// The classification thresholds ride in the shared block at both party counts:
// a positive count, a contingency cell or a pair class cannot be read without
// the threshold that produced it, and a file that carried only one of the three
// (as the 2026-09-09 schema did) left the other two to be remembered.
inline constexpr const char *METADATA_COLUMNS =
    "N,circ_type,circuit_name,realizations,p,periods,k,entropy_units,"
    "statevector_precision,boundary_implementation,master_seed,"
    "pair_zero_tol,occupation_mi_tol,channel_floor";

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
    out += ",bits,";
    // fp32 vs fp64 decides where the generic partial transpose stops being
    // able to resolve a negativity at all, so a positivity threshold cannot be
    // interpreted without it.
    out += std::to_string(config.statevector_precision);
    out += ',';
    out += csv_quote(config.boundary_implementation());
    out += ',';
    append_uint(out, config.seed);
    out += ',';
    append_double(out, config.pair_zero_tol);
    out += ',';
    append_double(out, config.occupation_mi_tol);
    out += ',';
    append_double(out, config.channel_floor);
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

inline void append_counter(std::string &out, std::uint64_t value)
{
    out += ',';
    append_uint(out, value);
}

inline double positive_fraction(std::uint64_t positive, std::uint64_t total)
{
    return total > 0 ? static_cast<double>(positive) / static_cast<double>(total)
                     : std::numeric_limits<double>::quiet_NaN();
}

inline void append_sdp_stats(std::string &out, const SdpStats &stats)
{
    // Mean, then the positive count and fraction, then the schedule
    // bookkeeping -- so the leading five fields read exactly like a pair
    // negativity's and a reader can find `<stem>_positive_count` in the same
    // place for both party counts.
    append_stats(out, stats.value);
    out += ',';
    append_uint(out, stats.positive);
    out += ',';
    append_double(out, positive_fraction(stats.positive, stats.value.count));
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
    // Run-level constants that a reader must have before it can interpret any
    // count below: which threshold decided "entangled", which measure it was
    // applied to, and which graph convention produced the flags.
    header +=
        ",contingency_measure,connectivity_graph_version,connectivity_graph,"
        "separation,chord_length,d,embedding_count,"
        "mi_mean,mi_stderr,mi_samples,"
        "mn_mean,mn_stderr,mn_samples,mn_positive_count,mn_positive_fraction,"
        "g2_mean,g2_stderr,g2_samples,"
        "f2_mean,f2_stderr,f2_samples";
    if (config.fermionic_outputs())
    {
        header +=
            ",fmi_mean,fmi_stderr,fmi_samples,"
            "fmn_mean,fmn_stderr,fmn_samples,fmn_positive_count,fmn_positive_fraction,"
            "fg2_mean,fg2_stderr,fg2_samples,"
            "ff2_mean,ff2_stderr,ff2_samples,"
            "fmn_residual_mean,fmn_residual_stderr,fmn_residual_samples,fmn_residual_max";
    }
    header +=
        ",n_i_mean,n_i_stderr,n_i_samples,"
        "n_j_mean,n_j_stderr,n_j_samples,"
        "dnn_mean,dnn_stderr,dnn_samples,"
        "rho_n_mean,rho_n_stderr,rho_n_samples,"
        "abs_rho_n_mean,abs_rho_n_stderr,abs_rho_n_samples,"
        "rho_n_sq_mean,rho_n_sq_stderr,rho_n_sq_samples,"
        "i_occ_mean,i_occ_stderr,i_occ_samples,"
        "conn_records,"
        "survive_i_count,survive_j_count,survive_both_count,"
        "interior_path_count,connected_count,"
        "connected_ent_positive_count,connected_ent_zero_count,"
        "disconnected_ent_positive_count,disconnected_ent_zero_count,"
        "classical_occ_correlated_count,classical_subthreshold_count,"
        "classical_silent_count,"
        "ent_given_connected_mean,ent_given_connected_stderr,ent_given_connected_samples,"
        "ent_given_disconnected_mean,ent_given_disconnected_stderr,"
        "ent_given_disconnected_samples,"
        "i_occ_connected_classical_mean,i_occ_connected_classical_stderr,"
        "i_occ_connected_classical_samples,"
        "component_size_mean,component_size_stderr,component_size_samples,"
        "shortest_path_mean,shortest_path_stderr,shortest_path_samples";
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
        "gmn_mean,gmn_stderr,gmn_samples,gmn_positive_count,gmn_positive_fraction,"
        "gmn_requested_count,"
        "gmn_zero_prefilter_count,gmn_solver_failure_count,"
        "joint_purity_mean,joint_purity_stderr,joint_purity_samples,"
        "mean_single_purity_mean,mean_single_purity_stderr,mean_single_purity_samples";
    if (config.fermionic_outputs())
    {
        header +=
            ",ftmi_mean,ftmi_stderr,ftmi_samples,"
            "faverage_mi_mean,faverage_mi_stderr,faverage_mi_samples,"
            "min_fbipneg_mean,min_fbipneg_stderr,min_fbipneg_samples,"
            "fgmn_mean,fgmn_stderr,fgmn_samples,fgmn_positive_count,"
            "fgmn_positive_fraction,fgmn_requested_count,"
            "fgmn_zero_prefilter_count,fgmn_solver_failure_count,"
            "fjoint_purity_mean,fjoint_purity_stderr,fjoint_purity_samples,"
            "fmean_single_purity_mean,fmean_single_purity_stderr,fmean_single_purity_samples";
    }
    header += '\n';
    return header;
}

inline std::string render_pair_csv(const RunConfig &config, const std::vector<PairBin> &bins)
{
    std::string out = pair_csv_header(config);
    out.reserve(out.size() + bins.size() * 640u);
    std::string line;
    for (const PairBin &bin : bins)
    {
        line.clear();
        append_metadata_fields(line, config);
        line += ',';
        line += config.contingency_measure();
        line += ',';
        line += std::to_string(CONNECTIVITY_GRAPH_VERSION);
        line += ',';
        line += csv_quote(CONNECTIVITY_GRAPH_DEFINITION);
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
        append_stats(line, bin.g2);
        append_stats(line, bin.f2);
        if (config.fermionic_outputs())
        {
            append_stats(line, bin.fmi);
            append_stats(line, bin.fmn);
            line += ',';
            append_uint(line, bin.fmn_positive);
            line += ',';
            append_double(line, positive_fraction(bin.fmn_positive, bin.fmn.count));
            append_stats(line, bin.fg2);
            append_stats(line, bin.ff2);
            append_stats(line, bin.fmn_residual);
            line += ',';
            append_double(line, bin.fmn_residual_max);
        }
        append_stats(line, bin.n_i);
        append_stats(line, bin.n_j);
        append_stats(line, bin.dnn);
        append_stats(line, bin.rho_n);
        append_stats(line, bin.abs_rho_n);
        append_stats(line, bin.rho_n_sq);
        append_stats(line, bin.i_occ);
        append_counter(line, bin.conn_records);
        append_counter(line, bin.survive_i);
        append_counter(line, bin.survive_j);
        append_counter(line, bin.survive_both);
        append_counter(line, bin.interior_path);
        append_counter(line, bin.connected);
        append_counter(line, bin.connected_ent_positive);
        append_counter(line, bin.connected_ent_zero);
        append_counter(line, bin.disconnected_ent_positive);
        append_counter(line, bin.disconnected_ent_zero);
        append_counter(line, bin.classical_occ_correlated);
        append_counter(line, bin.classical_subthreshold);
        append_counter(line, bin.classical_silent);
        append_stats(line, bin.ent_given_connected);
        append_stats(line, bin.ent_given_disconnected);
        append_stats(line, bin.i_occ_connected_classical);
        append_stats(line, bin.component_size);
        append_stats(line, bin.shortest_path);
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
