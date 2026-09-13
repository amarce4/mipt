#pragma once

// The connected-zero triple analysis: where does a graph-connected pair's
// missing entanglement go?
//
// The question
// ------------
// dist_scaling.exe's contingency table established that a spanning path through
// the circuit is necessary for a pair to be entangled (eta ~ 0) and not
// sufficient (kappa < 1). The pairs responsible for kappa < 1 are
//
//     C_ij = 1   and   N^f_ij <= eps_2,
//
// graph-connected but with no fermionic negativity above threshold. One
// explanation is that the correlation the path carries is genuinely tripartite:
// invisible in the (i, j) marginal, but present once a third site is included.
// This analysis tests that directly. For every such pair it adds *every* third
// site k != i, j, forms the fermionic three-site RDM, and computes everything
// that can say where the correlation went -- the three pair marginals, the three
// one-versus-rest cut negativities, and fGMN.
//
// What a result means, and does not
// ---------------------------------
// A positive fGMN for a triple containing a silent pair demonstrates genuine
// tripartite entanglement hidden from that pair's marginal. It does not
// demonstrate genuine network multipartite entanglement. Zero fGMN for every k
// does not exclude entanglement that needs four or more sites; it rules out the
// three-site explanation at the chosen tolerance, and no more.
//
// Design decisions, each of which the tests pin
// ---------------------------------------------
//  * Every outcome is recorded. No balance filter, no survival filter on k, no
//    geometry sampling, no content-dependent sampling: exactly L-2 rows per
//    qualifying pair. The only selection is the trigger itself.
//  * Shared triples are computed once. Two anchors can generate the same sorted
//    {i,j,k}; its RDM and its fGMN are computed once per trajectory and fanned
//    out to every row that references it, each row keeping its own anchor.
//  * A prefilter is a bound, not a zero. fGMN <= min_s N_s, so a minimum cut at
//    or below the prefilter tolerance skips the solve and records
//    `prefilter_bound` with the bound itself; it enters averages as 0 and is
//    never described as an exact zero.
//  * A failed solve is missing, never zero. It is retried, and if it still
//    fails its row carries NaN and the solver status, and the aggregate refuses
//    to call the run complete.
//  * Nothing is lost in a crash. A trajectory's triples are all solved before
//    its block is written, the block is written before the pair checkpoint can
//    count the trajectory, and on resume both files are cut back to the pair
//    checkpoint and the aggregate is rebuilt from the outcome rows themselves.
//    Unlike the k=3 background queue, there is no in-flight state to lose.
//
// Nothing here depends on CUDA-Q, MOSEK or CUDA: the three-site RDM batch and
// the fGMN solver are injected, which is what lets `make test-dist` drive the
// whole protocol with first-principles RDMs and a fake solver.

#include "mipt/analysis/cut_negativity.hpp"
#include "mipt/analysis/assisted.hpp"
#include "mipt/analysis/separability.hpp"
#include "mipt/dist_pair_four.hpp"
#include "mipt/analysis/fgmn.hpp"
#include "mipt/dist_connectivity.hpp"
#include "mipt/dist_metrics.hpp"
#include "mipt/dist_scaling_csv.hpp"
#include "mipt/small_rdm.hpp"
#include "mipt/types.hpp"
#include "mipt/util/resume_csv.hpp"
#include "mipt/util/spectral.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace mipt::dist::gap
{

// 2: certified fGMN intervals (lower/upper bound, residuals, solver gap, the
// four-way fgmn_class) and the per-pair summary file. Version 1 rows recorded a
// single fGMN value and a bare positive/failed flag, which cannot be
// reclassified at another threshold, so they are not readable here.
inline constexpr int GAP_FORMAT_VERSION = 2;
// FGMN_STATUS_* are the solver's own codes; this one is ours.
inline constexpr int STATUS_PREFILTERED = -1;

// ---------------------------------------------------------------------------
// Certified classification
//
// A value is not "zero" or "nonzero"; it is an interval compared against a
// threshold, and the comparison has four outcomes rather than two. This is the
// whole point of the certified solve: at MOSEK's default tolerance the interval
// around a separable state is ~1e-5 wide, so a 1e-10 positivity threshold
// resolves *nothing*, and saying "not positive" there would be a claim the
// numbers do not support.
// ---------------------------------------------------------------------------

enum class CertifiedClass : int
{
    Positive = 0,              // lower bound is above the threshold
    BoundedBelowThreshold = 1, // upper bound is below the threshold
    Unresolved = 2,            // the interval straddles the threshold
    Failed = 3,                // no interval: the solve did not converge
};
inline constexpr int CERTIFIED_CLASS_COUNT = 4;

inline const char *certified_class_name(CertifiedClass value)
{
    switch (value)
    {
    case CertifiedClass::Positive: return "positive";
    case CertifiedClass::BoundedBelowThreshold: return "bounded_below_threshold";
    case CertifiedClass::Unresolved: return "unresolved";
    case CertifiedClass::Failed: break;
    }
    return "failed";
}

inline bool parse_certified_class(const std::string &text, CertifiedClass &out)
{
    for (int c = 0; c < CERTIFIED_CLASS_COUNT; ++c)
    {
        if (text == certified_class_name(static_cast<CertifiedClass>(c)))
        {
            out = static_cast<CertifiedClass>(c);
            return true;
        }
    }
    return false;
}

// A NaN bound is no bound: it can neither certify positivity nor certify
// smallness, so it leaves the state unresolved. `ok` is false for a solve that
// failed outright, which is a distinct outcome from an inconclusive one.
inline CertifiedClass classify_certified(double lower, double upper, double threshold, bool ok)
{
    if (!ok)
    {
        return CertifiedClass::Failed;
    }
    if (lower == lower && lower > threshold)
    {
        return CertifiedClass::Positive;
    }
    if (upper == upper && upper < threshold)
    {
        return CertifiedClass::BoundedBelowThreshold;
    }
    return CertifiedClass::Unresolved;
}
inline constexpr std::size_t RHO3_DOUBLES = 128;
inline constexpr std::size_t RHO2_DOUBLES = 32;

enum class AnchorRole : int
{
    Anchor = 0,
    Control = 1,
};
inline constexpr int ROLE_COUNT = 2;

inline const char *role_name(AnchorRole role)
{
    return role == AnchorRole::Anchor ? "anchor" : "control";
}

enum class FgmnMethod : int
{
    PrefilterBound = 0,
    Mosek = 1,
};

inline const char *method_name(FgmnMethod method)
{
    return method == FgmnMethod::Mosek ? "mosek" : "prefilter_bound";
}

inline std::string status_name(int status)
{
    switch (status)
    {
    case STATUS_PREFILTERED: return "prefiltered";
    case FGMN_STATUS_OK: return "ok";
    case FGMN_STATUS_EXCEPTION: return "exception";
    case FGMN_STATUS_NONFINITE: return "nonfinite";
    case FGMN_STATUS_NULL_INPUT: return "null_input";
    default: break;
    }
    if (status >= FGMN_STATUS_NOT_OPTIMAL_BASE)
    {
        static const char *names[] = {"undefined", "unknown", "optimal",
                                      "feasible",  "certificate", "illposed_certificate"};
        const int index = status - FGMN_STATUS_NOT_OPTIMAL_BASE;
        if (index >= 0 && index < 6)
        {
            return std::string("not_optimal_") + names[index];
        }
    }
    return "status_" + std::to_string(status);
}

// ---------------------------------------------------------------------------
// One trajectory's pair metrics
//
// PairProtocol::measure fills this for every pair, not just the qualifying
// ones, because a row's other two edges (i,k) and (j,k) are arbitrary pairs of
// the same trajectory and must be looked up rather than recomputed.
// ---------------------------------------------------------------------------

struct PairSnapshot
{
    int i = 0;
    int j = 0;
    int separation = 0;
    double chord = 0.0;
    PairMetrics fermion;           // fermionic trace; mn is the closed form
    PairConnectivity conn;
    bool entangled = false;        // fermion.mn > pair_zero_tol
    ZeroClass zero_class = ZeroClass::Silent; // meaningful only when !entangled
    const double *fermion_rho_ri = nullptr;   // raw reducer output, 32 doubles
};

inline bool is_connected_zero(const PairSnapshot &pair)
{
    return pair.conn.evaluated && pair.conn.connected && !pair.entangled;
}

inline bool is_disconnected_zero(const PairSnapshot &pair)
{
    return pair.conn.evaluated && !pair.conn.connected && !pair.entangled;
}

// The spec's per-pair record of a trigger. PairMeasurement::connected_zero lists
// every graph-connected unentangled pair of the trajectory, whatever its class;
// the analysis then filters by MIPT_DIST_PAIR_GAP_CLASS.
struct ConnectedZeroPair
{
    std::uint32_t realization = 0;
    std::size_t pair_index = 0;
    int i = 0;
    int j = 0;
    int separation = 0;
    double chord = 0.0;
    PairConnectivity conn;
    PairMetrics fermion;
    ZeroClass zero_class = ZeroClass::Silent;
};

struct PairMeasurement
{
    std::uint32_t realization = 0;
    int n = 0;
    std::vector<PairSnapshot> pairs;       // pair-enumeration order
    std::vector<int> index_of;             // n*n, (i,j) -> pairs index, -1 on the diagonal
    std::vector<ConnectedZeroPair> connected_zero;

    void reset(int sites)
    {
        n = sites;
        pairs.clear();
        connected_zero.clear();
        index_of.assign(static_cast<std::size_t>(sites) * static_cast<std::size_t>(sites), -1);
    }

    // Register one pair, reachable as (i, j) or (j, i), and list it among the
    // connected-zero pairs if it is one.
    void add(const PairSnapshot &snapshot)
    {
        const int index = static_cast<int>(pairs.size());
        index_of[static_cast<std::size_t>(snapshot.i) * static_cast<std::size_t>(n) +
                 static_cast<std::size_t>(snapshot.j)] = index;
        index_of[static_cast<std::size_t>(snapshot.j) * static_cast<std::size_t>(n) +
                 static_cast<std::size_t>(snapshot.i)] = index;
        if (is_connected_zero(snapshot))
        {
            ConnectedZeroPair zero;
            zero.realization = realization;
            zero.pair_index = static_cast<std::size_t>(index);
            zero.i = snapshot.i;
            zero.j = snapshot.j;
            zero.separation = snapshot.separation;
            zero.chord = snapshot.chord;
            zero.conn = snapshot.conn;
            zero.fermion = snapshot.fermion;
            zero.zero_class = snapshot.zero_class;
            connected_zero.push_back(zero);
        }
        pairs.push_back(snapshot);
    }

    const PairSnapshot &pair(int a, int b) const
    {
        const int index = index_of[static_cast<std::size_t>(a) * static_cast<std::size_t>(n) +
                                   static_cast<std::size_t>(b)];
        if (index < 0)
        {
            throw std::out_of_range("No pair (" + std::to_string(a) + ", " + std::to_string(b) +
                                    ") in this trajectory's measurement.");
        }
        return pairs[static_cast<std::size_t>(index)];
    }
};

// ---------------------------------------------------------------------------
// Anchor selection
// ---------------------------------------------------------------------------

struct Anchor
{
    std::size_t pair_index = 0;
    AnchorRole role = AnchorRole::Anchor;
    // The matched partner: an anchor's control, or a control's anchor. -1 when
    // there is none -- a control is only drawn when an eligible pair exists.
    int partner_i = -1;
    int partner_j = -1;
};

// Every qualifying pair, in pair-enumeration order, then -- if controls are on
// -- one distance-matched disconnected-zero pair per anchor from the same
// trajectory.
//
// The control draw is uniform over the eligible pairs at the anchor's
// separation, without replacement, from a stream derived from the master seed
// and the realization. That makes it reproducible and value-blind beyond the
// eligibility condition itself, which is the definition of the control group.
inline std::vector<Anchor> select_anchors(const PairMeasurement &measurement,
                                          const PairGapSettings &settings,
                                          std::uint64_t master_seed)
{
    std::vector<Anchor> anchors;
    for (std::size_t index = 0; index < measurement.pairs.size(); ++index)
    {
        const PairSnapshot &pair = measurement.pairs[index];
        if (is_connected_zero(pair) && gap_selects(settings.selection, pair.zero_class))
        {
            anchors.push_back({index, AnchorRole::Anchor, -1, -1});
        }
    }
    if (!settings.controls || anchors.empty())
    {
        return anchors;
    }

    const int max_separation = measurement.n / 2;
    std::vector<std::vector<std::size_t>> eligible(static_cast<std::size_t>(max_separation + 1));
    for (std::size_t index = 0; index < measurement.pairs.size(); ++index)
    {
        const PairSnapshot &pair = measurement.pairs[index];
        if (is_disconnected_zero(pair))
        {
            eligible[static_cast<std::size_t>(pair.separation)].push_back(index);
        }
    }
    std::mt19937_64 rng(seeding::trajectory_seed(master_seed ^ 0x636f6e74726f6c73ull,
                                        static_cast<std::uint64_t>(measurement.realization)));
    for (auto &pool : eligible)
    {
        std::shuffle(pool.begin(), pool.end(), rng);
    }

    std::vector<std::size_t> used(static_cast<std::size_t>(max_separation + 1), 0);
    const std::size_t anchor_count = anchors.size();
    for (std::size_t a = 0; a < anchor_count; ++a)
    {
        const PairSnapshot &anchor = measurement.pairs[anchors[a].pair_index];
        auto &pool = eligible[static_cast<std::size_t>(anchor.separation)];
        std::size_t &next = used[static_cast<std::size_t>(anchor.separation)];
        if (next >= pool.size())
        {
            continue;
        }
        const PairSnapshot &control = measurement.pairs[pool[next]];
        anchors[a].partner_i = control.i;
        anchors[a].partner_j = control.j;
        anchors.push_back({pool[next], AnchorRole::Control, anchor.i, anchor.j});
        ++next;
    }
    return anchors;
}

// ---------------------------------------------------------------------------
// Planning: every third site, and every shared triple computed once
// ---------------------------------------------------------------------------

// A canonical id for the sorted triple a < b < c: its colexicographic rank, so
// the same three sites have the same id in every trajectory and every run.
inline std::uint32_t triple_rank(int a, int b, int c)
{
    const auto choose2 = [](std::uint64_t x) { return x * (x - 1u) / 2u; };
    const auto choose3 = [](std::uint64_t x) { return x * (x - 1u) * (x - 2u) / 6u; };
    return static_cast<std::uint32_t>(choose3(static_cast<std::uint64_t>(c)) +
                                      choose2(static_cast<std::uint64_t>(b)) +
                                      static_cast<std::uint64_t>(a));
}

inline std::array<int, 3> sorted_triple(int x, int y, int z)
{
    std::array<int, 3> t{x, y, z};
    std::sort(t.begin(), t.end());
    return t;
}

struct TriplePlan
{
    struct Third
    {
        int k = 0;
        std::size_t triple = 0;
        bool first_use = false;
    };
    std::vector<std::array<int, 3>> triples; // unique, sorted, in first-use order
    std::vector<std::uint32_t> triple_ids;
    std::vector<std::vector<Third>> thirds;  // per anchor, k ascending, exactly n-2
};

inline TriplePlan plan_triples(const PairMeasurement &measurement, const std::vector<Anchor> &anchors)
{
    TriplePlan plan;
    const int n = measurement.n;
    const std::size_t rank_count = static_cast<std::size_t>(triple_rank(n - 3, n - 2, n - 1)) + 1u;
    std::vector<std::int64_t> slot(rank_count, -1);
    plan.thirds.resize(anchors.size());
    for (std::size_t a = 0; a < anchors.size(); ++a)
    {
        const PairSnapshot &pair = measurement.pairs[anchors[a].pair_index];
        auto &thirds = plan.thirds[a];
        thirds.reserve(static_cast<std::size_t>(n - 2));
        for (int k = 0; k < n; ++k)
        {
            if (k == pair.i || k == pair.j)
            {
                continue;
            }
            const std::array<int, 3> triple = sorted_triple(pair.i, pair.j, k);
            const std::uint32_t id = triple_rank(triple[0], triple[1], triple[2]);
            bool first = false;
            if (slot[id] < 0)
            {
                slot[id] = static_cast<std::int64_t>(plan.triples.size());
                plan.triples.push_back(triple);
                plan.triple_ids.push_back(id);
                first = true;
            }
            thirds.push_back({k, static_cast<std::size_t>(slot[id]), first});
        }
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Evaluating one unique triple
// ---------------------------------------------------------------------------

struct TripleResult
{
    std::array<int, 3> sites{};
    std::uint32_t id = 0;
    // Trace-normalized and Hermitized, interleaved: what every metric and the
    // solver see. The raw reducer output is kept by the caller for the
    // companion file.
    std::array<double, RHO3_DOUBLES> rho_ri{};
    TripleEntropies parts;
    double ftmi = 0.0;
    double average_fmi = 0.0;
    double joint_purity = 0.0;
    double mean_single_purity = 0.0;
    std::array<double, 3> cuts{};     // by sorted position: cuts[p] = N_F(site p | rest)
    double min_cut = 0.0;
    int min_cut_position = 0;
    double trace_error = 0.0;
    double hermiticity_error = 0.0;
    double min_eigenvalue = 0.0;
    double parity_leakage = 0.0;
    // Pair marginal against the independently reduced pair RDM, for the sorted
    // position pairs (0,1), (0,2), (1,2). This is the sign-ordering check: the
    // fermionic partial trace of the triple has to reproduce the pair.
    std::array<double, 3> marginal_residual{};
    // Per-cut separability diagnostics: what kind of PPT a zero cut is. The
    // ordinary-trace members are labelled `qubit_` in the CSV and are a
    // comparison observable, never a statement about fermionic separability.
    analysis::TripleSeparability separability;
    // Assisted endpoint negativity, indexed by which sorted position acts as
    // the measured helper. One triple serves several anchors with different
    // (i, j, k) assignments, so all three are precomputed here and the row
    // picks the one its own k names.
    std::array<analysis::AssistedNegativity, 3> assisted{};

    FgmnMethod method = FgmnMethod::PrefilterBound;
    double fgmn_raw = std::numeric_limits<double>::quiet_NaN();
    double fgmn = std::numeric_limits<double>::quiet_NaN();
    // The certified interval. On the prefilter path lower is 0 (fGMN >= 0 is a
    // theorem) and upper is the minimum cut, so a prefiltered triple is a
    // *bound*, never an exact zero. On the solver path they come from MOSEK's
    // primal and dual objectives.
    double lower_bound = 0.0;
    double upper_bound = std::numeric_limits<double>::quiet_NaN();
    // How far the *input* matrix is from being a density matrix. An fGMN can
    // never be certified below the accuracy of the matrix it was computed from,
    // and on an fp32 state vector that is ~1e-7 -- far above a 1e-10 threshold.
    // Leaving it out is how a control triple whose true fGMN is exactly zero
    // came back "certified positive" at 1.8e-8, which is fp32 noise and nothing
    // else.
    double input_uncertainty = 0.0;
    double primal_residual = std::numeric_limits<double>::quiet_NaN();
    double dual_residual = std::numeric_limits<double>::quiet_NaN();
    double solver_gap = std::numeric_limits<double>::quiet_NaN();
    CertifiedClass certified = CertifiedClass::Unresolved;
    bool positive = false;
    int status = STATUS_PREFILTERED;
    int attempts = 0;

    bool failed() const { return method == FgmnMethod::Mosek && status != FGMN_STATUS_OK; }
};

inline int position_mask(int p, int q)
{
    return (1 << p) | (1 << q);
}

// Largest elementwise difference between two 4x4 density matrices.
inline double max_difference(const ancilla::SmallRdm &a, const Matrix4 &b)
{
    double worst = 0.0;
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            worst = std::max(worst, std::abs(a(r, c) - b[static_cast<std::size_t>(r * 4 + c)]));
        }
    }
    return worst;
}

inline TripleResult evaluate_triple(const double *raw, const std::array<int, 3> &sites,
                                    std::uint32_t id, const PairMeasurement &measurement,
                                    const PairGapSettings &settings)
{
    TripleResult out;
    out.sites = sites;
    out.id = id;

    double trace = 0.0;
    for (int i = 0; i < 8; ++i)
    {
        trace += raw[2u * static_cast<std::size_t>(i * 8 + i)];
    }
    out.trace_error = std::abs(trace - 1.0);
    for (int r = 0; r < 8; ++r)
    {
        for (int c = 0; c < 8; ++c)
        {
            const std::size_t rc = 2u * static_cast<std::size_t>(r * 8 + c);
            const std::size_t cr = 2u * static_cast<std::size_t>(c * 8 + r);
            const double re = raw[rc] - raw[cr];
            const double im = raw[rc + 1] + raw[cr + 1];
            out.hermiticity_error = std::max(out.hermiticity_error, std::hypot(re, im));
        }
    }
    if (trace > 0.0)
    {
        out.hermiticity_error /= trace;
    }

    const ancilla::SmallRdm rho = normalized_small_rdm(raw, 8);
    std::array<std::complex<double>, 64> dense{};
    for (int r = 0; r < 8; ++r)
    {
        for (int c = 0; c < 8; ++c)
        {
            const std::complex<double> value = rho(r, c);
            dense[static_cast<std::size_t>(r * 8 + c)] = value;
            out.rho_ri[2u * static_cast<std::size_t>(r * 8 + c)] = value.real();
            out.rho_ri[2u * static_cast<std::size_t>(r * 8 + c) + 1u] = value.imag();
            if ((__builtin_popcount(static_cast<unsigned>(r)) & 1) !=
                (__builtin_popcount(static_cast<unsigned>(c)) & 1))
            {
                out.parity_leakage += std::abs(value);
            }
        }
    }

    out.parts = three_party_entropies(rho, true);
    const auto &S = out.parts.entropy;
    out.ftmi = S[1] + S[2] + S[4] - S[3] - S[5] - S[6] + S[7];
    out.average_fmi = ((S[1] + S[2] - S[3]) + (S[1] + S[4] - S[5]) + (S[2] + S[4] - S[6])) / 3.0;
    out.joint_purity = out.parts.purity[7];
    out.mean_single_purity = (out.parts.purity[1] + out.parts.purity[2] + out.parts.purity[4]) / 3.0;

    analysis::cut_negativities<3>(out.rho_ri.data(), true, out.cuts);
    out.min_cut = std::numeric_limits<double>::infinity();
    for (int p = 0; p < 3; ++p)
    {
        const double cut = out.cuts[static_cast<std::size_t>(p)];
        // A non-converged cut is NaN, and NaN must never win the minimum: the
        // minimum certifies a skipped solve.
        if (!(cut == cut))
        {
            out.min_cut = std::numeric_limits<double>::quiet_NaN();
            break;
        }
        if (cut < out.min_cut)
        {
            out.min_cut = cut;
            out.min_cut_position = p;
        }
    }

    out.separability = analysis::triple_separability(out.rho_ri.data(), out.parts.entropy,
                                                     settings.cut_positive_tol,
                                                     settings.correlation_tol);
    if (settings.assisted)
    {
        // Single-mode occupation measurement on each candidate helper. It is
        // the only single-mode family a fermionic experiment can realize
        // without an external parity reference; see analysis/assisted.hpp.
        for (int helper = 0; helper < 3; ++helper)
        {
            const unsigned helper_mask = 1u << helper;
            const unsigned endpoints = 7u & ~helper_mask;
            out.assisted[static_cast<std::size_t>(helper)] = analysis::assisted_negativity(
                rho, 3, endpoints, helper_mask, analysis::occupation_family(),
                settings.assisted_tol);
        }
    }

    std::array<double, 8> eigenvalues{};
    out.min_eigenvalue = util::hermitian_eigenvalues<8>(dense, eigenvalues)
                             ? *std::min_element(eigenvalues.begin(), eigenvalues.end())
                             : std::numeric_limits<double>::quiet_NaN();

    const std::array<std::array<int, 2>, 3> position_pairs{{{0, 1}, {0, 2}, {1, 2}}};
    for (std::size_t e = 0; e < 3; ++e)
    {
        const int p = position_pairs[e][0];
        const int q = position_pairs[e][1];
        const PairSnapshot &pair = measurement.pair(sites[static_cast<std::size_t>(p)],
                                                    sites[static_cast<std::size_t>(q)]);
        if (pair.fermion_rho_ri == nullptr)
        {
            out.marginal_residual[e] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const ancilla::SmallRdm marginal =
            ancilla::partial_trace_fixed(rho, 3, static_cast<unsigned>(position_mask(p, q)), true);
        out.marginal_residual[e] =
            max_difference(marginal, detail::normalized_hermitian_rho(pair.fermion_rho_ri));
    }

    // The matrix's own error, from the two diagnostics that measure it: how far
    // its trace is from 1 and how negative its smallest eigenvalue is. Both are
    // zero for an exact RDM and both are ~1e-7 on an fp32 state vector.
    out.input_uncertainty = std::max(out.trace_error, 0.0);
    if (out.min_eigenvalue == out.min_eigenvalue && out.min_eigenvalue < 0.0)
    {
        out.input_uncertainty = std::max(out.input_uncertainty, -out.min_eigenvalue);
    }
    out.input_uncertainty = std::max(out.input_uncertainty, out.hermiticity_error);

    // The prefilter decision. A NaN minimum is never prefiltered: without a
    // converged bound there is nothing to certify, so the SDP decides.
    if (out.min_cut == out.min_cut && out.min_cut <= settings.prefilter_tol)
    {
        out.method = FgmnMethod::PrefilterBound;
        out.status = STATUS_PREFILTERED;
        out.fgmn_raw = std::numeric_limits<double>::quiet_NaN();
        // fGMN <= min_s N_s is the bound the prefilter rests on, and 0 <= fGMN
        // is a theorem, so the triple is certified to lie in [0, min_cut]. It
        // enters averages at 0 -- the best point estimate inside that interval
        // -- but it is classified from the interval, so a positivity threshold
        // below min_cut leaves it unresolved instead of silently negative.
        out.fgmn = 0.0;
        out.lower_bound = 0.0;
        // The cut bound is only as good as the matrix it came from, so the
        // ceiling carries the input error too.
        out.upper_bound = out.min_cut + out.input_uncertainty;
        out.solver_gap = out.min_cut;
        out.certified =
            classify_certified(out.lower_bound, out.upper_bound, settings.positive_tol, true);
        out.positive = out.certified == CertifiedClass::Positive;
    }
    else
    {
        out.method = FgmnMethod::Mosek;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Solving
// ---------------------------------------------------------------------------

struct SolveOutcome
{
    double value = std::numeric_limits<double>::quiet_NaN();
    // The certified interval, as FgmnCertificate defines it. A solver that
    // supplies no bounds leaves these NaN, and the triple is then unresolved at
    // every threshold -- except that `value` itself is taken as the lower bound
    // (see apply_solve), because for this model -primalObjValue() is attained
    // at a primal-feasible point and is a lower bound on fGMN by construction.
    double lower_bound = std::numeric_limits<double>::quiet_NaN();
    double upper_bound = std::numeric_limits<double>::quiet_NaN();
    // How far the *input* matrix is from being a density matrix. An fGMN can
    // never be certified below the accuracy of the matrix it was computed from,
    // and on an fp32 state vector that is ~1e-7 -- far above a 1e-10 threshold.
    // Leaving it out is how a control triple whose true fGMN is exactly zero
    // came back "certified positive" at 1.8e-8, which is fp32 noise and nothing
    // else.
    double input_uncertainty = 0.0;
    double primal_residual = std::numeric_limits<double>::quiet_NaN();
    double dual_residual = std::numeric_limits<double>::quiet_NaN();
    double solver_gap = std::numeric_limits<double>::quiet_NaN();
    int status = FGMN_STATUS_EXCEPTION;

    // An exactly-known value: the interval collapses to a point. Used by tests
    // and by any solver that certifies its own answer some other way.
    static SolveOutcome exact(double value, int status = FGMN_STATUS_OK)
    {
        SolveOutcome out;
        out.value = value;
        out.lower_bound = value;
        out.upper_bound = value;
        out.solver_gap = 0.0;
        out.primal_residual = 0.0;
        out.dual_residual = 0.0;
        out.status = status;
        return out;
    }
};

using FgmnSolver = std::function<SolveOutcome(const double *rho_ri)>;

struct SolveJob
{
    const double *rho = nullptr;
    SolveOutcome outcome;
    int attempts = 0;
};

// Retry a failed solve up to `retries` more times; the last attempt holds
// `serial` so it runs alone, which is the one remedy for a failure that only
// concurrency provokes.
inline void solve_with_retries(SolveJob &job, const FgmnSolver &solver, int retries,
                               std::mutex *serial)
{
    for (int attempt = 0; attempt <= retries; ++attempt)
    {
        ++job.attempts;
        if (attempt == retries && attempt > 0 && serial != nullptr)
        {
            std::lock_guard<std::mutex> lock(*serial);
            job.outcome = solver(job.rho);
        }
        else
        {
            job.outcome = solver(job.rho);
        }
        if (job.outcome.status == FGMN_STATUS_OK && std::isfinite(job.outcome.value))
        {
            return;
        }
    }
    // Whatever the solver said, a failure is reported as a failure. A finite
    // value alongside a non-OK status is discarded rather than trusted.
    if (job.outcome.status == FGMN_STATUS_OK)
    {
        job.outcome.status = FGMN_STATUS_NONFINITE;
    }
    job.outcome.value = std::numeric_limits<double>::quiet_NaN();
}

// A persistent pool, because every MOSEK Fusion workspace is thread_local: a
// fresh thread per trajectory would rebuild the Fusion model for every handful
// of solves. `run` blocks until every job of the batch is done -- the batch is
// one trajectory, and nothing of it may be written before all of it is solved.
class SolverPool
{
  public:
    SolverPool(int workers, FgmnSolver solver, int retries)
        : solver_(std::move(solver)), retries_(std::max(0, retries))
    {
        const int count = std::max(1, workers);
        for (int w = 0; w < count; ++w)
        {
            threads_.emplace_back([this] { worker(); });
        }
    }

    ~SolverPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        work_.notify_all();
        for (std::thread &thread : threads_)
        {
            thread.join();
        }
    }

    SolverPool(const SolverPool &) = delete;
    SolverPool &operator=(const SolverPool &) = delete;

    void run(std::vector<SolveJob> &jobs)
    {
        if (jobs.empty())
        {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        jobs_ = &jobs;
        next_ = 0;
        remaining_ = jobs.size();
        in_flight_.store(jobs.size());
        ++generation_;
        work_.notify_all();
        done_.wait(lock, [this] { return remaining_ == 0; });
        jobs_ = nullptr;
    }

    const std::atomic<std::uint64_t> &in_flight_counter() const { return in_flight_; }
    const std::atomic<std::uint64_t> &solved_counter() const { return solved_; }
    int workers() const { return static_cast<int>(threads_.size()); }

  private:
    void worker()
    {
        std::uint64_t seen = 0;
        for (;;)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_.wait(lock, [&] { return stop_ || (generation_ != seen && jobs_ != nullptr); });
            if (stop_)
            {
                return;
            }
            seen = generation_;
            while (jobs_ != nullptr && next_ < jobs_->size())
            {
                SolveJob &job = (*jobs_)[next_++];
                lock.unlock();
                solve_with_retries(job, solver_, retries_, &serial_);
                in_flight_.fetch_sub(1);
                solved_.fetch_add(1);
                lock.lock();
                if (--remaining_ == 0)
                {
                    done_.notify_all();
                }
            }
        }
    }

    FgmnSolver solver_;
    int retries_ = 0;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::mutex serial_;
    std::condition_variable work_;
    std::condition_variable done_;
    std::vector<SolveJob> *jobs_ = nullptr;
    std::size_t next_ = 0;
    std::size_t remaining_ = 0;
    std::uint64_t generation_ = 0;
    bool stop_ = false;
    std::atomic<std::uint64_t> in_flight_{0};
    std::atomic<std::uint64_t> solved_{0};
};

inline void apply_solve(TripleResult &result, const SolveJob &job, double positive_tol)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    result.attempts = job.attempts;
    result.status = job.outcome.status;
    result.fgmn_raw = job.outcome.value;
    result.primal_residual = job.outcome.primal_residual;
    result.dual_residual = job.outcome.dual_residual;
    const bool ok = result.status == FGMN_STATUS_OK && std::isfinite(result.fgmn_raw);
    if (ok)
    {
        // The SDP optimum can sit a hair below zero on a separable state; the
        // raw value keeps that, the classified one clamps it. fGMN >= 0 is a
        // theorem, so clamping is tightening a valid bound, not hiding a sign.
        result.fgmn = std::max(0.0, result.fgmn_raw);
        result.lower_bound = std::isfinite(job.outcome.lower_bound)
                                 ? std::max(0.0, job.outcome.lower_bound)
                                 : result.fgmn;
        // MOSEK's own duality gap, kept as the solver diagnostic it is. It can
        // be *negative*: at GMN_MOSEK_TOL=1e-5 the returned primal and dual
        // points are each infeasible by ~1e-5, so the two objectives routinely
        // cross by ~1e-4. Measured on real RPPU triples, median -1.4e-4.
        result.solver_gap = std::isfinite(job.outcome.solver_gap)
                                ? job.outcome.solver_gap
                                : job.outcome.upper_bound - result.lower_bound;

        // The certified interval.
        //
        // The solver's answer is `value` with an uncertainty, and MOSEK reports
        // three things that bound it: the duality gap and the two feasibility
        // residuals. The widest of them is how far the returned point can be
        // from the true optimum, so the interval is value +- that.
        //
        // Taking the *gap* as an uncertainty rather than as a bracket is what
        // makes a crossed bracket usable. At any tolerance the primal and dual
        // objectives routinely cross -- both points are only feasible to
        // ~GMN_MOSEK_TOL -- and reading -dualObj literally would put the
        // ceiling *below* zero, which fGMN >= 0 forbids. Discarding it entirely
        // was the first fix and it was too blunt: it threw away the very
        // information that says the optimum is pinned, and left near-separable
        // triples permanently unresolved. The crossing magnitude *is* the
        // uncertainty, and using it as one resolves them correctly.
        double uncertainty = result.input_uncertainty;
        for (double candidate : {job.outcome.solver_gap, job.outcome.primal_residual,
                                 job.outcome.dual_residual,
                                 job.outcome.upper_bound - result.fgmn_raw})
        {
            if (std::isfinite(candidate))
            {
                uncertainty = std::max(uncertainty, std::abs(candidate));
            }
        }
        result.lower_bound = std::max(0.0, result.fgmn_raw - uncertainty);
        double upper = result.fgmn_raw + uncertainty;
        if (upper < 0.0)
        {
            upper = 0.0; // fGMN >= 0 is a theorem
        }
        // fGMN <= min_s N_s is a theorem too, evaluated from eigenvalues with
        // no solver in it, so it tightens the ceiling whenever it is smaller.
        if (result.min_cut == result.min_cut && result.min_cut < upper)
        {
            upper = result.min_cut;
        }
        result.upper_bound = upper;

        // ...and the same theorem is a consistency check. A floor above that
        // exact ceiling is proof the primal point is not feasible enough to
        // bound anything: at GMN_MOSEK_TOL=1e-10 this fires on near-separable
        // *control* triples, where the solver returns 1.8e-8 against an exact
        // ceiling of 1.3e-8. Without it, the empty interval would read as
        // certified positive and manufacture tripartite entanglement out of an
        // exact product state. The value is still reported; it stops being a
        // bound.
        if (result.lower_bound > upper)
        {
            result.lower_bound = std::numeric_limits<double>::quiet_NaN();
        }
    }
    else
    {
        result.fgmn = nan;
        result.lower_bound = nan;
        result.upper_bound = nan;
        result.solver_gap = nan;
    }
    result.certified = classify_certified(result.lower_bound, result.upper_bound, positive_tol, ok);
    result.positive = result.certified == CertifiedClass::Positive;
}

// ---------------------------------------------------------------------------
// The outcome CSV
// ---------------------------------------------------------------------------

// The column list, once. The header is rendered from it and the resume reader
// looks columns up in it by name, so the writer and the reader cannot disagree
// about a position. `make test-dist` checks every rendered row has exactly this
// many fields.
inline const std::vector<std::string> &outcome_columns()
{
    static const std::vector<std::string> columns = [] {
        std::vector<std::string> c{
            // identity
            "run_id", "realization_id", "outcome_id", "anchor_role", "pair_i", "pair_j", "third_k",
            "triple_id", "triple_site_a", "triple_site_b", "triple_site_c", "triple_first_use",
            "matched_i", "matched_j",
            // provenance
            "N", "periods", "p", "circ_type", "circuit_name", "master_seed", "statevector_precision",
            "boundary_implementation", "connectivity_graph_version", "gap_format_version",
            // thresholds
            "pair_zero_tol", "occupation_mi_tol", "channel_floor", "fgmn_positive_tol",
            "prefilter_tol", "cut_positive_tol", "correlation_tol", "mosek_tol",
            // anchor pair
            "separation", "chord", "pair_class", "pair_fn", "pair_fn_generic", "pair_fmi", "pair_g2",
            "pair_f2", "pair_re_g", "pair_im_g", "pair_re_f", "pair_im_f", "pair_n_i", "pair_n_j",
            "pair_dnn", "pair_rho_n", "pair_abs_rho_n", "pair_rho_n_sq", "pair_i_occ",
            "pair_parity_leakage",
            // anchor pair graph
            "survives_i", "survives_j", "interior_path_ij", "connected_ij", "component_size_ij",
            "shortest_path_ij", "idle_i", "idle_j", "edge_disjoint_paths_ij",
            "vertex_disjoint_paths_ij",
            // Component anatomy: how much of the component carries the pair.
            // min_edge_cut and min_vertex_cut equal the disjoint-path counts
            // above by Menger, so they are not repeated.
            "final_sites_in_component", "backbone_nodes", "articulation_nodes",
            "dangling_nodes", "branches", "mean_branch_size", "max_branch_size",
            "endpoint_degree_i", "endpoint_degree_j", "shortest_path_gates",
            "shortest_path_temporal",
            // Channel-resolved connectivity. Diagnostics beside the percolation
            // event, never a redefinition of it.
            "f_channel_log_weight", "g_channel_log_weight", "f_channel_bottleneck",
            "g_channel_bottleneck", "f_channel_multiplicity", "g_channel_multiplicity",
            // third-site graph
            "survives_k", "idle_k", "interior_path_ik", "connected_ik", "shortest_path_ik",
            "interior_path_jk", "connected_jk", "shortest_path_jk", "same_component_ijk"};
        for (const char *edge : {"ik", "jk"})
        {
            for (const char *field : {"fn", "fmi", "g2", "f2", "rho_n", "abs_rho_n", "i_occ",
                                      "entangled", "class"})
            {
                c.push_back(std::string(edge) + "_" + field);
            }
        }
        for (const char *field :
             {// triple
              "ftmi", "average_fmi", "joint_purity", "mean_single_purity", "cut_i", "cut_j", "cut_k",
              "min_cut", "min_cut_site", "cmi_ij_given_k", "mi_ij_k",
              // Per-cut separability. `qubit_` is the ordinary trace: a
              // comparison observable for a fermionic ensemble, never a claim
              // about fermionic separability.
              "mi_i_rest", "mi_j_rest", "mi_k_rest",
              "product_distance_i", "product_distance_j", "product_distance_k",
              "qubit_cut_i", "qubit_cut_j", "qubit_cut_k",
              "qubit_pt_min_eig_i", "qubit_pt_min_eig_j", "qubit_pt_min_eig_k",
              "ccnr_i", "ccnr_j", "ccnr_k",
              "cut_character_i", "cut_character_j", "cut_character_k",
              "qubit_ppt_ccnr_candidate",
              // Assisted (localizable) endpoint negativity under an occupation
              // measurement of k. Parity-respecting by construction.
              "assisted_fn_avg", "assisted_fn_max", "assisted_fn_positive_probability",
              "assisted_fn_outcomes", "assisted_fn_best_outcome", "assisted_family",
              // fGMN: the certified interval, then the point value derived from
              // it. fgmn_class is what R_3 is counted from; fgmn_positive is
              // exactly (fgmn_class == positive), kept so a reader that only
              // wants the boolean does not have to know the vocabulary.
              "fgmn_method", "fgmn_raw", "fgmn_lower_bound", "fgmn_upper_bound",
              "fgmn_primal_residual", "fgmn_dual_residual", "fgmn_solver_gap",
              "fgmn_input_uncertainty", "fgmn_class",
              "fgmn_min_cut_bound", "fgmn", "fgmn_positive", "fgmn_status", "fgmn_attempts",
              // numerical checks
              "trace_error", "hermiticity_error", "min_eigenvalue", "triple_parity_leakage",
              "marginal_residual_ij", "marginal_residual_ik", "marginal_residual_jk"})
        {
            c.push_back(field);
        }
        return c;
    }();
    return columns;
}

inline std::string outcome_csv_header()
{
    std::string header;
    const auto &columns = outcome_columns();
    for (std::size_t i = 0; i < columns.size(); ++i)
    {
        if (i > 0)
        {
            header += ',';
        }
        header += columns[i];
    }
    header += '\n';
    return header;
}

inline std::string run_id(const RunConfig &config)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "s%016llx",
                  static_cast<unsigned long long>(config.seed));
    return buffer;
}

inline void field(std::string &out, double value)
{
    out += ',';
    append_double(out, value);
}

inline void field(std::string &out, long long value)
{
    out += ',';
    out += std::to_string(value);
}

inline void field(std::string &out, const std::string &value)
{
    out += ',';
    out += value;
}

inline void field_flag(std::string &out, bool value)
{
    out += value ? ",1" : ",0";
}

inline const char *edge_class(const PairSnapshot &pair)
{
    return pair.entangled ? "entangled" : zero_class_name(pair.zero_class);
}

// Positions of anchor sites within the sorted triple.
struct RolePositions
{
    int i = 0;
    int j = 0;
    int k = 0;
};

inline RolePositions role_positions(const std::array<int, 3> &sites, int i, int j, int k)
{
    RolePositions out;
    for (int p = 0; p < 3; ++p)
    {
        if (sites[static_cast<std::size_t>(p)] == i) out.i = p;
        if (sites[static_cast<std::size_t>(p)] == j) out.j = p;
        if (sites[static_cast<std::size_t>(p)] == k) out.k = p;
    }
    return out;
}

inline std::size_t edge_slot(int p, int q)
{
    // (0,1) -> 0, (0,2) -> 1, (1,2) -> 2 regardless of argument order
    const int lo = std::min(p, q);
    const int hi = std::max(p, q);
    return lo == 0 ? static_cast<std::size_t>(hi - 1) : 2u;
}

// Fixed per-run text: the provenance and threshold block, rendered once.
struct RowPrefix
{
    std::string provenance;
    std::string thresholds;
};

inline RowPrefix row_prefix(const RunConfig &config)
{
    RowPrefix out;
    std::string &p = out.provenance;
    p += std::to_string(config.n);
    p += ',' + std::to_string(config.periods);
    p += ',';
    append_double(p, config.p);
    p += ',' + std::to_string(static_cast<int>(config.type));
    p += ',' + csv_quote(circuit_type_name(config.type));
    p += ',' + std::to_string(config.seed);
    p += ',' + std::to_string(config.statevector_precision);
    p += ',' + csv_quote(config.boundary_implementation());
    p += ',' + std::to_string(CONNECTIVITY_GRAPH_VERSION);
    p += ',' + std::to_string(GAP_FORMAT_VERSION);

    std::string &t = out.thresholds;
    append_double(t, config.pair_zero_tol);
    t += ',';
    append_double(t, config.occupation_mi_tol);
    t += ',';
    append_double(t, config.channel_floor);
    t += ',';
    append_double(t, config.pair_gap.positive_tol);
    t += ',';
    append_double(t, config.pair_gap.prefilter_tol);
    t += ',';
    append_double(t, config.pair_gap.cut_positive_tol);
    t += ',';
    append_double(t, config.pair_gap.correlation_tol);
    t += ',' + csv_quote(config.pair_gap.mosek_tol_text);
    return out;
}

// One row. `pair` is the anchor, `third` its added site, `result` the shared
// triple; everything role-dependent (which cut is "i", the CMI) is derived
// here rather than stored, since one triple serves several anchors.
inline void append_outcome_row(std::string &out, const RowPrefix &prefix,
                               const std::string &run, std::uint32_t realization,
                               std::uint64_t outcome_id, const Anchor &anchor,
                               const PairSnapshot &pair, const PairMeasurement &measurement,
                               const ConnectivityIndex &connectivity, const DisjointPaths &paths,
                               const ComponentAnatomy &anatomy, const ChannelConnectivity &channels,
                               const TriplePlan::Third &third,
                               const TripleResult &result)
{
    const int i = pair.i;
    const int j = pair.j;
    const int k = third.k;
    const PairSnapshot &ik = measurement.pair(std::min(i, k), std::max(i, k));
    const PairSnapshot &jk = measurement.pair(std::min(j, k), std::max(j, k));
    const PairConnectivity conn_ik = connectivity.query(i, k);
    const PairConnectivity conn_jk = connectivity.query(j, k);
    const RolePositions pos = role_positions(result.sites, i, j, k);
    const auto &S = result.parts.entropy;
    const int bi = 1 << pos.i;
    const int bj = 1 << pos.j;
    const int bk = 1 << pos.k;

    out += run;
    field(out, static_cast<long long>(realization));
    field(out, static_cast<long long>(outcome_id));
    field(out, std::string(role_name(anchor.role)));
    field(out, static_cast<long long>(i));
    field(out, static_cast<long long>(j));
    field(out, static_cast<long long>(k));
    field(out, static_cast<long long>(result.id));
    field(out, static_cast<long long>(result.sites[0]));
    field(out, static_cast<long long>(result.sites[1]));
    field(out, static_cast<long long>(result.sites[2]));
    field_flag(out, third.first_use);
    field(out, static_cast<long long>(anchor.partner_i));
    field(out, static_cast<long long>(anchor.partner_j));

    out += ',';
    out += prefix.provenance;
    out += ',';
    out += prefix.thresholds;

    const PairMetrics &f = pair.fermion;
    field(out, static_cast<long long>(pair.separation));
    field(out, pair.chord);
    field(out, std::string(zero_class_name(pair.zero_class)));
    field(out, f.mn);
    field(out, f.mn_generic);
    field(out, f.mi);
    field(out, f.g2);
    field(out, f.f2);
    field(out, f.g.real());
    field(out, f.g.imag());
    field(out, f.f.real());
    field(out, f.f.imag());
    field(out, f.n_i);
    field(out, f.n_j);
    field(out, f.dnn);
    field(out, f.rho_n);
    field(out, std::abs(f.rho_n));
    field(out, f.rho_n * f.rho_n);
    field(out, f.i_occ);
    field(out, f.parity_leakage);

    const PairConnectivity &c = pair.conn;
    field_flag(out, c.survives_i);
    field_flag(out, c.survives_j);
    field_flag(out, c.interior_path);
    field_flag(out, c.connected);
    field(out, static_cast<long long>(c.component_size));
    field(out, static_cast<long long>(c.shortest_path));
    field(out, static_cast<long long>(c.idle_i));
    field(out, static_cast<long long>(c.idle_j));
    field(out, static_cast<long long>(paths.edge));
    field(out, static_cast<long long>(paths.vertex));
    field(out, static_cast<long long>(anatomy.final_sites_in_component));
    field(out, static_cast<long long>(anatomy.backbone_nodes));
    field(out, static_cast<long long>(anatomy.articulation_nodes));
    field(out, static_cast<long long>(anatomy.dangling_nodes));
    field(out, static_cast<long long>(anatomy.branches));
    field(out, anatomy.mean_branch_size);
    field(out, static_cast<long long>(anatomy.max_branch_size));
    field(out, static_cast<long long>(anatomy.degree_i));
    field(out, static_cast<long long>(anatomy.degree_j));
    field(out, static_cast<long long>(anatomy.shortest_path_gates));
    field(out, static_cast<long long>(anatomy.shortest_path_temporal));
    field(out, channels.pair_log_weight);
    field(out, channels.hop_log_weight);
    field(out, channels.pair_bottleneck);
    field(out, channels.hop_bottleneck);
    field(out, static_cast<long long>(channels.pair_multiplicity));
    field(out, static_cast<long long>(channels.hop_multiplicity));

    field_flag(out, conn_ik.survives_j);
    field(out, static_cast<long long>(conn_ik.idle_j));
    field_flag(out, conn_ik.interior_path);
    field_flag(out, conn_ik.connected);
    field(out, static_cast<long long>(conn_ik.shortest_path));
    field_flag(out, conn_jk.interior_path);
    field_flag(out, conn_jk.connected);
    field(out, static_cast<long long>(conn_jk.shortest_path));
    field_flag(out, c.interior_path && conn_ik.interior_path);

    for (const PairSnapshot *edge : {&ik, &jk})
    {
        const PairMetrics &m = edge->fermion;
        field(out, m.mn);
        field(out, m.mi);
        field(out, m.g2);
        field(out, m.f2);
        field(out, m.rho_n);
        field(out, std::abs(m.rho_n));
        field(out, m.i_occ);
        field_flag(out, edge->entangled);
        field(out, std::string(edge_class(*edge)));
    }

    field(out, result.ftmi);
    field(out, result.average_fmi);
    field(out, result.joint_purity);
    field(out, result.mean_single_purity);
    field(out, result.cuts[static_cast<std::size_t>(pos.i)]);
    field(out, result.cuts[static_cast<std::size_t>(pos.j)]);
    field(out, result.cuts[static_cast<std::size_t>(pos.k)]);
    field(out, result.min_cut);
    {
        const int p = result.min_cut_position;
        const char *site = p == pos.i ? "i" : (p == pos.j ? "j" : "k");
        field(out, std::string(result.min_cut == result.min_cut ? site : "nan"));
    }
    // I(i:j|k) = S_ik + S_jk - S_k - S_ijk and I(ij:k) = S_ij + S_k - S_ijk, in
    // bits. Higher-order correlation diagnostics, not entanglement witnesses.
    field(out, S[bi | bk] + S[bj | bk] - S[bk] - S[7]);
    field(out, S[bi | bj] + S[bk] - S[7]);

    {
        const analysis::TripleSeparability &sep = result.separability;
        const std::array<int, 3> order{pos.i, pos.j, pos.k};
        for (int p : order) { field(out, sep.mutual_information[static_cast<std::size_t>(p)]); }
        for (int p : order) { field(out, sep.product_distance[static_cast<std::size_t>(p)]); }
        for (int p : order) { field(out, sep.qubit_cut[static_cast<std::size_t>(p)]); }
        for (int p : order)
        {
            field(out, sep.qubit_pt_min_eigenvalue[static_cast<std::size_t>(p)]);
        }
        for (int p : order) { field(out, sep.realignment[static_cast<std::size_t>(p)]); }
        for (int p : order)
        {
            field(out, std::string(analysis::cut_character_name(
                           sep.character[static_cast<std::size_t>(p)])));
        }
        field_flag(out, sep.qubit_ppt_ccnr_candidate);
        const analysis::AssistedNegativity &assisted =
            result.assisted[static_cast<std::size_t>(pos.k)];
        field(out, assisted.average);
        field(out, assisted.maximum);
        field(out, assisted.positive_probability);
        field(out, static_cast<long long>(assisted.outcomes));
        field(out, static_cast<long long>(assisted.best_outcome));
        field(out, std::string(assisted.evaluated ? "occupation" : "off"));
    }

    field(out, std::string(method_name(result.method)));
    field(out, result.fgmn_raw);
    field(out, result.lower_bound);
    field(out, result.upper_bound);
    field(out, result.primal_residual);
    field(out, result.dual_residual);
    field(out, result.solver_gap);
    field(out, result.input_uncertainty);
    field(out, std::string(certified_class_name(result.certified)));
    field(out, result.min_cut);
    field(out, result.fgmn);
    field_flag(out, result.positive);
    field(out, status_name(result.status));
    field(out, static_cast<long long>(result.attempts));

    field(out, result.trace_error);
    field(out, result.hermiticity_error);
    field(out, result.min_eigenvalue);
    field(out, result.parity_leakage);
    field(out, result.marginal_residual[edge_slot(pos.i, pos.j)]);
    field(out, result.marginal_residual[edge_slot(pos.i, pos.k)]);
    field(out, result.marginal_residual[edge_slot(pos.j, pos.k)]);
    out += '\n';
}

// ---------------------------------------------------------------------------
// The aggregate
//
// Built from row summaries only -- the same summaries the resume path parses
// back out of the outcome CSV -- so an aggregate rebuilt from the file and one
// accumulated live are the same computation over the same inputs.
// ---------------------------------------------------------------------------

// One outcome row, reduced to what the aggregate and the pair summary need.
//
// Everything here is either written to the outcome CSV or derived from columns
// that are, and `scan_outcomes` parses exactly these fields back, so an
// aggregate accumulated live and one rebuilt after a kill are the same
// computation over the same inputs. That is what makes the pair summary
// regenerable offline -- the audit mode reuses this struct verbatim.
struct RowSummary
{
    std::uint32_t realization = 0;
    int role = 0;
    int zero_class = 0;
    int separation = 0;
    int pair_i = 0;
    int pair_j = 0;
    int third_k = 0;
    bool prefiltered = false;
    bool failed = false;
    bool positive = false;
    bool first_use = false;
    CertifiedClass certified = CertifiedClass::Unresolved;
    double fgmn = std::numeric_limits<double>::quiet_NaN();
    double lower_bound = std::numeric_limits<double>::quiet_NaN();
    double upper_bound = std::numeric_limits<double>::quiet_NaN();
    double min_cut = std::numeric_limits<double>::quiet_NaN();
    double cmi_ij_given_k = std::numeric_limits<double>::quiet_NaN();
    double mi_ij_k = std::numeric_limits<double>::quiet_NaN();
    bool same_component = false;
    bool all_cuts_entangled = false;
    // Which of the anchor's three parties owns the minimum cut: 0 = i, 1 = j,
    // 2 = the added third, -1 when the cut did not converge. This is the
    // bipartition that holds fGMN down, so it is the diagnostic for *why* a
    // triple is not genuinely entangled.
    int min_cut_role = -1;

    // Constant across a pair's rows; carried on every row so a group can be
    // summarized without a second source.
    double chord = 0.0;
    double pair_fn = 0.0;
    double pair_fmi = 0.0;
    double pair_g2 = 0.0;
    double pair_f2 = 0.0;
    double pair_rho_n = 0.0;
    double pair_i_occ = 0.0;
    long long component_size = 0;
    long long shortest_path = 0;
    long long edge_disjoint = 0;
    long long vertex_disjoint = 0;
    ComponentAnatomy anatomy;
    ChannelConnectivity channels;
    bool survives_i = false;
    bool survives_j = false;
    bool interior_path = false;
    bool connected = false;
};

// ---------------------------------------------------------------------------
// The per-pair summary
//
// The individual-outcome CSV carries one row per (pair, third), which is the
// right record to keep and the wrong one to read: a single anchor appears N-2
// times and its verdict is spread across all of them. This reduces each anchor
// to one row ending in a single mutually exclusive explanatory class, which is
// what makes "are there any all-cuts-entangled, fGMN-zero candidates?"
// answerable by a grep rather than a groupby.
// ---------------------------------------------------------------------------

enum class ExplanatoryClass : int
{
    // Some third site carries certified tripartite entanglement.
    ThreeSiteFgmn = 0,
    // No positive third, but some third is entangled across every cut while
    // its fGMN is certified below threshold. This is the bound-entanglement
    // candidate -- a PPT-mixture-decomposable state with no separable cut --
    // and the reason the summary exists.
    ThreeSiteAllCutsEntangledFgmnZero = 1,
    // Every in-component third has a separable cut, so a vanishing fGMN is
    // already explained by the cut structure and needs no further mechanism.
    ThreeSiteSeparableCut = 2,
    // No third shares a spacetime component with the pair: the triples are
    // products and the question does not arise.
    NoSameComponentThird = 3,
    // No positive third, but at least one third's interval straddles the
    // threshold or failed. Nothing is being claimed here.
    NumericallyUnresolved = 4,
    // Everything resolved and bounded, thirds in the component, no cut
    // structure to explain it: the residue four-site or channel work targets.
    HigherOrderOrGraph = 5,
};
inline constexpr int EXPLANATORY_CLASS_COUNT = 6;

inline const char *explanatory_class_name(ExplanatoryClass value)
{
    switch (value)
    {
    case ExplanatoryClass::ThreeSiteFgmn: return "three_site_fgmn";
    case ExplanatoryClass::ThreeSiteAllCutsEntangledFgmnZero:
        return "three_site_all_cuts_entangled_fgmn_zero";
    case ExplanatoryClass::ThreeSiteSeparableCut: return "three_site_separable_cut";
    case ExplanatoryClass::NoSameComponentThird: return "no_same_component_third";
    case ExplanatoryClass::NumericallyUnresolved: return "numerically_unresolved";
    case ExplanatoryClass::HigherOrderOrGraph: break;
    }
    return "higher_order_or_graph";
}

struct PairSummaryRow
{
    std::uint32_t realization = 0;
    int role = 0;
    int zero_class = 0;
    int separation = 0;
    int pair_i = 0;
    int pair_j = 0;
    double chord = 0.0;

    double pair_fn = 0.0;
    double pair_fmi = 0.0;
    double pair_g2 = 0.0;
    double pair_f2 = 0.0;
    double pair_rho_n = 0.0;
    double pair_i_occ = 0.0;
    long long component_size = 0;
    long long shortest_path = 0;
    long long edge_disjoint = 0;
    long long vertex_disjoint = 0;
    ComponentAnatomy anatomy;
    ChannelConnectivity channels;
    bool survives_i = false;
    bool survives_j = false;
    bool interior_path = false;
    bool connected = false;

    std::uint32_t thirds = 0;
    std::uint32_t thirds_same_component = 0;
    std::uint32_t thirds_positive = 0;
    std::uint32_t thirds_bounded = 0;
    std::uint32_t thirds_unresolved = 0;
    std::uint32_t thirds_failed = 0;
    std::uint32_t thirds_all_cuts_entangled = 0;
    // How often each party owns the minimum cut over this pair's thirds.
    std::uint32_t blocked_by_i = 0;
    std::uint32_t blocked_by_j = 0;
    std::uint32_t blocked_by_k = 0;

    double max_min_cut = std::numeric_limits<double>::quiet_NaN();
    double max_lower_bound = std::numeric_limits<double>::quiet_NaN();
    double max_upper_bound = std::numeric_limits<double>::quiet_NaN();
    double max_cmi = std::numeric_limits<double>::quiet_NaN();
    double max_mi_ij_k = std::numeric_limits<double>::quiet_NaN();
    int best_k = -1; // the third with the largest certified lower bound

    bool robust_fgmn = false;
    ExplanatoryClass explanation = ExplanatoryClass::HigherOrderOrGraph;
};

// The largest of a set that may contain NaN, with NaN meaning "no value" rather
// than poisoning the maximum.
inline void keep_max(double &best, double candidate)
{
    if (candidate == candidate && (!(best == best) || candidate > best))
    {
        best = candidate;
    }
}

// Reduce one anchor's N-2 rows to a verdict.
//
// The class order is a precedence, and it is deliberate: a factual finding
// (something is positive) outranks an inconclusive one, which outranks every
// structural explanation, because a structural explanation offered over an
// unresolved interval would be a claim the numbers do not support.
inline PairSummaryRow summarize_pair(const RowSummary *rows, std::size_t count)
{
    PairSummaryRow out;
    if (count == 0)
    {
        return out;
    }
    const RowSummary &head = rows[0];
    out.realization = head.realization;
    out.role = head.role;
    out.zero_class = head.zero_class;
    out.separation = head.separation;
    out.pair_i = head.pair_i;
    out.pair_j = head.pair_j;
    out.chord = head.chord;
    out.pair_fn = head.pair_fn;
    out.pair_fmi = head.pair_fmi;
    out.pair_g2 = head.pair_g2;
    out.pair_f2 = head.pair_f2;
    out.pair_rho_n = head.pair_rho_n;
    out.pair_i_occ = head.pair_i_occ;
    out.component_size = head.component_size;
    out.shortest_path = head.shortest_path;
    out.edge_disjoint = head.edge_disjoint;
    out.vertex_disjoint = head.vertex_disjoint;
    out.anatomy = head.anatomy;
    out.channels = head.channels;
    out.survives_i = head.survives_i;
    out.survives_j = head.survives_j;
    out.interior_path = head.interior_path;
    out.connected = head.connected;
    out.thirds = static_cast<std::uint32_t>(count);

    double best_lower = std::numeric_limits<double>::quiet_NaN();
    bool candidate_all_cuts = false;
    bool separable_cut_everywhere = true;
    for (std::size_t r = 0; r < count; ++r)
    {
        const RowSummary &row = rows[r];
        out.thirds_same_component += row.same_component ? 1u : 0u;
        switch (row.certified)
        {
        case CertifiedClass::Positive: ++out.thirds_positive; break;
        case CertifiedClass::BoundedBelowThreshold: ++out.thirds_bounded; break;
        case CertifiedClass::Unresolved: ++out.thirds_unresolved; break;
        case CertifiedClass::Failed: ++out.thirds_failed; break;
        }
        switch (row.min_cut_role)
        {
        case 0: ++out.blocked_by_i; break;
        case 1: ++out.blocked_by_j; break;
        case 2: ++out.blocked_by_k; break;
        default: break;
        }
        if (row.all_cuts_entangled)
        {
            ++out.thirds_all_cuts_entangled;
            separable_cut_everywhere = false;
            if (row.certified == CertifiedClass::BoundedBelowThreshold)
            {
                candidate_all_cuts = true;
            }
        }
        keep_max(out.max_min_cut, row.min_cut);
        keep_max(out.max_upper_bound, row.upper_bound);
        keep_max(out.max_cmi, row.cmi_ij_given_k);
        keep_max(out.max_mi_ij_k, row.mi_ij_k);
        const double before = best_lower;
        keep_max(best_lower, row.lower_bound);
        if (!(before == best_lower))
        {
            out.best_k = row.third_k;
        }
    }
    out.max_lower_bound = best_lower;
    out.robust_fgmn = out.thirds_positive > 0;

    if (out.thirds_positive > 0)
    {
        out.explanation = ExplanatoryClass::ThreeSiteFgmn;
    }
    else if (out.thirds_unresolved > 0 || out.thirds_failed > 0)
    {
        out.explanation = ExplanatoryClass::NumericallyUnresolved;
    }
    else if (candidate_all_cuts)
    {
        out.explanation = ExplanatoryClass::ThreeSiteAllCutsEntangledFgmnZero;
    }
    else if (out.thirds_same_component == 0)
    {
        out.explanation = ExplanatoryClass::NoSameComponentThird;
    }
    else if (separable_cut_everywhere)
    {
        out.explanation = ExplanatoryClass::ThreeSiteSeparableCut;
    }
    else
    {
        out.explanation = ExplanatoryClass::HigherOrderOrGraph;
    }
    return out;
}

struct GapCell
{
    std::uint64_t qualifying_pairs = 0;
    std::uint64_t outcomes = 0;
    std::uint64_t outcomes_prefiltered = 0;
    std::uint64_t outcomes_mosek = 0;
    std::uint64_t outcomes_positive = 0;
    std::uint64_t outcomes_failed = 0;
    // R3: at least one k with fGMN above tolerance. A pair none of whose k are
    // positive but some of whose k failed is *undetermined*, not negative.
    std::uint64_t pairs_with_positive_third = 0;
    std::uint64_t pairs_undetermined = 0;
    std::vector<std::uint64_t> positive_third_hist; // index = positive thirds, 0..n-2
    // The four-way certified split of every outcome, and the pair-level
    // explanatory split. Summing either over the cells of a separation gives
    // that separation's total, which is what makes the decomposition add up.
    std::array<std::uint64_t, CERTIFIED_CLASS_COUNT> outcomes_by_class{};
    std::array<std::uint64_t, EXPLANATORY_CLASS_COUNT> pairs_by_explanation{};
    RunningStats max_fgmn_over_k;
    RunningStats mean_fgmn_over_k;
    RunningStats fgmn;
    // Unique-triple counts, attributed to the anchor whose row first used the
    // triple, so summing a column over every cell gives the exact global count.
    std::uint64_t triples_first_use = 0;
    std::uint64_t triples_prefiltered = 0;
    std::uint64_t triples_mosek = 0;
    std::uint64_t triples_positive = 0;
    std::uint64_t triples_failed = 0;
};

class GapAggregate
{
  public:
    void reset(int n)
    {
        n_ = n;
        max_separation_ = n / 2;
        cells_.assign(static_cast<std::size_t>(ROLE_COUNT * ZERO_CLASS_COUNT * max_separation_),
                      GapCell{});
        for (GapCell &cell : cells_)
        {
            cell.positive_third_hist.assign(static_cast<std::size_t>(n - 1), 0u);
        }
        rows_ = 0;
        failed_triples_ = 0;
    }

    int n() const { return n_; }
    int max_separation() const { return max_separation_; }
    std::uint64_t rows() const { return rows_; }
    std::uint64_t failed_triples() const { return failed_triples_; }

    GapCell &cell(int role, int zero_class, int separation)
    {
        return cells_[index(role, zero_class, separation)];
    }
    const GapCell &cell(int role, int zero_class, int separation) const
    {
        return cells_[index(role, zero_class, separation)];
    }

    // Every row of one (trajectory, role, pair), in k order. Returns the pair
    // summary it computed, so the live path and the resume path build the
    // summary file and the aggregate from one reduction rather than two.
    PairSummaryRow absorb_pair(const RowSummary *rows, std::size_t count)
    {
        PairSummaryRow summary = summarize_pair(rows, count);
        if (count == 0)
        {
            return summary;
        }
        const RowSummary &head = rows[0];
        GapCell &target = cell(head.role, head.zero_class, head.separation);
        ++target.qualifying_pairs;
        target.pairs_by_explanation[static_cast<std::size_t>(summary.explanation)] += 1u;
        std::size_t positives = 0;
        double largest = -std::numeric_limits<double>::infinity();
        double sum = 0.0;
        std::size_t valued = 0;
        for (std::size_t r = 0; r < count; ++r)
        {
            const RowSummary &row = rows[r];
            ++target.outcomes;
            ++rows_;
            target.outcomes_by_class[static_cast<std::size_t>(row.certified)] += 1u;
            if (row.prefiltered)
            {
                ++target.outcomes_prefiltered;
            }
            else
            {
                ++target.outcomes_mosek;
            }
            if (row.failed)
            {
                ++target.outcomes_failed;
            }
            else
            {
                target.fgmn.add(row.fgmn);
                largest = std::max(largest, row.fgmn);
                sum += row.fgmn;
                ++valued;
            }
            if (row.positive)
            {
                ++target.outcomes_positive;
                ++positives;
            }
            if (row.first_use)
            {
                ++target.triples_first_use;
                if (row.prefiltered)
                {
                    ++target.triples_prefiltered;
                }
                else
                {
                    ++target.triples_mosek;
                }
                if (row.positive)
                {
                    ++target.triples_positive;
                }
                if (row.failed)
                {
                    ++target.triples_failed;
                    ++failed_triples_;
                }
            }
        }
        if (positives > 0)
        {
            ++target.pairs_with_positive_third;
        }
        else if (summary.explanation == ExplanatoryClass::NumericallyUnresolved)
        {
            // A pair with no positive third but an interval that straddles the
            // threshold -- or a solve that failed -- is undetermined, not
            // negative. `failures` alone used to decide this; an unresolved
            // interval is the same kind of ignorance and now counts the same.
            ++target.pairs_undetermined;
        }
        target.positive_third_hist[std::min(positives, target.positive_third_hist.size() - 1u)] += 1u;
        if (valued > 0)
        {
            target.max_fgmn_over_k.add(largest);
            target.mean_fgmn_over_k.add(sum / static_cast<double>(valued));
        }
        return summary;
    }

  private:
    std::size_t index(int role, int zero_class, int separation) const
    {
        if (role < 0 || role >= ROLE_COUNT || zero_class < 0 || zero_class >= ZERO_CLASS_COUNT ||
            separation < 1 || separation > max_separation_)
        {
            throw std::out_of_range("Pair-gap aggregate cell out of range.");
        }
        return static_cast<std::size_t>((role * ZERO_CLASS_COUNT + zero_class) * max_separation_ +
                                        (separation - 1));
    }

    int n_ = 0;
    int max_separation_ = 0;
    std::vector<GapCell> cells_;
    std::uint64_t rows_ = 0;
    std::uint64_t failed_triples_ = 0;
};

// ---------------------------------------------------------------------------
// The pair-summary CSV
// ---------------------------------------------------------------------------

inline const std::vector<std::string> &pair_summary_columns()
{
    static const std::vector<std::string> columns{
        // identity
        "run_id", "realization_id", "anchor_role", "pair_i", "pair_j", "separation", "d",
        "pair_class",
        // provenance
        "N", "periods", "p", "circ_type", "circuit_name", "master_seed", "statevector_precision",
        "boundary_implementation", "connectivity_graph_version", "gap_format_version",
        // thresholds
        "pair_zero_tol", "occupation_mi_tol", "channel_floor", "fgmn_positive_tol", "prefilter_tol",
        "cut_positive_tol", "correlation_tol", "mosek_tol",
        // the pair itself
        "pair_fn", "pair_fmi", "pair_g2", "pair_f2", "pair_rho_n", "pair_abs_rho_n", "pair_i_occ",
        // graph diagnostics
        "survives_i", "survives_j", "interior_path_ij", "connected_ij", "component_size_ij",
        "shortest_path_ij", "edge_disjoint_paths_ij", "vertex_disjoint_paths_ij",
        "final_sites_in_component", "backbone_nodes", "articulation_nodes", "dangling_nodes",
        "branches", "mean_branch_size", "max_branch_size", "endpoint_degree_i",
        "endpoint_degree_j", "shortest_path_gates", "shortest_path_temporal",
        "f_channel_log_weight", "g_channel_log_weight", "f_channel_bottleneck",
        "g_channel_bottleneck", "f_channel_multiplicity", "g_channel_multiplicity",
        // the thirds
        "thirds", "thirds_same_component", "thirds_positive", "thirds_bounded_below_threshold",
        "thirds_unresolved", "thirds_failed", "thirds_all_cuts_entangled",
        "thirds_blocked_by_i", "thirds_blocked_by_j", "thirds_blocked_by_k",
        "max_min_cut_fn", "max_fgmn_lower_bound", "max_fgmn_upper_bound", "max_cmi_ij_given_k",
        "max_mi_ij_k", "best_third_k", "robust_fgmn", "explanatory_class"};
    return columns;
}

inline std::string pair_summary_csv_header()
{
    std::string header;
    const auto &columns = pair_summary_columns();
    for (std::size_t i = 0; i < columns.size(); ++i)
    {
        if (i > 0)
        {
            header += ',';
        }
        header += columns[i];
    }
    header += '\n';
    return header;
}

inline void append_pair_summary_row(std::string &out, const RowPrefix &prefix,
                                    const std::string &run, const PairSummaryRow &row)
{
    out += run;
    field(out, static_cast<long long>(row.realization));
    field(out, std::string(role_name(static_cast<AnchorRole>(row.role))));
    field(out, static_cast<long long>(row.pair_i));
    field(out, static_cast<long long>(row.pair_j));
    field(out, static_cast<long long>(row.separation));
    field(out, row.chord);
    field(out, std::string(zero_class_name(static_cast<ZeroClass>(row.zero_class))));

    out += ',';
    out += prefix.provenance;
    out += ',';
    out += prefix.thresholds;

    field(out, row.pair_fn);
    field(out, row.pair_fmi);
    field(out, row.pair_g2);
    field(out, row.pair_f2);
    field(out, row.pair_rho_n);
    field(out, std::abs(row.pair_rho_n));
    field(out, row.pair_i_occ);

    field_flag(out, row.survives_i);
    field_flag(out, row.survives_j);
    field_flag(out, row.interior_path);
    field_flag(out, row.connected);
    field(out, row.component_size);
    field(out, row.shortest_path);
    field(out, row.edge_disjoint);
    field(out, row.vertex_disjoint);
    field(out, static_cast<long long>(row.anatomy.final_sites_in_component));
    field(out, static_cast<long long>(row.anatomy.backbone_nodes));
    field(out, static_cast<long long>(row.anatomy.articulation_nodes));
    field(out, static_cast<long long>(row.anatomy.dangling_nodes));
    field(out, static_cast<long long>(row.anatomy.branches));
    field(out, row.anatomy.mean_branch_size);
    field(out, static_cast<long long>(row.anatomy.max_branch_size));
    field(out, static_cast<long long>(row.anatomy.degree_i));
    field(out, static_cast<long long>(row.anatomy.degree_j));
    field(out, static_cast<long long>(row.anatomy.shortest_path_gates));
    field(out, static_cast<long long>(row.anatomy.shortest_path_temporal));
    field(out, row.channels.pair_log_weight);
    field(out, row.channels.hop_log_weight);
    field(out, row.channels.pair_bottleneck);
    field(out, row.channels.hop_bottleneck);
    field(out, static_cast<long long>(row.channels.pair_multiplicity));
    field(out, static_cast<long long>(row.channels.hop_multiplicity));

    field(out, static_cast<long long>(row.thirds));
    field(out, static_cast<long long>(row.thirds_same_component));
    field(out, static_cast<long long>(row.thirds_positive));
    field(out, static_cast<long long>(row.thirds_bounded));
    field(out, static_cast<long long>(row.thirds_unresolved));
    field(out, static_cast<long long>(row.thirds_failed));
    field(out, static_cast<long long>(row.thirds_all_cuts_entangled));
    field(out, static_cast<long long>(row.blocked_by_i));
    field(out, static_cast<long long>(row.blocked_by_j));
    field(out, static_cast<long long>(row.blocked_by_k));
    field(out, row.max_min_cut);
    field(out, row.max_lower_bound);
    field(out, row.max_upper_bound);
    field(out, row.max_cmi);
    field(out, row.max_mi_ij_k);
    field(out, static_cast<long long>(row.best_k));
    field_flag(out, row.robust_fgmn);
    field(out, std::string(explanatory_class_name(row.explanation)));
    out += '\n';
}

inline std::string pair_summary_path_for(const std::string &outcome_csv)
{
    std::filesystem::path path(outcome_csv);
    path.replace_extension();
    std::string stem = path.string();
    // The outcome file is <main>_connected_zero_thirds.csv; the summary sits
    // beside it as <main>_connected_zero_pair_summary.csv, so the two names
    // share a prefix and glob apart.
    static const std::string suffix = "_connected_zero_thirds";
    if (stem.size() >= suffix.size() &&
        stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
        stem.resize(stem.size() - suffix.size());
        return stem + "_connected_zero_pair_summary.csv";
    }
    return stem + "_pair_summary.csv";
}

// The aggregate CSV: one row per (role, pair class, separation), carrying the
// pair-level partition beside it so
//
//     P(C) - P(E_2) = P(C, !E_2, R_3) + P(C, !E_2, !R_3) - P(!C, E_2)
//
// can be read from this file alone. The last term is the eta contribution; the
// two-term partition in the request assumes it vanishes, and keeping it here is
// what lets that assumption be checked rather than made. The `sep_*` columns
// come from the pair bins, which cover exactly the same trajectories.
inline std::string aggregate_csv_header(const RunConfig &config)
{
    std::string header(METADATA_COLUMNS);
    header += ",fgmn_positive_tol,prefilter_tol,cut_positive_tol,correlation_tol,mosek_tol,"
              "gap_selection,controls,"
              "gap_format_version,completed_realizations,run_complete,unresolved_failed_triples,"
              "anchor_role,pair_class,separation,d,"
              "qualifying_pairs,outcomes,outcomes_prefiltered,outcomes_mosek,outcomes_positive,"
              "outcomes_failed,outcomes_bounded_below_threshold,outcomes_unresolved,"
              "pairs_with_positive_third,pairs_undetermined,r3_fraction";
    for (int c = 0; c < EXPLANATORY_CLASS_COUNT; ++c)
    {
        header += ",pairs_" + std::string(explanatory_class_name(static_cast<ExplanatoryClass>(c)));
    }
    for (int m = 0; m <= config.n - 2; ++m)
    {
        header += ",positive_thirds_eq_" + std::to_string(m);
    }
    header += ",max_fgmn_over_k_mean,max_fgmn_over_k_stderr,max_fgmn_over_k_samples,"
              "mean_fgmn_over_k_mean,mean_fgmn_over_k_stderr,mean_fgmn_over_k_samples,"
              "fgmn_mean,fgmn_stderr,fgmn_samples,"
              "triples_first_use,triples_prefiltered,triples_mosek,triples_positive,triples_failed,"
              "sep_pair_records,sep_connected,sep_entangled,sep_connected_zero,"
              "sep_disconnected_entangled,sep_connected_zero_analyzed,"
              "sep_connected_zero_with_positive_third\n";
    return header;
}

inline std::string render_aggregate_csv(const RunConfig &config, const GapAggregate &aggregate,
                                        const std::vector<PairBin> &pair_bins,
                                        std::uint64_t completed)
{
    const PairGapSettings &settings = config.pair_gap;
    std::string out = aggregate_csv_header(config);
    RunConfig pair_config = config;
    pair_config.k = 2;
    const bool complete = completed >= static_cast<std::uint64_t>(config.realizations) &&
                          aggregate.failed_triples() == 0;
    const int roles = settings.controls ? ROLE_COUNT : 1;
    for (int role = 0; role < roles; ++role)
    {
        for (int zero_class = 0; zero_class < ZERO_CLASS_COUNT; ++zero_class)
        {
            for (int separation = 1; separation <= aggregate.max_separation(); ++separation)
            {
                const GapCell &cell = aggregate.cell(role, zero_class, separation);
                const PairBin &bin = pair_bins[static_cast<std::size_t>(separation - 1)];
                std::string line;
                append_metadata_fields(line, pair_config);
                field(line, settings.positive_tol);
                field(line, settings.prefilter_tol);
                field(line, settings.cut_positive_tol);
                field(line, settings.correlation_tol);
                field(line, csv_quote(settings.mosek_tol_text));
                field(line, std::string(gap_selection_name(settings.selection)));
                field_flag(line, settings.controls);
                field(line, static_cast<long long>(GAP_FORMAT_VERSION));
                field(line, static_cast<long long>(completed));
                field_flag(line, complete);
                field(line, static_cast<long long>(aggregate.failed_triples()));
                field(line, std::string(role_name(static_cast<AnchorRole>(role))));
                field(line, std::string(zero_class_name(static_cast<ZeroClass>(zero_class))));
                field(line, static_cast<long long>(separation));
                field(line, bin.chord);
                field(line, static_cast<long long>(cell.qualifying_pairs));
                field(line, static_cast<long long>(cell.outcomes));
                field(line, static_cast<long long>(cell.outcomes_prefiltered));
                field(line, static_cast<long long>(cell.outcomes_mosek));
                field(line, static_cast<long long>(cell.outcomes_positive));
                field(line, static_cast<long long>(cell.outcomes_failed));
                field(line, static_cast<long long>(
                                cell.outcomes_by_class[static_cast<std::size_t>(
                                    CertifiedClass::BoundedBelowThreshold)]));
                field(line, static_cast<long long>(
                                cell.outcomes_by_class[static_cast<std::size_t>(
                                    CertifiedClass::Unresolved)]));
                field(line, static_cast<long long>(cell.pairs_with_positive_third));
                field(line, static_cast<long long>(cell.pairs_undetermined));
                field(line, positive_fraction(cell.pairs_with_positive_third, cell.qualifying_pairs));
                for (std::uint64_t count : cell.pairs_by_explanation)
                {
                    field(line, static_cast<long long>(count));
                }
                for (std::uint64_t count : cell.positive_third_hist)
                {
                    field(line, static_cast<long long>(count));
                }
                append_stats(line, cell.max_fgmn_over_k);
                append_stats(line, cell.mean_fgmn_over_k);
                append_stats(line, cell.fgmn);
                field(line, static_cast<long long>(cell.triples_first_use));
                field(line, static_cast<long long>(cell.triples_prefiltered));
                field(line, static_cast<long long>(cell.triples_mosek));
                field(line, static_cast<long long>(cell.triples_positive));
                field(line, static_cast<long long>(cell.triples_failed));

                std::uint64_t analyzed = 0;
                std::uint64_t with_positive = 0;
                for (int c = 0; c < ZERO_CLASS_COUNT; ++c)
                {
                    const GapCell &anchors = aggregate.cell(0, c, separation);
                    analyzed += anchors.qualifying_pairs;
                    with_positive += anchors.pairs_with_positive_third;
                }
                field(line, static_cast<long long>(bin.conn_records));
                field(line, static_cast<long long>(bin.connected));
                field(line, static_cast<long long>(bin.connected_ent_positive +
                                                   bin.disconnected_ent_positive));
                field(line, static_cast<long long>(bin.connected_ent_zero));
                field(line, static_cast<long long>(bin.disconnected_ent_positive));
                field(line, static_cast<long long>(analyzed));
                field(line, static_cast<long long>(with_positive));
                line += '\n';
                out += line;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Durable files
// ---------------------------------------------------------------------------

// Append-only, flushed per trajectory and fsync-ed at every publish. A FILE*
// rather than an ofstream because only a file descriptor can be fsync-ed, and
// the ordering guarantee this analysis rests on -- a trajectory's rows are on
// disk before the pair checkpoint counts it -- has to survive the OS as well as
// a killed process.
class DurableAppender
{
  public:
    DurableAppender() = default;
    DurableAppender(const DurableAppender &) = delete;
    DurableAppender &operator=(const DurableAppender &) = delete;
    ~DurableAppender() { close(); }

    void open(const std::string &path)
    {
        close();
        file_ = std::fopen(path.c_str(), "ab");
        if (file_ == nullptr)
        {
            throw std::runtime_error("Could not open " + path + " for appending.");
        }
        path_ = path;
    }

    bool active() const { return file_ != nullptr; }

    void write(const void *data, std::size_t bytes)
    {
        if (bytes == 0)
        {
            // An empty block is how a disabled companion reports "nothing to
            // write", so it is not an error.
            return;
        }
        if (file_ == nullptr)
        {
            // Bytes for a file nobody opened. This used to be a silent no-op,
            // and it cost a debugging session: the four-site pass reported 120
            // rows written while the file did not exist, because its appender
            // had never been opened. A caller that genuinely has nothing to say
            // passes zero bytes; anything else is a wiring bug.
            throw std::runtime_error(
                "Tried to append " + std::to_string(bytes) +
                " byte(s) to a pair-gap output that was never opened.");
        }
        if (std::fwrite(data, 1, bytes, file_) != bytes || std::fflush(file_) != 0)
        {
            throw std::runtime_error("Failed while writing " + path_ + ".");
        }
    }

    void sync()
    {
        if (file_ != nullptr)
        {
            std::fflush(file_);
            ::fsync(::fileno(file_));
        }
    }

    void close()
    {
        if (file_ != nullptr)
        {
            sync();
            std::fclose(file_);
            file_ = nullptr;
        }
    }

  private:
    std::FILE *file_ = nullptr;
    std::string path_;
};

// Write-then-rename, so a reader -- or a resume -- never sees half a file.
inline void publish_atomically(const std::string &path, const std::string &contents)
{
    const std::string temporary = path + ".tmp";
    std::FILE *file = std::fopen(temporary.c_str(), "wb");
    if (file == nullptr)
    {
        throw std::runtime_error("Could not create " + temporary + ".");
    }
    const bool ok = std::fwrite(contents.data(), 1, contents.size(), file) == contents.size() &&
                    std::fflush(file) == 0 && ::fsync(::fileno(file)) == 0;
    std::fclose(file);
    if (!ok)
    {
        throw std::runtime_error("Failed while writing " + temporary + ".");
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error)
    {
        throw std::runtime_error("Could not move " + temporary + " into place: " +
                                 error.message());
    }
}

// The RDM companion: every unique triple's raw fermionic 8x8, once.
//
//     bytes 0..7   magic "MIPTGAP3"
//     bytes 8..11  uint32 version
//     bytes 12..15 uint32 record bytes (1040)
//     bytes 16..19 uint32 header text bytes
//     bytes 20..23 uint32 reserved
//     header text  key=value lines
//     records      uint32 realization, uint32 triple_id, uint8 a, b, c, pad,
//                  uint32 reserved, then 128 float64: the raw reducer output,
//                  interleaved row-major, *before* normalization, so the trace
//                  and Hermiticity diagnostics can be reproduced from it.
inline constexpr char RHO3_MAGIC[8] = {'M', 'I', 'P', 'T', 'G', 'A', 'P', '3'};
inline constexpr std::uint32_t RHO3_VERSION = 1;
inline constexpr std::size_t RHO3_PREAMBLE = 24;
inline constexpr std::size_t RHO3_RECORD_BYTES = 16 + RHO3_DOUBLES * sizeof(double);

inline std::string rho3_header_text(const RunConfig &config)
{
    std::string out;
    out += "format=mipt_pair_gap_rho3\n";
    out += "version=" + std::to_string(RHO3_VERSION) + "\n";
    out += "N=" + std::to_string(config.n) + "\n";
    out += "circ_type=" + std::to_string(static_cast<int>(config.type)) + "\n";
    out += "periods=" + std::to_string(config.periods) + "\n";
    std::string p;
    append_double(p, config.p);
    out += "p=" + p + "\n";
    out += "realizations=" + std::to_string(config.realizations) + "\n";
    out += "master_seed=" + std::to_string(config.seed) + "\n";
    out += "statevector_precision=" + std::to_string(config.statevector_precision) + "\n";
    out += "trace=fermionic\n";
    out += "normalization=raw_reducer_output\n";
    out += "fields=realization_id:u4,triple_id:u4,site_a:u1,site_b:u1,site_c:u1,pad:u1,"
           "reserved:u4,rho:f8x128\n";
    return out;
}

inline std::string rho3_preamble(const std::string &header)
{
    std::string out(RHO3_PREAMBLE, '\0');
    std::memcpy(out.data(), RHO3_MAGIC, 8);
    const std::uint32_t version = RHO3_VERSION;
    const std::uint32_t bytes = static_cast<std::uint32_t>(RHO3_RECORD_BYTES);
    const std::uint32_t header_bytes = static_cast<std::uint32_t>(header.size());
    std::memcpy(out.data() + 8, &version, 4);
    std::memcpy(out.data() + 12, &bytes, 4);
    std::memcpy(out.data() + 16, &header_bytes, 4);
    return out + header;
}

inline void append_rho3_record(std::string &out, std::uint32_t realization, std::uint32_t id,
                               const std::array<int, 3> &sites, const double *raw)
{
    char head[16] = {};
    std::memcpy(head, &realization, 4);
    std::memcpy(head + 4, &id, 4);
    head[8] = static_cast<char>(sites[0]);
    head[9] = static_cast<char>(sites[1]);
    head[10] = static_cast<char>(sites[2]);
    out.append(head, 16);
    out.append(reinterpret_cast<const char *>(raw), RHO3_DOUBLES * sizeof(double));
}

// ---------------------------------------------------------------------------
// Resume
// ---------------------------------------------------------------------------

struct OutcomeScan
{
    std::uint64_t rows_kept = 0;
    std::uint64_t bytes_kept = 0;
    std::uint64_t pairs_kept = 0;
    bool trimmed = false;
};

// Column positions in the outcome CSV, resolved once by name.
//
// The reader addresses columns by name and the writer renders them from the
// same list, so neither can drift into the other's positions. `scan_outcomes`
// and the offline audit share this, which is what makes the audit's
// reconstruction identical to the resume's rather than merely similar.
struct OutcomeColumns
{
    std::size_t realization = 0, outcome = 0, role = 0, pair_i = 0, pair_j = 0, third_k = 0;
    std::size_t separation = 0, chord = 0, zero_class = 0, method = 0, fgmn = 0, lower = 0;
    std::size_t upper = 0, certified = 0, status = 0, first_use = 0, min_cut = 0;
    std::size_t cut_i = 0, cut_j = 0, cut_k = 0, cmi = 0, mi_ij_k = 0, same_component = 0;
    std::size_t pair_fn = 0, pair_fmi = 0, pair_g2 = 0, pair_f2 = 0, pair_rho_n = 0, pair_i_occ = 0;
    std::size_t survives_i = 0, survives_j = 0, interior = 0, connected = 0, component_size = 0;
    std::size_t shortest_path = 0, edge_disjoint = 0, vertex_disjoint = 0, min_cut_site = 0;
    std::size_t final_sites = 0, backbone = 0, articulations = 0, dangling = 0, branches = 0;
    std::size_t mean_branch = 0, max_branch = 0, degree_i = 0, degree_j = 0;
    std::size_t path_gates = 0, path_temporal = 0;
    std::size_t f_log = 0, g_log = 0, f_bottleneck = 0, g_bottleneck = 0, f_mult = 0, g_mult = 0;

    static OutcomeColumns resolve()
    {
        const auto &columns = outcome_columns();
        auto at = [&](const char *name) {
            const auto it = std::find(columns.begin(), columns.end(), name);
            if (it == columns.end())
            {
                throw std::logic_error(std::string("outcome column '") + name + "' does not exist.");
            }
            return static_cast<std::size_t>(it - columns.begin());
        };
        OutcomeColumns c;
        c.realization = at("realization_id");
        c.outcome = at("outcome_id");
        c.role = at("anchor_role");
        c.pair_i = at("pair_i");
        c.pair_j = at("pair_j");
        c.third_k = at("third_k");
        c.separation = at("separation");
        c.chord = at("chord");
        c.zero_class = at("pair_class");
        c.method = at("fgmn_method");
        c.fgmn = at("fgmn");
        c.lower = at("fgmn_lower_bound");
        c.upper = at("fgmn_upper_bound");
        c.certified = at("fgmn_class");
        c.status = at("fgmn_status");
        c.first_use = at("triple_first_use");
        c.min_cut = at("min_cut");
        c.cut_i = at("cut_i");
        c.cut_j = at("cut_j");
        c.cut_k = at("cut_k");
        c.cmi = at("cmi_ij_given_k");
        c.mi_ij_k = at("mi_ij_k");
        c.same_component = at("same_component_ijk");
        c.pair_fn = at("pair_fn");
        c.pair_fmi = at("pair_fmi");
        c.pair_g2 = at("pair_g2");
        c.pair_f2 = at("pair_f2");
        c.pair_rho_n = at("pair_rho_n");
        c.pair_i_occ = at("pair_i_occ");
        c.survives_i = at("survives_i");
        c.survives_j = at("survives_j");
        c.interior = at("interior_path_ij");
        c.connected = at("connected_ij");
        c.component_size = at("component_size_ij");
        c.shortest_path = at("shortest_path_ij");
        c.edge_disjoint = at("edge_disjoint_paths_ij");
        c.vertex_disjoint = at("vertex_disjoint_paths_ij");
        c.min_cut_site = at("min_cut_site");
        c.final_sites = at("final_sites_in_component");
        c.backbone = at("backbone_nodes");
        c.articulations = at("articulation_nodes");
        c.dangling = at("dangling_nodes");
        c.branches = at("branches");
        c.mean_branch = at("mean_branch_size");
        c.max_branch = at("max_branch_size");
        c.degree_i = at("endpoint_degree_i");
        c.degree_j = at("endpoint_degree_j");
        c.path_gates = at("shortest_path_gates");
        c.path_temporal = at("shortest_path_temporal");
        c.f_log = at("f_channel_log_weight");
        c.g_log = at("g_channel_log_weight");
        c.f_bottleneck = at("f_channel_bottleneck");
        c.g_bottleneck = at("g_channel_bottleneck");
        c.f_mult = at("f_channel_multiplicity");
        c.g_mult = at("g_channel_multiplicity");
        return c;
    }
};

// Rebuild one RowSummary from a parsed outcome row.
//
// `positive_tol` and `cut_positive_tol` are applied here rather than taken from
// the row's own recorded fgmn_class, which is what lets the offline audit
// reclassify a finished run at another threshold without re-solving: the
// certified interval is the datum, the class is a reading of it.
inline RowSummary row_summary_from_fields(const std::vector<std::string> &fields,
                                          const OutcomeColumns &c, const std::string &path,
                                          double positive_tol, double cut_positive_tol)
{
    auto number = [&](std::size_t index, const char *name) {
        return util::resume::parse_double_field(fields[index], path + " " + name);
    };
    auto flag = [&](std::size_t index) { return fields[index] == "1"; };
    auto integer = [&](std::size_t index) { return std::stol(fields[index]); };

    RowSummary row;
    row.realization = static_cast<std::uint32_t>(integer(c.realization));
    row.role = fields[c.role] == "control" ? 1 : 0;
    row.separation = static_cast<int>(integer(c.separation));
    row.pair_i = static_cast<int>(integer(c.pair_i));
    row.pair_j = static_cast<int>(integer(c.pair_j));
    row.third_k = static_cast<int>(integer(c.third_k));
    row.chord = number(c.chord, "chord");
    row.prefiltered = fields[c.method] == "prefilter_bound";
    row.first_use = flag(c.first_use);
    row.fgmn = number(c.fgmn, "fgmn");
    row.lower_bound = number(c.lower, "fgmn_lower_bound");
    row.upper_bound = number(c.upper, "fgmn_upper_bound");
    row.min_cut = number(c.min_cut, "min_cut");
    row.cmi_ij_given_k = number(c.cmi, "cmi_ij_given_k");
    row.mi_ij_k = number(c.mi_ij_k, "mi_ij_k");
    row.same_component = flag(c.same_component);
    row.failed = !row.prefiltered && fields[c.status] != "ok";
    row.certified =
        classify_certified(row.lower_bound, row.upper_bound, positive_tol, !row.failed);
    row.positive = row.certified == CertifiedClass::Positive;

    // Every cut strictly above the threshold: no bipartition of the triple is
    // separable, so a vanishing fGMN cannot be explained by a product cut.
    // A NaN cut is not "entangled" -- it is not anything.
    bool all_cuts = true;
    for (std::size_t index : {c.cut_i, c.cut_j, c.cut_k})
    {
        const double cut = number(index, "cut");
        all_cuts = all_cuts && cut == cut && cut > cut_positive_tol;
    }
    row.all_cuts_entangled = all_cuts;
    {
        const std::string &site = fields[c.min_cut_site];
        row.min_cut_role = site == "i" ? 0 : (site == "j" ? 1 : (site == "k" ? 2 : -1));
    }

    row.pair_fn = number(c.pair_fn, "pair_fn");
    row.pair_fmi = number(c.pair_fmi, "pair_fmi");
    row.pair_g2 = number(c.pair_g2, "pair_g2");
    row.pair_f2 = number(c.pair_f2, "pair_f2");
    row.pair_rho_n = number(c.pair_rho_n, "pair_rho_n");
    row.pair_i_occ = number(c.pair_i_occ, "pair_i_occ");
    row.component_size = integer(c.component_size);
    row.shortest_path = integer(c.shortest_path);
    row.edge_disjoint = integer(c.edge_disjoint);
    row.vertex_disjoint = integer(c.vertex_disjoint);
    row.anatomy.evaluated = true;
    row.anatomy.component_nodes = static_cast<std::uint32_t>(integer(c.component_size));
    row.anatomy.final_sites_in_component = static_cast<std::uint32_t>(integer(c.final_sites));
    row.anatomy.backbone_nodes = static_cast<std::uint32_t>(integer(c.backbone));
    row.anatomy.articulation_nodes = static_cast<std::uint32_t>(integer(c.articulations));
    row.anatomy.dangling_nodes = static_cast<std::uint32_t>(integer(c.dangling));
    row.anatomy.branches = static_cast<std::uint32_t>(integer(c.branches));
    row.anatomy.mean_branch_size = number(c.mean_branch, "mean_branch_size");
    row.anatomy.max_branch_size = static_cast<std::uint32_t>(integer(c.max_branch));
    row.anatomy.degree_i = static_cast<std::uint32_t>(integer(c.degree_i));
    row.anatomy.degree_j = static_cast<std::uint32_t>(integer(c.degree_j));
    row.anatomy.shortest_path_gates = static_cast<std::int32_t>(integer(c.path_gates));
    row.anatomy.shortest_path_temporal = static_cast<std::int32_t>(integer(c.path_temporal));
    row.anatomy.min_edge_cut = static_cast<int>(row.edge_disjoint);
    row.anatomy.min_vertex_cut = static_cast<int>(row.vertex_disjoint);
    row.channels.evaluated = true;
    row.channels.pair_log_weight = number(c.f_log, "f_channel_log_weight");
    row.channels.hop_log_weight = number(c.g_log, "g_channel_log_weight");
    row.channels.pair_bottleneck = number(c.f_bottleneck, "f_channel_bottleneck");
    row.channels.hop_bottleneck = number(c.g_bottleneck, "g_channel_bottleneck");
    row.channels.pair_multiplicity = static_cast<int>(integer(c.f_mult));
    row.channels.hop_multiplicity = static_cast<int>(integer(c.g_mult));
    row.survives_i = flag(c.survives_i);
    row.survives_j = flag(c.survives_j);
    row.interior_path = flag(c.interior);
    row.connected = flag(c.connected);
    return row;
}

// Stream the outcome CSV, rebuilding the aggregate from every row whose
// realization is below `completed`, and report where the kept prefix ends.
//
// This is the only place the aggregate comes back from, and it is not a
// shortcut: the aggregate CSV is published at progress intervals, so after a
// kill it can hold trajectories the pair checkpoint does not. The outcome rows
// are the record of what was measured; the aggregate is recomputed from them,
// which is also what makes the two reconcile exactly.
using PairSummarySink = std::function<void(const PairSummaryRow &)>;

inline OutcomeScan scan_outcomes(const std::string &path, std::uint64_t completed,
                                 GapAggregate &aggregate, const PairGapSettings &settings,
                                 const PairSummarySink &sink = PairSummarySink())
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open " + path + " for reading.");
    }
    const std::string expected = outcome_csv_header();
    std::string line;
    if (!std::getline(file, line) || line + '\n' != expected)
    {
        util::resume::refuse(
            path + " has a different column layout than this binary writes. Finish it with "
                   "the binary that wrote it, move it aside, or set MIPT_DIST_RESUME=0.");
    }
    const auto &columns = outcome_columns();
    const OutcomeColumns c = OutcomeColumns::resolve();

    OutcomeScan scan;
    scan.bytes_kept = expected.size();
    std::vector<RowSummary> group;
    std::uint32_t last_realization = 0;
    auto flush_group = [&] {
        if (!group.empty())
        {
            if (group.size() != static_cast<std::size_t>(aggregate.n() - 2))
            {
                util::resume::refuse(
                    path + " holds " + std::to_string(group.size()) + " rows for pair (" +
                    std::to_string(group.front().pair_i) + ", " +
                    std::to_string(group.front().pair_j) + ") of trajectory " +
                    std::to_string(group.front().realization) + "; every qualifying pair has "
                    "exactly N-2 = " + std::to_string(aggregate.n() - 2) + ". The file is "
                    "damaged.");
            }
            const PairSummaryRow summary = aggregate.absorb_pair(group.data(), group.size());
            ++scan.pairs_kept;
            if (sink)
            {
                sink(summary);
            }
            group.clear();
        }
    };
    auto parse_int = [&](const std::string &text) { return std::stol(text); };
    auto class_of = [&](const std::string &text) {
        for (int c = 0; c < ZERO_CLASS_COUNT; ++c)
        {
            if (text == zero_class_name(static_cast<ZeroClass>(c)))
            {
                return c;
            }
        }
        util::resume::refuse(path + " names an unknown pair class '" + text + "'.");
    };

    while (true)
    {
        const std::streampos start = file.tellg();
        if (!std::getline(file, line))
        {
            break;
        }
        if (file.eof())
        {
            // No terminating newline: a write the kill interrupted.
            scan.trimmed = true;
            break;
        }
        const std::vector<std::string> fields = util::resume::split_csv_row(line);
        if (fields.size() != columns.size())
        {
            scan.trimmed = true;
            break;
        }
        const std::uint32_t realization =
            static_cast<std::uint32_t>(parse_int(fields[c.realization]));
        if (realization < last_realization)
        {
            util::resume::refuse(path + " is not in trajectory order at outcome " +
                                 fields[c.outcome] + ".");
        }
        if (realization >= completed)
        {
            scan.trimmed = true;
            break;
        }
        if (static_cast<std::uint64_t>(parse_int(fields[c.outcome])) != scan.rows_kept)
        {
            util::resume::refuse(path + " skips or repeats an outcome_id near row " +
                                 std::to_string(scan.rows_kept + 2) + ".");
        }
        RowSummary row = row_summary_from_fields(fields, c, path, settings.positive_tol,
                                                 settings.cut_positive_tol);
        row.zero_class = class_of(fields[c.zero_class]);

        if (!group.empty() &&
            (group.front().realization != row.realization || group.front().role != row.role ||
             group.front().pair_i != row.pair_i || group.front().pair_j != row.pair_j))
        {
            flush_group();
        }
        group.push_back(row);
        last_realization = realization;
        ++scan.rows_kept;
        scan.bytes_kept = static_cast<std::uint64_t>(start) + line.size() + 1u;
    }
    flush_group();
    return scan;
}

// Keep the companion records whose realization is below `completed`. Fixed
// width and in realization order, so the cut is a binary search.
inline void trim_rho3(const std::string &path, const std::string &header, std::uint64_t completed)
{
    std::ifstream file(path, std::ios::binary);
    char preamble[RHO3_PREAMBLE];
    file.read(preamble, RHO3_PREAMBLE);
    if (!file || std::memcmp(preamble, RHO3_MAGIC, 8) != 0)
    {
        util::resume::refuse(path + " is not a pair-gap RDM companion.");
    }
    std::uint32_t header_bytes = 0;
    std::memcpy(&header_bytes, preamble + 16, 4);
    std::string stored(header_bytes, '\0');
    file.read(stored.data(), header_bytes);
    if (stored != header)
    {
        util::resume::refuse(path + " describes a different run. Move it aside, or set "
                                    "MIPT_DIST_PAIR_GAP_STORE_RHO3=0.");
    }
    const std::uint64_t offset = RHO3_PREAMBLE + header_bytes;
    const std::uint64_t size = std::filesystem::file_size(path);
    const std::uint64_t count = size > offset ? (size - offset) / RHO3_RECORD_BYTES : 0;
    auto realization_at = [&](std::uint64_t index) {
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset + index * RHO3_RECORD_BYTES));
        std::uint32_t value = 0;
        file.read(reinterpret_cast<char *>(&value), 4);
        return value;
    };
    std::uint64_t low = 0;
    std::uint64_t high = count;
    while (low < high)
    {
        const std::uint64_t mid = low + (high - low) / 2;
        if (realization_at(mid) < completed)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    file.close();
    std::filesystem::resize_file(path, offset + low * RHO3_RECORD_BYTES);
}

// ---------------------------------------------------------------------------
// The analysis driver
// ---------------------------------------------------------------------------

inline std::string default_gap_output(const std::string &pair_csv)
{
    std::filesystem::path path(pair_csv);
    path.replace_extension();
    return path.string() + "_connected_zero_thirds.csv";
}

inline std::string aggregate_path_for(const std::string &outcome_csv)
{
    std::filesystem::path path(outcome_csv);
    path.replace_extension();
    return path.string() + "_aggregate.csv";
}

inline std::string rho3_path_for(const std::string &outcome_csv)
{
    std::filesystem::path path(outcome_csv);
    path.replace_extension();
    return path.string() + "_rho3.bin";
}

// Ordinary-PPT-but-CCNR-positive triples, in the same record format as the
// companion. Always written, because these are the conventional
// bound-entanglement candidates and they are rare enough that keeping all of
// them costs nothing next to keeping every triple.
inline std::string ppt_ccnr_path_for(const std::string &outcome_csv)
{
    std::filesystem::path path(outcome_csv);
    path.replace_extension();
    return path.string() + "_ppt_ccnr.bin";
}

// Which explanatory classes earn a four-site pass.
//
// "No positive third, and not numerically unresolved" -- a pair with a positive
// third is already explained, and a pair whose intervals straddle the threshold
// is not a residue but an unanswered question, which a larger Hilbert space
// does not answer.
// Stated as the exclusion rather than the list, so a class added later is
// selected by default: the rule is a property of the verdict, not an
// enumeration that can silently fall behind. In particular
// `three_site_all_cuts_entangled_fgmn_zero` -- the bound-entanglement candidate
// -- qualifies: it has no positive third and is fully resolved, which is
// exactly the residue a four-body object would explain.
inline bool four_site_selects(ExplanatoryClass value)
{
    return value != ExplanatoryClass::ThreeSiteFgmn &&
           value != ExplanatoryClass::NumericallyUnresolved;
}

// One four-site row, rendered.
inline void append_four_site_row(std::string &out, const RowPrefix &prefix, const std::string &run,
                                 std::uint32_t realization, std::uint64_t four_id,
                                 const PairSummaryRow &summary, const HelperPair &helper,
                                 const FourResult &result)
{
    // Sorted position of each role, so a cut named "i" really is site i's.
    std::array<int, 4> role{};
    const std::array<int, 4> sites{summary.pair_i, summary.pair_j, helper.k, helper.l};
    for (std::size_t r = 0; r < 4; ++r)
    {
        for (int p = 0; p < 4; ++p)
        {
            if (result.sites[static_cast<std::size_t>(p)] == sites[r])
            {
                role[r] = p;
            }
        }
    }
    const auto &S = result.parts.entropy;
    const int bi = 1 << role[0];
    const int bj = 1 << role[1];
    const int bk = 1 << role[2];
    const int bl = 1 << role[3];

    out += run;
    field(out, static_cast<long long>(realization));
    field(out, static_cast<long long>(four_id));
    field(out, std::string(role_name(static_cast<AnchorRole>(summary.role))));
    field(out, static_cast<long long>(summary.pair_i));
    field(out, static_cast<long long>(summary.pair_j));
    field(out, static_cast<long long>(helper.k));
    field(out, static_cast<long long>(helper.l));
    field_flag(out, helper.control);
    field(out, static_cast<long long>(helper.same_component));
    field(out, static_cast<long long>(helper.on_backbone));
    field(out, static_cast<long long>(helper.distance));
    field(out, static_cast<long long>(summary.separation));
    field(out, summary.chord);
    field(out, std::string(zero_class_name(static_cast<ZeroClass>(summary.zero_class))));
    field(out, std::string(explanatory_class_name(summary.explanation)));
    out += ',';
    out += prefix.provenance;
    out += ',';
    out += prefix.thresholds;

    for (std::size_t r = 0; r < 4; ++r)
    {
        field(out, result.single_cuts[static_cast<std::size_t>(role[r])]);
    }
    field(out, result.min_single_cut);
    // The two-vs-two cut naming: which pair of roles sits together. The stored
    // masks are by sorted position, so the role masks are looked up rather than
    // assumed to line up.
    for (const std::array<int, 2> &together : std::array<std::array<int, 2>, 3>{
             {{0, 1}, {0, 2}, {0, 3}}})
    {
        const int mask = (1 << role[static_cast<std::size_t>(together[0])]) |
                         (1 << role[static_cast<std::size_t>(together[1])]);
        // double_cuts is indexed by the mask containing sorted position 0.
        const std::array<int, 3> masks{0b0011, 0b0101, 0b1001};
        const int canonical = (mask & 1) != 0 ? mask : (15 & ~mask);
        double value = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t d = 0; d < 3; ++d)
        {
            if (masks[d] == canonical)
            {
                value = result.double_cuts[d];
            }
        }
        field(out, value);
    }
    field(out, result.min_double_cut);

    for (const std::array<int, 2> &pair : std::array<std::array<int, 2>, 6>{
             {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}})
    {
        field(out, result.pair_marginal_fn[pair_slot_of(role[static_cast<std::size_t>(pair[0])],
                                                        role[static_cast<std::size_t>(pair[1])])]);
    }
    field(out, result.max_pair_marginal_fn);
    // A triple marginal is named by the role it *drops*, in the order ijk, ijl,
    // ikl, jkl -- so dropping l, k, j, i respectively.
    for (int dropped_role : {3, 2, 1, 0})
    {
        field(out, result.triple_marginal_min_cut[static_cast<std::size_t>(
                       role[static_cast<std::size_t>(dropped_role)])]);
    }
    field(out, result.max_triple_marginal_min_cut);

    field(out, S[bi] + S[bj] - S[bi | bj]);
    field(out, S[bk] + S[bl] - S[bk | bl]);
    field(out, S[bi | bj] + S[bk | bl] - S[15]);
    field(out, S[bi | bk | bl] + S[bj | bk | bl] - S[bk | bl] - S[15]);
    field(out, result.joint_purity);
    for (int bit : {bi, bj, bk, bl})
    {
        field(out, result.parts.purity[static_cast<std::size_t>(bit)]);
    }

    field_flag(out, result.globally_entangled_locally_separable);
    {
        // The helpers are roles 2 and 3, so the assisted entry is the slot for
        // that pair of sorted positions -- the same indexing the marginals use.
        const std::size_t slot = pair_slot_of(role[2], role[3]);
        const analysis::AssistedNegativity &occ = result.assisted_occupation[slot];
        const analysis::AssistedNegativity &par = result.assisted_parity[slot];
        field(out, occ.average);
        field(out, occ.maximum);
        field(out, occ.positive_probability);
        field(out, par.average);
        field(out, par.maximum);
        field(out, par.positive_probability);
        field(out, static_cast<long long>(par.best_outcome));
        field(out, par.unconditional);
    }
    field(out, result.trace_error);
    field(out, result.hermiticity_error);
    field(out, result.min_eigenvalue);
    field(out, result.parity_leakage);
    out += '\n';
}

// Fills `raw` with one raw fermionic 8x8 per triple, interleaved, 128 doubles
// each, in the order given.
using RdmBatch = std::function<void(const std::vector<std::array<int, 3>> &triples,
                                    std::vector<double> &raw)>;

struct TrajectoryStats
{
    std::size_t four_site_rows = 0;
    std::size_t anchors = 0;
    std::size_t controls = 0;
    std::size_t pairs = 0;
    std::size_t rows = 0;
    std::size_t unique_triples = 0;
    std::size_t solves = 0;
    std::size_t failures = 0;
};

class PairGapAnalysis
{
  public:
    // `pair_bins` must outlive this object; they supply the pair-level partition
    // and the reconciliation check. `completed` is the pair checkpoint: this
    // analysis has to cover exactly the trajectories the pair CSV counts.
    PairGapAnalysis(RunConfig config, const std::vector<PairBin> &pair_bins, FgmnSolver solver,
                    int workers, std::uint64_t completed, bool resume)
        : config_(std::move(config)), pair_bins_(pair_bins),
          pool_(workers, std::move(solver), config_.pair_gap.retries),
          prefix_(row_prefix(config_)), run_(run_id(config_))
    {
        settings_ = config_.pair_gap;
        outcome_path_ = settings_.output_path.empty() ? default_gap_output(config_.output_path)
                                                      : settings_.output_path;
        aggregate_path_ = aggregate_path_for(outcome_path_);
        summary_path_ = pair_summary_path_for(outcome_path_);
        rho3_path_ = rho3_path_for(outcome_path_);
        ppt_ccnr_path_ = ppt_ccnr_path_for(outcome_path_);
        four_site_path_ = four_site_path_for(outcome_path_);
        rho4_path_ = rho4_path_for(outcome_path_);
        aggregate_.reset(config_.n);
        completed_ = completed;
        restore(resume);
    }

    const std::string &outcome_path() const { return outcome_path_; }
    const std::string &aggregate_path() const { return aggregate_path_; }
    const std::string &summary_path() const { return summary_path_; }
    const std::string &rho3_path() const { return rho3_path_; }
    std::uint64_t rows() const { return aggregate_.rows(); }
    std::uint64_t solves() const { return solves_; }
    std::uint64_t failed_triples() const { return aggregate_.failed_triples(); }
    const GapAggregate &aggregate() const { return aggregate_; }
    const SolverPool &pool() const { return pool_; }
    std::uint64_t restored_rows() const { return restored_rows_; }
    bool restored_trimmed() const { return restored_trimmed_; }

    // Everything for one trajectory: select, plan, reduce, evaluate, solve,
    // write, absorb -- in that order, and nothing written until everything is
    // solved.
    TrajectoryStats process(std::uint32_t realization, const PairMeasurement &measurement,
                            const ConnectivityIndex &connectivity, const RdmBatch &rdm_batch,
                            const Rdm4Batch &rdm4_batch = Rdm4Batch())
    {
        TrajectoryStats stats;
        const std::vector<Anchor> anchors = select_anchors(measurement, settings_, config_.seed);
        for (const Anchor &anchor : anchors)
        {
            (anchor.role == AnchorRole::Anchor ? stats.anchors : stats.controls) += 1;
        }
        if (!anchors.empty())
        {
            const TriplePlan plan = plan_triples(measurement, anchors);
            stats.unique_triples = plan.triples.size();

            raw_.assign(plan.triples.size() * RHO3_DOUBLES, 0.0);
            rdm_batch(plan.triples, raw_);

            results_.clear();
            results_.reserve(plan.triples.size());
            for (std::size_t t = 0; t < plan.triples.size(); ++t)
            {
                results_.push_back(evaluate_triple(raw_.data() + t * RHO3_DOUBLES,
                                                   plan.triples[t], plan.triple_ids[t],
                                                   measurement, settings_));
            }

            jobs_.clear();
            job_of_.assign(results_.size(), static_cast<std::size_t>(-1));
            for (std::size_t t = 0; t < results_.size(); ++t)
            {
                if (results_[t].method == FgmnMethod::Mosek)
                {
                    job_of_[t] = jobs_.size();
                    jobs_.push_back({results_[t].rho_ri.data(), {}, 0});
                }
            }
            pool_.run(jobs_);
            for (std::size_t t = 0; t < results_.size(); ++t)
            {
                if (job_of_[t] != static_cast<std::size_t>(-1))
                {
                    apply_solve(results_[t], jobs_[job_of_[t]], settings_.positive_tol);
                    stats.failures += results_[t].failed() ? 1u : 0u;
                }
            }
            stats.solves = jobs_.size();
            solves_ += jobs_.size();

            block_.clear();
            rho3_block_.clear();
            ppt_block_.clear();
            summary_block_.clear();
            summaries_.clear();
            for (std::size_t a = 0; a < anchors.size(); ++a)
            {
                const Anchor &anchor = anchors[a];
                const PairSnapshot &pair = measurement.pairs[anchor.pair_index];
                const DisjointPaths paths = connectivity.disjoint_paths(pair.i, pair.j);
                const ComponentAnatomy anatomy =
                    settings_.anatomy ? connectivity.anatomy(pair.i, pair.j) : ComponentAnatomy{};
                const ChannelConnectivity channels =
                    settings_.channels ? connectivity.channels(pair.i, pair.j)
                                       : ChannelConnectivity{};
                summaries_.clear();
                for (const TriplePlan::Third &third : plan.thirds[a])
                {
                    const TripleResult &result = results_[third.triple];
                    append_outcome_row(block_, prefix_, run_, realization, next_outcome_id_,
                                       anchor, pair, measurement, connectivity, paths, anatomy,
                                       channels, third, result);
                    ++next_outcome_id_;
                    RowSummary row;
                    row.realization = realization;
                    row.role = static_cast<int>(anchor.role);
                    row.zero_class = static_cast<int>(pair.zero_class);
                    row.separation = pair.separation;
                    row.pair_i = pair.i;
                    row.pair_j = pair.j;
                    row.third_k = third.k;
                    row.prefiltered = result.method == FgmnMethod::PrefilterBound;
                    row.failed = result.failed();
                    row.positive = result.positive;
                    row.certified = result.certified;
                    row.first_use = third.first_use;
                    row.fgmn = result.fgmn;
                    row.lower_bound = result.lower_bound;
                    row.upper_bound = result.upper_bound;
                    row.min_cut = result.min_cut;
                    {
                        const RolePositions pos = role_positions(result.sites, pair.i, pair.j, third.k);
                        const auto &S = result.parts.entropy;
                        const int bi = 1 << pos.i;
                        const int bj = 1 << pos.j;
                        const int bk = 1 << pos.k;
                        row.cmi_ij_given_k = S[bi | bk] + S[bj | bk] - S[bk] - S[7];
                        row.mi_ij_k = S[bi | bj] + S[bk] - S[7];
                        bool all_cuts = true;
                        for (int p = 0; p < 3; ++p)
                        {
                            const double cut = result.cuts[static_cast<std::size_t>(p)];
                            all_cuts = all_cuts && cut == cut && cut > settings_.cut_positive_tol;
                        }
                        row.all_cuts_entangled = all_cuts;
                        const int p = result.min_cut_position;
                        row.min_cut_role = result.min_cut != result.min_cut
                                               ? -1
                                               : (p == pos.i ? 0 : (p == pos.j ? 1 : 2));
                    }
                    row.same_component =
                        pair.conn.interior_path && connectivity.query(pair.i, third.k).interior_path;
                    row.chord = pair.chord;
                    row.pair_fn = pair.fermion.mn;
                    row.pair_fmi = pair.fermion.mi;
                    row.pair_g2 = pair.fermion.g2;
                    row.pair_f2 = pair.fermion.f2;
                    row.pair_rho_n = pair.fermion.rho_n;
                    row.pair_i_occ = pair.fermion.i_occ;
                    row.component_size = pair.conn.component_size;
                    row.shortest_path = pair.conn.shortest_path;
                    row.edge_disjoint = paths.edge;
                    row.vertex_disjoint = paths.vertex;
                    row.anatomy = anatomy;
                    row.channels = channels;
                    row.survives_i = pair.conn.survives_i;
                    row.survives_j = pair.conn.survives_j;
                    row.interior_path = pair.conn.interior_path;
                    row.connected = pair.conn.connected;
                    summaries_.push_back(row);
                    if (third.first_use && rho3_.active())
                    {
                        append_rho3_record(rho3_block_, realization, result.id, result.sites,
                                           raw_.data() + third.triple * RHO3_DOUBLES);
                    }
                    if (third.first_use && result.separability.qubit_ppt_ccnr_candidate)
                    {
                        append_rho3_record(ppt_block_, realization, result.id, result.sites,
                                           raw_.data() + third.triple * RHO3_DOUBLES);
                        ++ppt_ccnr_candidates_;
                    }
                }
                const PairSummaryRow summary =
                    aggregate_.absorb_pair(summaries_.data(), summaries_.size());
                append_pair_summary_row(summary_block_, prefix_, run_, summary);
                stats.rows += summaries_.size();
                ++stats.pairs;
                if (settings_.four_site && rdm4_batch && four_site_selects(summary.explanation))
                {
                    pending_four_.push_back(summary);
                }
            }
            run_four_site(realization, connectivity, rdm4_batch, stats);
            outcomes_.write(block_.data(), block_.size());
            summaries_file_.write(summary_block_.data(), summary_block_.size());
            four_site_file_.write(four_block_.data(), four_block_.size());
            rho4_.write(rho4_block_.data(), rho4_block_.size());
            rho3_.write(rho3_block_.data(), rho3_block_.size());
            ppt_ccnr_.write(ppt_block_.data(), ppt_block_.size());
        }
        completed_ = static_cast<std::uint64_t>(realization) + 1u;
        return stats;
    }

    // Make every written trajectory durable, then publish the aggregate. The
    // caller publishes the pair CSV only after this returns, which is the
    // ordering the resume relies on: the aggregate's completed count is never
    // behind the pair checkpoint, and the rows behind it are on disk.
    void publish()
    {
        outcomes_.sync();
        summaries_file_.sync();
        rho3_.sync();
        ppt_ccnr_.sync();
        four_site_file_.sync();
        rho4_.sync();
        publish_atomically(aggregate_path_,
                           render_aggregate_csv(config_, aggregate_, pair_bins_, completed_));
    }

    void finish(std::ostream &out)
    {
        publish();
        outcomes_.close();
        summaries_file_.close();
        rho3_.close();
        ppt_ccnr_.close();
        four_site_file_.close();
        rho4_.close();
        out << "Pair-gap triples: " << aggregate_.rows() << " outcome row(s), " << solves_
            << " fGMN solve(s) -> " << outcome_path_ << '\n';
        out << "  per-pair summary -> " << summary_path_ << '\n';
        if (settings_.four_site)
        {
            out << "  four-site analysis: " << four_rows_ << " row(s) -> " << four_site_path_
                << '\n';
            if (four_site_signatures_ > 0)
            {
                out << "  " << four_site_signatures_
                    << " four-mode RDM(s) entangled across every cut with every pair and "
                       "triple marginal separable -- the GHZ-like signature.\n";
            }
        }
        if (ppt_ccnr_candidates_ > 0)
        {
            out << "  " << ppt_ccnr_candidates_
                << " ordinary-PPT but CCNR-positive triple(s) -> " << ppt_ccnr_path_
                << ". Both criteria are one-sided, so these are candidates for offline "
                   "study, not bound-entangled states.\n";
        }
        if (aggregate_.failed_triples() > 0)
        {
            out << "  WARNING: " << aggregate_.failed_triples()
                << " triple(s) still failed after retries. Their rows carry NaN and a solver "
                   "status, their pairs are counted undetermined rather than negative, and "
                   "the aggregate reports run_complete=0.\n";
        }
    }

    void announce(std::ostream &out) const
    {
        out << "pair_gap=" << outcome_path_ << " (class=" << gap_selection_name(settings_.selection)
            << ", controls=" << (settings_.controls ? 1 : 0)
            << ", prefilter_tol=" << settings_.prefilter_tol
            << ", fgmn_positive_tol=" << settings_.positive_tol << ", retries=" << settings_.retries
            << ", solver_workers=" << pool_.workers()
            << ", rho3=" << (settings_.store_rho3 ? rho3_path_ : std::string("off")) << ")\n";
        if (restored_rows_ > 0 || restored_trimmed_)
        {
            out << "  continuing from " << restored_rows_ << " outcome row(s) over " << completed_
                << " trajectory/ies";
            if (restored_trimmed_)
            {
                out << "; rows past the pair checkpoint were discarded";
            }
            out << ".\n";
        }
    }

  private:
    // The four-site pass for one trajectory's selected anchors.
    //
    // It runs after every summary is known, because the selection *is* the
    // summary's verdict, and before anything is written, so a trajectory's
    // four-site rows land in the same durable block as the outcome rows they
    // belong to. That is what makes the resume trim cut both at the same place.
    void run_four_site(std::uint32_t realization, const ConnectivityIndex &connectivity,
                       const Rdm4Batch &rdm4_batch, TrajectoryStats &stats)
    {
        four_block_.clear();
        rho4_block_.clear();
        if (pending_four_.empty())
        {
            return;
        }
        quads_.clear();
        helpers_.clear();
        owner_.clear();
        std::vector<std::uint8_t> same_component;
        std::vector<std::uint8_t> on_backbone;
        std::vector<std::int32_t> distance;
        for (std::size_t a = 0; a < pending_four_.size(); ++a)
        {
            const PairSummaryRow &summary = pending_four_[a];
            connectivity.site_roles(summary.pair_i, summary.pair_j, same_component, on_backbone,
                                    distance);
            // Seeded on the run, the trajectory and the pair, so a pair's
            // helper set is the same however the run was scheduled or resumed.
            const std::uint64_t seed = seeding::splitmix64(
                config_.seed ^ (static_cast<std::uint64_t>(realization) << 20) ^
                (static_cast<std::uint64_t>(summary.pair_i) << 8) ^
                static_cast<std::uint64_t>(summary.pair_j));
            const std::vector<HelperPair> chosen = select_helper_pairs(
                config_.n, summary.pair_i, summary.pair_j, same_component, on_backbone, distance,
                settings_.four_max_helper_pairs, seed);
            for (const HelperPair &helper : chosen)
            {
                std::array<int, 4> quad{summary.pair_i, summary.pair_j, helper.k, helper.l};
                std::sort(quad.begin(), quad.end());
                quads_.push_back(quad);
                helpers_.push_back(helper);
                owner_.push_back(a);
            }
        }
        if (quads_.empty())
        {
            return;
        }
        raw4_.assign(quads_.size() * RHO4_DOUBLES, 0.0);
        rdm4_batch(quads_, raw4_);
        for (std::size_t q = 0; q < quads_.size(); ++q)
        {
            const FourResult result =
                evaluate_four(raw4_.data() + q * RHO4_DOUBLES, quads_[q],
                              settings_.cut_positive_tol, settings_.assisted,
                              settings_.assisted_tol);
            append_four_site_row(four_block_, prefix_, run_, realization, next_four_id_,
                                 pending_four_[owner_[q]], helpers_[q], result);
            ++next_four_id_;
            ++stats.four_site_rows;
            four_rows_ += 1u;
            if (result.globally_entangled_locally_separable)
            {
                ++four_site_signatures_;
            }
            if (rho4_.active())
            {
                append_rho4_record(rho4_block_, realization, quads_[q],
                                   raw4_.data() + q * RHO4_DOUBLES);
            }
        }
        pending_four_.clear();
    }

    static void append_rho4_record(std::string &out, std::uint32_t realization,
                                   const std::array<int, 4> &sites, const double *raw)
    {
        char head[16] = {};
        std::memcpy(head, &realization, 4);
        for (int s = 0; s < 4; ++s)
        {
            head[8 + s] = static_cast<char>(sites[static_cast<std::size_t>(s)]);
        }
        out.append(head, 16);
        out.append(reinterpret_cast<const char *>(raw), RHO4_DOUBLES * sizeof(double));
    }

    void restore(bool resume)
    {
        namespace fs = std::filesystem;
        std::error_code error;
        const bool outcome_exists = fs::exists(outcome_path_, error);
        const bool aggregate_exists = fs::exists(aggregate_path_, error);
        const std::string rho3_header = rho3_header_text(config_);

        if (!resume)
        {
            // MIPT_DIST_RESUME=0: the pair CSV starts over, so this does too.
            completed_ = 0;
            fs::remove(aggregate_path_, error);
            start_fresh(rho3_header);
            return;
        }

        std::uint64_t aggregate_completed = 0;
        if (aggregate_exists)
        {
            const auto table = util::resume::CsvTable::read(aggregate_path_);
            std::string expected = aggregate_csv_header(config_);
            expected.pop_back();
            if (!table.empty() && table.header_line() != expected)
            {
                util::resume::refuse(aggregate_path_ + " has a different column layout than "
                                                       "this run writes.");
            }
            if (!table.empty())
            {
                verify_aggregate_identity(table);
                aggregate_completed = table.counter(0, "completed_realizations");
            }
        }

        // The ordering guarantee, checked. The aggregate is published before the
        // pair CSV every time, so it can be ahead but never behind; behind means
        // the gap analysis did not cover every trajectory the pair CSV counts.
        if (aggregate_completed < completed_)
        {
            util::resume::refuse(
                "the pair checkpoint counts " + std::to_string(completed_) +
                " trajectories but the pair-gap aggregate " +
                (aggregate_exists ? aggregate_path_ : std::string("(absent)")) + " covers " +
                std::to_string(aggregate_completed) +
                ". The triple analysis must cover every trajectory the pair CSV does, so it "
                "cannot be switched on partway through a run. Start it with a fresh pair "
                "CSV, or set MIPT_DIST_RESUME=0.");
        }

        if (!outcome_exists)
        {
            if (completed_ > 0)
            {
                util::resume::refuse(outcome_path_ + " is missing, but the pair checkpoint "
                                                      "counts " +
                                     std::to_string(completed_) +
                                     " trajectories whose triple outcomes it should hold.");
            }
            start_fresh(rho3_header);
            return;
        }

        // The summary is a pure function of the kept rows, so it is rewritten
        // from them rather than trimmed: a summary row and the N-2 outcome rows
        // behind it then cannot disagree, whatever the kill interrupted.
        std::string rebuilt = pair_summary_csv_header();
        const OutcomeScan scan =
            scan_outcomes(outcome_path_, completed_, aggregate_, settings_,
                          [&](const PairSummaryRow &summary) {
                              append_pair_summary_row(rebuilt, prefix_, run_, summary);
                          });
        restored_rows_ = scan.rows_kept;
        restored_trimmed_ = scan.trimmed;
        next_outcome_id_ = scan.rows_kept;
        fs::resize_file(outcome_path_, scan.bytes_kept, error);
        if (error)
        {
            throw std::runtime_error("Could not trim " + outcome_path_ + ": " + error.message());
        }
        reconcile();
        publish_atomically(summary_path_, rebuilt);
        outcomes_.open(outcome_path_);
        summaries_file_.open(summary_path_);

        if (settings_.four_site)
        {
            // A row-per-(pair, helper) CSV in trajectory order, so the trim is
            // a line scan: keep every row whose realization is below the pair
            // checkpoint. Far fewer rows than the outcome file, so a scan is
            // cheaper than maintaining an index for it.
            if (fs::exists(four_site_path_, error))
            {
                trim_four_site(four_site_path_, completed_, next_four_id_);
            }
            else
            {
                create_four_site_header();
            }
            four_site_file_.open(four_site_path_);
            if (settings_.store_rho4)
            {
                if (fs::exists(rho4_path_, error))
                {
                    trim_rho4(rho4_path_, rho3_header, completed_);
                }
                else
                {
                    write_rho3_header(rho4_path_, rho3_header);
                }
                rho4_.open(rho4_path_);
            }
        }

        // The candidate companion is always written, so it is always trimmed.
        // Missing is not an error: it is rare for one to exist at all, and a
        // run with no candidates yet legitimately has only a header.
        if (fs::exists(ppt_ccnr_path_, error))
        {
            trim_rho3(ppt_ccnr_path_, rho3_header, completed_);
        }
        else
        {
            write_rho3_header(ppt_ccnr_path_, rho3_header);
        }
        ppt_ccnr_.open(ppt_ccnr_path_);

        if (settings_.store_rho3)
        {
            if (fs::exists(rho3_path_, error))
            {
                trim_rho3(rho3_path_, rho3_header, completed_);
            }
            else if (completed_ > 0)
            {
                util::resume::refuse(
                    rho3_path_ + " is missing, but MIPT_DIST_PAIR_GAP_STORE_RHO3=1 and the run "
                                 "has already completed " +
                    std::to_string(completed_) +
                    " trajectories. The companion has to hold every unique triple from the "
                    "start; set MIPT_DIST_PAIR_GAP_STORE_RHO3=0 to continue without it.");
            }
            else
            {
                write_rho3_header(rho3_header);
            }
            rho3_.open(rho3_path_);
        }
    }

    void start_fresh(const std::string &rho3_header)
    {
        ensure_output_parent_directory(outcome_path_);
        auto create_with_header = [](const std::string &path, const std::string &header) {
            std::FILE *file = std::fopen(path.c_str(), "wb");
            if (file == nullptr)
            {
                throw std::runtime_error("Could not create " + path + ".");
            }
            std::fwrite(header.data(), 1, header.size(), file);
            std::fclose(file);
        };
        create_with_header(outcome_path_, outcome_csv_header());
        create_with_header(summary_path_, pair_summary_csv_header());
        outcomes_.open(outcome_path_);
        summaries_file_.open(summary_path_);
        write_rho3_header(ppt_ccnr_path_, rho3_header);
        ppt_ccnr_.open(ppt_ccnr_path_);
        if (settings_.store_rho3)
        {
            write_rho3_header(rho3_header);
            rho3_.open(rho3_path_);
        }
        if (settings_.four_site)
        {
            create_four_site_header();
            four_site_file_.open(four_site_path_);
            if (settings_.store_rho4)
            {
                write_rho3_header(rho4_path_, rho3_header);
                rho4_.open(rho4_path_);
            }
        }
    }

    void create_four_site_header()
    {
        std::FILE *file = std::fopen(four_site_path_.c_str(), "wb");
        if (file == nullptr)
        {
            throw std::runtime_error("Could not create " + four_site_path_ + ".");
        }
        const std::string header = four_site_csv_header();
        std::fwrite(header.data(), 1, header.size(), file);
        std::fclose(file);
    }

    // Keep the prefix whose realization is below `completed`, and report how
    // many rows survived so four_id stays contiguous across the restart.
    static void trim_four_site(const std::string &path, std::uint64_t completed,
                               std::uint64_t &rows_kept)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Could not open " + path + " for reading.");
        }
        std::string line;
        if (!std::getline(file, line) || line + '\n' != four_site_csv_header())
        {
            util::resume::refuse(path + " has a different column layout than this binary "
                                        "writes. Move it aside, or set MIPT_DIST_RESUME=0.");
        }
        const auto &columns = four_site_columns();
        const std::size_t realization_column = static_cast<std::size_t>(
            std::find(columns.begin(), columns.end(), "realization_id") - columns.begin());
        std::uint64_t bytes = four_site_csv_header().size();
        rows_kept = 0;
        while (true)
        {
            const std::streampos start = file.tellg();
            if (!std::getline(file, line) || file.eof())
            {
                break;
            }
            const std::vector<std::string> fields = util::resume::split_csv_row(line);
            if (fields.size() != columns.size())
            {
                break;
            }
            if (static_cast<std::uint64_t>(std::stoul(fields[realization_column])) >= completed)
            {
                break;
            }
            ++rows_kept;
            bytes = static_cast<std::uint64_t>(start) + line.size() + 1u;
        }
        file.close();
        std::error_code error;
        std::filesystem::resize_file(path, bytes, error);
        if (error)
        {
            throw std::runtime_error("Could not trim " + path + ": " + error.message());
        }
    }

    static void trim_rho4(const std::string &path, const std::string &header,
                          std::uint64_t completed)
    {
        // Same fixed-width layout as the triple companion, only wider.
        std::ifstream file(path, std::ios::binary);
        char preamble[RHO3_PREAMBLE];
        file.read(preamble, RHO3_PREAMBLE);
        if (!file || std::memcmp(preamble, RHO3_MAGIC, 8) != 0)
        {
            util::resume::refuse(path + " is not a pair-gap RDM companion.");
        }
        std::uint32_t header_bytes = 0;
        std::memcpy(&header_bytes, preamble + 16, 4);
        std::string stored(header_bytes, '\0');
        file.read(stored.data(), header_bytes);
        if (stored != header)
        {
            util::resume::refuse(path + " describes a different run.");
        }
        constexpr std::size_t record = 16 + RHO4_DOUBLES * sizeof(double);
        const std::uint64_t offset = RHO3_PREAMBLE + header_bytes;
        const std::uint64_t size = std::filesystem::file_size(path);
        const std::uint64_t count = size > offset ? (size - offset) / record : 0;
        std::uint64_t keep = 0;
        for (std::uint64_t index = 0; index < count; ++index)
        {
            file.clear();
            file.seekg(static_cast<std::streamoff>(offset + index * record));
            std::uint32_t value = 0;
            file.read(reinterpret_cast<char *>(&value), 4);
            if (value >= completed)
            {
                break;
            }
            keep = index + 1u;
        }
        file.close();
        std::filesystem::resize_file(path, offset + keep * record);
    }

    void write_rho3_header(const std::string &header) { write_rho3_header(rho3_path_, header); }

    void write_rho3_header(const std::string &path, const std::string &header)
    {
        std::FILE *file = std::fopen(path.c_str(), "wb");
        if (file == nullptr)
        {
            throw std::runtime_error("Could not create " + path + ".");
        }
        const std::string bytes = rho3_preamble(header);
        std::fwrite(bytes.data(), 1, bytes.size(), file);
        std::fclose(file);
    }

    // A checkpoint written under different thresholds classifies and solves
    // the same triples differently, so its rows cannot be continued.
    void verify_aggregate_identity(const util::resume::CsvTable &table)
    {
        const std::string &path = aggregate_path_;
        auto same_text = [&](const char *name, const std::string &expected) {
            if (table.text(0, name) != expected)
            {
                util::resume::refuse(path + " was produced with " + name + "=" +
                                     table.text(0, name) + ", but this run uses " + expected +
                                     ". Move it aside, or set MIPT_DIST_RESUME=0.");
            }
        };
        auto number = [](double value) {
            std::string text;
            append_double(text, value);
            return text;
        };
        same_text("pair_zero_tol", number(config_.pair_zero_tol));
        same_text("occupation_mi_tol", number(config_.occupation_mi_tol));
        same_text("channel_floor", number(config_.channel_floor));
        same_text("fgmn_positive_tol", number(settings_.positive_tol));
        same_text("prefilter_tol", number(settings_.prefilter_tol));
        same_text("cut_positive_tol", number(settings_.cut_positive_tol));
        same_text("correlation_tol", number(settings_.correlation_tol));
        same_text("gap_selection", gap_selection_name(settings_.selection));
        same_text("controls", settings_.controls ? "1" : "0");
        same_text("master_seed", std::to_string(config_.seed));
        same_text("statevector_precision", std::to_string(config_.statevector_precision));
    }

    // The rebuilt anchors must be exactly the pair bins' connected-zero
    // records, class by class and separation by separation. Both count the same
    // trajectories under the same classification, so any disagreement is a
    // damaged or mismatched file, and it is refused rather than reported.
    void reconcile() const
    {
        for (int separation = 1; separation <= aggregate_.max_separation(); ++separation)
        {
            const PairBin &bin = pair_bins_[static_cast<std::size_t>(separation - 1)];
            const std::array<std::uint64_t, ZERO_CLASS_COUNT> expected{
                bin.classical_occ_correlated, bin.classical_subthreshold, bin.classical_silent};
            for (int c = 0; c < ZERO_CLASS_COUNT; ++c)
            {
                if (!gap_selects(settings_.selection, static_cast<ZeroClass>(c)))
                {
                    continue;
                }
                const std::uint64_t found = aggregate_.cell(0, c, separation).qualifying_pairs;
                if (found != expected[static_cast<std::size_t>(c)])
                {
                    util::resume::refuse(
                        outcome_path_ + " holds " + std::to_string(found) + " " +
                        zero_class_name(static_cast<ZeroClass>(c)) + " anchor pair(s) at "
                        "separation " + std::to_string(separation) + ", but the pair "
                        "checkpoint counts " + std::to_string(expected[static_cast<std::size_t>(c)]) +
                        " over the same trajectories. The two files do not describe the same "
                        "run.");
                }
            }
        }
    }

    RunConfig config_;
    PairGapSettings settings_;
    const std::vector<PairBin> &pair_bins_;
    SolverPool pool_;
    RowPrefix prefix_;
    std::string run_;
    std::string outcome_path_;
    std::string aggregate_path_;
    std::string summary_path_;
    std::string rho3_path_;
    std::string ppt_ccnr_path_;
    std::string four_site_path_;
    std::string rho4_path_;
    GapAggregate aggregate_;
    DurableAppender outcomes_;
    DurableAppender summaries_file_;
    DurableAppender rho3_;
    DurableAppender ppt_ccnr_;
    DurableAppender four_site_file_;
    DurableAppender rho4_;
    std::uint64_t completed_ = 0;
    std::uint64_t next_outcome_id_ = 0;
    std::uint64_t solves_ = 0;
    std::uint64_t restored_rows_ = 0;
    std::uint64_t ppt_ccnr_candidates_ = 0;
    std::uint64_t next_four_id_ = 0;
    std::uint64_t four_rows_ = 0;
    std::uint64_t four_site_signatures_ = 0;
    bool restored_trimmed_ = false;

    std::vector<double> raw_;
    std::vector<TripleResult> results_;
    std::vector<SolveJob> jobs_;
    std::vector<std::size_t> job_of_;
    std::vector<RowSummary> summaries_;
    std::string block_;
    std::string summary_block_;
    std::string rho3_block_;
    std::string ppt_block_;
    std::string four_block_;
    std::string rho4_block_;
    std::vector<PairSummaryRow> pending_four_;
    std::vector<std::array<int, 4>> quads_;
    std::vector<HelperPair> helpers_;
    std::vector<std::size_t> owner_;
    std::vector<double> raw4_;
};

} // namespace mipt::dist::gap
