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

inline constexpr int GAP_FORMAT_VERSION = 1;
// FGMN_STATUS_* are the solver's own codes; this one is ours.
inline constexpr int STATUS_PREFILTERED = -1;
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

    FgmnMethod method = FgmnMethod::PrefilterBound;
    double fgmn_raw = std::numeric_limits<double>::quiet_NaN();
    double fgmn = std::numeric_limits<double>::quiet_NaN();
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

    // The prefilter decision. A NaN minimum is never prefiltered: without a
    // converged bound there is nothing to certify, so the SDP decides.
    if (out.min_cut == out.min_cut && out.min_cut <= settings.prefilter_tol)
    {
        out.method = FgmnMethod::PrefilterBound;
        out.status = STATUS_PREFILTERED;
        out.fgmn_raw = std::numeric_limits<double>::quiet_NaN();
        out.fgmn = 0.0;
        out.positive = false;
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
    int status = FGMN_STATUS_EXCEPTION;
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
    result.attempts = job.attempts;
    result.status = job.outcome.status;
    result.fgmn_raw = job.outcome.value;
    if (result.status == FGMN_STATUS_OK && std::isfinite(result.fgmn_raw))
    {
        // The SDP optimum can sit a hair below zero on a separable state; the
        // raw value keeps that, the classified one clamps it.
        result.fgmn = std::max(0.0, result.fgmn_raw);
        result.positive = result.fgmn > positive_tol;
    }
    else
    {
        result.fgmn = std::numeric_limits<double>::quiet_NaN();
        result.positive = false;
    }
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
            "prefilter_tol", "mosek_tol",
            // anchor pair
            "separation", "chord", "pair_class", "pair_fn", "pair_fn_generic", "pair_fmi", "pair_g2",
            "pair_f2", "pair_re_g", "pair_im_g", "pair_re_f", "pair_im_f", "pair_n_i", "pair_n_j",
            "pair_dnn", "pair_rho_n", "pair_abs_rho_n", "pair_rho_n_sq", "pair_i_occ",
            "pair_parity_leakage",
            // anchor pair graph
            "survives_i", "survives_j", "interior_path_ij", "connected_ij", "component_size_ij",
            "shortest_path_ij", "idle_i", "idle_j", "edge_disjoint_paths_ij",
            "vertex_disjoint_paths_ij",
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
              // fGMN
              "fgmn_method", "fgmn_raw", "fgmn_upper_bound", "fgmn", "fgmn_positive", "fgmn_status",
              "fgmn_attempts",
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
                               const TriplePlan::Third &third, const TripleResult &result)
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

    field(out, std::string(method_name(result.method)));
    field(out, result.fgmn_raw);
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

struct RowSummary
{
    std::uint32_t realization = 0;
    int role = 0;
    int zero_class = 0;
    int separation = 0;
    int pair_i = 0;
    int pair_j = 0;
    bool prefiltered = false;
    bool failed = false;
    bool positive = false;
    bool first_use = false;
    double fgmn = std::numeric_limits<double>::quiet_NaN();
};

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

    // Every row of one (trajectory, role, pair), in k order.
    void absorb_pair(const RowSummary *rows, std::size_t count)
    {
        if (count == 0)
        {
            return;
        }
        const RowSummary &head = rows[0];
        GapCell &target = cell(head.role, head.zero_class, head.separation);
        ++target.qualifying_pairs;
        std::size_t positives = 0;
        std::size_t failures = 0;
        double largest = -std::numeric_limits<double>::infinity();
        double sum = 0.0;
        std::size_t valued = 0;
        for (std::size_t r = 0; r < count; ++r)
        {
            const RowSummary &row = rows[r];
            ++target.outcomes;
            ++rows_;
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
                ++failures;
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
        else if (failures > 0)
        {
            ++target.pairs_undetermined;
        }
        target.positive_third_hist[std::min(positives, target.positive_third_hist.size() - 1u)] += 1u;
        if (valued > 0)
        {
            target.max_fgmn_over_k.add(largest);
            target.mean_fgmn_over_k.add(sum / static_cast<double>(valued));
        }
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
    header += ",fgmn_positive_tol,prefilter_tol,mosek_tol,gap_selection,controls,"
              "gap_format_version,completed_realizations,run_complete,unresolved_failed_triples,"
              "anchor_role,pair_class,separation,d,"
              "qualifying_pairs,outcomes,outcomes_prefiltered,outcomes_mosek,outcomes_positive,"
              "outcomes_failed,pairs_with_positive_third,pairs_undetermined,r3_fraction";
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
                field(line, static_cast<long long>(cell.pairs_with_positive_third));
                field(line, static_cast<long long>(cell.pairs_undetermined));
                field(line, positive_fraction(cell.pairs_with_positive_third, cell.qualifying_pairs));
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
        if (bytes == 0 || file_ == nullptr)
        {
            return;
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
    bool trimmed = false;
};

// Stream the outcome CSV, rebuilding the aggregate from every row whose
// realization is below `completed`, and report where the kept prefix ends.
//
// This is the only place the aggregate comes back from, and it is not a
// shortcut: the aggregate CSV is published at progress intervals, so after a
// kill it can hold trajectories the pair checkpoint does not. The outcome rows
// are the record of what was measured; the aggregate is recomputed from them,
// which is also what makes the two reconcile exactly.
inline OutcomeScan scan_outcomes(const std::string &path, std::uint64_t completed,
                                 GapAggregate &aggregate)
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
    auto column = [&](const char *name) {
        const auto it = std::find(columns.begin(), columns.end(), name);
        return static_cast<std::size_t>(it - columns.begin());
    };
    const std::size_t c_realization = column("realization_id");
    const std::size_t c_outcome = column("outcome_id");
    const std::size_t c_role = column("anchor_role");
    const std::size_t c_i = column("pair_i");
    const std::size_t c_j = column("pair_j");
    const std::size_t c_separation = column("separation");
    const std::size_t c_class = column("pair_class");
    const std::size_t c_method = column("fgmn_method");
    const std::size_t c_fgmn = column("fgmn");
    const std::size_t c_positive = column("fgmn_positive");
    const std::size_t c_status = column("fgmn_status");
    const std::size_t c_first = column("triple_first_use");

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
            aggregate.absorb_pair(group.data(), group.size());
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
            static_cast<std::uint32_t>(parse_int(fields[c_realization]));
        if (realization < last_realization)
        {
            util::resume::refuse(path + " is not in trajectory order at outcome " +
                                 fields[c_outcome] + ".");
        }
        if (realization >= completed)
        {
            scan.trimmed = true;
            break;
        }
        if (static_cast<std::uint64_t>(parse_int(fields[c_outcome])) != scan.rows_kept)
        {
            util::resume::refuse(path + " skips or repeats an outcome_id near row " +
                                 std::to_string(scan.rows_kept + 2) + ".");
        }
        RowSummary row;
        row.realization = realization;
        row.role = fields[c_role] == "control" ? 1 : 0;
        row.zero_class = class_of(fields[c_class]);
        row.separation = static_cast<int>(parse_int(fields[c_separation]));
        row.pair_i = static_cast<int>(parse_int(fields[c_i]));
        row.pair_j = static_cast<int>(parse_int(fields[c_j]));
        row.prefiltered = fields[c_method] == "prefilter_bound";
        row.failed = !row.prefiltered && fields[c_status] != "ok";
        row.positive = fields[c_positive] == "1";
        row.first_use = fields[c_first] == "1";
        row.fgmn = util::resume::parse_double_field(fields[c_fgmn], path + " fgmn");

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

// Fills `raw` with one raw fermionic 8x8 per triple, interleaved, 128 doubles
// each, in the order given.
using RdmBatch = std::function<void(const std::vector<std::array<int, 3>> &triples,
                                    std::vector<double> &raw)>;

struct TrajectoryStats
{
    std::size_t anchors = 0;
    std::size_t controls = 0;
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
        rho3_path_ = rho3_path_for(outcome_path_);
        aggregate_.reset(config_.n);
        completed_ = completed;
        restore(resume);
    }

    const std::string &outcome_path() const { return outcome_path_; }
    const std::string &aggregate_path() const { return aggregate_path_; }
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
                            const ConnectivityIndex &connectivity, const RdmBatch &rdm_batch)
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
            summaries_.clear();
            for (std::size_t a = 0; a < anchors.size(); ++a)
            {
                const Anchor &anchor = anchors[a];
                const PairSnapshot &pair = measurement.pairs[anchor.pair_index];
                const DisjointPaths paths = connectivity.disjoint_paths(pair.i, pair.j);
                summaries_.clear();
                for (const TriplePlan::Third &third : plan.thirds[a])
                {
                    const TripleResult &result = results_[third.triple];
                    append_outcome_row(block_, prefix_, run_, realization, next_outcome_id_,
                                       anchor, pair, measurement, connectivity, paths, third, result);
                    ++next_outcome_id_;
                    RowSummary row;
                    row.realization = realization;
                    row.role = static_cast<int>(anchor.role);
                    row.zero_class = static_cast<int>(pair.zero_class);
                    row.separation = pair.separation;
                    row.pair_i = pair.i;
                    row.pair_j = pair.j;
                    row.prefiltered = result.method == FgmnMethod::PrefilterBound;
                    row.failed = result.failed();
                    row.positive = result.positive;
                    row.first_use = third.first_use;
                    row.fgmn = result.fgmn;
                    summaries_.push_back(row);
                    if (third.first_use && rho3_.active())
                    {
                        append_rho3_record(rho3_block_, realization, result.id, result.sites,
                                           raw_.data() + third.triple * RHO3_DOUBLES);
                    }
                }
                aggregate_.absorb_pair(summaries_.data(), summaries_.size());
                stats.rows += summaries_.size();
            }
            outcomes_.write(block_.data(), block_.size());
            rho3_.write(rho3_block_.data(), rho3_block_.size());
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
        rho3_.sync();
        publish_atomically(aggregate_path_,
                           render_aggregate_csv(config_, aggregate_, pair_bins_, completed_));
    }

    void finish(std::ostream &out)
    {
        publish();
        outcomes_.close();
        rho3_.close();
        out << "Pair-gap triples: " << aggregate_.rows() << " outcome row(s), " << solves_
            << " fGMN solve(s) -> " << outcome_path_ << '\n';
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

        const OutcomeScan scan = scan_outcomes(outcome_path_, completed_, aggregate_);
        restored_rows_ = scan.rows_kept;
        restored_trimmed_ = scan.trimmed;
        next_outcome_id_ = scan.rows_kept;
        fs::resize_file(outcome_path_, scan.bytes_kept, error);
        if (error)
        {
            throw std::runtime_error("Could not trim " + outcome_path_ + ": " + error.message());
        }
        reconcile();
        outcomes_.open(outcome_path_);

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
        {
            std::FILE *file = std::fopen(outcome_path_.c_str(), "wb");
            if (file == nullptr)
            {
                throw std::runtime_error("Could not create " + outcome_path_ + ".");
            }
            const std::string header = outcome_csv_header();
            std::fwrite(header.data(), 1, header.size(), file);
            std::fclose(file);
        }
        outcomes_.open(outcome_path_);
        if (settings_.store_rho3)
        {
            write_rho3_header(rho3_header);
            rho3_.open(rho3_path_);
        }
    }

    void write_rho3_header(const std::string &header)
    {
        std::FILE *file = std::fopen(rho3_path_.c_str(), "wb");
        if (file == nullptr)
        {
            throw std::runtime_error("Could not create " + rho3_path_ + ".");
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
    std::string rho3_path_;
    GapAggregate aggregate_;
    DurableAppender outcomes_;
    DurableAppender rho3_;
    std::uint64_t completed_ = 0;
    std::uint64_t next_outcome_id_ = 0;
    std::uint64_t solves_ = 0;
    std::uint64_t restored_rows_ = 0;
    bool restored_trimmed_ = false;

    std::vector<double> raw_;
    std::vector<TripleResult> results_;
    std::vector<SolveJob> jobs_;
    std::vector<std::size_t> job_of_;
    std::vector<RowSummary> summaries_;
    std::string block_;
    std::string rho3_block_;
};

} // namespace mipt::dist::gap
