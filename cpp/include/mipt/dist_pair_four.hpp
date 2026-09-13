#pragma once

// Four-site analysis for the residue of the connected-zero triple analysis.
// --------------------------------------------------------------------------
//
// **The question.** A four-mode GHZ-like state has vanishing pair negativity
// *and* fully separable three-mode marginals, while being globally entangled.
// So a pair whose every third site came back certified-small is not thereby
// explained: the correlation could live in a four-body object that no
// three-site marginal sees. This adds, for exactly those pairs, a fourth site
// pair and measures the four-mode RDM.
//
// **Which pairs.** Only those whose per-pair summary says *no positive third
// and not numerically unresolved* -- the `three_site_separable_cut`,
// `no_same_component_third` and `higher_order_or_graph` classes. A pair with a
// positive third is already explained; a pair that is unresolved is not a
// residue, it is an unanswered question, and adding a bigger Hilbert space to
// an unresolved measurement makes it no better. This gate is the whole reason
// the four-site work is affordable.
//
// **What is measured.** No genuine-multipartite measure -- that is deliberately
// out of scope. What is computed is the full cut structure:
//
//   * four one-versus-three fermionic cut negativities;
//   * three two-versus-two fermionic cut negativities;
//   * the six pair marginal fNs and the four triple marginals' minimum cuts;
//   * subsystem mutual informations, conditional mutual informations, and the
//     joint and marginal purities.
//
// A four-mode GHZ has every one of those seven cuts positive while every pair
// marginal and every triple marginal is separable, which is the signature the
// residue is being tested against.
//
// **Helper selection is deterministic**, and that matters: a pair's four-site
// verdict must not depend on when it was analysed. Candidates are ranked by
// graph relevance -- same component first, then on the endpoint backbone, then
// closest to it -- with ties broken by site index, and a reproducibly seeded
// sample of the remainder as a control group. Nothing about a candidate's
// *value* enters the ranking, which is what keeps the control a control.

#include "mipt/analysis/assisted.hpp"
#include "mipt/analysis/cut_negativity.hpp"
#include "mipt/dist_connectivity.hpp"
#include "mipt/dist_metrics.hpp"
#include "mipt/dist_scaling_csv.hpp"
#include "mipt/small_rdm.hpp"
#include "mipt/types.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace mipt::dist::gap
{

inline constexpr std::size_t RHO4_DOUBLES = 512; // 16x16 interleaved complex

// All sixteen subsystem entropies and purities of a four-mode state, indexed by
// the retained bit mask. Entropies in bits, matching every other entropy this
// executable reports.
struct FourEntropies
{
    std::array<double, 16> entropy{};
    std::array<double, 16> purity{};
};

inline FourEntropies four_party_entropies(const ancilla::SmallRdm &rho, bool fermionic)
{
    FourEntropies out;
    for (unsigned mask = 1; mask < 16u; ++mask)
    {
        const ancilla::SmallRdm marginal = ancilla::partial_trace_fixed(rho, 4, mask, fermionic);
        out.entropy[mask] = ancilla::entropy_from_small_rdm(marginal);
        out.purity[mask] = ancilla::purity_from_small_rdm(marginal);
    }
    return out;
}

// One helper pair chosen for one anchor.
struct HelperPair
{
    int k = 0;
    int l = 0;
    bool control = false; // drawn from the reproducible sample rather than ranked
    // Why it was ranked where it was, carried onto the row so a selection bias
    // can be looked for rather than assumed absent.
    int same_component = 0; // how many of k, l share the anchor's component
    int on_backbone = 0;    // how many sit on the endpoint backbone
    int distance = 0;       // summed graph distance from k and l to i and j
};

// Rank every eligible {k, l} and keep the best `limit`, with a seeded sample of
// the rest as controls.
//
// `backbone` is per-site: 1 where that site's final node lies on the anchor's
// endpoint backbone. `distance_to` gives the graph distance from a site to the
// anchor's endpoints, or -1 when there is none.
inline std::vector<HelperPair> select_helper_pairs(
    int n, int i, int j, const std::vector<std::uint8_t> &same_component,
    const std::vector<std::uint8_t> &backbone, const std::vector<std::int32_t> &distance_to,
    int limit, std::uint64_t seed)
{
    std::vector<HelperPair> ranked;
    for (int k = 0; k < n; ++k)
    {
        if (k == i || k == j)
        {
            continue;
        }
        for (int l = k + 1; l < n; ++l)
        {
            if (l == i || l == j)
            {
                continue;
            }
            HelperPair candidate;
            candidate.k = k;
            candidate.l = l;
            const std::size_t sk = static_cast<std::size_t>(k);
            const std::size_t sl = static_cast<std::size_t>(l);
            candidate.same_component = (same_component[sk] != 0u ? 1 : 0) +
                                       (same_component[sl] != 0u ? 1 : 0);
            candidate.on_backbone =
                (backbone[sk] != 0u ? 1 : 0) + (backbone[sl] != 0u ? 1 : 0);
            const std::int32_t dk = distance_to[sk];
            const std::int32_t dl = distance_to[sl];
            // An unreachable site sorts last rather than first, which a plain
            // -1 would do.
            candidate.distance = (dk < 0 ? 1000 : dk) + (dl < 0 ? 1000 : dl);
            ranked.push_back(candidate);
        }
    }
    if (limit <= 0 || ranked.empty())
    {
        return {};
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const HelperPair &a, const HelperPair &b) {
        if (a.same_component != b.same_component) return a.same_component > b.same_component;
        if (a.on_backbone != b.on_backbone) return a.on_backbone > b.on_backbone;
        if (a.distance != b.distance) return a.distance < b.distance;
        if (a.k != b.k) return a.k < b.k;
        return a.l < b.l;
    });

    // A quarter of the budget, at least one slot when there is room, goes to a
    // seeded draw from everything the ranking did not take. Without it the
    // four-site sample is entirely the pairs the graph already likes, and a
    // positive result there would have no null to be read against.
    const std::size_t budget = static_cast<std::size_t>(limit);
    const std::size_t keep = std::min(budget, ranked.size());
    std::size_t controls = keep < ranked.size() ? std::max<std::size_t>(1u, budget / 4u) : 0u;
    controls = std::min(controls, keep);
    const std::size_t ranked_slots = keep - controls;

    std::vector<HelperPair> out(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(ranked_slots));
    if (controls > 0)
    {
        std::vector<HelperPair> remainder(ranked.begin() + static_cast<std::ptrdiff_t>(ranked_slots),
                                          ranked.end());
        std::mt19937_64 rng(seed);
        std::shuffle(remainder.begin(), remainder.end(), rng);
        for (std::size_t c = 0; c < controls && c < remainder.size(); ++c)
        {
            HelperPair chosen = remainder[c];
            chosen.control = true;
            out.push_back(chosen);
        }
    }
    return out;
}

// Everything measured on one four-mode RDM. Positions are the *sorted* site
// order, as the reducer returns them; the row writer maps them onto the roles
// i, j, k, l.
struct FourResult
{
    std::array<int, 4> sites{};
    std::array<double, RHO4_DOUBLES> rho_ri{};
    FourEntropies parts;
    // Cuts by the party cut off, in sorted position.
    std::array<double, 4> single_cuts{};
    // Two-versus-two cuts, by the mask of the smaller side containing position
    // 0: {0,1}|{2,3}, {0,2}|{1,3}, {0,3}|{1,2}.
    std::array<double, 3> double_cuts{};
    double min_single_cut = 0.0;
    double min_double_cut = 0.0;
    // The six pair marginals' fermionic negativities and the four triple
    // marginals' minimum one-vs-rest cut, in sorted-position order.
    std::array<double, 6> pair_marginal_fn{};
    std::array<double, 4> triple_marginal_min_cut{};
    double max_pair_marginal_fn = 0.0;
    double max_triple_marginal_min_cut = 0.0;
    double joint_purity = 0.0;
    double trace_error = 0.0;
    double hermiticity_error = 0.0;
    double min_eigenvalue = 0.0;
    double parity_leakage = 0.0;
    // The signature the residue is tested against: every cut entangled while
    // every pair and triple marginal is separable.
    bool globally_entangled_locally_separable = false;
    // Assisted endpoint negativity under the two parity-respecting helper
    // families, by which pair of sorted positions is measured. Indexed the same
    // way as pair_marginal_fn.
    std::array<analysis::AssistedNegativity, 6> assisted_occupation{};
    std::array<analysis::AssistedNegativity, 6> assisted_parity{};
};

inline std::size_t pair_slot_of(int p, int q)
{
    static const std::array<std::array<int, 2>, 6> pairs{
        {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    const int lo = std::min(p, q);
    const int hi = std::max(p, q);
    for (std::size_t s = 0; s < pairs.size(); ++s)
    {
        if (pairs[s][0] == lo && pairs[s][1] == hi)
        {
            return s;
        }
    }
    return 0;
}

inline FourResult evaluate_four(const double *raw, const std::array<int, 4> &sites,
                                double cut_tol, bool assisted = false,
                                double assisted_tol = 1.0e-10)
{
    FourResult out;
    out.sites = sites;

    double trace = 0.0;
    for (int d = 0; d < 16; ++d)
    {
        trace += raw[2u * static_cast<std::size_t>(d * 16 + d)];
    }
    out.trace_error = std::abs(trace - 1.0);
    for (int r = 0; r < 16; ++r)
    {
        for (int c = 0; c < 16; ++c)
        {
            const std::size_t rc = 2u * static_cast<std::size_t>(r * 16 + c);
            const std::size_t cr = 2u * static_cast<std::size_t>(c * 16 + r);
            out.hermiticity_error = std::max(
                out.hermiticity_error, std::hypot(raw[rc] - raw[cr], raw[rc + 1] + raw[cr + 1]));
        }
    }
    if (trace > 0.0)
    {
        out.hermiticity_error /= trace;
    }

    const ancilla::SmallRdm rho = normalized_small_rdm(raw, 16);
    std::array<std::complex<double>, 256> dense{};
    for (int r = 0; r < 16; ++r)
    {
        for (int c = 0; c < 16; ++c)
        {
            const std::complex<double> value = rho(r, c);
            dense[static_cast<std::size_t>(r * 16 + c)] = value;
            out.rho_ri[2u * static_cast<std::size_t>(r * 16 + c)] = value.real();
            out.rho_ri[2u * static_cast<std::size_t>(r * 16 + c) + 1u] = value.imag();
            if ((__builtin_popcount(static_cast<unsigned>(r)) & 1) !=
                (__builtin_popcount(static_cast<unsigned>(c)) & 1))
            {
                out.parity_leakage += std::abs(value);
            }
        }
    }

    out.parts = four_party_entropies(rho, true);
    out.joint_purity = ancilla::purity_from_small_rdm(rho);

    out.min_single_cut = std::numeric_limits<double>::infinity();
    for (int s = 0; s < 4; ++s)
    {
        const double cut = analysis::cut_negativity_mask<4>(out.rho_ri.data(), 1 << s, true);
        out.single_cuts[static_cast<std::size_t>(s)] = cut;
        out.min_single_cut = cut == cut ? std::min(out.min_single_cut, cut)
                                        : std::numeric_limits<double>::quiet_NaN();
    }
    const std::array<int, 3> double_masks{0b0011, 0b0101, 0b1001};
    out.min_double_cut = std::numeric_limits<double>::infinity();
    for (std::size_t d = 0; d < 3; ++d)
    {
        const double cut = analysis::cut_negativity_mask<4>(
            out.rho_ri.data(), double_masks[d], true);
        out.double_cuts[d] = cut;
        out.min_double_cut = cut == cut ? std::min(out.min_double_cut, cut)
                                        : std::numeric_limits<double>::quiet_NaN();
    }

    // Pair marginals: the two-mode fermionic negativity, through the generic
    // path -- the closed form wants parity weights this routine does not carry,
    // and six 4x4 trace norms are free next to the 16x16 above.
    std::size_t slot = 0;
    for (int p = 0; p < 4; ++p)
    {
        for (int q = p + 1; q < 4; ++q)
        {
            const unsigned mask = (1u << p) | (1u << q);
            const ancilla::SmallRdm marginal = ancilla::partial_trace_fixed(rho, 4, mask, true);
            std::array<double, 32> packed{};
            for (int r = 0; r < 4; ++r)
            {
                for (int c = 0; c < 4; ++c)
                {
                    packed[2u * static_cast<std::size_t>(r * 4 + c)] = marginal(r, c).real();
                    packed[2u * static_cast<std::size_t>(r * 4 + c) + 1u] = marginal(r, c).imag();
                }
            }
            const double fn = analysis::cut_negativity<2>(packed.data(), 0, true);
            out.pair_marginal_fn[slot++] = fn;
            if (fn == fn)
            {
                out.max_pair_marginal_fn = std::max(out.max_pair_marginal_fn, fn);
            }
        }
    }

    // Triple marginals: the minimum one-vs-rest fermionic cut, the same
    // quantity the three-site analysis prefilters on.
    for (int drop = 0; drop < 4; ++drop)
    {
        const unsigned mask = 15u & ~(1u << drop);
        const ancilla::SmallRdm marginal = ancilla::partial_trace_fixed(rho, 4, mask, true);
        std::array<double, 128> packed{};
        for (int r = 0; r < 8; ++r)
        {
            for (int c = 0; c < 8; ++c)
            {
                packed[2u * static_cast<std::size_t>(r * 8 + c)] = marginal(r, c).real();
                packed[2u * static_cast<std::size_t>(r * 8 + c) + 1u] = marginal(r, c).imag();
            }
        }
        std::array<double, 3> cuts{};
        analysis::cut_negativities<3>(packed.data(), true, cuts);
        double smallest = std::numeric_limits<double>::infinity();
        for (double cut : cuts)
        {
            smallest = cut == cut ? std::min(smallest, cut)
                                  : std::numeric_limits<double>::quiet_NaN();
        }
        out.triple_marginal_min_cut[static_cast<std::size_t>(drop)] = smallest;
        if (smallest == smallest)
        {
            out.max_triple_marginal_min_cut =
                std::max(out.max_triple_marginal_min_cut, smallest);
        }
    }

    std::array<double, 16> eigenvalues{};
    out.min_eigenvalue = util::hermitian_eigenvalues<16>(dense, eigenvalues)
                             ? *std::min_element(eigenvalues.begin(), eigenvalues.end())
                             : std::numeric_limits<double>::quiet_NaN();

    // The four-mode GHZ signature: entangled across every one of the seven
    // cuts, separable in every pair and every triple marginal. Both halves are
    // threshold statements, so a NaN anywhere denies the claim rather than
    // passing it.
    bool every_cut = out.min_single_cut == out.min_single_cut &&
                     out.min_double_cut == out.min_double_cut &&
                     out.min_single_cut > cut_tol && out.min_double_cut > cut_tol;
    bool marginals_separable = out.max_pair_marginal_fn <= cut_tol &&
                               out.max_triple_marginal_min_cut <= cut_tol;
    for (double fn : out.pair_marginal_fn)
    {
        marginals_separable = marginals_separable && fn == fn;
    }
    for (double cut : out.triple_marginal_min_cut)
    {
        marginals_separable = marginals_separable && cut == cut;
    }
    out.globally_entangled_locally_separable = every_cut && marginals_separable;

    if (assisted)
    {
        // Both families, for every choice of which two modes are the helpers.
        // The joint parity-sector family is the one that matters here: a
        // four-mode GHZ has zero endpoint negativity under any single-mode
        // occupation measurement and a full Bell pair under this one.
        std::size_t slot = 0;
        for (int p = 0; p < 4; ++p)
        {
            for (int q = p + 1; q < 4; ++q)
            {
                const unsigned helpers = (1u << p) | (1u << q);
                const unsigned endpoints = 15u & ~helpers;
                out.assisted_occupation[slot] = analysis::assisted_negativity(
                    rho, 4, endpoints, helpers, analysis::occupation_family(), assisted_tol);
                out.assisted_parity[slot] = analysis::assisted_negativity(
                    rho, 4, endpoints, helpers, analysis::parity_sector_family(), assisted_tol);
                ++slot;
            }
        }
    }
    return out;
}

// Fills `raw` with one raw fermionic 16x16 per quadruple, interleaved.
using Rdm4Batch = std::function<void(const std::vector<std::array<int, 4>> &quads,
                                     std::vector<double> &raw)>;

// ---------------------------------------------------------------------------
// The four-site CSV
// ---------------------------------------------------------------------------

inline const std::vector<std::string> &four_site_columns()
{
    static const std::vector<std::string> columns{
        "run_id", "realization_id", "four_id", "anchor_role", "pair_i", "pair_j", "helper_k",
        "helper_l", "helper_control", "helper_same_component", "helper_on_backbone",
        "helper_distance", "separation", "d", "pair_class", "explanatory_class",
        "N", "periods", "p", "circ_type", "circuit_name", "master_seed", "statevector_precision",
        "boundary_implementation", "connectivity_graph_version", "gap_format_version",
        "pair_zero_tol", "occupation_mi_tol", "channel_floor", "fgmn_positive_tol",
        "prefilter_tol", "cut_positive_tol", "correlation_tol", "mosek_tol",
        // one-vs-three, by role
        "cut_i", "cut_j", "cut_k", "cut_l", "min_single_cut",
        // two-vs-two, named by which two sit together
        "cut_ij_kl", "cut_ik_jl", "cut_il_jk", "min_double_cut",
        // marginals
        "fn_ij", "fn_ik", "fn_il", "fn_jk", "fn_jl", "fn_kl", "max_pair_marginal_fn",
        "triple_min_cut_ijk", "triple_min_cut_ijl", "triple_min_cut_ikl", "triple_min_cut_jkl",
        "max_triple_marginal_min_cut",
        // information
        "mi_ij", "mi_kl", "mi_ij_kl", "cmi_ij_given_kl", "joint_purity",
        "purity_i", "purity_j", "purity_k", "purity_l",
        // verdict and numerics
        "globally_entangled_locally_separable",
        // Assisted endpoint negativity when the two helpers are measured.
        // Occupation is the single-mode family applied to each helper
        // independently; parity_sector is the joint family inside each parity
        // block. Single-mode X/Y is absent on purpose -- it is not a fermionic
        // measurement. See analysis/assisted.hpp.
        "assisted_occ_avg", "assisted_occ_max", "assisted_occ_positive_probability",
        "assisted_parity_avg", "assisted_parity_max", "assisted_parity_positive_probability",
        "assisted_parity_best_outcome", "assisted_unconditional",
        "trace_error", "hermiticity_error", "min_eigenvalue", "parity_leakage"};
    return columns;
}

inline std::string four_site_csv_header()
{
    std::string header;
    const auto &columns = four_site_columns();
    for (std::size_t c = 0; c < columns.size(); ++c)
    {
        if (c > 0)
        {
            header += ',';
        }
        header += columns[c];
    }
    header += '\n';
    return header;
}

inline std::string four_site_path_for(const std::string &outcome_csv)
{
    std::filesystem::path path(outcome_csv);
    path.replace_extension();
    std::string stem = path.string();
    static const std::string suffix = "_connected_zero_thirds";
    if (stem.size() >= suffix.size() &&
        stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
        stem.resize(stem.size() - suffix.size());
        return stem + "_connected_zero_four_site.csv";
    }
    return stem + "_four_site.csv";
}

inline std::string rho4_path_for(const std::string &outcome_csv)
{
    std::filesystem::path path(outcome_csv);
    path.replace_extension();
    return path.string() + "_rho4.bin";
}

} // namespace mipt::dist::gap
