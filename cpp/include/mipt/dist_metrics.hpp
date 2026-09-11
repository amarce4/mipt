#pragma once

// Site-resolved entanglement measures for dist_scaling.exe.
//
// Two-party metrics act on one 4x4 RDM; three-party metrics act on one 8x8.
// Both take the interleaved row-major complex buffer the reduction kernels
// write, and both are expressed in *bits* -- dist_scaling has always used
// log2 for its mutual information, and the k=3 additions follow it rather
// than the nats convention of the probe executables. The CSV writer states
// the unit explicitly so no reader has to remember which is which.
//
// Nothing here needs CUDA-Q, MOSEK, or a state vector, which is what lets
// `make test-dist` check it on the host.

#include "mipt/small_rdm.hpp"
#include "mipt/util/geometry.hpp"
#include "mipt/util/spectral.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace mipt::dist
{
using Complex = std::complex<double>;
using Matrix2 = std::array<Complex, 4>;
using Matrix4 = std::array<Complex, 16>;

struct PairMetrics
{
    double mi = 0.0;
    // The reported negativity. For a fermionic trace on a parity-preserving
    // ensemble this is `mn_closed`; otherwise it is `mn_generic`.
    double mn = 0.0;
    // Squared magnitudes of the two fermionic two-point functions; see
    // two_point_correlators() for what they are and which trace they mean.
    double g2 = 0.0;
    double f2 = 0.0;

    // --- occupation-basis diagnostics ------------------------------------
    //
    // These are diagonal quantities, so they are *identical* under both trace
    // conventions: a Jordan-Wigner string changes reorder signs on entries
    // that connect different local parities, and the diagonal has none. That
    // is why dist_scaling records them once per pair rather than once per
    // trace.
    double n_i = 0.0;   // <n_i>
    double n_j = 0.0;   // <n_j>
    double dnn = 0.0;   // D = <n_i n_j>
    double rho_n = 0.0; // the connected density correlation D - <n_i><n_j>
    double i_occ = 0.0; // mutual information of the joint occupation
                        // distribution alone, in bits

    // --- the two-mode channel --------------------------------------------
    Complex g{};             // <c_i^dag c_j>
    Complex f{};             // <c_i c_j>
    double parity_even = 0.0; // p(00) + p(11)
    double parity_odd = 0.0;  // p(10) + p(01)

    // --- negativity, both ways -------------------------------------------
    double mn_generic = 0.0; // from the generic partial transpose
    double mn_closed = std::numeric_limits<double>::quiet_NaN(); // closed form
    double mn_residual = std::numeric_limits<double>::quiet_NaN();
    // Total weight of entries connecting different local parities. Zero in
    // exact arithmetic for any circuit that conserves computational parity, so
    // this is the numerical evidence that the closed form applies at all.
    double parity_leakage = 0.0;
};

// The two fermionic two-point functions of a mode pair (i, j) with i < j:
// the hopping correlator G_ij = <c_i^dag c_j> and the pairing (anomalous)
// correlator F_ij = <c_i c_j>.
//
// Both are *single entries* of the two-mode density matrix, because each
// operator connects exactly one pair of Fock states. In the basis
// |n_i n_j> = (c_i^dag)^{n_i} (c_j^dag)^{n_j} |vac> indexed as n_i + 2 n_j --
// which is what the fermionic partial trace produces, bit k of the RDM being
// retained[k], and the pair list being ascending --
//
//     c_i^dag c_j |0,1> = |1,0>,   zero on every other basis state
//     c_i c_j     |1,1> = -|0,0>,  zero on every other basis state
//
// so Tr(rho c_i^dag c_j) = rho[2][1] and Tr(rho c_i c_j) = -rho[3][0].
//
// The *positions* follow from particle number alone -- c_i^dag c_j moves one
// fermion from j to i, c_i c_j removes both -- so they carry no Jordan-Wigner
// phase convention. The signs do, but this returns squared magnitudes, in
// which they cancel: |G|^2 and |F|^2 are convention-independent given a
// correct fermionic reduced density matrix.
//
// F is identically zero whenever the circuit conserves particle number. RPPU
// conserves only parity, so it is a genuine observable there.
//
// **Which rho matters.** Fed the fermionic-trace RDM these are G and F. Fed
// the ordinary-trace one they are the spin correlators
// |<sigma_i^+ sigma_j^->|^2 and |<sigma_i^- sigma_j^->|^2 -- the same matrix
// entries of a different matrix. The Jordan-Wigner string running between the
// two modes is exactly what the fermionic partial trace restores and the
// ordinary one drops.
struct TwoPointCorrelators
{
    double g2 = 0.0; // |<c_i^dag c_j>|^2
    double f2 = 0.0; // |<c_i c_j>|^2
};

inline TwoPointCorrelators two_point_correlators(const Matrix4 &rho)
{
    return {std::norm(rho[2 * 4 + 1]), std::norm(rho[3 * 4 + 0])};
}

// The three-party analogue. `tmi` is the tripartite information
// I_3 = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC, which is the same
// combination three-probe mode 4 reports; `average_mi` is the mean of the
// three pairwise mutual informations, and the purities are diagnostics that
// say whether a small |I_3| means "weakly correlated" or "nearly pure".
struct TripleMetrics
{
    double tmi = 0.0;
    double average_mi = 0.0;
    double joint_purity = 0.0;
    double mean_single_purity = 0.0;
};

// The mutual information of the *diagonal* of a two-mode RDM: how much the
// occupation of one mode says about the other, with every coherence discarded.
//
// This is the classical part of the pair correlation, and it is what separates
// the two ways a graph-connected pair can carry no entanglement. A pair with
// I_occ > 0 is genuinely correlated and merely not entangled; a pair with
// I_occ ~ 0 and vanishing G and F carries no detectable two-mode correlation
// of any kind, which is a much stronger statement about the spanning path.
//
// Reported in bits, matching every other information measure in this header.
inline double occupation_mutual_information(double p00, double p10, double p01, double p11)
{
    const double n_i = p10 + p11;
    const double n_j = p01 + p11;
    const std::array<double, 4> joint{p00, p10, p01, p11};
    const std::array<double, 4> product{(1.0 - n_i) * (1.0 - n_j), n_i * (1.0 - n_j),
                                        (1.0 - n_i) * n_j, n_i * n_j};
    constexpr double eps = 1.0e-15;
    double total = 0.0;
    for (std::size_t index = 0; index < 4; ++index)
    {
        if (joint[index] > eps && product[index] > eps)
        {
            total += joint[index] * std::log2(joint[index] / product[index]);
        }
    }
    // Non-negative in exact arithmetic; the clamp only removes rounding noise
    // on a product state, where every term cancels against every other.
    return (total < 0.0 && total > -1.0e-10) ? 0.0 : total;
}

// The fermionic negativity of a parity-preserving two-mode state, in closed
// form.
//
// Derivation-free statement: for a state that is block diagonal in the local
// fermion parity, the fermionic partial transpose has exactly two negative
// eigenvalues, and summing them gives
//
//     N = (sqrt(p_e^2 + 4|G|^2) - p_e) / 2 + (sqrt(p_o^2 + 4|F|^2) - p_o) / 2
//
// with p_e = p(00) + p(11) and p_o = p(10) + p(01). The pairing is not the one
// parity bookkeeping suggests -- |G|^2, an *odd*-sector coherence, rides on the
// *even* weight -- because the fermionic partial transpose mixes the blocks.
// It is verified against the generic path in `make test-dist`, which is the
// only reason to trust it over intuition.
//
// **Why it is written this way.** The generic route computes the trace norm of
// the partial transpose and subtracts 1. On a weakly entangled pair that norm
// is 1 + O(10^-14), so the subtraction throws away every significant digit:
// measured against the exact value, the generic path is still good at
// |G|^2 = 10^-14, returns 2.2e-16 where the answer is 1.67e-16, and returns
// *exactly zero* from |G|^2 = 10^-18 down. Multiplying through by the
// conjugate surd turns the difference of two nearly equal numbers into a ratio
// and removes the cancellation entirely: the form below is accurate to full
// relative precision at any magnitude.
//
// That distinction is the whole point of the exercise. A fixed positivity
// threshold applied to the generic value cannot tell a pair whose negativity
// is algebraically zero from one whose negativity has merely shrunk below the
// solver's noise floor, and the two make opposite predictions for how
// P(fN > eps) behaves as eps is swept.
inline double parity_preserving_fermionic_negativity(double parity_even, double parity_odd,
                                                     double g2, double f2)
{
    double total = 0.0;
    if (g2 > 0.0)
    {
        total += 2.0 * g2 / (std::sqrt(parity_even * parity_even + 4.0 * g2) + parity_even);
    }
    if (f2 > 0.0)
    {
        total += 2.0 * f2 / (std::sqrt(parity_odd * parity_odd + 4.0 * f2) + parity_odd);
    }
    return total;
}

namespace detail
{
inline double entropy_term(double lambda)
{
    constexpr double eps = 1.0e-15;
    return lambda > eps ? -lambda * std::log2(lambda) : 0.0;
}

template <std::size_t D>
inline double von_neumann_entropy(const std::array<Complex, D * D> &rho)
{
    std::array<double, D> eigenvalues{};
    if (!util::hermitian_eigenvalues<D>(rho, eigenvalues))
    {
        throw std::runtime_error(
            "Two-site density-matrix eigensolver did not converge.");
    }

    double entropy = 0.0;
    for (double lambda : eigenvalues)
    {
        if (lambda < 0.0 && lambda > -1.0e-11)
        {
            lambda = 0.0;
        }
        entropy += entropy_term(lambda);
    }
    return std::abs(entropy) < 1.0e-12 ? 0.0 : entropy;
}

inline Matrix4 normalized_hermitian_rho(const double *rho_ri)
{
    if (rho_ri == nullptr)
    {
        throw std::invalid_argument("Null two-site density matrix.");
    }

    Matrix4 rho{};
    for (std::size_t row = 0; row < 4; ++row)
    {
        for (std::size_t col = row; col < 4; ++col)
        {
            const std::size_t rc = row * 4 + col;
            const std::size_t cr = col * 4 + row;
            const Complex a(rho_ri[2 * rc], rho_ri[2 * rc + 1]);
            const Complex b(rho_ri[2 * cr], rho_ri[2 * cr + 1]);
            Complex value = 0.5 * (a + std::conj(b));
            if (row == col)
            {
                value = Complex(value.real(), 0.0);
            }
            rho[rc] = value;
            rho[cr] = std::conj(value);
        }
    }

    double trace = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
    {
        trace += rho[i * 4 + i].real();
    }
    if (!(trace > 0.0) || !std::isfinite(trace))
    {
        throw std::runtime_error("Two-site density matrix has a non-positive or non-finite trace.");
    }
    for (Complex &value : rho)
    {
        value /= trace;
    }
    return rho;
}

inline Matrix2 trace_to_one_mode(const Matrix4 &rho,
                                 int kept_position,
                                 bool fermionic)
{
    if (kept_position != 0 && kept_position != 1)
    {
        throw std::invalid_argument("kept_position must be 0 or 1.");
    }

    Matrix2 out{};
    const int traced_position = 1 - kept_position;
    const std::array<int, 2> new_mode_order{kept_position, traced_position};

    auto old_index = [&](int kept, int env) {
        const int new_index = kept | (env << 1);
        return ((new_index >> 0) & 1) << new_mode_order[0] |
               ((new_index >> 1) & 1) << new_mode_order[1];
    };
    auto reorder_sign = [&](int kept, int env) {
        if (!fermionic || new_mode_order[0] < new_mode_order[1])
        {
            return 1;
        }
        return (kept && env) ? -1 : 1;
    };

    for (int row_kept = 0; row_kept < 2; ++row_kept)
    {
        for (int col_kept = 0; col_kept < 2; ++col_kept)
        {
            Complex sum(0.0, 0.0);
            for (int env = 0; env < 2; ++env)
            {
                const int row = old_index(row_kept, env);
                const int col = old_index(col_kept, env);
                const int sign = reorder_sign(row_kept, env) *
                                 reorder_sign(col_kept, env);
                sum += static_cast<double>(sign) * rho[row * 4 + col];
            }
            out[row_kept * 2 + col_kept] = sum;
        }
    }
    return out;
}

inline double negativity(const Matrix4 &rho, bool fermionic)
{
    Matrix4 partial_transpose{};
    constexpr int mask = 1; // transpose the first retained mode (local bit 0)
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const int source_row = (row & ~mask) | (col & mask);
            const int source_col = (col & ~mask) | (row & mask);
            Complex value = rho[source_row * 4 + source_col];

            // Match the existing fGMN convention: after ordinary partial
            // transpose, local-parity-violating entries acquire a factor i.
            if (fermionic && (((row & mask) != 0) != ((col & mask) != 0)))
            {
                value *= Complex(0.0, 1.0);
            }
            partial_transpose[row * 4 + col] = value;
        }
    }

    // The ordinary partial transpose stays Hermitian, so its singular values are
    // the absolute eigenvalues; only the fermionic one needs the Gram matrix.
    // Dilation rather than Gram for the fermionic case: this generic value is
    // the cross-check the closed form's residual is measured against, and a
    // ~1e-9 Gram floor would swamp the residual it exists to report.
    double norm = 0.0;
    const bool converged =
        fermionic ? util::dilation_trace_norm<4>(partial_transpose, norm)
                  : util::hermitian_trace_norm<4>(partial_transpose, norm);
    if (!converged)
    {
        throw std::runtime_error(
            "Two-site partial-transpose eigensolver did not converge.");
    }

    double result = 0.5 * (norm - 1.0);
    if (result < 0.0 && result > -1.0e-10)
    {
        result = 0.0;
    }
    if (result < -1.0e-8 || !std::isfinite(result))
    {
        throw std::runtime_error("Bipartite negativity calculation produced an invalid value.");
    }
    return std::max(0.0, result);
}
} // namespace detail

// Trace-normalized, Hermitized copy of an interleaved row-major complex block.
//
// The reduction kernels accumulate |psi|^2 in the working precision, so an
// fp32 trajectory arrives with a trace that is 1 only to about 1e-6 and with
// a small anti-Hermitian part. Both are removed once here so that every
// downstream consumer -- entropies, purities, and the negativity prefilter
// that decides whether an SDP runs at all -- sees the same matrix.
inline ancilla::SmallRdm normalized_small_rdm(const double *rho_ri, int dimension)
{
    if (rho_ri == nullptr || dimension < 2)
    {
        throw std::invalid_argument("Invalid interleaved density-matrix block.");
    }

    double trace = 0.0;
    for (int i = 0; i < dimension; ++i)
    {
        trace += rho_ri[2 * static_cast<std::size_t>(i * dimension + i)];
    }
    if (!(trace > 0.0) || !std::isfinite(trace))
    {
        throw std::runtime_error("Reduced density matrix has a non-positive or non-finite trace.");
    }

    ancilla::SmallRdm rho(dimension);
    for (int row = 0; row < dimension; ++row)
    {
        for (int col = row; col < dimension; ++col)
        {
            const std::size_t rc = 2u * static_cast<std::size_t>(row * dimension + col);
            const std::size_t cr = 2u * static_cast<std::size_t>(col * dimension + row);
            const Complex a(rho_ri[rc], rho_ri[rc + 1]);
            const Complex b(rho_ri[cr], rho_ri[cr + 1]);
            Complex value = 0.5 * (a + std::conj(b)) / trace;
            if (row == col)
            {
                value = Complex(value.real(), 0.0);
            }
            rho(row, col) = value;
            rho(col, row) = std::conj(value);
        }
    }
    return rho;
}

namespace detail
{
inline double bits_from_nats(double nats)
{
    static const double inverse_ln2 = 1.0 / std::log(2.0);
    return nats * inverse_ln2;
}
} // namespace detail

// Three-party information measures for one 8x8 site block.
//
// `fermionic` selects the Jordan-Wigner reorder signs inside the block. It
// must match the trace that produced `rho_ri`: the caller hands in the
// ordinary RDM with `fermionic=false` and the fermionic RDM with `true`,
// because the sub-traces taken here are the continuation of the same
// convention, not an independent choice.
// Every subsystem entropy of a three-mode block, indexed by the retained-mode
// bit mask (entry 0 unused), in bits, plus the purities three_party_metrics
// reports. Split out so a caller that needs a combination three_party_metrics
// does not report -- the conditional mutual information I(a:b|c), say -- takes
// it from the same partial traces rather than from a second implementation.
struct TripleEntropies
{
    std::array<double, 8> entropy{};
    std::array<double, 8> purity{};
};

inline TripleEntropies three_party_entropies(const ancilla::SmallRdm &rho, bool fermionic)
{
    TripleEntropies out;
    for (unsigned mask = 1; mask < 8; ++mask)
    {
        const ancilla::SmallRdm reduced = ancilla::partial_trace_fixed(rho, 3, mask, fermionic);
        out.entropy[mask] = detail::bits_from_nats(ancilla::entropy_from_small_rdm(reduced));
        out.purity[mask] = ancilla::purity_from_small_rdm(reduced);
    }
    return out;
}

inline TripleMetrics three_party_metrics(const double *rho_ri, bool fermionic)
{
    const ancilla::SmallRdm rho = normalized_small_rdm(rho_ri, 8);
    const TripleEntropies parts = three_party_entropies(rho, fermionic);
    const std::array<double, 8> &entropies = parts.entropy;

    const double i_ab = entropies[1] + entropies[2] - entropies[3];
    const double i_ac = entropies[1] + entropies[4] - entropies[5];
    const double i_bc = entropies[2] + entropies[4] - entropies[6];
    const double tmi = entropies[1] + entropies[2] + entropies[4] - entropies[3] - entropies[5] - entropies[6] +
                       entropies[7];
    if (!std::isfinite(tmi) || !std::isfinite(i_ab) || !std::isfinite(i_ac) || !std::isfinite(i_bc))
    {
        throw std::runtime_error("Three-party information calculation produced a non-finite value.");
    }

    const double mean_single_purity = (parts.purity[1] + parts.purity[2] + parts.purity[4]) / 3.0;
    return {tmi, (i_ab + i_ac + i_bc) / 3.0, parts.purity[7], mean_single_purity};
}

// `parity_preserving` says whether the ensemble conserves computational parity,
// which is what licenses the closed-form negativity above. It is a property of
// the circuit, so the caller supplies it rather than this function guessing
// from the matrix: on an fp32 trajectory the cross-parity entries are ~1e-7
// noise rather than exactly zero, and a threshold on them would silently
// switch formulas partway through a run. The measured leakage is reported
// instead, so the assumption can be checked after the fact.
inline PairMetrics two_party_metrics(const double *rho_ri, bool fermionic,
                                     bool parity_preserving = false)
{
    const Matrix4 rho = detail::normalized_hermitian_rho(rho_ri);
    const Matrix2 rho_a = detail::trace_to_one_mode(rho, 0, fermionic);
    const Matrix2 rho_b = detail::trace_to_one_mode(rho, 1, fermionic);

    double mi = detail::von_neumann_entropy<2>(rho_a) +
                detail::von_neumann_entropy<2>(rho_b) -
                detail::von_neumann_entropy<4>(rho);
    if (mi < 0.0 && mi > -1.0e-10)
    {
        mi = 0.0;
    }
    if (mi < -1.0e-8 || !std::isfinite(mi))
    {
        throw std::runtime_error("Mutual-information calculation produced an invalid value.");
    }

    PairMetrics out;
    out.mi = std::max(0.0, mi);

    // Basis |n_i n_j> indexed n_i + 2 n_j, as the fermionic partial trace
    // produces it; see two_point_correlators() for why these entries are the
    // ones they are.
    const double p00 = rho[0].real();
    const double p10 = rho[1 * 4 + 1].real();
    const double p01 = rho[2 * 4 + 2].real();
    const double p11 = rho[3 * 4 + 3].real();
    out.n_i = p10 + p11;
    out.n_j = p01 + p11;
    out.dnn = p11;
    out.rho_n = p11 - out.n_i * out.n_j;
    out.i_occ = occupation_mutual_information(p00, p10, p01, p11);
    out.parity_even = p00 + p11;
    out.parity_odd = p10 + p01;

    out.g = rho[2 * 4 + 1];
    out.f = -rho[3 * 4 + 0];
    out.g2 = std::norm(out.g);
    out.f2 = std::norm(out.f);

    // Entries joining a local-parity-even basis state to an odd one. Exactly
    // zero for a definite-global-parity state, because tracing out the
    // environment can only connect subsystem states of equal parity.
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const bool row_odd = ((row & 1) ^ ((row >> 1) & 1)) != 0;
            const bool col_odd = ((col & 1) ^ ((col >> 1) & 1)) != 0;
            if (row_odd != col_odd)
            {
                out.parity_leakage += std::abs(rho[row * 4 + col]);
            }
        }
    }

    out.mn_generic = detail::negativity(rho, fermionic);
    out.mn = out.mn_generic;
    if (fermionic && parity_preserving)
    {
        out.mn_closed = parity_preserving_fermionic_negativity(out.parity_even, out.parity_odd,
                                                              out.g2, out.f2);
        out.mn_residual = std::abs(out.mn_closed - out.mn_generic);
        out.mn = out.mn_closed;
    }
    return out;
}

using util::chord_length;
using util::periodic_distance;

inline double two_party_geometric_chord_distance(int n, int site_a, int site_b)
{
    if (site_a < 0 || site_a >= n || site_b < 0 || site_b >= n || site_a == site_b)
    {
        throw std::invalid_argument("Invalid two-site separation.");
    }
    const int forward = std::abs(site_b - site_a);
    const int backward = n - forward;
    return std::sqrt(chord_length(n, forward) * chord_length(n, backward));
}

// The k=3 analogue: the geometric mean of the triangle's three chord lengths.
//
// This is the same effective distance three-probe mode 4 writes as
// `chord_geometric_mean`, so a triangle measured by either protocol lands on
// the same abscissa. Note that the two-party version above is the geometric
// mean over the ring's *two arcs* rather than over pairs; on a periodic chain
// l(s) = l(N-s), so it reduces to the single chord and the two definitions
// agree on what "distance" means.
inline double triangle_chord_geometric_mean(int n, const std::array<int, 3> &separations)
{
    double product = 1.0;
    for (int separation : separations)
    {
        product *= chord_length(n, separation);
    }
    return std::cbrt(product);
}

inline std::array<int, 3> triangle_separations(int n, int site_a, int site_b, int site_c)
{
    if (site_a == site_b || site_a == site_c || site_b == site_c)
    {
        throw std::invalid_argument("Invalid three-site triangle: sites must be distinct.");
    }
    std::array<int, 3> separations{periodic_distance(site_a, site_b, n), periodic_distance(site_a, site_c, n),
                                   periodic_distance(site_b, site_c, n)};
    std::sort(separations.begin(), separations.end());
    return separations;
}

inline double three_party_geometric_chord_distance(int n, int site_a, int site_b, int site_c)
{
    return triangle_chord_geometric_mean(n, triangle_separations(n, site_a, site_b, site_c));
}
} // namespace mipt::dist
