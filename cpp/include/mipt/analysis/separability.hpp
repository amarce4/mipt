#pragma once

// Cut-by-cut separability diagnostics for a small multi-mode density matrix.
// ------------------------------------------------------------------------
//
// A vanishing cut negativity says a bipartition is PPT. It does not say *why*,
// and the three reasons make different predictions:
//
//   * a **product** cut: zero negativity and zero mutual information. The two
//     sides share nothing at all, so there is nothing for a third site to
//     reveal.
//   * a **classically correlated separable** cut: zero negativity, positive
//     mutual information. The sides are correlated, but the correlation is not
//     entanglement.
//   * an **entangled** cut: positive negativity.
//   * a **numerically unresolved** cut: neither is above its threshold and the
//     quantities are the size of the arithmetic, so nothing is being claimed.
//
// The distance from product form settles the first two without appealing to an
// entropy threshold:
//
//     delta_s = || rho - rho_s (x) rho_sbar ||_1
//
// which is exactly zero for a product cut and positive for any correlation,
// classical or quantum. Trace norm rather than a Frobenius one because it is
// the operationally meaningful distinguishability, and because it is the same
// norm the negativity is built from.
//
// **Ordinary-trace quantities are labelled `qubit_`, deliberately and
// everywhere.** For a fermionic ensemble the ordinary partial transpose is a
// different criterion from the fermionic one -- parity superselection makes
// fermionic separability the stronger requirement -- so an ordinary PPT witness
// is a comparison observable here, not a statement about fermionic
// separability. Mixing the two vocabularies is how a "bound entangled fermionic
// state" gets reported that is nothing of the kind.
//
// The realignment (computable cross-norm, CCNR) criterion is the classic
// companion to PPT: ||R(rho)||_1 > 1 implies entanglement, and it detects a
// family of PPT-entangled states the partial transpose misses. A cut that is
// ordinary-PPT *and* CCNR-positive is therefore a conventional
// bound-entanglement candidate, which is worth exporting for offline study and
// is not a claim on its own -- both criteria are one-sided.
//
// Header-only and dependency-free, so `make test-dist` pins it on the host.

#include "mipt/analysis/cut_negativity.hpp"
#include "mipt/small_rdm.hpp"
#include "mipt/util/spectral.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <limits>

namespace mipt::analysis
{

// Minimum eigenvalue of the ordinary partial transpose across cut `subsystem`.
//
// The ordinary partial transpose of a Hermitian matrix stays Hermitian, so this
// is well defined; the fermionic one is not Hermitian and has no such spectrum,
// which is why this is an ordinary-trace (`qubit_`) diagnostic only. A negative
// value is the PPT violation, and it is a finer instrument than the negativity:
// the negativity sums the negative eigenvalues and can round a single tiny one
// away, while this reports it.
template <int Parties>
inline double qubit_partial_transpose_min_eigenvalue(const double *rho_ri, int subsystem)
{
    constexpr int D = 1 << Parties;
    if (rho_ri == nullptr || subsystem < 0 || subsystem >= Parties)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int mask = 1 << subsystem;
    std::array<std::complex<double>, static_cast<std::size_t>(D * D)> pt{};
    for (int r = 0; r < D; ++r)
    {
        for (int c = 0; c < D; ++c)
        {
            const int source_row = (r & ~mask) | (c & mask);
            const int source_col = (c & ~mask) | (r & mask);
            const std::size_t source = 2u * static_cast<std::size_t>(source_row * D + source_col);
            pt[static_cast<std::size_t>(r * D + c)] =
                std::complex<double>(rho_ri[source], rho_ri[source + 1u]);
        }
    }
    std::array<double, static_cast<std::size_t>(D)> eigenvalues{};
    if (!util::hermitian_eigenvalues<static_cast<std::size_t>(D)>(pt, eigenvalues))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double smallest = eigenvalues[0];
    for (double value : eigenvalues)
    {
        smallest = value < smallest ? value : smallest;
    }
    return smallest;
}

// Realignment (CCNR) norm across the cut `subsystem` versus the rest.
//
//     R(rho)_{(a,a'),(b,b')} = rho_{(a,b),(a',b')}
//
// with `a` the cut party's index and `b` the rest's. ||R||_1 <= 1 for every
// separable state, so a value above 1 certifies entanglement. The matrix is
// dim_A^2 x dim_B^2 -- 4 x 16 for one qubit against two -- and rectangular, so
// it is zero-padded into a square before the trace norm. Padding adds only zero
// singular values, so the norm is unchanged.
template <int Parties>
inline double realignment_norm(const double *rho_ri, int subsystem)
{
    constexpr int D = 1 << Parties;
    constexpr int DA = 2;                 // one party
    constexpr int DB = D / DA;            // the rest
    constexpr int ROWS = DA * DA;         // 4
    constexpr int COLS = DB * DB;         // 16 at Parties = 3
    constexpr std::size_t PAD = ROWS > COLS ? ROWS : COLS;
    if (rho_ri == nullptr || subsystem < 0 || subsystem >= Parties)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int mask = 1 << subsystem;
    // Split a basis index into (cut bit, rest index), keeping the rest's bits in
    // their original relative order.
    auto split = [&](int x) {
        const int a = (x & mask) != 0 ? 1 : 0;
        const int low = x & (mask - 1);
        const int high = (x >> (subsystem + 1)) << subsystem;
        return std::pair<int, int>(a, low | high);
    };

    std::array<std::complex<double>, PAD * PAD> matrix{};
    for (int row = 0; row < D; ++row)
    {
        for (int col = 0; col < D; ++col)
        {
            const auto [a, b] = split(row);
            const auto [a2, b2] = split(col);
            const std::size_t source = 2u * static_cast<std::size_t>(row * D + col);
            matrix[static_cast<std::size_t>((a * DA + a2) * static_cast<int>(PAD) + (b * DB + b2))] =
                std::complex<double>(rho_ri[source], rho_ri[source + 1u]);
        }
    }
    double norm = 0.0;
    if (!util::dilation_trace_norm<PAD>(matrix, norm))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return norm;
}

// || rho - rho_s (x) rho_sbar ||_1 across cut `subsystem`.
//
// The marginals come from the same partial trace the entropies use, so
// `fermionic` selects the same convention; the product is taken in the
// occupation basis, which for states of definite global parity is the right
// notion of "no correlation at all" -- the Jordan-Wigner strings of two
// parity-definite factors cancel.
template <int Parties>
inline double product_distance(const double *rho_ri, int subsystem, bool fermionic)
{
    constexpr int D = 1 << Parties;
    if (rho_ri == nullptr || subsystem < 0 || subsystem >= Parties)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    ancilla::SmallRdm rho(D);
    for (int r = 0; r < D; ++r)
    {
        for (int c = 0; c < D; ++c)
        {
            const std::size_t source = 2u * static_cast<std::size_t>(r * D + c);
            rho(r, c) = std::complex<double>(rho_ri[source], rho_ri[source + 1u]);
        }
    }
    const unsigned cut_mask = 1u << subsystem;
    const unsigned rest_mask = static_cast<unsigned>((1 << Parties) - 1) & ~cut_mask;
    const ancilla::SmallRdm cut = ancilla::partial_trace_fixed(rho, Parties, cut_mask, fermionic);
    const ancilla::SmallRdm rest = ancilla::partial_trace_fixed(rho, Parties, rest_mask, fermionic);

    auto split = [&](int x) {
        const int a = (x & (1 << subsystem)) != 0 ? 1 : 0;
        const int low = x & ((1 << subsystem) - 1);
        const int high = (x >> (subsystem + 1)) << subsystem;
        return std::pair<int, int>(a, low | high);
    };

    std::array<std::complex<double>, static_cast<std::size_t>(D * D)> difference{};
    for (int r = 0; r < D; ++r)
    {
        for (int c = 0; c < D; ++c)
        {
            const auto [ra, rb] = split(r);
            const auto [ca, cb] = split(c);
            difference[static_cast<std::size_t>(r * D + c)] =
                rho(r, c) - cut(ra, ca) * rest(rb, cb);
        }
    }
    double norm = 0.0;
    if (!util::hermitian_trace_norm<static_cast<std::size_t>(D)>(difference, norm))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return norm;
}

// How a cut looks once both criteria have been applied.
enum class CutCharacter : int
{
    Product = 0,               // no correlation of any kind across the cut
    ClassicallyCorrelated = 1, // correlated, no negativity
    Entangled = 2,             // positive negativity
    Unresolved = 3,            // everything at the size of the arithmetic
};
inline constexpr int CUT_CHARACTER_COUNT = 4;

inline const char *cut_character_name(CutCharacter value)
{
    switch (value)
    {
    case CutCharacter::Product: return "product";
    case CutCharacter::ClassicallyCorrelated: return "classically_correlated";
    case CutCharacter::Entangled: return "entangled";
    case CutCharacter::Unresolved: break;
    }
    return "unresolved";
}

// `negativity` decides entanglement; below that threshold the product distance
// decides whether there is any correlation left to explain. A NaN in either is
// unresolved -- it is not evidence of anything.
inline CutCharacter classify_cut(double negativity, double product_distance_value,
                                 double negativity_tol, double correlation_tol)
{
    if (!(negativity == negativity) || !(product_distance_value == product_distance_value))
    {
        return CutCharacter::Unresolved;
    }
    if (negativity > negativity_tol)
    {
        return CutCharacter::Entangled;
    }
    if (product_distance_value > correlation_tol)
    {
        return CutCharacter::ClassicallyCorrelated;
    }
    return CutCharacter::Product;
}

// Every per-cut diagnostic of a three-party state, indexed by the party cut off.
struct TripleSeparability
{
    std::array<double, 3> mutual_information{};  // I(s : rest), bits, fermionic
    std::array<double, 3> product_distance{};    // ||rho - rho_s (x) rho_rest||_1
    std::array<double, 3> qubit_cut{};           // ordinary-trace cut negativity
    std::array<double, 3> qubit_pt_min_eigenvalue{};
    std::array<double, 3> realignment{};         // CCNR norm; > 1 certifies entanglement
    std::array<CutCharacter, 3> character{};
    // Ordinary-PPT on every cut but CCNR-positive on at least one: a
    // conventional bound-entanglement candidate. Both criteria are one-sided,
    // so this flags a state worth studying offline, it does not settle it.
    bool qubit_ppt_ccnr_candidate = false;
    double max_realignment = 0.0;
};

// `entropies[mask]` is the von Neumann entropy of the subsystem named by that
// bit mask, as three_party_entropies fills it.
inline TripleSeparability triple_separability(const double *rho_ri,
                                              const std::array<double, 8> &entropies,
                                              double negativity_tol, double correlation_tol)
{
    TripleSeparability out;
    bool all_ppt = true;
    bool any_ccnr = false;
    for (int s = 0; s < 3; ++s)
    {
        const std::size_t index = static_cast<std::size_t>(s);
        const int bit = 1 << s;
        out.mutual_information[index] = entropies[static_cast<std::size_t>(bit)] +
                                        entropies[static_cast<std::size_t>(7 ^ bit)] -
                                        entropies[7];
        out.product_distance[index] = product_distance<3>(rho_ri, s, true);
        out.qubit_cut[index] = cut_negativity<3>(rho_ri, s, false);
        out.qubit_pt_min_eigenvalue[index] = qubit_partial_transpose_min_eigenvalue<3>(rho_ri, s);
        out.realignment[index] = realignment_norm<3>(rho_ri, s);
        // The character is stated for the *fermionic* criterion's companion
        // quantities, so the negativity used here is the ordinary one only
        // because that is what this struct carries; the caller passes the
        // fermionic cut separately when it wants the fermionic reading.
        out.character[index] = classify_cut(out.qubit_cut[index], out.product_distance[index],
                                            negativity_tol, correlation_tol);
        if (!(out.qubit_cut[index] <= negativity_tol))
        {
            all_ppt = false;
        }
        if (out.realignment[index] == out.realignment[index])
        {
            out.max_realignment = std::max(out.max_realignment, out.realignment[index]);
            // 1 + tol rather than 1: the norm of a product state is exactly 1,
            // so the criterion has to clear the arithmetic around it.
            if (out.realignment[index] > 1.0 + correlation_tol)
            {
                any_ccnr = true;
            }
        }
    }
    out.qubit_ppt_ccnr_candidate = all_ppt && any_ccnr;
    return out;
}

} // namespace mipt::analysis
