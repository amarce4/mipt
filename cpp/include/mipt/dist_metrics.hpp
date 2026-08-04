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
    double mn = 0.0;
};

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
    double norm = 0.0;
    const bool converged =
        fermionic ? util::gram_trace_norm<4>(partial_transpose, norm)
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
inline TripleMetrics three_party_metrics(const double *rho_ri, bool fermionic)
{
    const ancilla::SmallRdm rho = normalized_small_rdm(rho_ri, 8);

    std::array<ancilla::SmallRdm, 8> reduced{ancilla::SmallRdm(1), ancilla::SmallRdm(2), ancilla::SmallRdm(2),
                                             ancilla::SmallRdm(4), ancilla::SmallRdm(2), ancilla::SmallRdm(4),
                                             ancilla::SmallRdm(4), ancilla::SmallRdm(8)};
    std::array<double, 8> entropies{};
    for (unsigned mask = 1; mask < 8; ++mask)
    {
        reduced[mask] = ancilla::partial_trace_fixed(rho, 3, mask, fermionic);
        entropies[mask] = detail::bits_from_nats(ancilla::entropy_from_small_rdm(reduced[mask]));
    }

    const double i_ab = entropies[1] + entropies[2] - entropies[3];
    const double i_ac = entropies[1] + entropies[4] - entropies[5];
    const double i_bc = entropies[2] + entropies[4] - entropies[6];
    const double tmi = entropies[1] + entropies[2] + entropies[4] - entropies[3] - entropies[5] - entropies[6] +
                       entropies[7];
    if (!std::isfinite(tmi) || !std::isfinite(i_ab) || !std::isfinite(i_ac) || !std::isfinite(i_bc))
    {
        throw std::runtime_error("Three-party information calculation produced a non-finite value.");
    }

    const double mean_single_purity = (ancilla::purity_from_small_rdm(reduced[1]) +
                                       ancilla::purity_from_small_rdm(reduced[2]) +
                                       ancilla::purity_from_small_rdm(reduced[4])) /
                                      3.0;
    return {tmi, (i_ab + i_ac + i_bc) / 3.0, ancilla::purity_from_small_rdm(reduced[7]), mean_single_purity};
}

inline PairMetrics two_party_metrics(const double *rho_ri, bool fermionic)
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

    return {std::max(0.0, mi), detail::negativity(rho, fermionic)};
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
