#pragma once

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
} // namespace mipt::dist
