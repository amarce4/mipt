#pragma once

#include "mipt/util/geometry.hpp"

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
template <std::size_t N>
using RealMatrix = std::array<std::array<double, N>, N>;

inline double entropy_term(double lambda)
{
    constexpr double eps = 1.0e-15;
    return lambda > eps ? -lambda * std::log2(lambda) : 0.0;
}

template <std::size_t N>
inline std::array<double, N> jacobi_eigenvalues(RealMatrix<N> a)
{
    constexpr int max_sweeps = 256;
    constexpr double relative_tolerance = 1.0e-14;

    double scale = 1.0;
    for (std::size_t i = 0; i < N; ++i)
    {
        scale = std::max(scale, std::abs(a[i][i]));
        for (std::size_t j = i + 1; j < N; ++j)
        {
            const double sym = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = a[j][i] = sym;
            scale = std::max(scale, std::abs(sym));
        }
    }
    const double tolerance = relative_tolerance * scale;

    for (int sweep = 0; sweep < max_sweeps; ++sweep)
    {
        std::size_t p = 0;
        std::size_t q = 1;
        double max_offdiag = 0.0;
        for (std::size_t i = 0; i + 1 < N; ++i)
        {
            for (std::size_t j = i + 1; j < N; ++j)
            {
                const double value = std::abs(a[i][j]);
                if (value > max_offdiag)
                {
                    max_offdiag = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_offdiag <= tolerance)
        {
            break;
        }

        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];
        const double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(theta);
        const double s = std::sin(theta);

        for (std::size_t k = 0; k < N; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }
            const double akp = a[k][p];
            const double akq = a[k][q];
            const double new_kp = c * akp - s * akq;
            const double new_kq = s * akp + c * akq;
            a[k][p] = a[p][k] = new_kp;
            a[k][q] = a[q][k] = new_kq;
        }

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = a[q][p] = 0.0;
    }

    std::array<double, N> eigenvalues{};
    for (std::size_t i = 0; i < N; ++i)
    {
        eigenvalues[i] = a[i][i];
    }
    return eigenvalues;
}

template <std::size_t D>
inline RealMatrix<2 * D> complex_hermitian_realification(
    const std::array<Complex, D * D> &matrix)
{
    RealMatrix<2 * D> out{};
    for (std::size_t i = 0; i < D; ++i)
    {
        for (std::size_t j = 0; j < D; ++j)
        {
            const Complex value = matrix[i * D + j];
            out[i][j] = value.real();
            out[i][j + D] = -value.imag();
            out[i + D][j] = value.imag();
            out[i + D][j + D] = value.real();
        }
    }
    return out;
}

template <std::size_t D>
inline double von_neumann_entropy(const std::array<Complex, D * D> &rho)
{
    const auto eigenvalues = jacobi_eigenvalues(complex_hermitian_realification<D>(rho));
    double entropy = 0.0;
    for (double lambda : eigenvalues)
    {
        if (lambda < 0.0 && lambda > -1.0e-11)
        {
            lambda = 0.0;
        }
        entropy += entropy_term(lambda);
    }
    // The realification duplicates every eigenvalue.
    entropy *= 0.5;
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

inline double trace_norm(const Matrix4 &matrix)
{
    Matrix4 gram{};
    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 4; ++j)
        {
            Complex sum(0.0, 0.0);
            for (std::size_t k = 0; k < 4; ++k)
            {
                sum += std::conj(matrix[k * 4 + i]) * matrix[k * 4 + j];
            }
            gram[i * 4 + j] = sum;
        }
    }

    const auto eigenvalues = jacobi_eigenvalues(complex_hermitian_realification<4>(gram));
    double norm = 0.0;
    for (double lambda : eigenvalues)
    {
        if (lambda < 0.0 && lambda > -1.0e-11)
        {
            lambda = 0.0;
        }
        if (lambda > 0.0)
        {
            norm += std::sqrt(lambda);
        }
    }
    // The realification duplicates every singular value.
    return 0.5 * norm;
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

    double result = 0.5 * (trace_norm(partial_transpose) - 1.0);
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
