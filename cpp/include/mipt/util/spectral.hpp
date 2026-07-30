#pragma once

// Small dense spectral helpers shared by the negativity paths.
//
// Everything here works directly on complex Hermitian matrices.  The code this
// replaced went through the real realification [[Re, -Im], [Im, Re]], which
// duplicates every eigenvalue and therefore did four times the arithmetic of a
// direct complex diagonalization for exactly the same spectrum.
//
// The diagonalization is a *cyclic* Jacobi sweep: every (p, q) pair is
// annihilated once per sweep, and sweeps repeat until the off-diagonal is
// negligible.  The previous implementations searched for the largest
// off-diagonal entry and then performed a single rotation, charging an O(N^2)
// scan per rotation and — more importantly — capping the total number of
// *rotations* at a value that reads like a sweep count.  fgmn.cpp allowed 128
// rotations for a 16x16, which needs several hundred to converge, so it
// silently returned a partially diagonalized matrix whenever the spectrum was
// not already nearly diagonal.  Non-convergence here is reported instead.

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>

namespace mipt::util
{
// Cyclic Jacobi converges quadratically; well-conditioned problems of the sizes
// used here (N <= 8) finish in under ten sweeps.  The cap exists to bound a
// pathological input, not to be reached.
inline constexpr int JACOBI_MAX_SWEEPS = 40;
inline constexpr double JACOBI_RELATIVE_TOLERANCE = 1.0e-15;

// Eigenvalues of a dense complex Hermitian matrix, row-major, by cyclic Jacobi.
//
// The matrix is taken by value and destroyed.  Returns false (leaving
// `eigenvalues` unspecified) if the sweeps did not converge, so callers can
// propagate a NaN rather than trust a half-diagonalized matrix.  Eigenvalues
// come out unordered.
template <std::size_t N>
inline bool hermitian_eigenvalues(std::array<std::complex<double>, N * N> a,
                                  std::array<double, N> &eigenvalues)
{
    static_assert(N >= 1, "hermitian_eigenvalues needs a non-empty matrix.");
    using Complex = std::complex<double>;

    auto at = [&a](std::size_t r, std::size_t c) -> Complex & {
        return a[r * N + c];
    };

    // Symmetrize away roundoff and force an exactly real diagonal, so that the
    // rotation below can treat the diagonal as real without drift.
    double scale = 0.0;
    for (std::size_t r = 0; r < N; ++r)
    {
        at(r, r) = Complex(at(r, r).real(), 0.0);
        scale = std::max(scale, std::abs(at(r, r).real()));
        for (std::size_t c = r + 1; c < N; ++c)
        {
            const Complex mean = 0.5 * (at(r, c) + std::conj(at(c, r)));
            at(r, c) = mean;
            at(c, r) = std::conj(mean);
            scale = std::max(scale, std::abs(mean));
        }
    }

    if (!std::isfinite(scale))
    {
        return false;
    }
    if (scale == 0.0)
    {
        eigenvalues.fill(0.0);
        return true;
    }
    const double tolerance = JACOBI_RELATIVE_TOLERANCE * scale;

    for (int sweep = 0; sweep <= JACOBI_MAX_SWEEPS; ++sweep)
    {
        double off_diagonal = 0.0;
        for (std::size_t r = 0; r + 1 < N; ++r)
        {
            for (std::size_t c = r + 1; c < N; ++c)
            {
                off_diagonal = std::max(off_diagonal, std::abs(at(r, c)));
            }
        }
        if (off_diagonal <= tolerance)
        {
            for (std::size_t r = 0; r < N; ++r)
            {
                eigenvalues[r] = at(r, r).real();
            }
            return true;
        }
        if (sweep == JACOBI_MAX_SWEEPS)
        {
            break;
        }

        for (std::size_t p = 0; p + 1 < N; ++p)
        {
            for (std::size_t q = p + 1; q < N; ++q)
            {
                const Complex apq = at(p, q);
                const double magnitude = std::abs(apq);
                if (magnitude <= tolerance)
                {
                    continue;
                }

                // Annihilate (p, q) with U = diag(1, conj(phase)) * [[c, s], [-s, c]]
                // restricted to the (p, q) block: the phase makes the pivot real,
                // then an ordinary Givens rotation zeroes it.  Writing
                // z = conj(phase), the columns transform as
                //     a'(k,p) = c a(k,p) - s z a(k,q)
                //     a'(k,q) = s a(k,p) + c z a(k,q)
                // and the 2x2 block picks up the usual real Jacobi updates with
                // |a(p,q)| in place of the off-diagonal entry.
                const Complex conj_phase = std::conj(apq / magnitude);
                const double app = at(p, p).real();
                const double aqq = at(q, q).real();
                const double theta = 0.5 * std::atan2(2.0 * magnitude, aqq - app);
                const double c = std::cos(theta);
                const double s = std::sin(theta);

                for (std::size_t k = 0; k < N; ++k)
                {
                    if (k == p || k == q)
                    {
                        continue;
                    }
                    const Complex akp = at(k, p);
                    const Complex akq = at(k, q);
                    const Complex new_kp = c * akp - s * conj_phase * akq;
                    const Complex new_kq = s * akp + c * conj_phase * akq;
                    at(k, p) = new_kp;
                    at(p, k) = std::conj(new_kp);
                    at(k, q) = new_kq;
                    at(q, k) = std::conj(new_kq);
                }

                at(p, p) = Complex(
                    c * c * app - 2.0 * s * c * magnitude + s * s * aqq, 0.0);
                at(q, q) = Complex(
                    s * s * app + 2.0 * s * c * magnitude + c * c * aqq, 0.0);
                at(p, q) = Complex(0.0, 0.0);
                at(q, p) = Complex(0.0, 0.0);
            }
        }
    }

    return false;
}

// Trace norm (sum of singular values) of a complex *Hermitian* matrix.
//
// For Hermitian input the singular values are the absolute eigenvalues, so no
// Gram matrix is needed.  That matters beyond the flop count: forming A^dagger A
// squares the condition number and halves the number of significant digits
// available to the small end of the spectrum.
template <std::size_t N>
inline bool hermitian_trace_norm(const std::array<std::complex<double>, N * N> &a,
                                 double &trace_norm)
{
    std::array<double, N> eigenvalues{};
    if (!hermitian_eigenvalues<N>(a, eigenvalues))
    {
        return false;
    }
    double total = 0.0;
    for (const double lambda : eigenvalues)
    {
        total += std::abs(lambda);
    }
    trace_norm = total;
    return true;
}

// Trace norm of a general (not necessarily Hermitian) complex matrix, via the
// eigenvalues of the Hermitian Gram matrix A^dagger A.  Required for the
// fermionic partial transpose, which is not Hermitian.
template <std::size_t N>
inline bool gram_trace_norm(const std::array<std::complex<double>, N * N> &a,
                            double &trace_norm)
{
    using Complex = std::complex<double>;

    std::array<Complex, N * N> gram{};
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = i; j < N; ++j)
        {
            Complex sum(0.0, 0.0);
            for (std::size_t k = 0; k < N; ++k)
            {
                sum += std::conj(a[k * N + i]) * a[k * N + j];
            }
            gram[i * N + j] = sum;
            gram[j * N + i] = std::conj(sum);
        }
    }

    std::array<double, N> eigenvalues{};
    if (!hermitian_eigenvalues<N>(gram, eigenvalues))
    {
        return false;
    }

    double total = 0.0;
    for (double lambda : eigenvalues)
    {
        // A^dagger A is PSD in exact arithmetic; clamp the roundoff-level
        // negative eigenvalues the squaring can introduce.
        if (lambda > 0.0)
        {
            total += std::sqrt(lambda);
        }
    }
    trace_norm = total;
    return true;
}
} // namespace mipt::util
