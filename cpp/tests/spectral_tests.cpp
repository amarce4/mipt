// Dependency-free checks for mipt::util::hermitian_eigenvalues and the two
// trace-norm wrappers built on it.  These replaced three separate hand-rolled
// Jacobi loops (fgmn.cpp, dist_metrics.hpp, ancilla_mi.hpp), one of which
// capped its rotation count low enough that it returned partially diagonalized
// matrices, so the point of this file is to pin the spectrum hard: every test
// that can be is stated against a *known* answer rather than a self-consistency
// invariant.

#include "mipt/util/spectral.hpp"
#include "mipt/dist_metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace
{
using Complex = std::complex<double>;

int failures = 0;

void expect_close(double actual, double expected, double tolerance, const char *label)
{
    if (!(std::abs(actual - expected) <= tolerance))
    {
        std::cerr << "FAIL " << label << ": expected " << expected << ", got "
                  << actual << " (tolerance " << tolerance << ")\n";
        ++failures;
    }
}

void expect_true(bool condition, const char *label)
{
    if (!condition)
    {
        std::cerr << "FAIL " << label << '\n';
        ++failures;
    }
}

// Deterministic generator so a failure is always reproducible.
struct Rng
{
    std::uint64_t state;

    explicit Rng(std::uint64_t seed) : state(seed) {}

    std::uint64_t next()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    double uniform()
    {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }

    double normal()
    {
        // Box-Muller; the 1e-12 floor keeps log() away from zero.
        const double u1 = std::max(1.0e-12, uniform());
        const double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

template <std::size_t N>
using Matrix = std::array<Complex, N * N>;

// Haar-ish random unitary by Gram-Schmidt on a complex Gaussian matrix.  Exact
// Haar measure is not needed; what matters is that the eigenvectors are dense
// and generic, which is the case the old max-pivot loops handled worst.
template <std::size_t N>
Matrix<N> random_unitary(Rng &rng)
{
    Matrix<N> u{};
    for (std::size_t c = 0; c < N; ++c)
    {
        for (std::size_t r = 0; r < N; ++r)
        {
            u[r * N + c] = Complex(rng.normal(), rng.normal());
        }

        for (std::size_t prev = 0; prev < c; ++prev)
        {
            Complex overlap(0.0, 0.0);
            for (std::size_t r = 0; r < N; ++r)
            {
                overlap += std::conj(u[r * N + prev]) * u[r * N + c];
            }
            for (std::size_t r = 0; r < N; ++r)
            {
                u[r * N + c] -= overlap * u[r * N + prev];
            }
        }

        double norm = 0.0;
        for (std::size_t r = 0; r < N; ++r)
        {
            norm += std::norm(u[r * N + c]);
        }
        norm = std::sqrt(norm);
        for (std::size_t r = 0; r < N; ++r)
        {
            u[r * N + c] /= norm;
        }
    }
    return u;
}

// U diag(spectrum) U^dagger.
template <std::size_t N>
Matrix<N> from_spectrum(const Matrix<N> &u, const std::array<double, N> &spectrum)
{
    Matrix<N> a{};
    for (std::size_t r = 0; r < N; ++r)
    {
        for (std::size_t c = 0; c < N; ++c)
        {
            Complex sum(0.0, 0.0);
            for (std::size_t k = 0; k < N; ++k)
            {
                sum += u[r * N + k] * spectrum[k] * std::conj(u[c * N + k]);
            }
            a[r * N + c] = sum;
        }
    }
    return a;
}

// The decisive test: conjugate a known spectrum by a dense random unitary and
// check that the solver recovers it.  A capped-rotation Jacobi fails this.
template <std::size_t N>
void check_known_spectrum(Rng &rng, int trials, const char *label)
{
    for (int trial = 0; trial < trials; ++trial)
    {
        std::array<double, N> spectrum{};
        for (std::size_t i = 0; i < N; ++i)
        {
            spectrum[i] = rng.normal();
        }
        // A deliberately wide dynamic range plus an exact degeneracy: both are
        // cases where a truncated sweep leaves visible off-diagonal mass.
        spectrum[0] *= 1.0e4;
        if (N > 2)
        {
            spectrum[N - 1] = spectrum[N - 2];
        }

        const Matrix<N> u = random_unitary<N>(rng);
        const Matrix<N> a = from_spectrum<N>(u, spectrum);

        std::array<double, N> computed{};
        if (!mipt::util::hermitian_eigenvalues<N>(a, computed))
        {
            std::cerr << "FAIL " << label << ": solver reported non-convergence\n";
            ++failures;
            return;
        }

        std::array<double, N> expected = spectrum;
        std::sort(expected.begin(), expected.end());
        std::sort(computed.begin(), computed.end());

        double magnitude = 0.0;
        for (std::size_t i = 0; i < N; ++i)
        {
            magnitude = std::max(magnitude, std::abs(expected[i]));
        }
        for (std::size_t i = 0; i < N; ++i)
        {
            expect_close(computed[i], expected[i], 1.0e-9 * magnitude, label);
        }
    }
}

// For a Hermitian matrix the singular values are the absolute eigenvalues, so
// the direct route and the Gram route must agree.  This is the cross-check that
// justifies skipping the Gram matrix in the bosonic negativity paths.
template <std::size_t N>
void check_trace_norm_routes(Rng &rng, int trials, const char *label)
{
    for (int trial = 0; trial < trials; ++trial)
    {
        std::array<double, N> spectrum{};
        for (std::size_t i = 0; i < N; ++i)
        {
            spectrum[i] = rng.normal();
        }
        const Matrix<N> a = from_spectrum<N>(random_unitary<N>(rng), spectrum);

        double direct = 0.0;
        double via_gram = 0.0;
        expect_true(mipt::util::hermitian_trace_norm<N>(a, direct), label);
        expect_true(mipt::util::gram_trace_norm<N>(a, via_gram), label);

        double reference = 0.0;
        for (const double lambda : spectrum)
        {
            reference += std::abs(lambda);
        }
        expect_close(direct, reference, 1.0e-9 * std::max(1.0, reference), label);
        expect_close(via_gram, reference, 1.0e-7 * std::max(1.0, reference), label);
    }
}

void check_trivial_cases()
{
    std::array<Complex, 4> zero{};
    std::array<double, 2> eigenvalues{};
    expect_true(mipt::util::hermitian_eigenvalues<2>(zero, eigenvalues),
                "zero matrix converges");
    expect_close(eigenvalues[0], 0.0, 0.0, "zero matrix eigenvalue 0");
    expect_close(eigenvalues[1], 0.0, 0.0, "zero matrix eigenvalue 1");

    // Already diagonal: must return immediately without rotating.
    std::array<Complex, 9> diagonal{};
    diagonal[0] = Complex(3.0, 0.0);
    diagonal[4] = Complex(-1.0, 0.0);
    diagonal[8] = Complex(0.5, 0.0);
    std::array<double, 3> diagonal_eigs{};
    expect_true(mipt::util::hermitian_eigenvalues<3>(diagonal, diagonal_eigs),
                "diagonal matrix converges");
    std::sort(diagonal_eigs.begin(), diagonal_eigs.end());
    expect_close(diagonal_eigs[0], -1.0, 1.0e-14, "diagonal eigenvalue 0");
    expect_close(diagonal_eigs[1], 0.5, 1.0e-14, "diagonal eigenvalue 1");
    expect_close(diagonal_eigs[2], 3.0, 1.0e-14, "diagonal eigenvalue 2");

    // Pauli Y: eigenvalues +/-1, and purely imaginary off-diagonals are exactly
    // the case a real-symmetric solver cannot see without realification.
    std::array<Complex, 4> pauli_y{};
    pauli_y[1] = Complex(0.0, -1.0);
    pauli_y[2] = Complex(0.0, 1.0);
    std::array<double, 2> y_eigs{};
    expect_true(mipt::util::hermitian_eigenvalues<2>(pauli_y, y_eigs),
                "Pauli Y converges");
    std::sort(y_eigs.begin(), y_eigs.end());
    expect_close(y_eigs[0], -1.0, 1.0e-14, "Pauli Y eigenvalue -1");
    expect_close(y_eigs[1], 1.0, 1.0e-14, "Pauli Y eigenvalue +1");
}

// Negativity of a maximally entangled pair is 1/2 across the cut, and of a
// product state it is 0.  Checked at D=8 through the same partial-transpose
// index arithmetic fgmn.cpp uses, so this pins the physical answer the GMN
// zero-prefilter depends on.
void check_ghz_negativity()
{
    constexpr int D = 8;
    std::array<Complex, D> psi{};
    psi[0] = Complex(1.0 / std::sqrt(2.0), 0.0);
    psi[7] = Complex(1.0 / std::sqrt(2.0), 0.0);

    std::array<Complex, D * D> rho{};
    for (int r = 0; r < D; ++r)
    {
        for (int c = 0; c < D; ++c)
        {
            rho[r * D + c] = psi[r] * std::conj(psi[c]);
        }
    }

    for (int subsystem = 0; subsystem < 3; ++subsystem)
    {
        const int mask = 1 << subsystem;
        std::array<Complex, D * D> pt{};
        for (int r = 0; r < D; ++r)
        {
            for (int c = 0; c < D; ++c)
            {
                const int source_row = (r & ~mask) | (c & mask);
                const int source_col = (c & ~mask) | (r & mask);
                pt[r * D + c] = rho[source_row * D + source_col];
            }
        }

        double norm = 0.0;
        expect_true(mipt::util::hermitian_trace_norm<D>(pt, norm),
                    "GHZ partial transpose converges");
        expect_close(0.5 * (norm - 1.0), 0.5, 1.0e-12, "GHZ bipartite negativity");
    }

    // |000>: product, so every cut is separable.
    std::array<Complex, D * D> product{};
    product[0] = Complex(1.0, 0.0);
    double product_norm = 0.0;
    expect_true(mipt::util::hermitian_trace_norm<D>(product, product_norm),
                "product state converges");
    expect_close(0.5 * (product_norm - 1.0), 0.0, 1.0e-12, "product negativity");
}

// dist_metrics is the consumer that changed shape most, so check it end to end
// against analytically known two-site answers.
void check_two_site_metrics()
{
    // Bell pair: MI = 2 bits, negativity = 1/2.
    std::array<Complex, 4> psi{};
    psi[0] = Complex(1.0 / std::sqrt(2.0), 0.0);
    psi[3] = Complex(1.0 / std::sqrt(2.0), 0.0);

    std::array<double, 32> rho_ri{};
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            const Complex value = psi[r] * std::conj(psi[c]);
            rho_ri[2 * (r * 4 + c)] = value.real();
            rho_ri[2 * (r * 4 + c) + 1] = value.imag();
        }
    }

    const auto bell = mipt::dist::two_party_metrics(rho_ri.data(), false);
    expect_close(bell.mi, 2.0, 1.0e-10, "Bell mutual information (bits)");
    expect_close(bell.mn, 0.5, 1.0e-10, "Bell negativity");

    // Maximally mixed: MI = 0, negativity = 0.
    std::array<double, 32> mixed{};
    for (int i = 0; i < 4; ++i)
    {
        mixed[2 * (i * 4 + i)] = 0.25;
    }
    const auto separable = mipt::dist::two_party_metrics(mixed.data(), false);
    expect_close(separable.mi, 0.0, 1.0e-10, "mixed mutual information");
    expect_close(separable.mn, 0.0, 1.0e-10, "mixed negativity");
}
} // namespace

int main()
{
    Rng rng(0x9E3779B97F4A7C15ULL);

    check_trivial_cases();
    check_known_spectrum<2>(rng, 200, "known spectrum N=2");
    check_known_spectrum<3>(rng, 200, "known spectrum N=3");
    check_known_spectrum<4>(rng, 200, "known spectrum N=4");
    check_known_spectrum<8>(rng, 200, "known spectrum N=8");
    check_trace_norm_routes<4>(rng, 100, "trace-norm routes N=4");
    check_trace_norm_routes<8>(rng, 100, "trace-norm routes N=8");
    check_ghz_negativity();
    check_two_site_metrics();

    if (failures != 0)
    {
        std::cerr << failures << " spectral test failure(s).\n";
        return 1;
    }
    std::cout << "All spectral tests passed.\n";
    return 0;
}
