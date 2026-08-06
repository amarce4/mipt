// Ground-truth checks for every GMN/fGMN solver entry point in the tree.
//
// Two implementations of the same quantity coexist here: the MOSEK C API path
// in src/analysis/gmn_mosek.c (used by gmn.exe) and the MOSEK Fusion path in
// src/analysis/fgmn.cpp (used by mipt_probed.exe, dist_scaling.exe, fgmn.exe).
// They silently disagreed by 20-40% on mixed states for as long as both
// existed, because the Fusion side bounded only Q in the JMG normalization
//
//     W = P_M + PT_M(Q_M),   0 <= P_M <= 1,   0 <= Q_M <= 1
//
// and left P free. That relaxation is invisible on the states people reach for
// when spot-checking -- GHZ, product states and biseparable states all still
// come out exactly right -- so the regression has to be stated against a state
// where the genuine multipartite value differs from the bipartite one.
//
// |W> is that state: its genuine multipartite negativity is 0.443, while its
// minimum bipartite negativity is ((1/sqrt3 + sqrt(2/3))^2 - 1)/2 = 0.4714.
// An unbounded P lets the witness saturate the bipartite value, so a solver
// that reports 0.4714 for |W> has lost the P <= 1 constraint.
//
// Needs MOSEK but no GPU. Run with `make test-gmn`.

#include "mipt/analysis/gmn.hpp"
#include "mipt/analysis/fgmn.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace
{
using Complex = std::complex<double>;

int failures = 0;

// Genuine multipartite negativity of |W>, PPT-mixture definition. Solver
// tolerance dominates here, so this is pinned to 3 decimals.
constexpr double W_GMN = 0.4428;
// Minimum bipartite negativity of |W>. The value a solver returns when the
// P <= 1 normalization is missing -- must NOT be what we get.
constexpr double W_MIN_BIPARTITE_NEGATIVITY = 0.4714;

void expect_close(double actual, double expected, double tolerance, const char *label)
{
    if (!(std::abs(actual - expected) <= tolerance))
    {
        std::cerr << "FAIL " << label << ": expected " << expected << ", got " << actual
                  << " (tolerance " << tolerance << ")\n";
        ++failures;
    }
}

void expect_far_from(double actual, double forbidden, double margin, const char *label)
{
    if (std::abs(actual - forbidden) < margin)
    {
        std::cerr << "FAIL " << label << ": got " << actual << ", which is the forbidden value "
                  << forbidden << " (margin " << margin << ")\n";
        ++failures;
    }
}

// Row-major 8x8 complex matrix, interleaved real/imaginary -- the layout every
// solver entry point in this tree takes.
using Packed = std::array<double, 128>;

Packed pack(const std::vector<Complex> &rho)
{
    Packed out{};
    for (std::size_t i = 0; i < 64; ++i)
    {
        out[2 * i] = rho[i].real();
        out[2 * i + 1] = rho[i].imag();
    }
    return out;
}

std::vector<Complex> pure(const std::array<Complex, 8> &amplitudes)
{
    std::vector<Complex> rho(64);
    for (std::size_t i = 0; i < 8; ++i)
    {
        for (std::size_t j = 0; j < 8; ++j)
        {
            rho[i * 8 + j] = amplitudes[i] * std::conj(amplitudes[j]);
        }
    }
    return rho;
}

std::array<Complex, 8> ghz_amplitudes()
{
    const double s = 1.0 / std::sqrt(2.0);
    std::array<Complex, 8> v{};
    v[0] = s;
    v[7] = s;
    return v;
}

std::array<Complex, 8> w_amplitudes()
{
    const double s = 1.0 / std::sqrt(3.0);
    std::array<Complex, 8> v{};
    v[1] = s; // |001>
    v[2] = s; // |010>
    v[4] = s; // |100>
    return v;
}

// A Bell pair on AB with C left in |0>: entangled, but biseparable, so every
// genuine multipartite measure must vanish.
std::array<Complex, 8> bell_ab_amplitudes()
{
    const double s = 1.0 / std::sqrt(2.0);
    std::array<Complex, 8> v{};
    v[0] = s; // |000>
    v[6] = s; // |110>
    return v;
}

std::array<Complex, 8> product_amplitudes()
{
    std::array<Complex, 8> v{};
    v[0] = 1.0;
    return v;
}

// p|GHZ><GHZ| + (1-p) 1/8. PPT mixtures detect genuine tripartite entanglement
// exactly above p = 3/7.
std::vector<Complex> ghz_white_noise(double p)
{
    std::vector<Complex> rho = pure(ghz_amplitudes());
    for (auto &value : rho)
    {
        value *= p;
    }
    for (std::size_t i = 0; i < 8; ++i)
    {
        rho[i * 8 + i] += (1.0 - p) / 8.0;
    }
    return rho;
}

struct Solver
{
    const char *name;
    double (*evaluate)(const double *);
};

// The pure-state closed forms. Every basis element appearing in these states
// has definite particle number, and each state's support lies in a single
// parity sector, so the fermionic partial transpose reduces to the ordinary one
// and the fermionic solver must reproduce the same values. This is the part of
// the suite that applies to all three entry points.
void check_pure_state_closed_forms(const Solver &solver)
{
    const std::string prefix = std::string(solver.name) + " ";

    const Packed ghz = pack(pure(ghz_amplitudes()));
    expect_close(solver.evaluate(ghz.data()), 0.5, 1.0e-4, (prefix + "GHZ").c_str());

    // The regression that matters. Both assertions are needed: the first pins
    // the right answer, the second names the specific wrong one.
    const Packed w = pack(pure(w_amplitudes()));
    const double w_value = solver.evaluate(w.data());
    expect_close(w_value, W_GMN, 1.0e-3, (prefix + "W").c_str());
    expect_far_from(w_value, W_MIN_BIPARTITE_NEGATIVITY, 0.01,
                    (prefix + "W is not the bipartite negativity").c_str());

    const Packed bell = pack(pure(bell_ab_amplitudes()));
    expect_close(solver.evaluate(bell.data()), 0.0, 1.0e-4, (prefix + "Bell(AB) x |0>_C").c_str());

    const Packed product = pack(pure(product_amplitudes()));
    expect_close(solver.evaluate(product.data()), 0.0, 1.0e-4, (prefix + "|000>").c_str());
}

// White-noise mixtures separate the two criteria, so these are stated for the
// ordinary-trace solvers only.
//
// Mixing in 1/8 populates every occupation-basis state, so the fermionic
// partial transpose no longer agrees with the ordinary one and fGMN is a
// genuinely different quantity: it reports 0.076 at p=0.40 where GMN reports a
// PPT mixture, and 0.125 against GMN's 0.0625 at p=0.50. That is expected --
// parity superselection makes fermionic separability the stronger requirement,
// so more states count as entangled -- and there is no closed form to pin it
// against here, so it is deliberately left unchecked rather than pinned to a
// number read off the current implementation.
void check_white_noise_threshold(const Solver &solver)
{
    const std::string prefix = std::string(solver.name) + " ";

    // Below and above the PPT-mixture threshold p* = 3/7.
    const Packed below = pack(ghz_white_noise(0.40));
    expect_close(solver.evaluate(below.data()), 0.0, 1.0e-4, (prefix + "GHZ+noise p=0.40").c_str());

    const Packed above = pack(ghz_white_noise(0.50));
    expect_close(solver.evaluate(above.data()), 0.0625, 1.0e-3,
                 (prefix + "GHZ+noise p=0.50").c_str());
}

// The C API and Fusion GMN paths must agree, not merely each look plausible.
void check_paths_agree()
{
    const std::array<std::vector<Complex>, 4> states = {
        pure(ghz_amplitudes()), pure(w_amplitudes()), ghz_white_noise(0.6), ghz_white_noise(0.8)};

    for (std::size_t i = 0; i < states.size(); ++i)
    {
        const Packed rho = pack(states[i]);
        const double c_api = compute_gmn_mosek_complex_8x8(rho.data());
        const double fusion = compute_gmn_mosek_complex_8x8_cpp(rho.data());
        expect_close(fusion, c_api, 1.0e-3,
                     ("C API and Fusion GMN agree on state " + std::to_string(i)).c_str());
    }
}

} // namespace

int main()
{
    const std::array<Solver, 2> ordinary_trace = {
        Solver{"GMN C API", compute_gmn_mosek_complex_8x8},
        Solver{"GMN Fusion", compute_gmn_mosek_complex_8x8_cpp},
    };
    const Solver fermionic{"fGMN Fusion", compute_fgmn_mosek_8x8_cpp};

    for (const auto &solver : ordinary_trace)
    {
        check_pure_state_closed_forms(solver);
        check_white_noise_threshold(solver);
    }
    check_pure_state_closed_forms(fermionic);
    check_paths_agree();

    if (failures != 0)
    {
        std::cerr << failures << " GMN solver check(s) failed.\n";
        return EXIT_FAILURE;
    }
    std::cout << "All GMN/fGMN solver checks passed.\n";
    return EXIT_SUCCESS;
}
