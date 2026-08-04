// Checks for mipt::front::block_metrics and the small-block partial trace it
// shares with the ancilla path.
//
// The mode-5 protocol reads a velocity off the *shape* of these quantities in
// (x, tau), so a constant offset or a swapped basis would not show up as an
// obviously wrong number in a production run -- it would show up as a slightly
// wrong exponent. Everything below is therefore pinned against a closed-form
// value where one exists, and against an inequality that must hold state by
// state where one does not.
//
// Conventions under test, both of which the runner depends on:
//   * the reference is the highest local mode, so index = r * 2^width + b;
//   * narrowing a width-3 block to width w traces out the *outer* sites and
//     leaves the reference on bit w.

#include "mipt/front_metrics.hpp"
#include "mipt/small_rdm.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
using Complex = std::complex<double>;
using mipt::ancilla::SmallRdm;
using mipt::front::BlockMetrics;

int failures = 0;
constexpr double LOG2 = 0.69314718055994530942;

void expect_close(double actual, double expected, double tolerance, const char *label)
{
    if (!(std::abs(actual - expected) <= tolerance))
    {
        std::cerr << "FAIL " << label << ": expected " << expected << ", got " << actual << " (tolerance " << tolerance
                  << ")\n";
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
        // Box-Muller; the tail cut is far enough out not to matter here.
        const double u = std::max(uniform(), 1.0e-12);
        const double v = uniform();
        return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v);
    }

    std::uint64_t state;
};

SmallRdm from_pure_state(const std::vector<Complex> &amplitudes)
{
    const int dimension = static_cast<int>(amplitudes.size());
    double norm = 0.0;
    for (const Complex &amplitude : amplitudes)
    {
        norm += std::norm(amplitude);
    }
    SmallRdm rho(dimension);
    for (int row = 0; row < dimension; ++row)
    {
        for (int col = 0; col < dimension; ++col)
        {
            rho(row, col) = amplitudes[static_cast<std::size_t>(row)] *
                            std::conj(amplitudes[static_cast<std::size_t>(col)]) / norm;
        }
    }
    return rho;
}

std::vector<Complex> random_amplitudes(Rng &rng, int dimension)
{
    std::vector<Complex> amplitudes(static_cast<std::size_t>(dimension));
    for (Complex &amplitude : amplitudes)
    {
        amplitude = Complex(rng.normal(), rng.normal());
    }
    return amplitudes;
}

// A Bell pair between the reference and the innermost block site, with every
// remaining block site maximally mixed and uncorrelated.
SmallRdm bell_plus_mixed(int width)
{
    const int block_dimension = 1 << width;
    SmallRdm rho(2 * block_dimension);
    const int spectators = 1 << (width - 1); // states of block sites 1..width-1
    const double weight = 0.5 / static_cast<double>(spectators);
    for (int spectator = 0; spectator < spectators; ++spectator)
    {
        // Block index is (spectator << 1) | b0, so the Bell partner is bit 0.
        const int b0 = spectator << 1;
        const int b1 = b0 | 1;
        const int r0 = 0 * block_dimension + b0; // |0>_R |0>_b0
        const int r1 = 1 * block_dimension + b1; // |1>_R |1>_b0
        rho(r0, r0) = weight;
        rho(r1, r1) = weight;
        rho(r0, r1) = weight;
        rho(r1, r0) = weight;
    }
    return rho;
}

void check_bell(int width)
{
    const std::string label = "Bell width=" + std::to_string(width);
    const BlockMetrics metrics = mipt::front::block_metrics(bell_plus_mixed(width), width, false);
    const double spectator_entropy = static_cast<double>(width - 1) * LOG2;

    expect_close(metrics.joint_entropy, spectator_entropy, 1e-10, (label + " S(RB)").c_str());
    expect_close(metrics.block_entropy, spectator_entropy + LOG2, 1e-10, (label + " S(B)").c_str());
    expect_close(metrics.reference_entropy, LOG2, 1e-10, (label + " S(R)").c_str());
    expect_close(metrics.mutual_information, 2.0 * LOG2, 1e-9, (label + " I(R:B)").c_str());
    expect_close(metrics.coherent_information, LOG2, 1e-9, (label + " I_c").c_str());
    // A maximally entangled pair carries a full bit in every basis.
    expect_close(metrics.classical_x, LOG2, 1e-9, (label + " C_X").c_str());
    expect_close(metrics.classical_y, LOG2, 1e-9, (label + " C_Y").c_str());
    expect_close(metrics.classical_z, LOG2, 1e-9, (label + " C_Z").c_str());
    expect_close(metrics.negativity, 0.5, 1e-10, (label + " negativity").c_str());
    expect_close(metrics.log_negativity, LOG2, 1e-10, (label + " log negativity").c_str());

    // The fermionic partial transpose gives the same value for a parity-even
    // Bell state, but must still run and stay in range.
    const BlockMetrics fermionic = mipt::front::block_metrics(bell_plus_mixed(width), width, true);
    expect_close(fermionic.negativity, 0.5, 1e-9, (label + " fermionic negativity").c_str());
    expect_close(fermionic.mutual_information, 2.0 * LOG2, 1e-9, (label + " fermionic I(R:B)").c_str());
}

// Perfectly correlated in Z and nothing else: the classical front is there, the
// entanglement front is not. This is exactly the state the protocol's
// "classical only" band is supposed to detect.
void check_classical_correlation()
{
    SmallRdm rho(4);
    rho(0, 0) = 0.5; // |0>_R |0>_B
    rho(3, 3) = 0.5; // |1>_R |1>_B
    const BlockMetrics metrics = mipt::front::block_metrics(rho, 1, false);

    expect_close(metrics.classical_z, LOG2, 1e-10, "classical C_Z");
    expect_close(metrics.classical_x, 0.0, 1e-10, "classical C_X");
    expect_close(metrics.classical_y, 0.0, 1e-10, "classical C_Y");
    expect_close(metrics.classical_max, LOG2, 1e-10, "classical C_max");
    expect_close(metrics.mutual_information, LOG2, 1e-9, "classical I(R:B)");
    expect_close(metrics.negativity, 0.0, 1e-10, "classical negativity");
    expect_close(metrics.coherent_information, 0.0, 1e-9, "classical I_c");
}

// Nothing has arrived yet: the reference is still maximally mixed and
// uncorrelated. Every signal must be identically zero, and the coherent
// information maximally negative.
void check_uncorrelated()
{
    SmallRdm rho(8);
    for (int i = 0; i < 8; ++i)
    {
        rho(i, i) = 0.125;
    }
    const BlockMetrics metrics = mipt::front::block_metrics(rho, 2, false);

    expect_close(metrics.classical_x, 0.0, 1e-10, "uncorrelated C_X");
    expect_close(metrics.classical_y, 0.0, 1e-10, "uncorrelated C_Y");
    expect_close(metrics.classical_z, 0.0, 1e-10, "uncorrelated C_Z");
    expect_close(metrics.mutual_information, 0.0, 1e-9, "uncorrelated I(R:B)");
    expect_close(metrics.negativity, 0.0, 1e-10, "uncorrelated negativity");
    expect_close(metrics.coherent_information, -LOG2, 1e-9, "uncorrelated I_c");
}

// GHZ over the reference and a two-site block: the R|B cut is maximally
// entangled, but neither block site alone is.
void check_ghz()
{
    std::vector<Complex> amplitudes(8, Complex(0.0, 0.0));
    amplitudes[0] = Complex(1.0, 0.0); // |000>
    amplitudes[7] = Complex(1.0, 0.0); // |111>
    const SmallRdm rho = from_pure_state(amplitudes);
    const BlockMetrics metrics = mipt::front::block_metrics(rho, 2, false);

    expect_close(metrics.joint_entropy, 0.0, 1e-9, "GHZ S(RB)");
    expect_close(metrics.mutual_information, 2.0 * LOG2, 1e-9, "GHZ I(R:B)");
    expect_close(metrics.negativity, 0.5, 1e-9, "GHZ R|B negativity");
    expect_close(metrics.classical_z, LOG2, 1e-9, "GHZ C_Z");
}

// Narrowing must agree with tracing the outer sites out of the state directly.
// This is the step that lets one width-3 RDM serve all three widths, and a bug
// here would silently mix up which sites a width-1 curve refers to.
void check_narrowing(Rng &rng)
{
    constexpr int full_width = 3;
    for (int trial = 0; trial < 50; ++trial)
    {
        const auto amplitudes = random_amplitudes(rng, 16);
        const SmallRdm full = from_pure_state(amplitudes);

        for (int width = 1; width <= 2; ++width)
        {
            const unsigned mask = ((1u << width) - 1u) | (1u << full_width);
            const SmallRdm narrowed =
                mipt::ancilla::partial_trace_fixed(full, full_width + 1, mask, false);

            // Independent reference: sum over the traced-out block sites, with
            // the reference relocated to bit `width`.
            const int kept = 1 << (width + 1);
            const int traced = 1 << (full_width - width);
            SmallRdm expected(kept);
            for (int r = 0; r < 2; ++r)
            {
                for (int inner_row = 0; inner_row < (1 << width); ++inner_row)
                {
                    for (int inner_col = 0; inner_col < (1 << width); ++inner_col)
                    {
                        Complex value(0.0, 0.0);
                        for (int outer = 0; outer < traced; ++outer)
                        {
                            const int row = (r << full_width) | (outer << width) | inner_row;
                            const int col = (r << full_width) | (outer << width) | inner_col;
                            value += full(row, col);
                        }
                        expected((r << width) | inner_row, (r << width) | inner_col) = value;
                    }
                }
            }
            // Only the reference-diagonal blocks were checked above; the
            // off-diagonal ones follow the same pattern.
            for (int r = 0; r < 2; ++r)
            {
                for (int s = 0; s < 2; ++s)
                {
                    if (r == s)
                    {
                        continue;
                    }
                    for (int inner_row = 0; inner_row < (1 << width); ++inner_row)
                    {
                        for (int inner_col = 0; inner_col < (1 << width); ++inner_col)
                        {
                            Complex value(0.0, 0.0);
                            for (int outer = 0; outer < traced; ++outer)
                            {
                                const int row = (r << full_width) | (outer << width) | inner_row;
                                const int col = (s << full_width) | (outer << width) | inner_col;
                                value += full(row, col);
                            }
                            expected((r << width) | inner_row, (s << width) | inner_col) = value;
                        }
                    }
                }
            }

            double largest = 0.0;
            for (int row = 0; row < kept; ++row)
            {
                for (int col = 0; col < kept; ++col)
                {
                    largest = std::max(largest, std::abs(narrowed(row, col) - expected(row, col)));
                }
            }
            expect_true(largest < 1e-12, "narrowed block RDM matches a direct partial trace");
        }
    }
}

// Inequalities that hold for every state, checked on random pure and mixed
// inputs at all three widths.
void check_invariants(Rng &rng)
{
    for (int width = 1; width <= 3; ++width)
    {
        const int dimension = 1 << (width + 1);
        for (int trial = 0; trial < 60; ++trial)
        {
            // Mix a few random pure states so the joint state is not always
            // pure and the Holevo terms have something to do.
            SmallRdm rho(dimension);
            const int components = 1 + static_cast<int>(rng.next() % 3u);
            for (int component = 0; component < components; ++component)
            {
                const SmallRdm pure = from_pure_state(random_amplitudes(rng, dimension));
                for (int row = 0; row < dimension; ++row)
                {
                    for (int col = 0; col < dimension; ++col)
                    {
                        rho(row, col) += pure(row, col) / static_cast<double>(components);
                    }
                }
            }

            for (const bool fermionic : {false, true})
            {
                const BlockMetrics metrics = mipt::front::block_metrics(rho, width, fermionic, true);
                const double slack = 1e-8;

                // The Pauli axes are three points of the landscape J maximizes
                // over. J <= I is only a theorem for the qubit trace: the
                // fermionic reference marginal is not the marginal the Holevo
                // terms saw, so there the difference is clamped instead.
                expect_true(metrics.classical_optimized >= metrics.classical_max - slack, "J >= C_max");
                expect_true(metrics.discord >= 0.0, "discord is nonnegative");
                if (!fermionic)
                {
                    expect_true(metrics.classical_optimized <= metrics.mutual_information + slack, "J <= I(R:B)");
                    expect_close(metrics.discord, metrics.mutual_information - metrics.classical_optimized, 1e-12,
                                 "D = I - J");
                }

                expect_true(metrics.mutual_information >= -slack, "I(R:B) is nonnegative");
                expect_true(metrics.negativity >= -slack, "negativity is nonnegative");
                // Holevo bound: the accessible classical information can never
                // exceed the total correlation.
                expect_true(metrics.classical_x <= metrics.mutual_information + slack, "C_X <= I(R:B)");
                expect_true(metrics.classical_y <= metrics.mutual_information + slack, "C_Y <= I(R:B)");
                expect_true(metrics.classical_z <= metrics.mutual_information + slack, "C_Z <= I(R:B)");
                // A single reference qubit cannot carry more than one bit.
                expect_true(metrics.classical_max <= LOG2 + slack, "C_max <= log 2");
                expect_close(metrics.classical_mean,
                             (metrics.classical_x + metrics.classical_y + metrics.classical_z) / 3.0, 1e-12,
                             "C_mean is the mean of the three bases");
                expect_close(metrics.coherent_information, metrics.block_entropy - metrics.joint_entropy, 1e-12,
                             "I_c = S(B) - S(RB)");
                expect_close(metrics.log_negativity, std::log1p(2.0 * metrics.negativity), 1e-12,
                             "E_N = log(1 + 2N)");
            }
        }
    }
}

double binary_entropy(double probability)
{
    if (probability <= 0.0 || probability >= 1.0)
    {
        return 0.0;
    }
    return -probability * std::log(probability) - (1.0 - probability) * std::log(1.0 - probability);
}

// Rotate the reference qubit by the unitary whose columns are the eigenvectors
// of n.sigma, leaving the block alone.
SmallRdm rotate_reference(const SmallRdm &rho, int width, double theta, double phi)
{
    const auto basis = mipt::front::detail::measurement_basis(theta, phi);
    const int block_dimension = 1 << width;
    SmallRdm rotated(rho.dimension);
    for (int r = 0; r < 2; ++r)
    {
        for (int rp = 0; rp < 2; ++rp)
        {
            for (int s = 0; s < 2; ++s)
            {
                for (int sp = 0; sp < 2; ++sp)
                {
                    // U(r,s) is component r of the s-th eigenvector.
                    const Complex weight = basis[static_cast<std::size_t>(s)][static_cast<std::size_t>(r)] *
                                           std::conj(basis[static_cast<std::size_t>(sp)][static_cast<std::size_t>(rp)]);
                    if (weight == Complex(0.0, 0.0))
                    {
                        continue;
                    }
                    for (int b = 0; b < block_dimension; ++b)
                    {
                        for (int bp = 0; bp < block_dimension; ++bp)
                        {
                            rotated(r * block_dimension + b, rp * block_dimension + bp) +=
                                weight * rho(s * block_dimension + b, sp * block_dimension + bp);
                        }
                    }
                }
            }
        }
    }
    return rotated;
}

// The optimized classical correlation J and the discord D = I - J.
//
// Three closed forms, each pinning a different failure mode:
//
//   * a state whose optimal measurement axis is deliberately *not* X, Y or Z.
//     J is invariant under a local unitary on the measured party, so rotating a
//     perfectly Z-correlated state leaves J = log 2 and D = 0 exactly while
//     driving C_max strictly below log 2. Nothing but a genuine search over the
//     Bloch sphere recovers that, so this is what separates J from C_max.
//   * a Bell pair, where J = log 2 and the discord takes the other half of
//     I = 2 log 2.
//   * a Werner state, whose Bell-diagonal classical correlation is
//     log 2 - H((1+w)/2) with H the binary entropy. Its optimum is isotropic,
//     so it checks the *value* the search converges to rather than its
//     direction.
void check_discord()
{
    // Perfectly correlated in Z, then rotated onto a generic axis.
    SmallRdm classical(4);
    classical(0, 0) = 0.5;
    classical(3, 3) = 0.5;
    const SmallRdm tilted = rotate_reference(classical, 1, 0.9, 2.0);
    const BlockMetrics rotated = mipt::front::block_metrics(tilted, 1, false, true);
    expect_close(rotated.mutual_information, LOG2, 1e-9, "tilted classical I(A:B)");
    expect_close(rotated.classical_optimized, LOG2, 1e-6, "tilted classical J");
    expect_close(rotated.discord, 0.0, 1e-6, "tilted classical discord vanishes");
    // If this ever fails the state stopped being a hard case and the test above
    // stopped testing the search.
    expect_true(rotated.classical_max < LOG2 - 0.01, "tilted state defeats the Pauli axes");
    expect_true(rotated.classical_optimized >= rotated.classical_max - 1e-12, "J >= C_max");

    // Unoptimized runs must report "not measured", not zero.
    const BlockMetrics unoptimized = mipt::front::block_metrics(tilted, 1, false);
    expect_true(!std::isfinite(unoptimized.classical_optimized), "J is NaN without the optimization");
    expect_true(!std::isfinite(unoptimized.discord), "discord is NaN without the optimization");

    for (int width = 1; width <= 3; ++width)
    {
        const std::string label = "Bell width=" + std::to_string(width);
        const BlockMetrics bell = mipt::front::block_metrics(bell_plus_mixed(width), width, false, true);
        expect_close(bell.classical_optimized, LOG2, 1e-6, (label + " J").c_str());
        expect_close(bell.discord, LOG2, 1e-6, (label + " discord").c_str());
    }

    for (const double w : {0.2, 0.5, 0.8, 0.95})
    {
        // (1-w) I/4 + w |Phi+><Phi+|, with |Phi+> = (|00> + |11>)/sqrt(2).
        SmallRdm werner(4);
        for (int i = 0; i < 4; ++i)
        {
            werner(i, i) = 0.25 * (1.0 - w);
        }
        werner(0, 0) += 0.5 * w;
        werner(3, 3) += 0.5 * w;
        werner(0, 3) += 0.5 * w;
        werner(3, 0) += 0.5 * w;

        const BlockMetrics metrics = mipt::front::block_metrics(werner, 1, false, true);
        const double expected_j = LOG2 - binary_entropy(0.5 * (1.0 + w));
        const double joint = -(0.25 * (1.0 + 3.0 * w) * std::log(0.25 * (1.0 + 3.0 * w)) +
                               0.75 * (1.0 - w) * std::log(0.25 * (1.0 - w)));
        const std::string label = "Werner w=" + std::to_string(w);
        expect_close(metrics.mutual_information, 2.0 * LOG2 - joint, 1e-8, (label + " I(A:B)").c_str());
        expect_close(metrics.classical_optimized, expected_j, 1e-6, (label + " J").c_str());
        expect_close(metrics.discord, 2.0 * LOG2 - joint - expected_j, 1e-6, (label + " discord").c_str());
    }
}

// For a pure joint state the two marginals share a spectrum, so I = 2 S(R) and
// I_c = S(R). This pins the entropy bookkeeping independently of the Holevo
// terms.
void check_pure_state_identities(Rng &rng)
{
    for (int width = 1; width <= 3; ++width)
    {
        for (int trial = 0; trial < 40; ++trial)
        {
            const SmallRdm rho = from_pure_state(random_amplitudes(rng, 1 << (width + 1)));
            const BlockMetrics metrics = mipt::front::block_metrics(rho, width, false);
            expect_close(metrics.joint_entropy, 0.0, 1e-8, "pure S(RB) vanishes");
            expect_close(metrics.mutual_information, 2.0 * metrics.reference_entropy, 1e-7,
                         "pure I(R:B) = 2 S(R)");
            expect_close(metrics.coherent_information, metrics.reference_entropy, 1e-7, "pure I_c = S(R)");
            expect_close(metrics.block_entropy, metrics.reference_entropy, 1e-7, "pure S(B) = S(R)");
        }
    }
}
} // namespace

int main()
{
    Rng rng(0xD1B54A32D192ED03ULL);

    for (int width = 1; width <= 3; ++width)
    {
        check_bell(width);
    }
    check_classical_correlation();
    check_uncorrelated();
    check_ghz();
    check_discord();
    check_narrowing(rng);
    check_invariants(rng);
    check_pure_state_identities(rng);

    if (failures != 0)
    {
        std::cerr << failures << " front-metric test failure(s).\n";
        return 1;
    }
    std::cout << "All front-metric tests passed.\n";
    return 0;
}
