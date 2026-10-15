// Host tests for the gate diagnostics in analysis/gate_entanglement.hpp and
// the way rppu_layer.hpp attaches them to a drawn gate.
//
// What is pinned, and why each one is a closed form rather than a regression
// value:
//
//   * The two ends of the entangling-power scale. Every local unitary, the
//     SWAP, and every local unitary times SWAP have e_p = 0 exactly; CNOT and
//     CZ sit at the maximum 2/9 (normalized 1). A sign or factor slip in
//     either E(U) or E(US) moves one of these, and nothing else in the tree
//     would notice -- the diagnostic is a continuous number nobody can eyeball.
//   * Local-unitary invariance. e_p is a property of the nonlocal content of
//     U only, so dressing a gate with random single-qubit unitaries on both
//     sides must leave it unchanged to rounding. This is what catches a
//     realignment that cuts along the wrong pair of indices.
//   * The fourth-moment shortcut. sum s^4 is computed as ||R R^dag||_F^2 with
//     no eigensolver; here it is compared against the singular values computed
//     independently, as eigenvalues of R^dag R by the cyclic Jacobi in
//     util/spectral.hpp.
//   * The draw. A drawn RPPU gate stores its diagnostic from the 2x2 blocks
//     *before* they are decomposed into U3 parameters; the engine applies the
//     matrix rebuilt *from* those parameters. The two must describe the same
//     gate, and the block mixings must equal sin^2(theta) of the stored U3.

#include "mipt/analysis/gate_entanglement.hpp"
#include "mipt/circuits/rppu_layer.hpp"
#include "mipt/util/spectral.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <string>

namespace
{
int failures = 0;

void expect(bool condition, const std::string &message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

using mipt::GateMatrix2;
using mipt::GateMatrix4;
using Complex = std::complex<double>;

constexpr double MAX_EP = 2.0 / 9.0;

GateMatrix4 identity4()
{
    GateMatrix4 m{};
    for (int k = 0; k < 4; ++k)
    {
        m[k * 4 + k] = 1.0;
    }
    return m;
}

GateMatrix2 identity2() { return {Complex(1.0), Complex(0.0), Complex(0.0), Complex(1.0)}; }

// CNOT with control on bit 0 and target on bit 1, in the bit-0-fastest layout.
GateMatrix4 cnot()
{
    GateMatrix4 m{};
    m[0 * 4 + 0] = 1.0; // |00> -> |00>
    m[2 * 4 + 2] = 1.0; // |0,1> (index 2: bit1 set) -> itself
    m[3 * 4 + 1] = 1.0; // |1,0> (index 1) -> |1,1> (index 3)
    m[1 * 4 + 3] = 1.0;
    return m;
}

GateMatrix4 cz()
{
    GateMatrix4 m = identity4();
    m[3 * 4 + 3] = -1.0;
    return m;
}

GateMatrix4 dagger(const GateMatrix4 &u)
{
    GateMatrix4 out{};
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            out[r * 4 + c] = std::conj(u[c * 4 + r]);
        }
    }
    return out;
}

double unitarity_error(const GateMatrix4 &u)
{
    const GateMatrix4 product = mipt::multiply(dagger(u), u);
    double worst = 0.0;
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            worst = std::max(worst, std::abs(product[r * 4 + c] - (r == c ? 1.0 : 0.0)));
        }
    }
    return worst;
}

GateMatrix4 random_local(std::mt19937 &rng)
{
    return mipt::kron(mipt::haar_unitary_2(rng), mipt::haar_unitary_2(rng));
}

// A random parity-preserving gate in the engine layout, from two Haar blocks.
GateMatrix4 random_parity_preserving(std::mt19937 &rng)
{
    const GateMatrix2 even = mipt::haar_unitary_2(rng);
    const GateMatrix2 odd = mipt::haar_unitary_2(rng);
    return mipt::parity_preserving_matrix(even, odd);
}

// sum_alpha s_alpha^4 of the realigned matrix, from its singular values: the
// eigenvalues of R^dag R are s^2, found by a Jacobi that shares no code with
// the Frobenius shortcut under test.
double fourth_moment_from_spectrum(const GateMatrix4 &u, bool &converged)
{
    const GateMatrix4 r = mipt::realign(u);
    const GateMatrix4 gram = mipt::multiply(dagger(r), r);
    std::array<double, 4> eigenvalues{};
    converged = mipt::util::hermitian_eigenvalues<4>(gram, eigenvalues);
    double total = 0.0;
    double second = 0.0;
    for (double s2 : eigenvalues)
    {
        total += s2 * s2;
        second += s2;
    }
    // sum s^2 = Tr U^dag U = 4 is itself a check on the realignment.
    converged = converged && std::abs(second - 4.0) < 1.0e-12;
    return total;
}

// ---------------------------------------------------------------------------

void test_non_entangling_gates()
{
    std::mt19937 rng(11);
    expect(mipt::entangling_power(identity4()) == 0.0, "the identity has e_p = 0");
    expect(mipt::entangling_power(mipt::swap_matrix()) == 0.0, "SWAP has e_p = 0");
    for (int trial = 0; trial < 50; ++trial)
    {
        const GateMatrix4 local = random_local(rng);
        expect(std::abs(mipt::entangling_power(local)) < 1.0e-12,
               "a Kronecker product of single-qubit unitaries has e_p = 0");
        expect(std::abs(mipt::entangling_power(mipt::multiply(local, mipt::swap_matrix()))) <
                   1.0e-12,
               "a local unitary times SWAP has e_p = 0");
        expect(std::abs(mipt::entangling_power(mipt::multiply(mipt::swap_matrix(), local))) <
                   1.0e-12,
               "SWAP times a local unitary has e_p = 0");
        expect(mipt::entangling_power(local) >= 0.0,
               "rounding below zero is clamped, never reported negative");
    }
    const mipt::GateEntanglement id = mipt::gate_entanglement(identity4());
    expect(id.entangling_power_normalized == 0.0, "the identity's normalized power is 0");
    expect(id.even_block_mixing == 0.0 && id.odd_block_mixing == 0.0,
           "the identity mixes neither parity block");
}

void test_maximal_gates()
{
    for (const auto &[name, gate] : {std::pair<const char *, GateMatrix4>{"CNOT", cnot()},
                                     std::pair<const char *, GateMatrix4>{"CZ", cz()}})
    {
        const mipt::GateEntanglement g = mipt::gate_entanglement(gate);
        expect(std::abs(g.entangling_power - MAX_EP) < 1.0e-14,
               std::string(name) + " has e_p = 2/9");
        expect(std::abs(g.entangling_power_normalized - 1.0) < 1.0e-14,
               std::string(name) + " has normalized e_p = 1");
    }
    // CNOT with the control on the other qubit is the same nonlocal content.
    const GateMatrix4 flipped =
        mipt::multiply(mipt::swap_matrix(), mipt::multiply(cnot(), mipt::swap_matrix()));
    expect(std::abs(mipt::entangling_power(flipped) - MAX_EP) < 1.0e-14,
           "a CNOT controlled on the other qubit is also maximal");
}

void test_local_invariance()
{
    std::mt19937 rng(23);
    for (int trial = 0; trial < 100; ++trial)
    {
        const GateMatrix4 u = random_parity_preserving(rng);
        const GateMatrix4 dressed =
            mipt::multiply(random_local(rng), mipt::multiply(u, random_local(rng)));
        expect(std::abs(mipt::entangling_power(u) - mipt::entangling_power(dressed)) < 1.0e-12,
               "e_p is invariant under local pre- and post-unitaries");
        // CZ dressed is still maximal, whatever the locals.
        const GateMatrix4 dressed_cz =
            mipt::multiply(random_local(rng), mipt::multiply(cz(), random_local(rng)));
        expect(std::abs(mipt::entangling_power(dressed_cz) - MAX_EP) < 1.0e-12,
               "a locally dressed CZ stays at 2/9");
    }
}

void test_random_rppu_ranges()
{
    std::mt19937 rng(37);
    double largest = 0.0;
    for (int trial = 0; trial < 2000; ++trial)
    {
        const GateMatrix4 u = random_parity_preserving(rng);
        expect(unitarity_error(u) < 1.0e-12, "parity_preserving_matrix is unitary");
        const mipt::GateEntanglement g = mipt::gate_entanglement(u);
        expect(g.entangling_power >= 0.0 && g.entangling_power <= MAX_EP + 1.0e-14,
               "a random RPPU gate has e_p in [0, 2/9]");
        expect(g.entangling_power_normalized >= 0.0 &&
                   g.entangling_power_normalized <= 1.0 + 1.0e-13,
               "and normalized e_p in [0, 1]");
        expect(g.even_block_mixing >= 0.0 && g.even_block_mixing <= 1.0 + 1.0e-14,
               "the even-block mixing is in [0, 1]");
        expect(g.odd_block_mixing >= 0.0 && g.odd_block_mixing <= 1.0 + 1.0e-14,
               "the odd-block mixing is in [0, 1]");
        largest = std::max(largest, g.entangling_power_normalized);
    }
    // Premise: the ensemble is not degenerate at zero, so the bound above is
    // tested on genuinely entangling gates.
    expect(largest > 0.5, "random RPPU gates reach well into the entangling range");
}

void test_mixing_matches_u3_theta()
{
    std::mt19937 rng(41);
    for (int trial = 0; trial < 200; ++trial)
    {
        const mipt::RppuBondGate gate = mipt::random_parity_preserving_gate(0, 1, rng);
        const double even_expected = std::pow(std::sin(gate.even.theta), 2);
        const double odd_expected = std::pow(std::sin(gate.odd.theta), 2);
        expect(std::abs(gate.entanglement.even_block_mixing - even_expected) < 1.0e-12,
               "the even-block mixing is sin^2(theta) of the stored U3");
        expect(std::abs(gate.entanglement.odd_block_mixing - odd_expected) < 1.0e-12,
               "the odd-block mixing is sin^2(theta) of the stored U3");
    }
    // A block that is a pure permutation (theta = pi) mixes nothing either.
    mipt::RppuBondGate swapish;
    swapish.kind = 0;
    swapish.even.theta = M_PI;
    swapish.odd.theta = 0.5 * M_PI;
    const mipt::GateEntanglement g = mipt::rppu_gate_entanglement(swapish);
    expect(std::abs(g.even_block_mixing) < 1.0e-14,
           "a fully off-diagonal block is a permutation and mixes nothing");
    expect(std::abs(g.odd_block_mixing - 1.0) < 1.0e-14, "a balanced block mixes fully");
}

void test_draw_time_matches_rebuilt()
{
    std::mt19937 rng(53);
    for (int trial = 0; trial < 500; ++trial)
    {
        const mipt::RppuBondGate gate = mipt::random_parity_preserving_gate(2, 3, rng);
        expect(std::isfinite(gate.entanglement.entangling_power_normalized),
               "a drawn gate stores its diagnostic");
        const GateMatrix4 rebuilt = mipt::rppu_gate_matrix(gate);
        expect(unitarity_error(rebuilt) < 1.0e-12, "the rebuilt engine matrix is unitary");
        const mipt::GateEntanglement again = mipt::gate_entanglement(rebuilt);
        expect(std::abs(again.entangling_power - gate.entanglement.entangling_power) < 1.0e-12,
               "draw-time e_p equals e_p of the matrix the engine applies");
        expect(std::abs(again.entangling_power_normalized -
                        gate.entanglement.entangling_power_normalized) < 1.0e-12,
               "and so does the normalized value");
        expect(std::abs(again.even_block_mixing - gate.entanglement.even_block_mixing) < 1.0e-12,
               "draw-time even mixing equals the rebuilt one");
        expect(std::abs(again.odd_block_mixing - gate.entanglement.odd_block_mixing) < 1.0e-12,
               "draw-time odd mixing equals the rebuilt one");

        // A hand-assembled gate (no stored diagnostic) falls back to the
        // rebuilt matrix and must give the same answer.
        mipt::RppuBondGate bare = gate;
        bare.entanglement = {};
        const mipt::GateEntanglement fallback = mipt::rppu_gate_entanglement(bare);
        expect(std::abs(fallback.entangling_power_normalized -
                        gate.entanglement.entangling_power_normalized) < 1.0e-12,
               "the parameter fallback agrees with the draw-time value");

        mipt::RppuBondGate jw = gate;
        jw.kind = 2;
        expect(mipt::rppu_gate_entanglement(jw).entangling_power_normalized ==
                   gate.entanglement.entangling_power_normalized,
               "a JW boundary gate carries its two-mode unitary's diagnostic");
    }
    const mipt::GateEntanglement fswap = mipt::rppu_gate_entanglement(mipt::fermionic_swap_gate(0, 1));
    expect(std::isnan(fswap.entangling_power_normalized) && std::isnan(fswap.even_block_mixing) &&
               std::isnan(fswap.odd_block_mixing),
           "an FSWAP is transport and carries no diagnostic (NaN, not 0)");
}

void test_fourth_moment_shortcut()
{
    std::mt19937 rng(67);
    auto check = [](const GateMatrix4 &u, const std::string &what) {
        bool converged = false;
        const double reference = fourth_moment_from_spectrum(u, converged);
        expect(converged, "Jacobi converges on R^dag R for " + what);
        expect(std::abs(mipt::operator_schmidt_fourth_moment(u) - reference) < 1.0e-12,
               "||R R^dag||_F^2 equals sum s^4 from the singular values for " + what);
    };
    check(identity4(), "the identity");      // one Schmidt term: s = 2, sum s^4 = 16
    check(mipt::swap_matrix(), "SWAP");      // four equal terms: s = 1, sum s^4 = 4
    check(cnot(), "CNOT");                   // two equal terms: s^2 = 2, sum s^4 = 8
    expect(std::abs(mipt::operator_schmidt_fourth_moment(identity4()) - 16.0) < 1.0e-14,
           "the identity's realignment is rank one with s = 2");
    expect(std::abs(mipt::operator_schmidt_fourth_moment(mipt::swap_matrix()) - 4.0) < 1.0e-14,
           "SWAP's realignment has four unit Schmidt coefficients");
    expect(std::abs(mipt::operator_schmidt_fourth_moment(cnot()) - 8.0) < 1.0e-14,
           "CNOT's realignment has two Schmidt coefficients sqrt 2");
    for (int trial = 0; trial < 200; ++trial)
    {
        check(random_parity_preserving(rng), "a random parity-preserving gate");
        check(mipt::multiply(random_local(rng),
                             mipt::multiply(random_parity_preserving(rng), random_local(rng))),
              "a dressed random gate");
    }
}

void test_layout_helpers()
{
    // kron(high, low): low on bit 0. X on bit 0 maps index 0 -> 1.
    const GateMatrix2 x{Complex(0.0), Complex(1.0), Complex(1.0), Complex(0.0)};
    const GateMatrix4 x_low = mipt::kron(identity2(), x);
    expect(x_low[1 * 4 + 0] == Complex(1.0) && x_low[2 * 4 + 0] == Complex(0.0),
           "kron puts its second factor on bit 0");
    const GateMatrix4 x_high = mipt::kron(x, identity2());
    expect(x_high[2 * 4 + 0] == Complex(1.0), "and its first factor on bit 1");
    // SWAP exchanges |01> and |10>.
    const GateMatrix4 s = mipt::swap_matrix();
    expect(s[1 * 4 + 2] == Complex(1.0) && s[2 * 4 + 1] == Complex(1.0) && unitarity_error(s) == 0.0,
           "swap_matrix exchanges indices 1 and 2");
    // Parity preservation: no amplitude leaves its parity sector.
    std::mt19937 rng(79);
    const GateMatrix4 u = random_parity_preserving(rng);
    double leak = 0.0;
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            const int pr = (r & 1) ^ ((r >> 1) & 1);
            const int pc = (c & 1) ^ ((c >> 1) & 1);
            if (pr != pc)
            {
                leak = std::max(leak, std::abs(u[r * 4 + c]));
            }
        }
    }
    expect(leak == 0.0, "parity_preserving_matrix has no entries between parity sectors");
}

} // namespace

int main()
{
    test_non_entangling_gates();
    test_maximal_gates();
    test_local_invariance();
    test_random_rppu_ranges();
    test_mixing_matches_u3_theta();
    test_draw_time_matches_rebuilt();
    test_fourth_moment_shortcut();
    test_layout_helpers();

    if (failures != 0)
    {
        std::cerr << "gate_entanglement_tests: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "gate_entanglement_tests: PASS\n";
    return 0;
}
