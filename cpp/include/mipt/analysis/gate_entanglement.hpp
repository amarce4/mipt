#pragma once

// How entangling a two-mode gate is, as a property of the unitary alone.
//
// The question this answers
// -------------------------
// The spacetime percolation graph treats every gate as an edge. A connected
// pair with no fermionic negativity could therefore be a pair whose every
// connecting path runs through a gate that is *intrinsically* unable to
// entangle -- a local unitary, or a local unitary dressed by a SWAP. This header
// measures that directly on the 4x4 matrix the engine applies, so the graph can
// carry it as an edge capacity without changing the percolation event itself.
//
// The quantities
// --------------
// For a 4x4 unitary U on qubits (A, B), realign it as R_{(a a'),(b b')} =
// U_{(a b),(a' b')} and take its singular values s_alpha (the operator Schmidt
// coefficients; sum_alpha s_alpha^2 = Tr U^dag U = 4). The operator
// entanglement is
//
//     E(U) = 1 - (1/16) sum_alpha s_alpha^4,
//
// and Zanardi's entangling power is
//
//     e_p(U) = (4/9) [ E(U) + E(U S) - E(S) ],   E(S) = 3/4,
//
// with S the SWAP. 0 <= e_p <= 2/9: zero for every local unitary and every
// local unitary times SWAP, 2/9 for CNOT and CZ. The stored
// `entangling_power_normalized` is (9/2) e_p in [0, 1].
//
// sum s^4 is computed as ||R R^dag||_F^2 = Tr (R R^dag)^2, which is the same
// number with no eigensolver: the singular values themselves are never needed.
//
// The parity-block mixing strengths m = 4 |U_00 U_10|^2 of the even and odd
// 2x2 blocks measure how well the gate turns an occupation-basis input into a
// superposition within that sector -- the pairing (F, even block) and hopping
// (G, odd block) channels. m = 0 for a block that is diagonal *or* fully
// off-diagonal (both are permutations up to phases), 1 for a balanced mixer.
// For a unitary block |U_00|^2 + |U_10|^2 = 1, so m = sin^2(theta) in U3 terms.
//
// No threshold lives here or anywhere in the simulator: the values are stored
// continuously and any "non-entangling" cut is a post-processing choice.
//
// Host-only, no CUDA-Q, so `make test-gate` pins it on closed forms.

#include <array>
#include <cmath>
#include <complex>
#include <limits>

namespace mipt
{

// Bump when any definition below changes. Written into every output that
// carries a gate diagnostic, beside the definition string.
inline constexpr int GATE_DIAGNOSTICS_VERSION = 1;
inline constexpr const char *GATE_DIAGNOSTICS_DEFINITION =
    "ep_norm=(9/2)*(4/9)*(E(U)+E(US)-3/4),E=1-sum(s^4)/16 over realigned U;"
    "mix=4|U00*U10|^2 per parity block;temporal_capacity=1;"
    "widest=max over paths of min unitary capacity;"
    "shortest_maximin=same over shortest paths";

using GateMatrix4 = std::array<std::complex<double>, 16>; // row-major, bit 0 = first qubit
using GateMatrix2 = std::array<std::complex<double>, 4>;  // row-major

struct GateEntanglement
{
    double entangling_power = std::numeric_limits<double>::quiet_NaN();            // e_p in [0, 2/9]
    double entangling_power_normalized = std::numeric_limits<double>::quiet_NaN(); // (9/2) e_p
    double even_block_mixing = std::numeric_limits<double>::quiet_NaN();
    double odd_block_mixing = std::numeric_limits<double>::quiet_NaN();
};

// The parity-preserving two-mode gate, laid out exactly as the cuStateVec
// engine applies it (cusv::rppu_bond_matrix delegates here, so there is one
// definition): bit 0 is the gate's first qubit; the even block acts on
// {|00>, |11>} = indices {0, 3} in that order and the odd block on
// {|10>, |01>} = indices {2, 1} -- the reversal comes from the CNOT-conjugated
// construction of the CUDA-Q kernel.
inline GateMatrix4 parity_preserving_matrix(const GateMatrix2 &even, const GateMatrix2 &odd)
{
    GateMatrix4 m{};
    m[0 * 4 + 0] = even[0];
    m[0 * 4 + 3] = even[1];
    m[3 * 4 + 0] = even[2];
    m[3 * 4 + 3] = even[3];
    m[2 * 4 + 2] = odd[0];
    m[2 * 4 + 1] = odd[1];
    m[1 * 4 + 2] = odd[2];
    m[1 * 4 + 1] = odd[3];
    return m;
}

inline GateMatrix4 swap_matrix()
{
    GateMatrix4 m{};
    m[0 * 4 + 0] = 1.0;
    m[1 * 4 + 2] = 1.0;
    m[2 * 4 + 1] = 1.0;
    m[3 * 4 + 3] = 1.0;
    return m;
}

inline GateMatrix4 multiply(const GateMatrix4 &a, const GateMatrix4 &b)
{
    GateMatrix4 out{};
    for (int r = 0; r < 4; ++r)
        for (int k = 0; k < 4; ++k)
            for (int c = 0; c < 4; ++c)
                out[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
    return out;
}

// Kronecker product with `low` on bit 0 and `high` on bit 1.
inline GateMatrix4 kron(const GateMatrix2 &high, const GateMatrix2 &low)
{
    GateMatrix4 out{};
    for (int r1 = 0; r1 < 2; ++r1)
        for (int r0 = 0; r0 < 2; ++r0)
            for (int c1 = 0; c1 < 2; ++c1)
                for (int c0 = 0; c0 < 2; ++c0)
                    out[(r0 + 2 * r1) * 4 + (c0 + 2 * c1)] =
                        high[r1 * 2 + c1] * low[r0 * 2 + c0];
    return out;
}

// The realignment with respect to the (bit 0 | bit 1) cut:
// R[(r0 c0), (r1 c1)] = U[r0 + 2 r1, c0 + 2 c1].
inline GateMatrix4 realign(const GateMatrix4 &u)
{
    GateMatrix4 r{};
    for (int r1 = 0; r1 < 2; ++r1)
        for (int r0 = 0; r0 < 2; ++r0)
            for (int c1 = 0; c1 < 2; ++c1)
                for (int c0 = 0; c0 < 2; ++c0)
                    r[(r0 * 2 + c0) * 4 + (r1 * 2 + c1)] = u[(r0 + 2 * r1) * 4 + (c0 + 2 * c1)];
    return r;
}

// sum_alpha s_alpha^4 of the realigned operator, as ||R R^dag||_F^2.
inline double operator_schmidt_fourth_moment(const GateMatrix4 &u)
{
    const GateMatrix4 r = realign(u);
    double total = 0.0;
    for (int a = 0; a < 4; ++a)
    {
        for (int b = 0; b < 4; ++b)
        {
            std::complex<double> g = 0.0;
            for (int k = 0; k < 4; ++k)
            {
                g += r[a * 4 + k] * std::conj(r[b * 4 + k]);
            }
            total += std::norm(g);
        }
    }
    return total;
}

inline double operator_entanglement(const GateMatrix4 &u)
{
    return 1.0 - operator_schmidt_fourth_moment(u) / 16.0;
}

inline double entangling_power(const GateMatrix4 &u)
{
    const double value =
        (4.0 / 9.0) * (operator_entanglement(u) + operator_entanglement(multiply(u, swap_matrix())) -
                       0.75);
    // Exact zero is reachable (local gates); rounding can land a hair below.
    return value < 0.0 && value > -1.0e-14 ? 0.0 : value;
}

inline double block_mixing(const std::complex<double> &u00, const std::complex<double> &u10)
{
    return 4.0 * std::norm(u00 * u10);
}

// Every diagnostic of one 4x4 gate. The block mixings read the engine layout
// above, so they are only meaningful for a parity-preserving matrix in it.
inline GateEntanglement gate_entanglement(const GateMatrix4 &u)
{
    GateEntanglement out;
    out.entangling_power = entangling_power(u);
    out.entangling_power_normalized = 4.5 * out.entangling_power;
    out.even_block_mixing = block_mixing(u[0 * 4 + 0], u[3 * 4 + 0]);
    out.odd_block_mixing = block_mixing(u[2 * 4 + 2], u[1 * 4 + 2]);
    return out;
}

inline GateEntanglement parity_preserving_gate_entanglement(const GateMatrix2 &even,
                                                            const GateMatrix2 &odd)
{
    return gate_entanglement(parity_preserving_matrix(even, odd));
}

} // namespace mipt
