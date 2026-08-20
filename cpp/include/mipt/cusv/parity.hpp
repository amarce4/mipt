#pragma once

// Parity-sector encoding for the cuStateVec circuit engine.
//
// A parity-preserving circuit started from |0...0> can never leave the even
// global-parity sector.  Every gate in the RPPU/qRPPU gate set is block
// diagonal in the pair parity (the even block acts on {|00>, |11>}, the odd
// block on {|01>, |10>}), an FSWAP is a weight-preserving permutation, and a
// Z-basis measurement only deletes basis states -- so every basis state in the
// support keeps the even Hamming weight the initial state had.  The odd half
// of the state vector is therefore *exactly* zero, bit for bit, at every
// layer, and the engine has been loading, multiplying and storing it on every
// pass.  Verified on the production layer builders: at n = 12..14, over full
// measured trajectories, the largest odd-parity amplitude seen at any layer is
// 0.0e+00 and the odd-parity nonzero count is 0.
//
// The encoding drops that half.  An encoded index e carries bits b_0..b_{n-2}
// and the missing bit is recovered as
//
//     b_{n-1} = parity(e),                                              (*)
//
// which is well defined precisely because the total weight is even.  The state
// becomes 2^(n-1) amplitudes and every pass moves half the bytes.
//
// Three kinds of operator have to be rewritten; everything else is untouched.
//
//   * A gate on qubits both <= n-2.  A parity-preserving gate maps the pair
//     (b_q0, b_q1) within a fixed pair parity, so it leaves the *global*
//     parity of the remaining bits alone and therefore leaves (*) consistent.
//     It acts on the encoded index by exactly the same 4x4 matrix, so
//     custatevecApplyMatrix drives it unchanged and blocking still fuses runs
//     of them.
//
//   * A gate on (q, n-1).  Since b_{n-1} = b_q XOR s with
//     s = parity(e with bit q cleared), the gate collapses to a 2x2 on the
//     encoded bit q whose matrix is picked by s -- one custom pass.  See
//     encode_boundary_bond().
//
//   * The Jordan-Wigner closure string.  prod_{k != lo, hi} Z_k equals
//     Z_lo Z_hi on the even sector (multiply by the identity prod_all Z_k),
//     so the two full-state diagonal passes that used to bracket the closure
//     bond fold into the bond's own 4x4 at zero cost -- and, because the
//     string was also a blocking barrier, the closure bond stops splitting the
//     odd layer's fusable run.  Checked exact (max |3-pass - 1-pass| = 0.0e+00
//     at n = 8, 12, 16).
//
// Measurement of a site <= n-2 is an ordinary Z-basis measurement of the
// encoded bit.  Measurement of site n-1 is a measurement of parity(e); the
// engine takes it as a separate chunk of the joint distribution, which is
// exact by the same chain-rule argument measure_layer() already rests on.
//
// The state is expanded back to the full 2^n vector once, at the end of the
// trajectory, so every downstream RDM/entropy/TMI consumer sees an ordinary
// state and needed no change.  That expansion is what still costs the
// reduction kernels their own factor of two: teaching them the encoding would
// halve the RDM sweeps as well, and halve peak device memory with them.

#include "mipt/cusv/gates.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace mipt::cusv
{

// MIPT_CUSV_PARITY=0 restores the full 2^n circuit path. It is the A/B control
// for this optimization and changes nothing else.
inline bool parity_encoding_enabled()
{
    static const bool on = backend::parity_sector_encoding_enabled();
    return on;
}

// Pair parity of a two-qubit basis index (bit 0 = first target).
inline int bond_index_parity(int index) { return (index & 1) ^ ((index >> 1) & 1); }

// Whether a 4x4 bond gate is block diagonal in the pair parity. This is the
// property the whole encoding rests on, so it is checked per bond rather than
// assumed from the circuit type -- the cost is 16 comparisons against a pass
// over the state vector.
inline bool is_parity_preserving_bond(const DenseMatrix &m)
{
    if (m.targets != 2)
    {
        return false;
    }
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            if (bond_index_parity(r) != bond_index_parity(c) && std::abs(m.at(r, c)) > 1e-12)
            {
                return false;
            }
        }
    }
    return true;
}

// Rewrites a bond touching the derived qubit as a parity-controlled 2x2.
//
// With `pos` the position of the derived qubit inside `targets` and `other`
// the position of the surviving one, the live 4x4 entries at a given s are the
// ones whose derived bit is (surviving bit XOR s), so the 2x2 selected by s is
//     M_s[rb][cb] = m[(rb << other) | ((rb ^ s) << pos)]
//                    [(cb << other) | ((cb ^ s) << pos)].
inline Op encode_boundary_bond(const DenseMatrix &m, const std::vector<int> &targets, int derived)
{
    const int pos = (targets[0] == derived) ? 0 : 1;
    const int other = 1 - pos;

    Op op;
    op.kind = OpKind::ParityGate;
    op.targets = {targets[static_cast<std::size_t>(other)]};
    for (int s = 0; s < 2; ++s)
    {
        auto &dst = (s == 0) ? op.parity_even : op.parity_odd;
        for (int rb = 0; rb < 2; ++rb)
        {
            for (int cb = 0; cb < 2; ++cb)
            {
                const int row = (rb << other) | ((rb ^ s) << pos);
                const int col = (cb << other) | ((cb ^ s) << pos);
                const Complex v = m.at(static_cast<std::size_t>(row), static_cast<std::size_t>(col));
                dst[static_cast<std::size_t>((2 * rb + cb) * 2 + 0)] = v.real();
                dst[static_cast<std::size_t>((2 * rb + cb) * 2 + 1)] = v.imag();
            }
        }
    }
    return op;
}

// Folds a Jordan-Wigner string into the bond it brackets.
//
// The string flips the sign wherever bit `hi` is set and the masked popcount
// is odd. When the mask is the complement of {lo, hi}, the even-sector
// identity prod_all Z_k = +1 turns that masked parity into b_lo XOR b_hi, so
// the sign is -1 exactly on (b_lo = 0, b_hi = 1) -- diagonal on the bond's own
// two qubits. The bracketed operator is then D U D, folded here into one 4x4.
inline void fold_jw_string_into_bond(DenseMatrix &m, const std::vector<int> &targets, int lo, int hi)
{
    const int hi_pos = (targets[0] == hi) ? 0 : 1;
    const int lo_pos = 1 - hi_pos;
    double d[4];
    for (int idx = 0; idx < 4; ++idx)
    {
        const int b_hi = (idx >> hi_pos) & 1;
        const int b_lo = (idx >> lo_pos) & 1;
        d[idx] = (b_hi == 1 && b_lo == 0) ? -1.0 : 1.0;
    }
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            m.at(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) *= d[r] * d[c];
        }
    }
}

// Rewrites one layer's operator list for the even-parity sector on n-1 encoded
// qubits. Returns nullopt when the layer holds anything the encoding does not
// cover -- a non-parity-preserving bond, a wider gate, or a Jordan-Wigner
// string that is not the closure sandwich -- in which case the caller falls
// back to the full 2^n path with nothing mutated.
//
// Must run on the *unblocked* op list: it inspects bonds one at a time.
inline std::optional<std::vector<Op>> encode_parity_sector(const std::vector<Op> &ops, int n)
{
    if (n < 4)
    {
        return std::nullopt;
    }
    const int derived = n - 1;
    std::vector<Op> out;
    out.reserve(ops.size());

    std::size_t i = 0;
    while (i < ops.size())
    {
        const Op &op = ops[i];

        if (op.kind == OpKind::JwString)
        {
            // Only the (D, U, D) sandwich is covered; a bare string is not.
            if (i + 2 >= ops.size())
            {
                return std::nullopt;
            }
            const Op &bond = ops[i + 1];
            const Op &back = ops[i + 2];
            if (bond.kind != OpKind::Matrix || bond.matrix.targets != 2 ||
                back.kind != OpKind::JwString || back.string_control != op.string_control ||
                back.string_mask != op.string_mask)
            {
                return std::nullopt;
            }
            const int lo = std::min(bond.targets[0], bond.targets[1]);
            const int hi = std::max(bond.targets[0], bond.targets[1]);
            if (op.string_control != hi || !is_parity_preserving_bond(bond.matrix))
            {
                return std::nullopt;
            }
            std::uint64_t complement = 0;
            for (int k = 0; k < n; ++k)
            {
                if (k != lo && k != hi)
                {
                    complement |= std::uint64_t{1} << k;
                }
            }
            if (op.string_mask != complement)
            {
                return std::nullopt;
            }

            DenseMatrix folded = bond.matrix;
            fold_jw_string_into_bond(folded, bond.targets, lo, hi);
            if (hi == derived || lo == derived)
            {
                out.push_back(encode_boundary_bond(folded, bond.targets, derived));
            }
            else
            {
                Op plain = bond;
                plain.matrix = std::move(folded);
                out.push_back(std::move(plain));
            }
            i += 3;
            continue;
        }

        if (op.kind != OpKind::Matrix || op.matrix.targets != 2 ||
            !is_parity_preserving_bond(op.matrix))
        {
            return std::nullopt;
        }
        if (op.targets[0] == derived || op.targets[1] == derived)
        {
            out.push_back(encode_boundary_bond(op.matrix, op.targets, derived));
        }
        else
        {
            // Acts on the encoded index by the same matrix.
            out.push_back(op);
        }
        ++i;
    }
    return out;
}

} // namespace mipt::cusv
