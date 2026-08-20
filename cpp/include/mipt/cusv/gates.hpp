#pragma once

// Layer -> operator list for the cuStateVec circuit engine.
//
// This is optimization #1: instead of emitting the 8-, 13-, or 20-gate
// decomposition that CUDA-Q applies one statevector pass at a time, each
// two-site bond is turned into the single 4x4 matrix that decomposition
// multiplies out to. The matrices are built to agree with CUDA-Q's gate
// conventions exactly; cpp/test/dump_bond.cpp extracts the same matrices from
// the CUDA-Q runtime and cusv_gate_tests asserts they match.
//
// Optimization #2 is the JwString op: the periodic-boundary Jordan-Wigner CZ
// chain in the fermionic RPPU circuit is a diagonal operator, so the whole
// chain of n-2 CZ gates becomes one sign pass instead of n-2 matrix passes.
//
// The layer builders never reorder gates. Blocking (#5, block_disjoint_ops)
// only groups *consecutive* ops with pairwise-disjoint supports, which commute,
// so the composed operator is unchanged.

#include "mipt/circuits/fermion.hpp"
#include "mipt/circuits/haar.hpp"
#include "mipt/circuits/mms.hpp"
#include "mipt/cusv/matrix.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mipt::cusv
{

enum class OpKind
{
    Matrix,     // dense matrix on `targets`
    JwString,   // diagonal: amplitude *= -1 where bit(string_control) & parity(idx & string_mask)
    ParityGate, // parity-sector encoding only; see mipt/cusv/parity.hpp
};

struct Op
{
    OpKind kind = OpKind::Matrix;
    std::vector<int> targets;
    DenseMatrix matrix;
    int string_control = -1;
    std::uint64_t string_mask = 0;
    // ParityGate only: the two 2x2 blocks a bond on (targets[0], n-1) reduces
    // to in the encoded picture, selected by s = parity(index with targets[0]
    // cleared). Row-major (re, im) pairs ordered M00, M01, M10, M11, indexed
    // by the value of the encoded bit.
    std::array<double, 8> parity_even{};
    std::array<double, 8> parity_odd{};
};

struct LayerOps
{
    std::vector<Op> ops;
    std::vector<int> measure_sites;
};

// ------------------------------------------------------------------ MMS bond

// One MMS bond: the chosen local rotation on each endpoint, then
// exp(-i*pi/4 X_a X_b) as h,h,cx,rz,cx,h,h. Bit 0 of the returned matrix is
// the bond's first site (the one the kernel names q[i]).
inline Matrix2 mms_local_rotation(int axis)
{
    if (axis == 0) { return mat_rx(PI_2); }
    if (axis == 1) { return mat_ry(PI_2); }
    // rz(-pi/4), rx(pi/2), rz(+pi/4) applied in that order.
    return mat_multiply(mat_rz(PI_4), mat_multiply(mat_rx(PI_2), mat_rz(-PI_4)));
}

inline int mms_axis_of(const MmsLayer &layer, int site)
{
    const auto s = static_cast<std::size_t>(site);
    if (layer.rot_x_flags[s]) { return 0; }
    if (layer.rot_y_flags[s]) { return 1; }
    if (layer.rot_xy_flags[s]) { return 2; }
    return -1;
}

inline DenseMatrix mms_bond_matrix(int axis_first, int axis_second)
{
    DenseMatrix m = identity_matrix(2);
    if (axis_first >= 0)
    {
        m = matrix_product(single_qubit_on(2, 0, mms_local_rotation(axis_first)), m);
    }
    if (axis_second >= 0)
    {
        m = matrix_product(single_qubit_on(2, 1, mms_local_rotation(axis_second)), m);
    }
    const auto h = mat_hadamard();
    m = matrix_product(single_qubit_on(2, 0, h), m);
    m = matrix_product(single_qubit_on(2, 1, h), m);
    m = matrix_product(cnot_on(2, 0, 1), m);
    m = matrix_product(single_qubit_on(2, 1, mat_rz(PI_2)), m);
    m = matrix_product(cnot_on(2, 0, 1), m);
    m = matrix_product(single_qubit_on(2, 0, h), m);
    m = matrix_product(single_qubit_on(2, 1, h), m);
    return m;
}

inline LayerOps build_mms_layer_ops(const MmsLayer &layer, int n, bool closed)
{
    LayerOps out;
    const int start = layer.start;
    for (int i = start; i < n - 1; i += 2)
    {
        Op op;
        op.targets = {i, i + 1};
        op.matrix = mms_bond_matrix(mms_axis_of(layer, i), mms_axis_of(layer, i + 1));
        out.ops.push_back(std::move(op));
    }
    if (closed && start == 1 && n > 2)
    {
        // Closure bond (n-1, 0): the kernel treats q[n-1] as the first site.
        Op op;
        op.targets = {n - 1, 0};
        op.matrix = mms_bond_matrix(mms_axis_of(layer, n - 1), mms_axis_of(layer, 0));
        out.ops.push_back(std::move(op));
    }
    for (int site = 0; site < n; ++site)
    {
        if (layer.measure_flags[static_cast<std::size_t>(site)])
        {
            out.measure_sites.push_back(site);
        }
    }
    return out;
}

// ----------------------------------------------------------------- Haar bond

// Replays the cosine-sine decomposition the CUDA-Q kernel applies. Bit 0 is
// gate.q0 (the CSD's least-significant basis bit), bit 1 is gate.q1.
inline DenseMatrix haar_bond_matrix(const HaarBondGate &gate)
{
    auto multiplexed = [](DenseMatrix m, const U2Params &high, const U2Params &low)
    {
        // r1(phase) on bit 1, controlled-U3 (control bit 1, target bit 0),
        // then the same with bit 1 inverted by X.
        const Matrix2 r1_high = {Complex{1.0, 0.0}, Complex{0.0, 0.0},
                                 Complex{0.0, 0.0}, std::polar(1.0, high.global_phase)};
        const Matrix2 u3_high = mat_from_u2_params(U2Params{0.0, high.theta, high.phi, high.lambda});
        const Matrix2 r1_low = {Complex{1.0, 0.0}, Complex{0.0, 0.0},
                                Complex{0.0, 0.0}, std::polar(1.0, low.global_phase)};
        const Matrix2 u3_low = mat_from_u2_params(U2Params{0.0, low.theta, low.phi, low.lambda});
        const Matrix2 x = {Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}};

        m = matrix_product(single_qubit_on(2, 1, r1_high), m);
        m = matrix_product(controlled_single_qubit_on(2, 1, 0, u3_high), m);
        m = matrix_product(single_qubit_on(2, 1, x), m);
        m = matrix_product(single_qubit_on(2, 1, r1_low), m);
        m = matrix_product(controlled_single_qubit_on(2, 1, 0, u3_low), m);
        m = matrix_product(single_qubit_on(2, 1, x), m);
        return m;
    };

    const Matrix2 x = {Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}};

    DenseMatrix m = identity_matrix(2);
    m = multiplexed(std::move(m), gate.b1, gate.b0);
    // Ry multiplexor: control bit 0, target bit 1.
    m = matrix_product(controlled_single_qubit_on(2, 0, 1, mat_ry(gate.theta1)), m);
    m = matrix_product(single_qubit_on(2, 0, x), m);
    m = matrix_product(controlled_single_qubit_on(2, 0, 1, mat_ry(gate.theta0)), m);
    m = matrix_product(single_qubit_on(2, 0, x), m);
    m = multiplexed(std::move(m), gate.a1, gate.a0);
    return m;
}

inline LayerOps build_haar_layer_ops(const HaarLayer &layer, int n)
{
    LayerOps out;
    for (const auto &gate : layer.gates)
    {
        Op op;
        op.targets = {gate.q0, gate.q1};
        op.matrix = haar_bond_matrix(gate);
        out.ops.push_back(std::move(op));
    }
    for (int site = 0; site < n; ++site)
    {
        if (layer.measure_flags[static_cast<std::size_t>(site)])
        {
            out.measure_sites.push_back(site);
        }
    }
    return out;
}

// ----------------------------------------------------------------- RPPU bond

// The parity-preserving two-mode gate is block diagonal: the even block acts on
// {|00>, |11>} in that order and the odd block acts on {|10>, |01>} -- note the
// reversal, which comes from the CNOT-conjugated construction in the kernel.
// Bit 0 is gate.q0.
inline DenseMatrix rppu_bond_matrix(const RppuBondGate &gate)
{
    const Matrix2 even = mat_from_u2_params(gate.even);
    const Matrix2 odd = mat_from_u2_params(gate.odd);
    DenseMatrix m(2);
    m.at(0, 0) = even[0];
    m.at(0, 3) = even[1];
    m.at(3, 0) = even[2];
    m.at(3, 3) = even[3];
    m.at(2, 2) = odd[0];
    m.at(2, 1) = odd[1];
    m.at(1, 2) = odd[2];
    m.at(1, 1) = odd[3];
    return m;
}

// FSWAP = SWAP * CZ, matching `z<ctrl>(q0,q1); swap(q0,q1);` in the kernel.
inline DenseMatrix fswap_matrix()
{
    DenseMatrix m(2);
    m.at(0, 0) = Complex{1.0, 0.0};
    m.at(2, 1) = Complex{1.0, 0.0};
    m.at(1, 2) = Complex{1.0, 0.0};
    m.at(3, 3) = Complex{-1.0, 0.0};
    return m;
}

inline LayerOps build_rppu_layer_ops(const RppuLayer &layer, int n)
{
    LayerOps out;
    for (const auto &gate : layer.gates)
    {
        if (gate.kind == 1)
        {
            Op op;
            op.targets = {gate.q0, gate.q1};
            op.matrix = fswap_matrix();
            out.ops.push_back(std::move(op));
            continue;
        }

        const bool jw = gate.kind == 2;
        Op string_op;
        if (jw)
        {
            const int lo = std::min(gate.q0, gate.q1);
            const int hi = std::max(gate.q0, gate.q1);
            std::uint64_t mask = 0;
            for (int k = lo + 1; k < hi; ++k)
            {
                mask |= std::uint64_t{1} << k;
            }
            string_op.kind = OpKind::JwString;
            string_op.string_control = hi;
            string_op.string_mask = mask;
            if (mask != 0)
            {
                out.ops.push_back(string_op);
            }
        }

        Op op;
        op.targets = {gate.q0, gate.q1};
        op.matrix = rppu_bond_matrix(gate);
        out.ops.push_back(std::move(op));

        if (jw && string_op.string_mask != 0)
        {
            out.ops.push_back(string_op);
        }
    }
    for (int site = 0; site < n; ++site)
    {
        if (layer.measure_flags[static_cast<std::size_t>(site)])
        {
            out.measure_sites.push_back(site);
        }
    }
    return out;
}

// -------------------------------------------------------------- #5 blocking

// Fuse runs of consecutive Matrix ops with pairwise-disjoint targets into one
// larger matrix, up to `max_block_targets` qubits.
//
// Two invariants make this exactly equivalent to applying the ops one by one:
//   * only *consecutive* ops are considered, so nothing is reordered past a
//     non-commuting neighbour (in particular past a JwString op); and
//   * grouped ops have disjoint supports, so they commute and their product is
//     the tensor product built by tensor_product().
// The composed matrix is therefore a Kronecker product of the original bond
// gates and cannot entangle sites the circuit did not entangle.
inline std::vector<Op> block_disjoint_ops(const std::vector<Op> &ops, int max_block_targets)
{
    if (max_block_targets < 2)
    {
        return ops;
    }
    std::vector<Op> out;
    std::size_t i = 0;
    while (i < ops.size())
    {
        if (ops[i].kind != OpKind::Matrix)
        {
            out.push_back(ops[i]);
            ++i;
            continue;
        }

        Op block = ops[i];
        std::size_t j = i + 1;
        while (j < ops.size() && ops[j].kind == OpKind::Matrix &&
               block.matrix.targets + ops[j].matrix.targets <= max_block_targets)
        {
            const bool disjoint = std::none_of(
                ops[j].targets.begin(), ops[j].targets.end(),
                [&](int t)
                { return std::find(block.targets.begin(), block.targets.end(), t) != block.targets.end(); });
            if (!disjoint)
            {
                break;
            }
            block.matrix = tensor_product(block.matrix, ops[j].matrix);
            block.targets.insert(block.targets.end(), ops[j].targets.begin(), ops[j].targets.end());
            ++j;
        }
        out.push_back(std::move(block));
        i = j;
    }
    return out;
}

} // namespace mipt::cusv
