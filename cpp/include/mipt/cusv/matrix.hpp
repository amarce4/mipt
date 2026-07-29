#pragma once

// Small dense complex matrices for the cuStateVec circuit engine.
//
// Every matrix here is row-major and indexed so that **bit k of the row/column
// index corresponds to targets[k]**, matching custatevecApplyMatrix. That is
// the same little-endian convention CUDA-Q uses for the state-vector index
// (qubit b <-> index bit b), so a gate's target list can be handed to
// cuStateVec unchanged.
//
// The single-qubit primitives reproduce CUDA-Q's gate definitions exactly;
// see cpp/test/dump_bond.cpp, which extracts the same matrices from the
// CUDA-Q runtime and is checked against these by the cusv_gate_tests.

#include "mipt/types.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mipt::cusv
{

using Complex = std::complex<double>;
using Matrix2 = std::array<Complex, 4>;

inline constexpr double PI_2 = 1.57079632679489661923;
inline constexpr double PI_4 = 0.78539816339744830962;

// ---------------------------------------------------------------- primitives

inline Matrix2 mat_identity()
{
    return {Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{1.0, 0.0}};
}

inline Matrix2 mat_rx(double theta)
{
    const double c = std::cos(0.5 * theta);
    const double s = std::sin(0.5 * theta);
    return {Complex{c, 0.0}, Complex{0.0, -s}, Complex{0.0, -s}, Complex{c, 0.0}};
}

inline Matrix2 mat_ry(double theta)
{
    const double c = std::cos(0.5 * theta);
    const double s = std::sin(0.5 * theta);
    return {Complex{c, 0.0}, Complex{-s, 0.0}, Complex{s, 0.0}, Complex{c, 0.0}};
}

inline Matrix2 mat_rz(double theta)
{
    return {std::polar(1.0, -0.5 * theta), Complex{0.0, 0.0},
            Complex{0.0, 0.0}, std::polar(1.0, 0.5 * theta)};
}

inline Matrix2 mat_hadamard()
{
    const double r = 1.0 / std::sqrt(2.0);
    return {Complex{r, 0.0}, Complex{r, 0.0}, Complex{r, 0.0}, Complex{-r, 0.0}};
}

// U = exp(i*global_phase) * U3(theta, phi, lambda), the exact inverse of
// mipt::decompose_u2_to_u3_phase in types.hpp.
inline Matrix2 mat_from_u2_params(const U2Params &p)
{
    const Complex g = std::polar(1.0, p.global_phase);
    const double c = std::cos(0.5 * p.theta);
    const double s = std::sin(0.5 * p.theta);
    return {g * c,
            -g * std::polar(1.0, p.lambda) * s,
            g * std::polar(1.0, p.phi) * s,
            g * std::polar(1.0, p.phi + p.lambda) * c};
}

inline Matrix2 mat_multiply(const Matrix2 &a, const Matrix2 &b) // a*b
{
    return {a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3]};
}

// ------------------------------------------------------------ dense matrices

// Row-major 2^t x 2^t matrix over t target qubits.
struct DenseMatrix
{
    int targets = 0;
    std::vector<Complex> values;

    DenseMatrix() = default;

    explicit DenseMatrix(int target_count)
        : targets(target_count), values(std::size_t{1} << (2 * target_count), Complex{0.0, 0.0})
    {
        if (target_count < 1 || target_count > 15)
        {
            throw std::invalid_argument("DenseMatrix supports 1..15 target qubits.");
        }
    }

    std::size_t dimension() const { return std::size_t{1} << targets; }

    Complex &at(std::size_t row, std::size_t col) { return values[row * dimension() + col]; }
    const Complex &at(std::size_t row, std::size_t col) const { return values[row * dimension() + col]; }
};

inline DenseMatrix identity_matrix(int target_count)
{
    DenseMatrix m(target_count);
    for (std::size_t i = 0; i < m.dimension(); ++i)
    {
        m.at(i, i) = Complex{1.0, 0.0};
    }
    return m;
}

// Embed a single-qubit matrix acting on local bit `bit` of a t-target matrix.
inline DenseMatrix single_qubit_on(int target_count, int bit, const Matrix2 &g)
{
    DenseMatrix m(target_count);
    const std::size_t dim = m.dimension();
    const std::size_t mask = std::size_t{1} << bit;
    for (std::size_t col = 0; col < dim; ++col)
    {
        const int in_bit = (col & mask) != 0 ? 1 : 0;
        for (int out_bit = 0; out_bit < 2; ++out_bit)
        {
            const Complex amp = g[static_cast<std::size_t>(2 * out_bit + in_bit)];
            if (amp == Complex{0.0, 0.0}) { continue; }
            const std::size_t row = out_bit ? (col | mask) : (col & ~mask);
            m.at(row, col) += amp;
        }
    }
    return m;
}

// CNOT with local control bit `control` and target bit `target`.
inline DenseMatrix cnot_on(int target_count, int control, int target)
{
    DenseMatrix m(target_count);
    const std::size_t dim = m.dimension();
    const std::size_t cmask = std::size_t{1} << control;
    const std::size_t tmask = std::size_t{1} << target;
    for (std::size_t col = 0; col < dim; ++col)
    {
        const std::size_t row = (col & cmask) ? (col ^ tmask) : col;
        m.at(row, col) = Complex{1.0, 0.0};
    }
    return m;
}

// Single-qubit gate on local bit `target`, applied only when local bit
// `control` is 1.
inline DenseMatrix controlled_single_qubit_on(int target_count, int control, int target,
                                              const Matrix2 &g)
{
    DenseMatrix m(target_count);
    const std::size_t dim = m.dimension();
    const std::size_t cmask = std::size_t{1} << control;
    const std::size_t tmask = std::size_t{1} << target;
    for (std::size_t col = 0; col < dim; ++col)
    {
        if ((col & cmask) == 0)
        {
            m.at(col, col) = Complex{1.0, 0.0};
            continue;
        }
        const int in_bit = (col & tmask) != 0 ? 1 : 0;
        for (int out_bit = 0; out_bit < 2; ++out_bit)
        {
            const Complex amp = g[static_cast<std::size_t>(2 * out_bit + in_bit)];
            if (amp == Complex{0.0, 0.0}) { continue; }
            const std::size_t row = out_bit ? (col | tmask) : (col & ~tmask);
            m.at(row, col) += amp;
        }
    }
    return m;
}

inline DenseMatrix matrix_product(const DenseMatrix &a, const DenseMatrix &b) // a*b
{
    if (a.targets != b.targets)
    {
        throw std::invalid_argument("matrix_product requires equal target counts.");
    }
    DenseMatrix out(a.targets);
    const std::size_t dim = out.dimension();
    for (std::size_t i = 0; i < dim; ++i)
    {
        for (std::size_t k = 0; k < dim; ++k)
        {
            const Complex aik = a.at(i, k);
            if (aik == Complex{0.0, 0.0}) { continue; }
            for (std::size_t j = 0; j < dim; ++j)
            {
                out.at(i, j) += aik * b.at(k, j);
            }
        }
    }
    return out;
}

// Tensor product of gates on *disjoint* target sets.
//
// The result's low index bits are `low`'s targets and its high bits are
// `high`'s, matching a concatenated target list {low_targets..., high_targets...}.
// This is the operation behind optimization #5: because the product is a
// Kronecker product of the individual bond gates, applying it is identical to
// applying those bonds separately, and it cannot entangle across the factors.
inline DenseMatrix tensor_product(const DenseMatrix &low, const DenseMatrix &high)
{
    DenseMatrix out(low.targets + high.targets);
    const std::size_t low_dim = low.dimension();
    const std::size_t high_dim = high.dimension();
    for (std::size_t hr = 0; hr < high_dim; ++hr)
    {
        for (std::size_t hc = 0; hc < high_dim; ++hc)
        {
            const Complex hv = high.at(hr, hc);
            if (hv == Complex{0.0, 0.0}) { continue; }
            for (std::size_t lr = 0; lr < low_dim; ++lr)
            {
                for (std::size_t lc = 0; lc < low_dim; ++lc)
                {
                    out.at(hr * low_dim + lr, hc * low_dim + lc) = hv * low.at(lr, lc);
                }
            }
        }
    }
    return out;
}

} // namespace mipt::cusv
