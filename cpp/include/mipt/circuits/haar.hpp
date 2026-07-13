#pragma once

#include "mipt/types.hpp"

#include <cudaq.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <vector>

namespace mipt
{
using ComplexD = std::complex<double>;
using Haar4 = std::array<ComplexD, 16>;
using Matrix2 = std::array<ComplexD, 4>;

struct HaarBondGate
{
    int q0 = 0; // Local least-significant basis bit in the historical Haar path.
    int q1 = 0; // Local most-significant basis bit / multiplexor control.
    U2Params a0;
    U2Params a1;
    U2Params b0;
    U2Params b1;
    double theta0 = 0.0;
    double theta1 = 0.0;
};

struct HaarLayer
{
    std::vector<HaarBondGate> gates;
    std::vector<int> measure_flags;
};

inline Haar4 haar_unitary_4(std::mt19937 &rng)
{
    // Draw U(4) from the Haar measure by QR/Gram-Schmidt orthonormalizing
    // four independent complex Gaussian columns. The matrix is row-major.
    std::normal_distribution<double> normal(0.0, 1.0);
    Haar4 q{};

    for (int col = 0; col < 4; ++col)
    {
        std::array<ComplexD, 4> v{};
        for (int row = 0; row < 4; ++row)
        {
            v[row] = ComplexD(normal(rng), normal(rng));
        }

        for (int prev = 0; prev < col; ++prev)
        {
            ComplexD dot = 0.0;
            for (int row = 0; row < 4; ++row)
            {
                dot += std::conj(q[static_cast<std::size_t>(row * 4 + prev)]) * v[row];
            }
            for (int row = 0; row < 4; ++row)
            {
                v[row] -= q[static_cast<std::size_t>(row * 4 + prev)] * dot;
            }
        }

        double norm2 = 0.0;
        for (const auto &x : v)
        {
            norm2 += std::norm(x);
        }
        const double norm = std::sqrt(norm2);
        if (norm < 1e-14)
        {
            --col;
            continue;
        }

        for (int row = 0; row < 4; ++row)
        {
            q[static_cast<std::size_t>(row * 4 + col)] = v[row] / norm;
        }
    }
    return q;
}

inline Matrix2 matrix2_adjoint(const Matrix2 &a)
{
    return {std::conj(a[0]), std::conj(a[2]),
            std::conj(a[1]), std::conj(a[3])};
}

inline Matrix2 matrix2_multiply(const Matrix2 &a, const Matrix2 &b)
{
    return {
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    };
}

inline Matrix2 block2(const Haar4 &u, int block_row, int block_col)
{
    Matrix2 out{};
    for (int r = 0; r < 2; ++r)
    {
        for (int c = 0; c < 2; ++c)
        {
            out[static_cast<std::size_t>(2 * r + c)] =
                u[static_cast<std::size_t>((2 * block_row + r) * 4 + (2 * block_col + c))];
        }
    }
    return out;
}

inline void normalize_vector2(ComplexD &x, ComplexD &y)
{
    const double n = std::sqrt(std::norm(x) + std::norm(y));
    if (!(n > 0.0))
    {
        throw std::runtime_error("Degenerate vector in Haar U(4) cosine-sine decomposition.");
    }
    x /= n;
    y /= n;
}

inline HaarBondGate decompose_haar_u4_csd(int q0, int q1, const Haar4 &u)
{
    // Cosine-sine decomposition in the historical local basis ordering
    // {00,01,10,11} = {q1 q0}. Thus q1 selects the 2x2 blocks and q0 is
    // the target of each multiplexed one-qubit operation:
    // U = (A0 (+) A1) [ C -S; S C ] (B0 (+) B1).
    const Matrix2 x = block2(u, 0, 0);
    const Matrix2 u01 = block2(u, 0, 1);
    const Matrix2 u10 = block2(u, 1, 0);

    const Matrix2 xdag = matrix2_adjoint(x);
    const Matrix2 h = matrix2_multiply(xdag, x);
    const double h00 = std::real(h[0]);
    const double h11 = std::real(h[3]);
    const ComplexD h01 = h[1];
    const double trace = h00 + h11;
    const double disc = std::sqrt(std::max(0.0,
        (h00 - h11) * (h00 - h11) + 4.0 * std::norm(h01)));
    const double lambda0 = std::clamp(0.5 * (trace + disc), 0.0, 1.0);
    const double lambda1 = std::clamp(0.5 * (trace - disc), 0.0, 1.0);

    ComplexD v00, v10, v01, v11;
    if (std::abs(h01) > 1e-14)
    {
        v00 = h01;
        v10 = lambda0 - h00;
        normalize_vector2(v00, v10);
        // Orthogonal complement fixes the second singular vector exactly.
        v01 = -std::conj(v10);
        v11 = std::conj(v00);
    }
    else if (h00 >= h11)
    {
        v00 = 1.0; v10 = 0.0;
        v01 = 0.0; v11 = 1.0;
    }
    else
    {
        v00 = 0.0; v10 = 1.0;
        v01 = 1.0; v11 = 0.0;
    }

    const Matrix2 v{v00, v01, v10, v11};
    const double c0 = std::sqrt(lambda0);
    const double c1 = std::sqrt(lambda1);
    const double s0 = std::sqrt(std::max(0.0, 1.0 - lambda0));
    const double s1 = std::sqrt(std::max(0.0, 1.0 - lambda1));
    constexpr double eps = 1e-12;
    if (c0 < eps || c1 < eps || s0 < eps || s1 < eps)
    {
        // This has probability zero for a Haar-random matrix. Regenerate rather
        // than introducing an unstable pseudo-inverse into the exact circuit.
        throw std::runtime_error("Numerically singular Haar U(4) cosine-sine decomposition.");
    }

    // A0 = X V diag(1/c), B0 = V^dagger.
    Matrix2 a0 = matrix2_multiply(x, v);
    a0[0] /= c0; a0[2] /= c0;
    a0[1] /= c1; a0[3] /= c1;
    const Matrix2 b0 = matrix2_adjoint(v);

    // A1 = U10 B0^dagger diag(1/s) = U10 V diag(1/s).
    Matrix2 a1 = matrix2_multiply(u10, v);
    a1[0] /= s0; a1[2] /= s0;
    a1[1] /= s1; a1[3] /= s1;

    // B1 = -diag(1/s) A0^dagger U01.
    Matrix2 b1 = matrix2_multiply(matrix2_adjoint(a0), u01);
    b1[0] = -b1[0] / s0; b1[1] = -b1[1] / s0;
    b1[2] = -b1[2] / s1; b1[3] = -b1[3] / s1;

    HaarBondGate gate;
    gate.q0 = q0;
    gate.q1 = q1;
    gate.a0 = decompose_u2_to_u3_phase(a0);
    gate.a1 = decompose_u2_to_u3_phase(a1);
    gate.b0 = decompose_u2_to_u3_phase(b0);
    gate.b1 = decompose_u2_to_u3_phase(b1);
    gate.theta0 = 2.0 * std::atan2(s0, c0);
    gate.theta1 = 2.0 * std::atan2(s1, c1);
    return gate;
}

inline HaarBondGate random_haar_bond_gate(int q0, int q1, std::mt19937 &rng)
{
    for (;;)
    {
        try
        {
            return decompose_haar_u4_csd(q0, q1, haar_unitary_4(rng));
        }
        catch (const std::runtime_error &)
        {
            // Regenerate only for the measure-zero numerically singular case.
        }
    }
}

inline void fill_haar_layer(HaarLayer &layer,
                     int n,
                     int start,
                     double p,
                     bool closed,
                     std::mt19937 &rng)
{
    const int bond_stop = (closed && start == 1 && n > 2) ? n : (n - 1);
    const int bond_count = (bond_stop > start) ? ((bond_stop - start + 1) / 2) : 0;
    layer.gates.resize(static_cast<std::size_t>(bond_count));
    layer.measure_flags.resize(static_cast<std::size_t>(n));

    int b = 0;
    for (int i = start; i < bond_stop; i += 2)
    {
        const int j = (i + 1) % n;
        layer.gates[static_cast<std::size_t>(b++)] = random_haar_bond_gate(i, j, rng);
    }

    if (p <= 0.0)
    {
        std::fill(layer.measure_flags.begin(), layer.measure_flags.end(), 0);
    }
    else if (p >= 1.0)
    {
        std::fill(layer.measure_flags.begin(), layer.measure_flags.end(), 1);
    }
    else
    {
        std::bernoulli_distribution measure_dist(p);
        for (int q = 0; q < n; ++q)
        {
            layer.measure_flags[static_cast<std::size_t>(q)] = measure_dist(rng) ? 1 : 0;
        }
    }
}

inline void build_haar_layers(std::vector<HaarLayer> &layers,
                           int n,
                           int periods,
                           double p,
                           std::mt19937 &rng,
                           bool closed = true)
{
    if ((n % 2) != 0)
    {
        throw std::invalid_argument("Random Haar 1D simulation requires even n for periodic brickwork geometry.");
    }
    layers.resize(static_cast<std::size_t>(2 * periods));
    std::size_t idx = 0;
    for (int period = 0; period < periods; ++period)
    {
        fill_haar_layer(layers[idx++], n, 0, p, closed, rng);
        fill_haar_layer(layers[idx++], n, 1, p, closed, rng);
    }
}

struct HaarKernel1D
{
    void operator()(int n, const std::vector<HaarLayer> &layers) __qpu__
    {
        cudaq::qvector q(n);
        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            for (std::size_t gi = 0; gi < layers[layer].gates.size(); ++gi)
            {
                const auto gate = layers[layer].gates[gi];
                // Right block-diagonal factor B0 (+) B1.
                r1(gate.b1.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.b1.theta, gate.b1.phi, gate.b1.lambda,
                                q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);
                r1(gate.b0.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.b0.theta, gate.b0.phi, gate.b0.lambda,
                                q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);

                // Cosine-sine core: Ry(theta0) for control=0 and
                // Ry(theta1) for control=1.
                ry<cudaq::ctrl>(gate.theta1, q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);
                ry<cudaq::ctrl>(gate.theta0, q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);

                // Left block-diagonal factor A0 (+) A1.
                r1(gate.a1.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.a1.theta, gate.a1.phi, gate.a1.lambda,
                                q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);
                r1(gate.a0.global_phase, q[gate.q1]);
                u3<cudaq::ctrl>(gate.a0.theta, gate.a0.phi, gate.a0.lambda,
                                q[gate.q1], q[gate.q0]);
                x(q[gate.q1]);
            }

            for (int i = 0; i < n; ++i)
            {
                if (layers[layer].measure_flags[static_cast<std::size_t>(i)])
                {
                    mz(q[i]);
                }
            }
        }
    }
};

} // namespace mipt
