#pragma once

#include "mipt/ancilla_rdm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace mipt::ancilla
{
inline double entropy_from_rdm(const Rdm &rho)
{
    double rho00 = rho.rho00;
    double rho11 = rho.rho11;
    std::complex<double> rho01 = rho.rho01;
    const double trace = rho00 + rho11;
    if (!(trace > 0.0) || !std::isfinite(trace))
    {
        throw std::runtime_error("Ancilla reduced density matrix has invalid trace.");
    }

    rho00 /= trace;
    rho11 /= trace;
    rho01 /= trace;
    const double discriminant = std::clamp(
        std::sqrt(std::max(
            0.0,
            (rho00 - rho11) * (rho00 - rho11) +
                4.0 * std::norm(rho01))),
        0.0,
        1.0);
    const double lambda0 = 0.5 * (1.0 + discriminant);
    const double lambda1 = 0.5 * (1.0 - discriminant);

    auto entropy_term = [](double lambda)
    {
        return lambda > 0.0 ? -lambda * std::log(lambda) : 0.0;
    };
    return entropy_term(lambda0) + entropy_term(lambda1);
}

inline void jacobi_eigenvalues_symmetric_8x8(
    std::array<std::array<double, 8>, 8> &matrix)
{
    constexpr int dimension = 8;
    constexpr int max_rotations = 2048;
    constexpr double tolerance = 1e-14;

    for (int rotation = 0; rotation < max_rotations; ++rotation)
    {
        int p = 0;
        int q = 1;
        double largest = std::abs(matrix[0][1]);
        for (int row = 0; row < dimension; ++row)
        {
            for (int col = row + 1; col < dimension; ++col)
            {
                const double candidate = std::abs(matrix[row][col]);
                if (candidate > largest)
                {
                    largest = candidate;
                    p = row;
                    q = col;
                }
            }
        }
        if (largest < tolerance)
        {
            return;
        }

        const double app = matrix[p][p];
        const double aqq = matrix[q][q];
        const double apq = matrix[p][q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = tau >= 0.0
            ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
            : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < dimension; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }
            const double akp = matrix[k][p];
            const double akq = matrix[k][q];
            const double new_kp = c * akp - s * akq;
            const double new_kq = s * akp + c * akq;
            matrix[k][p] = matrix[p][k] = new_kp;
            matrix[k][q] = matrix[q][k] = new_kq;
        }

        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = matrix[q][p] = 0.0;
    }

    throw std::runtime_error(
        "The 4x4 ancilla density-matrix eigensolver did not converge.");
}

inline double entropy_from_rdm2(const Rdm2 &rho)
{
    double trace = 0.0;
    for (std::size_t i = 0; i < Rdm2::dimension; ++i)
    {
        trace += std::real(rho(i, i));
    }
    if (!(trace > 0.0) || !std::isfinite(trace))
    {
        throw std::runtime_error(
            "Two-ancilla reduced density matrix has invalid trace.");
    }

    std::array<std::array<double, 8>, 8> real_embedding{};
    for (std::size_t row = 0; row < Rdm2::dimension; ++row)
    {
        for (std::size_t col = 0; col < Rdm2::dimension; ++col)
        {
            const std::complex<double> value =
                0.5 * (rho(row, col) + std::conj(rho(col, row))) / trace;
            const double real = std::real(value);
            const double imag = std::imag(value);
            real_embedding[row][col] = real;
            real_embedding[row][col + 4] = -imag;
            real_embedding[row + 4][col] = imag;
            real_embedding[row + 4][col + 4] = real;
        }
    }

    jacobi_eigenvalues_symmetric_8x8(real_embedding);

    double entropy_twice = 0.0;
    double eigenvalue_sum = 0.0;
    constexpr double negative_tolerance = 2e-6;
    for (std::size_t i = 0; i < 8; ++i)
    {
        double lambda = real_embedding[i][i];
        if (!std::isfinite(lambda))
        {
            throw std::runtime_error(
                "Two-ancilla density matrix produced a non-finite eigenvalue.");
        }
        if (lambda < -negative_tolerance)
        {
            throw std::runtime_error(
                "Two-ancilla density matrix is not positive semidefinite; eigenvalue=" +
                std::to_string(lambda));
        }
        lambda = std::max(0.0, lambda);
        eigenvalue_sum += lambda;
        if (lambda > 0.0)
        {
            entropy_twice -= lambda * std::log(lambda);
        }
    }

    if (std::abs(eigenvalue_sum - 2.0) > 2e-5)
    {
        throw std::runtime_error(
            "Two-ancilla eigenspectrum has invalid normalization.");
    }
    return 0.5 * entropy_twice;
}

inline double mutual_information(
    cudaq::state &state,
    int n_qubits,
    int ancilla_a,
    int ancilla_b,
    bool fermion_trace)
{
    const Rdm rho_a = reduced_density_matrix(
        state, n_qubits, ancilla_a, fermion_trace);
    const Rdm rho_b = reduced_density_matrix(
        state, n_qubits, ancilla_b, fermion_trace);
    const Rdm2 rho_ab = reduced_density_matrix_two(
        state, n_qubits, ancilla_a, ancilla_b, fermion_trace);

    double value = entropy_from_rdm(rho_a) + entropy_from_rdm(rho_b) -
                   entropy_from_rdm2(rho_ab);
    if (value < 0.0 && value > -2e-6)
    {
        value = 0.0;
    }
    if (!std::isfinite(value) || value < -2e-6)
    {
        throw std::runtime_error(
            "Ancilla mutual information is invalid: " + std::to_string(value));
    }
    return value;
}
} // namespace mipt::ancilla
