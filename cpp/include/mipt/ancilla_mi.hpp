#pragma once

#include "mipt/ancilla_rdm.hpp"
#include "mipt/util/spectral.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace mipt::ancilla
{
struct TwoProbeObservables
{
    double joint_entropy = 0.0;
    double mutual_information = 0.0;
    double negativity = 0.0;
    double log_negativity = 0.0;
};

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

// Eigenvalues of a 4x4 complex Hermitian ancilla block, as a flat row-major
// array for mipt::util::hermitian_eigenvalues.  This used to embed the block in
// a real symmetric 8x8, which duplicates every eigenvalue and quadruples the
// work of the diagonalization.
inline std::array<double, 4> ancilla_eigenvalues(const Rdm2 &matrix)
{
    std::array<std::complex<double>, 16> flat{};
    for (std::size_t row = 0; row < Rdm2::dimension; ++row)
    {
        for (std::size_t col = 0; col < Rdm2::dimension; ++col)
        {
            flat[row * Rdm2::dimension + col] = matrix(row, col);
        }
    }

    std::array<double, 4> eigenvalues{};
    if (!mipt::util::hermitian_eigenvalues<4>(flat, eigenvalues))
    {
        throw std::runtime_error(
            "The 4x4 ancilla density-matrix eigensolver did not converge.");
    }
    return eigenvalues;
}

inline Rdm2 normalized_hermitian_rdm2(const Rdm2 &rho)
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

    Rdm2 normalized;
    for (std::size_t row = 0; row < Rdm2::dimension; ++row)
    {
        for (std::size_t col = row; col < Rdm2::dimension; ++col)
        {
            std::complex<double> value =
                0.5 * (rho(row, col) + std::conj(rho(col, row))) / trace;
            if (row == col)
            {
                value = {std::real(value), 0.0};
            }
            normalized(row, col) = value;
            normalized(col, row) = std::conj(value);
        }
    }
    return normalized;
}

inline double entropy_from_normalized_rdm2(const Rdm2 &rho)
{
    const auto eigenvalues = ancilla_eigenvalues(rho);

    double entropy = 0.0;
    double eigenvalue_sum = 0.0;
    constexpr double negative_tolerance = 2e-6;
    for (double lambda : eigenvalues)
    {
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
            entropy -= lambda * std::log(lambda);
        }
    }

    if (std::abs(eigenvalue_sum - 1.0) > 1e-5)
    {
        throw std::runtime_error(
            "Two-ancilla eigenspectrum has invalid normalization.");
    }
    return entropy;
}

inline double entropy_from_rdm2(const Rdm2 &rho)
{
    return entropy_from_normalized_rdm2(normalized_hermitian_rdm2(rho));
}

// Trace norm of a partially transposed two-ancilla block.
//
// `hermitian` selects the cheap route: an ordinary partial transpose of a
// Hermitian matrix stays Hermitian, so its singular values are the absolute
// eigenvalues and no Gram matrix is needed.  Only the fermionic partial
// transpose, which multiplies parity-violating entries by i, is non-Hermitian.
inline double trace_norm_rdm2(const Rdm2 &matrix, bool hermitian)
{
    std::array<std::complex<double>, 16> flat{};
    for (std::size_t row = 0; row < Rdm2::dimension; ++row)
    {
        for (std::size_t col = 0; col < Rdm2::dimension; ++col)
        {
            flat[row * Rdm2::dimension + col] = matrix(row, col);
        }
    }

    double norm = 0.0;
    const bool converged = hermitian
        ? mipt::util::hermitian_trace_norm<4>(flat, norm)
        : mipt::util::gram_trace_norm<4>(flat, norm);
    if (!converged || !std::isfinite(norm))
    {
        throw std::runtime_error(
            "Two-ancilla partial-transpose eigensolver did not converge.");
    }
    return norm;
}

inline double negativity_from_normalized_rdm2(
    const Rdm2 &rho,
    bool fermion_trace)
{
    Rdm2 partial_transpose;
    constexpr int mask = 1; // transpose ancilla A, local bit 0
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const int source_row = (row & ~mask) | (col & mask);
            const int source_col = (col & ~mask) | (row & mask);
            std::complex<double> value = rho(source_row, source_col);

            // Match the fGMN/two-site convention for fermionic partial transpose.
            if (fermion_trace && (((row & mask) != 0) != ((col & mask) != 0)))
            {
                value *= std::complex<double>(0.0, 1.0);
            }
            partial_transpose(row, col) = value;
        }
    }

    double value = 0.5 * (trace_norm_rdm2(partial_transpose, !fermion_trace) - 1.0);
    if (value < 0.0 && value > -2e-8)
    {
        value = 0.0;
    }
    if (!std::isfinite(value) || value < -2e-8)
    {
        throw std::runtime_error(
            "Ancilla negativity is invalid: " + std::to_string(value));
    }
    return std::max(0.0, value);
}

inline double negativity_from_rdm2(const Rdm2 &rho, bool fermion_trace)
{
    return negativity_from_normalized_rdm2(
        normalized_hermitian_rdm2(rho), fermion_trace);
}

inline TwoProbeObservables two_probe_observables(
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
    const Rdm2 normalized_rho_ab = normalized_hermitian_rdm2(rho_ab);

    const double joint_entropy =
        entropy_from_normalized_rdm2(normalized_rho_ab);
    double mi = entropy_from_rdm(rho_a) + entropy_from_rdm(rho_b) -
                joint_entropy;
    if (mi < 0.0 && mi > -2e-6)
    {
        mi = 0.0;
    }
    if (!std::isfinite(mi) || mi < -2e-6)
    {
        throw std::runtime_error(
            "Ancilla mutual information is invalid: " + std::to_string(mi));
    }

    const double negativity = negativity_from_normalized_rdm2(
        normalized_rho_ab, fermion_trace);
    const double log_negativity = std::log1p(2.0 * negativity);
    if (!std::isfinite(log_negativity) || log_negativity < 0.0)
    {
        throw std::runtime_error(
            "Ancilla logarithmic negativity is invalid: " +
            std::to_string(log_negativity));
    }

    return {joint_entropy, std::max(0.0, mi), negativity, log_negativity};
}

inline double mutual_information(
    cudaq::state &state,
    int n_qubits,
    int ancilla_a,
    int ancilla_b,
    bool fermion_trace)
{
    return two_probe_observables(
        state, n_qubits, ancilla_a, ancilla_b, fermion_trace)
        .mutual_information;
}
} // namespace mipt::ancilla
