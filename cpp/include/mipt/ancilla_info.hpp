#pragma once

#include "mipt/ancilla_mi.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mipt::ancilla
{
struct ProbeInformation
{
    double entropy = 0.0;
    double i2 = 0.0;
    double i3 = 0.0;
    double i4 = 0.0;
};

struct SmallRdm
{
    explicit SmallRdm(int dimension_in)
        : dimension(dimension_in),
          values(static_cast<std::size_t>(dimension_in * dimension_in))
    {
    }

    std::complex<double> &operator()(int row, int col)
    {
        return values[static_cast<std::size_t>(dimension * row + col)];
    }

    const std::complex<double> &operator()(int row, int col) const
    {
        return values[static_cast<std::size_t>(dimension * row + col)];
    }

    int dimension = 0;
    std::vector<std::complex<double>> values;
};

inline int local_reordered_position(int mode, unsigned retained_mask)
{
    int retained_position = 0;
    int environment_position = 0;
    for (int candidate = 0; candidate < 4; ++candidate)
    {
        if ((retained_mask >> candidate) & 1u)
        {
            if (candidate == mode)
            {
                return retained_position;
            }
            ++retained_position;
        }
    }
    for (int candidate = 0; candidate < 4; ++candidate)
    {
        if (((retained_mask >> candidate) & 1u) == 0)
        {
            if (candidate == mode)
            {
                return retained_position + environment_position;
            }
            ++environment_position;
        }
    }
    throw std::logic_error("Local mode is outside the four-mode basis.");
}

inline double local_fermion_reorder_sign(unsigned occupation,
                                         unsigned retained_mask)
{
    unsigned seen = 0;
    int parity = 0;
    for (int mode = 0; mode < 4; ++mode)
    {
        if (((occupation >> mode) & 1u) == 0)
        {
            continue;
        }
        const int position = local_reordered_position(mode, retained_mask);
        const unsigned lower_or_equal = (1u << (position + 1)) - 1u;
#if defined(__GNUG__) || defined(__clang__)
        parity ^= (__builtin_popcount(seen & ~lower_or_equal) & 1);
#else
        unsigned inversions = seen & ~lower_or_equal;
        while (inversions)
        {
            parity ^= 1;
            inversions &= inversions - 1;
        }
#endif
        seen |= 1u << position;
    }
    return parity ? -1.0 : 1.0;
}

inline int embed_local_bits(int retained_value,
                            int environment_value,
                            unsigned retained_mask)
{
    int full = 0;
    int retained_bit = 0;
    int environment_bit = 0;
    for (int mode = 0; mode < 4; ++mode)
    {
        const bool retained = ((retained_mask >> mode) & 1u) != 0;
        const int bit = retained
            ? ((retained_value >> retained_bit++) & 1)
            : ((environment_value >> environment_bit++) & 1);
        full |= bit << mode;
    }
    return full;
}

inline SmallRdm partial_trace_four(const Rdm4 &rho,
                                   unsigned retained_mask,
                                   bool fermion_trace)
{
    retained_mask &= 0xFu;
    if (retained_mask == 0)
    {
        throw std::invalid_argument("At least one ancilla must be retained.");
    }
#if defined(__GNUG__) || defined(__clang__)
    const int retained_count = __builtin_popcount(retained_mask);
#else
    int retained_count = 0;
    for (unsigned value = retained_mask; value; value &= value - 1)
    {
        ++retained_count;
    }
#endif
    const int dimension = 1 << retained_count;
    const int environment_dimension = 1 << (4 - retained_count);
    SmallRdm reduced(dimension);

    for (int row = 0; row < dimension; ++row)
    {
        for (int col = 0; col < dimension; ++col)
        {
            std::complex<double> value = 0.0;
            for (int environment = 0;
                 environment < environment_dimension;
                 ++environment)
            {
                const int full_row = embed_local_bits(
                    row, environment, retained_mask);
                const int full_col = embed_local_bits(
                    col, environment, retained_mask);
                const double sign = fermion_trace
                    ? local_fermion_reorder_sign(
                          static_cast<unsigned>(full_row), retained_mask) *
                          local_fermion_reorder_sign(
                              static_cast<unsigned>(full_col), retained_mask)
                    : 1.0;
                value += sign * rho(
                    static_cast<std::size_t>(full_row),
                    static_cast<std::size_t>(full_col));
            }
            reduced(row, col) = value;
        }
    }
    return reduced;
}

inline double entropy_from_small_rdm(const SmallRdm &input)
{
    const int d = input.dimension;
    if (d <= 0)
    {
        throw std::invalid_argument("Reduced density matrix is empty.");
    }
    double trace = 0.0;
    for (int i = 0; i < d; ++i)
    {
        trace += std::real(input(i, i));
    }
    if (!(trace > 0.0) || !std::isfinite(trace))
    {
        throw std::runtime_error("Ancilla reduced density matrix has invalid trace.");
    }

    const int n = 2 * d;
    std::vector<double> matrix(static_cast<std::size_t>(n * n), 0.0);
    auto at = [&](int row, int col) -> double &
    {
        return matrix[static_cast<std::size_t>(n * row + col)];
    };
    for (int row = 0; row < d; ++row)
    {
        for (int col = 0; col < d; ++col)
        {
            const std::complex<double> value =
                0.5 * (input(row, col) + std::conj(input(col, row))) / trace;
            at(row, col) = std::real(value);
            at(row, col + d) = -std::imag(value);
            at(row + d, col) = std::imag(value);
            at(row + d, col + d) = std::real(value);
        }
    }

    constexpr int max_sweeps = 128;
    constexpr double tolerance = 1e-13;
    bool converged = false;
    for (int sweep = 0; sweep < max_sweeps; ++sweep)
    {
        double largest = 0.0;
        for (int p = 0; p < n; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                const double apq = at(p, q);
                largest = std::max(largest, std::abs(apq));
                if (std::abs(apq) <= tolerance)
                {
                    continue;
                }
                const double app = at(p, p);
                const double aqq = at(q, q);
                const double tau = (aqq - app) / (2.0 * apq);
                const double tangent = tau >= 0.0
                    ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                    : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = tangent * cosine;

                for (int k = 0; k < n; ++k)
                {
                    if (k == p || k == q)
                    {
                        continue;
                    }
                    const double akp = at(k, p);
                    const double akq = at(k, q);
                    at(k, p) = at(p, k) = cosine * akp - sine * akq;
                    at(k, q) = at(q, k) = sine * akp + cosine * akq;
                }
                at(p, p) = cosine * cosine * app -
                            2.0 * sine * cosine * apq +
                            sine * sine * aqq;
                at(q, q) = sine * sine * app +
                            2.0 * sine * cosine * apq +
                            cosine * cosine * aqq;
                at(p, q) = at(q, p) = 0.0;
            }
        }
        if (largest <= tolerance)
        {
            converged = true;
            break;
        }
    }
    if (!converged)
    {
        throw std::runtime_error("Ancilla density-matrix eigensolver did not converge.");
    }

    double entropy_twice = 0.0;
    double sum_twice = 0.0;
    constexpr double negative_tolerance = 2e-5;
    for (int i = 0; i < n; ++i)
    {
        double eigenvalue = at(i, i);
        if (!std::isfinite(eigenvalue) || eigenvalue < -negative_tolerance)
        {
            throw std::runtime_error(
                "Ancilla density matrix is not positive semidefinite.");
        }
        eigenvalue = std::max(0.0, eigenvalue);
        sum_twice += eigenvalue;
        if (eigenvalue > 0.0)
        {
            entropy_twice -= eigenvalue * std::log(eigenvalue);
        }
    }
    if (std::abs(sum_twice - 2.0) > 2e-4)
    {
        throw std::runtime_error("Ancilla eigenspectrum has invalid normalization.");
    }
    return 0.5 * entropy_twice;
}

inline ProbeInformation four_probe_information(cudaq::state &state,
                                               int n_qubits,
                                               const std::array<int, 4> &references,
                                               bool fermion_trace)
{
    const Rdm4 rho = reduced_density_matrix_four(
        state, n_qubits, references, fermion_trace);
    std::array<double, 16> entropies{};
    for (unsigned mask = 1; mask < 16; ++mask)
    {
        entropies[mask] = entropy_from_small_rdm(
            partial_trace_four(rho, mask, fermion_trace));
    }

    double i2_sum = 0.0;
    double i3_sum = 0.0;
    int pairs = 0;
    int triplets = 0;
    double i4 = 0.0;
    for (unsigned mask = 1; mask < 16; ++mask)
    {
#if defined(__GNUG__) || defined(__clang__)
        const int count = __builtin_popcount(mask);
#else
        int count = 0;
        for (unsigned value = mask; value; value &= value - 1)
        {
            ++count;
        }
#endif
        i4 += (count % 2 == 1 ? 1.0 : -1.0) * entropies[mask];
        if (count == 2)
        {
            double pair_information = -entropies[mask];
            for (int bit = 0; bit < 4; ++bit)
            {
                if ((mask >> bit) & 1u)
                {
                    pair_information += entropies[1u << bit];
                }
            }
            i2_sum += pair_information;
            ++pairs;
        }
        else if (count == 3)
        {
            double tripartite = entropies[mask];
            for (unsigned subset = mask; subset; subset = (subset - 1) & mask)
            {
#if defined(__GNUG__) || defined(__clang__)
                const int subset_count = __builtin_popcount(subset);
#else
                int subset_count = 0;
                for (unsigned value = subset; value; value &= value - 1)
                {
                    ++subset_count;
                }
#endif
                if (subset_count == 1)
                {
                    tripartite += entropies[subset];
                }
                else if (subset_count == 2)
                {
                    tripartite -= entropies[subset];
                }
            }
            i3_sum += tripartite;
            ++triplets;
        }
    }
    return {
        entropies[15],
        i2_sum / static_cast<double>(pairs),
        i3_sum / static_cast<double>(triplets),
        i4};
}

inline ProbeInformation two_probe_information(cudaq::state &state,
                                              int n_qubits,
                                              int reference_a,
                                              int reference_b,
                                              bool fermion_trace)
{
    const auto values = two_probe_observables(
        state,
        n_qubits,
        reference_a,
        reference_b,
        fermion_trace);
    return {values.joint_entropy, values.mutual_information, 0.0, 0.0};
}
} // namespace mipt::ancilla
