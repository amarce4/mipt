#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>

namespace mipt::free_energy
{
struct MeasurementResult
{
    int outcome = 0;
    double conditional_probability = 1.0;
    double surprisal = 0.0;
};

template <typename Real>
inline MeasurementResult measure_collapse_host(
    std::complex<Real> *state,
    int n,
    int measured_qubit,
    double uniform_01)
{
    if (state == nullptr || n < 1 || n >= 63 || measured_qubit < 0 ||
        measured_qubit >= n || !(uniform_01 >= 0.0 && uniform_01 < 1.0))
    {
        throw std::invalid_argument("Invalid sequential-measurement request.");
    }

    const std::uint64_t dimension = std::uint64_t{1} << n;
    const std::uint64_t mask = std::uint64_t{1} << measured_qubit;
    long double total_norm_sum = 0.0L;
    long double branch_one_norm_sum = 0.0L;
    for (std::uint64_t index = 0; index < dimension; ++index)
    {
        const long double weight =
            static_cast<long double>(std::norm(state[index]));
        total_norm_sum += weight;
        if ((index & mask) != 0)
        {
            branch_one_norm_sum += weight;
        }
    }

    const double total_norm = static_cast<double>(total_norm_sum);
    const double branch_one_norm = static_cast<double>(branch_one_norm_sum);
    if (!(total_norm > 0.0) || !std::isfinite(total_norm) ||
        !std::isfinite(branch_one_norm) || branch_one_norm < 0.0 ||
        branch_one_norm > total_norm * (1.0 + 1.0e-10))
    {
        throw std::runtime_error(
            "Sequential measurement received invalid branch norms.");
    }
    const double probability_one =
        std::clamp(branch_one_norm / total_norm, 0.0, 1.0);

    MeasurementResult result;
    result.outcome = uniform_01 < probability_one ? 1 : 0;
    result.conditional_probability =
        result.outcome ? probability_one : 1.0 - probability_one;
    const double realized_branch_norm =
        result.outcome ? branch_one_norm : total_norm - branch_one_norm;
    if (!(result.conditional_probability > 0.0) ||
        !std::isfinite(result.conditional_probability))
    {
        throw std::runtime_error(
            "Sampled an outcome with invalid conditional probability.");
    }
    if (!(realized_branch_norm > 0.0) ||
        !std::isfinite(realized_branch_norm))
    {
        throw std::runtime_error(
            "Sampled an outcome with invalid measurement branch norm.");
    }

    // The returned probability is conditional on a normalized input state, but
    // collapse uses the raw branch weight. This removes any small norm drift
    // accumulated by finite-precision unitary evolution.
    const Real inverse_norm =
        static_cast<Real>(1.0 / std::sqrt(realized_branch_norm));
    for (std::uint64_t index = 0; index < dimension; ++index)
    {
        const int bit = (index & mask) != 0 ? 1 : 0;
        if (bit == result.outcome)
        {
            state[index] *= inverse_norm;
        }
        else
        {
            state[index] = std::complex<Real>{0, 0};
        }
    }
    result.surprisal = -std::log(result.conditional_probability);
    return result;
}
} // namespace mipt::free_energy
