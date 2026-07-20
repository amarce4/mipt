#pragma once

#include "mipt/ancilla_rdm.hpp"
#include "mipt/backend.hpp"

#ifdef MIPT_ENABLE_CUDA_RHO
#include "mipt/cuda/ancilla_rdm4.hpp"
#endif

#include <cudaq.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::ancilla
{
struct Rdm4
{
    static constexpr std::size_t dimension = 16;
    std::array<std::complex<double>, dimension * dimension> values{};

    std::complex<double> &operator()(std::size_t row, std::size_t col)
    {
        return values[row * dimension + col];
    }

    const std::complex<double> &operator()(std::size_t row, std::size_t col) const
    {
        return values[row * dimension + col];
    }
};

inline double four_tail_fermion_reorder_sign(
    std::uint64_t environment,
    unsigned local)
{
    // The four retained modes are the final four modes. Reordering them ahead
    // of the environment creates one inversion for every occupied retained-mode
    // / occupied-environment-mode pair.
    return odd_popcount(environment) &&
           odd_popcount(static_cast<std::uint64_t>(local))
        ? -1.0
        : 1.0;
}

template <typename Real>
inline Rdm4 from_host_statevector_four_tail(
    const std::complex<Real> *state,
    int n_qubits,
    bool fermion_trace)
{
    if (state == nullptr)
    {
        throw std::invalid_argument(
            "Four-ancilla RDM received a null statevector.");
    }
    if (n_qubits < 4)
    {
        throw std::invalid_argument(
            "Four-ancilla RDM requires at least four total qubits.");
    }

    const int system_qubits = n_qubits - 4;
    const std::uint64_t environment_dimension =
        std::uint64_t{1} << system_qubits;
    Rdm4 rho;
    std::array<std::complex<double>, Rdm4::dimension> amplitudes{};

    for (std::uint64_t environment = 0;
         environment < environment_dimension;
         ++environment)
    {
        for (std::size_t local = 0; local < Rdm4::dimension; ++local)
        {
            const std::uint64_t index = environment |
                (static_cast<std::uint64_t>(local) << system_qubits);
            const double sign = fermion_trace
                ? four_tail_fermion_reorder_sign(
                      environment, static_cast<unsigned>(local))
                : 1.0;
            amplitudes[local] =
                static_cast<std::complex<double>>(state[index]) * sign;
        }

        for (std::size_t row = 0; row < Rdm4::dimension; ++row)
        {
            for (std::size_t col = 0; col < Rdm4::dimension; ++col)
            {
                rho(row, col) +=
                    amplitudes[row] * std::conj(amplitudes[col]);
            }
        }
    }
    return rho;
}

inline Rdm4 from_amplitude_queries_four_tail(
    cudaq::state &state,
    int n_qubits,
    bool fermion_trace)
{
    const int system_qubits = n_qubits - 4;
    const std::uint64_t environment_dimension =
        std::uint64_t{1} << system_qubits;
    const std::size_t amplitude_batch = static_cast<std::size_t>(
        backend::mps_amplitude_batch_size());
    const std::size_t environment_batch =
        std::max<std::size_t>(1, amplitude_batch / Rdm4::dimension);

    std::vector<std::vector<int>> basis_states;
    std::vector<double> signs;
    Rdm4 rho;

    for (std::uint64_t first_environment = 0;
         first_environment < environment_dimension;
         first_environment += environment_batch)
    {
        const std::uint64_t count = std::min<std::uint64_t>(
            environment_batch,
            environment_dimension - first_environment);
        basis_states.clear();
        signs.clear();
        basis_states.reserve(
            static_cast<std::size_t>(Rdm4::dimension * count));
        signs.reserve(
            static_cast<std::size_t>(Rdm4::dimension * count));

        for (std::uint64_t offset = 0; offset < count; ++offset)
        {
            const std::uint64_t environment = first_environment + offset;
            for (std::size_t local = 0; local < Rdm4::dimension; ++local)
            {
                const std::uint64_t index = environment |
                    (static_cast<std::uint64_t>(local) << system_qubits);
                basis_states.push_back(
                    basis_bits_from_index(n_qubits, index));
                signs.push_back(
                    fermion_trace
                        ? four_tail_fermion_reorder_sign(
                              environment, static_cast<unsigned>(local))
                        : 1.0);
            }
        }

        const auto queried = state.amplitudes(basis_states);
        if (queried.size() != basis_states.size())
        {
            throw std::runtime_error(
                "CUDA-Q state.amplitudes returned an unexpected count for the four-ancilla RDM.");
        }

        for (std::size_t environment_index = 0;
             environment_index < static_cast<std::size_t>(count);
             ++environment_index)
        {
            const std::size_t base =
                environment_index * Rdm4::dimension;
            for (std::size_t row = 0; row < Rdm4::dimension; ++row)
            {
                const std::complex<double> a =
                    queried[base + row] * signs[base + row];
                for (std::size_t col = 0; col < Rdm4::dimension; ++col)
                {
                    const std::complex<double> b =
                        queried[base + col] * signs[base + col];
                    rho(row, col) += a * std::conj(b);
                }
            }
        }
    }
    return rho;
}

inline Rdm4 reduced_density_matrix_four_tail(
    cudaq::state &state,
    int n_qubits,
    bool fermion_trace = false)
{
    if (n_qubits < 4)
    {
        throw std::invalid_argument(
            "Four-ancilla RDM requires at least four total qubits.");
    }

    const std::uint64_t dimension_u64 = checked_dimension(n_qubits);
    if (dimension_u64 >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument(
            "Four-ancilla statevector dimension does not fit in size_t.");
    }
    const std::size_t dimension = static_cast<std::size_t>(dimension_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();
    const bool dense_rank1 =
        tensor.get_rank() == 1 &&
        tensor.get_num_elements() >= dimension &&
        tensor.data != nullptr;

    if (!dense_rank1)
    {
        backend::warn_mps_amplitude_path_once(
            "four-ancilla reduced density matrix",
            n_qubits,
            false,
            0);
        return from_amplitude_queries_four_tail(
            state, n_qubits, fermion_trace);
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    if (state.is_on_gpu())
    {
        double values[2 * Rdm4::dimension * Rdm4::dimension]{};
        int status = 0;
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = fermion_trace
                ? mipt_cuda_rho4_tail_complex_fermion_f64(
                      tensor.data, n_qubits, values)
                : mipt_cuda_rho4_tail_complex_f64(
                      tensor.data, n_qubits, values);
        }
        else
        {
            status = fermion_trace
                ? mipt_cuda_rho4_tail_complex_fermion_f32(
                      tensor.data, n_qubits, values)
                : mipt_cuda_rho4_tail_complex_f32(
                      tensor.data, n_qubits, values);
        }
        if (status != 0)
        {
            throw std::runtime_error(
                "CUDA four-ancilla reduced-density-matrix calculation failed; status=" +
                std::to_string(status));
        }

        Rdm4 rho;
        for (std::size_t i = 0; i < rho.values.size(); ++i)
        {
            rho.values[i] = {values[2 * i], values[2 * i + 1]};
        }
        return rho;
    }
#endif

    if (state.is_on_gpu())
    {
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            std::vector<std::complex<double>> host_state(dimension);
            state.to_host(host_state.data(), host_state.size());
            return from_host_statevector_four_tail(
                host_state.data(), n_qubits, fermion_trace);
        }

        std::vector<std::complex<float>> host_state(dimension);
        state.to_host(host_state.data(), host_state.size());
        return from_host_statevector_four_tail(
            host_state.data(), n_qubits, fermion_trace);
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        return from_host_statevector_four_tail(
            reinterpret_cast<const std::complex<double> *>(tensor.data),
            n_qubits,
            fermion_trace);
    }

    return from_host_statevector_four_tail(
        reinterpret_cast<const std::complex<float> *>(tensor.data),
        n_qubits,
        fermion_trace);
}
} // namespace mipt::ancilla
