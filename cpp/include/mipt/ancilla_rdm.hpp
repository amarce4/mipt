#pragma once

#include "mipt/backend.hpp"

#ifdef MIPT_ENABLE_CUDA_RHO
#include "mipt/cuda/ancilla_rdm.hpp"
#endif

#include <cudaq.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::ancilla
{
struct Rdm
{
    double rho00 = 0.0;
    double rho11 = 0.0;
    std::complex<double> rho01 = 0.0;
};

inline std::uint64_t checked_dimension(int n_qubits)
{
    if (n_qubits < 1 || n_qubits >= 63)
    {
        throw std::invalid_argument(
            "Ancilla RDM requires 1 <= total_qubits < 63.");
    }
    return std::uint64_t{1} << n_qubits;
}

inline std::uint64_t embed_environment_bits(
    std::uint64_t environment,
    int retained_qubit)
{
    const std::uint64_t lower_mask = retained_qubit == 0
        ? std::uint64_t{0}
        : ((std::uint64_t{1} << retained_qubit) - 1u);
    const std::uint64_t lower = environment & lower_mask;
    const std::uint64_t upper = environment >> retained_qubit;
    return lower | (upper << (retained_qubit + 1));
}

inline bool odd_popcount(std::uint64_t value)
{
#if defined(__GNUG__) || defined(__clang__)
    return (__builtin_popcountll(
        static_cast<unsigned long long>(value)) & 1) != 0;
#else
    bool parity = false;
    while (value)
    {
        parity = !parity;
        value &= value - 1;
    }
    return parity;
#endif
}

template <typename Real>
inline Rdm from_host_statevector(
    const std::complex<Real> *state,
    int n_qubits,
    int retained_qubit,
    bool fermion_trace)
{
    if (state == nullptr)
    {
        throw std::invalid_argument("Ancilla RDM received a null statevector.");
    }
    if (retained_qubit < 0 || retained_qubit >= n_qubits)
    {
        throw std::invalid_argument("Ancilla qubit index is out of range.");
    }

    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::uint64_t environment_dimension = dimension >> 1;
    const std::uint64_t retained_mask =
        std::uint64_t{1} << retained_qubit;
    const std::uint64_t lower_environment_mask = retained_qubit == 0
        ? std::uint64_t{0}
        : ((std::uint64_t{1} << retained_qubit) - 1u);

    Rdm rho;
    for (std::uint64_t environment = 0;
         environment < environment_dimension;
         ++environment)
    {
        const std::uint64_t index0 =
            embed_environment_bits(environment, retained_qubit);
        const std::uint64_t index1 = index0 | retained_mask;
        const std::complex<double> a = state[index0];
        const std::complex<double> b = state[index1];

        rho.rho00 += std::norm(a);
        rho.rho11 += std::norm(b);
        const double sign = fermion_trace &&
            odd_popcount(environment & lower_environment_mask)
                ? -1.0
                : 1.0;
        rho.rho01 += sign * a * std::conj(b);
    }
    return rho;
}

inline std::vector<int> basis_bits_from_index(
    int n_qubits,
    std::uint64_t basis_index)
{
    std::vector<int> bits(static_cast<std::size_t>(n_qubits), 0);
    const bool reverse = backend::amplitude_reverse_bits() != 0;
    for (int q = 0; q < n_qubits; ++q)
    {
        const int bit = static_cast<int>(
            (basis_index >> q) & std::uint64_t{1});
        bits[static_cast<std::size_t>(reverse ? n_qubits - 1 - q : q)] = bit;
    }
    return bits;
}

inline Rdm from_amplitude_queries(
    cudaq::state &state,
    int n_qubits,
    int retained_qubit,
    bool fermion_trace)
{
    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::uint64_t environment_dimension = dimension >> 1;
    const std::uint64_t retained_mask =
        std::uint64_t{1} << retained_qubit;
    const std::uint64_t lower_environment_mask = retained_qubit == 0
        ? std::uint64_t{0}
        : ((std::uint64_t{1} << retained_qubit) - 1u);

    const std::size_t amplitude_batch = static_cast<std::size_t>(
        backend::mps_amplitude_batch_size());
    const std::size_t environment_batch =
        std::max<std::size_t>(1, amplitude_batch / 2);

    std::vector<std::vector<int>> basis_states;
    std::vector<double> signs;
    Rdm rho;

    for (std::uint64_t first_environment = 0;
         first_environment < environment_dimension;
         first_environment += environment_batch)
    {
        const std::uint64_t count = std::min<std::uint64_t>(
            environment_batch,
            environment_dimension - first_environment);
        basis_states.clear();
        signs.clear();
        basis_states.reserve(static_cast<std::size_t>(2 * count));
        signs.reserve(static_cast<std::size_t>(count));

        for (std::uint64_t offset = 0; offset < count; ++offset)
        {
            const std::uint64_t environment = first_environment + offset;
            const std::uint64_t index0 =
                embed_environment_bits(environment, retained_qubit);
            const std::uint64_t index1 = index0 | retained_mask;
            basis_states.push_back(basis_bits_from_index(n_qubits, index0));
            basis_states.push_back(basis_bits_from_index(n_qubits, index1));
            signs.push_back(
                fermion_trace &&
                odd_popcount(environment & lower_environment_mask)
                    ? -1.0
                    : 1.0);
        }

        const auto amplitudes = state.amplitudes(basis_states);
        if (amplitudes.size() != basis_states.size())
        {
            throw std::runtime_error(
                "CUDA-Q state.amplitudes returned an unexpected count for the ancilla RDM.");
        }

        for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i)
        {
            const std::complex<double> a = amplitudes[2 * i];
            const std::complex<double> b = amplitudes[2 * i + 1];
            rho.rho00 += std::norm(a);
            rho.rho11 += std::norm(b);
            rho.rho01 += signs[i] * a * std::conj(b);
        }
    }
    return rho;
}

inline Rdm reduced_density_matrix(
    cudaq::state &state,
    int n_qubits,
    int retained_qubit,
    bool fermion_trace = false)
{
    if (retained_qubit < 0 || retained_qubit >= n_qubits)
    {
        throw std::invalid_argument("Ancilla qubit index is out of range.");
    }

    const std::uint64_t dimension_u64 = checked_dimension(n_qubits);
    if (dimension_u64 >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument(
            "Ancilla statevector dimension does not fit in size_t.");
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
            "ancilla reduced density matrix",
            n_qubits,
            false,
            0);
        return from_amplitude_queries(
            state,
            n_qubits,
            retained_qubit,
            fermion_trace);
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    if (state.is_on_gpu())
    {
        double values[4]{};
        int status = 0;
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = fermion_trace
                ? mipt_cuda_rho1_complex_fermion_f64(
                      tensor.data,
                      n_qubits,
                      retained_qubit,
                      values)
                : mipt_cuda_rho1_complex_f64(
                      tensor.data,
                      n_qubits,
                      retained_qubit,
                      values);
        }
        else
        {
            status = fermion_trace
                ? mipt_cuda_rho1_complex_fermion_f32(
                      tensor.data,
                      n_qubits,
                      retained_qubit,
                      values)
                : mipt_cuda_rho1_complex_f32(
                      tensor.data,
                      n_qubits,
                      retained_qubit,
                      values);
        }
        if (status != 0)
        {
            throw std::runtime_error(
                "CUDA one-site ancilla reduced-density-matrix kernel failed; status=" +
                std::to_string(status));
        }
        return {values[0], values[1], {values[2], values[3]}};
    }
#endif

    if (state.is_on_gpu())
    {
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            std::vector<std::complex<double>> host_state(dimension);
            state.to_host(host_state.data(), host_state.size());
            return from_host_statevector(
                host_state.data(),
                n_qubits,
                retained_qubit,
                fermion_trace);
        }

        std::vector<std::complex<float>> host_state(dimension);
        state.to_host(host_state.data(), host_state.size());
        return from_host_statevector(
            host_state.data(),
            n_qubits,
            retained_qubit,
            fermion_trace);
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        return from_host_statevector(
            reinterpret_cast<const std::complex<double> *>(tensor.data),
            n_qubits,
            retained_qubit,
            fermion_trace);
    }

    return from_host_statevector(
        reinterpret_cast<const std::complex<float> *>(tensor.data),
        n_qubits,
        retained_qubit,
        fermion_trace);
}
} // namespace mipt::ancilla
