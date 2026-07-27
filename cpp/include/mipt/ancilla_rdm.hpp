#pragma once

#include "mipt/backend.hpp"

#ifdef MIPT_ENABLE_CUDA_RHO
#include "mipt/cuda/ancilla_rdm.hpp"
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
struct Rdm
{
    double rho00 = 0.0;
    double rho11 = 0.0;
    std::complex<double> rho01 = 0.0;
};

struct ParityWeights
{
    double even = 0.0;
    double odd = 0.0;
};

struct Rdm2
{
    static constexpr std::size_t dimension = 4;
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

struct Rdm3
{
    static constexpr std::size_t dimension = 8;
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

inline std::uint64_t checked_dimension(int n_qubits)
{
    if (n_qubits < 1 || n_qubits >= 63)
    {
        throw std::invalid_argument("Ancilla RDM requires 1 <= total_qubits < 63.");
    }
    return std::uint64_t{1} << n_qubits;
}

inline std::uint64_t embed_environment_bits(std::uint64_t environment, int retained_qubit)
{
    const std::uint64_t lower_mask =
        retained_qubit == 0 ? std::uint64_t{0} : ((std::uint64_t{1} << retained_qubit) - 1u);
    const std::uint64_t lower = environment & lower_mask;
    const std::uint64_t upper = environment >> retained_qubit;
    return lower | (upper << (retained_qubit + 1));
}

inline std::uint64_t embed_environment_bits_two(std::uint64_t environment, int retained_qubit0, int retained_qubit1)
{
    const int lower = std::min(retained_qubit0, retained_qubit1);
    const int upper = std::max(retained_qubit0, retained_qubit1);
    return embed_environment_bits(embed_environment_bits(environment, lower), upper);
}

inline std::uint64_t embed_environment_bits_three(std::uint64_t environment, const std::array<int, 3> &retained)
{
    auto sorted = retained;
    std::sort(sorted.begin(), sorted.end());
    std::uint64_t value = environment;
    for (int retained_qubit : sorted)
    {
        value = embed_environment_bits(value, retained_qubit);
    }
    return value;
}

inline std::uint64_t embed_environment_bits_four(std::uint64_t environment, const std::array<int, 4> &retained)
{
    auto sorted = retained;
    std::sort(sorted.begin(), sorted.end());
    std::uint64_t value = environment;
    for (int retained_qubit : sorted)
    {
        value = embed_environment_bits(value, retained_qubit);
    }
    return value;
}

inline bool odd_popcount(std::uint64_t value)
{
#if defined(__GNUG__) || defined(__clang__)
    return (__builtin_popcountll(static_cast<unsigned long long>(value)) & 1) != 0;
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

inline int two_mode_reordered_position(int mode, int retained_qubit0, int retained_qubit1)
{
    if (mode == retained_qubit0)
    {
        return 0;
    }
    if (mode == retained_qubit1)
    {
        return 1;
    }
    return 2 + mode - (retained_qubit0 < mode ? 1 : 0) - (retained_qubit1 < mode ? 1 : 0);
}

inline double two_mode_fermion_reorder_sign(std::uint64_t occupation_mask, int n_qubits, int retained_qubit0,
    int retained_qubit1)
{
    std::uint64_t seen_new_positions = 0;
    int parity = 0;
    for (int mode = 0; mode < n_qubits; ++mode)
    {
        if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
        {
            continue;
        }
        const int position = two_mode_reordered_position(mode, retained_qubit0, retained_qubit1);
        const std::uint64_t lower_or_equal = (std::uint64_t{1} << (position + 1)) - std::uint64_t{1};
        parity ^= odd_popcount(seen_new_positions & ~lower_or_equal) ? 1 : 0;
        seen_new_positions |= std::uint64_t{1} << position;
    }
    return parity ? -1.0 : 1.0;
}

inline int three_mode_reordered_position(int mode, const std::array<int, 3> &retained)
{
    for (int local = 0; local < 3; ++local)
    {
        if (mode == retained[static_cast<std::size_t>(local)])
        {
            return local;
        }
    }
    int before = 0;
    for (int retained_qubit : retained)
    {
        before += retained_qubit < mode ? 1 : 0;
    }
    return 3 + mode - before;
}

inline double three_mode_fermion_reorder_sign(std::uint64_t occupation_mask, int n_qubits,
                                              const std::array<int, 3> &retained)
{
    std::uint64_t seen_new_positions = 0;
    int parity = 0;
    for (int mode = 0; mode < n_qubits; ++mode)
    {
        if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
        {
            continue;
        }
        const int position = three_mode_reordered_position(mode, retained);
        const std::uint64_t lower_or_equal = (std::uint64_t{1} << (position + 1)) - std::uint64_t{1};
        parity ^= odd_popcount(seen_new_positions & ~lower_or_equal) ? 1 : 0;
        seen_new_positions |= std::uint64_t{1} << position;
    }
    return parity ? -1.0 : 1.0;
}

inline int four_mode_reordered_position(int mode, const std::array<int, 4> &retained)
{
    for (int local = 0; local < 4; ++local)
    {
        if (mode == retained[static_cast<std::size_t>(local)])
        {
            return local;
        }
    }
    int before = 0;
    for (int retained_qubit : retained)
    {
        before += retained_qubit < mode ? 1 : 0;
    }
    return 4 + mode - before;
}

inline double four_mode_fermion_reorder_sign(std::uint64_t occupation_mask, int n_qubits,
    const std::array<int, 4> &retained)
{
    std::uint64_t seen_new_positions = 0;
    int parity = 0;
    for (int mode = 0; mode < n_qubits; ++mode)
    {
        if (((occupation_mask >> mode) & std::uint64_t{1}) == 0)
        {
            continue;
        }
        const int position = four_mode_reordered_position(mode, retained);
        const std::uint64_t lower_or_equal = (std::uint64_t{1} << (position + 1)) - std::uint64_t{1};
        parity ^= odd_popcount(seen_new_positions & ~lower_or_equal) ? 1 : 0;
        seen_new_positions |= std::uint64_t{1} << position;
    }
    return parity ? -1.0 : 1.0;
}

template <typename Real>
inline Rdm from_host_statevector(const std::complex<Real> *state, int n_qubits, int retained_qubit, bool fermion_trace)
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
    const std::uint64_t retained_mask = std::uint64_t{1} << retained_qubit;
    const std::uint64_t lower_environment_mask =
        retained_qubit == 0 ? std::uint64_t{0} : ((std::uint64_t{1} << retained_qubit) - 1u);

    Rdm rho;
    for (std::uint64_t environment = 0; environment < environment_dimension; ++environment)
    {
        const std::uint64_t index0 = embed_environment_bits(environment, retained_qubit);
        const std::uint64_t index1 = index0 | retained_mask;
        const std::complex<double> a = state[index0];
        const std::complex<double> b = state[index1];

        rho.rho00 += std::norm(a);
        rho.rho11 += std::norm(b);
        const double sign = fermion_trace && odd_popcount(environment & lower_environment_mask) ? -1.0 : 1.0;
        rho.rho01 += sign * a * std::conj(b);
    }
    return rho;
}

template <typename Real>
inline Rdm2 from_host_statevector_two(const std::complex<Real> *state, int n_qubits, int retained_qubit0,
                                      int retained_qubit1, bool fermion_trace)
{
    if (state == nullptr)
    {
        throw std::invalid_argument("Two-ancilla RDM received a null statevector.");
    }
    if (n_qubits < 2 || retained_qubit0 < 0 || retained_qubit0 >= n_qubits || retained_qubit1 < 0 ||
        retained_qubit1 >= n_qubits || retained_qubit0 == retained_qubit1)
    {
        throw std::invalid_argument("Two-ancilla qubit indices are invalid.");
    }

    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::uint64_t environment_dimension = dimension >> 2;
    const std::uint64_t mask0 = std::uint64_t{1} << retained_qubit0;
    const std::uint64_t mask1 = std::uint64_t{1} << retained_qubit1;

    Rdm2 rho;
    std::array<std::complex<double>, 4> amplitudes{};
    for (std::uint64_t environment = 0; environment < environment_dimension; ++environment)
    {
        const std::uint64_t base = embed_environment_bits_two(environment, retained_qubit0, retained_qubit1);
        for (std::size_t local = 0; local < 4; ++local)
        {
            const std::uint64_t index =
                base | ((local & 1u) ? mask0 : std::uint64_t{0}) | ((local & 2u) ? mask1 : std::uint64_t{0});
            const double sign =
                fermion_trace ? two_mode_fermion_reorder_sign(index, n_qubits, retained_qubit0, retained_qubit1) : 1.0;
            amplitudes[local] = static_cast<std::complex<double>>(state[index]) * sign;
        }

        for (std::size_t row = 0; row < 4; ++row)
        {
            for (std::size_t col = 0; col < 4; ++col)
            {
                rho(row, col) += amplitudes[row] * std::conj(amplitudes[col]);
            }
        }
    }
    return rho;
}

template <typename Real>
inline Rdm3 from_host_statevector_three(const std::complex<Real> *state, int n_qubits,
                                        const std::array<int, 3> &retained, bool fermion_trace)
{
    if (state == nullptr)
    {
        throw std::invalid_argument("Three-ancilla RDM received a null statevector.");
    }
    auto sorted = retained;
    std::sort(sorted.begin(), sorted.end());
    if (n_qubits < 3 || sorted.front() < 0 || sorted.back() >= n_qubits ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    {
        throw std::invalid_argument("Three-ancilla qubit indices are invalid.");
    }

    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::uint64_t environment_dimension = dimension >> 3;
    std::array<std::uint64_t, 3> masks{};
    for (int local = 0; local < 3; ++local)
    {
        masks[static_cast<std::size_t>(local)] = std::uint64_t{1} << retained[static_cast<std::size_t>(local)];
    }

    Rdm3 rho;
    std::array<std::complex<double>, 8> amplitudes{};
    for (std::uint64_t environment = 0; environment < environment_dimension; ++environment)
    {
        const std::uint64_t base = embed_environment_bits_three(environment, retained);
        for (std::size_t local = 0; local < 8; ++local)
        {
            std::uint64_t index = base;
            for (int bit = 0; bit < 3; ++bit)
            {
                if ((local >> bit) & 1u)
                {
                    index |= masks[static_cast<std::size_t>(bit)];
                }
            }
            const double sign = fermion_trace ? three_mode_fermion_reorder_sign(index, n_qubits, retained) : 1.0;
            amplitudes[local] = static_cast<std::complex<double>>(state[index]) * sign;
        }
        for (std::size_t row = 0; row < 8; ++row)
        {
            for (std::size_t col = 0; col < 8; ++col)
            {
                rho(row, col) += amplitudes[row] * std::conj(amplitudes[col]);
            }
        }
    }
    return rho;
}

template <typename Real>
inline Rdm4 from_host_statevector_four(const std::complex<Real> *state, int n_qubits,
                                       const std::array<int, 4> &retained, bool fermion_trace)
{
    if (state == nullptr)
    {
        throw std::invalid_argument("Four-ancilla RDM received a null statevector.");
    }
    auto sorted = retained;
    std::sort(sorted.begin(), sorted.end());
    if (n_qubits < 4 || sorted.front() < 0 || sorted.back() >= n_qubits ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    {
        throw std::invalid_argument("Four-ancilla qubit indices are invalid.");
    }

    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::uint64_t environment_dimension = dimension >> 4;
    std::array<std::uint64_t, 4> masks{};
    for (int local = 0; local < 4; ++local)
    {
        masks[static_cast<std::size_t>(local)] = std::uint64_t{1} << retained[static_cast<std::size_t>(local)];
    }

    Rdm4 rho;
    std::array<std::complex<double>, 16> amplitudes{};
    for (std::uint64_t environment = 0; environment < environment_dimension; ++environment)
    {
        const std::uint64_t base = embed_environment_bits_four(environment, retained);
        for (std::size_t local = 0; local < 16; ++local)
        {
            std::uint64_t index = base;
            for (int bit = 0; bit < 4; ++bit)
            {
                if ((local >> bit) & 1u)
                {
                    index |= masks[static_cast<std::size_t>(bit)];
                }
            }
            const double sign = fermion_trace ? four_mode_fermion_reorder_sign(index, n_qubits, retained) : 1.0;
            amplitudes[local] = static_cast<std::complex<double>>(state[index]) * sign;
        }
        for (std::size_t row = 0; row < 16; ++row)
        {
            for (std::size_t col = 0; col < 16; ++col)
            {
                rho(row, col) += amplitudes[row] * std::conj(amplitudes[col]);
            }
        }
    }
    return rho;
}

inline std::vector<int> basis_bits_from_index(int n_qubits, std::uint64_t basis_index)
{
    std::vector<int> bits(static_cast<std::size_t>(n_qubits), 0);
    const bool reverse = backend::amplitude_reverse_bits() != 0;
    for (int q = 0; q < n_qubits; ++q)
    {
        const int bit = static_cast<int>((basis_index >> q) & std::uint64_t{1});
        bits[static_cast<std::size_t>(reverse ? n_qubits - 1 - q : q)] = bit;
    }
    return bits;
}

template <typename Complex>
inline ParityWeights parity_weights_from_host_statevector(const Complex *state, int n_qubits)
{
    if (state == nullptr)
    {
        throw std::invalid_argument("Parity reduction received a null statevector.");
    }
    const std::uint64_t dimension = checked_dimension(n_qubits);
    ParityWeights weights;
    for (std::uint64_t index = 0; index < dimension; ++index)
    {
        const double probability = std::norm(static_cast<std::complex<double>>(state[index]));
        (odd_popcount(index) ? weights.odd : weights.even) += probability;
    }
    return weights;
}

inline ParityWeights parity_weights_from_amplitude_queries(cudaq::state &state, int n_qubits)
{
    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::size_t batch_size = static_cast<std::size_t>(backend::mps_amplitude_batch_size());
    std::vector<std::vector<int>> basis_states;
    basis_states.reserve(batch_size);
    ParityWeights weights;
    for (std::uint64_t first = 0; first < dimension; first += batch_size)
    {
        const std::uint64_t count = std::min<std::uint64_t>(batch_size, dimension - first);
        basis_states.clear();
        for (std::uint64_t offset = 0; offset < count; ++offset)
        {
            basis_states.push_back(basis_bits_from_index(n_qubits, first + offset));
        }
        const auto amplitudes = state.amplitudes(basis_states);
        if (amplitudes.size() != basis_states.size())
        {
            throw std::runtime_error("CUDA-Q state.amplitudes returned an unexpected "
                                     "count for parity reduction.");
        }
        for (std::uint64_t offset = 0; offset < count; ++offset)
        {
            const double probability =
                std::norm(static_cast<std::complex<double>>(amplitudes[static_cast<std::size_t>(offset)]));
            (odd_popcount(first + offset) ? weights.odd : weights.even) += probability;
        }
    }
    return weights;
}

inline Rdm from_amplitude_queries(cudaq::state &state, int n_qubits, int retained_qubit, bool fermion_trace)
{
    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::uint64_t environment_dimension = dimension >> 1;
    const std::uint64_t retained_mask = std::uint64_t{1} << retained_qubit;
    const std::uint64_t lower_environment_mask =
        retained_qubit == 0 ? std::uint64_t{0} : ((std::uint64_t{1} << retained_qubit) - 1u);

    const std::size_t amplitude_batch = static_cast<std::size_t>(backend::mps_amplitude_batch_size());
    const std::size_t environment_batch = std::max<std::size_t>(1, amplitude_batch / 2);

    std::vector<std::vector<int>> basis_states;
    std::vector<double> signs;
    Rdm rho;

    for (std::uint64_t first_environment = 0; first_environment < environment_dimension;
         first_environment += environment_batch)
    {
        const std::uint64_t count =
            std::min<std::uint64_t>(environment_batch, environment_dimension - first_environment);
        basis_states.clear();
        signs.clear();
        basis_states.reserve(static_cast<std::size_t>(2 * count));
        signs.reserve(static_cast<std::size_t>(count));

        for (std::uint64_t offset = 0; offset < count; ++offset)
        {
            const std::uint64_t environment = first_environment + offset;
            const std::uint64_t index0 = embed_environment_bits(environment, retained_qubit);
            const std::uint64_t index1 = index0 | retained_mask;
            basis_states.push_back(basis_bits_from_index(n_qubits, index0));
            basis_states.push_back(basis_bits_from_index(n_qubits, index1));
            signs.push_back(fermion_trace && odd_popcount(environment & lower_environment_mask) ? -1.0 : 1.0);
        }

        const auto amplitudes = state.amplitudes(basis_states);
        if (amplitudes.size() != basis_states.size())
        {
            throw std::runtime_error("CUDA-Q state.amplitudes returned an unexpected "
                                     "count for the ancilla RDM.");
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

inline Rdm2 from_amplitude_queries_two(cudaq::state &state, int n_qubits, int retained_qubit0, int retained_qubit1,
    bool fermion_trace)
{
    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::uint64_t environment_dimension = dimension >> 2;
    const std::uint64_t mask0 = std::uint64_t{1} << retained_qubit0;
    const std::uint64_t mask1 = std::uint64_t{1} << retained_qubit1;

    const std::size_t amplitude_batch = static_cast<std::size_t>(backend::mps_amplitude_batch_size());
    const std::size_t environment_batch = std::max<std::size_t>(1, amplitude_batch / 4);

    std::vector<std::vector<int>> basis_states;
    std::vector<double> signs;
    Rdm2 rho;

    for (std::uint64_t first_environment = 0; first_environment < environment_dimension;
         first_environment += environment_batch)
    {
        const std::uint64_t count =
            std::min<std::uint64_t>(environment_batch, environment_dimension - first_environment);
        basis_states.clear();
        signs.clear();
        basis_states.reserve(static_cast<std::size_t>(4 * count));
        signs.reserve(static_cast<std::size_t>(4 * count));

        for (std::uint64_t offset = 0; offset < count; ++offset)
        {
            const std::uint64_t environment = first_environment + offset;
            const std::uint64_t base = embed_environment_bits_two(environment, retained_qubit0, retained_qubit1);
            for (std::size_t local = 0; local < 4; ++local)
            {
                const std::uint64_t index =
                    base | ((local & 1u) ? mask0 : std::uint64_t{0}) | ((local & 2u) ? mask1 : std::uint64_t{0});
                basis_states.push_back(basis_bits_from_index(n_qubits, index));
                signs.push_back(fermion_trace
                                    ? two_mode_fermion_reorder_sign(index, n_qubits, retained_qubit0, retained_qubit1)
                    : 1.0);
            }
        }

        const auto queried = state.amplitudes(basis_states);
        if (queried.size() != basis_states.size())
        {
            throw std::runtime_error("CUDA-Q state.amplitudes returned an unexpected "
                                     "count for the two-ancilla RDM.");
        }

        for (std::size_t environment = 0; environment < static_cast<std::size_t>(count); ++environment)
        {
            std::array<std::complex<double>, 4> amplitudes{};
            for (std::size_t local = 0; local < 4; ++local)
            {
                const std::size_t index = 4 * environment + local;
                amplitudes[local] = queried[index] * signs[index];
            }
            for (std::size_t row = 0; row < 4; ++row)
            {
                for (std::size_t col = 0; col < 4; ++col)
                {
                    rho(row, col) += amplitudes[row] * std::conj(amplitudes[col]);
                }
            }
        }
    }
    return rho;
}

inline Rdm3 from_amplitude_queries_three(cudaq::state &state, int n_qubits, const std::array<int, 3> &retained,
                                         bool fermion_trace)
{
    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::uint64_t environment_dimension = dimension >> 3;
    std::array<std::uint64_t, 3> masks{};
    for (int local = 0; local < 3; ++local)
    {
        masks[static_cast<std::size_t>(local)] = std::uint64_t{1} << retained[static_cast<std::size_t>(local)];
    }

    const std::size_t amplitude_batch = static_cast<std::size_t>(backend::mps_amplitude_batch_size());
    const std::size_t environment_batch = std::max<std::size_t>(1, amplitude_batch / 8);
    std::vector<std::vector<int>> basis_states;
    std::vector<double> signs;
    Rdm3 rho;

    for (std::uint64_t first_environment = 0; first_environment < environment_dimension;
         first_environment += environment_batch)
    {
        const std::uint64_t count =
            std::min<std::uint64_t>(environment_batch, environment_dimension - first_environment);
        basis_states.clear();
        signs.clear();
        basis_states.reserve(static_cast<std::size_t>(8 * count));
        signs.reserve(static_cast<std::size_t>(8 * count));

        for (std::uint64_t offset = 0; offset < count; ++offset)
        {
            const std::uint64_t base = embed_environment_bits_three(first_environment + offset, retained);
            for (std::size_t local = 0; local < 8; ++local)
            {
                std::uint64_t index = base;
                for (int bit = 0; bit < 3; ++bit)
                {
                    if ((local >> bit) & 1u)
                    {
                        index |= masks[static_cast<std::size_t>(bit)];
                    }
                }
                basis_states.push_back(basis_bits_from_index(n_qubits, index));
                signs.push_back(fermion_trace ? three_mode_fermion_reorder_sign(index, n_qubits, retained) : 1.0);
            }
        }

        const auto queried = state.amplitudes(basis_states);
        if (queried.size() != basis_states.size())
        {
            throw std::runtime_error("CUDA-Q state.amplitudes returned an unexpected "
                                     "count for the three-ancilla RDM.");
        }
        for (std::size_t environment = 0; environment < static_cast<std::size_t>(count); ++environment)
        {
            std::array<std::complex<double>, 8> amplitudes{};
            for (std::size_t local = 0; local < 8; ++local)
            {
                const std::size_t index = 8 * environment + local;
                amplitudes[local] = queried[index] * signs[index];
            }
            for (std::size_t row = 0; row < 8; ++row)
            {
                for (std::size_t col = 0; col < 8; ++col)
                {
                    rho(row, col) += amplitudes[row] * std::conj(amplitudes[col]);
                }
            }
        }
    }
    return rho;
}

inline Rdm4 from_amplitude_queries_four(cudaq::state &state, int n_qubits, const std::array<int, 4> &retained,
    bool fermion_trace)
{
    const std::uint64_t dimension = checked_dimension(n_qubits);
    const std::uint64_t environment_dimension = dimension >> 4;
    std::array<std::uint64_t, 4> masks{};
    for (int local = 0; local < 4; ++local)
    {
        masks[static_cast<std::size_t>(local)] = std::uint64_t{1} << retained[static_cast<std::size_t>(local)];
    }

    const std::size_t amplitude_batch = static_cast<std::size_t>(backend::mps_amplitude_batch_size());
    const std::size_t environment_batch = std::max<std::size_t>(1, amplitude_batch / 16);
    std::vector<std::vector<int>> basis_states;
    std::vector<double> signs;
    Rdm4 rho;

    for (std::uint64_t first_environment = 0; first_environment < environment_dimension;
         first_environment += environment_batch)
    {
        const std::uint64_t count =
            std::min<std::uint64_t>(environment_batch, environment_dimension - first_environment);
        basis_states.clear();
        signs.clear();
        basis_states.reserve(static_cast<std::size_t>(16 * count));
        signs.reserve(static_cast<std::size_t>(16 * count));

        for (std::uint64_t offset = 0; offset < count; ++offset)
        {
            const std::uint64_t base = embed_environment_bits_four(first_environment + offset, retained);
            for (std::size_t local = 0; local < 16; ++local)
            {
                std::uint64_t index = base;
                for (int bit = 0; bit < 4; ++bit)
                {
                    if ((local >> bit) & 1u)
                    {
                        index |= masks[static_cast<std::size_t>(bit)];
                    }
                }
                basis_states.push_back(basis_bits_from_index(n_qubits, index));
                signs.push_back(fermion_trace ? four_mode_fermion_reorder_sign(index, n_qubits, retained) : 1.0);
            }
        }

        const auto queried = state.amplitudes(basis_states);
        if (queried.size() != basis_states.size())
        {
            throw std::runtime_error("CUDA-Q state.amplitudes returned an unexpected "
                                     "count for the four-ancilla RDM.");
        }
        for (std::size_t environment = 0; environment < static_cast<std::size_t>(count); ++environment)
        {
            std::array<std::complex<double>, 16> amplitudes{};
            for (std::size_t local = 0; local < 16; ++local)
            {
                const std::size_t index = 16 * environment + local;
                amplitudes[local] = queried[index] * signs[index];
            }
            for (std::size_t row = 0; row < 16; ++row)
            {
                for (std::size_t col = 0; col < 16; ++col)
                {
                    rho(row, col) += amplitudes[row] * std::conj(amplitudes[col]);
                }
            }
        }
    }
    return rho;
}

inline Rdm reduced_density_matrix(cudaq::state &state, int n_qubits, int retained_qubit, bool fermion_trace = false)
{
    if (retained_qubit < 0 || retained_qubit >= n_qubits)
    {
        throw std::invalid_argument("Ancilla qubit index is out of range.");
    }

    const std::uint64_t dimension_u64 = checked_dimension(n_qubits);
    if (dimension_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("Ancilla statevector dimension does not fit in size_t.");
    }
    const std::size_t dimension = static_cast<std::size_t>(dimension_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();
    const bool dense_rank1 = tensor.get_rank() == 1 && tensor.get_num_elements() >= dimension && tensor.data != nullptr;

    if (!dense_rank1)
    {
        backend::warn_mps_amplitude_path_once("ancilla reduced density matrix", n_qubits, false, 0);
        return from_amplitude_queries(state, n_qubits, retained_qubit, fermion_trace);
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    if (state.is_on_gpu())
    {
        double values[4]{};
        int status = 0;
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = fermion_trace ? mipt_cuda_rho1_complex_fermion_f64(tensor.data, n_qubits, retained_qubit, values)
                                   : mipt_cuda_rho1_complex_f64(tensor.data, n_qubits, retained_qubit, values);
        }
        else
        {
            status = fermion_trace ? mipt_cuda_rho1_complex_fermion_f32(tensor.data, n_qubits, retained_qubit, values)
                                   : mipt_cuda_rho1_complex_f32(tensor.data, n_qubits, retained_qubit, values);
        }
        if (status != 0)
        {
            throw std::runtime_error("CUDA one-site ancilla reduced-density-matrix "
                                     "kernel failed; status=" +
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
            return from_host_statevector(host_state.data(), n_qubits, retained_qubit, fermion_trace);
        }

        std::vector<std::complex<float>> host_state(dimension);
        state.to_host(host_state.data(), host_state.size());
        return from_host_statevector(host_state.data(), n_qubits, retained_qubit, fermion_trace);
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        return from_host_statevector(reinterpret_cast<const std::complex<double> *>(tensor.data), n_qubits,
                                     retained_qubit, fermion_trace);
    }

    return from_host_statevector(reinterpret_cast<const std::complex<float> *>(tensor.data), n_qubits, retained_qubit,
        fermion_trace);
}

inline ParityWeights parity_sector_weights(cudaq::state &state, int n_qubits)
{
    const std::uint64_t dimension_u64 = checked_dimension(n_qubits);
    if (dimension_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("Parity statevector dimension does not fit in size_t.");
    }
    const std::size_t dimension = static_cast<std::size_t>(dimension_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();
    const bool dense_rank1 = tensor.get_rank() == 1 && tensor.get_num_elements() >= dimension && tensor.data != nullptr;

    if (!dense_rank1)
    {
        backend::warn_mps_amplitude_path_once("parity-sector reduction", n_qubits, false, 0);
        return parity_weights_from_amplitude_queries(state, n_qubits);
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    if (state.is_on_gpu())
    {
        double values[2]{};
        const int status = precision == cudaq::SimulationState::precision::fp64
                               ? mipt_cuda_parity_weights_f64(tensor.data, n_qubits, values)
                               : mipt_cuda_parity_weights_f32(tensor.data, n_qubits, values);
        if (status != 0)
        {
            throw std::runtime_error("CUDA parity-sector reduction failed; status=" + std::to_string(status));
        }
        return {values[0], values[1]};
    }
#endif

    if (state.is_on_gpu())
    {
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            std::vector<std::complex<double>> host_state(dimension);
            state.to_host(host_state.data(), host_state.size());
            return parity_weights_from_host_statevector(host_state.data(), n_qubits);
        }
        std::vector<std::complex<float>> host_state(dimension);
        state.to_host(host_state.data(), host_state.size());
        return parity_weights_from_host_statevector(host_state.data(), n_qubits);
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        return parity_weights_from_host_statevector(reinterpret_cast<const std::complex<double> *>(tensor.data),
            n_qubits);
    }
    return parity_weights_from_host_statevector(reinterpret_cast<const std::complex<float> *>(tensor.data), n_qubits);
}

inline Rdm2 reduced_density_matrix_two(cudaq::state &state, int n_qubits, int retained_qubit0, int retained_qubit1,
    bool fermion_trace = false)
{
    if (n_qubits < 2 || retained_qubit0 < 0 || retained_qubit0 >= n_qubits || retained_qubit1 < 0 ||
        retained_qubit1 >= n_qubits || retained_qubit0 == retained_qubit1)
    {
        throw std::invalid_argument("Two-ancilla qubit indices are invalid.");
    }

    const std::uint64_t dimension_u64 = checked_dimension(n_qubits);
    if (dimension_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("Two-ancilla statevector dimension does not fit in size_t.");
    }
    const std::size_t dimension = static_cast<std::size_t>(dimension_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();
    const bool dense_rank1 = tensor.get_rank() == 1 && tensor.get_num_elements() >= dimension && tensor.data != nullptr;

    if (!dense_rank1)
    {
        backend::warn_mps_amplitude_path_once("two-ancilla reduced density matrix", n_qubits, false, 0);
        return from_amplitude_queries_two(state, n_qubits, retained_qubit0, retained_qubit1, fermion_trace);
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    if (state.is_on_gpu())
    {
        double values[32]{};
        int status = 0;
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = fermion_trace
                         ? mipt_cuda_rho2_complex_fermion_f64(tensor.data, n_qubits, retained_qubit0, retained_qubit1,
                      values)
                         : mipt_cuda_rho2_complex_f64(tensor.data, n_qubits, retained_qubit0, retained_qubit1, values);
        }
        else
        {
            status = fermion_trace
                         ? mipt_cuda_rho2_complex_fermion_f32(tensor.data, n_qubits, retained_qubit0, retained_qubit1,
                      values)
                         : mipt_cuda_rho2_complex_f32(tensor.data, n_qubits, retained_qubit0, retained_qubit1, values);
        }
        if (status != 0)
        {
            throw std::runtime_error("CUDA two-site ancilla reduced-density-matrix "
                                     "kernel failed; status=" +
                std::to_string(status));
        }

        Rdm2 rho;
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
            return from_host_statevector_two(host_state.data(), n_qubits, retained_qubit0, retained_qubit1,
                fermion_trace);
        }

        std::vector<std::complex<float>> host_state(dimension);
        state.to_host(host_state.data(), host_state.size());
        return from_host_statevector_two(host_state.data(), n_qubits, retained_qubit0, retained_qubit1, fermion_trace);
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        return from_host_statevector_two(reinterpret_cast<const std::complex<double> *>(tensor.data), n_qubits,
                                         retained_qubit0, retained_qubit1, fermion_trace);
    }

    return from_host_statevector_two(reinterpret_cast<const std::complex<float> *>(tensor.data), n_qubits,
                                     retained_qubit0, retained_qubit1, fermion_trace);
}

inline Rdm3 reduced_density_matrix_three(cudaq::state &state, int n_qubits, const std::array<int, 3> &retained,
                                         bool fermion_trace = false)
{
    auto sorted = retained;
    std::sort(sorted.begin(), sorted.end());
    if (n_qubits < 3 || sorted.front() < 0 || sorted.back() >= n_qubits ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    {
        throw std::invalid_argument("Three-ancilla qubit indices are invalid.");
    }

    const std::uint64_t dimension_u64 = checked_dimension(n_qubits);
    if (dimension_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("Three-ancilla statevector dimension does not fit in size_t.");
    }
    const std::size_t dimension = static_cast<std::size_t>(dimension_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();
    const bool dense_rank1 = tensor.get_rank() == 1 && tensor.get_num_elements() >= dimension && tensor.data != nullptr;

    if (!dense_rank1)
    {
        backend::warn_mps_amplitude_path_once("three-ancilla reduced density matrix", n_qubits, false, 0);
        return from_amplitude_queries_three(state, n_qubits, retained, fermion_trace);
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    if (state.is_on_gpu())
    {
        double values[128]{};
        int status = 0;
        const int q0 = retained[0];
        const int q1 = retained[1];
        const int q2 = retained[2];
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = fermion_trace ? mipt_cuda_rho3_complex_fermion_f64(tensor.data, n_qubits, q0, q1, q2, values)
                                   : mipt_cuda_rho3_complex_f64(tensor.data, n_qubits, q0, q1, q2, values);
        }
        else
        {
            status = fermion_trace ? mipt_cuda_rho3_complex_fermion_f32(tensor.data, n_qubits, q0, q1, q2, values)
                                   : mipt_cuda_rho3_complex_f32(tensor.data, n_qubits, q0, q1, q2, values);
        }
        if (status != 0)
        {
            throw std::runtime_error("CUDA three-site ancilla reduced-density-matrix "
                                     "kernel failed; status=" +
                                     std::to_string(status));
        }
        Rdm3 rho;
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
            return from_host_statevector_three(host_state.data(), n_qubits, retained, fermion_trace);
        }
        std::vector<std::complex<float>> host_state(dimension);
        state.to_host(host_state.data(), host_state.size());
        return from_host_statevector_three(host_state.data(), n_qubits, retained, fermion_trace);
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        return from_host_statevector_three(reinterpret_cast<const std::complex<double> *>(tensor.data), n_qubits,
                                           retained, fermion_trace);
    }
    return from_host_statevector_three(reinterpret_cast<const std::complex<float> *>(tensor.data), n_qubits, retained,
        fermion_trace);
}

inline Rdm4 reduced_density_matrix_four(cudaq::state &state, int n_qubits, const std::array<int, 4> &retained,
    bool fermion_trace = false)
{
    auto sorted = retained;
    std::sort(sorted.begin(), sorted.end());
    if (n_qubits < 4 || sorted.front() < 0 || sorted.back() >= n_qubits ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    {
        throw std::invalid_argument("Four-ancilla qubit indices are invalid.");
    }

    const std::uint64_t dimension_u64 = checked_dimension(n_qubits);
    if (dimension_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::invalid_argument("Four-ancilla statevector dimension does not fit in size_t.");
    }
    const std::size_t dimension = static_cast<std::size_t>(dimension_u64);
    const auto precision = state.get_precision();
    const auto tensor = state.get_tensor();
    const bool dense_rank1 = tensor.get_rank() == 1 && tensor.get_num_elements() >= dimension && tensor.data != nullptr;

    if (!dense_rank1)
    {
        backend::warn_mps_amplitude_path_once("four-ancilla reduced density matrix", n_qubits, false, 0);
        return from_amplitude_queries_four(state, n_qubits, retained, fermion_trace);
    }

#ifdef MIPT_ENABLE_CUDA_RHO
    if (state.is_on_gpu())
    {
        double values[512]{};
        int status = 0;
        const int q0 = retained[0];
        const int q1 = retained[1];
        const int q2 = retained[2];
        const int q3 = retained[3];
        if (precision == cudaq::SimulationState::precision::fp64)
        {
            status = fermion_trace ? mipt_cuda_rho4_complex_fermion_f64(tensor.data, n_qubits, q0, q1, q2, q3, values)
                                   : mipt_cuda_rho4_complex_f64(tensor.data, n_qubits, q0, q1, q2, q3, values);
        }
        else
        {
            status = fermion_trace ? mipt_cuda_rho4_complex_fermion_f32(tensor.data, n_qubits, q0, q1, q2, q3, values)
                                   : mipt_cuda_rho4_complex_f32(tensor.data, n_qubits, q0, q1, q2, q3, values);
        }
        if (status != 0)
        {
            throw std::runtime_error("CUDA four-site ancilla reduced-density-matrix "
                                     "kernel failed; status=" +
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
            return from_host_statevector_four(host_state.data(), n_qubits, retained, fermion_trace);
        }
        std::vector<std::complex<float>> host_state(dimension);
        state.to_host(host_state.data(), host_state.size());
        return from_host_statevector_four(host_state.data(), n_qubits, retained, fermion_trace);
    }

    if (precision == cudaq::SimulationState::precision::fp64)
    {
        return from_host_statevector_four(reinterpret_cast<const std::complex<double> *>(tensor.data), n_qubits,
                                          retained, fermion_trace);
    }
    return from_host_statevector_four(reinterpret_cast<const std::complex<float> *>(tensor.data), n_qubits, retained,
        fermion_trace);
}
} // namespace mipt::ancilla
