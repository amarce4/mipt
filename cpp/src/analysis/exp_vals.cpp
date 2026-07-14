#include "mipt/analysis/exp_vals.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace
{
    using Complex = std::complex<double>;

    constexpr std::size_t DIM = 8;
    constexpr std::size_t MODE_COUNT = 3;
    constexpr std::size_t MAJORANA_COUNT = 2 * MODE_COUNT;

    using Matrix = std::array<Complex, DIM * DIM>;

    constexpr std::size_t index(std::size_t row, std::size_t col) noexcept
    {
        return row * DIM + col;
    }

    Matrix zero_matrix()
    {
        return Matrix{};
    }

    Matrix dagger(const Matrix &matrix)
    {
        Matrix result{};
        for (std::size_t row = 0; row < DIM; ++row)
        {
            for (std::size_t col = 0; col < DIM; ++col)
            {
                result[index(row, col)] = std::conj(matrix[index(col, row)]);
            }
        }
        return result;
    }

    Matrix add(const Matrix &lhs, const Matrix &rhs)
    {
        Matrix result{};
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            result[i] = lhs[i] + rhs[i];
        }
        return result;
    }

    Matrix subtract(const Matrix &lhs, const Matrix &rhs)
    {
        Matrix result{};
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            result[i] = lhs[i] - rhs[i];
        }
        return result;
    }

    Matrix scale(const Matrix &matrix, Complex factor)
    {
        Matrix result{};
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            result[i] = factor * matrix[i];
        }
        return result;
    }

    Matrix multiply(const Matrix &lhs, const Matrix &rhs)
    {
        Matrix result{};
        for (std::size_t row = 0; row < DIM; ++row)
        {
            for (std::size_t k = 0; k < DIM; ++k)
            {
                const Complex lhs_value = lhs[index(row, k)];
                if (lhs_value == Complex{})
                {
                    continue;
                }
                for (std::size_t col = 0; col < DIM; ++col)
                {
                    result[index(row, col)] += lhs_value * rhs[index(k, col)];
                }
            }
        }
        return result;
    }

    Complex trace_product(const Matrix &rho, const Matrix &op)
    {
        Complex result{};
        for (std::size_t row = 0; row < DIM; ++row)
        {
            for (std::size_t col = 0; col < DIM; ++col)
            {
                result += rho[index(row, col)] * op[index(col, row)];
            }
        }
        return result;
    }

    Matrix annihilation(std::size_t mode)
    {
        if (mode >= MODE_COUNT)
        {
            throw std::invalid_argument("Fermionic mode index is out of range.");
        }

        Matrix result{};
        const std::size_t mode_mask = std::size_t{1} << mode;
        const std::size_t lower_mask = mode_mask - 1;

        for (std::size_t ket = 0; ket < DIM; ++ket)
        {
            if ((ket & mode_mask) == 0)
            {
                continue;
            }

            const std::size_t bra = ket & ~mode_mask;
            const unsigned lower_occupancy =
                std::popcount(static_cast<unsigned>(ket & lower_mask));
            result[index(bra, ket)] = (lower_occupancy % 2 == 0) ? 1.0 : -1.0;
        }
        return result;
    }

    struct Operators
    {
        std::array<Matrix, MODE_COUNT> annihilation_ops{};
        std::array<Matrix, MODE_COUNT> creation_ops{};
        std::array<Matrix, MODE_COUNT> occupation_ops{};
        std::array<Matrix, MAJORANA_COUNT> majorana_ops{};
        Matrix total_number = zero_matrix();
        Matrix total_number_squared = zero_matrix();
        Matrix subsystem_parity = zero_matrix();
        std::array<std::array<Matrix, MODE_COUNT>, MODE_COUNT> hopping_ops{};
        std::array<std::array<Matrix, MODE_COUNT>, MODE_COUNT> pairing_ops{};
        std::array<std::array<Matrix, MODE_COUNT>, MODE_COUNT> density_pair_ops{};
        std::array<std::array<Matrix, MAJORANA_COUNT>, MAJORANA_COUNT>
            majorana_pair_ops{};
        std::array<Matrix, 15> majorana_four_ops{};
        std::array<std::array<std::size_t, 4>, 15> majorana_four_indices{};
        Matrix majorana_six_op = zero_matrix();

        Operators()
        {
            for (std::size_t mode = 0; mode < MODE_COUNT; ++mode)
            {
                annihilation_ops[mode] = annihilation(mode);
                creation_ops[mode] = dagger(annihilation_ops[mode]);
                occupation_ops[mode] =
                    multiply(creation_ops[mode], annihilation_ops[mode]);
                total_number = add(total_number, occupation_ops[mode]);

                majorana_ops[2 * mode] =
                    add(annihilation_ops[mode], creation_ops[mode]);
                majorana_ops[2 * mode + 1] = scale(
                    subtract(annihilation_ops[mode], creation_ops[mode]),
                    Complex{0.0, -1.0});
            }
            total_number_squared = multiply(total_number, total_number);

            for (std::size_t basis = 0; basis < DIM; ++basis)
            {
                const unsigned occupation = std::popcount(static_cast<unsigned>(basis));
                subsystem_parity[index(basis, basis)] =
                    (occupation % 2 == 0) ? 1.0 : -1.0;
            }

            for (std::size_t i = 0; i < MODE_COUNT; ++i)
            {
                for (std::size_t j = 0; j < MODE_COUNT; ++j)
                {
                    hopping_ops[i][j] = multiply(creation_ops[i], annihilation_ops[j]);
                    pairing_ops[i][j] = multiply(annihilation_ops[i], annihilation_ops[j]);
                    density_pair_ops[i][j] =
                        multiply(occupation_ops[i], occupation_ops[j]);
                }
            }

            for (std::size_t a = 0; a < MAJORANA_COUNT; ++a)
            {
                for (std::size_t b = 0; b < MAJORANA_COUNT; ++b)
                {
                    majorana_pair_ops[a][b] =
                        multiply(majorana_ops[a], majorana_ops[b]);
                }
            }

            std::size_t four_index = 0;
            for (std::size_t a = 0; a < MAJORANA_COUNT; ++a)
            {
                for (std::size_t b = a + 1; b < MAJORANA_COUNT; ++b)
                {
                    for (std::size_t c = b + 1; c < MAJORANA_COUNT; ++c)
                    {
                        for (std::size_t d = c + 1; d < MAJORANA_COUNT; ++d)
                        {
                            majorana_four_indices[four_index] = {a, b, c, d};
                            majorana_four_ops[four_index] = multiply(
                                multiply(majorana_pair_ops[a][b], majorana_ops[c]),
                                majorana_ops[d]);
                            ++four_index;
                        }
                    }
                }
            }
            if (four_index != majorana_four_ops.size())
            {
                throw std::logic_error("Incorrect number of four-Majorana operators.");
            }

            majorana_six_op = majorana_ops[0];
            for (std::size_t a = 1; a < MAJORANA_COUNT; ++a)
            {
                majorana_six_op = multiply(majorana_six_op, majorana_ops[a]);
            }
        }
    };

    double real_if_close(Complex value, const char *name)
    {
        const double tolerance = 1.0e-9 * (1.0 + std::abs(value.real()));
        if (std::abs(value.imag()) > tolerance)
        {
            static std::atomic<std::uint64_t> warning_count{0};
            const std::uint64_t warning_index = warning_count.fetch_add(
                1, std::memory_order_relaxed);
            if (warning_index < 8)
            {
                std::cerr << "Warning: " << name << " has imaginary component "
                          << value.imag() << "; using its real part.\n";
            }
        }
        return value.real();
    }

    using MajoranaMatrix =
        std::array<std::array<Complex, MAJORANA_COUNT>, MAJORANA_COUNT>;

    Complex pfaffian_mask(const MajoranaMatrix &matrix, unsigned mask)
    {
        if (mask == 0)
        {
            return Complex{1.0, 0.0};
        }

        const unsigned first = std::countr_zero(mask);
        const unsigned without_first = mask & ~(1u << first);
        Complex result{};

        for (unsigned paired = first + 1; paired < MAJORANA_COUNT; ++paired)
        {
            if ((without_first & (1u << paired)) == 0)
            {
                continue;
            }

            const unsigned between_mask =
                without_first & ((1u << paired) - 1u);
            const double sign =
                (std::popcount(between_mask) % 2 == 0) ? 1.0 : -1.0;
            result += sign * matrix[first][paired] * pfaffian_mask(
                matrix, without_first & ~(1u << paired));
        }
        return result;
    }

    Complex pfaffian(const MajoranaMatrix &matrix)
    {
        return pfaffian_mask(matrix, (1u << MAJORANA_COUNT) - 1u);
    }

    Matrix load_rho(const double *rho_ri)
    {
        if (rho_ri == nullptr)
        {
            throw std::invalid_argument("The 8x8 RDM pointer is null.");
        }

        Matrix rho{};
        for (std::size_t i = 0; i < DIM * DIM; ++i)
        {
            rho[i] = Complex{rho_ri[2 * i], rho_ri[2 * i + 1]};
        }
        return rho;
    }

    double clamp_small_negative(double value)
    {
        return (value < 0.0 && value > -1.0e-12) ? 0.0 : value;
    }
}

namespace mipt::analysis
{
    FermionExpVals8x8 compute_fermion_exp_vals_8x8(const double *rho_ri)
    {
        static const Operators operators;
        const Matrix rho = load_rho(rho_ri);

        FermionExpVals8x8 result;
        result.purity = real_if_close(trace_product(rho, rho), "purity");

        constexpr std::array<std::array<std::size_t, 2>, 3> pairs{{
            {{0, 1}},
            {{1, 2}},
            {{0, 2}},
        }};

        std::array<double, MODE_COUNT> occupations{};
        for (std::size_t mode = 0; mode < MODE_COUNT; ++mode)
        {
            occupations[mode] = real_if_close(
                trace_product(rho, operators.occupation_ops[mode]), "occupation");
        }

        for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index)
        {
            const std::size_t i = pairs[pair_index][0];
            const std::size_t j = pairs[pair_index][1];

            result.hopping_squared[pair_index] = std::norm(
                trace_product(rho, operators.hopping_ops[i][j]));
            result.pairing_squared[pair_index] = std::norm(
                trace_product(rho, operators.pairing_ops[i][j]));

            const double occupation_pair = real_if_close(
                trace_product(rho, operators.density_pair_ops[i][j]),
                "density pair");
            result.connected_density[pair_index] =
                occupation_pair - occupations[i] * occupations[j];
            result.connected_density_squared[pair_index] =
                result.connected_density[pair_index] *
                result.connected_density[pair_index];
        }

        result.number_mean = real_if_close(
            trace_product(rho, operators.total_number), "number mean");
        const double number_second_moment = real_if_close(
            trace_product(rho, operators.total_number_squared),
            "number second moment");
        result.number_variance = clamp_small_negative(
            number_second_moment - result.number_mean * result.number_mean);

        result.parity_expectation = real_if_close(
            trace_product(rho, operators.subsystem_parity), "parity expectation");
        result.parity_variance = clamp_small_negative(
            1.0 - result.parity_expectation * result.parity_expectation);

        std::array<std::array<Complex, MAJORANA_COUNT>, MAJORANA_COUNT>
            two_point{};
        for (std::size_t a = 0; a < MAJORANA_COUNT; ++a)
        {
            for (std::size_t b = 0; b < MAJORANA_COUNT; ++b)
            {
                two_point[a][b] = trace_product(
                    rho, operators.majorana_pair_ops[a][b]);
            }
        }

        for (std::size_t moment = 0;
             moment < operators.majorana_four_ops.size();
             ++moment)
        {
            const auto ids = operators.majorana_four_indices[moment];
            const std::size_t a = ids[0];
            const std::size_t b = ids[1];
            const std::size_t c = ids[2];
            const std::size_t d = ids[3];

            const Complex actual = trace_product(
                rho, operators.majorana_four_ops[moment]);
            const Complex wick =
                two_point[a][b] * two_point[c][d] -
                two_point[a][c] * two_point[b][d] +
                two_point[a][d] * two_point[b][c];
            result.wick4_residual_squared += std::norm(actual - wick);
        }

        MajoranaMatrix antisymmetric{};
        for (std::size_t a = 0; a < MAJORANA_COUNT; ++a)
        {
            for (std::size_t b = a + 1; b < MAJORANA_COUNT; ++b)
            {
                antisymmetric[a][b] = two_point[a][b];
                antisymmetric[b][a] = -two_point[a][b];
            }
        }

        const Complex actual_six =
            trace_product(rho, operators.majorana_six_op);
        result.wick6_residual_squared =
            std::norm(actual_six - pfaffian(antisymmetric));

        return result;
    }
}
