#pragma once

// Classical and quantum correlations between one reference qubit and a small
// receiver block, for the probe-mode-5 front-velocity protocol.
//
// The input is always a joint density matrix over `width` block modes plus the
// reference, with the reference as the *highest* local mode, so the basis index
// factorizes as `r * 2^width + b`.  Everything below is a function of that one
// matrix; extracting it from a state vector is probed/front.hpp's job, and the
// separation is what lets these formulas be tested without a GPU.
//
// Two families of quantity are computed:
//
//   * Classical.  Measuring the reference in Pauli basis mu prepares the
//     ensemble {p_a, rho_B^a} on the block, and the classical information that
//     has reached the block about that basis is its Holevo quantity
//
//         C_mu = S(rho_B) - sum_a p_a S(rho_B^a).
//
//     Because the reference was Bell-entangled with the injection site, this is
//     exactly the accessible information about a bit encoded in basis mu at the
//     source.  C_mu <= I(R:B) always, with equality only for a classically
//     correlated state, so the gap between max_mu C_mu and I(R:B) is itself a
//     coarse witness of quantum correlation.
//
//     Maximizing the same quantity over *all* projective measurements of the
//     reference, not just the three Pauli axes, gives the Henderson--Vedral
//     classical correlation J_{R->B}, and I(R:B) - J is the quantum discord.
//     That is the sharp version of the same witness and is what should be
//     reported; C_max is a lower bound on J and therefore only an upper bound
//     on the discord.  The optimization is not free, so it is opt-in.
//
//   * Quantum.  The R|B negativity and logarithmic negativity, plus the
//     coherent information I_c(R>B) = S(B) - S(RB), whose positivity certifies
//     that the block can recover the quantum state.
//
// Fermionic circuits: the reference is the last mode, so tracing it out costs
// no Jordan--Wigner sign and rho_B is an ordinary block sum, but tracing the
// *block* out to reach rho_R does, and the R|B partial transpose is the
// fermionic one (a factor of i on parity-violating entries, matching fgmn.cpp
// and ancilla_mi.hpp).  C_X and C_Y project the reference mode onto a
// superposition of occupations and so are not superselection-safe under a
// fermionic reading; C_Z is.  They are reported for all circuits because the
// probe ancilla is an explicit Bell partner throughout this codebase, but a
// fermionic front should be read off C_Z.  J and the discord inherit the same
// caveat, and more sharply: J maximizes over every measurement direction, so it
// is never superselection-safe.

#include "mipt/small_rdm.hpp"
#include "mipt/util/spectral.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace mipt::front
{
using Complex = std::complex<double>;
using ancilla::SmallRdm;

// The largest receiver block this path supports. One reference plus three block
// modes is a 16x16 joint matrix, which the dense Jacobi routines handle
// comfortably; going wider needs a different eigensolver, not just a bigger
// switch.
inline constexpr int MAX_BLOCK_WIDTH = 3;

struct BlockMetrics
{
    double joint_entropy = 0.0;        // S(rho_RB)
    double block_entropy = 0.0;        // S(rho_B)
    double reference_entropy = 0.0;    // S(rho_R)
    double mutual_information = 0.0;   // I(R:B)
    double coherent_information = 0.0; // I_c(R>B) = S(B) - S(RB)
    double negativity = 0.0;
    double log_negativity = 0.0;
    double classical_x = 0.0;
    double classical_y = 0.0;
    double classical_z = 0.0;
    double classical_mean = 0.0; // (C_X + C_Y + C_Z) / 3
    double classical_max = 0.0;
    // Filled only when the caller asks for the measurement optimization; NaN
    // otherwise, so an unoptimized run cannot be mistaken for a discord of zero.
    double classical_optimized = std::numeric_limits<double>::quiet_NaN(); // J_{R->B}
    double discord = std::numeric_limits<double>::quiet_NaN();             // D = I - J
};

namespace detail
{

inline SmallRdm normalized_hermitian(const SmallRdm &rho)
{
    double trace = 0.0;
    for (int i = 0; i < rho.dimension; ++i)
    {
        trace += std::real(rho(i, i));
    }
    if (!(trace > 0.0) || !std::isfinite(trace))
    {
        throw std::runtime_error("Receiver-block density matrix has invalid trace.");
    }

    SmallRdm normalized(rho.dimension);
    for (int row = 0; row < rho.dimension; ++row)
    {
        for (int col = row; col < rho.dimension; ++col)
        {
            Complex value = 0.5 * (rho(row, col) + std::conj(rho(col, row))) / trace;
            if (row == col)
            {
                value = Complex(std::real(value), 0.0);
            }
            normalized(row, col) = value;
            normalized(col, row) = std::conj(value);
        }
    }
    return normalized;
}

// Trace the reference out. It is the highest local mode, so it is already at
// the front of the Jordan--Wigner reordering used by partial_trace_fixed and
// the block modes keep their positions: the fermionic sign is identically +1
// and this is a plain sum over the reference's diagonal blocks. Writing it out
// directly keeps sum_a sigma_a == rho_B exact for the Holevo terms below.
inline SmallRdm trace_out_reference(const SmallRdm &rho, int width)
{
    const int block_dimension = 1 << width;
    SmallRdm reduced(block_dimension);
    for (int r = 0; r < 2; ++r)
    {
        const int offset = r * block_dimension;
        for (int row = 0; row < block_dimension; ++row)
        {
            for (int col = 0; col < block_dimension; ++col)
            {
                reduced(row, col) += rho(offset + row, offset + col);
            }
        }
    }
    return reduced;
}

// The conditional block state given outcome `vector` on the reference, left
// unnormalized so its trace is the outcome probability.
inline SmallRdm project_reference(const SmallRdm &rho, int width, const std::array<Complex, 2> &vector)
{
    const int block_dimension = 1 << width;
    SmallRdm conditional(block_dimension);
    for (int r = 0; r < 2; ++r)
    {
        for (int s = 0; s < 2; ++s)
        {
            const Complex weight = std::conj(vector[static_cast<std::size_t>(r)]) *
                                   vector[static_cast<std::size_t>(s)];
            if (weight == Complex(0.0, 0.0))
            {
                continue;
            }
            for (int row = 0; row < block_dimension; ++row)
            {
                for (int col = 0; col < block_dimension; ++col)
                {
                    conditional(row, col) +=
                        weight * rho(r * block_dimension + row, s * block_dimension + col);
                }
            }
        }
    }
    return conditional;
}

inline double trace_of(const SmallRdm &rho)
{
    double trace = 0.0;
    for (int i = 0; i < rho.dimension; ++i)
    {
        trace += std::real(rho(i, i));
    }
    return trace;
}

// Holevo information of the ensemble the reference measurement prepares.
//
// The two projectors are complementary, so sum_a sigma_a == rho_B and the
// leading term needs no recomputation. An outcome of vanishing probability
// contributes nothing: p S(rho) -> 0 as p -> 0 with S bounded by log(dim).
inline double holevo_information(const SmallRdm &rho, int width, double block_entropy,
                                 const std::array<Complex, 2> &plus, const std::array<Complex, 2> &minus)
{
    constexpr double negligible_probability = 1.0e-12;
    double conditional = 0.0;
    for (const auto &vector : {plus, minus})
    {
        const SmallRdm outcome = project_reference(rho, width, vector);
        const double probability = trace_of(outcome);
        if (!(probability > negligible_probability))
        {
            continue;
        }
        conditional += probability * ancilla::entropy_from_small_rdm(outcome);
    }

    double value = block_entropy - conditional;
    // The Holevo quantity is nonnegative by concavity; only roundoff can push
    // it below zero, and only by the entropy routine's own tolerance.
    if (value < 0.0 && value > -1.0e-6)
    {
        value = 0.0;
    }
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::runtime_error("Receiver-block classical information is invalid: " + std::to_string(value));
    }
    return value;
}

// The two orthogonal reference states a projective measurement along the Bloch
// direction (theta, phi) distinguishes.
//
// |+n> = (cos(theta/2), e^{i phi} sin(theta/2)) and its orthogonal partner. The
// two projectors are merely exchanged by n -> -n, so a measurement is a point of
// the *projective* sphere and a hemisphere of directions covers every distinct
// one.
inline std::array<std::array<Complex, 2>, 2> measurement_basis(double theta, double phi)
{
    const double cosine = std::cos(0.5 * theta);
    const double sine = std::sin(0.5 * theta);
    const Complex phase(std::cos(phi), std::sin(phi));
    return {std::array<Complex, 2>{Complex(cosine, 0.0), phase * sine},
            std::array<Complex, 2>{-std::conj(phase) * sine, Complex(cosine, 0.0)}};
}

// The Henderson--Vedral classical correlation J_{R->B}: the Holevo quantity
// maximized over every projective measurement of the reference.
//
// There is no closed form even for two qubits, so this is a numerical maximum
// over the Bloch sphere: a coarse sweep of the hemisphere, then a shrinking
// eight-neighbour search from the best point. The landscape is smooth and, on
// the weakly correlated states most of a front grid consists of, very flat, so
// the refinement matters less for the value than for making it independent of
// where the coarse grid happened to land. The three Pauli axes are points of the
// same landscape, hence J >= max(C_X, C_Y, C_Z) up to the search tolerance --
// which is the invariant worth testing, since a sign or indexing error in the
// basis would break it.
inline double optimized_classical_information(const SmallRdm &rho, int width, double block_entropy)
{
    constexpr double PI = 3.14159265358979323846;
    constexpr int THETA_STEPS = 6;
    constexpr int PHI_STEPS = 12;
    constexpr int REFINEMENTS = 16;

    const auto evaluate = [&](double theta, double phi) {
        const auto basis = measurement_basis(theta, phi);
        return holevo_information(rho, width, block_entropy, basis[0], basis[1]);
    };

    double best_theta = 0.0;
    double best_phi = 0.0;
    double best = evaluate(0.0, 0.0);
    for (int theta_index = 1; theta_index <= THETA_STEPS; ++theta_index)
    {
        const double theta = 0.5 * PI * static_cast<double>(theta_index) / THETA_STEPS;
        for (int phi_index = 0; phi_index < PHI_STEPS; ++phi_index)
        {
            const double phi = 2.0 * PI * static_cast<double>(phi_index) / PHI_STEPS;
            const double value = evaluate(theta, phi);
            if (value > best)
            {
                best = value;
                best_theta = theta;
                best_phi = phi;
            }
        }
    }

    double step = 0.5 * PI / THETA_STEPS;
    for (int round = 0; round < REFINEMENTS; ++round)
    {
        double candidate = best;
        double candidate_theta = best_theta;
        double candidate_phi = best_phi;
        for (int theta_offset = -1; theta_offset <= 1; ++theta_offset)
        {
            for (int phi_offset = -1; phi_offset <= 1; ++phi_offset)
            {
                if (theta_offset == 0 && phi_offset == 0)
                {
                    continue;
                }
                // theta beyond [0, pi] would re-parameterize a direction the
                // sweep already covers, so it is simply clipped away.
                const double theta = best_theta + theta_offset * step;
                if (theta < 0.0 || theta > PI)
                {
                    continue;
                }
                const double phi = best_phi + phi_offset * step;
                const double value = evaluate(theta, phi);
                if (value > candidate)
                {
                    candidate = value;
                    candidate_theta = theta;
                    candidate_phi = phi;
                }
            }
        }
        best = candidate;
        best_theta = candidate_theta;
        best_phi = candidate_phi;
        step *= 0.5;
    }
    return best;
}

// Trace norm of the R|B partial transpose, with the reference at bit `width`.
template <std::size_t Dimension>
inline double partial_transpose_trace_norm(const SmallRdm &rho, int width, bool fermionic)
{
    const int mask = 1 << width;
    std::array<Complex, Dimension * Dimension> transposed{};
    for (int row = 0; row < static_cast<int>(Dimension); ++row)
    {
        for (int col = 0; col < static_cast<int>(Dimension); ++col)
        {
            const int source_row = (row & ~mask) | (col & mask);
            const int source_col = (col & ~mask) | (row & mask);
            Complex value = rho(source_row, source_col);
            // Same convention as fgmn.cpp and ancilla_mi.hpp: after the ordinary
            // partial transpose, entries violating the transposed subsystem's
            // local parity pick up a factor of i.
            if (fermionic && (((row & mask) != 0) != ((col & mask) != 0)))
            {
                value *= Complex(0.0, 1.0);
            }
            transposed[static_cast<std::size_t>(row) * Dimension + static_cast<std::size_t>(col)] = value;
        }
    }

    double norm = 0.0;
    // An ordinary partial transpose of a Hermitian matrix stays Hermitian, so
    // its singular values are the absolute eigenvalues. Only the fermionic one
    // is non-Hermitian and needs the Gram matrix.
    const bool converged = fermionic ? util::gram_trace_norm<Dimension>(transposed, norm)
                                     : util::hermitian_trace_norm<Dimension>(transposed, norm);
    if (!converged || !std::isfinite(norm))
    {
        throw std::runtime_error("Receiver-block partial-transpose eigensolver did not converge.");
    }
    return norm;
}

inline double reference_negativity(const SmallRdm &rho, int width, bool fermionic)
{
    double norm = 0.0;
    switch (width)
    {
    case 1:
        norm = partial_transpose_trace_norm<4>(rho, width, fermionic);
        break;
    case 2:
        norm = partial_transpose_trace_norm<8>(rho, width, fermionic);
        break;
    case 3:
        norm = partial_transpose_trace_norm<16>(rho, width, fermionic);
        break;
    default:
        throw std::invalid_argument("Receiver-block width must lie in [1, 3].");
    }

    double value = 0.5 * (norm - 1.0);
    if (value < 0.0 && value > -2.0e-8)
    {
        value = 0.0;
    }
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::runtime_error("Receiver-block negativity is invalid: " + std::to_string(value));
    }
    return value;
}

} // namespace detail

// Every correlation measure for one (reference, receiver block) pair.
//
// `rho` spans `width` block modes plus the reference at local bit `width`; it
// need not be normalized or exactly Hermitian.
//
// `optimize_classical` adds J_{R->B} and the discord D = I - J. It costs a few
// hundred extra Holevo evaluations, each of them two entropies of a 2^width
// matrix, so it is cheap for the two-qubit probe pairs of the staggered
// protocol and distinctly not cheap for a width-3 receiver grid; the caller
// decides.
inline BlockMetrics block_metrics(const SmallRdm &rho, int width, bool fermionic, bool optimize_classical = false)
{
    if (width < 1 || width > MAX_BLOCK_WIDTH)
    {
        throw std::invalid_argument("Receiver-block width must lie in [1, " + std::to_string(MAX_BLOCK_WIDTH) + "].");
    }
    if (rho.dimension != (1 << (width + 1)))
    {
        throw std::invalid_argument("Receiver-block density matrix does not match its width.");
    }

    const SmallRdm normalized = detail::normalized_hermitian(rho);
    const SmallRdm block = detail::trace_out_reference(normalized, width);
    const SmallRdm reference =
        ancilla::partial_trace_fixed(normalized, width + 1, 1u << width, fermionic);

    BlockMetrics metrics;
    metrics.joint_entropy = ancilla::entropy_from_small_rdm(normalized);
    metrics.block_entropy = ancilla::entropy_from_small_rdm(block);
    metrics.reference_entropy = ancilla::entropy_from_small_rdm(reference);
    metrics.coherent_information = metrics.block_entropy - metrics.joint_entropy;

    double mutual = metrics.reference_entropy + metrics.block_entropy - metrics.joint_entropy;
    if (mutual < 0.0 && mutual > -2.0e-6)
    {
        mutual = 0.0;
    }
    if (!std::isfinite(mutual) || mutual < 0.0)
    {
        throw std::runtime_error("Receiver-block mutual information is invalid: " + std::to_string(mutual));
    }
    metrics.mutual_information = mutual;

    constexpr double inverse_root_two = 0.70710678118654752440;
    metrics.classical_z = detail::holevo_information(normalized, width, metrics.block_entropy, {Complex(1.0, 0.0), Complex(0.0, 0.0)},
                                                     {Complex(0.0, 0.0), Complex(1.0, 0.0)});
    metrics.classical_x =
        detail::holevo_information(normalized, width, metrics.block_entropy,
                                   {Complex(inverse_root_two, 0.0), Complex(inverse_root_two, 0.0)},
                                   {Complex(inverse_root_two, 0.0), Complex(-inverse_root_two, 0.0)});
    metrics.classical_y =
        detail::holevo_information(normalized, width, metrics.block_entropy,
                                   {Complex(inverse_root_two, 0.0), Complex(0.0, inverse_root_two)},
                                   {Complex(inverse_root_two, 0.0), Complex(0.0, -inverse_root_two)});
    metrics.classical_mean = (metrics.classical_x + metrics.classical_y + metrics.classical_z) / 3.0;
    metrics.classical_max = std::max(metrics.classical_z, std::max(metrics.classical_x, metrics.classical_y));

    if (optimize_classical)
    {
        // The Pauli axes are themselves candidate measurements, so taking the
        // better of the two costs nothing and makes J >= C_max hold exactly
        // rather than only to search tolerance -- which in turn keeps the
        // reported discord from ever exceeding I - C_max.
        metrics.classical_optimized =
            std::max(metrics.classical_max, detail::optimized_classical_information(normalized, width, metrics.block_entropy));
        double discord = metrics.mutual_information - metrics.classical_optimized;
        // Under the qubit trace J <= I is the data-processing inequality, so
        // only the search tolerance and the entropy routine's own roundoff can
        // push the difference below zero, and anything larger is a bug.
        //
        // Under the fermionic trace it is not a theorem at all. The reference
        // marginal there carries Jordan--Wigner signs, so S(R) is not the
        // entropy of the state the Holevo terms were computed from and the
        // difference is genuinely negative on states that violate the parity
        // superselection rule -- by O(0.1) on random ones. That is the same
        // caveat C_X and C_Y carry, and it is handled the same way: the value
        // is reported rather than suppressed, but a fermionic discord is an
        // approximate witness, not the Henderson--Vedral quantity.
        if (discord < 0.0 && (fermionic || discord > -1.0e-6))
        {
            discord = 0.0;
        }
        if (!std::isfinite(discord) || discord < 0.0)
        {
            throw std::runtime_error("Receiver-block discord is invalid: " + std::to_string(discord));
        }
        metrics.discord = discord;
    }

    metrics.negativity = detail::reference_negativity(normalized, width, fermionic);
    metrics.log_negativity = std::log1p(2.0 * metrics.negativity);
    if (!std::isfinite(metrics.log_negativity) || metrics.log_negativity < 0.0)
    {
        throw std::runtime_error("Receiver-block logarithmic negativity is invalid.");
    }
    return metrics;
}

} // namespace mipt::front
