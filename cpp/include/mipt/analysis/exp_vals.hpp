#pragma once

#include <array>

namespace mipt::analysis
{
    // Expectation-value diagnostics for one fermionic three-mode (8x8) RDM.
    // Pair order is (0,1), (1,2), (0,2), matching the CSV suffixes 01, 12, 02.
    struct FermionExpVals8x8
    {
        double purity = 0.0;
        std::array<double, 3> hopping_squared{};
        std::array<double, 3> pairing_squared{};
        double number_mean = 0.0;
        double number_variance = 0.0;
        double parity_expectation = 0.0;
        double parity_variance = 0.0;
        std::array<double, 3> connected_density{};
        std::array<double, 3> connected_density_squared{};
        double wick4_residual_squared = 0.0;
        double wick6_residual_squared = 0.0;
    };

    // rho_ri is an interleaved row-major complex 8x8 matrix:
    // rho_ri[2*(row*8+col)+0] = Re rho[row,col]
    // rho_ri[2*(row*8+col)+1] = Im rho[row,col]
    //
    // The local basis must use reduced-basis bit k for retained mode k, as in
    // mipt::io::DensityFileMetadata::subsystem_qubits.
    FermionExpVals8x8 compute_fermion_exp_vals_8x8(const double *rho_ri);
}
