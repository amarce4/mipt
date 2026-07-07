#pragma once

#include "fusion.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fully decomposable GMN SDP for one three-qubit, 8x8 RDM.
 *
 * compute_gmn_mosek_real_8x8:
 *   rho_real_row_major has 64 doubles: Re(rho) in row-major order.
 *   This is the faster legacy real-only SDP.
 *
 * compute_gmn_mosek_complex_8x8:
 *   rho_ri_row_major has 128 doubles: interleaved row-major complex entries,
 *   i.e. rho_ri[2*(row*8+col)+0] = Re(rho[row,col]) and
 *   rho_ri[2*(row*8+col)+1] = Im(rho[row,col]).
 *   This solves the complex Hermitian SDP by realifying each 8x8 Hermitian
 *   PSD block as a structured 16x16 real symmetric PSD block.
 */

double compute_gmn_mosek_real_8x8_cpp(const double *rho_real_row_major);
double compute_gmn_mosek_complex_8x8_cpp(const double *rho_ri_row_major);
double compute_fgmn_mosek_8x8_cpp(const double *rho_real_row_major);

#ifdef __cplusplus
}

// Some external definitions of the GMN workspace, so they can be referenced and 
// multiple different instances can be created, without having to go through the compute functions.
namespace fgmn{
    struct GmnWorkspace
    {
        mosek::fusion::Model::t M;
        mosek::fusion::Parameter::t objective_coeffs;
        std::shared_ptr<monty::ndarray<double, 1>> coeff_values;
        bool is_fermionic;
        bool is_complex;

        GmnWorkspace(bool complex_in = false,bool fermionic_in = false);

        double solve(const double *rho);
    };
    GmnWorkspace &workspace(bool is_complex = false, bool is_fermionic = false);
}

#endif

