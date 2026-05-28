#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Expects an 8x8 real density matrix in row-major order.
// This matches your current Python socket behavior, which only used real(rho(i,j)).
double compute_gmn_mosek_real_8x8(const double *rho_real_row_major);

#ifdef __cplusplus
}
#endif