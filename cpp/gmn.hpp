#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Real-valued fully decomposable GMN SDP for one three-qubit, 8x8 RDM.
 * It uses all three unique bipartitions. The saved MIPT density files contain
 * complete complex matrices; gmn.exe currently supplies Re(rho) to this
 * real-valued solver and reports discarded imaginary-component norms.
 */
double compute_gmn_mosek_real_8x8(const double *rho_real_row_major);

#ifdef __cplusplus
}
#endif
