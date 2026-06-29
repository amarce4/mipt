#pragma once

#include <stdint.h>

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
double compute_gmn_mosek_real_8x8(const double *rho_real_row_major);
double compute_gmn_mosek_complex_8x8(const double *rho_ri_row_major);

typedef struct GmnMosekDiagnostics
{
    uint64_t workspace_failures;
    uint64_t optimize_errors;
    uint64_t nonfinite_objectives;
    uint64_t accepted_nonoptimal_solutions;
} GmnMosekDiagnostics;

void gmn_mosek_reset_diagnostics(void);
GmnMosekDiagnostics gmn_mosek_get_diagnostics(void);

#ifdef __cplusplus
}
#endif
