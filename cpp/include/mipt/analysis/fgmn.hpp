#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Three-party GMN/fGMN entry points for one 8x8 reduced density matrix.
 *
 * Complex matrices use interleaved row-major storage:
 *   rho_ri[2*(row*8+col)+0] = Re(rho[row,col])
 *   rho_ri[2*(row*8+col)+1] = Im(rho[row,col])
 *
 * fgmn.exe uses compute_fgmn_mosek_8x8_cpp on the full complex 8x8 RDMs
 * written by mipt.exe/rho3.bin-style files.
 */
double compute_fgmn_mosek_8x8_cpp(const double *rho_ri_row_major);

/*
 * Cheap upper-bound prefilter helper.  Returns
 *   min_s N_F_s(rho)
 * where s runs over the three one-party-vs-rest bipartitions.  Since
 * fGMN(rho) <= min_s N_F_s(rho), a zero value certifies fGMN = 0
 * up to the caller's numerical tolerance.
 */
double compute_min_bipartite_fermionic_negativity_8x8_cpp(
    const double *rho_ri_row_major);

/* Non-fermionic helpers retained for validation and comparison. */
double compute_gmn_mosek_real_8x8_cpp(const double *rho_real_row_major);
double compute_gmn_mosek_complex_8x8_cpp(const double *rho_ri_row_major);

#ifdef __cplusplus
}

namespace fgmn
{
    struct GmnWorkspace;

    GmnWorkspace &workspace_real();
    GmnWorkspace &workspace_complex();
    GmnWorkspace &workspace_fermion();
    GmnWorkspace &workspace(bool is_complex = false, bool is_fermionic = false);

    // 0 means no in-process limiter.  Positive values limit concurrent
    // MOSEK/Fusion solve calls in this process; useful because Fusion can
    // segfault under some OpenMP-concurrent workloads.
    int configured_fusion_concurrency_limit();
}

#endif
