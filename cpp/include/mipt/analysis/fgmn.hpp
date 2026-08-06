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

/* Minimum ordinary bipartite negativity over the three one-vs-rest cuts. */
double compute_min_bipartite_negativity_8x8_cpp(
    const double *rho_ri_row_major);

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

#include <algorithm>
#include <string>
#include <thread>

namespace mipt::analysis
{

// How many GMN/fGMN solves the deferred-SDP runners should keep in flight.
//
// Shared by mipt_probed.exe and dist_scaling.exe so the two cannot drift.  The
// SDP dominates both -- it outweighs the circuit trajectory that produced its
// RDM by two orders of magnitude -- so this number, not anything on the GPU
// side, sets the wall time of a three-probe run.
//
// Measured on a 12-core box, mipt_probed mode 4 with three probes at N=12:
//
//     workers        2        4        6
//     small run   32.2 s   21.2 s   22.5 s
//     large run   88.5 s   53.0 s   43.5 s
//
// Going from the old hard-coded 2 to 4 is worth 1.5-1.7x.  Past that the wall
// time flattens while CPU keeps climbing (76 s -> 115 s of user time on the
// small run), which is contention rather than throughput, so the ratio is kept
// conservative and capped.  Both runners honour an explicit
// FGMN_MAX_CONCURRENT_MOSEK, so this is only the default.
inline int default_sdp_worker_count()
{
    const unsigned hardware = std::thread::hardware_concurrency();
    if (hardware == 0)
    {
        return 2;
    }
    return std::clamp(static_cast<int>(hardware / 3u), 2, 8);
}

inline std::string default_sdp_worker_count_text()
{
    return std::to_string(default_sdp_worker_count());
}

// Interior-point tolerance the runners ask MOSEK for.
//
// Both runners average thousands of samples per bin, so the standard error of a
// reported mean is ~1e-3; solving each individual SDP to MOSEK's default ~1e-8
// buys nothing.  Measured per solve on random rank-4 8x8 states, with the error
// against the |GHZ> and |W> closed forms:
//
//     tol        GMN      fGMN     error
//     default  219 ms    514 ms    3e-6
//     1e-6     202 ms    449 ms    1.4e-6
//     1e-5     169 ms    396 ms    1.4e-6      <- 1.30x, accuracy unchanged
//     1e-4     147 ms    336 ms    3e-4        <- falls off a cliff
//
// 1e-5 sits right at the knee: still five orders of magnitude tighter than the
// statistical error on any reported mean, and the last setting before accuracy
// degrades.  Set GMN_MOSEK_TOL explicitly to override, including back to
// MOSEK's own default.
inline constexpr const char *DEFAULT_SDP_TOLERANCE_TEXT = "1e-5";

} // namespace mipt::analysis

#endif
