#include "mipt/analysis/gmn.hpp"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "mosek.h"

/*
 * Fully decomposable GMN witness SDP through the MOSEK C API.
 *
 * For each unique bipartition mask m:
 *     W = P_m + PT_m(Q_m)
 *     0 <= P_m <= I, 0 <= Q_m <= I
 * and the objective is:
 *     minimize Re Tr(rho W).
 *
 * The returned quantity is -min Re Tr(rho W), matching the sign convention in
 * the previous implementation.
 *
 * The retained system is fixed at three qubits; its three unique
 * bipartitions are represented by masks 1, 2, and 3.
 *
 * MOSEK's C API bar variables are real symmetric PSD blocks. Complex mode uses
 * the standard realification of an 8x8 Hermitian matrix X = A + iB:
 *
 *     R(X) = [ A  -B ]
 *            [ B   A ]
 *
 * The old complex path explicitly enforced this block structure with equality
 * constraints. Keep those constraints enabled by default. Without them, MOSEK
 * can return non-OPTIMAL solution statuses consistently on the realified
 * relaxation even though the mathematical complex problem is bounded and
 * feasible. Set GMN_COMPLEX_STRUCTURE=0 only for profiling/experiments. Since
 * Tr(R(C)R(X)) = 2 Re Tr(CX), complex-mode objective coefficients use
 * 0.5 * R(rho) and 0.5 * R(PT(rho)).
 *
 * Performance notes:
 *   - A MOSEK task is cached per host thread and per mode (real/complex).
 *   - Per-solve heap allocation has been removed from the hot path.
 *   - The interior-point solution is explicitly deleted after each solve so
 *     MOSEK can release solution storage before the next objective update.
 *   - The task is periodically rebuilt by default to avoid long-run task-state
 *     accumulation on very large scans. Set GMN_MOSEK_RECYCLE=0 to disable.
 */

typedef struct
{
    int complex_mode;
    int cd; /* complex dimension, always 8 here */
    int d;  /* real bar-variable dimension: 8 in real mode, 16 in complex mode */
    int cuts;
    int upper_count;
    int primary_barvars;
    int num_barvars;
    int upper_bound_cons;
    int witness_equality_cons;
    int structure_cons;
    int num_cons;
    int recycle_after;
    int solves_since_rebuild;
    int *masks;
    MSKint64t *entry_basis;
    MSKrealt *coeff_p;
    MSKrealt *coeff_q;
    double *rho_pt;
    double *work_matrix;
    MSKenv_t env;
    MSKtask_t task;
} GmnWorkspace;

static _Thread_local GmnWorkspace *tls_ws_real = NULL;
static _Thread_local GmnWorkspace *tls_ws_complex = NULL;

static _Atomic uint64_t diag_workspace_failures = 0;
static _Atomic uint64_t diag_optimize_errors = 0;
static _Atomic uint64_t diag_nonfinite_objectives = 0;
static _Atomic uint64_t diag_accepted_nonoptimal_solutions = 0;

void gmn_mosek_reset_diagnostics(void)
{
    atomic_store(&diag_workspace_failures, 0);
    atomic_store(&diag_optimize_errors, 0);
    atomic_store(&diag_nonfinite_objectives, 0);
    atomic_store(&diag_accepted_nonoptimal_solutions, 0);
}

GmnMosekDiagnostics gmn_mosek_get_diagnostics(void)
{
    GmnMosekDiagnostics out;
    out.workspace_failures = atomic_load(&diag_workspace_failures);
    out.optimize_errors = atomic_load(&diag_optimize_errors);
    out.nonfinite_objectives = atomic_load(&diag_nonfinite_objectives);
    out.accepted_nonoptimal_solutions =
        atomic_load(&diag_accepted_nonoptimal_solutions);
    return out;
}

static int parse_int_env(const char *name, int default_value,
                         int min_value, int max_value)
{
    const char *s = getenv(name);
    if (s == NULL || *s == '\0')
    {
        return default_value;
    }

    char *end = NULL;
    long value;
    errno = 0;
    value = strtol(s, &end, 10);

    if (errno != 0 || end == s || *end != '\0' ||
        value < min_value || value > max_value)
    {
        return default_value;
    }

    return (int)value;
}

static int upper_index(int d, int row, int col)
{
    int r = row;
    int c = col;
    if (r > c)
    {
        const int tmp = r;
        r = c;
        c = tmp;
    }
    return r * d - (r * (r - 1)) / 2 + (c - r);
}

static void partial_transpose_source_index(int row,
                                           int col,
                                           int subsystem_mask,
                                           int *src_row,
                                           int *src_col)
{
    *src_row = (row & ~subsystem_mask) | (col & subsystem_mask);
    *src_col = (col & ~subsystem_mask) | (row & subsystem_mask);
}

static MSKrescodee set_fixed_bound(MSKtask_t task, int con, double rhs)
{
    return MSK_putconbound(task, (MSKint32t)con, MSK_BK_FX, rhs, rhs);
}

static void destroy_workspace(GmnWorkspace *ws)
{
    if (ws == NULL)
    {
        return;
    }
    if (ws->task != NULL)
    {
        MSK_deletetask(&ws->task);
    }
    if (ws->env != NULL)
    {
        MSK_deleteenv(&ws->env);
    }
    free(ws->work_matrix);
    free(ws->rho_pt);
    free(ws->coeff_q);
    free(ws->coeff_p);
    free(ws->entry_basis);
    free(ws->masks);
    free(ws);
}

static MSKrescodee append_entry_basis(GmnWorkspace *ws)
{
    MSKrescodee r = MSK_RES_OK;

    for (int row = 0; row < ws->d && r == MSK_RES_OK; ++row)
    {
        for (int col = row; col < ws->d && r == MSK_RES_OK; ++col)
        {
            const int k = upper_index(ws->d, row, col);
            const MSKint32t subi = (MSKint32t)col;
            const MSKint32t subj = (MSKint32t)row;
            const MSKrealt value = (row == col) ? 1.0 : 0.5;

            r = MSK_appendsparsesymmat(ws->task,
                                       ws->d,
                                       1,
                                       &subi,
                                       &subj,
                                       &value,
                                       &ws->entry_basis[k]);
        }
    }

    return r;
}

static MSKrescodee add_upper_bounds(GmnWorkspace *ws, int *constraint_index)
{
    MSKrescodee r = MSK_RES_OK;
    const MSKrealt one = 1.0;

    for (int x = 0; x < ws->primary_barvars && r == MSK_RES_OK; ++x)
    {
        const int slack = ws->primary_barvars + x;

        for (int row = 0; row < ws->d && r == MSK_RES_OK; ++row)
        {
            for (int col = row; col < ws->d && r == MSK_RES_OK; ++col)
            {
                const int con = (*constraint_index)++;
                const int k = upper_index(ws->d, row, col);
                const double rhs = (row == col) ? 1.0 : 0.0;

                r = MSK_putbaraij(ws->task, con, x, 1,
                                  &ws->entry_basis[k], &one);
                if (r == MSK_RES_OK)
                {
                    r = MSK_putbaraij(ws->task, con, slack, 1,
                                      &ws->entry_basis[k], &one);
                }
                if (r == MSK_RES_OK)
                {
                    r = set_fixed_bound(ws->task, con, rhs);
                }
            }
        }
    }

    return r;
}

static void realified_pt_source_index(int cd,
                                      int row,
                                      int col,
                                      int subsystem_mask,
                                      int *src_row,
                                      int *src_col)
{
    const int row_block = (row >= cd) ? 1 : 0;
    const int col_block = (col >= cd) ? 1 : 0;
    const int local_row = row_block ? row - cd : row;
    const int local_col = col_block ? col - cd : col;
    int pt_row, pt_col;

    partial_transpose_source_index(local_row, local_col, subsystem_mask,
                                   &pt_row, &pt_col);

    if (row_block == col_block)
    {
        /* Re(X[pt_row,pt_col]) */
        *src_row = pt_row;
        *src_col = pt_col;
    }
    else if (row_block == 0)
    {
        /* -Im(X[pt_row,pt_col]) = R(X)[pt_row, cd + pt_col]. */
        *src_row = pt_row;
        *src_col = cd + pt_col;
    }
    else
    {
        /* Im(X[pt_row,pt_col]) = R(X)[cd + pt_row, pt_col]. */
        *src_row = cd + pt_row;
        *src_col = pt_col;
    }
}

static void pt_source_for_workspace(const GmnWorkspace *ws,
                                    int row,
                                    int col,
                                    int subsystem_mask,
                                    int *src_row,
                                    int *src_col)
{
    if (ws->complex_mode)
    {
        realified_pt_source_index(ws->cd, row, col, subsystem_mask,
                                  src_row, src_col);
    }
    else
    {
        partial_transpose_source_index(row, col, subsystem_mask,
                                       src_row, src_col);
    }
}

static MSKrescodee add_witness_equalities(GmnWorkspace *ws,
                                           int *constraint_index)
{
    MSKrescodee r = MSK_RES_OK;
    const MSKrealt plus_one = 1.0;
    const MSKrealt minus_one = -1.0;
    const int anchor_mask = ws->masks[0];

    for (int cut = 1; cut < ws->cuts && r == MSK_RES_OK; ++cut)
    {
        const int p_other = 2 * cut;
        const int q_other = p_other + 1;
        const int other_mask = ws->masks[cut];

        for (int row = 0; row < ws->d && r == MSK_RES_OK; ++row)
        {
            for (int col = row; col < ws->d && r == MSK_RES_OK; ++col)
            {
                const int con = (*constraint_index)++;
                const int target = upper_index(ws->d, row, col);
                int src_anchor_row, src_anchor_col;
                int src_other_row, src_other_col;

                pt_source_for_workspace(ws, row, col, anchor_mask,
                                        &src_anchor_row, &src_anchor_col);
                pt_source_for_workspace(ws, row, col, other_mask,
                                        &src_other_row, &src_other_col);

                const int source_anchor =
                    upper_index(ws->d, src_anchor_row, src_anchor_col);
                const int source_other =
                    upper_index(ws->d, src_other_row, src_other_col);

                /* P_0 + PT_0(Q_0) - P_cut - PT_cut(Q_cut) = 0. */
                r = MSK_putbaraij(ws->task, con, 0, 1,
                                  &ws->entry_basis[target], &plus_one);
                if (r == MSK_RES_OK)
                {
                    r = MSK_putbaraij(ws->task, con, 1, 1,
                                      &ws->entry_basis[source_anchor],
                                      &plus_one);
                }
                if (r == MSK_RES_OK)
                {
                    r = MSK_putbaraij(ws->task, con, p_other, 1,
                                      &ws->entry_basis[target], &minus_one);
                }
                if (r == MSK_RES_OK)
                {
                    r = MSK_putbaraij(ws->task, con, q_other, 1,
                                      &ws->entry_basis[source_other],
                                      &minus_one);
                }
                if (r == MSK_RES_OK)
                {
                    r = set_fixed_bound(ws->task, con, 0.0);
                }
            }
        }
    }

    return r;
}

static int use_explicit_complex_structure_constraints(void)
{
    return parse_int_env("GMN_COMPLEX_STRUCTURE", 1, 0, 1) != 0;
}

static MSKrescodee add_complex_structure_constraints(GmnWorkspace *ws,
                                                     int *constraint_index)
{
    MSKrescodee r = MSK_RES_OK;
    const MSKrealt plus_one = 1.0;
    const MSKrealt minus_one = -1.0;
    const MSKrealt two = 2.0;
    const int n = ws->cd;

    if (!ws->complex_mode)
    {
        return MSK_RES_OK;
    }

    for (int var = 0; var < ws->num_barvars && r == MSK_RES_OK; ++var)
    {
        /* Top-left block equals bottom-right block: A == A. */
        for (int i = 0; i < n && r == MSK_RES_OK; ++i)
        {
            for (int j = i; j < n && r == MSK_RES_OK; ++j)
            {
                const int con = (*constraint_index)++;
                const int tl = upper_index(ws->d, i, j);
                const int br = upper_index(ws->d, n + i, n + j);

                const MSKint64t mats[2] = {ws->entry_basis[tl],
                                           ws->entry_basis[br]};
                const MSKrealt vals[2] = {plus_one, minus_one};

                r = MSK_putbaraij(ws->task, con, var, 2, mats, vals);
                if (r == MSK_RES_OK)
                {
                    r = set_fixed_bound(ws->task, con, 0.0);
                }
            }
        }

        /* Top-right block is skew-symmetric: T_ij + T_ji == 0. */
        for (int i = 0; i < n && r == MSK_RES_OK; ++i)
        {
            for (int j = i; j < n && r == MSK_RES_OK; ++j)
            {
                const int con = (*constraint_index)++;
                const int tij = upper_index(ws->d, i, n + j);
                const int tji = upper_index(ws->d, j, n + i);

                if (i == j)
                {
                    r = MSK_putbaraij(ws->task, con, var, 1,
                                      &ws->entry_basis[tij], &two);
                }
                else
                {
                    const MSKint64t mats[2] = {ws->entry_basis[tij],
                                               ws->entry_basis[tji]};
                    const MSKrealt vals[2] = {plus_one, plus_one};
                    r = MSK_putbaraij(ws->task, con, var, 2, mats, vals);
                }
                if (r == MSK_RES_OK)
                {
                    r = set_fixed_bound(ws->task, con, 0.0);
                }
            }
        }
    }

    return r;
}

static void set_optional_threads(MSKtask_t task)
{
    const char *s = getenv("GMN_MOSEK_NUM_THREADS");
    if (s != NULL && *s != '\0')
    {
        char *end = NULL;
        long value;
        errno = 0;
        value = strtol(s, &end, 10);
        if (errno == 0 && end != s && *end == '\0' &&
            value > 0 && value <= INT_MAX)
        {
            (void)MSK_putintparam(task,
                                  MSK_IPAR_NUM_THREADS,
                                  (MSKint32t)value);
        }
    }
}

static GmnWorkspace *create_workspace(int complex_mode)
{
    GmnWorkspace *ws = (GmnWorkspace *)calloc(1, sizeof(*ws));
    if (ws == NULL)
    {
        return NULL;
    }

    ws->complex_mode = complex_mode ? 1 : 0;
    ws->cd = 8;
    ws->d = complex_mode ? 16 : 8;
    ws->cuts = 3;
    ws->upper_count = ws->d * (ws->d + 1) / 2;
    ws->primary_barvars = 2 * ws->cuts;
    ws->num_barvars = 2 * ws->primary_barvars;
    ws->upper_bound_cons = ws->primary_barvars * ws->upper_count;
    ws->witness_equality_cons = (ws->cuts - 1) * ws->upper_count;
    ws->structure_cons = (complex_mode &&
                          use_explicit_complex_structure_constraints())
                             ? ws->num_barvars * ws->cd * (ws->cd + 1)
                             : 0;
    ws->num_cons = ws->upper_bound_cons +
                   ws->witness_equality_cons +
                   ws->structure_cons;
    ws->recycle_after = parse_int_env("GMN_MOSEK_RECYCLE", 10000, 0, INT_MAX);

    ws->masks = (int *)calloc((size_t)ws->cuts, sizeof(int));
    ws->entry_basis =
        (MSKint64t *)calloc((size_t)ws->upper_count, sizeof(MSKint64t));
    ws->coeff_p =
        (MSKrealt *)calloc((size_t)ws->upper_count, sizeof(MSKrealt));
    ws->coeff_q =
        (MSKrealt *)calloc((size_t)ws->upper_count, sizeof(MSKrealt));
    ws->rho_pt = (double *)calloc((size_t)2 * 8u * 8u, sizeof(double));
    ws->work_matrix =
        (double *)calloc((size_t)ws->d * (size_t)ws->d, sizeof(double));

    if (ws->masks == NULL || ws->entry_basis == NULL ||
        ws->coeff_p == NULL || ws->coeff_q == NULL ||
        ws->rho_pt == NULL || ws->work_matrix == NULL)
    {
        destroy_workspace(ws);
        return NULL;
    }

    for (int cut = 0; cut < ws->cuts; ++cut)
    {
        ws->masks[cut] = cut + 1;
    }

    MSKrescodee r = MSK_makeenv(&ws->env, NULL);
    if (r == MSK_RES_OK)
    {
        r = MSK_maketask(ws->env, ws->num_cons, 0, &ws->task);
    }
    if (r == MSK_RES_OK)
    {
        r = MSK_appendcons(ws->task, ws->num_cons);
    }
    if (r == MSK_RES_OK)
    {
        MSKint32t *dims =
            (MSKint32t *)calloc((size_t)ws->num_barvars, sizeof(MSKint32t));
        if (dims == NULL)
        {
            destroy_workspace(ws);
            return NULL;
        }
        for (int i = 0; i < ws->num_barvars; ++i)
        {
            dims[i] = (MSKint32t)ws->d;
        }
        r = MSK_appendbarvars(ws->task, ws->num_barvars, dims);
        free(dims);
    }
    if (r == MSK_RES_OK)
    {
        r = MSK_putobjsense(ws->task, MSK_OBJECTIVE_SENSE_MINIMIZE);
    }
    if (r == MSK_RES_OK)
    {
        r = MSK_putintparam(ws->task, MSK_IPAR_LOG, 0);
    }
    if (r == MSK_RES_OK)
    {
        set_optional_threads(ws->task);
        r = append_entry_basis(ws);
    }

    int con = 0;
    if (r == MSK_RES_OK)
    {
        r = add_upper_bounds(ws, &con);
    }
    if (r == MSK_RES_OK)
    {
        r = add_witness_equalities(ws, &con);
    }
    if (r == MSK_RES_OK && ws->structure_cons > 0)
    {
        r = add_complex_structure_constraints(ws, &con);
    }
    if (r == MSK_RES_OK && con != ws->num_cons)
    {
        r = MSK_RES_ERR_INTERNAL;
    }

    if (r != MSK_RES_OK)
    {
        destroy_workspace(ws);
        return NULL;
    }

    return ws;
}

static GmnWorkspace *workspace(int complex_mode)
{
    GmnWorkspace **slot = complex_mode ? &tls_ws_complex : &tls_ws_real;
    if (*slot == NULL)
    {
        *slot = create_workspace(complex_mode);
    }
    return *slot;
}

static void maybe_recycle_workspace(GmnWorkspace *ws)
{
    if (ws == NULL || ws->recycle_after <= 0)
    {
        return;
    }
    if (ws->solves_since_rebuild < ws->recycle_after)
    {
        return;
    }

    const int complex_mode = ws->complex_mode;
    if (complex_mode)
    {
        destroy_workspace(tls_ws_complex);
        tls_ws_complex = create_workspace(1);
    }
    else
    {
        destroy_workspace(tls_ws_real);
        tls_ws_real = create_workspace(0);
    }
}

static void fill_upper_coefficients(const GmnWorkspace *ws,
                                    const double *matrix,
                                    MSKrealt *out)
{
    for (int row = 0; row < ws->d; ++row)
    {
        for (int col = row; col < ws->d; ++col)
        {
            const int k = upper_index(ws->d, row, col);
            out[k] = (row == col)
                         ? matrix[row * ws->d + row]
                         : matrix[row * ws->d + col] +
                               matrix[col * ws->d + row];
        }
    }
}

static void partial_transpose_matrix_real(const GmnWorkspace *ws,
                                          const double *in,
                                          int subsystem_mask,
                                          double *out)
{
    for (int row = 0; row < ws->d; ++row)
    {
        for (int col = 0; col < ws->d; ++col)
        {
            int src_row, src_col;
            partial_transpose_source_index(row, col, subsystem_mask,
                                           &src_row, &src_col);
            out[row * ws->d + col] = in[src_row * ws->d + src_col];
        }
    }
}

static void partial_transpose_matrix_complex_ri(const double *in_ri,
                                                int subsystem_mask,
                                                double *out_ri)
{
    const int d = 8;
    for (int row = 0; row < d; ++row)
    {
        for (int col = 0; col < d; ++col)
        {
            int src_row, src_col;
            partial_transpose_source_index(row, col, subsystem_mask,
                                           &src_row, &src_col);
            const int dst = 2 * (row * d + col);
            const int src = 2 * (src_row * d + src_col);
            out_ri[dst + 0] = in_ri[src + 0];
            out_ri[dst + 1] = in_ri[src + 1];
        }
    }
}

static void realify_complex_8x8(const double *rho_ri,
                                double scale,
                                double *out_16x16)
{
    const int n = 8;
    const int d = 16;
    memset(out_16x16, 0, (size_t)d * (size_t)d * sizeof(double));

    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            const int src = 2 * (row * n + col);
            const double re = scale * rho_ri[src + 0];
            const double im = scale * rho_ri[src + 1];

            out_16x16[row * d + col] = re;
            out_16x16[row * d + (n + col)] = -im;
            out_16x16[(n + row) * d + col] = im;
            out_16x16[(n + row) * d + (n + col)] = re;
        }
    }
}

static double optimize_current_task(GmnWorkspace *ws)
{
    if (ws == NULL || ws->task == NULL)
    {
        atomic_fetch_add(&diag_workspace_failures, 1);
        return NAN;
    }

    (void)MSK_deletesolution(ws->task, MSK_SOL_ITR);

    MSKrescodee r =
        MSK_putbarcj(ws->task, 0, ws->upper_count, ws->entry_basis, ws->coeff_p);
    if (r == MSK_RES_OK)
    {
        r = MSK_putbarcj(ws->task, 1, ws->upper_count,
                         ws->entry_basis, ws->coeff_q);
    }

    MSKrescodee trm = MSK_RES_OK;
    if (r == MSK_RES_OK)
    {
        r = MSK_optimizetrm(ws->task, &trm);
    }

    if (r != MSK_RES_OK)
    {
        atomic_fetch_add(&diag_optimize_errors, 1);
        (void)MSK_deletesolution(ws->task, MSK_SOL_ITR);
        ++ws->solves_since_rebuild;
        maybe_recycle_workspace(ws);
        return NAN;
    }

    MSKsolstae solsta = MSK_SOL_STA_UNKNOWN;
    const MSKrescodee solsta_r = MSK_getsolsta(ws->task, MSK_SOL_ITR, &solsta);

    MSKrealt objective = 0.0;
    r = MSK_getprimalobj(ws->task, MSK_SOL_ITR, &objective);

    (void)MSK_deletesolution(ws->task, MSK_SOL_ITR);
    ++ws->solves_since_rebuild;
    maybe_recycle_workspace(ws);

    /* Do not discard usable complex solves just because MOSEK reports a
     * non-OPTIMAL solution status. In practice the realified complex SDP can
     * terminate with a feasible/unknown status while still providing a finite
     * primal objective. The problem is compact because every P_m,Q_m has an
     * explicit 0 <= X <= I constraint, so a finite primal objective is the best
     * available value and is far preferable to collapsing the entire column to 0.
     */
    if (r == MSK_RES_OK && isfinite((double)objective))
    {
        if (solsta_r != MSK_RES_OK || solsta != MSK_SOL_STA_OPTIMAL)
        {
            atomic_fetch_add(&diag_accepted_nonoptimal_solutions, 1);
        }
        return -(double)objective;
    }

    atomic_fetch_add(&diag_nonfinite_objectives, 1);
    return NAN;
}

double compute_gmn_mosek_real_8x8(const double *rho_real_row_major)
{
    if (rho_real_row_major == NULL)
    {
        return NAN;
    }

    GmnWorkspace *ws = workspace(0);
    if (ws == NULL)
    {
        atomic_fetch_add(&diag_workspace_failures, 1);
        return NAN;
    }

    fill_upper_coefficients(ws, rho_real_row_major, ws->coeff_p);
    partial_transpose_matrix_real(ws, rho_real_row_major, ws->masks[0], ws->rho_pt);
    fill_upper_coefficients(ws, ws->rho_pt, ws->coeff_q);

    return optimize_current_task(ws);
}

double compute_gmn_mosek_complex_8x8(const double *rho_ri_row_major)
{
    if (rho_ri_row_major == NULL)
    {
        return NAN;
    }

    GmnWorkspace *ws = workspace(1);
    if (ws == NULL)
    {
        atomic_fetch_add(&diag_workspace_failures, 1);
        return NAN;
    }

    realify_complex_8x8(rho_ri_row_major, 0.5, ws->work_matrix);
    fill_upper_coefficients(ws, ws->work_matrix, ws->coeff_p);

    partial_transpose_matrix_complex_ri(rho_ri_row_major, ws->masks[0], ws->rho_pt);
    realify_complex_8x8(ws->rho_pt, 0.5, ws->work_matrix);
    fill_upper_coefficients(ws, ws->work_matrix, ws->coeff_q);

    return optimize_current_task(ws);
}
