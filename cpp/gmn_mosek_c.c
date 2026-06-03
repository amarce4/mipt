#include "gmn.hpp"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "mosek.h"

/*
 * Fully decomposable real-valued GMN witness SDP through the MOSEK C API.
 *
 * For each unique bipartition mask m:
 *     W = P_m + PT_m(Q_m)
 *     0 <= P_m <= I, 0 <= Q_m <= I
 * and the objective is:
 *     minimize Tr(rho W).
 *
 * The returned quantity is -min Tr(rho W), matching the sign convention in
 * the previous implementation.
 *
 * The retained system is fixed at three qubits; its three unique
 * bipartitions are represented by masks 1, 2, and 3.
 *
 * This implementation is deliberately C-only to avoid the CUDA-Q libc++ /
 * MOSEK Fusion libstdc++ ABI boundary.
 */

typedef struct
{
    int d;
    int cuts;
    int upper_count;
    int primary_barvars;
    int num_barvars;
    int upper_bound_cons;
    int witness_equality_cons;
    int num_cons;
    int *masks;
    MSKint64t *entry_basis;
    MSKenv_t env;
    MSKtask_t task;
} GmnWorkspace;

static _Thread_local GmnWorkspace *tls_ws = NULL;

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

                partial_transpose_source_index(row, col, anchor_mask,
                                               &src_anchor_row,
                                               &src_anchor_col);
                partial_transpose_source_index(row, col, other_mask,
                                               &src_other_row,
                                               &src_other_col);

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

static GmnWorkspace *create_workspace(void)
{
    GmnWorkspace *ws = (GmnWorkspace *)calloc(1, sizeof(*ws));
    if (ws == NULL)
    {
        return NULL;
    }

    ws->d = 8;
    ws->cuts = 3;
    ws->upper_count = ws->d * (ws->d + 1) / 2;
    ws->primary_barvars = 2 * ws->cuts;
    ws->num_barvars = 2 * ws->primary_barvars;
    ws->upper_bound_cons = ws->primary_barvars * ws->upper_count;
    ws->witness_equality_cons = (ws->cuts - 1) * ws->upper_count;
    ws->num_cons = ws->upper_bound_cons + ws->witness_equality_cons;

    ws->masks = (int *)calloc((size_t)ws->cuts, sizeof(int));
    ws->entry_basis =
        (MSKint64t *)calloc((size_t)ws->upper_count, sizeof(MSKint64t));
    if (ws->masks == NULL || ws->entry_basis == NULL)
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

static GmnWorkspace *workspace(void)
{
    if (tls_ws == NULL)
    {
        tls_ws = create_workspace();
    }
    return tls_ws;
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

static void partial_transpose_matrix(const GmnWorkspace *ws,
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

double compute_gmn_mosek_real_8x8(const double *rho_real_row_major)
{
    if (rho_real_row_major == NULL)
    {
        return NAN;
    }

    GmnWorkspace *ws = workspace();
    if (ws == NULL)
    {
        return NAN;
    }

    MSKrealt *coeff_p =
        (MSKrealt *)calloc((size_t)ws->upper_count, sizeof(MSKrealt));
    MSKrealt *coeff_q =
        (MSKrealt *)calloc((size_t)ws->upper_count, sizeof(MSKrealt));
    double *rho_pt =
        (double *)calloc((size_t)ws->d * (size_t)ws->d, sizeof(double));
    if (coeff_p == NULL || coeff_q == NULL || rho_pt == NULL)
    {
        free(coeff_p);
        free(coeff_q);
        free(rho_pt);
        return NAN;
    }

    fill_upper_coefficients(ws, rho_real_row_major, coeff_p);
    partial_transpose_matrix(ws, rho_real_row_major, ws->masks[0], rho_pt);
    fill_upper_coefficients(ws, rho_pt, coeff_q);

    MSKrescodee r =
        MSK_putbarcj(ws->task, 0, ws->upper_count, ws->entry_basis, coeff_p);
    if (r == MSK_RES_OK)
    {
        r = MSK_putbarcj(ws->task, 1, ws->upper_count,
                         ws->entry_basis, coeff_q);
    }

    MSKrescodee trm = MSK_RES_OK;
    if (r == MSK_RES_OK)
    {
        r = MSK_optimizetrm(ws->task, &trm);
    }

    free(coeff_p);
    free(coeff_q);
    free(rho_pt);

    if (r != MSK_RES_OK)
    {
        return NAN;
    }

    MSKsolstae solsta;
    r = MSK_getsolsta(ws->task, MSK_SOL_ITR, &solsta);
    if (r != MSK_RES_OK || solsta != MSK_SOL_STA_OPTIMAL)
    {
        return NAN;
    }

    MSKrealt objective = 0.0;
    r = MSK_getprimalobj(ws->task, MSK_SOL_ITR, &objective);
    return (r == MSK_RES_OK) ? -(double)objective : NAN;
}

