#include "gmn.hpp"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mosek.h"

/*
 * GMN SDP implementation through the MOSEK Optimizer C API.
 *
 * This file intentionally contains no C++ and does not link against Fusion.
 * CUDA-Q/nvq++ may use libc++, whereas MOSEK Fusion on Linux is built with
 * GCC/libstdc++; calling the C Optimizer API avoids that ABI boundary.
 *
 * Model, for subsystem s in {A,B,C}:
 *   W = P_s + PT_s(Q_s)
 *   0 <= P_s <= I, 0 <= Q_s <= I
 *   minimize trace(rho W)
 *
 * Each 8x8 PSD upper-bound inequality uses a PSD slack:
 *   P_s + S^P_s = I,  Q_s + S^Q_s = I.
 *
 * The MOSEK task is cached per host thread. Each solve updates only the two
 * objective bar-matrix coefficients associated with P_A and Q_A.
 */

enum {
    PARTIES = 3,
    D = 8,
    UPPER_COUNT = D * (D + 1) / 2,
    PRIMARY_BARVARS = 2 * PARTIES,
    NUM_BARVARS = 2 * PRIMARY_BARVARS,
    UPPER_BOUND_CONS = PRIMARY_BARVARS * UPPER_COUNT,
    WITNESS_EQUALITY_CONS = (PARTIES - 1) * UPPER_COUNT,
    NUM_CONS = UPPER_BOUND_CONS + WITNESS_EQUALITY_CONS
};

enum {
    P_A = 0, Q_A = 1,
    P_B = 2, Q_B = 3,
    P_C = 4, Q_C = 5,
    S_P_A = 6, S_Q_A = 7,
    S_P_B = 8, S_Q_B = 9,
    S_P_C = 10, S_Q_C = 11
};

typedef struct {
    MSKenv_t env;
    MSKtask_t task;
    MSKint64t entry_basis[UPPER_COUNT];
} GmnWorkspace;

static _Thread_local GmnWorkspace *tls_ws = NULL;

static int subsystem_mask(int subsystem)
{
    /* A -> bit 2, B -> bit 1, C -> bit 0. */
    return 1 << (PARTIES - 1 - subsystem);
}

static void partial_transpose_source_index(int row, int col, int subsystem,
                                           int *src_row, int *src_col)
{
    const int mask = subsystem_mask(subsystem);
    const int row_bit = row & mask;
    const int col_bit = col & mask;
    *src_row = (row & ~mask) | col_bit;
    *src_col = (col & ~mask) | row_bit;
}

static int upper_index(int row, int col)
{
    int r = row;
    int c = col;
    if (r > c) {
        const int tmp = r;
        r = c;
        c = tmp;
    }

    /* Entries preceding row r plus offset c-r. */
    return r * D - (r * (r - 1)) / 2 + (c - r);
}

static MSKrescodee set_fixed_bound(MSKtask_t task, int con, double rhs)
{
    return MSK_putconbound(task, (MSKint32t)con, MSK_BK_FX, rhs, rhs);
}

static MSKrescodee append_entry_basis(GmnWorkspace *ws)
{
    MSKrescodee r = MSK_RES_OK;
    int row, col;

    for (row = 0; row < D && r == MSK_RES_OK; ++row) {
        for (col = row; col < D && r == MSK_RES_OK; ++col) {
            const int k = upper_index(row, col);
            /* MSK_appendsparsesymmat accepts lower-triangular locations. */
            const MSKint32t subi = (MSKint32t)col;
            const MSKint32t subj = (MSKint32t)row;
            /* <E_rc, X> = X_rc, including off-diagonal locations. */
            const MSKrealt value = (row == col) ? 1.0 : 0.5;
            r = MSK_appendsparsesymmat(ws->task, D, 1,
                                       &subi, &subj, &value,
                                       &ws->entry_basis[k]);
        }
    }
    return r;
}

static MSKrescodee add_upper_bounds(GmnWorkspace *ws, int *constraint_index)
{
    MSKrescodee r = MSK_RES_OK;
    const MSKrealt one = 1.0;
    int x, row, col;

    for (x = 0; x < PRIMARY_BARVARS && r == MSK_RES_OK; ++x) {
        const int slack = PRIMARY_BARVARS + x;
        for (row = 0; row < D && r == MSK_RES_OK; ++row) {
            for (col = row; col < D && r == MSK_RES_OK; ++col) {
                const int con = (*constraint_index)++;
                const int k = upper_index(row, col);
                const double rhs = (row == col) ? 1.0 : 0.0;

                r = MSK_putbaraij(ws->task, con, x, 1,
                                  &ws->entry_basis[k], &one);
                if (r == MSK_RES_OK) {
                    r = MSK_putbaraij(ws->task, con, slack, 1,
                                      &ws->entry_basis[k], &one);
                }
                if (r == MSK_RES_OK) {
                    r = set_fixed_bound(ws->task, con, rhs);
                }
            }
        }
    }
    return r;
}

static MSKrescodee add_witness_equalities(GmnWorkspace *ws, int *constraint_index)
{
    MSKrescodee r = MSK_RES_OK;
    const MSKrealt plus_one = 1.0;
    const MSKrealt minus_one = -1.0;
    int subsystem, row, col;

    for (subsystem = 1; subsystem < PARTIES && r == MSK_RES_OK; ++subsystem) {
        const int p_other = 2 * subsystem;
        const int q_other = p_other + 1;

        for (row = 0; row < D && r == MSK_RES_OK; ++row) {
            for (col = row; col < D && r == MSK_RES_OK; ++col) {
                const int con = (*constraint_index)++;
                const int target = upper_index(row, col);
                int src_a_row, src_a_col, src_s_row, src_s_col;
                int source_a, source_s;

                partial_transpose_source_index(row, col, 0,
                                               &src_a_row, &src_a_col);
                partial_transpose_source_index(row, col, subsystem,
                                               &src_s_row, &src_s_col);
                source_a = upper_index(src_a_row, src_a_col);
                source_s = upper_index(src_s_row, src_s_col);

                /* P_A + PT_A(Q_A) - P_s - PT_s(Q_s) = 0. */
                r = MSK_putbaraij(ws->task, con, P_A, 1,
                                  &ws->entry_basis[target], &plus_one);
                if (r == MSK_RES_OK) {
                    r = MSK_putbaraij(ws->task, con, Q_A, 1,
                                      &ws->entry_basis[source_a], &plus_one);
                }
                if (r == MSK_RES_OK) {
                    r = MSK_putbaraij(ws->task, con, p_other, 1,
                                      &ws->entry_basis[target], &minus_one);
                }
                if (r == MSK_RES_OK) {
                    r = MSK_putbaraij(ws->task, con, q_other, 1,
                                      &ws->entry_basis[source_s], &minus_one);
                }
                if (r == MSK_RES_OK) {
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
    if (s != NULL && *s != '\0') {
        char *end = NULL;
        long value;
        errno = 0;
        value = strtol(s, &end, 10);
        if (errno == 0 && end != s && *end == '\0' && value > 0 && value <= INT32_MAX) {
            (void)MSK_putintparam(task, MSK_IPAR_NUM_THREADS, (MSKint32t)value);
        }
    }
}

static GmnWorkspace *create_workspace(void)
{
    GmnWorkspace *ws = (GmnWorkspace *)calloc(1, sizeof(*ws));
    MSKrescodee r = MSK_RES_OK;
    MSKint32t dim[NUM_BARVARS];
    int i;
    int con = 0;

    if (ws == NULL) {
        return NULL;
    }

    for (i = 0; i < NUM_BARVARS; ++i) {
        dim[i] = D;
    }

    r = MSK_makeenv(&ws->env, NULL);
    if (r == MSK_RES_OK) {
        r = MSK_maketask(ws->env, NUM_CONS, 0, &ws->task);
    }
    if (r == MSK_RES_OK) {
        r = MSK_appendcons(ws->task, NUM_CONS);
    }
    if (r == MSK_RES_OK) {
        r = MSK_appendbarvars(ws->task, NUM_BARVARS, dim);
    }
    if (r == MSK_RES_OK) {
        r = MSK_putobjsense(ws->task, MSK_OBJECTIVE_SENSE_MINIMIZE);
    }
    if (r == MSK_RES_OK) {
        r = MSK_putintparam(ws->task, MSK_IPAR_LOG, 0);
    }
    if (r == MSK_RES_OK) {
        set_optional_threads(ws->task);
        r = append_entry_basis(ws);
    }
    if (r == MSK_RES_OK) {
        r = add_upper_bounds(ws, &con);
    }
    if (r == MSK_RES_OK) {
        r = add_witness_equalities(ws, &con);
    }
    if (r == MSK_RES_OK && con != NUM_CONS) {
        r = MSK_RES_ERR_INTERNAL;
    }

    if (r != MSK_RES_OK) {
        if (ws->task != NULL) {
            MSK_deletetask(&ws->task);
        }
        if (ws->env != NULL) {
            MSK_deleteenv(&ws->env);
        }
        free(ws);
        return NULL;
    }

    return ws;
}

static GmnWorkspace *workspace(void)
{
    if (tls_ws == NULL) {
        tls_ws = create_workspace();
    }
    return tls_ws;
}

static void fill_upper_coefficients(const double *matrix, MSKrealt out[UPPER_COUNT])
{
    int row, col;
    for (row = 0; row < D; ++row) {
        for (col = row; col < D; ++col) {
            const int k = upper_index(row, col);
            out[k] = (row == col)
                ? matrix[row * D + row]
                : matrix[row * D + col] + matrix[col * D + row];
        }
    }
}

static void partial_transpose_matrix(const double *in, int subsystem, double out[D * D])
{
    int row, col;
    for (row = 0; row < D; ++row) {
        for (col = 0; col < D; ++col) {
            int src_row, src_col;
            partial_transpose_source_index(row, col, subsystem, &src_row, &src_col);
            out[row * D + col] = in[src_row * D + src_col];
        }
    }
}

double compute_gmn_mosek_real_8x8(const double *rho_real_row_major)
{
    GmnWorkspace *ws;
    MSKrescodee r;
    MSKrescodee trm = MSK_RES_OK;
    MSKsolstae solsta;
    MSKrealt objective = 0.0;
    MSKrealt coeff_p[UPPER_COUNT];
    MSKrealt coeff_q[UPPER_COUNT];
    double rho_pt_a[D * D];

    if (rho_real_row_major == NULL) {
        return NAN;
    }

    ws = workspace();
    if (ws == NULL) {
        return NAN;
    }

    fill_upper_coefficients(rho_real_row_major, coeff_p);
    partial_transpose_matrix(rho_real_row_major, 0, rho_pt_a);
    fill_upper_coefficients(rho_pt_a, coeff_q);

    /* W = P_A + PT_A(Q_A): trace(rho W) = <rho,P_A> + <PT_A(rho),Q_A>. */
    r = MSK_putbarcj(ws->task, P_A, UPPER_COUNT, ws->entry_basis, coeff_p);
    if (r == MSK_RES_OK) {
        r = MSK_putbarcj(ws->task, Q_A, UPPER_COUNT, ws->entry_basis, coeff_q);
    }
    if (r == MSK_RES_OK) {
        r = MSK_optimizetrm(ws->task, &trm);
    }
    if (r != MSK_RES_OK) {
        return NAN;
    }

    r = MSK_getsolsta(ws->task, MSK_SOL_ITR, &solsta);
    if (r != MSK_RES_OK || solsta != MSK_SOL_STA_OPTIMAL) {
        return NAN;
    }

    r = MSK_getprimalobj(ws->task, MSK_SOL_ITR, &objective);
    return (r == MSK_RES_OK) ? -(double)objective : NAN;
}
