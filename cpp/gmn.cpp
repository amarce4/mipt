#include "gmn.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>

#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

namespace
{
    constexpr int PARTIES = 3;
    constexpr int D = 8;
    constexpr int D2 = D * D;
    constexpr int PSD_MATRICES = 2 * PARTIES;
    constexpr int ANCHOR_SUBSYSTEM = 0;
    constexpr int UPPER_COUNT = D * (D + 1) / 2;

    struct Index2
    {
        int r;
        int c;
    };

    using Int1D = std::shared_ptr<ndarray<int, 1>>;
    using Int2D = std::shared_ptr<ndarray<int, 2>>;
    using Double1D = std::shared_ptr<ndarray<double, 1>>;

    constexpr int subsystem_mask(int subsystem) noexcept
    {
        // subsystem 0 -> A -> bit 2
        // subsystem 1 -> B -> bit 1
        // subsystem 2 -> C -> bit 0
        return 1 << (PARTIES - 1 - subsystem);
    }

    constexpr Index2 partial_transpose_source_index(
        int row,
        int col,
        int subsystem) noexcept
    {
        const int mask = subsystem_mask(subsystem);

        const int row_bit = row & mask;
        const int col_bit = col & mask;

        return {
            (row & ~mask) | col_bit,
            (col & ~mask) | row_bit
        };
    }

    const Int1D &matrix_shape()
    {
        static const Int1D s = new_array_ptr<int, 1>({D, D});
        return s;
    }

    const Matrix::t &identity_matrix()
    {
        static const Matrix::t I = Matrix::eye(D);
        return I;
    }

    const std::array<Int2D, PARTIES> &partial_transpose_pick_tables()
    {
        static const std::array<Int2D, PARTIES> tables = []()
        {
            std::array<Int2D, PARTIES> t{};

            for (int subsystem = 0; subsystem < PARTIES; ++subsystem)
            {
                auto pick = std::make_shared<ndarray<int, 2>>(shape(D2, 2));

                // The picked vector is reshaped row-major into the target
                // matrix. Entry target(r,c) receives source(qr,qc).
                for (int r = 0; r < D; ++r)
                {
                    for (int c = 0; c < D; ++c)
                    {
                        const int k = r * D + c;
                        const Index2 q = partial_transpose_source_index(
                            r,
                            c,
                            subsystem);

                        (*pick)(k, 0) = q.r;
                        (*pick)(k, 1) = q.c;
                    }
                }

                t[subsystem] = pick;
            }

            return t;
        }();

        return tables;
    }

    const Int2D &partial_transpose_pick_table(int subsystem)
    {
        return partial_transpose_pick_tables()[subsystem];
    }

    const Int2D &upper_triangle_pick_table()
    {
        static const Int2D pick = []()
        {
            auto p = std::make_shared<ndarray<int, 2>>(shape(UPPER_COUNT, 2));

            int k = 0;
            for (int r = 0; r < D; ++r)
            {
                for (int c = r; c < D; ++c)
                {
                    (*p)(k, 0) = r;
                    (*p)(k, 1) = c;
                    ++k;
                }
            }

            return p;
        }();

        return pick;
    }

    Variable::t psd_slice(Variable::t X, int k)
    {
        return X->slice(
                    new_array_ptr<int, 1>({k, 0, 0}),
                    new_array_ptr<int, 1>({k + 1, D, D}))
                ->reshape(matrix_shape());
    }

    Expression::t partial_transpose_expr(
        Variable::t Q,
        int subsystem)
    {
        return Expr::reshape(
            Q->pick(partial_transpose_pick_table(subsystem)),
            matrix_shape());
    }

    Expression::t witness_expr(
        Variable::t P,
        Variable::t Q,
        int subsystem)
    {
        return Expr::add(P, partial_transpose_expr(Q, subsystem));
    }

    Expression::t upper_triangle(Expression::t X)
    {
        return X->pick(upper_triangle_pick_table());
    }

    void add_psd_upper_bound(
        Model::t M,
        Variable::t X)
    {
        // X <= I in the Loewner order:
        //     I - X is positive semidefinite.
        M->constraint(
            Expr::sub(identity_matrix(), X),
            Domain::inPSDCone(D));
    }

    void apply_optional_solver_settings(Model::t M)
    {
        M->setSolverParam("log", 0);

        // Useful when you run many independent GMN solves in parallel.
        // Example:
        //     GMN_MOSEK_NUM_THREADS=1 ./mipt.exe
        //
        // If unset, MOSEK chooses its default thread behavior.
        const char *threads = std::getenv("GMN_MOSEK_NUM_THREADS");
        if (threads != nullptr && *threads != '\0')
        {
            char *end = nullptr;
            errno = 0;

            const long value = std::strtol(threads, &end, 10);

            if (errno == 0 &&
                end != threads &&
                *end == '\0' &&
                value > 0 &&
                value <= std::numeric_limits<int>::max())
            {
                M->setSolverParam("numThreads", static_cast<int>(value));
            }
        }
    }

    void fill_upper_objective_coefficients(
        const double *rho,
        Double1D coeffs)
    {
        int k = 0;
        for (int r = 0; r < D; ++r)
        {
            for (int c = r; c < D; ++c)
            {
                // Objective is trace(rho * W) = sum_{i,j} rho(i,j) W(j,i).
                // rho and W are real symmetric here. When using only the upper
                // triangle, off-diagonal coefficients must include both full
                // matrix entries.
                if (r == c)
                {
                    (*coeffs)(k) = rho[r * D + r];
                }
                else
                {
                    (*coeffs)(k) = rho[r * D + c] + rho[c * D + r];
                }

                ++k;
            }
        }
    }

    struct GmnWorkspace
    {
        Model::t M;
        Parameter::t objective_coeffs;
        Double1D coeff_values;

        GmnWorkspace()
            : M(new Model("GMN_3Q_cached")),
              objective_coeffs(nullptr),
              coeff_values(std::make_shared<ndarray<double, 1>>(shape(UPPER_COUNT)))
        {
            // Helps Fusion avoid re-evaluating repeated pick/reshape expressions.
            M->expressionCache(true);

            apply_optional_solver_settings(M);

            // One stacked variable containing:
            //     P_A, Q_A, P_B, Q_B, P_C, Q_C
            //
            // Each slice is an 8x8 symmetric PSD matrix.
            Variable::t X = M->variable(Domain::inPSDCone(D, PSD_MATRICES));

            std::array<Variable::t, PARTIES> P;
            std::array<Variable::t, PARTIES> Q;

            for (int subsystem = 0; subsystem < PARTIES; ++subsystem)
            {
                P[subsystem] = psd_slice(X, 2 * subsystem);
                Q[subsystem] = psd_slice(X, 2 * subsystem + 1);

                add_psd_upper_bound(M, P[subsystem]);
                add_psd_upper_bound(M, Q[subsystem]);
            }

            // Deliberately no tr(W) = 1 constraint.
            //
            // Instead of creating an explicit dense unbounded W variable and
            // imposing W = P_s + PT_s(Q_s) for all s, use subsystem A as the
            // anchor witness:
            //
            //     W := P_A + PT_A(Q_A)
            //
            // and impose equality with the B and C decompositions.
            Expression::t W_anchor = witness_expr(
                P[ANCHOR_SUBSYSTEM],
                Q[ANCHOR_SUBSYSTEM],
                ANCHOR_SUBSYSTEM);

            Expression::t W_anchor_ut = upper_triangle(W_anchor);

            // Auxiliary scalar copy of the upper triangle. This is intentional:
            // MOSEK Fusion cannot parametrize coefficients that sit directly on
            // semidefinite terms, but parameters are valid objective
            // coefficients for ordinary scalar variables. Reusing this model and
            // only changing the 36 objective coefficients avoids rebuilding the
            // Fusion model for every realization.
            Variable::t W_ut = M->variable(UPPER_COUNT);
            M->constraint(
                Expr::sub(W_ut, W_anchor_ut),
                Domain::equalsTo(0.0));

            for (int subsystem = 0; subsystem < PARTIES; ++subsystem)
            {
                if (subsystem == ANCHOR_SUBSYSTEM)
                {
                    continue;
                }

                Expression::t W_other = witness_expr(
                    P[subsystem],
                    Q[subsystem],
                    subsystem);

                // Since every decomposition is symmetric, upper-triangular
                // equality is sufficient and avoids duplicate scalar equalities.
                M->constraint(
                    Expr::sub(W_anchor_ut, upper_triangle(W_other)),
                    Domain::equalsTo(0.0));
            }

            objective_coeffs = M->parameter(UPPER_COUNT);
            M->objective(
                ObjectiveSense::Minimize,
                Expr::dot(objective_coeffs, W_ut));

            M->acceptedSolutionStatus(AccSolutionStatus::Optimal);
        }

        ~GmnWorkspace()
        {
            if (M != nullptr)
            {
                M->dispose();
            }
        }

        double solve(const double *rho)
        {
            if (rho == nullptr)
            {
                throw std::invalid_argument("rho pointer is null.");
            }

            fill_upper_objective_coefficients(rho, coeff_values);
            objective_coeffs->setValue(coeff_values);

            M->solve();

            return -M->primalObjValue();
        }
    };

    GmnWorkspace &workspace()
    {
        static thread_local GmnWorkspace ws;
        return ws;
    }
}

extern "C" double compute_gmn_mosek_real_8x8(
    const double *rho_real_row_major)
{
    try
    {
        return workspace().solve(rho_real_row_major);
    }
    catch (...)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
}
