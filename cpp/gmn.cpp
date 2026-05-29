#include "gmn.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <array>

#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

namespace
{
    constexpr int PARTIES = 3;
    constexpr int D = 8;

    std::shared_ptr<ndarray<int, 1>> nint(const std::vector<int> &x)
    {
        return new_array_ptr<int>(x);
    }

    std::shared_ptr<ndarray<int, 1>> idx(int i, int j)
    {
        return nint({i, j});
    }

    std::array<int, PARTIES> unpack_index(int x)
    {
        return {
            (x >> 2) & 1,
            (x >> 1) & 1,
            x & 1
        };
    }

    int pack_index(const std::array<int, PARTIES> &bits)
    {
        return (bits[0] << 2) | (bits[1] << 1) | bits[2];
    }

    std::pair<int, int> partial_transpose_source_index(
        int row,
        int col,
        int subsystem)
    {
        auto r = unpack_index(row);
        auto c = unpack_index(col);

        std::swap(r[subsystem], c[subsystem]);

        return {pack_index(r), pack_index(c)};
    }

    Matrix::t make_objective_matrix_from_row_major_real(const double *rho)
    {
        auto C = std::make_shared<ndarray<double, 2>>(shape(D, D));

        // Objective: trace(rho * W) = sum_{i,j} rho(i,j) W(j,i)
        // Expr::dot(C,W) = sum_{r,c} C(r,c) W(r,c)
        // Therefore C(r,c) = rho(c,r).
        for (int r = 0; r < D; ++r)
        {
            for (int c = 0; c < D; ++c)
            {
                (*C)(r, c) = rho[c * D + r];
            }
        }

        return Matrix::dense(C);
    }

    void add_partition_constraints(
        Model::t M,
        Variable::t W,
        int subsystem,
        const std::string &name)
    {
        Variable::t P =
            M->variable("P_" + name, Domain::inPSDCone(D));

        Variable::t Q =
            M->variable("Q_" + name, Domain::inPSDCone(D));

        Variable::t P_slack =
            M->variable("P_slack_" + name, Domain::inPSDCone(D));

        Variable::t Q_slack =
            M->variable("Q_slack_" + name, Domain::inPSDCone(D));

        for (int i = 0; i < D; ++i)
        {
            for (int j = 0; j < D; ++j)
            {
                const auto [qi, qj] =
                    partial_transpose_source_index(i, j, subsystem);

                M->constraint(
                    "w_eq_" + name + "_" +
                        std::to_string(i) + "_" + std::to_string(j),
                    Expr::sub(
                        W->index(idx(i, j)),
                        Expr::add(
                            P->index(idx(i, j)),
                            Q->index(idx(qi, qj)))),
                    Domain::equalsTo(0.0));

                const double eye_ij = (i == j) ? 1.0 : 0.0;

                M->constraint(
                    "p_upper_" + name + "_" +
                        std::to_string(i) + "_" + std::to_string(j),
                    Expr::add(
                        P->index(idx(i, j)),
                        P_slack->index(idx(i, j))),
                    Domain::equalsTo(eye_ij));

                M->constraint(
                    "q_upper_" + name + "_" +
                        std::to_string(i) + "_" + std::to_string(j),
                    Expr::add(
                        Q->index(idx(i, j)),
                        Q_slack->index(idx(i, j))),
                    Domain::equalsTo(eye_ij));
            }
        }
    }

    double compute_gmn_internal(const double *rho)
    {
        if (rho == nullptr)
        {
            throw std::invalid_argument("rho pointer is null.");
        }

        Model::t M = new Model("GMN_3Q");
        auto cleanup = finally([&]() { M->dispose(); });

        M->setSolverParam("log", 0);

        Variable::t W =
            M->variable("W", Set::make(D, D), Domain::unbounded());

        // tr(W) = 1 bad.
        // M->constraint(
        //     "trace_W",
        //     Expr::dot(Matrix::eye(D), W),
        //     Domain::equalsTo(1.0));

        add_partition_constraints(M, W, 0, "A");
        add_partition_constraints(M, W, 1, "B");
        add_partition_constraints(M, W, 2, "C");

        Matrix::t C = make_objective_matrix_from_row_major_real(rho);

        M->objective(
            "min_trace_rho_W",
            ObjectiveSense::Minimize,
            Expr::dot(C, W));

        M->solve();

        M->acceptedSolutionStatus(AccSolutionStatus::Optimal);

        return -M->primalObjValue();
    }
}

extern "C" double compute_gmn_mosek_real_8x8(
    const double *rho_real_row_major)
{
    try
    {
        return compute_gmn_internal(rho_real_row_major);
    }
    catch (...)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
}