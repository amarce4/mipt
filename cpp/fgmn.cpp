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

namespace fgmn
{
    constexpr int PARTIES = 3;
    constexpr int D = 8;
    constexpr int D2 = D * D;
    constexpr int PSD_MATRICES = 2 * PARTIES;
    constexpr int ANCHOR_SUBSYSTEM = 0;
    constexpr int UPPER_COUNT = D * (D + 1) / 2;
    constexpr int DOUBLE_UPPER_COUNT = 2*D * (2*D + 1) / 2;
    constexpr int LOWER_COUNT = D*(D-1)/2;

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

    const Matrix::t &double_identity()
    {
        static const Matrix::t I = Matrix::eye(2*D);
        return I;
    }

    const Expression::t &double_identity_expr()
    {
        static const Expression::t I = Expr::constTerm(double_identity());
        return I;
    }

    const Int1D &real_pick_table()
    {
        static const Int1D table = []()
        {
            auto p = std::make_shared<ndarray<int, 1>>(shape(D2));

            int k = 0;
            for (int r = 0; r < D; ++r)
            {
                for (int c = 0; c < D; ++c)
                {
                    k=r*2*D+c;
                    (*p)(r*D+c) = k;
                }
            }

            return p;
        }();
        return table;
    }

        const Int1D &imag_pick_table()
    {
        static const Int1D table = []()
        {
            auto p = std::make_shared<ndarray<int, 1>>(shape(D2));

            int k = 0;
            for (int r = 0; r < D; ++r)
            {
                for (int c = 0; c < D; ++c)
                {
                    k=r*2*D+c+D;
                    (*p)(r*D+c) = k;
                }
            }

            return p;
        }();
        return table;
    }

    const std::array<Int2D, PARTIES> &partial_transpose_pick_tables()
    {
        static const std::array<Int2D, PARTIES> tables = []()
        {
            std::array<Int2D, PARTIES> t{};

            for (int subsystem = 0; subsystem < PARTIES; ++subsystem)
            {

                //std::printf("\nPartial transpose table, subsystem %d:", subsystem);
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
                        //std::printf("[%d,%d->%d,%d]",r,c,q.r,q.c);
                    }
                }

                t[subsystem] = pick;
                
            }
            //std::printf("\n");

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

    const Int2D &upper_triangle_complex_pick_table()
    {
        static const Int2D pick = []()
        {
            auto p = std::make_shared<ndarray<int, 2>>(shape(2*UPPER_COUNT, 2));

            int k = 0;
            for (int r = 0; r < D; ++r) //Real part
            {
                for (int c = r; c < D; ++c)
                {
                    (*p)(k, 0) = r; //Real part
                    (*p)(k, 1) = c;
                    ++k;
                    (*p)(k, 0) = r; //Imaginary part
                    (*p)(k, 1) = c+D;
                    ++k;
                }
            }
            return p;
        }();

        return pick;
    }

    //Takes a D*(D+1)/2 list of elements and converts to a D*D list of elements that can be reshaped into a matrix
    const Int1D &reverse_upper_triangle_pick_table()
    {
        static const Int1D pick = []()
        {
            auto p = std::make_shared<ndarray<int, 1>>(shape(D2));

            int k = 0;
            for (int r = 0; r < D; ++r)
            {
                for (int c = r; c < D; ++c)
                {
                    (*p)(r*D+c) = k;
                    (*p)(c*D+r) = k;
                    ++k;
                }
            }

            return p;
        }();

        return pick;
    }

    //Takes a D*(D-1)/2 list of elements and converts to a D*D list of elements that can be reshaped into the upper right triangular half of an antisymmetric matrix
    const Int1D &reverse_upper_triangle_antisymm_pick_table()
    {
        static const Int1D pick = []()
        {
            auto p = std::make_shared<ndarray<int, 1>>(shape(D2));

            int k = 0;
            for (int r = 0; r < D-1; ++r)
            {
                for (int c = r+1; c < D; ++c)
                {
                    (*p)(r*D+c) = k;
                    (*p)(c*D+r) = k;
                    ++k;
                }
                (*p)(r*D+r) = 0;
            }
            (*p)(D2-1) = 0;
            return p;
        }();

        return pick;
    }

    const std::array<Int2D, 2*PARTIES> &parity_violation_pick_tables()
    {
        //Picks out all elements (r,c) of a D by D matrix for which the parity of r and c differ on the bits corresponding to the given subsystem.
        
        static const std::array<Int2D, 2*PARTIES> tables = []()
        {
            std::array<Int2D, 2*PARTIES> t{};

            for (int subsystem = 0; subsystem < PARTIES; ++subsystem)
            {
                //if(subsystem==0) std::printf("\nParity violation table:");
                auto pick_violate = std::make_shared<ndarray<int, 2>>(shape(D2/2,2));
                auto pick_non_violate = std::make_shared<ndarray<int, 2>>(shape(D2/2,2));
                int count_violate = 0;
                int count_non_violate = 0;
                for (int r = 0; r < D; ++r)
                {
                    for (int c = 0; c < D; ++c)
                    {
                        const int k = r * D + c;
                        const int parity_r = (r & subsystem_mask(subsystem)) != 0;
                        const int parity_c = (c & subsystem_mask(subsystem)) != 0;
                        if(parity_r ^ parity_c){
                            (*pick_violate)(count_violate, 0) = r;
                            (*pick_violate)(count_violate, 1) = c;
                            //if(subsystem == 0) std::printf("[(%d,%d)->PV%d]",r,c,count_violate);
                            count_violate++;
                        }
                        else{
                            (*pick_non_violate)(count_non_violate, 0) = r;
                            (*pick_non_violate)(count_non_violate, 1) = c;
                            //if(subsystem == 0) std::printf("[(%d,%d)->PP%d]",r,c,count_non_violate);
                            count_non_violate++;
                        }
                    }
                }

                t[subsystem] = pick_violate;
                t[subsystem + PARTIES] = pick_non_violate;
                //if(subsystem==0) std::printf("\n");
            }

            return t;
        }();

        return tables;
    }
    const Int2D &parity_violation_pick_table(int subsystem, bool violate)
    {
        return parity_violation_pick_tables()[subsystem + (violate ? 0 : PARTIES)];
    }

    // Recombines a matrix that was separated into parity preserving and violating parts
    // Input is a 1D flattened list of entries where the parity preserving terms come first, followed by the parity violating terms.
    // Output is a reordering of the list so that all entries return to their original position upon reshaping the list to a matrix.
    const std::array<Int1D, PARTIES> &parity_recollection_pick_tables(){
        static const std::array<Int1D, PARTIES> tables = []()
        {
            std::array<Int1D, PARTIES> t{};

            for (int subsystem = 0; subsystem < PARTIES; ++subsystem)
            {
                //if(subsystem==0) std::printf("\nParity recollection table:");
                auto pick = std::make_shared<ndarray<int, 1>>(shape(D2));
                int count_violate = 0;
                int count_non_violate = 0;
                for (int r = 0; r < D; ++r)
                {
                    for (int c = 0; c < D; ++c)
                    {
                        const int k = r * D + c;
                        const int parity_r = (r & subsystem_mask(subsystem)) != 0;
                        const int parity_c = (c & subsystem_mask(subsystem)) != 0;
                        if(parity_r ^ parity_c){
                            (*pick)(k) = count_violate+D2/2;
                            //if(subsystem == 0) std::printf("[PV%d->%d,%d]",count_violate,r,c);
                            count_violate++;
                        }
                        else{
                            (*pick)(k) = count_non_violate;
                            //if(subsystem == 0) std::printf("[PP%d->%d,%d]",count_non_violate,r,c);
                            count_non_violate++;
                        }
                    }
                }
                //if(subsystem==0) std::printf("\n");

                t[subsystem] = pick;
            }

            return t;
        }();

        return tables;
    }
    const Int1D &parity_recollection_pick_table(int subsystem)
    {
        return parity_recollection_pick_tables()[subsystem];
    }

    void print_expression_shape(Expression::t X, std::string Xname="X", std::string endstring = "\n"){
        std::printf("%s shape:(",Xname.c_str());
        auto Xshape = (*X->getShape());
        for(int dim : Xshape) std::printf("%d ",dim);
        std::printf(")%s", endstring.c_str());
    }
    
    Variable::t psd_slice(Variable::t X, int k)
    {
        return X->slice(
                    new_array_ptr<int, 1>({k, 0, 0}),
                    new_array_ptr<int, 1>({k + 1, D, D}))
                ->reshape(matrix_shape());
    }

    Expression::t partial_transpose_expr(
        Expression::t Q,
        int subsystem)
    {
        return Expr::reshape(
            Q->pick(partial_transpose_pick_table(subsystem)),
            matrix_shape());
    }

    Expression::t partial_transpose_expr(
        Expression::t Q_real,
        Expression::t Q_imag,
        int subsystem)
    {
        Expression::t Q_real_pt = Expr::reshape(
            Q_real->pick(partial_transpose_pick_table(subsystem)),
            matrix_shape());
        Expression::t Q_imag_pt = Expr::reshape(
            Q_imag->pick(partial_transpose_pick_table(subsystem)),
            matrix_shape());
        return Expr::vstack(Expr::hstack(Q_real_pt, Expr::neg(Q_imag_pt)), Expr::hstack(Q_imag_pt, Q_real_pt));
    }

    Expression::t partial_transpose_expr(
        Variable::t Q,
        int subsystem)
    { return partial_transpose_expr(Expression::t(Q), subsystem);}

    Expression::t fermionic_partial_transpose_expr(
        Expression::t Q_real,
        Expression::t Q_imag,
        int subsystem)
    {
        // std::printf("\nApplying partial transpose on subsystem %d...", subsystem);
        Expression::t Q_real_pt = Expr::reshape(
            Q_real->pick(partial_transpose_pick_table(subsystem)),
            matrix_shape());
        Expression::t Q_imag_pt = Expr::reshape(
            Q_imag->pick(partial_transpose_pick_table(subsystem)),
            matrix_shape());
        // For fermionic systems, the partial transpose includes an additional factor of i if the local parity is violated.
        // This can be implemented by swapping the real and imaginary parts of the parts of Q that violate parity, 
        // and applying a sign flip to the new real part.
        
        // std::printf("Picking out parity preserving/violating parts...");
        // Pick out real and imaginary parity preserving and violating parts of Q separately so we can recombine them with the correct factors at the end.
        Expression::t Q_real_pp = Q_real_pt->pick(parity_violation_pick_table(subsystem, false));
        Expression::t Q_real_pv = Q_real_pt->pick(parity_violation_pick_table(subsystem, true));
        Expression::t Q_imag_pp = Q_imag_pt->pick(parity_violation_pick_table(subsystem, false));
        Expression::t Q_imag_pv = Q_imag_pt->pick(parity_violation_pick_table(subsystem, true));
        // print_expression_shape(Q_real_pp, "Real PP", "; ");
        // print_expression_shape(Q_real_pv, "Real PV", "; ");
        // print_expression_shape(Q_imag_pp, "Imag PP", "; ");
        // print_expression_shape(Q_imag_pv, "Imag PV", "\n");
        // std::printf("Recombining and phasing parts...");
        // Recombine the parts, applying the appropriate factors of i and -1 as needed. The parity preserving part is unchanged, the parity violating part has its real and imaginary parts swapped and the new real part gets a minus sign.
        Expression::t Q_real_recombined = Expr::vstack(Q_real_pp, Expr::neg(Q_imag_pv));
        Expression::t Q_imag_recombined = Expr::vstack(Q_imag_pp, Q_real_pv);
        //Expression::t Q_real_recombined = Expr::vstack(Q_real_pp, Q_real_pv);
        //Expression::t Q_imag_recombined = Expr::vstack(Q_imag_pp, Q_imag_pv);

        // print_expression_shape(Q_real_recombined, "Real ALL", "; ");
        // print_expression_shape(Q_imag_recombined, "Imag ALL", "\n");
        // std::printf("Reshaping real/imag matrices...");
        Expression::t Q_real_mat = Expr::reshape(Q_real_recombined->pick(parity_recollection_pick_table(subsystem)), matrix_shape());
        Expression::t Q_imag_mat = Expr::reshape(Q_imag_recombined->pick(parity_recollection_pick_table(subsystem)), matrix_shape());
        
        // std::printf("Restacking matrix...\n");
        // Block the real and imaginary parts together into a single matrix
        Expression::t Q_fpt = Expr::vstack(Expr::hstack(Q_real_mat, Expr::neg(Q_imag_mat)), Expr::hstack(Q_imag_mat, Q_real_mat));
        return Q_fpt;
    }

    Expression::t fermionic_partial_transpose_expr(
        Variable::t Q_real,
        Variable::t Q_imag,
        int subsystem)
    { return fermionic_partial_transpose_expr(Expression::t(Q_real), Expression::t(Q_imag), subsystem);}

    Expression::t witness_expr(
        Expression::t P,
        Expression::t Q,
        int subsystem)
    {
        return Expr::add(P, partial_transpose_expr(Q, subsystem));
    }

    Expression::t witness_expr(
        Variable::t P,
        Variable::t Q,
        int subsystem)
    { return Expr::add(P, partial_transpose_expr(Q, subsystem));}

    Expression::t witness_expr(
        Expression::t P_real,
        Expression::t P_imag,
        Expression::t Q_real,
        Expression::t Q_imag,
        int subsystem)
    {   
        auto Qrt = partial_transpose_expr(Q_real, subsystem);
        auto Qit = partial_transpose_expr(Q_imag, subsystem);
        auto Qt_complex = Expr::vstack(Expr::hstack(Qrt, Expr::neg(Qit)), Expr::hstack(Qit, Qrt));
        auto P_complex = Expr::vstack(Expr::hstack(P_real, Expr::neg(P_imag)), Expr::hstack(P_imag, P_real));
        return Expr::add(P_complex, Qt_complex);
    }

    Expression::t witness_expr(Variable::t P_real, Variable::t P_imag, Variable::t Q_real, Variable::t Q_imag, int subsystem)
    { return witness_expr(Expression::t(P_real), Expression::t(P_imag), Expression::t(Q_real), Expression::t(Q_imag), subsystem);}

    Expression::t fermionic_witness_expr(
        Expression::t P_real,
        Expression::t P_imag,
        Expression::t Q_real,
        Expression::t Q_imag,
        int subsystem)
    {   
        Expression::t Qt_complex = fermionic_partial_transpose_expr(Q_real, Q_imag, subsystem);
        Expression::t P_complex = Expr::vstack(Expr::hstack(P_real, Expr::neg(P_imag)), Expr::hstack(P_imag, P_real));
        return Expr::add(P_complex, Qt_complex);
    }

    Expression::t fermionic_witness_expr(Variable::t P_real, Variable::t P_imag, Variable::t Q_real, Variable::t Q_imag, int subsystem)
    { return fermionic_witness_expr(Expression::t(P_real), Expression::t(P_imag), Expression::t(Q_real), Expression::t(Q_imag), subsystem);}

    Expression::t upper_triangle_complex(Expression::t X)
    {
        return X->pick(upper_triangle_complex_pick_table());
    }

    Expression::t upper_triangle(Expression::t X)
    {
        return X->pick(upper_triangle_pick_table());
    }

    //Turn a D*(D+1)/2 sized variable in upper triangular form into a D*D matrix
    Expression::t reverse_upper_triangle(Expression::t X){
        auto Xpick = X->pick(reverse_upper_triangle_pick_table());
        //print_expression_shape(Expression::t(Xpick),"Xpick");
        return Expr::reshape(Xpick,matrix_shape());
    }
    Expression::t reverse_upper_triangle(Variable::t X){
        auto Xpick = X->pick(reverse_upper_triangle_pick_table());
        //print_expression_shape(Expression::t(Xpick),"Xpick");
        return Expr::reshape(Xpick,matrix_shape());
    }

    //Turn a D*(D-1)/2 sized variable in upper triangular form into a D*D antisymmetric matrix
    Expression::t reverse_upper_triangle_antisymm(Expression::t X){
        auto Xpick = X->pick(reverse_upper_triangle_antisymm_pick_table());
        auto Xmat = Expr::reshape(Xpick,matrix_shape());
        auto Xconstants =std::make_shared<ndarray<double, 2>>(shape(D,D));
        for(int r = 0; r < D; r++){
            for(int c = r+1; c < D; c++){
                (*Xconstants)(r,c) = 1.;
                (*Xconstants)(c,r) = -1.;
            }
            (*Xconstants)(r,r) = 0.;
        }
        //print_expression_shape(Expression::t(Xpick),"Xpick");
        //return Expr::sub(Xmat, Expr::transpose(Xmat));
        return Expr::mulElm(Xconstants,Xmat);
    }

    //Get a PSD slice from an upper triangular matrix
    Expression::t psd_slice_from_ut(Variable::t X, int k)
    {
        auto Xslice = Expr::flatten(X->slice(
                    new_array_ptr<int, 1>({k, 0}),
                    new_array_ptr<int, 1>({k + 1, UPPER_COUNT})));
        //print_expression_shape(Expression::t(Xslice),"Xslice");
        return reverse_upper_triangle(Xslice);
    }

    //Get an antisymmetric PSD slice from an upper triangular matrix
    Expression::t psd_slice_from_ut_antisymm(Variable::t X, int k)
    {
        auto Xslice = Expr::flatten(X->slice(
                    new_array_ptr<int, 1>({k, 0}),
                    new_array_ptr<int, 1>({k + 1, LOWER_COUNT})));
        //print_expression_shape(Expression::t(Xslice),"Xslice");
        return reverse_upper_triangle_antisymm(Xslice);
    }

    //Combine vectorized/upper triangular real and imaginary parts of X into a 2dim*2dim matrix
    // Expression::t complex_vec_M(Expression::t ReX, Expression::t ImX, std::size_t dim){
    //     Int1D combiner_pick_table; //Pick table that 
    // }

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

    void add_psd_upper_bound(
        Model::t M,
        Expression::t X)
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
                // If real: Objective is trace(rho * W) = sum_{i,j} rho(i,j) W(j,i).
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

    void fill_objective_coefficients_complex_direct(
        const double *rho,
        Double1D coeffs)
    {
        int k = 0;
        for (int r = 0; r < D; ++r)
        {
            for (int c = 0; c < D; ++c)
            {
                // Parameters for an objective that just tries to minimize ||rho-W||.
                (*coeffs)(r*2*D+c) = rho[2u*(r*D+c)]; //Real part (multiply by 0.5 because we're using the doubled realification of W)
                (*coeffs)(r*2*D+(c+D)) = rho[2u*(r*D+c)+1]; //Imag part (upper right block)
                (*coeffs)((r+D)*2*D+c) = -rho[2u*(r*D+c)+1]; //Imag part (lower right block) (negate because Im(W) is positive on the lower right block)
                (*coeffs)((r+D)*2*D+(c+D)) = rho[2u*(r*D+c)];
                ++k;
            }
        }
    }

    void fill_upper_objective_coefficients_complex(
        const double *rho,
        Double1D coeffs)
    {
        int k = 0;
        for (int r = 0; r < D; ++r)
        {
            for (int c = 0; c < D; ++c)
            {
                // If complex: Objective is Re(trace(rho * W)) = sum_{i,j} Re(rho(j,i)) Re(W(i,j)) - Im(rho(j,i)) Im(W(i,j)).
                // rho is Hermitian but no assumptions are made on W. When using only the upper
                // triangle, off-diagonal coefficients must include both full
                // matrix entries.
                (*coeffs)(r*2*D+c) = 0.5*rho[2u*(c*D+r)]; //Real part (multiply by 0.5 because we're using the doubled realification of W)
                (*coeffs)(r*2*D+(c+D)) = 0.5*rho[2u*(c*D+r)+1]; //Imag part (upper right block)
                (*coeffs)((r+D)*2*D+c) = -0.5*rho[2u*(c*D+r)+1]; //Imag part (lower right block) (negate because Im(W) is positive on the lower right block)
                (*coeffs)((r+D)*2*D+(c+D)) = 0.5*rho[2u*(c*D+r)];
                ++k;
            }
        }
    }

    struct GmnWorkspace
    {
        Model::t M;
        Parameter::t objective_coeffs;
        Double1D coeff_values;
        bool is_fermionic;
        bool is_complex;

        GmnWorkspace(bool complex_in = false,bool fermionic_in = false)
            : M(new Model("GMN_3Q_cached")),
              objective_coeffs(nullptr),
              is_complex(complex_in),
              is_fermionic(fermionic_in)
        {
            // Helps Fusion avoid re-evaluating repeated pick/reshape expressions.
            M->expressionCache(true);

            apply_optional_solver_settings(M);

            // One stacked variable containing:
            //     P_A, Q_A, P_B, Q_B, P_C, Q_C
            //

            // Each slice is an 8x8 symmetric PSD matrix.
            int num_psd = PSD_MATRICES;
            //if(is_complex) num_psd *= 2; //For complex mode, we need to double the size of each PSD matrix to capture the real and imaginary parts together and enforce the correct structure.
            
            int psds_per_subsystem = 2;
            //Matrix::t adjusted_identity = is_complex ? double_identity() : identity_matrix();

            std::array<Expression::t, PARTIES> P;
            std::array<Expression::t, PARTIES> Q;
            std::array<Expression::t, PARTIES> P_imag;
            std::array<Expression::t, PARTIES> Q_imag;

            if(is_complex){
                Variable::t X = M->variable("X",new_array_ptr<int,1>({num_psd*2,D,D}), Domain::unbounded()); //Collection of P and Q real and imaginary components
                //    Contains Preal_A, Qreal_A, Pimag_A, Qimag_A, Preal_B, Qreal_B etc.
                //    X is dense with no inherent symmetry/antisymmetry/PSD constraints.
                for (int subsystem = 0; subsystem < PARTIES; ++subsystem)
                {
                    P[subsystem] = psd_slice(X, psds_per_subsystem*subsystem*2);
                    Q[subsystem] = psd_slice(X, psds_per_subsystem*subsystem*2 + 1);
                    P_imag[subsystem] = psd_slice(X, psds_per_subsystem*subsystem*2 + 2);
                    Q_imag[subsystem] = psd_slice(X, psds_per_subsystem*subsystem*2 + 3);
                    // Imaginary components are used to construct the full realified PSD variables
                    // by stacking the real and imaginary parts in the [A -B; B A] pattern. 
                    Expression::t P_complex = Expr::vstack(Expr::hstack(P[subsystem], Expr::neg(P_imag[subsystem])), Expr::hstack(P_imag[subsystem], P[subsystem]));
                    Expression::t Q_complex = Expr::vstack(Expr::hstack(Q[subsystem], Expr::neg(Q_imag[subsystem])), Expr::hstack(Q_imag[subsystem], Q[subsystem]));
                    if (is_fermionic){
                        Expression::t Qshift = Expr::sub(Expr::mul(2,Q_complex), double_identity()); //2Q-I
                        // BigQ is the matrix [I, 2Q-I; (2Q-I)', I]. If ||2Q-I||<=1, BigQ needs to be PSD.
                        Expression::t BigQ = Expr::vstack(Expr::hstack(double_identity_expr(), Qshift), Expr::hstack(Expr::transpose(Qshift), double_identity_expr())); //The transpose here is actually the full realified conjugate transpose, since Qshift is symmetric in the real part and antisymmetric in the imaginary part.
                        
                        M->constraint(P_complex,Domain::inPSDCone(2*D)); //Real part is automatically taken by the constraint
                        // M->constraint(
                        //     Expr::sub(double_identity(), Expr::mul(0.1,P_complex)),
                        //     Domain::inPSDCone(2*D)); //Extremely weak upper bound on P, re-enable if Mosek complains about dual infeasibile/unbounded problem
                        M->constraint(BigQ,Domain::inPSDCone(4*D));
                    }
                    else{
                        M->constraint(P_complex, Domain::inPSDCone(2*D));
                        M->constraint(Q_complex, Domain::inPSDCone(2*D));
                        M->constraint(
                            Expr::sub(double_identity(), Q_complex),
                            Domain::inPSDCone(2*D));
                        // M->constraint(
                        //     Expr::sub(double_identity(), P_complex),
                        //     Domain::inPSDCone(2*D)); //Going to just use the renormalized version here (no P < I constraint)
                    }
                }
            }
            else{
                Variable::t X = M->variable(Domain::inPSDCone(D, num_psd));
                //Variable::t XP = M->variable("P",Domain::inPSDCone(D, PARTIES));
                //Variable::t XQ = M->variable("Q",Domain::inPSDCone(D, PARTIES));
                for (int subsystem = 0; subsystem < PARTIES; ++subsystem)
                {
                    P[subsystem] = psd_slice(X, psds_per_subsystem * subsystem);
                    Q[subsystem] = psd_slice(X, psds_per_subsystem * subsystem + 1);
                    //P[subsystem] = psd_slice(XP, subsystem);
                    //Q[subsystem] = psd_slice(XQ, subsystem);
                    add_psd_upper_bound(M, Q[subsystem]);
                }
            }

            
            // Instead of creating an explicit dense unbounded W variable and
            // imposing W = P_s + PT_s(Q_s) for all s, use subsystem A as the
            // anchor witness:
            //
            //     W := P_A + PT_A(Q_A)
            //
            // and impose equality with the B and C decompositions.
            
            
            //std::printf("\nBuilding witness...\n");
            Expression::t W_anchor;
            if(is_fermionic){
                W_anchor = fermionic_witness_expr(
                    P[ANCHOR_SUBSYSTEM],
                    P_imag[ANCHOR_SUBSYSTEM],
                    Q[ANCHOR_SUBSYSTEM],
                    Q_imag[ANCHOR_SUBSYSTEM],
                    ANCHOR_SUBSYSTEM);
            }
            else if(is_complex){
                W_anchor = witness_expr(
                    P[ANCHOR_SUBSYSTEM],
                    P_imag[ANCHOR_SUBSYSTEM],
                    Q[ANCHOR_SUBSYSTEM],
                    Q_imag[ANCHOR_SUBSYSTEM],
                    ANCHOR_SUBSYSTEM);
            }
            else{
                W_anchor = witness_expr(
                    P[ANCHOR_SUBSYSTEM],
                    Q[ANCHOR_SUBSYSTEM],
                    ANCHOR_SUBSYSTEM);
            }
            

            if(is_complex){ //Assume, for now, that the W matrix isn't symmetric or Hermitian in the complex case.
                Expression::t W_anchor_unravel = Expr::reshape(W_anchor, 4*D2);

                for (int subsystem = 0; subsystem < PARTIES; ++subsystem)
                {
                    if (subsystem == ANCHOR_SUBSYSTEM)
                    {
                        continue;
                    }
                    Expression::t W_other;
                    if(is_fermionic){
                        W_other = fermionic_witness_expr(
                            P[subsystem],
                            P_imag[subsystem],
                            Q[subsystem],
                            Q_imag[subsystem],
                            subsystem);
                    }
                    else{
                        W_other = witness_expr(
                            P[subsystem],
                            P_imag[subsystem],
                            Q[subsystem],
                            Q_imag[subsystem],
                            subsystem);
                    }
                    Expression::t W_other_unravel = Expr::reshape(W_other, 4*D2);
                    M->constraint(
                        Expr::sub(W_anchor_unravel, W_other_unravel),
                        Domain::equalsTo(0.0));
                }
                objective_coeffs = M->parameter(4*D2);

                M->objective(
                    ObjectiveSense::Minimize,
                    Expr::dot(objective_coeffs, W_anchor_unravel)); 

                M->acceptedSolutionStatus(AccSolutionStatus::Optimal);
                coeff_values = std::make_shared<ndarray<double, 1>>(shape(4*D2));
            }
            else{
                Expression::t W_anchor_ut;
                W_anchor_ut = upper_triangle(W_anchor);
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

                    objective_coeffs = M->parameter(UPPER_COUNT);
                    M->objective(
                        ObjectiveSense::Minimize,
                        Expr::dot(objective_coeffs, W_ut)); //Wouldn't this effectively double-count the diagonal entries of W? 
                        // Yes, but the fill_upper_objective_coefficients function accounts for this by only putting the original 
                        // rho coefficients on the diagonal entries of W_ut, and putting the sum of the symmetric rho coefficients 
                        // on the off-diagonal entries of W_ut, so that the dot product with objective_coeffs correctly computes the 
                        // trace of rho with the full witness matrix W.

                    M->acceptedSolutionStatus(AccSolutionStatus::Optimal);
                }
                coeff_values = std::make_shared<ndarray<double, 1>>(shape(UPPER_COUNT));
            }
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

            if(is_complex){
                fill_upper_objective_coefficients_complex(rho, coeff_values);
            }
            else{
                fill_upper_objective_coefficients(rho, coeff_values);
            }
            objective_coeffs->setValue(coeff_values);
            M->writeTask("mosek_task.ptf");

            M->solve();

            
            // Potential variable-printing templates for testing
            // if(M->hasVariable("Witness")){
            //     std::printf("Optimal witness: \n");
            //     auto Wvals = M->getVariable("Witness")->level();
            //     for(int wrow = 0; wrow < 2*D; wrow++){
            //         for(int wcol=0; wcol < 2*D; wcol++){
            //             std::printf(" %*.5f ", 8, (*Wvals)[wrow*2*D+wcol]);
            //             if(wcol == D-1) std::printf(" | ");
            //         }
            //         if(wrow == D-1) std::printf("\n--------------------------------------------------------------------------------------------------");
            //         std::printf("\n");
            //     }
            // }
            
            // if(M->hasVariable("X")){
            //     auto Xvals = M->getVariable("X")->level();
            //     int X_index = 0;
            //     for(int subsystem = 0; subsystem < PARTIES; subsystem++){
            //         std::printf("Optimal Preal%d|Pimag%d: \n",subsystem+1,subsystem+1);
            //         for(int wrow = 0; wrow < D; wrow++){
            //             for(int wcol=0; wcol < D; wcol++){
            //                 std::printf(" %*.5f ", 8, (*Xvals)[X_index+wcol]);
            //                 if(wcol == D-1) std::printf(" | ");
            //             }
            //             for(int wcol=0; wcol < D; wcol++){
            //                 std::printf(" %*.5f ", 8, (*Xvals)[X_index+wcol+2*D2]);
            //             }
            //             X_index+=D;
            //             std::printf("\n");
            //         }

            //         std::printf("Optimal Qreal%d|Qimag%d: \n",subsystem+1,subsystem+1);
            //         for(int wrow = 0; wrow < D; wrow++){
            //             for(int wcol=0; wcol < D; wcol++){
            //                 std::printf(" %*.5f ", 8, (*Xvals)[X_index+wcol]);
            //                 if(wcol == D-1) std::printf(" | ");
            //             }
            //             for(int wcol=0; wcol < D; wcol++){
            //                 std::printf(" %*.5f ", 8, (*Xvals)[X_index+wcol+2*D2]);
            //             }
            //             X_index+=D;
            //             std::printf("\n");
            //         }
            //         X_index += 2*D2;
            //     }
            // }
            return -M->primalObjValue();
        }
    };

    GmnWorkspace &workspace(bool is_complex = false, bool is_fermionic = false)
    {
        static thread_local GmnWorkspace ws(is_complex, is_fermionic);
        return ws;
    }
    GmnWorkspace &workspace_fermion()
    {
        static thread_local GmnWorkspace wsf(true, true);
        return wsf;
    }
    GmnWorkspace &workspace_complex()
    {
        static thread_local GmnWorkspace wsc(true, false);
        return wsc;
    }
    GmnWorkspace &workspace_real()
    {
        static thread_local GmnWorkspace wsr(false, false);
        return wsr;
    }
}

extern "C" double compute_fgmn_mosek_8x8_cpp( // Implement C linkage version later
    const double *rho_complex_row_major)
{
    try
    {
        return fgmn::workspace_fermion().solve(rho_complex_row_major);
    }
    catch (const std::exception& e)
    {
        std::cerr << "\nFound exception of type ";
        std::cerr << e.what() << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
    catch (...)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

extern "C" double compute_gmn_mosek_complex_8x8_cpp(
    const double *rho_complex_row_major)
{
    try
    {
        return fgmn::workspace_complex().solve(rho_complex_row_major);
    }
    catch (const std::exception& e)
    {
        std::cerr << "\nFound exception of type ";
        std::cerr << e.what() << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
    catch (...)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

extern "C" double compute_gmn_mosek_real_8x8_cpp(
    const double *rho_real_row_major)
{
    try
    {
        return fgmn::workspace_real().solve(rho_real_row_major);
    }
    catch (...)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
}
