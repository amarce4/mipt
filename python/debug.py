from gnme import *

def level_3_inflation_score(
        rho: ComplexMatrix,
        verbose: bool = False,
        max_iters: int = 10_000,
        tol: float = 1e-4):
    
    rho = np.asarray(rho, dtype=np.complex128)

    dims: list
    d: int

    if rho.shape == (8, 8):
        dims = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    else:
        raise ValueError(f"unsupported matrix shape {rho.shape}")
    
    d = int(np.prod(dims))
    
    rho_dims = rho.shape
    rho_d = np.prod(rho_dims)

    t = cp.Variable(name="t")
    tau = cp.Variable((d, d), hermitian=True, name="tau")
    gamma = cp.Variable((d, d), hermitian=True, name="gamma")
    sigma = cp.Variable((d, d), hermitian=True, name="sigma")

    I = np.eye(rho_dims[0], dtype=np.complex128)
    rho_vec = rho.reshape(rho_d, order="F")
    I_vec = I.reshape(rho_d, order="F")

    constraints: list[cp.Constraint] = []

    constraints += [
        t >= 0,
        t <= 1,          # Important safeguard, especially near rho = I/8.
        tau >> 0,
        gamma >> 0,
        sigma >> 0,
        cp.trace(tau) == 1,
        cp.trace(gamma) == 1,
        cp.trace(sigma) == 1
    ]

    # White-noise mixture:
    # sigma_012 = t*rho + (1-t)*I/rho_dims[0]
    constraints.append(
        marginal_vec_expr(sigma, dims, [0, 1, 2])
        ==
        t * rho_vec + (1 - t) * I_vec / rho_dims[0]
    )

    # Marginal equality constraints.
    constraints += [
        # gamma_7801245 = tau_4501278
        marginal_vec_expr(gamma, dims, [7,8,0,1,2,4,5])
        == marginal_vec_expr(tau, dims, [4,5,0,1,2,7,8]),
        # tau_1245678 = sigma_1245678
        marginal_vec_expr(tau, dims, [1,2,4,5,6,7,8])
        == marginal_vec_expr(sigma, dims, [1,2,4,5,6,7,8]),
        # gamma_0123467 = tau_0123467
        marginal_vec_expr(gamma, dims, [0,1,2,3,4,6,7])
        == marginal_vec_expr(tau, dims, [0,1,2,3,4,6,7]),
        # tau_0134678 = sigma_0134678
        marginal_vec_expr(tau, dims, [0,1,3,4,6,7,8])
        == marginal_vec_expr(sigma, dims, [0,1,3,4,6,7,8]),
        # gamma_8012356 = tau_5012386
        marginal_vec_expr(gamma, dims, [8,0,1,2,3,5,6])
        == marginal_vec_expr(tau, dims, [5,0,1,2,3,8,6]),
        # tau_2350678 = sigma_2053678
        marginal_vec_expr(tau, dims, [2,3,5,0,6,7,8])
        == marginal_vec_expr(sigma, dims, [2,0,5,3,6,7,8])
    ]

    sigma_vec = cp.vec(sigma, order="F")

    # Copy-swap / permutation symmetry:
    constraints += [
        # gamma_345678012 = gamma
        marginal_vec_expr(gamma, dims, [3,4,5,6,7,8,0,1,2])
        == cp.vec(gamma, order="F"),
        # tau_345012678 = tau
        marginal_vec_expr(tau, dims, [3,4,5,0,1,2,6,7,8])
        == cp.vec(tau, order="F"),
        # sigma_345012678 = sigma
        marginal_vec_expr(sigma, dims, [3,4,5,0,1,2,6,7,8])
        == sigma_vec,
        # sigma_012678345 = sigma
        marginal_vec_expr(sigma, dims, [0,1,2,6,7,8,3,4,5])
        == sigma_vec
    ]

    # PPT constraints.
    tau_T678 = partial_transpose_expr_sparse(tau, dims, [6,7,8])
    sigma_T678 = partial_transpose_expr_sparse(sigma, dims, [6,7,8])

    gamma_0234567 = marginal_matrix_expr(gamma, dims, [0,2,3,4,5,6,7])
    gamma_0124567 = marginal_matrix_expr(gamma, dims, [0,1,2,4,5,6,7])
    gamma_1345678 = marginal_matrix_expr(gamma, dims, [1,3,4,5,6,7,8])
    gamma_1235678 = marginal_matrix_expr(gamma, dims, [1,2,3,5,6,7,8])
    gamma_2456780 = marginal_matrix_expr(gamma, dims, [2,4,5,6,7,8,0])
    gamma_2346780 = marginal_matrix_expr(gamma, dims, [2,3,4,6,7,8,0])

    tau_0234678 = marginal_matrix_expr(tau, dims, [0,2,3,4,6,7,8])
    tau_1345678 = marginal_matrix_expr(tau, dims, [1,3,4,5,6,7,8])
    tau_2450678 = marginal_matrix_expr(tau, dims, [2,4,5,0,6,7,8])

    partial_dims = [2,2,2,2,2,2,2]
    gamma_0234567_T0 = partial_transpose_expr_sparse(gamma_0234567, partial_dims, [0])
    gamma_0124567_T012 = partial_transpose_expr_sparse(gamma_0124567, partial_dims, [0,1,2])
    gamma_1345678_T1 = partial_transpose_expr_sparse(gamma_1345678, partial_dims, [0])
    gamma_1235678_T123 = partial_transpose_expr_sparse(gamma_1235678, partial_dims, [0,1,2])
    gamma_2456780_T2 = partial_transpose_expr_sparse(gamma_2456780, partial_dims, [0])
    gamma_2346780_T234 = partial_transpose_expr_sparse(gamma_2346780, partial_dims, [0,1,2])

    tau_0234678_T1 = partial_transpose_expr_sparse(tau_0234678, partial_dims, [0])
    tau_0234678_T234 = partial_transpose_expr_sparse(tau_0234678, partial_dims, [1,2,3])
    tau_1345678_T2 = partial_transpose_expr_sparse(tau_1345678, partial_dims, [0])
    tau_1345678_T345 = partial_transpose_expr_sparse(tau_1345678, partial_dims, [1,2,3])
    tau_2450678_T3 = partial_transpose_expr_sparse(tau_2450678, partial_dims, [0])
    tau_2450678_T450 = partial_transpose_expr_sparse(tau_2450678, partial_dims, [1,2,3])

    constraints += [
        (tau_T678 + tau_T678.H) / 2  >> 0,
        (sigma_T678 + sigma_T678.H) / 2 >> 0,
        (gamma_0234567_T0 + gamma_0234567_T0.H) / 2 >> 0,
        (gamma_0124567_T012 + gamma_0124567_T012.H) / 2 >> 0,
        (gamma_1345678_T1 + gamma_1345678_T1.H) / 2 >> 0,
        (gamma_1235678_T123 + gamma_1235678_T123.H) / 2 >> 0,
        (gamma_2456780_T2 + gamma_2456780_T2.H) / 2 >> 0,
        (gamma_2346780_T234 + gamma_2346780_T234.H) / 2 >> 0,
        (tau_0234678_T1 + tau_0234678_T1.H) / 2 >> 0,
        (tau_0234678_T234 + tau_0234678_T234.H) / 2 >> 0,
        (tau_1345678_T2 + tau_1345678_T2.H) / 2 >> 0,
        (tau_1345678_T345 + tau_1345678_T345.H) / 2 >> 0,
        (tau_2450678_T3 + tau_2450678_T3.H) / 2 >> 0,
        (tau_2450678_T450 + tau_2450678_T450.H) / 2 >> 0
    ]

    objective = cp.Maximize(cp.real(t))
    problem = cp.Problem(objective, constraints)

    if verbose:
        print("is_dcp:", problem.is_dcp())
        print("num constraints:", len(problem.constraints))

        print("Compiling problem data...")
        data, chain, inverse_data = problem.get_problem_data(cp.MOSEK)
        print("Compilation succeeded.")

        print("Solving...")

    problem.solve(
        solver=cp.SCS,
        verbose=verbose,
        warm_start=True,
        max_iters=max_iters,
        eps=tol
    )

    if verbose:
        print("Solve succeeded.")

    score = problem.value

    if (score >= 1):
        score = 1

    return score