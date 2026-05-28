import socket
import numpy as np
import cvxpy as cp
import scipy.sparse as sp

# -----------------------------
# your GMN function (same as before)
# -----------------------------
def partial_transpose_expr(X, dims, subsys):
    d = int(np.prod(dims))

    rows, cols, data = [], [], []

    for i in range(d):
        ii = np.unravel_index(i, dims)

        for j in range(d):
            jj = np.unravel_index(j, dims)

            ii2 = list(ii)
            jj2 = list(jj)

            for s in subsys:
                ii2[s], jj2[s] = jj2[s], ii2[s]

            ip = np.ravel_multi_index(ii2, dims)
            jp = np.ravel_multi_index(jj2, dims)

            src = i + j*d
            dst = ip + jp*d

            rows.append(dst)
            cols.append(src)
            data.append(1.0)

    T = sp.coo_matrix((data, (rows, cols)), shape=(d*d, d*d)).tocsr()

    vecX = cp.reshape(X, (d*d, 1), order='F')
    vecPT = T @ vecX

    return cp.reshape(vecPT, (d, d), order='F')


def gmn(rho_vec):
    parties = 3
    dims = [2]*parties
    d = 2**parties

    rho = np.array(rho_vec).reshape((d,d))

    W = cp.Variable((d,d), hermitian=True)

    constraints = []
    constraints.append(cp.trace(W) == 1)

    I = np.eye(d)

    partitions = {"A":[0], "B":[1], "C":[2]}

    for name, subsys in partitions.items():
        P = cp.Variable((d,d), hermitian=True)
        Q = cp.Variable((d,d), hermitian=True)

        QT = partial_transpose_expr(Q, dims, subsys)

        constraints += [
            W == P + QT,
            P >> 0,
            I - P >> 0,
            Q >> 0,
            I - Q >> 0
        ]

    objective = cp.Minimize(cp.real(cp.trace(rho @ W)))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, warm_start=True)

    return float(-problem.value)


# -----------------------------
# TCP SERVER
# -----------------------------
HOST = "127.0.0.1"
PORT = 50007

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print("GMN server running...")

while True:
    conn, addr = server.accept()
    data = conn.recv(1000000)

    if not data:
        continue

    # decode incoming matrix
    rho = np.frombuffer(data, dtype=np.float64)

    try:
        val = gmn(rho)
        conn.sendall(str(val).encode())
    except Exception as e:
        conn.sendall(b"nan")

    conn.close()