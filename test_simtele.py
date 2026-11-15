import numpy as np

def jacobi(A, b, tol=1e-8, max_iter=100):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    x = np.zeros(n, dtype=float)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, True, k
        x = x_new
    return x, False, max_iter

def gauss_seidel(A, b, tol=1e-8, max_iter=100):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    x = np.zeros(n, dtype=float)
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, True, k
    return x, False, max_iter

def lu_solve(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    L = np.eye(n, dtype=float)
    U = A.copy()
    for k in range(n - 1):
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x, True, 1

def cg_solve(A, b, tol=1e-8, max_iter=100):
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros_like(b, dtype=float)
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    for k in range(1, max_iter + 1):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            return x, True, k
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, False, max_iter

# Pruebas automatizadas de todos los escenarios de la Semana 1
x_exact = np.array([1.0, 2.0, 3.0, 4.0])

escenarios = [
    ("NORMAL", np.array([[10., 2., 1., 0.], [2., 12., 2., 1.], [1., 2., 15., 3.], [0., 1., 3., 14.]])),
    ("ESTRÉS", np.array([[4.1, 2.0, 1.0, 1.0], [2.0, 4.2, 1.1, 1.0], [1.0, 1.1, 4.3, 2.0], [1.0, 1.0, 2.0, 4.4]])),
    ("MAL CONDICIONADO", np.array([[1.0 / (i + j + 1) for j in range(4)] for i in range(4)]))
]

for nombre_esc, A in escenarios:
    b = A @ x_exact
    print(f"\n--- PRUEBAS ESCENARIO {nombre_esc} (cond = {np.linalg.cond(A, 2):,.2f}) ---")
    for nombre, func in [("Jacobi", jacobi), ("Gauss-Seidel", gauss_seidel), ("LU Directo", lu_solve), ("Gradiente Conjugado", cg_solve)]:
        x_ap, conv, iters = func(A, b)
        err = np.linalg.norm(x_ap - x_exact, 2)
        err_str = f"{err:.2e}" if err < 1e6 else "Diverge"
        print(f"Método: {nombre:19} | Convergió: {str(conv):5} | Iteraciones: {iters:3} | Error Abs: {err_str}")