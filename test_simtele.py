"""
SIMTELE-RISK - Suite de Validación Numérica y Diagnóstico de Fallos (Semana 2)
Autores: Miguel Angel Benito Peñaloza & Daner Gabriel Huancollo Gaile
Descripción:
  Valida los 4 módulos numéricos (EDO, Raíces, Integración, Sistemas Lineales e Interpolación)
  bajo los 3 escenarios operativos (NORMAL, ESTRES, MAL_CONDICIONADO), verificando la
  consistencia con la aplicación web (index.html) y registrando los fallos numéricos detectados.
"""

import math
import numpy as np


# ==============================================================================
# 1. ALGORITMOS NUMÉRICOS (HOMÓLOGOS A index.html)
# ==============================================================================
def solve_bisection(f, a, b, tol=1e-7, max_iter=60):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return {"root": np.nan, "iter": 0, "error": np.nan, "status": "Fallo: Sin cambio de signo"}
    c = a
    prev_c = a
    for i in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        fc = f(c)
        err = abs(b - a) if i == 1 else abs(c - prev_c)
        if abs(fc) < tol or err < tol:
            return {"root": c, "iter": i, "error": err, "status": "Convergió"}
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        prev_c = c
    return {"root": c, "iter": max_iter, "error": abs(b - a), "status": "Máx. iteraciones"}


def solve_newton_raphson(f, df, x0, tol=1e-7, max_iter=50):
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            return {"root": np.nan, "iter": i, "error": np.nan, "status": "Fallo: Derivada cercana a cero"}
        x_next = x - fx / dfx
        err = abs(x_next - x)
        if not np.isfinite(x_next) or abs(x_next) > 1e6:
            return {"root": np.nan, "iter": i, "error": np.nan, "status": "Fallo: Divergencia"}
        if err < tol:
            return {"root": x_next, "iter": i, "error": err, "status": "Convergió"}
        x = x_next
    return {"root": x, "iter": max_iter, "error": abs(f(x)), "status": "No convergió"}


def solve_secant(f, x0, x1, tol=1e-7, max_iter=50):
    a, b = x0, x1
    for i in range(1, max_iter + 1):
        fa, fb = f(a), f(b)
        if abs(fb - fa) < 1e-13:
            return {"root": np.nan, "iter": i, "error": np.nan, "status": "Fallo: Pendiente nula"}
        c = b - fb * (b - a) / (fb - fa)
        err = abs(c - b)
        if not np.isfinite(c) or abs(c) > 1e6:
            return {"root": np.nan, "iter": i, "error": np.nan, "status": "Fallo: Divergencia"}
        if err < tol:
            return {"root": c, "iter": i, "error": err, "status": "Convergió"}
        a, b = b, c
    return {"root": b, "iter": max_iter, "error": abs(b - a), "status": "Máx. iteraciones"}


def integrate_trapezoid(f, a, b, n=16):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b)) + sum(f(a + i * h) for i in range(1, n))
    return s * h


def integrate_simpson(f, a, b, n=16):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (2 if i % 2 == 0 else 4) * f(a + i * h)
    return (s * h) / 3.0


def integrate_romberg(f, a, b, levels=4):
    R = np.zeros((levels, levels))
    for i in range(levels):
        R[i, 0] = integrate_trapezoid(f, a, b, 2 ** i)
    for j in range(1, levels):
        for i in range(j, levels):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4 ** j - 1)
    return float(R[levels - 1, levels - 1])


def solve_ode_1d(dydt, t0, y0, t_end, h):
    steps = max(1, round((t_end - t0) / h))
    dt = (t_end - t0) / steps
    t_arr = [t0]
    y_rk2 = [y0]
    y_rk4 = [y0]
    y2, y4, t = y0, y0, t0
    for _ in range(steps):
        k1_2 = dydt(t, y2)
        k2_2 = dydt(t + dt, y2 + dt * k1_2)
        y2 = y2 + 0.5 * dt * (k1_2 + k2_2)

        k1 = dydt(t, y4)
        k2 = dydt(t + 0.5 * dt, y4 + 0.5 * dt * k1)
        k3 = dydt(t + 0.5 * dt, y4 + 0.5 * dt * k2)
        k4 = dydt(t + dt, y4 + dt * k3)
        y4 = y4 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        t += dt
        t_arr.append(t)
        y_rk2.append(y2)
        y_rk4.append(y4)
    return np.array(t_arr), np.array(y_rk2), np.array(y_rk4)


def solve_ode_system_2d(f1, f2, t0, x0, v0, t_end, h):
    steps = max(1, round((t_end - t0) / h))
    dt = (t_end - t0) / steps
    t_arr = [t0]
    x_rk2, v_rk2 = [x0], [v0]
    x_rk4, v_rk4 = [x0], [v0]
    x2, v2, x4, v4, t = x0, v0, x0, v0, t0
    for _ in range(steps):
        kx1, kv1 = f1(t, x2, v2), f2(t, x2, v2)
        kx2, kv2 = f1(t + dt, x2 + dt * kx1, v2 + dt * kv1), f2(t + dt, x2 + dt * kx1, v2 + dt * kv1)
        x2 += 0.5 * dt * (kx1 + kx2)
        v2 += 0.5 * dt * (kv1 + kv2)

        k1x, k1v = f1(t, x4, v4), f2(t, x4, v4)
        k2x, k2v = f1(t + 0.5 * dt, x4 + 0.5 * dt * k1x, v4 + 0.5 * dt * k1v), f2(t + 0.5 * dt, x4 + 0.5 * dt * k1x, v4 + 0.5 * dt * k1v)
        k3x, k3v = f1(t + 0.5 * dt, x4 + 0.5 * dt * k2x, v4 + 0.5 * dt * k2v), f2(t + 0.5 * dt, x4 + 0.5 * dt * k2x, v4 + 0.5 * dt * k2v)
        k4x, k4v = f1(t + dt, x4 + dt * k3x, v4 + dt * k3v), f2(t + dt, x4 + dt * k3x, v4 + dt * k3v)
        x4 += (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v4 += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

        t += dt
        t_arr.append(t)
        x_rk2.append(x2)
        x_rk4.append(x4)
    return np.array(t_arr), np.array(x_rk2), np.array(x_rk4)


def solve_linear_systems(A, b, tol=1e-6, max_iter=80):
    n = len(b)
    x_lu = np.linalg.solve(A, b)

    # Jacobi
    x_j = np.zeros(n)
    err_j = []
    for _ in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * x_j[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        diff = np.max(np.abs(x_new - x_j))
        err_j.append(diff)
        x_j = x_new
        if diff < tol or diff > 1e8 or np.isnan(diff):
            break

    # Gauss-Seidel
    x_gs = np.zeros(n)
    err_gs = []
    for _ in range(max_iter):
        x_old = x_gs.copy()
        for i in range(n):
            s = sum(A[i, j] * x_gs[j] for j in range(n) if j != i)
            x_gs[i] = (b[i] - s) / A[i, i]
        diff = np.max(np.abs(x_gs - x_old))
        err_gs.append(diff)
        if diff < tol or diff > 1e8 or np.isnan(diff):
            break

    # Gradiente Conjugado
    x_cg = np.zeros(n)
    r = b - A @ x_cg
    p = r.copy()
    rsold = float(r @ r)
    err_cg = []
    for _ in range(max_iter):
        Ap = A @ p
        pAp = float(p @ Ap)
        if abs(pAp) < 1e-14:
            break
        alpha = rsold / pAp
        x_cg = x_cg + alpha * p
        r = r - alpha * Ap
        rsnew = float(r @ r)
        err_cg.append(math.sqrt(rsnew))
        if math.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x_lu, x_j, x_gs, x_cg, err_j, err_gs, err_cg


def eval_lagrange(xs, ys, x):
    total = 0.0
    n = len(xs)
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if i != j:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        total += term
    return total


def eval_natural_cubic_spline(xs, ys, eval_x):
    n = len(xs) - 1
    h = np.diff(xs)
    alpha = np.zeros(n)
    for i in range(1, n):
        alpha[i] = (3.0 / h[i]) * (ys[i + 1] - ys[i]) - (3.0 / h[i - 1]) * (ys[i] - ys[i - 1])
    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    for i in range(1, n):
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    out = []
    for x in eval_x:
        idx = n - 1
        for k in range(n):
            if x <= xs[k + 1]:
                idx = k
                break
        dx = x - xs[idx]
        out.append(ys[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3)
    return np.array(out)


# ==============================================================================
# 2. EJECUCIÓN DE SUITE DE VALIDACIÓN (4 MÓDULOS x 3 ESCENARIOS)
# ==============================================================================
def run_full_validation():
    print("=" * 84)
    print("SIMTELE-RISK | REPORTE DE VALIDACIÓN NUMÉRICA AUTOMATIZADA (SEMANA 2)")
    print("=" * 84)

    # --- MÓDULO 1 ---
    print("\n[MÓDULO 1] FRENADO DE EMERGENCIA (EDO, RAÍCES E INTEGRACIÓN)")
    m1_scenarios = {
        "NORMAL": (5.0, 1200, 380, 0.25),
        "ESTRES": (7.5, 1850, 290, 0.25),
        "MAL_CONDICIONADO": (8.5, 2100, 180, 1.80),
    }
    for name, (v0, m, cf, h) in m1_scenarios.items():
        k = cf / m
        v_th = 0.05
        v_exact = lambda t: v0 * math.exp(-k * t)
        dv_exact = lambda t: -k * v0 * math.exp(-k * t)
        t_stop_exact = -math.log(v_th / v0) / k
        d_stop_exact = (v0 / k) * (1.0 - math.exp(-k * t_stop_exact))

        t_arr, y_rk2, y_rk4 = solve_ode_1d(lambda t, v: -k * v, 0.0, v0, math.ceil(t_stop_exact + 2), h)
        v_true_arr = np.array([v_exact(t) for t in t_arr])
        err_rk2 = np.max(np.abs(y_rk2 - v_true_arr))
        err_rk4 = np.max(np.abs(y_rk4 - v_true_arr))

        f_root = lambda t: v_exact(t) - v_th
        bis = solve_bisection(f_root, 0.0, t_stop_exact * 2.2)
        x0_new = t_stop_exact * 4.5 if name == "MAL_CONDICIONADO" else 1.0
        newt = solve_newton_raphson(f_root, dv_exact, x0_new)
        sec = solve_secant(f_root, 0.5, 2.5)
        d_romb = integrate_romberg(v_exact, 0.0, t_stop_exact, 4)

        print(f"  * {name:17s} | t_stop={bis['root']:.4f}s | d_romb={d_romb:.4f}m (err={abs(d_romb-d_stop_exact):.2e}) "
              f"| Err_RK2={err_rk2:.2e} | Err_RK4={err_rk4:.2e} | Newton={newt['status']}")

    # --- MÓDULO 2 ---
    print("\n[MÓDULO 2] EQUILIBRIO DE TORRES (SISTEMAS LINEALES Ax = b)")
    m2_scenarios = {
        "NORMAL": (10.0, 120.0),
        "ESTRES": (4.5, 240.0),
        "MAL_CONDICIONADO": (2.02, 240.0),
    }
    for name, (d, load) in m2_scenarios.items():
        A = np.array([
            [d, -1.0, 0.0, -1.0],
            [-1.0, d, -1.0, 0.0],
            [0.0, -1.0, d, -1.0],
            [-1.0, 0.0, -1.0, d]
        ])
        b = np.array([load, load * 1.15, load * 0.95, load * 1.05])
        cond_num = np.linalg.cond(A)
        x_lu, x_j, x_gs, x_cg, err_j, err_gs, err_cg = solve_linear_systems(A, b)
        print(f"  * {name:17s} | cond(A)={cond_num:6.2f} | T_max(LU)={np.max(x_lu):6.2f}kN | "
              f"Iter(J/GS/CG)={len(err_j)}/{len(err_gs)}/{len(err_cg)} | Err_J_fin={err_j[-1]:.2e}")

    # --- MÓDULO 3 ---
    print("\n[MÓDULO 3] OSCILACIONES POR VIENTO (EDO 2° ORDEN, RAÍCES E INTEGRACIÓN)")
    m3_scenarios = {
        "NORMAL": (0.15, 0.18, 1.4, 0.10),
        "ESTRES": (0.42, 0.06, 1.8, 0.10),
        "MAL_CONDICIONADO": (0.55, 0.01, 2.2, 0.65),
    }
    for name, (th0, zeta, wn, h) in m3_scenarios.items():
        wd = wn * math.sqrt(max(1e-6, 1.0 - zeta ** 2))
        th_exact = lambda t: th0 * math.exp(-zeta * wn * t) * (math.cos(wd * t) + (zeta * wn / wd) * math.sin(wd * t))
        om_exact = lambda t: -th0 * (wn ** 2 / wd) * math.exp(-zeta * wn * t) * math.sin(wd * t)

        t_arr, x_rk2, x_rk4 = solve_ode_system_2d(
            lambda t, th, om: om,
            lambda t, th, om: -2.0 * zeta * wn * om - (wn ** 2) * th,
            0.0, th0, 0.0, 10.0, h
        )
        th_true = np.array([th_exact(t) for t in t_arr])
        err_rk2 = np.max(np.abs(x_rk2 - th_true))
        err_rk4 = np.max(np.abs(x_rk4 - th_true))
        bis = solve_bisection(th_exact, 0.1, (math.pi / (2.0 * wd)) * 1.8)
        e_romb = integrate_romberg(lambda t: om_exact(t) ** 2, 0.0, 10.0, 5)
        print(f"  * {name:17s} | t_cruce={bis['root']:.4f}s | E_romb={e_romb:.5f} | "
              f"Err_RK2={err_rk2:.2e} | Err_RK4={err_rk4:.2e}")

    # --- MÓDULO 4 ---
    print("\n[MÓDULO 4] RECONSTRUCCIÓN DE SENSORES (LAGRANGE, NEWTON Y SPLINES CÚBICOS)")
    m4_scenarios = {
        "NORMAL": (6, 0.0),
        "ESTRES": (9, 0.45),
        "MAL_CONDICIONADO": (13, 1.8),
    }
    true_profile = lambda u: 42.0 / (1.0 + 16.0 * u * u) + 12.0 * u
    eval_xs = np.linspace(-1.0, 1.0, 51)
    y_true = np.array([true_profile(u) for u in eval_xs])

    for name, (n_nodes, noise) in m4_scenarios.items():
        xs = np.linspace(-1.0, 1.0, n_nodes)
        ys = np.array([true_profile(xs[i]) + noise * math.sin(i * 2.7 + 1.1) for i in range(n_nodes)])
        y_lag = np.array([eval_lagrange(xs, ys, u) for u in eval_xs])
        y_spl = eval_natural_cubic_spline(xs, ys, eval_xs)
        err_lag = np.max(np.abs(y_lag - y_true))
        err_spl = np.max(np.abs(y_spl - y_true))
        print(f"  * {name:17s} | Nodos={n_nodes:2d} | Ruido={noise:.2f}m | "
              f"MaxErr_Lagrange={err_lag:7.4f}m | MaxErr_Spline={err_spl:6.4f}m | Mejora={err_lag/err_spl:5.2f}x")


if __name__ == "__main__":
    run_full_validation()
