#!/usr/bin/env python3
"""Independent adversarial check of track d6 (sections/d6_pair.tex, "Duplication in motion: the balanced pair").

Rebuilt from the sources only (tex/s02_arena.tex eq. (1)-(4) realisation and tower, s10 Thm 10.8 / Prop 10.9,
s13 Prop 13.14, s17 Props 17.8, 17.18, Thms 17.11-17.16, and the conventions of sections d1, d3, d4, d5); nothing is
imported or copied from the derive script verify/d6_pair_balance.py.

Methods chosen to differ from the derive script where possible:
  * the closure ODEs are integrated with LSODA (Adams/BDF; the derive script uses DOP853), and the frame
    drift off the balance surface is compared with an independent quadrature of Theorem 17.16 for n constituents;
  * M_pair = inf_t |P_sum(t)| is obtained by direct numerical minimisation over boosts of the two field strengths
    (Lorentz length of (E6_tot, S_sum)), not from the rest frame of z1 x z2; the stress-tensor identities of
    Thm d6:thm:pair(i) are checked in exact Gaussian-rational arithmetic at random points;
  * exact semisimplicity in Prop d6:prop:zpe uses the square-free part of the characteristic polynomial over Q;
  * Krein signatures are computed from the canonical Hessian on numerically computed eigenspaces.

Checks of statements that the referee found to be wrongly stated in the tex are written for the CORRECT statement
(they pass); they are tagged "[correction]" and explained in the findings file.
Prints one PASS/FAIL line per check; exit status 0 iff every check passes.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import time
import itertools
import numpy as np
import sympy as sp
from scipy.linalg import expm, null_space
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, minimize_scalar

T_START = time.time()
RESULTS = []


def report(cid, ok, claim):
    ok = bool(ok)
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} x6-{cid}: {claim}", flush=True)


rng = np.random.default_rng(20261018)

# ------------------------------------------------------------------ conventions (V = R^{3,3})
eta = np.diag([1.0, 1, 1, -1, -1, -1])
G = np.zeros((6, 6))
for i in range(3):
    G[i + 3, i] = 1.0      # G e_i = e_{i+3}
    G[i, i + 3] = -1.0     # G e_{i+3} = -e_i
I6 = np.eye(6)
eps2 = np.array([[0.0, -1.0], [1.0, 0.0]])
eta2 = np.kron(np.eye(2), eta)
G2 = np.kron(np.eye(2), G)
Ecr = np.kron(eps2, I6)          # copy rotation generator E = eps (x) id_6
g4 = np.diag([1.0, -1, -1, -1])  # Minkowski metric for stress tensors


def zf(X):
    X = np.asarray(X)
    return X[:3] + 1j * X[3:]


def Xf(z):
    return np.concatenate([np.real(z), np.imag(z)])


def qf(z):
    return z @ z


def E6f(z):
    return 0.5 * np.vdot(z, z).real


def E0f(z):
    return 0.5 * abs(z @ z)


def Sf(z):   # frame Poynting vector S = E x B = (i/2) conj(z) x z
    return np.real(0.5j * np.cross(np.conj(z), z))


def Tmix(z):
    """Mixed stress tensor T^mu_nu of (E, B_t), B_t = -B (d1:prop:rainich)."""
    B, E = np.real(z), np.imag(z)
    St = -np.cross(E, B)
    Tu = np.zeros((4, 4))
    Tu[0, 0] = E6f(z)
    Tu[0, 1:] = St
    Tu[1:, 0] = St
    Tu[1:, 1:] = E6f(z) * np.eye(3) - np.outer(E, E) - np.outer(B, B)
    return Tu @ g4


def cross_mat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=complex)


def boost(b):
    """Field-strength boost exp(i [b]_x) (K_i acts as z -> i e_i x z, d1/d4 conventions)."""
    return expm(1j * cross_mat(np.asarray(b, dtype=float)))


def Pnorm2(zs):
    """Lorentz length^2 of the summed density seen by the fixed observer, from (E6_tot, S_sum)."""
    e = sum(E6f(z) for z in zs)
    s = sum(Sf(z) for z in zs)
    return e * e - s @ s


def Mpair_numeric(z1, z2, starts=6):
    best = np.inf
    for k in range(starts):
        b0 = 0.3 * rng.normal(size=3) if k else np.zeros(3)
        f = lambda b: Pnorm2([boost(b) @ z1, boost(b) @ z2])
        r = minimize(f, b0, method='BFGS', options={'gtol': 1e-13, 'maxiter': 2000})
        best = min(best, r.fun)
    return np.sqrt(max(best, 0.0))


def Qgram(z1, z2):
    return np.array([[z1 @ z1, z1 @ z2], [z2 @ z1, z2 @ z2]])


def Mpair_formula(z1, z2):
    return 0.5 * np.linalg.svd(Qgram(z1, z2), compute_uv=False).sum()


def rz():
    return rng.normal(size=3) + 1j * rng.normal(size=3)


def rnull():
    u = rng.normal(size=3)
    v = rng.normal(size=3)
    v -= (u @ v) / (u @ u) * u
    v *= np.linalg.norm(u) / np.linalg.norm(v)
    return u + 1j * v


def Nlor(nu, m):
    return (nu.real * np.linalg.inv(eta) + nu.imag * G @ np.linalg.inv(eta)) / m


def wcov(p):   # w(p) = q(z(pi)), pi = eta^{-1} p
    return qf(zf(np.linalg.solve(eta, p)))


# Lorentz generators (binding l_G) on one copy, coordinates (B;E)
def lG_gens():
    gens = []
    for i in range(3):
        ei = np.zeros(3)
        ei[i] = 1
        C = np.real(cross_mat(ei))
        L = np.zeros((6, 6))
        L[:3, :3] = C
        L[3:, 3:] = C              # dB = e x B, dE = e x E
        K = np.zeros((6, 6))
        K[:3, 3:] = -C             # dB = -e x E
        K[3:, :3] = C              # dE = e x B
        gens += [L, K]
    return gens


LG = lG_gens()


def Rab(a, b, n=6, metric=None):
    """R_ab e_b = eta_bb e_a, R_ab e_a = -eta_aa e_b (d4 notation), on R^n."""
    met = np.asarray(metric, float) if metric is not None else np.diag(eta)
    R = np.zeros((n, n))
    R[a, b] = met[b]
    R[b, a] = -met[a]
    return R


def embed(M6, k):
    Z = np.zeros((12, 12))
    Z[6 * k:6 * k + 6, 6 * k:6 * k + 6] = M6
    return Z


def so66_basis():
    basis = []
    for a in range(12):
        for b in range(a + 1, 12):
            basis.append(Rab(a, b, 12, np.diag(eta2)))
    return basis


SO66 = so66_basis()


def commutant_in(basis, gens):
    """Subspace of span(basis) commuting with all gens; returns list of matrices."""
    A = np.array([np.concatenate([(B @ g - g @ B).ravel() for g in gens]) for B in basis]).T
    ns = null_space(A, rcond=1e-10)
    return [sum(c * B for c, B in zip(col, basis)) for col in ns.T]


def in_span(X, mats, tol=1e-9):
    if not mats:
        return np.linalg.norm(X) < tol
    A = np.array([m.ravel() for m in mats]).T
    c, *_ = np.linalg.lstsq(A, X.ravel(), rcond=None)
    return np.linalg.norm(A @ c - X.ravel()) < tol * max(1, np.linalg.norm(X))


# ===================================================================== x6-0 conventions
def check_conventions():
    ok = np.allclose(G @ G, -I6) and np.allclose(G.T, -G) and np.allclose(eta @ G, -G @ eta)
    errs = []
    for _ in range(20):
        X = rng.normal(size=6)
        z = zf(X)
        errs.append(abs(zf(G @ X) - 1j * z).max())
        errs.append(abs(X @ eta @ X - qf(z).real))
        errs.append(abs(E6f(z) ** 2 - E0f(z) ** 2 - Sf(z) @ Sf(z)))
        errs.append(np.abs(Tmix(z) @ Tmix(z) - E0f(z) ** 2 * np.eye(4)).max())
        # boost covariance of the fixed-frame length: (E6, S) has Lorentz length E0
        zb = boost(rng.normal(size=3)) @ z
        errs.append(abs(Pnorm2([zb]) - E0f(z) ** 2))
    ok &= max(errs) < 1e-10
    ok &= np.allclose(Ecr @ Ecr, -np.eye(12)) and np.allclose(Ecr @ G2, G2 @ Ecr)
    ok &= np.allclose(Ecr.T @ eta2 + eta2 @ Ecr, 0) and np.allclose(Ecr.T, -Ecr)
    # copy rotation acts on (z1,z2) as stated
    X = rng.normal(size=12)
    phi = 0.37
    Y = expm(phi * Ecr) @ X
    z1, z2 = zf(X[:6]), zf(X[6:])
    ok &= np.allclose(zf(Y[:6]), np.cos(phi) * z1 - np.sin(phi) * z2) and np.allclose(zf(Y[6:]), np.sin(phi) * z1 + np.cos(phi) * z2)
    report("0", ok, f"conventions: G^2=-1, G^T=-G, eta G=-G eta, G acts as i, E6^2=E0^2+|S|^2, Rainich T^2=E0^2, boosted "
                    f"(E6,S) has length E0, E^2=-1, [E,G2]=0, E antisymmetric for eta2 and Euclid, R_phi as stated (max err {max(errs):.1e})")


# ===================================================================== x6-1 Theorem d6:thm:balance (algebra)
def check_balance_algebra():
    errs = []
    for _ in range(20):
        nu = complex(*rng.normal(size=2))
        m = rng.uniform(0.5, 2)
        N = Nlor(nu, m)
        p = rng.normal(size=6)
        errs.append(abs(0.5 * p @ N @ p - (nu * wcov(p)).real / (2 * m)))
        errs.append(np.abs(N @ G + G @ N).max())
    ok_a = max(errs) < 1e-12
    # n=1: R_eff = |c| |w| = 2|c| E0
    nu, m = 0.8 - 0.6j, 1.7
    p = rng.normal(size=6)
    ok_a &= abs(abs(nu / m) * abs(wcov(p)) - 2 * abs(nu / m) * E0f(zf(np.linalg.solve(eta, p)))) < 1e-12
    report("1a", ok_a, "1/2 p^T N p = Re(nu w(p))/(2m) for Lorentz-class N (which anticommutes with G); for n=1 "
                       "R_eff=2|c|E0(pi), so a constant nonzero rate forces E0(pi)=0 (thm:balance(ii))")

    # (iii): rank of dF at random balanced points (n=2,3) = 2; the all-null set has dimension 4n
    def F(P, cs):
        n = len(cs)
        return sum(cs[k] * wcov(P[6 * k:6 * k + 6]) for k in range(n))

    def jac(fun, P, h=1e-6):
        cols = []
        for j in range(len(P)):
            d = np.zeros(len(P))
            d[j] = h
            v = (fun(P + d) - fun(P - d)) / (2 * h)
            cols.append(np.atleast_1d(v))
        J = np.array(cols).T
        return np.vstack([J.real, J.imag])

    ranks = []
    nullranks = []
    both_massive = True
    for n in (2, 3):
        for _ in range(3):
            cs = [complex(*rng.normal(size=2)) for _ in range(n)]
            # balanced point: random z_1..z_{n-1}, then z_n scaled so that sum c_k q_k = 0
            zs = [rz() for _ in range(n - 1)]
            target = -sum(c * qf(z) for c, z in zip(cs[:-1], zs)) / cs[-1]
            u = rz()
            zs.append(np.sqrt(target / qf(u)) * u)
            P = np.concatenate([eta @ Xf(z) for z in zs])
            assert abs(F(P, cs)) < 1e-10
            ranks.append(int(np.linalg.matrix_rank(jac(lambda Q: F(Q, cs), P), tol=1e-6)))
            if n == 2:
                both_massive &= abs(abs(cs[0]) * E0f(zs[0]) - abs(cs[1]) * E0f(zs[1])) < 1e-10 and E0f(zs[1]) > 0
            # all-null product: rank of the map (w_1,...,w_n) at a generic null point is 2n -> dimension 6n-2n = 4n
            Pn = np.concatenate([eta @ Xf(rnull()) for _ in range(n)])
            wmap = lambda Q: np.array([wcov(Q[6 * k:6 * k + 6]) for k in range(n)])
            nullranks.append((n, int(np.linalg.matrix_rank(jac(wmap, Pn), tol=1e-6))))
    ok_b = all(r == 2 for r in ranks) and all(r == 2 * n for n, r in nullranks) and both_massive
    ok_b &= all(4 * n < 6 * n - 2 for n in (2, 3, 4, 5)) and not (4 * 1 < 6 * 1 - 2)
    report("1b", ok_b, f"thm:balance(iii): rank dF = 2 at random balanced points for n=2,3 (ranks {ranks}); the all-null part has "
                       f"dimension 4n (Jacobian ranks of (w_k) = 2n: {nullranks}), 4n < 6n-2 iff n>=2; for n=2 |c1|E0^(1)=|c2|E0^(2)>0")


# ===================================================================== closure integrator (LSODA)
def closure_rhs_factory(I0, Ns):
    n = len(Ns)

    def rhs(t, y):
        th, h = y[0], y[1]
        ps = [y[2 + 6 * k:8 + 6 * k] for k in range(n)]
        out = np.zeros_like(y)
        kap = h / I0
        out[0] = kap
        out[1] = -sum(pk @ G @ Nk @ pk for pk, Nk in zip(ps, Ns))
        c, s = np.cos(th), np.sin(th)
        R = c * I6 + s * G          # e^{theta G}
        for k in range(n):
            out[2 + 6 * k:8 + 6 * k] = kap * (G.T @ ps[k])
            out[2 + 6 * n + 6 * k:8 + 6 * n + 6 * k] = R @ (Ns[k] @ ps[k])
        return out
    return rhs


def run_closure(I0, Ns, ps, h0, T, events=None, rtol=1e-12, atol=1e-13):
    n = len(Ns)
    y0 = np.concatenate([[0.0, h0]] + [np.asarray(p, float) for p in ps] + [np.zeros(6)] * n)
    return solve_ivp(closure_rhs_factory(I0, Ns), (0, T), y0, method='LSODA', rtol=rtol, atol=atol,
                     dense_output=True, events=events)


def frame_circle(kap, N, p0, t):
    # x(t) = (e^{2 kap t G} - 1)(2 kap G)^{-1} N p0, x(0)=0
    M = expm(2 * kap * t * G) - I6
    return M @ np.linalg.solve(2 * kap * G, N @ p0)


def check_balance_dynamics():
    I0 = 1.0
    ts = np.linspace(0, 40, 81)
    # (a) equal inertias, p2 = -G p1
    nu, m = 0.7 + 0.4j, 1.3
    N = Nlor(nu, m)
    p1 = rng.normal(size=6)
    p2 = -G @ p1
    h0 = 1.1
    sol = run_closure(I0, [N, N], [p1, p2], h0, 40.0)
    Y = sol.sol(ts)
    dh = np.abs(Y[1] - h0).max()
    dx = max(np.abs(Y[14 + 6 * k:20 + 6 * k, j] - frame_circle(h0 / I0, N, pk, t)).max()
             for k, pk in enumerate((p1, p2)) for j, t in enumerate(ts))
    diam_ok = True
    for k, pk in enumerate((p1, p2)):
        xs = Y[14 + 6 * k:20 + 6 * k]
        centre = -np.linalg.solve(2 * (h0 / I0) * G, N @ pk)
        rad = np.linalg.norm(xs - centre[:, None], axis=0)
        diam_ok &= abs(2 * rad.max() - np.linalg.norm(N @ pk) / (h0 / I0)) < 1e-8 and np.ptp(rad) < 1e-8
    E0s = [E0f(zf(np.linalg.solve(eta, p))) for p in (p1, p2)]
    ok = dh < 1e-9 and dx < 1e-8 and diam_ok and abs(E0s[0] - E0s[1]) < 1e-12 and E0s[0] > 0.1
    report("2a", ok, f"thm:balance(i) by LSODA: equal inertias, p2=-Gp1 (E0={E0s[0]:.3f} each): |h-h0|<={dh:.1e} on [0,40], "
                     f"frames on the circles of (i) to {dx:.1e}, diameter |N p|/|kappa|")

    # (b) unequal inertias, z2 = rho Lambda z1
    nu1, m1, nu2, m2 = 1 + 0.3j, 1.0, 0.4 - 0.7j, 2.5
    c1, c2 = nu1 / m1, nu2 / m2
    N1, N2 = Nlor(nu1, m1), Nlor(nu2, m2)
    z1 = rz()
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Lam = expm(0.4 * (A - A.T))
    z2 = np.sqrt(-c1 / c2) * Lam @ z1
    p1, p2 = eta @ Xf(z1), eta @ Xf(z2)
    Reff = abs(c1 * wcov(p1) + c2 * wcov(p2))
    h0 = 0.9
    sol = run_closure(I0, [N1, N2], [p1, p2], h0, 40.0)
    Y = sol.sol(ts)
    dh = np.abs(Y[1] - h0).max()
    dx = max(np.abs(Y[14 + 6 * k:20 + 6 * k, j] - frame_circle(h0 / I0, Nk, pk, t)).max()
             for k, (Nk, pk) in enumerate(((N1, p1), (N2, p2))) for j, t in enumerate(ts))
    ok = Reff < 1e-12 and dh < 1e-9 and dx < 1e-8 and abs(abs(c1) * E0f(z1) - abs(c2) * E0f(z2)) < 1e-10
    ok &= abs(np.linalg.det(Lam) - 1) < 1e-10 and np.abs(Lam.T @ Lam - np.eye(3)).max() < 1e-10
    report("2b", ok, f"prop:pairs(iv)+thm:balance(i): unequal inertias (m=1,2.5; nu=1+0.3i,0.4-0.7i), z2=rho Lambda z1, "
                     f"Lambda in SO(3,C): R_eff={Reff:.1e}, |h-h0|<={dh:.1e}, frames on circles to {dx:.1e}, |c1|E0^(1)=|c2|E0^(2)")

    # (c) off the surface: pendulum law and additive frame drift (Thm 17.16 for n constituents, by quadrature)
    p2b = p2 + 0.05 * rng.normal(size=6)
    S_eff = c1 * wcov(p1) + c2 * wcov(p2b)
    R, ph = abs(S_eff), np.angle(S_eff)
    h0 = 1.0
    epsm = R / (I0 * (h0 / I0) ** 2)

    def ev_factory(target):
        def ev(t, y):
            return y[0] - target
        ev.terminal = False
        ev.direction = 1
        return ev
    evs = [ev_factory(np.pi * k) for k in (1, 2, 3)]
    sol = run_closure(I0, [N1, N2], [p1, p2b], h0, 60.0, events=evs)
    tt = np.linspace(0, 60, 601)
    Y = sol.sol(tt)
    # pendulum: energy 1/2 I0 thdot^2 + 1/2 R cos(2 th + ph) conserved and equals H
    Epend = 0.5 * Y[1] ** 2 / I0 + 0.5 * R * np.cos(2 * Y[0] + ph)
    Udir = np.array([0.5 * (Y[2:8, j] @ N1 @ Y[2:8, j] + Y[8:14, j] @ N2 @ Y[8:14, j]) for j in range(len(tt))])
    dU = np.abs(Udir - 0.5 * R * np.cos(2 * Y[0] + ph)).max()
    dE = np.ptp(Epend)
    tev = list(sol.t_events)
    rotating = all(len(t) > 0 for t in tev)
    ok = rotating and dU < 1e-9 and dE < 1e-9 and epsm < 0.5
    if rotating:
        x_at = [sol.sol(t[0])[14:20] for t in tev]
        kap0 = h0 / I0
        psi0 = ph
        # periodic analytic integrand: the trapezoidal rule on a uniform grid converges geometrically
        sg = np.linspace(0, 2 * np.pi, 4097)[:-1]
        lam = 0.5 * (2 * np.pi / len(sg)) * np.sum(np.cos(sg) / np.sqrt(kap0 ** 2 + (R / I0) * (np.cos(psi0) - np.cos(sg))))
        lam2 = 0.5 * quad(lambda s: np.cos(s) / np.sqrt(kap0 ** 2 + (R / I0) * (np.cos(psi0) - np.cos(s))), 0, 2 * np.pi,
                          epsabs=1e-12, epsrel=1e-11, limit=200)[0]
        ok &= abs(lam - lam2) < 1e-10
        dpred = lam * expm(-ph * G) @ (N1 @ p1)
        d1 = x_at[0]
        ok &= np.abs(d1 - dpred).max() < 1e-8
        ok &= np.abs(x_at[1] - 2 * d1).max() < 1e-8 and np.abs(x_at[2] - 3 * d1).max() < 1e-8
        drift = np.linalg.norm(d1)
    else:
        drift = float('nan')
    report("2c", ok, f"off the balance surface (R_eff={R:.4f}, eps={epsm:.3f}, rotating): U = 1/2 R_eff cos(2theta+phi_eff) "
                     f"to {dU:.1e}, pendulum energy constant to {dE:.1e}; frame 1 drifts by lambda e^(-phi G) N1 p1 "
                     f"(|dx|={drift:.4f} per half turn, quadrature) and exactly additively over three half turns")


# ===================================================================== x6-1c Remark d6:rem:lorentzreq (+ correction)
def check_lorentzreq():
    m = 1.0
    N = np.diag([1.0, -1, 0, -1, 1, 0]) / m
    p0 = np.zeros(6)
    p0[0] = p0[1] = 1.0
    pi0 = np.linalg.solve(eta, p0)
    ok = np.allclose(N @ G + G @ N, 0) and np.allclose(zf(pi0), [1, 1, 0]) and abs(E0f(zf(pi0)) - 1) < 1e-14
    # not l_G-invariant: r N + N r^T != 0 for some r in l_G
    ok &= max(np.abs(r @ N + N @ r.T).max() for r in LG) > 0.5
    # U(theta) == 0 exactly (sympy)
    th = sp.symbols('theta', real=True)
    Gs = sp.Matrix(G.astype(int))
    eG = sp.cos(2 * th) * sp.eye(6) + sp.sin(2 * th) * Gs
    U = sp.simplify((sp.Matrix(p0.astype(int)).T * eG * sp.Matrix(N.astype(int)) * sp.Matrix(p0.astype(int)))[0, 0] / 2)
    ok &= U == 0
    sol = run_closure(1.0, [N], [p0], 0.8, 30.0)
    ts = np.linspace(0, 30, 61)
    Y = sol.sol(ts)
    dh = np.abs(Y[1] - 0.8).max()
    dx = max(np.abs(Y[8:14, j] - frame_circle(0.8, N, p0, t)).max() for j, t in enumerate(ts))
    ok &= dh < 1e-10 and dx < 1e-8
    report("1c", ok, f"rem:lorentzreq: N=diag(1,-1,0,-1,1,0) anticommutes with G, is not l_G-invariant, z(pi0)=(1,1,0), "
                     f"E0=1, U(theta)==0 symbolically; LSODA: |h-h0|<={dh:.1e}, frame on a circle to {dx:.1e}")
    # [correction] duality-class N = id/m: constant rate with any massive momentum field, no cancellation; frame drifts
    Nd = np.eye(6) / 1.3
    p0 = rng.normal(size=6)
    sol = run_closure(1.0, [Nd], [p0], 0.7, 20.0)
    Y = sol.sol(np.linspace(0, 20, 41))
    dh = np.abs(Y[1] - 0.7).max()
    U_t = [0.5 * Y[2:8, j] @ Nd @ Y[2:8, j] for j in range(41)]
    xend = np.linalg.norm(Y[8:14, -1])
    xmid = np.linalg.norm(Y[8:14, 20])
    ok2 = dh < 1e-10 and np.ptp(U_t) < 1e-10 and abs(U_t[0]) > 0.1 and E0f(zf(np.linalg.solve(eta, p0))) > 0.05 and xend > 1.8 * xmid
    report("1d", ok2, f"[correction to rem:lorentzreq 'only through a cancellation'] duality-class N=id/m on one constituent: "
                      f"rate constant (|h-h0|<={dh:.1e}) with E0(pi)={E0f(zf(np.linalg.solve(eta, p0))):.3f}>0 and U constant and "
                      f"nonzero (no cancellation), at the price of a drifting frame (|x| grows linearly)")


# ===================================================================== x6-3 Corollary d6:cor:nopair
def check_nopair():
    M1 = rng.normal(size=(12, 12))
    M1 = (M1 + M1.T) / 20
    M2 = rng.normal(size=(12, 12))
    M2 = (M2 + M2.T) / 20
    M3 = rng.normal(size=(6, 6)) / 5

    def grads(h, p):
        s1 = p @ M1 @ p
        s2 = p @ M2 @ p
        cpl = p[:6] @ M3 @ p[6:]
        Hh = h + 0.4 * h ** 3 + 0.2 * s2 + 0.3 * np.sin(h) * cpl ** 2
        gp = np.cos(s1) * 2 * M1 @ p + 0.4 * h * M2 @ p
        gc = np.zeros(12)
        gc[:6] = M3 @ p[6:]
        gc[6:] = M3.T @ p[:6]
        gp = gp + (1 - 0.3 * np.cos(h)) * 2 * cpl * gc
        return Hh, gp

    def Hfun(h, p):
        s1 = p @ M1 @ p
        s2 = p @ M2 @ p
        cpl = p[:6] @ M3 @ p[6:]
        return h * h / 2 + 0.1 * h ** 4 + np.sin(s1) + 0.2 * h * s2 + (1 - 0.3 * np.cos(h)) * cpl ** 2

    def rhs(t, y):
        th, h, p = y[0], y[1], y[2:]
        Hh, gp = grads(h, p)
        return np.concatenate([[Hh, -p @ G2 @ gp], Hh * (G2.T @ p)])

    p0 = rng.normal(size=12)
    sol = solve_ivp(rhs, (0, 8), np.concatenate([[0, 0.6], p0]), method='LSODA', rtol=1e-12, atol=1e-13, dense_output=True)
    ts = np.linspace(0, 8, 81)
    Y = sol.sol(ts)

    def invariants(p):
        z1 = zf(np.linalg.solve(eta, p[:6]))
        z2 = zf(np.linalg.solve(eta, p[6:]))
        Hg = np.array([[np.vdot(a, b) for b in (z1, z2)] for a in (z1, z2)])
        Q = Qgram(z1, z2)
        return np.concatenate([[E6f(z1), E6f(z2), E0f(z1), E0f(z2)], Sf(z1), Sf(z2), Hg.real.ravel(), Hg.imag.ravel(),
                               np.abs(Q).ravel()]), Q
    inv0, Q0 = invariants(p0)
    dinv = 0.0
    dQ = 0.0
    dH = 0.0
    H0 = Hfun(0.6, p0)
    for j in range(len(ts)):
        inv, Q = invariants(Y[2:, j])
        dinv = max(dinv, np.abs(inv - inv0).max())
        dQ = max(dQ, np.abs(Q - np.exp(2j * Y[0, j]) * Q0).max())
        dH = max(dH, abs(Hfun(Y[1, j], Y[2:, j]) - H0))
    phase_moved = abs(Y[0, -1])
    ok = dinv < 1e-8 and dQ < 1e-8 and dH < 1e-8 and phase_moved > 0.5
    report("3", ok, f"cor:nopair: LSODA with a random Hamiltonian coupling the copies (quartic in h, trigonometric in p, "
                    f"cross term (p1 M p2)^2 cos h): per-copy E6,E0,S, Hermitian Gram and |Q_kl| constant to {dinv:.1e}, "
                    f"Q = e^(2i theta)Q(0) to {dQ:.1e}, H to {dH:.1e}, theta moved {phase_moved:.2f} rad")


# ===================================================================== x6-4 Proposition d6:prop:pairs
def check_pairs():
    # (i) symbolic in a, b (complex 3-vectors)
    a = sp.Matrix(sp.symbols('a1:4'))
    b = sp.Matrix(sp.symbols('b1:4'))
    I = sp.I
    z1 = a + b
    z2 = I * (a - b)              # inverse of a=(z1-iz2)/2, b=(z1+iz2)/2
    ok = all(sp.expand(e) == 0 for e in (z1 - I * z2) / 2 - a)
    ok &= all(sp.expand(e) == 0 for e in (z1 + I * z2) / 2 - b)
    dot = lambda u, v: (u.T * v)[0, 0]
    ok &= sp.expand(dot(z1, z1) + dot(z2, z2) - 4 * dot(a, b)) == 0
    ok &= sp.expand(dot(z1, z1) - (dot(a, a) + 2 * dot(a, b) + dot(b, b))) == 0
    # on a.b = 0: q2 = -q1, E0 equal
    for _ in range(5):
        av = rz()
        bv = rz()
        bv = bv - (av @ bv) / (av @ av) * av
        z1v, z2v = av + bv, 1j * (av - bv)
        ok &= abs(qf(z1v) + qf(z2v)) < 1e-12 and abs(E0f(z1v) - 0.5 * abs(av @ av + bv @ bv)) < 1e-12
        ok &= abs(E0f(z1v) - E0f(z2v)) < 1e-12
    report("4a", ok, "prop:pairs(i): q1+q2=4a.b, q1=a.a+2a.b+b.b (symbolic); on a.b=0, q2=-q1 and E0^(1)=E0^(2)=|a.a+b.b|/2")

    # (ii) EG2 = eps (x) G
    EG = Ecr @ G2
    ok = np.allclose(EG, np.kron(eps2, G)) and np.allclose(EG @ EG, np.eye(12)) and np.allclose(EG.T @ eta2 + eta2 @ EG, 0)
    for sgn in (+1, -1):
        Vs = null_space(EG - sgn * np.eye(12))
        ok &= Vs.shape[1] == 6 and np.abs(Vs.T @ eta2 @ Vs).max() < 1e-12
        ok &= np.allclose(Ecr @ Vs, -sgn * G2 @ Vs)
        # V_+ = {X2 = G X1}, V_- = {X2 = -G X1}
        ok &= np.allclose(Vs[6:], sgn * G @ Vs[:6])
        # momentum field in V_sgn iff p2 = -sgn G p1
        p1 = rng.normal(size=6)
        p2 = -sgn * G @ p1
        pi = np.linalg.solve(eta2, np.concatenate([p1, p2]))
        ok &= np.allclose(EG @ pi, sgn * pi)
        z1v, z2v = zf(pi[:6]), zf(pi[6:])
        av, bv = (z1v - 1j * z2v) / 2, (z1v + 1j * z2v) / 2
        ok &= (np.abs(bv).max() < 1e-12) if sgn > 0 else (np.abs(av).max() < 1e-12)
        # copy rotation = duality rotation exp(-+ phi G2) on V_sgn
        phi = 0.81
        ok &= np.allclose(expm(phi * Ecr) @ Vs, expm(-sgn * phi * G2) @ Vs)
    report("4b", ok, "prop:pairs(ii): EG2=eps(x)G is an eta2-antisymmetric involution; V_pm={X2=pm G X1} six-dimensional, "
                     "totally eta2-null, E=-+G2 there; pi in V_+ iff p2=-Gp1 iff b=0, in V_- iff p2=Gp1 iff a=0; "
                     "R_phi = exp(-+phi G2) on V_pm")

    # (iii) on V_+: z2 = i z1, w2=-w1, equal E6, S, E0, T(z2)=T(z1), M_pair = 2E0^(1), q_tot = 0
    ok = True
    for _ in range(4):
        z1v = rz()
        if qf(z1v).real < 0:
            z1v = 1j * z1v
        z2v = 1j * z1v
        ok &= abs(qf(z2v) + qf(z1v)) < 1e-12 and abs(E6f(z1v) - E6f(z2v)) < 1e-12 and np.allclose(Sf(z1v), Sf(z2v))
        ok &= np.allclose(Tmix(z1v), Tmix(z2v))
        Mn = Mpair_numeric(z1v, z2v)
        ok &= abs(Mn - 2 * E0f(z1v)) < 1e-6 and abs(Mpair_formula(z1v, z2v) - 2 * E0f(z1v)) < 1e-12
        # magnetic / electric type when q1 > 0 (real part sign of q)
        ok &= qf(z1v).real > 0 and qf(z2v).real < 0
    report("4c", ok, "prop:pairs(iii): on V_+ (z2=iz1) w2=-w1, equal E6, S, E0, T(z2)=T(z1), M_pair=2E0^(1) (numerical inf over "
                     "boosts) although q_tot=0; Re q1>0 > Re q2 (magnetic/electric type)")

    # invariant symmetric forms on V2^*: 4, 6, 2, all Lorentz class
    def inv_forms(gens):
        basis = []
        for i in range(12):
            for j in range(i, 12):
                B = np.zeros((12, 12))
                B[i, j] = B[j, i] = 1
                basis.append(B)
        A = np.array([np.concatenate([(r @ B + B @ r.T).ravel() for r in gens]) for B in basis]).T
        ns = null_space(A, rcond=1e-10)
        return [sum(c * B for c, B in zip(col, basis)) for col in ns.T]
    g_split = [embed(r, 0) for r in LG] + [embed(r, 1) for r in LG]
    g_diag = [embed(r, 0) + embed(r, 1) for r in LG]
    cent = commutant_in(SO66, [G2])
    f4, f6, f2 = inv_forms(g_split), inv_forms(g_diag), inv_forms(cent)
    lor = all(np.abs(N @ G2 + G2 @ N).max() < 1e-9 for N in f4 + f6 + f2)
    # the six diagonal forms are U = 1/2 Re tr(C Q(pi)), C complex symmetric
    Cbasis = []
    for (k, l) in ((0, 0), (1, 1), (0, 1)):
        for ph in (1, 1j):
            C = np.zeros((2, 2), complex)
            C[k, l] = C[l, k] = ph
            Cbasis.append(C)

    def form_from_C(C):
        # Hessian of p -> 1/2 Re tr(C Q(pi)) with pi = eta2^{-1} p
        H = np.zeros((12, 12))
        f = lambda p: 0.5 * np.real(np.trace(C @ Qgram(zf(np.linalg.solve(eta, p[:6])), zf(np.linalg.solve(eta, p[6:])))))
        for i in range(12):
            for j in range(12):
                ei = np.zeros(12)
                ej = np.zeros(12)
                ei[i] = 1
                ej[j] = 1
                H[i, j] = f(ei + ej) - f(ei) - f(ej)     # quadratic: polarisation
        return H
    NC = [form_from_C(C) for C in Cbasis]
    span_ok = np.linalg.matrix_rank(np.array([N.ravel() for N in NC]), tol=1e-9) == 6 and all(in_span(N, f6) for N in NC)
    ok = len(f4) == 4 and len(f6) == 6 and len(f2) == 2 and lor and span_ok and len(cent) == 30
    report("4d", ok, f"invariant symmetric forms on V2^*: {len(f4)} under lG1+lG2, {len(f6)} under the diagonal l_G "
                     f"(= 1/2 Re tr(C Q(pi)), C complex symmetric), {len(f2)} under Cent(G2) cap so(6,6) (dim {len(cent)}); "
                     f"all of Lorentz class")


# ===================================================================== x6-5 Proposition d6:prop:covariant
def check_covariant():
    g_diag = [embed(r, 0) + embed(r, 1) for r in LG]
    gl_basis = []
    for i in range(12):
        for j in range(12):
            B = np.zeros((12, 12))
            B[i, j] = 1
            gl_basis.append(B)
    comm = commutant_in(gl_basis, g_diag)
    ok = len(comm) == 8 and all(np.allclose(C @ G2, G2 @ C) for C in comm)
    # each is A (x) id3 acting on Z: check by comparing with the complex 2x2 matrix read off
    for C in comm[:3]:
        X = rng.normal(size=12)
        Z = np.array([zf(X[:6]), zf(X[6:])])        # 2 x 3
        A = np.zeros((2, 2), complex)
        for l in range(2):
            e = np.zeros(12)
            e[6 * l] = 1                            # (z_l = e_1 real)
            Y = C @ e
            for k in range(2):
                A[k, l] = zf(Y[6 * k:6 * k + 6])[0]
        Y = C @ X
        ok &= np.allclose(np.array([zf(Y[:6]), zf(Y[6:])]), A @ Z)
    c_so = commutant_in(SO66, g_diag)
    ok &= len(c_so) == 2 and all(in_span(C, [Ecr, Ecr @ G2]) for C in c_so)
    ok &= np.allclose((Ecr @ G2) @ (Ecr @ G2), np.eye(12))       # noncompact generator
    # u(2): commuting with diag rotations and G2 inside so(12); its intersection with so(6,6) is R E
    rots = [embed(r, 0) + embed(r, 1) for r in LG[0::2]]
    so12 = []
    for a in range(12):
        for b in range(a + 1, 12):
            B = np.zeros((12, 12))
            B[a, b], B[b, a] = 1, -1
            so12.append(B)
    u2 = commutant_in(so12, rots + [G2])
    u2so66 = [X for X in u2 if np.allclose(X.T @ eta2 + eta2 @ X, 0)]
    # intersection dimension: solve within span(u2) for eta2-antisymmetry
    Amat = np.array([(X.T @ eta2 + eta2 @ X).ravel() for X in u2]).T
    dim_int = len(u2) - np.linalg.matrix_rank(Amat, tol=1e-9)
    ok &= len(u2) == 4 and dim_int == 1 and in_span(Ecr, u2)
    report("5a", ok, f"prop:covariant(i): commutant of the diagonal l_G in gl(12,R) has dim {len(comm)} (= A(x)id3, A in M2(C)), "
                     f"inside so(6,6) dim {len(c_so)} = span{{E,EG2}} = g(0,2,0); u(2) dim {len(u2)}, u(2) cap so(6,6) = R E")

    # (ii) Q -> A Q A^T ; unitary A keeps H_sum, sigmas, not q_tot; R_phi keeps q_tot, det Q too
    ok = True
    for _ in range(5):
        z1, z2 = rz(), rz()
        Z = np.array([z1, z2]).T          # 3 x 2, columns z_k
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        Zp = Z @ A.T
        ok &= np.allclose(Zp.T @ Zp, A @ (Z.T @ Z) @ A.T)
        Uu, _ = np.linalg.qr(A)
        Zu = Z @ Uu.T
        ok &= np.allclose(Zu @ Zu.conj().T, Z @ Z.conj().T)
        ok &= np.allclose(np.linalg.svd(Zu.T @ Zu, compute_uv=False), np.linalg.svd(Z.T @ Z, compute_uv=False))
        ok &= abs(np.trace(Zu.T @ Zu) - np.trace(Z.T @ Z)) > 1e-3
        Rp = expm(0.7 * eps2)
        Zr = Z @ Rp.T
        ok &= abs(np.trace(Zr.T @ Zr) - np.trace(Z.T @ Z)) < 1e-12 and abs(np.linalg.det(Zr.T @ Zr) - np.linalg.det(Z.T @ Z)) < 1e-10
        ok &= np.allclose(sum(Tmix(Zr[:, k]) for k in range(2)), sum(Tmix(Z[:, k]) for k in range(2)))
    report("5b", ok, "prop:covariant(ii): A(x)id3 maps Q->AQA^T; unitary A preserve H_sum and sigma_1,2 but not q_tot; R_phi also "
                     "preserves q_tot, det Q and the summed stress tensor")

    # (iii) stabiliser algebra of a null z in so(3,C) is C z; its common fixed vectors in C^6 are (alpha z, beta z)
    ok = True
    for _ in range(4):
        z = rnull()
        # w with w x z = 0
        Wz = -cross_mat(z)            # w x z = -(z x w)
        ns = null_space(Wz)
        ok &= ns.shape[1] == 1 and np.linalg.matrix_rank(np.column_stack([ns[:, 0], z]), tol=1e-10) == 1
        w = ns[:, 0]
        act = np.kron(np.eye(2), cross_mat(w))        # (z1,z2) -> (w x z1, w x z2)
        fix = null_space(act)
        ok &= fix.shape[1] == 2
        for col in fix.T:
            ok &= np.linalg.matrix_rank(np.column_stack([col[:3], z]), tol=1e-9) <= 1
            ok &= np.linalg.matrix_rank(np.column_stack([col[3:], z]), tol=1e-9) <= 1
        # and the group: a finite null rotation exp(t [w]_x) fixes z
        Lam = expm(cross_mat(0.9 * w))
        ok &= np.allclose(Lam @ z, z) and abs(np.linalg.det(Lam) - 1) < 1e-10
        # R_phi(z,0) = (cos z, sin z): both null, M_pair = 0
        phi = 0.6
        ok &= abs(qf(np.cos(phi) * z)) < 1e-12 and abs(qf(np.sin(phi) * z)) < 1e-12
        ok &= Mpair_formula(np.cos(phi) * z, np.sin(phi) * z) < 1e-12
    report("5c", ok, "prop:covariant(iii): the stabiliser algebra of a null z is C z, its common fixed vectors in C^6 are "
                     "(alpha z, beta z); so every diagonally SO(3,C)-equivariant map sends (z,0) to two null fields; "
                     "R_phi(z,0) has M_pair=0")


# ===================================================================== x6-6 Theorem d6:thm:pair
def check_pair():
    # (i) exact identities at random Gaussian-rational points
    def rat():
        return sp.Rational(int(rng.integers(-9, 10)), int(rng.integers(1, 6)))

    def symT(z):
        B = [sp.re(c) for c in z]
        E = [sp.im(c) for c in z]
        E6 = sum(b * b for b in B) / 2 + sum(e * e for e in E) / 2
        Bv, Ev = sp.Matrix(B), sp.Matrix(E)
        St = -Ev.cross(Bv)
        Tu = sp.zeros(4, 4)
        Tu[0, 0] = E6
        for i in range(3):
            Tu[0, i + 1] = Tu[i + 1, 0] = St[i]
            for j in range(3):
                Tu[i + 1, j + 1] = (E6 if i == j else 0) - Ev[i] * Ev[j] - Bv[i] * Bv[j]
        return Tu * sp.diag(1, -1, -1, -1)
    ok = True
    for _ in range(8):
        z1 = sp.Matrix([rat() + sp.I * rat() for _ in range(3)])
        z2 = sp.Matrix([rat() + sp.I * rat() for _ in range(3)])
        T1, T2 = symT(z1), symT(z2)
        q12 = (z1.T * z2)[0, 0]
        ok &= sp.simplify((T1 * T2).trace() - sp.expand(q12 * sp.conjugate(q12))) == 0
        Q = sp.Matrix([[(z1.T * z1)[0, 0], q12], [q12, (z2.T * z2)[0, 0]]])
        fro2 = sum(sp.expand(x * sp.conjugate(x)) for x in Q)
        w = z1.cross(z2)
        lhs = (T1 + T2) * (T1 + T2)
        rhs = fro2 / 4 * sp.eye(4) + symT(w)
        ok &= all(sp.simplify(sp.expand(e)) == 0 for e in (lhs - rhs))
        ok &= sp.expand((w.T * w)[0, 0] - Q.det()) == 0
    report("6a", ok, "thm:pair(i) in exact Gaussian-rational arithmetic (8 random points): tr(T1 T2)=|z1.z2|^2, "
                     "(T1+T2)^2 = 1/4||Q||_F^2 id + T(z1 x z2), (z1 x z2).(z1 x z2) = det Q")

    # (ii) M_pair = (sigma1+sigma2)/2 by direct minimisation over boosts; attained at the rest frame of w = z1 x z2
    errs = []
    rest_ok = True
    for _ in range(5):
        z1, z2 = rz(), rz()
        Mn = Mpair_numeric(z1, z2)
        Mf = Mpair_formula(z1, z2)
        errs.append(abs(Mn - Mf) / Mf)
        w = np.cross(z1, z2)
        # rest frame of w by minimising E6 over boosts (S(w') = 0 there)
        r = minimize(lambda b: E6f(boost(b) @ w), np.zeros(3), method='BFGS', options={'gtol': 1e-12})
        Lb = boost(r.x)
        rest_ok &= np.linalg.norm(Sf(Lb @ w)) < 1e-5 * E6f(w)
        val = np.sqrt(Pnorm2([Lb @ z1, Lb @ z2]))
        rest_ok &= abs(val - Mf) < 1e-8 * Mf
        for _ in range(100):
            Lr = boost(0.8 * rng.normal(size=3))
            rest_ok &= np.sqrt(Pnorm2([Lr @ z1, Lr @ z2])) >= Mf * (1 - 1e-12)
    ok = max(errs) < 1e-6 and rest_ok
    report("6b", ok, f"thm:pair(ii): inf over boosts of |P_sum| = (sigma1+sigma2)/2 (rel. dev. {max(errs):.1e}), attained in the "
                     f"rest frame of z1 x z2, and no random observer sees less")

    # (iii) chain, in random frames; (iv) max over phi; (v) null pairs
    ok3 = True
    for _ in range(10):
        z1, z2 = rz(), rz()
        Mf = Mpair_formula(z1, z2)
        for _ in range(5):
            L = boost(rng.normal(size=3))
            a, b = L @ z1, L @ z2
            ok3 &= E0f(a) + E0f(b) <= Mf + 1e-12
            ok3 &= Mf <= np.sqrt(Pnorm2([a, b])) + 1e-12
            ok3 &= np.sqrt(Pnorm2([a, b])) <= E6f(a) + E6f(b) + 1e-12
        ok3 &= Mf >= 0.5 * abs(qf(z1) + qf(z2)) - 1e-12
    report("6c", ok3, "thm:pair(iii): E0^(1)+E0^(2) <= M_pair <= |P_sum(t)| <= E6^(1)+E6^(2) in random frames, and "
                      "M_pair >= |q_tot|/2 (50 frames)")
    devs = []
    for _ in range(6):
        z1, z2 = rz(), rz()
        Mf = Mpair_formula(z1, z2)
        f = lambda ph: -(E0f(np.cos(ph) * z1 - np.sin(ph) * z2) + E0f(np.sin(ph) * z1 + np.cos(ph) * z2))
        grid = np.linspace(0, np.pi, 4001)
        vals = np.array([f(x) for x in grid])
        x0 = grid[np.argmin(vals)]
        r = minimize_scalar(f, bounds=(x0 - 1e-3, x0 + 1e-3), method='bounded', options={'xatol': 1e-13})
        devs.append(abs(-r.fun - Mf) / Mf)
        devs.append(max(0.0, (-vals.min() - Mf) / Mf))
    ok4 = max(devs) < 1e-9
    report("6d", ok4, f"thm:pair(iv): max over phi of E0^(1)+E0^(2) on R_phi Z equals M_pair (rel. dev. {max(devs):.1e}, grid+refine)")
    ok5 = True
    for _ in range(6):
        z1, z2 = rnull(), rnull()
        Mf = Mpair_formula(z1, z2)
        s1, s2 = Sf(z1), Sf(z2)
        cth = s1 @ s2 / (np.linalg.norm(s1) * np.linalg.norm(s2))
        ok5 &= abs(Mf - abs(z1 @ z2)) < 1e-12 and abs(Mf - np.sqrt(E6f(z1) * E6f(z2)) * (1 - cth)) < 1e-10
        ok5 &= abs(Mpair_numeric(z1, z2) - Mf) < 1e-5 * max(Mf, 1e-3)
        ok5 &= Mpair_formula(z1, 0 * z1) < 1e-14
    report("6e", ok5, "thm:pair(v): for two null fields M_pair=|z1.z2|=(E6^(1)E6^(2))^(1/2)(1-cos theta_12); (z,0) null has M_pair=0")

    # [correction / consistency with d5:rem:twoE0] the 'pair invariant' E_pair^2 = 1/4 tr T_tot^2 = ||Q||_F^2/4 is not M_pair
    ok = True
    for _ in range(5):
        z1, z2 = rz(), rz()
        Tt = Tmix(z1) + Tmix(z2)
        Ep2 = 0.25 * np.trace(Tt @ Tt)
        Q = Qgram(z1, z2)
        ok &= abs(Ep2 - 0.25 * np.linalg.norm(Q) ** 2) < 1e-10
        ok &= abs(Mpair_formula(z1, z2) ** 2 - (Ep2 + 0.5 * abs(np.linalg.det(Q)))) < 1e-9
        ok &= Mpair_formula(z1, z2) > np.sqrt(Ep2) + 1e-6
    z1, z2 = np.array([1, 1j, 0]), np.array([1, -1j, 0])
    Tt = Tmix(z1) + Tmix(z2)
    Ep2 = 0.25 * np.trace(Tt @ Tt)
    ok &= abs(Ep2 - 2) < 1e-14 and abs(Mpair_formula(z1, z2) - 2) < 1e-14
    report("6f", ok, "[consistency d5/d6] d5's pair invariant E_pair=(1/4 tr T_tot^2)^(1/2)=||Q||_F/2 satisfies "
                     "M_pair^2=E_pair^2+|det Q|/2 > E_pair^2 when det Q != 0; for (1,i,0),(1,-i,0): E_pair^2=2 but M_pair=2")

    # [correction] M_pair is invariant only under the diagonal Lorentz action: a boost of one copy changes it
    z1, z2 = np.array([1, 1j, 0]), np.array([1, -1j, 0])
    L = boost([0, 0, 0.7])
    Mdiag = Mpair_formula(L @ z1, L @ z2)
    Mone = Mpair_formula(z1, L @ z2)
    ok = abs(Mdiag - 2) < 1e-12 and abs(Mone - 2) > 0.1 and abs(E0f(L @ z2)) < 1e-12
    report("6g", ok, f"[precision] M_pair is a diagonal-Lorentz invariant only: boosting copy 2 alone along e3 (rapidity 0.7) "
                     f"takes M_pair of (1,i,0),(1,-i,0) from 2 to {Mone:.4f}, while each E0^(k) stays 0")


# ===================================================================== x6-7 Example d6:ex:bw (+ correction)
def check_bw():
    z1, z2 = np.array([1, 1j, 0]), np.array([1, -1j, 0])
    ok = abs(qf(z1)) < 1e-15 and abs(qf(z2)) < 1e-15
    ok &= np.allclose(Sf(z1), [0, 0, -1]) and np.allclose(Sf(z2), [0, 0, 1])
    ok &= abs(E6f(z1) + E6f(z2) - 2) < 1e-15 and np.allclose(Qgram(z1, z2), [[0, 2], [2, 0]])
    ok &= abs(Mpair_formula(z1, z2) - 2) < 1e-14 and abs(Mpair_numeric(z1, z2) - 2) < 1e-6
    c = s = np.sqrt(0.5)
    a, b = c * z1 - s * z2, s * z1 + c * z2
    ok &= np.allclose(a, [0, np.sqrt(2) * 1j, 0]) and np.allclose(b, [np.sqrt(2), 0, 0])
    ok &= abs(qf(a) + 2) < 1e-14 and abs(qf(b) - 2) < 1e-14 and abs(E0f(a) - 1) < 1e-14 and abs(E0f(b) - 1) < 1e-14
    ok &= np.allclose(Sf(a), 0) and np.allclose(Sf(b), 0)
    chain = [E0f(a) + E0f(b), Mpair_formula(a, b), np.sqrt(Pnorm2([a, b])), E6f(a) + E6f(b)]
    ok &= np.allclose(chain, 2)
    report("7", ok, "ex:bw: (1,i,0),(1,-i,0) null, S=-+e3, E6_tot=2, Q=[[0,2],[2,0]], M_pair=2; R_{pi/4} gives (0,sqrt2 i,0), "
                    "(sqrt2,0,0), q=-2,+2, E0=1 each, at rest; the chain E0^(1)+E0^(2)=M_pair=|P_sum|=E6_tot=2")
    ok2 = abs(0.5 * abs(qf(a) + qf(b))) < 1e-14 and Mpair_formula(a, b) > 1.9
    report("7b", ok2, "[correction to ex:bw 'every inequality of thm:pair(iii) is an equality'] the last inequality is strict: "
                      "M_pair = 2 > |q_tot|/2 = 0")
    # [correction] the copy rotation does not conserve the sum of the copies' rest energies
    vals = [E0f(np.cos(p) * z1 - np.sin(p) * z2) + E0f(np.sin(p) * z1 + np.cos(p) * z2) for p in np.linspace(0, np.pi / 2, 9)]
    ok3 = abs(vals[0]) < 1e-14 and abs(vals[4] - 2) < 1e-14 and np.ptp(vals) > 1.9
    report("7c", ok3, "[correction to 'moves rest energy between the manifolds'] along R_phi the sum E0^(1)+E0^(2) runs from 0 to "
                      "2 = M_pair: the copy rotation changes the per-copy rest energies and their sum, at fixed M_pair, T_sum, q_tot")


# ===================================================================== x6-8 Example d6:ex:R1
def interior_gens(k):
    idx = [0, 3, 4, 5]          # e1, e4, e5, e6 of copy k
    gens = []
    for a, b in itertools.combinations(idx, 2):
        gens.append(embed(Rab(a, b), k))
    return gens


def check_R1():
    def R1(phi):
        R = np.eye(12)
        for i, j in ((0, 6), (3, 9)):
            R[i, i] = R[j, j] = np.cos(phi)
            R[j, i] = np.sin(phi)
            R[i, j] = -np.sin(phi)
        return R
    X = np.zeros((12, 12))
    for i, j in ((0, 6), (3, 9)):
        X[j, i], X[i, j] = 1, -1
    phi = 0.73
    R = R1(phi)
    ok = np.allclose(R.T @ R, np.eye(12)) and np.allclose(R.T @ eta2 @ R, eta2) and np.allclose(R @ G2, G2 @ R)
    ok &= np.allclose(expm(phi * X), R)
    cent = commutant_in(SO66, [G2])
    ok &= in_span(X, cent)
    g_diag = [embed(r, 0) + embed(r, 1) for r in LG]
    ok &= max(np.abs(X @ g - g @ X).max() for g in g_diag) > 0.5
    ok &= not in_span(X, [Ecr, Ecr @ G2])
    # commutants of the k_I = 0 actions (Theorem 10.8): g(0,1,1)=so(3,3) (dim 15), g(0,2,0) (dim 2)
    c011 = commutant_in(SO66, [embed(r, 0) for r in LG])
    c020 = commutant_in(SO66, g_diag)
    # extension: the interior actions (1,0,1) (dim 28) and (2,0,0) (dim 7)
    c101 = commutant_in(SO66, interior_gens(0))
    c200 = commutant_in(SO66, [a + b for a, b in zip(interior_gens(0), interior_gens(1))])
    dims = (len(c011), len(c020), len(c101), len(c200))
    ok &= dims == (15, 2, 28, 7)
    notin = [not in_span(X, c) for c in (c011, c020, c101, c200)]
    ok &= all(notin)
    report("8a", ok, f"ex:R1: R^(1)_phi is orthogonal, eta2-isometric, commutes with G2, lies in Cent(G2) cap so(6,6), not in "
                     f"span{{E,EG2}}; commutant dims (0,1,1),(0,2,0),(1,0,1),(2,0,0) = {dims} (Thm 10.8); R^(1) lies in none "
                     f"(stronger than the k_I=0 statement)")
    ok = True
    for phi in np.linspace(0.1, np.pi / 2, 6):
        X0 = np.concatenate([Xf(np.array([1, 1j, 0])), np.zeros(6)])
        Y = R1(phi) @ X0
        a, b = zf(Y[:6]), zf(Y[6:])
        s, c = np.sin(phi), np.cos(phi)
        ok &= np.allclose(a, [c, 1j, 0]) and np.allclose(b, [s, 0, 0])
        ok &= abs(qf(a) + s * s) < 1e-14 and abs(qf(b) - s * s) < 1e-14 and abs(E0f(a) - s * s / 2) < 1e-14
        ok &= abs(E6f(a) + E6f(b) - 1) < 1e-14 and abs(qf(a) + qf(b)) < 1e-14
        ok &= np.allclose(Sf(a) + Sf(b), [0, 0, -c])
        Mf = Mpair_formula(a, b)
        ok &= abs(Mf - s) < 1e-12 and abs(Mpair_numeric(a, b) - s) < 1e-5
        ok &= 1 >= Mf - 1e-14 and Mf >= s * s - 1e-14
    a, b = zf((R1(np.pi / 2) @ np.concatenate([Xf(np.array([1, 1j, 0])), np.zeros(6)]))[:6]), \
        zf((R1(np.pi / 2) @ np.concatenate([Xf(np.array([1, 1j, 0])), np.zeros(6)]))[6:])
    ok &= np.allclose(a, [0, 1j, 0]) and np.allclose(b, [1, 0, 0])
    report("8b", ok, "ex:R1 numbers: on ((1,i,0),0) z1'=(cos,i,0), z2'=(sin,0,0), q=-+sin^2, E0=sin^2/2 each, E6_tot=1, q_tot=0, "
                     "S_sum=-cos e3, M_pair=sin phi, 1 >= sin >= sin^2, equality at pi/2 (electric e2, magnetic e1)")


# ===================================================================== Clifford realisation for x6-9, x6-13
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
ep = np.array([[0, -1], [1, 0]])
I2 = np.eye(2)
k3 = lambda a, b, c: np.kron(np.kron(a, b), c)
e8 = [k3(sx, I2, I2), k3(sz, sx, I2), k3(sz, sz, sx), k3(ep, I2, I2), k3(sz, ep, I2), k3(sz, sz, ep)]
omega8 = e8[0] @ e8[1] @ e8[2] @ e8[3] @ e8[4] @ e8[5]
I8 = np.eye(8)
E64 = [[np.kron(e, I8) for e in e8], [np.kron(omega8, e) for e in e8]]
I64 = np.eye(64)


def vec_action(Bv):
    """12x12 matrix of X -> [Bv, X] on V2 (basis e^(k)_a), via traces."""
    R = np.zeros((12, 12))
    flat = E64[0] + E64[1]
    etas = list(np.diag(eta)) * 2
    for A in range(12):
        Y = Bv @ flat[A] - flat[A] @ Bv
        for C in range(12):
            R[C, A] = np.trace(flat[C] @ Y) / (64 * etas[C])
    return R


def check_gauge():
    ok = True
    flat = E64[0] + E64[1]
    etas = list(np.diag(eta)) * 2
    for A in range(12):
        for B in range(12):
            ok &= np.allclose(flat[A] @ flat[B] + flat[B] @ flat[A], 2 * (etas[A] if A == B else 0) * I64)
    M = 0.5 * sum(eta[a, a] * E64[0][a] @ E64[1][a] for a in range(6))
    adM = vec_action(M)
    ok &= np.allclose(adM, -Ecr)
    ev = np.linalg.eigvals(M)
    ok &= np.allclose(ev.real, 0, atol=1e-10)
    ims = np.round(ev.imag, 8)
    ok &= set(np.unique(ims)) == set(float(k) for k in range(-3, 4)) and np.allclose(ims, np.round(ims))
    U = lambda ph: np.linalg.multi_dot([np.cos(ph / 2) * I64 + np.sin(ph / 2) * eta[a, a] * E64[0][a] @ E64[1][a] for a in range(6)])
    ok &= np.allclose(U(0.9), expm(0.9 * M)) and np.allclose(U(2 * np.pi), I64)
    J1 = E64[0][1] @ E64[0][2]
    J2 = E64[1][1] @ E64[1][2]
    Uq = U(np.pi / 2)
    ok &= np.allclose(Uq @ J1 @ np.linalg.inv(Uq), J2) and np.allclose(Uq @ J2 @ np.linalg.inv(Uq), J1)
    ok &= np.allclose(M @ (J1 + J2) - (J1 + J2) @ M, 0) and np.abs(M @ J1 - J1 @ M).max() > 0.5
    # compact, commuting with G2 and the diagonal l_G: (vector level)
    g_diag = [embed(r, 0) + embed(r, 1) for r in LG]
    ok &= all(np.allclose(Ecr @ g, g @ Ecr) for g in g_diag)
    report("9", ok, "prop:gauge(i): tower Clifford relations; M=1/2 sum eta_aa e^(1)_a e^(2)_a has ad M=-E on V2, eigenvalues "
                    "i{-3..3}; U_phi = product formula = exp(phi M), U_2pi=id; U_{pi/2} exchanges J^(1), J^(2); [M,J1+J2]=0 != [M,J1]")

    # (ii)-(iii)
    ok = True
    z1, z2 = np.array([1, 1j, 0]), np.array([1, -1j, 0])
    # on V_pm the E0^(k) are invariant
    z = rz()
    for sgn in (1, -1):
        w2 = sgn * 1j * z
        for ph in (0.3, 1.1):
            a, b = np.cos(ph) * z - np.sin(ph) * w2, np.sin(ph) * z + np.cos(ph) * w2
            ok &= abs(E0f(a) - E0f(z)) < 1e-12 and abs(E0f(b) - E0f(w2)) < 1e-12
    # equal-inertia closure invariant under O(eta2) cap Cent(G2): random real orthogonal and random complex orthogonal
    nu, m = 0.6 + 0.9j, 1.4
    N2 = np.kron(np.eye(2), Nlor(nu, m))

    def realify(Oc):
        # complex 6x6 acting on Z=(z1,z2) -> 12x12 real on (X1;X2), X_k = (B_k;E_k)
        L = np.zeros((12, 12))
        for col in range(12):
            e = np.zeros(12)
            e[col] = 1
            Z = np.concatenate([zf(e[:6]), zf(e[6:])])
            W = Oc @ Z
            L[:, col] = np.concatenate([Xf(W[:3]), Xf(W[3:])])
        return L
    Aso = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    Oc = expm(0.3 * (Aso - Aso.T))                         # SO(6,C)
    Ao = rng.normal(size=(6, 6))
    Or = expm(0.5 * (Ao - Ao.T))                           # SO(6)
    for Oo in (Oc, Or):
        L = realify(Oo)
        ok &= np.allclose(L.T @ eta2 @ L, eta2) and np.allclose(L @ G2, G2 @ L)
        Lmt = np.linalg.inv(L).T
        ok &= np.allclose(Lmt @ G2.T, G2.T @ Lmt)
        for _ in range(3):
            p = rng.normal(size=12)
            ok &= abs((Lmt @ p) @ N2 @ (Lmt @ p) - p @ N2 @ p) < 1e-9
    # transitivity of SO(6) on {q_tot=0, |Z|=r}: map a random balanced pair to (r/sqrt2 (1,i,0), 0)
    u = rng.normal(size=6)
    v = rng.normal(size=6)
    v -= (u @ v) / (u @ u) * u
    v *= np.linalg.norm(u) / np.linalg.norm(v)
    Z = u + 1j * v
    ok &= abs(Z @ Z) < 1e-12
    Bm = np.column_stack([u / np.linalg.norm(u), v / np.linalg.norm(v), rng.normal(size=(6, 4))])
    Qm, _ = np.linalg.qr(Bm)
    Qm[:, 0] *= np.sign(Qm[:, 0] @ u)
    Qm[:, 1] *= np.sign(Qm[:, 1] @ v)
    O = Qm.T
    if np.linalg.det(O) < 0:
        O[5] *= -1
    Zt = O @ Z
    r = np.linalg.norm(Z)
    ok &= np.allclose(Zt, r / np.sqrt(2) * np.array([1, 1j, 0, 0, 0, 0])) and abs(np.linalg.det(O) - 1) < 1e-10
    ok &= abs(Mpair_formula(Zt[:3], Zt[3:])) < 1e-12 and abs(sum(E6f(x) for x in (Zt[:3], Zt[3:])) - 0.5 * r * r) < 1e-12
    # the example (e1, i e1) -> ((1,i,0),0): M_pair 1 -> 0
    e1 = np.array([1, 0, 0], complex)
    ok &= abs(Mpair_formula(e1, 1j * e1) - 1) < 1e-14 and Mpair_formula(np.array([1, 1j, 0]), 0 * e1) < 1e-14
    P = np.eye(6)
    P[[1, 3]] = P[[3, 1]]
    P[3] *= -1                                     # rotation e4 -> e2 in R^6 (index 3 -> 1)
    ok &= np.allclose(P @ np.array([1, 0, 0, 1j, 0, 0]), [1, 1j, 0, 0, 0, 0]) and abs(np.linalg.det(P) - 1) < 1e-12
    # (iv) unequal c: no nonzero s E + t E G2 preserves the inertia; equal c: all do
    def preserving_dim(c1, c2):
        Nd = np.zeros((12, 12))
        Nd[:6, :6] = Nlor(c1, 1)
        Nd[6:, 6:] = Nlor(c2, 1)
        gens = [Ecr, Ecr @ G2]
        A = np.array([(g @ Nd + Nd @ g.T).ravel() for g in gens]).T
        return 2 - np.linalg.matrix_rank(A, tol=1e-10)
    ok &= preserving_dim(1 + 0.2j, 0.5 - 0.3j) == 0 and preserving_dim(0.7 + 0.1j, 0.7 + 0.1j) == 2
    # per-copy charges not invariant (already), q_k not invariant under R_phi on ex:bw
    ok &= abs(qf(np.cos(0.5) * z1 - np.sin(0.5) * z2)) > 0.1
    report("10", ok, "prop:gauge(ii)-(iv): on V_pm the E0^(k) are R_phi-invariant; equal Lorentz-class inertia invariant under "
                     "random SO(6,C) and SO(6) (Poisson-compatible); SO(6) carries a random balanced pair to (r(1,i,0)/sqrt2, 0); "
                     "(e1,ie1)->((1,i,0),0) with M_pair 1->0; for c1!=c2 no nonzero element of g(0,2,0) preserves the inertia")


# ===================================================================== x6-11 interior with b = beta E
def S_matrix(k1, k2, beta):
    return np.kron(np.diag([k1, k2]), G) + beta * Ecr


def check_interior():
    ok = True
    k1, k2, beta = 1.0, 3.0, 0.4
    S = S_matrix(k1, k2, beta)
    Sd = np.linalg.inv(eta2) @ S.T @ eta2
    ok &= np.allclose(Sd, np.kron(np.diag([k1, k2]), G) - beta * Ecr)
    h = np.array([[k1, -1j * beta], [1j * beta, k2]])
    z1, z2 = np.array([1, 1j, 0]), np.array([1, -1j, 0])
    pi0 = np.concatenate([Xf(z1), Xf(z2)])
    taus = np.linspace(0, 12, 6001)
    e0s = []
    qt = []
    Mp = []
    e6 = []
    for t in taus:
        pit = expm(Sd * t) @ pi0
        a, b = zf(pit[:6]), zf(pit[6:])
        U = expm(1j * h * t)
        if t in taus[::600]:
            ok &= np.allclose(np.array([a, b]), U @ np.array([z1, z2]), atol=1e-10)
            ok &= abs(E0f(a) - abs(z1 @ z2) * abs(U[0, 0] * U[0, 1])) < 1e-10
        e0s.append(E0f(a))
        qt.append(abs(qf(a) + qf(b)))
        Mp.append(Mpair_formula(a, b))
        e6.append(E6f(a) + E6f(b))
    e0s = np.array(e0s)
    j = int(np.argmax(e0s))
    f = lambda t: -E0f(zf((expm(Sd * t) @ pi0)[:6]))
    r = minimize_scalar(f, bounds=(taus[max(j - 2, 0)], taus[min(j + 2, len(taus) - 1)]), method='bounded',
                        options={'xatol': 1e-12})
    peak = -r.fun
    s = 4 * beta ** 2 / ((k2 - k1) ** 2 + 4 * beta ** 2)
    ok &= abs(peak - 2 * np.sqrt(s * (1 - s))) < 1e-9 and abs(round(peak, 4) - 0.6897) < 1e-12
    ok &= np.ptp(e6) < 1e-10 and np.ptp(Mp) < 1e-9
    # equal rates: exp(i kappa tau) R_{-beta tau}
    Sd_eq = np.linalg.inv(eta2) @ S_matrix(2.0, 2.0, 0.7).T @ eta2
    for t in (0.4, 1.7):
        Y = expm(Sd_eq * t) @ pi0
        expect = np.exp(2j * t) * (expm(-0.7 * t * eps2) @ np.array([z1, z2]))
        ok &= np.allclose(np.array([zf(Y[:6]), zf(Y[6:])]), expect)
    # from (z,0), z null: both copies stay null
    zn = rnull()
    pin = np.concatenate([Xf(zn), np.zeros(6)])
    for (a1, a2, bb) in ((1, 3, 0.4), (0, 5, 0.5), (2, 0.5, 1.3)):
        Sdx = np.linalg.inv(eta2) @ S_matrix(a1, a2, bb).T @ eta2
        for t in (0.3, 2.2, 7.1):
            Y = expm(Sdx * t) @ pin
            ok &= abs(qf(zf(Y[:6]))) < 1e-10 and abs(qf(zf(Y[6:]))) < 1e-10
    report("11", ok, f"interior b=beta E: S^dag = K(x)G - beta E, z(pi)(tau)=e^(i h tau) z(pi)(0); equal rates give "
                     f"e^(i kappa tau) R_(-beta tau); for ex:bw E0^(1)=|z1.z2||U11 U12| peaks at {peak:.6f} = 2 sqrt(s(1-s)) "
                     f"(kappa=(1,3), beta=0.4); E6_tot and M_pair constant; (z,0) null stays null-null")
    qt = np.array(qt)
    ok2 = qt[0] < 1e-14 and qt.max() > 0.1 and np.ptp(Mp) < 1e-9
    report("11b", ok2, f"[precision for the assessment 'q_tot invariant'] under the unequal-rate interior flow (kappa=(1,3), "
                       f"beta=0.4) |q_tot(pi)| is NOT conserved: it goes from 0 to {qt.max():.4f} for ex:bw, while M_pair "
                       f"and E6_tot are; q_tot is invariant only under the copy rotation itself or at equal rates (up to a phase)")


# ===================================================================== x6-12 Remark d6:rem:copyrotor
def check_copyrotor():
    ok = np.allclose(G2 @ Ecr - Ecr @ G2, 0)      # Jacobi for R^2 x| V2
    # structure constants: basis (g, c, X_1..X_12); Jacobi on random triples
    def br(u, v):
        a1, b1, X1 = u[0], u[1], u[2:]
        a2, b2, X2 = v[0], v[1], v[2:]
        return np.concatenate([[0, 0], (a1 * G2 + b1 * Ecr) @ X2 - (a2 * G2 + b2 * Ecr) @ X1])
    for _ in range(5):
        u, v, w = (rng.normal(size=14) for _ in range(3))
        jac = br(u, br(v, w)) + br(v, br(w, u)) + br(w, br(u, v))
        ok &= np.abs(jac).max() < 1e-12
    I0, Ic = 1.0, 0.8
    nu, m = 0.9 - 0.5j, 1.2
    N2 = np.kron(np.eye(2), Nlor(nu, m))

    def rhs(t, y):
        th, h, thc, hc, p = y[0], y[1], y[2], y[3], y[4:]
        Hh, Hc = h / I0, hc / Ic
        gp = N2 @ p
        return np.concatenate([[Hh, -p @ G2 @ gp, Hc, -p @ Ecr @ gp], (Hh * G2.T + Hc * Ecr.T) @ p])

    def run(p0, h0=1.0, hc0=0.6, T=20.0):
        return solve_ivp(rhs, (0, T), np.concatenate([[0, h0, 0, hc0], p0]), method='LSODA', rtol=1e-12, atol=1e-13,
                         dense_output=True)

    def per_copy(p):
        a = zf(np.linalg.solve(eta, p[:6]))
        b = zf(np.linalg.solve(eta, p[6:]))
        return a, b
    p0 = rng.normal(size=12)
    sol = run(p0)
    ts = np.linspace(0, 20, 201)
    Y = sol.sol(ts)
    hc_dev = np.ptp(Y[3])
    E0_1 = []
    Hs = []
    Mps = []
    qts = []
    Hsum0 = None
    for j in range(len(ts)):
        a, b = per_copy(Y[4:, j])
        E0_1.append(E0f(a))
        Hm = np.outer(a, a.conj()) + np.outer(b, b.conj())
        Hsum0 = Hm if Hsum0 is None else Hsum0
        Hs.append(np.abs(Hm - Hsum0).max())
        Mps.append(Mpair_formula(a, b))
        qts.append(abs(qf(a) + qf(b)))
    ok &= hc_dev < 1e-10 and max(Hs) < 1e-8 and np.ptp(Mps) < 1e-8 and np.ptp(qts) < 1e-8 and np.ptp(E0_1) > 1e-2
    # null-null from ((1,i,0),0)
    pn = np.concatenate([eta @ Xf(np.array([1, 1j, 0])), np.zeros(6)])
    sol = run(pn)
    Yn = sol.sol(ts)
    nn = max(max(abs(qf(per_copy(Yn[4:, j])[0])), abs(qf(per_copy(Yn[4:, j])[1]))) for j in range(len(ts)))
    ok &= nn < 1e-9
    # [correction] the dynamical copy rotor changes the SUM of the copies' rest energies too (ex:bw pair: from 0)
    pb = np.concatenate([eta @ Xf(np.array([1, 1j, 0])), eta @ Xf(np.array([1, -1j, 0]))])
    sol = run(pb, h0=0.0, hc0=0.5)
    Yb = sol.sol(ts)
    sums = [sum(E0f(x) for x in per_copy(Yb[4:, j])) for j in range(len(ts))]
    Mb = [Mpair_formula(*per_copy(Yb[4:, j])) for j in range(len(ts))]
    ok &= abs(sums[0]) < 1e-12 and max(sums) > 1.9 and np.ptp(Mb) < 1e-8
    report("12", ok, f"rem:copyrotor: R^2 x| V2 satisfies Jacobi; with equal Lorentz-class inertias h_c const ({hc_dev:.1e}), "
                     f"H_sum, M_pair, |q_tot| const, per-copy E0 oscillates (range {np.ptp(E0_1):.3f}); ((1,i,0),0) stays "
                     f"null-null; [correction] from the ex:bw pair sum_k E0^(k) goes 0 -> {max(sums):.3f} at fixed M_pair=2")


# ===================================================================== x6-13 Proposition d6:prop:dirac
def check_dirac():
    J = e8[1] @ e8[2]
    J1 = E64[0][1] @ E64[0][2]
    J2 = E64[1][1] @ E64[1][2]
    ok = np.allclose(J1, np.kron(J, I8)) and np.allclose(J2, np.kron(I8, J)) and np.allclose(J1 @ J2, J2 @ J1)
    for k in range(2):
        for a in (0, 3, 4, 5):
            for Jk in (J1, J2):
                ok &= np.allclose(Jk @ E64[k][a], E64[k][a] @ Jk)
    Qtot = -1j * (J1 + J2)
    ev = np.round(np.linalg.eigvals(Qtot).real, 8)
    ok &= {v: int(np.sum(ev == v)) for v in (-2.0, 0.0, 2.0)} == {-2.0: 16, 0.0: 32, 2.0: 16}
    ok &= np.allclose(np.linalg.eigvals(Qtot).imag, 0, atol=1e-9)
    report("13a", ok, "prop:dirac(i): J^(1)=J(x)id, J^(2)=id(x)J, commuting with each other and the eight interior gammas; "
                      "Q_tot=-i(J1+J2) has spectrum -2,0,2 with multiplicities 16,32,16")
    # (ii)
    m = 1.7
    H0 = m * e8[0]
    A = np.vstack([e8[0] - I8, J - 1j * I8])
    W = null_space(A)
    ok = W.shape[1] == 2
    psi = W @ (rng.normal(size=2) + 1j * rng.normal(size=2))
    phi = W @ (rng.normal(size=2) + 1j * rng.normal(size=2))
    C = e8[1]
    ok &= np.allclose(C @ e8[0], -e8[0] @ C) and np.allclose(C @ J, -J @ C)
    ok &= all(np.allclose(C @ e8[a] @ e8[b], e8[a] @ e8[b] @ C) for a, b in itertools.combinations((0, 3, 4, 5), 2))
    Psi = np.kron(psi, C @ phi)
    ok &= np.allclose(-1j * J1 @ Psi, Psi) and np.allclose(-1j * J2 @ Psi, -Psi) and np.allclose(Qtot @ Psi, 0)
    En = -1j * (np.kron(J @ H0, I8) + np.kron(I8, J @ H0))
    ok &= np.allclose(En @ Psi, 2 * m * Psi)
    # the graded (tower) energy built from T^(k)=e^(k)_1 e^(k)_2 e^(k)_3 gives +-sqrt2 m instead
    T1 = E64[0][0] @ E64[0][1] @ E64[0][2]
    T2 = E64[1][0] @ E64[1][1] @ E64[1][2]
    evg = np.linalg.eigvals(-1j * m * (T1 + T2))
    ok &= np.allclose(np.sort(np.abs(evg.real)), np.sqrt(2) * m) and np.allclose(evg.imag, 0, atol=1e-9)
    report("13b", ok, "prop:dirac(ii): psi, phi at rest (e1 psi=psi, J psi=i psi), C=e2 anticommutes with e1, J and commutes with "
                      "the interior Lorentz bivectors; Psi=psi(x)C phi has charges (1,-1), Q_tot Psi=0, and the ungraded energy "
                      "-i(JH(0)(x)id+id(x)JH(0)) gives 2m (the graded -i m(T^(1)+T^(2)) gives +-sqrt2 m)")
    # (iii)
    C1C2 = E64[0][1] @ E64[1][1]
    ok = np.allclose(C1C2 @ C1C2, -I64) and np.allclose(C1C2 @ J1, -J1 @ C1C2) and np.allclose(C1C2 @ J2, -J2 @ C1C2)
    om1 = np.kron(omega8, I8)
    CC = np.kron(C, C)
    ok &= np.allclose(CC, -om1 @ C1C2) and np.allclose(CC @ CC, I64)
    for sgn in (1, -1):
        Pp = np.kron(psi, C @ psi) + sgn * np.kron(C @ psi, psi)
        ok &= np.allclose(CC @ Pp, sgn * Pp) and np.allclose(Qtot @ Pp, 0) and np.linalg.norm(Pp) > 1e-6
    report("13c", ok, "prop:dirac(iii): (C1C2)^2=-id, C1C2 anticommutes with J1, J2; C(x)C=-omega^(1)C1C2 squares to id; "
                      "Psi_pm eigen with C-parity pm1 and Q_tot=0")
    # (iv)
    Ms = m * (E64[0][0] + E64[1][0])
    ok = np.allclose(Ms @ Ms, 2 * m * m * I64)
    ev = np.linalg.eigvalsh(Ms)
    ok &= np.allclose(np.abs(ev), np.sqrt(2) * m) and np.sum(ev > 0) == 32
    Mt = m * (np.kron(e8[0], I8) + np.kron(I8, e8[0]))
    ok &= np.allclose(Mt, m * (E64[0][0] + om1 @ E64[1][0]))
    ev = np.round(np.linalg.eigvalsh(Mt) / m, 8)
    ok &= {v: int(np.sum(ev == v)) for v in (-2.0, 0.0, 2.0)} == {-2.0: 16, 0.0: 32, 2.0: 16}
    report("13d", ok, "prop:dirac(iv): (e^(1)_1+e^(2)_1)^2=2id, spectrum +-sqrt2 m (32 each); e1(x)id+id(x)e1 = "
                      "e^(1)_1+omega^(1)e^(2)_1 has spectrum -2m,0,2m (16,32,16)")
    # (v) 2pi rotations
    rot = lambda k: 0.5 * E64[k][3] @ E64[k][4]
    one = expm(2 * np.pi * rot(0))
    both = expm(2 * np.pi * (rot(0) + rot(1)))
    surplus = expm(2 * np.pi * 0.5 * E64[0][1] @ E64[0][2])
    ok = np.allclose(one, -I64) and np.allclose(both, I64) and np.allclose(surplus, -I64)
    report("13e", ok, "prop:dirac(v): a 2pi interior rotation is -id on one copy and +id under the diagonal interior action "
                      "(k_I=2), as Prop 10.9 states")


# ===================================================================== x6-14 Proposition d6:prop:zpe
def krein_signatures(S, masses):
    n = S.shape[0]
    Minv = np.diag(np.repeat(1.0 / np.asarray(masses), 6))
    Hes = np.zeros((2 * n, 2 * n))
    Hes[:n, n:] = -S.T
    Hes[n:, :n] = -S
    Hes[n:, n:] = Minv @ np.linalg.inv(eta2)
    Jm = np.zeros((2 * n, 2 * n))
    Jm[:n, n:] = np.eye(n)
    Jm[n:, :n] = -np.eye(n)
    A = Jm @ Hes
    w, V = np.linalg.eig(A)
    out = {}
    idx = [i for i in range(len(w)) if w[i].imag > 1e-7]
    clusters = []
    for i in sorted(idx, key=lambda i: w[i].imag):
        if clusters and abs(w[i] - w[clusters[-1][-1]]) < 1e-6:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    for cl in clusters:
        f = round(float(np.mean([w[i].imag for i in cl])), 6)
        # eigenspace of A at i f (semisimple case): span of the clustered eigenvectors, orthonormalised
        U_, s_, _ = np.linalg.svd(V[:, cl], full_matrices=False)
        Ns = U_[:, s_ > 1e-8 * s_[0]]
        Hm = Ns.conj().T @ Hes @ Ns
        Hm = 0.5 * (Hm + Hm.conj().T)
        ev = np.linalg.eigvalsh(Hm)
        out[f] = (int(np.sum(ev > 1e-9)), int(np.sum(ev < -1e-9)), Ns.shape[1])
    return out, A


def exact_semisimple(k1, k2, beta, m=1):
    k1, k2, beta, m = map(sp.nsimplify, (k1, k2, beta, m))
    Gs = sp.Matrix(G.astype(int))
    S = sp.diag(k1 * Gs, k2 * Gs) + beta * sp.Matrix(Ecr.astype(int))
    eta2s = sp.Matrix(eta2.astype(int))
    Sd = eta2s.inv() * S.T * eta2s
    M = sp.zeros(24, 24)
    M[:12, :12] = -S
    M[:12, 12:] = sp.eye(12) / m
    M[12:, 12:] = Sd
    lam = sp.symbols('lam')
    chi = S.charpoly(lam).as_expr()
    P = sp.expand(chi * chi.subs(lam, -lam))
    Psf = sp.Poly(sp.sqf_part(P), lam)
    # Horner evaluation of Psf at M
    R = sp.zeros(24, 24)
    for c in Psf.all_coeffs():
        R = R * M + c * sp.eye(24)
    return R.is_zero_matrix, sp.factor(chi)


def check_zpe():
    # (i) two binding manifolds without b, unequal rates and masses
    sig, _ = krein_signatures(S_matrix(1.3, 2.1, 0.0), [1.0, 2.7])
    ok_a = sorted(sig) == [1.3, 2.1] and all(v[:2] == (3, 3) and v[2] == 6 for v in sig.values())
    sig_eq, _ = krein_signatures(S_matrix(1.5, 1.5, 0.0), [1.0, 1.0])
    ok_a &= list(sig_eq.values()) == [(6, 6, 12)]
    report("14a", ok_a, f"prop:zpe(i): without b, rates (1.3,2.1), masses (1,2.7): energy-form signatures {sig} = (3,3) per rate, "
                        f"zero-point energy sum hbar kappa/2 (3-3)=0 (at equal rates the common line has (6,6))")
    # (ii) with b: lines, Krein (3,3) each, seesaw
    ok_b = True
    for (a1, a2, bb) in ((1.0, 3.0, 0.4), (0.0, 5.0, 0.5), (2.0, 2.0, 0.3), (1.0, 4.0, 1.0)):
        S = S_matrix(a1, a2, bb)
        mu = 0.5 * (a1 + a2) + np.array([1, -1]) * np.sqrt(0.25 * (a1 - a2) ** 2 + bb ** 2)
        evS = np.linalg.eigvals(S)
        want = np.sort(np.concatenate([np.repeat(1j * mu, 3), np.repeat(-1j * mu, 3)]).imag)
        ok_b &= np.allclose(np.sort(evS.imag), want, atol=1e-9) and np.allclose(evS.real, 0, atol=1e-9)
        sig, A = krein_signatures(S, [1.0, 1.0])
        ok_b &= all(v[:2] == (3, 3) for v in sig.values()) and len(sig) == 2
    mum = 2.5 - np.sqrt(6.5)
    ok_b &= abs(abs(mum) - 0.0495097568) < 5e-11 and abs(abs(mum) - 0.05) < 1e-3
    report("14b", ok_b, f"prop:zpe(ii): spec S = pm i mu_pm (3 each) at four parameter points; Krein (3,3) on each line; seesaw "
                        f"|mu_-|={abs(mum):.10f} at (0,5,1/2) against beta^2/kappa2=0.05")
    # (iii) exact semisimplicity
    cases = {(1, 4, 2): False, (1, 4, 1): True, (2, 2, 2): True, (sp.Rational(1), 3, sp.Rational(2, 5)): True}
    ok_c = True
    res = {}
    for (a1, a2, bb), want in cases.items():
        ss, _ = exact_semisimple(a1, a2, bb)
        res[(a1, a2, bb)] = ss
        ok_c &= ss == want
    # linear growth at (1,4,2)
    _, A = krein_signatures(S_matrix(1, 4, 2), [1.0, 1.0])
    n1 = np.linalg.norm(expm(A * 100.0), 2)
    n2 = np.linalg.norm(expm(A * 200.0), 2)
    ok_c &= 1.8 < n2 / n1 < 2.2
    report("14c", ok_c, f"prop:zpe(iii) exactly over Q: semisimple {res} (False only at beta^2=kappa1 kappa2, kappa1!=kappa2); "
                        f"at (1,4,2) ||e^(M tau)|| doubles from tau=100 to 200 (ratio {n2 / n1:.3f}): linear growth")
    # [correction] the assessment's condition 'beta^2 = kappa1 kappa2 != kappa1^2' misses unbounded cases
    ok_d = True
    extra = {}
    for (a1, a2, bb) in ((0, 3, 0), (0, 0, sp.Rational(1, 2)), (3, 0, 0)):
        ss, _ = exact_semisimple(a1, a2, bb)
        _, A = krein_signatures(S_matrix(float(a1), float(a2), float(bb)), [1.0, 1.0])
        grow = np.linalg.norm(expm(A * 200.0), 2) / np.linalg.norm(expm(A * 100.0), 2)
        extra[(a1, a2, bb)] = (ss, round(grow, 2))
        ok_d &= (not ss) and grow > 1.5
    report("14d", ok_d, f"[correction to the assessment 'bounded except at beta^2=kappa1 kappa2 != kappa1^2'] unbounded also at "
                        f"(0,3,0) and (0,0,1/2), which that wording admits ((3,0,0) is the mirror case it does catch); (semisimple, "
                        f"growth ratio) = {extra}; the correct "
                        f"exception set (d5:thm:bfield2(v)) is beta^2=kappa1 kappa2 with kappa1!=kappa2, or kappa1=kappa2=0")


# ===================================================================== x6-15 group-level metric copy maps: O(2), not SO(2)
def check_O2():
    swap = np.kron(np.array([[0, 1], [1, 0]]), I6)
    refl = np.kron(np.diag([1, -1]), I6)
    g_diag = [embed(r, 0) + embed(r, 1) for r in LG]
    ok = True
    for Mx in (swap, refl):
        ok &= np.allclose(Mx.T @ eta2 @ Mx, eta2) and np.allclose(Mx.T @ Mx, np.eye(12)) and np.allclose(Mx @ G2, G2 @ Mx)
        ok &= all(np.allclose(Mx @ g, g @ Mx) for g in g_diag)
        ok &= not any(np.allclose(Mx, expm(ph * Ecr)) for ph in np.linspace(0, 2 * np.pi, 721))
        ok &= abs(np.linalg.det(Mx[[0, 6]][:, [0, 6]]) + 1) < 1e-12
    report("15", ok, "[correction to the assessment 'only the copy rotation SO(2) is a metric transformation'] the swap and "
                     "diag(1,-1) on the copy index are eta2-isometric, energy-preserving, commute with G2 and the diagonal l_G, "
                     "and are not copy rotations: U(2) cap O(eta2) = O(2), whose identity component is SO(2)")


# ===================================================================== x6-17 balanced pair from ONE massive field (non-metric U(2))
def check_massive_split():
    ok = True
    for _ in range(4):
        z = rz()
        a, b = z / np.sqrt(2), 1j * z / np.sqrt(2)          # iota_{alpha, i alpha}, |alpha|^2+|beta|^2 = 1
        ok &= abs(E6f(a) + E6f(b) - E6f(z)) < 1e-12 and abs(qf(a) + qf(b)) < 1e-12
        ok &= abs(E0f(a) - 0.5 * E0f(z)) < 1e-12 and abs(E0f(b) - 0.5 * E0f(z)) < 1e-12 and E0f(z) > 1e-3
        ok &= np.allclose(Tmix(a) + Tmix(b), Tmix(z)) and abs(Mpair_formula(a, b) - E0f(z)) < 1e-12
        # it is A (x) id3 with A in U(2) (relative phase after an equal split), not an eta2-isometry
        A = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)           # unitary, first column (1, i)/sqrt2
        ok &= np.allclose(A.conj().T @ A, np.eye(2)) and not np.allclose(A.T @ A, np.eye(2))
        # the admissible copy rotation never balances (z,0) with q(z) != 0: q_tot = q(z) is R_phi-invariant
        ok &= all(abs(qf(np.cos(p) * z) + qf(np.sin(p) * z) - qf(z)) < 1e-12 for p in np.linspace(0, np.pi, 7))
    report("17", ok, "[precision for the assessment 'a balanced pair requires a second null field or a non-covariant agent'] "
                     "from ONE massive field the covariant, energy-preserving but non-metric duplication z -> (z, iz)/sqrt2 "
                     "(d5:thm:dup(ii), U(2)) gives a balanced magnetic-electric pair with E0 = E0(z)/2 each; the admissible "
                     "R_phi never balances (z,0) since q_tot = q(z) is invariant; the claim holds as stated only from a null field")


def check_binding_commutator():
    ok = True
    for (k1, k2) in ((1.0, 1.0), (1.0, 3.0), (0.0, 2.0)):
        KG = np.kron(np.diag([k1, k2]), G)
        Cm = KG @ Ecr - Ecr @ KG
        ok &= np.allclose(Cm, (k2 - k1) * np.kron(np.array([[0, 1], [1, 0]]), G))
        ok &= (np.abs(Cm).max() < 1e-14) == (k1 == k2)
    ok &= np.allclose(G2 @ Ecr, Ecr @ G2)
    report("18", ok, "[correction to the assessment 'it commutes with the binding'] E commutes with the binding root G2, but "
                     "[K(x)G, E] = (kappa2-kappa1) sigma_x(x)G: it commutes with the binding K(x)G only at equal rates "
                     "(d5:thm:bfield2(iii))")


# ===================================================================== x6-16 citations against paper_labels.txt
def check_citations():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))
    path = os.path.join(root, 'paper_labels.txt')
    texroot = os.path.abspath(os.path.join(root, '..', '..', 'tex'))
    want = {'prop:several': '17.18', 'cor:constrate': '17.14', 'thm:closureunique': '17.7', 'prop:LP': '17.8',
            'thm:inertiareq': '17.11', 'thm:drift': '17.12', 'thm:pendulum': '17.13', 'thm:freedrift': '17.16',
            'thm:so9C': '17.19', 'thm:nocommon': '17.20', 'thm:inertia3': '17.24', 'prop:lGhoms': '17.4',
            'thm:algebra': '10.8', 'prop:orthospin': '10.9', 'prop:commutantsn': '13.14', 'thm:sig': '2.3',
            'thm:C': '5.1', 'thm:subfs': '18.6', 'eq:tower': '2', 'eq:bladecomm': '3', 'eq:Hp': '8'}
    ok = os.path.exists(path)
    got = {}
    if ok:
        for line in open(path):
            parts = line.split()
            if len(parts) >= 2:
                got[parts[0]] = parts[1]
        ok &= all(got.get(k) == v for k, v in want.items())
    s17 = os.path.join(texroot, 's17_closure.tex')
    if os.path.exists(s17):
        txt = open(s17).read()
        i = txt.find('label{prop:several}')
        seg = txt[i:i + 1200]
        ok &= 'p^{(2)}=-Gp^{(1)}' in seg and 'w_2=-w_1' in seg and 'every frame moves on a bounded orbit' in seg
        j = txt.find('label{thm:nocommon}')
        seg2 = txt[j:txt.find('end{theorem}', j)]
        ok &= 'meets the interior' in seg2 and 'open' not in seg2
    else:
        ok = False
    report("16", ok, "citations: the theorem/equation numbers used in d6 match paper_labels.txt; Prop 17.18 contains the balance "
                     "surface, the bounded frames and 'p2=-Gp1 gives w2=-w1'; Thm 17.20 is the no-common-Lorentz-algebra result "
                     "(it does not state that the Lorentz action at several constituents is open)")


def timed(f):
    t0 = time.time()
    f()
    if os.environ.get('X6_TIMING'):
        print(f"   [{f.__name__}: {time.time() - t0:.1f} s]")


def main():
    timed(check_conventions)
    timed(check_balance_algebra)
    timed(check_lorentzreq)
    timed(check_balance_dynamics)
    timed(check_nopair)
    timed(check_pairs)
    timed(check_covariant)
    timed(check_pair)
    timed(check_bw)
    timed(check_R1)
    timed(check_gauge)
    timed(check_interior)
    timed(check_copyrotor)
    timed(check_dirac)
    timed(check_zpe)
    timed(check_O2)
    timed(check_citations)
    timed(check_massive_split)
    timed(check_binding_commutator)
    n = len(RESULTS)
    npass = sum(RESULTS)
    print(f"x6: {npass}/{n} checks passed in {time.time() - T_START:.1f} s", flush=True)
    sys.exit(0 if npass == n else 1)


if __name__ == '__main__':
    main()
