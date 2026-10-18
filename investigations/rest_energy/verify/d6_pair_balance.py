#!/usr/bin/env python3
"""Track d6 -- duplication in motion: pair creation and the balance surface.

Verification script for investigations/rest_energy/sections/d6_pair.tex.
Conventions follow the unified paper of 17 October 2026 (tex/s02, s04, s05, s10, s12, s13, s16, s17) and the note's
sections d1, d3, d4:
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1); realisation of tex/s02 eq. (realisation); tower eq. (tower):
  e^(1)_a = e_a (x) 1, e^(2)_a = omega (x) e_a, omega = e_1...e_6;
  binding root G e_i = e_{i+3}, G e_{i+3} = -e_i; z(X) = x + i y for X = sum_j (x_j e_j + y_j e_{j+3}) (G acts as i);
  two constituents: V_2 = V (+) V = R^{6,6}, coordinates (X^(1); X^(2)), eta_2 = 1 (x) eta, G_2 = 1 (x) G,
  copy rotation E = eps (x) 1_6, eps = [[0,-1],[1,0]] on the copy index: z1 -> cos(phi) z1 - sin(phi) z2, z2 -> sin z1 + cos z2;
  Gram matrices Q_kl = z_k . z_l (complex bilinear), H_sum = sum_k z_k z_k^dagger;
  per copy E6 = |z|^2/2, E0 = |q|/2, q = z.z, S = E x B = (i/2) conj(z) x z (frame Poynting vector, minus the textbook one);
  stress tensor T(z) of note section d1 (prop. Rainich): T^00 = E6, T^0i = -S_i, T^ij = E6 delta_ij - E_i E_j - B_i B_j;
  P(t)^mu = T^mu_nu t^nu; M_pair := inf_t |P_sum(t)| over unit future timelike observers t;
  closure (tex/s17): covector p, pi = eta^{-1} p, w(p) = q(z(pi)); s_n = R ghat x|_{G_n} V_n;
  Lorentz-class inertia N_k = (nu_k1 eta^{-1} + nu_k2 G eta^{-1})/m_k, c_k := nu_k/m_k, R_eff e^{i phi_eff} = sum_k c_k w_k(0).
  MOMENTUM field = pi (closure Casimirs), FRAME field = X.
Claims d6-0 ... d6-14 match Table d6:tab:claims of the tex section.
Exact statements use sympy / integer numpy arithmetic; numerical ones use scipy solve_ivp (DOP853, rtol 1e-12, atol 1e-14)
and state their tolerances.
Run from any directory:  python3 d6_pair_balance.py   (exit 0 and one PASS line per claim iff all hold)
"""
import sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm, null_space
from scipy.optimize import minimize_scalar

FAILED = []


def report(cid, ok, text):
    if ok:
        print(f"PASS {cid}: {text}")
    else:
        print(f"FAIL {cid}: {text}")
        FAILED.append(cid)


def info(text):
    print(f"    . {text}")


RTOL, ATOL = 1e-12, 1e-14
rng = np.random.default_rng(20261018)

# ----------------------------------------------------------------------------- Clifford realisation and tower (tex/s02)
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def k3(a, b, c):
    return np.kron(np.kron(a, b), c)


e = [k3(sx, I2, I2), k3(sz, sx, I2), k3(sz, sz, sx), k3(ep, I2, I2), k3(sz, ep, I2), k3(sz, sz, ep)]
eta_d = [1, 1, 1, -1, -1, -1]
I8 = np.eye(8, dtype=np.int64)
I64 = np.eye(64, dtype=np.int64)
OM = e[0] @ e[1] @ e[2] @ e[3] @ e[4] @ e[5]                 # omega = e1...e6 (= Gamma at one constituent)
E1 = [np.kron(x, I8) for x in e]                             # e^(1)_a
E2 = [np.kron(OM, x) for x in e]                             # e^(2)_a  (Klein factor omega on copy 1)
EE = E1 + E2
ETA12 = eta_d + eta_d
J8 = e[1] @ e[2]


def vec_coeffs(Y, gens, etas):
    n = Y.shape[0]
    c = [sp.Rational(int(np.trace(Y @ g)), n * s) for g, s in zip(gens, etas)]
    rec = sum((int(ci) * g for ci, g in zip(c, gens) if ci.q == 1), np.zeros_like(Y))
    if any(ci.q != 1 for ci in c) or not np.array_equal(rec, Y):
        return None
    return c


def ad_vec(Bv, gens, etas, scale=sp.Rational(1, 2)):
    """matrix (columns = images of the generators) of v -> scale*[Bv, v] on the vectors spanned by gens."""
    n = len(gens)
    M = sp.zeros(n, n)
    for a in range(n):
        c = vec_coeffs(Bv @ gens[a] - gens[a] @ Bv, gens, etas)
        assert c is not None
        for b in range(n):
            M[b, a] = scale * c[b]
    return M


ETA = sp.diag(*eta_d)
G = sp.zeros(6, 6)
for i in range(3):
    G[i + 3, i] = 1
    G[i, i + 3] = -1
EPS = sp.Matrix([[0, -1], [1, 0]])


def kron(A, B):
    A, B = sp.Matrix(A), sp.Matrix(B)
    return sp.Matrix(sp.BlockMatrix([[A[i, j] * B for j in range(A.shape[1])] for i in range(A.shape[0])]).as_explicit())


ETA2 = kron(sp.eye(2), ETA)
G2 = kron(sp.eye(2), G)
ECOP = kron(EPS, sp.eye(6))           # copy rotation generator E = eps (x) 1_6
CYC = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
LG = [ad_vec(e[j + 3] @ e[k + 3] - e[j] @ e[k], e, eta_d) for (i, j, k) in CYC] + \
     [ad_vec(e[k] @ e[j + 3] - e[j] @ e[k + 3], e, eta_d) for (i, j, k) in CYC]       # L_i, K_i (note d4 conventions)
LGD = [kron(sp.eye(2), A) for A in LG]                     # diagonal Lorentz algebra lG_Delta on V_2
LG1 = [kron(sp.diag(1, 0), A) for A in LG]                  # lG on copy 1 only

Gn, ETAn = np.array(G.tolist(), dtype=float), np.diag(np.array(eta_d, dtype=float))
G2n, ETA2n, ECOPn = (np.array(M.tolist(), dtype=float) for M in (G2, ETA2, ECOP))
LGn = [np.array(A.tolist(), dtype=float) for A in LG]


# ----------------------------------------------------------------------------- field helpers
def q(z):
    return complex(np.dot(z, z))


def E6(z):
    return 0.5 * float(np.vdot(z, z).real)


def E0(z):
    return 0.5 * abs(q(z))


def Svec(z):
    return (0.5j * np.cross(np.conj(z), z)).real            # = E x B


gmin = np.diag([1.0, -1.0, -1.0, -1.0])


def Tmat(z):
    B, Ee = z.real, z.imag
    M = np.zeros((4, 4))
    M[0, 0] = E6(z)
    M[0, 1:] = -Svec(z)
    M[1:, 0] = -Svec(z)
    M[1:, 1:] = E6(z) * np.eye(3) - np.outer(Ee, Ee) - np.outer(B, B)
    return M


def Pobs(Tm, t):
    return Tm @ gmin @ t


def Llen(P):
    return float(np.sqrt(max(P @ gmin @ P, 0.0)))


def gram(z1, z2):
    return np.array([[np.dot(z1, z1), np.dot(z1, z2)], [np.dot(z2, z1), np.dot(z2, z2)]])


def Mpair_formula(z1, z2):
    s = np.linalg.svd(gram(z1, z2), compute_uv=False)
    return 0.5 * float(s[0] + s[1])


def Rphi(z1, z2, phi):
    c, s = np.cos(phi), np.sin(phi)
    return c * z1 - s * z2, s * z1 + c * z2


def randz():
    return rng.normal(size=3) + 1j * rng.normal(size=3)


def nullz():
    u = rng.normal(size=3)
    v = rng.normal(size=3)
    v -= u * np.dot(u, v) / np.dot(u, u)
    v *= np.linalg.norm(u) / np.linalg.norm(v)
    return (u + 1j * v) * np.exp(1j * rng.uniform(0, 6.28)) * rng.uniform(0.5, 2.0)


def zf(p):                       # z of the momentum field pi = eta^{-1} p of a covector p in R^6
    pi = ETAn @ p
    return pi[:3] + 1j * pi[3:]


def wf(p):
    return q(zf(p))


def p_from_z(z):                  # covector p whose momentum field has z(pi) = z
    return ETAn @ np.concatenate([z.real, z.imag])


def Nlor(nu, m):
    return (nu.real * ETAn + nu.imag * Gn @ ETAn) / m


def expGn(t):
    return np.cos(t) * np.eye(6) + np.sin(t) * Gn


# ============================================================================ d6-0 conventions
def claim_0():
    ok = all(np.array_equal(EE[a] @ EE[b] + EE[b] @ EE[a], 2 * ETA12[a] * I64 * (a == b)) for a in range(12) for b in range(12))
    ok &= G * G == -sp.eye(6) and G.T * ETA * G == -ETA and (ETA.inv() * G.T * ETA) == G
    # G acts as i on z
    X = rng.normal(size=6)
    zX = X[:3] + 1j * X[3:]
    GX = Gn @ X
    ok &= np.allclose(GX[:3] + 1j * GX[3:], 1j * zX)
    # w(p) = C - iD = q(z(pi))
    p = rng.normal(size=6)
    C, D = p @ ETAn @ p, p @ Gn @ ETAn @ p
    ok &= abs(wf(p) - (C - 1j * D)) < 1e-12
    # E commutes with G_2, preserves eta_2, E^2 = -1; eta_2 has signature (6,6)
    ok &= ECOP * G2 == G2 * ECOP and ECOP.T * ETA2 * ECOP == ETA2 and ECOP * ECOP == -sp.eye(12)
    ok &= sorted(ETA2.diagonal()) == [-1] * 6 + [1] * 6
    # the spin generator M = (1/2) sum_a eta_aa e^(1)_a e^(2)_a acts on the 12 tower vectors as the copy rotation
    Mspin = sum(ETA12[a] * E1[a] @ E2[a] for a in range(6))          # = 2 M (integer)
    Ad = ad_vec(Mspin, EE, ETA12)                                    # [M, v] with M = Mspin/2
    ok &= Ad == -ECOP or Ad == ECOP
    sgn = 1 if Ad == ECOP else -1
    report("d6-0", ok, "conventions: tower at n=2 (12 anticommuting generators, eta_2 of signature (6,6)); G acts as i; "
           f"w = q(z(pi)); E = eps(x)1_6 commutes with G_2, preserves eta_2; ad of (1/2)sum eta_aa e^(1)_a e^(2)_a is {'+' if sgn > 0 else '-'}E")
    return sgn


# ============================================================================ d6-1 constant rate with massive constituents
def claim_1():
    # (a) n = 1, Lorentz class: (1/2) p^T N p = Re(nu w(p))/(2m)  (symbolic), so U(theta) = (R0/2m) cos(2theta+phi0),
    #     and the rate is constant & nonzero only if R0 = |nu||w| = 0, i.e. E0(pi) = 0 (Corollary 17.14).
    ps = sp.symbols('p0:6', real=True)
    n1, n2, m = sp.symbols('nu1 nu2 m', real=True)
    P = sp.Matrix(ps)
    N = (n1 * ETA.inv() + n2 * G * ETA.inv()) / m
    pi = ETA.inv() * P
    z = [pi[i] + sp.I * pi[i + 3] for i in range(3)]
    w = sp.expand(sum(zz ** 2 for zz in z))
    lhs = sp.expand((P.T * N * P)[0] / 2)
    rhs = sp.expand(sp.re(sp.expand((n1 + sp.I * n2) * w)) / (2 * m))
    ok_a = sp.simplify(lhs - rhs) == 0
    report("d6-1a", ok_a, "n=1, Lorentz class: p^T N p/2 = Re(nu w(p))/(2m) (exact); with nu != 0 a constant nonzero rate forces "
           "w(p0) = 0, i.e. E0(pi) = 0 (Cor. 17.14 of ES17)")

    # (b) n >= 2: F(p) = sum_k c_k w(p_k); dF has real rank 2 at every nonzero point; the balance set has dim 6n-2 and
    #     the all-null part dim 4n (< 6n-2 iff n >= 2); explicit massive points.
    ok_b = True
    for n in (1, 2, 3):
        cs = rng.normal(size=n) + 1j * rng.normal(size=n)
        for trial in range(4):
            # a random point of the balance set with all constituents massive (n >= 2), or the null cone (n = 1)
            if n == 1:
                zs = [nullz()]
            else:
                zs = [randz() for _ in range(n - 1)]
                target = -sum(c * q(zz) for c, zz in zip(cs[:-1], zs)) / cs[-1]
                A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
                zs.append(np.sqrt(target) * expm(0.4 * (A - A.T)) @ np.array([1, 0, 0]))
            pvec = np.concatenate([p_from_z(zz) for zz in zs])
            F = sum(c * wf(pvec[6 * k:6 * k + 6]) for k, c in enumerate(cs))
            Jac = np.zeros((2, 6 * n))
            for j in range(6 * n):
                d = np.zeros(6 * n)
                d[j] = 1.0
                dF = sum(c * 2 * np.dot(zf(pvec[6 * k:6 * k + 6]), zf(d[6 * k:6 * k + 6])) for k, c in enumerate(cs))
                Jac[:, j] = [dF.real, dF.imag]
            ok_b &= abs(F) < 1e-10 and np.linalg.matrix_rank(Jac, tol=1e-9) == 2
            if n >= 2:
                ok_b &= all(E0(zz) > 1e-3 for zz in zs)
    # null cone of one constituent: w = 0 is a 4-dimensional cone (rank of dw at a nonzero null point = 2)
    ok_b &= all(4 * n < 6 * n - 2 for n in (2, 3, 4, 5)) and not (4 * 1 < 6 * 1 - 2)
    report("d6-1b", ok_b, "n>=2: Sum c_k w_k = 0 is a smooth codim-2 set away from 0 (dF of real rank 2 at random points, n=1,2,3), "
           "the all-null part has dim 4n < 6n-2, so massive balanced data are open dense in it; at n=1 it is the null cone")

    # (c) without the Lorentz requirement the n=1 statement fails: N = diag(1,-1,0,-1,1,0)/m anticommutes with G, is not
    #     lG-invariant, and p0 = e^1 + e^2 (E0(pi0) = 1) has a constant rate and a bounded frame.
    Nn = np.diag([1.0, -1.0, 0.0, -1.0, 1.0, 0.0])
    ok_c = np.allclose(Gn @ Nn, -Nn @ Gn)
    ok_c &= any(np.max(abs(r @ Nn + Nn @ r.T)) > 0.5 for r in LGn)
    p0 = np.array([1.0, 1.0, 0, 0, 0, 0])
    ok_c &= abs(E0(zf(p0)) - 1.0) < 1e-15
    I0, h0 = 1.0, 1.1

    def rhs(t, y):
        th, h, p, x = y[0], y[1], y[2:8], y[8:14]
        return np.concatenate([[h / I0, -p @ Gn @ Nn @ p], (h / I0) * Gn.T @ p, expGn(th) @ Nn @ p])
    sol = solve_ivp(rhs, (0, 60), np.concatenate([[0, h0], p0, np.zeros(6)]), method='DOP853', rtol=RTOL, atol=ATOL)
    dh = np.max(abs(sol.y[1] - h0))
    xmax = np.max(np.linalg.norm(sol.y[8:14], axis=0))
    ok_c &= dh < 1e-12 and xmax <= np.linalg.norm(Nn @ p0) / (h0 / I0) + 1e-9
    report("d6-1c", ok_c, f"the Lorentz requirement is essential: n=1 with the non-Lorentz anticommuting N=diag(1,-1,0,-1,1,0), "
           f"p0=e^1+e^2 (E0(pi)=1): rate constant to {dh:.1e}, frame bounded (max |x| = {xmax:.3f})")


# ============================================================================ d6-2 integrating the two-constituent closure
def closure_rhs(t, y, I0, Ns):
    n = len(Ns)
    th, h = y[0], y[1]
    ps = [y[2 + 6 * k:8 + 6 * k] for k in range(n)]
    out = np.zeros_like(y)
    out[0] = h / I0
    out[1] = -sum(ps[k] @ Gn @ Ns[k] @ ps[k] for k in range(n))
    eG = expGn(th)
    for k in range(n):
        out[2 + 6 * k:8 + 6 * k] = (h / I0) * Gn.T @ ps[k]
        out[2 + 6 * n + 6 * k:8 + 6 * n + 6 * k] = eG @ Ns[k] @ ps[k]
    return out


def run_closure(I0, Ns, ps, h0, T=60.0):
    n = len(Ns)
    y0 = np.concatenate([[0.0, h0]] + list(ps) + [np.zeros(6)] * n)
    return solve_ivp(closure_rhs, (0, T), y0, args=(I0, Ns), method='DOP853', rtol=RTOL, atol=ATOL, dense_output=True)


def frame_closed_form(kap, N, pk, t):
    return (expm(2 * kap * t * Gn) - np.eye(6)) @ np.linalg.inv(2 * kap * Gn) @ N @ pk


def claim_2():
    I0 = 1.0
    ts = np.linspace(0, 60, 301)
    # (a) equal inertias, p2 = -G p1
    nu, m = 1.0 + 0.3j, 1.0
    N = Nlor(nu, m)
    p1 = rng.normal(size=6)
    p2 = -Gn @ p1
    h0 = 1.3
    sol = run_closure(I0, [N, N], [p1, p2], h0)
    dh = np.max(abs(sol.y[1] - h0))
    xerr = max(np.max(abs(sol.sol(t)[14 + 6 * k:20 + 6 * k] - frame_closed_form(h0 / I0, N, pk, t)))
               for t in ts for k, pk in enumerate([p1, p2]))
    ok = dh < 1e-10 and xerr < 1e-9 and abs(wf(p1) + wf(p2)) < 1e-12 and E0(zf(p1)) > 0.1
    report("d6-2a", ok, f"equal inertias, p2=-Gp1 (E0(pi^(k))={E0(zf(p1)):.3f} each): |h-h0| <= {dh:.1e} on [0,60], frames on the "
           f"closed-form circles to {xerr:.1e}")
    # (b) equal inertias, generic balanced pair a.b = 0 with a, b != 0
    a = randz()
    b = randz()
    b -= a * np.dot(a, b) / np.dot(a, a)              # complex-bilinear projection: a.b = 0
    z1, z2 = a + b, 1j * (a - b)
    p1, p2 = p_from_z(z1), p_from_z(z2)
    sol = run_closure(I0, [N, N], [p1, p2], 0.8)
    dh = np.max(abs(sol.y[1] - 0.8))
    ok = abs(np.dot(a, b)) < 1e-12 and abs(q(z1) + q(z2)) < 1e-12 and dh < 1e-10 and E0(z1) > 0.1
    ok &= abs(E0(z1) - E0(z2)) < 1e-12 and abs(E6(z1) - E6(z2)) > 1e-3
    report("d6-2b", ok, f"equal inertias, generic balanced pair (a.b=0, a,b != 0; equal E0, unequal E6): |h-h0| <= {dh:.1e}")
    # (c) unequal inertias, z2 = rho Lambda z1, rho^2 = -c1/c2
    nu1, m1, nu2, m2 = 1.0 + 0.3j, 1.0, 0.4 - 0.7j, 2.5
    N1, N2 = Nlor(nu1, m1), Nlor(nu2, m2)
    c1, c2 = nu1 / m1, nu2 / m2
    z1 = randz()
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Lam = expm(0.3 * (A - A.T))
    rho = np.sqrt(-c1 / c2)
    z2 = rho * Lam @ z1
    p1, p2 = p_from_z(z1), p_from_z(z2)
    Reff = abs(c1 * wf(p1) + c2 * wf(p2))
    h0 = 0.9
    sol = run_closure(I0, [N1, N2], [p1, p2], h0)
    dh = np.max(abs(sol.y[1] - h0))
    xmax = [np.max(np.linalg.norm(sol.y[14 + 6 * k:20 + 6 * k], axis=0)) for k in range(2)]
    bounds = [np.linalg.norm(Nk @ pk) / (h0 / I0) for Nk, pk in ((N1, p1), (N2, p2))]
    ok = Reff < 1e-12 and dh < 1e-10 and all(x <= bd + 1e-9 for x, bd in zip(xmax, bounds))
    ok &= abs(abs(c1) * E0(z1) - abs(c2) * E0(z2)) < 1e-12
    report("d6-2c", ok, f"unequal inertias (m=1, 2.5; nu=1+0.3i, 0.4-0.7i), z2 = rho Lambda z1: R_eff={Reff:.1e}, |h-h0| <= {dh:.1e}, "
           f"frames within their circles; |c1|E0^(1) = |c2|E0^(2)")
    # (d) off the balance surface: the pendulum of Prop. 17.18 and drifting frames
    p2b = p2 + 0.05 * rng.normal(size=6)
    Rc = c1 * wf(p1) + c2 * wf(p2b)
    R, ph = abs(Rc), np.angle(Rc)
    sol = run_closure(I0, [N1, N2], [p1, p2b], h0, T=80.0)
    pend = solve_ivp(lambda t, y: [y[1] / I0, R * np.sin(2 * y[0] + ph)], (0, 80), [0, h0], method='DOP853', rtol=RTOL,
                     atol=ATOL, dense_output=True)
    tt = np.linspace(0, 80, 401)
    dth = np.max(abs(sol.sol(tt)[0] - pend.sol(tt)[0]))
    # energy budget H = K_rot + U, U = (R_eff/2) cos(2 theta + phi_eff), and K_rot sweeps an interval of length R_eff
    tf = np.linspace(0, 80, 40001)
    Y = sol.sol(tf)
    Krot = Y[1] ** 2 / (2 * I0)
    Ufun = 0.5 * R * np.cos(2 * Y[0] + ph)
    Udirect = np.array([0.5 * (Y[2:8, j] @ N1 @ Y[2:8, j] + Y[8:14, j] @ N2 @ Y[8:14, j]) for j in range(0, 40001, 400)])
    dU = np.max(abs(Udirect - Ufun[::400]))
    dH = np.max(abs(Krot + Ufun - (Krot[0] + Ufun[0])))
    Krange = np.ptp(Krot)
    # rotating regime: frames drift (displacement after k periods of theta by pi grows ~ linearly)
    eps_mult = R / (I0 * h0 ** 2)
    disp = []
    for kturn in (1, 2, 4):
        def ev(t, y, I0_, Ns_, kk=kturn):
            return y[0] - kk * np.pi
        ev.terminal = True
        s2 = solve_ivp(closure_rhs, (0, 200), np.concatenate([[0, h0], p1, p2b, np.zeros(12)]), args=(I0, [N1, N2]),
                       method='DOP853', rtol=RTOL, atol=ATOL, events=ev)
        disp.append(np.linalg.norm(s2.y_events[0][0][14:20]))
    ok = dth < 1e-9 and dU < 1e-10 and dH < 1e-10 and abs(Krange - R) < 1e-5 and R > 1e-3 and eps_mult < 0.5
    ok &= np.max(Y[0]) > 2 * np.pi and abs(disp[1] / disp[0] - 2) < 1e-8 and abs(disp[2] / disp[0] - 4) < 1e-8
    report("d6-2d", ok, f"off the balance surface (R_eff={R:.4f}, rotating, eps={eps_mult:.3f}): theta agrees with I0 theta''=R_eff "
           f"sin(2theta+phi_eff) to {dth:.1e}; U = (R_eff/2)cos(2theta+phi_eff) to {dU:.1e}, H to {dH:.1e}, K_rot sweeps R_eff "
           f"(to {abs(Krange - R):.1e}); frame 1 drifts {disp[0]:.4f} per half turn (x2, x4 after 2, 4)")


# ============================================================================ d6-3 the closure creates no pair
def claim_3():
    I0 = 1.0
    M1 = rng.normal(size=(12, 12))
    M1 = (M1 + M1.T) / 10
    M2 = rng.normal(size=(12, 12))
    M2 = (M2 + M2.T) / 40

    def H(h, p):
        return h ** 2 / (2 * I0) + h ** 4 / 10 + np.sin(p @ M1 @ p) + 0.2 * h * (p @ M2 @ p)

    def rhs(t, y):
        th, h, p = y[0], y[1], y[2:14]
        Hh = h / I0 + 0.4 * h ** 3 + 0.2 * (p @ M2 @ p)
        gradp = np.cos(p @ M1 @ p) * 2 * M1 @ p + 0.2 * h * 2 * M2 @ p
        return np.concatenate([[Hh, -p @ G2n @ gradp], Hh * G2n.T @ p])
    p0 = rng.normal(size=12)
    y0 = np.concatenate([[0.0, 1.5], p0])
    sol = solve_ivp(rhs, (0, 10), y0, method='DOP853', rtol=RTOL, atol=ATOL)
    def labels(p):
        z1, z2 = zf(p[:6]), zf(p[6:])
        Hs = np.outer(z1, np.conj(z1)) + np.outer(z2, np.conj(z2))
        return np.array([E0(z1), E0(z2), E6(z1), E6(z2), *Svec(z1), *Svec(z2), *abs(gram(z1, z2)).ravel(),
                         *Hs.real.ravel(), *Hs.imag.ravel(), np.vdot(z1, z2).real, np.vdot(z1, z2).imag])
    L0 = labels(p0)
    dev = max(np.max(abs(labels(sol.y[2:14, j]) - L0)) for j in range(sol.y.shape[1]))
    Q0 = gram(zf(p0[:6]), zf(p0[6:]))
    dQ = max(np.max(abs(gram(zf(sol.y[2:8, j]), zf(sol.y[8:14, j])) - np.exp(2j * sol.y[0, j]) * Q0)) for j in range(sol.y.shape[1]))
    th_end = np.max(abs(sol.y[0]))
    Hdev = max(abs(H(sol.y[1, j], sol.y[2:14, j]) - H(1.5, p0)) for j in range(sol.y.shape[1]))
    ok = dev < 1e-9 and dQ < 1e-9 and Hdev < 1e-9 and th_end > 1
    report("d6-3", ok, f"one rotor, generic copy-coupling Hamiltonian on s_2^*: per-copy E0, E6, S of pi^(k), the Hermitian Gram matrix "
           f"and |Q_kl| constant to {dev:.1e}; Q -> e^(2i theta) Q to {dQ:.1e} (max |theta| = {th_end:.2f}); H to {Hdev:.1e}")


# ============================================================================ d6-4 the balanced pairs
def claim_4(sgn):
    zs = sp.symbols('a0:6 b0:6', real=True)
    z1 = sp.Matrix([zs[0] + sp.I * zs[3], zs[1] + sp.I * zs[4], zs[2] + sp.I * zs[5]])
    z2 = sp.Matrix([zs[6] + sp.I * zs[9], zs[7] + sp.I * zs[10], zs[8] + sp.I * zs[11]])
    a = (z1 - sp.I * z2) / 2
    b = (z1 + sp.I * z2) / 2
    ok_a = sp.expand(z1.dot(z1) + z2.dot(z2) - 4 * a.dot(b)) == 0
    ok_a &= sp.expand(z1.dot(z1) - (a.dot(a) + 2 * a.dot(b) + b.dot(b))) == 0
    report("d6-4a", ok_a, "equal inertias: with a=(z1-iz2)/2, b=(z1+iz2)/2, q1+q2 = 4 a.b and q1 = a.a+2a.b+b.b (exact): "
           "balance <=> a.b = 0, and then q2 = -q1, E0^(1) = E0^(2) = |a.a+b.b|/2")

    # (b) V_pm: eigenspaces of the involution E G_2 = eps (x) G
    EG = ECOP * G2
    ok_b = EG * EG == sp.eye(12) and EG == kron(EPS, G) and EG.T * ETA2 + ETA2 * EG == sp.zeros(12)
    for s in (1, -1):
        V = (EG - s * sp.eye(12)).nullspace()
        Vm = sp.Matrix.hstack(*V)
        ok_b &= len(V) == 6 and (Vm.T * ETA2 * Vm).is_zero_matrix
        # X2 = s G X1 on V_s, and E = -s G_2 there
        ok_b &= all((v[6:, 0] - s * G * v[:6, 0]).is_zero_matrix for v in V)
        ok_b &= all((ECOP * v + s * G2 * v).is_zero_matrix for v in V)
    # momenta: pi in V_+ <=> p2 = -G p1
    p1 = sp.Matrix(sp.symbols('p0:6', real=True))
    pi1 = ETA.inv() * p1
    pi2 = ETA.inv() * (-G * p1)
    ok_b &= (pi2 - G * pi1).is_zero_matrix
    report("d6-4b", ok_b, "E G_2 = eps(x)G is an eta_2-antisymmetric involution (noncompact generator of so(2,C)); its eigenspaces "
           "V_pm = {X2 = pm G X1} are 6-dim and totally eta_2-null; E = -+G_2 on V_pm; pi in V_+ <=> p2 = -G p1 (exact)")

    # (c) p2 = -G p1: w2 = -w1, equal E6, S, E0; T(iz) = T(z); M_pair = 2 E0 while q_tot = 0
    ok_c = True
    for _ in range(5):
        pp = rng.normal(size=6)
        za, zb = zf(pp), zf(-Gn @ pp)
        ok_c &= np.allclose(zb, 1j * za) and abs(q(za) + q(zb)) < 1e-12 and abs(E6(za) - E6(zb)) < 1e-12
        ok_c &= np.allclose(Svec(za), Svec(zb)) and np.allclose(Tmat(zb), Tmat(za))
        ok_c &= abs(Mpair_formula(za, zb) - 2 * E0(za)) < 1e-12
    # duality phase of the pair: if q1 > 0 (magnetic type, |B|>|E|, B.E=0) then q2 = -q1 < 0 (electric type)
    zm = np.array([1.0, 0, 0]) + 1j * np.array([0, 0.5, 0])
    ok_c &= q(zm).real > 0 and q(1j * zm).real < 0 and abs(q(zm).imag) < 1e-15
    report("d6-4c", ok_c, "p2=-Gp1: z2 = i z1, w2 = -w1, equal E6, S, E0, T(z2) = T(z1) (summed tensor 2T(z1)), so M_pair = 2E0 "
           "although q_tot = 0; for q1 > 0 (magnetic type) q2 < 0 (electric type)")

    # (d) unequal inertias: z2 = rho Lambda z1 with rho^2 = -c1/c2 balances, E0 ratio |c1/c2|
    ok_d = True
    for _ in range(4):
        c1, c2 = rng.normal() + 1j * rng.normal(), rng.normal() + 1j * rng.normal()
        z1 = randz()
        A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        Lam = expm(0.5 * (A - A.T))
        z2 = np.sqrt(-c1 / c2) * Lam @ z1
        ok_d &= abs(c1 * q(z1) + c2 * q(z2)) < 1e-10 * (1 + abs(c1 * q(z1)))
        ok_d &= abs(E0(z2) / E0(z1) - abs(c1 / c2)) < 1e-10
    report("d6-4d", ok_d, "unequal inertias: z2 = rho Lambda z1 (Lambda in SO(3,C), rho^2 = -c1/c2) is balanced, E0^(2)/E0^(1) = |c1/c2|")

    # (e) Lorentz-class inertia freedom at n = 2: invariant symmetric forms on V_2^* (all anticommute with G_2)
    def inv_forms(gens):
        syms = sp.symbols('n0:78')
        Nf = sp.zeros(12, 12)
        k = 0
        for i in range(12):
            for j in range(i, 12):
                Nf[i, j] = syms[k]
                Nf[j, i] = syms[k]
                k += 1
        eqs = []
        for r in gens:
            eqs += list(r * Nf + Nf * r.T)
        Am = sp.Matrix([[sp.diff(eq, s) for s in syms] for eq in eqs if eq != 0])
        ns = Am.nullspace()
        sols = []
        for v in ns:
            Nv = Nf.subs(dict(zip(syms, list(v))))
            sols.append(Nv)
        return sols
    centG2 = cent_G2_so66()
    dims = []
    allanti = True
    for gens in (centG2, LGD + [kron(sp.diag(0, 1), A) for A in LG], LGD):
        sols = inv_forms(gens)
        dims.append(len(sols))
        allanti &= all((G2 * Nv + Nv * G2).is_zero_matrix for Nv in sols)
    ok_e = dims == [2, 4, 6] and allanti
    report("d6-4e", ok_e, f"invariant symmetric forms on V_2^* (all of Lorentz class): {dims[0]} under Cent(G_2) cap so(6,6), {dims[1]} under "
           f"lG^(1)+lG^(2), {dims[2]} under the diagonal lG_Delta (n=2 analogue of Thm 17.24)")


_CENT = []


def cent_G2_so66():
    """basis of {A in so(6,6): [A, G_2] = 0} (exact)."""
    if _CENT:
        return _CENT
    syms = sp.symbols('x0:144')
    A = sp.Matrix(12, 12, syms)
    eqs = list(A.T * ETA2 + ETA2 * A) + list(A * G2 - G2 * A)
    Am = sp.Matrix([[sp.diff(eq, s) for s in syms] for eq in eqs if eq != 0])
    for v in Am.nullspace():
        _CENT.append(sp.Matrix(12, 12, list(v)))
    return _CENT


def commutant_in_so66(gens):
    syms = sp.symbols('x0:144')
    A = sp.Matrix(12, 12, syms)
    eqs = list(A.T * ETA2 + ETA2 * A)
    for r in gens:
        eqs += list(A * r - r * A)
    Am = sp.Matrix([[sp.diff(eq, s) for s in syms] for eq in eqs if eq != 0])
    return [sp.Matrix(12, 12, list(v)) for v in Am.nullspace()]


def commutant_in_end(gens):
    syms = sp.symbols('x0:144')
    A = sp.Matrix(12, 12, syms)
    eqs = []
    for r in gens:
        eqs += list(A * r - r * A)
    Am = sp.Matrix([[sp.diff(eq, s) for s in syms] for eq in eqs if eq != 0])
    return [sp.Matrix(12, 12, list(v)) for v in Am.nullspace()]


def span_rank(mats):
    return sp.Matrix([list(Mx) for Mx in mats]).rank()


# ============================================================================ d6-5 what a Lorentz-covariant copy operation does
def claim_5():
    # (a) commutants
    cEnd = commutant_in_end(LGD)
    cso = commutant_in_so66(LGD)
    ok_a = len(cEnd) == 8 and all((Mx * G2 - G2 * Mx).is_zero_matrix for Mx in cEnd)
    ok_a &= len(cso) == 2 and span_rank(cso + [ECOP, ECOP * G2]) == 2
    ok_a &= (ECOP.T + ECOP).is_zero_matrix and not ((ECOP * G2).T + ECOP * G2).is_zero_matrix   # E compact, EG_2 a boost
    report("d6-5a", ok_a, "commutant of lG_Delta: 8-dim in End(V_2), all commuting with G_2 (= M_2(C)(x)1_3); in so(6,6) it is "
           "span{E, EG_2} = so(2,C) = g(0,2,0) of Thm 10.8 (Prop. 13.14), compact part SO(2)_cop = {R_phi}")
    # (b) R_phi preserves H_sum, q_tot, det Q, ||Q||_F, singular values; changes q1, q2
    ok_b = True
    for _ in range(5):
        z1, z2 = randz(), randz()
        phi = rng.uniform(0.2, 1.3)
        y1, y2 = Rphi(z1, z2, phi)
        Hs = lambda u, v: np.outer(u, np.conj(u)) + np.outer(v, np.conj(v))
        Q0, Q1 = gram(z1, z2), gram(y1, y2)
        ok_b &= np.allclose(Hs(z1, z2), Hs(y1, y2)) and np.allclose(Tmat(z1) + Tmat(z2), Tmat(y1) + Tmat(y2))
        ok_b &= abs(np.trace(Q0) - np.trace(Q1)) < 1e-12 and abs(np.linalg.det(Q0) - np.linalg.det(Q1)) < 1e-10
        ok_b &= np.allclose(np.linalg.svd(Q0, compute_uv=False), np.linalg.svd(Q1, compute_uv=False))
        ok_b &= abs(q(z1) - q(y1)) > 1e-6
    report("d6-5b", ok_b, "R_phi preserves H_sum = sum z_k z_k^dagger (hence every summed density sum_k T_k t), q_tot, det Q and the "
           "singular values of Q; it changes q_1, q_2")
    # (c) covariant maps send (z, 0), z null, to two null constituents
    zz = sp.symbols('u0:3')
    W = sp.Matrix(zz)
    znull = sp.Matrix([1, sp.I, 0])
    stab = sp.Matrix([list(W.cross(znull))]).jacobian(zz)          # w -> w x z ; kernel = Lie algebra of Stab(z)
    ker = stab.nullspace()
    ok_c = len(ker) == 1 and sp.simplify(ker[0].cross(znull)).is_zero_matrix and \
        sp.Matrix.hstack(ker[0], znull).rank() == 1
    # fixed vectors of {w x . : w in C z}: v with z x v = 0  -> C z
    fix = sp.Matrix([list(znull.cross(W))]).jacobian(zz).nullspace()
    ok_c &= len(fix) == 1 and sp.Matrix.hstack(fix[0], znull).rank() == 1
    # linear A (x) 1 with random complex A, and U(2) (x) 1: (z,0) null stays null-null; U(2) keeps H_sum and singular values
    for _ in range(5):
        zn = nullz()
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        y1, y2 = A[0, 0] * zn, A[1, 0] * zn
        ok_c &= abs(q(y1)) < 1e-12 and abs(q(y2)) < 1e-12
        Hh = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        U = expm(1j * (Hh + Hh.conj().T))
        z1, z2 = randz(), randz()
        Zm = np.stack([z1, z2], axis=1)          # 3x2, columns z_k
        Zu = Zm @ U.T
        ok_c &= np.allclose(Zm @ Zm.conj().T, Zu @ Zu.conj().T)
        ok_c &= np.allclose(np.linalg.svd(Zm.T @ Zm, compute_uv=False), np.linalg.svd(Zu.T @ Zu, compute_uv=False))
        ok_c &= abs(np.trace(Zm.T @ Zm) - np.trace(Zu.T @ Zu)) > 1e-6
    # u(2) (x) 1 inside so(6,6) is so(2): the anti-Hermitian 2x2 generators realised on V_2 and intersected with so(6,6)
    i2 = [sp.Matrix([[sp.I, 0], [0, 0]]), sp.Matrix([[0, 0], [0, sp.I]]), sp.Matrix([[0, 1], [-1, 0]]), sp.Matrix([[0, sp.I], [sp.I, 0]])]

    def realify(Mc):        # complex 2x2 acting on copy index, G_2 = i  ->  real 12x12
        return kron(Mc.applyfunc(sp.re), sp.eye(6)) + kron(Mc.applyfunc(sp.im), G)
    us = [realify(Mc) for Mc in i2]
    cs = sp.symbols('c0:4', real=True)
    Agen = sum((c * Mx for c, Mx in zip(cs, us)), sp.zeros(12, 12))
    sol = sp.solve(list(Agen.T * ETA2 + ETA2 * Agen), cs, dict=True)
    free = set(cs) - set(sol[0].keys()) if sol else set(cs)
    Afree = Agen.subs(sol[0]).subs({list(free)[0]: 1}) if len(free) == 1 else sp.zeros(12, 12)
    ok_c &= len(free) == 1 and ((Afree - ECOP).is_zero_matrix or (Afree + ECOP).is_zero_matrix)
    report("d6-5c", ok_c, "Stab_so(3,C)(null z) = C z and its fixed vectors are C z (exact): every lG_Delta-covariant map sends (z,0), "
           "q(z)=0, to two null constituents; A(x)1 (A in M_2(C)) checked; U(2)(x)1 keeps H_sum and sing(Q), moves q_tot; "
           "u(2) cap so(6,6) = R E (end of Sec. 13.5 of ES17)")


# ============================================================================ d6-6 the rest energy of a duplicated field point
def claim_6():
    x = sp.symbols('x0:12', real=True)
    Z1 = sp.Matrix([x[0] + sp.I * x[3], x[1] + sp.I * x[4], x[2] + sp.I * x[5]])
    Z2 = sp.Matrix([x[6] + sp.I * x[9], x[7] + sp.I * x[10], x[8] + sp.I * x[11]])

    def Tsym(zv):
        B = sp.Matrix([sp.re(c) for c in zv])
        Ev = sp.Matrix([sp.im(c) for c in zv])
        e6 = sp.Rational(1, 2) * (B.dot(B) + Ev.dot(Ev))
        Sv = Ev.cross(B)
        M = sp.zeros(4, 4)
        M[0, 0] = e6
        for i in range(3):
            M[0, i + 1] = -Sv[i]
            M[i + 1, 0] = -Sv[i]
            for j in range(3):
                M[i + 1, j + 1] = (e6 if i == j else 0) - Ev[i] * Ev[j] - B[i] * B[j]
        return M
    gs = sp.diag(1, -1, -1, -1)
    T1, T2 = Tsym(Z1), Tsym(Z2)
    Wv = Z1.cross(Z2).applyfunc(sp.expand)
    Q11, Q12, Q22 = (sp.expand(u.dot(v)) for u, v in ((Z1, Z1), (Z1, Z2), (Z2, Z2)))
    F2 = sp.expand(Q11 * sp.conjugate(Q11) + 2 * Q12 * sp.conjugate(Q12) + Q22 * sp.conjugate(Q22))
    Tm = (T1 + T2) * gs
    R = (Tm * Tm - sp.Rational(1, 4) * F2 * sp.eye(4) - Tsym(Wv) * gs).applyfunc(sp.expand)
    TT12 = sp.expand(sum((T1 * gs)[i, j] * (T2 * gs)[j, i] for i in range(4) for j in range(4)))
    ok_a = R.is_zero_matrix and sp.expand(TT12 - Q12 * sp.conjugate(Q12)) == 0
    ok_a &= sp.expand(Wv.dot(Wv) - (Q11 * Q22 - Q12 ** 2)) == 0
    report("d6-6a", ok_a, "exact: T(z1).T(z2) = |z1.z2|^2 and (T(z1)+T(z2))^2 = (1/4)||Q||_F^2 + T(z1 x z2) as mixed tensors, with "
           "(z1 x z2).(z1 x z2) = det Q")
    # (b) M_pair = (s1+s2)/2, attained at the rest frame of w = z1 x z2; every other observer sees more
    ok_b = True
    for _ in range(6):
        z1, z2 = randz(), randz()
        wv = np.cross(z1, z2)
        v = -Svec(wv) / (E6(wv) + E0(wv))
        tw = np.concatenate([[1.0], v]) / np.sqrt(1 - v @ v)
        Ts = Tmat(z1) + Tmat(z2)
        Mp = Mpair_formula(z1, z2)
        ok_b &= abs(Llen(Pobs(Ts, tw)) - Mp) < 1e-10 * (1 + Mp)
        for _ in range(50):
            rv = rng.normal(size=3) * rng.uniform(0, 1.5)
            r = np.linalg.norm(rv)
            t = np.concatenate([[np.cosh(r)], np.sinh(r) * rv / r])
            ok_b &= Llen(Pobs(Ts, t)) >= Mp - 1e-10
    report("d6-6b", ok_b, "M_pair := inf_t |P_sum(t)| = (s1+s2)/2 (s_i singular values of Q), attained in the rest frame of z1 x z2 "
           "(1e-10); 300 random observers see more")
    # (c) max over R_phi of E0^(1)+E0^(2) equals (s1+s2)/2
    ok_c = True
    worst = 0.0
    for _ in range(8):
        z1, z2 = randz(), randz()
        f = lambda ph: -(E0(Rphi(z1, z2, ph)[0]) + E0(Rphi(z1, z2, ph)[1]))
        grid = np.linspace(0, np.pi, 2001)
        k = int(np.argmin([f(g) for g in grid]))
        res = minimize_scalar(f, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, 2000)]), method='bounded',
                              options={'xatol': 1e-13})
        Mp = Mpair_formula(z1, z2)
        worst = max(worst, abs(-res.fun - Mp) / Mp)
        ok_c &= -res.fun <= Mp + 1e-12
    ok_c &= worst < 1e-9
    report("d6-6c", ok_c, f"max over the copy rotation of E0^(1)+E0^(2) = (s1+s2)/2 = M_pair (8 random pairs, rel. dev. {worst:.1e})")
    # (d) the chain, and M_pair >= |q_tot|/2; null pairs: M_pair = |z1.z2| = P1.P2
    ok_d = True
    for _ in range(20):
        z1, z2 = randz(), randz()
        Ts = Tmat(z1) + Tmat(z2)
        t0 = np.array([1.0, 0, 0, 0])
        Mp = Mpair_formula(z1, z2)
        chain = [E6(z1) + E6(z2), Llen(Pobs(Ts, t0)), Mp, E0(z1) + E0(z2)]
        ok_d &= all(chain[i] >= chain[i + 1] - 1e-12 for i in range(3)) and Mp >= 0.5 * abs(q(z1) + q(z2)) - 1e-12
    for _ in range(5):
        z1, z2 = nullz(), nullz()
        P1, P2 = Pobs(Tmat(z1), np.array([1.0, 0, 0, 0])), Pobs(Tmat(z2), np.array([1.0, 0, 0, 0]))
        St1, St2 = -Svec(z1), -Svec(z2)
        cth = St1 @ St2 / (np.linalg.norm(St1) * np.linalg.norm(St2))
        Mp = Mpair_formula(z1, z2)
        ok_d &= abs(Mp - abs(np.dot(z1, z2))) < 1e-12 and abs(P1 @ gmin @ P2 / np.sqrt(E6(z1) * E6(z2)) - Mp) < 1e-10
        ok_d &= abs(np.sqrt(E6(z1) * E6(z2)) * (1 - cth) - Mp) < 1e-10
        ok_d &= abs(Mpair_formula(nullz(), np.zeros(3))) < 1e-12
    report("d6-6d", ok_d, "E6_tot >= |P_sum(t0)| >= M_pair >= E0^(1)+E0^(2), M_pair >= |q_tot|/2 (20 random pairs); two null fields: "
           "M_pair = |z1.z2| = sqrt(E6^(1)E6^(2))(1 - cos angle(S1,S2)); one null field and an empty copy: M_pair = 0")


# ============================================================================ d6-7 Breit-Wheeler at a field point
def claim_7():
    z1 = sp.Matrix([1, sp.I, 0])
    z2 = sp.Matrix([1, -sp.I, 0])
    c = s = 1 / sp.sqrt(2)
    y1, y2 = c * z1 - s * z2, s * z1 + c * z2
    qe = lambda u: sp.simplify(u.dot(u))
    e6 = lambda u: sp.simplify(sum(sp.Abs(ui) ** 2 for ui in u) / 2)
    def Ssym(u):
        return sp.simplify((sp.I / 2) * u.conjugate().cross(u))
    ok = qe(z1) == 0 and qe(z2) == 0 and sp.simplify(Ssym(z1) + Ssym(z2)).is_zero_matrix
    ok &= sp.simplify(y1 - sp.Matrix([0, sp.sqrt(2) * sp.I, 0])).is_zero_matrix and sp.simplify(y2 - sp.Matrix([sp.sqrt(2), 0, 0])).is_zero_matrix
    ok &= qe(y1) == -2 and qe(y2) == 2 and Ssym(y1).is_zero_matrix and Ssym(y2).is_zero_matrix
    ok &= e6(z1) + e6(z2) == 2 and e6(y1) + e6(y2) == 2 and sp.simplify(z1.dot(z2)) == 2
    report("d6-7", ok, "exact: counter-propagating null fields (1,i,0),(1,-i,0) (S_sum=0, E6_tot=2, M_pair=|z1.z2|=2) -> R_(pi/4) -> "
           "pure electric (0,sqrt2 i,0) and pure magnetic (sqrt2,0,0), both at rest, q=-2,+2: E0^(1)+E0^(2) = M_pair = E6_tot = 2")


# ============================================================================ d6-8 the rotation of one direction (the task's example)
def claim_8():
    phi = sp.symbols('phi', real=True)
    # generator: rotate (e^(1)_1, e^(2)_1) and (e^(1)_4, e^(2)_4), indices 0,6 and 3,9 of V_2
    A1 = sp.zeros(12, 12)
    A1[6, 0], A1[0, 6] = 1, -1
    A1[9, 3], A1[3, 9] = 1, -1
    ok = (A1.T + A1).is_zero_matrix and (A1.T * ETA2 + ETA2 * A1).is_zero_matrix and (A1 * G2 - G2 * A1).is_zero_matrix
    Rm = sp.eye(12) + sp.sin(phi) * A1 + (1 - sp.cos(phi)) * A1 * A1
    X0 = sp.Matrix([1, 0, 0, 0, 1, 0] + [0] * 6)            # z1 = (1, i, 0), z2 = 0
    X1 = (Rm * X0).applyfunc(sp.simplify)
    zz1 = sp.Matrix([X1[i] + sp.I * X1[i + 3] for i in range(3)])
    zz2 = sp.Matrix([X1[6 + i] + sp.I * X1[9 + i] for i in range(3)])
    q1, q2 = sp.simplify(zz1.dot(zz1)), sp.simplify(zz2.dot(zz2))
    S1 = ((sp.I / 2) * zz1.conjugate().cross(zz1)).applyfunc(sp.simplify)
    S2 = ((sp.I / 2) * zz2.conjugate().cross(zz2)).applyfunc(sp.simplify)
    e6tot = sp.simplify(sum(sp.Abs(u) ** 2 for u in list(zz1) + list(zz2)) / 2)
    ok &= sp.simplify(zz1 - sp.Matrix([sp.cos(phi), sp.I, 0])).is_zero_matrix and sp.simplify(zz2 - sp.Matrix([sp.sin(phi), 0, 0])).is_zero_matrix
    ok &= sp.simplify(q1 + sp.sin(phi) ** 2) == 0 and sp.simplify(q2 - sp.sin(phi) ** 2) == 0 and sp.simplify(e6tot - 1) == 0
    ok &= sp.simplify(S1 + S2 - sp.Matrix([0, 0, -sp.cos(phi)])).is_zero_matrix
    # M_pair after: singular values of Q' are |sin phi| twice -> M_pair = sin phi (0 < phi < pi)
    Qp = sp.Matrix([[q1, zz1.dot(zz2)], [zz1.dot(zz2), q2]]).applyfunc(sp.simplify)
    ev = (Qp * Qp.conjugate().T).applyfunc(sp.simplify)
    ok &= sp.simplify(ev - sp.sin(phi) ** 2 * sp.eye(2)).is_zero_matrix
    # not Lorentz covariant, not in g(0,2,0), not in the commutant of lG on copy 1 (g(0,1,1) = so(3,3), dim 15)
    ok &= any(not (A1 * r - r * A1).is_zero_matrix for r in LGD)
    ok &= span_rank([A1, ECOP, ECOP * G2]) == 3
    c1 = commutant_in_so66(LG1)
    ok &= len(c1) == 15 and span_rank(c1 + [A1]) == 16
    report("d6-8", ok, "exact: the paired rotation of the e_1/e_4 directions of the two copies (compact, in Cent(G_2) cap so(6,6)) maps "
           "((1,i,0),0) to ((cos,i,0),(sin,0,0)): q = -+sin^2, E6_tot = 1, q_tot = 0, S_sum = -cos(phi) e3, M_pair 0 -> sin(phi); it does not "
           "commute with lG_Delta and lies in neither g(0,2,0) nor g(0,1,1) (dim 15)")


# ============================================================================ d6-9 the closure's own symmetry
def claim_9():
    cent = cent_G2_so66()
    ok_a = len(cent) == 30
    # compact part: antisymmetric elements (paired rotations), dim 15
    compact = [Mx for Mx in cent if (Mx.T + Mx).is_zero_matrix]
    syms = sp.symbols('y0:30')
    Agen = sum((s * Mx for s, Mx in zip(syms, cent)), sp.zeros(12, 12))
    ncomp = len(sp.Matrix([[sp.diff(eq, s) for s in syms] for eq in list(Agen.T + Agen) if eq != 0]).nullspace())
    ok_a &= ncomp == 15
    # equal Lorentz-class inertia N (x) 1 is invariant (Lambda N Lambda^T = N for Lambda = exp(A)) -> r N + N r^T = 0
    nu1, nu2 = sp.Rational(3, 2), sp.Rational(-2, 3)
    N2 = kron(sp.eye(2), nu1 * ETA.inv() + nu2 * G * ETA.inv())
    ok_a &= all((Mx * N2 + N2 * Mx.T).is_zero_matrix for Mx in cent)
    report("d6-9a", ok_a, "Cent(G_2) cap so(6,6) has dim 30 (= so(6,C), n=2 analogue of Thm 17.19), compact part 15 (= so(6)); it "
           "preserves the equal Lorentz-class inertia, so it is a symmetry of the equal-inertia two-constituent closure")
    # (b) an integer element of the compact part maps the balanced pair (e1, i e1) to ((1,i,0), 0)
    O6 = sp.eye(6)
    O6[1, 1], O6[3, 3] = 0, 0
    O6[1, 3], O6[3, 1] = 1, -1                    # epsilon_2 <-> epsilon_4 in C^6 = (z1; z2)
    # realise on V_2: z-components (k, i) -> real indices (B: 6(k)+i, E: 6(k)+3+i)
    idx = lambda c: (6 * (c // 3) + c % 3, 6 * (c // 3) + 3 + c % 3)
    O12 = sp.zeros(12, 12)
    for r in range(6):
        for c in range(6):
            if O6[r, c] != 0:
                (rb, re_), (cb, ce) = idx(r), idx(c)
                O12[rb, cb] = O6[r, c]
                O12[re_, ce] = O6[r, c]
    ok_b = (O12.T * O12 == sp.eye(12)) and (O12.T * ETA2 * O12 == ETA2) and (O12 * G2 == G2 * O12) and O12.det() == 1
    X = sp.Matrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])      # z1 = e1, z2 = i e1
    Y = O12 * X
    zy1 = [Y[i] + sp.I * Y[i + 3] for i in range(3)]
    zy2 = [Y[6 + i] + sp.I * Y[9 + i] for i in range(3)]
    ok_b &= zy1 == [1, sp.I, 0] and zy2 == [0, 0, 0]
    zx1, zx2 = np.array([1, 0, 0], complex), np.array([1j, 0, 0])
    ok_b &= abs(Mpair_formula(zx1, zx2) - 1.0) < 1e-15 and abs(Mpair_formula(np.array([1, 1j, 0]), np.zeros(3))) < 1e-15
    # random balanced pair -> single null constituent by a real orthogonal O (Gram-Schmidt), same E6_tot
    for _ in range(4):
        a = randz()
        b = randz()
        b -= a * np.dot(a, b) / np.dot(a, a)
        Z = np.concatenate([a + b, 1j * (a - b)])
        u, v = Z.real, Z.imag
        ok_b &= abs(u @ u - v @ v) < 1e-10 and abs(u @ v) < 1e-10
        Bm = np.column_stack([u / np.linalg.norm(u), v / np.linalg.norm(v), rng.normal(size=(6, 4))])
        Qm, _ = np.linalg.qr(Bm)
        Qm[:, 0] *= np.sign(Qm[:, 0] @ u)
        Qm[:, 1] *= np.sign(Qm[:, 1] @ v)
        O = Qm.T
        if np.linalg.det(O) < 0:
            O[5] *= -1
        Zn = O @ Z
        ok_b &= np.allclose(Zn[2:], 0) and abs(q(Zn[:3])) < 1e-10 and abs(np.vdot(Zn, Zn) - np.vdot(Z, Z)) < 1e-10
        ok_b &= E0(Z[:3]) > 1e-3
    report("d6-9b", ok_b, "a compact symmetry of the equal-inertia closure (integer, det 1) maps the balanced pair (e1, i e1) "
           "(E0 = 1/2 each, M_pair = 1) to ((1,i,0), 0) (one null constituent, M_pair = 0) at the same E6_tot; random balanced "
           "pairs likewise (SO(6) is transitive on {q_tot = 0, |Z| = r})")
    # (c) unequal inertias: no nonzero element of g(0,2,0) preserves the inertia
    s, t = sp.symbols('s t', real=True)
    gen = s * ECOP + t * ECOP * G2
    ok_c = True
    for (cA, cB, expect) in ((sp.Rational(1), sp.Rational(1), 2), (sp.Rational(1), sp.Rational(5, 2), 0),
                             (sp.Rational(2), sp.Rational(-1, 3), 0)):
        Nw = kron(sp.diag(cA, 0), ETA.inv() + G * ETA.inv() / 3) + kron(sp.diag(0, cB), ETA.inv() + G * ETA.inv() / 3)
        eqs = list(gen * Nw + Nw * gen.T)
        Am = sp.Matrix([[sp.diff(eq, v) for v in (s, t)] for eq in eqs if eq != 0])
        ok_c &= (2 - Am.rank()) == expect
    report("d6-9c", ok_c, "the copy algebra g(0,2,0) preserves Lorentz-class inertias diag(c1,c2) exactly when c1 = c2 "
           "(dimension 2 at c1=c2, 0 at c2/c1 = 5/2, -1/6)")


# ============================================================================ d6-10 spin lift of the copy rotation
def claim_10():
    U = I64.copy()
    for a in range(6):
        U = U @ (I64 + ETA12[a] * E1[a] @ E2[a])
    ok = np.array_equal(U.T @ U, 64 * I64)                    # U_{pi/2} = U/8 orthogonal
    Uq = U.astype(object)
    sgn_ok = all(np.array_equal(U @ E1[a] @ U.T, -64 * E2[a]) or np.array_equal(U @ E1[a] @ U.T, 64 * E2[a]) for a in range(6))
    ok &= sgn_ok
    J1, J2 = E1[1] @ E1[2], E2[1] @ E2[2]
    ok &= np.array_equal(U @ J1 @ U.T, 64 * J2) and np.array_equal(U @ J2 @ U.T, 64 * J1)
    U4 = np.linalg.matrix_power(U, 4)
    ok &= np.array_equal(U4, 8 ** 4 * I64)                    # rotation through 2 pi = +1
    Mspin = sum(ETA12[a] * E1[a] @ E2[a] for a in range(6))  # = 2M
    # eigenvalues of M are i*{-3..3}: (2M) has char. poly prod (x^2 + 4k^2)
    terms = [ETA12[a] * E1[a] @ E2[a] for a in range(6)]
    ok &= all(np.array_equal(T @ T, -I64) for T in terms) and all(np.array_equal(A @ B, B @ A) for A in terms for B in terms)
    ev = np.linalg.eigvals(Mspin.astype(float) / 2)
    ok &= np.allclose(ev.real, 0, atol=1e-9) and np.allclose(ev.imag, np.round(ev.imag), atol=1e-9) and set(np.round(ev.imag).astype(int)) == set(range(-3, 4))
    report("d6-10", ok, "spin lift of the copy rotation (Prop. 10.9 of ES17): U_(pi/2) = (1/8) prod_a (1 + eta_aa e^(1)_a e^(2)_a) is "
           "orthogonal, carries e^(1)_a to -+e^(2)_a and J^(1) <-> J^(2); U_(2pi) = +1; generator eigenvalues i{-3,...,3}")


# ============================================================================ d6-11 distinguished copies in the interior
def claim_11():
    def flow(k1, k2, beta, Z0, ts):
        S = np.array((kron(sp.diag(k1, k2), G) + beta * ECOP).tolist(), dtype=float)
        Sdag = ETA2n @ S.T @ ETA2n
        X0 = np.concatenate([Z0[:3].real, Z0[:3].imag, Z0[3:].real, Z0[3:].imag])
        out = []
        for t in ts:
            X = expm(Sdag * t) @ X0
            out.append((X[:3] + 1j * X[3:6], X[6:9] + 1j * X[9:12]))
        return out
    ts = np.linspace(0, 40, 2001)
    ok = True
    # (i) from (null, 0): both stay null for all tau, any rates and beta
    zn = np.array([1, 1j, 0])
    for (k1, k2, beta) in ((1.0, 3.0, 0.4), (0.0, 5.0, 0.5), (2.0, 2.0, 0.7)):
        tr = flow(k1, k2, beta, np.concatenate([zn, np.zeros(3)]), ts[::50])
        ok &= all(abs(q(a)) < 1e-12 and abs(q(b)) < 1e-12 for a, b in tr)
    # (ii) counter-propagating null pair as momentum field, kappa = (1,3), beta = 0.4: per-copy E0 oscillate, Rabi bound
    k1, k2, beta = 1.0, 3.0, 0.4
    Z0 = np.concatenate([np.array([1, 1j, 0]), np.array([1, -1j, 0])])
    tr = flow(k1, k2, beta, Z0, ts)
    e01 = np.array([E0(a) for a, b in tr])
    e6t = np.array([E6(a) + E6(b) for a, b in tr])
    hU = np.array([[k1, -1j * beta], [1j * beta, k2]])        # z(tau) = exp((iK - beta eps) tau) z(0) = exp(i hU tau) z(0)
    ok &= np.allclose(1j * hU, 1j * np.diag([k1, k2]) - beta * np.array([[0, -1], [1, 0]]))
    U = lambda t: expm(1j * hU * t)
    pred = np.array([2 * abs(U(t)[0, 0]) * abs(U(t)[0, 1]) for t in ts])          # |r| |U11 U12|, r = z1.z2 = 2
    s2 = 4 * beta ** 2 / ((k2 - k1) ** 2 + 4 * beta ** 2)
    rabi_max = 2 * np.sqrt(s2 * (1 - s2)) if s2 < 0.5 else 1.0
    ok &= np.max(abs(e01 - pred)) < 1e-10 and np.max(abs(e6t - 2)) < 1e-10
    ok &= abs(np.max(e01) - rabi_max) < 2e-3 and np.max(e01) <= rabi_max + 1e-12 and np.min(e01) < 1e-3
    # (iii) equal rates: the per-copy oscillation is the gauge rotation R_{-beta tau} (times a common duality phase)
    trE = flow(2.0, 2.0, 0.7, Z0, ts[::40])
    for (a, b), t in zip(trE, ts[::40]):
        g1, g2 = Rphi(Z0[:3], Z0[3:], -0.7 * t)
        ok &= abs(E0(a) - E0(g1)) < 1e-10 and abs(E0(b) - E0(g2)) < 1e-10
    report("d6-11", ok, f"interior with b = beta E: from (null,0) both copies stay null for every (kappa1,kappa2,beta); for kappa=(1,3), "
           f"beta=0.4 the per-copy E0(pi^(k)) of a counter-propagating null pair oscillate as |r||U11 U12| (Rabi max {rabi_max:.4f}, "
           f"E6_tot fixed); at kappa1=kappa2 the oscillation is the gauge rotation R_(-beta tau)")


# ============================================================================ d6-12 a copy rotor
def claim_12():
    I0, Ic = 1.0, 0.7
    nu, m = 1.0 + 0.3j, 1.0
    N2 = np.kron(np.eye(2), Nlor(nu, m))

    def rhs(t, y):
        th, ch, h, hc, p = y[0], y[1], y[2], y[3], y[4:16]
        Np = N2 @ p
        return np.concatenate([[h / I0, hc / Ic, -p @ G2n @ Np, -p @ ECOPn @ Np], (h / I0) * G2n.T @ p + (hc / Ic) * ECOPn.T @ p])
    ok = True
    for Z0 in (np.concatenate([randz(), randz()]), np.concatenate([np.array([1, 1j, 0]), np.zeros(3)])):
        p0 = np.concatenate([p_from_z(Z0[:3]), p_from_z(Z0[3:])])
        sol = solve_ivp(rhs, (0, 30), np.concatenate([[0, 0, 0.9, 0.5], p0]), method='DOP853', rtol=RTOL, atol=ATOL)
        zs = [(zf(sol.y[4:10, j]), zf(sol.y[10:16, j])) for j in range(sol.y.shape[1])]
        Ts0 = Tmat(zs[0][0]) + Tmat(zs[0][1])
        ok &= np.max(abs(sol.y[3] - 0.5)) < 1e-11
        ok &= max(np.max(abs(Tmat(a) + Tmat(b) - Ts0)) for a, b in zs) < 1e-9
        ok &= max(abs(Mpair_formula(a, b) - Mpair_formula(*zs[0])) for a, b in zs) < 1e-9
        ok &= max(abs(abs(q(a) + q(b)) - abs(q(zs[0][0]) + q(zs[0][1]))) for a, b in zs) < 1e-9
        e0 = [E0(a) for a, b in zs]
        if abs(q(Z0[:3])) > 1e-9:
            ok &= np.ptp(e0) > 0.05
        else:
            ok &= max(e0) < 1e-10 and max(E0(b) for a, b in zs) < 1e-10
    report("d6-12", ok, "a second rotor acting by E (lG_Delta-equivariant): with equal inertias h_c is conserved (1e-11), the summed "
           "momentum tensor, M_pair and |q_tot| are constant (1e-9) while the per-copy E0(pi^(k)) move; ((1,i,0),0) stays null-null")


# ============================================================================ d6-13 the Dirac side
def claim_13():
    J1, J2 = E1[1] @ E1[2], E2[1] @ E2[2]
    ok_a = np.array_equal(J1, np.kron(J8, I8)) and np.array_equal(J2, np.kron(I8, J8)) and np.array_equal(J1 @ J2, J2 @ J1)
    interior = [0, 3, 4, 5]
    ok_a &= all(np.array_equal(Jk @ Ea[a], Ea[a] @ Jk) for Jk in (J1, J2) for Ea in (E1, E2) for a in interior)
    Jsum = J1 + J2
    P2x = I64 - J1 @ J2                                   # = 2P
    ok_a &= np.array_equal(Jsum @ Jsum, -2 * P2x) and np.array_equal(P2x @ P2x, 2 * P2x) and int(np.trace(P2x)) == 64
    report("d6-13a", ok_a, "J^(1) = J(x)1 and J^(2) = 1(x)J exactly (the Klein factors cancel), commuting, both commuting with the 8 interior "
           "gammas; (J^(1)+J^(2))^2 = -4P, rank P = 32: Q_tot = -i(J^(1)+J^(2)) has spectrum {-2,0,2} with multiplicities 16,32,16")
    # (b) the pair at rest: psi, phi with e1 psi = psi, J psi = i psi; Psi = psi (x) C phi, C = e2
    e1s, Js, Cs = (sp.Matrix(x.tolist()) for x in (e[0], J8, e[1]))
    Vp = (sp.Matrix.vstack(e1s - sp.eye(8), Js - sp.I * sp.eye(8))).nullspace()
    ok_b = len(Vp) == 2
    psi, phi = Vp[0], Vp[1]
    Cphi = Cs * phi
    ok_b &= (e1s * Cphi + Cphi).is_zero_matrix and (Js * Cphi + sp.I * Cphi).is_zero_matrix
    Psi = sp.Matrix(np.kron(np.array(psi.tolist(), dtype=object).ravel(), np.array(Cphi.tolist(), dtype=object).ravel()).tolist())
    J1s, J2s = sp.Matrix(J1.tolist()), sp.Matrix(J2.tolist())
    ok_b &= (J1s * Psi - sp.I * Psi).is_zero_matrix and (J2s * Psi + sp.I * Psi).is_zero_matrix
    mm = sp.Integer(3)
    JH = Js * (mm * e1s)
    Hcal = -sp.I * (kron(JH, sp.eye(8)) + kron(sp.eye(8), JH))
    ok_b &= (Hcal * Psi - 2 * mm * Psi).is_zero_matrix
    report("d6-13b", ok_b, "exact: Psi = psi (x) C phi (psi, phi particles at rest, C = e2 of Thm 5.1) has (Q1,Q2) = (+1,-1), Q_tot = 0, "
           "and two-particle energy -i(JH(x)1 + 1(x)JH) Psi = 2m Psi")
    # (c) charge conjugation on the pair
    CC = E1[1] @ E2[1]
    CxC = np.kron(e[1], e[1])
    OM1 = np.kron(OM, I8)
    ok_c = np.array_equal(CC @ CC, -I64) and np.array_equal(CC @ J1, -J1 @ CC) and np.array_equal(CC @ J2, -J2 @ CC)
    ok_c &= np.array_equal(CxC, -OM1 @ CC) and np.array_equal(CxC @ CxC, I64)
    psin = np.array(psi.tolist(), dtype=complex).ravel()
    Cpsi = e[1] @ psin
    Pp = np.kron(psin, Cpsi) + np.kron(Cpsi, psin)
    Pm = np.kron(psin, Cpsi) - np.kron(Cpsi, psin)
    ok_c &= np.allclose(CxC @ Pp, Pp) and np.allclose(CxC @ Pm, -Pm)
    ok_c &= np.allclose((-1j * (J1 + J2)) @ Pp, 0) and np.allclose((-1j * (J1 + J2)) @ Pm, 0)
    report("d6-13c", ok_c, "C^(1)C^(2) = e^(1)_2 e^(2)_2 anticommutes with J^(1), J^(2) but squares to -1 (no state is mapped to +-itself); "
           "the Klein-corrected C(x)C = -omega^(1) C^(1)C^(2) squares to +1 and Psi_pm = psi(x)Cpsi +- Cpsi(x)psi have C(x)C = +-1, Q_tot = 0")
    # (d) graded vs ungraded mass terms
    Hg = E1[0] + E2[0]
    Hu = np.kron(e[0], I8) + np.kron(I8, e[0])
    ok_d = np.array_equal(Hg @ Hg, 2 * I64) and np.array_equal(np.kron(I8, e[0]), OM1 @ E2[0])
    A, B = np.kron(e[0], I8), np.kron(I8, e[0])
    ok_d &= np.array_equal(A @ B, B @ A) and int(np.trace(A)) == 0 and int(np.trace(B)) == 0 and int(np.trace(A @ B)) == 0
    evu = np.round(np.linalg.eigvalsh(Hu.astype(float)), 9)
    ok_d &= sorted(set(evu)) == [-2.0, 0.0, 2.0] and list(evu).count(2.0) == 16 and list(evu).count(0.0) == 32
    report("d6-13d", ok_d, "the tower's mass terms anticommute: (e^(1)_1 + e^(2)_1)^2 = 2, spectrum m(e^(1)_1+e^(2)_1) = {+-sqrt2 m} (one particle, "
           "masses in quadrature); the two-particle m(e1(x)1 + 1(x)e1) = m(e^(1)_1 + omega^(1)e^(2)_1) has {-2m, 0, 2m} (16, 32, 16)")
    # (e) integer spin of the pair under the diagonal interior rotations (Prop. 10.9, k_I = 2)
    S1 = [E1[a] @ E1[b] for a, b in ((4, 5), (5, 3), (3, 4))]      # 2 Sigma^(1) for the interior rotations
    S2 = [E2[a] @ E2[b] for a, b in ((4, 5), (5, 3), (3, 4))]
    ok_e = all(np.array_equal(X @ X, -I64) for X in S1 + S2) and all(np.array_equal(X @ Y, Y @ X) for X, Y in zip(S1, S2))
    # exp(2 pi Sigma^(k)) = cos(pi) + (2 Sigma) sin(pi) = -1 for each copy, so the diagonal rotation through 2 pi is (+1)
    report("d6-13e", ok_e, "interior rotation generators of the two copies square to -1/4 (in units of Sigma) and commute: the diagonal "
           "rotation through 2pi is (-1)(-1) = +1 on S_2, integer spin of the pair (Prop. 10.9, k_I = 2)")


# ============================================================================ d6-14 zero-point energy of the duplicated frame
def Amat(k1, k2, beta, m1=1, m2=1):
    S = kron(sp.diag(k1, k2), G) + beta * ECOP
    Minv = kron(sp.diag(sp.Integer(1) / m1, sp.Integer(1) / m2), ETA.inv())
    A = sp.zeros(24, 24)
    A[:12, :12] = -S
    A[:12, 12:] = Minv
    A[12:, 12:] = S.T
    Hs = sp.zeros(24, 24)
    Hs[:12, 12:] = -S.T
    Hs[12:, :12] = -S
    Hs[12:, 12:] = Minv
    return A, Hs


def semisimple(A):
    lam = sp.symbols('lam')
    chi = A.charpoly(lam).as_expr()
    sqf = sp.quo(chi, sp.gcd(chi, sp.diff(chi, lam)), lam)
    Pp = sp.Poly(sqf, lam)
    R = sp.zeros(*A.shape)
    Ap = sp.eye(A.shape[0])
    for c in reversed(Pp.all_coeffs()):
        R += c * Ap
        Ap = Ap * A
    return R.is_zero_matrix


def krein(k1, k2, beta, m1=1, m2=1):
    A, Hs = Amat(k1, k2, beta, m1, m2)
    A = np.array(A.tolist(), dtype=float)
    Hs = np.array(Hs.tolist(), dtype=float)
    ev, V = np.linalg.eig(A)
    rows, zpe = [], 0.0
    for o in sorted(set(np.round(ev.imag[ev.imag > 1e-9], 8))):
        idx = np.where(abs(ev - 1j * o) < 1e-6)[0]
        W = V[:, idx]
        F = W.conj().T @ Hs @ W
        s = np.linalg.eigvalsh((F + F.conj().T) / 2)
        npos, nneg = int(np.sum(s > 1e-9)), int(np.sum(s < -1e-9))
        rows.append((float(o), len(idx), npos, nneg))
        zpe += 0.5 * o * (npos - nneg)
    return rows, zpe


def claim_14():
    ok_a = True
    for args in ((1, 3, 0), (1, 3, 0, 1, sp.Rational(5, 2)), (sp.Rational(1, 2), 7, 0, 3, 1)):
        rows, zpe = krein(*args)
        k1, k2 = float(args[0]), float(args[1])
        ok_a &= sorted(r[0] for r in rows) == sorted({k1, k2}) and all(r[1] == 6 and r[2] == 3 and r[3] == 3 for r in rows)
        ok_a &= abs(zpe) < 1e-9
    report("d6-14a", ok_a, "two binding manifolds without b: Krein signature (3,3) at each rate kappa_k, zero-point energy "
           "sum_k (hbar kappa_k/2)(3-3) = 0 for unequal rates and masses (Thm 18.6 of ES17 per copy)")
    ok_b = True
    for (k1, k2, beta) in ((1.0, 3.0, 0.4), (0.0, 5.0, 0.5), (2.0, 2.0, 0.7)):
        rows, zpe = krein(k1, k2, beta)
        mp = (k1 + k2) / 2 + np.sqrt((k1 - k2) ** 2 / 4 + beta ** 2)
        mm = (k1 + k2) / 2 - np.sqrt((k1 - k2) ** 2 / 4 + beta ** 2)
        ok_b &= np.allclose(sorted(r[0] for r in rows), sorted({abs(mp), abs(mm)}), atol=1e-7)
        ok_b &= all(r[1] == 6 and r[2] == 3 and r[3] == 3 for r in rows) and abs(zpe) < 1e-8
    rows, _ = krein(0.0, 5.0, 0.5)
    seesaw = min(r[0] for r in rows)
    ok_b &= abs(seesaw - 0.0495097568) < 1e-8 and abs(seesaw - 0.25 / 5) < 1e-3
    report("d6-14b", ok_b, f"with b = beta E: lines mu_pm = (k1+k2)/2 +- sqrt((k1-k2)^2/4+beta^2), Krein (3,3) each, zero-point energy 0 "
           f"(1e-8); seesaw line at (0,5,1/2): {seesaw:.10f} ~ beta^2/kappa2 = 0.05")
    ok_c = (not semisimple(Amat(1, 4, 2)[0])) and semisimple(Amat(1, 4, 1)[0]) and semisimple(Amat(2, 2, 2)[0]) \
        and semisimple(Amat(1, 3, sp.Rational(2, 5))[0])
    # at the exceptional ratio the kernel of S^dagger is not eta_2-null: v1^2+v2^2 = v1^2 (1 - kappa1/kappa2)
    S = kron(sp.diag(1, 4), G) + 2 * ECOP
    Sdag = ETA2.inv() * S.T * ETA2
    K = sp.Matrix.hstack(*Sdag.nullspace())
    ok_c &= K.shape[1] == 6 and not (K.T * ETA2 * K).is_zero_matrix
    report("d6-14c", ok_c, "exact (rationals): at beta^2 = kappa1 kappa2 with kappa1 != kappa2 (1,4,2) the lower line vanishes and M is "
           "not semisimple (Jordan block, linear growth; ker S^dagger is 6-dim and not eta_2-null); (1,4,1), (2,2,2), (1,3,2/5) semisimple")


def main():
    sgn = claim_0()
    claim_1()
    claim_2()
    claim_3()
    claim_4(sgn)
    claim_5()
    claim_6()
    claim_7()
    claim_8()
    claim_9()
    claim_10()
    claim_11()
    claim_12()
    claim_13()
    claim_14()
    if FAILED:
        print(f"FAILED: {FAILED}")
        return 1
    print("all d6 claims verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
