#!/usr/bin/env python3
"""Track d3 -- the unity and its differentiation in motion: the closure of the force law.

Verification script for investigations/rest_energy/sections/d3_closure.tex.
Conventions follow the unified paper of 17 October 2026 (tex/s02, s03, s04, s12, s15, s16, s17):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1); realisation of tex/s02_arena.tex eq. (realisation);
  binding root G e_i = e_{i+3}, G e_{i+3} = -e_i;  covector p (column), pi = eta^{-1} p,
  z(X) = x + i y for X = sum_j (x_j e_j + y_j e_{j+3})  (so (B;E) = pi, z = B + iE, G acts as i);
  C(p) = p^T eta^{-1} p,  D(p) = p^T G eta^{-1} p,  w(p) = C - iD = q(z(pi))        (tex/s17 eq. wdef);
  closure s = R ghat x|_G V, [ghat, X] = G X, coordinates (h, p),
  H = h^2/(2 I0) + p^T N p / 2, N = (a eta^{-1} + b G eta^{-1})/m (Lorentz class)     (tex/s17 prop:LP);
  for a field point Y with z(Y) = B + iE:  E6(Y) = |z|^2/2,  E0(Y) = |q(z)|/2,  S(Y) = E x B = (i/2) conj(z) x z
  (frame Poynting vector, minus the textbook one since B_t = -B); the MOMENTUM field is pi = eta^{-1} p, with
  E6(pi) = |p|^2/2, E0(pi) = |w(p)|/2; the FRAME field is X (X = e^{-theta G} x in the closure).
  Lorentz-class inertia N = (nu1 eta^{-1} + nu2 G eta^{-1})/m, nu = nu1 + i nu2 (the (a, b) of the paper; variables a, b below).
Claims d3-0 ... d3-23 match the labels of the tex section (Table d3:tab:claims).
Exact statements use sympy (integer/rational arithmetic); numerical ones use scipy solve_ivp (DOP853,
rtol = 1e-12) or mpmath (30 digits) and state their tolerances.
Run from any directory:  python3 d3_closure_dynamics.py   (exit 0 and one PASS line per claim iff all hold)
"""
import sys
import numpy as np
import sympy as sp
import mpmath as mpm
from scipy.integrate import solve_ivp
from scipy.linalg import expm, eigh
from scipy.optimize import brentq

FAILED = []


def report(cid, ok, text):
    if ok:
        print(f"PASS {cid}: {text}")
    else:
        print(f"FAIL {cid}: {text}")
        FAILED.append(cid)


def info(text):
    print(f"    . {text}")


def is_zero(M):
    M = sp.Matrix(M)
    return all(sp.simplify(x) == 0 for x in M)


RTOL, ATOL = 1e-12, 1e-13
rng = np.random.default_rng(20261017)
rng2 = np.random.default_rng(20261018)          # stream for the checks added in the revision (keeps rng's numbers unchanged)

# ----------------------------------------------------------------------------- Clifford realisation (tex/s02)
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def k3(a, b, c):
    return np.kron(np.kron(a, b), c)


e = [k3(sx, I2, I2), k3(sz, sx, I2), k3(sz, sz, sx),
     k3(ep, I2, I2), k3(sz, ep, I2), k3(sz, sz, ep)]
eta_d = [1, 1, 1, -1, -1, -1]
I8 = np.eye(8, dtype=np.int64)


def vec_coeffs(Y):
    """Exact coefficients of an 8x8 integer matrix Y along e_1..e_6; None if Y is not a vector."""
    c = [sp.Rational(int(np.trace(Y @ e[b])), 8 * eta_d[b]) for b in range(6)]
    rec = sum((float(c[b]) * e[b] for b in range(6)), np.zeros((8, 8)))
    return c if np.allclose(rec, Y) else None


def ad6(Bv, scale=sp.Rational(1, 2)):
    """6x6 matrix (columns = images of e_a) of v -> scale*[Bv, v] on vectors."""
    M = sp.zeros(6, 6)
    for a in range(6):
        c = vec_coeffs(Bv @ e[a] - e[a] @ Bv)
        assert c is not None
        for b in range(6):
            M[b, a] = scale * c[b]
    return M


ETA = sp.diag(*eta_d)
G = sp.zeros(6, 6)
for i in range(3):
    G[i + 3, i] = 1
    G[i, i + 3] = -1
I6 = sp.eye(6)
Gn = np.array(G.tolist(), dtype=float)
ETAn = np.diag(np.array(eta_d, dtype=float))
J8 = e[1] @ e[2]                       # J = e2 e3
OM8 = e[0] @ e[3] @ e[4] @ e[5]        # Omega = e1 e4 e5 e6
GAM8 = J8 @ OM8                        # Gamma = J Omega
T8 = e[0] @ e[1] @ e[2]                # mass form T = gamma^0 J = e1 e2 e3

CYC = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
Lmat = [ad6(e[j + 3] @ e[k + 3] - e[j] @ e[k]) for (i, j, k) in CYC]     # L_i
Kmat = [ad6(e[k] @ e[j + 3] - e[j] @ e[k + 3]) for (i, j, k) in CYC]     # K_i
LG = Lmat + Kmat
LGn = [np.array(A.tolist(), dtype=float) for A in LG]


# ----------------------------------------------------------------------------- momentum-field helpers
def zfield(p):
    pi = ETAn @ p                        # eta^{-1} = eta
    return pi[:3] + 1j * pi[3:]


def wfun(p):
    z = zfield(p)
    return complex(np.dot(z, z))         # q(z) = z.z  (no conjugation)


def E6fun(p):
    return 0.5 * float(np.vdot(zfield(p), zfield(p)).real)


def E0fun(p):
    return 0.5 * abs(wfun(p))


def Sfun(p):
    pi = ETAn @ p
    B, E = pi[:3], pi[3:]
    return np.cross(E, B)


def expG(t):
    return np.cos(t) * np.eye(6) + np.sin(t) * Gn


def lorentzN(a, b, m):
    return (a * ETAn + b * Gn @ ETAn) / m


def closure_rhs(Hh, gradp):
    """Lie-Poisson field of tex/s17 eq. (LProtor) for a general Hamiltonian, with the angle theta and the frame x.
    State y = (theta, h, p[6], x[6]); theta' = H_h, h' = -p^T G grad_p H, p' = H_h G^T p, x' = e^{theta G} grad_p H."""
    def f(t, y):
        th, h, p = y[0], y[1], y[2:8]
        hh = Hh(h, p)
        gp = gradp(h, p)
        return np.concatenate(([hh, -p @ Gn @ gp], hh * (Gn.T @ p), expG(th) @ gp))
    return f


def quad_closure(I0, N):
    return closure_rhs(lambda h, p: h / I0, lambda h, p: N @ p)


def integrate(f, y0, T, **kw):
    return solve_ivp(f, (0.0, T), y0, method="DOP853", rtol=RTOL, atol=ATOL, dense_output=True, **kw)


def pendulum_data(p0, a, b):
    W = (a + 1j * b) * wfun(p0)
    return abs(W), np.angle(W)


# ============================================================================= d3-0 conventions
ok = all(np.array_equal(e[a] @ e[b] + e[b] @ e[a], 2 * eta_d[a] * I8 * (a == b)) for a in range(6) for b in range(6))
# binding root from Clifford data (tex/s03 lem:Gclifford): G = -K o S, K(v) = [P, v]/2, P = sum e_i e_{i+3}, S(v) = T v T^{-1}
Pp = sum(e[i] @ e[i + 3] for i in range(3))
Kc = ad6(Pp)
Tinv = -T8                                           # T^2 = -1
S6 = sp.zeros(6, 6)
for a_ in range(6):
    c = vec_coeffs(T8 @ e[a_] @ Tinv)
    for b_ in range(6):
        S6[b_, a_] = c[b_]
ok &= (S6 == ETA) and (-Kc * S6 == G)
ok &= (G * G == -I6) and (ETA.inv() * G.T * ETA == G) and (G.T * ETA * G == -ETA) and (G.T == -G)
ok &= all((A.T * ETA + ETA * A).is_zero_matrix and (A * G - G * A).is_zero_matrix for A in LG)
ok &= sp.Matrix([list(A) for A in LG]).rank() == 6
report("d3-0", ok, "realisation; G = -K o Ad_T is integer, G^2=-1, G^dagger=G, anti-isometry, G^T=-G; "
       "the six L_i, K_i lie in so(3,3) and commute with G")

# ============================================================================= d3-1 Lie-Poisson equations, H and |p|^2 conserved
hs, I0s, ms, as_, bs = sp.symbols("h I_0 m a b", real=True)
ps = sp.Matrix(sp.symbols("p1:7", real=True))
Nsym = (as_ * ETA.inv() + bs * G * ETA.inv()) / ms
Hsym = hs ** 2 / (2 * I0s) + (ps.T * Nsym * ps)[0] / 2
# structure constants of s = R ghat + V, basis (E_0 = ghat, E_1..E_6 = e_1..e_6): [ghat, e_j] = G e_j, [e_i, e_j] = 0
cst = [[[sp.Integer(0)] * 7 for _ in range(7)] for _ in range(7)]   # cst[j][k][l] = c^l_{jk}
for j in range(6):
    for l in range(6):
        cst[0][j + 1][l + 1] = G[l, j]
        cst[j + 1][0][l + 1] = -G[l, j]
mu = [hs] + list(ps)
dH = [sp.diff(Hsym, v) for v in mu]
# Lie-Poisson equation mu_dot = ad^*_{dH} mu, i.e. mu_dot_k = sum_{j,l} dH_j c^l_{jk} mu_l
mudot = [sp.expand(sum(dH[j] * cst[j][k][l] * mu[l] for j in range(7) for l in range(7))) for k in range(7)]
pdot_paper = (hs / I0s) * G.T * ps
hdot_paper = -(ps.T * G * Nsym * ps)[0]
ok = sp.simplify(mudot[0] - hdot_paper) == 0 and all(sp.simplify(mudot[k + 1] - pdot_paper[k]) == 0 for k in range(6))
Hdot = sp.simplify(sum(sp.diff(Hsym, mu[k]) * mudot[k] for k in range(7)))
Cdot = sp.simplify(sum(sp.diff((ps.T * ps)[0], mu[k]) * mudot[k] for k in range(7)))
Nfree = sp.Matrix(6, 6, lambda r, c_: sp.Symbol(f"n{min(r, c_)}{max(r, c_)}", real=True))
Hgen = hs ** 2 / (2 * I0s) + (ps.T * Nfree * ps)[0] / 2
pdg = (hs / I0s) * G.T * ps
hdg = -(ps.T * G * Nfree * ps)[0]
Hdot_gen = sp.expand(sp.diff(Hgen, hs) * hdg + sum(sp.diff(Hgen, ps[k]) * pdg[k] for k in range(6)))
ok &= Hdot == 0 and Cdot == 0 and Hdot_gen == 0
# the bracket of the text, {F,K} = p^T (K_h G grad F - F_h G grad K), gives K' = {K, H} for the coordinate functions
def pb(Fx, Kx):
    gF = sp.Matrix([sp.diff(Fx, v) for v in ps]); gK = sp.Matrix([sp.diff(Kx, v) for v in ps])
    return sp.expand((ps.T * (sp.diff(Kx, hs) * G * gF - sp.diff(Fx, hs) * G * gK))[0])


ok &= sp.simplify(pb(hs, Hsym) - hdot_paper) == 0 and all(sp.simplify(pb(ps[k], Hsym) - pdot_paper[k]) == 0 for k in range(6))
report("d3-1", ok, "eq. d3:eq:LPgen: mu' = ad*_{dH} mu reproduces p' = (h/I0) G^T p, h' = -p^T G N p (Prop 17.8, eq. LProtor) and "
       "equals F' = {F,H} for {F1,F2} = p^T(F2_h G grad F1 - F1_h G grad F2); dH/dtau = 0 for every symmetric N and "
       "d|p|^2/dtau = 0 (exact)")

# ============================================================================= d3-2 Casimirs: E6, E0, S; leaf; phase law
Pi = sp.Matrix(7, 7, lambda k, j: sum(cst[j][k][l] * mu[l] for l in range(7)))   # mu_dot_k = sum_j Pi[k,j] dH_j
pis = ETA.inv() * ps
Bv, Ev = pis[:3, 0], pis[3:, 0]
Cp = (ps.T * ETA.inv() * ps)[0]
Dp = (ps.T * G * ETA.inv() * ps)[0]
Sv = Ev.cross(Bv)
cands = {"|p|^2": (ps.T * ps)[0], "|w|^2=C^2+D^2": Cp ** 2 + Dp ** 2, "S1": Sv[0], "S2": Sv[1], "S3": Sv[2]}


def casimir(f):
    grad = sp.Matrix([sp.diff(f, v) for v in mu]).T
    return (grad * Pi).applyfunc(sp.expand).is_zero_matrix


ok = all(casimir(f) for f in cands.values())
ok &= (not casimir(Cp)) and (not casimir(Dp))                                   # C and D alone are not Casimirs
ok &= sp.expand((Bv.dot(Bv) + Ev.dot(Ev)) / 2 - (ps.T * ps)[0] / 2) == 0          # E6(p) = |p|^2/2
ok &= sp.expand(Cp - (Bv.dot(Bv) - Ev.dot(Ev))) == 0 and sp.expand(Dp + 2 * Bv.dot(Ev)) == 0   # w = |B|^2-|E|^2+2i B.E
# leaf dimension 2 and five independent Casimirs at a generic rational point
pt = dict(zip(mu, [sp.Rational(3, 7), 1, -2, sp.Rational(1, 3), 5, -1, 2]))
ok &= Pi.subs(pt).rank() == 2
zs = [Bv[k] + sp.I * Ev[k] for k in range(3)]
herm = []
for r in range(3):
    for c_ in range(r, 3):
        herm += [sp.re(sp.expand(sp.conjugate(zs[r]) * zs[c_])), sp.im(sp.expand(sp.conjugate(zs[r]) * zs[c_]))]
herm = [x for x in herm if x != 0]
ok &= all(casimir(f) for f in herm)
Jh = sp.Matrix([[sp.diff(f, v) for v in ps] for f in herm]).subs(pt)
ok &= Jh.rank() == 5
E0sq = (Cp ** 2 + Dp ** 2) / 4
J5 = sp.Matrix([[sp.diff(f, v) for v in ps] for f in [(ps.T * ps)[0] / 2, E0sq, Sv[0], Sv[1], Sv[2]]]).subs(pt)
ok &= J5.rank() == 4
ok &= sp.expand(((ps.T * ps)[0] / 2) ** 2 - E0sq - Sv.dot(Sv)) == 0              # E6^2 = E0^2 + |S|^2
# phase law: e^{theta G} = cos + sin G, eta e^{-theta G} = e^{theta G} eta, z(G X) = i z(X), hence w(e^{-theta G} p) = e^{2 i theta} w(p)
th = sp.Symbol("theta", real=True)
eG = sp.cos(th) * I6 + sp.sin(th) * G
ok &= is_zero(sp.diff(eG, th) - G * eG) and eG.subs(th, 0) == I6
ok &= is_zero(ETA * (sp.cos(th) * I6 - sp.sin(th) * G) - eG * ETA)
pth = (sp.cos(th) * I6 - sp.sin(th) * G) * ps
wth = sp.expand(((pth.T * ETA.inv() * pth)[0]) - sp.I * (pth.T * G * ETA.inv() * pth)[0])
ok &= sp.simplify(sp.expand_complex(wth - sp.exp(2 * sp.I * th) * (Cp - sp.I * Dp))) == 0
report("d3-2", ok, "thm:frozen: E6(pi)=|p|^2/2, E0(pi)^2=|w|^2/4 and the vector S(pi) of the momentum field are Casimirs of s* "
       "(C, D alone are not); leaves are 2-dim, 5 independent Casimirs; E6^2=E0^2+|S|^2; p=e^{-theta G}p0 gives "
       "z(pi)->e^{i theta}z(pi), w->e^{2i theta}w (exact; in substance Prop 17.8 of ES17)")

# ============================================================================= d3-3 numerical conservation along the closure
worst = 0.0
worst_phase = 0.0
for trial in range(5):
    p0 = rng.normal(size=6)
    h0 = rng.normal() * 2
    a, b = rng.normal(size=2)
    m, I0 = rng.uniform(0.5, 2.0, size=2)
    N = lorentzN(a, b, m)
    y0 = np.concatenate(([0.0, h0], p0, np.zeros(6)))
    sol = integrate(quad_closure(I0, N), y0, 40.0)
    ts = np.linspace(0, 40.0, 401)
    Y = sol.sol(ts)
    H0 = h0 ** 2 / (2 * I0) + 0.5 * p0 @ N @ p0
    for k in range(len(ts)):
        thk, hk, pk = Y[0, k], Y[1, k], Y[2:8, k]
        Hk = hk ** 2 / (2 * I0) + 0.5 * pk @ N @ pk
        scale = abs(h0 ** 2 / (2 * I0)) + abs(0.5 * p0 @ p0 * (abs(a) + abs(b)) / m)
        errs = [abs(Hk - H0) / scale, abs(pk @ pk - p0 @ p0) / (p0 @ p0),
                abs(E0fun(pk) - E0fun(p0)) / (p0 @ p0), np.linalg.norm(Sfun(pk) - Sfun(p0)) / (p0 @ p0),
                abs(E6fun(pk) - E6fun(p0)) / (p0 @ p0)]
        worst = max(worst, max(errs))
        worst_phase = max(worst_phase, abs(wfun(pk) - np.exp(2j * thk) * wfun(p0)) / (p0 @ p0),
                          np.linalg.norm(pk - expG(-thk) @ p0) / np.linalg.norm(p0))
info(f"5 random data, tau in [0,40]: max relative drift of H, |p|^2, E6, E0, S = {worst:.2e}; "
     f"max |w(p) - e^(2i theta) w(p0)|, |p - e^(-theta G) p0| = {worst_phase:.2e}")
report("d3-3", worst < 1e-9 and worst_phase < 1e-9,
       "thm:frozen (numerical): closure integrated (DOP853, rtol 1e-12) for 5 random data: H, |p|^2, E6(pi), E0(pi) and the "
       "vector S(pi) conserved to <1e-9; w(p(tau)) = e^{2i theta(tau)} w(p0) and p = e^{-theta G} p0 to <1e-9")

# ============================================================================= d3-4 no Hamiltonian on s* changes E6 or E0 (obstruction)
M1 = rng.normal(size=(6, 6)); M1 = (M1 + M1.T) / 2
M2 = rng.normal(size=(6, 6)); M2 = (M2 + M2.T) / 2
I0g = 0.8


def Hgen_num(h, p):
    return h ** 2 / (2 * I0g) + 0.1 * h ** 4 + np.sin(p @ M1 @ p) + 0.2 * h * (p @ M2 @ p)


fgen = closure_rhs(lambda h, p: h / I0g + 0.4 * h ** 3 + 0.2 * (p @ M2 @ p),
                   lambda h, p: 2 * np.cos(p @ M1 @ p) * (M1 @ p) + 0.4 * h * (M2 @ p))
p0 = rng.normal(size=6)
y0 = np.concatenate(([0.0, 0.7], p0, np.zeros(6)))
sol = integrate(fgen, y0, 15.0)
Y = sol.sol(np.linspace(0, 15, 301))
errg = max(max(abs(E6fun(Y[2:8, k]) - E6fun(p0)), abs(E0fun(Y[2:8, k]) - E0fun(p0)),
               np.linalg.norm(Sfun(Y[2:8, k]) - Sfun(p0)), abs(Hgen_num(Y[1, k], Y[2:8, k]) - Hgen_num(0.7, p0)))
           for k in range(Y.shape[1])) / (p0 @ p0)
moved = np.ptp(np.unwrap(np.angle([wfun(Y[2:8, k]) for k in range(Y.shape[1])])))
info(f"non-quadratic H with random symmetric forms: drift of E6, E0, S, H = {errg:.2e}; duality phase of w moved by {moved:.2f} rad")
report("d3-4", errg < 1e-9 and moved > 1.0,
       "cor:nogen(i): for an arbitrary (non-quadratic, non-Lorentz) Hamiltonian on s* the rest energy E0(pi), undivided energy "
       "E6(pi) and S(pi) of the MOMENTUM field stay fixed while only the duality phase moves: no Hamiltonian on s* generates "
       "or changes the rest energy of the momentum field (for the frame field see d3-22)")

# ============================================================================= d3-5 the pendulum potential
ok = sp.simplify((ps.T * Nsym * ps)[0] / 2 - sp.re(sp.expand((as_ + sp.I * bs) * (Cp - sp.I * Dp))) / (2 * ms)) == 0
Uth = sp.expand((pth.T * Nsym * pth)[0] / 2)
W0 = sp.expand((as_ + sp.I * bs) * (Cp - sp.I * Dp))
Uformula = sp.re(sp.expand_complex(W0 * sp.exp(2 * sp.I * th))) / (2 * ms)
ok &= sp.simplify(sp.expand_trig(Uth - Uformula)) == 0
hdot_th = sp.expand(-(pth.T * G * Nsym * pth)[0])
ok &= sp.simplify(sp.expand_trig(hdot_th + sp.diff(Uth, th))) == 0                # h' = -dU/dtheta
ok &= sp.simplify(sp.expand_trig(hdot_th - sp.im(sp.expand_complex(W0 * sp.exp(2 * sp.I * th))) / ms)) == 0
# numerically along trajectories: U = (R0/2m) cos(2 theta + phi0), h' = (R0/m) sin(2 theta + phi0), R0 = 2|a+ib| E0
worst = 0.0
for trial in range(3):
    p0 = rng.normal(size=6)
    a, b = rng.normal(size=2)
    m, I0 = rng.uniform(0.5, 2.0, size=2)
    N = lorentzN(a, b, m)
    R0, phi0 = pendulum_data(p0, a, b)
    worst = max(worst, abs(R0 - 2 * abs(a + 1j * b) * E0fun(p0)))
    sol = integrate(quad_closure(I0, N), np.concatenate(([0.0, 1.3], p0, np.zeros(6))), 20.0)
    for tk in np.linspace(0, 20, 101):
        y = sol.sol(tk)
        psi = 2 * y[0] + phi0
        rhs = quad_closure(I0, N)(tk, y)
        worst = max(worst, abs(0.5 * y[2:8] @ N @ y[2:8] - R0 / (2 * m) * np.cos(psi)) / R0,
                    abs(rhs[1] - R0 / m * np.sin(psi)) / R0)
ok &= worst < 1e-9
report("d3-5", ok, "prop:pendulum (restates Thm 17.13): U = p^T N p/2 = Re(nu w)/(2m) = (R0/2m)cos(2theta+phi0) with "
       "R0 = |nu||w(p0)| = 2|nu|E0(pi0), nu = a+ib, I0 theta'' = (R0/m) sin psi (exact; along 3 random flows to <1e-9)")

# ============================================================================= d3-6 small oscillations: Omega^2 = 2 R0/(m I0)
def period_about_pi(delta0, p0, a, b, m, I0):
    """Start at psi(0) = pi + delta0, h(0) = 0; return the measured period of psi."""
    R0, phi0 = pendulum_data(p0, a, b)
    ths = (np.pi + delta0 - phi0) / 2
    pst = expG(-ths) @ p0
    R0s, phis = pendulum_data(pst, a, b)
    assert abs(R0s - R0) < 1e-12 and abs(np.angle(np.exp(1j * (phis - np.pi - delta0)))) < 1e-12
    Om = np.sqrt(2 * R0 / (m * I0))
    Tguess = 4 * float(mpm.ellipk(np.sin(delta0 / 2) ** 2)) / Om
    sol = integrate(quad_closure(I0, lorentzN(a, b, m)), np.concatenate(([0.0, 0.0], pst, np.zeros(6))), 3.3 * Tguess)
    hfun = lambda t: sol.sol(t)[1]
    grid = np.linspace(1e-3 * Tguess, 3.2 * Tguess, 4000)
    vals = np.array([hfun(t) for t in grid])
    zeros = [brentq(hfun, grid[k], grid[k + 1], xtol=1e-15, rtol=1e-15)
             for k in range(len(grid) - 1) if vals[k] * vals[k + 1] < 0]
    # h vanishes at the turning points: every half period
    Tmeas = (zeros[4] - zeros[0]) / 2 if len(zeros) >= 5 else (zeros[2] - zeros[0])
    return Tmeas, Om, R0


p0 = rng.normal(size=6)
a, b, m, I0 = 0.8, -0.5, 1.3, 0.7
T_small, Om, R0 = period_about_pi(1e-3, p0, a, b, m, I0)
T_big, _, _ = period_about_pi(2.0, p0, a, b, m, I0)
rel_small = abs(T_small - 2 * np.pi / Om) / (2 * np.pi / Om)
T_exact = 4 * float(mpm.ellipk(np.sin(1.0) ** 2)) / Om
rel_big = abs(T_big - T_exact) / T_exact
info(f"Omega = sqrt(2R0/(m I0)) = {Om:.10f}; small amplitude: T_meas/(2pi/Omega) - 1 = {rel_small:.2e} "
     f"(expected xi0^2/16 = 6.3e-08); amplitude 2 rad: T_meas/(4K(sin^2 1)/Omega) - 1 = {rel_big:.2e}")
ok = rel_small < 2e-7 and rel_big < 1e-9
ok &= abs(Om ** 2 - 4 * abs(a + 1j * b) * E0fun(p0) / (m * I0)) < 1e-12
report("d3-6", ok, "prop:omega: small oscillations about psi = pi have Omega^2 = 2R0/(m I0) = 4|nu|E0(pi0)/(m I0): measured period "
       "matches 2pi/Omega to 2e-7 at amplitude 1e-3, and the exact 4K(sin^2(xi0/2))/Omega to 1e-9 at amplitude 2")

# ============================================================================= d3-7 flat direction on null data and in the duality class
# null momentum field: |B| = |E|, B.E = 0  (pi = (B; E))
Bn = np.array([1.0, 0.3, -0.4]); En = np.cross(Bn, np.array([0.2, 1.0, 0.5])); En *= np.linalg.norm(Bn) / np.linalg.norm(En)
p_null = ETAn @ np.concatenate((Bn, En))
ok = abs(wfun(p_null)) < 1e-14
sol = integrate(quad_closure(0.9, lorentzN(0.7, 1.1, 1.2)), np.concatenate(([0.0, 1.5], p_null, np.zeros(6))), 30.0)
hs_ = sol.sol(np.linspace(0, 30, 301))[1]
dev1 = np.max(np.abs(hs_ - 1.5))
p0 = rng.normal(size=6)
sol = integrate(quad_closure(0.9, np.eye(6) / 1.2), np.concatenate(([0.0, 1.5], p0, np.zeros(6))), 30.0)
hs2 = sol.sol(np.linspace(0, 30, 301))[1]
dev2 = np.max(np.abs(hs2 - 1.5))
# symbolic: N = 1/m commutes with G and p^T G p = 0, so h' = 0
ok &= sp.expand((ps.T * G * ps)[0]) == 0
# (iii) the whole duality class [N, G] = 0 (dimension 9): h' = -p^T G N p = 0 and U(theta) = p0^T N p0 / 2 is constant
NcomG = sp.solve(list(Nfree * G - G * Nfree), list(Nfree.free_symbols), dict=True)[0]
Ndual = Nfree.subs(NcomG)
ok &= len(Ndual.free_symbols) == 9 and (Ndual * G - G * Ndual).is_zero_matrix
ok &= sp.expand((ps.T * G * Ndual * ps)[0]) == 0
Udual = sp.expand(((sp.cos(th) * I6 - sp.sin(th) * G) * ps).T * Ndual * ((sp.cos(th) * I6 - sp.sin(th) * G) * ps))[0] / 2
ok &= sp.simplify(sp.expand_trig(Udual - (ps.T * Ndual * ps)[0] / 2)) == 0
Ndn = np.array(Ndual.subs({v: rng2.normal() for v in sorted(Ndual.free_symbols, key=str)}).tolist(), dtype=float)
p0 = rng2.normal(size=6)
sol = integrate(quad_closure(0.9, Ndn), np.concatenate(([0.0, 1.5], p0, np.zeros(6))), 30.0)
Yd7 = sol.sol(np.linspace(0, 30, 301))
dev3 = np.max(np.abs(Yd7[1] - 1.5))
U7 = np.array([0.5 * Yd7[2:8, k] @ Ndn @ Yd7[2:8, k] for k in range(Yd7.shape[1])])
dev3 = max(dev3, np.max(np.abs(U7 - 0.5 * p0 @ Ndn @ p0)))
ok &= abs(0.5 * p0 @ Ndn @ p0) > 1e-3                                           # in general nonzero
# (i) with E0(pi0) > 0 and nu != 0: h constant => sin psi == 0 => theta constant => h = 0 (equilibria), and there the frame
# moves with the constant velocity N p0, |N p0| = |nu||p0|/m (N^2 = |nu|^2/m^2)
ok &= (Nsym * Nsym - (as_ ** 2 + bs ** 2) / ms ** 2 * I6).applyfunc(sp.simplify).is_zero_matrix
a7, b7, m7, I07 = 0.8, -0.6, 1.3, 0.7
N7 = lorentzN(a7, b7, m7)
p0 = rng2.normal(size=6)
R07, phi07 = pendulum_data(p0, a7, b7)
dev_eq = 0.0
for psi_eq in (np.pi, 0.0):                                                      # stable and unstable equilibrium
    peq = expG(-(psi_eq - phi07) / 2) @ p0                                       # theta(0) = 0, psi(0) = phi0' = psi_eq
    _, ph_eq = pendulum_data(peq, a7, b7)
    ok &= abs(np.angle(np.exp(1j * (ph_eq - psi_eq)))) < 1e-12
    T7 = 5.0 if psi_eq == np.pi else 2.0                                          # the unstable one only for a short time
    sol = integrate(quad_closure(I07, N7), np.concatenate(([0.0, 0.0], peq, np.zeros(6))), T7)
    tt7 = np.linspace(0, T7, 101)
    Y7 = sol.sol(tt7)
    dev_eq = max(dev_eq, np.max(np.abs(Y7[1])), np.max(np.abs(Y7[8:14] - np.outer(N7 @ peq, tt7))) / T7)
ok &= dev_eq < 1e-9
ok &= abs(np.linalg.norm(N7 @ p0) - abs(a7 + 1j * b7) * np.linalg.norm(p0) / m7) < 1e-12
info(f"null data, Lorentz class: max |h - h0| = {dev1:.1e}; duality class N = 1/m, random data: max |h - h0| = {dev2:.1e}; "
     f"random 9-parameter duality-class N: |h - h0|, |U - U0| <= {dev3:.1e}; equilibria h = 0, psi in {{0, pi}}: h stays 0 and "
     f"x = tau N p0 to {dev_eq:.1e}")
report("d3-7", ok and dev1 < 1e-11 and dev2 < 1e-11 and dev3 < 1e-10,
       "cor:flat: with nu != 0 the rate is constant and nonzero iff R0 = 0 (null momentum field) and h0 != 0 (exact flat "
       "direction); for E0(pi0) > 0 only at the equilibria h = 0, psi in pi Z, where it vanishes and the frame moves at the "
       "constant velocity N p0, |N p0| = |nu||p0|/m; nu = 0: constant for all data; the whole 9-dim duality class [N,G] = 0 "
       "keeps the rate for all data with U = p0^T N p0/2 constant, in general nonzero (exact + integrations)")

# ============================================================================= d3-8 pseudo-Goldstone reading: I0 Omega^2 = U''(min)
Rs, phs = sp.symbols("R_0 varphi_0", positive=True)
Ups = Rs / (2 * ms) * sp.cos(2 * th + phs)
thmin = (sp.pi - phs) / 2
ok = sp.simplify(sp.diff(Ups, th).subs(th, thmin)) == 0 and sp.simplify(sp.diff(Ups, th, 2).subs(th, thmin) - 2 * Rs / ms) == 0
ok &= sp.simplify(sp.diff(Ups, th, 2).subs(th, thmin) / I0s - 2 * Rs / (ms * I0s)) == 0
# the first equality I0 Omega^2 = U''(theta_min) holds for EVERY one-degree-of-freedom oscillator H = h^2/2I0 + V(theta):
Vf = sp.Function("V")
ok &= sp.simplify(sp.diff(-sp.diff(Vf(th), th) / I0s, th).subs(th, thmin)
                  + sp.diff(Vf(th), th, 2).subs(th, thmin) / I0s) == 0      # linearised theta'' = -(V''/I0) delta theta
report("d3-8", ok, "rem:goldstone: I0 Omega^2 = U''(theta_min) holds for every one-degree-of-freedom oscillator; the closure-specific "
       "content U''(theta_min) = 2R0/m = 4|nu|E0(pi0)/m is Theorem 17.13 of ES17 (exact); the GMOR / pseudo-Goldstone reading "
       "is interpretive and generic")

# ============================================================================= d3-9 energy budget and Lorentz invariance of the split
ok = True
worst = 0.0
p0 = rng.normal(size=6)
a, b, m, I0 = 0.6, 0.9, 1.1, 0.8
N = lorentzN(a, b, m)
R0, phi0 = pendulum_data(p0, a, b)
h0 = 2.5
sol = integrate(quad_closure(I0, N), np.concatenate(([0.0, h0], p0, np.zeros(6))), 25.0)
H0 = h0 ** 2 / (2 * I0) + 0.5 * p0 @ N @ p0
Tps = np.linspace(0, 25, 200001)
Yd = sol.sol(Tps)
Kk = Yd[1] ** 2 / (2 * I0)
Uk = np.array([0.5 * Yd[2:8, k] @ N @ Yd[2:8, k] for k in range(0, len(Tps), 50)])
worst = np.max(np.abs(Kk[::50] + Uk - H0)) / abs(H0)
# dK/dtau = (R0/m) kappa sin psi = -dU/dtau
for tk in np.linspace(0, 25, 51):
    y = sol.sol(tk)
    rhs = quad_closure(I0, N)(tk, y)
    dK = y[1] * rhs[1] / I0
    worst = max(worst, abs(dK - R0 / m * (y[1] / I0) * np.sin(2 * y[0] + phi0)) / R0)
psi_span = 2 * (Yd[0, -1] - Yd[0, 0])
exch = (Kk.max() - Kk.min()) / (R0 / m) - 1
info(f"H = K + U to {worst:.1e}; psi swept {psi_span:.1f} rad (rotating regime); (max K - min K)/(R0/m) - 1 = {exch:.1e}")
ok &= worst < 1e-9 and psi_span > 4 * np.pi and abs(exch) < 1e-6
# K and U are separately invariant under l_G acting on covectors (p -> Lambda^{-T} p); w is l_G-invariant
coef = rng.normal(size=6)
Lam = expm(sum(c_ * A for c_, A in zip(coef, LGn)))
ok &= np.allclose(Lam.T @ ETAn @ Lam, ETAn, atol=1e-12) and np.allclose(Lam @ Gn, Gn @ Lam, atol=1e-12)
p0b = np.linalg.inv(Lam).T @ p0
ok &= abs(wfun(p0b) - wfun(p0)) < 1e-10 * abs(wfun(p0)) and abs(0.5 * p0b @ N @ p0b - 0.5 * p0 @ N @ p0) < 1e-10
solb = integrate(quad_closure(I0, N), np.concatenate(([0.0, h0], p0b, np.zeros(6))), 25.0)
tt = np.linspace(0, 25, 501)
dth = np.max(np.abs(solb.sol(tt)[0] - sol.sol(tt)[0]))
dE6 = abs(E6fun(p0b) - E6fun(p0))
info(f"boosted momentum p0' = Lambda^-T p0: |w'-w| = {abs(wfun(p0b) - wfun(p0)):.1e}, E6' - E6 = {E6fun(p0b) - E6fun(p0):+.3f}, "
     f"max |theta'(tau) - theta(tau)| = {dth:.1e}")
ok &= dth < 1e-9 and dE6 > 0.1
# symbolic: l_G invariance of the Lorentz-class N, i.e. r N + N r^T = 0 for r in l_G (covector action -r^T)
ok &= all((A * Nsym + Nsym * A.T).applyfunc(sp.simplify).is_zero_matrix for A in LG)
report("d3-9", ok, "prop:budget: H = K_rot + U with K_rot = I0 kappa^2/2 and U = (R0/2m)cos psi exchanging energy "
       "(dK_rot/dtau = -dU/dtau); in the rotating regime the exchange amplitude is exactly R0/m; K_rot and U are separately "
       "l_G-invariant and the rotor dynamics sees p0 only through nu w(p0): a boosted p0 with different E6(pi) gives the same "
       "theta(tau) to 1e-9")

# ============================================================================= d3-10 dimensions: the c^2 part and the factor 1/2
# dimension vectors (M, L, T)
dims = {"tau": (0, 0, 1), "c": (0, 1, -1), "ell": (0, 1, 0), "I0": (1, 2, 0), "m": (1, 0, 0), "X": (0, 1, 0),
        "hbar": (1, 2, -1)}


def dsum(*pairs):
    out = np.zeros(3, dtype=int)
    for name, k in pairs:
        out += k * np.array(dims[name])
    return tuple(out)


kappa_dim = dsum(("c", 1), ("ell", -1))
energy = (1, 2, -2)
ok = kappa_dim == (0, 0, -1)
ok &= dsum(("I0", 1), ("c", 2), ("ell", -2)) == energy                     # (1/2) I0 kappa^2 is an energy
ok &= dsum(("I0", 1), ("ell", -2)) == dims["m"]                           # M_rot := I0/ell^2 is a mass
ok &= dsum(("I0", 1), ("c", 1), ("ell", -1)) == dims["hbar"]              # h = I0 kappa has the dimension of an action
p_dim = dsum(("m", 1), ("X", 1), ("tau", -1))
ok &= tuple(2 * np.array(p_dim) - np.array(dims["m"])) == energy          # p^T N p, N ~ 1/m, is an energy
cs, ells = sp.symbols("c ell", positive=True)
ok &= sp.simplify(sp.Rational(1, 2) * I0s * (cs / ells) ** 2 - sp.Rational(1, 2) * (I0s / ells ** 2) * cs ** 2) == 0
ok &= (sp.Matrix([[0, -1], [1, 0]]) * (cs / ells)) ** 2 == -(cs / ells) ** 2 * sp.eye(2)   # (kappa G)^2 = -c^2/ell^2 per plane
ok &= tuple(2 * np.array(p_dim)) == (2, 2, -2)                             # E0(pi) ~ p^2: a momentum squared
report("d3-10", ok, "units: with kappa = c/ell the rotor term is K_rot = (1/2)(I0/ell^2)c^2 = (1/2)M_rot c^2, M_rot = I0/ell^2 a mass, "
       "h = I0 kappa an action, U an energy, E0(pi) a momentum squared; (kappa G)^2 = -c^2/ell^2 (dimension bookkeeping exact; "
       "the 'c^2 part' reading is interpretive)")

# ============================================================================= d3-11 multiscale criterion
eps_s, c0, c1 = sp.symbols("epsilon cos_psi0 cos_psi", real=True)
k0 = sp.Symbol("kappa_0", positive=True)
# energy conservation: I0 k^2/2 + (R0/2m) cos psi = I0 k0^2/2 + (R0/2m) cos psi0, with R0 = eps m I0 k0^2
ksq = sp.solve(sp.Eq(I0s * sp.Symbol("K2") / 2 + eps_s * ms * I0s * k0 ** 2 / (2 * ms) * c1,
                     I0s * k0 ** 2 / 2 + eps_s * ms * I0s * k0 ** 2 / (2 * ms) * c0), sp.Symbol("K2"))[0]
ok = sp.simplify(ksq - k0 ** 2 * (1 + eps_s * (c0 - c1))) == 0
worst = 0.0
bound_ok = True
regimes = []
for (eps_target, psi0) in [(0.05, 0.3), (0.3, 2.0), (0.49, -1.0), (0.45, np.pi)]:
    p0 = rng.normal(size=6)
    a, b, m, I0 = rng.normal(), rng.normal(), 1.2, 0.9
    R0, phi0 = pendulum_data(p0, a, b)
    ths = (psi0 - phi0) / 2
    pst = expG(-ths) @ p0
    kap0 = np.sqrt(R0 / (m * I0 * eps_target))
    sol = integrate(quad_closure(I0, lorentzN(a, b, m)), np.concatenate(([0.0, I0 * kap0], pst, np.zeros(6))), 30.0)
    Yd = sol.sol(np.linspace(0, 30, 3001))
    _, phis = pendulum_data(pst, a, b)
    ksq_num = (Yd[1] / I0) ** 2
    psi = 2 * Yd[0] + phis
    pred = kap0 ** 2 * (1 + eps_target * (np.cos(psi0) - np.cos(psi)))
    worst = max(worst, np.max(np.abs(ksq_num - pred)) / kap0 ** 2)
    bound_ok &= np.max(np.abs(ksq_num / kap0 ** 2 - 1)) <= 2 * eps_target + 1e-12
    bound_ok &= np.all(Yd[1] > 0)                                       # rotates for all time when eps < 1/2
# the criterion: the rotor turns for all time iff eps (1 - cos psi0) < 1; otherwise the rate changes sign (libration)
p0 = rng.normal(size=6)
a, b, m, I0 = 0.7, 0.2, 1.0, 1.0
R0, phi0 = pendulum_data(p0, a, b)
for psi0, eps_target in [(np.pi, 0.45), (np.pi, 0.55), (np.pi / 2, 0.9), (np.pi / 2, 1.1), (0.0, 5.0)]:
    pst = expG(-(psi0 - phi0) / 2) @ p0
    expect_rotating = eps_target * (1 - np.cos(psi0)) < 1
    kap0 = np.sqrt(R0 / (m * I0 * eps_target))
    sol = integrate(quad_closure(I0, lorentzN(a, b, m)), np.concatenate(([0.0, I0 * kap0], pst, np.zeros(6))), 60.0)
    hh = sol.sol(np.linspace(0, 60, 6001))[1]
    regimes.append((np.all(hh > 0)) == expect_rotating)
info(f"kappa^2 law along 4 flows: max deviation {worst:.1e} (units kappa0^2); |kappa^2/kappa0^2-1| <= 2 eps held: {bound_ok}; "
     f"criterion eps(1-cos psi0) < 1 decides rotation vs libration in 5 cases (psi0 = pi: eps 0.45/0.55; psi0 = pi/2: "
     f"0.9/1.1; psi0 = 0: eps 5 rotates): {all(regimes)}")
report("d3-11", ok and worst < 1e-9 and bound_ok and all(regimes),
       "prop:multiscale (restates Cor 17.15): kappa^2 = kappa0^2[1 + eps(cos psi0 - cos psi)], eps = R0/(m I0 kappa0^2) = "
       "U_max/K_rot(0); |kappa^2/kappa0^2 - 1| <= 2 eps and perpetual rotation for eps < 1/2; libration once "
       "eps(1 - cos psi0) > 1 (exact + 9 integrations)")

# ============================================================================= d3-12 the clamped rate as the heavy-rotor limit
# along a rotating solution kappa(tau) = h/I0 and ell(tau) = c/kappa(tau); relative spread of ell is O(eps)
spreads = []
for eps_target in [1e-1, 1e-2, 1e-3]:
    p0 = rng.normal(size=6)
    a, b, m, I0 = 0.5, 0.5, 1.0, 1.0
    R0, phi0 = pendulum_data(p0, a, b)
    kap0 = np.sqrt(R0 / (m * I0 * eps_target))
    sol = integrate(quad_closure(I0, lorentzN(a, b, m)), np.concatenate(([0.0, I0 * kap0], p0, np.zeros(6))), 20.0 / kap0 * 10)
    kk = sol.sol(np.linspace(0, 200.0 / kap0, 20001))[1] / I0
    ell = 1.0 / kk                                                    # c = 1
    spreads.append((eps_target, (ell.max() - ell.min()) / ell.mean()))
ok = all(abs(s / ept - 1) <= 2 * ept for ept, s in spreads) and spreads[2][1] < spreads[1][1] < spreads[0][1]
info("relative spread of ell = c/kappa along the flow: " + ", ".join(f"eps={ept:g}: {s:.2e}" for ept, s in spreads))
report("d3-12", ok, "heavy rotor (interpretive): a constant 'c/ell' is the heavy-rotor limit: the relative spread of ell = c/kappa "
       "along a rotating flow is eps(1 + O(eps)), |spread/eps - 1| <= 2 eps, and vanishes with eps (numerical, 3 values of eps)")

# ============================================================================= d3-13 rest energy as a rotation rate (inherited)
ms8 = sp.Symbol("m", positive=True)
lam = sp.Symbol("lambda", real=True)
J8s, e1s, T8s = sp.Matrix(J8), sp.Matrix(e[0]), sp.Matrix(T8)
ok = (J8s * e1s == T8s) and (T8s * T8s == -sp.eye(8)) and (T8s.T == -T8s)
JH0 = J8s * (ms8 * e1s)                                                # H(0) = m gamma^0 = m e_1
Psi = sp.cos(ms8 * lam) * sp.eye(8) - sp.sin(ms8 * lam) * T8s          # proposed flow operator
ok &= is_zero(sp.diff(Psi, lam) + JH0 * Psi) and Psi.subs(lam, 0) == sp.eye(8)
# frame zero-momentum orbit X = e^{-kappa G tau} X0: alpha_+ X = 0 and X'' = -kappa^2 X, period 2 pi/kappa
kap, tau = sp.symbols("kappa tau", positive=True)
Xs = sp.Matrix(sp.symbols("X1:7", real=True))
eGt = sp.cos(kap * tau) * I6 - sp.sin(kap * tau) * G
Xt = eGt * Xs
ok &= is_zero(sp.diff(Xt, tau) + kap * G * Xt) and is_zero(sp.diff(Xt, tau, 2) + kap ** 2 * Xt)
ok &= is_zero(eGt.subs(tau, 2 * sp.pi / kap) - I6)
# the other sense: X = e^{+kappa G tau} X0 is a duality rotation at the same rate, X'' = -kappa^2 X, but alpha_+ X = 2 kappa G X != 0
eGp = sp.cos(kap * tau) * I6 + sp.sin(kap * tau) * G
Xp = eGp * Xs
ok &= is_zero(sp.diff(Xp, tau, 2) + kap ** 2 * Xp) and is_zero(sp.diff(Xp, tau) + kap * G * Xp - 2 * kap * G * Xp)
ok &= not is_zero(2 * kap * G * Xs)
report("d3-13", ok, "prop:tworot: Dirac at rest: J e1 = T, Psi(lambda) = cos(m lambda)Psi0 - sin(m lambda) T Psi0 = e^{-m lambda T}Psi0 "
       "(rate m, i.e. mc^2/hbar; inherited); frame zero-momentum orbit X = e^{-kappa G tau}X0 has alpha_+X = 0, rate kappa, and the "
       "other sense X = e^{+kappa G tau}X0 turns at the same rate with momentum alpha_+X = 2 kappa G X != 0 (restates Prop 16.5(i)-(ii); exact)")

# ============================================================================= d3-14 the matching hbar kappa = m c^2 (interpretive)
hb, mm = sp.symbols("hbar m", positive=True)
ellsol = sp.solve(sp.Eq(hb * cs / ells, mm * cs ** 2), ells)
ok = len(ellsol) == 1 and sp.simplify(ellsol[0] - hb / (mm * cs)) == 0
# the alternative matching of the Lorentz-invariant rate 2 kappa (q turns as e^{-2 i kappa tau}) to the de Broglie rate
ellsol2 = sp.solve(sp.Eq(2 * hb * cs / ells, mm * cs ** 2), ells)
ok &= sp.simplify(ellsol2[0] - 2 * hb / (mm * cs)) == 0
# the dynamics does not supply it: X'' = -kappa^2 X contains no m, and the action-preserving rescaling
# (tau, kappa, I0, m) -> (s tau, kappa/s, s I0, s m) of prop:scalings(ii)-(iii) keeps m kappa but multiplies m/kappa by s^2
ssc = sp.Symbol("s", positive=True)
ok &= sp.simplify((ssc * mm) * (kap / ssc) - mm * kap) == 0 and sp.simplify((ssc * mm) / (kap / ssc) - ssc ** 2 * mm / kap) == 0
ok &= sp.simplify(hb * (kap / ssc) / ((ssc * mm) * cs ** 2) - (hb * kap / (mm * cs ** 2)) / ssc ** 2) == 0   # hbar kappa/(m c^2) not invariant
report("d3-14", ok, "matching: hbar kappa = m c^2 with kappa = c/ell holds iff ell = hbar/(m c), the reduced Compton wavelength; matching "
       "the rate 2kappa of the invariant q instead gives ell = 2 hbar/(m c): nothing in the algebra decides; the action-preserving "
       "rescaling keeps m kappa and multiplies m/kappa (and hbar kappa/(m c^2)) by s^2 (resp. s^-2): an added identification (exact)")

# ============================================================================= d3-15 the two rotations are not one operator (obstruction)
# Ad_{exp(alpha T)} preserves V only for alpha in (pi/2)Z; e^{theta G} is an eta-isometry only for theta in pi Z
alpha = 0.37
eaT = np.cos(alpha) * np.eye(8) + np.sin(alpha) * T8                 # T^2 = -1
ok = vec_coeffs(np.rint(1e6 * (eaT @ e[3] @ eaT.T)).astype(np.int64)) is None
img = eaT @ e[3] @ eaT.T
resid = img - np.cos(2 * alpha) * e[3]
ok &= np.allclose(resid, np.sin(2 * alpha) * (T8 @ e[3]), atol=1e-12)
ok &= all(np.allclose(eaT @ e[i] @ eaT.T, e[i], atol=1e-12) for i in range(3))
EtaG = sp.simplify(eG.T * ETA * eG - (sp.cos(2 * th) * ETA + sp.sin(2 * th) * ETA * G))
ok &= EtaG.is_zero_matrix
ok &= (ETA * G).T == ETA * G and ETA * G != sp.zeros(6, 6)
report("d3-15", ok, "prop:norel: the Dirac rest flow e^{-m lambda T} is generated by a trivector: it fixes e1,e2,e3 and sends e_a -> "
       "cos(2vartheta)e_a + sin(2vartheta)T e_a (not a vector); e^{theta G} preserves eta only for theta in pi Z: no Spin(3,3) or "
       "O(3,3) map relates the two rates, so this route supplies no relation between m and kappa (independent inputs)")

# ============================================================================= d3-16 integer Clifford data; rescaling of (I0, m)
beta0 = {}
for i_ in range(6):
    for j_ in range(6):
        zi = np.zeros(3, complex); zj = np.zeros(3, complex)
        zi[i_ % 3] = 1 if i_ < 3 else 1j
        zj[j_ % 3] = 1 if j_ < 3 else 1j
        cr = np.cross(zi, zj)
        beta0[(i_, j_)] = np.concatenate((cr.real, cr.imag))
ints = [np.array(ETA.tolist(), float), Gn, Gn @ ETAn, ETAn @ Gn] + list(beta0.values()) + [np.array(Kc.tolist(), float)]
ok = all(np.array_equal(M, np.rint(M)) for M in ints)
# beta0 reproduces the so(1,3) brackets of thm:emfield with L_i = e_i, K_i = e_{i+3}
for (i_, j_, k_) in CYC:
    ok &= np.array_equal(beta0[(i_, j_)], np.eye(6)[k_]) and np.array_equal(beta0[(i_, j_ + 3)], np.eye(6)[k_ + 3])
    ok &= np.array_equal(beta0[(i_ + 3, j_ + 3)], -np.eye(6)[k_])
# eps -> eps/(s t) under (I0, m) -> (s I0, t m); the form of N (Lorentz class) and I0 > 0 are preserved
s_, t_ = sp.symbols("s t", positive=True)
R0s_ = sp.Symbol("R_0", positive=True)
epsf = lambda I0_, m_: R0s_ / (m_ * I0_ * k0 ** 2)
ok &= sp.simplify(epsf(s_ * I0s, t_ * ms) - epsf(I0s, ms) / (s_ * t_)) == 0
# at fixed h0 instead, kappa0 = h0/I0 -> kappa0/s and eps is multiplied by s/t (erratum 12 of the paper)
h0s = sp.Symbol("h_0", positive=True)
eps_h = lambda I0_, m_: R0s_ / (m_ * I0_ * (h0s / I0_) ** 2)
ok &= sp.simplify(eps_h(s_ * I0s, t_ * ms) - eps_h(I0s, ms) * s_ / t_) == 0
ok &= all((A * (Nsym / t_) + (Nsym / t_) * A.T).applyfunc(sp.simplify).is_zero_matrix for A in LG)
# numerical: measured 2*eps from the kappa^2 range of a rotating solution scales as 1/(s t)
p0 = rng.normal(size=6)
a, b, m, I0, kap0 = 0.4, 0.3, 1.0, 1.0, 3.0


def eps_measured(I0_, m_):
    R0, phi0 = pendulum_data(p0, a, b)
    sol = integrate(quad_closure(I0_, lorentzN(a, b, m_)), np.concatenate(([0.0, I0_ * kap0], p0, np.zeros(6))), 20.0)
    kk = (sol.sol(np.linspace(0, 20, 40001))[1] / I0_) ** 2
    return (kk.max() - kk.min()) / (2 * kap0 ** 2), R0 / (m_ * I0_ * kap0 ** 2)


e1m, e1f = eps_measured(I0, m)
e2m, e2f = eps_measured(2.0 * I0, 3.0 * m)
ok &= abs(e1m / e1f - 1) < 1e-6 and abs(e2m / e2f - 1) < 1e-6 and abs(e2f * 6.0 / e1f - 1) < 1e-12


def eps_measured_h0(I0_, m_, h0_):
    R0, phi0 = pendulum_data(p0, a, b)
    k0 = h0_ / I0_
    sol = integrate(quad_closure(I0_, lorentzN(a, b, m_)), np.concatenate(([0.0, h0_], p0, np.zeros(6))), 20.0)
    kk = (sol.sol(np.linspace(0, 20, 40001))[1] / I0_) ** 2
    return (kk.max() - kk.min()) / (2 * k0 ** 2), R0 / (m_ * I0_ * k0 ** 2)


e3m, e3f = eps_measured_h0(2.0 * I0, 3.0 * m, I0 * kap0)                       # same h0 as the first run
ok &= abs(e3m / e3f - 1) < 1e-6 and abs(e3f / e1f - 2.0 / 3.0) < 1e-12
info(f"eps measured from the kappa^2 range: {e1m:.8f} (formula {e1f:.8f}); after (I0,m)->(2I0,3m) at fixed kappa0: "
     f"{e2m:.8f} = eps/6; at fixed h0: {e3m:.8f} = (2/3) eps")
report("d3-16", ok, "prop:scalings(i),(iv): eta, G, G eta^{-1}, eta G, K and the structure constants of beta0 are integer matrices; "
       "(I0, m) -> (s I0, t m) keeps N Lorentz class and I0 > 0 and multiplies eps by 1/(s t) at fixed initial rate kappa0 "
       "(h0 -> s h0), by s/t at fixed h0 (exact and measured)")

# ============================================================================= d3-17 kappa is the only scale of the interior
A6 = sp.Matrix(sp.symbols("A1:7", real=True))
B6 = sp.Matrix(sp.symbols("B1:7", real=True))
lm, sc = sp.symbols("lambda s", positive=True)
Xgen = sp.cos(kap * tau) * A6 + sp.sin(kap * tau) * B6
ok = is_zero(sp.diff(Xgen, tau, 2) + kap ** 2 * Xgen)                      # thm:neutral: X'' = -kappa^2 X
# split-form second-order equation with G = G^dagger: X'' + 2 kappa a X' - kappa^2 G^dagger G X, a = 0, G^dagger G = -1
Gdag = ETA.inv() * G.T * ETA
ok &= (G - Gdag).is_zero_matrix and Gdag * G == -I6
# X -> lambda X maps solutions to solutions; the Lagrangian scales by lambda^2, absorbed by m -> m/lambda^2
Lsplit = lambda X, Xd, m_: m_ / 2 * ((Xd + kap * G * X).T * ETA * (Xd + kap * G * X))[0]
Xv = sp.Matrix(sp.symbols("x1:7", real=True)); Vv = sp.Matrix(sp.symbols("v1:7", real=True))
ok &= sp.expand(Lsplit(lm * Xv, lm * Vv, ms / lm ** 2) - Lsplit(Xv, Vv, ms)) == 0
# (tau, kappa) -> (s tau, kappa/s): Y(tau') = X(tau'/s) solves Y'' = -(kappa/s)^2 Y
taup = sp.Symbol("tau_p", positive=True)
Y = Xgen.subs(tau, taup / sc)
ok &= is_zero(sp.diff(Y, taup, 2) + (kap / sc) ** 2 * Y)
# closure: (tau, I0, m) -> (s tau, s I0, s m) maps solutions to solutions with (h, p) unchanged and eps invariant
p0 = rng.normal(size=6)
a, b, m, I0, h0, sfac = 0.9, -0.4, 1.1, 0.7, 1.6, 2.5
solA = integrate(quad_closure(I0, lorentzN(a, b, m)), np.concatenate(([0.0, h0], p0, np.zeros(6))), 10.0)
solB = integrate(quad_closure(sfac * I0, lorentzN(a, b, sfac * m)), np.concatenate(([0.0, h0], p0, np.zeros(6))), 10.0 * sfac)
tt = np.linspace(0, 10, 201)
dev = np.max(np.abs(solB.sol(sfac * tt)[:8] - solA.sol(tt)[:8]))
ok &= dev < 1e-9
ok &= sp.simplify(epsf(sc * I0s, sc * ms).subs(k0, k0 / sc) - epsf(I0s, ms)) == 0
# the interior action int L dtau is invariant under (tau, kappa, m) -> (s tau, kappa/s, s m): L' = (s m/2) eta(u/s, u/s), dtau' = s dtau
L_scaled = sp.expand((sc * ms) / 2 * ((Vv / sc + (kap / sc) * G * Xv).T * ETA * (Vv / sc + (kap / sc) * G * Xv))[0] * sc)
ok &= sp.simplify(L_scaled - Lsplit(Xv, Vv, ms)) == 0
# the closure Hamiltonian scales as H' = H/s with dtau' = s dtau, so int (h dtheta - H dtau) is invariant; m kappa is invariant
Hs_ = hs ** 2 / (2 * sc * I0s) + (ps.T * (Nsym.subs(ms, sc * ms)) * ps)[0] / 2
ok &= sp.simplify(Hs_ - Hsym / sc) == 0 and sp.simplify((sc * ms) * (kap / sc) - ms * kap) == 0
info(f"closure time rescaling s = {sfac}: max deviation of (theta, h, p) = {dev:.1e}")
report("d3-17", ok, "prop:scalings(ii),(iii): the interior law X'' = -kappa^2 X is invariant under X -> sigma X (L -> sigma^2 L, absorbed "
       "in m) and (tau, kappa) -> (s tau, kappa/s), which with m -> s m preserves the action; the closure under (tau, I0, m) -> "
       "(s tau, s I0, s m), with H' = H/s, which fixes eps, m kappa and the action: kappa only sets the unit of tau")

# ============================================================================= d3-18 no length or mass scale from the algebra (inherited)
# Schur: the commutant of so(3) (vector representation) in M_3(R) is R 1
Cm = sp.Matrix(3, 3, sp.symbols("c0:9"))
gens3 = [sp.Matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]]), sp.Matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
         sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
eqs = []
for g_ in gens3:
    eqs += list(Cm * g_ - g_ * Cm)
solC = sp.solve(eqs, list(Cm), dict=True)[0]
Csol = Cm.subs(solC)
ok = len(Csol.free_symbols) == 1 and Csol == Csol[0, 0] * sp.eye(3)
# dimensional analysis: c^a hbar^b is never a length, a mass or a time; with m one length, hbar/(m c), up to powers
Dch = sp.Matrix([[0, 1], [1, 2], [-1, -1]])                   # columns: [c], [hbar] in (M, L, T)
for target in ([0, 1, 0], [1, 0, 0], [0, 0, 1]):
    aug = Dch.row_join(sp.Matrix(target))
    ok &= aug.rank() > Dch.rank()
Dchm = sp.Matrix([[0, 1, 1], [1, 2, 0], [-1, -1, 0]])         # columns: [c], [hbar], [m]
xs = Dchm.LUsolve(sp.Matrix([0, 1, 0]))
ok &= list(xs) == [-1, 1, -1] and Dchm.rank() == 3            # length = c^-1 hbar m^-1
# scaling orbit: every Clifford datum is invariant while kappa -> kappa/s covers (0, oo): a condition built from the data
# that selected kappa would hold on the whole orbit (checked: the data used above contain no symbol)
ok &= all(not sp.Matrix(M).free_symbols for M in [ETA, G, Kc, S6] + LG)
report("d3-18", ok, "thm:noscale (inherited dimensional analysis): Schur gives C = c 1 (commutant of so(3) one-dimensional); c and "
       "hbar form no length, mass or time, and with m the only length is hbar/(m c); the Clifford data are dimensionless integers "
       "invariant under the scaling of d3-17, so nothing in the algebra fixes ell (equivalently the mass scale)")

# ============================================================================= d3-19 frame drift (ex:drift)
mpm.mp.dps = 30
Kc2, Ec2 = mpm.ellipk(-2), mpm.ellipe(-2)
target_full = 8 * Kc2 - 4 * Ec2
target_half = 4 * Kc2 - 2 * Ec2
quad_full = mpm.quad(lambda t: mpm.cos(2 * t) / mpm.sqrt(1 + 2 * mpm.sin(t) ** 2), [0, mpm.pi / 2, mpm.pi, 3 * mpm.pi / 2, 2 * mpm.pi])
lam_thm = mpm.quad(lambda ps_: mpm.cos(ps_) / mpm.sqrt(1 + (1 - mpm.cos(ps_))), [0, mpm.pi, 2 * mpm.pi]) / 2   # thm:freedrift
ok = abs(target_full - quad_full) < mpm.mpf(10) ** -25 and abs(target_half - lam_thm) < mpm.mpf(10) ** -25
ok &= abs(target_full - mpm.mpf("0.6336081022")) < 5e-11
# full system: p0 = e^1, m = I0 = kappa0 = 1, N = eta^{-1} (a = 1, b = 0): w = 1, phi0 = 0, eps = 1
p0 = np.eye(6)[0]
ok &= abs(wfun(p0) - 1) < 1e-15
f = quad_closure(1.0, lorentzN(1.0, 0.0, 1.0))
ev_full = lambda t, y: y[0] - 2 * np.pi
ev_full.terminal, ev_full.direction = True, 1
ev_half = lambda t, y: y[0] - np.pi
ev_half.direction = 1
sol = solve_ivp(f, (0, 20), np.concatenate(([0.0, 1.0], p0, np.zeros(6))), method="DOP853", rtol=1e-13, atol=1e-14,
                events=[ev_full, ev_half], dense_output=True)
yf = sol.y_events[0][0]
yh = sol.y_events[1][0]
dx_full = yf[8:14]
dx_half = yh[8:14]
err_full = np.linalg.norm(dx_full - float(target_full) * np.eye(6)[0])
err_half = np.linalg.norm(dx_half - float(target_half) * np.eye(6)[0])
ret = max(abs(yf[1] - 1.0), np.linalg.norm(yf[2:8] - p0))
half_flip = max(abs(yh[1] - 1.0), np.linalg.norm(yh[2:8] + p0))
info(f"8K(-2)-4E(-2) = {mpm.nstr(target_full, 16)}; integrated over theta in [0,2pi]: dx = {dx_full[0]:.12f} e1 "
     f"(error {err_full:.1e}); half turn: {dx_half[0]:.12f} e1 = 4K-2E (error {err_half:.1e}); "
     f"after a full turn |h-h0|,|p-p0| <= {ret:.1e}; after a half turn p = -p0 to {half_flip:.1e}")
ok &= err_full < 1e-10 and err_half < 1e-10 and ret < 1e-10 and half_flip < 1e-10
report("d3-19", ok, "ex:drift (restates Ex 17.17): displacement per full turn of the rotor (theta: 0 -> 2pi) is (8K(-2)-4E(-2))e1 = 0.6336081022 e1 "
       "(mpmath, two quadratures) and by integrating (theta,h,p,x) to <1e-10; per period of the rate (theta by pi) it is "
       "(4K(-2)-2E(-2))e1, where p = -p0 and h = h0")

# ============================================================================= d3-20 the half turn
ok = eG.subs(th, sp.pi) == -I6 and eG.subs(th, 2 * sp.pi) == I6
ok &= all(np.array_equal(GAM8 @ e[a_] + e[a_] @ GAM8, np.zeros((8, 8), dtype=np.int64)) for a_ in range(6))
ok &= np.array_equal(GAM8 @ GAM8, I8)
VOL8 = e[0] @ e[1] @ e[2] @ e[3] @ e[4] @ e[5]
ok &= np.array_equal(GAM8, VOL8)                                       # Gamma = J e1e4e5e6 = +e1...e6 in the realisation
ok &= all(np.array_equal(VOL8 @ e[a_] @ VOL8, -e[a_]) for a_ in range(6))  # Ad_Gamma = -1 on V (Gamma^{-1} = Gamma)
# U(theta + pi) = U(theta) for every symmetric N (H even in p); the leaf circle {e^{theta G} p0} has period exactly 2 pi
Ugen = lambda tt_: sp.expand(((sp.cos(tt_) * I6 - sp.sin(tt_) * G) * ps).T * Nfree * ((sp.cos(tt_) * I6 - sp.sin(tt_) * G) * ps))[0]
ok &= sp.simplify(sp.expand_trig(Ugen(th + sp.pi) - Ugen(th))) == 0
pr = sp.Matrix([1, 2, 0, -1, 3, 1])
# (e^{theta G} - 1) p = (cos theta - 1) p + sin theta G p with p orthogonal to G p and |G p| = |p| (G orthogonal), so it
# vanishes iff cos theta = 1 and sin theta = 0
ok &= (pr.T * G * pr)[0] == 0 and (G * pr).norm() == pr.norm() and G.T * G == I6
report("d3-20", ok, "prop:halfturn: e^{pi G} = -1 and e^{2pi G} = 1 on V; Gamma = e1e2...e6 (= J e1e4e5e6 in the realisation) "
       "anticommutes with every e_a and squares to 1, so Ad_Gamma = -1 implements the half turn on spinors; U(theta+pi) = U(theta) "
       "for every N; the leaf circle has period exactly 2pi (exact)")

# ============================================================================= d3-21 quantum rotor (conditional)
def rotor_matrix(nmax, hbar, I0, R0m, phi0, delta=0.0):
    ns = np.arange(-nmax, nmax + 1)
    Hm = np.diag(hbar ** 2 * (ns + delta) ** 2 / (2 * I0)).astype(complex)
    amp = R0m / 4.0                                          # (R0/2m) cos(2 theta + phi0) = (R0/4m)(e^{i..} + e^{-i..})
    for k in range(len(ns) - 2):
        Hm[k + 2, k] += amp * np.exp(1j * phi0)             # <n+2| e^{2 i theta} |n>
        Hm[k, k + 2] += amp * np.exp(-1j * phi0)
    return ns, Hm


hbar, I0 = 1.0, 1.0
ok = True
for dlt in (0.0, 0.3):                                       # (i) flat leaves: exact ladder hbar^2 (n+delta)^2/(2 I0)
    ns, Hm = rotor_matrix(40, hbar, I0, 0.0, 0.0, dlt)
    ev = np.sort(np.linalg.eigvalsh(Hm))
    ok &= np.allclose(ev, np.sort(hbar ** 2 * (ns + dlt) ** 2 / (2 * I0)), atol=1e-12)
# (ii) R0 > 0: the potential couples n only to n +- 2, for every delta; deep-well gap in each sector
Om = np.sqrt(2 * 2000.0 / I0)                               # Omega^2 = 2 R0/(m I0) with R0/m = 2000
pred_gap = hbar * Om - hbar ** 2 / (2 * I0)
gaps = {}
for dlt in (0.0, 0.3):
    ns, Hm = rotor_matrix(60, hbar, I0, 2000.0, 0.7, dlt)
    even = ns % 2 == 0
    ok &= np.allclose(Hm[np.ix_(even, ~even)], 0) and np.allclose(Hm[np.ix_(~even, even)], 0)
    ev_even = np.sort(np.linalg.eigvalsh(Hm[np.ix_(even, even)]))
    ev_odd = np.sort(np.linalg.eigvalsh(Hm[np.ix_(~even, ~even)]))
    gaps[dlt] = (ev_even[1] - ev_even[0], ev_odd[1] - ev_odd[0], abs(ev_even[0] - ev_odd[0]))
    ok &= all(abs(g - pred_gap) / (hbar * Om) < 2e-3 for g in gaps[dlt][:2])
    ok &= abs(ev_even[0] - (-2000.0 / 2 + hbar * Om / 2)) / (hbar * Om) < 2e-2
ok &= gaps[0.0][2] < 1e-8 * hbar * Om                        # delta = 0: sector ground states degenerate up to tunnelling
ok &= gaps[0.3][2] < 1e-6 * hbar * Om
# the parity of n labels the half-turn eigenvalue: e^{i(n+delta)(theta+pi)} = (-1)^n e^{i pi delta} e^{i(n+delta)theta}
nn = sp.Symbol("n", integer=True)
dd = sp.Symbol("delta", real=True)
ratio = sp.exp(sp.I * (nn + dd) * (th + sp.pi)) / (sp.exp(sp.I * sp.pi * dd) * sp.exp(sp.I * (nn + dd) * th))
ok &= sp.simplify(ratio - (-1) ** nn) == 0 or sp.simplify(ratio) == (-1) ** nn
# duality-class leaves: U is the constant p0^T N p0 / 2 (d3-7), so the ladder is shifted by it
ok &= sp.simplify(sp.expand_trig(Udual - (ps.T * Ndual * ps)[0] / 2)) == 0
# Bohr correspondence and the two matchings: hbar kappa_1 = m c^2 and h = hbar give M_rot = m and E_1 = (1/2) m c^2 = K_rot(h = hbar)
En = lambda k: hb ** 2 * (k + dd) ** 2 / (2 * I0s)
kn = lambda k: (k + dd) * hb / I0s
ok &= sp.simplify(En(nn + 1) - En(nn) - hb * (kn(nn) + kn(nn + 1)) / 2) == 0
En0 = lambda k: En(k).subs(dd, 0)
kn0 = lambda k: kn(k).subs(dd, 0)
I0_match = sp.solve(sp.Eq(kn0(1), cs / (hb / (mm * cs))), I0s)[0]
ok &= sp.simplify(I0_match - hb ** 2 / (mm * cs ** 2)) == 0
ok &= sp.simplify(I0_match / (hb / (mm * cs)) ** 2 - mm) == 0 and sp.simplify(En0(1).subs(I0s, I0_match) - mm * cs ** 2 / 2) == 0
ok &= sp.simplify(En0(1) - hb * kn0(1) / 2) == 0                # E_1 = K_rot = (1/2) h kappa at h = hbar: the 1/2 is built in
info(f"R0/m = 2000, hbar = I0 = 1: Omega = {Om:.4f}; lowest gaps (even, odd sector) at delta = 0: {gaps[0.0][0]:.4f}, "
     f"{gaps[0.0][1]:.4f}; at delta = 0.3: {gaps[0.3][0]:.4f}, {gaps[0.3][1]:.4f}; vs hbar Omega - hbar^2/2I0 = {pred_gap:.4f}; "
     f"even/odd ground states split by {gaps[0.0][2]:.1e} (delta = 0), {gaps[0.3][2]:.1e} (delta = 0.3)")
report("d3-21", ok, "prop:ladder (constructed, under ass:quant): flat leaves give E_n = hbar^2 (n+delta)^2/(2I0) (plus p0^T N p0/2 on "
       "duality-class leaves), kappa_n = (n+delta) hbar/I0, E_{n+1}-E_n = hbar(kappa_n+kappa_{n+1})/2; the potential couples n to "
       "n+-2 only, so for every delta the parity of n (half-turn eigenvalue +-e^{i pi delta}) is conserved; deep-well gap "
       "hbar Omega - hbar^2/(2I0) in each sector (delta = 0, 0.3); with the separate matchings ell = hbar/(mc), h = hbar: "
       "I0 = hbar^2/(mc^2), M_rot = m, E_1 = K_rot = mc^2/2")

# ============================================================================= d3-22 the frame field is not frozen (cor:nogen(ii))
# clamped split-form interior, Proposition 16.5(ii): z(X) = c_- e^{-i kappa tau} + c_+ e^{i kappa tau}
cm_ = sp.Matrix(sp.symbols("cmr1:4", real=True)) + sp.I * sp.Matrix(sp.symbols("cmi1:4", real=True))
cp_ = sp.Matrix(sp.symbols("cpr1:4", real=True)) + sp.I * sp.Matrix(sp.symbols("cpi1:4", real=True))
tq = sp.Symbol("tau", real=True)
kq = sp.Symbol("kappa", positive=True)
zX = cm_ * sp.exp(-sp.I * kq * tq) + cp_ * sp.exp(sp.I * kq * tq)
conj = lambda v: v.applyfunc(sp.conjugate)
E6X = sp.expand((conj(zX).T * zX)[0] / 2)
E6X_formula = ((conj(cm_).T * cm_)[0] + (conj(cp_).T * cp_)[0]) / 2 + sp.re(sp.expand((conj(cm_).T * cp_)[0] * sp.exp(2 * sp.I * kq * tq)))
ok = sp.simplify(sp.expand_complex(E6X - E6X_formula)) == 0
qX = sp.expand((zX.T * zX)[0])
qX_formula = (cm_.T * cm_)[0] * sp.exp(-2 * sp.I * kq * tq) + 2 * (cm_.T * cp_)[0] + (cp_.T * cp_)[0] * sp.exp(2 * sp.I * kq * tq)
ok &= sp.simplify(sp.expand(qX - qX_formula)) == 0
# E6(X) is constant iff conj(c_-).c_+ = 0: its tau-derivative is -2 kappa Im(conj(c_-).c_+ e^{2 i kappa tau})
dE6 = sp.diff(E6X_formula, tq)
ok &= sp.simplify(sp.expand_complex(dE6 + 2 * kq * sp.im(sp.expand((conj(cm_).T * cp_)[0] * sp.exp(2 * sp.I * kq * tq))))) == 0
# exact example c_- = (1, i, 0), c_+ = (1, 0, 0): |q(z(X))|^2 = 9 - 8 sin^2(kappa tau)
ex_sub = {**{v: 0 for v in cm_.free_symbols | cp_.free_symbols}}
ex_sub.update({sp.Symbol("cmr1", real=True): 1, sp.Symbol("cmi2", real=True): 1, sp.Symbol("cpr1", real=True): 1})
qex = sp.expand(qX.subs(ex_sub))
absq2 = sp.simplify(sp.expand_complex(qex * sp.conjugate(qex)))
ok &= sp.simplify(sp.expand_trig(absq2 - (9 - 8 * sp.sin(kq * tq) ** 2))) == 0
ok &= sp.simplify((conj(cm_).T * cp_)[0].subs(ex_sub)) == 1                  # this orbit has E6(X) = 3/2 + cos(2 kappa tau)
# free-rotor drift of ex:drift with x0 = 0: the frame field X = e^{-theta G} x has E0(X) = 0 at tau = 0 and (1/2)(3 Dx)^2 at theta = 6 pi
p0 = np.eye(6)[0]
f22 = quad_closure(1.0, lorentzN(1.0, 0.0, 1.0))
ev6 = lambda t, y: y[0] - 6 * np.pi
ev6.terminal, ev6.direction = True, 1
sol = solve_ivp(f22, (0, 60), np.concatenate(([0.0, 1.0], p0, np.zeros(6))), method="DOP853", rtol=1e-13, atol=1e-14,
                events=[ev6], dense_output=True)
yend = sol.y_events[0][0]
Xend = expG(-yend[0]) @ yend[8:14]
E0X = lambda X: 0.5 * abs(complex(np.dot(X[:3] + 1j * X[3:], X[:3] + 1j * X[3:])))
E6X_n = lambda X: 0.5 * float(X @ X)
Dx = float(8 * mpm.ellipk(-2) - 4 * mpm.ellipe(-2))
ts22 = np.linspace(0, sol.t_events[0][0], 3001)
E0Xs = np.array([E0X(expG(-sol.sol(t)[0]) @ sol.sol(t)[8:14]) for t in ts22])
ok &= abs(E0X(Xend) - 0.5 * (3 * Dx) ** 2) < 1e-9 and E0Xs[0] == 0.0 and abs(E6X_n(Xend) - 0.5 * (3 * Dx) ** 2) < 1e-9
ok &= abs(E0fun(yend[2:8]) - 0.5) < 1e-12                                    # while the momentum field keeps E0(pi) = 1/2
# growth like k^2 on the turns theta = 2 pi k
E0_turns = []
for k in (1, 2, 3):
    tk = brentq(lambda t: sol.sol(t)[0] - 2 * np.pi * k, 0, sol.t_events[0][0] + 1e-9 if k == 3 else sol.t_events[0][0], xtol=1e-14)
    yk = sol.sol(tk)
    E0_turns.append(E0X(expG(-yk[0]) @ yk[8:14]))
ok &= all(abs(E0_turns[k - 1] - 0.5 * (k * Dx) ** 2) < 1e-8 for k in (1, 2, 3))
info(f"clamped interior: E6(X) = (|c-|^2+|c+|^2)/2 + Re(conj(c-).c+ e^(2i kappa tau)), q(z(X)) expansion, |q|^2 = 9 - 8 sin^2 "
     f"(exact); free-rotor drift: E0(X) at theta = 2pi k, k = 1, 2, 3: {', '.join(f'{v:.4f}' for v in E0_turns)} "
     f"(= (k Dx)^2/2), max on the way {E0Xs.max():.3f}; E0(pi) stays 1/2")
report("d3-22", ok, "cor:nogen(ii): the FRAME field is not frozen: in the clamped interior (Prop 16.5(ii)) E6(X) is constant iff "
       "conj(c_-).c_+ = 0 and |q(z(X))|^2 = 9 - 8 sin^2(kappa tau) for c_- = (1,i,0), c_+ = (1,0,0); under the free-rotor drift of "
       "ex:drift (x0 = 0) E0(X) goes from 0 to (1/2)(3 Dx)^2 = 1.807 in three full turns, growing like k^2, while E0(pi) = 1/2")

# ============================================================================= d3-23 compact transports (assessment; restated from d1, d4)
# one constituent, S = b + kappa G, M = [[-S, 1/m], [0, S^dagger]] (d4 lem:M); bounded iff M semisimple with imaginary spectrum
lamM = sp.Symbol("lambda")
wM = sp.Symbol("omega", real=True)


def crossmat(v):
    a_, b_, c_ = [sp.Rational(x) for x in v]
    return sp.Matrix([[0, -c_, b_], [c_, 0, -a_], [-b_, a_, 0]])


def compact_b(ad_, aa_):
    """b = (A_d + A_a, A_d - A_a) on the temporal and spatial triples (rotation vectors ad_, aa_)."""
    Ad_, Aa_ = crossmat(ad_), crossmat(aa_)
    return sp.diag(Ad_ + Aa_, Ad_ - Aa_)


def interior_M(bm, kap_=2, m_=1):
    S_ = bm + kap_ * G
    Sd_ = ETA.inv() * S_.T * ETA
    return sp.Matrix(sp.BlockMatrix([[-S_, sp.eye(6) / m_], [sp.zeros(6), Sd_]])), S_


def semisimple(Mx):
    cp = sp.Poly(Mx.charpoly(lamM).as_expr(), lamM)
    r = sp.quo(cp, sp.gcd(cp, cp.diff(lamM)))
    R = sp.zeros(*Mx.shape)
    for co in r.all_coeffs():
        R = R * Mx + co * sp.eye(Mx.shape[0])
    return R.is_zero_matrix


def imaginary_spectrum(S_):
    cp = sp.Poly(S_.charpoly(lamM).as_expr(), lamM)
    r = sp.Poly(sp.quo(cp, sp.gcd(cp, cp.diff(lamM))), lamM)            # squarefree part
    f = sp.expand(r.as_expr().subs(lamM, sp.I * wM))
    g = sp.gcd(sp.Poly(sp.re(f), wM), sp.Poly(sp.im(f), wM))
    return g.degree() == r.degree() and g.count_roots() == r.degree()


def freqs(S_):
    return sorted({round(abs(complex(x).imag), 10) for x in np.linalg.eigvals(np.array(S_.tolist(), dtype=float))})


ok = True
results = {}
cases = {"diagonal": ((0, 0, 1), (0, 0, 0)), "anti-diagonal": ((0, 0, 0), (1, 0, 0)),
         "screened a.c=0 (1)": ((0, 0, 1), (1, 0, 0)), "screened a.c=0 (2)": ((1, 2, 0), (2, -1, 3)),
         "generic a.c!=0 (1)": ((0, 0, 1), (1, 0, 1)), "generic a.c!=0 (2)": ((1, 2, 0), (2, 1, 3))}
for name, (ad_, aa_) in cases.items():
    bm = compact_b(ad_, aa_)
    ok &= (bm.T * ETA + ETA * bm).is_zero_matrix                               # compact element of so(3,3)
    Mx, S_ = interior_M(bm)
    results[name] = (semisimple(Mx) and imaginary_spectrum(S_), (bm * G - G * bm).is_zero_matrix, freqs(S_))
ok &= results["diagonal"][0] and results["diagonal"][1] and results["diagonal"][2] == [1.0, 2.0, 3.0]   # kappa, kappa +- |a_d|
ok &= (not results["anti-diagonal"][0]) and not results["anti-diagonal"][1]
ok &= abs(max(results["anti-diagonal"][2]) - np.sqrt(5.0)) < 1e-9                                  # sqrt(kappa^2 + |a_a|^2)
ok &= all(results[k][0] and not results[k][1] for k in ("screened a.c=0 (1)", "screened a.c=0 (2)"))
ok &= all(not results[k][0] for k in ("generic a.c!=0 (1)", "generic a.c!=0 (2)"))
# three constituents, so(4)_Delta-type paired rotation: keeps (1/2)|Z|^2 and q3, moves sum_k E0^(k) from 1/2 to 3/2 and the
# Lorentz length of the summed density (sum E6; sum S_t) from sqrt5/2 to 3/2 (d1:prop:bfield(3)-(4); x3-22)
z2 = sp.Matrix([1, sp.I, 0]); z3 = sp.Matrix([1, 0, 0])
cq, sq = sp.cos(sp.pi / 4), sp.sin(sp.pi / 4)
z2n = sp.Matrix([cq * z2[0] - sq * z3[0], z2[1], z2[2]]); z3n = sp.Matrix([sq * z2[0] + cq * z3[0], z3[1], z3[2]])
E6c = lambda z: sp.simplify((conj(z).T * z)[0] / 2)
E0c = lambda z: sp.simplify(sp.Abs((z.T * z)[0]) / 2)
St = lambda z: -(sp.I / 2 * conj(z).cross(z)).applyfunc(sp.simplify)          # textbook Poynting vector S_t = -S
Llen = lambda zs: sp.sqrt(sum(E6c(z) for z in zs) ** 2 - sum((St(z) for z in zs), sp.zeros(3, 1)).dot(sum((St(z) for z in zs), sp.zeros(3, 1))))
ok &= sp.simplify(E6c(z2) + E6c(z3) - E6c(z2n) - E6c(z3n)) == 0
ok &= sp.simplify((z2.T * z2)[0] + (z3.T * z3)[0] - (z2n.T * z2n)[0] - (z3n.T * z3n)[0]) == 0
ok &= sp.simplify(E0c(z2) + E0c(z3) - sp.Rational(1, 2)) == 0 and sp.simplify(E0c(z2n) + E0c(z3n) - sp.Rational(3, 2)) == 0
ok &= sp.simplify(Llen([z2, z3]) - sp.sqrt(5) / 2) == 0 and sp.simplify(Llen([z2n, z3n]) - sp.Rational(3, 2)) == 0
info("one constituent, kappa = 2, m = 1: " + "; ".join(f"{k}: bounded={v[0]}, [b,G]=0: {v[1]}, rates {v[2]}" for k, v in results.items()))
report("d3-23", ok, "assessment (restated from d1:prop:bfield and d4): compact b = (A_d+A_a, A_d-A_a): diagonal part alone gives the bounded "
       "lines kappa, kappa +- |a_d|; anti-diagonal alone Jordan blocks at sqrt(kappa^2+|a_a|^2); a_d.a_a = 0, a_d != 0 bounded "
       "although [b,G] != 0; generic compact b unbounded (exact semisimplicity); a paired rotation of two constituents keeps "
       "|Z|^2/2 and q3 but moves sum E0 from 1/2 to 3/2 and the Lorentz length from sqrt5/2 to 3/2 (exact)")

# ----------------------------------------------------------------------------- summary
if FAILED:
    print(f"FAIL summary: {len(FAILED)} claim(s) failed: {', '.join(FAILED)}")
    sys.exit(1)
print("ALL d3 CLAIMS PASS")
sys.exit(0)
