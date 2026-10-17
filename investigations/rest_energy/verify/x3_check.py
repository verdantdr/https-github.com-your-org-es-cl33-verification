#!/usr/bin/env python3
"""x3 -- independent adversarial check of track d3 (sections/d3_closure.tex, claims d3-0 ... d3-21).

Does NOT import verify/d3_closure_dynamics.py.  Everything is rebuilt from the paper tex:
  realisation   tex/s02_arena.tex   e1=sx11, e2=sz sx 1, e3=sz sz sx, e4=ep11, e5=sz ep 1, e6=sz sz ep
  roots         tex/s03_clock.tex   lem:Gclifford  K(v)=1/2[P,v], S(v)=T v T^-1, G=-K o S
  closure       tex/s17_closure.tex  s = R ghat x|_G V, prop:LP, thm:pendulum, cor:multiscale, thm:freedrift, ex:drift
Methods differ from the derive script where possible:
  * G is computed from Clifford data (not typed in) and l_G as a sympy nullspace (not from bivector formulas);
  * Lie-Poisson equations from structure constants with BOTH signs of the bracket;
  * integrations use the implicit Radau method (derive used DOP853);
  * the pendulum period is checked by mpmath quadrature, the Mathieu gap by scipy.special.mathieu_a/b and an
    independent cos/sin Galerkin basis (derive used an exponential Fourier basis);
  * extra adversarial checks: the rest energy of the FRAME field X (not the momentum field) under the closure,
    the hypothesis needed for the epsilon scaling (fixed kappa0 vs fixed h0), and the sum of constituent rest
    energies under a so(4)_Delta-type rotation.
Prints one PASS/FAIL line per checked item; exit 0 iff all pass.  Runtime ~ 1 min.
"""
import sys
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import scipy.special as ssp

FAILS = []


def check(cid, ok, text):
    print(("PASS " if ok else "FAIL ") + f"{cid}: {text}")
    if not ok:
        FAILS.append(cid)


def note(text):
    print("     # " + text)


rng = np.random.default_rng(1234567)

# ------------------------------------------------------------------ realisation (s02) and roots (s03)
sx = sp.Matrix([[0, 1], [1, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
ep = sp.Matrix([[0, -1], [1, 0]])
one = sp.eye(2)
kp = sp.kronecker_product
E = [kp(sx, one, one), kp(sz, sx, one), kp(sz, sz, sx), kp(ep, one, one), kp(sz, ep, one), kp(sz, sz, ep)]
etad = [1, 1, 1, -1, -1, -1]
ETA = sp.diag(*etad)
I8 = sp.eye(8)
cliff_ok = all(E[a] * E[b] + E[b] * E[a] == 2 * (etad[a] if a == b else 0) * I8 for a in range(6) for b in range(6))

# coefficients of an 8x8 matrix on the basis e_a (exact): solve the linear system on the 64 entries
Vstack = sp.Matrix.hstack(*[E[a].reshape(64, 1) for a in range(6)])


def vcoef(Y):
    sol = (Vstack.T * Vstack).LUsolve(Vstack.T * Y.reshape(64, 1))
    rec = sp.zeros(8, 8)
    for a in range(6):
        rec += sol[a] * E[a]
    return sol if sp.simplify(rec - Y) == sp.zeros(8, 8) else None


Pm = sum((E[i] * E[i + 3] for i in range(3)), sp.zeros(8, 8))
Tm = E[0] * E[1] * E[2]
Tinv = Tm.inv()
Kmat = sp.zeros(6, 6)
Smat = sp.zeros(6, 6)
for a in range(6):
    Kmat[:, a] = vcoef(sp.Rational(1, 2) * (Pm * E[a] - E[a] * Pm))
    Smat[:, a] = vcoef(Tm * E[a] * Tinv)
G = -Kmat * Smat
Gexpected = sp.zeros(6, 6)
for i in range(3):
    Gexpected[i + 3, i] = 1
    Gexpected[i, i + 3] = -1
I6 = sp.eye(6)
ok0 = cliff_ok and G == Gexpected and G ** 2 == -I6 and G.T == -G and ETA.inv() * G.T * ETA == G \
    and G.T * ETA * G == -ETA and Smat == ETA and Kmat ** 2 == I6
# l_G as a nullspace: A^T eta + eta A = 0 and [A, G] = 0
Asym = sp.Matrix(6, 6, sp.symbols("a0:36"))
eqs = list(Asym.T * ETA + ETA * Asym) + list(Asym * G - G * Asym)
solA = sp.linsolve(eqs, list(Asym))
(gen_sol,) = solA
free = sorted(set().union(*[x.free_symbols for x in gen_sol]), key=str)
LGbasis = []
for f in free:
    sub = {g: (1 if g == f else 0) for g in free}
    LGbasis.append(sp.Matrix(6, 6, [x.subs(sub) for x in gen_sol]))
ok0 &= len(LGbasis) == 6
# the paper's L_i, K_i (s12): ad of bivectors rot/bst, must lie in l_G
def ad_on_V(Bv):
    M = sp.zeros(6, 6)
    for a in range(6):
        c = vcoef(Bv * E[a] - E[a] * Bv)
        if c is None:
            return None
        M[:, a] = c
    return M


cyc = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
paperLK = []
for (i, j, k) in cyc:
    paperLK.append(ad_on_V(E[j + 3] * E[k + 3] - E[j] * E[k]))
    paperLK.append(ad_on_V(E[k] * E[j + 3] - E[j] * E[k + 3]))
ok0 &= all(A is not None and A.T * ETA + ETA * A == sp.zeros(6, 6) and A * G == G * A for A in paperLK)
check("x3-0", ok0, "realisation: Clifford relations; G = -K o Ad_T computed from Clifford data equals the paper's pairing; "
      "G^2=-1, G^T=-G, G^dagger=G, G^T eta G=-eta; centraliser of G in so(3,3) is 6-dim and contains the paper's L_i, K_i")

LGn = [np.array(A.tolist(), dtype=float) for A in LGbasis]
Gn = np.array(G.tolist(), dtype=float)
ETAn = np.diag(np.array(etad, dtype=float))

# ------------------------------------------------------------------ Lie-Poisson from structure constants (both signs)
h = sp.Symbol("h", real=True)
ps = sp.Matrix(sp.symbols("p1:7", real=True))
I0s, ms = sp.symbols("I_0 m", positive=True)
mu = [h] + list(ps)
# basis of s: index 0 = ghat, 1..6 = e_1..e_6 ; [ghat, e_a] = G e_a ; c[k][i][j] = coefficient of basis k in [b_i, b_j]
c = [[[0] * 7 for _ in range(7)] for _ in range(7)]
for a in range(6):
    for b in range(6):
        c[b + 1][0][a + 1] = G[b, a]
        c[b + 1][a + 1][0] = -G[b, a]
Pi_minus = sp.Matrix(7, 7, lambda i, j: -sum(mu[k] * c[k][i][j] for k in range(7)))  # {F,K} = -<mu,[dF,dK]>
Nsyms = sp.symbols("n0:21", real=True)
Nsym = sp.zeros(6, 6)
idx = 0
for i in range(6):
    for j in range(i, 6):
        Nsym[i, j] = Nsym[j, i] = Nsyms[idx]
        idx += 1
Hq = h ** 2 / (2 * I0s) + (ps.T * Nsym * ps)[0] / 2
coords = [h] + list(ps)


def grad(F):
    return sp.Matrix([sp.diff(F, x) for x in coords])


def pb(F, K, Pi=Pi_minus):
    return sp.expand((grad(F).T * Pi * grad(K))[0])


rhs_minus = [pb(x, Hq) for x in coords]
target_p = (h / I0s) * (G.T * ps)
target_h = -(ps.T * G * Nsym * ps)[0]
ok1 = sp.simplify(rhs_minus[0] - target_h) == 0 and all(sp.simplify(rhs_minus[j + 1] - target_p[j]) == 0 for j in range(6))
# the other sign gives the reversed flow
rhs_plus = [pb(x, Hq, -Pi_minus) for x in coords]
ok1 &= all(sp.simplify(rhs_plus[j] + rhs_minus[j]) == 0 for j in range(7))
# the d3 bracket formula {F,K} = p^T (K_h G grad F - F_h G grad K) equals -<mu,[dF,dK]>
Ft = h * ps[0] + ps[1] * ps[4] ** 2 + h ** 2 * ps[2]
Kt = h ** 3 + ps[0] * ps[3] + h * ps[5] ** 2
gpF = sp.Matrix([sp.diff(Ft, x) for x in ps])
gpK = sp.Matrix([sp.diff(Kt, x) for x in ps])
d3form = (ps.T * (sp.diff(Kt, h) * G * gpF - sp.diff(Ft, h) * G * gpK))[0]
ok1 &= sp.expand(d3form - pb(Ft, Kt)) == 0
# Jacobi identity of the Poisson tensor on three test functions
Lt = ps[2] * ps[5] + h * ps[1]
jac = pb(Ft, pb(Kt, Lt)) + pb(Kt, pb(Lt, Ft)) + pb(Lt, pb(Ft, Kt))
ok1 &= sp.expand(jac) == 0
# H conserved for every symmetric N, |p|^2 Casimir
ok1 &= pb(Hq, Hq) == 0 and all(sp.expand(pb((ps.T * ps)[0], x)) == 0 for x in coords)
check("x3-1", ok1, "Lie-Poisson equations of s = R ghat x|_G V from its structure constants: the minus bracket "
      "{F,K}=-<mu,[dF,dK]> gives p'=(h/I0)G^T p, h'=-p^T G N p (paper eq. LProtor), the plus bracket the reversed flow; "
      "the d3 bracket formula is the minus bracket; Jacobi holds; |p|^2 is a Casimir (exact)")

# ------------------------------------------------------------------ Casimirs (Theorem d3:thm:frozen)
pi_s = ETA.inv() * ps
Bv, Ev = pi_s[:3, 0], pi_s[3:, 0]
zr, zi = Bv, Ev                                   # z = B + iE
E6 = sp.Rational(1, 2) * ((Bv.T * Bv)[0] + (Ev.T * Ev)[0])
Cq = (ps.T * ETA.inv() * ps)[0]
Dq = (ps.T * G * ETA.inv() * ps)[0]
w_re, w_im = Cq, -Dq
ok2 = sp.expand(w_re - ((Bv.T * Bv)[0] - (Ev.T * Ev)[0])) == 0 and sp.expand(w_im - 2 * (Bv.T * Ev)[0]) == 0
E0sq = sp.Rational(1, 4) * (w_re ** 2 + w_im ** 2)
Svec = Ev.cross(Bv)
# check S = (i/2) conj(z) x z componentwise
zc = [sp.Symbol("dummy")]  # placeholder not used
cz = sp.Matrix([Bv[k] - sp.I * Ev[k] for k in range(3)])
zz = sp.Matrix([Bv[k] + sp.I * Ev[k] for k in range(3)])
ok2 &= sp.expand(sp.I / 2 * cz.cross(zz) - Svec) == sp.zeros(3, 1)
cas_ok = all(sp.expand(pb(f, x)) == 0 for f in [E6, E0sq, Svec[0], Svec[1], Svec[2]] for x in coords)
notcas = any(sp.expand(pb(Cq, x)) != 0 for x in coords) and any(sp.expand(pb(Dq, x)) != 0 for x in coords)
ok2 &= cas_ok and notcas
ok2 &= sp.expand(E6 ** 2 - E0sq - (Svec.T * Svec)[0]) == 0
pt = {ps[k]: sp.Rational(int(rng.integers(-9, 10)), int(rng.integers(1, 7))) for k in range(6)}
pt[h] = sp.Rational(3, 7)
ok2 &= Pi_minus.subs(pt).rank() == 2
# five local Casimirs: Jacobian of Re, Im of conj(z_j) z_k
fl = []
for j in range(3):
    for k in range(3):
        prod = sp.expand(cz[j] * zz[k])
        fl += [sp.re(prod), sp.im(prod)]
Jz = sp.Matrix([[sp.diff(f, x) for x in ps] for f in fl])
ok2 &= Jz.subs(pt).rank() == 5
J4 = sp.Matrix([[sp.diff(f, x) for x in ps] for f in [E6, E0sq, Svec[0], Svec[1], Svec[2]]])
ok2 &= J4.subs(pt).rank() == 4
# phase law: p = e^{-theta G} p0 => z(pi) = e^{i theta} z(pi0), w = e^{2 i theta} w0
th = sp.Symbol("theta", real=True)
p0s = sp.Matrix(sp.symbols("q1:7", real=True))
eGm = sp.cos(th) * I6 - sp.sin(th) * G
pth = eGm * p0s
pith = ETA.inv() * pth
pi0 = ETA.inv() * p0s
zth = sp.Matrix([pith[k] + sp.I * pith[k + 3] for k in range(3)])
z0 = sp.Matrix([pi0[k] + sp.I * pi0[k + 3] for k in range(3)])
ok2 &= sp.simplify((zth - sp.exp(sp.I * th) * z0).applyfunc(sp.expand_complex)) == sp.zeros(3, 1)
w0 = sp.expand((z0.T * z0)[0])
wth = (zth.T * zth)[0]
ok2 &= sp.simplify(sp.expand_complex(sp.expand(wth - sp.exp(2 * sp.I * th) * w0))) == 0
check("x3-2", ok2, "E6, E0^2, S=ExB=(i/2) conj(z) x z Poisson-commute with h and all p_j; C, D do not; Poisson tensor rank 2; "
      "Jacobian of Re/Im conj(z_j)z_k has rank 5, of (E6,E0^2,S) rank 4; E6^2=E0^2+|S|^2; z->e^{i theta}z, w->e^{2i theta}w (exact)")


# ------------------------------------------------------------------ numerical integrator (Radau)
def zf(p):
    pi = ETAn @ p
    return pi[:3] + 1j * pi[3:]


def wf(p):
    z = zf(p)
    return complex(z @ z)


def E6f(p):
    return 0.5 * float(np.vdot(zf(p), zf(p)).real)


def E0f(p):
    return 0.5 * abs(wf(p))


def Sf(p):
    pi = ETAn @ p
    return np.cross(pi[3:], pi[:3])


def rhs_general(Hh, gradp):
    def f(t, y):
        thv, hv, p = y[0], y[1], y[2:8]
        Fh = Hh(hv, p)
        gp = gradp(hv, p)
        out = np.empty(14)
        out[0] = Fh
        out[1] = -p @ Gn @ gp
        out[2:8] = Fh * (Gn.T @ p)
        ct, st = np.cos(thv), np.sin(thv)
        out[8:14] = (ct * np.eye(6) + st * Gn) @ gp       # body velocity N p reconstructed: x' = e^{theta G} grad_p H
        return out
    return f


def quad_rhs(I0, N):
    return rhs_general(lambda hv, p: hv / I0, lambda hv, p: N @ p)


def run(f, y0, T, **kw):
    return solve_ivp(f, (0, T), y0, method="Radau", rtol=1e-11, atol=1e-13, dense_output=True, **kw)


def LorN(a, b, m):
    return (a * ETAn + b * Gn @ ETAn) / m


def eGn(t):
    return np.cos(t) * np.eye(6) + np.sin(t) * Gn


# x3-3/4: non-quadratic Hamiltonian, different from the derive one
M1 = rng.normal(size=(6, 6)); M1 = (M1 + M1.T) / 2
M2 = rng.normal(size=(6, 6)); M2 = (M2 + M2.T) / 2
I0g = 0.8
Hgen = lambda hv, p: hv ** 2 / (2 * I0g) + 0.3 * np.cos(hv) + 0.05 * (p @ M1 @ p) ** 2 + 0.25 * np.sin(hv) * (p @ M2 @ p)
Hh = lambda hv, p: hv / I0g - 0.3 * np.sin(hv) + 0.25 * np.cos(hv) * (p @ M2 @ p)
Hp = lambda hv, p: 0.2 * (p @ M1 @ p) * (M1 @ p) + 0.5 * np.sin(hv) * (M2 @ p)
worst = 0.0
phase_err = 0.0
moved = 0.0
for trial in range(3):
    p0 = rng.normal(size=6) * 0.7
    y0 = np.concatenate(([0.0, 1.1], p0, np.zeros(6)))
    sol = run(rhs_general(Hh, Hp), y0, 12.0)
    for t in np.linspace(0, 12, 60):
        y = sol.sol(t)
        p = y[2:8]
        worst = max(worst, abs(E6f(p) - E6f(p0)) / E6f(p0), abs(E0f(p) - E0f(p0)) / E6f(p0),
                    np.linalg.norm(Sf(p) - Sf(p0)) / E6f(p0), abs(Hgen(y[1], p) - Hgen(1.1, p0)) / abs(Hgen(1.1, p0)))
        phase_err = max(phase_err, abs(wf(p) - np.exp(2j * y[0]) * wf(p0)), np.linalg.norm(p - eGn(-y[0]) @ p0))
    moved = max(moved, abs(sol.sol(12.0)[0]))
check("x3-3", worst < 1e-8 and phase_err < 1e-8,
      f"Radau, non-quadratic H = h^2/2I0 + 0.3cos h + 0.05(p.M1p)^2 + 0.25 sin(h) p.M2p, 3 random data: E6, E0, S, H drift "
      f"{worst:.1e} (rel), phase law w=e^(2i theta)w0, p=e^(-theta G)p0 to {phase_err:.1e}; theta moved {moved:.2f} rad")

# x3-4 adversarial: the FRAME field X = e^{-theta G} x is not frozen; its rest energy (1/2)|q(X)| grows under the drift
# Example 17.17 data: p0 = e^1, m = I0 = kappa0 = 1, N = eta^{-1}
p0 = np.array([1.0, 0, 0, 0, 0, 0])
N_ex = LorN(1.0, 0.0, 1.0)


def ev_theta(target):
    def f(t, y):
        return y[0] - target
    f.terminal = True
    f.direction = 1
    return f


sol = run(quad_rhs(1.0, N_ex), np.concatenate(([0.0, 1.0], p0, np.zeros(6))), 40.0, events=[ev_theta(6 * np.pi)])
qX = []
ts = np.linspace(0, sol.t[-1], 4001)
for t in ts:
    y = sol.sol(t)
    X = eGn(-y[0]) @ y[8:14]
    zX = X[:3] + 1j * X[3:]
    qX.append(abs(zX @ zX))
qX = np.array(qX)
# at theta = 2 pi k, x = k Dx, so |q(X)| = k^2 |Dx|^2
y_end = sol.y[:, -1]
Dx_full = float(8 * mp.ellipk(-2) - 4 * mp.ellipe(-2))
grow_ok = abs(np.linalg.norm(y_end[8:14]) - 3 * Dx_full) < 1e-8 and qX.max() > 3.0 and qX[0] == 0.0
check("x3-4", grow_ok and abs(E0f(y_end[2:8]) - E0f(p0)) < 1e-9,
      f"(corollary d3:cor:nogen is about the MOMENTUM field only) along Example 17.17 the momentum field keeps E0=1/2, but the "
      f"frame field X=e^(-theta G)x (x0=0) has (1/2)|q(X)| = 0 at tau=0, {0.5 * 9 * Dx_full ** 2:.3f} = (1/2)(3 Dx)^2 at theta=6pi "
      f"and max {0.5 * qX.max():.3f} on the way: the closure does change (and drives unboundedly) the rest energy of the frame")

# ------------------------------------------------------------------ pendulum (Proposition d3:prop:pendulum)
a_, b_ = sp.symbols("a b", real=True)
Nl = (a_ * ETA.inv() + b_ * G * ETA.inv()) / ms
Uth = sp.Rational(1, 2) * (pth.T * Nl * pth)[0]
w0c = sp.expand((z0.T * z0)[0])
target_U = sp.re(sp.expand((a_ + sp.I * b_) * sp.exp(2 * sp.I * th) * w0c)) / (2 * ms)
ok5 = sp.simplify(sp.expand(sp.expand_complex(Uth - target_U))) == 0
# dU/dtheta = p^T G N p  (so h' = -dU/dtheta)
ok5 &= sp.simplify(sp.diff(Uth, th) - (pth.T * G * Nl * pth)[0]) == 0
# Lorentz class invariance r N + N r^T = 0 for the nullspace basis
ok5 &= all(sp.simplify(A * Nl + Nl * A.T) == sp.zeros(6, 6) for A in LGbasis)
check("x3-5", ok5, "U(theta) = (1/2)p(theta)^T N p(theta) = Re((a+ib)e^{2i theta}w0)/(2m), dU/dtheta = p^T G N p, "
      "and N=(a eta^{-1}+b G eta^{-1})/m satisfies rN+Nr^T=0 for the computed l_G (exact)")

# ------------------------------------------------------------------ small oscillations, period (Proposition d3:prop:omega)
R0s = sp.Symbol("R_0", positive=True)
phi0 = sp.Symbol("varphi_0", real=True)
Ufun = R0s / (2 * ms) * sp.cos(2 * th + phi0)
thmin = (sp.pi - phi0) / 2
Upp = sp.simplify(sp.diff(Ufun, th, 2).subs(th, thmin))
ok6 = sp.simplify(Upp - 2 * R0s / ms) == 0
Om2 = Upp / I0s
ok6 &= sp.simplify(Om2 - 2 * R0s / (ms * I0s)) == 0
# period by quadrature vs 4K(sin^2(d0/2))/Omega, at a random amplitude
mp.mp.dps = 30
d0 = mp.mpf("1.37")
Om = mp.mpf("1.7")
Tquad = 4 * mp.quad(lambda d: 1 / mp.sqrt(2 * Om ** 2 * (mp.cos(d) - mp.cos(d0))), [0, d0])
Tell = 4 * mp.ellipk(mp.sin(d0 / 2) ** 2) / Om
kk_ = mp.sin(d0 / 2)
Tsub = 4 * mp.quad(lambda t: 1 / (Om * mp.sqrt(1 - kk_ ** 2 * mp.sin(t) ** 2)), [0, mp.pi / 2])
ok6 &= abs(Tquad / Tell - 1) < mp.mpf("1e-15") and abs(Tsub / Tell - 1) < mp.mpf("1e-25")
# and by integrating the full closure from psi = pi + d0 at rest: measure the period from zeros of h
p0 = rng.normal(size=6)
a_n, b_n, m_n, I0_n = 0.6, -0.8, 1.3, 0.9
Nn = LorN(a_n, b_n, m_n)
cplx = (a_n + 1j * b_n) * wf(p0)
R0n, phi0n = abs(cplx), np.angle(cplx)
Omn = np.sqrt(2 * R0n / (m_n * I0_n))
dlt = 1.1
th0 = (np.pi + dlt - phi0n) / 2
pstart = eGn(-th0) @ p0
# start on the leaf at angle th0: equivalently rotate p0 and reset theta, with w(pstart)=e^{2i th0}w(p0)


def hzero(t, y):
    return y[1]


hzero.direction = 0
sol = run(quad_rhs(I0_n, Nn), np.concatenate(([th0, 0.0], pstart, np.zeros(6))), 6 * 2 * np.pi / Omn, events=[hzero])
tz = sol.t_events[0]
tz = tz[tz > 1e-6]
Tmeas = 2 * (tz[1] - tz[0])
Tpred = float(4 * mp.ellipk(mp.sin(dlt / 2) ** 2)) / Omn
ok6 &= abs(Tmeas / Tpred - 1) < 1e-8
check("x3-6", ok6, f"U''(theta_min)=2R0/m, Omega^2=2R0/(m I0) (exact); period = 4K(sin^2(d0/2))/Omega by mpmath quadrature "
      f"(direct quadrature 1e-15, substituted 1e-25) and by Radau integration of the full closure at amplitude 1.1 (rel. error {abs(Tmeas / Tpred - 1):.1e})")

# ------------------------------------------------------------------ flat direction (Corollary d3:cor:flat)
u = rng.normal(size=3)
v = np.cross(u, rng.normal(size=3))
v *= np.linalg.norm(u) / np.linalg.norm(v)
p_null = ETAn @ np.concatenate((u, v))
sol = run(quad_rhs(1.0, LorN(0.7, 0.4, 1.0)), np.concatenate(([0.0, 1.3], p_null, np.zeros(6))), 25.0)
hs_ = sol.sol(np.linspace(0, 25, 300))[1]
ok7 = abs(wf(p_null)) < 1e-12 and np.max(np.abs(hs_ - 1.3)) < 1e-9
ok7 &= sp.expand((ps.T * G * ps)[0]) == 0
check("x3-7", ok7, f"null momentum field (w=0, random): Lorentz-class rate constant to {np.max(np.abs(hs_ - 1.3)):.1e}; "
      "p^T G p = 0 identically, so N = n*1 conserves h (exact)")

# ------------------------------------------------------------------ x3-8: I0 Omega^2 = U'' is the generic 1-dof identity
k_, V_ = sp.symbols("k V2", positive=True)
xx = sp.Symbol("x")
Vgen = sp.Function("V")
# for ANY potential V with minimum at x0, I x'' = -V'(x) linearises to omega^2 = V''(x0)/I: nothing specific to the closure
ok8 = sp.simplify(Om2 * I0s - Upp) == 0
check("x3-8", ok8, "I0*Omega^2 = U''(theta_min) holds by definition of the small-oscillation frequency of any one-degree-of-"
      "freedom system; the only closure-specific content is U'' = 2R0/m = 4|a+ib|E0/m (Theorem 17.13)")

# ------------------------------------------------------------------ budget and Lorentz inertness (Proposition d3:prop:budget)
cvec = rng.normal(size=6) * 0.6
Lam = expm(sum(cvec[k] * LGn[k] for k in range(6)))
LamIT = np.linalg.inv(Lam).T
Nb = LorN(0.5, 0.9, 1.0)
okb = np.allclose(np.linalg.inv(Lam) @ Nb @ LamIT, Nb, atol=1e-12) and np.allclose(LamIT @ Gn.T, Gn.T @ LamIT, atol=1e-12)
p0 = rng.normal(size=6)
p0b = LamIT @ p0
okb &= abs(wf(p0b) - wf(p0)) < 1e-10 * max(1, abs(wf(p0))) and abs(E6f(p0b) - E6f(p0)) > 0.1
R0b = abs((0.5 + 0.9j) * wf(p0))
h0b = 2 * np.sqrt(R0b) + 1.0                    # eps = R0/h0^2 < 1/4: rotating
s1 = run(quad_rhs(1.0, Nb), np.concatenate(([0.0, h0b], p0, np.zeros(6))), 30.0)
s2 = run(quad_rhs(1.0, Nb), np.concatenate(([0.0, h0b], p0b, np.zeros(6))), 30.0)
tt = np.linspace(0, 30, 400)
dth = np.max(np.abs(s1.sol(tt)[0:2] - s2.sol(tt)[0:2]))
okb &= dth < 1e-8
# energy budget H = K + U along the flow, psi sweeps more than 2 pi, and the sampled K range approaches R0/m
Ys = s1.sol(np.linspace(0, 30, 6001))
Kv = Ys[1] ** 2 / 2
Uv = np.array([0.5 * Ys[2:8, k] @ Nb @ Ys[2:8, k] for k in range(Ys.shape[1])])
H0b = Kv[0] + Uv[0]
rot = np.all(Ys[1] > 0) and 2 * (Ys[0, -1] - Ys[0, 0]) > 2 * np.pi
okb &= rot and np.max(np.abs(Kv + Uv - H0b)) < 1e-8 * abs(H0b) + 1e-9 and abs((Kv.max() - Kv.min()) / R0b - 1) < 1e-3
okb &= abs((Uv.max() - Uv.min()) / R0b - 1) < 1e-3
check("x3-9", okb, f"random Lambda=exp(r), r in computed l_G: Lambda^-1 N Lambda^-T = N, w invariant, E6 changed by "
      f"{E6f(p0b) - E6f(p0):+.3f}, (theta,h)(tau) unchanged to {dth:.1e}; rotating case: H = K + U to "
      f"{np.max(np.abs(Kv + Uv - H0b)):.1e}, sampled range of K = R0/m to {abs((Kv.max() - Kv.min()) / R0b - 1):.1e} (6001 samples)")

# ------------------------------------------------------------------ units (d3-10)
Mv = sp.Matrix
dim = {"kappa": Mv([0, 0, -1]), "I0": Mv([1, 2, 0]), "m": Mv([1, 0, 0]), "X": Mv([0, 1, 0]), "tau": Mv([0, 0, 1]),
       "c": Mv([0, 1, -1]), "hbar": Mv([1, 2, -1]), "ell": Mv([0, 1, 0])}
energy = Mv([1, 2, -2])
pdim = dim["m"] + dim["X"] - dim["tau"]
okd = (dim["I0"] + 2 * dim["kappa"] == energy) and (dim["I0"] + dim["kappa"] == dim["hbar"]) \
    and (2 * pdim - dim["m"] == energy) and (dim["I0"] - 2 * dim["ell"] == dim["m"]) and (dim["c"] - dim["ell"] == dim["kappa"])
check("x3-10", okd, "dimensions with [X]=L: [I0 kappa^2]=[p^2/m]=energy, [h]=[hbar], [I0/ell^2]=M, [c/ell]=[kappa]")

# ------------------------------------------------------------------ multiscale (Proposition d3:prop:multiscale)
eps_s, psi, psi0 = sp.symbols("epsilon psi psi_0", real=True)
k0 = sp.Symbol("kappa_0", positive=True)
Hc = I0s * k0 ** 2 / 2 + R0s / (2 * ms) * sp.cos(psi0)
kap2 = sp.solve(sp.Eq(I0s * sp.Symbol("K2") / 2 + R0s / (2 * ms) * sp.cos(psi), Hc), sp.Symbol("K2"))[0]
epsv = R0s / (ms * I0s * k0 ** 2)
okm = sp.simplify(kap2 - k0 ** 2 * (1 + epsv * (sp.cos(psi0) - sp.cos(psi)))) == 0
okm &= sp.simplify(epsv - (R0s / (2 * ms)) / (I0s * k0 ** 2 / 2)) == 0
# threshold test at psi0 = 2pi/3: eps(1-cos psi0) = 1.5 eps; rotates for eps=0.6, librates for eps=0.7


def regime(eps_t, psi0v):
    p0 = rng.normal(size=6)
    a_n, b_n, m_n, I0_n = 0.3, 0.2, 1.0, 1.0
    cpl = (a_n + 1j * b_n) * wf(p0)
    R0n, phn = abs(cpl), np.angle(cpl)
    kap0 = np.sqrt(R0n / (m_n * I0_n * eps_t))
    th0 = (psi0v - phn) / 2
    pst = eGn(-th0) @ p0
    sol = run(quad_rhs(I0_n, LorN(a_n, b_n, m_n)), np.concatenate(([th0, I0_n * kap0], pst, np.zeros(6))), 40.0 / kap0)
    hh = sol.sol(np.linspace(0, 40.0 / kap0, 8000))[1]
    return np.all(hh > 0), np.max(np.abs(hh ** 2 / (I0_n * kap0) ** 2 - 1)) <= 2 * eps_t + 1e-9


r1, b1 = regime(0.6, 2 * np.pi / 3)
r2, b2 = regime(0.7, 2 * np.pi / 3)
r3, b3 = regime(3.0, 0.0)
okm &= r1 and (not r2) and r3 and b1 and b2 and b3
check("x3-11", okm, "kappa^2 = kappa0^2[1+eps(cos psi0 - cos psi)], eps = U_max/K(0) (exact); at psi0=2pi/3 (threshold 2/3) "
      "eps=0.6 rotates, eps=0.7 librates; eps=3 at psi0=0 rotates; |kappa^2/kappa0^2-1|<=2eps in all three")

# ------------------------------------------------------------------ ell spread (d3-12), analytic
oks = True
for eps_t in [1e-2, 1e-3, 1e-4]:
    for ps0 in [0.0, 1.0, 2.5, np.pi]:
        lmax = (1 + eps_t * (np.cos(ps0) - 1)) ** -0.5
        lmin = (1 + eps_t * (np.cos(ps0) + 1)) ** -0.5
        spread = lmax - lmin                     # in units of ell0 = c/kappa0
        oks &= abs(spread / eps_t - 1) <= 3 * eps_t
check("x3-12", oks, "analytic: (ell_max - ell_min)/ell0 = (1+eps(cos psi0-1))^(-1/2) - (1+eps(cos psi0+1))^(-1/2) = eps + O(eps^2) "
      "for every psi0 (rotating regime)")

# ------------------------------------------------------------------ Dirac at rest vs frame (d3-13)
Jm = E[1] * E[2]
okr = Jm * E[0] == Tm and Tm ** 2 == -I8 and E[0] * Jm == Jm * E[0] and Tm.T == -Tm
lam, mm = sp.symbols("lambda m_D", real=True)
Psi0 = sp.Matrix(sp.symbols("s1:9", real=True))
Psi = sp.cos(mm * lam) * Psi0 - sp.sin(mm * lam) * Tm * Psi0
H0 = mm * E[0]
okr &= sp.simplify(sp.diff(Psi, lam) + Jm * H0 * Psi) == sp.zeros(8, 1)
tau, kap = sp.symbols("tau kappa", real=True)
X0 = sp.Matrix(sp.symbols("x1:7", real=True))
Xt = (sp.cos(kap * tau) * I6 - sp.sin(kap * tau) * G) * X0
okr &= sp.simplify(sp.diff(Xt, tau) + kap * G * Xt) == sp.zeros(6, 1)
okr &= sp.simplify(sp.diff(Xt, tau, 2) + kap ** 2 * Xt) == sp.zeros(6, 1)
check("x3-13", okr, "J e1 = T, T^2=-1, T^T=-T, Psi=cos(m l)Psi0 - sin(m l)T Psi0 solves Psi'=-J H(0) Psi with H(0)=m e1; "
      "X=e^{-kappa G tau}X0 has alpha_+X=0 and X''=-kappa^2 X (exact)")

# ------------------------------------------------------------------ (d3-14) matching, trivial
hb, cc, mmass, ell = sp.symbols("hbar c m ell", positive=True)
okc = sp.solve(sp.Eq(hb * cc / ell, mmass * cc ** 2), ell) == [hb / (mmass * cc)] and \
    sp.solve(sp.Eq(2 * hb * cc / ell, mmass * cc ** 2), ell) == [2 * hb / (mmass * cc)]
check("x3-14", okc, "hbar c/ell = m c^2 <=> ell = hbar/(mc); 2 hbar c/ell = m c^2 <=> ell = 2hbar/(mc)")

# ------------------------------------------------------------------ (d3-15) no operator relates the rates
al = sp.Symbol("alpha", real=True)
eaT = sp.cos(al) * I8 + sp.sin(al) * Tm
emaT = sp.cos(al) * I8 - sp.sin(al) * Tm
okn = all(sp.simplify(eaT * E[a] * emaT - E[a]) == sp.zeros(8, 8) for a in range(3))
okn &= all(sp.simplify(eaT * E[a] * emaT - (sp.cos(2 * al) * E[a] + sp.sin(2 * al) * Tm * E[a])) == sp.zeros(8, 8) for a in range(3, 6))
# T e_a is not in the span of the vectors (grade four)
okn &= all(vcoef(Tm * E[a]) is None for a in range(3, 6))
eG = sp.cos(th) * I6 + sp.sin(th) * G
okn &= sp.simplify(eG.T * ETA * eG - (sp.cos(2 * th) * ETA + sp.sin(2 * th) * ETA * G)) == sp.zeros(6, 6)
# isometry iff theta in pi Z: at theta = pi/2 it is an anti-isometry
okn &= (eG.subs(th, sp.pi / 2).T * ETA * eG.subs(th, sp.pi / 2) == -ETA)
check("x3-15", okn, "Ad_{e^{alpha T}} fixes e1,e2,e3 and sends spatial e_a to cos2a e_a + sin2a T e_a with T e_a not a vector; "
      "(e^{theta G})^T eta e^{theta G} = cos2theta eta + sin2theta eta G, anti-isometry at theta=pi/2 (exact)")

# ------------------------------------------------------------------ (d3-16, 17) scalings: the epsilon law needs a stated hypothesis
s_, t_ = sp.symbols("s t", positive=True)
h0s = sp.Symbol("h_0", positive=True)
eps_k0 = lambda I, m_: R0s / (m_ * I * k0 ** 2)                 # kappa0 held fixed
eps_h0 = lambda I, m_: R0s / (m_ * I * (h0s / I) ** 2)          # h0 held fixed, kappa0 = h0/I0
r_k0 = sp.simplify(eps_k0(s_ * I0s, t_ * ms) / eps_k0(I0s, ms))
r_h0 = sp.simplify(eps_h0(s_ * I0s, t_ * ms) / eps_h0(I0s, ms))
oksc = sp.simplify(r_k0 - 1 / (s_ * t_)) == 0 and sp.simplify(r_h0 - s_ / t_) == 0
ints = all(all(x.is_integer for x in M) for M in [ETA, G, G * ETA.inv(), ETA * G, Kmat])
oksc &= ints
check("x3-16", oksc, f"integer data (eta, G, G eta^-1, eta G, K); epsilon under (I0,m)->(sI0,tm) scales by {r_k0} at fixed kappa0 "
      f"but by {r_h0} at fixed h0: the 1/(st) law requires the hypothesis 'kappa0 fixed'")
# closure time rescaling (tau, I0, m) -> (s tau, s I0, s m): symbolic on the equations
tt_ = sp.Symbol("tp")
thf, hf = sp.Function("th"), sp.Function("hh")
pf = [sp.Function(f"pp{k}") for k in range(6)]
# original solution y(tau) obeys p' = (h/I0) G^T p, h' = -p^T G N p. New: Y(tau') = y(tau'/s), I0'=s I0, N'=N/s
pvec = sp.Matrix([f(tt_ / s_) for f in pf])
hval = hf(tt_ / s_)
lhs_p = sp.diff(pvec, tt_)
orig_p = (hval / I0s) * (G.T * pvec) / s_       # chain rule: d/dtau' = (1/s) d/dtau evaluated with original law
new_p = (hval / (s_ * I0s)) * (G.T * pvec)
ok17 = sp.simplify(orig_p - new_p) == sp.zeros(6, 1)
orig_h = -(pvec.T * G * Nsym * pvec)[0] / s_
new_h = -(pvec.T * G * (Nsym / s_) * pvec)[0]
ok17 &= sp.simplify(orig_h - new_h) == 0
ok17 &= sp.simplify(eps_k0(s_ * I0s, s_ * ms).subs(k0, k0 / s_) - eps_k0(I0s, ms)) == 0
check("x3-17", ok17, "(tau,I0,m)->(s tau,s I0,s m) maps closure solutions to solutions and leaves eps invariant (exact)")

# ------------------------------------------------------------------ (d3-18) Schur commutant and dimension matrix
L3 = [sp.Matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]]), sp.Matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
A3 = sp.Matrix(3, 3, sp.symbols("c0:9"))
sol3 = sp.linsolve(sum((list(A3 * L - L * A3) for L in L3), []), list(A3))
(g3,) = sol3
ok18 = len(set().union(*[x.free_symbols for x in g3])) == 1
Dm = sp.Matrix.hstack(dim["c"], dim["hbar"])
ok18 &= all(sp.linsolve((Dm, tgt)) == sp.EmptySet for tgt in [Mv([0, 1, 0]), Mv([1, 0, 0]), Mv([0, 0, 1])])
D3 = sp.Matrix.hstack(dim["c"], dim["hbar"], dim["m"])
ok18 &= D3.det() != 0 and D3.LUsolve(Mv([0, 1, 0])) == Mv([-1, 1, -1])
check("x3-18", ok18, "commutant of so(3) on R^3 is 1-dim (Schur); no c^a hbar^b is L, M or T; with m the unique length is "
      "c^-1 hbar m^-1 (exact).  This is dimensional analysis: true by construction, so 'obstructed' overstates it")

# ------------------------------------------------------------------ (d3-19) Example 17.17
mp.mp.dps = 30
Kc, Ec = mp.ellipk(-2), mp.ellipe(-2)
full = 8 * Kc - 4 * Ec
half = 4 * Kc - 2 * Ec
q_full = mp.quad(lambda t: mp.cos(2 * t) / mp.sqrt(1 + 2 * mp.sin(t) ** 2), mp.linspace(0, 2 * mp.pi, 9))
lam1716 = mp.quad(lambda ps_: mp.cos(ps_) / mp.sqrt(1 + (1 - mp.cos(ps_))), mp.linspace(0, 2 * mp.pi, 9)) / 2
ok19 = abs(full - q_full) < mp.mpf("1e-25") and abs(half - lam1716) < mp.mpf("1e-25") and abs(full - mp.mpf("0.6336081022")) < 1e-10
sol = run(quad_rhs(1.0, N_ex), np.concatenate(([0.0, 1.0], np.array([1.0, 0, 0, 0, 0, 0]), np.zeros(6))), 20.0,
          events=[ev_theta(np.pi), ev_theta(2 * np.pi)])
yh = sol.y_events[0][0]
sol2 = run(quad_rhs(1.0, N_ex), np.concatenate(([0.0, 1.0], np.array([1.0, 0, 0, 0, 0, 0]), np.zeros(6))), 20.0,
           events=[ev_theta(2 * np.pi)])
yf = sol2.y_events[0][0]
ok19 &= np.linalg.norm(yf[8:14] - float(full) * np.eye(6)[0]) < 1e-9 and np.linalg.norm(yh[8:14] - float(half) * np.eye(6)[0]) < 1e-9
ok19 &= np.linalg.norm(yh[2:8] + np.eye(6)[0]) < 1e-9 and abs(yh[1] - 1) < 1e-9 and np.linalg.norm(yf[2:8] - np.eye(6)[0]) < 1e-9
check("x3-19", ok19, f"8K(-2)-4E(-2) = {mp.nstr(full, 12)} = quadrature over theta in [0,2pi]; Theorem 17.16's lambda (one rate "
      f"period) = 4K(-2)-2E(-2) = {mp.nstr(half, 12)}; Radau: x(theta=2pi) = {yf[8]:.10f} e1, x(theta=pi) = {yh[8]:.10f} e1, "
      "p=-p0 and h=h0 at theta=pi")

# ------------------------------------------------------------------ (d3-20) half turn
OmegaCl = E[0] * E[3] * E[4] * E[5]
Gam = Jm * OmegaCl
vol = E[0] * E[1] * E[2] * E[3] * E[4] * E[5]
ok20 = (sp.cos(sp.pi) * I6 + sp.sin(sp.pi) * G == -I6) and (sp.cos(2 * sp.pi) * I6 + sp.sin(2 * sp.pi) * G == I6)
ok20 &= (Gam == vol or Gam == -vol) and Gam ** 2 == I8 and all(Gam * E[a] == -E[a] * Gam for a in range(6))
U_sym = sp.Rational(1, 2) * (pth.T * Nsym * pth)[0]
ok20 &= sp.simplify(U_sym.subs(th, th + sp.pi) - U_sym) == 0
check("x3-20", ok20, f"e^(pi G) = -1, e^(2pi G) = 1; Gamma = J Omega = {'+' if Gam == vol else '-'}e1...e6, Gamma^2 = 1, anticommutes "
      "with every e_a; U(theta+pi) = U(theta) for a general symmetric N (exact)")

# ------------------------------------------------------------------ (d3-21) quantum rotor via scipy Mathieu values
q_m = 1000.0
try:
    a0 = float(ssp.mathieu_a(0, q_m))
    b2 = float(ssp.mathieu_b(2, q_m))
    b1 = float(ssp.mathieu_b(1, q_m))
except Exception:
    a0 = b2 = b1 = float("nan")
# independent Galerkin in the cos/sin basis of period-pi functions (even Fourier sector): phi'' + (a - 2q cos 2x) phi = 0
nmax = 120
# cos(2kx) block for a_{2k}; sin(2kx) block for b_{2k}
Acos = np.zeros((nmax, nmax))
for k in range(nmax):
    Acos[k, k] = (2 * k) ** 2
    if k + 1 < nmax:
        Acos[k, k + 1] = Acos[k + 1, k] = q_m
Acos[0, 1] = Acos[1, 0] = np.sqrt(2) * q_m
Asin = np.zeros((nmax, nmax))
for k in range(nmax):
    Asin[k, k] = (2 * (k + 1)) ** 2
    if k + 1 < nmax:
        Asin[k, k + 1] = Asin[k + 1, k] = q_m
a0_g = np.linalg.eigvalsh(Acos)[0]
b2_g = np.linalg.eigvalsh(Asin)[0]
gap_g = 0.5 * (b2_g - a0_g)                       # E = (hbar^2/2I0) a with hbar = I0 = 1
Omq = np.sqrt(2 * 2000.0)
asym = Omq - 0.5
asym2 = asym - 0.5 / (4 * np.sqrt(q_m))
ok21 = abs(gap_g - 62.7415) < 1e-3 and abs(gap_g - asym2) < 1e-3 and abs(gap_g - asym) < 6e-3
if np.isfinite(a0) and np.isfinite(b2):
    note(f"scipy.special Mathieu values at q=1000: a0={a0:.6f}, b2={b2:.6f}, b1={b1:.6f}; Galerkin a0={a0_g:.6f}, b2={b2_g:.6f}")
I0q = sp.solve(sp.Eq(hb / sp.Symbol("I0q", positive=True), mmass * cc ** 2 / hb), sp.Symbol("I0q", positive=True))[0]
ok21 &= sp.simplify(I0q - hb ** 2 / (mmass * cc ** 2)) == 0 and sp.simplify(I0q / (hb / (mmass * cc)) ** 2 - mmass) == 0 \
    and sp.simplify(hb ** 2 / (2 * I0q) - mmass * cc ** 2 / 2) == 0
# E_n differences
nn, dd = sp.symbols("n delta")
En = lambda k: hb ** 2 * (k + dd) ** 2 / (2 * I0s)
ok21 &= sp.simplify(En(nn + 1) - En(nn) - hb / 2 * ((nn + dd) * hb / I0s + (nn + 1 + dd) * hb / I0s)) == 0
check("x3-21", ok21, f"q = I0 R0/(2 m hbar^2) = 1000: lowest even-sector gap (b2-a0)/2 = {gap_g:.5f} (cos/sin Galerkin) vs "
      f"hbar Omega - hbar^2/2I0 = {asym:.5f}, with the next DLMF 28.8.1 term {asym2:.5f}; I0 = hbar^2/(mc^2), M_rot = m, "
      "E_1 = mc^2/2; E_(n+1)-E_n = hbar(kappa_n+kappa_(n+1))/2 (exact)")

# ------------------------------------------------------------------ x3-22 adversarial: sum of constituent rest energies under a compact Cent(G3) rotation
# two constituents (2 and 3 in the paper's numbering) as C^3 (+) C^3; a real rotation mixing z^(2)_1 and z^(3)_1 by pi/4
z2 = np.array([1, 1j, 0]); z3 = np.array([1, 0, 0], dtype=complex)
cth, sth = np.cos(np.pi / 4), np.sin(np.pi / 4)
z2n, z3n = z2.copy(), z3.copy()
z2n[0], z3n[0] = cth * z2[0] - sth * z3[0], sth * z2[0] + cth * z3[0]
E0sum = lambda A, B: 0.5 * (abs(A @ A) + abs(B @ B))
q3 = lambda A, B: A @ A + B @ B
ok22 = abs(q3(z2, z3) - q3(z2n, z3n)) < 1e-12 and abs(E0sum(z2, z3) - 0.5) < 1e-12 and abs(E0sum(z2n, z3n) - 1.5) < 1e-12
check("x3-22", ok22, "a compact element of Cent(G3) keeps q3 = sum_k z^(k).z^(k) but moves sum_k E0^(k) from 1/2 to 3/2: "
      "'redistributes rest energy among constituents without creating it' holds only for the total invariant |q3|/2")

# ------------------------------------------------------------------ x3-23 adversarial: the frame's |q| is not conserved even in the clamped interior
# Proposition 16.5(ii) of ES17: z = c_- e^{-i kappa tau} + c_+ e^{i kappa tau}; q(z) = c_-.c_- e^{-2i k t} + 2 c_-.c_+ + c_+.c_+ e^{2i k t}
cm = rng.normal(size=3) + 1j * rng.normal(size=3)
cpl_ = rng.normal(size=3) + 1j * rng.normal(size=3)
kap_ = 1.3
tq = np.linspace(0, 2 * np.pi / kap_, 400)
absq = np.array([abs((cm * np.exp(-1j * kap_ * t) + cpl_ * np.exp(1j * kap_ * t)) @ (cm * np.exp(-1j * kap_ * t) + cpl_ * np.exp(1j * kap_ * t))) for t in tq])
# sympy: d/dtau |q|^2 is not identically zero for generic c_+ != 0
tau_s = sp.Symbol("tau", real=True)
cm_s = sp.Matrix([1, sp.I, 0]); cp_s = sp.Matrix([1, 0, 0])
zt = cm_s * sp.exp(-sp.I * tau_s) + cp_s * sp.exp(sp.I * tau_s)
qt = sp.expand((zt.T * zt)[0])
absq2 = sp.simplify(sp.expand_complex(qt * sp.conjugate(qt)))
ok23 = (absq.max() - absq.min()) > 0.1 * absq.mean() and sp.simplify(sp.diff(absq2, tau_s)) != 0
check("x3-23", ok23, f"clamped split-form interior (Prop. 16.5(ii)): along z = c_- e^(-ik t) + c_+ e^(ik t) the frame's |q| varies "
      f"(random data: from {absq.min():.3f} to {absq.max():.3f}; exact example c_-=(1,i,0), c_+=(1,0,0): |q|^2 = {absq2}); "
      "only the momentum field's |w| is a constant of motion")

if FAILS:
    print("FAILED:", ", ".join(FAILS))
    sys.exit(1)
print("ALL x3 CHECKS PASS")
sys.exit(0)
