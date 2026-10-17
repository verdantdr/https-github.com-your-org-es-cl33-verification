#!/usr/bin/env python3
"""Independent adversarial check of track d4 (the Yang--Mills b-field as the antisymmetric part of the shift).

Rebuilt from the paper sources (tex/s02_arena.tex eq. realisation, s03 prop:twoK / lem:Gclifford,
s12 thm:emfield, s16, s17) without importing the derive script d4_bfield.py.

Methods differ from the derive script where possible:
  * vector actions of bivectors are extracted from the 8x8 (and 512x512) realisation by traces;
  * exact semisimplicity uses the square-free part of the characteristic polynomial over QQ
    (DomainMatrix), characteristic polynomials of block-triangular M are taken blockwise;
  * random rational samples replace special cases; the bounded-data subspace is computed as an exact
    kernel of a real polynomial in M instead of from complex eigenvectors;
  * Krein signatures are computed from the canonical Hessian on numerically computed kernels;
  * the dimension of the bounded set is checked by a finite-difference Jacobian at an exact point.

Besides re-verifying the d4 claims, it records three findings of the verifier:
  x4-N1  compact (gauge-admissible) connections can be screened as well: b = diag(A+C, A-C) with
         a.c = 0 is bounded although [b,G] != 0 (no boost needed);
  x4-N2  at three constituents b = R(7,13) + R(10,11) in ps_W, not in so(4)_Delta, [b,G3] != 0, is
         bounded: 'only the locked diagonal is compatible with binding' is false;
  x4-N3  for b in l_G the boundedness criterion 'w = 0 or w.w > 0' is exactly boundedness of the gauge
         factor exp(-b tau); the co-moving dynamics is the b = 0 binding for every b in l_G.
Prints PASS/FAIL lines; exit 0 iff every check passes.
"""
import sys, time, itertools
from fractions import Fraction
import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import scipy.sparse as sps
import mpmath as mp

T0 = time.time()
FAILS = []


def report(cid, ok, text):
    if ok:
        print(f"PASS {cid}: {text}")
    else:
        print(f"FAIL {cid}: {text}")
        FAILS.append(cid)


QQ = sp.QQ
lam = sp.symbols('lam')


def DM(M):
    return DomainMatrix.from_Matrix(sp.Matrix(M)).convert_to(QQ)


def charpoly_QQ(M):
    """characteristic polynomial (as sympy Poly in lam) of a rational matrix"""
    d = DM(M)
    coeffs = d.charpoly()
    return sp.Poly([sp.Rational(c) for c in coeffs], lam)


def is_semisimple_poly(Mdm, p):
    """M (DomainMatrix over QQ) semisimple iff square-free part of p annihilates M"""
    g = sp.gcd(p, p.diff(lam))
    sf = sp.quo(p, g)
    n = Mdm.shape[0]
    R = DomainMatrix.zeros((n, n), QQ)
    I = DomainMatrix.eye(n, QQ)
    for c in sf.all_coeffs():
        R = R * Mdm + I * QQ(sp.Rational(c).p, sp.Rational(c).q)
    return R.is_zero_matrix


def M_of(b, kap, G, m=1):
    n = b.shape[0]
    S = b + kap * G
    D = -b + kap * G
    return S, D, sp.BlockMatrix([[-S, sp.eye(n) / m], [sp.zeros(n), D]]).as_explicit()


def exact_bounded(b, kap, G, m=1):
    """returns (semisimple, imaginary_spectrum) exactly; bounded iff both"""
    S, D, M = M_of(sp.Matrix(b), kap, sp.Matrix(G), m)
    pS = charpoly_QQ(S)
    pM = charpoly_QQ(-S) * charpoly_QQ(D)
    ss = is_semisimple_poly(DM(M), pM)
    return ss, imag_spectrum(pS)


def imag_spectrum(pS):
    """exact: every root of the real polynomial pS lies on the imaginary axis.
    pS must be even; then roots lam = +-i w with x = w^2 roots of q(x) = pS(sqrt(-x)); all must be real and >= 0.
    count_roots counts distinct roots, so it is applied to the square-free part of q."""
    coeffs = pS.all_coeffs()
    deg = len(coeffs) - 1
    even = all(coeffs[i] == 0 for i in range(len(coeffs)) if (deg - i) % 2 == 1)
    if not even:
        return False
    x = sp.symbols('x')
    q = sp.Poly(sum(c * (-x) ** ((deg - i) // 2) for i, c in enumerate(coeffs)), x)
    qs = sp.Poly(sp.quo(q, sp.gcd(q, q.diff(x))), x)
    return qs.count_roots() == qs.degree() and qs.count_roots(0, sp.oo) == qs.degree()


# ---------------------------------------------------------------------------------------------
# 0. the realisation (s02 eq. realisation) and the vector actions
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)
k3 = lambda a, b, c: np.kron(np.kron(a, b), c)
E8 = [k3(sx, I2, I2), k3(sz, sx, I2), k3(sz, sz, sx), k3(ep, I2, I2), k3(sz, ep, I2), k3(sz, sz, ep)]
etad = [1, 1, 1, -1, -1, -1]
eta_np = np.diag(etad)
eta = sp.diag(*etad)
I8 = np.eye(8, dtype=np.int64)


def vec_action(B):
    """matrix of v -> (1/2)[B, v] on V = span(e_a), via traces"""
    M = sp.zeros(6, 6)
    for c in range(6):
        X = B @ E8[c] - E8[c] @ B
        for d in range(6):
            M[d, c] = sp.Rational(int(np.trace(X @ E8[d])), 2 * 8 * etad[d])
    return M


ok = all((E8[a] @ E8[b] + E8[b] @ E8[a] == 2 * eta_np[a, b] * I8).all() for a in range(6) for b in range(6))


def Rab(a, b, n=6, et=None):  # 1-based
    et = etad if et is None else et
    M = sp.zeros(n, n)
    M[a - 1, b - 1] = et[b - 1]
    M[b - 1, a - 1] = -et[a - 1]
    return M


ok_R = all(vec_action(E8[a - 1] @ E8[b - 1]) == Rab(a, b) for a in range(1, 7) for b in range(a + 1, 7))
Kgr = vec_action(sum(E8[i] @ E8[i + 3] for i in range(3)))
G = sp.zeros(6, 6)
for i in range(3):
    G[i + 3, i] = 1
    G[i, i + 3] = -1
ok_K = all(Kgr[:, i] == -sp.eye(6)[:, i + 3] and Kgr[:, i + 3] == -sp.eye(6)[:, i] for i in range(3))
ok_G = (G == -Kgr * eta) and (G * G == -sp.eye(6)) and (eta.inv() * G.T * eta == G) and (G.T * eta * G == -eta)
# L_i = (1/2)ad(e_{j+3}e_{k+3} - e_j e_k), K_i = ad((1/2)(e_k e_{j+3} - e_j e_{k+3})) for cyclic (i,j,k)
cyc = {1: (2, 3), 2: (3, 1), 3: (1, 2)}
L, K = {}, {}
for i, (j, k) in cyc.items():
    L[i] = vec_action(E8[j + 2] @ E8[k + 2] - E8[j - 1] @ E8[k - 1])
    K[i] = vec_action(E8[k - 1] @ E8[j + 2] - E8[j - 1] @ E8[k + 2])


def so3(v):
    x, y, z = v
    return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])


Z3 = sp.zeros(3, 3)
ok_z = True
for i in range(1, 4):
    Ji = so3([1 if t == i - 1 else 0 for t in range(3)])
    ok_z &= (L[i] == sp.diag(Ji, Ji))  # z -> e_i x z
    ok_z &= (K[i] == sp.BlockMatrix([[Z3, -Ji], [Ji, Z3]]).as_explicit())  # z -> i e_i x z
    ok_z &= (K[i] == G * L[i])
br = lambda A, B: A * B - B * A
eps = lambda i, j, k: sp.LeviCivita(i, j, k)
ok_br = all(br(L[i], L[j]) == sum((eps(i, j, k) * L[k] for k in range(1, 4)), sp.zeros(6))
            and br(L[i], K[j]) == sum((eps(i, j, k) * K[k] for k in range(1, 4)), sp.zeros(6))
            and br(K[i], K[j]) == -sum((eps(i, j, k) * L[k] for k in range(1, 4)), sp.zeros(6))
            for i in range(1, 4) for j in range(1, 4))
ok_misc = (Rab(1, 2) - Rab(4, 5) == -L[3]) and ((Rab(1, 2) + Rab(4, 5)) * G == -G * (Rab(1, 2) + Rab(4, 5)))
# paper example 12.4: [K_3, e_1] = e_5
ok_misc &= (K[3][:, 0] == sp.eye(6)[:, 4])
report("d4-0", ok and ok_R and ok_K and ok_G and ok_z and ok_br and ok_misc,
       "8x8 realisation re-built: Cl(3,3); (1/2)ad(e_a e_b) = R(a,b) by traces; K = (1/2)ad(sum e_i e_{i+3}) "
       "with K e_i = -e_{i+3}; G = -K eta, G^2 = -1, G^dag = G, anti-isometry; L_i acts as e_i x, K_i as i e_i x, "
       "K_i = G L_i, Lorentz brackets of Thm 12.3; R12 - R45 = -L_3, R12 + R45 anticommutes with G")

# ---------------------------------------------------------------------------------------------
# 1. EL equations by direct differentiation of the scalar Lagrangian
bs = sp.symbols('b1:16')
so_basis = [Rab(a, b) for a in range(1, 7) for b in range(a + 1, 7)]
bsym = sum((c * B for c, B in zip(bs, so_basis)), sp.zeros(6))
kap, m = sp.symbols('kappa m', positive=True)
Xs = sp.Matrix(sp.symbols('x1:7'))
Vs = sp.Matrix(sp.symbols('v1:7'))
As = sp.Matrix(sp.symbols('a1:7'))
S = bsym + kap * G
u = Vs + S * Xs
Lag = (m / 2 * (u.T * eta * u))[0]
dLdV = sp.Matrix([sp.diff(Lag, v) for v in Vs])
dLdX = sp.Matrix([sp.diff(Lag, x) for x in Xs])
# total derivative of dL/dV along (X, V, A): d/dtau = V.d/dX + A.d/dV
ddt = sp.Matrix([sum(sp.diff(f, x) * v for x, v in zip(Xs, Vs)) + sum(sp.diff(f, v) * a for v, a in zip(Vs, As)) for f in dLdV])
EL = ddt - dLdX
Dop = lambda Y, Ydot: Ydot + bsym * Y  # D acting on (Y, Ydot)
# (D - kG)(D + kG)X = A + 2bV + b^2 X + k[b,G]X + k^2 X
target = m * eta * (As + 2 * bsym * Vs + bsym * bsym * Xs + kap * br(bsym, G) * Xs + kap ** 2 * Xs)
ok1 = sp.expand(EL - target) == sp.zeros(6, 1)
# decomposition End(V) = so(3,3) + Sym_eta: rank of 15 + 21 basis matrices
dag = lambda A: eta.inv() * A.T * eta
sym_basis = []
for a in range(6):
    for b in range(a, 6):
        Ms = sp.zeros(6, 6)
        Ms[a, b] = 1
        Ms[b, a] = 1
        sym_basis.append(eta.inv() * Ms)  # eta^{-1} times symmetric = eta-self-adjoint
ok1 &= all(dag(A) == A for A in sym_basis) and all(dag(A) == -A for A in so_basis)
rk = sp.Matrix([list(A) for A in so_basis + sym_basis]).rank()
ok1 &= (rk == 36) and dag(G) == G and dag(Kgr) == -Kgr
# the eta-antisymmetric part of S is b
ok1 &= sp.expand((S - dag(S)) / 2 - bsym) == sp.zeros(6) and sp.expand((S + dag(S)) / 2 - kap * G) == sp.zeros(6)
report("d4-1", ok1, "End(V) = so(3,3) (+) Sym_eta (rank 36); EL of (m/2)eta(Xdot+SX,Xdot+SX) obtained by differentiating "
       "the scalar Lagrangian equals m eta (D - kG)(D + kG)X, D = d/dtau + b, for symbolic b (15 coords)")

# ---------------------------------------------------------------------------------------------
# 2. gauge covariance, numerically along a matrix-exponential curve g = g0 exp(tau A)
rng = np.random.default_rng(20261017)
eta_f = np.array(eta_np, dtype=float)
G_f = np.array(G, dtype=float)
sob_f = [np.array(B, dtype=float) for B in so_basis]


def rand_so(scale=1.0):
    return sum(rng.normal() * scale * B for B in sob_f)


dag_f = lambda A: np.linalg.inv(eta_f) @ A.T @ eta_f
ok2 = True
for trial in range(5):
    b = rand_so()
    A = rand_so(0.7)
    g0 = expm(rand_so(0.5))
    kk = rng.uniform(0.3, 2.0)
    for tau in [0.3, 1.1]:
        g = g0 @ expm(tau * A)
        gd = g @ A
        gi = np.linalg.inv(g)
        Sg = gi @ (b + kk * G_f) @ g + gi @ gd
        so_part = (Sg - dag_f(Sg)) / 2
        sy_part = (Sg + dag_f(Sg)) / 2
        ok2 &= np.allclose(so_part, gi @ b @ g + gi @ gd, atol=1e-9)
        ok2 &= np.allclose(sy_part, kk * gi @ G_f @ g, atol=1e-9)
        ok2 &= np.allclose(g.T @ eta_f @ g, eta_f, atol=1e-9)
        # covariant derivative of G' = g^-1 G g: dG'/dtau + [b^g, G'] = g^-1 [b,G] g ; dG'/dtau by central differences
        h = 1e-5
        Gp = lambda t: np.linalg.inv(g0 @ expm(t * A)) @ G_f @ (g0 @ expm(t * A))
        dGp = (Gp(tau + h) - Gp(tau - h)) / (2 * h)
        bg = gi @ b @ g + gi @ gd
        lhs = dGp + bg @ Gp(tau) - Gp(tau) @ bg
        ok2 &= np.allclose(lhs, gi @ (b @ G_f - G_f @ b) @ g, atol=1e-6)
report("d4-2", ok2, "gauge covariance (numeric, 5 random b, g = g0 exp(tau A) in SO(3,3)): so(3,3) part of S^g is "
       "g^-1 b g + g^-1 gdot (connection), Sym part is k g^-1 G g (tensor), D_tau G' = g^-1 [b,G] g (tol 1e-6)")

# ---------------------------------------------------------------------------------------------
# 3. l_G (+) p_G; Killing form from structure constants
def coords_so(A):
    # coordinates of A in so_basis (R(a,b) has entry eta_bb at (a,b))
    return [A[a - 1, b - 1] * etad[b - 1] for a in range(1, 7) for b in range(a + 1, 7)]


assert all(coords_so(B) == [1 if t == s else 0 for t in range(15)] for s, B in enumerate(so_basis))
adm = []
for X in so_basis:
    adm.append(sp.Matrix([coords_so(br(X, Y)) for Y in so_basis]).T)  # 15x15 ad(X)
Kill = sp.Matrix(15, 15, lambda i, j: (adm[i] * adm[j]).trace())
Tr = sp.Matrix(15, 15, lambda i, j: (so_basis[i] * so_basis[j]).trace())
ok3 = (Kill == 4 * Tr)
# l_G and p_G as kernels
Wl = sp.Matrix([list(br(B, G)) for B in so_basis]).T  # 36 x 15
lG = Wl.nullspace()
Wp = sp.Matrix([list(B * G + G * B) for B in so_basis]).T
pG = Wp.nullspace()
ok3 &= (len(lG) == 6 and len(pG) == 9)


def inertia(Gram):
    ev = np.linalg.eigvalsh(np.array(Gram, dtype=float))
    return int((ev > 1e-9).sum()), int((ev < -1e-9).sum())


Pl = sp.Matrix.hstack(*lG)
Pp = sp.Matrix.hstack(*pG)
sig_l = inertia(Pl.T * Kill * Pl)
sig_p = inertia(Pp.T * Kill * Pp)
ok3 &= (sig_l == (3, 3) and sig_p == (6, 3))
# compact (matrix-antisymmetric) part of p_G has dim 3, boost part (matrix-symmetric) dim 6
pmats = [sum((c * B for c, B in zip(v, so_basis)), sp.zeros(6)) for v in pG]
Wc = sp.Matrix([list(A + A.T) for A in pmats]).T
Ws = sp.Matrix([list(A - A.T) for A in pmats]).T
ok3 &= (len(Wc.nullspace()) == 3 and len(Ws.nullspace()) == 6)
sig1 = Rab(1, 4)  # mixed bivector: a pairing-plane boost (up to sign)
ok3 &= (Kgr * G == -G * Kgr) and (sig1 * G == -G * sig1)
report("d4-3", ok3, "Killing form from structure constants = 4 Tr; l_G = ker[.,G] dim 6, p_G = ker{.,G} dim 9; "
       "Killing signature (3+,3-) on l_G and (6+,3-) on p_G; p_G = 3 compact + 6 boosts; K and pairing boosts in p_G")

# ---------------------------------------------------------------------------------------------
# 4. phase space: canonical Hessian, Hamiltonian flow, char poly identity (random rational b)
ok4 = True
Jc = np.block([[np.zeros((6, 6)), np.eye(6)], [-np.eye(6), np.zeros((6, 6))]])
for trial in range(4):
    b = rand_so()
    kk = rng.uniform(0.2, 2)
    mm = rng.uniform(0.5, 2)
    Sf = b + kk * G_f
    Hess = np.block([[np.zeros((6, 6)), -Sf.T], [-Sf, np.linalg.inv(eta_f) / mm]])
    Mcan = Jc @ Hess
    # canonical flow: Xdot = eta^-1 p/m - S X, pdot = S^T p
    ok4 &= np.allclose(Mcan, np.block([[-Sf, np.linalg.inv(eta_f) / mm], [np.zeros((6, 6)), Sf.T]]))
    ok4 &= np.allclose(Mcan.T @ Jc + Jc @ Mcan, 0)  # Hamiltonian matrix
    # energy equals (m/2)eta(Xdot,Xdot) - (m/2) eta(SX,SX)
    X0 = rng.normal(size=6)
    p0 = rng.normal(size=6)
    Hval = p0 @ np.linalg.inv(eta_f) @ p0 / (2 * mm) - p0 @ Sf @ X0
    Xd = np.linalg.inv(eta_f) @ p0 / mm - Sf @ X0
    ok4 &= abs(Hval - (mm / 2 * Xd @ eta_f @ Xd - mm / 2 * (Sf @ X0) @ eta_f @ (Sf @ X0))) < 1e-10
for trial in range(3):
    b = sum((sp.Rational(int(rng.integers(-5, 6)), int(rng.integers(1, 4))) * B for B in so_basis), sp.zeros(6))
    S_, D_, M_ = M_of(b, sp.Rational(3, 2), G, sp.Rational(2, 3))
    pM = charpoly_QQ(M_)
    pS = charpoly_QQ(S_)
    ok4 &= (pM.as_expr() - sp.expand(pS.as_expr() * pS.as_expr().subs(lam, -lam))) == 0
report("d4-4", ok4, "canonical Hessian of H = p eta^-1 p/2m - p S X gives M (numeric, 4 random b), H = (m/2)eta(Xdot,Xdot) "
       "- (m/2)eta(SX,SX); det(lam - M) = chi_S(lam) chi_S(-lam) exact at 3 random rational b")

# ---------------------------------------------------------------------------------------------
# 5/6/7. connections commuting with G, parametrised by w in C^3; random rational tests
def b_of_w(wr, wb):
    return sum((wr[i] * L[i + 1] + wb[i] * K[i + 1] for i in range(3)), sp.zeros(6))


ok5 = True
ws = sp.symbols('w1:7', real=True)
bl_sym = b_of_w(ws[:3], ws[3:])
Ysyl = -G / (2 * kap * m)
A_ = -(bl_sym + kap * G)
D_ = -bl_sym + kap * G
ok5 &= sp.simplify(A_ * Ysyl - Ysyl * D_ + sp.eye(6) / m) == sp.zeros(6)
# rotation beta(R12 - R45): spectrum
beta = sp.Rational(3, 7)
S_, Dd, M_ = M_of(beta * (Rab(1, 2) - Rab(4, 5)), 1, G)
pM = sp.factor(charpoly_QQ(M_).as_expr())
ok5 &= sp.expand(charpoly_QQ(M_).as_expr() - ((lam ** 2 + 1) * (lam ** 2 + (1 + beta) ** 2) * (lam ** 2 + (1 - beta) ** 2)) ** 2) == 0
ok5 &= all(exact_bounded(bb * (Rab(1, 2) - Rab(4, 5)), 1, G) == (True, True) for bb in [sp.Rational(3, 7), sp.Rational(3, 2), 2])
report("d4-5", ok5, "Y = -G/(2 k m) solves A Y - Y D = -1/m for symbolic b in l_G; b = beta(R12-R45): chi_M = "
       "[(l^2+k^2)(l^2+(k+beta)^2)(l^2+(k-beta)^2)]^2 and bounded (exact, beta = 3/7, 3/2, 2 incl. k - beta = -k)")


def rr(lo=-4, hi=5, den=4):
    return sp.Rational(int(rng.integers(lo, hi)), int(rng.integers(1, den + 1)))


ok6 = True
kq = sp.Rational(1)
n_ell = n_hyp = n_null = 0
for trial in range(6):
    # elliptic: w_r perp w_b, |w_r| > |w_b|
    wr = [rr() for _ in range(3)]
    if all(x == 0 for x in wr):
        wr[0] = 1
    uu = [rr() for _ in range(3)]
    dot = sum(a * c for a, c in zip(uu, wr))
    nr = sum(a * a for a in wr)
    wb = [(uu[i] - dot / nr * wr[i]) for i in range(3)]
    nb = sum(a * a for a in wb)
    t = sp.Rational(1, 2) if nb == 0 else sp.sqrt(nr / nb) / 2
    t = sp.nsimplify(sp.Rational(int(sp.floor(t * 100)), 100)) if t.is_irrational else t
    wb = [t * x for x in wb]
    qq = sum(a * a for a in wr) - sum(a * a for a in wb)
    b = b_of_w(wr, wb)
    S_ = b + kq * G
    # chi_S formula
    wq = qq  # w.w real here since w_r.w_b = 0
    formula = sp.expand((lam ** 2 + 1) * ((lam - sp.I) ** 2 + wq) * ((lam + sp.I) ** 2 + wq))
    ok6 &= sp.expand(charpoly_QQ(S_).as_expr() - formula) == 0
    ok6 &= (qq > 0) and exact_bounded(b, kq, G) == (True, True)
    n_ell += 1
    # hyperbolic: swap roles (|w_b| > |w_r|)
    b2 = b_of_w(wb, wr)
    ss2, im2 = exact_bounded(b2, kq, G)
    ok6 &= not (ss2 and im2)
    n_hyp += 1
# generic complex w (w_r . w_b != 0)
for trial in range(3):
    wr = [rr() for _ in range(3)]
    wb = [rr() for _ in range(3)]
    if sum(a * c for a, c in zip(wr, wb)) == 0:
        wb[0] += 1
    ss2, im2 = exact_bounded(b_of_w(wr, wb), kq, G)
    ok6 &= not im2
# null w: w = (1,0,0) + i(0,1,0): S itself not semisimple
bn = b_of_w([1, 0, 0], [0, 1, 0])
Sn = bn + G
ok6 &= not is_semisimple_poly(DM(Sn), charpoly_QQ(Sn))
n_null += 1
# symbolic chi_S for general w (6 real params): compare with |(l - i k)^2 + w.w|^2 (l^2 + k^2)
wr_s, wb_s = ws[:3], ws[3:]
qre = sum(a * a for a in wr_s) - sum(a * a for a in wb_s)
qim = 2 * sum(a * c for a, c in zip(wr_s, wb_s))
chiS = (bl_sym + kap * G).charpoly(lam).as_expr()
form = (lam ** 2 + kap ** 2) * sp.expand(((lam - sp.I * kap) ** 2 + qre + sp.I * qim) * ((lam + sp.I * kap) ** 2 + qre - sp.I * qim))
ok6 &= sp.expand(chiS - form) == 0
report("d4-6", ok6, f"chi_S = (l^2+k^2)|(l - i k)^2 + w.w|^2 symbolic in 6 real params; {n_ell} random rational elliptic w "
       f"(w.w > 0) bounded exactly; {n_hyp} hyperbolic (w.w < 0) and 3 generic complex w.w unbounded; null w gives non-semisimple S")

bK = sp.Rational(2, 5) * K[3]
S_, D_, M_ = M_of(bK, 1, G)
be = sp.Rational(2, 5)
ok7 = sp.expand(charpoly_QQ(M_).as_expr() - ((lam ** 2 + 1) * ((lam - be) ** 2 + 1) * ((lam + be) ** 2 + 1)) ** 2) == 0
report("d4-7", ok7, "b = beta K_3 (beta = 2/5, k = 1): chi_M = [(l^2+k^2)((l-beta)^2+k^2)((l+beta)^2+k^2)]^2 exactly: exponents "
       "+-beta +- i k (complex, not real)")

# x4-N3: for b in l_G, boundedness <=> exp(-b tau) bounded (b elliptic); co-moving dynamics is the b = 0 binding
okN3 = True
for (wr, wb) in [([2, 0, 0], [0, 1, 0]), ([1, 0, 0], [0, 2, 0]), ([1, 1, 0], [0, 0, 1]), ([1, 0, 0], [1, 1, 0]), ([1, 0, 0], [0, 1, 0])]:
    b = b_of_w(wr, wb)
    bdd_M = all(exact_bounded(b, 1, G))
    # exp(-b tau) bounded iff b semisimple with imaginary spectrum
    pb = charpoly_QQ(b)
    ss_b = is_semisimple_poly(DM(b), pb)
    imag_b = imag_spectrum(pb)
    okN3 &= (bdd_M == (ss_b and imag_b))
    # co-moving frame: (d/dtau + b + kG)(exp(-b tau) Y) = exp(-b tau)(Ydot + k G Y), numerically
    bf = np.array(b, dtype=float)
    tau = 0.77
    Y = rng.normal(size=6)
    Yd = rng.normal(size=6)
    g = expm(-bf * tau)
    lhs = (-bf @ g @ Y + g @ Yd) + (bf + G_f) @ g @ Y
    okN3 &= np.allclose(lhs, g @ (Yd + G_f @ Y), atol=1e-10)
report("x4-N3", okN3, "for b in l_G (5 cases: elliptic, hyperbolic, loxodromic, null) 'all motion bounded' <=> exp(-b tau) "
       "bounded (b semisimple, imaginary spectrum); the co-moving variable Y = exp(b tau) X always obeys the b = 0 binding")

# ---------------------------------------------------------------------------------------------
# 8. compact p_G element: Jordan structure and the bounded subspace as an exact kernel
bc = 3 * (Rab(1, 2) + Rab(4, 5))
S_, D_, M_ = M_of(bc, 4, G)
pM = charpoly_QQ(M_).as_expr()
ok8 = sp.expand(pM - (lam ** 2 + 16) ** 2 * (lam ** 2 + 25) ** 4) == 0
Mi = sp.Matrix(M_)
I12 = sp.eye(12)
ok8 &= ((Mi ** 2 + 25 * I12).rank() == 8 and ((Mi ** 2 + 25 * I12) ** 2).rank() == 4
        and (Mi ** 2 + 16 * I12).rank() == 8 and ((Mi ** 2 + 16 * I12) ** 2).rank() == 8)
ker = ((Mi ** 2 + 16 * I12) * (Mi ** 2 + 25 * I12)).nullspace()  # = sum of eigenspaces (eigenvectors only)
ok8 &= len(ker) == 8 and all(v[6 + j] == 0 for v in ker for j in (0, 1, 3, 4))
# zero-momentum flow is bounded and turns at sqrt(k^2+beta^2) on the support: S^2 = -(k^2+beta^2) there
ok8 &= (S_ ** 2)[:2, :2] == -25 * sp.eye(2) and (S_ ** 2)[2, 2] == -16
report("d4-8", ok8, "b = 3(R12+R45), k = 4: chi_M = (l^2+16)^2 (l^2+25)^4; ranks 8,4 at 25 and 8,8 at 16 (Jordan 2-blocks at "
       "+-5i); the bounded subspace ker[(M^2+16)(M^2+25)] has dim 8 and pi_1=pi_2=pi_4=pi_5=0 on it")

# ---------------------------------------------------------------------------------------------
# 9. pairing-plane boost and grading root as a connection
ok9 = True
sig3 = sp.zeros(6)
sig3[5, 2] = 1
sig3[2, 5] = 1  # e3 <-> e6
ok9 &= dag(sig3) == -sig3
for cc, kk_ in [(sp.Rational(1, 2), 1), (2, 1), (1, 1)]:
    S_, D_, M_ = M_of(cc * sig3, kk_, G)
    idx = [2, 5, 8, 11]
    Mp = sp.Matrix(M_)[idx, idx]
    N = Mp ** 2 - (cc ** 2 - kk_ ** 2) * sp.eye(4)
    ok9 &= N[:2, :2] == sp.zeros(2) and N[2:, 2:] == sp.zeros(2) and N[:2, 2:] == -2 * cc * sp.Matrix([[0, 1], [1, 0]])
    ok9 &= not is_semisimple_poly(DM(Mp), charpoly_QQ(Mp))
    if cc == kk_:
        Sp_ = sp.Matrix(S_)[[2, 5], [2, 5]]
        ok9 &= Sp_ ** 2 == sp.zeros(2) and Sp_ != sp.zeros(2)
Mh = sp.Matrix(M_of(sp.Rational(1, 2) * sig3, 1, G)[2])[[2, 5, 8, 11], [2, 5, 8, 11]]
ok9 &= (Mh ** 2 + sp.Rational(3, 4) * sp.eye(4)).rank() == 2
cs = sp.symbols('c')
ok9 &= sp.expand((cs * Kgr + kap * G) ** 2 - (cs ** 2 - kap ** 2) * sp.eye(6)) == sp.zeros(6)
report("d4-9", ok9, "pairing boost c sigma_3 (sigma e3<->e6, in so(3,3)): on the plane M^2-(c^2-k^2) = [[0,-2c sigma],[0,0]], "
       "M non-semisimple at c = 1/2, 2, 1 (k = 1), nilpotent nonzero S at threshold, rank(M^2+3/4) = 2 (as in the text after "
       "Prop 16.7); (cK + kG)^2 = (c^2-k^2) 1")

# ---------------------------------------------------------------------------------------------
# 10. every b in p_G \ {0} is non-semisimple: symbolic identities + random rational
ps_ = sp.symbols('p1:10')
bp_sym = sum((c * sum((vi * B for vi, B in zip(v, so_basis)), sp.zeros(6)) for c, v in zip(ps_, pG)), sp.zeros(6))
ok10 = sp.expand(bp_sym * G + G * bp_sym) == sp.zeros(6)
ok10 &= sp.expand((bp_sym + kap * G) ** 2 - (bp_sym ** 2 - kap ** 2 * sp.eye(6))) == sp.zeros(6)
ok10 &= sp.expand((-bp_sym + kap * G) ** 2 - (bp_sym ** 2 - kap ** 2 * sp.eye(6))) == sp.zeros(6)
Sx, Dx, Mx = M_of(bp_sym, kap, G, m)
M2 = sp.expand(Mx * Mx)
ok10 &= sp.expand(M2[:6, 6:] + 2 * bp_sym / m) == sp.zeros(6)
for trial in range(4):
    cvec = [rr() for _ in range(9)]
    if all(c == 0 for c in cvec):
        cvec[0] = 1
    b = sum((c * sum((vi * B for vi, B in zip(v, so_basis)), sp.zeros(6)) for c, v in zip(cvec, pG)), sp.zeros(6))
    ss_, im_ = exact_bounded(b, sp.Rational(5, 4), G)
    ok10 &= not ss_
report("d4-10", ok10, "symbolic in 9 p_G coordinates: (+-b + kG)^2 = b^2 - k^2, off-diagonal block of M^2 = -2b/m; M not "
       "semisimple at 4 random rational b in p_G (exact)")

# ---------------------------------------------------------------------------------------------
# 11. criterion: exact semisimplicity vs numerical bilinear orthogonality on random/mixed examples
def criterion_num(b, kk):
    """[E_mu(D) eta-orthogonal (bilinear) to E_-mu(D) for all mu], D = S^dag; kernels by SVD (tol 1e-7)"""
    Df = -np.array(b, dtype=float) + kk * G_f
    ev = np.linalg.eigvals(Df)
    reps = []
    for z in ev:
        if all(abs(z - r) > 1e-6 for r in reps):
            reps.append(z)

    def kern(mu):
        U, s, Vh = np.linalg.svd(Df - mu * np.eye(6))
        k = int((s < 1e-7).sum())
        return Vh[6 - k:].conj().T

    for mu in reps:
        for nu in reps:
            if abs(mu + nu) < 1e-6:
                Bm = kern(nu).T @ eta_f @ kern(mu)
                if np.abs(Bm).max() > 1e-8:
                    return False
    return True


tests11 = [("l_G rot", Rab(1, 2) - Rab(4, 5), 1), ("p_G compact", Rab(1, 2) + Rab(4, 5), 1),
           ("mixed", Rab(1, 2) + sp.Rational(1, 3) * Rab(2, 6), 1), ("pairing boost", sp.Rational(1, 2) * sig3, 1)]
for t in range(4):
    tests11.append((f"random{t}", sum((rr() * B for B in so_basis), sp.zeros(6)), sp.Rational(3, 2)))
# compact screening and gyroscopic examples (expected bounded)
tests11.append(("compact screened", sp.diag(so3([sp.Rational(1, 5), 0, 1]), so3([-sp.Rational(1, 5), 0, 1])), 1))
Ygy = sp.Matrix([[sp.Rational(1, 4), 0, 0], [0, -sp.Rational(1, 4), 0], [0, 0, 0]])
tests11.append(("gyroscopic", sp.Rational(1, 2) * L[3] + sp.BlockMatrix([[Z3, Ygy], [Ygy, Z3]]).as_explicit(), 1))
ok11 = True
agree = 0
for name, b, kk in tests11:
    ss_, im_ = exact_bounded(b, kk, G)
    S_ = sp.Matrix(b) + kk * G
    Sss = is_semisimple_poly(DM(S_), charpoly_QQ(S_))
    crit = Sss and criterion_num(b, float(kk))
    ok11 &= (crit == ss_)
    agree += (crit == ss_)
report("d4-11", ok11, f"Roth/virial criterion: exact semisimplicity of M agrees with [S semisimple and E_mu(S^dag) "
       f"eta-orthogonal to E_-mu(S^dag)] (numeric, tol 1e-8) on {agree}/{len(tests11)} connections incl. 4 random, "
       "a compact screened and a gyroscopic one")

# ---------------------------------------------------------------------------------------------
# 12. gyroscopic family: symbolic chi_S, random exact points, Jacobian rank at an exact point
a_, c_, d_ = sp.symbols('a c d', real=True)
Ys = sp.Matrix([[c_, d_, 0], [d_, -c_, 0], [0, 0, 0]])
bY = sp.BlockMatrix([[Z3, Ys], [Ys, Z3]]).as_explicit()  # z -> i Y conj(z)
bgy = a_ * L[3] + bY
r2 = c_ ** 2 + d_ ** 2
ok12 = sp.expand((bgy + kap * G).charpoly(lam).as_expr()
                 - (lam ** 2 + kap ** 2) * (lam ** 2 + (kap - a_) ** 2 - r2) * (lam ** 2 + (kap + a_) ** 2 - r2)) == 0
ok12 &= sp.expand(bY * G + G * bY) == sp.zeros(6) and sp.expand(br(bgy, G)) != sp.zeros(6)
# K_3 commutes with G, L_3, b_Y; eigenplanes totally null
ok12 &= sp.expand(br(K[3], bY)) == sp.zeros(6) and br(K[3], G) == sp.zeros(6) and br(K[3], L[3]) == sp.zeros(6)
Upl = (K[3] - sp.eye(6)).nullspace()
Umi = (K[3] + sp.eye(6)).nullspace()
ok12 &= len(Upl) == 2 and len(Umi) == 2 and all((x.T * eta * y)[0] == 0 for x in Upl for y in Upl) and all((x.T * eta * y)[0] == 0 for x in Umi for y in Umi)
cnt_b = 0
for trial in range(4):
    aa = sp.Rational(int(rng.integers(1, 7)), 8) * (1 if trial % 2 == 0 else -1)
    lim = min(abs(1 - aa), abs(1 + aa))
    cc_ = sp.Rational(int(rng.integers(1, 5)), 10) * lim
    dd_ = sp.Rational(int(rng.integers(0, 5)), 10) * lim
    b = bgy.subs({a_: aa, c_: cc_, d_: dd_})
    ok12 &= exact_bounded(b, 1, G) == (True, True)
    cnt_b += 1
# a = 0: not bounded; r between |k-a| and |k+a|: not bounded
ok12 &= not all(exact_bounded(bgy.subs({a_: 0, c_: sp.Rational(1, 4), d_: 0}), 1, G))
ok12 &= not all(exact_bounded(bgy.subs({a_: sp.Rational(1, 2), c_: sp.Rational(3, 4), d_: 0}), 1, G))


def five_conditions(bvec, kk=1.0):
    bmat = sum(c * B for c, B in zip(bvec, sob_f))
    Sf = bmat + kk * G_f
    Df = -bmat + kk * G_f
    co = np.poly(Sf)  # [1, c5, c4, c3, c2, c1, c0]
    ev, V = np.linalg.eig(Df)
    idx = sorted([i for i in range(6) if ev[i].imag > 1e-6], key=lambda i: ev[i].imag)
    iso = [float(np.real(V[:, i].conj() @ eta_f @ V[:, i]) / np.real(V[:, i].conj() @ V[:, i])) for i in idx]
    return np.array([co[3], co[5]] + iso)


def jac_rank(b0):
    x0 = np.array([float(v) for v in coords_so(sp.Matrix(b0))])
    F0 = five_conditions(x0)
    h = 1e-6
    J = np.array([(five_conditions(x0 + h * e) - five_conditions(x0 - h * e)) / (2 * h) for e in np.eye(15)]).T
    sv = np.linalg.svd(J, compute_uv=False)
    return np.abs(F0).max(), sv


from scipy.optimize import least_squares


def local_probe(b0sym, t=1e-2, ntrial=3):
    """rank of the 5x15 Jacobian and least-squares residual of F(b0 + t v + R s) = 0 along random kernel directions v"""
    x0 = np.array([float(v) for v in coords_so(sp.Matrix(b0sym))])
    F0, sv = jac_rank(b0sym)
    h = 1e-6
    J = np.array([(five_conditions(x0 + h * e) - five_conditions(x0 - h * e)) / (2 * h) for e in np.eye(15)]).T
    U_, s_, Vt = np.linalg.svd(J)
    r = int((s_ > 1e-6 * s_[0]).sum())
    Kn, Rw = Vt[r:].T, Vt[:r].T
    res = []
    for _ in range(ntrial):
        v = Kn @ rng.normal(size=Kn.shape[1])
        v /= np.linalg.norm(v)
        out = []
        for tt in (t, t / 2):
            sol = least_squares(lambda s0: five_conditions(x0 + tt * v + Rw @ s0), np.zeros(r), xtol=1e-15, ftol=1e-15, gtol=1e-15)
            out.append(np.linalg.norm(sol.fun))
        res.append(out)
    return F0, r, res


F0c, rc, resc = local_probe(sp.diag(so3([sp.Rational(1, 5), 0, 1]), so3([-sp.Rational(1, 5), 0, 1])))
ok12 &= F0c < 1e-10 and rc == 5 and max(max(x) for x in resc) < 1e-11
report("d4-12", ok12, f"gyroscopic family: chi_S symbolic in (a,c,d,k); [b,G] != 0; K_3 commutes with G, L_3, b_Y, eigenplanes "
       f"totally null; {cnt_b} random exact points with r < min|k+-a| bounded; a = 0 and |k-a| < r < |k+a| not. Dimension: "
       f"at the exact compact screened point the 5 conditions vanish, the finite-difference Jacobian has rank 5 and the "
       f"conditions are solvable along random kernel directions (residual < 1e-11): a 10-dim bounded manifold there")

# x4-N4: the gyroscopic family and the elliptic part of l_G lie on the singular locus of the five conditions
F0g, rg, resg = local_probe(bgy.subs({a_: sp.Rational(1, 2), c_: sp.Rational(1, 4), d_: 0}))
F0l, rl, resl = local_probe(b_of_w([2, 0, 0], [0, 1, 0]))
ratio_g = [x[0] / x[1] for x in resg]
ratio_l = [x[0] / x[1] for x in resl]
okN4 = F0g < 1e-10 and F0l < 1e-10 and rg == 4 and rl == 4 and all(3.5 < q < 4.5 for q in ratio_g + ratio_l)
report("x4-N4", okN4, f"at the exact gyroscopic point (1/2,1/4,0) and at the l_G point w = (2,i,0) the conditions hold but the "
       f"Jacobian has rank 4, and along random kernel directions the least-squares residual scales as t^2 (halving ratios "
       f"{', '.join(f'{q:.2f}' for q in ratio_g + ratio_l)}): both lie on the singular locus, so the '10-dim submanifold' "
       f"statement is established only at regular points, not along the gyroscopic family itself")

# x4-N1: compact (gauge-admissible, Cor 2.4) screening without any boost
okN1 = True
cntN1 = 0
for (av, cv) in [([0, 0, 1], [sp.Rational(1, 5), 0, 0]), ([0, 0, 3], [sp.Rational(1, 7), 0, 0]),
                 ([1, 1, 0], [sp.Rational(1, 3), -sp.Rational(1, 3), sp.Rational(1, 2)]), ([0, 0, sp.Rational(1, 2)], [0, 2, 0])]:
    A = so3(av)
    C = so3(cv)
    b = sp.diag(A + C, A - C)
    assert sum(x * y for x, y in zip(av, cv)) == 0
    ok_b = exact_bounded(b, 1, G) == (True, True)
    okN1 &= ok_b and br(b, G) != sp.zeros(6) and (b + b.T) == sp.zeros(6)
    cntN1 += 1
# unequal rates: not bounded; anti-aligned (pure p_G compact): not bounded
okN1 &= not all(exact_bounded(sp.diag(so3([0, 0, 1]), so3([0, 0, 2])), 1, G))
okN1 &= not all(exact_bounded(sp.diag(so3([0, 0, 1]), so3([0, 0, -1])), 1, G))
# random numeric scan: generic compact b is unbounded, a.c = 0 ones bounded
def bounded_num(bf, kk):
    Sf = bf + kk * G_f
    Df = -bf + kk * G_f
    Mf = np.block([[-Sf, np.eye(6)], [np.zeros((6, 6)), Df]])
    ev, V = np.linalg.eig(Mf)
    if np.max(np.abs(ev.real)) > 1e-8:
        return False
    s = np.linalg.svd(V, compute_uv=False)
    return s[-1] / s[0] > 1e-6


so3n = lambda v: np.array(so3(list(v)), dtype=float)
gen = sum(bounded_num(np.block([[so3n(rng.normal(size=3)), np.zeros((3, 3))], [np.zeros((3, 3)), so3n(rng.normal(size=3))]]), 1.0) for _ in range(500))
scr = 0
for _ in range(500):
    av = rng.normal(size=3)
    cv = rng.normal(size=3)
    cv -= av * (av @ cv) / (av @ av)
    cv *= rng.uniform(0.05, 1.5)
    scr += bounded_num(np.block([[so3n(av + cv), np.zeros((3, 3))], [np.zeros((3, 3)), so3n(av - cv)]]), rng.uniform(0.3, 3))
okN1 &= gen == 0 and scr >= 495
report("x4-N1", okN1, f"compact (gauge-admissible) screening: b = diag(A+C, A-C), A = so3(a), C = so3(c), a.c = 0, is bounded "
       f"exactly at {cntN1} rational points though [b,G] != 0 and b has no boost part; unequal rates and anti-aligned are not; "
       f"numerically 0/500 generic compact b and {scr}/500 random a.c = 0 ones are bounded")

# ---------------------------------------------------------------------------------------------
# 13. gauge removal: kappa = 0 drift, Larmor energy identity, p_G conjugation
ok13 = True
for trial in range(3):
    b = rand_so()
    Mf = np.block([[-b, np.eye(6)], [np.zeros((6, 6)), -b]])
    # never semisimple: rank test on the nilpotent part (M + diag(b,b))^2 = 0 but != 0
    Nn = Mf + np.block([[b, np.zeros((6, 6))], [np.zeros((6, 6)), b]])
    ok13 &= np.allclose(Nn @ Nn, 0) and not np.allclose(Nn, 0)
for trial in range(3):
    wr = rng.normal(size=3)
    wb = rng.normal(size=3)
    b = np.array(b_of_w(list(wr), list(wb)), dtype=float)
    kk = rng.uniform(0.3, 2)
    mm = rng.uniform(0.5, 2)
    tau = rng.uniform(-2, 2)
    Y = rng.normal(size=6)
    pY = rng.normal(size=6)
    X = expm(-b * tau) @ Y
    pX = expm(b.T * tau) @ pY
    Hl = pX @ np.linalg.inv(eta_f) @ pX / (2 * mm) - pX @ (b + kk * G_f) @ X
    Hr = pY @ np.linalg.inv(eta_f) @ pY / (2 * mm) - pY @ (kk * G_f) @ Y
    ok13 &= abs(Hl - (Hr - pY @ b @ Y)) < 1e-9 * max(1, abs(Hl))
for trial in range(3):
    v = rng.normal(size=9)
    bp = sum(c * np.array(sum((vi * B for vi, B in zip(w, so_basis)), sp.zeros(6)), dtype=float) for c, w in zip(v, pG))
    tau = 0.6
    ok13 &= np.allclose(expm(bp * tau) @ G_f @ expm(-bp * tau), expm(2 * bp * tau) @ G_f, atol=1e-9)
    ok13 &= np.allclose(bp @ G_f - G_f @ bp, 2 * bp @ G_f, atol=1e-12)
report("d4-13", ok13, "k = 0: M + diag(b,b) is nonzero nilpotent (free drift); Larmor H_lab = H_rot - p_Y^T b Y at 3 random "
       "b in l_G (incl. boosts); for b in p_G exp(b t)G exp(-b t) = exp(2bt)G and [b,G] = 2bG (numeric)")

# ---------------------------------------------------------------------------------------------
# 14. split virial: exact at random rational states, integral form along a screened trajectory
ok14 = True
for trial in range(5):
    Sx_ = sp.Matrix(6, 6, lambda i, j: rr())
    Dx_ = eta.inv() * Sx_.T * eta
    X = sp.Matrix([rr() for _ in range(6)])
    P = sp.Matrix([rr() for _ in range(6)])
    mm_ = sp.Rational(3, 2)
    Xd = P / mm_ - Sx_ * X
    Pd = Dx_ * P
    ok14 &= sp.simplify((Xd.T * eta * P + X.T * eta * Pd)[0] - (P.T * eta * P)[0] / mm_) == 0
bscr = np.array(sp.diag(so3([sp.Rational(1, 5), 0, 1]), so3([-sp.Rational(1, 5), 0, 1])), dtype=float)
Sf = bscr + G_f
Df = -bscr + G_f


def rhs(t, y):
    return np.concatenate([y[6:] - Sf @ y[:6], Df @ y[6:]])


y0 = rng.normal(size=12)
sol = solve_ivp(rhs, (0, 30), y0, method='DOP853', rtol=1e-12, atol=1e-12, dense_output=True)
ts = np.linspace(0, 30, 3001)
Ys_ = sol.sol(ts)
vir = np.array([Ys_[:6, i] @ eta_f @ Ys_[6:, i] for i in range(len(ts))])
pp = np.array([Ys_[6:, i] @ eta_f @ Ys_[6:, i] for i in range(len(ts))])
integ = np.concatenate([[0], np.cumsum((pp[1:] + pp[:-1]) / 2 * np.diff(ts))])
ok14 &= np.abs((vir - vir[0]) - integ).max() < 1e-4
report("d4-14", ok14, "split virial d/dt eta(X,pi) = eta(pi,pi)/m exact at 5 random rational (S, X, pi); integral form along a "
       "DOP853 trajectory (rtol 1e-12) of the compact screened connection, trapezoid error < 1e-4")

# ---------------------------------------------------------------------------------------------
# 15. Krein signatures from the canonical Hessian
def krein_check(b, kk, mm=1.0, eta_m=None, n=6):
    eta_m = eta_f if eta_m is None else eta_m
    Gm = G_f if n == 6 else G3_f
    Sf = np.array(b, dtype=float) + kk * Gm
    Hess = np.block([[np.zeros((n, n)), -Sf.T], [-Sf, np.linalg.inv(eta_m) / mm]])
    Jn = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    Mc = Jn @ Hess
    ev = np.linalg.eigvals(Mc)
    oms = sorted(set(round(z.imag, 6) for z in ev if z.imag > 1e-6))
    zp = 0.0
    sigs = []
    for om in oms:
        U, s, Vh = np.linalg.svd(Mc - 1j * om * np.eye(2 * n))
        Kb = Vh[s.size - int((s < 1e-6).sum()):].conj().T  # kernel basis
        H = Kb.conj().T @ Hess @ Kb
        e = np.linalg.eigvalsh((H + H.conj().T) / 2)
        npos, nneg = int((e > 1e-9).sum()), int((e < -1e-9).sum())
        sigs.append((om, npos, nneg))
        zp += 0.5 * om * (npos - nneg)
    return sigs, zp


ok15 = True
cases15 = [("Zeeman 3/7", Rab(1, 2) - Rab(4, 5), 3 / 7), ("Zeeman 3/2", Rab(1, 2) - Rab(4, 5), 1.5), ("b=0", sp.zeros(6), 1),
           ("w=(2,i,0)", b_of_w([2, 0, 0], [0, 1, 0]), 1),
           ("gyro", bgy.subs({a_: sp.Rational(1, 2), c_: sp.Rational(1, 4), d_: 0}), 1),
           ("compact screened", sp.diag(so3([sp.Rational(1, 5), 0, 1]), so3([-sp.Rational(1, 5), 0, 1])), 1)]
desc = []
for name, b, scale in cases15:
    bb = np.array(b, dtype=float) * (scale if name.startswith("Zeeman") else 1)
    sigs, zp = krein_check(bb, 1.0)
    ok15 &= all(p == q and p > 0 for _, p, q in sigs) and abs(zp) < 1e-12
    desc.append(name + ":" + ",".join(f"({p},{q})" for _, p, q in sigs))
report("d4-15", ok15, "Krein: energy form (canonical Hessian on ker(M - i w)) has signature (k,k) at every frequency and the "
       "zero-point sum vanishes: " + "; ".join(desc))

# ---------------------------------------------------------------------------------------------
# 16. model block
q1, q2, p1, p2, om = sp.symbols('q1 q2 p1 p2 omega')
Hm = om * (q1 * p2 - q2 * p1) + (q1 ** 2 + q2 ** 2) / 2
vars_ = [q1, q2, p1, p2]
Hh = sp.hessian(Hm, vars_)
J4 = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]])
Mm = (J4 * Hh).subs(om, 3)
ok16 = sp.expand(Mm.charpoly(lam).as_expr() - (lam ** 2 + 9) ** 2) == 0 and not is_semisimple_poly(DM(Mm), charpoly_QQ(Mm))
pb = sum(sp.diff(q1 * p2 - q2 * p1, q) * sp.diff(q1 ** 2 + q2 ** 2, p) - sp.diff(q1 * p2 - q2 * p1, p) * sp.diff(q1 ** 2 + q2 ** 2, q)
         for q, p in [(q1, p1), (q2, p2)])
ok16 &= sp.simplify(pb) == 0
# a signed oscillator sum generates a semisimple flow
ws_ = [1, 2, 2]
sg = [1, -1, 1]
Hosc = sum(s_ * (pp_ ** 2 + w_ ** 2 * qq_ ** 2) / 2 for s_, w_, qq_, pp_ in zip(sg, ws_, sp.symbols('Q1:4'), sp.symbols('P1:4')))
vv = list(sp.symbols('Q1:4')) + list(sp.symbols('P1:4'))
J6 = sp.BlockMatrix([[sp.zeros(3), sp.eye(3)], [-sp.eye(3), sp.zeros(3)]]).as_explicit()
Mo = J6 * sp.hessian(Hosc, vv)
ok16 &= is_semisimple_poly(DM(Mo), charpoly_QQ(Mo))
report("d4-16", ok16, "model block H = w(q1p2 - q2p1) + (q1^2+q2^2)/2: chi = (l^2+9)^2 at w = 3, not semisimple, {L_z, r^2} = 0; "
       "a signed-oscillator sum (with degenerate frequencies) has a semisimple flow")

# ---------------------------------------------------------------------------------------------
# 17. three constituents: 512-dim sparse realisation, Clifford relations, vector action, Thm 17.21 for all 70 coordinate U
eta3d = etad * 3
omega8 = E8[0] @ E8[1] @ E8[2] @ E8[3] @ E8[4] @ E8[5]
Id8 = sps.identity(8, dtype=np.int64, format='csr')
Om = sps.csr_matrix(omega8)
Es = []
for kq_ in range(3):
    for a in range(6):
        factors = [Om] * kq_ + [sps.csr_matrix(E8[a])] + [Id8] * (2 - kq_)
        X = factors[0]
        for f in factors[1:]:
            X = sps.kron(X, f, format='csr')
        Es.append(X)
I512 = sps.identity(512, dtype=np.int64, format='csr')
ok17 = True
for A in range(18):
    for B in range(A, 18):
        Ac = Es[A] @ Es[B] + Es[B] @ Es[A]
        target = (2 * eta3d[A] * I512) if A == B else None
        ok17 &= (abs(Ac - target).sum() == 0) if A == B else (Ac.count_nonzero() == 0)
# vector action: (1/2)[E_A E_B, E_C] = eta_BC E_A - eta_AC E_B, sampled on all A<B and all C
for A in range(18):
    for B in range(A + 1, 18):
        BV = Es[A] @ Es[B]
        for C in range(18):
            Xc = (BV @ Es[C] - Es[C] @ BV)
            tgt = sps.csr_matrix((512, 512), dtype=np.int64)
            if B == C:
                tgt = tgt + 2 * eta3d[B] * Es[A]
            if A == C:
                tgt = tgt - 2 * eta3d[A] * Es[B]
            ok17 &= abs(Xc - tgt).sum() == 0
n18 = 18


def R3(a, b):
    return Rab(a, b, n=18, et=eta3d)


G3 = sp.zeros(18)
for kq_ in range(3):
    for i in range(3):
        a = 6 * kq_ + i
        G3[a + 3, a] = 1
        G3[a, a + 3] = -1
G3_f = np.array(G3, dtype=float)
eta3 = sp.diag(*eta3d)
ok17 &= G3 * G3 == -sp.eye(18) and G3.T * eta3 * G3 == -eta3
so99 = [R3(a, b) for a in range(1, 19) for b in range(a + 1, 19)]
Wc3 = DomainMatrix.from_Matrix(sp.Matrix([list(br(B, G3)) for B in so99]).T).convert_to(QQ)
ok17 &= (len(so99) - Wc3.rank() == 72)
internal = [2, 3, 7, 8, 9, 13, 14, 15]
colour = [10, 11, 12, 16, 17, 18]
Eset = {7, 8, 9, 13, 14, 15}
soC = [R3(a, b) for a, b in itertools.combinations(colour, 2)]


def nullity(mats):
    W = DomainMatrix.from_Matrix(sp.Matrix([list(br(B, G3)) for B in mats]).T).convert_to(QQ)
    return len(mats) - W.rank()


okU = True
nU = 0
nullC = nullity(soC)
for U in itertools.combinations(internal, 4):
    soU = [R3(a, b) for a, b in itertools.combinations(U, 2)]
    dd = len(set(U) & Eset)
    okU &= nullity(soU) == 0 and nullity(soU + soC) == sp.binomial(dd, 2)
    nU += 1
okU &= nullC == 0


# two non-coordinate U (random rational): generic (d = 2) and with three vectors in E (d = 3)
def Rvec(x, y):
    return x * y.T * eta3 - y * x.T * eta3


def evec(i):
    v = sp.zeros(18, 1)
    v[i - 1] = 1
    return v


for mode in ("generic", "d3"):
    vecs = []
    for j in range(4):
        if mode == "d3" and j < 3:
            v = sum((rr() * evec(i) for i in sorted(Eset)), sp.zeros(18, 1))
        else:
            v = sum((rr() * evec(i) for i in internal), sp.zeros(18, 1))
        vecs.append(v)
    soU = [Rvec(x, y) for x, y in itertools.combinations(vecs, 2)]
    Umat = sp.Matrix.hstack(*vecs)
    Emat = sp.Matrix.hstack(*[evec(i) for i in sorted(Eset)])
    dd = Umat.rank() + Emat.rank() - sp.Matrix.hstack(Umat, Emat).rank()
    okU &= nullity(soU) == 0 and nullity(soU + soC) == sp.binomial(dd, 2) and dd == (2 if mode == "generic" else 3)
report("d4-17", ok17 and okU, f"512-dim sparse realisation: Cl(9,9) relations and (1/2)[E_A E_B, E_C] = eta_BC E_A - eta_AC E_B "
       f"for all A<B, C; G3 anti-isometry, G3^2 = -1; dim Cent(G3) in so(9,9) = 72; Thm 17.21 checked on all {nU} coordinate "
       "U and two random rational U (d = 2, 3): so(U), so(6)_C meet Cent in 0, their sum in C(d,2)")

# ---------------------------------------------------------------------------------------------
# 18. so(4)_Delta
def rot3(i, j):  # e_{i'}e_{j'} - e_i e_j acting by (1/2)ad
    return R3(i + 3, j + 3) - R3(i, j)


a1, b1, c1, d1 = 7, 13, 14, 15
Tp = {1: (rot3(b1, c1) + rot3(a1, d1)) / 2, 2: (rot3(a1, c1) - rot3(b1, d1)) / 2, 3: (rot3(a1, b1) + rot3(c1, d1)) / 2}
Tm = {1: (rot3(b1, c1) - rot3(a1, d1)) / 2, 2: (rot3(a1, c1) + rot3(b1, d1)) / 2, 3: (rot3(a1, b1) - rot3(c1, d1)) / 2}
ok18 = all(br(Tp[i], Tm[j]) == sp.zeros(18) for i in range(1, 4) for j in range(1, 4))
for T in (Tp, Tm):
    ok18 &= br(T[1], T[2]) == -T[3] and br(T[2], T[3]) == -T[1] and br(T[3], T[1]) == -T[2]
    ok18 &= all(br(T[i], G3) == sp.zeros(18) for i in range(1, 4))
basis6 = [Tp[1], Tp[2], Tp[3], Tm[1], Tm[2], Tm[3]]
TrF = sp.Matrix(6, 6, lambda i, j: (basis6[i] * basis6[j]).trace())
ok18 &= inertia(TrF) == (0, 6)
pT = sp.factor(Tp[3].charpoly(lam).as_expr())
ok18 &= sp.expand(Tp[3].charpoly(lam).as_expr() - lam ** 10 * (lam ** 2 + sp.Rational(1, 4)) ** 4) == 0
ok18 &= sp.expand(Tm[3].charpoly(lam).as_expr() - lam ** 10 * (lam ** 2 + sp.Rational(1, 4)) ** 4) == 0
# so(4)_Delta = span rot(i,j), i<j in W
spanW = [rot3(i, j) for i, j in itertools.combinations([7, 13, 14, 15], 2)]
ok18 &= sp.Matrix([list(X) for X in spanW + basis6]).rank() == 6
report("d4-18", ok18, "T+-_i (rot acting by (1/2)ad): [T+,T-] = 0, [T1,T2] = -T3 cyclically, commute with G3, trace form "
       "negative definite on the 6-dim span = span{rot(i,j): i<j in W}; chi(T+-_3) = l^10 (l^2+1/4)^4")

# ---------------------------------------------------------------------------------------------
# 19. spectra on 18 dims and the tearing examples
def chi18(b, kk):
    return charpoly_QQ(sp.Matrix(b) + kk * G3).as_expr()


ok19 = sp.expand(chi18(2 * Tp[3], 5) - (lam ** 2 + 25) ** 5 * (lam ** 2 + 16) ** 2 * (lam ** 2 + 36) ** 2) == 0
ok19 &= sp.expand(chi18(2 * Tp[3] + 4 * Tm[3], 5)
                  - (lam ** 2 + 25) ** 5 * (lam ** 2 + 64) * (lam ** 2 + 36) * (lam ** 2 + 16) * (lam ** 2 + 4)) == 0
for b in (2 * Tp[3], 2 * Tp[3] + 4 * Tm[3]):
    ok19 &= exact_bounded(b, 5, G3) == (True, True)
    A_ = -(b + 5 * G3)
    D_ = -b + 5 * G3
    Y_ = -G3 / 10
    ok19 &= A_ * Y_ - Y_ * D_ == -sp.eye(18)
    sigs, zp = krein_check(np.array(b, dtype=float), 5.0, n=18, eta_m=np.array(eta3, dtype=float))
    ok19 &= all(p == q for _, p, q in sigs) and abs(zp) < 1e-10
ok19 &= not exact_bounded(R3(7, 13), 5, G3)[0] and not exact_bounded(R3(10, 16), 5, G3)[0]
report("d4-19", ok19, "k = 5: chi_S(2T+_3) = (l^2+25)^5 (l^2+16)^2 (l^2+36)^2; chi_S(2T+_3 + 4T-_3) has lines 8,6,4,2 and 5^5; "
       "both bounded exactly, Y = -G3/(2k) solves Sylvester, Krein (k,k), zero-point 0; R(7,13) alone and R(10,16) alone "
       "give non-semisimple M")

# x4-N2: an unlocked Pati-Salam connection that keeps the binding
okN2 = True
descN2 = []
for name, b in [("R(7,13)+R(10,11)", R3(7, 13) + R3(10, 11)), ("R(7,13)-R(10,11)", R3(7, 13) - R3(10, 11))]:
    inPS = True  # R(7,13) in so(W), R(10,11) in so(6)_C
    comm = br(b, G3) == sp.zeros(18)
    ss_, im_ = exact_bounded(b, 5, G3)
    okN2 &= inPS and (not comm) and ss_ and im_
    descN2.append(f"{name}: [b,G3]!=0, bounded")
# and the locked/anti-locked controls
okN2 &= br(R3(7, 13) - R3(10, 16), G3) == sp.zeros(18) and all(exact_bounded(R3(7, 13) - R3(10, 16), 5, G3))
okN2 &= not exact_bounded(R3(7, 13) + R3(10, 16), 5, G3)[0]
report("x4-N2", okN2, "COUNTEREXAMPLE to 'only the locked diagonal is compatible with binding' (d4-19 / Prop so4spec text): "
       + "; ".join(descN2) + " (exact, k = 5); controls: locked R(7,13)-R(10,16) commutes and is bounded, anti-locked is not")

# ---------------------------------------------------------------------------------------------
# 20-24 inherited Yang-Mills facts
def gellmann(N):
    gens = []
    for j in range(N):
        for k in range(j + 1, N):
            Msym = sp.zeros(N)
            Msym[j, k] = Msym[k, j] = sp.Rational(1, 2)
            gens.append(Msym)
            Masy = sp.zeros(N)
            Masy[j, k] = -sp.I / 2
            Masy[k, j] = sp.I / 2
            gens.append(Masy)
    for l in range(1, N):
        Md = sp.zeros(N)
        for i in range(l):
            Md[i, i] = 1
        Md[l, l] = -l
        gens.append(Md / sp.sqrt(2 * l * (l + 1)))
    return gens


T4 = gellmann(4)
ok20 = len(T4) == 15 and all(sp.simplify((T4[i] * T4[j]).trace() - (sp.Rational(1, 2) if i == j else 0)) == 0 for i in range(15) for j in range(15))
for i in range(4):
    for j in range(4):
        Xu = sp.zeros(4)
        Xu[i, j] = 1
        lhs = sum((T * Xu * T for T in T4), sp.zeros(4))
        ok20 &= sp.simplify(lhs - (Xu.trace() * sp.eye(4) - Xu / 4) / 2) == sp.zeros(4)
report("d4-20", ok20, "SU(4) generalised Gell-Mann basis (tr T^A T^B = delta/2): sum T X T = (tr X 1 - X/4)/2 on all 16 matrix units")

T4n = [np.array(T, dtype=complex) for T in T4]
s2 = [np.array([[0, 1], [1, 0]]) / 2, np.array([[0, -1j], [1j, 0]]) / 2, np.array([[1, 0], [0, -1]]) / 2]


def cas(gens):
    return sum(g @ g for g in gens)


I4 = np.eye(4)
I2c = np.eye(2)
g44 = [np.kron(T, I4) + np.kron(I4, T) for T in T4n]
C44 = cas(g44)
Pswap = np.zeros((16, 16))
for i in range(4):
    for j in range(4):
        Pswap[i * 4 + j, j * 4 + i] = 1
Psym = (np.eye(16) + Pswap) / 2
Pas = (np.eye(16) - Pswap) / 2
ok21 = np.allclose(C44 @ Psym, 4.5 * Psym) and np.allclose(C44 @ Pas, 2.5 * Pas)
gb4 = [np.kron(-T.conj(), I4) + np.kron(I4, T) for T in T4n]
Cb = cas(gb4)
evb = np.round(np.linalg.eigvalsh(Cb), 8)
ok21 &= sorted(set(evb)) == [0.0, 4.0] and (evb == 0).sum() == 1
C4 = cas(T4n)
ok21 &= np.allclose(C4, 15 / 8 * I4)
f = lambda Cp, C1, C2: (Cp - C1 - C2) / 2
ok21 &= np.isclose(f(4.5, 15 / 8, 15 / 8), 3 / 8) and np.isclose(f(2.5, 15 / 8, 15 / 8), -5 / 8)
ok21 &= np.isclose(f(0, 15 / 8, 15 / 8), -15 / 8) and np.isclose(f(4, 15 / 8, 15 / 8), 1 / 8)
ok21 &= np.isclose(f(2, 0.75, 0.75), 0.25) and np.isclose(f(0, 0.75, 0.75), -0.75)
# Sym^2(C^4 (x) C^2): joint Casimir eigenspaces of dims 30 and 6
g8_4 = [np.kron(T, I2c) for T in T4n]
g8_2 = [np.kron(I4, s) for s in s2]
dim8 = 8
Psw8 = np.zeros((64, 64))
for i in range(8):
    for j in range(8):
        Psw8[i * 8 + j, j * 8 + i] = 1
Ps8 = (np.eye(64) + Psw8) / 2
U8, s8, _ = np.linalg.svd(Ps8)
Bsym = U8[:, :36]
lift = lambda g: np.kron(g, np.eye(8)) + np.kron(np.eye(8), g)
C4s = Bsym.conj().T @ cas([lift(g) for g in g8_4]) @ Bsym
C2s = Bsym.conj().T @ cas([lift(g) for g in g8_2]) @ Bsym
ev4, V4 = np.linalg.eigh(C4s)
ok21 &= np.isclose(sorted(set(np.round(ev4, 8)))[0], 2.5) and (np.isclose(ev4, 4.5)).sum() == 30 and (np.isclose(ev4, 2.5)).sum() == 6
Vten = V4[:, np.isclose(ev4, 4.5)]
Vsix = V4[:, np.isclose(ev4, 2.5)]
ok21 &= np.allclose(np.linalg.eigvalsh(Vten.conj().T @ C2s @ Vten), 2.0) and np.allclose(np.linalg.eigvalsh(Vsix.conj().T @ C2s @ Vsix), 0.0)
report("d4-21", ok21, "Casimirs 15/8, 9/2 (Sym^2), 5/2 (Lambda^2), 4 (adjoint, 4bar x 4) computed numerically; channel factors "
       "+3/8, -5/8, -15/8, +1/8, +1/4, -3/4; Sym^2(C^4 x C^2) = 30 (C2 = 9/2, 2) + 6 (C2 = 5/2, 0)")

mp.mp.dps = 30
Lc, Mc_ = mp.mpf(1), mp.mpf('0.3')
num = 2 * mp.pi ** 2 / (2 * mp.pi) ** 4 * mp.quad(lambda k: k ** 3 / (k ** 2 + Mc_ ** 2), [0, Lc])
closed = (Lc ** 2 - Mc_ ** 2 * mp.log(1 + Lc ** 2 / Mc_ ** 2)) / (16 * mp.pi ** 2)
fx = lambda x: 1 - x ** 2 * mp.log(1 + 1 / x ** 2)
root = mp.findroot(lambda x: mp.mpf(7) / 4 * fx(x) - 1, 0.5)
xs = sp.symbols('x', positive=True)
fsym = 1 - xs ** 2 * sp.log(1 + 1 / xs ** 2)
ok22 = abs(num - closed) < mp.mpf(10) ** -25 and abs(root - mp.mpf('0.5330234964')) < 1e-10
ok22 &= sp.simplify(sp.diff(fsym, xs) - 2 * xs * (1 / (1 + xs ** 2) - sp.log(1 + xs ** -2))) == 0
grid = [mp.mpf(i) / 50 for i in range(1, 500)]
vals = [fx(x) for x in grid]
ok22 &= all(vals[i + 1] < vals[i] for i in range(len(vals) - 1)) and abs(fx(mp.mpf('1e-8')) - 1) < 1e-12 and abs(fx(mp.mpf(1e4))) < 1e-7
report("d4-22", ok22, f"gap integral: mpmath quadrature = closed form to 1e-25; f strictly decreasing on a grid, limits 1 and 0, "
       f"f' formula symbolic; root of (7/4) f(x) = 1 is x = {mp.nstr(root, 12)}")

CF = sp.Rational(15, 8)
Nf = sp.symbols('N_f')
b0 = sp.Rational(11, 3) * 2 - sp.Rational(4, 3) * sp.Rational(1, 2) * Nf
g_, mu_ = sp.symbols('g mu', positive=True)
Lam = mu_ * sp.exp(-8 * sp.pi ** 2 / (b0.subs(Nf, 2) * g_ ** 2))
# d Lambda / d ln mu along dg/dln mu = -b0 g^3/(16 pi^2)
dL = sp.diff(Lam, mu_) * mu_ + sp.diff(Lam, g_) * (-b0.subs(Nf, 2) * g_ ** 3 / (16 * sp.pi ** 2))
ok23 = sp.simplify(sp.pi / (3 * CF) - 8 * sp.pi / 45) == 0 and sp.simplify(b0 - (22 - 2 * Nf) / 3) == 0 and b0.subs(Nf, 2) == 6
ok23 &= sp.simplify(dL) == 0 and abs(float(8 * sp.pi / 45) - 0.5585) < 1e-4
report("d4-23", ok23, "alpha_c = pi/(3 C_F) = 8 pi/45 = 0.5585; beta0 = 22/3 - 2N_f/3 = 6 at N_f = 2; d Lambda/d ln mu = 0 along "
       "the one-loop flow (symbolic)")

Omg = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
rows = []
basis_su4 = [1j * T for T in T4n]  # anti-Hermitian traceless
for X in basis_su4:
    Cx = X.T @ Omg + Omg @ X
    rows.append(np.concatenate([Cx.real.ravel(), Cx.imag.ravel()]))
rkk = np.linalg.matrix_rank(np.array(rows).T, tol=1e-10)
ok24 = (15 - rkk == 10)
report("d4-24", ok24, "dim{X in su(4): X^T Omega + Omega X = 0} = 10 (numerical rank), so SU(4)/Sp(4) has 5 Goldstone directions")

# ---------------------------------------------------------------------------------------------
# 25. rescaling and kappa = 0
ok25 = True
for trial in range(3):
    b = rand_so()
    kk = rng.uniform(0.3, 2)
    mm = rng.uniform(0.5, 2)
    s = rng.uniform(0.2, 5)
    Mf = lambda bb, k_, m_: np.block([[-(bb + k_ * G_f), np.eye(6) / m_], [np.zeros((6, 6)), -bb + k_ * G_f]])
    ok25 &= np.allclose(Mf(b / s, kk / s, s * mm), Mf(b, kk, mm) / s)
b = sum((rr() * B for B in so_basis), sp.zeros(6))
S_, D_, M_ = M_of(b, 0, G)
ok25 &= sp.expand(charpoly_QQ(M_).as_expr() - charpoly_QQ(-b).as_expr() ** 2) == 0
# m does not enter the spectrum of M: chi_M independent of m
S1, D1, M1 = M_of(b, 1, G, 1)
S2, D2, M2_ = M_of(b, 1, G, sp.Rational(7, 3))
ok25 &= sp.expand(charpoly_QQ(M1).as_expr() - charpoly_QQ(M2_).as_expr()) == 0
report("d4-25", ok25, "(tau,k,b,m) -> (s tau,k/s,b/s,s m) maps M to M/s (numeric); at k = 0 chi_M = chi_{-b}^2 (exact); the "
       "spectrum of M does not depend on m (so m*kappa, not m/kappa, is the rescaling invariant)")

print(f"# x4 checks finished in {time.time() - T0:.1f}s")
if FAILS:
    print("# FAILED:", ", ".join(FAILS))
    sys.exit(1)
sys.exit(0)
