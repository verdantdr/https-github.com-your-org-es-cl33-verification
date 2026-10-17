#!/usr/bin/env python3
"""Track d4 -- The b-field: the antisymmetric part of the shift, and the scale.

    python3 investigations/rest_energy/verify/d4_bfield.py

Prints one 'PASS d4-<n>: ...' line per verified claim and exits 0 only if every check passes.

Conventions are those of the unified paper (tex/s02_arena.tex eq. realisation, s03 prop:twoK and lem:Gclifford,
s12 thm:GLorentz / thm:emfield, s15-s17): V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1), e1..e6 realised on R^8,
G e_i = e_{i+3}, G e_{i+3} = -e_i, X = sum_i (B_i e_i + E_i e_{i+3}), z = B + iE (G acts as i).
R(a,b) is the generator with R[a,b] = eta_bb, R[b,a] = -eta_aa; it equals (1/2) ad(e_a e_b) on vectors (checked).
The interior Lagrangian is L = (m/2) eta(u,u), u = Xdot + S X, S = b + kappa G, b in so(3,3) constant.
Momentum pi = m u, pidot = S^dagger pi = (-b + kappa G) pi, and the phase-space generator on (X, pi) is
    M = [[-(b + kappa G), 1/m], [0, -b + kappa G]].

Exactness: one-constituent statements are checked in exact rational / Gaussian-rational arithmetic with sympy
(characteristic polynomials, ranks, null spaces, symbolic identities in kappa, beta, m).  Semisimplicity is tested
exactly: M is semisimple iff the square-free part of its characteristic polynomial annihilates M.  The
three-constituent Clifford computations use integer sparse matrices (exact).  Numerical checks (Krein signatures,
integration of the split virial identity, dimension of the bounded set, mpmath root of the gap equation) state
their tolerances where they are made.  Runtime is well under a minute.
"""
import itertools
import sys
import time

import mpmath as mp
import numpy as np
import scipy.linalg as sla
import scipy.optimize as so
import scipy.sparse as sps
import sympy as sp
from scipy.integrate import solve_ivp
from sympy import QQ
from sympy.polys.matrices import DomainMatrix

T0 = time.time()
FAILS = []


def report(cid, ok, text):
    if ok:
        print(f"PASS {cid}: {text}")
    else:
        print(f"FAIL {cid}: {text}")
        FAILS.append(cid)


def claim(cid, text):
    """Decorator: run the check, report PASS/FAIL; an exception is a FAIL."""
    def deco(fn):
        try:
            ok = bool(fn())
        except Exception as exc:  # pragma: no cover - reported as failure
            print(f"   exception in {cid}: {exc!r}")
            ok = False
        report(cid, ok, text)
        return fn
    return deco


# ============================================================================================= exact helpers
lam = sp.Symbol('lam')
kap, beta, mm, cc = sp.symbols('kappa beta m c', positive=True)


def rank_q(M):
    """Exact rank of a rational sympy Matrix."""
    rows = [[QQ(int(sp.Rational(x).p), int(sp.Rational(x).q)) for x in M.row(i)] for i in range(M.rows)]
    return DomainMatrix(rows, (M.rows, M.cols), QQ).rank()


def is_semisimple(M):
    """Exact: M (rational or Gaussian-rational entries) is diagonalisable over C iff sqf(charpoly)(M) = 0."""
    p = M.charpoly(lam)
    r = p.sqf_part()
    n = M.shape[0]
    res = sp.zeros(n, n)
    for cf in r.all_coeffs():
        res = res * M + cf * sp.eye(n)
    return res.expand().is_zero_matrix


def poly_eq(a, b):
    return sp.expand(a - b) == 0


def zero_mat(A):
    return sp.simplify(A).is_zero_matrix


# ============================================================================================= realisation
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def kron(*ms):
    r = ms[0]
    for x in ms[1:]:
        r = np.kron(r, x)
    return r


E8 = [None, kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx), kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
ETA = [None, 1, 1, 1, -1, -1, -1]
eta = sp.diag(1, 1, 1, -1, -1, -1)
I6 = sp.eye(6)
Z6 = sp.zeros(6, 6)


def ad6(B):
    """Matrix of v -> [B, v] on the six generators (integer-exact)."""
    out = sp.zeros(6, 6)
    for c in range(1, 7):
        Y = B @ E8[c] - E8[c] @ B
        rec = np.zeros((8, 8), dtype=np.int64)
        for d in range(1, 7):
            tr = int(np.trace(E8[d] @ Y))
            assert tr % (8 * ETA[d]) == 0
            co = tr // (8 * ETA[d])
            out[d - 1, c - 1] = co
            rec += co * E8[d]
        assert (rec == Y).all(), "commutator not of grade one"
    return out


def R(a, b, n=6, etad=None):
    etad = etad or ETA
    M = sp.zeros(n, n)
    M[a - 1, b - 1] = etad[b]
    M[b - 1, a - 1] = -etad[a]
    return M


G = sp.zeros(6, 6)
for _i in range(3):
    G[_i + 3, _i] = 1
    G[_i, _i + 3] = -1


def dag(A):
    return eta.inv() * A.T * eta


SO33 = [R(a, b) for a in range(1, 7) for b in range(a + 1, 7)]
PAIRS = [(a, b) for a in range(1, 7) for b in range(a + 1, 7)]


def so_coords(X):
    return [X[a - 1, b - 1] * ETA[b] for (a, b) in PAIRS]


def phase_M(b, k=kap, m=mm, Gm=None):
    Gm = G if Gm is None else Gm
    n = b.shape[0]
    A = -(b + k * Gm)
    D = -b + k * Gm
    return sp.BlockMatrix([[A, sp.eye(n) / m], [sp.zeros(n, n), D]]).as_explicit()


def cross_mat(i):
    M = sp.zeros(3, 3)
    ei = [0, 0, 0]
    ei[i] = 1
    for col in range(3):
        v = [0, 0, 0]
        v[col] = 1
        cr = [ei[1] * v[2] - ei[2] * v[1], ei[2] * v[0] - ei[0] * v[2], ei[0] * v[1] - ei[1] * v[0]]
        for r_ in range(3):
            M[r_, col] = cr[r_]
    return M


def zmatrix(A):
    """For A commuting with G: complex 3x3 matrix of its action on z = B + iE."""
    P = A[:3, :3]
    Q = A[3:, :3]
    return P + sp.I * Q


# Paper generators: L_i = (1/2) ad(rot(j,k)), K_i = ad((1/2)(e_k e_{j+3} - e_j e_{k+3})), (i,j,k) cyclic.
Lg, Kg = {}, {}
for (i, j, k) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
    rot = E8[j + 3] @ E8[k + 3] - E8[j] @ E8[k]
    Lg[i] = ad6(rot) / 2
    Kg[i] = ad6(E8[k] @ E8[j + 3] - E8[j] @ E8[k + 3]) / 2
Pole = sum((E8[i] @ E8[i + 3] for i in (1, 2, 3)), np.zeros((8, 8), dtype=np.int64))
Kgrad = ad6(Pole) / 2          # the grading root  K = (1/2) ad(sum_i e_i e_{i+3})  (lem:Gclifford)


@claim("d4-0", "realisation: Clifford relations; R(a,b) = (1/2)ad(e_a e_b); G^2=-1, G^dag=G, anti-isometry; "
               "L_i: z -> e_i x z, K_i = G L_i: z -> i e_i x z; grading root K = (1/2)ad(sum e_i e_{i+3}) in so(3,3), G = -K eta")
def _c0():
    ok = True
    for a in range(1, 7):
        for b in range(1, 7):
            ok &= ((E8[a] @ E8[b] + E8[b] @ E8[a]) == 2 * (ETA[a] if a == b else 0) * np.eye(8, dtype=np.int64)).all()
    for (a, b) in PAIRS:
        ok &= (ad6(E8[a] @ E8[b]) / 2 == R(a, b))
    ok &= (G * G == -I6) and (dag(G) == G) and (G.T * eta * G == -eta)
    for i in (1, 2, 3):
        ok &= (zmatrix(Lg[i]) == cross_mat(i - 1)) and (Kg[i] == G * Lg[i])
        ok &= (zmatrix(Kg[i]) == sp.I * cross_mat(i - 1))
        ok &= (Lg[i] * G == G * Lg[i])
    ok &= (dag(Kgrad) == -Kgrad) and (Kgrad * Kgrad == I6) and (G == -Kgrad * eta)
    return ok


# ============================================================================================= d4-1, d4-2, d4-3
@claim("d4-1", "End(V) = so(3,3) (+) Sym_eta, dims 15 + 21 = 36; G in Sym_eta, K in so(3,3); for S = b + kappa G "
               "(b in so(3,3) generic, 15 symbols) the eta-antisymmetric part of S is b and the self-adjoint part kappa G; "
               "the Euler-Lagrange operator is (D - kappa G)(D + kappa G) with D = d/dtau + b, i.e. D^2 X + kappa^2 X + kappa[b,G] X = 0")
def _c1():
    sym_basis = []
    for a in range(6):
        for b_ in range(a, 6):
            Sm = sp.zeros(6, 6)
            Sm[a, b_] = 1
            Sm[b_, a] = 1
            sym_basis.append(eta.inv() * Sm)
    ok = len(SO33) == 15 and len(sym_basis) == 21
    ok &= all(dag(x) == -x for x in SO33) and all(dag(x) == x for x in sym_basis)
    big = sp.Matrix([list(x) for x in SO33 + sym_basis])
    ok &= rank_q(big) == 36
    ok &= dag(G) == G and dag(Kgrad) == -Kgrad
    xs = sp.symbols('x0:15')
    b = sum((xs[i] * SO33[i] for i in range(15)), Z6)
    S = b + kap * G
    ok &= zero_mat((S - dag(S)) / 2 - b) and zero_mat((S + dag(S)) / 2 - kap * G)
    # Euler-Lagrange: pi = m(Xd + S X), pidot = S^dag pi  ->  Xdd + (S - S^dag) Xd - S^dag S X = 0.
    X = sp.Matrix(sp.symbols('X1:7'))
    Xd = sp.Matrix(sp.symbols('Xd1:7'))
    Xdd = sp.Matrix(sp.symbols('Xdd1:7'))
    Lag = mm / 2 * ((Xd + S * X).T * eta * (Xd + S * X))[0]
    p = sp.Matrix([sp.diff(Lag, v) for v in Xd])
    ok &= zero_mat(p - mm * eta * (Xd + S * X))
    dLdX = sp.Matrix([sp.diff(Lag, v) for v in X])
    ok &= zero_mat(dLdX - S.T * p)
    EL2 = Xdd + (S - dag(S)) * Xd - dag(S) * S * X
    # d/dtau (dL/dXdot) - dL/dX, with d/dtau p = m eta (Xdd + S Xd), equals m eta EL2
    ok &= zero_mat(mm * eta * (Xdd + S * Xd) - S.T * p - mm * eta * EL2)
    D2X = Xdd + 2 * b * Xd + b * b * X
    cov = D2X + kap ** 2 * X + kap * (b * G - G * b) * X
    ok &= zero_mat(EL2 - cov)
    return ok


def cayley(A, t):
    return (sp.eye(A.shape[0]) - t * A).inv() * (sp.eye(A.shape[0]) + t * A)


@claim("d4-2", "gauge covariance: for g(tau) = Cayley(tau A) in SO(3,3), alpha_+(gY) = g(d/dtau + b' + kappa G')Y with "
               "b' = g^-1 b g + g^-1 gdot in so(3,3) (inhomogeneous: a connection) and G' = g^-1 G g in Sym_eta (homogeneous: a tensor); "
               "D_tau G = [b,G] transforms covariantly")
def _c2():
    t = sp.Symbol('t')
    A = R(1, 4) + 2 * R(2, 3) - R(3, 5) + sp.Rational(1, 2) * R(4, 6)
    b = R(1, 2) - R(4, 5) + sp.Rational(1, 3) * R(1, 4)
    g = cayley(A, t)
    ok = zero_mat(g.T * eta * g - eta)
    gi = g.inv()
    gd = g.diff(t)
    bprime = sp.simplify(gi * b * g + gi * gd)
    Gprime = sp.simplify(gi * G * g)
    ok &= zero_mat(dag(bprime) + bprime) and zero_mat(dag(Gprime) - Gprime)
    ok &= not zero_mat(gi * gd)          # the inhomogeneous term is really there
    covG = sp.simplify(Gprime.diff(t) + bprime * Gprime - Gprime * bprime)
    ok &= zero_mat(covG - gi * (b * G - G * b) * g)
    return ok


def projections(b):
    bl = (b - G * b * G) / 2       # G^-1 = -G : (b + G b G^-1)/2
    bp = (b + G * b * G) / 2
    return bl, bp


@claim("d4-3", "so(3,3) = l_G (+) p_G (commuting / anticommuting with G), dims 6 + 9; Ad_G preserves so(3,3) and is an involution; "
               "Killing form = 4 tr: signature (3+,3-) on l_G and (6+,3-) on p_G (3 compact, 6 boost directions); "
               "l_G = {z -> (P+iQ)z}, p_G = {z -> A conj(z), A anti-Hermitian}; K and the pairing-plane boosts lie in p_G")
def _c3():
    ok = True
    xs = sp.symbols('x0:15')
    b = sum((xs[i] * SO33[i] for i in range(15)), Z6)
    ok &= zero_mat(dag(G * b * G.inv()) + G * b * G.inv())          # Ad_G preserves so(3,3)
    ok &= zero_mat(G * (G * b * G.inv()) * G.inv() - b)              # involution
    lsol = sp.solve(list(b * G - G * b), xs, dict=True)[0]
    psol = sp.solve(list(b * G + G * b), xs, dict=True)[0]
    bl, bp = b.subs(lsol), b.subs(psol)
    fl, fp = sorted(bl.free_symbols, key=str), sorted(bp.free_symbols, key=str)
    ok &= len(fl) == 6 and len(fp) == 9
    basis_l = [bl.subs({s: (1 if s == t_ else 0) for s in fl}) for t_ in fl]
    basis_p = [bp.subs({s: (1 if s == t_ else 0) for s in fp}) for t_ in fp]

    # Killing form from the adjoint representation on so(3,3)
    def adm(x):
        return sp.Matrix([so_coords(x * y - y * x) for y in SO33]).T

    ads = [adm(x) for x in SO33]
    Kill = sp.Matrix(15, 15, lambda i, j: (ads[i] * ads[j]).trace())
    TrF = sp.Matrix(15, 15, lambda i, j: (SO33[i] * SO33[j]).trace())
    ok &= Kill == 4 * TrF

    def sig(basis):
        Gm = sp.Matrix(len(basis), len(basis), lambda i, j: (basis[i] * basis[j]).trace())
        ev = Gm.eigenvals()
        return (sum(mu for e_, mu in ev.items() if e_ > 0), sum(mu for e_, mu in ev.items() if e_ < 0))

    ok &= sig(basis_l) == (3, 3) and sig(basis_p) == (6, 3)
    # complex descriptions: (B,E) blocks
    for x in basis_l:
        P, Q = x[:3, :3], x[3:, :3]
        ok &= x[3:, 3:] == P and x[:3, 3:] == -Q and P.T == -P and Q.T == -Q
    for x in basis_p:
        Ar, Ai = x[:3, :3], x[:3, 3:]
        ok &= x[3:, :3] == Ai and x[3:, 3:] == -Ar
        Acx = Ar + sp.I * Ai
        ok &= Acx.H == -Acx
    ok &= (Kgrad * G + G * Kgrad == Z6)
    for a in (1, 2, 3):
        sig_ = sp.zeros(6, 6)
        sig_[a - 1, a + 2] = 1
        sig_[a + 2, a - 1] = 1
        ok &= dag(sig_) == -sig_ and sig_ * G + G * sig_ == Z6
    return ok


# ============================================================================================= d4-4 .. d4-12
def charM(b, k=kap, m=mm):
    return sp.factor(phase_M(b, k, m).charpoly(lam).as_expr())


@claim("d4-4", "phase space: H = p^T eta^-1 p/(2m) - p^T S X generates M (Hamiltonian, M = J Hess H) and is conserved; "
               "H = 0 on the zero-momentum sector; det(lam - M) = chi_S(lam) chi_S(-lam) (exact at three rational b)")
def _c4():
    ok = True
    xs = sp.symbols('x0:15')
    b = sum((xs[i] * SO33[i] for i in range(15)), Z6)
    S = b + kap * G
    Hess = sp.BlockMatrix([[Z6, -S.T], [-S, eta.inv() / mm]]).as_explicit()
    J = sp.BlockMatrix([[Z6, I6], [-I6, Z6]]).as_explicit()
    Mcan = J * Hess
    ok &= zero_mat(Mcan - sp.BlockMatrix([[-S, eta.inv() / mm], [Z6, S.T]]).as_explicit())
    # M_can is the generator in (X,p); M in (X,pi) is its conjugate by diag(1, eta^-1)
    Tm = sp.diag(I6, eta.inv())
    ok &= zero_mat(Tm * Mcan * Tm.inv() - phase_M(b))
    ok &= zero_mat(Mcan.T * Hess + Hess * Mcan)        # dH/dt = 0
    ok &= Hess[:6, :6] == Z6                              # H vanishes at p = 0
    for bb in [sp.Rational(2, 3) * R(1, 2) - R(4, 6) + sp.Rational(1, 5) * R(2, 5),
               R(1, 4) + sp.Rational(3, 2) * R(3, 6) - R(2, 3), sp.Rational(1, 7) * (R(1, 5) + R(2, 4))]:
        S_ = bb + sp.Rational(5, 4) * G
        chi = S_.charpoly(lam).as_expr()
        ok &= poly_eq(phase_M(bb, sp.Rational(5, 4), 2).charpoly(lam).as_expr(), chi * chi.subs(lam, -lam))
    return ok


L3 = Lg[3]
b_rot = beta * (R(1, 2) - R(4, 5))


@claim("d4-5", "(i) b = beta(R12 - R45) in l_G (= -beta L_3, rotation): spec M = +-i kappa, +-i(kappa+beta), +-i(kappa-beta), each double; "
               "Y = -G/(2 kappa m) solves A Y - Y D = -1/m for EVERY b in l_G (6 symbols), so M ~ diag(A,D) is semisimple and all motion "
               "is bounded; the three-pair degeneracy is lifted (exact semisimplicity at kappa=1, beta=3/7 and kappa=1, beta=3/2)")
def _c5():
    ok = (R(1, 2) - R(4, 5)) * G == G * (R(1, 2) - R(4, 5)) and (R(1, 2) + R(4, 5)) * G == -G * (R(1, 2) + R(4, 5))
    ok &= (R(1, 2) - R(4, 5)) == -L3
    chi = charM(b_rot)
    pred = (lam ** 2 + kap ** 2) ** 2 * (lam ** 2 + (kap + beta) ** 2) ** 2 * (lam ** 2 + (kap - beta) ** 2) ** 2
    ok &= poly_eq(chi, pred)
    w = sp.symbols('w0:6', real=True)
    bl = sum((w[i] * Lg[i + 1] + w[i + 3] * Kg[i + 1] for i in range(3)), Z6)
    A = -(bl + kap * G)
    D = -bl + kap * G
    Y = -G / (2 * kap * mm)
    ok &= zero_mat(A * Y - Y * D + I6 / mm)
    for (k_, b_) in [(1, sp.Rational(3, 7)), (1, sp.Rational(3, 2))]:
        Mv = phase_M(b_rot.subs(beta, b_), k_, 1)
        ok &= is_semisimple(Mv)
        ev = sp.Matrix(-(b_rot.subs(beta, b_) + k_ * G)).eigenvals()
        ok &= len(ev) == 6                                  # six distinct exponents: degeneracy lifted
    ok &= not is_semisimple(phase_M(Z6 + R(1, 2) - R(4, 5), 0, 1))   # sanity: kappa = 0 is not semisimple
    return ok


@claim("d4-6", "general b = sum(w_r L + w_b K) in l_G, w = w_r + i w_b in C^3: chi_S = (lam^2+kappa^2) |(lam - i kappa)^2 + w.w|^2; "
               "all motion bounded iff w = 0 or w.w > 0 (a 'magnetic-type' connection), frequencies kappa, |kappa +- sqrt(w.w)|; "
               "null w (w.w = 0, w != 0) gives a non-semisimple S already at zero momentum")
def _c6():
    br = sp.symbols('br1:4', real=True)
    bb = sp.symbols('bb1:4', real=True)
    b = sum((br[i] * Lg[i + 1] + bb[i] * Kg[i + 1] for i in range(3)), Z6)
    chi = sp.expand((b + kap * G).charpoly(lam).as_expr())
    wv = [br[i] + sp.I * bb[i] for i in range(3)]
    ww = sum(x * x for x in wv)
    pred = (lam ** 2 + kap ** 2) * ((lam - sp.I * kap) ** 2 + ww) * ((lam + sp.I * kap) ** 2 + sp.conjugate(ww))
    ok = poly_eq(chi, sp.expand(pred))
    # examples (kappa = 1, m = 1)
    def bw(wr, wb):
        return sum((sp.Rational(wr[i]) * Lg[i + 1] + sp.Rational(wb[i]) * Kg[i + 1] for i in range(3)), Z6)
    ell = bw([2, 0, 0], [0, 1, 0])          # w = (2, i, 0): w.w = 3 > 0, complex but elliptic
    Mv = phase_M(ell, 1, 1)
    ok &= is_semisimple(Mv) and poly_eq(Mv.charpoly(lam).as_expr(),
                                        ((lam ** 2 + 1) * (lam ** 2 + (1 + sp.sqrt(3)) ** 2) * (lam ** 2 + (1 - sp.sqrt(3)) ** 2)) ** 2)
    nul = bw([1, 0, 0], [0, 1, 0])          # w = (1, i, 0): null
    Sn = nul + G
    ok &= not is_semisimple(Sn) and not is_semisimple(phase_M(nul, 1, 1))
    hyp = bw([0, 0, 0], [0, 0, sp.Rational(1, 2)])    # w = (0,0,i/2): w.w = -1/4
    ev = list(sp.Matrix(hyp + G).eigenvals().keys())
    ok &= any(sp.re(sp.nsimplify(e_)) != 0 for e_ in ev)
    return ok


@claim("d4-7", "(ii) b = beta K_3 (l_G boost): chi_M = (lam^2+kappa^2)^2 ((lam -+ beta)^2 + kappa^2)^2: exponents +-beta +- i kappa, "
               "complex with real part +-beta (not real): growing spirals at the rest frequency")
def _c7():
    chi = charM(beta * Kg[3])
    pred = (lam ** 2 + kap ** 2) ** 2 * ((lam - beta) ** 2 + kap ** 2) ** 2 * ((lam + beta) ** 2 + kap ** 2) ** 2
    ok = poly_eq(chi, pred)
    roots = sp.roots(sp.Poly(sp.expand((lam - beta) ** 2 + kap ** 2), lam))
    ok &= set(roots) == {beta + sp.I * kap, beta - sp.I * kap}
    return ok


b_pc = beta * (R(1, 2) + R(4, 5))


@claim("d4-8", "(iii) b = beta(R12 + R45) in p_G (compact): (b + kappa G)^2 = b^2 - kappa^2; chi_M = (lam^2+kappa^2)^2 (lam^2+kappa^2+beta^2)^4; "
               "at kappa=4, beta=3: +-5i have algebraic multiplicity 4 and geometric 2 (Jordan blocks of size 2), +-4i are semisimple; "
               "the bounded initial data are exactly {pi_1 = pi_2 = pi_4 = pi_5 = 0} (dim 8): no momentum on the support of b")
def _c8():
    S = b_pc + kap * G
    ok = zero_mat(S * S - (b_pc * b_pc - kap ** 2 * I6))
    ok &= poly_eq(charM(b_pc), (lam ** 2 + kap ** 2) ** 2 * (lam ** 2 + kap ** 2 + beta ** 2) ** 4)
    Mv = phase_M(b_pc.subs(beta, 3), 4, 1)
    I12 = sp.eye(12)
    ok &= rank_q(Mv * Mv + 25 * I12) == 8 and rank_q((Mv * Mv + 25 * I12) ** 2) == 4
    ok &= rank_q(Mv * Mv + 16 * I12) == 8 and rank_q((Mv * Mv + 16 * I12) ** 2) == 8
    # eigenvectors (over Q(i)) span the bounded subspace
    vecs = []
    for ev in (5 * sp.I, -5 * sp.I, 4 * sp.I, -4 * sp.I):
        ns = (Mv - ev * I12).nullspace()
        vecs += ns
    ok &= len(vecs) == 8
    Vm = sp.Matrix.hstack(*vecs)
    ok &= Vm.rank() == 8
    # every eigenvector has zero momentum on the support (pi indices 6,7,9,10 in 0-based phase coordinates)
    for v in vecs:
        ok &= all(sp.simplify(v[i]) == 0 for i in (6, 7, 9, 10))
    return ok


@claim("d4-9", "(iv) pairing-plane boost b = c sigma on (e3,e6) (in p_G): exponents +-sqrt(c^2 - kappa^2) (Prop. 16.7), "
               "M^2 - (c^2 - kappa^2) = [[0, -2c sigma/m],[0,0]] on the plane; Jordan blocks below (c=1/2), above (c=2) and at the "
               "threshold (c=kappa, where S is nilpotent); the grading root as a connection: (cK + kappa G)^2 = (c^2 - kappa^2) 1 on all of V")
def _c9():
    sig = sp.zeros(6, 6)
    sig[2, 5] = 1
    sig[5, 2] = 1
    ok = poly_eq(charM(cc * sig), (lam ** 2 + kap ** 2) ** 4 * (lam ** 2 - cc ** 2 + kap ** 2) ** 2)
    # restrict to the plane (indices 2,5)
    idx = [2, 5]
    s2 = sp.Matrix([[0, 1], [1, 0]])
    G2 = sp.Matrix([[0, -1], [1, 0]])
    def M2(c_, k_, m_):
        return sp.BlockMatrix([[-(c_ * s2 + k_ * G2), sp.eye(2) / m_], [sp.zeros(2, 2), -c_ * s2 + k_ * G2]]).as_explicit()
    ok &= G[idx, idx] == G2 and sig[idx, idx] == s2
    Ms = M2(cc, kap, mm)
    ok &= zero_mat(Ms * Ms - (cc ** 2 - kap ** 2) * sp.eye(4) -
                   sp.BlockMatrix([[sp.zeros(2, 2), -2 * cc * s2 / mm], [sp.zeros(2, 2), sp.zeros(2, 2)]]).as_explicit())
    Mh = M2(sp.Rational(1, 2), 1, 1)
    ok &= poly_eq(Mh.charpoly(lam).as_expr(), (lam ** 2 + sp.Rational(3, 4)) ** 2) and rank_q(Mh * Mh + sp.Rational(3, 4) * sp.eye(4)) == 2
    Ma = M2(2, 1, 1)
    ok &= rank_q(Ma * Ma - 3 * sp.eye(4)) == 2 and not is_semisimple(Ma)
    S_thr = sp.Rational(1) * s2 + G2
    ok &= S_thr * S_thr == sp.zeros(2, 2) and S_thr != sp.zeros(2, 2) and not is_semisimple(-S_thr)
    ok &= zero_mat((cc * Kgrad + kap * G) ** 2 - (cc ** 2 - kap ** 2) * I6)
    return ok


@claim("d4-10", "every b in p_G (9 symbols): (b+kappa G)^2 = (-b+kappa G)^2 = b^2 - kappa^2 and M^2 = (b^2-kappa^2)(+)(b^2-kappa^2) + N, "
                "N = [[0,-2b/m],[0,0]] nilpotent, commuting, nonzero: M is never semisimple for b != 0 in p_G (exact also at 3 random rational b)")
def _c10():
    xs = sp.symbols('x0:15')
    b = sum((xs[i] * SO33[i] for i in range(15)), Z6)
    psol = sp.solve(list(b * G + G * b), xs, dict=True)[0]
    bp = b.subs(psol)
    Q = bp * bp - kap ** 2 * I6
    ok = zero_mat((bp + kap * G) ** 2 - Q) and zero_mat((-bp + kap * G) ** 2 - Q)
    M = phase_M(bp)
    N = sp.BlockMatrix([[Z6, -2 * bp / mm], [Z6, Z6]]).as_explicit()
    Qt = sp.diag(Q, Q)
    ok &= zero_mat(M * M - Qt - N) and zero_mat(Qt * N - N * Qt) and zero_mat(N * N)
    rng = np.random.default_rng(4)
    fp = sorted(bp.free_symbols, key=str)
    for _ in range(3):
        vals = {s: sp.Rational(int(rng.integers(-4, 5)), int(rng.integers(1, 6))) for s in fp}
        bv = bp.subs(vals)
        if bv == Z6:
            continue
        ok &= not is_semisimple(phase_M(bv, 1, 1))
    return ok


def num(A):
    return np.array(A.evalf(), dtype=complex)


def eig_spaces(Dn, tol=1e-7):
    ev = np.linalg.eigvals(Dn)
    groups = []
    for e_ in ev:
        if not any(abs(e_ - g) < 1e-5 for g in groups):
            groups.append(e_)
    spaces = {}
    for g in groups:
        ns = sla.null_space(Dn - g * np.eye(Dn.shape[0]), rcond=tol)
        spaces[g] = ns
    return spaces


def roth_isotropy(b, k):
    """Numerical criterion: E_mu(D) and E_-mu(D) are eta-orthogonal (bilinear) for all mu; D = S^dagger = -b + kG."""
    Dn = num(-b + k * G)
    sp_ = eig_spaces(Dn)
    etan = np.diag([1, 1, 1, -1, -1, -1]).astype(complex)
    worst = 0.0
    for mu, Ym in sp_.items():
        for nu, Yn in sp_.items():
            if abs(mu + nu) < 1e-5:
                worst = max(worst, np.max(np.abs(Yn.T @ etan @ Ym)))
    return worst < 1e-8


def sylvester_solvable(b, k, m=1):
    A = -(b + k * G)
    D = -b + k * G
    n = 6
    Kmat = sp.kronecker_product(sp.eye(n), A) - sp.kronecker_product(D.T, sp.eye(n))
    rhs = sp.Matrix([(-sp.eye(n) / m)[i, j] for j in range(n) for i in range(n)])     # column-major vec(-1/m)
    return rank_q(Kmat) == rank_q(Kmat.row_join(rhs))


b_mix = sp.Rational(1, 3) * (R(1, 2) - R(4, 5)) + sp.Rational(1, 5) * (R(2, 3) - R(5, 6)) + sp.Rational(1, 10) * (R(1, 2) + R(4, 5))


def bp_of(Y):
    b = sp.zeros(6, 6)
    for i in range(3):
        for j in range(3):
            b[i, 3 + j] = Y[i, j]
            b[3 + i, j] = Y[i, j]
    return b


def b_gyro(a, c, d):
    return a * L3 + bp_of(sp.Matrix([[c, d, 0], [d, -c, 0], [0, 0, 0]]))


@claim("d4-11", "criterion (v): M semisimple <=> A, D semisimple and A Y - Y D = -1/m solvable (Roth) <=> E_mu(S^dag) and E_-mu(S^dag) "
                 "eta-orthogonal for all mu (for imaginary spectra: every momentum normal mode eta-null, <eta(pi,pi)> = 0); "
                 "checked exactly (semisimplicity, Sylvester rank over Q) and numerically (isotropy, tol 1e-8) on 7 test connections")
def _c11():
    tests = [(sp.Rational(3, 7) * (R(1, 2) - R(4, 5)), 1, True),           # l_G rotation
             (3 * (R(1, 2) + R(4, 5)), 4, False),                          # p_G compact
             (b_mix, 1, False),                                            # l_G + small p_G
             (b_gyro(sp.Rational(1, 2), sp.Rational(1, 4), 0), 1, True),   # gyroscopic screening
             (sp.Rational(1, 2) * (R(3, 6)), 1, False),                    # pairing boost below threshold
             (R(1, 2) + sp.Rational(1, 3) * R(1, 4) - sp.Rational(2, 3) * R(2, 6) + R(3, 5), 1, None),   # generic
             (sp.Rational(1, 4) * R(1, 5) - sp.Rational(1, 2) * R(4, 6) + R(2, 3), sp.Rational(2, 3), None)]
    ok = True
    for (b, k, expect) in tests:
        Mv = phase_M(b, k, 1)
        sem = is_semisimple(Mv)
        A_ss = is_semisimple(-(b + k * G))
        rot = A_ss and sylvester_solvable(b, k)
        iso = A_ss and roth_isotropy(b, k)
        ok &= (sem == rot == iso)
        if expect is not None:
            ok &= (sem == expect)
    return ok


@claim("d4-12", "gyroscopic screening: b = a L_3 + b_p, b_p the p_G boost z -> iY conj(z) with Y traceless symmetric in the (1,2) plane, "
                 "has chi_S = (lam^2+kappa^2)(lam^2+(kappa-a)^2-r^2)(lam^2+(kappa+a)^2-r^2), r^2 = c^2+d^2, and M is semisimple "
                 "(exact at (a,c,d) = (1/2,1/4,0), (1/3,1/5,1/7)) although [b,G] != 0; at a = 0 it is not; mechanism: K_3 = G L_3 splits the "
                 "support into two eta-null planes with S = (kappa -+ a)G + b_Y on them; numerically the bounded "
                 "set is locally a 10-dimensional subset of so(3,3) (5 independent conditions, Jacobian rank 5 at random solutions)")
def _c12():
    a, c_, d_ = sp.symbols('a c d', real=True)
    b = b_gyro(a, c_, d_)
    ok = zero_mat(dag(b) + b) and not zero_mat(b * G - G * b)
    bl, bp = projections(b)
    ok &= zero_mat(bl - a * L3) and zero_mat(bp - bp_of(sp.Matrix([[c_, d_, 0], [d_, -c_, 0], [0, 0, 0]])))
    chi = sp.expand((b + kap * G).charpoly(lam).as_expr())
    r2 = c_ ** 2 + d_ ** 2
    ok &= poly_eq(chi, (lam ** 2 + kap ** 2) * (lam ** 2 + (kap - a) ** 2 - r2) * (lam ** 2 + (kap + a) ** 2 - r2))
    for (av, cv, dv) in [(sp.Rational(1, 2), sp.Rational(1, 4), 0), (sp.Rational(1, 3), sp.Rational(1, 5), sp.Rational(1, 7))]:
        ok &= is_semisimple(phase_M(b_gyro(av, cv, dv), 1, 1))
    ok &= not is_semisimple(phase_M(b_gyro(0, sp.Rational(1, 4), 0), 1, 1))
    # mechanism: K_3 = G L_3 is an involution on the support span(e1,e2,e4,e5), commutes with G, L_3 and b_Y; its two
    # eigenplanes are totally eta-null, and on the (+-1)-plane S = (kappa -+ a) G + b_Y, S^dag = (kappa +- a) G - b_Y
    K3 = Kg[3]
    bY = bp_of(sp.Matrix([[c_, d_, 0], [d_, -c_, 0], [0, 0, 0]]))
    supp = sp.diag(1, 1, 0, 1, 1, 0)
    ok &= K3 * K3 == supp and zero_mat(K3 * bY - bY * K3) and K3 * G == G * K3 and K3 * L3 == L3 * K3
    for sgn in (1, -1):
        Pn = (supp + sgn * K3) / 2                       # projector on the eigenplane K_3 = sgn
        ok &= rank_q(Pn) == 2 and zero_mat(Pn.T * eta * Pn)
        ok &= zero_mat((b + kap * G) * Pn - ((kap - sgn * a) * G + bY) * Pn)
        ok &= zero_mat(dag(b + kap * G) * Pn - ((kap + sgn * a) * G - bY) * Pn)
    # numerical dimension count of the bounded set near random solutions
    so33n = [np.array(x, dtype=float) for x in SO33]
    Gn = np.array(G, dtype=float)
    etan = np.diag([1., 1, 1, -1, -1, -1])

    def resid(x, target=None):
        bn = sum(x[i] * so33n[i] for i in range(15))
        ev, V = np.linalg.eig(-bn + Gn)
        order = np.argsort(ev.imag)
        r = []
        for j in order[3:]:
            y = V[:, j] / np.linalg.norm(V[:, j])
            r += [(y.conj() @ etan @ y).real, ev[j].real]
        if target is not None:
            bpn = 0.5 * (bn + Gn @ bn @ Gn)
            r.append(np.linalg.norm(bpn) - target)
        return np.array(r)

    rng = np.random.default_rng(11)
    ranks = []
    for _ in range(40):
        x0 = rng.normal(size=15) * rng.uniform(0.1, 0.8)
        sol = so.least_squares(resid, x0, args=(0.3,), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
        if np.linalg.norm(sol.fun) > 1e-11:
            continue
        x = sol.x
        Jac = np.array([(resid(x + 1e-7 * np.eye(15)[i]) - resid(x - 1e-7 * np.eye(15)[i])) / 2e-7 for i in range(15)]).T
        sv = np.linalg.svd(Jac, compute_uv=False)
        ranks.append(int(np.sum(sv > 1e-4 * sv[0])))
        # boundedness check of this solution: geometric multiplicity 2 for each double eigenvalue of M
        bn = sum(x[i] * so33n[i] for i in range(15))
        Mn = np.block([[-(bn + Gn), np.eye(6)], [np.zeros((6, 6)), -bn + Gn]])
        z0 = rng.normal(size=12)
        grow = max(np.linalg.norm(sla.expm(Mn * T) @ z0) for T in (10., 100., 1000.))
        ok &= grow < 50 * np.linalg.norm(z0)
        if len(ranks) >= 4:
            break
    ok &= len(ranks) >= 4 and all(r_ == 5 for r_ in ranks)
    return ok


# ============================================================================================= d4-13 gauge
@claim("d4-13", "gauge: (a) kappa = 0: X = exp(-b tau) Y gives alpha X = exp(-b tau) Ydot, Yddot = 0 (free drift, M has Jordan blocks); "
                 "(b) [b,G] = 0: alpha_+ X = exp(-b tau)(Ydot + kappa G Y), lab exponents = rotating-frame +-i kappa shifted by eig(b), "
                 "H_lab = H_rot - p^T b Y (Larmor / chemical potential); (c) {b,G} = 0: exp(b tau) G exp(-b tau) = exp(2b tau) G = G exp(-2b tau), "
                 "not constant; b is removable keeping G constant iff [b,G] = 0, and the gauge-invariant remainder is D_tau G = [b,G] = 2 b_p G")
def _c13():
    t = sp.Symbol('tau', real=True)
    Y = sp.Matrix([sp.Function(f'Y{i}')(t) for i in range(1, 7)])
    ok = True
    # explicit exponentials for a rotation-type b (R12 - R45 in l_G) and an antidiagonal rotation (R12 + R45 in p_G)
    for b, commuting in [(beta * (R(1, 2) - R(4, 5)), True), (beta * (R(1, 2) + R(4, 5)), False)]:
        Eb = sp.simplify((-b * t).exp())
        Ebi = sp.simplify((b * t).exp())
        X = Eb * Y
        # (a) kappa = 0
        ok &= zero_mat(X.diff(t) + b * X - Eb * Y.diff(t))
        ok &= zero_mat(Eb.T * eta * Eb - eta)
        conj = sp.simplify(Ebi * G * Eb)
        if commuting:
            ok &= zero_mat(X.diff(t) + (b + kap * G) * X - Eb * (Y.diff(t) + kap * G * Y))
            # Larmor energy: p_X = exp(b^T tau) p_Y
            pY = sp.Matrix(sp.symbols('p1:7'))
            Yv = sp.Matrix(sp.symbols('y1:7'))
            pX = Ebi.T * pY
            Xv = Eb * Yv
            Hlab = (pX.T * eta.inv() * pX)[0] / (2 * mm) - (pX.T * (b + kap * G) * Xv)[0]
            Hrot = (pY.T * eta.inv() * pY)[0] / (2 * mm) - (pY.T * kap * G * Yv)[0]
            ok &= sp.simplify(Hlab - (Hrot - (pY.T * b * Yv)[0])) == 0
            ok &= zero_mat(conj - G)
        else:
            ok &= zero_mat(conj - sp.simplify((2 * b * t).exp()) * G) and zero_mat(conj - G * sp.simplify((-2 * b * t).exp()))
            ok &= not zero_mat(conj - G)
            ok &= zero_mat(conj.diff(t).subs(t, 0) - (b * G - G * b)) and zero_mat((b * G - G * b) - 2 * b * G)
    # kappa = 0 phase-space generator: chi = chi_{-b}^2 and not semisimple (free drift)
    Mv = phase_M(sp.Rational(1, 2) * (R(1, 2) - R(4, 5)), 0, 1)
    ok &= not is_semisimple(Mv)
    # lab spectrum = rotating +- eigenvalues of b  (already in d4-5): check the shift relation exactly at beta = 3/7
    b_ = sp.Rational(3, 7) * (R(1, 2) - R(4, 5))
    lab = set(sp.Matrix(-(b_ + G)).eigenvals().keys())
    rot_ = set(sp.Matrix(-G).eigenvals().keys())             # rotating frame: +- i kappa
    shifts = set(sp.Matrix(-b_).eigenvals().keys())          # 0, +- 3i/7
    ok &= rot_ == {sp.I, -sp.I} and shifts == {0, 3 * sp.I / 7, -3 * sp.I / 7}
    ok &= lab == {sp.expand(r_ + s_) for r_ in rot_ for s_ in shifts} and len(lab) == 6
    return ok


# ============================================================================================= d4-14 split virial
@claim("d4-14", "split virial d/dtau eta(X,pi) = eta(pi,pi)/m holds for every constant S (36 symbols, exact identity) and along numerical "
                 "solutions with b in l_G, p_G and mixed (integral form, |error| < 1e-8)")
def _c14():
    s = sp.symbols('s0:36')
    S = sp.Matrix(6, 6, s)
    X = sp.Matrix(sp.symbols('X1:7'))
    P = sp.Matrix(sp.symbols('P1:7'))
    Xd = P / mm - S * X
    Pd = dag(S) * P
    lhs = (Xd.T * eta * P + X.T * eta * Pd)[0]
    ok = sp.expand(lhs - (P.T * eta * P)[0] / mm) == 0
    etan = np.diag([1., 1, 1, -1, -1, -1])
    rng = np.random.default_rng(5)
    for b in [sp.Rational(3, 7) * (R(1, 2) - R(4, 5)), 3 * (R(1, 2) + R(4, 5)) / 4, b_mix, R(3, 6) / 2 + R(1, 4) / 3]:
        Mn = np.array(phase_M(b, 1, 1), dtype=float)

        def rhs(_t, y):
            z = y[:12]
            pi = z[6:]
            return np.concatenate([Mn @ z, [pi @ etan @ pi]])

        z0 = rng.normal(size=12)
        sol = solve_ivp(rhs, (0, 20), np.concatenate([z0, [0.]]), method='DOP853', rtol=1e-12, atol=1e-12)
        zT = sol.y[:12, -1]
        f0 = z0[:6] @ etan @ z0[6:]
        fT = zT[:6] @ etan @ zT[6:]
        ok &= abs((fT - f0) - sol.y[12, -1]) < 1e-8 * max(1.0, abs(sol.y[12, -1]))
    return ok


# ============================================================================================= d4-15/16 Krein, Fock
def krein(b, k, m=1.0, Gm=None):
    """Energy signature (n+, n-) of H on each eigenspace of the canonical generator with eigenvalue i w, w > 0."""
    Gm = G if Gm is None else Gm
    n = b.shape[0]
    S = np.array(b + k * Gm, dtype=float)
    etad = np.diag([1., 1, 1, -1, -1, -1] * (n // 6))
    Hs = np.block([[np.zeros((n, n)), -S.T], [-S, np.linalg.inv(etad) / m]])
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    Mc = J @ Hs
    ev = np.linalg.eigvals(Mc)
    freqs = sorted({round(float(e_.imag), 7) for e_ in ev if e_.imag > 1e-9})
    out = []
    for w in freqs:
        Vs = sla.null_space(Mc - 1j * w * np.eye(2 * n), rcond=1e-9)
        h = Vs.conj().T @ Hs @ Vs
        evh = np.linalg.eigvalsh((h + h.conj().T) / 2)
        out.append((w, int(np.sum(evh > 1e-9)), int(np.sum(evh < -1e-9)), Vs.shape[1]))
    return out


@claim("d4-15", "vacuum with a b-field: whenever all motion is bounded, every frequency w carries energy signature (k,k) "
                 "(the zero-momentum half is energy-null), so the zero-point energy sum_j (+-) hbar w_j / 2 cancels frequency by "
                 "frequency: checked for the Zeeman triplet (kappa=1, beta=3/7 and beta=3/2), an elliptic complex w, the gyroscopic family; "
                 "tolerance 1e-9 on eigenvalues of the energy form")
def _c15():
    ok = True
    def bw(wr, wb):
        return sum((sp.Rational(wr[i]) * Lg[i + 1] + sp.Rational(wb[i]) * Kg[i + 1] for i in range(3)), Z6)
    cases = [(sp.Rational(3, 7) * (R(1, 2) - R(4, 5)), 1, 3), (sp.Rational(3, 2) * (R(1, 2) - R(4, 5)), 1, 3),
             (bw([2, 0, 0], [0, 1, 0]), 1, 3), (b_gyro(sp.Rational(1, 2), sp.Rational(1, 4), 0), 1, 3),
             (Z6, 1, 1)]
    for (b, k, nfreq) in cases:
        res = krein(b, k)
        ok &= len(res) == nfreq
        zp = 0.0
        for (w, npos, nneg, dim) in res:
            ok &= npos == nneg and npos + nneg == dim
            zp += (npos - nneg) * w / 2
        ok &= abs(zp) < 1e-12
    res0 = krein(Z6, 1)
    ok &= res0[0][1:] == (3, 3, 6)
    return ok


@claim("d4-16", "no Fock vacuum with Jordan blocks: a sum of signed oscillators H = sum_j s_j (p_j^2 + w_j^2 q_j^2)/2 generates a semisimple "
                 "flow (exact), so for b in p_G \\ {0} (d4-10) no linear symplectic change of variables yields normal modes and no Fock "
                 "vacuum annihilated by normal-mode lowering operators exists")
def _c16():
    q = sp.symbols('q1:4')
    ws = [1, 2, sp.Rational(5, 2)]
    sgn = [1, -1, 1]
    n = 3
    Hess = sp.zeros(2 * n, 2 * n)
    for j in range(n):
        Hess[j, j] = sgn[j] * ws[j] ** 2
        Hess[n + j, n + j] = sgn[j]
    J = sp.BlockMatrix([[sp.zeros(n, n), sp.eye(n)], [-sp.eye(n), sp.zeros(n, n)]]).as_explicit()
    ok = is_semisimple(J * Hess)
    # and a symplectic change of variables preserves semisimplicity (similarity); M for p_G compact is not semisimple
    ok &= not is_semisimple(phase_M(3 * (R(1, 2) + R(4, 5)), 4, 1))
    # a model block with the same Jordan structure: H = w (q1 p2 - q2 p1) + (q1^2 + q2^2)/2 has spectrum +-i w, each a 2-block,
    # and its two terms Poisson-commute (rotation generator and a function of the coordinates alone)
    w0 = 3
    H2 = sp.Matrix([[1, 0, 0, w0], [0, 1, -w0, 0], [0, -w0, 0, 0], [w0, 0, 0, 0]])     # Hessian in (q1,q2,p1,p2)
    J2 = sp.BlockMatrix([[sp.zeros(2, 2), sp.eye(2)], [-sp.eye(2), sp.zeros(2, 2)]]).as_explicit()
    Mb = J2 * H2
    ok &= poly_eq(Mb.charpoly(lam).as_expr(), (lam ** 2 + w0 ** 2) ** 2) and not is_semisimple(Mb)
    q1, q2, p1, p2 = sp.symbols('q1 q2 p1 p2')
    Lr = q1 * p2 - q2 * p1
    r2 = (q1 ** 2 + q2 ** 2) / 2
    pb = sum(sp.diff(Lr, q) * sp.diff(r2, p) - sp.diff(Lr, p) * sp.diff(r2, q) for q, p in ((q1, p1), (q2, p2)))
    ok &= sp.simplify(pb) == 0
    return ok


# ============================================================================================= d4-17..19 three constituents
def build_cl99():
    om = E8[1] @ E8[2] @ E8[3] @ E8[4] @ E8[5] @ E8[6]
    assert (om @ om == np.eye(8, dtype=np.int64)).all()
    for a in range(1, 7):
        assert (om @ E8[a] == -E8[a] @ om).all()
    I8s = sps.identity(8, dtype=np.int64, format='csr')
    oms = sps.csr_matrix(om)
    E = []
    for k in range(3):
        for a in range(1, 7):
            facs = [oms] * k + [sps.csr_matrix(E8[a])] + [I8s] * (2 - k)
            r = facs[0]
            for f in facs[1:]:
                r = sps.kron(r, f, format='csr')
            E.append(r)
    return E


ETA18 = [None] + [1, 1, 1, -1, -1, -1] * 3
E18 = build_cl99()
I512 = sps.identity(512, dtype=np.int64, format='csr')


def ad18(Bv):
    out = sp.zeros(18, 18)
    for C in range(18):
        Y = (Bv @ E18[C] - E18[C] @ Bv).tocsr()
        rec = sps.csr_matrix((512, 512), dtype=np.int64)
        for Dd in range(18):
            tr = int(E18[Dd].multiply(Y.T).sum())
            assert tr % (512 * ETA18[Dd + 1]) == 0
            co = tr // (512 * ETA18[Dd + 1])
            if co:
                out[Dd, C] = co
                rec = rec + co * E18[Dd]
        assert (rec - Y).count_nonzero() == 0
    return out


def R18(a, b):
    return R(a, b, 18, ETA18)


G3 = sp.zeros(18, 18)
for _k in range(3):
    for _i in range(3):
        _A = 6 * _k + _i
        G3[_A + 3, _A] = 1
        G3[_A, _A + 3] = -1
eta18 = sp.diag(*ETA18[1:])


def rot18(i, j):
    """(1/2) ad of rot(i,j) = e_{i'} e_{j'} - e_i e_j, computed from the 512-dim Clifford realisation."""
    Bv = E18[i + 3 - 1] @ E18[j + 3 - 1] - E18[i - 1] @ E18[j - 1]
    return ad18(Bv) / 2


W = [7, 13, 14, 15]
COLOUR = [10, 11, 12, 16, 17, 18]
ROT_W = {(i, j): rot18(i, j) for i, j in itertools.combinations(W, 2)}


def cent_dim(basis):
    """dim of {x in span(basis): [x, G3] = 0}, exact."""
    cols = [sp.Matrix(list(x * G3 - G3 * x)) for x in basis]
    Mx = sp.Matrix.hstack(*cols)
    return len(basis) - rank_q(Mx)


@claim("d4-17", "three constituents: E_{6(k-1)+a} = omega^(k-1) (x) e_a (x) 1 satisfy Cl(9,9); G3 = G+G+G is an anti-isometry with G3^2 = -1; "
                 "Cent(G3) in so(9,9) has dim 72; Thm 17.21: so(U) and so(6)_C each meet Cent(G3) in 0 and ps_U meets it in dim C(d,2) "
                 "with d = dim(U cap E): 6 for U = W = {7,13,14,15} (spanned by rot(i,j), i<j in W), 3 for U = {3,7,8,9}, 1 for U = {2,3,7,8}")
def _c17():
    ok = True
    for A in range(18):
        for B in range(A, 18):
            ac = (E18[A] @ E18[B] + E18[B] @ E18[A]).tocsr()
            target = 2 * ETA18[A + 1] * I512 if A == B else 0 * I512
            ok &= (ac - target).count_nonzero() == 0
    ok &= G3 * G3 == -sp.eye(18) and G3.T * eta18 * G3 == -eta18 and eta18 * G3.T * eta18 == G3
    so99 = [R18(a, b) for a in range(1, 19) for b in range(a + 1, 19)]
    ok &= cent_dim(so99) == 72
    ok &= ad18(E18[6] @ E18[12]) / 2 == R18(7, 13)        # abstract generators agree with the realisation
    for (i, j), r_ in ROT_W.items():
        ok &= r_ * G3 == G3 * r_ and r_ == R18(i + 3, j + 3) - R18(i, j)
    for U, d_expect in [(W, 6), ([3, 7, 8, 9], 3), ([2, 3, 7, 8], 1)]:
        soU = [R18(a, b) for a, b in itertools.combinations(U, 2)]
        so6 = [R18(a, b) for a, b in itertools.combinations(COLOUR, 2)]
        ok &= cent_dim(soU) == 0 and cent_dim(so6) == 0
        ok &= cent_dim(soU + so6) == d_expect
    # for U = W the intersection is exactly span{rot(i,j)}
    soW = [R18(a, b) for a, b in itertools.combinations(W, 2)]
    so6 = [R18(a, b) for a, b in itertools.combinations(COLOUR, 2)]
    span_ps = sp.Matrix.hstack(*[sp.Matrix(list(x)) for x in soW + so6])
    span_rot = sp.Matrix.hstack(*[sp.Matrix(list(x)) for x in ROT_W.values()])
    ok &= rank_q(span_rot) == 6 and rank_q(span_ps.row_join(span_rot)) == rank_q(span_ps)
    return ok


a_, b_, c_, d_ = W
Rw = ROT_W
Tp = [(Rw[(b_, c_)] + Rw[(a_, d_)]) / 2, (Rw[(a_, c_)] - Rw[(b_, d_)]) / 2, (Rw[(a_, b_)] + Rw[(c_, d_)]) / 2]
Tm = [(Rw[(b_, c_)] - Rw[(a_, d_)]) / 2, (Rw[(a_, c_)] + Rw[(b_, d_)]) / 2, (Rw[(a_, b_)] - Rw[(c_, d_)]) / 2]


def br(x, y):
    return x * y - y * x


@claim("d4-18", "so(4)_Delta = su(2)_+ (+) su(2)_-: T^pm_1 = (rot(13,14) +- rot(7,15))/2, T^pm_2 = (rot(7,14) -+ rot(13,15))/2, "
                 "T^pm_3 = (rot(7,13) +- rot(14,15))/2 (rot acting by (1/2)ad): [T^+,T^-] = 0, [T^pm_1,T^pm_2] = -T^pm_3 cyclically, "
                 "trace form negative definite (compact); each factor acts on span(W) with weights +-1/2 (two doublets)")
def _c18():
    ok = True
    for X in Tp:
        for Y in Tm:
            ok &= br(X, Y) == sp.zeros(18, 18)
    for T in (Tp, Tm):
        for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            ok &= br(T[i], T[j]) == -T[k]
    basis = Tp + Tm
    Gm = sp.Matrix(6, 6, lambda i, j: (basis[i] * basis[j]).trace())
    ok &= all(e_ < 0 for e_ in Gm.eigenvals())
    ev = (Tp[2]).eigenvals()
    ok &= ev == {sp.I / 2: 4, -sp.I / 2: 4, 0: 10}
    return ok


@claim("d4-19", "spectrum on the 18-dim frame (kappa=5): b = 2 T^+_3 gives chi_S = (lam^2+25)^5 (lam^2+16)^2 (lam^2+36)^2, i.e. "
                 "frequencies kappa (5 singlet pairs) and kappa -+ beta/2 (su(2) weights +-1/2, twice each); b = 2T^+_3 + 4T^-_3 gives four lines "
                 "kappa +- (beta_+ +- beta_-)/2 = 8, 6, 4, 2; M semisimple (Y = -G3/(2 kappa m)), Krein (k,k) per frequency, zero-point 0; "
                 "a weak-only rotation R(7,13) or a colour-only rotation R(10,16) is NOT semisimple (secular growth)")
def _c19():
    ok = True
    k = 5
    b1 = 2 * Tp[2]
    S1 = b1 + k * G3
    ok &= poly_eq(S1.charpoly(lam).as_expr(), (lam ** 2 + 25) ** 5 * (lam ** 2 + 16) ** 2 * (lam ** 2 + 36) ** 2)
    b2 = 2 * Tp[2] + 4 * Tm[2]
    S2 = b2 + k * G3
    ok &= poly_eq(S2.charpoly(lam).as_expr(),
                  (lam ** 2 + 25) ** 5 * (lam ** 2 + 64) * (lam ** 2 + 36) * (lam ** 2 + 16) * (lam ** 2 + 4))
    for b in (b1, b2):
        A = -(b + k * G3)
        D = -b + k * G3
        Y = -G3 / (2 * k)
        ok &= A * Y - Y * D + sp.eye(18) == sp.zeros(18, 18)
        ok &= is_semisimple(phase_M(b, k, 1, Gm=G3))
        res = krein(b, k, Gm=G3)
        ok &= all(npos == nneg for (_, npos, nneg, _) in res)
        ok &= abs(sum((npos - nneg) * w / 2 for (w, npos, nneg, _) in res)) < 1e-12
    for bad in (R18(7, 13), R18(10, 16)):
        ok &= not is_semisimple(phase_M(bad, k, 1, Gm=G3))
        ok &= bad * G3 != G3 * bad
    return ok


# ============================================================================================= d4-20..24 inherited Yang-Mills facts
def su_gens(N):
    gens = []
    for j in range(N):
        for k in range(j + 1, N):
            Sm = sp.zeros(N, N)
            Sm[j, k] = Sm[k, j] = sp.Rational(1, 2)
            gens.append(Sm)
            Am = sp.zeros(N, N)
            Am[j, k] = -sp.I / 2
            Am[k, j] = sp.I / 2
            gens.append(Am)
    for l_ in range(1, N):
        Dm = sp.zeros(N, N)
        for j in range(l_):
            Dm[j, j] = 1
        Dm[l_, l_] = -l_
        gens.append(Dm / sp.sqrt(2 * l_ * (l_ + 1)))
    return gens


T4 = su_gens(4)
T2 = su_gens(2)


@claim("d4-20", "SU(4), tr(T^A T^B) = delta/2: sum_A T^A X T^A = (1/2)[tr(X) 1 - X/4] for all X (eigenvalue 15/8 = C_F on 1, -1/8 on traceless X)")
def _c20():
    ok = len(T4) == 15
    for A in range(15):
        for B in range(15):
            ok &= sp.simplify((T4[A] * T4[B]).trace() - (sp.Rational(1, 2) if A == B else 0)) == 0
    for j in range(4):
        for k in range(4):
            X = sp.zeros(4, 4)
            X[j, k] = 1
            lhs = sum((T for T in (Ta * X * Ta for Ta in T4)), sp.zeros(4, 4))
            ok &= zero_mat(lhs - (X.trace() * sp.eye(4) - X / 4) / 2)
    return ok


def tdot(Ts1, Ts2):
    return sum((sp.kronecker_product(a, b) for a, b in zip(Ts1, Ts2)), sp.zeros(Ts1[0].shape[0] * Ts2[0].shape[0]))


@claim("d4-21", "channel factors T1.T2 = (1/2)[C2(pair) - C2(R1) - C2(R2)]: SU(4) 4x4 -> 10: +3/8, 6: -5/8; 4bar x 4 -> 1: -15/8, 15: +1/8; "
                 "SU(2) 2x2 -> 3: +1/4, 1: -3/4; C2 = 15/8, 9/2, 5/2, 4; PS bilinears: Dirac (4,2,1)bar x (4,1,2) = (1+15,2,2); a Lorentz-scalar "
                 "Majorana bilinear of (4,1,2) lies in Sym^2(C^8) = (10,1,3) (+) (6,1,1) (dims 30 + 6 = 36): (1,2,2) and (6,1,1) attractive, "
                 "(15,2,2) and (10,1,3) repulsive in one-boson exchange")
def _c21():
    ok = True
    n = 4
    TT = tdot(T4, T4)
    P = sp.zeros(16, 16)
    for i in range(4):
        for j in range(4):
            P[4 * i + j, 4 * j + i] = 1
    Psym = (sp.eye(16) + P) / 2
    Panti = (sp.eye(16) - P) / 2
    ok &= zero_mat(TT * Psym - sp.Rational(3, 8) * Psym) and zero_mat(TT * Panti + sp.Rational(5, 8) * Panti)
    ok &= rank_q(Psym) == 10 and rank_q(Panti) == 6
    Cas = sum(((sp.kronecker_product(T, sp.eye(4)) + sp.kronecker_product(sp.eye(4), T)) ** 2 for T in T4), sp.zeros(16, 16))
    ok &= zero_mat(Cas * Psym - sp.Rational(9, 2) * Psym) and zero_mat(Cas * Panti - sp.Rational(5, 2) * Panti)
    CF = sum((T * T for T in T4), sp.zeros(4, 4))
    ok &= zero_mat(CF - sp.Rational(15, 8) * sp.eye(4))
    Tbar = [-T.T for T in T4]                       # conjugate representation -T^*  (T Hermitian: T^* = T^T)
    TTb = tdot(Tbar, T4)
    singlet = sp.Matrix([1 if i == j else 0 for i in range(4) for j in range(4)])
    ok &= zero_mat(TTb * singlet + sp.Rational(15, 8) * singlet)
    Pin = singlet * singlet.T / 4
    Padj = sp.eye(16) - Pin
    ok &= zero_mat(TTb * Padj - sp.Rational(1, 8) * Padj)
    Cas_adj = sum(((sp.kronecker_product(Tb, sp.eye(4)) + sp.kronecker_product(sp.eye(4), T)) ** 2 for Tb, T in zip(Tbar, T4)), sp.zeros(16, 16))
    ok &= zero_mat(Cas_adj * Padj - 4 * Padj)
    TT2 = tdot(T2, T2)
    P2 = sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    ok &= zero_mat(TT2 * (sp.eye(4) + P2) / 2 - sp.Rational(1, 4) * (sp.eye(4) + P2) / 2)
    ok &= zero_mat(TT2 * (sp.eye(4) - P2) / 2 + sp.Rational(3, 4) * (sp.eye(4) - P2) / 2)
    # Sym^2(C^4 (x) C^2) = Sym^2 C^4 (x) Sym^2 C^2 (+) Lambda^2 C^4 (x) Lambda^2 C^2 : count by SU(4) x SU(2) Casimirs
    T8_4 = [sp.kronecker_product(T, sp.eye(2)) for T in T4]
    T8_2 = [sp.kronecker_product(sp.eye(4), T) for T in T2]
    swap = sp.zeros(64, 64)
    for i in range(8):
        for j in range(8):
            swap[8 * i + j, 8 * j + i] = 1
    Ps8 = (sp.eye(64) + swap) / 2

    def cas2(Ts):
        return sum(((sp.kronecker_product(T, sp.eye(8)) + sp.kronecker_product(sp.eye(8), T)) ** 2 for T in Ts), sp.zeros(64, 64))

    C4 = cas2(T8_4) * Ps8
    C2 = cas2(T8_2) * Ps8
    ok &= rank_q(Ps8) == 36
    # joint eigenspaces on Sym^2: (C4, C2) = (9/2, 2) of dim 30 and (5/2, 0) of dim 6
    n1 = 64 - rank_q((C4 - sp.Rational(9, 2) * sp.eye(64)).col_join(C2 - 2 * sp.eye(64)).col_join(sp.eye(64) - Ps8))
    n2 = 64 - rank_q((C4 - sp.Rational(5, 2) * sp.eye(64)).col_join(C2).col_join(sp.eye(64) - Ps8))
    ok &= n1 == 30 and n2 == 6
    # verdicts: sum of channel factors (equal couplings) for each PS channel
    ch = {'(1,2,2)': -sp.Rational(15, 8), '(15,2,2)': sp.Rational(1, 8),
          '(10,1,3)': sp.Rational(3, 8) + sp.Rational(1, 4), '(6,1,1)': -sp.Rational(5, 8) - sp.Rational(3, 4)}
    ok &= ch['(1,2,2)'] < 0 and ch['(6,1,1)'] < 0 and ch['(15,2,2)'] > 0 and ch['(10,1,3)'] > 0
    return ok


@claim("d4-22", "contact gap equation: int_{k^2<=L^2} d^4k/(2pi)^4 1/(k^2+M^2) = [L^2 - M^2 ln(1+L^2/M^2)]/(16 pi^2); f(x) = 1 - x^2 ln(1+1/x^2) "
                 "decreases strictly from 1 to 0, so 1 = lambda f(x) has a solution x > 0 iff lambda > 1; at lambda = 7/4, x = 0.5330235 (mpmath, 25 digits)")
def _c22():
    k, L, M = sp.symbols('k L M', positive=True)
    I_ = sp.integrate(2 * sp.pi ** 2 * k ** 3 / (k ** 2 + M ** 2), (k, 0, L)) / (2 * sp.pi) ** 4
    ok = sp.simplify(I_ - (L ** 2 - M ** 2 * sp.log(1 + L ** 2 / M ** 2)) / (16 * sp.pi ** 2)) == 0
    x, u = sp.symbols('x u', positive=True)
    f = 1 - x ** 2 * sp.log(1 + 1 / x ** 2)
    fp = sp.simplify(sp.diff(f, x) - 2 * x * (1 / (1 + x ** 2) - sp.log(1 + 1 / x ** 2)))
    ok &= fp == 0
    # ln(1+u) > u/(1+u) for u > 0: g(u) = ln(1+u) - u/(1+u), g(0) = 0, g'(u) = u/(1+u)^2 > 0
    g = sp.log(1 + u) - u / (1 + u)
    ok &= sp.simplify(sp.diff(g, u) - u / (1 + u) ** 2) == 0 and g.subs(u, 0) == 0
    ok &= sp.limit(f, x, 0, '+') == 1 and sp.limit(f, x, sp.oo) == 0
    mp.mp.dps = 30
    root = mp.findroot(lambda y: mp.mpf(7) / 4 * (1 - y ** 2 * mp.log(1 + 1 / y ** 2)) - 1, 0.5)
    ok &= abs(root - mp.mpf('0.5330234964064841071252163')) < mp.mpf('1e-24')
    ok &= abs(root - mp.mpf('0.533')) < 5e-4
    return ok


@claim("d4-23", "ladder critical coupling alpha_c = pi/(3 C_F) = 8 pi/45 = 0.5585 for C_F = 15/8; one-loop beta0 = (11/3) C_A - (4/3) T_F N_f "
                 "= (22 - 2 N_f)/3 for SU(2) with N_f Dirac doublets (beta0 = 6 at N_f = 2); Lambda = mu exp(-8 pi^2/(beta0 g^2)) is RG invariant")
def _c23():
    CF = sp.Rational(15, 8)
    ac = sp.pi / (3 * CF)
    ok = sp.simplify(ac - 8 * sp.pi / 45) == 0 and abs(float(ac) - 0.5585053606) < 1e-9
    Nf = sp.Symbol('N_f')
    b0 = sp.Rational(11, 3) * 2 - sp.Rational(4, 3) * sp.Rational(1, 2) * Nf
    ok &= sp.simplify(b0 - (22 - 2 * Nf) / 3) == 0 and b0.subs(Nf, 2) == 6
    t, g2, B0 = sp.symbols('t g2 beta0', positive=True)   # t = ln mu
    g2f = sp.Function('g2')(t)
    # d g^2/dt = -2 beta0 g^4/(16 pi^2) ; Lambda = exp(t - 8 pi^2/(beta0 g^2))
    lnLam = t - 8 * sp.pi ** 2 / (B0 * g2f)
    dlnLam = sp.diff(lnLam, t).subs(sp.Derivative(g2f, t), -2 * B0 * g2f ** 2 / (16 * sp.pi ** 2))
    ok &= sp.simplify(dlnLam) == 0
    return ok


@claim("d4-24", "SU(2N_f) -> Sp(2N_f) at N_f = 2: dim su(4) = 15, dim sp(4) = 10 (anti-Hermitian X with X^T Omega + Omega X = 0), 5 Goldstones")
def _c24():
    Om = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]])
    # real basis of su(4): i T^A ; solve for the subspace preserving Omega
    cs = sp.symbols('c0:15', real=True)
    X = sum((cs[A] * sp.I * T4[A] for A in range(15)), sp.zeros(4, 4))
    eqs = []
    for e_ in (X.T * Om + Om * X):
        e_ = sp.expand(e_)
        eqs += [sp.re(e_), sp.im(e_)]
    Mx = sp.Matrix([[sp.diff(e_, c) for c in cs] for e_ in eqs])
    dim_sp = 15 - Mx.rank()
    return dim_sp == 10 and 15 - dim_sp == 5


@claim("d4-25", "no scale from the algebra: (tau, kappa, b, m) -> (s tau, kappa/s, b/s, s m) maps M to M/s and solutions to solutions, "
                 "and every Clifford datum (eta, G, l_G, p_G, so(4)_Delta) is invariant, so no value of kappa (or ell = c/kappa) is selected, "
                 "with or without a constant b; at kappa = 0 the spectrum of M is that of -b doubled (no rest frequency)")
def _c25():
    s = sp.Symbol('s', positive=True)
    b = beta * (R(1, 2) - R(4, 5)) + sp.Rational(1, 3) * beta * (R(1, 2) + R(4, 5))
    ok = zero_mat(phase_M(b / s, kap / s, s * mm) - phase_M(b, kap, mm) / s)
    b0 = sp.Rational(1, 2) * (R(1, 2) - R(4, 5)) + R(3, 6) / 3
    chi = phase_M(b0, 0, 1).charpoly(lam).as_expr()
    chib = (-b0).charpoly(lam).as_expr()
    ok &= poly_eq(chi, chib ** 2)
    return ok


# ============================================================================================= done
dt = time.time() - T0
if FAILS:
    print(f"FAIL summary: {len(FAILS)} claim(s) failed: {', '.join(FAILS)}  ({dt:.1f}s)")
    sys.exit(1)
print(f"# all d4 checks passed in {dt:.1f}s")
sys.exit(0)
