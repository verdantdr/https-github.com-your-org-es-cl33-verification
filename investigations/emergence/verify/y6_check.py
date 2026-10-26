#!/usr/bin/env python3
"""Adversarial referee check y6 for investigations/emergence/sections/e6_bfield_triples.tex.

    python3 y6_check.py        (from any directory)

Independent of verify/e6_bfield_triples.py (nothing is imported from it).  Where possible a claim is re-derived by a different
method than the derivation script: random group elements and matrix exponentials instead of single bivectors, a Cartan-support
reduction instead of orbit counts, character (Frobenius--Schur) sums over a finite spanning group instead of null spaces, the
torus integral of SU(4) = Spin(3,3)_C instead of the Weyl multiplicity formula, the boundedness criterion of RE Thm 5.5 (eta-pairing of
eigenvectors) besides exact Jordan ranks over Q(i), and symbolic parameters where the derivation used rational points.
Claims found false in the section appear here as checks of the CORRECT statement (ids y6-5d, y6-6f, y6-4f).

Exactness: algebraic identities in integer/rational arithmetic (signed-permutation matrices, sympy, exact ranks over ZZ, QQ, QQ<I>);
floating point only for exponentials, spectra and random tests, with tolerance 1e-9 unless stated.  Runtime: well under a minute.
"""
import itertools
import sys
import time
from fractions import Fraction
from math import comb

import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sympy.polys.matrices import DomainMatrix

T0 = time.time()
FAILED = []
RNG = np.random.default_rng(20261026)
TOL = 1e-9


def check(cid, text):
    def deco(fn):
        try:
            ok = bool(fn())
        except Exception as exc:  # a crash is a failure
            import traceback
            traceback.print_exc()
            print(f"   exception in {cid}: {exc!r}")
            ok = False
        print(f"{'PASS' if ok else 'FAIL'} {cid}: {text}")
        if not ok:
            FAILED.append(cid)
        return fn
    return deco


# ------------------------------------------------------------------------------------------------ realisation (ES17 eq. (1))
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def kr(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = np.kron(out, m)
    return out


e = {1: kr(sx, I2, I2), 2: kr(sz, sx, I2), 3: kr(sz, sz, sx), 4: kr(ep, I2, I2), 5: kr(sz, ep, I2), 6: kr(sz, sz, ep)}
eta = {a: (1 if a <= 3 else -1) for a in range(1, 7)}
I8 = np.eye(8, dtype=np.int64)
Z8 = 0 * I8
T = e[1] @ e[2] @ e[3]
WS = e[4] @ e[5] @ e[6]                  # omega_S
OM = T @ WS                              # omega = Gamma = e1...e6
J = e[2] @ e[3]
Omg = e[1] @ e[4] @ e[5] @ e[6]          # Omega
gam = [e[1], e[4], e[5], e[6]]
Kin = [-(e[1] @ g) for g in gam]         # K^mu
BIV = {(a, b): e[a] @ e[b] for a, b in itertools.combinations(range(1, 7), 2)}
COMPACT = [k for k in BIV if (k[0] <= 3) == (k[1] <= 3)]
BOOST = [k for k in BIV if (k[0] <= 3) != (k[1] <= 3)]


def com(A, B):
    return A @ B - B @ A


def acom(A, B):
    return A @ B + B @ A


def same(A, B):
    return np.array_equal(A, B)


def close(A, B, tol=TOL):
    return np.max(np.abs(np.asarray(A, dtype=complex) - np.asarray(B, dtype=complex))) < tol


def zrank(rows):
    """exact rank of an integer matrix (list of rows or 2-D array)"""
    A = np.asarray(rows, dtype=object)
    if A.size == 0:
        return 0
    return DomainMatrix([[sp.ZZ(int(x)) for x in r] for r in A.tolist()], A.shape, sp.ZZ).rank()


def blade_coeffs(X):
    """coefficients of X in the 64 blades of Cl(3,3) (trace form; blades are signed permutations)"""
    out = {}
    for r in range(7):
        for I in itertools.combinations(range(1, 7), r):
            B = I8.copy()
            for a in I:
                B = B @ e[a]
            c = np.trace(B.T @ X) / 8.0
            if abs(c) > 1e-12:
                out[I] = c
    return out


# ------------------------------------------------------------------------------------------------ signed permutations (large rungs)
class SPm:
    """signed permutation matrix M with M[i, p[i]] = s[i]"""
    __slots__ = ("p", "s")

    def __init__(self, p, s):
        self.p, self.s = p, s

    @staticmethod
    def dense(M):
        M = np.asarray(M)
        p = np.argmax(M != 0, axis=1)
        s = M[np.arange(M.shape[0]), p]
        assert np.all(np.abs(s) == 1) and np.count_nonzero(M) == M.shape[0]
        return SPm(p.astype(np.int64), s.astype(np.int64))

    def __matmul__(self, o):
        return SPm(o.p[self.p], self.s * o.s[self.p])

    def kron(self, o):
        d = len(o.p)
        return SPm((self.p[:, None] * d + o.p[None, :]).ravel(), (self.s[:, None] * o.s[None, :]).ravel())

    @property
    def T(self):
        pinv = np.argsort(self.p)
        return SPm(pinv, self.s[pinv])

    def same(self, o, sign=1):
        return np.array_equal(self.p, o.p) and np.array_equal(self.s, sign * o.s)

    def trace(self):
        return int(np.sum(self.s[self.p == np.arange(len(self.p))]))

    def neg(self):
        return SPm(self.p, -self.s)


SPe = {a: SPm.dense(e[a]) for a in e}
SPom = SPm.dense(OM)
SPI8 = SPm.dense(I8)


def tower(n):
    """graded generators E_(k,a) = omega^(x)(k-1) (x) e_a (x) 1^(x)(n-k) as signed permutations"""
    gens = {}
    for k in range(1, n + 1):
        for a in range(1, 7):
            fac = [SPom] * (k - 1) + [SPe[a]] + [SPI8] * (n - k)
            M = fac[0]
            for f in fac[1:]:
                M = M.kron(f)
            gens[(k, a)] = M
    return gens


def prod(ms, dim):
    M = SPm(np.arange(dim), np.ones(dim, dtype=np.int64))
    for m in ms:
        M = M @ m
    return M


# ================================================================================================ 0. conventions
@check("y6-0", "conventions: the six e_a of ES17 eq. (1) satisfy the Clifford relations of R^{3,3}, e_a^T = eta_aa e_a (ES17 Lemma 2.1, eq. (4)); "
       "T^2 = -1, omega_S^2 = 1, T omega_S = Gamma; the Fock number N = sum a+_i a-_i is diagonal with spectrum 0..3 (multiplicities 1,3,3,1)")
def _():
    ok = all(same(acom(e[a], e[b]), 2 * eta[a] * (a == b) * I8) for a in e for b in e)
    ok &= all(same(e[a].T, eta[a] * e[a]) for a in e)
    ok &= same(T @ T, -I8) and same(WS @ WS, I8) and same(T @ WS, OM)
    ap = {i: (e[i] + e[i + 3]) / 2 for i in (1, 2, 3)}
    am = {i: (e[i] - e[i + 3]) / 2 for i in (1, 2, 3)}
    ok &= all(close(ap[i] @ am[j] + am[j] @ ap[i], (i == j) * np.eye(8)) for i in ap for j in am)
    Nf = sum(ap[i] @ am[i] for i in ap)
    ok &= close(Nf, np.diag(np.diag(Nf))) and sorted(np.round(np.diag(Nf)).astype(int).tolist()) == [0, 1, 1, 1, 2, 2, 2, 3]
    return ok


# ================================================================================================ 1. T as form and operator
def rand_so33(compact_only=False, boost_only=False):
    keys = COMPACT if compact_only else (BOOST if boost_only else list(BIV))
    c = RNG.normal(size=len(keys))
    return sum(ci * BIV[k] for ci, k in zip(c, keys))


@check("y6-1a", "Prop. Tform(i),(ii) at group level, by matrix exponentials of random elements of so(3,3) (not single bivectors): (e^Y)^T T e^Y = T and "
       "(e^Y)^T omega_S e^Y = omega_S for 20 random Y; e^Y commutes with T for random compact Y; for a random boost combination Y in span{e_ie_(a+3)}, "
       "e^Y T e^-Y = e^{2Y} T")
def _():
    ok = True
    for _ in range(20):
        Y = 0.4 * rand_so33()
        g = expm(Y)
        ok &= close(g.T @ T @ g, T, 1e-8) and close(g.T @ WS @ g, WS, 1e-8)
    for _ in range(5):
        Y = rand_so33(compact_only=True)
        ok &= close(com(expm(Y), T), np.zeros((8, 8)), 1e-9)
        B = 0.5 * rand_so33(boost_only=True)
        ok &= close(expm(B) @ T @ expm(-B), expm(2 * B) @ T, 1e-8)
    return ok


def invariant_forms_cartan(gens_biv, cartan, dim):
    """exact basis of {M : Y^T M + M Y = 0 for all Y} using the Cartan support reduction: the pole operators in `cartan` are diagonal and
    symmetric, so M is supported on index pairs of opposite joint weight; the remaining conditions are solved exactly."""
    W = np.stack([np.diag(c) for c in cartan], axis=1)         # dim x r joint weights
    key = {}
    for i in range(dim):
        key.setdefault(tuple(W[i]), []).append(i)
    pairs = [(i, j) for i in range(dim) for j in key.get(tuple(-W[i]), [])]
    idx = {p: t for t, p in enumerate(pairs)}
    rows = []
    for Y in gens_biv:
        # (Y^T M + M Y)_{ij} = sum_k Y_{ki} M_{kj} + sum_k M_{ik} Y_{kj}
        Ysp = SPm.dense(Y)
        R = {}
        for (k, j), t in idx.items():
            # Y_{k,i} nonzero for i with Ysp.p[k] = i  -> contributes to entry (Ysp.p[k], j) with sign s[k]
            i = Ysp.p[k]
            R.setdefault((i, j), {}).setdefault(t, 0)
            R[(i, j)][t] += int(Ysp.s[k])
            # M_{j? } : (M Y)_{a,b} = M_{a,k'} Y_{k',b}; Y_{k',b} nonzero iff b = p[k'] -> entry (k, p[j]) from M_{k,j}
            b = Ysp.p[j]
            R.setdefault((k, b), {}).setdefault(t, 0)
            R[(k, b)][t] += int(Ysp.s[j])
        for ent, coeffs in R.items():
            row = [0] * len(pairs)
            for t, v in coeffs.items():
                row[t] += v
            if any(row):
                rows.append(row)
    nunk = len(pairs)
    rows = [list(r) for r in {tuple(r) for r in rows}]
    if not rows:
        return pairs, [sp.eye(nunk)[:, t] for t in range(nunk)]
    D = DomainMatrix([[sp.ZZ(int(x)) for x in r] for r in rows], (len(rows), nunk), sp.ZZ).convert_to(sp.QQ)
    ns = D.nullspace().to_Matrix()
    return pairs, [ns[t, :].T for t in range(ns.shape[0])]


def forms_from_nullspace(pairs, ns, dim):
    out = []
    for v in ns:
        M = np.zeros((dim, dim), dtype=object)
        for (i, j), c in zip(pairs, list(v)):
            M[i, j] = sp.Rational(c)
        out.append(M)
    return out


@check("y6-1b", "Prop. Tform(i), last sentence, by an independent exact solve (Cartan-support reduction to 8 unknowns, then all 15 bivectors): "
       "the forms on R^8 invariant under so(3,3) are exactly span{T, omega_S}, one antisymmetric (T) and one symmetric (omega_S)")
def _():
    cart = [e[i] @ e[i + 3] for i in (1, 2, 3)]
    ok = all(same(c, np.diag(np.diag(c))) and same(c.T, c) for c in cart)
    pairs, ns = invariant_forms_cartan(list(BIV.values()), cart, 8)
    ok &= len(ns) == 2
    forms = forms_from_nullspace(pairs, ns, 8)
    span = sp.Matrix([[sp.Rational(x) for x in F.ravel()] for F in forms])
    tgt = sp.Matrix([[int(x) for x in T.ravel()], [int(x) for x in WS.ravel()]])
    ok &= span.rank() == 2 and span.col_join(tgt).rank() == 2
    return ok


@check("y6-1c", "Prop. Tform(ii)-(iv): among the 15 bivectors T commutes with exactly the 6 compact ones and anticommutes with the 9 boosts; the commutant "
       "of T in M_8(R) has exact dimension 32 = dim_R gl(4,C), equals the span of the 8 x 4 products (temporal blade)(even spatial blade), is closed "
       "under products, has centre span{1, T}; Cl^0(V^-) obeys the quaternion relations; Ad_T = eta on V")
def _():
    ok = sorted(k for k in BIV if same(com(T, BIV[k]), Z8)) == sorted(COMPACT)
    ok &= sorted(k for k in BIV if same(acom(T, BIV[k]), Z8)) == sorted(BOOST)
    L = np.kron(T, I8) - np.kron(I8, T.T)                  # vec(TX - XT), row-major
    ok &= 64 - zrank(L) == 32
    prods = []
    for r in range(4):
        for I in itertools.combinations((1, 2, 3), r):
            for Jsp in [(), (4, 5), (4, 6), (5, 6)]:
                B = I8.copy()
                for a in I + Jsp:
                    B = B @ e[a]
                prods.append(B)
    ok &= all(same(com(B, T), Z8) for B in prods) and zrank([B.ravel() for B in prods]) == 32
    # closed under products: products of two basis elements stay in the span (blades are +- blades, so check membership)
    basis = np.array([B.ravel() for B in prods])
    ok &= all(zrank(np.vstack([basis, (A @ B).ravel()])) == 32 for A in prods[::5] for B in prods[::7])
    # centre: elements of the commutant commuting with all of it
    rows = []
    for B in prods:
        rows.append(np.array([com(A, B).ravel() for A in prods]).T)  # columns = basis coefficient directions
    big = np.vstack(rows)
    ok &= 32 - zrank(big) == 2 and same(com(T, T), Z8)
    q1, q2 = e[4] @ e[5], e[5] @ e[6]
    q3 = q1 @ q2
    ok &= same(q1 @ q1, -I8) and same(q2 @ q2, -I8) and same(q3 @ q3, -I8) and same(q1 @ q2, -q2 @ q1) and same(q3, -(e[4] @ e[6]))
    ok &= all(same(T @ e[a] @ np.linalg.inv(T).round().astype(np.int64), eta[a] * e[a]) for a in e)
    return ok


# ================================================================================================ 2. Dirac data
al, be, ga = sp.symbols('alpha beta gamma', real=True)
S8 = {a: sp.Matrix(e[a].tolist()) for a in e}
Ts, Js = sp.Matrix(T.tolist()), sp.Matrix(J.tolist())
Kins = [sp.Matrix(K.tolist()) for K in Kin]
gams = [sp.Matrix(g.tolist()) for g in gam]


def Ysym(y):
    """element of so(3)_+ with axial vector y: y1 e2e3 + y2 e3e1 + y3 e1e2 (acts on V^+ as v -> -y x v)"""
    return y[0] * S8[2] * S8[3] + y[1] * S8[3] * S8[1] + y[2] * S8[1] * S8[2]


def fro2(M):
    return sp.expand(sum(x ** 2 for x in M))


@check("y6-2a", "Prop. dirac(i) and Remark compare(c), symbolic in (alpha, beta, gamma) and for a general clock: Y^T K^a + K^a Y = 2 Y(e1) gamma^a with "
       "Y(e1) = (1/2)[Y, e1] = beta e3 - gamma e2, Y^T K^0 + K^0 Y = 0, sum_a ||.||_F^2 = 96(beta^2 + gamma^2); for the clock n = (2,3,6)/7 with "
       "K^a(n) = -n gamma^a, sum_a ||Y^T K^a(n) + K^a(n) Y||^2 = 96 |y x n|^2, a quadratic form with kernel R J(n), J(n) = T n; the covariance system "
       "(15 bivector + 16 Lambda unknowns, 256 equations) has a 7-dimensional solution space meeting so(3)_+ in R J")
def _():
    Y = Ysym([al, be, ga])
    ok = sp.expand(Y.T * Kins[0] + Kins[0] * Y) == sp.zeros(8)
    Ye1 = (Y * S8[1] - S8[1] * Y) / 2
    ok &= sp.expand(Ye1 - (be * S8[3] - ga * S8[2])) == sp.zeros(8)
    tot = 0
    for a in (1, 2, 3):
        D = Y.T * Kins[a] + Kins[a] * Y
        ok &= sp.expand(D - 2 * Ye1 * gams[a]) == sp.zeros(8)
        tot += fro2(D)
    ok &= sp.expand(tot - 96 * (be ** 2 + ga ** 2)) == 0
    n = [sp.Rational(2, 7), sp.Rational(3, 7), sp.Rational(6, 7)]
    nv = sum((c * S8[i + 1] for i, c in enumerate(n)), sp.zeros(8))
    tot = sum(fro2(Y.T * (-nv * gams[a]) + (-nv * gams[a]) * Y) for a in (1, 2, 3))
    y = sp.Matrix([al, be, ga])
    ok &= sp.expand(tot - 96 * (y.cross(sp.Matrix(n))).dot(y.cross(sp.Matrix(n)))) == 0
    Jn = Ts * nv
    ok &= sp.expand(Jn - Ysym(n)) == sp.zeros(8)
    Yn = Ysym(n)
    ok &= all(sp.expand(Yn.T * (-nv * gams[a]) + (-nv * gams[a]) * Yn) == sp.zeros(8) for a in (1, 2, 3))
    # covariance system, exact
    cb = sp.symbols('c0:15')
    lm = sp.symbols('l0:16')
    keys = list(BIV)
    Yg = sum((cb[t] * sp.Matrix(BIV[k].tolist()) for t, k in enumerate(keys)), sp.zeros(8))
    eqs = []
    for mu in range(4):
        R = Yg.T * Kins[mu] + Kins[mu] * Yg - sum((lm[4 * mu + nu] * Kins[nu] for nu in range(4)), sp.zeros(8))
        eqs += list(R)
    A, _ = sp.linear_eq_to_matrix(eqs, list(cb) + list(lm))
    ns = A.nullspace()
    ok &= len(ns) == 7
    sol = sp.Matrix.hstack(*[v[:15, 0] for v in ns])
    ok &= sol.rank() == 7
    so3p = [keys.index((2, 3)), keys.index((1, 3)), keys.index((1, 2))]
    U = sp.zeros(15, 3)
    for c, t in enumerate(so3p):
        U[t, c] = 1
    inter = sol.rank() + U.rank() - sp.Matrix.hstack(sol, U).rank()
    ok &= inter == 1
    Jvec = sp.zeros(15, 1)
    Jvec[keys.index((2, 3))] = 1
    ok &= sp.Matrix.hstack(sol, Jvec).rank() == 7
    return ok


@check("y6-2b", "Prop. dirac(ii), symbolic in phi and by the rotor (1 + n e1) for the clock n = (2,3,6)/7: e^{phi e1e2} e1 e^{-phi e1e2} = cos2phi e1 - sin2phi e2; "
       "the rotor carries e1 to n, J to T n, K^a to -n gamma^a, fixes T and K^0, and n J(n) = J(n) n = T; the stabiliser of e1 in so(3)_+ is R J")
def _():
    ph = sp.symbols('phi', real=True)
    E12 = S8[1] * S8[2]
    R = sp.cos(ph) * sp.eye(8) + sp.sin(ph) * E12
    Ri = sp.cos(ph) * sp.eye(8) - sp.sin(ph) * E12
    ok = sp.simplify(R * S8[1] * Ri - (sp.cos(2 * ph) * S8[1] - sp.sin(2 * ph) * S8[2])) == sp.zeros(8)
    E13 = S8[1] * S8[3]
    R3 = sp.cos(ph) * sp.eye(8) + sp.sin(ph) * E13
    R3i = sp.cos(ph) * sp.eye(8) - sp.sin(ph) * E13
    ok &= sp.simplify(R3 * S8[1] * R3i - (sp.cos(2 * ph) * S8[1] - sp.sin(2 * ph) * S8[3])) == sp.zeros(8)
    n = [sp.Rational(2, 7), sp.Rational(3, 7), sp.Rational(6, 7)]
    nv = sum((c * S8[i + 1] for i, c in enumerate(n)), sp.zeros(8))
    Rn = sp.eye(8) + nv * S8[1]
    Rni = (sp.eye(8) + S8[1] * nv) / (2 * (1 + n[0]))
    ok &= Rn * Rni == sp.eye(8)
    ok &= Rn * S8[1] * Rni == nv and Rn * Js * Rni == Ts * nv and Rn * Ts * Rni == Ts
    ok &= all(Rn * Kins[a] * Rni == -nv * gams[a] for a in (1, 2, 3)) and Rn * Kins[0] * Rni == Kins[0]
    ok &= nv * (Ts * nv) == Ts and (Ts * nv) * nv == Ts
    Y = Ysym([al, be, ga])
    sol = sp.solve(list(Y * S8[1] - S8[1] * Y), [be, ga], dict=True)
    ok &= sol == [{be: 0, ga: 0}]
    return ok


def fock_vacuum():
    am = {i: e[i] - e[i + 3] for i in (1, 2, 3)}        # 2 a-
    Mst = np.vstack([am[i] for i in (1, 2, 3)])
    # common kernel of the a-: one dimensional
    null = sp.Matrix(Mst.tolist()).nullspace()
    assert len(null) == 1
    v = np.array([int(x) for x in null[0]], dtype=np.int64)
    return v


@check("y6-2c", "Prop. dirac(iii),(iv): the spin lift (1/2)e_je_k of so(3)_+ has Casimir -3/4 on R^8 and e^{pi e_je_k} = -1 (expm); e1e2, e3e1 anticommute "
       "with J, commute with T, and Ad_{e1e2} maps (e1, J, T, gamma^a) to (-e1, -J, T, gamma^a); with the field-octant map built from the Fock vacuum "
       "(f_k = a+_k|0>, g_m = a+_j a+_k|0>, Phi(X) = sum E_k f_k + B_m g_m, TR Thm 4.7) -T Phi = Phi G, and e^{-m lam T} Phi = Phi e^{m lam G} (expm)")
def _():
    gens = [e[2] @ e[3] / 2, e[3] @ e[1] / 2, e[1] @ e[2] / 2]
    ok = close(sum(g @ g for g in gens), -0.75 * np.eye(8))
    ok &= all(close(expm(np.pi * (e[a] @ e[b]).astype(float)), -np.eye(8)) for a, b in ((2, 3), (3, 1), (1, 2)))
    for X in (e[1] @ e[2], e[3] @ e[1]):
        ok &= same(acom(X, J), Z8) and same(com(X, T), Z8)
    X = e[1] @ e[2]
    Xi = -X
    ok &= same(X @ e[1] @ Xi, -e[1]) and same(X @ J @ Xi, -J) and same(X @ T @ Xi, T)
    ok &= all(same(X @ g @ Xi, g) for g in gam[1:])
    ok &= same(X, e[1] @ e[2]) and same(X @ e[2] @ Xi, -e[2])
    vac = fock_vacuum()
    ap = {i: (e[i] + e[i + 3]) for i in (1, 2, 3)}       # 2 a+
    f = {k: ap[k] @ vac for k in (1, 2, 3)}
    g = {1: ap[2] @ ap[3] @ vac, 2: ap[3] @ ap[1] @ vac, 3: ap[1] @ ap[2] @ vac}
    # columns: X = (B1, B2, B3, E1, E2, E3); scale 2a+ -> factors 2 and 4 cancel against the normalisation below
    Phi = np.zeros((8, 6))
    for m in (1, 2, 3):
        Phi[:, m - 1] = g[m] / 4.0
        Phi[:, 3 + m - 1] = f[m] / 2.0
    G6 = np.block([[np.zeros((3, 3)), -np.eye(3)], [np.eye(3), np.zeros((3, 3))]])
    ok &= close(-T @ Phi, Phi @ G6)
    ok &= close(Phi.T @ Phi, np.diag(np.diag(Phi.T @ Phi))) and np.min(np.diag(Phi.T @ Phi)) > 0.5
    for lam in (0.3, 1.7):
        mm = 1.3
        ok &= close(expm(-mm * lam * T) @ Phi, Phi @ expm(mm * lam * G6))
    return ok


@check("y6-2d", "Prop. twosu2: su(2)_PG = span{J, e2 Omega, e3 Omega} equals {Y P+ + (J Y J^-1) P- : Y in so(3)_+}, P+- = (1 +- omega_S)/2 of rank 4; "
       "so(3)_+ and su(2)_PG meet in R J; the Lie algebra they generate (computed by bracket closure) is 6-dimensional and equals "
       "span{Y P+, Y P-}; g = P+ + J P- = expm((pi/2) J P-) is orthogonal, conjugates so(3)_+ onto su(2)_PG and back (g^2 = omega_S); 2g has blade "
       "components of grades 0, 2, 3, 5 only; su(2)_PG contains grade-5 elements, so no grade-preserving (Clifford-group) conjugation maps so(3)_+ onto it")
def _():
    Pp, Pm = (I8 + WS) / 2, (I8 - WS) / 2
    ok = close(Pp @ Pp, Pp) and round(np.trace(Pp)) == 4 and round(np.trace(Pm)) == 4 and close(Pp.T, Pp)
    so3p = [e[2] @ e[3], e[3] @ e[1], e[1] @ e[2]]
    su2 = [J, e[2] @ Omg, e[3] @ Omg]
    Jinv = -J
    tw = [Y @ Pp + (J @ Y @ Jinv) @ Pm for Y in so3p]
    rk = lambda Ms: np.linalg.matrix_rank(np.array([M.ravel() for M in Ms], dtype=float), tol=1e-9)
    ok &= rk(tw) == 3 and rk(tw + su2) == 3
    ok &= rk(so3p) + rk(su2) - rk(so3p + su2) == 1 and rk(so3p + su2 + [J]) == rk(so3p + su2)
    # bracket closure
    span = [M.astype(float) for M in so3p + su2]
    while True:
        new = [com(A, B) for A in span for B in span]
        r0 = rk(span)
        cand = span + new
        if rk(cand) == r0:
            break
        basis = []
        for M in cand:
            if rk(basis + [M]) > len(basis):
                basis.append(M)
        span = basis
    ok &= rk(span) == 6
    target = [Y @ Pp for Y in so3p] + [Y @ Pm for Y in so3p]
    ok &= rk(target) == 6 and rk(span + target) == 6
    gmat = Pp + J @ Pm
    ok &= close(gmat.T @ gmat, np.eye(8)) and close(expm(np.pi / 2 * (J @ Pm)), gmat) and close(gmat @ gmat, WS)
    conj = [gmat @ Y @ gmat.T for Y in so3p]
    ok &= rk(conj + su2) == 3 and rk([gmat @ Y @ gmat.T for Y in su2] + so3p) == 3
    bc = blade_coeffs(2 * gmat)
    ok &= sorted({len(I) for I in bc}) == [0, 2, 3, 5] and all(abs(abs(c) - 1) < 1e-12 for c in bc.values())
    ok &= {len(I) for I in blade_coeffs(e[2] @ Omg)} == {5} and {len(I) for I in blade_coeffs(e[3] @ Omg)} == {5}
    return ok


# ================================================================================================ 3. a temporal b-field and the binding root
r1, r2, r3, kap, lam_, mms = sp.symbols('r1 r2 r3 kappa lambda m', real=True)
Z3, I3 = sp.zeros(3), sp.eye(3)
ETA6 = sp.diag(1, 1, 1, -1, -1, -1)
G6s = sp.Matrix(sp.BlockMatrix([[Z3, -I3], [I3, Z3]]))


def crossm(v):
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def bd(A, B):
    return sp.diag(A, B)


def dag(F, et=ETA6):
    return et.inv() * F.T * et


def GC(C):
    return sp.Matrix(sp.BlockMatrix([[Z3, -C.inv()], [C, Z3]]))


@check("y6-3a", "Prop. binding(i), symbolic in r: b = diag([r]x, 0) = b_l + b_p with b_l = (1/2)diag(R,R) commuting with G, b_p = (1/2)diag(R,-R) "
       "anticommuting with G, [b_l, b_p] = 0, [b, G] = 2 b_p G; in RE eq. (11) form diag(A + C, A - C) this is A = C = R/2")
def _():
    R = crossm([r1, r2, r3])
    b = bd(R, Z3)
    bl, bp = bd(R, R) / 2, bd(R, -R) / 2
    ok = b == bl + bp and bl * G6s == G6s * bl and bp * G6s == -G6s * bp and bl * bp == bp * bl
    ok &= sp.expand(b * G6s - G6s * b - 2 * bp * G6s) == sp.zeros(6)
    ok &= dag(b) == -b
    return ok


@check("y6-3b", "Prop. binding(ii): e^{theta b} G e^{-theta b} = G_{e^{-theta R}} (expm, random r, theta); G_C^2 = -1 for a random C; G_C is eta-self-adjoint for "
       "a random orthogonal C (both determinants) and not for a non-orthogonal C; the self-adjoint part of S = b + kappa G is kappa G with spectrum "
       "+-i kappa, unchanged under the gauge change X = e^{-theta b}Y; [b, G] != 0")
def _():
    ok = True
    G = np.array(G6s.tolist(), dtype=float)
    E6 = np.diag([1, 1, 1, -1, -1, -1.0])
    for _ in range(5):
        r = RNG.normal(size=3)
        R = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        b = np.zeros((6, 6))
        b[:3, :3] = R
        th = RNG.normal()
        lhs = expm(th * b) @ G @ expm(-th * b)
        C = expm(-th * R)
        rhs = np.block([[np.zeros((3, 3)), -np.linalg.inv(C)], [C, np.zeros((3, 3))]])
        ok &= close(lhs, rhs)
        kap0 = 1.7
        S = b + kap0 * G
        Ssa = (S + np.linalg.inv(E6) @ S.T @ E6) / 2
        ok &= close(Ssa, kap0 * G) and close(sorted(np.linalg.eigvals(Ssa).imag), [-kap0] * 3 + [kap0] * 3, 1e-8)
        g = expm(-th * b)                                   # eta-isometry
        Sg = np.linalg.inv(g) @ S @ g                       # constant gauge change (no gdot term for the self-adjoint part)
        Sgsa = (Sg + np.linalg.inv(E6) @ Sg.T @ E6) / 2
        ok &= close(sorted(np.linalg.eigvals(Sgsa).imag), [-kap0] * 3 + [kap0] * 3, 1e-8)
        ok &= np.max(np.abs(b @ G - G @ b)) > 1e-3
    Cr = sp.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 1]])
    ok &= GC(Cr) ** 2 == -sp.eye(6) and dag(GC(Cr)) != GC(Cr)
    Q = (I3 + crossm([1, 2, sp.Rational(1, 2)])) * (I3 - crossm([1, 2, sp.Rational(1, 2)])).inv()
    for Cq in (Q, -Q, Q * sp.diag(1, 1, -1)):
        ok &= sp.simplify(dag(GC(Cq)) - GC(Cq)) == sp.zeros(6)
    return ok


@check("y6-3c", "Prop. binding(iii), symbolic in r = (r1, r2, r3) and kappa: det(lambda - S) = (lambda^2 + kappa^2)(lambda^4 + (2kappa^2 + |r|^2)lambda^2 + kappa^4); "
       "at kappa = 2 and r = (3/7)(2,3,6) (omega = 3) the spectrum of S is +-i{1, 2, 4}, and the line kappa = 2 has its eigenvectors in the plane <(r^,0), (0,r^)>")
def _():
    R = crossm([r1, r2, r3])
    S = bd(R, Z3) + kap * G6s
    p = sp.expand((lam_ * sp.eye(6) - S).det())
    w2 = r1 ** 2 + r2 ** 2 + r3 ** 2
    ok = sp.expand(p - (lam_ ** 2 + kap ** 2) * (lam_ ** 4 + (2 * kap ** 2 + w2) * lam_ ** 2 + kap ** 4)) == 0
    rv = [sp.Rational(6, 7), sp.Rational(9, 7), sp.Rational(18, 7)]
    S0 = (bd(crossm(rv), Z3) + 2 * G6s)
    ev = sorted(float(sp.im(x)) for x in S0.eigenvals(multiple=True))
    ok &= np.allclose(ev, [-4, -2, -1, 1, 2, 4])
    ns = (S0 - 2 * sp.I * sp.eye(6)).nullspace()
    rh = sp.Matrix(rv) / 3
    plane = sp.Matrix.hstack(sp.Matrix.vstack(rh, sp.zeros(3, 1)), sp.Matrix.vstack(sp.zeros(3, 1), rh))
    ok &= len(ns) == 1 and sp.Matrix.hstack(plane, ns[0]).rank() == 2
    return ok


def Mgen(S, m, et=ETA6):
    n = S.shape[0]
    return sp.Matrix(sp.BlockMatrix([[-S, sp.eye(n) / m], [sp.zeros(n), dag(S, et)]]))


def jordan_profile_exact(M, lams):
    """{lambda0: (geometric, algebraic)} over QQ<I> for the listed eigenvalues"""
    out = {}
    n = M.shape[0]
    for l0 in lams:
        A = DomainMatrix.from_Matrix(M - l0 * sp.eye(n)).convert_to(sp.QQ_I)
        geo = n - A.rank()
        P = A
        alg_prev, k = geo, 1
        while True:
            P = P * A
            k += 1
            nk = n - P.rank()
            if nk == alg_prev or k > n:
                break
            alg_prev = nk
        out[l0] = (geo, alg_prev)
    return out


@check("y6-3d", "Prop. binding(iv), exactly over Q(i) at (kappa, omega) = (2,3), (6,5), (4,6) with r in the generic directions (2,3,6)/7, (1,4,8)/9 and m = 1, 3/2: "
       "M of RE Lemma 5.4 has, at each split line s +- omega/2, algebraic multiplicity 2 and geometric multiplicity 1 (a Jordan block), and is semisimple "
       "(2, 2) at kappa; independently, by the criterion of RE Thm 5.5 (eta-pairing h(v,v) of the eigenvector of S^dagger), at 30 random real (kappa, r) "
       "h(v,v) != 0 exactly at the two split lines and = 0 at kappa; numerically ||e^{tau M}|| grows linearly")
def _():
    ok = True
    cases = [(2, 3, (2, 3, 6), 7), (6, 5, (2, 3, 6), 7), (4, 6, (1, 4, 8), 9)]
    for kp, om, dirv, nrm in cases:
        rv = [sp.Rational(om * c, nrm) for c in dirv]
        S = bd(crossm(rv), Z3) + kp * G6s
        s = sp.sqrt(sp.Rational(kp ** 2) + sp.Rational(om ** 2, 4))
        assert s.is_rational
        lines = [s + sp.Rational(om, 2), s - sp.Rational(om, 2)]
        for m in (1, sp.Rational(3, 2)):
            M = Mgen(S, m)
            lams = [sgn * sp.I * L for L in lines for sgn in (1, -1)] + [sp.I * kp, -sp.I * kp]
            prof = jordan_profile_exact(M, lams)
            ok &= all(prof[sgn * sp.I * L] == (1, 2) for L in lines for sgn in (1, -1))
            ok &= prof[sp.I * kp] == (2, 2) and prof[-sp.I * kp] == (2, 2)
    E6 = np.diag([1, 1, 1, -1, -1, -1.0])
    G = np.array(G6s.tolist(), dtype=float)
    for _ in range(30):
        kp = RNG.uniform(0.3, 3)
        r = RNG.normal(size=3) * RNG.uniform(0.2, 3)
        om = np.linalg.norm(r)
        R = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        S = np.zeros((6, 6))
        S[:3, :3] = R
        S += kp * G
        Sd = np.linalg.inv(E6) @ S.T @ E6
        w, V = np.linalg.eig(Sd)
        s = np.sqrt(kp ** 2 + om ** 2 / 4)
        for wi, v in zip(w, V.T):
            v = v / np.linalg.norm(v)
            hv = np.conj(v) @ E6 @ v
            f = abs(wi.imag)
            if abs(f - kp) < 1e-7:
                ok &= abs(hv) < 1e-8
            else:
                ok &= (abs(f - (s + om / 2)) < 1e-7 or abs(f - (s - om / 2)) < 1e-7) and abs(hv) > 1e-3
    S = np.array((bd(crossm([6 / 7, 9 / 7, 18 / 7]), Z3) + 2 * G6s).tolist(), dtype=float)
    Mn = np.block([[-S, np.eye(6)], [np.zeros((6, 6)), np.linalg.inv(E6) @ S.T @ E6]])
    n1_, n2_ = np.linalg.norm(expm(100 * Mn), 2), np.linalg.norm(expm(1000 * Mn), 2)
    ok &= 7 < n2_ / n1_ < 13
    return ok


def Nmat():
    return sum((I8 - e[i] @ e[i + 3]) for i in (1, 2, 3)) // 2


@check("y6-3e", "Prop. binding(v) on R^8 for all three temporal planes (J included): (1/2)(e_je_k - e_{j+3}e_{k+3}) commutes with N, and "
       "(1/2)(e_je_k + e_{j+3}e_{k+3}) has nonzero blocks only between N-eigenspaces differing by exactly 2, in both directions, so e_je_k does not conserve N")
def _():
    N = Nmat()
    d = np.diag(N)
    ok = same(N, np.diag(d))
    for j, k in ((2, 3), (3, 1), (1, 2)):
        P1 = e[j] @ e[k] - e[j + 3] @ e[k + 3]
        P2 = e[j] @ e[k] + e[j + 3] @ e[k + 3]
        ok &= same(com(N, P1), Z8)
        nz = np.nonzero(P2)
        diffs = set((d[nz[0]] - d[nz[1]]).tolist())
        ok &= diffs == {2, -2}
        ok &= not same(com(N, e[j] @ e[k]), Z8)
    return ok


# ================================================================================================ 4. conversions: polar form
@check("y6-4a", "Prop. polar(i) for a random integer C: G_C^2 = -1, G_C^T N_C + N_C G_C = 0 with N_C = diag(C^T C, 1), and z_C(G_C X) = i z_C(X) "
       "symbolically in X, z_C = C B + i E")
def _():
    C = sp.Matrix(RNG.integers(-3, 4, size=(3, 3)).tolist())
    while C.det() == 0:
        C = sp.Matrix(RNG.integers(-3, 4, size=(3, 3)).tolist())
    G = GC(C)
    NC = bd(C.T * C, I3)
    ok = G * G == -sp.eye(6) and G.T * NC + NC * G == sp.zeros(6)
    X = sp.Matrix(sp.symbols('B1 B2 B3 E1 E2 E3'))
    z = lambda Y: C * Y[:3, 0] + sp.I * Y[3:, 0]
    ok &= sp.expand(z(G * X) - sp.I * z(X)) == sp.zeros(3, 1)
    return ok


@check("y6-4b", "Prop. polar(ii): G_C^T N_c + N_c G_C has blocks C^T - c^2 C^{-1}, C - c^2 C^{-T} (exact for 3 random integer C); it vanishes for C = cQ with "
       "Q orthogonal of either determinant and fails for C = c diag(1,2,3)")
def _():
    c = sp.Rational(5, 2)
    Nc = bd(c ** 2 * I3, I3)
    ok = True
    for _ in range(3):
        C = sp.Matrix(RNG.integers(-3, 4, size=(3, 3)).tolist())
        while C.det() == 0:
            C = sp.Matrix(RNG.integers(-3, 4, size=(3, 3)).tolist())
        L = GC(C).T * Nc + Nc * GC(C)
        ok &= L[:3, :3] == Z3 and L[3:, 3:] == Z3
        ok &= L[:3, 3:] == C.T - c ** 2 * C.inv() and L[3:, :3] == C - c ** 2 * C.inv().T
    Q = (I3 + crossm([2, -1, 1])) * (I3 - crossm([2, -1, 1])).inv()
    for Cq in (c * Q, -c * Q, c * Q * sp.diag(1, -1, 1)):
        ok &= sp.simplify(GC(Cq).T * Nc + Nc * GC(Cq)) == sp.zeros(6)
    ok &= GC(c * sp.diag(1, 2, 3)).T * Nc + Nc * GC(c * sp.diag(1, 2, 3)) != sp.zeros(6)
    return ok


@check("y6-4c", "Prop. polar(iii), exact linear solves: for C = cQ the elements (A, A') of so(3)+so(3) commuting with G_C form the 3-dim space {(A, QAQ^-1)}; "
       "the Y in gl(6) preserving eta and E_c form a 6-dim space, exactly the block-diagonal antisymmetric ones; adding [Y, G_c] = 0 leaves 3 dims (diagonal); "
       "Ad_{(Q+, Q-)} G_c = G_{c Q- Q+^-1} for orthogonal Q+, Q- of both determinants")
def _():
    c = 2
    Q = (I3 + crossm([1, sp.Rational(1, 2), 3])) * (I3 - crossm([1, sp.Rational(1, 2), 3])).inv()
    a = sp.symbols('a0:6')
    A, Ap = crossm(a[:3]), crossm(a[3:])
    eqs = list(GC(c * Q) * bd(A, Ap) - bd(A, Ap) * GC(c * Q))
    Mx, _ = sp.linear_eq_to_matrix(eqs, a)
    ns = Mx.nullspace()
    ok = len(ns) == 3
    for v in ns:
        Av, Apv = crossm(list(v[:3])), crossm(list(v[3:]))
        ok &= sp.simplify(Apv - Q * Av * Q.inv()) == Z3
    ys = sp.symbols('y0:36')
    Y = sp.Matrix(6, 6, ys)
    Nc = bd(c ** 2 * I3, I3)
    eq1 = list(Y.T * ETA6 + ETA6 * Y) + list(Y.T * Nc + Nc * Y)
    M1, _ = sp.linear_eq_to_matrix(eq1, ys)
    ns1 = M1.nullspace()
    ok &= len(ns1) == 6
    for v in ns1:
        Yv = sp.Matrix(6, 6, list(v))
        ok &= Yv[:3, 3:] == Z3 and Yv[3:, :3] == Z3 and Yv.T == -Yv
    eq2 = eq1 + list(Y * GC(c * I3) - GC(c * I3) * Y)
    M2, _ = sp.linear_eq_to_matrix(eq2, ys)
    ok &= len(M2.nullspace()) == 3
    Q2 = (I3 + crossm([0, 1, sp.Rational(1, 3)])) * (I3 - crossm([0, 1, sp.Rational(1, 3)])).inv()
    for Qp in (Q, Q * sp.diag(-1, 1, 1)):
        for Qm in (Q2, -Q2):
            g = bd(Qp, Qm)
            ok &= sp.simplify(g * GC(c * I3) * g.inv() - GC(c * Qm * Qp.inv())) == sp.zeros(6)
    return ok


@check("y6-4d", "Prop. polar(iv) and Remark csym(a),(b) for a random integer (non-normal) C: a temporal rotation (Q+, 1) carries G_C to G_{C Q+^-1} and the form "
       "N_C to N_{C Q+^-1}; it fixes C C^T and conjugates C^T C (which changes); conversely, for 20 random C and C' = (C C^T)^{1/2} U' with random "
       "orthogonal U', C^{-1} C' is orthogonal, so C C^T is a complete invariant of the O(3) action of the temporal triple; a spatial rotation fixes C^T C")
def _():
    C = sp.Matrix([[3, 1, 0], [0, 1, 2], [1, -1, 1]])
    Qp = (I3 + crossm([1, -2, sp.Rational(1, 2)])) * (I3 - crossm([1, -2, sp.Rational(1, 2)])).inv()
    g = bd(Qp, I3)
    Cn = C * Qp.inv()
    ok = sp.simplify(g * GC(C) * g.inv() - GC(Cn)) == sp.zeros(6)
    ok &= sp.simplify(g.inv().T * bd(C.T * C, I3) * g.inv() - bd(Cn.T * Cn, I3)) == sp.zeros(6)
    ok &= sp.simplify(Cn * Cn.T - C * C.T) == Z3 and sp.simplify(Cn.T * Cn - C.T * C) != Z3
    ok &= sp.simplify(Cn.T * Cn - Qp * C.T * C * Qp.T) == Z3
    Qm = Qp.T
    ok &= sp.simplify((Qm * C).T * (Qm * C) - C.T * C) == Z3
    from scipy.linalg import sqrtm
    for _ in range(20):
        Cf = RNG.normal(size=(3, 3))
        U, _ = np.linalg.qr(RNG.normal(size=(3, 3)))
        Cp = np.real(sqrtm(Cf @ Cf.T)) @ U
        ok &= close(Cp @ Cp.T, Cf @ Cf.T, 1e-8)
        W = np.linalg.inv(Cf) @ Cp
        ok &= close(W.T @ W, np.eye(3), 1e-7)
    return ok


@check("y6-4e", "Prop. polar(v), symbolic in theta: for B = e1, E = e2, C = R(theta) about e3, q_C = 2i sin theta and E_c = 1; at theta = pi/2, E_0 = |q_C|/2 = 1 = E_6; "
       "for random B, E and an orthogonal R, Re q_C and E_c do not depend on R while Im q_C = 2c (RB).E does")
def _():
    th = sp.symbols('theta', real=True)
    R = sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])
    z = R * sp.Matrix([1, 0, 0]) + sp.I * sp.Matrix([0, 1, 0])
    q = sp.simplify(sp.expand((z.T * z)[0, 0]))
    ok = sp.simplify(q - 2 * sp.I * sp.sin(th)) == 0 and sp.simplify((z.H * z)[0, 0] / 2) == 1
    ok &= sp.Abs(q.subs(th, sp.pi / 2)) / 2 == 1
    c = 3
    B = sp.Matrix([1, -2, 2])
    E = sp.Matrix([0, 1, 4])
    vals = set()
    for v in ([1, 0, 0], [0, 1, 2]):
        Q = (I3 + crossm(v)) * (I3 - crossm(v)).inv()
        zz = c * Q * B + sp.I * E
        qq = sp.expand((zz.T * zz)[0, 0])
        ok &= sp.re(qq) == c ** 2 * B.dot(B) - E.dot(E) and sp.im(qq) == 2 * c * (Q * B).dot(E)
        ok &= sp.simplify((zz.H * zz)[0, 0] - (c ** 2 * B.dot(B) + E.dot(E))) == 0
        vals.add(sp.im(qq))
    ok &= len(vals) == 2
    return ok


@check("y6-4f", "CORRECTED Remark csym(a): the temporal b-field (a flow in SO(3)_+) keeps C C^T and also sgn det C, which O(3)_+ does not: C and C diag(1,1,-1) "
       "have the same C C^T, are related by an orthogonal map of the temporal triple, but by no rotation (det C changes sign); the energy E_C sees the "
       "other factor C^T C, a form on the temporal triple, which the temporal rotation conjugates")
def _():
    C = sp.Matrix([[3, 1, 0], [0, 1, 2], [1, -1, 1]])
    Cr = C * sp.diag(1, 1, -1)
    ok = Cr * Cr.T == C * C.T and sp.sign(Cr.det()) == -sp.sign(C.det())
    Wq = C.inv() * Cr
    ok &= Wq.T * Wq == I3 and Wq.det() == -1
    Qp = (I3 + crossm([1, -2, sp.Rational(1, 2)])) * (I3 - crossm([1, -2, sp.Rational(1, 2)])).inv()
    ok &= sp.sign((C * Qp.inv()).det()) == sp.sign(C.det())
    return ok


# ================================================================================================ 5. the six-momentum read by a clock
def rodr(eh, x, t):
    eh = np.asarray(eh, float)
    x = np.asarray(x, float)
    return np.cos(t) * x + np.sin(t) * np.cross(eh, x) + (1 - np.cos(t)) * eh.dot(x) * eh


@check("y6-5a", "Prop. clock(i)-(iii) by a random test (300 draws of unit n, unit e^, E, p; 720 times): |E| and mu are constant; E_n(t) = A + B cos(t - phi) with "
       "A = (n.e^)(e^.E), B = |n_perp||E_perp|; max/min of m_4^2 = mu^2 - |E|^2 + E_n^2 as stated (min with (|A| - B)^2 iff |A| >= B); max <= mu^2; "
       "constant for n || e^ and for E || e^, not otherwise; rotation about n keeps E_n; E || n stays on R n iff e^ || n")
def _():
    ok = True
    ts = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for _ in range(300):
        n = RNG.normal(size=3)
        n /= np.linalg.norm(n)
        eh = RNG.normal(size=3)
        eh /= np.linalg.norm(eh)
        E = RNG.normal(size=3) * 2
        p = RNG.normal(size=3)
        mu2 = E @ E - p @ p
        Ets = np.array([rodr(eh, E, t) for t in ts])
        ok &= np.allclose(np.sum(Ets ** 2, axis=1), E @ E)
        En = Ets @ n
        A = (n @ eh) * (eh @ E)
        B = np.linalg.norm(n - (n @ eh) * eh) * np.linalg.norm(E - (E @ eh) * eh)
        ok &= abs(En.mean() - A) < 1e-9 and abs((En.max() - En.min()) / 2 - B) < 1e-4
        m4 = mu2 - E @ E + En ** 2
        mx = mu2 - E @ E + (abs(A) + B) ** 2
        mn = mu2 - E @ E + ((abs(A) - B) ** 2 if abs(A) >= B else 0.0)
        ok &= abs(m4.max() - mx) < 1e-3 and abs(m4.min() - mn) < 1e-3 and mx <= mu2 + 1e-12
    for _ in range(20):
        n = RNG.normal(size=3)
        n /= np.linalg.norm(n)
        E = RNG.normal(size=3)
        ok &= np.ptp([n @ rodr(n, E, t) for t in ts]) < 1e-12                 # e^ = n
        eh = RNG.normal(size=3)
        eh /= np.linalg.norm(eh)
        ok &= np.ptp([n @ rodr(eh, 2.0 * eh, t) for t in ts]) < 1e-12          # E || e^
        ok &= np.ptp([(n @ rodr(eh, E, t)) ** 2 for t in ts]) > 1e-6          # generic
        Es = 3.0 * n
        ok &= max(np.linalg.norm(np.cross(rodr(eh, Es, t), n)) for t in ts) > 1e-3
        ok &= max(np.linalg.norm(np.cross(rodr(n, Es, t), n)) for t in ts) < 1e-12
    return ok


@check("y6-5b", "Prop. clock(iv), exact (sympy): E = (2,3,6), p = (0,0,3), mu^2 = 40; about e3 read by e1: E_n = 2cos t - 3 sin t, m_4^2 in [-9, 4] (and -5 at t = 0, the "
       "value of TR Prop. 3.13); with W^ = (1,1,1)/sqrt3 (TR, RE Prop. 2.10) about e3: m_4^2 in [35/3 - 4 sqrt26, 35/3 + 4 sqrt26]; about W^ read by W^: m_4^2 = 94/3")
def _():
    t = sp.symbols('t', real=True)
    E = sp.Matrix([2, 3, 6])
    mu2 = E.dot(E) - 9
    ok = mu2 == 40

    def rot(eh, x):
        eh = sp.Matrix(eh)
        return sp.cos(t) * x + sp.sin(t) * eh.cross(x) + (1 - sp.cos(t)) * eh.dot(x) * eh
    Et = rot([0, 0, 1], E)
    En = sp.simplify(Et[0])
    ok &= sp.simplify(En - (2 * sp.cos(t) - 3 * sp.sin(t))) == 0
    m4 = mu2 - 49 + En ** 2
    ok &= sp.simplify(En ** 2 + (3 * sp.cos(t) + 2 * sp.sin(t)) ** 2 - 13) == 0          # so 0 <= E_n^2 <= 13
    t0, t1 = sp.atan(sp.Rational(2, 3)), sp.atan(sp.Rational(-3, 2))                     # E_n = 0, resp. 3cos + 2sin = 0
    ok &= sp.simplify(m4.subs(t, t0)) == -9 and sp.simplify(m4.subs(t, t1)) == 4 and m4.subs(t, 0) == -5
    W = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    EW = sp.simplify(W.dot(Et))
    m4W = sp.expand(mu2 - 49 + EW ** 2)
    f = sp.lambdify(t, m4W)
    grid = np.linspace(0, 2 * np.pi, 200001)
    fv = f(grid)
    ok &= abs(fv.min() - float(sp.Rational(35, 3) - 4 * sp.sqrt(26))) < 1e-8 and abs(fv.max() - float(sp.Rational(35, 3) + 4 * sp.sqrt(26))) < 1e-8
    EtW = rot(list(W), E)
    ok &= sp.simplify(mu2 - 49 + W.dot(EtW) ** 2 - sp.Rational(94, 3)) == 0
    return ok


@check("y6-5d", "CORRECTED discussion after Prop. clock: read passively, the clock n(t) = R(-Omega t) n stays on the circle {x : x.e^ = n.e^} of radius |n_perp| about e^, "
       "and E_n(t) = n(t).E; it is a great circle only for n perpendicular to e^ (for W^ about e3 the radius is sqrt(2/3))")
def _():
    ok = True
    ts = np.linspace(0, 2 * np.pi, 361)
    for _ in range(20):
        n = RNG.normal(size=3)
        n /= np.linalg.norm(n)
        eh = RNG.normal(size=3)
        eh /= np.linalg.norm(eh)
        E = RNG.normal(size=3)
        nt = np.array([rodr(eh, n, -t) for t in ts])
        ok &= np.allclose(nt @ eh, n @ eh) and np.allclose([nt[i] @ E for i in range(len(ts))], [n @ rodr(eh, E, t) for t in ts])
        centre = (n @ eh) * eh
        rad = np.linalg.norm(nt - centre, axis=1)
        ok &= np.allclose(rad, np.sqrt(1 - (n @ eh) ** 2))
    W = np.ones(3) / np.sqrt(3)
    ok &= abs(np.sqrt(1 - W[2] ** 2) - np.sqrt(2 / 3)) < 1e-12 and np.sqrt(2 / 3) < 1
    return ok


# ================================================================================================ 6. between constituents (n = 2)
I64 = np.eye(64, dtype=np.int64)
Z64 = 0 * I64
E2 = {}
for a in range(1, 7):
    E2[(1, a)] = np.kron(e[a], I8)
    E2[(2, a)] = np.kron(OM, e[a])
Cp = -0.5 * sum(E2[(1, i)] @ E2[(2, i)] for i in (1, 2, 3))
Cm = 0.5 * sum(E2[(1, a)] @ E2[(2, a)] for a in (4, 5, 6))
Cb = Cp + Cm
Cx = Cp - Cm
Ntot = sum((I64 - E2[(k, i)] @ E2[(k, i + 3)]) for k in (1, 2) for i in (1, 2, 3)) // 2
N1 = sum((I64 - E2[(1, i)] @ E2[(1, i + 3)]) for i in (1, 2, 3)) // 2
N2 = Ntot - N1


@check("y6-6a", "Prop. copyrot(i): C_+ and C_- are antisymmetric and commute; the spectrum of C_+ (and of C_-) is +-i/2 with multiplicity 24 each and +-3i/2 "
       "with multiplicity 8 each (numerical eigenvalues); expm(2 pi C_+) = expm(2 pi C_-) = -1 and expm(2 pi C_b) = +1")
def _():
    ok = close(Cp.T, -Cp) and close(Cm.T, -Cm) and close(com(Cp, Cm), np.zeros((64, 64)))
    for C in (Cp, Cm):
        ev = np.round(2 * np.linalg.eigvals(C).imag).astype(int)
        cnt = {v: int(np.sum(ev == v)) for v in set(ev.tolist())}
        ok &= cnt == {1: 24, -1: 24, 3: 8, -3: 8} and np.max(np.abs(np.linalg.eigvals(C).real)) < 1e-9
        ok &= close(expm(2 * np.pi * C), -np.eye(64), 1e-8)
    ok &= close(expm(2 * np.pi * Cb), np.eye(64), 1e-8)
    return ok


def vec_action(X):
    """matrix of v -> [X, v] on the twelve generators (basis order E_(1,1..6), E_(2,1..6)), read off by the trace form"""
    basis = [E2[(1, a)] for a in range(1, 7)] + [E2[(2, a)] for a in range(1, 7)]
    M = np.zeros((12, 12))
    for j, v in enumerate(basis):
        w = com(X, v)
        for i, u in enumerate(basis):
            M[i, j] = np.trace(u.T @ w) / 64.0
        ok = close(sum(M[i, j] * basis[i] for i in range(12)), w)
        assert ok
    return M


@check("y6-6b", "Prop. copyrot(ii), from the spin action v -> [C, v] on the 12 generators: V_+ = eps(x)P_+ (e^(1)_i -> e^(2)_i, temporal only), V_- = eps(x)P_-; "
       "V_+ and V_- do not commute with G_2 = 1(x)G, V_+ + V_- = eps(x)1 does, V_+ - V_- = eps(x)eta is antisymmetric (compact), lies in so(6,6) "
       "and anticommutes with G_2")
def _():
    Vp, Vm = vec_action(Cp), vec_action(Cm)
    epsm = np.array([[0, -1], [1, 0.0]])
    Pp = np.diag([1, 1, 1, 0, 0, 0.0])
    Pm = np.eye(6) - Pp
    ok = close(Vp, np.kron(epsm, Pp)) and close(Vm, np.kron(epsm, Pm))
    G2 = np.kron(np.eye(2), np.array(G6s.tolist(), dtype=float))
    E12 = np.kron(np.eye(2), np.diag([1, 1, 1, -1, -1, -1.0]))
    ok &= not close(com(Vp, G2), np.zeros((12, 12))) and not close(com(Vm, G2), np.zeros((12, 12)))
    ok &= close(com(Vp + Vm, G2), np.zeros((12, 12))) and close(acom(Vp - Vm, G2), np.zeros((12, 12)))
    D = Vp - Vm
    ok &= close(D, np.kron(epsm, np.diag([1, 1, 1, -1, -1, -1.0]))) and close(D.T, -D) and close(D.T @ E12 + E12 @ D, np.zeros((12, 12)))
    return ok


@check("y6-6c", "Prop. copyrot(iii): C_x = C_+ - C_- equals -sum_i (a+_{1i}a+_{2i} + a-_{1i}a-_{2i}) and changes N by +-2 only, C_b commutes with N; from the "
       "Fock vacuum, at theta = 0.9 and 2.3 (expm), the weight of e^{theta C_+}|0> in N = 2k is C(3,k) cos^{2(3-k)}(theta/2) sin^{2k}(theta/2), zero in odd N, "
       "N^(1) = N^(2) on the state, <N> = 3(1 - cos theta); at theta = pi the state has N = 6")
def _():
    ap = {(k, i): (E2[(k, i)] + E2[(k, i + 3)]) / 2 for k in (1, 2) for i in (1, 2, 3)}
    am = {(k, i): (E2[(k, i)] - E2[(k, i + 3)]) / 2 for k in (1, 2) for i in (1, 2, 3)}
    ok = close(Cx, -sum(ap[(1, i)] @ ap[(2, i)] + am[(1, i)] @ am[(2, i)] for i in (1, 2, 3)))
    ok &= close(com(Cb, Ntot), np.zeros((64, 64))) and not close(com(Cp, Ntot), np.zeros((64, 64)))
    d = np.diag(Ntot)
    nz = np.nonzero(np.abs(Cx) > 1e-12)
    ok &= set((d[nz[0]] - d[nz[1]]).tolist()) == {2, -2}
    vac = np.zeros(64)
    vac[int(np.argmin(d))] = 1.0
    ok &= d[int(np.argmin(d))] == 0 and close(Cb @ vac, np.zeros(64))
    for th in (0.9, 2.3):
        psi = expm(th * Cp) @ vac
        for k in range(4):
            w = np.sum(psi[d == 2 * k] ** 2)
            ok &= abs(w - comb(3, k) * np.cos(th / 2) ** (2 * (3 - k)) * np.sin(th / 2) ** (2 * k)) < 1e-10
        ok &= np.sum(psi[d % 2 == 1] ** 2) < 1e-20
        ok &= close((N1 - N2) @ psi, np.zeros(64)) and abs(psi @ Ntot @ psi - 3 * (1 - np.cos(th))) < 1e-10
    psi = expm(np.pi * Cp) @ vac
    ok &= abs(np.sum(psi[d == 6] ** 2) - 1) < 1e-10
    return ok


@check("y6-6d", "Prop. transfer(i),(iii) by expm at theta = 0.7: Ad(e^{theta C_+}) E_(1,1) = cos theta E_(1,1) + sin theta E_(2,1), E_(2,1) -> cos theta E_(2,1) - sin theta E_(1,1); "
       "H_g keeps spectrum {+-M} (32 each); C_- commutes with H_g; Ad(e^{theta C_+}) u_1 = cos theta u_1 + sin theta omega(x)e1, and omega(x)e1 is trace-orthogonal "
       "to Cl(3,3)(x)1 + 1(x)Cl(3,3); spec(m1 u1 + m2 u2) = {+-m1 +- m2} with each joint sign on 16 dimensions")
def _():
    th = 0.7
    g = expm(th * Cp)
    gi = expm(-th * Cp)
    ok = close(g @ E2[(1, 1)] @ gi, np.cos(th) * E2[(1, 1)] + np.sin(th) * E2[(2, 1)])
    ok &= close(g @ E2[(2, 1)] @ gi, np.cos(th) * E2[(2, 1)] - np.sin(th) * E2[(1, 1)])
    m1, m2 = 1.3, 0.4
    Hg = m1 * E2[(1, 1)] + m2 * E2[(2, 1)]
    M = np.hypot(m1, m2)
    ev = np.linalg.eigvalsh(g @ Hg @ gi)
    ok &= np.allclose(ev[:32], -M) and np.allclose(ev[32:], M) and close(com(Cm, Hg), np.zeros((64, 64)))
    u1, u2 = np.kron(e[1], I8), np.kron(I8, e[1])
    ok &= close(g @ u1 @ gi, np.cos(th) * u1 + np.sin(th) * np.kron(OM, e[1]))
    W = np.kron(OM, e[1])
    for r in range(7):
        for I in itertools.combinations(range(1, 7), r):
            B = I8.copy()
            for a in I:
                B = B @ e[a]
            ok &= np.trace(W.T @ np.kron(B, I8)) == 0 and np.trace(W.T @ np.kron(I8, B)) == 0
    Hu = m1 * u1 + m2 * u2
    ev = np.round(np.linalg.eigvalsh(Hu), 9)
    tgt = sorted([s1 * m1 + s2 * m2 for s1 in (1, -1) for s2 in (1, -1)] * 16)
    ok &= np.allclose(sorted(ev), tgt)
    return ok


@check("y6-6f", "CORRECTED Prop. transfer(ii) and Remark transfer: the flip P(x(x)y) = y(x)x is orthogonal and conjugates u_1 = e1(x)1 into u_2 = 1(x)e1, and "
       "H_u(m1, m2) into H_u(m2, m1): the full exchange (M, 0) -> (0, M) is a conjugation at fixed spectrum {+-M}; the rotated family cos t u1 + sin t u2 has "
       "spectrum {+-1} at t = pi/2 and squares to 1 + sin2t u1u2, so it is conjugate to u_1 iff sin 2t = 0; at fixed spectrum {|m1|, |m2|} is fixed, so no "
       "continuous transfer exists")
def _():
    P = np.zeros((64, 64), dtype=np.int64)
    for i in range(8):
        for j in range(8):
            P[j * 8 + i, i * 8 + j] = 1
    u1, u2 = np.kron(e[1], I8), np.kron(I8, e[1])
    ok = same(P @ P.T, I64) and same(P @ u1 @ P.T, u2) and same(P @ u2 @ P.T, u1)
    m1, m2 = 3, 1
    ok &= same(P @ (m1 * u1 + m2 * u2) @ P.T, m2 * u1 + m1 * u2)
    for t in (np.pi / 2, 0.0, np.pi):
        H = np.cos(t) * u1 + np.sin(t) * u2
        ok &= np.allclose(sorted(np.linalg.eigvalsh(H)), [-1] * 32 + [1] * 32)
    for t in (0.3, 1.1, 2.0):
        H = np.cos(t) * u1 + np.sin(t) * u2
        ok &= close(H @ H, np.eye(64) + np.sin(2 * t) * (u1 @ u2)) and not np.allclose(sorted(np.abs(np.linalg.eigvalsh(H))), 1)
    return ok


@check("y6-6e", "Prop. copyrot(iv), symbolic in (kappa, beta): S = kappa G_2 + beta V_+ has det(lambda - S) = (lambda^4 + (2kappa^2 + beta^2)lambda^2 + kappa^4)^3, self-adjoint part "
       "kappa G_2 (rates kappa); exactly over Q(i) at (kappa, beta) = (2,3) and (4,6), M has a Jordan block at each line s +- beta/2 (geometric 3 < algebraic 6)")
def _():
    bt = sp.symbols('beta', real=True)
    epsm = sp.Matrix([[0, -1], [1, 0]])
    Vp = sp.kronecker_product(epsm, sp.diag(1, 1, 1, 0, 0, 0))
    G2 = sp.kronecker_product(sp.eye(2), G6s)
    E12 = sp.kronecker_product(sp.eye(2), ETA6)
    S = kap * G2 + bt * Vp
    p = sp.expand((lam_ * sp.eye(12) - S).det(method='berkowitz'))
    ok = sp.expand(p - (lam_ ** 4 + (2 * kap ** 2 + bt ** 2) * lam_ ** 2 + kap ** 4) ** 3) == 0
    ok &= (S + dag(S, E12)) / 2 == kap * G2
    for kp, b in ((2, 3), (4, 6)):
        s = sp.sqrt(sp.Rational(kp ** 2) + sp.Rational(b ** 2, 4))
        lines = [s + sp.Rational(b, 2), s - sp.Rational(b, 2)]
        M = Mgen(kp * G2 + b * Vp, 1, E12)
        prof = jordan_profile_exact(M, [sgn * sp.I * L for L in lines for sgn in (1, -1)])
        ok &= all(v == (3, 6) for v in prof.values())
    return ok


# ================================================================================================ 7. up the ladder
@check("y6-7a", "Thm ladder(ii) for n = 1..6 (signed permutations, up to 8^6 = 262144): T_n^T = (-1)^{3n(3n-1)/2} T_n and S_n^T = (-1)^{3n(3n+1)/2} S_n, giving "
       "(anti,sym), (anti,anti), (sym,anti), (sym,sym), (anti,sym), (anti,anti) for n = 1..6: period four, so the antisymmetric invariant kernel is unique "
       "exactly at odd n (T_n at n = 1, 5; S_n at n = 3), a plane at n = 2, 6 and absent at n = 4; T_n S_n = +-Gamma_n")
def _():
    ok = True
    pattern = []
    for n in range(1, 7):
        gens = tower(n)
        dim = 8 ** n
        Tn = prod([gens[(k, i)] for k in range(1, n + 1) for i in (1, 2, 3)], dim)
        Sn = prod([gens[(k, a)] for k in range(1, n + 1) for a in (4, 5, 6)], dim)
        Gn = prod([gens[(k, a)] for k in range(1, n + 1) for a in range(1, 7)], dim)
        st = 1 if Tn.T.same(Tn) else (-1 if Tn.T.same(Tn, -1) else 0)
        ss = 1 if Sn.T.same(Sn) else (-1 if Sn.T.same(Sn, -1) else 0)
        m = 3 * n
        ok &= st == (-1) ** (m * (m - 1) // 2) and ss == (-1) ** (m * (m + 1) // 2)
        TS = Tn @ Sn
        ok &= TS.same(Gn) or TS.same(Gn, -1)
        pattern.append((st, ss))
    ok &= pattern == [(-1, 1), (-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    return ok


@check("y6-7b", "Thm ladder(i) for n = 1..4: T_n E_A T_n^-1 = (-1)^{3n-1} E_A^T and S_n E_A S_n^-1 = (-1)^{3n} E_A^T for all 6n generators (hence Y^T M + M Y = 0); "
       "the commutant of all bivectors, computed by the Cartan reduction (the 3n pole operators are diagonal with all 8^n joint signs distinct) and the "
       "connected components of the bivector permutation graph, has dimension 2 = span{1, Gamma_n}: the invariant forms are span{T_n, S_n}")
def _():
    ok = True
    for n in range(1, 5):
        gens = tower(n)
        dim = 8 ** n
        Tn = prod([gens[(k, i)] for k in range(1, n + 1) for i in (1, 2, 3)], dim)
        Sn = prod([gens[(k, a)] for k in range(1, n + 1) for a in (4, 5, 6)], dim)
        sT, sS = (-1) ** (3 * n - 1), (-1) ** (3 * n)
        for A, EA in gens.items():
            ok &= (Tn @ EA @ Tn.T).same(EA.T, sT) and (Sn @ EA @ Sn.T).same(EA.T, sS)
        poles = [gens[(k, i)] @ gens[(k, i + 3)] for k in range(1, n + 1) for i in (1, 2, 3)]
        ok &= all(np.array_equal(P.p, np.arange(dim)) for P in poles)
        sig = np.stack([P.s for P in poles], axis=1)
        ok &= len({tuple(r) for r in sig.tolist()}) == dim
        rows, cols = [], []
        keys = list(gens)
        for x, y in itertools.combinations(keys, 2):
            Y = gens[x] @ gens[y]
            rows.append(np.arange(dim))
            cols.append(Y.p)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim))
        ncomp, lab = connected_components(graph, directed=False)
        ok &= ncomp == 2
        Gn = prod([gens[k] for k in keys], dim)
        ok &= np.array_equal(Gn.p, np.arange(dim)) and all(len(set(Gn.s[lab == c].tolist())) == 1 for c in range(2))
    return ok


def temporal_group_counts(n):
    """(dim commutant, sym, antisym) of so(3n)_+ by the character sums over the finite group {+- e_I : I temporal, |I| even}"""
    gens = tower(n)
    dim = 8 ** n
    temps = [gens[(k, i)] for k in range(1, n + 1) for i in (1, 2, 3)]
    s_chi2, s_chisq = 0, 0
    K = 0
    for r in range(0, 3 * n + 1, 2):
        for I in itertools.combinations(range(3 * n), r):
            B = prod([temps[t] for t in I], dim)
            tr = B.trace()
            B2 = B @ B
            assert np.array_equal(B2.p, np.arange(dim)) and len(set(B2.s.tolist())) == 1
            s_chi2 += tr * tr
            s_chisq += int(B2.s[0]) * dim
            K += 1
    order = 2 * K                       # the group has +- each blade
    tot = Fraction(2 * s_chi2, order)
    sym = Fraction(2 * s_chi2 + 2 * s_chisq, 2 * order)
    asym = Fraction(2 * s_chi2 - 2 * s_chisq, 2 * order)
    return tot, sym, asym


@check("y6-7c", "Thm ladder(iv) by character (Frobenius-Schur) sums over the finite group {+- e_I : I temporal, |I| even}, which spans the lift of so(3n)_+: "
       "commutant dimension (1/|G|) sum chi^2 = 16, 128, 1024 and (sym, antisym) = (1/|G|) sum (chi^2 +- chi(g^2))/2 = (6,10), (64,64), (528,496) at n = 1, 2, 3; "
       "at n = 1 also by an exact linear solve (16 = 6 + 10); T_n and S_n are among them")
def _():
    ok = True
    exp = {1: (16, 6, 10), 2: (128, 64, 64), 3: (1024, 528, 496)}
    for n in (1, 2, 3):
        ok &= temporal_group_counts(n) == exp[n]
    so3p = [e[2] @ e[3], e[3] @ e[1], e[1] @ e[2]]
    L = np.vstack([np.kron(Y, I8) - np.kron(I8, Y.T) for Y in so3p])
    ok &= 64 - zrank(L) == 16
    Lsym = []
    for i in range(8):
        for j in range(i + 1, 8):
            row = np.zeros(64, dtype=np.int64)
            row[i * 8 + j], row[j * 8 + i] = 1, -1
            Lsym.append(row)
    ok &= 64 - zrank(np.vstack([L] + [np.array(Lsym)])) == 6
    Lanti = []
    for i in range(8):
        for j in range(i, 8):
            row = np.zeros(64, dtype=np.int64)
            row[i * 8 + j] += 1
            row[j * 8 + i] += 1
            Lanti.append(row)
    ok &= 64 - zrank(np.vstack([L] + [np.array(Lanti)])) == 10
    ok &= all(same(Y.T @ T + T @ Y, Z8) and same(Y.T @ WS + WS @ Y, Z8) for Y in so3p)
    return ok


def laurent_mul(f, g):
    h = {}
    for a, ca in f.items():
        for b, cb in g.items():
            k = (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])
            h[k] = h.get(k, 0) + ca * cb
    return {k: v for k, v in h.items() if v}


def su4_invariants(f):
    """dimension of SU(4)-invariants of a virtual character f (Laurent polynomial in x1..x4, x1x2x3x4 = 1) by Weyl integration"""
    # Delta = prod_{i<j} (x_i - x_j); |Delta|^2 = Delta(x) Delta(1/x)
    D = {(0, 0, 0, 0): 1}
    for i, j in itertools.combinations(range(4), 2):
        ei = [0] * 4
        ej = [0] * 4
        ei[i] = 1
        ej[j] = 1
        D = laurent_mul(D, {tuple(ei): 1, tuple(ej): -1})
    Dbar = {tuple(-x for x in k): v for k, v in D.items()}
    D2 = laurent_mul(D, Dbar)
    norm = lambda k: (k[0] - k[3], k[1] - k[3], k[2] - k[3])
    fn, dn = {}, {}
    for k, v in f.items():
        fn[norm(k)] = fn.get(norm(k), 0) + v
    for k, v in D2.items():
        dn[norm(k)] = dn.get(norm(k), 0) + v
    tot = sum(v * dn.get(tuple(-x for x in k), 0) for k, v in fn.items())
    assert tot % 24 == 0
    return tot // 24


def laurent_pow(f, n):
    out = {(0, 0, 0, 0): 1}
    for _ in range(n):
        out = laurent_mul(out, f)
    return out


@check("y6-7d", "Thm ladder(v) by an independent method: Spin(3,3) = SL(4,R) with S = 4 + 4*, so the diagonal invariant forms on S^(x)n are the SU(4)-invariants "
       "of Sym^2 and Lambda^2 of (4 + 4*)^(x)n, computed by the Weyl integration formula on the maximal torus: (sym, antisym) = (1,1), (10,4), (84,84), totals "
       "2, 14, 168 at n = 1, 2, 3; at n = 2 also an exact solve with the Cartan-support reduction (216 unknowns) gives 14 = 10 + 4")
def _():
    chi = {}
    chi2 = {}
    for i in range(4):
        k = [0] * 4
        k[i] = 1
        chi[tuple(k)] = 1
        chi[tuple(-x for x in k)] = 1
        k2 = [0] * 4
        k2[i] = 2
        chi2[tuple(k2)] = 1
        chi2[tuple(-x for x in k2)] = 1
    ok = True
    exp = {1: (1, 1), 2: (10, 4), 3: (84, 84)}
    for n in (1, 2, 3):
        cn = laurent_pow(chi, n)
        sq = laurent_mul(cn, cn)
        c2n = laurent_pow(chi2, n)
        sym = {k: sq.get(k, 0) + c2n.get(k, 0) for k in set(sq) | set(c2n)}
        asym = {k: sq.get(k, 0) - c2n.get(k, 0) for k in set(sq) | set(c2n)}
        s_, a_ = su4_invariants(sym), su4_invariants(asym)
        ok &= s_ % 2 == 0 and a_ % 2 == 0 and (s_ // 2, a_ // 2) == exp[n]
    # n = 2 exact solve, diagonal so(3,3): Y (x) 1 + 1 (x) Y
    diag_biv = [np.kron(BIV[k], I8) + np.kron(I8, BIV[k]) for k in BIV]
    cart = [np.kron(e[i] @ e[i + 3], I8) + np.kron(I8, e[i] @ e[i + 3]) for i in (1, 2, 3)]
    # the diagonal generators are not signed permutations; solve directly on the opposite-weight support
    W = np.stack([np.diag(c) for c in cart], axis=1)
    pairs = [(i, j) for i in range(64) for j in range(64) if np.array_equal(W[i], -W[j])]
    ok &= len(pairs) == 216
    rows = []
    for Y in diag_biv:
        A = np.zeros((64 * 64, len(pairs)), dtype=np.int64)
        for t, (i, j) in enumerate(pairs):
            Mt = np.zeros((64, 64), dtype=np.int64)
            Mt[i, j] = 1
            A[:, t] = (Y.T @ Mt + Mt @ Y).ravel()
        A = A[np.any(A != 0, axis=1)]
        rows.append(A)
    A = np.unique(np.vstack(rows), axis=0)
    nul = len(pairs) - zrank(A)
    ok &= nul == 14
    # symmetric part: add M = M^T constraints
    idx = {p: t for t, p in enumerate(pairs)}
    symc, antc = [], []
    for (i, j), t in idx.items():
        if i < j:
            r = np.zeros(len(pairs), dtype=np.int64)
            r[t], r[idx[(j, i)]] = 1, -1
            symc.append(r)
            r2 = np.zeros(len(pairs), dtype=np.int64)
            r2[t], r2[idx[(j, i)]] = 1, 1
            antc.append(r2)
        elif i == j:
            r2 = np.zeros(len(pairs), dtype=np.int64)
            r2[t] = 1
            antc.append(r2)
    ok &= len(pairs) - zrank(np.vstack([A, np.array(symc)])) == 10
    ok &= len(pairs) - zrank(np.vstack([A, np.array(antc)])) == 4
    return ok


@check("y6-7e", "Thm ladder(iii) at n = 2: T_2 = -omega_S(x)T and S_2 = -T(x)omega_S; the forms invariant under so(3,3)^(1) + so(3,3)^(2) (30 copy bivectors, exact "
       "Cartan-support solve, 64 unknowns) are a 4-dim space spanned by M1(x)M2, M_k in {T, omega_S}; of these, exactly the antisymmetric omega_S(x)T and "
       "T(x)omega_S are invariant under all 66 bivectors of so(6,6); T(x)1 is invariant under the compact bivectors of copy 2 but not under its boosts")
def _():
    keys = list(E2)
    Tn = E2[(1, 1)] @ E2[(1, 2)] @ E2[(1, 3)] @ E2[(2, 1)] @ E2[(2, 2)] @ E2[(2, 3)]
    Sn = E2[(1, 4)] @ E2[(1, 5)] @ E2[(1, 6)] @ E2[(2, 4)] @ E2[(2, 5)] @ E2[(2, 6)]
    ok = same(Tn, -np.kron(WS, T)) and same(Sn, -np.kron(T, WS))
    copyb = [E2[x] @ E2[y] for x, y in itertools.combinations(keys, 2) if x[0] == y[0]]
    allb = [E2[x] @ E2[y] for x, y in itertools.combinations(keys, 2)]
    ok &= len(copyb) == 30 and len(allb) == 66
    cart = [E2[(k, i)] @ E2[(k, i + 3)] for k in (1, 2) for i in (1, 2, 3)]
    pairs, ns = invariant_forms_cartan(copyb, cart, 64)
    ok &= len(pairs) == 64 and len(ns) == 4
    prods = {(A, B): np.kron(Ma, Mb) for A, Ma in (('T', T), ('S', WS)) for B, Mb in (('T', T), ('S', WS))}
    forms = forms_from_nullspace(pairs, ns, 64)
    span = sp.Matrix([[sp.Rational(x) for x in F.ravel()] for F in forms])
    tgt = sp.Matrix([[int(x) for x in P.ravel()] for P in prods.values()])
    ok &= tgt.rank() == 4 and span.col_join(tgt).rank() == 4
    inv_all = {k: all(same(Y.T @ P + P @ Y, Z64) for Y in allb) for k, P in prods.items()}
    ok &= inv_all == {('T', 'T'): False, ('T', 'S'): True, ('S', 'T'): True, ('S', 'S'): False}
    ok &= same(prods[('T', 'T')].T, prods[('T', 'T')]) and same(prods[('S', 'S')].T, prods[('S', 'S')])
    ok &= same(prods[('T', 'S')].T, -prods[('T', 'S')]) and same(prods[('S', 'T')].T, -prods[('S', 'T')])
    T1 = np.kron(T, I8)
    c2c = [E2[(2, a)] @ E2[(2, b)] for a, b in COMPACT]
    c2b = [E2[(2, a)] @ E2[(2, b)] for a, b in BOOST]
    ok &= all(same(Y.T @ T1 + T1 @ Y, Z64) for Y in c2c) and not any(same(Y.T @ T1 + T1 @ Y, Z64) for Y in c2b)
    return ok


# ------------------------------------------------------------------------------------------------ summary
print(f"# runtime {time.time() - T0:.1f}s")
if FAILED:
    print("FAILED:", ", ".join(FAILED))
    sys.exit(1)
sys.exit(0)
