#!/usr/bin/env python3
"""Track e5 -- The closure of the force law at two constituents (n = 2).

    python3 investigations/emergence/verify/e5_closure_two.py

Prints one 'PASS e5-<id>: ...' line per verified claim of sections/e5_closure_two.tex and exits 0 only if every check
passes.  Runnable from any directory; numpy, sympy and scipy only.

Conventions (tex/s02_arena.tex eq. (1)-(2), tex/sA_realisation.tex):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1); on R^8: e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1,
  e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep; omega = e1...e6.  Tower: E_(1,a) = e_a (x) 1, E_(2,a) = omega (x) e_a on R^64.
  G e_i = e_{i+3}, G e_{i+3} = -e_i; z = x + i y for X = sum x_j e_j + y_j e_{j+3} (G acts as i).
  V_2 = V (+) V with basis e^(1)_1..6, e^(2)_1..6; eta_2 = eta (+) eta, G_2 = G (+) G.
  Spin lift of a vector-level M in so(eta_n):  S(M) = 1/4 sum_{a,b} (M eta)_{ab} E_a E_b,  so that [S(M), E_c] = sum_a M_{ac} E_a.

Exactness: every algebraic statement is checked in exact arithmetic: Clifford generators are integer (signed
permutation) matrices, spin lifts are integer matrices divided by 4, vector-level Lie algebra statements use exact
rational null spaces (sympy DomainMatrix over QQ).  Floating point is used only for (a) eigenvalue listings of Casimir
operators (tolerance 1e-9), (b) the pair invariant M_pair and its derivatives (tolerances stated), (c) the numerical
integrations of the closure (DOP853, rtol = atol = 1e-12; tolerances stated per check).
Runtime: about half a minute.
"""
import itertools
import sys
import time
from fractions import Fraction

import numpy as np
import scipy.linalg as sla
import sympy as sp
from scipy.integrate import solve_ivp
from sympy.polys.matrices import DomainMatrix

T0 = time.time()
FAILS = []
NPASS = [0]


def report(cid, ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {cid}: {text}")
    if ok:
        NPASS[0] += 1
    else:
        FAILS.append(cid)


def claim(cid, text):
    def deco(fn):
        try:
            ok = bool(fn())
        except Exception as exc:  # reported as a failure
            import traceback
            traceback.print_exc()
            print(f"   exception in {cid}: {exc!r}")
            ok = False
        report(cid, ok, text)
        return fn
    return deco


# ============================================================================== Clifford realisation
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def kr(*ms):
    out = np.array([[1]], dtype=np.int64)
    for m in ms:
        out = np.kron(out, m)
    return out


e = {1: kr(sx, I2, I2), 2: kr(sz, sx, I2), 3: kr(sz, sz, sx),
     4: kr(ep, I2, I2), 5: kr(sz, ep, I2), 6: kr(sz, sz, ep)}
ETA1 = [1, 1, 1, -1, -1, -1]
I8 = np.eye(8, dtype=np.int64)
omega = e[1] @ e[2] @ e[3] @ e[4] @ e[5] @ e[6]
I64 = np.eye(64, dtype=np.int64)

# graded tower at n = 2:  E[(k,a)]
E = {}
for a in range(1, 7):
    E[(1, a)] = np.kron(e[a], I8)
    E[(2, a)] = np.kron(omega, e[a])
GEN = [E[(k, a)] for k in (1, 2) for a in range(1, 7)]       # index A = 6(k-1) + (a-1)
ETA2 = np.array(ETA1 + ETA1, dtype=np.int64)


def comm(A, B):
    return A @ B - B @ A


def iszero(A):
    return not np.any(A)


# ============================================================================== vector level
def G_one():
    G = np.zeros((6, 6), dtype=np.int64)
    for i in range(3):
        G[i + 3, i] = 1      # G e_i = e_{i+3}
        G[i, i + 3] = -1     # G e_{i+3} = -e_i
    return G


G1 = G_one()
ETA1M = np.diag(ETA1).astype(np.int64)


def blockdiag(*ms):
    n = sum(m.shape[0] for m in ms)
    out = np.zeros((n, n), dtype=ms[0].dtype)
    i = 0
    for m in ms:
        k = m.shape[0]
        out[i:i + k, i:i + k] = m
        i += k
    return out


def Gn(n):
    return blockdiag(*([G1] * n))


def etan(n):
    return blockdiag(*([ETA1M] * n))


G2 = Gn(2)
ETA2M = etan(2)
Z6 = np.zeros((6, 6), dtype=np.int64)


def unit(n, i, j):
    M = np.zeros((n, n), dtype=np.int64)
    M[i, j] = 1
    return M


def rot_vec(n, a, b, eta):
    """vector-level generator of the plane (a,b) in so(eta): image of ad(1/2 E_a E_b), i.e. M e_c = eta_bc e_a - eta_ac e_b."""
    M = np.zeros((n, n), dtype=np.int64)
    M[a, b] += eta[b]
    M[b, a] -= eta[a]
    return M


def so_basis(n, eta):
    return [rot_vec(n, a, b, eta) for a in range(n) for b in range(a + 1, n)]


def nullspace_QQ(rows, ncols):
    """exact rational null space of an integer matrix given as a list of rows."""
    if len(rows) == 0:
        return [sp.Matrix.eye(ncols)[:, i] for i in range(ncols)]
    M = DomainMatrix([[sp.QQ(int(x)) for x in r] for r in rows], (len(rows), ncols), sp.QQ)
    ns = M.nullspace().to_Matrix()
    return [ns.row(i).T for i in range(ns.rows)]


def rank_QQ(rows, ncols):
    if len(rows) == 0:
        return 0
    M = DomainMatrix([[sp.QQ(int(x)) if not isinstance(x, Fraction) else sp.QQ(x.numerator, x.denominator) for x in r]
                      for r in rows], (len(rows), ncols), sp.QQ)
    return M.rank()


def span_rank(mats):
    """rank of a list of integer matrices as vectors (exact)."""
    if not mats:
        return 0
    rows = [np.asarray(m).flatten().tolist() for m in mats]
    return rank_QQ(rows, len(rows[0]))


def lie_subspace(basis, conds):
    """exact subspace {sum c_i basis_i : cond(X) = 0 for each cond} where each cond is a linear integer-valued map.
    returns list of integer matrices spanning it (denominators cleared)."""
    cols = []
    for B in basis:
        col = np.concatenate([np.asarray(c(B)).flatten() for c in conds])
        cols.append(col)
    A = np.array(cols).T          # rows: equations
    rows = [r.tolist() for r in A if np.any(r)]
    ns = nullspace_QQ(rows, len(basis))
    out = []
    for v in ns:
        den = sp.ilcm(*[sp.fraction(x)[1] for x in v]) if len(v) else 1
        coeffs = [int(x * den) for x in v]
        M = sum(c * B for c, B in zip(coeffs, basis))
        out.append(np.asarray(M, dtype=np.int64))
    return out


SO66 = so_basis(12, ETA2.tolist())


def in_so(M, eta):
    return iszero(M.T @ eta + eta @ M)


# ============================================================================== spin lift on S_2
def spin_lift(M):
    """4 * S(M) as an integer 64x64 matrix (S(M) = 1/4 sum (M eta)_{ab} E_a E_b)."""
    A = M @ ETA2M
    out = np.zeros((64, 64), dtype=np.int64)
    for a in range(12):
        for b in range(12):
            if A[a, b]:
                out += A[a, b] * (GEN[a] @ GEN[b])
    return out


# ============================================================================== named subalgebras (vector level, n = 2)
def emb(k, M6):
    """embed a 6x6 matrix into constituent k of V_2."""
    out = np.zeros((12, 12), dtype=np.int64)
    o = 6 * (k - 1)
    out[o:o + 6, o:o + 6] = M6
    return out


def lG_basis():
    """L_i: paired rotations (same so(3) generator on (e1,e2,e3) and on (e4,e5,e6)); K_i = G L_i."""
    Ls = []
    for (i, j) in [(1, 2), (2, 0), (0, 1)]:     # L_1 rotates (e2,e3), L_2 (e3,e1), L_3 (e1,e2)
        L = np.zeros((6, 6), dtype=np.int64)
        L[j, i] -= 1
        L[i, j] += 1
        L[j + 3, i + 3] -= 1
        L[i + 3, j + 3] += 1
        Ls.append(-L)                            # sign: [L_i, L_j] = eps_ijk L_k
    Ks = [G1 @ L for L in Ls]
    return Ls, Ks


LG_L, LG_K = lG_basis()
LG1 = [emb(1, M) for M in LG_L + LG_K]
LG2 = [emb(2, M) for M in LG_L + LG_K]
LGD = [emb(1, M) + emb(2, M) for M in LG_L + LG_K]

INT0 = [0, 3, 4, 5]             # interior four-plane of one constituent (0-based: e1,e4,e5,e6)
SUR0 = [1, 2]                   # surplus pair (e2,e3)


def lint_basis(k):
    idx = [6 * (k - 1) + i for i in INT0]
    return [rot_vec(12, idx[a], idx[b], ETA2.tolist()) for a in range(4) for b in range(a + 1, 4)]


LINT1 = lint_basis(1)
LINT2 = lint_basis(2)

CENT = lie_subspace(SO66, [lambda X: comm(X, G2)])


def sym_basis(n):
    out = []
    for i in range(n):
        for j in range(i, n):
            M = np.zeros((n, n), dtype=np.int64)
            M[i, j] = 1
            M[j, i] = 1
            out.append(M)
    return out


def invariant_forms(gens, n, G=None, cls=None, on_dual=True):
    """exact basis of symmetric forms invariant under gens.  on_dual: N on V^*, r N + N r^T = 0 (Def. 17.10 of ES17);
    otherwise S on V, r^T S + S r = 0.  cls = 'lor' ({N,G} = 0), 'dual' ([N,G] = 0) or None."""
    conds = []
    for r in gens:
        if on_dual:
            conds.append(lambda N, r=r: r @ N + N @ r.T)
        else:
            conds.append(lambda S, r=r: r.T @ S + S @ r)
    if cls == 'lor':
        conds.append(lambda N: N @ G + G @ N)
    elif cls == 'dual':
        conds.append(lambda N: N @ G - G @ N)
    if not conds:
        return sym_basis(n)
    return lie_subspace(sym_basis(n), conds)


# temporal and partner indices of V_2 (0-based)
TEMP = [0, 1, 2, 6, 7, 8]
PART = [3, 4, 5, 9, 10, 11]


def complex_form(X):
    """the complex 6x6 matrix by which X in Cent(G_2) acts on (z1, z2): R - iS."""
    R = X[np.ix_(TEMP, TEMP)]
    mS = X[np.ix_(PART, TEMP)]
    return R.astype(complex) + 1j * mS


def zpair(Y):
    z1 = Y[0:3] + 1j * Y[3:6]
    z2 = Y[6:9] + 1j * Y[9:12]
    return z1, z2


def from_z(z1, z2):
    return np.concatenate([z1.real, z1.imag, z2.real, z2.imag])


def gram(z1, z2):
    return np.array([[z1 @ z1, z1 @ z2], [z2 @ z1, z2 @ z2]])


def Mpair(z1, z2):
    s = np.linalg.svd(gram(z1, z2), compute_uv=False)
    return 0.5 * (s[0] + s[1])


# copy rotation C_v = eps (x) 1_6 and copy boost eps (x) G on vectors
EPS = np.array([[0, -1], [1, 0]], dtype=np.int64)
CV = np.kron(EPS, np.eye(6, dtype=np.int64))
CBOOST = np.kron(EPS, G1)

print("# e5 -- closure at two constituents; conventions as in the docstring")


# ----------------------------------------------------------------------------- 1. closure symmetry and its invariants
@claim("e5-1a", "G_2 = G+G: G_2^T eta_2 G_2 = -eta_2, G_2^2 = -1; Cent(G_2) in so(6,6) has dim 30 = 15 paired rotations "
                "+ 15 paired boosts, every element is [[R,S],[-S,R]] (temporal|partner basis) with R,S antisymmetric, R-iS is a "
                "Lie isomorphism onto so(6,C), and the Euclidean-antisymmetric (compact) part is the 15 paired rotations "
                "(restates RE Prop. 7.8(i))")
def _():
    ok = iszero(G2.T @ ETA2M @ G2 + ETA2M) and iszero(G2 @ G2 + np.eye(12, dtype=np.int64))
    ok &= len(CENT) == 30
    # paired rotations / boosts
    rots, bsts = [], []
    for a, b in itertools.combinations(range(6), 2):
        ta, tb, pa, pb = TEMP[a], TEMP[b], PART[a], PART[b]
        R = np.zeros((12, 12), dtype=np.int64)
        R[ta, tb], R[tb, ta], R[pa, pb], R[pb, pa] = -1, 1, -1, 1
        rots.append(R)
        bsts.append(G2 @ R)                      # paired boost
    allp = rots + bsts
    ok &= all(in_so(M, ETA2M) and iszero(comm(M, G2)) for M in allp)
    ok &= span_rank(allp) == 30 and span_rank(allp + CENT) == 30
    for X in CENT:
        R = X[np.ix_(TEMP, TEMP)]
        S = X[np.ix_(TEMP, PART)]
        ok &= iszero(X[np.ix_(PART, PART)] - R) and iszero(X[np.ix_(PART, TEMP)] + S)
        ok &= iszero(R + R.T) and iszero(S + S.T)
    # R - iS is a Lie homomorphism (checked on all pairs of the basis) and injective
    cf = [complex_form(X) for X in allp]
    for i, j in itertools.combinations(range(30), 2):
        lhs = complex_form(comm(allp[i], allp[j]))
        ok &= np.array_equal(lhs, cf[i] @ cf[j] - cf[j] @ cf[i])
    ok &= all(np.array_equal(c, -c.T) for c in cf)
    flat = np.array([np.concatenate([c.real.flatten(), c.imag.flatten()]) for c in cf])
    ok &= np.linalg.matrix_rank(flat) == 30
    comp = lie_subspace(CENT, [lambda X: X + X.T])
    ok &= len(comp) == 15 and span_rank(comp + rots) == 15
    return ok


def holo_invariants(cplx_gens):
    """complex dimension of the complex symmetric 6x6 B with A^T B + B A = 0 (holomorphic quadratic invariants);
    unknowns: real and imaginary parts of the 21 entries."""
    basis = []
    for i in range(6):
        for j in range(i, 6):
            M = np.zeros((6, 6))
            M[i, j] = M[j, i] = 1
            basis.append(M)
    cols = []
    for part in (1.0, 1j):
        for M in basis:
            Bm = part * M
            col = np.concatenate([np.concatenate([(A.T @ Bm + Bm @ A).real.flatten(), (A.T @ Bm + Bm @ A).imag.flatten()])
                                  for A in cplx_gens])
            cols.append(col)
    A = np.array(cols).T
    rows = [[int(round(x)) for x in r] for r in A if np.any(np.abs(r) > 0.5)]
    assert np.allclose(A, np.round(A))
    nul = 42 - rank_QQ(rows, 42)
    assert nul % 2 == 0
    return nul // 2


@claim("e5-1b", "quadratic invariants of the pair (z1,z2): holomorphic ones have complex dimension 1 (q1+q2) under "
                "so(6,C), 2 (q1, q2) under lG^(1)+lG^(2), 3 (q1, q2, z1.z2) under the diagonal lG; the invariant real "
                "quadratic forms on V_2 have real dimension 2, 4, 6 and are spanned by the real and imaginary parts of these")
def _():
    ok = True
    groups = [(CENT, 1), (LG1 + LG2, 2), (LGD, 3)]
    for gens, d in groups:
        ok &= holo_invariants([complex_form(X) for X in gens]) == d
    # real forms on V_2
    def qform(f):
        # symmetric integer matrix S with Y^T S Y = 2 f(Y), f = Re or Im of a complex quadratic in (z1,z2)
        S = np.zeros((12, 12))
        Ib = np.eye(12)
        for a in range(12):
            for b in range(12):
                S[a, b] = f(Ib[a] + Ib[b]) - f(Ib[a]) - f(Ib[b])     # twice the polar form: integer
        return S
    q1 = lambda Y: zpair(Y)[0] @ zpair(Y)[0]
    q2 = lambda Y: zpair(Y)[1] @ zpair(Y)[1]
    q12 = lambda Y: zpair(Y)[0] @ zpair(Y)[1]
    cand = {1: [lambda Y: q1(Y) + q2(Y)], 2: [q1, q2], 3: [q1, q2, q12]}
    for gens, d in groups:
        inv = invariant_forms(gens, 12, on_dual=False)
        ok &= len(inv) == 2 * d
        Ss = []
        for f in cand[d]:
            for part in (np.real, np.imag):
                S = qform(lambda Y, f=f, part=part: part(f(Y)))
                ok &= np.allclose(S, np.round(S))
                Ss.append(np.round(S).astype(np.int64))
        ok &= span_rank(Ss) == 2 * d and span_rank(Ss + inv) == 2 * d
    return ok


RNG = np.random.default_rng(20261020)


def dMpair_sq(Y, X):
    """derivative of (sigma1+sigma2)^2 = |Q|_F^2 + 2|det Q| at Y along the vector field X Y."""
    z1, z2 = zpair(Y)
    d1, d2 = zpair(X @ Y)
    Z = np.stack([z1, z2], axis=1)
    dZ = np.stack([d1, d2], axis=1)
    Q = Z.T @ Z
    dQ = dZ.T @ Z + Z.T @ dZ
    det = Q[0, 0] * Q[1, 1] - Q[0, 1] * Q[1, 0]
    adj = np.array([[Q[1, 1], -Q[0, 1]], [-Q[1, 0], Q[0, 0]]])
    ddet = np.trace(adj @ dQ)
    return 2 * np.real(np.trace(Q.conj().T @ dQ)) + 2 * np.real(np.conj(det) / abs(det) * ddet)


@claim("e5-1c", "the pair invariant M_pair = (sigma1+sigma2)/2 (RE Thm 8.6) is invariant under the diagonal lG and the "
                "copy rotation C_v (|d M_pair^2| < 1e-11 at 40 random pairs), and the subalgebra of so(6,C) annihilating it "
                "has real dimension 7 = diag lG + R C_v (numerical rank 23 of 30 at 40 random pairs, singular-value gap "
                "> 1e8); hence the closure symmetry does not preserve M_pair")
def _():
    pts = [RNG.normal(size=12) for _ in range(40)]
    ok = True
    for Y in pts:
        for X in LGD + [CV]:
            ok &= abs(dMpair_sq(Y, X)) < 1e-11 * (1 + np.dot(Y, Y) ** 2)
    D = np.array([[dMpair_sq(Y, X) for X in CENT] for Y in pts])
    s = np.linalg.svd(D, compute_uv=False)
    rank = int(np.sum(s > 1e-8 * s[0]))
    ok &= rank == 23 and s[22] / max(s[23], 1e-300) > 1e8
    # the 7 known elements lie in CENT
    ok &= span_rank(CENT + LGD + [CV]) == 30
    return ok


@claim("e5-1d", "explicit non-invariance under the closure symmetry: the copy boost eps(x)G in so(6,C) is +1 on V_+, so "
                "e^{t eps(x)G} maps the balanced pair (e1, i e1) (M_pair = 1, q_tot = 0) to e^t (e1, i e1), with q_tot = 0 "
                "and M_pair = e^{2t} (7 values of t, to 1e-12); the compact R^(1)_phi of RE Ex. 8.8 takes ((1,i,0),0) from M_pair = 0 to "
                "sin(phi); on {q_tot = c} M_pair attains its lower bound |c|/2 at (z,0), and the copy boost carries (z,0) to "
                "M_pair = |c| cosh(2t)/2 at fixed q_tot (5 random z, 3 values of t, to 1e-12)")
def _():
    ok = True
    z1 = np.array([1, 0, 0], dtype=complex)
    z2 = 1j * z1
    Y = from_z(z1, z2)
    ok &= abs(Mpair(z1, z2) - 1) < 1e-14 and abs(z1 @ z1 + z2 @ z2) < 1e-14
    for t in [-1.3, -0.4, -0.1, 0.0, 0.2, 0.7, 1.1]:
        Yt = (np.cosh(t) * np.eye(12) + np.sinh(t) * CBOOST) @ Y
        a, b = zpair(Yt)
        ok &= np.allclose(Yt, np.exp(t) * Y, atol=1e-14) and abs(Mpair(a, b) - np.exp(2 * t)) < 1e-12 * np.exp(2 * abs(t))
        ok &= abs(a @ a + b @ b) < 1e-12 * np.cosh(2 * t)          # q_tot preserved
    # R^(1)_phi: rotates the planes (e^(1)_1, e^(2)_1) and (e^(1)_4, e^(2)_4)
    R1 = np.zeros((12, 12), dtype=np.int64)
    R1[6, 0], R1[0, 6], R1[9, 3], R1[3, 9] = 1, -1, 1, -1
    ok &= span_rank(CENT + [R1]) == 30 and iszero(R1 + R1.T)
    z1 = np.array([1, 1j, 0])
    z2 = np.zeros(3, dtype=complex)
    Y = from_z(z1, z2)
    ok &= Mpair(z1, z2) < 1e-14
    for phi in [0.3, 0.9, 1.4]:
        a, b = zpair(sla.expm(phi * R1) @ Y)
        ok &= abs(Mpair(a, b) - np.sin(phi)) < 1e-12
    for _ in range(5):
        z = RNG.normal(size=3) + 1j * RNG.normal(size=3)
        ok &= abs(Mpair(z, 0 * z) - abs(z @ z) / 2) < 1e-12
        for t in [-0.8, 0.5, 1.5]:      # the copy boost carries (z,0) to M_pair = |q(z)| cosh(2t) / 2 at fixed q_tot
            a, b = zpair((np.cosh(t) * np.eye(12) + np.sinh(t) * CBOOST) @ from_z(z, 0 * z))
            ok &= abs(Mpair(a, b) - abs(z @ z) * np.cosh(2 * t) / 2) < 1e-12 * np.cosh(2 * t) * abs(z @ z)
            ok &= abs(a @ a + b @ b - z @ z) < 1e-12 * np.cosh(2 * t) * abs(z @ z)
    return ok


# ----------------------------------------------------------------------------- 2. no common Lorentz algebra
@claim("e5-2a", "Cent(G_2) meets lint^(1) in 0, and the elements of so(6,6) commuting with G_2 and lint^(1) form lG^(2) "
                "(dim 6; restates RE Prop. 7.8(iv)); for comparison the same algebra is 0 at n = 1 and 30-dimensional at "
                "n = 3 (so(6,C) of constituents 2,3, ES17 Thm 17.20): dimension 3(n-1)(3n-4)")
def _():
    ok = len(lie_subspace(LINT1, [lambda X: comm(X, G2)])) == 0
    both = lie_subspace(SO66, [lambda X: comm(X, G2)] + [lambda X, L=L: comm(X, L) for L in LINT1])
    ok &= len(both) == 6 and span_rank(both + LG2) == 6
    for n, d in [(1, 0), (3, 30)]:
        N = 6 * n
        eta = etan(n)
        etal = np.diag(eta).tolist()
        idx = INT0
        lint = [rot_vec(N, idx[a], idx[b], etal) for a in range(4) for b in range(a + 1, 4)]
        Gk = Gn(n)
        sol = lie_subspace(so_basis(N, etal), [lambda X: comm(X, Gk)] + [lambda X, L=L: comm(X, L) for L in lint])
        ok &= len(sol) == d == 3 * (n - 1) * (3 * n - 4)
    return ok


def casimir_content(Ls, Ks, P=None, cplx=None):
    """(j_L, j_R) content of a real module under so(1,3) with generators L_i, K_i ([L,L]=L, [L,K]=K, [K,K]=-L) acting by
    real matrices; with cplx (a complex structure commuting with everything) the content is counted over C = (R, cplx);
    otherwise over the complexification.  Returns dict {(jL,jR): multiplicity}."""
    n = Ls[0].shape[0]
    if cplx is not None:
        # complex structure: work on the +i eigenspace of cplx
        w, U = np.linalg.eig(cplx.astype(float))
        B = U[:, np.abs(w - 1j) < 1e-9]
        proj = lambda M: np.linalg.lstsq(B, M @ B, rcond=None)[0]
        Ls = [proj(L) for L in Ls]
        Ks = [proj(K) for K in Ks]
        if P is not None:
            P = proj(P)
        n = B.shape[1]
    A = [0.5 * (L + 1j * K) for L, K in zip(Ls, Ks)]
    Bm = [0.5 * (L - 1j * K) for L, K in zip(Ls, Ks)]
    CA = -sum(a @ a for a in A)
    CB = -sum(b @ b for b in Bm)
    if P is not None:
        # restrict to the image of the projector P (commuting with everything)
        w, U = np.linalg.eig(P.astype(complex))
        Bs = U[:, np.abs(w - 1) < 1e-9]
        CA = np.linalg.lstsq(Bs, CA @ Bs, rcond=None)[0]
        CB = np.linalg.lstsq(Bs, CB @ Bs, rcond=None)[0]
    # CA and CB commute: diagonalise a generic combination
    M = CA + np.pi * CB
    w, U = np.linalg.eig(M)
    out = {}
    Ui = np.linalg.inv(U)
    a_vals = np.diag(Ui @ CA @ U)
    b_vals = np.diag(Ui @ CB @ U)
    for a, b in zip(a_vals, b_vals):
        jl = (-1 + np.sqrt(1 + 4 * a.real)) / 2
        jr = (-1 + np.sqrt(1 + 4 * b.real)) / 2
        key = (round(2 * jl) / 2, round(2 * jr) / 2)
        assert abs(2 * jl - round(2 * jl)) < 1e-9 and abs(2 * jr - round(2 * jr)) < 1e-9
        out[key] = out.get(key, 0) + 1
    dims = {k: (int(2 * k[0] + 1) * int(2 * k[1] + 1)) for k in out}
    return {k: v // dims[k] for k, v in out.items()}


def lift_LK(Lv, Kv):
    return [spin_lift(M) / 4.0 for M in Lv], [spin_lift(M) / 4.0 for M in Kv]


# interior Lorentz algebra of constituent k with so(1,3) brackets: rotations S_i of (e5,e6),(e6,e4),(e4,e5), boosts T_i (e1,e_{i+3})
def lint_LK(k):
    o = 6 * (k - 1)
    sp_ = [o + 3, o + 4, o + 5]
    S_ = []
    for (i, j) in [(1, 2), (2, 0), (0, 1)]:
        M = np.zeros((12, 12), dtype=np.int64)
        M[sp_[j], sp_[i]] += 1
        M[sp_[i], sp_[j]] -= 1
        S_.append(M)
    T_ = []
    for i in range(3):
        M = np.zeros((12, 12), dtype=np.int64)
        M[o + 0, sp_[i]] = 1
        M[sp_[i], o + 0] = 1
        T_.append(M)
    return S_, T_


def check_lorentz_brackets(Ls, Ks):
    ok = True
    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        ok &= np.array_equal(comm(Ls[i], Ls[j]), Ls[k]) and np.array_equal(comm(Ls[i], Ks[j]), Ks[k])
        ok &= np.array_equal(comm(Ks[i], Ks[j]), -Ls[k])
    return ok


def fix_lint_signs():
    S_, T_ = lint_LK(1)
    for sS in (1, -1):
        for sT in (1, -1):
            Ls = [sS * M for M in S_]
            Ks = [sT * M for M in T_]
            if check_lorentz_brackets(Ls, Ks):
                return sS, sT
    raise RuntimeError("no sign choice")


SGN_S, SGN_T = fix_lint_signs()


def lint_vec(k):
    S_, T_ = lint_LK(k)
    return [SGN_S * M for M in S_], [SGN_T * M for M in T_]


@claim("e5-2b", "spinor content: constituent-1 bivectors of the tower are e_a e_b (x) 1 exactly, so S_n (x) C carries "
                "8^(n-1) copies of the one-constituent content; at n = 1, S (x) C = 2(1/2,1/2) under lG and "
                "2(1/2,0)+2(0,1/2) under lint (Casimirs, tol 1e-9); hence S_2 (x) C = 16 (1/2,1/2) under lG^(1) (no Weyl "
                "spinor; 128 at n = 3) and 16(1/2,0) + 16(0,1/2) under lint^(1); computed directly on S_2 as well")
def _():
    ok = all(np.array_equal(E[(1, a)] @ E[(1, b)], np.kron(e[a] @ e[b], I8)) for a in range(1, 7) for b in range(1, 7))
    Lv = [emb(1, M) for M in LG_L]
    Kv = [emb(1, M) for M in LG_K]
    ok &= check_lorentz_brackets(Lv, Kv)
    Ls, Ks = lift_LK(Lv, Kv)
    ok &= casimir_content(Ls, Ks) == {(0.5, 0.5): 16}
    # n = 1 content from the 8x8 block (the tower lift is (8x8) (x) 1_8)
    Ls1 = [L[np.ix_(range(0, 64, 8), range(0, 64, 8))] for L in Ls]
    ok &= all(np.allclose(L, np.kron(l1, np.eye(8))) for L, l1 in zip(Ls, Ls1))
    Ks1 = [K[np.ix_(range(0, 64, 8), range(0, 64, 8))] for K in Ks]
    ok &= casimir_content(Ls1, Ks1) == {(0.5, 0.5): 2}
    Sv, Tv = lint_vec(1)
    Ls, Ks = lift_LK(Sv, Tv)
    c = casimir_content(Ls, Ks)
    ok &= c == {(0.5, 0.0): 16, (0.0, 0.5): 16}
    Ls1 = [L[np.ix_(range(0, 64, 8), range(0, 64, 8))] for L in Ls]
    Ks1 = [K[np.ix_(range(0, 64, 8), range(0, 64, 8))] for K in Ks]
    ok &= casimir_content(Ls1, Ks1) == {(0.5, 0.0): 2, (0.0, 0.5): 2}
    ok &= 2 * 8 ** 2 == 128
    return ok


@claim("e5-2c", "under lG^(2) alone S_2 (x) C = 16 (1/2,1/2); under the diagonal lG (common Lorentz transformation of "
                "both field strengths) S_2 (x) C = 4[(0,0)+(1,0)+(0,1)+(1,1)]: integer spin only")
def _():
    Lv = [emb(2, M) for M in LG_L]
    Kv = [emb(2, M) for M in LG_K]
    Ls, Ks = lift_LK(Lv, Kv)
    ok = casimir_content(Ls, Ks) == {(0.5, 0.5): 16}
    Lv = [emb(1, M) + emb(2, M) for M in LG_L]
    Kv = [emb(1, M) + emb(2, M) for M in LG_K]
    ok &= check_lorentz_brackets(Lv, Kv)
    Ls, Ks = lift_LK(Lv, Kv)
    ok &= casimir_content(Ls, Ks) == {(0.0, 0.0): 4, (1.0, 0.0): 4, (0.0, 1.0): 4, (1.0, 1.0): 4}
    return ok


# ----------------------------------------------------------------------------- 3. gauge algebras at n = 2 and the binding root
def euclid_compact(basis):
    return lie_subspace(basis, [lambda X: X + X.T])


GSPEC = lie_subspace(SO66, [lambda X, L=L: comm(X, L) for L in LINT1])          # g(1,0,1) = so(5,3)
GSPEC_C = euclid_compact(GSPEC)                                                  # so(5) + so(3)
LG2_ROT = [emb(2, M) for M in LG_L]
TRIV = [E[(1, 2)], E[(1, 3)]] + [E[(2, a)] for a in range(1, 7)]               # Lorentz-trivial generators (spectator)
W_CS = TRIV[0]
for g_ in TRIV[1:]:
    W_CS = W_CS @ g_
OMEGA1 = E[(1, 1)] @ E[(1, 4)] @ E[(1, 5)] @ E[(1, 6)]
GAMMA_SPEC = W_CS @ OMEGA1
PL_SPEC = (I64 - GAMMA_SPEC) // 1        # 2 P_L (integer)
PR_SPEC = (I64 + GAMMA_SPEC) // 1        # 2 P_R
GAMMAS = [E[(1, 1)], E[(1, 4)], E[(1, 5)], E[(1, 6)]]
KIN = [GAMMAS[0] @ g_ for g_ in GAMMAS]                                           # K^mu = gamma^0 gamma^mu (constituent 1)


@claim("e5-3a", "spectator action: g(1,0,1) = Cent(lint^(1)) in so(6,6) has dim 28 (so(5,3)), compact part dim 13 "
                "(so(5)+so(3)); g(1,0,1) meets Cent(G_2) in lG^(2) (dim 6), and its compact part meets Cent(G_2) in the "
                "three paired rotations of constituent 2: su(2), compact and non-abelian (brackets [L_i,L_j] = eps L_k)")
def _():
    ok = len(GSPEC) == 28 and len(GSPEC_C) == 13
    inter = lie_subspace(GSPEC, [lambda X: comm(X, G2)])
    ok &= len(inter) == 6 and span_rank(inter + LG2) == 6
    interc = lie_subspace(GSPEC_C, [lambda X: comm(X, G2)])
    ok &= len(interc) == 3 and span_rank(interc + LG2_ROT) == 3
    ok &= np.array_equal(comm(LG2_ROT[0], LG2_ROT[1]), LG2_ROT[2])
    return ok


@claim("e5-3b", "spectator Weyl block: w = product of the 8 Lorentz-trivial generators has w^2 = -1 and commutes with "
                "lint^(1) and with the spin lift of so(5,3); Gamma = w Omega^(1) is an involution commuting with w, "
                "rank P_L = 32; with w as i, P_L and P_R each carry 8 copies of one Weyl representation of lint^(1), the two "
                "halves opposite (8 (1/2,0) and 8 (0,1/2)), and the so(5) and "
                "so(3) Casimirs are 5/2 and 3/4 identically (the (4,2) of ES17 Prop. 10.12, restated)")
def _():
    ok = np.array_equal(W_CS @ W_CS, -I64)
    lifts = [spin_lift(M) for M in GSPEC]
    Sv, Tv = lint_vec(1)
    lint_l = [spin_lift(M) for M in Sv + Tv]
    ok &= all(iszero(comm(W_CS, X)) for X in lifts + lint_l)
    ok &= np.array_equal(GAMMA_SPEC @ GAMMA_SPEC, I64) and iszero(comm(GAMMA_SPEC, W_CS))
    ok &= all(iszero(comm(GAMMA_SPEC, X)) for X in lifts + lint_l)
    ok &= np.trace(PL_SPEC) == 64 and np.linalg.matrix_rank(PL_SPEC) == 32
    Ls, Ks = lift_LK(Sv, Tv)
    cl = casimir_content(Ls, Ks, P=PL_SPEC / 2, cplx=W_CS)
    cr = casimir_content(Ls, Ks, P=PR_SPEC / 2, cplx=W_CS)
    ok &= sorted([cl, cr], key=str) == sorted([{(0.5, 0.0): 8}, {(0.0, 0.5): 8}], key=str)
    pos = [E[(1, 2)], E[(1, 3)], E[(2, 1)], E[(2, 2)], E[(2, 3)]]
    neg = [E[(2, 4)], E[(2, 5)], E[(2, 6)]]
    C5 = -sum((pos[a] @ pos[b]) @ (pos[a] @ pos[b]) for a, b in itertools.combinations(range(5), 2))   # 4 x Casimir
    C3 = -sum((neg[a] @ neg[b]) @ (neg[a] @ neg[b]) for a, b in itertools.combinations(range(3), 2))
    ok &= np.array_equal(C5, 10 * I64) and np.array_equal(C3, 3 * I64)
    return ok


def su2_content_on(Xs, P, cplx):
    """multiplicities of su(2) spins j on the image of the projector P, counted over C = (R, cplx)."""
    w, U = np.linalg.eig(cplx.astype(float))
    B = U[:, np.abs(w - 1j) < 1e-9]
    proj = lambda M: np.linalg.lstsq(B, M @ B, rcond=None)[0]
    C = -sum(proj(X) @ proj(X) for X in Xs)
    Pp = proj(P)
    wp, Up = np.linalg.eig(Pp)
    Bs = Up[:, np.abs(wp - 1) < 1e-9]
    Cr = np.linalg.lstsq(Bs, C @ Bs, rcond=None)[0]
    ev = np.linalg.eigvals(Cr).real
    out = {}
    for x in ev:
        j = (-1 + np.sqrt(1 + 4 * x)) / 2
        assert abs(2 * j - round(2 * j)) < 1e-9
        j = round(2 * j) / 2
        out[j] = out.get(j, 0) + 1
    return {j: m // int(2 * j + 1) for j, m in out.items()}


@claim("e5-3c", "the binding-compatible su(2) of the spectator algebra acts on the Weyl block with integer isospin: "
                "P_L = (1/2,0) (x) (2.[0] + 2.[1]) and P_R = (0,1/2) (x) (2.[0] + 2.[1]) (w as i; Casimirs, tol 1e-9), the "
                "same real representation on both halves: vector-like; with charges h = -w S(X), A(h) = A(h^3) = 0 exactly "
                "(Tr(h^k P_L) = Tr(h^k P_R) = 0 for k = 1, 3), no doublet, hence no global SU(2) anomaly")
def _():
    Xs = [spin_lift(M) / 4.0 for M in LG2_ROT]
    ok = su2_content_on(Xs, PL_SPEC / 2, W_CS) == {0.0: 4, 1.0: 4}       # 4 = 2 singlets x 2 Lorentz components
    ok &= su2_content_on(Xs, PR_SPEC / 2, W_CS) == {0.0: 4, 1.0: 4}
    # exact traces: 4 h = -w (4 S(X)); integers
    for M in LG2_ROT:
        H4 = -W_CS @ spin_lift(M)
        for k in (1, 3):
            Hk = np.linalg.matrix_power(H4, k)
            ok &= np.trace(Hk @ PL_SPEC) == 0 and np.trace(Hk @ PR_SPEC) == 0
    return ok


@claim("e5-3d", "kinetic condition (ES17 Prop. 5.5: X^T K^mu + K^mu X = 0, K^mu = gamma^0 gamma^mu of constituent 1): among the "
                "spin lifts of the 30 elements of Cent(G_2) exactly a 3-dimensional subspace satisfies it, the paired "
                "rotations of constituent 2; the paired boosts of constituent 2 fail it")
def _():
    conds = [lambda M, K=K: spin_lift(M).T @ K + K @ spin_lift(M) for K in KIN]
    sol = lie_subspace(CENT, conds)
    ok = len(sol) == 3 and span_rank(sol + LG2_ROT) == 3
    for M in [emb(2, K_) for K_ in LG_K]:
        S4 = spin_lift(M)
        ok &= any(not iszero(S4.T @ K + K @ S4) for K in KIN)
    return ok


# diagonal interior action
def lint_diag():
    S1, T1 = lint_vec(1)
    S2, T2 = lint_vec(2)
    return [a + b for a, b in zip(S1, S2)], [a + b for a, b in zip(T1, T2)]


LD_S, LD_T = lint_diag()
GDIAG = lie_subspace(SO66, [lambda X, L=L: comm(X, L) for L in LD_S + LD_T])     # so(2) + so(4)
CB4 = 2 * sum(ETA1[a - 1] * (E[(1, a)] @ E[(2, a)]) for a in range(1, 7))        # 4 C_b
GAMMA12 = I64.copy()
for g_ in GEN:
    GAMMA12 = GAMMA12 @ g_


@claim("e5-3e", "diagonal interior action: Cent(diag lint) in so(6,6) has dim 7 (so(2)+so(4)), all compact; it meets "
                "Cent(G_2) in R C_v, the copy rotation of all six directions (= M_12 + the surplus copy rotation), abelian "
                "and compact; its spin lift is S(C_v) = -C_b with C_b = (1/2) sum_a eta_aa E_(1,a)E_(2,a) (RE Prop. 7.3)")
def _():
    ok = len(GDIAG) == 7 and len(euclid_compact(GDIAG)) == 7
    inter = lie_subspace(GDIAG, [lambda X: comm(X, G2)])
    ok &= len(inter) == 1 and span_rank(inter + [CV]) == 1
    ok &= np.array_equal(spin_lift(CV), -CB4)
    # C_v = M12 (interior four-planes) + surplus copy rotation
    M12 = np.zeros((12, 12), dtype=np.int64)
    MS = np.zeros((12, 12), dtype=np.int64)
    for a in range(6):
        tgt = M12 if a in INT0 else MS
        tgt[6 + a, a] = 1
        tgt[a, 6 + a] = -1
    ok &= np.array_equal(M12 + MS, CV) and span_rank(GDIAG + [M12, MS]) == 7
    return ok


@claim("e5-3f", "kinetic condition for C_b: with the summed kernels K^mu_S = sum_k gamma^(k)_0 gamma^(k)_mu it holds; with "
                "the kernels of the averaged gammas f_mu = (gamma^(1)_mu + gamma^(2)_mu)/sqrt2 it fails (exact)")
def _():
    g1 = [E[(1, a)] for a in (1, 4, 5, 6)]
    g2 = [E[(2, a)] for a in (1, 4, 5, 6)]
    ok = iszero(CB4 + CB4.T)
    for mu in range(4):
        KS = g1[0] @ g1[mu] + g2[0] @ g2[mu]
        ok &= iszero(CB4.T @ KS + KS @ CB4)
    bad = False
    for mu in range(1, 4):
        Kf2 = (g1[0] + g2[0]) @ (g1[mu] + g2[mu])         # 2 f_0 f_mu
        bad |= not iszero(CB4.T @ Kf2 + Kf2 @ CB4)
    return ok and bad


DIAG_CONTENT = {
    # (Gamma_12, (jL, jR), copy charge q of C_b = i q): multiplicity, over the complexification
    (-1, (0.0, 0.0), -3): 1, (-1, (0.0, 0.0), 3): 1, (-1, (0.0, 0.0), -1): 1, (-1, (0.0, 0.0), 1): 1,
    (-1, (0.5, 0.5), -1): 2, (-1, (0.5, 0.5), 1): 2,
    (-1, (1.0, 0.0), -1): 1, (-1, (1.0, 0.0), 1): 1, (-1, (0.0, 1.0), -1): 1, (-1, (0.0, 1.0), 1): 1,
    (1, (0.0, 0.0), -2): 2, (1, (0.0, 0.0), 2): 2,
    (1, (0.5, 0.5), -2): 1, (1, (0.5, 0.5), 2): 1, (1, (0.5, 0.5), 0): 2,
    (1, (1.0, 0.0), 0): 2, (1, (0.0, 1.0), 0): 2,
}


@claim("e5-3g", "content of S_2 (x) C under the diagonal lint, the binding-compatible copy charge C_b and the chirality "
                "Gamma_12 = E_1...E_12 (Gamma_12^2 = 1, commuting with both): integer spin only; Gamma_12 = -1 carries the odd "
                "and Gamma_12 = +1 the even copy charges; every chirality half is invariant under (q, (jL,jR)) -> "
                "(-q, (jR,jL)), so it is self-conjugate: vector-like, and Tr(C_b^k P_+-) = 0 for odd k exactly")
def _():
    ok = np.array_equal(GAMMA12 @ GAMMA12, I64)
    ok &= all(iszero(comm(GAMMA12, X)) for X in [CB4] + [spin_lift(M) for M in LD_S + LD_T])
    ok &= check_lorentz_brackets(LD_S, LD_T)
    Ls, Ks = lift_LK(LD_S, LD_T)
    A = [0.5 * (L + 1j * K) for L, K in zip(Ls, Ks)]
    B = [0.5 * (L - 1j * K) for L, K in zip(Ls, Ks)]
    CA = -sum(a @ a for a in A)
    CBm = -sum(b @ b for b in B)
    Cb = CB4 / 4.0
    M = CA + np.pi * CBm + np.sqrt(2) * Cb / 1j + 0.37 * np.e * GAMMA12
    w, U = np.linalg.eig(M)
    Ui = np.linalg.inv(U)
    cnt = {}
    for X in range(64):
        u, ui = U[:, X], Ui[X]
        a, b = (ui @ CA @ u).real, (ui @ CBm @ u).real
        q, g = (ui @ Cb @ u / 1j).real, (ui @ GAMMA12 @ u).real
        jl, jr = (-1 + np.sqrt(1 + 4 * a)) / 2, (-1 + np.sqrt(1 + 4 * b)) / 2
        for v in (2 * jl, 2 * jr, q, g):
            ok &= abs(v - round(v)) < 1e-9
        key = (int(round(g)), (round(2 * jl) / 2, round(2 * jr) / 2), int(round(q)))
        cnt[key] = cnt.get(key, 0) + 1
    cnt = {k: v // int((2 * k[1][0] + 1) * (2 * k[1][1] + 1)) for k, v in cnt.items()}
    ok &= cnt == DIAG_CONTENT
    ok &= all(DIAG_CONTENT.get((g, (jr, jl), -q)) == m for (g, (jl, jr), q), m in DIAG_CONTENT.items())
    ok &= all((g == -1) == (q % 2 != 0) for (g, _, q) in DIAG_CONTENT)
    for sgn in (1, -1):
        P2 = I64 + sgn * GAMMA12
        for k in (1, 3, 5):
            ok &= np.trace(np.linalg.matrix_power(CB4, k) @ P2) == 0
    return ok


def lmix():
    S1, T1 = lint_vec(1)
    Lv = [emb(2, M) for M in LG_L]
    Kv = [emb(2, M) for M in LG_K]
    return [a + b for a, b in zip(S1, Lv)], [a + b for a, b in zip(T1, Kv)]


LM_S, LM_T = lmix()
J1V = rot_vec(12, 1, 2, ETA2.tolist())          # rotation of the surplus pair (e^(1)_2, e^(1)_3)
G2ND = emb(2, G1)                                # binding root of constituent 2 alone


@claim("e5-3h", "mixed rung (lint on constituent 1 paired with lG on constituent 2, ES17 Prop. 13.16): l_mix has the "
                "so(1,3) brackets; its centraliser in so(6,6) is R J_1 (ES17 Prop. 13.17, restated), which meets Cent(G_2) in "
                "0; G_2 itself does not commute with l_mix; the l_mix-equivariant endomorphisms of V_2 form a 7-dimensional "
                "algebra whose Euclidean-antisymmetric (non-dilating) part is span{G^(2) = 0+G, J_1}, and J_1 commutes with G^(2)")
def _():
    ok = check_lorentz_brackets(LM_S, LM_T)
    cen = lie_subspace(SO66, [lambda X, L=L: comm(X, L) for L in LM_S + LM_T])
    ok &= len(cen) == 1 and span_rank(cen + [J1V]) == 1
    ok &= len(lie_subspace(cen, [lambda X: comm(X, G2)])) == 0
    ok &= any(not iszero(comm(G2, L)) for L in LM_S + LM_T)
    endo = [unit(12, i, j) for i in range(12) for j in range(12)]
    eq_ = lie_subspace(endo, [lambda X, L=L: comm(X, L) for L in LM_S + LM_T])
    ok &= len(eq_) == 7
    na = lie_subspace(eq_, [lambda X: X + X.T])
    ok &= len(na) == 2 and span_rank(na + [G2ND, J1V]) == 2
    ok &= iszero(comm(J1V, G2ND)) and all(iszero(comm(G2ND, L)) for L in LM_S + LM_T)
    return ok


# ----------------------------------------------------------------------------- 4. inertia freedom at two constituents
INERTIA_TABLE = [
    # (name, generators, Lorentz class {N,G_2}=0, duality class [N,G_2]=0)
    ("Cent(G_2) = so(6,C)", CENT, 2, 0),
    ("diagonal lG + R C_v", LGD + [CV], 2, 0),
    ("lG^(1) + lG^(2)", LG1 + LG2, 4, 0),
    ("diagonal lG", LGD, 6, 0),
    ("lG^(2)", LG2, 14, 9),
    ("su(2) = paired rotations of constituent 2 (spectator, binding-compatible)", LG2_ROT, 14, 10),
    ("R C_v (diagonal connection algebra, binding-compatible)", [CV], 18, 18),
    ("none", [], 42, 36),
]


@claim("e5-4a", "inertia freedom at n = 2 (symmetric N on V_2^* invariant under the symmetry, r N + N r^T = 0; exact ranks): "
                + "; ".join(f"{nm}: Lorentz {a}, duality {b}" for nm, _, a, b in INERTIA_TABLE))
def _():
    ok = True
    for nm, gens, a, b in INERTIA_TABLE:
        la = invariant_forms(gens, 12, G=G2, cls='lor')
        lb = invariant_forms(gens, 12, G=G2, cls='dual')
        ok &= len(la) == a and len(lb) == b
    # the full-symmetry Lorentz forms are eta_2^{-1} and G_2 eta_2^{-1}
    la = invariant_forms(CENT, 12, G=G2, cls='lor')
    ok &= span_rank(la + [ETA2M, G2 @ ETA2M]) == 2
    return ok


@claim("e5-4b", "comparison along the ladder: under the full closure symmetry so(3n,C) the inertia freedom is (Lorentz, "
                "duality) = (2, 0) at n = 1, 2, 3; with no symmetry it is (3n(3n+1), (3n)^2) = (12, 9), (42, 36), (90, 81)")
def _():
    ok = True
    for n in (1, 2, 3):
        N = 6 * n
        etal = np.diag(etan(n)).tolist()
        Gk = Gn(n)
        cent = lie_subspace(so_basis(N, etal), [lambda X: comm(X, Gk)])
        ok &= len(cent) == 3 * n * (3 * n - 1)
        ok &= len(invariant_forms(cent, N, G=Gk, cls='lor')) == 2
        ok &= len(invariant_forms(cent, N, G=Gk, cls='dual')) == 0
        ok &= len(invariant_forms([], N, G=Gk, cls='lor')) == 3 * n * (3 * n + 1)
        ok &= len(invariant_forms([], N, G=Gk, cls='dual')) == (3 * n) ** 2
    return ok
