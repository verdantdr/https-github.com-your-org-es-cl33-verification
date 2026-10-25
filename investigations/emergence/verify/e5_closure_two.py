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
integrations of the closure (DOP853, rtol = atol = 1e-12; tolerances stated per check), (d) the quadratures of the drift
(trapezoidal rule on 4096 points, spectrally accurate for the smooth periodic integrands) and the Jacobian ranks of e5-5h.
Runtime: under twenty seconds.
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


@claim("e5-1e", "M_pair through the three diagonal invariants: q(z1+z2) = q1+q2+2 z1.z2 and det Q = q1 q2 - (z1.z2)^2 exactly "
                "(symbolic), M_pair^2 = (|q1|^2+|q2|^2+2|z1.z2|^2)/4 + |q1 q2 - (z1.z2)^2|/2 at 30 random pairs (1e-12, against the "
                "singular values); M_pair is not invariant under lG^(1)+lG^(2): its derivative along each of the three boosts of "
                "constituent 2 alone is nonzero at a random pair, while z1.z2 changes and q1, q2 do not")
def _():
    a = sp.symbols("a1:4") + sp.symbols("b1:4")
    c = sp.symbols("c1:4") + sp.symbols("d1:4")
    z1 = sp.Matrix([a[i] + sp.I * a[i + 3] for i in range(3)])
    z2 = sp.Matrix([c[i] + sp.I * c[i + 3] for i in range(3)])
    dot = lambda u, v: (u.T * v)[0]
    ok = sp.expand(dot(z1 + z2, z1 + z2) - dot(z1, z1) - dot(z2, z2) - 2 * dot(z1, z2)) == 0
    Qs = sp.Matrix([[dot(z1, z1), dot(z1, z2)], [dot(z2, z1), dot(z2, z2)]])
    ok &= sp.expand(Qs.det() - (dot(z1, z1) * dot(z2, z2) - dot(z1, z2) ** 2)) == 0
    rng = np.random.default_rng(77)
    for _ in range(30):
        u = rng.normal(size=3) + 1j * rng.normal(size=3)
        v = rng.normal(size=3) + 1j * rng.normal(size=3)
        q1, q2, r = u @ u, v @ v, u @ v
        f = np.sqrt((abs(q1) ** 2 + abs(q2) ** 2 + 2 * abs(r) ** 2) / 4 + abs(q1 * q2 - r ** 2) / 2)
        ok &= abs(Mpair(u, v) - f) < 1e-12 * (1 + f)
    Y = rng.normal(size=12)
    for K in [emb(2, M) for M in LG_K]:
        ok &= abs(dMpair_sq(Y, K)) > 1e-3
        z1n, z2n = zpair(Y)
        d1, d2 = zpair(K @ Y)
        ok &= abs(2 * z1n @ d1) < 1e-12 and abs(2 * z2n @ d2) < 1e-12 and abs(d1 @ z2n + z1n @ d2) > 1e-3
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
                "(so(5)+so(3), so(5) on the positive Lorentz-trivial directions e^(1)_2,e^(1)_3,e^(2)_1,e^(2)_2,e^(2)_3 and so(3) on "
                "e^(2)_4,e^(2)_5,e^(2)_6); g(1,0,1) meets Cent(G_2) in lG^(2) (dim 6); neither so(5) nor so(3) meets Cent(G_2), and "
                "(so(5)+so(3)) meets it in {A + G_2 A G_2^{-1}: A in so(E_2)}, E_2 = span{e^(2)_1,e^(2)_2,e^(2)_3}: the three paired "
                "rotations of constituent 2, su(2), compact and non-abelian ([L_i,L_j] = eps_ijk L_k)")
def _():
    ok = len(GSPEC) == 28 and len(GSPEC_C) == 13
    pos, neg = [1, 2, 6, 7, 8], [9, 10, 11]
    so5 = [rot_vec(12, pos[a], pos[b], ETA2.tolist()) for a, b in itertools.combinations(range(5), 2)]
    so3 = [rot_vec(12, neg[a], neg[b], ETA2.tolist()) for a, b in itertools.combinations(range(3), 2)]
    ok &= span_rank(so5 + so3) == 13 and span_rank(so5 + so3 + GSPEC_C) == 13
    ok &= len(lie_subspace(so5, [lambda X: comm(X, G2)])) == 0 and len(lie_subspace(so3, [lambda X: comm(X, G2)])) == 0
    inter = lie_subspace(GSPEC, [lambda X: comm(X, G2)])
    ok &= len(inter) == 6 and span_rank(inter + LG2) == 6
    interc = lie_subspace(GSPEC_C, [lambda X: comm(X, G2)])
    ok &= len(interc) == 3 and span_rank(interc + LG2_ROT) == 3
    E2 = [6, 7, 8]
    G2inv = -G2
    graph = [A + G2 @ A @ G2inv for A in [rot_vec(12, E2[a], E2[b], ETA2.tolist()) for a, b in itertools.combinations(range(3), 2)]]
    ok &= span_rank(graph + interc) == 3 and span_rank(graph) == 3
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


@claim("e5-3j", "the vanishing odd traces of e5-3c are not a consequence of integer isospin: the so(3) on N = <e^(2)_4,e^(2)_5,"
                "e^(2)_6>, which acts on the Weyl block by isospin doublets (Casimir 3/4, e5-3b), also has Tr(h P_L) = Tr(h P_R) = 0 "
                "and vanishing symmetrised cubic traces Tr(h_(a h_b h_c) P_L,R), h = -w S(X), exactly as the binding-compatible su(2) "
                "does: every su(2) action has weights symmetric under m -> -m; integer isospin excludes doublets (the global anomaly)")
def _():
    neg = [9, 10, 11]
    so3N = [rot_vec(12, neg[a], neg[b], ETA2.tolist()) for a, b in itertools.combinations(range(3), 2)]
    ok = True
    for gens in (so3N, LG2_ROT):
        H = [-W_CS @ spin_lift(M) for M in gens]
        for P2 in (PL_SPEC, PR_SPEC):
            ok &= all(np.trace(h @ P2) == 0 for h in H)
            for a, b, c in itertools.combinations_with_replacement(range(3), 3):
                t = sum(np.trace(H[x] @ H[y] @ H[z] @ P2) for x, y, z in set(itertools.permutations((a, b, c))))
                ok &= t == 0
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


@claim("e5-3i", "the two binding actions at n = 2: under lG^(1) (constituent 2 a spectator, (k_I,k_B,k_S) = (0,1,1)) the commutant in "
                "so(6,6) is so(3,3) of constituent 2 (dim 15, compact part 6), meeting Cent(G_2) in lG^(2) (dim 6) with compact part the 3 "
                "paired rotations of constituent 2; under the diagonal lG ((0,2,0)) the commutant is so(2,C) = span{C_v, eps(x)G} (dim 2), "
                "contained in Cent(G_2), with compact part R C_v (restates RE Prop. 7.8(iii) and Table 8)")
def _():
    c011 = lie_subspace(SO66, [lambda X, L=L: comm(X, L) for L in LG1])
    so33_2 = [emb(2, M) for M in so_basis(6, ETA1)]
    ok = len(c011) == 15 and span_rank(c011 + so33_2) == 15 and len(euclid_compact(c011)) == 6
    i011 = lie_subspace(c011, [lambda X: comm(X, G2)])
    ok &= len(i011) == 6 and span_rank(i011 + LG2) == 6
    ok &= len(euclid_compact(i011)) == 3 and span_rank(euclid_compact(i011) + LG2_ROT) == 3
    c020 = lie_subspace(SO66, [lambda X, L=L: comm(X, L) for L in LGD])
    ok &= len(c020) == 2 and span_rank(c020 + [CV, CBOOST]) == 2
    ok &= all(iszero(comm(X, G2)) for X in c020)
    comp = euclid_compact(c020)
    ok &= len(comp) == 1 and span_rank(comp + [CV]) == 1
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


# ----------------------------------------------------------------------------- 5. two constituents and one rotor
def Nform(a, b, m):
    """Lorentz-class inverse inertia N = (a eta^{-1} + b G eta^{-1}) / m on V^* (eta^{-1} = eta)."""
    return (a * ETA1M + b * (G1 @ ETA1M)) / m


def wfun(p):
    """w(p) = C - i D = q(z(eta^{-1} p))  (ES17 eq. (29))."""
    return p @ ETA1M @ p - 1j * (p @ G1 @ ETA1M @ p)


def expG(t):
    return np.cos(t) * np.eye(6) + np.sin(t) * G1


def rhs_closure(Ns, I0):
    """full one-rotor closure on n constituents: y = (theta, h, p_1..p_n, x_1..x_n)."""
    n = len(Ns)
    GT = G1.T.astype(float)

    def f(t, y):
        th, h = y[0], y[1]
        ps = [y[2 + 6 * k: 8 + 6 * k] for k in range(n)]
        dy = np.zeros_like(y)
        dy[0] = h / I0
        dy[1] = -sum(p @ G1 @ N @ p for p, N in zip(ps, Ns))
        E = expG(th)
        for k in range(n):
            dy[2 + 6 * k: 8 + 6 * k] = (h / I0) * GT @ ps[k]
            dy[2 + 6 * n + 6 * k: 8 + 6 * n + 6 * k] = E @ Ns[k] @ ps[k]
        return dy
    return f


def integrate(Ns, I0, h0, ps, T, npts=2001, events=None):
    n = len(Ns)
    y0 = np.concatenate([[0.0, h0]] + list(ps) + [np.zeros(6)] * n)
    ts = np.linspace(0, T, npts)
    return solve_ivp(rhs_closure(Ns, I0), (0, T), y0, method="DOP853", rtol=1e-12, atol=1e-12, t_eval=ts,
                     events=events, dense_output=True)


def Reff(cs, ps):
    F = sum(c * wfun(p) for c, p in zip(cs, ps))
    return abs(F), np.angle(F), F


@claim("e5-5a", "rotor law at n = 2: (1/2) p^T N p = Re((a+ib) w(p))/(2m) exactly (symbolic, restates ES17 Thm 17.11(ii)); the "
                "Lorentz-class forms invariant under the diagonal lG are exactly Re and Im of the six products z_k.z_l, z_k = z(eta^{-1}p^(k)) "
                "(exact span), so with a diagonal-Lorentz inertia U = (1/2) Re tr(C Q(pi)) with C complex symmetric and the rotor obeys "
                "I_0 theta'' = R sin(2 theta + phi), R e^{i phi} = tr(C Q(pi(0))): checked along a flow with a random C to 1e-9")
def _():
    xs = sp.symbols("p1:7", real=True)
    a, b, m = sp.symbols("a b m", positive=True)
    P = sp.Matrix(xs)
    Gs, Es = sp.Matrix(G1.tolist()), sp.Matrix(ETA1M.tolist())
    N = (a * Es + b * Gs * Es) / m
    lhs = sp.Rational(1, 2) * (P.T * N * P)[0]
    C = (P.T * Es * P)[0]
    D = (P.T * Gs * Es * P)[0]
    ok = sp.expand(lhs - (a * C + b * D) / (2 * m)) == 0
    # Re and Im of z_k.z_l as integer forms on p-space (p = (p1, p2) in R^12); pi_k = eta p_k
    def zz_forms():
        out = []
        for k, l in [(0, 0), (1, 1), (0, 1)]:
            for part in (0, 1):
                S = np.zeros((12, 12))
                Ib = np.eye(12)
                def f(Y, k=k, l=l, part=part):
                    zk = (ETA1M @ Y[6 * k:6 * k + 6]); zl = (ETA1M @ Y[6 * l:6 * l + 6])
                    v = (zk[:3] + 1j * zk[3:]) @ (zl[:3] + 1j * zl[3:])
                    return v.real if part == 0 else v.imag
                for i in range(12):
                    for j in range(12):
                        S[i, j] = f(Ib[i] + Ib[j]) - f(Ib[i]) - f(Ib[j])
                out.append(np.round(S).astype(np.int64))
        return out
    Ss = zz_forms()
    inv = invariant_forms(LGD, 12, G=G2, cls='lor')
    ok &= span_rank(Ss) == 6 and span_rank(Ss + inv) == 6 and len(inv) == 6
    # flow with a random complex symmetric C: U = 1/2 Re sum C_kl z_k.z_l  <->  N = sum Re/Im parts
    Cc = np.array([[0.7 + 0.2j, -0.4 + 0.5j], [-0.4 + 0.5j, 1.1 - 0.3j]])
    coef = [Cc[0, 0], Cc[1, 1], 2 * Cc[0, 1]]
    Nbig = np.zeros((12, 12))
    for i, c in enumerate(coef):
        # Y^T S Y = 2 f(Y), so Y^T [sum 0.5 (Re c S_re - Im c S_im)] Y = Re tr(C Q) and U = 1/2 Y^T N Y = 1/2 Re tr(C Q)
        Nbig += 0.5 * (c.real * Ss[2 * i] - c.imag * Ss[2 * i + 1])
    ok &= np.allclose(Nbig @ G2 + G2 @ Nbig, 0)
    I0, h0 = 1.3, 0.8
    p0 = RNG.normal(size=12)
    G2f = G2.astype(float)

    def f(t, y):
        th, h, p = y[0], y[1], y[2:]
        return np.concatenate([[h / I0, -p @ G2f @ Nbig @ p], (h / I0) * G2f.T @ p])
    sol = solve_ivp(f, (0, 20), np.concatenate([[0, h0], p0]), method="DOP853", rtol=1e-12, atol=1e-12,
                    t_eval=np.linspace(0, 20, 801))
    def Qmat(p):
        z = [(ETA1M @ p[6 * k:6 * k + 6]) for k in (0, 1)]
        z = [v[:3] + 1j * v[3:] for v in z]
        return np.array([[z[0] @ z[0], z[0] @ z[1]], [z[1] @ z[0], z[1] @ z[1]]])
    F0 = np.trace(Cc @ Qmat(p0))
    ok &= abs(0.5 * p0 @ Nbig @ p0 - 0.5 * F0.real) < 1e-12 * (1 + p0 @ p0)
    R, ph = abs(F0), np.angle(F0)
    err = 0.0
    for th, h, p in zip(sol.y[0], sol.y[1], sol.y[2:].T):
        hd = -p @ G2f @ Nbig @ p
        err = max(err, abs(I0 * 0 + hd - R * np.sin(2 * th + ph)))
    ok &= err < 1e-9
    return ok


DATA_RNG = np.random.default_rng(1729)
P1_BAL = DATA_RNG.normal(size=6)
P2_BAL = -(G1 @ P1_BAL)                                   # p^(2) = -G p^(1)
A_, B_, M_ = 1.0, 0.3, 1.0                                # equal inertias, N = (a eta^{-1} + b G eta^{-1}) / m
N_EQ = Nform(A_, B_, M_)
C_EQ = (A_ + 1j * B_) / M_


@claim("e5-5b", "balanced pair, equal inertias, p^(2) = -G p^(1), I_0 = 1, h(0) = 2 (frames included, DOP853, rtol = atol = 1e-12, "
                "tau in [0,60]): w_2 = -w_1 exactly, both constituents massive (E_0(pi^(k)) = |w_k|/2 > 0), rate constant to 1e-11, frames on "
                "the circles x_k = (e^{2 kappa tau G} - 1)(2 kappa G)^{-1} N p^(k)(0) to 1e-9 (restates RE Thm 8.1(i), d6-2a); a generic "
                "balanced pair with unequal inertias (m = 1, 2.5; a+ib = 1+0.3i, 0.4-0.7i) gives the same to the same tolerances")
def _():
    ok = abs(wfun(P2_BAL) + wfun(P1_BAL)) < 1e-12 and abs(wfun(P1_BAL)) > 0.1
    cases = [([N_EQ, N_EQ], [P1_BAL, P2_BAL])]
    # unequal inertias: adjust p2 by a scale and a duality rotation so that c1 w1 + c2 w2 = 0
    c1, c2 = (1 + 0.3j) / 1.0, (0.4 - 0.7j) / 2.5
    N1, N2 = Nform(1.0, 0.3, 1.0), Nform(0.4, -0.7, 2.5)
    q1 = DATA_RNG.normal(size=6)
    q2 = DATA_RNG.normal(size=6)
    target = -c1 * wfun(q1) / c2                         # want w(q2') = target; w(s e^{alpha G} q) = s^2 e^{-2 i alpha} w(q)
    ratio = target / wfun(q2)
    s = np.sqrt(abs(ratio))
    alpha = -np.angle(ratio) / 2
    q2b = s * (expG(alpha) @ q2)
    ok &= abs(c1 * wfun(q1) + c2 * wfun(q2b)) < 1e-12
    cases.append(([N1, N2], [q1, q2b]))
    for Ns, ps in cases:
        sol = integrate(Ns, 1.0, 2.0, ps, 60.0)
        ok &= sol.success
        h = sol.y[1]
        ok &= np.max(np.abs(h - 2.0)) < 1e-11
        kap = 2.0
        for k in range(2):
            xk = sol.y[14 + 6 * k: 20 + 6 * k]
            Np = Ns[k] @ ps[k]
            inv2kG = np.linalg.inv(2 * kap * G1)
            pred = np.array([(expG(2 * kap * t) - np.eye(6)) @ inv2kG @ Np for t in sol.t]).T
            ok &= np.max(np.abs(xk - pred)) < 1e-9
    return ok


def period_event(theta_target):
    def ev(t, y):
        return y[0] - theta_target
    ev.terminal = True
    ev.direction = 1
    return ev


def _periodic_mean(f, M=4096):
    """2 pi times the mean of a smooth 2 pi-periodic function on M equispaced points (spectrally accurate)."""
    psi = np.linspace(0.0, 2 * np.pi, M, endpoint=False)
    return 2 * np.pi * np.mean(f(psi))


def lam_eff(eps, psi0, kappa0):
    """lambda of ES17 Thm 17.16: (1/2) int_0^{2pi} cos psi dpsi / (|kappa0| sqrt(1 + eps (cos psi0 - cos psi)))."""
    return 0.5 * _periodic_mean(lambda psi: np.cos(psi) / (abs(kappa0) * np.sqrt(1 + eps * (np.cos(psi0) - np.cos(psi)))))


def period_eff(eps, psi0, kappa0):
    """duration of one period of the rate (theta: 0 -> pi): (1/2) int_0^{2pi} dpsi / (|kappa0| sqrt(...))."""
    return 0.5 * _periodic_mean(lambda psi: 1.0 / (abs(kappa0) * np.sqrt(1 + eps * (np.cos(psi0) - np.cos(psi)))))


U_PERT = np.random.default_rng(4242).normal(size=6)        # own generator: independent of the order of the checks


def unbalanced(delta):
    return [P1_BAL, P2_BAL + delta * U_PERT]


@claim("e5-5c", "off the balance surface (equal inertias, p^(2) = -G p^(1) + delta u, delta = 0.05, h(0) = 2): R_eff = |sum c_k w_k| and "
                "the relative phase w_1 conj(w_2) are constant (1e-10), the rotor obeys I_0 theta'' = R_eff sin(2 theta + phi_eff) to 1e-9, "
                "kappa^2 = kappa_0^2 [1 + eps (cos psi_0 - cos psi)] with eps = R_eff/(I_0 kappa_0^2) to 1e-9, |kappa^2/kappa_0^2 - 1| <= 2 eps, "
                "and over one period of the rate (theta: 0 -> pi) each frame moves by lambda_eff e^{-phi_eff G} N_k p^(k)(0) with lambda_eff "
                "of ES17 Thm 17.16 at eps (relative 1e-8): the several-constituent pendulum and drift")
def _():
    ps = unbalanced(0.05)
    R, ph, F = Reff([C_EQ, C_EQ], ps)
    ok = R > 1e-3
    I0, h0 = 1.0, 2.0
    sol = integrate([N_EQ, N_EQ], I0, h0, ps, 30.0)
    ok &= sol.success
    eps = R / (I0 * h0 ** 2)
    psi0 = ph
    for t, y in zip(sol.t, sol.y.T):
        th, h = y[0], y[1]
        p1, p2 = y[2:8], y[8:14]
        Rt, _, Ft = Reff([C_EQ, C_EQ], [p1, p2])
        ok &= abs(Rt - R) < 1e-10
        ok &= abs(wfun(p1) * np.conj(wfun(p2)) - wfun(ps[0]) * np.conj(wfun(ps[1]))) < 1e-10 * (1 + abs(wfun(ps[0])) ** 2)
        hd = -(p1 @ G1 @ N_EQ @ p1 + p2 @ G1 @ N_EQ @ p2)
        ok &= abs(hd - R * np.sin(2 * th + ph)) < 1e-9
        psi = 2 * th + ph
        ok &= abs((h / I0) ** 2 - h0 ** 2 * (1 + eps * (np.cos(psi0) - np.cos(psi)))) < 1e-9
        ok &= abs((h / h0) ** 2 - 1) <= 2 * eps + 1e-12
    # one period of the rate
    sol2 = integrate([N_EQ, N_EQ], I0, h0, ps, 10.0, events=period_event(np.pi))
    ok &= len(sol2.t_events[0]) == 1
    yT = sol2.y_events[0][0]
    lam = lam_eff(eps, psi0, h0 / I0)
    for k in range(2):
        pred = lam * (expG(-ph) @ N_EQ @ ps[k])
        dx = yT[14 + 6 * k: 20 + 6 * k]
        ok &= np.linalg.norm(dx - pred) < 1e-8 * np.linalg.norm(pred)
    # rate and momenta return after the period; theta has advanced by pi
    ok &= abs(yT[1] - h0) < 1e-9 and np.allclose(yT[2:14], -np.concatenate(ps), atol=1e-9)
    return ok


@claim("e5-5d", "transverse stability of the balance surface (equal inertias, h(0) = 2, delta = 1e-3, 2e-3, 4e-3, 5e-2): R_eff is a "
                "constant of motion, so the surface and each level R_eff = r are invariant (neither attracting nor repelling); the rate "
                "excursion obeys max|h - h_0| <= R_eff/|kappa_0| (I_0 = 1); R_eff, the excursion and the mean frame velocity over a period "
                "scale linearly in delta (doubling ratios 2 within 2%); the period equals the quadrature (1/2) int dpsi/(|kappa_0| sqrt(..)) "
                "and the mean velocity equals lambda_eff |N_k p^(k)(0)| / period (relative 1e-8); to first order it is (eps/4)|N_k p^(k)(0)| "
                "(relative deviation <= 1.5 eps) and to second order (eps/4)(1 - eps cos psi_0)|N_k p^(k)(0)| (relative deviation <= eps^2): "
                "the rate is Lyapunov stable, the frames are not (secular drift linear in the imbalance)")
def _():
    ok = True
    I0, h0 = 1.0, 2.0
    out = []
    for d in (1e-3, 2e-3, 4e-3, 5e-2):
        ps = unbalanced(d)
        R, ph, _ = Reff([C_EQ, C_EQ], ps)
        sol = integrate([N_EQ, N_EQ], I0, h0, ps, 10.0, npts=11, events=period_event(np.pi))
        T = sol.t_events[0][0]
        yT = sol.y_events[0][0]
        exc = np.max(np.abs(sol.sol(np.linspace(0, T, 4001))[1] - h0))
        ok &= exc <= (1 + 1e-9) * R / h0
        v = [np.linalg.norm(yT[14 + 6 * k: 20 + 6 * k]) / T for k in range(2)]
        eps = R / (I0 * h0 ** 2)
        kap0 = h0 / I0
        Tq = period_eff(eps, ph, kap0)
        lam = lam_eff(eps, ph, kap0)
        ok &= abs(T / Tq - 1) < 1e-8
        for k in range(2):
            nk = np.linalg.norm(N_EQ @ ps[k])
            ok &= abs(v[k] / (lam * nk / Tq) - 1) < 1e-8
            first = eps / 4 * nk
            ok &= abs(v[k] / first - 1) <= 1.5 * eps
            ok &= abs(v[k] / (first * (1 - eps * np.cos(ph))) - 1) <= eps ** 2
        if d < 1e-2:
            out.append((R, exc, v[0], v[1]))
    for i in range(2):
        for j in range(4):
            ok &= abs(out[i + 1][j] / out[i][j] - 2) < 2e-2
    # exactly balanced: no drift over a period
    sol = integrate([N_EQ, N_EQ], I0, h0, [P1_BAL, P2_BAL], 10.0, events=period_event(np.pi))
    yT = sol.y_events[0][0]
    ok &= np.linalg.norm(yT[14:26]) < 1e-10
    return ok


@claim("e5-5e", "libration and the h(0) = 0 precision: off balance with h(0) = 0 the rotor librates (h changes sign, theta stays in an "
                "interval of length < pi, period 4K(sin^2(xi_0/2))/Omega with Omega^2 = 2 R_eff/I_0 to 1e-8) and the frames drift at this "
                "amplitude (the general law is e5-5i); on the "
                "balance surface with h(0) = 0 the state is an equilibrium of the rotor (|h| < 1e-12) while each frame moves uniformly, "
                "x_k = tau N_k p^(k)(0) (to 1e-10): bounded frames on the balance surface need h(0) != 0")
def _():
    from scipy.special import ellipk
    ok = True
    I0 = 1.0
    ps = unbalanced(0.05)
    R, ph, _ = Reff([C_EQ, C_EQ], ps)
    sol = integrate([N_EQ, N_EQ], I0, 0.0, ps, 200.0, npts=40001)
    th, h = sol.y[0], sol.y[1]
    ok &= np.min(h) < -1e-3 and np.max(h) > 1e-3 and (np.max(th) - np.min(th)) < np.pi
    Omega = np.sqrt(2 * R / I0)
    xi0 = abs(((ph + np.pi) % (2 * np.pi)) - np.pi)       # psi_0 - pi folded: amplitude of xi = psi - pi
    xi0 = np.pi - xi0 if False else abs(np.angle(np.exp(1j * (ph - np.pi))))
    Tpred = 4 * ellipk(np.sin(xi0 / 2) ** 2) / Omega
    # measured period of h: zero crossings of theta' with the same sign of slope
    ev = lambda t, y: y[1]
    ev.direction = -1
    sol2 = solve_ivp(rhs_closure([N_EQ, N_EQ], I0), (0, 4 * Tpred),
                     np.concatenate([[0.0, 0.0]] + ps + [np.zeros(12)]), method="DOP853", rtol=1e-12, atol=1e-12,
                     events=ev)
    te = sol2.t_events[0]
    te = te[te > 1e-6]
    ok &= len(te) >= 2 and abs((te[1] - te[0]) / Tpred - 1) < 1e-8
    x1 = sol.y[14:20, -1]
    ok &= np.linalg.norm(x1) > 1.0
    # balanced, h(0) = 0
    sol = integrate([N_EQ, N_EQ], I0, 0.0, [P1_BAL, P2_BAL], 20.0)
    ok &= np.max(np.abs(sol.y[1])) < 1e-12
    for k in range(2):
        pred = np.outer(N_EQ @ [P1_BAL, P2_BAL][k], sol.t)
        ok &= np.max(np.abs(sol.y[14 + 6 * k: 20 + 6 * k] - pred)) < 1e-10
    return ok


def dF_rank(cs, ps):
    """real rank of dF, F = sum c_k w(p^(k)), at p (as a map R^{6n} -> R^2)."""
    n = len(ps)
    J = np.zeros((2, 6 * n))
    for k in range(n):
        g = 2 * cs[k] * (ETA1M @ ps[k] - 1j * ((G1 @ ETA1M + (G1 @ ETA1M).T) / 2) @ ps[k])
        # w(p) = p^T eta p - i p^T (G eta) p; gradient of p^T A p is (A + A^T) p
        g = cs[k] * (2 * ETA1M @ ps[k] - 1j * ((G1 @ ETA1M) + (G1 @ ETA1M).T) @ ps[k])
        J[0, 6 * k:6 * k + 6] = g.real
        J[1, 6 * k:6 * k + 6] = g.imag
    s = np.linalg.svd(J, compute_uv=False)
    return int(np.sum(s > 1e-9 * max(s[0], 1e-300)))


def random_balanced(cs, rng):
    q1 = rng.normal(size=6)
    q2 = rng.normal(size=6)
    target = -cs[0] * wfun(q1) / cs[1]
    ratio = target / wfun(q2)
    return [q1, np.sqrt(abs(ratio)) * (expG(-np.angle(ratio) / 2) @ q2)]


def random_null(rng):
    """a random p in R^6 with w(p) = 0: pi = eta p with z = u + i v, |u| = |v|, u . v = 0."""
    u = rng.normal(size=3)
    v = np.cross(u, rng.normal(size=3))
    v *= np.linalg.norm(u) / np.linalg.norm(v)
    return ETA1M @ np.concatenate([u, v])


@claim("e5-5f", "geometry of the balance surface at n = 2 (20 random points each): dF has real rank 2 on B \\ 0 for equal and for "
                "unequal coefficients (codimension 2; restates RE Thm 8.1(iii)); at equal coefficients the so(6,C)-orbit through a "
                "point of B \\ 0 has real dimension 10 = dim B, so B \\ 0 = {Z.Z = 0, Z != 0} is one orbit, connected, = R_{>0} x V_2(R^6); "
                "the all-null part N x N has dimension 8 (rank of d(w_1,w_2) = 4 there); on the massive part w_1 : B -> C^* is a "
                "submersion (rank 2), whose fibre over s is the product of the two complex spheres {w = s} and {w = -c_1 s/c_2}")
def _():
    ok = True
    rng = np.random.default_rng(99)
    cases = [[C_EQ, C_EQ], [(1 + 0.3j) / 1.0, (0.4 - 0.7j) / 2.5]]
    for cs in cases:
        for _ in range(20):
            ps = random_balanced(cs, rng)
            ok &= abs(cs[0] * wfun(ps[0]) + cs[1] * wfun(ps[1])) < 1e-11 * (1 + sum(p @ p for p in ps))
            ok &= dF_rank(cs, ps) == 2
    # orbit dimension at equal coefficients: tangent vectors X Y for X in Cent(G_2), acting on momenta p by -X^T
    for _ in range(20):
        ps = random_balanced([1.0, 1.0], rng)
        Y = np.concatenate(ps)
        T_ = np.array([(-X.T) @ Y for X in CENT])
        s = np.linalg.svd(T_, compute_uv=False)
        ok &= int(np.sum(s > 1e-9 * s[0])) == 10 and s[9] / s[0] > 1e-6
    # the null part: rank of d(w1, w2) as R^12 -> R^4 at points of N x N
    for _ in range(20):
        ps = [random_null(rng), random_null(rng)]
        ok &= abs(wfun(ps[0])) < 1e-12 and abs(wfun(ps[1])) < 1e-12
        J = np.zeros((4, 12))
        for k in range(2):
            g = 2 * ETA1M @ ps[k] - 1j * ((G1 @ ETA1M) + (G1 @ ETA1M).T) @ ps[k]
            J[2 * k, 6 * k:6 * k + 6], J[2 * k + 1, 6 * k:6 * k + 6] = g.real, g.imag
        s = np.linalg.svd(J, compute_uv=False)
        ok &= int(np.sum(s > 1e-9 * s[0])) == 4
    # w1 restricted to B_massive is a submersion: rank of (dF; dw1) = 4
    for cs in cases:
        for _ in range(20):
            ps = random_balanced(cs, rng)
            J = np.zeros((4, 12))
            for k in range(2):
                g = cs[k] * (2 * ETA1M @ ps[k] - 1j * ((G1 @ ETA1M) + (G1 @ ETA1M).T) @ ps[k])
                J[0, 6 * k:6 * k + 6], J[1, 6 * k:6 * k + 6] = g.real, g.imag
            g = 2 * ETA1M @ ps[0] - 1j * ((G1 @ ETA1M) + (G1 @ ETA1M).T) @ ps[0]
            J[2, 0:6], J[3, 0:6] = g.real, g.imag
            s = np.linalg.svd(J, compute_uv=False)
            ok &= int(np.sum(s > 1e-9 * s[0])) == 4
    return ok


SWAP = np.zeros((12, 12), dtype=np.int64)
SWAP[0:6, 6:12] = np.eye(6, dtype=np.int64)
SWAP[6:12, 0:6] = np.eye(6, dtype=np.int64)


@claim("e5-5g", "exchange of the constituents: the swap sigma is an eta_2-isometry commuting with G_2 (so with the rotor), preserves the "
                "equal-inertia Hamiltonian and the closure bracket, and as an element of O(6,C) has determinant -1; sigma maps "
                "V_+ = {p^(2) = -G p^(1)} onto V_- = {p^(2) = G p^(1)}; exchange-even and -odd data p^(2) = +-p^(1) both have w_2 = w_1, "
                "so R_eff = 2|c||w_1| and they are balanced only when null; with equal coefficients R_eff = 2|c||w_s|, w_s = (w_1+w_2)/2 "
                "the exchange-even part of (w_1, w_2): balanced iff (w_1, w_2) is exchange-odd")
def _():
    ok = iszero(SWAP.T @ ETA2M @ SWAP - ETA2M) and iszero(comm(SWAP, G2))
    NB = blockdiag(N_EQ, N_EQ)
    ok &= np.allclose(SWAP @ NB @ SWAP.T, NB)
    cf = complex_form(SWAP) if False else None
    # complex 6x6 matrix of the swap on (z1, z2): permutation of three pairs
    R = SWAP[np.ix_(TEMP, TEMP)].astype(float)
    ok &= iszero(SWAP[np.ix_(PART, PART)] - SWAP[np.ix_(TEMP, TEMP)]) and iszero(SWAP[np.ix_(TEMP, PART)])
    ok &= abs(np.linalg.det(R) + 1) < 1e-12
    rng = np.random.default_rng(5)
    for _ in range(10):
        p1 = rng.normal(size=6)
        Y = np.concatenate([p1, -G1 @ p1])
        Ys = SWAP @ Y
        ok &= np.allclose(Ys[6:], G1 @ Ys[:6])
        for sgn in (1, -1):
            ok &= abs(wfun(sgn * p1) - wfun(p1)) < 1e-12
            ok &= abs(Reff([C_EQ, C_EQ], [p1, sgn * p1])[0] - 2 * abs(C_EQ) * abs(wfun(p1))) < 1e-12
        p2 = rng.normal(size=6)
        ws = (wfun(p1) + wfun(p2)) / 2
        ok &= abs(Reff([C_EQ, C_EQ], [p1, p2])[0] - 2 * abs(C_EQ) * abs(ws)) < 1e-12
    return ok


@claim("e5-5i", "libration drift, Thm stab (iv) as corrected: off the surface with h(0) = 0 and psi_0 = pi - xi_0, the displacement "
                "of each frame after 1, 2, 3 periods 4K(m)/Omega (m = sin^2(xi_0/2), Omega^2 = 2R_eff/I_0) equals j T (1 - 2E(m)/K(m)) "
                "e^{-phi_eff G} N_k p^(k)(0) to relative 1e-9 at xi_0 = 0.6, 1.5, 2.6 and at the root of 2E(m) = K(m) (m = 0.8261, "
                "xi_0 = 130.7 deg), where it vanishes: the frames drift except at that amplitude, where they return after every "
                "period; at psi_0 = 0 the datum is an equilibrium of (theta, h, p)")
def _():
    from scipy.special import ellipk, ellipe
    from scipy.optimize import brentq
    ok = True
    I0 = 1.0
    mstar = brentq(lambda m: 2 * ellipe(m) - ellipk(m), 0.5, 0.99)
    xistar = 2 * np.arcsin(np.sqrt(mstar))
    ok &= abs(mstar - 0.8261) < 1e-4 and abs(np.degrees(xistar) - 130.7) < 0.05
    base = unbalanced(0.05)
    R, ph, _ = Reff([C_EQ, C_EQ], base)
    for xi0 in (0.6, 1.5, 2.6, xistar):
        al = (np.pi - xi0 - ph) / 2                      # w -> e^{2 i al} w under p -> e^{-al G} p
        ps = [expG(-al) @ p for p in base]
        R_, ph_, _ = Reff([C_EQ, C_EQ], ps)
        ok &= abs(R_ - R) < 1e-12 and abs(np.angle(np.exp(1j * (ph_ - (np.pi - xi0))))) < 1e-12
        m = np.sin(xi0 / 2) ** 2
        T = 4 * ellipk(m) / np.sqrt(2 * R_ / I0)
        sol = integrate([N_EQ, N_EQ], I0, 0.0, ps, 3 * T + 1e-9, npts=11)
        ok &= sol.success
        for k in range(2):
            Np = N_EQ @ ps[k]
            pred = (1 - 2 * ellipe(m) / ellipk(m)) * (expG(-ph_) @ Np) * T
            for j in (1, 2, 3):
                xk = sol.sol(j * T)[14 + 6 * k: 20 + 6 * k]
                ok &= np.linalg.norm(xk - j * pred) < 1e-9 * np.linalg.norm(Np) * T
            if xi0 == xistar:
                ok &= np.linalg.norm(pred) < 1e-12 * np.linalg.norm(Np) * T
            else:
                ok &= np.linalg.norm(pred) > 1e-2 * np.linalg.norm(Np) * T
    al = (0.0 - ph) / 2
    ps = [expG(-al) @ p for p in base]
    y0 = np.concatenate([[0.0, 0.0]] + ps + [np.zeros(12)])
    ok &= np.max(np.abs(rhs_closure([N_EQ, N_EQ], I0)(0.0, y0)[:14])) < 1e-12
    return ok


@claim("e5-5j", "exchange eigenstates for general coefficients: F(p, +-p) = (c_1 + c_2) w(p) (symbolic), so for c_2 = -c_1 every "
                "exchange eigenstate is balanced, massive ones included (10 random data); the statement 'a massive balanced pair is "
                "never an exchange eigenstate' holds exactly when c_1 + c_2 != 0, in particular for equal inertias")
def _():
    ps_ = sp.symbols("p1:7", real=True)
    c1, c2 = sp.symbols("c1 c2")
    P = sp.Matrix(ps_)
    Es, Gs = sp.Matrix(ETA1M.tolist()), sp.Matrix(G1.tolist())
    wsym = lambda Q: sp.expand((Q.T * Es * Q)[0] - sp.I * (Q.T * Gs * Es * Q)[0])
    ok = all(sp.expand(c1 * wsym(P) + c2 * wsym(s_ * P) - (c1 + c2) * wsym(P)) == 0 for s_ in (1, -1))
    rng = np.random.default_rng(12)
    for _ in range(10):
        p1 = rng.normal(size=6)
        ok &= abs(wfun(p1)) > 1e-6
        for s_ in (1, -1):
            ok &= Reff([C_EQ, -C_EQ], [p1, s_ * p1])[0] < 1e-12 and Reff([C_EQ, C_EQ], [p1, s_ * p1])[0] > 1e-6
    return ok


@claim("e5-5h", "carry-over of the rest-closure results: Gamma_12 = E_1...E_12 satisfies Gamma_12^2 = 1 and anticommutes with all twelve "
                "generators, so it implements e^{pi G_2} = -id on S_2 (the half turn of RE Prop. 4.14 at n = 2); "
                "G_2^T eta_2 = eta_2 G_2 and G_2^T eta_2 G_2 = -eta_2, so (e^{theta G_2})^T eta_2 e^{theta G_2} = cos 2theta eta_2 + "
                "sin 2theta eta_2 G_2 and the rotor lifts to Pin(6,6) only at theta in pi Z (RE Prop. 4.10(ii) at n = 2); the coadjoint "
                "orbits of s_2 at p != 0 are two-dimensional (rank of the Lie-Poisson tensor = 2 at 10 random points), so there are "
                "11 = 13 - 2 local Casimirs, and the real and imaginary parts of conj(z_j) z_k, z = (z(pi^(1)), z(pi^(2))) in C^6, provide "
                "them (Jacobian rank 11 at 10 random points; 5 at n = 1, RE Thm 4.1(ii)) and are invariant under p -> e^{tG_2} p")
def _():
    rng0 = np.random.default_rng(81)

    def herm_funcs(p, n):
        z = []
        for k in range(n):
            pi = ETA1M @ p[6 * k:6 * k + 6]
            z.extend(list(pi[:3] + 1j * pi[3:]))
        z = np.array(z)
        H = np.outer(np.conj(z), z)
        iu = np.triu_indices(3 * n)
        return np.concatenate([H[iu].real, H[np.triu_indices(3 * n, 1)].imag])

    okc = True
    for n, r in [(1, 5), (2, 11)]:
        for _ in range(10):
            p = rng0.normal(size=6 * n)
            J = np.array([(herm_funcs(p + 1e-6 * e_, n) - herm_funcs(p - 1e-6 * e_, n)) / 2e-6 for e_ in np.eye(6 * n)]).T
            s = np.linalg.svd(J, compute_uv=False)
            okc &= int(np.sum(s > 1e-6 * s[0])) == r
            Gk = Gn(n).astype(float)
            Et = np.cos(0.7) * np.eye(6 * n) + np.sin(0.7) * Gk
            okc &= np.allclose(herm_funcs(Et @ p, n), herm_funcs(p, n), atol=1e-12)
    if not okc:
        return False
    ok = np.array_equal(GAMMA12 @ GAMMA12, I64)
    ok &= all(np.array_equal(GAMMA12 @ g_ + g_ @ GAMMA12, 0 * I64) for g_ in GEN)
    ok &= iszero(G2.T @ ETA2M - ETA2M @ G2) and iszero(G2.T @ ETA2M @ G2 + ETA2M)
    for t in (0.3, 1.1, np.pi):
        Et = np.cos(t) * np.eye(12) + np.sin(t) * G2
        ok &= np.allclose(Et.T @ ETA2M @ Et, np.cos(2 * t) * ETA2M + np.sin(2 * t) * ETA2M @ G2, atol=1e-13)
    # Lie-Poisson tensor of s_2 = R x_G V_2 at (h, p): {h, p_a} = (G^T p)_a-type; only brackets with h are nonzero
    rng = np.random.default_rng(8)
    for _ in range(10):
        p = rng.normal(size=12)
        Pi = np.zeros((13, 13))
        v = G2.T @ p
        Pi[0, 1:] = v
        Pi[1:, 0] = -v
        ok &= np.linalg.matrix_rank(Pi) == 2
    return ok


# ----------------------------------------------------------------------------- 6. the ladder n = 1, 2, 3
@claim("e5-6a", "ladder entries (exact): n = 1: Cent(lint) in so(3,3) = R J (J = e2e3), which meets Cent(G) in 0, and Cent(G) meets lint in 0; "
                "n = 3: Cent(G_3) meets lint^(1) in 0, and for the Pati-Salam algebra of P_* (U = span{e_7,e_13,e_14,e_15}, so(6) on the colour "
                "directions {10,11,12,16,17,18}) ps_U meets Cent(G_3) in 6 dimensions, all compact (so4_Delta, restates ES17 Thm 17.21); "
                "the compact spectator algebra (Euclidean-antisymmetric part of Cent(lint^(1)), dim 43 = so(8)+so(6)) meets Cent(G_3) in "
                "the 15 paired rotations of constituents 2, 3 (so(6)), and the diagonal connection algebra Cent(diag lint) (dim 18) meets "
                "it in the 3 copy rotations (E_lk - E_kl) (x) 1_6: the binding-compatible parts are so(3n-3) and so(n) at n = 1, 2, 3; "
                "balance: rank dF = 2 at random nonzero points for n = 1, 2, 3 and on B for n = 2, 3, while for n = 1 B = {w = 0}")
def _():
    ok = True
    eta3l = np.diag(etan(3)).tolist()
    so99 = so_basis(18, eta3l)
    G3_ = Gn(3)
    lint31_ = [rot_vec(18, INT0[a], INT0[b], eta3l) for a in range(4) for b in range(a + 1, 4)]
    spec3 = lie_subspace(so99, [lambda X, L=L: comm(X, L) for L in lint31_])
    spec3c = euclid_compact(spec3)
    ok &= len(spec3) == 91 and len(spec3c) == 43
    bc3 = lie_subspace(spec3c, [lambda X: comm(X, G3_)])
    temp23, part23 = [6, 7, 8, 12, 13, 14], [9, 10, 11, 15, 16, 17]
    pr23 = []
    for a, b in itertools.combinations(range(6), 2):
        R_ = np.zeros((18, 18), dtype=np.int64)
        R_[temp23[a], temp23[b]], R_[temp23[b], temp23[a]] = -1, 1
        R_[part23[a], part23[b]], R_[part23[b], part23[a]] = -1, 1
        pr23.append(R_)
    ok &= len(bc3) == 15 and span_rank(bc3 + pr23) == 15
    lintd =[sum(rot_vec(18, 6 * k + INT0[a], 6 * k + INT0[b], eta3l) for k in range(3)) for a in range(4) for b in range(a + 1, 4)]
    diag3 = lie_subspace(so99, [lambda X, L=L: comm(X, L) for L in lintd])
    ok &= len(diag3) == 18 and len(euclid_compact(diag3)) == 18
    dbc3 = lie_subspace(diag3, [lambda X: comm(X, G3_)])
    copyrots = []
    for k, l in [(0, 1), (0, 2), (1, 2)]:
        B_ = np.zeros((3, 3), dtype=np.int64)
        B_[l, k], B_[k, l] = 1, -1
        copyrots.append(np.kron(B_, np.eye(6, dtype=np.int64)))
    ok &= len(dbc3) == 3 and span_rank(dbc3 + copyrots) == 3
    so33 = so_basis(6, ETA1)
    lint1 = [rot_vec(6, INT0[a], INT0[b], ETA1) for a in range(4) for b in range(a + 1, 4)]
    cl = lie_subspace(so33, [lambda X, L=L: comm(X, L) for L in lint1])
    Jv = rot_vec(6, 1, 2, ETA1)
    ok &= len(cl) == 1 and span_rank(cl + [Jv]) == 1
    ok &= len(lie_subspace(cl, [lambda X: comm(X, G1)])) == 0
    ok &= len(lie_subspace(lint1, [lambda X: comm(X, G1)])) == 0
    G3 = Gn(3)
    eta3 = np.diag(etan(3)).tolist()
    lint31 = [rot_vec(18, INT0[a], INT0[b], eta3) for a in range(4) for b in range(a + 1, 4)]
    ok &= len(lie_subspace(lint31, [lambda X: comm(X, G3)])) == 0
    W = [6, 12, 13, 14]                           # 0-based of 7, 13, 14, 15
    col = [9, 10, 11, 15, 16, 17]                 # 0-based of 10, 11, 12, 16, 17, 18
    ps_ = [rot_vec(18, W[a], W[b], eta3) for a in range(4) for b in range(a + 1, 4)] + \
          [rot_vec(18, col[a], col[b], eta3) for a in range(6) for b in range(a + 1, 6)]
    inter = lie_subspace(ps_, [lambda X: comm(X, G3)])
    ok &= len(inter) == 6 and len(euclid_compact(inter)) == 6
    rng = np.random.default_rng(31)
    for n in (1, 2, 3):
        cs = [complex(*rng.normal(size=2)) for _ in range(n)]
        for _ in range(5):
            ps = [rng.normal(size=6) for _ in range(n)]
            ok &= dF_rank(cs, ps) == 2
            if n >= 2:
                q = [rng.normal(size=6) for _ in range(n - 2)]
                bal = random_balanced(cs[:2], rng)
                # rescale so that the first two absorb the rest: c1 w1 + c2 w2 = -sum_{k>2} c_k w_k
                if n == 3:
                    rest = cs[2] * wfun(q[0])
                    q1 = rng.normal(size=6)
                    tgt = -(cs[0] * wfun(q1) + rest) / cs[1]
                    q2 = rng.normal(size=6)
                    ratio = tgt / wfun(q2)
                    bal = [q1, np.sqrt(abs(ratio)) * (expG(-np.angle(ratio) / 2) @ q2), q[0]]
                Fv = sum(c * wfun(p) for c, p in zip(cs, bal))
                ok &= abs(Fv) < 1e-10 and dF_rank(cs, bal) == 2
    return ok


@claim("e5-6b", "binding-compatible parts at every rung (the proofs of the spectator and diagonal propositions apply verbatim), "
                "exact at n = 4: the compact spectator algebra meets Cent(G_4) in 36 = dim so(9) dimensions, the paired rotations of "
                "constituents 2, 3, 4, and the diagonal connection algebra meets it in 6 = dim so(4), the copy rotations")
def _():
    N4 = 24
    eta4 = np.diag(etan(4)).tolist()
    comp = [rot_vec(N4, a, b, eta4) for a in range(N4) for b in range(a + 1, N4) if eta4[a] == eta4[b]]
    G4 = Gn(4)
    lint41 = [rot_vec(N4, INT0[a], INT0[b], eta4) for a in range(4) for b in range(a + 1, 4)]
    spec = lie_subspace(comp, [lambda X, L=L: comm(X, L) for L in lint41] + [lambda X: comm(X, G4)])
    temp = [6 * k + i for k in (1, 2, 3) for i in range(3)]
    pr = []
    for a, b in itertools.combinations(temp, 2):
        R_ = np.zeros((N4, N4), dtype=np.int64)
        R_[a, b], R_[b, a] = -1, 1
        R_[a + 3, b + 3], R_[b + 3, a + 3] = -1, 1
        pr.append(R_)
    ok = len(spec) == 36 and span_rank(spec + pr) == 36
    lintd = [sum(rot_vec(N4, 6 * k + INT0[a], 6 * k + INT0[b], eta4) for k in range(4)) for a in range(4) for b in range(a + 1, 4)]
    dg = lie_subspace(comp, [lambda X, L=L: comm(X, L) for L in lintd] + [lambda X: comm(X, G4)])
    copyrots = []
    for k, l in itertools.combinations(range(4), 2):
        B_ = np.zeros((4, 4), dtype=np.int64)
        B_[l, k], B_[k, l] = 1, -1
        copyrots.append(np.kron(B_, np.eye(6, dtype=np.int64)))
    ok &= len(dg) == 6 and span_rank(dg + copyrots) == 6
    return ok


# ------------------------------------------------------------------------------------------------- summary
print(f"# {NPASS[0]} PASS, {len(FAILS)} FAIL; runtime {time.time() - T0:.1f}s")
if FAILS:
    print("FAILED:", ", ".join(FAILS))
    sys.exit(1)
sys.exit(0)
