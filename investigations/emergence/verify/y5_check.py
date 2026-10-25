#!/usr/bin/env python3
"""Referee check y5 of sections/e5_closure_two.tex (the closure at two constituents).

    python3 investigations/emergence/verify/y5_check.py

Independent of verify/e5_closure_two.py (nothing is imported from it).  Prints one 'PASS y5-<id>: ...' line per check
and exits 0 only if every check passes.  Where the section states something false, the check below tests the
corrected statement (y5-5f, y5-5g, y5-3c).  numpy, scipy, sympy and mpmath only; runtime well under a minute.

Methods, chosen to differ from the derivation script where possible:
  * exact linear algebra: Gaussian elimination modulo the prime p = 2^31 - 1, rational reconstruction of the null
    vectors, and exact integer verification of every reconstructed vector against the full system.  A matrix with
    nullity k mod p has nullity at most k over Q, and k verified rational null vectors give nullity at least k, so the
    dimensions reported are exact;
  * Lie algebras built from Clifford bivectors e_a e_b and their adjoint action on vectors, not from a spin-lift
    formula (the formula S(M) of the section is checked separately);
  * the stabiliser of M_pair by exact rational ranks at Gaussian-integer points, not by a numerical rank;
  * M_pair also against its definition as an infimum over observers (RE Thm 8.6), by direct minimisation;
  * the drift law by exact series (sympy) and by mpmath quadrature, the rotor law without integration;
  * a different integrator (Radau) and different random data for the dynamics.

Conventions (tex/s02_arena.tex, tex/sA_realisation.tex): eta = diag(1,1,1,-1,-1,-1); on R^8
e1 = sx.1.1, e2 = sz.sx.1, e3 = sz.sz.sx, e4 = ep.1.1, e5 = sz.ep.1, e6 = sz.sz.ep; omega = e1...e6; tower
E_(1,a) = e_a (x) 1, E_(2,a) = omega (x) e_a.  Vector index 6(k-1)+(a-1) for e^(k)_a.  G e_i = e_{i+3},
G e_{i+3} = -e_i, z = x + i y.
"""
import itertools
import sys
import time
from math import gcd

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import ellipe, ellipk

T_START = time.time()
RESULTS = []


def report(cid, ok, text):
    ok = bool(ok)
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {cid}: {text}")


def check(cid, text):
    def deco(fn):
        try:
            ok = fn()
        except Exception as exc:  # a crash is a failure, reported as such
            print(f"   exception in {cid}: {exc!r}")
            ok = False
        report(cid, ok, text)
        return fn
    return deco


# ----------------------------------------------------------------------------------------------------------------
# exact linear algebra modulo p with rational reconstruction
# ----------------------------------------------------------------------------------------------------------------
PR = 2147483647


def _normalise_rows(M):
    """reduce mod p, drop zero rows, scale each row to leading entry 1, drop duplicates."""
    M = np.asarray(M, dtype=np.int64) % PR
    M = M[np.any(M != 0, axis=1)]
    if len(M) == 0:
        return M
    lead = M[np.arange(len(M)), np.argmax(M != 0, axis=1)]
    inv = np.array([pow(int(x), PR - 2, PR) for x in lead], dtype=np.int64)
    M = (M * inv[:, None]) % PR
    return np.unique(M, axis=0)


def rref_modp(M):
    M = _normalise_rows(M).copy()
    if len(M) == 0:
        return M, []
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        i = r + nz[0]
        if i != r:
            M[[r, i]] = M[[i, r]]
        M[r] = (M[r] * pow(int(M[r, c]), PR - 2, PR)) % PR
        f = M[:, c].copy()
        f[r] = 0
        idx = np.nonzero(f)[0]
        if len(idx):
            M[idx] = (M[idx] - (f[idx, None] * M[r][None, :]) % PR) % PR
        piv.append(c)
        r += 1
    return M[:r], piv


def ratrec(a):
    """rational reconstruction of a mod PR: r/s with |r|, s <= sqrt(PR/2)."""
    a = int(a) % PR
    bound = int((PR // 2) ** 0.5)
    r0, r1, s0, s1 = PR, a, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        raise ValueError("rational reconstruction failed")
    if s1 < 0:
        r1, s1 = -r1, -s1
    return r1, s1


def exact_nullspace(M):
    """integer basis of the rational null space of the integer matrix M (rows = equations), exact."""
    M = np.asarray(M, dtype=object)
    ncols = M.shape[1]
    R, piv = rref_modp(np.array(M, dtype=np.int64))
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for f in free:
        num = [0] * ncols
        den = [1] * ncols
        num[f] = 1
        for i, c in enumerate(piv):
            r, s = ratrec((-int(R[i, f])) % PR)
            num[c], den[c] = r, s
        L = 1
        for d in den:
            L = L * d // gcd(L, d)
        v = np.array([num[i] * (L // den[i]) for i in range(ncols)], dtype=object)
        if any(x != 0 for x in M.dot(v)):
            raise ValueError("reconstructed null vector fails exact verification")
        basis.append(v)
    return basis


def exact_rank_int(M):
    """exact rank over Q of an integer matrix: rank mod p plus the verified null space."""
    M = np.asarray(M, dtype=object)
    return M.shape[1] - len(exact_nullspace(M))


def span_dim(vecs):
    """exact dimension of the rational span of integer vectors."""
    if len(vecs) == 0:
        return 0
    A = np.array([np.array(v, dtype=object).ravel() for v in vecs], dtype=object)
    return exact_rank_int(A.T)


def in_span(vecs, w):
    return span_dim(list(vecs) + [w]) == span_dim(vecs)


# ----------------------------------------------------------------------------------------------------------------
# Clifford algebra Cl(3,3) on R^8 and the tower at n = 2
# ----------------------------------------------------------------------------------------------------------------
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def kron(*ms):
    out = np.array([[1]], dtype=np.int64)
    for m in ms:
        out = np.kron(out, m)
    return out


e = [kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx), kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
ETA = np.diag([1, 1, 1, -1, -1, -1]).astype(np.int64)
I8 = np.eye(8, dtype=np.int64)
OMEGA = e[0] @ e[1] @ e[2] @ e[3] @ e[4] @ e[5]
E2T = [np.kron(x, I8) for x in e] + [np.kron(OMEGA, x) for x in e]       # E_(1,a), E_(2,a) on R^64
ETA2 = np.diag([1, 1, 1, -1, -1, -1] * 2).astype(np.int64)
I64 = np.eye(64, dtype=np.int64)


def Gn(n):
    G = np.zeros((6 * n, 6 * n), dtype=np.int64)
    for k in range(n):
        for i in range(3):
            G[6 * k + i + 3, 6 * k + i] = 1
            G[6 * k + i, 6 * k + i + 3] = -1
    return G


def etan(n):
    return np.diag([1, 1, 1, -1, -1, -1] * n).astype(np.int64)


G1, G2 = Gn(1), Gn(2)


def bv(n, a, b):
    """vector action of the Clifford bivector e_a e_b (global 0-based indices a != b) on V_n: ad(e_a e_b)."""
    eta = np.diag(etan(n))
    M = np.zeros((6 * n, 6 * n), dtype=np.int64)
    M[a, b] += 2 * eta[b]
    M[b, a] -= 2 * eta[a]
    return M


def comm(A, B):
    return A @ B - B @ A


def is0(A):
    return not np.any(A)


def in_so(X, eta):
    return is0(X.T @ eta + eta @ X)


def so_basis(n):
    """basis of so(eta_n): the bivector actions, integer."""
    N = 6 * n
    return [bv(n, a, b) for a in range(N) for b in range(a + 1, N)]


def commutant_in(basis, gens, extra=None):
    """integer basis of {X in span(basis): [X, g] = 0 for all g in gens} (and extra(X)-linear conditions), exact."""
    cols = []
    for X in basis:
        rows = [comm(X, g).ravel() for g in gens]
        if extra is not None:
            rows += [r.ravel() for r in extra(X)]
        cols.append(np.concatenate(rows) if rows else np.zeros(1, dtype=np.int64))
    A = np.array(cols, dtype=np.int64).T
    null = exact_nullspace(A)
    out = []
    for v in null:
        X = sum(int(c) * B for c, B in zip(v, basis) if c != 0)
        out.append(np.array(X, dtype=np.int64))
    return out


def lie_closed(mats):
    return all(in_span(mats, comm(A, B).ravel() * 1) for A, B in itertools.combinations(mats, 2)) if mats else True


def flat(ms):
    return [np.array(m, dtype=np.int64).ravel() for m in ms]


def same_span(A, B):
    return span_dim(flat(A)) == span_dim(flat(B)) == span_dim(flat(A) + flat(B))


# vector-level generators -----------------------------------------------------------------------------------------
def idx(k, a):
    """global 0-based index of e^(k)_a, k and a 1-based."""
    return 6 * (k - 1) + (a - 1)


def rot(n, k, i, j):
    """paired rotation of constituent k (temporal 1..3): e_{i'}e_{j'} - e_i e_j, acting on vectors."""
    return bv(n, idx(k, i + 3), idx(k, j + 3)) - bv(n, idx(k, i), idx(k, j))


def bst(n, k, i, j):
    return bv(n, idx(k, j), idx(k, i + 3)) - bv(n, idx(k, i), idx(k, j + 3))


def lint_vec(n, k):
    """interior Lorentz algebra of constituent k on V_n: S_1..3 = (1/2) e5e6, e6e4, e4e5; T_1..3 = (1/2) e1 e_{4,5,6};
    returned doubled (integer): 2S, 2T."""
    S = [bv(n, idx(k, 5), idx(k, 6)), bv(n, idx(k, 6), idx(k, 4)), bv(n, idx(k, 4), idx(k, 5))]
    T = [bv(n, idx(k, 1), idx(k, 4)), bv(n, idx(k, 1), idx(k, 5)), bv(n, idx(k, 1), idx(k, 6))]
    return S, T


def lG_vec(n, k):
    """frame Lorentz algebra of constituent k: 2L_1 = rot(2,3), 2L_2 = rot(3,1), 2L_3 = rot(1,2), 2K likewise with bst."""
    L = [rot(n, k, 2, 3), rot(n, k, 3, 1), rot(n, k, 1, 2)]
    K = [bst(n, k, 2, 3), bst(n, k, 3, 1), bst(n, k, 1, 2)]
    return L, K


def lorentz_ok(L, K, scale=2):
    """[L_i,L_j] = eps L_k, [L_i,K_j] = eps K_k, [K_i,K_j] = -eps L_k for L = L2/scale, K = K2/scale (integer test)."""
    ok = True
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        ok &= is0(comm(L[i], L[j]) - scale * L[k])
        ok &= is0(comm(L[i], K[j]) - scale * K[k])
        ok &= is0(comm(K[i], K[j]) + scale * L[k])
    return ok


# Clifford-level generators on S_2 (the same algebras acting on spinors) -------------------------------------------
def cl(k, a):
    return E2T[idx(k, a)]


def cbv(k, a, l, b):
    return cl(k, a) @ cl(l, b)


def lint_cl(k):
    S = [cbv(k, 5, k, 6), cbv(k, 6, k, 4), cbv(k, 4, k, 5)]
    T = [cbv(k, 1, k, 4), cbv(k, 1, k, 5), cbv(k, 1, k, 6)]
    return S, T            # doubled: S_k = S[k]/2


def rot_cl(k, i, j):
    return cbv(k, i + 3, k, j + 3) - cbv(k, i, k, j)


def bst_cl(k, i, j):
    return cbv(k, j, k, i + 3) - cbv(k, i, k, j + 3)


def lG_cl(k):
    L = [rot_cl(k, 2, 3), rot_cl(k, 3, 1), rot_cl(k, 1, 2)]
    K = [bst_cl(k, 2, 3), bst_cl(k, 3, 1), bst_cl(k, 1, 2)]
    return L, K            # doubled


def spin_lift4(M):
    """4 S(M) = sum_{a,b} (M eta_2)_{ab} E_a E_b on S_2 (integer)."""
    Me = M @ ETA2
    out = np.zeros((64, 64), dtype=np.int64)
    for a in range(12):
        for b in range(12):
            if Me[a, b]:
                out += Me[a, b] * (E2T[a] @ E2T[b])
    return out


# ----------------------------------------------------------------------------------------------------------------
# representation content from Casimirs
# ----------------------------------------------------------------------------------------------------------------
def lorentz_casimirs(L, K, scale=2):
    """C_A, C_B for A = (L - iK)/2, B = (L + iK)/2 with L = L_in/scale etc.: eigenvalues jL(jL+1), jR(jR+1)."""
    L = [x / scale for x in L]
    K = [x / scale for x in K]
    A = [(l - 1j * k) / 2 for l, k in zip(L, K)]
    B = [(l + 1j * k) / 2 for l, k in zip(L, K)]
    return -sum(a @ a for a in A), -sum(b @ b for b in B)


def su2_casimir(L, scale=2):
    L = [x / scale for x in L]
    return -sum(l @ l for l in L)


def jfrom(c):
    j = (-1 + np.sqrt(1 + 4 * max(c, 0))) / 2
    jr = round(2 * j) / 2
    return jr if abs(j - jr) < 1e-7 else None


def content(ops, basis=None, labels=None, tol=1e-7):
    """joint eigenvalues of commuting Hermitian operators, restricted to the orthonormal columns of basis;
    returns {tuple of values: dimension}."""
    if basis is not None:
        ops = [basis.conj().T @ op @ basis for op in ops]
    rng = np.random.default_rng(3)
    coeffs = rng.uniform(1, 2, size=len(ops)) * np.array([1.0, np.sqrt(2), np.pi, np.e, 1.7][:len(ops)])
    H = sum(c * op for c, op in zip(coeffs, ops))
    H = (H + H.conj().T) / 2
    w, U = np.linalg.eigh(H)
    out = {}
    for col in range(U.shape[1]):
        v = U[:, col]
        vals = tuple(round(float(np.real(v.conj() @ op @ v)) * 4) / 4 for op in ops)
        for op, val in zip(ops, vals):
            if np.linalg.norm(op @ v - val * v) > tol * 10:
                raise ValueError("not a joint eigenvector")
        out[vals] = out.get(vals, 0) + 1
    return out


def orth_range(P, tol=1e-9):
    w, U = np.linalg.eigh((P + P.conj().T) / 2)
    return U[:, w > 0.5]


print("# y5 -- referee check of the closure at two constituents (e5); independent of e5_closure_two.py")


# ================================================================================================================
# 1. closure symmetry and the invariants of a pair (Prop e5:prop:so6C, e5:prop:quad, Thm e5:thm:pair)
# ================================================================================================================
SO66 = so_basis(2)
CENT2 = commutant_in(SO66, [G2])                       # Cent(G_2) in so(6,6), exact integer basis
TEMP = [idx(k, a) for k in (1, 2) for a in (1, 2, 3)]  # temporal directions, then their partners
PART = [t + 3 for t in TEMP]
PERM = TEMP + PART


def phi(X):
    """complex 6x6 matrix by which a G_2-commuting X acts on Z = (z1, z2): column b is z(X e_b), e_b temporal."""
    cols = []
    for t in TEMP:
        v = X[:, t]
        cols.append(v[TEMP] + 1j * v[PART])
    return np.array(cols).T


def zvec(v):
    return v[TEMP] + 1j * v[PART]


def vec_from_z(z):
    v = np.zeros(12)
    v[TEMP] = z.real
    v[PART] = z.imag
    return v


PAIRED_ROT = [rot(2, k, i, j) for k in (1, 2) for (i, j) in [(1, 2), (1, 3), (2, 3)]]


def cross_rot(k, i, l, j):
    """paired rotation between temporal direction i of constituent k and j of constituent l."""
    return bv(2, idx(k, i + 3), idx(l, j + 3)) - bv(2, idx(k, i), idx(l, j))


def cross_bst(k, i, l, j):
    return bv(2, idx(l, j), idx(k, i + 3)) - bv(2, idx(k, i), idx(l, j + 3))


ALLROT = PAIRED_ROT + [cross_rot(1, i, 2, j) for i in (1, 2, 3) for j in (1, 2, 3)]
ALLBST = [bst(2, k, i, j) for k in (1, 2) for (i, j) in [(1, 2), (1, 3), (2, 3)]] + \
         [cross_bst(1, i, 2, j) for i in (1, 2, 3) for j in (1, 2, 3)]


@check("y5-1a", "Prop. so6C: G_2^T eta_2 G_2 = -eta_2, G_2^2 = -1; Cent(G_2) in so(6,6) has dimension 30 (exact, modular "
                "elimination with verified rational null vectors) and is spanned by the 15 paired rotations and 15 paired boosts "
                "(within and across the constituents); in the basis temporal|partner every element is [[R,S],[-S,R]], R, S "
                "antisymmetric; X -> R - iS is the action on Z, injective, a Lie homomorphism on 200 random pairs, onto so(6,C) "
                "by dimension; the Euclidean-antisymmetric part is 15-dimensional = the paired rotations")
def _():
    ok = is0(G2.T @ ETA2 @ G2 + ETA2) and is0(G2 @ G2 + np.eye(12, dtype=np.int64))
    ok &= len(CENT2) == 30
    ok &= all(is0(comm(X, G2)) and in_so(X, ETA2) for X in ALLROT + ALLBST)
    ok &= same_span(CENT2, ALLROT + ALLBST) and span_dim(flat(ALLROT + ALLBST)) == 30
    for X in CENT2:
        Y = X[np.ix_(PERM, PERM)]
        R, S, S2, R2 = Y[:6, :6], Y[:6, 6:], Y[6:, :6], Y[6:, 6:]
        ok &= is0(R - R2) and is0(S + S2) and is0(R + R.T) and is0(S + S.T)
        A = phi(X)
        ok &= np.allclose(A, R - 1j * S) and np.allclose(A, -A.T)
    rng = np.random.default_rng(11)
    for _ in range(200):
        a, b = rng.integers(-3, 4, size=(2, 30))
        X = sum(int(c) * B for c, B in zip(a, CENT2))
        Y = sum(int(c) * B for c, B in zip(b, CENT2))
        ok &= np.allclose(phi(comm(X, Y)), comm(phi(X), phi(Y)))
        v = rng.normal(size=12)
        ok &= np.allclose(zvec(X @ v), phi(X) @ zvec(v))
    Phi = np.array([np.concatenate([phi(X).real.ravel(), phi(X).imag.ravel()]) for X in CENT2])
    ok &= np.linalg.matrix_rank(Phi) == 30            # injective; so(6,C) has real dimension 30
    compact = [X for X in CENT2]
    Kc = commutant_in(SO66, [G2], extra=lambda X: [X + X.T])
    ok &= len(Kc) == 15 and same_span(Kc, ALLROT)
    return ok


def real_forms_invariant(gens, n=12):
    """invariant real quadratic forms S (symmetric n x n) with X^T S + S X = 0 for all X in gens; exact basis."""
    basis = []
    for i in range(n):
        for j in range(i, n):
            S = np.zeros((n, n), dtype=np.int64)
            S[i, j] = S[j, i] = 1
            basis.append(S)
    cols = [np.concatenate([(X.T @ S + S @ X).ravel() for X in gens]) for S in basis]
    null = exact_nullspace(np.array(cols, dtype=np.int64).T)
    return [sum(int(c) * B for c, B in zip(v, basis) if c != 0) for v in null]


def holo_form(k, l, part):
    """2 x the real or imaginary part of z_k . z_l as an integer symmetric form on V_2 (k, l in {0,1})."""
    S = np.zeros((12, 12), dtype=np.int64)
    for j in range(3):
        xk, yk = idx(k + 1, j + 1), idx(k + 1, j + 4)
        xl, yl = idx(l + 1, j + 1), idx(l + 1, j + 4)
        # z_k.z_l = sum (x_k + i y_k)(x_l + i y_l): real part x_k x_l - y_k y_l, imaginary x_k y_l + y_k x_l
        if part == 0:
            S[xk, xl] += 1; S[xl, xk] += 1; S[yk, yl] -= 1; S[yl, yk] -= 1
        else:
            S[xk, yl] += 1; S[yl, xk] += 1; S[yk, xl] += 1; S[xl, yk] += 1
    return S


LG1 = [x for pair in zip(*lG_vec(2, 1)) for x in pair]
LG2 = [x for pair in zip(*lG_vec(2, 2)) for x in pair]
LGD = [a + b for a, b in zip(LG1, LG2)]
CV = np.zeros((12, 12), dtype=np.int64)
for a in range(6):
    CV[idx(2, a + 1), idx(1, a + 1)] = 1
    CV[idx(1, a + 1), idx(2, a + 1)] = -1
CBOOST = CV @ G2                                        # eps (x) G


@check("y5-1b", "Prop. quad: invariant real quadratic forms on V_2 have real dimension 2 under Cent(G_2), 4 under "
                "lG^(1)+lG^(2), 6 under the diagonal lG (exact), spanned respectively by Re/Im of q_tot; of q1, q2; of q1, q2, "
                "z1.z2; and the invariant holomorphic quadratics on C^6 (complex symmetric B with A^T B + B A = 0) have complex "
                "dimension 1, 2, 3 (independent computation on the complex side)")
def _():
    ok = True
    q = {(k, l, p): holo_form(k, l, p) for k in (0, 1) for l in (0, 1) for p in (0, 1)}
    qtot = [q[0, 0, p] + q[1, 1, p] for p in (0, 1)]
    cases = [(CENT2, qtot, 2), (LG1 + LG2, [q[0, 0, p] for p in (0, 1)] + [q[1, 1, p] for p in (0, 1)], 4),
             (LGD, [q[k, l, p] for (k, l) in [(0, 0), (1, 1), (0, 1)] for p in (0, 1)], 6)]
    for gens, expect, d in cases:
        F = real_forms_invariant(gens)
        ok &= len(F) == d and same_span(F, expect)
    # complex side: B = Br + i Bi complex symmetric 6 x 6, A^T B + B A = 0 for A = phi(X); work with integer real forms
    for gens, d in [(CENT2, 1), (LG1 + LG2, 2), (LGD, 3)]:
        As = [phi(X) for X in gens]
        basis = []
        for i in range(6):
            for j in range(i, 6):
                for part in (1, 1j):
                    B = np.zeros((6, 6), dtype=complex)
                    B[i, j] = B[j, i] = part
                    basis.append(B)
        cols = []
        for B in basis:
            r = np.concatenate([(A.T @ B + B @ A).ravel() for A in As])
            cols.append(np.round(np.concatenate([r.real, r.imag])).astype(np.int64))
        ok &= len(exact_nullspace(np.array(cols).T)) == 2 * d     # real dimension 2d = complex dimension d
    return ok


def gramQ(z1, z2):
    return np.array([[z1 @ z1, z1 @ z2], [z2 @ z1, z2 @ z2]])


def Mpair(z1, z2):
    s = np.linalg.svd(gramQ(z1, z2), compute_uv=False)
    return 0.5 * (s[0] + s[1])


def stress(z):
    """Maxwell stress-energy T^{mu nu} of the field z = B + iE (Minkowski (+,-,-,-)), as a matrix acting on t_nu."""
    B, E = z.real, z.imag
    u = 0.5 * (E @ E + B @ B)
    S = np.cross(E, B)
    T = np.zeros((4, 4))
    T[0, 0] = u
    T[0, 1:] = T[1:, 0] = S
    T[1:, 1:] = -(np.outer(E, E) + np.outer(B, B)) + u * np.eye(3)
    return T


def Mpair_observers(z1, z2, starts=6, seed=0):
    """inf over unit future timelike t of the Minkowski length of P_sum(t) = sum_k T(z_k) t (RE's definition)."""
    T = stress(z1) + stress(z2)
    g = np.diag([1.0, -1, -1, -1])

    def f(v):
        t = np.concatenate([[np.sqrt(1 + v @ v)], v])
        P = T @ (g @ t)
        return np.sqrt(max(P @ g @ P, 0.0))
    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(starts):
        r = minimize(f, rng.normal(size=3), method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
        best = min(best, r.fun)
    return best


def cplx_rot(w):
    """complex orthogonal matrix exp(w x .) for w in C^3 (so(3,C) acting on C^3)."""
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=complex)
    from scipy.linalg import expm
    return expm(W)


@check("y5-1c", "Thm pair (i): q(z1+z2) = q1+q2+2 z1.z2 and det Q = q1 q2 - (z1.z2)^2 (symbolic); M_pair^2 = (|q1|^2+|q2|^2+"
                "2|z1.z2|^2)/4 + |det Q|/2 = (sigma1+sigma2)^2/4 at 50 random pairs (1e-12), and = (inf over observers of "
                "|sum_k T(z_k) t|)^2 at 4 random pairs by direct minimisation (1e-7; RE Thm 8.6 re-verified from its "
                "definition); a common SO(3,C) transformation keeps Q, M_pair, z1.z2, q(z1+z2); a complex rotation of "
                "constituent 2 alone (a boost and a rotation of lG^(2)) keeps q1, q2 and changes z1.z2, q(z1+z2), M_pair")
def _():
    zs = sp.symbols("a1:4 b1:4")
    z1, z2 = sp.Matrix(zs[:3]), sp.Matrix(zs[3:])
    d = lambda u, v: (u.T * v)[0]
    ok = sp.expand(d(z1 + z2, z1 + z2) - d(z1, z1) - d(z2, z2) - 2 * d(z1, z2)) == 0
    Qs = sp.Matrix([[d(z1, z1), d(z1, z2)], [d(z2, z1), d(z2, z2)]])
    ok &= sp.expand(Qs.det() - (d(z1, z1) * d(z2, z2) - d(z1, z2) ** 2)) == 0
    rng = np.random.default_rng(12)
    for _ in range(50):
        a, b = rng.normal(size=3) + 1j * rng.normal(size=3), rng.normal(size=3) + 1j * rng.normal(size=3)
        q1, q2, r = a @ a, b @ b, a @ b
        M2 = 0.25 * (abs(q1) ** 2 + abs(q2) ** 2 + 2 * abs(r) ** 2) + 0.5 * abs(q1 * q2 - r * r)
        ok &= abs(M2 - Mpair(a, b) ** 2) < 1e-12 * (1 + M2)
    for s in range(4):
        a, b = rng.normal(size=3) + 1j * rng.normal(size=3), rng.normal(size=3) + 1j * rng.normal(size=3)
        ok &= abs(Mpair_observers(a, b, seed=s) - Mpair(a, b)) < 1e-7 * (1 + Mpair(a, b))
    a, b = rng.normal(size=3) + 1j * rng.normal(size=3), rng.normal(size=3) + 1j * rng.normal(size=3)
    Lam = cplx_rot(rng.normal(size=3) + 1j * rng.normal(size=3))
    ok &= np.allclose(Lam.T @ Lam, np.eye(3))
    ok &= np.allclose(gramQ(Lam @ a, Lam @ b), gramQ(a, b)) and abs(Mpair(Lam @ a, Lam @ b) - Mpair(a, b)) < 1e-10
    for w in (1j * np.array([0.3, -0.2, 0.5]), np.array([0.4, 0.1, -0.3])):     # boost, rotation of constituent 2
        Lb = cplx_rot(w)
        b2 = Lb @ b
        ok &= abs(b2 @ b2 - b @ b) < 1e-10 and abs(a @ b2 - a @ b) > 1e-3
        ok &= abs((a + b2) @ (a + b2) - (a + b) @ (a + b)) > 1e-3 and abs(Mpair(a, b2) - Mpair(a, b)) > 1e-4
    return ok


# exact stabiliser of M_pair -------------------------------------------------------------------------------------------
def gauss_points(count, seed):
    """Gaussian-integer pairs (z1, z2) with det Q real and nonzero, so that |det Q| is rational."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < count:
        a = [int(x) + int(y) * sp.I for x, y in rng.integers(-3, 4, size=(3, 2))]
        b = [int(x) + int(y) * sp.I for x, y in rng.integers(-3, 4, size=(3, 2))]
        za, zb = sp.Matrix(a), sp.Matrix(b)
        D = sp.expand((za.T * za)[0] * (zb.T * zb)[0] - ((za.T * zb)[0]) ** 2)
        if sp.im(D) == 0 and D != 0:
            pts.append((za, zb, D))
    return pts


def dM2_exact(za, zb, D, A):
    """exact derivative of M_pair^2 = ||Q||_F^2/4 + |det Q|/2 along dZ = A Z (A Gaussian-integer 6x6), det Q = D real."""
    Z = sp.Matrix.vstack(za, zb)
    dZ = A * Z
    da, db = dZ[:3, 0], dZ[3:, 0]
    d = lambda u, v: sp.expand((u.T * v)[0])
    Q = sp.Matrix([[d(za, za), d(za, zb)], [d(zb, za), d(zb, zb)]])
    dQ = sp.Matrix([[2 * d(da, za), d(da, zb) + d(za, db)], [d(db, za) + d(zb, da), 2 * d(db, zb)]])
    t1 = sum(dQ[i, j] * sp.conjugate(Q[i, j]) for i in range(2) for j in range(2))
    ddet = Q[1, 1] * dQ[0, 0] - Q[0, 1] * dQ[1, 0] - Q[1, 0] * dQ[0, 1] + Q[0, 0] * dQ[1, 1]
    return sp.nsimplify(sp.re(sp.expand(t1)) / 2 + sp.sign(D) * sp.re(sp.expand(ddet)) / 2)


def phi_exact(X):
    A = phi(X)
    return sp.Matrix(6, 6, lambda i, j: int(round(A[i, j].real)) + int(round(A[i, j].imag)) * sp.I)


STAB7 = LGD + [CV]


@check("y5-1d", "Thm pair (ii), exact: at 30 Gaussian-integer pairs with real det Q != 0 the derivatives of M_pair^2 along the "
                "30 generators of Cent(G_2) form a rational 30 x 30 matrix of exact rank 23, so the subalgebra of so(6,C) "
                "leaving M_pair invariant has dimension <= 7; the 7 elements of lG^Delta + R C_v give exact zero derivatives "
                "there, so the stabiliser is exactly lG^Delta + R C_v, real dimension 7")
def _():
    pts = gauss_points(30, 21)
    As = [phi_exact(X) for X in CENT2]
    rows = []
    for za, zb, D in pts:
        rows.append([dM2_exact(za, zb, D, A) for A in As])
    Mx = sp.Matrix(rows)
    den = sp.ilcm(*[sp.fraction(x)[1] for x in Mx])
    rk = exact_rank_int(np.array((Mx * den).tolist(), dtype=object).astype(np.int64))
    ok = rk == 23
    ok &= span_dim(flat(STAB7)) == 7 and all(in_span(flat(CENT2), x.ravel()) for x in flat(STAB7))
    for za, zb, D in pts[:6]:
        ok &= all(dM2_exact(za, zb, D, phi_exact(X)) == 0 for X in STAB7)
    return ok


@check("y5-1e", "Thm pair (iii): the copy boost eps(x)G lies in Cent(G_2) cap so(6,6), squares to +1, acts on Z as "
                "[[0,-i],[i,0]], has eigenvector (e1, i e1) (eigenvalue 1, a vector of V_+) and e^{t eps(x)G} = cosh t + "
                "sinh t eps(x)G (exact); M_pair(e^t(e1,ie1)) = e^{2t}, q_tot = 0 (7 values of t); R^(1)_phi of RE Ex. 8.8 "
                "is generated by an element of Cent(G_2) cap so(12) and takes ((1,i,0),0) to M_pair = sin(phi), 0 <= phi <= pi")
def _():
    ok = is0(comm(CBOOST, G2)) and in_so(CBOOST, ETA2) and is0(CBOOST @ CBOOST - np.eye(12, dtype=np.int64))
    ok &= np.allclose(phi(CBOOST), np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(3)))
    v = vec_from_z(np.array([1, 0, 0, 1j, 0, 0]))
    ok &= np.allclose(CBOOST @ v, v) and np.allclose(v[6:], G1 @ v[:6])       # V_+ = {X^(2) = G X^(1)}
    for t in (-1.0, -0.5, -0.1, 0.0, 0.3, 0.8, 1.5):
        Z = np.exp(t) * np.array([1, 0, 0, 1j, 0, 0])
        ok &= abs(Mpair(Z[:3], Z[3:]) - np.exp(2 * t)) < 1e-12 * np.exp(2 * t) and abs(Z @ Z) < 1e-12
    gen = cross_rot(1, 1, 2, 1) // 2                      # rotation of (e^(1)_1, e^(2)_1) and (e^(1)_4, e^(2)_4)
    ok &= is0(comm(gen, G2)) and is0(gen + gen.T) and in_so(gen, ETA2)
    from scipy.linalg import expm
    for ph in np.linspace(0, np.pi, 7):
        R = expm(ph * gen.astype(float))
        Z = phi(R) @ np.array([1, 1j, 0, 0, 0, 0]) if False else zvec(R @ vec_from_z(np.array([1, 1j, 0, 0, 0, 0])))
        ok &= abs(Mpair(Z[:3], Z[3:]) - np.sin(ph)) < 1e-12
    return ok


@check("y5-1f", "Thm pair (iv): at random Z with q_tot = c != 0 and at random nonzero null Z the tangent space of the "
                "Cent(G_2)-orbit has real dimension 10 = dim {q_tot = c} (open orbits of a connected group on a connected "
                "level set: one orbit, Witt); M_pair >= |q_tot|/2 at 200 random pairs, with equality at (z,0); the copy boost "
                "takes (z,0) to (cosh t z, i sinh t z) with q_tot = q(z) and M_pair = |q(z)| cosh(2t)/2")
def _():
    ok = True
    rng = np.random.default_rng(13)
    As = [phi(X) for X in CENT2]
    for kind in ("generic", "null"):
        for _ in range(5):
            if kind == "generic":
                Z = rng.normal(size=6) + 1j * rng.normal(size=6)
            else:
                u = rng.normal(size=6)
                v = rng.normal(size=6)
                v -= (v @ u) / (u @ u) * u
                v *= np.linalg.norm(u) / np.linalg.norm(v)
                Z = u + 1j * v
            T = np.array([np.concatenate([(A @ Z).real, (A @ Z).imag]) for A in As])
            ok &= np.linalg.matrix_rank(T, tol=1e-9) == 10
    for _ in range(200):
        Z = rng.normal(size=6) + 1j * rng.normal(size=6)
        ok &= Mpair(Z[:3], Z[3:]) >= 0.5 * abs(Z @ Z) - 1e-12
    for _ in range(5):
        z = rng.normal(size=3) + 1j * rng.normal(size=3)
        ok &= abs(Mpair(z, 0 * z) - 0.5 * abs(z @ z)) < 1e-12
        for t in (0.2, 0.7, 1.3):
            Bt = np.cosh(t) * np.eye(12) + np.sinh(t) * CBOOST
            Z = zvec(Bt @ vec_from_z(np.concatenate([z, 0 * z])))
            ok &= np.allclose(Z, np.concatenate([np.cosh(t) * z, 1j * np.sinh(t) * z]))
            ok &= abs(Z @ Z - z @ z) < 1e-10 and abs(Mpair(Z[:3], Z[3:]) - 0.5 * abs(z @ z) * np.cosh(2 * t)) < 1e-10
    return ok


# ================================================================================================================
# 2. no common Lorentz algebra (Prop e5:prop:nocommon)
# ================================================================================================================
def lint_list(n, k):
    S, T = lint_vec(n, k)
    return S + T


def lG_list(n, k):
    L, K = lG_vec(n, k)
    return L + K


def intersect_dim(A, B):
    return span_dim(flat(A)) + span_dim(flat(B)) - span_dim(flat(A) + flat(B))


def supported_on_tail(X):
    """X vanishes on constituent 1 and has no component into it."""
    return is0(X[:, :6]) and is0(X[:6, :])


@check("y5-2a", "Prop. nocommon (i): Cent(G_n) meets lint^(1) in 0 and the elements of so(3n,3n) commuting with G_n and lint^(1) "
                "have dimension 0, 6, 30, 72 at n = 1, 2, 3, 4 = 3(n-1)(3n-4); at n = 2 they are exactly lG^(2), at n = 3, 4 "
                "they vanish on constituent 1 and span the centraliser of G_{n-1} on constituents 2..n (so(3n-3,C)); n = 4 by "
                "elimination mod p with 72 verified rational solutions")
def _():
    ok = True
    for n, expect in [(1, 0), (2, 6), (3, 30), (4, 72)]:
        G = Gn(n)
        li = lint_list(n, 1)
        basis = so_basis(n)
        cent = commutant_in(basis, [G])
        ok &= intersect_dim(cent, li) == 0
        both = commutant_in(basis, [G] + li)
        ok &= len(both) == expect == 3 * (n - 1) * (3 * n - 4)
        if n == 2:
            ok &= same_span(both, LG2)
        if n >= 3:
            ok &= all(is0(X[:, :6]) and is0(X[:6, :]) for X in both)
            tail = [X[6:, 6:] for X in both]
            ok &= len(commutant_in(so_basis(n - 1), [Gn(n - 1)])) == len(tail) and \
                same_span(tail, commutant_in(so_basis(n - 1), [Gn(n - 1)]))
    return ok


@check("y5-2b", "Prop. nocommon (ii): the bivectors of constituent 1 on S_2 are e_a e_b (x) 1 and those of constituent 2 are "
                "1 (x) e_a e_b (exact); at one constituent S (x) C = 2(1/2,1/2) under lG and 2(1/2,0)+2(0,1/2) under lint "
                "(Casimirs); on S_2, computed directly: 16(1/2,1/2) under lG^(1) and 16(1/2,0)+16(0,1/2) under lint^(1)")
def _():
    ok = True
    for a in range(6):
        for b in range(6):
            if a != b:
                ok &= np.array_equal(cbv(1, a + 1, 1, b + 1), np.kron(e[a] @ e[b], I8))
                ok &= np.array_equal(cbv(2, a + 1, 2, b + 1), np.kron(I8, e[a] @ e[b]))
    # one constituent, on R^8
    def one(k):
        c = [e[k[0]] @ e[k[1]]]
        return c[0]
    L8 = [e[4] @ e[5] - e[1] @ e[2], e[5] @ e[3] - e[2] @ e[0], e[3] @ e[4] - e[0] @ e[1]]
    K8 = [e[2] @ e[4] - e[1] @ e[5], e[0] @ e[5] - e[2] @ e[3], e[1] @ e[3] - e[0] @ e[4]]
    S8 = [e[4] @ e[5], e[5] @ e[3], e[3] @ e[4]]
    T8 = [e[0] @ e[3], e[0] @ e[4], e[0] @ e[5]]
    ok &= lorentz_ok(L8, K8) and lorentz_ok(S8, T8)
    ok &= content(list(lorentz_casimirs(L8, K8))) == {(0.75, 0.75): 8}
    ok &= content(list(lorentz_casimirs(S8, T8))) == {(0.75, 0.0): 4, (0.0, 0.75): 4}
    L, K = lG_cl(1)
    ok &= content(list(lorentz_casimirs(L, K))) == {(0.75, 0.75): 64}
    S, T = lint_cl(1)
    ok &= content(list(lorentz_casimirs(S, T))) == {(0.75, 0.0): 32, (0.0, 0.75): 32}
    return ok


@check("y5-2c", "Prop. nocommon (iii): S_2 (x) C = 16(1/2,1/2) under lG^(2), and under the diagonal lG (the sum of the two "
                "spin lifts) = 4[(0,0)+(1,0)+(0,1)+(1,1)]: dimensions 4, 12, 12, 36; integer spin only")
def _():
    L2, K2 = lG_cl(2)
    ok = content(list(lorentz_casimirs(L2, K2))) == {(0.75, 0.75): 64}
    L1, K1 = lG_cl(1)
    LD = [a + b for a, b in zip(L1, L2)]
    KD = [a + b for a, b in zip(K1, K2)]
    ok &= lorentz_ok(LD, KD)
    ok &= content(list(lorentz_casimirs(LD, KD))) == {(0.0, 0.0): 4, (2.0, 0.0): 12, (0.0, 2.0): 12, (2.0, 2.0): 36}
    return ok


# ================================================================================================================
# 3. gauge algebras at two constituents and the binding root (Props noPS, spectator, diagonal, mixed)
# ================================================================================================================
LINT1, LINT2 = lint_list(2, 1), lint_list(2, 2)
LINTD = [a + b for a, b in zip(LINT1, LINT2)]
LMIX = [a + b for a, b in zip(LINT1, LG2)]             # S^(1)_k + L^(2)_k, T^(1)_k + K^(2)_k (doubled)
ACTIONS = {"spectator (1,0,1)": LINT1, "binding+spectator (0,1,1)": LG1, "diagonal interior (2,0,0)": LINTD,
           "mixed (1,1,0)": LMIX, "binding on both (0,2,0)": LGD}


def antisym_extra(X):
    return [X + X.T]


@check("y5-3a", "Prop. noPS: the grade-two commutants g(k_I,k_B,k_S) of the five Lorentz actions at n = 2 have dimensions "
                "28, 15, 7, 1, 2, are closed under transposition (so their compact parts are their Euclidean-antisymmetric "
                "parts) and have compact parts of dimensions 13, 6, 7, 1, 1 (exact); all are below 15 = dim su(4) < 21 = "
                "dim(so4+so6), so none contains the Pati-Salam algebra or its colour factor (dimension count)")
def _():
    ok = lorentz_ok(LINT1[:3], LINT1[3:]) and lorentz_ok(LMIX[:3], LMIX[3:]) and lorentz_ok(LINTD[:3], LINTD[3:])
    expect = [(28, 13), (15, 6), (7, 7), (1, 1), (2, 1)]
    for (name, gens), (d, dc) in zip(ACTIONS.items(), expect):
        g = commutant_in(SO66, gens)
        gc = commutant_in(SO66, gens, extra=antisym_extra)
        ok &= len(g) == d and len(gc) == dc
        ok &= all(in_span(flat(g), X.T.ravel()) for X in g)
        ok &= dc < 15
    return ok


P_IDX = [idx(1, 2), idx(1, 3), idx(2, 1), idx(2, 2), idx(2, 3)]
N_IDX = [idx(2, 4), idx(2, 5), idx(2, 6)]
E2_IDX = [idx(2, 1), idx(2, 2), idx(2, 3)]
SO_P = [bv(2, a, b) for a, b in itertools.combinations(P_IDX, 2)]
SO_N = [bv(2, a, b) for a, b in itertools.combinations(N_IDX, 2)]
ROT2 = [rot(2, 2, 2, 3), rot(2, 2, 3, 1), rot(2, 2, 1, 2)]       # 2 L^(2)_k, the paired rotations of constituent 2
BST2 = [bst(2, 2, 2, 3), bst(2, 2, 3, 1), bst(2, 2, 1, 2)]


@check("y5-3b", "Prop. spectator (i): g(1,0,1) = Cent(lint^(1)) in so(6,6) has dimension 28, compact part so(P) + so(N) "
                "(10 + 3) on P = <e^(1)_2,e^(1)_3,e^(2)_1,e^(2)_2,e^(2)_3>, N = <e^(2)_4,e^(2)_5,e^(2)_6>; g(1,0,1) cap Cent(G_2) "
                "= lG^(2); so(P) and so(N) each meet Cent(G_2) in 0, and so(P)+so(N) meets it in {A + G_2 A G_2^-1: A in "
                "so(E_2)}, the three paired rotations of constituent 2, with su(2) brackets")
def _():
    g = commutant_in(SO66, LINT1)
    gc = commutant_in(SO66, LINT1, extra=antisym_extra)
    ok = len(g) == 28 and same_span(gc, SO_P + SO_N)
    ok &= same_span(commutant_in(SO66, LINT1 + [G2]), LG2)
    cent = CENT2
    ok &= intersect_dim(SO_P, cent) == 0 and intersect_dim(SO_N, cent) == 0
    inter = commutant_in(SO66, LINT1 + [G2], extra=antisym_extra)
    ok &= len(inter) == 3 and same_span(inter, ROT2)
    G2inv = -G2
    graph = [A + G2 @ A @ G2inv for A in [bv(2, a, b) for a, b in itertools.combinations(E2_IDX, 2)]]
    ok &= same_span(graph, ROT2) and all(is0(comm(X, G2)) for X in graph)
    ok &= lorentz_ok(ROT2, BST2)                         # in particular [L_i, L_j] = eps_ijk L_k
    return ok


GAM1 = [cl(1, 1), cl(1, 4), cl(1, 5), cl(1, 6)]
KER1 = [-(GAM1[0] @ g) for g in GAM1]


def kinetic_ok(X, kernels):
    return all(is0(X.T @ Km + Km @ X) for Km in kernels)


@check("y5-3c", "Prop. spectator (ii): among the spin lifts of Cent(G_2) the solutions of X^T K^mu + K^mu X = 0 (K^mu = "
                "-gamma^0 gamma^mu of constituent 1, mu = 0..3) form exactly the span of the lifts of the three paired "
                "rotations of constituent 2 (exact null space of a 16384 x 30 integer system); each paired boost of "
                "constituent 2 fails; the spin lift used is that of the section, checked against the bivectors")
def _():
    lifts = [spin_lift4(X) for X in CENT2]
    cols = [np.concatenate([(S.T @ Km + Km @ S).ravel() for Km in KER1]) for S in lifts]
    null = exact_nullspace(np.array(cols, dtype=np.int64).T)
    sols = [sum(int(c) * S for c, S in zip(v, lifts) if c != 0) for v in null]
    ok = len(sols) == 3 and same_span(sols, [spin_lift4(X) for X in ROT2])
    ok &= all(np.array_equal(spin_lift4(X), 2 * rot_cl(2, *ij)) for X, ij in zip(ROT2, [(2, 3), (3, 1), (1, 2)]))
    ok &= all(not kinetic_ok(spin_lift4(X), KER1) for X in BST2)
    return ok


W8 = cl(1, 2) @ cl(1, 3)
for a in range(1, 7):
    W8 = W8 @ cl(2, a)
OMEGA1 = cl(1, 1) @ cl(1, 4) @ cl(1, 5) @ cl(1, 6)
GAMMA_SP = W8 @ OMEGA1
PL_SP, PR_SP = (I64 - GAMMA_SP), (I64 + GAMMA_SP)          # doubled projectors 2 P_L, 2 P_R


def sym3(hs, P2):
    """symmetrised integer traces Tr(h_(a h_b h_c) P) of integer matrices hs and doubled projector P2."""
    vals = []
    for a, b, c in itertools.combinations_with_replacement(range(len(hs)), 3):
        t = 0
        for p in set(itertools.permutations((a, b, c))):
            t += int(np.trace(hs[p[0]] @ hs[p[1]] @ hs[p[2]] @ P2))
        vals.append(t)
    return vals


@check("y5-3d", "Prop. spectator (iii): w = product of the 8 Lorentz-trivial generators has w^2 = -1 and commutes with "
                "lint^(1) and with the spin lifts of all 28 elements of so(5,3); Gamma = w Omega^(1), Omega^(1) = "
                "E_(1,1)E_(1,4)E_(1,5)E_(1,6), is an involution commuting with w and with both; with w as i each chirality "
                "half is C^16 = (1/2,0)(x)(4,2), respectively (0,1/2)(x)(4,2) (so(5), so(3) Casimirs 5/2, 3/4); under the "
                "binding-compatible su(2) each half is (Weyl)(x)(2.1 + 2.3): su(2) Casimir 0 on 4 and 2 on 12 dimensions, no "
                "isospin 1/2, the same content on both halves")
def _():
    ok = is0(W8 @ W8 + I64)
    S1, T1 = lint_cl(1)
    g101 = commutant_in(SO66, LINT1)
    lifts = [spin_lift4(X) for X in g101]
    ok &= all(is0(comm(W8, X)) for X in S1 + T1 + lifts)
    ok &= is0(GAMMA_SP @ GAMMA_SP - I64) and is0(comm(GAMMA_SP, W8))
    ok &= all(is0(comm(GAMMA_SP, X)) for X in S1 + T1 + lifts)
    Uw = orth_range((I64 - 1j * W8) / 2)
    ok &= Uw.shape[1] == 32
    CA, CB = lorentz_casimirs(S1, T1)
    soP = [spin_lift4(X) for X in SO_P]                  # 4 x (1/2) e_a e_b ... spin_lift4(bv) = 4 e_a e_b
    soN = [spin_lift4(X) for X in SO_N]
    CP = -sum((x / 8) @ (x / 8) for x in soP)            # (1/2) e_a e_b = spin_lift4(bv)/8
    CN = -sum((x / 8) @ (x / 8) for x in soN)
    Csu = su2_casimir([spin_lift4(X) / 2 for X in ROT2])  # spin_lift4(rot) = 2 rot_cl; L = rot_cl / 2
    halves = []
    for sgn in (-1, 1):
        Ph = Uw.conj().T @ ((I64 + sgn * GAMMA_SP) / 2) @ Uw
        Uh = Uw @ orth_range(Ph)
        ok &= Uh.shape[1] == 16
        halves.append(content([CA, CB, CP, CN, Csu], basis=Uh))
    ok &= halves[0] == {(0.75, 0.0, 2.5, 0.75, 0.0): 4, (0.75, 0.0, 2.5, 0.75, 2.0): 12} and \
        halves[1] == {(0.0, 0.75, 2.5, 0.75, 0.0): 4, (0.0, 0.75, 2.5, 0.75, 2.0): 12}
    return ok


@check("y5-3e", "Prop. spectator (iii), anomaly traces, CORRECTED REASON: with h = -w S(X) the symmetrised traces Tr(h P_L), "
                "Tr(h^3 P_L) and the same for P_R vanish exactly for the binding-compatible su(2) (integer isospin) and equally "
                "for so(N) = su(2) of the spectator algebra, which acts on the Weyl block by isospin doublets (so(N) Casimir "
                "3/4 on all of S_2): the vanishing of odd traces holds for every su(2) action (weights symmetric under m -> -m); "
                "what integer isospin adds is the absence of doublets, hence of the global SU(2) anomaly")
def _():
    ok = True
    for gens in (ROT2, SO_N):
        lifts = [spin_lift4(X) for X in gens]           # 4 S(X) for SO_N bivectors; 2 x (2 L) for ROT2
        hs = [-(W8 @ S) for S in lifts]                  # integer multiples of the charges h = -w S(X)
        for P2 in (PL_SP, PR_SP):
            ok &= all(int(np.trace(h @ P2)) == 0 for h in hs)
            ok &= all(v == 0 for v in sym3(hs, P2))
    CN = -sum((spin_lift4(X) / 8) @ (spin_lift4(X) / 8) for X in SO_N)
    ok &= np.allclose(CN, 0.75 * np.eye(64))
    return ok


CB2 = sum(ETA2[idx(1, a), idx(1, a)] * (cl(1, a) @ cl(2, a)) for a in range(1, 7))     # 2 C_b
GAMMA12 = I64.copy()
for X in E2T:
    GAMMA12 = GAMMA12 @ X


@check("y5-3f", "Prop. diagonal (i): Cent(diagonal lint) in so(6,6) has dimension 7, all of it Euclidean-antisymmetric "
                "(compact); it meets Cent(G_2) in R C_v, C_v = eps (x) 1_6, which is the copy rotation of the interior four-planes "
                "plus that of the surplus pairs, neither of which commutes with G_2 alone; S(C_v) = -C_b with C_b = (1/2) sum_a "
                "eta_aa E_(1,a)E_(2,a) (exact)")
def _():
    g = commutant_in(SO66, LINTD)
    ok = len(g) == 7 and all(is0(X + X.T) for X in g)
    inter = commutant_in(SO66, LINTD + [G2])
    ok &= len(inter) == 1 and same_span(inter, [CV])
    four = [idx(k, a) for k in (1, 2) for a in (1, 4, 5, 6)]
    Pm = np.zeros((12, 12), dtype=np.int64)
    Pm[four, four] = 1
    M12, Cs = CV @ Pm, CV @ (np.eye(12, dtype=np.int64) - Pm)
    ok &= all(is0(comm(X, Y)) for X in (M12, Cs) for Y in LINTD)
    ok &= not is0(comm(M12, G2)) and not is0(comm(Cs, G2))
    ok &= np.array_equal(spin_lift4(CV), -2 * CB2)
    return ok


@check("y5-3g", "Prop. diagonal (ii): C_b satisfies the kinetic condition with the summed kernels K^mu_S = sum_k "
                "gamma^(k)_0 gamma^(k)_mu (mu = 0..3), and fails it with the kernels f_0 f_mu of f_a = (E_(1,a)+E_(2,a))/sqrt2 "
                "for each mu = 1, 2, 3 (it holds at mu = 0, where f_0 f_0 = 1) (exact)")
def _():
    gam = {k: [cl(k, 1), cl(k, 4), cl(k, 5), cl(k, 6)] for k in (1, 2)}
    KS = [gam[1][0] @ gam[1][m] + gam[2][0] @ gam[2][m] for m in range(4)]
    ok = kinetic_ok(CB2, KS)
    f2 = [gam[1][m] + gam[2][m] for m in range(4)]       # sqrt2 f_mu
    Kf = [f2[0] @ f2[m] for m in range(4)]               # 2 f_0 f_mu
    ok &= kinetic_ok(CB2, [Kf[0]]) and all(not kinetic_ok(CB2, [Kf[m]]) for m in (1, 2, 3))
    return ok


@check("y5-3h", "Prop. diagonal (iii): Gamma_12^2 = 1, Gamma_12 commutes with the diagonal lint and C_b; the joint content "
                "(diagonal-lint Casimirs, copy charge q = -i C_b, Gamma_12) of S_2 (x) C is exactly the table of the section: "
                "Gamma_12 = -1: (0,0)_{+-3} + (0,0)_{+-1} + 2(1/2,1/2)_{+-1} + (1,0)_{+-1} + (0,1)_{+-1}; Gamma_12 = +1: "
                "2(0,0)_{+-2} + (1/2,1/2)_{+-2} + 2(1/2,1/2)_0 + 2(1,0)_0 + 2(0,1)_0; Tr(C_b^k P_+-) = 0 for k = 1, 3, 5 (exact)")
def _():
    ok = is0(GAMMA12 @ GAMMA12 - I64) and is0(comm(GAMMA12, CB2))
    S1, T1 = lint_cl(1)
    S2, T2 = lint_cl(2)
    SD, TD = [a + b for a, b in zip(S1, S2)], [a + b for a, b in zip(T1, T2)]
    ok &= all(is0(comm(GAMMA12, X)) for X in SD + TD) and all(is0(comm(CB2, X)) for X in SD + TD)
    CA, CB = lorentz_casimirs(SD, TD)
    q = -1j * CB2 / 2
    got = content([CA, CB, q, GAMMA12.astype(float)])
    want = {}

    def add(jl, jr, qq, g, mult):
        key = (jl * (jl + 1), jr * (jr + 1), float(qq), float(g))
        want[key] = want.get(key, 0) + mult * int((2 * jl + 1) * (2 * jr + 1))
    for s in (1, -1):
        add(0, 0, 3 * s, -1, 1); add(0, 0, s, -1, 1); add(.5, .5, s, -1, 2); add(1, 0, s, -1, 1); add(0, 1, s, -1, 1)
        add(0, 0, 2 * s, 1, 2); add(.5, .5, 2 * s, 1, 1)
    add(.5, .5, 0, 1, 2); add(1, 0, 0, 1, 2); add(0, 1, 0, 1, 2)
    ok &= got == want
    for P2 in (I64 - GAMMA12, I64 + GAMMA12):
        for k in (1, 3, 5):
            ok &= int(np.trace(np.linalg.matrix_power(CB2, k) @ P2)) == 0
    return ok


J1V = bv(2, idx(1, 2), idx(1, 3))
GV2 = np.zeros((12, 12), dtype=np.int64)
GV2[6:, 6:] = G1


@check("y5-3i", "Prop. mixed: l_mix (lint^(1) paired with lG^(2)) has the so(1,3) brackets; V_2 = (1/2,1/2) + (1,0)+(0,1) + "
                "2(0,0) under it; its centraliser in so(6,6) is R J_1 (J_1 the rotation of e^(1)_2, e^(1)_3), which meets "
                "Cent(G_2) in 0; [G_2, l_mix] != 0; the l_mix-equivariant endomorphisms of V_2 form a 7-dimensional space "
                "whose Euclidean-antisymmetric part is span{G^(2) = 0+G, J_1}, and [J_1, G^(2)] = 0 (exact)")
def _():
    ok = lorentz_ok(LMIX[:3], LMIX[3:])
    CA, CB = lorentz_casimirs(LMIX[:3], LMIX[3:])
    ok &= content([CA, CB]) == {(0.75, 0.75): 4, (2.0, 0.0): 3, (0.0, 2.0): 3, (0.0, 0.0): 2}
    c = commutant_in(SO66, LMIX)
    ok &= len(c) == 1 and same_span(c, [J1V]) and intersect_dim(c, CENT2) == 0
    ok &= any(not is0(comm(G2, X)) for X in LMIX)
    units = []
    for i in range(12):
        for j in range(12):
            U = np.zeros((12, 12), dtype=np.int64)
            U[i, j] = 1
            units.append(U)
    endo = commutant_in(units, LMIX)
    ok &= len(endo) == 7
    endo_a = commutant_in(units, LMIX, extra=antisym_extra)
    ok &= len(endo_a) == 2 and same_span(endo_a, [GV2, J1V]) and is0(comm(J1V, GV2))
    return ok


@check("y5-3j", "binding actions: the commutant of lG^(1) in so(6,6) is the so(3,3) of constituent 2 (dimension 15, supported "
                "on constituent 2, compact part 6) and meets Cent(G_2) in lG^(2), with compact part the three paired rotations "
                "of constituent 2; the commutant of the diagonal lG is span{C_v, eps(x)G} (dimension 2), inside Cent(G_2), "
                "with compact part R C_v (exact)")
def _():
    g = commutant_in(SO66, LG1)
    ok = len(g) == 15 and all(is0(X[:, :6]) and is0(X[:6, :]) for X in g)
    ok &= len(commutant_in(SO66, LG1, extra=antisym_extra)) == 6
    ok &= same_span(commutant_in(SO66, LG1 + [G2]), LG2)
    ok &= same_span(commutant_in(SO66, LG1 + [G2], extra=antisym_extra), ROT2)
    d = commutant_in(SO66, LGD)
    ok &= len(d) == 2 and same_span(d, [CV, CBOOST]) and all(is0(comm(X, G2)) for X in d)
    ok &= same_span(commutant_in(SO66, LGD, extra=antisym_extra), [CV])
    return ok
