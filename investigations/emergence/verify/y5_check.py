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
