#!/usr/bin/env python3
"""Track e4 -- The b-field up the ladder.

Verification script for investigations/emergence/sections/e4_bfield_ladder.tex.

    python3 e4_bfield_ladder.py        (from any directory)

Prints one 'PASS e4-<id>: <claim>' (or FAIL) line per claim and exits 0 only if every check passes.

Conventions (tex/s02_arena.tex, tex/sA_realisation.tex; RE sections d5-d7; emergence sections e1, e2):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1); e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1,
  e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep; omega = e1...e6; tower E_(k,a) = omega^{(x)(k-1)} (x) e_a (x) 1^{(x)(n-k)}.
  Vector index 6k + a (copy-major, k = 0..n-1, a = 0..5).  G e_i = e_{i+3}, G e_{i+3} = -e_i.
  Fock modes p = (k, i), i = 0, 1, 2:  a+_p = (E_(k,i) + E_(k,i+3))/2,  a-_p = (E_(k,i) - E_(k,i+3))/2,  N = sum a+_p a-_p.
  z_p = X_(k,i) + i X_(k,i+3) (z = B + iE), index p = 3k + i.
  Spin lift of A in so(eta_n): sigma(A) = 1/4 sum_ab A_ab eta_b E_a E_b; lift4(A) = 4 sigma(A) is an integer matrix.
  Copy rotation C^{kl} = (E_lk - E_kl) (x) 1_6, spin lift -(1/2) sum_a eta_a E_(k,a) E_(l,a); copy boost (E_lk - E_kl) (x) G.
  Copy field b = B (x) 1_6, B real antisymmetric; binding K (x) G, K = diag(kappa); on z: S = iH, H = K~ - iR, R = B (x) 1_3.
  Frame brackets: beta_0(x, y) = z(x) x z(y) (complex cross product), as in RE d7 (d7-2b).

Exactness: every algebraic identity is checked in integer or rational arithmetic (scipy.sparse int64 matrices, which are
signed permutations or integer combinations of them; sympy for symbolic and rational linear algebra).  Floating point is
used only for spectra (numpy.linalg.eigh/eigvals, tolerance 1e-8), for numerical ranks (SVD, tolerance 1e-9 relative)
and for random-sample checks of polynomial identities (tolerance 1e-10), as stated in each PASS line.
Runtime: well under three minutes.
"""
import itertools
import sys
import time
from collections import Counter
from fractions import Fraction
from math import comb

import numpy as np
import scipy.sparse as sps
import sympy as sp

T0 = time.time()
FAILED = []


def report(cid, ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {cid}: {text}")
    if not ok:
        FAILED.append(cid)


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


# ================================================================================================ realisation and tower
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def k3(a, b, c):
    return np.kron(np.kron(a, b), c)


E8 = [k3(sx, I2, I2), k3(sz, sx, I2), k3(sz, sz, sx), k3(ep, I2, I2), k3(sz, ep, I2), k3(sz, sz, ep)]
ETA = [1, 1, 1, -1, -1, -1]
OM = E8[0] @ E8[1] @ E8[2] @ E8[3] @ E8[4] @ E8[5]


def tower(n):
    """sparse integer generators E_(k,a), index 6k + a."""
    OMs, Is = sps.csr_matrix(OM), sps.identity(8, dtype=np.int64, format="csr")
    es = [sps.csr_matrix(x) for x in E8]
    gens = []
    for k in range(n):
        for a in range(6):
            M = sps.identity(1, dtype=np.int64, format="csr")
            for j in range(n):
                M = sps.kron(M, OMs if j < k else (es[a] if j == k else Is), format="csr")
            gens.append(M)
    return gens


def iszero(M):
    M = sps.csr_matrix(M)
    M.eliminate_zeros()
    return M.nnz == 0


def eq(A, B):
    return iszero(sps.csr_matrix(A) - sps.csr_matrix(B))


def comm(A, B):
    return A @ B - B @ A


def ident(d):
    return sps.identity(d, dtype=np.int64, format="csr")


class Tower:
    def __init__(self, n):
        self.n = n
        self.E = tower(n)
        self.eta = ETA * n
        self.d = 8 ** n
        # 2 a+_p and 2 a-_p, p = 3k + i
        self.ap2 = [self.E[6 * k + i] + self.E[6 * k + i + 3] for k in range(n) for i in range(3)]
        self.am2 = [self.E[6 * k + i] - self.E[6 * k + i + 3] for k in range(n) for i in range(3)]
        self.N4p = [(self.ap2[p] @ self.am2[p]).tocsr() for p in range(3 * n)]  # 4 N_p
        self.Np = [np.rint(x.diagonal() / 4).astype(np.int64) for x in self.N4p]  # diagonal occupation numbers
        self.Ntot = sum(self.Np)
        self.Nk = [sum(self.Np[3 * k + i] for i in range(3)) for k in range(n)]
        self.Ndir = [sum(self.Np[3 * k + i] for k in range(n)) for i in range(3)]

    def lift4(self, A):
        """4 sigma(A) for a rational vector-level generator A given as integer matrix (or sympy)."""
        A = np.asarray(A)
        out = sps.csr_matrix((self.d, self.d), dtype=np.int64)
        for a in range(6 * self.n):
            for b in range(6 * self.n):
                if A[a, b] != 0:
                    out = out + int(A[a, b]) * self.eta[b] * (self.E[a] @ self.E[b])
        return out.tocsr()

    def diag(self, v):
        return sps.diags(np.asarray(v, dtype=np.int64), format="csr", dtype=np.int64)

    def onebody4(self, Rint):
        """sum_{pq} R_pq (2a+_p)(2a-_q) = 4 sum R_pq a+_p a-_q (integer R)."""
        out = sps.csr_matrix((self.d, self.d), dtype=np.int64)
        m = 3 * self.n
        for p in range(m):
            for q in range(m):
                if Rint[p, q] != 0:
                    out = out + int(Rint[p, q]) * (self.ap2[p] @ self.am2[q])
        return out.tocsr()


def vec_rot(n, p, q):
    """paired rotation at vector level: mode p -> mode q (temporal and partner alike); on z it is R with R_qp = 1."""
    A = np.zeros((6 * n, 6 * n), dtype=np.int64)
    for off in (0, 3):
        ip = 6 * (p // 3) + p % 3 + off
        iq = 6 * (q // 3) + q % 3 + off
        A[iq, ip], A[ip, iq] = 1, -1
    R = np.zeros((3 * n, 3 * n), dtype=np.int64)
    R[q, p], R[p, q] = 1, -1
    return A, R


G1 = np.block([[np.zeros((3, 3), dtype=np.int64), -np.eye(3, dtype=np.int64)],
               [np.eye(3, dtype=np.int64), np.zeros((3, 3), dtype=np.int64)]])


def Ekl(k, l, n):
    M = np.zeros((n, n), dtype=np.int64)
    M[k, l] = 1
    return M


def copy_rot_vec(k, l, n):
    return np.kron(Ekl(l, k, n) - Ekl(k, l, n), np.eye(6, dtype=np.int64))


def copy_boost_vec(k, l, n):
    return np.kron(Ekl(l, k, n) - Ekl(k, l, n), G1)


def block(M, rows, cols):
    return M[rows][:, cols]


def graded_parts(L, Ndiag):
    """set of differences N(row) - N(col) over the nonzero entries of L."""
    L = sps.coo_matrix(L)
    nz = L.data != 0
    return set((Ndiag[L.row[nz]] - Ndiag[L.col[nz]]).tolist())


TW = {n: Tower(n) for n in (2, 3)}


# ================================================================================================ 1. Fock reading
@claim("e4-1a", "conventions at n = 2, 3 (exact): the modes a+-_p satisfy the CAR, a-_p is the transpose of a+_p, the first basis "
                "vector is the Fock vacuum (a-_p|0> = 0 for all p), N = 3n/2 + sigma(K_n) is diagonal with spectrum 0..3n and "
                "multiplicities binom(3n, m)")
def _():
    ok = True
    for n, T in TW.items():
        m = 3 * n
        for p in range(m):
            ok &= eq(T.am2[p], T.ap2[p].T)
            for q in range(m):
                ok &= eq(T.ap2[p] @ T.am2[q] + T.am2[q] @ T.ap2[p], 4 * (p == q) * ident(T.d))
                ok &= eq(T.ap2[p] @ T.ap2[q] + T.ap2[q] @ T.ap2[p], sps.csr_matrix((T.d, T.d), dtype=np.int64))
            ok &= np.all(T.am2[p][:, 0].toarray() == 0)
        Kn = np.kron(np.eye(n, dtype=np.int64), np.block([[np.zeros((3, 3), dtype=np.int64), np.eye(3, dtype=np.int64)],
                                                           [np.eye(3, dtype=np.int64), np.zeros((3, 3), dtype=np.int64)]]))
        N4 = sum(T.N4p)
        ok &= eq(N4, 6 * n * ident(T.d) + T.lift4(Kn))
        ok &= eq(N4, T.diag(4 * T.Ntot))
        ok &= Counter(T.Ntot.tolist()) == Counter({k: comb(m, k) for k in range(m + 1)})
    return ok


@claim("e4-1b", "paired rotations at n = 2, 3 (exact, all 15 resp. 36 basis elements of so(3n)): sigma(X) equals the one-body "
                "operator sum_pq R_pq a+_p a-_q, where R is the real antisymmetric matrix of X on z; hence [sigma(X), N] = 0, and "
                "sigma(X) maps the one-quantum state a+_q|0> to sum_p R_pq a+_p|0>")
def _():
    ok = True
    for n, T in TW.items():
        m = 3 * n
        cnt = 0
        vac = np.zeros(T.d, dtype=np.int64)
        vac[0] = 1
        for p, q in itertools.combinations(range(m), 2):
            A, R = vec_rot(n, p, q)
            L4 = T.lift4(A)
            ok &= eq(L4, T.onebody4(R))
            ok &= eq(comm(L4, T.diag(T.Ntot)), sps.csr_matrix((T.d, T.d), dtype=np.int64))
            # one-quantum action: sigma(X) (2a+_r|0>) = sum_s R_sr (2a+_s|0>)
            for r in range(m):
                lhs = L4 @ (T.ap2[r] @ vac)
                rhs = 4 * sum(R[s, r] * (T.ap2[s] @ vac) for s in range(m))
                ok &= np.array_equal(lhs, rhs)
            cnt += 1
        ok &= cnt == m * (m - 1) // 2
    return ok


@claim("e4-1c", "copy rotations at n = 3 (exact): sigma(C^{kl}) commutes with N, with N^(k) + N^(l), with N^(m) for m not in {k,l}, and "
                "with each direction number N_i = sum_k N_(k,i); it does not commute with N^(k); it equals "
                "sum_i (a+_(l,i) a-_(k,i) - a+_(k,i) a-_(l,i)); the three lifts close on so(3) as on vectors ([C^12, C^13] = C^23)")
def _():
    T = TW[3]
    n = 3
    ok = True
    Z = sps.csr_matrix((T.d, T.d), dtype=np.int64)
    lifts = {}
    for k, l in itertools.combinations(range(n), 2):
        L4 = T.lift4(copy_rot_vec(k, l, n))
        lifts[(k, l)] = L4
        spin2 = sum(-ETA[a] * (T.E[6 * k + a] @ T.E[6 * l + a]) for a in range(6))  # 2 * (-(1/2) sum eta E E)
        ok &= eq(L4, 2 * spin2)
        Rint = np.zeros((3 * n, 3 * n), dtype=np.int64)
        for i in range(3):
            Rint[3 * l + i, 3 * k + i], Rint[3 * k + i, 3 * l + i] = 1, -1
        ok &= eq(L4, T.onebody4(Rint))
        ok &= eq(comm(L4, T.diag(T.Ntot)), Z)
        ok &= eq(comm(L4, T.diag(T.Nk[k] + T.Nk[l])), Z)
        mm = [x for x in range(n) if x not in (k, l)][0]
        ok &= eq(comm(L4, T.diag(T.Nk[mm])), Z)
        for i in range(3):
            ok &= eq(comm(L4, T.diag(T.Ndir[i])), Z)
        ok &= not eq(comm(L4, T.diag(T.Nk[k])), Z)
    # closure: vector level [C01, C02] = c C12, the same at spin level (lift4 is 4 sigma: [4s,4s] = 16 [s,s] = 4 * 4 sigma)
    C01, C02, C12 = (copy_rot_vec(0, 1, 3), copy_rot_vec(0, 2, 3), copy_rot_vec(1, 2, 3))
    cv = C01 @ C02 - C02 @ C01
    c = -1 if np.array_equal(cv, -C12) else (1 if np.array_equal(cv, C12) else 0)
    ok &= c == 1
    ok &= eq(comm(lifts[(0, 1)], lifts[(0, 2)]), 4 * c * lifts[(1, 2)])
    return ok


@claim("e4-1d", "copy boosts at n = 3 (exact): sigma((E_lk - E_kl)(x)G) = -sum_i (a+_(k,i) a+_(l,i) + a-_(l,i) a-_(k,i)); its matrix "
                "elements join only sectors with N differing by +2 or -2, both occurring; it commutes with N^(k) - N^(l) and with "
                "N^(m), m not in {k,l}, so each created pair has one quantum in constituent k and one in l; a generic paired boost of "
                "so(9,C) (random integer S) also changes N by exactly +-2, both occurring")
def _():
    T = TW[3]
    n = 3
    ok = True
    Z = sps.csr_matrix((T.d, T.d), dtype=np.int64)
    for k, l in itertools.combinations(range(n), 2):
        L4 = T.lift4(copy_boost_vec(k, l, n))
        expect = sum(-(T.ap2[3 * k + i] @ T.ap2[3 * l + i] + T.am2[3 * l + i] @ T.am2[3 * k + i]) for i in range(3))
        ok &= eq(L4, expect)
        ok &= graded_parts(L4, T.Ntot) == {2, -2}
        ok &= eq(comm(L4, T.diag(T.Nk[k] - T.Nk[l])), Z)
        mm = [x for x in range(n) if x not in (k, l)][0]
        ok &= eq(comm(L4, T.diag(T.Nk[mm])), Z)
    # generic paired boost: on z it acts as iQ with Q real antisymmetric; at vector level [[0,-Q],[Q,0]] per (temporal, partner)
    rng = np.random.default_rng(4)
    Q = rng.integers(-3, 4, size=(9, 9))
    Q = Q - Q.T
    A = np.zeros((18, 18), dtype=np.int64)
    for p in range(9):
        for q in range(9):
            tp, sp_ = 6 * (p // 3) + p % 3, 6 * (p // 3) + p % 3 + 3
            tq, sq = 6 * (q // 3) + q % 3, 6 * (q // 3) + q % 3 + 3
            A[tp, sq] += -Q[p, q]
            A[sp_, tq] += Q[p, q]
    Gn = np.kron(np.eye(3, dtype=np.int64), G1)
    etan = np.diag(ETA * 3)
    ok &= np.array_equal(A @ Gn, Gn @ A) and np.array_equal(A.T @ etan + etan @ A, np.zeros((18, 18), dtype=np.int64))
    ok &= graded_parts(T.lift4(A), T.Ntot) == {2, -2}
    return ok


KAP = (Fraction(1), Fraction(2), Fraction(4))
BV = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 5))  # B12, B23, B31


def Bmat_frac(b12, b23, b31):
    return [[0, b12, -b31], [-b12, 0, b23], [b31, -b23, 0]]


@claim("e4-1e", "second quantisation at n = 3, rates (1,2,4), (B12,B23,B31) = (1/2,1/3,1/5): HH = sum_k kappa_k N^(k) - i sigma(b) is "
                "Hermitian and commutes with N; its block on the one-quantum sector, in the basis a+_p|0>, equals H = K~ - iR exactly "
                "(rational); its eigenvalues on the sectors N = 2 and N = 3 are the sums of 2 resp. 3 eigenvalues of H with distinct "
                "indices (floating point, 1e-9)")
def _():
    T = TW[3]
    n = 3
    ok = True
    B = Bmat_frac(*BV)
    B30 = np.array([[int(30 * x) for x in row] for row in B], dtype=np.int64)
    b30 = np.kron(B30, np.eye(6, dtype=np.int64))
    L4 = T.lift4(b30)  # 4 * 30 sigma(b)
    # one-quantum block: basis a+_p|0> = (2a+_p)|0>/2
    vac = np.zeros(T.d, dtype=np.int64)
    vac[0] = 1
    one = np.array([(T.ap2[p] @ vac) // 2 for p in range(9)])  # rows: states (integer, unit vectors up to sign)
    ok &= np.array_equal(one @ one.T, np.eye(9, dtype=np.int64))
    Lblk = one @ (L4 @ one.T)  # 120 sigma(b) on N = 1
    R30 = np.kron(B30, np.eye(3, dtype=np.int64))
    ok &= np.array_equal(Lblk, 4 * R30)
    Nblk = [one @ (T.diag(T.Nk[k]) @ one.T) for k in range(n)]
    Kt = np.kron(np.diag([1, 2, 4]), np.eye(3, dtype=np.int64))
    ok &= np.array_equal(sum(int(KAP[k]) * Nblk[k] for k in range(n)), Kt)
    # numerics: HH on S_3
    Bf = np.array([[float(x) for x in row] for row in B])
    sig = (L4.toarray() / 120.0)
    HH = sum(float(KAP[k]) * np.diag(T.Nk[k].astype(float)) for k in range(n)) - 1j * sig
    ok &= np.allclose(HH, HH.conj().T, atol=1e-12)
    ok &= np.allclose(HH @ np.diag(T.Ntot) - np.diag(T.Ntot) @ HH, 0, atol=1e-12)
    H = np.kron(np.diag([1.0, 2.0, 4.0]), np.eye(3)) - 1j * np.kron(Bf, np.eye(3))
    ev = np.linalg.eigvalsh(H)
    for m in (2, 3):
        idx = np.where(T.Ntot == m)[0]
        w = np.sort(np.linalg.eigvalsh(HH[np.ix_(idx, idx)]))
        sums = np.sort([sum(c) for c in itertools.combinations(ev, m)])
        ok &= len(w) == len(sums) and np.allclose(w, sums, atol=1e-9)
    return ok


@claim("e4-1f", "gauge reading at n = 3 (exact): the quarter turn U = e^{(pi/2) sigma(C^{12})} = (1/8) prod_a (1 + 2 t_a), t_a the six "
                "commuting terms of sigma(C^{12}), is orthogonal, commutes with N and with N^(3), and maps N^(1) to N^(2): "
                "U N^(1) U^T = N^(2); the conjugate of sum_k kappa_k N^(k) has the same spectrum and, for kappa_1 != kappa_2, differs "
                "from it")
def _():
    T = TW[3]
    ok = True
    t2 = [(-ETA[a] * (T.E[0 + a] @ T.E[6 + a])).tocsr() for a in range(6)]  # 2 t_a, signed permutations
    for a in range(6):
        ok &= eq(t2[a] @ t2[a], -ident(T.d))
        for b in range(6):
            ok &= eq(comm(t2[a], t2[b]), sps.csr_matrix((T.d, T.d), dtype=np.int64))
    U8 = ident(T.d)
    for a in range(6):
        U8 = (U8 @ (ident(T.d) + t2[a])).tocsr()  # 8 U
    ok &= eq(U8 @ U8.T, 64 * ident(T.d))
    ok &= eq(comm(U8, T.diag(T.Ntot)), sps.csr_matrix((T.d, T.d), dtype=np.int64))
    ok &= eq(comm(U8, T.diag(T.Nk[2])), sps.csr_matrix((T.d, T.d), dtype=np.int64))
    ok &= eq(U8 @ T.diag(T.Nk[0]) @ U8.T, 64 * T.diag(T.Nk[1]))
    Hk = T.diag(1 * T.Nk[0] + 2 * T.Nk[1] + 4 * T.Nk[2])
    conj = (U8 @ Hk @ U8.T)
    ok &= not eq(conj, 64 * Hk)
    w = np.sort(np.linalg.eigvalsh(conj.toarray() / 64.0))
    ok &= np.allclose(w, np.sort(Hk.diagonal().astype(float)), atol=1e-9)
    return ok


# ================================================================================================ 2. time-odd invariants
def eta_n(n):
    return sp.diag(*(ETA * n))


def sym_kron(A, B):
    return sp.kronecker_product(sp.Matrix(A), sp.Matrix(B))


def commutant_exact(mats, N):
    """exact basis of {X in gl(N): XM = MX for all M}."""
    rows = []
    for M in mats:
        M = sp.Matrix(M)
        # vec(XM - MX) = (M^T (x) I - I (x) M) vec X  (column-major vec)
        rows.append(sp.kronecker_product(M.T, sp.eye(N)) - sp.kronecker_product(sp.eye(N), M))
    big = sp.Matrix.vstack(*rows)
    ns = big.nullspace()
    return [sp.Matrix(N, N, lambda i, j, v=v: v[j * N + i]) for v in ns]


@claim("e4-2a", "n = 4, rates (1,2,3,5) (exact): the endomorphisms of V_4 commuting with G_4, the diagonal l_G and K(x)G are "
                "sum_k (a_k + b_k G) on copy k (dimension 8), and the eta_4-isometries among them are the 16 copy reflections "
                "diag(+-1,+-1,+-1,+-1)(x)1_6")
def _():
    n = 4
    Z3, I3 = sp.zeros(3), sp.eye(3)
    G = sp.Matrix(sp.BlockMatrix([[Z3, -I3], [I3, Z3]]))
    R3 = []
    for i, j in ((1, 2), (2, 0), (0, 1)):
        r = sp.zeros(3)
        r[i, j], r[j, i] = -1, 1
        R3.append(r)
    LG = [sp.diag(r, r) for r in R3] + [sp.Matrix(sp.BlockMatrix([[Z3, -r], [r, Z3]])) for r in R3]
    mats = [sym_kron(sp.eye(n), G)] + [sym_kron(sp.eye(n), g) for g in LG] + [sym_kron(sp.diag(1, 2, 3, 5), G)]
    basis = commutant_exact(mats, 6 * n)
    ok = len(basis) == 8
    expect = [sym_kron(Ekl(k, k, n), sp.eye(6)) for k in range(n)] + [sym_kron(Ekl(k, k, n), G) for k in range(n)]
    span = sp.Matrix.hstack(*[b.reshape(36 * n * n, 1) for b in basis])
    both = sp.Matrix.hstack(span, *[e_.reshape(36 * n * n, 1) for e_ in expect])
    ok &= span.rank() == 8 and both.rank() == 8
    # isometry condition for X = sum_k (a_k + b_k G)_k: X^T eta X = eta  <=>  a_k b_k = 0, a_k^2 - b_k^2 = 1 for each k
    a, b = sp.symbols('a b', real=True)
    X = a * sp.eye(6) + b * G
    cond = sp.simplify(X.T * sp.diag(*ETA) * X - sp.diag(*ETA))
    sols = sp.solve([cond[0, 0], cond[0, 3]], [a, b], dict=True)
    real_sols = {(s[a], s[b]) for s in sols if s[a].is_real and s[b].is_real}
    ok &= real_sols == {(1, 0), (-1, 0)}
    return ok


def edges(n):
    return list(itertools.combinations(range(n), 2))


def rank_gf2(M):
    M = (np.array(M, dtype=np.int64) % 2).copy()
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


@claim("e4-2b", "copy reflections on sign patterns, n = 3..6 (exact, exhaustive): the coboundary map Z_2^n -> Z_2^E (s -> s_k s_l) has "
                "rank n-1 over Z_2, so the cycle space of the complete graph has rank c = (n-1)(n-2)/2 = 1, 3, 6, 10; the 2^E sign "
                "patterns fall into exactly 2^c orbits, and the signs of the c triangles (1,k,l), 2 <= k < l <= n, are constant on "
                "orbits and separate them")
def _():
    ok = True
    for n in (3, 4, 5, 6):
        Es = edges(n)
        Emat = np.zeros((len(Es), n), dtype=np.int64)
        for e_, (k, l) in enumerate(Es):
            Emat[e_, k] = Emat[e_, l] = 1
        r = rank_gf2(Emat)
        c = len(Es) - r
        ok &= r == n - 1 and c == (n - 1) * (n - 2) // 2
        # exhaustive orbits: encode sign pattern as bits (1 = negative)
        nE = len(Es)
        refl_masks = []
        for s in range(2 ** n):
            mask = 0
            for e_, (k, l) in enumerate(Es):
                if ((s >> k) & 1) ^ ((s >> l) & 1):
                    mask |= 1 << e_
            refl_masks.append(mask)
        refl_masks = sorted(set(refl_masks))
        ok &= len(refl_masks) == 2 ** (n - 1)
        eidx = {e_: i for i, e_ in enumerate(Es)}
        tris = [(0, k, l) for k, l in itertools.combinations(range(1, n), 2)]
        ok &= len(tris) == c

        def trisig(pat):
            out = 0
            for t, (x, y, z) in enumerate(tris):
                par = ((pat >> eidx[(x, y)]) ^ (pat >> eidx[(y, z)]) ^ (pat >> eidx[(x, z)])) & 1
                out |= par << t
            return out
        seen = np.zeros(2 ** nE, dtype=bool)
        orbits = 0
        labels = set()
        for pat in range(2 ** nE):
            if seen[pat]:
                continue
            orbits += 1
            orb = [pat ^ m for m in refl_masks]
            sig = {trisig(x) for x in orb}
            ok &= len(sig) == 1
            labels |= sig
            seen[orb] = True
        ok &= orbits == 2 ** c and len(labels) == 2 ** c
    return ok


@claim("e4-2c", "interior time reversal B -> -B, n = 3..6 (exact, exhaustive): it flips the sign of every triangle and multiplies "
                "the sign of a cycle of length L by (-1)^L; no sign pattern of the complete graph is reflection-equivalent to its "
                "negative (the complete graph on n >= 3 vertices is not bipartite)")
def _():
    ok = True
    for n in (3, 4, 5, 6):
        Es = edges(n)
        nE = len(Es)
        full = 2 ** nE - 1
        refl = set()
        for s in range(2 ** n):
            mask = 0
            for e_, (k, l) in enumerate(Es):
                if ((s >> k) & 1) ^ ((s >> l) & 1):
                    mask |= 1 << e_
            refl.add(mask)
        ok &= full not in refl  # -B = B * (all signs flipped) is not s_k s_l B for any s
        # a cycle of length L picks (-1)^L: for every simple cycle and every sign pattern (n <= 5 exhaustive, n = 6 sampled)
        eidx = {e_: i for i, e_ in enumerate(Es)}
        pats = range(2 ** nE) if n <= 5 else np.random.default_rng(1).integers(0, 2 ** nE, size=64).tolist()
        for L in range(3, n + 1):
            for cyc in itertools.permutations(range(n), L):
                if cyc[0] != min(cyc) or cyc[1] > cyc[-1]:
                    continue
                bits = [eidx[tuple(sorted((cyc[i], cyc[(i + 1) % L])))] for i in range(L)]
                for pat in pats:
                    par = sum((pat >> b_) & 1 for b_ in bits) % 2
                    parf = sum(((pat ^ full) >> b_) & 1 for b_ in bits) % 2
                    ok &= parf == (par + L) % 2
    return ok


def components(n, Es):
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for k, l in Es:
        par[find(k)] = find(l)
    return len({find(x) for x in range(n)})


@claim("e4-2f", "general coupling graphs (exact): for every graph E' of nonzero couplings on n = 4 vertices (all 64, exhaustive orbit count) "
                "the copy reflections act on the sign patterns of E' with exactly 2^{|E'| - n + c(E')} orbits, c the number of connected "
                "components; on n = 5 (all 1024 graphs) and n = 6 (all 32768 graphs) the rank of the coboundary map over Z_2 is n - c(E')")
def _():
    ok = True
    for n in (4, 5, 6):
        Es = edges(n)
        for gmask in range(2 ** len(Es)):
            Eg = [e_ for i, e_ in enumerate(Es) if (gmask >> i) & 1]
            c = components(n, Eg)
            if n == 4:
                masks = set()
                for s_ in range(2 ** n):
                    m_ = 0
                    for i, (k, l) in enumerate(Eg):
                        if ((s_ >> k) & 1) ^ ((s_ >> l) & 1):
                            m_ |= 1 << i
                    masks.add(m_)
                seen, orbits = set(), 0
                for pat in range(2 ** len(Eg)):
                    if pat in seen:
                        continue
                    orbits += 1
                    seen |= {pat ^ m_ for m_ in masks}
                ok &= orbits == 2 ** (len(Eg) - n + c)
            else:
                if not Eg:
                    ok &= c == n
                    continue
                M = np.zeros((len(Eg), n), dtype=np.int64)
                for i, (k, l) in enumerate(Eg):
                    M[i, k] = M[i, l] = 1
                ok &= rank_gf2(M) == n - c
    return ok


def H_and_K(n):
    kap = sp.symbols(f'k1:{n + 1}', real=True)
    Bs = {}
    for k, l in edges(n):
        Bs[(k, l)] = sp.Symbol(f'B{k + 1}{l + 1}', real=True)
    B = sp.zeros(n)
    for (k, l), s in Bs.items():
        B[k, l], B[l, k] = s, -s
    K = sp.diag(*kap)
    H = K - sp.I * B
    return H, K, B, kap, Bs


def cycle_cover_det(D, n):
    """det of a symmetric zero-diagonal matrix via cycle covers: sum over derangements grouped by cycle type."""
    total = 0
    for perm in itertools.permutations(range(n)):
        if any(perm[i] == i for i in range(n)):
            continue
        # sign = (-1)^(n - #cycles)
        seen, ncyc = set(), 0
        for i in range(n):
            if i not in seen:
                ncyc += 1
                j = i
                while j not in seen:
                    seen.add(j)
                    j = perm[j]
        term = (-1) ** (n - ncyc)
        for i in range(n):
            term *= D[i, perm[i]]
        total += term
    return sp.expand(total)


@claim("e4-2d", "triangle traces (symbolic, n = 4): with P_k the spectral projectors of K (distinct rates), "
                "Im tr(P_k H P_l H P_m H) = B_kl B_lm B_mk for every triangle (k,l,m) of copy indices, invariant under the copy "
                "reflections and odd under B -> -B; the product of the four triangle invariants is a product of squares")
def _():
    n = 4
    H, K, B, kap, Bs = H_and_K(n)
    ok = True
    P = [Ekl(k, k, n) for k in range(n)]
    P = [sp.Matrix(p) for p in P]
    # P_k are polynomials in K for distinct rates: Lagrange interpolation
    for k in range(n):
        Lk = sp.eye(n)
        for j in range(n):
            if j != k:
                Lk = Lk * (K - kap[j] * sp.eye(n)) / (kap[k] - kap[j])
        ok &= sp.simplify(Lk - P[k]) == sp.zeros(n)
    prods = []
    for (k, l, m) in itertools.combinations(range(n), 3):
        val = sp.expand((P[k] * H * P[l] * H * P[m] * H).trace())
        target = B[k, l] * B[l, m] * B[m, k]
        ok &= sp.simplify(sp.im(val) - target) == 0 and sp.simplify(sp.re(val)) == 0
        prods.append(target)
        for s in itertools.product((1, -1), repeat=n):
            S = sp.diag(*s)
            Bs_ = S * B * S
            ok &= sp.simplify(Bs_[k, l] * Bs_[l, m] * Bs_[m, k] - target) == 0
    tot = sp.factor(prods[0] * prods[1] * prods[2] * prods[3])
    ok &= all(e_ % 2 == 0 for _, e_ in sp.factor_list(tot)[1])
    return ok


@claim("e4-2e", "Jarlskog-type determinant (symbolic): [H,K] = iD with D_kl = (kappa_k - kappa_l) B_kl symmetric with zero diagonal, "
                "so det[H,K] = i^n det D; n = 3: det[H,K] = 2iJ(k1-k2)(k1-k3)(k2-k3); n = 4: det[H,K] = D12^2 D34^2 + D13^2 D24^2 + "
                "D14^2 D23^2 - 2(D12 D23 D34 D41 + D12 D24 D43 D31 + D13 D32 D24 D41), even under B -> -B; n = 4, 5: det D equals "
                "the cycle-cover sum sum_sigma (-1)^(n - #cycles) prod D_{k sigma(k)} over derangements, and at n = 5 it is odd under "
                "B -> -B")
def _():
    ok = True
    # n = 3
    H, K, B, kap, Bs = H_and_K(3)
    J = B[0, 1] * B[1, 2] * B[2, 0]
    d3 = sp.expand((H * K - K * H).det(method='berkowitz'))
    ok &= sp.expand(d3 - 2 * sp.I * J * (kap[0] - kap[1]) * (kap[0] - kap[2]) * (kap[1] - kap[2])) == 0
    for n in (4, 5):
        H, K, B, kap, Bs = H_and_K(n)
        C = H * K - K * H
        D = sp.Matrix(n, n, lambda k, l: (kap[k] - kap[l]) * B[k, l])
        ok &= (C - sp.I * D).expand() == sp.zeros(n)
        ok &= (D - D.T).expand() == sp.zeros(n) and all(D[k, k] == 0 for k in range(n))
        detC = sp.expand(C.det(method='berkowitz'))
        detD = sp.expand(D.det(method='berkowitz'))
        ok &= sp.expand(detC - sp.I ** n * detD) == 0
        ok &= sp.expand(detD - cycle_cover_det(D, n)) == 0
        flip = {s: -s for s in Bs.values()}
        if n == 4:
            d = lambda a, b: D[a - 1, b - 1]
            formula = (d(1, 2) ** 2 * d(3, 4) ** 2 + d(1, 3) ** 2 * d(2, 4) ** 2 + d(1, 4) ** 2 * d(2, 3) ** 2
                       - 2 * (d(1, 2) * d(2, 3) * d(3, 4) * d(4, 1) + d(1, 2) * d(2, 4) * d(4, 3) * d(3, 1)
                              + d(1, 3) * d(3, 2) * d(2, 4) * d(4, 1)))
            ok &= sp.expand(detC - formula) == 0
            ok &= sp.expand(detC.subs(flip, simultaneous=True) - detC) == 0
        else:
            ok &= sp.expand(detC.subs(flip, simultaneous=True) + detC) == 0
    return ok


# ================================================================================================ 3. copy isospin at n = 4
T4 = Tower(4)


def spin_copy2(T, k, l):
    """2 sigma(C^{kl}) = -sum_a eta_a E_(k,a) E_(l,a) (integer)."""
    return sum(-ETA[a] * (T.E[6 * k + a] @ T.E[6 * l + a]) for a in range(6)).tocsr()


C2 = {(k, l): spin_copy2(T4, k, l) for k, l in itertools.combinations(range(4), 2)}


def C2get(k, l):
    return C2[(k, l)] if k < l else -C2[(l, k)]


# self-dual / anti-self-dual combinations on the copy index (0-based): J^+-_1 = (C23 +- C14)/2, J^+-_2 = (C31 +- C24)/2,
# J^+-_3 = (C12 +- C34)/2 (1-based labels); 4 J = 2C + 2C' is an integer matrix
PAIRS = [((1, 2), (0, 3)), ((2, 0), (1, 3)), ((0, 1), (2, 3))]


def J4(sign):
    return [(C2get(*p) + sign * C2get(*q)).tocsr() for p, q in PAIRS]


JP, JM = J4(+1), J4(-1)


def vecJ(sign):
    out = []
    for p, q in PAIRS:
        Lp = Ekl(p[1], p[0], 4) - Ekl(p[0], p[1], 4)
        Lq = Ekl(q[1], q[0], 4) - Ekl(q[0], q[1], 4)
        out.append((Lp + sign * Lq))  # 2 J on vectors
    return out


@claim("e4-3a", "n = 4, S_4 of dimension 4096 (exact, sparse integer): each copy rotation sigma(C^{kl}) is a sum of six commuting terms t_a "
                "with t_a^2 = -1/4, so exp(2 pi sigma(C^{kl})) = prod_a (-1) = +1; the same holds on S_5 (dimension 32768) for "
                "C^{12}; each sigma(C^{kl}) equals the one-body operator sum_i (a+_(l,i) a-_(k,i) - a+_(k,i) a-_(l,i)) and commutes with "
                "the direction numbers N_i; on vectors and on S_4 the six copy rotations close on so(4)")
def _():
    ok = True
    Z4 = sps.csr_matrix((T4.d, T4.d), dtype=np.int64)
    for (k, l), Cm in C2.items():
        Rint = np.zeros((12, 12), dtype=np.int64)
        for i in range(3):
            Rint[3 * l + i, 3 * k + i], Rint[3 * k + i, 3 * l + i] = 1, -1
        ok &= eq(2 * Cm, T4.onebody4(Rint)) and eq(2 * Cm, T4.lift4(copy_rot_vec(k, l, 4)))
        for i in range(3):
            ok &= eq(comm(Cm, T4.diag(T4.Ndir[i])), Z4)
    for T in (T4, Tower(5)):
        t2 = [(-ETA[a] * (T.E[a] @ T.E[6 + a])).tocsr() for a in range(6)]
        Z = sps.csr_matrix((T.d, T.d), dtype=np.int64)
        for a in range(6):
            ok &= eq(t2[a] @ t2[a], -ident(T.d))
            for b in range(a + 1, 6):
                ok &= eq(comm(t2[a], t2[b]), Z)
        # exp(2 pi t) = cos(pi) + sin(pi) * 2t = -1 for each term; product of six = +1 (exact consequence)
    # so(4) closure: [C^{kl}, C^{lm}] as on vectors
    Z = sps.csr_matrix((T4.d, T4.d), dtype=np.int64)
    for (k, l), (k2, l2) in itertools.combinations(list(C2), 2):
        Lv = copy_rot_vec(k, l, 4)[::6, ::6]
        Lv2 = copy_rot_vec(k2, l2, 4)[::6, ::6]
        cv = Lv @ Lv2 - Lv2 @ Lv
        # express cv in the basis of copy rotations
        target = Z
        for (a, b) in C2:
            coeff = cv[b, a]  # (E_ba - E_ab) has entry +1 at (b, a)
            if coeff:
                target = target + int(coeff) * 2 * C2[(a, b)]
        ok &= eq(comm(C2[(k, l)], C2[(k2, l2)]), target)
    return ok


@claim("e4-3b", "n = 4 (exact): with J^+-_1 = (C^{23} +- C^{14})/2, J^+-_2 = (C^{31} +- C^{24})/2, J^+-_3 = (C^{12} +- C^{34})/2, "
                "[J^+_a, J^-_b] = 0 and [J^+-_1, J^+-_2] = J^+-_3 (cyclic) on vectors and on S_4; on vectors 4(J^+-_a)^2 = -1, "
                "the vector is (1/2,1/2); on S_4, exp(2 pi J^+_3) = exp(2 pi J^-_3) = (-1)^N (a product of twelve commuting signed "
                "permutations), so J^+_3 = T = (C^{12}+C^{34})/2 has half-integer eigenvalues exactly on odd N, and exp(2 pi T) = -1 "
                "on the one-quantum state a+_(1,1)|0>")
def _():
    ok = True
    Z = sps.csr_matrix((T4.d, T4.d), dtype=np.int64)
    for s in (+1, -1):
        Jv = vecJ(s)  # 2J on vectors (4x4)
        for a in range(3):
            ok &= np.array_equal(Jv[a] @ Jv[a], -np.eye(4, dtype=np.int64))  # (2J)^2 = -1  <=> 4 J^2 = -1
        ok &= np.array_equal(Jv[0] @ Jv[1] - Jv[1] @ Jv[0], 2 * Jv[2])  # [2J1, 2J2] = 2 (2J3)  <=> [J1,J2] = J3
        ok &= np.array_equal(Jv[1] @ Jv[2] - Jv[2] @ Jv[1], 2 * Jv[0])
        ok &= np.array_equal(Jv[2] @ Jv[0] - Jv[0] @ Jv[2], 2 * Jv[1])
    JvP, JvM = vecJ(1), vecJ(-1)
    for a in range(3):
        for b in range(3):
            ok &= np.array_equal(JvP[a] @ JvM[b] - JvM[b] @ JvP[a], np.zeros((4, 4), dtype=np.int64))
    # spin level: JP[a] = 4 J^+_a
    for J in (JP, JM):
        ok &= eq(comm(J[0], J[1]), 4 * J[2]) and eq(comm(J[1], J[2]), 4 * J[0]) and eq(comm(J[2], J[0]), 4 * J[1])
    for a in range(3):
        for b in range(3):
            ok &= eq(comm(JP[a], JM[b]), Z)
    parity = T4.diag((-1) ** T4.Ntot)
    for s in (+1, -1):
        # J^s_3 = (1/2)(sigma(C12) + s sigma(C34)) = sum of 12 commuting terms u with u^2 = -1/16; exp(2 pi u) = 4u = 2t
        fac = [(-ETA[a] * (T4.E[a] @ T4.E[6 + a])).tocsr() for a in range(6)]
        fac += [(s * (-ETA[a]) * (T4.E[12 + a] @ T4.E[18 + a])).tocsr() for a in range(6)]
        for x in fac:
            ok &= eq(x @ x, -ident(T4.d))
        for x, y in itertools.combinations(fac, 2):
            ok &= eq(comm(x, y), Z)
        Ex = ident(T4.d)
        for x in fac:
            Ex = (Ex @ x).tocsr()
        ok &= eq(Ex, parity)
    vac = np.zeros(T4.d, dtype=np.int64)
    vac[0] = 1
    v = T4.ap2[0] @ vac
    ok &= np.array_equal(parity @ v, -v)
    return ok


def cg(a, b):
    """Clebsch-Gordan for doubled spins: a (x) b = sum_{c = |a-b|..a+b step 2} c."""
    return list(range(abs(a - b), a + b + 1, 2))


def tensor_reps(R1, R2):
    out = Counter()
    for (a1, b1), m1 in R1.items():
        for (a2, b2), m2 in R2.items():
            for a in cg(a1, a2):
                for b in cg(b1, b2):
                    out[(a, b)] += m1 * m2
    return out


@claim("e4-3c", "n = 4, joint spectrum of the two su(2) Casimirs on S_4 (block-diagonalised by the direction numbers N_1, N_2, N_3, which "
                "commute with every copy rotation, exact; eigenvalues in floating point, 1e-8): the multiplicities of (j1,j2) agree "
                "with the character of (2(0,0) + 2(1/2,1/2) + (1,0) + (0,1))^{(x)3}, the so(4) module Lambda(R^4)^{(x)3}; every (j1,j2) "
                "has j1 - j2 integer, the doublet-containing ones (j1 half-integer) fill exactly the odd-N sectors (2048 dimensions), "
                "and the one-quantum sector is (1/2,1/2)(x)R^3")
def _():
    ok = True
    T = T4
    Z = sps.csr_matrix((T.d, T.d), dtype=np.int64)
    for (k, l), Cm in C2.items():
        for i in range(3):
            ok &= eq(comm(Cm, T.diag(T.Ndir[i])), Z)
    # 16 Cas = -sum (4J)^2  -> eigenvalue 16 j (j+1)
    CasP = (-(JP[0] @ JP[0] + JP[1] @ JP[1] + JP[2] @ JP[2])).tocsr()
    CasM = (-(JM[0] @ JM[0] + JM[1] @ JM[1] + JM[2] @ JM[2])).tocsr()
    ok &= eq(comm(CasP, CasM), Z)
    keys = T.Ndir[0] * 25 + T.Ndir[1] * 5 + T.Ndir[2]
    coo = CasP.tocoo()
    ok &= np.all(keys[coo.row] == keys[coo.col])
    coo = CasM.tocoo()
    ok &= np.all(keys[coo.row] == keys[coo.col])
    CP, CM = CasP.toarray().astype(float), CasM.toarray().astype(float)
    found = Counter()
    odd_dims = 0
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        a = CP[np.ix_(idx, idx)]
        w, V = np.linalg.eigh(a)
        lev = np.round(w).astype(int)
        ok &= np.allclose(w, lev, atol=1e-8)
        for L in np.unique(lev):
            sub = V[:, lev == L]
            b = sub.T @ CM[np.ix_(idx, idx)] @ sub
            w2 = np.linalg.eigvalsh(b)
            lev2 = np.round(w2).astype(int)
            ok &= np.allclose(w2, lev2, atol=1e-8)
            for L2 in np.unique(lev2):
                mult = int(np.sum(lev2 == L2))
                # 16 j(j+1) = L  -> 2j from (2j)(2j+2) = L/4 ... solve  (2j+1)^2 = L/4 + 1
                tj1 = int(round(np.sqrt(L / 4 + 1))) - 1
                tj2 = int(round(np.sqrt(L2 / 4 + 1))) - 1
                ok &= tj1 * (tj1 + 2) * 4 == L and tj2 * (tj2 + 2) * 4 == L2
                dim = (tj1 + 1) * (tj2 + 1)
                ok &= mult % dim == 0
                found[(tj1, tj2)] += mult // dim
                if tj1 % 2 == 1:
                    odd_dims += mult
        # parity of N inside the block is fixed by the key
    M = Counter({(0, 0): 2, (1, 1): 2, (2, 0): 1, (0, 2): 1})
    char = tensor_reps(tensor_reps(M, M), M)
    ok &= found == char
    ok &= all((a - b) % 2 == 0 for a, b in found)
    ok &= odd_dims == 2048
    # odd-N sectors are exactly the half-integer-j1 part: check via (-1)^N = exp(2 pi J3) (e4-3b) and dimension count
    ok &= int(np.sum(T.Ntot % 2 == 1)) == 2048
    # one-quantum sector: 12 states, Casimirs both 3/4 (16 * 3/4 = 12)
    idx1 = np.where(T.Ntot == 1)[0]
    ok &= len(idx1) == 12
    ok &= np.allclose(CP[np.ix_(idx1, idx1)], 12 * np.eye(12)) and np.allclose(CM[np.ix_(idx1, idx1)], 12 * np.eye(12))
    print("    . (2j1,2j2): multiplicity on S_4 =", dict(sorted(found.items())))
    return ok


@claim("e4-3d", "n = 3 cross-check (floating point, 1e-8): the character of (2(j=0) + 2(j=1))^{(x)3}, the so(3) module Lambda(R^3)^{(x)3}, "
                "gives j = 0..3 with dimensions 40, 216, 200, 56, the Casimir spectrum of the copy so(3) on S_3 recomputed here "
                "(restated: RE Prop. 9.6(ii))")
def _():
    T = TW[3]
    Cs = [T.lift4(copy_rot_vec(k, l, 3)) for k, l in ((1, 2), (2, 0), (0, 1))]
    Cas = -(Cs[0] @ Cs[0] + Cs[1] @ Cs[1] + Cs[2] @ Cs[2]).toarray().astype(float) / 16.0
    w = np.linalg.eigvalsh(Cas)
    lev = np.round(w).astype(int)
    ok = np.allclose(w, lev, atol=1e-8)
    cnt = Counter(lev.tolist())
    ok &= cnt == Counter({0: 40, 2: 216, 6: 200, 12: 56})
    # character: doubled spins
    M = Counter({0: 2, 2: 2})
    def tens(R1, R2):
        out = Counter()
        for a, m1 in R1.items():
            for b, m2 in R2.items():
                for c in cg(a, b):
                    out[c] += m1 * m2
        return out
    ch = tens(tens(M, M), M)
    dims = {c // 2 * (c // 2 + 1): m * (c + 1) for c, m in ch.items()}
    ok &= Counter(dims) == cnt
    return ok


@claim("e4-3e", "n = 5, S_5 of dimension 32768 (exact, sparse integer): for the so(4) of the four constituents (1,2,3,4) and of (2,3,4,5), "
                "exp(2 pi J^+-_3) = (-1)^{N'} with N' the number of quanta in those four constituents (a product of twelve commuting signed "
                "permutations): its su(2) factors carry half-integral isospin exactly where N' is odd")
def _():
    T = Tower(5)
    ok = True
    for quad in ((0, 1, 2, 3), (1, 2, 3, 4)):
        q0, q1, q2, q3 = quad
        Nq = sum(T.Nk[k] for k in quad)
        par = T.diag((-1) ** Nq)
        for s_ in (+1, -1):
            fac = [(-ETA[a] * (T.E[6 * q0 + a] @ T.E[6 * q1 + a])).tocsr() for a in range(6)]
            fac += [(s_ * (-ETA[a]) * (T.E[6 * q2 + a] @ T.E[6 * q3 + a])).tocsr() for a in range(6)]
            Ex = ident(T.d)
            for x in fac:
                Ex = (Ex @ x).tocsr()
            ok &= eq(Ex, par)
        ok &= not eq(par, T.diag((-1) ** T.Ntot))
    return ok


# ================================================================================================ 4. rate patterns
def so_basis_int(m):
    out = []
    for p, q in itertools.combinations(range(m), 2):
        X = np.zeros((m, m), dtype=np.int64)
        X[q, p], X[p, q] = 1, -1
        out.append(X)
    return out


def removable_dim(rates, mult=1):
    """dim of {X in so(n*mult): [K (x) 1_mult, X] = 0}, exact rank over Q."""
    n = len(rates)
    K = sp.Matrix(np.kron(np.diag(rates), np.eye(mult, dtype=np.int64)).tolist())
    basis = so_basis_int(n * mult)
    cols = [sp.Matrix(K * sp.Matrix(X.tolist()) - sp.Matrix(X.tolist()) * K).reshape((n * mult) ** 2, 1) for X in basis]
    rk = sp.Matrix.hstack(*cols).rank()
    return len(basis) - rk


def partition_rates(parts):
    out = []
    for i, p in enumerate(parts):
        out += [i + 1] * p
    return out


@claim("e4-4a", "rate patterns (exact): for rates with multiplicities n_1 + ... + n_r = n the removable part of the copy so(n) has "
                "dimension sum dim so(n_i) and the non-removable part sum_{i<j} n_i n_j, for every partition of n = 2..5; within the "
                "compact binding-compatible algebra so(3n) the removable part has dimension sum dim so(3 n_i) and the non-removable "
                "part 9 sum_{i<j} n_i n_j, for every partition of n = 2, 3, 4")
def _():
    ok = True

    def partitions(n, maxp=None):
        if n == 0:
            yield []
            return
        maxp = n if maxp is None else maxp
        for p in range(min(n, maxp), 0, -1):
            for rest in partitions(n - p, p):
                yield [p] + rest
    rows = []
    for n in (2, 3, 4, 5):
        for parts in partitions(n):
            rates = partition_rates(parts)
            d = removable_dim(rates)
            exp_rem = sum(p * (p - 1) // 2 for p in parts)
            exp_non = sum(parts[i] * parts[j] for i, j in itertools.combinations(range(len(parts)), 2))
            ok &= d == exp_rem and n * (n - 1) // 2 - d == exp_non
            if n <= 4:
                d3 = removable_dim(rates, 3)
                ok &= d3 == sum(3 * p * (3 * p - 1) // 2 for p in parts) and 3 * n * (3 * n - 1) // 2 - d3 == 9 * exp_non
                rows.append((tuple(parts), d, exp_non, d3))
    print("    . (partition, removable in so(n), non-removable, removable in so(3n)):", rows)
    return ok


@claim("e4-4b", "the non-removable part as a mass matrix (exact, n = 4, rates (1,1,2,5)): the quadratic form B -> ||[K,B]||^2 on so(n) "
                "is diagonal in the basis of copy rotations C^{kl} with eigenvalue 2(kappa_k - kappa_l)^2; its kernel is the "
                "removable so(2) = span C^{12} and its nonzero eigenvalues are 2, 2, 18, 32, 32")
def _():
    rates = [1, 1, 2, 5]
    n = 4
    K = np.diag(rates)
    basis = [Ekl(l, k, n) - Ekl(k, l, n) for k, l in edges(n)]
    Gm = np.array([[int(np.sum((K @ X - X @ K) * (K @ Y - Y @ K))) for Y in basis] for X in basis])
    expect = np.diag([2 * (rates[k] - rates[l]) ** 2 for k, l in edges(n)])
    ok = np.array_equal(Gm, expect)
    ok &= Gm[0, 0] == 0 and sorted(np.diag(Gm).tolist()) == [0, 2, 2, 18, 32, 32]
    return ok


# ================================================================================================ 5. rotors up the ladder
def beta0_tensor():
    """integer tensor T[c, a, b] with beta_0(x, y)_c = sum T[c,a,b] x_a y_b, z = x[:3] + i x[3:], beta_0 = z_x x z_y."""
    T = np.zeros((6, 6, 6), dtype=np.int64)
    eps = np.zeros((3, 3, 3), dtype=np.int64)
    for (i, j, k), s in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1), ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)):
        eps[i, j, k] = s
    for c in range(3):
        for a in range(3):
            for b in range(3):
                e_ = eps[a, b, c]
                if e_:
                    T[c, a, b] += e_          # Re: Re x Re
                    T[c, a + 3, b + 3] -= e_  # Re: - Im x Im
                    T[c + 3, a, b + 3] += e_  # Im: Re x Im
                    T[c + 3, a + 3, b] += e_  # Im: Im x Re
    return T


B0 = beta0_tensor()
G6 = G1


def LG_gens():
    Z3, I3 = np.zeros((3, 3), dtype=np.int64), np.eye(3, dtype=np.int64)
    R3 = []
    for i, j in ((1, 2), (2, 0), (0, 1)):
        r = np.zeros((3, 3), dtype=np.int64)
        r[i, j], r[j, i] = -1, 1
        R3.append(r)
    return [np.block([[r, Z3], [Z3, r]]) for r in R3] + [np.block([[Z3, -r], [r, Z3]]) for r in R3]


@claim("e4-5a", "frame brackets, one block (exact, rational null space of the 1296 x 216 system): the real bilinear maps V x V -> V "
                "equivariant under l_G form a 2-dimensional space, spanned by beta_0 and G beta_0 (the complex multiples of the "
                "complex cross product); all are antisymmetric; hence the equivariant antisymmetric brackets on V_n are "
                "[X,Y]_k = sum Gamma^k_ij beta_0(X_i,Y_j) with complex Gamma^k_ij = Gamma^k_ji, of real dimension n^2(n+1) = 12, 36, 80")
def _():
    gens = LG_gens()
    rows = []
    # unknown T[c,a,b], vec index c*36 + a*6 + b; equivariance: g T(x,y) - T(gx,y) - T(x,gy) = 0
    for g in gens:
        M = np.zeros((216, 216), dtype=np.int64)
        for c in range(6):
            for a in range(6):
                for b in range(6):
                    r = c * 36 + a * 6 + b
                    # (g T)[c,a,b] = sum_d g[c,d] T[d,a,b]
                    for d in range(6):
                        if g[c, d]:
                            M[r, d * 36 + a * 6 + b] += g[c, d]
                    # T(gx, y)[c] coefficient of x_a y_b: sum_d T[c,d,b] g[d,a]
                    for d in range(6):
                        if g[d, a]:
                            M[r, c * 36 + d * 6 + b] -= g[d, a]
                        if g[d, b]:
                            M[r, c * 36 + a * 6 + d] -= g[d, b]
        rows.append(M)
    big = sp.Matrix(np.vstack(rows).tolist())
    ns = big.nullspace()
    ok = len(ns) == 2
    span = sp.Matrix.hstack(*ns)
    b0 = sp.Matrix(B0.reshape(216).tolist())
    gb0 = sp.Matrix(np.einsum('cd,dab->cab', G6, B0).reshape(216).tolist())
    ok &= sp.Matrix.hstack(span, b0, gb0).rank() == 2
    for v in ns:
        Tm = np.array([int(x * sp.ilcm(*[sp.fraction(y)[1] for y in v])) for x in v]).reshape(6, 6, 6)
        ok &= np.array_equal(Tm, -Tm.transpose(0, 2, 1))
    ok &= [n * n * (n + 1) for n in (2, 3, 4)] == [12, 36, 80]
    return ok


def rotor_block_dim(ki, kj, kk):
    """real dim of {mu in span(beta0, G beta0): kk G mu(x,y) = mu(ki G x, y) + mu(x, kj G y)} (exact)."""
    basis = [B0, np.einsum('cd,dab->cab', G6, B0)]
    cols = []
    for Tm in basis:
        lhs = kk * np.einsum('cd,dab->cab', G6, Tm)
        rhs = ki * np.einsum('cdb,da->cab', Tm, G6) + kj * np.einsum('cad,db->cab', Tm, G6)
        cols.append(sp.Matrix((lhs - rhs).reshape(216).tolist()))
    return 2 - sp.Matrix.hstack(*cols).rank()


def rotor_dim(rates):
    n = len(rates)
    tot = 0
    for k in range(n):
        for i in range(n):
            for j in range(i, n):
                tot += rotor_block_dim(rates[i], rates[j], rates[k])
    return tot


@claim("e4-5b", "the rotor condition A[X,Y] = [AX,Y] + [X,AY] for A = K(x)G (exact, block by block): a block (i,j -> k) survives "
                "iff kappa_k = kappa_i + kappa_j; real dimensions of the rotor-compatible equivariant brackets: (1,1,1): 0, (1,2,3): 4, "
                "(1,3,5): 0, (1,2,4): 4; n = 2: (1,1): 0, (1,3): 0, (1,2): 2 (restated d7-2b); n = 4: (1,2,3,4): 8; equally spaced "
                "rates (1..n) give 2 sum_k floor(k/2) = 2, 4, 8, 12, 18 for n = 2..6")
def _():
    ok = True
    ok &= rotor_dim((1, 1, 1)) == 0 and rotor_dim((1, 2, 3)) == 4 and rotor_dim((1, 3, 5)) == 0 and rotor_dim((1, 2, 4)) == 4
    ok &= rotor_dim((1, 1)) == 0 and rotor_dim((1, 3)) == 0 and rotor_dim((1, 2)) == 2
    ok &= rotor_dim((1, 2, 3, 4)) == 8
    for n in range(2, 7):
        brute = sum(1 for k in range(1, n + 1) for i in range(1, n + 1) for j in range(i, n + 1) if i + j == k)
        ok &= 2 * brute == 2 * sum(k // 2 for k in range(1, n + 1))
        if n <= 4:
            ok &= rotor_dim(tuple(range(1, n + 1))) == 2 * brute
    ok &= [2 * sum(k // 2 for k in range(1, n + 1)) for n in range(2, 7)] == [2, 4, 8, 12, 18]
    return ok


def cross_c(u, v):
    return np.cross(u, v)


def make_bracket(n, Gam):
    """Gam: dict (k, i, j) -> complex (i <= j); bracket on C^{3n} (z coordinates) [X,Y]_k = sum Gamma^k_ij beta(X_i, Y_j)."""
    def br(X, Y):
        X, Y = X.reshape(n, 3), Y.reshape(n, 3)
        out = np.zeros((n, 3), dtype=complex)
        for (k, i, j), g in Gam.items():
            out[k] += g * cross_c(X[i], Y[j])
            if i != j:
                out[k] += g * cross_c(X[j], Y[i])
        return out.reshape(3 * n)
    return br


def to_z(x, n):
    x = x.reshape(n, 6)
    return (x[:, :3] + 1j * x[:, 3:]).reshape(3 * n)


def from_z(z, n):
    z = z.reshape(n, 3)
    return np.concatenate([z.real, z.imag], axis=1).reshape(6 * n)


def jacobiator(br, X, Y, Z):
    return br(br(X, Y), Z) + br(br(Y, Z), X) + br(br(Z, X), Y)


@claim("e4-5c", "n = 3, rotor diag(1,2,3)(x)G (floating point, random samples, 1e-10): for every (a,b) in C^2 the bracket "
                "[X,Y]_2 = a beta_0(X_1,Y_1), [X,Y]_3 = b(beta_0(X_1,Y_2) + beta_0(X_2,Y_1)) is antisymmetric, l_G-equivariant, "
                "has the rotor as a derivation and satisfies the Jacobi identity; for a b != 0 the frame is a graded nilpotent Lie "
                "algebra V_1 + V_2 + V_3 generated by V_1, with lower central series of real dimensions 18, 12, 6, 0; the "
                "non-resonant block Gamma^1_11 violates the rotor condition")
def _():
    n = 3
    rng = np.random.default_rng(7)
    ok = True
    A = np.kron(np.diag([1.0, 2.0, 3.0]), G6.astype(float))
    gens = [np.kron(np.eye(n), g.astype(float)) for g in LG_gens()]
    for trial in range(4):
        a, b = rng.normal(size=2) + 1j * rng.normal(size=2)
        brz = make_bracket(n, {(1, 0, 0): a, (2, 0, 1): b})
        br = lambda x, y: from_z(brz(to_z(x, n), to_z(y, n)), n)
        for _ in range(5):
            x, y, w = rng.normal(size=(3, 6 * n))
            ok &= np.allclose(br(x, y), -br(y, x), atol=1e-10)
            ok &= np.allclose(A @ br(x, y), br(A @ x, y) + br(x, A @ y), atol=1e-10)
            ok &= np.allclose(jacobiator(br, x, y, w), 0, atol=1e-10)
            for g in gens:
                ok &= np.allclose(g @ br(x, y), br(g @ x, y) + br(x, g @ y), atol=1e-10)
        # lower central series
        X = rng.normal(size=(60, 6 * n))
        g2 = np.array([br(X[i], X[j]) for i in range(60) for j in range(i + 1, 60)][:600])
        r2 = np.linalg.matrix_rank(g2, tol=1e-9 * np.abs(g2).max())
        g3 = np.array([br(g2[i], X[j % 60]) for i in range(300) for j in range(i, i + 2)])
        r3 = np.linalg.matrix_rank(g3, tol=1e-9 * np.abs(g3).max())
        g4 = np.array([br(g3[i], X[i % 60]) for i in range(300)])
        ok &= r2 == 12 and r3 == 6 and np.abs(g4).max() < 1e-9
    # non-resonant block fails the rotor condition
    brz = make_bracket(n, {(0, 0, 0): 1.0})
    br = lambda x, y: from_z(brz(to_z(x, n), to_z(y, n)), n)
    x, y = rng.normal(size=(2, 6 * n))
    ok &= not np.allclose(A @ br(x, y), br(A @ x, y) + br(x, A @ y), atol=1e-6)
    return ok


@claim("e4-5d", "n = 4, rotor diag(1,2,3,4)(x)G (floating point, 1e-10): with a = Gamma^2_11, b = Gamma^3_12, c = Gamma^4_13, d = "
                "Gamma^4_22, the Jacobiator of X, Y in V_1 and Z in V_2 equals (ad - bc) beta_0(beta_0(X,Y),Z) in V_4, all other "
                "components vanish, and the Jacobi identity holds on all of V_4 iff ad = bc (random samples with ad = bc pass, "
                "with ad != bc fail)")
def _():
    n = 4
    rng = np.random.default_rng(9)
    ok = True
    for trial in range(4):
        a, b, c, d = rng.normal(size=4) + 1j * rng.normal(size=4)
        brz = make_bracket(n, {(1, 0, 0): a, (2, 0, 1): b, (3, 0, 2): c, (3, 1, 1): d})
        X1, Y1, Z2 = (rng.normal(size=3) + 1j * rng.normal(size=3) for _ in range(3))
        X = np.concatenate([X1, np.zeros(9)])
        Y = np.concatenate([Y1, np.zeros(9)])
        Z = np.concatenate([np.zeros(3), Z2, np.zeros(6)])
        jac = jacobiator(brz, X, Y, Z)
        expect = np.concatenate([np.zeros(9), (a * d - b * c) * np.cross(np.cross(X1, Y1), Z2)])
        ok &= np.allclose(jac, expect, atol=1e-10) and np.abs(expect).max() > 1e-3
        # general elements
        Xg, Yg, Zg = (rng.normal(size=12) + 1j * rng.normal(size=12) for _ in range(3))
        ok &= np.abs(jacobiator(brz, Xg, Yg, Zg)).max() > 1e-6
        d2 = b * c / a
        brz2 = make_bracket(n, {(1, 0, 0): a, (2, 0, 1): b, (3, 0, 2): c, (3, 1, 1): d2})
        for _ in range(5):
            Xg, Yg, Zg = (rng.normal(size=12) + 1j * rng.normal(size=12) for _ in range(3))
            ok &= np.allclose(jacobiator(brz2, Xg, Yg, Zg), 0, atol=1e-10)
    return ok


@claim("e4-5e", "n = 3, rotor diag(1,2,4)(x)G (floating point, 1e-10): the rotor-compatible brackets are [X,Y]_2 = a beta_0(X_1,Y_1), "
                "[X,Y]_3 = b beta_0(X_2,Y_2); for X, Y in V_1 and Z in V_2 the Jacobiator is a b beta_0(beta_0(X,Y),Z) in V_3, so the Jacobi "
                "identity holds iff a b = 0 (samples with a = 0 or b = 0 pass, with a b != 0 fail)")
def _():
    n = 3
    rng = np.random.default_rng(11)
    ok = True
    A = np.kron(np.diag([1.0, 2.0, 4.0]), G6.astype(float))
    for trial in range(4):
        a, b = rng.normal(size=2) + 1j * rng.normal(size=2)
        brz = make_bracket(n, {(1, 0, 0): a, (2, 1, 1): b})
        br = lambda x, y: from_z(brz(to_z(x, n), to_z(y, n)), n)
        x, y = rng.normal(size=(2, 6 * n))
        ok &= np.allclose(A @ br(x, y), br(A @ x, y) + br(x, A @ y), atol=1e-10)
        X1, Y1, Z2 = (rng.normal(size=3) + 1j * rng.normal(size=3) for _ in range(3))
        X = np.concatenate([X1, np.zeros(6)])
        Y = np.concatenate([Y1, np.zeros(6)])
        Z = np.concatenate([np.zeros(3), Z2, np.zeros(3)])
        expect = np.concatenate([np.zeros(6), a * b * np.cross(np.cross(X1, Y1), Z2)])
        ok &= np.allclose(jacobiator(brz, X, Y, Z), expect, atol=1e-10) and np.abs(expect).max() > 1e-3
        for g in ({(1, 0, 0): a}, {(2, 1, 1): b}):
            brz0 = make_bracket(n, g)
            for _ in range(3):
                Xg, Yg, Zg = (rng.normal(size=9) + 1j * rng.normal(size=9) for _ in range(3))
                ok &= np.allclose(jacobiator(brz0, Xg, Yg, Zg), 0, atol=1e-10)
    return ok


# ================================================================================================ summary
if __name__ == "__main__":
    dt = time.time() - T0
    print(f"    . runtime {dt:.1f} s")
    if FAILED:
        print("FAILED:", " ".join(FAILED))
        sys.exit(1)
    sys.exit(0)
