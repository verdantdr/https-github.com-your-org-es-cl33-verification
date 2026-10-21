#!/usr/bin/env python3
"""Referee check y2 for investigations/emergence/sections/e2_processes.tex ("How one becomes two: processes").

Independent of verify/e2_processes.py (nothing imported or copied).  Methods are chosen to differ from the derivation:
Hom dimensions from sl(2,C) Casimir decompositions and from finite-field ranks, threshold identities at other data
and against total-momentum kinematics, invariant states by exact sympy kernels, Bogoliubov amplitudes by
matrix exponentials at other parameters, citation numbers by reading the label files.

Conventions (tex/s02_arena.tex): e1 = sx.1.1, e2 = sz.sx.1, e3 = sz.sz.sx, e4 = ep.1.1, e5 = sz.ep.1, e6 = sz.sz.ep,
omega = e1...e6, tower E_(k,a) = omega^{(x)(k-1)} (x) e_a (x) 1^{(x)(n-k)}.  Vector-level generators A in so(eta_n)
are lifted by sigma(A) = 1/4 sum A_ab eta_bb E_a E_b.

Exactness: all algebraic matrices have dyadic-rational entries of small size (0, +-1/4, +-1/2, +-1, ...); they are
held in float64, where such numbers and their sums and products are represented exactly, and algebraic statements
are tested by exact equality (== 0).  Ranks of linear systems are computed exactly over the prime fields F_p for two
primes p (integer arithmetic) and compared with a floating SVD rank; sympy is used for exact kernels.  Floating
point with stated tolerances is used only for spectra and exponentials.

Run from any directory: python3 y2_check.py   -> one PASS/FAIL line per claim, exit 0 iff all pass.
"""
import os
import sys
import itertools
from math import comb, cosh, sinh, sqrt, cos, sin, pi, tanh

import numpy as np
import sympy as sp
import scipy.sparse as sps
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
EMERG = os.path.dirname(HERE)
FAIL = []


def rep(cid, ok, text):
    print(f"{'PASS' if ok else 'FAIL'} y2-{cid}: {text}")
    if not ok:
        FAIL.append(cid)


# ------------------------------------------------------------------ realisation
sx = np.array([[0., 1.], [1., 0.]])
sz = np.array([[1., 0.], [0., -1.]])
ep = np.array([[0., -1.], [1., 0.]])
I2 = np.eye(2)


def k3(a, b, c):
    return np.kron(np.kron(a, b), c)


e = [k3(sx, I2, I2), k3(sz, sx, I2), k3(sz, sz, sx), k3(ep, I2, I2), k3(sz, ep, I2), k3(sz, sz, ep)]
eta1 = np.array([1, 1, 1, -1, -1, -1])
om = e[0] @ e[1] @ e[2] @ e[3] @ e[4] @ e[5]


def tower(n):
    gens = []
    for k in range(n):
        for a in range(6):
            M = np.eye(1)
            for j in range(n):
                M = np.kron(M, om if j < k else (e[a] if j == k else np.eye(8)))
            gens.append(M)
    return gens


def tower_sparse(n):
    gens = []
    oms, I8 = sps.csr_matrix(om), sps.identity(8, format='csr')
    for k in range(n):
        for a in range(6):
            M = sps.identity(1, format='csr')
            for j in range(n):
                M = sps.kron(M, oms if j < k else (sps.csr_matrix(e[a]) if j == k else I8), format='csr')
            gens.append(M)
    return gens


def etan(n):
    return np.tile(eta1, n)


def sigma(A, E, et):
    d = E[0].shape[0]
    S = np.zeros((d, d)) if not sps.issparse(E[0]) else sps.csr_matrix((d, d))
    for a, b in zip(*np.nonzero(A)):
        S = S + 0.25 * A[a, b] * et[b] * (E[a] @ E[b])
    return S


def vec_of_biv(a, b, n=1):
    """vector-level matrix of ad(e_a e_b) (0-based indices in V_n): [e_a e_b, e_c] = 2 eta_bc e_a - 2 eta_ac e_b"""
    et = etan(n)
    A = np.zeros((6 * n, 6 * n))
    A[a, b] += 2 * et[b]
    A[b, a] -= 2 * et[a]
    return A


def comm(X, Y):
    return X @ Y - Y @ X


def iszero(X):
    if sps.issparse(X):
        return X.count_nonzero() == 0 or abs(X).max() == 0
    return not np.any(X)


# Fock structure
def fock(E, n):
    ap = [0.5 * (E[6 * k + i] + E[6 * k + i + 3]) for k in range(n) for i in range(3)]
    am = [0.5 * (E[6 * k + i] - E[6 * k + i + 3]) for k in range(n) for i in range(3)]
    Nk = [sum(ap[3 * k + i] @ am[3 * k + i] for i in range(3)) for k in range(n)]
    return ap, am, Nk


# vector-level structures
G1 = np.zeros((6, 6)); K1v = np.zeros((6, 6))
for i in range(3):
    G1[i + 3, i] = 1; G1[i, i + 3] = -1
    K1v[i + 3, i] = 1; K1v[i, i + 3] = 1
epsm = np.array([[0., -1.], [1., 0.]])

# lint generators (vector level), half-bivectors
S_v = [0.5 * vec_of_biv(4, 5), 0.5 * vec_of_biv(5, 3), 0.5 * vec_of_biv(3, 4)]   # S1=e5e6/2, S2=e6e4/2, S3=e4e5/2
T_v = [0.5 * vec_of_biv(0, 3), 0.5 * vec_of_biv(0, 4), 0.5 * vec_of_biv(0, 5)]   # T_a = e1 e_{a+3}/2
J_v = vec_of_biv(1, 2)                                                            # J = e2 e3 (not halved)
# lG boosts K_i = (e_k e_{j+3} - e_j e_{k+3})/2, (i,j,k) cyclic
cyc = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
KG_v = [0.5 * (vec_of_biv(k, j + 3) - vec_of_biv(j, k + 3)) for (i, j, k) in cyc]
LG_v = [-comm(KG_v[j], KG_v[k]) for (i, j, k) in cyc]

E1 = tower(1)
eta_1 = etan(1)


def lift1(A):
    return sigma(A, E1, eta_1)


# ================================================================== 1. conventions
ok = all(np.array_equal(E1[a] @ E1[b] + E1[b] @ E1[a], 2 * eta1[a] * (a == b) * np.eye(8)) for a in range(6) for b in range(6))
ap1, am1, N1 = fock(E1, 1)
ok &= all(np.array_equal(am1[i], ap1[i].T) for i in range(3))
vac = np.zeros(8); vac[0] = 1
ok &= all(not np.any(am1[i] @ vac) for i in range(3))
# |n1 n2 n3> is the basis vector with index 4 n1 + 2 n2 + n3
Nmode1 = [ap1[i] @ am1[i] for i in range(3)]
Ndiag = [np.diag(Nmode1[i]) for i in range(3)]
ok &= all(np.array_equal(Nmode1[i], np.diag(Ndiag[i])) for i in range(3))
ok &= all(Ndiag[i][4 * b[0] + 2 * b[1] + b[2]] == b[i] for b in itertools.product([0, 1], repeat=3) for i in range(3))
E2 = tower(2)
ok &= all(np.array_equal(E2[6 + a] @ E2[6 + b], np.kron(np.eye(8), e[a] @ e[b])) for a in range(6) for b in range(6))
# lifts reproduce Clifford half-bivectors
ok &= np.array_equal(lift1(T_v[1]), 0.5 * e[0] @ e[4]) and np.array_equal(lift1(S_v[0]), 0.5 * e[4] @ e[5])
ok &= np.array_equal(lift1(J_v), e[1] @ e[2])
ok &= all(np.array_equal(lift1(KG_v[i]), 0.5 * (e[k] @ e[j + 3] - e[j] @ e[k + 3])) for (i, j, k) in cyc)
rep(1, ok, "realisation and tower: Clifford relations, a- = (a+)^T, Fock vacuum = first basis vector, |n1n2n3> = basis "
    "vector 4n1+2n2+n3; graded copy-2 bivectors = 1(x)e_a e_b; sigma reproduces T_2, S_1, J, K_i (exact)")

# ================================================================== helpers for Hom dimensions
P1, P2 = 2147483629, 2147483587


def rank_modp(M, p):
    A = np.array(np.rint(M), dtype=np.int64) % p
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = None
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        piv = r + nz[0]
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        others = np.nonzero(A[:, c])[0]
        others = others[others != r]
        if others.size:
            f = A[others, c].copy()
            # row ops in two halves to avoid int64 overflow: f < p < 2^31, A < 2^31
            A[others] = (A[others] - (f[:, None] * A[r][None, :]) % p) % p
        r += 1
        if r == rows:
            break
    return r


def hom_dim(gx, gy):
    """dim of {T: rho_y(X) T = T rho_x(X) for all X}; gx, gy lists of matrices (scaled to integers)."""
    dx, dy = gx[0].shape[0], gy[0].shape[0]
    blocks = [np.kron(Y, np.eye(dx)) - np.kron(np.eye(dy), X.T) for X, Y in zip(gx, gy)]
    M = np.vstack(blocks)
    rs = [rank_modp(M, p) for p in (P1, P2)]
    sv = np.linalg.svd(M, compute_uv=False)
    rf = int(np.sum(sv > 1e-8 * sv[0]))
    return dx * dy - rs[0], (rs[0] == rs[1] == rf)


lint_v = S_v + T_v                      # 6 generators
lint_s = [2 * lift1(A) for A in lint_v]  # integer matrices
lint_vec = [2 * A for A in lint_v]


def diag_spin(gens_list, n):
    """diagonal action on S^{(x)n} of single-copy spin matrices (ungraded tensor sum; equals the bivector lift)"""
    out = []
    for X in gens_list:
        M = np.zeros((8 ** n, 8 ** n))
        for k in range(n):
            T = np.eye(1)
            for j in range(n):
                T = np.kron(T, X if j == k else np.eye(8))
            M += T
        out.append(M)
    return out


# ================================================================== 2. vertices (i)
lint2 = diag_spin(lint_s, 2)
# the diagonal action equals the tower lift of id_2 (x) A
lift_ok = all(np.array_equal(sigma(np.kron(np.eye(2), A), E2, etan(2)) * 2, L) for A, L in zip(lint_v, lint2))
d12, c12 = hom_dim(lint_s, lint2)
d21, c21 = hom_dim(lint2, lint_s)
R3 = expm(2 * pi * lift1(S_v[2]))
rot_ok = np.allclose(R3, -np.eye(8), atol=1e-12)
R3_2 = expm(2 * pi * sigma(np.kron(np.eye(2), S_v[2]), E2, etan(2)))
rot_ok &= np.allclose(R3_2, np.eye(64), atol=1e-11)
rep(2, lift_ok and c12 and c21 and d12 == 0 and d21 == 0 and rot_ok,
    f"diagonal lint at n=2 (tower lift = tensor-sum action, exact): dim Hom(S,S_2) = {d12}, dim Hom(S_2,S) = {d21} "
    "(ranks over F_p for two primes = SVD rank); exp(2 pi S_3) = -1 on S, +1 on S_2 (tol 1e-11)")


# Casimir decomposition: find signs so that (R_k, B_k) satisfy so(1,3) brackets in the A/B form
def sl2_pair(Sg, Tg):
    """return R, B with [R_i,R_j]=eps R_k, [R_i,B_j]=eps B_k, [B_i,B_j]=-eps R_k (vector-level check)."""
    for sr, sb in itertools.product([1, -1], repeat=2):
        R = [sr * X for X in Sg]; B = [sb * X for X in Tg]
        good = True
        for (i, j, k) in cyc:
            good &= np.allclose(comm(R[i], R[j]), R[k]) and np.allclose(comm(R[i], B[j]), B[k]) \
                and np.allclose(comm(B[i], B[j]), -R[k])
        if good:
            return R, B
    return None


RB = sl2_pair(S_v, T_v)
R_int, B_int = RB
sr = 1 if np.allclose(R_int[0], S_v[0]) else -1
sb = 1 if np.allclose(B_int[0], T_v[0]) else -1


def casimirs(Rs, Bs):
    A = [0.5 * (r + 1j * b) for r, b in zip(Rs, Bs)]
    Bp = [0.5 * (r - 1j * b) for r, b in zip(Rs, Bs)]
    return -sum(x @ x for x in A), -sum(x @ x for x in Bp)


def joint_mult(CA, CB, ja, jb):
    """multiplicity of (ja, jb) from the joint eigenspace dimension of two commuting diagonalisable Casimirs."""
    ta, tb = ja * (ja + 1), jb * (jb + 1)
    d = CA.shape[0]
    # joint eigenspace: kernel of (CA - ta)^dagger(CA-ta) + (CB-tb)^dagger(CB-tb)
    M = (CA - ta * np.eye(d)).conj().T @ (CA - ta * np.eye(d)) + (CB - tb * np.eye(d)).conj().T @ (CB - tb * np.eye(d))
    w = np.linalg.eigvalsh(M)
    dim = int(np.sum(w < 1e-8))
    return dim // int(round((2 * ja + 1) * (2 * jb + 1))), dim


R_s = [sr * lift1(X) for X in S_v]; B_s = [sb * lift1(X) for X in T_v]
CA1, CB1 = casimirs(R_s, B_s)
m10, _ = joint_mult(CA1, CB1, 0.5, 0); m01, _ = joint_mult(CA1, CB1, 0, 0.5)
# S_3
R_3 = diag_spin(R_s, 3); B_3 = diag_spin(B_s, 3)
CA3, CB3 = casimirs(R_3, B_3)
n10, _ = joint_mult(CA3, CB3, 0.5, 0); n01, _ = joint_mult(CA3, CB3, 0, 0.5)
R3_3 = expm(2 * pi * diag_spin([lift1(S_v[2])], 3)[0])
rep(3, (m10, m01) == (2, 2) and (n10, n01) == (40, 40) and 2 * n10 + 2 * n01 == 160 and np.allclose(R3_3, -np.eye(512), atol=1e-9),
    f"diagonal lint at n=3 by sl(2,C)xsl(2,C) Casimirs (not weights): S(x)C = {m10}(1/2,0)+{m01}(0,1/2), S_3(x)C contains "
    f"(1/2,0) {n10} and (0,1/2) {n01} times, so dim Hom(S,S_3) = 2*{n10}+2*{n01} = 160; exp(2 pi S_3) = -1 on S_3 (tol 1e-9)")

# ================================================================== 4. vertices (ii), (iii)
Eidx = [0, 3, 4, 5]
Pfour = np.zeros((6, 4))
for c, a in enumerate(Eidx):
    Pfour[a, c] = 1
lintE = [Pfour.T @ X @ Pfour for X in lint_vec]          # action on E (E is invariant)
ok_inv = all(np.allclose(Pfour @ Y, X @ Pfour) for X, Y in zip(lint_vec, lintE))
# Lambda^2 W from vector action
def wedge2(Xs, d):
    pairs = list(itertools.combinations(range(d), 2))
    idx = {p: t for t, p in enumerate(pairs)}
    out = []
    for X in Xs:
        M = np.zeros((len(pairs), len(pairs)))
        for (a, b), t in idx.items():
            # X(e_a ^ e_b) = X e_a ^ e_b + e_a ^ X e_b
            for c in range(d):
                for (u, v, s) in ((c, b, X[c, a]), (a, c, X[c, b])):
                    if s == 0 or u == v:
                        continue
                    if u < v:
                        M[idx[(u, v)], t] += s
                    else:
                        M[idx[(v, u)], t] -= s
        out.append(M)
    return out


def tensor_act(A, B):
    return [np.kron(a, np.eye(b.shape[0])) + np.kron(np.eye(a.shape[0]), b) for a, b in zip(A, B)]


L2E = wedge2(lintE, 4); L2V = wedge2(lint_vec, 6)
dims = {}
cons = True
for name, W in (("End(S)", lint_s), ("S->S(x)E", tensor_act(lint_s, lintE)), ("S->S(x)V", tensor_act(lint_s, lint_vec)),
                ("S->S(x)L2E", tensor_act(lint_s, L2E)), ("S->S(x)L2V", tensor_act(lint_s, L2V))):
    d, c = hom_dim(lint_s, W); dims[name] = d; cons &= c
rep(4, ok_inv and cons and list(dims.values()) == [8, 8, 24, 8, 32],
    f"lint-equivariant maps by direct kernels (F_p ranks = SVD rank): {dims}")


# Clifford multiplications
def c1_matrix(with_eta=True):
    # psi -> sum_mu eta_mumu gamma_mu psi (x) e_mu ; output index (s, mu) row-major in S (x) E
    M = np.zeros((8 * 4, 8))
    for c, a in enumerate(Eidx):
        g = (eta1[a] if with_eta else 1) * e[a]
        for s in range(8):
            M[s * 4 + c, :] += g[s, :]
    return M


def c2_matrix():
    pairs = list(itertools.combinations(range(4), 2))
    M = np.zeros((8 * 6, 8))
    for t, (u, v) in enumerate(pairs):
        a, b = Eidx[u], Eidx[v]
        g = eta1[a] * eta1[b] * e[a] @ e[b]
        for s in range(8):
            M[s * 6 + t, :] += g[s, :]
    return M


def is_equiv(T, gx, gy):
    return all(not np.any(Y @ T - T @ X) for X, Y in zip(gx, gy))


WE = tensor_act(lint_s, lintE); WL = tensor_act(lint_s, L2E)
C1, C1n, C2 = c1_matrix(True), c1_matrix(False), c2_matrix()
eq1, eq1n, eq2 = is_equiv(C1, lint_s, WE), is_equiv(C1n, lint_s, WE), is_equiv(C2, lint_s, WL)
# basis of End_lint(S) via exact sympy kernel
Mend = np.vstack([np.kron(X, np.eye(8)) - np.kron(np.eye(8), X.T) for X in lint_s])
ker = sp.Matrix(np.rint(Mend).astype(int).tolist()).nullspace()
Ends = [np.array(v, dtype=float).reshape(8, 8) for v in ker]
r1 = np.linalg.matrix_rank(np.array([(C1 @ T).ravel() for T in Ends]))
r2 = np.linalg.matrix_rank(np.array([(C2 @ T).ravel() for T in Ends]))
rep(5, eq1 and eq2 and not eq1n and len(Ends) == 8 and r1 == 8 and r2 == 8,
    "c1 = sum eta_mumu gamma_mu (x) e_mu and c2 (bivector Clifford multiplication) are lint-equivariant, c1 without eta is not; "
    "c1 o End_lint(S) and c2 o End_lint(S) have rank 8 = dim Hom (exact equivariance; exact sympy kernel)")

# ================================================================== 6. vertices (iv): spectator, mixed, lG Casimir
spec_act = [np.kron(X, np.eye(8)) for X in lint_s]
dsp, csp = hom_dim(lint_s, spec_act)
# isomorphism lint -> lG: find signs/permutation so that brackets match
RBG = sl2_pair(LG_v, KG_v)
lg_ok = RBG is not None
RG, BG = RBG
# lint generators in the same normal form: R_int (rotations), B_int (boosts); map R_int[k] -> RG[k], B_int[k] -> BG[k]
# spin lifts of lG on S (Clifford), consistent with vector map
RGs = [lift1(X) for X in RG]; BGs = [lift1(X) for X in BG]
lint_norm_s = [lift1(X) for X in R_int] + [lift1(X) for X in B_int]
lG_norm_s = RGs + BGs
lG_norm_vec = list(RG) + list(BG)
mixed = [np.kron(X, np.eye(8)) + np.kron(np.eye(8), Y) for X, Y in zip(lint_norm_s, lG_norm_s)]
gx = [2 * X for X in lint_norm_s]
dmix, cmix = hom_dim(gx, [2 * M for M in mixed])
dmixV, cmixV = hom_dim(gx, [2 * M for M in tensor_act(lint_norm_s, lG_norm_vec)])
casG = sum(lift1(K) @ lift1(K) for K in KG_v) - sum(lift1(L) @ lift1(L) for L in LG_v)
# vector Casimir of lG on V for comparison: 4 (field strength), and lint on E: 3
casGV = sum(K @ K for K in KG_v) - sum(L @ L for L in LG_v)
L3s = lift1(LG_v[2])
rep(6, lg_ok and csp and cmix and cmixV and dsp == 64 and dmix == 16 and dmixV == 8 and np.array_equal(casG, 3 * np.eye(8))
    and np.array_equal(casGV, 4 * np.eye(6)) and np.allclose(expm(2 * pi * L3s), np.eye(8), atol=1e-12),
    f"spectator dim Hom(S,S_2) = {dsp}; mixed (lint on 1, lG on 2, via a bracket-preserving identification) "
    f"dim Hom(S,S_2) = {dmix}, dim Hom(S,S(x)V^(2)) = {dmixV}; lG Casimir sum K^2 - sum L^2 = 3 on S, 4 on V; "
    "exp(2 pi L_3) = +1 on S")

# ================================================================== 7-9. thresholds
E2g = E2


def Hg_mat(m1, m2, p, q):
    H = m1 * E2g[0] + m2 * E2g[6]
    for a in range(3):
        H = H + p[a] * E2g[0] @ E2g[3 + a] + q[a] * E2g[6] @ E2g[9 + a]
    return H


def H1(m, p):
    return m * e[0] + sum(p[a] * e[0] @ e[3 + a] for a in range(3))


exact_ok = True
for (m1, m2, p, q) in [(2, 5, (1, -3, 2), (4, 0, -1)), (0, 3, (2, 2, 1), (1, -1, 0)), (7, 1, (0, 0, 0), (3, 4, 0)),
                       (1, 1, (1, 2, 2), (-2, 1, 0)), (3, 0, (0, 1, 0), (0, 0, 0))]:
    H = Hg_mat(m1, m2, p, q)
    pp, qq = sum(x * x for x in p), sum(x * x for x in q)
    a = m1 ** 2 + m2 ** 2 + pp + qq
    b = m1 ** 2 * qq + m2 ** 2 * pp + pp * qq
    X = H @ H - a * np.eye(64)
    W = E2g[0] @ E2g[6]
    exact_ok &= np.array_equal(X @ X, 4 * b * np.eye(64)) and np.trace(X) == 0 and np.array_equal(W @ H, -H @ W)
rng = np.random.default_rng(20261020)
spec_ok = True
for _ in range(15):
    m1, m2 = rng.uniform(0.1, 3, 2)
    p, q = rng.normal(size=3), rng.normal(size=3)
    E1_, E2_ = sqrt(m1 ** 2 + p @ p), sqrt(m2 ** 2 + q @ q)
    b = E1_ ** 2 * E2_ ** 2 - m1 ** 2 * m2 ** 2
    Ep, Em = sqrt(E1_ ** 2 + E2_ ** 2 + 2 * sqrt(b)), sqrt(E1_ ** 2 + E2_ ** 2 - 2 * sqrt(b))
    w = np.sort(np.linalg.eigvalsh(Hg_mat(m1, m2, p, q)))
    target = np.sort(np.repeat([-Ep, -Em, Em, Ep], 16))
    spec_ok &= np.allclose(w, target, atol=1e-10)
    Hu = np.kron(H1(m1, p), np.eye(8)) + np.kron(np.eye(8), H1(m2, q))
    wu = np.sort(np.linalg.eigvalsh(Hu))
    spec_ok &= np.allclose(wu, np.sort(np.repeat([E1_ + E2_, E1_ - E2_, -E1_ + E2_, -E1_ - E2_], 16)), atol=1e-10)
    spec_ok &= max(E1_, E2_) <= Ep + 1e-12 and Ep < E1_ + E2_
rep(7, exact_ok and spec_ok,
    "thresholds: (H_g^2-a)^2 = 4b, X traceless, E_(1,1)E_(2,1) anticommutes with H_g at five other integer data (incl. "
    "m1 m2 = 0), exact; spec H_g = {+-E_+,+-E_-} (16 each), spec H_u = {+-E1+-E2} (16 each), max(E1,E2) <= E_+ < E1+E2 "
    "on 15 random data (tol 1e-10)")


def Eplus(m1, m2, p, q):
    E1_, E2_ = sqrt(m1 ** 2 + p @ p), sqrt(m2 ** 2 + q @ q)
    return sqrt(E1_ ** 2 + E2_ ** 2 + 2 * sqrt(E1_ ** 2 * E2_ ** 2 - m1 ** 2 * m2 ** 2))


# (refutation) dissociation: compare with the free-pair threshold at the SAME TOTAL momentum
m = 1.0
p = np.array([0, 0, 2.0])
Ep_rel = np.max(np.linalg.eigvalsh(Hg_mat(m, m, p, -p)))
above = Ep_rel > 2 * m + 1  # 4.45 > 2: composite level above the free-pair threshold at P = 0
# crossover at |p| = m/2 for equal masses, P = 0
cross = abs(Eplus(m, m, np.array([0, 0, 0.5]), np.array([0, 0, -0.5])) - 2 * m) < 1e-12
below_small = Eplus(m, m, np.array([0, 0, 0.4]), np.array([0, 0, -0.4])) < 2 * m
# lowest composite level at each total momentum lies below the threshold sqrt((m1+m2)^2+P^2)
low_ok = True
for _ in range(20):
    m1, m2 = rng.uniform(0.2, 2, 2)
    P = rng.normal(size=3)
    pstar = m1 / (m1 + m2) * P
    low_ok &= Eplus(m1, m2, pstar, P - pstar) < sqrt((m1 + m2) ** 2 + P @ P)
rep(8, above and cross and below_small and low_ok,
    f"(corrects the dissociation reading) at m1=m2=1 and p=-q=(0,0,2) (total momentum 0) the top graded level is "
    f"E_+ = {Ep_rel:.4f} > m1+m2 = 2, above the free-pair threshold at that total momentum; at P=0 the crossover is "
    "|p| = m/2 (exact equality); only the lowest composite level at each total momentum P lies below sqrt((m1+m2)^2+P^2) "
    "(20 random samples)")
# deficit is not proportional to m1 m2
defs = [(mm, 2 * mm - sqrt(2) * mm) for mm in (0.5, 1, 2, 4)]
lin = all(abs(d / mm - (2 - sqrt(2))) < 1e-14 for mm, d in defs)
notprod = abs(defs[2][1] / 4 - defs[1][1] / 1) > 0.1
sqdef = all(abs((m1 + m2) ** 2 - (m1 ** 2 + m2 ** 2) - 2 * m1 * m2) < 1e-12 for m1, m2 in [(1, 2), (0.3, 5), (2, 2)])
rep(9, lin and notprod and sqdef,
    "(corrects the abstract of the section) the rest deficit m1+m2-sqrt(m1^2+m2^2) is not proportional to m1 m2: at m1=m2=m it "
    "is (2-sqrt2) m, linear in m; what is proportional to m1 m2 is the difference of squares (m1+m2)^2 - M^2 = 2 m1 m2")

# ================================================================== 10. fields (restated)
okf = True
for _ in range(10):
    u = rng.normal(size=3); v = rng.normal(size=3)
    v = v - (u @ v) / (u @ u) * u; v *= np.linalg.norm(u) / np.linalg.norm(v)
    z = u + 1j * v                     # null: z.z = |u|^2-|v|^2 + 2i u.v = 0
    al, be = rng.normal(size=2) + 1j * rng.normal(size=2)
    Z = np.array([al * z, be * z])
    Q = Z @ Z.T
    okf &= abs(z @ z) < 1e-12 and np.allclose(Q, 0, atol=1e-12)
    u2 = rng.normal(size=3); v2 = rng.normal(size=3)
    v2 = v2 - (u2 @ v2) / (u2 @ u2) * u2; v2 *= np.linalg.norm(u2) / np.linalg.norm(v2)
    z2 = u2 + 1j * v2
    okf &= abs(z @ z2) > 1e-6       # non-collinear null pair: z1.z2 != 0
    sv = np.linalg.svd(np.array([[0, z @ z2], [z @ z2, 0]]), compute_uv=False)
    okf &= abs(0.5 * sv.sum() - abs(z @ z2)) < 1e-12
rep(10, okf, "restated fields: duplicates (alpha z, beta z) of a null z have Gram matrix 0; two random non-collinear null "
    "fields have z1.z2 != 0 and (sigma1+sigma2)/2 = |z1.z2| (tol 1e-12)")

# ================================================================== 11. number operator
okN = True
for n in (1, 2, 3):
    E = tower(n) if n < 3 else tower(3)
    et = etan(n)
    Kn = np.kron(np.eye(n), K1v)
    ap, am, Nk = fock(E, n)
    N = sum(Nk)
    okN &= np.array_equal(N, 1.5 * n * np.eye(8 ** n) + sigma(Kn, E, et))
    okN &= all(np.array_equal(comm(N, E[c]), sum(Kn[a, c] * E[a] for a in range(6 * n))) for c in range(6 * n))
    if n == 1:
        par = np.diag((-1.0) ** np.rint(np.diag(N)))
        okN &= np.array_equal(par, -om)
    if n == 2:
        par = np.diag((-1.0) ** np.rint(np.diag(N)))
        vol = np.eye(64)
        for Ea in E:
            vol = vol @ Ea
        okN &= np.array_equal(par, vol)
        okN &= all(np.array_equal(par @ E[a] @ E[b], E[a] @ E[b] @ par) for a in range(12) for b in range(12))
    del E
rep(11, okN, "N = 3n/2 + sigma(K_n) and [N, E_c] = K_n E_c at n=1,2,3; (-1)^N = -omega at n=1 and = +E_1...E_12 at n=2, "
    "commuting with every bivector (exact)")


# Cent/Anti dimensions by an exact basis count (vector level)
def so_basis(n):
    et = etan(n); d = 6 * n
    B = []
    for a in range(d):
        for b in range(a + 1, d):
            B.append(vec_of_biv(a, b, n) / 2)
    return B


dimok = True
dl = []
for n in (1, 2, 3, 4):
    Kn = np.kron(np.eye(n), K1v)
    B = so_basis(n)
    Mc = np.array([(comm(Kn, X)).ravel() for X in B]).T
    Ma = np.array([(Kn @ X + X @ Kn).ravel() for X in B]).T
    dc = len(B) - rank_modp(2 * Mc, P1)
    da = len(B) - rank_modp(2 * Ma, P1)
    dl.append((dc, da))
    dimok &= dc == 9 * n * n and da == 9 * n * n - 3 * n
# Anti(K) lift changes N by exactly +-2 at n=2: random integer element of Anti(K_2)
Kn = np.kron(np.eye(2), K1v)
B = so_basis(2)
ap2, am2, Nk2 = fock(E2, 2)
N2 = sum(Nk2)
Nd = np.rint(np.diag(N2)).astype(int)
X = sum(rng.integers(-3, 4) * Bx for Bx in B)
Xa = 0.5 * (X - Kn @ X @ Kn)     # Anti part (K^{-1} = K)
Xc = 0.5 * (X + Kn @ X @ Kn)
Sa = sigma(Xa, E2, etan(2)); Sc = sigma(Xc, E2, etan(2))
shift_ok = all((Sa[i, j] == 0) or abs(Nd[i] - Nd[j]) == 2 for i in range(64) for j in range(64))
shift_ok &= all((Sc[i, j] == 0) or Nd[i] == Nd[j] for i in range(64) for j in range(64))
shift_ok &= np.any(Sa) and np.any(Sc)
rep(12, dimok and shift_ok, f"(dim Cent(K_n), dim Anti(K_n)) = {dl} for n=1..4 = (9n^2, 9n^2-3n) (exact F_p rank); the lift "
    "of the Anti part of a random integer element changes N by exactly +-2 and the Cent part conserves N (n=2, exact)")

# ================================================================== 13. lemma (iii) and its hypothesis
# compact = block diagonal in (temporal, spatial) basis; commuting with N iff A = D, iff commuting with G
okc = True
for n in (1, 2):
    d3 = 3 * n
    perm = [6 * k + i for k in range(n) for i in range(3)] + [6 * k + i + 3 for k in range(n) for i in range(3)]
    P = np.eye(6 * n)[:, perm]           # columns: temporal then spatial
    Kn = np.kron(np.eye(n), K1v); Gn = np.kron(np.eye(n), G1)
    for trial in range(6):
        A = rng.integers(-2, 3, (d3, d3)); A = A - A.T
        D = A.copy() if trial % 2 == 0 else (lambda M: M - M.T)(rng.integers(-2, 3, (d3, d3)))
        Xb = np.zeros((6 * n, 6 * n)); Xb[:d3, :d3] = A; Xb[d3:, d3:] = D
        Xv = P @ Xb @ P.T
        cK = not np.any(comm(Xv, Kn)); cG = not np.any(comm(Xv, Gn))
        okc &= (cK == cG == np.array_equal(A, D))
# elliptic element in Cent(K) outside the maximal compact subalgebra (n=1): X = [[A,B],[B,A]], A rotation, B symmetric
A = np.zeros((3, 3)); A[0, 1], A[1, 0] = -1, 1
Bm = np.diag([0.5, -0.5, 0])
Xe = np.block([[A, Bm], [Bm, A]])
eta_ok = not np.any(np.diag(eta1) @ Xe + Xe.T @ np.diag(eta1))
evs = np.linalg.eigvals(Xe)
w0 = sqrt(1 - 0.25)
periodic = np.allclose(expm(2 * pi / w0 * Xe), np.eye(6), atol=1e-10)
commK = not np.any(comm(Xe, K1v)); commG = not np.any(comm(Xe, G1))
Ne = sum(fock(E1, 1)[2]); commN = not np.any(comm(lift1(Xe), Ne))
rep(13, okc and eta_ok and periodic and commK and commN and not commG and not np.allclose(Xe, -Xe.T),
    "Lemma (iii) holds for X in the maximal compact subalgebra so(3n)+so(3n) (12 random block-diagonal elements, n=1,2, "
    "exact); it needs that meaning of 'compact': X = [[A,B],[B,A]] with A = rotation of (e1,e2), B = diag(1/2,-1/2,0) lies in "
    "so(3,3), generates a periodic one-parameter group (period 2 pi/sqrt(3/4), tol 1e-10), commutes with K and with N, and "
    "does not commute with G (exact)")

# lemma (iv) and its hypothesis
okiv = True
vac2 = np.zeros(64); vac2[0] = 1
Kn = np.kron(np.eye(2), K1v)
for trial in range(10):
    X = sum(rng.integers(-2, 3) * Bx for Bx in B)
    Xs = 0.5 * (X - X.T) if trial % 2 == 0 else 0.5 * (X + X.T)
    if not np.any(Xs):
        continue
    v = sigma(Xs, E2, etan(2)) @ vac2
    has2 = np.any(v[Nd == 2])
    okiv &= has2 == bool(np.any(comm(Xs, Kn)))
# hypothesis is needed: a pure lowering element (Lambda^2 L_-) is neither, is outside Cent(K), and has no N=2 component
Lm = [np.kron(np.eye(2)[k], (np.eye(6)[i] - np.eye(6)[i + 3])) for k in range(2) for i in range(3)]
u, w = Lm[0], Lm[4]
etaM = np.diag(etan(2))
Xlow = np.outer(u, w) @ etaM - np.outer(w, u) @ etaM       # x -> u eta(w,x) - w eta(u,x), in so(eta)
inso = not np.any(etaM @ Xlow + Xlow.T @ etaM)
vlow = sigma(Xlow, E2, etan(2)) @ vac2
rep(14, okiv and inso and np.any(comm(Xlow, Kn)) and not np.any(vlow[Nd == 2]),
    "Lemma (iv): for random antisymmetric and symmetric X at n=2, sigma(X)|0> has an N=2 component iff [X,K_2] != 0 (exact); "
    "the hypothesis is needed: the lowering element u^w (u,w in L_-) lies outside Cent(K_2) and sigma of it gives no N=2 component")

# ================================================================== 15. classification of generators (e2-13)
def cent_dim(gens_v, Kv):
    """dim of span(gens) cap Cent(K) via exact kernel of the commutator map"""
    M = np.array([comm(X, Kv).ravel() for X in gens_v]).T
    return len(gens_v) - np.linalg.matrix_rank(M)


lint_cent = cent_dim(lint_v, K1v)
T1_in = not np.any(comm(T_v[0], K1v))
other_pair = all(np.any(comm(X, K1v)) for X in [S_v[0], S_v[1], S_v[2], T_v[1], T_v[2], J_v])
lG_cent = cent_dim(LG_v + KG_v, K1v)
L_in = all(not np.any(comm(L, K1v)) for L in LG_v)
K_anti = all(not np.any(Kx @ K1v + K1v @ Kx) for Kx in KG_v)
K2 = np.kron(np.eye(2), K1v)
crot = np.kron(epsm, np.eye(6)); cboost = np.kron(epsm, G1)
copy_ok = not np.any(comm(crot, K2)) and not np.any(cboost @ K2 + K2 @ cboost)
rep(15, lint_cent == 1 and T1_in and other_pair and lG_cent == 3 and L_in and K_anti and copy_ok,
    "lint cap Cent(K) = R T_1 (dim 1); S_1,S_2,S_3,T_2,T_3 and J do not commute with K; lG cap Cent(K) = span L_k (dim 3), "
    "the K_i anticommute with K; copy rotation in Cent(K_2), copy boost in Anti(K_2) (exact)")

# ================================================================== 16. Bogoliubov amplitudes
def ket(b):
    v = np.zeros(8); v[4 * b[0] + 2 * b[1] + b[2]] = 1; return v


k000, k110, k011 = ket((0, 0, 0)), ket((1, 1, 0)), ket((0, 1, 1))
okb = True
T2s, K1s, S1s, Js = lift1(T_v[1]), lift1(KG_v[0]), lift1(S_v[0]), lift1(J_v)
evT = np.sort(np.linalg.eigvals(T_v[1]).real); evK = np.sort(np.linalg.eigvals(KG_v[0]).real)
okb &= np.allclose(evT, [-1, 0, 0, 0, 0, 1]) and np.allclose(evK, [-1, -1, 0, 0, 1, 1])
for z in (0.7, -1.4, 3.1):
    U = expm(z * T2s)
    okb &= np.allclose(U @ k000, cosh(z / 2) * k000 + sinh(z / 2) * k110, atol=1e-12)
    okb &= abs(np.linalg.norm(U @ k000) ** 2 - cosh(z)) < 1e-11 and np.allclose(U, U.T) and not np.allclose(U.T @ U, np.eye(8))
    V = expm(z * K1s)
    okb &= np.allclose(V @ k000, cosh(z) * k000 - sinh(z) * k011, atol=1e-11)
for t in (0.9, 2.2, pi):
    okb &= np.allclose(expm(t * S1s) @ k000, cos(t / 2) * k000 + sin(t / 2) * k011, atol=1e-12)
    okb &= np.allclose(expm(t * Js) @ k000, cos(t) * k000 + sin(t) * k011, atol=1e-12)
rep(16, okb, "Bogoliubov amplitudes at other parameters (zeta = 0.7,-1.4,3.1; theta = 0.9,2.2,pi; tol 1e-11): "
    "e^{zeta T_2}|000> = cosh(zeta/2)|000>+sinh(zeta/2)|110>, norm^2 cosh zeta, symmetric, not orthogonal; "
    "e^{zeta K_1}|000> = cosh zeta|000> - sinh zeta|011>; e^{theta S_1}, e^{theta J} as stated; ad T_2, ad K_1 eigenvalues as stated")

# Pauli bound refutation: unitary (compact) generator reaches pair probability 1; boost pair probability < 1/2
pS = (expm(pi * S1s) @ k000)[np.argmax(k011)] ** 2
pairs_boost = [tanh(z / 2) ** 2 / (1 + tanh(z / 2) ** 2) for z in (1, 5, 30)]
rep(17, abs(pS - 1) < 1e-12 and all(x < 0.5 for x in pairs_boost),
    f"(corrects the 'Pauli bound') the normalised pair probability of a boost, tanh^2/(1+tanh^2), stays below 1/2 "
    f"(values {', '.join(f'{x:.4f}' for x in pairs_boost)}), but under the unitary rotation e^{{pi S_1}} the pair probability "
    "is 1; the bound 1/2 is the large-rapidity limit cosh ~ sinh of the non-unitary boost, not a Pauli bound, which is 1 per "
    "mode pair")

# ================================================================== 18. copy boost
et2 = etan(2)
sb = sigma(cboost, E2, et2)
lift_cb = all(np.array_equal(comm(sb, E2[c]), sum(cboost[a, c] * E2[a] for a in range(12))) for c in range(12))
form = -sum(ap2[i] @ ap2[3 + i] + am2[3 + i] @ am2[i] for i in range(3))
okcb = lift_cb and np.array_equal(sb, form) and np.allclose(sb, sb.T)
for z in (0.45, -0.8, 1.7):
    psi = expm(z * sb) @ vac2
    okcb &= abs(psi[0] - cosh(z) ** 3) < 1e-10 * cosh(z) ** 3
    for k in range(4):
        wk = np.sum(psi[Nd == 2 * k] ** 2)
        okcb &= abs(wk - comb(3, k) * cosh(z) ** (2 * (3 - k)) * sinh(z) ** (2 * k)) < 1e-9 * cosh(2 * z) ** 3
    okcb &= np.sum(psi[Nd % 2 == 1] ** 2) < 1e-20
    N1d, N2d = np.diag(Nk2[0]), np.diag(Nk2[1])
    okcb &= np.sum(psi[np.abs(N1d - N2d) > 0.5] ** 2) < 1e-20
rep(18, okcb, "copy boost: sigma(eps(x)G) lifts eps(x)G on all 12 generators and equals -sum_i(a+_(1,i)a+_(2,i)+a-_(2,i)a-_(1,i)), "
    "symmetric (exact); at zeta = 0.45,-0.8,1.7: <0|U|0> = cosh^3, weight in N=2k = C(3,k)cosh^{2(3-k)}sinh^{2k}, none in odd N, "
    "N^(1) = N^(2) on U|0> (relative tol 1e-9)")

# ================================================================== 19. Lorentz-invariant states
def exact_inv_projector(gens):
    M = sp.Matrix(np.vstack([np.rint(2 * g).astype(int) for g in gens]).tolist())
    ns = M.nullspace()
    if not ns:
        return 0, None
    Bm = sp.Matrix.hstack(*ns)
    P = Bm * (Bm.T * Bm).inv() * Bm.T
    return len(ns), P


# n=1: none
d1, _ = exact_inv_projector(lint_s)
lintdiag2 = [sigma(np.kron(np.eye(2), A), E2, et2) for A in lint_v]
d2, Pinv = exact_inv_projector(lintdiag2)
weights = [sum(Pinv[i, i] for i in range(64) if Nd[i] == k) for k in range(7)]
# states of definite N inside the invariant space: for each k, dim(Inv cap H_k) = rank(P restricted ...) via kernel
defN = {}
for k in range(7):
    idx = [i for i in range(64) if Nd[i] == k]
    other = [i for i in range(64) if Nd[i] != k]
    # v invariant with support in H_k: solve generators v = 0 on columns idx
    M = sp.Matrix(np.vstack([np.rint(2 * g)[:, idx].astype(int) for g in lintdiag2]).tolist())
    defN[k] = len(M.nullspace())
idx3 = [i for i in range(64) if Nd[i] == 3]
M3 = sp.Matrix(np.vstack([np.rint(2 * g)[:, idx3].astype(int) for g in lintdiag2]).tolist())
ns3 = M3.nullspace()
modes_ok = True
for v in ns3:
    full = np.zeros(64)
    full[idx3] = np.array(v, dtype=float).ravel()
    # occupation numbers of mode i in constituents 1 and 2
    for i in range(3):
        Ni1 = ap2[i] @ am2[i]; Ni2 = ap2[3 + i] @ am2[3 + i]
        modes_ok &= np.allclose((Ni1 + Ni2) @ full, full)
# spectator: none
spect = [np.kron(lift1(A), np.eye(8)) for A in lint_v]
dsp_inv, _ = exact_inv_projector(spect)
# n=1 projectors
Nd1 = np.rint(np.diag(sum(N1))).astype(int)
par_ok = np.array_equal(0.5 * (np.eye(8) + om), np.diag((Nd1 % 2 == 1).astype(float))) and \
    np.array_equal(0.5 * (np.eye(8) - om), np.diag((Nd1 % 2 == 0).astype(float)))
Pmode = sum(e[i] @ e[i + 3] for i in range(3))
Pi_pm = 0.5 * np.eye(8) - 0.25 * (Pmode + om)
pm_ok = np.array_equal(Pi_pm, np.diag((Nd1 >= 2).astype(float)))
# Pi_0 of Thm 26.6
Hb = E2[1] @ E2[2] @ E2[6] @ E2[7]
Pi0 = 0.5 * (np.eye(64) + Hb)
par2 = np.diag((-1.0) ** Nd)
pi0_ok = np.array_equal(Pi0 @ par2, par2 @ Pi0) and np.any(comm(Pi0, N2))
pi0_ok &= all(abs(sum(Pi0[i, i] for i in range(64) if Nd[i] == k) - 0.5 * comb(6, k)) == 0 for k in range(7))
rep(19, d1 == 0 and d2 == 8 and [sp.nsimplify(w) for w in weights] == [0, sp.Rational(1, 2), 2, 3, 2, sp.Rational(1, 2), 0]
    and defN == {0: 0, 1: 0, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0} and modes_ok and dsp_inv == 0 and par_ok and pm_ok and pi0_ok,
    f"invariant states (exact sympy kernels): none at n=1; dim 8 at n=2 (diagonal lint), weights {[str(w) for w in weights]} "
    f"in N=0..6; invariants of definite N: {defN} with N_i^(1)+N_i^(2) = 1; none under the spectator action; (1+omega)/2, "
    "(1-omega)/2 = odd, even N; paired-mode projector = N >= 2; Pi_0 commutes with (-1)^N, not with N, trace half of each sector")

# ================================================================== 20. invariant number
a_, b_ = sp.symbols('a b', real=True)
Xs = a_ * sp.Matrix(np.kron(epsm, np.eye(6)).astype(int).tolist()) + b_ * sp.Matrix(np.kron(epsm, G1).astype(int).tolist())
sol = sp.solve(list(Xs * Xs - sp.eye(12)), [a_, b_], dict=True)
sols = sorted([(s[a_], s[b_]) for s in sol])
# commutant of diagonal lG at n=2 (exact)
def commutant_dim(gens_v, n):
    B = so_basis(n)
    rows = []
    for g in gens_v:
        rows.append(np.array([comm(g, X).ravel() for X in B]).T)
    M = np.vstack(rows)
    return len(B) - rank_modp(np.rint(4 * M), P1)


lGdiag2 = [np.kron(np.eye(2), X) for X in LG_v + KG_v]
c2dim = commutant_dim(lGdiag2, 2)
Vp_null = True
Xk = np.kron(epsm, G1)
for sgn in (1, -1):
    # eigenvectors: (x, sgn G x)
    basisV = np.array([np.concatenate([np.eye(6)[i], sgn * G1 @ np.eye(6)[i]]) for i in range(6)]).T
    Vp_null &= np.array_equal(Xk @ basisV, sgn * basisV) and not np.any(basisV.T @ np.diag(et2) @ basisV)
    Vp_null &= all(abs(np.linalg.norm(basisV[:6, i]) - np.linalg.norm(basisV[6:, i])) == 0 for i in range(6))
NKs = 3 * np.eye(64) + sigma(Xk, E2, et2)
evNK = np.rint(np.linalg.eigvalsh(NKs)).astype(int)
mult_ok = all(np.sum(evNK == k) == comb(6, k) for k in range(7)) and np.allclose(np.linalg.eigvalsh(NKs), evNK, atol=1e-10)
lGdiag2_s = [sigma(X, E2, et2) for X in lGdiag2]
inv_ok = all(not np.any(comm(NKs, g)) for g in lGdiag2_s)
NK_N1 = np.any(comm(NKs, Nk2[0]))
rep(20, sols == [(0, -1), (0, 1)] and c2dim == 2 and Vp_null and mult_ok and inv_ok and NK_N1,
    "n=2, diagonal lG: commutant in so(6,6) has dim 2 (exact F_p), (a eps(x)1 + b eps(x)G)^2 = 1 iff (a,b) = (0,+-1) "
    "(sympy); V_pm = {X2 = +-G X1} totally null with equal Euclidean lengths in the two constituents; N_K = 3 + sigma(eps(x)G) "
    "has spectrum 0..6 with binomial multiplicities, commutes with the lifted diagonal lG, and not with N^(1)")

# n=3: commutant of diagonal lG is so(3,C) and has no involution; (0,2,1) admits (eps(x)G)+K
lGdiag3 = [np.kron(np.eye(3), X) for X in LG_v + KG_v]
c3dim = commutant_dim(lGdiag3, 3)
Bs = [np.zeros((3, 3)) for _ in range(3)]
Bs[0][0, 1], Bs[0][1, 0] = -1, 1; Bs[1][1, 2], Bs[1][2, 1] = -1, 1; Bs[2][0, 2], Bs[2][2, 0] = -1, 1
fam = [np.kron(Bq, np.eye(6)) for Bq in Bs] + [np.kron(Bq, G1) for Bq in Bs]
fam_comm = all(not np.any(comm(F, g)) for F in fam for g in lGdiag3)
sing_ok = True
cs = sp.symbols('c0:6')
Xsym = sum((cs[i] * sp.Matrix(fam[i].astype(int).tolist()) for i in range(6)), sp.zeros(18, 18))
# each such X acts on z in C^3 (x) C^3 as (A+iB) (x) 1_3 with A+iB complex antisymmetric 3x3: det = 0 identically
Acx = sp.Matrix([[0, -cs[0] - sp.I * cs[3], -cs[2] - sp.I * cs[5]], [cs[0] + sp.I * cs[3], 0, -cs[1] - sp.I * cs[4]],
                 [cs[2] + sp.I * cs[5], cs[1] + sp.I * cs[4], 0]])
sing_ok &= sp.expand(Acx.det()) == 0
for _ in range(3):
    vals = {c: int(rng.integers(-3, 4)) for c in cs}
    sing_ok &= sp.Matrix(Xsym.subs(vals)).rank() <= 12
X021 = np.zeros((18, 18)); X021[:12, :12] = np.kron(epsm, G1); X021[12:, 12:] = K1v
lG021 = [np.kron(np.diag([1, 1, 0]), X) for X in LG_v + KG_v]
inv021 = np.array_equal(X021 @ X021, np.eye(18)) and all(not np.any(comm(X021, g)) for g in lG021) and \
    not np.any(np.diag(etan(3)) @ X021 + X021.T @ np.diag(etan(3)))
rep(21, c3dim == 6 and fam_comm and sing_ok and inv021,
    "n=3: commutant of the diagonal lG has dim 6 = {A(x)1 + B(x)G} (exact F_p), every element singular (det of the complex "
    "antisymmetric copy matrix vanishes identically; integer ranks <= 12), so no involution; under (0,2,1) "
    "(eps(x)G)+K is an eta-antisymmetric involution commuting with the action (exact)")

# other actions at n=2
def action_gens(kind):
    """vector-level generators of the five actions at n=2: per constituent 'I' (lint), 'B' (lG), 'S' (trivial)"""
    out = []
    lintn = R_int + B_int
    lGn = list(RG) + list(BG)
    for t in range(6):
        blocks = []
        for c in kind:
            blocks.append(lintn[t] if c == 'I' else (lGn[t] if c == 'B' else np.zeros((6, 6))))
        M = np.zeros((12, 12)); M[:6, :6] = blocks[0]; M[6:, 6:] = blocks[1]
        out.append(M)
    return out


def commutant_basis(gens_v, n):
    B = so_basis(n)
    M = np.vstack([np.array([comm(g, X).ravel() for X in B]).T for g in gens_v])
    ns = sp.Matrix(np.rint(4 * M).astype(int).tolist()).nullspace()
    return [sum(float(v[i]) * B[i] for i in range(len(B))) for v in ns]


cdims = {}
kinds_ok = True
for kind, lab in (("II", "(2,0,0)"), ("IB", "(1,1,0)"), ("IS", "(1,0,1)"), ("BS", "(0,1,1)"), ("BB", "(0,2,0)")):
    cb = commutant_basis(action_gens(kind), 2)
    cdims[lab] = len(cb)
    if lab in ("(2,0,0)", "(1,1,0)"):
        kinds_ok &= all(np.allclose(X, -X.T) for X in cb)
    if lab in ("(1,0,1)", "(0,1,1)"):
        cols = [0, 3, 4, 5] if kind[0] == 'I' else list(range(6))
        kinds_ok &= all(not np.any(X[:, cols]) and not np.any(X[cols, :]) for X in cb)
rep(22, cdims == {"(2,0,0)": 7, "(1,1,0)": 1, "(1,0,1)": 28, "(0,1,1)": 15, "(0,2,0)": 2} and kinds_ok,
    f"commutants at n=2 (exact sympy kernels): {cdims}; Euclidean antisymmetric for (2,0,0),(1,1,0) (so X^2 = 1 impossible), "
    "zero on and into the Lorentz-carrying subspace of constituent 1 for (1,0,1),(0,1,1)")

# invariant states of the diagonal lG and |0_K>
dG, PG = exact_inv_projector(lGdiag2_s)
wNK, VNK = np.linalg.eigh(NKs)
PGf = np.array(PG.evalf(), dtype=float)
wK = [np.trace(PGf @ (VNK[:, np.abs(wNK - k) < 0.5] @ VNK[:, np.abs(wNK - k) < 0.5].T)) for k in range(7)]
v0 = VNK[:, np.abs(wNK) < 0.5][:, 0]
vac_inv = all(np.allclose(g @ v0, 0, atol=1e-12) for g in lGdiag2_s)
wv = [np.sum(v0[Nd == k] ** 2) for k in range(7)]
lim = expm(-15 * sb) @ vac2; lim /= np.linalg.norm(lim)
rep(23, dG == 4 and np.allclose(wK, [1, 0, 0, 2, 0, 0, 1], atol=1e-10) and vac_inv
    and np.allclose(wv, [1 / 8, 0, 3 / 8, 0, 3 / 8, 0, 1 / 8], atol=1e-12) and abs(abs(lim @ v0) - 1) < 1e-9,
    "diagonal-lG invariant states: dim 4 (exact), weights 1,2,1 at N_K = 0,3,6; |0_K> is lG-invariant, with weights "
    "1/8,3/8,3/8,1/8 at N = 0,2,4,6, and equals the normalised e^{-15 sigma(eps(x)G)}|0> up to sign (tol 1e-9)")

# n=4: sparse lift of (eps+eps)(x)G
E4 = tower_sparse(4)
et4 = etan(4)
X4 = np.kron(np.kron(np.eye(2), epsm), G1)
inv4 = np.array_equal(X4 @ X4, np.eye(24)) and not np.any(np.diag(et4) @ X4 + X4.T @ np.diag(et4))
s4 = sigma(X4, E4, et4).tocsr()
lift4_ok = all(iszero(s4 @ E4[c] - E4[c] @ s4 - sum((X4[a, c] * E4[a] for a in range(24) if X4[a, c] != 0),
                                                     sps.csr_matrix((4096, 4096)))) for c in range(24))
lG4 = [sigma(np.kron(np.eye(4), X), E4, et4).tocsr() for X in LG_v + KG_v]
comm4 = all(iszero(s4 @ g - g @ s4) for g in lG4)
ev4 = np.linalg.eigvalsh(s4.toarray() + 6 * np.eye(4096))
evr = np.rint(ev4).astype(int)
mult4 = np.allclose(ev4, evr, atol=1e-8) and all(np.sum(evr == k) == comb(12, k) for k in range(13))
# other pairings also give involutions commuting with the action
perm_ok = True
for pairing in [((0, 2), (1, 3)), ((0, 3), (1, 2))]:
    Cm = np.zeros((4, 4))
    for (i, j) in pairing:
        Cm[j, i], Cm[i, j] = 1, -1
    Xp = np.kron(Cm, G1)
    perm_ok &= np.array_equal(Xp @ Xp, np.eye(24)) and all(not np.any(comm(Xp, np.kron(np.eye(4), X))) for X in LG_v + KG_v)
rep(24, inv4 and lift4_ok and comm4 and mult4 and perm_ok,
    "n=4 (S_4 of dimension 4096, sparse): (eps+eps)(x)G is an eta-antisymmetric involution, its lift has ad = X on all 24 "
    "generators and commutes with the six lifted diagonal lG generators (exact); 6 + sigma has spectrum 0..12 with "
    "multiplicities C(12,k) (eigvalsh, tol 1e-8); the pairings (13)(24), (14)(23) give further involutions (exact)")
del E4, s4, lG4

# ================================================================== 25. hopping and Rabi
Ch = sigma(crot, E2, et2)
form_h = sum(ap2[3 + i] @ am2[i] - ap2[i] @ am2[3 + i] for i in range(3))
okh = np.array_equal(Ch, form_h) and np.array_equal(Ch, -0.5 * sum(et2[a] * E2[a] @ E2[6 + a] for a in range(6)))
okh &= not np.any(comm(Ch, N2)) and np.any(comm(Ch, Nk2[0]))
okh &= all(np.array_equal(Ch @ ap2[i] @ vac2, ap2[3 + i] @ vac2) and np.array_equal(Ch @ ap2[3 + i] @ vac2, -ap2[i] @ vac2)
           for i in range(3))
okh &= np.allclose(expm(2 * pi * Ch), np.eye(64), atol=1e-11)
th = 0.77
okh &= np.allclose(expm(th * Ch) @ ap2[0] @ vac2, cos(th) * ap2[0] @ vac2 + sin(th) * ap2[3] @ vac2, atol=1e-12)
U = expm(th * Ch)
okh &= all(abs(U[i, j]) < 1e-13 for i in range(64) for j in range(64) if Nd[i] != Nd[j])
# n=3 copy rotations and boosts
K3 = np.kron(np.eye(3), K1v)
c3 = True
for (i, j) in [(0, 1), (1, 2), (0, 2)]:
    Cm = np.zeros((3, 3)); Cm[j, i], Cm[i, j] = 1, -1
    c3 &= not np.any(comm(np.kron(Cm, np.eye(6)), K3)) and not np.any(np.kron(Cm, G1) @ K3 + K3 @ np.kron(Cm, G1))
# Rabi at other parameters, numerical time evolution
k1, k2, be = 0.5, 2.0, 0.7
Hr = k1 * Nk2[0] + k2 * Nk2[1] + 1j * be * Ch
herm = np.allclose(Hr, Hr.conj().T) and np.allclose(comm(Hr, N2), 0)
psi0 = ap2[0] @ vac2
Om = sqrt((k2 - k1) ** 2 + 4 * be ** 2); s = 4 * be ** 2 / Om ** 2
rabi = True
for t in (0.3, 1.1, 4.0):
    psi = expm(-1j * Hr * t) @ psi0
    prob2 = sum(abs(psi @ (ap2[3 + i] @ vac2)) ** 2 for i in range(3))
    rabi &= abs(prob2 - s * sin(Om * t / 2) ** 2) < 1e-12
w_om = np.sort(np.linalg.eigvalsh(np.array([[k1, -1j * be], [1j * be, k2]])))
omp = 0.5 * (k1 + k2) + sqrt(0.25 * (k1 - k2) ** 2 + be ** 2); omm = 0.5 * (k1 + k2) - sqrt(0.25 * (k1 - k2) ** 2 + be ** 2)
rabi &= np.allclose(w_om, [omm, omp], atol=1e-14)
psi_eq = expm(-1j * (1.3 * N2 + 1j * be * Ch) * (pi / (2 * be))) @ psi0
rabi &= abs(sum(abs(psi_eq @ (ap2[3 + i] @ vac2)) ** 2 for i in range(3)) - 1) < 1e-12
rep(25, okh and c3 and herm and rabi,
    "copy rotation: sigma(eps(x)1) = -1/2 sum eta_aa E_(1,a)E_(2,a) = sum_i(a+_(2,i)a-_(1,i) - a+_(1,i)a-_(2,i)) (exact), commutes "
    "with N, not N^(1), hops a+_(1,i)|0> <-> a+_(2,i)|0> with the stated signs, e^{2 pi C} = 1, no matrix element between "
    "different N; n=3 copy rotations commute and copy boosts anticommute with K_3 (exact); Rabi at (0.5,2,0.7): transfer "
    "s sin^2(Omega t/2) (tol 1e-12), lines = omega_pm of RE Thm 7.12(ii), complete transfer at equal rates at t = pi/(2 beta)")

# paired rotations: dimension 3n(3n-1)/2 commuting with K and G (ladder statement)
lad = True
for n in (1, 2, 3, 4):
    Kn = np.kron(np.eye(n), K1v); Gn = np.kron(np.eye(n), G1)
    B = so_basis(n)
    M = np.vstack([np.array([comm(Kn, X).ravel() for X in B]).T, np.array([comm(Gn, X).ravel() for X in B]).T])
    dimKG = len(B) - rank_modp(np.rint(2 * M), P1)
    # compact part: antisymmetric elements of Cent(K)
    Mc = np.vstack([np.array([comm(Kn, X).ravel() for X in B]).T, np.array([(X + X.T).ravel() for X in B]).T])
    dimKc = len(B) - rank_modp(np.rint(2 * Mc), P1)
    lad &= dimKG == dimKc == 3 * n * (3 * n - 1) // 2
rep(26, lad, "ladder: Cent(K_n) cap Cent(G_n) = Cent(K_n) cap so(6n) has dimension 3n(3n-1)/2 at n=1..4 (exact F_p)")

# ================================================================== 27. citation numbers
def labels(path):
    d = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                d[parts[0]] = parts[1]
    return d


pl = labels(os.path.join(EMERG, "paper_labels.txt"))
rl = labels(os.path.join(EMERG, "re_labels.txt"))
es_ok = (pl.get("eq:realisation") == "1" and pl.get("eq:tower") == "2" and pl.get("eq:bladecomm") == "3" and pl.get("eq:Hp") == "8"
         and pl.get("thm:algebra") == "10.8" and pl.get("prop:orthospin") == "10.9" and pl.get("thm:GLorentz") == "12.1"
         and pl.get("thm:emfield") == "12.3" and pl.get("prop:rigid") == "26.4" and pl.get("thm:towervac") == "26.6"
         and pl.get("cor:noboost") == "2.4" and pl.get("ex:pairedmode") == "2.6" and pl.get("sec:interior") == "2.4"
         and pl.get("sec:dirac") == "4")
re_ok = (rl.get("d5:prop:graded") == "7.1" and rl.get("d5:prop:fock") == "7.2" and rl.get("d5:prop:cmrel") == "7.3"
         and rl.get("d5:sec:actions") == "7.5" and rl.get("d5:thm:bfield2") == "7.12" and rl.get("d6:cor:nopair") == "8.3"
         and rl.get("d6:prop:pairs") == "8.4" and rl.get("d6:prop:covariant") == "8.5" and rl.get("d6:thm:pair") == "8.6"
         and rl.get("d6:ex:bw") == "8.7" and rl.get("d6:ex:R1") == "8.8" and rl.get("d6:prop:gauge") == "8.9"
         and rl.get("d6:sec:gauge") == "8.4" and rl.get("d6:prop:dirac") == "8.11" and rl.get("d7:prop:exist") == "9.1"
         and rl.get("d7:prop:boost") == "9.4" and rl.get("d5:prop:so6C") == "7.8")
rep(27, es_ok and re_ok,
    "citation numbers: ES17 eq. (1),(2),(3),(8), Cor. 2.4, Ex. 2.6, Thm 10.8, Prop. 10.9, Thms 12.1, 12.3, Prop. 26.4, Thm 26.6 "
    "and RE 7.1-7.3, 7.5, 7.12, 8.3-8.9, 8.11, 9.1, 9.4 match the label files; lint is defined in ES17 Section 2.4 "
    "(sec:interior), Section 4 is the Dirac section; the copy algebra so(n,C) of the diagonal lG alone is RE Prop. 7.8(iii) / "
    "ES17 Thm 10.8 (RE Prop. 9.1(i) adds commutation with G_n)")

print()
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("all y2 checks passed")
sys.exit(0)
