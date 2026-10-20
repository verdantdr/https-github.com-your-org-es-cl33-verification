#!/usr/bin/env python3
"""Track e2 -- How one becomes two: vertices, thresholds and pair emergence.

Verification script for investigations/emergence/sections/e2_processes.tex.
Conventions (tex/s02_arena.tex, tex/sA_realisation.tex; note RE sections d5-d7):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1); e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1,
  e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep; omega = e1...e6; tower E_(k,a) = omega^{(x)(k-1)} (x) e_a (x) 1^{(x)(n-k)}.
  G e_i = e_{i+3}, G e_{i+3} = -e_i (binding root); K e_i = e_{i+3}, K e_{i+3} = e_i (para-complex structure).
  Interior Lorentz algebra lint: S_1 = e5e6/2 (cyclic in 4,5,6), T_a = e1 e_{a+3}/2.  Field-strength algebra lG:
  K_i = (e_k e_{j+3} - e_j e_{k+3})/2 (cyclic (i,j,k)), L_k = -[K_i,K_j] (tex/s12_emfield.tex, Theorem 12.3).
  Fock reading (RE Prop. 7.2): a+_(k,i) = (E_(k,i)+E_(k,i+3))/2, a-_(k,i) = (E_(k,i)-E_(k,i+3))/2, N = sum a+ a-;
  the Fock vacuum is the first basis vector (all bits zero).
  Spin lift of a vector-level generator A in so(eta_n) (A e_b = sum_a A_ab e_a): sigma(A) = 1/4 sum_ab A_ab eta_bb E_a E_b,
  so that [sigma(A), E_c] = A e_c.  The script works with lift4(A) = 4 sigma(A), an integer matrix.
Algebraic statements are checked in exact integer/rational arithmetic (numpy int64, sympy); spectra and exponentials in
floating point with the tolerances stated in the PASS lines.
Run from any directory:  python3 e2_processes.py   (exit 0 and one PASS line per claim iff all hold)
"""
import sys
import itertools
from collections import Counter
from math import comb

import numpy as np
import sympy as sp
import scipy.sparse as sps
from scipy.linalg import expm
from sympy.polys.matrices import DomainMatrix
from sympy.polys.domains import ZZ, QQ

FAILED = []


def report(cid, ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {cid}: {text}")
    if not ok:
        FAILED.append(cid)


def info(text):
    print(f"    . {text}")


# ----------------------------------------------------------------------------------------------- exact linear algebra
def rank_exact(M):
    M = np.asarray(M)
    if M.size == 0:
        return 0
    return DomainMatrix([[ZZ(int(x)) for x in row] for row in M.tolist()], M.shape, ZZ).convert_to(QQ).rank()


def nullity_exact(M):
    return np.asarray(M).shape[1] - rank_exact(M)


def nullspace_exact(M):
    """Rational basis of the right kernel of an integer matrix, returned as a list of integer numpy vectors."""
    S = sp.Matrix(np.asarray(M).tolist())
    out = []
    for v in S.nullspace():
        den = sp.ilcm(*[sp.fraction(x)[1] for x in v])
        out.append(np.array([int(x * den) for x in v], dtype=np.int64))
    return out


# ----------------------------------------------------------------------------------------------- realisation and tower
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def k3(a, b, c):
    return np.kron(np.kron(a, b), c)


e = [k3(sx, I2, I2), k3(sz, sx, I2), k3(sz, sz, sx), k3(ep, I2, I2), k3(sz, ep, I2), k3(sz, sz, ep)]
ETA = [1, 1, 1, -1, -1, -1]
I8 = np.eye(8, dtype=np.int64)
OM = e[0] @ e[1] @ e[2] @ e[3] @ e[4] @ e[5]


def tower(n):
    gens = []
    for k in range(n):
        for a in range(6):
            M = np.eye(1, dtype=np.int64)
            for j in range(n):
                M = np.kron(M, OM if j < k else (e[a] if j == k else I8))
            gens.append(M)
    return gens


def tower_sparse(n):
    OMs, Is = sps.csr_matrix(OM), sps.identity(8, dtype=np.int64, format="csr")
    es = [sps.csr_matrix(x) for x in e]
    gens = []
    for k in range(n):
        for a in range(6):
            M = sps.identity(1, dtype=np.int64, format="csr")
            for j in range(n):
                M = sps.kron(M, OMs if j < k else (es[a] if j == k else Is), format="csr")
            gens.append(M)
    return gens


def lift4(A, gens, etas):
    """4 sigma(A) for an integer vector-level generator A."""
    d = gens[0].shape[0]
    out = None
    for a in range(len(gens)):
        for b in range(len(gens)):
            if A[a, b] != 0:
                t = int(A[a, b]) * etas[b] * (gens[a] @ gens[b])
                out = t if out is None else out + t
    return out if out is not None else np.zeros((d, d), dtype=np.int64)


def advec(X, gens, etas):
    """Integer matrix of ad X on the span of the generators: [X, E_b] = sum_a M_ab E_a (X must be an integer matrix)."""
    d, m = gens[0].shape[0], len(gens)
    M = np.zeros((m, m), dtype=np.int64)
    for b in range(m):
        C = X @ gens[b] - gens[b] @ X
        for a in range(m):
            t = int(np.trace(gens[a] @ C))
            assert t % (d) == 0
            M[a, b] = t // d * etas[a]
    return M


def comm(A, B):
    return A @ B - B @ A


def number_ops(gens, n):
    """2 N^(k) (integer) for k = 1..n and 2N."""
    d = gens[0].shape[0]
    Nk = []
    for k in range(n):
        M = np.zeros((d, d), dtype=np.int64)
        for i in range(3):
            M = M + np.eye(d, dtype=np.int64) - gens[6 * k + i] @ gens[6 * k + i + 3]
        Nk.append(M)
    return Nk, sum(Nk)


def rot(i, j, d):
    R = np.zeros((d, d), dtype=np.int64)
    R[j, i], R[i, j] = 1, -1
    return R


Gv = np.zeros((6, 6), dtype=np.int64)
Kv = np.zeros((6, 6), dtype=np.int64)
for i in range(3):
    Gv[i + 3, i], Gv[i, i + 3] = 1, -1
    Kv[i + 3, i], Kv[i, i + 3] = 1, 1
EPS = np.array([[0, -1], [1, 0]], dtype=np.int64)
ETAM = np.diag(ETA)


def in_so(A, etas):
    E = np.diag(etas)
    return np.array_equal((E @ A).T, -(E @ A))


def so_basis(etas):
    """Basis of so(eta): A = eta^{-1}(E_ab - E_ba), a<b."""
    m = len(etas)
    out = []
    for a in range(m):
        for b in range(a + 1, m):
            M = np.zeros((m, m), dtype=np.int64)
            M[a, b], M[b, a] = 1, -1
            out.append(np.diag(etas) @ M)
    return out


def commutant_in_so(gens_vec, etas):
    """Exact basis of the elements of so(eta) commuting with every matrix in gens_vec."""
    basis = so_basis(etas)
    rows = []
    for X in gens_vec:
        rows.append(np.stack([comm(B, X).flatten() for B in basis], axis=1))
    sol = nullspace_exact(np.vstack(rows))
    return [sum(int(c) * B for c, B in zip(v, basis)) for v in sol]


# =============================================================================================== n = 1 and n = 2 data
g1 = e
g2 = tower(2)
ET2 = ETA + ETA
d2 = 64
I64 = np.eye(64, dtype=np.int64)
twoNk1, twoN1 = number_ops(g1, 1)
twoNk2, twoN2 = number_ops(g2, 2)
Ndiag1 = np.diag(twoN1) // 2
Ndiag2 = np.diag(twoN2) // 2

# Lorentz generators (doubled, integer): 2 S_k, 2 T_k, 2 L_k, 2 K_k on S
S2x = [e[4] @ e[5], e[5] @ e[3], e[3] @ e[4]]
T2x = [e[0] @ e[3], e[0] @ e[4], e[0] @ e[5]]
K2x = [e[2] @ e[4] - e[1] @ e[5], e[0] @ e[5] - e[2] @ e[3], e[1] @ e[3] - e[0] @ e[4]]
L2x = []
for (i, j) in [(1, 2), (2, 0), (0, 1)]:
    C = -comm(K2x[i], K2x[j])
    assert np.all(C % 2 == 0)
    L2x.append(C // 2)
LINT = S2x + T2x
LG = L2x + K2x
LINTV = [advec(X, g1, ETA) for X in LINT]   # 2 x (vector-level generator), integer
LGV = [advec(X, g1, ETA) for X in LG]


def diag_lift_n(Xs, gens, n):
    """Sum over copies of the copy-k bivector built from a one-copy bivector given as a sum of e_a e_b."""
    out = []
    for X in Xs:
        A = advec(X, g1, ETA)   # 2 * vector generator
        out.append(lift4(np.kron(np.eye(n, dtype=np.int64), A), gens, [x for _ in range(n) for x in ETA]) // 4)
    return out


# ----------------------------------------------------------------------------------------------- e2-1 conventions
ok = all(np.array_equal(g2[a] @ g2[b] + g2[b] @ g2[a], 2 * ET2[a] * (a == b) * I64) for a in range(12) for b in range(12))
OM1 = np.kron(OM, I8)
ok &= all(np.array_equal(np.kron(I8, e[a]), OM1 @ g2[6 + a]) for a in range(6))
ok &= all(np.array_equal(g2[6 + a] @ g2[6 + b], np.kron(I8, e[a] @ e[b])) for a in range(6) for b in range(6))
ok &= np.array_equal(np.diag(np.diag(twoN2)), twoN2) and Ndiag2[0] == 0 and Ndiag1[0] == 0
ok &= Counter(Ndiag2.tolist()) == Counter({k: comb(6, k) for k in range(7)})
# CAR at n = 2 in doubled form: {2a+,2a-} = 4 delta, {2a+,2a+} = 0
ap2 = lambda k, i: g2[6 * k + i] + g2[6 * k + i + 3]
am2 = lambda k, i: g2[6 * k + i] - g2[6 * k + i + 3]
for (k, i), (l, j) in itertools.product(itertools.product(range(2), range(3)), repeat=2):
    ok &= np.array_equal(ap2(k, i) @ am2(l, j) + am2(l, j) @ ap2(k, i), 4 * ((k, i) == (l, j)) * I64)
    ok &= np.array_equal(ap2(k, i) @ ap2(l, j) + ap2(l, j) @ ap2(k, i), 0 * I64)
ok &= all(np.array_equal(ap2(k, i).T, am2(k, i)) for k in range(2) for i in range(3))
report("e2-1", ok, "tower Clifford relations at n=2; ungraded copy-2 generators 1(x)e_a = omega^(1) E_(2,a); bivectors of the graded "
       "and ungraded copy 2 coincide (so every diagonal Lorentz action is the same on S_2 = S(x)S); CAR with a- = (a+)^T; "
       "N has spectrum 0..6, multiplicities binomial(6,k), Fock vacuum = first basis vector (exact)")

# ----------------------------------------------------------------------------------------------- e2-2 .. e2-6 vertices
def homdim(rx, ry):
    """Exact real dimension of Hom_l(X, Y) for doubled integer generators."""
    dx, dy = rx[0].shape[0], ry[0].shape[0]
    rows = [np.kron(np.eye(dx, dtype=np.int64), Y) - np.kron(X.T, np.eye(dy, dtype=np.int64)) for X, Y in zip(rx, ry)]
    return nullity_exact(np.vstack(rows))


def tens(r1, r2):
    return [np.kron(A, np.eye(B.shape[0], dtype=np.int64)) + np.kron(np.eye(A.shape[0], dtype=np.int64), B) for A, B in zip(r1, r2)]


LINT_S2 = [np.kron(X, I8) + np.kron(I8, X) for X in LINT]
ok = all(np.array_equal(Y, Z) for Y, Z in zip(LINT_S2, diag_lift_n(LINT, g2, 2)))
h12 = homdim(LINT, LINT_S2)
h21 = homdim(LINT_S2, LINT)
R2pi_S = expm(np.pi * S2x[2].astype(float))           # exp(2 pi S_3), S_3 = e4e5/2
R2pi_S2 = expm(np.pi * (LINT_S2[2]).astype(float))
ok &= h12 == 0 and h21 == 0 and np.allclose(R2pi_S, -np.eye(8), atol=1e-12) and np.allclose(R2pi_S2, np.eye(64), atol=1e-12)
report("e2-2", ok, f"diagonal lint: dim Hom(S, S_2) = {h12}, dim Hom(S_2, S) = {h21} (exact); exp(2 pi S_3) = -1 on S and +1 on S_2 "
       "(tol 1e-12): no vertex takes one spin-1/2 module into the two-fermion space")

IDX_E = [0, 3, 4, 5]
rhoE = [A[np.ix_(IDX_E, IDX_E)] for A in LINTV]
ok_inv = all(np.array_equal(A[np.ix_([1, 2], IDX_E)], 0 * A[np.ix_([1, 2], IDX_E)]) for A in LINTV)
pairs6 = [(a, b) for a in range(6) for b in range(a + 1, 6)]


def adBV(X, basis):
    """Integer matrix of ad X (X doubled) on a basis of blades (each blade squares to +-1)."""
    m = len(basis)
    M = np.zeros((m, m), dtype=np.int64)
    for j, Bj in enumerate(basis):
        C = comm(X, Bj)
        for i, Bi in enumerate(basis):
            M[i, j] = int(np.trace(Bi.T @ C)) // int(np.trace(Bi.T @ Bi))
    return M


bv6 = [e[a] @ e[b] for a, b in pairs6]
pE = [(a, b) for a, b in pairs6 if a in IDX_E and b in IDX_E]
bvE = [e[a] @ e[b] for a, b in pE]
rhoL2V = [adBV(X, bv6) for X in LINT]
rhoL2E = [adBV(X, bvE) for X in LINT]
dims = {
    "End": homdim(LINT, LINT),
    "E": homdim(LINT, tens(LINT, rhoE)),
    "V": homdim(LINT, tens(LINT, LINTV)),
    "L2E": homdim(LINT, tens(LINT, rhoL2E)),
    "L2V": homdim(LINT, tens(LINT, rhoL2V)),
}
ok = ok_inv and dims == {"End": 8, "E": 8, "V": 24, "L2E": 8, "L2V": 32}
report("e2-3", ok, f"lint-equivariant maps (real dimensions, exact): End(S) = {dims['End']}, S -> S(x)E = {dims['E']}, "
       f"S -> S(x)V = {dims['V']} (= 8 + 2*8), S -> S(x)Lambda^2 E = {dims['L2E']}, S -> S(x)Lambda^2 V = {dims['L2V']} "
       "(= 8 + 16 + 8): fermion -> fermion + boson vertices exist")


def homspace(rx, ry):
    dx, dy = rx[0].shape[0], ry[0].shape[0]
    rows = [np.kron(np.eye(dx, dtype=np.int64), Y) - np.kron(X.T, np.eye(dy, dtype=np.int64)) for X, Y in zip(rx, ry)]
    vs = nullspace_exact(np.vstack(rows))
    return [v.reshape(dx, dy).T for v in vs]       # column-major vec


def equiv(M, rx, ry):
    return all(np.array_equal(Y @ M, M @ X) for X, Y in zip(rx, ry))


C1 = sum(ETA[a] * np.kron(e[a], np.eye(4, dtype=np.int64)[:, [m]]) for m, a in enumerate(IDX_E))       # psi -> sum eta gamma_mu psi (x) e_mu
C2 = sum(ETA[a] * ETA[b] * np.kron(e[a] @ e[b], np.eye(6, dtype=np.int64)[:, [m]]) for m, (a, b) in enumerate(pE))
C1n = sum(np.kron(e[a], np.eye(4, dtype=np.int64)[:, [m]]) for m, a in enumerate(IDX_E))
EndS = homspace(LINT, LINT)
ok = equiv(C1, LINT, tens(LINT, rhoE)) and equiv(C2, LINT, tens(LINT, rhoL2E)) and not equiv(C1n, LINT, tens(LINT, rhoE))
ok &= rank_exact(np.array([(C1 @ X).flatten() for X in EndS])) == 8
ok &= rank_exact(np.array([(C2 @ X).flatten() for X in EndS])) == 8
EndS_blades = [I8, e[1], e[2], e[1] @ e[2], e[0] @ e[3] @ e[4] @ e[5]]
ok &= all(equiv(B, LINT, LINT) for B in EndS_blades)
report("e2-4", ok, "Clifford multiplication by the interior gammas, psi -> sum_mu eta_mumu gamma_mu psi (x) e_mu, and by their "
       "bivectors are lint-equivariant (the version without eta is not); composed with End_lint(S) (dimension 8, containing "
       "1, e2, e3, J, Omega) they span the whole of Hom(S, S(x)E) and Hom(S, S(x)Lambda^2 E) (ranks 8, exact)")

# other Lorentz actions at n = 2: spectator (1,0,1) and mixed (1,1,0) (tex/s13 Prop. 13.16)
SPEC2 = [np.kron(X, I8) for X in LINT]
MIX2 = [np.kron(X, I8) + np.kron(I8, Y) for X, Y in zip(LINT, LG)]
h_spec = homdim(LINT, SPEC2)
h_mix = homdim(LINT, MIX2)
h_mixV = homdim(LINT, tens(LINT, LGV))
CasG = sum(x @ x for x in K2x) - sum(x @ x for x in L2x)   # 4 x Casimir
CasI = sum(x @ x for x in T2x) - sum(x @ x for x in S2x)
ok = h_spec == 64 and h_mix == 16 and h_mixV == 8 and np.array_equal(CasG, 12 * I8) and np.array_equal(CasI, 6 * I8)
ok &= np.allclose(expm(np.pi * L2x[2].astype(float)), np.eye(8), atol=1e-12)
report("e2-5", ok, f"spectator action: dim Hom(S, S_2) = {h_spec} = 8 x 8 (copy 2 an internal label); mixed action (lint on copy 1, "
       f"lG on copy 2): dim Hom(S, S_2) = {h_mix}, dim Hom(S, S (x) V^(2)) = {h_mixV} (fermion -> fermion + frame field); "
       "lG Casimir sum K^2 - sum L^2 = 3 on all of S (two four-vectors; lint Casimir 3/2), exp(2 pi L_3) = +1 on S (exact; tol 1e-12)")

# ladder: n = 3 under the diagonal interior action
g3 = tower(3)
S3_3 = sum(g3[6 * k + 3] @ g3[6 * k + 4] for k in range(3))      # 2 x lift of S_3
T3_3 = sum(g3[6 * k + 0] @ g3[6 * k + 5] for k in range(3))      # 2 x lift of T_3
ok = np.array_equal(S3_3 @ T3_3, T3_3 @ S3_3)
ok &= np.allclose(expm(np.pi * S3_3.astype(float)), -np.eye(512), atol=1e-10)
lam = np.linalg.eigvals(-1j * S3_3 + np.sqrt(2) * T3_3)       # eigenvalues 2j + sqrt2 * 2b
wts = Counter()
for x in lam:
    hit = None
    for j2 in range(-12, 13):
        b2 = (x.real - j2) / np.sqrt(2)
        if abs(b2 - round(b2)) < 1e-7 and abs(x.imag) < 1e-7:
            hit = (j2, int(round(b2)))
            break
    ok &= hit is not None
    if hit:
        wts[((hit[0] + hit[1]) // 2, (hit[0] - hit[1]) // 2)] += 1


def mult(tL, tR):
    f = lambda a, b: wts.get((a, b), 0)
    return f(tL, tR) - f(tL + 2, tR) - f(tL, tR + 2) + f(tL + 2, tR + 2)


mL, mR = mult(1, 0), mult(0, 1)
ok &= mL == 40 and mR == 40
report("e2-6", ok, f"diagonal lint at n=3: exp(2 pi S_3) = -1 on S_3 (tol 1e-10), so the 2 pi sign is (-1)^n for n = 1,2,3; joint "
       f"weights of (S_3, T_3) give multiplicity {mL} of (1/2,0) and {mR} of (0,1/2) in S_3 (x) C, hence dim Hom(S, S_3) = "
       f"2*40 + 2*40 = 160: one -> three constituents is allowed (weights to 1e-7)")

# ----------------------------------------------------------------------------------------------- e2-7, e2-8 thresholds
rng = np.random.default_rng(20261020)


def h1(m, p):
    return m * e[0] + sum(p[a] * e[0] @ e[3 + a] for a in range(3))


def Hu(m1, m2, p, q):
    return np.kron(h1(m1, p), I8) + np.kron(I8, h1(m2, q))


def Hg(m1, m2, p, q):
    H = m1 * g2[0] + m2 * g2[6]
    for a in range(3):
        H = H + p[a] * g2[0] @ g2[3 + a] + q[a] * g2[6] @ g2[9 + a]
    return H


ok = True
worst = 0.0
for _ in range(20):
    m1, m2 = rng.uniform(0.2, 3, 2)
    p, q = rng.normal(size=3), rng.normal(size=3)
    E1, E2 = np.sqrt(m1 ** 2 + p @ p), np.sqrt(m2 ** 2 + q @ q)
    ev = np.sort(np.linalg.eigvalsh(Hu(m1, m2, p, q)))
    want = np.sort(np.repeat([E1 + E2, E1 - E2, -E1 + E2, -E1 - E2], 16))
    worst = max(worst, np.max(np.abs(ev - want)))
    ok &= E1 + E2 >= m1 + m2
ev0 = np.linalg.eigvalsh(Hu(1, 2, [0, 0, 0], [0, 0, 0]))
ok &= worst < 1e-10 and Counter(np.round(ev0).astype(int).tolist()) == Counter({3: 16, 1: 16, -1: 16, -3: 16})
report("e2-7", ok, f"ungraded two-particle operator H(p)(x)1 + 1(x)H(q): spectrum {{+-E1 +- E2}}, 16 each (20 random samples, "
       f"max error {worst:.1e}); at rest with (m1,m2)=(1,2): {{+-3,+-1}}; E1+E2 >= m1+m2 in every sample (free-pair threshold, inherited)")

ok = True
for (m1, m2, p, q) in [(1, 2, (0, 0, 0), (0, 0, 0)), (1, 2, (1, -2, 0), (0, 1, 3)), (3, 1, (2, 0, 1), (-2, 0, -1)), (2, 2, (1, 1, 1), (0, 0, 0))]:
    H = Hg(m1, m2, p, q)
    pp, qq = int(np.dot(p, p)), int(np.dot(q, q))
    a = m1 ** 2 + m2 ** 2 + pp + qq
    b = m1 ** 2 * qq + m2 ** 2 * pp + pp * qq
    X = H @ H - a * I64
    ok &= np.array_equal(X @ X, 4 * b * I64) and np.array_equal(H, H.T)
evr = np.linalg.eigvalsh(Hg(1, 2, [0] * 3, [0] * 3).astype(float))
ok &= np.allclose(np.abs(evr), np.sqrt(5), atol=1e-12) and int(np.sum(evr > 0)) == 32
below = True
for _ in range(200):
    m1, m2 = rng.uniform(0.1, 3, 2)
    p, q = rng.normal(size=3) * rng.uniform(0, 3), rng.normal(size=3) * rng.uniform(0, 3)
    E1, E2 = np.sqrt(m1 ** 2 + p @ p), np.sqrt(m2 ** 2 + q @ q)
    a = m1 ** 2 + m2 ** 2 + p @ p + q @ q
    b = m1 ** 2 * (q @ q) + m2 ** 2 * (p @ p) + (p @ p) * (q @ q)
    Ep = np.sqrt(a + 2 * np.sqrt(b))
    evp = np.max(np.linalg.eigvalsh(Hg(m1, m2, p, q)))
    below &= abs(evp - Ep) < 1e-9 and max(E1, E2) - 1e-12 <= Ep < E1 + E2
B11 = 2 - np.sqrt(2)
ok &= below and abs(B11 / 2 - (1 - 1 / np.sqrt(2))) < 1e-15
report("e2-8", ok, "graded composite H_g = H^(1)(p) + H^(2)(q) in tower generators: (H_g^2 - a)^2 = 4b exactly with "
       "a = m1^2+m2^2+p^2+q^2, b = m1^2 q^2 + m2^2 p^2 + p^2 q^2 (four integer data); at rest spectrum +-sqrt(m1^2+m2^2), 32 each "
       "((1,2): +-sqrt5, tol 1e-12); top branch sqrt(a+2 sqrt b) satisfies max(E1,E2) <= E_+ < E1+E2 in 200 random samples "
       "(tol 1e-9); relative binding 1-1/sqrt2 = 0.2929 at m1=m2")

# ----------------------------------------------------------------------------------------------- e2-9 field level (restated)
al, be = sp.symbols("alpha beta")
z = sp.Matrix([1, sp.I, 0])
Z = sp.Matrix.hstack(al * z, be * z)
Q = (Z.T * Z).applyfunc(sp.expand)
z1, z2 = sp.Matrix([1, sp.I, 0]), sp.Matrix([1, -sp.I, 0])
QBW = sp.Matrix.hstack(z1, z2).T * sp.Matrix.hstack(z1, z2)
svals = sorted([sp.sqrt(x) for x in (QBW.H * QBW).eigenvals(multiple=True)])
ok = Q == sp.zeros(2, 2) and QBW == sp.Matrix([[0, 2], [2, 0]]) and sp.Rational(1, 2) * sum(svals) == 2
report("e2-9", ok, "restated (RE Prop. 8.5(iii), Thm 8.6): every covariant duplicate (alpha z, beta z) of a null z has Gram matrix Q = 0, "
       "so M_pair = (sigma1+sigma2)/2 = 0; the counter-propagating null pair has Q = [[0,2],[2,0]] and M_pair = 2 (exact)")

# ----------------------------------------------------------------------------------------------- e2-10 .. e2-13 number and K
ok = True
Kn = {n: np.kron(np.eye(n, dtype=np.int64), Kv) for n in (1, 2, 3, 4)}
Gn = {n: np.kron(np.eye(n, dtype=np.int64), Gv) for n in (1, 2, 3, 4)}
ETn = {n: [x for _ in range(n) for x in ETA] for n in (1, 2, 3, 4)}
ok &= np.array_equal(advec(twoN1, g1, ETA), 2 * Kn[1]) and np.array_equal(advec(twoN2, g2, ET2), 2 * Kn[2])
ok &= np.array_equal(lift4(Kn[1], g1, ETA) + 6 * I8, 2 * twoN1)      # 4 sigma(K) + 6 = 4N
par1 = np.diag((-1) ** Ndiag1)
par2 = np.diag((-1) ** Ndiag2)
vol2 = I64.copy()
for x in g2:
    vol2 = vol2 @ x
ok &= np.array_equal(OM, -par1) and np.array_equal(vol2, par2)
ok &= all(np.array_equal(comm(OM, e[a] @ e[b]), 0 * I8) for a in range(6) for b in range(6))
report("e2-10", ok, "ad N = K on V and V_2 (N = 3n/2 + sigma(K), the Fock number is the spin lift of the para-complex structure K); "
       "(-1)^N = -omega at n=1 and = +volume element at n=2, commuting with every bivector: quanta change in pairs (exact)")

ok = True
dims_cent, dims_anti = [], []
for n in (1, 2, 3):
    basis = so_basis(ETn[n])
    rows = np.stack([comm(B, Kn[n]).flatten() for B in basis], axis=1)
    dc = nullity_exact(rows)
    rows_a = np.stack([(B @ Kn[n] + Kn[n] @ B).flatten() for B in basis], axis=1)
    da = nullity_exact(rows_a)
    dims_cent.append(dc)
    dims_anti.append(da)
    ok &= dc == 9 * n * n and da == 9 * n * n - 3 * n and dc + da == len(basis)
# pure pair: lifts of K-anticommuting elements map H_k -> H_{k+-2}; check on a basis at n = 2
basis2 = so_basis(ET2)
rows_a = np.stack([(B @ Kn[2] + Kn[2] @ B).flatten() for B in basis2], axis=1)
anti2 = [sum(int(c) * B for c, B in zip(v, basis2)) for v in nullspace_exact(rows_a)]
rows_c = np.stack([comm(B, Kn[2]).flatten() for B in basis2], axis=1)
cent2 = [sum(int(c) * B for c, B in zip(v, basis2)) for v in nullspace_exact(rows_c)]
for A in anti2:
    L = lift4(A, g2, ET2)
    for j in range(7):
        for k in range(7):
            if abs(j - k) != 2:
                blk = L[np.ix_(Ndiag2 == j, Ndiag2 == k)]
                ok &= not blk.any()
for A in cent2:
    L = lift4(A, g2, ET2)
    ok &= np.array_equal(L @ twoN2, twoN2 @ L)
report("e2-11", ok, f"so(3n,3n) = Cent(K_n) + Anti(K_n) with dimensions {dims_cent} + {dims_anti} at n=1,2,3 (= 9n^2 + 9n^2-3n, exact); "
       "at n=2 every lift from Cent(K) commutes with N (gl(6,R), hopping) and every lift from Anti(K) maps H_k into H_{k-2}+H_{k+2} "
       "(pure pair creation and annihilation), exact")

ok = True
d_list = []
for n in (1, 2, 3):
    basis = so_basis(ETn[n])
    # compact (Euclidean antisymmetric) and commuting with K, resp. with G
    sols = {}
    for name, Y in (("K", Kn[n]), ("G", Gn[n])):
        rows = [np.stack([comm(B, Y).flatten() for B in basis], axis=1), np.stack([(B + B.T).flatten() for B in basis], axis=1)]
        sols[name] = [sum(int(c) * B for c, B in zip(v, basis)) for v in nullspace_exact(np.vstack(rows))]
    dK, dG = len(sols["K"]), len(sols["G"])
    both = np.array([x.flatten() for x in sols["K"] + sols["G"]])
    d_list.append(dK)
    ok &= dK == dG == 3 * n * (3 * n - 1) // 2 and rank_exact(both) == dK
    # paired rotations: block form diag(a, a) in the (temporal; spatial) ordering
    perm = [6 * k + i for k in range(n) for i in range(3)] + [6 * k + i + 3 for k in range(n) for i in range(3)]
    for X in sols["K"]:
        Y = X[np.ix_(perm, perm)]
        m = 3 * n
        ok &= np.array_equal(Y[:m, :m], Y[m:, m:]) and not Y[:m, m:].any() and not Y[m:, :m].any()
report("e2-12", ok, f"compact generators conserving N are exactly the compact binding-compatible ones: so(6n) cap Cent(K_n) = "
       f"so(6n) cap Cent(G_n) = paired rotations so(3n), dimensions {d_list} at n=1,2,3 (exact)")


def split_K(A, K):
    """Parts of A commuting and anticommuting with the involution K."""
    return (A + K @ A @ K), (A - K @ A @ K)     # doubled


ok = True
lint_a = np.array([split_K(A, Kv)[1].flatten() for A in LINTV])
lint_dim = len(LINTV) - rank_exact(lint_a)          # dim (lint cap Cent K)
pole = split_K(LINTV[3], Kv)        # 2 T_1 = e1e4, the pole boost
ok &= lint_dim == 1 and not pole[1].any()
ok &= all(split_K(LINTV[k], Kv)[1].any() for k in (0, 1, 2, 4, 5))
ok &= all(not split_K(A, Kv)[1].any() for A in LGV[:3]) and all(not split_K(A, Kv)[0].any() for A in LGV[3:])
JV = advec(e[1] @ e[2], g1, ETA)
Jc, Ja = split_K(JV, Kv)
ok &= Jc.any() and Ja.any()
# n = 2: diagonal lint cap gl(6), M_12, C_b
LINT2V = [np.kron(np.eye(2, dtype=np.int64), A) for A in LINTV]
lint2_dim = len(LINT2V) - rank_exact(np.array([split_K(A, Kn[2])[1].flatten() for A in LINT2V]))
M12v = np.kron(EPS, np.diag([1, 0, 0, 1, 1, 1]))
Cv = np.kron(EPS, np.eye(6, dtype=np.int64))
CGv = np.kron(EPS, Gv)
ok &= lint2_dim == 1 and split_K(M12v, Kn[2])[1].any() and not split_K(Cv, Kn[2])[1].any() and not split_K(CGv, Kn[2])[0].any()
report("e2-13", ok, "lint cap gl(3,R) = R e1e4 (the pole boost T_1; dim 1) and the rotations S_k and boosts T_2, T_3 all have pair parts; "
       "lG cap gl(3,R) = span L_k while the lG boosts K_k are pure pair; J = e2e3 has both parts; at n=2 diag lint cap gl(6) has "
       "dim 1, M_12 has a pair part, the copy rotation eps(x)1 is number-conserving and the copy boost eps(x)G is pure pair (exact)")

# ----------------------------------------------------------------------------------------------- e2-14 .. e2-17 amplitudes
vac1 = np.zeros(8)
vac1[0] = 1
vac2 = np.zeros(64)
vac2[0] = 1
idx = lambda bits: int(bits, 2)
ok = True
T2 = 0.5 * T2x[1]      # T_2 = e1e5/2
adT2 = advec(T2x[1], g1, ETA) / 2
ok &= np.allclose(np.sort(np.linalg.eigvals(adT2).real), [-1, 0, 0, 0, 0, 1])
for zeta in (0.3, 1.0, 2.5):
    v = expm(zeta * T2) @ vac1
    w = np.zeros(8)
    w[idx("000")], w[idx("110")] = np.cosh(zeta / 2), np.sinh(zeta / 2)
    ok &= np.allclose(v, w, atol=1e-12) and abs(v @ v - np.cosh(zeta)) < 1e-12
    P0 = v[0] ** 2 / (v @ v)
    ok &= abs(P0 - (1 + np.cosh(zeta)) / (2 * np.cosh(zeta))) < 1e-12
U = expm(1.0 * T2)
ok &= np.allclose(U, U.T) and not np.allclose(U.T @ U, np.eye(8))
report("e2-14", ok, "interior boost U = exp(zeta T_2), T_2 = e1e5/2 (ad T_2 has eigenvalues +-1: rapidity zeta): U|0> = cosh(zeta/2)|000> + "
       "sinh(zeta/2)|110> exactly (zeta = 0.3, 1, 2.5; tol 1e-12); |U|0>|^2 = cosh zeta, U symmetric and not orthogonal; normalised "
       "persistence cosh^2(zeta/2)/cosh zeta -> 1/2, tan(theta) = tanh(zeta/2)")

ok = True
for th in (0.4, 1.3, np.pi):
    v = expm(th * 0.5 * S2x[0]) @ vac1      # S_1 = e5e6/2
    w = np.zeros(8)
    w[idx("000")], w[idx("011")] = np.cos(th / 2), np.sin(th / 2)
    ok &= np.allclose(v, w, atol=1e-12)
    v = expm(th * (e[1] @ e[2])) @ vac1
    w = np.zeros(8)
    w[idx("000")], w[idx("011")] = np.cos(th), np.sin(th)
    ok &= np.allclose(v, w, atol=1e-12)
v = expm(np.pi * 0.5 * S2x[0]) @ vac1
ok &= abs(v[idx("011")] - 1) < 1e-12
report("e2-15", ok, "compact generators create pairs unitarily: exp(theta S_1)|0> = cos(theta/2)|000> + sin(theta/2)|011> (a spatial rotation "
       "through pi turns the vacuum into a two-quantum state), exp(theta J)|0> = cos(theta)|000> + sin(theta)|011> (tol 1e-12)")

ok = True
K1 = 0.5 * K2x[0]
ok &= np.allclose(np.sort(np.linalg.eigvals(advec(K2x[0], g1, ETA) / 2).real), [-1, -1, 0, 0, 1, 1])
for zeta in (0.3, 1.0, 2.5):
    v = expm(zeta * K1) @ vac1
    w = np.zeros(8)
    w[idx("000")], w[idx("011")] = np.cosh(zeta), -np.sinh(zeta)
    ok &= np.allclose(v, w, atol=1e-11)
for L in L2x:
    ok &= np.array_equal(L @ twoN1, twoN1 @ L)
report("e2-16", ok, "lG boost K_1 of rapidity zeta (ad K_1 eigenvalues +-1, +-1, 0, 0): exp(zeta K_1)|0> = cosh(zeta)|000> - sinh(zeta)|011> "
       "(tol 1e-11), normalised persistence cosh^2 zeta / cosh 2zeta; the lG rotations L_k commute with N (exact)")

ok = True
CG4 = lift4(CGv, g2, ET2)
ok &= np.array_equal(advec(CG4, g2, ET2), 4 * CGv) and np.array_equal(CG4, CG4.T)
sCG = CG4 / 4.0
twoN1_2, twoN2_2 = twoNk2
for zeta in (0.3, 1.0, 2.0):
    v = expm(zeta * sCG) @ vac2
    ok &= abs(v[0] - np.cosh(zeta) ** 3) < 1e-10 * np.cosh(zeta) ** 3
    for k in range(4):
        Pk = np.sum(v[Ndiag2 == 2 * k] ** 2)
        ok &= abs(Pk - comb(3, k) * np.cosh(zeta) ** (2 * (3 - k)) * np.sinh(zeta) ** (2 * k)) < 1e-9 * np.cosh(2 * zeta) ** 3
        ok &= np.sum(v[Ndiag2 == 2 * k + 1] ** 2) < 1e-20
    ok &= np.allclose((twoN1_2 - twoN2_2) @ v, 0, atol=1e-9)
report("e2-17", ok, "copy boost eps(x)G (lift checked, rapidity zeta): <0|U|0> = cosh^3 zeta, weight in N = 2k equal to "
       "C(3,k) cosh^{2(3-k)} zeta sinh^{2k} zeta (total cosh^3 2zeta), and N^(1) = N^(2) on U|0>: every emerging pair has one quantum on "
       "each rung (zeta = 0.3, 1, 2; relative tol 1e-9)")

# ----------------------------------------------------------------------------------------------- e2-18 invariant states
ok = True
inv1 = nullity_exact(np.vstack(LINT))
ok &= inv1 == 0
# Prop. 26.4 projectors are parity sectors; paired-mode projector (Ex. 2.6) = projector on N >= 2
Pmode4 = 2 * I8 - (sum(e[i] @ e[i + 3] for i in range(3)) + OM)            # 4 Pi = 2 - (P + V)
ok &= np.array_equal(Pmode4, 4 * np.diag((Ndiag1 >= 2).astype(np.int64)))
ok &= np.array_equal(I8 + OM, 2 * np.diag((Ndiag1 % 2 == 1).astype(np.int64)))
invvec = nullspace_exact(np.vstack(LINT_S2))
Wv = np.array(invvec).T.astype(float)
Qw, _ = np.linalg.qr(Wv)
Pinv = Qw @ Qw.T
content = [float(np.trace(Pinv[np.ix_(Ndiag2 == k, Ndiag2 == k)])) for k in range(7)]
ok &= len(invvec) == 8 and np.allclose(content, [0, 0.5, 2, 3, 2, 0.5, 0], atol=1e-12)
defN = []
for k in range(7):
    sel = np.where(Ndiag2 == k)[0]
    defN.append(nullity_exact(np.vstack([X[:, sel] for X in LINT_S2])))
ok &= defN == [0, 0, 0, 2, 0, 0, 0]
sel = np.where(Ndiag2 == 3)[0]
Ksub = nullspace_exact(np.vstack([X[:, sel] for X in LINT_S2]))
for v in Ksub:
    full = np.zeros(64, dtype=np.int64)
    full[sel] = v
    for i in range(3):
        n1 = (I64 - g2[i] @ g2[i + 3]) // 2 + (I64 - g2[6 + i] @ g2[9 + i]) // 2
        ok &= np.array_equal(n1 @ full, full)
inv_spec = nullity_exact(np.vstack(SPEC2))
H26 = g2[1] @ g2[2] @ g2[6] @ g2[7]
Pi0x2 = I64 + H26
ok &= inv_spec == 0 and np.array_equal(Pi0x2 @ par2, par2 @ Pi0x2) and not np.array_equal(Pi0x2 @ twoN2, twoN2 @ Pi0x2)
pi_content = [int(np.trace(Pi0x2[np.ix_(Ndiag2 == k, Ndiag2 == k)])) for k in range(7)]
ok &= pi_content == [comb(6, k) for k in range(7)]          # trace of 2 Pi0 = full sector dimension: half of each sector
report("e2-18", ok, "invariant states: none at n=1 under lint; Prop. 26.4's projectors (1+-omega)/2 are the parity sectors and the paired-mode "
       "projector of Ex. 2.6 is the projector on N >= 2 (exact); at n=2 the diagonal-lint invariants have dimension 8 with weight "
       f"{[round(c, 3) for c in content]} in N = 0..6, contain states of definite N only at N = 3 (dimension 2, with "
       "N_i^(1)+N_i^(2) = 1 for i=1,2,3); none under the spectator action; Thm 26.6's Pi_0 commutes with (-1)^N, not with N, and "
       "has rank exactly half of every N-sector (exact)")

# ----------------------------------------------------------------------------------------------- e2-19 .. e2-21 invariant number
ok = True
LG2V = [np.kron(np.eye(2, dtype=np.int64), A) for A in LGV]
g020 = commutant_in_so(LG2V, ET2)
ok &= len(g020) == 2 and rank_exact(np.array([x.flatten() for x in g020 + [Cv, CGv]])) == 2
a_, b_ = sp.symbols("a b", real=True)
Xs = sp.Matrix(a_ * Cv + b_ * CGv)
sq = (Xs * Xs - sp.eye(12)).applyfunc(sp.expand)
solset = sp.solve(list(set(sq)), [a_, b_], dict=True)
ok &= sorted([(s[a_], s[b_]) for s in solset]) == [(0, -1), (0, 1)]
# eigenspaces of eps(x)G are V_pm = {X2 = pm G X1}, Lagrangian
for sgn in (1, -1):
    W = np.vstack([np.eye(6, dtype=np.int64), sgn * Gv])
    ok &= np.array_equal(CGv @ W, sgn * W)
    ok &= not (W.T @ np.diag(ET2) @ W).any()
evK = np.linalg.eigvalsh(sCG)
ok &= Counter(np.round(evK).astype(int).tolist()) == Counter({k - 3: comb(6, k) for k in range(7)})
LG_S2 = diag_lift_n(LG, g2, 2)
ok &= all(np.array_equal(CG4 @ X, X @ CG4) for X in LG_S2)
report("e2-19", ok, "common binding action (0,2,0) at n=2: its commutant in so(6,6) is so(2,C) = span{eps(x)1, eps(x)G}, whose only "
       "involutions are K = +-eps(x)G (exact, sympy); their eigenspaces are the totally null V_pm = {X2 = +-G X1} of RE Prop. 8.4(ii); "
       "N_K = sigma(eps(x)G)+3 has spectrum 0..6 with multiplicities binomial(6,k) and commutes with every lifted diagonal lG generator: "
       "a Lorentz-invariant particle number at the second rung")

ok = True
invG = nullspace_exact(np.vstack(LG_S2))
WG = np.array(invG).T.astype(float)
QG, _ = np.linalg.qr(WG)
PG = QG @ QG.T
wK, VK = np.linalg.eigh(sCG)
cont_K = [float(np.trace(PG @ VK[:, np.isclose(wK, k - 3)] @ VK[:, np.isclose(wK, k - 3)].T)) for k in range(7)]
ok &= len(invG) == 4 and np.allclose(cont_K, [1, 0, 0, 2, 0, 0, 1], atol=1e-10)
vK = VK[:, 0]
ok &= abs(wK[0] + 3) < 1e-12 and all(np.allclose(X @ vK, 0, atol=1e-12) for X in LG_S2)
contN = [float(np.sum(vK[Ndiag2 == k] ** 2)) for k in range(7)]
ok &= np.allclose(contN, [1 / 8, 0, 3 / 8, 0, 3 / 8, 0, 1 / 8], atol=1e-12)
v = expm(-12.0 * sCG) @ vac2
v /= np.linalg.norm(v)
ok &= abs(abs(v @ vK) - 1) < 1e-9
report("e2-20", ok, "the diagonal-lG invariant states of S_2 have dimension 4, with weight 1, 2, 1 at N_K = 0, 3, 6; the N_K-vacuum is "
       "lG-invariant and has weight 1/8, 3/8, 3/8, 1/8 at N = 0, 2, 4, 6; it is the infinite-rapidity limit of the copy-boosted Fock "
       "vacuum (overlap at zeta = -12 equal to 1 within 1e-9)")

ok = True
# the other four Lorentz actions at n = 2 admit no involution in their commutant
MIXV = [np.block([[A, np.zeros((6, 6), dtype=np.int64)], [np.zeros((6, 6), dtype=np.int64), B]]) for A, B in zip(LINTV, LGV)]
Z6 = np.zeros((6, 6), dtype=np.int64)
act = {
    "(2,0,0)": LINT2V,
    "(1,1,0)": MIXV,
    "(1,0,1)": [np.block([[A, Z6], [Z6, Z6]]) for A in LINTV],
    "(0,1,1)": [np.block([[A, Z6], [Z6, Z6]]) for A in LGV],
}
gdims = {}
for name, gens_ in act.items():
    cm = commutant_in_so(gens_, ET2)
    gdims[name] = len(cm)
    if name in ("(2,0,0)", "(1,1,0)"):
        ok &= all(np.array_equal(X.T, -X) for X in cm)                  # compact: X^2 <= 0, never +1
    elif name == "(1,0,1)":
        ok &= all(not X[:, IDX_E].any() for X in cm)                     # kills the interior four-plane of copy 1
    else:
        ok &= all(not X[:, :6].any() for X in cm)                        # kills the whole of copy 1
ok &= gdims == {"(2,0,0)": 7, "(1,1,0)": 1, "(1,0,1)": 28, "(0,1,1)": 15}
# n = 3: (0,3,0) has commutant so(3,C) (dim 6) of singular elements; (0,2,1) admits an invariant number
LG3V = [np.kron(np.eye(3, dtype=np.int64), A) for A in LGV]
g030 = commutant_in_so(LG3V, ETn[3])
ok &= len(g030) == 6
cs = sp.symbols("c0:6")
copyB = [(rot(i, j, 3)) for i, j in ((0, 1), (0, 2), (1, 2))]
span030 = [np.kron(B, np.eye(6, dtype=np.int64)) for B in copyB] + [np.kron(B, Gv) for B in copyB]
ok &= rank_exact(np.array([x.flatten() for x in g030 + span030])) == 6
Acx = sum((cs[t] * sp.Matrix(copyB[t]) for t in range(3)), sp.zeros(3, 3)) + sp.I * sum((cs[3 + t] * sp.Matrix(copyB[t]) for t in range(3)), sp.zeros(3, 3))
ok &= sp.expand(Acx.det()) == 0
for _ in range(5):
    cvec = rng.integers(-3, 4, size=6)
    X = sum(int(c) * B for c, B in zip(cvec, span030))
    ok &= rank_exact(X) <= 12
K021 = np.zeros((18, 18), dtype=np.int64)
K021[:12, :12] = CGv
K021[12:, 12:] = Kv
L021 = [np.block([[np.kron(np.eye(2, dtype=np.int64), A), np.zeros((12, 6), dtype=np.int64)], [np.zeros((6, 12), dtype=np.int64), Z6]]) for A in LGV]
ok &= np.array_equal(K021 @ K021, np.eye(18, dtype=np.int64)) and in_so(K021, ETn[3]) and all(np.array_equal(comm(K021, X), 0 * X) for X in L021)
# n = 4: (0,4,0) admits K = (eps+eps)(x)G
A4 = np.kron(np.eye(2, dtype=np.int64), EPS)
K040 = np.kron(A4, Gv)
LG4V = [np.kron(np.eye(4, dtype=np.int64), A) for A in LGV]
ok &= np.array_equal(K040 @ K040, np.eye(24, dtype=np.int64)) and in_so(K040, ETn[4])
ok &= all(np.array_equal(comm(K040, X), 0 * X) for X in LG4V) and np.array_equal(comm(K040, Gn[4]), 0 * Gn[4])
report("e2-21", ok, f"no invariant number under the other actions at n=2: commutants of dimension {gdims} (Thm 10.8) are compact for "
       "(2,0,0), (1,1,0) and vanish on the Lorentz-carrying subspace of copy 1 for (1,0,1), (0,1,1); at n=3 the commutant of (0,3,0) is "
       "so(3,C) (dim 6) whose copy matrix B+iB' is complex antisymmetric with det 0 (sympy; integer ranks <= 12), while (0,2,1) admits "
       "K = (eps(x)G) + K; at n=4 (0,4,0) admits K = ((eps+eps)(x)G) (exact)")

# spinor level at n = 4 (sparse): the lift of K_(0,4,0) commutes with every lifted diagonal lG generator
g4 = tower_sparse(4)
ET4 = ETn[4]


def lift4_sparse(A, gens, etas):
    out = None
    for a in range(A.shape[0]):
        for b in range(A.shape[1]):
            if A[a, b] != 0:
                t = (int(A[a, b]) * etas[b]) * (gens[a] @ gens[b])
                out = t if out is None else out + t
    return out.tocsr()


def ad_sparse_ok(L, A, gens):
    """check [L, E_c] = 4 sum_a A_ac E_a for all c (L = 4 sigma(A))."""
    for c in range(len(gens)):
        lhs = L @ gens[c] - gens[c] @ L
        rhs = None
        for a in range(len(gens)):
            if A[a, c] != 0:
                t = (4 * int(A[a, c])) * gens[a]
                rhs = t if rhs is None else rhs + t
        diff = lhs - (rhs if rhs is not None else 0 * lhs)
        diff.eliminate_zeros()
        if diff.nnz:
            return False
    return True


ok = True
LK4 = lift4_sparse(K040, g4, ET4)
ok &= ad_sparse_ok(LK4, K040, g4)
for A in LG4V:
    LA = lift4_sparse(A, g4, ET4)
    D = LK4 @ LA - LA @ LK4
    D.eliminate_zeros()
    ok &= D.nnz == 0
diagK = LK4 @ LK4
report("e2-22", ok, "spinor level at n=4 (S_4 of dimension 4096, sparse integer matrices): sigma(K_(0,4,0)) has ad = K_(0,4,0) on all 24 "
       "generators and commutes with the six lifted diagonal lG generators: the invariant number exists on the fourth rung (exact)")

# ----------------------------------------------------------------------------------------------- e2-23 .. e2-25 copy rotation
ok = True
Cb2 = -sum(ET2[a] * g2[a] @ g2[6 + a] for a in range(6))        # 2 C_b (task normalisation C_b = -1/2 sum eta E1a E2a)
ok &= np.array_equal(Cb2, lift4(Cv, g2, ET2) // 2) and np.array_equal(advec(Cb2, g2, ET2), 2 * Cv)
hop = sum(ap2(1, i) @ am2(0, i) - ap2(0, i) @ am2(1, i) for i in range(3))   # 4 sum (a+_2 a-_1 - a+_1 a-_2)
ok &= np.array_equal(2 * Cb2, hop)
ok &= np.array_equal(Cb2 @ twoN2, twoN2 @ Cb2) and not np.array_equal(Cb2 @ twoN1_2, twoN1_2 @ Cb2)
one = [ap2(k, i) @ vac2.astype(np.int64) // 2 for k in range(2) for i in range(3)]
for i in range(3):
    ok &= np.array_equal(Cb2 @ one[i], 2 * one[3 + i]) and np.array_equal(Cb2 @ one[3 + i], -2 * one[i])
Cb = Cb2 / 2.0
for th in (0.3, np.pi / 2, 2.0):
    U = expm(th * Cb)
    v = U @ one[0]
    ok &= np.allclose(v, np.cos(th) * one[0] + np.sin(th) * one[3], atol=1e-12)
    for k in range(7):
        for j in range(7):
            if j != k:
                ok &= np.max(np.abs(U[np.ix_(Ndiag2 == j, Ndiag2 == k)])) < 1e-12
ok &= np.allclose(expm(2 * np.pi * Cb), np.eye(64), atol=1e-10)
report("e2-23", ok, "the copy rotation C_b = -1/2 sum eta_aa E_(1,a)E_(2,a) is the lift of eps(x)1 and equals sum_i (a+_(2,i)a-_(1,i) - "
       "a+_(1,i)a-_(2,i)) (exact); it commutes with N but not with N^(1); C_b a+_(1,i)|0> = a+_(2,i)|0>, C_b a+_(2,i)|0> = -a+_(1,i)|0>; "
       "exp(theta C_b) a+_(1,1)|0> = cos(theta) a+_(1,1)|0> + sin(theta) a+_(2,1)|0> (full transfer at pi/2) and exp(theta C_b) has no "
       "matrix element between different N (tol 1e-12): one quantum is moved, never doubled")

ok = True
C12v = np.kron(rot(0, 1, 3), np.eye(6, dtype=np.int64))
C23v = np.kron(rot(1, 2, 3), np.eye(6, dtype=np.int64))
B12v = np.kron(rot(0, 1, 3), Gv)
ok &= all(np.array_equal(comm(X, Kn[3]), 0 * X) for X in (C12v, C23v)) and np.array_equal(B12v @ Kn[3], -(Kn[3] @ B12v))
twoNk3, twoN3 = number_ops(g3, 3)
L12 = lift4(C12v, g3, ETn[3])
L23 = lift4(C23v, g3, ETn[3])
ok &= np.array_equal(L12 @ twoN3, twoN3 @ L12) and np.array_equal(L23 @ twoN3, twoN3 @ L23)
report("e2-24", ok, "up the ladder: at n=3 the copy rotations commute with K_3 and their lifts with N (hopping among three rungs); "
       "the copy boosts anticommute with K_3 (pure pair creation across rungs), exact")

ok = True
k1, k2, beta = 1.0, 3.0, 0.4
HF = k1 * twoN1_2 / 2 + k2 * twoN2_2 / 2 + 1j * beta * Cb
ok &= np.allclose(HF, HF.conj().T) and np.allclose(HF @ (twoN2 / 2), (twoN2 / 2) @ HF)
s_rabi = 4 * beta ** 2 / ((k2 - k1) ** 2 + 4 * beta ** 2)
Om = np.sqrt((k2 - k1) ** 2 + 4 * beta ** 2)
ts = np.linspace(0, 2 * np.pi / Om, 41)
Pmax, err, leak = 0.0, 0.0, 0.0
psi0 = one[0].astype(complex)
for t in ts:
    psi = expm(-1j * HF * t) @ psi0
    P12 = sum(abs(np.vdot(one[3 + i], psi)) ** 2 for i in range(3))
    err = max(err, abs(P12 - s_rabi * np.sin(Om * t / 2) ** 2))
    leak = max(leak, float(np.sum(np.abs(psi[Ndiag2 != 1]) ** 2)))
    Pmax = max(Pmax, P12)
HFe = 2.0 * (twoN2 / 2) + 1j * beta * Cb
psi = expm(-1j * HFe * (np.pi / (2 * beta))) @ psi0
full = sum(abs(np.vdot(one[3 + i], psi)) ** 2 for i in range(3))
ok &= err < 1e-12 and leak < 1e-24 and abs(Pmax - s_rabi) < 1e-12 and abs(full - 1) < 1e-12
sector10 = int(np.sum((np.diag(twoN1_2) == 2) & (np.diag(twoN2_2) == 0)))
ok &= sector10 == 3
report("e2-25", ok, f"Rabi transfer of one quantum (external i): H = k1 N^(1) + k2 N^(2) + i beta C_b is Hermitian and commutes with N; from "
       f"a+_(1,1)|0> the rung-2 probability is s sin^2(Omega t/2), s = 4beta^2/((k2-k1)^2+4beta^2) = {s_rabi:.5f} at (1,3,0.4) "
       f"(error {err:.1e}), the weight outside N=1 stays below 1e-24, and at equal rates the transfer is complete at t = pi/(2 beta); the sector N^(1)=1, N^(2)=0 has odd real dimension 3, so no real complex structure preserves it and the i is external")

print()
if FAILED:
    print(f"FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("all e2 checks passed")
sys.exit(0)
