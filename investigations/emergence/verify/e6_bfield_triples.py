#!/usr/bin/env python3
"""Track e6 -- The b-field on the temporal triple, c on the spatial triple, and the mass form T = e1e2e3.

Verification script for investigations/emergence/sections/e6_bfield_triples.tex.

    python3 e6_bfield_triples.py        (from any directory)

Prints one 'PASS e6-<id>: <claim>' (or FAIL) line per claim and exits 0 only if every check passes.

Conventions (tex/s02_arena.tex eq. (1)-(2), tex/sA_realisation.tex; RE sections d1-d7; emergence sections e1, e2, e5):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1); on R^8: e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1,
  e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep; omega = e1...e6 = Gamma.  T = omega_T = e1e2e3, omega_S = e4e5e6, J = e2e3,
  Omega = e1e4e5e6, gamma^mu = (e1, e4, e5, e6), K^mu = -gamma^0 gamma^mu.  Tower: E_(k,a) = omega^{(x)(k-1)} (x) e_a (x) 1....
  G e_i = e_{i+3}, G e_{i+3} = -e_i; K e_i = e_{i+3}, K e_{i+3} = e_i.  Fock modes a+-_{k,i} = (E_(k,i) +- E_(k,i+3))/2,
  N = sum_{k,i} (1 - E_(k,i)E_(k,i+3))/2.  Copy rotation C_b = -(1/2) sum_a eta_aa E_(1,a)E_(2,a) = C_+ + C_-, with
  C_+ = -(1/2) sum_{i<=3} E_(1,i)E_(2,i) (temporal part) and C_- = +(1/2) sum_{a>=4} E_(1,a)E_(2,a) (spatial part).
  A bivector ab acts on vectors by v -> (1/2)[ab, v].

Exactness: every algebraic identity is checked in integer or rational arithmetic (signed-permutation matrices; sympy for
symbolic identities and exact null spaces).  Spin rotors at rational points use (cos phi, sin phi) = (3/5, 4/5) and are
stored as integer matrices times a known power of 5.  Invariant bilinear forms on S_n are counted exactly: for a Lie algebra
spanned by blades, by signed orbits of the matrix entries; for the diagonal so(3,3), by the Weyl-group multiplicity
formula on exact integer weights.  Floating point is used only in e6-7d's dense cross-check at n = 2 (none: exact) and
nowhere else.  Runtime: well under a minute.
"""
import itertools
import sys
import time
from collections import Counter
from fractions import Fraction

import numpy as np
import sympy as sp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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


# ============================================================================== one constituent (integer matrices)
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def kr(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = np.kron(out, m)
    return out


e = [None, kr(sx, I2, I2), kr(sz, sx, I2), kr(sz, sz, sx), kr(ep, I2, I2), kr(sz, ep, I2), kr(sz, sz, ep)]
ETA = [0, 1, 1, 1, -1, -1, -1]
I8 = np.eye(8, dtype=np.int64)
Z8 = np.zeros((8, 8), dtype=np.int64)
om = e[1] @ e[2] @ e[3] @ e[4] @ e[5] @ e[6]
T = e[1] @ e[2] @ e[3]
OS = e[4] @ e[5] @ e[6]
J = e[2] @ e[3]
Om = e[1] @ e[4] @ e[5] @ e[6]
GAM = [e[1], e[4], e[5], e[6]]
KIN = [-(e[1] @ g) for g in GAM]              # K^mu = -gamma^0 gamma^mu
TEMP_ROT = {(2, 3): e[2] @ e[3], (3, 1): e[3] @ e[1], (1, 2): e[1] @ e[2]}
COMPACT = [(a, b) for a, b in itertools.combinations(range(1, 7), 2) if (a <= 3) == (b <= 3)]
MIXED = [(a, b) for a, b in itertools.combinations(range(1, 7), 2) if (a <= 3) != (b <= 3)]


def comm(A, B):
    return A @ B - B @ A


def acomm(A, B):
    return A @ B + B @ A


def eq(A, B):
    return np.array_equal(A, B)


def biv(a, b):
    return e[a] @ e[b]


def rank_exact(A):
    """exact rank of an integer matrix (fraction-free elimination over ZZ)"""
    from sympy.polys.matrices import DomainMatrix
    from sympy import ZZ
    A = np.asarray(A, dtype=np.int64)
    if A.shape[0] > A.shape[1]:
        A = A.T
    A = A[:, np.any(A != 0, axis=0)]
    if A.size == 0:
        return 0
    rows = [[ZZ(int(x)) for x in r] for r in A.tolist()]
    return DomainMatrix(rows, A.shape, ZZ).rank()


def nullspace_Q(rows, ncols):
    """exact rational null space of a list of integer/rational rows"""
    if not rows:
        return [sp.eye(ncols)[:, i] for i in range(ncols)]
    M = sp.Matrix(rows)
    return M.nullspace()


# vector level (6 x 6)
Zs3 = sp.zeros(3)
Is3 = sp.eye(3)
ETAs = sp.diag(1, 1, 1, -1, -1, -1)
Gs = sp.Matrix(sp.BlockMatrix([[Zs3, -Is3], [Is3, Zs3]]))
Ks = sp.Matrix(sp.BlockMatrix([[Zs3, Is3], [Is3, Zs3]]))


def cross(v):
    v = sp.Matrix(v)
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def blk(A, B):
    return sp.Matrix(sp.BlockMatrix([[A, sp.zeros(A.shape[0], B.shape[1])], [sp.zeros(B.shape[0], A.shape[1]), B]]))


def G_C(C):
    return sp.Matrix(sp.BlockMatrix([[Zs3, -C.inv()], [C, Zs3]]))


def cayley(v):
    """rational rotation matrix (Cayley transform of [v]x)"""
    A = cross(v)
    return (Is3 + A) * (Is3 - A).inv()


def dagger(F, eta=ETAs):
    return eta.inv() * F.T * eta


# ------------------------------------------------------------------------------------------------ conventions
@claim("e6-0", "conventions: e_a satisfy the Clifford relations of R^{3,3}; e_i^T = e_i (temporal), e_{i+3}^T = -e_{i+3}; "
       "T = e1e2e3 = J e1 = e1 J has T^2 = -1, T^T = -T; omega_S = e4e5e6 has omega_S^2 = 1, omega_S^T = omega_S; T omega_S = omega = Gamma = J Omega; "
       "the Fock modes obey the CAR and N has spectrum 0..3 with multiplicities 1,3,3,1; G^2 = -1 is eta-self-adjoint, K^2 = 1 is eta-skew")
def _():
    ok = all(eq(acomm(e[a], e[b]), 2 * ETA[a] * (a == b) * I8) for a in range(1, 7) for b in range(1, 7))
    ok &= all(eq(e[a].T, ETA[a] * e[a]) for a in range(1, 7))
    ok &= eq(T, J @ e[1]) and eq(T, e[1] @ J) and eq(T @ T, -I8) and eq(T.T, -T)
    ok &= eq(OS @ OS, I8) and eq(OS.T, OS) and eq(T @ OS, om) and eq(om, J @ Om)
    a2p = {i: e[i] + e[i + 3] for i in (1, 2, 3)}   # 2 a+
    a2m = {i: e[i] - e[i + 3] for i in (1, 2, 3)}   # 2 a-
    ok &= all(eq(acomm(a2p[i], a2m[j]), 4 * (i == j) * I8) and eq(acomm(a2p[i], a2p[j]), Z8) and eq(acomm(a2m[i], a2m[j]), Z8)
              for i in (1, 2, 3) for j in (1, 2, 3))
    N2 = sum((I8 - e[i] @ e[i + 3]) for i in (1, 2, 3))       # 2N
    d = np.diag(N2) // 2
    ok &= eq(N2, np.diag(np.diag(N2))) and Counter(d.tolist()) == {0: 1, 1: 3, 2: 3, 3: 1}
    ok &= Gs * Gs == -sp.eye(6) and dagger(Gs) == Gs and Ks * Ks == sp.eye(6) and dagger(Ks) == -Ks
    return ok


# ======================================================================================== 1. T as form and operator
@claim("e6-1a", "T as a form: Y^T T + T Y = 0 for all fifteen bivectors, and Y^T T Y = -Y^2 T, so (e^{sY})^T T e^{sY} = T for every s "
       "(restates ES17 Cor. 18.4 and RE d2:prop:Tinv(i)); the same holds for omega_S = e4e5e6, the symmetric invariant pairing of ES17 Sec. 28")
def _():
    ok = True
    for a, b in itertools.combinations(range(1, 7), 2):
        Y = biv(a, b)
        Y2 = (Y @ Y)[0, 0]
        ok &= eq(Y @ Y, Y2 * I8) and eq(Y.T @ T + T @ Y, Z8) and eq(Y.T @ T @ Y, -Y2 * T)
        ok &= eq(Y.T @ OS + OS @ Y, Z8) and eq(Y.T @ OS @ Y, -Y2 * OS)
    return ok


@claim("e6-1b", "T as an operator commutes with exactly the six compact bivectors (so(3)_+ + so(3)_-) and anticommutes with the nine "
       "mixed ones; its commutant in M_8(R) has dimension 32, is spanned by the 32 blades with an even number of spatial indices, equals "
       "Cl(V+) Cl^0(V-) with Cl^0(V-) = H commuting with Cl(V+), has centre span{1, T} (so it is M_4(C)), and meets so(3,3) in so(3)_+ + so(3)_-; "
       "the 32 blades with an odd number of spatial indices anticommute with T")
def _():
    ok = all(eq(comm(T, biv(a, b)), Z8) for a, b in COMPACT) and all(eq(acomm(T, biv(a, b)), Z8) for a, b in MIXED)
    # exact commutant dimension: kernel of X -> TX - XT on M_8(R)
    L = np.kron(T, I8) - np.kron(I8, T.T)            # acts on row-major vec(X)
    rank = rank_exact(L)
    ok &= (64 - rank) == 32
    blades = {}
    for mask in range(64):
        M = I8.copy()
        for a in range(6):
            if mask >> a & 1:
                M = M @ e[a + 1]
        blades[mask] = M
    nsp = lambda m: bin(m >> 3).count('1')
    com = [m for m in range(64) if eq(comm(T, blades[m]), Z8)]
    anti = [m for m in range(64) if eq(acomm(T, blades[m]), Z8)]
    ok &= sorted(com) == sorted(m for m in range(64) if nsp(m) % 2 == 0) and len(com) == 32
    ok &= sorted(anti) == sorted(m for m in range(64) if nsp(m) % 2 == 1)
    # Cl^0(V-) is the quaternions and commutes with Cl(V+)
    qi, qj = e[4] @ e[5], e[5] @ e[6]
    qk = qi @ qj
    ok &= eq(qi @ qi, -I8) and eq(qj @ qj, -I8) and eq(qk @ qk, -I8)
    ok &= all(eq(comm(e[i], q), Z8) for i in (1, 2, 3) for q in (qi, qj))
    # products of the 8 temporal blades and the 4 even spatial blades give the 32 commuting blades (up to sign)
    temp = [m for m in range(64) if m >> 3 == 0]
    evsp = [m for m in range(64) if m & 7 == 0 and nsp(m) % 2 == 0]
    ok &= sorted(t | s for t in temp for s in evsp) == sorted(com)
    # centre of the commutant: blades in com commuting with all of com
    centre = [m for m in com if all(eq(comm(blades[m], blades[k]), Z8) for k in com)]
    ok &= sorted(centre) == [0, 7]
    # intersection with so(3,3): bivector blades in com
    ok &= sorted(m for m in com if bin(m).count('1') == 2) == sorted((1 << (a - 1)) | (1 << (b - 1)) for a, b in COMPACT)
    return ok


@claim("e6-1c", "group level: e^{sY} commutes with T for compact Y; for a mixed boost Y, e^{sY} T e^{-sY} = e^{2sY} T (checked as Y T Y^{-1} = -T); "
       "Ad_T on V is eta (restates RE d1:prop:Tparity(i))")
def _():
    ok = all(eq(biv(a, b) @ T @ biv(a, b), -T) for a, b in MIXED)    # Y^{-1} = Y for boosts
    ok &= all(eq(T @ e[a] @ (-T), ETA[a] * e[a]) for a in range(1, 7))   # T^{-1} = -T
    return ok


# ======================================================================================== 2. the Dirac data
a_, b_, g_ = sp.symbols('alpha beta gamma', real=True)
Ysym = a_ * sp.Matrix(J.tolist()) + b_ * sp.Matrix((e[3] @ e[1]).tolist()) + g_ * sp.Matrix((e[1] @ e[2]).tolist())


@claim("e6-2a", "kinetic forms: Y = alpha J + beta e3e1 + gamma e1e2 in so(3)_+ satisfies Y^T K^mu + K^mu Y = Lambda^mu_nu K^nu for some Lambda iff "
       "beta = gamma = 0 (so(3)_+ meets lint + RJ in RJ, from ES17/RE mb:Tinv(ii)); explicitly Y^T K^0 + K^0 Y = 0 and "
       "Y^T K^a + K^a Y = -2 Y(e1) gamma^a, with Y(e1) = (1/2)[Y, e1] = beta e3 - gamma e2 the rotated clock, so "
       "sum_a ||Y^T K^a + K^a Y||^2 = 96 (beta^2 + gamma^2)")
def _():
    Ks_ = [sp.Matrix(k.tolist()) for k in KIN]
    ok = True
    v = b_ * sp.Matrix(e[3].tolist()) - g_ * sp.Matrix(e[2].tolist())
    ok &= sp.simplify((Ysym * sp.Matrix(e[1].tolist()) - sp.Matrix(e[1].tolist()) * Ysym) / 2 - v) == sp.zeros(8)
    for mu in range(4):
        lhs = Ysym.T * Ks_[mu] + Ks_[mu] * Ysym
        target = sp.zeros(8) if mu == 0 else -2 * v * sp.Matrix(GAM[mu].tolist())
        ok &= sp.expand(lhs - target) == sp.zeros(8)
    tot = sum(sum(x ** 2 for x in (Ysym.T * Ks_[mu] + Ks_[mu] * Ysym)) for mu in range(1, 4))
    ok &= sp.expand(tot - 96 * (b_ ** 2 + g_ ** 2)) == 0
    # covariance system: unknowns (alpha, beta, gamma, Lambda 16)
    Lam = sp.symbols('L0:16')
    eqs = []
    for mu in range(4):
        lhs = Ysym.T * Ks_[mu] + Ks_[mu] * Ysym - sum((Lam[4 * mu + nu] * Ks_[nu] for nu in range(4)), sp.zeros(8))
        eqs += list(lhs)
    unknowns = [a_, b_, g_] + list(Lam)
    A, _b = sp.linear_eq_to_matrix(eqs, unknowns)
    ns = A.nullspace()
    proj = sp.Matrix.hstack(*[v_[:3, 0] for v_ in ns]) if ns else sp.zeros(3, 0)
    ok &= proj.rank() == 1 and all(proj[1, i] == 0 and proj[2, i] == 0 for i in range(proj.shape[1]))
    return ok


def rotor5(B):
    """5 * exp(phi B) with cos phi = 3/5, sin phi = 4/5, for B^2 = -1 (integer matrix); inverse is 5 * exp(-phi B)"""
    Id = np.eye(B.shape[0], dtype=np.int64)
    return 3 * Id + 4 * B, 3 * Id - 4 * B


@claim("e6-2b", "rest Hamiltonian: for R = exp(phi e1e2), R e1 R^{-1} = cos2phi e1 - sin2phi e2 =: n (and for e1e3, cos2phi e1 - sin2phi e3), exact at "
       "(cos phi, sin phi) = (3/5, 4/5); R J R^{-1} = J(n) = T n, n J(n) = J(n) n = T, R K^mu R^{-1} = -n gamma^mu, R T R^{-1} = T: the clock rotation "
       "carries the Dirac data at e1 to those at n (TR Prop. 2.4, Thm 2.5); the stabiliser of H(0) = m e1 in so(3)_+ is RJ")
def _():
    ok = True
    for (B, other) in ((e[1] @ e[2], e[2]), (e[1] @ e[3], e[3])):
        R5, Ri5 = rotor5(B)
        n25 = R5 @ e[1] @ Ri5                       # 25 * n
        ok &= eq(n25, -7 * e[1] - 24 * other)
        ok &= eq(n25 @ n25, 625 * I8)
        Jn25 = R5 @ J @ Ri5
        ok &= eq(Jn25, T @ n25) and eq(n25 @ Jn25, 625 * T) and eq(Jn25 @ n25, 625 * T)
        ok &= eq(R5 @ T @ Ri5, 25 * T)
        ok &= all(eq(R5 @ KIN[mu] @ Ri5, -(n25 @ GAM[mu]) if mu > 0 else -25 * I8) for mu in range(4))
    # stabiliser of e1: [Y, e1] = 0 iff beta = gamma = 0
    Cm = Ysym * sp.Matrix(e[1].tolist()) - sp.Matrix(e[1].tolist()) * Ysym
    sol = sp.solve(list(Cm), [a_, b_, g_], dict=True)
    ok &= sol == [{b_: 0, g_: 0}]
    return ok


@claim("e6-2c", "rest flow: e^{-m lambda T} commutes with Spin(3)_+ x Spin(3)_-; the spin lift (1/2)e_je_k of so(3)_+ has Casimir -3/4 on all of S "
       "(isospin 1/2) and exp(pi e_je_k) = -1; the clock rotations e1e2, e3e1 anticommute with J and commute with T, and the half-turn Ad_{e1e2} = Ad_{e1}Ad_{e2} "
       "maps (e1, J, T, gamma^a) to (-e1, -J, T, gamma^a); on the field octants -T Phi = Phi G (TR Prop. 3.12(i)), so e^{-m lambda T} Phi = Phi e^{m lambda G}")
def _():
    ok = all(eq(comm(T, biv(a, b)), Z8) for a, b in COMPACT)
    cas4 = sum(Y @ Y for Y in TEMP_ROT.values())          # 4 * Casimir of (1/2) e_je_k
    ok &= eq(cas4, -3 * I8)
    ok &= all(eq(Y @ Y, -I8) for Y in TEMP_ROT.values())   # exp(pi Y) = cos pi + sin pi Y = -1
    for Y in (e[1] @ e[2], e[3] @ e[1]):
        ok &= eq(acomm(Y, J), Z8) and eq(comm(Y, T), Z8)
    Y = e[1] @ e[2]
    Yi = -Y
    ok &= eq(Y @ e[1] @ Yi, -e[1]) and eq(Y @ J @ Yi, -J) and eq(Y @ T @ Yi, T) and all(eq(Y @ e[a] @ Yi, e[a]) for a in (4, 5, 6))
    ok &= all(eq(e[1] @ (e[2] @ e[a] @ e[2]) @ e[1], Y @ e[a] @ Yi) for a in range(1, 7))
    # field-octant map Phi (TR Thm 3.7): Phi(X) = sum E_k f_k + sum B_m g_m, f_k = a+_k|vac>, g_m = a+_j a+_k |vac>
    N2 = sum((I8 - e[i] @ e[i + 3]) for i in (1, 2, 3))
    vac = np.zeros(8, dtype=np.int64)
    vac[int(np.argmin(np.diag(N2)))] = 1
    ap2 = {i: e[i] + e[i + 3] for i in (1, 2, 3)}          # 2 a+
    f = {k: (ap2[k] @ vac) // 2 for k in (1, 2, 3)}
    g = {1: (ap2[2] @ ap2[3] @ vac) // 4, 2: (ap2[3] @ ap2[1] @ vac) // 4, 3: (ap2[1] @ ap2[2] @ vac) // 4}
    ok &= all(eq(ap2[k] @ vac, 2 * f[k]) for k in (1, 2, 3))
    Phi = np.zeros((8, 6), dtype=np.int64)
    for m in (1, 2, 3):
        Phi[:, m - 1] = g[m]
        Phi[:, m + 2] = f[m]
    Gi = np.array(Gs.tolist(), dtype=np.int64)
    ok &= eq(-T @ Phi, Phi @ Gi) and eq(Phi.T @ Phi, np.eye(6, dtype=np.int64))
    return ok


@claim("e6-2d", "the two su(2)'s: so(3)_+ = span{J, e3e1, e1e2} meets g_massless = span{Omega, J, e2Omega, e3Omega} (ES17 Prop. 5.5) in RJ; "
       "e2Omega = -(e1e2) omega_S and e3Omega = (e3e1) omega_S with omega_S commuting with so(3)_+; with P+- = (1 +- omega_S)/2 (rank 4), "
       "su(2)_PG = {Y P+ + (J Y J^{-1}) P- : Y in so(3)_+}; the Lie algebra generated by both is so(3)_+ P+ + so(3)_+ P- (dim 6, = so(4)); "
       "g = P+ + J P- is orthogonal, conjugates so(3)_+ onto su(2)_PG, and has components of grades 0, 2, 3, 5 (not in Pin(3,3))")
def _():
    R3 = [J, e[3] @ e[1], e[1] @ e[2]]
    PG = [J, e[2] @ Om, e[3] @ Om]
    gm = [Om, J, e[2] @ Om, e[3] @ Om]
    vecs = lambda L: sp.Matrix([list(x.flatten()) for x in L]).T
    ok = vecs(R3 + gm).rank() == 3 + 4 - 1
    ok &= eq(e[2] @ Om, -(e[1] @ e[2]) @ OS) and eq(e[3] @ Om, (e[3] @ e[1]) @ OS)
    ok &= all(eq(comm(OS, Y), Z8) for Y in R3)
    Pp2, Pm2 = I8 + OS, I8 - OS                              # 2 P+, 2 P-
    ok &= sp.Matrix(Pp2.tolist()).rank() == 4 and eq(Pp2 @ Pp2, 2 * Pp2)
    Jinv = -J
    for Y, Ypg in zip([J, -(e[1] @ e[2]), e[3] @ e[1]], [J, e[2] @ Om, e[3] @ Om]):
        ok &= eq(Y @ Pp2 + (J @ Y @ Jinv) @ Pm2, 2 * Ypg)
    # generated Lie algebra
    basis = []

    def add(M):
        cand = basis + [M]
        if sp.Matrix([list(x.flatten()) for x in cand]).rank() > len(basis):
            basis.append(M)
            return True
        return False
    for x in R3 + PG:
        add(x)
    changed = True
    while changed:
        changed = False
        for x in list(basis):
            for y in list(basis):
                if add(comm(x, y)):
                    changed = True
    ok &= len(basis) == 6
    target = [Y @ Pp2 for Y in R3] + [Y @ Pm2 for Y in R3]
    ok &= vecs(basis + target).rank() == 6 and vecs(target).rank() == 6
    g2 = Pp2 + J @ Pm2                                       # 2 g
    ok &= eq(g2.T @ g2, 4 * I8)
    ginv2 = Pp2 - J @ Pm2                                    # 2 g^{-1}
    ok &= eq(g2 @ ginv2, 4 * I8)
    ok &= all(eq(g2 @ Y @ ginv2, 4 * Ypg) for Y, Ypg in zip([J, -(e[1] @ e[2]), e[3] @ e[1]], [J, e[2] @ Om, e[3] @ Om]))
    # 2g = 1 + J + omega_S - J omega_S : grades 0, 2, 3, 5
    ok &= eq(g2, I8 + J + OS - J @ OS)
    return ok


# ======================================================================================== 3. temporal b and the binding root
r1, r2, r3 = sp.symbols('r1 r2 r3', real=True)
kap, w_ = sp.symbols('kappa omega', positive=True)
lam = sp.symbols('lambda')


@claim("e6-3a", "b = (R, 0) on V+ + V-, R = [r]x: b = (1/2)(R, R) + (1/2)(R, -R) with [(R,R), G] = 0 and {(R,-R), G} = 0; in the notation of RE eq. (11) "
       "b = diag(A + C, A - C) with A = C = R/2 (a = c = r/2), so a.c = |r|^2/4; [b, G] = 2 b_p G = (R,-R)G")
def _():
    R = cross([r1, r2, r3])
    b = blk(R, Zs3)
    bl, bp = blk(R, R) / 2, blk(R, -R) / 2
    ok = sp.simplify(b - bl - bp) == sp.zeros(6)
    ok &= sp.expand(bl * Gs - Gs * bl) == sp.zeros(6) and sp.expand(bp * Gs + Gs * bp) == sp.zeros(6)
    ok &= sp.expand(b * Gs - Gs * b - 2 * bp * Gs) == sp.zeros(6)
    ok &= dagger(b) == -b
    return ok


@claim("e6-3b", "conjugation: diag(Q, 1) G diag(Q, 1)^{-1} = G_{Q^{-1}} for a temporal rotation Q (exact for Cayley rotations), so e^{theta b} G e^{-theta b} = G_{C(theta)}, "
       "C(theta) = e^{-theta R}; G_C is eta-self-adjoint with (kappa G_C)^2 = -kappa^2 iff C is orthogonal, so the rates (spectrum +-i kappa of the self-adjoint part) "
       "are unchanged (RE d7:thm:schr(iii)); [b, G] != 0, so b is not removable along a worldline (RE d4:prop:remove)")
def _():
    ok = True
    for v in ([1, 2, 3], [sp.Rational(1, 2), 0, sp.Rational(1, 3)], [0, 0, 1]):
        Q = cayley(v)
        ok &= sp.simplify(Q.T * Q - Is3) == sp.zeros(3)
        g = blk(Q, Is3)
        ok &= sp.simplify(g * Gs * g.inv() - G_C(Q.inv())) == sp.zeros(6)
        GC = G_C(Q.inv())
        ok &= sp.simplify(dagger(GC) - GC) == sp.zeros(6) and sp.simplify(GC * GC + sp.eye(6)) == sp.zeros(6)
        ok &= sp.expand((lam * sp.eye(6) - kap * GC).det() - (lam ** 2 + kap ** 2) ** 3) == 0
    Cn = sp.diag(1, 2, 3)
    ok &= sp.simplify(dagger(G_C(Cn)) - G_C(Cn)) != sp.zeros(6)
    R = cross([0, 0, 1])
    ok &= blk(R, Zs3) * Gs - Gs * blk(R, Zs3) != sp.zeros(6)
    return ok


@claim("e6-3c", "lines: for S = b + kappa G with b = (R,0), |r| = omega, det(lambda - S) = (lambda^2 + kappa^2)(lambda^4 + (2kappa^2 + omega^2) lambda^2 + kappa^4) "
       "(symbolic in r); the lines are kappa and s +- omega/2, s = (kappa^2 + omega^2/4)^{1/2}, whose product is kappa^2; at (kappa, omega) = (2, 3) they are 1, 2, 4")
def _():
    R = cross([r1, r2, r3])
    S = blk(R, Zs3) + kap * Gs
    p = sp.expand((lam * sp.eye(6) - S).det())
    om2 = r1 ** 2 + r2 ** 2 + r3 ** 2
    ok = sp.expand(p - (lam ** 2 + kap ** 2) * (lam ** 4 + (2 * kap ** 2 + om2) * lam ** 2 + kap ** 4)) == 0
    s = sp.sqrt(kap ** 2 + w_ ** 2 / 4)
    q = lam ** 4 + (2 * kap ** 2 + w_ ** 2) * lam ** 2 + kap ** 4
    ok &= all(sp.simplify(q.subs(lam, sp.I * x)) == 0 for x in (s + w_ / 2, s - w_ / 2))
    ok &= sp.simplify((s + w_ / 2) * (s - w_ / 2) - kap ** 2) == 0
    S23 = blk(3 * cross([0, 0, 1]), Zs3) + 2 * Gs
    ok &= sp.expand((lam * sp.eye(6) - S23).det() - (lam ** 2 + 1) * (lam ** 2 + 4) * (lam ** 2 + 16)) == 0
    return ok


def jordan_profile(S, m=1):
    """geometric and algebraic multiplicities of the eigenvalues of M = [[-S, 1/m],[0, S^dagger]] (exact)"""
    n = S.shape[0]
    eta = ETAs if n == 6 else sp.diag(*([1, 1, 1, -1, -1, -1] * (n // 6)))
    M = sp.Matrix(sp.BlockMatrix([[-S, sp.eye(n) / m], [sp.zeros(n), dagger(S, eta)]]))
    out = {}
    for ev, mult in sp.roots(M.charpoly(lam).as_expr(), lam).items():
        A = M - ev * sp.eye(2 * n)
        out[ev] = (2 * n - A.rank(), mult)
    return out


@claim("e6-3d", "the temporal b-field tears the binding: for (kappa, omega) = (2, 3) and (6, 5) the generator M of RE d4:lem:M has Jordan blocks exactly at the two "
       "split lines s +- omega/2 (geometric multiplicity 1, algebraic 2 at each sign) and is semisimple at kappa, as RE d4:thm:screen(i) requires (a.c = omega^2/4 != 0)")
def _():
    ok = True
    for (k, w, lines) in ((2, 3, (1, 4)), (6, 5, (4, 9))):
        S = blk(w * cross([0, 0, 1]), Zs3) + k * Gs
        prof = jordan_profile(S)
        for ev, (geo, alg) in prof.items():
            mag = abs(sp.im(ev))
            if mag in lines:
                ok &= (geo, alg) == (1, 2)
            elif mag == k:
                ok &= geo == alg == 2
            else:
                ok = False
        ok &= len(prof) == 6
    return ok


@claim("e6-3e", "on spinors a temporal rotation e_je_k is half a paired rotation and half a counter-rotation: e_je_k = (1/2)(e_je_k - e_{j+3}e_{k+3}) + "
       "(1/2)(e_je_k + e_{j+3}e_{k+3}); the first commutes with N (emergence e1:thm:Nrule), the second changes N by +-2 only, so e_je_k does not commute with N")
def _():
    N2 = sum((I8 - e[i] @ e[i + 3]) for i in (1, 2, 3))
    dN = (np.diag(N2)[:, None] - np.diag(N2)[None, :]) // 2
    ok = True
    for (j, k) in ((2, 3), (3, 1), (1, 2)):
        P = e[j] @ e[k] - e[j + 3] @ e[k + 3]
        Q = e[j] @ e[k] + e[j + 3] @ e[k + 3]
        ok &= eq(P + Q, 2 * e[j] @ e[k]) and eq(comm(P, N2), Z8)
        ok &= set(dN[Q != 0].tolist()) == {2, -2} and not eq(comm(e[j] @ e[k], N2), Z8)
    return ok


# ======================================================================================== 4. c on the spatial triple: polar form
Csym = sp.Matrix(3, 3, sp.symbols('c0:9'))
cc = sp.symbols('c', positive=True)
Bv = sp.Matrix(sp.symbols('B1:4', real=True))
Ev = sp.Matrix(sp.symbols('E1:4', real=True))


@claim("e6-4a", "for a generic invertible conversion C (nine symbols): G_C^2 = -1 and G_C^T N_C + N_C G_C = 0 with N_C = diag(C^T C, 1), so e^{theta G_C} preserves "
       "E_C = (1/2)(|CB|^2 + |E|^2), which depends on C only through C^T C; and G_C acts as i on z_C = CB + iE")
def _():
    GC = G_C(Csym)
    ok = sp.simplify(GC * GC + sp.eye(6)) == sp.zeros(6)
    NC = blk(Csym.T * Csym, Is3)
    ok &= sp.simplify(GC.T * NC + NC * GC) == sp.zeros(6)
    X = sp.Matrix.vstack(Bv, Ev)
    Y = GC * X
    zC = lambda Z: Csym * Z[:3, 0] + sp.I * Z[3:, 0]
    ok &= sp.simplify(zC(Y) - sp.I * zC(X)) == sp.zeros(3, 1)
    return ok


@claim("e6-4b", "G_C preserves E_c = (1/2)(c^2|B|^2 + |E|^2) iff C^T C = c^2: G_C^T N_c + N_c G_C = [[0, C^T - c^2 C^{-1}], [C - c^2 C^{-T}, 0]] (symbolic), so the conversions "
       "whose duality preserves the weighted energy are exactly C = cR, R in O(3); C = diag(1,2,3) fails, C = 2Q passes for Cayley rotations and for -Q")
def _():
    GC = G_C(Csym)
    Nc = blk(cc ** 2 * Is3, Is3)
    lhs = GC.T * Nc + Nc * GC
    rhs = sp.Matrix(sp.BlockMatrix([[Zs3, Csym.T - cc ** 2 * Csym.inv()], [Csym - cc ** 2 * Csym.inv().T, Zs3]]))
    ok = sp.simplify(lhs - rhs) == sp.zeros(6)
    test = lambda C, c: sp.simplify(G_C(C).T * blk(c ** 2 * Is3, Is3) + blk(c ** 2 * Is3, Is3) * G_C(C)) == sp.zeros(6)
    ok &= not test(sp.diag(1, 2, 3), 1) and not test(sp.diag(2, 2, 2), 1)
    for v in ([1, 2, 3], [sp.Rational(2, 3), -1, 0]):
        Q = cayley(v)
        ok &= test(2 * Q, 2) and test(-2 * Q, 2)
    return ok


@claim("e6-4c", "for C = cQ (c = 2, Q a Cayley rotation): the elements of so(3) + so(3) commuting with G_C are exactly the twisted diagonal rotations (A, QAQ^{-1}) (dim 3); "
       "G_{cQ} = Ad_{(Q^{-1},1)} G_c = Ad_{(1,Q)} G_c, and Ad_{(Q+,Q-)} G_c = G_{c Q- Q+^{-1}} for (Q+, Q-) in O(3) x O(3); the Lie algebra preserving eta and E_c is "
       "so(3) + so(3) (dim 6), and adding [Y, G_c] = 0 leaves the diagonal so(3) (dim 3): {C : C^T C = c^2} is one O(3) x O(3)-orbit of dimension 3")
def _():
    c = 2
    Q = cayley([1, 2, 3])
    GC = G_C(c * Q)
    xs = sp.symbols('x0:6')
    A, Ap = cross(xs[:3]), cross(xs[3:])
    eqs = list(GC * blk(A, Ap) - blk(A, Ap) * GC)
    M_, _ = sp.linear_eq_to_matrix(eqs, xs)
    ns = M_.nullspace()
    ok = len(ns) == 3
    for v in ns:
        Av, Apv = cross(v[:3]), cross(v[3:])
        ok &= sp.simplify(Apv - Q * Av * Q.inv()) == sp.zeros(3)
    Gc = G_C(c * Is3)
    ok &= sp.simplify(blk(Q.inv(), Is3) * Gc * blk(Q, Is3) - GC) == sp.zeros(6)
    ok &= sp.simplify(blk(Is3, Q) * Gc * blk(Is3, Q.inv()) - GC) == sp.zeros(6)
    Qp, Qm = cayley([0, 1, sp.Rational(1, 2)]), -cayley([2, 0, 1])
    g = blk(Qp, Qm)
    ok &= sp.simplify(g * Gc * g.inv() - G_C(c * Qm * Qp.inv())) == sp.zeros(6)
    # stabilisers (eta and E_c with c = 2)
    ys = sp.symbols('y0:36')
    Y = sp.Matrix(6, 6, ys)
    Nc = blk(c ** 2 * Is3, Is3)
    eqs = list(Y.T * ETAs + ETAs * Y) + list(Y.T * Nc + Nc * Y)
    M1, _ = sp.linear_eq_to_matrix(eqs, ys)
    ok &= 36 - M1.rank() == 6
    eqs2 = eqs + list(Y * Gc - Gc * Y)
    M2, _ = sp.linear_eq_to_matrix(eqs2, ys)
    ok &= 36 - M2.rank() == 3
    return ok


@claim("e6-4d", "a temporal rotation (Q+, 1) carries G_C to G_{C Q+^{-1}} and N_C to N_{C Q+^{-1}}: C^T C -> Q+ C^T C Q+^T (unchanged iff [Q+, C^T C] = 0, always at C = cR), "
       "C C^T unchanged; a spatial rotation (1, Q-) carries C to Q- C, with C^T C unchanged (exact for an integer C)")
def _():
    C = sp.Matrix([[2, 1, 0], [0, 1, 3], [1, 0, 1]])
    Qp, Qm = cayley([1, 0, 2]), cayley([sp.Rational(1, 3), 1, 0])
    g = blk(Qp, Is3)
    Cn = C * Qp.inv()
    ok = sp.simplify(g * G_C(C) * g.inv() - G_C(Cn)) == sp.zeros(6)
    NC = blk(C.T * C, Is3)
    ok &= sp.simplify(g.inv().T * NC * g.inv() - blk(Cn.T * Cn, Is3)) == sp.zeros(6)
    ok &= sp.simplify(Cn.T * Cn - Qp * C.T * C * Qp.T) == sp.zeros(3) and sp.simplify(Cn * Cn.T - C * C.T) == sp.zeros(3)
    ok &= sp.simplify(Cn.T * Cn - C.T * C) != sp.zeros(3)
    h = blk(Is3, Qm)
    Cm = Qm * C
    ok &= sp.simplify(h * G_C(C) * h.inv() - G_C(Cm)) == sp.zeros(6)
    ok &= sp.simplify(Cm.T * Cm - C.T * C) == sp.zeros(3)
    return ok


@claim("e6-4e", "for C = cQ: q_C = z_C.z_C has Re q_C = c^2|B|^2 - |E|^2 and E_c independent of Q, Im q_C = 2c (QB).E; at c = 1, B = e1, E = e2 the rotation Q(theta) about e3 "
       "gives q_C = 2i sin theta: the field is null for Q = 1 and at rest (E_0 = 1 = E_6) for theta = pi/2")
def _():
    Q = cayley([sp.Rational(1, 2), 1, 2])
    z = cc * Q * Bv + sp.I * Ev
    q = sp.expand((z.T * z)[0, 0])
    ok = sp.simplify(sp.re(q) - (cc ** 2 * (Bv.T * Bv)[0, 0] - (Ev.T * Ev)[0, 0])) == 0
    ok &= sp.simplify(sp.im(q) - 2 * cc * ((Q * Bv).T * Ev)[0, 0]) == 0
    th = sp.symbols('theta', real=True)
    Qt = sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])
    zt = Qt * sp.Matrix([1, 0, 0]) + sp.I * sp.Matrix([0, 1, 0])
    qt = sp.simplify((zt.T * zt)[0, 0])
    ok &= sp.simplify(qt - 2 * sp.I * sp.sin(th)) == 0
    ok &= sp.simplify((zt.H * zt)[0, 0] / 2) == 1
    return ok


# ======================================================================================== 5. six-momentum and the clock
n1, n2, n3, E1, E2, E3 = sp.symbols('n1 n2 n3 E1 E2 E3', real=True)
tt = sp.symbols('t', real=True)


def rodrigues(eh, x, t):
    eh, x = sp.Matrix(eh), sp.Matrix(x)
    return sp.cos(t) * x + sp.sin(t) * eh.cross(x) + (1 - sp.cos(t)) * (eh.dot(x)) * eh


@claim("e6-5a", "under a temporal rotation of rate Omega about a unit axis e^, E(t) = R(Omega t)E keeps |E| and mu; the energy read by a fixed clock n is "
       "E_n(t) = A + B cos(Omega t - phi) with A = (n.e^)(e^.E) and B = |n_perp||E_perp| (parts orthogonal to e^), so m_4^2 c^4 = mu^2 c^4 - |E|^2 + E_n(t)^2 "
       "(TR Prop. 2.13) oscillates, and is constant iff n || e^ or E || e^ (symbolic in n, E for three rational axes)")
def _():
    ok = True
    nv, Evv = sp.Matrix([n1, n2, n3]), sp.Matrix([E1, E2, E3])
    for eh in ([0, 0, 1], [sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)], [sp.Rational(2, 7), sp.Rational(3, 7), sp.Rational(6, 7)]):
        ehv = sp.Matrix(eh)
        Et = rodrigues(eh, Evv, tt)
        ok &= sp.simplify(Et.dot(Et) - Evv.dot(Evv)) == 0
        En = sp.expand(nv.dot(Et))
        A = nv.dot(ehv) * ehv.dot(Evv)
        cA = nv.dot(Evv) - A                    # coefficient of cos t
        sA = nv.dot(ehv.cross(Evv))             # coefficient of sin t
        ok &= sp.simplify(En - (A + cA * sp.cos(tt) + sA * sp.sin(tt))) == 0
        Bsq = (nv.dot(nv) - nv.dot(ehv) ** 2) * (Evv.dot(Evv) - ehv.dot(Evv) ** 2)
        ok &= sp.expand(cA ** 2 + sA ** 2 - Bsq) == 0
    return ok


@claim("e6-5b", "examples (c = 1, E = (2,3,6), p = (0,0,3), mu^2 = 40): rotation about e3 read by e1 gives E_n = 2cos t - 3sin t and m_4^2 in [-9, 4]; read by W it gives "
       "m_4^2 in [35/3 - 4 sqrt26, 35/3 + 4 sqrt26]; rotation about W read by W keeps m_4^2 = 94/3 while E_perp turns; rotation about the clock (spin lift prop. to J(n)) "
       "keeps E(n); the reduced shell E || n is preserved iff e^ || n")
def _():
    Evv = sp.Matrix([2, 3, 6])
    p2 = 9
    mu2 = Evv.dot(Evv) - p2
    ok = mu2 == 40
    Et = rodrigues([0, 0, 1], Evv, tt)
    En = sp.simplify(Et[0])
    ok &= sp.simplify(En - (2 * sp.cos(tt) - 3 * sp.sin(tt))) == 0
    m4 = sp.expand(mu2 - Evv.dot(Evv) + En ** 2)
    # range of m4^2: E_n^2 in [0, 13]
    ok &= sp.simplify(sp.expand(En ** 2 + (3 * sp.cos(tt) + 2 * sp.sin(tt)) ** 2) - 13) == 0
    ok &= (mu2 - 49 + 0, mu2 - 49 + 13) == (-9, 4)
    W = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    EW = sp.simplify(W.dot(Et))
    A = W.dot(sp.Matrix([0, 0, 1])) * 6
    B = sp.sqrt((1 - sp.Rational(1, 3)) * 13)
    ok &= sp.simplify(sp.expand((EW - A) ** 2 + (W.dot(sp.Matrix([0, 0, 1]).cross(Et))) ** 2) - B ** 2) == 0
    lo, hi = sp.simplify(-9 + (A - B) ** 2), sp.simplify(-9 + (A + B) ** 2)
    ok &= sp.simplify(lo - (sp.Rational(35, 3) - 4 * sp.sqrt(26))) == 0 and sp.simplify(hi - (sp.Rational(35, 3) + 4 * sp.sqrt(26))) == 0
    EtW = rodrigues(list(W), Evv, tt)
    ok &= sp.simplify(W.dot(EtW) - 11 / sp.sqrt(3)) == 0 and sp.simplify(40 - 49 + sp.Rational(121, 3) - sp.Rational(94, 3)) == 0
    Eperp0 = Evv - W.dot(Evv) * W
    ok &= sp.simplify((EtW - W.dot(EtW) * W) - Eperp0) != sp.zeros(3, 1)
    # rotation about the clock n keeps E(n) (n = (2,3,6)/7 and n = e1)
    for nn in (sp.Matrix([2, 3, 6]) / 7, sp.Matrix([1, 0, 0])):
        ok &= sp.simplify(nn.dot(rodrigues(list(nn), sp.Matrix([E1, E2, E3]), tt)) - nn.dot(sp.Matrix([E1, E2, E3]))) == 0
    # reduced shell: E = |E| n preserved iff axis || n (n = e1)
    ok &= sp.simplify(rodrigues([1, 0, 0], sp.Matrix([5, 0, 0]), tt) - sp.Matrix([5, 0, 0])) == sp.zeros(3, 1)
    ok &= sp.simplify(rodrigues([0, 0, 1], sp.Matrix([5, 0, 0]), tt)[1]) != 0
    return ok


# ======================================================================================== 6. between constituents (n = 2, dense 64 x 64)
I64 = np.eye(64, dtype=np.int64)
E2 = {}
for a in range(1, 7):
    E2[(1, a)] = np.kron(e[a], I8)
    E2[(2, a)] = np.kron(om, e[a])
U2 = {a: np.kron(I8, e[a]) for a in range(1, 7)}
N2_2 = sum((I64 - E2[(k, i)] @ E2[(k, i + 3)]) for k in (1, 2) for i in (1, 2, 3))     # 2N on S_2
dN2 = (np.diag(N2_2)[:, None] - np.diag(N2_2)[None, :]) // 2
Xp = {i: E2[(1, i)] @ E2[(2, i)] for i in (1, 2, 3)}         # C_+ = -(1/2) sum Xp
Xm = {a: E2[(1, a)] @ E2[(2, a)] for a in (4, 5, 6)}         # C_- = +(1/2) sum Xm
Cp2 = -sum(Xp.values())                                      # 2 C_+
Cm2 = sum(Xm.values())                                       # 2 C_-
Cb2 = Cp2 + Cm2


def rotC5(Xs, sgn):
    """125 * exp(theta * sgn * (1/2) sum X) and 125 * its inverse, with cos(theta/2) = 3/5, sin(theta/2) = 4/5:
    the terms commute and X^2 = -1, so each factor is cos(theta/2) + sgn sin(theta/2) X.  For C_+ = -(1/2) sum X take sgn = -1."""
    R = I64.copy()
    Ri = I64.copy()
    for X in Xs:
        R = R @ (3 * I64 + sgn * 4 * X)
        Ri = Ri @ (3 * I64 - sgn * 4 * X)
    return R, Ri


def vec_copy_rotation(D):
    """vector-level copy rotation on the directions D: e^(1)_a -> e^(2)_a"""
    M = sp.zeros(12)
    for a in D:
        M[6 + a - 1, a - 1] = 1
        M[a - 1, 6 + a - 1] = -1
    return M


G2s = blk(Gs, Gs)


@claim("e6-6a", "C_+ and C_- are antisymmetric and commute; each is a sum of three commuting terms of square -1/4, so exp(2pi C_+-) = -1 and exp(2pi C_b) = +1, with "
       "eigenvalues i{+-1/2, +-3/2} (restates RE d7:prop:isospin(i)); on vectors neither commutes with G_2, their sum does, and their difference C_x = C_+ - C_- "
       "is compact and anticommutes with G_2")
def _():
    ok = eq(Cp2.T, -Cp2) and eq(Cm2.T, -Cm2) and eq(comm(Cp2, Cm2), 0 * I64)
    ok &= all(eq(X @ X, -I64) for X in list(Xp.values()) + list(Xm.values()))
    ok &= all(eq(comm(X, Y), 0 * I64) for X in Xp.values() for Y in Xp.values())
    ok &= all(eq(comm(X, Y), 0 * I64) for X in Xm.values() for Y in Xm.values())
    # exp(2 pi C) = prod (cos pi + sin pi X) = (-1)^3 ; spectrum of 2C_+ : 2i*{+-1/2,+-3/2} -> char poly of 2C_+ over the joint signs
    # joint eigenvalues of the three commuting X (each +-i): 2C_+ = -(X1+X2+X3) has eigenvalues -i(s1+s2+s3)
    # 2C_+ = -(X1 + X2 + X3) with commuting X_i, X_i^2 = -1: minimal polynomial (l^2 + 1)(l^2 + 9), and
    # Tr (2C_+)^2 = -192 gives multiplicities 24 (eigenvalues +-i) and 8 (+-3i): C_+ has i{+-1/2 (24 each), +-3/2 (8 each)}
    for C2 in (Cp2, Cm2):
        A2 = C2 @ C2
        ok &= eq((A2 + I64) @ (A2 + 9 * I64), 0 * I64) and not eq(A2 + I64, 0 * I64) and not eq(A2 + 9 * I64, 0 * I64)
        ok &= int(np.trace(A2)) == -192
    Vp, Vm = vec_copy_rotation([1, 2, 3]), vec_copy_rotation([4, 5, 6])
    ok &= Vp * G2s != G2s * Vp and Vm * G2s != G2s * Vm and (Vp + Vm) * G2s == G2s * (Vp + Vm)
    ok &= (Vp - Vm) * G2s == -G2s * (Vp - Vm) and (Vp - Vm).T == -(Vp - Vm)
    return ok


@claim("e6-6b", "mass transfer: Ad(e^{theta C_+}) E_(1,1) = cos theta E_(1,1) + sin theta E_(2,1) and E_(2,1) -> cos theta E_(2,1) - sin theta E_(1,1) (exact at cos(theta/2) = 3/5), "
       "so H_g = m1 E_(1,1) + m2 E_(2,1) has its masses rotated at fixed M, from (M, 0) to (M cos theta, M sin theta), with spectrum +-M throughout; C_- commutes with H_g "
       "(compare emergence e1:prop:rest(ii))")
def _():
    R, Ri = rotC5(Xp.values(), -1)          # exp(theta C_+) = prod (cos(th/2) - sin(th/2) X), C_+ = -(1/2) sum X
    S = 125 ** 2
    ok = eq(R @ Ri, S * I64)
    cth, sth = -7, 24                          # 25 cos theta, 25 sin theta
    ok &= eq(25 * (R @ E2[(1, 1)] @ Ri), S * (cth * E2[(1, 1)] + sth * E2[(2, 1)]))
    ok &= eq(25 * (R @ E2[(2, 1)] @ Ri), S * (cth * E2[(2, 1)] - sth * E2[(1, 1)]))
    # H_g from (M,0) = E_(1,1): after the rotation, H^2 = M^2
    H = R @ E2[(1, 1)] @ Ri                    # = S * (cos E11 + sin E21)
    ok &= eq(H @ H, S * S * I64)
    ok &= all(eq(comm(X, E2[(k, 1)]), 0 * I64) for X in Xm.values() for k in (1, 2))
    return ok


@claim("e6-6c", "C_+ does not conserve N: C_+ = (1/2)C_b + (1/2)C_x, where C_b conserves N (it hops a quantum) and C_x changes N by +-2 only (one quantum in each constituent); "
       "from the Fock vacuum <N>(theta) = 3(1 - cos theta), exact at cos(theta/2) = 3/5 (<N> = 96/25), and theta = pi fills all six modes")
def _():
    Cx2 = Cp2 - Cm2
    ok = eq(Cp2, (Cb2 + Cx2) // 2) and eq(2 * Cp2, Cb2 + Cx2)
    ok &= eq(comm(Cb2, N2_2), 0 * I64) and not eq(comm(Cp2, N2_2), 0 * I64)
    ok &= set(dN2[Cx2 != 0].tolist()) == {2, -2} and set(dN2[Cb2 != 0].tolist()) == {0}
    vac = np.zeros(64, dtype=np.int64)
    vac[int(np.argmin(np.diag(N2_2)))] = 1
    R, _ = rotC5(Xp.values(), -1)
    v = R @ vac                                # 125 * state
    expN = Fraction(int(v @ (N2_2 @ v)), 2 * 125 ** 2)
    ok &= expN == Fraction(96, 25) and expN == 3 * (1 - Fraction(-7, 25))
    # theta = pi: cos(pi/2) = 0, sin = 1: prod(-X) applied to vac lands in N = 6
    w = vac.copy()
    for X in Xp.values():
        w = -X @ w
    ok &= int(np.diag(N2_2)[np.nonzero(w)[0][0]]) // 2 == 6 and np.count_nonzero(w) == 1
    return ok


@claim("e6-6d", "ungraded copies: spec(m1 e1(x)1 + m2 1(x)e1) = {+-m1 +- m2}, each value 16 times, and this multiset determines {|m1|, |m2|}; Ad(e^{theta C_+}) maps e1(x)1 to "
       "cos theta e1(x)1 + sin theta omega(x)e1, outside Cl(3,3)(x)1 + 1(x)Cl(3,3) unless sin theta = 0; rotating (m1, m2) inside the ungraded family from (M, 0) changes "
       "the spectrum to {+-M(cos theta + sin theta), +-M(cos theta - sin theta)} (exact: 25 x spectrum = {+-17, +-31} at cos theta = -7/25)")
def _():
    u1, u2 = np.kron(e[1], I8), np.kron(I8, e[1])
    ok = eq(comm(u1, u2), 0 * I64) and np.trace(u1) == 0 and np.trace(u2) == 0 and np.trace(u1 @ u2) == 0
    # joint sign multiplicities 16 each: projectors (1 +- u1)(1 +- u2)/4 have trace 16
    ok &= all(np.trace((I64 + s1 * u1) @ (I64 + s2 * u2)) // 4 == 16 for s1 in (1, -1) for s2 in (1, -1))
    # multiset determines {|m1|,|m2|}: largest |value| = |m1|+|m2|, the other = ||m1|-|m2||
    for (m1, m2) in ((3, 1), (1, 3), (2, 0), (5, 4)):
        vals = sorted({abs(s1 * m1 + s2 * m2) for s1 in (1, -1) for s2 in (1, -1)})
        big, small = max(vals), min(vals)
        ok &= sorted(((big + small) // 2, (big - small) // 2)) == sorted((abs(m1), abs(m2)))
    R, Ri = rotC5(Xp.values(), -1)
    S = 125 ** 2
    ok &= eq(25 * (R @ u1 @ Ri), S * (-7 * u1 + 24 * np.kron(om, e[1])))
    # omega (x) e1 is a single blade with both factors nontrivial: orthogonal (trace form) to all A(x)1 and 1(x)B blades
    W = np.kron(om, e[1])
    ok &= all(np.trace(W.T @ np.kron(A, I8)) == 0 and np.trace(W.T @ np.kron(I8, A)) == 0 for A in [I8, om, e[1], T, OS, J])
    # ungraded rotation of the masses: 25 H = 25M(cos u1 + sin u2) = -7 u1 + 24 u2
    H25 = -7 * u1 + 24 * u2
    vals = sorted({s1 * (-7) + s2 * 24 for s1 in (1, -1) for s2 in (1, -1)})
    ok &= vals == [-31, -17, 17, 31]
    ok &= all(eq((H25 - (s1 * -7 + s2 * 24) * I64) @ (I64 + s1 * u1) @ (I64 + s2 * u2), 0 * I64) for s1 in (1, -1) for s2 in (1, -1))
    return ok


@claim("e6-6e", "the temporal copy rotation as a constant inter-manifold field: S = kappa G_2 + beta V_+ has det(lambda - S) = (lambda^4 + (2kappa^2 + beta^2)lambda^2 + kappa^4)^3, "
       "lines s +- beta/2 (s = (kappa^2 + beta^2/4)^{1/2}), rates unchanged; at (kappa, beta) = (2, 3) M has Jordan blocks at the lines 1 and 4 (it is outside "
       "the binding-compatible class of RE d7:thm:schr)")
def _():
    Vp = vec_copy_rotation([1, 2, 3])
    S = kap * G2s + w_ * Vp
    p = sp.expand((lam * sp.eye(12) - S).det())
    ok = sp.expand(p - (lam ** 4 + (2 * kap ** 2 + w_ ** 2) * lam ** 2 + kap ** 4) ** 3) == 0
    S23 = 2 * G2s + 3 * Vp
    prof = jordan_profile(S23)
    ok &= set(abs(sp.im(ev)) for ev in prof) == {1, 4}
    ok &= all(geo < alg for (geo, alg) in prof.values())
    return ok


# ======================================================================================== 7. up the ladder
def sp_of(M):
    """signed-permutation form of an integer matrix: M e_j = s_j e_{pi(j)}"""
    pi = np.argmax(np.abs(M), axis=0)
    s = M[pi, np.arange(M.shape[0])]
    assert np.all(np.abs(s) == 1) and np.count_nonzero(M) == M.shape[0]
    return pi.astype(np.int64), s.astype(np.int64)


SINGLE = {a: sp_of(e[a]) for a in range(1, 7)}
OMSP = sp_of(om)
IDSP = (np.arange(8, dtype=np.int64), np.ones(8, dtype=np.int64))


def tower_sp(n):
    gens = {}
    for k in range(1, n + 1):
        for a in range(1, 7):
            factors = [OMSP] * (k - 1) + [SINGLE[a]] + [IDSP] * (n - k)
            pi = np.zeros(1, dtype=np.int64)
            s = np.ones(1, dtype=np.int64)
            for (p, ss) in factors:
                pi = (pi[:, None] * 8 + p[None, :]).reshape(-1)
                s = (s[:, None] * ss[None, :]).reshape(-1)
            gens[(k, a)] = (pi, s)
    return gens


def compose(A, B):
    """signed permutation of A @ B"""
    pa, sa = A
    pb, sb = B
    return pa[pb], sb * sa[pb]


def sp_dense(P):
    pi, s = P
    d = len(pi)
    M = np.zeros((d, d), dtype=np.int64)
    M[pi, np.arange(d)] = s
    return M


def sp_transpose(P):
    pi, s = P
    inv = np.empty_like(pi)
    inv[pi] = np.arange(len(pi))
    return inv, s[inv]


def sp_eq(A, B, sign=1):
    return np.array_equal(A[0], B[0]) and np.array_equal(A[1], sign * B[1])


def sp_prod(gens, keys):
    d = len(gens[keys[0]][0])
    P = (np.arange(d, dtype=np.int64), np.ones(d, dtype=np.int64))
    for k in keys:
        P = compose(P, gens[k])
    return P


def inv_forms(gen_list, d):
    """dimension of {M : M_{pi(i)pi(j)} = tau s_i s_j M_ij for all (pi, s, tau)}, and its split into symmetric and antisymmetric M
    (exact: signed orbits of the entries, by connected components of the doubled graph)"""
    Nn = d * d
    idx = np.arange(Nn, dtype=np.int64)
    I, Jx = idx // d, idx % d
    rows, cols = [], []
    for (pi, s, tau) in gen_list:
        tgt = pi[I] * d + pi[Jx]
        sg = tau * s[I] * s[Jx]
        rows += [idx, idx + Nn]
        cols += [np.where(sg == 1, tgt, tgt + Nn), np.where(sg == 1, tgt + Nn, tgt)]
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(2 * Nn, 2 * Nn))
    _, lab = connected_components(g, directed=False)
    lp, lm = lab[:Nn], lab[Nn:]
    good = lp != lm
    nodes = idx[good]
    orb = np.minimum(lp, lm)[good]
    sign = np.where(lp[good] == orb, 1, -1)
    uo, inv = np.unique(orb, return_inverse=True)
    total = len(uo)
    pos = -np.ones(Nn, dtype=np.int64)
    pos[nodes] = np.arange(len(nodes))
    tp = pos[Jx[good] * d + I[good]]
    assert np.all(tp >= 0)
    torb, tsign = inv[tp], sign[tp]
    first = np.zeros(total, dtype=np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    sym = asym = 0
    for o in range(total):
        r_ = first[o]
        o2 = torb[r_]
        if o2 == o:
            if sign[r_] * tsign[r_] == 1:
                sym += 1
            else:
                asym += 1
        elif o2 > o:
            sym += 1
            asym += 1
    return total, sym, asym


def eta_of(a):
    return 1 if a <= 3 else -1


def biv_cond(gens, A, B):
    """(pi, s, tau) for the invariance condition Y^T M + M Y = 0 of the bivector blade Y = E_A E_B: Y M Y^{-1} = tau M"""
    pi, s = compose(gens[A], gens[B])
    epsT = -eta_of(A[1]) * eta_of(B[1])          # Y^T = epsT Y
    return pi, s, -epsT


VOL_TYPES = {}


@claim("e6-7a", "at every rung n = 1..4 the temporal volume T_n (product of the 3n temporal E_(k,i)) and the spatial volume S_n satisfy M E_A M^{-1} = sigma eta_AA E_A^T-type "
       "relations that make Y^T M + M Y = 0 for every bivector (exact, signed permutations); T_n^T = (-1)^{3n(3n-1)/2} T_n and S_n^T = (-1)^{3n+3n(3n-1)/2} S_n: "
       "(T_n, S_n) are (anti, sym), (anti, anti), (sym, anti), (sym, sym) for n = 1, 2, 3, 4; at n = 2, T_2 = -omega_S(x)T and S_2 = -T(x)omega_S; T_n S_n = +-Gamma_n")
def _():
    ok = True
    expected = {1: (-1, 1), 2: (-1, -1), 3: (1, -1), 4: (1, 1)}
    for n in (1, 2, 3, 4):
        gens = tower_sp(n)
        tkeys = [(k, a) for k in range(1, n + 1) for a in (1, 2, 3)]
        skeys = [(k, a) for k in range(1, n + 1) for a in (4, 5, 6)]
        allkeys = [(k, a) for k in range(1, n + 1) for a in range(1, 7)]
        Tn, Sn, Gn = sp_prod(gens, tkeys), sp_prod(gens, skeys), sp_prod(gens, allkeys)
        for M in (Tn, Sn):
            Mt = sp_transpose(M)                 # M^{-1} = +-M^T for a signed permutation: M^{-1} = M^T exactly
            # condition for each bivector Y = E_A E_B: Y^T M + M Y = 0  <=>  M Y M^{-1} = -Y^T
            for A, B in itertools.combinations(allkeys, 2):
                Y = compose(gens[A], gens[B])
                lhs = compose(compose(M, Y), Mt)
                YT = sp_transpose(Y)
                ok &= sp_eq(lhs, YT, -1)
        tT = 1 if sp_eq(sp_transpose(Tn), Tn) else (-1 if sp_eq(sp_transpose(Tn), Tn, -1) else 0)
        tS = 1 if sp_eq(sp_transpose(Sn), Sn) else (-1 if sp_eq(sp_transpose(Sn), Sn, -1) else 0)
        m = 3 * n
        ok &= (tT, tS) == expected[n] == ((-1) ** (m * (m - 1) // 2), (-1) ** (m + m * (m - 1) // 2))
        TS = compose(Tn, Sn)
        ok &= sp_eq(TS, Gn) or sp_eq(TS, Gn, -1)
        VOL_TYPES[n] = (tT, tS)
        if n == 2:
            ok &= eq(sp_dense(Tn), -np.kron(OS, T)) and eq(sp_dense(Sn), -np.kron(T, OS))
    return ok


def masks_commuting_with_all_bivectors(nbits):
    """blades (bitmasks) commuting with every adjacent bivector e_A e_{A+1}; blade rule: commute iff |I cap {A, A+1}| even"""
    m = np.arange(1 << nbits, dtype=np.uint32)
    good = np.ones(len(m), dtype=bool)
    for A in range(nbits - 1):
        good &= (((m >> np.uint32(A)) ^ (m >> np.uint32(A + 1))) & np.uint32(1)) == 0
    return [int(x) for x in m[good]]


@claim("e6-7b", "uniqueness up the ladder: for an invariant M, T_n^{-1} M commutes with every bivector, and the blades commuting with all bivectors are 1 and Gamma_n only "
       "(blade rule, n = 1..4); so the forms invariant under so(3n,3n) are span{T_n, S_n} at every rung; the exact orbit count gives (total, sym, antisym) = "
       "(2,1,1), (2,0,2), (2,1,1) at n = 1, 2, 3: the Grassmann (antisymmetric) mass kernel is T at n = 1, two-dimensional at n = 2, S_3 at n = 3 and absent at n = 4")
def _():
    ok = True
    for n in (1, 2, 3, 4):
        ok &= masks_commuting_with_all_bivectors(6 * n) == [0, (1 << (6 * n)) - 1]
    expect = {1: (2, 1, 1), 2: (2, 0, 2), 3: (2, 1, 1)}
    for n in (1, 2, 3):
        gens = tower_sp(n)
        order = [(k, a) for k in range(1, n + 1) for a in range(1, 7)]
        conds = [biv_cond(gens, order[i], order[i + 1]) for i in range(len(order) - 1)]
        ok &= inv_forms(conds, 8 ** n) == expect[n]
    # n = 4: no antisymmetric invariant kernel, since both volumes are symmetric (e6-7a)
    ok &= VOL_TYPES.get(4) == (1, 1)
    return ok


@claim("e6-7c", "temporal rotations: the forms invariant under so(3n)_+ (the bivectors of the 3n temporal generators) are its commutant, of dimension 16, 128, 1024 at "
       "n = 1, 2, 3, with (sym, antisym) = (6, 10), (64, 64), (528, 496) (exact orbit count); they contain T_n and S_n, so so(3n)_+ alone selects no mass form")
def _():
    expect = {1: (16, 6, 10), 2: (128, 64, 64), 3: (1024, 528, 496)}
    ok = True
    for n in (1, 2, 3):
        gens = tower_sp(n)
        temporal = [(k, a) for k in range(1, n + 1) for a in (1, 2, 3)]
        conds = [biv_cond(gens, temporal[i], temporal[i + 1]) for i in range(len(temporal) - 1)]
        ok &= all(c[2] == 1 for c in conds)       # compact: invariance = commutation
        ok &= inv_forms(conds, 8 ** n) == expect[n]
    return ok


def weyl_D3():
    els = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            if np.prod(signs) != 1:
                continue
            M = np.zeros((3, 3), dtype=np.int64)
            for i in range(3):
                M[perm[i], i] = signs[i]
            els.append(M)
    return els


WEYL = weyl_D3()
RHO2 = np.array([4, 2, 0], dtype=np.int64)       # 2 rho of D3 in doubled weights (spinor weights +-1)


def mult_trivial(counter):
    tot = 0
    for wm in WEYL:
        mu = tuple((RHO2 - wm @ RHO2).tolist())
        tot += int(round(np.linalg.det(wm))) * counter.get(mu, 0)
    return tot


def diag_weights(n):
    gens = tower_sp(n)
    d = 8 ** n
    wts = np.zeros((d, 3), dtype=np.int64)
    for i in range(3):
        for k in range(1, n + 1):
            pi, s = compose(gens[(k, i + 1)], gens[(k, i + 4)])
            assert np.array_equal(pi, np.arange(d))
            wts[:, i] += s
    return wts


@claim("e6-7d", "diagonal so(3,3) (the same bivector on every constituent): the invariant forms on S_n have dimension 2, 14, 168 at n = 1, 2, 3 with (sym, antisym) = "
       "(1, 1), (10, 4), (84, 84) (Weyl multiplicity formula on the exact weights of the split Cartan e_ie_{i+3}); at n = 2 an exact direct solution gives 14 = 10 + 4")
def _():
    ok = mult_trivial(Counter([tuple(x) for x in (RHO2 - RHO2)[None, :].tolist()])) == 1      # trivial rep sanity
    expect = {1: (2, 1, 1), 2: (14, 10, 4), 3: (168, 84, 84)}
    for n in (1, 2, 3):
        wts = diag_weights(n)
        d = len(wts)
        Bw = wts[:, None, :] + wts[None, :, :]
        tot = mult_trivial(Counter(map(tuple, Bw.reshape(-1, 3).tolist())))
        iu = np.triu_indices(d, 1)
        il = np.triu_indices(d, 0)
        alt = mult_trivial(Counter(map(tuple, Bw[iu].tolist())))
        sym = mult_trivial(Counter(map(tuple, Bw[il].tolist())))
        ok &= (tot, sym, alt) == expect[n]
    # direct exact solution at n = 2: M supported on pairs of opposite weight, then all 15 diagonal bivectors
    wts = diag_weights(2)
    d = 64
    supp = [(i, j) for i in range(d) for j in range(d) if np.all(wts[i] + wts[j] == 0)]
    col = {p: c for c, p in enumerate(supp)}
    diagY = []
    for a, b in itertools.combinations(range(1, 7), 2):
        Y = E2[(1, a)] @ E2[(1, b)] + E2[(2, a)] @ E2[(2, b)]
        diagY.append(Y)

    def solve(extra_sym):
        rows = []
        for Y in diagY:
            # (Y^T M + M Y)_{kl} = sum_i Y_ik M_il + sum_j M_kj Y_jl
            for k in range(d):
                for l in range(d):
                    row = {}
                    for i in np.nonzero(Y[:, k])[0]:
                        if (i, l) in col:
                            row[col[(i, l)]] = row.get(col[(i, l)], 0) + int(Y[i, k])
                    for j in np.nonzero(Y[:, l])[0]:
                        if (k, j) in col:
                            row[col[(k, j)]] = row.get(col[(k, j)], 0) + int(Y[j, l])
                    row = {c: v for c, v in row.items() if v != 0}
                    if row:
                        rows.append(row)
        if extra_sym is not None:
            for (i, j) in supp:
                if i < j:
                    rows.append({col[(i, j)]: 1, col[(j, i)]: -extra_sym})
                elif i == j and extra_sym == -1:
                    rows.append({col[(i, j)]: 1})
        uniq = {tuple(sorted(r.items())) for r in rows}
        Mx = np.zeros((len(uniq), len(supp)), dtype=np.int64)
        for ri, r in enumerate(uniq):
            for c, v in r:
                Mx[ri, c] = v
        return len(supp) - rank_exact(Mx)
    ok &= len(supp) == 216
    ok &= (solve(None), solve(1), solve(-1)) == (14, 10, 4)
    return ok


@claim("e6-7e", "the ungraded products M1(x)M2, M_k in {T, omega_S}, span the forms invariant under so(3,3)^(1) + so(3,3)^(2) (exact orbit count: 4, with T(x)T and "
       "omega_S(x)omega_S symmetric, T(x)omega_S and omega_S(x)T antisymmetric); so(6,6) keeps exactly span{omega_S(x)T, T(x)omega_S}; the per-copy T(x)1 is not "
       "invariant under the boosts of copy 2")
def _():
    gens = tower_sp(2)
    conds = []
    for k in (1, 2):
        keys = [(k, a) for a in range(1, 7)]
        conds += [biv_cond(gens, keys[i], keys[i + 1]) for i in range(5)]
    ok = inv_forms(conds, 64) == (4, 2, 2)
    TT, SS, TSx, STx = np.kron(T, T), np.kron(OS, OS), np.kron(T, OS), np.kron(OS, T)
    ok &= eq(TT.T, TT) and eq(SS.T, SS) and eq(TSx.T, -TSx) and eq(STx.T, -STx)
    allkeys = [(k, a) for k in (1, 2) for a in range(1, 7)]
    bivs = [sp_dense(compose(gens[A], gens[B])) for A, B in itertools.combinations(allkeys, 2)]
    inv_all = lambda M: all(eq(Y.T @ M + M @ Y, 0 * I64) for Y in bivs)
    ok &= inv_all(TSx) and inv_all(STx) and not inv_all(TT) and not inv_all(SS)
    T1 = np.kron(T, I8)
    boost2 = np.kron(I8, e[1] @ e[4])
    ok &= not eq(boost2.T @ T1 + T1 @ boost2, 0 * I64)
    return ok


# ------------------------------------------------------------------------------------------------- summary
print(f"# runtime {time.time() - T0:.1f}s")
if FAILED:
    print("FAILED:", ", ".join(FAILED))
    sys.exit(1)
sys.exit(0)
