#!/usr/bin/env python3
"""Track e1 -- One particle or two: the second rung Cl(6,6).

    python3 investigations/emergence/verify/e1_two_from_one.py

Prints one 'PASS e1-<id>: ...' line per verified claim of sections/e1_two_from_one.tex and exits 0 only if every check
passes.  Runnable from any directory; numpy, sympy and scipy only.

Conventions (tex/s02_arena.tex eq. (1)-(2), tex/sA_realisation.tex):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1); on R^8: e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1,
  e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep; omega = e1...e6 (diagonal, = Gamma = J Omega).
  Tower: E_(k,a) = omega^{(x)(k-1)} (x) e_a (x) 1^{(x)(n-k)} on R^(8^n).  Ungraded copy k: 1^{(x)(k-1)} (x) e_a (x) 1....
  J = e2e3, T = e1e2e3, Omega = e1e4e5e6 (interior volume), gamma^mu = (e1, e4, e5, e6).
  G e_i = e_{i+3}, G e_{i+3} = -e_i; K e_i = e_{i+3}, K e_{i+3} = e_i (K = -calK of Appendix A of ES17).
  Fock modes a+-_{k,i} = (E_(k,i) +- E_(k,i+3))/2, N = sum a+ a-.

Exactness: every algebraic statement is checked in integer (or rational) arithmetic: the generators and their blades
are signed permutation matrices; at n = 3, 4 they are stored as (permutation, sign) arrays; vector-level Lie algebra
statements use exact rational null spaces (sympy).  Floating point is used only for the rest-flow exponentials
(tolerance 1e-9) and for one eigenvalue listing at n = 3 (tolerance 1e-9), both stated where made.
Runtime: well under a minute.
"""
import itertools
import sys
import time

import numpy as np
import scipy.linalg as sla
import sympy as sp

T0 = time.time()
FAILS = []


def report(cid, ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {cid}: {text}")
    if not ok:
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
I64 = np.eye(64, dtype=np.int64)
om = e[1] @ e[2] @ e[3] @ e[4] @ e[5] @ e[6]
J1c = e[2] @ e[3]                       # J on one constituent
T1c = e[1] @ e[2] @ e[3]                # T = gamma^0 J
Om1c = e[1] @ e[4] @ e[5] @ e[6]        # interior volume Omega
INT = [1, 4, 5, 6]                      # interior (gamma) indices
SUR = [2, 3]                            # surplus indices


def rank_exact(A):
    """exact rank of an integer matrix (fraction-free elimination over ZZ)"""
    from sympy.polys.matrices import DomainMatrix
    from sympy import ZZ
    A = np.asarray(A)
    if A.dtype.kind == 'f':
        assert np.all(A == np.round(A))
    A = A.astype(np.int64)
    if A.shape[0] > A.shape[1]:
        A = A.T
    keep = np.any(A != 0, axis=0)
    A = A[:, keep]
    rows = [[ZZ(int(x)) for x in r] for r in A.tolist()]
    if not rows or not rows[0]:
        return 0
    return DomainMatrix(rows, A.shape, ZZ).rank()


def comm(A, B):
    return A @ B - B @ A


def acomm(A, B):
    return A @ B + B @ A


def eq(A, B):
    return np.array_equal(A, B)


# ============================================================================== second rung, dense 64 x 64
E = {}
for a in range(1, 7):
    E[(1, a)] = np.kron(e[a], I8)
    E[(2, a)] = np.kron(om, e[a])
U2 = {a: np.kron(I8, e[a]) for a in range(1, 7)}      # ungraded copy-2 generators 1 (x) e_a
gens12 = [E[(1, a)] for a in range(1, 7)] + [E[(2, a)] for a in range(1, 7)]
ETA12 = ETA[1:] + ETA[1:]
OM1 = np.kron(om, I8)     # omega^(1)
OM2u = np.kron(I8, om)    # ungraded chirality of copy 2, 1 (x) omega
J1 = np.kron(J1c, I8)
J2 = np.kron(I8, J1c)
T1 = np.kron(T1c, I8)
T2 = np.kron(I8, T1c)
P12 = J1 @ J2
Psw = np.zeros((64, 64), dtype=np.int64)            # ungraded swap x (x) y -> y (x) x
for i in range(8):
    for j in range(8):
        Psw[j * 8 + i, i * 8 + j] = 1
pbit = (1 - np.diag(om)) // 2                       # omega = (-1)^p on the basis
Dk = np.diag([(-1) ** (int(pbit[i]) * int(pbit[j])) for i in range(8) for j in range(8)]).astype(np.int64)
tauK = Dk @ Psw                                     # Koszul swap of RE d5:prop:cmrel(iii)


def blade_mat(mask, gens):
    """product of the generators in the bitmask, in increasing order"""
    M = np.eye(gens[0].shape[0], dtype=np.int64)
    for i in range(len(gens)):
        if mask >> i & 1:
            M = M @ gens[i]
    return M


def bcomm_sign(A, B):
    """blade rule (3) of ES17: e_A e_B = s e_B e_A with s = (-1)^{|A||B| - |A cap B|}"""
    pa, pb, pab = bin(A).count('1'), bin(B).count('1'), bin(A & B).count('1')
    return (-1) ** ((pa * pb - pab) % 2)


def blade_commutant(gen_masks, nbits=12):
    """masks of the blades commuting with every blade in gen_masks.  Conjugation by an invertible blade is diagonal
    (entries +-1) in the blade basis, so the commutant of a set of blades is spanned by the blades it contains."""
    return [m for m in range(1 << nbits) if all(bcomm_sign(m, g) == 1 for g in gen_masks)]


def bit(k, a):        # index of E_(k,a) in the 12-bit mask
    return 1 << (6 * (k - 1) + a - 1)


OMEGA1_MASK = sum(bit(1, a) for a in range(1, 7))

# ------------------------------------------------------------------------------------------------- section 1
@claim("e1-1a", "the twelve graded E_(k,a) satisfy the Clifford relations of R^{6,6}; the ungraded 1(x)e_a satisfy those of "
       "R^{3,3}, commute with every e_a(x)1, and 1(x)e_a = omega^(1) E_(2,a) = E_1...E_6 E_{6+a}, a grade-7 blade")
def _():
    ok = all(eq(acomm(gens12[i], gens12[j]), 2 * ETA12[i] * (i == j) * I64) for i in range(12) for j in range(12))
    ok &= all(eq(acomm(U2[a], U2[b]), 2 * ETA[a] * (a == b) * I64) for a in range(1, 7) for b in range(1, 7))
    ok &= all(eq(comm(U2[a], E[(1, b)]), 0 * I64) for a in range(1, 7) for b in range(1, 7))
    ok &= all(eq(U2[a], OM1 @ E[(2, a)]) for a in range(1, 7))
    ok &= all(eq(U2[a], blade_mat(OMEGA1_MASK | bit(2, a), gens12)) for a in range(1, 7))
    ok &= eq(OM1, blade_mat(OMEGA1_MASK, gens12))
    return ok


@claim("e1-1b", "the commutant of {e_a(x)1} in M_64(R) is spanned by 64 blades of Cl(6,6), each equal to +-1(x)e_J: it is "
       "exactly 1(x)M_8(R) (restates RE d5:prop:graded(iii)); its generators 1(x)e_a are odd elements of grade 7")
def _():
    cm = blade_commutant([bit(1, a) for a in range(1, 7)])
    if len(cm) != 64:
        return False
    targets = {}
    for J in range(64):
        M = I8.copy()
        for a in range(6):
            if J >> a & 1:
                M = M @ e[a + 1]
        targets[J] = np.kron(I8, M)
    found = set()
    for m in cm:
        B = blade_mat(m, gens12)
        hit = [J for J in range(64) if eq(B, targets[J]) or eq(B, -targets[J])]
        if len(hit) != 1:
            return False
        found.add(hit[0])
    return len(found) == 64


@claim("e1-1c", "the Klein factor is the chirality and the Fock parity of copy 1: omega = J Omega = Gamma, omega = -(-1)^(N1+N2+N3); "
       "Cl(3,3)(x)Cl(3,3) and Cl(3,3)^Cl(3,3) are both M_64(R) (4096 independent blades), but (e1(x)1 + 1(x)e1)^2 = 2 + 2 e1(x)e1 "
       "is not a scalar, so the ungraded twelve span no quadratic space")
def _():
    ok = eq(om, J1c @ Om1c)
    Nn = sum(((e[i] + e[i + 3]) @ (e[i] - e[i + 3])) for i in (1, 2, 3)) // 4
    ok &= eq(om, -np.diag([(-1) ** int(x) for x in np.diag(Nn)]).astype(np.int64))
    # 4096 blades are pairwise trace-orthogonal signed permutations (Tr B^T B' = 0 for B != B')
    tr0 = all(np.trace(blade_mat(m, gens12)) == 0 for m in range(1, 4096))
    ok &= tr0
    x = E[(1, 1)] + U2[1]
    ok &= eq(x @ x, 2 * I64 + 2 * np.kron(e[1], e[1]))
    return ok


# ---------------------------------------------------------------- signed permutations for rungs n = 3, 4
class SP:
    """signed permutation matrix: M e_j = s[j] e_{p[j]}"""

    def __init__(self, p, s):
        self.p, self.s = np.asarray(p, dtype=np.int64), np.asarray(s, dtype=np.int64)

    @staticmethod
    def from_dense(M):
        n = M.shape[0]
        p = np.argmax(np.abs(M), axis=0)
        s = M[p, np.arange(n)]
        assert np.all(np.abs(s) == 1) and np.count_nonzero(M) == n
        return SP(p, s)

    def __matmul__(self, o):
        return SP(self.p[o.p], o.s * self.s[o.p])

    def kron(self, o):
        nb = len(o.p)
        p = (self.p[:, None] * nb + o.p[None, :]).ravel()
        s = (self.s[:, None] * o.s[None, :]).ravel()
        return SP(p, s)

    def __eq__(self, o):
        return np.array_equal(self.p, o.p) and np.array_equal(self.s, o.s)

    def neg(self):
        return SP(self.p, -self.s)

    def trace(self):
        fix = self.p == np.arange(len(self.p))
        return int(self.s[fix].sum())

    def ident(n):
        return SP(np.arange(n), np.ones(n, dtype=np.int64))


sp_e = [None] + [SP.from_dense(e[a]) for a in range(1, 7)]
sp_om = SP.from_dense(om)
sp_I8 = SP.ident(8)


def tower_sp(n):
    """graded generators E_(k,a) and ungraded u_(k,a) at rung n as signed permutations"""
    Eg, Eu = {}, {}
    for k in range(1, n + 1):
        for a in range(1, 7):
            fg = [sp_om] * (k - 1) + [sp_e[a]] + [sp_I8] * (n - k)
            fu = [sp_I8] * (k - 1) + [sp_e[a]] + [sp_I8] * (n - k)
            g, u = fg[0], fu[0]
            for f in fg[1:]:
                g = g.kron(f)
            for f in fu[1:]:
                u = u.kron(f)
            Eg[(k, a)], Eu[(k, a)] = g, u
    return Eg, Eu


def anticommute_sp(A, B):
    return (A @ B) == (B @ A).neg()


def commute_sp(A, B):
    return (A @ B) == (B @ A)


@claim("e1-1d", "rungs n=3,4: the ungraded generator of copy k equals Omega^(1)...Omega^(k-1) E_(k,a) (Omega^(j) the volume "
       "element of constituent j, a blade of grade 6(k-1)+1); ungraded copies commute with each other, graded ones anticommute")
def _():
    ok = True
    for n in (3, 4):
        Eg, Eu = tower_sp(n)
        Omj = {}
        for j in range(1, n + 1):
            M = SP.ident(8 ** n)
            for a in range(1, 7):
                M = M @ Eg[(j, a)]
            Omj[j] = M
        for k in range(1, n + 1):
            for a in range(1, 7):
                M = SP.ident(8 ** n)
                for j in range(1, k):
                    M = M @ Omj[j]
                ok &= (M @ Eg[(k, a)]) == Eu[(k, a)]
        for k in range(1, n + 1):
            for l in range(k + 1, n + 1):
                for a, b in [(1, 1), (2, 5), (6, 3)]:
                    ok &= commute_sp(Eu[(k, a)], Eu[(l, b)]) and anticommute_sp(Eg[(k, a)], Eg[(l, b)])
    return ok


# ------------------------------------------------------------------------------------------------- section 2
m1s, m2s = sp.symbols('m1 m2', positive=True)


@claim("e1-2a", "graded rest Hamiltonian: E_(1,1), E_(2,1) are anticommuting traceless involutions, so H_g = m1 E_(1,1) + m2 E_(2,1) "
       "has H_g^2 = (m1^2+m2^2) 1 and spectrum +-sqrt(m1^2+m2^2), multiplicity 32 each (at (m1,m2) = (3,4): +-5)")
def _():
    A, B = E[(1, 1)], E[(2, 1)]
    ok = eq(A @ A, I64) and eq(B @ B, I64) and eq(acomm(A, B), 0 * I64) and np.trace(A) == 0 and np.trace(B) == 0
    ok &= np.trace(A @ B) == 0
    H = 3 * A + 4 * B
    ok &= eq(H @ H, 25 * I64) and np.trace(H) == 0
    ok &= rank_exact(H - 5 * I64) == 32 and rank_exact(H + 5 * I64) == 32
    return ok


@claim("e1-2b", "ungraded rest Hamiltonian: e1(x)1 and 1(x)e1 are commuting involutions with traceless products, so "
       "H_u = m1 e1(x)1 + m2 1(x)e1 has spectrum {+-m1 +- m2}, multiplicity 16 each (exact at (m1,m2) = (1,2): -3,-1,1,3)")
def _():
    A, B = E[(1, 1)], U2[1]
    ok = eq(comm(A, B), 0 * I64) and eq(A @ A, I64) and eq(B @ B, I64)
    ok &= np.trace(A) == 0 and np.trace(B) == 0 and np.trace(A @ B) == 0
    H = A + 2 * B
    mp = (H - 3 * I64) @ (H - I64) @ (H + I64) @ (H + 3 * I64)
    ok &= eq(mp, 0 * I64)
    for lam in (-3, -1, 1, 3):
        ok &= rank_exact(H - lam * I64) == 48
    return ok


@claim("e1-2c", "H_g is one Dirac mass in a rotated frame: with B = E_(2,1)E_(1,1) (B^2 = -1, compact) and cos^2 phi = 4/5, "
       "U = cos phi + sin phi B gives U (5 E_(1,1)) U^-1 = 3 E_(1,1) + 4 E_(2,1) (exact, rational coefficients)")
def _():
    A, Bv = E[(1, 1)], E[(2, 1)]
    B = Bv @ A
    ok = eq(B @ B, -I64)
    # U X U^-1 = cos^2 X + sin cos [B, X] - sin^2 B X B   (U^-1 = cos - sin B)
    c2, sc, s2 = sp.Rational(4, 5), sp.Rational(2, 5), sp.Rational(1, 5)
    X = 5 * A
    R = sp.Matrix(X) * c2 + sp.Matrix(comm(B, X)) * sc - sp.Matrix(B @ X @ B) * s2
    ok &= R == sp.Matrix(3 * A + 4 * Bv)
    return ok


@claim("e1-2d", "H_u is not a vector of Cl(6,6): it is m1 E_(1,1) + m2 (grade-7 blade), and for m1 != m2 both nonzero it has four "
       "distinct eigenvalues, whereas every v in R^{6,6} satisfies v^2 = eta(v,v) 1 and has at most two")
def _():
    H = E[(1, 1)] + 2 * blade_mat(OMEGA1_MASK | bit(2, 1), gens12)
    ok = eq(H, E[(1, 1)] + 2 * U2[1])
    ev = np.round(np.linalg.eigvalsh(H.astype(float)), 9)
    ok &= sorted(set(ev.tolist())) == [-3.0, -1.0, 1.0, 3.0]
    # every vector squares to a scalar: v = sum c_A E_A with random integers
    rng = np.random.default_rng(1)
    c = rng.integers(-5, 6, size=12)
    v = sum(int(c[i]) * gens12[i] for i in range(12))
    ok &= eq(v @ v, int(sum(ETA12[i] * c[i] ** 2 for i in range(12))) * I64)
    return ok


@claim("e1-2e", "rest flows: J^(1), J^(2) commute with H_g and H_u; Z_u = m1 T(x)1 + m2 1(x)T has (T(x)1)^2 = (1(x)T)^2 = -1, "
       "commuting, eigenvalues i(+-m1+-m2); Z_g = J^(1) H_g has Z_g^2 = -(m1^2+m2^2): exp(-lam Z_g) has the single frequency M, "
       "exp(-lam Z_u) the two frequencies m1, m2 (float check at (m1,m2) = (3,4) and (1,sqrt2), tol 1e-9)")
def _():
    Hg = 3 * E[(1, 1)] + 4 * E[(2, 1)]
    Hu = 3 * E[(1, 1)] + 4 * U2[1]
    ok = all(eq(comm(Jx, H), 0 * I64) for Jx in (J1, J2) for H in (Hg, Hu, E[(1, 1)], E[(2, 1)], U2[1]))
    ok &= eq(T1, J1 @ E[(1, 1)]) and eq(T2, J2 @ U2[1])
    ok &= eq(T1 @ T1, -I64) and eq(T2 @ T2, -I64) and eq(comm(T1, T2), 0 * I64)
    Zg = J1 @ Hg
    ok &= eq(Zg @ Zg, -25 * I64)
    ok &= np.allclose(sla.expm(-2 * np.pi / 5 * Zg.astype(float)), np.eye(64), atol=1e-9)
    Zu = 1.0 * T1 + np.sqrt(2) * T2
    w = np.sort(np.abs(np.linalg.eigvals(Zu).imag))
    ok &= np.allclose(np.unique(np.round(w, 9)), [np.sqrt(2) - 1, np.sqrt(2) + 1], atol=1e-9)
    # quasi-periodic: no return at the period of either frequency
    ok &= not np.allclose(sla.expm(-2 * np.pi * Zu), np.eye(64), atol=1e-3)
    return ok


@claim("e1-2f", "rungs n=3,4: the graded E_(k,1) are pairwise anticommuting involutions (H_g^2 = sum m_k^2), the ungraded u_(k,1) "
       "pairwise commuting involutions whose products over every nonempty subset are traceless, so H_u has spectrum "
       "{sum_k +-m_k} with multiplicity 8^n/2^n; at n=3, (m)=(1,2,4): H_u eigenvalues -7,-5,...,7, H_g eigenvalues +-sqrt(21) (tol 1e-9)")
def _():
    ok = True
    for n in (3, 4):
        Eg, Eu = tower_sp(n)
        Id = SP.ident(8 ** n)
        for k in range(1, n + 1):
            ok &= (Eg[(k, 1)] @ Eg[(k, 1)]) == Id and (Eu[(k, 1)] @ Eu[(k, 1)]) == Id
            for l in range(k + 1, n + 1):
                ok &= anticommute_sp(Eg[(k, 1)], Eg[(l, 1)]) and commute_sp(Eu[(k, 1)], Eu[(l, 1)])
        for r in range(1, n + 1):
            for sub in itertools.combinations(range(1, n + 1), r):
                M = Id
                for k in sub:
                    M = M @ Eu[(k, 1)]
                ok &= M.trace() == 0
    # dense eigenvalue check at n = 3
    Eg, Eu = tower_sp(3)

    def dense(spm):
        n = len(spm.p)
        M = np.zeros((n, n))
        M[spm.p, np.arange(n)] = spm.s
        return M
    Hu = sum(m * dense(Eu[(k, 1)]) for k, m in zip((1, 2, 3), (1, 2, 4)))
    Hg = sum(m * dense(Eg[(k, 1)]) for k, m in zip((1, 2, 3), (1, 2, 4)))
    evu = np.round(np.linalg.eigvalsh(Hu), 9)
    ok &= sorted(set(evu.tolist())) == [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
    ok &= all(np.count_nonzero(np.abs(evu - v) < 1e-9) == 64 for v in (-7, -5, -3, -1, 1, 3, 5, 7))
    evg = np.linalg.eigvalsh(Hg)
    ok &= np.allclose(np.abs(evg), np.sqrt(21), atol=1e-9) and np.count_nonzero(evg > 0) == 256
    return ok


# ------------------------------------------------------------------------------------------------- section 3
def rank_int(M):
    return int(sp.Matrix(M).rank())


W_proj2 = I64 - P12            # 2 x projector onto W = {J1 psi = J2 psi} = {P12 = -1}
Wb_proj2 = I64 + P12


def preserves(X, Pr2):
    """X maps the image of the projector Pr2/2 into itself"""
    Q2 = 2 * I64 - Pr2
    return eq(Q2 @ X @ Pr2, 0 * I64)


@claim("e1-3a", "J^(1) = e2e3(x)1 and J^(2) = 1(x)e2e3 = E_(2,2)E_(2,3) commute and square to -1; W = ker(J^(1)-J^(2)) is the "
       "(-1)-eigenspace of the symmetric involution J^(1)J^(2), of dimension 32, with complement im(J^(1)-J^(2)) = Wbar = ker(J^(1)+J^(2)), "
       "also 32: W = S(x)_C S and Wbar = S(x)_C Sbar")
def _():
    ok = eq(J2, E[(2, 2)] @ E[(2, 3)]) and eq(J1, E[(1, 2)] @ E[(1, 3)])
    ok &= eq(comm(J1, J2), 0 * I64) and eq(J1 @ J1, -I64) and eq(J2 @ J2, -I64)
    ok &= eq(P12 @ P12, I64) and eq(P12.T, P12) and np.trace(P12) == 0
    ok &= 64 - rank_int(J1 - J2) == 32 and 64 - rank_int(J1 + J2) == 32
    ok &= eq((J1 - J2) @ W_proj2, 0 * I64)                # W is the kernel
    ok &= eq((P12 - I64) @ (J1 - J2), 0 * I64)            # image of J1 - J2 lies in {P12 = +1}
    return ok


@claim("e1-3b", "W is the (-1)-eigenspace of the surplus volume omega_S = E_(1,2)E_(1,3)E_(2,2)E_(2,3) = J^(1)J^(2); a blade preserves W iff it "
       "has an even number of surplus indices and otherwise maps W onto Wbar (e.g. 1(x)e2, E_(1,1)E_(2,2), e1e2(x)1). Commuting with "
       "J^(1) and J^(2), hence C-linear on W and on Wbar: lint^(1), lint^(2) (graded = ungraded bivectors), H_g, H_u, the eight graded and the "
       "eight ungraded interior gammas, the interior copy rotation M_12")
def _():
    ok = eq(P12, blade_mat(bit(1, 2) | bit(1, 3) | bit(2, 2) | bit(2, 3), gens12))
    sur = {bit(1, 2), bit(1, 3), bit(2, 2), bit(2, 3)}
    # every blade: commutes with omega_S iff even surplus count; then preserves W, else W -> Wbar
    smask = sum(sur)
    for m in range(1 << 12):
        cnt = bin(m & smask).count('1')
        if bcomm_sign(m, smask) != (1 if cnt % 2 == 0 else -1):
            return False
    for m in (bit(1, 1) | bit(2, 2), bit(1, 1) | bit(1, 2), OMEGA1_MASK | bit(2, 2), bit(1, 2) | bit(2, 2), bit(1, 4) | bit(2, 5)):
        B = blade_mat(m, gens12)
        cnt = bin(m & smask).count('1')
        ok &= preserves(B, W_proj2) == (cnt % 2 == 0)
        if cnt % 2:
            ok &= eq(B @ W_proj2, Wb_proj2 @ B)
    lst = []
    for k in (1, 2):
        for a, b in itertools.combinations(INT, 2):
            lst.append(E[(k, a)] @ E[(k, b)])
    ok &= all(eq(E[(2, a)] @ E[(2, b)], U2[a] @ U2[b]) for a in range(1, 7) for b in range(1, 7))
    lst += [E[(1, 1)] + E[(2, 1)], E[(1, 1)] + U2[1]]
    lst += [E[(k, mu)] for k in (1, 2) for mu in INT] + [U2[mu] for mu in INT]
    lst += [sum(ETA[a] * E[(1, a)] @ E[(2, a)] for a in INT)]
    ok &= all(eq(comm(X, Jx), 0 * I64) for X in lst for Jx in (J1, J2))
    return ok


@claim("e1-3f", "on W the surplus part of the copy rotation vanishes, E_(1,3)E_(2,3) = E_(1,2)E_(2,2) omega_S, so C_b = (1/2) sum_a eta_aa E_(1,a)E_(2,a) restricted to W equals "
       "the interior copy rotation M_12 = (1/2) sum_mu eta_mumu E_(1,mu)E_(2,mu) and is C-linear there; the four cross surplus bivectors "
       "E_(1,s)E_(2,t) preserve W but anticommute with J^(1) and J^(2), so J := J^(1)|W = J^(2)|W is not invariant under the surplus so(4)")
def _():
    ok = eq(E[(1, 3)] @ E[(2, 3)], E[(1, 2)] @ E[(2, 2)] @ P12)
    Cb2 = sum(ETA[a] * E[(1, a)] @ E[(2, a)] for a in range(1, 7))      # 2 C_b (RE Prop. 7.3(ii))
    M12x2 = sum(ETA[a] * E[(1, a)] @ E[(2, a)] for a in INT)           # 2 M_12
    ok &= eq((Cb2 - M12x2) @ W_proj2, 0 * I64)
    ok &= not eq((Cb2 - M12x2) @ Wb_proj2, 0 * I64)
    ok &= eq(comm(Cb2, J1) @ W_proj2, 0 * I64) and not eq(comm(Cb2, J1), 0 * I64)
    for s_ in (2, 3):
        for t in (2, 3):
            X = E[(1, s_)] @ E[(2, t)]
            ok &= preserves(X, W_proj2) and eq(acomm(X, J1), 0 * I64) and eq(acomm(X, J2), 0 * I64)
    return ok


@claim("e1-2g", "joint rest eigenstates: E_(1,1) and E_(2,1) anticommute, so no nonzero psi has E_(1,1)psi = s1 psi and E_(2,1)psi = s2 psi "
       "(graded copies have no joint mass shell; only H_g is sharp), while e1(x)1 and 1(x)e1 commute and every joint sign (s1,s2) occurs on 16 "
       "dimensions; likewise the graded Dirac operators D_k = sum_mu p_mu E_(k,mu) of the two copies anticommute and D_1 + D_2 squares to "
       "(p_1^2 + p_2^2) 1, whereas the ungraded ones commute")
def _():
    ok = True
    for s1 in (1, -1):
        for s2 in (1, -1):
            A = np.vstack([E[(1, 1)] - s1 * I64, E[(2, 1)] - s2 * I64])
            ok &= rank_exact(A) == 64          # trivial joint kernel
            Bm = np.vstack([E[(1, 1)] - s1 * I64, U2[1] - s2 * I64])
            ok &= 64 - rank_exact(Bm) == 16
    p1, p2 = [5, 1, 2, 3], [7, -2, 1, 4]      # components along gamma^mu = (e1, e4, e5, e6)
    D1 = sum(c * E[(1, mu)] for c, mu in zip(p1, INT))
    D2 = sum(c * E[(2, mu)] for c, mu in zip(p2, INT))
    sq = lambda p: p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
    ok &= eq(acomm(D1, D2), 0 * I64) and eq((D1 + D2) @ (D1 + D2), (sq(p1) + sq(p2)) * I64)
    D2u = sum(c * U2[mu] for c, mu in zip(p2, INT))
    ok &= eq(comm(D1, D2u), 0 * I64) and eq(D2u @ D2u, sq(p2) * I64)
    return ok


Jsel = blade_mat(bit(1, 2) | bit(1, 3) | sum(bit(2, a) for a in range(1, 7)), gens12)


@claim("e1-3c", "the selected unit of ES17 Thm 4.1 at n=2 (spectator action) is Jsel = (e2e3)(x)omega = J^(1) Gamma^(2), Gamma^(2) = 1(x)omega; "
       "it preserves W, equals J^(1) on {Gamma^(2)=+1} and -J^(1) on {Gamma^(2)=-1} (32 each), J^(2) on {Gamma^(2) omega_S = -1} and "
       "-J^(2) on {Gamma^(2) omega_S = +1} (32 each; Jsel -+ J^(2) = J^(1)(Gamma^(2) +- omega_S)), and on W equals J on 16 and -J on 16 "
       "dimensions; it anticommutes with every copy-2 generator, graded or ungraded, so neither H_g nor H_u nor any copy-2 gamma "
       "is Jsel-linear, while m1 E_(1,1) is")
def _():
    ok = eq(Jsel, np.kron(J1c, om)) and eq(Jsel, J1 @ OM2u) and eq(Jsel @ Jsel, -I64)
    ok &= eq(comm(Jsel, J1), 0 * I64) and eq(comm(Jsel, J2), 0 * I64)
    ok &= 64 - rank_int(Jsel - J1) == 32 and 64 - rank_int(Jsel + J1) == 32
    ok &= eq((Jsel - J1) @ (I64 + OM2u), 0 * I64)
    # Jsel - J2 = J1 (Gamma2 + omega_S): Jsel = +-J^(2) on {Gamma^(2) omega_S = -+1}, 32 dimensions each
    ok &= eq(Jsel - J2, J1 @ (OM2u + P12)) and eq(Jsel + J2, J1 @ (OM2u - P12))
    ok &= 64 - rank_int(Jsel - J2) == 32 and 64 - rank_int(Jsel + J2) == 32
    ok &= eq((Jsel - J2) @ (I64 - OM2u @ P12), 0 * I64) and eq((Jsel + J2) @ (I64 + OM2u @ P12), 0 * I64)
    # on W: dims where Jsel = J (= J1 = J2) and Jsel = -J
    WJp = (I64 - P12) @ (I64 + OM2u)
    WJm = (I64 - P12) @ (I64 - OM2u)
    ok &= np.trace(WJp) // 4 == 16 and np.trace(WJm) // 4 == 16
    ok &= eq((Jsel - J1) @ WJp, 0 * I64) and eq((Jsel + J1) @ WJm, 0 * I64)
    ok &= all(eq(acomm(Jsel, E[(2, a)]), 0 * I64) and eq(acomm(Jsel, U2[a]), 0 * I64) for a in range(1, 7))
    ok &= eq(comm(Jsel, E[(1, 1)]), 0 * I64)
    ok &= not eq(comm(Jsel, E[(1, 1)] + E[(2, 1)]), 0 * I64) and not eq(comm(Jsel, E[(1, 1)] + U2[1]), 0 * I64)
    return ok


@claim("e1-3d", "per-particle selection: the commutant of the eight ungraded gammas is spanned by 16 blades (H(x)H = M_4(R)); the joint "
       "commutant in M_64(R) of the eight ungraded gammas and of T^(1), T^(2) is "
       "span{1, J^(1), J^(2), J^(1)J^(2)} (all blades of Cl(6,6)); its solutions of x^2 = -1 are exactly +-J^(1), +-J^(2)")
def _():
    # ungraded gammas and T's as blades: 1(x)e_a = omega^(1) E_(2,a); 1(x)T = omega^(1) E_(2,1)E_(2,2)E_(2,3)
    gm = [bit(1, mu) for mu in INT] + [OMEGA1_MASK | bit(2, mu) for mu in INT]
    t1 = bit(1, 1) | bit(1, 2) | bit(1, 3)
    t2 = OMEGA1_MASK | bit(2, 1) | bit(2, 2) | bit(2, 3)
    ok = all(eq(blade_mat(OMEGA1_MASK | bit(2, mu), gens12), U2[mu]) for mu in INT)
    ok &= eq(blade_mat(t1, gens12), T1) and eq(blade_mat(t2, gens12), T2)
    ok &= len(blade_commutant(gm)) == 16            # H (x) H = M_4(R)
    cm = blade_commutant(gm + [t1, t2])
    want = {0, bit(1, 2) | bit(1, 3), bit(2, 2) | bit(2, 3), bit(1, 2) | bit(1, 3) | bit(2, 2) | bit(2, 3)}
    ok &= set(cm) == want
    a, b, c, d = sp.symbols('a b c d', real=True)
    # x = a + b J1 + c J2 + d J1J2 ; x^2 = -1 componentwise (J1^2 = J2^2 = -1, (J1J2)^2 = 1)
    eqs = [a * a - b * b - c * c + d * d + 1, 2 * a * b - 2 * c * d, 2 * a * c - 2 * b * d, 2 * a * d + 2 * b * c]
    sols = sp.solve(eqs, [a, b, c, d], dict=True)
    sols = {(s[a], s[b], s[c], s[d]) for s in sols}
    ok &= sols == {(0, 1, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, -1, 0)}
    # cross-check the multiplication table in the realisation
    X = 2 * I64 + 3 * J1 - J2 + 5 * P12
    ok &= eq(X @ X, (4 - 9 - 1 + 25) * I64 + (12 + 10) * J1 + (-4 - 30) * J2 + (20 - 6) * P12)
    return ok


@claim("e1-3e", "graded diagonal action (k_I = 2): the commutant of the eight graded interior gammas E_(k,mu) is spanned by 16 blades "
       "(M_2(H)); adding the six surplus bivectors leaves span{1, E_(1,2)E_(1,3)E_(2,2)E_(2,3)} with E_(1,2)E_(1,3)E_(2,2)E_(2,3) = "
       "J^(1)J^(2) of square +1: no complex structure commutes with both")
def _():
    gm = [bit(k, mu) for k in (1, 2) for mu in INT]
    cm = blade_commutant(gm)
    ok = len(cm) == 16
    sur = [bit(1, 2), bit(1, 3), bit(2, 2), bit(2, 3)]
    sbiv = [x | y for x, y in itertools.combinations(sur, 2)]
    cm2 = blade_commutant(gm + sbiv)
    ok &= set(cm2) == {0, sum(sur)}
    ok &= eq(blade_mat(sum(sur), gens12), P12)
    return ok


def restrict_to(X, basis):
    """matrix of X on the span of the orthogonal integer vectors in basis (columns), assuming X preserves it"""
    B = basis
    norms = np.sum(B * B, axis=0)
    XB = X @ B
    R = (B.T @ XB)
    assert np.all(R % norms[:, None] == 0)
    R = R // norms[:, None]
    assert eq(B @ R, XB)
    return R


def W_basis():
    cols, seen = [], set()
    for j in range(64):
        v = (I64 - P12)[:, j]
        if not v.any():
            continue
        key = tuple(np.sign(v[np.nonzero(v)[0][0]]) * v)
        if key in seen:
            continue
        seen.add(key)
        cols.append(v)
    B = np.array(cols).T
    assert B.shape == (64, 32) and eq(B.T @ B, np.diag(np.diag(B.T @ B)))
    return B


@claim("e1-3g", "restricted to W (dim 32), the eight graded interior gammas and the six surplus bivectors have commutant R (only the "
       "scalars): no complex structure on W commutes with them (the minimal eigenvalue of sum ad^T ad is 0 once; next eigenvalue "
       "> 1/2, floating point); the eight ungraded gammas together with J|W have commutant span{1, J|W} on W (dimension 2)")
def _():
    B = W_basis()
    gens = [E[(k, mu)] for k in (1, 2) for mu in INT]
    sur = [(1, 2), (1, 3), (2, 2), (2, 3)]
    gens += [E[x] @ E[y] for x, y in itertools.combinations(sur, 2)]
    R = [restrict_to(X, B) for X in gens]
    I32 = np.eye(32)
    M = sum((np.kron(I32, r) - np.kron(r.T, I32)).T @ (np.kron(I32, r) - np.kron(r.T, I32)) for r in R)
    ev = np.linalg.eigvalsh(M.astype(float))
    ok = np.count_nonzero(ev < 1e-9) == 1 and ev[1] > 0.5
    # ungraded gammas with J on W: commutant span{1, J}
    gu = [E[(1, mu)] for mu in INT] + [U2[mu] for mu in INT] + [J1]
    Ru = [restrict_to(X, B) for X in gu]
    Mu = sum((np.kron(I32, r) - np.kron(r.T, I32)).T @ (np.kron(I32, r) - np.kron(r.T, I32)) for r in Ru)
    evu = np.linalg.eigvalsh(Mu.astype(float))
    print(f"   W-commutant dims: graded gammas + surplus bivectors {np.count_nonzero(ev < 1e-9)} (gap {ev[1]:.3f}); "
          f"ungraded gammas + J {np.count_nonzero(evu < 1e-9)} (gap {evu[np.count_nonzero(evu < 1e-9)]:.3f})")
    ok &= np.count_nonzero(evu < 1e-9) == 2
    return ok


@claim("e1-3h", "up the ladder: under the graded diagonal action the commutant of the 4n interior gammas and the surplus bivectors is "
       "span{1, omega_S} (exhaustive over blades at n = 2, 3), omega_S = J^(1)...J^(n) with omega_S^2 = (-1)^n (n = 2, 3, 4): an imaginary "
       "unit is selected exactly at odd n (ES17 Prop 10.10(ii)); the common subspace J^(1) = ... = J^(n) has dimension 2*4^n (n = 2, 3, 4), and at n = 3 omega_S = -J there")
def _():
    ok = True
    for n in (2, 3):
        inter = [1 << (6 * k + mu - 1) for k in range(n) for mu in INT]
        surm = [1 << (6 * k + s_ - 1) for k in range(n) for s_ in SUR]
        sbiv = [x | y for x, y in itertools.combinations(surm, 2)]
        gm = inter + sbiv
        cm = [m for m in range(1 << (6 * n)) if all(bcomm_sign(m, g) == 1 for g in gm)]
        ok &= set(cm) == {0, sum(surm)}
    for n in (2, 3, 4):
        Eg, _ = tower_sp(n)
        Id = SP.ident(8 ** n)
        Js = [Eg[(k, 2)] @ Eg[(k, 3)] for k in range(1, n + 1)]
        om_S = Id
        for k in range(1, n + 1):
            om_S = om_S @ Eg[(k, 2)] @ Eg[(k, 3)]
        sq = om_S @ om_S
        ok &= sq == (Id if n % 2 == 0 else Id.neg())
        # dim of {J_k J_{k+1} = -1 for all k}: trace of prod (1 - J_k J_{k+1}) / 2^{n-1}, expanded over subsets
        Pk = [Js[k] @ Js[k + 1] for k in range(n - 1)]
        tot = 0
        for r in range(n):
            for sub in itertools.combinations(range(n - 1), r):
                M = Id
                for k in sub:
                    M = M @ Pk[k]
                tot += (-1) ** r * M.trace()
        ok &= tot % 2 ** (n - 1) == 0 and tot // 2 ** (n - 1) == 2 * 4 ** n
        if n == 3:
            def dense(spm):
                d = len(spm.p)
                M = np.zeros((d, d), dtype=np.int64)
                M[spm.p, np.arange(d)] = spm.s
                return M
            Id3 = np.eye(512, dtype=np.int64)
            proj = (Id3 - dense(Pk[0])) @ (Id3 - dense(Pk[1]))
            ok &= eq((dense(om_S) + dense(Js[0])) @ proj, 0 * Id3)      # omega_S = -J on the common subspace
            ok &= eq((dense(Js[0]) - dense(Js[2])) @ proj, 0 * Id3)
    return ok


# ------------------------------------------------------------------------------------------------- section 4
def number_op(gens_by_copy):
    """N = sum a+ a-, a+- = (e_i +- e_{i+3})/2; returns 4N as an integer matrix"""
    tot = 0
    for Ek in gens_by_copy:
        for i in (1, 2, 3):
            tot = tot + (Ek[i] + Ek[i + 3]) @ (Ek[i] - Ek[i + 3])
    return tot


N4 = number_op([{a: E[(1, a)] for a in range(1, 7)}, {a: E[(2, a)] for a in range(1, 7)}])
assert np.all(N4 % 4 == 0)
Nop = N4 // 4
N1op = number_op([{a: E[(1, a)] for a in range(1, 7)}]) // 4
N2op = number_op([{a: E[(2, a)] for a in range(1, 7)}]) // 4
VOL12 = blade_mat((1 << 12) - 1, gens12)


@claim("e1-4a", "N is diagonal with spectrum 0..6, multiplicities 1,6,15,20,15,6,1; N = N^(1) + N^(2) with N^(2) = 1(x)N_1c (graded and "
       "ungraded readings give the same N); (-1)^N = Omega_12 = omega(x)omega (volume element of Cl(6,6)); [N, v] = K_2 v for every v in R^{6,6} "
       "(K e_i = e_{i+3}, K e_{i+3} = e_i), i.e. ad N = K_2 = -calK on vectors")
def _():
    d = np.diag(Nop)
    ok = eq(Nop, np.diag(d))
    ok &= [int(np.count_nonzero(d == j)) for j in range(7)] == [1, 6, 15, 20, 15, 6, 1]
    N1c = number_op([{a: e[a] for a in range(1, 7)}]) // 4
    ok &= eq(Nop, N1op + N2op) and eq(N2op, np.kron(I8, N1c)) and eq(N1op, np.kron(N1c, I8))
    ok &= eq(np.diag([(-1) ** int(x) for x in d]).astype(np.int64), VOL12) and eq(VOL12, np.kron(om, om))
    for k in (1, 2):
        for i in (1, 2, 3):
            ok &= eq(comm(Nop, E[(k, i)]), E[(k, i + 3)]) and eq(comm(Nop, E[(k, i + 3)]), E[(k, i)])
    return ok


def ad_vec(a, b, n):
    """matrix of ad(E_a E_b) on V_n (basis E_1..E_6n): e_c -> 2 eta_bc e_a - 2 eta_ac e_b"""
    et = ETA[1:] * n
    M = sp.zeros(6 * n, 6 * n)
    for c in range(6 * n):
        if c == b:
            M[a, c] += 2 * et[b]
        if c == a:
            M[b, c] -= 2 * et[a]
    return M


def Kn(n):
    M = sp.zeros(6 * n, 6 * n)
    for k in range(n):
        for i in range(3):
            M[6 * k + i + 3, 6 * k + i] = 1
            M[6 * k + i, 6 * k + i + 3] = 1
    return M


def Gn(n):
    M = sp.zeros(6 * n, 6 * n)
    for k in range(n):
        for i in range(3):
            M[6 * k + i + 3, 6 * k + i] = 1
            M[6 * k + i, 6 * k + i + 3] = -1
    return M


def lie_solve(conds, n):
    """basis of {sum c_ab E_aE_b : conditions} as coefficient vectors; conds: list of functions(ad_matrix)->Matrix"""
    pairs = list(itertools.combinations(range(6 * n), 2))
    cols = []
    for (a, b) in pairs:
        X = ad_vec(a, b, n)
        cols.append(sp.Matrix.vstack(*[f(X).reshape(len(f(X)), 1) for f in conds]))
    A = sp.Matrix.hstack(*cols)
    return pairs, A.nullspace()


def ad_of(vecs, pairs, n):
    return [sum((v[i] * ad_vec(a, b, n) for i, (a, b) in enumerate(pairs)), sp.zeros(6 * n, 6 * n)) for v in vecs]


@claim("e1-4b", "bivectors of Cl(6,6) commuting with N: dimension 36, exactly those whose ad commutes with K_2 (gl(6,R), the stabiliser "
       "of the two isotropic sixes L+- = ker(K_2 -+ 1)); the other 30 split 15 + 15 with [N, X] = +-2X (a+a+ and a-a- pairs)")
def _():
    bis = [(a, b) for a, b in itertools.combinations(range(12), 2)]
    mats = [gens12[a] @ gens12[b] for a, b in bis]
    # [N, X] for each bivector, decomposed: N diagonal, so entrywise (N_t - N_s) X_ts
    d = np.diag(Nop)
    dN = d[:, None] - d[None, :]
    cnt = {}
    # dimension of {X in span of the 66 bivectors : [N, X] = val X}; N is diagonal, so [N, X]_ts = (N_t - N_s) X_ts
    for val in (0, 2, -2):
        A = np.array([((dN - val) * m).ravel() for m in mats]).T   # columns: [N,X] - val X
        cnt[val] = 66 - rank_exact(A)
    ok = cnt == {0: 36, 2: 15, -2: 15}
    # vector level: commutant of K_2 in so(6,6)
    K = Kn(2)
    pairs, ns = lie_solve([lambda X: X * K - K * X], 2)
    ok &= len(ns) == 36
    # each K-commuting bivector commutes with N at spin level
    for v in ns:
        X = sum(int(v[i]) * mats[i] for i in range(66)) if all(x.q == 1 for x in v) else None
        if X is None:
            den = sp.ilcm(*[x.q for x in v])
            X = sum(int(v[i] * den) * mats[i] for i in range(66))
        ok &= eq(comm(Nop, X), 0 * I64)
    return ok


@claim("e1-4c", "rungs n=1..4 (vectors, exact): Cent(K_n) in so(3n,3n) has dimension 9n^2 = 9,36,81,144; Cent(G_n) in so(3n,3n) has "
       "dimension 3n(3n-1) = 6,30,72,132; Cent(G_n) cap Cent(K_n) = compact part (antisymmetric matrices) = so(3n), dimension "
       "3n(3n-1)/2; on Cent(G_n), Ad_K equals the Cartan involution X -> -X^T, so the boost part anticommutes with K_n")
def _():
    ok = True
    for n in (1, 2, 3, 4):
        K, G = Kn(n), Gn(n)
        pairs, nsK = lie_solve([lambda X: X * K - K * X], n)
        _, nsG = lie_solve([lambda X: X * G - G * X], n)
        _, nsGK = lie_solve([lambda X: X * G - G * X, lambda X: X * K - K * X], n)
        _, nsGa = lie_solve([lambda X: X * G - G * X, lambda X: X * K + K * X], n)
        m = 3 * n
        ok &= len(nsK) == 9 * n * n and len(nsG) == m * (m - 1) and len(nsGK) == m * (m - 1) // 2 and len(nsGa) == m * (m - 1) // 2
        for X in ad_of(nsG, pairs, n):
            ok &= K * X * K == -X.T
        for X in ad_of(nsGK, pairs, n):
            ok &= X.T == -X
        for X in ad_of(nsGa, pairs, n):
            ok &= X.T == X
    return ok


@claim("e1-4d", "rungs n=1..4 (spin level, signed permutations): every paired rotation e_ie_j - e_{i+3}e_{j+3} (same or different "
       "copies) commutes with N, and every paired boost e_ie_{j+3} - e_je_{i+3} changes N by exactly +-2 on every matrix entry, "
       "with both signs present; their ads lie in Cent(G_n), compact and non-compact respectively")
def _():
    ok = True
    for n in (1, 2, 3, 4):
        Eg, _ = tower_sp(n)
        Eidx = {6 * (k - 1) + a - 1: Eg[(k, a)] for k in range(1, n + 1) for a in range(1, 7)}
        # N = sum_k sum_i (1 - E_ki E_k,i+3)/2, diagonal
        dim = 8 ** n
        Nd = np.zeros(dim, dtype=np.int64)
        for k in range(1, n + 1):
            for i in (1, 2, 3):
                pole = Eg[(k, i)] @ Eg[(k, i + 3)]
                ok &= bool(np.all(pole.p == np.arange(dim)))
                Nd += (1 - pole.s) // 2
        temporal = [6 * k + i for k in range(n) for i in range(3)]
        G = Gn(n)
        for A, B in itertools.combinations(temporal, 2):
            # paired rotation: E_A E_B - E_{A+3} E_{B+3}; paired boost: E_A E_{B+3} - E_B E_{A+3}
            for kind, (p1, p2) in (("rot", ((A, B), (A + 3, B + 3))), ("boost", ((A, B + 3), (B, A + 3)))):
                X1 = Eidx[p1[0]] @ Eidx[p1[1]]
                X2 = Eidx[p2[0]] @ Eidx[p2[1]]
                adX = ad_vec(p1[0], p1[1], n) - ad_vec(p2[0], p2[1], n)
                ok &= adX * G == G * adX
                ok &= (adX.T == -adX) if kind == "rot" else (adX.T == adX)
                # entrywise check of the combination X1 - X2 (signed permutations: sparse dictionary)
                ent = {}
                for X, c in ((X1, 1), (X2, -1)):
                    for j in range(dim):
                        key = (int(X.p[j]), j)
                        ent[key] = ent.get(key, 0) + c * int(X.s[j])
                ent = {k: v for k, v in ent.items() if v != 0}
                dNs = {int(Nd[t] - Nd[s]) for (t, s) in ent}
                if kind == "rot":
                    ok &= dNs == {0}
                else:
                    ok &= dNs == {2, -2}
    return ok


def spin_lift(Xvec):
    """bivector B in Cl(6,6) with ad(B) = Xvec on V_2 (exact); returns integer-scaled matrix and scale"""
    pairs = list(itertools.combinations(range(12), 2))
    A = sp.Matrix.hstack(*[ad_vec(a, b, 2).reshape(144, 1) for a, b in pairs])
    sol, params = A.gauss_jordan_solve(Xvec.reshape(144, 1))
    sol = sol.subs({p: 0 for p in params})
    den = sp.ilcm(*[x.q for x in sol])
    B = sum(int(sol[i] * den) * (gens12[a] @ gens12[b]) for i, (a, b) in enumerate(pairs))
    return B, den


@claim("e1-4e", "second rung: the copy rotation C_b = (1/2) sum eta_aa E_(1,a)E_(2,a) commutes with N but not with N^(1); "
       "U = (1/8) prod_a (1 + eta_aa E_(1,a)E_(2,a)) = exp(pi C_b / 2) (float check, tol 1e-12) carries N^(1) to N^(2); the spin lift of the copy boost eps(x)G "
       "changes N by +-2 only (both signs)")
def _():
    Cb2 = sum(ETA[a] * E[(1, a)] @ E[(2, a)] for a in range(1, 7))        # 2 C_b
    ok = eq(comm(Nop, Cb2), 0 * I64) and not eq(comm(N1op, Cb2), 0 * I64)
    U8 = I64.copy()
    for a in range(1, 7):
        U8 = U8 @ (I64 + ETA[a] * E[(1, a)] @ E[(2, a)])
    Uinv8 = I64.copy()
    for a in reversed(range(1, 7)):
        Uinv8 = Uinv8 @ (I64 - ETA[a] * E[(1, a)] @ E[(2, a)])
    ok &= eq(U8 @ Uinv8, 64 * I64)
    ok &= np.allclose(sla.expm(np.pi / 4 * Cb2.astype(float)), U8 / 8.0, atol=1e-12)
    ok &= eq(U8 @ N1op @ Uinv8, 64 * N2op)
    # copy boost eps (x) G on V_2: e^(1)_a -> ..., in the basis (copy1 e1..e6, copy2 e1..e6)
    eps = sp.Matrix([[0, -1], [1, 0]])
    Gs = Gn(1)
    Xv = sp.kronecker_product(eps, Gs)
    B, den = spin_lift(Xv)
    # check it is a lift: [B, E_c] = den * sum_d Xv[d,c] E_d
    ok &= all(eq(comm(B, gens12[c]), den * sum(int(Xv[dd, c]) * gens12[dd] for dd in range(12))) for c in range(12))
    d = np.diag(Nop)
    dN = d[:, None] - d[None, :]
    vals = set(dN[B != 0].tolist())
    ok &= vals == {2, -2}
    ok &= Xv * Gn(2) == Gn(2) * Xv
    ok &= Xv.T == Xv     # non-compact (symmetric)
    return ok


@claim("e1-4f", "N against the two Lorentz algebras of one constituent: in lint = span{e_ae_b: a,b in {1,4,5,6}} the elements commuting "
       "with N form R e1e4 (a boost, the pole operator); the rotations e4e5, e4e6, e5e6 and the boosts e1e5, e1e6 do not commute with "
       "N, nor does the charge J = e2e3; in lG the commutant of N is the compact so(3) (paired rotations), and no nonzero boost of lG "
       "commutes with N")
def _():
    N1c = number_op([{a: e[a] for a in range(1, 7)}]) // 4
    lint = [e[a] @ e[b] for a, b in itertools.combinations(INT, 2)]
    A = np.array([comm(N1c, X).ravel() for X in lint]).T
    ok = 6 - rank_exact(A) == 1 and eq(comm(N1c, e[1] @ e[4]), 0 * I8)
    ok &= all(not eq(comm(N1c, X), 0 * I8) for X in (e[4] @ e[5], e[4] @ e[6], e[5] @ e[6], e[1] @ e[5], e[1] @ e[6], J1c))
    # lG: Cent(G) in so(3,3), 6-dim; its intersection with Cent(N)
    pairs, nsG = lie_solve([lambda X: X * Gn(1) - Gn(1) * X], 1)
    mats = [e[a + 1] @ e[b + 1] for a, b in pairs]

    def tomat(v):
        den = sp.ilcm(*[x.q for x in v])
        return sum(int(v[i] * den) * mats[i] for i in range(15))
    lG = [tomat(v) for v in nsG]
    A = np.array([comm(N1c, X).ravel() for X in lG]).T
    ok &= len(lG) == 6 and 6 - rank_exact(A) == 3
    rots = [e[a] @ e[b] - e[a + 3] @ e[b + 3] for a, b in ((1, 2), (1, 3), (2, 3))]
    boosts = [e[a] @ e[b + 3] - e[b] @ e[a + 3] for a, b in ((1, 2), (1, 3), (2, 3))]
    ok &= all(eq(comm(N1c, X), 0 * I8) for X in rots)
    B = np.array([comm(N1c, X).ravel() for X in boosts]).T
    ok &= rank_exact(B) == 3          # no nonzero combination of the boosts commutes with N
    # rots and boosts lie in lG
    Lspan = np.array([X.ravel() for X in lG]).T
    ok &= all(rank_exact(np.column_stack([Lspan, X.ravel()])) == 6 for X in rots + boosts)
    return ok


@claim("e1-4g", "(-1)^N = Omega_12 commutes with every bivector and anticommutes with every generator of Cl(6,6): even elements "
       "change N by an even amount, so no element of so(6,6) takes an odd number of quanta to an even one")
def _():
    par = np.diag([(-1) ** int(x) for x in np.diag(Nop)]).astype(np.int64)
    ok = all(eq(acomm(par, g), 0 * I64) for g in gens12)
    ok &= all(eq(comm(par, gens12[a] @ gens12[b]), 0 * I64) for a, b in itertools.combinations(range(12), 2))
    return ok


# ------------------------------------------------------------------------------------------------- section 5
DeltaGen = {}
for a, b in itertools.combinations(INT, 2):
    DeltaGen[(a, b)] = E[(1, a)] @ E[(1, b)] + E[(2, a)] @ E[(2, b)]      # 2 Delta_ab
C4 = -sum(ETA[a] * ETA[b] * DeltaGen[(a, b)] @ DeltaGen[(a, b)] for a, b in itertools.combinations(INT, 2))   # 4 C_Delta
G1 = np.kron(om, I8)
G2 = OM2u


def tr_dim(*proj2):
    """dimension of the joint image of commuting 2x-projectors (each = 1 +- involution)"""
    M = I64.copy()
    for p in proj2:
        M = M @ p
    t = np.trace(M)
    assert t % (2 ** len(proj2)) == 0
    return int(t // 2 ** len(proj2))


def Cproj(c):
    """4 x 64 ... exact spectral projector of C (values 0, 3, 4), scaled to an integer matrix; returns (M, scale)"""
    vals = [0, 3, 4]
    M, sc = I64.copy(), 1
    for v in vals:
        if v != c:
            M = M @ (C4 - 4 * v * I64)
            sc *= 4 * (c - v)
    return M, sc


def content(*proj2):
    out = {}
    for c in (0, 3, 4):
        Pc, sc = Cproj(c)
        M = Pc.copy()
        for p in proj2:
            M = M @ p
        t = np.trace(M)
        den = sc * 2 ** len(proj2)
        assert t % den == 0
        if t // den:
            out[c] = int(t // den)
    return out


@claim("e1-5a", "the ungraded swap P (x(x)y -> y(x)x) satisfies P(e_a(x)1)P = 1(x)e_a, so P E_(1,a) P = omega^(1)E_(2,a) has grade 7; "
       "P commutes with every diagonal Lorentz generator, with N, with J^(1)J^(2) (so it preserves W) and with H_u at m1 = m2, not with H_g")
def _():
    ok = eq(Psw @ Psw, I64) and all(eq(Psw @ E[(1, a)] @ Psw, U2[a]) for a in range(1, 7))
    ok &= all(eq(comm(Psw, D), 0 * I64) for D in DeltaGen.values())
    ok &= eq(comm(Psw, Nop), 0 * I64) and eq(comm(Psw, P12), 0 * I64)
    ok &= eq(comm(Psw, E[(1, 1)] + U2[1]), 0 * I64) and not eq(comm(Psw, E[(1, 1)] + E[(2, 1)]), 0 * I64)
    return ok


@claim("e1-5b", "the Koszul swap tau_K = D P (RE d5:prop:cmrel(iii)) satisfies tau_K E_(1,a) tau_K = E_(2,a); it commutes with the diagonal "
       "Lorentz generators, N, J^(1)J^(2) and H_g at m1 = m2, not with H_u; D = (1 + Gamma1 + Gamma2 - Gamma1 Gamma2)/2, so tau_K = P except "
       "on the sector Gamma^(1) = Gamma^(2) = -1, where tau_K = -P")
def _():
    ok = eq(tauK @ tauK, I64) and all(eq(tauK @ E[(1, a)] @ tauK, E[(2, a)]) for a in range(1, 7))
    ok &= all(eq(comm(tauK, D), 0 * I64) for D in DeltaGen.values())
    ok &= eq(comm(tauK, Nop), 0 * I64) and eq(comm(tauK, P12), 0 * I64)
    ok &= eq(comm(tauK, E[(1, 1)] + E[(2, 1)]), 0 * I64) and not eq(comm(tauK, E[(1, 1)] + U2[1]), 0 * I64)
    ok &= eq(2 * Dk, I64 + G1 + G2 - G1 @ G2)
    ok &= eq((tauK + Psw) @ (I64 - G1) @ (I64 - G2), 0 * I64)
    ok &= eq((tauK - Psw) @ (4 * I64 - (I64 - G1) @ (I64 - G2)), 0 * I64)
    return ok


@claim("e1-5c", "Casimir C = -sum eta^aa eta^bb Delta_ab^2 (ES17 Sec. 10.2 normalisation) is 3/2 on one constituent; on S_2 it satisfies "
       "C(C-3)(C-4) = 0; on one constituent the boosts are J Gamma times the dual rotations, e1e_{i+3} = J Gamma (dual), so Gamma = +-1 "
       "are the two Weyl halves")
def _():
    C1 = -sum(ETA[a] * ETA[b] * (e[a] @ e[b]) @ (e[a] @ e[b]) for a, b in itertools.combinations(INT, 2))  # 4 C on one copy
    ok = eq(C1, 6 * I8)
    ok &= eq(C4 @ (C4 - 12 * I64) @ (C4 - 16 * I64), 0 * I64)
    dual = {4: e[5] @ e[6], 5: e[6] @ e[4], 6: e[4] @ e[5]}
    ok &= all(eq(e[1] @ e[b], J1c @ om @ dual[b]) for b in (4, 5, 6))
    ok &= all(eq(comm(om, e[a] @ e[b]), 0 * I8) for a, b in itertools.combinations(INT, 2))
    return ok


@claim("e1-5d", "on W = S(x)_C S (dim 32): P = -1 (Lambda^2_C, dim 12) carries C = 0 on 4 and C = 3 on 8, i.e. 2(0,0) + (1/2,1/2); "
       "P = +1 (Sym^2_C, dim 20) carries C = 4 on 12 and C = 3 on 8, i.e. (1,0) + (0,1) + (1/2,1/2); by chirality: Gamma1 = Gamma2 = +1 "
       "gives P=+1: C=4 on 6, P=-1: C=0 on 2, and the same for Gamma1 = Gamma2 = -1; the mixed sector (16) has C = 3, split 8 + 8 by P")
def _():
    Wp = I64 - P12
    ok = tr_dim(Wp) == 32
    ok &= content(Wp, I64 - Psw) == {0: 4, 3: 8} and content(Wp, I64 + Psw) == {3: 8, 4: 12}
    for s in (1, -1):
        sec = [Wp, I64 + s * G1, I64 + s * G2]
        ok &= content(*sec, I64 + Psw) == {4: 6} and content(*sec, I64 - Psw) == {0: 2}
    mixed = [Wp, I64 - G1 @ G2]
    ok &= content(*mixed) == {3: 16} and content(*mixed, I64 + Psw) == {3: 8} and content(*mixed, I64 - Psw) == {3: 8}
    return ok


@claim("e1-5e", "on W the Koszul swap has eigenspaces of dimension 16 with identical Casimir content (C = 0, 3, 4 on 2, 8, 6), but the "
       "spin-one part of tau_K = +1 lies in Gamma1 = Gamma2 = +1 and that of tau_K = -1 in Gamma1 = Gamma2 = -1; on all of S_2 each tau_K "
       "sector has C = 0, 3, 4 on 4, 16, 12 (restates RE d5:prop:spin(iii))")
def _():
    Wp = I64 - P12
    ok = content(Wp, I64 + tauK) == {0: 2, 3: 8, 4: 6} and content(Wp, I64 - tauK) == {0: 2, 3: 8, 4: 6}
    ok &= content(Wp, I64 + tauK, I64 + G1, I64 + G2) == {4: 6} and content(Wp, I64 - tauK, I64 + G1, I64 + G2) == {0: 2}
    ok &= content(Wp, I64 - tauK, I64 - G1, I64 - G2) == {4: 6} and content(Wp, I64 + tauK, I64 - G1, I64 - G2) == {0: 2}
    ok &= content(I64 + tauK) == {0: 4, 3: 16, 4: 12} and content(I64 - tauK) == {0: 4, 3: 16, 4: 12}
    return ok


@claim("e1-5f", "on Wbar = S(x)_C Sbar the ungraded swap is C-antilinear (P J^(1) = -J^(1) P there), J^(1) exchanges its two eigenspaces, "
       "and both carry C = 0, 3, 4 on 2, 8, 6; on all of S_2, P = -1 (dim 28) carries C = 0, 3, 4 on 6, 16, 6 and P = +1 (dim 36) on "
       "2, 16, 18: the ungraded swap separates Lorentz content on W only; on Wbar both tau_K sectors carry C = 0, 3, 4 on 2, 8, 6")
def _():
    Wb = I64 + P12
    ok = eq((Psw @ J1 + J1 @ Psw) @ Wb, 0 * I64) and eq((Psw @ J1 - J1 @ Psw) @ (I64 - P12), 0 * I64)
    ok &= eq(J1 @ (I64 + Psw) @ Wb, (I64 - Psw) @ J1 @ Wb)
    ok &= content(Wb, I64 + Psw) == {0: 2, 3: 8, 4: 6} and content(Wb, I64 - Psw) == {0: 2, 3: 8, 4: 6}
    ok &= content(I64 - Psw) == {0: 6, 3: 16, 4: 6} and content(I64 + Psw) == {0: 2, 3: 16, 4: 18}
    ok &= content(Wb, I64 + tauK) == {0: 2, 3: 8, 4: 6} and content(Wb, I64 - tauK) == {0: 2, 3: 8, 4: 6}
    return ok


# ------------------------------------------------------------------------------------------------- summary
print(f"# runtime {time.time() - T0:.1f}s")
if FAILS:
    print("FAILED:", ", ".join(FAILS))
    sys.exit(1)
sys.exit(0)
