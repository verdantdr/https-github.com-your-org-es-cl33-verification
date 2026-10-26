#!/usr/bin/env python3
"""Independent referee re-check of section r4 ("The bridge T = nJ(n) between the two readings").

    python3 investigations/two_readings/verify/s4_check.py

Prints one line 'PASS s4-<id>: <claim>' (or FAIL) per claim and exits 0 iff every check passes.  Written without
importing or copying verify/r4_bridge.py, and with different methods where possible:

  * identities in a general clock n are checked in an abstract Clifford algebra of blades (bitmask products with
    the reordering sign and the metric), with sympy coefficients, not in the 8x8 realisation;
  * module statements use the realisation of eq. (1) of the paper as numpy int64 matrices; rational clocks are
    scaled to integer vectors N = |N| n with |N| an integer, and every factor |N| is kept explicit;
  * dimensions of solution spaces are bounded above by the rank modulo the prime p = 2^31 - 1 (rank_p <= rank_Q,
    so nullity_Q <= nullity_p) and bounded below by exhibiting exact integer solutions that are checked to be
    independent; equality of the two bounds gives the exact dimension over Q;
  * linear (bilinear) identities in X or n are checked exactly on bases.
No floating point is used.  Runtime: a few seconds.
"""
import itertools
import sys
from fractions import Fraction
from functools import reduce

import numpy as np
import sympy as sp

FAILS = []


def check(cid, claim, ok):
    ok = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'} s4-{cid}: {claim}")
    if not ok:
        FAILS.append(cid)


# =====================================================================================================================
# Abstract Clifford algebra Cl(3,3): multivectors as {bitmask: coefficient}; bit i <-> e_{i+1}
# =====================================================================================================================
SIG = [1, 1, 1, -1, -1, -1]


def _reorder(a, b):
    a >>= 1
    s = 0
    while a:
        s += bin(a & b).count("1")
        a >>= 1
    return -1 if s & 1 else 1


def _bprod(a, b):
    sgn = _reorder(a, b)
    common = a & b
    for i in range(6):
        if common >> i & 1:
            sgn *= SIG[i]
    return sgn, a ^ b


class MV:
    def __init__(self, d=None):
        self.d = {k: v for k, v in (d or {}).items() if sp.simplify(v) != 0} if d else {}

    def __add__(self, o):
        o = o if isinstance(o, MV) else MV({0: o})
        d = dict(self.d)
        for k, v in o.d.items():
            d[k] = d.get(k, 0) + v
        return MV(d)

    __radd__ = __add__

    def __neg__(self):
        return MV({k: -v for k, v in self.d.items()})

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        if not isinstance(o, MV):
            return MV({k: v * o for k, v in self.d.items()})
        d = {}
        for a, x in self.d.items():
            for b, y in o.d.items():
                s, c = _bprod(a, b)
                d[c] = d.get(c, 0) + s * x * y
        return MV({k: sp.expand(v) for k, v in d.items()})

    def __rmul__(self, o):
        return MV({k: v * o for k, v in self.d.items()})

    def iszero(self):
        return all(sp.expand(v) == 0 for v in self.d.values())

    def grade(self, r):
        return MV({k: v for k, v in self.d.items() if bin(k).count("1") == r})


def eb(*idx):           # blade e_{i1} e_{i2} ... (indices 1..6, increasing or not)
    out = MV({0: 1})
    for i in idx:
        out = out * MV({1 << (i - 1): 1})
    return out


E = [None] + [eb(i) for i in range(1, 7)]
One = MV({0: 1})


def mveq(a, b):
    return (a - b).iszero()


def acomm(a, b):
    return a * b + b * a


def comm(a, b):
    return a * b - b * a


# =====================================================================================================================
# The 8x8 realisation (integer matrices)
# =====================================================================================================================
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)
I8 = np.eye(8, dtype=np.int64)
Z8 = np.zeros((8, 8), dtype=np.int64)
kr = lambda *m: reduce(np.kron, m)
e = [None, kr(sx, I2, I2), kr(sz, sx, I2), kr(sz, sz, sx), kr(ep, I2, I2), kr(sz, ep, I2), kr(sz, sz, ep)]
s = [None, e[4], e[5], e[6]]
cyc = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
eta = [None, 1, 1, 1, -1, -1, -1]


def mp(*ms):
    return reduce(lambda a, b: a @ b, ms, I8)


def blade(I):
    return mp(*[e[i] for i in I])


def to_mat(x):
    """realise an abstract multivector with integer coefficients"""
    out = np.zeros((8, 8), dtype=object)
    for k, v in x.d.items():
        out = out + int(v) * blade([i + 1 for i in range(6) if k >> i & 1])
    return out.astype(np.int64)


def eqm(a, b):
    return np.array_equal(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))


def iszm(a):
    return not np.any(np.asarray(a))


def tv(N):          # sum N_i e_i  (integer N)
    return sum((int(N[i]) * e[i + 1] for i in range(3)), Z8)


def sv(v):
    return sum((int(v[i]) * s[i + 1] for i in range(3)), Z8)


T = blade([1, 2, 3])
S3 = blade([4, 5, 6])
GAM = blade([1, 2, 3, 4, 5, 6])
OMI = [None] + [e[i] @ e[i + 3] for i in (1, 2, 3)]


def J(N):           # J(N) = T N  (linear in N)
    return T @ tv(N)


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# coordinates of a vector element of V along e_1..e_6 (coordinates (B;E))
def vcoords(v):
    c = []
    for d in range(1, 7):
        t = int(np.trace(e[d] @ v)) * eta[d]
        assert t % 8 == 0
        c.append(t // 8)
    assert eqm(sum((c[d - 1] * e[d] for d in range(1, 7)), Z8), v), "not a vector"
    return c


def admat(Y, scale=1):
    """6x6 integer matrix of X -> [Y,X] on V (columns: images of e_1..e_6), for Y with [Y,e_c] integer*scale"""
    cols = []
    for c in range(1, 7):
        w = Y @ e[c] - e[c] @ Y
        cols.append(vcoords(w))
    return np.array(cols, dtype=np.int64).T


# Fock modes, octants
vac = np.zeros(8, dtype=np.int64); vac[0] = 1
AP2 = [None] + [e[i] + e[i + 3] for i in (1, 2, 3)]       # 2 a^+_i
AM2 = [None] + [e[i] - e[i + 3] for i in (1, 2, 3)]       # 2 a^-_i
f = [None] + [e[k] @ vac for k in (1, 2, 3)]
g = [None, None, None, None]
for (m, j, k) in cyc:
    g[m] = e[j] @ e[k] @ vac
full = e[1] @ e[2] @ e[3] @ vac


def fv(v):
    return sum((int(v[i]) * f[i + 1] for i in range(3)), np.zeros(8, dtype=np.int64))


def gv(v):
    return sum((int(v[i]) * g[i + 1] for i in range(3)), np.zeros(8, dtype=np.int64))


def Phi(X):          # X = (B1,B2,B3,E1,E2,E3)
    return fv(X[3:]) + gv(X[:3])


PHI = np.array([Phi([1 if r == c else 0 for r in range(6)]) for c in range(6)], dtype=np.int64).T   # 8x6
Gm = np.zeros((6, 6), dtype=np.int64)
for i in range(3):
    Gm[3 + i, i] = 1          # G e_i = e_{i+3}
    Gm[i, 3 + i] = -1         # G e_{i+3} = -e_i
ETA6 = np.diag([1, 1, 1, -1, -1, -1]).astype(np.int64)

# Lorentz generators of l_G (Clifford elements, times 2 to stay integer)
L2 = [None] * 4
K2 = [None] * 4
for (i, j, k) in cyc:
    L2[i] = e[j + 3] @ e[k + 3] - e[j] @ e[k]
    K2[i] = e[k] @ e[j + 3] - e[j] @ e[k + 3]
Q2 = [None] + [-(T @ L2[i]) for i in (1, 2, 3)]          # 2 Q_i = 2 rho_oct(K_i)


def rhoL2(i):        # 2 rho_n(L_i) = s_j s_k  (clock independent)
    j, k = [c for c in cyc if c[0] == i][0][1:]
    return s[j] @ s[k]


def rhoK2N(i, N):    # 2|N| rho_n(K_i) = s_i N
    return s[i] @ tv(N)


# ---------------------------------------------------------------------------------------------------------------------
# exact linear algebra helpers
# ---------------------------------------------------------------------------------------------------------------------
P = 2147483647


def rank_mod_p(A):
    A = np.array(A, dtype=np.int64) % P
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), P - 2, P)
        A[r] = (A[r] * inv) % P
        below = np.nonzero(A[r + 1:, c])[0] + r + 1
        if len(below):
            fac = A[below, c].copy()
            A[below] = (A[below] - (fac[:, None] * A[r][None, :]) % P) % P
        r += 1
    return r


def nullity_p(A):
    A = np.asarray(A)
    return A.shape[1] - rank_mod_p(A)


def rank_Q(vectors):
    """exact rank of a list of integer vectors (sympy, small sizes only)"""
    if not vectors:
        return 0
    return sp.Matrix([[int(x) for x in np.ravel(v)] for v in vectors]).rank()


def kron_r(A, B):
    return np.kron(np.asarray(A, dtype=np.int64), np.asarray(B, dtype=np.int64))


def lin_XA_minus_BX(A, B):
    """matrix of vec(X) -> vec(X A - B X), row-major vec"""
    return kron_r(I8, np.asarray(A).T) - kron_r(B, I8)


def lin_XA_plus_BX(A, B):
    return kron_r(I8, np.asarray(A).T) + kron_r(B, I8)


def satisfies(X, A, B, sign=-1):
    return iszm(X @ A + sign * B @ X)


# integer clocks N = |N| n with integer norm, and oriented frames (N, U1, U2): N U1 U2 = |N|^3 T
CLOCKS = {
    'e1': ([1, 0, 0], [0, 1, 0], [0, 0, 1], 1),
    '236': ([2, 3, 6], [3, -6, 2], [6, 2, -3], 7),
    '122': ([1, 2, 2], [2, 1, -2], [-2, 2, -1], 3),
    '-148': ([-1, 4, 8], [4, -7, 4], [8, 4, -1], 9),
}
for key, (N, U1, U2, r) in CLOCKS.items():
    assert dot(N, N) == r * r and dot(U1, U1) == r * r and dot(U2, U2) == r * r and dot(N, U1) == 0
    assert cross(N, U1) == [r * x for x in U2]

# =====================================================================================================================
# s4-1  conventions
# =====================================================================================================================
ok = all(eqm(e[a] @ e[b] + e[b] @ e[a], 2 * eta[a] * I8 if a == b else Z8) for a in range(1, 7) for b in range(1, 7))
ok &= all(eqm(OMI[i], np.diag(np.diag(OMI[i]))) for i in (1, 2, 3))
NOP4 = sum((AP2[i] @ AM2[i] for i in (1, 2, 3)), Z8)          # 4N
signs = [tuple(int(OMI[i][b, b]) for i in (1, 2, 3)) for b in range(8)]
ok &= eqm(NOP4, np.diag(np.diag(NOP4))) and all(int(NOP4[b, b]) == 4 * sgs.count(-1) for b, sgs in enumerate(signs))
ok &= signs[0] == (1, 1, 1) and signs[7] == (-1, -1, -1) and full[7] == 1 and np.count_nonzero(full) == 1
ok &= all(iszm(AM2[i] @ vac) for i in (1, 2, 3))
ok &= all(eqm(4 * f[k], 2 * AP2[k] @ vac) for k in (1, 2, 3))
ok &= all(eqm(4 * g[m], AP2[j] @ AP2[k] @ vac) for (m, j, k) in cyc)
ok &= eqm(T @ T, -I8) and eqm(T.T, -T) and eqm(GAM.T, GAM) and eqm(GAM @ GAM, I8) and eqm(T @ GAM, -S3)
ok &= eqm(S3.T, S3) and eqm(S3 @ S3, I8)
# T pairs a+_i|vac> with a+_j a+_k|vac> (j<k) with signs (-1,+1,-1)
pairs = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
vals = [int(f[i] @ T @ (e[pairs[i][0]] @ e[pairs[i][1]] @ vac)) for i in (1, 2, 3)]
ok &= vals == [-1, 1, -1]
ok &= all(int(f[i] @ T @ (e[a] @ e[b] @ vac)) == 0 for i in (1, 2, 3) for (a, b) in pairs.values() if (a, b) != pairs[i])
check(1, "conventions: Clifford relations; omega_i diagonal; 4N = sum (2a+_i)(2a-_i) diagonal, N = number of minus signs; "
         "vacuum (+,+,+) N=0, full (-,-,-) N=3; f_k=a+_k|vac>, g_m=a+_ja+_k|vac>; T^2=-1, T^T=-T, Gamma^T=Gamma, Gamma^2=1, "
         "T Gamma=-e4e5e6 (symmetric, square 1); T pairs a+_i|vac> with a+_ja+_k|vac> (j<k) with signs (-1,+1,-1), zero otherwise", ok)

# =====================================================================================================================
# s4-2  the identity, symbolic in n, in the abstract blade algebra
# =====================================================================================================================
n1, n2, n3 = sp.symbols('n1 n2 n3', real=True)
nn2 = n1 ** 2 + n2 ** 2 + n3 ** 2
nA = n1 * E[1] + n2 * E[2] + n3 * E[3]
TA = eb(1, 2, 3)
JA = TA * nA
ok = mveq(JA, nA * TA) and mveq(JA, n1 * eb(2, 3) + n2 * eb(3, 1) + n3 * eb(1, 2))
ok &= mveq(nA * nA, nn2 * One) and mveq(JA * JA, -nn2 * One)
ok &= mveq(nA * JA, nn2 * TA) and mveq(JA * nA, nn2 * TA)
ok &= mveq(comm(JA, nA), MV()) and all(mveq(comm(JA, E[a]), MV()) for a in (4, 5, 6))
ok &= all(mveq(acomm(TA, E[a]), MV()) for a in (4, 5, 6)) and all(mveq(comm(TA, E[a]), MV()) for a in (1, 2, 3))
# injectivity: the coefficient map n -> J(n) has rank 3
ok &= sp.Matrix([[sp.diff(JA.d.get(k, 0), x) for x in (n1, n2, n3)] for k in (6, 5, 3)]).rank() == 3
# J(n) = u1 u2 on the rational frames
for key, (N, U1, U2, r) in CLOCKS.items():
    ok &= eqm(tv(N) @ tv(U1) @ tv(U2), r ** 3 * T) and eqm(tv(U1) @ tv(U2), r * J(N))
check(2, "J(n)=Tn=nT=n1e2e3+n2e3e1+n3e1e2, n^2=|n|^2, J(n)^2=-|n|^2, nJ(n)=J(n)n=|n|^2 T (symbolic n, abstract blade algebra); "
         "J(n) commutes with n and s_a; n -> J(n) injective; n u1 u2 = T and J(n) = u1 u2 on four rational oriented frames", ok)

# =====================================================================================================================
# s4-3  rotations: so(3)_+ and so(3)_Delta; rotors of Spin(V+); a finite diagonal rotation
# =====================================================================================================================
ok = True
tA = {i: eb(j, k) * sp.Rational(1, 2) for (i, j, k) in cyc}
LA = {i: (eb(j + 3, k + 3) - eb(j, k)) * sp.Rational(1, 2) for (i, j, k) in cyc}
same_sign = [eb(a, b) for a, b in itertools.combinations((1, 2, 3), 2)] + [eb(a, b) for a, b in itertools.combinations((4, 5, 6), 2)]
mixed = [eb(a, b) for a in (1, 2, 3) for b in (4, 5, 6)]
ok &= all(mveq(comm(Y, TA), MV()) for Y in same_sign) and all(mveq(acomm(Y, TA), MV()) for Y in mixed)
for Y in list(tA.values()) + list(LA.values()):
    Yn = comm(Y, nA)
    ok &= mveq(Yn.grade(1), Yn) and all(k in (1, 2, 4) for k in Yn.d)          # [Y,n] in V^+
    ok &= mveq(comm(Y, JA), TA * Yn)                                           # [Y,J(n)] = J([Y,n])
# so(3)_+ does not commute with G, so(3)_Delta does (6x6 action on V)
ok &= all(not eqm(admat(2 * (e[j] @ e[k])) @ Gm, Gm @ admat(2 * (e[j] @ e[k]))) for (i, j, k) in cyc)
ok &= all(eqm(admat(L2[i]) @ Gm, Gm @ admat(L2[i])) for i in (1, 2, 3))
# rotor q = |N| + N e1 (proportional to 1 + n e1): q e1 = n q
for key, (N, U1, U2, r) in CLOCKS.items():
    if key == 'e1':
        continue
    q = r * I8 + tv(N) @ e[1]
    qt = r * I8 + e[1] @ tv(N)                      # reverse
    den = 2 * r * (r + N[0])
    ok &= eqm(q @ qt, den * I8) and eqm(r * (q @ e[1]), tv(N) @ q)
    conj = lambda X: q @ X @ qt                      # = den * q X q^-1
    ok &= eqm(conj(T), den * T) and all(eqm(conj(s[a]), den * s[a]) for a in (1, 2, 3)) and eqm(conj(S3), den * S3)
    ok &= eqm(r * conj(J([1, 0, 0])), den * J(N))
    ok &= eqm(r * conj(e[1] @ S3), den * tv(N) @ S3)                        # Omega_e1 -> Omega_n
    ok &= all(eqm(r * conj(s[i] @ e[1]), den * s[i] @ tv(N)) for i in (1, 2, 3))   # phi_e1(K_i) -> phi_n(K_i)
    ok &= eqm(q.T @ q, den * I8)                                             # orthogonal up to scale
# finite diagonal rotation about e3, rational angle: R = 1 + sin L_3 + (1-cos) L_3^2 with L_3 = L2[3]/2
tt = Fraction(1, 2)
sn, cs = Fraction(2) * tt / (1 + tt * tt), (1 - tt * tt) / (1 + tt * tt)   # 4/5, 3/5
L3 = L2[3]
ok &= eqm(L3 @ L3 @ L3, -4 * L3)                                            # (L2/2)^3 = -(L2/2)
# 4R = 4 + 2 sn L2 + (1-cs) L2^2 ; with sn=4/5, cs=3/5: 20 R = 20 + 8 L2 + 2 L2^2
R20 = 20 * I8 + 8 * L3 + 2 * (L3 @ L3)
Rinv20 = 20 * I8 - 8 * L3 + 2 * (L3 @ L3)
ok &= eqm(R20 @ Rinv20, 400 * I8)
rot = lambda v: [int(cs * 5) * v[0] - int(sn * 5) * v[1], int(sn * 5) * v[0] + int(cs * 5) * v[1], 5 * v[2]]   # 5 * R v
for key, (N, U1, U2, r) in CLOCKS.items():
    ok &= eqm(R20 @ T @ Rinv20, 400 * T) and eqm(5 * (R20 @ J(N) @ Rinv20), 400 * J(rot(N)))
    ok &= eqm(5 * (R20 @ tv(N) @ Rinv20), 400 * tv(rot(N))) and eqm(5 * (R20 @ sv(N) @ Rinv20), 400 * sv(rot(N)))
check(3, "same-sign bivectors commute with T, mixed ones anticommute; for Y in so(3)_+ + so(3)_Delta: [Y,n] in V^+ and "
         "[Y,J(n)]=J([Y,n]) (symbolic n); so(3)_+ does not commute with G, so(3)_Delta does; the rotors q ~ 1+n e1 (three rational "
         "clocks) are orthogonal up to scale, fix T, s_a, e4e5e6 and carry J(e1), Omega_e1, phi_e1(K_i) to J(n), Omega_n, phi_n(K_i); "
         "the diagonal rotation exp(theta L_3) (cos=3/5) fixes T, maps J(n) to J(Rn) and rotates both triples alike", ok)

# =====================================================================================================================
# s4-4  permutations of the temporal axes, alone and paired, built from reflections
# =====================================================================================================================
def U_transp(i, j, paired):
    """implementing element of e_i <-> e_j (and e_{i+3} <-> e_{j+3} if paired), all else fixed"""
    u = e[i] - e[j]
    if not paired:
        return u @ GAM                        # odd: X -> u Gamma X (u Gamma)^-1 = reflection in u^perp
    w = e[i + 3] - e[j + 3]
    return u @ w                              # even: product of the two reflections (each -uXu^-1)


def perm_U(p, paired):
    """element implementing e_{b+1} -> e_{p[b]+1} (and the partners if paired), as a product of at most two transpositions"""
    ts = [(0, 1), (0, 2), (1, 2)]
    words = [[]] + [[t] for t in ts] + [[t1, t2] for t1 in ts for t2 in ts]
    for w in words:
        img = list(range(3))
        for (a, b) in reversed(w):           # the rightmost factor acts first
            img = [b if x == a else a if x == b else x for x in img]
        if img == list(p):
            U = I8.copy()
            for (a, b) in w:
                U = U @ U_transp(a + 1, b + 1, paired)
            return U
    raise RuntimeError(p)


def psign(p):
    return int(round(sp.Matrix(3, 3, lambda r, c: 1 if p[c] == r else 0).det()))


ok4 = ok5 = True
for p in itertools.permutations(range(3)):
    sg = psign(p)
    for paired in (False, True):
        U = perm_U(p, paired)
        imgs = [e[p[0] + 1], e[p[1] + 1], e[p[2] + 1]] + ([e[p[0] + 4], e[p[1] + 4], e[p[2] + 4]] if paired else [e[4], e[5], e[6]])
        ok4 &= all(eqm(U @ e[a], imgs[a - 1] @ U) for a in range(1, 7))
        sysm = np.vstack([lin_XA_minus_BX(e[a], imgs[a - 1]) for a in range(1, 7)])
        ok4 &= nullity_p(sysm) == 1                                      # unique up to scale
        UtU = U.T @ U
        ok4 &= eqm(UtU, UtU[0, 0] * I8) and UtU[0, 0] > 0                 # orthogonal up to scale
        c = int(UtU[0, 0])
        Ui = U.T                                                          # U^-1 = U^T / c
        Ad = lambda X: U @ X @ Ui                                         # = c * A(X)
        ok4 &= eqm(Ad(T), c * sg * T)
        for b in range(3):
            nb = [1 if i == b else 0 for i in range(3)]
            snb = [1 if i == p[b] else 0 for i in range(3)]
            ok4 &= eqm(Ad(tv(nb)), c * tv(snb)) and eqm(Ad(J(nb)), c * sg * J(snb))   # linear in n: basis suffices
        ok4 &= eqm(Ad(GAM), c * (GAM if paired else sg * GAM))
        Pm = np.zeros((3, 3), dtype=np.int64)
        for b in range(3):
            Pm[p[b], b] = 1
        P6 = np.block([[Pm, np.zeros((3, 3), dtype=np.int64)], [np.zeros((3, 3), dtype=np.int64), Pm if paired else np.eye(3, dtype=np.int64)]])
        ok4 &= (eqm(P6 @ Gm, Gm @ P6)) == (paired or p == (0, 1, 2))
        if paired:
            ok5 &= all(eqm(Ad(OMI[i]), c * OMI[p[i - 1] + 1]) for i in (1, 2, 3))
            v0 = U @ vac
            ok5 &= np.count_nonzero(v0) == 1 and v0[0] != 0
            lam = int(v0[0])                                              # U vac = lam vac
            Xt = np.block([[sg * Pm, np.zeros((3, 3), dtype=np.int64)], [np.zeros((3, 3), dtype=np.int64), Pm]])
            ok5 &= eqm(U @ PHI, lam * PHI @ Xt)                           # U Phi(X) = lam Phi(Xt)
            Pinv = P6.T
            PHIp = U @ PHI @ Pinv                                         # = lam Phi'
            target = PHI if sg == 1 else PHI @ (-ETA6)
            ok5 &= eqm(PHIp, lam * target)
            if sg == -1:
                parN = OMI[1] @ OMI[2] @ OMI[3]
                ok5 &= eqm(PHI @ (-ETA6), -(parN @ PHI))                  # Phi o(-eta) = -(-1)^N Phi = Phi_t
            ok5 &= eqm(Ad(-T) @ PHIp, c * PHIp @ Gm)                      # A'(-T) Phi' = Phi' G
check(4, "the six temporal permutations, alone and paired: U built from reflections implements e_i -> e_sigma(i) (and "
         "e_{i+3} -> e_{sigma(i)+3}), is unique up to scale (nullity 1 mod p) and orthogonal up to scale; A(T)=sgn T, A(n)=sigma n, "
         "A(J(n))=sgn J(sigma n), A(Gamma)=sgn Gamma alone and Gamma paired; on V the paired ones and only they (with the identity) commute with G", ok4)
check(5, "paired permutations: A'(omega_i)=omega_sigma(i), U|vac> ~ |vac>, U Phi(X) = Phi(Xt), Xt=(sgn P B; P E) (B axial); "
         "the transported octant map is Phi (even) and Phi o(-eta) = -(-1)^N Phi = Phi_t (odd); A'(-T) Phi' = Phi' G", ok5)

# =====================================================================================================================
# s4-5  clock reversal is a half-turn; C = R T; the antilinear intertwiners at clock n are U_n
# =====================================================================================================================
ok = True
for key, (N, U1, U2, r) in CLOCKS.items():
    nM, u1, u2 = tv(N), tv(U1), tv(U2)
    R = nM @ u2                                           # r^2 * rotor
    Rt = u2 @ nM                                          # R Rt = r^4
    ok &= eqm(R @ Rt, r ** 4 * I8)
    cj = lambda X: R @ X @ Rt                             # = r^4 R X R^-1
    ok &= eqm(cj(nM), -r ** 4 * nM) and eqm(cj(T), r ** 4 * T) and eqm(cj(J(N)), -r ** 4 * J(N))
    ok &= eqm(cj(u1), r ** 4 * u1) and eqm(cj(u2), -r ** 4 * u2)          # half-turn about u1
    ok &= eqm(R @ T, r * u1)                                               # (n u2) T = u1
    # antilinear operators intertwining the Dirac operator at clock n: X J = -J X, X gamma^mu = -gamma^mu X
    gams = [nM] + [s[a] for a in (1, 2, 3)]
    sysm = np.vstack([lin_XA_plus_BX(J(N), J(N))] + [lin_XA_plus_BX(gm, gm) for gm in gams])
    ok &= nullity_p(sysm) == 2
    ok &= all(satisfies(C, J(N), J(N), +1) and all(satisfies(C, gm, gm, +1) for gm in gams) for C in (u1, u2))
    ok &= rank_Q([u1, u2]) == 2
    # every unit C of U_n is R_C T with R_C the half-turn about C: C = (3u1+4u2)/5
    Cv = [3 * U1[i] + 4 * U2[i] for i in range(3)]          # = 5 r C
    W = cross(N, Cv)                                         # = 5 r^2 (n x C)
    RC = tv(N) @ tv(W)                                       # = 5 r^3 * (n (n x C))
    ok &= eqm(RC @ T, r * r * tv(Cv)) and eqm(RC @ tv(Cv) @ RC.T, (5 * r ** 3) ** 2 * tv(Cv))
check(6, "clock reversal: R = n u2 satisfies R n R^-1 = -n, R T R^-1 = T, R J(n) R^-1 = -J(n), fixes u1 and reverses u2 (the "
         "half-turn about u1), and R T = u1; at each of four clocks the antilinear operators intertwining J(n)gamma^mu are exactly "
         "U_n (nullity 2 mod p, spanned by u1,u2): charge conjugation at clock n; every unit C in U_n is (half-turn about C) times T", ok)

# =====================================================================================================================
# s4-7  the two senses; odd temporal permutation on the Dirac data; the axial rotation; signs of all couplings
# =====================================================================================================================
def S2(N, X, r):
    """2|N| S_n(X) = sum |N| B_i s_j s_k + E_i s_i N   (integer)"""
    out = Z8.copy()
    for (i, j, k) in cyc:
        out = out + r * X[i - 1] * (s[j] @ s[k]) + X[3 + i - 1] * (s[i] @ tv(N))
    return out


def Mn2(N, X, r):          # 2|N|^2 M_n(X) = 2|N|^2 T J(n) S_n(X) = T J(N) (2|N| S_n)
    return T @ J(N) @ S2(N, X, r)


def Ms2(N, X, r):          # 2|N| M*_n(X) = T Gamma (2|N| S_n)
    return T @ GAM @ S2(N, X, r)


Xb = [[1 if a == b else 0 for a in range(6)] for b in range(6)]
ok = True
for key, (N, U1, U2, r) in CLOCKS.items():
    nM = tv(N)
    Pp, Pm_ = r * I8 + nM, r * I8 - nM               # 2r Pi_+-
    ok &= eqm(r * (T @ Pp), J(N) @ Pp) and eqm(r * (T @ Pm_), -(J(N) @ Pm_))
    OmN = nM @ S3                                    # r Omega_n
    K = [-(nM @ nM)] + [-(nM @ s[a]) for a in (1, 2, 3)]   # r^2 K^mu_n (K^0 = -1)
    ok &= eqm(OmN.T @ T @ OmN, -r * r * T) and all(eqm(OmN.T @ Km @ OmN, r * r * Km) for Km in K) and eqm(OmN @ J(N), J(N) @ OmN)
    for X in Xb:
        M, Ms = Mn2(N, X, r), Ms2(N, X, r)
        ok &= eqm(OmN.T @ M @ OmN, -r * r * M) and eqm(OmN.T @ Ms @ OmN, -r * r * Ms)     # axial rotation reverses both
        C1 = tv(U1)
        ok &= eqm(C1.T @ M @ C1, -r * r * M) and eqm(C1.T @ Ms @ C1, -r * r * Ms)        # C reverses both
        ok &= eqm(C1.T @ T @ C1, r * r * T) and eqm(C1.T @ J(N) @ C1, -r * r * J(N)) and all(eqm(C1.T @ Km @ C1, r * r * Km) for Km in K)
# odd temporal permutation e1 <-> e2 on the Dirac data (K_n, T, J(n)) -> (K_{sigma n}, -T, -J(sigma n)); kernels M, M* fixed
U = perm_U((1, 0, 2), False)
c = int((U.T @ U)[0, 0])
for key, (N, U1, U2, r) in CLOCKS.items():
    sN = [N[1], N[0], N[2]]
    K = [-(tv(N) @ tv(N))] + [-(tv(N) @ s[a]) for a in (1, 2, 3)]
    Ks = [-(tv(sN) @ tv(sN))] + [-(tv(sN) @ s[a]) for a in (1, 2, 3)]
    ok &= all(eqm(U @ K[m] @ U.T, c * Ks[m]) for m in range(4)) and eqm(U @ T @ U.T, -c * T) and eqm(U @ J(N) @ U.T, -c * J(sN))
    for X in Xb:
        ok &= eqm(U @ Mn2(N, X, r) @ U.T, c * Mn2(sN, X, r)) and eqm(U @ Ms2(N, X, r) @ U.T, c * Ms2(sN, X, r))
check(7, "T Pi_+- = +-J(n) Pi_+- (rest flow exp(-m lambda T) = exp(-+m lambda J(n)) on n=+-1); Omega_n reverses T and keeps "
         "every K^mu_n and J(n); an odd temporal permutation maps (K_n, T, J(n)) to (K_{sigma n}, -T, -J(sigma n)) and M_n, M*_n to "
         "M_{sigma n}, M*_{sigma n}; Omega_n and C each reverse M_n and M*_n, C keeps T and K and reverses J: after the permutation and "
         "the axial rotation (m,q,mu,d) -> (m,-q,-mu,-d), the effect of C (four clocks)", ok)

# =====================================================================================================================
# s4-8  three roles of T: mass form, parity (unique; admissible parities at clock n), duality
# =====================================================================================================================
ok = True
sysm = np.vstack([lin_XA_minus_BX(e[a], eta[a] * e[a]) for a in range(1, 7)])   # Y e_a = eta_aa e_a Y
ok &= nullity_p(sysm) == 1 and all(eqm(T @ e[a], eta[a] * e[a] @ T) for a in range(1, 7))
ok &= all(eqm(T @ e[c] @ T.T, (ETA6 @ np.eye(6, dtype=np.int64))[c - 1, c - 1] * e[c]) for c in range(1, 7))   # Ad_T = eta
p1, p2, p3, mm = sp.symbols('p1 p2 p3 m')
for key, (N, U1, U2, r) in CLOCKS.items():
    nM = tv(N)
    ok &= eqm(nM @ J(N), r * r * T) and eqm(T @ J(N), -nM)                       # gamma^0 J = T, T J(n) = -n
    ok &= all(eqm((T @ J(N) @ g_).T, T @ J(N) @ g_) for g_ in [nM] + [s[a] for a in (1, 2, 3)])   # K^mu symmetric
    ok &= eqm(T @ J(N), J(N) @ T)
    ok &= all(eqm(T @ nM @ s[a] @ T.T, -(nM @ s[a])) for a in (1, 2, 3)) and eqm(T @ nM @ T.T, nM)   # T H(p) T^-1 = H(-p)
    # admissible parities at clock n: linear (commute with J(n)), X n = n X, X (n s_a) = -(n s_a) X
    sysm = np.vstack([lin_XA_minus_BX(J(N), J(N)), lin_XA_minus_BX(nM, nM)] + [lin_XA_plus_BX(nM @ s[a], nM @ s[a]) for a in (1, 2, 3)])
    ok &= nullity_p(sysm) == 2 and rank_Q([nM, T]) == 2
ok &= eqm(-(T @ PHI), PHI @ Gm) and all(eqm(T @ f[k], g[k]) and eqm(T @ g[k], -f[k]) for k in (1, 2, 3)) and eqm(T @ vac, full)
check(8, "three roles: gamma^0 J(n) = T and K^mu_n = T J(n) gamma^mu_n = -n gamma^mu_n symmetric (four clocks); Ad_T|V = eta, "
         "implemented by T alone (nullity 1); T commutes with J(n) and T H_n(p) T^-1 = H_n(-p); the admissible parities at each clock "
         "form span{n, T} (nullity 2), P^2=+1 for n and -1 for T; -T Phi = Phi G, T f_k = g_k, T g_k = -f_k, T|vac> = |full>", ok)

# =====================================================================================================================
# s4-9  J(n) relative to the field structure
# =====================================================================================================================
ok = True
A2 = [None] * 4
for (i, j, k) in cyc:
    A2[i] = e[j] @ e[k] + e[j + 3] @ e[k + 3]
for key, (N, U1, U2, r) in CLOCKS.items():
    Ln2 = sum((N[i - 1] * L2[i] for i in (1, 2, 3)), Z8)        # 2 L(N)
    An2 = sum((N[i - 1] * A2[i] for i in (1, 2, 3)), Z8)
    ok &= eqm(2 * J(N), -Ln2 + An2) and iszm(Ln2 @ An2) and iszm(An2 @ Ln2)
    PL, PA = -(Ln2 @ Ln2), -(An2 @ An2)                         # 4 r^2 * projectors
    ok &= eqm(PL @ PL, 4 * r * r * PL) and eqm(PA @ PA, 4 * r * r * PA) and eqm(PL + PA, 4 * r * r * I8)
    ok &= rank_Q([PL[:, b] for b in range(8)]) == 4 and rank_Q([PA[:, b] for b in range(8)]) == 4
    ok &= eqm(Ln2 @ Ln2 @ Ln2, -4 * r * r * Ln2) and eqm(An2 @ An2 @ An2, -4 * r * r * An2)
    rhoLN2 = sum((N[i - 1] * rhoL2(i) for i in (1, 2, 3)), Z8)  # 2 rho_n(L(N))
    ok &= eqm(Ln2, rhoLN2 - J(N)) and eqm(rhoLN2 @ rhoLN2, -r * r * I8) and eqm(rhoLN2 @ J(N), J(N) @ rhoLN2)
    # rho_n(L(n)) acts on V as the rotation of V^- about the spatial axis sum n_i s_i and kills V^+
    Ar = admat(rhoLN2)                 # 2 * action
    ok &= iszm(Ar[:, :3]) and iszm(Ar[:3, :]) and all(eqm(Ar[3:, 3 + b], 2 * np.array(cross(N, [1 if c_ == b else 0 for c_ in range(3)]))) for b in range(3))
    # exp(2 pi L(n)) = 1: L^3 = -L gives exp(theta L) = 1 + sin(theta) L + (1 - cos(theta)) L^2; at 2 pi: 1.
    # exp(2 pi rho_n(L(n))) = cos(pi) + sin(pi) 2 rho = -1 and exp(-pi J(n)) = -1 since both square to -1 (unit n)
    # kernel of L(n): poles and longitudinal octants
    for vv in (vac, full, fv(N), gv(N)):
        ok &= iszm(Ln2 @ vv)
    ok &= rank_Q([Ln2[:, b] for b in range(8)]) == 4
for i in (1, 2, 3):
    ok &= eqm(L2[i], rhoL2(i) - J([1 if c_ == i - 1 else 0 for c_ in range(3)]))      # rho_oct(L_i) = rho_n(L_i) - J(e_i)/2
# J(n) Phi(X) = -Phi(n x B; n x E) + (E.n)|full> - (B.n)|vac>, bilinear: check on bases
for b in range(3):
    nb = [1 if c_ == b else 0 for c_ in range(3)]
    for X in Xb:
        B_, E_ = X[:3], X[3:]
        rhs = -Phi(cross(nb, B_) + cross(nb, E_)) + dot(E_, nb) * full - dot(B_, nb) * vac
        ok &= eqm(J(nb) @ Phi(X), rhs)
    ok &= eqm(J(nb) @ vac, gv(nb)) and eqm(J(nb) @ full, -fv(nb))
    for c_ in range(3):
        uc = [1 if d == c_ else 0 for d in range(3)]
        Ln2b = sum((nb[i - 1] * L2[i] for i in (1, 2, 3)), Z8)
        An2b = sum((nb[i - 1] * A2[i] for i in (1, 2, 3)), Z8)
        ok &= eqm(Ln2b @ fv(uc), 2 * fv(cross(nb, uc))) and eqm(Ln2b @ gv(uc), 2 * gv(cross(nb, uc)))
        ok &= eqm(An2b @ fv(uc), 2 * dot(nb, uc) * full) and eqm(An2b @ gv(uc), -2 * dot(nb, uc) * vac)
# J(n) does not commute with rho_oct and does not preserve Phi(V)
ok &= not eqm(J([1, 0, 0]) @ Q2[2], Q2[2] @ J([1, 0, 0])) and rank_Q([PHI[:, b] for b in range(6)] + [J([1, 0, 0]) @ f[1]]) == 7
check(9, "J(n) = -L(n) + A(n), L(n)A(n) = A(n)L(n) = 0, -L^2, -A^2 complementary rank-4 projectors, L^3=-L, A^3=-A (four clocks); "
         "L(n) = rho_n(L(n)) - J(n)/2 with rho_n(L(n))^2 = -1/4, and rho_n(L(n)) rotates V^- about the spatial axis sum n_i s_i and "
         "kills V^+ (so it is a rotation about the axis paired with n, not about n); rho_oct(L_i) = rho_n(L_i) - J(e_i)/2; "
         "J(n)Phi(X) = -Phi(n x B; n x E) + (E.n)|full> - (B.n)|vac>, J(n)|vac> = g(n), J(n)|full> = -f(n); ker L(n) = poles + f(n), g(n); "
         "L(n)f(u)=f(n x u), L(n)g(u)=g(n x u), A(n)f(u)=(n.u)|full>, A(n)g(u)=-(n.u)|vac>; J(n) leaves Phi(V) and does not commute with rho_oct", ok)

# =====================================================================================================================
# s4-10  commutants: 8, 6, 1
# =====================================================================================================================
def gens_rho_n(N, r):
    """integer multiples of the generators of rho_n: 2 rho_n(L_i), 2|N| rho_n(K_i)"""
    return [rhoL2(i) for i in (1, 2, 3)] + [rhoK2N(i, N) for i in (1, 2, 3)]


GENS_OCT = [L2[i] for i in (1, 2, 3)] + [Q2[i] for i in (1, 2, 3)]
PI_F = np.diag([0, 1, 1, 1, 1, 1, 1, 0]).astype(np.int64)
ok = True
for key in ('e1', '236', '-148'):
    N, U1, U2, r = CLOCKS[key]
    Gn = gens_rho_n(N, r)
    sys_n = np.vstack([lin_XA_minus_BX(Y, Y) for Y in Gn])
    sys_o = np.vstack([lin_XA_minus_BX(Y, Y) for Y in GENS_OCT])
    ok &= nullity_p(sys_n) == 8 and nullity_p(sys_o) == 6 and nullity_p(np.vstack([sys_n, sys_o])) == 1
    OmN = tv(N) @ S3
    base = [I8, tv(U1), tv(U2), J(N)]
    cand = base + [b @ OmN for b in base]
    ok &= all(all(eqm(C @ Y, Y @ C) for Y in Gn) for C in cand) and rank_Q(cand) == 8
    # the algebra span{1,u1,u2,J} = M_2(R): u1u2 = r J(n), u_a^2 = r^2
    ok &= eqm(tv(U1) @ tv(U2), r * J(N)) and eqm(OmN @ OmN, -r * r * I8)
pole = lambda a, b: np.outer(a, b)
cand_o = [PI_F, T @ PI_F, pole(vac, vac), pole(vac, full), pole(full, vac), pole(full, full)]
ok &= all(all(eqm(C @ Y, Y @ C) for Y in GENS_OCT) for C in cand_o) and rank_Q(cand_o) == 6
check(10, "commutants (nullity mod p, exact bases exhibited): rho_n(l_G) has dimension 8, spanned by {1,u1,u2,J(n)} x {1,Omega_n} "
          "(u1u2 = J(n), = M_2(R) x C = M_2(C)); rho_oct(l_G) has dimension 6, span{Pi_F, T Pi_F} + End(poles); joint commutant R 1 "
          "(clocks e1, (2,3,6)/7, (-1,4,8)/9)", ok)

# =====================================================================================================================
# s4-11  three inequivalent representations on S
# =====================================================================================================================
ok = True
N, U1, U2, r = CLOCKS['236']
Gn = gens_rho_n(N, r)
# Casimirs (generators scaled: rho(L) = L2/2, rho_n(K) = s_i N/(2r))
cas_n = sum((Gn[3 + i] @ Gn[3 + i] for i in range(3)), Z8) * 1 - r * r * sum((Gn[i] @ Gn[i] for i in range(3)), Z8)   # 4 r^2 C1
ok &= eqm(cas_n, 4 * r * r * 3 * I8 // 2)                                              # C1 = 3/2
KL = sum((Gn[3 + i] @ Gn[i] for i in range(3)), Z8)                                       # 4 r sum K L
ok &= eqm(KL @ KL, -9 * r * r * I8)                                                       # (sum K L)^2 = -9/16
cas_o = sum((Q2[i] @ Q2[i] - L2[i] @ L2[i] for i in (1, 2, 3)), Z8)                       # 4 C1
KLo = sum((Q2[i] @ L2[i] for i in (1, 2, 3)), Z8)                                         # 4 sum K L
ok &= eqm(cas_o, 16 * PI_F) and eqm(KLo @ KLo, -64 * PI_F)                               # 4 and -4 on field octants, 0 on poles
cas_g = sum((K2[i] @ K2[i] - L2[i] @ L2[i] for i in (1, 2, 3)), Z8)
KLg = sum((K2[i] @ L2[i] for i in (1, 2, 3)), Z8)
ok &= eqm(cas_g, 12 * I8) and iszm(KLg @ KLg)                                              # 3 and 0
ok &= eqm(rhoL2(3) @ rhoL2(3), -I8) and eqm(L2[3] @ L2[3] @ L2[3], -4 * L2[3])             # rho_n(L3)^2=-1/4; L^3=-L
# no nonzero intertwiner between any two of them (the Lie brackets are those of l_G for each)
GENS_G = [L2[i] for i in (1, 2, 3)] + [K2[i] for i in (1, 2, 3)]
for A_, B_ in ((Gn, GENS_OCT), (Gn, GENS_G), (GENS_OCT, GENS_G)):
    # U A(Y) = B(Y) U ; K-generators of rho_n carry the extra factor r: scale B accordingly
    rows = []
    for idx in range(6):
        a, b = A_[idx], B_[idx]
        if A_ is Gn and idx >= 3:
            b = r * b
        rows.append(lin_XA_minus_BX(a, b))
    ok &= nullity_p(np.vstack(rows)) == 0
# the brackets: rho_n, rho_oct and the spin lift of l_G are representations of l_G (check [K1,K2] = -L3 etc.)
def br_ok(Lg, Kg, kscale):
    o = True
    for (i, j, k) in cyc:
        o &= eqm(Lg[i - 1] @ Lg[j - 1] - Lg[j - 1] @ Lg[i - 1], 2 * Lg[k - 1])
        o &= eqm(Lg[i - 1] @ Kg[j - 1] - Kg[j - 1] @ Lg[i - 1], 2 * Kg[k - 1])
        o &= eqm(Kg[i - 1] @ Kg[j - 1] - Kg[j - 1] @ Kg[i - 1], -2 * kscale * kscale * Lg[k - 1])
    return o
ok &= br_ok(Gn[:3], Gn[3:], r) and br_ok(GENS_OCT[:3], GENS_OCT[3:], 1) and br_ok(GENS_G[:3], GENS_G[3:], 1)
check(11, "rho_n, rho_oct and the spin lift of l_G satisfy the l_G brackets; Casimirs sum K^2-L^2, (sum K.L)^2: 3/2, -9/16 (rho_n); "
          "4, -4 on the field octants and 0 on the poles (rho_oct); 3, 0 (spin lift of l_G); rho_n(L3)^2=-1/4, L3^3=-L3 in the "
          "others; the intertwiner spaces between each pair are 0 (nullity 0 mod p), so no U, invertible or not, relates them", ok)

# =====================================================================================================================
# s4-12  no invariant bilinear coupling psi^T M Phi(X)
# =====================================================================================================================
ok = True
for key in ('e1', '236'):
    N, U1, U2, r = CLOCKS[key]
    Gn = gens_rho_n(N, r)
    rows_all, rows_F = [], []
    for idx in range(6):
        a = Gn[idx]
        b = GENS_OCT[idx] * (r if idx >= 3 else 1)
        # rho_n(Y)^T M + M rho_oct(Y) = 0  (M: 8x8 on all of S)
        rows_all.append(kron_r(a.T, I8) + kron_r(I8, b.T))
        # M: 8x6 with Phi(X) = PHI X carrying rho_oct, equivalently X carrying l_G: rho_n^T M' + M' A(Y) = 0
        Yg = (L2 if idx < 3 else K2)[idx % 3 + 1]
        Ay = admat(Yg) * (r if idx >= 3 else 1)
        rows_F.append(kron_r(a.T, np.eye(6, dtype=np.int64)) + kron_r(I8, Ay.T))
    ok &= nullity_p(np.vstack(rows_all)) == 0 and nullity_p(np.vstack(rows_F)) == 0
# consistency: Phi intertwines l_G on V with rho_oct
ok &= all(eqm(GENS_OCT[idx] @ PHI, PHI @ admat((L2 if idx < 3 else K2)[idx % 3 + 1])) for idx in range(6))
check(12, "no nonzero invariant bilinear psi^T M Phi(X) with psi under rho_n and Phi(X) under rho_oct (on Phi(V), equivalently X "
          "under l_G, and on all of S), at n = e1 and (2,3,6)/7 (nullity 0 mod p); Phi intertwines l_G on V with rho_oct", ok)

# =====================================================================================================================
# s4-13  trilinear couplings: 8 = 2 (antisymmetric) + 6 (symmetric), exact
# =====================================================================================================================
def coupling_system(N, r):
    Gn = gens_rho_n(N, r)
    blocks = []
    for idx in range(6):
        a = Gn[idx]
        Yg = (L2 if idx < 3 else K2)[idx % 3 + 1]
        Ay = admat(Yg) * (r if idx >= 3 else 1)          # same scale as a
        for cc in range(6):
            blk = np.zeros((64, 384), dtype=np.int64)
            blk[:, cc * 64:(cc + 1) * 64] += kron_r(a.T, I8) + kron_r(I8, a.T)
            for d in range(6):
                if Ay[d, cc]:
                    blk[:, d * 64:(d + 1) * 64] += Ay[d, cc] * np.eye(64, dtype=np.int64)
            blocks.append(blk)
    return np.vstack(blocks)


def kernel_from_C(N, r, C):
    """the map X -> n C S_n(X) as 6 matrices (scaled by 2|N|^2: N C (2|N| S_n))"""
    return [tv(N) @ C @ S2(N, Xb[cc], r) for cc in range(6)]


def is_solution(N, r, Ms):
    Gn = gens_rho_n(N, r)
    o = True
    for idx in range(6):
        a = Gn[idx]
        Yg = (L2 if idx < 3 else K2)[idx % 3 + 1]
        Ay = admat(Yg) * (r if idx >= 3 else 1)
        for cc in range(6):
            lhs = a.T @ Ms[cc] + Ms[cc] @ a + sum((Ay[d, cc] * Ms[d] for d in range(6)), Z8)
            o &= iszm(lhs)
    return o


ok = True
for key in ('e1', '236', '-148'):
    N, U1, U2, r = CLOCKS[key]
    A_sys = coupling_system(N, r)
    ok &= A_sys.shape == (2304, 384) and nullity_p(A_sys) == 8
    OmN = tv(N) @ S3
    base = [I8, tv(U1), tv(U2), J(N)]
    Cs = base + [b @ OmN for b in base]
    sols = [kernel_from_C(N, r, C) for C in Cs]
    ok &= all(is_solution(N, r, Ms) for Ms in sols) and rank_Q([np.concatenate([m.ravel() for m in Ms]) for Ms in sols]) == 8
    sym = [Ms for Ms in sols if all(eqm(m.T, m) for m in Ms)]
    anti = [Ms for Ms in sols if all(eqm(m.T, -m) for m in Ms)]
    ok &= len(sym) == 6 and len(anti) == 2 and len(sym) + len(anti) == 8
    ok &= all(eqm(m.T, -m) for C in (I8, OmN) for m in kernel_from_C(N, r, C))          # C = 1, Omega_n antisymmetric
# representation content: invariant bilinear forms on S under rho_n: 2 symmetric (1, Omega in n.Comm) and 6 antisymmetric
N, U1, U2, r = CLOCKS['236']
Gn = gens_rho_n(N, r)
ok &= nullity_p(np.vstack([kron_r(a.T, I8) + kron_r(I8, a.T) for a in Gn])) == 8
OmN = tv(N) @ S3
forms = [tv(N) @ C for C in [I8, tv(U1), tv(U2), J(N)] + [b @ OmN for b in [I8, tv(U1), tv(U2), J(N)]]]
ok &= sum(eqm(F.T, F) for F in forms) == 2 and sum(eqm(F.T, -F) for F in forms) == 6
check(13, "invariant trilinear kernels M: V -> End(S), rho_n(Y)^T M(X) + M(X) rho_n(Y) + M(Y.X) = 0: the 2304 x 384 system has "
          "nullity 8 mod p and the eight n C S_n(X), C in {1,u1,u2,J(n)} x {1,Omega_n}, are independent exact solutions, so the "
          "dimension is 8 over Q; 2 antisymmetric (C = 1, Omega_n: Grassmann) and 6 symmetric (commuting psi), at e1, (2,3,6)/7, (-1,4,8)/9; "
          "invariant forms on S: 2 symmetric, 6 antisymmetric (Sym^2 S has 2, Lambda^2 S has 6 singlets)", ok)

# =====================================================================================================================
# s4-14  the explicit kernels, their symmetries, parity; l equivariant; the bivector of F_X
# =====================================================================================================================
ok = True
# l(X) = sum B_i L_i + E_i K_i is equivariant: [Y, l(X)] = l(Y.X)
def ell2(X):
    return sum((X[i - 1] * L2[i] + X[3 + i - 1] * K2[i] for i in (1, 2, 3)), Z8)
for Yg in GENS_G:
    Ay = admat(Yg)                                    # 2 * action
    for X in Xb:
        lhs = Yg @ ell2(X) - ell2(X) @ Yg             # 4 [Y, l(X)]
        ok &= eqm(lhs, ell2(list(Ay @ np.array(X))))
for key, (N, U1, U2, r) in CLOCKS.items():
    nM = tv(N)
    for idx, a in enumerate(gens_rho_n(N, r)):
        ok &= eqm(r * r * a.T, -(nM @ a @ nM))                         # rho_n(Y)^T = -n rho_n(Y) n
    for X in Xb:
        B_, E_ = X[:3], X[3:]
        M2 = Mn2(N, X, r)                                              # 2 r^2 M_n
        expl = r * r * sv(E_) - r * nM @ sum((B_[i - 1] * (s[j] @ s[k]) for (i, j, k) in cyc), Z8)   # r^2 (E.s - n B ss) ... times 1
        ok &= eqm(M2, expl) and eqm(M2, -(nM @ S2(N, X, r)))        # = -n S_n
        Ms = Ms2(N, X, r)                                              # 2 r M*_n
        expls = r * sv(B_) + nM @ sum((E_[i - 1] * (s[j] @ s[k]) for (i, j, k) in cyc), Z8)
        ok &= eqm(Ms, expls) and eqm(r * Ms, Mn2(N, list(Gm @ np.array(X)), r))           # M*(X) = M(GX)
        ok &= eqm(r * r * Ms, T @ J(N) @ nM @ S3 @ S2(N, X, r))                            # T Gamma = T J(n) Omega_n
        ok &= eqm(M2.T, -M2) and eqm(Ms.T, -Ms)
        ok &= eqm(M2 @ J(N), J(N) @ M2) and eqm(Ms @ J(N), J(N) @ Ms)
        Xe = list(ETA6 @ np.array(X))
        for Pp in (nM, T):
            ok &= eqm(Pp.T @ Mn2(N, Xe, r) @ Pp, (r * r if Pp is nM else 1) * M2)
            ok &= eqm(Pp.T @ Ms2(N, Xe, r) @ Pp, -(r * r if Pp is nM else 1) * Ms)
        ok &= eqm(nM.T @ T @ nM, r * r * T) and eqm(T.T @ T @ T, T)
        # bivector of F_X: (F_X)_{0i} = E_i, (F_X)_{jk} = eps_ijk B_i; gamma^mu = (n, s_a), gamma_mu = eta_{mu nu} gamma^nu
        F = sp.zeros(4, 4)
        for (i, j, k) in cyc:
            F[0, i] = E_[i - 1]; F[i, 0] = -E_[i - 1]
            F[j, k] = B_[i - 1]; F[k, j] = -B_[i - 1]
        etaE = sp.diag(1, -1, -1, -1)
        Fup = etaE * F * etaE
        gup = [nM] + [r * s[a] for a in (1, 2, 3)]                      # r gamma^mu
        gdn = [nM] + [-r * s[a] for a in (1, 2, 3)]                     # r gamma_mu
        wedge = lambda a, b: a @ b                                      # 2 a^b for orthogonal a, b
        F_low_low = sum((int(F[mu, nu]) * wedge(gdn[mu], gdn[nu]) for mu in range(4) for nu in range(mu + 1, 4)), Z8)
        F_up_low = sum((int(Fup[mu, nu]) * wedge(gdn[mu], gdn[nu]) for mu in range(4) for nu in range(mu + 1, 4)), Z8)
        ok &= eqm(F_low_low, r * S2(N, X, r))                          # sum_{mu<nu} F_{mu nu} gamma_mu ^ gamma_nu = S_n(X)
        ok &= eqm(F_up_low, r * S2(N, Xe, r))                          # sum F^{mu nu} gamma_mu ^ gamma_nu = S_n(eta X)
check(14, "l(X) = sum(B_i L_i + E_i K_i) is l_G-equivariant; rho_n(Y)^T = -n rho_n(Y) n; M_n = T J(n) S_n = -n S_n = "
          "E.s/2 - n(sum B_i s_js_k)/2 and M*_n = T Gamma S_n = T J(n) Omega_n S_n = B.s/2 + n(sum E_i s_js_k)/2 = M_n(GX), both "
          "antisymmetric and commuting with J(n); under psi -> n psi or T psi with X -> eta X, M_n and T are even, M*_n is odd; "
          "S_n(X) is the Clifford image of sum_{mu<nu} F_{mu nu} gamma_mu ^ gamma_nu (gamma_mu = eta_{mu nu} gamma^nu_n), while "
          "sum F^{mu nu} gamma_mu ^ gamma_nu is S_n(eta X), the electric part reversed (four clocks)", ok)

# =====================================================================================================================
# s4-15  the couplings when the clock moves or the axes are permuted
# =====================================================================================================================
ok = True
for key, (N, U1, U2, r) in CLOCKS.items():
    if key == 'e1':
        continue
    q = r * I8 + tv(N) @ e[1]
    qt = r * I8 + e[1] @ tv(N)
    den = 2 * r * (r + N[0])
    for X in Xb:
        # M_n = q M_e1 q^-1 : 2 r^2 M_n vs 2 M_e1
        ok &= eqm(r * r * (q @ Mn2([1, 0, 0], X, 1) @ qt), den * Mn2(N, X, r))
        ok &= eqm(r * (q @ Ms2([1, 0, 0], X, 1) @ qt), den * Ms2(N, X, r))
    ok &= eqm(q @ T @ qt, den * T) and eqm(q @ T @ GAM @ qt, den * T @ GAM)
ok &= not eqm(Mn2([1, 0, 0], [1, 0, 0, 0, 0, 0], 1), Mn2([0, 1, 0], [1, 0, 0, 0, 0, 0], 1))
# diagonal rotation exp(theta L_3), cos = 3/5: M_{Rn}(RX) = R M_n(X) R^-1
rot5 = lambda v: [3 * v[0] - 4 * v[1], 4 * v[0] + 3 * v[1], 5 * v[2]]
for key, (N, U1, U2, r) in CLOCKS.items():
    RN = rot5(N)                                       # = 5 R N, |RN| = 5 r
    for X in Xb:
        RX = rot5(X[:3]) + rot5(X[3:])                 # 5 R X
        lhs = R20 @ Mn2(N, X, r) @ Rinv20              # 400 * 2 r^2 R M R^-1
        rhs = Mn2(RN, RX, 5 * r)                       # 2 (5r)^2 M_{Rn}(5 RX) = 2 r^2 * 125 M_{Rn}(RX)
        ok &= eqm(125 * lhs, 400 * rhs)
        lhs = R20 @ Ms2(N, X, r) @ Rinv20
        rhs = Ms2(RN, RX, 5 * r)
        ok &= eqm(25 * lhs, 400 * rhs)
# permutations (alone: M, M* fixed; paired: M -> M(Xt), M* -> sgn M*(Xt))
for p in itertools.permutations(range(3)):
    sg = psign(p)
    Pm = np.zeros((3, 3), dtype=np.int64)
    for b in range(3):
        Pm[p[b], b] = 1
    for paired in (False, True):
        U = perm_U(p, paired)
        c = int((U.T @ U)[0, 0])
        for key, (N, U1, U2, r) in CLOCKS.items():
            sN = list(Pm @ np.array(N))
            for X in Xb:
                Xt = list(np.concatenate([sg * Pm @ np.array(X[:3]), Pm @ np.array(X[3:])])) if paired else X
                ok &= eqm(U @ Mn2(N, X, r) @ U.T, c * Mn2(sN, Xt, r))
                ok &= eqm(U @ Ms2(N, X, r) @ U.T, c * (sg if paired else 1) * Ms2(sN, Xt, r))
            ok &= eqm(U @ T @ U.T, c * sg * T)
check(15, "M_n(X) = q M_e1(X) q^-1, M*_n(X) = q M*_e1(X) q^-1, q T q^-1 = T, q T Gamma q^-1 = T Gamma (three clocks); M_e1 != M_e2; "
          "M_{Rn}(RX) = R M_n(X) R^-1 and likewise for M* (diagonal rotation, cos = 3/5); all six permutations at four clocks: "
          "alone A(M_n(X)) = M_{sigma n}(X), A(M*_n(X)) = M*_{sigma n}(X) while A(T) = sgn T; paired A'(M_n(X)) = M_{sigma n}(Xt), "
          "A'(M*_n(X)) = sgn M*_{sigma n}(Xt): only the dual term under the paired odd permutation changes sign with T", ok)

# =====================================================================================================================
# s4-16  two Pauli triples; one-sided relations excluded for every clock
# =====================================================================================================================
ok = True
k1, k2, k3 = sp.symbols('k1 k2 k3')
for key, (N, U1, U2, r) in CLOCKS.items():
    al = [tv(N) @ s[a] for a in (1, 2, 3)]                # r alpha_a
    ok &= all(eqm(x.T, x) and eqm(x @ x, r * r * I8) for x in al)
    ok &= all(iszm(al[a] @ al[b] + al[b] @ al[a]) for a in range(3) for b in range(3) if a < b)
    ok &= all(eqm(J(N) @ al[a], r * r * T @ s[a + 1]) and eqm(al[a], -(T @ J(N) @ s[a + 1])) for a in range(3))
ok &= all(eqm(e[a], -(T @ J([1 if c_ == a - 1 else 0 for c_ in range(3)]))) for a in (1, 2, 3))
ok &= all(eqm(x.T, x) and eqm(x @ x, I8) for x in e[1:4]) and all(iszm(e[a] @ e[b] + e[b] @ e[a]) for a in (1, 2, 3) for b in (1, 2, 3) if a < b)
# (sum k pi)^2 = |k|^2 for both triples, symbolic k, abstract algebra (alpha with symbolic unit-free n: |n|^2 |k|^2)
alA = [nA * E[a] for a in (4, 5, 6)]
kk = [k1, k2, k3]
Ksum = alA[0] * k1 + alA[1] * k2 + alA[2] * k3
ok &= mveq(Ksum * Ksum, nn2 * (k1 ** 2 + k2 ** 2 + k3 ** 2) * One)
Ksum = E[1] * k1 + E[2] * k2 + E[3] * k3
ok &= mveq(Ksum * Ksum, (k1 ** 2 + k2 ** 2 + k3 ** 2) * One)
# no Z with alpha_a(n) = Z e_a for all a: Z = alpha_a e_a would need n s1 e1 = n s2 e2, i.e. s1e1 = s2e2 (n invertible)
ok &= not mveq(E[4] * E[1], E[5] * E[2])
# no Z with alpha_a(n) = e_a Z: e1 n s1 = e2 n s2 is linear in n; the map n -> e1 n s1 - e2 n s2 is injective
diffA = E[1] * nA * E[4] - E[2] * nA * E[5]
coeffs = sp.Matrix([[sp.diff(diffA.d.get(kb, 0), x) for x in (n1, n2, n3)] for kb in sorted(diffA.d)])
ok &= coeffs.rank() == 3
check(16, "alpha_a(n) = n s_a and e_a are Pauli triples (symmetric, square 1, anticommuting), (sum k_a pi_a)^2 = |k|^2 "
          "(symbolic k and n); J(n) alpha_a = T s_a, alpha_a = -T J(n) s_a, e_a = -T J(e_a); for every n there is no Z with "
          "alpha_a(n) = Z e_a (it would need e4e1 = e5e2) nor with alpha_a(n) = e_a Z (n -> e1ns1 - e2ns2 is injective)", ok)

# =====================================================================================================================
# s4-17  intertwiners U alpha_a(n) = e_a U
# =====================================================================================================================
ok = True
even_bl = [blade(I) for rr in range(0, 7, 2) for I in itertools.combinations(range(1, 7), rr)]
odd_bl = [blade(I) for rr in range(1, 7, 2) for I in itertools.combinations(range(1, 7), rr)]
EVEN_B = np.array([b.ravel() for b in even_bl]).T            # 64 x 32
ODD_B = np.array([b.ravel() for b in odd_bl]).T
for key in ('e1', '236', '-148'):
    N, U1, U2, r = CLOCKS[key]
    al = [tv(N) @ s[a] for a in (1, 2, 3)]                    # r alpha_a
    sysm = np.vstack([lin_XA_minus_BX(al[a], r * e[a + 1]) for a in range(3)])
    ok &= nullity_p(sysm) == 8
    ok &= nullity_p(sysm @ EVEN_B) == 0 and nullity_p(sysm @ ODD_B) == 0
    # an exact invertible solution: sympy nullspace of the 192 x 64 system (small)
    ns = sp.Matrix(sysm.tolist()).nullspace()
    ok &= len(ns) == 8
    Uc = sum((sp.Integer(i + 1) * ns[i] for i in range(8)), sp.zeros(64, 1))
    Uc = Uc * sp.ilcm(*[sp.fraction(x)[1] for x in Uc])
    Um = sp.Matrix(8, 8, list(Uc))
    ok &= Um.det() != 0
    Uinv = Um.inv()
    toS = lambda A: sp.Matrix(A.tolist())
    for (i, j, k) in cyc:
        ok &= Um * toS(rhoL2(i)) * Uinv == -toS(e[j] @ e[k])                      # U rho_n(L_i) U^-1 = -e_j e_k / 2
        ok &= Um * toS(rhoK2N(i, N)) * Uinv == -r * toS(e[i])                     # U rho_n(K_i) U^-1 = -e_i / 2
        ok &= eqm(r * r * rhoL2(i), -(al[j - 1] @ al[k - 1]))
        ok &= eqm(rhoK2N(i, N), -al[i - 1])
# commutant of e1,e2,e3: span{1,T} x Cl^0(V^-), dimension 8
sysm = np.vstack([lin_XA_minus_BX(e[a], e[a]) for a in (1, 2, 3)])
cand = [I8, s[1] @ s[2], s[2] @ s[3], s[3] @ s[1]]
cand = cand + [T @ x for x in cand]
ok &= nullity_p(sysm) == 8 and all(all(eqm(x @ e[a], e[a] @ x) for a in (1, 2, 3)) for x in cand) and rank_Q(cand) == 8
ok &= all(eqm(Q2[i], -(e[i] + s[i] @ GAM)) and eqm(L2[i], -(e[j] @ e[k]) + s[j] @ s[k]) for (i, j, k) in cyc)
check(17, "the U with U alpha_a(n) = e_a U form an 8-dimensional space (nullity 8 mod p; 8 exact solutions), none of whose nonzero "
          "elements is even or odd (nullity 0 on each parity); an exact invertible solution carries rho_n(L_i) to -e_je_k/2 and "
          "rho_n(K_i) to -e_i/2 (n = e1, (2,3,6)/7, (-1,4,8)/9); rho_n(L_i) = -alpha_j alpha_k/2, rho_n(K_i) = -alpha_i/2; the commutant "
          "of e_1,e_2,e_3 is span{1,T} x Cl^0(V^-); rho_oct(K_i) = -(e_i + e_{i+3} omega)/2, rho_oct(L_i) = -e_je_k/2 + s_js_k/2", ok)

# =====================================================================================================================
# s4-18  the Fock picture: temporal generators left, spatial ones twisted right
# =====================================================================================================================
ok = True
for rr in range(4):
    for I in itertools.combinations((1, 2, 3), rr):
        x = blade(I)
        for a in (1, 2, 3):
            ok &= eqm(e[a] @ (x @ vac), (e[a] @ x) @ vac)
            ok &= eqm(s[a] @ (x @ vac), (-1) ** rr * (x @ e[a]) @ vac)
            ok &= eqm(e[1] @ s[a] @ (x @ vac), (-1) ** rr * (e[1] @ x @ e[a]) @ vac)
        ok &= eqm(GAM @ vac, -vac)
check(18, "on the eight blades x of Cl(V^+): e_a(x|vac>) = (e_a x)|vac>, s_a(x|vac>) = ((-1)^{|x|} x e_a)|vac>, hence "
          "alpha_a(e1)(x|vac>) = (e1 x^ e_a)|vac>; Gamma|vac> = -|vac>", ok)

# =====================================================================================================================
# s4-19  uniqueness of T: three clocks; the two-clock exception; the joint-commutant version
# =====================================================================================================================
ok = True


def gam_list(N):
    return [tv(N)] + [s[a] for a in (1, 2, 3)]


def sys_clock(Ns, ops):
    rows = []
    for N in Ns:
        nM = tv(N)
        for gm in ops(N):
            # (n Y) gm - gm (n Y) = 0  <=>  vec: (n x I) ... use X = nY: Y -> nY linear: kron(nM, I8)
            rows.append(lin_XA_minus_BX(gm, gm) @ kron_r(nM, I8))
    return np.vstack(rows)


three = [[1, 0, 0], [1, 2, 2], [2, 3, 6]]
ok &= sp.Matrix(three).det() != 0
ok &= nullity_p(sys_clock(three, gam_list)) == 1 and all(eqm(tv(N) @ T @ g_, g_ @ tv(N) @ T) for N in three for g_ in gam_list(N))
for two in ([[1, 0, 0], [0, 3, 4]], [[1, 2, 2], [2, 3, 6]]):
    w = cross(two[0], two[1])
    Y2 = tv(w) @ S3
    ok &= nullity_p(sys_clock(two, gam_list)) == 2
    ok &= all(eqm(tv(N) @ Y2 @ g_, g_ @ tv(N) @ Y2) for N in two for g_ in gam_list(N)) and rank_Q([T, Y2]) == 2
# two clocks e1, (0,3,4)/5: n x n' = (0,-4,3)/5 is a unit vector; Y = (3T + 4 w e4e5e6)/5 gives a square root of -1 at both
two = [[1, 0, 0], [0, 3, 4]]
w = cross(two[0], two[1])                       # = 5 (n x n'), |w| = 5
Y5 = 3 * 5 * T + 4 * tv(w) @ S3                 # = 25 Y
for N, rN in zip(two, (1, 5)):
    x = tv(N) @ Y5                              # = 25 rN n^-1 Y  (n^-1 = n for unit n)
    ok &= eqm(x @ x, -(25 * rN) ** 2 * I8) and all(eqm(x @ g_, g_ @ x) for g_ in gam_list(N))
ok &= rank_Q([Y5, T]) == 2                       # not a multiple of T
# joint commutant of l_int(n) and J(n) (Theorem 4.1 conditions), two clocks: span{T, e4e5e6}
def lint_and_J(N):
    vs = [tv(N)] + [s[a] for a in (1, 2, 3)]
    return [vs[a] @ vs[b] for a in range(4) for b in range(a + 1, 4)] + [J(N)]
for two in ([[1, 0, 0], [1, 2, 2]], [[1, 2, 2], [2, 3, 6]]):
    ok &= nullity_p(sys_clock(two, lint_and_J)) == 2
    ok &= all(all(eqm(tv(N) @ Yc @ g_, g_ @ tv(N) @ Yc) for g_ in lint_and_J(N)) for N in two for Yc in (T, S3))
# n Omega_n = |n|^2 e4e5e6, J(n) Omega_n = |n|^2 Gamma (symbolic n, abstract algebra), T Gamma = -e4e5e6
OmA = nA * eb(4, 5, 6)
ok &= mveq(nA * OmA, nn2 * eb(4, 5, 6)) and mveq(JA * OmA, nn2 * eb(1, 2, 3, 4, 5, 6)) and mveq(TA * eb(1, 2, 3, 4, 5, 6), -eb(4, 5, 6))
# n u_a Omega_n = -u_a e4e5e6
for key, (N, U1, U2, r) in CLOCKS.items():
    ok &= all(eqm(tv(N) @ tv(Ua) @ (tv(N) @ S3), -r * r * tv(Ua) @ S3) for Ua in (U1, U2))
check(19, "uniqueness of T: {Y : n^-1 Y commutes with gamma^mu_n} is R T for the three clocks e1, (1,2,2)/3, (2,3,6)/7 and "
          "span{T, (n x n') e4e5e6} for two clocks (nullity 1, 2 mod p; solutions exhibited); for the clocks e1, (0,3,4)/5 the "
          "element Y = (3T + 4 w e4e5e6)/5, w = n x n', makes n^-1 Y a square root of -1 commuting with the gamma^mu_n at both "
          "clocks and is not a multiple of T, so two clocks do not single out +-T even with Y^2 fixed; with the conditions of "
          "Theorem 4.1 (l_int(n) and J(n)) two clocks give span{T, e4e5e6}; n Omega_n = |n|^2 e4e5e6, J(n)Omega_n = |n|^2 Gamma, "
          "n u_a Omega_n = -u_a e4e5e6", ok)

# =====================================================================================================================
# s4-20  the earlier covariant mass e4e5e6
# =====================================================================================================================
ok = True
TRI = [blade(I) for I in itertools.combinations(range(1, 7), 3)]
TRI_B = np.array([b.ravel() for b in TRI]).T                    # 64 x 20
for N in ([1, 0, 0], [0, 1, 0], [1, 1, 1], [2, 3, 6], [-1, 4, 8]):   # (1,1,1): the symmetric clock W, up to scale
    al = [tv(N) @ s[a] for a in (1, 2, 3)]
    sysm = np.vstack([lin_XA_plus_BX(x, x) for x in al]) @ TRI_B
    ok &= nullity_p(sysm) == 2 and all(iszm(T @ x + x @ T) and iszm(S3 @ x + x @ S3) for x in al)
p1, p2, p3, mm = sp.symbols('p1 p2 p3 m')
HA = (nA * E[4]) * p1 + (nA * E[5]) * p2 + (nA * E[6]) * p3 + eb(4, 5, 6) * mm
ok &= mveq((HA * HA).grade(0), HA * HA) and sp.expand((HA * HA).d.get(0, 0) - (nn2 * (p1 ** 2 + p2 ** 2 + p3 ** 2) + mm ** 2)) == 0
ok &= eqm(S3.T, S3) and eqm(S3 @ S3, I8) and all(eqm(S3 @ J(N), J(N) @ S3) for N in ([1, 0, 0], [2, 3, 6]))
ok &= eqm(J([2, 3, 6]) @ S3, T @ (tv([2, 3, 6]) @ S3))                     # J(n) e4e5e6 = T Omega_n (both scaled by |N|)
ok &= not eqm(7 * J([1, 0, 0]) @ S3, J([2, 3, 6]) @ S3)
ok &= eqm(T.T, -T) and eqm(T @ T, -I8)
check(20, "the trivectors anticommuting with alpha_1(n), alpha_2(n), alpha_3(n) form span{e1e2e3, e4e5e6} at e1, e2, the symmetric "
          "clock W ~ (1,1,1), (2,3,6)/7, (-1,4,8)/9 (nullity 2 on the 20 trivectors); H = alpha.p + m e4e5e6 squares to |p|^2+m^2 "
          "(symbolic); e4e5e6 symmetric, square 1, commutes with J(n); its Grassmann kernel J(n) e4e5e6 = T Omega_n depends on the "
          "clock; T is antisymmetric with T^2 = -1, so it is no beta", ok)

# =====================================================================================================================
# s4-21  clock dependence
# =====================================================================================================================
ok = True
def lint_span(N):
    vs = [tv(N)] + [s[a] for a in (1, 2, 3)]
    return [vs[a] @ vs[b] for a in range(4) for b in range(a + 1, 4)]
ok &= rank_Q(lint_span([1, 0, 0]) + lint_span([-1, 0, 0])) == 6
ok &= rank_Q(lint_span([1, 0, 0]) + lint_span([2, 3, 6])) > 6 and rank_Q(lint_span([2, 3, 6]) + lint_span([-1, 4, 8])) > 6
for key, (N, U1, U2, r) in CLOCKS.items():
    ok &= eqm(T @ GAM, -S3) and eqm(tv(N) @ (tv(N) @ S3), r * r * S3) and eqm(J(N) @ (tv(N) @ S3), r * r * GAM)
check(21, "l_int(n) = l_int(n') for n' = -n and not for distinct clocks (e1 vs (2,3,6)/7, (2,3,6)/7 vs (-1,4,8)/9); "
          "T Gamma = -e4e5e6 = -n Omega_n and Gamma = J(n) Omega_n at four clocks (clock independent)", ok)

# =====================================================================================================================
# s4-22  units of the Pauli Lagrangian (SI exponent vectors over kg, m, s, A)
# =====================================================================================================================
def U_(kg=0, m=0, s_=0, A=0):
    return np.array([kg, m, s_, A])
J_ = U_(1, 2, -2)
V_ = J_ - U_(A=1, s_=1)            # V = J/C, C = A s
Tesla = V_ + U_(s_=1) - U_(m=2)    # T = V s / m^2
mu_u = J_ - Tesla                  # J/T
d_u = U_(A=1, s_=1, m=1)           # C m
c_u = U_(m=1, s_=-1)
Efield = V_ - U_(m=1)
psi2 = U_(m=-3)
ok = np.array_equal(Tesla + c_u, Efield)                                 # [cB] = [E]
ok &= np.array_equal(mu_u - c_u + Efield + psi2, J_ - U_(m=3))           # (mu/c) X psi^2 : J m^-3
ok &= np.array_equal(d_u + Efield + psi2, J_ - U_(m=3))                  # d X psi^2 : J m^-3
ok &= np.array_equal(mu_u, U_(A=1, m=2))                                 # J/T = A m^2
check(22, "units: [cB] = [E] = V/m; (mu/c) X_EM psi^T psi and d X_EM psi^T psi have the unit J m^-3 for [mu] = J/T (= A m^2), "
          "[d] = C m, [psi] = m^-3/2", ok)

# =====================================================================================================================
# s4-23  couplings beyond linear order in X exist: the claim 'only trilinear' needs the restriction to linear in X
# =====================================================================================================================
ok = True
for key in ('e1', '236'):
    N, U1, U2, r = CLOCKS[key]
    ok &= all(iszm(a.T @ T + T @ a) for a in gens_rho_n(N, r))          # T is a rho_n-invariant form
    ok &= all(iszm(admat(Y).T @ ETA6 + ETA6 @ admat(Y)) for Y in GENS_G)  # eta(X,X) is l_G-invariant
    ok &= eqm(T.T, -T) and not iszm(T)                                    # psi^T T psi nonzero for Grassmann psi
check(23, "the quartic coupling (psi^T T psi) eta(X,X) is covariant and derivative-free (T is rho_n-invariant and antisymmetric, "
          "eta(X,X) is l_G-invariant): the trilinear couplings are the only covariant derivative-free ones that are linear in X "
          "and quadratic in psi, not the only ones", ok)

# =====================================================================================================================
# s4-24  r3 / r4 agreement: Dirac boosts even, octant boosts odd; Dirac couples d_a to s_a through gamma^0 gamma^a
# =====================================================================================================================
ok = True
for key, (N, U1, U2, r) in CLOCKS.items():
    ok &= all(eqm(GAM @ rhoK2N(i, N), rhoK2N(i, N) @ GAM) for i in (1, 2, 3))     # Dirac boosts even
    ok &= all(eqm(tv(N) @ (tv(N) @ s[a]), r * r * s[a]) for a in (1, 2, 3))      # gamma^0 alpha_a ~ s_a
ok &= all(eqm(GAM @ Q2[i], -(Q2[i] @ GAM)) for i in (1, 2, 3)) and all(eqm(GAM @ L2[i], L2[i] @ GAM) for i in (1, 2, 3))
check(24, "r3/r4 agreement: the Dirac boosts rho_n(K_i) are even and the octant boosts Q_i odd (r3 'main obstruction'); "
          "gamma^0 gamma^a = alpha_a(n) with n alpha_a(n) = s_a: the Dirac operator couples d_a to the spatial generators, the octant "
          "operator D = sum e_a d_a to the temporal ones (r3 remark after its Maxwell theorem)", ok)

print(f"# s4: {24 - len(FAILS)} of 24 checks pass")
sys.exit(1 if FAILS else 0)
