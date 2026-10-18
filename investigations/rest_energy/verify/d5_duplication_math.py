#!/usr/bin/env python3
"""Track d5 -- Duplication: the mathematics of Cl(3,3) -> Cl(6,6).

    python3 investigations/rest_energy/verify/d5_duplication_math.py

Prints one 'PASS d5-<id>: ...' line per verified claim and exits 0 only if every check passes.

Conventions are those of the unified paper (tex/s02_arena.tex eq. realisation and eq. tower; s03 prop:twoK; s12
thm:GLorentz / thm:emfield; s13 prop:commutantsn, prop:mixed, prop:mixedcost; s10 thm:algebra, prop:orthospin; s17
eq:wdef, prop:several, thm:so9C, thm:nocommon; s26 prop:rigid, def:towervac, thm:towervac, thm:S4).
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1), e_1..e_6 realised on R^8 by signed permutation matrices,
  omega = e_1...e_6, tower generators e^(1)_a = e_a (x) 1, e^(2)_a = omega (x) e_a on S_2 = R^8 (x) R^8 = R^64,
  single index E_{6(k-1)+a} = e^(k)_a.  G e_i = e_{i+3}, G e_{i+3} = -e_i; X = sum_i (B_i e_i + E_i e_{i+3}),
  z = B + iE (G acts as i), q(z) = z.z, E6 = |z|^2/2, E0 = |q|/2, S = E x B (frame Poynting vector).
  V_2 = V + V with basis (copy 1: e_1..e_6, copy 2: e_1..e_6), eta_2 = 1_2 (x) eta, G_2 = 1_2 (x) G,
  eps = [[0,-1],[1,0]] on the copy index, so eps (x) 1_6 maps copy 1 to copy 2 (the copy rotation).

Exactness: every Clifford statement is checked in integer arithmetic (signed permutation matrices, or an exact
bitmask blade algebra with rational coefficients); commutants are exact null spaces over Q (sympy DomainMatrix);
symbolic identities are expanded with sympy; semisimplicity is tested exactly (square-free part of the
characteristic polynomial annihilates the matrix), with rational parameters only.  The only floating-point checks
are the spin lift of the pi/4 copy rotation (tolerance 1e-12), the seesaw numerics and the sup-norm of e^{Mt}
(stated where made).  Runtime: a few seconds.
"""
import itertools
import math
import sys
import time

import numpy as np
import scipy.linalg as sla
import sympy as sp
from sympy import QQ
from sympy.polys.matrices import DomainMatrix

T0 = time.time()
FAILS = []


def report(cid, ok, text):
    if ok:
        print(f"PASS {cid}: {text}")
    else:
        print(f"FAIL {cid}: {text}")
        FAILS.append(cid)


def claim(cid, text):
    """Decorator: run the check, report PASS/FAIL; an exception is a FAIL."""
    def deco(fn):
        try:
            ok = bool(fn())
        except Exception as exc:  # pragma: no cover - reported as failure
            import traceback
            traceback.print_exc()
            print(f"   exception in {cid}: {exc!r}")
            ok = False
        report(cid, ok, text)
        return fn
    return deco


# ============================================================================================ exact linear algebra
def to_dm(M):
    """sympy Matrix with rational entries -> DomainMatrix over QQ."""
    rows = [[QQ(int(sp.Rational(x).p), int(sp.Rational(x).q)) for x in M.row(i)] for i in range(M.rows)]
    return DomainMatrix(rows, (M.rows, M.cols), QQ)


def rank_q(M):
    return to_dm(sp.Matrix(M)).rank()


def nullspace_q(M):
    """Basis (list of sympy column vectors) of the exact rational null space of M."""
    M = sp.Matrix(M)
    if M.rows == 0:
        return [sp.eye(M.cols)[:, j] for j in range(M.cols)]
    ns = to_dm(M).nullspace().to_Matrix()
    return [ns.row(i).T for i in range(ns.rows)]


lam = sp.Symbol('lam')


def is_semisimple(M):
    """Exact: M (rational) is diagonalisable over C iff sqf(charpoly)(M) = 0.  Returns (bool, charpoly Poly)."""
    dM = to_dm(M)
    coeffs = [sp.Rational(int(c.numerator), int(c.denominator)) for c in dM.charpoly()]
    P = sp.Poly(coeffs, lam)
    r = sp.Poly(sp.sqf_part(P.as_expr()), lam)
    n = M.shape[0]
    res = DomainMatrix.zeros((n, n), QQ)
    Id = DomainMatrix.eye(n, QQ)
    for c in r.all_coeffs():
        c = sp.Rational(c)
        res = res * dM + Id * QQ(int(c.p), int(c.q))
    return res.is_zero_matrix, P


def kr(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = sp.kronecker_product(out, m)
    return out


def to_int_np(M):
    """rational sympy Matrix -> integer numpy array, scaled by the lcm of the denominators (scaling does not change
    commutation conditions)."""
    M = sp.Matrix(M)
    den = 1
    for x in M:
        den = sp.ilcm(den, sp.Rational(x).q)
    return np.array((M * den).tolist(), dtype=np.int64)


def nullspace_int(A):
    """exact rational null space of an integer numpy matrix; returns a list of sympy column vectors."""
    A = np.asarray(A, dtype=np.int64)
    if A.shape[0] == 0:
        return [sp.eye(A.shape[1])[:, j] for j in range(A.shape[1])]
    A = A[np.any(A != 0, axis=1)]
    if A.shape[0] == 0:
        return [sp.eye(A.shape[1])[:, j] for j in range(A.shape[1])]
    dm = DomainMatrix([[QQ(int(x)) for x in row] for row in A.tolist()], A.shape, QQ)
    ns = dm.nullspace().to_Matrix()
    return [ns.row(i).T for i in range(ns.rows)]


def unvec(v, n):
    return sp.Matrix(n, n, lambda i, j: v[j * n + i])


def algebra_basis(n, commute_with=(), eta=None, antisym=False, sym=False):
    """Exact basis of {X in gl(n): [g,X] = 0 for g in commute_with; X^T eta + eta X = 0 if eta (diagonal) is given;
    X^T = -X if antisym; X^T = X if sym}.  vec is column-major: X_ij <-> j*n + i."""
    rows = []
    Id = np.eye(n, dtype=np.int64)
    for g in commute_with:
        gi_ = to_int_np(g)
        rows.append(np.kron(Id, gi_) - np.kron(gi_.T, Id))       # vec(gX - Xg)
    idx = lambda i, j: j * n + i
    if eta is not None:
        d = [int(eta[i, i]) for i in range(n)]
        assert sp.Matrix(eta).is_diagonal()
        R = []
        for i in range(n):
            for j in range(i, n):
                r = np.zeros(n * n, dtype=np.int64)
                r[idx(j, i)] += d[j]     # (X^T eta)_ij = X_ji eta_jj
                r[idx(i, j)] += d[i]     # (eta X)_ij = eta_ii X_ij
                R.append(r)
        rows.append(np.array(R))
    if antisym or sym:
        R = []
        for i in range(n):
            for j in range(i, n):
                r = np.zeros(n * n, dtype=np.int64)
                r[idx(i, j)] += 1
                r[idx(j, i)] += 1 if antisym else -1
                R.append(r)
        rows.append(np.array(R))
    if not rows:
        return [unvec(sp.eye(n * n)[:, j], n) for j in range(n * n)]
    return [unvec(v, n) for v in nullspace_int(np.vstack(rows))]


def span_dim(mats):
    if not mats:
        return 0
    return rank_q(sp.Matrix.hstack(*[sp.Matrix(m).reshape(m.shape[0] * m.shape[1], 1) for m in mats]))


def same_span(A, B):
    return span_dim(A) == span_dim(B) == span_dim(list(A) + list(B))


# ============================================================================================ realisation on R^8
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def nkron(*ms):
    r = ms[0]
    for x in ms[1:]:
        r = np.kron(r, x)
    return r


e8 = [None, nkron(sx, I2, I2), nkron(sz, sx, I2), nkron(sz, sz, sx),
      nkron(ep, I2, I2), nkron(sz, ep, I2), nkron(sz, sz, ep)]
ETA1 = [None, 1, 1, 1, -1, -1, -1]
I8 = np.eye(8, dtype=np.int64)
I64 = np.eye(64, dtype=np.int64)
OM = e8[1] @ e8[2] @ e8[3] @ e8[4] @ e8[5] @ e8[6]
# tower at n = 2
E = [None] + [np.kron(e8[a], I8) for a in range(1, 7)] + [np.kron(OM, e8[a]) for a in range(1, 7)]
ETA = [None] + ETA1[1:] * 2


def gi(k, a):
    """single index of e^(k)_a."""
    return 6 * (k - 1) + a


def prod(mats, n=64):
    out = np.eye(n, dtype=np.int64)
    for m in mats:
        out = out @ m
    return out


def anti(A, B):
    return A @ B + B @ A


def comm(A, B):
    return A @ B - B @ A


# signed-permutation representation (fast exact blade enumeration)
def to_sp(M):
    """M signed permutation matrix -> (perm, sign) with M[perm[j], j] = sign[j]."""
    n = M.shape[0]
    perm = np.argmax(np.abs(M), axis=0)
    sign = M[perm, np.arange(n)]
    ok = (np.abs(M).sum(axis=0) == 1).all() and (np.abs(sign) == 1).all()
    return perm, sign, ok


def sp_mul(A, B):
    pa, sa = A
    pb, sb = B
    return pa[pb], sa[pb] * sb


def sp_trace(A):
    p, s = A
    idx = np.nonzero(p == np.arange(len(p)))[0]
    return int(s[idx].sum())


def sp_kron(A, B):
    pa, sa = A
    pb, sb = B
    nb = len(pb)
    perm = (pa[:, None] * nb + pb[None, :]).reshape(-1)
    sign = (sa[:, None] * sb[None, :]).reshape(-1)
    return perm, sign


def sp_eq(A, B):
    return (A[0] == B[0]).all() and (A[1] == B[1]).all()


# ============================================================================================ exact blade algebra
SIG = [None] + [1, 1, 1, -1, -1, -1] * 2  # signature of the twelve generators E_1..E_12


def reorder_sign(a, b):
    s = 0
    x = a >> 1
    while x:
        s += bin(x & b).count('1')
        x >>= 1
    return -1 if s % 2 else 1


def gp(a, b):
    """geometric product of basis blades (bitmasks); returns (sign, mask)."""
    s = reorder_sign(a, b)
    c = a & b
    k = 1
    while c:
        if c & 1 and SIG[k] == -1:
            s = -s
        c >>= 1
        k += 1
    return s, a ^ b


def mv_mul(X, Y):
    out = {}
    for a, x in X.items():
        for b, y in Y.items():
            s, c = gp(a, b)
            out[c] = out.get(c, 0) + s * x * y
    return {k: v for k, v in out.items() if v != 0}


def mv_add(*Xs):
    out = {}
    for X in Xs:
        for k, v in X.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def mv_sc(c, X):
    return {k: c * v for k, v in X.items() if c * v != 0}


def mv_comm(X, Y):
    return mv_add(mv_mul(X, Y), mv_sc(-1, mv_mul(Y, X)))


def bl(*ks):
    X = {0: sp.Integer(1)}
    for k in ks:
        X = mv_mul(X, {1 << (k - 1): sp.Integer(1)})
    return X


def mv_commutant(gens, basis):
    """Exact null space: coefficient vectors c with [g, sum c_j basis_j] = 0 for all g."""
    cols, keys = [], set()
    for B in basis:
        col = {}
        for gi_, g in enumerate(gens):
            for k, v in mv_comm(g, B).items():
                col[(gi_, k)] = v
        cols.append(col)
        keys |= set(col)
    keys = sorted(keys)
    if not keys:
        return [sp.eye(len(basis))[:, j] for j in range(len(basis))]
    idx = {k: i for i, k in enumerate(keys)}
    M = sp.zeros(len(keys), len(basis))
    for j, col in enumerate(cols):
        for k, v in col.items():
            M[idx[k], j] = v
    return nullspace_q(M)


H = sp.Rational(1, 2)


def lint_mv(k):
    """interior Lorentz algebra of constituent k: S_1 = 1/2 e5 e6 (cyclic), T_a = 1/2 e1 e_{a+3}."""
    S = [mv_sc(H, bl(gi(k, 5), gi(k, 6))), mv_sc(H, bl(gi(k, 6), gi(k, 4))), mv_sc(H, bl(gi(k, 4), gi(k, 5)))]
    T = [mv_sc(H, bl(gi(k, 1), gi(k, a + 3))) for a in (1, 2, 3)]
    return S, T


CYC = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]


def lG_mv(k):
    """l_G of constituent k: L_i = 1/2(e_{j+3}e_{k+3} - e_j e_k), K_i = 1/2(e_k e_{j+3} - e_j e_{k+3})."""
    L, K = [], []
    for (i, j, kk) in CYC:
        L.append(mv_sc(H, mv_add(bl(gi(k, j + 3), gi(k, kk + 3)), mv_sc(-1, bl(gi(k, j), gi(k, kk))))))
        K.append(mv_sc(H, mv_add(bl(gi(k, kk), gi(k, j + 3)), mv_sc(-1, bl(gi(k, j), gi(k, kk + 3))))))
    return L, K


def lorentz_brackets(R, B):
    ok = True
    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        ok &= mv_comm(R[i], R[j]) == R[k]
        ok &= mv_comm(R[i], B[j]) == B[k]
        ok &= mv_comm(B[i], B[j]) == mv_sc(-1, R[k])
    return ok


BIVS = [bl(i, j) for i in range(1, 13) for j in range(i + 1, 13)]
TRIVS = [bl(i, j, k) for i in range(1, 13) for j in range(i + 1, 13) for k in range(j + 1, 13)]


def blade_square_sign(m):
    return gp(m, m)[0]


def mv_to_mat(X):
    """multivector (dict) -> 64x64 sympy-free numpy object matrix with rational entries via Fraction scaling."""
    out = np.zeros((64, 64), dtype=object)
    for m, v in X.items():
        ks = [k for k in range(1, 13) if m >> (k - 1) & 1]
        out = out + prod([E[k] for k in ks]).astype(object) * v
    return out


# ============================================================================================ vector-level (V, V_2)
Gs = sp.zeros(6, 6)
for i in range(3):
    Gs[i + 3, i] = 1
    Gs[i, i + 3] = -1
eta = sp.diag(1, 1, 1, -1, -1, -1)
eps = sp.Matrix([[0, -1], [1, 0]])
sgx = sp.Matrix([[0, 1], [1, 0]])
sgz = sp.Matrix([[1, 0], [0, -1]])
I6 = sp.eye(6)
eta2 = kr(sp.eye(2), eta)
G2 = kr(sp.eye(2), Gs)
Cv = kr(eps, I6)  # copy rotation on V_2


def ad6(B):
    """matrix of v -> [B, v] on the six generators of one constituent (B an 8x8 integer matrix)."""
    out = sp.zeros(6, 6)
    for c in range(1, 7):
        Y = B @ e8[c] - e8[c] @ B
        rec = np.zeros((8, 8), dtype=np.int64)
        for d in range(1, 7):
            tr = int(np.trace(e8[d] @ Y))
            assert tr % (8 * ETA1[d]) == 0
            co = tr // (8 * ETA1[d])
            out[d - 1, c - 1] = co
            rec += co * e8[d]
        assert (rec == Y).all(), "commutator not of grade one"
    return out


# vector-level generators (from the Clifford realisation, so conventions are the paper's)
LV = [ad6(e8[j + 3] @ e8[kk + 3] - e8[j] @ e8[kk]) / 2 for (i, j, kk) in CYC]   # L_i on V
KV = [ad6(e8[kk] @ e8[j + 3] - e8[j] @ e8[kk + 3]) / 2 for (i, j, kk) in CYC]   # K_i on V
SV = [ad6(e8[5] @ e8[6]) / 2, ad6(e8[6] @ e8[4]) / 2, ad6(e8[4] @ e8[5]) / 2]     # interior rotations
TV = [ad6(e8[1] @ e8[a + 3]) / 2 for a in (1, 2, 3)]                             # interior boosts


def diag2(A, B=None):
    B = A if B is None else B
    return sp.diag(A, B)


def in_so(X, et):
    return (X.T * et + et * X).is_zero_matrix


# ============================================================================================ claims d5-1: the algebra
@claim("d5-1a", "omega^2 = 1, omega^T = omega, Tr omega = 0, omega anticommutes with e_a; the 12 tower generators "
                "satisfy the Clifford relations of R^{6,6} (integer exact)")
def _():
    ok = (OM @ OM == I8).all() and (OM.T == OM).all() and np.trace(OM) == 0
    ok &= all((anti(OM, e8[a]) == 0).all() for a in range(1, 7))
    for A in range(1, 13):
        for B in range(1, 13):
            target = 2 * ETA[A] * I64 if A == B else 0 * I64
            ok &= (anti(E[A], E[B]) == target).all()
    return ok


SPG = []  # signed-permutation form of the 12 generators
for A in range(1, 13):
    p, s, okp = to_sp(E[A])
    assert okp
    SPG.append((p, s))
ID_SP = (np.arange(64), np.ones(64, dtype=np.int64))
BLADES = {}
for mask in range(1 << 12):
    if mask == 0:
        BLADES[0] = ID_SP
    else:
        low = (mask & -mask).bit_length()  # lowest generator index in mask
        rest = mask & (mask - 1)
        BLADES[mask] = sp_mul(SPG[low - 1], BLADES[rest]) if rest else SPG[low - 1]
        # ordering: e_low * e_rest = blade in increasing order since low < every index in rest


@claim("d5-1b", "the 4096 blades of the 12 generators are signed permutation matrices, every non-identity blade is "
                "traceless, hence they are trace-orthogonal and span M_64(R): Cl(6,6) = M_64(R)")
def _():
    ok = all(sp_trace(BLADES[m]) == 0 for m in range(1, 1 << 12)) and sp_trace(BLADES[0]) == 64
    # orthogonality follows: B_I^T B_J = B_I^{-1} B_J = +-B_{I xor J}; check a sample identity exactly
    for (I, J) in [(0b101101, 0b110011000111), (4095, 1), (0b111111, 0b111111000000)]:
        p1, s1 = BLADES[I]
        inv = (np.argsort(p1), s1[np.argsort(p1)])  # inverse of a signed permutation
        pr = sp_mul(inv, BLADES[J])
        pk, sk = BLADES[I ^ J]
        ok &= (pr[0] == pk).all() and (abs(pr[1] * sk) == 1).all() and len(set((pr[1] * sk).tolist())) == 1
    return ok and len(BLADES) == 64 * 64


def copy_blade8(J):
    """8x8 blade e_J of one constituent in signed-permutation form (J a 6-bit mask)."""
    out = (np.arange(8), np.ones(8, dtype=np.int64))
    for a in range(1, 7):
        if J >> (a - 1) & 1:
            p, s, _ = to_sp(e8[a])
            out = sp_mul(out, (p, s))
    return out


OM_SP = to_sp(OM)[:2]


@claim("d5-1c", "graded tensor rule: for all 64 x 64 pairs, e^(1)_I e^(2)_J = (e_I omega^{|J|}) (x) e_J and "
                "e^(2)_J e^(1)_I = (-1)^{|I||J|} e^(1)_I e^(2)_J (the Klein / Jordan-Wigner sign)")
def _():
    ok = True
    for I in range(64):
        eI = copy_blade8(I)
        for J in range(64):
            eJ = copy_blade8(J)
            lhs = sp_mul(BLADES[I], BLADES[J << 6])
            left = sp_mul(eI, OM_SP) if bin(J).count('1') % 2 else eI
            rhs = sp_kron(left, eJ)
            ok &= sp_eq(lhs, rhs)
            other = sp_mul(BLADES[J << 6], BLADES[I])
            sgn = (-1) ** (bin(I).count('1') * bin(J).count('1'))
            ok &= sp_eq(other, (lhs[0], sgn * lhs[1]))
    return ok


@claim("d5-1d", "the Klein factor is forced: the blades commuting with all e_a (x) 1 are exactly 64, each +-1 (x) e_J "
                "(the commutant 1 (x) M_8); those anticommuting with all are 64, each +-omega (x) e_J")
def _():
    comm_set, anti_set = [], []
    for m in range(1 << 12):
        signs = []
        for a in range(1, 7):
            s1, _ = gp(m, 1 << (a - 1))
            s2, _ = gp(1 << (a - 1), m)
            signs.append(s1 * s2)
        if all(s == 1 for s in signs):
            comm_set.append(m)
        if all(s == -1 for s in signs):
            anti_set.append(m)
    ok = len(comm_set) == 64 and len(anti_set) == 64
    for m in comm_set:
        J = m >> 6
        target = sp_kron((np.arange(8), np.ones(8, dtype=np.int64)), copy_blade8(J))
        ok &= (BLADES[m][0] == target[0]).all() and (abs(BLADES[m][1] * target[1]) == 1).all() \
            and len(set((BLADES[m][1] * target[1]).tolist())) == 1
    for m in anti_set:
        J = m >> 6
        target = sp_kron(OM_SP, copy_blade8(J))
        ok &= (BLADES[m][0] == target[0]).all() and len(set((BLADES[m][1] * target[1]).tolist())) == 1
    return ok


@claim("d5-1e", "Fock reading: n+-_i = (e_i +- e_{i+3})/2 satisfy the CAR of 3 modes at n=1 and, with the tower, of 6 "
                "modes at n=2 (copy-2 modes = omega (x) n+-: Jordan-Wigner strings); N_i = (1 - omega_i)/2 with "
                "omega_i = e_i e_{i+3}; spec N = 0..6 with multiplicities 1,6,15,20,15,6,1; (-1)^{N_1} = -omega (x) 1")
def _():
    cp = [None] + [e8[i] + e8[i + 3] for i in (1, 2, 3)]   # 2 n+
    cm = [None] + [e8[i] - e8[i + 3] for i in (1, 2, 3)]   # 2 n-
    ok = True
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            ok &= (anti(cp[i], cp[j]) == 0).all() and (anti(cm[i], cm[j]) == 0).all()
            ok &= (anti(cp[i], cm[j]) == (4 * I8 if i == j else 0 * I8)).all()
        ok &= (cp[i] @ cm[i] == 2 * (I8 - e8[i] @ e8[i + 3])).all()   # 4 N_i = 2(1 - omega_i)
    CP = [None] + [E[i] + E[i + 3] for i in (1, 2, 3)] + [E[i + 6] + E[i + 9] for i in (1, 2, 3)]
    CM = [None] + [E[i] - E[i + 3] for i in (1, 2, 3)] + [E[i + 6] - E[i + 9] for i in (1, 2, 3)]
    for i in range(1, 7):
        for j in range(1, 7):
            ok &= (anti(CP[i], CP[j]) == 0).all() and (anti(CM[i], CM[j]) == 0).all()
            ok &= (anti(CP[i], CM[j]) == (4 * I64 if i == j else 0 * I64)).all()
    for i in (4, 5, 6):  # copy-2 modes carry the string omega (x) .
        ok &= (CP[i] == np.kron(OM, cp[i - 3])).all() and (CM[i] == np.kron(OM, cm[i - 3])).all()
    N4 = sum(CP[i] @ CM[i] for i in range(1, 7))
    ok &= (N4 == np.diag(np.diag(N4))).all()
    vals, counts = np.unique(np.diag(N4) // 4, return_counts=True)
    ok &= list(vals) == list(range(7)) and list(counts) == [1, 6, 15, 20, 15, 6, 1]
    N1 = sum(CP[i] @ CM[i] for i in (1, 2, 3)) // 4
    ok &= (np.diag((-1) ** np.diag(N1)) == -np.kron(OM, I8)).all()
    ok &= (np.diag((-1) ** (np.diag(N4) // 4)) == prod([E[A] for A in range(1, 13)])).all()
    return ok


PAR = (1 - np.diag(OM)) // 2          # omega = (-1)^parity on the standard basis (omega is diagonal)
DK = np.diag([(-1) ** (PAR[i] * PAR[j]) for i in range(8) for j in range(8)]).astype(np.int64)
TAU = np.zeros((64, 64), dtype=np.int64)
for i in range(8):
    for j in range(8):
        TAU[j * 8 + i, i * 8 + j] = 1    # naive swap x_i (x) x_j -> x_j (x) x_i
TK = DK @ TAU                            # Koszul (graded) swap
FV = [None] + [E[a] + E[a + 6] for a in range(1, 7)]     # sqrt2 f_a
GV = [None] + [E[a] - E[a + 6] for a in range(1, 7)]     # sqrt2 g_a


@claim("d5-1f", "exchange of the constituents: the Koszul swap tau_K = D tau conjugates e^(1)_a <-> e^(2)_a, "
                "tau_K^2 = 1, Tr tau_K = 0 (32 + 32), and tau_K = g_1 g_2 ... g_6 exactly (the volume of the "
                "relative half); the naive swap does not implement the exchange")
def _():
    ok = (TK @ TK == I64).all() and np.trace(TK) == 0
    ok &= all((TK @ E[a] @ TK == E[a + 6]).all() and (TK @ E[a + 6] @ TK == E[a]).all() for a in range(1, 7))
    ok &= (prod([GV[a] for a in range(1, 7)]) == 8 * TK).all()
    ok &= not (TAU @ E[1] @ TAU == E[7]).all()
    return ok


# ============================================================================================ claims d5-2: CM / relative
@claim("d5-2a", "f_a = (e^(1)_a + e^(2)_a)/sqrt2 and g_a = (e^(1)_a - e^(2)_a)/sqrt2: {f_a,f_b} = {g_a,g_b} = "
                "2 eta_ab, {f_a, g_b} = 0 for all a, b: Cl(6,6) = Cl(f) (graded x) Cl(g)")
def _():
    ok = True
    for a in range(1, 7):
        for b in range(1, 7):
            t = 4 * ETA1[a] * I64 if a == b else 0 * I64
            ok &= (anti(FV[a], FV[b]) == t).all() and (anti(GV[a], GV[b]) == t).all()
            ok &= (anti(FV[a], GV[b]) == 0).all()
    return ok


CB = sum(ETA[a] * E[a] @ E[a + 6] for a in range(1, 7))   # 2 * copy rotation C_b = sum eta_aa e1_a e2_a / 2


@claim("d5-2b", "the copy rotation C_b = (1/2) sum_a eta_aa e^(1)_a e^(2)_a acts by e^(1) -> -e^(2), e^(2) -> e^(1); "
                "its spin lift through pi/4 carries e^(1)_a -> g_a, e^(2)_a -> f_a (float, 1e-12); "
                "C_b = -(1/2) sum eta_aa f_a g_a; exp(2 pi C_b) = 1")
def _():
    ok = True
    for a in range(1, 7):
        ok &= (comm(CB, E[a]) == -2 * E[a + 6]).all() and (comm(CB, E[a + 6]) == 2 * E[a]).all()
    # spin lift U = prod_a (cos(pi/8) + sin(pi/8) eta_aa e1_a e2_a), commuting factors
    c, s = math.cos(math.pi / 8), math.sin(math.pi / 8)
    U = np.eye(64)
    for a in range(1, 7):
        U = U @ (c * np.eye(64) + s * ETA[a] * (E[a] @ E[a + 6]))
    Ui = np.linalg.inv(U)
    for a in range(1, 7):
        ok &= np.allclose(U @ E[a] @ Ui, GV[a] / math.sqrt(2), atol=1e-12)
        ok &= np.allclose(U @ E[a + 6] @ Ui, FV[a] / math.sqrt(2), atol=1e-12)
    ok &= (2 * CB == -sum(ETA[a] * FV[a] @ GV[a] for a in range(1, 7))).all()   # 4 C_b = -sum eta (sqrt2 f)(sqrt2 g)
    # exp(2 pi C_b): six commuting terms t_a = eta e1 e2 / 2 with (e1_a e2_a)^2 = -1, so each factor is
    # cos(pi) + sin(pi) e1e2 = -1 and the product is +1
    for a in range(1, 7):
        ok &= ((E[a] @ E[a + 6]) @ (E[a] @ E[a + 6]) == -I64).all()
        for b in range(1, 7):
            ok &= (comm(E[a] @ E[a + 6], E[b] @ E[b + 6]) == 0).all()
    return ok


@claim("d5-2c", "what the halves carry: sum_k e^(k)_1 = sqrt2 f_1 (the duplicated clock), sum_k e^(k)_a e^(k)_b = "
                "f_a f_b + g_a g_b for all a < b (diagonal l_int, l_G, J_1 + J_2 split without cross terms), G_2 "
                "preserves span f and span g, tau_K commutes with every f_a and anticommutes with every g_a, and "
                "anticommutes with the copy rotations C_b and M_12 (exchange reverses copy charge)")
def _():
    ok = (E[1] + E[7] == FV[1]).all()
    for a in range(1, 7):
        for b in range(a + 1, 7):
            ok &= (2 * (E[a] @ E[b] + E[a + 6] @ E[b + 6]) == FV[a] @ FV[b] + GV[a] @ GV[b]).all()
        ok &= (TK @ FV[a] == FV[a] @ TK).all() and (TK @ GV[a] == -GV[a] @ TK).all()
    # G_2 on V_2 in the (f, g) basis is block diagonal: the change of basis commutes with 1_2 (x) G
    Rfg = kr(sp.Matrix([[1, 1], [1, -1]]), I6)  # sqrt2 * (f, g) coordinates
    ok &= (Rfg * G2 - G2 * Rfg).is_zero_matrix
    M12 = sum(e * (E[gi(1, mu)] @ E[gi(2, mu)]) for mu, e in [(1, 1), (4, -1), (5, -1), (6, -1)])
    ok &= (TK @ CB == -CB @ TK).all() and (TK @ M12 == -M12 @ TK).all()
    return ok


# ============================================================================================ claims d5-3: the energy
def rot_diag():
    return [diag2(L) for L in LV]


@claim("d5-3a", "symmetry of the duplicated energy: Cent_gl(12)(diag rotations, G_2) = gl(2) (x) span{1,G} (dim 8); "
                "its part in so(12) is u(2) = span{eps(x)1, 1(x)G, sz(x)G, sx(x)G} (dim 4); in so(6,6) so(2,C) = "
                "span{eps(x)1, eps(x)G} (dim 2); in so(12) and so(6,6) only so(2) = R eps(x)1; the other three "
                "generators of u(2) are eta_2-self-adjoint")
def _():
    gens = rot_diag() + [G2]
    full = algebra_basis(12, gens)
    u2 = algebra_basis(12, gens, antisym=True)
    so2C = algebra_basis(12, gens, eta=eta2)
    so2 = algebra_basis(12, gens, eta=eta2, antisym=True)
    expected_full = [kr(A, B) for A in [sp.eye(2), eps, sgx, sgz] for B in [I6, Gs]]
    u2_named = [kr(eps, I6), kr(sp.eye(2), Gs), kr(sgz, Gs), kr(sgx, Gs)]
    ok = len(full) == 8 and same_span(full, expected_full)
    ok &= len(u2) == 4 and same_span(u2, u2_named)
    ok &= len(so2C) == 2 and same_span(so2C, [kr(eps, I6), kr(eps, Gs)])
    ok &= len(so2) == 1 and same_span(so2, [kr(eps, I6)])
    for X in u2_named[1:]:
        ok &= (eta2.inv() * X.T * eta2 - X).is_zero_matrix      # eta_2-self-adjoint
    return ok


@claim("d5-3b", "dual pair U(3) x U(2) in U(6): inside u(6) = so(12) cap Cent(G_2) (dim 36) the commutant of the "
                "diagonal u(3) (dim 9) is u(2) (x) 1 (dim 4), and the commutant of u(2) (x) 1 is the diagonal u(3)")
def _():
    u6 = algebra_basis(12, [G2], antisym=True)
    u3 = algebra_basis(6, [Gs], antisym=True)
    diag_u3 = [diag2(A) for A in u3]
    u2_named = [kr(eps, I6), kr(sp.eye(2), Gs), kr(sgz, Gs), kr(sgx, Gs)]
    c1 = algebra_basis(12, [G2] + diag_u3, antisym=True)
    c2 = algebra_basis(12, [G2] + u2_named, antisym=True)
    return len(u6) == 36 and len(u3) == 9 and len(c1) == 4 and same_span(c1, u2_named) \
        and len(c2) == 9 and same_span(c2, diag_u3)


@claim("d5-3c", "duplication maps: the G-linear, rotation-equivariant maps V -> V_2 form the 4-dim space "
                "z -> (alpha z, beta z), alpha, beta in C; such a map is E6-isometric iff |alpha|^2 + |beta|^2 = 1 "
                "(an S^3) and eta-isometric iff alpha, beta are real with alpha^2 + beta^2 = 1 (a circle)")
def _():
    # unknown 12x6 matrix L, vec column-major
    blocks = []
    for (A, B) in [(G2, Gs)] + [(diag2(L), L) for L in LV]:
        # L B - A L = 0  ->  (B^T (x) I_12 - I_6 (x) A) vec L
        blocks.append(kr(B.T, sp.eye(12)) - kr(sp.eye(6), A))
    ns = nullspace_q(sp.Matrix.vstack(*blocks))
    sols = [sp.Matrix(12, 6, lambda i, j: v[j * 12 + i]) for v in ns]
    named = [sp.Matrix.vstack(I6, sp.zeros(6)), sp.Matrix.vstack(Gs, sp.zeros(6)),
             sp.Matrix.vstack(sp.zeros(6), I6), sp.Matrix.vstack(sp.zeros(6), Gs)]
    ok = len(sols) == 4 and same_span(sols, named)
    a1, a2, b1, b2 = sp.symbols('a1 a2 b1 b2', real=True)
    Lg = sp.Matrix.vstack(a1 * I6 + a2 * Gs, b1 * I6 + b2 * Gs)
    ok &= (Lg.T * Lg - (a1**2 + a2**2 + b1**2 + b2**2) * I6).expand().is_zero_matrix
    al, be = a1 + sp.I * a2, b1 + sp.I * b2
    s2 = sp.expand(al**2 + be**2)
    ok &= (Lg.T * eta2 * Lg - (sp.re(s2) * eta + 2 * (a1 * a2 + b1 * b2) * eta * Gs)).expand().is_zero_matrix
    ok &= sp.expand(sp.im(s2) - 2 * (a1 * a2 + b1 * b2)) == 0
    # E6-isometric (|alpha|^2+|beta|^2 = 1) and eta-isometric (Re s2 = 1) together force a2 = b2 = 0, because
    # |alpha|^2 + |beta|^2 - Re(alpha^2 + beta^2) = 2 (a2^2 + b2^2) identically
    ok &= sp.expand((a1**2 + a2**2 + b1**2 + b2**2) - sp.re(s2) - 2 * (a2**2 + b2**2)) == 0
    return ok


# symbolic field-strength quantities
bx = sp.symbols('B1:4', real=True)
ex = sp.symbols('E1:4', real=True)
zsym = sp.Matrix([bx[i] + sp.I * ex[i] for i in range(3)])


def qf(z):
    return sp.expand((z.T * z)[0])


def stress(z):
    """Maxwell stress tensor T^{mu nu} of the field z = B + iE (frame variables), 4x4 symbolic.
    T^00 = |z|^2/2, T^{0i} = textbook Poynting = -(E x B) = -(i/2)(conj z x z), T^{ij} = -Re(conj z_i z_j)
    + delta_ij |z|^2 / 2.  (The sign of T^{0i} does not affect any statement below.)"""
    zc = z.conjugate()
    T = sp.zeros(4, 4)
    T[0, 0] = sp.expand((zc.T * z)[0] / 2)
    S = (sp.I / 2) * zc.cross(z)
    for i in range(3):
        T[0, i + 1] = T[i + 1, 0] = sp.expand(-S[i])
        for j in range(3):
            T[i + 1, j + 1] = sp.expand(-sp.re(zc[i] * z[j]) + (T[0, 0] if i == j else 0))
    return T.applyfunc(lambda x: sp.expand(sp.re(x)) if x.is_real is not True else x)


ETA4 = sp.diag(1, -1, -1, -1)


@claim("d5-3d", "on the image of a duplication z -> (alpha z, beta z): q_2 = (alpha^2 + beta^2) q, sum_k E0^(k) = "
                "(|alpha|^2 + |beta|^2) E0, and the summed stress tensor T(alpha z) + T(beta z) = (|alpha|^2+|beta|^2) "
                "T(z); T(z) is duality invariant and obeys Rainich's identity (T eta)^2 = E0^2 1; at the equal split "
                "with relative duality angle theta, |q_2| = |cos theta| |q|")
def _():
    a1, a2, b1, b2, th = sp.symbols('a1 a2 b1 b2 theta', real=True)
    al, be = a1 + sp.I * a2, b1 + sp.I * b2
    ok = sp.expand(qf(al * zsym) + qf(be * zsym) - (al**2 + be**2) * qf(zsym)) == 0
    # |q(alpha z)| = |alpha|^2 |q(z)|: compare squares
    ok &= sp.expand(sp.Abs(al)**2 * sp.Abs(al)**2 - (a1**2 + a2**2)**2) == 0
    T = stress(zsym)
    Tal = stress(al * zsym).applyfunc(sp.expand)
    Tbe = stress(be * zsym).applyfunc(sp.expand)
    ok &= (Tal + Tbe - (a1**2 + a2**2 + b1**2 + b2**2) * T).applyfunc(sp.expand).is_zero_matrix
    Tdual = stress(sp.exp(sp.I * th) * zsym).applyfunc(lambda x: sp.expand(sp.simplify(x)))
    ok &= (Tdual - T).applyfunc(sp.simplify).is_zero_matrix
    q = qf(zsym)
    E0sq = sp.expand(sp.re(q)**2 / 4 + sp.im(q)**2 / 4)
    Tm = T * ETA4
    ok &= (Tm * Tm - E0sq * sp.eye(4)).applyfunc(sp.expand).is_zero_matrix
    # equal split, relative duality angle theta: alpha = 1/sqrt2, beta = e^{i theta}/sqrt2
    fac = sp.Rational(1, 2) * (1 + sp.exp(2 * sp.I * th))
    ok &= sp.simplify(sp.Abs(fac)**2 - sp.cos(th)**2) == 0
    return ok


@claim("d5-3e", "X -> (cos(phi) X, sin(phi) R X) with R in U(3): the summed stress tensor is preserved for all X iff "
                "R is a duality rotation (T determines z z^* bijectively); q_2 iff R is real orthogonal; both iff "
                "R = +-1; and a G-linear W = U + iV preserving both E6 (+) E6 and q_2 has V = 0, "
                "since W^*W - W^T W = 2 V^T V - 2i V^T U")
def _():
    # T^{ij}, T^{0i} <-> Re and Im of z z^*: linear bijection on the 9 real entries (exact)
    T = stress(zsym)
    zz = zsym * zsym.conjugate().T
    herm_coords = [sp.expand(sp.re(zz[i, j])) for i in range(3) for j in range(i, 3)] + \
                  [sp.expand(sp.im(zz[i, j])) for i in range(3) for j in range(i + 1, 3)]
    t_coords = [T[i + 1, j + 1] for i in range(3) for j in range(i, 3)] + [T[0, i] for i in (1, 2, 3)]
    # express both as linear forms in the 9 monomial-combinations; check the linear map between them is invertible
    mons = sorted({m for c in herm_coords + t_coords for m in sp.Poly(c, *bx, *ex).monoms()})
    def coeffs(c):
        P = sp.Poly(c, *bx, *ex)
        return [P.coeff_monomial(m) for m in mons]
    Hm = sp.Matrix([coeffs(c) for c in herm_coords])
    Tm = sp.Matrix([coeffs(c) for c in t_coords])
    ok = Hm.rank() == 9 and sp.Matrix.vstack(Hm, Tm).rank() == 9 and Tm.rank() == 9
    # a non-scalar unitary changes T (numeric witness)
    rng = np.random.default_rng(5)
    Rn, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    zn = rng.normal(size=3) + 1j * rng.normal(size=3)
    def Tnum(z):
        zc = np.conj(z)
        return np.real(np.outer(zc, z)), np.real(0.5j * np.cross(zc, z))
    a, b = Tnum(zn), Tnum(Rn @ zn)
    ok &= not (np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]))
    ph = np.exp(0.7j)
    c = Tnum(ph * zn)
    ok &= np.allclose(a[0], c[0]) and np.allclose(a[1], c[1])
    # W^*W - W^T W identity
    U = sp.Matrix(6, 3, sp.symbols('u0:18', real=True))
    V = sp.Matrix(6, 3, sp.symbols('v0:18', real=True))
    W = U + sp.I * V
    ok &= (W.conjugate().T * W - W.T * W - (2 * V.T * V - 2 * sp.I * V.T * U)).applyfunc(sp.expand).is_zero_matrix
    return ok


@claim("d5-3f", "diagonal Lorentz invariants of two field strengths: the Gram matrix Q_kl = z_k . z_l is invariant "
                "under the diagonal so(3,C); Tr Q = q_2 and det Q are invariant under the copy so(2,C), while Q_11 "
                "(hence E0^(1)) is not")
def _():
    z1 = sp.Matrix(sp.symbols('x1:4')) + sp.I * sp.Matrix(sp.symbols('y1:4'))
    z2 = sp.Matrix(sp.symbols('u1:4')) + sp.I * sp.Matrix(sp.symbols('v1:4'))
    w = sp.Matrix(sp.symbols('w1:4')) + sp.I * sp.Matrix(sp.symbols('s1:4'))
    cr, ci = sp.symbols('cr ci', real=True)
    c = cr + sp.I * ci
    Q = lambda a, b: (a.T * b)[0]
    ok = True
    for (a, b) in [(z1, z1), (z1, z2), (z2, z2)]:
        ok &= sp.expand(Q(w.cross(a), b) + Q(a, w.cross(b))) == 0
    d1, d2 = -c * z2, c * z1
    dQ11 = sp.expand(2 * Q(z1, d1))
    dQ22 = sp.expand(2 * Q(z2, d2))
    dQ12 = sp.expand(Q(d1, z2) + Q(z1, d2))
    ok &= sp.expand(dQ11 + dQ22) == 0
    detd = sp.expand(dQ11 * Q(z2, z2) + Q(z1, z1) * dQ22 - 2 * Q(z1, z2) * dQ12)
    ok &= detd == 0 and dQ11 != 0
    return ok


@claim("d5-3g", "an eta-isometric complex structure exists on R^{6,6}: C_v = eps (x) 1_6 has C_v^2 = -1, "
                "C_v^T C_v = 1, C_v^T eta_2 C_v = eta_2, and commutes with G_2 and the diagonal l_G")
def _():
    ok = (Cv * Cv + sp.eye(12)).is_zero_matrix and (Cv.T * Cv - sp.eye(12)).is_zero_matrix
    ok &= (Cv.T * eta2 * Cv - eta2).is_zero_matrix and (Cv * G2 - G2 * Cv).is_zero_matrix
    for X in LV + KV:
        ok &= (Cv * diag2(X) - diag2(X) * Cv).is_zero_matrix
    return ok


@claim("d5-3h", "Lorentz-compatible part of the duplicated energy symmetry: u(6) cap so(6,6) = so(6) (dim 15, the "
                "paired rotations) against u(3) cap so(3,3) = so(3) (dim 3) at n=1; inside so(6) the centraliser "
                "of C_v is u(3)' = {1 (x) diag(A,A) + eps (x) diag(B,B): A^T = -A, B^T = B} (dim 9), whose part "
                "commuting with the diagonal boosts is R C_v only")
def _():
    so6 = algebra_basis(12, [G2], eta=eta2, antisym=True)
    so3 = algebra_basis(6, [Gs], eta=eta, antisym=True)
    u3p = algebra_basis(12, [G2, Cv], eta=eta2, antisym=True)
    named = []
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        A = sp.zeros(3)
        A[i, j], A[j, i] = 1, -1
        named.append(kr(sp.eye(2), sp.diag(A, A)))
    for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        B = sp.zeros(3)
        B[i, j] = B[j, i] = 1
        named.append(kr(eps, sp.diag(B, B)))
    ok = len(so6) == 15 and len(so3) == 3 and len(u3p) == 9 and same_span(u3p, named)
    ok &= all(in_so(X, eta2) and (X.T + X).is_zero_matrix for X in named)
    # u(3)' is not Lorentz invariant: its part commuting with the diagonal boosts K_i is R C_v only
    u3p_inv = algebra_basis(12, [G2, Cv] + [diag2(K) for K in KV], eta=eta2, antisym=True)
    ok &= len(u3p_inv) == 1 and same_span(u3p_inv, [Cv])
    return ok


@claim("d5-3i", "pair invariant: E_pair^2 = (1/4) Tr((T_tot eta)^2), a full contraction and hence invariant under the "
                "diagonal Lorentz group, equals E0^2 on the duplication image; for the two counter-propagating null "
                "fields z1 = (1,i,0), z2 = (1,-i,0) (E0^(k) = 0, q_2 = 0) it equals 2 > 0, for z2 = z1 it is 0")
def _():
    a1, a2, b1, b2 = sp.symbols('a1 a2 b1 b2', real=True)
    T = stress(zsym)
    q = qf(zsym)
    E0sq = sp.expand(sp.re(q)**2 / 4 + sp.im(q)**2 / 4)
    nrm = a1**2 + a2**2 + b1**2 + b2**2
    Ttot = (nrm * T) * ETA4
    ok = sp.expand((Ttot * Ttot).trace() / 4 - nrm**2 * E0sq) == 0
    z1 = sp.Matrix([1, sp.I, 0])
    z2 = sp.Matrix([1, -sp.I, 0])
    ok &= qf(z1) == 0 and qf(z2) == 0
    Tp = (stress(z1) + stress(z2)) * ETA4
    ok &= sp.simplify((Tp * Tp).trace() / 4) == 2
    Tc = (stress(z1) + stress(z1)) * ETA4
    ok &= sp.simplify((Tc * Tc).trace() / 4) == 0
    return ok


# ============================================================================================ claims d5-4: closure symmetry
@claim("d5-4a", "Cent(G_n) in so(3n,3n) has dimension 3n(3n-1) = 6, 30, 72 at n = 1, 2, 3 (so(3n,C); Theorem 17.19 at "
                "n=3), compact part so(3n) of dimension 3, 15, 36; at n=2 every element is [[R,S],[-S,R]] in the "
                "(temporal; spatial) basis with R, S antisymmetric 6x6")
def _():
    ok = True
    for n, d, dk in [(1, 6, 3), (2, 30, 15), (3, 72, 36)]:
        Gn = kr(sp.eye(n), Gs)
        etan = kr(sp.eye(n), eta)
        ok &= len(algebra_basis(6 * n, [Gn], eta=etan)) == d
        ok &= len(algebra_basis(6 * n, [Gn], eta=etan, antisym=True)) == dk
    # block form at n = 2: permute to (temporal of both copies; spatial of both copies)
    perm = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]
    Pm = sp.zeros(12)
    for new, old in enumerate(perm):
        Pm[new, old] = 1
    for X in algebra_basis(12, [G2], eta=eta2):
        Y = Pm * X * Pm.T
        R, S, S2, R2 = Y[:6, :6], Y[:6, 6:], Y[6:, :6], Y[6:, 6:]
        ok &= (R - R2).is_zero_matrix and (S + S2).is_zero_matrix and (R + R.T).is_zero_matrix \
            and (S + S.T).is_zero_matrix
    return ok


@claim("d5-4b", "so(6,C) = l_G^(1) + l_G^(2) (6+6) + cross part (18); the diagonal rotations act on the cross part with "
                "Casimir j(j+1) in {0 (2), 2 (6), 6 (10)}: complex 1 + 3 + 5; the commutant of the diagonal l_G is the "
                "copy algebra so(2,C) = span{eps (x) 1 (square -1, compact), eps (x) G (square +1, a copy boost)} "
                "and lies in the cross part")
def _():
    so6C = algebra_basis(12, [G2], eta=eta2)
    # cross part: elements with vanishing diagonal blocks (exact subspace)
    coeff = sp.symbols('c0:30')
    Xg = sum((c * X for c, X in zip(coeff, so6C)), sp.zeros(12))
    eqs = list(Xg[:6, :6]) + list(Xg[6:, 6:])
    A, _ = sp.linear_eq_to_matrix(eqs, coeff)
    cr_basis = [sum((v[i] * so6C[i] for i in range(30)), sp.zeros(12)) for v in nullspace_q(A)]
    lG1 = [diag2(X, sp.zeros(6)) for X in LV + KV]
    lG2 = [diag2(sp.zeros(6), X) for X in LV + KV]
    ok = len(cr_basis) == 18 and span_dim(lG1 + lG2 + cr_basis) == 30
    # adjoint Casimir of the diagonal rotations on the cross part, in coordinates of cr_basis
    Bm = sp.Matrix.hstack(*[X.reshape(144, 1) for X in cr_basis])
    # exact coordinates via a left inverse on a pivot set
    piv = to_dm(Bm.T).rref()[1]
    Bsub = Bm.extract(list(piv), list(range(18)))
    Binv = Bsub.inv()
    ad = []
    for L in LV:
        Ld = diag2(L)
        cols = []
        for X in cr_basis:
            Y = Ld * X - X * Ld
            c = Binv * Y.reshape(144, 1).extract(list(piv), [0])
            ok &= (Bm * c - Y.reshape(144, 1)).is_zero_matrix
            cols.append(c)
        ad.append(sp.Matrix.hstack(*cols))
    Cas = -(ad[0]**2 + ad[1]**2 + ad[2]**2)
    ok &= (Cas * (Cas - 2 * sp.eye(18)) * (Cas - 6 * sp.eye(18))).is_zero_matrix
    tr1, tr2 = Cas.trace(), (Cas**2).trace()
    # multiplicities m0, m2, m6 with m0+m2+m6 = 18, 2 m2 + 6 m6 = tr1, 4 m2 + 36 m6 = tr2
    m6 = sp.Rational(tr2 - 2 * tr1, 24)
    m2 = sp.Rational(tr1 - 6 * m6, 2)
    ok &= (18 - m2 - m6, m2, m6) == (2, 6, 10)
    copy = algebra_basis(12, [diag2(X) for X in LV + KV], eta=eta2)
    ok &= len(copy) == 2 and same_span(copy, [kr(eps, I6), kr(eps, Gs)])
    ok &= span_dim(cr_basis + copy) == 18
    ok &= (kr(eps, I6)**2 + sp.eye(12)).is_zero_matrix and (kr(eps, Gs)**2 - sp.eye(12)).is_zero_matrix
    ok &= (kr(eps, I6) * kr(eps, Gs) - kr(eps, Gs) * kr(eps, I6)).is_zero_matrix
    return ok


@claim("d5-4c", "n=2 analogue of Theorem 17.20: Cent(G_2) meets l_int^(1) in 0, and the elements of so(6,6) commuting "
                "with G_2 and l_int^(1) form l_G^(2) = so(3,C) of copy 2 (dim 6)")
def _():
    lint1 = [diag2(X, sp.zeros(6)) for X in SV + TV]
    so6C = algebra_basis(12, [G2], eta=eta2)
    joint = algebra_basis(12, [G2] + lint1, eta=eta2)
    lG2 = [diag2(sp.zeros(6), X) for X in LV + KV]
    inter = span_dim(so6C) + span_dim(lint1) - span_dim(so6C + lint1)
    return inter == 0 and len(joint) == 6 and same_span(joint, lG2)


# ============================================================================================ claims d5-5: Lorentz actions
S1, T1 = lint_mv(1)
S2, T2 = lint_mv(2)
L1, K1 = lG_mv(1)
L2, K2 = lG_mv(2)
ACTIONS = {
    "spectator (k_I,k_B,k_S)=(1,0,1)": ((1, 0, 1), S1 + T1),
    "binding+spectator (0,1,1)": ((0, 1, 1), L1 + K1),
    "diagonal interior (2,0,0)": ((2, 0, 0), [mv_add(a, b) for a, b in zip(S1 + T1, S2 + T2)]),
    "mixed (1,1,0)": ((1, 1, 0), [mv_add(a, b) for a, b in zip(S1 + T1, L2 + K2)]),
    "binding both (0,2,0)": ((0, 2, 0), [mv_add(a, b) for a, b in zip(L1 + K1, L2 + K2)]),
}
COMPACT_BIVS = [b for b in BIVS if blade_square_sign(list(b)[0]) == -1]
BOOST_BIVS = [b for b in BIVS if blade_square_sign(list(b)[0]) == 1]


def thm108(kI, kB, kS):
    """dimension and compact dimension of so(kI) + so(2kI+3kS, 3kS) + so(kB, C)."""
    so = lambda n: n * (n - 1) // 2
    p, q = 2 * kI + 3 * kS, 3 * kS
    return so(kI) + so(p + q) + 2 * so(kB), so(kI) + so(p) + so(q) + so(kB)


@claim("d5-5a", "Lorentz actions at n=2 (Theorem 10.8): commutants in grade two of dimensions 28, 15, 7, 1, 2 with "
                "compact parts 13, 6, 7, 1, 1 for spectator, binding+spectator, diagonal interior, mixed, binding "
                "on both; theta-stable; each matches so(k_I) + so(2k_I+3k_S,3k_S) + so(k_B,C)")
def _():
    ok = lorentz_brackets(S1, T1) and lorentz_brackets(L1, K1)
    ok &= lorentz_brackets([mv_add(a, b) for a, b in zip(S1, L2)], [mv_add(a, b) for a, b in zip(T1, K2)])
    got = []
    for name, (k, gens) in ACTIONS.items():
        d = len(mv_commutant(gens, BIVS))
        dk = len(mv_commutant(gens, COMPACT_BIVS))
        dp = len(mv_commutant(gens, BOOST_BIVS))
        got.append((d, dk))
        ok &= d == dk + dp and (d, dk) == thm108(*k)
    return ok and got == [(28, 13), (15, 6), (7, 7), (1, 1), (2, 1)]


def cas_vec(R, B):
    """Lorentz Casimir on V_2 (12x12) from rotation and boost generators given as multivectors."""
    def admat(X):
        M = sp.zeros(12, 12)
        for c in range(1, 13):
            Y = mv_comm(X, bl(c))
            for m, v in Y.items():
                assert bin(m).count('1') == 1
                d = m.bit_length()
                M[d - 1, c - 1] += v
        return M
    Rm = [admat(x) for x in R]
    Bm = [admat(x) for x in B]
    return -(Rm[0]**2 + Rm[1]**2 + Rm[2]**2) + (Bm[0]**2 + Bm[1]**2 + Bm[2]**2)


def spec_counts(C, values):
    """exact multiplicities of the given eigenvalues of a diagonalisable rational matrix C (checks min. poly.)."""
    n = C.shape[0]
    P = sp.eye(n)
    for v in values:
        P = P * (C - v * sp.eye(n))
    if not P.is_zero_matrix:
        return None
    out = []
    for v in values:
        Pv = sp.eye(n)
        for w in values:
            if w != v:
                Pv = Pv * (C - w * sp.eye(n)) / (v - w)
        out.append(int(Pv.trace()))
    return out


MIXR = [mv_add(a, b) for a, b in zip(S1, L2)]
MIXB = [mv_add(a, b) for a, b in zip(T1, K2)]


@claim("d5-5b", "mixed rung (restating Propositions 13.16-13.17): centraliser R e^(1)_2 e^(1)_3; grade-one content "
                "C = 3 (4), 4 (6), 0 (2); on S_2, (C_mix - 3/2)(C_mix - 11/2) = 0 with ranks 16, 48; for comparison "
                "spectator V + 8.1, binding-on-one Lambda^2 V + 6.1, diagonal interior 2V + 4.1, binding on both "
                "2 Lambda^2 V")
def _():
    gens = MIXR + MIXB
    ns = mv_commutant(gens, BIVS)
    ok = len(ns) == 1
    nz = [(i, ns[0][i]) for i in range(66) if ns[0][i] != 0]
    ok &= len(nz) == 1 and BIVS[nz[0][0]] == bl(gi(1, 2), gi(1, 3))
    ok &= spec_counts(cas_vec(MIXR, MIXB), [0, 3, 4]) == [2, 4, 6]
    ok &= spec_counts(cas_vec(S1, T1), [0, 3]) == [8, 4]
    ok &= spec_counts(cas_vec(L1, K1), [0, 4]) == [6, 6]
    DR = [mv_add(a, b) for a, b in zip(S1, S2)]
    DB = [mv_add(a, b) for a, b in zip(T1, T2)]
    ok &= spec_counts(cas_vec(DR, DB), [0, 3]) == [4, 8]                     # diagonal interior: 2V + 4.1
    BR = [mv_add(a, b) for a, b in zip(L1, L2)]
    BB = [mv_add(a, b) for a, b in zip(K1, K2)]
    ok &= spec_counts(cas_vec(BR, BB), [4]) == [12]                          # binding on both: 2 Lambda^2 V
    # Casimir on S_2 (exact integer arithmetic: 4 C_mix)
    Rm = [mv_to_mat(x) for x in MIXR]
    Bm = [mv_to_mat(x) for x in MIXB]
    C4 = (-(Rm[0] @ Rm[0] + Rm[1] @ Rm[1] + Rm[2] @ Rm[2]) + (Bm[0] @ Bm[0] + Bm[1] @ Bm[1] + Bm[2] @ Bm[2])) * 4
    C4 = np.array(C4, dtype=object)
    ok &= all(sp.Rational(x).q == 1 for x in C4.flat)
    C4 = C4.astype(np.int64)
    ok &= ((C4 - 6 * I64) @ (C4 - 22 * I64) == 0).all()
    # rank of the 3/2 block: P = (11 - 2C)/8 = (22 - C4)/16, Tr P = 16
    ok &= np.trace(22 * I64 - C4) == 16 * 16
    return ok


@claim("d5-5c", "the identification Theta_F of the field strength of copy 1 with the frame field of copy 2 and "
                "Theta_F followed by G are l_mix-invariant and span the invariants in Lambda^2(four-plane_1) (x) V_2 "
                "(dim 2, as in Section 13.6); the full grade-three invariant space has dim 4, the other two lying in "
                "Lambda^3 V_2 (copy 2 alone)")
def _():
    gens = MIXR + MIXB
    gam = lambda mu: gi(1, 1) if mu == 0 else gi(1, mu + 3)
    ThF = mv_add(bl(gam(2), gam(3), gi(2, 1)), bl(gam(3), gam(1), gi(2, 2)), bl(gam(1), gam(2), gi(2, 3)),
                 mv_sc(-1, bl(gam(0), gam(1), gi(2, 4))), mv_sc(-1, bl(gam(0), gam(2), gi(2, 5))),
                 mv_sc(-1, bl(gam(0), gam(3), gi(2, 6))))

    def g_slot(X):
        out = {}
        for m, v in X.items():
            for a in range(1, 7):
                if m >> (gi(2, a) - 1) & 1:
                    rest = m ^ (1 << (gi(2, a) - 1))
                    na, sg = (a + 3, 1) if a <= 3 else (a - 3, -1)
                    nm = rest | (1 << (gi(2, na) - 1))
                    out[nm] = out.get(nm, 0) + sg * v
        return out
    GTh = g_slot(ThF)
    ok = all(mv_comm(x, ThF) == {} for x in gens) and all(mv_comm(x, GTh) == {} for x in gens)
    four = [gi(1, 1), gi(1, 4), gi(1, 5), gi(1, 6)]
    sub = [bl(a, b, c) for (a, b) in itertools.combinations(four, 2) for c in range(7, 13)]
    ns = mv_commutant(gens, sub)
    ok &= len(ns) == 2
    # Theta_F, G Theta_F in this span: express as coefficient vectors over sub
    def coeffvec(X):
        return sp.Matrix([X.get(list(b)[0], 0) for b in sub])
    M = sp.Matrix.hstack(*ns)
    ok &= M.rank() == 2 and sp.Matrix.hstack(M, coeffvec(ThF), coeffvec(GTh)).rank() == 2
    ok &= coeffvec(ThF).rank() == 1 and sp.Matrix.hstack(coeffvec(ThF), coeffvec(GTh)).rank() == 2
    full = mv_commutant(gens, TRIVS)
    ok &= len(full) == 4
    copy2 = [bl(a, b, c) for (a, b, c) in itertools.combinations(range(7, 13), 3)]
    ok &= len(mv_commutant(gens, copy2)) == 2
    return ok


# ============================================================================================ claims d5-6: spin, statistics
def mvm(X):
    return np.array(mv_to_mat(X), dtype=object)


@claim("d5-6a", "Proposition 10.9 at n=2: the copy rotations M_12 (interior four-planes: 4 commuting terms of square "
                "-1/4) and C_b (binding frames: 6 terms) have eigenvalues i.{-2..2} and i.{-3..3} and exp(2 pi .) = 1; "
                "a surplus rotation (1/2) e_s e_t has exp(2 pi .) = -1")
def _():
    gam = lambda k, mu: E[gi(k, 1)] if mu == 0 else E[gi(k, mu + 3)]
    terms = [e * gam(1, mu) @ gam(2, mu) for mu, e in [(0, 1), (1, -1), (2, -1), (3, -1)]]  # 2 x terms of M_12
    ok = all((t @ t == -I64).all() for t in terms) and all((comm(a, b) == 0).all() for a in terms for b in terms)
    M2 = sum(terms)                          # 2 M_12, antisymmetric integer matrix
    ok &= (M2.T == -M2).all()
    # eigenvalues of M_12 = M2/2 are i m with m in {-2..2}: check prod (M_12^2 + m^2) = 0 for m = 0, 1, 2
    Msq = M2 @ M2                             # = 4 M_12^2
    ok &= (Msq @ (Msq + 4 * I64) @ (Msq + 16 * I64) == 0).all()
    ok &= not (Msq @ (Msq + 4 * I64) == 0).all()
    CBsq = CB @ CB                            # (2 C_b)^2
    ok &= (CBsq @ (CBsq + 4 * I64) @ (CBsq + 16 * I64) @ (CBsq + 36 * I64) == 0).all()
    ok &= not (CBsq @ (CBsq + 4 * I64) @ (CBsq + 16 * I64) == 0).all()
    # exp(2 pi M_12) = prod over commuting terms of (cos pi + sin pi t) = (-1)^4 = 1, exp(2 pi C_b) = (-1)^6 = 1;
    # numerical confirmation
    ok &= np.allclose(sla.expm(2 * math.pi * M2 / 2), np.eye(64), atol=1e-9)
    ok &= np.allclose(sla.expm(2 * math.pi * CB / 2), np.eye(64), atol=1e-9)
    st = E[gi(1, 2)] @ E[gi(1, 3)]
    ok &= (st @ st == -I64).all() and np.allclose(sla.expm(2 * math.pi * st / 2), -np.eye(64), atol=1e-9)
    return ok


def int4(X):
    A = np.array(X, dtype=object) * 4
    assert all(sp.Rational(x).q == 1 for x in A.flat)
    return A.astype(np.int64)


SDm = [mvm(mv_add(a, b)) for a, b in zip(S1, S2)]
TDm = [mvm(mv_add(a, b)) for a, b in zip(T1, T2)]


def trace_counts(C4, vals4):
    """eigenvalue multiplicities of integer matrix C4 (= 4 C) with known minimal polynomial roots vals4."""
    P = np.eye(64, dtype=np.int64)
    for v in vals4:
        P = P @ (C4 - v * I64)
    if not (P == 0).all():
        return None
    out = []
    for v in vals4:
        num = np.eye(64, dtype=object)
        den = 1
        for w in vals4:
            if w != v:
                num = num.dot((C4 - w * I64).astype(object))
                den *= (v - w)
        out.append(sp.Rational(int(np.trace(num)), den))
    return out


@claim("d5-6b", "spin at n=2: a 2 pi rotation is +1 on S_2 under the diagonal interior action, -1 under the spectator "
                "action, +1 under l_G (one or both copies), -1 under the mixed action; spin Casimir: diagonal {0 (16), 2 (48)}, spectator "
                "{3/4 (64)}; diagonal Lorentz Casimir {0 (8), 3 (32), 4 (24)}; C_cop {0 (24), 1 (32), 4 (8)} and "
                "C_cop + C_Delta = 4 (Theorem 10.4 at n=2)")
def _():
    # 2 pi rotations about the third axis, via commuting square -1 blades
    b45 = [E[gi(k, 4)] @ E[gi(k, 5)] for k in (1, 2)]
    b12 = [E[gi(k, 1)] @ E[gi(k, 2)] for k in (1, 2)]
    ok = all((x @ x == -I64).all() for x in b45 + b12)
    ok &= all((comm(x, y) == 0).all() for x in b45 + b12 for y in b45 + b12)
    # exp(2 pi * (1/2) sum of r commuting square -1 blades) = (-1)^r
    rot = lambda blades: sla.expm(math.pi * sum(blades).astype(float))
    ok &= np.allclose(rot(b45), np.eye(64), atol=1e-9)              # diagonal interior S_3
    ok &= np.allclose(rot(b45[:1]), -np.eye(64), atol=1e-9)         # spectator S_3
    ok &= np.allclose(rot([b45[0], -b12[0]]), np.eye(64), atol=1e-9)  # l_G L_3 on one copy
    ok &= np.allclose(rot([b45[0], -b12[0], b45[1], -b12[1]]), np.eye(64), atol=1e-9)
    ok &= np.allclose(rot([b45[0], b45[1], -b12[1]]), -np.eye(64), atol=1e-9)    # mixed: S_3 (x) 1 + 1 (x) L_3
    spin4 = int4(-sum(s.dot(s) for s in SDm))
    ok &= trace_counts(spin4, [0, 8]) == [16, 48]
    sp1 = int4(-sum(mvm(s).dot(mvm(s)) for s in S1))
    ok &= (sp1 == 3 * I64).all()
    lor4 = int4(-sum(s.dot(s) for s in SDm) + sum(t.dot(t) for t in TDm))
    ok &= trace_counts(lor4, [0, 12, 16]) == [8, 32, 24]
    gam = lambda k, mu: E[gi(k, 1)] if mu == 0 else E[gi(k, mu + 3)]
    M2 = sum(e * gam(1, mu) @ gam(2, mu) for mu, e in [(0, 1), (1, -1), (2, -1), (3, -1)])
    ccop4 = -(M2 @ M2)          # 4 C_cop = -(2 M_12)^2
    ok &= trace_counts(ccop4, [0, 4, 16]) == [24, 32, 8]
    ok &= (ccop4 + lor4 == 16 * I64).all()
    return ok


@claim("d5-6c", "exchange and spin: tau_K commutes with the diagonal interior Lorentz action, anticommutes with M_12, "
                "and Tr(tau_K P) = 0 on every Lorentz-Casimir eigenspace, so the exchange-symmetric and "
                "-antisymmetric halves carry identical content {0 (4), 3 (16), 4 (12)}")
def _():
    ok = all((TK.astype(object).dot(x) == x.dot(TK.astype(object))).all() for x in SDm + TDm)
    gam = lambda k, mu: E[gi(k, 1)] if mu == 0 else E[gi(k, mu + 3)]
    M2 = sum(e * gam(1, mu) @ gam(2, mu) for mu, e in [(0, 1), (1, -1), (2, -1), (3, -1)])
    ok &= (TK @ M2 == -M2 @ TK).all()
    lor4 = int4(-sum(s.dot(s) for s in SDm) + sum(t.dot(t) for t in TDm))
    vals = [0, 12, 16]
    for v in vals:
        num = np.eye(64, dtype=object)
        den = 1
        for w in vals:
            if w != v:
                num = num.dot((lor4 - w * I64).astype(object))
                den *= (v - w)
        ok &= np.trace(TK.astype(object).dot(num)) == 0
    return ok


# ============================================================================================ claims d5-7: photon vacuum
@claim("d5-7a", "Proposition 26.4 at n=1: the commutant of l_int and Q = (1/2) e_2 e_3 in M_8(R) has dimension 4 and "
                "symmetric part span{1, V} (V = e_1...e_6); the rank-four projectors in it are (1 +- V)/2, and V "
                "commutes with every bivector (tangent space 0: rigid)")
def _():
    gens = [e8[5] @ e8[6], e8[6] @ e8[4], e8[4] @ e8[5]] + [e8[1] @ e8[a + 3] for a in (1, 2, 3)] + [e8[2] @ e8[3]]
    gens_s = [sp.Matrix(g.tolist()) for g in gens]
    basis = algebra_basis(8, gens_s)
    Vm = sp.Matrix(OM.tolist())
    sym = algebra_basis(8, gens_s, sym=True)
    ok = len(basis) == 4 and len(sym) == 2 and same_span(sym, [sp.eye(8), Vm])
    a, b = sp.symbols('a b')
    P = a * sp.eye(8) + b * Vm
    sols = sp.solve(list((P * P - P).applyfunc(sp.expand)) + [P.trace() - 4], [a, b], dict=True)
    ok &= sorted([(s[a], s[b]) for s in sols]) == [(H, -H), (H, H)]
    ok &= all((comm(e8[i] @ e8[j], OM) == 0).all() for i in range(1, 7) for j in range(i + 1, 7))
    return ok


A_, B_, C_, D_, E_ = E[gi(1, 2)], E[gi(1, 3)], E[gi(2, 1)], E[gi(2, 2)], E[gi(2, 3)]
WI = prod([E[A] for A in (2, 3, 7, 8, 9, 10, 11, 12)])        # internal volume = J at n = 2
GAMMAS = [E[1], E[4], E[5], E[6]]


@claim("d5-7b", "Theorem 26.6 in the realisation of Cl(6,6): H = abcd has H^2 = 1, H^T = H, Tr H = 0; [Q, Pi_0] = 0 for "
                "Q = ab/2; Pi_0 commutes with the four interior gammas and with the internal volume; for B = be/2, "
                "pi = [B, Pi_0] = -acde/2, symmetric, Tr pi^2 = 16, [Q, pi] = bcde/2")
def _():
    Hm = A_ @ B_ @ C_ @ D_
    ok = (Hm @ Hm == I64).all() and (Hm.T == Hm).all() and np.trace(Hm) == 0
    Q2 = A_ @ B_                       # 2Q
    P2 = I64 + Hm                      # 2 Pi_0
    ok &= (comm(Q2, P2) == 0).all() and all((comm(g, P2) == 0).all() for g in GAMMAS) and (comm(WI, P2) == 0).all()
    B2 = B_ @ E_                       # 2B
    pi4 = comm(B2, P2)                 # 4 pi
    ok &= (pi4 == -2 * (A_ @ C_ @ D_ @ E_)).all()      # pi = -acde/2  <=> 4 pi = -2 acde
    ok &= (pi4.T == pi4).all() and np.trace(pi4 @ pi4) == 16 * 16
    ok &= (comm(Q2, pi4) == 4 * (B_ @ C_ @ D_ @ E_)).all()   # [Q,pi] = bcde/2 <=> [2Q, 4pi] = 4 bcde
    return ok


@claim("d5-7c", "Theorem 26.7: Z_i = I_5 v_i are symmetric of grade four with {Z_i, Z_j} = 2 delta_ij; Pi(n) for the "
                "rational unit n = (2,3,6,0,0)/7 is a rank-32 projector commuting with the interior gammas; tangents "
                "T_i = [v_i e/2, Pi_0] = I_5 v_i / 2, Tr(T_i T_j) = 16 delta_ij, [Q,T_a] = -T_b, [Q,T_b] = T_a, "
                "[Q,T_c] = [Q,T_d] = 0")
def _():
    vs = [A_, B_, C_, D_, E_]
    I5 = A_ @ B_ @ C_ @ D_ @ E_
    Z = [I5 @ v for v in vs]
    ok = all((z.T == z).all() for z in Z)
    ok &= all((anti(Z[i], Z[j]) == (2 * I64 if i == j else 0 * I64)).all() for i in range(5) for j in range(5))
    nvec = [2, 3, 6, 0, 0]
    P14 = 7 * I64 + sum(c * z for c, z in zip(nvec, Z))     # 14 Pi(n)
    ok &= (P14 @ P14 == 14 * P14).all() and np.trace(P14) == 14 * 32 and (P14.T == P14).all()
    ok &= all((comm(g, P14) == 0).all() for g in GAMMAS)
    P2 = I64 + I5 @ E_                                      # 2 Pi(e) = 1 + I5 e
    Hm = A_ @ B_ @ C_ @ D_
    ok &= (I5 @ E_ == Hm).all()                             # I_5 e = abcd e^2 = H, so Pi(e) = Pi_0
    T4 = [comm(v @ E_, P2) for v in vs[:4]]                 # 4 T_i
    ok &= all((T4[i] == 2 * I5 @ vs[i]).all() for i in range(4))
    ok &= all(np.trace(T4[i] @ T4[j]) == (16 * 16 if i == j else 0) for i in range(4) for j in range(4))
    Q2 = A_ @ B_
    ok &= (comm(Q2, T4[0]) == -2 * T4[1]).all() and (comm(Q2, T4[1]) == 2 * T4[0]).all()
    ok &= (comm(Q2, T4[2]) == 0).all() and (comm(Q2, T4[3]) == 0).all()
    return ok


@claim("d5-7d", "the second-rung vacuum straddles the two copies: the grade-four blades in the five positive internal "
                "generators that commute with Q = ab/2 are exactly abcd, abce, abde, each using a, b of copy 1 and two "
                "directions of copy 2")
def _():
    names = {'a': gi(1, 2), 'b': gi(1, 3), 'c': gi(2, 1), 'd': gi(2, 2), 'e': gi(2, 3)}
    good = []
    for combo in itertools.combinations('abcde', 4):
        m = sum(1 << (names[x] - 1) for x in combo)
        qm = (1 << (names['a'] - 1)) | (1 << (names['b'] - 1))
        s1, _ = gp(m, qm)
        s2, _ = gp(qm, m)
        if s1 == s2:
            good.append(''.join(combo))
    return sorted(good) == ['abcd', 'abce', 'abde']


# ============================================================================================ claims d5-8: the imaginary unit
@claim("d5-8", "imaginary unit at n=2 (Theorem 4.1): J = product of the 8 internal generators = (e_2 e_3) (x) omega, "
                "J^2 = -1; the joint commutant of l_int^(1) and the 28 internal bivectors in M_64(R) is spanned by "
                "the blades 1, Omega, J, Omega J; Omega^2 = -1, (Omega J)^2 = 1, [Omega, J] = 0; the solutions of "
                "x^2 = -1 there are exactly +-Omega, +-J; J commutes and Omega anticommutes with the four gammas")
def _():
    ok = (WI == np.kron(e8[2] @ e8[3], OM)).all() and (WI @ WI == -I64).all()
    internal = [2, 3, 7, 8, 9, 10, 11, 12]
    gens = [(1 << (gi(1, 5) - 1)) | (1 << (gi(1, 6) - 1)), (1 << (gi(1, 6) - 1)) | (1 << (gi(1, 4) - 1)),
            (1 << (gi(1, 4) - 1)) | (1 << (gi(1, 5) - 1))] + \
           [(1 << (gi(1, 1) - 1)) | (1 << (gi(1, a + 3) - 1)) for a in (1, 2, 3)] + \
           [(1 << (s - 1)) | (1 << (t - 1)) for s, t in itertools.combinations(internal, 2)]
    ok &= len(gens) == 6 + 28
    survivors = []
    for m in range(1 << 12):
        if all(gp(m, g)[0] == gp(g, m)[0] for g in gens):
            survivors.append(m)
    om_mask = sum(1 << (k - 1) for k in (1, 4, 5, 6))
    j_mask = sum(1 << (k - 1) for k in internal)
    ok &= sorted(survivors) == sorted([0, om_mask, j_mask, om_mask | j_mask])
    Om = prod([E[1], E[4], E[5], E[6]])
    ok &= (Om @ Om == -I64).all() and (comm(Om, WI) == 0).all() and ((Om @ WI) @ (Om @ WI) == I64).all()
    x0, x1, x2, x3 = sp.symbols('x0:4')
    # (x0 + x1 Om + x2 J + x3 OmJ)^2 in the commutative algebra with Om^2 = J^2 = -1, (OmJ)^2 = 1
    c0 = x0**2 - x1**2 - x2**2 + x3**2
    c1 = 2 * x0 * x1 - 2 * x2 * x3
    c2 = 2 * x0 * x2 - 2 * x1 * x3
    c3 = 2 * x0 * x3 + 2 * x1 * x2
    sols = sp.solve([c0 + 1, c1, c2, c3], [x0, x1, x2, x3], dict=True)
    real_sols = sorted(tuple(int(s[v]) for v in (x0, x1, x2, x3)) for s in sols if all(s[v].is_real for v in s))
    ok &= real_sols == sorted([(0, 1, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, -1, 0)])
    ok &= all((comm(WI, g) == 0).all() and (anti(Om, g) == 0).all() for g in GAMMAS)
    return ok


# ============================================================================================ claims d5-9: inter-copy b-field
k1, k2, bt, ms = sp.symbols('kappa1 kappa2 beta m')


def S_of(a, b, c):
    return kr(sp.diag(a, b), Gs) + c * kr(eps, I6)


def M_of(a, b, c, m=1):
    S = S_of(a, b, c)
    Sd = eta2.inv() * S.T * eta2
    M = sp.zeros(24, 24)
    M[:12, :12] = -S
    M[:12, 12:] = sp.eye(12) / m
    M[12:, 12:] = Sd
    return M


@claim("d5-9a", "the compact inter-copy field b = beta eps (x) 1_6 is, up to scale, the only element of so(6,6) cap "
                "so(12) commuting with G_2 and the diagonal rotations (restates Proposition 13.14); with "
                "S = K (x) G + b, K = diag(kappa1, kappa2): S^dagger = K (x) G - b, charpoly(S) = (lam^4 + (kappa1^2 + "
                "kappa2^2 + 2 beta^2) lam^2 + (kappa1 kappa2 - beta^2)^2)^3, i.e. spec S = +-i omega_+-, "
                "omega_+- = (kappa1+kappa2)/2 +- sqrt((kappa1-kappa2)^2/4 + beta^2)")
def _():
    S = S_of(k1, k2, bt)
    ok = (eta2.inv() * S.T * eta2 - (kr(sp.diag(k1, k2), Gs) - bt * kr(eps, I6))).is_zero_matrix
    cp = S.charpoly(lam).as_expr()
    ok &= sp.expand(cp - (lam**4 + (k1**2 + k2**2 + 2 * bt**2) * lam**2 + (k1 * k2 - bt**2)**2)**3) == 0
    wp = (k1 + k2) / 2 + sp.sqrt((k1 - k2)**2 / 4 + bt**2)
    wm = (k1 + k2) / 2 - sp.sqrt((k1 - k2)**2 / 4 + bt**2)
    ok &= sp.simplify(sp.expand((lam**2 + wp**2) * (lam**2 + wm**2)) -
                      (lam**4 + (k1**2 + k2**2 + 2 * bt**2) * lam**2 + (k1 * k2 - bt**2)**2)) == 0
    so2 = algebra_basis(12, rot_diag() + [G2], eta=eta2, antisym=True)
    ok &= len(so2) == 1 and same_span(so2, [kr(eps, I6)])
    return ok


@claim("d5-9b", "removability: [K (x) G, eps (x) 1] = 0 iff kappa1 = kappa2; dim Cent_so(6,6)(K (x) G) = 30, 12, 21 "
                "for kappa1 = kappa2 != 0, kappa1 != kappa2 both nonzero (sample 1, 3), exactly one zero (sample 0, 5)")
def _():
    Cm = kr(sp.diag(k1, k2), Gs) * kr(eps, I6) - kr(eps, I6) * kr(sp.diag(k1, k2), Gs)
    ok = Cm.subs(k2, k1).is_zero_matrix and not Cm.subs({k1: 1, k2: 3}).is_zero_matrix
    ok &= (Cm - (k2 - k1) * kr(sgx, Gs)).applyfunc(sp.expand).is_zero_matrix   # = (kappa2 - kappa1) sx (x) G
    dims = [len(algebra_basis(12, [kr(sp.diag(a, b), Gs)], eta=eta2)) for (a, b) in [(2, 2), (1, 3), (0, 5)]]
    return ok and dims == [30, 12, 21]


@claim("d5-9c", "the phase-space generator M = [[-S, 1/m], [0, S^dagger]] on (frame field X, momentum field pi) obeys "
                "(M^2 + omega_+^2)(M^2 + omega_-^2) = 0 identically in kappa1, kappa2, beta, m")
def _():
    M = M_of(k1, k2, bt, ms)
    M2 = (M * M).applyfunc(sp.expand)
    M4 = (M2 * M2).applyfunc(sp.expand)
    q = M4 + (k1**2 + k2**2 + 2 * bt**2) * M2 + (k1 * k2 - bt**2)**2 * sp.eye(24)
    return q.applyfunc(sp.expand).is_zero_matrix


@claim("d5-9d", "boundedness (exact semisimplicity, rational data, m = 1): bounded at (kappa1,kappa2,beta) = (1,3,2/5), "
                "(0,5,1/2), (2,2,1/3), (1,1,1), (2,2,0); unbounded (Jordan blocks) at (0,0,1/2), (1,4,2), (1,9,3), "
                "(0,3,0); at kappa1 kappa2 = beta^2 the kernel of S is eta_2-totally isotropic iff kappa1 = kappa2")
def _():
    R = sp.Rational
    good = [(1, 3, R(2, 5)), (0, 5, R(1, 2)), (2, 2, R(1, 3)), (1, 1, 1), (2, 2, 0)]
    bad = [(0, 0, R(1, 2)), (1, 4, 2), (1, 9, 3), (0, 3, 0)]
    ok = True
    for pt in good:
        ss, P = is_semisimple(M_of(*pt))
        roots = sp.Poly(sp.sqf_part(P.as_expr()), lam).nroots(n=30, maxsteps=200)
        ok &= ss and all(abs(sp.re(r)) < 1e-25 for r in roots)
    for pt in bad:
        ss, _ = is_semisimple(M_of(*pt))
        ok &= not ss
    for pt, iso in [((1, 1, 1), True), ((1, 4, 2), False), ((1, 9, 3), False), ((4, 9, 6), False)]:
        S = S_of(*pt)
        ker = nullspace_q(S)
        Kb = sp.Matrix.hstack(*ker)
        ok &= len(ker) == 6 and ((Kb.T * eta2 * Kb).is_zero_matrix == iso)
    return ok


@claim("d5-9e", "seesaw (inherited two-level mixing): at kappa1 = 0, kappa2 = 5, beta = 1/2 the slow line is "
                "omega_- = -0.0495098 (beta^2/kappa2 = 0.05, beta^2/kappa2 - beta^4/kappa2^3 = 0.0495); the free "
                "copy is bound, sup_{0<=t<=600} ||e^{Mt}|| = 19.86, of the order of 1/|omega_-| = 20.20 (numerical, "
                "601 sample times)")
def _():
    wm = 2.5 - math.sqrt(6.25 + 0.25)
    ok = abs(wm + 0.0495098) < 1e-6 and abs(0.25 / 5 - 0.05) < 1e-15 and abs(0.25 / 5 - 0.0625 / 125 - 0.0495) < 1e-12
    Mn = np.array(M_of(0, 5, sp.Rational(1, 2)).tolist(), dtype=float)
    mx = max(np.linalg.norm(sla.expm(Mn * t), 2) for t in np.linspace(0, 600, 601))
    ok &= abs(mx - 19.86) < 0.05 and abs(1 / abs(wm) - 20.198) < 1e-3
    # without coupling the free copy drifts: ||e^{Mt}|| grows linearly
    Mn0 = np.array(M_of(0, 5, 0).tolist(), dtype=float)
    ok &= np.linalg.norm(sla.expm(Mn0 * 600), 2) > 500
    return ok


@claim("d5-9g", "rates against confinement at kappa1 = 0 < kappa2: omega_+ omega_- = -beta^2 and omega_+ + omega_- = kappa2 "
                "(exact); at (kappa1,kappa2,beta) = (0,1/100,1) M is semisimple (bounded), the lines are 1.00501 and "
                "-0.99501 (about +-beta), and sup ||e^{Mt}|| over a beat period 2 pi/kappa2 is 200.0 = 2/(m kappa2) "
                "(numerical, 3000 sample times): b sets the rates, kappa2 the size of the orbit")
def _():
    wp = (k1 + k2) / 2 + sp.sqrt((k1 - k2)**2 / 4 + bt**2)
    wm = (k1 + k2) / 2 - sp.sqrt((k1 - k2)**2 / 4 + bt**2)
    ok = sp.simplify((wp * wm).subs(k1, 0) + bt**2) == 0 and sp.simplify((wp + wm).subs(k1, 0) - k2) == 0
    pt = (0, sp.Rational(1, 100), 1)
    ss, _ = is_semisimple(M_of(*pt))
    ok &= ss
    wpn = 0.005 + math.sqrt(0.005**2 + 1)
    wmn = 0.005 - math.sqrt(0.005**2 + 1)
    ok &= abs(wpn - 1.00501) < 1e-5 and abs(wmn + 0.99501) < 1e-5
    Mn = np.array(M_of(*pt).tolist(), dtype=float)
    T = 1.2 * 2 * math.pi / 0.01
    mx = max(np.linalg.norm(sla.expm(Mn * t), 2) for t in np.linspace(0, T, 3000))
    ok &= abs(mx - 200.0) < 0.5
    return ok


@claim("d5-9f", "duplicated momentum fields in the closure (restating Proposition 17.18): for pi^(2) = G pi^(1) "
                "(p^(2) = -G p^(1)) w_2 = -w_1; for the duplication pi -> (alpha pi, beta pi), w_1 + w_2 = "
                "(alpha^2 + beta^2) w(pi), so R_eff = |a + i b| |q_2(pi)| / m for identical inertias, and the "
                "relative-duality quarter turn puts two massive copies (E0^(k) = E0(pi)/2) on the balance surface")
def _():
    pv = sp.Matrix(sp.symbols('p1:7', real=True))
    w = lambda p: sp.expand((p.T * eta.inv() * p)[0] - sp.I * (p.T * Gs * eta.inv() * p)[0])  # eq. (30)
    pi = eta.inv() * pv
    zq = sp.Matrix([pi[i] + sp.I * pi[i + 3] for i in range(3)])
    ok = sp.expand(w(pv) - qf(zq)) == 0                    # w(p) = q(z(pi))
    p2 = -Gs * pv
    ok &= sp.expand(w(p2) + w(pv)) == 0
    ok &= ((eta.inv() * p2) - Gs * pi).is_zero_matrix      # p^(2) = -G p^(1)  <=>  pi^(2) = G pi^(1)
    # duplication in the momentum field: pi -> (alpha pi, beta pi) with alpha, beta acting through 1, G
    a1, a2, b1, b2 = sp.symbols('a1 a2 b1 b2', real=True)
    def wpi(pin):
        z = sp.Matrix([pin[i] + sp.I * pin[i + 3] for i in range(3)])
        return qf(z)
    s = wpi((a1 * I6 + a2 * Gs) * pi) + wpi((b1 * I6 + b2 * Gs) * pi)
    ok &= sp.expand(s - ((a1 + sp.I * a2)**2 + (b1 + sp.I * b2)**2) * qf(zq)) == 0
    # quarter turn at equal split: alpha = 1/sqrt2, beta = i/sqrt2 -> sum zero, each |w_k| = |q|/2
    r2 = sp.sqrt(2)
    s0 = wpi(pi / r2) + wpi(Gs * pi / r2)
    ok &= sp.expand(s0) == 0 and sp.expand(wpi(pi / r2) - qf(zq) / 2) == 0
    return ok


# ============================================================================================ claims d5-10: comparison
@claim("d5-10", "n=1 against n=2: commutant of the diagonal l_G in so(3n,3n) 0 -> 2 (the copy algebra); Cent(G_n) 6 -> 30; "
                "Lorentz-compatible energy symmetry so(3) -> so(6) (3 -> 15); eta-compatible copy symmetry 0 -> 1; "
                "internal centraliser of the spectator action 1 -> 28 (Theorem 6.1)")
def _():
    c1 = algebra_basis(6, LV + KV, eta=eta)
    c2 = algebra_basis(12, [diag2(X) for X in LV + KV], eta=eta2)
    g1 = algebra_basis(6, [Gs], eta=eta)
    g2 = algebra_basis(12, [G2], eta=eta2)
    e1 = algebra_basis(6, [Gs], eta=eta, antisym=True)
    e2 = algebra_basis(12, [G2], eta=eta2, antisym=True)
    # at n=1 the copy-compatible symmetry is trivially absent; at n=2 so(2)
    s2 = algebra_basis(12, rot_diag() + [G2], eta=eta2, antisym=True)
    # spectator internal centraliser at n=1 (blade algebra restricted to copy 1 bivectors) and n=2
    biv1 = [bl(i, j) for i in range(1, 7) for j in range(i + 1, 7)]
    gint1 = len(mv_commutant(S1 + T1, biv1))
    gint2 = len(mv_commutant(S1 + T1, BIVS))
    return (len(c1), len(c2), len(g1), len(g2), len(e1), len(e2), len(s2), gint1, gint2) == \
        (0, 2, 6, 30, 3, 15, 1, 1, 28)


# ============================================================================================ summary
print(f"# d5_duplication_math.py: {time.time() - T0:.1f} s")
if FAILS:
    print("FAILED:", ", ".join(FAILS))
    sys.exit(1)
sys.exit(0)
