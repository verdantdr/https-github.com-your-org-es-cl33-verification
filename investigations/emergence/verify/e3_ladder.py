#!/usr/bin/env python3
"""Track e3 -- Up the ladder: the rung Cl(3n,3n).

    python3 investigations/emergence/verify/e3_ladder.py

Prints one 'PASS e3-<id>: ...' line per verified claim of sections/e3_ladder.tex and exits 0 only if every check
passes.  Runnable from any directory; numpy, scipy and sympy only.

Conventions (tex/s02_arena.tex eq. (1)-(2), tex/sA_realisation.tex):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1); on R^8: e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1,
  e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep; omega = e1...e6 (diagonal).
  Tower: E_(k,a) = omega^{(x)(k-1)} (x) e_a (x) 1^{(x)(n-k)} on S_n = R^(8^n).  Ungraded copy k: u_(k,a) = 1..(x) e_a (x)..1.
  Interior Lorentz algebra lint on <e1,e4,e5,e6>; lG = commutant of G in so(3,3) (G e_i = e_{i+3}, G e_{i+3} = -e_i):
  paired rotations e_ie_j - e_{i+3}e_{j+3} and paired boosts e_ie_{j+3} - e_je_{i+3}.
  Copy rotations C^{kl} = (1/2) sum_a eta_aa E_(k,a)E_(l,a) (all six directions); interior copy rotations M_kl (four).
  Fock modes a+-_{k,i} = (E_(k,i) +- E_(k,i+3))/2, N = sum a+ a-.
  Koszul swap tau_K = D P on R^8 (x) R^8 (RE Prop. 7.3(iii)); tau_k acts on the factors k, k+1.

Exactness.  Generators and blades are signed permutation matrices.  Algebraic identities are checked in integer
arithmetic: scipy.sparse int64 matrices, (permutation, sign) arrays, and dense float64 arrays whose entries are integers
of modulus < 2^40 (asserted), for which float64 arithmetic is exact.  Vector-level Lie algebra dimensions are exact
ranks over Q (sympy DomainMatrix).  Floating point proper is used only for spectra of symmetric operators (tolerance
1e-6, stated with each check), for matrix exponentials (tolerance 1e-9) and for one nullity (gap stated).
Runtime: about a minute.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # dense BLAS on small matrices: threads only add contention
import itertools
import math
import sys
import time
from collections import Counter
from fractions import Fraction

import numpy as np
import scipy.linalg as sla
import scipy.sparse as ss
from sympy import QQ, ZZ
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
        t0 = time.time()
        try:
            ok = bool(fn())
        except Exception as exc:  # reported as a failure
            import traceback
            traceback.print_exc()
            print(f"   exception in {cid}: {exc!r}")
            ok = False
        report(cid, ok, text)
        if os.environ.get("E3_TIMING"):
            print(f"   [{cid}: {time.time() - t0:.1f}s]")
        return fn
    return deco


# ============================================================================== one constituent
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
om = e[1] @ e[2] @ e[3] @ e[4] @ e[5] @ e[6]
INT = [1, 4, 5, 6]
SUR = [2, 3]


# ============================================================================== sparse tower
def S(M):
    return ss.csr_matrix(np.asarray(M, dtype=np.int64))


def skron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = ss.kron(out, m, format='csr')
    return out


_TOWER = {}


def tower(n):
    """graded E[(k,a)] and ungraded U[(k,a)] at rung n, sparse int64"""
    if n not in _TOWER:
        E, U = {}, {}
        for k in range(1, n + 1):
            for a in range(1, 7):
                E[(k, a)] = skron(*([S(om)] * (k - 1) + [S(e[a])] + [S(I8)] * (n - k)))
                U[(k, a)] = skron(*([S(I8)] * (k - 1) + [S(e[a])] + [S(I8)] * (n - k)))
        _TOWER[n] = (E, U)
    return _TOWER[n]


def Id(n):
    return ss.identity(8 ** n, dtype=np.int64, format='csr')


def seq(A, B):
    """exact equality of sparse integer matrices"""
    D = (A - B)
    return D.count_nonzero() == 0 if ss.issparse(D) else not np.any(D)


def is_zero(A):
    return A.count_nonzero() == 0 if ss.issparse(A) else not np.any(A)


def dense_f(A):
    """dense float64 copy of an integer matrix; exact while entries stay below 2^40"""
    M = A.toarray() if ss.issparse(A) else np.asarray(A)
    assert np.abs(M).max() < 2 ** 40
    return M.astype(np.float64)


def exact_f(M):
    """assert that a float64 array produced from integer arithmetic is still integral and in the exact range"""
    assert np.abs(M).max() < 2 ** 40 and np.array_equal(M, np.round(M))
    return M


def spec_counts(M, tol=1e-6):
    ev = np.linalg.eigvalsh((M + M.T) / 2)
    vals = []
    for x in np.sort(ev):
        if vals and abs(x - vals[-1][0]) < tol:
            vals[-1][1] += 1
        else:
            vals.append([x, 1])
    return [(round(v, 6), c) for v, c in vals]


def number_diag(n):
    E, _ = tower(n)
    Nd = np.zeros(8 ** n, dtype=np.int64)
    for k in range(1, n + 1):
        for i in (1, 2, 3):
            P = E[(k, i)] @ E[(k, i + 3)]
            assert seq(P, ss.diags(P.diagonal(), dtype=np.int64).tocsr())
            Nd += (1 - P.diagonal()) // 2
    return Nd


def vol(n):
    """the volume element E_1 ... E_6n of Cl(3n,3n)"""
    E, _ = tower(n)
    M = Id(n)
    for k in range(1, n + 1):
        for a in range(1, 7):
            M = M @ E[(k, a)]
    return M


# ============================================================================== signed permutations (rung 5)
class SP:
    """signed permutation matrix: M e_j = s[j] e_{p[j]}"""

    def __init__(self, p, s):
        self.p, self.s = np.asarray(p, dtype=np.int64), np.asarray(s, dtype=np.int64)

    @staticmethod
    def from_dense(M):
        M = np.asarray(M)
        p = np.argmax(np.abs(M), axis=0)
        s = M[p, np.arange(M.shape[0])]
        assert np.all(np.abs(s) == 1) and np.count_nonzero(M) == M.shape[0]
        return SP(p, s)

    def __matmul__(self, o):
        return SP(self.p[o.p], o.s * self.s[o.p])

    def kron(self, o):
        nb = len(o.p)
        return SP((self.p[:, None] * nb + o.p[None, :]).ravel(), (self.s[:, None] * o.s[None, :]).ravel())

    def __eq__(self, o):
        return np.array_equal(self.p, o.p) and np.array_equal(self.s, o.s)

    def neg(self):
        return SP(self.p, -self.s)

    def trace(self):
        fix = self.p == np.arange(len(self.p))
        return int(self.s[fix].sum())

    @staticmethod
    def ident(d):
        return SP(np.arange(d), np.ones(d, dtype=np.int64))


sp_e = [None] + [SP.from_dense(e[a]) for a in range(1, 7)]
sp_om = SP.from_dense(om)
sp_I8 = SP.ident(8)


def tower_sp(n):
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


def factor_perm_sp(n, sigma):
    """plain permutation of the tensor factors of (R^8)^{(x)n}: the factor in slot i moves to slot sigma[i]"""
    d = 8 ** n
    idx = np.indices((8,) * n).reshape(n, -1)
    new = np.empty_like(idx)
    for i in range(n):
        new[sigma[i]] = idx[i]
    src = np.ravel_multi_index(tuple(idx), (8,) * n)
    dst = np.ravel_multi_index(tuple(new), (8,) * n)
    p = np.empty(d, dtype=np.int64)
    p[src] = dst
    return SP(p, np.ones(d, dtype=np.int64))


def sp_to_sparse(A):
    d = len(A.p)
    return ss.csr_matrix((A.s, (A.p, np.arange(d))), shape=(d, d), dtype=np.int64)


# ============================================================================== vector level (exact over Q)
def ad_vec(a, b, n):
    """matrix of ad(E_a E_b) on V_n (basis E_1..E_6n, 0-based): e_c -> 2 eta_bc e_a - 2 eta_ac e_b"""
    et = ETA[1:] * n
    M = np.zeros((6 * n, 6 * n), dtype=np.int64)
    M[a, b] += 2 * et[b]
    M[b, a] -= 2 * et[a]
    return M


def so_basis(n):
    pairs = list(itertools.combinations(range(6 * n), 2))
    return pairs, [ad_vec(a, b, n) for a, b in pairs]


def commutant_dim(gens, n, extra=None):
    """dim of {X in so(3n,3n) : [X, g] = 0 for g in gens} (and, if given, X g = g X for the matrices in extra), exact"""
    pairs, B = so_basis(n)
    rows = []
    for g in list(gens) + list(extra or []):
        rows.append(np.array([(X @ g - g @ X).ravel() for X in B]).T)
    A = np.vstack(rows)
    A = A[np.any(A != 0, axis=1)]
    AtA = A.T @ A
    M = DomainMatrix([[ZZ(int(x)) for x in r] for r in AtA.tolist()], AtA.shape, ZZ)
    return len(pairs) - M.rank()


def biv_vec(n, terms):
    """sum of c * ad(E_(k,a) E_(k,b)) for terms (c, k, a, b) (1-based copies and directions)"""
    M = np.zeros((6 * n, 6 * n), dtype=np.int64)
    for c, k, a, b in terms:
        M += c * ad_vec(6 * (k - 1) + a - 1, 6 * (k - 1) + b - 1, n)
    return M


def diag_vec(n, a, b, c=1):
    return biv_vec(n, [(c, k, a, b) for k in range(1, n + 1)])


def G_vec(n):
    M = np.zeros((6 * n, 6 * n), dtype=np.int64)
    for k in range(n):
        for i in range(3):
            M[6 * k + i + 3, 6 * k + i] = 1
            M[6 * k + i, 6 * k + i + 3] = -1
    return M


def copy_vec(n, k, l, weighted=True):
    """ad of sum_a w_a E_(k,a) E_(l,a), w_a = eta_aa (weighted) or 1"""
    M = np.zeros((6 * n, 6 * n), dtype=np.int64)
    for a in range(1, 7):
        w = ETA[a] if weighted else 1
        M += w * ad_vec(6 * (k - 1) + a - 1, 6 * (l - 1) + a - 1, n)
    return M


def so33_diag_vec(n):
    return [diag_vec(n, a, b) for a, b in itertools.combinations(range(1, 7), 2)]


def lint_diag_vec(n):
    return [diag_vec(n, a, b) for a, b in itertools.combinations(INT, 2)]


def lG_diag_vec(n):
    rots = [biv_vec(n, [(1, k, i, j) for k in range(1, n + 1)] + [(-1, k, i + 3, j + 3) for k in range(1, n + 1)])
            for i, j in ((1, 2), (1, 3), (2, 3))]
    boosts = [biv_vec(n, [(1, k, i, j + 3) for k in range(1, n + 1)] + [(-1, k, j, i + 3) for k in range(1, n + 1)])
              for i, j in ((1, 2), (1, 3), (2, 3))]
    return rots, boosts


# ============================================================================== section 1: the rung and the step
@claim("e3-1a", "Cl(3n,3n) = M_(8^n)(R) in the tower (n = 1..4, exhaustive over all 64^n blades via the factorisation "
       "e_I = (x)_j e_(I_j) omega^(|I_(j+1)|+...+|I_n|)): every blade except 1 is traceless, so the 64^n blades are "
       "trace-orthogonal and independent; the volume element E_1...E_6n equals omega^(x)n, and (-1)^N = (-1)^n omega^(x)n")
def _():
    ok = True
    # one-constituent table: trace of e_J omega^p for the 64 blades J and p in {0,1}
    tr = np.zeros((64, 2), dtype=np.int64)
    for J in range(64):
        M = I8.copy()
        for a in range(6):
            if J >> a & 1:
                M = M @ e[a + 1]
        tr[J, 0] = np.trace(M)
        tr[J, 1] = np.trace(M @ om)
    pop = np.array([bin(J).count('1') for J in range(64)])
    for n in (1, 2, 3, 4):
        # all blades as n-tuples (J_1..J_n); vectorised over the 64^n tuples
        grids = np.indices((64,) * n).reshape(n, -1)
        total = np.ones(grids.shape[1], dtype=np.int64)
        tail = np.zeros(grids.shape[1], dtype=np.int64)
        for j in range(n - 1, -1, -1):
            total *= tr[grids[j], tail % 2]
            tail += pop[grids[j]]
        nz = np.nonzero(total)[0]
        ok &= len(nz) == 1 and nz[0] == 0 and total[0] == 8 ** n
        # cross-check the factorisation against the sparse realisation on a sample of blades at n = 2
        if n == 2:
            E, _ = tower(2)
            gens = [E[(k, a)] for k in (1, 2) for a in range(1, 7)]
            rng = np.random.default_rng(3)
            for m in rng.integers(0, 4096, size=60).tolist() + [4095, 0]:
                M = Id(2)
                for i in range(12):
                    if m >> i & 1:
                        M = M @ gens[i]
                ok &= int(M.diagonal().sum()) == int(tr[m & 63, bin(m >> 6).count('1') % 2] * tr[m >> 6, 0])
                ok &= int(total[64 * (m & 63) + (m >> 6)]) == int(M.diagonal().sum())
        Vn = vol(n)
        ok &= seq(Vn, skron(*([S(om)] * n)))
        Nd = number_diag(n)
        ok &= seq(ss.diags((-1) ** Nd, dtype=np.int64).tocsr(), (-1) ** n * Vn)
    return ok


@claim("e3-1b", "the step n -> n+1 (n = 1..4): E^(n+1)_(k,a) = E^(n)_(k,a) (x) 1 for k <= n and E_(n+1,a) = vol_n (x) e_a, "
       "vol_n = omega^(x)n; so S_(n+1) restricted to Cl(3n,3n) is 8 copies of S_n; the ungraded new copy is "
       "1 (x) e_a = vol_n E_(n+1,a) (the dressing omega^(1)...omega^(n)); among the 64^(n+1) blades exactly 64, each "
       "+-1(x)e_J, commute with all 6n lower generators (n = 1, 2 exhaustive), i.e. the commutant is 1(x)M_8(R)")
def _():
    ok = True
    for n in (1, 2, 3, 4):
        E0, _ = tower(n)
        E1, U1 = tower(n + 1)
        for k in range(1, n + 1):
            for a in range(1, 7):
                ok &= seq(E1[(k, a)], skron(E0[(k, a)], S(I8)))
        Vn = vol(n)
        for a in range(1, 7):
            ok &= seq(E1[(n + 1, a)], skron(Vn, S(e[a])))
            ok &= seq(U1[(n + 1, a)], skron(Id(n), S(e[a])))
            ok &= seq(U1[(n + 1, a)], skron(Vn, S(I8)) @ E1[(n + 1, a)])
    # blade commutant: blade rule (3) of ES17 on bitmasks of 6(n+1) generators
    for n in (1, 2):
        nb = 6 * (n + 1)
        masks = np.arange(1 << nb, dtype=np.int64)
        pc = np.zeros_like(masks)
        for i in range(nb):
            pc += (masks >> i) & 1
        good = np.ones(len(masks), dtype=bool)
        for g in range(6 * n):
            inter = (masks >> g) & 1
            good &= ((pc * 1 - inter) % 2 == 0)
        cm = np.nonzero(good)[0]
        ok &= len(cm) == 64
        # each such blade is +-1 (x) e_J: check by building it at n = 1 (64 x 64)
        if n == 1:
            E1, _ = tower(2)
            gens = [E1[(k, a)] for k in (1, 2) for a in range(1, 7)]
            targets = []
            for J in range(64):
                M = I8.copy()
                for a in range(6):
                    if J >> a & 1:
                        M = M @ e[a + 1]
                targets.append(skron(S(I8), S(M)))
            for m in cm.tolist():
                B = Id(2)
                for i in range(12):
                    if m >> i & 1:
                        B = B @ gens[i]
                ok &= sum(1 for T in targets if seq(B, T) or seq(B, -T)) == 1
    return ok


@claim("e3-1c", "Fock modes up the ladder (n = 1..4; spectrum also at n = 5): N_(n+1) = N_n (x) 1 + 1 (x) N_1, so each step adds "
       "three modes; spec N = {0,...,3n} with multiplicities binom(3n,j); [N, E_(k,i)] = E_(k,i+3), [N, E_(k,i+3)] = E_(k,i)")
def _():
    ok = True
    Eg5, _ = tower_sp(5)
    N5 = np.zeros(8 ** 5, dtype=np.int64)
    for k in range(1, 6):
        for i in (1, 2, 3):
            P = Eg5[(k, i)] @ Eg5[(k, i + 3)]
            ok &= np.array_equal(P.p, np.arange(8 ** 5))       # pole operators are diagonal
            N5 += (1 - P.s) // 2
    cnt5 = Counter(N5.tolist())
    ok &= all(cnt5[j] == math.comb(15, j) for j in range(16))
    for n in (1, 2, 3, 4):
        Nd = number_diag(n)
        cnt = Counter(Nd.tolist())
        ok &= all(cnt[j] == math.comb(3 * n, j) for j in range(3 * n + 1)) and sum(cnt.values()) == 8 ** n
        if n < 4:
            ok &= np.array_equal(number_diag(n + 1), np.add.outer(Nd, number_diag(1)).ravel())
        E, _ = tower(n)
        Nm = ss.diags(Nd, dtype=np.int64).tocsr()
        for k in range(1, n + 1):
            for i in (1, 2, 3):
                ok &= seq(Nm @ E[(k, i)] - E[(k, i)] @ Nm, E[(k, i + 3)])
                ok &= seq(Nm @ E[(k, i + 3)] - E[(k, i + 3)] @ Nm, E[(k, i)])
    return ok


# ============================================================================== section 2: mass up the ladder
@claim("e3-2a", "graded mass ladder (n = 1..5, signed permutations): the E_(k,1) are pairwise anticommuting traceless involutions, "
       "so H_g = m sum_k E_(k,1) has H_g^2 = n m^2 and spectrum +-sqrt(n) m, each of multiplicity 8^n/2; Z_g = J^(1) H_g "
       "(J^(1) = E_(1,2)E_(1,3) commutes with every E_(k,1)) has Z_g^2 = -n m^2: one clock of frequency sqrt(n) m "
       "(float exp(2 pi Z_g/sqrt(n)) = 1 at m = 1, n = 2, 3, tol 1e-9)")
def _():
    ok = True
    for n in (1, 2, 3, 4, 5):
        Eg, _ = tower_sp(n)
        Iden = SP.ident(8 ** n)
        J1 = Eg[(1, 2)] @ Eg[(1, 3)]
        for k in range(1, n + 1):
            ok &= (Eg[(k, 1)] @ Eg[(k, 1)]) == Iden and Eg[(k, 1)].trace() == 0
            ok &= (J1 @ Eg[(k, 1)]) == (Eg[(k, 1)] @ J1)
            for l in range(k + 1, n + 1):
                ok &= (Eg[(k, 1)] @ Eg[(l, 1)]) == (Eg[(l, 1)] @ Eg[(k, 1)]).neg()
        ok &= (J1 @ J1) == Iden.neg()
    for n in (2, 3):
        E, _ = tower(n)
        H = sum(E[(k, 1)] for k in range(1, n + 1))
        ok &= seq(H @ H, n * Id(n))
        ev = np.linalg.eigvalsh(dense_f(H))
        ok &= np.allclose(np.abs(ev), np.sqrt(n), atol=1e-9) and np.count_nonzero(ev > 0) == 8 ** n // 2
        Z = E[(1, 2)] @ E[(1, 3)] @ H
        ok &= seq(Z @ Z, -n * Id(n))
        ok &= np.allclose(sla.expm(2 * np.pi / np.sqrt(n) * dense_f(Z)), np.eye(8 ** n), atol=1e-9)
    return ok


@claim("e3-2b", "ungraded mass ladder: u_(k,a) = vol_1...vol_(k-1) E_(k,a) (vol_j = E_(j,1)...E_(j,6); n = 2..5, all k, a); the u_(k,1) "
       "commute, square to 1, and every product over a nonempty subset is traceless (n = 1..5), so H_u = m sum_k u_(k,1) has "
       "eigenvalues m(n-2j) with multiplicity binom(n,j) 4^n (float eigenvalue count at n = 1..3, tol 1e-9); the ungraded "
       "rest generators T_u^(k) = J^(k) u_(k,1) = 1..(x) T (x)..1 commute and square to -1: n clocks of frequency m")
def _():
    ok = True
    for n in (1, 2, 3, 4, 5):
        Eg, Eu = tower_sp(n)
        Iden = SP.ident(8 ** n)
        vols = {}
        for j in range(1, n + 1):
            M = Iden
            for a in range(1, 7):
                M = M @ Eg[(j, a)]
            vols[j] = M
        for k in range(1, n + 1):
            for a in range(1, 7):
                M = Iden
                for j in range(1, k):
                    M = M @ vols[j]
                ok &= (M @ Eg[(k, a)]) == Eu[(k, a)]
        for k in range(1, n + 1):
            for l in range(k + 1, n + 1):
                ok &= (Eu[(k, 1)] @ Eu[(l, 1)]) == (Eu[(l, 1)] @ Eu[(k, 1)])
        for r in range(1, n + 1):
            for sub in itertools.combinations(range(1, n + 1), r):
                M = Iden
                for k in sub:
                    M = M @ Eu[(k, 1)]
                ok &= M.trace() == 0
        T1 = SP.from_dense(e[1] @ e[2] @ e[3])
        for k in range(1, n + 1):
            Tk = Eu[(k, 2)] @ Eu[(k, 3)] @ Eu[(k, 1)]
            f = [sp_I8] * (k - 1) + [T1] + [sp_I8] * (n - k)
            g = f[0]
            for x in f[1:]:
                g = g.kron(x)
            ok &= Tk == g and (Tk @ Tk) == Iden.neg()
    for n in (1, 2, 3):
        _, U = tower(n)
        H = sum(U[(k, 1)] for k in range(1, n + 1))
        ev = np.linalg.eigvalsh(dense_f(H))
        for j in range(n + 1):
            ok &= np.count_nonzero(np.abs(ev - (n - 2 * j)) < 1e-9) == math.comb(n, j) * 4 ** n
    return ok


# ============================================================================== section 3: spin parity
def lint_rot(n):
    E, _ = tower(n)
    return [E[(k, 4)] @ E[(k, 5)] for k in range(1, n + 1)]          # 2R = sum of these


def lG_rot(n):
    E, _ = tower(n)
    return [E[(k, 1)] @ E[(k, 2)] for k in range(1, n + 1)] + [-(E[(k, 4)] @ E[(k, 5)]) for k in range(1, n + 1)]


def sp_commuting_minus_one(n, pairs, signs=None):
    """exact, signed permutations: the blades E_(k,a)E_(k,b) for (k,a,b) in pairs square to -1 and commute pairwise"""
    Eg, _ = tower_sp(n)
    Iden = SP.ident(8 ** n)
    bs = [Eg[(k, a)] @ Eg[(k, b)] for k, a, b in pairs]
    ok = all((b @ b) == Iden.neg() for b in bs)
    ok &= all((x @ y) == (y @ x) for x, y in itertools.combinations(bs, 2))
    return ok


@claim("e3-3a", "spin parity under the diagonal interior algebra (n = 1..5): the rotation R = (1/2) sum_k E_(k,4)E_(k,5) is a sum of n "
       "commuting terms (1/2)b_k with b_k^2 = -1, so exp(2 pi R) = (-1)^n; float check exp(2 pi R) = (-1)^n and spec(2R) = "
       "i x {integers = n mod 2} at n = 1..3 (tol 1e-9); the diagonal interior composite is a fermion exactly at odd n")
def _():
    ok = sp_commuting_minus_one(5, [(k, 4, 5) for k in range(1, 6)])
    for n in (1, 2, 3, 4):
        bs = lint_rot(n)
        ok &= all(seq(b @ b, -Id(n)) for b in bs)
        ok &= all(seq(x @ y, y @ x) for x, y in itertools.combinations(bs, 2))
        if n <= 3:
            R = dense_f(sum(bs)) / 2
            ok &= np.allclose(sla.expm(2 * np.pi * R), (-1) ** n * np.eye(8 ** n), atol=1e-9)
            w = np.linalg.eigvals(2 * R).imag
            ok &= np.allclose(w, np.round(w), atol=1e-9) and np.all(np.round(w).astype(int) % 2 == n % 2)
    return ok


@claim("e3-3b", "spin parity under the diagonal field-strength algebra lG (n = 1..5): the rotation L = (1/2) sum_k (E_(k,1)E_(k,2) - "
       "E_(k,4)E_(k,5)) commutes with G_n on vectors and is a sum of 2n commuting terms (1/2)b with b^2 = -1, so "
       "exp(2 pi L) = +1 at every rung (float check n = 1..3, tol 1e-9; spec(L) in i Z); exp(2 pi (1/2)e4e5) = exp(2 pi (1/2)e1e2) "
       "= -1 on S is the central element of Spin(3,3), which acts on S_n as (-1)^n")
def _():
    ok = sp_commuting_minus_one(5, [(k, 1, 2) for k in range(1, 6)] + [(k, 4, 5) for k in range(1, 6)])
    for n in (1, 2, 3, 4):
        ok &= np.array_equal(biv_vec(n, [(1, k, 1, 2) for k in range(1, n + 1)] + [(-1, k, 4, 5) for k in range(1, n + 1)]) @ G_vec(n),
                             G_vec(n) @ biv_vec(n, [(1, k, 1, 2) for k in range(1, n + 1)] + [(-1, k, 4, 5) for k in range(1, n + 1)]))
        bs = lG_rot(n)
        ok &= all(seq(b @ b, -Id(n)) for b in bs)
        ok &= all(seq(x @ y, y @ x) for x, y in itertools.combinations(bs, 2))
        if n <= 3:
            L = dense_f(sum(bs)) / 2
            ok &= np.allclose(sla.expm(2 * np.pi * L), np.eye(8 ** n), atol=1e-9)
            w = np.linalg.eigvals(L).imag
            ok &= np.allclose(w, np.round(w), atol=1e-9)
    for a, b in ((4, 5), (1, 2)):
        ok &= np.allclose(sla.expm(np.pi * (e[a] @ e[b]).astype(float)), -np.eye(8), atol=1e-12)
    return ok


def casimir_lint(n):
    """4 C_Delta = -sum_{a<b} eta eta (2 Delta_ab)^2 over the interior four-plane (Dirac value 4 x 3/2 = 6)"""
    E, _ = tower(n)
    D = {(a, b): sum(E[(k, a)] @ E[(k, b)] for k in range(1, n + 1)) for a, b in itertools.combinations(INT, 2)}
    return sum(-ETA[a] * ETA[b] * (D[(a, b)] @ D[(a, b)]) for (a, b) in D)


def casimir_lG(n):
    """4 C_G = -sum (2 L_i)^2 + sum (2 K_i)^2, L_i, K_i the paired rotations and boosts with a factor 1/2 (value 3 on S)"""
    E, _ = tower(n)

    def b2(a, b):
        return sum(E[(k, a)] @ E[(k, b)] for k in range(1, n + 1))
    rots = [b2(1, 2) - b2(4, 5), b2(1, 3) - b2(4, 6), b2(2, 3) - b2(5, 6)]
    boosts = [b2(1, 5) - b2(2, 4), b2(1, 6) - b2(3, 4), b2(2, 6) - b2(3, 5)]
    return sum(-(r @ r) for r in rots) + sum(b @ b for b in boosts)


@claim("e3-3c", "Casimir content (float spectra, tol 1e-6): diagonal lint, C_Delta = 3/2 (8) at n=1; 0,3,4 (8,32,24) at n=2; "
       "3/2,11/2,15/2 (160,288,64) at n=3; 0,3,4,8,9,12 (160,1024,864,864,1024,160) at n=4: half-integer spin exactly at odd n; "
       "diagonal lG, C_G = 3 (8) at n=1; 0,4,8 (4,24,36) at n=2; 3,9,15 (128,256,128) at n=3: (1/2,1/2)^(x)n, integer spin at "
       "every rung")
def _():
    want_i = {1: [(1.5, 8)], 2: [(0.0, 8), (3.0, 32), (4.0, 24)], 3: [(1.5, 160), (5.5, 288), (7.5, 64)],
              4: [(0.0, 160), (3.0, 1024), (4.0, 864), (8.0, 864), (9.0, 1024), (12.0, 160)]}
    want_g = {1: [(3.0, 8)], 2: [(0.0, 4), (4.0, 24), (8.0, 36)], 3: [(3.0, 128), (9.0, 256), (15.0, 128)]}
    ok = True
    for n in (1, 2, 3, 4):
        C = dense_f(casimir_lint(n)) / 4
        got = spec_counts(C)
        print(f"   n={n}: C_Delta spectrum {got}")
        ok &= got == want_i[n]
    for n in (1, 2, 3):
        C = dense_f(casimir_lG(n)) / 4
        got = spec_counts(C)
        print(f"   n={n}: C_G spectrum {got}")
        ok &= got == want_g[n]
    return ok


# ============================================================================== section 4: the dual pair
def copy_rot2(n):
    """2 C^{kl} = sum_a eta_aa E_(k,a)E_(l,a)"""
    E, _ = tower(n)
    return {(k, l): sum(ETA[a] * E[(k, a)] @ E[(l, a)] for a in range(1, 7)) for k in range(1, n + 1) for l in range(k + 1, n + 1)}


def diag_so33_2(n):
    E, _ = tower(n)
    return {(a, b): sum(E[(k, a)] @ E[(k, b)] for k in range(1, n + 1)) for a, b in itertools.combinations(range(1, 7), 2)}


def casimir_diag4(n):
    D = diag_so33_2(n)
    return sum(-ETA[a] * ETA[b] * (D[(a, b)] @ D[(a, b)]) for (a, b) in D)


def casimir_cop4(n):
    C = copy_rot2(n)
    return sum((-(X @ X) for X in C.values()), 0 * Id(n))


@claim("e3-4a", "vectors (exact over Q, n = 1..5): the commutant of the diagonal so(3,3) in so(3n,3n) has dimension n(n-1)/2 = "
       "0,1,3,6,10 and is spanned by the copy rotations C^{kl} (all six directions, weighted by eta_aa); the commutant of "
       "the copy so(n) together with the copy permutations is the diagonal so(3,3) (dimension 15, n = 2..5); the copy so(n) "
       "alone has commutant of dimension 36 at n = 2 and 15 at n = 3, 4, 5; the unweighted sum_a E_(k,a)E_(l,a) does not "
       "commute with the diagonal so(3,3)")
def _():
    ok = True
    for n in (1, 2, 3, 4, 5):
        D = so33_diag_vec(n)
        dimc = commutant_dim(D, n)
        ok &= dimc == n * (n - 1) // 2
        Cs = [copy_vec(n, k, l) for k in range(1, n + 1) for l in range(k + 1, n + 1)]
        ok &= all(np.array_equal(X @ Y, Y @ X) for X in Cs for Y in D)
        if n >= 2:
            perms = []
            for k in range(1, n):
                P = np.zeros((6 * n, 6 * n), dtype=np.int64)
                for j in range(1, n + 1):
                    jj = k + 1 if j == k else (k if j == k + 1 else j)
                    for a in range(6):
                        P[6 * (jj - 1) + a, 6 * (j - 1) + a] = 1
                perms.append(P)
            ok &= commutant_dim(Cs, n, extra=perms) == 15
            ok &= commutant_dim(Cs, n) == (36 if n == 2 else 15)
            Cu = copy_vec(n, 1, 2, weighted=False)
            ok &= not all(np.array_equal(Cu @ Y, Y @ Cu) for Y in D)
    return ok


@claim("e3-4b", "spin level (exact, n = 2, 3, 4): every C^{kl} commutes with the fifteen diagonal so(3,3) generators; "
       "[C^{kl}, E_(l,a)] = E_(k,a) and [C^{kl}, E_(k,a)] = -E_(l,a) (C^{kl} lifts the copy matrix e_k e_l^T - e_l e_k^T), so the "
       "C^{kl} close on so(n) with [C^{kl}, C^{lm}] = C^{km} (n = 3, 4); each is a sum of six commuting terms of square -1/4, so "
       "exp(2 pi C^{kl}) = +1 (single-valued copy rotations)")
def _():
    ok = True
    for n in (2, 3, 4):
        C = copy_rot2(n)
        D = diag_so33_2(n)
        ok &= all(seq(X @ Y, Y @ X) for X in C.values() for Y in D.values())
        E, _ = tower(n)
        for (k, l) in C:
            terms = [ETA[a] * E[(k, a)] @ E[(l, a)] for a in range(1, 7)]
            ok &= all(seq(t @ t, -Id(n)) for t in terms) and all(seq(x @ y, y @ x) for x, y in itertools.combinations(terms, 2))
        for (k, l), X in C.items():                       # X = 2 C^{kl}
            for a in range(1, 7):
                ok &= seq(X @ E[(l, a)] - E[(l, a)] @ X, 2 * E[(k, a)])
                ok &= seq(X @ E[(k, a)] - E[(k, a)] @ X, -2 * E[(l, a)])
        for k, l, m in itertools.permutations(range(1, n + 1), 3):
            if k < l and l < m:                               # [C^{kl}, C^{lm}] = C^{km}: [2C, 2C] = 2 (2C)
                ok &= seq(C[(k, l)] @ C[(l, m)] - C[(l, m)] @ C[(k, l)], 2 * C[(k, m)])
    return ok


def blade_mul(A, B, eta):
    """product of the blades with generator bitmasks A, B in the abstract Clifford algebra of diag(eta): (sign, mask)"""
    swaps = 0                                  # transpositions needed to bring the generators of B past those of A
    for b in range(len(eta)):
        if B >> b & 1:
            swaps += bin(A >> (b + 1)).count('1')
    sgn = -1 if swaps % 2 else 1
    common = A & B
    for g in range(len(eta)):
        if common >> g & 1:
            sgn *= eta[g]
    return sgn, A ^ B


def cl_mul(X, Y, eta):
    out = {}
    for A, x in X.items():
        for B, y in Y.items():
            s_, M = blade_mul(A, B, eta)
            out[M] = out.get(M, 0) + s_ * x * y
    return {M: c for M, c in out.items() if c}


def cl_add(X, Y, c=1):
    out = dict(X)
    for M, y in Y.items():
        out[M] = out.get(M, 0) + c * y
    return {M: v for M, v in out.items() if v}


def casimir_sum_abstract(n, dirs=range(1, 7)):
    """4 (C_cop + C_diag) in the abstract Clifford algebra Cl(3n,3n), for the copy rotations and the diagonal algebra of the
    directions dirs (all six, or the interior four)"""
    eta = ETA[1:] * n
    gen = lambda k, a: 1 << (6 * (k - 1) + a - 1)
    tot = {}
    for a, b in itertools.combinations(dirs, 2):
        D = {}
        for k in range(1, n + 1):
            s_, M = blade_mul(gen(k, a), gen(k, b), eta)
            D[M] = D.get(M, 0) + s_
        tot = cl_add(tot, cl_mul(D, D, eta), -ETA[a] * ETA[b])
    for k, l in itertools.combinations(range(1, n + 1), 2):
        Cm = {}
        for a in dirs:
            s_, M = blade_mul(gen(k, a), gen(l, a), eta)
            Cm[M] = Cm.get(M, 0) + ETA[a] * s_
        tot = cl_add(tot, cl_mul(Cm, Cm, eta), -1)
    return tot


@claim("e3-4c", "Casimir identity: C_cop + C_diag = (3/4) n (n+4) on S_n, i.e. 15/4, 9, 63/4, 24, 135/4, 81/2, with "
       "C_diag = -sum_{a<b} eta^aa eta^bb Delta_ab^2 over all six directions and C_cop = -sum_{k<l} (C^{kl})^2 (exact: as matrices "
       "on S_n for n = 1..4, and in the abstract Clifford algebra Cl(3n,3n), blade by blade, for n = 1..6, the abstract product "
       "checked against the matrices at n = 2); the interior version of ES17 Thm 10.4, C_cop^int + C_Delta = n(n+2)/2, re-verified "
       "in the same two ways; both are nd(d+n-2)/8 at d = 6, 4")
def _():
    ok = True
    for n in range(1, 7):
        ok &= casimir_sum_abstract(n) == {0: 3 * n * (n + 4)}
        ok &= casimir_sum_abstract(n, INT) == {0: 2 * n * (n + 2)}
    # the abstract blade product agrees with the realisation (n = 2, all pairs of single generators and a sample of blades)
    E, _ = tower(2)
    gens = [E[(k, a)] for k in (1, 2) for a in range(1, 7)]
    eta2 = ETA[1:] * 2

    def realise(M):
        R = Id(2)
        for i in range(12):
            if M >> i & 1:
                R = R @ gens[i]
        return R
    rng = np.random.default_rng(5)
    for A, B in [(1 << i, 1 << j) for i in range(12) for j in range(12)] + \
            [tuple(rng.integers(0, 4096, size=2).tolist()) for _ in range(40)]:
        s_, M = blade_mul(A, B, eta2)
        ok &= seq(realise(A) @ realise(B), s_ * realise(M))
    for n in (1, 2, 3, 4):
        ok &= seq(casimir_diag4(n) + casimir_cop4(n), 3 * n * (n + 4) * Id(n))
        E, _ = tower(n)
        Mi = [sum(ETA[a] * E[(k, a)] @ E[(l, a)] for a in INT) for k in range(1, n + 1) for l in range(k + 1, n + 1)]
        Ccop_int4 = sum((-(X @ X) for X in Mi), 0 * Id(n))
        ok &= seq(Ccop_int4 + casimir_lint(n), 2 * n * (n + 2) * Id(n))
        ok &= Fraction(n * 6 * (6 + n - 2), 8) == Fraction(3 * n * (n + 4), 4)
        ok &= Fraction(n * 4 * (4 + n - 2), 8) == Fraction(n * (n + 2), 2)
    return ok


def so6_casimir(lam):
    """<lam, lam + 2 rho> for so(6), rho = (2,1,0); vector 5, spinor 15/4"""
    a, b, c = lam
    return a * (a + 4) + b * (b + 2) + c * c


def so6_dim(lam):
    """Weyl dimension formula for D_3: prod over i<j of (l_i^2 - l_j^2)/(r_i^2 - r_j^2), l = lam + rho"""
    l = [Fraction(lam[0]) + 2, Fraction(lam[1]) + 1, Fraction(lam[2])]
    r = [2, 1, 0]
    num = den = Fraction(1)
    for i, j in itertools.combinations(range(3), 2):
        num *= (l[i] - l[j]) * (l[i] + l[j])
        den *= (r[i] - r[j]) * (r[i] + r[j])
    return num / den


def spectral_ranks_exact(C4, vals):
    """exact: (prod (C4 - v)) = 0 and the rank of each eigenprojector as a trace, in exact float64 integer arithmetic"""
    d = C4.shape[0]
    M = dense_f(C4)
    I = np.eye(d)
    P = I.copy()
    for v in vals:
        P = exact_f(P @ (M - v * I))
    if np.any(P):
        return None
    out = {}
    for v in vals:
        Q = I.copy()
        den = 1
        for w in vals:
            if w != v:
                Q = exact_f(Q @ (M - w * I))
                den *= (v - w)
        out[v] = Fraction(int(round(np.trace(Q))), den)
    return out


@claim("e3-4d", "joint decomposition (O(n), so(6)) [Howe skew duality]: n = 2 (exact): C_cop = j^2 = 0,1,4,9 on 20,30,12,2 and C_diag = 9,8,5,0, "
       "the so(6) modules (1,1,+-1), (1,1,0), (1,0,0), (0,0,0) of dimensions 10+10, 15, 6, 1; n = 3 (exact): C_cop = j(j+1) = 0,2,6,12 "
       "on 40,216,200,56 (RE d7) and C_diag = 63/4, 55/4, 39/4, 15/4, the modules (3/2,3/2,+-3/2), (3/2,3/2,+-1/2), (3/2,1/2,+-1/2), "
       "(1/2,1/2,+-1/2) of dimensions 20, 36, 20, 4 each; n = 4 (float, per Fock block, tol 1e-6): C_cop = 0,3,4,8,9,12,15,16,19,24 on "
       "70,560,504,810,1024,200,320,450,144,14, matching so(4) = su(2)+su(2) modules times the so(6) module of Casimir 24 - C_cop; "
       "every rank equals dim(copy module) x dim(so(6) module)")
def _():
    ok = True
    # n = 2
    r2 = spectral_ranks_exact(casimir_cop4(2), [0, 4, 16, 36])
    ok &= r2 == {0: 20, 4: 30, 16: 12, 36: 2}
    pairs2 = {0: ([(1, 1, 1), (1, 1, -1)], 1), 1: ([(1, 1, 0)], 2), 2: ([(1, 0, 0)], 2), 3: ([(0, 0, 0)], 2)}
    for j, (lams, dcop) in pairs2.items():
        ok &= all(so6_casimir(l) == 9 - j * j for l in lams)
        ok &= dcop * sum(so6_dim(l) for l in lams) == r2[4 * j * j]
    # n = 3
    r3 = spectral_ranks_exact(casimir_cop4(3), [0, 8, 24, 48])
    ok &= r3 == {0: 40, 8: 216, 24: 200, 48: 56}
    h = Fraction(1, 2)
    pairs3 = {0: [(3 * h, 3 * h, 3 * h), (3 * h, 3 * h, -3 * h)], 1: [(3 * h, 3 * h, h), (3 * h, 3 * h, -h)],
              2: [(3 * h, h, h), (3 * h, h, -h)], 3: [(h, h, h), (h, h, -h)]}
    for j, lams in pairs3.items():
        ok &= all(so6_casimir(l) == Fraction(63, 4) - j * (j + 1) for l in lams)
        ok &= (2 * j + 1) * sum(so6_dim(l) for l in lams) == r3[4 * j * (j + 1)]
    ok &= [so6_dim(pairs3[j][0]) for j in range(4)] == [20, 36, 20, 4]
    # n = 4: C_cop commutes with N (paired rotations), so diagonalise block by block
    n = 4
    C4 = casimir_cop4(n).tocsr()
    Nd = number_diag(n)
    cnt = Counter()
    for j in range(3 * n + 1):
        idx = np.nonzero(Nd == j)[0]
        rest = np.nonzero(Nd != j)[0]
        ok &= C4[idx][:, rest].count_nonzero() == 0
        for v, c in spec_counts(dense_f(C4[idx][:, idx]) / 4):
            cnt[v] += c
    got = sorted(cnt.items())
    print(f"   n=4: C_cop spectrum {got}")
    ok &= got == [(0.0, 70), (3.0, 560), (4.0, 504), (8.0, 810), (9.0, 1024), (12.0, 200), (15.0, 320), (16.0, 450),
                  (19.0, 144), (24.0, 14)]
    # so(4) modules (j1,j2)+(j2,j1) with C = 2[j1(j1+1)+j2(j2+1)], and the so(6) partners
    so4 = {0: [(0, 0)], 3: [(h, h)], 4: [(1, 0), (0, 1)], 8: [(1, 1)], 9: [(3 * h, h), (h, 3 * h)], 12: [(2, 0), (0, 2)],
           15: [(3 * h, 3 * h)], 16: [(2, 1), (1, 2)], 19: [(5 * h, h), (h, 5 * h)], 24: [(3, 0), (0, 3)]}
    so6 = {0: [(2, 2, 2), (2, 2, -2)], 3: [(2, 2, 1), (2, 2, -1)], 4: [(2, 2, 0)], 8: [(2, 1, 1), (2, 1, -1)], 9: [(2, 1, 0)],
           12: [(2, 0, 0)], 15: [(1, 1, 1), (1, 1, -1)], 16: [(1, 1, 0)], 19: [(1, 0, 0)], 24: [(0, 0, 0)]}
    for c, mods in so4.items():
        ok &= all(2 * (a * (a + 1) + b * (b + 1)) == c for a, b in mods)
        dcop = sum((2 * a + 1) * (2 * b + 1) for a, b in mods)
        ok &= all(so6_casimir(l) == 24 - c for l in so6[c])
        ok &= dcop * sum(so6_dim(l) for l in so6[c]) == cnt[float(c)]
    return ok


# Koszul swap on R^8 (x) R^8
Psw = np.zeros((64, 64), dtype=np.int64)
for _i in range(8):
    for _j in range(8):
        Psw[_j * 8 + _i, _i * 8 + _j] = 1
pbit = (1 - np.diag(om)) // 2
Dk = np.diag([(-1) ** (int(pbit[i]) * int(pbit[j])) for i in range(8) for j in range(8)]).astype(np.int64)
tauK = Dk @ Psw


def koszul(n, k):
    """tau_k on S_n: Koszul swap of the factors k, k+1"""
    return skron(*([S(I8)] * (k - 1) + [S(tauK)] + [S(I8)] * (n - k - 1)))


@claim("e3-4e", "at n = 2 the commutant of the diagonal so(3,3) in M_64(R) has dimension 14 (float nullity of sum ad^T ad, gap > 1) and "
       "equals the span of the words in C_b = C^{12} and the Koszul swap tau_K (exact rank 14): the image of the group algebra of "
       "O(2), with C_b tau_K = -tau_K C_b; so(2) alone generates only 7 dimensions")
def _():
    n = 2
    D = list(diag_so33_2(n).values())
    Cb2 = copy_rot2(n)[(1, 2)]
    tau = koszul(2, 1)
    ok = all(seq(tau @ X, X @ tau) for X in D) and seq(Cb2 @ tau, -(tau @ Cb2))
    d = 64
    I = ss.identity(d, format='csr')
    M = 0
    for X in D:
        Xf = X.astype(float)
        A = ss.kron(Xf, I) - ss.kron(I, Xf.T)
        M = M + (A.T @ A)
    ev = np.linalg.eigvalsh(M.toarray())
    null = int(np.count_nonzero(ev < 1e-8))
    ok &= null == 14 and ev[14] > 1
    # span of words in C_b (powers 0..6) and tau
    words = []
    P = Id(2)
    for p in range(7):
        words.append(P)
        words.append(P @ tau)
        P = P @ Cb2
    Wm = np.array([dense_f(w).ravel() for w in words])
    ok &= np.linalg.matrix_rank(Wm) == 14
    Wc = np.array([dense_f(w).ravel() for w in words[0::2]])
    ok &= np.linalg.matrix_rank(Wc) == 7
    # exact rank of the 14 words (integer Gram matrix over ZZ)
    G = (Wm @ Wm.T).astype(np.int64)
    ok &= DomainMatrix([[ZZ(int(x)) for x in r] for r in G.tolist()], G.shape, ZZ).rank() == 14
    return ok


# ============================================================================== section 5: the symmetric group
@claim("e3-5a", "Koszul swaps (exact, n = 3, 4): tau_k = 1..(x) tau_K (x)..1 satisfy tau_k^2 = 1, (tau_k tau_(k+1))^3 = 1 and "
       "tau_k tau_l = tau_l tau_k for |k-l| > 1, so they define an action of S_n on S_n; tau_k E_(j,a) tau_k = E_(s_k j, a); "
       "tau_k = (1/8) prod_a (E_(k,a) - E_(k+1,a)); tau_k commutes with the diagonal so(3,3), the copy Casimir, N and H_g "
       "(equal masses), and conjugates C^{kl} as the permutation does")
def _():
    ok = True
    for n in (3, 4):
        E, _ = tower(n)
        I = Id(n)
        t = {k: koszul(n, k) for k in range(1, n)}
        for k in range(1, n):
            ok &= seq(t[k] @ t[k], I)
            if k + 1 < n:
                x = t[k] @ t[k + 1]
                ok &= seq(x @ x @ x, I)
            for l in range(k + 2, n):
                ok &= seq(t[k] @ t[l], t[l] @ t[k])
            for j in range(1, n + 1):
                jj = k + 1 if j == k else (k if j == k + 1 else j)
                for a in range(1, 7):
                    ok &= seq(t[k] @ E[(j, a)] @ t[k], E[(jj, a)])
            prod = I
            for a in range(1, 7):
                prod = prod @ (E[(k, a)] - E[(k + 1, a)])
            ok &= seq(prod, 8 * t[k])
            ok &= all(seq(t[k] @ X, X @ t[k]) for X in diag_so33_2(n).values())
            Nm = ss.diags(number_diag(n), dtype=np.int64).tocsr()
            H = sum(E[(j, 1)] for j in range(1, n + 1))
            ok &= seq(t[k] @ Nm, Nm @ t[k]) and seq(t[k] @ H, H @ t[k])
            ok &= seq(t[k] @ casimir_cop4(n), casimir_cop4(n) @ t[k])
        C = copy_rot2(n)
        ok &= seq(t[1] @ C[(1, 3)] @ t[1], C[(2, 3)])
    return ok


def s3_projectors():
    """isotypic projectors of the Koszul S_3 on S_3, times 6 (exact integer float64)"""
    t1, t2 = dense_f(koszul(3, 1)), dense_f(koszul(3, 2))
    I = np.eye(512)
    grp = [I, t1, t2, t1 @ t2, t2 @ t1, t1 @ t2 @ t1]
    sg = [1, -1, -1, 1, 1, -1]
    ch = [2, 0, 0, -1, -1, 0]
    return {'triv': sum(grp), 'sign': sum(s * g for s, g in zip(sg, grp)), 'std': 2 * sum(c * g for c, g in zip(ch, grp))}


def isotypic_table(P6, C4, vals):
    M = dense_f(C4)
    I = np.eye(M.shape[0])
    out = {}
    for name, P in P6.items():
        row = []
        for v in vals:
            Q = I.copy()
            den = 1
            for w in vals:
                if w != v:
                    Q = exact_f(Q @ (M - w * I))
                    den *= (v - w)
            row.append(Fraction(int(round(np.trace(exact_f(P @ Q)))), 6 * den))
        out[name] = row
    return out


@claim("e3-5b", "n = 3, Koszul S_3 isotypic content (exact traces): under the diagonal interior algebra (C_Delta = 3/2, 11/2, 15/2) "
       "trivial 24,48,16; sign 24,48,16; standard 112,192,32 (= 2 x 56,96,16); under the copy so(3) (C_cop = 0,2,6,12) trivial "
       "20,36,20,12; sign 20,36,20,12; standard 0,144,160,32; the Dirac block (160) contains 24 totally symmetric and 24 totally "
       "antisymmetric dimensions")
def _():
    P6 = s3_projectors()
    ti = isotypic_table(P6, casimir_lint(3), [6, 22, 30])
    tc = isotypic_table(P6, casimir_cop4(3), [0, 8, 24, 48])
    ok = ti == {'triv': [24, 48, 16], 'sign': [24, 48, 16], 'std': [112, 192, 32]}
    ok &= tc == {'triv': [20, 36, 20, 12], 'sign': [20, 36, 20, 12], 'std': [0, 144, 160, 32]}
    return ok


@claim("e3-5c", "the mirror: for a surplus direction s, W_s = E_(1,s)...E_(n,s) is invertible, anticommutes with every Koszul "
       "transposition tau_k, commutes with the diagonal interior algebra and with every copy rotation C^{kl} (exact, n = 2, 3, 4), "
       "and anticommutes with the cubic so(6) invariant C_3 = sum eps L L L of the diagonal so(3,3) (n = 2, 3): the totally symmetric "
       "and totally antisymmetric Koszul sectors are isomorphic under lint x so(n) and are mirror images (opposite C_3) under "
       "so(3,3); at n = 3 the C_3 eigenvalues on them are -630,-210,-126 against +630,+210,+126 in the copy sectors j = 0,1,2 (float, tol 1e-6)")
def _():
    ok = True
    for n in (2, 3, 4):
        E, _ = tower(n)
        for s in SUR:
            W = Id(n)
            for k in range(1, n + 1):
                W = W @ E[(k, s)]
            ok &= seq(W @ W, Id(n)) or seq(W @ W, -Id(n))
            for k in range(1, n):
                t = koszul(n, k)
                ok &= seq(W @ t, -(t @ W))
            for a, b in itertools.combinations(INT, 2):
                X = sum(E[(k, a)] @ E[(k, b)] for k in range(1, n + 1))
                ok &= seq(W @ X, X @ W)
            ok &= all(seq(W @ X, X @ W) for X in copy_rot2(n).values())
    from sympy.combinatorics import Permutation
    for n in (2, 3):
        E, _ = tower(n)
        Ed = {k: dense_f(v) for k, v in E.items()}
        L = {}
        for a in range(6):
            for b in range(6):
                if a != b:
                    L[(a, b)] = sum(Ed[(k, a + 1)] @ Ed[(k, b + 1)] for k in range(1, n + 1)) * ETA[a + 1] * ETA[b + 1]  # 2 L^{ab}
        C3 = np.zeros((8 ** n, 8 ** n))
        for p in itertools.permutations(range(6)):
            if p[0] < p[1] and p[2] < p[3] and p[4] < p[5]:
                C3 += Permutation(list(p)).signature() * (L[(p[0], p[1])] @ L[(p[2], p[3])] @ L[(p[4], p[5])])
        C3 = exact_f(C3)                     # = 8 C_3 with L = (1/2) sum E E
        W = dense_f(E[(1, 2)])
        for k in range(2, n + 1):
            W = W @ dense_f(E[(k, 2)])
        ok &= np.array_equal(W @ C3, -(C3 @ W))
        if n == 3:
            P6 = s3_projectors()
            M = dense_f(casimir_cop4(3))
            I = np.eye(512)
            vals = [0, 8, 24, 48]
            want = {0: 630, 8: 210, 24: 126}
            for v in (0, 8, 24):
                Q = I.copy()
                for w in vals:
                    if w != v:
                        Q = Q @ (M - w * I) / (v - w)
                for nm, sgn in (('triv', -1), ('sign', 1)):
                    R = P6[nm] @ Q / 6
                    ev = np.linalg.eigvals(C3 @ R)
                    ev = ev[np.abs(ev) > 1e-6].real
                    ok &= len(ev) == round(np.trace(R)) and np.allclose(ev, sgn * want[v], atol=1e-6)
    return ok


def W_projector_sp_terms(n, Eg):
    """terms of 2^(n-1) P_W = prod_k (1 - J_k J_(k+1)) as (sign, SP) pairs"""
    Js = [Eg[(k, 2)] @ Eg[(k, 3)] for k in range(1, n + 1)]
    Pk = [Js[k] @ Js[k + 1] for k in range(n - 1)]
    terms = []
    for r in range(n):
        for sub in itertools.combinations(range(n - 1), r):
            M = SP.ident(8 ** n)
            for k in sub:
                M = M @ Pk[k]
            terms.append(((-1) ** r, M))
    return terms


@claim("e3-5d", "the ungraded n-particle reading: the plain factor permutations P_sigma commute with J^(k)-permutations and preserve "
       "W_n = {J^(1) = ... = J^(n)} (dim 2 x 4^n, a complex space C^(4^n) = (C^4)^(x_C n)); the totally antisymmetric part of W_n has "
       "real dimension 2 binom(4,n) = 8,12,8,2,0 and the totally symmetric part 2 binom(n+3,3) = 8,20,40,70,112 for n = 1..5 "
       "(exact traces over signed permutations); its Lorentz content (exact): n = 2: C_Delta = 0 on 4, 3 on 8 (2(0,0)+(1/2,1/2)); "
       "n = 3: C_Delta = 3/2 on all 8 (Lambda^3 C^4 = Dirac); n = 4: C_Delta = 0 on 2 (Lambda^4 C^4 = det, a scalar)")
def _():
    ok = True
    for n in (1, 2, 3, 4, 5):
        Eg, _ = tower_sp(n)
        terms = W_projector_sp_terms(n, Eg)
        tot_a = tot_s = 0
        perms = list(itertools.permutations(range(n)))
        for sigma in perms:
            Pp = factor_perm_sp(n, sigma)
            sgn = 1
            for i, j in itertools.combinations(range(n), 2):
                if sigma[i] > sigma[j]:
                    sgn = -sgn
            for s, M in terms:
                tr = (Pp @ M).trace()
                tot_a += sgn * s * tr
                tot_s += s * tr
        den = math.factorial(n) * 2 ** (n - 1)
        ok &= Fraction(tot_a, den) == 2 * math.comb(4, n) and Fraction(tot_s, den) == 2 * math.comb(n + 3, 3)
        # P_sigma permutes the J^(k)
        if n >= 2:
            Pp = factor_perm_sp(n, (1, 0) + tuple(range(2, n)))
            ok &= (Pp @ (Eg[(1, 2)] @ Eg[(1, 3)]) @ Pp) == (Eg[(2, 2)] @ Eg[(2, 3)])
        if n in (2, 3, 4):
            # projector R = (1/(n! 2^(n-1))) sum sgn P_sigma prod(1 - J J), as a sparse integer matrix times den
            R = 0
            for sigma in perms:
                Pp = sp_to_sparse(factor_perm_sp(n, sigma))
                sgn = 1
                for i, j in itertools.combinations(range(n), 2):
                    if sigma[i] > sigma[j]:
                        sgn = -sgn
                for s, M in terms:
                    R = R + (sgn * s) * (Pp @ sp_to_sparse(M))
            R = R.tocsr()
            C4 = casimir_lint(n)
            ok &= seq(R @ R, den * R)                      # R/den is a projector
            if n == 2:
                ok &= seq(C4 @ (C4 - 12 * Id(n)) @ R, 0 * R)
                rk0 = Fraction(int(((C4 - 12 * Id(n)) @ R).diagonal().sum()), -12 * den)
                ok &= rk0 == 4
            if n == 3:
                ok &= seq(C4 @ R, 6 * R)
            if n == 4:
                ok &= seq(C4 @ R, 0 * R)
    return ok


# ============================================================================== section 6: the ladder table
@claim("e3-6a", "ladder table (vectors, exact over Q, n = 1..5): commutant in so(3n,3n) of the spectator lint 1,28,91,190,325 = "
       "dim so(3n-1,3n-3); of the diagonal lint 1,7,18,34,55 = dim so(n)+so(2n); of the diagonal lG 0,2,6,12,20 = dim so(n,C) with "
       "compact part 0,1,3,6,10; of the diagonal rotations and G_n in so(6n) 1,4,9,16,25 = dim u(n), and the same inside so(3n,3n) "
       "(u(n) cap so(3n,3n)) 0,1,3,6,10 = dim so(n) (restates ES17 Prop. 13.14); the full commutant of the diagonal rotations and "
       "G_n in so(3n,3n) is 0,2,6,12,20 = dim so(n,C), the commutant of the diagonal lG")
def _():
    ok = True
    want = {1: (1, 1, 0, 0, 1, 0, 0), 2: (28, 7, 2, 1, 4, 1, 2), 3: (91, 18, 6, 3, 9, 3, 6), 4: (190, 34, 12, 6, 16, 6, 12),
            5: (325, 55, 20, 10, 25, 10, 20)}
    for n in (1, 2, 3, 4, 5):
        spect = [biv_vec(n, [(1, 1, a, b)]) for a, b in itertools.combinations(INT, 2)]
        d1 = commutant_dim(spect, n)
        d2 = commutant_dim(lint_diag_vec(n), n)
        rots, boosts = lG_diag_vec(n)
        G = G_vec(n)
        ok &= all(np.array_equal(X @ G, G @ X) for X in rots + boosts)
        d3 = commutant_dim(rots + boosts, n)
        # compact part: antisymmetric (Euclidean) elements of the commutant: commutant of lG and of the Cartan-type K_n test
        # handled by the vector-level identity: compact = commuting additionally with eta (i.e. X^T = -X)
        eta = np.diag(ETA[1:] * n)
        d4 = commutant_dim(rots + boosts, n, extra=[eta])
        # u(n): antisymmetric 6n x 6n matrices commuting with the diagonal rotations and G_n
        N6 = 6 * n
        Abasis = []
        for a, b in itertools.combinations(range(N6), 2):
            X = np.zeros((N6, N6), dtype=np.int64)
            X[a, b], X[b, a] = 1, -1
            Abasis.append(X)
        conds = rots + [G]
        A = np.vstack([np.array([(X @ g - g @ X).ravel() for X in Abasis]).T for g in conds])
        A = A[np.any(A != 0, axis=1)]
        AtA = A.T @ A
        d5 = len(Abasis) - DomainMatrix([[ZZ(int(x)) for x in r] for r in AtA.tolist()], AtA.shape, ZZ).rank()
        d6 = commutant_dim(rots + [G], n, extra=[eta])  # u(n) cap so(3n,3n): X in so(3n,3n), X^T = -X, [X, G] = [X, rots] = 0
        d7 = commutant_dim(rots + [G], n)  # full commutant in so(3n,3n) of the diagonal rotations and G_n
        got = (d1, d2, d3, d4, d5, d6, d7)
        print(f"   n={n}: {got}")
        ok &= got == want[n]
        p, q = 3 * n - 1, 3 * n - 3
        ok &= d1 == (p + q) * (p + q - 1) // 2 and d2 == n * (n - 1) // 2 + n * (2 * n - 1) and d3 == n * (n - 1)
    return ok


@claim("e3-6b", "the copy-singlet four-vector at n = 3 (ES17 Thm 13.15 at g = 3, Thm 10.6): Q_mu = sum_nu eta_nunu (g1_mu g2_nu g3_nu + "
       "g1_nu g2_mu g3_nu + g1_nu g2_nu g3_mu) commutes with the interior copy rotations M_kl and satisfies [Delta_ab, Q_c] = "
       "eta_bc Q_a - eta_ac Q_b (exact); at n = 2 the analogous sum_nu eta_nunu (g1_mu g2_nu + g1_nu g2_mu) commutes with M_12 but is "
       "a symmetric tensor, not a vector: [Delta_ab, X_c] != eta_bc X_a - eta_ac X_b")
def _():
    ok = True
    E, _ = tower(3)
    g = {(k, i): E[(k, mu)] for k in (1, 2, 3) for i, mu in enumerate(INT)}
    et = [ETA[mu] for mu in INT]
    Q = {}
    for i in range(4):
        Q[i] = sum(et[j] * (g[(1, i)] @ g[(2, j)] @ g[(3, j)] + g[(1, j)] @ g[(2, i)] @ g[(3, j)] + g[(1, j)] @ g[(2, j)] @ g[(3, i)])
                   for j in range(4))
    M = [sum(et[j] * g[(k, j)] @ g[(l, j)] for j in range(4)) for k, l in ((1, 2), (1, 3), (2, 3))]
    ok &= all(seq(X @ Q[i], Q[i] @ X) for X in M for i in range(4))
    for a, b in itertools.combinations(range(4), 2):
        D2 = sum(g[(k, a)] @ g[(k, b)] for k in (1, 2, 3))     # 2 Delta_ab
        for c in range(4):
            lhs = D2 @ Q[c] - Q[c] @ D2
            rhs = 2 * ((et[b] if b == c else 0) * Q[a] - (et[a] if a == c else 0) * Q[b])
            ok &= seq(lhs, rhs)
    ok &= not is_zero(Q[0])
    E2, _ = tower(2)
    g2 = {(k, i): E2[(k, mu)] for k in (1, 2) for i, mu in enumerate(INT)}
    X = {i: sum(et[j] * (g2[(1, i)] @ g2[(2, j)] + g2[(1, j)] @ g2[(2, i)]) for j in range(4)) for i in range(4)}
    M12 = sum(et[j] * g2[(1, j)] @ g2[(2, j)] for j in range(4))
    ok &= all(seq(M12 @ X[i], X[i] @ M12) for i in range(4))
    vec = True
    for a, b in itertools.combinations(range(4), 2):
        D2 = sum(g2[(k, a)] @ g2[(k, b)] for k in (1, 2))
        for c in range(4):
            rhs = 2 * ((et[b] if b == c else 0) * X[a] - (et[a] if a == c else 0) * X[b])
            vec &= seq(D2 @ X[c] - X[c] @ D2, rhs)
    ok &= not vec
    return ok


@claim("e3-6c", "the step n -> n+1 preserves the lower rung and adds in two ways (exact, n = 1..3): H_g^(n+1) = H_g^(n) (x) 1 + "
       "vol_n (x) e_1 with anticommuting summands (masses in quadrature), H_u^(n+1) = H_u^(n) (x) 1 + 1 (x) e_1 with commuting "
       "summands (masses add); every diagonal bivector satisfies X^(n+1) = X^(n) (x) 1 + 1 (x) X^(1), so S_(n+1) = S_n (x) S as a "
       "module of the diagonal so(3,3) and of its subalgebras lint, lG; the Casimir constant grows by (3/4)(2n+5)")
def _():
    ok = True
    for n in (1, 2, 3):
        E0, U0 = tower(n)
        E1, U1 = tower(n + 1)
        Hg0 = sum(E0[(k, 1)] for k in range(1, n + 1))
        Hu0 = sum(U0[(k, 1)] for k in range(1, n + 1))
        Hg1 = sum(E1[(k, 1)] for k in range(1, n + 2))
        Hu1 = sum(U1[(k, 1)] for k in range(1, n + 2))
        A = skron(Hg0, S(I8))
        B = skron(vol(n), S(e[1]))
        ok &= seq(Hg1, A + B) and is_zero(A @ B + B @ A)
        A = skron(Hu0, S(I8))
        B = skron(Id(n), S(e[1]))
        ok &= seq(Hu1, A + B) and is_zero(A @ B - B @ A)
        for a, b in itertools.combinations(range(1, 7), 2):
            X0 = sum(E0[(k, a)] @ E0[(k, b)] for k in range(1, n + 1))
            X1 = sum(E1[(k, a)] @ E1[(k, b)] for k in range(1, n + 2))
            ok &= seq(X1, skron(X0, S(I8)) + skron(Id(n), S(e[a] @ e[b])))
        ok &= Fraction(3 * (n + 1) * (n + 5), 4) - Fraction(3 * n * (n + 4), 4) == Fraction(3 * (2 * n + 5), 4)
    return ok


if __name__ == "__main__":
    print(f"e3_ladder.py: {NPASS[0]} PASS, {len(FAILS)} FAIL, {time.time() - T0:.1f}s")
    sys.exit(1 if FAILS else 0)
