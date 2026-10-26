#!/usr/bin/env python3
"""p2: alignment by a gauged subalgebra on the compact families of vacuum projectors (second and third rung).

    python3 investigations/photon_selection/verify/p2_alignment.py

Prints one 'PASS p2-<n>: <claim>' (or 'FAIL ...') line per claim of sections/p2_alignment.tex and exits 0 iff all pass.

Realisation (tex/s02_arena.tex, tex/sA_realisation.tex), as in verify/p1_families.py: e1=sx(x)1(x)1, e2=sz(x)sx(x)1,
e3=sz(x)sz(x)sx, e4=ep(x)1(x)1, e5=sz(x)ep(x)1, e6=sz(x)sz(x)ep, ep=[[0,-1],[1,0]], omega=e1...e6,
E_{6(k-1)+a}=omega^{(x)(k-1)} (x) e_a (x) 1...  Every generator and blade is a signed permutation matrix (class SP); linear
combinations of blades with dyadic coefficients (class Comb) are expanded exactly, and float64 is exact on them.  Tangent
representations are converted to exact fractions and the conversion is checked to be exact.  Floating point with stated
tolerances is used only at random points of the families, for spectra, null spaces and optimisation.

Normalisation: on the compact internal algebra k the invariant form is <X,Y> = -Tr(XY)/(N/4) (N = 512 at n=3, 64 at n=2), in
which the unit-rate rotations e_ie_j/2 are orthonormal and rho(e_ie_j/2) = E_ij - E_ji on the positive span.
"""
import itertools
import sys
import time
from fractions import Fraction

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm, null_space
from scipy.optimize import minimize

T_START = time.time()
TOL = 1e-9
RNG = np.random.default_rng(20261026)
RESULTS = []


def report(n, ok, claim):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} p2-{n}: {claim}", flush=True)


# ============================================================================================ exact arithmetic
class SP:
    """Signed permutation matrix M with M e_i = s_i e_{p_i}, i.e. M[p_i, i] = s_i."""
    __slots__ = ("p", "s")

    def __init__(self, p, s):
        self.p = np.asarray(p, dtype=np.int64)
        self.s = np.asarray(s, dtype=np.int64)

    @property
    def N(self):
        return len(self.p)

    def __matmul__(self, o):
        if isinstance(o, SP):
            return SP(self.p[o.p], o.s * self.s[o.p])
        if isinstance(o, Comb):
            return Comb([(1.0, self)]) @ o
        return NotImplemented

    def __neg__(self):
        return SP(self.p, -self.s)

    def __eq__(self, o):
        return np.array_equal(self.p, o.p) and np.array_equal(self.s, o.s)

    __hash__ = None

    @property
    def T(self):
        inv = np.empty_like(self.p)
        inv[self.p] = np.arange(self.N)
        return SP(inv, self.s[inv])

    def trace(self):
        return int(np.sum(self.s[self.p == np.arange(self.N)]))

    def dense(self):
        D = np.zeros((self.N, self.N))
        D[self.p, np.arange(self.N)] = self.s
        return D

    def lmul(self, X):
        out = np.empty_like(X)
        out[self.p] = self.s[:, None] * X
        return out

    def rmul(self, X):
        return X[:, self.p] * self.s[None, :]


class Comb:
    """Finite real linear combination of signed permutations."""

    def __init__(self, terms):
        self.terms = [(float(c), S) for c, S in terms if c != 0]

    @staticmethod
    def of(x):
        return x if isinstance(x, Comb) else Comb([(1.0, x)])

    def __add__(self, o):
        return Comb(self.terms + Comb.of(o).terms)

    def __sub__(self, o):
        return Comb(self.terms + [(-c, S) for c, S in Comb.of(o).terms])

    def __mul__(self, a):
        return Comb([(a * c, S) for c, S in self.terms])

    __rmul__ = __mul__

    def __neg__(self):
        return self * (-1.0)

    def __matmul__(self, o):
        o = Comb.of(o)
        return Comb([(c * d, S @ R) for c, S in self.terms for d, R in o.terms])

    def __rmatmul__(self, o):
        return Comb.of(o) @ self

    @property
    def T(self):
        return Comb([(c, S.T) for c, S in self.terms])

    def dense(self):
        N = self.terms[0][1].N
        out = np.zeros((N, N))
        ar = np.arange(N)
        for c, S in self.terms:
            out[S.p, ar] += c * S.s
        return out

    def trace(self):
        return sum(c * S.trace() for c, S in self.terms)

    def lmul(self, X):
        out = np.zeros_like(X)
        for c, S in self.terms:
            out += c * S.lmul(X)
        return out

    def rmul(self, X):
        out = np.zeros_like(X)
        for c, S in self.terms:
            out += c * S.rmul(X)
        return out


def cm(a, b):
    a, b = Comb.of(a), Comb.of(b)
    return a @ b - b @ a


def dcm(X, D):
    """[X, D] for X exact and D dense."""
    X = Comb.of(X)
    return X.lmul(D) - X.rmul(D)


def nz(x):
    if isinstance(x, (SP, Comb)):
        if isinstance(x, Comb) and not x.terms:
            return 0.0
        x = Comb.of(x).dense()
    return float(np.max(np.abs(x))) if x.size else 0.0


def iszero(x, tol=TOL):
    return nz(x) < tol


def close(a, b, tol=TOL):
    a = Comb.of(a).dense() if isinstance(a, (SP, Comb)) else a
    b = Comb.of(b).dense() if isinstance(b, (SP, Comb)) else b
    return float(np.max(np.abs(a - b))) < tol


def commutes(A, B):
    return A @ B == B @ A


def anticommutes(A, B):
    return A @ B == -(B @ A)


def sp_from_dense(M):
    p = np.argmax(np.abs(M), axis=0)
    return SP(p, M[p, np.arange(M.shape[1])])


def sp_kron(ops):
    p = np.array([0], dtype=np.int64)
    s = np.array([1], dtype=np.int64)
    for o in ops:
        m = o.N
        p = (p[:, None] * m + o.p[None, :]).ravel()
        s = (s[:, None] * o.s[None, :]).ravel()
    return SP(p, s)


def ident(N):
    return SP(np.arange(N), np.ones(N, dtype=np.int64))


def blade(E, idx):
    out = E[idx[0]]
    for i in idx[1:]:
        out = out @ E[i]
    return out


def frac(x, den=10 ** 6):
    """Exact conversion of a dyadic float to a Fraction; asserts exactness."""
    f = Fraction(float(x)).limit_denominator(den)
    assert abs(float(f) - x) < 1e-12, "non-rational entry"
    return f


def frac_matrix(M):
    return [[frac(v) for v in row] for row in M]


def frac_is_scalar(F, val):
    n = len(F)
    return all(F[i][j] == (val if i == j else 0) for i in range(n) for j in range(n))


# ----------------------------------------------------------------------------------------------- realisation
_sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
_sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
_ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
_I2 = np.eye(2, dtype=np.int64)


def _kron(*ms):
    out = np.array([[1]], dtype=np.int64)
    for m in ms:
        out = np.kron(out, m)
    return out


e8_dense = [_kron(_sx, _I2, _I2), _kron(_sz, _sx, _I2), _kron(_sz, _sz, _sx),
            _kron(_ep, _I2, _I2), _kron(_sz, _ep, _I2), _kron(_sz, _sz, _ep)]
e8 = [sp_from_dense(m) for m in e8_dense]
om8 = e8[0] @ e8[1] @ e8[2] @ e8[3] @ e8[4] @ e8[5]
ETA1 = [1, 1, 1, -1, -1, -1]


def tower(n):
    E, eta = [None], [0]
    for k in range(1, n + 1):
        for a in range(6):
            E.append(sp_kron([om8] * (k - 1) + [e8[a]] + [ident(8)] * (n - k)))
            eta.append(ETA1[a])
    return E, eta


def vec_rep(X, E, idx):
    """[X, E_l] = sum_k M_kl E_k on span{E_i : i in idx}; asserts that the image lies in the span (exactly)."""
    n = len(idx)
    N = E[idx[0]].N
    M = np.zeros((n, n))
    for c_, l_ in enumerate(idx):
        img = cm(X, E[l_])
        for r_, k_ in enumerate(idx):
            M[r_, c_] = (E[k_].T @ img).trace() / N
        rec = img - Comb([(M[r_, c_], E[k_]) for r_, k_ in enumerate(idx)])
        assert iszero(rec), "vector image leaves the span"
    return M


def orth_basis(mats, tol=1e-8):
    A = np.array([m.ravel() for m in mats])
    G = A @ A.T
    w, V = np.linalg.eigh(G)
    keep = w > tol * max(1.0, float(np.max(w)))
    B = (V[:, keep] / np.sqrt(w[keep])).T @ A
    return [b.reshape(mats[0].shape) for b in B]


def span_rank(mats, tol=1e-8):
    if not mats:
        return 0
    A = np.array([m.ravel() for m in mats])
    G = A @ A.T
    return int(np.linalg.matrix_rank(G, tol=tol * max(1.0, float(np.max(np.abs(G))))))


# =============================================================================================== n = 3 data
E, eta = tower(3)
N = 512
Id = np.eye(N)
Isp = ident(N)
interior = [1, 4, 5, 6]
internal = [i for i in range(1, 19) if i not in interior]
Vp = [i for i in internal if eta[i] > 0]
Vm = [i for i in internal if eta[i] < 0]
Aidx, Widx, Cidx = [2, 3, 8, 9], [7, 13, 14, 15], [10, 11, 12, 16, 17, 18]
kpos_pairs = list(itertools.combinations(Vp, 2))
kneg_pairs = list(itertools.combinations(Vm, 2))
kbiv = kpos_pairs + kneg_pairs
gam = [E[i] for i in interior]
J = blade(E, internal)
Gam = J @ E[1] @ E[4] @ E[5] @ E[6]
chiW = blade(E, Widx)
chiC = blade(E, Cidx)
qA = E[2] @ E[3]
volA = blade(E, Aidx)
volW = chiW
volPlus = blade(E, Vp)
PiA = Comb([(0.5, Isp), (0.5, volA)])
PiW = Comb([(0.5, Isp), (0.5, volW)])
PiA_d, PiW_d = PiA.dense(), PiW.dense()
Xinv = Gam @ chiW
Zinv = -(J @ chiC)
Uinv = -(J @ qA)
Pst = Comb([(1.0 / 8, Isp)]) @ (Comb.of(Isp) + Xinv) @ (Comb.of(Isp) + Zinv) @ (Comb.of(Isp) + Uinv)
PL = Pst @ Comb([(0.5, Isp), (-0.5, Gam)])
PR = Pst @ Comb([(0.5, Isp), (0.5, Gam)])


def ip(X, Y, Nn=N):
    """invariant form <X,Y> = -Tr(XY)/(N/4) on k."""
    return -(Comb.of(X) @ Comb.of(Y)).trace() / (Nn / 4)


def unit(i, j):
    return Comb([(0.5, E[i] @ E[j])])


w = [E[i] for i in Widx]
tau = [Comb([(-0.5, w[1] @ w[2])]), Comb([(-0.5, w[2] @ w[0])]), Comb([(-0.5, w[0] @ w[1])])]
PmW = Comb([(0.5, Isp), (-0.5, chiW)])
PpW = Comb([(0.5, Isp), (0.5, chiW)])
tL = [PmW @ t for t in tau]
tR = [PpW @ t for t in tau]
Hc = Comb([(-1.0, E[10] @ E[16]), (-1.0, E[11] @ E[17]), (-1.0, E[12] @ E[18])])
tY = tR[2] + Hc * (1.0 / 6)
Bem = tau[2] + Hc * (1.0 / 6)
QA = Comb([(0.5, qA)])


def lift(M, idx):
    sgn = 1.0 if eta[idx[0]] > 0 else -1.0
    return Comb([(sgn * M[p_, q_] * 0.5, E[idx[p_]] @ E[idx[q_]]) for p_, q_ in itertools.combinations(range(len(idx)), 2) if M[p_, q_] != 0])


# colour su(3) as the derived centraliser of H_C in so(6)_C (as in p1), orthonormal in the unit-rate form
rhoH = vec_rep(Hc, E, Vm)
so6 = []
for p_, q_ in itertools.combinations(range(6), 2):
    M = np.zeros((6, 6))
    M[p_, q_], M[q_, p_] = 1.0, -1.0
    so6.append(M)
nsu = null_space(np.array([(M @ rhoH - rhoH @ M).ravel() for M in so6]).T)
u3 = [sum(nsu[k_, m_] * so6[k_] for k_ in range(15)) for m_ in range(nsu.shape[1])]
hn = rhoH / np.linalg.norm(rhoH)
su3v = orth_basis([M - np.sum(M * hn) * hn for M in u3])
su3v = [M * np.sqrt(2.0) / np.linalg.norm(M) for M in su3v]  # -1/2 tr M^2 = 1
su3 = [lift(M, Vm) for M in su3v]

soA = [unit(i, j) for i, j in itertools.combinations(Aidx, 2)]
soW = [unit(i, j) for i, j in itertools.combinations(Widx, 2)]
soC = [unit(i, j) for i, j in itertools.combinations(Cidx, 2)]
k_units = [unit(i, j) for i, j in kbiv]

# the P_*-surviving algebra of ES17 Thm 17.22(ii): rot(i,j) = e_{i'}e_{j'} - e_ie_j, i' = i+3
def rot(i, j):
    return Comb([(1.0, E[i + 3] @ E[j + 3]), (-1.0, E[i] @ E[j])])


so4D = [rot(i, j) * (1 / (2 * np.sqrt(2))) for i, j in itertools.combinations(Widx, 2)]
rot89 = rot(8, 9) * (1 / (2 * np.sqrt(2)))
rot23 = rot(2, 3) * (1 / (2 * np.sqrt(2)))


def rho_plus(X):
    return vec_rep(X, E, Vp)


iA = [Vp.index(i) for i in Aidx]
iW = [Vp.index(i) for i in Widx]
pA8 = np.zeros((8, 8))
pA8[iA, iA] = 1.0
pW8 = np.eye(8) - pA8
U_A = [Vp.index(2), Vp.index(3)]
U_em = [Vp.index(7), Vp.index(13)]

# ---- the gauged subalgebras: list of ideals (name, generators); one coupling per ideal
ALG = {
    "u1A": [("A", [QA])],
    "u1em": [("em", [Bem])],
    "u1em+su3": [("em", [Bem]), ("3", su3)],
    "SM": [("3", su3), ("2", tL), ("Y", [tY])],
    "PS": [("L", tL), ("R", tR), ("4", soC)],
    "PS+u1A": [("L", tL), ("R", tR), ("4", soC), ("A", [QA])],
    "s*": [("D", so4D), ("89", [rot89])],
}
RHO = {name: [(lab, [rho_plus(X) for X in gens]) for lab, gens in ideals] for name, ideals in ALG.items()}


def rand_couplings(name):
    return {lab: float(RNG.uniform(0.4, 1.6)) for lab, _ in ALG[name]}


def V512(name, g, Pi):
    return sum(g[lab] ** 2 * sum(float(np.sum(dcm(X, Pi) ** 2)) for X in gens) for lab, gens in ALG[name])


def V8(name, g, p):
    return 64.0 * sum(g[lab] ** 2 * sum(float(np.sum((x @ p - p @ x) ** 2)) for x in xs) for lab, xs in RHO[name])


def frame_vol(F, idx):
    out = None
    for r in range(F.shape[0]):
        u = Comb([(F[r, s], E[idx[s]]) for s in range(len(idx)) if F[r, s] != 0])
        out = u if out is None else out @ u
    return out


def rand_frame():
    return np.linalg.qr(RNG.normal(size=(8, 4)))[0].T


def point(F):
    vP = frame_vol(F, Vp).dense()
    return 0.5 * (Id + vP), F.T @ F


def angles2(p, U):
    """squared cosines of the principal angles between P (projector p) and the coordinate plane U."""
    return np.sort(np.linalg.eigvalsh(p[np.ix_(U, U)]))


def f_weak(p):
    g_ = np.linalg.eigvalsh(p[np.ix_(iW, iW)])
    return 0.75 * np.sum(g_) - 0.5 * sum(g_[k] * g_[l] for k in range(4) for l in range(k + 1, 4))


JY = 2.0 * rho_plus(tY)[np.ix_(iW, iW)]


def y_hyp(p):
    M = p[np.ix_(iW, iW)]
    return 0.25 * (np.trace(M) - np.trace(JY.T @ M @ JY @ M))


def s_plane(p, U):
    c = angles2(p, U)
    return c[0] + c[1] - 2 * c[0] * c[1]


# =============================================================================================== 1. mass matrix
ok = all((E[i] @ E[i]) == (Isp if eta[i] > 0 else -Isp) for i in range(1, 19))
ok = ok and all(anticommutes(E[i], E[j]) for i in range(1, 19) for j in range(i + 1, 19))
om8_dense = e8_dense[0] @ e8_dense[1] @ e8_dense[2] @ e8_dense[3] @ e8_dense[4] @ e8_dense[5]
ok = ok and all(np.array_equal(E[6 * (k - 1) + a + 1].dense(), _kron(*([om8_dense] * (k - 1) + [e8_dense[a]] + [np.eye(8, dtype=np.int64)] * (3 - k))))
                for k in (1, 2, 3) for a in range(6))
ok = ok and sorted(Aidx + Widx) == Vp and Cidx == Vm and len(kbiv) == 43
G43 = np.array([[ip(X, Y) for Y in k_units] for X in k_units])
ok = ok and np.allclose(G43, np.eye(43))
ok = ok and all(np.allclose(rho_plus(unit(i, j)), np.eye(8)[:, [Vp.index(i)]] @ np.eye(8)[[Vp.index(j)], :] - np.eye(8)[:, [Vp.index(j)]] @ np.eye(8)[[Vp.index(i)], :]) for i, j in kpos_pairs[:6])
norms = (ip(QA, QA), ip(Bem, Bem), [ip(t, t) for t in tL + tR], ip(tY, tY), [ip(x, x) for x in so4D], ip(rot89, rot89))
ok = ok and abs(norms[0] - 1) < TOL and abs(norms[1] - 4 / 3) < TOL and np.allclose(norms[2], 0.5) and abs(norms[3] - 5 / 6) < TOL
ok = ok and np.allclose(norms[4], 1) and abs(norms[5] - 1) < TOL
ok = ok and np.allclose([[ip(X, Y) for Y in su3] for X in su3], np.eye(8)) and all(iszero(cm(X, Hc)) for X in su3)
ok = ok and all(abs(ip(X, Y)) < TOL for X in tL for Y in tR) and all(abs(ip(X, tY) - (0.5 if X is tR[2] else 0)) < TOL for X in tL + tR)
report(1, ok, "n=3 realisation (E_1..E_18 = the Kronecker products of ES17 App. A, Clifford relations of R^{9,9}); in <X,Y>=-Tr(XY)/128 the 43 unit-rate rotations e_ie_j/2 are orthonormal with rho(e_ie_j/2)=E_ij-E_ji; |Q_A|^2=1, |B_em|^2=4/3, |t^{L,R}_i|^2=1/2, |t_Y|^2=5/6, rot(i,j)/(2 sqrt2) orthonormal, su(3)_C basis orthonormal")

# mass matrix at random points
ok_psd, ok_tr, ok_ker, ok_sym = True, True, True, True
for name in ("SM", "PS+u1A", "u1em"):
    g = rand_couplings(name)
    gens = [(g[lab], X) for lab, Xs in ALG[name] for X in Xs]
    for Pi in (point(rand_frame())[0], PiA_d):
        C = [dcm(X, Pi) for _, X in gens]
        ok_sym = ok_sym and all(close(c_, c_.T) for c_ in C)
        M2 = np.array([[ga * gb * np.sum(ca * cb.T) for (gb, _), cb in zip(gens, C)] for (ga, _), ca in zip(gens, C)])
        ev = np.linalg.eigvalsh(M2)
        ok_psd = ok_psd and np.allclose(M2, M2.T) and ev[0] > -1e-7
        ok_tr = ok_tr and abs(np.trace(M2) - V512(name, g, Pi)) < 1e-6 * max(1, np.trace(M2))
        nker = int(np.sum(ev < 1e-7 * max(1.0, ev[-1])))
        stab = len(gens) - span_rank(C)
        ok_ker = ok_ker and nker == stab
report(2, ok_psd and ok_tr and ok_ker and ok_sym, "n=3 (f=1): for h_SM, h_PS+u(1)_A and u(1)_em with random couplings, at random points of F_3 and at Pi_A, the [T_a,Pi] are symmetric, M^2_ab=g_ag_bTr([T_a,Pi][T_b,Pi]) is symmetric positive semidefinite, Tr M^2 = V_h(Pi), and dim ker M^2 = dim of the stabiliser of Pi in h (massless vectors = unbroken generators)")

# invariance under the group generated by h (non-abelian cases)
ok = True
for name in ("SM", "PS+u1A"):
    g = rand_couplings(name)
    gens = [X for _, Xs in ALG[name] for X in Xs]
    for _ in range(2):
        Pi, _p = point(rand_frame())
        X = Comb([])
        for Xg in gens:
            X = X + Comb.of(Xg) * float(RNG.normal())
        U = expm(X.dense())
        v0, v1 = V512(name, g, Pi), V512(name, g, U @ Pi @ U.T)
        ok = ok and close(U @ U.T, Id, 1e-9) and abs(v1 - v0) < 1e-8 * v0 and v0 > 1
report(3, ok, "n=3: V_h(g Pi g^-1) = V_h(Pi) for g = exp(X) with X a random element of h, h = h_SM and h_PS+u(1)_A with unequal couplings, at random points of F_3 (invariance under the group generated by h)")
