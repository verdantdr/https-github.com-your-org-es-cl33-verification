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

# vector-model reduction (p1:lem:vector) for every subalgebra of the list
ok = True
for name in ALG:
    g = rand_couplings(name)
    for _ in range(2):
        F = rand_frame()
        Pi, p = point(F)
        a, b = V512(name, g, Pi), V8(name, g, p)
        ok = ok and abs(a - b) < 1e-8 * max(1.0, a) and a > 1e-3
report(4, ok, "n=3: for every h in {u(1)_A, u(1)_em, u(1)_em+su(3)_C, h_SM, h_PS, h_PS+u(1)_A, s_*} with random couplings, at random points of F_3, V_h(Pi_P) computed on R^512 equals 64 sum_a g_a^2 ||[rho_+T_a,p_P]||^2 computed on R^8 (the colour parts drop out)")

# ============================================================================== Coleman-Weinberg: the cutoff integral
ok = True
details = []
for r in (1e-1, 3e-2, 1e-2):
    Lam, m2 = 1.0, r ** 2
    exact = quad(lambda k: k ** 3 * np.log1p(m2 / k ** 2), 0, Lam, epsabs=1e-16, epsrel=1e-13, limit=200)[0] / (16 * np.pi ** 2)
    closed = (0.5 * (0.5 * Lam ** 4 * np.log1p(m2 / Lam ** 2) + 0.5 * m2 * (Lam ** 2 - m2 * np.log1p(Lam ** 2 / m2)))) / (16 * np.pi ** 2)
    approx = Lam ** 2 * m2 / (32 * np.pi ** 2) + m2 ** 2 / (64 * np.pi ** 2) * (np.log(m2 / Lam ** 2) - 0.5)
    rem = (exact - approx) / (m2 ** 2 / (64 * np.pi ** 2))
    details.append(rem / (m2 / Lam ** 2))
    ok = ok and abs(exact - closed) < 1e-12 * abs(exact) + 1e-18 and abs(rem) < 2 * m2 / Lam ** 2 * 64 * np.pi ** 2 / (16 * np.pi ** 2)
ok = ok and np.allclose(details, details[-1], rtol=0.05)
report(5, ok, f"inherited (Coleman-Weinberg, Euclidean cutoff): per real degree of freedom (1/2)int^Lambda d^4k/(2pi)^4 log(1+m^2/k^2) = Lambda^2 m^2/(32pi^2) + m^4/(64pi^2)(log(m^2/Lambda^2)-1/2) + O(m^6/Lambda^2) (quadrature against the closed form; remainder/(m^4/64pi^2) = {details[-1]:.3f} m^2/Lambda^2); times 3 polarisations for a vector in Landau gauge")

# ============================================================================== 2. the full alignment energy is constant
ok = True
spec_seen = []
for trial in range(3):
    Pi, p = point(rand_frame()) if trial else (PiA_d, pA8)
    C = [dcm(X, Pi) for X in k_units]
    G = np.array([[np.sum(a_ * b_) for b_ in C] for a_ in C])
    g8, g6 = RNG.uniform(0.5, 1.5, size=2)
    gv = np.array([g8] * 28 + [g6] * 15)
    M2 = gv[:, None] * gv[None, :] * G
    ev = np.sort(np.linalg.eigvalsh(M2))
    ok = ok and np.allclose(ev[:27], 0, atol=1e-7) and np.allclose(ev[27:], 128 * g8 ** 2, rtol=1e-9)
    ok = ok and abs(np.trace(M2) - 2048 * g8 ** 2) < 1e-6 and np.allclose(G[28:, :], 0) and np.allclose(G[:, 28:], 0)
    spec_seen.append((round(ev[-1] / g8 ** 2, 6), int(np.sum(ev > 1e-6))))
report(6, ok, f"n=3: with h=k (all 43 compact internal bivectors) and Ad(K)-invariant couplings g_8 (so(8)), g_6 (so(6)), the mass matrix M^2 at Pi_A and at random points of F_3 has spectrum {{128 g_8^2 (x16), 0 (x27)}} (f=1): Tr M^2 = V_k = 2048 g_8^2, independent of g_6 and of the point, so every spectral function of M^2, the full one-loop potential included, is constant on F_3 (computed (eig/g_8^2, rank) = {spec_seen})")

# comparison with the normalisation of ES17 Prop. 21.6
Pd, PLd, PRd = Pst.dense(), PL.dense(), PR.dense()
blade_vals = {}
for lab, P_ in (("PiA", PiA_d), ("P*", Pd), ("PL", PLd), ("PR", PRd)):
    v = np.array([np.sum(dcm(E[i] @ E[j], P_) ** 2) for i, j in kbiv])
    rk = int(round(np.trace(P_)))
    blade_vals[lab] = (rk, sorted(set(np.round(v, 6).tolist())), int(np.sum(v > 1e-6)), v.sum() / 512, v.sum() / (2 * min(rk, 512 - rk)))
ok = (blade_vals["PiA"][1:] == ([0.0, 512.0], 16, 16.0, 16.0) and blade_vals["P*"][1:] == ([0.0, 128.0], 20, 5.0, 20.0)
      and blade_vals["PL"][1:] == ([0.0, 64.0], 20, 2.5, 20.0) and blade_vals["PR"][1:] == ([0.0, 64.0], 20, 2.5, 20.0))
report(7, ok, f"n=3, normalisation of ES17 Prop. 21.6: (1/512) sum_43 ||[e_ie_j,P]||^2 = 16 on F_3 (restated p1-34), 5 at P_*, 5/2 at P_L and P_R (restated); blade by blade every moving blade moves maximally, ||[e_ie_j,P]||^2 = 2 min(rk P, 512 - rk P): 16 movers at Pi_A, 20 at P_*, P_L, P_R, so (1/(2 rk P)) sum = number of broken blades = 16 on F_3 and 20 at P_* (computed (rank, values, movers, ES17 norm, rank norm) = {blade_vals})")

# ============================================================================== 3. calibration at the second rung
E2, eta2 = tower(2)
N2 = 64
Id2 = np.eye(N2)
I2sp = ident(N2)
pos2, neg2 = [2, 3, 7, 8, 9], [10, 11, 12]
a2, b2, c2, d2, e2 = (E2[i] for i in pos2)
I5 = a2 @ b2 @ c2 @ d2 @ e2
Z5 = [I5 @ v for v in (a2, b2, c2, d2, e2)]
Z5d = [z.dense() for z in Z5]
Q2 = Comb([(0.5, a2 @ b2)])
k2pairs = list(itertools.combinations(pos2, 2)) + list(itertools.combinations(neg2, 2))
k2 = [Comb([(0.5, E2[i] @ E2[j])]) for i, j in k2pairs]
so3cde = [Comb([(0.5, E2[i] @ E2[j])]) for i, j in itertools.combinations([7, 8, 9], 2)]
ok = all(abs(-(X @ X).trace() / 16 - 1) < TOL for X in k2) and abs(-(Q2 @ Q2).trace() / 16 - 1) < TOL


def Pi2(n):
    return 0.5 * (Id2 + sum(n[i] * Z5d[i] for i in range(5)))


def vrep2(X):
    return vec_rep(X, E2, pos2)


rho2Q = vrep2(Q2)
rho2_3 = [vrep2(X) for X in so3cde]


def V2_vm(n, gQ, g3):
    p = np.eye(5) - np.outer(n, n)
    return 8.0 * (gQ ** 2 * np.sum((rho2Q @ p - p @ rho2Q) ** 2) + g3 ** 2 * sum(np.sum((x @ p - p @ x) ** 2) for x in rho2_3))


ok_c, ok_Q, ok_vm, ok_2c = True, True, True, True
for _ in range(4):
    n = RNG.normal(size=5)
    n /= np.linalg.norm(n)
    P2 = Pi2(n)
    C = [dcm(X, P2) for X in k2]
    G = np.array([[np.sum(x_ * y_) for y_ in C] for x_ in C])
    g5, g3m = RNG.uniform(0.5, 1.5, size=2)
    gv = np.array([g5] * 10 + [g3m] * 3)
    ev = np.sort(np.linalg.eigvalsh(gv[:, None] * gv[None, :] * G))
    ok_c = ok_c and np.allclose(ev[:9], 0, atol=1e-8) and np.allclose(ev[9:], 16 * g5 ** 2) and abs(np.sum(ev) - 64 * g5 ** 2) < 1e-8
    gQ, g3 = RNG.uniform(0.5, 1.5, size=2)
    VQ = gQ ** 2 * np.sum(dcm(Q2, P2) ** 2)
    ok_Q = ok_Q and abs(VQ - 16 * gQ ** 2 * (n[0] ** 2 + n[1] ** 2)) < 1e-9
    V3 = g3 ** 2 * sum(np.sum(dcm(X, P2) ** 2) for X in so3cde)
    rho = n[0] ** 2 + n[1] ** 2
    ok_2c = ok_2c and abs(VQ + V3 - (16 * gQ ** 2 * rho + 32 * g3 ** 2 * (1 - rho))) < 1e-9
    ok_vm = ok_vm and abs(VQ + V3 - V2_vm(n, gQ, g3)) < 1e-9
    X = k2[int(RNG.integers(13))] * 1.0 + k2[int(RNG.integers(13))] * 0.3
    Y = k2[int(RNG.integers(13))]
    xv, yv = vrep2(X), vrep2(Y)
    p = np.eye(5) - np.outer(n, n)
    ok_vm = ok_vm and abs(np.sum(dcm(X, P2) * dcm(Y, P2)) - 8 * np.trace((xv @ p - p @ xv) @ (yv @ p - p @ yv))) < 1e-9
report(8, ok and ok_c and ok_Q and ok_vm and ok_2c, "n=2 (f=1): at random points Pi(n) of S^4, the full spectator algebra so(5)+so(3) with couplings g_5, g_3' gives M^2 spectrum {16 g_5^2 (x4), 0 (x9)} and V=64 g_5^2 (constant); V_Q = 16 g_Q^2 (n_a^2+n_b^2) = 16 g_Q^2|phi|^2 for Q=ab/2; with so(3)_cde added, V = 16 g_Q^2 rho + 32 g_3^2 (1-rho), rho = n_a^2+n_b^2; Tr([X,Pi(n)][Y,Pi(n)]) = 8 tr_5([x,p][y,p]) with p = 1-nn^T")

# minimum set of V_Q, residual symmetry, Hessian at Pi_0 = Pi(e)
Pi0 = Pi2(np.array([0, 0, 0, 0, 1.0]))
T2 = [Comb([(0.5, Z5[i])]) for i in range(4)]
ok = all(close(dcm(Comb([(0.5, (a2, b2, c2, d2)[i] @ e2)]), Pi0), T2[i].dense()) for i in range(4))
rhoT = np.array([[(T2[r] @ cm(Q2, T2[c_])).trace() / 16 for c_ in range(4)] for r in range(4)])
HQ = frac_matrix(-rhoT @ rhoT)
ok = ok and HQ == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
cent = [X for X in k2 if iszero(cm(X, Q2))]
stab = [X for X in cent if iszero(dcm(X, Pi0))]
gold = span_rank([dcm(X, Pi0) for X in cent])
gold_in_cd = all(abs(np.sum(dcm(X, Pi0) * T2[0].dense())) < TOL and abs(np.sum(dcm(X, Pi0) * T2[1].dense())) < TOL for X in cent)
ok = ok and len(cent) == 7 and len(stab) == 5 and gold == 2 and gold_in_cd
# SO(3)_cde acts on n by rotations of (c,d,e): transitive on the uncharged two-sphere
for X in so3cde:
    th = RNG.uniform(0.2, 2.5)
    g = expm(th * X.dense())
    R = expm(th * vrep2(X))
    n = np.array([0, 0, *RNG.normal(size=3)])
    n /= np.linalg.norm(n)
    ok = ok and close(g @ Pi2(n) @ g.T, Pi2(R @ n), 1e-9) and abs((R @ n)[0]) < TOL and abs((R @ n)[1]) < TOL
# numerical minimisation of V_Q over S^4
ends = []
for _ in range(30):
    gQ = RNG.uniform(0.5, 1.5)
    res = minimize(lambda v: V2_vm(v / np.linalg.norm(v), gQ, 0.0), RNG.normal(size=5), method="BFGS", options={"gtol": 1e-12})
    nn_ = res.x / np.linalg.norm(res.x)
    ends.append((res.fun, nn_[0] ** 2 + nn_[1] ** 2))
ok = ok and max(e_[0] for e_ in ends) < 1e-10 and max(e_[1] for e_ in ends) < 1e-8
report(9, ok, "n=2: the minimum set of V_Q on S^4 is the uncharged two-sphere n_a=n_b=0 (30 random starts all reach V<1e-10 there), a single orbit of SO(3)_cde (exp(theta X) Pi(n) exp(-theta X) = Pi(Rn)); at Pi_0 the Hessian form -(ad Q)^2 on the tangent T_i is exactly diag(1,1,0,0): the charged phi has eigenvalue g_Q^2 and the neutral T_c, T_d are zero modes; the centraliser of Q in k_2 has dim 7, its stabiliser at Pi_0 dim 5, and [centraliser, Pi_0] = span(T_c,T_d): two Goldstones of so(3)_cde -> so(2)_cd")

# two couplings: phase boundary
ok = True
summ = {}
for ratio in (0.5, 0.70, 0.7072, 1.0):
    gQ = 1.0
    g3 = ratio * gQ
    vals = []
    for _ in range(25):
        res = minimize(lambda v: V2_vm(v / np.linalg.norm(v), gQ, g3), RNG.normal(size=5), method="BFGS", options={"gtol": 1e-12})
        nn_ = res.x / np.linalg.norm(res.x)
        vals.append((res.fun, nn_[0] ** 2 + nn_[1] ** 2))
    vmin = min(v_[0] for v_ in vals)
    rhos = [v_[1] for v_ in vals if v_[0] < vmin + 1e-9]
    summ[ratio] = (round(float(vmin), 6), round(float(np.mean(rhos)), 6))
    if ratio ** 2 < 0.5:
        ok = ok and abs(vmin - 32 * g3 ** 2) < 1e-8 and max(rhos) < 1e-6
    else:
        ok = ok and abs(vmin - 16 * gQ ** 2) < 1e-8 and min(rhos) > 1 - 1e-6
nb = RNG.normal(size=(6, 5))
flat = [V2_vm(v / np.linalg.norm(v), 1.0, np.sqrt(0.5)) for v in nb]
ok = ok and np.allclose(flat, 16.0)
# Hessians off the zero set, from V(n): at Pi(e) on (n_a,n_b) and at Pi(a) towards c,d,e, relative to Tr(dPi^2)=16|dn|^2
gQ, g3 = 1.3, 0.6
h = 1e-4
fe = lambda t, i: V2_vm(np.cos(t) * np.eye(5)[4] + np.sin(t) * np.eye(5)[i], gQ, g3)
fa = lambda t, i: V2_vm(np.cos(t) * np.eye(5)[0] + np.sin(t) * np.eye(5)[i], gQ, g3)
hess_e = [(fe(h, i) - 2 * fe(0, i) + fe(-h, i)) / h ** 2 / 2 / 16 for i in range(4)]
hess_a = [(fa(h, i) - 2 * fa(0, i) + fa(-h, i)) / h ** 2 / 2 / 16 for i in range(1, 5)]
ok = ok and np.allclose(hess_e, [gQ ** 2 - 2 * g3 ** 2] * 2 + [0, 0], atol=1e-5) and np.allclose(hess_a, [0] + [2 * g3 ** 2 - gQ ** 2] * 3, atol=1e-5)
# vector masses at the two minimum sets (f=1)
gens2 = [(gQ, Q2)] + [(g3, X) for X in so3cde]
for nvec, expect in ((np.eye(5)[4], [0, 0, 16 * g3 ** 2, 16 * g3 ** 2]), (np.eye(5)[0], [0, 0, 0, 16 * gQ ** 2])):
    C = [dcm(X, Pi2(nvec)) for _, X in gens2]
    M2 = np.array([[ga * gb * np.sum(ca * cb) for (gb, _), cb in zip(gens2, C)] for (ga, _), ca in zip(gens2, C)])
    ok = ok and np.allclose(np.sort(np.linalg.eigvalsh(M2)), sorted(expect), atol=1e-9)
report(10, ok, f"n=2 with h=u(1)_Q+so(3)_cde: global minimisation over S^4 gives the uncharged S^2 (V=32 g_3^2, photon massless, so(3)->so(2), vector masses^2 16 g_3^2 (x2)) for g_3^2<g_Q^2/2 and the charged circle n in span(a,b) (V=16 g_Q^2, so(3) unbroken, photon mass^2 16 g_Q^2) for g_3^2>g_Q^2/2; V is constant on S^4 at g_3^2=g_Q^2/2; Hessians relative to Tr(dPi^2): g_Q^2-2g_3^2 on phi at Pi(e), 2g_3^2-g_Q^2 on the so(3) triplet at Pi(a) (computed (V_min, <rho>) for g_3/g_Q: {summ})")

# ============================================================================== 4. the third rung: closed forms
rA, rem_ = rho_plus(QA), rho_plus(Bem)
xL = [rho_plus(t) for t in tL]
xR = [rho_plus(t) for t in tR]
xY = rho_plus(tY)
ok = np.allclose(rA, np.eye(8)[:, [U_A[1]]] @ np.eye(8)[[U_A[0]], :] - np.eye(8)[:, [U_A[0]]] @ np.eye(8)[[U_A[1]], :]) or \
    np.allclose(-rA, np.eye(8)[:, [U_A[1]]] @ np.eye(8)[[U_A[0]], :] - np.eye(8)[:, [U_A[0]]] @ np.eye(8)[[U_A[1]], :])
ok = ok and np.allclose(rA @ rA, -np.diag([1.0 if i in U_A else 0.0 for i in range(8)]))
ok = ok and np.allclose(rem_ @ rem_, -np.diag([1.0 if i in U_em else 0.0 for i in range(8)]))
for _ in range(20):
    F = rand_frame()
    p = F.T @ F
    for x, U in ((rA, U_A), (rem_, U_em)):
        c = angles2(p, U)
        ok = ok and abs(np.sum((x @ p - p @ x) ** 2) - 2 * (c[0] + c[1] - 2 * c[0] * c[1])) < 1e-10
        M = p[np.ix_(U, U)]
        ok = ok and abs((c[0] + c[1] - 2 * c[0] * c[1]) - (np.trace(M) - 2 * np.linalg.det(M))) < 1e-12
report(11, ok, "n=3 (lemma, two-plane charges): rho_+(Q_A) and rho_+(B_em) are the unit-rate rotations of U_A=span(E2,E3) and U_em=span(E7,E13); at random P, ||[j_U,p]||^2 = 2(c_1+c_2-2c_1c_2) = 2(tr p_U - 2 det p_U), c_i the squared cosines of the principal angles between P and U")

ok = True
for xs in (xL, xR):
    Ls = [2 * x[np.ix_(iW, iW)] for x in xs]
    ok = ok and all(np.allclose(x[np.ix_(iA, iA)], 0) and np.allclose(x[np.ix_(iA, iW)], 0) for x in xs)
    ok = ok and all(np.allclose(L @ L, -np.eye(4)) for L in Ls) and np.allclose(Ls[0] @ Ls[1], Ls[2]) and np.allclose(Ls[0] @ Ls[1], -Ls[1] @ Ls[0])
    for _ in range(3):
        Ms = RNG.normal(size=(4, 4))
        Ms = Ms + Ms.T
        ok = ok and np.allclose(sum(L @ Ms @ L.T for L in Ls), np.trace(Ms) * np.eye(4) - Ms)
LsL = [2 * x[np.ix_(iW, iW)] for x in xL]
ok = ok and np.allclose(JY @ JY, -np.eye(4)) and all(np.allclose(JY @ L, L @ JY) for L in LsL) and np.allclose(xY[np.ix_(iA, iA)], 0)
ok = ok and np.allclose(xY, xR[2])
for _ in range(20):
    p = (lambda F: F.T @ F)(rand_frame())
    gw = np.linalg.eigvalsh(p[np.ix_(iW, iW)])
    fv = f_weak(p)
    ok = ok and abs(sum(np.sum((x @ p - p @ x) ** 2) for x in xL) - 2 * fv) < 1e-10
    ok = ok and abs(sum(np.sum((x @ p - p @ x) ** 2) for x in xR) - 2 * fv) < 1e-10
    ok = ok and abs(fv - 0.25 * sum(gw[k] * (3 - (np.sum(gw) - gw[k])) for k in range(4))) < 1e-12 and fv > 0
    yv = y_hyp(p)
    ok = ok and abs(np.sum((xY @ p - p @ xY) ** 2) - 2 * yv) < 1e-10 and yv >= -1e-14
    ok = ok and abs(V8("PS", {"L": 1.0, "R": 0.0, "4": 0.0}, p) - 128 * fv) < 1e-8
ok = ok and abs(f_weak(pA8)) < TOL and abs(f_weak(pW8)) < TOL and abs(y_hyp(pA8)) < TOL and abs(y_hyp(pW8)) < TOL
report(12, ok, "n=3 (lemma, weak isospin and hypercharge): rho(t^L_i), rho(t^R_i) vanish on A and 2rho(t^{L,R}_i) are quaternion units on W (L_1L_2=L_3, anticommuting complex structures) with sum_i L_i M L_i^T = (tr M)1 - M for symmetric M; hence sum_i||[rho t^L_i,p]||^2 = sum_i||[rho t^R_i,p]||^2 = 2f(P), f = (3/4)sum gamma_k - (1/2)sum_{k<l} gamma_k gamma_l = (1/4)sum_k gamma_k(3 - sum_{l!=k} gamma_l) >= 0, gamma_k the squared cosines between P and W; ||[rho t_Y,p]||^2 = 2y(P), y = (1/4)(tr p_W - tr(J_Y^T p_W J_Y p_W)) >= 0, J_Y = 2rho(t_Y)|_W = 2rho(t^R_3)|_W a complex structure commuting with su(2)_L; f and y vanish at A and W")

ok = True
for _ in range(10):
    p = (lambda F: F.T @ F)(rand_frame())
    s89 = s_plane(p, [Vp.index(8), Vp.index(9)])
    fv, yv, sA, sem = f_weak(p), y_hyp(p), s_plane(p, U_A), s_plane(p, U_em)
    gg = {lab: float(RNG.uniform(0.4, 1.6)) for lab in ("A", "em", "3", "2", "Y", "L", "R", "4", "D", "89")}
    pred = {"u1A": 128 * gg["A"] ** 2 * sA, "u1em": 128 * gg["em"] ** 2 * sem, "u1em+su3": 128 * gg["em"] ** 2 * sem,
            "SM": 128 * (gg["2"] ** 2 * fv + gg["Y"] ** 2 * yv), "PS": 128 * (gg["L"] ** 2 + gg["R"] ** 2) * fv,
            "PS+u1A": 128 * ((gg["L"] ** 2 + gg["R"] ** 2) * fv + gg["A"] ** 2 * sA), "s*": 128 * (2 * gg["D"] ** 2 * fv + 0.5 * gg["89"] ** 2 * s89)}
    for name in ALG:
        ok = ok and abs(V8(name, gg, p) - pred[name]) < 1e-8 * max(1, pred[name])
Fr = rand_frame()
Pi, p = point(Fr)
gg = {lab: float(RNG.uniform(0.4, 1.6)) for lab in ("2", "Y", "3")}
ok = ok and abs(V512("SM", gg, Pi) - 128 * (gg["2"] ** 2 * f_weak(p) + gg["Y"] ** 2 * y_hyp(p))) < 1e-8 * V512("SM", gg, Pi)
report(13, ok, "n=3 closed forms at random points of F_3 (vector model, and R^512 for h_SM): V_A=128g_A^2 s_A, V_em=V_{em+su3}=128e^2 s_em, V_SM=128(g_2^2 f+g_Y^2 y), V_PS=128(g_L^2+g_R^2)f, V_{PS+A}=V_PS+128g_A^2 s_A, V_{s*}=128(2g_D^2 f+(1/2)g_89^2 s_89), with s_U=c_1+c_2-2c_1c_2 for the plane U (the colour couplings g_3, g_4 drop out)")

# ============================================================================== exact Hessians at Pi_A and Pi_W
mix = [(i, j) for i in Aidx for j in Widx]
TA = {m: Comb([(0.5, E[m[0]] @ E[m[1]] @ volA)]) for m in mix}
TW = {m: Comb([(0.5, E[m[0]] @ E[m[1]] @ volW)]) for m in mix}


def tangent_matrix(X, Td):
    keys = list(Td)
    M = np.zeros((16, 16))
    for c_, k_ in enumerate(keys):
        img = cm(X, Td[k_])
        for r_, k2_ in enumerate(keys):
            M[r_, c_] = (Td[k2_] @ img).trace() / 128.0
        rec = img - Comb([(M[r_, c_] * cc, S) for r_, k2_ in enumerate(keys) for cc, S in Td[k2_].terms])
        assert iszero(rec), "tangent image leaves the tangent span"
    return M


def proj(pred):
    return [[Fraction(1) if (i == j and pred(mix[i])) else Fraction(0) for j in range(16)] for i in range(16)]


def scal(F, a):
    return [[a * v for v in row] for row in F]


EXPECT = {"A": proj(lambda m: m[0] in (2, 3)), "em": proj(lambda m: m[1] in (7, 13)), "3": scal(proj(lambda m: True), 0),
          "4": scal(proj(lambda m: True), 0), "2": scal(proj(lambda m: True), Fraction(3, 4)), "Y": scal(proj(lambda m: True), Fraction(1, 4)),
          "L": scal(proj(lambda m: True), Fraction(3, 4)), "R": scal(proj(lambda m: True), Fraction(3, 4)),
          "D": scal(proj(lambda m: True), Fraction(3, 2)), "89": scal(proj(lambda m: m[0] in (8, 9)), Fraction(1, 2))}
CASIMIR = {}
ok = True
# rot(i,j) is dyadic; its orthonormal multiple rot/(2 sqrt2) contributes 1/8 of the exact block of rot
for lab, gens, scale in [("A", [QA], 1), ("em", [Bem], 1), ("3", su3, 1), ("2", tL, 1), ("Y", [tY], 1), ("L", tL, 1), ("R", tR, 1), ("4", soC, 1),
                         ("D", [rot(i, j) for i, j in itertools.combinations(Widx, 2)], Fraction(1, 8)), ("89", [rot(8, 9)], Fraction(1, 8))]:
    for vac, Td in (("A", TA), ("W", TW)):
        R = [tangent_matrix(X, Td) for X in gens]
        Cm = -sum(r_ @ r_ for r_ in R)
        Cf = scal(frac_matrix(np.round(Cm, 12)) if lab == "3" else frac_matrix(Cm), scale)
        ok = ok and Cf == EXPECT[lab] and all(np.allclose(r_, -r_.T) for r_ in R)
        CASIMIR[(lab, vac)] = Cf
report(14, ok, "n=3 (exact): on the 16-dim tangent space at Pi_A and at Pi_W (basis T_aw), the coupling-free Hessian blocks C_I = -sum_{a in I}(ad T_a)^2 are, exactly: u(1)_A: the projector onto span(E2,E3)(x)W (rank 8); u(1)_em: the projector onto A(x)span(E7,E13) (rank 8); su(3)_C and so(6)_C: 0; su(2)_L, su(2)_R: (3/4)1; u(1)_Y: (1/4)1; so(4)_Delta: (3/2)1; rot(8,9): (1/2) the projector onto span(E8,E9)(x)W; so the Hessian of V_h is sum_I g_I^2 C_I relative to Tr(dPi^2)")

# Hessian kernels = Goldstones of the residual global symmetry (centraliser of h in k)
def rho14(X):
    out = np.zeros((14, 14))
    out[:8, :8] = vec_rep(X, E, Vp)
    out[8:, 8:] = vec_rep(X, E, Vm)
    return out


K14 = [rho14(X) for X in k_units]
EXPECT_DIMS = {"u1A": (31, 23, 8), "u1em": (25, 17, 8), "u1em+su3": (17, 9, 8), "SM": (8, 8, 0), "PS": (6, 6, 0), "PS+u1A": (2, 2, 0), "s*": (3, 3, 0)}
ok = True
dims = {}
for name in ALG:
    H14 = [rho14(X) for _, Xs in ALG[name] for X in Xs]
    rows = np.vstack([np.array([(Kx @ Hx - Hx @ Kx).ravel() for Kx in K14]).T for Hx in H14])
    cent = null_space(rows)  # columns: coefficients on the 43 unit rotations
    stab_rows = np.array([(Kx[:8, :8] @ pA8 - pA8 @ Kx[:8, :8]).ravel() for Kx in K14]).T @ cent
    nstab = cent.shape[1] - np.linalg.matrix_rank(stab_rows, tol=1e-9)
    coords = np.array([[cent[kbiv.index(tuple(sorted(m))), c_] for m in mix] for c_ in range(cent.shape[1])])
    gold = np.linalg.matrix_rank(coords, tol=1e-9) if coords.size else 0
    g = rand_couplings(name)
    Hs = sum(float(g[lab] ** 2) * np.array(CASIMIR[(lab, "A")], dtype=float) for lab, _ in ALG[name])
    HsW = sum(float(g[lab] ** 2) * np.array(CASIMIR[(lab, "W")], dtype=float) for lab, _ in ALG[name])
    kerH = null_space(Hs, rcond=1e-10)
    same = (kerH.shape[1] == gold) and (gold == 0 or np.linalg.matrix_rank(np.hstack([kerH, coords.T]), tol=1e-8) == gold)
    dims[name] = (int(cent.shape[1]), int(nstab), int(gold))
    ok = ok and dims[name] == EXPECT_DIMS[name] and same and np.allclose(Hs, HsW) and np.linalg.eigvalsh(Hs)[0] > -1e-12
report(15, ok, f"n=3: the Hessian of V_h at Pi_A (equal to that at Pi_W) is positive semidefinite and its kernel is exactly [c_k(h),Pi_A], c_k(h) the centraliser of h in k: (dim c_k(h), dim of its stabiliser at Pi_A, Goldstones) = {dims}; u(1)_A, u(1)_em (with or without su(3)_C) leave 8 Goldstones, every h containing su(2)_L or so(4)_W leaves none")

# ============================================================================== global minimisation over F_3
def p_of(Fm):
    S = Fm.T @ Fm
    return Fm @ np.linalg.solve(S, Fm.T)


def V_and_grad(flat, xs_g):
    """V(p(F)) = 64 sum g^2 ||[x,p]||^2 for p = F (F^T F)^-1 F^T, and its gradient in the 8x4 matrix F."""
    Fm = flat.reshape(8, 4)
    S = Fm.T @ Fm
    Si = np.linalg.inv(S)
    p = Fm @ Si @ Fm.T
    V = 0.0
    G = np.zeros((8, 8))
    for gs, x in xs_g:
        c = x @ p - p @ x
        V += 64 * gs * np.sum(c * c)
        G += 128 * gs * (c @ x - x @ c)
    Kq = Si @ Fm.T @ G @ Fm @ Si
    grad = 2 * G @ Fm @ Si - 2 * Fm @ Kq
    return V, grad.ravel()


# gradient check
xs_test = [(1.3, x) for x in xL] + [(0.7, xY)]
f0 = RNG.normal(size=32)
V0, g0 = V_and_grad(f0, xs_test)
dv = RNG.normal(size=32)
fd = (V_and_grad(f0 + 1e-6 * dv, xs_test)[0] - V_and_grad(f0 - 1e-6 * dv, xs_test)[0]) / 2e-6
ok_grad = abs(fd - g0 @ dv) < 1e-5 * max(1, abs(fd))



def riem_grad(p, xs_g):
    G = sum(128 * gs * ((x @ p - p @ x) @ x - x @ (x @ p - p @ x)) for gs, x in xs_g)
    return float(np.linalg.norm(p @ G - G @ p))


def minimise_F3(xs_g, x0):
    """BFGS on frames, restarted with L-BFGS-B while the Riemannian gradient is not small (precision-loss stalls)."""
    res = minimize(V_and_grad, x0, args=(xs_g,), jac=True, method="BFGS", options={"gtol": 1e-11, "maxiter": 5000})
    xb, fb = res.x, res.fun
    for _ in range(4):
        p = p_of(xb.reshape(8, 4))
        if riem_grad(p, xs_g) < 1e-6 * max(1.0, sum(gs for gs, _ in xs_g)):
            break
        xb = xb / np.linalg.norm(xb) * np.sqrt(32)
        r2 = minimize(V_and_grad, xb, args=(xs_g,), jac=True, method="L-BFGS-B", options={"gtol": 1e-14, "ftol": 1e-16, "maxiter": 20000})
        xb, fb = r2.x, r2.fun
    p = p_of(xb.reshape(8, 4))
    return fb, p, riem_grad(p, xs_g)


def couplings_for(name):
    g = rand_couplings(name)
    if name == "SM":  # the regime g_Y < g_2 (the regime g_Y > g_2 is p2-17)
        g["Y"] = g["2"] * float(RNG.uniform(0.2, 0.95))
    return g


starts = 60
ok = ok_grad
glob = {}
for name in ALG:
    ends = []
    for _ in range(starts):
        g = couplings_for(name)
        xs_g = [(g[lab] ** 2, x) for lab, xs in RHO[name] for x in xs]
        fb, p, rg = minimise_F3(xs_g, RNG.normal(size=32))
        scale = 128 * max(g.values()) ** 2
        ends.append((fb / scale, p))
    vmax = max(e_[0] for e_ in ends)
    ok = ok and vmax < 1e-12
    if name in ("u1A", "u1em", "u1em+su3"):
        U = U_A if name == "u1A" else U_em
        comp = [int(round(np.trace(p[np.ix_(U, U)]))) for _, p in ends]
        ok = ok and set(comp) == {0, 2} and all(np.allclose(angles2(p, U), [c_ / 2] * 2, atol=1e-6) for (_, p), c_ in zip(ends, comp))
        glob[name] = (f"{vmax:.1e}", {"P contains U": comp.count(2), "P perp U": comp.count(0)})
    else:
        lab = [("A" if np.max(np.abs(p - pA8)) < 1e-6 else "W" if np.max(np.abs(p - pW8)) < 1e-6 else "?") for _, p in ends]
        ok = ok and "?" not in lab and {"A", "W"} <= set(lab)
        glob[name] = (f"{vmax:.1e}", {"A": lab.count("A"), "W": lab.count("W")})
report(16, ok, f"n=3: numerical global minimisation of V_h over F_3 (oriented frames in R^8, analytic gradient checked against finite differences, restarts until the Riemannian gradient is below 1e-6; {starts} random starts per h, random couplings in [0.4,1.6], for h_SM with g_Y/g_2 in [0.2,0.95]): every run reaches V_h/(128 g_max^2) < 1e-12, i.e. the zero set; for u(1)_A, u(1)_em, u(1)_em+su(3)_C the end points lie on the two components P contains U or P perp U (both reached), for h_SM, h_PS, h_PS+u(1)_A and s_* at the planes A or W (both reached, within 1e-6): no other local minima found (computed (max V, end points) = {glob})")

# ============================================================================== h_SM: the hypercharge-fixed component as a metastable set
def d2h(x, p, X, Y):
    """mixed second derivative at 0 of ||[x, e^{sX+tY} p e^{-sX-tY}]||^2."""
    c = x @ p - p @ x
    Xp, Yp = X @ p - p @ X, Y @ p - p @ Y
    XYp = X @ Yp - Yp @ X + Y @ Xp - Xp @ Y
    return 2 * np.trace((x @ Xp - Xp @ x) @ (x @ Yp - Yp @ x)) + np.trace(c @ (x @ XYp - XYp @ x))


iP0 = [Vp.index(i) for i in (2, 3, 7, 13)]
iQ0 = [Vp.index(i) for i in (8, 9, 14, 15)]
p0 = np.diag([1.0 if i in iP0 else 0.0 for i in range(8)])
dirs = []
dlab = []
for i in iP0:
    for j in iQ0:
        X = np.zeros((8, 8))
        X[j, i], X[i, j] = 1.0, -1.0
        dirs.append(X)
        dlab.append((Vp[i], Vp[j]))
ok = all(abs(np.trace((x @ p0 - p0 @ x) @ (x @ (X @ p0 - p0 @ X) - (X @ p0 - p0 @ X) @ x))) < TOL for x in xL + [xY] for X in dirs)
ok = ok and all(abs(64 * np.sum((X @ p0 - p0 @ X) ** 2) - 128) < TOL for X in dirs)
B2 = frac_matrix(np.array([[0.5 * sum(d2h(x, p0, X, Y) for x in xL) for Y in dirs] for X in dirs]))
BY = frac_matrix(np.array([[0.5 * d2h(xY, p0, X, Y) for Y in dirs] for X in dirs]))
mixAW = [k_ for k_, (i, j) in enumerate(dlab) if (i in Aidx) != (j in Aidx)]
ok = ok and all(B2[k_][l_] == (Fraction(-1, 2) if (k_ == l_ and k_ in mixAW) else 0) for k_ in range(16) for l_ in range(16))
ok = ok and all(BY[k_][l_] == Fraction(1, 2) * (k_ == l_) for k_ in mixAW for l_ in range(16))
BYf = np.array(BY, dtype=float)
evs = {}
for rr in (0.5, 2.0):
    lam = np.sort(np.linalg.eigvalsh(np.array(B2, dtype=float) + rr * BYf))
    evs[rr] = np.round(lam, 9).tolist()
    ok = ok and np.allclose(lam, sorted([0.0] * 6 + [(rr - 1) / 2] * 8 + [2 * rr] * 2))
# CS family: along P(phi) with fixed principal frames V_SM is multilinear in gamma = cos^2 phi
F = rand_frame()
p = F.T @ F
gam_, Wv = np.linalg.eigh(p[np.ix_(iW, iW)])
wk = np.zeros((8, 4))
wk[iW, :] = Wv
ak = np.zeros((8, 4))
for k_ in range(4):
    v = p @ wk[:, k_]
    a_ = v.copy()
    a_[iW] = 0
    ak[:, k_] = a_ / np.linalg.norm(a_)
ok_cs = np.allclose(ak.T @ ak, np.eye(4), atol=1e-9) and np.allclose(ak[iW, :], 0)
phi = np.arccos(np.sqrt(np.clip(gam_, 0, 1)))
PF = lambda ph: sum(np.outer(np.cos(ph[k_]) * wk[:, k_] + np.sin(ph[k_]) * ak[:, k_], np.cos(ph[k_]) * wk[:, k_] + np.sin(ph[k_]) * ak[:, k_]) for k_ in range(4))
ok_cs = ok_cs and np.allclose(PF(phi), p, atol=1e-9)
jkl = np.array([[(wk[iW, k_] @ JY @ wk[iW, l_]) ** 2 for l_ in range(4)] for k_ in range(4)])
ok_cs = ok_cs and np.allclose(jkl.sum(axis=1), 1)
for _ in range(5):
    g2, gY = RNG.uniform(0.4, 1.6, size=2)
    ph = RNG.uniform(0, np.pi / 2, size=4)
    gm = np.cos(ph) ** 2
    Phi = (3 * g2 ** 2 + gY ** 2) / 4 * gm.sum() - sum((g2 ** 2 + gY ** 2 * jkl[k_, l_]) / 2 * gm[k_] * gm[l_] for k_ in range(4) for l_ in range(k_ + 1, 4))
    ok_cs = ok_cs and abs(V8("SM", {"3": 1.0, "2": g2, "Y": gY}, PF(ph)) - 128 * Phi) < 1e-8
# numerical: for g_Y > g_2 runs end either at the four points (V=0) or on the component A'+W' (V=128 g_2^2)
endv = []
ok_run = True
for _ in range(40):
    g2 = float(RNG.uniform(0.5, 1.0))
    g = {"3": 1.0, "2": g2, "Y": g2 * float(RNG.uniform(1.3, 3.0))}
    xs_g = [(g[lab] ** 2, x) for lab, xs in RHO["SM"] for x in xs]
    fb, pe, rg = minimise_F3(xs_g, RNG.normal(size=32))
    lv = fb / (128 * g2 ** 2)
    if abs(lv) < 1e-10:
        endv.append(0)
    elif abs(lv - 1) < 1e-8:
        endv.append(1)
        gw = np.sort(np.linalg.eigvalsh(pe[np.ix_(iW, iW)]))
        ok_run = ok_run and np.allclose(gw, [0, 0, 1, 1], atol=1e-6) and abs(y_hyp(pe)) < 1e-10
    else:
        endv.append(lv)
ok_run = ok_run and set(endv) == {0, 1}
report(17, ok and ok_cs and ok_run, f"n=3, h_SM (exact): P_0=span(E2,E3,E7,E13) (A' + a J_Y-line W') is a critical point of f and y; relative to Tr(dPi^2) the Hessians of f and y on the 16 tangent directions are -1/2 and +1/2 on the 8 A-W mixing directions, y has eigenvalues 2,2 on the anti-complex part of W'(x)W'', so V_SM has Hessian eigenvalues 0 (x6, the tangent of Gr~_2(A) x CP^1), (g_Y^2-g_2^2)/2 (x8), 2g_Y^2 (x2) (computed for g_Y^2/g_2^2 = 0.5, 2: {evs}); along the CS family P(phi) with fixed principal frames V_SM = 128[(3g_2^2+g_Y^2)/4 sum gamma_k - sum_{{k<l}} (g_2^2+g_Y^2 j_kl)/2 gamma_k gamma_l], j_kl=(w_k.J_Y w_l)^2 with row sums 1; for g_Y/g_2 in [1.3,3] minimisation ends at the four points (V=0) or on A'+W' (V=128 g_2^2, gamma=(0,0,1,1), y=0) and nowhere else (end values / 128g_2^2 = {sorted(set(endv))}, counts {endv.count(0)}, {endv.count(1)})")

# Hessian on R^512 by finite differences at Pi_A (h_SM and h_PS+u(1)_A)
ok = True
fdv = {}
for name in ("SM", "PS+u1A"):
    g = rand_couplings(name)
    Hs = sum(float(g[lab] ** 2) * np.array(CASIMIR[(lab, "A")], dtype=float) for lab, _ in ALG[name])
    for _ in range(2):
        xv = RNG.normal(size=16)
        X = Comb([(xv[k_] * 0.5, E[m[0]] @ E[m[1]]) for k_, m in enumerate(mix)]).dense()
        t = 1e-3
        vals = []
        for s in (t, -t):
            U = expm(s * X)
            vals.append(V512(name, g, U @ PiA_d @ U.T))
        second = (vals[0] + vals[1]) / (2 * t ** 2)
        kin = np.sum(dcm(Comb([(xv[k_] * 0.5, E[m[0]] @ E[m[1]]) for k_, m in enumerate(mix)]), PiA_d) ** 2)
        ok = ok and abs(kin - 128 * xv @ xv) < 1e-8 and abs(second - 128 * xv @ Hs @ xv) < 1e-4 * second
        fdv[name] = round(second / (128 * xv @ Hs @ xv), 7)
report(18, ok, f"n=3: on R^512, V_h(exp(tX)Pi_A exp(-tX)) = t^2 Tr([X,Pi_A]^2) (x^T H x)/|x|^2 + O(t^3) for random X = sum x_aw aw/2 (h_SM and h_PS+u(1)_A, random couplings), with Tr([X,Pi_A]^2) = 128|x|^2: the exact Hessian of p2-14 is the second variation of the alignment energy (finite-difference ratio {fdv})")

# the P_*-surviving algebra of ES17 Thm 17.22(ii), restated, and its action on F_3
Pd = Pst.dense()
ok = all(iszero(dcm(rot(i, j), Pd)) for i, j in itertools.combinations(Widx, 2)) and iszero(dcm(rot(8, 9), Pd)) and iszero(dcm(rot(2, 3), Pd))
ok = ok and np.allclose(rho_plus(rot(8, 9)), -2 * rho_plus(unit(8, 9))) and np.allclose(rho_plus(rot(2, 3)), -2 * rho_plus(QA))
ok = ok and all(np.allclose(rho_plus(rot(i, j)), -2 * rho_plus(unit(i, j))) for i, j in itertools.combinations(Widx, 2))
ok = ok and all(np.allclose(vec_rep(rot(i, j), E, Vm), 2 * vec_rep(unit(i + 3, j + 3), E, Vm)) for i, j in itertools.combinations(Widx, 2))
Pi, p = point(rand_frame())
ok = ok and close(dcm(E[5] @ E[6], Pi), np.zeros((N, N)), 1e-9) and close(dcm(rot(2, 3), Pi), -dcm(qA, Pi), 1e-9)
ok = ok and iszero(cm(rot(8, 9), volA)) and iszero(cm(rot(2, 3), volA)) and all(iszero(cm(rot(i, j), volA)) for i, j in itertools.combinations(Widx, 2))
report(19, ok, "restated (ES17 Thm 17.22(ii)): rot(i,j)=e_{i'}e_{j'}-e_ie_j (i'=i+3) for i<j in W, rot(8,9) and rot(2,3)=E5E6-E2E3 commute with P_*; on V_+ the first act as -2 x (unit rotations of W), so so(4)_Delta acts on F_3 through all of so(4)_W, rot(8,9) as -2 x the rotation of (E8,E9); rot(2,3) is not internal, E5E6 commutes with every Pi_P and rot(2,3) acts on F_3 as -q_A; all of them fix Pi_A")

# what is unbroken on the selected sets
ok = True
for P_ in (PiA_d, PiW_d, Id - PiA_d, Id - PiW_d):
    ok = ok and iszero(dcm(Bem, P_)) and all(iszero(dcm(X, P_)) for X in su3)
others_A = [i for i in range(8) if i not in U_A]
others_em = [i for i in range(8) if i not in U_em]


def frame_with(fixed, k, span):
    Fm = np.zeros((4, 8))
    for r, i in enumerate(fixed):
        Fm[r, i] = 1.0
    Qm = np.linalg.qr(RNG.normal(size=(len(span), k)))[0]
    for c_ in range(k):
        Fm[len(fixed) + c_, span] = Qm[:, c_]
    return Fm


sem_on_A, f_on_em = [], []
for _ in range(5):
    for Fm in (frame_with(U_A, 2, others_A), frame_with([], 4, others_A)):
        p = Fm.T @ Fm
        ok = ok and abs(s_plane(p, U_A)) < 1e-12
        sem_on_A.append(s_plane(p, U_em))
    for Fm in (frame_with(U_em, 2, others_em), frame_with([], 4, others_em)):
        p = Fm.T @ Fm
        ok = ok and abs(s_plane(p, U_em)) < 1e-12
        f_on_em.append(f_weak(p))
ok = ok and min(sem_on_A) > 1e-3 and min(f_on_em) > 1e-3
# on the hypercharge-fixed component: unbroken su(3)+u(1)_{T3L}+u(1)_Y, containing B_em; only W^+- massive
P0d = point(np.eye(8)[iP0])[0]
gensSM = su3 + tL + [tY]
stab0 = len(gensSM) - span_rank([dcm(X, P0d) for X in gensSM])
ok = ok and stab0 == 10 and iszero(dcm(Bem, P0d)) and iszero(dcm(tL[2], P0d)) and iszero(dcm(tY, P0d))
C = [dcm(X, P0d) for X in tL]
M2W = np.array([[np.sum(a_ * b_) for b_ in C] for a_ in C])
evW = np.sort(np.linalg.eigvalsh(M2W))
ok = ok and np.allclose(evW, [0, 64, 64])
report(20, ok, f"n=3: at Pi_+-A, Pi_+-W u(1)_em and su(3)_C are unbroken (restated p1-21); on the u(1)_A-selected components u(1)_em is broken at generic points (min s_em at random points {min(sem_on_A):.3f} > 0), on the u(1)_em-selected components su(2)_L is broken at generic points (min f {min(f_on_em):.3f} > 0); on the metastable component A'+W' of h_SM the unbroken algebra is su(3)+u(1)_(T3L)+u(1)_Y (dim {stab0}), containing B_em, and only W^+- are massive (su(2)_L mass matrix eigenvalues {evW.round(9).tolist()} g_2^2 f^2)")
