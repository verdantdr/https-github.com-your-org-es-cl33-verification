#!/usr/bin/env python3
"""p3: which charge is the photon at the third rung, and the residual choice among the four aligned vacua.

    python3 investigations/photon_selection/verify/p3_identification.py

Prints one 'PASS p3-<n>: <claim>' (or 'FAIL ...') line per claim of sections/p3_identification.tex and exits 0 iff all pass.

Realisation (tex/s02_arena.tex, tex/sA_realisation.tex), identical to verify/p1_families.py: e1=sx(x)1(x)1, e2=sz(x)sx(x)1,
e3=sz(x)sz(x)sx, e4=ep(x)1(x)1, e5=sz(x)ep(x)1, e6=sz(x)sz(x)ep, ep=[[0,-1],[1,0]], omega=e1...e6,
E_{6(k-1)+a}=omega^{(x)(k-1)} (x) e_a (x) 1...  Every generator and blade is a signed permutation matrix of size 512 (class SP);
dyadic linear combinations of blades (class Comb) are expanded exactly, and float64 is exact on them.  Floating point with
tolerance TOL is used only at random points of F_3 (orthonormal frames, matrix exponentials), for spectra, ranks and null spaces,
and in the numerical optimisation of p3-14 and p3-22, whose tolerances are stated in the claims.
"""
import itertools
import sys
import time

import numpy as np
from scipy.linalg import expm, null_space
from scipy.optimize import minimize

T_START = time.time()
TOL = 1e-9
RNG = np.random.default_rng(20261026)
RESULTS = []


def report(n, ok, claim):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} p3-{n}: {claim}", flush=True)
    print(f"  [p3-{n} at {time.time() - T_START:.1f}s]", file=sys.stderr, flush=True)


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

    def lmul(self, X):  # self @ X for a dense X
        out = np.empty_like(X)
        out[self.p] = self.s[:, None] * X
        return out

    def rmul(self, X):  # X @ self for a dense X
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
    """Commutator of two exact operators (SP or Comb) as a Comb."""
    a, b = Comb.of(a), Comb.of(b)
    return a @ b - b @ a


def dcm(X, D):
    """[X, D] for X exact (SP or Comb) and D dense."""
    X = Comb.of(X)
    return X.lmul(D) - X.rmul(D)


def nz(x):
    """Max abs entry of an exact operator or dense matrix."""
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


def span_rank(mats, tol=1e-8):
    if not mats:
        return 0
    A = np.array([m.ravel() for m in mats])
    G = A @ A.T
    return int(np.linalg.matrix_rank(G, tol=tol * max(1.0, float(np.max(np.abs(G))))))


def orth_basis(mats, tol=1e-8):
    """Orthonormal (Frobenius) basis of the span of dense matrices, via the Gram matrix."""
    A = np.array([m.ravel() for m in mats])
    G = A @ A.T
    w, V = np.linalg.eigh(G)
    keep = w > tol * max(1.0, float(np.max(w)))
    B = (V[:, keep] / np.sqrt(w[keep])).T @ A
    shape = mats[0].shape
    return [b.reshape(shape) for b in B]


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
    """Generators E_1..E_{6n} (index 0 unused) as signed permutations of size 8^n, and their squares eta."""
    E, eta = [None], [0]
    for k in range(1, n + 1):
        for a in range(6):
            E.append(sp_kron([om8] * (k - 1) + [e8[a]] + [ident(8)] * (n - k)))
            eta.append(ETA1[a])
    return E, eta


def vec_rep(X, E, idx):
    """Vector representation of an exact operator X on span{E_i : i in idx}: [X, E_l] = sum_k M_kl E_k.
    Asserts that the image lies in the span (exactly)."""
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


# =============================================================================================== n = 3 definitions
E, eta = tower(3)
N = 512
Id = np.eye(N)
Isp = ident(N)
I1 = Comb.of(Isp)
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
Xinv = Gam @ chiW
Zinv = -(J @ chiC)
Uinv = -(J @ qA)
Pst = Comb([(1.0 / 8, Isp)]) @ (I1 + Xinv) @ (I1 + Zinv) @ (I1 + Uinv)
Pst_d = Pst.dense()
PL = Pst @ Comb([(0.5, Isp), (-0.5, Gam)])
PR = Pst @ Comb([(0.5, Isp), (0.5, Gam)])
PL_d, PR_d = PL.dense(), PR.dense()
w = [E[i] for i in Widx]
tau = [Comb([(-0.5, w[1] @ w[2])]), Comb([(-0.5, w[2] @ w[0])]), Comb([(-0.5, w[0] @ w[1])])]
PmW = Comb([(0.5, Isp), (-0.5, chiW)])
PpW = Comb([(0.5, Isp), (0.5, chiW)])
tL = [PmW @ t for t in tau]
tR = [PpW @ t for t in tau]
Hc = Comb([(-1.0, E[10] @ E[16]), (-1.0, E[11] @ E[17]), (-1.0, E[12] @ E[18])])
Yb = tR[2] + Hc * (1.0 / 6)
Bem = tau[2] + Hc * (1.0 / 6)
QA = Comb([(0.5, qA)])
mJ = Comb.of(-J)
PiA = Comb([(0.5, Isp), (0.5, volA)])
PiW = Comb([(0.5, Isp), (0.5, volW)])
PimA = Comb([(0.5, Isp), (-0.5, volA)])
PimW = Comb([(0.5, Isp), (-0.5, volW)])
FOUR = {"A": (PiA, volA, 1), "-A": (PimA, volA, -1), "W": (PiW, volW, 1), "-W": (PimW, volW, -1)}
Zp = Comb([(0.5, Isp), (0.5, Zinv)])  # projector onto Z=+1, which contains P_*
Zm = Comb([(0.5, Isp), (-0.5, Zinv)])


def lift(M, idx):
    """sum_{i<j} M_ij e_ie_j/2 on an orthonormal positive frame, with an overall sign -1 on a negative frame (as in p1)."""
    sgn = 1.0 if eta[idx[0]] > 0 else -1.0
    return Comb([(sgn * M[p_, q_] * 0.5, E[idx[p_]] @ E[idx[q_]]) for p_, q_ in itertools.combinations(range(len(idx)), 2) if M[p_, q_] != 0])


def rho_plus(X):
    return vec_rep(X, E, Vp)


def simplify(X, tol=1e-14):
    """merge equal signed permutations (up to an overall sign) in a Comb; exact on dyadic coefficients."""
    acc = {}
    for c_, S in Comb.of(X).terms:
        sg = int(S.s[0])
        key = S.p.tobytes() + (S.s * sg).tobytes()
        if key in acc:
            acc[key][0] += c_ * sg
        else:
            acc[key] = [c_ * sg, SP(S.p, S.s * sg)]
    return Comb([(c_, S) for c_, S in acc.values() if abs(c_) > tol])


# colour su(3): derived centraliser of H_C in so(6)_C (vector model of V_-, lifted), as in p1-15
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
su3 = [simplify(lift(M, Vm)) for M in su3v]


def tform(X, Y):
    """invariant inner product <X,Y> = -Tr(XY)/512 (unit blades e_ie_j have norm 1)."""
    return -(Comb.of(X) @ Comb.of(Y)).trace() / N


# orthonormal bases (for tform) of the gauged algebras
G3 = np.array([[tform(x, y) for y in su3] for x in su3])
w3_, V3_ = np.linalg.eigh(G3)
su3_on = [simplify(Comb([(c_, S) for k_ in range(8) for c_, S in (su3[k_] * (V3_[k_, m_] / np.sqrt(w3_[m_]))).terms])) for m_ in range(8)]
su2L_on = [x * (2 * np.sqrt(2)) for x in tL]
Y_on = Yb * np.sqrt(24.0 / 5.0)
soW_on = [Comb([(1.0, E[i] @ E[j])]) for i, j in itertools.combinations(Widx, 2)]
so6_on = [Comb([(1.0, E[i] @ E[j])]) for i, j in kneg_pairs]
hSM_on = su3_on + su2L_on + [Y_on]
hPS_on = soW_on + so6_on

# ----------------------------------------------------------------------------------------------- p3-1
ok = all((E[i] @ E[i]) == (Isp if eta[i] > 0 else -Isp) for i in range(1, 19))
ok = ok and all(anticommutes(E[i], E[j]) for i in range(1, 19) for j in range(i + 1, 19))
ok = ok and sorted(Aidx + Widx) == Vp and Cidx == Vm
ok = ok and all(commutes(a_, b_) for a_ in (Xinv, Zinv, Uinv) for b_ in (Xinv, Zinv, Uinv))
ok = ok and all(a_ @ a_ == Isp and a_.T == a_ and a_.trace() == 0 for a_ in (Xinv, Zinv, Uinv))
ok = ok and abs(np.trace(Pst_d) - 64) < TOL and close(Pst_d @ Pst_d, Pst_d)
QhatA = mJ @ QA
ok = ok and close(QhatA, Comb([(0.5, Uinv)]))
ok = ok and commutes(qA, Xinv) and commutes(qA, Zinv) and commutes(qA, Uinv)
ok = ok and close(Uinv @ Pst, Pst) and close(qA @ Pst, J @ Pst) and close(QhatA @ Pst, Pst * 0.5)
report(1, ok, "n=3: the charge operator of the construction's charge, Qhat_A=-J Q_A=-(1/2)J q_A, equals U/2 exactly, U=-J q_A the third involution "
              "of P_*=(1/8)(1+X)(1+Z)(1+U) (rank 64); q_A commutes with X,Z,U and U P_*=P_*, so q_A P_*=J P_* and Qhat_A P_*=(1/2)P_*: "
              "the construction's charge is 1/2 on every state of the block")


# ----------------------------------------------------------------------------------------------- p3-2 joint charge table on V_*
def im_basis(Pd):
    w_, V_ = np.linalg.eigh(0.5 * (Pd + Pd.T))
    return V_[:, w_ > 0.5]


def joint_table(Bas, ops, scale=6):
    mats = [Bas.T @ Comb.of(o).dense() @ Bas for o in ops]
    mats = [0.5 * (m + m.T) for m in mats]
    okc = all(np.allclose(m1 @ m2, m2 @ m1, atol=1e-10) for m1 in mats for m2 in mats)
    coeffs = RNG.normal(size=len(mats))
    _, Vv = np.linalg.eigh(sum(c_ * m for c_, m in zip(coeffs, mats)))
    vals = np.array([np.diag(Vv.T @ m @ Vv) for m in mats]).T
    offd = max(float(np.max(np.abs(Vv.T @ m @ Vv - np.diag(np.diag(Vv.T @ m @ Vv))))) for m in mats)
    rv = np.round(vals * scale).astype(int)
    okr = float(np.max(np.abs(vals * scale - rv))) < 1e-8 and offd < 1e-8
    tab = {}
    for row in rv:
        tab[tuple(row.tolist())] = tab.get(tuple(row.tolist()), 0) + 1
    return tab, okc and okr


BasStar = im_basis(Pst_d)
That_L = mJ @ tL[2]
Yhat = mJ @ Yb
Qhat_em = mJ @ Bem
BLhat = (mJ @ Hc) * (1.0 / 3)
tab, okt = joint_table(BasStar, [Gam, That_L, Yhat, Qhat_em, QhatA, BLhat])
# (Gamma, T3L, Y, Q, Q_A, B-L) in units of 1/6, real multiplicities (4 real = 2 complex = one internal Weyl component)
expect = {(-6, 3, 1, 4, 3, 2): 12, (-6, -3, 1, -2, 3, 2): 12, (-6, 3, -3, 0, 3, -6): 4, (-6, -3, -3, -6, 3, -6): 4,
          (6, 0, 4, 4, 3, 2): 12, (6, 0, -2, -2, 3, 2): 12, (6, 0, -6, -6, 3, -6): 4, (6, 0, 0, 0, 3, -6): 4}
report(2, okt and tab == expect and BasStar.shape[1] == 64,
       "n=3: on the range of P_* the operators Gamma, -J t^3_L, -J t_Y, Qhat_em=-J B_em, Qhat_A, B-L=-J H_C/3 commute, and their joint "
       "eigenvalues with real multiplicities (complex multiplicity in the J-structure = half) are the Standard-Model family with nu_R: "
       "Q_L (Gamma=-1, T3=+-1/2, Y=1/6, Q=2/3,-1/3) 12+12, L_L (Y=-1/2, Q=0,-1) 4+4, u_R (2/3) 12, d_R (-1/3) 12, e_R (-1) 4, nu_R (0) 4, "
       "with Qhat_A=1/2 on every row")

# ----------------------------------------------------------------------------------------------- p3-3 the spectator Weyl block
PG_d = Comb([(0.5, Isp), (-0.5, Gam)]).dense()
BasW = im_basis(PG_d)
tabA, ok1 = joint_table(BasW, [QhatA], scale=2)
tabQ, ok2 = joint_table(BasW, [Qhat_em], scale=6)
tabU, ok3 = joint_table(BasW, [Uinv, QhatA], scale=2)
symQ = all(tabQ.get((k_[0],), 0) == tabQ.get((-k_[0],), 0) for k_ in tabQ)
ok = ok1 and ok2 and ok3 and BasW.shape[1] == 256 and tabA == {(1,): 128, (-1,): 128} and symQ
ok = ok and tabU == {(2, 1): 128, (-2, -1): 128}
ok = ok and tabQ == {(6,): 16, (-6,): 16, (4,): 48, (-4,): 48, (2,): 48, (-2,): 48, (0,): 32}
report(3, ok, f"n=3: on the Weyl block (1/2)(1-Gamma) of the spectator action (rank 256), Qhat_A=U/2 takes the values +-1/2 (128 real each) and "
              f"Qhat_em takes the symmetric values +-1 (16), +-2/3 (48), +-1/3 (48), 0 (32) of two families and two mirror families: Q_A is "
              f"universal only on the half U=+1, which P_* selects (computed Qhat_em multiplicities x6: {dict(sorted(tabQ.items()))})")

# ----------------------------------------------------------------------------------------------- p3-4 the two charges are distinct
ok = iszero(cm(QA, Bem)) and abs(tform(QA, Bem)) < TOL and iszero(cm(J, QA)) and iszero(cm(J, Bem))
ok = ok and all(iszero(cm(g_, QA)) and iszero(cm(g_, Bem)) for g_ in gam)
ok = ok and all(iszero(cm(QA, x)) for x in hSM_on + hPS_on)
# centraliser of h_PS in k is so(4)_A, and it contains Q_A
kbl = [Comb([(1.0, E[i] @ E[j])]) for i, j in kbiv]
rho_int = {}


def rho_full(X):
    return vec_rep(X, E, internal)


rk = [rho_full(Bb) for Bb in kbl]
rows = []
for Tg in hPS_on:
    rT = rho_full(Tg)
    rows.append(np.array([(rT @ r_ - r_ @ rT).ravel() for r_ in rk]).T)
cent = null_space(np.vstack(rows))
cent_pairs = [kbiv[k_] for k_ in range(43) if np.max(np.abs(cent[k_])) > 1e-9]
ok = ok and cent.shape[1] == 6 and set(cent_pairs) == set(itertools.combinations(Aidx, 2))
QemV = BasStar.T @ Qhat_em.dense() @ BasStar
QAV = BasStar.T @ QhatA.dense() @ BasStar
ok = ok and np.allclose(QAV, 0.5 * np.eye(64)) and span_rank([QemV, QAV]) == 2 and np.min(np.abs(np.linalg.eigvalsh(QemV))) < 1e-10
ok = ok and eta[2] == eta[3] == eta[7] == eta[13] == eta[1] == 1 and all(eta[i] == -1 for i in Cidx)
report(4, ok, "n=3: Q_A=(1/2)E2E3 and B_em commute, are orthogonal in the trace form, commute with J, the interior gammas and every generator of h_SM "
              "and h_PS; the centraliser of h_PS in k is so(4)_A (dim 6) and contains Q_A; on V_* no multiple of Qhat_em equals Qhat_A=1/2 "
              "(Qhat_em has the eigenvalue 0 there); the rotation planes span(E2,E3) of Q_A and span(E7,E13) of the weak part of B_em consist "
              "of generators of square +1, the sign of the time generator E1, and the colour planes of generators of square -1")

# ----------------------------------------------------------------------------------------------- p3-5 anomalies on the block
BL_ = im_basis(PL_d)
BR_ = im_basis(PR_d)
specL = np.sort(np.round(6 * np.linalg.eigvalsh(BL_.T @ Qhat_em.dense() @ BL_)).astype(int))
specR = np.sort(np.round(6 * np.linalg.eigvalsh(BR_.T @ Qhat_em.dense() @ BR_)).astype(int))
ok = np.array_equal(specL, specR)
Tc_hat = [simplify(mJ @ x).dense() for x in su3_on]
Qem_d, QA_d, TL_d = Qhat_em.dense(), QhatA.dense(), That_L.dense()


def tr3(P_, A_, B_):
    return float(np.sum((P_ @ A_) * (B_ @ B_).T))


ok = ok and abs(tr3(PL_d, Qem_d, TL_d)) < TOL
ok = ok and all(abs(tr3(PL_d, Qem_d, t_) - tr3(PR_d, Qem_d, t_)) < 1e-8 for t_ in Tc_hat)
aA_L = tr3(PL_d, QA_d, TL_d)
ok = ok and abs(aA_L - 4) < 1e-9
ok = ok and abs(np.trace(PL_d @ QA_d) - np.trace(PR_d @ QA_d)) < TOL and abs(np.trace(PL_d @ QA_d @ QA_d @ QA_d) - np.trace(PR_d @ QA_d @ QA_d @ QA_d)) < 1e-9
ok = ok and all(abs(tr3(PL_d, QA_d, t_) - tr3(PR_d, QA_d, t_)) < 1e-8 for t_ in Tc_hat)
report(5, ok, f"n=3: on the block Qhat_em has the same spectrum on P_L and P_R (u(1)_em is vector-like: its cubic, gravitational and colour anomalies "
              f"cancel between the chiralities) and Tr(P_L Qhat_em That_L^2)=0; Qhat_A is also vector-like in Gamma, but Tr(P_L Qhat_A That_L^2)={aA_L:.0f}: "
              f"its only nonvanishing anomaly on the block is the mixed SU(2)_L^2 U(1)_A one (restated, p1-28)")

# ----------------------------------------------------------------------------------------------- vector model helpers
rho_k8 = [rho_plus(Comb([(0.5, E[i] @ E[j])])) for i, j in kpos_pairs]
rhoQA = rho_plus(QA)
rhoBem = rho_plus(Bem)
rho_su2L_on = [rho_plus(x) for x in su2L_on]
rho_Y_on = rho_plus(Y_on)
rho_soW_on = [rho_plus(x) for x in soW_on]
vpos = {i: k_ for k_, i in enumerate(Vp)}


def frame_of(vectors):
    """rows: oriented frame in V_+ coordinates from a list of coordinate vectors."""
    return np.array(vectors, dtype=float)


def e8(i):
    v = np.zeros(8)
    v[vpos[i]] = 1.0
    return v


def rand_2plane(idx):
    M = np.zeros((2, 8))
    Qm = np.linalg.qr(RNG.normal(size=(len(idx), 2)))[0]
    for c_ in range(2):
        for s, i in enumerate(idx):
            M[c_, vpos[i]] = Qm[s, c_]
    return M


def vm_tangent_kernel(gens8, p8):
    tans = [Yv @ p8 - p8 @ Yv for Yv in rho_k8]
    basis = orth_basis(tans)
    if not gens8:
        return len(basis), len(basis)
    rows_ = [np.array([[np.sum(bb * (g8 @ ba - ba @ g8)) for ba in basis] for bb in basis]) for g8 in gens8]
    return len(basis), null_space(np.vstack(rows_)).shape[1]


def vm_fixed(gens8, p8):
    return all(np.allclose(g8 @ p8, p8 @ g8, atol=1e-10) for g8 in gens8)


# ----------------------------------------------------------------------------------------------- p3-6 joint fixed set of Q_A and B_em
pts = {
    "A": frame_of([e8(i) for i in Aidx]),
    "W": frame_of([e8(i) for i in Widx]),
    "UA+Uem": frame_of([e8(2), e8(3), e8(7), e8(13)]),
    "R4": frame_of([e8(8), e8(9), e8(14), e8(15)]),
    "UA+P'": np.vstack([e8(2), e8(3), rand_2plane([8, 9, 14, 15])]),
    "Uem+P'": np.vstack([e8(7), e8(13), rand_2plane([8, 9, 14, 15])]),
}
res6 = {}
ok = True
for nm, F in pts.items():
    p8 = F.T @ F
    fx = vm_fixed([rhoQA, rhoBem], p8)
    dim_, ker = vm_tangent_kernel([rhoQA, rhoBem], p8)
    res6[nm] = (fx, ker)
    ok = ok and fx and dim_ == 16
ok = ok and [res6[k_][1] for k_ in pts] == [4, 4, 0, 0, 4, 4]
Fg = np.linalg.qr(RNG.normal(size=(8, 4)))[0].T
pg = Fg.T @ Fg
ok = ok and not vm_fixed([rhoQA], pg) and not vm_fixed([rhoBem], pg)
report(6, ok, f"n=3: the points of F_3 commuting with both Q_A and B_em are P=+-(U_A+U_em) and P=+-span(E8,E9,E14,E15) (isolated: joint tangent "
              f"kernel 0) and the two components {{U_A+P'}}, {{U_em+P'}} with P' an oriented 2-plane in span(E8,E9,E14,E15), each a copy of "
              f"G~r_2(R^4) of dim 4; Pi_+-A lie on the first and Pi_+-W on the second (joint kernel 4); a generic point commutes with neither "
              f"(computed (fixed, kernel) {res6})")


# ----------------------------------------------------------------------------------------------- tangent matrices on R^512
mix = [(i, j) for i in Aidx for j in Widx]


def tangent_basis(nm):
    Pi0, v0, s0 = FOUR[nm]
    return {(i, j): Comb([(0.5 * s0, E[i] @ E[j] @ v0)]) for i, j in mix}


TB = {nm: tangent_basis(nm) for nm in FOUR}
ok = all(close(cm(Comb([(0.5, E[i] @ E[j])]), FOUR[nm][0]), TB[nm][(i, j)]) for nm in FOUR for i, j in mix)


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


# ----------------------------------------------------------------------------------------------- p3-7 torus weights
r2 = np.sqrt(2.0)
res7 = {}
for nm in FOUR:
    MA = tangent_matrix(QA, TB[nm])
    Mem = tangent_matrix(Bem, TB[nm])
    okc = np.allclose(MA @ Mem, Mem @ MA)
    ev = np.abs(np.linalg.eigvals(MA + r2 * Mem).imag)
    cnt = {}
    for lab, val in (("(1,1)", 1 + r2), ("(1,-1)", r2 - 1), ("(1,0)", 1.0), ("(0,1)", r2), ("(0,0)", 0.0)):
        cnt[lab] = int(np.sum(np.abs(ev - val) < 1e-8))
    res7[nm] = (okc, cnt)
want7 = {"(1,1)": 2, "(1,-1)": 2, "(1,0)": 4, "(0,1)": 4, "(0,0)": 4}
ok = ok and all(r_[0] and r_[1] == want7 for r_ in res7.values())
report(7, ok, "n=3: at each of Pi_A, Pi_-A, Pi_W, Pi_-W the tangent space (16 real) decomposes under the torus u(1)_A x u(1)_em into the weights "
              "(1,1) and (1,-1) (one complex line each: U_A(x)U_em), (1,0) (two complex: U_A(x)W_0), (0,1) (two complex: A_0(x)U_em) and (0,0) "
              "(four real: A_0(x)W_0), W_0=span(E14,E15), A_0=span(E8,E9) (real multiplicities of |weight| read from ad Q_A+sqrt2 ad B_em)")

# ----------------------------------------------------------------------------------------------- p3-8 the unit of charge
Bem2V = BasStar.T @ (Comb.of(Bem) @ Bem).dense() @ BasStar
ok = np.allclose(Bem2V, -(QemV @ QemV)) and np.allclose(BasStar.T @ (QA @ QA).dense() @ BasStar, -0.25 * np.eye(64))
Mem = tangent_matrix(Bem, TB["A"])
MA = tangent_matrix(QA, TB["A"])
ok = ok and np.allclose(Mem @ Mem @ Mem, -Mem) and np.allclose(MA @ MA @ MA, -MA)
ok = ok and np.sum(np.abs(np.abs(np.linalg.eigvals(Mem)) - 1) < 1e-9) == 8
ok = ok and all(iszero(cm(x, PiA)) for x in hSM_on)
evB2 = sorted(set(np.round(np.linalg.eigvalsh(-Bem2V) * 9).astype(int).tolist()))
ok = ok and evB2 == [0, 1, 4, 9]
report(8, ok, "n=3: on V_* the real generator satisfies B_em^2=-Qhat_em^2 (squared magnitudes 0, 1/9, 4/9, 1; electron 1) and Q_A^2=-1/4, while on the "
              "charged tangent directions at Pi_A (ad B_em)^2=-1 and (ad Q_A)^2=-1: the projector's charged scalars carry the electron's charge "
              "magnitude under u(1)_em, and twice the universal fermion charge 1/2 under u(1)_A; at Pi_A all twelve h_SM generators commute "
              "with the vacuum (no h_SM boson is massive there)")

# ----------------------------------------------------------------------------------------------- p3-9 electroweak breaking by the projector's own doublet
th = 0.9
gth = Comb([(np.cos(th / 2), Isp), (np.sin(th / 2), E[8] @ E[14])])
Pith = (gth @ PiA @ gth.T).dense()
ok = close(Pith @ Pith, Pith) and close(Pith, Pith.T) and abs(np.trace(Pith) - 256) < 1e-9
ok = ok and iszero(dcm(Bem, Pith)) and iszero(dcm(QA, Pith)) and all(iszero(dcm(x, Pith)) for x in su3_on)
Gram = np.array([[tform(x, y) for y in hSM_on] for x in hSM_on])
ok = ok and np.allclose(Gram, np.eye(12), atol=1e-10)
comms = [dcm(x, Pith) for x in hSM_on]
Kmat = np.array([[np.sum(a_ * b_.T) for b_ in comms] for a_ in comms])
res9 = []
for g3, g2, gY in ((0.7, 1.0, 1.0), (1.3, 0.8, 0.45)):
    Gc = np.diag([g3] * 8 + [g2] * 3 + [gY])
    Mm = Gc @ Kmat @ Gc
    ev = np.sort(np.linalg.eigvalsh(Mm))
    zero = int(np.sum(np.abs(ev) < 1e-8 * ev[-1]))
    mW2, mZ2 = ev[-2], ev[-1]
    ratio = mW2 / mZ2
    ok = ok and zero == 9 and abs(ev[-3] - ev[-2]) < 1e-8 * mZ2 and abs(ratio - 1.0 / (1.0 + 0.6 * gY ** 2 / g2 ** 2)) < 1e-10
    res9.append(round(ratio, 6))
    nullv = null_space(Mm[8:, 8:])
    gen = Comb([])
    for k_ in range(4):
        gen = gen + (su2L_on + [Y_on])[k_] * (nullv[k_, 0] * [g2, g2, g2, gY][k_])
    gn = gen * (1.0 / np.sqrt(tform(gen, gen)))
    bn = Bem * (1.0 / np.sqrt(tform(Bem, Bem)))
    ok = ok and nullv.shape[1] == 1 and (close(gn, bn) or close(gn, -bn))
report(9, ok and res9[0] == 0.625, f"n=3: along the neutral doublet direction, Pi_theta=Ad(exp(theta E8E14/2))Pi_A (theta=0.9) is a point of F_3 commuting with "
                                   f"B_em, Q_A and su(3)_C; the h_SM mass matrix (orthonormal basis for -Tr/512, couplings g3,g2,gY) has eigenvalues 0 (9 times: "
                                   f"gluons and photon), m_W^2 (twice), m_Z^2, with m_W^2/m_Z^2=1/(1+(3/5)gY^2/g2^2) (= 5/8 = 1-3/8 at gY=g2) and massless "
                                   f"electroweak combination B_em (computed ratios {res9})")

# ----------------------------------------------------------------------------------------------- matter projectors
X1 = E[2] @ E[3] @ E[8] @ E[9]
X2 = blade(E, [13, 14, 15, 16, 17, 18])
X3 = blade(E, [1, 2, 3, 4, 5, 6]) @ E[13] @ E[16]
Pfam = Comb([(1.0 / 8, Isp)]) @ (I1 + X1) @ (I1 + X2) @ (I1 + X3)
sC = [J @ E[10] @ E[16], J @ E[11] @ E[17], J @ E[12] @ E[18]]
Qc = mJ @ Hc
Plep = Comb([(1.0 / 8, Isp)]) @ (I1 - sC[0]) @ (I1 - sC[1]) @ (I1 - sC[2])
g0 = E[1]
Qv = PL + g0 @ PL @ g0.T
Gm = Comb([(0.5, Isp), (-0.5, Gam)])
Gp = Comb([(0.5, Isp), (0.5, Gam)])
MATTER = {"P_*": Pst, "P_L": PL, "P_R": PR, "Pfam": Pfam, "P_lep": Plep, "(1-G)/2": Gm, "(1+G)/2": Gp, "Q_v": Qv}
MATTER_d = {k_: v_.dense() for k_, v_ in MATTER.items()}

# ----------------------------------------------------------------------------------------------- p3-10 Plucker lemma
pos4 = list(itertools.combinations(range(8), 4))
eI = [blade(E, [Vp[k_] for k_ in I]) for I in pos4]
ok = all(b_ @ b_ == Isp and b_.T == b_ for b_ in eI)


def frame_vol(F):
    out = None
    for r in range(F.shape[0]):
        u = Comb([(F[r, s], E[Vp[s]]) for s in range(8) if F[r, s] != 0])
        out = u if out is None else out @ u
    return out


def plucker(F):
    return np.array([np.linalg.det(F[:, list(I)]) for I in pos4])


for _ in range(2):
    F = np.linalg.qr(RNG.normal(size=(8, 4)))[0].T
    ok = ok and close(frame_vol(F), Comb([(c_, b_) for c_, b_ in zip(plucker(F), eI)]), 1e-10)
mcomp = {k_: np.array([(b_ @ M).trace() for b_ in eI]) for k_, M in MATTER.items()}
iA, iW = pos4.index(tuple(vpos[i] for i in Aidx)), pos4.index(tuple(sorted(vpos[i] for i in Widx)))
ok = ok and tuple(vpos[i] for i in Widx) == tuple(sorted(vpos[i] for i in Widx))


def only(vec, d):
    ref = np.zeros(70)
    for k_, v_ in d.items():
        ref[k_] = v_
    return np.allclose(vec, ref, atol=1e-9)


ok = ok and all(only(mcomp[k_], {}) for k_ in ("P_*", "P_lep", "(1-G)/2", "(1+G)/2"))
ok = ok and only(mcomp["Pfam"], {iA: 64}) and only(mcomp["P_L"], {iA: 32, iW: -32}) and only(mcomp["P_R"], {iA: -32, iW: 32})
ok = ok and only(mcomp["Q_v"], {iA: 64, iW: -64}) and only(np.array([(b_ @ PiA).trace() for b_ in eI]), {iA: 256})
report(10, ok, "n=3: vol_P = sum_I p_I(P) e_I over the 70 positive four-blades, p_I the Plucker coordinates of the oriented frame, so "
               "Tr(Pi_P M)=(1/2)Tr M+(1/2)sum_I p_I Tr(e_I M); the four-blade components Tr(e_I M) vanish for P_*, P_lep and (1+-Gamma)/2, "
               "are 64 e_A for Pfam, 32(e_A-e_W) for P_L, -32(e_A-e_W) for P_R, 64(e_A-e_W) for Q_v and 256 e_A for Pi_A")

# ----------------------------------------------------------------------------------------------- p3-11 overlaps at the four points
tab11 = {}
for k_, M in MATTER_d.items():
    row = []
    for nm in ("A", "-A", "W", "-W"):
        Pd = FOUR[nm][0].dense()
        lin = float(np.sum(Pd * M.T))
        PM = Pd @ M
        quad = float(np.sum(PM * PM.T))
        row.append((round(lin, 9), round(quad, 9)))
    tab11[k_] = row
want11 = {"P_*": [(32, 32)] * 4, "P_L": [(32, 32), (0, 0), (0, 0), (32, 32)], "P_R": [(0, 0), (32, 32), (32, 32), (0, 0)],
          "Pfam": [(64, 64), (0, 0), (32, 16), (32, 16)], "P_lep": [(32, 32)] * 4, "(1-G)/2": [(128, 128)] * 4, "(1+G)/2": [(128, 128)] * 4,
          "Q_v": [(64, 64), (0, 0), (0, 0), (64, 64)]}
ok = all(np.allclose(np.array(tab11[k_], dtype=float), np.array(want11[k_], dtype=float), atol=1e-8) for k_ in want11)
report(11, ok, "n=3: the overlaps (Tr(Pi P), Tr(Pi P Pi P)) at (Pi_A, Pi_-A, Pi_W, Pi_-W) are: P_* (32,32) at all four; P_lep (32,32) at all four; "
               "(1+-Gamma)/2 (128,128) at all four; P_L (32,32),(0,0),(0,0),(32,32); P_R (0,0),(32,32),(32,32),(0,0); Q_v (64,64),(0,0),(0,0),(64,64); "
               "Pfam (64,64),(0,0),(32,16),(32,16)")

# ----------------------------------------------------------------------------------------------- p3-12 overlaps on all of F_3
ok = True
PiA_d = PiA.dense()
for _ in range(3):
    F = np.linalg.qr(RNG.normal(size=(8, 4)))[0].T
    pl = plucker(F)
    pA, pW = pl[iA], pl[iW]
    Pd = 0.5 * (Id + frame_vol(F).dense())
    tr = {k_: float(np.sum(Pd * M.T)) for k_, M in MATTER_d.items()}
    ok = ok and abs(tr["P_*"] - 32) < 1e-8 and abs(tr["P_lep"] - 32) < 1e-8 and abs(tr["(1-G)/2"] - 128) < 1e-8 and abs(tr["(1+G)/2"] - 128) < 1e-8
    ok = ok and abs(tr["Pfam"] - 32 * (1 + pA)) < 1e-8 and abs(tr["Pfam"] - 0.25 * np.sum(Pd * PiA_d)) < 1e-8
    ok = ok and abs(tr["P_L"] - 16 * (1 + pA - pW)) < 1e-8 and abs(tr["P_R"] - 16 * (1 - pA + pW)) < 1e-8 and abs(tr["Q_v"] - 2 * tr["P_L"]) < 1e-8
    PP = Pst_d @ Pd @ Pst_d
    quad = float(np.sum((Pd @ Pst_d) * (Pd @ Pst_d).T))
    ok = ok and abs(quad - np.sum(PP * PP)) < 1e-8 and quad < 32 - 1e-3
report(12, ok, "n=3: at random points of F_3, Tr(Pi P_*)=32, Tr(Pi P_lep)=32 and Tr(Pi (1+-Gamma)/2)=128 (constant on F_3); Tr(Pi Pfam)=32(1+p_A)="
               "(1/4)Tr(Pi Pi_A); Tr(Pi P_L)=16(1+p_A-p_W), Tr(Pi P_R)=16(1-p_A+p_W), Tr(Pi Q_v)=2Tr(Pi P_L); Tr(Pi P_* Pi P_*)=||P_* Pi P_*||^2<32 "
               "at a generic point")

# ----------------------------------------------------------------------------------------------- p3-13 commutant of P_* in F_3
ok = True
for nm in FOUR:
    imgs = [dcm(T_, Pst_d) for T_ in TB[nm].values()]
    ok = ok and span_rank(imgs) == 16
kb_pos = [Comb([(0.5, E[i] @ E[j])]) for i, j in kpos_pairs]


def tangent_kernel_commuting(Pd, M):
    basis = orth_basis([dcm(Bb, Pd) for Bb in kb_pos])
    imgs = [b_ @ M - M @ b_ for b_ in basis]
    G = np.array([[np.sum(a_ * b_) for b_ in imgs] for a_ in imgs])
    return len(basis), int(np.sum(np.linalg.eigvalsh(G) < 1e-8 * max(1.0, np.max(G))))


res13 = []
for comp in ([2, 3], [8, 9]):
    Fm = np.vstack([e8(comp[0]), e8(comp[1]), rand_2plane(Widx)])
    Pd = 0.5 * (Id + frame_vol(Fm).dense())
    okc = close(Pd @ Pst_d, Pst_d @ Pd, 1e-9)
    dim_, ker = tangent_kernel_commuting(Pd, Pst_d)
    res13.append((okc, dim_, ker))
    ok = ok and okc and dim_ == 16 and ker == 4
Fg = np.linalg.qr(RNG.normal(size=(8, 4)))[0].T
Pd = 0.5 * (Id + frame_vol(Fg).dense())
ok = ok and not close(Pd @ Pst_d, Pst_d @ Pd, 1e-6)
report(13, ok, f"n=3: the points of F_3 commuting with P_* are Pi_+-A, Pi_+-W, isolated (the 16 tangents there all fail to commute with P_*), and the two "
               f"components {{U_A+W'}}, {{span(E8,E9)+W'}}, W' an oriented 2-plane in W, of dimension 4 (tangent kernel 4 at random points of each); a "
               f"generic point does not commute (computed {res13})")

# ----------------------------------------------------------------------------------------------- p3-14 the comass inequality
ok = True
for _ in range(300):
    F = np.linalg.qr(RNG.normal(size=(8, 4)))[0].T
    MA_ = F[:, [vpos[i] for i in Aidx]].T  # <a_i, f_j>
    MW_ = F[:, [vpos[i] for i in Widx]].T  # <w_i, f_j>
    c2 = np.linalg.eigvalsh(MA_.T @ MA_)
    ok = ok and np.allclose(MA_.T @ MA_ + MW_.T @ MW_, np.eye(4)) and np.allclose(np.sort(np.linalg.eigvalsh(MW_.T @ MW_)), np.sort(1 - c2))
    ok = ok and abs(abs(np.linalg.det(MA_)) - np.prod(np.sqrt(np.clip(c2, 0, 1)))) < 1e-10
    ok = ok and abs(abs(np.linalg.det(MW_)) - np.prod(np.sqrt(np.clip(1 - c2, 0, 1)))) < 1e-10
    ok = ok and abs(np.linalg.det(MA_)) + abs(np.linalg.det(MW_)) <= 1 + 1e-12
FA = frame_of([e8(i) for i in Aidx])
ia_cols = [vpos[i] for i in Aidx]
iw_cols = [vpos[i] for i in Widx]
tri = np.triu_indices(8, 1)


def frame_from(y):
    Yv = np.zeros((8, 8))
    Yv[tri] = y
    Yv = Yv - Yv.T
    return FA @ expm(Yv).T


def phiLW(y):
    F = frame_from(y)
    return np.linalg.det(F[:, ia_cols]) - np.linalg.det(F[:, iw_cols])


best, near = -9.0, True
for _ in range(40):
    y0 = RNG.normal(scale=1.5, size=28)
    r_ = minimize(lambda y: -phiLW(y), y0, method="BFGS", options={"gtol": 1e-10})
    val = -r_.fun
    best = max(best, val)
    if val > 1 - 1e-6:
        F = frame_from(r_.x)
        pA_, pW_ = np.linalg.det(F[:, ia_cols]), np.linalg.det(F[:, iw_cols])
        near = near and (abs(pA_ - 1) < 1e-5 or abs(pW_ + 1) < 1e-5)
ok = ok and best <= 1 + 1e-9 and best > 1 - 1e-9 and near
report(14, ok, f"n=3: for every oriented 4-plane P in V_+=A+W the blocks M_A, M_W of an orthonormal frame (M_A=<a_i,f_j>, M_W=<w_i,f_j>) satisfy M_A^T M_A+M_W^T M_W=1 with "
               f"|p_A|=prod cos(theta_i), |p_W|=prod sin(theta_i) (300 random planes), hence |p_A|+|p_W|<=1; maximising p_A-p_W over SO(8) from 40 "
               f"random starts gives the maximum 1 (found {best:.12f}), attained only at P=A or P=-W")

# ----------------------------------------------------------------------------------------------- p3-15 restated ES17 data on the projectors
allint = list(itertools.combinations(internal, 2))
ok = X1 == volA and close(PiA @ Pfam, Pfam) and close(Pfam @ PiA, Pfam)
gphys = [(i, j) for i, j in allint if commutes(E[i] @ E[j], volA)]
kP1 = [(i, j) for i, j in gphys if eta[i] == eta[j]]
ann_dim = len(kP1) - span_rank([(Comb([(1.0, E[i] @ E[j])]) @ PiA).dense() for i, j in kP1])
ok = ok and len(gphys) == 51 and len(kP1) == 27 and ann_dim == 3
Qcd = Qc.dense()
Plep_d = MATTER_d["P_lep"]
ok = ok and close(Qc, Comb.of(sC[0]) + sC[1] + sC[2]) and close(Plep_d @ Plep_d, Plep_d) and abs(np.trace(Plep_d) - 64) < TOL
ok = ok and close(Qcd @ Plep_d, -3 * Plep_d) and all(iszero(dcm(g_, Plep_d)) for g_ in gam) and iszero(dcm(J, Plep_d))


def V43(M):
    return sum(float(np.sum(dcm(E[i] @ E[j], M) ** 2)) for i, j in kbiv) / 512.0


def stab43(M):
    return 43 - span_rank([dcm(E[i] @ E[j], M) for i, j in kbiv])


tab15 = {k_: (round(V43(MATTER_d[k_]), 9), stab43(MATTER_d[k_])) for k_ in ("P_*", "Pfam", "P_lep", "Q_v")}
ok = ok and tab15 == {"P_*": (5.0, 23), "Pfam": (9.0, 9), "P_lep": (3.0, 37), "Q_v": (5.0, 25)}
report(15, ok, f"n=3 restated (Thm 7.5, Prop 21.6): X_1=J^(1)J^(2)=vol_A, so the rank-256 projector P_1 of the projector family is Pi_A, with "
               f"dim g_phys=51, dim k=27 and a three-dimensional annihilator, and Pfam<=Pi_A; P_lep, the spectral projector of Q_c=-J H_C at -3 "
               f"(rank 64, Lorentz invariant, commuting with J), reproduces the table of Prop 21.6: (V, stabiliser) = {tab15}")


# ----------------------------------------------------------------------------------------------- p3-16 the Klein group of the projector sector
def Dmap(Pd):
    return 0.5 * (Id + volPlus.lmul(2 * Pd - Id))


def Cmap(Pd):
    return Id - Pd


def Emap(Pd):
    return Cmap(Dmap(Pd))


kb_all = [Comb([(1.0, E[i] @ E[j])]) for i, j in kbiv]


def massmat(Pd):
    cs = [dcm(Bb, Pd) for Bb in kb_all]
    Am = np.array([c_.ravel() for c_ in cs])
    Bt = np.array([c_.T.ravel() for c_ in cs])
    return Am @ Bt.T


ok = True
Zp_d = Zp.dense()
for _ in range(2):
    F = np.linalg.qr(RNG.normal(size=(8, 8)))[0]
    if np.linalg.det(F) < 0:
        F[:, 7] *= -1
    Fp, Fperp = F[:, :4].T, F[:, 4:].T
    Pd = 0.5 * (Id + frame_vol(Fp).dense())
    vperp = frame_vol(Fperp).dense()
    sgn = 1.0 if close(volPlus.lmul(frame_vol(Fp).dense()), vperp, 1e-9) else -1.0
    ok = ok and close(volPlus.lmul(frame_vol(Fp).dense()), sgn * vperp, 1e-9)
    ok = ok and close(Emap(Pd), 0.5 * (Id - sgn * vperp), 1e-9)
    M0 = massmat(Pd)
    for f_ in (Dmap, Cmap, Emap):
        Pf_ = f_(Pd)
        ok = ok and close(Pf_ @ Pf_, Pf_, 1e-9) and np.allclose(massmat(Pf_), M0, atol=1e-7)
        Y8 = RNG.normal(size=(8, 8))
        Y8 = Y8 - Y8.T
        Yc = lift(Y8, Vp)
        for Qx in (QA, Bem):
            ok = ok and abs(np.sum(dcm(Qx, Pf_) * dcm(Yc, Pf_).T) - np.sum(dcm(Qx, Pd) * dcm(Yc, Pd).T)) < 1e-7
    ok = ok and close(Emap(Pd) @ Zp_d, Pd @ Zp_d, 1e-9) and close(Pst_d @ Emap(Pd) @ Pst_d, Pst_d @ Pd @ Pst_d, 1e-9)
four_d = {nm: FOUR[nm][0].dense() for nm in FOUR}


def which(Pd):
    hits = [nm for nm in four_d if close(Pd, four_d[nm])]
    return hits[0] if len(hits) == 1 else None


perm = {f_.__name__: {nm: which(f_(four_d[nm])) for nm in four_d} for f_ in (Dmap, Cmap, Emap)}
ok = ok and perm == {"Dmap": {"A": "W", "-A": "-W", "W": "A", "-W": "-A"}, "Cmap": {"A": "-A", "-A": "A", "W": "-W", "-W": "W"},
                     "Emap": {"A": "-W", "-A": "W", "W": "-A", "-W": "A"}}
report(16, ok, f"n=3: C(Pi)=1-Pi, D(Pi)=(1+vol_+(2Pi-1))/2 and E=CD, E(Pi)=(1+vol_+)/2-vol_+Pi, map F_3 to itself (E(Pi_P)=Pi_(-P^perp)) and at random "
               f"points preserve the 43x43 mass matrix and the current bilinear forms -Tr([Q,Pi][Y,Pi]) for Q=Q_A, B_em and Y in so(8); E is the identity "
               f"on the range of (1+Z)/2, which contains P_*, so P_* E(Pi) P_*=P_* Pi P_*; on the four points {perm}: the Klein group acts simply transitively")

# ----------------------------------------------------------------------------------------------- p3-17 the four points on the two halves of Z
ok = close(PiA @ Zp, PimW @ Zp) and close(PiA @ Zp, Comb([(0.5, Isp), (-0.5, chiW)]) @ Zp)
ok = ok and close(PiW @ Zp, PimA @ Zp) and close(PiW @ Zp, Comb([(0.5, Isp), (0.5, chiW)]) @ Zp)
ok = ok and close(PiA @ Zm, PiW @ Zm) and close(PimA @ Zm, PimW @ Zm)
ok = ok and all(close(tL[i] @ Zp, PiA @ tau[i] @ Zp) and close(tR[i] @ Zp, PiW @ tau[i] @ Zp) for i in range(3))
ok = ok and close(Zp @ Pst, Pst)
report(17, ok, "n=3: on the half Z=+1 (range of (1+Z)/2, rank 256, containing P_*) Pi_A=Pi_-W=(1/2)(1-chi_W) and Pi_W=Pi_-A=(1/2)(1+chi_W), and "
               "t^L_i=Pi_A tau_i, t^R_i=Pi_W tau_i there: the vacuum Pi_A is, on the matter half, the support of su(2)_L; on the half Z=-1, Pi_A=Pi_W "
               "and Pi_-A=Pi_-W")

# ----------------------------------------------------------------------------------------------- p3-18 normalisers
ok = True
fixSM = null_space(np.vstack([rho_plus(x) for x in hSM_on]))
fixPS = null_space(np.vstack([rho_plus(x) for x in hPS_on]))
projA = np.zeros((8, 8))
for i in Aidx:
    projA[vpos[i], vpos[i]] = 1.0
ok = ok and fixSM.shape[1] == 4 and fixPS.shape[1] == 4 and np.allclose(fixSM @ fixSM.T, projA) and np.allclose(fixPS @ fixPS.T, projA)
refl = np.eye(8)
refl[vpos[15], vpos[15]] = -1.0
rL = [rho_plus(x) for x in tL]
rR = [rho_plus(x) for x in tR]
ok = ok and all(span_rank(rR + [refl @ m @ refl]) == 3 for m in rL) and all(span_rank(rL + [refl @ m @ refl]) == 3 for m in rR)
gPS = E[9] @ E[15]


def Ad(g, X):
    return Comb.of(g) @ Comb.of(X) @ Comb.of(g).T


ok = ok and (gPS @ gPS.T) == Isp and close(Ad(gPS, volA), -volA) and close(Ad(gPS, volW), -volW)


def in_span(X, basis):
    Xd = Comb.of(X).dense().ravel()
    Bm = np.array([Comb.of(b_).dense().ravel() for b_ in basis]).T
    c_ = np.linalg.lstsq(Bm, Xd, rcond=None)[0]
    return np.max(np.abs(Bm @ c_ - Xd)) < 1e-9


ok = ok and all(in_span(Ad(gPS, x), hPS_on) for x in hPS_on)
ok = ok and all(in_span(Ad(gPS, x), tR) for x in tL) and not in_span(Ad(gPS, tL[0]), hSM_on)
ok = ok and close(Ad(gPS, Xinv), -Xinv) and not close(Ad(gPS, Pst), Pst)
gAW = Comb([(0.25, Isp)])
for a_, w_ in zip(Aidx, Widx):
    gAW = gAW @ Comb([(1.0, Isp), (1.0, E[a_] @ E[w_])])
gAW_d = gAW.dense()
ok = ok and close(gAW_d @ gAW_d.T, Id)
adA = Ad(gAW, volA)
adW = Ad(gAW, volW)
ok = ok and close(adA, volW) and close(adW, volA)
ok = ok and not in_span(Ad(gAW, tL[2]), hSM_on) and not in_span(Ad(gAW, tL[2]), hPS_on)
ok = ok and volA @ volW == volPlus
report(18, ok, "n=3: the fixed vectors of h_SM and of h_PS in V_+ are exactly A, so every rotation normalising either preserves A and W; a reflection "
               "of W conjugates su(2)_L onto su(2)_R, so a rotation normalising h_SM preserves the orientations of A and W and fixes Pi_+-A, Pi_+-W; "
               "g=E9E15=exp(pi E9E15/2) normalises h_PS, exchanges su(2)_L and su(2)_R, sends vol_A to -vol_A, vol_W to -vol_W and X to -X (so "
               "Ad(g)P_*!=P_*); g_AW=prod_i (1+a_iw_i)/sqrt2 exchanges vol_A and vol_W and normalises neither h_SM nor h_PS; vol_A vol_W=vol_+")

# ----------------------------------------------------------------------------------------------- p3-19 invariant tangent vectors
algs = {"h_SM": hSM_on, "h_PS": hPS_on, "su2_L": su2L_on, "u1_em+su3": [Bem] + su3_on, "u1_Y+su3": [Y_on] + su3_on,
        "u1_A": [QA], "u1_A+u1_em": [QA, Bem]}
res19 = {}
ok = True
for nm in ("A", "W"):
    for an, gens in algs.items():
        Ms = [tangent_matrix(x, TB[nm]) for x in gens]
        res19[(nm, an)] = null_space(np.vstack(Ms)).shape[1]
want19 = {"h_SM": 0, "h_PS": 0, "su2_L": 0, "u1_em+su3": 8, "u1_Y+su3": 0, "u1_A": 8, "u1_A+u1_em": 4}
ok = all(res19[(nm, an)] == want19[an] for nm in ("A", "W") for an in algs)
report(19, ok, f"n=3: dimensions of the invariant tangent vectors at Pi_A and Pi_W (and hence at Pi_-A, Pi_-W, whose tangent spaces are the same up to "
               f"sign): h_SM 0, h_PS 0, su(2)_L 0, u(1)_Y+su(3)_C 0, u(1)_em+su(3)_C 8 (the neutral components A(x)W_0), u(1)_A 8, u(1)_A+u(1)_em 4 "
               f"(computed {res19})")

# ----------------------------------------------------------------------------------------------- p3-20 invariance of the overlaps, vanishing gradients
ok = True
for _ in range(2):
    F = np.linalg.qr(RNG.normal(size=(8, 4)))[0].T
    Pd = 0.5 * (Id + frame_vol(F).dense())
    MPM = {k_: M @ Pd @ M for k_, M in MATTER_d.items()}
    for x in hSM_on:
        cx = dcm(x, Pd)
        for k_, M in MATTER_d.items():
            ok = ok and abs(np.sum(cx * M.T)) < 1e-8
            if k_ in ("P_*", "P_L", "P_R", "Q_v", "P_lep"):
                ok = ok and abs(2 * np.sum(cx * MPM[k_].T)) < 1e-8
ok = ok and any(not iszero(cm(x, Pfam)) for x in su3_on)
for nm in four_d:
    MPM = {k_: M @ four_d[nm] @ M for k_, M in MATTER_d.items()}
    for Bb in kb_all:
        cx = dcm(Bb, four_d[nm])
        for k_, M in MATTER_d.items():
            ok = ok and abs(np.sum(cx * M.T)) < 1e-9 and abs(np.sum(cx * MPM[k_].T)) < 1e-9
report(20, ok, "n=3: at random points of F_3 the first variation of every linear overlap Tr(Pi M) (M in P_*, P_L, P_R, Pfam, P_lep, (1+-Gamma)/2, Q_v) "
               "and of the quadratic overlaps Tr(Pi M Pi M) for M commuting with h_SM vanishes along h_SM: they are h_SM-invariant functions on F_3, "
               "Pfam included although su(3)_C does not commute with Pfam; at the four points their first variation vanishes along all 43 generators of k")

# ----------------------------------------------------------------------------------------------- p3-21 Hessians
ok = True
res21 = {}
for nm in ("A", "-A", "W", "-W"):
    Pi0 = FOUR[nm][0]
    Pi0d = four_d[nm]
    vals = []
    for trial in range(2):
        x = RNG.normal(size=16)
        Xc = Comb([(0.5 * x[k_], E[i] @ E[j]) for k_, (i, j) in enumerate(mix)])
        g2, gY, gPSc = RNG.uniform(0.3, 1.5, size=3)
        qSM = sum(cc ** 2 * float(np.sum(dcm(simplify(cm(T_, Xc), 1e-12), Pi0d) ** 2)) for cc, T_ in zip([1.0] * 8 + [g2] * 3 + [gY], hSM_on))
        qPS = gPSc ** 2 * sum(float(np.sum(dcm(simplify(cm(T_, Xc), 1e-12), Pi0d) ** 2)) for T_ in hPS_on)
        cSM = 6 * g2 ** 2 + 1.2 * gY ** 2
        ok = ok and abs(qSM - 128 * cSM * x @ x) < 1e-7 * (1 + qSM) and abs(qPS - 128 * 12 * gPSc ** 2 * x @ x) < 1e-7 * (1 + qPS)
        XXP = dcm(Xc, dcm(Xc, Pi0d))
        hL = float(np.sum(XXP * PL_d.T))
        hF = float(np.sum(XXP * MATTER_d["Pfam"].T))
        vals.append((round(hL / (x @ x), 8), round(hF / (x @ x), 8)))
    res21[nm] = vals[0]
ok = ok and res21 == {"A": (-16.0, -32.0), "-A": (16.0, 32.0), "W": (16.0, 0.0), "-W": (-16.0, 0.0)}
# finite-difference check of the expansion at Pi_A
x = RNG.normal(size=16)
x /= np.linalg.norm(x)
Xd = Comb([(0.5 * x[k_], E[i] @ E[j]) for k_, (i, j) in enumerate(mix)]).dense()
t = 1e-3
VSM, fL = 0.0, 0.0
for sg in (1.0, -1.0):
    gt = expm(sg * t * Xd)
    Pt = gt @ PiA_d @ gt.T
    VSM += 0.5 * sum(float(np.sum(dcm(T_, Pt) ** 2)) for T_ in su2L_on + [Y_on])  # g2=gY=1, su(3) contributes zero on F_3
    fL += 0.5 * float(np.sum(Pt * PL_d.T))
ok = ok and abs(VSM / t ** 2 - 128 * 7.2) < 1e-4 * 128 * 7.2 and abs((32 - fL) / t ** 2 - 8) < 1e-3
report(21, ok, f"n=3: with Pi(x)=Ad(exp X)Pi_0, X=sum x_aw aw/2, at each of the four points V_SM=128(6g2^2+(6/5)gY^2)|x|^2+O(x^3) and "
               f"V_PS=128*12 g^2|x|^2+O(x^3) for bases orthonormal in -Tr/512; the second derivatives Tr([X,[X,Pi_0]]P)/|x|^2 of the overlaps with "
               f"(P_L, Pfam) are {res21} (so Tr(Pi P_L)=32-8|x|^2 at Pi_A and Pi_-W, 8|x|^2 at Pi_-A and Pi_W); finite differences at Pi_A agree")

# ----------------------------------------------------------------------------------------------- p3-22 the selection on F_3
cSM1 = 7.2  # 6 g2^2 + (6/5) gY^2 at g2 = gY = 1
ok = True
for _ in range(2):
    F = np.linalg.qr(RNG.normal(size=(8, 4)))[0].T
    Pd = 0.5 * (Id + frame_vol(F).dense())
    p8 = F.T @ F
    vR = sum(float(np.sum(dcm(T_, Pd) ** 2)) for T_ in su2L_on + [Y_on])
    vV = 64 * sum(float(np.sum((r_ @ p8 - p8 @ r_) ** 2)) for r_ in rho_su2L_on + [rho_Y_on])
    ok = ok and abs(vR - vV) < 1e-7


def Vsel(y, kappa, lam, mu):
    F = frame_from(y)
    p8 = F.T @ F
    vh = 64 * sum(float(np.sum((r_ @ p8 - p8 @ r_) ** 2)) for r_ in rho_su2L_on + [rho_Y_on])
    pA_, pW_ = np.linalg.det(F[:, ia_cols]), np.linalg.det(F[:, iw_cols])
    return kappa * vh - lam * 16 * (1 + pA_ - pW_) - mu * 32 * (1 + pA_)


res22 = []
for kappa, lam, mu in ((1.0, 0.5, 0.0), (1.0, 300.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.3)):
    vmin = 1e99
    winners = set()
    for _ in range(12):
        r_ = minimize(Vsel, RNG.normal(scale=1.5, size=28), args=(kappa, lam, mu), method="BFGS", options={"gtol": 1e-9})
        vmin = min(vmin, r_.fun)
        F = frame_from(r_.x)
        if r_.fun < -32 * lam - 64 * mu + 1e-6 * (1 + lam):
            pA_, pW_ = np.linalg.det(F[:, ia_cols]), np.linalg.det(F[:, iw_cols])
            winners.add("A" if abs(pA_ - 1) < 1e-4 else ("-W" if abs(pW_ + 1) < 1e-4 else "other"))
    target = -32 * lam - 64 * mu if mu > 0 else -32 * lam
    ok = ok and abs(vmin - target) < 1e-6 * (1 + abs(target))
    ok = ok and winners == ({"A"} if mu > 0 else {"A", "-W"})
    res22.append(((kappa, lam, mu), round(vmin, 6), sorted(winners)))
report(22, ok, f"n=3 (vector model, checked against R^512 at random points): the global minimum of V=kappa V_SM - lam Tr(Pi P_L) - mu Tr(Pi Pfam) on F_3 "
               f"(g2=gY=1) is -32 lam - 64 mu; for mu=0 it is attained exactly at Pi_A and Pi_-W, also for lam>16 kappa c_SM and for kappa=0, and for "
               f"mu>0 only at Pi_A (12 random starts each; computed {res22})")

# =============================================================================================== summary
n_ok = sum(RESULTS)
print(f"{n_ok}/{len(RESULTS)} checks passed in {time.time() - T_START:.1f}s")
sys.exit(0 if n_ok == len(RESULTS) else 1)
