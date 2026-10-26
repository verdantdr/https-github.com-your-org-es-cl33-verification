#!/usr/bin/env python3
"""p1: the compact families of Lorentz-invariant vacuum projectors at the second, third and fourth rung.

    python3 investigations/photon_selection/verify/p1_families.py

Prints one 'PASS p1-<n>: <claim>' (or 'FAIL ...') line per claim of sections/p1_families.tex and exits 0 iff all pass.

Realisation (tex/s02_arena.tex, tex/sA_realisation.tex): e1=sx(x)1(x)1, e2=sz(x)sx(x)1, e3=sz(x)sz(x)sx, e4=ep(x)1(x)1,
e5=sz(x)ep(x)1, e6=sz(x)sz(x)ep, ep=[[0,-1],[1,0]], omega=e1...e6, E_{6(k-1)+a}=omega^{(x)(k-1)} (x) e_a (x) 1...
Every generator and every blade is a signed permutation matrix.  Blades are handled exactly as signed permutations (class SP);
linear combinations of blades with dyadic coefficients (class Comb) are expanded exactly, and float64 is exact on them.
Floating point with tolerance TOL is used only at random points of the families (orthonormal frames, matrix exponentials),
for ranks of real spans and for null spaces; every such use is a comparison with an exact value.
"""
import itertools
import sys
import time

import numpy as np
from scipy.linalg import expm, null_space

T_START = time.time()
TOL = 1e-9
RNG = np.random.default_rng(20261021)
RESULTS = []


def report(n, ok, claim):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} p1-{n}: {claim}", flush=True)


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


# =============================================================================================== n = 2
E2, eta2 = tower(2)
N2 = 64
Id2 = np.eye(N2)
I2sp = ident(N2)
ok = all((E2[a] @ E2[a]) == (I2sp if eta2[a] > 0 else -I2sp) for a in range(1, 13))
ok = ok and all(anticommutes(E2[a], E2[b]) for a in range(1, 13) for b in range(a + 1, 13))
om8_dense = e8_dense[0] @ e8_dense[1] @ e8_dense[2] @ e8_dense[3] @ e8_dense[4] @ e8_dense[5]
ok = ok and all(np.array_equal(E2[6 * (k - 1) + a + 1].dense(), _kron(*([om8_dense] * (k - 1) + [e8_dense[a]] + [np.eye(8, dtype=np.int64)] * (2 - k))))
                for k in (1, 2) for a in range(6))
interior2 = [1, 4, 5, 6]
internal2 = [i for i in range(1, 13) if i not in interior2]
pos2 = [i for i in internal2 if eta2[i] > 0]
neg2 = [i for i in internal2 if eta2[i] < 0]
report(1, ok and pos2 == [2, 3, 7, 8, 9] and neg2 == [10, 11, 12],
       "n=2: E_1..E_12 (equal to the dense Kronecker products) satisfy E_aE_b+E_bE_a=2eta_ab; spectator internal complement has positive {2,3,7,8,9} and negative {10,11,12}, signature (5,3)")

D2 = [None] + [E2[i].dense() for i in range(1, 13)]
a, b, c, d, e = (D2[i] for i in (2, 3, 7, 8, 9))


def comm(x, y):
    return x @ y - y @ x


def acomm(x, y):
    return x @ y + y @ x


def prod(ms, N):
    out = np.eye(N)
    for m in ms:
        out = out @ m
    return out


Q2 = 0.5 * a @ b
H2 = a @ b @ c @ d
Pi0 = 0.5 * (Id2 + H2)
w2 = prod([D2[i] for i in internal2], N2)
B = 0.5 * b @ e
pi = comm(B, Pi0)
ok = (close(H2 @ H2, Id2) and close(H2.T, H2) and abs(np.trace(H2)) < TOL and abs(np.trace(Pi0) - 32) < TOL
      and close(Pi0 @ Pi0, Pi0) and iszero(comm(Q2, Pi0)) and all(iszero(comm(D2[i], Pi0)) for i in interior2)
      and iszero(comm(w2, Pi0)) and close(w2 @ w2, -Id2)
      and close(pi, -0.5 * a @ c @ d @ e) and close(comm(Q2, pi), 0.5 * b @ c @ d @ e) and close(pi.T, pi)
      and abs(np.trace(pi @ pi) - 16) < TOL)
report(2, ok, "n=2 restated (Thm 26.6): H=abcd=E2E3E7E8 symmetric traceless involution, Pi_0 rank 32, [Q,Pi_0]=0, Pi_0 commutes with E1,E4,E5,E6 and the internal volume w (w^2=-1); pi=[be/2,Pi_0]=-acde/2, [Q,pi]=bcde/2, Tr pi^2=16")

v5 = [a, b, c, d, e]
I5 = a @ b @ c @ d @ e
Z = [I5 @ v for v in v5]
ok = all(close(Z[i].T, Z[i]) for i in range(5)) and all(close(acomm(Z[i], Z[j]), 2 * (i == j) * Id2) for i in range(5) for j in range(5))
ok = ok and close(Z[4], H2)
T = [comm(0.5 * v5[i] @ e, Pi0) for i in range(4)]
ok = ok and all(close(T[i], 0.5 * Z[i]) for i in range(4))
ok = ok and all(abs(np.trace(T[i] @ T[j]) - 16 * (i == j)) < TOL for i in range(4) for j in range(4))
ok = ok and close(comm(Q2, T[0]), -T[1]) and close(comm(Q2, T[1]), T[0]) and iszero(comm(Q2, T[2])) and iszero(comm(Q2, T[3]))
report(3, ok, "n=2 restated (Thm 26.7): Z_i=I_5 v_i symmetric with Z_iZ_j+Z_jZ_i=2delta_ij, Z_e=H; T_i=[v_ie/2,Pi_0]=I_5v_i/2, Tr(T_iT_j)=16delta_ij; [Q,T_a]=-T_b, [Q,T_b]=T_a, [Q,T_c]=[Q,T_d]=0")


def Pi_n2(n):
    return 0.5 * (Id2 + sum(n[i] * Z[i] for i in range(5)))


ok = True
for _ in range(5):
    n = RNG.normal(size=5)
    n /= np.linalg.norm(n)
    dn = RNG.normal(size=5)
    dn -= (dn @ n) * n
    P = Pi_n2(n)
    dP = 0.5 * sum(dn[i] * Z[i] for i in range(5))
    ok = ok and close(P @ P, P) and close(P.T, P) and abs(np.trace(P) - 32) < 1e-8
    ok = ok and all(iszero(comm(D2[i], P)) for i in interior2)
    ok = ok and abs(np.trace(dP @ dP) - 16 * dn @ dn) < 1e-8
    ok = ok and abs(-np.trace(comm(Q2, P) @ dP) - 16 * (n[0] * dn[1] - n[1] * dn[0])) < 1e-8
    nv = sum(n[i] * v5[i] for i in range(5))
    dnv = sum(dn[i] * v5[i] for i in range(5))
    ok = ok and close(comm(P, dP), 0.25 * comm(nv, dnv))
    ev = np.linalg.eigvalsh(P)
    ok = ok and np.allclose(np.sort(ev), np.r_[np.zeros(32), np.ones(32)], atol=1e-8)
report(4, ok, "n=2 restated (Thms 26.7, 26.8): at random n, Pi(n) is a rank-32 Lorentz-invariant orthogonal projector with spectrum {0^32,1^32}; Tr(dPi^2)=16|dn|^2, -Tr([Q,Pi]dPi)=16(n_a dn_b-n_b dn_a), [Pi,dPi]=(1/4)[n,dn]")

# K_2 orbit
k2pairs = list(itertools.combinations(pos2, 2)) + list(itertools.combinations(neg2, 2))
k2 = [0.5 * D2[i] @ D2[j] for i, j in k2pairs]
ok = len(k2) == 13 and all(close(X @ X, -0.25 * Id2) for X in k2)
Zspan = np.array([z.ravel() for z in Z])
ok_inv = True
for (i, j), X in zip(k2pairs, k2):
    for zi, z in enumerate(Z):
        Cz = comm(X, z).ravel()
        coef = np.linalg.lstsq(Zspan.T, Cz, rcond=None)[0]
        ok_inv = ok_inv and np.max(np.abs(Zspan.T @ coef - Cz)) < TOL
        if eta2[i] < 0:
            ok_inv = ok_inv and np.max(np.abs(Cz)) < TOL  # so(3) acts trivially on span{Z}
stab = sum(1 for X in k2 if iszero(comm(X, Pi0)))
orbit_tan = span_rank([comm(X, Pi0) for X in k2])
ok_tr = True
for i in range(4):
    th = RNG.uniform(0, np.pi)
    g = expm(0.5 * th * v5[i] @ e)
    n = np.zeros(5)
    n[4], n[i] = np.cos(th), np.sin(th)
    ok_tr = ok_tr and close(g @ Pi0 @ g.T, Pi_n2(n)) and close(g @ g.T, Id2)
# every point of S^4 is Pi_P for the oriented 4-plane P = n^perp in V_+ (5-dim): I_5 n = vol(n^perp)
ok_gr = True
for _ in range(3):
    n = RNG.normal(size=5)
    n /= np.linalg.norm(n)
    Fm = null_space(n[None, :]).T
    if np.linalg.det(np.vstack([Fm, n])) < 0:
        Fm[0] *= -1
    u = [sum(Fm[r, s] * v5[s] for s in range(5)) for r in range(4)]
    ok_gr = ok_gr and close(sum(n[k] * Z[k] for k in range(5)), u[0] @ u[1] @ u[2] @ u[3])
    # the 13-blade alignment energy is constant on S^4
    VK = sum(np.sum(comm(2 * X, Pi_n2(n)) ** 2) for X in k2) / 64.0
    ok_gr = ok_gr and abs(VK - 4.0) < 1e-8
report(5, ok and ok_inv and stab == 9 and orbit_tan == 4 and ok_tr and ok_gr,
       "n=2: the 13 compact internal bivectors preserve span{Z_i} (so(5) by the vector representation, so(3) trivially); the stabiliser of Pi_0 in k_2 has dim 9, the orbit tangent dim 4, and exp(+theta v_i e/2) carries Pi(e) to Pi(cos theta e+sin theta v_i): the K_2-orbit of Pi_0 is S^4; I_5 n = vol(n^perp), so S^4 is the oriented Grassmannian of 4-planes in V_+; (1/64)sum_13||[e_ie_j,Pi(n)]||^2=4 on S^4")

ok = iszero(comm(Q2, Z[2])) and iszero(comm(Q2, Z[3])) and iszero(comm(Q2, Z[4])) and close(comm(Q2, Z[0]), -Z[1]) and close(comm(Q2, Z[1]), Z[0])
report(6, ok, "n=2: [Q,I_5 v]=I_5[Q,v]: Q fixes Z_c,Z_d,Z_e and rotates (Z_a,Z_b), so [Q,Pi(n)]=0 iff n_a=n_b=0 (a two-sphere: the planes containing span(a,b))")

allbiv2 = list(itertools.combinations(internal2, 2))
H2sp = blade(E2, [2, 3, 7, 8])
stab_all = [(i, j) for i, j in allbiv2 if commutes(E2[i] @ E2[j], H2sp)]
stab_c = [(i, j) for i, j in stab_all if eta2[i] == eta2[j]]
P2 = {2, 3, 7, 8}
ok = len(allbiv2) == 28 and len(stab_all) == 12 and len(stab_c) == 9
ok = ok and all(({i, j} <= P2) or not ({i, j} & P2) for i, j in stab_all)
ok = ok and all(anticommutes(E2[i] @ E2[j], H2sp) for i, j in allbiv2 if (i, j) not in stab_all)
report(7, ok, "n=2 restated (Prop 26.9): stabiliser of H in the internal so(5,3) (dim 28) is so(P)+so(P^perp)=so(4)+so(1,3), dim 12, compact part 9 = so(4)+so(3); every other internal bivector anticommutes with H")

# =============================================================================================== n = 3
t_n3 = time.time()
E, eta = tower(3)
N = 512
Id = np.eye(N)
Isp = ident(N)
AR = np.arange(N)
ok = all((E[i] @ E[i]) == (Isp if eta[i] > 0 else -Isp) for i in range(1, 19))
ok = ok and all(anticommutes(E[i], E[j]) for i in range(1, 19) for j in range(i + 1, 19))
ok = ok and all(E[i].T == (E[i] if eta[i] > 0 else -E[i]) for i in range(1, 19))
interior = [1, 4, 5, 6]
internal = [i for i in range(1, 19) if i not in interior]
Vp = [i for i in internal if eta[i] > 0]
Vm = [i for i in internal if eta[i] < 0]
Aidx, Widx, Cidx = [2, 3, 8, 9], [7, 13, 14, 15], [10, 11, 12, 16, 17, 18]
kpos_pairs = list(itertools.combinations(Vp, 2))
kneg_pairs = list(itertools.combinations(Vm, 2))
kbiv = kpos_pairs + kneg_pairs
ok = ok and sorted(Aidx + Widx) == Vp and Cidx == Vm and len(kbiv) == 43 and len(list(itertools.combinations(internal, 2))) == 91
ok = ok and all((E[i] @ E[j]) @ (E[i] @ E[j]) == -Isp for i, j in kbiv)
ok = ok and all((E[i] @ E[j]) @ (E[i] @ E[j]) == Isp for i in Vp for j in Vm)
report(8, ok, "n=3: E_1..E_18 satisfy the Clifford relations of R^{9,9} (E_i^T=eta_i E_i); V_+=span(A u W)={2,3,7,8,9,13,14,15}, V_-=span(C)={10,11,12,16,17,18}; the 43 compact internal bivectors (28+15) square to -1 and the 48 mixed ones to +1")

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
PiA_d = PiA.dense()
PiW_d = PiW.dense()

# ---- classification of four-generator blades commuting with the interior gammas
types = {}
ok = True
n_lor = 0
for S in itertools.combinations(range(1, 19), 4):
    Bm = blade(E, S)
    lor = all(commutes(g_, Bm) for g_ in gam)
    if lor != (not set(S) & set(interior)):
        ok = False
    if not lor:
        continue
    n_lor += 1
    q = sum(1 for i in S if eta[i] < 0)
    p = 4 - q
    sq = Bm @ Bm
    sym = Bm.T == Bm
    invol = sq == Isp
    cplx = sq == -Isp
    ok = ok and Bm.trace() == 0 and (invol == (q % 2 == 0)) and (cplx == (q % 2 == 1)) and (sym == invol)
    types[(p, q)] = types.get((p, q), 0) + 1
report(9, ok and n_lor == 1001 and types == {(4, 0): 70, (3, 1): 336, (2, 2): 420, (1, 3): 160, (0, 4): 15},
       "n=3: of the 3060 four-generator blades exactly the 1001 internal ones commute with the four interior gammas; all are traceless; a blade of type (p,q) is a symmetric involution iff q is even and an antisymmetric complex structure iff q is odd; counts (4,0):70, (3,1):336, (2,2):420, (1,3):160, (0,4):15")

# census of all internal blades of even grade: symmetric involution iff r(r-1)/2+q is even (theta-stable orbit types)
ok = True
census = {}
kinv = []
for r in range(2, 15, 2):
    for S in itertools.combinations(internal, r):
        q = sum(1 for i in S if eta[i] < 0)
        Bm = blade(E, S)
        inv = (Bm @ Bm) == Isp
        if inv != ((r * (r - 1) // 2 + q) % 2 == 0):
            ok = False
        if inv:
            ok = ok and Bm.T == Bm and all(commutes(g_, Bm) for g_ in gam)
            census[(r, q)] = census.get((r, q), 0) + 1
            if all(commutes(E[i] @ E[j], Bm) for i, j in kbiv):
                kinv.append(S)
ok = ok and (14, 6) not in census and census.get((8, 0)) == 1 and (6, 6) not in census and kinv == [tuple(Vp)]
okZ = volPlus == -(J @ chiC) or volPlus == (J @ chiC)
okZ = okZ and all(commutes(E[i] @ E[j], volPlus) for i, j in kbiv) and all(commutes(g_, volPlus) for g_ in gam)
report(10, ok and okZ, "n=3: an internal blade of even grade r with q negative factors is a (symmetric) involution iff r(r-1)/2+q is even (checked on all 8191 even internal blades); the only K-invariant one is vol_+=E2E3E7E8E9E13E14E15 (grade 8), equal to +-J chi_C = -+Z and commuting with all 43 compact bivectors and the interior gammas")

# stabilisers of representatives
reps = {(4, 0): [2, 3, 8, 9], (2, 2): [2, 3, 10, 16], (0, 4): [10, 16, 11, 17]}
allint = list(itertools.combinations(internal, 2))
stabdat = {}
ok_split = True
for t_, S in reps.items():
    V = blade(E, S)
    st_all = [(i, j) for i, j in allint if commutes(E[i] @ E[j], V)]
    ok_split = ok_split and all(anticommutes(E[i] @ E[j], V) for i, j in allint if (i, j) not in st_all)
    ok_split = ok_split and all(({i, j} <= set(S)) or not ({i, j} & set(S)) for i, j in st_all)
    st_k = [(i, j) for i, j in st_all if eta[i] == eta[j]]
    movers = sum(1 for i, j in kbiv if (i, j) not in st_all)
    stabdat[t_] = (len(st_all), len(st_k), movers)
ok = stabdat == {(4, 0): (51, 27, 16), (2, 2): (51, 23, 20), (0, 4): (51, 35, 8)} and ok_split
report(11, ok, f"n=3: stabiliser of vol_P in so(8,6) is so(P)+so(P^perp) (dim 51 for every type); compact part / K-orbit dimension: (4,0) 27/16 [so(4)+so(4)+so(6)], (2,2) 23/20, (0,4) 35/8; every other bivector anticommutes with vol_P (computed {stabdat})")

# ---- vector representation and lift
def lift(M, idx):
    """sum_{i<j} M_ij e_ie_j/2 on an orthonormal positive frame, with an overall sign -1 on a negative frame."""
    sgn = 1.0 if eta[idx[0]] > 0 else -1.0
    return Comb([(sgn * M[p_, q_] * 0.5, E[idx[p_]] @ E[idx[q_]]) for p_, q_ in itertools.combinations(range(len(idx)), 2) if M[p_, q_] != 0])


ok = True
for idx in (Vp, Vm):
    dim = len(idx)
    for _ in range(2):
        X = RNG.normal(size=(dim, dim))
        X = X - X.T
        L = lift(X, idx)
        ok = ok and np.allclose(vec_rep(L, E, idx), X, atol=1e-12)
        other = Vm if idx is Vp else Vp
        ok = ok and np.allclose(vec_rep(L, E, other), 0, atol=1e-12) and all(iszero(cm(L, g_)) for g_ in gam)
        Y = RNG.normal(size=(dim, dim))
        Y = Y - Y.T
        ok = ok and close(cm(lift(X, idx), lift(Y, idx)), lift(X @ Y - Y @ X, idx).dense())
report(12, ok, "n=3: the lift M -> sum_{i<j} M_ij e_ie_j/2 (sign -1 on the negative frame) is a Lie algebra map so(V_+)+so(V_-) -> k with [lift M,u]=Mu on V_+ (resp. V_-), acting trivially on the other span and commuting with the interior gammas")

# ---- transpose lemma: vol_P^T = vol_{theta P}; P is the -1 eigenspace of Ad(vol_P) on internal vectors
eta_int = np.array([eta[i] for i in internal], dtype=float)


def eta_ip(x, y):
    return float(np.sum(eta_int * x * y))


def vec_op(x):
    return Comb([(x[k_], E[i]) for k_, i in enumerate(internal) if x[k_] != 0])


ok = True
iVp_int = [internal.index(i) for i in Vp]
iVm_int = [internal.index(i) for i in Vm]
for _ in range(2):
    # boosted orthonormal frame: u_i = (a_i + t_i b_i)/sqrt(1-t_i^2), a_i orthonormal in V_+, b_i orthonormal in V_-
    Aq = np.linalg.qr(RNG.normal(size=(8, 4)))[0]
    Bq = np.linalg.qr(RNG.normal(size=(6, 4)))[0]
    tt = RNG.uniform(0.2, 0.6, size=4)
    us = []
    for r in range(4):
        x = np.zeros(14)
        x[iVp_int] = Aq[:, r]
        x[iVm_int] = tt[r] * Bq[:, r]
        us.append(x / np.sqrt(1 - tt[r] ** 2))
    ok = ok and all(abs(eta_ip(us[i], us[j]) - (i == j)) < 1e-12 for i in range(4) for j in range(4))
    vP = (vec_op(us[0]) @ vec_op(us[1]) @ vec_op(us[2]) @ vec_op(us[3])).dense()
    th = [x * eta_int for x in us]  # theta(u) = u_+ - u_-
    vthP = (vec_op(th[0]) @ vec_op(th[1]) @ vec_op(th[2]) @ vec_op(th[3])).dense()
    ok = ok and close(vP @ vP, Id) and all(iszero(dcm(g_, vP)) for g_ in gam) and close(vP.T, vthP) and not close(vP.T, vP)
    for x in us:
        ok = ok and close(vec_op(x).lmul(vP), -vec_op(x).rmul(vP))
    yv = RNG.normal(size=14)
    for x in us:
        yv = yv - eta_ip(yv, x) * x
    ok = ok and close(vec_op(yv).lmul(vP), vec_op(yv).rmul(vP))
report(13, ok, "n=3: for a positive-definite internal 4-plane P not inside V_+ (tilted into V_-), vol_P is a Lorentz-invariant involution with vol_P^T = vol_{theta P} != vol_P (theta = +1 on V_+, -1 on V_-), so Pi_P is not an orthogonal projector; internal vectors in P anticommute and vectors eta-orthogonal to P commute with vol_P (P is recovered from vol_P)")

# ---- K-orbits of the three types: Ad(g) vol_P = vol_{RP}
lift8 = lambda M: lift(M, Vp)
lift6 = lambda M: lift(M, Vm)


def frame_vol(F, idx):
    """vol of the oriented frame with rows F (coordinates on the generators idx), as a Comb."""
    out = None
    for r in range(F.shape[0]):
        u = Comb([(F[r, s], E[idx[s]]) for s in range(len(idx)) if F[r, s] != 0])
        out = u if out is None else out @ u
    return out


ok = True
okc = True
for trial in range(2):
    Y8 = RNG.normal(size=(8, 8))
    Y8 = Y8 - Y8.T
    Y6 = RNG.normal(size=(6, 6))
    Y6 = Y6 - Y6.T
    g = expm((lift8(Y8) + lift6(Y6)).dense())
    R8, R6 = expm(Y8), expm(Y6)
    ok = ok and close(g @ g.T, Id, 1e-8)
    for t_, S in reps.items():
        V = blade(E, S)
        rot = V.rmul(g) @ g.T  # g vol g^T
        rows = []
        for i in S:
            if i in Vp:
                rows.append((R8[:, Vp.index(i)], Vp))
            else:
                rows.append((R6[:, Vm.index(i)], Vm))
        out = None
        for vecc, idx in rows:
            u = Comb([(vecc[s], E[idx[s]]) for s in range(len(idx))])
            out = u if out is None else out @ u
        ok = ok and close(rot, out.dense(), 1e-8)
report(14, ok, "n=3: for g=exp(lift(Y8+Y6)) in K and (R8,R6)=(exp Y8,exp Y6), g vol_P g^T = vol_{RP} for representatives of all three types: the K-orbits are the theta-stable oriented planes of each type, (4,0): G~r(4,V_+) dim 16, (2,2): G~r(2,V_+)xG~r(2,V_-)/+-1 dim 20, (0,4): G~r(4,V_-) dim 8")

# ---- colour su(3) in the vector model of V_- and its lift
Hc = Comb([(-1.0, E[10] @ E[16]), (-1.0, E[11] @ E[17]), (-1.0, E[12] @ E[18])])
rhoH = vec_rep(Hc, E, Vm)
so6 = []
for p_, q_ in itertools.combinations(range(6), 2):
    M = np.zeros((6, 6))
    M[p_, q_], M[q_, p_] = 1.0, -1.0
    so6.append(M)
Acm = np.array([(M @ rhoH - rhoH @ M).ravel() for M in so6]).T
nsu = null_space(Acm)
u3 = [sum(nsu[k_, m_] * so6[k_] for k_ in range(15)) for m_ in range(nsu.shape[1])]
hn = rhoH / np.linalg.norm(rhoH)
su3v = [M - np.sum(M * hn) * hn for M in u3]
su3v = orth_basis(su3v)
der = span_rank([X @ Y - Y @ X for X in su3v for Y in su3v])
closed = all(span_rank(su3v + [X @ Y - Y @ X]) == 8 for X in su3v for Y in su3v)
# commutant of su(3) on R^6
Ms = np.array([np.kron(np.eye(6), R) - np.kron(R.T, np.eye(6)) for R in su3v]).reshape(-1, 36)
commutant = null_space(Ms)
Jc = rhoH / 2.0
ok = (nsu.shape[1] == 9 and len(su3v) == 8 and der == 8 and closed and commutant.shape[1] == 2
      and np.allclose(Jc @ Jc, -np.eye(6)) and span_rank([commutant[:, 0].reshape(6, 6), commutant[:, 1].reshape(6, 6), np.eye(6), Jc]) == 2)
su3 = [lift6(M) for M in su3v]
ok = ok and all(iszero(cm(X, Hc)) for X in su3) and all(np.allclose(vec_rep(X, E, Vm), M, atol=1e-12) for X, M in zip(su3, su3v))
ok = ok and all(np.allclose(vec_rep(X, E, Vp), 0, atol=1e-12) for X in su3)
report(15, ok, "n=3: the centraliser of the colour Cartan element H=-(E10E16+E11E17+E12E18) in so(6)_C is u(3) (dim 9); its part orthogonal to H is su(3)_C (dim 8, perfect), lifted to k it commutes with H and acts trivially on V_+; its commutant on V_-=R^6 is span(1,H/2) with (H/2)^2=-1, a field: su(3)_C acts irreducibly on V_-")

ok = True
for t_ in ((2, 2), (0, 4)):
    V = blade(E, reps[t_])
    ok = ok and max(nz(cm(X, V)) for X in su3) > 0.1
    for _ in range(2):
        Y6 = RNG.normal(size=(6, 6))
        Y6 = Y6 - Y6.T
        Y8 = RNG.normal(size=(8, 8))
        Y8 = Y8 - Y8.T
        g = expm((lift8(Y8) + lift6(Y6)).dense())
        Vg = V.rmul(g) @ g.T
        ok = ok and max(nz(dcm(X, Vg)) for X in su3) > 0.1
okA = all(iszero(cm(E[i] @ E[j], volA)) for i, j in kneg_pairs)
report(16, ok and okA, "n=3 (colour test): all of so(6)_C commutes with vol_A; at the (2,2) and (0,4) representatives and at random K-translates of them some su(3)_C generator fails to commute, as irreducibility of su(3)_C on V_- requires for every point of those two orbits")

# ---- weak algebra, hypercharge, electric charge (Section 8 of ES17)
w = [E[i] for i in Widx]
tau = [Comb([(-0.5, w[1] @ w[2])]), Comb([(-0.5, w[2] @ w[0])]), Comb([(-0.5, w[0] @ w[1])])]
PmW = Comb([(0.5, Isp), (-0.5, chiW)])
PpW = Comb([(0.5, Isp), (0.5, chiW)])
tL = [PmW @ t for t in tau]
tR = [PpW @ t for t in tau]
Wbiv = [E[i] @ E[j] for i, j in itertools.combinations(Widx, 2)]


def in_blade_span(X, blades):
    X = Comb.of(X)
    coef = [(Bb.T @ X).trace() / N for Bb in blades]
    return iszero(X - Comb([(c_, Bb) for c_, Bb in zip(coef, blades)]))


ok = all(in_blade_span(x, Wbiv) for x in tL + tR)
ok = ok and close(tL[2], Comb([(-0.25, w[0] @ w[1]), (-0.25, w[2] @ w[3])])) and close(tR[2], Comb([(-0.25, w[0] @ w[1]), (0.25, w[2] @ w[3])]))
ok = ok and all(close(cm(tL[i], tL[(i + 1) % 3]), tL[(i + 2) % 3]) for i in range(3))
ok = ok and all(close(cm(tR[i], tR[(i + 1) % 3]), tR[(i + 2) % 3]) for i in range(3))
ok = ok and all(iszero(cm(x, y)) for x in tL for y in tR)
report(17, ok, "n=3 restated (Thm 8.2): t^{L,R}_i=(1/2)(1-+chi_W)tau_i, tau_i=-w_jw_k/2, are bivectors in so(4)_W, t^L_3=-(w1w2+w3w4)/4, t^R_3=-(w1w2-w3w4)/4, two commuting su(2) algebras")

Xinv = Gam @ chiW
Zinv = -(J @ chiC)
Uinv = -(J @ qA)
Pst = Comb([(1.0 / 8, Isp)]) @ (Comb.of(Isp) + Xinv) @ (Comb.of(Isp) + Zinv) @ (Comb.of(Isp) + Uinv)
Pst_d = Pst.dense()
Yb = tR[2] + Hc * (1.0 / 6)
Bem = tau[2] + Hc * (1.0 / 6)
ok = close(Bem, Comb([(-0.5, E[7] @ E[13])]) + Hc * (1.0 / 6)) and close(tL[2] + tR[2], tau[2])
Qop = Pst_d @ (Comb.of(-J) @ Bem).lmul(Pst_d)
ev = np.linalg.eigvalsh(0.5 * (Qop + Qop.T))
vals, cnts = np.unique(np.round(ev * 3).astype(int), return_counts=True)
dist = dict(zip(vals.tolist(), cnts.tolist()))
dist[0] -= 448
ok = ok and abs(np.trace(Pst_d) - 64) < TOL and dist == {-3: 8, -1: 24, 0: 8, 2: 24} and iszero(cm(J, Bem))
report(18, ok, "n=3 restated (Prop 8.4): u(1)_em is spanned by B_em=tau_3+H/6=-E7E13/2-(E10E16+E11E17+E12E18)/6, the bivector part of Q=-J(t^3_L+t^3_R+H/6); on the range of P_* (rank 64) -J B_em has eigenvalues -1 (8), -1/3 (24), 0 (8), 2/3 (24)")

# ---- Clifford spheres in F_3
I5A = E[2] @ E[3] @ E[8] @ E[9] @ E[7]
Lgen_idx = [2, 3, 8, 9, 7]
ZA = [I5A @ E[i] for i in Lgen_idx]
ok = all(Zi.T == Zi for Zi in ZA) and all((ZA[i] @ ZA[i]) == Isp for i in range(5)) and all(anticommutes(ZA[i], ZA[j]) for i in range(5) for j in range(i + 1, 5))
ok = ok and ZA[4] == volA
U3 = E[3] @ E[8] @ E[9]
Ogen_idx = [2, 7, 13, 14, 15]
ZO = [E[i] @ U3 for i in Ogen_idx]
ok = ok and all(Zi.T == Zi for Zi in ZO) and all((ZO[i] @ ZO[i]) == Isp for i in range(5)) and all(anticommutes(ZO[i], ZO[j]) for i in range(5) for j in range(i + 1, 5))
ok = ok and ZO[0] == volA
TA = {(i, j): Comb([(0.5, E[i] @ E[j] @ volA)]) for i in Aidx for j in Widx}
ok = ok and all(close(Comb([(0.5, ZA[k_])]), TA[(Aidx[k_], 7)]) or close(Comb([(0.5, ZA[k_])]), -TA[(Aidx[k_], 7)]) for k_ in range(4))
ok = ok and all(close(Comb([(0.5, ZO[k_])]), TA[(2, Widx[k_ - 1])]) or close(Comb([(0.5, ZO[k_])]), -TA[(2, Widx[k_ - 1])]) for k_ in range(1, 5))
for _ in range(2):
    n = RNG.normal(size=5)
    n /= np.linalg.norm(n)
    Zn = Comb([(n[i], ZA[i]) for i in range(5)])
    Fm = null_space(n[None, :]).T
    if np.linalg.det(np.vstack([Fm, n])) < 0:
        Fm[0] *= -1
    ok = ok and close(Zn, frame_vol(Fm, Lgen_idx))
    m3 = RNG.normal(size=5)
    m3 /= np.linalg.norm(m3)
    Zm = Comb([(m3[i], ZO[i]) for i in range(5)])
    # plane spanned by m (in span(E2,W)) and E3,E8,E9: vol = m E3E8E9
    u_m = Comb([(m3[i], E[Ogen_idx[i]]) for i in range(5)])
    ok = ok and close(Zm, u_m @ E[3] @ E[8] @ E[9])
report(19, ok, "n=3: two kinds of Clifford four-spheres through Pi_A: inner {(1+I_5 n)/2}, I_5=E2E3E8E9E7, = {Pi_P : P in L=A+R E7} with tangent A(x)E7, and outer {(1+n E3E8E9)/2}, n in span(E2,E7,E13,E14,E15), = {Pi_P : P contains span(E3,E8,E9)} with tangent E2(x)W; both Z-systems are symmetric Clifford systems and (1+I_5 n)/2 = Pi_P with P = n^perp in L")

# duality P -> P^perp by vol_+
ok = (volPlus @ volA == volW) or (volPlus @ volA == -volW)
for _ in range(2):
    Rr = np.linalg.qr(RNG.normal(size=(8, 8)))[0]
    F = Rr[:4]
    vP = frame_vol(F, Vp).dense()
    dual = volPlus.lmul(vP)
    ok = ok and close(dual, dual.T) and close(dual @ dual, Id) and close(volPlus.lmul(vP), volPlus.rmul(vP))
    for r in range(8):
        u = Comb([(Rr[r, s], E[Vp[s]]) for s in range(8)])
        sgn = -1 if r >= 4 else 1  # vectors of P^perp anticommute with vol_+ vol_P, vectors of P commute
        ok = ok and close(u.lmul(dual), sgn * u.rmul(dual))
report(20, ok, "n=3: vol_+ commutes with every vol_P, vol_+ vol_A = +-vol_W, and at random P the operator vol_+ vol_P is a symmetric involution whose anticommuting vectors are exactly P^perp in V_+: P -> P^perp is the K-equivariant involution Pi -> (1+vol_+(2Pi-1))/2 of F_3, exchanging Pi_A and Pi_{+-W}")

# =============================================================================================== the vacuum Pi_A
QA = Comb([(0.5, qA)])
hPS = [Comb([(0.5, E[i] @ E[j])]) for i, j in itertools.combinations(Widx, 2)] + [Comb([(0.5, E[i] @ E[j])]) for i, j in itertools.combinations(Cidx, 2)]
hSM = su3 + tL + [Yb]
ok = (volA @ volA == Isp and volA.T == volA and volA.trace() == 0 and all(commutes(g_, volA) for g_ in gam)
      and commutes(qA, volA) and commutes(J, volA) and iszero(cm(Bem, volA))
      and all(iszero(cm(x, volA)) for x in hPS) and all(iszero(cm(x, volA)) for x in hSM)
      and not iszero(cm(Comb([(0.5, E[3] @ E[7])]), volA)))
okW = (volW @ volW == Isp and volW.T == volW and all(commutes(g_, volW) for g_ in gam) and commutes(qA, volW) and commutes(J, volW)
       and iszero(cm(Bem, volW)) and all(iszero(cm(x, volW)) for x in hPS) and all(iszero(cm(x, volW)) for x in hSM)
       and not iszero(cm(Comb([(0.5, E[3] @ E[7])]), volW)))
report(21, ok and okW, "n=3: Pi_A=(1+E2E3E8E9)/2 and Pi_W=(1+chi_W)/2 are rank-256 orthogonal projectors commuting with the interior gammas, J, q_A, B_em, h_PS (21 generators) and h_SM (12 generators), and are moved by E3E7/2")

Hlit = blade(E, [2, 3, 7, 8])
ok = commutes(qA, Hlit) and not iszero(cm(Bem, Hlit)) and any(not iszero(cm(x, Hlit)) for x in tL) and all(commutes(g_, Hlit) for g_ in gam)
ok = ok and Hlit.T == Hlit and Hlit @ Hlit == Isp
report(22, ok, "n=3: the literal embedding H=E2E3E7E8 of the second-rung vacuum (c,d=E7,E8) is a point of F_3 commuting with q_A, but not with B_em nor with su(2)_L, because E7 lies in W")

# tangent space at Pi_A and Pi_W
mix = [(i, j) for i in Aidx for j in Widx]
TW = {(i, j): Comb([(0.5, E[i] @ E[j] @ volW)]) for i, j in mix}
ok = all(close(cm(Comb([(0.5, E[i] @ E[j])]), PiA), TA[(i, j)]) for i, j in mix)
ok = ok and all(close(cm(Comb([(0.5, E[i] @ E[j])]), PiW), TW[(i, j)]) for i, j in mix)
moversA = sorted((i, j) for i, j in kbiv if not commutes(E[i] @ E[j], volA))
moversW = sorted((i, j) for i, j in kbiv if not commutes(E[i] @ E[j], volW))
ok = ok and moversA == sorted(tuple(sorted(m)) for m in mix) and moversW == moversA
gramA = np.array([[(TA[x] @ TA[y]).trace() for y in mix] for x in mix])
gramW = np.array([[(TW[x] @ TW[y]).trace() for y in mix] for x in mix])
ok = ok and np.array_equal(gramA, 128 * np.eye(16)) and np.array_equal(gramW, 128 * np.eye(16))
ok = ok and all(close(TA[m].T, TA[m]) for m in mix)
report(23, ok, "n=3: the compact bivectors moving Pi_A (and Pi_W) are exactly the 16 mixing A with W; T_aw=[aw/2,Pi_A]=aw vol_A/2 are symmetric with Tr(T_aw T_a'w')=128 delta (the real trace on R^512); likewise at Pi_W")


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


soA = [Comb([(0.5, E[i] @ E[j])]) for i, j in itertools.combinations(Aidx, 2)]
soW = [Comb([(0.5, E[i] @ E[j])]) for i, j in itertools.combinations(Widx, 2)]
ok = True
for Td in (TA, TW):
    rho = [tangent_matrix(X, Td) for X in soA + soW]
    Ms = np.array([np.kron(np.eye(16), R) - np.kron(R.T, np.eye(16)) for R in rho])
    comm_dim = null_space(Ms.reshape(-1, 256)).shape[1]
    ok = ok and comm_dim == 1
    for basis in (soA, soW):
        Mb = [tangent_matrix(X, Td) for X in basis]
        sdual = [Mb[0] + Mb[5], Mb[1] - Mb[4], Mb[2] + Mb[3]]
        adual = [Mb[0] - Mb[5], Mb[1] + Mb[4], Mb[2] - Mb[3]]
        ok = ok and np.allclose(-sum((0.5 * m) @ (0.5 * m) for m in sdual), 0.75 * np.eye(16))
        ok = ok and np.allclose(-sum((0.5 * m) @ (0.5 * m) for m in adual), 0.75 * np.eye(16))
report(24, ok, "n=3: at Pi_A and at Pi_W the tangent space is the real irreducible (2,2;2,2)=R^4(x)R^4 of so(4)_A+so(4)_W (commutant of dimension 1; each of the four su(2) Casimirs equals 3/4)")

ok = True
summary = {}
for name, Td in (("A", TA), ("W", TW)):
    rhoL = [tangent_matrix(X, Td) for X in tL]
    rhoY = tangent_matrix(Yb, Td)
    rho3 = [tangent_matrix(X, Td) for X in su3]
    ok = ok and np.allclose(-sum(m @ m for m in rhoL), 0.75 * np.eye(16)) and np.allclose(rhoY @ rhoY, -0.25 * np.eye(16))
    ok = ok and all(np.allclose(m, 0) for m in rho3)
    ok = ok and all(np.allclose(r_ @ rhoY, rhoY @ r_) for r_ in rhoL)
    # T^3_L + Y on a doublet: eigenvalues +-i (charge 1) on 8 real directions and 0 on 8
    for qname, Qx in (("em", Bem), ("A", QA)):
        m = tangent_matrix(Qx, Td)
        evq = np.linalg.eigvals(m)
        nch = int(np.sum(np.abs(np.abs(evq.imag) - 1) < 1e-8))
        nneu = int(np.sum(np.abs(evq) < 1e-8))
        summary[(name, qname)] = (nch, nneu)
        ok = ok and nch == 8 and nneu == 8 and np.allclose(m @ m @ m, -m)
    mem = tangent_matrix(Bem, Td)
    ok = ok and np.allclose(mem, rhoL[2] + rhoY)
report(25, ok, f"n=3: on the tangent space at Pi_A and at Pi_W, su(3)_C acts trivially, su(2)_L with Casimir 3/4 and Y with Y^2=-1/4 commuting with it: four complex doublets (1,2)_{{1/2}}; ad B_em = ad(t^3_L+Y); under u(1)_em and under u(1)_A=(1/2)q_A: 8 real directions of charge +-1 (four charged complex scalars) and 8 neutral real directions ({summary})")

ok = True
for w_ in Widx:
    ok = ok and close(cm(QA, TA[(2, w_)]), -TA[(3, w_)]) and close(cm(QA, TA[(3, w_)]), TA[(2, w_)])
    ok = ok and iszero(cm(QA, TA[(8, w_)])) and iszero(cm(QA, TA[(9, w_)]))
for a_ in Aidx:
    ok = ok and close(cm(Bem, TA[(a_, 7)]), TA[(a_, 13)]) and close(cm(Bem, TA[(a_, 13)]), -TA[(a_, 7)])
    ok = ok and iszero(cm(Bem, TA[(a_, 14)])) and iszero(cm(Bem, TA[(a_, 15)]))
report(26, ok, "n=3: [Q_A,T_{2w}]=-T_{3w}, [Q_A,T_{3w}]=T_{2w}, [Q_A,T_{8w}]=[Q_A,T_{9w}]=0; [B_em,T_{a7}]=T_{a13}, [B_em,T_{a13}]=-T_{a7}, [B_em,T_{a14}]=[B_em,T_{a15}]=0")

# ---- relation to P_* and the construction's charge
PL = Pst @ Comb([(0.5, Isp), (-0.5, Gam)])
PR = Pst @ Comb([(0.5, Isp), (0.5, Gam)])
ok = commutes(volA, Xinv) and commutes(volA, Zinv) and commutes(volA, Uinv) and commutes(chiW, Xinv) and commutes(chiW, Zinv) and commutes(chiW, Uinv)
ok = ok and close(volA @ Pst, -(Gam @ Pst)) and close(chiW @ Pst, Gam @ Pst)
ok = ok and close(qA @ Pst, J @ Pst) and close(QA @ Pst, Comb([(0.5, J)]) @ Pst)
ok = ok and close(PiA @ Pst, PL) and close(PiW @ Pst, PR) and abs(PL.trace() - 32) < TOL
ok = ok and all(anticommutes(E[i] @ E[j] @ volA, Xinv) for i, j in mix) and all(iszero(Pst @ TA[m] @ Pst) for m in mix)
report(27, ok, "n=3: vol_A and chi_W commute with X, Z, U, vol_A P_*=-Gamma P_*, chi_W P_*=Gamma P_*, so Pi_A P_*=P_L and Pi_W P_*=P_R (rank 32); q_A P_*=J P_*, so Q_A=q_A/2 acts on the block as J/2; every tangent T_aw at Pi_A anticommutes with X=Gamma chi_W and P_* T_aw P_*=0")

# mixed anomaly with su(2)_L on the block: Q_A versus Q_em
That = Comb.of(-J) @ tL[2]
QhatA = Comb.of(-J) @ QA
Qhatem = Comb.of(-J) @ Bem
aL_A = (PL @ QhatA @ That @ That).trace()
aR_A = (PR @ QhatA @ That @ That).trace()
aL_em = (PL @ Qhatem @ That @ That).trace()
ThatR = Comb.of(-J) @ tR[2]
aR_AR = (PR @ QhatA @ ThatR @ ThatR).trace()
aR_emR = (PR @ Qhatem @ ThatR @ ThatR).trace()
grav_A = (PL @ QhatA).trace() - (PR @ QhatA).trace()
ok = abs(aL_A - 4) < TOL and abs(aR_A) < TOL and abs(aL_em) < TOL and abs(aR_AR - 4) < TOL and abs(grav_A) < TOL
report(28, ok, f"n=3 restated (Prop 8.5 with a=1/2, b=0): on the block, Tr(P_L Qhat_A That_L^2)={aL_A:.0f}, Tr(P_R Qhat_A That_L^2)={aR_A:.0f}, Tr(P_R Qhat_A That_R^2)={aR_AR:.0f}, so the mixed SU(2)_L^2 U(1)_A and SU(2)_R^2 U(1)_A anomalies of Qhat_A=-J q_A/2 do not cancel, while Tr(P_L Qhat_em That_L^2)={aL_em:.0f} (gravitational U(1)_A: {grav_A:.0f})")

# =============================================================================================== kinetic term, current, transport
rho_p = {}


def rho_plus(X):
    return vec_rep(X, E, Vp)


def rand_frame(span_idx, k, fixed_idx=()):
    """oriented orthonormal 4-frame in V_+: fixed generators followed by k random orthonormal vectors in span(span_idx)."""
    F = np.zeros((4, 8))
    r = 0
    for i in fixed_idx:
        F[r, Vp.index(i)] = 1.0
        r += 1
    M = RNG.normal(size=(len(span_idx), k))
    Qm, _ = np.linalg.qr(M)
    for c_ in range(k):
        for s, i in enumerate(span_idx):
            F[r, Vp.index(i)] = Qm[s, c_]
        r += 1
    return F


def point(F):
    """(Pi_P dense, p 8x8 projector, vol_P dense) for an oriented orthonormal frame F (rows, V_+ coordinates)."""
    vP = frame_vol(F, Vp).dense()
    return 0.5 * (Id + vP), F.T @ F, vP


rhoQA = rho_plus(QA)
rhoBem = rho_plus(Bem)
ok_bil, ok_cur = True, True
for trial in range(3):
    PiP, p8, vP = point(rand_frame(Vp, 4))
    ok_bil = ok_bil and close(PiP @ PiP, PiP, 1e-8) and abs(np.trace(PiP) - 256) < 1e-8
    Xs = []
    for _ in range(2):
        Y8 = RNG.normal(size=(8, 8))
        Y8 = Y8 - Y8.T
        Y6 = RNG.normal(size=(6, 6))
        Y6 = Y6 - Y6.T
        Xs.append((lift8(Y8) + lift6(Y6), Y8))
    dP = [dcm(X, PiP) for X, _ in Xs]
    dp = [Yv @ p8 - p8 @ Yv for _, Yv in Xs]
    for i in range(2):
        for j in range(2):
            ok_bil = ok_bil and abs(np.sum(dP[i] * dP[j].T) - 64 * np.trace(dp[i] @ dp[j])) < 1e-7
        ok_bil = ok_bil and close(dP[i], dP[i].T, 1e-9)
        ok_bil = ok_bil and abs(np.sum(dP[i] ** 2) - 64 * np.sum(dp[i] ** 2)) < 1e-7
    for Qx, qv in ((QA, rhoQA), (Bem, rhoBem)):
        lhs = -np.sum(dcm(Qx, PiP) * dP[0].T)
        rhs = -64 * np.trace((qv @ p8 - p8 @ qv) @ dp[0])
        ok_cur = ok_cur and abs(lhs - rhs) < 1e-7
report(29, ok_bil, "n=3 (vector model): at random points Pi_P of F_3 and random X,Y in k (with so(6) parts), Tr([X,Pi_P][Y,Pi_P]) = 64 tr_8([x,p][y,p]) and ||[X,Pi_P]||^2 = 64||[x,p]||^2, x the so(8) part of the vector representation and p the 8x8 projector onto P")
report(30, ok_cur, "n=3: at random points of F_3, -Tr([Q,Pi]dPi) = -64 tr_8([q,p]dp) for Q=q_A/2 and Q=B_em (q their images in so(8)); hence L=(f^2/2)Tr(DPi DPi)=32 f^2 tr_8(Dp Dp) and j=-64 f^2 tr_8([q,p]Dp) exactly on F_3")

# quadratic current at Pi_A
ok = True
for _ in range(3):
    x = RNG.normal(size=16)
    y = RNG.normal(size=16)
    X = Comb([(x[k_] * 0.5, E[i] @ E[j]) for k_, (i, j) in enumerate(mix)])
    Yc = Comb([(y[k_] * 0.5, E[i] @ E[j]) for k_, (i, j) in enumerate(mix)])
    xs = {m: x[k_] for k_, m in enumerate(mix)}
    ys = {m: y[k_] for k_, m in enumerate(mix)}
    dY = cm(Yc, PiA)
    jA = -(cm(cm(QA, X), PiA) @ dY).trace()
    jem = -(cm(cm(Bem, X), PiA) @ dY).trace()
    ok = ok and abs((cm(X, PiA) @ dY).trace() - 128 * x @ y) < 1e-8
    ok = ok and abs(jA - 128 * sum(xs[(2, w_)] * ys[(3, w_)] - xs[(3, w_)] * ys[(2, w_)] for w_ in Widx)) < 1e-8
    ok = ok and abs(jem - 128 * sum(xs[(a_, 13)] * ys[(a_, 7)] - xs[(a_, 7)] * ys[(a_, 13)] for a_ in Aidx)) < 1e-8
report(31, ok, "n=3: at Pi_A, for X=sum x_aw aw/2 and Y=sum y_aw aw/2: Tr([X,Pi_A][Y,Pi_A])=128 x.y, -Tr([[Q_A,X],Pi_A][Y,Pi_A]) = 128 sum_w (x_2w y_3w - x_3w y_2w), -Tr([[B_em,X],Pi_A][Y,Pi_A]) = 128 sum_a (x_a13 y_a7 - x_a7 y_a13): L=64 f^2 Dx.Dx and j=128 f^2 sum Im(phibar D phi) with four charged complex scalars for each charge")

# exact kinetic term and current on the two Clifford spheres; adapted connection on them
ok = True
for Zs, gens, Qx, (ia, ib) in ((ZA, Lgen_idx, QA, (0, 1)), (ZO, Ogen_idx, Bem, (1, 2))):
    for _ in range(2):
        n = RNG.normal(size=5)
        n /= np.linalg.norm(n)
        dn = RNG.normal(size=5)
        dn -= (dn @ n) * n
        P = Comb([(0.5, Isp)]) + Comb([(0.5 * n[i], Zs[i]) for i in range(5)])
        dP = Comb([(0.5 * dn[i], Zs[i]) for i in range(5)])
        ok = ok and abs((dP @ dP).trace() - 128 * dn @ dn) < 1e-8
        ok = ok and abs(-(cm(Qx, P) @ dP).trace() - (128 * (n[ia] * dn[ib] - n[ib] * dn[ia]) if Zs is ZA else 128 * (n[ib] * dn[ia] - n[ia] * dn[ib]))) < 1e-8
        if Zs is ZA:
            nv = Comb([(n[i], E[gens[i]]) for i in range(5)])
            dnv = Comb([(dn[i], E[gens[i]]) for i in range(5)])
            ok = ok and close(cm(P, dP), cm(nv, dnv) * 0.25)
report(32, ok, "n=3: on the inner sphere with Q=q_A/2 and on the outer sphere with Q=B_em: Tr(dPi^2)=128|dn|^2 and -Tr([Q,Pi]dPi)=+-128(n_a dn_b-n_b dn_a) (one charged complex scalar, two neutral real scalars), and on the inner sphere [Pi,dPi]=(1/4)[n,dn]: the n=2 structure with 16 replaced by 128")

# adapted connection across F_3 and K-invariance of the full alignment energy
ok_C, ok_inv = True, True
kpos_blades = [E[i] @ E[j] for i, j in kpos_pairs]
for trial in range(3):
    PiP, p8, vP = point(rand_frame(Vp, 4))
    Y8 = RNG.normal(size=(8, 8))
    Y8 = Y8 - Y8.T
    Y6 = RNG.normal(size=(6, 6))
    Y6 = Y6 - Y6.T
    X = lift8(Y8) + lift6(Y6) + Comb([(0.7, qA)]) + Bem * 0.3
    dPi = dcm(X, PiP)
    Cm = PiP @ dPi - dPi @ PiP
    Xd = X.dense()
    Xm = 0.5 * (Xd - vP @ Xd @ vP)
    ok_C = ok_C and close(Cm, -Xm, 1e-8)
    coef = [Bb.T.lmul(Cm).trace() / N for Bb in kpos_blades]
    rec = Cm - Comb([(c_, Bb) for c_, Bb in zip(coef, kpos_blades)]).dense()
    ok_C = ok_C and float(np.max(np.abs(rec))) < 1e-8
    VK = sum(float(np.sum(dcm(E[i] @ E[j], PiP) ** 2)) for i, j in kbiv) / 512.0
    ok_inv = ok_inv and abs(VK - 16) < 1e-7
report(33, ok_C, "n=3: at random points of F_3 and a random compact connection value X (so(8)+so(6)+q_A+B_em), the adapted connection [Pi,[X,Pi]] equals -X_m=-(X-vol_P X vol_P)/2 and lies in the span of the 28 compact bivectors of V_+")
report(34, ok_inv, "n=3: the full alignment energy (1/512) sum over the 43 compact blades of ||[e_ie_j,Pi]||^2 equals 16 at random points of F_3 (constant on the K-orbit), and is 16 = (1/512)*16*512 at Pi_A")

lorb = [E[i] @ E[j] for i, j in itertools.combinations(interior, 2)]
ok = all(commutes(x, volA) and commutes(x, volW) for x in lorb) and all(iszero(cm(x, y)) for x in lorb for y in [QA, Bem] + hPS + hSM)
stabk = [(i, j) for i, j in kbiv if commutes(E[i] @ E[j], volA)]
ok = ok and len(stabk) == 27 and set(stabk) == set(itertools.combinations(Aidx, 2)) | set(itertools.combinations(Widx, 2)) | set(kneg_pairs)
report(35, ok, "n=3: the six interior bivectors commute with Pi_A, Pi_W and with q_A, B_em and every generator of h_PS and h_SM, so Omega=Omega_Lorentz+A^a T_a gives D Pi_A=0 and [R,Pi_A]=0 exactly for T_a in the stabiliser so(4)_A+so(4)_W+so(6)_C (27 compact blades), which contains u(1)_A, u(1)_em, h_PS and h_SM")

# =============================================================================================== fixed sets
t_fix = time.time()


def tangent_basis_at(PiP):
    tans = [dcm(Bb, PiP) * 0.5 for Bb in kpos_blades]
    return orth_basis(tans)


def kernel_on_tangent(gens, PiP, basis=None):
    basis = tangent_basis_at(PiP) if basis is None else basis
    r = len(basis)
    if not gens:
        return r, r
    rows = []
    for Xg in gens:
        imgs = [dcm(Xg, bb) for bb in basis]
        rows.append(np.array([[np.sum(bb * im) for im in imgs] for bb in basis]))
    return r, null_space(np.vstack(rows)).shape[1]


def stab_dim(gens, PiP):
    imgs = [dcm(Xg, PiP) for Xg in gens]
    return len(gens) - span_rank(imgs)


# vector-model data
rho_su2L = [rho_plus(x) for x in tL]
rho_Y = rho_plus(Yb)
rho_soW = [rho_plus(x) for x in soW]
rho_k8 = [rho_plus(Comb([(0.5, Bb)])) for Bb in kpos_blades]


def vm_tangent_kernel(gens8, p8):
    """dimension of the joint kernel of ad(gens) on T_P = {[Y,p]: Y in so(8)} (vector model)."""
    tans = []
    for Yv in rho_k8:
        tans.append(Yv @ p8 - p8 @ Yv)
    basis = orth_basis(tans)
    if not gens8:
        return len(basis), len(basis)
    rows = []
    for g8 in gens8:
        rows.append(np.array([[np.sum(bb * (g8 @ ba - ba @ g8)) for ba in basis] for bb in basis]))
    return len(basis), null_space(np.vstack(rows)).shape[1]


def vm_fixed(gens8, p8):
    return all(np.allclose(g8 @ p8, p8 @ g8, atol=1e-10) for g8 in gens8)


def vm_stab(gens8, p8):
    imgs = [g8 @ p8 - p8 @ g8 for g8 in gens8]
    return len(gens8) - span_rank(imgs) if gens8 else 0


others_A = [i for i in Vp if i not in (2, 3)]
others_em = [i for i in Vp if i not in (7, 13)]
ok = True
for _ in range(3):
    for Qv, others, pl in ((rhoQA, others_A, (2, 3)), (rhoBem, others_em, (7, 13))):
        F1 = rand_frame(others, 2, fixed_idx=pl)
        F2 = rand_frame(others, 4)
        for F in (F1, F2):
            p8 = F.T @ F
            ok = ok and vm_fixed([Qv], p8) and vm_tangent_kernel([Qv], p8) == (16, 8)
    Fg = rand_frame(Vp, 4)
    ok = ok and not vm_fixed([rhoQA], Fg.T @ Fg) and not vm_fixed([rhoBem], Fg.T @ Fg)
# spot checks on R^512
for Qx, others, pl in ((QA, others_A, (2, 3)), (Bem, others_em, (7, 13))):
    for F in (rand_frame(others, 2, fixed_idx=pl), rand_frame(others, 4)):
        PiP, p8, vP = point(F)
        ok = ok and iszero(dcm(Qx, PiP), 1e-9) and kernel_on_tangent([Qx], PiP) == (16, 8)
ok = ok and kernel_on_tangent([QA], PiA_d) == (16, 8) and kernel_on_tangent([Bem], PiA_d) == (16, 8)
report(36, ok, "n=3: the u(1)_A-fixed and u(1)_em-fixed sets in F_3: planes containing the charge plane, or orthogonal to it, commute with the charge (vector model and R^512), and the kernel of ad Q on their 16-dim tangent space is 8; a generic plane does not commute; Pi_A has kernel 8 for both charges")

# irreducibility of su(2)_L on W, triviality of h_SM on A, so(8) irreducible on V_+
iA = [Vp.index(i) for i in Aidx]
iW = [Vp.index(i) for i in Widx]
ok = all(np.allclose(r_[:, iA], 0) and np.allclose(r_[iA, :], 0) for r_ in rho_su2L + [rho_Y])
blocks = [r_[np.ix_(iW, iW)] for r_ in rho_su2L]
Msu = np.array([np.kron(np.eye(4), R) - np.kron(R.T, np.eye(4)) for R in blocks]).reshape(-1, 16)
comW = null_space(Msu)
cW = [comW[:, k_].reshape(4, 4) for k_ in range(comW.shape[1])]
# the commutant is a quaternion algebra: it contains three anticommuting complex structures
Iq = 2 * rho_Y[np.ix_(iW, iW)]
ok = ok and comW.shape[1] == 4 and np.allclose(Iq @ Iq, -np.eye(4)) and all(np.allclose(Iq @ R, R @ Iq) for R in blocks)
skew = [Cc - Cc.T for Cc in cW]
skew = orth_basis([s_ for s_ in skew if np.max(np.abs(s_)) > 1e-9])
ok = ok and len(skew) == 3 and all(np.allclose((2 * s_) @ (2 * s_), -np.eye(4)) for s_ in skew)
ok = ok and all(np.allclose(skew[i] @ skew[j] + skew[j] @ skew[i], 0) for i in range(3) for j in range(i + 1, 3))
Mk8 = np.array([np.kron(np.eye(8), R) - np.kron(R.T, np.eye(8)) for R in rho_k8]).reshape(-1, 64)
ok = ok and null_space(Mk8).shape[1] == 1
report(37, ok, "n=3 (vector model): h_SM acts on V_+ only through su(2)_L+u(1)_Y on W and trivially on A; the commutant of su(2)_L on W is a quaternion algebra (dim 4, three anticommuting complex structures, one of them 2Y), so su(2)_L is irreducible on W; so(8) has commutant R on V_+")

okh = True
dims = {}
PimA = Id - PiA_d
PimW = Id - PiW_d
for name, gens in (("PS", hPS), ("SM", hSM), ("su2L", tL)):
    for lab, Td in (("A", TA), ("W", TW)):
        rows = [tangent_matrix(X, Td) for X in gens]
        dims[(name, lab)] = null_space(np.vstack(rows)).shape[1]
okh = all(v == 0 for v in dims.values())
gpi = expm(np.pi * 0.5 * E[2].dense() @ E[7].dense())
okh = okh and close(gpi @ PiA_d @ gpi.T, PimA, 1e-9) and close(gpi @ PiW_d @ gpi.T, PimW, 1e-9)
okh = okh and all(iszero(dcm(x, PimA)) and iszero(dcm(x, PimW)) for x in hPS + hSM)
report(38, okh, f"n=3: h_PS, h_SM and su(2)_L have no invariant tangent vector at Pi_A and Pi_W (hence none at Pi_-A=1-Pi_A, Pi_-W: same tangent space up to sign), so these four fixed points are isolated; Pi_-A is reached from Pi_A by exp(pi E2E7/2) (kernel dims {dims})")

# u(1)_Y alone
PY, pY, _ = point(np.eye(8)[[Vp.index(i) for i in (2, 3, 7, 13)]])
ok = iszero(dcm(Yb, PY)) and kernel_on_tangent([Yb], PY) == (16, 6) and vm_tangent_kernel([rho_Y], pY) == (16, 6)
ok = ok and vm_tangent_kernel([rho_Y], np.eye(8)[iA].T @ np.eye(8)[iA]) == (16, 0)
for _ in range(2):  # random point of the component: A' (2-plane in A) + W' (complex line of 2Y in W)
    FA = rand_frame(Aidx, 2)[:2]
    wv = np.zeros(8)
    wv[iW] = RNG.normal(size=4)
    wv /= np.linalg.norm(wv)
    Jw = np.zeros((8, 8))
    Jw[np.ix_(iW, iW)] = 2 * rho_Y[np.ix_(iW, iW)]
    F = np.vstack([FA, wv, Jw @ wv])
    pP = F.T @ F
    ok = ok and np.allclose(F @ F.T, np.eye(4)) and vm_fixed([rho_Y], pP) and vm_tangent_kernel([rho_Y], pP) == (16, 6)
    ok = ok and vm_stab(rho_su2L + [rho_Y], pP) == 2
ok = ok and vm_stab(rho_su2L + [rho_Y], pY) == 2 and stab_dim(hSM, PY) == 10
report(39, ok, "n=3: the u(1)_Y-fixed set contains, besides Pi_+-A and Pi_+-W (isolated), the 6-dim component {A'+W'}: A' a 2-plane in A, W' a complex line of 2Y in W; through span(E2,E3,E7,E13) the kernel of ad Y is 6 (vector model and R^512), and on this component the stabiliser in h_SM is su(3)+u(1)+u(1) (dim 10)")

# generic stabilisers: em components carry the electroweak pattern
Wbasis = np.eye(8)[iW]


def int_dim(F1, F2):
    """dimension of the intersection of the row spans of two orthonormal frames (principal angles zero)."""
    sv = np.linalg.svd(F1 @ F2.T, compute_uv=False)
    return int(np.sum(sv > 1 - 1e-9))


ok = True
res = {}
for lab, F in (("em_perp", rand_frame(others_em, 4)), ("em_cont", rand_frame(others_em, 2, fixed_idx=(7, 13))),
               ("A_cont", rand_frame(others_A, 2, fixed_idx=(2, 3))), ("A_perp", rand_frame(others_A, 4)), ("generic", rand_frame(Vp, 4))):
    p8 = F.T @ F
    sSM = 8 + vm_stab(rho_su2L + [rho_Y], p8)
    sPS = 15 + vm_stab(rho_soW, p8)
    Fperp = null_space(F).T
    d1, d2 = int_dim(F, Wbasis), int_dim(Fperp, Wbasis)
    ok = ok and sPS == 15 + d1 * (d1 - 1) // 2 + d2 * (d2 - 1) // 2
    res[lab] = (sSM, sPS, d1, d2)
PiP, p8, _ = point(rand_frame(others_em, 4))
ok = ok and {k_: v_[:2] for k_, v_ in res.items()} == {"em_perp": (9, 16), "em_cont": (9, 16), "A_cont": (8, 16), "A_perp": (8, 16), "generic": (8, 15)}
ok = ok and stab_dim(hSM, PiP) == 9 and iszero(dcm(Bem, PiP), 1e-9) and stab_dim(hPS, PiP) == 16
ok = ok and stab_dim(hSM, PiA_d) == 12 and stab_dim(hPS, PiA_d) == 21
report(40, ok, f"n=3: the stabiliser of Pi_P in h_PS is so(6)_C+so(P cap W)+so(P^perp cap W); dimensions in (h_SM, h_PS) at random points: u(1)_em-fixed components (9,16) = su(3)+u(1)_em and so(6)+u(1)_em; u(1)_A-fixed components (8,16) (su(3) only, so(6)+so(2) of a 2-plane of W); generic points (8,15); Pi_A (12,21) (computed (SM,PS,dim P cap W,dim P^perp cap W) = {res}; R^512 spot check on the em component)")

# h-invariant symmetric bilinear forms on the tangent space (Hessians at isolated fixed points)
def inv_sym_forms(gens, Td):
    rho = [tangent_matrix(X, Td) for X in gens]
    idx = [(i, j) for i in range(16) for j in range(i, 16)]
    cols = []
    for (i, j) in idx:
        Sb = np.zeros((16, 16))
        Sb[i, j] = Sb[j, i] = 1.0
        cols.append(np.concatenate([(Sb @ R - R @ Sb).ravel() for R in rho]))
    return null_space(np.array(cols).T).shape[1]


nPS = inv_sym_forms(hPS, TA)
nSM = inv_sym_forms(hSM, TA)
nPSW = inv_sym_forms(hPS, TW)
nSMW = inv_sym_forms(hSM, TW)
report(41, nPS == 10 and nSM == 16 and nPSW == 10 and nSMW == 16,
       f"n=3: the h-invariant symmetric bilinear forms on the tangent space at Pi_A (and at Pi_W) form a space of dimension 10 for h_PS (S(x)delta_W, S symmetric on A) and 16 for h_SM (adding N(x)omega_Y, N antisymmetric on A) (computed {nPS},{nSM},{nPSW},{nSMW})")

# spectrum of the three involutive types, from exact traces
ok = True
for t_, S in reps.items():
    V = blade(E, S)
    ok = ok and V @ V == Isp and V.T == V and V.trace() == 0
report(42, ok, "n=3: representatives of the three involutive types (4,0),(2,2),(0,4) are symmetric involutions of trace 0, hence all have spectrum {+1^256,-1^256}: a spectrum-only potential is constant on F_3 and does not separate the types")

# =============================================================================================== n = 4
t_n4 = time.time()
E4, eta4 = tower(4)
Nn4 = 8 ** 4
ident4 = ident(Nn4)
ok = all((E4[i] @ E4[i]) == (ident4 if eta4[i] > 0 else -ident4) for i in range(1, 25))
ok = ok and all(anticommutes(E4[i], E4[j]) for i in range(1, 25) for j in range(i + 1, 25))
int4 = [i for i in range(1, 25) if i not in interior]
Vp4 = [i for i in int4 if eta4[i] > 0]
Vm4 = [i for i in int4 if eta4[i] < 0]
H4 = blade(E4, [2, 3, 8, 9])
st_all4 = st_k4 = 0
for i, j in itertools.combinations(int4, 2):
    Bij = E4[i] @ E4[j]
    if commutes(Bij, H4):
        st_all4 += 1
        st_k4 += eta4[i] == eta4[j]
    else:
        ok = ok and anticommutes(Bij, H4)
k4 = len(Vp4) * (len(Vp4) - 1) // 2 + len(Vm4) * (len(Vm4) - 1) // 2
ok = ok and len(Vp4) == 11 and len(Vm4) == 9 and len(list(itertools.combinations(int4, 2))) == 190
ok = ok and st_all4 == 126 and st_k4 == 63 and k4 == 91 and k4 - st_k4 == 28
ok = ok and all(commutes(g_, H4) for g_ in (E4[1], E4[4], E4[5], E4[6]))
report(43, ok, f"n=4 restated (Prop 26.9 table): the 24 generators (signed permutations of size 4096) satisfy the Clifford relations; internal signature (11,9), so(11,9) dim 190; stabiliser of E2E3E8E9 dim {st_all4}, compact part {st_k4}, K_4 dim {k4}, orbit dimension {k4 - st_k4} = dim of the oriented Grassmannian of 4-planes in R^11")

# fixed-set dimensions at n=4 by blade counting at Pi_A
A4 = [2, 3, 8, 9]
perp4 = [i for i in Vp4 if i not in A4]
T4 = {(a_, b_): E4[a_] @ E4[b_] @ H4 for a_ in A4 for b_ in perp4}


def kernel_count(Qblades):
    """joint kernel of ad(Q) on the coordinate tangent basis: a blade either commutes with a basis element or maps it to
    a distinct blade, so the kernel is spanned by the basis elements commuting with every Q."""
    return sum(1 for t in T4.values() if all(commutes(Qb, t) for Qb in Qblades))


W4 = [7, 13, 14, 15]
d_in = kernel_count([E4[2] @ E4[3]])
d_perp = kernel_count([E4[19] @ E4[20]])
d_W = kernel_count([E4[i] @ E4[j] for i, j in itertools.combinations(W4, 2)])
d_col = all(commutes(E4[i] @ E4[j], H4) for i, j in itertools.combinations(Vm4, 2))
ok = len(T4) == 28 and d_in == 14 and d_perp == 20 and d_W == 12 and d_col
report(44, ok, f"n=4: at Pi_A the tangent space is A(x)A^perp (28); a u(1) rotating a plane inside A fixes a component of dim {d_in}=dim G~r(2,9), one rotating a plane orthogonal to A a component of dim {d_perp}=dim G~r(4,9); so(4)_W (W={{7,13,14,15}}) fixes a component of dim {d_W}=dim G~r(4,7) through Pi_A; all 36 bivectors of V_- (dim 9) commute with vol_A")


def grdim(k, m):
    return k * (m - k)


tab = {nn: (3 * nn - 1, grdim(4, 3 * nn - 1), 2 * (3 * nn - 1 - 4), max(0, 4 * (3 * nn - 1 - 6))) for nn in (2, 3, 4)}
ok = tab == {2: (5, 4, 2, 0), 3: (8, 16, 8, 8), 4: (11, 28, 14, 20)} and (43 - 27) == 16 and (13 - 9) == 4 and (91 - 63) == 28
report(45, ok, "dimension table: F_n = oriented Grassmannian of 4-planes in V_+ (dim 3n-1) has dim 4, 16, 28 at n=2,3,4, equal to the K_n-orbit dimensions 13-9, 43-27, 91-63; a u(1) on one positive plane fixes components of dims (2), (8,8), (14,20)")

# =============================================================================================== duality and the embedded second-rung family
sgnAW = 1 if (volPlus @ volA) == volW else -1
ok = (volPlus @ volA) == (volW if sgnAW > 0 else -volW)
for _ in range(2):
    PiP, p8, vP = point(rand_frame(Vp, 4))
    PiD = 0.5 * (Id + volPlus.lmul(vP))
    cA = [dcm(E[i] @ E[j], PiP).ravel() for i, j in kbiv]
    cD = [dcm(E[i] @ E[j], PiD).ravel() for i, j in kbiv]
    MA = np.array(cA) @ np.array(cA).T
    MD = np.array(cD) @ np.array(cD).T
    ok = ok and np.allclose(MA, MD, atol=1e-7) and np.linalg.matrix_rank(MA, tol=1e-6) == 16
PiAd = Comb([(0.5, Isp), (0.5, volPlus @ volA)])
ok = ok and close(PiAd, PiW if sgnAW > 0 else Comb([(0.5, Isp), (-0.5, volW)]))
ok = ok and abs((PiA @ PL).trace() - 32) < TOL and abs((PiW @ PL).trace()) < TOL and abs((PiW @ PR).trace() - 32) < TOL
report(46, ok, f"n=3: the duality D(Pi)=(1+vol_+(2Pi-1))/2, P->P^perp, maps Pi_A to Pi_{{{'+' if sgnAW > 0 else '-'}W}} (vol_+ vol_A = {'+' if sgnAW > 0 else '-'}vol_W) and at random P preserves the full 43x43 mass matrix Tr([e_ie_j,Pi][e_ke_l,Pi]) (rank 16): every alignment energy V_h, h in k, takes equal values at P and P^perp; on the block Tr(Pi_A P_L)=32, Tr(Pi_W P_L)=0, Tr(Pi_W P_R)=32, so a coupling to the chiral block separates what alignment cannot")

# the second-rung family embedded with its own generators a,b,c,d,e = E2,E3,E7,E8,E9
I5e = blade(E, [2, 3, 7, 8, 9])
Ze = [I5e @ E[i] for i in (2, 3, 7, 8, 9)]


def kernel_of_n(gens):
    vecs = []
    for i in range(5):
        vecs.append(np.concatenate([cm(Xg, Ze[i]).dense().ravel() for Xg in gens]))
    G = np.array(vecs) @ np.array(vecs).T
    w_, V_ = np.linalg.eigh(G)
    return V_[:, w_ < 1e-8 * max(1.0, float(np.max(w_)))]


kem = kernel_of_n([Bem])
kPS = kernel_of_n(hPS)
kSM = kernel_of_n(hSM)
kQ = kernel_of_n([QA])
ec = np.zeros(5)
ec[2] = 1.0
ok = Ze[2] == volA and Ze[4] == Hlit
ok = ok and kem.shape[1] == 1 and abs(abs(kem[:, 0] @ ec) - 1) < 1e-9
ok = ok and kPS.shape[1] == 1 and abs(abs(kPS[:, 0] @ ec) - 1) < 1e-9 and kSM.shape[1] == 1 and abs(abs(kSM[:, 0] @ ec) - 1) < 1e-9
ok = ok and kQ.shape[1] == 3 and np.allclose(kQ[:2, :], 0)
report(47, ok, "n=3: the second-rung family embedded with its own generators (a,b,c,d,e)=(E2,E3,E7,E8,E9) is the inner Clifford sphere {(1+I_5 n)/2}; its point n=c is Pi_A (I_5 c = vol_A) and its point n=e is the literal H=E2E3E7E8; on it Q_A=ab/2 is unbroken on the two-sphere n_a=n_b=0, while u(1)_em, h_SM and h_PS are each unbroken only at n=+-c, i.e. at Pi_{+-A}")

npass = sum(RESULTS)
print(f"{npass}/{len(RESULTS)} checks passed in {time.time() - T_START:.1f}s (n=3 from {t_n3 - T_START:.1f}s, fixed sets from {t_fix - T_START:.1f}s, n=4 from {t_n4 - T_START:.1f}s)")
sys.exit(0 if all(RESULTS) else 1)
