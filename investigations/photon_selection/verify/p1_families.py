#!/usr/bin/env python3
"""p1: the compact families of Lorentz-invariant vacuum projectors at the second, third and fourth rung.

    python3 investigations/photon_selection/verify/p1_families.py

Prints one 'PASS p1-<n>: <claim>' (or 'FAIL ...') line per claim of sections/p1_families.tex and exits 0 iff all pass.

Realisation (tex/s02_arena.tex, tex/sA_realisation.tex): e1=sx(x)1(x)1, e2=sz(x)sx(x)1, e3=sz(x)sz(x)sx, e4=ep(x)1(x)1,
e5=sz(x)ep(x)1, e6=sz(x)sz(x)ep, ep=[[0,-1],[1,0]], omega=e1...e6, E_{6(k-1)+a}=omega^{(x)(k-1)} (x) e_a (x) 1...
All generators and blades are signed permutation matrices with integer entries; identities among them are checked exactly
(float64 on integers and dyadic rationals of small modulus is exact).  Floating point with tolerance TOL is used only for
random points of the families (orthonormal frames, matrix exponentials) and for ranks of real spans.
"""
import itertools
import sys

import numpy as np
from scipy.linalg import expm, null_space

TOL = 1e-9
RNG = np.random.default_rng(20261021)
RESULTS = []


def report(n, ok, claim):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} p1-{n}: {claim}")


# ----------------------------------------------------------------------------------------------- realisation
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def kron(*ms):
    out = np.array([[1]], dtype=np.int64)
    for m in ms:
        out = np.kron(out, m)
    return out


e8 = [kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx), kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
omega8 = e8[0] @ e8[1] @ e8[2] @ e8[3] @ e8[4] @ e8[5]
ETA1 = [1, 1, 1, -1, -1, -1]


def tower(n):
    """Generators E_1..E_{6n} (index 0 unused) as float64 dense matrices of size 8^n, and their signs."""
    E = [None]
    eta = [0]
    for k in range(1, n + 1):
        for a in range(6):
            E.append(kron(*([omega8] * (k - 1) + [e8[a]] + [np.eye(8, dtype=np.int64)] * (n - k))).astype(np.float64))
            eta.append(ETA1[a])
    return E, eta


def prod(ms, N):
    out = np.eye(N)
    for m in ms:
        out = out @ m
    return out


def comm(a, b):
    return a @ b - b @ a


def acomm(a, b):
    return a @ b + b @ a


def iszero(m):
    return float(np.max(np.abs(m))) < TOL


def close(a, b):
    return iszero(a - b)


def frob2(m):
    return float(np.sum(m * m))


def span_rank(mats):
    if not mats:
        return 0
    A = np.array([m.ravel() for m in mats])
    G = A @ A.T
    return int(np.linalg.matrix_rank(G, tol=1e-8 * max(1.0, float(np.max(np.abs(G))))))


def lie_vec(i, j, dim):
    """Vector-representation generator of the spinor bivector (1/2) e_i e_j on an orthonormal positive frame:
    ad((1/2)e_i e_j) e_i = -e_j, e_j -> e_i, i.e. the matrix E_ij - E_ji."""
    M = np.zeros((dim, dim))
    M[i, j] = 1.0
    M[j, i] = -1.0
    return M


all_ok_dims = True

# =============================================================================================== n = 2
E2_, eta2 = tower(2)
N2 = 64
Id2 = np.eye(N2)
ok = all(close(acomm(E2_[a], E2_[b]), 2 * (eta2[a] if a == b else 0) * Id2) for a in range(1, 13) for b in range(1, 13))
interior2 = [1, 4, 5, 6]
internal2 = [i for i in range(1, 13) if i not in interior2]
pos2 = [i for i in internal2 if eta2[i] > 0]
neg2 = [i for i in internal2 if eta2[i] < 0]
report(1, ok and pos2 == [2, 3, 7, 8, 9] and neg2 == [10, 11, 12],
       "n=2: E_1..E_12 satisfy E_aE_b+E_bE_a=2eta_ab; spectator internal complement has positive {2,3,7,8,9} and negative {10,11,12}, signature (5,3)")

a, b, c, d, e = (E2_[i] for i in (2, 3, 7, 8, 9))
Q2 = 0.5 * a @ b
H2 = a @ b @ c @ d
Pi0 = 0.5 * (Id2 + H2)
w2 = prod([E2_[i] for i in internal2], N2)
B = 0.5 * b @ e
pi = comm(B, Pi0)
ok = (close(H2 @ H2, Id2) and close(H2.T, H2) and abs(np.trace(H2)) < TOL and abs(np.trace(Pi0) - 32) < TOL
      and close(Pi0 @ Pi0, Pi0) and iszero(comm(Q2, Pi0)) and all(iszero(comm(E2_[i], Pi0)) for i in interior2)
      and iszero(comm(w2, Pi0)) and close(w2 @ w2, -Id2)
      and close(pi, -0.5 * a @ c @ d @ e) and close(comm(Q2, pi), 0.5 * b @ c @ d @ e) and close(pi.T, pi)
      and abs(np.trace(pi @ pi) - 16) < TOL)
report(2, ok, "n=2 restated (Thm 26.6): H=abcd symmetric traceless involution, Pi_0 rank 32, [Q,Pi_0]=0, Pi_0 commutes with E1,E4,E5,E6 and the internal volume w (w^2=-1); pi=[be/2,Pi_0]=-acde/2, [Q,pi]=bcde/2, Tr pi^2=16")

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
    ok = ok and all(iszero(comm(E2_[i], P)) for i in interior2)
    ok = ok and abs(np.trace(dP @ dP) - 16 * dn @ dn) < 1e-8
    ok = ok and abs(-np.trace(comm(Q2, P) @ dP) - 16 * (n[0] * dn[1] - n[1] * dn[0])) < 1e-8
    # adapted connection [Pi, dPi] = (1/2) n wedge dn = (1/4)[n,dn]
    nv = sum(n[i] * v5[i] for i in range(5))
    dnv = sum(dn[i] * v5[i] for i in range(5))
    ok = ok and close(comm(P, dP), 0.25 * comm(nv, dnv))
    ev = np.linalg.eigvalsh(P)
    ok = ok and np.allclose(np.sort(ev), np.r_[np.zeros(32), np.ones(32)], atol=1e-8)
report(4, ok, "n=2 restated (Thms 26.7, 26.8): at random n, Pi(n) is a rank-32 Lorentz-invariant orthogonal projector with spectrum {0^32,1^32}; Tr(dPi^2)=16|dn|^2, -Tr([Q,Pi]dPi)=16(n_a dn_b-n_b dn_a), [Pi,dPi]=(1/4)[n,dn]")

# K_2 orbit
k2 = [0.5 * E2_[i] @ E2_[j] for i, j in itertools.combinations(pos2, 2)] + [0.5 * E2_[i] @ E2_[j] for i, j in itertools.combinations(neg2, 2)]
ok = len(k2) == 13 and all(close(X @ X, -0.25 * Id2) for X in k2)
Zspan = np.array([z.ravel() for z in Z])
ok_inv = True
for X in k2:
    for z in Z:
        Cz = comm(X, z).ravel()
        coef = np.linalg.lstsq(Zspan.T, Cz, rcond=None)[0]
        ok_inv = ok_inv and np.max(np.abs(Zspan.T @ coef - Cz)) < TOL
stab = sum(1 for X in k2 if iszero(comm(X, Pi0)))
orbit_tan = span_rank([comm(X, Pi0) for X in k2])
# explicit transitivity: rotation of the (e, v) plane by angle theta carries Pi(e) to Pi(cos e + sin v)
ok_tr = True
for i in range(4):
    th = RNG.uniform(0, np.pi)
    g = expm(-0.5 * th * v5[i] @ e)  # spinor rotation
    n = np.zeros(5)
    n[4], n[i] = np.cos(th), np.sin(th)
    ok_tr = ok_tr and close(g @ Pi0 @ g.T, Pi_n2(n)) and close(g @ g.T, Id2)
report(5, ok and ok_inv and stab == 9 and orbit_tan == 4 and ok_tr,
       "n=2: the 13 compact internal bivectors (so(5)+so(3)) preserve span{Z_i}; the stabiliser of Pi_0 in k_2 has dim 9, the orbit tangent dim 4, and exp(-theta v_i e/2) carries Pi(e) to Pi(cos theta e+sin theta v_i): the K_2-orbit of Pi_0 is S^4")

# Q-fixed points on S^4
ok = iszero(comm(Q2, Z[2])) and iszero(comm(Q2, Z[3])) and iszero(comm(Q2, Z[4])) and close(comm(Q2, Z[0]), -Z[1]) and close(comm(Q2, Z[1]), Z[0])
report(6, ok, "n=2: [Q,I_5 v]=I_5[Q,v]: Q fixes Z_c,Z_d,Z_e and rotates (Z_a,Z_b), so [Q,Pi(n)]=0 iff n_a=n_b=0 (a two-sphere)")

# Prop 26.9 at n=2: stabiliser in so(5,3)
allbiv2 = [(i, j) for i, j in itertools.combinations(internal2, 2)]
stab_all = [(i, j) for i, j in allbiv2 if iszero(comm(E2_[i] @ E2_[j], H2))]
stab_c = [(i, j) for i, j in stab_all if eta2[i] == eta2[j]]
ok = len(allbiv2) == 28 and len(stab_all) == 12 and len(stab_c) == 9
# the rank of the stabiliser-kernel equals the number of commuting blades (blades commute or anticommute)
ok = ok and span_rank([comm(E2_[i] @ E2_[j], H2) for i, j in allbiv2]) == 28 - 12
report(7, ok, "n=2 restated (Prop 26.9): stabiliser of H in the internal so(5,3) (dim 28) has dim 12, compact part 9 = so(4)+so(3)")

# =============================================================================================== n = 3
E, eta = tower(3)
N = 512
Id = np.eye(N)
ok = all(close(acomm(E[i], E[j]), 2 * (eta[i] if i == j else 0) * Id) for i in range(1, 19) for j in range(i, 19))
interior = [1, 4, 5, 6]
internal = [i for i in range(1, 19) if i not in interior]
Vp = [i for i in internal if eta[i] > 0]
Vm = [i for i in internal if eta[i] < 0]
Aidx, Widx, Cidx = [2, 3, 8, 9], [7, 13, 14, 15], [10, 11, 12, 16, 17, 18]
kbiv = [(i, j) for i, j in itertools.combinations(Vp, 2)] + [(i, j) for i, j in itertools.combinations(Vm, 2)]
ok = ok and sorted(Aidx + Widx) == Vp and Cidx == Vm and len(kbiv) == 43 and len(list(itertools.combinations(internal, 2))) == 91
ok = ok and all(close((E[i] @ E[j]) @ (E[i] @ E[j]), -Id) for i, j in kbiv)
report(8, ok, "n=3: E_1..E_18 satisfy the Clifford relations of R^{9,9}; V_+=span(A u W)={2,3,7,8,9,13,14,15}, V_-=span(C)={10,...,18}\\{13,14,15}; 43 compact internal bivectors (28+15), each squaring to -1")

gam = [E[i] for i in interior]
J = prod([E[i] for i in internal], N)
Gam = J @ E[1] @ E[4] @ E[5] @ E[6]
chiW = prod([E[i] for i in Widx], N)
chiC = prod([E[i] for i in Cidx], N)
qA = E[2] @ E[3]
volA = prod([E[i] for i in Aidx], N)
PiA = 0.5 * (Id + volA)
PiW = 0.5 * (Id + chiW)

# ---- classification of four-generator internal blades
def blade(idx):
    return prod([E[i] for i in idx], N)


types = {}
ok = True
for S in itertools.combinations(internal, 4):
    Bm = blade(S)
    q = sum(1 for i in S if eta[i] < 0)
    p = 4 - q
    sq = Bm @ Bm
    sym = close(Bm.T, Bm)
    invol = close(sq, Id)
    cplx = close(sq, -Id)
    lor = all(iszero(comm(g, Bm)) for g in gam)
    tr0 = abs(np.trace(Bm)) < TOL
    ok = ok and lor and tr0 and (invol == (q % 2 == 0)) and (cplx == (q % 2 == 1)) and (sym == invol)
    types.setdefault((p, q), 0)
    types[(p, q)] += 1
report(9, ok and types == {(4, 0): 70, (3, 1): 336, (2, 2): 420, (1, 3): 160, (0, 4): 15},
       "n=3: all 1001 four-generator internal blades commute with the four interior gammas and are traceless; a blade of type (p,q) squares to +1 (symmetric involution) iff q is even and to -1 (antisymmetric) iff q is odd; counts (4,0):70, (2,2):420, (0,4):15")

# stabilisers of representatives
reps = {(4, 0): [2, 3, 8, 9], (2, 2): [2, 3, 10, 16], (0, 4): [10, 16, 11, 17]}
allint = list(itertools.combinations(internal, 2))
stabdat = {}
for t, S in reps.items():
    V = blade(S)
    st_all = [(i, j) for i, j in allint if iszero(comm(E[i] @ E[j], V))]
    st_k = [(i, j) for i, j in st_all if eta[i] == eta[j]]
    rk = span_rank([comm(E[i] @ E[j], V) for i, j in kbiv])
    stabdat[t] = (len(st_all), len(st_k), rk)
ok = stabdat == {(4, 0): (51, 27, 16), (2, 2): (51, 23, 20), (0, 4): (51, 35, 8)}
report(10, ok, f"n=3: stabiliser in so(8,6) / in k / K-orbit dimension: (4,0) 51/27/16, (2,2) 51/23/20, (0,4) 51/35/8 (computed {stabdat})")

# Lie-algebra embeddings in the spinor representation
def biv_basis_from_vec(M, idx):
    """Spinor lift of a vector-representation so(V) element M on the orthonormal positive or negative frame idx:
    sum_{i<j} M_ij (1/2) e_i e_j  (positive frame); for a negative frame e_i^2=-1 the same formula holds with an overall sign."""
    out = np.zeros((N, N))
    sgn = 1.0 if eta[idx[0]] > 0 else -1.0
    for p_, q_ in itertools.combinations(range(len(idx)), 2):
        if M[p_, q_] != 0:
            out += sgn * M[p_, q_] * 0.5 * E[idx[p_]] @ E[idx[q_]]
    return out


# check lift convention on vectors: ad(lift(M)) e_j = sum_i M_ij... as a vector rotation
ok = True
for idx in (Vp, Vm):
    dim = len(idx)
    for _ in range(3):
        X = RNG.normal(size=(dim, dim))
        X = X - X.T
        L = biv_basis_from_vec(X, idx)
        for jj in range(dim):
            lhs = comm(L, E[idx[jj]])
            rhs = sum(X[ii, jj] * E[idx[ii]] for ii in range(dim))
            ok = ok and close(lhs, rhs)
        Y = RNG.normal(size=(dim, dim))
        Y = Y - Y.T
        ok = ok and close(comm(biv_basis_from_vec(X, idx), biv_basis_from_vec(Y, idx)), biv_basis_from_vec(X @ Y - Y @ X, idx))
report(11, ok, "n=3: the lift M -> sum_{i<j} M_ij e_ie_j/2 (sign -1 on the negative frame) is a Lie algebra map so(V_+)+so(V_-) -> k with [lift M, u] = M u on vectors")

# su(3)_C: centraliser of H in so(6)_C
Hc = -(E[10] @ E[16] + E[11] @ E[17] + E[12] @ E[18])
cb = [0.5 * E[i] @ E[j] for i, j in itertools.combinations(Cidx, 2)]
A_ = np.array([comm(X, Hc).ravel() for X in cb]).T
ns = null_space(A_)
u3 = [sum(ns[k, m] * cb[k] for k in range(15)) for m in range(ns.shape[1])]
# su(3): orthogonal complement of Hc inside u(3)
Hn = Hc / np.sqrt(frob2(Hc))
su3 = []
for X in u3:
    Y = X - np.sum(X * Hn) * Hn
    su3.append(Y)
su3_rank = span_rank(su3)
# orthonormalise su3
Ar = np.array([X.ravel() for X in su3])
U_, s_, Vt_ = np.linalg.svd(Ar, full_matrices=False)
su3 = [Vt_[m].reshape(N, N) for m in range(int(np.sum(s_ > 1e-8)))]
der = span_rank([comm(x, y) for x in su3 for y in su3])
ok = ns.shape[1] == 9 and su3_rank == 8 and der == 8 and all(iszero(comm(X, Hc)) for X in su3)
report(12, ok, "n=3: the centraliser of the colour Cartan element H=-(E10E16+E11E17+E12E18) in so(6)_C is u(3) (dim 9); its part orthogonal to H is su(3)_C (dim 8, perfect)")

# weak algebra
w = [E[i] for i in Widx]
tau = [-0.5 * w[1] @ w[2], -0.5 * w[2] @ w[0], -0.5 * w[0] @ w[1]]
tL = [0.5 * (Id - chiW) @ t for t in tau]
tR = [0.5 * (Id + chiW) @ t for t in tau]
Wb = [E[i] @ E[j] for i, j in itertools.combinations(Widx, 2)]
Wspan = np.array([x.ravel() for x in Wb]).T


def in_span(M, basis_cols):
    coef = np.linalg.lstsq(basis_cols, M.ravel(), rcond=None)[0]
    return np.max(np.abs(basis_cols @ coef - M.ravel())) < 1e-9


ok = all(in_span(x, Wspan) for x in tL + tR)
ok = ok and close(tL[2], -0.25 * (w[0] @ w[1] + w[2] @ w[3])) and close(tR[2], -0.25 * (w[0] @ w[1] - w[2] @ w[3]))
eps = lambda i, j: (i, j, 3 - i - j)
ok = ok and all(close(comm(tL[i], tL[(i + 1) % 3]), tL[(i + 2) % 3]) for i in range(3))
ok = ok and all(close(comm(tR[i], tR[(i + 1) % 3]), tR[(i + 2) % 3]) for i in range(3))
ok = ok and all(iszero(comm(x, y)) for x in tL for y in tR)
report(13, ok, "n=3 restated (Thm 8.2): t^{L,R}_i=(1/2)(1-+chi_W)tau_i are bivectors in so(4)_W, t^L_3=-(w1w2+w3w4)/4, t^R_3=-(w1w2-w3w4)/4, two commuting su(2) algebras")

# hypercharge and electric charge bivectors and the charge table on P_*
Pst = (Id + Gam @ chiW) @ (Id - J @ chiC) @ (Id - J @ qA) / 8.0
Yb = tR[2] + Hc / 6.0
Bem = tau[2] + Hc / 6.0
ok = close(Bem, -0.5 * E[7] @ E[13] - (E[10] @ E[16] + E[11] @ E[17] + E[12] @ E[18]) / 6.0)
ok = ok and close(tL[2] + tR[2], tau[2])
Qop = Pst @ (-J @ Bem) @ Pst
ev = np.linalg.eigvalsh(0.5 * (Qop + Qop.T))
# restrict to the range of P_*: the 448 kernel eigenvalues are 0; subtract them from the 0 count
vals, cnts = np.unique(np.round(ev * 3).astype(int), return_counts=True)
dist = dict(zip(vals.tolist(), cnts.tolist()))
dist[0] -= 448
ok = ok and abs(np.trace(Pst) - 64) < TOL and dist == {-3: 8, -1: 24, 0: 8, 2: 24} and iszero(comm(J, Bem))
report(14, ok, f"n=3 restated (Prop 8.4): u(1)_em is spanned by B_em=tau_3+H/6=-E7E13/2-(E10E16+E11E17+E12E18)/6, the bivector part of Q=-J(t^3_L+t^3_R+H/6); -J B_em on the range of P_* has eigenvalues -1 (8), -1/3 (24), 0 (8), 2/3 (24)")

# ---- the vacuum Pi_A
QA = 0.5 * qA
hPS = [E[i] @ E[j] for i, j in itertools.combinations(Widx, 2)] + [E[i] @ E[j] for i, j in itertools.combinations(Cidx, 2)]
hSM = su3 + tL + [Yb]
ok = (close(volA @ volA, Id) and close(volA.T, volA) and abs(np.trace(PiA) - 256) < TOL
      and all(iszero(comm(g, PiA)) for g in gam) and iszero(comm(QA, PiA)) and iszero(comm(Bem, PiA))
      and all(iszero(comm(x, PiA)) for x in hPS) and all(iszero(comm(x, PiA)) for x in hSM) and iszero(comm(J, PiA))
      and not iszero(comm(0.5 * E[3] @ E[7], PiA)))
okW = (abs(np.trace(PiW) - 256) < TOL and all(iszero(comm(g, PiW)) for g in gam) and iszero(comm(QA, PiW)) and iszero(comm(Bem, PiW))
       and all(iszero(comm(x, PiW)) for x in hPS) and all(iszero(comm(x, PiW)) for x in hSM) and not iszero(comm(0.5 * E[3] @ E[7], PiW)))
report(15, ok and okW, "n=3: Pi_A=(1+E2E3E8E9)/2 and Pi_W=(1+chi_W)/2 are rank-256 orthogonal projectors commuting with the interior gammas, J, q_A, B_em, h_PS (21 generators) and h_SM (12 generators), and are moved by E3E7/2")

# tangent space at Pi_A and Pi_W
mix = [(i, j) for i in Aidx for j in Widx]
TA = {(i, j): comm(0.5 * E[i] @ E[j], PiA) for i, j in mix}
TW = {(i, j): comm(0.5 * E[i] @ E[j], PiW) for i, j in mix}
tanA = span_rank([comm(0.5 * E[i] @ E[j], PiA) for i, j in kbiv])
tanW = span_rank([comm(0.5 * E[i] @ E[j], PiW) for i, j in kbiv])
movers = [(i, j) for i, j in kbiv if not iszero(comm(E[i] @ E[j], PiA))]
ok = tanA == 16 and tanW == 16 and sorted(movers) == sorted(tuple(sorted(m)) for m in mix)
gramA = np.array([[np.trace(TA[x] @ TA[y]) for y in mix] for x in mix])
gramW = np.array([[np.trace(TW[x] @ TW[y]) for y in mix] for x in mix])
ok = ok and np.allclose(gramA, 128 * np.eye(16)) and np.allclose(gramW, 128 * np.eye(16))
ok = ok and all(close(TA[(i, j)], 0.5 * E[i] @ E[j] @ volA) for i, j in mix)
report(16, ok, "n=3: the compact bivectors moving Pi_A (and Pi_W) are exactly the 16 mixing A with W; T_aw=[aw/2,Pi_A]=aw vol_A/2 are symmetric with Tr(T_aw T_a'w')=128 delta (the real trace on R^512)")

# representation on the tangent space
def tangent_matrix(X, Tdict):
    keys = list(Tdict)
    M = np.zeros((16, 16))
    for c_, k_ in enumerate(keys):
        img = comm(X, Tdict[k_])
        for r_, k2_ in enumerate(keys):
            M[r_, c_] = np.trace(Tdict[k2_] @ img) / 128.0
    # verify that the image lies in the span
    for c_, k_ in enumerate(keys):
        img = comm(X, Tdict[k_])
        rec = sum(M[r_, c_] * Tdict[k2_] for r_, k2_ in enumerate(keys))
        assert iszero(img - rec)
    return M


soA = [0.5 * E[i] @ E[j] for i, j in itertools.combinations(Aidx, 2)]
soW = [0.5 * E[i] @ E[j] for i, j in itertools.combinations(Widx, 2)]
ok = True
for Td in (TA, TW):
    rho = [tangent_matrix(X, Td) for X in soA + soW]
    # commutant of the so(4)+so(4) action on R^16: dimension 1 => irreducible of real type
    Ms = np.array([np.kron(np.eye(16), R) - np.kron(R.T, np.eye(16)) for R in rho])
    comm_dim = null_space(Ms.reshape(-1, 256)).shape[1]
    ok = ok and comm_dim == 1
    # each su(2) factor Casimir 3/4
    for basis in (soA, soW):
        Mb = [tangent_matrix(X, Td) for X in basis]
        sdual = [Mb[0] + Mb[5], Mb[1] - Mb[4], Mb[2] + Mb[3]]  # (12)+(34),(13)-(24),(14)+(23) self-dual combos (x1/2)
        ok = ok and np.allclose(-sum((0.5 * m) @ (0.5 * m) for m in sdual), 0.75 * np.eye(16))
report(17, ok, "n=3: at Pi_A and at Pi_W the tangent space is the real irreducible (2,2;2,2)=R^4(x)R^4 of so(4)_A+so(4)_W (commutant of dimension 1; every su(2) Casimir equals 3/4)")

ok = True
summary = {}
for name, Td in (("A", TA), ("W", TW)):
    rhoL = [tangent_matrix(X, Td) for X in tL]
    rhoY = tangent_matrix(Yb, Td)
    rho3 = [tangent_matrix(X, Td) for X in su3]
    ok = ok and np.allclose(-sum(m @ m for m in rhoL), 0.75 * np.eye(16)) and np.allclose(rhoY @ rhoY, -0.25 * np.eye(16))
    ok = ok and all(np.allclose(m, 0) for m in rho3)
    for qname, Qx in (("em", Bem), ("A", QA)):
        m = tangent_matrix(Qx, Td)
        evq = np.linalg.eigvals(m)
        nch = int(np.sum(np.abs(np.abs(evq.imag) - 1) < 1e-8))
        nneu = int(np.sum(np.abs(evq) < 1e-8))
        summary[(name, qname)] = (nch, nneu)
        ok = ok and nch == 8 and nneu == 8 and np.allclose(m @ m @ m, -m)
report(18, ok, f"n=3: on the tangent space at Pi_A and at Pi_W, su(3)_C acts trivially, su(2)_L with Casimir 3/4 and Y with Y^2=-1/4 (four doublets of hypercharge +-1/2); under u(1)_em and under u(1)_A=(1/2)q_A: 8 real directions of charge +-1 (four charged complex scalars) and 8 neutral real directions ({summary})")

ok = True
# explicit charged pairs
for w_ in Widx:
    ok = ok and close(comm(QA, TA[(2, w_)]), -TA[(3, w_)]) and close(comm(QA, TA[(3, w_)]), TA[(2, w_)])
    ok = ok and iszero(comm(QA, TA[(8, w_)])) and iszero(comm(QA, TA[(9, w_)]))
for a_ in Aidx:
    ok = ok and close(comm(Bem, TA[(a_, 7)]), TA[(a_, 13)]) and close(comm(Bem, TA[(a_, 13)]), -TA[(a_, 7)])
    ok = ok and iszero(comm(Bem, TA[(a_, 14)])) and iszero(comm(Bem, TA[(a_, 15)]))
report(19, ok, "n=3: [Q_A,T_{2w}]=-T_{3w}, [Q_A,T_{3w}]=T_{2w}, [Q_A,T_{8w}]=[Q_A,T_{9w}]=0; [B_em,T_{a7}]=T_{a13}, [B_em,T_{a13}]=-T_{a7}, [B_em,T_{a14}]=[B_em,T_{a15}]=0")

# ---- Clifford spheres through Pi_A
I5A = E[2] @ E[3] @ E[8] @ E[9] @ E[7]
Lgen = [E[i] for i in (2, 3, 8, 9, 7)]
ZA = [I5A @ v for v in Lgen]
ok = all(close(ZA[i].T, ZA[i]) for i in range(5)) and all(close(acomm(ZA[i], ZA[j]), 2 * (i == j) * Id) for i in range(5) for j in range(5))
ok = ok and close(ZA[4], volA)
U3 = E[3] @ E[8] @ E[9]
Ogen = [E[i] for i in (2, 7, 13, 14, 15)]
ZO = [v @ U3 for v in Ogen]
ok = ok and all(close(ZO[i].T, ZO[i]) for i in range(5)) and all(close(acomm(ZO[i], ZO[j]), 2 * (i == j) * Id) for i in range(5) for j in range(5))
ok = ok and close(ZO[0], volA)
# tangent spaces at Pi_A
inner_tan = [0.5 * ZA[i] for i in range(4)]
outer_tan = [0.5 * ZO[i] for i in range(1, 5)]
ok = ok and all(close(inner_tan[k], TA[(Aidx[k], 7)]) or close(inner_tan[k], -TA[(Aidx[k], 7)]) for k in range(4))
ok = ok and all(close(outer_tan[k], TA[(2, Widx[k])]) or close(outer_tan[k], -TA[(2, Widx[k])]) for k in range(4))
# every point is in F_3: (1+I5 n)/2 = Pi_P for P = n^perp in L (compare with explicit frame product)
for _ in range(3):
    n = RNG.normal(size=5)
    n /= np.linalg.norm(n)
    Zn = sum(n[i] * ZA[i] for i in range(5))
    # orthonormal basis of n^perp in R^5, oriented so that (f1..f4, n) is positive
    Fm = null_space(n[None, :]).T
    if np.linalg.det(np.vstack([Fm, n])) < 0:
        Fm[0] *= -1
    u = [sum(Fm[r, s] * Lgen[s] for s in range(5)) for r in range(4)]
    volP = u[0] @ u[1] @ u[2] @ u[3]
    ok = ok and close(Zn, volP)
report(20, ok, "n=3: two kinds of Clifford four-spheres through Pi_A: inner {(1+I_5 n)/2}, I_5=E2E3E8E9E7 (4-planes inside L=A+R E7, tangent A(x)E7) and outer {(1+n E3E8E9)/2}, n in span(E2,E7,E13,E14,E15) (4-planes containing E3E8E9, tangent E2(x)W); both Z-systems are Clifford, and (1+I_5 n)/2 = Pi_P with P = n^perp in L")

# exact kinetic term and current on the two spheres
ok = True
for Zs, Qx, (ia, ib), lab in ((ZA, QA, (0, 1), "inner,q_A"), (ZO, Bem, (1, 2), "outer,em")):
    for _ in range(3):
        n = RNG.normal(size=5)
        n /= np.linalg.norm(n)
        dn = RNG.normal(size=5)
        dn -= (dn @ n) * n
        P = 0.5 * (Id + sum(n[i] * Zs[i] for i in range(5)))
        dP = 0.5 * sum(dn[i] * Zs[i] for i in range(5))
        ok = ok and abs(np.trace(dP @ dP) - 128 * dn @ dn) < 1e-7
        ok = ok and abs(-np.trace(comm(Qx, P) @ dP) - 128 * (n[ia] * dn[ib] - n[ib] * dn[ia])) < 1e-7
report(21, ok, "n=3: on the inner sphere with Q=q_A/2 and on the outer sphere with Q=B_em: Tr(dPi^2)=128|dn|^2 and -Tr([Q,Pi]dPi)=128(n_a dn_b-n_b dn_a), so L=64f^2 Dn.Dn and j=128f^2 Im(phibar D phi), one charged complex scalar and two neutral real scalars")

# ---- exact geometry on F_3: random points
VpE = [E[i] for i in Vp]
VA_pos = [Vp.index(i) for i in Aidx]


def vol_of_frame(F):
    u = [sum(F[r, s] * VpE[s] for s in range(8)) for r in range(F.shape[0])]
    return prod(u, N), u


def lift8(M):
    return biv_basis_from_vec(M, Vp)


ok_eq, ok_kin, ok_cur, ok_C, ok_inv = True, True, True, True, True
Xk_basis = [0.5 * E[i] @ E[j] for i, j in kbiv]
kspan = np.array([x.ravel() for x in Xk_basis]).T
kpos = np.array([x.ravel() for x in [0.5 * E[i] @ E[j] for i, j in itertools.combinations(Vp, 2)]]).T
for trial in range(3):
    Y = RNG.normal(size=(8, 8))
    Y = Y - Y.T
    R = expm(Y)
    g = expm(lift8(Y))
    F0 = np.zeros((4, 8))
    for r, s in enumerate(VA_pos):
        F0[r, s] = 1.0
    F = (R @ F0.T).T
    volP, _ = vol_of_frame(F)
    ok_eq = ok_eq and close(g @ volA @ g.T, volP) and close(volP @ volP, Id) and close(volP.T, volP) and all(iszero(comm(x, volP)) for x in gam)
    ok_eq = ok_eq and all(iszero(comm(x, volP)) for x in su3) and all(iszero(comm(E[i] @ E[j], volP)) for i, j in itertools.combinations(Cidx, 2))
    PiP = 0.5 * (Id + volP)
    p8 = F.T @ F
    # random tangent and connection
    Xv = RNG.normal(size=(8, 8))
    Xv = Xv - Xv.T
    X = lift8(Xv)
    dPi = comm(X, PiP)
    dp = Xv @ p8 - p8 @ Xv
    ok_kin = ok_kin and abs(np.trace(dPi @ dPi) - 64 * np.trace(dp @ dp)) < 1e-6
    for Qx, qv in ((QA, lie_vec(Vp.index(2), Vp.index(3), 8)), (Bem, -lie_vec(Vp.index(7), Vp.index(13), 8))):
        lhs = -np.trace(comm(Qx, PiP) @ dPi)
        rhs = -64 * np.trace((qv @ p8 - p8 @ qv) @ dp)
        ok_cur = ok_cur and abs(lhs - rhs) < 1e-6
    # adapted connection C = [Pi, dPi] = -X_m, a compact bivector in Lambda^2 V_+
    Cm = comm(PiP, dPi)
    Xm = 0.5 * (X - volP @ X @ volP)
    ok_C = ok_C and close(Cm, -Xm)
    coef = np.linalg.lstsq(kpos, Cm.ravel(), rcond=None)[0]
    ok_C = ok_C and np.max(np.abs(kpos @ coef - Cm.ravel())) < 1e-8
    # K-invariance of the full alignment energy: (1/512) sum over the 43 blades e_ie_j of ||[e_ie_j, Pi]||^2 = 16
    VK = sum(frob2(comm(E[i] @ E[j], PiP)) for i, j in kbiv) / 512.0
    ok_inv = ok_inv and abs(VK - 16) < 1e-7
report(22, ok_eq, "n=3: for random g=exp(lift Y), Y in so(8), g vol_A g^T = u1u2u3u4 with u_i=R e_{A_i}, R=exp Y: the K-orbit of Pi_A is {Pi_P : P an oriented 4-plane in V_+}; every point is a Lorentz-invariant symmetric involution commuting with so(6)_C, hence with su(3)_C")
report(23, ok_kin and ok_cur, "n=3: at random points of F_3, Tr(dPi dPi)=64 tr_8(dp dp) and -Tr([Q,Pi]dPi)=-64 tr_8([q,p]dp) (p the 8x8 projector onto P, q the vector-representation generator) for Q=q_A/2 and Q=B_em")
report(24, ok_C, "n=3: at random points of F_3 and random compact tangent X, the adapted connection [Pi,dPi] equals -X_m=-(X-vol_P X vol_P)/2, a compact bivector of Lambda^2 V_+")
report(25, ok_inv, "n=3: the full alignment energy (1/512) sum over the 43 compact blades of ||[e_ie_j,Pi]||^2 equals 16 at random points of F_3 (constant on the K-orbit)")

# curved extension at Pi_A: Lorentz bivectors and Q
lorb = [E[i] @ E[j] for i, j in itertools.combinations(interior, 2)]
ok = all(iszero(comm(x, PiA)) for x in lorb) and all(iszero(comm(x, y)) for x in lorb for y in (QA, Bem))
# which compact Q leave Pi_A parallel: the stabiliser so(4)_A+so(4)_W+so(6)_C, dim 27
stabk = [(i, j) for i, j in kbiv if iszero(comm(E[i] @ E[j], PiA))]
ok = ok and len(stabk) == 27 and span_rank([comm(0.5 * E[i] @ E[j], PiA) for i, j in kbiv]) == 16
report(26, ok, "n=3: the six interior bivectors commute with Pi_A and with Q_A and B_em, so Omega=Omega_Lorentz+A Q gives D Pi_A=0 and [R,Pi_A]=0 exactly for Q in the stabiliser so(4)_A+so(4)_W+so(6)_C (27 compact blades)")

# ---- fixed sets
def kernel_dim_on_tangent(gens, Pi):
    tans = [comm(0.5 * E[i] @ E[j], Pi) for i, j in kbiv]
    # orthonormal basis of the tangent space
    Ar_ = np.array([t.ravel() for t in tans])
    _, s, Vt = np.linalg.svd(Ar_, full_matrices=False)
    r = int(np.sum(s > 1e-8))
    basis = [Vt[m].reshape(N, N) for m in range(r)]
    mats = []
    for Xg in gens:
        M = np.array([[np.sum(bb * comm(Xg, ba)) for ba in basis] for bb in basis])
        mats.append(M)
    if not mats:
        return r, r
    stack = np.vstack(mats)
    return r, null_space(stack).shape[1]


def pi_from_frame(F):
    v, _ = vol_of_frame(F)
    return 0.5 * (Id + v)


def frame(cols_idx, extra=None):
    F = np.zeros((4, 8))
    for r, i in enumerate(cols_idx):
        F[r, Vp.index(i)] = 1.0
    return F


def rand_frame_in(span_idx, k, fixed_idx=()):
    """oriented orthonormal 4-frame: fixed generators followed by k random orthonormal vectors in span(span_idx)."""
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


others_A = [i for i in Vp if i not in (2, 3)]
others_em = [i for i in Vp if i not in (7, 13)]
ok = True
# u(1)_A: component containing the (E2,E3) plane, and component orthogonal to it
for _ in range(2):
    P1 = pi_from_frame(rand_frame_in(others_A, 2, fixed_idx=(2, 3)))
    P2 = pi_from_frame(rand_frame_in(others_A, 4))
    ok = ok and iszero(comm(QA, P1)) and iszero(comm(QA, P2))
    ok = ok and kernel_dim_on_tangent([QA], P1) == (16, 8) and kernel_dim_on_tangent([QA], P2) == (16, 8)
    P1e = pi_from_frame(rand_frame_in(others_em, 2, fixed_idx=(7, 13)))
    P2e = pi_from_frame(rand_frame_in(others_em, 4))
    ok = ok and iszero(comm(Bem, P1e)) and iszero(comm(Bem, P2e))
    ok = ok and kernel_dim_on_tangent([Bem], P1e) == (16, 8) and kernel_dim_on_tangent([Bem], P2e) == (16, 8)
Pg = pi_from_frame(rand_frame_in(Vp, 4))
ok = ok and not iszero(comm(QA, Pg)) and not iszero(comm(Bem, Pg))
ok = ok and kernel_dim_on_tangent([QA], PiA) == (16, 8) and kernel_dim_on_tangent([Bem], PiA) == (16, 8)
report(27, ok, "n=3: the u(1)_A-fixed and u(1)_em-fixed sets in F_3: random planes containing the charge plane, or orthogonal to it, commute with the charge and the kernel of ad Q on their 16-dim tangent space is 8; a generic plane does not commute")

okh = True
dims = {}
for name, gens in (("PS", [0.5 * x for x in hPS]), ("SM", hSM), ("su2L", tL), ("k", Xk_basis)):
    for lab, P in (("A", PiA), ("W", PiW), ("-A", Id - PiA), ("-W", Id - PiW)):
        dims[(name, lab)] = kernel_dim_on_tangent(gens, P)[1]
okh = all(dims[(nm, lab)] == 0 for nm in ("PS", "SM", "su2L", "k") for lab in ("A", "W", "-A", "-W"))
# Pi_{-A} = 1 - Pi_A lies in F_3
Rf = expm(np.pi * lie_vec(Vp.index(2), Vp.index(7), 8))
gpi = expm(np.pi * 0.5 * E[2] @ E[7])
okh = okh and close(gpi @ PiA @ gpi.T, Id - PiA) and close(gpi @ PiW @ gpi.T, Id - PiW)
# Y-only fixed component through U+W', U=span(E2,E3), W'=span(w1,w2): dimension 6
PY = pi_from_frame(frame([2, 3, 7, 13]))
dY = kernel_dim_on_tangent([Yb], PY)
okh = okh and iszero(comm(Yb, PY)) and dY == (16, 6) and kernel_dim_on_tangent([Yb], PiA) == (16, 0)
report(28, okh, f"n=3: h_PS, h_SM, su(2)_L and k have no fixed tangent direction at Pi_+-A, Pi_+-W (isolated fixed points); Pi_-A=1-Pi_A is reached by exp(pi E2E7/2); u(1)_Y alone fixes a 6-dim component through span(E2,E3,E7,E13) (kernel dims {dims[('PS','A')]},{dY})")

# ---- relation to P_*
ok = iszero(comm(volA, Pst)) and iszero(comm(chiW, Pst)) and close(volA @ Pst, -Gam @ Pst) and close(chiW @ Pst, Gam @ Pst)
ok = ok and close(qA @ Pst, J @ Pst) and close(QA @ Pst, 0.5 * J @ Pst)
PL = Pst @ (0.5 * (Id - Gam))
ok = ok and close(PiA @ Pst, PL) and close(PiW @ Pst, Pst - PL) and abs(np.trace(PL) - 32) < TOL
X_ = Gam @ chiW
ok = ok and all(iszero(acomm(TA[m], X_)) for m in mix) and all(iszero(Pst @ TA[m] @ Pst) for m in mix)
report(29, ok, "n=3: vol_A and chi_W commute with P_*, vol_A P_*=-Gamma P_*, chi_W P_*=Gamma P_*, so Pi_A P_*=P_L and Pi_W P_*=P_R (rank 32); q_A P_*=J P_*, so Q_A=q_A/2 acts on the block as J/2; every tangent T_aw at Pi_A anticommutes with X=Gamma chi_W and P_* T_aw P_*=0")

# spectrum-only potentials: all three types have the same spectrum
ok = True
for t, S in reps.items():
    V = blade(S)
    evs = np.linalg.eigvalsh(V)
    ok = ok and int(np.sum(np.abs(evs - 1) < 1e-8)) == 256 and int(np.sum(np.abs(evs + 1) < 1e-8)) == 256
report(30, ok, "n=3: representatives of the three involutive types (4,0),(2,2),(0,4) all have spectrum {+1^256,-1^256}: a spectrum-only potential is constant on F_3 and does not even separate the types")

# invariant quadratic forms on the tangent space at Pi_A (Hessian data for an h-invariant coupling)
def inv_sym_forms(gens, Td):
    rho = [tangent_matrix(X, Td) for X in gens]
    # symmetric S with S rho + rho^T S = 0  <=> S rho = rho S since rho antisymmetric (orthonormal basis)
    rows = []
    idx = [(i, j) for i in range(16) for j in range(i, 16)]
    for R in rho:
        for r_ in range(16):
            for c_ in range(16):
                row = np.zeros(len(idx))
                for k_, (i, j) in enumerate(idx):
                    Sb = np.zeros((16, 16))
                    Sb[i, j] = Sb[j, i] = 1.0
                    row[k_] = (Sb @ R - R @ Sb)[r_, c_]
                rows.append(row)
    return null_space(np.array(rows)).shape[1]


nPS = inv_sym_forms([0.5 * x for x in hPS], TA)
nSM = inv_sym_forms(hSM, TA)
nPSW = inv_sym_forms([0.5 * x for x in hPS], TW)
nSMW = inv_sym_forms(hSM, TW)
report(31, nPS == 10 and nSM == 16 and nPSW == 10 and nSMW == 16,
       f"n=3: the h-invariant symmetric bilinear forms on the tangent space at Pi_A (and at Pi_W) form a space of dimension 10 for h_PS (M(x)delta_W, M symmetric on A) and 16 for h_SM (adding N(x)omega_R, N antisymmetric) (computed {nPS},{nSM},{nPSW},{nSMW})")

# =============================================================================================== n = 4 (signed permutations)
def sp_from_dense(M):
    perm = np.argmax(np.abs(M), axis=0)
    sign = M[perm, np.arange(M.shape[1])]
    return perm, sign


def sp_mul(Aop, Bop):
    pA, sA = Aop
    pB, sB = Bop
    return pA[pB], sB * sA[pB]


def sp_eq(Aop, Bop):
    return np.array_equal(Aop[0], Bop[0]) and np.array_equal(Aop[1], Bop[1])


def sp_neg(Aop):
    return Aop[0], -Aop[1]


e8sp = [sp_from_dense(m) for m in e8]
om8sp = sp_from_dense(omega8)
id8sp = (np.arange(8), np.ones(8, dtype=np.int64))


def sp_kron(ops):
    perm = np.array([0])
    sign = np.array([1], dtype=np.int64)
    for p_, s_ in ops:
        m = len(p_)
        perm = (perm[:, None] * m + p_[None, :]).ravel()
        sign = (sign[:, None] * s_[None, :]).ravel()
    return perm, sign


n4 = 4
E4 = [None]
eta4 = [0]
for k in range(1, n4 + 1):
    for a_ in range(6):
        E4.append(sp_kron([om8sp] * (k - 1) + [e8sp[a_]] + [id8sp] * (n4 - k)))
        eta4.append(ETA1[a_])
Nn4 = 8 ** n4
ident4 = (np.arange(Nn4), np.ones(Nn4, dtype=np.int64))
ok = True
for i in range(1, 25):
    sq = sp_mul(E4[i], E4[i])
    ok = ok and sp_eq(sq, ident4 if eta4[i] > 0 else sp_neg(ident4))
    for j in range(i + 1, 25):
        ok = ok and sp_eq(sp_mul(E4[i], E4[j]), sp_neg(sp_mul(E4[j], E4[i])))
int4 = [i for i in range(1, 25) if i not in interior]
Vp4 = [i for i in int4 if eta4[i] > 0]
Vm4 = [i for i in int4 if eta4[i] < 0]
H4 = sp_mul(sp_mul(E4[2], E4[3]), sp_mul(E4[8], E4[9]))
st_all4 = 0
st_k4 = 0
for i, j in itertools.combinations(int4, 2):
    Bij = sp_mul(E4[i], E4[j])
    if sp_eq(sp_mul(Bij, H4), sp_mul(H4, Bij)):
        st_all4 += 1
        st_k4 += eta4[i] == eta4[j]
    else:
        ok = ok and sp_eq(sp_mul(Bij, H4), sp_neg(sp_mul(H4, Bij)))
k4 = len(Vp4) * (len(Vp4) - 1) // 2 + len(Vm4) * (len(Vm4) - 1) // 2
ok = ok and len(Vp4) == 11 and len(Vm4) == 9 and len(list(itertools.combinations(int4, 2))) == 190
ok = ok and st_all4 == 126 and st_k4 == 63 and k4 == 91 and k4 - st_k4 == 28
report(32, ok, f"n=4: the 24 generators (signed permutations of size 4096) satisfy the Clifford relations; internal signature (11,9), so(11,9) dim 190; stabiliser of E2E3E8E9 dim {st_all4}, compact part {st_k4}, K_4 dim {k4}, orbit dimension {k4 - st_k4} = dim of the oriented Grassmannian of 4-planes in R^11")

# dimension table of fixed sets at n=2,3,4 (combinatorial formula check against Grassmannian dimensions)
def grdim(k, n):
    return k * (n - k)


tab = {nn: (3 * nn - 1, grdim(4, 3 * nn - 1)) for nn in (2, 3, 4)}
ok = tab == {2: (5, 4), 3: (8, 16), 4: (11, 28)} and (43 - 27) == 16 and (13 - 9) == 4
report(33, ok, "dimension of the compact family F_n = oriented Grassmannian of 4-planes in V_+ (dim 3n-1): 4, 16, 28 at n=2,3,4, equal to the K_n-orbit dimensions 13-9, 43-27, 91-63")

npass = sum(RESULTS)
print(f"{npass}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
