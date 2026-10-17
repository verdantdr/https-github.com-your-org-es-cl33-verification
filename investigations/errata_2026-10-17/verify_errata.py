#!/usr/bin/env python3
"""Independent checks of the points raised in an external review of ES_electromagnetic_sphere_2026-10-17.

    python3 investigations/errata_2026-10-17/verify_errata.py

Prints one PASS/FAIL line per check and exits 0 iff all pass.  Each check either confirms a defect in the
paper as written (the line says 'defect confirmed') or confirms that the corrected statement holds.
"""
import itertools
import sys
from functools import reduce

import numpy as np
import sympy as sp

FAILS = []


def report(ok, tag, text):
    print(f"{'PASS' if ok else 'FAIL'} {tag}: {text}")
    if not ok:
        FAILS.append(tag)


# ----------------------------------------------------------------------------- realisation of Cl(3,3)
sx = np.array([[0, 1], [1, 0]]); sz = np.array([[1, 0], [0, -1]]); ep = np.array([[0, -1], [1, 0]]); I2 = np.eye(2, dtype=int)
kr = lambda *m: reduce(np.kron, m)
e = [None, kr(sx, I2, I2), kr(sz, sx, I2), kr(sz, sz, sx), kr(ep, I2, I2), kr(sz, ep, I2), kr(sz, sz, ep)]
eta = [None, 1, 1, 1, -1, -1, -1]
I8 = np.eye(8, dtype=int)
for a in range(1, 7):
    for b in range(1, 7):
        assert np.array_equal(e[a] @ e[b] + e[b] @ e[a], 2 * (eta[a] if a == b else 0) * I8)

# ============================================================================= E1  Theorem 18.7 (universal Fock vacuum)
# Fock module of eight real Majorana components: theta_i = i*gamma_i on C^16 (Jordan-Wigner), rho(A) = -1/4 sum A_ij theta_i theta_j.
Z = np.diag([1, -1]).astype(complex); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Id = np.eye(2)
gam = []
for k in range(4):
    for P in (X, Y):
        gam.append(reduce(np.kron, [Z] * k + [P] + [Id] * (3 - k)))
theta = [1j * g for g in gam]


def rho(A):
    return -0.25 * sum(A[i, j] * theta[i] @ theta[j] for i in range(8) for j in range(8))


def joint_spectrum(A, Q):
    HF, QF = -1j * rho(A), -1j * rho(Q)
    assert np.allclose(HF @ QF, QF @ HF)
    w, V = np.linalg.eigh(HF + 1e-3 * QF)          # generic combination separates joint eigenspaces
    pairs = [(round(float(np.real(v.conj() @ HF @ v)), 6), round(float(np.real(v.conj() @ QF @ v)), 6)) for v in V.T]
    out = {}
    for p in pairs:
        out[p] = out.get(p, 0) + 1
    return out


E = 2.0
# the paper's own case (Theorem 5.7): A = J H(p) with H = gamma^0 gamma^a p_a + m gamma^0, Q = J; Tr(AQ) = 0
J = e[2] @ e[3]
H = 1.0 * e[1] @ e[6] + np.sqrt(3.0) * e[1]                    # |p| = 1, m = sqrt 3, E = 2
spec_dirac = joint_spectrum(J @ H, J)
ground = min(spec_dirac)
report(ground == (-4.0, 0.0) and spec_dirac[ground] == 1, "E1a",
       f"Dirac case A=JH, Q=J: ground state (H,Q)={ground}, unique and neutral; Tr(AQ)={np.trace(J @ H @ J):.0f}")
# the review's counterexample: A = E e4, Q = e4 satisfy every stated hypothesis
A, Q = E * e[4].astype(float), e[4].astype(float)
hyp = (np.allclose(A.T, -A) and np.allclose(A @ A, -E**2 * np.eye(8)) and np.allclose(Q.T, -Q)
       and np.allclose(Q @ Q, -np.eye(8)) and np.allclose(A @ Q, Q @ A))
spec_ce = joint_spectrum(A, Q)
ground_ce = min(spec_ce)
report(hyp and abs(ground_ce[1]) == 2.0, "E1b",
       f"defect confirmed: A=E e4, Q=e4 meet the hypotheses of Thm 18.7 but the ground state {ground_ce} carries charge {ground_ce[1]:+.0f}")
# corrected hypothesis Tr(AQ)=0 is necessary and sufficient: scan all sign patterns on four common planes
target = {(-2 * E, 0.0): 1, (-E, 1.0): 2, (-E, -1.0): 2, (0.0, 0.0): 4, (0.0, 2.0): 1, (0.0, -2.0): 1,
          (E, 1.0): 2, (E, -1.0): 2, (2 * E, 0.0): 1}
J2 = np.array([[0, -1], [1, 0]])
rng = np.random.default_rng(7)
Oq, _ = np.linalg.qr(rng.normal(size=(8, 8)))
ok_scan = True
for a_signs in itertools.product([1, -1], repeat=4):
    for s_signs in itertools.product([1, -1], repeat=4):
        A = Oq @ np.kron(np.diag(a_signs), J2) @ Oq.T * E
        Q = Oq @ np.kron(np.diag(s_signs), J2) @ Oq.T
        balanced = abs(np.trace(A @ Q)) < 1e-9
        sp_ = {k: v for k, v in joint_spectrum(A, Q).items()}
        ok_scan &= (sp_ == target) == balanced
report(ok_scan, "E1c", "over all 256 commuting pairs (A^2=-E^2, Q^2=-1, both antisymmetric) the table of Thm 18.7 holds iff Tr(AQ)=0")
# the paper's intended charges satisfy Tr(AQ)=0
wT = e[1] @ e[2] @ e[3]
pv = np.array([0.3, -1.1, 0.7])
cases = {"T: A=omega_T p.e, Q=omega_T": (wT @ sum(pv[i] * e[i + 1] for i in range(3)), wT),
         "S: A=p.(e4,e5,e6), Q=J": (sum(pv[i] * e[i + 4] for i in range(3)), J),
         "M12c: A=Dirac 2+1, Q=omega_W": (e[1] @ (pv[0] * e[5] + pv[1] * e[6]) + pv[2] * e[1], e[1] @ e[5] @ e[6])}
ok_cases = True
for name, (A, Q) in cases.items():
    ok_cases &= np.allclose(A @ Q, Q @ A) and abs(np.trace(A @ Q)) < 1e-12
report(ok_cases, "E1d", "the charges named after Thm 18.7 (omega_T on T, J on S, omega_W on a (1,2) plane) commute with A and have Tr(AQ)=0")

# ============================================================================= E2  Theorem 18.1 (transpose rule)
t = 0.4
f1 = np.cosh(t) * e[4] + np.sinh(t) * e[1]
w_t = f1 @ e[5] @ e[6]
sq_ok = np.allclose(w_t @ w_t, np.eye(8))                    # -(-1)^q with q = 3 gives +1
sym_fail = not np.allclose(w_t.T, w_t)
report(sq_ok and sym_fail and np.allclose(f1 @ f1, -np.eye(8)), "E2a",
       "defect confirmed: for the tilted (0,3) plane <cosh t e4 + sinh t e1, e5, e6> the square rule holds (omega^2=+1) but omega^T != omega")
ok_coord = True
for W in itertools.combinations(range(1, 7), 3):
    w = e[W[0]] @ e[W[1]] @ e[W[2]]
    q = sum(1 for a in W if eta[a] < 0)
    ok_coord &= np.array_equal(w @ w, -(-1) ** q * I8) and np.array_equal(w.T, -(-1) ** q * w)
    for a in range(1, 7):
        c = e[a] @ w - (1 if a in W else -1) * w @ e[a]
        ok_coord &= not c.any()
report(ok_coord, "E2b", "for all 20 coordinate three-planes the square, transpose and (anti)commutation rules of Thm 18.1 hold exactly")
# general nondegenerate planes: square rule and (anti)commutation with W and W^perp survive
ok_gen = True
for trial in range(20):
    L = np.eye(6)
    for (a, b) in [(0, 3), (1, 4), (2, 5), (0, 4), (1, 5), (2, 3)]:   # random boosts mixing V+ and V-
        s = rng.normal()
        Bm = np.eye(6); Bm[a, a] = Bm[b, b] = np.cosh(s); Bm[a, b] = Bm[b, a] = np.sinh(s)
        L = L @ Bm
    for (a, b) in [(0, 1), (3, 4), (1, 2)]:
        s = rng.normal()
        R = np.eye(6); R[a, a] = R[b, b] = np.cos(s); R[a, b] = -np.sin(s); R[b, a] = np.sin(s)
        L = L @ R
    vecs = [sum(L[i, j] * e[i + 1] for i in range(6)) for j in range(6)]     # images of e1..e6, still orthonormal
    W, Wp = [vecs[0], vecs[1], vecs[3]], [vecs[2], vecs[4], vecs[5]]           # a (2,1) plane and its complement
    w = W[0] @ W[1] @ W[2]
    ok_gen &= np.allclose(w @ w, -(-1) ** 1 * np.eye(8))
    ok_gen &= all(np.allclose(v @ w, w @ v) for v in W) and all(np.allclose(v @ w, -w @ v) for v in Wp)
report(ok_gen, "E2c", "for 20 random tilted (2,1) planes the square rule and the (anti)commutation with W and W^perp still hold")

# planes adapted to V = V+ (+) V-: rotate inside the two triples by the spin lift exp(theta e_a e_b / 2); transpose rule survives
from scipy.linalg import expm
ok_adapt = True
for trial in range(10):
    Rs = np.eye(8)
    for (a, b) in [(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)]:
        Rs = Rs @ expm(rng.normal() * (e[a] @ e[b]) / 2)
    ok_adapt &= np.allclose(Rs @ Rs.T, np.eye(8))
    for W in [(1, 2, 4), (1, 5, 6), (4, 5, 6), (1, 2, 3)]:
        w = Rs @ (e[W[0]] @ e[W[1]] @ e[W[2]]) @ Rs.T
        q = sum(1 for a in W if eta[a] < 0)
        ok_adapt &= np.allclose(w.T, -(-1) ** q * w)
report(ok_adapt, "E2d", "for planes spanned by vectors of V+ and V- (coordinate planes rotated inside the triples) the transpose rule holds: the spin lifts are orthogonal")

# ============================================================================= E3  benchmark sentence after Theorem 20.5
# tensor model: T = Sym^2(C^4) (x) Sym^2(C^2) inside Sym^2(C^8), S -> U S U^T
def sym_basis(n):
    B = []
    for i in range(n):
        for j in range(i, n):
            M = np.zeros((n, n)); M[i, j] = M[j, i] = 1.0
            B.append(M / np.linalg.norm(M))
    return B


basisT = [np.kron(A4, A2) for A4 in sym_basis(4) for A2 in sym_basis(2)]           # orthonormal, 30 elements
BT = np.array([b.reshape(-1) for b in basisT]).T                                    # 64 x 30


def coords(S):
    return BT.T @ S.reshape(-1)


def gens():
    g4, g2 = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            M = np.zeros((4, 4), complex); M[i, j] = M[j, i] = 0.5; g4.append(M)
            M = np.zeros((4, 4), complex); M[i, j] = -0.5j; M[j, i] = 0.5j; g4.append(M)
    for k in range(1, 4):
        d = np.zeros(4); d[:k] = 1; d[k] = -k; g4.append(np.diag(d / np.sqrt(2 * k * (k + 1))).astype(complex))
    g2 = [X / 2, Y / 2, Z / 2]
    return [np.kron(m, Id) for m in g4], [np.kron(np.eye(4), m) for m in g2]


G4, G2 = gens()


def rep_on_T(Tg):
    return np.array([coords(Tg @ b + b @ Tg.T) for b in basisT]).T                   # 30 x 30 matrix of S -> TS + ST^T


R4 = [rep_on_T(g) for g in G4]; R2 = [rep_on_T(g) for g in G2]
I30 = np.eye(30)
C4 = sum((np.kron(r, I30) + np.kron(I30, r)) @ (np.kron(r, I30) + np.kron(I30, r)) for r in R4)
C2 = sum((np.kron(r, I30) + np.kron(I30, r)) @ (np.kron(r, I30) + np.kron(I30, r)) for r in R2)
swap = np.zeros((900, 900))
for i in range(30):
    for j in range(30):
        swap[i * 30 + j, j * 30 + i] = 1
Psym = (np.eye(900) + swap) / 2
labels = {}
w4, _ = np.linalg.eig(C4)
vals4 = sorted(set(np.round(np.real(w4), 6)))
vals2 = sorted(set(np.round(np.real(np.linalg.eig(C2)[0]), 6)))


def proj(Cop, val, others):
    P = np.eye(900, dtype=complex)
    for o in others:
        if abs(o - val) > 1e-6:
            P = P @ (Cop - o * np.eye(900)) / (val - o)
    return P


comps = []
for a in vals4:
    for b in vals2:
        P = Psym @ proj(C4, a, vals4) @ proj(C2, b, vals2)
        r = int(round(np.real(np.trace(P))))
        if r > 0:
            comps.append(((a, b), r, P))
dims = sorted(r for _, r, _ in comps)
report(dims == sorted([175, 35, 100, 20, 135]), "E3a", f"Sym^2 T splits into components of dimensions {dims} (expected 175,35,100,20,135)")
# order as in the paper: (35,5),(35,1),(20',5),(20',1),(45,3) -> identify by dimension
order = {175: 0, 35: 1, 100: 2, 20: 3, 135: 4}
Pk = [None] * 5
for _, r, P in comps:
    Pk[order[r]] = P


def I_k(S):
    s = coords(S).astype(complex)
    ss = np.kron(s, s)
    return np.array([np.real(ss.conj() @ P @ ss) for P in Pk])


epsC = np.zeros((4, 4, 4, 4))
for perm in itertools.permutations(range(4)):
    epsC[perm] = np.linalg.det(np.eye(4)[list(perm)])
eps2 = np.array([[0, 1], [-1, 0]])


def Dhol(S):
    S6 = S.reshape(4, 2, 4, 2)                                         # S[a,s,e,t]
    Pm = np.einsum('su,tv,aset,bufv->abef', eps2, eps2, S6, S6)
    return np.einsum('abcd,efgh,abef,cdgh->', epsC, epsC, Pm, Pm) / 24


u0 = np.kron([0, 0, 0, 1], [1, 0]).astype(float)
S0 = np.outer(u0, u0)
Sid = np.kron(np.eye(4), np.eye(2)) / np.sqrt(8)
f0, fid = I_k(S0), I_k(Sid)
did = abs(Dhol(Sid))
report(np.allclose(f0, [1, 0, 0, 0, 0]) and np.isclose(fid.sum(), 1.0), "E3b",
       f"f_k(S0) = {np.round(f0, 6)}, f_k(1x1/sqrt8) = {np.round(fid, 6)}, d(1x1) = {did:.6f}")
report(np.isclose(did, 3 / 8 * fid[3]), "E3c", "|D(1x1)| = (3/8) I_4 at the identity configuration (equality case of Thm 20.2(ii))")
lam = np.array([1, 1.7, 2.3, 3.1, 2.9])
diff = lambda k: lam @ fid - 2 * k * did - lam @ f0
report(np.isclose(diff(0), 0.9) and np.isclose(diff(7.2), 0, atol=1e-9) and diff(7.0) > 0 and diff(7.5) < 0, "E3d",
       f"defect confirmed: Lambda_kappa(1x1) - Lambda(SM) = 0.9 - |kappa|/8 ({diff(7.0):+.4f} at 7, {diff(7.5):+.4f} at 7.5): "
       "the identity orbit is the HIGHER minimum for |kappa|<=7 and the LOWER one for |kappa|>=7.5")

Gk = np.array([90, 108, -18, 84]); Fk = np.array([3, 3, 1.5, 2])
report(np.isclose(Gk @ fid[1:], 48) and np.isclose(Fk @ fid[1:], 7 / 4), "E3e",
       "cross-check against the paper: with these f_k, Theorem 20.9 gives the excess (L/64pi^2) g^4 (48 - 7x/4) of 1x1 over S0, as stated")
# gauge part at S0: Tr M_V^4 = g^4 * 33 (c_1 = 33 in Theorem 20.9)
# ============================================================================= E4  Carroll-Field-Jackiw dispersion
w, k, p = sp.symbols('omega k p', real=True)
kk = w**2 - k**2          # k_mu k^mu for wave vector along the same axis as the gradient, k = |k|
# (k.k)^2 + (k.k)(p.p) - (k.p)^2 = 0 with p^mu the Chern-Simons vector
spacelike = sp.expand(kk**2 + kk * (-p**2) - (-k * p)**2)
timelike = sp.expand(kk**2 + kk * (p**2) - (w * p)**2)
sl0 = sp.solve(spacelike.subs(k, 0), w)
tl0 = sp.solve(timelike.subs(k, 0), w)
report(set(sl0) == {0, p, -p} or set(sl0) == {0, -p, p}, "E4a",
       f"defect confirmed: spacelike constant gradient, omega at zero wave vector = {sl0}: one branch gapped (|omega|=|p|), one gapless")
report(set(tl0) == {0}, "E4b", f"timelike constant gradient: omega at zero wave vector = {tl0}: both branches gapless")
# no Lorentz-invariant mass: the gapped branch is not omega^2 = k^2 + m^2
branch = sp.solve(spacelike.subs(p, 1), w)
report(not any(sp.simplify(b**2 - (k**2 + 1)) == 0 for b in branch), "E4c",
       "the gapped branch is not a Proca shell omega^2 = k^2 + m^2 (Lorentz-violating dispersion)")

# ============================================================================= E5  normalisations of Sections 19-21
# tensor-model vector masses at S = r S0 from (M_V^2)_ab = 2 g^2 Re Tr (T_a S + S T_a^T)^dag (T_b S + S T_b^T), one coupling g, r = 1
gens_all = G4 + G2
Mv = np.array([[2 * np.real(np.trace((Ta @ S0 + S0 @ Ta.T).conj().T @ (Tb @ S0 + S0 @ Tb.T))) for Tb in gens_all] for Ta in gens_all])
ev = np.sort(np.round(np.linalg.eigvalsh(Mv), 9))
nz = ev[ev > 1e-9]
report(len(nz) == 9 and np.allclose(nz[:8], 1.0) and np.isclose(nz[8], 5.0) and np.isclose(np.sum(nz**2), 33), "E5a",
       f"tensor normalisation: nonzero squared vector masses at S = |S| S0 are {np.round(nz, 6)} x g^2 |S|^2: M_WR = M_X = g|S|, M_Z' = sqrt5 g|S|, Tr M_V^4 = 33 g^4|S|^4 (c_1 = 33 of Thm 20.9)")
# (not a computation) |S| = r: Prop 20.1 gives Tr Phi^T Phi = 4|S|^2 and Thm 19.1(v) gives Tr Phi0^T Phi0 = Tr P_nu = 4, so
# |iota(r Phi0)| = r, while Sec. 21 uses v = |Phi|/sqrt2 = sqrt2 r; M_N = |y| r in both (Sec. 19: H^2 = p^2 + y^2 r^2 P_nu; Sec. 20: M_F = yS).
# kernel convention of Sec 20.5: Tr M_V^4 = (1/16) g^4 v^4 T(u), Sigma m_Weyl^4 = y^4 v^4 Q / ... -> x consistency
T_u0 = 8 * 4**2 + 20**2                               # eigenvalues of G^{-1}K at u0: 4 (x8), 20 (x1) (Thm 21.1(b))
report(T_u0 * 3 / 16 == 99, "E5c", "Sec. 20.5: (3/16) T(u0) = 99 = 3 x 33, i.e. Tr M_V^4 = (1/16) g^4 v^4 T: M_WR = g v with v = |Phi|/sqrt2, "
       "which agrees with E5a for the canonical field S = sqrt2 iota(Phi) (|S| = v): one and the same g")
# fermion term: 2 * (1/4) y^4 Tr(Phi^T Phi)^2 = 2 y^4 v^4 Q (|Phi| = sqrt2 v); matching 8 x Q g^4 v^4 needs x = y^4/(4 g_kernel^4) = y^4/g_tensor^4
yv, gk = sp.symbols('y g_k', positive=True)
xk = sp.Rational(1, 4) * yv**4 / gk**4
# fermion side: Sigma_Weyl m^4 = (1/4) Tr((y Phi)^T (y Phi))^2 = y^4 v^4 Q (|Phi| = sqrt2 v); the term 8 x Q g^4 v^4 of E_x then needs
# x = y^4/(4 g^4) with y of eq. (Saction); with M_N = |y| r = |y| v/sqrt2 and M_WR = g v this is (M_N/M_WR)^4.
MN, MW = yv * sp.Symbol('r') , gk * sp.sqrt(2) * sp.Symbol('r')
report(sp.simplify(xk - (MN / MW)**4) == 0, "E5d",
       "Sec. 20.5's x = y^4/(4 g^4) (couplings of eq. Saction and Thm 21.1) equals (M_N/M_WR)^4; Sec. 20.4's x = y'^4/g^4 with y' = y/sqrt2 is the same number")

# ============================================================================= E6  tau_i convention (s08 vs s19 / Appendix A)
# any three anticommuting generators of square +1 (the weak directions E7, E13, E14 are temporal) model w1, w2, w3
w1, w2, w3 = e[1], e[2], e[3]
cyc = [(w2, w3), (w3, w1), (w1, w2)]
tau08 = [-0.5 * a @ b for a, b in cyc]                       # s08: tau_i = -1/2 w_j w_k, (i,j,k) cyclic, no sum
tau19 = [-0.5 * (a @ b - b @ a) for a, b in cyc]             # s19/sA read with summation: -1/2 eps_ijk w_j w_k = -w_j w_k
br = lambda X, Y: X @ Y - Y @ X
ok08 = all(np.allclose(t @ t, -0.25 * np.eye(8)) for t in tau08) and np.allclose(br(tau08[0], tau08[1]), tau08[2])
bad19 = all(np.allclose(t @ t, -np.eye(8)) for t in tau19) and np.allclose(br(tau19[0], tau19[1]), 2 * tau19[2])
report(ok08 and bad19, "E6a", "defect confirmed: s08's tau_i = -1/2 w_j w_k (cyclic) has tau^2 = -1/4 and closes on su(2); the s19/App. A formula "
       "-1/2 eps_ijk w_j w_k summed equals -w_j w_k = 2 tau_i (tau^2 = -1, [tau1,tau2] = 2 tau3)")
# ============================================================================= E7  Omega and gamma_5 (Prop 4.6 vs Sec. 5)
Om = e[1] @ e[4] @ e[5] @ e[6]
Gam = J @ Om
gamma5 = -Gam                                                # Theorem 4.1 with i = +J: gamma_5 = -Gamma
report(np.allclose(Om, J @ gamma5) and np.allclose(Om, -J @ Gam), "E7a",
       "defect confirmed: with i = J, Theorem 4.1 gives gamma_5 = -Gamma, so Omega = -J Gamma = i gamma_5 (Prop 4.6); Sec. 5's 'real form of -i gamma_5' has the sign of i = -J")
# ============================================================================= E8  rescaling remark at the end of Sec. 17.5
s_, t_, R0, m_, I0, k0, h0 = sp.symbols('s t R0 m I0 kappa0 h0', positive=True)
eps = lambda m, I, k: R0 / (m * I * k**2)
fixed_k = sp.simplify(eps(t_ * m_, s_ * I0, k0) / eps(m_, I0, k0))
fixed_h = sp.simplify(eps(t_ * m_, s_ * I0, h0 / (s_ * I0)) / eps(m_, I0, h0 / I0))
report(fixed_k == 1 / (s_ * t_) and fixed_h == s_ / t_, "E8a",
       f"defect confirmed: (I0,m) -> (sI0,tm) multiplies eps by {fixed_k} at fixed kappa0 but by {fixed_h} at fixed h0; the paper's 1/(st) needs kappa0 fixed")

print()
print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
