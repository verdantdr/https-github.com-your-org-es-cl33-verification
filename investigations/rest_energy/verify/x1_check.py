#!/usr/bin/env python3
"""Adversarial check x1 of track d1 ("the undivided six-dimensional energy and its differentiation").

Independent of investigations/rest_energy/verify/d1_unitary_energy.py (nothing is imported from it).
Conventions are rebuilt from the paper sources (tex/s02 realisation, s03 Lemma 3.4, s12 Theorem 12.3,
s16 Theorem 16.2 / Proposition 16.5, s17 Propositions 17.8, Theorems 17.13, 17.19, 17.21):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1);  X = sum_i (B_i e_i + E_i e_{i+3}),  z = B + iE;
  G e_i = e_{i+3}, G e_{i+3} = -e_i;  q = z.z;  E6 = |z|^2/2;  E0 = |q|/2;  S = E x B (frame), S_t = E x B_t = -S.
Methods differ from the derive script where possible: adjoint matrices from the abstract Clifford rule
[e_A e_B, e_C] = 2 eta_BC e_A - 2 eta_AC e_B cross-checked against the 8x8 realisation by least squares;
dimensions by numerical SVD (tolerances stated) plus exact checks of the claimed bases; the rest frame by
the textbook finite Lorentz boost; the interior Euler-Lagrange equation derived symbolically from the
Lagrangian; the Lie-Poisson flow with RK45.
Prints one PASS/FAIL line per checked item and exits 0 only if all pass.  Seeded: default_rng(4242).
"""
import sys
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.linalg import expm, null_space
from scipy.integrate import solve_ivp

RNG = np.random.default_rng(4242)
FAILS = []


def check(cid, ok, text):
    if ok:
        print(f"PASS {cid}: {text}")
    else:
        print(f"FAIL {cid}: {text}")
        FAILS.append(cid)


def note(text):
    print(f"    . {text}")


def numrank(M, tol=1e-9):
    s = np.linalg.svd(np.asarray(M, dtype=float), compute_uv=False)
    return int(np.sum(s > tol * max(1.0, s[0] if len(s) else 1.0)))


# ------------------------------------------------------------------ Clifford data (abstract and realised)
ETAD = [1, 1, 1, -1, -1, -1]
ETA = np.diag(ETAD).astype(float)


def adbiv(A, B, etad):
    """matrix of v -> [e_A e_B, v] on vectors, from the abstract rule (A != B, 0-based)."""
    n = len(etad)
    M = np.zeros((n, n))
    M[A, B] += 2 * etad[B]      # [e_A e_B, e_B] = 2 eta_BB e_A
    M[B, A] -= 2 * etad[A]      # [e_A e_B, e_A] = -2 eta_AA e_B
    return M


sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
ep = np.array([[0, -1], [1, 0]])
I2 = np.eye(2, dtype=int)
kr = lambda a, b, c: np.kron(np.kron(a, b), c)
e8 = [kr(sx, I2, I2), kr(sz, sx, I2), kr(sz, sz, sx), kr(ep, I2, I2), kr(sz, ep, I2), kr(sz, sz, ep)]
VECBASIS = np.array([m.reshape(-1) for m in e8], dtype=float).T   # 64 x 6


def vcoef(Y):
    c, res, *_ = np.linalg.lstsq(VECBASIS, Y.reshape(-1).astype(float), rcond=None)
    assert np.allclose(VECBASIS @ c, Y.reshape(-1)), "not a vector"
    return c


def adreal(Bv):
    return np.array([vcoef(Bv @ e8[a] - e8[a] @ Bv) for a in range(6)]).T


CYC = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
adL = [0.5 * (adbiv(j + 3, k + 3, ETAD) - adbiv(j, k, ETAD)) for (i, j, k) in CYC]
adK = [0.5 * (adbiv(k, j + 3, ETAD) - adbiv(j, k + 3, ETAD)) for (i, j, k) in CYC]
G = np.zeros((6, 6))
for i in range(3):
    G[i + 3, i] = 1
    G[i, i + 3] = -1
I6 = np.eye(6)


def zof(x):
    x = np.asarray(x)
    return x[:3] + 1j * x[3:]


def xof(z):
    z = np.asarray(z)
    return np.concatenate([z.real, z.imag])


def E6(z):
    return 0.5 * np.vdot(z, z).real


def E0(z):
    return 0.5 * abs(np.dot(z, z))


def Sframe(z):
    B, E = z.real, z.imag
    return np.cross(E, B)


def lorentz_ad():
    return adL + adK


def comm(A, B):
    return A @ B - B @ A


# =============================================================== d1-0 conventions
ok = True
for a in range(6):
    for b in range(6):
        ok &= np.array_equal(e8[a] @ e8[b] + e8[b] @ e8[a], 2 * ETAD[a] * (a == b) * np.eye(8, dtype=int))
for A in range(6):
    for B in range(6):
        if A != B:
            ok &= np.allclose(adreal(e8[A] @ e8[B]), adbiv(A, B, ETAD))
eps = lambda i, j, k: int(np.sign((j - i) * (k - i) * (k - j)))
for i in range(3):
    for j in range(3):
        for k in range(3):
            if eps(i, j, k) == 0:
                continue
            # [L_i,L_j] = eps L_k, [L_i,K_j] = eps K_k, [K_i,K_j] = -eps L_k (ad is a faithful Lie hom here)
            ok &= np.allclose(comm(adL[i], adL[j]), eps(i, j, k) * adL[k])
            ok &= np.allclose(comm(adL[i], adK[j]), eps(i, j, k) * adK[k])
            ok &= np.allclose(comm(adK[i], adK[j]), -eps(i, j, k) * adL[k])
ok &= np.allclose(adK[2] @ I6[:, 0], I6[:, 4])           # [K3, e1] = e5 (Example 12.4)
ok &= np.allclose(adK[2] @ I6[:, 4], I6[:, 0])           # [K3, e5] = e1
# action laws on (B,E), symbolic
Bs = sp.Matrix(sp.symbols('B1:4', real=True))
Es = sp.Matrix(sp.symbols('E1:4', real=True))
xs = sp.Matrix.vstack(Bs, Es)
for i in range(3):
    ei = sp.Matrix([1 if r == i else 0 for r in range(3)])
    dK = sp.Matrix(np.round(adK[i]).astype(int)) * xs if np.allclose(adK[i], np.round(adK[i])) else sp.nsimplify(sp.Matrix(adK[i])) * xs
    dL = sp.nsimplify(sp.Matrix(adL[i])) * xs
    ok &= sp.simplify(dK[3:, 0] - ei.cross(Bs)) == sp.zeros(3, 1)
    ok &= sp.simplify(dK[:3, 0] + ei.cross(Es)) == sp.zeros(3, 1)
    ok &= sp.simplify(dL[:3, 0] - ei.cross(Bs)) == sp.zeros(3, 1)
    ok &= sp.simplify(dL[3:, 0] - ei.cross(Es)) == sp.zeros(3, 1)
# Lemma 3.4: K(v) = (1/2)[sum e_i e_{i+3}, v], S(v) = T v T^{-1} with T = e1e2e3; S = eta and G = -K o S
Pole = sum(e8[i] @ e8[i + 3] for i in range(3))
Kgr = 0.5 * adreal(Pole)
T8 = e8[0] @ e8[1] @ e8[2]
Tinv = np.linalg.inv(T8)
Smap = np.array([vcoef(T8 @ e8[a] @ Tinv) for a in range(6)]).T
ok &= np.allclose(Smap, ETA) and np.allclose(-Kgr @ Smap, G)
# G: G^2 = -1, eta-self-adjoint, anti-isometry, commutes with l_G
ok &= np.allclose(G @ G, -I6) and np.allclose(ETA @ G.T @ ETA, G) and np.allclose(G.T @ ETA @ G, -ETA)
ok &= all(np.allclose(comm(M, G), 0) for M in lorentz_ad())
ok &= all(np.allclose(M.T @ ETA + ETA @ M, 0) for M in lorentz_ad())
check("x1-d1-0", ok, "realisation, abstract ad rule, brackets of Thm 12.3, [K3,e1]=e5, K_i: dE=e_i x B, dB=-e_i x E, "
      "L_i rotate B,E; Ad_T|V=eta, G=-K o Ad_T (Lemma 3.4); G^2=-1, G=G^dagger, anti-isometry, [l_G,G]=0")


# =============================================================== d1-1a invariant forms (+ class test)
def sym_basis(n):
    out = []
    for a in range(n):
        for b in range(a, n):
            M = np.zeros((n, n))
            M[a, b] = M[b, a] = 1
            out.append(M)
    return out


SB = sym_basis(6)


def invariant_forms(gens_anti, gens_dual=()):
    """symmetric N with N R + R^T N = 0 for R in gens_anti (infinitesimal isometries)."""
    rows = []
    for R in list(gens_anti) + list(gens_dual):
        rows.append(np.array([(N @ R + R.T @ N).reshape(-1) for N in SB]).T)
    A = np.vstack(rows)
    ns = null_space(A, rcond=1e-10)
    return [sum(ns[k, c] * SB[k] for k in range(len(SB))) for c in range(ns.shape[1])]


rotinv = invariant_forms(adL)
both = invariant_forms(adL, [G])
lginv = invariant_forms(adL + adK)
Pblk = np.diag([1, 1, 1, 0, 0, 0.])
Qblk = np.diag([0, 0, 0, 1, 1, 1.])
Cblk = np.block([[np.zeros((3, 3)), np.eye(3)], [np.eye(3), np.zeros((3, 3))]])
claimed_rot = [Pblk, Qblk, Cblk]
in_span = lambda Ms, basis: numrank(np.array([m.reshape(-1) for m in basis + Ms])) == numrank(np.array([m.reshape(-1) for m in basis]))
ok = len(rotinv) == 3 and in_span(claimed_rot, rotinv)
ok &= len(both) == 1 and in_span([I6], both)
ok &= len(lginv) == 2 and in_span([ETA, ETA @ G], lginv)
# none of the l_G-invariant forms is definite: a*eta + b*eta G has trace 0
al, be = sp.symbols('alpha beta', real=True)
Nsym = al * sp.Matrix(ETA.astype(int)) + be * sp.Matrix((ETA @ G).astype(int))
ok &= sp.simplify(Nsym.trace()) == 0
# commuting / anticommuting split: (1/2)(N - G N G) and (1/2)(N + G N G)
ok &= np.allclose(0.5 * (Cblk + G @ Cblk @ G), -(ETA @ G)) and np.allclose(0.5 * (Pblk - G @ Pblk @ G), 0.5 * I6)
check("x1-d1-1a", ok, "rotation-invariant forms dim 3 (|B|^2,|E|^2,B.E); rotation+duality invariant dim 1 = R*id "
      "(E6, positive); l_G-invariant dim 2 = span{eta, eta G}, traceless hence indefinite (SVD rank, tol 1e-10)")

# class test: conjugate G by an SO(3,3) element b outside l_G.  The tex says the statement 'could have failed' this way.
Bgen = 0.5 * adbiv(0, 3, ETAD) * 0.8 + 0.5 * adbiv(1, 5, ETAD) * 0.5 + 0.5 * adbiv(2, 4, ETAD) * (-0.3)
assert not np.allclose(comm(Bgen, G), 0)
b = expm(Bgen)
binv = np.linalg.inv(b)
Gp = b @ G @ binv
Rp = [b @ R @ binv for R in adL]
ok = np.allclose(Gp @ Gp, -I6) and np.allclose(ETA @ Gp.T @ ETA, Gp)          # still a binding root
ok &= not np.allclose(Gp.T @ Gp, I6)                                             # not orthogonal for delta
ok &= all(np.allclose(comm(R, Gp), 0) and np.allclose(R.T @ ETA + ETA @ R, 0) for R in Rp)   # compact part of l_G'
formsp = invariant_forms(Rp, [Gp])
Npos = binv.T @ binv
ok &= len(formsp) == 1 and in_span([Npos], formsp) and np.all(np.linalg.eigvalsh(Npos) > 0)
check("x1-1a-class", ok, "for a conjugated binding root G'=bGb^-1 (b in SO(3,3) outside l_G, G' not delta-orthogonal) the forms "
      "invariant under the compact part of l_G' and under G' are again 1-dim and positive definite (b^-T b^-1): the "
      "'it could have failed' argument for d1-1a does not hold; the uniqueness is inherited")

# =============================================================== d1-1b u(3), Lie(O(3,C)xU(1)), intersections
full = []
for a in range(6):
    for c in range(6):
        M = np.zeros((6, 6))
        M[a, c] = 1
        full.append(M)


def solve_lin(conds, extra_params=0):
    """nullspace of linear conditions on M (36 entries) plus extra scalar params."""
    rows = []
    for cond in conds:
        cols = [cond(M, np.zeros(extra_params)).reshape(-1) for M in full]
        for p in range(extra_params):
            ph = np.zeros(extra_params)
            ph[p] = 1
            cols.append(cond(np.zeros((6, 6)), ph).reshape(-1))
        rows.append(np.array(cols).T)
    return null_space(np.vstack(rows), rcond=1e-10)


u3 = solve_lin([lambda M, ph: comm(M, G), lambda M, ph: M + M.T])
lie7 = solve_lin([lambda M, ph: comm(M, G), lambda M, ph: M.T @ ETA + ETA @ M - ph[0] * ETA @ G], 1)
inter = solve_lin([lambda M, ph: comm(M, G), lambda M, ph: M.T @ ETA + ETA @ M - ph[0] * ETA @ G,
                   lambda M, ph: M + M.T], 1)
u3lg = solve_lin([lambda M, ph: comm(M, G), lambda M, ph: M.T @ ETA + ETA @ M, lambda M, ph: M + M.T])
ok = u3.shape[1] == 9 and lie7.shape[1] == 7 and inter.shape[1] == 4 and u3lg.shape[1] == 3
inter_mats = [inter[:36, c].reshape(6, 6) for c in range(inter.shape[1])]
ok &= in_span(adL + [G], inter_mats) and numrank(np.array([m.reshape(-1) for m in adL + [G]])) == 4
# explicit unitary moving a rest field to a null field at fixed E6
Uex = np.array([[1, 1j, 0], [1j, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
ok &= np.allclose(Uex.conj().T @ Uex, np.eye(3))
z0 = np.array([1, 0, 0], dtype=complex)
z1 = Uex @ z0
ok &= np.allclose(z1, np.array([1, 1j, 0]) / np.sqrt(2)) and abs(E6(z1) - 0.5) < 1e-15 and abs(E0(z0) - 0.5) < 1e-15 and E0(z1) < 1e-15
check("x1-d1-1b", ok, "dims (SVD, tol 1e-10): u(3)=9, Lie(O(3,C)xU(1)_G)=7, intersection=4=span{L_i,G}, u(3) cap l_G=3; "
      "explicit unitary sends (1,0,0) (E0=E6=1/2) to (1,i,0)/sqrt2 (E0=0)")

# =============================================================== d1-8b stabiliser of |q| in U(3)
u3basis = []
for a in range(3):
    for c in range(a + 1, 3):
        M = np.zeros((3, 3), dtype=complex)
        M[a, c], M[c, a] = 1, -1
        u3basis.append(M)
        M = np.zeros((3, 3), dtype=complex)
        M[a, c], M[c, a] = 1j, 1j
        u3basis.append(M)
for a in range(3):
    M = np.zeros((3, 3), dtype=complex)
    M[a, a] = 1j
    u3basis.append(M)
rows = []
for _ in range(60):
    z = RNG.normal(size=3) + 1j * RNG.normal(size=3)
    q = z @ z
    rows.append([np.real(np.conj(q) * (z @ (M + M.T) @ z)) for M in u3basis])
nsq = null_space(np.array(rows), rcond=1e-10)
stab = [sum(nsq[k, c] * u3basis[k] for k in range(9)) for c in range(nsq.shape[1])]
# expected: real antisymmetric (so(3)) + i*1
expected = [u3basis[0], u3basis[2], u3basis[4], 1j * np.eye(3)]
cv = lambda M: np.concatenate([M.real.reshape(-1), M.imag.reshape(-1)])
ok = len(stab) == 4 and numrank(np.array([cv(m) for m in stab + expected])) == 4
# group level: e^{i phi} O preserves |q|; a generic unitary does not; and U^T U = e^{i phi} for the stabiliser
O = expm(np.array([[0, .3, -.7], [-.3, 0, .2], [.7, -.2, 0]]))
Ust = np.exp(0.9j) * O
Hm = RNG.normal(size=(3, 3)) + 1j * RNG.normal(size=(3, 3))
Ugen = expm(1j * (Hm + Hm.conj().T) / 2)
zt = RNG.normal(size=3) + 1j * RNG.normal(size=3)
ok &= abs(abs((Ust @ zt) @ (Ust @ zt)) - abs(zt @ zt)) < 1e-12 and abs(abs((Ugen @ zt) @ (Ugen @ zt)) - abs(zt @ zt)) > 1e-3
ok &= np.allclose(Ust.T @ Ust, np.exp(1.8j) * np.eye(3))
# orbit dimension 5 = real structures up to phase: tangent of U -> U U^T at 1 is M + M^T (rank 6), minus the phase
img = [cv(M + M.T) for M in u3basis]
ok &= numrank(np.array(img)) == 6
check("x1-d1-8b", ok, "Lie stabiliser of |q| in u(3): dim 4 = so(3)+R i (60 random z, SVD tol 1e-10); e^{i phi}O preserves |q|, "
      "a generic unitary does not; symmetric unitaries U U^T form a 6-dim family, 5-dim up to phase")

# =============================================================== d1-2a/b orbits
def boost_textbook(E, Bt, v):
    """fields seen in a frame moving with velocity v (c=1), Jackson's finite transformation."""
    vv = np.dot(v, v)
    g = 1 / np.sqrt(1 - vv)
    Ep = g * (E + np.cross(v, Bt)) - g ** 2 / (g + 1) * v * np.dot(v, E)
    Bp = g * (Bt - np.cross(v, E)) - g ** 2 / (g + 1) * v * np.dot(v, Bt)
    return Ep, Bp


def z_to_EBt(z):
    return z.imag.copy(), -z.real.copy()


def EBt_to_z(E, Bt):
    return -Bt + 1j * E


def rest_frame(z, textbook=True):
    E, Bt = z_to_EBt(z)
    u = 0.5 * (E @ E + Bt @ Bt)
    St = np.cross(E, Bt)
    S = St if textbook else -St
    v = S / (u + E0(z))
    Ep, Bp = boost_textbook(E, Bt, v)
    return EBt_to_z(Ep, Bp)


def rotation_to_e1(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u2 = a - n * (a @ n)
    u2 /= np.linalg.norm(u2)
    u3 = np.cross(n, u2)
    return np.array([n, u2, u3])     # rows: R n = e1


maxres = 0.0
maxorth = 0.0
for _ in range(1000):
    z = RNG.normal(size=3) + 1j * RNG.normal(size=3)
    zr = rest_frame(z)
    # linearity check of the boost as a complex-orthogonal map (Lorentz = SO(3,C))
    E, Bt = z_to_EBt(z)
    St = np.cross(E, Bt)
    v = St / (0.5 * (E @ E + Bt @ Bt) + E0(z))
    Lam = np.array([EBt_to_z(*boost_textbook(*z_to_EBt(col), v)) for col in np.eye(3, dtype=complex)]).T
    Lam_i = np.array([EBt_to_z(*boost_textbook(*z_to_EBt(1j * col), v)) for col in np.eye(3, dtype=complex)]).T
    maxorth = max(maxorth, np.abs(Lam.T @ Lam - np.eye(3)).max(), np.abs(Lam_i - 1j * Lam).max())
    n = zr.real if np.linalg.norm(zr.real) > np.linalg.norm(zr.imag) else zr.imag
    R = rotation_to_e1(n)
    zz = R @ zr
    alpha = np.angle(zz[0])
    zf = np.exp(-1j * alpha) * zz
    target = np.array([np.sqrt(abs(z @ z)), 0, 0])
    maxres = max(maxres, np.abs(zf - target).max() / np.linalg.norm(z))
ok = maxres < 1e-10 and maxorth < 1e-10
note(f"1000 random z: textbook rest-frame boost + rotation + duality phase reaches (sqrt|q|,0,0), max rel. residual {maxres:.1e}; "
     f"boost is C-linear and complex orthogonal to {maxorth:.1e} (tol 1e-10)")


def tangent(z, with_G=True):
    vecs = [xof(np.cross(np.eye(3)[i], z)) for i in range(3)] + [xof(1j * np.cross(np.eye(3)[i], z)) for i in range(3)]
    if with_G:
        vecs.append(xof(1j * z))
    return np.array(vecs)


rk_ok = all(numrank(tangent(RNG.normal(size=3) + 1j * RNG.normal(size=3))) == 5 for _ in range(200))
# exact rank at rational points (sympy): generators L_i, K_i, G applied to X
def exact_rank(xvec, gens):
    xv = sp.Matrix(xvec)
    cols = [sp.nsimplify(sp.Matrix(M)) * xv for M in gens]
    return sp.Matrix.hstack(*cols).rank()


rk_ok &= all(exact_rank(xv, adL + adK + [G]) == 5 for xv in ([1, 2, 0, 0, 1, 3], [3, -1, 2, 1, 1, 0], [1, 0, 0, 0, 0, 0]))
check("x1-d1-2a", ok and rk_ok, "Lorentz x duality: every non-null z reaches (sqrt(2E0),0,0) via the textbook boost to the rest "
      "frame; orbit (tangent) rank 5 at 200 random and 3 exact rational points, so stabiliser dim 7-5=2")

ok = True
res = 0.0
for _ in range(300):
    a = RNG.normal(size=3)
    bb = np.cross(a, RNG.normal(size=3))
    bb *= np.linalg.norm(a) / np.linalg.norm(bb)
    z = a + 1j * bb
    ok &= abs(z @ z) < 1e-12 * np.vdot(z, z).real
    ok &= numrank(tangent(z, False)) == 4 and numrank(tangent(z, True)) == 4
    Rm = np.array([a / np.linalg.norm(a), bb / np.linalg.norm(bb), np.cross(a, bb) / (np.linalg.norm(a) * np.linalg.norm(bb))])
    zr = Rm @ z
    chi = -np.log(np.linalg.norm(a))
    X = expm(chi * adK[2]) @ xof(zr)
    res = max(res, np.abs(zof(X) - np.array([1, 1j, 0])).max())
ok &= res < 1e-10
ok &= exact_rank([1, 0, 0, 0, 1, 0], adL + adK) == 4 and exact_rank([1, 0, 0, 0, 1, 0], adL + adK + [G]) == 4
check("x1-d1-2b", ok, f"300 random null z: tangent rank 4 with and without G z; rotation + exp(chi K3), chi=-log|a|, gives (1,i,0) "
      f"(residual {res:.1e}); exact rank 4 at (1,i,0)")

# =============================================================== d1-3a Lagrange identity, H
b1, b2, b3, f1, f2, f3 = sp.symbols('b1 b2 b3 f1 f2 f3', real=True)
Bv = sp.Matrix([b1, b2, b3])
Ev = sp.Matrix([f1, f2, f3])
zs = Bv + sp.I * Ev
zbar = Bv - sp.I * Ev
q = (zs.T * zs)[0]
nz2 = (zbar.T * zs)[0]
zxz = zbar.cross(zs)
Sfr = Ev.cross(Bv)
ok = sp.expand(zxz + 2 * sp.I * Sfr) == sp.zeros(3, 1)
ok &= sp.expand(nz2 ** 2 - q * sp.conjugate(q) - (zxz.T * zxz.conjugate())[0]) == 0
ok &= sp.expand((nz2 / 2) ** 2 - (q * sp.conjugate(q)) / 4 - (Sfr.T * Sfr)[0]) == 0
H = zs * zbar.T
ok &= sp.expand(H.trace() - nz2) == 0
ok &= all(sp.expand(sp.im(sp.expand(H[j, k])) - sum(sp.LeviCivita(j, k, l) * Sfr[l] for l in range(3))) == 0
          for j in range(3) for k in range(3))
ok &= sp.expand((H * H.conjugate()).trace() - q * sp.conjugate(q)) == 0
ok &= sp.expand(H * H - H.trace() * H) == sp.zeros(3, 3)
mp.mp.dps = 40
for _ in range(50):
    zz = [mp.mpc(mp.mpf(RNG.normal()), mp.mpf(RNG.normal())) for _ in range(3)]
    lhs = (sum(abs(x) ** 2 for x in zz)) ** 2
    qq = sum(x * x for x in zz)
    cr = [mp.conj(zz[1]) * zz[2] - mp.conj(zz[2]) * zz[1], mp.conj(zz[2]) * zz[0] - mp.conj(zz[0]) * zz[2],
          mp.conj(zz[0]) * zz[1] - mp.conj(zz[1]) * zz[0]]
    ok &= abs(lhs - abs(qq) ** 2 - sum(abs(x) ** 2 for x in cr)) < mp.mpf(10) ** -30
check("x1-d1-3a", ok, "|z|^4=|q|^2+|zbar x z|^2, zbar x z=-2i E x B, E6^2=E0^2+|S|^2, tr H=2E6, Im H=eps.S, tr(H Hbar)=|q|^2, "
      "H^2=(tr H)H (symbolic; 50 random points at 40 digits)")

# =============================================================== d1-3b Rainich
Ex = sp.Matrix(sp.symbols('Ex Ey Ez', real=True))
Bt = sp.Matrix(sp.symbols('Bx By Bz', real=True))


def Tmn(E, B):
    u = ((E.T * E)[0] + (B.T * B)[0]) / 2
    S = E.cross(B)
    T = sp.zeros(4, 4)
    T[0, 0] = u
    for i in range(3):
        T[0, i + 1] = T[i + 1, 0] = S[i]
        for j in range(3):
            T[i + 1, j + 1] = (u if i == j else 0) - E[i] * E[j] - B[i] * B[j]
    return T


eta4 = sp.diag(1, -1, -1, -1)
T = Tmn(Ex, Bt)
Tmix = T * eta4
E0sq = (((Bt.T * Bt)[0] - (Ex.T * Ex)[0]) ** 2 + 4 * ((Ex.T * Bt)[0]) ** 2) / 4
ok = sp.expand(Tmix * Tmix - E0sq * sp.eye(4)) == sp.zeros(4, 4) and sp.expand(Tmix.trace()) == 0
th = sp.symbols('theta', real=True)
# duality z -> e^{i th} z  in textbook variables: Bt' = Bt cos + E sin, E' = E cos - Bt sin
Ed = Ex * sp.cos(th) - Bt * sp.sin(th)
Bd = Bt * sp.cos(th) + Ex * sp.sin(th)
ok &= sp.simplify(Tmn(Ed, Bd) - T) == sp.zeros(4, 4)
zd = -Bd + sp.I * Ed
zo = -Bt + sp.I * Ex
ok &= sp.simplify(sp.expand((zd.T * zd)[0] - sp.exp(2 * sp.I * th) * (zo.T * zo)[0]).rewrite(sp.cos)) == 0
fc = True
for _ in range(400):
    Ev_, Bv_ = RNG.normal(size=3), RNG.normal(size=3)
    if _ % 4 == 0:                      # include null fields
        Bv_ = np.cross(Ev_, RNG.normal(size=3))
        Bv_ *= np.linalg.norm(Ev_) / np.linalg.norm(Bv_)
    u = 0.5 * (Ev_ @ Ev_ + Bv_ @ Bv_)
    S = np.cross(Ev_, Bv_)
    Tn = np.zeros((4, 4))
    Tn[0, 0] = u
    Tn[0, 1:] = Tn[1:, 0] = S
    Tn[1:, 1:] = u * np.eye(3) - np.outer(Ev_, Ev_) - np.outer(Bv_, Bv_)
    w = RNG.normal(size=3)
    w = 0.95 * w / np.linalg.norm(w) * RNG.uniform()
    t = np.concatenate([[1.0], w]) / np.sqrt(1 - w @ w)
    Pt = Tn @ np.diag([1, -1, -1, -1.]) @ t            # P(t)^mu = T^mu_nu t^nu
    e0 = 0.5 * np.hypot(Bv_ @ Bv_ - Ev_ @ Ev_, 2 * Ev_ @ Bv_)
    fc &= Pt[0] > 0 and Pt[0] ** 2 - Pt[1:] @ Pt[1:] >= -1e-12 * u ** 2
    fc &= abs(Pt[0] ** 2 - Pt[1:] @ Pt[1:] - e0 ** 2) < 1e-10 * u ** 2 * (1 + t[0] ** 4)
check("x1-d1-3b", ok and fc, "T^mu_l T^l_nu = E0^2 delta, trace 0, T duality invariant while q -> e^{2i theta} q (symbolic); "
      "T.t future causal for 400 random (field, observer)")

# =============================================================== d1-4a/b/c minimum, two boost factors, rest-frame velocity
ok = True
worst = 0.0
for _ in range(40):
    z = RNG.normal(size=3) + 1j * RNG.normal(size=3)
    e0 = E0(z)
    mn = np.inf
    for _ in range(1500):
        n = RNG.normal(size=3)
        n /= np.linalg.norm(n)
        beta = np.tanh(RNG.uniform(0, 3))
        E, Bt_ = z_to_EBt(z)
        zp = EBt_to_z(*boost_textbook(E, Bt_, beta * n))
        mn = min(mn, E6(zp))
    zr = rest_frame(z)
    worst = max(worst, abs(E6(zr) - e0) / e0, np.linalg.norm(Sframe(zr)) / e0)
    ok &= mn >= e0 * (1 - 1e-12)
    ok &= abs(E6(z) - e0 / np.sqrt(1 - (np.linalg.norm(Sframe(z)) / E6(z)) ** 2)) < 1e-10 * E6(z)
ok &= worst < 1e-10
check("x1-d1-4a", ok, f"40 random fields x 1500 random boosts never go below E0; the rest-frame boost attains E0 with S=0 "
      f"(worst rel. {worst:.1e}, tol 1e-10); E6 = E0/sqrt(1-v_E^2)")

chi, a_, b_ = sp.symbols('chi a b', real=True)
ok = True
for idx, Kc in [(2, adK[2]), (1, adK[1])]:
    M = sp.nsimplify(sp.Matrix(Kc))
    Mexp = sp.simplify((chi * M).exp())
    X = Mexp * sp.Matrix([a_, 0, 0, b_, 0, 0])
    Bc, Ec = X[:3, 0], X[3:, 0]
    e6 = sp.simplify(((Bc.T * Bc)[0] + (Ec.T * Ec)[0]) / 2)
    Sv = sp.simplify(Ec.cross(Bc))
    ok &= sp.simplify(e6 - (a_ ** 2 + b_ ** 2) / 2 * sp.cosh(2 * chi)) == 0
    ok &= sp.simplify((Sv.T * Sv)[0] - ((a_ ** 2 + b_ ** 2) / 2 * sp.sinh(2 * chi)) ** 2) == 0
    qq = (Bc + sp.I * Ec).T * (Bc + sp.I * Ec)
    ok &= sp.simplify(sp.expand(qq[0]) - (a_ + sp.I * b_) ** 2) == 0       # q unchanged (Lorentz invariant)
M1 = sp.nsimplify(sp.Matrix(adK[0]))
ok &= sp.simplify((chi * M1).exp() * sp.Matrix([a_, 0, 0, b_, 0, 0]) - sp.Matrix([a_, 0, 0, b_, 0, 0])) == sp.zeros(6, 1)
# v_E = tanh 2chi = 2beta/(1+beta^2); |S|/(E6+E0) = tanh chi
bet = sp.tanh(chi)
ok &= sp.simplify((sp.tanh(2 * chi) - 2 * bet / (1 + bet ** 2)).rewrite(sp.exp)) == 0
ok &= sp.simplify((sp.sinh(2 * chi) / (sp.cosh(2 * chi) + 1) - sp.tanh(chi)).rewrite(sp.exp)) == 0
# riding with the energy velocity: frame moving with +tanh(2chi) e3 relative to the lab = exp(-2chi K3) (Section 12: v=-u)
M3 = sp.nsimplify(sp.Matrix(adK[2]))
Xlab = (chi * M3).exp() * sp.Matrix([a_, 0, 0, b_, 0, 0])
Xride = (-2 * chi * M3).exp() * Xlab
Bl, El = Xlab[:3, 0], Xlab[3:, 0]
Br, Er = Xride[:3, 0], Xride[3:, 0]
ok &= sp.simplify((Br.T * Br)[0] + (Er.T * Er)[0] - (Bl.T * Bl)[0] - (El.T * El)[0]) == 0
ok &= sp.simplify(Er.cross(Br) + El.cross(Bl)) == sp.zeros(3, 1)
check("x1-d1-4b", ok, "sympy Matrix.exp of the paper's K3, K2 on a e1 + b e4: E6 = E0 cosh 2chi, |S| = E0 sinh 2chi, q fixed; "
      "K1 fixes it; v_E = tanh 2chi = 2beta/(1+beta^2); rider at the energy velocity sees the same E6 and -S")

ok = True
wt = 0.0
wf = np.inf
agree = 0.0
for _ in range(200):
    z = RNG.normal(size=3) + 1j * RNG.normal(size=3)
    zt = rest_frame(z, textbook=True)
    zf_ = rest_frame(z, textbook=False)
    wt = max(wt, np.linalg.norm(Sframe(zt)) / E0(z))
    wf = min(wf, np.linalg.norm(Sframe(zf_)) / E0(z))
    E, Bt_ = z_to_EBt(z)
    v = np.cross(E, Bt_) / (E6(z) + E0(z))
    ok &= abs(np.linalg.norm(v) - np.tanh(0.5 * np.arctanh(np.linalg.norm(Sframe(z)) / E6(z)))) < 1e-10
    # the paper's generator for passing to a frame moving with u = beta n is exp(artanh(beta) * sum(-n_i K_i))
    beta = np.linalg.norm(v)
    nn = v / beta
    Xp = expm(np.arctanh(beta) * sum(-nn[i] * adK[i] for i in range(3))) @ xof(z)
    agree = max(agree, np.abs(zof(Xp) - zt).max() / np.linalg.norm(z))
ok &= wt < 1e-10 and wf > 1e-3 and agree < 1e-10
check("x1-d1-4c", ok, f"200 random fields: textbook boost with v = (E x B_t)/(u+E0) leaves |S|/E0 <= {wt:.1e}; with the frame "
      f"vector E x B it leaves >= {wf:.2f} E0; |v| = beta; the paper's exp(artanh(beta) sum(-n_i K_i)) equals the textbook "
      f"boost to {agree:.1e}")

# =============================================================== d1-5a SI identity and dimensions
c, e0s = sp.symbols('c epsilon0', positive=True)
mu0 = 1 / (e0s * c ** 2)
Bt3 = sp.Matrix(sp.symbols('bx by bz', real=True))
E3 = sp.Matrix(sp.symbols('ex ey ez', real=True))
u = e0s / 2 * ((E3.T * E3)[0] + c ** 2 * (Bt3.T * Bt3)[0])
St = E3.cross(Bt3) / mu0
g = St / c ** 2
zc = -c * Bt3 + sp.I * E3
qc = sp.expand((zc.T * zc)[0])
ok = sp.expand(qc - (c ** 2 * (Bt3.T * Bt3)[0] - (E3.T * E3)[0] - 2 * sp.I * c * (Bt3.T * E3)[0])) == 0
ok &= sp.expand(u - e0s / 2 * (zc.T * zc.conjugate())[0]) == 0
u0sq = e0s ** 2 / 4 * sp.expand(qc * sp.conjugate(qc))
ok &= sp.simplify(u ** 2 - c ** 2 * (g.T * g)[0] - u0sq) == 0
# dimensions as exponent vectors over (kg, m, s, A)
D = lambda kg, m, s, A: np.array([kg, m, s, A])
dE, dB, deps, dc = D(1, 1, -3, -1), D(1, 0, -2, -1), D(-1, -3, 4, 2), D(0, 1, -1, 0)
dmu = -deps - 2 * dc
du = deps + 2 * dE
dS = dE + dB - dmu
dg = dS - 2 * dc
dJm3 = D(1, -1, -2, 0)
ok &= np.array_equal(du, dJm3) and np.array_equal(dg + dc, dJm3) and np.array_equal(deps + 2 * dc + 2 * dB, dJm3)
ok &= np.array_equal(du - 2 * dc, D(1, -3, 0, 0)) and np.array_equal(2 * dc + 2 * dB, 2 * dE)
check("x1-d1-5a", ok, "SI: q_c = c^2|B_t|^2-|E|^2-2ic B_t.E, u=(eps0/2)|z_c|^2, u^2 = c^2|g|^2 + u0^2 (symbolic); "
      "[u]=[cg]=[u0]=J m^-3, [rho0]=kg m^-3, [q_c]=(V/m)^2 (exponent arithmetic)")

# =============================================================== d1-5b Schur, isotropic weight, and its removability
Cm = sp.Matrix(3, 3, sp.symbols('c0:9'))
so3 = [sp.Matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]]), sp.Matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
       sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
eqs = []
for R in so3:
    eqs += list(Cm * R - R * Cm)
sol = sp.solve(eqs, list(Cm), dict=True)[0]
Csol = Cm.subs(sol)
ok = Csol == Csol[0, 0] * sp.eye(3)
c1, c2, c3 = 1, 2, 3
Cd = sp.diag(c1, c2, c3)
ok &= any((Cd * R - R * Cd) != sp.zeros(3, 3) for R in so3)
al, be, ga = sp.symbols('alpha beta gamma', real=True)
Gc = sp.Matrix(sp.BlockMatrix([[sp.zeros(3, 3), -sp.eye(3) / c], [c * sp.eye(3), sp.zeros(3, 3)]]))
N = sp.Matrix(sp.BlockMatrix([[al * sp.eye(3), ga * sp.eye(3)], [ga * sp.eye(3), be * sp.eye(3)]]))
cond = sp.simplify(Gc.T * N + N * Gc)
solN = sp.solve(list(cond), [al, ga], dict=True)
ok &= len(solN) == 1 and sp.simplify(solN[0][al] - c ** 2 * be) == 0 and solN[0][ga] == 0
Phi = sp.diag(c, c, c, 1, 1, 1)
Gs = sp.Matrix(G.astype(int))
ok &= sp.simplify(Phi * Gc * Phi.inv() - Gs) == sp.zeros(6, 6)
ok &= sp.simplify(Phi.T * Phi - sp.diag(c ** 2, c ** 2, c ** 2, 1, 1, 1)) == sp.zeros(6, 6)
kap, ell = sp.symbols('kappa ell', positive=True)
ok &= sp.simplify(((c / ell) * Gs) ** 2 + c ** 2 / ell ** 2 * sp.eye(6)) == sp.zeros(6, 6)
check("x1-d1-5b", ok, "[C,so(3)]=0 forces C=c*1 (diag(1,2,3) fails); G_c-invariant rotation forms = multiples of c^2|B|^2+|E|^2; "
      "(kappa G)^2=-c^2/ell^2; and Phi=diag(c,c,c,1,1,1) conjugates G_c to G and turns c^2|B|^2+|E|^2 into |Phi X|^2: "
      "the weight c^2 is removed by measuring B as cB (unit convention)")

# =============================================================== d1-6a standing wave
zc_, t_, k_, a0 = sp.symbols('z t k a', positive=True)
om = c * k_
Exs = 2 * a0 * sp.cos(k_ * zc_) * sp.cos(om * t_)
cBy = 2 * a0 * sp.sin(k_ * zc_) * sp.sin(om * t_)
By = cBy / c
ok = sp.simplify(sp.diff(By, t_) + sp.diff(Exs, zc_)) == 0        # dB_y/dt = -(curl E)_y = -dE_x/dz
ok &= sp.simplify(sp.diff(Exs, t_) + c ** 2 * sp.diff(By, zc_)) == 0   # dE_x/dt = c^2 (curl B)_x = -c^2 dB_y/dz
uu = e0s / 2 * (Exs ** 2 + c ** 2 * By ** 2)
Sz = Exs * By / mu0
L = 2 * sp.pi / k_
Ucell = sp.simplify(sp.integrate(uu, (zc_, 0, L)))
Pcell = sp.simplify(sp.integrate(Sz / c ** 2, (zc_, 0, L)))
Tzz = sp.simplify(sp.integrate(uu, (zc_, 0, L)))            # T^zz = u for transverse E_x, B_y
Txx = sp.simplify(sp.integrate(uu - e0s * Exs ** 2, (zc_, 0, L)))
qcs = sp.simplify(c ** 2 * By ** 2 - Exs ** 2)
ok &= sp.simplify(Ucell - 2 * sp.pi * e0s * a0 ** 2 / k_) == 0 and Pcell == 0
ok &= sp.simplify(Tzz - Ucell) == 0 and sp.simplify(Txx + Ucell * sp.cos(2 * om * t_)) == 0
ok &= sp.simplify(sp.expand_trig(qcs + 2 * a0 ** 2 * (sp.cos(2 * k_ * zc_) + sp.cos(2 * om * t_)))) == 0
# each constituent: E_x = a cos(kz - wt), cB_y = a cos(kz - wt): U = |P| c per cell, q_c = 0
Ep_ = a0 * sp.cos(k_ * zc_ - om * t_)
Up = sp.integrate(e0s / 2 * (Ep_ ** 2 + Ep_ ** 2), (zc_, 0, L))
Pp = sp.integrate(Ep_ * (Ep_ / c) / mu0 / c ** 2, (zc_, 0, L))
ok &= sp.simplify(Up - Pp * c) == 0
# integrated rest energy over the cell vs s = sin^2(wt): mean of |s - cos^2 kz|, quadrature (mpmath, 30 digits)
mp.mp.dps = 30
ratio = lambda s: (2 / mp.pi) * mp.quad(lambda x: abs(s - mp.cos(x) ** 2), [0, mp.acos(mp.sqrt(s)) if 0 < s < 1 else 0, mp.pi / 2]) * 2
vals = [ratio(mp.mpf(s)) for s in ['0', '0.1', '0.25', '0.5', '0.75', '1']]
# ∫u0/U = (2 eps0 a^2 * (2pi/k) * mean|s-cos^2|) / (2 pi eps0 a^2/k) = 2 * mean|s - cos^2|
ok &= abs(vals[3] - 2 / mp.pi) < 1e-20 and abs(vals[0] - 1) < 1e-20 and abs(vals[5] - 1) < 1e-20
ok &= all(2 / mp.pi - 1e-20 <= v <= 1 + 1e-20 for v in vals)
check("x1-d1-6a", ok, "standing wave solves Maxwell; per cell U = 2 pi eps0 a^2/k, P = 0, int T^zz = U, int T^xx = -U cos 2wt, "
      "q_c = -2a^2(cos2kz+cos2wt); constituents U=|P|c; int u0/U = 2/pi at s=1/2 and 1 at s=0,1 (mpmath quad)")

# =============================================================== d1-6b pulses, M >= int E0
s_ = sp.symbols('s', real=True)
f = lambda x: (1 - x ** 2) ** 2


def pulse_integrals(tv):
    tv = sp.Rational(tv)
    lo, hi = -tv - 1, tv + 1
    zz = sp.symbols('zz', real=True)
    fm = sp.Piecewise((f(zz - tv), sp.And(zz - tv >= -1, zz - tv <= 1)), (0, True))
    fp = sp.Piecewise((f(zz + tv), sp.And(zz + tv >= -1, zz + tv <= 1)), (0, True))
    U = sp.integrate(fm ** 2 + fp ** 2, (zz, lo, hi))
    P = sp.integrate(fm ** 2 - fp ** 2, (zz, lo, hi))
    IE0 = sp.integrate(2 * fm * fp, (zz, lo, hi))
    return sp.nsimplify(U), sp.nsimplify(P), sp.nsimplify(IE0)


r0 = pulse_integrals(0)
rh = pulse_integrals(sp.Rational(1, 2))
r2 = pulse_integrals(2)
ok = r0 == (sp.Rational(512, 315), 0, sp.Rational(512, 315))
ok &= rh[0] == sp.Rational(512, 315) and rh[1] == 0 and rh[2] == sp.Rational(103, 315)
ok &= r2 == (sp.Rational(512, 315), 0, 0)
# fields: E_x = f(z-t)+f(z+t), B_ty = f(z-t)-f(z+t): u, S_z, q
fm_, fp_ = sp.symbols('fm fp', real=True)
Exp, Byp = fm_ + fp_, fm_ - fp_
ok &= sp.expand((Exp ** 2 + Byp ** 2) / 2 - (fm_ ** 2 + fp_ ** 2)) == 0 and sp.expand(Exp * Byp - (fm_ ** 2 - fp_ ** 2)) == 0
ok &= sp.expand(Byp ** 2 - Exp ** 2 + 4 * fm_ * fp_) == 0
# reverse triangle inequality for sums of Rainich densities, and its equality case
ineq = True
for _ in range(300):
    K = RNG.integers(2, 8)
    tot = np.zeros(4)
    sE0 = 0.0
    for _ in range(K):
        z = RNG.normal(size=3) + 1j * RNG.normal(size=3)
        if RNG.uniform() < 0.3:
            a = RNG.normal(size=3)
            bb = np.cross(a, RNG.normal(size=3))
            bb *= np.linalg.norm(a) / np.linalg.norm(bb)
            z = a + 1j * bb
        E, Bt_ = z_to_EBt(z)
        tot += np.concatenate([[E6(z)], np.cross(E, Bt_)])
        sE0 += E0(z)
    M = np.sqrt(max(tot[0] ** 2 - tot[1:] @ tot[1:], 0))
    ineq &= M >= sE0 * (1 - 1e-12) - 1e-14
zq = RNG.normal(size=3) + 1j * RNG.normal(size=3)
E, Bt_ = z_to_EBt(zq)
Pq = np.concatenate([[E6(zq)], np.cross(E, Bt_)])
tot = Pq * (1 + 2.5 + 0.3)
eqcase = abs(np.sqrt(tot[0] ** 2 - tot[1:] @ tot[1:]) - 3.8 * E0(zq)) < 1e-12
check("x1-d1-6b", ok and ineq and eqcase, "pulses (exact sympy piecewise): U=512/315, P=0 at t=0,1/2,2; int E0 = 512/315, 103/315, 0; "
      "q = -4 f(z-t) f(z+t); M >= sum E0 at 300 random sets incl. null points, equality for proportional densities")

# =============================================================== d1-6c blind to orthogonality
bvec = sp.Matrix(sp.symbols('p1 p2 p3', real=True))
TA = Tmn(sp.zeros(3, 1), bvec)
TB = Tmn(bvec * sp.sin(th), bvec * sp.cos(th))
ok = sp.simplify(TA - TB) == sp.zeros(4, 4)
ok &= sp.simplify((bvec * sp.sin(th)).dot(bvec * sp.cos(th)) - sp.sin(2 * th) / 2 * bvec.dot(bvec)) == 0
check("x1-d1-6c", ok, "(E,B_t)=(0,b) and (b sin th, b cos th) have identical T^{mu nu}; E.B_t = 0 vs (1/2) sin 2th |b|^2")

# =============================================================== d1-7 momentum reading
Ei = sp.symbols('E1:4', real=True)
pp = sp.symbols('p1:4', real=True)
Evec = sp.Matrix(Ei)
pvec = sp.Matrix(pp)
etaPP = (Evec.T * Evec)[0] / c ** 2 - (pvec.T * pvec)[0]
W = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
EW = (W.T * Evec)[0]
Eperp = Evec - EW * W
m4sq_c4 = EW ** 2 - (pvec.T * pvec)[0] * c ** 2
ok = sp.simplify(m4sq_c4 - (c ** 2 * etaPP - (Eperp.T * Eperp)[0])) == 0
m4e1 = Ei[0] ** 2 - (pvec.T * pvec)[0] * c ** 2
ok &= sp.simplify(m4e1 - (c ** 2 * etaPP - Ei[1] ** 2 - Ei[2] ** 2)) == 0
subs = {Ei[0]: 3, Ei[1]: 4, Ei[2]: 0, pp[0]: 5, pp[1]: 0, pp[2]: 0, c: 1}
ok &= etaPP.subs(subs) == 0 and m4e1.subs(subs) == -16 and sp.nsimplify(m4sq_c4.subs(subs)) == sp.Rational(-26, 3)
check("x1-d1-7", ok, "m4^2c^4 = mu^2c^4 - |E_perp|^2 (symmetric clock) and mu^2c^4 - E2^2 - E3^2 (clock e1); E=(3,4,0), p=(5,0,0): "
      "mu=0, m4^2=-16 with clock e1 (and -26/3 with the symmetric clock)")

# =============================================================== d1-8a T on the frame and as a kernel
PT = Smap
ok = np.allclose(PT, ETA) and np.allclose(PT @ G, -G @ PT)
ok &= all(np.allclose(PT @ K @ PT, -K) for K in adK) and all(np.allclose(PT @ L_ @ PT, L_) for L_ in adL)
xr = RNG.normal(size=6)
zr_ = zof(xr)
ok &= np.allclose(zof(PT @ xr), np.conj(zr_)) and abs(np.vdot(np.conj(zr_), zr_) - zr_ @ zr_) < 1e-12
# max over theta of eta(e^{theta G}X, e^{theta G}X) = |q|
thv = np.linspace(0, np.pi, 20001)
vals_ = [xof(np.exp(1j * t) * zr_) @ ETA @ xof(np.exp(1j * t) * zr_) for t in thv]
ok &= abs(max(vals_) - abs(zr_ @ zr_)) < 1e-6 * abs(zr_ @ zr_)
Gsym = sp.Matrix(G.astype(int))
Rth = sp.cos(th) * sp.eye(6) + sp.sin(th) * Gsym
Esym = sp.Matrix(ETA.astype(int))
ok &= sp.simplify(Rth.T * Esym * Rth - (sp.cos(2 * th) * Esym + sp.sin(2 * th) * Esym * Gsym)) == sp.zeros(6, 6)
# T as a Grassmann kernel (Def. 18.2): antisymmetric, commutes with rotations, anticommutes with boosts
L8 = [0.5 * (e8[j + 3] @ e8[k + 3] - e8[j] @ e8[k]) for (i, j, k) in CYC]
K8 = [0.5 * (e8[k] @ e8[j + 3] - e8[j] @ e8[k + 3]) for (i, j, k) in CYC]
ok &= np.array_equal(T8.T, -T8)
ok &= all(np.allclose(T8 @ X - X @ T8, 0) for X in L8) and all(np.allclose(T8 @ X + X @ T8, 0) for X in K8)
same = [(a, bq) for a in range(6) for bq in range(a + 1, 6) if ETAD[a] == ETAD[bq]]
mixed = [(a, bq) for a in range(6) for bq in range(a + 1, 6) if ETAD[a] != ETAD[bq]]
ok &= all(np.array_equal(T8 @ e8[a] @ e8[bq], e8[a] @ e8[bq] @ T8) for a, bq in same)
ok &= all(np.array_equal(T8 @ e8[a] @ e8[bq], -e8[a] @ e8[bq] @ T8) for a, bq in mixed)
ok &= all(np.allclose(e8[a] @ T8 @ np.linalg.inv(e8[a]), (1 if a < 3 else -1) * T8) for a in range(6))
check("x1-d1-8a", ok, "Ad_T|V = eta: fixes B, reverses E, anticommutes with G and K_i, commutes with L_i; q = h(Pz,z); "
      "max_theta eta(e^{theta G}X,e^{theta G}X) = |q|; as an 8x8 kernel T^T=-T commutes with L_i and anticommutes with K_i "
      "(so on spinors it IS an l_G-invariant kernel), sign flipped only by e4,e5,e6")

# =============================================================== d1-9 interior dynamics from the Lagrangian
tau = sp.symbols('tau', real=True)
m_ = sp.symbols('m', positive=True)
Xf = sp.Matrix([sp.Function(f'x{i}')(tau) for i in range(6)])
Xd = Xf.diff(tau)
Lag = m_ / 2 * ((Xd + kap * Gsym * Xf).T * Esym * (Xd + kap * Gsym * Xf))[0]
EL = sp.Matrix([sp.diff(sp.diff(Lag, Xd[i]), tau) - sp.diff(Lag, Xf[i]) for i in range(6)])
target = m_ * Esym * (Xf.diff(tau, 2) + kap ** 2 * Xf)
ok = sp.simplify(EL - target) == sp.zeros(6, 1)       # eta invertible: Xddot = -kappa^2 X, m an overall factor
Av = sp.Matrix(sp.symbols('A0:6', real=True))
Bvv = sp.Matrix(sp.symbols('C0:6', real=True))
Xsol = Av * sp.cos(kap * tau) + Bvv * sp.sin(kap * tau)
ok &= sp.simplify(Xsol.diff(tau, 2) + kap ** 2 * Xsol) == sp.zeros(6, 1)
Xsd = Xsol.diff(tau)
pmom = m_ * Esym * (Xsd + kap * Gsym * Xsol)
Lsol = m_ / 2 * ((Xsd + kap * Gsym * Xsol).T * Esym * (Xsd + kap * Gsym * Xsol))[0]
Enoether = sp.simplify((Xsd.T * pmom)[0] - Lsol)
Eform = sp.simplify(m_ / 2 * (Xsd.T * Esym * Xsd)[0] - m_ * kap ** 2 / 2 * ((Gsym * Xsol).T * Esym * (Gsym * Xsol))[0])
zA = Av[:3, 0] + sp.I * Av[3:, 0]
zB = Bvv[:3, 0] + sp.I * Bvv[3:, 0]
cm = (zA + sp.I * zB) / 2
cp = (zA - sp.I * zB) / 2
ok &= sp.simplify(Enoether - Eform) == 0
ok &= sp.simplify(sp.expand(Enoether - 2 * m_ * kap ** 2 * sp.re(sp.expand((cm.T * cp)[0])))) == 0
# zero momentum orbits: X = exp(-kappa G tau) X0  <=>  c_+ = 0  (B = -G A)
zero_mom = {Bvv[i]: (-(Gsym * Av))[i] for i in range(6)}
ok &= sp.simplify(pmom.subs(zero_mom)) == sp.zeros(6, 1) and sp.simplify(Enoether.subs(zero_mom)) == 0
Xz = Xsol.subs(zero_mom)
zz_ = Xz[:3, 0] + sp.I * Xz[3:, 0]
ok &= sp.simplify(sp.diff(sp.expand((zz_.T * zz_.conjugate())[0]), tau)) == 0
qz = sp.expand((zz_.T * zz_)[0])
ok &= sp.simplify(sp.diff(sp.expand(qz * sp.conjugate(qz)), tau)) == 0
# indefinite: two explicit orbits
Epos = Enoether.subs({**{Av[i]: [1, 0, 0, 0, 0, 0][i] for i in range(6)}, **{Bvv[i]: 0 for i in range(6)}})
Eneg = Enoether.subs({**{Av[i]: [0, 0, 0, 1, 0, 0][i] for i in range(6)}, **{Bvv[i]: 0 for i in range(6)}})
ok &= sp.simplify(Epos - m_ * kap ** 2 / 2) == 0 and sp.simplify(Eneg + m_ * kap ** 2 / 2) == 0
check("x1-d1-9", ok, "Euler-Lagrange of L=(m/2)eta(alpha_+X,alpha_+X) is m eta(Xddot+kappa^2 X): m an overall factor; Noether "
      "energy = (m/2)eta(Xd,Xd)-(m k^2/2)eta(GX,GX) = 2 m kappa^2 Re(c_-.c_+); zero on p=0 orbits, where E6 and |q| are constant; "
      "takes both signs (+-m kappa^2/2)")
note("with X a length and kappa = c/ell: E_N = m c^2 * 2Re(c_-.c_+)/ell^2, i.e. m c^2 times an unselected, indefinite "
     "dimensionless amplitude")

# =============================================================== d1-10 closure (Lie-Poisson)
I0 = 1.3
mm = 0.7


def wfun(p):
    pi = ETA @ p
    zp = zof(pi)
    return zp @ zp


def rhs(t, y, N):
    h, p = y[0], y[1:7]
    dp = (h / I0) * (G.T @ p)
    dh = -p @ G @ N @ p
    return np.concatenate([[dh], dp, [h / I0]])


def runLP(p0, N, h0, T=25.0):
    y0 = np.concatenate([[h0], p0, [0.0]])
    return solve_ivp(rhs, (0, T), y0, args=(N,), method='RK45', rtol=1e-11, atol=1e-13, dense_output=True,
                     t_eval=np.linspace(0, T, 400))


ok = True
# w(p) = C - iD with C = p^T eta^-1 p, D = p^T G eta^-1 p (eq. wdef)
for _ in range(20):
    p = RNG.normal(size=6)
    ok &= abs(wfun(p) - (p @ ETA @ p - 1j * (p @ G @ ETA @ p))) < 1e-12
maxcons = 0.0
hvar = 0.0
pend = 0.0
for trial in range(4):
    p0 = RNG.normal(size=6)
    aa, bb_ = RNG.normal(), RNG.normal()
    N = (aa * ETA + bb_ * G @ ETA) / mm
    ok &= np.allclose(N, N.T)
    sol = runLP(p0, N, h0=1.7)
    w0 = wfun(p0)
    R0 = abs((aa + 1j * bb_) * w0)
    phi0 = np.angle((aa + 1j * bb_) * w0)
    H0 = 1.7 ** 2 / (2 * I0) + 0.5 * p0 @ N @ p0
    hs = []
    for k in range(sol.y.shape[1]):
        h, p, theta = sol.y[0, k], sol.y[1:7, k], sol.y[7, k]
        pi = ETA @ p
        zp = zof(pi)
        H = h ** 2 / (2 * I0) + 0.5 * p @ N @ p
        Hred = h ** 2 / (2 * I0) + (abs(aa + 1j * bb_) * E0(zof(ETA @ p0)) / mm) * np.cos(2 * theta + phi0)
        maxcons = max(maxcons, abs(p @ p - 2 * E6(zp)) / (p0 @ p0), abs(p @ p - p0 @ p0) / (p0 @ p0),
                      abs(abs(wfun(p)) - abs(w0)) / (p0 @ p0), np.abs(Sframe(zp) - Sframe(zof(ETA @ p0))).max() / (p0 @ p0),
                      abs(H - H0) / abs(H0), abs(Hred - H0) / abs(H0))
        dh = -p @ G @ N @ p
        pend = max(pend, abs(dh - (R0 / mm) * np.sin(2 * theta + phi0)) / max(R0 / mm, 1e-300))
        hs.append(h)
    hvar = max(hvar, np.ptp(hs))
ok &= maxcons < 1e-8 and pend < 1e-8 and hvar > 1e-3
note(f"Lie-Poisson (RK45, rtol 1e-11): |p|^2=2E6(pi), |w|, S(pi), H and the reduced H with amplitude |a+ib|E0(pi0)/m "
     f"conserved/matched to {maxcons:.1e}; hdot = (R0/m) sin(2theta+phi0) to {pend:.1e}; h varies by up to {hvar:.2f}")
# null data or N = 1/m: rate constant
an = RNG.normal(size=3)
bn = np.cross(an, RNG.normal(size=3))
bn *= np.linalg.norm(an) / np.linalg.norm(bn)
p_null = ETA @ xof(an + 1j * bn)
ok &= abs(wfun(p_null)) < 1e-12
sol = runLP(p_null, (0.4 * ETA + 0.9 * G @ ETA) / mm, h0=1.1)
ok &= np.ptp(sol.y[0]) < 1e-10
sol = runLP(RNG.normal(size=6), I6 / mm, h0=1.1)
ok &= np.ptp(sol.y[0]) < 1e-10
check("x1-d1-10", ok, "closure: Casimir |p|^2 = 2E6(pi); E0(pi)=|w|/2 and S(pi) conserved for every N; Lorentz-class reduced "
      "H = h^2/2I0 + (|a+ib|E0(pi0)/m) cos(2theta+phi0); rate constant for null data and for N = 1/m")

# counterexample to the unqualified 'constant iff E0(pi0)=0': an exact equilibrium of the pendulum
p_eq = np.zeros(6)
p_eq[0] = 1.0                      # pi = eta p = e_1: pure magnetic, w = 1, E0 = 1/2
for (aa, bb_) in [(1.0, 0.0), (-1.0, 0.0)]:      # phi0 = 0 (unstable top) and phi0 = pi (stable bottom)
    N = (aa * ETA + bb_ * G @ ETA) / mm
    vf = rhs(0, np.concatenate([[0.0], p_eq, [0.0]]), N)
    exact_fixed = np.allclose(vf[:7], 0, atol=0)
    sol = runLP(p_eq, N, h0=0.0)
    ok_c = exact_fixed and E0(zof(ETA @ p_eq)) == 0.5 and np.ptp(sol.y[0]) == 0.0
    check(f"x1-10-iff-{'unstable' if aa > 0 else 'stable'}", ok_c,
          f"(a,b)=({aa:+.0f},0), pi0=e1 (E0(pi0)=1/2), h(0)=0: the vector field vanishes exactly, h stays 0, so the rate is "
          f"constant although E0(pi0) != 0; the correct statement is Corollary 17.14's 'constant and nonzero iff R0=0 and h(0)!=0'")
# and (a,b) = (0,0): constant for all data
sol = runLP(RNG.normal(size=6), 0 * ETA, h0=0.8)
check("x1-10-ab0", np.ptp(sol.y[0]) < 1e-14, "(a,b)=(0,0) (N=0): rate constant for all data, so the iff needs (a,b)!=(0,0)")

# =============================================================== d1-11 b-field placement
gens15 = [0.5 * adbiv(A, B, ETAD) for A in range(6) for B in range(A + 1, 6)]
rows = []
for Mg in gens15:
    Sm = Mg.T @ ETA @ G + ETA @ G @ Mg
    rows.append(0.5 * (Sm + Sm.T))
Amat = np.array([r.reshape(-1) for r in rows]).T
ns = null_space(Amat, rcond=1e-10)
pres = [sum(ns[k, cc] * gens15[k] for k in range(15)) for cc in range(ns.shape[1])]
ok = len(pres) == 6 and in_span(adL + adK, pres) and all(np.allclose(comm(M, G), 0) for M in pres)
# tower n=3: eta3, G3, rot(i,j) for i<j in W={7,13,14,15} (1-based), from the abstract rule
ETAD3 = ETAD * 3
G3 = np.kron(np.eye(3), G)
ETA3 = np.diag(ETAD3).astype(float)


def rot(i, j):
    i0, j0 = i - 1, j - 1
    return adbiv(i0 + 3, j0 + 3, ETAD3) - adbiv(i0, j0, ETAD3)


Wd = [7, 13, 14, 15]
so4 = [rot(i, j) for ii, i in enumerate(Wd) for j in Wd[ii + 1:]]
ok &= numrank(np.array([m.reshape(-1) for m in so4])) == 6
ok &= all(np.allclose(comm(M, G3), 0) and np.allclose(M, -M.T) and np.allclose(M.T @ ETA3 + ETA3 @ M, 0)
          and np.allclose(M.T @ ETA3 @ G3 + ETA3 @ G3 @ M, 0) for M in so4)
# closed Lie algebra, compact semisimple (Killing form negative definite) of dim 6 -> su(2)+su(2)
basis = np.array([m.reshape(-1) for m in so4]).T
coef = lambda X: np.linalg.lstsq(basis, X.reshape(-1), rcond=None)[0]
closed = all(np.allclose(basis @ coef(comm(X, Y)), comm(X, Y).reshape(-1)) for X in so4 for Y in so4)
adm = [np.array([coef(comm(X, Y)) for Y in so4]).T for X in so4]
Kill = np.array([[np.trace(A1 @ A2) for A2 in adm] for A1 in adm])
ok &= closed and np.all(np.linalg.eigvalsh(0.5 * (Kill + Kill.T)) < -1e-9)
# the example
Z0 = np.concatenate([np.zeros(6), xof(np.array([1, 1j, 0])), xof(np.array([1, 0, 0], dtype=complex))])
Z1 = expm((np.pi / 4) * 0.5 * rot(7, 13)) @ Z0
zs1 = [zof(Z1[6 * k:6 * k + 6]) for k in range(3)]
zs0 = [zof(Z0[6 * k:6 * k + 6]) for k in range(3)]
ok &= np.allclose(zs1[1], [0, 1j, 0]) and np.allclose(zs1[2], [np.sqrt(2), 0, 0]) and np.allclose(zs1[0], 0)
tE6 = lambda zl: sum(E6(z) for z in zl)
tq = lambda zl: sum(z @ z for z in zl)
sE0 = lambda zl: sum(E0(z) for z in zl)
ok &= abs(tE6(zs0) - 1.5) < 1e-14 and abs(tE6(zs1) - 1.5) < 1e-12 and abs(tq(zs0) - 1) < 1e-14 and abs(tq(zs1) - 1) < 1e-12
ok &= abs(sE0(zs0) - 0.5) < 1e-14 and abs(sE0(zs1) - 1.5) < 1e-12
check("x1-d1-11", ok, "A in so(3,3) preserves E.B iff A in l_G (dim 6, SVD); so(4)_Delta: rank 6, commutes with G3, antisymmetric, "
      "preserves eta3 and eta3 G3, closed, Killing form negative definite; example: totals E6=3/2, q3=1 fixed, sum E0 1/2 -> 3/2")

# the 'cannot create total rest energy' reading: the Lorentz length of the summed density changes
def summed_density(zl):
    tot = np.zeros(4)
    for z in zl:
        E, Bt_ = z_to_EBt(z)
        tot += np.concatenate([[E6(z)], np.cross(E, Bt_)])
    return tot


d0, d1_ = summed_density(zs0), summed_density(zs1)
M0 = np.sqrt(d0[0] ** 2 - d0[1:] @ d0[1:])
M1 = np.sqrt(d1_[0] ** 2 - d1_[1:] @ d1_[1:])
Ldiag3 = [np.kron(np.eye(3), L_) for L_ in adL]
nc = any(not np.allclose(comm(rot(7, 13), Ld), 0) for Ld in Ldiag3)
ok = abs(M0 - np.sqrt(5) / 2) < 1e-14 and abs(M1 - 1.5) < 1e-12 and np.allclose(d0[1:], [0, 0, 1]) and np.allclose(d1_[1:], 0) and nc
check("x1-11-total", ok, "in the same example the summed energy-momentum density (sum E6, sum S_t) goes from (3/2;0,0,1) to "
      "(3/2;0,0,0): its Lorentz length goes sqrt5/2 -> 3/2, and rot(7,13) does not commute with the diagonal rotations; "
      "'total rest energy is not created' holds only for the so(9,C) invariant |q3|/2")

# a compact element of Cent(G3) = so(9,C) need not be real antisymmetric (proof step of Prop. d1:prop:bfield(2))
def bst(i, j):
    i0, j0 = i - 1, j - 1
    return adbiv(j0, i0 + 3, ETAD3) - adbiv(i0, j0 + 3, ETAD3)


bconj = expm(0.6 * bst(7, 8))
Rc = bconj @ rot(7, 13) @ np.linalg.inv(bconj)
ev = np.linalg.eigvals(Rc)
ok = np.allclose(comm(bst(7, 8), G3), 0) and np.allclose(comm(Rc, G3), 0) and np.allclose(ev.real, 0, atol=1e-10)
ok &= np.allclose(expm(np.pi * rot(7, 13)), np.eye(18), atol=1e-9) and np.allclose(expm(np.pi * Rc), np.eye(18), atol=1e-8)
ok &= not np.allclose(Rc, -Rc.T)
Zt = RNG.normal(size=18)
Zt2 = expm(0.7 * Rc) @ Zt
ok &= abs(Zt2 @ Zt2 - Zt @ Zt) > 1e-6
check("x1-11-compact", ok, "b rot(7,13) b^-1 (b = exp(0.6 bst(7,8)) in Cent(G3)) commutes with G3 and generates a compact group "
      "(imaginary spectrum) but is not antisymmetric and does not preserve |Z|^2: 'compact elements of so(9,C) are real "
      "antisymmetric' is false in general; the proof must use that so(4)_Delta consists of paired rotations (Thm 17.19)")

# =============================================================== scope of U(3): pointwise form only
# (1) a U(3) element maps a Maxwell plane wave (Example 12.11) to a non-solution of dz/dt = -i curl z
x1s, x2s, x3s, ts, ks = sp.symbols('x1 x2 x3 t k', real=True)
phase = sp.exp(sp.I * ks * (x3s - ts))
zwave = sp.Matrix([1, sp.I, 0]) * phase
Usym = sp.Matrix([[1, sp.I, 0], [sp.I, 1, 0], [0, 0, sp.sqrt(2)]]) / sp.sqrt(2)


def curlv(F):
    return sp.Matrix([sp.diff(F[2], x2s) - sp.diff(F[1], x3s), sp.diff(F[0], x3s) - sp.diff(F[2], x1s),
                      sp.diff(F[1], x1s) - sp.diff(F[0], x2s)])


maxwell = lambda F: sp.simplify(sp.diff(F, ts) + sp.I * curlv(F))
ok = maxwell(zwave) == sp.zeros(3, 1) and sp.simplify(Usym.H * Usym - sp.eye(3)) == sp.zeros(3, 3)
ok &= maxwell(Usym * zwave) != sp.zeros(3, 1)
# (2) the split-form Lagrangian (eta) is not U(3) invariant: G = i*1 lies in u(3) and G^T eta G = -eta
ok &= np.allclose(G.T @ ETA @ G, -ETA)
# its U(3)-symmetries are u(3) cap {eta-preserving} = so(3) (dim 3, computed above as u3lg)
ok &= u3lg.shape[1] == 3
# (3) the duality-class closure Hamiltonian |p|^2/2m is U(3) invariant (U commutes with G, hence with the rotor action)
Mu = u3[:36, 0].reshape(6, 6)
ok &= np.allclose(Mu + Mu.T, 0) and np.allclose(comm(Mu, G), 0)
check("x1-u3-scope", ok, "U(3) is the symmetry of the pointwise form E6 (and of the duality-class closure H with N=1/m), not of "
      "Maxwell's equations on the frame (a unitary maps the wave (1,i,0)e^{ik(x3-t)} to a non-solution) nor of the split-form "
      "Lagrangian (G = i*1 reverses eta; U(3) cap O(eta) is SO(3))")

# =============================================================== summary
if FAILS:
    print("FAILED:", ", ".join(FAILS))
    sys.exit(1)
print("ALL x1 CHECKS PASSED")
sys.exit(0)
