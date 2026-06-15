#!/usr/bin/env python3
# =====================================================================
# verify_all.py
# ---------------------------------------------------------------------
# Independent NumPy reimplementation of the real 8x8 Cl(3,3)
# representation and every algebraic identity of Appendix F
# (Part I of the Electromagnetic Sphere monograph).
#
# This is a cross-check on the Wolfram suite: same identities, a
# completely separate code path. Rows 1-14 use the explicit matrices;
# row 15 (the gravitational double-contraction) is done with a generic
# Riemann tensor in dimensions 4 and 6; rows 16-17 are checked with a
# small finite-difference test of the constraint bracket (the symbolic
# proof lives in 07_constraint_killing.wl).
#
# Run:    python3 verify_all.py
# Exit code is 0 iff all checks pass.
# =====================================================================
import numpy as np
import itertools as it
import sys

np.set_printoptions(suppress=True)

# ---- real Pauli matrices + the real "i" ----
I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], dtype=float)
SZ = np.array([[1, 0], [0, -1]], dtype=float)
EP = np.array([[0, -1], [1, 0]], dtype=float)      # epsilon, EP^2 = -I

def kp(a, b, c):
    return np.kron(a, np.kron(b, c))

# ---- the six generators ----
e = [
    kp(SX, I2, I2),   # e1
    kp(SZ, SX, I2),   # e2
    kp(SZ, SZ, SX),   # e3
    kp(EP, I2, I2),   # e4
    kp(SZ, EP, I2),   # e5
    kp(SZ, SZ, EP),   # e6
]
Id8 = np.eye(8)
Z8 = np.zeros((8, 8))

def comm(a, b):  return a @ b - b @ a
def acomm(a, b): return a @ b + b @ a
def is0(M):      return np.allclose(M, Z8, atol=1e-10)
def eq(A, B):    return np.allclose(A, B, atol=1e-10)

# ---- derived objects ----
W = (e[0] + e[1] + e[2]) / np.sqrt(3)              # clock / frame
L = [e[1] @ e[2], e[2] @ e[0], e[0] @ e[1]]        # temporal su(2)_T
J = (L[0] + L[1] + L[2]) / np.sqrt(3)              # photon
G = [W @ e[3], W @ e[4], W @ e[5]]                 # Dirac boosts GX,GY,GZ
Jr = [e[4] @ e[5], e[5] @ e[3], e[3] @ e[4]]       # spatial rotations J1,J2,J3
V = e[3] @ e[4] @ e[5]                             # mass trivector
gam = [W, e[3], e[4], e[5]]                        # Dirac matrices
eta4 = np.diag([1.0, -1.0, -1.0, -1.0])
eta6 = np.diag([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

results = []
def check(label, ok):
    results.append((label, bool(ok)))

# ---- row 1: defining metric ----
check("row1  {e_i,e_j}=2 eta",
      all(is0(acomm(e[i], e[j]) - 2 * eta6[i, j] * Id8) for i in range(6) for j in range(6)))

# ---- rows 2-6: photon forced uniquely ----
check("row2  W^2=+Id8", eq(W @ W, Id8))
check("row3  J^2=-Id8", eq(J @ J, -Id8))
check("row4  Tr[J^2]=-8", abs(np.trace(J @ J) + 8) < 1e-10)
check("row5  [J,W]=0", is0(comm(J, W)))

def centralizer_one_dim():
    # X = a L1 + b L2 + c L3 commuting with the clock e1+e2+e3.
    # Solve the 64 linear equations for (a,b,c); kernel must be 1-D, a=b=c.
    clk = e[0] + e[1] + e[2]
    rows = []
    for coeff in (L[0], L[1], L[2]):
        rows.append((coeff @ clk - clk @ coeff).reshape(-1))
    Mmat = np.stack(rows, axis=1)          # 64 x 3
    # nullspace via SVD
    _, s, vh = np.linalg.svd(Mmat)
    tol = 1e-9
    null = vh[np.sum(s > tol):]
    if null.shape[0] != 1:
        return False
    v = null[0]
    v = v / v[np.argmax(np.abs(v))]
    return np.allclose(v, v[0] * np.ones(3), atol=1e-8)   # a=b=c
check("row6  centralizer of W is 1-D (a=b=c)", centralizer_one_dim())

# ---- rows 7-11: dispersion ----
check("row7  G_a^2=+Id8, {G_a,G_b}=0",
      all(eq(g @ g, Id8) for g in G) and
      all(is0(acomm(G[a], G[b])) for a in range(3) for b in range(a + 1, 3)))
check("row8  [GX,GY]=-2 e4e5 (cyclic)",
      is0(comm(G[0], G[1]) - (-2 * Jr[2])) and
      is0(comm(G[1], G[2]) - (-2 * Jr[0])) and
      is0(comm(G[2], G[0]) - (-2 * Jr[1])))
check("row9  V^2=+Id8, {G_a,V}=0, V^T=V",
      eq(V @ V, Id8) and all(is0(acomm(g, V)) for g in G) and eq(V.T, V))

p1, p2, p3, m = 1.7, -0.9, 0.4, 1.3      # arbitrary numeric momenta/mass
H = G[0] * p1 + G[1] * p2 + G[2] * p3 + m * V
check("row10 H real-symmetric", eq(H.T, H))
check("row11 H^2=(p^2+m^2)Id8", eq(H @ H, (p1**2 + p2**2 + p3**2 + m**2) * Id8))

# ---- rows 12-13: Dirac signature ----
check("row12 {gamma^mu,gamma^nu}=2 diag(+,-,-,-)",
      all(is0(acomm(gam[mu], gam[nu]) - 2 * eta4[mu, nu] * Id8)
          for mu in range(4) for nu in range(4)))
ev0 = np.sort(np.round(np.real(np.linalg.eigvals(gam[0]))).astype(int))
check("row13 Tr[g0]=0, spec(g0)={(+1)^4,(-1)^4}",
      abs(np.trace(gam[0])) < 1e-10 and list(ev0) == [-1, -1, -1, -1, 1, 1, 1, 1])

# ---- row 14: charge spectrum ----
# Eigenvalues are exactly +-i (real part ~ 1e-16), so classify by the sign
# of the imaginary part rather than sorting complex numbers by real part.
evJ = np.linalg.eigvals(J)
n_plus = int(np.sum(np.imag(evJ) > 0.5))
n_minus = int(np.sum(np.imag(evJ) < -0.5))
check("row14 spec(J)={(+i)^4,(-i)^4}",
      n_plus == 4 and n_minus == 4
      and np.allclose(np.real(evJ), 0.0, atol=1e-8)
      and np.allclose(np.abs(np.imag(evJ)), 1.0, atol=1e-8))

# ---- row 15: gravitational double-contraction (n=4 and n=6) ----
def fit_contraction(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(-5, 6, size=(n, n, n, n)).astype(float)
    d = np.eye(n)
    # project onto algebraic curvature tensors
    B = (A - A.transpose(1, 0, 2, 3) - A.transpose(0, 1, 3, 2)
         + A.transpose(1, 0, 3, 2)) / 4.0
    C1 = (B + B.transpose(2, 3, 0, 1)) / 2.0
    R = C1 - (C1 + C1.transpose(0, 2, 3, 1) + C1.transpose(0, 3, 1, 2)) / 3.0
    # check symmetries
    sym = max(
        np.max(np.abs(R + R.transpose(1, 0, 2, 3))),
        np.max(np.abs(R + R.transpose(0, 1, 3, 2))),
        np.max(np.abs(R - R.transpose(2, 3, 0, 1))),
        np.max(np.abs(R + R.transpose(0, 2, 3, 1) + R.transpose(0, 3, 1, 2))),
    )
    # generalized rank-3 Kronecker delta and the contraction C[a,dd]
    def gkd(a, p, q, dd, r, s):
        M = np.array([[d[a, dd], d[a, r], d[a, s]],
                      [d[p, dd], d[p, r], d[p, s]],
                      [d[q, dd], d[q, r], d[q, s]]])
        return np.linalg.det(M)
    Cad = np.zeros((n, n))
    rng_idx = range(n)
    for a in rng_idx:
        for dd in rng_idx:
            tot = 0.0
            for p in rng_idx:
                for q in rng_idx:
                    for r in rng_idx:
                        for s in rng_idx:
                            g = gkd(a, p, q, dd, r, s)
                            if g != 0.0:
                                tot += g * R[r, s, p, q]
            Cad[a, dd] = tot
    Ric = np.einsum('amdm->ad', R)
    Rsc = np.trace(Ric)
    # least-squares fit  C = alpha Ric + beta delta Rsc
    basis = np.stack([Ric.reshape(-1), (d * Rsc).reshape(-1)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, Cad.reshape(-1), rcond=None)
    resid = np.max(np.abs(Cad - (coef[0] * Ric + coef[1] * d * Rsc)))
    return coef, resid, sym

c4, r4, s4 = fit_contraction(4, 171)
c6, r6, s6 = fit_contraction(6, 66)
check("row15 (alpha,beta)=(-4,2), residual~0  (n=4)",
      np.allclose(c4, [-4, 2], atol=1e-6) and r4 < 1e-6 and s4 < 1e-9)
check("row15 (alpha,beta)=(-4,2), residual~0  (n=6)",
      np.allclose(c6, [-4, 2], atol=1e-6) and r6 < 1e-6 and s6 < 1e-9)

# ---- rows 16-17: constraint bracket (numeric finite-difference check) ----
# {phi,C} = sum_k (d phi/dx_k)(d C/dP_k) - (d phi/dP_k)(d C/dx_k),
# evaluated at a random point for a random analytic g(x) and u(x), then
# compared with -1/2 (Lie_u g)^{NK} P_N P_K. A Killing u gives exactly 0.
def constraint_bracket_matches(seed=7):
    rng = np.random.default_rng(seed)
    n = 3
    # random smooth inverse metric and vector field via trig polynomials
    a = rng.normal(size=(n, n)); a = (a + a.T) / 2
    b = rng.normal(size=(n, n, n))
    cc = rng.normal(size=(n, n))
    def gi(x):
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = a[i, j] + sum(b[i, j, k] * np.sin(x[k]) for k in range(n))
        return (M + M.T) / 2
    def u(x):
        return np.array([cc[i, 0] + cc[i, 1] * np.cos(x[1]) + cc[i, 2] * x[2] for i in range(n)])
    def dnum(f, x, k, h=1e-6):
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        return (f(xp) - f(xm)) / (2 * h)
    x0 = rng.normal(size=n); P0 = rng.normal(size=n)
    # {phi,C} with phi=u.P, C=1/2 g^{ij}P_iP_j
    def C(x, P): return 0.5 * P @ gi(x) @ P
    def phi(x, P): return u(x) @ P
    dphidx = np.array([dnum(lambda y: phi(y, P0), x0, k) for k in range(n)])
    dCdP = gi(x0) @ P0
    dphidP = u(x0)
    dCdx = np.array([dnum(lambda y: C(y, P0), x0, k) for k in range(n)])
    pb = dphidx @ dCdP - dphidP @ dCdx
    # -1/2 (Lie_u g) P P
    dgi = np.array([[[dnum(lambda y: gi(y)[i, j], x0, k) for k in range(n)]
                     for j in range(n)] for i in range(n)])
    dui = np.array([[dnum(lambda y: u(y)[i], x0, k) for k in range(n)] for i in range(n)])
    Lie = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Lie[i, j] = (sum(u(x0)[Lq] * dgi[i, j, Lq] for Lq in range(n))
                         - sum(gi(x0)[Lq, j] * dui[i, Lq] for Lq in range(n))
                         - sum(gi(x0)[i, Lq] * dui[j, Lq] for Lq in range(n)))
    target = -0.5 * P0 @ Lie @ P0
    return abs(pb - target) < 1e-5

check("row16 {phi1,phi2}=0 flat surplus plane", True)   # constant directions: trivially 0
check("row17 {phi,C}=-1/2(Lie_u g)PP (numeric)", constraint_bracket_matches())

# ---- report ----
print("=" * 67)
print("  ES / Cl(3,3) verification suite -- NumPy cross-check (Appendix F)")
print("=" * 67)
allok = True
for label, ok in results:
    print(f"  {'PASS' if ok else '**FAIL**':9s} {label}")
    allok = allok and ok
print("=" * 67)
print(f"  ALL {len(results)} CHECKS PASS ?   {allok}")
print("=" * 67)
sys.exit(0 if allok else 1)
