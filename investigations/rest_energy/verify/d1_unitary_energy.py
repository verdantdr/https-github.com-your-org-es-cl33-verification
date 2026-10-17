#!/usr/bin/env python3
"""Track d1 -- the undivided ("unitary") six-dimensional energy and its internal differentiation.

Verification script for investigations/rest_energy/sections/d1_unitary_energy.tex.
Conventions follow the unified paper of 17 October 2026 (tex/s02, s03, s12, s14, s16, s17):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1), realisation of tex/s02_arena.tex eq. (realisation);
  X = sum_i (B_i e_i + E_i e_{i+3}),  z = B + iE in C^3,  G e_i = e_{i+3}, G e_{i+3} = -e_i (G acts as i);
  q(z) = z.z, E6 = |z|^2/2, E0 = |q|/2, frame Poynting vector S = E x B, textbook S_t = E x B_t = -E x B.
Exact statements use sympy (integer/rational arithmetic); numerical ones state their tolerances.
Run:  python3 d1_unitary_energy.py     (exit 0 and one PASS line per claim if everything holds)
"""
import sys
import itertools
import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

FAILED = []


def report(cid, ok, text):
    if ok:
        print(f"PASS {cid}: {text}")
    else:
        print(f"FAIL {cid}: {text}")
        FAILED.append(cid)


def info(text):
    print(f"    . {text}")


def is_zero(M):
    M = sp.Matrix(M)
    return all(sp.simplify(x) == 0 for x in M)


# ----------------------------------------------------------------------------- Clifford realisation
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def k3(a, b, c):
    return np.kron(np.kron(a, b), c)


e = [k3(sx, I2, I2), k3(sz, sx, I2), k3(sz, sz, sx),
     k3(ep, I2, I2), k3(sz, ep, I2), k3(sz, sz, ep)]
eta_d = [1, 1, 1, -1, -1, -1]
ETA = sp.diag(*eta_d)
I8 = np.eye(8, dtype=np.int64)
ok = all(np.array_equal(e[a] @ e[b] + e[b] @ e[a], 2 * eta_d[a] * I8 * (a == b))
         for a in range(6) for b in range(6))
assert ok, "Clifford relations fail"


def vec_coeffs(Y):
    """Coefficients of an 8x8 matrix Y along e_1..e_6 (Y assumed a vector), exact rationals."""
    c = [sp.Rational(int(np.trace(Y @ e[b])), 8 * eta_d[b]) for b in range(6)]
    rec = sum((float(c[b]) * e[b] for b in range(6)), np.zeros((8, 8)))
    assert np.allclose(rec, Y), "not a vector"
    return c


def ad6(Bv, half=False):
    """6x6 matrix of v -> [Bv, v] on vectors (columns = images of e_a). Bv integer 8x8; half -> times 1/2."""
    M = sp.zeros(6, 6)
    for a in range(6):
        Y = Bv @ e[a] - e[a] @ Bv
        c = vec_coeffs(Y)
        for b in range(6):
            M[b, a] = c[b] / (2 if half else 1)
    return M


def comm(A, B):
    return A * B - B * A


# cyclic triples (i,j,k), 0-based temporal indices
CYC = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
# L_i = (1/2)(e_{j+3}e_{k+3} - e_j e_k),  K_i = (1/2)(e_k e_{j+3} - e_j e_{k+3})   (tex/s12, s17)
Lcl = [(e[j + 3] @ e[k + 3] - e[j] @ e[k]) for (i, j, k) in CYC]   # = 2 L_i
Kcl = [(e[k] @ e[j + 3] - e[j] @ e[k + 3]) for (i, j, k) in CYC]   # = 2 K_i
ell = [ad6(X, half=True) for X in Lcl]
kap = [ad6(X, half=True) for X in Kcl]

G = sp.zeros(6, 6)
for i in range(3):
    G[i + 3, i] = 1
    G[i, i + 3] = -1
I6 = sp.eye(6)
ETAG = ETA * G
eps = sp.LeviCivita


def sym_basis(n):
    out = []
    for a in range(n):
        for b in range(a, n):
            M = sp.zeros(n, n)
            M[a, b] = 1
            M[b, a] = 1
            out.append(M)
    return out


def full_basis(n):
    out = []
    for a in range(n):
        for b in range(n):
            M = sp.zeros(n, n)
            M[a, b] = 1
            out.append(M)
    return out


def solve_space(basis, conds):
    """Return a basis of {sum c_k basis_k : cond(M) = 0 for all conds} (conds linear maps to matrices)."""
    rows = []
    for cond in conds:
        imgs = [sp.Matrix(cond(Bk)).reshape(len(cond(Bk)), 1) for Bk in basis]
        rows.append(sp.Matrix.hstack(*imgs))
    A = sp.Matrix.vstack(*rows)
    ns = A.nullspace()
    return [sum((v[k] * basis[k] for k in range(len(basis))), sp.zeros(*basis[0].shape)) for v in ns]


def span_rank(mats):
    if not mats:
        return 0
    return sp.Matrix.hstack(*[sp.Matrix(M).reshape(len(M), 1) for M in mats]).rank()


def zfromX(X):
    X = list(X)
    return [X[j] + sp.I * X[j + 3] for j in range(3)]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ============================================================================= conventions check
conv = True
# brackets of l_G (Clifford commutators, normalised generators L_i = Lcl/2, K_i = Kcl/2)
Lh = [sp.Matrix(X) / 2 for X in Lcl]
Kh = [sp.Matrix(X) / 2 for X in Kcl]
for a in range(3):
    for b in range(3):
        rhsL = sum((eps(a, b, c) * Lh[c] for c in range(3)), sp.zeros(8, 8))
        rhsK = sum((eps(a, b, c) * Kh[c] for c in range(3)), sp.zeros(8, 8))
        conv &= comm(Lh[a], Lh[b]) == rhsL
        conv &= comm(Lh[a], Kh[b]) == rhsK
        conv &= comm(Kh[a], Kh[b]) == -rhsL
# ad6 is a Lie homomorphism on these
for a in range(3):
    for b in range(3):
        conv &= comm(ell[a], ell[b]) == sum((eps(a, b, c) * ell[c] for c in range(3)), sp.zeros(6, 6))
        conv &= comm(kap[a], kap[b]) == -sum((eps(a, b, c) * ell[c] for c in range(3)), sp.zeros(6, 6))
# ex:emboost: [K3, e1] = e5
conv &= kap[2][:, 0] == sp.Matrix([0, 0, 0, 0, 1, 0])
# l_G commutes with G, preserves eta
conv &= all(comm(M, G) == sp.zeros(6, 6) for M in ell + kap)
conv &= all(M.T * ETA + ETA * M == sp.zeros(6, 6) for M in ell + kap)
# G from Clifford data: G = -calK o S (lem:Gclifford), calK = (1/2) ad_P, S = Ad_T
Pcl = sum(e[i] @ e[i + 3] for i in range(3))
calK = ad6(Pcl, half=True)
Tcl = e[0] @ e[1] @ e[2]
Tinv = np.round(np.linalg.inv(Tcl)).astype(np.int64)
S_ad = sp.zeros(6, 6)
for a in range(6):
    c = vec_coeffs(Tcl @ e[a] @ Tinv)
    for b in range(6):
        S_ad[b, a] = c[b]
conv &= S_ad == ETA
conv &= -calK * S_ad == G
conv &= G * G == -I6 and ETA * G.T * ETA == G and G.T * ETA * G == -ETA
# K_i acts by dE = e_i x B, dB = -e_i x E ; L_i rotates B and E alike
Bs = sp.symbols('B1:4', real=True)
Es = sp.symbols('E1:4', real=True)
Xs = sp.Matrix(list(Bs) + list(Es))
for i in range(3):
    v = [1 if r == i else 0 for r in range(3)]
    dX = kap[i] * Xs
    conv &= list(dX[3:, 0]) == cross(v, list(Bs)) and list(dX[:3, 0]) == [-x for x in cross(v, list(Es))]
    dX = ell[i] * Xs
    # rotation generator about axis i: dB = e_i x B, dE = e_i x E
    conv &= list(dX[:3, 0]) == cross(v, list(Bs)) and list(dX[3:, 0]) == cross(v, list(Es))
report("d1-0", conv, "conventions: realisation, l_G = span{L_i,K_i} brackets, [K3,e1]=e5, G=-K o Ad_T, "
       "Ad_T|V = eta, K_i: dE = e_i x B, dB = -e_i x E, L_i rotate B and E alike")

# ============================================================================= d1-1a uniqueness of E6
SB = sym_basis(6)
rot_inv = solve_space(SB, [lambda N, M=M: M.T * N + N * M for M in ell])
dual_cls = solve_space(SB, [lambda N, M=M: M.T * N + N * M for M in ell] + [lambda N: N * G - G * N])
lor_part = [(N + G * N * G) / 2 for N in rot_inv]       # anticommuting with G
com_part = [(N - G * N * G) / 2 for N in rot_inv]       # commuting with G
lG_inv = solve_space(SB, [lambda N, M=M: M.T * N + N * M for M in ell + kap])
ok1 = (len(rot_inv) == 3 and len(dual_cls) == 1 and span_rank(dual_cls + [I6]) == 1
       and span_rank(com_part) == 1 and span_rank(com_part + [I6]) == 1
       and span_rank(lor_part) == 2 and span_rank(lor_part + [ETA, ETAG]) == 2
       and len(lG_inv) == 2 and span_rank(lG_inv + [ETA, ETAG]) == 2
       and G.T * G == I6 and ETAG.T == ETAG)
E6form = sp.expand((Xs.T * Xs)[0] / 2)
ok1 &= sp.expand(E6form - sum(x**2 for x in Bs + Es) / 2) == 0
info(f"rotation-invariant symmetric forms: dim {len(rot_inv)}; commuting with G: dim {len(dual_cls)} (=R*id); "
     f"l_G-invariant: dim {len(lG_inv)} (=span{{eta, eta G}})")
report("d1-1a", ok1, "rotation-invariant quadratic forms = R*id (+) span{eta, eta G}; the only rotation- and "
       "duality-invariant one is E6 = (|B|^2+|E|^2)/2 (positive); l_G-invariant forms = span{eta, eta G}")

# ============================================================================= d1-1b u(3), Lorentz x duality, intersection
FB = full_basis(6)
u3 = solve_space(FB, [lambda A: A.T + A, lambda A: A * G - G * A])
# Lie algebra of O(3,C) x U(1)_G: C-linear maps preserving q up to a phase, A^T eta + eta A = phi * eta G;
# unknowns: the 36 matrix entries and phi
rows = []
for cond in [lambda A, ph: A * G - G * A, lambda A, ph: A.T * ETA + ETA * A - ph * ETAG]:
    cols = []
    for k in range(37):
        if k < 36:
            img = cond(FB[k], 0)
        else:
            img = cond(sp.zeros(6, 6), 1)
        cols.append(sp.Matrix(img).reshape(36, 1))
    rows.append(sp.Matrix.hstack(*cols))
ns = sp.Matrix.vstack(*rows).nullspace()
g7 = [sum((v[k] * FB[k] for k in range(36)), sp.zeros(6, 6)) for v in ns]
lGG = ell + kap + [G]
# compute intersection via combined linear system on coefficients of g7 basis
cvars = sp.symbols('c0:%d' % len(g7))
Agen = sum((cvars[k] * g7[k] for k in range(len(g7))), sp.zeros(6, 6))
sol = sp.solve(list(Agen.T + Agen), cvars, dict=True)
Aint = Agen.subs(sol[0]) if sol else Agen
free = sorted(Aint.free_symbols, key=lambda s: s.name)
int_basis = [Aint.diff(s) for s in free]
lint_cpt = solve_space(FB, [lambda A: A.T + A, lambda A: A * G - G * A,
                            lambda A: A.T * ETA + ETA * A])
ok2 = (len(u3) == 9 and len(g7) == 7 and span_rank(g7 + lGG) == 7 and span_rank(lGG) == 7
       and len(int_basis) == 4 and span_rank(int_basis + ell + [G]) == 4
       and len(lint_cpt) == 3 and span_rank(lint_cpt + ell) == 3)
# u(3) structure: A = [[P,Q],[-Q,P]] -> M = P - iQ anti-Hermitian, Lie homomorphism


def tocx(A):
    P = A[:3, :3]
    Q = A[:3, 3:]
    return P - sp.I * Q


ok2 &= all(sp.simplify(tocx(A).H + tocx(A)) == sp.zeros(3, 3) for A in u3)
ok2 &= all(sp.simplify(tocx(comm(A, B)) - comm(tocx(A), tocx(B))) == sp.zeros(3, 3)
           for A in u3[:5] for B in u3[:5])
# U(3) does not preserve E0: U e1 = (1,i,0)/sqrt2
U = sp.Matrix([[1, 1, 0], [sp.I, -sp.I, 0], [0, 0, sp.sqrt(2)]]) / sp.sqrt(2)
ok2 &= sp.simplify(U.H * U) == sp.eye(3)
z1 = sp.Matrix([1, 0, 0])
z2 = U * z1
E6a, E6b = (z1.H * z1)[0] / 2, sp.simplify((z2.H * z2)[0] / 2)
E0a, E0b = sp.Abs((z1.T * z1)[0]) / 2, sp.Abs(sp.simplify((z2.T * z2)[0])) / 2
ok2 &= (E6a == E6b == sp.Rational(1, 2)) and E0a == sp.Rational(1, 2) and E0b == 0
info(f"dim u(3)-type algebra = {len(u3)}, dim Lie(O(3,C)xU(1)_G) = {len(g7)}, intersection dim = "
     f"{len(int_basis)} = span{{L_i, G}}, u(3) cap l_G = so(3) (dim {len(lint_cpt)}); "
     f"U(3) moves E0 from 1/2 to 0 at E6 = 1/2")
report("d1-1b", ok2, "maps preserving E6 and commuting with G form u(3) (dim 9); Lie(O(3,C)xU(1)_G) = l_G + R G "
       "(dim 7); intersection = so(3) + u(1) (dim 4); U(3) does not preserve E0")

# ============================================================================= d1-2a orbits off the null cone
rng = np.random.default_rng(20261017)
ellf = [np.array(M, dtype=float) for M in ell]
kapf = [np.array(M, dtype=float) for M in kap]
Gf = np.array(G, dtype=float)
etaf = np.diag(eta_d).astype(float)


def cdot(a, b):
    return np.sum(a * b)


def ccross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def to_rest(z):
    """Constructive Lorentz x duality element bringing a non-null z to (r,0,0), r = sqrt|q| > 0."""
    q = cdot(z, z)
    r = np.sqrt(abs(q))
    ph = np.exp(-0.5j * np.angle(q))          # duality factor e^{-i arg(q)/2}
    w = ph * z / r                            # q(w) = 1
    best, bestval = None, -1.0
    for _k in range(8):                       # pick the best-conditioned candidate in the q-complement of w
        v = rng.normal(size=3) + 1j * rng.normal(size=3)
        f = v - cdot(v, w) * w
        val = abs(cdot(f, f)) / np.vdot(f, f).real
        if val > bestval:
            best, bestval = f, val
    f = best
    f2 = f / np.sqrt(cdot(f, f))
    f3 = ccross(w, f2)
    M = np.column_stack([w, f2, f3])
    A = M.T                                   # A in SO(3,C), A w = e1
    return A, ph, r


maxres = 0.0
maxorth = 0.0
for _ in range(2000):
    z = rng.normal(size=3) + 1j * rng.normal(size=3)
    A, ph, r = to_rest(z)
    out = A @ (ph * z)
    # real 6x6 form of A on V: commutes with G and preserves eta and eta G (an element of O(3,C) acting on V)
    Pm, Qm_ = A.real, -A.imag
    A6 = np.block([[Pm, Qm_], [-Qm_, Pm]])
    cond6 = np.linalg.norm(A6, 2)**2
    maxorth = max(maxorth, (np.max(np.abs(A6 @ Gf - Gf @ A6)) + np.max(np.abs(A6.T @ etaf @ A6 - etaf))
                            + np.max(np.abs(A6.T @ etaf @ Gf @ A6 - etaf @ Gf))) / cond6)
    maxres = max(maxres, np.max(np.abs(out - np.array([r, 0, 0]))) / max(1, r))
    cond = np.linalg.norm(A, 2)**2            # A is complex orthogonal but not unitary: errors scale with |A|^2
    maxorth = max(maxorth, (np.max(np.abs(A.T @ A - np.eye(3))) + abs(np.linalg.det(A) - 1)) / cond)
# exact tangent ranks at integer points
tang_ok = True
for X in [[1, 2, 0, 3, -1, 2], [2, -1, 1, 0, 1, 3], [1, 0, 0, 0, 0, 0], [3, 1, -2, 1, 1, 1]]:
    Xv = sp.Matrix(X)
    z = zfromX(X)
    assert sp.expand(dot(z, z)) != 0
    Tm = sp.Matrix.hstack(*[M * Xv for M in ell + kap] + [G * Xv])
    tang_ok &= Tm.rank() == 5
nrank = []
for _ in range(200):
    X = rng.normal(size=6)
    Tm = np.column_stack([M @ X for M in ellf + kapf] + [Gf @ X])
    nrank.append(np.linalg.matrix_rank(Tm, tol=1e-9))
ok3 = maxres < 1e-10 and maxorth < 1e-10 and tang_ok and set(nrank) == {5}
info(f"2000 random non-null z mapped to (sqrt|q|,0,0): max residual {maxres:.1e}, max (|A^T A - 1|+|det A - 1|)/|A|^2 "
     f"{maxorth:.1e} (tol 1e-10); exact tangent rank 5 at 4 integer points, numeric rank 5 at 200 points")
report("d1-2a", ok3, "Lorentz x duality acts transitively on each level set |q| = const > 0 (constructive), "
       "orbit dimension 5, stabiliser dimension 2")

# ============================================================================= d1-2b null cone, U(3) orbits
null_ok = True
for a, b in [((1, 2, 2), (2, 1, -2)), ((3, 4, 0), (0, 0, 5)), ((2, 3, 6), (6, 2, -3))]:
    assert dot(a, b) == 0 and dot(a, a) == dot(b, b)
    X = sp.Matrix(list(a) + list(b))
    TL = sp.Matrix.hstack(*[M * X for M in ell + kap])
    TLG = sp.Matrix.hstack(TL, G * X)
    null_ok &= TL.rank() == 4 and TLG.rank() == 4
# constructive: null z = a + i b  ->  (1, i, 0)
maxres_n = 0.0
k3f = kapf[2]
for _ in range(500):
    a = rng.normal(size=3)
    b = rng.normal(size=3)
    b = b - a * (a @ b) / (a @ a)
    b = b * np.linalg.norm(a) / np.linalg.norm(b)
    n3 = np.cross(a, b)
    R = np.vstack([a / np.linalg.norm(a), b / np.linalg.norm(b), n3 / np.linalg.norm(n3)])
    R6 = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    X = np.concatenate([a, b])
    chi = -np.log(np.linalg.norm(a))
    Y = expm(chi * k3f) @ (R6 @ X)
    maxres_n = max(maxres_n, np.max(np.abs(Y - np.array([1, 0, 0, 0, 1, 0]))))
# U(3) transitive on spheres: unitary with first column z/|z|
maxu = 0.0
for _ in range(200):
    z = rng.normal(size=3) + 1j * rng.normal(size=3)
    Mz = np.column_stack([z, rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))])
    Qm, Rm = np.linalg.qr(Mz)
    Qm = Qm * (Rm[0, 0] / abs(Rm[0, 0]))
    maxu = max(maxu, np.max(np.abs(Qm @ np.array([np.linalg.norm(z), 0, 0]) - z)))
ok4 = null_ok and maxres_n < 1e-10 and maxu < 1e-10
info(f"null points: Lorentz tangent rank 4, adding G X keeps rank 4 (exact, 3 points); 500 random null z mapped "
     f"to (1,i,0) by rotation + boost along the axis, max residual {maxres_n:.1e}; U(3) transitive on spheres "
     f"(residual {maxu:.1e})")
report("d1-2b", ok4, "nonzero null fields form one orbit of SO(3,C) alone (dim 4, duality tangent inside); "
       "U(3) orbits are the spheres |z| = const")

# ============================================================================= d1-3a Lagrange identity, H = z z^dagger
B = sp.Matrix(Bs)
E = sp.Matrix(Es)
z = [Bs[j] + sp.I * Es[j] for j in range(3)]
zb = [Bs[j] - sp.I * Es[j] for j in range(3)]
nz2 = sp.expand(dot(z, zb))
q = sp.expand(dot(z, z))
qb = sp.expand(dot(zb, zb))
zbxz = [sp.expand(x) for x in cross(zb, z)]
zxzb = [sp.expand(x) for x in cross(z, zb)]
lag = sp.expand(nz2**2 - q * qb - dot(zbxz, zxzb))
Sfr = cross(list(Es), list(Bs))                          # frame Poynting vector E x B
ok5 = lag == 0
ok5 &= all(sp.expand(zbxz[l] + 2 * sp.I * Sfr[l]) == 0 for l in range(3))
ok5 &= all(sp.expand(sp.I / 2 * zbxz[l] - Sfr[l]) == 0 for l in range(3))
E6s = nz2 / 2
E0sq = sp.expand(q * qb / 4)
ok5 &= sp.expand(E6s**2 - dot(Sfr, Sfr) - E0sq) == 0
H = sp.Matrix(3, 3, lambda j, k: z[j] * zb[k])
ok5 &= sp.expand(H.trace() - 2 * E6s) == 0
ImH = H.applyfunc(lambda x: sp.im(sp.expand(x)))
ReH = H.applyfunc(lambda x: sp.re(sp.expand(x)))
ok5 &= all(sp.expand(ImH[j, k] - sum(eps(j, k, l) * Sfr[l] for l in range(3))) == 0
           for j in range(3) for k in range(3))
ok5 &= sp.expand((H * H.conjugate()).trace() - q * qb) == 0
ok5 &= sp.expand((ReH * ReH).trace() - 2 * dot(Sfr, Sfr) - q * qb) == 0
ok5 &= sp.expand(H * H - H.trace() * H) == sp.zeros(3, 3)
info("|z|^4 = |z.z|^2 + |zbar x z|^2 identically; zbar x z = -2i E x B; tr H = 2E6, Im H_jk = eps_jkl S_l, "
     "tr(H Hbar) = |q|^2 = tr(ReH^2) - 2|S|^2, H^2 = (tr H) H")
report("d1-3a", ok5, "complex Lagrange identity: E6^2 = |S|^2 + E0^2 with S = E x B = -S_text; "
       "H = z z^dagger carries 2E6 (trace), S (imaginary part) and |q|^2 = tr(H Hbar)")

# ============================================================================= d1-3b Rainich identity
Bt = [-x for x in Bs]                     # textbook magnetic field
Ev = list(Es)


def Tmunu(Ev, Bt):
    u = (dot(Ev, Ev) + dot(Bt, Bt)) / 2
    Sv = cross(Ev, Bt)
    T = sp.zeros(4, 4)
    T[0, 0] = u
    for i in range(3):
        T[0, i + 1] = Sv[i]
        T[i + 1, 0] = Sv[i]
        for j in range(3):
            T[i + 1, j + 1] = -(Ev[i] * Ev[j] + Bt[i] * Bt[j]) + (u if i == j else 0)
    return T


eta4 = sp.diag(1, -1, -1, -1)
T = Tmunu(Ev, Bt)
Tm = T * eta4                             # T^mu_nu
E0sq_t = sp.expand(((dot(Bt, Bt) - dot(Ev, Ev))**2 + 4 * dot(Ev, Bt)**2) / 4)
ok6 = sp.expand(E0sq_t - E0sq) == 0
ok6 &= (Tm * Tm - E0sq_t * sp.eye(4)).applyfunc(sp.expand) == sp.zeros(4, 4)
ok6 &= sp.expand(Tm.trace()) == 0
ok6 &= sp.expand(T[0, 0] - E6s) == 0
# observer-independent mass of the energy-momentum density T(t)
tt = sp.symbols('t0:4', real=True)
tv = sp.Matrix(tt)
Pv = Tm * tv
ok6 &= sp.expand((Pv.T * eta4 * Pv)[0] - E0sq_t * (tv.T * eta4 * tv)[0]) == 0
# duality invariance of T: z -> e^{i th} z
cth, sth = sp.symbols('c s', real=True)
Bn = [cth * Bs[j] - sth * Es[j] for j in range(3)]
En = [sth * Bs[j] + cth * Es[j] for j in range(3)]
Tn = Tmunu(En, [-x for x in Bn])
diff = (Tn - T).applyfunc(lambda x: sp.expand(sp.expand(x).subs(sth**2, 1 - cth**2)))
ok6 &= diff == sp.zeros(4, 4)
qn = sp.expand(dot([Bn[j] + sp.I * En[j] for j in range(3)], [Bn[j] + sp.I * En[j] for j in range(3)]))
ok6 &= sp.expand(qn - (cth + sp.I * sth)**2 * q) == 0
# dominant energy: T(t) future causal for random t (numeric)
Tf = sp.lambdify((Bs, Es), Tm, 'numpy')
dec = True
for _ in range(200):
    Bn_ = rng.normal(size=3)
    En_ = rng.normal(size=3)
    vv = rng.normal(size=3)
    vv = 0.99 * vv / np.linalg.norm(vv) * rng.random()
    g_ = 1 / np.sqrt(1 - vv @ vv)
    t_ = g_ * np.array([1, *vv])
    P_ = np.array(Tf(Bn_, En_), dtype=float) @ t_
    dec &= P_[0] > 0 and P_[0]**2 - P_[1:] @ P_[1:] > -1e-9
ok6 &= dec
info("T^mu_l T^l_nu = E0^2 delta, T^mu_mu = 0, T^00 = E6; eta(T t, T t) = E0^2 eta(t,t) for every t; "
     "T invariant under duality while q -> e^{2i theta} q; T(t) future-causal at 200 random (field, observer)")
report("d1-3b", ok6, "Rainich identity T.T = E0^2 * 1 and trace zero (Heaviside-Lorentz, u = (|E|^2+|B|^2)/2): "
       "every observer's energy-momentum density has the same invariant length E0")

# ============================================================================= d1-4a variational characterisation
ok7 = True
worst = 0.0
for _ in range(25):
    X0 = rng.normal(size=6)
    zc = X0[:3] + 1j * X0[3:]
    E0v = abs(cdot(zc, zc)) / 2

    def f(par):
        Y = expm(sum(par[i] * kapf[i] for i in range(3))) @ X0
        return 0.5 * Y @ Y

    best = None
    for _s in range(4):
        res = minimize(f, rng.normal(size=3) * 0.3, method='BFGS', options={'gtol': 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    Y = expm(sum(best.x[i] * kapf[i] for i in range(3))) @ X0
    Sy = np.cross(Y[3:], Y[:3])
    worst = max(worst, abs(best.fun - E0v) / E0v)
    ok7 &= abs(best.fun - E0v) / E0v < 1e-8 and np.linalg.norm(Sy) / E0v < 1e-3
# exact: E6 = E0/sqrt(1-v_E^2), v_E = |S|/E6
ok7 &= sp.simplify(E6s**2 * (1 - dot(Sfr, Sfr) / E6s**2) - E0sq) == 0
info(f"min over boosts of E6 equals E0 at 25 random fields (worst rel. dev. {worst:.1e}, tol 1e-8), "
     f"attained where S ~ 0")
report("d1-4a", ok7, "E0 = min of E6 over the Lorentz (and Lorentz x duality) orbit, attained exactly at S = 0; "
       "E6 = E0/sqrt(1-v_E^2) with v_E = |S|/E6")

# ============================================================================= d1-4b explicit boosts with the K_i
chi = sp.Symbol('chi', positive=True)


def expgen(M, t):
    """exp(t M) for M with M^3 = M (boost) -- exact."""
    assert M**3 == M
    return sp.eye(6) + sp.sinh(t) * M + (sp.cosh(t) - 1) * M * M


def invariants(Y):
    Bq = list(Y[:3, 0])
    Eq = list(Y[3:, 0])
    e6 = sp.simplify((dot(Bq, Bq) + dot(Eq, Eq)) / 2)
    S2 = sp.simplify(dot(cross(Eq, Bq), cross(Eq, Bq)))
    qq = sp.expand((dot(Bq, Bq) - dot(Eq, Eq)) + 2 * sp.I * dot(Eq, Bq))
    e0 = sp.sqrt(sp.factor(sp.simplify(sp.expand(qq * sp.conjugate(qq))))) / 2
    return e6, S2, e0


ok8 = True
av, bv = sp.symbols('a b', positive=True)
for X0 in [sp.Matrix([1, 0, 0, 0, 0, 0]), sp.Matrix([av, 0, 0, bv, 0, 0])]:
    e60, _, e00 = invariants(X0)
    ok8 &= sp.simplify(e60 - e00) == 0
    for i in (1, 2):           # boosts perpendicular to n = e1
        Y = expgen(kap[i], chi) * X0
        e6, S2, e0 = invariants(Y)
        ok8 &= sp.simplify((e6 - e00 * sp.cosh(2 * chi)).rewrite(sp.exp)) == 0
        ok8 &= sp.simplify((S2 - e00**2 * sp.sinh(2 * chi)**2).rewrite(sp.exp)) == 0
        ok8 &= sp.simplify(e0 - e00) == 0
        # rest-frame velocity |S|/(E6+E0) = tanh(chi); v_E = tanh(2 chi) = 2 beta/(1+beta^2)
        vrest = sp.sinh(2 * chi) / (sp.cosh(2 * chi) + 1)
        ok8 &= sp.simplify((vrest - sp.tanh(chi)).rewrite(sp.exp)) == 0
    Y = expgen(kap[0], chi) * X0                 # boost along n
    e6, S2, e0 = invariants(Y)
    ok8 &= sp.simplify(e6 - e60) == 0 and sp.simplify(S2) == 0
    # observer riding with the lab energy velocity tanh(2chi) e3 : exp(-2chi K3) exp(chi K3) = exp(-chi K3)
    Ylab = expgen(kap[2], chi) * X0
    Yobs = expgen(kap[2], -2 * chi) * Ylab
    Slab = cross(list(Ylab[3:, 0]), [-x for x in Ylab[:3, 0]])      # textbook S_t = E x B_t
    Sobs = cross(list(Yobs[3:, 0]), [-x for x in Yobs[:3, 0]])
    ok8 &= all(sp.simplify((Sobs[l] + Slab[l]).rewrite(sp.exp)) == 0 for l in range(3))
    ok8 &= sp.simplify((invariants(Yobs)[0] - invariants(Ylab)[0]).rewrite(sp.exp)) == 0
    ok8 &= sp.simplify((Slab[2] - e00 * sp.sinh(2 * chi)).rewrite(sp.exp)) == 0
beta = sp.Symbol('beta', positive=True)
ok8 &= sp.simplify((sp.tanh(2 * chi) - 2 * sp.tanh(chi) / (1 + sp.tanh(chi)**2)).rewrite(sp.exp)) == 0
info("pure magnetic field along n (and B || E along n): boosts K_2, K_3 of rapidity chi give E6 = E0 cosh 2chi, "
     "|S| = E0 sinh 2chi, E0 fixed; boost K_1 along n leaves E6 fixed; |S|/(E6+E0) = tanh chi; an observer riding "
     "with the energy velocity tanh(2chi) sees the same E6 and the reversed Poynting vector")
report("d1-4b", ok8, "with the paper's K_i: E6 = E0 cosh(2chi), |S| = E0 sinh(2chi), v_E = tanh(2chi) = "
       "2beta/(1+beta^2); boost along the field axis leaves E6 unchanged")

# ============================================================================= d1-4c the rest-frame velocity of s14
ok9 = True
worst_t, worst_f = 0.0, np.inf
for _ in range(100):
    X0 = rng.normal(size=6)
    Bq, Eq = X0[:3], X0[3:]
    u = 0.5 * X0 @ X0
    zc = Bq + 1j * Eq
    E0v = abs(cdot(zc, zc)) / 2
    for label, Svec in (("text", np.cross(Eq, -Bq)), ("frame", np.cross(Eq, Bq))):
        vobs = Svec / (u + E0v)
        sp_ = np.linalg.norm(vobs)
        n = vobs / sp_
        # frame moving with velocity v_obs: generator sum_i(-n_i) K_i (tex/s12: v = -u)
        Y = expm(np.arctanh(sp_) * sum(-n[i] * kapf[i] for i in range(3))) @ X0
        Sres = np.linalg.norm(np.cross(Y[3:], Y[:3])) / E0v
        if label == "text":
            worst_t = max(worst_t, Sres, abs(0.5 * Y @ Y - E0v) / E0v)
        else:
            worst_f = min(worst_f, Sres)
ok9 = worst_t < 1e-9 and worst_f > 1e-3
info(f"observer at v = S_text/(u+|q|/2) sees S = 0 and u = |q|/2 (worst rel. {worst_t:.1e}, tol 1e-9); with the "
     f"frame vector E x B instead the residual Poynting vector stays >= {worst_f:.2f} E0")
report("d1-4c", ok9, "s14's rest-frame velocity S/(u+|q|/2) is confirmed with S the textbook Poynting vector "
       "E x B_t (= -E x B); the energy velocity S/u is beta (+) beta, not beta")

# ============================================================================= d1-5a SI units
eps0, cc = sp.symbols('epsilon0 c', positive=True)
mu0 = 1 / (eps0 * cc**2)
Ev3 = list(Es)
Bt3 = sp.symbols('b1:4', real=True)           # textbook B in tesla
u_SI = eps0 / 2 * (dot(Ev3, Ev3) + cc**2 * dot(Bt3, Bt3))
S_SI = [x / mu0 for x in cross(Ev3, list(Bt3))]
g_SI = [x / cc**2 for x in S_SI]
qc_text = cc**2 * dot(Bt3, Bt3) - dot(Ev3, Ev3) - 2 * sp.I * cc * dot(Bt3, Ev3)   # z_c = cB + iE, B = -B_t
Bfr = [-x for x in Bt3]
zc_ = [cc * Bfr[j] + sp.I * Ev3[j] for j in range(3)]
qc_paper = sp.expand(dot(zc_, zc_))
ok10 = sp.expand(qc_paper - qc_text) == 0
absqc2 = sp.expand(qc_text * sp.conjugate(qc_text))
u0sq = eps0**2 / 4 * absqc2
ok10 &= sp.simplify(u_SI**2 - cc**2 * dot(g_SI, g_SI) - u0sq) == 0
# u = (eps0/2)|z_c|^2
ok10 &= sp.expand(u_SI - eps0 / 2 * sp.expand(dot(zc_, [sp.conjugate(x) for x in zc_]))) == 0
info("u = (eps0/2)(|E|^2 + c^2|B_t|^2) = (eps0/2)|z_c|^2, z_c = cB + iE; g = S/c^2, S = E x B_t/mu0; "
     "q_c = c^2|B_t|^2 - |E|^2 - 2ic B_t.E (paper sign: c^2|B|^2 - |E|^2 + 2ic B.E)")
report("d1-5a", ok10, "SI form: u^2 = c^2|g|^2 + u0^2 with u0 = (eps0/2)|q_c|, i.e. E^2 = (pc)^2 + (mc^2)^2 per unit "
       "volume with rest-mass density rho0 = u0/c^2 (eps0 mu0 c^2 = 1)")

# ============================================================================= d1-5b c^2 as the isotropic weight
Cs = sp.Matrix(3, 3, sp.symbols('C0:9'))
R = [sp.Matrix(3, 3, lambda j, k: -eps(i, j, k)) for i in range(3)]
eqs = []
for Ri in R:
    eqs += list(Cs * Ri - Ri * Cs)
solC = sp.solve(eqs, list(Cs), dict=True)[0]
Csol = Cs.subs(solC)
ok11 = len(Csol.free_symbols) == 1 and sp.simplify(Csol - Csol[0, 0] * sp.eye(3)) == sp.zeros(3, 3)
cs = sp.Symbol('c', positive=True)
Gc = sp.zeros(6, 6)
Gc[3:, :3] = cs * sp.eye(3)
Gc[:3, 3:] = -sp.eye(3) / cs
ok11 &= Gc * Gc == -I6
ok11 &= all(comm(Gc, M) == sp.zeros(6, 6) for M in ell)
# a conversion with distinct c_i does not commute with the diagonal rotations
c1, c2, c3 = sp.symbols('c1 c2 c3', positive=True)
Gd = sp.zeros(6, 6)
Gd[3:, :3] = sp.diag(c1, c2, c3)
Gd[:3, 3:] = -sp.diag(1 / c1, 1 / c2, 1 / c3)
commd = [sp.simplify(comm(Gd, M)) for M in ell]
ok11 &= any(x != 0 for M in commd for x in M)
formsc = solve_space(SB, [lambda N, M=M: M.T * N + N * M for M in ell] + [lambda N: Gc.T * N + N * Gc])
ok11 &= len(formsc) == 1 and span_rank(formsc + [sp.diag(cs**2, cs**2, cs**2, 1, 1, 1)]) == 1
kap_ell = sp.Symbol('ell', positive=True)
ok11 &= ((cs / kap_ell) * G)**2 == -(cs**2 / kap_ell**2) * I6
info("Schur: SO(3)-intertwiners V+ -> V- are c*1; with C = c*1 the duality root G_c = [[0,-1/c],[c,0]] commutes "
     "with the diagonal rotations (distinct c_i would not), and the rotation- and duality-invariant energy is "
     "c^2|B|^2 + |E|^2 up to scale; (kappa G)^2 = -c^2/ell^2")
report("d1-5b", ok11, "c enters the undivided energy only as the isotropic weight c^2 of |B|^2, forced by "
       "Theorem 3.2 (one conversion constant); it is a unit conversion, not a symmetry")

# ============================================================================= d1-6a standing wave in a periodic cell
zz, tt_, kk, aa = sp.symbols('z t k a', positive=True)
w_ = cc * kk
lam = 2 * sp.pi / kk
ExR, ByR = aa * sp.cos(kk * zz - w_ * tt_), aa / cc * sp.cos(kk * zz - w_ * tt_)
ExL, ByL = aa * sp.cos(kk * zz + w_ * tt_), -aa / cc * sp.cos(kk * zz + w_ * tt_)
Ex, By = ExR + ExL, ByR + ByL
# Maxwell (SI, vacuum): dB_y/dt = -dE_x/dz, dE_x/dt = -c^2 dB_y/dz
ok12 = sp.simplify(sp.diff(By, tt_) + sp.diff(Ex, zz)) == 0 and sp.simplify(sp.diff(Ex, tt_) + cc**2 * sp.diff(By, zz)) == 0


def cell(expr):
    return sp.simplify(sp.integrate(sp.expand(sp.expand_trig(expr)), (zz, 0, lam)))


uSW = eps0 / 2 * (Ex**2 + cc**2 * By**2)
gSW = eps0 * Ex * By            # g_z = S_z/c^2 = (E_x B_y/mu0)/c^2 = eps0 E_x B_y
U_cell = cell(uSW)
P_cell = cell(gSW)
ok12 &= sp.simplify(U_cell - eps0 * aa**2 * lam) == 0 and sp.simplify(P_cell) == 0
for (ex_, by_) in ((ExR, ByR), (ExL, ByL)):
    qcon = sp.simplify(cc**2 * by_**2 - ex_**2)          # E.B = 0 for these polarisations
    ok12 &= qcon == 0
    Uc = cell(eps0 / 2 * (ex_**2 + cc**2 * by_**2))
    Pc = cell(eps0 * ex_ * by_)
    ok12 &= sp.simplify(Uc**2 - cc**2 * Pc**2) == 0       # each constituent: M = 0
qtot = sp.simplify(cc**2 * By**2 - Ex**2)
ok12 &= qtot != 0
# Laue: cell stress T^zz = u - (eps0 Ez^2 + Bz^2/mu0) = u
Tzz_cell = cell(uSW)
Txx_cell = cell(uSW - eps0 * Ex**2)
Tyy_cell = cell(uSW - eps0 * cc**2 * By**2)
ok12 &= sp.simplify(Tzz_cell - U_cell) == 0 and sp.simplify(Txx_cell + Tyy_cell) == 0
info(f"per cell (area 1, length 2pi/k): U = {sp.simplify(U_cell)}, P = {P_cell}, M c^2 = U; each constituent "
     f"has q = 0 and U = |P| c; total q = {sp.factor(qtot)}; Laue stress int T^zz = U != 0, int T^xx = "
     f"{sp.simplify(Txx_cell)}")
report("d1-6a", ok12, "standing wave of two null waves: P = 0 and M c^2 = U > 0 per cell although each constituent "
       "is null; the cell stress int T^zz = U is nonzero, so the confining walls belong to the system (Laue)")

# ============================================================================= d1-6b mass of the whole is not interference
ok13 = True
Zr = sp.Symbol('Z', real=True)
for tval in (sp.Integer(0), sp.Rational(1, 2), sp.Integer(2)):
    g1 = sp.Piecewise(((1 - (Zr - tval)**2)**2, sp.Abs(Zr - tval) <= 1), (0, True))
    g2 = sp.Piecewise(((1 - (Zr + tval)**2)**2, sp.Abs(Zr + tval) <= 1), (0, True))
    Ex_, Bty = g1 + g2, g1 - g2                   # c = 1: E_x, (B_t)_y
    u_ = sp.expand((Ex_**2 + Bty**2) / 2)
    Sz = sp.expand(Ex_ * Bty)
    Uv = sp.integrate(u_, (Zr, -4, 4))
    Pv_ = sp.integrate(Sz, (Zr, -4, 4))
    E0int = sp.integrate(2 * sp.Abs(g1 * g2), (Zr, -4, 4))      # E0 = |q|/2 = 2|f1 f2|
    ok13 &= sp.simplify(Uv - sp.Rational(512, 315)) == 0 and sp.simplify(Pv_) == 0
    if tval == 2:
        ok13 &= sp.simplify(E0int) == 0
    if tval == 0:
        ok13 &= sp.simplify(E0int - Uv) == 0
    info(f"two bump pulses, t = {tval}: U = {sp.nsimplify(Uv)}, P = {Pv_}, int E0 = {sp.nsimplify(E0int)}")
# standing wave (c = 1): int E0 over a cell oscillates between (2/pi) U and U
xg = np.linspace(0, 2 * np.pi, 200001)[:-1]
ratios = []
for wt in np.linspace(0, np.pi / 2, 91):
    qloc = 4 * (np.sin(xg)**2 * np.sin(wt)**2 - np.cos(xg)**2 * np.cos(wt)**2)
    ratios.append(np.mean(np.abs(qloc) / 2) / 1.0)      # U per unit length = a^2 = 1 with a = 1
ratios = np.array(ratios)
ok13 &= abs(ratios.min() - 2 / np.pi) < 1e-6 and abs(ratios.max() - 1) < 1e-6 and abs(np.argmin(ratios) - 45) <= 0
info(f"standing wave: int E0 / U ranges over [{ratios.min():.6f}, {ratios.max():.6f}], 2/pi = {2/np.pi:.6f} "
     f"(min at omega t = pi/4; grid 2e5 points, tol 1e-6)")
# reverse triangle inequality M >= int E0 for random superpositions of future causal densities (numeric)
rti = True
for _ in range(300):
    Xs_ = rng.normal(size=(40, 6))
    Ps = []
    m_sum = 0.0
    for X in Xs_:
        zc = X[:3] + 1j * X[3:]
        u = 0.5 * X @ X
        Sv = np.cross(X[3:], -X[:3])
        Ps.append(np.concatenate([[u], Sv]))
        m_sum += abs(cdot(zc, zc)) / 2
    Ptot = np.sum(Ps, axis=0)
    M = np.sqrt(Ptot[0]**2 - Ptot[1:] @ Ptot[1:])
    rti &= M >= m_sum - 1e-9
ok13 &= rti
report("d1-6b", ok13, "M >= int E0 d^3x with equality only when all T^{0mu} are parallel; two separated "
       "counter-propagating null pulses have q = 0 everywhere and M = U > 0: the mass of the whole is not interference")

# ============================================================================= d1-6c orthogonality blind
bb = sp.symbols('beta1:4', real=True)
th = sp.Symbol('theta', real=True)
TA = Tmunu([0, 0, 0], list(bb))
TB = Tmunu([x * sp.sin(th) for x in bb], [x * sp.cos(th) for x in bb])
ok14 = (TB - TA).applyfunc(lambda x: sp.simplify(x)) == sp.zeros(4, 4)
EBB = sp.simplify(dot([x * sp.sin(th) for x in bb], [x * sp.cos(th) for x in bb]))
ok14 &= sp.simplify(EBB - sp.sin(2 * th) / 2 * dot(bb, bb)) == 0
report("d1-6c", ok14, "Cauchy data (0, b) and (b sin th, b cos th) have the same T^{mu nu} at every point, hence the "
       "same U, P, M, while E.B_t = 0 versus (1/2) sin 2th |b|^2: M does not detect orthogonality")

# ============================================================================= d1-7 the momentum reading
E1, E2, E3, p1, p2, p3, mu = sp.symbols('E1 E2 E3 p1 p2 p3 mu', real=True)
Pvec = sp.Matrix([E1 / cc, E2 / cc, E3 / cc, p1, p2, p3])
etaPP = (Pvec.T * ETA * Pvec)[0]
W = sp.Matrix([1, 1, 1, 0, 0, 0]) / sp.sqrt(3)
EW = cc * (W.T * ETA * Pvec)[0]                    # clock energy (E1+E2+E3)/sqrt3
Evec = sp.Matrix([E1, E2, E3])
Eperp = Evec - EW * sp.Matrix([1, 1, 1]) / sp.sqrt(3)
m4sq_c4 = EW**2 - cc**2 * (p1**2 + p2**2 + p3**2)
mu2c4 = cc**2 * etaPP                              # eta(P,P) = mu^2 c^2  ->  mu^2 c^4 = c^2 eta(P,P)
ok15 = sp.simplify(m4sq_c4 - (mu2c4 - (Eperp.T * Eperp)[0])) == 0
# clock e1 (interior gamma^0): relative energies are E2, E3  -> s27 prop:toy
m4e1 = E1**2 - cc**2 * (p1**2 + p2**2 + p3**2)
ok15 &= sp.simplify(m4e1 - (mu2c4 - E2**2 - E3**2)) == 0
# example: mu = 0 6-D momentum with relative-time energy is tachyonic in 4-D
ex = {E1: 3, E2: 4, E3: 0, p1: 5, p2: 0, p3: 0, cc: 1}
ok15 &= etaPP.subs(ex) == 0 and m4e1.subs(ex) == -16
# field reading: E6 >= E0 always (Lagrange), already exact
ok15 &= sp.expand(E6s**2 - E0sq - dot(Sfr, Sfr)) == 0
info("eta(P,P) c^2 = mu^2 c^4 = E_W^2 + |E_perp|^2 - |p|^2 c^2 ; m4^2 c^4 = mu^2 c^4 - |E_perp|^2 (clock W), "
     "= mu^2 c^4 - E2^2 - E3^2 (clock e1); example (3,4,0;5,0,0): mu = 0, m4^2 = -16")
report("d1-7", ok15, "momentum reading: splitting the temporal energy lowers the interior mass, m4^2 c^4 = "
       "mu^2 c^4 - |E_perp|^2 <= mu^2 c^4 (signature obstruction); the field-strength reading has E6 >= E0")

# ============================================================================= d1-8a the mass form on the frame
P6 = S_ad                                        # Ad_T on V
ok16 = P6 == ETA
ok16 &= P6 * G == -G * P6
ok16 &= all(P6 * M * P6 == M for M in ell) and all(P6 * M * P6 == -M for M in kap)
# q(z) = h(Pz, z) with Pz the field of Ad_T X:  Ad_T X <-> zbar
Xp = P6 * Xs
zP = zfromX(list(Xp))
ok16 &= all(sp.expand(zP[j] - zb[j]) == 0 for j in range(3))
hPz = sp.expand(sum(sp.conjugate(zP[j]) * z[j] for j in range(3)))
ok16 &= sp.expand(hPz - q) == 0
# E0 = (1/2) max_theta eta(e^{th G}X, e^{th G}X)
Re_rot = []
for _ in range(50):
    X0 = rng.normal(size=6)
    zc = X0[:3] + 1j * X0[3:]
    ths = np.linspace(0, np.pi, 20001)
    vals = [0.5 * (expm(t * Gf) @ X0) @ etaf @ (expm(t * Gf) @ X0) for t in ths[::200]]
    # analytic: (1/2) Re(e^{2i th} q)
    vals2 = 0.5 * np.real(np.exp(2j * ths) * cdot(zc, zc))
    Re_rot.append(abs(vals2.max() - abs(cdot(zc, zc)) / 2) < 1e-6 and
                  np.allclose(vals, vals2[::200], atol=1e-10))
ok16 &= all(Re_rot)
# T as a Grassmann kernel: antisymmetric, commutes with compact bivectors, anticommutes with boosts
Tk = Tcl
kern = np.array_equal(Tk.T, -Tk)
for a, b in itertools.combinations(range(6), 2):
    Bv = e[a] @ e[b]
    same = (eta_d[a] == eta_d[b])
    if same:
        kern &= np.array_equal(Tk @ Bv, Bv @ Tk)
    else:
        kern &= np.array_equal(Tk @ Bv, -Bv @ Tk)
signs = [np.array_equal(e[a] @ Tk @ np.linalg.inv(e[a]).round().astype(np.int64), Tk) for a in range(6)]
kern &= signs == [True, True, True, False, False, False]
ok16 &= kern
info("Ad_T|V = eta = parity P (B -> B, E -> -E): anticommutes with G and with the boosts K_i, commutes with L_i; "
     "q(z) = h(Pz, z); E0 = max over duality of (1/2) eta(X,X); T is an so(3,3)-invariant mass kernel whose sign "
     "is flipped by the spatial generators only")
report("d1-8a", ok16, "on the frame the Dirac mass form acts as parity, not as an l_G-invariant; the rest energy is "
       "the modulus of the overlap of the field with its mirror image, E0 = |h(Pz,z)|/2, while E6 = h(z,z)/2")

# ============================================================================= d1-8b the stabiliser of E0 in U(3)
# M = P - iQ in u(3) (P antisymmetric, Q symmetric) preserves |q|^2 iff Re(conj(q) z^T M z) = 0 identically
pa = sp.symbols('pa0:3', real=True)
qs = sp.symbols('qs0:6', real=True)
Pm_ = sp.Matrix([[0, pa[0], pa[1]], [-pa[0], 0, pa[2]], [-pa[1], -pa[2], 0]])
Qm_ = sp.Matrix([[qs[0], qs[1], qs[2]], [qs[1], qs[3], qs[4]], [qs[2], qs[4], qs[5]]])
Mu = Pm_ - sp.I * Qm_
zv = sp.Matrix(z)
dq = sp.expand((zv.T * Mu * zv)[0])
expr = sp.expand(sp.re(sp.expand(sp.conjugate(q) * dq, complex=True)))
poly = sp.Poly(expr, *(list(Bs) + list(Es)))
lin = [c for c in poly.coeffs()]
sol8 = sp.solve(lin, list(pa) + list(qs), dict=True)
Mstab = Mu.subs(sol8[0]) if sol8 else Mu
freeS = sorted(Mstab.free_symbols - set(Bs) - set(Es), key=lambda s: s.name)
stab_basis = [Mstab.diff(v) for v in freeS]
# expected: real antisymmetric (so(3)) and i*1 (u(1))
expected = [sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), sp.Matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
            sp.Matrix([[0, 0, 0], [0, 0, 1], [0, -1, 0]]), sp.I * sp.eye(3)]


def cvec(M):
    return sp.Matrix([sp.re(x) for x in M] + [sp.im(x) for x in M])


ok16b = len(stab_basis) == 4
ok16b &= sp.Matrix.hstack(*[cvec(M) for M in stab_basis + expected]).rank() == 4
info(f"u(3) elements preserving |q|: dim {len(stab_basis)} = so(3) + u(1); the U(3)-orbit of the rest-energy "
     f"function |q|/2 has dimension 9 - 4 = 5 (real structures up to phase)")
report("d1-8b", ok16b, "the stabiliser of the rest energy E0 inside the symmetry U(3) of the undivided energy is "
       "U(1).O(3) (Lie algebra so(3)+u(1), dim 4): the differentiation E6 -> (E0, |S|) is the reduction U(3) -> "
       "U(1).O(3) effected by the real structure P = Ad_T")

# ============================================================================= d1-9 interior dynamics
kp, mm, tau = sp.symbols('kappa m tau', positive=True)
cm = sp.Matrix(sp.symbols('am1:4', real=True)) + sp.I * sp.Matrix(sp.symbols('bm1:4', real=True))
cpl = sp.Matrix(sp.symbols('ap1:4', real=True)) + sp.I * sp.Matrix(sp.symbols('bp1:4', real=True))
zt = cm * sp.exp(-sp.I * kp * tau) + cpl * sp.exp(sp.I * kp * tau)
zdot = zt.diff(tau)
# eta(X,X) = Re q(z), eta(GX,GX) = -eta(X,X); Noether energy
EN = mm / 2 * sp.re(sp.expand((zdot.T * zdot)[0])) + mm * kp**2 / 2 * sp.re(sp.expand((zt.T * zt)[0]))
ENexp = 2 * mm * kp**2 * sp.re(sp.expand((cm.T * cpl)[0]))
ok17 = sp.simplify(sp.expand(EN - ENexp, complex=True)) == 0
# momentum <-> zdot + i kappa z = 2 i kappa c_+ e^{i kappa tau}
ok17 &= sp.simplify(zdot + sp.I * kp * zt - 2 * sp.I * kp * cpl * sp.exp(sp.I * kp * tau)) == sp.zeros(3, 1)
# zero-momentum orbits: E_N = 0, E6 and E0 constant
z0 = zt.subs({s: 0 for s in cpl.free_symbols})
EN0 = sp.simplify(EN.subs({s: 0 for s in cpl.free_symbols}))
ok17 &= EN0 == 0
E6t = sp.simplify(sp.expand((z0.H * z0)[0], complex=True))
ok17 &= sp.diff(E6t, tau) == 0
q0t = sp.expand((z0.T * z0)[0])
ok17 &= sp.simplify(sp.diff(sp.expand(q0t * sp.conjugate(q0t), complex=True), tau)) == 0
# general orbit: |z|^2 oscillates unless h(c_-, c_+) = 0
nz = sp.expand((zt.H * zt)[0], complex=True)
osc = sp.simplify(nz - (cm.H * cm)[0] - (cpl.H * cpl)[0])
target = 2 * sp.re(sp.expand((cm.H * cpl)[0] * sp.exp(2 * sp.I * kp * tau)))
ok17 &= sp.simplify(sp.expand(osc - target, complex=True)) == 0
# the equation of motion is independent of m: Xddot = -kappa^2 X (check on the general solution)
ok17 &= sp.simplify(zt.diff(tau, 2) + kp**2 * zt) == sp.zeros(3, 1)
info("Noether energy = 2 m kappa^2 Re(c_-.c_+): zero on every zero-momentum orbit, indefinite otherwise; on those "
     "orbits E6 and E0 are constant; |z|^2 = |c_-|^2 + |c_+|^2 + 2 Re(h(c_-,c_+) e^{2i kappa tau})")
report("d1-9", ok17, "interior dynamics: m is an overall factor, the Noether energy vanishes on the zero-momentum "
       "(duality-rotation) orbits that carry constant E6 and E0, so the interior energy is not a rest energy")

# ============================================================================= d1-10 the closure
mval, I0 = 1.0, 1.3
ab = 0.7 + 0.4j
Nlor = (ab.real * etaf + ab.imag * (Gf @ etaf)) / mval
Ndual = np.eye(6) / mval


def w_of(p):
    pi_ = etaf @ p
    zc = pi_[:3] + 1j * pi_[3:]
    return cdot(zc, zc), zc


def rhs(t, y, N):
    h, p = y[0], y[1:7]
    dp = (h / I0) * (Gf.T @ p)
    dh = -p @ Gf @ N @ p
    return np.concatenate([[dh], dp, [h / I0]])


def run(p0, N, h0=2.0, T=30.0):
    y0 = np.concatenate([[h0], p0, [0.0]])
    sol = solve_ivp(rhs, (0, T), y0, args=(N,), method='DOP853', rtol=1e-12, atol=1e-12, dense_output=False,
                    t_eval=np.linspace(0, T, 3001))
    return sol


ok18 = True
for trial in range(4):
    p0 = rng.normal(size=6)
    w0, z0c = w_of(p0)
    E0pi = abs(w0) / 2
    R0e = ab * w0
    R0, ph0 = abs(R0e), np.angle(R0e)
    sol = run(p0, Nlor)
    h, P, thv = sol.y[0], sol.y[1:7], sol.y[7]
    cas = np.sum(P**2, axis=0)
    ws = np.array([w_of(P[:, k])[0] for k in range(P.shape[1])])
    Ss = np.array([np.cross(w_of(P[:, k])[1].imag, w_of(P[:, k])[1].real) for k in range(P.shape[1])])
    Hs = h**2 / (2 * I0) + 0.5 * np.einsum('ik,ij,jk->k', P, Nlor, P)
    Eth = 0.5 * I0 * (h / I0)**2 + R0 / (2 * mval) * np.cos(2 * thv + ph0)
    ok18 &= np.max(np.abs(cas - cas[0])) / cas[0] < 1e-9
    ok18 &= np.max(np.abs(np.abs(ws) - abs(w0))) / abs(w0) < 1e-9
    ok18 &= np.max(np.abs(ws - np.exp(2j * thv) * w0)) / abs(w0) < 1e-8
    ok18 &= np.max(np.abs(Ss - Ss[0])) / np.linalg.norm(Ss[0]) < 1e-8
    ok18 &= np.max(np.abs(Hs - Hs[0])) / abs(Hs[0]) < 1e-9 and np.max(np.abs(Eth - Hs[0])) / abs(Hs[0]) < 1e-9
    ok18 &= abs(R0 / (2 * mval) - abs(ab) * E0pi / mval) < 1e-12
    ok18 &= np.ptp(h) > 1e-3                          # rate modulated when E0(pi) > 0
    # duality-class inertia: rate constant, frame energy E6(pi)/m
    sol2 = run(p0, Ndual)
    ok18 &= np.ptp(sol2.y[0]) < 1e-10
    ok18 &= abs(0.5 * p0 @ Ndual @ p0 - 0.5 * (p0 @ p0) / mval) < 1e-12
# null momentum data: the rate is constant for Lorentz-class inertia
pnull = etaf @ np.array([1.0, 0, 0, 0, 1.0, 0]) * 0.8          # pi = (1,0,0;0,1,0)*0.8, z(pi) ~ (1,i,0)
soln = run(pnull, Nlor)
ok18 &= abs(w_of(pnull)[0]) < 1e-14 and np.ptp(soln.y[0]) < 1e-10
info("Lie-Poisson flow (DOP853, rtol=atol=1e-12, 4 random momenta): |p|^2 = 2 E6(pi), |w| = 2 E0(pi), S(pi) and H "
     "conserved to 1e-9; w = e^{2i theta} w0; pendulum energy matches H; R0/(2m) = |a+ib| E0(pi)/m; h varies; "
     "null data or N = 1/m: h constant to 1e-10")
report("d1-10", ok18, "in the closure the undivided energy of the momentum is the Casimir |p|^2/2, and its rest part "
       "E0(pi) is exactly the amplitude that breaks the rotor's duality translation: the rate kappa = h/I0 is constant "
       "iff E0(pi) = 0")

# ============================================================================= d1-11 where the b-field belongs
# one constituent: elements of so(3,3) preserving E.B (equivalently q, equivalently |q|) are exactly l_G
so33 = [ad6(e[a] @ e[b]) for a, b in itertools.combinations(range(6), 2)]
ok19 = span_rank(so33) == 15
cv = sp.symbols('w0:15')
Agen = sum((cv[k] * so33[k] for k in range(15)), sp.zeros(6, 6))
solq = sp.solve(list(Agen.T * ETAG + ETAG * Agen), cv, dict=True)
Aq = Agen.subs(solq[0])
bas_q = [Aq.diff(s) for s in sorted(Aq.free_symbols, key=lambda s: s.name)]
ok19 &= len(bas_q) == 6 and span_rank(bas_q + ell + kap) == 6
# abstract ad formula [e_A e_B, e_C] = 2 eta_BC e_A - 2 eta_AC e_B, checked against the realisation


def ad_abstract(A, Bi, etad):
    n = len(etad)
    M = sp.zeros(n, n)
    for C in range(n):
        if Bi == C:
            M[A, C] += 2 * etad[Bi]
        if A == C:
            M[Bi, C] -= 2 * etad[A]
    return M


ok19 &= all(ad_abstract(a, b, eta_d) == ad6(e[a] @ e[b]) for a, b in itertools.combinations(range(6), 2))
# three constituents
eta3 = eta_d * 3
ETA3 = sp.diag(*eta3)
G3 = sp.diag(G, G, G)
Wd = [7, 13, 14, 15]           # 1-based weak directions (tex/sA)


def rot(i, j):                 # rot(i,j) = e_{i'}e_{j'} - e_i e_j, 1-based, i' = i+3
    return ad_abstract(i + 2, j + 2, eta3) - ad_abstract(i - 1, j - 1, eta3)


so4D = [rot(i, j) for i, j in itertools.combinations(Wd, 2)]
ok19 &= span_rank(so4D) == 6
ok19 &= all(comm(M, G3) == sp.zeros(18, 18) for M in so4D)
ok19 &= all(M.T == -M for M in so4D)
ok19 &= all(M.T * ETA3 + ETA3 * M == sp.zeros(18, 18) for M in so4D)
ok19 &= all(M.T * ETA3 * G3 + ETA3 * G3 * M == sp.zeros(18, 18) for M in so4D)
# redistribution example: z2 = (1,i,0) null, z3 = (1,0,0) at rest; rotate by pi/4 with rot(7,13)
X3 = sp.zeros(18, 1)
X3[6, 0] = 1     # e^{(2)}_1 : B_1 of constituent 2
X3[10, 0] = 1    # e^{(2)}_5 : E_2 of constituent 2
X3[12, 0] = 1    # e^{(3)}_1 : B_1 of constituent 3
Rg = rot(7, 13)
thq = sp.pi / 4
expR = sp.eye(18) + sp.sin(thq) / 2 * Rg + (1 - sp.cos(thq)) / 4 * Rg * Rg
ok19 &= (Rg / 2)**3 == -(Rg / 2)
Y3 = sp.simplify(expR * X3)


def consts(Y):
    out = []
    for k in range(3):
        zk = [Y[6 * k + j] + sp.I * Y[6 * k + 3 + j] for j in range(3)]
        out.append((sp.simplify(sum(sp.Abs(x)**2 for x in zk) / 2), sp.simplify(sp.expand(dot(zk, zk)))))
    return out


before, after = consts(X3), consts(Y3)
tot6_b, tot6_a = sum(x[0] for x in before), sum(x[0] for x in after)
totq_b, totq_a = sum(x[1] for x in before), sum(x[1] for x in after)
sE0_b = sum(sp.Abs(x[1]) / 2 for x in before)
sE0_a = sum(sp.Abs(x[1]) / 2 for x in after)
ok19 &= sp.simplify(tot6_b - tot6_a) == 0 and sp.simplify(totq_b - totq_a) == 0
ok19 &= sp.simplify(sE0_b - sp.Rational(1, 2)) == 0 and sp.simplify(sE0_a - sp.Rational(3, 2)) == 0
info(f"so(3,3) elements preserving E.B: dim {len(bas_q)} = l_G; so(4)_Delta (dim 6) commutes with G_3, is compact and "
     f"preserves q_3 = sum_k z^(k).z^(k); example: per-constituent (E6, q) {before} -> {after}: total E6 "
     f"{tot6_b} and total q {totq_b} fixed, sum of E0^(k) {sE0_b} -> {sE0_a}")
report("d1-11", ok19, "a transport generator of so(3,3) preserves E0 for all fields iff it lies in l_G; the "
       "binding-compatible so(4)_Delta = su(2)+su(2) preserves the total E6 and total q at three constituents, so it "
       "redistributes rest energy between constituents but cannot create it")

# ============================================================================= done
if FAILED:
    print(f"FAILED: {FAILED}")
    sys.exit(1)
print("ALL d1 CLAIMS VERIFIED")
sys.exit(0)
