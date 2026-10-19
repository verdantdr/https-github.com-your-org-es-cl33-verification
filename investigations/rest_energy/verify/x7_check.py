#!/usr/bin/env python3
"""Independent adversarial check of track d7 (the b-field between manifolds).

Rebuilt from the paper sources (tex/s02_arena.tex eq. realisation and tower, s10 thm:diagcent and
prop:orthospin, s13 prop:commutantsn, s16 prop:tear, s17 thm:closureunique, prop:groupred,
prop:several, thm:nocommon, thm:PSbinding) and the context sections d3/d4/d5, without importing or
copying the derive script d7_bfield_between.py.

Methods differ from the derive script where possible:
  * lG is computed as the centraliser of G in so(3,3) (a null space), lint from the bivectors of
    E=<e1,e4,e5,e6>; every commutant is a numerical null space, not a parametrisation;
  * vector actions of bivectors are extracted from the 8^n realisation by traces;
  * exact semisimplicity uses the Jordan rank test rank p(M) = rank p(M)^2 for every repeated
    irreducible factor p of the characteristic polynomial over QQ (DomainMatrix), not a square-free
    annihilation test; growth of ||exp(Mt)|| and direct ODE integration (solve_ivp) are used as
    independent numerical evidence;
  * the circulation numbers are obtained by integrating the real 18-dimensional frame flow;
  * the boundedness criterion of d7:thm:schr(vi) is also tested on cases the derive script does
    not use (a coincidence lambda = +-1 that is bounded, one that is not, and random compact b).
Claims found to be false are checked in their corrected form (see x7_findings.md); the
corresponding check ids end in 'N' (new finding).
Prints PASS/FAIL lines; exit 0 iff every check passes.
"""
import sys, time, itertools
import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

T0 = time.time()
FAILS = []
QQ = sp.QQ


import os
_TIMING = bool(os.environ.get("X7_TIME"))


def report(cid, ok, text):
    print(("PASS" if ok else "FAIL") + f" x7-{cid}: {text}" + (f"  [{time.time() - T0:.1f}s]" if _TIMING else ""),
          flush=True)
    if not ok:
        FAILS.append(cid)


# ----------------------------------------------------------------------------------------------
# Conventions (tex/s02_arena.tex): eta=diag(1,1,1,-1,-1,-1), G e_i=e_{i+3}, G e_{i+3}=-e_i
# ----------------------------------------------------------------------------------------------
eta1 = np.diag([1., 1, 1, -1, -1, -1])
G1 = np.zeros((6, 6))
for i in range(3):
    G1[i + 3, i] = 1.
    G1[i, i + 3] = -1.
I2 = np.eye(2)
sx = np.array([[0, 1], [1, 0.]])
sz = np.diag([1., -1])
sy = np.array([[0, -1j], [1j, 0]])
epsm = np.array([[0, -1], [1, 0.]])          # epsilon of d5: maps copy 1 -> copy 2


def kronn(*a):
    r = np.array([[1.]])
    for x in a:
        r = np.kron(r, x)
    return r


ecl = [kronn(sx, I2, I2), kronn(sz, sx, I2), kronn(sz, sz, sx),
       kronn(epsm, I2, I2), kronn(sz, epsm, I2), kronn(sz, sz, epsm)]
omega = ecl[0] @ ecl[1] @ ecl[2] @ ecl[3] @ ecl[4] @ ecl[5]


def tower(n):
    gens = []
    for k in range(n):
        for a in range(6):
            gens.append(kronn(*([omega] * k + [ecl[a]] + [np.eye(8)] * (n - k - 1))))
    return gens


def etan(n):
    return np.kron(np.eye(n), eta1)


def Gn(n):
    return np.kron(np.eye(n), G1)


def vec_action(X, gens, etad):
    """matrix of v -> [X, v] on the span of the generators, extracted by traces."""
    N = len(gens)
    D = gens[0].shape[0]
    A = np.zeros((N, N))
    for c in range(N):
        Y = X @ gens[c] - gens[c] @ X
        for d in range(N):
            A[d, c] = etad[d] * np.sum(gens[d].T * Y) / D
    return A


def Rab(a, b, N, etad):
    """R_ab = 1/2 ad(e_a e_b) on vectors: R e_b = eta_bb e_a, R e_a = -eta_aa e_b (0-based)."""
    R = np.zeros((N, N))
    R[a, b] = etad[b]
    R[b, a] = -etad[a]
    return R


def nullsp(Amat, tol=1e-9):
    """orthonormal basis (columns) of the null space; economy SVD when the matrix is tall."""
    m, nn = Amat.shape
    if m == 0:
        return np.eye(nn)
    if m >= nn:
        _, sv, Vh = np.linalg.svd(Amat, full_matrices=False)
    else:
        _, sv, Vh = np.linalg.svd(Amat, full_matrices=True)
        sv = np.concatenate([sv, np.zeros(nn - m)])
    smax = sv.max() if sv.size and sv.max() > 0 else 1.0
    return Vh[sv <= tol * smax].conj().T


def antisym_basis(N):
    B = []
    for i in range(N):
        for j in range(i + 1, N):
            A = np.zeros((N, N))
            A[i, j] = 1
            A[j, i] = -1
            B.append(A)
    return B


def commutant(gens, N, space="so", etam=None):
    """basis of {X in space : [X,g]=0 for g in gens}; space in {'so','end','euclid'}."""
    if space == "end":
        basis = []
        for i in range(N):
            for j in range(N):
                A = np.zeros((N, N))
                A[i, j] = 1
                basis.append(A)
    elif space == "so":
        basis = [etam @ A for A in antisym_basis(N)]
    elif space == "euclid":
        basis = antisym_basis(N)
    cols = []
    for Bm in basis:
        cols.append(np.concatenate([(Bm @ g - g @ Bm).ravel() for g in gens]) if gens else np.zeros(0))
    Amat = np.array(cols).T
    ns = nullsp(Amat) if gens else np.eye(len(basis))
    return [sum(c * Bm for c, Bm in zip(v, basis)) for v in ns.T]


def span_dim(mats, tol=1e-8):
    if not mats:
        return 0
    return np.linalg.matrix_rank(np.array([m.ravel() for m in mats]), tol=tol)


def same_span(A, B):
    return span_dim(A) == span_dim(B) == span_dim(list(A) + list(B))


# lG = centraliser of G in so(3,3) (d4:prop:lp); lint = bivectors of E=<e1,e4,e5,e6> (s02 sec:interior)
lG = commutant([G1], 6, "so", eta1)
lint = [Rab(a, b, 6, np.diag(eta1)) for a, b in itertools.combinations([0, 3, 4, 5], 2)]


def diag_action(alg, n):
    return [np.kron(np.eye(n), g) for g in alg]


def single_action(alg, n, k=0):
    P = np.zeros((n, n))
    P[k, k] = 1
    return [np.kron(P, g) for g in alg]


def Mof(S, n, m=1.0):
    N = 6 * n
    Sd = etan(n) @ S.T @ etan(n)
    return np.block([[-S, np.eye(N) / m], [np.zeros((N, N)), Sd]])


def growth_ratio(M, T=200.):
    a = np.linalg.norm(expm(M * T), 2)
    b = np.linalg.norm(expm(M * 10 * T), 2)
    return b / a, b


# ---------- exact helpers ----------
def to_dm(Mnum, den=1):
    """numeric matrix with entries that are integers/den -> DomainMatrix over QQ."""
    rows = [[QQ(int(round(x * den)), den) for x in r] for r in Mnum]
    return DomainMatrix(rows, Mnum.shape, QQ)


def sym_to_dm(Msym):
    return DomainMatrix.from_Matrix(sp.Matrix(Msym)).convert_to(QQ)


def poly_of_dm(coeffs, M):
    N = M.shape[0]
    Id = DomainMatrix.eye(N, QQ)
    P = DomainMatrix.zeros((N, N), QQ)
    for c in coeffs:
        P = P * M + Id * c
    return P


x = sp.symbols('x')


def jordan_semisimple(M):
    """exact: M semisimple iff rank p(M) == rank p(M)^2 for every repeated irreducible factor p."""
    cp = M.charpoly()
    poly = sp.Poly([sp.Rational(int(c.numerator), int(c.denominator)) for c in cp], x)
    _, facs = sp.factor_list(poly.as_expr())
    ok = True
    for f, mult in facs:
        if mult > 1:
            coeffs = sp.Poly(f, x).all_coeffs()
            P = poly_of_dm([QQ.from_sympy(sp.Rational(cc)) for cc in coeffs], M)
            if P.rank() != (P * P).rank():
                ok = False
    return ok, facs


# ==============================================================================================
# 1. d7:prop:exist
# ==============================================================================================
# sanity: d4's L_i, K_i lie in lG, and lG is 6-dimensional
ok = len(lG) == 6
for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    L = Rab(j + 3, k + 3, 6, np.diag(eta1)) - Rab(j, k, 6, np.diag(eta1))
    K = Rab(k, j + 3, 6, np.diag(eta1)) - Rab(j, k + 3, 6, np.diag(eta1))
    ok &= span_dim(lG + [L]) == 6 and span_dim(lG + [K]) == 6
report("0", ok, "conventions: lG = Cent(G) in so(3,3) is 6-dim and contains d4's L_i, K_i")

dims, cdims, ok_form = [], [], True
for n in range(1, 5):
    N = 6 * n
    C = commutant(diag_action(lG, n) + [Gn(n)], N, "so", etan(n))
    dims.append(len(C))
    # compact part: dimension of the Euclidean-antisymmetric subspace of span C
    if C:
        Sym = np.array([(X + X.T).ravel() for X in C]).T
        cdims.append(nullsp(Sym).shape[1])
    else:
        cdims.append(0)
    form = []
    for Bm in antisym_basis(n):
        form += [np.kron(Bm, np.eye(6)), np.kron(Bm, G1)]
    ok_form &= same_span(C, form) if C else (n == 1)
report("1a", dims == [0, 2, 6, 12] and cdims == [0, 1, 3, 6] and ok_form,
       "prop:exist(i): commutant of diag lG and G_n in so(3n,3n) = {B(x)1+B'(x)G} dims 0,2,6,12, compact parts 0,1,3,6")
cb = np.kron(epsm, G1)
report("1a2", np.allclose(cb @ cb, np.eye(12)) and np.allclose(cb, cb.T),
       "prop:exist(i): copy boost eps(x)G squares to +1 and is Euclidean symmetric (noncompact)")

dims2, ok2 = [], True
dims_lint_only = []
for n in range(1, 4):
    N = 6 * n
    C = commutant(diag_action(lint, n) + [Gn(n)], N, "so", etan(n))
    dims2.append(len(C))
    dims_lint_only.append(len(commutant(diag_action(lint, n), N, "so", etan(n))))
    rots = [np.kron(Bm, np.eye(6)) for Bm in antisym_basis(n)]
    ok2 &= same_span(C, rots) if C else (n == 1)
    if n >= 2:  # no copy boost commutes with diag lint
        for Bm in antisym_basis(n):
            Xb = np.kron(Bm, G1)
            ok2 &= any(np.abs(Xb @ g - g @ Xb).max() > 1e-9 for g in diag_action(lint, n))
report("1b", dims2 == [0, 1, 3] and ok2 and dims_lint_only == [1, 7, 18],
       "prop:exist(ii): diag lint & G_n -> copy rotations only (0,1,3); no copy boost commutes with lint; "
       "diag lint alone gives 1,7,18 (Prop. 13.14 row 2)")

# n=2 decomposition C^12 = M12 + C_surp and the sign of the paper's M_12
Pint = np.diag([1., 0, 0, 1, 1, 1])
Psur = np.diag([0., 1, 1, 0, 0, 0])
C12 = np.kron(epsm, np.eye(6))
M12d7 = np.kron(epsm, Pint)
Csur = np.kron(epsm, Psur)
E2 = tower(2)
eta2d = np.diag(etan(2))
M12paper = 0.5 * sum(eta1[a, a] * E2[a] @ E2[6 + a] for a in [0, 3, 4, 5])   # (1/2) sum eta^{mu mu} g1 g2
M12paper_vec = vec_action(M12paper, E2, eta2d)
ok = np.allclose(C12, M12d7 + Csur)
ok &= all(np.allclose(X @ g, g @ X) for X in [M12d7, Csur] for g in diag_action(lint, 2))
ok &= not np.allclose(M12d7 @ Gn(2), Gn(2) @ M12d7) and not np.allclose(Csur @ Gn(2), Gn(2) @ Csur)
report("1c", ok, "prop:exist(ii): C^12 = eps(x)P_int + eps(x)P_surp, both commute with diag lint, neither with G_2")
report("1cN", np.allclose(M12paper_vec, -M12d7),
       "sign: the paper's M_12 = 1/2 sum eta^{mu mu} g^(1)_mu g^(2)_mu acts on vectors as (E_12-E_21)(x)P_int = "
       "-eps(x)P_int, so C^12 = -M_12 + C_surp in the paper's sign")

ok, dims3 = True, []
for n in [2, 3]:
    N = 6 * n
    for alg in [lG, lint]:
        C = commutant(single_action(alg, n) + [Gn(n)], N, "so", etan(n))
        dims3.append(len(C))
        for X in C:
            ok &= np.abs(X[:6, :]).max() < 1e-9 and np.abs(X[:, :6]).max() < 1e-9
report("1d", dims3 == [6, 6, 30, 30] and ok,
       "prop:exist(iii): spectator lG^(1) or lint^(1) with G_n -> so(3n-3,C) of copies 2..n (6, 30), zero on and into copy 1")
# proof gap: under lint^(1) V^(1) is reducible and the commutant (without G) reaches into copy 1
ok = True
d_end = len(commutant(lint, 6, "end"))
dl = []
for n in [2, 3]:
    C = commutant(single_action(lint, n), 6 * n, "so", etan(n))
    dl.append(len(C))
    ok &= max(np.abs(X[6:, :6]).max() for X in C) > 0.1
report("1dN", d_end == 5 and dl == [28, 91] and ok,
       "proof of (iii): V is reducible under lint (End_lint(V) has dim 5, not 1); the commutant of lint^(1) in "
       "so(3n,3n) (dims 28, 91) has components into copy 1; only G_n removes them (Schur alone covers lG^(1) only)")

# ==============================================================================================
# 2. d7:prop:rotor and the paper's uniqueness argument
# ==============================================================================================
ok, dd = True, []
for n in [1, 2, 3]:
    N = 6 * n
    Cend = commutant(diag_action(lG, n), N, "end")
    Ceu = commutant(diag_action(lG, n), N, "euclid")
    dd.append((len(Cend), len(Ceu)))
    form_end = [np.kron(E, Y) for E in [np.eye(n)[:, [i]] @ np.eye(n)[[j], :] for i in range(n) for j in range(n)]
                for Y in [np.eye(6), G1]]
    ok &= same_span(Cend, form_end)
    so_part = [np.kron(Bm, np.eye(6)) for Bm in antisym_basis(n)]
    sym_part = []
    for i in range(n):
        for j in range(i, n):
            Sm = np.zeros((n, n))
            Sm[i, j] = Sm[j, i] = 1
            sym_part.append(np.kron(Sm, G1))
    ok &= same_span(Ceu, so_part + sym_part)
    en = etan(n)
    ok &= all(np.allclose(en @ X.T @ en, -X) for X in so_part)
    ok &= all(np.allclose(en @ X.T @ en, X) for X in sym_part)
    # centre of u(n): elements of u(n) commuting with all of u(n)
    cols = [np.concatenate([(X @ Y - Y @ X).ravel() for Y in Ceu]) for X in Ceu]
    cen_in = [sum(c * X for c, X in zip(v, Ceu)) for v in nullsp(np.array(cols).T).T]
    ok &= span_dim(cen_in) == 1 and span_dim(cen_in + [Gn(n)]) == 1
report("2a", dd == [(2, 1), (8, 4), (18, 9)] and ok,
       "prop:rotor: End_{diag lG}(V_n) = gl(n)(x){1,G} (2n^2); Euclidean-antisymmetric part u(n) = so(n)(x)1 + Sym(n)(x)G "
       "(n^2), eta-antisym + eta-selfadjoint; centre spanned by G_n")

# substitution X = exp(-theta A) x, theta = Omega tau  =>  exp(-theta A) xdot = Xdot + Omega A X
rng = np.random.default_rng(7)
n = 3
Bc = rng.normal(size=(3, 3)); Bc = Bc - Bc.T
Sc = rng.normal(size=(3, 3)); Sc = Sc + Sc.T
A = np.kron(Bc, np.eye(6)) + np.kron(Sc, G1)
Om = 0.7
Xf = lambda t: np.array([np.sin(t + q) for q in range(18)])
xf = lambda t: expm(Om * t * A) @ Xf(t)
t0, h = 0.4, 1e-5
xdot = (xf(t0 + h) - xf(t0 - h)) / (2 * h)
Xdot = (Xf(t0 + h) - Xf(t0 - h)) / (2 * h)
report("2b", np.allclose(expm(-Om * t0 * A) @ xdot, Xdot + Om * A @ Xf(t0), atol=1e-7),
       "prop:rotor: X = exp(-theta A) x gives exp(-theta A) xdot = Xdot + Omega A X, so S = Omega A")

# per-constituent action: Theorem 17.7 factor by factor already admits unequal rates
ok = True
dpc = []
for n in [2, 3]:
    N = 6 * n
    gens = []
    for k in range(n):
        gens += single_action(lG, n, k)
    Ceu = commutant(gens, N, "euclid")
    dpc.append(len(Ceu))
    diagK = []
    for k in range(n):
        P = np.zeros((n, n)); P[k, k] = 1
        diagK.append(np.kron(P, G1))
    ok &= same_span(Ceu, diagK)
# common action with a common rate does not select G_n either
A2 = Gn(2) + np.kron(epsm, np.eye(6))
inu2 = span_dim(commutant(diag_action(lG, 2), 12, "euclid") + [A2]) == 4
selfadj = 0.5 * (A2 + etan(2) @ A2.T @ etan(2))
report("2cN", dpc == [2, 3] and ok and inu2 and np.allclose(selfadj, Gn(2)) and not np.allclose(A2, Gn(2)),
       "per-constituent action: non-dilating rotors are exactly K(x)G with K diagonal arbitrary (unequal rates allowed); "
       "common action + common rate admits G_2 + eps(x)1 != G_2")

# complex cross product beta0 on V (s17 prop:lGhoms): z(X) x z(Y) returned to V
def zof(X):
    return X[:3] + 1j * X[3:]


def Vof(z):
    return np.concatenate([z.real, z.imag])


def beta0(X, Y):
    return Vof(np.cross(zof(X), zof(Y)))


Xr, Yr, Zr = rng.normal(size=6), rng.normal(size=6), rng.normal(size=6)
ok = all(np.allclose(g @ beta0(Xr, Yr), beta0(g @ Xr, Yr) + beta0(Xr, g @ Yr)) for g in lG)
ok &= np.allclose(beta0(G1 @ Xr, Yr), G1 @ beta0(Xr, Yr))
report("2d", ok, "beta0 (complex cross product) is lG-equivariant and G-bilinear: beta0(GX,Y) = G beta0(X,Y)")


def equivariant_brackets(n):
    """basis of lG-equivariant antisymmetric bilinear maps c: V_n x V_n -> V_n, as tensors c[a,i,j]."""
    N = 6 * n
    pairs = list(itertools.combinations(range(N), 2))
    nunk = len(pairs) * N
    cb = np.zeros((nunk, N, N, N))
    u = 0
    for (i, j) in pairs:
        for a in range(N):
            cb[u, a, i, j] = 1.0
            cb[u, a, j, i] = -1.0
            u += 1
    rows = []
    for g in diag_action(lG, n):
        rows.append(der_defect(cb, g).reshape(nunk, -1))
    R = np.concatenate(rows, axis=1)
    w, V = np.linalg.eigh(R @ R.T)
    ns = V[:, w < 1e-9 * w.max()]
    return np.einsum('uk,uaij->kaij', ns, cb)


def der_defect(c, g):
    """g c(X,Y) - c(gX,Y) - c(X,gY) for a batch of tensors c[u,a,i,j]."""
    return (np.einsum('ad,udij->uaij', g, c) - np.einsum('uadj,di->uaij', c, g)
            - np.einsum('uaid,dj->uaij', c, g))


def derivation_dim(basis, A):
    D = der_defect(basis, A).reshape(basis.shape[0], -1)
    return basis.shape[0] - np.linalg.matrix_rank(D, tol=1e-8)


EB2 = equivariant_brackets(2)
dfree = EB2.shape[0]
dG2 = derivation_dim(EB2, Gn(2))
A12 = np.kron(np.diag([1., 2]), G1)
A13 = np.kron(np.diag([1., 3]), G1)
d12 = derivation_dim(EB2, A12)
d13 = derivation_dim(EB2, A13)
dcr = derivation_dim(EB2, np.kron(epsm, np.eye(6)))
report("2e", dfree == 12 and dG2 == 0 and d13 == 0 and dcr == 0,
       "Jacobi (rotor,X,Y) under the common action: equivariant frame brackets on V_2 form a 12-dim space; "
       "none survive for A = G_2, diag(1,3)(x)G or the copy rotation eps(x)1, so [X,Y]=0 there")


def bracket12(X, Y):
    out = np.zeros(12)
    out[6:] = beta0(X[:6], Y[:6])
    return out


# full Jacobi of s = R ghat + V_2 with [ghat,X] = A12 X, [X,Y] = (0, beta0(X_1,Y_1))
def br(u, v):
    # elements as (t, X) with t the ghat coefficient
    t1, X1 = u
    t2, X2 = v
    return (0.0, t1 * (A12 @ X2) - t2 * (A12 @ X1) + bracket12(X1, X2))


ok = True
for _ in range(5):
    a, b, c = [(rng.normal(), rng.normal(size=12)) for _ in range(3)]
    J = [br(br(a, b), c), br(br(b, c), a), br(br(c, a), b)]
    ok &= abs(sum(j[0] for j in J)) < 1e-12 and np.allclose(sum(j[1] for j in J), 0, atol=1e-10)
    X, Y = rng.normal(size=12), rng.normal(size=12)
    for g in diag_action(lG, 2):
        ok &= np.allclose(g @ bracket12(X, Y), bracket12(g @ X, Y) + bracket12(X, g @ Y))
ok &= np.allclose(A12.T, -A12) and np.abs(bracket12(X, Y)).max() > 1e-3
report("2fN", d12 == 2 and ok,
       "REFUTES '[X,Y]=0' for non-central rotors: A = diag(1,2)(x)G in u(2) (rates 1:2) admits the lG-equivariant, "
       "non-dilating closure [ghat,X]=AX, [X,Y]=(0, beta0(X_1,Y_1)) (all Jacobi identities hold; 2-dim family)")


# the ghat-component channel: Hom_lG(Lambda^2 V_n, R) is nonzero at n=2 (Prop. 17.4 covers n=1 only)
def equivariant_forms(n):
    N = 6 * n
    basis = antisym_basis(N)
    gens = diag_action(lG, n)
    cols = [np.concatenate([(g.T @ W + W @ g).ravel() for g in gens]) for W in basis]
    return nullsp(np.array(cols).T).shape[1]


report("2gN", equivariant_forms(1) == 0 and equivariant_forms(2) == 2,
       "Hom_lG(Lambda^2 V_n, R) is 0 at n=1 but 2-dim at n=2 under the common action (eta and eta G between copies): "
       "the step 'the ghat-component vanishes by Prop. 17.4' of Thm 17.7 does not transfer to n>=2")

# ==============================================================================================
# 3. d7:thm:schr
# ==============================================================================================
def z_matrix(S, n):
    """complex 3n x 3n matrix of a G_n-commuting real S on z=(B+iE) coordinates (copy-major)."""
    N = 3 * n
    Zm = np.zeros((N, N), complex)
    for col in range(N):
        k, a = divmod(col, 3)
        X = np.zeros(6 * n); X[6 * k + a] = 1.0          # z = e_col (real)
        Y = S @ X
        Zm[:, col] = np.array([Y[6 * kk + aa] + 1j * Y[6 * kk + 3 + aa] for kk in range(n) for aa in range(3)])
    return Zm


n = 3
CentG3 = commutant([Gn(3)], 18, "so", etan(3))
ok = len(CentG3) == 72
coef = rng.normal(size=len(CentG3))
bgen = sum(c * X for c, X in zip(coef, CentG3))          # generic (noncompact) binding-compatible b
bcomp = 0.5 * (bgen - bgen.T)                             # its compact part (Euclidean antisymmetric)
ok &= np.allclose(bcomp @ Gn(3), Gn(3) @ bcomp) and np.allclose(etan(3) @ bcomp.T @ etan(3), -bcomp)
kap = np.array([0.7, 1.9, 3.1])
KG = np.kron(np.diag(kap), G1)
Kt = np.kron(np.diag(kap), np.eye(3))
okA = True
for b in [bgen, bcomp]:
    Zb = z_matrix(b, 3)
    Rr, Qr = Zb.real, Zb.imag
    okA &= np.allclose(Rr, -Rr.T) and np.allclose(Qr, -Qr.T)
    H = Kt - 1j * Rr + Qr
    S = KG + b
    okA &= np.allclose(z_matrix(S, 3), 1j * H)
    X0 = rng.normal(size=18)
    tau = 0.83
    Xt = expm(-S * tau) @ X0
    z0 = X0.reshape(3, 6)[:, :3].ravel() + 1j * X0.reshape(3, 6)[:, 3:].ravel()
    zt = Xt.reshape(3, 6)[:, :3].ravel() + 1j * Xt.reshape(3, 6)[:, 3:].ravel()
    okA &= np.allclose(zt, expm(-1j * H * tau) @ z0)
report("3a", ok and okA, "thm:schr(i): Cent(G_3) in so(9,9) is 72-dim; b acts on z as R+iQ (R,Q in so(9)); S acts as iH, "
       "z(tau) = exp(-iH tau) z(0) for compact and noncompact b")

# (ii) E6 = |X|^2/2 conserved for every initial value iff Q = 0
Zc = z_matrix(bcomp, 3)
Hc = Kt - 1j * Zc.real
Sg = KG + bgen
dE = lambda S, X: -X @ S @ X                            # d/dtau (1/2|X|^2) along Xdot=-SX
ok = np.allclose(Hc, Hc.conj().T) and np.abs(Zc.imag).max() < 1e-12
ok &= max(abs(dE(KG + bcomp, rng.normal(size=18))) for _ in range(5)) < 1e-10
ok &= max(abs(dE(Sg, rng.normal(size=18))) for _ in range(5)) > 1e-3
report("3b", ok, "thm:schr(ii): compact b gives Q=0, H Hermitian, |X|^2 conserved; generic noncompact b does not conserve it")

# (iii) rates are gauge invariants: Cayley curve U(tau) in O(9,9)
W0 = etan(3) @ (lambda A: A - A.T)(rng.normal(size=(18, 18))) * 0.2
W1 = etan(3) @ (lambda A: A - A.T)(rng.normal(size=(18, 18))) * 0.2
Id18 = np.eye(18)
ok = True
for b in [bcomp, bgen]:
    S = KG + b
    for tau in [0.0, 0.6, 1.3]:
        W = W0 + tau * W1
        U = np.linalg.solve(Id18 - W, Id18 + W)
        Ui = np.linalg.inv(U)
        Ud = np.linalg.solve(Id18 - W, W1 @ (Id18 + np.linalg.solve(Id18 - W, Id18 + W)))   # d/dtau Cayley
        ok &= np.allclose(U.T @ etan(3) @ U, etan(3))
        SU = Ui @ S @ U + Ui @ Ud
        sa = 0.5 * (SU + etan(3) @ SU.T @ etan(3))
        anti = 0.5 * (SU - etan(3) @ SU.T @ etan(3))
        ev = np.sort(np.linalg.eigvals(sa).imag)
        ok &= np.allclose(ev, np.sort(np.concatenate([kap, -kap]).repeat(3)), atol=1e-8)
        ok &= np.allclose(sa, Ui @ KG @ U)
report("3c", ok, "thm:schr(iii): under eta-isometric frame changes on a Cayley curve the eta-selfadjoint part is "
       "U^-1 (K(x)G) U, spectrum {+-i kappa_k} unchanged (compact and noncompact b)")

# (iv) sum rules and spread
ev0 = np.linalg.eigvalsh(Kt)
ev1 = np.linalg.eigvalsh(Hc)
Rn = np.linalg.norm(Zc.real) ** 2
ok = np.isclose(np.trace(Hc).real, 3 * kap.sum()) and np.isclose(np.trace(Hc @ Hc).real, 3 * (kap ** 2).sum() + Rn)
ok &= ev1.var() > ev0.var() and ev1.min() <= ev0.min() + 1e-12 and ev1.max() >= ev0.max() - 1e-12
# caveat: an individual gap can shrink (rates 1,2,4 with the copy field (1/2,1/3,1/5))
def Bm3(b12, b23, b31):
    return np.array([[0, b12, -b31], [-b12, 0, b23], [b31, -b23, 0.]])


evg = np.linalg.eigvalsh(np.diag([1., 2, 4]) - 1j * Bm3(.5, 1 / 3, .2))
report("3d", ok and (evg[2] - evg[1]) < 2.0,
       "thm:schr(iv): tr H = 3 sum kappa, tr H^2 = 3 sum kappa^2 + ||R||^2; variance and range widen "
       "(but an individual gap can shrink: rates (1,2,4) upper gap 2 -> %.3f)" % (evg[2] - evg[1]))

# (v) removable dimensions at n=3
compact = commutant([Gn(3), etan(3)], 18, "euclid")      # Euclidean- and eta-antisymmetric
dimc = span_dim(compact)
rem = []
copyrem = []
for k3 in [(1., 1, 1), (1., 1, 2.5), (1., 2, 3.7)]:
    KGx = np.kron(np.diag(k3), G1)
    C = commutant([Gn(3), etan(3), KGx], 18, "euclid")
    rem.append(span_dim(C))
    copyrem.append(nullsp(np.array([(np.kron(Bm, np.eye(6)) @ KGx - KGx @ np.kron(Bm, np.eye(6))).ravel()
                                    for Bm in antisym_basis(3)]).T).shape[1])
report("3e", dimc == 36 and rem == [36, 18, 9] and copyrem == [3, 1, 0],
       "thm:schr(v): compact binding-compatible fields 36-dim; removable 36/18/9 for rates (k,k,k)/(k,k,k')/distinct; "
       "in copy so3 3/1/0")


# (vi) criterion vs exact Jordan test vs growth
def crit_ok(H, tol=1e-7):
    w, V = np.linalg.eigh(H)
    worst = 0.0
    for i in range(len(w)):
        for j in range(len(w)):
            if abs(w[i] + w[j]) < tol:
                Ei = V[:, np.abs(w - w[i]) < tol]
                Ej = V[:, np.abs(w - w[j]) < tol]
                worst = max(worst, np.abs(Ei.T @ Ej).max())
    return worst < 1e-6


def copyfield_S_exact(kq, bq):
    """exact S = K(x)G + B(x)1_6 as DomainMatrix (entries rational)."""
    K = sp.diag(*kq)
    b12, b23, b31 = bq
    B = sp.Matrix([[0, b12, -b31], [-b12, 0, b23], [b31, -b23, 0]])
    Gs = sp.Matrix(G1.astype(int))
    S = sp.kronecker_product(K, Gs) + sp.kronecker_product(B, sp.eye(6))
    return S


def M_exact(S, n):
    N = 6 * n
    et = sp.kronecker_product(sp.eye(n), sp.diag(1, 1, 1, -1, -1, -1))
    Sd = et * S.T * et
    M = sp.zeros(2 * N, 2 * N)
    M[:N, :N] = -S
    M[:N, N:] = sp.eye(N)
    M[N:, N:] = Sd
    return M


R_ = sp.Rational
cases = [((1, 2, 4), (R_(1, 2), R_(1, 3), R_(1, 5)), True, "rates (1,2,4), B=(1/2,1/3,1/5)"),
         ((0, 3, 5), (1, R_(1, 2), R_(1, 3)), True, "seesaw (0,3,5), B=(1,1/2,1/3)"),
         ((1, 1, 1), (1, 0, 0), True, "equal rates, B12=1"),
         ((1, 2, 3), (1, 1, 1), False, "(1,2,3), B=(1,1,1): spec H={0,2,4}"),
         ((0, 0, 0), (1, 1, 1), False, "zero rates, B=(1,1,1)"),
         ((1, 1, 1), (R_(4, 3), R_(4, 3), R_(2, 3)), True, "NEW: equal rates, |B|=2: spec H={-1,1,3}, bounded"),
         ((1, 2, 0), (1, 1, 1), False, "NEW: (1,2,0), B=(1,1,1): spec H={-1,1,3}, unbounded")]
ok = True
details = []
for kq, bq, bounded, name in cases:
    S = copyfield_S_exact(kq, bq)
    M = M_exact(S, 3)
    ss, _ = jordan_semisimple(sym_to_dm(M))
    Snum = np.array(S.tolist(), dtype=float)
    H = np.kron(np.diag(np.array(kq, float)) - 1j * Bm3(*[float(v) for v in bq]), np.eye(3))
    cr = crit_ok(H)
    gr, _ = growth_ratio(Mof(Snum, 3))
    grows = gr > 3
    ok &= (ss == bounded) and (cr == bounded) and (grows == (not bounded))
    details.append(name)
report("3f", ok, "thm:schr(vi): exact Jordan test, bilinear criterion and growth of ||e^{Mt}|| agree on the five cases of "
       "the text and two new ones with a coincidence lambda=+-1 (one bounded, one not)")
# n=3 spectrum claim (1,2,3),(1,1,1)
Hq = sp.diag(1, 2, 3) - sp.I * sp.Matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
ev = sorted([sp.nsimplify(v) for v in Hq.eigenvals()])
v0 = (Hq).nullspace()[0]
report("3f2", ev == [0, 2, 4] and sp.simplify((v0.T * v0)[0]) != 0,
       "thm:schr(vi) text: (1,2,3), B=(1,1,1) has spec H = {0,2,4} and zero mode with v.v != 0")

# random compact b (general R in so(9)) tuned to a coincidence lambda_i + lambda_j = 0
ok = True
ntest = 0
for trial in range(12):
    Rg = rng.normal(size=(9, 9)); Rg = Rg - Rg.T
    kq = rng.uniform(0.2, 2.0, 3)
    Ktr = np.kron(np.diag(kq), np.eye(3))

    def fpair(s):
        w = np.linalg.eigvalsh(Ktr - 1j * s * Rg)
        return min((w[i] + w[j] for i in range(9) for j in range(i, 9)), key=abs)

    ss_ = np.linspace(0.01, 3, 60)
    vals = [fpair(s) for s in ss_]
    for i in range(len(ss_) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]):
            try:
                s0 = brentq(fpair, ss_[i], ss_[i + 1], xtol=1e-15)
            except ValueError:
                continue
            if abs(fpair(s0)) > 1e-10:
                continue
            H = Ktr - 1j * s0 * Rg
            # real S from H: S acts on z as iH
            A = 1j * H
            Sreal = np.zeros((18, 18))
            iB = [6 * k + a for k in range(3) for a in range(3)]
            iE = [6 * k + 3 + a for k in range(3) for a in range(3)]
            Sreal[np.ix_(iB, iB)] = A.real; Sreal[np.ix_(iB, iE)] = -A.imag
            Sreal[np.ix_(iE, iB)] = A.imag; Sreal[np.ix_(iE, iE)] = A.real
            cr = crit_ok(H, tol=1e-8)
            gr, _ = growth_ratio(Mof(Sreal, 3), T=100.)
            ok &= (cr == (gr < 3))
            ntest += 1
            break
    if ntest >= 4:
        break
report("3g", ok and ntest >= 3, f"thm:schr(vi) on {ntest} random compact b (general R in so(9)) tuned to a coincidence: "
       "criterion <=> bounded growth")

# n=2 remark: coincidences are lambda_- = 0 (beta^2 = k1 k2) when k1+k2>0, but lambda_+ = -lambda_- when k1=k2=0
def H2(k1, k2, b):
    return np.array([[k1, 1j * b], [-1j * b, k2]])


w = np.linalg.eigvalsh(H2(0, 0, 0.7))
report("3hN", np.isclose(w[0], -w[1]) and not crit_ok(np.kron(H2(0, 0, .7), np.eye(3))),
       "n=2 remark: at k1=k2=0 the coincidence is lambda_+ = -lambda_- = beta (not lambda_- = 0); "
       "'the only coincidence is lambda_-=0' needs k1+k2>0")

# ==============================================================================================
# 4. d7:prop:boost
# ==============================================================================================
k1, k2, bp = sp.symbols('k1 k2 bp', real=True)
Hb = sp.Matrix([[k1, -bp], [bp, k2]])
evs = list(Hb.eigenvals().keys())
target = [(k1 + k2) / 2 + sp.sqrt((k1 - k2) ** 2 / 4 - bp ** 2), (k1 + k2) / 2 - sp.sqrt((k1 - k2) ** 2 / 4 - bp ** 2)]
okf = all(any(sp.simplify(e - t) == 0 for t in target) for e in evs)
# check H = K + beta' eps from the real S
Sb = np.kron(np.diag([1., 4]), G1) + 0.5 * np.kron(epsm, G1)
okf &= np.allclose(z_matrix(Sb, 2), 1j * np.kron(np.diag([1., 4]) + 0.5 * epsm, np.eye(3)))
report("4a", okf, "prop:boost: b = beta' eps(x)G acts with H = K + beta' eps; eigenvalues (k1+k2)/2 +- sqrt((k1-k2)^2/4 - beta'^2)")


def boost_S_exact(kq, b):
    Gs = sp.Matrix(G1.astype(int))
    return sp.kronecker_product(sp.diag(*kq), Gs) + b * sp.kronecker_product(sp.Matrix([[0, -1], [1, 0]]), Gs)


ok = True
for b in [R_(1, 2), 1, R_(6, 5)]:
    S = boost_S_exact((1, 4), b)
    M = M_exact(S, 2)
    ss, _ = jordan_semisimple(sym_to_dm(M))
    Mn = np.array(M.tolist(), dtype=float)
    gr, _ = growth_ratio(Mn)
    evM = np.linalg.eigvals(Mn)
    ok &= ss and gr < 3 and np.abs(evM.real).max() < 1e-8
    # conserved positive form z^dag Q z with H^dag Q = Q H, Q != multiple of 1
    H = np.array([[1, -float(b)], [float(b), 4]])
    wv, Vv = np.linalg.eig(H)
    Vi = np.linalg.inv(Vv)
    Qf = Vi.conj().T @ Vi
    ok &= np.allclose(H.conj().T @ Qf, Qf @ H) and np.all(np.linalg.eigvalsh(Qf) > 0)
    ok &= not np.allclose(Qf / Qf[0, 0], np.eye(2))
    sol = solve_ivp(lambda t, z: -1j * H @ z, [0, 7.3], np.array([1.0, 0.3j]), rtol=1e-11, atol=1e-12)
    zt = sol.y[:, -1]
    z0 = np.array([1.0, 0.3j])
    ok &= np.isclose((zt.conj() @ Qf @ zt).real, (z0.conj() @ Qf @ z0).real, rtol=1e-8)
    ok &= not np.isclose(np.vdot(zt, zt).real, np.vdot(z0, z0).real, rtol=1e-4)
report("4b", ok, "prop:boost(i): rates (1,4), beta'=1/2, 1, 6/5: M exactly semisimple, imaginary spectrum, bounded; "
       "a positive Q with H^dag Q = QH is conserved (ODE) while |z|^2 is not")
S = boost_S_exact((1, 4), R_(3, 2))
ssEP, _ = jordan_semisimple(sym_to_dm(M_exact(S, 2)))
S2 = np.array(boost_S_exact((1, 4), 2).tolist(), dtype=float)
evM = np.linalg.eigvals(Mof(S2, 2))
re = np.unique(np.round(np.abs(evM.real), 9))
S3 = np.array(boost_S_exact((2, 2), R_(1, 3)).tolist(), dtype=float)
bb = (1 / 3) * np.kron(epsm, G1)
KG2 = np.kron(2 * np.eye(2), G1)
gr3, nrm3 = growth_ratio(Mof(S3, 2), T=20.)
lam_max = np.abs(np.linalg.eigvals(Mof(S3, 2)).real).max()
ok3 = np.allclose(bb @ KG2, KG2 @ bb) and np.isclose(lam_max, 1 / 3) and gr3 > 100 and np.linalg.norm(expm(-bb * 30), 2) > 1e3
# removal: exp(-b tau) intertwines (d/dtau + S) with (d/dtau + K(x)G)
tau = 0.9
okr = np.allclose(expm(bb * tau) @ (S3 @ expm(-bb * tau)) - bb, KG2)
report("4c", (not ssEP) and np.allclose(re, [np.sqrt(7) / 2]) and ok3 and okr,
       "prop:boost(ii)-(iii): at beta'=3/2 M is not semisimple; at beta'=2 every eigenvalue of M has real part +-sqrt(7)/2; at rates (2,2), "
       "beta'=1/3 b commutes with K(x)G, growth rate 1/3, removable by the unbounded frame exp(-b tau)")
# comparison with Prop. 16.7 (pairing boost): Jordan blocks strictly below threshold
c_, kk = 0.5, 1.0
sig = np.array([[0, 1], [1, 0.]]); Gb = np.array([[0, -1], [1, 0.]])
Mpair = np.block([[-c_ * sig - kk * Gb, np.eye(2)], [np.zeros((2, 2)), -c_ * sig + kk * Gb]])
Nil = Mpair @ Mpair - (c_ ** 2 - kk ** 2) * np.eye(4)
report("4dN", np.linalg.matrix_rank(Nil, tol=1e-10) > 0 and np.allclose(Nil @ Nil, 0),
       "caveat on 'copy analogue of the pairing boosts of Prop. 16.7': the pairing boost has Jordan blocks below its "
       "threshold (M^2-(c^2-k^2) nilpotent, nonzero), the copy boost is bounded below its threshold")

# ==============================================================================================
# 5. d7:thm:three
# ==============================================================================================
xs, ka, kb, kc, B12, B23, B31 = sp.symbols('x ka kb kc B12 B23 B31', real=True)
Ks = sp.diag(ka, kb, kc)
Bs = sp.Matrix([[0, B12, -B31], [-B12, 0, B23], [B31, -B23, 0]])
Hs = Ks - sp.I * Bs
lhs = sp.expand((xs * sp.eye(3) - Ks + sp.I * Bs).det())
rhs = sp.expand(xs ** 3 - (ka + kb + kc) * xs ** 2 + (ka * kb + ka * kc + kb * kc - B12 ** 2 - B23 ** 2 - B31 ** 2) * xs
                - (ka * kb * kc - kc * B12 ** 2 - ka * B23 ** 2 - kb * B31 ** 2))
report("5a", sp.simplify(lhs - rhs) == 0, "thm:three(i): det(x-K+iB) formula; couplings enter only through squares")
J = B12 * B23 * B31
plaq = sp.expand(Hs[0, 1] * Hs[1, 2] * Hs[2, 0])
comm = Hs * Ks - Ks * Hs
ok = sp.simplify(plaq - sp.I * J) == 0
ok &= sp.simplify(sp.expand(comm.det()) - sp.expand(2 * sp.I * J * (ka - kb) * (ka - kc) * (kb - kc))) == 0
# per-copy rephasings (duality rotations e^{theta_k G} on copy k) leave the plaquette invariant
th = sp.symbols('t1 t2 t3', real=True)
D = sp.diag(*[sp.exp(sp.I * t) for t in th])
Hp = D.inv() * Hs * D
ok_inv = sp.simplify(sp.expand(Hp[0, 1] * Hp[1, 2] * Hp[2, 0]) - plaq) == 0
ok_move = sp.simplify(Hp[0, 1] - Hs[0, 1]) != 0
report("5b", ok, "thm:three(ii): plaquette H12 H23 H31 = iJ, det[H,K] = 2iJ(k1-k2)(k1-k3)(k2-k3)")
report("5bN", ok_inv and ok_move,
       "plaquette phase: per-copy duality rotations change each H_kl but leave H12 H23 H31 invariant (a Wilson loop), "
       "so they could not 'move' the phase; it is +-pi/2 because the compact field is real")

Kd = np.diag([0.6, 1.7, 2.9])
KGd = np.kron(Kd, G1)
C = commutant(diag_action(lG, 3) + [Gn(3), KGd], 18, "end")
form = []
for k in range(3):
    P = np.zeros((3, 3)); P[k, k] = 1
    form += [np.kron(P, np.eye(6)), np.kron(P, G1)]
a_, b_ = sp.symbols('a b', real=True)
Gs = sp.Matrix(G1.astype(int)); etas = sp.diag(1, 1, 1, -1, -1, -1)
Lm = a_ * sp.eye(6) + b_ * Gs
eqs = list(set(sp.expand(e) for e in (Lm.T * etas * Lm - etas)))
sols = sp.solve([e for e in eqs if e != 0], [a_, b_], dict=True)
okiso = sorted([(s[a_], s[b_]) for s in sols]) == [(-1, 0), (1, 0)]
okrefl = True
Bn = Bm3(0.3, -0.8, 1.1)
for s in itertools.product([1, -1], repeat=3):
    Sg_ = np.kron(np.diag(s), np.eye(6))
    Bt = np.diag(s) @ Bn @ np.diag(s)
    okrefl &= np.allclose(Bt, np.outer(s, s) * Bn)
    okrefl &= np.isclose(Bt[0, 1] * Bt[1, 2] * Bt[2, 0], Bn[0, 1] * Bn[1, 2] * Bn[2, 0])
report("5c", same_span(C, form) and len(C) == 6 and okiso and okrefl,
       "thm:three(iii): commutant of G_3, diag lG, K(x)G (distinct rates) = sum_k (a_k+b_k G); eta-isometries: a=+-1, b=0 "
       "(8 reflections); B_kl -> s_k s_l B_kl, J invariant")

# (iv) P and T
Pm = np.kron(np.eye(3), np.diag([1., 1, 1, -1, -1, -1]))
Tm = -Pm
Bt = Bm3(0.4, 0.9, -0.5)
Kv = np.diag([0.5, 1.3, 2.2])
SKB = lambda K, B: np.kron(K, G1) + np.kron(B, np.eye(6))
ok = np.allclose(Pm @ Gn(3) @ Pm, -Gn(3))
ok &= np.allclose(Pm @ SKB(Kv, Bt) @ Pm, SKB(-Kv, Bt))
ok &= np.allclose(-Tm @ SKB(Kv, Bt) @ np.linalg.inv(Tm), SKB(Kv, -Bt))
ok &= np.allclose(Pm.T @ etan(3) @ Pm, etan(3))
# direct ODE: X'(t) = T X(-t), pi'(t) = -T pi(-t) solves the (K,-B) system
S1 = SKB(Kv, Bt); S2 = SKB(Kv, -Bt)
M1 = Mof(S1, 3); M2 = Mof(S2, 3)
y0 = rng.normal(size=36)
TT = np.block([[Tm, np.zeros((18, 18))], [np.zeros((18, 18)), -Tm]])
for t in [0.5, 1.7]:
    yback = expm(-M1 * t) @ y0                       # (X,pi) at time -t for system 1
    yprime = expm(M2 * t) @ (TT @ y0)                # system 2 started at T-image of y(0)
    ok &= np.allclose(yprime, TT @ yback)
report("5d", ok, "thm:three(iv): P G P^-1 = -G, P: (K,B)->(-K,B); interior time reversal X(t)->TX(-t), pi(t)->-T pi(-t) "
       "maps solutions of (K,B) to solutions of (K,-B) (checked on the full (X,pi) flow)")


# bipartite criterion by brute force
def bipartite(nv, edges):
    col = {}
    for st in range(nv):
        if st in col:
            continue
        col[st] = 0
        stack = [st]
        while stack:
            u = stack.pop()
            for (a, b) in edges:
                for (p, q) in [(a, b), (b, a)]:
                    if p == u:
                        if q not in col:
                            col[q] = 1 - col[u]; stack.append(q)
                        elif col[q] == col[u]:
                            return False
    return True


ok = True
for nv in [2, 3, 4, 5]:
    alle = list(itertools.combinations(range(nv), 2))
    for r in range(1, len(alle) + 1):
        for E in itertools.combinations(alle, r):
            ex = any(all(s[a] * s[b] == -1 for a, b in E) for s in itertools.product([1, -1], repeat=nv))
            ok &= ex == bipartite(nv, E)
report("5e", ok, "thm:three(iv): (K,B) ~ (K,-B) by a copy reflection iff the coupling graph is bipartite (all graphs n<=5)")


# (v) circulation, via the real 18-dim frame flow
def copy_energies(Kv, Bv, tau):
    S = SKB(Kv, Bv)
    X0 = np.zeros(18); X0[0] = 1.0
    sol = solve_ivp(lambda t, X: -S @ X, [0, tau], X0, rtol=1e-12, atol=1e-13)
    X = sol.y[:, -1].reshape(3, 6)
    return (X ** 2).sum(axis=1) / (X0 ** 2).sum()


Kc = np.diag([1., 2, 4]); Bc1 = Bm3(1, 1, 1)
Pp = copy_energies(Kc, Bc1, 0.5); Pm_ = copy_energies(Kc, -Bc1, 0.5)
# P_k(tau;-B) = P_k(-tau;B): integrate backwards
S = SKB(Kc, Bc1)
X0 = np.zeros(18); X0[0] = 1.0
Xm = (expm(S * 0.5) @ X0).reshape(3, 6)
Pback = (Xm ** 2).sum(axis=1)
ok = np.allclose(Pm_, Pback, atol=1e-9) and np.isclose(Pp[0], Pm_[0], atol=1e-9)
ok &= round(Pp[1], 3) == 0.293 and round(Pm_[1], 3) == 0.116 and not np.isclose(Pp[2], Pm_[2])
report("5f", ok, "thm:three(v): P_k(tau;-B) = P_k(-tau;B), P1 equal, P2 = %.4f vs %.4f, P3 = %.4f vs %.4f at (1,2,4), B=1, tau=1/2"
       % (Pp[1], Pm_[1], Pp[2], Pm_[2]))

# ==============================================================================================
# 6. d7:prop:isospin
# ==============================================================================================
E2 = tower(2)
ok = True
spec_txt = []
for D, expect, sgn in [([0, 1, 2, 3, 4, 5], np.arange(-3, 4), 1), ([0, 3, 4, 5], np.arange(-2, 3), 1),
                       ([0, 1, 2], np.array([-1.5, -.5, .5, 1.5]), -1)]:
    L = -0.5 * sum(eta1[a, a] * E2[a] @ E2[6 + a] for a in D)
    ev = np.unique(np.round(np.linalg.eigvals(L).imag, 8))
    ok &= np.allclose(ev, expect)
    ok &= np.allclose(expm(2 * np.pi * L), sgn * np.eye(64), atol=1e-8)
    vec = vec_action(L, E2, eta2d)
    Pd = np.diag([1. if a in D else 0. for a in range(6)])
    ok &= np.allclose(vec, np.kron(epsm, Pd))
    commutesG = np.allclose(vec @ Gn(2), Gn(2) @ vec)
    ok &= commutesG == (len(D) == 6)          # neither the temporal nor the interior (M_12) rotation commutes
# all subsets: binding-compatible iff closed under partners; then |D| even and exp(2 pi lift) = +1
ok_all = True
for r in range(1, 7):
    for D in itertools.combinations(range(6), r):
        Pd = np.diag([1. if a in D else 0. for a in range(6)])
        vec = np.kron(epsm, Pd)
        bc = np.allclose(vec @ Gn(2), Gn(2) @ vec)
        closed = all(((a + 3) % 6) in D for a in D)
        L = -0.5 * sum(eta1[a, a] * E2[a] @ E2[6 + a] for a in D)
        ex = expm(2 * np.pi * L)
        ok_all &= (bc == closed) and (not bc or len(D) % 2 == 0)
        ok_all &= np.allclose(ex, (-1) ** len(D) * np.eye(64), atol=1e-8)
report("6a", ok and ok_all, "prop:isospin(i): lift -1/2 sum eta_aa e^(k)_a e^(l)_a acts as the copy rotation of D; spectra "
       "i{-3..3}, i{-2..2}, i{+-1/2,+-3/2}; exp(2 pi .) = (-1)^|D|; binding-compatible <=> D closed under partners (|D| even)")

E3 = tower(3)
eta3d = np.diag(etan(3))
Ls = {}
for (k, l) in [(0, 1), (1, 2), (2, 0)]:
    Ls[(k, l)] = -0.5 * sum(eta1[a, a] * E3[6 * k + a] @ E3[6 * l + a] for a in range(6))
L12, L23, L31 = Ls[(0, 1)], Ls[(1, 2)], Ls[(2, 0)]
okc = np.allclose(L12 @ L23 - L23 @ L12, -L31) or np.allclose(L12 @ L23 - L23 @ L12, L31)
Cas = -(L12 @ L12 + L23 @ L23 + L31 @ L31)
evC = np.round(np.linalg.eigvalsh(0.5 * (Cas + Cas.T)), 6)
vals, cnts = np.unique(evC, return_counts=True)
okm = list(vals) == [0, 2, 6, 12] and list(cnts) == [40, 216, 200, 56]
ok2pi = np.allclose(expm(2 * np.pi * L12), np.eye(512), atol=1e-7)
vecs = [np.kron(Bm, np.eye(6)) for Bm in [np.eye(3)[:, [1]] @ np.eye(3)[[0], :] - np.eye(3)[:, [0]] @ np.eye(3)[[1], :]]]
v12 = vec_action(L12, E3, eta3d)
CasV = -(sum(vec_action(L, E3, eta3d) @ vec_action(L, E3, eta3d) for L in [L12, L23, L31]))
report("6b", okc and okm and ok2pi and np.allclose(CasV, 2 * np.eye(18)) and np.allclose(v12, vecs[0]),
       "prop:isospin(ii): n=3 binding copy rotations close on so3; Casimir 0,2,6,12 on 40,216,200,56; exp(2 pi C12)=+1; "
       "frames are j=1")

# (iii) doublet
gens_u2 = {"1": Gn(2), "x": np.kron(sx, G1), "y": -np.kron(epsm, np.eye(6)), "z": np.kron(sz, G1)}
cmat = {"1": 1j * np.eye(2), "x": 1j * sx, "y": 1j * sy, "z": 1j * sz}
ok = True
for key, Xg in gens_u2.items():
    ok &= np.allclose(z_matrix(Xg, 2), np.kron(cmat[key], np.eye(3)))
    ok &= np.allclose(Xg.T, -Xg)                          # preserves E6+E6
    ok &= np.allclose(Xg @ Gn(2), Gn(2) @ Xg)
    ok &= all(np.allclose(Xg @ g, g @ Xg) for g in diag_action(lG, 2))
    adj = etan(2) @ Xg.T @ etan(2)
    ok &= np.allclose(adj, -Xg) if key == "y" else np.allclose(adj, Xg)
kk = 1.3
for _ in range(3):
    bx, by, bz = rng.normal(size=3)
    S = kk * Gn(2) + bx * gens_u2["x"] + by * gens_u2["y"] + bz * gens_u2["z"]
    lines = np.sort(np.abs(np.linalg.eigvals(S).imag))
    nb = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    ok &= np.allclose(np.unique(np.round(lines, 9)), np.sort(np.round([abs(kk - nb), kk + nb], 9)))
    sa = 0.5 * (S + etan(2) @ S.T @ etan(2))
    rates = np.sort(np.abs(np.linalg.eigvals(sa).imag))
    nr = np.sqrt(bx ** 2 + bz ** 2)
    ok &= np.allclose(np.unique(np.round(rates, 9)), np.sort(np.round([abs(kk - nr), kk + nr], 9)))
report("6c", ok, "prop:isospin(iii): u(2) real forms; only i sigma_y (= -eps(x)1) eta-antisymmetric; lines k+-|beta|, "
       "rates k+-sqrt(bx^2+bz^2)")

# 'doublets only under the spectator action': n=4, lG diagonal on copies 1,2, copies 3,4 free
P12 = np.diag([1., 1, 0, 0])
gens = [np.kron(P12, g) for g in lG] + [Gn(4)]
C = commutant(gens, 24, "so", etan(4))
Sym = np.array([(X + X.T).ravel() for X in C]).T
dcompact = nullsp(Sym).shape[1]
# an su2 weight-1/2 generator on copies 3,4: T = 1/2(rot(a,b)+rot(c,d)) with temporal a,b,c,d of copies 3,4
eta4 = np.diag(etan(4))
def rot(i, j, N, etad):  # 1-based temporal indices, partner +3; acts by 1/2 ad
    return Rab(i + 2, j + 2, N, etad) - Rab(i - 1, j - 1, N, etad)
Tq = 0.5 * (rot(13, 19, 24, eta4) + rot(14, 20, 24, eta4))
inC = span_dim(C + [Tq]) == len(C)
evT = np.unique(np.round(np.linalg.eigvals(Tq).imag, 8))
c12 = np.kron(np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.]]), np.eye(6))
inC12 = span_dim(C + [c12]) == len(C)
report("6dN", len(C) == 32 and dcompact == 16 and inC and np.allclose(evT, [-0.5, 0, 0.5]) and inC12,
       "'doublets only under the spectator action' fails at n=4: with lG diagonal on copies 1,2 (coupled by C^12) and "
       "copies 3,4 free, the binding-compatible algebra so(2,C)+so(6,C) (32-dim, compact 16) contains weight-1/2 generators")

# ==============================================================================================
# 7. d7:prop:so4unequal
# ==============================================================================================
eta3d = np.diag(etan(3))
rotv = lambda i, j: rot(i, j, 18, eta3d)
Tp = {1: 0.5 * (rotv(13, 14) + rotv(7, 15)), 2: 0.5 * (rotv(7, 14) - rotv(13, 15)), 3: 0.5 * (rotv(7, 13) + rotv(14, 15))}
Tm_ = {1: 0.5 * (rotv(13, 14) - rotv(7, 15)), 2: 0.5 * (rotv(7, 14) + rotv(13, 15)), 3: 0.5 * (rotv(7, 13) - rotv(14, 15))}
# cross-check the 1/2 ad convention against the 512-dim realisation for one generator
r713 = vec_action(0.5 * (E3[9] @ E3[15] - E3[6] @ E3[12]), E3, eta3d)
ok = np.allclose(r713, rotv(7, 13))
br_ = lambda A, B: A @ B - B @ A
for T in [Tp, Tm_]:
    ok &= np.allclose(br_(T[1], T[2]), -T[3]) and np.allclose(br_(T[2], T[3]), -T[1]) and np.allclose(br_(T[3], T[1]), -T[2])
ok &= all(np.allclose(br_(Tp[i], Tm_[j]), 0) for i in range(1, 4) for j in range(1, 4))
ok &= np.allclose(np.sort(np.linalg.eigvals(Tp[3]).imag), np.sort([-.5] * 4 + [0] * 10 + [.5] * 4))
W = [7, 13, 14, 15]
so4 = [rotv(i, j) for i, j in itertools.combinations(W, 2)]
ok &= all(np.allclose(X @ Gn(3), Gn(3) @ X) for X in so4)
report("7a0", ok, "so4_Delta conventions: rot(i,j) = 1/2 ad(e_i'e_j' - e_ie_j) (checked on R^512); T^+-_i obey "
       "[T1,T2]=-T3 cyclically, [T+,T-]=0; T^+_3 eigenvalues +-i/2 (x4), 0 (x10)")

ok = True
for kq in [(5., 3, 5), (0.2, 1.1, 2.9), (4., 2, 2.5)]:
    KGq = np.kron(np.diag(kq), G1)
    cols = np.array([(X @ KGq - KGq @ X).ravel() for X in so4]).T
    ns = nullsp(cols)
    sub = [sum(c * X for c, X in zip(v, so4)) for v in ns.T]
    ok &= len(sub) == 3 and same_span(sub, [rotv(13, 14), rotv(13, 15), rotv(14, 15)])
ok &= same_span([rotv(13, 14), rotv(13, 15), rotv(14, 15)], [Tp[1] + Tm_[1], Tp[2] - Tm_[2], Tp[3] - Tm_[3]])
# graph of the isomorphism phi: T+_1 -> T-_1, T+_2 -> -T-_2, T+_3 -> -T-_3
phi = {1: Tm_[1], 2: -Tm_[2], 3: -Tm_[3]}
ok &= np.allclose(br_(phi[1], phi[2]), -phi[3]) and np.allclose(br_(phi[2], phi[3]), -phi[1]) and np.allclose(br_(phi[3], phi[1]), -phi[2])
# the three joining directions are not removable, and no nonzero combination of them is
KGq = np.kron(np.diag([5., 3, 5]), G1)
join = [rotv(7, 13), rotv(7, 14), rotv(7, 15)]
ok &= nullsp(np.array([(X @ KGq - KGq @ X).ravel() for X in join]).T).shape[1] == 0
# the surviving so3 is the rotation part of lG^(3)
rotpart3 = [np.kron(np.diag([0, 0, 1.]), Rab(j + 3, k + 3, 6, np.diag(eta1)) - Rab(j, k, 6, np.diag(eta1)))
            for (j, k) in [(1, 2), (2, 0), (0, 1)]]
ok_rot = same_span(rotpart3, [rotv(13, 14), rotv(13, 15), rotv(14, 15)])
report("7a", ok and ok_rot, "prop:so4unequal(i): commutant with K(x)G (k2 != k3) = span rot(13,14), rot(13,15), rot(14,15) = "
       "graph of an iso su2+ -> su2-; rot(7,13..15) not removable; the surviving so3 is the rotation algebra of lG^(3)")
# 'four of its six generators join constituent two to constituent three': count
def joins23(X):
    return np.abs(X[6:12, 12:18]).max() > 1e-12 or np.abs(X[12:18, 6:12]).max() > 1e-12
nrot = sum(joins23(X) for X in so4)
nT = sum(joins23(X) for X in list(Tp.values()) + list(Tm_.values()))
off = [np.concatenate([X[6:12, 12:18].ravel(), X[12:18, 6:12].ravel()]) for X in so4]
rank_off = np.linalg.matrix_rank(np.array(off))
report("7bN", nrot == 3 and nT == 6 and rank_off == 3,
       "'four of its six generators join constituent two to three' is false: 3 of the 6 rot(i,j) do (6 of the 6 T's; the "
       "joining part has rank 3)")

lamS = sp.symbols('lam')
def so4_S_exact(kq, Tmat, scale):
    Gs = sp.Matrix(G1.astype(int))
    Tq_ = sp.Matrix(np.round(2 * Tmat).astype(int)) / 2
    return sp.kronecker_product(sp.diag(*kq), Gs) + scale * Tq_


S = so4_S_exact((5, 3, 5), Tp[3], 2)
cp = sym_to_dm(S).charpoly()
cpoly = sp.Poly([sp.Rational(int(c.numerator), int(c.denominator)) for c in cp], lamS)
tgt = sp.Poly(sp.expand((lamS ** 2 + 9) ** 2 * (lamS ** 2 + 16) * (lamS ** 2 + 25) ** 3 * (lamS ** 2 + 36)
                        * (lamS ** 4 + 36 * lamS ** 2 + 196)), lamS)
ssM, _ = jordan_semisimple(sym_to_dm(M_exact(S, 3)))
Sn = np.array(S.tolist(), dtype=float)
gr, _ = growth_ratio(Mof(Sn, 3))
lines = np.unique(np.round(np.abs(np.linalg.eigvals(Sn).imag), 9))
S5 = so4_S_exact((5, 5, 5), Tp[3], 2)
cp5 = sym_to_dm(S5).charpoly()
cpoly5 = sp.Poly([sp.Rational(int(c.numerator), int(c.denominator)) for c in cp5], lamS)
tgt5 = sp.Poly(sp.expand((lamS ** 2 + 25) ** 5 * (lamS ** 2 + 16) ** 2 * (lamS ** 2 + 36) ** 2), lamS)
okl = any(np.isclose(lines, 4 + np.sqrt(2)).tolist()) and any(np.isclose(lines, 4 - np.sqrt(2)).tolist())
report("7b", cpoly == tgt and ssM and gr < 3 and okl and cpoly5 == tgt5,
       "prop:so4unequal(ii): kappa=(5,3,5), b=2T+_3: exact char poly as stated, M exactly semisimple and bounded, lines "
       "4+-sqrt2; at (5,5,5): (l^2+25)^5(l^2+16)^2(l^2+36)^2")

# ==============================================================================================
# 8. rem:verdict(e)
# ==============================================================================================
kk3 = sp.symbols('q1 q2 q3', real=True)
Kq = sp.diag(*kk3)
Gs = sp.Matrix(G1.astype(int))
cm = sp.kronecker_product(Kq, Gs) * sp.kronecker_product(Bs, sp.eye(6)) - sp.kronecker_product(Bs, sp.eye(6)) * sp.kronecker_product(Kq, Gs)
KB = Kq * Bs - Bs * Kq
ok = sp.simplify(cm - sp.kronecker_product(KB, Gs)) == sp.zeros(18, 18)
ok &= all(sp.simplify(KB[k, l] - (kk3[k] - kk3[l]) * Bs[k, l]) == 0 for k in range(3) for l in range(3))
nrm = sp.expand(sum(e ** 2 for e in cm))
ok &= sp.simplify(nrm - 6 * sum((kk3[k] - kk3[l]) ** 2 * Bs[k, l] ** 2 for k in range(3) for l in range(3) if k != l)) == 0
report("8", ok, "rem:verdict(e): [K(x)G, B(x)1] = [K,B](x)G, [K,B]_kl = (k_k-k_l)B_kl, ||.||^2 = 6 sum_{k!=l}(k_k-k_l)^2 B_kl^2")

el = time.time() - T0
print(f"x7_check: {len(FAILS)} failures, {el:.1f} s")
sys.exit(0 if not FAILS else 1)
