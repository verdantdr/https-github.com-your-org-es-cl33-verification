#!/usr/bin/env python3
"""Track d7 -- The b-field between manifolds.

    python3 investigations/rest_energy/verify/d7_bfield_between.py

Prints one 'PASS d7-<id>: ...' line per verified claim and exits 0 only if every check passes.

Conventions are those of the unified paper (tex/s02_arena.tex eq. realisation and eq. tower; s10 def. of M_kl and
thm:diagcent; s12 thm:GLorentz; s13 prop:commutantsn; s17 thm:closureunique, prop:several, thm:so9C, thm:nocommon,
thm:PSbinding) and of the note (d4_bfield.tex: thm:covariant, thm:criterion, prop:PS; d5_duplication.tex: thm:bfield2).
  V_n = R^n (x) R^6, copy-major basis (index 6(k-1)+a), eta_n = 1_n (x) eta, eta = diag(1,1,1,-1,-1,-1),
  G e_i = e_{i+3}, G e_{i+3} = -e_i, G_n = 1_n (x) G; z = B + iE per copy, so G acts as i.
  Diagonal l_G: paired rotations (r, r) and paired boosts [[0,-r],[r,0]] on each copy; l_int = so(1,3) on
  span{e_1, e_4, e_5, e_6}; surplus pair {e_2, e_3}.
  Copy rotation C^{kl} = (E_lk - E_kl) (x) 1_6 (maps copy k to copy l); copy boost (E_lk - E_kl) (x) G.
  Shift S = K (x) G + b with K = diag(kappa_k); frame flow dX/dtau = -S X + pi/m, dpi/dtau = S^dagger pi,
  phase-space generator M = [[-S, 1/m], [0, S^dagger]].
  Spinor side: tower generators E_(k,a) = omega^{(x)(k-1)} (x) e_a (x) 1^{(x)(n-k)} on R^(8^n); the spin lift of the
  copy rotation of the directions a in D is -(1/2) sum_{a in D} eta_aa E_(k,a) E_(l,a).

Exactness: commutants are exact rational null spaces (n <= 3) or numerical ranks (n = 4, stated); characteristic
polynomials and semisimplicity are exact over Q (square-free part of the characteristic polynomial annihilates M);
symbolic identities are expanded with sympy.  Floating point is used only for matrix exponentials (tolerance 1e-9)
and sup-norms, where stated.  Runtime: well under a minute.
"""
import itertools
import sys
import time

import numpy as np
import scipy.linalg as sla
import sympy as sp
from sympy import QQ
from sympy.polys.matrices import DomainMatrix

T0 = time.time()
FAILS = []


def report(cid, ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {cid}: {text}")
    if not ok:
        FAILS.append(cid)


def claim(cid, text):
    def deco(fn):
        try:
            ok = bool(fn())
        except Exception as exc:  # reported as failure
            import traceback
            traceback.print_exc()
            print(f"   exception in {cid}: {exc!r}")
            ok = False
        report(cid, ok, text)
        return fn
    return deco


# ============================================================================================ exact linear algebra
lam = sp.Symbol('lam')


def to_dm(M):
    M = sp.Matrix(M)
    rows = [[QQ(int(sp.Rational(x).p), int(sp.Rational(x).q)) for x in M.row(i)] for i in range(M.rows)]
    return DomainMatrix(rows, (M.rows, M.cols), QQ)


def charpoly_q(M):
    coeffs = [sp.Rational(int(c.numerator), int(c.denominator)) for c in to_dm(M).charpoly()]
    return sp.Poly(coeffs, lam)


def is_semisimple(M):
    """Exact: rational M is diagonalisable over C iff sqf(charpoly)(M) = 0."""
    dM = to_dm(M)
    P = charpoly_q(M)
    r = sp.Poly(sp.sqf_part(P.as_expr()), lam)
    n = M.shape[0]
    res = DomainMatrix.zeros((n, n), QQ)
    Id = DomainMatrix.eye(n, QQ)
    for c in r.all_coeffs():
        c = sp.Rational(c)
        res = res * dM + Id * QQ(int(c.p), int(c.q))
    return res.is_zero_matrix


def nullspace_int(A):
    A = np.asarray(A, dtype=np.int64)
    A = A[np.any(A != 0, axis=1)]
    if A.shape[0] == 0:
        return [sp.eye(A.shape[1])[:, j] for j in range(A.shape[1])]
    dm = DomainMatrix([[QQ(int(x)) for x in row] for row in A.tolist()], A.shape, QQ)
    ns = dm.nullspace().to_Matrix()
    return [ns.row(i).T for i in range(ns.rows)]


def to_int(M):
    M = sp.Matrix(M)
    den = 1
    for x in M:
        den = sp.ilcm(den, sp.Rational(x).q)
    return np.array((M * den).tolist(), dtype=np.int64)


def unvec(v, n):
    return sp.Matrix(n, n, lambda i, j: v[j * n + i])


def algebra_basis(n, commute_with=(), eta=None, antisym=False):
    """Exact basis of {X in gl(n): [g,X]=0 for g in commute_with; X^T eta + eta X = 0 if eta; X^T = -X if antisym}."""
    rows = []
    Id = np.eye(n, dtype=np.int64)
    for g in commute_with:
        g_ = to_int(g)
        rows.append(np.kron(Id, g_) - np.kron(g_.T, Id))
    idx = lambda i, j: j * n + i
    if eta is not None:
        d = [int(eta[i, i]) for i in range(n)]
        R = []
        for i in range(n):
            for j in range(i, n):
                r = np.zeros(n * n, dtype=np.int64)
                r[idx(j, i)] += d[j]
                r[idx(i, j)] += d[i]
                R.append(r)
        rows.append(np.array(R))
    if antisym:
        R = []
        for i in range(n):
            for j in range(i, n):
                r = np.zeros(n * n, dtype=np.int64)
                r[idx(i, j)] += 1
                r[idx(j, i)] += 1
                R.append(r)
        rows.append(np.array(R))
    return [unvec(v, n) for v in nullspace_int(np.vstack(rows))]


def span_dim(mats):
    if not mats:
        return 0
    return sp.Matrix.hstack(*[sp.Matrix(m).reshape(m.shape[0] * m.shape[1], 1) for m in mats]).rank()


def same_span(A, B):
    return span_dim(A) == span_dim(B) == span_dim(list(A) + list(B))


def kr(*ms):
    out = sp.Matrix(ms[0])
    for m in ms[1:]:
        out = sp.kronecker_product(out, sp.Matrix(m))
    return out


def Ekl(k, l, n):
    M = sp.zeros(n)
    M[k, l] = 1
    return M


def adj(X, eta):
    return eta * X.T * eta


# ============================================================================================ vector-level arena
I3, Z3 = sp.eye(3), sp.zeros(3)
G1 = sp.Matrix(sp.BlockMatrix([[Z3, -I3], [I3, Z3]]))
ETA1 = sp.diag(1, 1, 1, -1, -1, -1)
P_INT = sp.diag(1, 0, 0, 1, 1, 1)
P_SURP = sp.diag(0, 1, 1, 0, 0, 0)
P_TEMP = sp.diag(1, 1, 1, 0, 0, 0)


def rot3(i, j):
    r = sp.zeros(3)
    r[i, j], r[j, i] = -1, 1
    return r


R3 = [rot3(1, 2), rot3(2, 0), rot3(0, 1)]
LG = [sp.diag(r, r) for r in R3] + [sp.Matrix(sp.BlockMatrix([[Z3, -r], [r, Z3]])) for r in R3]
LG_ROT = LG[:3]
LINT = []
for a_, b_ in itertools.combinations([0, 3, 4, 5], 2):
    A_ = sp.zeros(6)
    A_[a_, b_], A_[b_, a_] = 1, -1
    LINT.append(ETA1 * A_)


def Gn(n):
    return kr(sp.eye(n), G1)


def etan(n):
    return kr(sp.eye(n), ETA1)


def diag_n(gens, n):
    return [kr(sp.eye(n), g) for g in gens]


def copy_rot(k, l, n, P=None):
    return kr(Ekl(l, k, n) - Ekl(k, l, n), sp.eye(6) if P is None else P)


def copy_boost(k, l, n):
    return kr(Ekl(l, k, n) - Ekl(k, l, n), G1)


def Bmat(b12, b23, b31):
    """real antisymmetric 3x3 with B12 = b12, B23 = b23, B31 = b31."""
    return sp.Matrix([[0, b12, -b31], [-b12, 0, b23], [b31, -b23, 0]])


def shift(kap, B=None, Bp=None):
    n = len(kap)
    S = kr(sp.diag(*kap), G1)
    if B is not None:
        S += kr(sp.Matrix(B), sp.eye(6))
    if Bp is not None:
        S += kr(sp.Matrix(Bp), G1)
    return S


def phase_gen(S, m=1):
    N = S.shape[0]
    n = N // 6
    return sp.Matrix(sp.BlockMatrix([[-S, sp.eye(N) / m], [sp.zeros(N), adj(S, etan(n))]]))


def complex_form(A):
    """A real 6n x 6n commuting with G_n -> complex 3n x 3n acting on z = (B + iE) per copy."""
    n = A.shape[0] // 6
    T = [6 * k + a for k in range(n) for a in range(3)]
    Sx = [6 * k + 3 + a for k in range(n) for a in range(3)]
    P = A.extract(T, T)
    Q = A.extract(Sx, T)
    return P + sp.I * Q


def z_of(X):
    n = X.shape[0] // 6
    return sp.Matrix([X[6 * k + a] + sp.I * X[6 * k + 3 + a] for k in range(n) for a in range(3)])


def npf(M):
    return np.array(sp.Matrix(M).evalf(), dtype=complex)


def sup_norm(M, T=400.0, steps=81):
    Mn = np.real_if_close(npf(M))
    return max(np.linalg.norm(sla.expm(Mn * t), 2) for t in np.linspace(0, T, steps))


R = sp.Rational

# ============================================================================================ 1. which fields act between manifolds


@claim("d7-1a", "commutant of the diagonal l_G and G_n in so(3n,3n): dimensions 0, 2, 6 (n = 1, 2, 3, exact) and 12 "
                "(n = 4, numerical rank); it is so(n,C) = span{B (x) 1_6, B (x) G : B in so(n)}, with compact part "
                "(copy rotations) of dimension 0, 1, 3, 6")
def _():
    ok = True
    for n in (1, 2, 3):
        basis = algebra_basis(6 * n, diag_n(LG, n) + [Gn(n)], eta=etan(n))
        comp = algebra_basis(6 * n, diag_n(LG, n) + [Gn(n)], eta=etan(n), antisym=True)
        expect = [copy_rot(k, l, n) for k, l in itertools.combinations(range(n), 2)] + \
                 [copy_boost(k, l, n) for k, l in itertools.combinations(range(n), 2)]
        ok &= len(basis) == n * (n - 1) and len(comp) == n * (n - 1) // 2
        if n > 1:
            ok &= same_span(basis, expect)
            ok &= same_span(comp, [copy_rot(k, l, n) for k, l in itertools.combinations(range(n), 2)])
    # n = 4 numerically
    n = 4
    N = 24
    gens = [np.array(g, dtype=float) for g in diag_n(LG, n) + [Gn(n)]]
    eta = np.array(etan(n), dtype=float)
    basis = []
    for i in range(N):
        for j in range(i + 1, N):
            A_ = np.zeros((N, N))
            A_[i, j], A_[j, i] = 1, -1
            basis.append(eta @ A_)
    Amat = np.vstack([np.array([(g @ b - b @ g).ravel() for b in basis]).T for g in gens])
    ok &= sla.null_space(Amat).shape[1] == 12
    return ok


@claim("d7-1b", "commutant of the diagonal interior Lorentz algebra l_int and G_n in so(3n,3n): dimensions 0, 1, 3 "
                "(n = 1, 2, 3), spanned by the six-direction copy rotations C^{kl}; no copy boost commutes with l_int")
def _():
    ok = True
    for n in (1, 2, 3):
        basis = algebra_basis(6 * n, diag_n(LINT, n) + [Gn(n)], eta=etan(n))
        ok &= len(basis) == n * (n - 1) // 2
        if n > 1:
            ok &= same_span(basis, [copy_rot(k, l, n) for k, l in itertools.combinations(range(n), 2)])
            cb = copy_boost(0, 1, n)
            ok &= any(not (g * cb - cb * g).is_zero_matrix for g in diag_n(LINT, n))
    return ok


@claim("d7-1c", "at n = 2 the binding copy rotation is the paper's interior copy rotation M_12 plus the surplus copy "
                "rotation: C_v = eps(x)P_int + eps(x)P_surp; both summands commute with the diagonal l_int, neither "
                "commutes with G_2, their sum does")
def _():
    n = 2
    C = copy_rot(0, 1, n)
    M12 = copy_rot(0, 1, n, P_INT)
    Cs = copy_rot(0, 1, n, P_SURP)
    G2 = Gn(2)
    ok = (C - M12 - Cs).is_zero_matrix
    ok &= all((g * M12 - M12 * g).is_zero_matrix and (g * Cs - Cs * g).is_zero_matrix for g in diag_n(LINT, 2))
    ok &= not (M12 * G2 - G2 * M12).is_zero_matrix and not (Cs * G2 - G2 * Cs).is_zero_matrix
    ok &= (C * G2 - G2 * C).is_zero_matrix
    return ok


@claim("d7-1d", "spectator assignment: the elements of so(3n,3n) commuting with G_n and with the Lorentz algebra of "
                "copy 1 alone (l_G^(1) or l_int^(1)) form so(3n-3,C) of copies 2..n, of dimension 6 and 30 at n = 2, 3; "
                "every one of them vanishes on copy 1 and has no component into it")
def _():
    ok = True
    for n, d in ((2, 6), (3, 30)):
        for L in (LG, LINT):
            gens = [kr(Ekl(0, 0, n), g) for g in L] + [Gn(n)]
            basis = algebra_basis(6 * n, gens, eta=etan(n))
            ok &= len(basis) == d
            ok &= all(X[:6, :].is_zero_matrix and X[:, :6].is_zero_matrix for X in basis)
    return ok


@claim("d7-2a", "rotor generators: the endomorphisms of V_n commuting with the diagonal l_G form gl(n,C) (dimensions 2, 8, "
                "18); the non-dilating ones (Euclidean antisymmetric) form u(n) (1, 4, 9) = so(n)(x)1 + Sym(n)(x)G, "
                "the first summand eta-antisymmetric (copy rotations), the second eta-self-adjoint (rate matrices); "
                "the paper's rotor G_n is the centre")
def _():
    ok = True
    for n in (1, 2, 3):
        N = 6 * n
        full = algebra_basis(N, diag_n(LG, n))
        nd = algebra_basis(N, diag_n(LG, n), antisym=True)
        ok &= (len(full), len(nd)) == (2 * n * n, n * n)
        rots = [copy_rot(k, l, n) for k, l in itertools.combinations(range(n), 2)]
        rates = [kr(Ekl(k, l, n) + Ekl(l, k, n), G1) for k in range(n) for l in range(k, n)]
        ok &= same_span(nd, rots + rates)
        ok &= all((adj(X, etan(n)) + X).is_zero_matrix for X in rots)
        ok &= all((adj(X, etan(n)) - X).is_zero_matrix for X in rates)
        # the centre of u(n) acting on V_n: elements of nd commuting with all of nd
        cen = [X for X in [Gn(n)] if all((X * Y - Y * X).is_zero_matrix for Y in nd)]
        ok &= len(cen) == 1
    # clamped reduction: with X = e^{-theta A} x and theta = Omega tau, e^{-theta A} xdot = Xdot + Omega A X, because
    # d/dtau e^{-Omega tau A} = -Omega A e^{-Omega tau A}; checked by a central difference for A = S of (1,3) with B12 = -2
    An = np.array(shift([1, 3], [[0, -2], [2, 0]]), dtype=float)
    t0, h, Om = 0.37, 1e-6, 1.3
    U = lambda t: sla.expm(-Om * t * An)
    ok &= np.allclose((U(t0 + h) - U(t0 - h)) / (2 * h), -Om * An @ U(t0), atol=1e-6)
    return ok


# ============================================================================================ 2. Schroedinger form


def random_cent_element(n, seed):
    """random rational b in Cent(G_n) cap so(3n,3n): paired rotation (R,R) plus paired boost [[0,-Q],[Q,0]] in the
    (all temporal, all spatial) ordering, returned in the copy-major basis."""
    rng = np.random.default_rng(seed)
    m = 3 * n

    def ranti():
        A_ = sp.zeros(m)
        for i in range(m):
            for j in range(i + 1, m):
                v = R(int(rng.integers(-4, 5)), int(rng.integers(1, 4)))
                A_[i, j], A_[j, i] = v, -v
        return A_
    Rr, Qq = ranti(), ranti()
    block_rot = sp.diag(Rr, Rr)
    block_boost = sp.Matrix(sp.BlockMatrix([[sp.zeros(m), -Qq], [Qq, sp.zeros(m)]]))
    T = [6 * k + a for k in range(n) for a in range(3)]
    Sx = [6 * k + 3 + a for k in range(n) for a in range(3)]
    perm = T + Sx            # position in block ordering -> copy-major index
    P = sp.zeros(6 * n)
    for pos, idx in enumerate(perm):
        P[idx, pos] = 1
    return P * block_rot * P.T, P * block_boost * P.T, Rr, Qq


@claim("d7-3a", "Schroedinger form: for b = (paired rotation R) + (paired boost Q) in Cent(G_n) and K = diag(kappa_k), "
                "S = K(x)G + b acts on z as i H with H = K(x)1_3 - iR + Q (n = 2, 3, random rational data); the frame "
                "flow is z(tau) = exp(-iH tau) z(0)")
def _():
    ok = True
    for n, seed in ((2, 1), (3, 2)):
        br, bb, Rr, Qq = random_cent_element(n, seed)
        kap = [R(1, 2), 2, R(7, 3)][:n]
        S = shift(kap) + br + bb
        ok &= (S * Gn(n) - Gn(n) * S).is_zero_matrix and (adj(br + bb, etan(n)) + br + bb).is_zero_matrix
        Kt = kr(sp.diag(*kap), sp.eye(3))
        H = Kt - sp.I * Rr + Qq
        ok &= sp.simplify(complex_form(S) - sp.I * H).is_zero_matrix
        X = sp.Matrix(6 * n, 1, lambda i, j: R(i * i - 3 * i + 1, 1 + i % 4))
        ok &= sp.simplify(z_of(S * X) - sp.I * H * z_of(X)).is_zero_matrix
    return ok


@claim("d7-3b", "unitarity <=> compactness: for b in Cent(G_n) cap so(3n,3n), the frame flow preserves E_6 = |X|^2/2 for "
                "every K and every X iff S is Euclidean antisymmetric iff the paired-boost part vanishes (H Hermitian)")
def _():
    ok = True
    for n, seed in ((2, 3), (3, 4)):
        br, bb, _, _ = random_cent_element(n, seed)
        S1 = shift([1, 2, 5][:n]) + br
        S2 = S1 + bb
        ok &= (S1 + S1.T).is_zero_matrix and not (S2 + S2.T).is_zero_matrix
        ok &= (bb + bb.T) == 2 * bb      # the paired boost is Euclidean symmetric
    return ok


@claim("d7-3c", "rates are gauge invariants: for an eta_2-isometric Cayley curve U(tau) and S' = U^-1 S U + U^-1 U', the "
                "eta-self-adjoint part of S' has the characteristic polynomial of K(x)G (exact, tau = 1/3, 2/5)")
def _():
    n = 2
    S = shift([1, 3], [[0, -2], [2, 0]], [[0, R(1, 2)], [-R(1, 2), 0]])
    eta = etan(n)
    A_ = sp.zeros(12)
    rng = np.random.default_rng(7)
    for i in range(12):
        for j in range(i + 1, 12):
            A_[i, j] = R(int(rng.integers(-3, 4)), 2)
            A_[j, i] = -A_[i, j]
    A_ = eta * A_                  # eta-antisymmetric generator
    ok = (adj(A_, eta) + A_).is_zero_matrix
    Id = sp.eye(12)
    sym_ref = charpoly_q((S + adj(S, eta)) / 2)
    for t in (R(1, 3), R(2, 5)):
        Minv = (Id - t * A_ / 2).inv()
        U = Minv * (Id + t * A_ / 2)
        dU = Minv * (A_ / 2) * (U + Id)
        ok &= (U.T * eta * U - eta).is_zero_matrix
        Sp = U.inv() * S * U + U.inv() * dU
        ok &= charpoly_q((Sp + adj(Sp, eta)) / 2) == sym_ref
        ok &= charpoly_q((S + adj(S, eta)) / 2) == charpoly_q(kr(sp.diag(1, 3), G1))
    return ok


@claim("d7-3d", "sum rules for compact b: tr H = 3 sum kappa_k and tr H^2 = 3 sum kappa_k^2 + |R|_F^2 (symbolic, n = 2, "
                "3): the field redistributes the rates at fixed sum and widens their spread")
def _():
    ok = True
    for n in (2, 3):
        ks = sp.symbols(f'k1:{n + 1}', real=True)
        m = 3 * n
        rs = sp.symbols(f'r1:{m * (m - 1) // 2 + 1}', real=True)
        Rr = sp.zeros(m)
        it = iter(rs)
        for i in range(m):
            for j in range(i + 1, m):
                v = next(it)
                Rr[i, j], Rr[j, i] = v, -v
        H = kr(sp.diag(*ks), sp.eye(3)) - sp.I * Rr
        ok &= sp.expand(H.trace() - 3 * sum(ks)) == 0
        ok &= sp.expand((H * H).trace() - 3 * sum(k * k for k in ks) - sum(x * x for x in Rr)) == 0
    return ok


@claim("d7-3e", "removability: a compact b in Cent(G_3) commutes with K(x)G iff it preserves the rate eigenspaces; "
                "centraliser dimensions in so(9) cap Cent(G_3): 36, 18, 9 for rates (k,k,k), (k,k,k'), distinct; "
                "in the copy so(3): 3, 1, 0")
def _():
    n = 3
    comp = algebra_basis(18, [Gn(3)], eta=etan(3), antisym=True)
    copies = [copy_rot(k, l, 3) for k, l in itertools.combinations(range(3), 2)]
    ok = len(comp) == 36
    for kap, d1, d2 in (((2, 2, 2), 36, 3), ((2, 2, 5), 18, 1), ((1, 2, 5), 9, 0)):
        KG = kr(sp.diag(*kap), G1)
        c1 = algebra_basis(18, [Gn(3), KG], eta=etan(3), antisym=True)
        c2 = [X for X in copies if (X * KG - KG * X).is_zero_matrix]
        ok &= len(c1) == d1 and len(c2) == d2
    return ok


def bilinear_criterion(H):
    """Corollary of d4:thm:criterion in the Schroedinger form: for every eigenvalue lam of H with -lam also an
    eigenvalue, the eigenspaces E(lam), E(-lam) must be orthogonal for v.w = sum v_a w_a."""
    Hn = npf(H)
    w, V = np.linalg.eig(Hn)
    ok = True
    for i in range(len(w)):
        for j in range(len(w)):
            if abs(w[i] + w[j]) < 1e-9 and abs(V[:, i] @ V[:, j]) > 1e-9:
                ok = False
    return ok


@claim("d7-3f", "boundedness at n = 3 (copy so(3) field, m = 1): exact semisimplicity of M agrees with the bilinear "
                "criterion on five cases -- bounded: rates (1,2,4) with B = (1/2,1/3,1/5); seesaw (0,3,5) with "
                "B = (1,1/2,1/3); equal rates (1,1,1) with B12 = 1; unbounded: (1,2,3) with B = (1,1,1) (a zero mode, "
                "v.v != 0); zero rates with B = (1,1,1); sup |e^{Mt}| over t <= 400 agrees")
def _():
    cases = [([1, 2, 4], Bmat(R(1, 2), R(1, 3), R(1, 5)), True),
             ([0, 3, 5], Bmat(1, R(1, 2), R(1, 3)), True),
             ([1, 1, 1], Bmat(1, 0, 0), True),
             ([1, 2, 3], Bmat(1, 1, 1), False),
             ([0, 0, 0], Bmat(1, 1, 1), False)]
    ok = True
    for kap, B, bounded in cases:
        S = shift(kap, B)
        M = phase_gen(S)
        H = sp.diag(*kap) - sp.I * B
        ss = is_semisimple(M)
        crit = bilinear_criterion(H)
        sup = sup_norm(M)
        ok &= (ss == bounded) and (crit == bounded) and ((sup < 10) == bounded)
    # the zero mode of the fourth case
    H = np.diag([1, 2, 3.]) - 1j * np.array(Bmat(1, 1, 1), dtype=float)
    w, V = np.linalg.eig(H)
    v = V[:, np.argmin(abs(w))]
    ok &= abs(min(abs(w))) < 1e-12 and abs(v @ v) > 0.1
    return ok


# ============================================================================================ 3. copy boosts


@claim("d7-4a", "copy boost between two manifolds (b = beta' eps(x)G, rates (1,4)): H = K + beta' eps has real "
                "spectrum iff |beta'| <= |k1-k2|/2 = 3/2; M is semisimple with imaginary spectrum for beta' = 1/2, 1 "
                "(bounded), not semisimple at the exceptional point 3/2, and has Re lam = sqrt(beta'^2-9/4) at beta' = 2; "
                "at equal rates (2,2) every beta' != 0 grows (beta' = 1/3: Re lam = 1/3)")
def _():
    ok = True
    for bp, kind in ((R(1, 2), 'bounded'), (R(1), 'bounded'), (R(3, 2), 'ep'), (R(2), 'exp')):
        S = shift([1, 4], None, [[0, -bp], [bp, 0]])
        M = phase_gen(S)
        ss = is_semisimple(M)
        ev = np.linalg.eigvals(np.real(npf(M)))
        mre = max(abs(ev.real))
        Hc = sp.Matrix([[1, -bp], [bp, 4]])
        disc = sp.Rational(9, 4) - bp ** 2
        if kind == 'bounded':
            ok &= ss and mre < 1e-9 and disc > 0
        elif kind == 'ep':
            ok &= (not ss) and disc == 0
        else:
            ok &= ss and abs(mre - float(sp.sqrt(bp ** 2 - sp.Rational(9, 4)))) < 1e-9 and disc < 0
        ok &= sp.expand(Hc.charpoly(lam).as_expr() - ((lam - sp.Rational(5, 2)) ** 2 - disc)) == 0
    S = shift([2, 2], None, [[0, -R(1, 3)], [R(1, 3), 0]])
    ev = np.linalg.eigvals(np.real(npf(phase_gen(S))))
    ok &= abs(max(ev.real) - 1 / 3) < 1e-9
    return ok


@claim("d7-4b", "below the exceptional point the copy-boost flow conserves a positive form Q != E_6: for beta' = 1, "
                "Q = (V V^dagger)^-1 with V the eigenvectors of H satisfies H^dagger Q = Q H, Q > 0, and Q is not a "
                "multiple of the identity (pseudo-Hermiticity)")
def _():
    H = np.array([[1, -1], [1, 4.]])
    w, V = np.linalg.eig(H)
    Q = np.linalg.inv(V @ V.conj().T)
    return np.allclose(H.conj().T @ Q, Q @ H) and np.all(np.linalg.eigvalsh((Q + Q.conj().T) / 2) > 0) and \
        abs(Q[0, 1]) > 1e-3


# ============================================================================================ 4. three manifolds

k1, k2, k3, b12, b23, b31, xx = sp.symbols('k1 k2 k3 b12 b23 b31 x', real=True)
K3 = sp.diag(k1, k2, k3)
B3 = Bmat(b12, b23, b31)
H3 = K3 - sp.I * B3
J3 = b12 * b23 * b31


@claim("d7-5a", "three manifolds: det(x - H) = x^3 - (sum k) x^2 + (sum k_k k_l - sum B^2) x - (k1k2k3 - k3 B12^2 - "
                "k1 B23^2 - k2 B31^2): the spectrum depends on the B_kl only through their squares")
def _():
    cp = sp.expand((xx * sp.eye(3) - H3).det())
    ref = xx ** 3 - (k1 + k2 + k3) * xx ** 2 + (k1 * k2 + k1 * k3 + k2 * k3 - b12 ** 2 - b23 ** 2 - b31 ** 2) * xx \
        - (k1 * k2 * k3 - k3 * b12 ** 2 - k1 * b23 ** 2 - k2 * b31 ** 2)
    return sp.expand(cp - ref) == 0


@claim("d7-5b", "the plaquette: H12 H23 H31 = i B12 B23 B31 = i J (purely imaginary), and the Jarlskog-type commutator "
                "determinant det[H,K] = 2i J (k1-k2)(k1-k3)(k2-k3)")
def _():
    pl = sp.expand(H3[0, 1] * H3[1, 2] * H3[2, 0])
    D = sp.expand((H3 * K3 - K3 * H3).det())
    return sp.expand(pl - sp.I * J3) == 0 and sp.expand(D - 2 * sp.I * J3 * (k1 - k2) * (k1 - k3) * (k2 - k3)) == 0


@claim("d7-5c", "equivalences: the endomorphisms commuting with G_3, the diagonal l_G and K(x)G (distinct rates) are "
                "sum_k (a_k + b_k G) on copy k (dimension 6), and the eta_3-isometries among them are exactly the eight "
                "copy reflections diag(+-1,+-1,+-1)(x)1_6; under them B_kl -> s_k s_l B_kl, so |B_kl| and sign J are "
                "the invariants")
def _():
    KG = kr(sp.diag(1, 2, 5), G1)
    basis = algebra_basis(18, diag_n(LG, 3) + [Gn(3), KG])
    expect = [kr(Ekl(k, k, 3), sp.eye(6)) for k in range(3)] + [kr(Ekl(k, k, 3), G1) for k in range(3)]
    ok = len(basis) == 6 and same_span(basis, expect)
    a = sp.symbols('a1:4', real=True)
    b = sp.symbols('c1:4', real=True)
    g = sum((a[k] * kr(Ekl(k, k, 3), sp.eye(6)) + b[k] * kr(Ekl(k, k, 3), G1) for k in range(3)), sp.zeros(18))
    eqs = set(sp.expand(x) for x in (g.T * etan(3) * g - etan(3)) if sp.expand(x) != 0)
    sols = sp.solve(list(eqs), list(a) + list(b), dict=True)
    ok &= len(sols) == 8 and all(all(s[bb] == 0 for bb in b) and all(abs(s[aa]) == 1 for aa in a) for s in sols)
    # J is invariant under sign changes
    for s in itertools.product((1, -1), repeat=3):
        Bs = sp.Matrix(3, 3, lambda i, j: s[i] * s[j] * B3[i, j])
        ok &= sp.expand(Bs[0, 1] * Bs[1, 2] * Bs[2, 0] - J3) == 0
    return ok


@claim("d7-5d", "parity and interior time reversal: with P = 1_n(x)diag(1_3,-1_3) and T = -P, P S P^-1 maps (K, B) to "
                "(-K, B) and -T S T^-1 maps (K, B) to (K, -B); at n = 2 the copy reflection diag(1,-1) undoes B -> -B, at "
                "n = 3 with J != 0 no copy reflection does (J -> -J)")
def _():
    ok = True
    for n, kap, B in ((2, [1, 3], sp.Matrix([[0, -2], [2, 0]])), (3, [1, 2, 5], Bmat(1, 2, 3))):
        P = kr(sp.eye(n), sp.diag(1, 1, 1, -1, -1, -1))
        T = -P
        S = shift(kap, B)
        ok &= (P * S * P.inv() - shift([-k for k in kap], B)).is_zero_matrix
        ok &= (-T * S * T.inv() - shift(kap, -B)).is_zero_matrix
        refl = []
        for s in itertools.product((1, -1), repeat=n):
            D = kr(sp.diag(*s), sp.eye(6))
            if (D * shift(kap, B) * D - shift(kap, -B)).is_zero_matrix:
                refl.append(s)
        ok &= (len(refl) > 0) == (n == 2)
    return ok


@claim("d7-5e", "general n: (K, B) and (K, -B) are related by a copy reflection iff the coupling graph {kl : B_kl != 0} "
                "is bipartite (all graphs on n = 3, 4, 5 vertices)")
def _():
    ok = True
    for n in (3, 4, 5):
        pairs = list(itertools.combinations(range(n), 2))
        for mask in range(1 << len(pairs)):
            edges = [p for i, p in enumerate(pairs) if mask >> i & 1]
            refl = any(all(s[k] * s[l] == -1 for k, l in edges) for s in itertools.product((1, -1), repeat=n))
            # bipartite test by BFS two-colouring
            col = [None] * n
            bip = True
            for st in range(n):
                if col[st] is None:
                    col[st] = 0
                    stack = [st]
                    while stack:
                        u = stack.pop()
                        for k, l in edges:
                            for x_, y_ in ((k, l), (l, k)):
                                if x_ == u:
                                    if col[y_] is None:
                                        col[y_] = 1 - col[u]
                                        stack.append(y_)
                                    elif col[y_] == col[u]:
                                        bip = False
            ok &= refl == bip
    return ok


@claim("d7-5f", "chiral circulation: rates (1,2,4), B = (1,1,1), energy started in copy 1: the copy energies P_k(tau) "
                "for B and -B agree for k = 1 and differ for k = 2, 3 (at tau = 0.5: P_2 = 0.293 against 0.116), and "
                "P_k(tau; -B) = P_k(-tau; B)")
def _():
    K = np.diag([1., 2., 4.])
    B = np.array(Bmat(1, 1, 1), dtype=float)
    Hp, Hm = K - 1j * B, K + 1j * B
    ok = True
    for t in (0.5, 1.0, 1.7):
        Pp = np.abs(sla.expm(-1j * Hp * t)[:, 0]) ** 2
        Pm = np.abs(sla.expm(-1j * Hm * t)[:, 0]) ** 2
        Pr = np.abs(sla.expm(1j * Hp * t)[:, 0]) ** 2
        ok &= abs(Pp[0] - Pm[0]) < 1e-12 and abs(Pp[1] - Pm[1]) > 1e-2 and np.allclose(Pm, Pr)
        ok &= abs(Pp.sum() - 1) < 1e-12
    Pp = np.abs(sla.expm(-1j * Hp * 0.5)[:, 0]) ** 2
    Pm = np.abs(sla.expm(-1j * Hm * 0.5)[:, 0]) ** 2
    ok &= abs(Pp[1] - 0.2929) < 5e-4 and abs(Pm[1] - 0.1159) < 5e-4
    return ok


# ============================================================================================ 5. isospin

sx2 = np.array([[0, 1], [1, 0]])
sz2 = np.array([[1, 0], [0, -1]])
ep2 = np.array([[0, -1], [1, 0]])
I2 = np.eye(2, dtype=np.int64)


def nk(*ms):
    r = ms[0]
    for x in ms[1:]:
        r = np.kron(r, x)
    return r


e8 = [None, nk(sx2, I2, I2), nk(sz2, sx2, I2), nk(sz2, sz2, sx2), nk(ep2, I2, I2), nk(sz2, ep2, I2), nk(sz2, sz2, ep2)]
ETA = [None, 1, 1, 1, -1, -1, -1]
OM8 = e8[1] @ e8[2] @ e8[3] @ e8[4] @ e8[5] @ e8[6]
I8 = np.eye(8, dtype=np.int64)


def tower(n):
    return {(k, a): nk(*([OM8] * (k - 1) + [e8[a]] + [I8] * (n - k))) for k in range(1, n + 1) for a in range(1, 7)}


def spin_copy(E, k, l, dirs):
    return sum(-0.5 * ETA[a] * (E[(k, a)] @ E[(l, a)]) for a in dirs)


@claim("d7-6a", "spin lifts at n = 2: the copy rotation of the six binding directions has eigenvalues i{-3..3} and "
                "exp(2 pi C) = +1; that of the four interior directions (M_12) i{-2..2}, +1; that of the three temporal "
                "directions i{+-1/2, +-3/2}, exp(2 pi C) = -1; on vectors each lift acts as the copy rotation of its "
                "directions, and only the six-direction one commutes with G_2")
def _():
    E = tower(2)
    ok = True
    for a in range(1, 7):
        for b in range(1, 7):
            for k in (1, 2):
                for l in (1, 2):
                    ok &= (E[(k, a)] @ E[(l, b)] + E[(l, b)] @ E[(k, a)] ==
                           2 * ETA[a] * (a == b) * (k == l) * np.eye(64, dtype=np.int64)).all()
    for dirs, spec, sign, P in (((1, 2, 3, 4, 5, 6), {-3, -2, -1, 0, 1, 2, 3}, 1, sp.eye(6)),
                                ((1, 4, 5, 6), {-2, -1, 0, 1, 2}, 1, P_INT),
                                ((1, 2, 3), {-1.5, -0.5, 0.5, 1.5}, -1, P_TEMP)):
        C = spin_copy(E, 1, 2, dirs)
        w = np.linalg.eigvals(C)
        ok &= set(np.round((w / 1j).real, 9)) == set(float(s) for s in spec) and np.allclose(w.real, 0)
        ok &= np.allclose(sla.expm(2 * np.pi * C), sign * np.eye(64), atol=1e-9)
        # action on the 12 vectors: [C, E_(k,a)] in the span of the E's equals the copy rotation of dirs
        vec = np.zeros((12, 12))
        for k in (1, 2):
            for a in range(1, 7):
                Y = C @ E[(k, a)] - E[(k, a)] @ C
                for l in (1, 2):
                    for b in range(1, 7):
                        vec[6 * (l - 1) + b - 1, 6 * (k - 1) + a - 1] = np.trace(Y @ E[(l, b)]) / (64 * ETA[b])
        ok &= np.allclose(vec, np.array(copy_rot(0, 1, 2, P), dtype=float))
        Cv = copy_rot(0, 1, 2, P)
        ok &= (Cv * Gn(2) - Gn(2) * Cv).is_zero_matrix == (len(dirs) == 6)
    return ok


@claim("d7-6b", "n = 3: the three binding copy rotations close on so(3) ([C12, C13] = C23) on the 512-dimensional "
                "module; the Casimir has eigenvalues j(j+1) = 0, 2, 6, 12 with multiplicities 40, 216, 200, 56 "
                "(j = 0..3, all integer), and exp(2 pi C12) = +1")
def _():
    E = tower(3)
    d6 = range(1, 7)
    C12, C13, C23 = spin_copy(E, 1, 2, d6), spin_copy(E, 1, 3, d6), spin_copy(E, 2, 3, d6)
    ok = np.allclose(C12 @ C13 - C13 @ C12, C23) and np.allclose(C13 @ C23 - C23 @ C13, C12) and \
        np.allclose(C23 @ C12 - C12 @ C23, C13)
    Cas = -(C12 @ C12 + C13 @ C13 + C23 @ C23)
    w = np.round(np.linalg.eigvalsh((Cas + Cas.T) / 2), 8)
    vals, cnt = np.unique(w, return_counts=True)
    ok &= list(vals) == [0, 2, 6, 12] and list(cnt) == [40, 216, 200, 56]
    ok &= np.allclose(sla.expm(2 * np.pi * C12), np.eye(512), atol=1e-8)
    return ok


@claim("d7-6c", "the Yang-Mills doublet on two copies of the undivided energy: the real forms of i1, i sigma_x, "
                "i sigma_y, i sigma_z on (z_1, z_2) are 1(x)G, sigma_x(x)G, -eps(x)1, sigma_z(x)G; all commute with G_2 and "
                "the diagonal l_G and preserve E_6 + E_6; only i sigma_y is eta_2-antisymmetric (a connection), the other "
                "three are eta_2-self-adjoint (rate terms); for S = kappa G_2 + b_YM the lines are kappa +- |beta|, and the "
                "rates (spectrum of the self-adjoint part) are kappa +- (beta_x^2 + beta_z^2)^(1/2)")
def _():
    eta2 = etan(2)
    G2 = Gn(2)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])

    def realform(A):
        return kr(A.applyfunc(sp.re), sp.eye(6)) + kr(A.applyfunc(sp.im), G1)
    forms = {n_: realform(sp.I * A) for n_, A in (('1', sp.eye(2)), ('x', sx), ('y', sy), ('z', sz))}
    ok = (forms['1'] - G2).is_zero_matrix and (forms['x'] - kr(sx, G1)).is_zero_matrix
    ok &= (forms['y'] + kr(sp.Matrix([[0, -1], [1, 0]]), sp.eye(6))).is_zero_matrix
    ok &= (forms['z'] - kr(sz, G1)).is_zero_matrix
    for n_, X in forms.items():
        ok &= (X * G2 - G2 * X).is_zero_matrix and (X + X.T).is_zero_matrix
        ok &= all((X * g - g * X).is_zero_matrix for g in diag_n(LG, 2))
        anti = (adj(X, eta2) + X).is_zero_matrix
        selfa = (adj(X, eta2) - X).is_zero_matrix
        ok &= (anti and not selfa) if n_ == 'y' else (selfa and not anti)
    kap, bx, by, bz = 3, R(3, 5), R(4, 5), R(6, 5)
    S = kap * G2 + bx * forms['x'] + by * forms['y'] + bz * forms['z']
    bb = sp.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    ok &= sp.expand(charpoly_q(S).as_expr() - ((lam ** 2 + (kap + bb) ** 2) * (lam ** 2 + (kap - bb) ** 2)) ** 3) == 0
    Ssym = (S + adj(S, eta2)) / 2
    br = sp.sqrt(bx ** 2 + bz ** 2)
    ok &= sp.expand(charpoly_q(Ssym).as_expr() - ((lam ** 2 + (kap + br) ** 2) * (lam ** 2 + (kap - br) ** 2)) ** 3) == 0
    return ok


# ============================================================================================ 6. weak isospin between unequal constituents


def rot(i, j, N=18):
    """paper's rot(i,j) = e_{i'}e_{j'} - e_ie_j acting by (1/2) ad on vectors (1-based temporal i < j): e_i -> e_j,
    e_j -> -e_i, and the same on the partners i' = i+3, j' = j+3."""
    M = sp.zeros(N)
    for a, b in ((i, j), (i + 3, j + 3)):
        M[b - 1, a - 1] += 1
        M[a - 1, b - 1] -= 1
    return M


T1 = {s: (rot(13, 14) + s * rot(7, 15)) / 2 for s in (1, -1)}
T2 = {s: (rot(7, 14) - s * rot(13, 15)) / 2 for s in (1, -1)}
T3 = {s: (rot(7, 13) + s * rot(14, 15)) / 2 for s in (1, -1)}


@claim("d7-7a", "so4_Delta on the frame: the T^+-_i of d4:prop:PS commute with G_3, are eta_3-antisymmetric, satisfy "
                "[T^+_i, T^-_j] = 0 and [T^s_1, T^s_2] = -T^s_3 cyclically; with rates (5,3,5) the part commuting with "
                "K(x)G is span{rot(13,14), rot(13,15), rot(14,15)} = span{T^+_1 + T^-_1, T^+_2 - T^-_2, T^+_3 - T^-_3}, the "
                "graph of an isomorphism su2_+ -> su2_- (a diagonal so(3)), while rot(7,13), rot(7,14), rot(7,15) are not "
                "removable")
def _():
    G3 = Gn(3)
    eta3 = etan(3)
    ok = True
    gens = [T1[1], T2[1], T3[1], T1[-1], T2[-1], T3[-1]]
    ok &= all((X * G3 - G3 * X).is_zero_matrix and (adj(X, eta3) + X).is_zero_matrix for X in gens)
    ok &= all((A * B - B * A).is_zero_matrix for A in gens[:3] for B in gens[3:])
    for s in (1, -1):
        Ts = [T1[s], T2[s], T3[s]]
        for i in range(3):
            A, B, C = Ts[i], Ts[(i + 1) % 3], Ts[(i + 2) % 3]
            ok &= (A * B - B * A + C).is_zero_matrix
    KG = kr(sp.diag(5, 3, 5), G1)
    so4 = [rot(i, j) for i, j in itertools.combinations([7, 13, 14, 15], 2)]
    keep = [X for X in so4 if (X * KG - KG * X).is_zero_matrix]
    ok &= len(keep) == 3 and same_span(keep, [rot(13, 14), rot(13, 15), rot(14, 15)])
    ok &= same_span(keep, [T1[1] + T1[-1], T2[1] - T2[-1], T3[1] - T3[-1]])
    # the commuting part of the whole so4_Delta (linear combinations), exactly
    c = sp.symbols('c1:7')
    X = sum((ci * Y for ci, Y in zip(c, so4)), sp.zeros(18))
    sol = sp.solve(list(set(sp.expand(x) for x in (X * KG - KG * X) if sp.expand(x) != 0)), c, dict=True)
    ok &= len(sol) == 1 and set(sol[0]) == {c[0], c[1], c[2]} and all(v == 0 for v in sol[0].values())
    return ok


@claim("d7-7b", "with rates (5,3,5) and b = 2 T^+_3: det(lam - S) = (lam^2+9)^2 (lam^2+16) (lam^2+25)^3 (lam^2+36) "
                "(lam^4+36 lam^2+196); the quartic gives the lines 4 +- sqrt2, the two-level mixing of z_7 and z_13 "
                "(rates 3, 5, coupling 1), and M is semisimple (bounded); at equal rates 5 the same b gives "
                "(lam^2+25)^5 (lam^2+16)^2 (lam^2+36)^2, as in d4:prop:so4spec")
def _():
    N = 18
    ok = True
    for kap, ref in (((5, 3, 5), (lam ** 2 + 9) ** 2 * (lam ** 2 + 16) * (lam ** 2 + 25) ** 3 * (lam ** 2 + 36) *
                      (lam ** 4 + 36 * lam ** 2 + 196)),
                     ((5, 5, 5), (lam ** 2 + 25) ** 5 * (lam ** 2 + 16) ** 2 * (lam ** 2 + 36) ** 2)):
        S = kr(sp.diag(*kap), G1) + 2 * T3[1]
        ok &= sp.expand(charpoly_q(S).as_expr() - sp.expand(ref)) == 0
        M = sp.Matrix(sp.BlockMatrix([[-S, sp.eye(N)], [sp.zeros(N), adj(S, etan(3))]]))
        ok &= is_semisimple(M)
    r = sp.Poly(lam ** 4 + 36 * lam ** 2 + 196, lam)
    ok &= all(sp.expand(r.as_expr().subs(lam, sp.I * w)) == 0 for w in (4 + sp.sqrt(2), 4 - sp.sqrt(2)))
    w2 = sp.Rational(3 + 5, 2) + sp.sqrt(sp.Rational((3 - 5) ** 2, 4) + 1)
    ok &= sp.simplify(w2 - (4 + sp.sqrt(2))) == 0
    return ok


# ============================================================================================ 7. the commutator that makes b physical


@claim("d7-8", "[K(x)G, B(x)1] = [K,B](x)G with [K,B]_kl = (k_k - k_l) B_kl, so |[K(x)G, B(x)1]|_F^2 = 6 sum_{k != l} "
               "(k_k - k_l)^2 B_kl^2: the non-removable part of the copy field is proportional to the rate differences "
               "(symbolic, n = 3)")
def _():
    S1 = kr(K3, G1)
    S2 = kr(B3, sp.eye(6))
    C = S1 * S2 - S2 * S1
    ok = (C - kr(K3 * B3 - B3 * K3, G1)).is_zero_matrix
    KB = K3 * B3 - B3 * K3
    ok &= all(sp.expand(KB[i, j] - (K3[i, i] - K3[j, j]) * B3[i, j]) == 0 for i in range(3) for j in range(3))
    nrm = sp.expand(sum(x ** 2 for x in C))
    ref = 6 * sum((K3[i, i] - K3[j, j]) ** 2 * B3[i, j] ** 2 for i in range(3) for j in range(3) if i != j)
    ok &= sp.expand(nrm - ref) == 0
    return ok


# ============================================================================================ summary
print(f"# d7_bfield_between.py: {time.time() - T0:.1f} s")
if FAILS:
    print("FAILED:", ", ".join(FAILS))
    sys.exit(1)
sys.exit(0)
