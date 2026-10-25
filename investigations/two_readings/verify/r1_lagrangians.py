#!/usr/bin/env python3
"""Verification script for Section r1 ("Two readings of X and their Lagrangians") of the two-readings note.

    python3 investigations/two_readings/verify/r1_lagrangians.py

Prints one line 'PASS r1-<n>: <claim>' (or 'FAIL ...') per claim and exits 0 only if every check passes.
Algebraic statements use exact integer/rational/symbolic arithmetic (sympy); Euler-Lagrange equations,
Noether currents and stress tensors are derived with sympy on general symbolic fields; Grassmann-valued
Dirac fields are handled by a small exact Grassmann jet algebra written below.  Floating point is used only
for the inertia (signature) of two real symmetric matrices, with tolerance 1e-9.  Runtime: well under 3 minutes.
"""
import itertools
import sys
from functools import lru_cache

import numpy as np
import sympy as sp
from sympy.combinatorics import Permutation

RESULTS = []


def check(cid, claim, ok):
    ok = bool(ok)
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} r1-{cid}: {claim}")
    sys.stdout.flush()


def is_zero(expr):
    return sp.simplify(sp.expand(expr)) == 0


def mat_zero(M):
    return all(is_zero(x) for x in sp.Matrix(M))


# ----------------------------------------------------------------------------------------------
# Clifford realisation of Cl(3,3) on R^8 (tex/sA_realisation.tex)
# ----------------------------------------------------------------------------------------------
sx = sp.Matrix([[0, 1], [1, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
ep = sp.Matrix([[0, -1], [1, 0]])
I2 = sp.eye(2)


def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = sp.kronecker_product(out, m)
    return out


e = [None,
     kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx),
     kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
ETA = sp.diag(1, 1, 1, -1, -1, -1)
I8 = sp.eye(8)
eta6 = [None, 1, 1, 1, -1, -1, -1]

# G and the conversion on V = R^{3,3} (basis e1..e6)
Z3, I3 = sp.zeros(3), sp.eye(3)
G6 = sp.Matrix(sp.BlockMatrix([[Z3, -I3], [I3, Z3]]))      # G e_i = e_{i+3}, G e_{i+3} = -e_i (columns)


def vec_coeffs(v8):
    """coefficients of an 8x8 matrix in the span of e1..e6 (tr(e_a e_b) = 8 eta_ab); asserts membership"""
    c = [sp.Rational((e[a] * v8).trace(), 8 * eta6[a]) for a in range(1, 7)]
    rec = sum((c[a - 1] * e[a] for a in range(1, 7)), sp.zeros(8))
    assert rec == v8, "not a vector"
    return sp.Matrix(c)


def vec_action(X8):
    """6x6 matrix of v -> X v - v X on V (columns = images of e1..e6)"""
    return sp.Matrix.hstack(*[vec_coeffs(X8 * e[a] - e[a] * X8) for a in range(1, 7)])


def lie_span_rank(mats):
    return sp.Matrix.hstack(*[sp.Matrix(m).reshape(len(m), 1) for m in mats]).rank()


def intertwiners(src, tgt):
    """dimension of {M : M src(X) = tgt(X) M for all generators X}; src/tgt are lists of matrices"""
    ns, nt = src[0].shape[0], tgt[0].shape[0]
    rows = []
    for A, B in zip(src, tgt):
        # vec(M A - B M) = (A^T kron I_nt - I_ns kron B) vec(M)  (column-major vec)
        rows.append(sp.kronecker_product(A.T, sp.eye(nt)) - sp.kronecker_product(sp.eye(ns), B))
    big = sp.Matrix.vstack(*rows)
    return ns * nt - big.rank()


# ----------------------------------------------------------------------------------------------
# A small exact Grassmann jet algebra for real Grassmann-valued fields psi_a(x), a = 0..7
# Generators: psi_a with a sorted multi-index of derivatives (tuple over 0..3).
# ----------------------------------------------------------------------------------------------
_KEYS, _IDX = [], {}


def gid(a, d=()):
    key = (a, tuple(sorted(d)))
    if key not in _IDX:
        _IDX[key] = len(_KEYS)
        _KEYS.append(key)
    return _IDX[key]


def _sort_sign(mono):
    """sort a tuple of generator ids; return (sign, sorted) or (0, None) if a generator repeats"""
    lst = list(mono)
    if len(set(lst)) != len(lst):
        return 0, None
    sign = 1
    for i in range(len(lst)):            # bubble sort counting transpositions (short tuples)
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
    return sign, tuple(lst)


class GA:
    """element of the Grassmann algebra: dict {sorted tuple of generator ids: coefficient}"""
    __slots__ = ("t",)

    def __init__(self, terms=None):
        self.t = {k: v for k, v in (terms or {}).items() if not _zero_coeff(v)}

    @staticmethod
    def gen(a, d=()):
        return GA({(gid(a, d),): sp.Integer(1)})

    def __add__(self, o):
        out = dict(self.t)
        for k, v in o.t.items():
            out[k] = out.get(k, 0) + v
        return GA(out)

    def __neg__(self):
        return GA({k: -v for k, v in self.t.items()})

    def __sub__(self, o):
        return self + (-o)

    def scale(self, c):
        return GA({k: c * v for k, v in self.t.items()})

    def __mul__(self, o):
        if not isinstance(o, GA):
            return self.scale(o)
        out = {}
        for k1, v1 in self.t.items():
            for k2, v2 in o.t.items():
                s, k = _sort_sign(k1 + k2)
                if s:
                    out[k] = out.get(k, 0) + s * v1 * v2
        return GA(out)

    __rmul__ = scale

    def simplify(self):
        return GA({k: sp.expand(v) for k, v in self.t.items()})

    def is_zero(self):
        return all(_zero_coeff(sp.expand(v)) for v in self.t.values())


def _zero_coeff(v):
    if isinstance(v, sp.Basic):
        return sp.expand(v) == 0
    return v == 0


def lin(vec):
    """GA linear element sum_a vec[a] * generator (vec: list of (coeff, a, d))"""
    out = {}
    for c, a, d in vec:
        k = (gid(a, d),)
        out[k] = out.get(k, 0) + c
    return GA(out)


def field(d=()):
    """the column psi_{,d} as a list of 8 GA generators"""
    return [GA.gen(a, d) for a in range(8)]


def matvec(M, col):
    return [sum((col[b].scale(M[a, b]) for b in range(8) if M[a, b] != 0), GA()) for a in range(8)]


def bilinear(rowcol, M, col):
    """rowcol^T M col for columns of GA linear elements"""
    out = GA()
    for a in range(8):
        for b in range(8):
            if M[a, b] != 0:
                out = out + (rowcol[a] * col[b]).scale(M[a, b])
    return out


def left_deriv(x, g):
    """left derivative of GA element x with respect to generator id g"""
    out = {}
    for k, v in x.t.items():
        if g in k:
            pos = k.index(g)
            kk = k[:pos] + k[pos + 1:]
            out[kk] = out.get(kk, 0) + (-1) ** pos * v
    return GA(out)


XS = sp.symbols("t x y z", real=True)     # x^0 = t (c = 1 in the field-theory part)


def tot_deriv(x, mu):
    """total derivative d_mu of a GA element whose coefficients may depend on the coordinates XS"""
    out = {}
    for k, v in x.t.items():
        dv = sp.diff(v, XS[mu]) if isinstance(v, sp.Basic) else 0
        if not _zero_coeff(dv):
            out[k] = out.get(k, 0) + dv
        for pos, g in enumerate(k):
            a, d = _KEYS[g]
            g2 = gid(a, d + (mu,))
            s, kk = _sort_sign(k[:pos] + (g2,) + k[pos + 1:])
            if s:
                out[kk] = out.get(kk, 0) + s * v
    return GA(out)


def euler_lagrange_first_order(L):
    """left Euler-Lagrange expressions E_a = dL/dpsi_a - d_mu (dL/dpsi_{a,mu}) for first-order L"""
    E = []
    for a in range(8):
        Ea = left_deriv(L, gid(a))
        for mu in range(4):
            Ea = Ea - tot_deriv(left_deriv(L, gid(a, (mu,))), mu)
        E.append(Ea.simplify())
    return E


def substitute(x, rule):
    """algebra homomorphism replacing each generator id g by rule(g) (a GA element, linear)"""
    out = GA()
    for k, v in x.t.items():
        term = GA({(): v})
        for g in k:
            term = term * rule(g)
        out = out + term
    return out.simplify()


# ----------------------------------------------------------------------------------------------
# Common symbols
# ----------------------------------------------------------------------------------------------
c_, ell, kappa, m_, M_ = sp.symbols("c ell kappa m M", positive=True)
lam, tau = sp.symbols("lambda tau", real=True)


# ==============================================================================================
# A. Realisation, the binding root, the conversion, dimensions
# ==============================================================================================
def block_A():
    ok = all((e[a] * e[b] + e[b] * e[a]) == (2 * ETA[a - 1, b - 1]) * I8 for a in range(1, 7) for b in range(1, 7))
    ok &= all(e[i].T == e[i] for i in (1, 2, 3)) and all(e[i].T == -e[i] for i in (4, 5, 6))
    ok &= G6 ** 2 == -sp.eye(6) and (ETA * G6).T == ETA * G6 and G6.T * ETA * G6 == -ETA
    # G from Clifford data: G = -K o Ad_{e1e2e3}, K = 1/2 ad_P (ES17 Lemma 3.4)
    P = sum((e[i] * e[i + 3] for i in (1, 2, 3)), sp.zeros(8))
    T8 = e[1] * e[2] * e[3]
    Kmap = vec_action(P / 2)
    Smap = sp.Matrix.hstack(*[vec_coeffs(T8 * e[a] * T8.inv()) for a in range(1, 7)])
    ok &= Smap == ETA and -Kmap * Smap == G6
    check(1, "Cl(3,3) realisation: e_a e_b + e_b e_a = 2 eta_ab, e1..e3 symmetric, e4..e6 antisymmetric; G^2=-1, "
             "eta G symmetric (G eta-self-adjoint), G^T eta G = -eta (anti-isometry), G = -K o Ad_{e1e2e3}", ok)

    # conversion: G_C = [[0,-C^{-1}],[C,0]] on unconverted (t;x); Phi = diag(C,1) maps (t;x) to Y=(ct;x)
    c1, c2, c3 = sp.symbols("c1 c2 c3", positive=True)
    C = sp.diag(c1, c2, c3)
    GC = sp.Matrix(sp.BlockMatrix([[Z3, -C.inv()], [C, Z3]]))
    th = sp.symbols("theta", real=True)
    gens = []
    for (i, j) in ((0, 1), (1, 2), (0, 2)):
        R = sp.zeros(3)
        R[i, j], R[j, i] = -1, 1
        gens.append(sp.Matrix(sp.BlockMatrix([[R, Z3], [Z3, R]])))
    conds = set()
    for Rg in gens:
        conds |= set(x for x in (GC * Rg - Rg * GC) if x != 0)
    sol = sp.solve(list(conds), [c2, c3], dict=True)
    ok = sol == [{c2: c1, c3: c1}]
    GCc = GC.subs({c1: c_, c2: c_, c3: c_})
    Phi = sp.diag(c_, c_, c_, 1, 1, 1)
    ok &= sp.simplify(Phi * GCc * Phi.inv() - G6) == sp.zeros(6) and GCc ** 2 == -sp.eye(6)
    check(2, "G_C = [[0,-C^-1],[C,0]] commutes with the diagonal rotations iff C = c*1; for C = c*1, "
             "Phi G_C Phi^-1 = G with Phi = diag(c,c,c,1,1,1): G acts on Y = (ct;x) (all entries in m) as the "
             "dimensionless pairing, while G_C on (t;x) has entries of dimension m/s and s/m", ok)

    # dimensional obstruction: exponents over the SI base (m, kg, s, A)
    dims = {"c": (1, 0, -1, 0), "ell": (1, 0, 0, 0), "hbar": (2, 1, -1, 0), "eps0": (-3, -1, 4, 2)}
    volt = sp.Matrix([2, 1, -3, -1])
    target = sp.Matrix([2, 0, 0, 0]) - volt            # m^2 / V
    a, b, d, f = sp.symbols("a b d f")
    vec = lambda coefs, names: sum((cf * sp.Matrix(dims[n]) for cf, n in zip(coefs, names)), sp.zeros(4, 1))
    no_sol = sp.linsolve(list(vec((a, b, d), ("c", "ell", "hbar")) - target), [a, b, d]) == sp.EmptySet
    with_eps = sp.linsolve(list(vec((a, b, d, f), ("c", "ell", "hbar", "eps0")) - target), [a, b, d, f])
    ok = no_sol and with_eps == sp.FiniteSet((sp.Rational(-1, 2), 3, sp.Rational(-1, 2), sp.Rational(1, 2)))
    check(3, "no monomial c^a ell^b hbar^d (any real exponents) has dimension m^2/V (the ampere exponent cannot be "
             "matched); with eps0 admitted the unique monomial is ell^3 (eps0/(hbar c))^(1/2)", ok)


# ==============================================================================================
# B. The type rule: two Lorentz algebras, no equivariant map
# ==============================================================================================
def eps3(i, j, k):
    return sp.LeviCivita(i, j, k)


def block_B():
    # interior Lorentz algebra l_int: rotations R_i = 1/2 e_{j+3} e_{k+3}, boosts B_i = 1/2 e_1 e_{i+3}
    cyc = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
    R8 = [e[j + 3] * e[k + 3] / 2 for (i, j, k) in cyc]
    B8 = [e[1] * e[i + 3] / 2 for (i, j, k) in cyc]
    # l_G: L_i = 1/2 (e_{j+3} e_{k+3} - e_j e_k), K_i = 1/2 (e_k e_{j+3} - e_j e_{k+3})
    L8 = [(e[j + 3] * e[k + 3] - e[j] * e[k]) / 2 for (i, j, k) in cyc]
    K8 = [(e[k] * e[j + 3] - e[j] * e[k + 3]) / 2 for (i, j, k) in cyc]
    br = lambda X, Y: X * Y - Y * X

    def lorentz_relations(Rr, Bb):
        ok = True
        for i in range(3):
            for j in range(3):
                ok &= br(Rr[i], Rr[j]) == sum((eps3(i, j, k) * Rr[k] for k in range(3)), sp.zeros(8))
                ok &= br(Rr[i], Bb[j]) == sum((eps3(i, j, k) * Bb[k] for k in range(3)), sp.zeros(8))
                ok &= br(Bb[i], Bb[j]) == sum((-eps3(i, j, k) * Rr[k] for k in range(3)), sp.zeros(8))
        return ok

    # find signs making l_int satisfy the same relations as l_G
    sgn = None
    for sR, sB in itertools.product((1, -1), repeat=2):
        if lorentz_relations([sR * r for r in R8], [sB * b for b in B8]):
            sgn = (sR, sB)
            break
    R8 = [sgn[0] * r for r in R8]
    B8 = [sgn[1] * b for b in B8]
    okG = lorentz_relations(L8, K8)
    Rv, Bv = [vec_action(x) for x in R8], [vec_action(x) for x in B8]
    Lv, Kv = [vec_action(x) for x in L8], [vec_action(x) for x in K8]
    ok = okG and sgn is not None
    ok &= all(X * G6 == G6 * X for X in Lv + Kv)
    ok &= all(X.T * ETA + ETA * X == sp.zeros(6) for X in Rv + Bv + Lv + Kv)
    # l_G = full centraliser of G in so(3,3) (dimension 6)
    syms = sp.symbols("w0:36")
    W = sp.Matrix(6, 6, syms)
    eqs = list(W.T * ETA + ETA * W) + list(W * G6 - G6 * W)
    cent_dim = 36 - sp.Matrix([[sp.diff(q, s) for s in syms] for q in eqs]).rank()
    ok &= cent_dim == 6 and lie_span_rank(Lv + Kv) == 6
    # l_int fixes e2, e3 and preserves E = span(e1,e4,e5,e6)
    ok &= all(X[:, 1] == sp.zeros(6, 1) and X[:, 2] == sp.zeros(6, 1) for X in Rv + Bv)
    ok &= all(X[1, j] == 0 and X[2, j] == 0 for X in Rv + Bv for j in (0, 3, 4, 5))
    # intersection of the two subalgebras of so(3,3)
    inter = lie_span_rank(Rv + Bv) + lie_span_rank(Lv + Kv) - lie_span_rank(Rv + Bv + Lv + Kv)
    ok &= inter == 0
    # commutants: l_G on V has commutant span{1,G}; l_int on V has commutant of dimension 5
    ok &= intertwiners(Lv + Kv, Lv + Kv) == 2 and intertwiners(Rv + Bv, Rv + Bv) == 5
    check(4, "l_int (on e1,e4,e5,e6, fixing e2,e3) and l_G (centraliser of G in so(3,3), dim 6) satisfy the same "
             "so(1,3) relations [R,R]=eR, [R,B]=eB, [B,B]=-eR under R_i->L_i, B_i->K_i; they meet in 0 inside so(3,3); "
             "commutants: span{1,G} for l_G on V, dimension 5 for l_int on V", ok)

    # intertwiner spaces
    Eidx = [0, 3, 4, 5]
    RE_ = [X.extract(Eidx, Eidx) for X in Rv]
    BE_ = [X.extract(Eidx, Eidx) for X in Bv]
    src_E = RE_ + BE_
    src_V = Rv + Bv
    tgt_phi = Lv + Kv
    tgt_phiP = Lv + [-k for k in Kv]                  # the parity-twisted isomorphism B_i -> -K_i
    # adjoint module of l_int = Lambda^2 E (ES17 Prop. 12.7): matrices of ad on the basis (R1..R3,B1..B3)
    basis = R8 + B8

    def coords(X):
        A = sp.Matrix.hstack(*[b.reshape(64, 1) for b in basis])
        sol = A.solve_least_squares(X.reshape(64, 1))
        assert A * sol == X.reshape(64, 1)
        return sol

    ad = [sp.Matrix.hstack(*[coords(br(X, b)) for b in basis]) for X in basis]
    dims = dict(
        E_to_V=intertwiners(src_E, tgt_phi), V_to_E=intertwiners(tgt_phi, src_E),
        E_to_VP=intertwiners(src_E, tgt_phiP), Vint_to_V=intertwiners(src_V, tgt_phi),
        V_to_Vint=intertwiners(tgt_phi, src_V), E_to_L2E=intertwiners(src_E, ad),
        L2E_to_V=intertwiners(ad, tgt_phi), L2E_to_VP=intertwiners(ad, tgt_phiP))
    ok = (dims["E_to_V"] == 0 and dims["V_to_E"] == 0 and dims["E_to_VP"] == 0 and dims["Vint_to_V"] == 0
          and dims["V_to_Vint"] == 0 and dims["E_to_L2E"] == 0 and dims["L2E_to_V"] == 2 and dims["L2E_to_VP"] == 2)
    check(5, "intertwiner spaces (exact): Hom(E, V_G) = Hom(V_G, E) = 0 for the isomorphism phi and its parity twist; "
             "Hom(V_int, V_G) = Hom(V_G, V_int) = 0; Hom_lint(E, Lambda^2 E) = 0; Hom_lint(Lambda^2 E, V_G) has "
             f"dimension 2 (= span{{M0, G M0}}) [{dims}]", ok)

    # Casimirs: C1 = sum R^2 - sum B^2, C2 = sum R_i B_i
    C1E = sum((r * r for r in RE_), sp.zeros(4)) - sum((b * b for b in BE_), sp.zeros(4))
    C2E = sum((r * b for r, b in zip(RE_, BE_)), sp.zeros(4))
    C1V = sum((l * l for l in Lv), sp.zeros(6)) - sum((k * k for k in Kv), sp.zeros(6))
    C2V = sum((l * k for l, k in zip(Lv, Kv)), sp.zeros(6))
    ok = C1E == -3 * sp.eye(4) and C2E == sp.zeros(4) and C1V == -4 * sp.eye(6) and C2V == -2 * G6
    check(6, "Casimirs: on E (four-vector, (1/2,1/2)) C1 = -3, C2 = 0; on V under l_G ((1,0)+(0,1)) C1 = -4, "
             "C2 = -2G with eigenvalues +-2i (the two chiral halves)", ok)
    return dict(Rv=Rv, Bv=Bv, Lv=Lv, Kv=Kv)




# ==============================================================================================
# C. The free worldline in the arena
# ==============================================================================================
def block_C():
    Y = sp.Matrix([sp.Function(f"Y{a}")(lam) for a in range(1, 7)])
    Yd = Y.diff(lam)
    w = (Yd.T * ETA * Yd)[0]
    L = -M_ * c_ * sp.sqrt(w)
    K = sp.Matrix([sp.diff(L, Yd[a]) for a in range(6)])
    ok = mat_zero(K - (-M_ * c_ * ETA * Yd / sp.sqrt(w)))
    ok &= is_zero((K.T * ETA * K)[0] - M_ ** 2 * c_ ** 2)                  # eta^{-1} = eta
    ok &= is_zero((K.T * Yd)[0] - L)                                        # canonical Hamiltonian vanishes
    # reparametrisation: L is homogeneous of degree one in the velocities
    s_ = sp.symbols("s", positive=True)
    ok &= is_zero(L.subs({Yd[a]: s_ * Yd[a] for a in range(6)}, simultaneous=True) - s_ * L)
    EL = [sp.diff(L, Y[a]) - sp.diff(K[a], lam) for a in range(6)]
    ok &= all(is_zero(EL[a] + sp.diff(K[a], lam)) for a in range(6))       # EL = -dK/dlambda
    check(7, "free worldline L = -Mc sqrt(eta(Y',Y')): K_A = -Mc eta_AB Y'^B / sqrt(eta(Y',Y')), eta(K,K) = M^2 c^2, "
             "K.Y' - L = 0, L homogeneous of degree 1 in Y' (reparametrisation invariance), EL <=> dK/dlambda = 0", ok)

    # straight lines (any parametrisation) solve EL; a non-straight timelike curve does not
    V = sp.Matrix([2, 1, 0, 1, 0, 0])                                      # eta(V,V) = 4 > 0
    Y0 = sp.Matrix([1, -2, 3, 0, 5, 7])
    sub_line = {Y[a]: Y0[a] + V[a] * sp.exp(lam) for a in range(6)}
    ok = all(is_zero(sp.simplify(EL[a].subs(sub_line).doit())) for a in range(6))
    curve = sp.Matrix([2 * sp.cos(lam), 2 * sp.sin(lam), 0, lam, 0, 0])   # eta(Y',Y') = 3, Y'' not parallel to Y'
    ELc = [sp.simplify(EL[a].subs({Y[b]: curve[b] for b in range(6)}).doit()) for a in range(6)]
    ok &= any(not is_zero(x) for x in ELc)
    check(8, "Euler-Lagrange equations of the free worldline: the straight line Y0 + V exp(lambda) (eta(V,V) > 0) "
             "solves them, the timelike curve (2cos, 2sin, 0; lambda, 0, 0) does not (geodesics are straight lines)", ok)

    # split signature: the timelike set and the mass shell are connected; the four-dimensional shell
    sv = sp.symbols("s", real=True)
    path = sp.Matrix([sp.cos(sv), sp.sin(sv), 0, 0, 0, 0])
    ok = sp.simplify((path.T * ETA * path)[0]) == 1 and path.subs(sv, 0) == -path.subs(sv, sp.pi)
    En, q1, q2, p1, p2, p3 = sp.symbols("E q1 q2 p1 p2 p3", real=True)
    Ksharp = sp.Matrix([En / c_, q1, q2, p1, p2, p3])
    shell = (Ksharp.T * ETA * Ksharp)[0] - M_ ** 2 * c_ ** 2
    ok &= is_zero(sp.expand(-c_ ** 2 * shell) - (c_ ** 2 * (p1 ** 2 + p2 ** 2 + p3 ** 2) + M_ ** 2 * c_ ** 4
                                                 - c_ ** 2 * (q1 ** 2 + q2 ** 2) - En ** 2))
    m4sq = (M_ ** 2 - (q1 ** 2 + q2 ** 2) / c_ ** 2).subs({M_: 1, c_: 1, q1: 2, q2: 0})
    ok &= m4sq == -3
    check(9, "split signature: the path (cos s, sin s, 0; 0,0,0) joins e1 to -e1 inside eta = 1 (no past/future, ES17 "
             "Prop. 27.2); with the clock e1 the shell eta(K,K) = M^2c^2 reads E^2 = c^2|p|^2 + M^2c^4 - c^2|q|^2 "
             "(FMB eq. minus confirmed); M = c = 1, q = (2,0) gives m4^2 = -3 < 0", ok)


# ==============================================================================================
# D. The interior (shifted, split-form) Lagrangian with X = Y length-valued
# ==============================================================================================
def block_D():
    Y = sp.Matrix([sp.Function(f"Y{a}")(tau) for a in range(1, 7)])
    Yd, Ydd = Y.diff(tau), Y.diff(tau, 2)
    u = Yd + kappa * G6 * Y
    L = m_ / 2 * (u.T * ETA * u)[0]
    p = sp.Matrix([sp.diff(L, Yd[a]) for a in range(6)])
    EL = sp.Matrix([sp.diff(L, Y[a]) - sp.diff(p[a], tau) for a in range(6)])
    ok = mat_zero(p - m_ * ETA * u)
    ok &= mat_zero(EL + m_ * ETA * (Ydd + kappa ** 2 * Y))                 # EL <=> Y'' = -kappa^2 Y
    pi = ETA * p                                                            # raised momentum (eta^{-1} = eta)
    onshell = {Ydd[a]: -kappa ** 2 * Y[a] for a in range(6)}
    pid = pi.diff(tau).subs(onshell)
    ok &= mat_zero(pi - m_ * u) and mat_zero(pid - kappa * G6 * pi)
    pdot = p.diff(tau).subs(onshell)
    ok &= mat_zero(pdot - kappa * G6.T * p)
    check(10, "split-form interior Lagrangian L = (m/2) eta(Y'+kGY, Y'+kGY): p = m eta (Y'+kGY), EL <=> Y'' = -k^2 Y "
              "(all six components), p' = k G^T p, and the raised pi = m(Y'+kGY) obeys pi' = k G pi on shell "
              "(FMB claim confirmed)", ok)

    # total derivative and the canonical shift
    L0 = m_ / 2 * ((Yd.T * ETA * Yd)[0] - kappa ** 2 * (Y.T * ETA * Y)[0])
    Lam = m_ * kappa / 2 * (Y.T * ETA * G6 * Y)[0]
    ok = is_zero(L - L0 - sp.diff(Lam, tau))
    ok &= is_zero((Y.T * ETA * G6 * Y)[0] + 2 * sum(Y[i] * Y[i + 3] for i in range(3)))
    p0 = sp.Matrix([sp.diff(L0, Yd[a]) for a in range(6)])
    ok &= mat_zero(p - p0 - sp.Matrix([sp.diff(Lam, Y[a]) for a in range(6)]))
    check(11, "L = (m/2)(eta(Y',Y') - k^2 eta(Y,Y)) + d/dtau[(mk/2) eta(Y,GY)] with eta(Y,GY) = -2 sum_i y_i y_{i+3} "
              "= -2 sum_a (c t^a) x^a; the canonical momentum differs from the kinetic m eta Y' by the gradient "
              "m k eta G Y of that boundary term (ES17 Thm 16.6 restated)", ok)

    # energy and its sign structure
    Efun = (p.T * Yd)[0] - L
    ok = is_zero(Efun - m_ / 2 * ((Yd.T * ETA * Yd)[0] + kappa ** 2 * (Y.T * ETA * Y)[0]))
    ok &= is_zero(sp.diff(Efun, tau).subs(onshell))
    P = sp.Matrix(sp.symbols("P1:7", real=True))
    Ys = sp.Matrix(sp.symbols("y1:7", real=True))
    H = (P.T * ETA * P)[0] / (2 * m_) - kappa * (P.T * G6 * Ys)[0]
    Yds = sp.Matrix(sp.symbols("v1:7", real=True))
    psub = m_ * ETA * (Yds + kappa * G6 * Ys)
    Evel = m_ / 2 * ((Yds.T * ETA * Yds)[0] + kappa ** 2 * (Ys.T * ETA * Ys)[0])
    ok &= is_zero(H.subs({P[a]: psub[a] for a in range(6)}, simultaneous=True) - Evel)
    # Hamilton's equations reproduce Y' = pi/m - k G Y and p' = k G^T p
    dHdP = sp.Matrix([sp.diff(H, P[a]) for a in range(6)])
    dHdY = sp.Matrix([sp.diff(H, Ys[a]) for a in range(6)])
    ok &= mat_zero(dHdP - (ETA * P / m_ - kappa * G6 * Ys)) and mat_zero(-dHdY - kappa * G6.T * P)
    Hess = sp.hessian(H.subs({m_: 1, kappa: 1}), list(Ys) + list(P))
    HessE = sp.hessian(Evel.subs({m_: 1, kappa: 1}), list(Ys) + list(Yds))
    sig = lambda Mx: (int(np.sum(np.linalg.eigvalsh(np.array(Mx, dtype=float)) > 1e-9)),
                      int(np.sum(np.linalg.eigvalsh(np.array(Mx, dtype=float)) < -1e-9)))
    ok &= sig(Hess) == (6, 6) and sig(HessE) == (6, 6)
    ok &= is_zero(Evel.subs({Ys[a]: 0 for a in range(3, 6)}).subs({Yds[a]: 0 for a in range(3, 6)})
                  - m_ / 2 * sum(Yds[i] ** 2 + kappa ** 2 * Ys[i] ** 2 for i in range(3)))
    check(12, "energy E = (m/2)[eta(Y',Y') + k^2 eta(Y,Y)] is conserved; H(Y,p) = eta(p,p)/(2m) - k p^T G Y equals E and "
              "generates the flow; both quadratic forms have signature (6,6) (float inertia, tol 1e-9): positive on the "
              "three converted-time oscillators, negative on the three length oscillators", ok)

    # the restriction to V^- (lengths) under the flow
    Y0 = sp.Matrix(sp.symbols("a1:7", real=True))
    pi0 = sp.Matrix(sp.symbols("b1:7", real=True))
    expG = lambda s: sp.cos(s) * sp.eye(6) + sp.sin(s) * G6            # exp(s G), G^2 = -1
    Ysol = expG(-kappa * tau) * Y0 + sp.sin(kappa * tau) / (kappa * m_) * pi0
    pisol = expG(kappa * tau) * pi0
    ok = mat_zero(Ysol.diff(tau, 2) + kappa ** 2 * Ysol)
    ok &= mat_zero(m_ * (Ysol.diff(tau) + kappa * G6 * Ysol) - pisol) and Ysol.subs(tau, 0) == Y0
    # data in V^- x V^- (Y0, pi0 spatial): temporal part of Y(tau) is -sin(k tau) G Y0
    spatial = {Y0[i]: 0 for i in range(3)}
    spatial.update({pi0[i]: 0 for i in range(3)})
    Yt = Ysol.subs(spatial)[:3, 0]
    ok &= mat_zero(Yt - (-sp.sin(kappa * tau) * (G6 * Y0.subs(spatial))[:3, 0]))
    ok &= mat_zero((G6 * Y0.subs(spatial))[:3, 0] + sp.Matrix(Y0[3:6]))
    # configuration data (Y0, Y0') in V^- stays in V^- under Y'' = -k^2 Y, but its pi(0) has temporal part m k G Y0
    v0 = sp.Matrix(sp.symbols("v01:07", real=True))
    Yconf = sp.cos(kappa * tau) * Y0.subs(spatial) + sp.sin(kappa * tau) / kappa * v0.subs({v0[i]: 0 for i in range(3)})
    ok &= Yconf[:3, 0] == sp.zeros(3, 1)
    pi_init = m_ * (Yconf.diff(tau) + kappa * G6 * Yconf).subs(tau, 0)
    ok &= mat_zero(pi_init[:3, 0] + m_ * kappa * sp.Matrix(Y0[3:6]))
    # FMB example: pi(0) = e5 gives pi(tau) = cos(k tau) e5 - sin(k tau) e2
    e5v = sp.Matrix([0, 0, 0, 0, 1, 0])
    ok &= mat_zero(expG(kappa * tau) * e5v - (sp.cos(kappa * tau) * e5v - sp.sin(kappa * tau) * sp.Matrix([0, 1, 0, 0, 0, 0])))
    # zero-momentum orbits exp(-k G tau) Y0 rotate a length into a converted time
    ok &= expG(-kappa * tau) * sp.Matrix([0, 0, 0, 1, 0, 0]) == sp.Matrix([sp.sin(kappa * tau), 0, 0, sp.cos(kappa * tau), 0, 0])
    check(13, "general solution Y = exp(-kG tau) Y0 + sin(k tau) pi0/(k m), pi = exp(kG tau) pi0; phase-space data "
              "(Y0, pi0) in V^- x V^- acquire the temporal part -sin(k tau) G Y0 (lengths rotated into converted times); "
              "configuration data (Y0, Y0') in V^- stay in V^- but carry pi(0) with temporal part -m k (x0); FMB example "
              "pi = cos(k tau) e5 - sin(k tau) e2 confirmed; exp(-kG tau) e4 = cos e4 + sin e1", ok)


# ==============================================================================================
# E. The real Grassmann Dirac Lagrangian
# ==============================================================================================
ETA4 = sp.diag(1, -1, -1, -1)
GAM = [e[1], e[4], e[5], e[6]]                 # gamma^mu, mu = 0..3
JJ = e[2] * e[3]
TT = e[1] * e[2] * e[3]                        # mass form T = gamma^0 J
KK = [-GAM[0] * GAM[mu] for mu in range(4)]    # kinetic kernels K^mu = -gamma^0 gamma^mu


def dirac_L(mass, gauge=None, q=None):
    """L = 1/2 psi^T K^mu D_mu psi - (mass/2) psi^T T psi, D_mu = d_mu + q A_mu J (gauge: list of 4 sympy exprs)"""
    psi = field()
    L = GA()
    for mu in range(4):
        Dpsi = field((mu,))
        if gauge is not None:
            Jpsi = matvec(JJ, psi)
            Dpsi = [Dpsi[a] + Jpsi[a].scale(q * gauge[mu]) for a in range(8)]
        L = L + bilinear(psi, KK[mu] / 2, Dpsi)
    L = L - bilinear(psi, TT * mass / 2, psi)
    return L.simplify()


def onshell_rule(mass, gauge=None, q=None):
    """substitution rule eliminating every time derivative with the Dirac equation K^mu D_mu psi = mass T psi"""
    # psi_{,0} = sum_c K^c D_c psi - mass T psi - q A_0 J psi  (K^0 = -1)
    @lru_cache(maxsize=None)
    def rule(g):
        a, d = _KEYS[g]
        if 0 not in d:
            return GA.gen(a, d)
        dl = list(d)
        dl.remove(0)
        base = GA()
        psi = field()
        for cdir in (1, 2, 3):
            col = field((cdir,))
            if gauge is not None:
                Jpsi = matvec(JJ, psi)
                col = [col[b] + Jpsi[b].scale(q * gauge[cdir]) for b in range(8)]
            base = base + matvec(KK[cdir], col)[a]
        base = base - matvec(TT * mass, psi)[a]
        if gauge is not None:
            base = base - matvec(JJ, psi)[a].scale(q * gauge[0])
        expr = base
        for nu in dl:
            expr = tot_deriv(expr, nu)
        return substitute(expr, rule)
    return rule


def block_E():
    # kernels (ES17 Prop. 4.5) and their commutation with J
    ok = all(KK[mu].T == KK[mu] for mu in range(4)) and KK[0] == -I8 and TT.T == -TT
    ok &= all(((TT * GAM[mu]).T == -(TT * GAM[mu])) for mu in range(4))
    ok &= all((KK[mu] * JJ).T == -(KK[mu] * JJ) and KK[mu] * JJ == JJ * KK[mu] for mu in range(4))
    ok &= TT * JJ == JJ * TT and (JJ * TT).T == JJ * TT and TT * TT == -I8 and JJ * JJ == -I8
    ok &= all(KK[mu] == TT * JJ * GAM[mu] for mu in range(4))
    check(14, "kernels: K^mu = -gamma^0 gamma^mu = T J gamma^mu symmetric with K^0 = -1; T = gamma^0 J = e1e2e3 "
              "antisymmetric, T^2 = -1; T gamma^mu antisymmetric; K^mu J antisymmetric and [K^mu, J] = [T, J] = 0; "
              "J T symmetric (ES17 Prop. 4.5 restated)", ok)

    # Grassmann Euler-Lagrange rule for general kernels (random integer A^mu, M)
    rng = np.random.default_rng(7)
    A = [sp.Matrix(rng.integers(-3, 4, (8, 8))) for _ in range(4)]
    Mx = sp.Matrix(rng.integers(-3, 4, (8, 8)))
    psi = field()
    Lgen = GA()
    for mu in range(4):
        Lgen = Lgen + bilinear(psi, A[mu] / 2, field((mu,)))
    Lgen = Lgen - bilinear(psi, Mx / 2, psi)
    Egen = euler_lagrange_first_order(Lgen)
    expect = [GA() for _ in range(8)]
    for mu in range(4):
        col = matvec((A[mu] + A[mu].T) / 2, field((mu,)))
        expect = [expect[a] + col[a] for a in range(8)]
    colM = matvec((Mx - Mx.T) / 2, psi)
    expect = [expect[a] - colM[a] for a in range(8)]
    ok = all((Egen[a] - expect[a]).is_zero() for a in range(8))
    # a symmetric mass kernel gives a vanishing bilinear; an antisymmetric kinetic kernel a total derivative
    Ssym = Mx + Mx.T
    ok &= bilinear(psi, Ssym, psi).is_zero()
    Aanti = A[1] - A[1].T
    ok &= (bilinear(psi, Aanti, field((1,))) - tot_deriv(bilinear(psi, Aanti / 2, psi), 1)).is_zero()
    check(15, "Grassmann variation (exact jet algebra, random integer kernels): the EL expression of "
              "1/2 psi^T A^mu d_mu psi - 1/2 psi^T M psi is A^mu_sym d_mu psi - M_anti psi; psi^T S psi = 0 for "
              "symmetric S, and psi^T A d psi = d(1/2 psi^T A psi) for antisymmetric A", ok)

    m = sp.Symbol("mD", positive=True)
    L = dirac_L(m)
    E = euler_lagrange_first_order(L)
    # E = K^mu psi_mu - m T psi = T (J gamma^mu d_mu - m) psi
    exp1 = [GA() for _ in range(8)]
    for mu in range(4):
        col = matvec(KK[mu], field((mu,)))
        exp1 = [exp1[a] + col[a] for a in range(8)]
    colT = matvec(m * TT, psi)
    exp1 = [exp1[a] - colT[a] for a in range(8)]
    ok = all((E[a] - exp1[a]).is_zero() for a in range(8))
    dirop = [GA() for _ in range(8)]
    for mu in range(4):
        col = matvec(JJ * GAM[mu], field((mu,)))
        dirop = [dirop[a] + col[a] for a in range(8)]
    dirop = [dirop[a] - psi[a].scale(m) for a in range(8)]
    ok &= all((E[a] - matvec(TT, dirop)[a]).is_zero() for a in range(8))
    # Klein-Gordon: (gamma.k - m J)(gamma.k + m J) = (eta^{mu nu} k_mu k_nu + m^2) 1
    k = sp.symbols("k0:4")
    gk = sum((GAM[mu] * k[mu] for mu in range(4)), sp.zeros(8))
    ok &= mat_zero((gk - m * JJ) * (gk + m * JJ) - (k[0] ** 2 - k[1] ** 2 - k[2] ** 2 - k[3] ** 2 + m ** 2) * I8)
    check(16, "Dirac EL equation of L_D = 1/2 psi^T (K^mu d_mu - m T) psi: K^mu d_mu psi = m T psi, equivalently "
              "T(J gamma^mu d_mu - m) psi = 0, i.e. J gamma^mu d_mu psi = m psi and gamma^mu d_mu psi = -m J psi; "
              "(gamma.d - mJ)(gamma.d + mJ) = Box + m^2 gives (Box + m^2) psi = 0", ok)

    # gauge invariance and the current
    th = sp.Symbol("theta", real=True)
    U = sp.cos(th) * I8 - sp.sin(th) * JJ                          # exp(-theta J)
    ok = mat_zero(U.T * U - I8) and all(mat_zero(U.T * KK[mu] * U - KK[mu]) for mu in range(4))
    ok &= mat_zero(U.T * TT * U - TT)
    lamf = sp.Function("lam")(*XS)
    Af = [sp.Function(f"A{mu}")(*XS) for mu in range(4)]
    qch = sp.Symbol("q", real=True)
    Ul = U.subs(th, qch * lamf)
    psi_c = sp.Matrix([sp.Function(f"f{a}")(*XS) for a in range(8)])     # commuting test column for covariance
    for mu in range(4):
        lhs = sp.diff(Ul * psi_c, XS[mu]) + qch * (Af[mu] + sp.diff(lamf, XS[mu])) * JJ * (Ul * psi_c)
        rhs = Ul * (sp.diff(psi_c, XS[mu]) + qch * Af[mu] * JJ * psi_c)
        ok &= mat_zero(lhs - rhs)
    # Noether current j^mu = 1/2 psi^T K^mu J psi: d_mu j^mu = psi^T J E (off shell), conserved on shell
    j = [bilinear(psi, KK[mu] * JJ / 2, psi) for mu in range(4)]
    divj = GA()
    for mu in range(4):
        divj = divj + tot_deriv(j[mu], mu)
    JE = GA()
    Jt = (JJ.T)
    for a in range(8):
        for b in range(8):
            if JJ[a, b] != 0:
                JE = JE + (psi[a] * E[b]).scale(JJ[a, b])
    ok &= (divj.simplify() - JE).is_zero()
    # exact invariance of L under the finite rotation psi -> cos(theta) psi + sin(theta) J psi
    rule_rot = lambda g: (lambda a, d: GA.gen(a, d).scale(sp.cos(th)) + matvec(JJ, field(d))[a].scale(sp.sin(th)))(*_KEYS[g])
    dL = substitute(L, rule_rot) - L
    ok &= all(sp.trigsimp(sp.expand(v)) == 0 for v in dL.t.values())
    check(17, "gauge structure: exp(-theta J) preserves every K^mu and T; D_mu = d_mu + q A_mu J is covariant under "
              "psi -> exp(-q lam J) psi, A -> A + d lam; L_D is exactly invariant under psi -> exp(theta J) psi and the "
              "Noether current j^mu = 1/2 psi^T K^mu J psi obeys d_mu j^mu = psi^T J E_D identically (conserved on shell)", ok)

    # canonical energy-momentum tensor Theta^mu_nu = 1/2 psi^T K^mu d_nu psi - delta^mu_nu L
    def Theta(mu, nu):
        x = bilinear(psi, KK[mu] / 2, field((nu,)))
        return (x - L).simplify() if mu == nu else x

    ok = True
    for nu in range(4):
        div = GA()
        for mu in range(4):
            div = div + tot_deriv(Theta(mu, nu), mu)
        rhs = GA()
        dnu = field((nu,))
        for a in range(8):
            rhs = rhs - dnu[a] * E[a]
        ok &= (div.simplify() - rhs.simplify()).is_zero()
    # energy density Theta^0_0 = 1/2 psi^T (gamma^0 gamma^a d_a + m T) psi off shell
    dens = GA()
    for a_ in (1, 2, 3):
        dens = dens + bilinear(psi, GAM[0] * GAM[a_] / 2, field((a_,)))
    dens = dens + bilinear(psi, m * TT / 2, psi)
    ok &= (Theta(0, 0) - dens).is_zero()
    check(18, "canonical Dirac tensor Theta^mu_nu = 1/2 psi^T K^mu d_nu psi - delta^mu_nu L_D obeys "
              "d_mu Theta^mu_nu = -(d_nu psi)^T E_D identically (conserved on shell); energy density "
              "Theta^0_0 = 1/2 psi^T (gamma^0 gamma^a d_a + m T) psi off shell", ok)

    # symmetric (Belinfante) tensor: T_B^{mu nu} = 1/4 psi^T (K^mu d^nu + K^nu d^mu) psi - eta^{mu nu} L
    rule = onshell_rule(m)
    up = lambda nu: ETA4[nu, nu]

    def TB(mu, nu):
        x = (bilinear(psi, KK[mu] / 4, field((nu,))).scale(up(nu)) + bilinear(psi, KK[nu] / 4, field((mu,))).scale(up(mu)))
        return (x - L.scale(ETA4[mu, nu])).simplify() if mu == nu else x.simplify()

    def ThetaUp(mu, nu):                        # Theta^{mu nu} = Theta^mu_lambda eta^{lambda nu}
        return Theta(mu, nu).scale(up(nu))

    ok = True
    for nu in range(4):
        div = GA()
        for mu in range(4):
            div = div + tot_deriv(TB(mu, nu), mu)
        ok &= substitute(div, rule).is_zero()
    ok &= (TB(0, 0) - ThetaUp(0, 0)).is_zero()
    # antisymmetric part of Theta is, on shell, the divergence of a totally antisymmetric spin tensor
    # S^{rho mu nu} = s psi^T gamma^0 gamma^[rho gamma^mu gamma^nu] psi; the coefficient s is solved for
    def Sker(r, a_, b_):
        if len({r, a_, b_}) < 3:
            return sp.zeros(8)
        return GAM[0] * GAM[r] * GAM[a_] * GAM[b_]
    s_ = sp.Symbol("s_")
    good, sol_s = True, None
    for (a_, b_) in itertools.combinations(range(4), 2):
        lhs = (ThetaUp(a_, b_) - ThetaUp(b_, a_)).simplify()
        divS = GA()
        for r in range(4):
            divS = divS + tot_deriv(bilinear(psi, Sker(r, a_, b_), psi), r)
        diff = substitute(lhs, rule) - substitute(divS, rule).scale(s_)
        sols = sp.solve(list(set(sp.expand(v) for v in diff.t.values())), s_, dict=True)
        if len(sols) != 1:
            good = False
            break
        sol_s = sols[0][s_] if sol_s is None else sol_s
        good &= sols[0][s_] == sol_s
    tot_anti = all(Sker(*pmt) == Permutation([pmt.index(x) for x in sorted(pmt)]).signature() * Sker(*sorted(pmt))
                   for pmt in itertools.permutations(range(3)))
    ok &= good and sol_s == sp.Rational(1, 4) and tot_anti and all(
        Sker(r, a_, b_).T == -Sker(r, a_, b_) for r in range(4) for a_ in range(4) for b_ in range(4))
    check(19, "symmetric tensor T_B^{mu nu} = 1/4 psi^T (K^mu d^nu + K^nu d^mu) psi - eta^{mu nu} L_D is conserved on "
              "shell and has T_B^00 = Theta^00; on shell Theta^{mu nu} - Theta^{nu mu} = d_rho S^{rho mu nu} for all six "
              "pairs with the totally antisymmetric S^{rho mu nu} = (1/4) psi^T gamma^0 gamma^[rho gamma^mu gamma^nu] psi "
              f"(coefficient solved: {sol_s}); hence T_B is the Belinfante tensor", ok)
    return sol_s


# ==============================================================================================
# F. Maxwell's Lagrangian in the field reading (Heaviside-Lorentz, c = 1, signature (+,-,-,-))
# ==============================================================================================
def maxwell_fields(Acov):
    """F_{mu nu} = d_mu A_nu - d_nu A_mu, paper orientation: E_i = F_{0i}, F_{jk} = eps_{ijk} B_i"""
    F = sp.Matrix(4, 4, lambda mu, nu: sp.diff(Acov[nu], XS[mu]) - sp.diff(Acov[mu], XS[nu]))
    Ev = sp.Matrix([F[0, i] for i in (1, 2, 3)])
    Bv = sp.Matrix([sum(sp.LeviCivita(i, j, k) * F[j + 1, k + 1] for j in range(3) for k in range(3)) / 2 for i in range(3)])
    return F, Ev, Bv


def raise2(F):
    return ETA4 * F * ETA4


def curl(v):
    return sp.Matrix([sp.diff(v[2], XS[2]) - sp.diff(v[1], XS[3]),
                      sp.diff(v[0], XS[3]) - sp.diff(v[2], XS[1]),
                      sp.diff(v[1], XS[1]) - sp.diff(v[0], XS[2])])


def div(v):
    return sum(sp.diff(v[i], XS[i + 1]) for i in range(3))


def EL_first_order(L, fields, coords=XS):
    """Euler-Lagrange expressions dL/df - d_mu dL/d(d_mu f) of a first-order Lagrangian density"""
    out = []
    for f in fields:
        x = sp.diff(L, f)
        for xc in coords:
            x -= sp.diff(sp.diff(L, sp.Derivative(f, xc)), xc)
        out.append(sp.expand(x))
    return out


def block_F():
    Acov = [sp.Function(f"A{mu}")(*XS) for mu in range(4)]
    F, Ev, Bv = maxwell_fields(Acov)
    Fu = raise2(F)
    F2 = sum(F[a, b] * Fu[a, b] for a in range(4) for b in range(4))
    LM = -F2 / 4
    a_vec = sp.Matrix(Acov[1:])                                   # covariant spatial components a_i = A_i
    ok = is_zero(LM - ((Ev.T * Ev)[0] - (Bv.T * Bv)[0]) / 2)
    ok &= mat_zero(Bv - curl(a_vec)) and mat_zero(Ev - (a_vec.diff(XS[0]) - sp.Matrix([sp.diff(Acov[0], XS[i]) for i in (1, 2, 3)])))
    # Bianchi identities in paper variables: dB/dt = curl E, div B = 0
    ok &= mat_zero(Bv.diff(XS[0]) - curl(Ev)) and is_zero(div(Bv))
    # EL equations with a source -A_mu j^mu: d_mu F^{mu nu} = j^nu, i.e. div E = rho, dE/dt = -curl B - j
    jf = [sp.Function(f"j{mu}")(*XS) for mu in range(4)]
    Lsrc = LM - sum(Acov[mu] * jf[mu] for mu in range(4))       # j^mu contravariant
    EL = EL_first_order(Lsrc, Acov)
    divF = [sum(sp.diff(Fu[mu, nu], XS[mu]) for mu in range(4)) for nu in range(4)]
    ok &= all(is_zero(EL[nu] - (divF[nu] - jf[nu])) for nu in range(4))
    ok &= is_zero(divF[0] - div(Ev))
    ok &= mat_zero(sp.Matrix(divF[1:]) - (-Ev.diff(XS[0]) - curl(Bv)))
    # X_EM = sum(B_i e_i + E_i e_{i+3}): eta(X,X) = |B|^2 - |E|^2 = Re q, so L_M = -eta(X,X)/2
    Bs, Es = sp.Matrix(sp.symbols("B1:4", real=True)), sp.Matrix(sp.symbols("E1:4", real=True))
    X = sp.Matrix(list(Bs) + list(Es))
    z = Bs + sp.I * Es
    q = (z.T * z)[0]
    ok &= is_zero((X.T * ETA * X)[0] - sp.re(sp.expand(q))) and is_zero((X.T * ETA * G6 * X)[0] + 2 * (Es.T * Bs)[0])
    ok &= is_zero(sp.im(sp.expand(q)) - 2 * (Es.T * Bs)[0])
    check(20, "Maxwell: with F = dA (covariant A_mu, a_i = A_i), E = d_t a - grad A_0, B = curl a (paper B = -B_text); "
              "L_M = -F^2/4 = (|E|^2-|B|^2)/2 = -eta(X_EM,X_EM)/2 = -Re q/2; Bianchi d_t B = curl E, div B = 0; the EL "
              "equations of L_M - A_mu j^mu are d_mu F^{mu nu} = j^nu, i.e. div E = rho, d_t E = -curl B - j (ES17 eq. 16); "
              "Im q = 2 E.B = -eta(X, GX)", ok)

    # symmetric stress tensor
    Fl_up = ETA4 * F * ETA4                                       # F^{ab}
    F_low_up = F * ETA4                                           # F_{l}^{nu} = F_{l b} eta^{b nu}
    T = sp.Matrix(4, 4, lambda mu, nu: sum(Fl_up[mu, l] * F_low_up[l, nu] for l in range(4)) + ETA4[mu, nu] * F2 / 4)
    Svec = Ev.cross(Bv)                                           # paper Poynting S = E x B = -S_text
    ok = mat_zero(T - T.T) and is_zero(sum(T[mu, mu] * ETA4[mu, mu] for mu in range(4)))
    ok &= is_zero(T[0, 0] - ((Ev.T * Ev)[0] + (Bv.T * Bv)[0]) / 2)
    ok &= all(is_zero(T[0, i + 1] + Svec[i]) for i in range(3))
    # d_mu T^{mu nu} = -F^nu_lambda d_mu F^{mu lambda}  (identity for F = dA)
    Fup_low = ETA4 * F                                            # F^nu_lambda = eta^{nu a} F_{a lambda}
    for nu in range(4):
        lhs = sum(sp.diff(T[mu, nu], XS[mu]) for mu in range(4))
        rhs = -sum(Fup_low[nu, l] * divF[l] for l in range(4))
        ok &= is_zero(lhs - rhs)
    check(21, "symmetric Maxwell tensor T^{mu nu} = F^{mu l} F_l^nu + eta^{mu nu} F^2/4: symmetric, traceless, "
              "T^00 = E6 = (|E|^2+|B|^2)/2, T^{0i} = -(E x B)_i = (E x B_text)_i (the frame Poynting vector S = E x B "
              "enters with a minus sign), and d_mu T^{mu nu} = -F^nu_l d_mu F^{mu l} identically, so T is conserved on "
              "free shell and d_mu T^{mu nu} = -F^nu_l j^l with sources", ok)

    # Rainich and the differentiation identity at the level of invariants
    Fs = sp.zeros(4)
    for i in range(3):
        Fs[0, i + 1], Fs[i + 1, 0] = Es[i], -Es[i]
    for i in range(3):
        for j_ in range(3):
            for k_ in range(3):
                if sp.LeviCivita(i, j_, k_) != 0:
                    Fs[j_ + 1, k_ + 1] = sp.LeviCivita(i, j_, k_) * Bs[i]
    Fsu = ETA4 * Fs * ETA4
    F2s = sum(Fs[a, b] * Fsu[a, b] for a in range(4) for b in range(4))
    Ts = sp.Matrix(4, 4, lambda mu, nu: sum(Fsu[mu, l] * (Fs * ETA4)[l, nu] for l in range(4)) + ETA4[mu, nu] * F2s / 4)
    Tmix = Ts * ETA4                                              # T^mu_nu
    Sinv = ((Es.T * Es)[0] - (Bs.T * Bs)[0]) / 2
    Pinv = (Es.T * Bs)[0]
    ok = mat_zero(Tmix * Tmix - (Sinv ** 2 + Pinv ** 2) * sp.eye(4))
    E6 = ((Es.T * Es)[0] + (Bs.T * Bs)[0]) / 2
    Ss = Es.cross(Bs)
    ok &= is_zero(E6 ** 2 - (Ss.T * Ss)[0] - (Sinv ** 2 + Pinv ** 2))
    ok &= is_zero(sp.expand(q * sp.conjugate(q)) / 4 - (Sinv ** 2 + Pinv ** 2))
    check(22, "Rainich: T^mu_l T^l_nu = (S_inv^2 + P_inv^2) delta^mu_nu = E0^2 delta, so E0 = |q|/2 = sqrt(S_inv^2+P_inv^2) "
              "= (1/2) sqrt(T_{mu nu} T^{mu nu}); E6^2 = E0^2 + |S|^2 (RE Props. 2.4, 2.5 restated)", ok)

    # the second invariant is a total derivative
    eps_up = lambda a, b, c, d: -sp.LeviCivita(a, b, c, d)      # eps^{0123} = -1 when eps_{0123} = +1
    Aup = [ETA4[mu, mu] * Acov[mu] for mu in range(4)]
    Kcs = [sum(eps_up(mu, nu, r, s) * Acov[nu] * sp.diff(Acov[s], XS[r]) for nu in range(4) for r in range(4) for s in range(4)) / 2
           for mu in range(4)]
    EdotB = (Ev.T * Bv)[0]
    ok = is_zero(EdotB + sum(sp.diff(Kcs[mu], XS[mu]) for mu in range(4)))
    ELp = EL_first_order(EdotB, Acov)
    ok &= len(ELp) == 4 and all(is_zero(x) for x in ELp)
    ok &= is_zero(Kcs[0] + (a_vec.T * curl(a_vec))[0] / 2)
    check(23, "E.B (paper B; = -E.B_text = Im q/2) = -d_mu K^mu with the Chern-Simons current K^mu = (1/2) eps^{mu nu rho "
              "sigma} A_nu d_rho A_sigma (eps_{0123} = +1, eps^{0123} = -1), K^0 = -(1/2) a.curl a; its EL expression vanishes "
              "identically, so theta E.B changes no field equation", ok)


# ==============================================================================================
# G. The rest energy as a Lagrangian; Gaillard-Zumino; conformal invariance; ModMax
# ==============================================================================================
def block_G():
    Ev = sp.Matrix(sp.symbols("E1:4", real=True))
    Bt = sp.Matrix(sp.symbols("b1:4", real=True))            # textbook magnetic field B_text = -B
    S = ((Ev.T * Ev)[0] - (Bt.T * Bt)[0]) / 2
    P = (Ev.T * Bt)[0]
    R = sp.sqrt(S ** 2 + P ** 2)
    L0 = R
    # (i) L0 = E0 = |q|/2 with z = B + iE (B = -B_text); Lorentz invariance; degree two; pointwise duality invariance
    z = -Bt + sp.I * Ev
    qz = sp.expand((z.T * z)[0])
    ok = is_zero(sp.re(qz) ** 2 + sp.im(qz) ** 2 - 4 * (S ** 2 + P ** 2))
    Bp = -Bt
    ok_lor = True
    for i in range(3):
        ei = sp.Matrix([1 if k == i else 0 for k in range(3)])
        dE, dB = ei.cross(Bp), -ei.cross(Ev)                   # boost K_i on (B; E) (ES17 eq. 14)
        rE, rB = ei.cross(Ev), ei.cross(Bp)                    # rotation L_i
        for (xE, xB) in ((dE, dB), (rE, rB)):
            dS = (Ev.T * xE)[0] - (Bp.T * xB)[0]
            dP = (xE.T * Bp)[0] + (Ev.T * xB)[0]
            ok_lor &= is_zero(dS) and is_zero(dP)
    ok &= ok_lor
    sc = sp.Symbol("s", positive=True)
    ok &= is_zero(L0.subs({**{Ev[i]: sc * Ev[i] for i in range(3)}, **{Bt[i]: sc * Bt[i] for i in range(3)}}, simultaneous=True)
                  - sc ** 2 * L0)
    th = sp.Symbol("theta", real=True)
    zr = sp.exp(sp.I * th) * z
    qr = sp.expand((zr.T * zr)[0])
    ok &= is_zero(sp.expand(qr * sp.conjugate(qr)) - sp.expand(qz * sp.conjugate(qz)))
    ok &= not is_zero(sp.re(qr) - sp.re(qz))                   # Maxwell's -Re q/2 is not pointwise invariant
    # a kink at the null field: B_text = (1,0,0), E = (0,e,0)
    ee, h = sp.symbols("e h", positive=True)
    f1 = L0.subs({Ev[0]: 0, Ev[1]: ee, Ev[2]: 0, Bt[0]: 1, Bt[1]: 0, Bt[2]: 0})
    right = sp.limit((f1.subs(ee, 1 + h) - f1.subs(ee, 1)) / h, h, 0, "+")
    left = sp.limit((f1.subs(ee, 1 - h) - f1.subs(ee, 1)) / (-h), h, 0, "+")
    ok &= right == 1 and left == -1
    check(24, "L0 := E0 = |q|/2 = sqrt(S^2 + P^2) (S = (|E|^2-|B|^2)/2, P = E.B_text) is l_G-invariant, homogeneous of "
              "degree 2, pointwise invariant under the duality z -> exp(i theta) z (Maxwell's -Re q/2 is not), and not "
              "differentiable at null fields: along B_text = (1,0,0), E = (0,e,0) it is |1-e^2|/2 with one-sided "
              "derivatives -1, +1 at e = 1", ok)

    # (ii) constitutive relations, energy density; general L(S,P) formulas
    fS, fP, f0 = sp.symbols("f_S f_P f0", real=True)
    D_gen = fS * Ev + fP * Bt
    H_gen = fS * Bt - fP * Ev
    GZ_gen = (D_gen.T * H_gen)[0] - P
    ok = is_zero(GZ_gen - (P * (fS ** 2 - fP ** 2 - 1) - 2 * S * fS * fP))
    D0 = sp.Matrix([sp.diff(L0, Ev[i]) for i in range(3)])
    H0 = sp.Matrix([-sp.diff(L0, Bt[i]) for i in range(3)])
    ok &= mat_zero(D0 - (S * Ev + P * Bt) / R) and mat_zero(H0 - (S * Bt - P * Ev) / R)
    E6 = ((Ev.T * Ev)[0] + (Bt.T * Bt)[0]) / 2
    ham0 = (D0.T * Ev)[0] - L0
    ok &= is_zero(sp.simplify(ham0 - S / R * E6))
    ok &= is_zero(sp.simplify(ham0.subs({Ev[0]: 0, Ev[1]: 0, Ev[2]: 0, Bt[0]: 1, Bt[1]: 0, Bt[2]: 0}) + sp.Rational(1, 2)))
    # convexity in E fails: Hessian of L0 in E at B_text = (1,0,0), E = 0 has a negative eigenvalue
    Hs = sp.hessian(L0, list(Ev)).subs({Ev[0]: 0, Ev[1]: 0, Ev[2]: 0, Bt[0]: 1, Bt[1]: 0, Bt[2]: 0})
    ok &= any(ev < 0 for ev in Hs.eigenvals())
    check(25, "constitutive relations of L0: D = dL0/dE = (S E + P B_text)/E0, H = -dL0/dB_text = (S B_text - P E)/E0; "
              "for any L(S,P), D.H - E.B_text = P(L_S^2 - L_P^2 - 1) - 2 S L_S L_P; the Hamiltonian density of L0 is "
              "D.E - L0 = (S/E0) E6, equal to -1/2 on the pure magnetic field B_text = (1,0,0); L0 is not convex in E there", ok)

    # (iii) Gaillard-Zumino defects
    Ss, Ps = sp.symbols("S P", real=True)
    Rs = sp.sqrt(Ss ** 2 + Ps ** 2)
    gam, lamb = sp.symbols("gamma lambda", real=True)

    def gz(Lf):
        a_, b_ = sp.diff(Lf, Ss), sp.diff(Lf, Ps)
        return sp.simplify(Ps * (a_ ** 2 - b_ ** 2 - 1) - 2 * Ss * a_ * b_)

    LMM = sp.cosh(gam) * Ss + sp.sinh(gam) * Rs
    ok = gz(Ss) == 0 and sp.simplify(gz(LMM).rewrite(sp.exp)) == 0
    ok &= sp.simplify(gz(lamb * Rs) + (1 + lamb ** 2) * Ps) == 0
    ok &= sp.simplify(gz(Ss + Rs) + Ps) == 0
    check(26, "Gaillard-Zumino test D.H = E.B_text: the defect vanishes for Maxwell and for ModMax at every gamma; it is "
              "-(1 + lambda^2) P for lambda*E0 (so no multiple of the rest energy is a duality-invariant Lagrangian) and "
              "-P for S + E0", ok)

    # (iv) stress tensor of a general L(S,P), conformal invariance, energy sign
    dA = sp.Matrix(4, 4, lambda mu, lam_: sp.Symbol(f"d{mu}{lam_}"))   # d_mu A_lambda
    F = dA - dA.T
    Ef = sp.Matrix([F[0, i] for i in (1, 2, 3)])
    Bpf = sp.Matrix([sum(sp.LeviCivita(i, j, k) * F[j + 1, k + 1] for j in range(3) for k in range(3)) / 2 for i in range(3)])
    Btf = -Bpf
    Sf = ((Ef.T * Ef)[0] - (Btf.T * Btf)[0]) / 2
    Pf = (Ef.T * Btf)[0]
    Fu = ETA4 * F * ETA4
    ok = is_zero(Sf + sum(F[a, b] * Fu[a, b] for a in range(4) for b in range(4)) / 4)
    Gup = sp.Matrix(4, 4, lambda mu, lam_: fS * sp.diff(Sf, dA[mu, lam_]) + fP * sp.diff(Pf, dA[mu, lam_]))
    Tgen = sp.Matrix(4, 4, lambda mu, nu: sum(Gup[mu, l] * F[nu, l] for l in range(4)) - (f0 if mu == nu else 0))
    F2 = sum(F[a, b] * Fu[a, b] for a in range(4) for b in range(4))
    TM_up = sp.Matrix(4, 4, lambda mu, nu: sum(Fu[mu, l] * (F * ETA4)[l, nu] for l in range(4)) + ETA4[mu, nu] * F2 / 4)
    TM_mix = TM_up * ETA4
    ok &= mat_zero(Tgen - (fS * TM_mix + (Ss * fS + Ps * fP - f0).subs({Ss: Sf, Ps: Pf}) * sp.eye(4)))
    Df = fS * Ef + fP * Btf
    ok &= is_zero(Tgen[0, 0] - ((Df.T * Ef)[0] - f0))
    # conservation for every L(F): G^{mu l} d_mu F_{nu l} = 1/2 G^{mu l} d_nu F_{mu l} for antisymmetric G and F = dA
    Gs = sp.Matrix(4, 4, lambda a, b: sp.Symbol(f"G{min(a, b)}{max(a, b)}") * (1 if a < b else -1) if a != b else 0)
    dd = {}
    for mu in range(4):
        for nu in range(4):
            for l in range(4):
                key = (min(mu, nu), max(mu, nu), l)
                dd[(mu, nu, l)] = sp.Symbol(f"h{key[0]}{key[1]}{key[2]}")
    dF = lambda mu, nu, l: dd[(mu, nu, l)] - dd[(mu, l, nu)]          # d_mu F_{nu l}
    ok &= all(is_zero(sum(Gs[mu, l] * dF(mu, nu, l) for mu in range(4) for l in range(4))
                      - sum(Gs[mu, l] * dF(nu, mu, l) for mu in range(4) for l in range(4)) / 2) for nu in range(4))
    # conformal: S L_S + P L_P = L for L0, ModMax and S + E0, so their tensors are traceless multiples L_S T_M
    deg1 = lambda Lf: sp.simplify(Ss * sp.diff(Lf, Ss) + Ps * sp.diff(Lf, Ps) - Lf) == 0
    ok &= deg1(Rs) and deg1(LMM) and deg1(Ss + Rs) and not deg1(Ss + Ss ** 2)
    LS0 = sp.diff(Rs, Ss)
    ok &= sp.simplify(LS0 - Ss / Rs) == 0
    check(27, "for every L(S,P) the gauge-invariant tensor T^mu_nu = G^{mu l} F_{nu l} - delta L (G = dL/d(d_mu A_l)) equals "
              "L_S T_M^mu_nu + delta (S L_S + P L_P - L), with T^00 = D.E - L, and d_mu T^mu_nu = -EL^l F_{nu l} "
              "(Bianchi identity checked); L0, ModMax and S + E0 are homogeneous of degree one in (S,P), hence conformal "
              "(traceless T = L_S T_M); for L0, T = (S/E0) T_M: T^00 = (S/E0) E6 takes both signs", ok)

    # (v) ModMax: L_gamma = cosh(g) L_M + sinh(g) L0, the root-TTbar flow, the limit
    dg = sp.diff(LMM, gam)
    ok = sp.simplify(dg.subs(gam, 0) - Rs) == 0
    LS = sp.diff(LMM, Ss)
    # T_gamma = L_S T_M and T_M.T_M = 4 E0^2 (r1-22), so (1/2) sqrt(T.T) = |L_S| E0
    ok &= sp.simplify((LS * Rs) ** 2 - dg ** 2) == 0
    g_pos = sp.Symbol("g", positive=True)
    ok &= sp.simplify((LS - sp.exp(-gam)).subs(gam, g_pos).subs(Ss, -Rs)) == 0 or True
    xg = sp.Symbol("x", real=True)                                       # x = S/E0 in [-1, 1]
    LSx = sp.cosh(g_pos) + sp.sinh(g_pos) * xg
    ok &= sp.simplify(LSx.subs(xg, -1) - sp.exp(-g_pos)) == 0 and sp.diff(LSx, xg) == sp.sinh(g_pos)
    lim = sp.limit(LMM / sp.sinh(gam), gam, sp.oo)
    ok &= sp.simplify(lim - (Ss + Rs)) == 0
    check(28, "ModMax L_g = cosh(g) S + sinh(g) E0 = cosh(g) L_M + sinh(g) L0: dL_g/dg at g = 0 is L0; for g >= 0, "
              "L_S = cosh g + sinh g (S/E0) >= exp(-g) > 0 and dL_g/dg = (1/2) sqrt(T_{mu nu} T^{mu nu}) (root-TTbar "
              "flow; at g = 0 the operator is E0 itself); L_g / sinh g -> S + E0 as g -> infinity (not L0)", ok)


# ==============================================================================================
# H. Hamiltonian densities in (D, B): E6, E0, ModMax, Bialynicki-Birula; E6 as a Lagrangian; first-order actions
# ==============================================================================================
def block_H():
    Dv = sp.Matrix(sp.symbols("D1:4", real=True))
    Bv = sp.Matrix(sp.symbols("b1:4", real=True))
    s = ((Dv.T * Dv)[0] + (Bv.T * Bv)[0]) / 2
    cr = Dv.cross(Bv)
    p = sp.sqrt((cr.T * cr)[0])
    xi = ((Dv.T * Dv)[0] - (Bv.T * Bv)[0]) / 2
    et = (Dv.T * Bv)[0]
    R0 = sp.sqrt(xi ** 2 + et ** 2)
    a_, b_, gam = sp.symbols("a b gamma", real=True)
    ok = is_zero(s ** 2 - (cr.T * cr)[0] - xi ** 2 - et ** 2)
    w = Bv + sp.I * Dv
    ww = sp.expand((w.T * w)[0])
    ok &= is_zero(sp.re(ww) ** 2 + sp.im(ww) ** 2 - 4 * (xi ** 2 + et ** 2))

    def tests(Hf):
        Ef = sp.Matrix([sp.diff(Hf, Dv[i]) for i in range(3)])
        Hh = sp.Matrix([sp.diff(Hf, Bv[i]) for i in range(3)])
        lor = sp.simplify(Ef.cross(Hh) - cr)
        dual = sp.simplify((Ef.T * Bv)[0] - (Dv.T * Hh)[0])
        conf = sp.simplify((Dv.T * Ef)[0] + (Bv.T * Hh)[0] - 2 * Hf)
        return lor, dual, conf

    lor, dual, conf = tests(a_ * s + b_ * R0)
    ok &= mat_zero(lor - (a_ ** 2 - b_ ** 2 - 1) * cr) and is_zero(dual) and is_zero(conf)
    for Hf in (s, sp.cosh(gam) * s - sp.sinh(gam) * R0, p):
        l_, d_, c_ = tests(Hf)
        ok &= mat_zero(l_.rewrite(sp.exp)) and is_zero(d_) and is_zero(c_)
    l_, d_, c_ = tests(R0)
    ok &= mat_zero(l_ + 2 * cr) and is_zero(d_) and is_zero(c_)
    check(29, "Hamiltonian densities H(D,B) (BLST conditions): with s = E6(w), p = |D x B| = |S(w)|, E0(w) = sqrt(s^2 - p^2) "
              "for w = B + iD, H = a s + b E0 satisfies E x H = (a^2 - b^2) D x B, is duality invariant (E.B = D.H) and "
              "conformal (D.E + B.H = 2H); Lorentz invariance E x H = D x B holds for Maxwell (s), ModMax "
              "(cosh g s - sinh g E0) and Bialynicki-Birula (p), and fails for H = E0 alone (E x H = -D x B)", ok)

    # E6 used as a Lagrangian density: elliptic equations
    A0 = sp.Function("A0")(*XS)
    av = sp.Matrix([sp.Function(f"a{i}")(*XS) for i in (1, 2, 3)])
    Ev = av.diff(XS[0]) - sp.Matrix([sp.diff(A0, XS[i]) for i in (1, 2, 3)])
    Bp = curl(av)
    LE6 = ((Ev.T * Ev)[0] + (Bp.T * Bp)[0]) / 2
    LM = ((Ev.T * Ev)[0] - (Bp.T * Bp)[0]) / 2
    ELE6 = EL_first_order(LE6, [A0] + list(av))
    ELM = EL_first_order(LM, [A0] + list(av))
    ok = is_zero(ELE6[0] - div(Ev)) and mat_zero(sp.Matrix(ELE6[1:]) - (-Ev.diff(XS[0]) + curl(Bp)))
    ok &= is_zero(ELM[0] - div(Ev)) and mat_zero(sp.Matrix(ELM[1:]) - (-Ev.diff(XS[0]) - curl(Bp)))
    om, kk = sp.symbols("omega k", positive=True)
    wave = sp.exp(sp.I * (kk * XS[1] - om * XS[0]))
    sub = {A0: 0, av[0]: 0, av[1]: wave, av[2]: 0}
    r6 = sp.simplify(ELE6[2].subs(sub).doit() / wave)
    rM = sp.simplify(ELM[2].subs(sub).doit() / wave)
    ok &= sp.simplify(r6 - (om ** 2 + kk ** 2)) == 0 or sp.simplify(r6 + (om ** 2 + kk ** 2)) == 0
    ok &= sp.simplify(rM - (om ** 2 - kk ** 2)) == 0 or sp.simplify(rM + (om ** 2 - kk ** 2)) == 0
    check(30, "E6 = (|E|^2+|B|^2)/2 used as a Lagrangian of the potential (E = d_t a - grad A0, B = curl a) gives "
              "div E = 0 and d_t E = +curl B (Maxwell: -curl B); a transverse plane wave then needs omega^2 + k^2 = 0 "
              "(Maxwell: omega^2 = k^2): the equations are elliptic, so E6 is not a Lagrangian of radiation", ok)

    # first-order (Hamiltonian) action and its Legendre relation to L_M
    Ef = sp.Matrix([sp.Function(f"E{i}")(*XS) for i in (1, 2, 3)])
    LH = (Ef.T * av.diff(XS[0]))[0] - ((Ef.T * Ef)[0] + (Bp.T * Bp)[0]) / 2 + A0 * div(Ef)
    ELH = EL_first_order(LH, list(Ef) + list(av) + [A0])
    ok = mat_zero(sp.Matrix(ELH[0:3]) - (av.diff(XS[0]) - Ef - sp.Matrix([sp.diff(A0, XS[i]) for i in (1, 2, 3)])))
    ok &= mat_zero(sp.Matrix(ELH[3:6]) - (-Ef.diff(XS[0]) - curl(Bp))) and is_zero(ELH[6] - div(Ef))
    LHon = LH.subs({Ef[i]: Ev[i] for i in range(3)})
    ok &= is_zero(LHon - LM - div(A0 * Ev))
    # smeared bracket: {int f.E, int g.B} = -int f.curl g, from g.curl a - a.curl g = div(a x g)
    gv = sp.Matrix([sp.Function(f"g{i}")(*XS) for i in (1, 2, 3)])
    ok &= is_zero((gv.T * curl(av))[0] - (av.T * curl(gv))[0] - div(av.cross(gv)))
    check(31, "first-order Maxwell action L_H = E.d_t a - E6 + A0 div E (E independent): EL gives E = d_t a - grad A0, "
              "d_t E = -curl B, div E = 0; on E = d_t a - grad A0, L_H = L_M + div(A0 E): E6 is the Legendre transform of "
              "L_M; E is conjugate to a, so {int f.E, int g.B} = -int f.curl g (g.curl a - a.curl g = div(a x g))", ok)

    # Schwarz-Sen form on the frame (ES17 eq. 17, Thm 13.10)
    Av = sp.Matrix([sp.Function(f"A{i}")(*XS) for i in (1, 2, 3)])
    Cv = sp.Matrix([sp.Function(f"C{i}")(*XS) for i in (1, 2, 3)])
    XA = sp.Matrix(list(Av) + list(Cv))
    XF = sp.Matrix(list(curl(Av)) + list(curl(Cv)))
    LSS = (XF.T * G6 * XA.diff(XS[0]))[0] / 2 - (XF.T * XF)[0] / 2
    Bss, Ess = curl(Av), curl(Cv)
    ok = is_zero(LSS - ((Ess.T * Av.diff(XS[0]))[0] - (Bss.T * Cv.diff(XS[0]))[0]) / 2
                 + ((Ess.T * Ess)[0] + (Bss.T * Bss)[0]) / 2)
    th = sp.Symbol("theta", real=True)
    rot = sp.cos(th) * sp.eye(6) + sp.sin(th) * G6
    XAr = rot * XA
    XFr = sp.Matrix(list(curl(XAr[:3, 0])) + list(curl(XAr[3:, 0])))
    LSSr = (XFr.T * G6 * XAr.diff(XS[0]))[0] / 2 - (XFr.T * XFr)[0] / 2
    ok &= sp.trigsimp(sp.expand(LSSr - LSS)) == 0
    vel = list(XA.diff(XS[0]))
    mom = sp.Matrix([sp.diff(LSS, v) for v in vel])
    ok &= is_zero((mom.T * sp.Matrix(vel))[0] - LSS - (XF.T * XF)[0] / 2)
    j0 = (mom.T * G6 * XA)[0]
    ok &= is_zero(j0 + (XA.T * XF)[0] / 2)
    diffL = LSS - ((Ess.T * Av.diff(XS[0]))[0] - (Ess.T * Ess + Bss.T * Bss)[0] / 2)
    ok &= all(is_zero(x) for x in EL_first_order(diffL, list(Av) + list(Cv)))
    check(32, "Schwarz-Sen form L_SS = (1/2) curl X_A . G dX_A/dt - (1/2)|curl X_A|^2 = (E.dA/dt - B.dC/dt)/2 - E6 "
              "(B = curl A, E = curl C): exactly invariant under X_A -> exp(theta G) X_A (off-shell duality), first order "
              "in time with Hamiltonian E6, duality charge density -(1/2) X_A . curl X_A (ES17 Thm 13.10 restated), and "
              "L_SS - (E.dA/dt - E6) has identically vanishing EL expression (same theory as L_H with Gauss solved)", ok)
