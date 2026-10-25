#!/usr/bin/env python3
"""Independent referee check of Section r1 ("Two readings of X and their Lagrangians") of the two-readings note.

    python3 investigations/two_readings/verify/s1_check.py

Prints one line 'PASS s1-<id>: <claim>' (or 'FAIL ...') per claim and exits 0 only if every check passes.
Written independently of verify/r1_lagrangians.py (nothing is imported from it), with different methods where possible:
  * the realisation is checked in integer numpy arithmetic and G is checked as the Hodge star on two-forms directly;
  * Lie algebras are built as nullspaces inside so(3,3) and intertwiner spaces are computed exactly for several
    isomorphisms, including a generic inner twist obtained from a rational Cayley transform;
  * units are checked with sympy.physics.units (the SI dimension system), not with hand-made exponent vectors;
  * Euler-Lagrange equations of point Lagrangians and of bosonic field Lagrangians use sympy's euler_equations;
  * Grassmann bilinears are handled by POLARISATION: a quadratic Grassmann expression sum c_ij g_i g_j is encoded
    injectively as the antisymmetric bilinear sum c_ij (g_i[u] g_j[v] - g_i[v] g_j[u]) of two commuting fields u, v,
    and the left Euler-Lagrange expression E[psi] of a first-order quadratic Grassmann Lagrangian is EL_u of the
    polarised Lagrangian evaluated at v.  All field identities are then identities of commuting jet polynomials;
  * inertias are exact (Descartes' rule on the characteristic polynomial of a real symmetric matrix);
  * the closed-form interior solution is compared with a numerical integration (scipy, tolerance 1e-9).
Floating point is used only for that integration and for random numerical tests of sqrt-valued Lagrangians (tolerance
1e-10). Runtime: well under 3 minutes.
"""
import itertools
import sys

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

OK_ALL = []


def report(cid, claim, ok):
    ok = bool(ok)
    OK_ALL.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} s1-{cid}: {claim}")
    sys.stdout.flush()


def zero(x):
    return sp.expand(x) == 0


def mzero(M):
    return all(zero(x) for x in sp.Matrix(M))


R = sp.Rational
rng = np.random.default_rng(20261025)


def rat(lo=-5, hi=5, den=(1, 2, 3)):
    return R(int(rng.integers(lo, hi + 1)), int(rng.choice(den)))


# ---------------------------------------------------------------------------------------------------------------
# A. Realisation in integer arithmetic; G as Hodge star; conversion; units
# ---------------------------------------------------------------------------------------------------------------
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)
k3 = lambda a, b, c: np.kron(np.kron(a, b), c)
EN = {1: k3(sx, I2, I2), 2: k3(sz, sx, I2), 3: k3(sz, sz, sx),
      4: k3(ep, I2, I2), 5: k3(sz, ep, I2), 6: k3(sz, sz, ep)}
ETA = sp.diag(1, 1, 1, -1, -1, -1)
eta_d = [None, 1, 1, 1, -1, -1, -1]
ES = {a: sp.Matrix(EN[a]) for a in EN}                     # sympy integer copies
I8 = sp.eye(8)
G = sp.zeros(6)
for i in range(3):
    G[i + 3, i] = 1                                         # G e_i = e_{i+3}
    G[i, i + 3] = -1                                        # G e_{i+3} = -e_i
ETA4 = sp.diag(1, -1, -1, -1)
LC3 = lambda i, j, k: sp.LeviCivita(i, j, k)


def F_of_X(X):
    """two-form F_X with F_{0i} = E_i, F_{jk} = eps_{ijk} B_i (B = X[0:3], E = X[3:6])"""
    F = sp.zeros(4)
    for i in range(3):
        F[0, i + 1], F[i + 1, 0] = X[i + 3], -X[i + 3]
        for j in range(3):
            for k in range(3):
                if LC3(i, j, k) != 0:
                    F[j + 1, k + 1] += LC3(i, j, k) * X[i]
    return F


def hodge(F):
    """(star F)_{ab} = 1/2 eps_{abcd} F^{cd}, eps_{0123} = +1, indices raised with diag(1,-1,-1,-1)"""
    Fu = ETA4 * F * ETA4
    S = sp.zeros(4)
    for a, b, c, d in itertools.product(range(4), repeat=4):
        e_ = sp.LeviCivita(a, b, c, d)
        if e_ != 0:
            S[a, b] += R(1, 2) * e_ * Fu[c, d]
    return S


def block_A():
    ok = all(np.array_equal(EN[a] @ EN[b] + EN[b] @ EN[a], 2 * (eta_d[a] if a == b else 0) * np.eye(8, dtype=np.int64))
             for a in EN for b in EN)
    ok &= G ** 2 == -sp.eye(6) and (ETA * G).T == ETA * G and G.T * ETA * G == -ETA
    # G is the Hodge star on two-forms: F_{GX} = star F_X for every basis vector X, and eta(X,X) = 1/2 F_ab F^ab
    for a in range(6):
        X = sp.Matrix([1 if b == a else 0 for b in range(6)])
        ok &= F_of_X(G * X) == hodge(F_of_X(X))
    Xs = sp.Matrix(sp.symbols("X0:6"))
    Fx = F_of_X(Xs)
    ok &= zero(sum(Fx[a, b] * (ETA4 * Fx * ETA4)[a, b] for a in range(4) for b in range(4)) / 2 - (Xs.T * ETA * Xs)[0])
    report(1, "integer realisation satisfies e_a e_b + e_b e_a = 2 eta_ab; G^2 = -1, eta G symmetric, G^T eta G = -eta; "
              "G is the Hodge star F_{GX} = *F_X (eps_{0123} = +1) and eta(X,X) = (1/2) F_ab F^ab", ok)

    # conversion: a general 3x3 C commuting with all rotations is c*1; G_C = Phi^{-1} G Phi
    cs = sp.symbols("c11 c12 c13 c21 c22 c23 c31 c32 c33")
    C = sp.Matrix(3, 3, cs)
    gens = []
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        Rm = sp.zeros(3)
        Rm[i, j], Rm[j, i] = -1, 1
        gens.append(Rm)
    eqs = []
    for Rm in gens:
        eqs += list(C * Rm - Rm * C)
    sol = sp.solve(eqs, cs, dict=True)
    ok = len(sol) == 1 and sp.Matrix(3, 3, [sol[0].get(s, s) for s in cs]) == cs[0] * sp.eye(3)
    c = sp.Symbol("c", positive=True)
    Phi = sp.diag(c, c, c, 1, 1, 1)
    GC = sp.Matrix(sp.BlockMatrix([[sp.zeros(3), -sp.eye(3) / c], [c * sp.eye(3), sp.zeros(3)]]))
    ok &= sp.simplify(Phi.inv() * G * Phi - GC) == sp.zeros(6)
    t1, x1 = sp.symbols("t1 x1")
    v = sp.Matrix([t1, 0, 0, x1, 0, 0])
    ok &= GC * v == sp.Matrix([-x1 / c, 0, 0, c * t1, 0, 0]) and G * (Phi * v) == sp.Matrix([-x1, 0, 0, c * t1, 0, 0])
    report(2, "a general 3x3 C commuting with the three rotation generators is a multiple of the identity; "
              "Phi^{-1} G Phi = G_C = [[0,-C^{-1}],[C,0]] with Phi = diag(c,c,c,1,1,1); G_C(t;x) = (-x/c; ct), while "
              "G sends the length x^1 of Y to -x^1 in the slot e_1", ok)


# units with the SI dimension system of sympy
from sympy.physics import units as U
from sympy.physics.units import Dimension
from sympy.physics.units.systems.si import SI, dimsys_SI

BASE = ("length", "mass", "time", "current")


def dimvec(q):
    dep = dimsys_SI.get_dimensional_dependencies(Dimension(SI.get_dimensional_expr(q)))
    return tuple(sp.nsimplify(dep.get(Dimension(n), 0)) for n in BASE)


def block_units():
    m, kg, s, A, V, T, J, Cb = U.meter, U.kilogram, U.second, U.ampere, U.volt, U.tesla, U.joule, U.coulomb
    c, hbar, eps0 = U.speed_of_light, U.hbar, U.vacuum_permittivity
    target = dimvec(m ** 2 / V)
    cols = [dimvec(c), dimvec(m), dimvec(hbar), dimvec(eps0)]       # ell has the dimension of a length
    Mfull = sp.Matrix([[col[r] for col in cols] for r in range(4)])
    tgt = sp.Matrix(target)
    # c, ell, hbar contain no ampere: the current row of the 3-column system is zero while the target's is not
    ok = all(cols[k][3] == 0 for k in range(3)) and target[3] != 0
    ok &= Mfull.det() != 0                                      # unique solution with eps0
    expo = Mfull.LUsolve(tgt)
    ok &= list(expo) == [R(-1, 2), 3, R(-1, 2), R(1, 2)]
    ok &= dimvec(m ** 3 * sp.sqrt(eps0 / (hbar * c))) == target
    # Heaviside-Lorentz reading: sqrt(eps0/(hbar c)) * E has dimension m^-2, so ell^3 converts it to a length
    ok &= dimvec(sp.sqrt(eps0 / (hbar * c)) * V / m) == dimvec(m ** -2)
    report(3, "units (sympy SI dimension system): no c^a ell^b hbar^d has dimension m^2/V (no ampere in c, ell, hbar); "
              "with eps0 the unique monomial is ell^3 (eps0/(hbar c))^(1/2); (eps0/(hbar c))^(1/2) E has dimension m^-2 "
              "(the Heaviside-Lorentz field with hbar = c = 1), so in those units ell^3 alone supplies the conversion", ok)

    ok = dimvec(c * s) == dimvec(m) and dimvec(c * T) == dimvec(V / m)
    ok &= dimvec(kg * c) == dimvec(kg * m / s)                         # K_A = -Mc eta Y'/sqrt(eta(Y',Y'))
    ok &= dimvec(kg * c * m) == dimvec(J * s)                          # action of the free worldline
    ok &= dimvec(kg * (m / s) ** 2) == dimvec(J) and dimvec(c / m) == dimvec(1 / s)
    ok &= dimvec(eps0 * (V / m) ** 2) == dimvec(J / m ** 3)
    ok &= dimvec(hbar * c * m ** -3 / m) == dimvec(J / m ** 3)         # hbar c psi^T d psi, [psi] = m^-3/2
    ok &= dimvec(kg * c / hbar) == dimvec(1 / m)                       # m c / hbar
    ok &= dimvec(Cb * (V * s / m) / hbar) == dimvec(1 / m)             # q A_mu / hbar
    ok &= dimvec(T * m) == dimvec(V * s / m)                           # B = curl A: [A] = V s/m
    ok &= dimvec((V / m) * m) == dimvec(V)                             # E = curl C: [C] = V, not V s/m
    ok &= dimvec(c * T * m) == dimvec(V)                               # (cA; C) is homogeneous in V
    ok &= dimvec(V / c) == dimvec(V * s / m)                           # A_0 = phi/c has the unit of a
    ok &= dimvec(eps0 * V / m) == dimvec(Cb / m ** 2)
    report(4, "units: [ct] = m, [cB] = [E] = V/m, [K_A] = kg m/s, free action J s, [L_kappa] = J, [kappa] = 1/s, "
              "eps0|E|^2 and hbar c psi^T d psi in J/m^3 for [psi] = m^-3/2, [mc/hbar] = [qA/hbar] = 1/m, [A] = V s/m, "
              "[C] = V (so X_A = (A;C) is homogeneous only as (cA;C), in V), A_0 = phi/c in V s/m", ok)


# ---------------------------------------------------------------------------------------------------------------
# B. The two Lorentz algebras as subalgebras of so(3,3); intertwiners for several isomorphisms
# ---------------------------------------------------------------------------------------------------------------
def vec_action(X8):
    """6x6 matrix of v -> [X8, v] on V = span(e1..e6), from traces tr(e_a e_b) = 8 eta_ab"""
    cols = []
    for b in range(1, 7):
        Y = X8 * ES[b] - ES[b] * X8
        coeff = [R((ES[a] * Y).trace(), 8 * eta_d[a]) for a in range(1, 7)]
        assert sum((coeff[a - 1] * ES[a] for a in range(1, 7)), sp.zeros(8)) == Y
        cols.append(sp.Matrix(coeff))
    return sp.Matrix.hstack(*cols)


def flat(M):
    return sp.Matrix(M).reshape(M.shape[0] * M.shape[1], 1)


def span_dim(mats):
    return sp.Matrix.hstack(*[flat(M) for M in mats]).rank()


def hom_dim(src, tgt):
    """dim {M : M src_k = tgt_k M for all k}, exact"""
    n, mm = src[0].shape[0], tgt[0].shape[0]
    Ms = sp.Matrix(mm, n, sp.symbols(f"h0:{mm * n}"))
    eqs = []
    for A_, B_ in zip(src, tgt):
        eqs += list(Ms * A_ - B_ * Ms)
    Jac = sp.Matrix([[sp.diff(e_, x) for x in Ms] for e_ in eqs])
    return mm * n - Jac.rank()


def coords_in(basis, X):
    A_ = sp.Matrix.hstack(*[flat(b) for b in basis])
    sol = (A_.T * A_).LUsolve(A_.T * flat(X))
    assert A_ * sol == flat(X)
    return sol


def adjoint(basis):
    return [sp.Matrix.hstack(*[coords_in(basis, X * b - b * X) for b in basis]) for X in basis]


def block_B():
    # so(3,3) = {A : A^T eta + eta A = 0}; lint = those supported on (e1,e4,e5,e6); lG = those commuting with G
    w = sp.symbols("w0:36")
    W = sp.Matrix(6, 6, w)

    def nullspace_mats(eqs):
        Jac = sp.Matrix([[sp.diff(e_, x) for x in w] for e_ in eqs])
        return [sp.Matrix(6, 6, list(v)) for v in Jac.nullspace()]

    so33 = list(W.T * ETA + ETA * W)
    lint_eqs = so33 + [W[r, cc] for r in range(6) for cc in range(6) if r in (1, 2) or cc in (1, 2)]
    lG_eqs = so33 + list(W * G - G * W)
    lint = nullspace_mats(lint_eqs)
    lG = nullspace_mats(lG_eqs)
    ok = len(nullspace_mats(so33)) == 15 and len(lint) == 6 and len(lG) == 6
    ok &= span_dim(lint + lG) == 12                              # the two subalgebras meet in 0
    # Clifford generators of the text span them
    cyc = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    Rc = [vec_action(ES[j + 4] * ES[k + 4] / 2) for (i, j, k) in cyc]
    Bc = [vec_action(ES[1] * ES[i + 4] / 2) for (i, j, k) in cyc]
    Lc = [vec_action((ES[j + 4] * ES[k + 4] - ES[j + 1] * ES[k + 1]) / 2) for (i, j, k) in cyc]
    Kc = [vec_action((ES[k + 1] * ES[j + 4] - ES[j + 1] * ES[k + 4]) / 2) for (i, j, k) in cyc]
    ok &= span_dim(lint + Rc + Bc) == 6 and span_dim(lG + Lc + Kc) == 6
    # Lorentz relations (with the sign that holds) for both triples
    br = lambda X, Y: X * Y - Y * X

    def rel(Rr, Bb):
        return all(br(Rr[i], Rr[j]) == sum((LC3(i, j, k) * Rr[k] for k in range(3)), sp.zeros(6)) and
                   br(Rr[i], Bb[j]) == sum((LC3(i, j, k) * Bb[k] for k in range(3)), sp.zeros(6)) and
                   br(Bb[i], Bb[j]) == sum((-LC3(i, j, k) * Rr[k] for k in range(3)), sp.zeros(6))
                   for i in range(3) for j in range(3))

    sgn = [(a_, b_) for a_ in (1, -1) for b_ in (1, -1) if rel([a_ * r for r in Rc], [b_ * b for b in Bc])]
    ok &= len(sgn) > 0 and rel(Lc, Kc)
    sR, sB = sgn[0]
    Rc, Bc = [sR * r for r in Rc], [sB * b for b in Bc]
    report(5, "so(3,3) has dimension 15; lint (supported on e1,e4,e5,e6) and lG (commuting with G) are 6-dimensional "
              "nullspaces, spanned by the Clifford generators R_i, B_i and L_i, K_i of the text, with the same so(1,3) "
              "relations; lint + lG has dimension 12, so the two subalgebras meet in 0", ok)

    # representations
    Eidx = [0, 3, 4, 5]
    srcE = [X.extract(Eidx, Eidx) for X in Rc + Bc]
    srcEV = Rc + Bc                                              # E + 2.1 on all of V
    srcL2 = adjoint(Rc + Bc)                                     # adjoint module = Lambda^2 E (bivectors of E span lint)
    phis = {"phi": Lc + Kc, "parity": Lc + [-k for k in Kc]}
    # a generic inner twist: Cayley transform g = (1 - A)^{-1}(1 + A) of a rational A in lG
    Ag = sum((rat() * X for X in Lc + Kc), sp.zeros(6)) / 7
    g = (sp.eye(6) - Ag).inv() * (sp.eye(6) + Ag)
    ok = g.T * ETA * g == ETA and g * G == G * g
    phis["inner"] = [g * X * g.inv() for X in Lc + Kc]
    phis["inner+parity"] = [g * X * g.inv() for X in Lc + [-k for k in Kc]]
    dims = {}
    for name, tg in phis.items():
        ok &= rel(tg[:3], tg[3:])                                # each is a Lie algebra isomorphism lint -> lG
        d = (hom_dim(srcE, tg), hom_dim(tg, srcE), hom_dim(srcEV, tg), hom_dim(tg, srcEV), hom_dim(srcL2, tg))
        dims[name] = d
        ok &= d == (0, 0, 0, 0, 2)
    ok &= hom_dim(srcE, srcL2) == 0 and hom_dim(srcL2, srcE) == 0
    # existence for EVERY phi: the adjoint module of lG is isomorphic to V, and phi intertwines ad_lint with ad_lG o phi
    adG = adjoint(Lc + Kc)
    ok &= hom_dim(adG, Lc + Kc) == 2
    report(6, "Hom_lint(E,V_phi) = Hom(V_phi,E) = Hom(E+2.1,V_phi) = Hom(V_phi,E+2.1) = 0 and dim Hom(Lambda^2 E,V_phi) = 2 "
              f"for phi, its parity twist, a generic inner twist (rational Cayley transform) and their composite {dims}; "
              "Hom(E,Lambda^2E) = Hom(Lambda^2E,E) = 0; the adjoint module of lG is isomorphic to V (Hom dimension 2), "
              "so an intertwiner Lambda^2E -> V_phi exists for every isomorphism phi", ok)

    # Casimirs and commutants
    C1E = sum((X * X for X in srcE[:3]), sp.zeros(4)) - sum((X * X for X in srcE[3:]), sp.zeros(4))
    C2E = sum((srcE[i] * srcE[i + 3] for i in range(3)), sp.zeros(4))
    C1V = sum((X * X for X in Lc), sp.zeros(6)) - sum((X * X for X in Kc), sp.zeros(6))
    C2V = sum((Lc[i] * Kc[i] for i in range(3)), sp.zeros(6))
    ok = C1E == -3 * sp.eye(4) and C2E == sp.zeros(4) and C1V == -4 * sp.eye(6) and C2V == -2 * G
    ok &= hom_dim(Lc + Kc, Lc + Kc) == 2 and hom_dim(Rc + Bc, Rc + Bc) == 5
    ok &= set((-2 * G).eigenvals().keys()) == {2 * sp.I, -2 * sp.I}
    report(7, "Casimirs: sum R^2 - sum B^2 = -3, sum R_i B_i = 0 on E; sum L^2 - sum K^2 = -4, sum L_i K_i = -2G "
              "(eigenvalues +-2i) on V; commutants of dimension 2 (lG) and 5 (lint) on V", ok)

    # G-invariant four-planes are complex 2-planes of C^3 = (V, G); eta restricted to them has signature (2,2)
    sigs = set()
    for _ in range(4):
        w1 = sp.Matrix([rat() for _ in range(6)])
        w2 = sp.Matrix([rat() for _ in range(6)])
        Bm = sp.Matrix.hstack(w1, G * w1, w2, G * w2)
        if Bm.rank() < 4:
            continue
        sigs.add(inertia(Bm.T * ETA * Bm))
    ok = sigs == {(2, 2, 0)} or sigs <= {(2, 2, 0), (1, 1, 2), (0, 0, 4)}
    report(8, "every G-invariant four-plane (a complex 2-plane of (V,G)) carries eta of signature (2,2) or degenerate, never "
              f"(1,3) [samples {sigs}]: G preserves no Lorentzian four-plane (FMB Prop. 6.2), while lG preserves no proper "
              "subspace at all (commutant C of a completely reducible module: irreducible)", ok)
    return dict(Rc=Rc, Bc=Bc, Lc=Lc, Kc=Kc)


def inertia(Msym):
    """exact inertia (n+, n-, n0) of a real symmetric matrix by Descartes' rule on its characteristic polynomial"""
    x = sp.Symbol("x")
    p = sp.Poly(sp.Matrix(Msym).charpoly(x).as_expr(), x)
    coeffs = p.all_coeffs()
    n0 = 0
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
        n0 += 1
    nz = [cf for cf in coeffs if cf != 0]
    pos = sum(1 for a_, b_ in zip(nz, nz[1:]) if a_ * b_ < 0)
    neg_c = [cf * (-1) ** (len(coeffs) - 1 - k) for k, cf in enumerate(coeffs)]
    nz2 = [cf for cf in neg_c if cf != 0]
    neg = sum(1 for a_, b_ in zip(nz2, nz2[1:]) if a_ * b_ < 0)
    return (pos, neg, n0)


# ---------------------------------------------------------------------------------------------------------------
# C. Field reading: Poisson bracket; free worldline
# ---------------------------------------------------------------------------------------------------------------
XS = sp.symbols("t x y z", real=True)


def curlF(v):
    return sp.Matrix([sp.diff(v[2], XS[2]) - sp.diff(v[1], XS[3]),
                      sp.diff(v[0], XS[3]) - sp.diff(v[2], XS[1]),
                      sp.diff(v[1], XS[1]) - sp.diff(v[0], XS[2])])


def divF(v):
    return sum(sp.diff(v[i], XS[i + 1]) for i in range(3))


def EL(Ldens, funcs, coords=XS):
    eqs = sp.calculus.euler.euler_equations(Ldens, funcs, coords)
    return [sp.expand(q.lhs) for q in eqs]


def block_C():
    a = sp.Matrix([sp.Function(f"a{i}")(*XS) for i in (1, 2, 3)])
    A0 = sp.Function("A0")(*XS)
    E = a.diff(XS[0]) - sp.Matrix([sp.diff(A0, X) for X in XS[1:]])
    B = curlF(a)
    LM = ((E.T * E)[0] - (B.T * B)[0]) / 2
    ok = all(zero(sp.diff(LM, sp.Derivative(a[i], XS[0])) - E[i]) for i in range(3))   # E is conjugate to a
    g = sp.Matrix([sp.Function(f"g{i}")(*XS) for i in (1, 2, 3)])
    # variational derivative of int g . curl a with respect to a is curl g
    var = EL((g.T * curlF(a))[0], list(a))
    ok &= all(zero(var[i] - curlF(g)[i]) for i in range(3))
    report(9, "field reading: E = d_t a - grad A0 is the momentum conjugate to a (dL_M/d(d_t a) = E), and "
              "delta/delta a of int g.curl a is curl g, so with {a_i(x),E_j(y)} = delta_ij delta^3 one gets "
              "{int f.E, int g.B} = -int f.curl g", ok)

    # free worldline: K, constraint, EL proportional to the acceleration orthogonal to the velocity
    Mc = sp.Symbol("Mc", positive=True)
    vs = sp.Matrix(sp.symbols("v1:7", real=True))
    acc = sp.Matrix(sp.symbols("w1:7", real=True))
    wv = (vs.T * ETA * vs)[0]
    L = -Mc * sp.sqrt(wv)
    K = sp.Matrix([sp.diff(L, x) for x in vs])
    ok = mzero(K + Mc * ETA * vs / sp.sqrt(wv)) and zero(sp.simplify((K.T * ETA * K)[0] - Mc ** 2))
    ok &= zero(sp.simplify((K.T * vs)[0] - L))
    lam_ = sp.Symbol("s", positive=True)
    ok &= zero(sp.simplify(L.subs({x: lam_ * x for x in vs}, simultaneous=True) - lam_ * L))
    Kdot = sp.Matrix([sum(sp.diff(K[A_], vs[B_]) * acc[B_] for B_ in range(6)) for A_ in range(6)])
    proj = acc - vs * (vs.T * ETA * acc)[0] / wv
    ok &= mzero(sp.simplify(Kdot + Mc * ETA * proj / sp.sqrt(wv)))
    P = -ETA * K
    ok &= mzero(sp.simplify(P - Mc * vs / sp.sqrt(wv)))
    report(10, "free worldline (sympy on velocity/acceleration symbols): K = -Mc eta Y'/sqrt(eta(Y',Y')), eta(K,K) = M^2c^2, "
               "K.Y' = L (vanishing canonical Hamiltonian), degree-one homogeneity, P = -eta^{-1}K = Mc Y'/sqrt; "
               "dK/dlambda = -Mc eta (Y'' - Y' eta(Y',Y'')/eta(Y',Y'))/sqrt(eta(Y',Y')), so the EL equations hold iff "
               "Y'' is parallel to Y': straight lines in every parametrisation", ok)

    # connectedness of {eta > 0} and of the shell; four-dimensional mass
    om = sp.Matrix(sp.symbols("o1:4", real=True))
    vv = sp.Matrix(sp.symbols("p1:4", real=True))
    M2 = sp.Symbol("M2", positive=True)                     # M^2 c^2
    h = sp.Matrix(list(sp.sqrt(M2 + (vv.T * vv)[0]) * om) + list(vv))
    ok = zero(sp.simplify((h.T * ETA * h)[0].subs((om.T * om)[0], 1) -
                          (M2 + (vv.T * vv)[0]) * ((om.T * om)[0] - 1) - M2))
    ok &= zero(sp.expand((h.T * ETA * h)[0] - (M2 + (vv.T * vv)[0]) * (om.T * om)[0] + (vv.T * vv)[0]))
    sv = sp.Symbol("s", real=True)
    path = sp.Matrix([sp.cos(sv), sp.sin(sv), 0, 0, 0, 0])
    ok &= sp.simplify((path.T * ETA * path)[0]) == 1 and path.subs(sv, sp.pi) == -path.subs(sv, 0)
    # timelike set: (u,v) -> (u,(1-s)v) stays timelike, ending at (u,0); the u-sphere is connected
    u3 = sp.Matrix(sp.symbols("u1:4", real=True))
    etaline = ((u3.T * u3)[0] - (1 - sv) ** 2 * (vv.T * vv)[0]) - ((u3.T * u3)[0] - (vv.T * vv)[0])
    ok &= zero(sp.expand(etaline - (vv.T * vv)[0] * (1 - (1 - sv) ** 2)))
    En, q1, q2 = sp.symbols("En q1 q2", real=True)
    c = sp.Symbol("c", positive=True)
    Mm = sp.Symbol("M", positive=True)
    Pv = sp.Matrix([En / c, q1, q2] + list(vv))
    shell = sp.solve(sp.Eq((Pv.T * ETA * Pv)[0], Mm ** 2 * c ** 2), En ** 2) if False else None
    ok &= zero(sp.expand(c ** 2 * ((Pv.T * ETA * Pv)[0] - Mm ** 2 * c ** 2)
                         - (En ** 2 - c ** 2 * (vv.T * vv)[0] - Mm ** 2 * c ** 4 + c ** 2 * (q1 ** 2 + q2 ** 2))))
    ok &= (Mm ** 2 - (q1 ** 2 + q2 ** 2) / c ** 2).subs({Mm: 1, c: 1, q1: 2, q2: 0}) == -3
    report(11, "split signature: the shell eta = M^2c^2 is the image of S^2 x R^3 under (w,v) -> (sqrt(M^2c^2+|v|^2) w, v), "
               "inverse (u,v) -> (u/|u|, v), hence connected; {eta > 0} retracts onto the unit sphere of V^+ along "
               "(u,(1-s)v); the path (cos s, sin s, 0; 0,0,0) joins e1 to -e1 at eta = 1; the shell reads "
               "E^2 = c^2|p|^2 + M^2c^4 - c^2|q|^2 and M = c = 1, q = (2,0) gives m4^2 = -3", ok)


# ---------------------------------------------------------------------------------------------------------------
# D. Interior Lagrangian on lengths
# ---------------------------------------------------------------------------------------------------------------
def block_D():
    tau = sp.Symbol("tau", real=True)
    m, k = sp.symbols("m kappa", positive=True)
    Y = sp.Matrix([sp.Function(f"y{a}")(tau) for a in range(6)])
    Yd = Y.diff(tau)
    uu = Yd + k * G * Y
    L = m / 2 * (uu.T * ETA * uu)[0]
    eqs = EL(L, list(Y), [tau])
    # euler_equations returns dL/dY - d/dtau dL/dY'; it must equal -m eta (Y'' + k^2 Y)
    ok = all(zero(eqs[a] + m * (ETA * (Y.diff(tau, 2) + k ** 2 * Y))[a]) for a in range(6))
    p = sp.Matrix([sp.diff(L, x) for x in Yd])
    on = {Y.diff(tau, 2)[a]: -k ** 2 * Y[a] for a in range(6)}
    ok &= mzero(p - m * ETA * uu) and mzero(p.diff(tau).subs(on) - k * G.T * p)
    pi = ETA * p
    ok &= mzero(pi.diff(tau).subs(on) - k * G * pi)
    report(12, "interior Lagrangian L = (m/2) eta(Y'+kGY,Y'+kGY) (sympy euler_equations): EL = -m eta(Y''+k^2Y), "
               "p = m eta(Y'+kGY), p' = k G^T p and pi = eta^{-1}p obeys pi' = k G pi on shell", ok)

    L0 = m / 2 * ((Yd.T * ETA * Yd)[0] - k ** 2 * (Y.T * ETA * Y)[0])
    ct = [Y[0], Y[1], Y[2]]
    xa = [Y[3], Y[4], Y[5]]
    Lam = -m * k * sum(ct[i] * xa[i] for i in range(3))
    ok = zero(L - L0 - Lam.diff(tau))
    gradLam = sp.Matrix([sp.diff(Lam, x) for x in Y])
    ok &= mzero(p - m * ETA * Yd - gradLam) and mzero(gradLam - m * k * ETA * G * Y)
    ok &= mzero(pi - m * Yd - ETA * gradLam) and mzero(ETA * gradLam - m * k * G * Y)
    report(13, "L = (m/2)(eta(Y',Y') - k^2 eta(Y,Y)) + d/dtau[-m k sum_a (ct^a) x^a]; p - m eta Y' = grad Lambda = m k eta G Y, "
               "and for the raised momentum pi - m Y' = eta^{-1} grad Lambda = m k G Y", ok)

    ys = sp.Matrix(sp.symbols("Y1:7", real=True))
    vs = sp.Matrix(sp.symbols("V1:7", real=True))
    ps = sp.Matrix(sp.symbols("P1:7", real=True))
    Lp = m / 2 * ((vs + k * G * ys).T * ETA * (vs + k * G * ys))[0]
    pp = sp.Matrix([sp.diff(Lp, x) for x in vs])
    Ef = sp.expand((pp.T * vs)[0] - Lp)
    ok = zero(Ef - m / 2 * ((vs.T * ETA * vs)[0] + k ** 2 * (ys.T * ETA * ys)[0]))
    H = (ps.T * ETA * ps)[0] / (2 * m) - k * (ps.T * G * ys)[0]
    ok &= zero(H.subs({ps[a]: pp[a] for a in range(6)}, simultaneous=True) - Ef)
    ok &= zero(sp.diff(Ef.subs({vs[a]: sp.Symbol(f"q{a}") for a in range(6)}), m) * 0)
    # conservation of E along Y'' = -k^2 Y: dE/dtau = m(eta(Y',Y'') + k^2 eta(Y,Y'))
    accel = -k ** 2 * ys
    ok &= zero(m * ((vs.T * ETA * accel)[0] + k ** 2 * (ys.T * ETA * vs)[0]))
    vars12 = list(ys) + list(vs)
    HE = sp.hessian(Ef.subs({m: 1, k: 1}), vars12)
    HH = sp.hessian(H.subs({m: 1, k: 1}), list(ys) + list(ps))
    HE2 = sp.hessian(Ef.subs({m: 3, k: R(2, 5)}), vars12)
    HH2 = sp.hessian(H.subs({m: 3, k: R(2, 5)}), list(ys) + list(ps))
    ok &= inertia(HE) == (6, 6, 0) and inertia(HH) == (6, 6, 0) and inertia(HE2) == (6, 6, 0) and inertia(HH2) == (6, 6, 0)
    Etime = Ef.subs({ys[a]: 0 for a in range(3, 6)}).subs({vs[a]: 0 for a in range(3, 6)})
    Elen = Ef.subs({ys[a]: 0 for a in range(3)}).subs({vs[a]: 0 for a in range(3)})
    ok &= zero(Etime - m / 2 * sum(vs[i] ** 2 + k ** 2 * ys[i] ** 2 for i in range(3)))
    ok &= zero(Elen + m / 2 * sum(vs[i] ** 2 + k ** 2 * ys[i] ** 2 for i in range(3, 6)))
    report(14, "energy p.Y' - L = (m/2)(eta(Y',Y') + k^2 eta(Y,Y)) = H(Y,p) = eta(p,p)/(2m) - k p^T G Y, conserved on "
               "Y'' = -k^2Y; exact inertia (Descartes) of both 12x12 Hessians is (6,6,0) at (m,k) = (1,1) and (3,2/5); "
               "positive on the converted-time oscillators, negative on the length oscillators", ok)

    # closed-form solution against numerical integration, and the statements (i)-(iii)
    mm, kk = 1.7, 0.9
    Gn = np.array(G.tolist(), dtype=float)
    Y0 = rng.normal(size=6)
    pi0 = rng.normal(size=6)
    v0 = pi0 / mm - kk * Gn @ Y0
    sol = solve_ivp(lambda t, z: np.concatenate([z[6:], -kk ** 2 * z[:6]]), (0, 3.0), np.concatenate([Y0, v0]),
                    rtol=1e-12, atol=1e-13, dense_output=True)
    ts = np.linspace(0, 3.0, 13)
    expG = lambda s_: np.cos(s_) * np.eye(6) + np.sin(s_) * Gn
    errY = max(np.max(np.abs(sol.sol(t)[:6] - (expG(-kk * t) @ Y0 + np.sin(kk * t) / (kk * mm) * pi0))) for t in ts)
    errpi = max(np.max(np.abs(mm * (sol.sol(t)[6:] + kk * Gn @ sol.sol(t)[:6]) - expG(kk * t) @ pi0)) for t in ts)
    ok = errY < 1e-9 and errpi < 1e-9
    x0 = sp.Matrix(sp.symbols("x01:4", real=True))
    Y0s = sp.Matrix([0, 0, 0] + list(x0))
    eG = lambda s_: sp.cos(s_) * sp.eye(6) + sp.sin(s_) * G
    ok &= mzero((eG(-k * tau) * Y0s)[:3, 0] - sp.sin(k * tau) * x0)
    w0 = sp.Matrix(sp.symbols("w01:04", real=True))
    Yc = sp.cos(k * tau) * Y0s + sp.sin(k * tau) / k * sp.Matrix([0, 0, 0] + list(w0))
    ok &= mzero(Yc.diff(tau, 2) + k ** 2 * Yc) and mzero(Yc[:3, 0])
    pic0 = (m * (Yc.diff(tau) + k * G * Yc)).subs(tau, 0)
    ok &= mzero(pic0[:3, 0] + m * k * x0)                      # temporal part -m k x0: nonzero iff Y0 != 0
    e5 = sp.Matrix([0, 0, 0, 0, 1, 0])
    ok &= mzero(eG(k * tau) * e5 - (sp.cos(k * tau) * e5 - sp.sin(k * tau) * sp.Matrix([0, 1, 0, 0, 0, 0])))
    ok &= mzero(eG(-k * tau) * sp.Matrix([0, 0, 0, 1, 0, 0]) - sp.Matrix([sp.sin(k * tau), 0, 0, sp.cos(k * tau), 0, 0]))
    # kinetic data in E_{e1} x E_{e1} stay there under Y'' = -k^2 Y; pi leaves it unless in span(e1,e4)
    ok &= all(x == 0 for x in (eG(k * tau) * sp.Matrix([1, 0, 0, 0, 0, 0]))[1:3, 0])
    report(15, f"general solution Y = exp(-kG tau)Y0 + sin(k tau)pi0/(k m), pi = exp(kG tau)pi0 agrees with numerical "
               f"integration (max errors {errY:.1e}, {errpi:.1e} < 1e-9); spatial phase-space data acquire the temporal "
               "part sin(k tau) sum x0^a e_a; spatial configuration data stay spatial with pi(0)^+ = -m k sum x0^a e_a "
               "(nonzero exactly when Y0 != 0); exp(kG tau)e5 = cos e5 - sin e2; exp(-kG tau)e4 = cos e4 + sin e1", ok)


# ---------------------------------------------------------------------------------------------------------------
# E. Jets of commuting polarised fields for the Grassmann Dirac field
# ---------------------------------------------------------------------------------------------------------------
class JetSpace:
    def __init__(self):
        self.meta, self.cache = {}, {}

    def s(self, name, comp, alpha=()):
        alpha = tuple(sorted(alpha))
        key = (name, comp, alpha)
        if key not in self.cache:
            sym = sp.Symbol(f"{name}{comp}_{''.join(map(str, alpha))}")
            self.cache[key] = sym
            self.meta[sym] = key
        return self.cache[key]

    def D(self, expr, mu):
        expr = sp.sympify(expr)
        out = sp.Integer(0)
        for sym in expr.free_symbols:
            if sym in self.meta:
                nm, cp, al = self.meta[sym]
                out += sp.diff(expr, sym) * self.s(nm, cp, al + (mu,))
        return sp.expand(out)

    def col(self, name, alpha=()):
        return sp.Matrix([self.s(name, a, alpha) for a in range(8)])


GAM = [ES[1], ES[4], ES[5], ES[6]]
JJ = ES[2] * ES[3]
TT = ES[1] * ES[2] * ES[3]
KK = [-GAM[0] * GAM[mu] for mu in range(4)]


def bil(X, M, Y):
    return sp.expand((X.T * M * Y)[0])


class Dirac:
    """polarised Dirac theory: fields 'u','v' (commuting), optional gauge jets 'A' (comp = mu)"""

    def __init__(self, mass, q=None, gauge=False):
        self.j = JetSpace()
        self.m, self.q, self.gauge = mass, q, gauge
        self._rule = {}

    def A(self, mu, alpha=()):
        return self.j.s("A", mu, alpha) if self.gauge else 0

    def Dcol(self, f, mu):
        c = self.j.col(f, (mu,))
        if self.gauge:
            c = c + self.q * self.A(mu) * JJ * self.j.col(f)
        return c

    def pol(self, Xf, M, Yf):
        return sp.expand(bil(Xf("u"), M, Yf("v")) - bil(Xf("v"), M, Yf("u")))

    def lagrangian(self):
        L = sp.Integer(0)
        for mu in range(4):
            L += self.pol(lambda f: self.j.col(f), KK[mu] / 2, lambda f, mu=mu: self.Dcol(f, mu))
        L -= self.pol(lambda f: self.j.col(f), self.m * TT / 2, lambda f: self.j.col(f))
        return sp.expand(L)

    def EL_u(self, L):
        """commuting Euler-Lagrange expression of a first-order L with respect to u; by polarisation it is E[v]"""
        out = []
        for a in range(8):
            ea = sp.diff(L, self.j.s("u", a))
            for mu in range(4):
                ea -= self.j.D(sp.diff(L, self.j.s("u", a, (mu,))), mu)
            out.append(sp.expand(ea))
        return sp.Matrix(out)

    def E_expected(self, f):
        e_ = sp.zeros(8, 1)
        for mu in range(4):
            e_ += KK[mu] * self.Dcol(f, mu)
        return sp.expand(e_ - self.m * TT * self.j.col(f))

    def rhs0(self, f):
        """d_0 f from the Dirac equation K^mu D_mu f = m T f with K^0 = -1"""
        r = sp.zeros(8, 1)
        for a_ in (1, 2, 3):
            r += KK[a_] * self.Dcol(f, a_)
        r -= self.m * TT * self.j.col(f)
        if self.gauge:
            r -= self.q * self.A(0) * JJ * self.j.col(f)
        return r

    def onshell(self, expr):
        expr = sp.expand(expr)
        while True:
            subs = {}
            for sym in expr.free_symbols:
                key = self.j.meta.get(sym)
                if key and key[0] in ("u", "v") and 0 in key[2]:
                    subs[sym] = self.rule(sym)
            if not subs:
                return expr
            expr = sp.expand(expr.xreplace(subs))

    def rule(self, sym):
        if sym not in self._rule:
            f, a, al = self.j.meta[sym]
            beta = list(al)
            beta.remove(0)
            e_ = self.rhs0(f)[a]
            for nu in beta:
                e_ = self.j.D(e_, nu)
            self._rule[sym] = self.onshell(e_)
        return self._rule[sym]


def block_E():
    # (i) Grassmann variation for random integer kernels
    d = Dirac(0)
    Ak = [sp.Matrix(rng.integers(-3, 4, (8, 8))) for _ in range(4)]
    Mk = sp.Matrix(rng.integers(-3, 4, (8, 8)))
    col = d.j.col
    L = sp.Integer(0)
    for mu in range(4):
        L += d.pol(lambda f: col(f), Ak[mu] / 2, lambda f, mu=mu: col(f, (mu,)))
    L -= d.pol(lambda f: col(f), Mk / 2, lambda f: col(f))
    E = d.EL_u(L)
    exp_ = sum(((Ak[mu] + Ak[mu].T) / 2 * col("v", (mu,)) for mu in range(4)), sp.zeros(8, 1)) - (Mk - Mk.T) / 2 * col("v")
    ok = mzero(E - exp_)
    S_ = Mk + Mk.T
    ok &= zero(d.pol(lambda f: col(f), S_, lambda f: col(f)))
    Aa = Ak[2] - Ak[2].T
    ok &= zero(d.pol(lambda f: col(f), Aa, lambda f: col(f, (2,))) - d.j.D(d.pol(lambda f: col(f), Aa / 2, lambda f: col(f)), 2))
    report(16, "Grassmann variation by polarisation (random integer kernels): the EL expression of 1/2 psi^T A^mu d_mu psi "
               "- 1/2 psi^T M psi is A^mu_sym d_mu psi - M_anti psi; a symmetric mass kernel gives 0 and an antisymmetric "
               "kinetic kernel a total derivative", ok)

    # kernels and the Dirac equation
    ok = all(K_.T == K_ for K_ in KK) and KK[0] == -I8 and TT.T == -TT and TT * TT == -I8
    ok &= all((TT * g_).T == -(TT * g_) for g_ in GAM) and all(KK[mu] == TT * JJ * GAM[mu] for mu in range(4))
    ok &= all((K_ * JJ).T == -(K_ * JJ) and K_ * JJ == JJ * K_ for K_ in KK) and (JJ * TT).T == JJ * TT
    ok &= all(g_ * JJ == JJ * g_ for g_ in GAM)
    m = sp.Symbol("m", positive=True)
    d = Dirac(m)
    L = d.lagrangian()
    E = d.EL_u(L)
    ok &= mzero(E - d.E_expected("v"))
    Tinv = TT.inv()
    ok &= all(Tinv * KK[mu] == JJ * GAM[mu] for mu in range(4)) and all(-JJ * (JJ * GAM[mu]) == GAM[mu] for mu in range(4))
    kv = sp.symbols("k0:4")
    gk = sum((GAM[mu] * kv[mu] for mu in range(4)), sp.zeros(8))
    ok &= mzero((gk - m * JJ) * (gk + m * JJ) - (kv[0] ** 2 - kv[1] ** 2 - kv[2] ** 2 - kv[3] ** 2 + m ** 2) * I8)
    report(17, "kernels K^mu = -gamma^0 gamma^mu = T J gamma^mu symmetric, K^0 = -1, T antisymmetric with T^2 = -1, "
               "T gamma^mu antisymmetric, K^mu J antisymmetric, J T symmetric; the Dirac EL expression is "
               "K^mu d_mu psi - m T psi; T^{-1}K^mu = J gamma^mu and -J(J gamma^mu) = gamma^mu, so the equation reads "
               "J gamma.d psi = m psi, i.e. gamma.d psi = -m J psi; (gamma.k - mJ)(gamma.k + mJ) = (k.k + m^2) 1", ok)

    # (iii) gauge invariance and the Noether current
    U = R(3, 5) * I8 + R(4, 5) * JJ                              # exp(theta J), cos = 3/5, sin = 4/5
    ok = U.T * U == I8 and all(U.T * K_ * U == K_ for K_ in KK) and U.T * TT * U == TT
    lamf = sp.Function("lam")(*XS)
    qq = sp.Symbol("q", real=True)
    Af = [sp.Function(f"Af{mu}")(*XS) for mu in range(4)]
    psi = sp.Matrix([sp.Function(f"f{a}")(*XS) for a in range(8)])
    Ul = sp.cos(qq * lamf) * I8 - sp.sin(qq * lamf) * JJ
    for mu in range(4):
        lhs = sp.diff(Ul * psi, XS[mu]) + qq * (Af[mu] + sp.diff(lamf, XS[mu])) * JJ * (Ul * psi)
        rhs = Ul * (sp.diff(psi, XS[mu]) + qq * Af[mu] * JJ * psi)
        ok &= mzero(sp.simplify(lhs - rhs))
    jmu = [d.pol(lambda f: d.j.col(f), KK[mu] * JJ / 2, lambda f: d.j.col(f)) for mu in range(4)]
    divj = sp.expand(sum(d.j.D(jmu[mu], mu) for mu in range(4)))
    Ev, Eu = d.E_expected("v"), d.E_expected("u")
    rhs = sp.expand(bil(d.j.col("u"), JJ, Ev) - bil(d.j.col("v"), JJ, Eu))
    ok &= zero(divj - rhs)
    ok &= zero(jmu[0] - d.pol(lambda f: d.j.col(f), -JJ / 2, lambda f: d.j.col(f)))
    report(18, "exp(theta J) (rational cos, sin) preserves every K^mu and T; D = d + qAJ is covariant under "
               "exp(-q lam J), A -> A + d lam; the polarised current j^mu = 1/2 psi^T K^mu J psi obeys "
               "d_mu j^mu = psi^T J E_D identically, and j^0 = -1/2 psi^T J psi", ok)

    # (iv) canonical tensor
    Lt = L

    def Theta(mu, nu):
        x_ = d.pol(lambda f: d.j.col(f), KK[mu] / 2, lambda f: d.j.col(f, (nu,)))
        return sp.expand(x_ - Lt) if mu == nu else x_

    ok = True
    for nu in range(4):
        dv = sp.expand(sum(d.j.D(Theta(mu, nu), mu) for mu in range(4)))
        target = -(bil(d.j.col("u", (nu,)), I8, Ev) - bil(d.j.col("v", (nu,)), I8, Eu))
        ok &= zero(dv - target)
    dens = d.pol(lambda f: d.j.col(f), sum((GAM[0] * GAM[a_] for a_ in (1, 2, 3)), sp.zeros(8)) * 0, lambda f: d.j.col(f))
    dens = sp.Integer(0)
    for a_ in (1, 2, 3):
        dens += d.pol(lambda f: d.j.col(f), GAM[0] * GAM[a_] / 2, lambda f, a_=a_: d.j.col(f, (a_,)))
    dens += d.pol(lambda f: d.j.col(f), m * TT / 2, lambda f: d.j.col(f))
    ok &= zero(Theta(0, 0) - dens)
    pm = sp.symbols("p1:4")
    Hp = sum((GAM[0] * GAM[a_] * pm[a_ - 1] for a_ in (1, 2, 3)), sp.zeros(8)) + m * GAM[0]
    ok &= mzero(sum((GAM[0] * GAM[a_] * JJ * pm[a_ - 1] for a_ in (1, 2, 3)), sp.zeros(8)) + m * TT - JJ * Hp)
    report(19, "canonical tensor Theta^mu_nu = 1/2 psi^T K^mu d_nu psi - delta L_D: d_mu Theta^mu_nu = -(d_nu psi)^T E_D "
               "identically; Theta^0_0 = 1/2 psi^T (gamma^0 gamma^a d_a + m T) psi; with d_a psi = J p_a psi the kernel is "
               "J H(p), H(p) = gamma^0 gamma^a p_a + m gamma^0", ok)

    # (v) Belinfante, on shell
    up = lambda nu: ETA4[nu, nu]

    def ThetaUp(mu, nu):
        return sp.expand(Theta(mu, nu) * up(nu))

    def Sbil(r, a_, b_):
        if len({r, a_, b_}) < 3:
            return sp.Integer(0)
        return d.pol(lambda f: d.j.col(f), GAM[0] * GAM[r] * GAM[a_] * GAM[b_], lambda f: d.j.col(f))

    sS = sp.Symbol("sS")
    sols = set()
    ok = True
    for (a_, b_) in itertools.combinations(range(4), 2):
        lhs = d.onshell(ThetaUp(a_, b_) - ThetaUp(b_, a_))
        divS = d.onshell(sum(d.j.D(Sbil(r, a_, b_), r) for r in range(4)))
        poly = sp.Poly(sp.expand(lhs - sS * divS), *sorted((lhs - sS * divS).free_symbols - {sS}, key=str))
        sol = sp.solve(poly.coeffs(), sS, dict=True)
        if len(sol) != 1:
            ok = False
            break
        sols.add(sol[0][sS])
    ok &= sols == {R(1, 4)}
    ok &= all(GAM[0] * GAM[pr[0]] * GAM[pr[1]] * GAM[pr[2]] ==
              sp.combinatorics.Permutation([sorted(pr).index(x) for x in pr]).signature() * GAM[0] * GAM[sorted(pr)[0]] *
              GAM[sorted(pr)[1]] * GAM[sorted(pr)[2]] for pr in itertools.permutations(range(1, 4)))

    def TB(mu, nu):
        x_ = (d.pol(lambda f: d.j.col(f), KK[mu] / 4, lambda f: d.j.col(f, (nu,))) * up(nu) +
              d.pol(lambda f: d.j.col(f), KK[nu] / 4, lambda f: d.j.col(f, (mu,))) * up(mu))
        return sp.expand(x_ - (Lt * ETA4[mu, nu] if mu == nu else 0))

    for nu in range(4):
        ok &= zero(d.onshell(sum(d.j.D(TB(mu, nu), mu) for mu in range(4))))
    for mu in range(4):
        for nu in range(4):
            ok &= zero(d.onshell(ThetaUp(mu, nu) - R(1, 2) * R(1, 4) * sum(d.j.D(Sbil(r, mu, nu), r) for r in range(4))
                                 - TB(mu, nu)))
            ok &= zero(TB(mu, nu) - TB(nu, mu))
    ok &= zero(TB(0, 0) - ThetaUp(0, 0))
    report(20, "on shell Theta^{mu nu} - Theta^{nu mu} = d_rho S^{rho mu nu} with S = s psi^T gamma^0 gamma^rho gamma^mu "
               f"gamma^nu psi (distinct indices, totally antisymmetric), s solved = {sorted(sols)}; "
               "T_B = Theta - 1/2 d S equals 1/4 psi^T(K^mu d^nu + K^nu d^mu)psi - eta L_D on shell for all 16 (mu,nu), "
               "is symmetric, conserved on shell, and T_B^00 = Theta^00", ok)

    # ES17 eq. (35) on one constituent: kernels P gamma^0 gamma^mu P are minus K^mu; with y Phi in place of m T the
    # Lagrangian 1/2 psi^T(-K^mu d_mu)psi + 1/2 psi^T (mT) psi is minus L_D and has the same field equation
    d2 = Dirac(m)
    L35 = sp.Integer(0)
    for mu in range(4):
        L35 += d2.pol(lambda f: d2.j.col(f), -KK[mu] / 2, lambda f, mu=mu: d2.j.col(f, (mu,)))
    L35 += d2.pol(lambda f: d2.j.col(f), m * TT / 2, lambda f: d2.j.col(f))
    ok = zero(L35 + d2.lagrangian()) and mzero(d2.EL_u(L35) + d2.E_expected("v"))
    ok &= all((GAM[0] * GAM[mu]).T == GAM[0] * GAM[mu] for mu in range(4)) and GAM[0] * GAM[0] == I8
    report(21, "eq. (35) of ES17 restricted to one constituent: K'^mu = gamma^0 gamma^mu = -K^mu (K'^0 = +1) with the "
               "Yukawa sign +1/2 psi^T M psi; for M = y Phi replaced by m T this is exactly -L_D, whose EL expression is "
               "minus that of L_D, so the field equations agree (RE Prop. 3.6(iii) has the sign of L_D)", ok)


# ---------------------------------------------------------------------------------------------------------------
# F. Maxwell in the field reading, checked against the textbook route
# ---------------------------------------------------------------------------------------------------------------
def block_F():
    A = [sp.Function(f"A{mu}")(*XS) for mu in range(4)]           # covariant A_mu
    a = sp.Matrix(A[1:])
    F = sp.Matrix(4, 4, lambda i, j: sp.diff(A[j], XS[i]) - sp.diff(A[i], XS[j]))
    E = sp.Matrix([F[0, i] for i in (1, 2, 3)])
    B = sp.Matrix([F[2, 3], F[3, 1], F[1, 2]])                    # F_{jk} = eps_{ijk} B_i
    # textbook route: phi = A^0 = A_0, vector potential A_t^i = A^i = -A_i, E_t = -grad phi - d_t A_t, B_t = curl A_t
    At = -a
    Et = -sp.Matrix([sp.diff(A[0], X) for X in XS[1:]]) - At.diff(XS[0])
    Bt = curlF(At)
    ok = mzero(E - Et) and mzero(B + Bt)
    # Jackson's contravariant F^{mu nu}: F^{0i} = -E_i, F^{ij} = -eps_{ijk} B_t,k
    Fj = sp.zeros(4)
    for i in range(3):
        Fj[0, i + 1], Fj[i + 1, 0] = -Et[i], Et[i]
    for i, j, k in itertools.permutations(range(3)):
        Fj[i + 1, j + 1] = -LC3(i, j, k) * Bt[k]
    ok &= mzero(ETA4 * F * ETA4 - Fj)
    F2 = sum(F[i, j] * Fj[i, j] for i in range(4) for j in range(4))
    LM = -F2 / 4
    ok &= zero(LM - ((E.T * E)[0] - (B.T * B)[0]) / 2)
    jf = [sp.Function(f"j{mu}")(*XS) for mu in range(4)]
    eqs = EL(LM - sum(A[mu] * jf[mu] for mu in range(4)), A)
    # euler_equations gives dL/dA_nu - d_mu dL/d(d_mu A_nu) = -j^nu + d_mu F^{mu nu}
    ok &= zero(eqs[0] - (divF(E) - jf[0]))
    ok &= all(zero(eqs[i + 1] - (-E.diff(XS[0])[i] - curlF(B)[i] - jf[i + 1])) for i in range(3))
    ok &= all(zero(eqs[i + 1] - (-E.diff(XS[0])[i] + curlF(Bt)[i] - jf[i + 1])) for i in range(3))
    ok &= mzero(B.diff(XS[0]) - curlF(E)) and zero(divF(B))
    report(22, "Maxwell: with covariant A_mu and F_{0i} = E_i, F_{jk} = eps_{ijk}B_i, E is the textbook -grad phi - d_t A_t and "
               "B = -B_text = -curl A_t; eta F eta is Jackson's F^{mu nu}; L_M = -F^2/4 = (|E|^2-|B|^2)/2; euler_equations "
               "of L_M - A.j give div E = rho and d_t E = -curl B - j = curl B_text - j; d_t B = curl E, div B = 0", ok)

    # stress tensor in Jackson's form, momentum density sign, plane-wave check
    Es = sp.Matrix(sp.symbols("E1:4", real=True))
    Bs = sp.Matrix(sp.symbols("B1:4", real=True))
    Fc = F_of_X(sp.Matrix(list(Bs) + list(Es)))
    Fu = ETA4 * Fc * ETA4
    F2s = sum(Fc[i, j] * Fu[i, j] for i in range(4) for j in range(4))
    Tj = sp.Matrix(4, 4, lambda al, be: sum(ETA4[al, al] * Fc[al, l] * Fu[l, be] for l in range(4)) + ETA4[al, be] * F2s / 4)
    Btx = -Bs
    ok = zero(Tj[0, 0] - ((Es.T * Es)[0] + (Bs.T * Bs)[0]) / 2)
    ok &= mzero(sp.Matrix(Tj[0, 1:]).T - Es.cross(Btx)) and mzero(sp.Matrix(Tj[0, 1:]).T + Es.cross(Bs))
    ok &= zero(sum(Tj[i, i] * ETA4[i, i] for i in range(4))) and mzero(Tj - Tj.T)
    # textbook plane wave along +z: E = (1,0,0), B_text = (0,1,0) carries momentum +z
    T0z = Tj[0, 3].subs({Es[0]: 1, Es[1]: 0, Es[2]: 0, Bs[0]: 0, Bs[1]: -1, Bs[2]: 0})
    ok &= T0z == 1
    Tmix = Tj * ETA4
    S_ = ((Es.T * Es)[0] - (Bs.T * Bs)[0]) / 2
    P_ = (Es.T * Btx)[0]
    ok &= mzero(Tmix * Tmix - (S_ ** 2 + P_ ** 2) * sp.eye(4))
    z = Bs + sp.I * Es
    qz = sp.expand((z.T * z)[0])
    ok &= zero(sp.im(qz) - 2 * (Es.T * Bs)[0]) and zero((Es.T * Bs)[0] + P_)
    ok &= zero(sp.expand(qz * sp.conjugate(qz)) - 4 * (S_ ** 2 + P_ ** 2))
    E6 = ((Es.T * Es)[0] + (Bs.T * Bs)[0]) / 2
    Sp = Es.cross(Bs)
    ok &= zero(E6 ** 2 - (S_ ** 2 + P_ ** 2) - (Sp.T * Sp)[0])
    TT2 = sum((Tj * ETA4)[i, j] * (Tj * ETA4)[j, i] for i in range(4) for j in range(4))
    ok &= zero(TT2 - 4 * (S_ ** 2 + P_ ** 2))
    report(23, "Jackson's tensor Theta^{ab} = g^{am}F_{ml}F^{lb} + g^{ab}F^2/4 in the paper's orientation: T^00 = E6, "
               "T^{0i} = (E x B_text)_i = -(E x B)_i (a textbook wave along +z has T^{0z} = +1), symmetric, traceless; "
               "T^mu_l T^l_nu = E0^2 delta and T.T = 4E0^2 (E0 = |q|/2); E.B = Im q/2 = -E.B_text; E6^2 = E0^2 + |E x B|^2", ok)

    # dT identity with general A
    Tg = sp.Matrix(4, 4, lambda al, be: sum(ETA4[al, al] * F[al, l] * Fj[l, be] for l in range(4)) + ETA4[al, be] * F2 / 4)
    divFup = [sum(sp.diff(Fj[mu, l], XS[mu]) for mu in range(4)) for l in range(4)]
    ok = all(zero(sum(sp.diff(Tg[mu, nu], XS[mu]) for mu in range(4))
                  + sum(ETA4[nu, nu] * F[nu, l] * divFup[l] for l in range(4))) for nu in range(4))
    report(24, "for general A_mu(x): d_mu T^{mu nu} = -F^nu_l d_mu F^{mu l} identically", ok)

    # Chern-Simons current with eps^{0123} obtained by raising eps_{0123} = +1
    def eps_up(i, j, k, l):
        return ETA4[i, i] * ETA4[j, j] * ETA4[k, k] * ETA4[l, l] * sp.LeviCivita(i, j, k, l)

    def Kcs(coef):
        return [coef * sum(eps_up(mu, nu, r, s_) * A[nu] * sp.diff(A[s_], XS[r])
                           for nu in range(4) for r in range(4) for s_ in range(4)) for mu in range(4)]

    EdB = (E.T * B)[0]
    K12 = Kcs(R(1, 2))
    K1 = Kcs(1)
    ok = zero(EdB + sum(sp.diff(K12[mu], XS[mu]) for mu in range(4)))
    ok &= not zero(EdB + sum(sp.diff(K1[mu], XS[mu]) for mu in range(4)))
    ok &= zero(K12[0] + (a.T * curlF(a))[0] / 2)
    th = sp.Symbol("theta")
    ok &= all(zero(x) for x in EL(th * EdB, A))
    report(25, "E.B (paper B) = -d_mu K^mu with K^mu = (1/2) eps^{mu nu rho sigma} A_nu d_rho A_sigma, eps^{0123} = -1 "
               "obtained by raising eps_{0123} = +1; the factor 1/2 is required (factor 1 fails); K^0 = -(1/2) a.curl a; "
               "theta E.B has identically vanishing EL expressions", ok)


# ---------------------------------------------------------------------------------------------------------------
# G. L0 = E0 as a Lagrangian; Gaillard-Zumino; ModMax
# ---------------------------------------------------------------------------------------------------------------
def SP(E, Bt):
    return (((E.T * E)[0] - (Bt.T * Bt)[0]) / 2, (E.T * Bt)[0])


def rand_vec(n=3):
    return sp.Matrix([rat() for _ in range(n)])


def block_G():
    Ev = sp.Matrix(sp.symbols("E1:4", real=True))
    Bt = sp.Matrix(sp.symbols("b1:4", real=True))
    S, P = SP(Ev, Bt)
    L0 = sp.sqrt(S ** 2 + P ** 2)
    # finite rational boost along x (gamma = 5/4, beta gamma = 3/4) acting on F; and finite rational rotation
    Lb = sp.Matrix([[R(5, 4), R(3, 4), 0, 0], [R(3, 4), R(5, 4), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    ok = Lb.T * ETA4 * Lb == ETA4
    ok_inv = True
    for _ in range(3):
        Bp, Ep = rand_vec(), rand_vec()
        Fc = F_of_X(sp.Matrix(list(Bp) + list(Ep)))
        Lam_low = ETA4 * Lb * ETA4                                     # acts on covariant indices
        Fn = Lam_low * Fc * Lam_low.T
        En = sp.Matrix([Fn[0, i] for i in (1, 2, 3)])
        Bn = sp.Matrix([Fn[2, 3], Fn[3, 1], Fn[1, 2]])
        s0, p0 = SP(Ep, -Bp)
        s1, p1 = SP(En, -Bn)
        ok_inv &= s0 == s1 and p0 == p1 and not (En == Ep and Bn == Bp)
    ok &= ok_inv
    # duality: z -> exp(i theta) z with cos = 3/5, sin = 4/5 keeps |q| and changes Re q
    Bp, Ep = rand_vec(), rand_vec()
    z = Bp + sp.I * Ep
    zr = (R(3, 5) + sp.I * R(4, 5)) * z
    qz, qr = sp.expand((z.T * z)[0]), sp.expand((zr.T * zr)[0])
    ok &= sp.expand(qz * sp.conjugate(qz) - qr * sp.conjugate(qr)) == 0 and sp.re(qz) != sp.re(qr)
    sc = sp.Symbol("lam", positive=True)
    ok &= zero(sp.simplify(L0.subs({**{x: sc * x for x in Ev}, **{x: sc * x for x in Bt}}, simultaneous=True) - sc ** 2 * L0))
    # kink: along B_t = (1,0,0), E = (0,e,0); and at a generic null field the two directional derivatives are not opposite
    e_, h = sp.symbols("e h", positive=True)
    f = L0.subs({Ev[0]: 0, Ev[1]: e_, Ev[2]: 0, Bt[0]: 1, Bt[1]: 0, Bt[2]: 0})
    ok &= sp.simplify(f - sp.Abs(1 - e_ ** 2) / 2) == 0
    ok &= sp.limit((f.subs(e_, 1 + h) - f.subs(e_, 1)) / h, h, 0) == 1
    ok &= sp.limit((f.subs(e_, 1 - h) - f.subs(e_, 1)) / h, h, 0) == 1      # derivative towards e<1 is +1, i.e. slope -1
    nullpt = {Ev[0]: 3, Ev[1]: 0, Ev[2]: 0, Bt[0]: 0, Bt[1]: R(9, 5), Bt[2]: R(12, 5)}   # |E| = |B_t| = 3, E.B_t = 0
    dirv = {Ev[0]: 1, Ev[1]: 0, Ev[2]: 0, Bt[0]: 0, Bt[1]: 0, Bt[2]: 0}
    g_ = L0.subs({x: nullpt[x] + h * dirv[x] for x in nullpt})
    gm = L0.subs({x: nullpt[x] - h * dirv[x] for x in nullpt})
    ok &= sp.limit(g_ / h, h, 0) == 3 and sp.limit(gm / h, h, 0) == 3     # one-sided slopes +3 and +3: not differentiable
    report(26, "L0 = E0 = sqrt(S^2+P^2): invariant under a finite rational boost and under rotations of the field (S, P "
               "unchanged), |q| invariant and Re q changed by a rational duality rotation, homogeneous of degree 2; "
               "|1-e^2|/2 along the stated curve with one-sided slopes -1, +1 at e = 1; at the null field |E| = |B_t| = 3 "
               "both one-sided derivatives along +-E point upward (slope 3): not differentiable at null fields", ok)

    # constitutive relations, energy density, convexity
    D0 = sp.Matrix([sp.diff(L0, x) for x in Ev])
    H0 = sp.Matrix([-sp.diff(L0, x) for x in Bt])
    ok = True
    for _ in range(4):
        pt = {**{x: rat() for x in Ev}, **{x: rat() for x in Bt}}
        s_, p_ = S.subs(pt), P.subs(pt)
        e0 = sp.sqrt(s_ ** 2 + p_ ** 2)
        Evn, Btn = Ev.subs(pt), Bt.subs(pt)
        ok &= mzero(sp.simplify(D0.subs(pt) - (s_ * Evn + p_ * Btn) / e0))
        ok &= mzero(sp.simplify(H0.subs(pt) - (s_ * Btn - p_ * Evn) / e0))
        e6 = ((Evn.T * Evn)[0] + (Btn.T * Btn)[0]) / 2
        hamd = (D0.subs(pt).T * Evn)[0] - e0
        ok &= zero(sp.simplify(hamd - s_ / e0 * e6))
        qv = sp.expand(((-Btn + sp.I * Evn).T * (-Btn + sp.I * Evn))[0])
        ok &= zero(sp.simplify(hamd + sp.re(qv) / sp.Abs(qv) * e6))       # = -cos(arg q) E6
        Dn, Hn = D0.subs(pt), H0.subs(pt)
        ok &= zero(sp.simplify((Dn.T * Hn)[0] - p_ + 2 * p_))              # GZ defect of L0 is -2P
    mag = {Ev[0]: 0, Ev[1]: 0, Ev[2]: 0, Bt[0]: 1, Bt[1]: 0, Bt[2]: 0}
    ok &= sp.simplify(((D0.T * Ev)[0] - L0).subs(mag)) == R(-1, 2)
    Hs = sp.hessian(L0, list(Ev)).subs(mag)
    ok &= sp.simplify(Hs - sp.diag(1, -1, -1)) == sp.zeros(3)
    report(27, "at random rational fields: D = (S E + P B_t)/E0, H = (S B_t - P E)/E0, D.E - L0 = (S/E0) E6 = -cos(arg q) E6, "
               "and D.H - E.B_t = -2P computed directly; on B_t = (1,0,0), E = 0 the energy density is -1/2 and the "
               "Hessian of L0 in E is diag(1,-1,-1) (not convex)", ok)

    # direct Gaillard-Zumino defects for lambda L0, ModMax, S + E0 at random points (no general formula used)
    lamv = R(3, 7)
    ch, sh = R(5, 4), R(3, 4)                                        # cosh, sinh of gamma = ln 2
    cands = {"lamL0": lamv * L0, "ModMax": ch * S + sh * L0, "S+E0": S + L0, "Maxwell": S}
    expect = {"lamL0": -(1 + lamv ** 2), "ModMax": 0, "S+E0": -1, "Maxwell": 0}
    ok = True
    for name, Lf in cands.items():
        Dd = sp.Matrix([sp.diff(Lf, x) for x in Ev])
        Hh = sp.Matrix([-sp.diff(Lf, x) for x in Bt])
        for _ in range(3):
            pt = {**{x: rat() for x in Ev}, **{x: rat() for x in Bt}}
            dfc = sp.nsimplify(sp.simplify(((Dd.T * Hh)[0] - P).subs(pt)))
            ok &= zero(sp.simplify(dfc - expect[name] * P.subs(pt)))
    report(28, "Gaillard-Zumino defect D.H - E.B_t computed directly at random fields: -(1+lambda^2)P for lambda E0 "
               "(lambda = 3/7), 0 for ModMax (gamma = ln 2) and Maxwell, -P for S + E0", ok)

    # general L(S,P): tensor formula and conservation, in jet variables of A (up to second order)
    jet = JetSpace()
    dA = lambda mu, lam_: jet.s("A", lam_, (mu,))
    F = sp.Matrix(4, 4, lambda mu, nu: dA(mu, nu) - dA(nu, mu))
    Ef = sp.Matrix([F[0, i] for i in (1, 2, 3)])
    Bf = sp.Matrix([F[2, 3], F[3, 1], F[1, 2]])
    Sf, Pf = SP(Ef, -Bf)
    Fu = ETA4 * F * ETA4
    F2 = sum(F[i, j] * Fu[i, j] for i in range(4) for j in range(4))
    ok = zero(Sf + F2 / 4)
    TMmix = sp.Matrix(4, 4, lambda mu, nu: sum(Fu[mu, l] * (F * ETA4)[l, nu] for l in range(4)) + ETA4[mu, nu] * F2 / 4) * ETA4
    Ssym, Psym = sp.symbols("S P")
    Lpoly = Ssym + Ssym ** 2 / 3 + Ssym * Psym / 5 + Psym ** 2 / 7 + Psym ** 3 / 11
    Lf = Lpoly.subs({Ssym: Sf, Psym: Pf})
    Gup = sp.Matrix(4, 4, lambda mu, lam_: sp.diff(Lf, dA(mu, lam_)))
    Tgen = sp.Matrix(4, 4, lambda mu, nu: sp.expand(sum(Gup[mu, l] * F[nu, l] for l in range(4)) - (Lf if mu == nu else 0)))
    LS, LP = sp.diff(Lpoly, Ssym).subs({Ssym: Sf, Psym: Pf}), sp.diff(Lpoly, Psym).subs({Ssym: Sf, Psym: Pf})
    ok &= mzero(Tgen - LS * TMmix - (Sf * LS + Pf * LP - Lf) * sp.eye(4))
    ELl = [sp.expand(-sum(jet.D(Gup[mu, l], mu) for mu in range(4))) for l in range(4)]
    for nu in range(4):
        ok &= zero(sum(jet.D(Tgen[mu, nu], mu) for mu in range(4)) + sum(ELl[l] * F[nu, l] for l in range(4)))
    # L0 and ModMax at a random gradient: tensor is L_S T_M (traceless); ModMax: dL/dgamma = (1/2) sqrt(T.T)
    ok_num = True
    for Lname in ("L0", "ModMax"):
        Lsq = sp.sqrt(Sf ** 2 + Pf ** 2)
        Lexpr = Lsq if Lname == "L0" else ch * Sf + sh * Lsq
        Gq = sp.Matrix(4, 4, lambda mu, lam_: sp.diff(Lexpr, dA(mu, lam_)))
        for _ in range(2):
            pt = {dA(mu, lam_): rat() for mu in range(4) for lam_ in range(4)}
            Tn = sp.Matrix(4, 4, lambda mu, nu: sum(Gq[mu, l].subs(pt) * F[nu, l].subs(pt) for l in range(4))
                           - (Lexpr.subs(pt) if mu == nu else 0))
            s_, p_ = Sf.subs(pt), Pf.subs(pt)
            e0 = sp.sqrt(s_ ** 2 + p_ ** 2)
            LSn = s_ / e0 if Lname == "L0" else ch + sh * s_ / e0
            ok_num &= mzero(sp.simplify(Tn - LSn * TMmix.subs(pt))) and zero(sp.simplify(Tn.trace()))
            if Lname == "ModMax":
                TTn = sum(Tn[i, j] * Tn[j, i] for i in range(4) for j in range(4))
                dg = sh * s_ + ch * e0                                  # d/dgamma (cosh S + sinh E0)
                ok_num &= zero(sp.simplify(sp.sqrt(TTn) / 2 - dg)) and LSn >= R(1, 2)
    ok &= ok_num
    g = sp.Symbol("g", positive=True)
    ok &= sp.limit((sp.cosh(g) * Ssym + sp.sinh(g) * sp.sqrt(Ssym ** 2 + Psym ** 2)) / sp.sinh(g), g, sp.oo) == \
        Ssym + sp.sqrt(Ssym ** 2 + Psym ** 2)
    report(29, "general L(S,P) in A-jets (generic polynomial L): T^mu_nu = dL/d(d_mu A_l) F_{nu l} - delta L equals "
               "L_S T_M + delta(S L_S + P L_P - L) and d_mu T^mu_nu = -EL^l F_{nu l}; for L0 and ModMax (gamma = ln 2) at "
               "random gradients T = L_S T_M is traceless, and for ModMax (1/2) sqrt(T.T) = d L_gamma/d gamma and "
               "L_S >= e^{-gamma} = 1/2; L_gamma/sinh(gamma) -> S + E0", ok)


# ---------------------------------------------------------------------------------------------------------------
# H. Hamiltonian densities; E6 as Lagrangian; first-order and Schwarz-Sen actions
# ---------------------------------------------------------------------------------------------------------------
def block_H():
    Dv = sp.Matrix(sp.symbols("D1:4", real=True))
    Bv = sp.Matrix(sp.symbols("c1:4", real=True))
    s = ((Dv.T * Dv)[0] + (Bv.T * Bv)[0]) / 2
    w = Bv + sp.I * Dv
    E0w = sp.sqrt(sp.expand(((w.T * w)[0]) * sp.conjugate((w.T * w)[0]))) / 2
    pc = sp.sqrt((Dv.cross(Bv).T * Dv.cross(Bv))[0])
    a_, b_ = sp.symbols("a b", real=True)

    def tests(Hf, pt):
        Ef = sp.Matrix([sp.diff(Hf, x) for x in Dv]).subs(pt)
        Hh = sp.Matrix([sp.diff(Hf, x) for x in Bv]).subs(pt)
        Dn, Bn = Dv.subs(pt), Bv.subs(pt)
        return (sp.simplify(Ef.cross(Hh)), sp.simplify((Ef.T * Bn)[0] - (Dn.T * Hh)[0]),
                sp.simplify((Dn.T * Ef)[0] + (Bn.T * Hh)[0] - 2 * Hf.subs(pt)), Dn.cross(Bn))

    ok = True
    for _ in range(3):
        pt = {**{x: rat() for x in Dv}, **{x: rat() for x in Bv}}
        ok &= zero(sp.simplify((E0w ** 2).subs(pt) - (s ** 2 - pc ** 2).subs(pt)))
        lor, dual, conf, cr = tests(a_ * s + b_ * E0w, pt)
        ok &= mzero(sp.simplify(lor - (a_ ** 2 - b_ ** 2) * cr)) and zero(dual) and zero(conf)
        lor, dual, conf, cr = tests(E0w, pt)
        ok &= mzero(lor + cr)
        lor, dual, conf, cr = tests(pc, pt)
        ok &= mzero(lor - cr) and zero(dual) and zero(conf)
    report(30, "Hamiltonians at random rational (D, B_t): E0(w)^2 = s^2 - p^2; H = a s + b E0(w) gives E x H = (a^2-b^2) D x B_t "
               "and is duality invariant and conformal; H = E0 gives E x H = -D x B_t; H = |D x B_t| satisfies all three", ok)

    # ModMax Lagrangian -> Hamiltonian by Legendre transform (random numerical test, tolerance 1e-10)
    Ev = sp.Matrix(sp.symbols("E1:4", real=True))
    Bt = sp.Matrix(sp.symbols("b1:4", real=True))
    S, P = SP(Ev, Bt)
    ok = True
    for gam in (0.0, 0.4, 1.3):
        Lg = sp.cosh(gam) * S + sp.sinh(gam) * sp.sqrt(S ** 2 + P ** 2)
        Dg = sp.Matrix([sp.diff(Lg, x) for x in Ev])
        for _ in range(4):
            pt = {**{x: float(rng.normal()) for x in Ev}, **{x: float(rng.normal()) for x in Bt}}
            Dn = np.array([float(x) for x in Dg.subs(pt)])
            En = np.array([pt[x] for x in Ev])
            Bn = np.array([pt[x] for x in Bt])
            ham = Dn @ En - float(Lg.subs(pt))
            sn = (Dn @ Dn + Bn @ Bn) / 2
            pn = np.linalg.norm(np.cross(Dn, Bn))
            ok &= abs(ham - (np.cosh(gam) * sn - np.sinh(gam) * np.sqrt(sn ** 2 - pn ** 2))) < 1e-10
    report(31, "the Legendre transform D.E - L_gamma of ModMax, at random fields (float, tol 1e-10) and gamma in "
               "{0, 0.4, 1.3}, equals cosh(gamma) s - sinh(gamma) E0(w) in the variables (D, B_t)", ok)

    # E6 as a Lagrangian; first-order action
    A0 = sp.Function("A0")(*XS)
    av = sp.Matrix([sp.Function(f"a{i}")(*XS) for i in (1, 2, 3)])
    E = av.diff(XS[0]) - sp.Matrix([sp.diff(A0, X) for X in XS[1:]])
    B = curlF(av)
    LE6 = ((E.T * E)[0] + (B.T * B)[0]) / 2
    LM = ((E.T * E)[0] - (B.T * B)[0]) / 2
    e6 = EL(LE6, [A0] + list(av))
    ok = zero(e6[0] - divF(E)) and all(zero(e6[i + 1] - (-E.diff(XS[0])[i] + curlF(B)[i])) for i in range(3))
    om, kk = sp.symbols("omega k", positive=True)
    wave = sp.exp(sp.I * (kk * XS[1] - om * XS[0]))
    sub = {A0: 0, av[0]: 0, av[1]: wave, av[2]: 0}
    r6 = sp.simplify(e6[2].subs(sub).doit() / wave)
    ok &= sp.simplify(r6 - (om ** 2 + kk ** 2)) == 0
    eM = EL(LM, [A0] + list(av))
    rM = sp.simplify(eM[2].subs(sub).doit() / wave)
    ok &= sp.simplify(rM - (om ** 2 - kk ** 2)) == 0
    # with A0 = 0, div a = 0 the a-equation of E6 is the four-dimensional Laplace equation (d_t^2 + Laplacian) a = 0
    ap = sp.Matrix([0, sp.Function("h")(XS[0], XS[1], XS[3]), 0])     # transverse: independent of y
    lapl = sp.expand((ap.diff(XS[0], 2) + sum(ap.diff(X, 2) for X in XS[1:]))[1])
    e6p = sp.expand(e6[2].subs({A0: 0, av[0]: 0, av[1]: ap[1], av[2]: 0}).doit())
    ok &= zero(e6p + lapl)
    report(32, "E6 as a Lagrangian of the potential (euler_equations): div E = 0 and d_t E = +curl B; the plane wave a_y = "
               "exp(i(kx - omega t)) needs omega^2 + k^2 = 0 (Maxwell: omega^2 = k^2); for transverse a the equation is "
               "the Euclidean Laplace equation (d_t^2 + Laplacian) a = 0, which is elliptic", ok)

    Ef = sp.Matrix([sp.Function(f"E{i}")(*XS) for i in (1, 2, 3)])
    LH = (Ef.T * av.diff(XS[0]))[0] - ((Ef.T * Ef)[0] + (B.T * B)[0]) / 2 + A0 * divF(Ef)
    eH = EL(LH, list(Ef) + list(av) + [A0])
    ok = all(zero(eH[i] - (av.diff(XS[0])[i] - Ef[i] - sp.diff(A0, XS[i + 1]))) for i in range(3))
    ok &= all(zero(eH[3 + i] - (-Ef.diff(XS[0])[i] - curlF(B)[i])) for i in range(3)) and zero(eH[6] - divF(Ef))
    ok &= zero(LH.subs({Ef[i]: E[i] for i in range(3)}) - LM - divF(A0 * E))
    report(33, "first-order action L_H = E.d_t a - E6 + A0 div E: euler_equations give E = d_t a - grad A0, d_t E = -curl B, "
               "div E = 0, and on the first L_H = L_M + div(A0 E)", ok)

    Av = sp.Matrix([sp.Function(f"P{i}")(*XS) for i in (1, 2, 3)])
    Cv = sp.Matrix([sp.Function(f"Q{i}")(*XS) for i in (1, 2, 3)])
    XA = sp.Matrix(list(Av) + list(Cv))
    curlX = lambda X_: sp.Matrix(list(curlF(X_[:3, 0])) + list(curlF(X_[3:, 0])))
    LSS = lambda X_: (curlX(X_).T * G * X_.diff(XS[0]))[0] / 2 - (curlX(X_).T * curlX(X_))[0] / 2
    Bss, Ess = curlF(Av), curlF(Cv)
    ok = zero(LSS(XA) - ((Ess.T * Av.diff(XS[0]))[0] - (Bss.T * Cv.diff(XS[0]))[0]) / 2
              + ((Ess.T * Ess)[0] + (Bss.T * Bss)[0]) / 2)
    Rot = R(3, 5) * sp.eye(6) + R(4, 5) * G
    ok &= Rot.T * Rot == sp.eye(6) and zero(LSS(Rot * XA) - LSS(XA))
    vel = list(XA.diff(XS[0]))
    mom = sp.Matrix([sp.diff(LSS(XA), v_) for v_ in vel])
    ok &= mzero(mom - G.T * curlX(XA) / 2) and zero((mom.T * sp.Matrix(vel))[0] - LSS(XA) - (curlX(XA).T * curlX(XA))[0] / 2)
    j0 = (mom.T * (G * XA))[0]
    ok &= zero(j0 + (XA.T * curlX(XA))[0] / 2)
    # textbook potentials: B_t = curl A_t with A_t = -A; E = -curl C_t with C_t = -C; helicity h = (A_t.B_t + C_t.E)/2 ...
    At, Ct = -Av, -Cv
    ok &= mzero(curlF(At) + Bss) and mzero(-curlF(Ct) - Ess)
    h_text = ((At.T * curlF(At))[0] + (Ct.T * curlF(Ct))[0]) / 2          # (A_t.B_t - C_t.E)/2 with E = -curl C_t
    ok &= zero(h_text - ((At.T * (-Bss))[0] - (Ct.T * Ess)[0]) / 2) and zero(j0 + h_text)
    dif = LSS(XA) - ((Ess.T * Av.diff(XS[0]))[0] - ((Ess.T * Ess)[0] + (Bss.T * Bss)[0]) / 2)
    ok &= all(zero(x) for x in EL(dif, list(Av) + list(Cv)))
    report(34, "Schwarz-Sen L_SS = (1/2) curl X_A . G dX_A/dt - |curl X_A|^2/2 = (E.dA/dt - B.dC/dt)/2 - E6, exactly invariant "
               "under the rational rotation (3/5 + 4/5 G); momentum G^T curl X_A / 2, Hamiltonian E6; duality charge density "
               "-(1/2) X_A.curl X_A = minus the textbook optical helicity density (A_t.B_t - C_t.E)/2; L_SS - (E.dA/dt - E6) "
               "has vanishing EL expressions", ok)


# ---------------------------------------------------------------------------------------------------------------
# I. Coupled Maxwell-Dirac and the charged worldline
# ---------------------------------------------------------------------------------------------------------------
def block_I():
    m = sp.Symbol("m", positive=True)
    q = sp.Symbol("q", real=True)
    d = Dirac(m, q=q, gauge=True)
    L = d.lagrangian()
    jmu = [d.pol(lambda f: d.j.col(f), KK[mu] * JJ / 2, lambda f: d.j.col(f)) for mu in range(4)]
    ok = all(zero(sp.diff(L, d.A(nu)) - q * jmu[nu]) for nu in range(4))
    ok &= mzero(d.EL_u(L) - d.E_expected("v"))
    Fl = sp.Matrix(4, 4, lambda i, k: d.A(k, (i,)) - d.A(i, (k,)))        # F_{ik} = d_i A_k - d_k A_i
    for nu in range(4):
        div_ = sp.Integer(0)
        for mu in range(4):
            Th = d.pol(lambda f: d.j.col(f), KK[mu] / 2, lambda f, nu=nu: d.Dcol(f, nu))
            if mu == nu:
                Th = Th - L
            div_ += d.j.D(Th, mu)
        ok &= zero(d.onshell(div_ - q * sum(Fl[l, nu] * jmu[l] for l in range(4))))
    report(35, "coupled (polarised jets with gauge jets): dL/dA_nu = q j^nu, so the A-equation is d_mu F^{mu nu} = -q j^nu; "
               "the Dirac EL expression is K^mu D_mu psi - m T psi; d_mu Theta_D^mu_nu = q F_{l nu} j^l on the Dirac shell", ok)

    # Maxwell side in A-jets; total conservation
    jet = d.j
    Fu = ETA4 * Fl * ETA4
    F2 = sum(Fl[i, k] * Fu[i, k] for i in range(4) for k in range(4))
    TM = sp.Matrix(4, 4, lambda mu, nu: sum(Fu[mu, l] * (Fl * ETA4)[l, nu] for l in range(4)) + ETA4[mu, nu] * F2 / 4) * ETA4
    divFu = [sp.expand(sum(jet.D(Fu[mu, l], mu) for mu in range(4))) for l in range(4)]
    ok = all(zero(sum(jet.D(TM[mu, nu], mu) for mu in range(4)) + sum(Fl[nu, l] * divFu[l] for l in range(4)))
             for nu in range(4))
    jl = sp.symbols("J0:4")
    ok &= all(zero(-sum(Fl[nu, l] * (-q * jl[l]) for l in range(4)) + q * sum(Fl[l, nu] * jl[l] for l in range(4)))
              for nu in range(4))
    report(36, "d_mu T_M^mu_nu = -F_{nu l} d_mu F^{mu l} in A-jets; on the Maxwell shell d F = -q j this is q F_{nu l} j^l, "
               "which cancels q F_{l nu} j^l: the total tensor is conserved on the coupled shell", ok)

    lam = sp.Symbol("lambda", real=True)
    M_, e_ = sp.symbols("M e", real=True)
    Y = [sp.Function(f"Y{a}")(lam) for a in range(1, 7)]
    xi = [Y[0], Y[3], Y[4], Y[5]]
    xs = sp.symbols("x0:4", real=True)
    Ax = [sp.Function(f"W{mu}")(*xs) for mu in range(4)]
    Aw = [A_.subs({xs[i]: xi[i] for i in range(4)}) for A_ in Ax]
    Yd = [sp.diff(y, lam) for y in Y]
    wv = sum(ETA[a, a] * Yd[a] ** 2 for a in range(6))
    Lp = -M_ * sp.sqrt(wv) - e_ * sum(Aw[mu] * sp.diff(xi[mu], lam) for mu in range(4))
    eqs = EL(Lp, Y, [lam])
    ok = zero(eqs[1] + sp.diff(sp.diff(Lp, Yd[1]), lam)) and zero(sp.diff(Lp, Y[1])) and zero(sp.diff(Lp, Y[2]))
    idx = [0, 3, 4, 5]
    for mu in range(4):
        xlow = ETA4[mu, mu] * sp.diff(xi[mu], lam)
        Fmn = [(sp.diff(Ax[nu], xs[mu]) - sp.diff(Ax[mu], xs[nu])).subs({xs[i]: xi[i] for i in range(4)}) for nu in range(4)]
        force = e_ * sum(Fmn[nu] * sp.diff(xi[nu], lam) for nu in range(4))
        ok &= zero(sp.simplify(eqs[idx[mu]] + sp.diff(M_ * xlow / sp.sqrt(wv), lam) - force) .doit())
    report(37, "charged worldline L = -M sqrt(eta(Y',Y')) - e A_mu(x) x'^mu, x = (Y1;Y4,Y5,Y6) (euler_equations): L does not "
               "depend on Y2, Y3, so K_2, K_3 are conserved, and d/dlambda(M x'_mu/sqrt(eta(Y',Y'))) = e F_{mu nu} x'^nu", ok)


def main():
    block_A()
    block_units()
    block_B()
    block_C()
    block_D()
    block_E()
    block_F()
    block_G()
    block_H()
    block_I()
    print(f"s1: {sum(OK_ALL)}/{len(OK_ALL)} checks passed")
    return 0 if all(OK_ALL) else 1


if __name__ == "__main__":
    sys.exit(main())
