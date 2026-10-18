#!/usr/bin/env python3
"""Track d2 -- What a mass breaks: duality, chirality, and the two senses of rotation.

    python3 investigations/rest_energy/verify/d2_symmetry_breaking.py

Prints one 'PASS d2-<n>: ...' line per verified claim and exits 0 only if every check passes.

Conventions are those of the unified paper (tex/s02_arena.tex eq. realisation, s03 prop:twoK, s12 thm:emfield,
s16, s17):  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1), e1..e6 realised on R^8, G e_i = e_{i+3}, G e_{i+3} = -e_i,
X = sum_i (B_i e_i + E_i e_{i+3}), z = B + iE, q = z.z.  Units hbar = c = 1 unless a symbol is shown.

Exactness: every one-constituent statement is checked in integer / rational arithmetic (numpy int64 for blade
products, sympy DomainMatrix over QQ for ranks and null spaces, sympy for symbolic identities).  The
three-constituent statements (d2-21a, d2-21, d2-22, d2-24) use float64 matrices whose entries are dyadic rationals with
small numerators; every result is checked to be dyadic (tolerance 1e-9 on 2^12 * entry) and then converted to integers
before any rank is taken (integer Gram matrices, ranks over QQ), so those ranks are exact as well.  d2-24 re-derives the
electroweak table of Theorem 22.3 of ES17 in integer arithmetic (products of integer-valued float64 matrices, checked
exact below 2^50).

Claim d2-0 is a consistency check of the constructed realisation, not a claim of the section.
"""
import itertools
import sys
from fractions import Fraction

import numpy as np
import sympy as sp
from sympy import QQ
from sympy.polys.matrices import DomainMatrix

FAILS = []


def report(cid, ok, text):
    if ok:
        print(f"PASS {cid}: {text}")
    else:
        print(f"FAIL {cid}: {text}")
        FAILS.append(cid)


# ----------------------------------------------------------------------------------------------- exact linear algebra
def _qq(x):
    if isinstance(x, Fraction):
        return QQ(int(x.numerator), int(x.denominator))
    if isinstance(x, (sp.Rational, sp.Integer)):
        return QQ(int(x.p), int(x.q))
    return QQ(int(x))


def to_dm(rows):
    rows = [[_qq(x) for x in r] for r in rows]
    return DomainMatrix(rows, (len(rows), len(rows[0])), QQ)


def rank_q(rows):
    return to_dm(rows).rank()


def nullspace_q(rows):
    """Basis of the right null space, as lists of Fractions."""
    ns = to_dm(rows).nullspace()
    out = []
    for r in ns.to_Matrix().tolist():
        out.append([Fraction(int(sp.Rational(x).p), int(sp.Rational(x).q)) for x in r])
    return out


def vec(M):
    return [int(x) for x in np.asarray(M).reshape(-1)]


# ----------------------------------------------------------------------------------------------- the realisation
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def kron(*ms):
    r = ms[0]
    for m_ in ms[1:]:
        r = np.kron(r, m_)
    return r


e = {1: kron(sx, I2, I2), 2: kron(sz, sx, I2), 3: kron(sz, sz, sx),
     4: kron(ep, I2, I2), 5: kron(sz, ep, I2), 6: kron(sz, sz, ep)}
I8 = np.eye(8, dtype=np.int64)
eta6 = np.diag([1, 1, 1, -1, -1, -1]).astype(np.int64)
I6 = np.eye(6, dtype=np.int64)


def mm(*ms):
    r = ms[0]
    for m_ in ms[1:]:
        r = r @ m_
    return r


ok = all(np.array_equal(e[a] @ e[b] + e[b] @ e[a], 2 * eta6[a - 1, b - 1] * I8) for a in e for b in e)
J = e[2] @ e[3]
Om = mm(e[1], e[4], e[5], e[6])
Gam = J @ Om
T = mm(e[1], e[2], e[3])
C = e[2]
vol = mm(e[1], e[2], e[3], e[4], e[5], e[6])
ok &= np.array_equal(J @ J, -I8) and np.array_equal(Om @ Om, -I8) and np.array_equal(Gam @ Gam, I8)
ok &= np.array_equal(T @ T, -I8) and np.array_equal(T.T, -T) and np.array_equal(vol, Gam)
ok &= all(np.array_equal(J @ e[a], e[a] @ J) for a in (1, 4, 5, 6))
report("d2-0", ok, "consistency check of the constructed realisation (eq. (1) of ES17, not a claim): Clifford relations of "
       "e1..e6 on R^8; J^2=Om^2=T^2=-1, Gamma^2=1, Gamma=e1...e6, T antisymmetric, J commutes with gamma^mu")

# G on V, in coordinates (B_1,B_2,B_3,E_1,E_2,E_3)
G = np.zeros((6, 6), dtype=np.int64)
for i in range(3):
    G[3 + i, i] = 1
    G[i, 3 + i] = -1


def vec_action(Bv):
    """6x6 matrix of v -> [Bv, v] on V = span(e_1..e_6) (coordinates in the basis e_a)."""
    M = np.zeros((6, 6), dtype=object)
    for c in range(1, 7):
        w = Bv @ e[c] - e[c] @ Bv
        for a in range(1, 7):
            M[a - 1, c - 1] = Fraction(int(eta6[a - 1, a - 1]) * int(np.trace(e[a] @ w)), 8)
    return M


def frac_mat(M):
    return np.array([[Fraction(x) for x in r] for r in np.asarray(M)], dtype=object)


def as_int(M):
    M = np.asarray(M, dtype=object)
    assert all(Fraction(x).denominator == 1 for x in M.reshape(-1))
    return np.array([[int(x) for x in r] for r in M], dtype=np.int64)


def half(Bv):
    # bivectors here are (1/2) e_a e_b; their vector action has integer entries
    return as_int(frac_mat(vec_action(Bv)) * Fraction(1, 2))


L = [half(e[5] @ e[6] - e[2] @ e[3]), half(e[6] @ e[4] - e[3] @ e[1]), half(e[4] @ e[5] - e[1] @ e[2])]
K = [half(e[3] @ e[5] - e[2] @ e[6]), half(e[1] @ e[6] - e[3] @ e[4]), half(e[2] @ e[4] - e[1] @ e[5])]
lG = L + K

# ================================================================================================= d2-1 duality
th, ka, m, tau, al = sp.symbols('theta kappa m tau alpha', real=True)
Bs = sp.symbols('B1:4', real=True)
Es = sp.symbols('E1:4', real=True)
Xs = sp.Matrix(list(Bs) + list(Es))
Gs = sp.Matrix(G)
etas = sp.Matrix(eta6)
R = sp.cos(th) * sp.eye(6) + sp.sin(th) * Gs
ok = (Gs * Gs == -sp.eye(6))
# R(theta) = exp(theta G): R(0)=1 and dR/dtheta = G R, which characterises the exponential
ok &= sp.simplify(R.diff(th) - Gs * R) == sp.zeros(6, 6) and R.subs(th, 0) == sp.eye(6)


def zvec(X):
    return [X[i] + sp.I * X[3 + i] for i in range(3)]


def qf(z):
    return sum(zz ** 2 for zz in z)


zX = zvec(Xs)
zR = zvec(R * Xs)
ok &= all(sp.simplify(sp.expand(zR[i] - sp.exp(sp.I * th) * zX[i], complex=True)) == 0 for i in range(3))
u_X = sum(x ** 2 for x in Xs) / 2
ok &= sp.simplify((R * Xs).dot(R * Xs) / 2 - u_X) == 0
ok &= sp.simplify(sp.expand(qf(zR) - sp.exp(2 * sp.I * th) * qf(zX), complex=True)) == 0
ok &= sp.simplify(sp.expand(sp.exp(2 * sp.I * th) - 1 - 2 * sp.I * sp.sin(th) * sp.exp(sp.I * th), complex=True)) == 0
# Rainich complexion (Prop 12.12 of ES17): an electric plane of strength f_e and a magnetic one of strength f_m is
# z = (f_m + i f_e, 0, 0).  With phi = arg(f_m + i f_e) (cos phi = f_m/r, sin phi = f_e/r, r = |z|, any real f_m, f_e
# not both zero): q = r^2 e^{2 i phi}, and e^{-i phi} z = r e_1 is purely magnetic.  The principal arctan(f_e/f_m)
# gives -r e_1 when f_m < 0, so the branch has to be fixed as arg.
fm_, fe_ = sp.symbols('f_m f_e', real=True)
r_ = sp.sqrt(fm_ ** 2 + fe_ ** 2)
cph, sph = fm_ / r_, fe_ / r_
zc = fm_ + sp.I * fe_
qc = sp.expand(zc ** 2)
ok &= sp.simplify(sp.re(qc) - r_ ** 2 * (cph ** 2 - sph ** 2)) == 0
ok &= sp.simplify(sp.im(qc) - r_ ** 2 * 2 * sph * cph) == 0                      # q = r^2 e^{2 i phi}
ok &= sp.simplify(sp.expand((cph - sp.I * sph) * zc) - r_) == 0                  # e^{-i phi} z = r
ok &= sp.simplify(cph ** 2 + sph ** 2 - 1) == 0
for fmv, fev in ((4, 3), (-4, 3), (-4, -3), (0, 2)):                             # exact checks of the branch
    zv = sp.Integer(fmv) + sp.I * fev
    av = sp.arg(zv)
    ok &= sp.expand((sp.cos(av) - sp.I * sp.sin(av)) * zv) == sp.Abs(zv)
at = sp.atan(sp.Rational(3, -4))                                                 # principal arctan fails for f_m < 0
ok &= sp.expand((sp.cos(at) - sp.I * sp.sin(at)) * (sp.Integer(-4) + 3 * sp.I)) == -5
report("d2-1", ok, "exp(theta G)=cos+sin G acts as z->e^{i theta}z, fixes u=|z|^2/2, sends q->e^{2i theta}q; "
       "e^{2i theta}-1 = 2i sin(theta) e^{i theta}, so the stabiliser of q is U(1) iff q=0 and pi*Z iff q!=0; for "
       "z=(f_m+i f_e)e_1, q=|z|^2 e^{2i phi} with phi=arg(f_m+i f_e) (not the principal arctan(f_e/f_m), which fails for "
       "f_m<0): arg q = twice the Rainich complexion and e^{-phi G} makes the field purely magnetic")

# ================================================================================================= d2-2 not an isometry
ok = sp.simplify(R.T * etas * R - (sp.cos(2 * th) * etas + sp.sin(2 * th) * etas * Gs)) == sp.zeros(6, 6)
ok &= sp.simplify(R.T * R - sp.eye(6)) == sp.zeros(6, 6)
ok &= np.array_equal(G.T @ eta6, eta6 @ G) and not np.array_equal(G.T @ eta6, -eta6 @ G)  # G eta-self-adjoint, not in so(3,3)
# isometry iff cos 2theta = 1 and sin 2theta = 0 (eta and eta G are linearly independent)
ok &= rank_q([vec(eta6), vec(eta6 @ G)]) == 2
ok &= sp.solveset(sp.Eq(sp.cos(2 * th), 1), th, domain=sp.Interval(0, 2 * sp.pi)) == sp.FiniteSet(0, sp.pi, 2 * sp.pi)
# the quadratic part of the interior Lagrangian is rotated, not kept
Xd = sp.Matrix(sp.symbols('Bd1:4 Ed1:4', real=True))
Lq = lambda X, V: (V.T * etas * V)[0] - ka ** 2 * (X.T * etas * X)[0]
Lq_GX = (Xd.T * etas * Gs * Xd)[0] - ka ** 2 * (Xs.T * etas * Gs * Xs)[0]
ok &= sp.simplify(Lq(R * Xs, R * Xd) - (sp.cos(2 * th) * Lq(Xs, Xd) + sp.sin(2 * th) * Lq_GX)) == 0
# why (ii) is inherited: under X -> F_X (F_{0i}=E_i, F_{jk}=eps_ijk B_i; signature (+,-,-,-), eps_0123=+1) G is the
# Hodge star, (1/2)F.F = eta(X,X) and (1/2)F.*F = eta(X,GX): (ii) is the textbook rotation of F.F into F.*F
etaM4 = sp.diag(1, -1, -1, -1)


def F_of(X):
    Fm = sp.zeros(4, 4)
    for i in range(3):
        Fm[0, i + 1], Fm[i + 1, 0] = X[3 + i], -X[3 + i]
        j, k = (i + 1) % 3, (i + 2) % 3
        Fm[j + 1, k + 1] += X[i]
        Fm[k + 1, j + 1] -= X[i]
    return Fm


def hodge(Fm):
    Fu = etaM4 * Fm * etaM4
    return sp.Matrix(4, 4, lambda a_, b_: sum(sp.LeviCivita(a_, b_, c_, d_) * Fu[c_, d_]
                                             for c_ in range(4) for d_ in range(4)) / 2)


def hcontr(F1, F2):
    return sp.expand(sum(F1[a_, b_] * (etaM4 * F2 * etaM4)[a_, b_] for a_ in range(4) for b_ in range(4)) / 2)


FX = F_of(Xs)
ok &= sp.expand(F_of(Gs * Xs) - hodge(FX)) == sp.zeros(4, 4)
ok &= sp.expand(hcontr(FX, FX) - (Xs.T * etas * Xs)[0]) == 0
ok &= sp.expand(hcontr(FX, hodge(FX)) - (Xs.T * etas * Gs * Xs)[0]) == 0
report("d2-2", ok, "(e^{theta G})^T eta e^{theta G} = cos2theta eta + sin2theta eta G and (e^{theta G})^T e^{theta G}=1: "
       "duality is an isometry of the positive form for all theta and of eta only for theta in pi*Z; G is eta-self-adjoint, "
       "not in so(3,3); the split-form Lagrangian is rotated into its eta G partner; under X->F_X (Thm 12.3) G is the Hodge "
       "star and eta(X,X), eta(X,GX) are (1/2)F.F, (1/2)F.*F, so this is the textbook rotation of F.F into F.*F (inherited)")

# ================================================================================================= d2-3 rest energy
Bv_ = sp.Matrix(Bs)
Ev_ = sp.Matrix(Es)
S_poy = Ev_.cross(Bv_)
uu = (Bv_.dot(Bv_) + Ev_.dot(Ev_)) / 2
q_re = Bv_.dot(Bv_) - Ev_.dot(Ev_)
q_im = 2 * Ev_.dot(Bv_)
ok = sp.expand(uu ** 2 - S_poy.dot(S_poy) - (q_re ** 2 + q_im ** 2) / 4) == 0
# q is l_G-invariant: d/de q(z + e A z) = 2 z.(A z) = 0 for every generator; u is not boost invariant
for A in lG:
    As = sp.Matrix(A)
    ok &= sp.expand((Xs.T * etas * As * Xs)[0]) == 0
    ok &= sp.expand((Xs.T * etas * Gs * As * Xs)[0]) == 0
    ok &= np.array_equal(A @ G, G @ A) and np.array_equal(A.T @ eta6 + eta6 @ A, np.zeros((6, 6), dtype=np.int64))
ok &= any(sp.expand((Xs.T * sp.Matrix(A) * Xs)[0]) != 0 for A in K)
# frame Poynting vector S = E x B (frame variables) is minus the textbook one: B_t = -B (Section 1.4 of ES17)
S_text = Ev_.cross(-Bv_)
ok &= sp.expand(S_text + S_poy) == sp.zeros(3, 1) and sp.expand(S_text.dot(S_text) - S_poy.dot(S_poy)) == 0
report("d2-3", ok, "u^2-|S|^2=|q|^2/4 with the frame Poynting vector S=E x B=-E x B_t=-S_t (so u>=|q|/2, equality iff "
       "S=0); Re q and Im q are invariant under the six generators of l_G, which commute with G and lie in so(3,3); u is "
       "not boost invariant: E0=|q|/2 is the only one of u, Re q, |q| invariant under both l_G and U(1)_G")

# ================================================================================================= d2-4 interior energy
a = sp.Matrix(sp.symbols('a1:4', real=True))
b = sp.Matrix(sp.symbols('b1:4', real=True))
c = sp.Matrix(sp.symbols('c1:4', real=True))
d = sp.Matrix(sp.symbols('d1:4', real=True))
Cs, Sn = sp.symbols('Cs Sn', real=True)  # cos(kappa tau), sin(kappa tau)


def ddt(expr):
    return sp.diff(expr, Cs) * (-ka * Sn) + sp.diff(expr, Sn) * (ka * Cs)


def red(expr):
    return sp.reduced(sp.expand(expr), [Sn ** 2 + Cs ** 2 - 1], Sn, Cs)[1]


# z(tau) = c_- e^{-i k tau} + c_+ e^{i k tau}, c_- = a + i b, c_+ = c + i d
Bt = (a + c) * Cs + (b - d) * Sn
Et = (b + d) * Cs + (c - a) * Sn
Xt = sp.Matrix(list(Bt) + list(Et))
Xtd = Xt.applyfunc(ddt)
ok = all(red(v) == 0 for v in (Xtd.applyfunc(ddt) + ka ** 2 * Xt))  # solves Xddot = -kappa^2 X
GX = Gs * Xt
E_noether = m / 2 * (Xtd.T * etas * Xtd)[0] - m * ka ** 2 / 2 * (GX.T * etas * GX)[0]
cm_cp_re = a.dot(c) - b.dot(d)
cm_cp_im = a.dot(d) + b.dot(c)
ok &= red(E_noether - 2 * m * ka ** 2 * cm_cp_re) == 0
# complex constant  (m/2)(q(zdot) + kappa^2 q(z)) = 2 m kappa^2 c_-.c_+
zt = [Bt[i] + sp.I * Et[i] for i in range(3)]
ztd = [ddt(x) for x in zt]
EC = m / 2 * (qf(ztd) + ka ** 2 * qf(zt))
ok &= red(sp.re(sp.expand(EC)) - 2 * m * ka ** 2 * cm_cp_re) == 0
ok &= red(sp.im(sp.expand(EC)) - 2 * m * ka ** 2 * cm_cp_im) == 0
# constant duality rotation multiplies it by e^{2 i theta}
XR = R * Xt
XRd = XR.applyfunc(ddt)
zR_ = [XR[i] + sp.I * XR[3 + i] for i in range(3)]
zRd_ = [XRd[i] + sp.I * XRd[3 + i] for i in range(3)]
EC_R = m / 2 * (qf(zRd_) + ka ** 2 * qf(zR_))
diffR = sp.expand(EC_R - (sp.cos(2 * th) + sp.I * sp.sin(2 * th)) * EC, complex=True)
ok &= red(sp.simplify(sp.re(diffR))) == 0 and red(sp.simplify(sp.im(diffR))) == 0
# E_C is conserved, and the Noether energy vanishes when either sense is absent
ok &= red(ddt(EC)) == 0
ok &= red(E_noether.subs({x: 0 for x in list(c) + list(d)})) == 0 and red(E_noether.subs({x: 0 for x in list(a) + list(b)})) == 0
report("d2-4", ok, "on the frame field z=c_-e^{-ik tau}+c_+e^{ik tau}: E_C=(m/2)(q(zdot)+k^2 q(z))=2 m k^2 c_-.c_+ is "
       "conserved, the Noether energy is Re E_C and vanishes when either sense is absent, and a constant duality "
       "rotation maps solutions to solutions and multiplies E_C by e^{2i theta}")

# ------------------------------------------------------------------ d2-4b momentum field versus frame field
# momentum p = m eta (Xdot + kappa G X); momentum field pi = eta^{-1} p = m alpha_+ X, z(pi) = 2 i m kappa c_+ e^{i kappa tau}
pi_ = Xtd + ka * GX                                    # pi / m
z_pi = [m * (pi_[i] + sp.I * pi_[3 + i]) for i in range(3)]
cpv_ = [c[i] + sp.I * d[i] for i in range(3)]
cmv_ = [a[i] + sp.I * b[i] for i in range(3)]
eik = Cs + sp.I * Sn                                   # e^{i kappa tau}
ok = all(red(sp.expand(z_pi[i] - 2 * sp.I * m * ka * cpv_[i] * eik)) == 0 for i in range(3))
ok &= red(m ** 2 * pi_.dot(pi_) - 4 * m ** 2 * ka ** 2 * (c.dot(c) + d.dot(d))) == 0          # |p|^2 = |pi|^2
pB = [sp.re(sp.expand(zz)) for zz in z_pi]
pE = [sp.im(sp.expand(zz)) for zz in z_pi]
u_pi = sum(pB[i] ** 2 + pE[i] ** 2 for i in range(3)) / 2
ok &= red(u_pi - 2 * m ** 2 * ka ** 2 * (c.dot(c) + d.dot(d))) == 0                        # u(pi) = |p|^2/2
q_pi = sp.expand(sum(zz ** 2 for zz in z_pi))
cpcp = sp.expand(sum(x ** 2 for x in cpv_))
ok &= red(sp.expand(q_pi * sp.conjugate(q_pi)) - 16 * m ** 4 * ka ** 4 * sp.expand(cpcp * sp.conjugate(cpcp))) == 0
S_pi = sp.Matrix(pE).cross(sp.Matrix(pB))
ok &= all(red(ddt(red(sv))) == 0 for sv in S_pi)                                              # S(pi) constant
ok &= all(red(ddt(red(v))) == 0 for v in (u_pi, sp.expand(q_pi * sp.conjugate(q_pi))))
# p = e^{-kappa tau G} p_0 : the momentum field rotates by pure duality, pi(tau) = e^{kappa tau G} pi_0
pi0 = pi_.subs({Cs: 1, Sn: 0})
ok &= all(red(v) == 0 for v in (pi_ - (Cs * sp.eye(6) + Sn * Gs) * pi0))
# frame field: u(X) = (|c_-|^2+|c_+|^2)/2 + Re(conj(c_-).c_+ e^{2 i k tau}), constant iff conj(c_-).c_+ = 0;
# q(X) = c_-.c_- e^{-2ik tau} + 2 c_-.c_+ + c_+.c_+ e^{2ik tau}, so E_0(X) oscillates in general
w_re = a.dot(c) + b.dot(d)
w_im = a.dot(d) - b.dot(c)
u_t = Xt.dot(Xt) / 2
u_claim = (a.dot(a) + b.dot(b) + c.dot(c) + d.dot(d)) / 2 + w_re * (Cs ** 2 - Sn ** 2) - w_im * 2 * Cs * Sn
ok &= red(u_t - u_claim) == 0
du = red(ddt(u_t))
ok &= du != 0 and red(du.subs({x: 0 for x in list(c) + list(d)})) == 0
emk = Cs - sp.I * Sn
q_X = sp.expand(qf(zt))
q_claim = sp.expand(sum(x ** 2 for x in cmv_) * emk ** 2 + 2 * sum(cmv_[i] * cpv_[i] for i in range(3))
                    + sum(x ** 2 for x in cpv_) * eik ** 2)
ok &= red(sp.re(q_X - q_claim)) == 0 and red(sp.im(q_X - q_claim)) == 0
ex = {a[0]: 1, c[0]: 1}
ex.update({x: 0 for x in list(a[1:]) + list(b) + list(c[1:]) + list(d)})                   # c_- = c_+ = (1,0,0)
q_ex = sp.expand(q_X.subs(ex))
absq2 = red(sp.expand(q_ex * sp.conjugate(q_ex)))
ok &= red(ddt(absq2)) != 0                                                                  # |q(X)| oscillates
# both pure senses are duality rotations of the frame field, with Noether energy 0 and constant u(X), E_0(X)
zm = {x: 0 for x in list(c) + list(d)}                 # c_+ = 0: zero momentum, z = e^{-ik tau} c_-
op = {x: 0 for x in list(a) + list(b)}                 # c_- = 0: z = e^{+ik tau} c_+
ok &= all(red(v.subs(zm)) == 0 for v in pi_) and red(E_noether.subs(zm)) == 0 and red(E_noether.subs(op)) == 0
for sub, ph in ((zm, emk), (op, eik)):
    z0 = [sp.expand(zz.subs(sub).subs({Cs: 1, Sn: 0})) for zz in zt]
    ok &= all(red(sp.expand(zt[i].subs(sub) - ph * z0[i])) == 0 for i in range(3))
    q_s = sp.expand(q_X.subs(sub))
    ok &= red(ddt(u_t.subs(sub))) == 0 and red(ddt(red(sp.expand(q_s * sp.conjugate(q_s))))) == 0
ok &= red(m ** 2 * pi_.dot(pi_).subs(op) - 8 * m ** 2 * ka ** 2 * u_t.subs(op)) == 0       # |p|^2 = 8 m^2 k^2 u(X)
ok &= red(u_t.subs(zm) - (a.dot(a) + b.dot(b)) / 2) == 0
report("d2-4b", ok, "momentum field pi=eta^{-1}p=m alpha_+X: z(pi)=2imk c_+e^{ik tau}=e^{k tau G}pi_0, so u(pi)=|p|^2/2"
       "=2m^2k^2|c_+|^2, E_0(pi)=2m^2k^2|c_+.c_+| and S(pi) are constant; frame field: u(X)=(|c_-|^2+|c_+|^2)/2+"
       "Re(conj(c_-).c_+e^{2ik tau}) is constant iff conj(c_-).c_+=0, and q(X)=c_-.c_-e^{-2ik tau}+2c_-.c_++c_+.c_+e^{2ik tau}, "
       "so E_0(X) oscillates in general; both pure senses z=e^{-+ik tau}c_-+ are duality rotations with Noether energy 0 "
       "and constant u(X), E_0(X): on c_+=0, p=0 while u(X)=|c_-|^2/2>0; on c_-=0, |p|^2=8m^2k^2 u(X)")

# ================================================================================================= d2-5..7 little groups


def stabiliser(gens, X0):
    """Exact basis (coefficient vectors) of {sum c_r A_r : (sum c_r A_r) X0 = 0}."""
    cols = [list(np.asarray(A, dtype=object) @ np.array(X0, dtype=object)) for A in gens]
    rows = [[Fraction(cols[r][i]) for r in range(len(gens))] for i in range(6)]
    ns = nullspace_q(rows)
    return [sum((Fraction(v[r]) * np.asarray(gens[r], dtype=object) for r in range(len(gens))),
                np.zeros((6, 6), dtype=object)) for v in ns], ns


def comm(A, B_):
    return A.dot(B_) - B_.dot(A)


def is_zero(A):
    return all(Fraction(x) == 0 for x in np.asarray(A).reshape(-1))


def span_rank(mats):
    return rank_q([[Fraction(x) for x in np.asarray(M_).reshape(-1)] for M_ in mats]) if mats else 0


def to_sym(M_):
    return sp.Matrix([[sp.Rational(Fraction(x).numerator, Fraction(x).denominator) for x in r] for r in np.asarray(M_)])


def X_of_z(z):
    return [sp.re(zz) for zz in z] + [sp.im(zz) for zz in z]


# brackets of l_G and the action on z: L acts by rotation, K by z -> i v x z
ok = True
for i in range(3):
    for j in range(3):
        k = 3 - i - j if i != j else None
        if i == j:
            continue
        sgn = 1 if (i, j) in ((0, 1), (1, 2), (2, 0)) else -1
        ok &= np.array_equal(comm(L[i], L[j]), sgn * L[k]) and np.array_equal(comm(L[i], K[j]), sgn * K[k])
        ok &= np.array_equal(comm(K[i], K[j]), -sgn * L[k])
zsym = sp.Matrix([Bs[i] + sp.I * Es[i] for i in range(3)])
for i in range(3):
    ei = sp.Matrix([1 if t == i else 0 for t in range(3)])
    LX = sp.Matrix(L[i]) * Xs
    KX = sp.Matrix(K[i]) * Xs
    zL = sp.Matrix([LX[t] + sp.I * LX[3 + t] for t in range(3)])
    zK = sp.Matrix([KX[t] + sp.I * KX[3 + t] for t in range(3)])
    ok &= sp.expand(zL - ei.cross(zsym)) == sp.zeros(3, 1)
    ok &= sp.expand(zK - sp.I * ei.cross(zsym)) == sp.zeros(3, 1)
report("d2-5a", ok, "l_G = span{L_i,K_i}: so(1,3) brackets of thm:emfield; on z=B+iE, L_i acts as z -> e_i x z and "
       "K_i as z -> i e_i x z (delta E = v x B, delta B = -v x E), so omega.L + v.K acts as z -> (omega + i v) x z: "
       "l_G = so(3,C) acting by complex cross products")

# non-null example with a rational square root of q: z=(1+i,1,i), q=2i=(1+i)^2 (not a complex multiple of a real vector)
z_nn = [1 + sp.I, sp.Integer(1), sp.I]
z_nul = [1 + 2 * sp.I, 2 + sp.I, 2 - 2 * sp.I]          # u=(1,2,2), v=(2,1,-2): |u|=|v|, u.v=0
z_nul2 = [sp.Integer(1), sp.I, sp.Integer(0)]
ok = sp.expand(qf(z_nn)) == 2 * sp.I and sp.expand(qf(z_nul)) == 0 and sp.expand(qf(z_nul2)) == 0
X_nn = [Fraction(int(x)) for x in X_of_z(z_nn)]
X_nul = [Fraction(int(x)) for x in X_of_z(z_nul)]
X_nul2 = [Fraction(int(x)) for x in X_of_z(z_nul2)]
Gf = np.asarray(G, dtype=object)

S_nn, _ = stabiliser(lG, X_nn)
ok &= len(S_nn) == 2 and is_zero(comm(S_nn[0], S_nn[1]))
# the stabiliser is a complex line: closed under multiplication by G
ok &= span_rank(S_nn + [Gf.dot(S_nn[0]), Gf.dot(S_nn[1])]) == 2
# compact and boost elements about a common complex axis a = z/sqrt(q), sqrt(q) = 1+i
acx = [sp.nsimplify(zz / (1 + sp.I)) for zz in z_nn]
Acx = sp.Matrix([[0, -acx[2], acx[1]], [acx[2], 0, -acx[0]], [-acx[1], acx[0], 0]])
Mr, Mi = Acx.applyfunc(sp.re), Acx.applyfunc(sp.im)
Rr = sp.BlockMatrix([[Mr, -Mi], [Mi, Mr]]).as_explicit()
Rf = np.array([[Fraction(int(sp.Rational(x).p), int(sp.Rational(x).q)) for x in row] for row in Rr.tolist()], dtype=object)
ok &= span_rank(S_nn + [Rf]) == 2 and span_rank(lG + [Rf]) == 6
Rs = to_sym(Rf)
GR = sp.Matrix(G) * Rs
ok &= Rs ** 3 == -Rs and GR ** 3 == GR and Rs * GR == GR * Rs and Rs != sp.zeros(6, 6)
ok &= Rs * sp.Matrix(X_of_z(z_nn)) == sp.zeros(6, 1)
# null: two-dimensional, abelian, every element nilpotent of order three (null rotations)
for Xn in (X_nul, X_nul2):
    S_n, _ = stabiliser(lG, Xn)
    ok &= len(S_n) == 2 and is_zero(comm(S_n[0], S_n[1]))
    A1, A2 = to_sym(S_n[0]), to_sym(S_n[1])
    s1, s2 = sp.symbols('s1 s2', real=True)
    Nn = s1 * A1 + s2 * A2
    ok &= sp.expand(Nn ** 3) == sp.zeros(6, 6) and sp.expand(Nn ** 2) != sp.zeros(6, 6)
report("d2-5", ok, "little groups of field points in l_G: for q!=0 (z=(1+i,1,i), q=2i) the stabiliser is 2-dim abelian "
       "C.(a x), containing R with R^3=-R (rotation) and GR with (GR)^3=GR (boost) about the common axis a=z/sqrt(q); "
       "for q=0 (two null z) it is 2-dim abelian with N^3=0 (null rotations)")

# with the duality generator adjoined
gensG = lG + [G]
S_nnG, ns_nnG = stabiliser(gensG, X_nn)
ok = len(S_nnG) == 2 and all(v[6] == 0 for v in ns_nnG)   # no G component
for Xn in (X_nul, X_nul2):
    S_nG, ns_nG = stabiliser(gensG, Xn)
    ok &= len(S_nG) == 3 and any(v[6] != 0 for v in ns_nG)
    brs = [comm(S_nG[i], S_nG[j]) for i in range(3) for j in range(i + 1, 3)]
    der = [M_ for M_ in brs if not is_zero(M_)]
    ok &= span_rank(der) == 2                                       # derived algebra 2-dim
    S_n, _ = stabiliser(lG, Xn)
    ok &= span_rank(der + S_n) == 2                                  # = the null rotations, abelian
    # the element with a G component acts on the null rotations as a rotation (eigenvalues +-i): e(2)
    gidx = [i for i, v in enumerate(ns_nG) if v[6] != 0][0]
    W_ = S_nG[gidx]
    Nb = S_n
    coeffs = []
    for Nk in Nb:
        br = comm(W_, Nk)
        sol = nullspace_q([[Fraction(x) for x in np.asarray(Nb[0]).reshape(-1)][t:t + 1]
                           + [Fraction(x) for x in np.asarray(Nb[1]).reshape(-1)][t:t + 1]
                           + [Fraction(-x) for x in np.asarray(br).reshape(-1)][t:t + 1] for t in range(36)])
        assert len(sol) == 1 and sol[0][2] != 0
        coeffs.append([sol[0][0] / sol[0][2], sol[0][1] / sol[0][2]])
    adm = sp.Matrix([[sp.Rational(coeffs[0][0].numerator, coeffs[0][0].denominator),
                      sp.Rational(coeffs[1][0].numerator, coeffs[1][0].denominator)],
                     [sp.Rational(coeffs[0][1].numerator, coeffs[0][1].denominator),
                      sp.Rational(coeffs[1][1].numerator, coeffs[1][1].denominator)]])
    lam = sp.symbols('lam')
    cp = sp.factor((adm - lam * sp.eye(2)).det())
    ok &= all(sp.re(r) == 0 and sp.im(r) != 0 for r in sp.solve(cp, lam))
# for z=(1,i,0) the compensating element is L_3 -+ G: a rotation about the propagation axis plus a duality rotation
Xz = np.array(X_nul2, dtype=object)
ok &= is_zero((np.asarray(L[2], dtype=object) + Gf).dot(Xz)) and not is_zero(np.asarray(L[2], dtype=object).dot(Xz))
# finite form for the circular null field z=(1,i,0): e^{theta G} z = e^{-theta L_3} z
L3s = sp.Matrix(L[2])
X0s = sp.Matrix([1, 0, 0, 0, 1, 0])
eL3 = sp.simplify((-th * L3s).exp())
ok &= sp.simplify(R * X0s - eL3 * X0s) == sp.zeros(6, 1)
# the compensating rotation is about the propagation axis n = n1 x n2/|n1|^2 = B x E/|B x E| of Theorem 13.8 of ES17
for zn in (z_nul, z_nul2):
    n1 = sp.Matrix([sp.re(x) for x in zn])
    n2 = sp.Matrix([sp.im(x) for x in zn])
    nax = n1.cross(n2) / n1.dot(n1)
    ok &= sp.simplify(nax - n1.cross(n2) / sp.sqrt(n1.cross(n2).dot(n1.cross(n2)))) == sp.zeros(3, 1)
    Ln = sum((nax[i] * sp.Matrix(L[i]) for i in range(3)), sp.zeros(6, 6))
    ok &= sp.expand((Ln + sp.Matrix(G)) * sp.Matrix(X_of_z(zn))) == sp.zeros(6, 1)
report("d2-6", ok, "with duality adjoined (l_G + R G): a non-null field point keeps a 2-dim stabiliser with no G "
       "component, a null one acquires a 3-dim stabiliser = e(2), whose compact generator is L_n + G with n=n1 x n2/|n1|^2 "
       "= B x E/|B x E| the propagation axis of Thm 13.8 (for z=(1,i,0): L_3 + G, and e^{theta G}z = e^{-theta L_3}z), "
       "acting on the null rotations with eigenvalues +-i")

# Wigner, inherited: the interior Lorentz algebra on the four-plane E=<e1,e4,e5,e6>
lint_biv = [(a_, b_) for a_, b_ in itertools.combinations((1, 4, 5, 6), 2)]
lint = [half(e[a_] @ e[b_]) for a_, b_ in lint_biv]
e1v = [Fraction(1), 0, 0, 0, 0, 0]
nullv = [Fraction(1), 0, 0, 0, 0, Fraction(1)]   # e1 + e6
S_t, _ = stabiliser(lint, e1v)
ok = len(S_t) == 3 and all(np.array_equal(as_int(M_).T, -as_int(M_)) for M_ in S_t)
ok &= span_rank([comm(S_t[i], S_t[j]) for i in range(3) for j in range(i + 1, 3)]) == 3   # so(3), semisimple
S_l, _ = stabiliser(lint, nullv)
brl = [comm(S_l[i], S_l[j]) for i in range(3) for j in range(i + 1, 3)]
ok &= len(S_l) == 3 and span_rank(brl) == 2
dl = [M_ for M_ in brl if not is_zero(M_)]
ok &= all(is_zero(comm(x_, y_)) for x_ in dl for y_ in dl)
report("d2-7", ok, "Wigner (inherited), in l_int on the interior four-plane <e1,e4,e5,e6>: Stab(e1)=so(3) (3-dim, compact, "
       "semisimple), Stab(e1+e6)=e(2) (3-dim, abelian 2-dim derived algebra); momenta have 3-dim little groups, field "
       "points 2-dim ones")

# ================================================================================================= d2-8 the mass form
biv = [(a_, b_) for a_, b_ in itertools.combinations(range(1, 7), 2)]
Bvs = [e[a_] @ e[b_] for a_, b_ in biv]
ok = all(np.array_equal(X.T @ T + T @ X, np.zeros((8, 8), dtype=np.int64)) for X in Bvs)
n_rot = sum(1 for X in Bvs if np.array_equal(X.T, -X))
n_bst = sum(1 for X in Bvs if np.array_equal(X.T, X))
ok &= n_rot == 6 and n_bst == 9
# all antisymmetric M with X^T M + M X = 0 for the 15 generators
asym_basis = []
for i in range(8):
    for j in range(i + 1, 8):
        A = np.zeros((8, 8), dtype=np.int64)
        A[i, j], A[j, i] = 1, -1
        asym_basis.append(A)
rows = []
for X in Bvs:
    imgs = [vec(X.T @ A + A @ X) for A in asym_basis]
    rows += [[imgs[r][t] for r in range(28)] for t in range(64)]
ns = nullspace_q(rows)
Tcoef = [Fraction(int(T[i, j])) for i in range(8) for j in range(i + 1, 8)]   # coordinates of T in asym_basis
ok &= len(ns) == 1 and rank_q([ns[0], Tcoef]) == 1
# group level: S^T T S = T for the one-parameter subgroups of every generator
for X in Bvs:
    Xs8 = sp.Matrix(X)
    sq = int((X @ X)[0, 0])
    S8 = (sp.cos(th) * sp.eye(8) + sp.sin(th) * Xs8) if sq == -1 else (sp.cosh(th) * sp.eye(8) + sp.sinh(th) * Xs8)
    ok &= sp.simplify(S8.T * sp.Matrix(T) * S8 - sp.Matrix(T)) == sp.zeros(8, 8)
report("d2-8", ok, "Dirac mass form T=e1e2e3: Y^T T + T Y = 0 for all 15 bivectors Y (6 rotations antisymmetric, 9 boosts "
       "symmetric), exp(sY)^T T exp(sY) = T on every one-parameter subgroup, and the antisymmetric solutions of the 15 "
       "conditions form exactly R T (restates Cor 18.4 of ES17): the mass term is invariant under all of Spin(3,3)")

# what the kinetic term keeps of spin(3,3): X^T K^mu + K^mu X = lambda^mu_nu K^nu
gam = {0: e[1], 1: e[4], 2: e[5], 3: e[6]}
Kin = {mu: -(gam[0] @ gam[mu]) for mu in range(4)}
ok = np.array_equal(Kin[0], -I8) and all(np.array_equal(Kin[mu].T, Kin[mu]) for mu in range(4))
unk = 15 + 16      # 15 bivector coefficients, 16 entries lambda^mu_nu
rows = []
for mu in range(4):
    cols = [vec(X.T @ Kin[mu] + Kin[mu] @ X) for X in Bvs]
    lam_cols = []
    for mu2 in range(4):
        for nu in range(4):
            lam_cols.append(vec(-Kin[nu]) if mu2 == mu else [0] * 64)
    allc = cols + lam_cols
    rows += [[allc[r][t] for r in range(unk)] for t in range(64)]
ns = nullspace_q(rows)
xpart = [v[:15] for v in ns]
ok &= len(ns) == 7 and rank_q(xpart) == 7
keep = [biv.index((a_, b_)) for a_, b_ in lint_biv] + [biv.index((2, 3))]
ok &= rank_q(xpart + [[Fraction(1 if r == kk else 0) for r in range(15)] for kk in keep]) == 7
report("d2-8b", ok, "the kinetic kernels K^mu=-gamma^0 gamma^mu are covariant (Y^T K^mu + K^mu Y = Lambda^mu_nu K^nu) "
       "exactly for Y in so(1,3)_int + R J (dim 7 of 15): within Spin(3,3) it is the kinetic term, not the mass, "
       "that reduces the six-dimensional symmetry to the interior one")

# ================================================================================================= d2-9 massless algebra
gl_basis = []
for i in range(8):
    for j in range(8):
        A = np.zeros((8, 8), dtype=np.int64)
        A[i, j] = 1
        gl_basis.append(A)


def solve_invariance(forms):
    rows_ = []
    for F in forms:
        imgs = [vec(X.T @ F + F @ X) for X in gl_basis]
        rows_ += [[imgs[r][t] for r in range(64)] for t in range(64)]
    return nullspace_q(rows_)


ns_ml = solve_invariance([Kin[mu] for mu in range(4)])
gml = [Om, J, e[2] @ Om, e[3] @ Om]
ok = len(ns_ml) == 4
ok &= rank_q([v for v in ns_ml] + [vec(X) for X in gml]) == 4
ok &= all(int(np.trace(gml[i] @ gml[j])) == (-8 if i == j else 0) for i in range(4) for j in range(4))
# su(2) closure of {J, e2 Om, e3 Om}, Om central
su2 = gml[1:]
ok &= all(rank_q([vec(X) for X in su2] + [vec(comm(A, B_))]) == 3 for A in su2 for B_ in su2)
ok &= all(np.array_equal(comm(Om, X), np.zeros((8, 8), dtype=np.int64)) for X in gml)
ns_m = solve_invariance([Kin[mu] for mu in range(4)] + [T])
ok &= len(ns_m) == 1 and rank_q([ns_m[0], vec(J)]) == 1
# orbit of T under g_massless: M -> X^T M + M X
orb = [X.T @ T + T @ X for X in gml]
ok &= rank_q([vec(M_) for M_ in orb]) == 3
dirs = [T @ Om, T @ e[2] @ Om, T @ e[3] @ Om]
ok &= rank_q([vec(M_) for M_ in orb] + [vec(M_) for M_ in dirs]) == 3
ok &= np.array_equal(orb[0], 2 * T @ Om) and np.array_equal(orb[1], np.zeros((8, 8), dtype=np.int64))
ok &= np.array_equal(J @ dirs[0], dirs[0] @ J) and all(np.array_equal(J @ D, -D @ J) for D in dirs[1:])
# the 6-dim kernel space of the interior Lorentz algebra contains the orbit
lint8 = [e[a_] @ e[b_] for a_, b_ in lint_biv]
rows = []
for X in lint8:
    imgs = [vec(X.T @ A + A @ X) for A in asym_basis]
    rows += [[imgs[r][t] for r in range(28)] for t in range(64)]
ns_K = nullspace_q(rows)
Kc = [sum((int(v[r]) * asym_basis[r] for r in range(28)), np.zeros((8, 8), dtype=np.int64)) for v in ns_K]
ok &= len(Kc) == 6 and all(Fraction(x).denominator == 1 for v in ns_K for x in v)
ok &= rank_q([vec(M_) for M_ in Kc] + [vec(M_) for M_ in dirs + [T]]) == 6
report("d2-9", ok, "massless algebra: Y^T K^mu + K^mu Y = 0 (mu=0..3) over all of End(R^8) gives exactly span{Om, J, "
       "e2 Om, e3 Om} = u(1)_Om + su(2), trace form -8*1, and adding the mass form leaves R J (restates Prop 5.5 of ES17); "
       "the orbit of T has dim 3 (Thm 18.5(a)), spanned by T Om (J-linear, pseudoscalar) and T e2 Om, T e3 Om "
       "(J-antilinear, Majorana type), inside the 6-dim space of interior Lorentz-invariant kernels")

# ================================================================================================= d2-10 chiral mass angle
p1, p2, p3, ms_, mp_ = sp.symbols('p1 p2 p3 m_s m_p', real=True)
ps = [p1, p2, p3]
gS = {k: sp.Matrix(v) for k, v in gam.items()}
JS, OmS, TS = sp.Matrix(J), sp.Matrix(Om), sp.Matrix(T)
Mk = TS * (ms_ * sp.eye(8) + mp_ * OmS)
Hs = sum((gS[0] * gS[a_ + 1] * ps[a_] for a_ in range(3)), sp.zeros(8, 8)) + gS[0] * (ms_ * sp.eye(8) + mp_ * OmS)
KinS = {mu: sp.Matrix(Kin[mu]) for mu in range(4)}
ok = Mk.T == -Mk and Hs.T == Hs and JS * Hs == Hs * JS and JS * Mk == Mk * JS
ok &= all(sp.expand(sp.Matrix(X).T * Mk + Mk * sp.Matrix(X)) == sp.zeros(8, 8) for X in lint8)
# Grassmann Lagrangian (1/2)Psi^T (K^mu d_mu - M) Psi: K^mu d_mu Psi = M Psi; with d_a Psi = J p_a Psi
lhs = sum((KinS[a_ + 1] * JS * ps[a_] for a_ in range(3)), sp.zeros(8, 8)) - Mk     # = d_0 Psi operator (K^0=-1)
ok &= sp.expand(lhs + JS * Hs) == sp.zeros(8, 8)
ok &= sp.expand(Hs * Hs - (p1 ** 2 + p2 ** 2 + p3 ** 2 + ms_ ** 2 + mp_ ** 2) * sp.eye(8)) == sp.zeros(8, 8)
ok &= sp.expand((JS * Hs) ** 2 + (p1 ** 2 + p2 ** 2 + p3 ** 2 + ms_ ** 2 + mp_ ** 2) * sp.eye(8)) == sp.zeros(8, 8)
# axial rotation e^{alpha Om}: kinetic kernels kept, M -> M e^{2 alpha Om}
Sax = sp.cos(al) * sp.eye(8) + sp.sin(al) * OmS
ok &= all(sp.simplify(Sax.T * KinS[mu] * Sax - KinS[mu]) == sp.zeros(8, 8) for mu in range(4))
ok &= sp.simplify(Sax.T * Mk * Sax - Mk * (sp.cos(2 * al) * sp.eye(8) + sp.sin(2 * al) * OmS)) == sp.zeros(8, 8)
# stabiliser of a nonzero mass in U(1)_Om: e^{2 alpha Om} = 1 iff alpha in pi Z
ok &= sp.solveset(sp.Eq(sp.cos(2 * al), 1), al, domain=sp.Interval(0, 2 * sp.pi)) == sp.FiniteSet(0, sp.pi, 2 * sp.pi)
report("d2-10", ok, "chiral mass angle: the Grassmann field equation K^mu d_mu Psi = M Psi with M=T(m_s+m_p Om) is "
       "Psi_dot = -J H Psi, H = gamma^0 gamma^a p_a + gamma^0(m_s+m_p Om) symmetric and J-linear, "
       "H^2 = (|p|^2+m_s^2+m_p^2) 1; e^{xi Om} keeps K^mu and sends M -> M e^{2 xi Om}, stabiliser {+-1}")

# ================================================================================================= d2-11 rest flow
lam_ = sp.symbols('lambda', real=True)
H0 = sp.Matrix(e[1])
ok = np.array_equal(J @ e[1], T) and np.array_equal(-T @ J, e[1])          # J H(0)/m = T ; gamma^0 = -T J
Pp = (sp.eye(8) + H0) / 2
Pm = (sp.eye(8) - H0) / 2
ok &= TS * Pp == JS * Pp and TS * Pm == -JS * Pm
flow = sp.cos(m * lam_) * sp.eye(8) - sp.sin(m * lam_) * TS                 # exp(-m lambda T)
ok &= sp.simplify(flow.diff(lam_) + m * TS * flow) == sp.zeros(8, 8)         # Psi_dot = -J H(0) Psi = -m T Psi
rotJ = lambda s: sp.cos(m * lam_) * sp.eye(8) - s * sp.sin(m * lam_) * JS   # exp(-s m lambda J)
ok &= sp.simplify(flow * Pp - rotJ(1) * Pp) == sp.zeros(8, 8) and sp.simplify(flow * Pm - rotJ(-1) * Pm) == sp.zeros(8, 8)
ok &= int(np.trace(e[1])) == 0 and sp.Matrix(e[1]).rank() == 8                # e1 = +-1 eigenspaces, 4 + 4
report("d2-11", ok, "rest flow: J H(0) = m T, so Psi(lambda)=exp(-m lambda T)Psi_0 with T=e1e2e3; T = +J on e1=+1 "
       "and -J on e1=-1, hence Psi = exp(-+ m J lambda)Psi_0 there; gamma^0 = -T J is the ratio of the mass rotation "
       "to the charge rotation (restored units: angular frequency m c^2/hbar)")

# ================================================================================================= d2-12/13/14 senses
Trev = mm(e[2], e[4], e[5], e[6])          # T = C gamma^1 gamma^2 gamma^3 (thm:PT)
P_d = e[1]
P_dJ = e[1] @ J
ok = np.array_equal(C @ e[1], -e[1] @ C) and np.array_equal(C @ J, -J @ C) and np.array_equal(C @ T, T @ C)
ok &= np.array_equal(C @ C, I8) and np.array_equal(C.T, C)
H0i = e[1]


def energy_sign(O):
    """+1 if O^T H(0) O = H(0) (rest energy kept), -1 if = -H(0)."""
    W_ = O.T @ H0i @ O
    return 1 if np.array_equal(W_, H0i) else (-1 if np.array_equal(W_, -H0i) else 0)


sig_d = {'C': energy_sign(C), 'P': energy_sign(P_d), 'T': energy_sign(Trev)}
ok &= sig_d == {'C': -1, 'P': 1, 'T': 1} and energy_sign(P_dJ) == 1
phi_ = sp.symbols('phi', real=True)
Cphi = sp.cos(phi_) * sp.Matrix(e[2]) + sp.sin(phi_) * sp.Matrix(e[3])          # any unit vector of U
ok &= sp.simplify(Cphi * Cphi - sp.eye(8)) == sp.zeros(8, 8)
ok &= sp.simplify(Cphi.T * H0 * Cphi + H0) == sp.zeros(8, 8) and sp.simplify(Cphi * JS + JS * Cphi) == sp.zeros(8, 8)
ok &= sp.simplify(Cphi * TS - TS * Cphi) == sp.zeros(8, 8)
# C maps solutions to solutions (commutes with the rest generator T) and maps e1=+1 to e1=-1
ok &= sp.Matrix(C) * Pp * sp.Matrix(C) == Pm
# T_rev is antilinear and reverses lambda: it anticommutes with the generator T and keeps e1 eigenspaces
ok &= np.array_equal(Trev @ T, -T @ Trev) and np.array_equal(Trev @ J, -J @ Trev) and np.array_equal(Trev @ e[1], e[1] @ Trev)
# composites
comp_d = {'CP': energy_sign(C @ P_d), 'CT': energy_sign(C @ Trev), 'PT': energy_sign(P_d @ Trev),
          'CPT': energy_sign(C @ P_d @ Trev)}
ok &= comp_d == {'CP': -1, 'CT': -1, 'PT': 1, 'CPT': -1}
# C is a symmetry of the Grassmann Lagrangian (it keeps T and every K^mu, Thm 5.1) but reverses the sign of the commuting
# first-order action (1/2)Psi^T J Psi_dot - (1/2)Psi^T H(p) Psi: C^T J C = -J and C^T H(p) C = -H(-p)
Hneg = Hs.subs({p1: -p1, p2: -p2, p3: -p3}, simultaneous=True)
ok &= sp.simplify(Cphi.T * JS * Cphi + JS) == sp.zeros(8, 8)
ok &= sp.simplify(Cphi.T * Hs * Cphi + Hneg) == sp.zeros(8, 8)
ok &= sp.simplify(Cphi.T * TS * Cphi - TS) == sp.zeros(8, 8)
ok &= all(sp.simplify(Cphi.T * KinS[mu] * Cphi - KinS[mu]) == sp.zeros(8, 8) for mu in range(4))
report("d2-12", ok, "Dirac at rest: every unit vector C=cos(sigma)e2+sin(sigma)e3 of U commutes with the rest generator T, "
       "anticommutes with J and with e1, so it maps the e1=+1 rotation exp(-mJ lambda) onto the e1=-1 rotation "
       "exp(+mJ lambda) and reverses the rest energy (C^T H(0) C = -H(0)); C keeps T and every K^mu (a symmetry of the "
       "Grassmann Lagrangian) but C^T J C=-J, C^T H(p) C=-H(-p) reverse the sign of the commuting first-order action; "
       "P=gamma^0 (or gamma^0 J) and T_rev=e2e4e5e6 keep the rest energy; CP, CT, CPT reverse it, PT keeps it")

# frame binding: P = eta (E -> -E), C = -1, T: X(tau) -> -eta X(-tau)
Pf, Cf, Tf = eta6, -I6, -eta6
ok = np.array_equal(Pf @ G, -G @ Pf) and np.array_equal(Cf @ G, G @ Cf) and np.array_equal(Tf @ G, -G @ Tf)
Xf = sp.Matrix([sp.Function(f'x{i}')(tau) for i in range(6)])
Lk = lambda X, k_: (m / 2 * ((X.diff(tau) + k_ * Gs * X).T * etas * (X.diff(tau) + k_ * Gs * X)))[0]
ok &= sp.simplify(Lk(sp.Matrix(Pf) * Xf, ka) - Lk(Xf, -ka)) == 0
ok &= sp.simplify(Lk(sp.Matrix(Cf) * Xf, ka) - Lk(Xf, ka)) == 0
Xrev = (sp.Matrix(Tf) * Xf).subs(tau, -tau)
ok &= sp.simplify(Lk(Xrev, ka) - Lk(Xf, ka).subs(tau, -tau)) == 0
# L_kappa - L_{-kappa} is the total derivative -2 m kappa d/dtau (E.B)
EB = sum(Xf[i] * Xf[3 + i] for i in range(3))
ok &= sp.simplify(Lk(Xf, ka) - Lk(Xf, -ka) + 2 * m * ka * EB.diff(tau)) == 0


def read_cpm(Xvec):
    """Read (c_-, c_+) off X(tau) = (B,E) written in Cs, Sn (as in d2-4)."""
    Bc = [sp.expand(Xvec[i]).coeff(Cs) for i in range(3)]
    Bs_ = [sp.expand(Xvec[i]).coeff(Sn) for i in range(3)]
    Ec = [sp.expand(Xvec[3 + i]).coeff(Cs) for i in range(3)]
    Es_ = [sp.expand(Xvec[3 + i]).coeff(Sn) for i in range(3)]
    # B = (a+c)Cs + (b-d)Sn, E = (b+d)Cs + (c-a)Sn
    aa = [(Bc[i] - Es_[i]) / 2 for i in range(3)]
    cc = [(Bc[i] + Es_[i]) / 2 for i in range(3)]
    bb = [(Ec[i] + Bs_[i]) / 2 for i in range(3)]
    dd = [(Ec[i] - Bs_[i]) / 2 for i in range(3)]
    return [sp.expand(aa[i] + sp.I * bb[i]) for i in range(3)], [sp.expand(cc[i] + sp.I * dd[i]) for i in range(3)]


cm0, cp0 = read_cpm(Xt)
cmv = [a[i] + sp.I * b[i] for i in range(3)]
cpv = [c[i] + sp.I * d[i] for i in range(3)]
ok &= all(sp.expand(cm0[i] - cmv[i]) == 0 and sp.expand(cp0[i] - cpv[i]) == 0 for i in range(3))
cmP, cpP = read_cpm(sp.Matrix(Pf) * Xt)
ok &= all(sp.expand(cmP[i] - sp.conjugate(cpv[i])) == 0 and sp.expand(cpP[i] - sp.conjugate(cmv[i])) == 0 for i in range(3))
cmC, cpC = read_cpm(sp.Matrix(Cf) * Xt)
ok &= all(sp.expand(cmC[i] + cmv[i]) == 0 and sp.expand(cpC[i] + cpv[i]) == 0 for i in range(3))
cmT, cpT = read_cpm((sp.Matrix(Tf) * Xt).subs(Sn, -Sn))
ok &= all(sp.expand(cmT[i] + sp.conjugate(cmv[i])) == 0 and sp.expand(cpT[i] + sp.conjugate(cpv[i])) == 0 for i in range(3))
# convention: T_f bundles the reversal of tau with the field map of C_f P_f, (B,E) -> (-B,E)
ok &= np.array_equal(Tf, Cf @ Pf)
report("d2-13", ok, "frame binding: P_f G P_f^-1 = -G, L_kappa(P_f X) = L_{-kappa}(X), L_kappa - L_{-kappa} = -2 m kappa "
       "d(E.B)/dtau; C_f (X->-X) and T_f (X(tau)->-eta X(-tau), the field map of C_f P_f combined with tau->-tau) leave "
       "L_kappa invariant; on z=c_-e^{-ik tau}+c_+e^{ik tau}: P_f: (c_-,c_+)->(conj c_+, conj c_-) exchanges the senses, "
       "C_f: -> (-c_-,-c_+), T_f: -> (-conj c_-,-conj c_+) keep them")

# contrast: which operations reverse the sense
sig_f = {'C': 1, 'P': -1, 'T': 1}
flip = lambda sig, word: int(np.prod([sig[ch] for ch in word]))
words = ['C', 'P', 'T', 'CP', 'CT', 'PT', 'CPT']
dirac_rev = [w for w in words if (sig_d[w] if w in sig_d else comp_d[w]) == -1]
frame_rev = [w for w in words if flip(sig_f, w) == -1]
ok = all(flip(sig_d, w) == (sig_d[w] if w in sig_d else comp_d[w]) for w in words)
ok &= dirac_rev == ['C', 'CP', 'CT', 'CPT'] and frame_rev == ['P', 'CP', 'PT', 'CPT']
report("d2-14", ok, f"contrast: the sense of the Dirac rest rotation is reversed by {dirac_rev}, that of the frame "
       f"binding by {frame_rev}; C-odd versus P-odd, and both CPT-odd")

# ================================================================================================= d2-15/16 Fock module
cands = [w for w in itertools.product('IXZE', repeat=4) if w.count('E') % 2 == 1]


def anti(w1, w2):
    return sum(1 for x_, y_ in zip(w1, w2) if x_ != 'I' and y_ != 'I' and x_ != y_) % 2 == 1


def search(chosen, start):
    if len(chosen) == 8:
        return chosen
    for i in range(start, len(cands)):
        if all(anti(cands[i], w) for w in chosen):
            r_ = search(chosen + [cands[i]], i + 1)
            if r_:
                return r_
    return None


words8 = search([], 0)
P2 = {'I': I2, 'X': sx, 'Z': sz, 'E': ep}
theta = [kron(*[P2[ch] for ch in w]) for w in words8]
I16 = np.eye(16, dtype=np.int64)
ok = all(np.array_equal(t.T, -t) and np.array_equal(t @ t, -I16) for t in theta)
ok &= all(np.array_equal(theta[i] @ theta[j] + theta[j] @ theta[i], np.zeros((16, 16), dtype=np.int64))
          for i in range(8) for j in range(i + 1, 8))
monos = []
for S_ in itertools.product((0, 1), repeat=8):
    M_ = I16.copy()
    for i, s in enumerate(S_):
        if s:
            M_ = M_ @ theta[i]
    monos.append(vec(M_))
ok &= rank_q(monos) == 256
report("d2-15a", ok, "eight 16x16 integer matrices theta_i (tensor words in the 2x2 matrices of eq. (1) of ES17), "
       "antisymmetric, theta_i^2=-1, pairwise anticommuting; the 256 Clifford monomials have rank 256, so their "
       "commutant is R*1 (Thm 5.6 of ES17)")


def rho4(A):
    """4*rho(A) = -sum A_ij theta_i theta_j, integer for integer A."""
    out = np.zeros((16, 16), dtype=np.int64)
    for i in range(8):
        for j in range(8):
            if A[i, j]:
                out -= int(A[i, j]) * (theta[i] @ theta[j])
    return out


so8 = asym_basis
ok = True
for A in so8:
    for B_ in so8:
        lhs_ = rho4(A) @ rho4(B_) - rho4(B_) @ rho4(A)        # 16 [rho A, rho B]
        rhs_ = 4 * rho4(A @ B_ - B_ @ A)                      # 16 rho([A,B])
        ok &= np.array_equal(lhs_, rhs_)
    tk = [rho4(A) @ theta[k] - theta[k] @ rho4(A) for k in range(8)]
    ok &= all(np.array_equal(tk[k], 4 * sum(int(A[j, k]) * theta[j] for j in range(8))) for k in range(8))
report("d2-15b", ok, "rho(A) = -(1/4) sum A_ij theta_i theta_j is a Lie homomorphism so(8) -> so(16) with "
       "[rho(A), theta_k] = sum_j A_jk theta_j")

xx = sp.symbols('t')


def Hp(pvec, mass):
    return sum((pvec[a_] * (e[1] @ e[a_ + 4]) for a_ in range(3)), np.zeros((8, 8), dtype=np.int64)) + mass * e[1]


cases = [((0, 0, 0), 1, 1), ((0, 0, 3), 4, 5), ((1, 2, 2), 0, 3), ((2, 3, 6), 0, 7)]
ok = True
chiJ = sp.Matrix(rho4(J)).applyfunc(lambda v: sp.Rational(v, 2)).charpoly(xx).as_expr()   # 2 rho(J)
ok &= sp.expand(chiJ - xx ** 6 * (xx ** 2 + 4) ** 4 * (xx ** 2 + 16)) == 0
for pv, mass, En in cases:
    JH = J @ Hp(pv, mass)
    chi = sp.Matrix(rho4(JH)).applyfunc(lambda v: sp.Rational(v, 2)).charpoly(xx).as_expr()
    ok &= sp.expand(chi - xx ** 6 * (xx ** 2 + 4 * En ** 2) ** 4 * (xx ** 2 + 16 * En ** 2)) == 0


def joint_dim(R1_4, q_, R2_4, h_):
    """complex dim of {v: -i rho(J) v = q v, -i rho(JH) v = h v}; R?_4 = 4 rho(.)"""
    Z = np.zeros((16, 16), dtype=np.int64)
    blk = lambda R4, lamb: np.block([[R4, 4 * lamb * I16], [-4 * lamb * I16, R4]])
    Msys = np.vstack([blk(R1_4, q_), blk(R2_4, h_)])
    return (32 - rank_q(Msys.tolist())) // 2


table_expected = {(-2, 0): 1, (-1, 1): 2, (-1, -1): 2, (0, 2): 1, (0, -2): 1, (0, 0): 4, (1, 1): 2, (1, -1): 2, (2, 0): 1}
for pv, mass, En in cases[:2]:
    R1, R2 = rho4(J), rho4(J @ Hp(pv, mass))
    ok &= np.array_equal(R1 @ R2, R2 @ R1)
    tab = {}
    for hk in range(-2, 3):
        for qk in range(-2, 3):
            dd_ = joint_dim(R1, qk, R2, hk * En)
            if dd_:
                tab[(hk, qk)] = dd_
    ok &= tab == table_expected
    mult = [sum(v for (hk, qk), v in tab.items() if hk == lev) for lev in range(-2, 3)]
    ok &= mult == [1, 4, 6, 4, 1]
report("d2-16", ok, "Fock space of one momentum (restates Thm 5.7 of ES17): chi_{2rho(J)} = t^6(t^2+4)^4(t^2+16), "
       "chi_{2rho(JH)} = t^6(t^2+4 eps^2)^4(t^2+16 eps^2), eps=sqrt(|p|^2+m^2), at (p,m,eps) = (0,1,1),(3e3,4,5),"
       "((1,2,2),0,3),((2,3,6),0,7); joint (H,Q) table (exact kernels at the first two); multiplicities 1,4,6,4,1; "
       "unique charge-0 vacuum at H=-2 eps, i.e. -2m at p=0, minus half the rest energies of the four modes")

# lift of C = e2 to the module as a product of four integral reflections
ns_C = nullspace_q((C + I8).tolist())
vs = [np.array([int(x * 1) for x in v], dtype=np.int64) for v in ns_C]
ok = len(vs) == 4 and all(int(vs[i] @ vs[j]) == 0 for i in range(4) for j in range(i + 1, 4))
nrm = [int(v @ v) for v in vs]
thv = [sum(int(v[i]) * theta[i] for i in range(8)) for v in vs]
U4 = thv[0] @ thv[1] @ thv[2] @ thv[3]
scale = int(np.prod(nrm))      # U = U4 / sqrt(prod |v|^2)
ok &= int(round(np.sqrt(scale))) ** 2 == scale
sU = int(round(np.sqrt(scale)))
ok &= np.array_equal(U4.T @ U4, scale * I16) and np.array_equal(U4 @ U4, scale * I16)
# U theta_k U^-1 = sum_j C_jk theta_j
ok &= all(np.array_equal(U4 @ theta[k] @ U4.T, scale * sum(int(C[j, k]) * theta[j] for j in range(8))) for k in range(8))
ok &= np.array_equal(U4 @ rho4(J) @ U4.T, -scale * rho4(J))
for pv, mass, En in cases[1:]:
    mp_v = tuple(-x for x in pv)
    ok &= np.array_equal(U4 @ rho4(J @ Hp(pv, mass)) @ U4.T, scale * rho4(J @ Hp(mp_v, mass)))
# +1 eigenspace of U (real dim 8) meets ker rho(J) in a 3-dim subspace
Uplus = nullspace_q((U4 - sU * I16).tolist())
kerQ = nullspace_q(rho4(J).tolist())
inter = 16 - rank_q((np.vstack([U4 - sU * I16, rho4(J)])).tolist())
ok &= len(Uplus) == 8 and len(kerQ) == 6 and inter == 3
report("d2-17", ok, "C=e2 lifts to the module as C^ = theta(v1)...theta(v4)/4, |v_k|^2=2 (four integral reflections, "
       "C^2=1, orthogonal); C^ theta_k C^-1 = sum_j C_jk theta_j, C^ Q C^-1 = -Q, C^ H(p) C^-1 = H(-p); the +1 "
       "eigenspace of C^ (real dim 8) meets ker Q in real dim 3 (restates Prop 5.8 of ES17)")

# ================================================================================================= d2-18/19 frame vacuum
hb = sp.symbols('hbar', positive=True)
mpos, kpos = sp.symbols('m kappa', positive=True)
Xd6 = sp.Matrix(sp.symbols('v1:7', real=True))
X6 = sp.Matrix(sp.symbols('y1:7', real=True))
E_cl = mpos / 2 * ((Xd6.T * etas * Xd6)[0] + kpos ** 2 * (X6.T * etas * X6)[0])
E_no = mpos / 2 * (Xd6.T * etas * Xd6)[0] - mpos * kpos ** 2 / 2 * ((Gs * X6).T * etas * (Gs * X6))[0]
ok = sp.expand(E_cl - E_no) == 0
ok &= sp.expand(E_cl - sum(int(eta6[i, i]) * mpos / 2 * (Xd6[i] ** 2 + kpos ** 2 * X6[i] ** 2) for i in range(6))) == 0
# canonical Hamiltonian of a spatial mode: L = -(m/2)(ydot^2 - k^2 y^2), p_y = -m ydot, H = -(p^2/2m + m k^2 y^2/2)
yv, ydv, pyv = sp.symbols('y ydot p_y', real=True)
Ly = -mpos / 2 * (ydv ** 2 - kpos ** 2 * yv ** 2)
py_ = sp.diff(Ly, ydv)
Hy = sp.expand((py_ * ydv - Ly).subs(ydv, -pyv / mpos))
ok &= sp.simplify(Hy + pyv ** 2 / (2 * mpos) + mpos * kpos ** 2 * yv ** 2 / 2) == 0 and sp.simplify(py_ + mpos * ydv) == 0
# truncated ladder operators: H_+ = P^2/2m + m k^2 X^2/2 = hbar k (n + 1/2) on the levels n < N-1
N = 10
aop = sp.zeros(N, N)
for n in range(1, N):
    aop[n - 1, n] = sp.sqrt(n)
ad_ = aop.T
Xq = sp.sqrt(hb / (2 * mpos * kpos)) * (aop + ad_)
Pq = sp.I * sp.sqrt(hb * mpos * kpos / 2) * (ad_ - aop)
Hplus = sp.simplify(Pq * Pq / (2 * mpos) + mpos * kpos ** 2 * Xq * Xq / 2)
ok &= all(sp.simplify(Hplus[i, j] - (hb * kpos * (i + sp.Rational(1, 2)) if i == j else 0)) == 0
          for i in range(N - 1) for j in range(N - 1))
comm_ok = sp.simplify((Xq * Pq - Pq * Xq)[:N - 1, :N - 1] - sp.I * hb * sp.eye(N - 1)) == sp.zeros(N - 1, N - 1)
ok &= comm_ok
Hminus = -Hplus
ok &= all(sp.simplify(Hminus[i, i] + hb * kpos * (i + sp.Rational(1, 2))) == 0 for i in range(N - 1))
# zero-point energies of all 20 coordinate three-planes and the full space
zp = {}
for W in itertools.combinations(range(1, 7), 3):
    pW = sum(1 for x in W if x <= 3)
    zp[W] = sp.Rational(pW - (3 - pW), 2)
named = {(1, 2, 3): sp.Rational(3, 2), (4, 5, 6): -sp.Rational(3, 2), (1, 2, 4): sp.Rational(1, 2),
         (2, 3, 4): sp.Rational(1, 2), (1, 4, 5): -sp.Rational(1, 2), (2, 4, 5): -sp.Rational(1, 2),
         (1, 5, 6): -sp.Rational(1, 2)}
ok &= all(zp[W] == v for W, v in named.items())
ok &= sum(int(eta6[i, i]) for i in range(6)) == 0
report("d2-18", ok, "frame vacuum (restates Thm 18.6 of ES17): energy (m/2)(eta(Xd,Xd)+k^2 eta(X,X)) = Noether energy = "
       "sum_a eta_aa (m/2)(xd_a^2+k^2x_a^2); a spatial mode x has H=-(p_x^2/2m+mk^2x^2/2); positive-norm quantisation gives "
       "+-hbar k(j+1/2) per temporal/spatial mode, so on k temporal and l spatial coordinate directions the zero-point "
       "energy is (hbar kappa/2)(k-l): 0 on R^{3,3}, +-3/2 on the temporal/spatial triples, +-1/2 on (2,1)/(1,2) planes")

# spectrum of the full frame: hbar k (sum n_T - sum n_S)
ok = True
deg0 = []
for Ncut in (2, 3, 4):
    levels = {}
    for ns6 in itertools.product(range(Ncut + 1), repeat=6):
        lev = sum(ns6[:3]) - sum(ns6[3:])
        levels[lev] = levels.get(lev, 0) + 1
    ok &= sorted(levels) == list(range(-3 * Ncut, 3 * Ncut + 1))
    deg0.append(levels[0])
ok &= deg0[0] < deg0[1] < deg0[2]
report("d2-19", ok, f"no ground state (obstructed under positive norm): the frame spectrum is hbar*kappa*Z, every level "
       f"reached from both sides (adding a spatial quantum lowers the energy by hbar kappa), and the zero level "
       f"degeneracy grows without bound with the cutoff ({deg0} at n_max = 2,3,4)")

# ================================================================================================= d2-20 Majorana phase
# kernels supported on Gamma=+1: (1-Gamma) M = 0 = M (1-Gamma)
rows = []
for t in range(64):
    rows.append([vec((I8 - Gam) @ M_)[t] for M_ in Kc])
for t in range(64):
    rows.append([vec(M_ @ (I8 - Gam))[t] for M_ in Kc])
ns_maj = nullspace_q(rows)
ok = len(ns_maj) == 2
Mj = [sum((ns_maj[s][r] * np.asarray(Kc[r], dtype=object) for r in range(6)), np.zeros((8, 8), dtype=object)) for s in range(2)]
Mj = [np.array([[Fraction(x) for x in row] for row in M_], dtype=object) for M_ in Mj]
Jo, Go, Omo = (np.asarray(X, dtype=object) for X in (J, Gam, Om))
# complex line: J M lies in the span
ok &= span_rank(Mj + [Jo.dot(Mj[0])]) == 2
Mmaj = Mj[0]
ok &= all(is_zero(Jo.dot(M_) + M_.dot(Jo)) for M_ in Mj)
MtM = Mmaj.T.dot(Mmaj)
Pplus = (np.asarray(I8, dtype=object) + Go) * Fraction(1, 2)
cfac = MtM[np.nonzero(np.asarray(Pplus.astype(float)))[0][0], np.nonzero(np.asarray(Pplus.astype(float)))[1][0]] / \
    Pplus[np.nonzero(np.asarray(Pplus.astype(float)))[0][0], np.nonzero(np.asarray(Pplus.astype(float)))[1][0]]
ok &= is_zero(MtM - cfac * Pplus) and cfac != 0
Nn_ = sum(MtM[i, i] for i in range(8))
Qn_ = sum((MtM.dot(MtM))[i, i] for i in range(8))
ok &= Fraction(Qn_) / Fraction(Nn_) ** 2 == Fraction(1, 4)
ok &= rank_q([[Fraction(x) for x in row] for row in Mmaj]) == 4
# stabiliser in g_I = span{Om, J, e2 Om, e3 Om} is R(J - J Gamma) = R(J + Om)
gI = [np.asarray(X, dtype=object) for X in gml]
rows = [[Fraction(x) for x in np.asarray(comm(X, Mmaj)).reshape(-1)] for X in gI]
ns_st = nullspace_q([list(r) for r in zip(*rows)])
ok &= len(ns_st) == 1
stab = sum((ns_st[0][r] * gI[r] for r in range(4)), np.zeros((8, 8), dtype=object))
ok &= span_rank([stab, Jo - Jo.dot(Go)]) == 1 and np.array_equal(J @ Gam, -Om)
ok &= rank_q([[Fraction(x) for x in np.asarray(comm(X, Mmaj)).reshape(-1)] for X in gI]) == 3
# Dirac phase keeps J and breaks J - J Gamma; Majorana phase the reverse
ok &= is_zero(comm(Jo, np.asarray(T, dtype=object))) and not is_zero(comm(Jo - Jo.dot(Go), np.asarray(T, dtype=object)))
ok &= not is_zero(comm(Jo, Mmaj))
report("d2-20", ok, "Majorana phase (restates Thm 18.5(b) of ES17): interior Lorentz kernels supported on Gamma=+1 form a "
       "complex line; each anticommutes with J, has rank 4, M^T M = c(1+Gamma)/2, Tr(M^TM)^2/(Tr M^TM)^2=1/4; its "
       "stabiliser in g_massless is R(J - J Gamma) = R(J + Om), orbit dim 3; the Dirac mass T keeps J and moves J - J Gamma, "
       "the Majorana mass the reverse")

# ================================================================================================= d2-21 three constituents
omega1 = vol.astype(float)
I8f = np.eye(8)


def Ebig(A):
    k_ = (A - 1) // 6
    a_ = (A - 1) % 6 + 1
    mats = [omega1] * k_ + [e[a_].astype(float)] + [I8f] * (2 - k_)
    return kron(*mats)


EB_ = {A: Ebig(A) for A in range(1, 19)}
I512 = np.eye(512)
eta18 = {A: (1 if ((A - 1) % 6) < 3 else -1) for A in range(1, 19)}
ok = all(np.array_equal(EB_[A] @ EB_[B_] + EB_[B_] @ EB_[A], (2 * eta18[A] * I512 if A == B_ else 0 * I512))
         for A in range(1, 19) for B_ in range(A, 19))


def prod(idx):
    r_ = I512
    for A in idx:
        r_ = r_ @ EB_[A]
    return r_


Jset = [A for A in range(1, 19) if A not in (1, 4, 5, 6)]
J3 = prod(Jset)
G3 = J3 @ prod([1, 4, 5, 6])
chiW = prod([7, 13, 14, 15])
chiC = prod([10, 11, 12, 16, 17, 18])
qA = prod([2, 3])
ok &= np.array_equal(J3 @ J3, -I512) and np.array_equal(G3 @ G3, I512)
Pst = (I512 + G3 @ chiW) @ (I512 - J3 @ chiC) @ (I512 - J3 @ qA) / 8
ok &= np.allclose(Pst @ Pst, Pst, atol=1e-12) and np.allclose(Pst, Pst.T) and abs(np.trace(Pst) - 64) < 1e-9
w1, w2 = EB_[7], EB_[13]
tau3 = -0.5 * (w1 @ w2)
tR3 = 0.5 * (I512 + chiW) @ tau3
Pnu = 0.5 * (I512 + G3) @ (0.5 * (I512 + chiW)) @ (0.5 * (I512 + J3 @ w1 @ w2))
for aidx in (10, 11, 12):
    Pnu = Pnu @ (0.5 * (I512 - J3 @ EB_[aidx] @ EB_[aidx + 6]))
Bl = prod([1, 3, 9, 13, 14, 16, 17, 18])
Phi0 = Pnu @ Pst @ Bl @ Pst @ Pnu


def dyadic_exact(M_, k_=12):
    S_ = M_ * (2 ** k_)
    return np.allclose(S_, np.round(S_), atol=1e-9, rtol=0)


ok &= dyadic_exact(Phi0) and np.linalg.norm(Phi0) > 1e-6
ok &= np.allclose(Phi0.T, -Phi0) and np.allclose(J3 @ Phi0, -Phi0 @ J3)
PtP = Phi0.T @ Phi0
ok &= np.allclose(PtP, Pnu @ Pst) and abs(np.trace(PtP) - 4) < 1e-9 and np.allclose(Pnu @ Pst, Pst @ Pnu)
ok &= abs(np.trace(Pnu) - 8) < 1e-9                                   # rank 8 on R^512 (erratum 10)
ok &= np.allclose(tR3 @ Phi0 - Phi0 @ tR3, J3 @ Phi0)
report("d2-21a", ok, "three constituents rebuilt (Cl(9,9) on R^512, Appendix A of ES17): P_* of rank 64 and Phi_0 = P_nu "
       "P_* gamma^0 E3E9E13E14E16E17E18 P_* P_nu: Phi_0 antisymmetric, {J,Phi_0}=0, Phi_0^T Phi_0 = P_nu P_* (the rank-4 nu_R "
       "projector of the block; P_nu on R^512 has rank 8), [t^R_3, Phi_0] = J Phi_0 with the su(2) generators normalised as in "
       "Section 8 (tau_3=-(1/2)E7E13; restates Thm 19.2(v))")

cJ = J3 @ Phi0 - Phi0 @ J3
cJG = (J3 @ G3) @ Phi0 - Phi0 @ (J3 @ G3)
ctR = tR3 @ Phi0 - Phi0 @ tR3
ok = np.allclose(cJ, cJG, atol=1e-12) and np.allclose(cJ, 2 * ctR, atol=1e-12) and np.linalg.norm(cJ) > 1e-6
Wset, Cset = [7, 13, 14, 15], [10, 11, 12, 16, 17, 18]
ps21 = [EB_[x] @ EB_[y] for x, y in itertools.combinations(Cset, 2)] + [EB_[x] @ EB_[y] for x, y in itertools.combinations(Wset, 2)]
ok &= len(ps21) == 21 and all(np.allclose(X @ Pst, Pst @ X) for X in ps21)
gen23 = ps21 + [J3, J3 @ G3]
tang = [X @ Phi0 - Phi0 @ X for X in gen23]
ok &= all(dyadic_exact(t_) for t_ in tang)
scale_ = 2 ** 12
Ti = [np.round(t_ * scale_).astype(np.int64).reshape(-1) for t_ in tang]
Gram = [[int(Ti[i] @ Ti[j]) for j in range(23)] for i in range(23)]
r21 = rank_q([row[:21] for row in Gram[:21]])
r23 = rank_q(Gram)
ok &= r21 == 9 and r23 == 9
report("d2-21", ok, "no Majoron (restates Thm 22.2 of ES17), re-verified exactly at three constituents: [J,Phi_0] = "
       "[J Gamma, Phi_0] = 2 [t^R_3, Phi_0] != 0, and the tangent spaces of the 21-orbit (Pati-Salam) and of the 23-orbit "
       "(with J, J Gamma) at Phi_0 both have dimension 9: nine Goldstone directions, all eaten")

# 1 - Gamma: kept by Phi0, moved by every Dirac-type kernel (blade search with the sign rule)
JmJG = J3 - J3 @ G3
ok = np.allclose(JmJG @ Phi0 - Phi0 @ JmJG, 0, atol=1e-12)
internal = [A for A in range(1, 19) if A not in (1, 4, 5, 6)]


def blade_comm(S1, S2):
    return (len(S1) * len(S2) - len(set(S1) & set(S2))) % 2 == 0


def blade_sq(Sset):
    r_ = len(Sset)
    return (-1) ** (r_ * (r_ - 1) // 2) * int(np.prod([eta18[A] for A in Sset]))


allset = list(range(1, 19))
inv1 = sorted(set(allset) ^ set(Wset))           # Gamma chi_W ~ blade on all \ W
inv2 = sorted(set(Jset) ^ set(Cset))             # J chi_C
inv3 = sorted(set(Jset) ^ {2, 3})                # J q_A
found = None
for r_ in range(0, 8):
    for I_ in itertools.combinations(internal, r_):
        S_ = sorted({1} | set(I_))
        if blade_sq(S_) != -1:
            continue
        if not all(blade_comm(S_, inv) for inv in (inv1, inv2, inv3)):
            continue
        if not blade_comm(S_, Jset) or blade_comm(S_, allset):
            continue
        found = S_
        break
    if found:
        break
Kd = Pst @ prod(found) @ Pst
ok &= found is not None and np.linalg.norm(Kd) > 1e-6 and np.allclose(Kd.T, -Kd)
ok &= np.allclose(J3 @ Kd, Kd @ J3) and np.allclose(G3 @ Kd, -Kd @ G3)
ok &= all(np.allclose(Kd @ (EB_[x] @ EB_[y]), (EB_[x] @ EB_[y]) @ Kd) for x, y in itertools.combinations((4, 5, 6), 2))
ok &= all(np.allclose(Kd @ (EB_[1] @ EB_[x]), -(EB_[1] @ EB_[x]) @ Kd) for x in (4, 5, 6))
ok &= all(np.allclose(Kd @ X, X @ Kd) for X in ps21[:15])          # colour singlet: a (1,2,2) kernel
cK = JmJG @ Kd - Kd @ JmJG
ok &= np.allclose(cK, -2 * J3 @ G3 @ Kd) and np.linalg.norm(cK) > 1e-6
report("d2-22", ok, f"U(1)_(1-Gamma) (inherited; the arena's input is J - J Gamma = J + Om at one constituent, Thm 18.5(b)): "
       f"[J - J Gamma, Phi_0] = 0, while for a Dirac-type kernel M (e.g. the colour-singlet bidoublet kernel P_* E{found} P_*, "
       f"Lorentz invariant, J-linear, Gamma-odd) [J - J Gamma, M] = -2 J Gamma M != 0: every bidoublet value breaks it")

# ================================================================================================= d2-23 conclusion checks
ok = sp.Matrix(G) ** 2 * ka ** 2 == -ka ** 2 * sp.eye(6)
cs_, ell_ = sp.symbols('c ell', positive=True)
ok &= sp.simplify(((cs_ / ell_) * sp.Matrix(G)) ** 2 + (cs_ / ell_) ** 2 * sp.eye(6)) == sp.zeros(6, 6)
report("d2-23", ok, "(kappa G)^2 = -(c/ell)^2 1: definitional, G^2=-1 (adopted in Thm 16.2) with kappa=c/ell (eq. (7) of "
       "ES17); c also enters through the conversion inside eta (Thm 3.2); the constant c generates nothing")

# ================================================================================================= d2-24 electroweak count
# Theorem 22.3 of ES17 re-derived in exact integer arithmetic (the adversarial check x2-24 does it with numerical SVD):
# kernel blades (Thm 19.2(i)), the Dirac-type colour-singlet (1,2,2) kernels, their Q-commuting up- and down-type
# parts (Prop 21.2(i)), and the ranks of the 21- and 23-orbits at Phi = vR Phi0 + ku K1u + kd K1d.
def to_int(M_, scale=1):
    S_ = M_ * scale
    R_ = np.rint(S_)
    assert np.array_equal(R_, S_) and np.max(np.abs(R_)) < 2 ** 40
    return R_


def imul(*ms):
    """product of integer-valued float64 matrices, checked exact (all values stay below 2^53)"""
    r_ = ms[0]
    for m_ in ms[1:]:
        r_ = r_ @ m_
        assert np.max(np.abs(r_)) < 2 ** 50 and np.array_equal(r_, np.rint(r_))
    return r_


def gram_rank(mats):
    V_ = np.array([np.rint(M_).astype(np.int64).reshape(-1) for M_ in mats])
    bound = int(np.max(np.abs(V_))) ** 2 * V_.shape[1]
    assert bound < 2 ** 62
    Gm = V_ @ V_.T
    return sp.Matrix(Gm.tolist()).rank(), Gm


def int_null(Gm):
    """integer basis of the null space of an integer Gram matrix (exact)"""
    out = []
    for v in sp.Matrix(Gm.tolist()).nullspace():
        den = sp.ilcm(*[sp.fraction(x)[1] for x in v])
        w = [int(x * den) for x in v]
        g_ = int(np.gcd.reduce(np.abs(np.array(w, dtype=np.int64))))
        out.append([x // g_ for x in w])
    return out


Pst8 = to_int(Pst, 8)                                  # 8 P_*
Phi_i = to_int(Phi0, 4096)                             # 4096 Phi_0
J3i, G3i = to_int(J3), to_int(G3)
Hcol = -(prod([10, 16]) + prod([11, 17]) + prod([12, 18]))          # H of Section 19, B-L = -(1/3) J H
Q12 = to_int(6 * J3 @ EB_[7] @ EB_[13] - 2 * J3 @ Hcol)              # 12 Q = 12 T^L_3 + 12 T^R_3 + 6 (B-L)
ok = np.allclose(Q12, 12 * (-J3 @ (tR3 + 0.5 * (I512 - chiW) @ tau3)) + 6 * (-(J3 @ Hcol) / 3))
# on the block 12 Q takes the values 8, -4, 0, -12 (u, d, nu, e)
QP = imul(Q12, Pst8)
ok &= not np.any(imul(QP - 8 * Pst8, Q12 + 4 * I512, Q12 + 12 * I512, Q12)) and np.array_equal(Q12 @ Pst8, Pst8 @ Q12)
# the 2176 kernel blades of Theorem 19.2(i)
kern = []
for r_ in range(len(Jset) + 1):
    for I_ in itertools.combinations(Jset, r_):
        for pre in ((1,), (4, 5, 6)):
            S_ = sorted(set(pre) | set(I_))
            if blade_sq(S_) == -1 and all(blade_comm(S_, inv) for inv in (inv1, inv2, inv3)):
                kern.append(S_)
ok &= len(kern) == 2176
# Dirac-type (J-linear, Gamma-odd) colour singlets, compressed to the block: they span the 8-dim (1,2,2)
dsing = [S_ for S_ in kern if blade_comm(S_, Jset) and not blade_comm(S_, allset)
         and all(blade_comm(S_, [x, y]) for x, y in itertools.combinations(Cset, 2))]
comp = [M_ for M_ in (imul(Pst8, prod(S_), Pst8) for S_ in dsing) if np.any(M_)]
basis = []
for M_ in comp:
    if gram_rank(basis + [M_])[0] == len(basis) + 1:
        basis.append(M_)
ok &= len(basis) == 8          # every other compression lies in their span
# the Q-commuting part is 4-dim (two up-type, two down-type: Prop 21.2(i))
_, Gq = gram_rank([imul(M_, Q12) - imul(Q12, M_) for M_ in basis])
nQ = int_null(Gq)
KQ = [sum(int(v[i]) * basis[i] for i in range(8)) for v in nQ]
ok &= len(KQ) == 4
W_up = imul(Q12 - 8 * I512, Q12, Pst8)                # zero on u, nu; invertible on d, e
W_dn = imul(Q12 + 4 * I512, Q12 + 12 * I512, Pst8)    # zero on d, e; invertible on u, nu
nU = int_null(gram_rank([imul(M_, W_up) for M_ in KQ])[1])
nD = int_null(gram_rank([imul(M_, W_dn) for M_ in KQ])[1])
ok &= len(nU) == 2 and len(nD) == 2
K1u = sum(int(nU[0][i]) * KQ[i] for i in range(4))
K1d = sum(int(nD[0][i]) * KQ[i] for i in range(4))
ok &= np.array_equal(K1u.T, -K1u) and np.array_equal(K1u @ J3i, J3i @ K1u) and np.array_equal(K1u @ G3i, -G3i @ K1u)
ok &= np.array_equal(K1d.T, -K1d) and np.array_equal(K1d @ J3i, J3i @ K1d) and np.array_equal(K1d @ G3i, -G3i @ K1d)
gen23i = [to_int(X) for X in gen23]
JmJGi = to_int(JmJG)


def orbit_ranks(vR, ku, kd):
    Ph = vR * Phi_i + ku * K1u + kd * K1d
    tg = [imul(X, Ph) - imul(Ph, X) for X in gen23i]
    r21_ = gram_rank(tg[:21])[0]
    r23_ = gram_rank(tg)[0]
    r21g = gram_rank(tg[:21] + [imul(JmJGi, Ph) - imul(Ph, JmJGi)])[0]     # is the 1-Gamma tangent eaten?
    return r21_, r23_, r21g


ew = {pt: orbit_ranks(*pt) for pt in ((3, 2, 5), (2, -1, 3), (3, 2, 0), (0, 2, 5))}
ok &= ew[(3, 2, 5)] == (12, 13, 13) and ew[(2, -1, 3)] == (12, 13, 13)    # generic: one uneaten, the 1-Gamma one
ok &= ew[(3, 2, 0)][:2] == (12, 12) and ew[(0, 2, 5)][:2] == (5, 6)
report("d2-24", ok, f"electroweak count (restates Thm 22.3 of ES17), re-derived exactly: {len(kern)} kernel blades; the "
       f"Dirac-type colour-singlet compressions span the 8-dim (1,2,2), whose Q-commuting part is 4-dim, 2 up- + 2 "
       f"down-type; at Phi = vR Phi0 + ku K1u + kd K1d the (21,23)-orbit dimensions are {ew[(3, 2, 5)][:2]} and "
       f"{ew[(2, -1, 3)][:2]} at two generic points (one uneaten direction, the tangent of J - J Gamma), "
       f"{ew[(3, 2, 0)][:2]} at kd=0, {ew[(0, 2, 5)][:2]} at vR=0 (integer Gram ranks over Q)")

if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("ALL d2 CHECKS PASSED")
sys.exit(0)
