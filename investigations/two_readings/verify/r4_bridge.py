#!/usr/bin/env python3
"""Verification script for section r4 ("The bridge T = nJ(n) between the two readings") of the two-readings note.

    python3 investigations/two_readings/verify/r4_bridge.py

Prints one line 'PASS r4-<n>: <claim>' (or FAIL) per claim and exits 0 iff every check passes.  Everything is exact:
integer/rational matrices in the realisation of eq. (1) of the paper (tex/s02_arena.tex, tex/sA_realisation.tex),
identities in a general clock n = (n1,n2,n3) checked as polynomial identities (with the factor |n|^2 kept explicit) or on
the rational parametrisation n = (2a, 2b, 1-a^2-b^2)/(1+a^2+b^2) of the unit sphere, and linear solution spaces computed by
exact rational row reduction (sympy DomainMatrix over QQ).  No floating point is used anywhere.
"""
import itertools
import sys
import time
from fractions import Fraction
from functools import reduce

import sympy as sp
from sympy import QQ
from sympy.combinatorics import Permutation
from sympy.polys.matrices import DomainMatrix

T0 = time.time()
FAILS = []


def check(cid, claim, ok):
    ok = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'} r4-{cid}: {claim}")
    if not ok:
        FAILS.append(cid)


# ---------------------------------------------------------------------------------------------------------------------
# The realisation of Cl(3,3) on R^8; basis |b1 b2 b3>, index 4 b1 + 2 b2 + b3.
# ---------------------------------------------------------------------------------------------------------------------
def kron(*ms):
    return reduce(lambda a, b: sp.kronecker_product(a, b), ms)


sx = sp.Matrix([[0, 1], [1, 0]])
sz = sp.diag(1, -1)
ep = sp.Matrix([[0, -1], [1, 0]])
I2 = sp.eye(2)
I8 = sp.eye(8)
Z8 = sp.zeros(8)
Z81 = sp.zeros(8, 1)
e = [None, kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx), kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
s = [None, e[4], e[5], e[6]]                      # spatial triple s_a = e_{a+3}
etad = [None, 1, 1, 1, -1, -1, -1]
cyc = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]


def mprod(ms):
    return reduce(lambda a, b: a * b, ms, I8)


def blade(I):
    return mprod([e[a] for a in I])


def comm(a, b):
    return a * b - b * a


def acomm(a, b):
    return a * b + b * a


def isZ(M):
    return sp.expand(M) == sp.zeros(*M.shape)


def tvec(n):
    """temporal vector n_1 e_1 + n_2 e_2 + n_3 e_3"""
    return sum((n[i] * e[i + 1] for i in range(3)), Z8)


def svec(v):
    return sum((v[i] * s[i + 1] for i in range(3)), Z8)


def vec(c):
    return sum((c[a - 1] * e[a] for a in range(1, 7)), Z8)


def vec_coords(v):
    c = [sp.Rational(1, 8) * etad[a] * (e[a] * v).trace() for a in range(1, 7)]
    assert isZ(v - vec(c))
    return c


def admat(beta):
    """6x6 matrix of X -> [beta, X] on V in the basis e_1..e_6 (coordinates (B;E))"""
    return sp.Matrix(6, 6, lambda r, c: vec_coords(comm(beta, e[c + 1]))[r])


def conjmat(g, gi=None):
    gi = g.inv() if gi is None else gi
    return sp.Matrix(6, 6, lambda r, c: vec_coords(g * e[c + 1] * gi)[r])


def basis(k):
    v = sp.zeros(8, 1)
    v[k] = 1
    return v


def flat(M):
    return list(M)


T = blade([1, 2, 3])                              # mass form, I_T
s123 = blade([4, 5, 6])
omega = blade([1, 2, 3, 4, 5, 6])                 # = Gamma
om = [None] + [e[i] * e[i + 3] for i in (1, 2, 3)]
ap = [None] + [(e[i] + e[i + 3]) / 2 for i in (1, 2, 3)]
am = [None] + [(e[i] - e[i + 3]) / 2 for i in (1, 2, 3)]
Nop = sum((ap[i] * am[i] for i in (1, 2, 3)), Z8)
parN = om[1] * om[2] * om[3]                      # (-1)^N
vac, full = basis(0), basis(7)
f = [None] + [ap[k] * vac for k in (1, 2, 3)]
g = [None, ap[2] * ap[3] * vac, ap[3] * ap[1] * vac, ap[1] * ap[2] * vac]
Gm = sp.Matrix(sp.BlockMatrix([[sp.zeros(3), -sp.eye(3)], [sp.eye(3), sp.zeros(3)]]))   # G e_i = e_{i+3}, G e_{i+3} = -e_i
etaM = sp.diag(1, 1, 1, -1, -1, -1)
Lb = {i: (e[j + 3] * e[k + 3] - e[j] * e[k]) / 2 for (i, j, k) in cyc}                 # diagonal rotations
Kb = {i: (e[k] * e[j + 3] - e[j] * e[k + 3]) / 2 for (i, j, k) in cyc}                 # boosts of l_G
Ab = {i: (e[j] * e[k] + e[j + 3] * e[k + 3]) / 2 for (i, j, k) in cyc}                 # anti-paired rotations
tb = {i: e[j] * e[k] / 2 for (i, j, k) in cyc}                                          # so(3)_+ (temporal triple alone)
Qb = {i: -T * Lb[i] for i in (1, 2, 3)}                                                 # rho_oct(K_i)
LV = {i: admat(Lb[i]) for i in (1, 2, 3)}
KV = {i: admat(Kb[i]) for i in (1, 2, 3)}
KEYS = [('L', 1), ('L', 2), ('L', 3), ('K', 1), ('K', 2), ('K', 3)]
rhoV = {('L', i): LV[i] for i in (1, 2, 3)} | {('K', i): KV[i] for i in (1, 2, 3)}     # l_G on V_EM
rho_oct = {('L', i): Lb[i] for i in (1, 2, 3)} | {('K', i): Qb[i] for i in (1, 2, 3)}  # section r3
spin_lG = {('L', i): Lb[i] for i in (1, 2, 3)} | {('K', i): Kb[i] for i in (1, 2, 3)}


def J(n):
    return T * tvec(n)


def Omega(n):
    return tvec(n) * s[1] * s[2] * s[3]


def phi(n):
    """dictionary phi_n : l_G -> l_int(n) of section r2 (L_i -> s_j^s_k, K_i -> s_i^n), as Clifford elements = spin lift"""
    nv = tvec(n)
    return {('L', i): s[j] * s[k] / 2 for (i, j, k) in cyc} | {('K', i): s[i] * nv / 2 for i in (1, 2, 3)}


def lint(n):
    return list(phi(n).values())


def Phi_of(x):
    """field-octant map of section r3: Phi(X) = E.f + B.g, x = (B1,B2,B3,E1,E2,E3)"""
    return sum((x[k + 2] * f[k] + x[k - 1] * g[k] for k in (1, 2, 3)), sp.zeros(8, 1))


PhiM = sp.Matrix.hstack(*[Phi_of([1 if c == r else 0 for c in range(6)]) for r in range(6)])


def S_n(n, x):
    """Clifford image of phi_n(l(X)), l(X) = sum B_i L_i + E_i K_i : the field strength as a bivector of E_n"""
    nv = tvec(n)
    return sum((x[i - 1] * s[j] * s[k] / 2 + x[i + 2] * s[i] * nv / 2 for (i, j, k) in cyc), Z8)


def M_P(n, x):
    """Pauli kernel T J(n) S_n(X)"""
    return T * J(n) * S_n(n, x)


def M_D(n, x):
    """dual (electric-dipole) kernel T Gamma S_n(X)"""
    return T * omega * S_n(n, x)


def Gx(x):
    return [-x[3], -x[4], -x[5], x[0], x[1], x[2]]


def etax(x):
    return list(x[:3]) + [-a for a in x[3:]]


def frame(n):
    """rational orthogonal (unnormalised) u, u' with (n, u, u') positively oriented"""
    nv = sp.Matrix(n)
    u = sp.Matrix([n[1], -n[0], 0]) if (n[0], n[1]) != (0, 0) else sp.Matrix([0, 1, 0])
    return nv, u, nv.cross(u)


# exact linear algebra ------------------------------------------------------------------------------------------------
EU = [sp.Matrix(8, 8, lambda r, c: 1 if (r, c) == (i, j) else 0) for i in range(8) for j in range(8)]


def nullspace_rows(rows, ncols):
    rows = [r for r in rows if any(x != 0 for x in r)]
    if not rows:
        return [[1 if i == j else 0 for i in range(ncols)] for j in range(ncols)]

    def q(x):
        if isinstance(x, Fraction):
            return QQ(x.numerator, x.denominator)
        if isinstance(x, int):
            return QQ(x)
        return QQ.from_sympy(sp.Rational(x))
    DM = DomainMatrix([[q(x) for x in r] for r in rows], (len(rows), ncols), QQ)
    ns = DM.nullspace()
    if ns.shape[0] == 0:
        return []
    M = ns.to_Matrix()
    return [list(M.row(i)) for i in range(M.rows)]


def solve_linear(unknown_basis, constraint):
    cols = [constraint(b) for b in unknown_basis]
    m = len(cols[0])
    return nullspace_rows([[cols[j][i] for j in range(len(cols))] for i in range(m)], len(unknown_basis))


def combo(coeffs, B):
    return sum((sp.Rational(c) * b for c, b in zip(coeffs, B)), Z8)


def rank_of(mats):
    return sp.Matrix([flat(m) for m in mats]).rank()


def commutant(gens):
    return [combo(v, EU) for v in solve_linear(EU, lambda X: sum((flat(comm(X, y)) for y in gens), []))]


# clocks --------------------------------------------------------------------------------------------------------------
n1, n2, n3 = sp.symbols('n1 n2 n3', real=True)
NS = [n1, n2, n3]
N2 = n1 ** 2 + n2 ** 2 + n3 ** 2
pa, pb = sp.symbols('a b', real=True)
DEN = 1 + pa ** 2 + pb ** 2
NU = [2 * pa / DEN, 2 * pb / DEN, (1 - pa ** 2 - pb ** 2) / DEN]      # unit clock, rational parametrisation


def isZu(M):
    return sp.simplify(sp.together(M)) == sp.zeros(*M.shape)


C1 = [sp.Integer(1), sp.Integer(0), sp.Integer(0)]
C7 = [sp.Rational(2, 7), sp.Rational(3, 7), sp.Rational(6, 7)]
C3 = [sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)]
C9 = [sp.Rational(-1, 9), sp.Rational(4, 9), sp.Rational(8, 9)]
Bs = sp.symbols('B1:4', real=True)
Es = sp.symbols('E1:4', real=True)
X = list(Bs) + list(Es)
Xv = sp.Matrix(X)

# =====================================================================================================================
# r4-1  conventions
# =====================================================================================================================
ok = all(acomm(e[a], e[b]) == (2 * etad[a] if a == b else 0) * I8 for a in range(1, 7) for b in range(1, 7))
ok &= all(om[i].is_diagonal() for i in (1, 2, 3)) and Nop.is_diagonal()
ok &= all(Nop[k, k] == sum(1 for i in (1, 2, 3) if om[i][k, k] == -1) for k in range(8))
ok &= om[1][0, 0] == om[2][0, 0] == om[3][0, 0] == 1 and Nop[0, 0] == 0 and Nop[7, 7] == 3
ok &= T * T == -I8 and T.T == -T and omega * omega == I8 and omega.T == omega and s123.T == s123 and s123 * s123 == I8
ok &= T * omega == -s123
ok &= all(e[a].T == etad[a] * e[a] for a in range(1, 7))
check(1, "realisation: Clifford relations, omega_i diagonal, N = number of minus signs, vacuum (+,+,+) at N=0 and (-,-,-) at N=3; "
         "T^2=-1, T antisymmetric, Gamma=omega symmetric with Gamma^2=1, T Gamma = -e4e5e6 (symmetric, square 1)", ok)

# =====================================================================================================================
# r4-2  the identity T = n J(n) = J(n) n
# =====================================================================================================================
nv, Jn = tvec(NS), J(NS)
ok = isZ(nv * nv - N2 * I8) and isZ(Jn * Jn + N2 * I8) and isZ(T * nv - nv * T)
ok &= isZ(Jn - nv * T) and isZ(Jn - (n1 * e[2] * e[3] + n2 * e[3] * e[1] + n3 * e[1] * e[2]))
ok &= isZ(nv * Jn - N2 * T) and isZ(Jn * nv - N2 * T)
ok &= all(isZ(comm(Jn, y)) for y in [nv] + [s[a] for a in (1, 2, 3)])
nu = tvec(NU)
ok &= isZu(nu * J(NU) - T) and isZu(J(NU) * nu - T) and isZu(J(NU) * J(NU) + I8)
# the clock dependence: n -> J(n) is injective, and J(n) is an antisymmetric matrix
ok &= sp.Matrix([flat(J([1, 0, 0])), flat(J([0, 1, 0])), flat(J([0, 0, 1]))]).rank() == 3 and isZ(Jn + Jn.T)
check(2, "for every n in V^+: n^2=|n|^2, J(n)=Tn=nT=n1e2e3+n2e3e1+n3e1e2, J(n)^2=-|n|^2, nJ(n)=J(n)n=|n|^2 T; hence "
         "T = nJ(n) = J(n)n for every unit clock (also on the rational parametrisation of S^2); J(n) commutes with n,s_1,s_2,s_3", ok)

# =====================================================================================================================
# r4-3  covariance under so(3)_+ (temporal triple alone)
# =====================================================================================================================
ok = True
for i in (1, 2, 3):
    c = vec_coords(comm(tb[i], nv))
    ok &= comm(tb[i], T) == Z8 and isZ(comm(tb[i], Jn) - J(c[:3])) and all(x == 0 for x in c[3:])
    ok &= admat(tb[i]) * Gm != Gm * admat(tb[i])
# group level: rational rotor q = u2 u1 with u1 = e1 - n, u2 perp n, maps e1 -> n
rotors = {}
for name, n in [('C7', C7), ('C3', C3), ('C9', C9)]:
    u1 = tvec([1 - n[0], -n[1], -n[2]])
    q = tvec([n[1], -n[0], 0]) * u1
    qi = q.inv()
    rotors[name] = (q, qi)
    ok &= q * e[1] * qi == tvec(n) and q * T * qi == T and q * J(C1) * qi == J(n) and q * s123 * qi == s123
    ok &= all(q * s[a] * qi == s[a] for a in (1, 2, 3)) and q * Omega(C1) * qi == Omega(n)
    ok &= (q.T * q).is_diagonal() and len(set((q.T * q).diagonal())) == 1
check(3, "so(3)_+: [t_i,T]=0 and [t_i,J(n)]=J([t_i,n]) for symbolic n; rational rotors q (q e1 q^-1 = n at (2,3,6)/7,(1,2,2)/3,"
         "(-1,4,8)/9) fix T, s_a, e4e5e6 and carry J(e1), Omega_e1 to J(n), Omega_n; so(3)_+ does not commute with G", ok)

# =====================================================================================================================
# r4-4  covariance under the diagonal rotations so(3)_Delta
# =====================================================================================================================
ok = True
for i in (1, 2, 3):
    c = vec_coords(comm(Lb[i], nv))
    ok &= comm(Lb[i], T) == Z8 and isZ(comm(Lb[i], Jn) - J(c[:3])) and all(x == 0 for x in c[3:])
    ok &= LV[i] * Gm == Gm * LV[i]
RD = (I8 + sp.Rational(1, 2) * Lb[3]) * (I8 - sp.Rational(1, 2) * Lb[3]).inv()      # Cayley transform: rational, orthogonal
RDi = RD.inv()
RV = conjmat(RD, RDi)
RN = list(RV[:3, :3] * sp.Matrix(C7))
ok &= RD.T * RD == I8 and RD * T * RDi == T and RD * J(C7) * RDi == J(RN) and RV * Gm == Gm * RV
ok &= RV[:3, 3:] == sp.zeros(3) and RV[:3, :3] == RV[3:, 3:]
check(4, "so(3)_Delta: [L_i,T]=0, [L_i,J(n)]=J([L_i,n]), [L_i,G]=0; a finite diagonal rotation R (Cayley, exact) satisfies "
         "R T R^-1 = T, R J(n) R^-1 = J(Rn) and rotates both triples alike", ok)


# =====================================================================================================================
# r4-5, r4-6  permutations of the temporal axes, alone and paired
# =====================================================================================================================
def implement(images):
    """Skolem-Noether: the U (unique up to scale) with U e_a = images[a] U; integer, normalised by U|vac> having
    first nonzero entry positive"""
    sol = solve_linear(EU, lambda U: sum((flat(U * e[a] - images[a] * U) for a in range(1, 7)), []))
    assert len(sol) == 1
    U = combo(sol[0], EU)
    U = U * sp.ilcm(*[sp.fraction(x)[1] for x in U])
    U = U / sp.igcd(*[int(x) for x in U if x != 0])
    v = U * vac
    first = next(x for x in v if x != 0)
    return U if first > 0 else -U


def perm_images(sig, paired):
    im = [None] * 7
    for i in (1, 2, 3):
        im[i] = e[sig[i - 1]]
        im[i + 3] = e[sig[i - 1] + 3] if paired else e[i + 3]
    return im


def sgn(sig):
    return Permutation([x - 1 for x in sig]).signature()


def permute(sig, v):
    out = [0, 0, 0]
    for i in range(3):
        out[sig[i] - 1] = v[i]
    return out


PERM = {}
ok5, ok6 = True, True
for sig in itertools.permutations((1, 2, 3)):
    for paired in (False, True):
        U = implement(perm_images(sig, paired))
        Ui = U.inv()
        PERM[(sig, paired)] = (U, Ui)
        A = lambda y: U * y * Ui
        UTU = U.T * U
        ok5 &= UTU == UTU[0, 0] * I8 and UTU[0, 0] > 0          # orthogonal up to a positive scale
        ok5 &= A(T) == sgn(sig) * T and isZ(A(Jn) - sgn(sig) * J(permute(sig, NS))) and isZ(A(nv) - tvec(permute(sig, NS)))
        ok5 &= A(omega) == (1 if paired else sgn(sig)) * omega
        P6 = conjmat(U, Ui)
        commutesG = P6 * Gm == Gm * P6
        ok5 &= commutesG == (paired or sig == (1, 2, 3))
        if paired:
            ok5 &= all(A(om[i]) == om[sig[i - 1]] for i in (1, 2, 3)) and U * vac == vac
            Xt = [sgn(sig) * b for b in permute(sig, Bs)] + permute(sig, Es)
            ok6 &= isZ(U * Phi_of(X) - Phi_of(Xt))
            Phip = U * PhiM * P6.inv()                       # transported octant map
            ok6 &= Phip == (PhiM * (-etaM) if sgn(sig) == -1 else PhiM)
            ok6 &= A(-T) * Phip == Phip * Gm
            ok6 &= PhiM * (-etaM) == -parN * PhiM            # Phi o (-eta) = Phi_t of section r3
check(5, "all six permutations sigma of the temporal axes, alone and paired with the spatial axes: the implementing U is "
         "orthogonal up to scale, A(T)=sgn(sigma)T, A(J(n))=sgn(sigma)J(sigma n), A(n)=sigma n (symbolic n); the paired ones commute "
         "with G, permute the omega_i and fix the vacuum; no nontrivial temporal permutation alone commutes with G", ok5)
check(6, "paired permutation: U Phi(X) = Phi(Xt), Xt=(sgn(sigma) P B; P E) (B axial); the transported octant map is "
         "Phi o(-eta) = Phi_t = -(-1)^N Phi for odd sigma, and A(-T) = sgn(sigma)(-T) is its duality operator: "
         "A(-T) Phi' = Phi' G", ok6)

# =====================================================================================================================
# r4-7  time reversal of the clock as a half-turn; charge conjugation C = R_pi T
# =====================================================================================================================
ok = True
for n in (C1, C7, C3):
    nvec, u, up = frame(n)
    R = tvec(n) * tvec(list(up))            # rotor of the half-turn about u (plane (n,u'))
    Ri = R.inv()
    ok &= R * tvec(n) * Ri == -tvec(n) and R * T * Ri == T and R * J(n) * Ri == J([-a for a in n]) == -J(n)
    ok &= R * tvec(list(u)) * Ri == tvec(list(u)) and all(R * s[a] * Ri == s[a] for a in (1, 2, 3))
    C = R * T
    # C is a positive multiple (|n||u'|/|u|) of the vector u of the surplus plane U_n
    ok &= rank_of([C, tvec(list(u))]) == 1 and next(x for x in C if x != 0) * next(x for x in tvec(list(u)) if x != 0) > 0
    ok &= acomm(C, tvec(n)) == Z8 and acomm(C, J(n)) == Z8 and comm(C, T) == Z8
check(7, "clock reversal n -> -n is the half-turn R = n u' in Spin(V^+): R n R^-1 = -n, R T R^-1 = T, R J(n) R^-1 = J(-n) = -J(n); "
         "the charge conjugation of RE is C = R T, a vector of U_n anticommuting with n and J(n) and commuting with T", ok)

# =====================================================================================================================
# r4-8  the Dirac reading under these operations
# =====================================================================================================================
PI = {sg: (I8 + sg * nv) / 2 for sg in (1, -1)}
ok = all(isZ(T * PI[sg] - sg * Jn * PI[sg] - (1 - N2) * T / 2) for sg in (1, -1))
ok &= all(isZu(T * (I8 + sg * nu) / 2 - sg * J(NU) * (I8 + sg * nu) / 2) for sg in (1, -1))
# odd temporal permutation alone: kinetic kernels K^a_n = -n s_a -> K^a_{sigma n}, mass form -> -T, charge J -> -J(sigma n)
U, Ui = PERM[((2, 1, 3), False)]
n = C7
sn = permute((2, 1, 3), n)
ok &= all(U * (-tvec(n) * s[a]) * Ui == -tvec(sn) * s[a] for a in (1, 2, 3)) and U * T * Ui == -T and U * J(n) * Ui == -J(sn)
# the chiral (axial) rotation by Omega_n restores the sign of the mass form and preserves every kinetic kernel
Om = Omega(sn)
ok &= Om * Om == -I8 and Om.T == -Om and Om.T * T * Om == -T and all(Om.T * (-tvec(sn) * g_) * Om == -tvec(sn) * g_ for g_ in [tvec(sn)] + s[1:])
ok &= comm(Om, J(sn)) == Z8
check(8, "on the eigenspaces n=+-1 of every unit clock T=+-J(n) (rest flow exp(-m lambda T) = exp(-+m lambda J(n))); an odd "
         "temporal permutation carries the Dirac data (K^mu_n, T, J(n)) to (K^mu_{sigma n}, -T, -J(sigma n)): mass and "
         "charge reversed; Omega restores T and keeps every K^mu and J", ok)

# =====================================================================================================================
# r4-9  the three roles of T
# =====================================================================================================================
ok = isZu(tvec(NU) * J(NU) - T)                                       # (a) gamma^0 J(n) = T, every unit clock
ps = sp.symbols('p1:4', real=True)
mm = sp.Symbol('m', real=True)
H = lambda n, p: sum((p[a - 1] * tvec(n) * s[a] for a in (1, 2, 3)), Z8) + mm * tvec(n)
ok &= isZ(comm(T, Jn)) and isZ(T * H(NS, ps) * T.inv() - H(NS, [-x for x in ps]))      # T is a Dirac parity at every clock
ok &= isZ(T * T + I8)
ok &= conjmat(T) == etaM                                               # (b) Ad_T = eta
# uniqueness of the implementer of eta up to scale
sol = solve_linear(EU, lambda Y: sum((flat(Y * e[a] - etad[a] * e[a] * Y) for a in range(1, 7)), []))
ok &= len(sol) == 1 and rank_of([combo(sol[0], EU), T]) == 1
ok &= (-T) * PhiM == PhiM * Gm                                         # (c) duality on the octants
ok &= T * vac == full and all(T * f[k] == g[k] and T * g[k] == -f[k] for k in (1, 2, 3))
check(9, "three roles of T: (a) gamma^0 J(n) = T at every unit clock; T commutes with J(n) and T H_n(p) T^-1 = H_n(-p), so T is the "
         "admissible Dirac parity gamma^0 J of ES17 Thm 5.3 at every clock; (b) Ad_T = eta on V, T unique up to scale; "
         "(c) -T Phi = Phi G, T f_k = g_k, T g_k = -f_k", ok)

# =====================================================================================================================
# r4-10  J(n) = -L(n) + A(n); L(n) = (Dirac rotation about n) - J(n)/2
# =====================================================================================================================
Ln = sum((NS[i - 1] * Lb[i] for i in (1, 2, 3)), Z8)
An = sum((NS[i - 1] * Ab[i] for i in (1, 2, 3)), Z8)
sn_ = sum((NS[i - 1] * s[j] * s[k] for (i, j, k) in cyc), Z8)          # 2 x spin lift of the spatial rotation about n
ok = isZ(Jn - (-Ln + An)) and isZ(Ln * An) and isZ(An * Ln) and isZ(sn_ * sn_ + N2 * I8)
ok &= isZ(Ln - (sn_ / 2 - Jn / 2)) and isZ(sn_ / 2 - (Ln + An) / 2)
ok &= isZ(sn_ / 2 - sum((NS[i - 1] * phi(NS)[('L', i)] for i in (1, 2, 3)), Z8))
# the Dirac rotation rho_n(L(n)) = sn_/2 acts on V (by the commutator) trivially on V^+ and as v -> n x v on V^-:
# a rotation about the spatial axis sum n_i s_i paired with n, not about the clock n itself
nvec3 = sp.Matrix(NS)
for a in (1, 2, 3):
    ok &= isZ(comm(sn_ / 2, e[a]))
    ua = sp.Matrix([1 if b == a - 1 else 0 for b in range(3)])
    ok &= isZ(comm(sn_ / 2, s[a]) - svec(list(nvec3.cross(ua))))
ok &= all(Lb[i] == phi(C7)[('L', i)] - J([1 if a == i - 1 else 0 for a in range(3)]) / 2 for i in (1, 2, 3))
# unit clock: -L^2 and -A^2 are complementary rank-4 projectors; L^3 = -L, A^3 = -A, so exp(2 pi L(n)) = 1 = exp(2 pi A(n)),
# while exp(2 pi s(n)/2) = cos(pi) + sin(pi) s(n) = -1 and exp(-pi J(n)) = -1
Lu = sum((NU[i - 1] * Lb[i] for i in (1, 2, 3)), Z8)
Au = sum((NU[i - 1] * Ab[i] for i in (1, 2, 3)), Z8)
Pp, Pl = sp.simplify(-Lu * Lu), sp.simplify(-Au * Au)
ok &= isZu(Pp * Pp - Pp) and isZu(Pl * Pl - Pl) and isZu(Pp + Pl - I8) and isZu(Pp * Pl)
ok &= isZu(Lu * Lu * Lu + Lu) and isZu(Au * Au * Au + Au)
ok &= Pp.subs({pa: 0, pb: 0}).rank() == 4 and Pl.subs({pa: 0, pb: 0}).rank() == 4
check(10, "J(n) = -L(n) + A(n) with L(n)=sum n_i L_i (diagonal rotation), A(n)=sum n_i (e_je_k+e_{j+3}e_{k+3})/2 (anti-paired), "
          "L(n)A(n)=A(n)L(n)=0; L(n) = s(n)/2 - J(n)/2 where s(n)/2 = spin lift of phi_n(L(n)) is the Dirac rotation of V^- about the "
          "spatial axis sum n_i s_i (trivial on V^+) "
          "(rho_oct(L_i) = rho_n(L_i) - J(e_i)/2); for unit n -L^2, -A^2 are complementary rank-4 projectors, L^3=-L", ok)

# =====================================================================================================================
# r4-11  what J(n) does on the octants
# =====================================================================================================================
nvec = sp.Matrix(NS)
fn = sum((NS[k - 1] * f[k] for k in (1, 2, 3)), Z81)
gn = sum((NS[k - 1] * g[k] for k in (1, 2, 3)), Z81)
ok = isZ(Jn * Phi_of(X) - (-Phi_of(list(nvec.cross(sp.Matrix(Bs))) + list(nvec.cross(sp.Matrix(Es))))
                          + nvec.dot(sp.Matrix(Es)) * full - nvec.dot(sp.Matrix(Bs)) * vac))
ok &= isZ(Jn * vac - gn) and isZ(Jn * gn + N2 * vac) and isZ(Jn * full + fn) and isZ(Jn * fn - N2 * full)
ok &= all(isZ(Ln * v) for v in (vac, full, fn, gn))
us = sp.symbols('u1:4', real=True)
uvec = sp.Matrix(us)
fu = sum((us[k - 1] * f[k] for k in (1, 2, 3)), Z81)
gu = sum((us[k - 1] * g[k] for k in (1, 2, 3)), Z81)
w = nvec.cross(uvec)
ok &= isZ(Ln * fu - sum((w[k - 1] * f[k] for k in (1, 2, 3)), Z81)) and isZ(Ln * gu - sum((w[k - 1] * g[k] for k in (1, 2, 3)), Z81))
ok &= isZ(An * fu - nvec.dot(uvec) * full) and isZ(An * gu + nvec.dot(uvec) * vac)
# J(n) is not in the commutant of rho_oct and does not preserve the field octants
Jc = J(C7)
ok &= any(comm(Jc, r) != Z8 for r in rho_oct.values())
ok &= sp.Matrix.hstack(PhiM, Jc * PhiM).rank() == 8
check(11, "J(n)Phi(X) = -Phi(n x B; n x E) + (E.n)|full> - (B.n)|vac> for every n: on transverse octants J(n) = -L(n), the rotation "
          "about n; J(n): f(n) -> |full>, g(n) -> -|vac>, |vac> -> g(n), |full> -> -f(n); ker L(n) = poles + longitudinal "
          "octants; J(n) does not commute with rho_oct and does not preserve Phi(V)", ok)

# =====================================================================================================================
# r4-12  commutants
# =====================================================================================================================
c_oct = commutant(list(rho_oct.values()))
ok = len(c_oct) == 6
PiP = vac * vac.T + full * full.T
cand = [I8 - PiP, T * (I8 - PiP), vac * vac.T, vac * full.T, full * vac.T, full * full.T]
ok &= all(all(comm(c, r) == Z8 for r in rho_oct.values()) for c in cand) and rank_of(cand) == 6
for n in (C1, C7, C3):
    cs = commutant(lint(n))
    nvec_, u, up = frame(n)
    cand = [I8, tvec(list(u)), tvec(list(up)), J(n)]
    cand = cand + [c * Omega(n) for c in cand]
    ok &= len(cs) == 8 and all(all(comm(c, y) == Z8 for y in lint(n)) for c in cand) and rank_of(cand) == 8
    joint = commutant(lint(n) + list(rho_oct.values()))
    ok &= len(joint) == 1 and rank_of([joint[0], I8]) == 1
    ph = phi(n)
    ok &= all(comm(ph[('L', i)], ph[('L', j)]) == ph[('L', k)] and comm(ph[('L', i)], ph[('K', j)]) == ph[('K', k)]
              and comm(ph[('K', i)], ph[('K', j)]) == -ph[('L', k)] for (i, j, k) in cyc)
ok &= all(comm(Lb[i], Lb[j]) == Lb[k] and comm(Lb[i], Qb[j]) == Qb[k] and comm(Qb[i], Qb[j]) == -Lb[k] for (i, j, k) in cyc)
check(12, "commutant of the spin lift of l_int(n): dimension 8, span{1,u_1,u_2,J(n)} x {1,Omega_n} = M_2(C) (n=e1,(2,3,6)/7,"
          "(1,2,2)/3); of rho_oct(l_G): dimension 6, C on the field octants + M_2(R) on the poles; joint commutant R 1; "
          "phi_n and rho_oct are Lie homomorphisms", ok)


# =====================================================================================================================
# r4-13  three inequivalent Lorentz representations on S
# =====================================================================================================================
def casimirs(rep):
    C1_ = sum((rep[('K', i)] ** 2 - rep[('L', i)] ** 2 for i in (1, 2, 3)), Z8)
    C2_ = sum((rep[('K', i)] * rep[('L', i)] for i in (1, 2, 3)), Z8)
    return sp.simplify(C1_), sp.simplify(C2_ * C2_)


ok = True
c1, c2 = casimirs(phi(C7))
ok &= c1 == sp.Rational(3, 2) * I8 and c2 == -sp.Rational(9, 16) * I8
c1, c2 = casimirs(rho_oct)
ok &= c1 == 4 * (I8 - PiP) and c2 == -4 * (I8 - PiP)
c1, c2 = casimirs(spin_lG)
ok &= c1 == 3 * I8 and c2 == Z8
# rotation generator about e3: spin lift has L^2 = -1/4 (eigenvalues +-i/2, exp(2 pi L) = -1); rho_oct has L^3 = -L
Ls = phi(C7)[('L', 3)]
ok &= Ls * Ls == -I8 / 4 and Lb[3] ** 3 == -Lb[3] and (Lb[3] ** 2).rank() == 4
ok &= sp.Matrix(Ls.charpoly().all_coeffs()) == sp.Matrix((sp.Poly((sp.Symbol('x') ** 2 + sp.Rational(1, 4)) ** 4)).all_coeffs())
ok &= any(comm(rho_oct[k], phi(C1)[kk]) != Z8 for k in KEYS for kk in KEYS)          # rho_oct does not commute with rho_e1
check(13, "Casimirs sum K^2-L^2 and (sum K.L)^2: spin lift of l_int(n) via phi_n: 3/2 and -9/16 (2(1/2,0)+2(0,1/2)); rho_oct: 4 and "
          "-4 on the field octants, 0 on the poles ((1,0)+(0,1)+2(0,0)); spin lift of l_G: 3 and 0 (2(1/2,1/2)); L_3 has "
          "spectrum +-i/2 (exp 2pi L_3 = -1) in the first, L_3^3=-L_3 (exp 2pi L_3 = +1) in the others: pairwise inequivalent; rho_oct does not commute with rho_e1", ok)


# =====================================================================================================================
# r4-14, r4-15  covariant couplings
# =====================================================================================================================
def constraint_columns(n):
    """columns of the linear map M=(M_c)_{c=0..5} -> (s(phi Y)^T M_c + M_c s(phi Y) + sum_d R[d,c] M_d)_{Y,c}, exactly,
    indexed by (c, i, j) with M_c[i, j] = 1; returned as dict (c,i,j) -> list of 6*6*64 rationals"""
    ph = phi(n)
    Ys = [[[Fraction(int(sp.fraction(x)[0]), int(sp.fraction(x)[1])) for x in ph[k].row(r)] for r in range(8)] for k in KEYS]
    Rs = [[[int(x) for x in rhoV[k].row(r)] for r in range(6)] for k in KEYS]
    cols = {}
    for c in range(6):
        for i in range(8):
            for j in range(8):
                col = []
                for kk in range(6):
                    Y, R = Ys[kk], Rs[kk]
                    for cc in range(6):
                        rc = R[c][cc]
                        for r in range(8):
                            for t in range(8):
                                # (Y^T E_ij + E_ij Y)[r,t] = Y[i][r] d_{tj} + d_{ri} Y[j][t]  (slot cc = c only)
                                v = Fraction(0)
                                if cc == c:
                                    if t == j:
                                        v += Y[i][r]
                                    if r == i:
                                        v += Y[j][t]
                                if rc and r == i and t == j:
                                    v += rc
                                col.append(v)
                cols[(c, i, j)] = col
    return cols


def coupling_spaces(n):
    cols = constraint_columns(n)
    m = len(next(iter(cols.values())))
    anti = [(c, i, j) for c in range(6) for i in range(8) for j in range(i + 1, 8)]
    sym = [(c, i, j) for c in range(6) for i in range(8) for j in range(i, 8)]
    ca = [[cols[(c, i, j)][r] - cols[(c, j, i)][r] for r in range(m)] for (c, i, j) in anti]
    cs = [[cols[(c, i, j)][r] + (cols[(c, j, i)][r] if i != j else 0) for r in range(m)] for (c, i, j) in sym]
    na = nullspace_rows([[ca[q][r] for q in range(len(anti))] for r in range(m)], len(anti))
    nsym = nullspace_rows([[cs[q][r] for q in range(len(sym))] for r in range(m)], len(sym))

    def to_maps(vecs, labels, symmetric):
        out = []
        for v in vecs:
            Ms = [sp.zeros(8) for _ in range(6)]
            for coef, (c, i, j) in zip(v, labels):
                if coef != 0:
                    Ms[c][i, j] += coef
                    if i != j:
                        Ms[c][j, i] += coef if symmetric else -coef
            out.append(Ms)
        return out
    return to_maps(na, anti, False), to_maps(nsym, sym, True)


def as_map(Mfun, n):
    return [Mfun(n, [1 if c == r else 0 for c in range(6)]) for r in range(6)]


ok14 = True
for n in (C1, C7):
    ph = phi(n)
    Eb = [sp.Matrix(8, 6, lambda r, c: 1 if (r, c) == (i, j) else 0) for i in range(8) for j in range(6)]
    ok14 &= len(solve_linear(Eb, lambda M: sum((flat(ph[k].T * M + M * rhoV[k]) for k in KEYS), []))) == 0
    ok14 &= len(solve_linear(EU, lambda M: sum((flat(ph[k].T * M + M * rho_oct[k]) for k in KEYS), []))) == 0
# the 2 pi rotation argument: exp(2 pi s(phi L_3)) = -1 on psi, +1 on the field octants
ok14 &= (Lb[3] ** 3 == -Lb[3])
check(14, "no invariant bilinear psi^T M Phi(X) (psi under the spin lift of l_int(n), Phi(X) under rho_oct o phi_n^-1, or all of "
          "S under rho_oct), at n=e1 and (2,3,6)/7: the 2 pi rotation is -1 on psi and +1 on the octants", ok14)

ok15 = True
SPACES = {}
for n in (C1, C7):
    anti, sym = coupling_spaces(n)
    SPACES[tuple(n)] = (anti, sym)
    ok15 &= len(anti) == 2 and len(sym) == 6
check(15, "trilinear couplings psi^T M(Phi(X)) psi invariant under the joint action: dimension 2 for Grassmann psi "
          "(antisymmetric kernels, Lambda^2) and 6 for commuting psi (Sym^2), at n=e1 and n=(2,3,6)/7 (exact rational "
          f"row reduction of 2304 x 168 and 2304 x 216 systems; {time.time() - T0:.0f}s so far)", ok15)

# =====================================================================================================================
# r4-16  the couplings written through T and J(n)
# =====================================================================================================================
ok = True
# l(X) = sum B_i L_i + E_i K_i is l_G-equivariant: [Y, l(X)] = l(Y.X)
ellX = lambda x: sum((x[i - 1] * Lb[i] + x[i + 2] * Kb[i] for i in (1, 2, 3)), Z8)
ok &= all(isZ(comm(spin_lG[k], ellX(X)) - ellX(list(rhoV[k] * Xv))) for k in KEYS)
ok &= all(isZ(S_n(n, X) - sum((X[i - 1] * phi(n)[('L', i)] + X[i + 2] * phi(n)[('K', i)] for i in (1, 2, 3)), Z8)) for n in (C1, C7))
# transpose rule rho_n(Y)^T = -n rho_n(Y) n, the reason the kernels n C S_n(X) are invariant
ok &= all(phi(n)[k].T == -tvec(n) * phi(n)[k] * tvec(n) for n in (C1, C7) for k in KEYS)
for n in (C1, C7):
    nv_ = tvec(n)
    MP, MD = M_P(n, X), M_D(n, X)
    ok &= isZ(MP + nv_ * S_n(n, X)) and isZ(MD + s123 * S_n(n, X)) and isZ(MD - T * J(n) * Omega(n) * S_n(n, X))
    ok &= isZ(MP + MP.T) and isZ(MD + MD.T) and isZ(MD - M_P(n, Gx(X)))
    ok &= isZ(comm(MP, J(n))) and isZ(comm(MD, J(n)))
    ok &= isZ(MP - (svec(Es) / 2 - nv_ * sum((X[i - 1] * s[j] * s[k] for (i, j, k) in cyc), Z8) / 2))
    ok &= isZ(MD - (svec(Bs) / 2 + nv_ * sum((X[i + 2] * s[j] * s[k] for (i, j, k) in cyc), Z8) / 2))
    # S_n(X) = Clifford image of sum_{mu<nu} F_{mu nu} gamma_mu ^ gamma_nu with gamma_mu = eta_{mu nu} gamma^nu_n,
    # (F_X)_{0i} = E_i, (F_X)_{jk} = eps_ijk B_i; the same sum with F^{mu nu} gives S_n(eta X)
    Fl = sp.zeros(4, 4)
    for (i, j, k) in cyc:
        Fl[0, i], Fl[i, 0], Fl[j, k], Fl[k, j] = X[i + 2], -X[i + 2], X[i - 1], -X[i - 1]
    etaE = sp.diag(1, -1, -1, -1)
    Fu = etaE * Fl * etaE
    gdn = [nv_] + [-s[a] for a in (1, 2, 3)]
    biv = lambda F_: sum((F_[m, v] * gdn[m] * gdn[v] / 2 for m in range(4) for v in range(m + 1, 4)), Z8)
    ok &= isZ(biv(Fl) - S_n(n, X)) and isZ(biv(Fu) - S_n(n, etax(X)))
    anti, sym = SPACES[tuple(n)]
    flatmap = lambda Ms: sum((flat(m) for m in Ms), [])
    ok &= sp.Matrix([flatmap(m) for m in anti] + [flatmap(as_map(M_P, n)), flatmap(as_map(M_D, n))]).rank() == 2
    nvec_, u, up = frame(n)
    Cs = [tvec(list(u)), tvec(list(up)), J(n)]
    Cs = Cs + [c * Omega(n) for c in Cs]
    symmaps = [[nv_ * c * S_n(n, [1 if cc == r else 0 for cc in range(6)]) for r in range(6)] for c in Cs]
    ok &= all(all(m.T == m for m in Ms) for Ms in symmaps)
    ok &= sp.Matrix([flatmap(m) for m in sym] + [flatmap(m) for m in symmaps]).rank() == 6
check(16, "l(X) = sum(B_i L_i + E_i K_i) is l_G-equivariant and rho_n(Y)^T = -n rho_n(Y) n; the Grassmann couplings are spanned by M_n(X) = T J(n) S_n(X) = -n S_n(X) (real form of psibar sigma^{mu nu} psi F_{mu nu}) "
          "and M*_n(X) = T Gamma S_n(X) = T J(n) Omega_n S_n(X) = M_n(GX) (real form of i psibar sigma^{mu nu} gamma_5 psi F_{mu nu}), "
          "S_n(X) the Clifford image of phi_n(l(X)) = sum_{mu<nu} F_{mu nu} gamma_mu^gamma_nu (gamma_mu = eta_{mu nu} gamma^nu_n; with F^{mu nu} "
          "the sum is S_n(eta X)); both commute with J(n); M_n = E.s/2 - n(sum B_i s_js_k)/2; the "
          "commuting-psi couplings are n C S_n(X), C in {u_1,u_2,J(n)} x {1,Omega_n}", ok)

# =====================================================================================================================
# r4-17  parity
# =====================================================================================================================
ok = True
for n in (C1, C7):
    nv_ = tvec(n)
    ok &= isZ(nv_ * M_P(n, X) * nv_ - M_P(n, etax(X))) and isZ(nv_ * M_D(n, X) * nv_ + M_D(n, etax(X)))
    ok &= isZ(T.T * M_P(n, X) * T - M_P(n, etax(X))) and isZ(T.T * M_D(n, X) * T + M_D(n, etax(X)))
    ok &= nv_.T * T * nv_ == T and T.T * T * T == T
check(17, "under the Dirac parity psi -> n psi (or psi -> T psi) with X -> eta X: M_n is even, M*_n is odd, the mass form T is even", ok)

# =====================================================================================================================
# r4-18  changing the clock
# =====================================================================================================================
ok = True
for name, n in [('C7', C7), ('C3', C3), ('C9', C9)]:
    q, qi = rotors[name]
    ok &= isZ(M_P(n, X) - q * M_P(C1, X) * qi) and isZ(M_D(n, X) - q * M_D(C1, X) * qi)
    ok &= all(q * (-tvec(C1) * g_) * qi == -tvec(n) * gn_ for g_, gn_ in zip([tvec(C1)] + s[1:], [tvec(n)] + s[1:]))
    ok &= isZ(q * S_n(C1, X) * qi - S_n(n, X))
RX = list(RV * Xv)
ok &= isZ(M_P(RN, RX) - RD * M_P(C7, X) * RDi) and isZ(M_D(RN, RX) - RD * M_D(C7, X) * RDi)
ok &= not isZ(M_P(C1, X) - M_P([0, 1, 0], X)) and not isZ(M_D(C1, X) - M_D([0, 1, 0], X))
# kinetic kernels K^mu = T J(n) gamma^mu at every clock: the Dirac + Pauli Lagrangian is 1/2 psi^T T D_n psi
ok &= isZ(T * Jn * nv - (-N2 * I8)) and all(isZ(T * Jn * s[a] + nv * s[a]) for a in (1, 2, 3))
check(18, "M_n(X) = q M_e1(X) q^-1 and M*_n(X) = q M*_e1(X) q^-1 for the rotor q taking e1 to n (three rational clocks), X fixed; "
          "M_{Rn}(RX) = R M_n(X) R^-1 for a diagonal rotation R; the couplings depend on n (M_e1 != M_e2); the kinetic, mass, "
          "Pauli and dual kernels are T J(n) gamma^mu, T, T J(n) S_n, T J(n) Omega_n S_n, with T fixed by every q", ok)

# =====================================================================================================================
# r4-19  permutations and the couplings
# =====================================================================================================================
ok = True
n = C7
for sig in [(1, 3, 2), (2, 1, 3), (3, 2, 1), (2, 3, 1)]:
    sn = permute(sig, n)
    for paired in (False, True):
        U, Ui = PERM[(sig, paired)]
        A = lambda y: U * y * Ui
        Xt = ([sgn(sig) * b for b in permute(sig, Bs)] + permute(sig, Es)) if paired else X
        ok &= isZ(A(M_P(n, X)) - M_P(sn, Xt))
        ok &= isZ(A(M_D(n, X)) - (sgn(sig) if paired else 1) * M_D(sn, Xt))
        ok &= A(T) == sgn(sig) * T
check(19, "odd temporal permutation alone: A(M_n(X)) = M_{sigma n}(X), A(M*_n(X)) = M*_{sigma n}(X) while A(T) = -T: the Pauli "
          "couplings do not change sign with T (refuted), only relative to the mass; paired: A(M_n(X)) = M_{sigma n}(Xt), "
          "A(M*_n(X)) = sgn(sigma) M*_{sigma n}(Xt): the dual (electric-dipole) coupling changes sign with T", ok)

# =====================================================================================================================
# r4-20  the two first-order operators
# =====================================================================================================================
al = {a: nv * s[a] for a in (1, 2, 3)}
ok = all(al[a].T == al[a] for a in al) and all(isZ(al[a] ** 2 - N2 * I8) for a in al)
ok &= all(isZ(acomm(al[a], al[b])) for a in al for b in al if a < b)
ok &= all(isZ(Jn * al[a] - N2 * T * s[a]) for a in al)
ok &= all(e[a] == -T * J([1 if i == a - 1 else 0 for i in range(3)]) for a in (1, 2, 3))
ok &= isZ(nv * nv - N2 * I8) and all(isZ(nv * (nv * s[a]) - N2 * s[a]) for a in (1, 2, 3))   # gamma^0 gamma^mu d_mu
ks = sp.symbols('k1:4', real=True)
for trip in ([al[a].subs({n1: 1, n2: 0, n3: 0}) for a in (1, 2, 3)], [e[1], e[2], e[3]], [al[a].subs(dict(zip(NS, C7))) for a in (1, 2, 3)]):
    K_ = sum((ks[a] * trip[a] for a in range(3)), Z8)
    ok &= isZ(K_ * K_ - (ks[0] ** 2 + ks[1] ** 2 + ks[2] ** 2) * I8)
a1 = {a: tvec(C1) * s[a] for a in (1, 2, 3)}
# for a general clock: Z = n s_a e_a would need s1e1 = s2e2; Z = e_a n s_a would need e1 n s1 = e2 n s2, and the linear map
# n -> e1 n s1 - e2 n s2 is injective (its matrix in the 64 entries has rank 3)
ok &= s[1] * e[1] != s[2] * e[2]
dmap = e[1] * nv * s[1] - e[2] * nv * s[2]
ok &= sp.Matrix([[sp.diff(x, v) for v in NS] for x in flat(dmap)]).rank() == 3
ok &= len({tuple(e[a] * a1[a]) for a in (1, 2, 3)}) == 3 and len({tuple(a1[a] * e[a]) for a in (1, 2, 3)}) == 3
check(20, "alpha_a(n) = n e_{a+3} is a Pauli triple (symmetric, square |n|^2, anticommuting) with J(n) alpha_a(n) = |n|^2 T e_{a+3}; "
          "the octant triple is e_a = -T J(e_a); (sum k_a pi_a)^2 = |k|^2 for both triples, so both first-order operators square "
          "to the wave operator; for every clock n (symbolic) e_a and alpha_a(n) are not related by one-sided multiplication with a "
          "fixed element", ok)

# =====================================================================================================================
# r4-21  intertwiners of the two triples
# =====================================================================================================================
even_b = [blade(I) for r in range(0, 7, 2) for I in itertools.combinations(range(1, 7), r)]
odd_b = [blade(I) for r in range(1, 7, 2) for I in itertools.combinations(range(1, 7), r)]
ok = True
for n in (C1, C7):
    a_ = {k: tvec(n) * s[k] for k in (1, 2, 3)}
    con = lambda U: sum((flat(U * a_[k] - e[k] * U) for k in (1, 2, 3)), [])
    sols = solve_linear(EU, con)
    ok &= len(sols) == 8 and len(solve_linear(even_b, con)) == 0 and len(solve_linear(odd_b, con)) == 0
    U = sum((sp.Rational(k + 1) * combo(sols[k], EU) for k in range(8)), Z8)
    ok &= U.det() != 0
    Ui = U.inv()
    ph = phi(n)
    lam = {('L', i): -e[j] * e[k] / 2 for (i, j, k) in cyc} | {('K', i): -e[i] / 2 for i in (1, 2, 3)}
    ok &= all(isZ(U * ph[kk] * Ui - lam[kk]) for kk in KEYS)
ok &= all(Qb[i] == -(e[i] + s[i] * omega) / 2 and Lb[i] == -e[j] * e[k] / 2 + s[j] * s[k] / 2 for (i, j, k) in cyc)
ce = commutant([e[1], e[2], e[3]])
cand = [I8, s[1] * s[2], s[2] * s[3], s[3] * s[1]]
cand = cand + [T * c for c in cand]
ok &= len(ce) == 8 and all(all(comm(c, e[a]) == Z8 for a in (1, 2, 3)) for c in cand) and rank_of(cand) == 8
check(21, "the U with U alpha_a(n) U^-1 = e_a form (with 0) an 8-dimensional space (C U_0 for an invertible solution U_0, C = span{1,T} x Cl^0(V^-) = M_2(C) the commutant of e_1,e_2,e_3), contain no "
          "element of definite parity, and every invertible one carries the spin lift of l_int(n) to the left-multiplication "
          "representation L_i -> -e_je_k/2, K_i -> -e_i/2 of the temporal Pauli algebra; rho_oct(K_i) = -(e_i + e_{i+3} omega)/2 "
          "adds the right multiplication (n = e1, (2,3,6)/7)", ok)

# =====================================================================================================================
# r4-22  the Fock picture: temporal = left, spatial = twisted right multiplication
# =====================================================================================================================
ok = True
for r in range(4):
    for I in itertools.combinations((1, 2, 3), r):
        aI = blade(I)
        for k in (1, 2, 3):
            ok &= s[k] * aI * vac == (-1) ** r * aI * e[k] * vac and e[k] * aI * vac == (e[k] * aI) * vac
check(22, "on S = Cl(V^+)|vac>: e_a acts by left multiplication and e_{a+3} by the twisted right multiplication a -> (-1)^{|a|} a e_a; "
          "hence alpha_a(n)(a|vac>) = n (-1)^{|a|} a e_a |vac>", ok)

# =====================================================================================================================
# r4-23  uniqueness of T
# =====================================================================================================================
def gam(n):
    return [tvec(n)] + [s[a] for a in (1, 2, 3)]


ok = True
three = [C1, C3, C7]                       # three linearly independent clocks of the Fisher-Rao octant
ok &= sp.Matrix(three).det() != 0
solA = solve_linear(EU, lambda Y: sum((flat(comm(tvec(n) * Y, gm)) for n in three for gm in gam(n)), []))
ok &= len(solA) == 1 and rank_of([combo(solA[0], EU), T]) == 1
two = [C3, C7]
solA2 = solve_linear(EU, lambda Y: sum((flat(comm(tvec(n) * Y, gm)) for n in two for gm in gam(n)), []))
nn = sp.Matrix(C3).cross(sp.Matrix(C7))
ok &= len(solA2) == 2 and rank_of([combo(v, EU) for v in solA2] + [T, tvec(list(nn)) * s123]) == 2
solB = solve_linear(EU, lambda Y: sum((flat(comm(tvec(n) * Y, gm)) for n in two for gm in lint(n) + [J(n)]), []))
ok &= len(solB) == 2 and rank_of([combo(v, EU) for v in solB] + [T, s123]) == 2
ok &= isZ(nv * Omega(NS) - N2 * s123) and isZ(Jn * Omega(NS) - N2 * omega) and T * omega == -s123
# for two clocks the products 'clock times a square root of -1 commuting with the gamma^mu' are not only +-T:
# at e1 and (0,3,4)/5, Y = (3 T + 4 w e4e5e6)/5 with w = n x n' a unit vector works at both
Cb = [sp.Integer(0), sp.Rational(3, 5), sp.Rational(4, 5)]
wv = list(sp.Matrix(C1).cross(sp.Matrix(Cb)))
Yc = (3 * T + 4 * tvec(wv) * s123) / 5
for n in (C1, Cb):
    xx = tvec(n) * Yc
    ok &= isZ(xx * xx + I8) and all(isZ(comm(xx, gm)) for gm in gam(n))
ok &= rank_of([Yc, T]) == 2
check(23, "{Y : n^-1 Y commutes with gamma^mu(n) for three independent clocks of the Fisher-Rao octant} = R T (for two clocks "
          "n,n' it is span{T, (n x n') e4e5e6}); {Y : n^-1 Y in span{1,Omega_n,J(n),Omega_nJ(n)} for two clocks} = span{T, e4e5e6}; "
          "n Omega_n = |n|^2 e4e5e6 = -|n|^2 T Gamma and J(n) Omega_n = |n|^2 Gamma for every n; for two clocks (e1, (0,3,4)/5) "
          "Y = (3T + 4 (n x n') e4e5e6)/5 makes n^-1 Y a square root of -1 commuting with the gamma^mu_n at both, so (iii) also "
          "needs three clocks", ok)

# =====================================================================================================================
# r4-24  the earlier "covariant mass" e4e5e6
# =====================================================================================================================
tri = [blade(I) for I in itertools.combinations(range(1, 7), 3)]
ok = True
for n in (C1, [0, 1, 0], C3, C7):
    a_ = {k: tvec(n) * s[k] for k in (1, 2, 3)}
    sol = solve_linear(tri, lambda Y: sum((flat(acomm(Y, a_[k])) for k in (1, 2, 3)), []))
    ok &= len(sol) == 2 and rank_of([combo(v, tri) for v in sol] + [T, s123]) == 2
Hp = sum((ps[k - 1] * tvec(C7) * s[k] for k in (1, 2, 3)), Z8) + mm * s123
ok &= isZ(Hp * Hp - (ps[0] ** 2 + ps[1] ** 2 + ps[2] ** 2 + mm ** 2) * I8) and Hp.T == Hp and isZ(comm(Hp, J(C7)))
ok &= J(C7) * s123 == T * Omega(C7) and J(C7) * tvec(C7) == T
ok &= not isZ(J(C1) * s123 - J(C7) * s123)
check(24, "at every clock tested the trivectors anticommuting with alpha_1,alpha_2,alpha_3 are span{e1e2e3, e4e5e6}; beta = e4e5e6 gives "
          "a symmetric H = alpha.p + m e4e5e6 with H^2 = p^2+m^2, [H,J(n)]=0, whose Grassmann kernel J(n)e4e5e6 = T Omega_n is the "
          "pseudoscalar mass and depends on the clock, while beta = n has the clock-independent kernel J(n) n = T", ok)

# =====================================================================================================================
# r4-25  couplings beyond linear order in X: (psi^T T psi) eta(X,X)
# =====================================================================================================================
ok = T.T == -T
for n in (C1, C7):
    ok &= all(isZ(phi(n)[k].T * T + T * phi(n)[k]) for k in KEYS)            # T is a rho_n-invariant form
etaM = sp.diag(1, 1, 1, -1, -1, -1)
ok &= all(isZ(rhoV[k].T * etaM + etaM * rhoV[k]) for k in KEYS)                # eta(X,X) is l_G-invariant
check(25, "(psi^T T psi) eta(X,X) is a covariant derivative-free coupling of a Grassmann psi to X (T antisymmetric and "
          "rho_n-invariant at e1 and (2,3,6)/7, eta(X,X) l_G-invariant): the trilinear couplings are the only covariant "
          "derivative-free ones linear in X, not the only ones", ok)

print(f"# r4: {25 - len(FAILS)} of 25 claims pass; runtime {time.time() - T0:.1f}s")
sys.exit(1 if FAILS else 0)
