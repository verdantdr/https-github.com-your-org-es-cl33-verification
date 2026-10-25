#!/usr/bin/env python3
"""Verification script for section r3 ("X = E + B on the octants") of the two-readings note.

    python3 investigations/two_readings/verify/r3_octants.py

Prints one line 'PASS r3-<n>: <claim>' (or FAIL) per claim and exits 0 iff every check passes.
Algebraic statements are checked in exact integer/rational/symbolic arithmetic (sympy) in the realisation of
Section 2 of the paper (tex/s02_arena.tex, eq. (1)); field equations are checked with sympy on general symbolic fields
E_k(t,x), B_k(t,x) or on first-order jets.  No floating point is used except in r3-6 (a random-rational test,
still exact) -- every comparison below is exact.
"""
import itertools
import sys
from functools import reduce

import sympy as sp

FAILS = []


def check(cid, claim, ok):
    ok = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'} r3-{cid}: {claim}")
    if not ok:
        FAILS.append(cid)


# ----------------------------------------------------------------------------------------------------------------
# The realisation of Cl(3,3) on R^8 (paper, eq. (1)); basis |b1 b2 b3>, index 4 b1 + 2 b2 + b3.
# ----------------------------------------------------------------------------------------------------------------
def kron(*ms):
    return reduce(lambda a, b: sp.kronecker_product(a, b), ms)


sx = sp.Matrix([[0, 1], [1, 0]])
sz = sp.diag(1, -1)
ep = sp.Matrix([[0, -1], [1, 0]])
I2 = sp.eye(2)
I8 = sp.eye(8)
Z8 = sp.zeros(8)
e = [None, kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx), kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
etad = [None, 1, 1, 1, -1, -1, -1]
eta6 = sp.diag(1, 1, 1, -1, -1, -1)


def bits(n):
    return ((n >> 2) & 1, (n >> 1) & 1, n & 1)


def basis(n):
    v = sp.zeros(8, 1)
    v[n] = 1
    return v


def blade(I):
    return reduce(lambda a, b: a * b, [e[a] for a in I], I8)


def comm(a, b):
    return a * b - b * a


def acomm(a, b):
    return a * b + b * a


def vec_coords(v):
    """coefficients of a vector element v = sum_a c_a e_a (checks it is a vector)."""
    c = [sp.Rational(1, 8) * etad[a] * (e[a] * v).trace() for a in range(1, 7)]
    assert sp.simplify(v - sum((c[a - 1] * e[a] for a in range(1, 7)), Z8)) == Z8
    return c


def admat(beta):
    """6x6 matrix of v -> [beta, v] in the basis e_1..e_6 (columns = images)."""
    return sp.Matrix(6, 6, lambda r, c: vec_coords(comm(beta, e[c + 1]))[r])


def conjmat(g):
    """6x6 matrix of v -> g v g^{-1} (g invertible)."""
    gi = g.inv()
    return sp.Matrix(6, 6, lambda r, c: vec_coords(g * e[c + 1] * gi)[r])


omega = blade([1, 2, 3, 4, 5, 6])          # volume element, = Gamma
Gam = omega
om = [None] + [e[i] * e[i + 3] for i in (1, 2, 3)]
ap = [None] + [(e[i] + e[i + 3]) / 2 for i in (1, 2, 3)]
am = [None] + [(e[i] - e[i + 3]) / 2 for i in (1, 2, 3)]
Nop = sum((ap[i] * am[i] for i in (1, 2, 3)), Z8)
T = e[1] * e[2] * e[3]
P = om[1] + om[2] + om[3]
vac = basis(0)
full = basis(7)
f = [None] + [ap[k] * vac for k in (1, 2, 3)]
g = [None, ap[2] * ap[3] * vac, ap[3] * ap[1] * vac, ap[1] * ap[2] * vac]
parN = om[1] * om[2] * om[3]               # (-1)^N
cyc = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
eps = sp.LeviCivita

# coordinates on V: x = (B1,B2,B3,E1,E2,E3) <-> X = sum B_i e_i + E_i e_{i+3}
Gm = sp.Matrix(sp.BlockMatrix([[sp.zeros(3), -sp.eye(3)], [sp.eye(3), sp.zeros(3)]]))   # G e_i = e_{i+3}, G e_{i+3} = -e_i
Kpar = admat(P) / 2                                                                        # paper's grading root
Lb = {i: (e[j + 3] * e[k + 3] - e[j] * e[k]) / 2 for (i, j, k) in cyc}                    # diagonal rotations (RE)
Kb = {i: (e[k] * e[j + 3] - e[j] * e[k + 3]) / 2 for (i, j, k) in cyc}                    # boosts of l_G (paper)
LV = {i: admat(Lb[i]) for i in (1, 2, 3)}
KV = {i: admat(Kb[i]) for i in (1, 2, 3)}


def Phi_mat(a=1, b=0, c=0, d=1):
    """8x6 matrix of X -> (aE + bB).f + (cE + dB).g ; Phi = Phi_mat() = E.f + B.g ."""
    M = sp.zeros(8, 6)
    for k in (1, 2, 3):
        M[:, k - 1] += b * f[k] + d * g[k]      # column of B_k
        M[:, k + 2] += a * f[k] + c * g[k]      # column of E_k
    return M


Phi = Phi_mat()
Bs = sp.symbols('B1:4', real=True)
Es = sp.symbols('E1:4', real=True)
xs = sp.Matrix(list(Bs) + list(Es))
Xel = sum((Bs[i - 1] * e[i] + Es[i - 1] * e[i + 3] for i in (1, 2, 3)), Z8)
Bv = sp.Matrix(Bs)
Ev = sp.Matrix(Es)


def solve_space(unknowns, eqs):
    """dimension of the solution space and a particular solution of a linear system (exact)."""
    A, rhs = sp.linear_eq_to_matrix(eqs, unknowns)
    sol = sp.linsolve((A, rhs), unknowns)
    return A, rhs, sol


# ================================================================================================================
# r3-1  Octants and the Fock grading
# ================================================================================================================
ok = all(om[i].is_diagonal() and om[i] ** 2 == I8 for i in (1, 2, 3))
ok &= all(comm(om[i], om[j]) == Z8 for i in (1, 2, 3) for j in (1, 2, 3))
ok &= all(om[i][n, n] == (-1) ** bits(n)[i - 1] for i in (1, 2, 3) for n in range(8))
signs = {tuple(om[i][n, n] for i in (1, 2, 3)) for n in range(8)}
ok &= len(signs) == 8
car = all(acomm(ap[i], ap[j]) == Z8 and acomm(am[i], am[j]) == Z8 and acomm(ap[i], am[j]) == (I8 if i == j else Z8)
          for i in (1, 2, 3) for j in (1, 2, 3))
Nok = Nop.is_diagonal() and all(Nop[n, n] == sum(bits(n)) == sum(1 for i in (1, 2, 3) if om[i][n, n] == -1)
                                for n in range(8))
omN = all(om[i] == I8 - 2 * ap[i] * am[i] for i in (1, 2, 3))
kern = sp.Matrix.vstack(am[1], am[2], am[3]).nullspace()
vac_ok = len(kern) == 1 and all(am[i] * vac == sp.zeros(8, 1) for i in (1, 2, 3))
fockb = True
for r in range(4):
    for I in itertools.combinations((1, 2, 3), r):
        v = reduce(lambda acc, i: ap[i] * acc, reversed(I), vac)     # a+_{i1} ... a+_{ir} |vac>
        n = sum(1 << (3 - i) for i in I)
        fockb &= (v == basis(n))
check(1, "omega_i are diagonal commuting involutions, omega_i|b>=(-1)^{b_i}|b>, joint spectrum = all 8 sign triples on the "
         "basis lines; a+-_i satisfy the CAR; N=sum a+_i a-_i = number of minus signs, omega_i = 1-2a+_i a-_i; |000>=(+,+,+) "
         "spans the common kernel of the a-_i; a+_{i1}..a+_{ir}|vac> = +|b_I> for i1<..<ir",
      ok and car and Nok and omN and vac_ok and fockb)

# r3-2  Pure spinors: octant sigma is the pure spinor line of N_sigma = span(e_i - sigma_i e_{i+3})
ok = True
for n in range(8):
    sig = [om[i][n, n] for i in (1, 2, 3)]
    cs = sp.symbols('c1:7')
    v = sum((cs[a - 1] * e[a] for a in range(1, 7)), Z8)
    sol = sp.solve(list(v * basis(n)), cs, dict=True)
    A = sp.Matrix(8, 6, lambda r, a: (e[a + 1] * basis(n))[r])
    ann = A.nullspace()                       # coefficient vectors c with (sum c_a e_a)|b> = 0
    Nsig = sp.Matrix.hstack(*[sp.Matrix([1 if a == i else (-sig[i - 1] if a == i + 3 else 0) for a in range(1, 7)])
                              for i in (1, 2, 3)])
    ok &= len(ann) == 3 and sp.Matrix.hstack(*ann).rank() == 3 and sp.Matrix.hstack(*ann, Nsig).rank() == 3
    ok &= (Nsig.T * eta6 * Nsig) == sp.zeros(3)        # totally null, dimension 3: maximal isotropic
check(2, "the annihilator {v in V: v phi_sigma = 0} of each octant line is N_sigma = span(e_i - sigma_i e_{i+3}), a maximal "
         "isotropic three-plane: each octant is the pure spinor line of N_sigma", ok)

# r3-3  The monograph's Prop. 2.9: I = e1..e6 = -omega1 omega2 omega3 with eigenvalue (-1)^w, w = number of plus signs
w = [sum(1 for i in (1, 2, 3) if om[i][n, n] == 1) for n in range(8)]
check(3, "Gamma = e1...e6 = -omega1 omega2 omega3 = -(-1)^N, with eigenvalue (-1)^w on the octant of w plus signs; "
         "(-1)^N = omega1 omega2 omega3 is the chirality gamma_5 = -Gamma",
      Gam == -parN and all(Gam[n, n] == (-1) ** w[n] for n in range(8)) and parN == -Gam
      and all(parN[n, n] == (-1) ** sum(bits(n)) for n in range(8)))

# ================================================================================================================
# r3-4/5  The octant group on spinors
# ================================================================================================================
Ms = sp.symbols('m0:64')
Mg = sp.Matrix(8, 8, Ms)


def implementers(refl):
    """solution space of M e_a = refl(e_a) M for all a, refl = list of 6 signs."""
    eqs = []
    for a in range(1, 7):
        eqs += list(Mg * e[a] - refl[a - 1] * e[a] * Mg)
    A, _ = sp.linear_eq_to_matrix(eqs, Ms)
    return A.nullspace()


ok = True
lifts = {}
for a in range(1, 7):
    refl = [(-1 if b == a else 1) for b in range(1, 7)]
    ns = implementers(refl)
    cand = e[a] * omega
    vecc = sp.Matrix(list(cand))
    ok &= len(ns) == 1 and sp.Matrix.hstack(ns[0], vecc).rank() == 1
    ok &= cand.T * cand == I8 and all(sum(abs(x) for x in cand.row(r)) == 1 for r in range(8))   # signed permutation
    i = a if a <= 3 else a - 3
    ok &= all(cand * om[j] * cand.inv() == (-om[j] if j == i else om[j]) for j in (1, 2, 3))
    # maps the line with signs sigma to the line with sigma_i flipped
    for n in range(8):
        img = cand * basis(n)
        m = [r for r in range(8) if img[r] != 0]
        ok &= len(m) == 1 and m[0] == n ^ (1 << (3 - i))
    lifts[a] = cand
pairlift = all(lifts[i] * lifts[i + 3] in (om[i], -om[i]) for i in (1, 2, 3))
check(4, "for each axis a the automorphisms implementing the reflection e_a -> -e_a form the line R e_a omega (omega=e1..e6); "
         "e_a omega is a signed permutation, conjugates omega_i (i = a or a-3) to -omega_i and fixes the other two, so it maps "
         "the octant sigma to the octant with sigma_i flipped; reflecting both axes of a pair lifts to +-omega_i",
      ok and pairlift)

gens = [lifts[1], lifts[2], lifts[3]]
group = {tuple(I8)}
frontier = [I8]
while frontier:
    new = []
    for x in frontier:
        for gg in gens:
            y = gg * x
            if tuple(y) not in group:
                group.add(tuple(y))
                new.append(y)
    frontier = new
anti = all(acomm(gens[i], gens[j]) == Z8 for i in range(3) for j in range(3) if i != j)
sq = all(gg * gg == -I8 for gg in gens)
lines = set()
regular = True
for el in group:
    M = sp.Matrix(8, 8, el)
    perm = tuple(next(r for r in range(8) if (M * basis(n))[r] != 0) for n in range(8))
    lines.add(perm)
regular = len(lines) == 8 and all(len(set(p)) == 8 for p in lines) and \
    len({p[0] for p in lines}) == 8      # each line is reached from the vacuum by exactly one element of Z2^3
gmats = [sp.Matrix(8, 8, el) for el in group]
centre = [M for M in gmats if all(M * gg == gg * M for gg in gens)]
g123 = gens[0] * gens[1] * gens[2]
centre_ok = len(centre) == 4 and set(map(tuple, centre)) == {tuple(I8), tuple(-I8), tuple(g123), tuple(-g123)}
centre_ok &= g123 == blade([4, 5, 6]) and g123 == -T * omega and g123 * g123 == I8
centre_ok &= all((g123 * basis(n))[7 - n] != 0 for n in range(8))          # maps each octant to its antipode
trivial = [M for M in gmats if all((M * basis(n))[n] != 0 for n in range(8))]
centre_ok &= set(map(tuple, trivial)) == {tuple(I8), tuple(-I8)}
check(5, "the lifts g_i = e_i omega (i=1,2,3) satisfy g_i^2 = -1, anticommute pairwise and generate a group of order 16 "
         "whose centre is {+-1, +-g_1g_2g_3} (order 4), with g_1g_2g_3 = -T omega = e4e5e6 mapping each octant to its antipode; "
         "the kernel of the action on the octant lines is {+-1}, and the action is the regular action of Z_2^3 (simply transitive)",
      len(group) == 16 and anti and sq and regular and centre_ok)

# r3-6  Clocks: the conversion operator B_diag(t) = sum t_i omega_i has the octant of sign(t) as its unique top eigenline
ts = sp.symbols('t1:4', real=True)
Bt = sum((ts[i - 1] * e[i] * e[i + 3] for i in (1, 2, 3)), Z8)
ok = Bt.is_diagonal() and all(Bt[n, n] == sum(om[i][n, n] * ts[i - 1] for i in (1, 2, 3)) for n in range(8))
import random
random.seed(3)
for trial in range(20):
    tv = [sp.Rational(random.randint(1, 50), random.randint(1, 50)) * random.choice((-1, 1)) for _ in range(3)]
    Bn = Bt.subs(dict(zip(ts, tv)))
    vals = [Bn[n, n] for n in range(8)]
    top = max(vals)
    nt = [n for n in range(8) if vals[n] == top]
    target = sum(1 << (3 - i) for i in (1, 2, 3) if tv[i - 1] < 0)
    ok &= nt == [target]
# equivariance under the octant group: g_i B_t g_i^{-1} = B_{R_i t}
for i in (1, 2, 3):
    Rt = {ts[j - 1]: (-ts[j - 1] if j == i else ts[j - 1]) for j in (1, 2, 3)}
    ok &= sp.simplify(lifts[i] * Bt * lifts[i].inv() - Bt.subs(Rt, simultaneous=True)) == Z8
BW = Bt.subs({t: 1 / sp.sqrt(3) for t in ts})
ok &= BW == P / sp.sqrt(3) and max(BW[n, n] for n in range(8)) == sp.sqrt(3) and BW[0, 0] == sp.sqrt(3)
# Fisher-Rao: w_i = x_i^2/4 is invariant under every sign flip of the amplitudes
xa = sp.symbols('x1:4', real=True)
ok &= all(sp.simplify((s * xa[i]) ** 2 / 4 - xa[i] ** 2 / 4) == 0 for s in (-1, 1) for i in range(3))
check(6, "B_diag(t) = sum t_i e_i e_{i+3} (the conversion operator of ES17 Thm 3.6 with C = diag t) has eigenvalue "
         "sum sigma_i t_i on the octant sigma; for t with nonzero entries its unique top eigenline is the octant sign(t) (20 "
         "random rational t); g_i B_t g_i^-1 = B_{R_i t}; for the symmetric clock B = P/sqrt3 with top eigenvalue sqrt3 on "
         "the vacuum; the clock weights w_i = x_i^2/4 do not see the sign of x_i", ok)

# ================================================================================================================
# r3-7/8  The exterior algebra and gl(3)
# ================================================================================================================
Lvecs = [sp.Matrix([1 if a in (i, i + 3) else 0 for a in range(1, 7)]) for i in (1, 2, 3)]
Lpvecs = [sp.Matrix([1 if a == i else (-1 if a == i + 3 else 0) for a in range(1, 7)]) for i in (1, 2, 3)]
Lm = sp.Matrix.hstack(*Lvecs)
Lpm = sp.Matrix.hstack(*Lpvecs)
ok = Lm.T * eta6 * Lm == sp.zeros(3) and Lpm.T * eta6 * Lpm == sp.zeros(3) and sp.Matrix.hstack(Lm, Lpm).rank() == 6
ok &= Lm.T * eta6 * Lpm == 2 * sp.eye(3)                    # eta pairs L with L' as dual spaces
# Lambda^*(L) -> S, l1 ^ ... ^ lr -> (l1 ... lr)|vac>/2^r, is well defined (the l's anticommute) and bijective
ok &= all(acomm(2 * ap[i], 2 * ap[j]) == Z8 for i in (1, 2, 3) for j in (1, 2, 3))
imgs = []
dims = []
for r in range(4):
    sub = [reduce(lambda acc, i: ap[i] * acc, reversed(I), vac) for I in itertools.combinations((1, 2, 3), r)]
    dims.append(sp.Matrix.hstack(*sub).rank())
    ok &= all(Nop * v == r * v for v in sub)
    imgs += sub
ok &= sp.Matrix.hstack(*imgs).rank() == 8 and dims == [1, 3, 3, 1]
# Clifford action: X = sum (B+E) a+ + (B-E) a-
Xfock = sum(((Bs[i - 1] + Es[i - 1]) * ap[i] + (Bs[i - 1] - Es[i - 1]) * am[i] for i in (1, 2, 3)), Z8)
ok &= sp.expand(Xel - Xfock) == Z8
# the temporal subalgebra Cl(3,0) = <e1,e2,e3> acts freely: a -> a|vac> is an isomorphism, grade = N
tb = [blade(list(I)) for r in range(4) for I in itertools.combinations((1, 2, 3), r)]
ok &= sp.Matrix.hstack(*[b * vac for b in tb]).rank() == 8
ok &= all(Nop * (blade(list(I)) * vac) == len(I) * (blade(list(I)) * vac)
          for r in range(4) for I in itertools.combinations((1, 2, 3), r))
ok &= all(blade(list(I)) * vac == reduce(lambda acc, i: ap[i] * acc, reversed(I), vac)
          for r in range(4) for I in itertools.combinations((1, 2, 3), r))
# with l_I := l_{i1} ^ .. ^ l_{ir} <-> a+_I|vac>, Clifford multiplication by l_i = e_i + e_{i+3} is 2 (l_i ^ .) and by
# l'_i = e_i - e_{i+3} the contraction with the form eta(l'_i, .) on L (eta(l'_i, l_j) = 2 delta_ij)
lam = {I: reduce(lambda acc, i: ap[i] * acc, reversed(I), vac) for r in range(4) for I in itertools.combinations((1, 2, 3), r)}
for i in (1, 2, 3):
    form = [(Lpvecs[i - 1].T * eta6 * Lvecs[k])[0] for k in range(3)]
    for I, v in lam.items():
        if i in I:
            wedge = sp.zeros(8, 1)
        else:
            J = tuple(sorted(I + (i,)))
            wedge = (-1) ** sum(1 for k in I if k < i) * lam[J]
        contr = sum(((-1) ** pos * form[k - 1] * lam[tuple(x for x in I if x != k)] for pos, k in enumerate(I)), sp.zeros(8, 1))
        ok &= (e[i] + e[i + 3]) * v == 2 * wedge and (e[i] - e[i + 3]) * v == contr
check(7, "L = span(e_i+e_{i+3}) and L' = span(e_i-e_{i+3}) are maximal isotropic, transverse and eta-dual; "
         "l1^..^lr -> l1..lr|vac>/2^r identifies Lambda^*(L) with S, graded by N with dimensions 1,3,3,1; a vector X = X_L + X_L' "
         "acts as exterior multiplication by 2 X_L plus contraction with eta(X_L', .); "
         "X = sum (B_i+E_i) a+_i + (B_i-E_i) a-_i; moreover a -> a|vac> is an isomorphism Cl(V^+) -> S carrying the "
         "Clifford grade of the temporal blade e_I to N = |I|, e_I|vac> = a+_I|vac>", ok)

# commutant of the grading root in so(3,3)
As = sp.symbols('A0:36')
Am = sp.Matrix(6, 6, As)
eqs = list(Am.T * eta6 + eta6 * Am) + list(Am * Kpar - Kpar * Am)
Aeq, _ = sp.linear_eq_to_matrix(eqs, As)
comm_dim = 36 - Aeq.rank()
gl_biv = {}
for i in (1, 2, 3):
    for j in (1, 2, 3):
        gl_biv[(i, j)] = ap[i] * am[j] - (sp.Rational(1, 2) * I8 if i == j else Z8)
ok = comm_dim == 9
ok &= all(comm(b, Nop) == Z8 for b in gl_biv.values())
# each E_ij is a bivector: its expansion in blades has grade 2 only
def grade_parts(Mx):
    parts = {}
    for r in range(7):
        for I in itertools.combinations(range(1, 7), r):
            B = blade(list(I))
            cf = sp.Rational(1, 8) * (B.inv() * Mx).trace()
            if cf != 0:
                parts.setdefault(r, []).append((I, cf))
    return parts
ok &= all(set(grade_parts(b).keys()) == {2} for b in gl_biv.values())
adg = {k: admat(b) for k, b in gl_biv.items()}
ok &= sp.Matrix.hstack(*[sp.Matrix(list(m)) for m in adg.values()]).rank() == 9
ok &= all(m * Kpar == Kpar * m and m.T * eta6 + eta6 * m == sp.zeros(6) for m in adg.values())
# vector action: on L the matrix unit e_ij, on L' minus its transpose
for (i, j), m in adg.items():
    Eij = sp.zeros(3)
    Eij[i - 1, j - 1] = 1
    ok &= m * Lm == Lm * Eij and m * Lpm == -Lpm * Eij.T
# spin action on Lambda^w: Lambda^w(A) - (1/2) tr A, checked on all eight basis states for every matrix unit
for (i, j), b in gl_biv.items():
    for r in range(4):
        for I in itertools.combinations((1, 2, 3), r):
            st = reduce(lambda acc, k: ap[k] * acc, reversed(I), vac)
            # derivation: replace one factor a+_l (l in I, l == j) by a+_i
            expect = sp.zeros(8, 1)
            for pos, l in enumerate(I):
                if l == j:
                    J = list(I)
                    J[pos] = i
                    expect += reduce(lambda acc, k: ap[k] * acc, reversed(J), vac)
            if i == j:
                expect -= sp.Rational(1, 2) * st
            ok &= b * st == expect
check(8, "the commutant of the grading root K in so(3,3) has dimension 9 and is the adjoint image of the nine bivectors "
         "a+_i a-_j - delta_ij/2 (= a+_i a-_j for i != j, -omega_i/2 for i = j), which conserve N; they act on L by the matrix "
         "unit e_ij and on L' by -e_ji (gl(3,R) = gl(L)), and on Lambda^w(L) as Lambda^w(A) - (1/2) tr A", ok)

# ================================================================================================================
# r3-9  The diagonal rotations and the boosts of l_G in Fock form
# ================================================================================================================
ok = True
for (i, j, k) in cyc:
    Ri = sp.zeros(3)
    for a in (1, 2, 3):
        for b in (1, 2, 3):
            Ri[a - 1, b - 1] = eps(i, b, a)        # delta x = e_i x x  on each triple
    ok &= LV[i] == sp.diag(Ri, Ri)
    ok &= LV[i] * Gm == Gm * LV[i] and LV[i] * Kpar == Kpar * LV[i]
    paired = admat(e[j] * e[k] + e[j + 3] * e[k + 3])
    ok &= paired[:3, :3] == -paired[3:, 3:] and paired * Gm != Gm * paired and paired * Kpar != Kpar * paired
    ok &= Lb[i] == ap[k] * am[j] - ap[j] * am[k]
    ok &= Kb[i] == am[j] * am[k] - ap[j] * ap[k]
    ok &= KV[i] == Gm * LV[i]
    ok &= Lb[i] ** 3 == -Lb[i]                     # spectrum {0, +-i}: exp(2 pi L_i) = 1 on S
    ok &= all(Lb[i] * f[m] == sum((eps(i, m, n) * f[n] for n in (1, 2, 3)), sp.zeros(8, 1)) for m in (1, 2, 3))
    ok &= all(Lb[i] * g[m] == sum((eps(i, m, n) * g[n] for n in (1, 2, 3)), sp.zeros(8, 1)) for m in (1, 2, 3))
br = all(comm(Lb[a], Lb[b]) == sum((eps(a, b, c) * Lb[c] for c in (1, 2, 3)), Z8) and
         comm(Lb[a], Kb[b]) == sum((eps(a, b, c) * Kb[c] for c in (1, 2, 3)), Z8) and
         comm(Kb[a], Kb[b]) == -sum((eps(a, b, c) * Lb[c] for c in (1, 2, 3)), Z8) for a in (1, 2, 3) for b in (1, 2, 3))
inv = sp.Matrix.vstack(Lb[1], Lb[2], Lb[3]).nullspace()
ok &= br and len(inv) == 2 and sp.Matrix.hstack(*inv, vac, full).rank() == 2
# l_G cap gl(3) = so(3)_Delta
lGV = [LV[i] for i in (1, 2, 3)] + [KV[i] for i in (1, 2, 3)]
glV = list(adg.values())
Ml = sp.Matrix.hstack(*[sp.Matrix(list(m)) for m in lGV])
Mg9 = sp.Matrix.hstack(*[sp.Matrix(list(m)) for m in glV])
inter = Ml.rank() + Mg9.rank() - sp.Matrix.hstack(Ml, Mg9).rank()
ok &= inter == 3
check(9, "L_i = (e_{j+3}e_{k+3} - e_j e_k)/2 rotates both triples alike (dB = e_i x B, dE = e_i x E) and commutes with G and K, "
         "while e_j e_k + e_{j+3}e_{k+3} rotates them oppositely and commutes with neither; L_i = a+_k a-_j - a+_j a-_k, "
         "K_i = a-_j a-_k - a+_j a+_k, K_i = G L_i on V, the brackets of ES17 Thm 12.3 hold; on S L_i^3 = -L_i (so "
         "exp(2 pi L_i) = 1), L_i f_m = eps_imn f_n, L_i g_m = eps_imn g_n, the invariant spinors are span(vac, full); "
         "l_G cap gl(3) = so(3)_Delta (dimension 3)", ok)

# ================================================================================================================
# r3-10/11  The field-octant map
# ================================================================================================================
Hs = sp.symbols('h0:48')
Hm = sp.Matrix(8, 6, Hs)
eqs = []
for i in (1, 2, 3):
    eqs += list(Lb[i] * Hm - Hm * LV[i])
Ah, _ = sp.linear_eq_to_matrix(eqs, Hs)
dim_eq = 48 - Ah.rank()
# with E -> N=1 and B -> N=2 imposed
mask = []
for r in range(8):
    for c in range(6):
        Nr = sum(bits(r))
        if (c < 3 and Nr != 2) or (c >= 3 and Nr != 1):
            mask.append(Hm[r, c])
Ah2, _ = sp.linear_eq_to_matrix(eqs + mask, Hs)
dim_eq2 = 48 - Ah2.rank()
ok = dim_eq == 4 and dim_eq2 == 2
ok &= all(Lb[i] * Phi == Phi * LV[i] for i in (1, 2, 3)) and Phi.T * Phi == sp.eye(6)
psi = Phi * xs
ok &= psi == sum((Es[k - 1] * f[k] + Bs[k - 1] * g[k] for k in (1, 2, 3)), sp.zeros(8, 1))
check(10, "the linear maps V -> S intertwining so(3)_Delta (L_i on V, its spin lift on S) form a 4-dimensional space; those "
          "with E on weight 1 and B on weight 2 form the 2-dimensional family Phi_{alpha,beta} = alpha E.f + beta B.g, "
          "f_k = a+_k|vac>, g_m = (1/2) eps_mjk a+_j a+_k|vac>; Phi = Phi_{1,1} is an isometry onto N=1 + N=2", ok)

ok = True
for (i, j), b in gl_biv.items():
    A3 = sp.zeros(3)
    A3[i - 1, j - 1] = 1
    tr = A3.trace()
    # induced action on the image: E -> (A - tr/2) E, B -> (tr/2 - A^T) B
    induced = sp.diag((sp.Rational(1, 2) * tr) * sp.eye(3) - A3.T, A3 - (sp.Rational(1, 2) * tr) * sp.eye(3))
    ok &= b * Phi == Phi * induced
Asym = gl_biv[(1, 2)] + gl_biv[(2, 1)]
Aasym = gl_biv[(1, 2)] - gl_biv[(2, 1)]
ok &= Asym * Phi != Phi * admat(Asym) and Aasym * Phi == Phi * admat(Aasym)
# the vector action of the symmetric part mixes E and B: B -> A E, E -> A B
Avec = admat(Asym)
S12 = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
ok &= Avec == sp.Matrix(sp.BlockMatrix([[sp.zeros(3), S12], [S12, sp.zeros(3)]]))
check(11, "the spin lift of gl(3) preserves the image of Phi and induces E -> (A - tr A/2)E (Lambda^1 (x) det^-1/2), "
          "B -> (tr A/2 - A^T)B (contragredient, Lambda^2 (x) det^-1/2); for the antisymmetric part this equals the vector "
          "action, but the symmetric part acts on V by B -> A E, E -> A B (it is a set of boosts) and Phi does not intertwine it",
      ok)

# ================================================================================================================
# r3-12  Parity
# ================================================================================================================
ok = conjmat(T) == eta6
ns = implementers([1, 1, 1, -1, -1, -1])
ok &= len(ns) == 1 and sp.Matrix.hstack(ns[0], sp.Matrix(list(T))).rank() == 1
ok &= all(T * basis(n) in (basis(7 - n), -basis(7 - n)) for n in range(8))      # octant sigma -> -sigma
ok &= all(T * f[k] == g[k] and T * g[k] == -f[k] for k in (1, 2, 3)) and T * vac == full and T * full == -vac
ok &= T * Phi == -Phi * Gm
ok &= parN * Phi == Phi * eta6
ok &= conjmat(parN) == -sp.eye(6)
ok &= conjmat(e[1]) == sp.diag(1, -1, -1, -1, -1, -1)
ok &= e[1] * f[1] == vac                                                            # e1 maps f_1 onto the vacuum
check(12, "Ad_T = eta on V (frame parity) and T = e1e2e3 is the unique element (up to scale) implementing it; T maps each "
          "octant to its antipode (T f_k = g_k, T g_k = -f_k, T|vac> = |full>, T|full> = -|vac>) and T Phi = -Phi G (on the octants the lift of parity is minus the Hodge star); "
          "(-1)^N Phi = Phi eta (the weight parity, i.e. the chirality, implements E polar / B axial) while Ad_{(-1)^N} = -1 on V; "
          "the Dirac parity gamma^0 = e1 of ES17 Thm 5.3 acts on the frame as (B1,-B2,-B3;-E) and maps f_1 to the vacuum", ok)

# ================================================================================================================
# r3-13  The field point on the poles
# ================================================================================================================
ok = sp.expand(Xel * vac - sum(((Bs[i - 1] + Es[i - 1]) * f[i] for i in (1, 2, 3)), sp.zeros(8, 1))) == sp.zeros(8, 1)
ok &= sp.expand(Xel * full - sum(((Bs[i - 1] - Es[i - 1]) * g[i] for i in (1, 2, 3)), sp.zeros(8, 1))) == sp.zeros(8, 1)
ok &= sp.expand(Xel * full - sum(((-1) ** (i - 1) * (Bs[i - 1] - Es[i - 1]) * basis(7 ^ (1 << (3 - i))) for i in (1, 2, 3)),
                                 sp.zeros(8, 1))) == sp.zeros(8, 1)
ok &= sp.expand(Xel * Xel - sum(((Bs[i] + Es[i]) * (Bs[i] - Es[i]) for i in range(3))) * I8) == Z8
eG4 = (sp.eye(6) + Gm) / sp.sqrt(2)                       # exp(pi G/4)
ok &= sp.expand(Xel * (vac + full) - sp.sqrt(2) * Phi * eG4 * xs) == sp.zeros(8, 1)
# no spinor chi gives Phi as a Clifford multiplication
chi = sp.Matrix(sp.symbols('q0:8'))
eqs = []
for a in range(1, 7):
    eqs += list(e[a] * chi - Phi[:, a - 1])
_, _, sol = solve_space(list(chi), eqs)
ok &= sol == sp.EmptySet
# rotation-equivariant Clifford maps X -> X chi need chi in span(vac, full); none separates E from B
eqs = []
for i in (1, 2, 3):
    eqs += list(Lb[i] * chi)
_, _, sol2 = solve_space(list(chi), eqs)
gen = list(sol2)[0]
free = sorted(gen.free_symbols, key=str)
ok &= len(free) == 2
cmap = sp.Matrix.hstack(*[e[a] * sp.Matrix(gen) for a in range(1, 7)])
sepeq = [cmap[r, c] for r in range(8) for c in range(6) if (c < 3 and sum(bits(r)) != 2) or (c >= 3 and sum(bits(r)) != 1)]
ok &= set(sp.solve(sepeq, free, dict=True)[0].values()) == {0}
check(13, "X|vac> = sum (B_i+E_i) f_i, X|full> = sum (B_i-E_i) g_i = sum (-1)^{i-1}(B_i-E_i)|111 xor delta_i>, "
          "X^2 = sum (B_i+E_i)(B_i-E_i) = eta(X,X); X(|vac>+|full>) = sqrt2 Phi(e^{pi G/4} X); no spinor chi has X chi = Phi(X) "
          "for all X, and the rotation-equivariant Clifford multiplications X -> X chi (chi in span(vac,full)) never put E and B "
          "on separate weights", ok)

# ================================================================================================================
# r3-14/15  Boosts and duality: the even obstruction and the odd intertwiners
# ================================================================================================================
even_blades = [blade(list(I)) for r in (0, 2, 4, 6) for I in itertools.combinations(range(1, 7), r)]
odd_blades = [blade(list(I)) for r in (1, 3, 5) for I in itertools.combinations(range(1, 7), r)]
ok = len(even_blades) == 32 and all(comm(b, parN) == Z8 for b in even_blades) and all(acomm(b, parN) == Z8 for b in odd_blades)
cs32 = sp.symbols('u0:32')


def intertwiner_space(blades, target):
    Q = sum((cs32[n] * blades[n] for n in range(len(blades))), Z8)
    eqs = list(Q * Phi - Phi * target)
    A, rhs = sp.linear_eq_to_matrix(eqs, cs32[:len(blades)])
    sol = sp.linsolve((A, rhs), cs32[:len(blades)])
    return A, rhs, sol


for tgt in [KV[1], KV[2], KV[3], Gm]:
    _, _, sol = intertwiner_space(even_blades, tgt)
    ok &= sol == sp.EmptySet
check(14, "every even element of Cl(3,3) commutes with (-1)^N (maps weight w into w, w+-2) and every odd one anticommutes; "
          "no even Q satisfies Q Phi = Phi K_k (k=1,2,3) or Q Phi = Phi G", ok)

ok = True
vec_blades = [e[a] for a in range(1, 7)]
for k in (1, 2, 3):
    (i_, j_, l_) = [c for c in cyc if c[0] == k][0]
    Qk = -T * Lb[k]
    ok &= Qk * Phi == Phi * KV[k]
    ok &= Qk == -e[k] * (I8 - om[j_] * om[l_]) / 2 and Qk == -(e[k] + e[k + 3] * omega) / 2
    ok &= set(grade_parts(Qk).keys()) == {1, 5}
    A, rhs, sol = intertwiner_space(odd_blades, KV[k])
    ok &= sol != sp.EmptySet and 32 - A.rank() == 8
    Hfull = sp.Matrix(8, 8, sp.symbols('z0:64'))
    Af, _ = sp.linear_eq_to_matrix(list(Hfull * Phi - Phi * KV[k]), list(Hfull))
    ok &= 64 - Af.rank() == 16
    _, _, solv = intertwiner_space(vec_blades, KV[k])
    ok &= solv == sp.EmptySet
ok &= (-T) * Phi == Phi * Gm
_, _, solvG = intertwiner_space(vec_blades, Gm)
ok &= solvG == sp.EmptySet
# Q_k(a|vac>) = -(1/2)[e_k, a]|vac> for a in the temporal algebra: the Pauli-algebra commutator
for k in (1, 2, 3):
    Qk = -T * Lb[k]
    ok &= all(Qk * (b * vac) == -(e[k] * b - b * e[k]) * vac / 2 for b in tb)
# group level: exp(s ad K_3) e_1 = cosh s e_1 + sinh s e_5 mixes the weights for every s != 0
s = sp.symbols('s', real=True)
expK = sp.simplify((s * KV[3]).exp())
ok &= sp.simplify(expK[:, 0] - sp.Matrix([sp.cosh(s), 0, 0, 0, sp.sinh(s), 0])) == sp.zeros(6, 1)
check(15, "the odd operators Q_k = -T L_k = -e_k (1 - omega_j omega_l)/2 = -(e_k + e_{k+3} omega)/2 (grades 1 and 5) satisfy "
          "Q_k Phi = Phi K_k, and -T Phi = Phi G; the solutions of Q Phi = Phi K_k form an affine space of dimension 16, of "
          "dimension 8 among odd elements, containing no Clifford vector, and no Clifford vector implements G; on "
          "S = Cl(V^+)|vac>, Q_k a|vac> = -(1/2)[e_k, a]|vac> (the Pauli-algebra boost); exp(s K_3) e_1 = cosh s e_1 + sinh s e_5",
      ok)

# ================================================================================================================
# r3-16  Two representations of l_G on S
# ================================================================================================================
ro_L = {i: Lb[i] for i in (1, 2, 3)}
ro_K = {i: -T * Lb[i] for i in (1, 2, 3)}
ok = all(comm(ro_L[a], ro_L[b]) == sum((eps(a, b, c) * ro_L[c] for c in (1, 2, 3)), Z8) and
         comm(ro_L[a], ro_K[b]) == sum((eps(a, b, c) * ro_K[c] for c in (1, 2, 3)), Z8) and
         comm(ro_K[a], ro_K[b]) == -sum((eps(a, b, c) * ro_L[c] for c in (1, 2, 3)), Z8) for a in (1, 2, 3) for b in (1, 2, 3))
Cas_oct = sum((ro_K[i] ** 2 - ro_L[i] ** 2 for i in (1, 2, 3)), Z8)
Cas_spin = sum((Kb[i] ** 2 - Lb[i] ** 2 for i in (1, 2, 3)), Z8)
Cas_V = sum((KV[i] ** 2 - LV[i] ** 2 for i in (1, 2, 3)), sp.zeros(6))
ok &= Cas_oct == sp.diag(*[4 if sum(bits(n)) in (1, 2) else 0 for n in range(8)])
ok &= Cas_spin == 3 * I8 and Cas_V == 4 * sp.eye(6)
eqs_spin = []
eqs_oct = []
for i in (1, 2, 3):
    eqs_spin += list(Lb[i] * Hm - Hm * LV[i]) + list(Kb[i] * Hm - Hm * KV[i])
    eqs_oct += list(ro_L[i] * Hm - Hm * LV[i]) + list(ro_K[i] * Hm - Hm * KV[i])
A1, _ = sp.linear_eq_to_matrix(eqs_spin, Hs)
A2, _ = sp.linear_eq_to_matrix(eqs_oct, Hs)
ok &= 48 - A1.rank() == 0
ok &= 48 - A2.rank() == 2 and sp.Matrix.hstack(sp.Matrix(list(Phi)), sp.Matrix(list(Phi * Gm))).rank() == 2
ok &= all(Kb[k] * f[k] == -full and Kb[k] * vac == -g[k] for k in (1, 2, 3))
check(16, "rho_oct(L_k) = L_k, rho_oct(K_k) = -T L_k is a representation of l_G on S with Casimir 4 on the field octants and 0 "
          "on the poles, whose intertwiners V -> S are span(Phi, Phi G); the spin lift has Casimir 3 on all of S and admits no "
          "nonzero intertwiner V -> S; the spin-lifted boosts carry f_k to -|full> and |vac> to -g_k", ok)

# ================================================================================================================
# r3-17/18  Energies and invariant forms
# ================================================================================================================
psiE = Phi * sp.Matrix([0, 0, 0] + list(Es))
psiB = Phi * sp.Matrix(list(Bs) + [0, 0, 0])
B2 = (Bv.T * Bv)[0]
E2 = (Ev.T * Ev)[0]
EB = (Ev.T * Bv)[0]
zc = [Bs[i] + sp.I * Es[i] for i in range(3)]
q = sp.expand(sum(z * z for z in zc))
ok = sp.expand((psi.T * psi)[0] - (E2 + B2)) == 0
ok &= sp.expand((psi.T * parN * psi)[0] - sp.re(q)) == 0 and sp.expand(sp.re(q) - (B2 - E2)) == 0
ok &= sp.expand((psiE.T * T * psiB)[0] + EB) == 0 and sp.expand(sp.im(q) - 2 * EB) == 0
ok &= sp.expand((psi.T * T * psi)[0]) == 0
ok &= sp.expand((psi.T * T * Gam * psi)[0] - sp.im(q)) == 0
ok &= sp.expand(((psi.T * parN * psi)[0]) ** 2 + 4 * ((psiE.T * T * psiB)[0]) ** 2 - sp.expand(q * sp.conjugate(q))) == 0
# SI form: Phi_SI = E.f + c B.g, u = eps0/2 |psi|^2 = eps0/2 (|E|^2 + c^2 |B|^2) = eps0/2 |z_c|^2
cc, e0 = sp.symbols('c epsilon0', positive=True)
psiSI = Phi * sp.Matrix([cc * b for b in Bs] + list(Es))
ok &= sp.expand(e0 / 2 * (psiSI.T * psiSI)[0] - e0 / 2 * (E2 + cc ** 2 * B2)) == 0
qc = sp.expand(sum((cc * Bs[i] + sp.I * Es[i]) ** 2 for i in range(3)))
ok &= sp.expand((psiSI.T * parN * psiSI)[0] - sp.re(qc)) == 0
check(17, "for psi = Phi(X): |psi|^2 = |E|^2+|B|^2 = 2 E6; psi^T (-1)^N psi = |B|^2-|E|^2 = Re q; psi_E^T T psi_B = -E.B = -Im q/2; "
          "psi^T T psi = 0; psi^T T Gamma psi = 2 E.B = Im q; (psi^T(-1)^N psi)^2 + 4 (psi_E^T T psi_B)^2 = |q|^2 = 4 E0^2; in SI "
          "with psi = Phi(cB; E): (eps0/2)|psi|^2 = u and psi^T(-1)^N psi = Re q_c", ok)

Ws = sp.symbols('w0:64')
Wm = sp.Matrix(8, 8, Ws)
eqs = []
for a, b in itertools.combinations(range(1, 7), 2):
    beta = e[a] * e[b]
    eqs += list(beta.T * Wm + Wm * beta)
Aw, _ = sp.linear_eq_to_matrix(eqs, Ws)
inv_dim = 64 - Aw.rank()
ok = inv_dim == 2 and all((beta.T * M + M * beta) == Z8 for M in (T, T * Gam)
                          for beta in [e[a] * e[b] for a, b in itertools.combinations(range(1, 7), 2)])
ok &= T.T == -T and (T * Gam).T == T * Gam


# Chevalley-Mukai pairing on Lambda^*(L): (phi, psi) = [tau(phi) ^ psi]_top, basis a+_I|vac> <-> a+_I
def wedge_sign(I, J):
    if set(I) & set(J):
        return 0
    seq = list(I) + list(J)
    inv = sum(1 for x in range(len(seq)) for y in range(x + 1, len(seq)) if seq[x] > seq[y])
    return (-1) ** inv


subsets = [I for r in range(4) for I in itertools.combinations((1, 2, 3), r)]
idx = {I: sum(1 << (3 - i) for i in I) for I in subsets}
mukai_ok = True
for I in subsets:
    for J in subsets:
        if len(I) + len(J) == 3:
            val = (-1) ** (len(I) * (len(I) - 1) // 2) * wedge_sign(I, J)
        else:
            val = 0
        mukai_ok &= val == -(basis(idx[I]).T * T * basis(idx[J]))[0]
ok &= mukai_ok
# pairing of pure spinors: nonzero iff N_sigma cap N_tau = 0 iff tau = -sigma
pp = True
for n in range(8):
    for m in range(8):
        sn = [om[i][n, n] for i in (1, 2, 3)]
        sm = [om[i][m, m] for i in (1, 2, 3)]
        inter_dim = sum(1 for i in range(3) if sn[i] == sm[i])
        pp &= ((basis(n).T * T * basis(m))[0] != 0) == (inter_dim == 0)
ok &= pp
# (-1)^N: invariant under the six compact bivectors, not under any of the nine boosts; invariant under rho_oct
compact = [e[a] * e[b] for a, b in itertools.combinations(range(1, 7), 2) if (a <= 3) == (b <= 3)]
boosts = [e[a] * e[b] for a in (1, 2, 3) for b in (4, 5, 6)]
ok &= all(beta.T * parN + parN * beta == Z8 for beta in compact) and all(beta.T * parN + parN * beta != Z8 for beta in boosts)
ok &= all(M.T * parN + parN * M == Z8 for M in list(ro_L.values()) + list(ro_K.values()))
# the Euclidean norm (= 2 E6 on the image of Phi) is invariant under the compact bivectors and rho_oct(L_k), not under the
# symmetric rho_oct(K_k) = Q_k: boosts change the energy density, |Phi(e^{s K_3} e_1)|^2 = cosh 2s
ok &= all(beta.T + beta == Z8 for beta in compact) and all(M.T + M == Z8 for M in ro_L.values())
ok &= all(M.T == M and M != Z8 for M in ro_K.values())
s_ = sp.symbols('s', real=True)
v_s = Phi * (KV[3] * s_).exp()[:, 0]
ok &= sp.simplify(((v_s.T * v_s)[0] - sp.cosh(2 * s_)).rewrite(sp.exp)) == 0
check(18, "the Spin(3,3)-invariant bilinear forms on S form the 2-dimensional span of T (antisymmetric) and T Gamma (symmetric); "
          "the Chevalley-Mukai pairing [tau(phi) ^ psi]_top on Lambda^*(L) equals -phi^T T psi on all 64 basis pairs; two octants "
          "pair to a nonzero value iff their null planes N_sigma, N_tau are transverse iff they are antipodal; (-1)^N is invariant "
          "under the compact bivectors and rho_oct, under none of the nine boosts; the Euclidean norm is invariant under the "
          "compact bivectors and rho_oct(L_k) but not under rho_oct(K_k) = Q_k, which are symmetric (|Phi(e^{sK_3}e_1)|^2 = cosh 2s)", ok)

# ================================================================================================================
# r3-19..24  Maxwell on the octants
# ================================================================================================================
t, x1, x2, x3 = sp.symbols('t x1 x2 x3', real=True)
X3 = (x1, x2, x3)
EF = [sp.Function(f'E{k}')(t, x1, x2, x3) for k in (1, 2, 3)]
BF = [sp.Function(f'B{k}')(t, x1, x2, x3) for k in (1, 2, 3)]
EFv = sp.Matrix(EF)
BFv = sp.Matrix(BF)


def curl(F):
    return sp.Matrix([sp.diff(F[2], x2) - sp.diff(F[1], x3), sp.diff(F[0], x3) - sp.diff(F[2], x1),
                      sp.diff(F[1], x1) - sp.diff(F[0], x2)])


def div(F):
    return sum(sp.diff(F[i], X3[i]) for i in range(3))


def grad(phi):
    return sp.Matrix([sp.diff(phi, x) for x in X3])


def apply_op(ops, psi_):
    """sum_i ops[i] d_i psi_ for a list of 8x8 matrices ops (i = 1,2,3)."""
    return sum((ops[i] * psi_.diff(X3[i]) for i in range(3)), sp.zeros(8, 1))


def fsum(coeffs, basis_vecs):
    return sum((coeffs[k] * basis_vecs[k + 1] for k in range(3)), sp.zeros(8, 1))


psiF = fsum(EF, f) + fsum(BF, g)
d_ops = [ap[i] for i in (1, 2, 3)]
del_ops = [-am[i] for i in (1, 2, 3)]
ok = all(acomm(ap[i], ap[j]) == Z8 and acomm(am[i], am[j]) == Z8 for i in (1, 2, 3) for j in (1, 2, 3))   # d^2 = delta^2 = 0
gen8 = sp.Matrix([sp.Function(f'p{n}')(t, x1, x2, x3) for n in range(8)])
ok &= sp.expand(apply_op(d_ops, apply_op(d_ops, gen8))) == sp.zeros(8, 1)
ok &= sp.expand(apply_op(del_ops, apply_op(del_ops, gen8))) == sp.zeros(8, 1)
ok &= all(d_ops[i] - del_ops[i] == e[i + 1] and d_ops[i] + del_ops[i] == e[i + 4] for i in range(3))
Dop = [e[1], e[2], e[3]]
Dp = [e[4], e[5], e[6]]
lap = lambda v: sum((v.diff(x, 2) for x in X3), sp.zeros(8, 1))
ok &= sp.expand(apply_op(Dop, apply_op(Dop, gen8)) - lap(gen8)) == sp.zeros(8, 1)
ok &= sp.expand(apply_op(Dp, apply_op(Dp, gen8)) + lap(gen8)) == sp.zeros(8, 1)
ok &= sp.expand(apply_op(d_ops, fsum(EF, f)) - fsum(list(curl(EFv)), g)) == sp.zeros(8, 1)
ok &= sp.expand(apply_op(del_ops, fsum(BF, g)) - fsum(list(curl(BFv)), f)) == sp.zeros(8, 1)
ok &= sp.expand(apply_op(del_ops, fsum(EF, f)) + div(EFv) * vac) == sp.zeros(8, 1)
ok &= sp.expand(apply_op(d_ops, fsum(BF, g)) - div(BFv) * full) == sp.zeros(8, 1)
check(19, "d = sum a+_i d_i and delta = -sum a-_i d_i satisfy d^2 = delta^2 = 0, d - delta = sum e_i d_i, d + delta = sum "
          "e_{i+3} d_i, (sum e_i d_i)^2 = Laplacian, (sum e_{i+3} d_i)^2 = -Laplacian; d(E.f) = (curl E).g, delta(B.g) = "
          "(curl B).f, delta(E.f) = -(div E)|vac>, d(B.g) = (div B)|full>: the exterior derivative and codifferential of "
          "Lambda^*(R^3) under f_k <-> dx^k, g_m <-> *dx^m", ok)

res = psiF.diff(t) - apply_op(Dop, psiF)
expect = -div(EFv) * vac - div(BFv) * full + fsum(list(EFv.diff(t) + curl(BFv)), f) + fsum(list(BFv.diff(t) - curl(EFv)), g)
ok = sp.expand(res - expect) == sp.zeros(8, 1)
# with the sources of ES17 eq. (16): d_t z = -i curl z - i j, div z = i rho  <=>  (d_t - D) Phi(X) = -(rho|vac> + j.f)
rho = sp.Function('rho')(t, x1, x2, x3)
jF = sp.Matrix([sp.Function(f'j{k}')(t, x1, x2, x3) for k in (1, 2, 3)])
zF = BFv + sp.I * EFv
RS1 = zF.diff(t) + sp.I * curl(zF) + sp.I * jF
RS2 = div(zF) - sp.I * rho
lhs = res + rho * vac + fsum(list(jF), f)


def re_im(expr):
    """real and imaginary parts of an expression linear in I with real (generic) fields."""
    ex = sp.expand(expr)
    rp = ex.subs(sp.I, 0)
    return rp, sp.expand((ex - rp) / sp.I)


# the components of lhs are exactly the real/imaginary parts of the residuals of the RS system
ok &= sp.expand(lhs[0] + re_im(RS2)[1]) == 0 and sp.expand(lhs[7] + re_im(RS2)[0]) == 0
ok &= all(sp.expand((lhs.T * f[k])[0] - re_im(RS1[k - 1])[1]) == 0 for k in (1, 2, 3))
ok &= all(sp.expand((lhs.T * g[k])[0] - re_im(RS1[k - 1])[0]) == 0 for k in (1, 2, 3))
ok &= all(sp.expand(re_im(RS1[k])[1] - (EFv.diff(t) + curl(BFv) + jF)[k]) == 0 and
          sp.expand(re_im(RS1[k])[0] - (BFv.diff(t) - curl(EFv))[k]) == 0 for k in range(3))
ok &= sp.expand(re_im(RS2)[1] - (div(EFv) - rho)) == 0 and sp.expand(re_im(RS2)[0] - div(BFv)) == 0
check(20, "for general fields, (d_t - sum e_i d_i) Phi(X) = -(div E)|vac> - (div B)|full> + (d_t E + curl B).f + "
          "(d_t B - curl E).g; hence (d_t - D) Phi(X) = -(rho|vac> + j.f) is exactly the system of ES17 eq. (16) "
          "(div E = rho, div B = 0, d_t E = -curl B - j, d_t B = curl E): Gauss laws on the poles, evolution on the field octants",
      ok)

box = gen8.diff(t, 2) - lap(gen8)
lhsw = (gen8.diff(t) + apply_op(Dop, gen8))
lhsw = lhsw.diff(t) - apply_op(Dop, lhsw)
ok = sp.expand(lhsw - box) == sp.zeros(8, 1)
lhs_e = (gen8.diff(t) + apply_op(Dp, gen8))
lhs_e = lhs_e.diff(t) - apply_op(Dp, lhs_e)
ok &= sp.expand(lhs_e - (gen8.diff(t, 2) + lap(gen8))) == sp.zeros(8, 1)
# extended system with pole components and constraint propagation
phi0 = sp.Function('phi0')(t, x1, x2, x3)
phi3 = sp.Function('phi3')(t, x1, x2, x3)
psiX = psiF + phi0 * vac + phi3 * full
resX = psiX.diff(t) - apply_op(Dop, psiX)
expX = (phi0.diff(t) - div(EFv)) * vac + (phi3.diff(t) - div(BFv)) * full + \
    fsum(list(EFv.diff(t) + curl(BFv) - grad(phi0)), f) + fsum(list(BFv.diff(t) - curl(EFv) - grad(phi3)), g)
ok &= sp.expand(resX - expX) == sp.zeros(8, 1)
check(21, "(d_t + D)(d_t - D) = d_t^2 - Laplacian on all eight components for D = sum e_i d_i, while D' = sum e_{i+3} d_i gives "
          "d_t^2 + Laplacian; with pole components phi0, phi3 the equation (d_t - D) psi = 0 reads d_t phi0 = div E, "
          "d_t phi3 = div B, d_t E = -curl B + grad phi0, d_t B = curl E + grad phi3", ok)

# uniqueness: rotation-equivariant maps Phi_M compatible with Maxwell (first-order jets)
aa, bb, cc_, dd = sp.symbols('a b c d')
jet = {}
for nm in ('E', 'B'):
    for k in (1, 2, 3):
        for mu in ('t', '1', '2', '3'):
            jet[(nm, k, mu)] = sp.Symbol(f'{nm}{k}_{mu}')


def jvec(nm, mu):
    return sp.Matrix([jet[(nm, k, mu)] for k in (1, 2, 3)])


def jcurl(nm):
    return sp.Matrix([jet[(nm, 3, '2')] - jet[(nm, 2, '3')], jet[(nm, 1, '3')] - jet[(nm, 3, '1')],
                      jet[(nm, 2, '1')] - jet[(nm, 1, '2')]])


PhiM = Phi_mat(aa, bb, cc_, dd)
dX = {mu: sp.Matrix(list(jvec('B', mu)) + list(jvec('E', mu))) for mu in ('t', '1', '2', '3')}
resM = PhiM * dX['t'] - sum((e[i] * PhiM * dX[str(i)] for i in (1, 2, 3)), sp.zeros(8, 1))
subsM = {}
for k in (1, 2, 3):
    subsM[jet[('E', k, 't')]] = -jcurl('B')[k - 1]
    subsM[jet[('B', k, 't')]] = jcurl('E')[k - 1]
subsM[jet[('E', 1, '1')]] = -jet[('E', 2, '2')] - jet[('E', 3, '3')]
subsM[jet[('B', 1, '1')]] = -jet[('B', 2, '2')] - jet[('B', 3, '3')]
resM = sp.expand(resM.subs(subsM, simultaneous=True))
coeff_eqs = []
for r in range(8):
    poly = sp.Poly(resM[r], *[v for v in jet.values()])
    coeff_eqs += list(poly.coeffs())
solM = sp.solve(coeff_eqs, [cc_, dd], dict=True)
ok = solM == [{cc_: -bb, dd: aa}]
th = sp.symbols('theta', real=True)
eth = sp.cos(th) * sp.eye(6) + sp.sin(th) * Gm
ok &= sp.simplify(Phi * eth - Phi_mat(sp.cos(th), sp.sin(th), -sp.sin(th), sp.cos(th))) == sp.zeros(8, 6)
ok &= Phi * Gm == Phi_mat(0, 1, -1, 0)
check(22, "among the rotation-equivariant maps Phi_M = (aE+bB).f + (cE+dB).g, (d_t - D) Phi_M(X) vanishes on every first-order "
          "jet of a source-free Maxwell field iff c = -b, d = a, i.e. Phi_M = r Phi o e^{theta G}; E and B lie on single "
          "weights iff theta is a multiple of pi/2: Phi (theta = 0) or Phi o G (theta = pi/2, E on weight 2)", ok)

# plane wave of ES17 Example 12.11 (real part), and a non-solution
k_ = sp.symbols('k', positive=True)
ph = k_ * (x3 - t)
zw = sp.Matrix([sp.exp(sp.I * ph), sp.I * sp.exp(sp.I * ph), 0])
Bw = sp.Matrix([sp.re(sp.expand_complex(zz)) for zz in zw])
Ew = sp.Matrix([sp.im(sp.expand_complex(zz)) for zz in zw])
psiw = fsum(list(Ew), f) + fsum(list(Bw), g)
ok = sp.simplify(psiw.diff(t) - apply_op(Dop, psiw)) == sp.zeros(8, 1)
psibad = fsum(list(Ew), f) - fsum(list(Bw), g)
ok &= sp.simplify(psibad.diff(t) - apply_op(Dop, psibad)) != sp.zeros(8, 1)
check(23, "the circularly polarised wave z = (1,i,0) e^{ik(x3-t)} of ES17 Ex. 12.11 gives (d_t - D) Phi(X) = 0, while "
          "the same wave with B reversed does not", ok)

# SI form and the textbook (Hestenes) form
e0s, mu0 = sp.symbols('epsilon0 mu0', positive=True)
rho_s = sp.Symbol('rho')
js = sp.Matrix(sp.symbols('j1:4'))
psiSIj = PhiM.subs({aa: 1, bb: 0, cc_: 0, dd: cc})
resSI = psiSIj * dX['t'] / cc - sum((e[i] * psiSIj * dX[str(i)] for i in (1, 2, 3)), sp.zeros(8, 1))
subsSI = {}
for k in (1, 2, 3):
    subsSI[jet[('E', k, 't')]] = -cc ** 2 * jcurl('B')[k - 1] - js[k - 1] / e0s
    subsSI[jet[('B', k, 't')]] = jcurl('E')[k - 1]
subsSI[jet[('E', 1, '1')]] = rho_s / e0s - jet[('E', 2, '2')] - jet[('E', 3, '3')]
subsSI[jet[('B', 1, '1')]] = -jet[('B', 2, '2')] - jet[('B', 3, '3')]
resSI = sp.expand(resSI.subs(subsSI, simultaneous=True))
ok = sp.simplify(resSI + (cc * rho_s * vac + fsum(list(js), f)) / (e0s * cc)) == sp.zeros(8, 1)
# textbook variables, HL c = 1: psi_t = E.f + B_t.g, (d_t + D) psi_t = rho|vac> - j.f
Phit = Phi_mat(1, 0, 0, -1)                    # columns in terms of the frame B = -B_t
rest = Phit * dX['t'] + sum((e[i] * Phit * dX[str(i)] for i in (1, 2, 3)), sp.zeros(8, 1))
subst = {}
for k in (1, 2, 3):        # textbook: d_t E = curl B_t - j, d_t B_t = -curl E, div E = rho ; B_t = -B
    subst[jet[('E', k, 't')]] = -jcurl('B')[k - 1] - js[k - 1]
    subst[jet[('B', k, 't')]] = jcurl('E')[k - 1]
subst[jet[('E', 1, '1')]] = rho_s - jet[('E', 2, '2')] - jet[('E', 3, '3')]
subst[jet[('B', 1, '1')]] = -jet[('B', 2, '2')] - jet[('B', 3, '3')]
rest = sp.expand(rest.subs(subst, simultaneous=True))
ok &= sp.expand(rest - (rho_s * vac - fsum(list(js), f))) == sp.zeros(8, 1)
ok &= Phit == -parN * Phi and T * Phit == Phit * Gm
# Phi_t(X) = (sum_k E_k e_k + B_tk T e_k)|vac>: Hestenes' F = E + I B_t with I = T = e1e2e3
Fp = sum((Es[k - 1] * e[k] - Bs[k - 1] * T * e[k] for k in (1, 2, 3)), Z8)
ok &= sp.expand(Fp * vac - Phit * xs) == sp.zeros(8, 1)
ok &= all(T * e[k] == e[[c for c in cyc if c[0] == k][0][1]] * e[[c for c in cyc if c[0] == k][0][2]] for k in (1, 2, 3))
ok &= all(comm(T, e[k]) == Z8 for k in (1, 2, 3)) and T * T == -I8
check(24, "SI: ((1/c) d_t - D) Phi(cB; E) = -(c rho|vac> + j.f)/(eps0 c) on Maxwell jets; textbook variables: "
          "(d_t + D)(E.f + B_t.g) = rho|vac> - j.f, and E.f + B_t.g = (E + T B_t)|vac> with E, B_t written on e1,e2,e3 and "
          "T = e1e2e3 central in Cl(V^+), T^2 = -1, T e_i = e_j e_k: Hestenes' (d_t + grad)(E + I B) = rho - j", ok)

# ================================================================================================================
# r3-25/26  Pooling structure and the poles under the interior algebra
# ================================================================================================================
ok = True
for i in (1, 2, 3):
    for n in range(8):
        img = e[i] * basis(n)
        nz = [r for r in range(8) if img[r] != 0]
        ok &= nz == [n ^ (1 << (3 - i))]
    ok &= e[i + 3] == e[i] * om[i]
# component of D psi on octant sigma involves only the three neighbours of sigma
for n in range(8):
    row = sum((e[i][n, :] for i in (1, 2, 3)), sp.zeros(1, 8))
    ok &= {c for c in range(8) if row[c] != 0} == {n ^ 4, n ^ 2, n ^ 1}
check(25, "each e_i maps the octant sigma to the octant with sigma_i flipped (the generator R_i of the octant group) and "
          "e_{i+3} = e_i omega_i; hence the octant-sigma component of D psi = sum e_i d_i psi collects d_i of the three "
          "neighbouring octants only", ok)

lint = [e[a] * e[b] for a, b in itertools.combinations((1, 4, 5, 6), 2)]
lintV = sp.Matrix.hstack(*[sp.Matrix(list(L_)) for L_ in lint])


def in_span(M, span):
    return sp.Matrix.hstack(span, sp.Matrix(list(M))).rank() == span.rank()


ok = True
any_inv = False
cs6 = sp.symbols('y0:6')
comb = sum((cs6[m] * lint[m] for m in range(6)), Z8)
for n in range(8):
    v = basis(n)
    s1 = om[1][n, n]
    img = comb * v
    A, _ = sp.linear_eq_to_matrix([img[r] for r in range(8) if r != n], cs6)
    stab = A.nullspace()                      # beta in l_int with beta v in R v
    expected = [e[1] * e[4], (e[1] - s1 * e[4]) * e[5], (e[1] - s1 * e[4]) * e[6]]
    Sm = sp.Matrix.hstack(*[sp.Matrix(list(sum((st[m] * lint[m] for m in range(6)), Z8))) for st in stab])
    ok &= len(stab) == 3 and all(in_span(x, Sm) for x in expected)
    ok &= all((x * v) == sp.zeros(8, 1) for x in expected[1:])
    # the null rotations fix the null vector n_sigma = e1 - sigma_1 e4 of the interior four-plane
    nv = e[1] - s1 * e[4]
    ok &= (nv * nv == Z8) and all(comm(x, nv) == Z8 for x in expected[1:])
    any_inv |= all(all((L_ * v)[r] == 0 for r in range(8) if r != n) for L_ in lint)
# g_2, g_3 commute with l_int; g_1 (and g_4) normalise it; together they permute the octants transitively
ok &= all(comm(lifts[i], L_) == Z8 for i in (2, 3) for L_ in lint)
ok &= all(in_span(lifts[i] * L_ * lifts[i].inv(), lintV) for i in (1, 4) for L_ in lint)
ok &= not any_inv and lint[0] == om[1]
check(26, "under the interior Lorentz algebra l_int (bivectors of e1,e4,e5,e6) no octant line is invariant; the stabiliser of the "
          "line sigma is three-dimensional, spanned by e1e4 = omega_1 and the null rotations (e1 - sigma_1 e4)e5, "
          "(e1 - sigma_1 e4)e6, which annihilate it and fix the null vector e1 - sigma_1 e4; it depends on sigma_1 only, and the "
          "lifts g_2, g_3 commute with l_int while g_1 normalises it: l_int distinguishes no octant, in particular neither pole", ok)

print()
if FAILS:
    print(f"r3_octants.py: {len(FAILS)} FAIL ({', '.join(map(str, FAILS))})")
    sys.exit(1)
print("r3_octants.py: all checks pass")
sys.exit(0)
