#!/usr/bin/env python3
"""Independent referee check of section r3 ("X = E + B on the octants") of the two-readings note.

    python3 investigations/two_readings/verify/s3_check.py

Prints one line 'PASS s3-<id>: <claim>' (or FAIL) per claim and exits 0 iff every check passes.  Written without
reference to the derivation script r3_octants.py: the generators are built from the bit-flip formulas of Appendix A of
the paper (and only compared with the Kronecker products of its eq. (1)), the octant data are read off integer
matrices, every dimension count is the exact rank of an integer linear system, and the field equations are checked as
exact residual identities on general symbolic fields E_k(t,x), B_k(t,x), rho, j (sympy), not on jets, except for the
uniqueness statement s3-25, which needs the solution space of the source-free system and uses first-order jets.
Where the section was found to be wrong, the check is of the corrected statement (s3-7, s3-19, s3-16b).
"""
import itertools
import random
import sys

import sympy as sp

FAILS = []


def check(cid, claim, ok):
    ok = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'} s3-{cid}: {claim}")
    if not ok:
        FAILS.append(cid)


R = sp.Rational
I8 = sp.eye(8)
Z8 = sp.zeros(8, 8)


# ---------------------------------------------------------------------------------------------------------------
# Generators from the bit rules  e_i|b> = (-1)^{b_1+..+b_{i-1}} |b xor delta_i>,  e_{i+3}|b> = (-1)^{b_1+..+b_i} |...>
# ---------------------------------------------------------------------------------------------------------------
def bits(n):
    return [(n >> 2) & 1, (n >> 1) & 1, n & 1]


def num(b):
    return 4 * b[0] + 2 * b[1] + b[2]


def gen(a):
    M = sp.zeros(8, 8)
    i = (a - 1) % 3
    for n in range(8):
        b = bits(n)
        s = sum(b[:i]) + (b[i] if a > 3 else 0)
        bb = b[:]
        bb[i] ^= 1
        M[num(bb), n] = (-1) ** s
    return M


e = [None] + [gen(a) for a in range(1, 7)]
eta = [None, 1, 1, 1, -1, -1, -1]


def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = sp.kronecker_product(out, m)
    return out


sx, sz, ep, I2 = sp.Matrix([[0, 1], [1, 0]]), sp.diag(1, -1), sp.Matrix([[0, -1], [1, 0]]), sp.eye(2)
ekron = [None, kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx), kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
cliff = all(e[a] * e[b] + e[b] * e[a] == (2 * eta[a] * I8 if a == b else Z8) for a in range(1, 7) for b in range(1, 7))
check(1, "the bit-flip rules of Appendix A give the six matrices of eq. (1) of ES17 (sx,sz,ep Kronecker products) and they "
         "satisfy e_a e_b + e_b e_a = 2 eta_ab", all(e[a] == ekron[a] for a in range(1, 7)) and cliff)


def prod(ms):
    out = I8
    for m in ms:
        out = out * m
    return out


def blade(I):
    return prod([e[a] for a in I])


def comm(a, b):
    return a * b - b * a


om = blade(range(1, 7))                       # omega = e1...e6 = Gamma
Gam = om
T = blade([1, 2, 3])
w = [None] + [e[i] * e[i + 3] for i in (1, 2, 3)]   # pole operators
ap = [None] + [(e[i] + e[i + 3]) / 2 for i in (1, 2, 3)]
am = [None] + [(e[i] - e[i + 3]) / 2 for i in (1, 2, 3)]
Nop = sum((ap[i] * am[i] for i in (1, 2, 3)), Z8)
parN = w[1] * w[2] * w[3]


def perm_parity(seq):
    return (-1) ** sum(1 for x in range(len(seq)) for y in range(x + 1, len(seq)) if seq[x] > seq[y])


def ket(n):
    v = sp.zeros(8, 1)
    v[n] = 1
    return v


vac, full = ket(0), ket(7)


def sigma_of(n):
    return tuple((-1) ** bit for bit in bits(n))


# ---------------------------------------------------------------------------------------------------------------
# s3-2, s3-3  octants, Fock structure
# ---------------------------------------------------------------------------------------------------------------
diag_ok = all(w[i] == sp.diag(*[(-1) ** bits(n)[i - 1] for n in range(8)]) for i in (1, 2, 3))
labels = {sigma_of(n) for n in range(8)}
check(2, "omega_i = e_i e_{i+3} are diagonal with omega_i|b> = (-1)^{b_i}|b>; the joint eigenlines are the eight basis lines "
         "and carry the eight distinct sign triples", diag_ok and len(labels) == 8)

car = all(ap[i] * ap[j] + ap[j] * ap[i] == Z8 and am[i] * am[j] + am[j] * am[i] == Z8 and
          ap[i] * am[j] + am[j] * ap[i] == (I8 if i == j else Z8) for i in (1, 2, 3) for j in (1, 2, 3))
wform = all(w[i] == I8 - 2 * ap[i] * am[i] for i in (1, 2, 3))
Ncount = Nop == sp.diag(*[sum(bits(n)) for n in range(8)])
vacker = all(am[i] * vac == sp.zeros(8, 1) for i in (1, 2, 3))
commonker = sp.Matrix.vstack(am[1], am[2], am[3]).nullspace()
creat = all(prod([ap[i] for i in I]) * vac == ket(sum(1 << (3 - i) for i in I))
            for r in range(4) for I in itertools.combinations((1, 2, 3), r))
check(3, "a+-_i = (e_i +- e_{i+3})/2 satisfy the CAR, omega_i = 1 - 2 a+_i a-_i, N = sum a+_i a-_i = number of minus signs; "
         "|000> (signs +++) spans the common kernel of the a-_i, a+_{i1}..a+_{ir}|vac> = +|b_I> (i1<..<ir), |111> has N = 3",
      car and wform and Ncount and vacker and len(commonker) == 1 and creat and (Nop * full == 3 * full))


# annihilators
def vecel(c):
    return sum((c[a - 1] * e[a] for a in range(1, 7)), Z8)


ann_ok = True
for n in range(8):
    s = sigma_of(n)
    A = sp.Matrix.hstack(*[e[a] * ket(n) for a in range(1, 7)])      # columns: e_a |b>
    ns = A.nullspace()
    Nsig = sp.Matrix.hstack(*[sp.Matrix([1 if a == i else (-s[i - 1] if a == i + 3 else 0) for a in range(1, 7)])
                              for i in (1, 2, 3)])
    ann_ok &= len(ns) == 3 and sp.Matrix.hstack(*ns, Nsig).rank() == 3
    iso = all(sum(eta[a] * Nsig[a - 1, p] * Nsig[a - 1, q] for a in range(1, 7)) == 0 for p in range(3) for q in range(3))
    ann_ok &= iso
check(4, "for every octant sigma the annihilator {v in V: v|b> = 0} is the maximal isotropic three-plane "
         "N_sigma = span(e_i - sigma_i e_{i+3}) (rank computation): each octant is a pure spinor line", ann_ok)

gam_ok = Gam == -w[1] * w[2] * w[3] and parN == -Gam
wplus = all((Gam * ket(n))[n] == (-1) ** (3 - sum(bits(n))) for n in range(8))
check(5, "Gamma = e1...e6 = -omega1 omega2 omega3, (-1)^N = omega1 omega2 omega3 = -Gamma (the chirality gamma_5 = -Gamma), "
         "and Gamma = (-1)^w on the octant with w = 3 - N plus signs", gam_ok and wplus)


# ---------------------------------------------------------------------------------------------------------------
# s3-6, s3-7  octant group
# ---------------------------------------------------------------------------------------------------------------
def implementers(images):
    """dimension of {M in M_8 : M e_a = images[a] M for all a} and a basis."""
    syms = sp.symbols('m0:64')
    M = sp.Matrix(8, 8, syms)
    eqs = []
    for a in range(1, 7):
        eqs += list(M * e[a] - images[a] * M)
    A, _ = sp.linear_eq_to_matrix(eqs, syms)
    ns = A.nullspace()
    return [sp.Matrix(8, 8, list(v)) for v in ns]


lift_ok = True
g = {}
for a in range(1, 7):
    imgs = {b: (-e[b] if b == a else e[b]) for b in range(1, 7)}
    sols = implementers(imgs)
    ga = e[a] * om
    lift_ok &= len(sols) == 1 and sp.Matrix.hstack(sols[0].reshape(64, 1), ga.reshape(64, 1)).rank() == 1
    signed_perm = all(sum(1 for x in ga[:, c] if x != 0) == 1 and all(x in (0, 1, -1) for x in ga[:, c]) for c in range(8))
    lift_ok &= signed_perm
    i = (a - 1) % 3 + 1
    lift_ok &= all(ga * w[j] * ga.inv() == (-w[j] if j == i else w[j]) for j in (1, 2, 3))
    g[a] = ga
pair = all(g[i] * g[i + 3] == -w[i] for i in (1, 2, 3))
check(6, "the reflection e_a -> -e_a (a = 1..6) is implemented exactly by the multiples of g_a = e_a omega (Skolem-Noether, "
         "solution space of dimension 1), g_a is a signed permutation, flips omega_i for a in {i, i+3} and fixes the other two; "
         "g_i g_{i+3} = -omega_i fixes every octant", lift_ok and pair)

gens = [g[1], g[2], g[3]]
group = {tuple(I8)}
front = [I8]
while front:
    new = []
    for x in front:
        for y in gens:
            z = x * y
            if tuple(z) not in group:
                group.add(tuple(z))
                new.append(z)
    front = new
G16 = [sp.Matrix(8, 8, list(t)) for t in group]
sq = all(x * x == -I8 for x in gens)
anti = all(gens[p] * gens[q] == -gens[q] * gens[p] for p in range(3) for q in range(3) if p != q)
centre = [x for x in G16 if all(x * y == y * x for y in G16)]
g123 = g[1] * g[2] * g[3]
cent_ok = len(centre) == 4 and all(any(c == s * m for s in (1, -1) for m in (I8, g123)) for c in centre)
g123_is = g123 == blade([4, 5, 6]) and g123 * g123 == I8
antip = all(g123 * ket(n) in (ket(7 - n), -ket(7 - n)) for n in range(8))


def line_perm(x):
    return tuple(next(r for r in range(8) if x[r, n] != 0) for n in range(8))


kernel = [x for x in G16 if line_perm(x) == tuple(range(8))]
H8 = {tuple(I8)}
fr = [I8]
while fr:
    nw = []
    for x in fr:
        for y in gens[:2]:
            zz = x * y
            if tuple(zz) not in H8:
                H8.add(tuple(zz))
                nw.append(zz)
    fr = nw
H8m = [sp.Matrix(8, 8, list(tt)) for tt in H8]
quat = len(H8m) == 8 and [x for x in H8m if x * x == I8 and x != I8] == [-I8] and tuple(g123) not in H8
perms = {line_perm(x) for x in G16}
simply = len(perms) == 8 and len({p[0] for p in perms}) == 8
check(7, "CORRECTED: g_1, g_2, g_3 square to -1, anticommute pairwise and generate a group of order 16, Q_8 x Z_2 (<g_1,g_2> is quaternionic, g_1g_2g_3 lies outside it), whose centre "
         "is {+-1, +-g_1g_2g_3} of order 4, not {+-1}: g_1g_2g_3 = e4e5e6 is central, squares to +1 and maps each octant to its "
         "antipode; the kernel of the action on the octant lines is {+-1}, and the quotient Z_2^3 acts simply transitively",
      len(G16) == 16 and sq and anti and cent_ok and g123_is and antip and len(kernel) == 2 and simply and quat)

# clock
random.seed(3)
clock_ok = True
for _ in range(20):
    x = [R(random.choice([-1, 1]) * random.randint(1, 9), random.randint(1, 5)) for _ in range(3)]
    Bx = sum((x[i - 1] * w[i] for i in (1, 2, 3)), Z8)
    ev = [Bx[n, n] for n in range(8)]
    clock_ok &= Bx == sp.diag(*ev) and all(ev[n] == sum(s * xi for s, xi in zip(sigma_of(n), x)) for n in range(8))
    top = max(ev)
    tops = [n for n in range(8) if ev[n] == top]
    clock_ok &= tops == [num([0 if xi > 0 else 1 for xi in x])]
    for i in (1, 2, 3):
        xr = [(-xi if j == i else xi) for j, xi in zip((1, 2, 3), x)]
        clock_ok &= g[i] * Bx * g[i].inv() == sum((xr[j - 1] * w[j] for j in (1, 2, 3)), Z8)
P = w[1] + w[2] + w[3]
BW = P / sp.sqrt(3)
sym_ok = sp.simplify(BW * vac - sp.sqrt(3) * vac) == sp.zeros(8, 1) and max(BW[n, n] for n in range(8)) == sp.sqrt(3)
xs = sp.symbols('x1:4', real=True)
wts = [xi ** 2 / 4 for xi in xs]
blind = all(sp.simplify(wt.subs(xs[i], -xs[i]) - wt) == 0 for i, wt in enumerate(wts))
check(8, "B_x = sum x_i e_i e_{i+3} = sum x_i omega_i is diagonal with eigenvalue sum sigma_i x_i on the octant sigma; for x with "
         "nonzero entries its unique top eigenline is the octant sign(x) (20 random rationals); g_i B_x g_i^-1 = B_{R_i x}; "
         "B_W = P/sqrt3 has top eigenvalue sqrt3 on |vac>; the weights x_i^2/4 do not see the signs", clock_ok and sym_ok and blind)

# ---------------------------------------------------------------------------------------------------------------
# s3-9  exterior algebra
# ---------------------------------------------------------------------------------------------------------------
etaM = sp.diag(1, 1, 1, -1, -1, -1)
Lb = [sp.Matrix([1 if a in (i, i + 3) else 0 for a in range(1, 7)]) for i in (1, 2, 3)]
Lp = [sp.Matrix([1 if a == i else (-1 if a == i + 3 else 0) for a in range(1, 7)]) for i in (1, 2, 3)]
iso = all((u.T * etaM * v)[0] == 0 for u in Lb for v in Lb) and all((u.T * etaM * v)[0] == 0 for u in Lp for v in Lp)
dual = all((Lb[i].T * etaM * Lp[j])[0] == (2 if i == j else 0) for i in range(3) for j in range(3))
transv = sp.Matrix.hstack(*Lb, *Lp).rank() == 6
Bs = sp.symbols('B1:4', real=True)
Es = sp.symbols('E1:4', real=True)
X = vecel(list(Bs) + list(Es))
Xfock = sum(((Bs[i - 1] + Es[i - 1]) * ap[i] + (Bs[i - 1] - Es[i - 1]) * am[i] for i in (1, 2, 3)), Z8)
# Clifford action of l in L is 2 (l wedge); of l' in L' contraction with eta(l', .)
basisL = {I: prod([ap[i] for i in I]) * vac for r in range(4) for I in itertools.combinations((1, 2, 3), r)}


def wedge_ell(i, I):
    """ell_i wedge ell_I in the sorted basis: (sign, J) or (0, None)."""
    if i in I:
        return 0, None
    J = tuple(sorted(I + (i,)))
    return (-1) ** sum(1 for k in I if k < i), J


def contract(alpha, I):
    """contraction of ell_I with the functional alpha (alpha[k] = alpha(ell_k)): sum_s (-1)^s alpha(ell_{i_s}) ell_{I minus i_s}."""
    out = sp.zeros(8, 1)
    for s, k in enumerate(I):
        J = tuple(x for x in I if x != k)
        out += (-1) ** s * alpha[k - 1] * basisL[J]
    return out


ext_ok = True
for i in (1, 2, 3):
    ell = e[i] + e[i + 3]
    ellp = e[i] - e[i + 3]
    alpha = [(Lp[i - 1].T * etaM * Lb[k])[0] for k in range(3)]       # eta(l'_i, l_k) = 2 delta
    for I, v in basisL.items():
        sgn, J = wedge_ell(i, I)
        ext_ok &= ell * v == (2 * sgn * basisL[J] if J else sp.zeros(8, 1))
        ext_ok &= ellp * v == contract(alpha, I)
grade_ok = all(blade(I) * vac == basisL[I] for I in basisL) and \
    sp.Matrix.hstack(*[blade(I) * vac for I in basisL]).rank() == 8
check(9, "L = span(e_i + e_{i+3}), L' = span(e_i - e_{i+3}) are maximal isotropic, transverse, eta(l_i, l'_j) = 2 delta_ij; "
         "X = sum (B_i+E_i) a+_i + (B_i-E_i) a-_i; CORRECTED NORMALISATION: under ell_I <-> a+_I|vac> a vector of L acts as "
         "exterior multiplication by TWICE itself and a vector l' of L' as contraction with eta(l',.); e_I|vac> = a+_I|vac>, "
         "so a -> a|vac> is an isomorphism Cl(V^+) -> S carrying the blade grade to N",
      iso and dual and transv and sp.expand(X - Xfock) == Z8 and ext_ok and grade_ok)


# ---------------------------------------------------------------------------------------------------------------
# helpers on V: matrix of ad_beta on V (basis e1..e6), bivector basis
# ---------------------------------------------------------------------------------------------------------------
def coords(Y):
    return sp.Matrix([R(1, 8) * eta[a] * (e[a] * Y).trace() for a in range(1, 7)])


def admat(beta):
    return sp.Matrix.hstack(*[coords(comm(beta, e[a])) for a in range(1, 7)])


biv = [e[a] * e[b] for a in range(1, 7) for b in range(a + 1, 7)]
Kpar = admat(P) / 2
Gm = sp.Matrix([[0, 0, 0, -1, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, -1],
                [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])     # G e_i = e_{i+3}, G e_{i+3} = -e_i
Kexp = sp.Matrix(6, 6, lambda r, c: -1 if abs(r - c) == 3 else 0)
check("10a", "the grading root is K = (1/2) ad_P with K e_i = -e_{i+3}, K e_{i+3} = -e_i (ES17 Prop. 3.3, Lemma 3.4); the "
             "para-complex swap e_i <-> e_{i+3} of the earlier version is -K; G is minus K composed with Ad_T",
      Kpar == Kexp and Gm == -Kpar * sp.Matrix.hstack(*[coords(T * e[a] * T.inv()) for a in range(1, 7)]))

cs = sp.symbols('c0:15')
gen_biv = sum((cs[k] * biv[k] for k in range(15)), Z8)
adg = admat(gen_biv)
A, _ = sp.linear_eq_to_matrix(list(adg * Kpar - Kpar * adg), cs)
comm_dim = 15 - A.rank()
Eij = {(i, j): ap[i] * am[j] - (R(1, 2) * I8 if i == j else Z8) for i in (1, 2, 3) for j in (1, 2, 3)}
Eij_ok = all(admat(Eij[i, j]) * Kpar == Kpar * admat(Eij[i, j]) and comm(Eij[i, j], Nop) == Z8 for (i, j) in Eij)
Eij_ok &= sp.Matrix.hstack(*[admat(Eij[k]).reshape(36, 1) for k in Eij]).rank() == 9
Eij_ok &= all(Eij[i, i] == -w[i] / 2 for i in (1, 2, 3))
# action on L (basis ap) and L' (basis am): [E_ij, a+_k] = delta_jk a+_i, [E_ij, a-_k] = -delta_ik a-_j
Eij_ok &= all(comm(Eij[i, j], ap[k]) == (ap[i] if j == k else Z8) and comm(Eij[i, j], am[k]) == (-am[j] if i == k else Z8)
              for (i, j) in Eij for k in (1, 2, 3))
# spin lift of A acts on Lambda^w as Lambda^w(A) - tr A / 2 : test on a random integer matrix A
random.seed(7)
Am = sp.Matrix(3, 3, lambda r, c: random.randint(-4, 4))
lift = sum((Am[i - 1, j - 1] * Eij[i, j] for (i, j) in Eij), Z8)
lam_ok = True
for I, v in basisL.items():
    # Lambda^w(A) on ell_I: sum over positions, replace ell_k by A ell_k
    out = sp.zeros(8, 1)
    for pos, k in enumerate(I):
        for i in (1, 2, 3):
            newI = list(I)
            newI[pos] = i
            if len(set(newI)) < len(newI):
                continue
            sgn = perm_parity(newI)
            out += Am[i - 1, k - 1] * sgn * basisL[tuple(sorted(newI))]
    lam_ok &= lift * v == out - Am.trace() / 2 * v
check("10b", "the elements of so(3,3) commuting with K form a 9-dimensional space, the adjoint image of E_ij = a+_i a-_j - delta_ij/2 "
             "(E_ii = -omega_i/2), which commute with N, act on L by the matrix unit e_ij and on L' by -e_ji (gl(3) = gl(L)); the "
             "spin lift of A acts on Lambda^w(L) as Lambda^w(A) - tr(A)/2 (random integer A)",
      comm_dim == 9 and Eij_ok and lam_ok)

# ---------------------------------------------------------------------------------------------------------------
# s3-11  diagonal rotations
# ---------------------------------------------------------------------------------------------------------------
cyc = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
Lr = {i: (e[j + 3] * e[k + 3] - e[j] * e[k]) / 2 for (i, j, k) in cyc}
Kb = {i: (e[k] * e[j + 3] - e[j] * e[k + 3]) / 2 for (i, j, k) in cyc}
LV = {i: admat(Lr[i]) for i in (1, 2, 3)}
KV = {i: admat(Kb[i]) for i in (1, 2, 3)}


def cross_mat(i):
    M = sp.zeros(3, 3)
    for m in (1, 2, 3):
        for n in (1, 2, 3):
            M[n - 1, m - 1] = sp.LeviCivita(i, m, n)
    return M


rot_ok = all(LV[i] == sp.diag(cross_mat(i), cross_mat(i)) for i in (1, 2, 3))
rot_ok &= all(LV[i] * Gm == Gm * LV[i] and LV[i] * Kpar == Kpar * LV[i] for i in (1, 2, 3))
paired_ok = all((lambda M: M * Gm != Gm * M and M * Kpar != Kpar * M and M[:3, :3] == -M[3:, 3:])(admat(e[j] * e[k] + e[j + 3] * e[k + 3]))
                for (i, j, k) in cyc)
fock_ok = all(Lr[i] == ap[k] * am[j] - ap[j] * am[k] and Kb[i] == am[j] * am[k] - ap[j] * ap[k] for (i, j, k) in cyc)
KGL = all(KV[i] == Gm * LV[i] for i in (1, 2, 3))
br = all(comm(Lr[i], Lr[j]) == sp.LeviCivita(i, j, k) * Lr[k] and comm(Lr[i], Kb[j]) == sp.LeviCivita(i, j, k) * Kb[k]
         and comm(Kb[i], Kb[j]) == -sp.LeviCivita(i, j, k) * Lr[k] for (i, j, k) in cyc)
cube = all(Lr[i] ** 3 == -Lr[i] for i in (1, 2, 3))
f = [None] + [ap[k] * vac for k in (1, 2, 3)]
gq = [None] + [sum((R(1, 2) * sp.LeviCivita(m, j, k) * ap[j] * ap[k] * vac for j in (1, 2, 3) for k in (1, 2, 3)), sp.zeros(8, 1))
               for m in (1, 2, 3)]
inv_sp = sp.Matrix.vstack(Lr[1], Lr[2], Lr[3]).nullspace()
inv_ok = len(inv_sp) == 2 and sp.Matrix.hstack(*inv_sp, vac, full).rank() == 2
act_ok = all(Lr[i] * f[m] == sum((sp.LeviCivita(i, m, n) * f[n] for n in (1, 2, 3)), sp.zeros(8, 1)) and
             Lr[i] * gq[m] == sum((sp.LeviCivita(i, m, n) * gq[n] for n in (1, 2, 3)), sp.zeros(8, 1))
             for i in (1, 2, 3) for m in (1, 2, 3))
# lG cap gl(3): bivectors commuting with G and with K
A2, _ = sp.linear_eq_to_matrix(list(adg * Kpar - Kpar * adg) + list(adg * Gm - Gm * adg), cs)
cap_dim = 15 - A2.rank()
check(11, "L_i = (e_{j+3}e_{k+3} - e_j e_k)/2 acts by dB = e_i x B, dE = e_i x E and commutes with G and K, while "
          "e_j e_k + e_{j+3}e_{k+3} turns the triples oppositely and commutes with neither; L_i = a+_k a-_j - a+_j a-_k, "
          "K_i = a-_j a-_k - a+_j a+_k, K_i = G L_i on V, the brackets of ES17 Thm 12.3; on S L_i^3 = -L_i, the common kernel "
          "is span(vac, full), L_i f_m = eps_imn f_n, L_i g_m = eps_imn g_n; l_G cap gl(3) has dimension 3 (= so(3)_Delta)",
      rot_ok and paired_ok and fock_ok and KGL and br and cube and inv_ok and act_ok and cap_dim == 3)


# ---------------------------------------------------------------------------------------------------------------
# s3-12, 13  the field-octant map
# ---------------------------------------------------------------------------------------------------------------
def intertwiners(Vops, Sops):
    syms = sp.symbols('n0:48')
    M = sp.Matrix(8, 6, syms)
    eqs = []
    for Vo, So in zip(Vops, Sops):
        eqs += list(M * Vo - So * M)
    A, _ = sp.linear_eq_to_matrix(eqs, syms)
    return [sp.Matrix(8, 6, list(v)) for v in A.nullspace()]


Phi = sp.Matrix.hstack(gq[1], gq[2], gq[3], f[1], f[2], f[3])     # columns: B1 B2 B3 E1 E2 E3
so3_int = intertwiners([LV[i] for i in (1, 2, 3)], [Lr[i] for i in (1, 2, 3)])
Nproj1 = sp.diag(*[1 if sum(bits(n)) == 1 else 0 for n in range(8)])
Nproj2 = sp.diag(*[1 if sum(bits(n)) == 2 else 0 for n in range(8)])
al, be = sp.symbols('alpha beta')
gen_int = sum((sp.Symbol(f't{k}') * so3_int[k] for k in range(len(so3_int))), sp.zeros(8, 6))
cond = list((I8 - Nproj1) * gen_int[:, 3:]) + list((I8 - Nproj2) * gen_int[:, :3])
tsyms = [sp.Symbol(f't{k}') for k in range(len(so3_int))]
Acond, _ = sp.linear_eq_to_matrix(cond, tsyms)
fam_dim = len(tsyms) - Acond.rank()
fam = sp.Matrix.hstack(be * gq[1], be * gq[2], be * gq[3], al * f[1], al * f[2], al * f[3])
fam_in = all(fam.subs({al: a_, be: b_}) * LV[i] == Lr[i] * fam.subs({al: a_, be: b_}) for i in (1, 2, 3) for (a_, b_) in ((1, 0), (0, 1)))
check(12, "the so(3)_Delta-intertwiners V -> S form a 4-dimensional space; those with E in N=1 and B in N=2 form the "
          "2-dimensional family Phi_{alpha,beta} = alpha E.f + beta B.g; Phi = Phi_{1,1} is an isometry (Phi^T Phi = 1_6) onto the "
          "six field octants", len(so3_int) == 4 and fam_dim == 2 and fam_in and Phi.T * Phi == sp.eye(6))

# gl(3): spin lift preserves image, induced action; vector action through light-cone components
img_ok = True
for (i, j) in Eij:
    Aij = sp.zeros(3, 3)
    Aij[i - 1, j - 1] = 1
    lift = Eij[i, j]
    newE = (Aij - Aij.trace() / 2 * sp.eye(3))
    newB = (Aij.trace() / 2 * sp.eye(3) - Aij.T)
    M6 = sp.diag(newB, newE)           # induced on (B;E)
    img_ok &= lift * Phi == Phi * M6
    VA = admat(lift)
    u = sp.Matrix(Bs) + sp.Matrix(Es)
    v = sp.Matrix(Bs) - sp.Matrix(Es)
    Xc = sp.Matrix(list(Bs) + list(Es))
    dX = VA * Xc
    dB, dE = dX[:3, :], dX[3:, :]
    img_ok &= sp.expand(dB + dE - Aij * u) == sp.zeros(3, 1) and sp.expand(dB - dE + Aij.T * v) == sp.zeros(3, 1)
# Phi intertwines vector action and spin lift exactly for antisymmetric A
asym = sp.symbols('A0:9')
Ag = sp.Matrix(3, 3, asym)
liftg = sum((Ag[i - 1, j - 1] * Eij[i, j] for (i, j) in Eij), Z8)
Aint, _ = sp.linear_eq_to_matrix(list(liftg * Phi - Phi * admat(liftg)), asym)
ints = [sp.Matrix(3, 3, list(vv)) for vv in Aint.nullspace()]
only_antisym = len(ints) == 3 and all(M == -M.T for M in ints)
# symmetric A acts by B -> A E, E -> A B
Sym = sp.Matrix([[1, 2, 0], [2, -1, 3], [0, 3, 5]])
liftS = sum((Sym[i - 1, j - 1] * Eij[i, j] for (i, j) in Eij), Z8)
sym_act = admat(liftS) == sp.Matrix(sp.BlockMatrix([[sp.zeros(3), Sym], [Sym, sp.zeros(3)]]))
check(13, "the spin lift of gl(3) preserves the image of Phi with E -> (A - trA/2)E, B -> (trA/2 - A^T)B; on V it acts by "
          "B+E -> A(B+E), B-E -> -A^T(B-E), for symmetric A by B -> AE, E -> AB; Phi intertwines vector action and spin lift "
          "exactly for antisymmetric A (a 3-dimensional space)", img_ok and only_antisym and sym_act)

# ---------------------------------------------------------------------------------------------------------------
# s3-14  parity
# ---------------------------------------------------------------------------------------------------------------
AdT = sp.Matrix.hstack(*[coords(T * e[a] * T.inv()) for a in range(1, 7)])
Tsols = implementers({a: eta[a] * e[a] for a in range(1, 7)})
Tuni = len(Tsols) == 1 and sp.Matrix.hstack(Tsols[0].reshape(64, 1), T.reshape(64, 1)).rank() == 1
Tvals = T * vac == full and T * full == -vac and all(T * f[k] == gq[k] and T * gq[k] == -f[k] for k in (1, 2, 3))
TPhi = T * Phi == -Phi * Gm
parPhi = parN * Phi == Phi * etaM and all(parN * e[a] * parN.inv() == -e[a] for a in range(1, 7))
Ade1 = sp.Matrix.hstack(*[coords(e[1] * e[a] * e[1].inv()) for a in range(1, 7)])
e1_ok = Ade1 == sp.diag(1, -1, -1, -1, -1, -1) and e[1] * f[1] == vac
check(14, "Ad_T = eta on V (fixes B, reverses E) and T is the unique implementer up to scale; T|vac> = |full>, T f_k = g_k, "
          "T g_k = -f_k, T|full> = -|vac>; T Phi = -Phi G; (-1)^N Phi = Phi eta while Ad_(-1)^N = -1 on V; Ad_e1 = "
          "diag(1,-1,-1;-1,-1,-1) and e1 f1 = |vac>", AdT == etaM and Tuni and Tvals and TPhi and parPhi and e1_ok)

# ---------------------------------------------------------------------------------------------------------------
# s3-15  the field point on the poles
# ---------------------------------------------------------------------------------------------------------------
Xv = sp.expand(X * vac) == sp.expand(sum(((Bs[i - 1] + Es[i - 1]) * f[i] for i in (1, 2, 3)), sp.zeros(8, 1)))
Xf = sp.expand(X * full) == sp.expand(sum(((Bs[i - 1] - Es[i - 1]) * gq[i] for i in (1, 2, 3)), sp.zeros(8, 1)))
Xf2 = sp.expand(X * full) == sp.expand(sum(((-1) ** (i - 1) * (Bs[i - 1] - Es[i - 1]) * ket(7 ^ (1 << (3 - i))) for i in (1, 2, 3)), sp.zeros(8, 1)))
Xsq = sp.expand(X * X - sum(((Bs[i] + Es[i]) * (Bs[i] - Es[i]) for i in range(3))) * I8) == Z8
zc = [Bs[i] + sp.I * Es[i] for i in range(3)]
lc = all(sp.simplify(sp.sqrt(2) * sp.re(sp.exp(-sp.I * sp.pi / 4) * zc[i]) - (Bs[i] + Es[i])) == 0 and
         sp.simplify(-sp.sqrt(2) * sp.im(sp.exp(-sp.I * sp.pi / 4) * zc[i]) - (Bs[i] - Es[i])) == 0 for i in range(3))
rot8 = (sp.eye(6) + Gm) / sp.sqrt(2)                       # e^{pi G/4}
Xcol = sp.Matrix(list(Bs) + list(Es))
eighth = sp.simplify(X * (vac + full) - sp.sqrt(2) * Phi * rot8 * Xcol) == sp.zeros(8, 1)
chi = sp.Matrix(sp.symbols('h0:8'))
eqs = []
for a in range(1, 7):
    col = Phi[:, a - 1]
    eqs += list(e[a] * chi - col)
nochi = sp.solve(eqs, list(chi)) == []
# rotation-equivariant Clifford maps X -> X chi : chi in common kernel of the L_i
check(15, "X|vac> = sum (B_i+E_i) f_i, X|full> = sum (B_i-E_i) g_i = sum (-1)^{i-1}(B_i-E_i)|111 xor delta_i>, X^2 = "
          "sum (B_i+E_i)(B_i-E_i); B+E = sqrt2 Re(e^{-i pi/4} z), B-E = -sqrt2 Im(e^{-i pi/4} z); X(|vac>+|full>) = "
          "sqrt2 Phi(e^{pi G/4} X); no chi has X chi = Phi(X) for all X", Xv and Xf and Xf2 and Xsq and lc and eighth and nochi)

# ---------------------------------------------------------------------------------------------------------------
# s3-16  even obstruction
# ---------------------------------------------------------------------------------------------------------------
even_blades = [blade(I) for r in (0, 2, 4, 6) for I in itertools.combinations(range(1, 7), r)]
odd_blades = [blade(I) for r in (1, 3, 5) for I in itertools.combinations(range(1, 7), r)]
ev_comm = all(x * parN == parN * x for x in even_blades) and all(x * parN == -parN * x for x in odd_blades)


def solve_space(basis_elems, target_cols):
    """affine dimension of {Q in span(basis_elems): Q Phi = target}, or None if inconsistent."""
    qs = sp.symbols(f'q0:{len(basis_elems)}')
    Q = sum((qs[k] * basis_elems[k] for k in range(len(basis_elems))), Z8)
    A, b = sp.linear_eq_to_matrix(list(Q * Phi - target_cols), qs)
    rA = A.rank()
    if sp.Matrix.hstack(A, b).rank() != rA:
        return None
    return len(qs) - rA


no_even = all(solve_space(even_blades, Phi * KV[k]) is None for k in (1, 2, 3)) and solve_space(even_blades, Phi * Gm) is None
s = sp.symbols('s', real=True)
th = sp.symbols('theta', real=True)
expK3 = (KV[3] * s).exp()
grp = sp.simplify(expK3[:, 0] - sp.Matrix([sp.cosh(s), 0, 0, 0, sp.sinh(s), 0])) == sp.zeros(6, 1)
expG = sp.simplify((Gm * th).exp()[:, 0] - sp.Matrix([sp.cos(th), 0, 0, sp.sin(th), 0, 0])) == sp.zeros(6, 1)
img1 = Phi * expK3[:, 0]
mixed = sp.simplify((Nproj1 * img1).norm()) != 0 and sp.simplify((Nproj2 * img1).norm()) != 0
check(16, "even elements of Cl(3,3) commute and odd ones anticommute with (-1)^N (all 64 blades); no even Q satisfies "
          "Q Phi = Phi K_k (k=1,2,3) or Q Phi = Phi G (inconsistent linear systems); e^{sK_3}e_1 = cosh s e_1 + sinh s e_5 and "
          "e^{theta G}e_1 = cos theta e_1 + sin theta e_4, so Phi(e^{sK_3}e_1) has both N-parities for s != 0",
      ev_comm and no_even and grp and expG and mixed)

# ---------------------------------------------------------------------------------------------------------------
# s3-17, 18  odd intertwiners, rho_oct
# ---------------------------------------------------------------------------------------------------------------
Q = {k: -T * Lr[k] for k in (1, 2, 3)}
qforms = all(Q[k] == -R(1, 2) * e[k] * (I8 - w[j] * w[l]) and Q[k] == -R(1, 2) * (e[k] + e[k + 3] * om) and
             e[k + 3] * om == -e[k] * w[j] * w[l] for (k, j, l) in cyc)
qint = all(Q[k] * Phi == Phi * KV[k] for k in (1, 2, 3)) and -T * Phi == Phi * Gm
grades15 = all(Q[k] + R(1, 2) * e[k] == -R(1, 2) * e[k + 3] * om for k in (1, 2, 3))
full_basis = [blade(I) for r in range(7) for I in itertools.combinations(range(1, 7), r)]
dims = [solve_space(full_basis, Phi * KV[k]) for k in (1, 2, 3)]
dims_odd = [solve_space(odd_blades, Phi * KV[k]) for k in (1, 2, 3)]
vec_basis = [e[a] for a in range(1, 7)]
novec = all(solve_space(vec_basis, Phi * KV[k]) is None for k in (1, 2, 3)) and solve_space(vec_basis, Phi * Gm) is None
# (ii) on Cl(V^+)|vac>
temporal_blades = [blade(I) for r in range(4) for I in itertools.combinations((1, 2, 3), r)]
pauli = all(Q[k] * a * vac == -R(1, 2) * comm(e[k], a) * vac for k in (1, 2, 3) for a in temporal_blades)
central = all(T * e[i] == e[i] * T for i in (1, 2, 3)) and T * T == -I8
check(17, "Q_k = -T L_k = -(1/2) e_k (1 - omega_j omega_l) = -(1/2)(e_k + e_{k+3} omega) (grades 1 and 5) satisfies Q_k Phi = "
          "Phi K_k, and -T Phi = Phi G; the solutions of Q Phi = Phi K_k form affine spaces of dimension 16 (all of Cl(3,3)) and "
          "8 (odd elements); no Clifford vector solves Q Phi = Phi K_k or Q Phi = Phi G; Q_k a|vac> = -(1/2)[e_k, a]|vac> on "
          "Cl(V^+)|vac>, and T is central in Cl(V^+) with T^2 = -1",
      qforms and qint and grades15 and dims == [16, 16, 16] and dims_odd == [8, 8, 8] and novec and pauli and central)

rho_br = all(comm(Lr[i], Q[j]) == sp.LeviCivita(i, j, k) * Q[k] and comm(Q[i], Q[j]) == -sp.LeviCivita(i, j, k) * Lr[k]
             for (i, j, k) in cyc)
cas_oct = sum((Q[k] ** 2 - Lr[k] ** 2 for k in (1, 2, 3)), Z8)
cas_ok = cas_oct == 4 * (Nproj1 + Nproj2)
rho_int = intertwiners([LV[i] for i in (1, 2, 3)] + [KV[i] for i in (1, 2, 3)], [Lr[i] for i in (1, 2, 3)] + [Q[i] for i in (1, 2, 3)])
rho_int_ok = len(rho_int) == 2 and sp.Matrix.hstack(*[m.reshape(48, 1) for m in rho_int], Phi.reshape(48, 1), (Phi * Gm).reshape(48, 1)).rank() == 2
cas_spin = sum((Kb[k] ** 2 - Lr[k] ** 2 for k in (1, 2, 3)), Z8) == 3 * I8
spin_int = intertwiners([LV[i] for i in (1, 2, 3)] + [KV[i] for i in (1, 2, 3)], [Lr[i] for i in (1, 2, 3)] + [Kb[i] for i in (1, 2, 3)])
boosts = all(Kb[k] * f[k] == -full and Kb[k] * vac == -gq[k] for k in (1, 2, 3))
Splus = sp.Matrix.hstack(f[1], f[2], f[3], full)
Sminus = sp.Matrix.hstack(vac, gq[1], gq[2], gq[3])
inv_pm = all(sp.Matrix.hstack(Splus, op * Splus).rank() == 4 and sp.Matrix.hstack(Sminus, op * Sminus).rank() == 4
             for op in list(Lr.values()) + list(Kb.values()))
check(18, "rho_oct(L_k) = L_k, rho_oct(K_k) = Q_k satisfy the brackets of l_G; Casimir sum Q_k^2 - L_k^2 = 4 on the field octants, "
          "0 on the poles; its intertwiners V -> S are span(Phi, Phi G) (dimension 2); the spin lift of l_G has Casimir 3 on all of "
          "S and no nonzero intertwiner V -> S; K_k f_k = -|full>, K_k|vac> = -g_k; S+ = span(f, full) and S- = span(vac, g) are "
          "invariant", rho_br and cas_ok and rho_int_ok and cas_spin and len(spin_int) == 0 and boosts and inv_pm)

# ---------------------------------------------------------------------------------------------------------------
# s3-19  invariance of the octant forms  (CORRECTED: the Euclidean norm is not rho_oct-invariant)
# ---------------------------------------------------------------------------------------------------------------
compact = [e[a] * e[b] for a in range(1, 7) for b in range(a + 1, 7) if (a <= 3) == (b <= 3)]
boosts9 = [e[a] * e[b] for a in (1, 2, 3) for b in (4, 5, 6)]


def inv(M, beta):
    return beta.T * M + M * beta == Z8


parity_inv = all(inv(parN, b) for b in compact) and all(inv(parN, x) for x in list(Lr.values()) + list(Q.values()))
eucl_compact = all(inv(I8, b) for b in compact) and all(inv(I8, Lr[k]) for k in (1, 2, 3))
Qsym = all(Q[k].T == Q[k] and Q[k] != Z8 for k in (1, 2, 3))
eucl_not_Q = all(not inv(I8, Q[k]) for k in (1, 2, 3))
cb = sp.symbols('b0:9')
gb = sum((cb[k] * boosts9[k] for k in range(9)), Z8)
Ab, _ = sp.linear_eq_to_matrix(list(gb.T * parN + parN * gb), cb)
none9 = Ab.rank() == 9
# the energy density changes under a boost: |Phi(e^{sK_3} e_1)|^2 = cosh^2 s + sinh^2 s
en_s = ((Phi * expK3[:, 0]).T * (Phi * expK3[:, 0]))[0]
energy_moves = sp.simplify((en_s - sp.cosh(2 * s)).rewrite(sp.exp)) == 0 and sp.simplify(en_s.diff(s)) != 0
check(19, "CORRECTED: (-1)^N is invariant under the six compact bivectors and under rho_oct, and under none of the nine boosts; "
          "the Euclidean norm is invariant under the compact bivectors and the rho_oct(L_k), but NOT under rho_oct(K_k) = Q_k, "
          "which are symmetric: |psi|^2 = 2 E_6 is the energy density and changes under boosts (|Phi(e^{sK_3}e_1)|^2 = "
          "cosh 2s)", parity_inv and eucl_compact and Qsym and eucl_not_Q and none9 and energy_moves)

# ---------------------------------------------------------------------------------------------------------------
# s3-20, 21  energies and invariant forms
# ---------------------------------------------------------------------------------------------------------------
psiE = sum((Es[k - 1] * f[k] for k in (1, 2, 3)), sp.zeros(8, 1))
psiB = sum((Bs[k - 1] * gq[k] for k in (1, 2, 3)), sp.zeros(8, 1))
psi = psiE + psiB
q = sp.expand(sum(z_ ** 2 for z_ in zc))
E2, B2, EB = sum(x ** 2 for x in Es), sum(x ** 2 for x in Bs), sum(Es[i] * Bs[i] for i in range(3))
en = [sp.expand((psi.T * psi)[0] - (E2 + B2)) == 0,
      sp.expand((psi.T * parN * psi)[0] - sp.re(q)) == 0,
      sp.expand((psiE.T * T * psiB)[0] + EB) == 0,
      sp.expand((psiE.T * T * psiB)[0] + sp.im(q) / 2) == 0,
      sp.expand((psi.T * T * psi)[0]) == 0,
      sp.expand((psi.T * T * Gam * psi)[0] - sp.im(q)) == 0,
      sp.expand(((psi.T * parN * psi)[0]) ** 2 + 4 * ((psiE.T * T * psiB)[0]) ** 2 - sp.expand(q * sp.conjugate(q))) == 0]
c_, eps0 = sp.symbols('c varepsilon_0', positive=True)
psiSI = sum((Es[k - 1] * f[k] + c_ * Bs[k - 1] * gq[k] for k in (1, 2, 3)), sp.zeros(8, 1))
qc = sp.expand(sum((c_ * Bs[i] + sp.I * Es[i]) ** 2 for i in range(3)))
en.append(sp.expand(eps0 / 2 * (psiSI.T * psiSI)[0] - eps0 / 2 * (E2 + c_ ** 2 * B2)) == 0)
en.append(sp.expand((psiSI.T * parN * psiSI)[0] - sp.re(qc)) == 0)
check(20, "psi = Phi(X): |psi|^2 = |E|^2+|B|^2 = 2E_6; psi^T(-1)^N psi = Re q; psi_E^T T psi_B = -E.B = -Im q/2; psi^T T psi = 0; "
          "psi^T T Gamma psi = Im q; (psi^T(-1)^N psi)^2 + 4(psi_E^T T psi_B)^2 = |q|^2; SI: (eps0/2)|Phi(cB;E)|^2 = "
          "(eps0/2)(|E|^2 + c^2|B|^2) and psi^T(-1)^N psi = Re q_c", all(en))

ms = sp.symbols('M0:64')
Mf = sp.Matrix(8, 8, ms)
eqs = []
for b in biv:
    eqs += list(b.T * Mf + Mf * b)
Af, _ = sp.linear_eq_to_matrix(eqs, ms)
invforms = [sp.Matrix(8, 8, list(v)) for v in Af.nullspace()]
TG = T * Gam
forms_ok = len(invforms) == 2 and sp.Matrix.hstack(*[m.reshape(64, 1) for m in invforms], T.reshape(64, 1), TG.reshape(64, 1)).rank() == 2
forms_ok &= T.T == -T and TG.T == TG
# Mukai pairing [tau(phi) ^ chi]_top on Lambda^*(L), basis ell_I <-> a+_I|vac> = |b_I>
subsets = [I for r in range(4) for I in itertools.combinations((1, 2, 3), r)]


mukai = True
for I in subsets:
    for J in subsets:
        val = 0
        if not set(I) & set(J) and len(I) + len(J) == 3:
            val = (-1) ** (len(I) * (len(I) - 1) // 2) * perm_parity(I + J)
        mukai &= val == -(basisL[I].T * T * basisL[J])[0]
transverse = True
for n in range(8):
    for m in range(8):
        agree = sum(1 for x, y in zip(sigma_of(n), sigma_of(m)) if x == y)
        Nn = sp.Matrix.hstack(*[sp.Matrix([1 if a == i else (-sigma_of(n)[i - 1] if a == i + 3 else 0) for a in range(1, 7)]) for i in (1, 2, 3)])
        Nm = sp.Matrix.hstack(*[sp.Matrix([1 if a == i else (-sigma_of(m)[i - 1] if a == i + 3 else 0) for a in range(1, 7)]) for i in (1, 2, 3)])
        inter = 6 - sp.Matrix.hstack(Nn, Nm).rank()
        pair = (ket(n).T * T * ket(m))[0]
        transverse &= inter == agree and ((pair != 0) == (inter == 0) == (m == 7 - n))
check(21, "the Spin(3,3)-invariant bilinear forms on S are span(T, T Gamma) (dimension 2), T antisymmetric, T Gamma symmetric; "
          "the Chevalley-Mukai pairing [tau(phi) ^ chi]_top on Lambda^*(L) equals -phi^T T chi on all 64 basis pairs; dim(N_sigma "
          "cap N_tau) = number of agreeing signs, and two octants pair nonzero iff transverse iff antipodal",
      forms_ok and mukai and transverse)

# ---------------------------------------------------------------------------------------------------------------
# s3-22 .. 26  Maxwell on the octants (general symbolic fields; residual identities)
# ---------------------------------------------------------------------------------------------------------------
t, x1, x2, x3 = sp.symbols('t x1 x2 x3', real=True)
xx = (x1, x2, x3)
Ef = [sp.Function(f'E{k}', real=True)(t, *xx) for k in (1, 2, 3)]
Bf = [sp.Function(f'B{k}', real=True)(t, *xx) for k in (1, 2, 3)]
rho = sp.Function('rho', real=True)(t, *xx)
jf = [sp.Function(f'j{k}', real=True)(t, *xx) for k in (1, 2, 3)]
ph0 = sp.Function('phi0')(t, *xx)
ph3 = sp.Function('phi3')(t, *xx)


def D(psi_, mats=None):
    mats = mats or [e[1], e[2], e[3]]
    return sum((mats[i] * psi_.diff(xx[i]) for i in range(3)), sp.zeros(8, 1))


def dext(psi_):
    return sum((ap[i] * psi_.diff(xx[i - 1]) for i in (1, 2, 3)), sp.zeros(8, 1))


def codiff(psi_):
    return -sum((am[i] * psi_.diff(xx[i - 1]) for i in (1, 2, 3)), sp.zeros(8, 1))


def curl(F):
    return [F[2].diff(x2) - F[1].diff(x3), F[0].diff(x3) - F[2].diff(x1), F[1].diff(x1) - F[0].diff(x2)]


def div(F):
    return sum(F[i].diff(xx[i]) for i in range(3))


def onF(F):
    return sum((F[k] * f[k + 1] for k in range(3)), sp.zeros(8, 1))


def onG(F):
    return sum((F[k] * gq[k + 1] for k in range(3)), sp.zeros(8, 1))


gen8 = sp.Matrix([sp.Function(f'u{n}')(t, *xx) for n in range(8)])
dd = sp.simplify(dext(dext(gen8))) == sp.zeros(8, 1) and sp.simplify(codiff(codiff(gen8))) == sp.zeros(8, 1)
dm = sp.expand(dext(gen8) - codiff(gen8) - D(gen8)) == sp.zeros(8, 1)
dp = sp.expand(dext(gen8) + codiff(gen8) - D(gen8, [e[4], e[5], e[6]])) == sp.zeros(8, 1)
lap = sum((gen8.diff(xi, 2) for xi in xx), sp.zeros(8, 1))
DD = sp.expand(D(D(gen8)) - lap) == sp.zeros(8, 1)
DpDp = sp.expand(D(D(gen8, [e[4], e[5], e[6]]), [e[4], e[5], e[6]]) + lap) == sp.zeros(8, 1)
four = [sp.expand(dext(onF(Ef)) - onG(curl(Ef))) == sp.zeros(8, 1),
        sp.expand(dext(onG(Bf)) - div(Bf) * full) == sp.zeros(8, 1),
        sp.expand(codiff(onF(Ef)) + div(Ef) * vac) == sp.zeros(8, 1),
        sp.expand(codiff(onG(Bf)) - onF(curl(Bf))) == sp.zeros(8, 1)]
check(22, "d = sum a+_i d_i and delta = -sum a-_i d_i satisfy d^2 = delta^2 = 0, d - delta = sum e_i d_i = D, d + delta = "
          "sum e_{i+3} d_i = D', D^2 = Laplacian, D'^2 = -Laplacian, and d(E.f) = (curl E).g, d(B.g) = (div B)|full>, "
          "delta(E.f) = -(div E)|vac>, delta(B.g) = (curl B).f (general fields)", dd and dm and dp and DD and DpDp and all(four))

PhiX = onF(Ef) + onG(Bf)
lhs = PhiX.diff(t) - D(PhiX)
rhs = -div(Ef) * vac + onF([Ef[k].diff(t) + curl(Bf)[k] for k in range(3)]) + onG([Bf[k].diff(t) - curl(Ef)[k] for k in range(3)]) - div(Bf) * full
ident = sp.expand(lhs - rhs) == sp.zeros(8, 1)
# residuals of ES17 eq. (16): d_t z = -i curl z - i j, div z = i rho, z = B + iE
z = [Bf[k] + sp.I * Ef[k] for k in range(3)]
curlz = curl(z)
res_t = [sp.expand(z[k].diff(t) + sp.I * curlz[k] + sp.I * jf[k]) for k in range(3)]
res_d = sp.expand(div(z) - sp.I * rho)
full_eq = lhs + rho * vac + onF(jf)


def re_(x):     # real and imaginary parts for expressions linear over C in real fields and their derivatives
    return sp.expand((x + x.subs(sp.I, -sp.I)) / 2)


def im_(x):
    return sp.expand((x - x.subs(sp.I, -sp.I)) / (2 * sp.I))


srcs = sp.expand(full_eq - (-im_(res_d) * vac + onF([im_(r_) for r_ in res_t]) + onG([re_(r_) for r_ in res_t]) - re_(res_d) * full))
check(23, "(d_t - D)Phi(X) = -(div E)|vac> + (d_t E + curl B).f + (d_t B - curl E).g - (div B)|full> on general fields, and "
          "(d_t - D)Phi(X) + rho|vac> + j.f has as its components exactly the real and imaginary parts of the residuals of ES17 "
          "eq. (16): Gauss on the vacuum, div B = 0 on the filled pole, d_t E = -curl B - j on the electric and d_t B = curl E "
          "on the magnetic octants", ident and srcs == sp.zeros(8, 1))

# SI, textbook/Hestenes, wave operator, extended system
PhiSI = onF(Ef) + onG([c_ * Bf[k] for k in range(3)])
lhsSI = PhiSI.diff(t) / c_ - D(PhiSI)
rG = div(Ef) - rho / eps0
rA = [Ef[k].diff(t) + c_ ** 2 * curl(Bf)[k] + jf[k] / eps0 for k in range(3)]
rF = [Bf[k].diff(t) - curl(Ef)[k] for k in range(3)]
rD = div(Bf)
si_ok = sp.simplify(lhsSI + (c_ * rho * vac + onF(jf)) / (eps0 * c_) - (-rG * vac + onF([r_ / c_ for r_ in rA]) + onG(rF) - c_ * rD * full)) == sp.zeros(8, 1)
Bt = [-b for b in Bf]
Phit = onF(Ef) + onG(Bt)
tb = sp.expand(Phit + parN * PhiX) == sp.zeros(8, 1)
hes = sp.expand(Phit.diff(t) + D(Phit) - (rho * vac - onF(jf)) + parN * full_eq) == sp.zeros(8, 1)
hes_form = sp.expand(Phit - sum((Ef[k] * e[k + 1] + Bt[k] * T * e[k + 1] for k in range(3)), Z8) * vac) == sp.zeros(8, 1)
Te = all(T * e[k] == e[j] * e[l] for (k, j, l) in cyc)
TPhit = T * Phi * sp.diag(-1, -1, -1, 1, 1, 1) == Phi * sp.diag(-1, -1, -1, 1, 1, 1) * Gm
check(24, "SI: (d_t/c - D)Phi(cB;E) + (c rho|vac> + j.f)/(eps0 c) = -(div E - rho/eps0)|vac> + (1/c)(Ampere residual).f + "
          "(Faraday residual).g - c(div B)|full>; textbook: Phi_t = E.f + B_t.g = -(-1)^N Phi, T Phi_t = Phi_t G, "
          "(d_t + D)Phi_t - (rho|vac> - j.f) = -(-1)^N[(d_t - D)Phi + rho|vac> + j.f], Phi_t = (sum E_k e_k + B_tk T e_k)|vac>, "
          "T e_k = e_j e_l: Hestenes' (d_t + grad)(E + I B_t) = rho - j with I = T",
      si_ok and tb and hes and hes_form and Te and TPhit)

waveop = sp.expand(gen8.diff(t, 2) - D(D(gen8)) - (gen8.diff(t, 2) - lap)) == sp.zeros(8, 1)
waveop2 = sp.expand(gen8.diff(t, 2) - D(D(gen8, [e[4], e[5], e[6]]), [e[4], e[5], e[6]]) - (gen8.diff(t, 2) + lap)) == sp.zeros(8, 1)
ext = PhiX + ph0 * vac + ph3 * full
ext_res = sp.expand(ext.diff(t) - D(ext))
gradf = [ph0.diff(xi) for xi in xx]
grad3 = [ph3.diff(xi) for xi in xx]
ext_ok = sp.expand(ext_res - ((ph0.diff(t) - div(Ef)) * vac + onF([Ef[k].diff(t) + curl(Bf)[k] - gradf[k] for k in range(3)]) +
                              onG([Bf[k].diff(t) - curl(Ef)[k] - grad3[k] for k in range(3)]) + (ph3.diff(t) - div(Bf)) * full)) == sp.zeros(8, 1)
check(25, "(d_t + D)(d_t - D) = d_t^2 - Laplacian on all eight components, (d_t + D')(d_t - D') = d_t^2 + Laplacian; with pole "
          "components phi0, phi3, (d_t - D)psi = 0 is d_t phi0 = div E, d_t phi3 = div B, d_t E = -curl B + grad phi0, "
          "d_t B = curl E + grad phi3", waveop and waveop2 and ext_ok)

# uniqueness: first-order jets of source-free Maxwell fields
a_, b_, c2, d_ = sp.symbols('a b cc d')
J = sp.symbols('J0:24')       # J[4k+mu] = d_mu E_k (mu = t,1,2,3), J[12+4k+mu] = d_mu B_k
dE = [[J[4 * k + mu] for mu in range(4)] for k in range(3)]
dB = [[J[12 + 4 * k + mu] for mu in range(4)] for k in range(3)]


def jcurl(dF):
    return [dF[2][2] - dF[1][3], dF[0][3] - dF[2][1], dF[1][1] - dF[0][2]]


cons = [dE[k][0] + jcurl(dB)[k] for k in range(3)] + [dB[k][0] - jcurl(dE)[k] for k in range(3)] + \
    [sum(dE[k][k + 1] for k in range(3)), sum(dB[k][k + 1] for k in range(3))]
Acons, _ = sp.linear_eq_to_matrix(cons, J)
nsj = Acons.nullspace()
params = sp.symbols(f'p0:{len(nsj)}')
jet = sum((params[i] * nsj[i] for i in range(len(nsj))), sp.zeros(24, 1))
sub = {J[i]: jet[i] for i in range(24)}
# Phi_M = (aE + bB).f + (cE + dB).g; (d_t - D) on jets
Fcomp = [[a_ * dE[k][mu] + b_ * dB[k][mu] for mu in range(4)] for k in range(3)]
Gcomp = [[c2 * dE[k][mu] + d_ * dB[k][mu] for mu in range(4)] for k in range(3)]
resM = -sum(Fcomp[k][k + 1] for k in range(3)) * vac + onF([Fcomp[k][0] + jcurl(Gcomp)[k] for k in range(3)]) + \
    onG([Gcomp[k][0] - jcurl(Fcomp)[k] for k in range(3)]) - sum(Gcomp[k][k + 1] for k in range(3)) * full
resM = sp.expand(resM.subs(sub))
coef = [cf for comp in resM for cf in sp.Poly(comp, *params).coeffs()]
solM = sp.solve(coef, [c2, d_], dict=True)
uniq = solM == [{c2: -b_, d_: a_}]
r_ = sp.symbols('r', nonnegative=True)
rotG = sp.cos(th) * sp.eye(6) + sp.sin(th) * Gm
PhiTheta = sp.simplify(r_ * Phi * rotG)
formM = sp.simplify(PhiTheta - (sp.Matrix.hstack(*[r_ * sp.sin(th) * f[k] + r_ * sp.cos(th) * gq[k] for k in (1, 2, 3)],
                                                *[r_ * sp.cos(th) * f[k] - r_ * sp.sin(th) * gq[k] for k in (1, 2, 3)]))) == sp.zeros(8, 6)
def single_weight(cols):
    return (Nproj2 * cols == sp.zeros(8, 3)) or (Nproj1 * cols == sp.zeros(8, 3))


sep = [tt for tt in (0, sp.pi / 6, sp.pi / 4, sp.pi / 2, 2 * sp.pi / 3, sp.pi, 3 * sp.pi / 2) if
       single_weight((Phi * rotG.subs(th, tt))[:, 3:]) and single_weight((Phi * rotG.subs(th, tt))[:, :3])]
sep_ok = sep == [0, sp.pi / 2, sp.pi, 3 * sp.pi / 2]
# Phi o G places electric sources on the filled pole and the N=2 octants
PhiGX = onF(Bf) + onG([-x for x in Ef])
srcG = sp.expand((PhiGX.diff(t) - D(PhiGX)) - (-div(Bf) * vac + onF([Bf[k].diff(t) - curl(Ef)[k] for k in range(3)]) +
                                               onG([-(Ef[k].diff(t) + curl(Bf)[k]) for k in range(3)]) + div(Ef) * full)) == sp.zeros(8, 1)
check(26, "on first-order jets of source-free Maxwell fields (d_t - D)Phi_M = 0 for Phi_M = (aE+bB).f + (cE+dB).g iff c = -b, d = a, "
          "which is r Phi o e^{theta G}; E and B lie on single weights iff theta in (pi/2)Z; Phi o G puts the charge density on "
          "the filled pole and the current on the N=2 octants", uniq and formM and sep_ok and srcG)

# circular wave of ES17 Ex. 12.11
k_ = sp.symbols('k', positive=True)
phase = k_ * (x3 - t)
zw = [sp.exp(sp.I * phase), sp.I * sp.exp(sp.I * phase), 0]
Bw = [sp.re(sp.expand_complex(v)) for v in zw]
Ew = [sp.im(sp.expand_complex(v)) for v in zw]
psiw = onF(Ew) + onG(Bw)
psir = onF(Ew) + onG([-v for v in Bw])
circ = sp.simplify(psiw.diff(t) - D(psiw)) == sp.zeros(8, 1) and sp.simplify(psir.diff(t) - D(psir)) != sp.zeros(8, 1)
check(27, "the circularly polarised wave z = (1,i,0)e^{ik(x3-t)} of ES17 Ex. 12.11 solves (d_t - D)Phi(X) = 0; with B reversed "
          "it does not", circ)

# pooling and the monograph labels
pool = all(all(e[i] * ket(n) in (ket(n ^ (1 << (3 - i))), -ket(n ^ (1 << (3 - i)))) for n in range(8)) for i in (1, 2, 3)) and \
    all(e[i + 3] == e[i] * w[i] for i in (1, 2, 3))
mono = True
for n in range(8):
    sg = sigma_of(n)
    wt = sum(1 for x in sg if x == 1)              # earlier 'temporal weight' = number of plus signs
    mono &= wt == 3 - sum(bits(n))
for k in (1, 2, 3):
    # octant with a single plus in pair k is the line of g_k; single minus in pair k is the line of f_k
    nk_plus = num([0 if i == k else 1 for i in (1, 2, 3)])
    nk_minus = num([1 if i == k else 0 for i in (1, 2, 3)])
    mono &= gq[k] in (ket(nk_plus), -ket(nk_plus)) and f[k] == ket(nk_minus)
PhiG = Phi * Gm
mono &= all(PhiG[:, k - 1] == f[k] and PhiG[:, k + 2] == -gq[k] for k in (1, 2, 3))
check(28, "each e_i and e_{i+3} = e_i omega_i flip exactly sigma_i (nearest-neighbour structure of D); the earlier 'temporal "
          "weight' (number of plus signs) is 3 - N, so its electric octants (one plus, in pair k) are the lines of g_k (N = 2) and "
          "its magnetic ones (one minus, in pair m) the lines of f_m (N = 1): Phi o G = B.f - E.g, not Phi", pool and mono)

# l_int stabilisers, octant-group lifts versus l_int
lint = [e[a] * e[b] for (a, b) in itertools.combinations((1, 4, 5, 6), 2)]
stab_ok = True
for n in range(8):
    s1 = sigma_of(n)[0]
    ls = sp.symbols('l0:6')
    beta = sum((ls[i] * lint[i] for i in range(6)), Z8)
    v = beta * ket(n)
    eqs = [v[r] for r in range(8) if r != n]
    Al, _ = sp.linear_eq_to_matrix(eqs, ls)
    stab = Al.nullspace()
    expect = [w[1], (e[1] - s1 * e[4]) * e[5], (e[1] - s1 * e[4]) * e[6]]
    stab_ok &= len(stab) == 3
    stab_mats = [sum((vv[i] * lint[i] for i in range(6)), Z8) for vv in stab]
    stab_ok &= sp.Matrix.hstack(*[m.reshape(64, 1) for m in stab_mats + expect]).rank() == 3
    stab_ok &= all(m * ket(n) == sp.zeros(8, 1) for m in expect[1:])
lifts_ok = all(g[i] * b == b * g[i] for i in (2, 3) for b in lint)
gl1 = [g[1] * b * g[1].inv() for b in lint]
lifts_ok &= all(sp.Matrix.hstack(*[m.reshape(64, 1) for m in lint], x.reshape(64, 1)).rank() == 6 for x in gl1)
check(29, "the stabiliser in l_int of every octant line is 3-dimensional, spanned by e1e4 = omega_1 and the null rotations "
          "(e1 - sigma_1 e4)e5, (e1 - sigma_1 e4)e6 (which annihilate it), depending on sigma_1 only; g_2, g_3 commute with l_int "
          "and g_1 normalises it", stab_ok and lifts_ok)

# FMB Sec. 5: I_T = e1e2e3
check(30, "FMB Sec. 5: I_T = e1e2e3 = T has I_T^2 = -1 and commutes with e1, e2, e3; Q_k and -T are odd (anticommute with Gamma)",
      T * T == -I8 and all(comm(T, e[i]) == Z8 for i in (1, 2, 3)) and all(Q[k] * Gam == -Gam * Q[k] for k in (1, 2, 3))
      and T * Gam == -Gam * T)

if FAILS:
    print(f"s3_check.py: {len(FAILS)} check(s) failed: {FAILS}")
    sys.exit(1)
print("s3_check.py: all checks pass")
sys.exit(0)
