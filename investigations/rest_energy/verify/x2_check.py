#!/usr/bin/env python3
"""Adversarial check x2 of track d2 ("What mass breaks: duality, chirality, particle/antiparticle, vacuum").

Independent of investigations/rest_energy/verify/d2_symmetry_breaking.py (nothing is imported from it).
Conventions are rebuilt from the paper sources:
  tex/s02 eq. (realisation): e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1, e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep;
  tex/s03 Prop. 3.3 / Lemma 3.4: G e_i = e_{i+3}, G e_{i+3} = -e_i, and G = -K o Ad_T with K = (1/2) ad_P;
  tex/s12 Thm 12.3: X = sum_i (B_i e_i + E_i e_{i+3}), z = B + iE, K_i = (1/2)(e_k e_{j+3} - e_j e_{k+3});
  tex/s04 Prop 4.5: K^mu = -gamma^0 gamma^mu, T = gamma^0 J;  tex/s05 Thms 5.1, 5.3, 5.6, 5.7, Prop 5.5, 5.8;
  tex/s16 eq. (splitL), Thm 16.2, Prop 16.5, Thm 16.6;  tex/s18 Def 18.2, Thms 18.3, 18.5, 18.6;
  tex/s08 (tau_i = -(1/2) w_j w_k cyclic), tex/s19 Thm 19.2, tex/s22 Thm 22.2, 22.3, tex/sA (three-constituent data).
Methods differ from the derive script where possible: G rebuilt from the Clifford adjoint action (Lemma 3.4);
blade-combinatorial solutions cross-checked with sympy nullspaces; rational points on the circle and hyperbola in
place of symbolic trig; Euler-Lagrange equations derived from the Lagrangian; random rational null vectors via the
Cayley transform; joint Fock spectrum by numerical simultaneous diagonalisation (tolerance 1e-8); the lift of C by
the exponential map instead of reflections; the three-constituent ranks on integer Gram matrices with sympy; the
electroweak orbit count of Theorem 22.3 (cited, not checked, by d2) re-derived from blade-level kernel enumeration with
numerical SVD ranks (relative tolerance 1e-8; the observed gap is 16 versus 4e-13).
Extra checks x2-N1..N3 support corrections: N1 (Phi0 masses only nu_R), N2 (d2-2 is the textbook F.F -> F.*F rotation),
N3 (both pure senses of the binding rotation are duality rotations).
Prints one PASS/FAIL line per checked item and exits 0 only if all pass.  Seeded: default_rng(2026).
"""
import sys
import itertools
from fractions import Fraction
import numpy as np
import sympy as sp

RNG = np.random.default_rng(2026)
FAILS = []


def check(cid, ok, text):
    if ok:
        print(f"PASS {cid}: {text}")
    else:
        print(f"FAIL {cid}: {text}")
        FAILS.append(cid)


def note(text):
    print(f"    . {text}")


def kron(*ms):
    out = np.array([[1]], dtype=np.int64)
    for m in ms:
        out = np.kron(out, np.asarray(m, dtype=np.int64))
    return out


def symrank(rows):
    return sp.Matrix(rows).rank()


def is_zero(M):
    return not np.any(M)


# ====================================================================== one constituent
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
ep = np.array([[0, -1], [1, 0]])
I2 = np.eye(2, dtype=np.int64)
I8 = np.eye(8, dtype=np.int64)
e = [None,
     kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx),
     kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
etad = [None, 1, 1, 1, -1, -1, -1]


def blade(I):
    M = I8.copy()
    for a in I:
        M = M @ e[a]
    return M


def bl_sign_comm(I, Jb):
    """blade rule (Lemma 2.1): e_I e_J = (-1)^{|I||J|-|I cap J|} e_J e_I"""
    return (-1) ** (len(I) * len(Jb) - len(set(I) & set(Jb)))


def bl_transpose_sign(I):
    """Lemma 2.1: B^T = (-1)^{r(r-1)/2} prod eta_aa B"""
    r = len(I)
    s = (-1) ** (r * (r - 1) // 2)
    for a in I:
        s *= etad[a]
    return s


J = blade([2, 3])
Om = blade([1, 4, 5, 6])
T = blade([1, 2, 3])
Gam = J @ Om
g0 = e[1]
gam = [e[1], e[4], e[5], e[6]]

ok = all(np.array_equal(e[a] @ e[b] + e[b] @ e[a], 2 * (etad[a] if a == b else 0) * I8)
         for a in range(1, 7) for b in range(1, 7))
ok &= np.array_equal(J @ J, -I8) and np.array_equal(Om @ Om, -I8) and np.array_equal(T @ T, -I8)
ok &= np.array_equal(Gam @ Gam, I8) and np.array_equal(Gam, blade([1, 2, 3, 4, 5, 6]))
ok &= np.array_equal(T.T, -T) and np.array_equal(T, g0 @ J)
ok &= all(np.array_equal(J @ g, g @ J) for g in gam)
# the blade rule itself, on all pairs of blades of grade <= 3 (64 x 64 would be slow-ish; use all)
allblades = [tuple(I) for r in range(7) for I in itertools.combinations(range(1, 7), r)]
BL = {I: blade(I) for I in allblades}
ok_bl = all(np.array_equal(BL[I] @ BL[K], bl_sign_comm(I, K) * BL[K] @ BL[I]) for I in allblades for K in allblades)
ok_tr = all(np.array_equal(BL[I].T, bl_transpose_sign(I) * BL[I]) for I in allblades)
check("x2-0", ok and ok_bl and ok_tr,
      "realisation of tex/s02: Clifford relations; J^2=Om^2=T^2=-1, Gamma^2=1, Gamma=JOm=e1...e6, T=gamma^0 J antisymmetric, "
      "[J,gamma^mu]=0; blade commutation and transpose rules (Lemma 2.1) hold on all 64x64 blade pairs")

# ---------------------------------------------------------------------- G rebuilt from Clifford data (Lemma 3.4)
def vec_coords(W):
    """coordinates of a grade-one element W = sum c_b e_b, via Tr(e_b W) = 8 eta_bb c_b"""
    c = [Fraction(int(np.trace(e[b] @ W)), 8 * etad[b]) for b in range(1, 7)]
    back = sum(int(c[b - 1]) * e[b] for b in range(1, 7))
    assert np.array_equal(back, W)
    return [int(x) for x in c]


Psum = sum(e[i] @ e[i + 3] for i in range(1, 4))
Tinv = -T
Kmat = np.zeros((6, 6), dtype=np.int64)
Smat = np.zeros((6, 6), dtype=np.int64)
for a in range(1, 7):
    Kv = (Psum @ e[a] - e[a] @ Psum)
    assert np.all(Kv % 2 == 0)
    Kmat[:, a - 1] = vec_coords(Kv // 2)
    Smat[:, a - 1] = vec_coords(T @ e[a] @ Tinv)
Gcl = -Kmat @ Smat
G = np.zeros((6, 6), dtype=np.int64)
for i in range(3):
    G[i + 3, i] = 1
    G[i, i + 3] = -1
ETA = np.diag([1, 1, 1, -1, -1, -1]).astype(np.int64)
ok = np.array_equal(Gcl, G) and np.array_equal(G @ G, -np.eye(6, dtype=np.int64))
ok &= np.array_equal(G.T, -G) and np.array_equal(ETA @ G.T @ ETA, G) and np.array_equal(G.T @ ETA @ G, -ETA)
check("x2-G", ok, "binding root rebuilt as G=-K o Ad_T with K=(1/2)ad_P (Lemma 3.4) equals the block matrix of Prop 3.3: "
      "G^2=-1, G antisymmetric (orthogonal), eta-self-adjoint, anti-isometry")

# ---------------------------------------------------------------------- field-strength dictionary
Gs = sp.Matrix(G)
ETAs = sp.Matrix(ETA)
xs = sp.symbols('x1:7', real=True)
Xs = sp.Matrix(xs)


def zvec(X):
    return [X[i] + sp.I * X[i + 3] for i in range(3)]


def qf(z):
    return sp.expand(sum(zi * zi for zi in z))


zX = zvec(Xs)
zGX = zvec(Gs * Xs)
ok = all(sp.expand(zGX[i] - sp.I * zX[i]) == 0 for i in range(3))
ok &= sp.expand((Xs.T * ETAs * Xs)[0] - sp.re(qf(zX))) == 0
B = Xs[0:3, 0]
E = Xs[3:6, 0]
ok &= sp.expand((Xs.T * ETAs * Gs * Xs)[0] + 2 * B.dot(E)) == 0
ok &= sp.expand(sp.im(qf(zX)) - 2 * B.dot(E)) == 0
check("x2-z", ok, "z(GX)=i z(X); eta(X,X)=Re q=|B|^2-|E|^2; eta(X,GX)=-2E.B=-Im q (Thm 12.3 conventions)")

# ---------------------------------------------------------------------- d2-1 / d2-2 duality
t = sp.symbols('t', real=True)
D = sp.cos(t) * sp.eye(6) + sp.sin(t) * Gs
ok1 = sp.simplify(D.T * D - sp.eye(6)) == sp.zeros(6)
ok2 = sp.simplify(D.T * ETAs * D - (sp.cos(2 * t) * ETAs + sp.sin(2 * t) * ETAs * Gs)) == sp.zeros(6)
zD = zvec(D * Xs)
cis = sp.cos(t) + sp.I * sp.sin(t)
ok3 = all(sp.expand(zD[i] - cis * zX[i]) == 0 for i in range(3))
ok4 = sp.expand(qf(zD) - cis ** 2 * qf(zX)) == 0
ok5 = sp.simplify((D * Xs).dot(D * Xs) - Xs.dot(Xs)) == 0
# rational points: (c,s) = ((1-r^2)/(1+r^2), 2r/(1+r^2))
ok6 = True
for r in [Fraction(1, 2), Fraction(2, 3), Fraction(-3, 7), Fraction(5, 1)]:
    c = sp.Rational((1 - r * r) / (1 + r * r))
    s = sp.Rational(2 * r / (1 + r * r))
    Dr = c * sp.eye(6) + s * Gs
    ok6 &= (Dr.T * Dr == sp.eye(6))
    ok6 &= (Dr.T * ETAs * Dr == (c * c - s * s) * ETAs + 2 * c * s * ETAs * Gs)
    ok6 &= (Dr.T * ETAs * Dr != ETAs)
indep = sp.Matrix([list(ETAs), list(ETAs * Gs)]).rank() == 2
check("x2-1", ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and indep,
      "duality D_t=cos t+sin t G: z->e^{it}z, |z|^2 kept, q->e^{2it}q; D^T D=1 and D^T eta D=cos2t eta+sin2t eta G "
      "(symbolic and at rational points of the circle); eta, eta G independent, so eta-isometry iff t in pi Z")

# complexion (Prop 12.12 frame): z = (b+ia) e_1; D_{-phi} carries it to a pure magnetic plane
ok = True
for (a_, b_) in [(3, 4), (5, 12), (-8, 15), (7, -24)]:
    rr = sp.sqrt(a_ * a_ + b_ * b_)
    cphi, sphi = sp.Rational(b_) / rr, sp.Rational(a_) / rr     # phi = arg(b + i a)
    X0 = sp.Matrix([b_, 0, 0, a_, 0, 0])
    Xr = (cphi * sp.eye(6) - sphi * Gs) * X0
    ok &= sp.simplify(Xr - sp.Matrix([rr, 0, 0, 0, 0, 0])) == sp.zeros(6, 1)
    qv = sp.expand((b_ + sp.I * a_) ** 2)
    ok &= sp.simplify(qv - (a_ ** 2 + b_ ** 2) * (cphi + sp.I * sphi) ** 2) == 0
check("x2-1b", ok, "Prop 12.12 frame z=(b+ia)e1: q=(a^2+b^2)e^{2i phi}, phi=arg(b+ia) (tan phi=a/b), and D_{-phi}X = sqrt(a^2+b^2) e1, "
      "a single magnetic plane (F_23=B_1)")

# ---------------------------------------------------------------------- l_G from the Clifford adjoint action
def adj6(Bv):
    M = np.zeros((6, 6), dtype=object)
    for a in range(1, 7):
        W = Bv @ e[a] - e[a] @ Bv
        c = [sp.Rational(int(np.trace(e[b] @ W)), 8 * etad[b]) for b in range(1, 7)]
        for b in range(6):
            M[b, a - 1] = c[b]
    return sp.Matrix(M.tolist())


cyc = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
Lm, Km = [], []
for (i, j, k) in cyc:
    Lm.append(adj6(e[k] @ e[j] + e[j + 3] @ e[k + 3]) / 2)
    Km.append(adj6(e[k] @ e[j + 3] - e[j] @ e[k + 3]) / 2)
Lm = [sp.Matrix(L) for L in Lm]
Km = [sp.Matrix(K) for K in Km]


def crossmat(v):
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


ok = True
for idx in range(3):
    ei = [1 if r_ == idx else 0 for r_ in range(3)]
    Cx = crossmat(ei)
    ok &= Lm[idx] == sp.diag(Cx, Cx)
    # K_i: delta E = e_i x B, delta B = -e_i x E
    Kexp = sp.zeros(6)
    Kexp[0:3, 3:6] = -Cx
    Kexp[3:6, 0:3] = Cx
    ok &= Km[idx] == Kexp


def br(A, B_):
    return A * B_ - B_ * A


eps = sp.LeviCivita
for a_ in range(3):
    for b_ in range(3):
        ok &= br(Lm[a_], Lm[b_]) == sum((eps(a_, b_, c_) * Lm[c_] for c_ in range(3)), sp.zeros(6))
        ok &= br(Lm[a_], Km[b_]) == sum((eps(a_, b_, c_) * Km[c_] for c_ in range(3)), sp.zeros(6))
        ok &= br(Km[a_], Km[b_]) == sum((-eps(a_, b_, c_) * Lm[c_] for c_ in range(3)), sp.zeros(6))
lGb = Lm + Km
ok &= all(A_ * Gs == Gs * A_ and A_.T * ETAs + ETAs * A_ == sp.zeros(6) for A_ in lGb)
# centraliser of G in so(3,3): solve linear system over all 6x6 matrices
avars = sp.symbols('a0:36')
Agen = sp.Matrix(6, 6, avars)
eqs = list(Agen.T * ETAs + ETAs * Agen) + list(Agen * Gs - Gs * Agen)
sol = sp.linsolve(eqs, avars)
free = set().union(*[s_.free_symbols for s_ in list(sol)[0]])
ok &= len(free) == 6
ok &= sp.Matrix([list(A_) for A_ in lGb]).rank() == 6
check("x2-5a", ok, "l_G rebuilt from the Clifford adjoint: L_i=(1/2)ad(e_k e_j+e_{j+3}e_{k+3}) acts as z->e_i x z, "
      "K_i=(1/2)ad(e_k e_{j+3}-e_j e_{k+3}) as delta E=e_i x B, delta B=-e_i x E (z->i e_i x z); brackets of Thm 12.3; "
      "centraliser of G in so(3,3) has dim 6 = span{L_i,K_i}")

# ---------------------------------------------------------------------- d2-3 rest energy
Bx, By, Bz, Ex, Ey, Ez = xs
Bv = sp.Matrix([Bx, By, Bz])
Ev = sp.Matrix([Ex, Ey, Ez])
u = (Bv.dot(Bv) + Ev.dot(Ev)) / 2
S = Ev.cross(Bv)
qre = Bv.dot(Bv) - Ev.dot(Ev)
qim = 2 * Ev.dot(Bv)
ok = sp.expand(u ** 2 - S.dot(S) - (qre ** 2 + qim ** 2) / 4) == 0
# invariance: d/ds f(X + s A X) at s=0
s_ = sp.symbols('s')


def deriv_inv(f, A_):
    Xn = Xs + s_ * A_ * Xs
    sub = dict(zip(xs, list(Xn)))
    return sp.expand(sp.diff(f.subs(sub, simultaneous=True), s_).subs(s_, 0))


ok &= all(deriv_inv(qre, A_) == 0 and deriv_inv(qim, A_) == 0 for A_ in lGb)
ok &= all(deriv_inv(u, L_) == 0 for L_ in Lm) and all(deriv_inv(u, K_) != 0 for K_ in Km)
ok &= deriv_inv(u, Gs) == 0 and deriv_inv(qre, Gs) != 0 and deriv_inv(qim, Gs) != 0
check("x2-3", ok, "u^2-|E x B|^2=|q|^2/4; Re q, Im q killed by all six l_G generators, u by L_i but not K_i; "
      "u killed by G, Re q and Im q not: |q|/2 is the invariant of both")

# ---------------------------------------------------------------------- class check for d2-2: the textbook duality identity
etaM = sp.diag(1, -1, -1, -1)                  # spacetime signature (+,-,-,-), tex/s01 conventions


def F_of(X):
    Fm = sp.zeros(4)
    for i in range(3):
        Fm[0, i + 1] = X[i + 3]
        Fm[i + 1, 0] = -X[i + 3]
    for i in range(3):
        for j in range(3):
            for k_ in range(3):
                if sp.LeviCivita(i, j, k_) != 0:
                    Fm[j + 1, k_ + 1] += sp.LeviCivita(i, j, k_) * X[i]
    return Fm


def up2(Fm):
    return etaM * Fm * etaM


def hodge(Fm):
    Fu = up2(Fm)
    Hm = sp.zeros(4)
    for a_ in range(4):
        for b_ in range(4):
            Hm[a_, b_] = sum(sp.LeviCivita(a_, b_, c_, d_) * Fu[c_, d_] for c_ in range(4) for d_ in range(4)) / 2
    return Hm


def half_contr(F1, F2):
    return sp.expand(sum(F1[a_, b_] * up2(F2)[a_, b_] for a_ in range(4) for b_ in range(4)) / 2)


FX = F_of(Xs)
ok = sp.simplify(F_of(Gs * Xs) - hodge(FX)) == sp.zeros(4)
ok &= sp.expand(half_contr(FX, FX) - (Xs.T * ETAs * Xs)[0]) == 0
ok &= sp.expand(half_contr(FX, hodge(FX)) - (Xs.T * ETAs * Gs * Xs)[0]) == 0
ok &= sp.expand(half_contr(hodge(FX), hodge(FX)) + half_contr(FX, FX)) == 0
Frot = sp.cos(t) * FX + sp.sin(t) * hodge(FX)
ok &= sp.simplify(half_contr(Frot, Frot) - (sp.cos(2 * t) * half_contr(FX, FX) + sp.sin(2 * t) * half_contr(FX, hodge(FX)))) == 0
check("x2-N2", ok, "class of d2-2: with F_X of Thm 12.3, (+,-,-,-), eps_0123=+1: F_{GX}=*F_X, (1/2)F.F=eta(X,X), (1/2)F.*F=eta(X,GX), "
      "(1/2)*F.*F=-(1/2)F.F, hence D^T eta D=cos2t eta+sin2t eta G is the textbook rotation of F.F into F.*F (inherited, not fixed)")

# ---------------------------------------------------------------------- d2-5 / d2-6 / d2-7 stabilisers
def gen_matrix_cols(gens, X):
    return sp.Matrix.hstack(*[g_ * X for g_ in gens])


def X_of_z(z):
    return sp.Matrix([sp.re(z[0]), sp.re(z[1]), sp.re(z[2]), sp.im(z[0]), sp.im(z[1]), sp.im(z[2])])


def cayley(Avec):
    A_ = crossmat(Avec)
    return (sp.eye(3) - A_).inv() * (sp.eye(3) + A_)


def stab_data(z):
    X = X_of_z(z)
    N1 = gen_matrix_cols(lGb, X).nullspace()
    N2 = gen_matrix_cols(lGb + [Gs], X).nullspace()
    return X, N1, N2


ok = True
zlist_nonnull = [[1 + sp.I, 1, sp.I], [2, -1 + 3 * sp.I, sp.I], [3, 0, 0], [sp.I, 0, 0], [1, 1, 0]]
for _ in range(4):
    v = [sp.Integer(int(x)) for x in RNG.integers(-4, 5, 6)]
    z = [v[0] + sp.I * v[3], v[1] + sp.I * v[4], v[2] + sp.I * v[5]]
    if sp.expand(sum(w * w for w in z)) != 0:
        zlist_nonnull.append(z)
for z in zlist_nonnull:
    X, N1, N2 = stab_data(z)
    ok &= len(N1) == 2 and len(N2) == 2 and all(n_[6] == 0 for n_ in N2)
    qz = sp.expand(sum(w * w for w in z))
    # R = a x with a = z/sqrt(q): R^3 = -R ;  (iR)^3 = iR, checked on complex 3x3 matrices
    a = sp.Matrix(z) / sp.sqrt(qz)
    R = crossmat(a)
    ok &= sp.simplify(R ** 3 + R) == sp.zeros(3) and sp.simplify((sp.I * R) ** 3 - sp.I * R) == sp.zeros(3)
    ok &= sp.simplify(R * sp.Matrix(z)) == sp.zeros(3, 1)
check("x2-5", ok, f"non-null field points ({len(zlist_nonnull)} incl. random): Stab in l_G and in l_G+RG both 2-dim, no G component; "
      "R=a x (a=z/sqrt q) has R^3=-R, (iR)^3=iR and kills z")

# symbolic identity (z x)^3 = -q (z x) for generic complex z
z1, z2, z3 = sp.symbols('z1:4')
Zx = crossmat([z1, z2, z3])
ok_sym = sp.expand(Zx ** 3 + (z1 ** 2 + z2 ** 2 + z3 ** 2) * Zx) == sp.zeros(3)
ok_sym &= sp.expand(Zx ** 2 * sp.Matrix([sp.Symbol('w1'), sp.Symbol('w2'), sp.Symbol('w3')])
                    - sp.Matrix([z1, z2, z3]) * (z1 * sp.Symbol('w1') + z2 * sp.Symbol('w2') + z3 * sp.Symbol('w3'))
                    + (z1 ** 2 + z2 ** 2 + z3 ** 2) * sp.Matrix([sp.Symbol('w1'), sp.Symbol('w2'), sp.Symbol('w3')])) == sp.zeros(3, 1)

# null points: random rational rotations of (1,i,0) via Cayley; z=u+iv with n=u x v/|u|^2
ok = ok_sym
nulls = [[1, sp.I, 0], [1 + 2 * sp.I, 2 + sp.I, 2 - 2 * sp.I]]
for _ in range(4):
    Avec = [sp.Rational(int(x), int(y)) for x, y in zip(RNG.integers(-3, 4, 3), RNG.integers(1, 4, 3))]
    Rm = cayley(Avec)
    lam = int(RNG.integers(1, 4))
    nulls.append(list(lam * Rm * sp.Matrix([1, sp.I, 0])))
for z in nulls:
    z = [sp.nsimplify(w) for w in z]
    assert sp.simplify(sum(w * w for w in z)) == 0
    X, N1, N2 = stab_data(z)
    ok &= len(N1) == 2 and len(N2) == 3
    # null rotations: N^3=0 != N^2 for N=(lambda z)x
    N = crossmat(z)
    ok &= sp.simplify(N ** 3) == sp.zeros(3) and sp.simplify(N ** 2) != sp.zeros(3)
    uv = sp.Matrix([sp.re(w) for w in z])
    vv = sp.Matrix([sp.im(w) for w in z])
    n = uv.cross(vv) / uv.dot(uv)
    Ln = sum((n[i] * Lm[i] for i in range(3)), sp.zeros(6))
    ok &= sp.simplify((Ln + Gs) * X) == sp.zeros(6, 1)
    ok &= sp.simplify(n.dot(n) - 1) == 0
    # e(2) structure: derived algebra 2-dim abelian; ad(L_n+G) on it has char poly x^2+1
    basis = [sum((nv[i] * (lGb + [Gs])[i] for i in range(7)), sp.zeros(6)) for nv in N2]
    der = [br(basis[i], basis[j]) for i in range(3) for j in range(i + 1, 3)]
    rk = sp.Matrix([list(d) for d in der]).rank()
    ok &= rk == 2
    dbasis = [sp.Matrix(d) for d in sp.Matrix([list(d) for d in der]).rowspace()]
    dmats = [sp.Matrix(6, 6, list(d)) for d in dbasis]
    ok &= sp.simplify(br(dmats[0], dmats[1])) == sp.zeros(6)
    comp = Ln + Gs
    coords = []
    Mflat = sp.Matrix.hstack(*[sp.Matrix(list(dm)) for dm in dmats])
    for dm in dmats:
        img = sp.Matrix(list(br(comp, dm)))
        c_ = Mflat.solve_least_squares(img)
        ok &= sp.simplify(Mflat * c_ - img) == sp.zeros(36, 1)
        coords.append(list(c_))
    adm = sp.Matrix(coords).T
    x_ = sp.symbols('x_')
    ok &= sp.simplify(adm.charpoly(x_).as_expr() - (x_ ** 2 + 1)) == 0
# duality on (1,i,0) is a rotation: e^{tG}z = e^{-t L3} z
X1 = X_of_z([1, sp.I, 0])
expL3 = sp.simplify((-t * Lm[2]).exp())
ok &= sp.simplify(D * X1 - expL3 * X1) == sp.zeros(6, 1)
check("x2-6", ok, f"null field points ({len(nulls)} incl. random Cayley rotations): (z x)^3=-q(z x) symbolically; Stab in l_G 2-dim of null "
      "rotations (N^3=0!=N^2), in l_G+RG 3-dim containing L_n+G (n=u x v/|u|^2, |n|=1), derived algebra 2-dim abelian, "
      "ad(L_n+G) with char poly x^2+1: e(2); e^{tG}(1,i,0)=e^{-tL_3}(1,i,0)")

# Wigner in l_int on E
lint_biv = [(1, 4), (1, 5), (1, 6), (4, 5), (4, 6), (5, 6)]
lint6 = [adj6(blade(list(p))) for p in lint_biv]
e1v = sp.Matrix([1, 0, 0, 0, 0, 0])
nullv = sp.Matrix([1, 0, 0, 0, 0, 1])
Ns1 = gen_matrix_cols(lint6, e1v).nullspace()
Ns2 = gen_matrix_cols(lint6, nullv).nullspace()
ok = len(Ns1) == 3 and len(Ns2) == 3
b1 = [sum((nv[i] * lint6[i] for i in range(6)), sp.zeros(6)) for nv in Ns1]
ok &= all(bb.T == -bb for bb in b1)  # compact (antisymmetric) -> so(3)
ok &= sp.Matrix([list(br(b1[i], b1[j])) for i in range(3) for j in range(i + 1, 3)]).rank() == 3
b2 = [sum((nv[i] * lint6[i] for i in range(6)), sp.zeros(6)) for nv in Ns2]
der2 = [br(b2[i], b2[j]) for i in range(3) for j in range(i + 1, 3)]
ok &= sp.Matrix([list(d) for d in der2]).rank() == 2
check("x2-7", ok, "Wigner in l_int: Stab(e1) 3-dim, all antisymmetric, perfect (so(3)); Stab(e1+e6) 3-dim with 2-dim derived algebra (e(2))")

# ---------------------------------------------------------------------- d2-4 interior dynamics
m, k = sp.symbols('m kappa', positive=True)
xd = sp.symbols('xd1:7', real=True)
xdd = sp.symbols('xdd1:7', real=True)
Xd = sp.Matrix(xd)
up = Xd + k * Gs * Xs
Lag = sp.expand(m / 2 * (up.T * ETAs * up)[0])
pvec = sp.Matrix([sp.diff(Lag, v) for v in xd])
EL = []
for i in range(6):
    dtp = sum(sp.diff(pvec[i], xs[j]) * xd[j] + sp.diff(pvec[i], xd[j]) * xdd[j] for j in range(6))
    EL.append(sp.expand(dtp - sp.diff(Lag, xs[i])))
solxdd = sp.solve(EL, xdd, dict=True)[0]
ok = all(sp.simplify(solxdd[xdd[i]] + k ** 2 * xs[i]) == 0 for i in range(6))
ok &= sp.simplify(pvec - m * ETAs * up) == sp.zeros(6, 1)
Enoe = sp.expand(sum(pvec[i] * xd[i] for i in range(6)) - Lag)
ok &= sp.expand(Enoe - (m / 2 * (Xd.T * ETAs * Xd)[0] - m * k ** 2 / 2 * ((Gs * Xs).T * ETAs * (Gs * Xs))[0])) == 0
ok &= sp.expand(Enoe - (m / 2 * ((Xd.T * ETAs * Xd)[0] + k ** 2 * (Xs.T * ETAs * Xs)[0]))) == 0
zd = zvec(Xd)
EC = m / 2 * (qf(zd) + k ** 2 * qf(zX))
ok &= sp.expand(sp.re(sp.expand(EC)) - Enoe) == 0
# conservation along the flow: d/dtau EC with xdd = -k^2 x
dEC = sum(sp.diff(EC, xs[i]) * xd[i] + sp.diff(EC, xd[i]) * (-k ** 2 * xs[i]) for i in range(6))
ok &= sp.expand(dEC) == 0
# E_C = 2 m k^2 c_-.c_+ with c_+ e^{ik tau} = (z + zd/(ik))/2, c_- e^{-ik tau} = (z - zd/(ik))/2
cp = [(zX[i] + zd[i] / (sp.I * k)) / 2 for i in range(3)]
cm = [(zX[i] - zd[i] / (sp.I * k)) / 2 for i in range(3)]
ok &= sp.expand(EC - 2 * m * k ** 2 * sum(cm[i] * cp[i] for i in range(3))) == 0
# momentum <-> 2 i k c_+ e^{ik tau}
zp = zvec(up)
ok &= all(sp.expand(zp[i] - 2 * sp.I * k * cp[i]) == 0 for i in range(3))
p2 = sp.expand((m * ETAs * up).dot(m * ETAs * up))
ok &= sp.expand(p2 - 4 * m ** 2 * k ** 2 * sum(sp.expand(cp[i] * sp.conjugate(cp[i])) for i in range(3))) == 0
# duality multiplies E_C by e^{2it}
Dsub = dict(zip(list(xs) + list(xd), list(D * Xs) + list(D * Xd)))
ECD = EC.subs(Dsub, simultaneous=True)
ok &= sp.expand(ECD - (sp.cos(t) + sp.I * sp.sin(t)) ** 2 * EC) == 0
check("x2-4a", ok, "split-form interior from the Lagrangian: EL gives Xdd=-k^2 X, p=m eta(alpha_+X), Noether energy "
      "=(m/2)(eta(Xd,Xd)+k^2 eta(X,X)) = Re E_C, E_C=(m/2)(q(zd)+k^2 q(z)) conserved, =2mk^2 c_-.c_+, momentum <-> 2ik c_+e^{ik tau}, "
      "|p|^2=4m^2k^2|c_+|^2, duality multiplies E_C by e^{2it}")

# u along a general solution, and zero-momentum orbits
tau = sp.symbols('tau', real=True)
cr = sp.symbols('cr1:4 ci1:4 dr1:4 di1:4', real=True)
cmv = [cr[i] + sp.I * cr[3 + i] for i in range(3)]
cpv = [cr[6 + i] + sp.I * cr[9 + i] for i in range(3)]
zt = [cmv[i] * sp.exp(-sp.I * k * tau) + cpv[i] * sp.exp(sp.I * k * tau) for i in range(3)]
abs2 = sum(sp.expand(w * sp.conjugate(w)) for w in zt)
target = sum(sp.expand(cmv[i] * sp.conjugate(cmv[i]) + cpv[i] * sp.conjugate(cpv[i])) for i in range(3)) + \
    2 * sp.re(sp.expand(sum(sp.conjugate(cmv[i]) * cpv[i] for i in range(3)) * sp.exp(2 * sp.I * k * tau)))
ok = sp.simplify(sp.expand((abs2 - target).rewrite(sp.cos), complex=True)) == 0
# u constant iff conj(c_-).c_+ = 0: du/dtau = Re(2ik w e^{2ik tau}); vanishing at tau=0 and tau=pi/(4k) forces w=0
wsym = sp.symbols('w_r w_i', real=True)
wv = wsym[0] + sp.I * wsym[1]
du = lambda tt: sp.re(sp.expand(2 * sp.I * k * wv * sp.exp(2 * sp.I * k * tt)))
sol_w = sp.solve([sp.simplify(du(0)), sp.simplify(du(sp.pi / (4 * k)))], wsym, dict=True)
ok &= sol_w == [{wsym[0]: 0, wsym[1]: 0}]
# example with conj(c_-).c_+ != 0
ex_cm = [1, 0, 0]
ex_cp = [1, 0, 0]
ex_z = [ex_cm[i] * sp.exp(-sp.I * k * tau) + ex_cp[i] * sp.exp(sp.I * k * tau) for i in range(3)]
ex_u = sp.simplify(sum(sp.expand(w * sp.conjugate(w)) for w in ex_z) / 2)
ok &= sp.simplify(sp.diff(ex_u, tau)) != 0
# zero momentum orbit X = exp(-k G tau) X0
X0s = sp.Matrix(sp.symbols('y1:7', real=True))
Xzm = (sp.cos(k * tau) * sp.eye(6) - sp.sin(k * tau) * Gs) * X0s
Xzm_d = Xzm.diff(tau)
ok &= sp.simplify(Xzm_d + k * Gs * Xzm) == sp.zeros(6, 1)
ok &= sp.simplify(sp.expand((m / 2 * ((Xzm_d.T * ETAs * Xzm_d)[0] + k ** 2 * (Xzm.T * ETAs * Xzm)[0])))) == 0
ok &= sp.simplify(Xzm.dot(Xzm) - X0s.dot(X0s)) == 0
zzm = zvec(Xzm)
z0 = zvec(X0s)
ok &= all(sp.simplify(sp.expand(zzm[i] - sp.exp(-sp.I * k * tau) * z0[i], complex=True)) == 0 for i in range(3))
check("x2-4b", ok, "u(tau)=(|c_-|^2+|c_+|^2)/2+Re(conj(c_-).c_+ e^{2ik tau}); constant iff conj(c_-).c_+=0 (and an example where it is not); "
      "zero-momentum orbit X=e^{-kG tau}X0: p=0, Noether energy 0, u=|X0|^2/2 constant, z=e^{-ik tau}z0: u is not a function of |p|^2")

# the other pure sense: X = e^{+k G tau} X0 is also a duality rotation, but carries momentum, |p|^2 = 8 m^2 k^2 u
Xpl = (sp.cos(k * tau) * sp.eye(6) + sp.sin(k * tau) * Gs) * X0s
Xpl_d = Xpl.diff(tau)
ok = sp.simplify(Xpl.diff(tau, 2) + k ** 2 * Xpl) == sp.zeros(6, 1)
zpl = zvec(Xpl)
ok &= all(sp.simplify(sp.expand(zpl[i] - sp.exp(sp.I * k * tau) * z0[i], complex=True)) == 0 for i in range(3))
p_pl = m * ETAs * (Xpl_d + k * Gs * Xpl)
u_pl = Xpl.dot(Xpl) / 2
ok &= sp.simplify(p_pl.dot(p_pl) - 8 * m ** 2 * k ** 2 * u_pl) == 0 and sp.simplify(u_pl - X0s.dot(X0s) / 2) == 0
ok &= sp.simplify(m / 2 * ((Xpl_d.T * ETAs * Xpl_d)[0] + k ** 2 * (Xpl.T * ETAs * Xpl)[0])) == 0
check("x2-N3", ok, "the c_- = 0 sense X=e^{+kG tau}X0 is also a duality rotation (z=e^{ik tau}z0), with Noether energy 0 but |p|^2=8m^2k^2 u != 0: "
      "duality acts along both pure senses, not only along the zero-momentum orbits (corrects a phrase of d2 after Prop. d2:prop:interior)")

# ---------------------------------------------------------------------- d2-8 T invariant under Spin(3,3)
biv = list(itertools.combinations(range(1, 7), 2))
ok = True
for (a, b) in biv:
    X = blade([a, b])
    ok &= is_zero(X.T @ T + T @ X)
# blade method: antisymmetric blades with X^T M + M X = 0 for all 15 bivectors
sol_bl = []
for I in allblades:
    if bl_transpose_sign(I) != -1:
        continue
    good = True
    for (a, b) in biv:
        sX = bl_transpose_sign((a, b))
        # X^T M + M X = sX X M + M X = (sX * c + 1) M X, with X M = c M X
        cc = bl_sign_comm((a, b), I)
        if sX * cc + 1 != 0:
            good = False
            break
    if good:
        sol_bl.append(I)
ok &= sol_bl == [(1, 2, 3)]
# linear algebra method: 28 antisymmetric unknowns, sympy nullspace
basisA = []
for i in range(8):
    for j in range(i + 1, 8):
        Mm = np.zeros((8, 8), dtype=np.int64)
        Mm[i, j] = 1
        Mm[j, i] = -1
        basisA.append(Mm)
rows = []
for (a, b) in biv:
    X = blade([a, b])
    cols = [(X.T @ Mm + Mm @ X).flatten() for Mm in basisA]
    rows.extend(np.array(cols).T.tolist())
ns = sp.Matrix(rows).nullspace()
ok &= len(ns) == 1
Msol_s = sum((ns[0][i] * sp.Matrix(basisA[i]) for i in range(28)), sp.zeros(8))
ok &= Msol_s.rank() == 8 and sp.Matrix([list(Msol_s), list(sp.Matrix(T))]).rank() == 1
# group level at rational points: rotations (c,s), c^2+s^2=1; boosts (ch,sh), ch^2-sh^2=1
for (a, b) in biv:
    X = sp.Matrix(blade([a, b]))
    Ts = sp.Matrix(T)
    if bl_transpose_sign((a, b)) == -1:
        c_, s2 = sp.Rational(3, 5), sp.Rational(4, 5)
    else:
        c_, s2 = sp.Rational(5, 4), sp.Rational(3, 4)
    Sg = c_ * sp.eye(8) + s2 * X
    ok &= Sg.T * Ts * Sg == Ts
check("x2-8", ok, "T=e1e2e3: X^T T+TX=0 for all 15 bivectors; antisymmetric invariant kernels = R T by blade combinatorics "
      "and by sympy nullspace (1-dim); S^T T S=T at rational points of every rotation circle and boost hyperbola (Cor 18.4)")

# ---------------------------------------------------------------------- d2-8b covariance of kinetic kernels
Kmu = [-(g0 @ g) for g in gam]
ok = np.array_equal(Kmu[0], -I8) and all(np.array_equal(Kk.T, Kk) for Kk in Kmu)
nb = len(biv)
cols = []
for (a, b) in biv:
    X = blade([a, b])
    cols.append(np.concatenate([(X.T @ Kmu[mu] + Kmu[mu] @ X).flatten() for mu in range(4)]))
for mu in range(4):
    for nu in range(4):
        v = np.zeros(4 * 64, dtype=np.int64)
        v[mu * 64:(mu + 1) * 64] = -Kmu[nu].flatten()
        cols.append(v)
Amat = sp.Matrix(np.array(cols).T.tolist())
nsk = Amat.nullspace()
ok &= len(nsk) == 7
proj = sp.Matrix([list(v[:nb]) for v in nsk])
ok &= proj.rank() == 7
target_biv = [(1, 4), (1, 5), (1, 6), (4, 5), (4, 6), (5, 6), (2, 3)]
tmat = sp.Matrix([[1 if biv[i] == p else 0 for i in range(nb)] for p in target_biv])
ok &= sp.Matrix.vstack(proj, tmat).rank() == 7
check("x2-8b", ok, "kinetic kernels K^mu=-gamma^0 gamma^mu (K^0=-1, symmetric); bivectors X with X^T K^mu+K^mu X=lambda^mu_nu K^nu: "
      "7-dim, projecting injectively onto span(l_int bivectors, e2e3)")

# ---------------------------------------------------------------------- d2-9 massless algebra, orbit of T
# blade method: X^T K + K X = 0 is diagonal on blades
gm_bl = []
for I in allblades:
    X = BL[I]
    if all(is_zero(X.T @ Kk + Kk @ X) for Kk in Kmu):
        gm_bl.append(I)
rowsg = []
for Kk in Kmu:
    for p in range(8):
        for q_ in range(8):
            # (X^T K + K X)_{pq} = sum_r X_rp K_rq + K_pr X_rq
            row = np.zeros(64, dtype=np.int64)
            for r_ in range(8):
                row[r_ * 8 + p] += Kk[r_, q_]
                row[r_ * 8 + q_] += Kk[p, r_]
            rowsg.append(row.tolist())
ns_g = sp.Matrix(rowsg).nullspace()
gm_named = {"Om": Om, "J": J, "e2Om": e[2] @ Om, "e3Om": e[3] @ Om}
ok = len(ns_g) == 4 and len(gm_bl) == 4
span_ns = sp.Matrix([list(v) for v in ns_g])
span_named = sp.Matrix([M_.flatten().tolist() for M_ in gm_named.values()])
ok &= sp.Matrix.vstack(span_ns, span_named).rank() == 4
gl = list(gm_named.values())
ok &= all(int(np.trace(gl[i] @ gl[j])) == (-8 if i == j else 0) for i in range(4) for j in range(4))
# su(2) = span{J, e2Om, e3Om}: closed, Om central
su2 = gl[1:]
ok &= all(is_zero(Om @ X - X @ Om) for X in su2)
for X, Y in itertools.combinations(su2, 2):
    C_ = X @ Y - Y @ X
    coeffs = [Fraction(int(np.trace(C_ @ Z)), -8) for Z in su2]
    recon = sum((c_ * Z.astype(object) for c_, Z in zip(coeffs, su2)), np.zeros((8, 8), dtype=object))
    ok &= np.all(recon == C_.astype(object)) and any(c_ != 0 for c_ in coeffs)
# with mass invariance: only J
mass_ok = [nm for nm, X in gm_named.items() if is_zero(X.T @ T + T @ X)]
ok &= mass_ok == ["J"]
# combined linear solve (all of End(R^8)) with mass condition
rowsT = []
for p in range(8):
    for q_ in range(8):
        row = np.zeros(64, dtype=np.int64)
        for r_ in range(8):
            row[r_ * 8 + p] += T[r_, q_]
            row[r_ * 8 + q_] += T[p, r_]
        rowsT.append(row.tolist())
ns_gT = sp.Matrix(rowsg + rowsT).nullspace()
ok &= len(ns_gT) == 1 and sp.Matrix([list(ns_gT[0]), J.flatten().tolist()]).rank() == 1
# orbit map
imgs = [(X.T @ T + T @ X) for X in gl]
ok &= sp.Matrix([im.flatten().tolist() for im in imgs]).rank() == 3
named_img = [T @ Om, T @ e[2] @ Om, T @ e[3] @ Om]
ok &= sp.Matrix([im.flatten().tolist() for im in imgs] + [x.flatten().tolist() for x in named_img]).rank() == 3
ok &= is_zero(J @ named_img[0] - named_img[0] @ J)
ok &= all(is_zero(J @ x + x @ J) for x in named_img[1:])
# interior kernel space (Def 18.2): antisymmetric, commute with l_int rotations, anticommute with boosts
rot_int = [(4, 5), (4, 6), (5, 6)]
boost_int = [(1, 4), (1, 5), (1, 6)]
Kint = []
for I in allblades:
    if bl_transpose_sign(I) != -1:
        continue
    if all(bl_sign_comm(I, r_) == 1 for r_ in rot_int) and all(bl_sign_comm(I, b_) == -1 for b_ in boost_int):
        Kint.append(I)
ok &= len(Kint) == 6
Kint_m = [BL[I] for I in Kint]
ok &= sp.Matrix([x.flatten().tolist() for x in Kint_m] + [x.flatten().tolist() for x in named_img + [T]]).rank() == 6
check("x2-9", ok, "massless algebra: X^T K^mu+K^mu X=0 over End(R^8): 4-dim (sympy nullspace and blade enumeration agree) = "
      "span{Om,J,e2Om,e3Om}, trace form -8*1, Om central, su(2) closed; with T-invariance only RJ; orbit of T rank 3 = "
      "span{T Om (J-linear), T e2Om, T e3Om (J-antilinear)}; l_int kernel space 6-dim (blades) contains T and the orbit")
note("l_int kernel blades: " + ", ".join("e" + "".join(map(str, I)) for I in Kint))

# ---------------------------------------------------------------------- d2-10 field equation and chiral angle
p1, p2, p3, ms, mp = sp.symbols('p1 p2 p3 m_s m_p', real=True)
pv = [p1, p2, p3]
eS = [None] + [sp.Matrix(x) for x in e[1:]]
Js, Oms, Ts = sp.Matrix(J), sp.Matrix(Om), sp.Matrix(T)
Hs = sum((pv[a] * eS[1] * eS[a + 4] for a in range(3)), sp.zeros(8)) + eS[1] * (ms * sp.eye(8) + mp * Oms)
Ms_ = Ts * (ms * sp.eye(8) + mp * Oms)
Kms = [sp.Matrix(Kk) for Kk in Kmu]
ok = Hs.T == Hs and Hs * Js == Js * Hs and Ms_.T == -Ms_
ok &= sp.expand(Hs * Hs - (p1 ** 2 + p2 ** 2 + p3 ** 2 + ms ** 2 + mp ** 2) * sp.eye(8)) == sp.zeros(8)
# K^0 dot Psi + K^a J p_a Psi = M Psi  =>  Psi_dot = (K^a J p_a - M) Psi  must equal -J H Psi
ok &= sp.expand(sum((Kms[a + 1] * Js * pv[a] for a in range(3)), sp.zeros(8)) - Ms_ + Js * Hs) == sp.zeros(8)
# chiral rotation at rational points: R = c + s Om
okc = True
for (cc, ss) in [(sp.Rational(3, 5), sp.Rational(4, 5)), (sp.Rational(-5, 13), sp.Rational(12, 13))]:
    R = cc * sp.eye(8) + ss * Oms
    okc &= all(R.T * Kq * R == Kq for Kq in Kms)
    okc &= sp.expand(R.T * Ms_ * R - Ms_ * ((cc ** 2 - ss ** 2) * sp.eye(8) + 2 * cc * ss * Oms)) == sp.zeros(8)
# stabiliser of M != 0: M R^2 = M forces R^2 = 1 since (m_s+m_p Om)(m_s-m_p Om) = (m_s^2+m_p^2) 1
okc &= sp.expand((ms * sp.eye(8) + mp * Oms) * (ms * sp.eye(8) - mp * Oms) - (ms ** 2 + mp ** 2) * sp.eye(8)) == sp.zeros(8)
cc_, ss_ = sp.symbols('c s', real=True)
solR = sp.solve([cc_ ** 2 - ss_ ** 2 - 1, 2 * cc_ * ss_, cc_ ** 2 + ss_ ** 2 - 1], [cc_, ss_], dict=True)
okc &= sorted([(d[cc_], d[ss_]) for d in solR]) == [(-1, 0), (1, 0)]
check("x2-10", ok and okc, "M=T(m_s+m_p Om) antisymmetric; the Grassmann field equation K^mu d_mu Psi=M Psi with d_a=J p_a is "
      "Psi_dot=-JH Psi, H=gamma^0gamma^a p_a+gamma^0(m_s+m_p Om) symmetric, J-linear, H^2=(p^2+m_s^2+m_p^2)1; e^{alpha Om} "
      "keeps K^mu and sends M->M e^{2 alpha Om} (rational points); stabiliser of M!=0 is {+-1}")

# ---------------------------------------------------------------------- d2-11 rest flow
msym = sp.symbols('m_D', positive=True)
H0 = msym * eS[1]
ok = sp.expand(Js * H0 - msym * Ts) == sp.zeros(8)
ok &= is_zero(T @ (I8 + e[1]) - J @ (I8 + e[1])) and is_zero(T @ (I8 - e[1]) + J @ (I8 - e[1]))
ok &= np.linalg.matrix_rank(I8 + e[1]) == 4 and is_zero(J @ (I8 + e[1]) - (I8 + e[1]) @ J)
ok &= np.array_equal(-T @ J, g0)
lam = sp.symbols('lambda', real=True)
flow = sp.cos(msym * lam) * sp.eye(8) - sp.sin(msym * lam) * Ts
ok &= sp.simplify(flow.diff(lam) + msym * Ts * flow) == sp.zeros(8) and flow.subs(lam, 0) == sp.eye(8)
check("x2-11", ok, "rest flow: J H(0)=mT, exp(-m lambda T)=cos-sin T; T=+J on e1=+1 and -J on e1=-1 (each 4-dim, J-invariant); gamma^0=-TJ")

# ---------------------------------------------------------------------- d2-12 C, P, T at rest
phi = sp.symbols('phi', real=True)
Cs = sp.cos(phi) * eS[2] + sp.sin(phi) * eS[3]
Hp = sum((pv[a] * eS[1] * eS[a + 4] for a in range(3)), sp.zeros(8)) + msym * eS[1]
Hm = Hp.subs({p1: -p1, p2: -p2, p3: -p3}, simultaneous=True)
ok = sp.simplify(Cs * Cs - sp.eye(8)) == sp.zeros(8)
ok &= sp.simplify(Cs * Ts - Ts * Cs) == sp.zeros(8) and sp.simplify(Cs * Js + Js * Cs) == sp.zeros(8)
ok &= sp.simplify(Cs * eS[1] + eS[1] * Cs) == sp.zeros(8)
ok &= sp.simplify(Cs.T * H0 * Cs + H0) == sp.zeros(8)
ok &= sp.simplify(Cs * Hp * Cs + Hm) == sp.zeros(8)   # Thm 5.1 (C^-1 = C)
Ps = [eS[1], eS[1] * Js]
Trev = sp.Matrix(blade([2, 4, 5, 6]))
ok &= Trev == eS[2] * eS[4] * eS[5] * eS[6]
for Pm in Ps:
    ok &= sp.simplify(Pm * Hp * Pm.inv() - Hm) == sp.zeros(8) and Pm * Js == Js * Pm
ok &= sp.simplify(Trev * Hp * Trev.inv() - Hm) == sp.zeros(8) and Trev * Js == -Js * Trev
# rest-energy sign under composites
ops = {"C": sp.Matrix(e[2]), "P": eS[1], "T": Trev}
signs = {}
for combo in ["C", "P", "T", "CP", "CT", "PT", "CPT"]:
    Mc = sp.eye(8)
    for ch in combo:
        Mc = Mc * ops[ch]
    Hc = Mc.inv() * H0 * Mc
    signs[combo] = 1 if Hc == H0 else (-1 if Hc == -H0 else 0)
ok &= signs == {"C": -1, "P": 1, "T": 1, "CP": -1, "CT": -1, "PT": 1, "CPT": -1}
# C maps the e1=+1 solution exp(-m lam J) Psi0 to exp(+m lam J) C Psi0 on e1=-1
Psi0 = sp.Matrix(I8 + e[1])[:, 0]
sol_plus = (sp.cos(msym * lam) * sp.eye(8) - sp.sin(msym * lam) * Js) * Psi0
img = sp.Matrix(e[2]) * sol_plus
target2 = (sp.cos(msym * lam) * sp.eye(8) + sp.sin(msym * lam) * Js) * sp.Matrix(e[2]) * Psi0
ok &= sp.simplify(img - target2) == sp.zeros(8, 1) and sp.Matrix(e[1]) * sp.Matrix(e[2]) * Psi0 == -sp.Matrix(e[2]) * Psi0
ok &= sp.simplify(img.diff(lam) + Js * H0 * img) == sp.zeros(8, 1)
check("x2-12", ok, "C=cos(phi)e2+sin(phi)e3: C^2=1, [C,T]=0, {C,J}={C,e1}=0, C^T H(0)C=-H(0), CH(p)C^-1=-H(-p); P=gamma^0, gamma^0 J and "
      "T=e2e4e5e6 satisfy XH(p)X^-1=H(-p) (Thm 5.3); rest-energy sign reversed by C,CP,CT,CPT and kept by P,T,PT; "
      "C carries the e1=+1 solution to the e1=-1 solution of the same flow")
note(f"rest-energy signs: {signs}")

# ---------------------------------------------------------------------- d2-13/14 frame P, C, T
xf = [sp.Function(f'x{i}')(tau) for i in range(1, 7)]
Xf = sp.Matrix(xf)
kap = sp.symbols('kappa_s', real=True)


def Lk(Xv, kk):
    upv = Xv.diff(tau) + kk * Gs * Xv
    return sp.expand(m / 2 * (upv.T * ETAs * upv)[0])


ok = (ETAs * Gs * ETAs.inv() == -Gs)
ok &= sp.simplify(Lk(ETAs * Xf, kap) - Lk(Xf, -kap)) == 0
EdotB = sum(xf[i] * xf[i + 3] for i in range(3))
ok &= sp.simplify(Lk(Xf, kap) - Lk(Xf, -kap) + 2 * m * kap * sp.diff(EdotB, tau)) == 0
ok &= sp.simplify(Lk(-Xf, kap) - Lk(Xf, kap)) == 0
Xrev = -ETAs * Xf.subs(tau, -tau)
LT = Lk(Xrev, kap)
ok &= sp.simplify(LT - Lk(Xf, kap).subs(tau, -tau)) == 0
# action on c_-+
ck = [sp.Symbol('cm', complex=True), sp.Symbol('cp', complex=True)]
zt1 = ck[0] * sp.exp(-sp.I * k * tau) + ck[1] * sp.exp(sp.I * k * tau)
Pz = sp.conjugate(zt1)
Cz = -zt1
Tz = -sp.conjugate(zt1.subs(tau, -tau))


def coeffs_of(expr):
    ex = sp.expand(expr)
    return (sp.simplify(ex.coeff(sp.exp(-sp.I * k * tau))), sp.simplify(ex.coeff(sp.exp(sp.I * k * tau))))


cP, cC, cT = coeffs_of(Pz), coeffs_of(Cz), coeffs_of(Tz)
ok &= cP == (sp.conjugate(ck[1]), sp.conjugate(ck[0])) and cC == (-ck[0], -ck[1]) and cT == (-sp.conjugate(ck[0]), -sp.conjugate(ck[1]))
# P on z is conjugation (E -> -E), T on z is -conj with tau reversed
ok &= all(sp.expand(zvec(ETAs * Xs)[i] - sp.conjugate(zX[i])) == 0 for i in range(3))
check("x2-13", ok, "frame: P=eta has PGP^-1=-G, L_k(PX)=L_{-k}(X), L_k-L_{-k}=-2mk d(E.B)/dtau; X->-X and X(tau)->-eta X(-tau) keep L_k; "
      "on c_-+: P->(conj c_+, conj c_-), C->(-c_-,-c_+), T->(-conj c_-,-conj c_+)")


# composite senses on the frame: compose the verified maps on (c_-, c_+) symbolically
def frame_apply(name, pair):
    a_, b_ = pair
    if name == "P":
        return (sp.conjugate(b_), sp.conjugate(a_))
    if name == "C":
        return (-a_, -b_)
    return (-sp.conjugate(a_), -sp.conjugate(b_))


def exch(combo):
    pair = (ck[0], ck[1])
    for ch in reversed(combo):
        pair = frame_apply(ch, pair)
    new_minus = pair[0]
    return new_minus.has(ck[1]) and not new_minus.has(ck[0])


frame_rev = [c for c in ["C", "P", "T", "CP", "CT", "PT", "CPT"] if exch(c)]
dirac_rev = [c for c, s_v in signs.items() if s_v == -1]
check("x2-14", frame_rev == ["P", "CP", "PT", "CPT"] and dirac_rev == ["C", "CP", "CT", "CPT"],
      f"sense reversal: Dirac rest rotation by {dirac_rev} (from the matrices), frame binding by {frame_rev}; both CPT-odd")

# ---------------------------------------------------------------------- d2-15 module, rho
paulis = {"1": np.eye(2, dtype=np.int64), "x": sx, "z": sz, "e": ep}
words = []
for w in itertools.product("1xze", repeat=4):
    M_ = kron(*[paulis[c] for c in w])
    if np.array_equal(M_.T, -M_) and np.array_equal(M_ @ M_, -np.eye(16, dtype=np.int64)):
        words.append(M_)


def search(chosen, start):
    if len(chosen) == 8:
        return chosen
    for i in range(start, len(words)):
        W = words[i]
        if all(is_zero(W @ C_ + C_ @ W) for C_ in chosen):
            r_ = search(chosen + [W], i + 1)
            if r_:
                return r_
    return None


th = search([], 0)
ok = th is not None and len(th) == 8
I16 = np.eye(16, dtype=np.int64)
ok &= all(np.array_equal(th[i] @ th[j] + th[j] @ th[i], -2 * I16 * (i == j)) for i in range(8) for j in range(8))
mons = []
for r_ in range(9):
    for I in itertools.combinations(range(8), r_):
        M_ = I16.copy()
        for i in I:
            M_ = M_ @ th[i]
        mons.append(M_.flatten())
ok &= np.linalg.matrix_rank(np.array(mons, dtype=float)) == 256
check("x2-15a", ok, "eight antisymmetric 16x16 integer theta_i (4-fold tensor words), theta_i^2=-1, pairwise anticommuting; "
      "256 monomials of rank 256 (commutant R*1, Thm 5.6)")


def rho(A):
    out = np.zeros((16, 16), dtype=object)
    for i in range(8):
        for j in range(8):
            if A[i][j] != 0:
                out = out + Fraction(-1, 4) * A[i][j] * (th[i] @ th[j]).astype(object)
    return out


def tofloat(M_):
    return np.array(M_, dtype=float)


# homomorphism on random antisymmetric integer matrices (not the basis pairs)
ok = True
for _ in range(6):
    A1 = RNG.integers(-3, 4, (8, 8))
    A1 = A1 - A1.T
    A2 = RNG.integers(-3, 4, (8, 8))
    A2 = A2 - A2.T
    R1, R2 = rho(A1.tolist()), rho(A2.tolist())
    lhs = R1.dot(R2) - R2.dot(R1)
    rhs = rho((A1 @ A2 - A2 @ A1).tolist())
    ok &= np.all(lhs == rhs)
    for kk in range(8):
        comm = R1.dot(th[kk].astype(object)) - th[kk].astype(object).dot(R1)
        ok &= np.all(comm == sum((int(A1[j, kk]) * th[j] for j in range(8)), np.zeros((16, 16), dtype=np.int64)).astype(object))
check("x2-15b", ok, "rho(A)=-(1/4)sum A_ij theta_i theta_j is a Lie homomorphism with [rho(A),theta_k]=sum_j A_jk theta_j "
      "(exact Fractions, six random integer pairs)")


# ---------------------------------------------------------------------- d2-16 Fock spectrum (new momenta)
def H_of(p, mm):
    return sum(p[a] * (e[1] @ e[a + 4]) for a in range(3)) + mm * e[1]


xx = sp.symbols('xx')
ok = True
cases = [((1, 2, 2), 4, 5), ((0, 0, 0), 2, 2), ((2, 6, 9), 0, 11), ((1, 1, 1), 1, 2)]
for (p, mm, Ev) in cases:
    Hm_ = H_of(p, mm)
    assert np.array_equal(Hm_ @ Hm_, Ev ** 2 * I8)
    RJ = sp.Matrix(2 * rho(J.tolist()))
    RJH = sp.Matrix(2 * rho((J @ Hm_).tolist()))
    ok &= sp.factor(RJ.charpoly(xx).as_expr()) == sp.factor(xx ** 6 * (xx ** 2 + 4) ** 4 * (xx ** 2 + 16))
    ok &= sp.expand(RJH.charpoly(xx).as_expr() - xx ** 6 * (xx ** 2 + 4 * Ev ** 2) ** 4 * (xx ** 2 + 16 * Ev ** 2)) == 0
    # joint spectrum numerically: Q=-i rho(J), H=-i rho(JH) commute; diagonalise a generic combination
    Q = -1j * tofloat(rho(J.tolist()))
    Hf = -1j * tofloat(rho((J @ Hm_).tolist()))
    ok &= np.allclose(Q @ Hf, Hf @ Q, atol=1e-12)
    w, V = np.linalg.eigh(Hf + 0.1234567 * Q)
    hv = np.real(np.einsum('ij,jk,ki->i', V.conj().T, Hf, V))
    qv = np.real(np.einsum('ij,jk,ki->i', V.conj().T, Q, V))
    ok &= np.allclose(np.round(hv / Ev), hv / Ev, atol=1e-8) and np.allclose(np.round(qv), qv, atol=1e-8)
    table = {}
    for h_, q_ in zip(np.round(hv / Ev).astype(int), np.round(qv).astype(int)):
        table[(h_, q_)] = table.get((h_, q_), 0) + 1
    expected = {(-2, 0): 1, (-1, 1): 2, (-1, -1): 2, (0, 2): 1, (0, -2): 1, (0, 0): 4, (1, 1): 2, (1, -1): 2, (2, 0): 1}
    ok &= table == expected
check("x2-16", ok, "Fock space of one momentum at (p,m,E)=((1,2,2),4,5),(0,2,2),((2,6,9),0,11),((1,1,1),1,2): exact char polys "
      "x^6(x^2+4)^4(x^2+16), x^6(x^2+4E^2)^4(x^2+16E^2); joint (H/E,Q) table by numerical simultaneous diagonalisation (tol 1e-8) "
      "equals Thm 5.7: unique (-2E,0) vacuum, multiplicities 1,4,6,4,1; at p=0 the vacuum is -2m")

# ---------------------------------------------------------------------- d2-17 lift of C by the exponential map
from scipy.linalg import expm
C8 = e[2].astype(float)
wC, VC = np.linalg.eigh(C8)
Vm = VC[:, wC < 0]          # -1 eigenspace (dim 4)
ok = Vm.shape[1] == 4
# complex structure A on the -1 eigenspace: pair basis vectors; C = exp(pi A)
Acs = np.zeros((8, 8))
for kk in range(2):
    a_, b_ = Vm[:, 2 * kk], Vm[:, 2 * kk + 1]
    Acs += np.outer(b_, a_) - np.outer(a_, b_)
ok &= np.allclose(expm(np.pi * Acs), C8, atol=1e-10)


def rho_f(A):
    out = np.zeros((16, 16))
    for i in range(8):
        for j in range(8):
            out += -0.25 * A[i, j] * (th[i] @ th[j])
    return out


Uc = expm(np.pi * rho_f(Acs))
ok &= np.allclose(Uc @ Uc.T, np.eye(16), atol=1e-10) and np.allclose(Uc @ Uc, np.eye(16), atol=1e-10)
ok &= all(np.allclose(Uc @ th[kk] @ Uc.T, sum(C8[j, kk] * th[j] for j in range(8)), atol=1e-10) for kk in range(8))
Qr = rho_f(J.astype(float))
ok &= np.allclose(Uc @ Qr @ Uc.T, -Qr, atol=1e-10)
for p in [(1, 2, 2), (0, 3, 1)]:
    Hp_ = rho_f((J @ H_of(p, 3)).astype(float))
    Hm2 = rho_f((J @ H_of(tuple(-x for x in p), 3)).astype(float))
    ok &= np.allclose(Uc @ Hp_ @ Uc.T, Hm2, atol=1e-10)
# +1 eigenspace of U (real) meets ker rho(J) in real dim 3
wU, VU = np.linalg.eigh((Uc + Uc.T) / 2)
Vp = VU[:, wU > 0]
ok &= Vp.shape[1] == 8
s_int = np.linalg.svd(Qr @ Vp, compute_uv=False)
ok &= int(np.sum(s_int < 1e-9)) == 3
check("x2-17", ok, "C=e2 lifted by the exponential map U=exp(pi rho(A)), C=exp(pi A): U orthogonal, U^2=1, U theta_k U^-1=sum C_jk theta_j, "
      "UQU^-1=-Q, UH(p)U^-1=H(-p); +1 eigenspace of U real dim 8, meeting ker Q in real dim 3 (Prop 5.8; numerics tol 1e-9)")

# ---------------------------------------------------------------------- d2-18/19 frame vacuum
y, yd, py = sp.symbols('y yd p_y', real=True)
Lsp = -m / 2 * (yd ** 2 - k ** 2 * y ** 2)
pys = sp.diff(Lsp, yd)
ydsol = sp.solve(sp.Eq(py, pys), yd)[0]
Hsp = sp.simplify((py * yd - Lsp).subs(yd, ydsol))
ok = sp.simplify(pys + m * yd) == 0 and sp.simplify(Hsp + (py ** 2 / (2 * m) + m * k ** 2 * y ** 2 / 2)) == 0
# truncated ladder
hbar = sp.symbols('hbar', positive=True)
Nc = 12
a = sp.zeros(Nc)
for n_ in range(1, Nc):
    a[n_ - 1, n_] = sp.sqrt(n_)
ad_ = a.T
xop = sp.sqrt(hbar / (2 * m * k)) * (a + ad_)
pop = sp.I * sp.sqrt(hbar * m * k / 2) * (ad_ - a)
Hosc = sp.simplify(pop * pop / (2 * m) + m * k ** 2 * xop * xop / 2)
for n_ in range(Nc - 1):
    ok &= sp.simplify(Hosc[n_, n_] - hbar * k * (n_ + sp.Rational(1, 2))) == 0
    ok &= all(sp.simplify(Hosc[n_, j_]) == 0 for j_ in range(Nc - 1) if j_ != n_)
ok &= sp.simplify((xop * pop - pop * xop)[0:Nc - 1, 0:Nc - 1] - sp.I * hbar * sp.eye(Nc - 1)) == sp.zeros(Nc - 1)
# zero-point energies of all 20 coordinate three-planes and the full space
zp_table = {}
for W in itertools.combinations(range(1, 7), 3):
    pp = sum(1 for a_ in W if a_ <= 3)
    qq = 3 - pp
    zp_table.setdefault((pp, qq), set()).add(sp.Rational(pp - qq, 2))
ok &= zp_table == {(3, 0): {sp.Rational(3, 2)}, (2, 1): {sp.Rational(1, 2)}, (1, 2): {sp.Rational(-1, 2)}, (0, 3): {sp.Rational(-3, 2)}}
ok &= sp.Rational(3 - 3, 2) == 0
# energy on a subfield: the restriction of (m/2)(eta(Xd,Xd)+k^2 eta(X,X)) is sum eta_aa (m/2)(xd_a^2+k^2 x_a^2)
ok &= sp.expand(m / 2 * ((Xd.T * ETAs * Xd)[0] + k ** 2 * (Xs.T * ETAs * Xs)[0])
                - sum(ETA[i, i] * m / 2 * (xd[i] ** 2 + k ** 2 * xs[i] ** 2) for i in range(6))) == 0
check("x2-18", ok, "spatial mode: p_y=-m yd, H=-(p^2/2m+mk^2y^2/2) (Legendre); x,p ladder: p^2/2m+mk^2x^2/2=hbar k(n+1/2) and [x,p]=i hbar "
      "below the cut (N=12); zero-point (hbar k/2)(p-q): +3/2 (T), +1/2 (all (2,1) planes), -1/2 ((1,2)), -3/2 (S), 0 on R^{3,3} (Thm 18.6)")

# spectrum of the full frame under positive norm: hbar k (sum n_T - sum n_S)
levels = {}
nmax = 4
for nT in itertools.product(range(nmax + 1), repeat=3):
    for nS in itertools.product(range(nmax + 1), repeat=3):
        lv = sum(nT) - sum(nS)
        levels[lv] = levels.get(lv, 0) + 1
ok = set(levels) == set(range(-3 * nmax, 3 * nmax + 1))
deg0 = []
for nm_ in [2, 3, 4, 5]:
    cnt = sum(1 for nT in itertools.product(range(nm_ + 1), repeat=3) for nS in itertools.product(range(nm_ + 1), repeat=3)
              if sum(nT) == sum(nS))
    deg0.append(cnt)
ok &= all(deg0[i] < deg0[i + 1] for i in range(len(deg0) - 1))
check("x2-19", ok, f"positive-norm frame spectrum: every integer level in [-{3 * nmax},{3 * nmax}] reached, zero-level degeneracy "
      f"{deg0} at cutoffs 2..5, unbounded: no ground state")

# ---------------------------------------------------------------------- d2-20 Majorana phase
Pg = (I8 + Gam)
ok = True
# kernels in span(Kint) with Gamma M = M = M Gamma
cvars = sp.symbols('c0:6')
Mgen = sum((cvars[i] * sp.Matrix(Kint_m[i]) for i in range(6)), sp.zeros(8))
Gs8 = sp.Matrix(Gam)
eqsM = list(Gs8 * Mgen - Mgen) + list(Mgen * Gs8 - Mgen)
solM = sp.linsolve(eqsM, cvars)
solv = list(solM)[0]
freeM = sorted(set().union(*[x.free_symbols for x in solv]), key=str)
ok &= len(freeM) == 2
Mbasis = []
for fv in freeM:
    sub = {f: (1 if f == fv else 0) for f in freeM}
    Mbasis.append(sp.Matrix(sum((solv[i].subs(sub) * sp.Matrix(Kint_m[i]) for i in range(6)), sp.zeros(8))))
# complex line: J M in the space
JM = Js * Mbasis[0]
ok &= sp.Matrix([list(Mbasis[0]), list(Mbasis[1]), list(JM)]).rank() == 2
gmS = [sp.Matrix(x) for x in gl]
for Mb in Mbasis + [Mbasis[0] + 2 * Mbasis[1]]:
    ok &= Mb * Js == -Js * Mb and Mb.rank() == 4 and Mb.T == -Mb
    MtM = Mb.T * Mb
    cval = MtM.trace() / 4
    ok &= MtM == cval * (sp.eye(8) + Gs8) / 2
    ok &= sp.simplify((MtM * MtM).trace() / MtM.trace() ** 2 - sp.Rational(1, 4)) == 0
    # stabiliser in g_massless
    ss_ = sp.symbols('g0:4')
    Xg = sum((ss_[i] * gmS[i] for i in range(4)), sp.zeros(8))
    solS = sp.linsolve(list(Xg.T * Mb + Mb * Xg), ss_)
    sv = list(solS)[0]
    fr = sorted(set().union(*[x.free_symbols for x in sv]), key=str)
    ok &= len(fr) == 1
    Xstab = sum((sv[i].subs(fr[0], 1) * gmS[i] for i in range(4)), sp.zeros(8))
    ok &= sp.Matrix([list(Xstab), list(Js - Js * Gs8)]).rank() == 1
ok &= Js - Js * Gs8 == Js + Oms
# Dirac mass keeps J, moves J - J Gamma; Majorana the reverse
ok &= (Js.T * Ts + Ts * Js == sp.zeros(8)) and ((Js + Oms).T * Ts + Ts * (Js + Oms) != sp.zeros(8))
ok &= (Js.T * Mbasis[0] + Mbasis[0] * Js != sp.zeros(8))
check("x2-20", ok, "Majorana phase (Thm 18.5b): l_int kernels on Gamma=+1 form a real 2-plane closed under J (complex line); each "
      "anticommutes with J, rank 4, M^T M=c(1+Gamma)/2, Q/N^2=1/4, stabiliser in g_massless R(J-J Gamma)=R(J+Om); T keeps J, moves J+Om")

# ====================================================================== three constituents
import scipy.sparse as sps

Om_bar = blade([1, 2, 3, 4, 5, 6])  # omega = e1...e6
I8f = np.eye(8, dtype=np.int64)
E3 = [None]
for kc in range(3):
    for a_ in range(1, 7):
        fac = [Om_bar] * kc + [e[a_]] + [I8f] * (2 - kc)
        E3.append(sps.csr_matrix(kron(*fac)))
etad3 = [None] + [1, 1, 1, -1, -1, -1] * 3
I512s = sps.identity(512, dtype=np.int64, format='csr')
ok = all((E3[A] @ E3[B_] + E3[B_] @ E3[A] - 2 * etad3[A] * (A == B_) * I512s).count_nonzero() == 0
         for A in range(1, 19) for B_ in range(A, 19))


def prods(idx):
    M_ = I512s.copy()
    for a_ in idx:
        M_ = M_ @ E3[a_]
    return M_


def prod(idx):
    return prods(idx).toarray().astype(float)


def isint(M_):
    return np.all(M_ == np.rint(M_)) and np.max(np.abs(M_)) < 2 ** 50


def eq(A_, B_):
    return np.array_equal(A_, B_)


I512 = np.eye(512)
Eint = [1, 4, 5, 6]
outside = [a_ for a_ in range(1, 19) if a_ not in Eint]
J3 = prod(outside)
ok &= eq(J3 @ J3, -I512) and eq(J3.T, -J3)
Gam3 = J3 @ prod([1, 4, 5, 6])
chiW = prod([7, 13, 14, 15])
chiC = prod([10, 11, 12, 16, 17, 18])
qA = prod([2, 3])
Ps8 = (I512 + Gam3 @ chiW) @ (I512 - J3 @ chiC) @ (I512 - J3 @ qA)   # = 8 P_*
ok &= eq(Ps8 @ Ps8, 8 * Ps8) and eq(Ps8.T, Ps8) and int(round(np.trace(Ps8))) == 8 * 64
Hc = -(prod([10, 16]) + prod([11, 17]) + prod([12, 18]))
w_ = [None, 7, 13, 14, 15]
tau2 = {1: -prod([w_[2], w_[3]]), 2: -prod([w_[3], w_[1]]), 3: -prod([w_[1], w_[2]])}   # = 2 tau_i, s08: tau_i=-(1/2)w_j w_k
ok &= all(eq(tau2[i] @ tau2[i], -I512) for i in (1, 2, 3))
ok &= eq(tau2[1] @ tau2[2] - tau2[2] @ tau2[1], 2 * tau2[3])          # [tau_1,tau_2]=tau_3
PR2 = I512 + chiW
PL2 = I512 - chiW
tR4 = {i: PR2 @ tau2[i] for i in (1, 2, 3)}     # = 4 t^R_i
tL4 = {i: PL2 @ tau2[i] for i in (1, 2, 3)}     # = 4 t^L_i
Pg2 = I512 + Gam3                                # = 2 P(Gamma=+1)
YR4 = -J3 @ tR4[3]                               # = 4 T^R_3, symmetric
ok &= eq(YR4.T, YR4)
PT4 = PR2 + YR4                                  # = 4 [ (1/2)(1+chi_W)/... ] = 4 * projector onto T^R_3 = +1/2
ok &= eq(PT4 @ PT4, 4 * PT4)
s_list = [J3 @ prod([10, 16]), J3 @ prod([11, 17]), J3 @ prod([12, 18])]
ok &= all(eq(sv @ sv, I512) and eq(sv.T, sv) for sv in s_list)
ok &= eq(-(J3 @ Hc), sum(s_list))
PBL8 = (I512 - s_list[0]) @ (I512 - s_list[1]) @ (I512 - s_list[2])   # = 8 * projector onto B-L = -(1/3)JH = -1
ok &= eq(PBL8 @ PBL8, 8 * PBL8)
Pnu = Pg2 @ PT4 @ PBL8          # = 64 P_nu
ok &= eq(Pnu @ Pnu, 64 * Pnu) and eq(Pnu @ Ps8, Ps8 @ Pnu)
rank_Pnu = int(round(np.trace(Pnu))) // 64
blade0 = prod([1, 3, 9, 13, 14, 16, 17, 18])     # gamma^0 E3 E9 E13 E14 E16 E17 E18
Phi_s = Pnu @ Ps8 @ blade0 @ Ps8 @ Pnu           # = 64*8*8*64 Phi0
ok &= isint(Phi_s / 64)
Phi = Phi_s / 64                                 # = 4096 Phi0
ok &= isint(Phi) and eq(Phi.T, -Phi) and eq(J3 @ Phi, -(Phi @ J3))
PtP1 = (Phi.T @ Phi) / 4096 ** 2                 # = Phi0^T Phi0
ok &= isint(PtP1 * 512) and eq(PtP1 @ PtP1, PtP1) and int(round(np.trace(PtP1))) == 4
ok &= eq(PtP1 * 512, Pnu @ Ps8)                  # = P_nu P_*
ok &= eq(tR4[3] @ Phi - Phi @ tR4[3], 4 * (J3 @ Phi))        # [t^R_3,Phi0] = J Phi0
ok &= eq(Hc @ Phi - Phi @ Hc, -6 * (J3 @ Phi))               # [H/3,Phi0] = -2 J Phi0
rank_Phi_block = np.linalg.matrix_rank(Phi)
check("x2-21a", ok, f"Cl(9,9) on R^512 rebuilt from sA (sparse signed permutations; omega=e1...e6 tower; J the ordered product of the 14 "
      f"generators outside E; tau_i=-(1/2)w_j w_k cyclic as in s08): P_* rank 64; Phi0 antisymmetric, {{J,Phi0}}=0, Phi0^T Phi0 = P_nu P_* "
      f"(rank 4; P_nu alone has rank {rank_Pnu} on R^512), [t^R_3,Phi0]=J Phi0, [H/3,Phi0]=-2J Phi0 (Thm 19.2 v), integer-exact in float64")

Ccol = [10, 11, 12, 16, 17, 18]
gens21 = [4 * prod([a_, b_]) for a_, b_ in itertools.combinations(Ccol, 2)]
for i in (1, 2, 3):
    gens21 += [tL4[i], tR4[i]]
gens23 = gens21 + [4 * J3, 4 * (J3 @ Gam3)]
ok = all(eq(X @ Ps8, Ps8 @ X) and eq(X @ J3, J3 @ X) for X in gens23)
tang = [(X @ Phi - Phi @ X) for X in gens23]
ok &= all(isint(tt) for tt in tang)
cJ = J3 @ Phi - Phi @ J3
cJG = (J3 @ Gam3) @ Phi - Phi @ (J3 @ Gam3)
ctR = tR4[3] @ Phi - Phi @ tR4[3]          # = 4 * 4096 [t^R_3, Phi0]
ok &= eq(cJ, cJG) and eq(2 * cJ, ctR) and np.any(cJ)
Tm = np.array([np.rint(tt).astype(np.int64).flatten() for tt in tang], dtype=np.int64)
gram = Tm @ Tm.T
r21 = sp.Matrix(gram[:21, :21].tolist()).rank()
r23 = sp.Matrix(gram.tolist()).rank()
ok &= r21 == 9 and r23 == 9
check("x2-21", ok, f"no Majoron (Thm 22.2): [J,Phi0]=[J Gamma,Phi0]=2[t^R_3,Phi0]!=0; tangent spaces at Phi0 of the 21-orbit and 23-orbit "
      f"have dims {r21} and {r23} (int64 Gram matrices, sympy rank)")

JmJG = J3 - J3 @ Gam3
ok = eq(JmJG @ Phi, Phi @ JmJG)
Kd = Ps8 @ prod([1, 7, 8]) @ Ps8      # 64 * P_* E1E7E8 P_*
ok &= np.any(Kd) and eq(Kd.T, -Kd)
rots = [Ps8 @ prod([a_, b_]) @ Ps8 for a_, b_ in [(4, 5), (4, 6), (5, 6)]]
boosts = [Ps8 @ prod([1, a_]) @ Ps8 for a_ in (4, 5, 6)]
ok &= all(eq(Kd @ R_, R_ @ Kd) for R_ in rots) and all(eq(Kd @ B_, -(B_ @ Kd)) for B_ in boosts)
ok &= eq(Kd @ J3, J3 @ Kd) and eq(Kd @ Gam3, -(Gam3 @ Kd))
ok &= eq(JmJG @ Kd - Kd @ JmJG, -2 * (J3 @ Gam3 @ Kd)) and np.any(J3 @ Gam3 @ Kd)
ok &= all(eq(Kd @ prod([a_, b_]), prod([a_, b_]) @ Kd) for a_, b_ in itertools.combinations(Ccol, 2))
check("x2-22", ok, "U(1)_{1-Gamma}: [J-J Gamma,Phi0]=0; K=P_*E1E7E8P_* is a Lorentz-invariant, antisymmetric, J-linear, Gamma-odd, "
      "colour-singlet kernel with [J-J Gamma,K]=-2J Gamma K!=0")

# the Pati-Salam condensate gives mass to nu_R only: rank of Phi0 on the 64-dim block
ok = rank_Phi_block == 4
check("x2-N1", ok, f"rank Phi0 = {rank_Phi_block} on the 64-dim block (Phi0^T Phi0 = P_nu P_*): the condensate Phi0 masses only nu_R; "
      "the charged fermions stay massless until the bidoublet stage (Thm 21.1(d), Prop 21.2), contrary to d2's sentence "
      "'the fermion masses come from the Yukawa coupling to the same condensate'")


# ---------------------------------------------------------------------- d2-24 (cited by d2): Theorem 22.3 re-derived
temporal3 = {1, 2, 3, 7, 8, 9, 13, 14, 15}


def bcomm(I, K):
    I, K = set(I), set(K)
    return (len(I) * len(K) - len(I & K)) % 2 == 0


def btsign(I):
    r_ = len(I)
    s = (-1) ** (r_ * (r_ - 1) // 2)
    for a_ in I:
        s *= (1 if a_ in temporal3 else -1)
    return s


set_out = set(outside)
set_all = set(range(1, 19))
inv_sets = [set_all ^ {7, 13, 14, 15}, set_out ^ set(Ccol), set_out ^ {2, 3}]
kern_blades = []
for r_ in range(15):
    for I in itertools.combinations(sorted(set_out), r_):
        for pre in ([1], [4, 5, 6]):
            Bset = tuple(sorted(set(pre) | set(I)))
            if btsign(Bset) == -1 and all(bcomm(Bset, S_) for S_ in inv_sets):
                kern_blades.append(Bset)
n_kern = len(kern_blades)
dirac_singlet = [Bb for Bb in kern_blades if bcomm(Bb, set_out) and not bcomm(Bb, set_all)
                 and all(bcomm(Bb, (a_, b_)) for a_, b_ in itertools.combinations(Ccol, 2))]
comp = []
for Bb in dirac_singlet:
    Mc_ = Ps8 @ prod(list(Bb)) @ Ps8
    if np.any(Mc_):
        comp.append(Mc_)
Cmat = np.array([c_.flatten() for c_ in comp])
rk_bid = np.linalg.matrix_rank(Cmat)
# basis of the (1,2,2) space
_, _, Vt = np.linalg.svd(Cmat, full_matrices=False)
# pick rows of comp forming a basis (greedy, exact rank via integer Gram)
basis_bid = []
for c_ in comp:
    trial = basis_bid + [c_]
    Gt = np.array([[int(round(np.vdot(x, y))) for y in trial] for x in trial], dtype=object)
    if sp.Matrix(Gt.tolist()).rank() == len(trial):
        basis_bid.append(c_)
ok = n_kern == 2176 and rk_bid == 8 and len(basis_bid) == 8
ok &= all(eq(Kb.T, -Kb) and eq(Kb @ J3, J3 @ Kb) and eq(Kb @ Gam3, -(Gam3 @ Kb)) for Kb in basis_bid)
ok &= all(all(eq(Kb @ R_, R_ @ Kb) for R_ in rots) and all(eq(Kb @ B_, -(B_ @ Kb)) for B_ in boosts) for Kb in basis_bid)
# charges on the block: T^L_3 = -J t^L_3, Y = T^R_3 + (1/2)(B-L), B-L = -(1/3) J H ; scaled Q12 = 12 Q
TL4 = -J3 @ tL4[3]                     # 4 T^L_3
BL3 = -(J3 @ Hc)                        # 3 (B-L)
Q12 = 3 * TL4 + 3 * YR4 + 2 * BL3       # 12 Q = 12 T^L_3 + 12 T^R_3 + 6 (B-L)
QP = Q12 @ Ps8 / 8
wq = np.linalg.eigvalsh(QP)
qvals = sorted(set(np.round(wq[np.abs(wq) > 1e-9] / 12, 6)))
# Q-commuting part of the (1,2,2) space
cf = sp.symbols('f0:8')
comm_cols = np.array([(Kb @ Q12 - Q12 @ Kb).flatten() for Kb in basis_bid]).T
nz = np.where(np.any(comm_cols != 0, axis=1))[0]
nsQ = sp.Matrix(np.rint(comm_cols[nz]).astype(np.int64).tolist()).nullspace()
ok &= len(nsQ) == 4
KQ = [sum(float(v[i]) * basis_bid[i] for i in range(8)) for v in nsQ]
# sector projector onto d and e (Q = -1/3, -1) on the block, by Lagrange interpolation in 12Q on im P_*
Pblk = Ps8 / 8
qs12 = [8, -4, 0, -12]                  # 12 Q for u, d, nu, e
def lag(target):
    Mx = Pblk.copy()
    for qq in qs12:
        if qq != target:
            Mx = Mx @ (QP - qq * I512) / (target - qq)
    return Mx
Pde = lag(-4) + lag(-12)
Puv = lag(8) + lag(0)
ok &= np.allclose(Pde + Puv, Pblk, atol=1e-9)
# up-type: K Pde = 0 ; down-type: K Puv = 0  (within the Q-commuting 4-space)
def subspace(P_):
    cols = np.array([(Kq @ P_).flatten() for Kq in KQ]).T
    _, sv, vt = np.linalg.svd(cols, full_matrices=False)
    null = vt[sv < 1e-8 * sv.max()] if np.any(sv < 1e-8 * sv.max()) else np.zeros((0, len(KQ)))
    return [sum(v_[i] * KQ[i] for i in range(len(KQ))) for v_ in null]
Kup = subspace(Pde)
Kdn = subspace(Puv)
ok &= len(Kup) == 2 and len(Kdn) == 2
K1u, K1d = Kup[0] / np.abs(Kup[0]).max(), Kdn[0] / np.abs(Kdn[0]).max()
Phi0f = Phi / 4096


def orbit_dims(vR, ku, kd):
    Phv = vR * Phi0f + ku * K1u + kd * K1d
    tg = np.array([(X @ Phv - Phv @ X).flatten() for X in gens23])
    s21 = np.linalg.svd(tg[:21], compute_uv=False)
    s23 = np.linalg.svd(tg, compute_uv=False)
    tol = 1e-8
    return int(np.sum(s21 > tol * s23.max())), int(np.sum(s23 > tol * s23.max())), (s21, s23)


d_gen = orbit_dims(1.0, 0.37, 0.61)
d_gen2 = orbit_dims(2.0, 0.53, -0.29)
d_kd0 = orbit_dims(1.0, 0.37, 0.0)
d_vr0 = orbit_dims(0.0, 0.37, 0.61)
ok &= d_gen[:2] == (12, 13) and d_gen2[:2] == (12, 13) and d_kd0[:2] == (12, 12) and d_vr0[:2] == (5, 6)
note(f"kernel blades {n_kern}; Dirac colour-singlet span rank {rk_bid}; Q values on block {qvals}; up/down-type dims "
     f"{len(Kup)}/{len(Kdn)}; (21-orbit, 23-orbit) generic {d_gen[:2]}, {d_gen2[:2]}, kd=0 {d_kd0[:2]}, vR=0 {d_vr0[:2]}; "
     f"smallest kept / largest dropped singular value (generic, 23) {min(x for x in d_gen[2][1] if x > 1e-8 * d_gen[2][1].max()):.3e} / "
     f"{max([x for x in d_gen[2][1] if x <= 1e-8 * d_gen[2][1].max()] + [0.0]):.3e}")
check("x2-24", ok, "Thm 22.3 re-derived (d2 cites it): 2176 kernel blades (Thm 19.2 i); the Dirac-type colour-singlet compressions span the "
      "8-dim (1,2,2); 4 of them commute with Q, split 2+2 into up- and down-type; at Phi=vR Phi0+ku K1u+kd K1d the (21,23)-orbit dims "
      "are (12,13) generically (one uneaten direction), (12,12) at kd=0, (5,6) at vR=0 (numerical SVD rank, rel. tol 1e-8)")

# ---------------------------------------------------------------------- d2-23
ok = sp.simplify((k * Gs) ** 2 + k ** 2 * sp.eye(6)) == sp.zeros(6)
check("x2-23", ok, "(kappa G)^2=-kappa^2 1 = -(c/ell)^2 1: a restatement of G^2=-1 and the definition kappa=c/ell (no content beyond Thm 16.2)")

if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL x2 CHECKS PASSED")
sys.exit(0)
