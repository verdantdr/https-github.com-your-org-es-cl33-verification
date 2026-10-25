#!/usr/bin/env python3
"""Independent referee check of sections/r2_fourmomentum.tex (the four-momentum with X on the spatial triple).

Written independently of verify/r2_fourmomentum.py (nothing is imported from it).  Different methods are used where
possible: the Clifford generators are built from the bit-string rule of Appendix A of ES17 (not from Kronecker
products), clocks and frames come from the Cayley transform of rational skew matrices (not quaternions), intertwiner
spaces are null spaces of Kronecker systems, signatures are counted exactly by Descartes' rule on characteristic
polynomials, field integrals are done by numerical quadrature and the boosts of fields by an explicit Lorentz
transformation of (E, B) and of the coordinates.

Every line reads "PASS s2-<id>: <claim>"; a claim of the section found false appears as a check of the CORRECT
statement.  Exact sympy arithmetic for algebraic statements; floating point only where stated, with the tolerance in
the line.  Exit status 0 iff every check passes.  Run from any directory:  python3 s2_check.py
"""
import sys
import random
import numpy as np
import sympy as sp
from scipy import integrate
from scipy.linalg import expm
from sympy.polys.matrices import DomainMatrix

FAILS = []


def report(cid, ok, text):
    ok = bool(ok)
    print(("PASS" if ok else "FAIL") + f" s2-{cid}: {text}")
    if not ok:
        FAILS.append(cid)


def is_zero(M):
    if isinstance(M, sp.MatrixBase):
        return all(sp.simplify(x) == 0 for x in M)
    return sp.simplify(M) == 0


def rank_q(M):
    return DomainMatrix.from_Matrix(sp.Matrix(M)).convert_to(sp.QQ).rank()


def null_dim(rows, ncols):
    if not rows:
        return ncols
    M = sp.Matrix.vstack(*rows)
    return ncols - rank_q(M)


def inertia(S):
    """Exact (n_plus, n_minus, n_zero) of a rational symmetric matrix by Descartes' rule (real-rooted char. poly)."""
    lam = sp.Symbol("lam")
    p = sp.Poly(sp.Matrix(S).charpoly(lam).as_expr(), lam)
    coeffs = p.all_coeffs()
    nzero = 0
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
        nzero += 1

    def changes(cs):
        cs = [c for c in cs if c != 0]
        return sum(1 for a, b in zip(cs, cs[1:]) if a * b < 0)
    npos = changes(coeffs)
    deg = len(coeffs) - 1
    neg_coeffs = [c * (-1) ** (deg - k) for k, c in enumerate(coeffs)]
    nneg = changes(neg_coeffs)
    return npos, nneg, nzero


# ----------------------------------------------------------------------------- Clifford algebra from bit strings
def gen_bits(a):
    """e_{a+1} on R^8 from the rule of ES17 Appendix A: e_i|b> = (-1)^{b_1+..+b_{i-1}}|b+d_i>, e_{i+3} with b_i included."""
    i = a % 3
    spatial = a >= 3
    M = sp.zeros(8, 8)
    for b in range(8):
        bits = [(b >> (2 - k)) & 1 for k in range(3)]
        ex = sum(bits[:i]) + (bits[i] if spatial else 0)
        M[b ^ (1 << (2 - i)), b] = (-1) ** ex
    return M


E = [gen_bits(a) for a in range(6)]
ETA = sp.diag(1, 1, 1, -1, -1, -1)
I8, I6, I4 = sp.eye(8), sp.eye(6), sp.eye(4)
G6 = sp.zeros(6, 6)
for i in range(3):
    G6[i + 3, i] = 1
    G6[i, i + 3] = -1
KC6 = sp.zeros(6, 6)          # the section's para-complex K: e_i -> e_{i+3}, e_{i+3} -> e_i
for i in range(3):
    KC6[i + 3, i] = 1
    KC6[i, i + 3] = 1
PIP = sp.diag(1, 1, 1, 0, 0, 0)
PIM = sp.diag(0, 0, 0, 1, 1, 1)


def cl(v):
    return sum((v[A] * E[A] for A in range(6)), sp.zeros(8, 8))


def comps(X):
    """components of a grade-one Clifford element"""
    return sp.Matrix([sp.nsimplify((X * E[A]).trace() / (8 * ETA[A, A])) for A in range(6)])


def eta(v, w):
    return (v.T * ETA * w)[0, 0]


def wedge(a, b):
    """action of the bivector a^b on V: v -> a eta(b,v) - b eta(a,v)"""
    return a * (ETA * b).T - b * (ETA * a).T


def tvec(t):
    return sp.Matrix(list(t) + [0, 0, 0])


def svec(s):
    return sp.Matrix([0, 0, 0] + list(s))


def unitv(k):
    v = sp.zeros(6, 1)
    v[k] = 1
    return v


def cayley_frame(a, b, c):
    A = sp.Matrix([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    R = (sp.eye(3) - A).inv() * (sp.eye(3) + A)
    return [tvec(R[:, k]) for k in range(3)], R


FRAMES = [cayley_frame(sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(-1, 5)),
          cayley_frame(sp.Rational(-2, 3), sp.Rational(1, 4), sp.Rational(3, 7)),
          cayley_frame(sp.Rational(0), sp.Rational(0), sp.Rational(0))]


def projectors(n):
    P = n * (ETA * n).T + PIM
    Q = PIP - n * (ETA * n).T
    return P, Q


def S_(a):
    return unitv(a + 3)


L_cl, K_cl = [], []
for i in range(3):
    j, k = (i + 1) % 3, (i + 2) % 3
    L_cl.append(sp.Rational(1, 2) * (E[j + 3] * E[k + 3] - E[j] * E[k]))
    K_cl.append(sp.Rational(1, 2) * (E[k] * E[j + 3] - E[j] * E[k + 3]))


def ad_on_V(x):
    return sp.Matrix.hstack(*[comps(x * E[A] - E[A] * x) for A in range(6)])


LG = [ad_on_V(x) for x in L_cl + K_cl]          # lG on V: L1,L2,L3,K1,K2,K3


def lint_gens(n):
    basis = [n] + [S_(a) for a in range(3)]
    return [wedge(basis[p], basis[q]) for p in range(4) for q in range(p + 1, 4)]


# ============================================================================= checks
def chk_conventions():
    kron = lambda a, b, c: sp.kronecker_product(sp.kronecker_product(a, b), c)
    sx, sz, ep, I2 = sp.Matrix([[0, 1], [1, 0]]), sp.diag(1, -1), sp.Matrix([[0, -1], [1, 0]]), sp.eye(2)
    ek = [kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx), kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
    ok = all(E[a] == ek[a] for a in range(6))
    ok &= all(E[a] * E[b] + E[b] * E[a] == 2 * ETA[a, b] * I8 for a in range(6) for b in range(6))
    om = [E[i] * E[i + 3] for i in range(3)]
    ok &= om[0] == kron(sz, I2, I2) and om[1] == kron(I2, sz, I2) and om[2] == kron(I2, I2, sz)
    # a^b acts as (1/2)[ab, v] for orthogonal a, b
    v = sp.Matrix(sp.symbols("v1:7"))
    for (a, b) in [(0, 1), (0, 4), (3, 5), (2, 3)]:
        lhs = wedge(unitv(a), unitv(b)) * v
        rhs = comps(sp.Rational(1, 2) * ((E[a] * E[b]) * cl(v) - cl(v) * (E[a] * E[b])))
        ok &= is_zero(lhs - rhs)
    ok &= G6 * G6 == -I6 and G6.T * ETA == ETA * G6 and G6.T * ETA * G6 == -ETA
    ok &= KC6 * KC6 == I6 and KC6.T * ETA == -ETA * KC6
    # ES17 Lemma 3.4: the grading root is K = (1/2) ad_P, P = sum omega_i: K e_i = -e_{i+3}, so K = -(section's K)
    Psum = om[0] + om[1] + om[2]
    Kpaper = sp.Matrix.hstack(*[comps(sp.Rational(1, 2) * (Psum * E[A] - E[A] * Psum)) for A in range(6)])
    ok_k = Kpaper == -KC6
    report("0a", ok and ok_k,
           "the bit-string generators of ES17 App. A equal the Kronecker realisation (1); Clifford relations; omega_i = "
           "sz(x)1(x)1, 1(x)sz(x)1, 1(x)1(x)sz; a^b acts as (1/2)[ab,.]; G^2=-1, G eta-self-adjoint, anti-isometry; the "
           "section's K (e_i->e_{i+3}) has K^2=1 and is eta-skew, and ES17's grading root (1/2)ad_P of Lemma 3.4 is -K")
    # Fock modes
    ap = [sp.Rational(1, 2) * (E[i] + E[i + 3]) for i in range(3)]
    am = [sp.Rational(1, 2) * (E[i] - E[i + 3]) for i in range(3)]
    ok = all(ap[i] * am[j] + am[j] * ap[i] == (I8 if i == j else sp.zeros(8, 8)) for i in range(3) for j in range(3))
    ok &= all(ap[i] * ap[j] + ap[j] * ap[i] == sp.zeros(8, 8) for i in range(3) for j in range(3))
    N = sum((ap[i] * am[i] for i in range(3)), sp.zeros(8, 8))
    ok &= N.is_diagonal()
    for b in range(8):
        signs = [om[i][b, b] for i in range(3)]
        ok &= N[b, b] == sum(1 for s in signs if s == -1)
    vac = [b for b in range(8) if N[b, b] == 0]
    full = [b for b in range(8) if N[b, b] == 3]
    ok &= vac == [0] and full == [7] and all(om[i][7, 7] == -1 for i in range(3))
    report("0b", ok,
           "a+-_i=(e_i+-e_{i+3})/2 obey the CAR; N=sum a+_i a-_i is diagonal and counts the minus signs of the omega_i; "
           "the (+,+,+) line is the vacuum N=0 and the (-,-,-) line is the filled line N=3")


def chk_planes():
    ok = True
    for (n, u1, u2), R in FRAMES:
        P, Q = projectors(n)
        ok &= P * P == P and Q * Q == Q and P + Q == I6 and P * Q == sp.zeros(6, 6)
        ok &= (ETA * P).T == ETA * P and (ETA * Q).T == ETA * Q and P.T * ETA * Q == sp.zeros(6, 6)
        ok &= rank_q(P) == 4 and rank_q(Q) == 2 and P * n == n and Q * u1 == u1 and Q * u2 == u2
        ok &= all(P * S_(a) == S_(a) for a in range(3))
        ok &= inertia(sp.Matrix([[eta(x, y) for y in [n] + [S_(a) for a in range(3)]] for x in [n] + [S_(a) for a in range(3)]])) == (1, 3, 0)
        ok &= inertia(sp.Matrix([[eta(x, y) for y in (u1, u2)] for x in (u1, u2)])) == (2, 0, 0)
        ok &= R.det() == 1
    # every four-plane containing V- is E_n: V- + R v contains n = (temporal part of v)/|.|
    v = sp.Matrix([2, 3, 6, 5, -1, 4])
    n = tvec(sp.Matrix([2, 3, 6]) / 7)
    P, Q = projectors(n)
    ok &= P * v == v and Q * v == sp.zeros(6, 1)
    # Theorem 28.1(ii) of ES17 for the other Lorentzian type: a (3,1) four-plane has a NEGATIVE normal plane
    four31 = [unitv(0), unitv(1), unitv(2), unitv(3)]
    ok &= inertia(sp.Matrix([[eta(x, y) for y in four31] for x in four31])) == (3, 1, 0)
    ok &= inertia(sp.Matrix([[eta(x, y) for y in (unitv(4), unitv(5))] for x in (unitv(4), unitv(5))])) == (0, 2, 0)
    report("1", ok,
           "three Cayley frames: P_n, Q_n complementary eta-orthogonal eta-self-adjoint projectors of ranks 4, 2; E_n of "
           "signature (1,3), U_n positive; V- + R(2,3,6;5,-1,4) = E_(2,3,6)/7; the normal plane has the sign of time for a "
           "(1,3) four-plane, while for a (3,1) four-plane such as <e1,e2,e3,e4> it is negative (ES17 Thm 28.1(ii))")


def chk_covariance():
    ok = True
    so33 = [wedge(unitv(a), unitv(b)) for a in range(6) for b in range(a + 1, 6)]
    for (n, u1, u2), R in FRAMES[:2]:
        P, Q = projectors(n)
        coeffs = sp.symbols("c0:15")
        X = sum((coeffs[k] * so33[k] for k in range(15)), sp.zeros(6, 6))
        eqs = list(X * P - P * X)
        sol = sp.linsolve(eqs, coeffs)
        (s,) = sol
        free = set().union(*[sp.sympify(x).free_symbols for x in s])
        ok &= len(free) == 7
        gens = lint_gens(n) + [wedge(u1, u2)]
        ok &= all(g * P == P * g for g in gens)
        cross = [wedge(a, b) for a in [n] + [S_(k) for k in range(3)] for b in (u1, u2)]
        comms = [(g * P - P * g).reshape(36, 1) for g in cross]
        ok &= rank_q(sp.Matrix.hstack(*comms)) == 8
    phi = sp.Symbol("phi", real=True)
    B = wedge(unitv(0), unitv(1))
    expB = I6 + sp.sin(phi) * B + (1 - sp.cos(phi)) * B * B
    ok &= is_zero(B ** 3 + B)
    Kv = expB * unitv(0)          # rest momentum (E/c) e1 with E/c = 1
    ok &= is_zero(eta(unitv(1), Kv) + sp.sin(phi))
    report("2", ok,
           "two Cayley clocks: the commutant of P_n in so(3,3) is 7-dimensional (solved over all 15 bivectors), contains "
           "lint(n)+so(U_n); the 8 commutators [g,P_n], g in E_n^U_n, are independent; exp(phi e1^e2) moves (E/c)e1 to "
           "q_1 = -(E/c) sin(phi)")


def chk_J():
    n1, n2, n3 = sp.symbols("n1 n2 n3", real=True)
    nv = tvec([n1, n2, n3])
    IT = E[0] * E[1] * E[2]
    Jn = IT * cl(nv)
    ok = IT * IT == -I8 and all(IT * E[a] == E[a] * IT for a in range(3)) and all(IT * E[a] == -E[a] * IT for a in range(3, 6))
    ok &= is_zero(Jn - (n1 * E[1] * E[2] + n2 * E[2] * E[0] + n3 * E[0] * E[1]))
    ok &= is_zero(Jn * Jn + (n1 ** 2 + n2 ** 2 + n3 ** 2) * I8)
    ok &= is_zero(Jn * cl(nv) - cl(nv) * Jn) and all(is_zero(Jn * E[a] - E[a] * Jn) for a in range(3, 6))
    for (n, u1, u2), R in FRAMES:
        ok &= cl(n) * cl(u1) * cl(u2) == IT and IT * cl(n) == cl(u1) * cl(u2)
        Om = cl(n) * E[3] * E[4] * E[5]
        Omm = cl(-n) * E[3] * E[4] * E[5]
        Jm = IT * cl(-n)
        Pm, Qm = projectors(-n)
        P, Q = projectors(n)
        ok &= Pm == P and Qm == Q and Jm == -(IT * cl(n)) and Omm == -Om
        Gam = (IT * cl(n)) * Om
        ok &= Gam == Jm * Omm and Gam == E[0] * E[1] * E[2] * E[3] * E[4] * E[5]
        # energy read by n reverses: E(-n) = c eta(-n, K) = -E(n)
        Ks = sp.Matrix(sp.symbols("k1:7"))
        ok &= is_zero(eta(-n, Ks) + eta(n, Ks))
    report("3", ok,
           "I_T=e1e2e3: I_T^2=-1, commutes with V+, anticommutes with V-; J(n)=I_T n = n1e2e3+n2e3e1+n3e1e2, J(n)^2=-|n|^2, "
           "[J(n),n]=[J(n),s_a]=0 (symbolic n); n u1 u2 = I_T and J(n)=u1u2 for three Cayley frames; n->-n keeps P_n, Q_n and "
           "reverses J(n), Omega_n and the energy E(n) read by the clock; Gamma_n = J(n)Omega_n = e1...e6 for every clock")


def chk_dirac():
    ok = True
    IT = E[0] * E[1] * E[2]
    clocks = [tvec(sp.Matrix([2, -6, 9]) / 11), tvec(sp.Matrix([12, -4, 3]) / 13)] + [FRAMES[0][0][0]]
    for n in clocks:
        gam = [cl(n)] + [E[a] for a in range(3, 6)]
        Jn = IT * cl(n)
        Om = gam[0] * gam[1] * gam[2] * gam[3]
        biv = [gam[p] * gam[q] for p in range(4) for q in range(p + 1, 4)]
        xs = sp.symbols("x0:64")
        X = sp.Matrix(8, 8, xs)
        rows = []
        for g in biv + [Jn]:
            rows.append(sp.Matrix(list(X * g - g * X)).jacobian(xs))
        dim_joint = null_dim(rows, 64)
        span = sp.Matrix.hstack(*[M.reshape(64, 1) for M in (I8, Om, Jn, Om * Jn)])
        ok &= dim_joint == 4 and rank_q(span) == 4
        ok &= all(M * g == g * M for M in (I8, Om, Jn, Om * Jn) for g in biv + [Jn])
        a, b, c, d = sp.symbols("a b c d", real=True)
        x = a * I8 + b * Om + c * Jn + d * Om * Jn
        sols = sp.solve(list(x * x + I8), [a, b, c, d], dict=True)
        roots = sorted([(s.get(a, a), s.get(b, b), s.get(c, c), s.get(d, d)) for s in sols])
        ok &= roots == sorted([(0, 1, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, -1, 0)])
        ok &= all(Jn * g == g * Jn for g in gam) and not all(Om * g == g * Om for g in gam)
        rows = [sp.Matrix(list(X * g - g * X)).jacobian(xs) for g in gam]
        ok &= null_dim(rows, 64) == 4
        rows.append(sp.Matrix(list(X * IT - IT * X)).jacobian(xs))
        ok &= null_dim(rows, 64) == 2
        # mass form, kinetic kernels, Hamiltonian
        ok &= cl(n) * Jn == IT and all(IT * Jn * g == -cl(n) * g for g in gam)
        p1, p2, p3, m, cc = sp.symbols("p1 p2 p3 m c", positive=True)
        H = cc * cl(n) * (p1 * E[3] + p2 * E[4] + p3 * E[5]) + m * cc ** 2 * cl(n)
        ok &= is_zero(H - H.T) and is_zero(H * Jn - Jn * H)
        ok &= is_zero(H * H - (cc ** 2 * (p1 ** 2 + p2 ** 2 + p3 ** 2) + m ** 2 * cc ** 4) * I8)
    # the quaternionic commutant contains u_a Omega_n (Cayley frame)
    (n, u1, u2), R = FRAMES[0]
    gam = [cl(n)] + [E[a] for a in range(3, 6)]
    Om = gam[0] * gam[1] * gam[2] * gam[3]
    ok &= all((cl(u) * Om) * g == g * (cl(u) * Om) for u in (u1, u2) for g in gam)
    ok &= all((cl(u) * Om) * IT == -IT * (cl(u) * Om) for u in (u1, u2))
    report("4", ok,
           "ES17 Thm 4.1 at the clocks (2,-6,9)/11, (12,-4,3)/13 and a Cayley clock, by solving the 64 linear equations: the "
           "joint commutant of lint(n) and J(n) is span{1,Omega_n,J(n),Omega_nJ(n)}; x^2=-1 has exactly the roots +-Omega_n, "
           "+-J(n); only +-J(n) commute with the gamma's; the gamma commutant is 4-dim (contains u_aOmega_n) and adding I_T "
           "leaves span{1,J(n)}; n J(n) = I_T; I_T J(n) gamma^mu = -n gamma^mu; H_n(p) symmetric, commutes with J(n), "
           "H_n^2 = c^2|p|^2+m^2c^4")


def chk_poisson():
    Y = sp.Matrix(sp.symbols("Y1:7"))
    K = sp.Matrix(sp.symbols("K1:7"))
    ok = True
    for (n, u1, u2), R in FRAMES[:2]:
        Ks = ETA.inv() * K
        ctn = eta(n, Y)
        xa = [-eta(S_(a), Y) for a in range(3)]
        ua = [eta(u1, Y), eta(u2, Y)]
        Ec = eta(n, Ks)
        pa = [-eta(S_(a), Ks) for a in range(3)]
        qa = [eta(u1, Ks), eta(u2, Ks)]
        coords = [ctn] + xa + ua + [Ec] + pa + qa

        def pb(f, g):
            return sp.expand(sum(sp.diff(f, Y[A]) * sp.diff(g, K[A]) - sp.diff(f, K[A]) * sp.diff(g, Y[A]) for A in range(6)))
        Bm = sp.Matrix(12, 12, lambda r, s: pb(coords[r], coords[s]))
        target = sp.zeros(12, 12)
        target[0, 6] = 1
        for a in range(3):
            target[1 + a, 7 + a] = -1
        for a in range(2):
            target[4 + a, 10 + a] = 1
        target = target - target.T
        ok &= Bm == target
        # the phase-space form K_A dY^A = E dt_n - p.dx + q.du
        dY = sp.Matrix(sp.symbols("dY1:7"))
        lhs = (K.T * dY)[0, 0]
        rhs = Ec * eta(n, dY) - sum(pa[a] * (-eta(S_(a), dY)) for a in range(3)) + qa[0] * eta(u1, dY) + qa[1] * eta(u2, dY)
        ok &= sp.expand(lhs - rhs) == 0
    # Noether from the bracket table: {f,H} = sum_l {f,z_l} dH/dz_l
    z = sp.symbols("ct x1 x2 x3 u1 u2 Ec p1 p2 p3 q1 q2")
    Hf = sp.Function("H")
    pbH = lambda r, H: sum(target[r, s] * sp.diff(H, z[s]) for s in range(12))
    H1 = Hf(*z[4:])                                   # independent of (ct, x)
    ok1 = all(sp.simplify(pbH(r, H1)) == 0 for r in range(6, 10))
    H2 = H1 + sp.Function("V")(z[0])
    ok2 = sp.simplify(pbH(6, H2)) != 0 and all(sp.simplify(pbH(r, H2)) == 0 for r in range(7, 10))
    H3 = Hf(z[5], *z[6:])                             # depends on u2 but not on u1
    ok3 = sp.simplify(pbH(10, H3)) == 0 and sp.simplify(pbH(11, H3)) != 0
    report("5", ok and ok1 and ok2 and ok3,
           "two Cayley frames: the 12x12 bracket table of (ct_n,x^a,u^a,E/c,p_a,q_a) has {ct_n,E/c}=1, {x^a,p_b}=-delta, "
           "{u^a,q_b}=delta and nothing else; K_A dY^A = E dt_n - p.dx + q.du; every H independent of (ct_n,x^a) conserves "
           "E/c and p; adding V(ct_n) breaks only E; q_a is conserved iff H does not depend on u^a (q_1 is conserved by an "
           "H that depends on u^2 only)")


def chk_shells():
    Ec, p1, p2, p3, q1, q2, cc, M6 = sp.symbols("Ec p1 p2 p3 q1 q2 c M6", real=True)
    ok = True
    for (n, u1, u2), R in FRAMES[:2]:
        Ks = Ec * n + p1 * S_(0) + p2 * S_(1) + p3 * S_(2) + q1 * u1 + q2 * u2
        ok &= sp.expand(eta(Ks, Ks) - (Ec ** 2 - p1 ** 2 - p2 ** 2 - p3 ** 2 + q1 ** 2 + q2 ** 2)) == 0
    Esq = sp.solve(sp.Eq(Ec ** 2 - p1 ** 2 - p2 ** 2 - p3 ** 2 + q1 ** 2 + q2 ** 2, M6 ** 2 * cc ** 2), Ec)
    ok &= all(sp.expand((s * cc) ** 2 - (cc ** 2 * (p1 ** 2 + p2 ** 2 + p3 ** 2) + M6 ** 2 * cc ** 4 - cc ** 2 * (q1 ** 2 + q2 ** 2))) == 0 for s in Esq)
    report("6", ok,
           "eta(K,K) = (E/c)^2-|p|^2+|q|^2 in two Cayley frames; on eta(K,K)=M6^2c^2, E^2 = c^2|p|^2+M6^2c^4-c^2|q|^2, so "
           "m_eff^2 = M6^2-|q|^2/c^2 (FMB eqs. 7, 8)")
    # momentum reading of RE
    E1, E2, E3 = sp.symbols("E1 E2 E3", real=True)
    ok = True
    for (n, u1, u2), R in FRAMES:
        Ks = (E1 * unitv(0) + E2 * unitv(1) + E3 * unitv(2)) / cc + p1 * S_(0) + p2 * S_(1) + p3 * S_(2)
        Ev = sp.Matrix([E1, E2, E3])
        nn = sp.Matrix(n[:3])
        ok &= sp.simplify(cc * eta(n, Ks) - nn.dot(Ev)) == 0
        ok &= sp.simplify(cc * eta(u1, Ks) - sp.Matrix(u1[:3]).dot(Ev)) == 0
        qsq = eta(u1, Ks) ** 2 + eta(u2, Ks) ** 2
        ok &= sp.simplify(cc ** 2 * qsq - (Ev.dot(Ev) - nn.dot(Ev) ** 2)) == 0
        ok &= sp.simplify(Ev.dot(Ev) - nn.dot(Ev) ** 2 - nn.cross(Ev).dot(nn.cross(Ev))) == 0
    Ev = sp.Matrix([3, 4, 0])
    mu2 = Ev.dot(Ev) - 25
    W = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    ok &= mu2 == 0 and Ev[0] ** 2 - 25 == -16 and sp.simplify(W.dot(Ev) ** 2 - 25 + sp.Rational(26, 3)) == 0
    report("7", ok,
           "RE's six-momentum K = sum(E_i/c)e_i + sum p_a s_a, three Cayley clocks: E(n) = n.E, c q_a = u_a.E, "
           "c^2|q|^2 = |E|^2-(n.E)^2 = |n x E|^2; E=(3,4,0), p=(5,0,0), c=1: mu=0, m_4^2=-16 at e1, -26/3 at W")


def chk_option2():
    xs = sp.symbols("x0:4")
    Pm = sp.symbols("P0:4")
    u, v, pu, pv, M6, cc = sp.symbols("u v p_u p_v M6 c", real=True)
    h = sp.diag(1, -1, -1, -1)
    A = [sp.Function(f"A{m}")(*xs) for m in range(4)]
    ell = v * pu - u * pv
    k = [Pm[m] - A[m] * ell for m in range(4)]
    H = sp.Rational(1, 2) * (sum(h[m, m] * k[m] ** 2 for m in range(4)) + pu ** 2 + pv ** 2)
    q_ = list(xs) + [u, v]
    p_ = list(Pm) + [pu, pv]

    def pb(f, g):
        return sp.expand(sum(sp.diff(f, q_[i]) * sp.diff(g, p_[i]) - sp.diff(f, p_[i]) * sp.diff(g, q_[i]) for i in range(6)))
    C = sum(A[m] * h[m, m] * k[m] for m in range(4))
    ok = pb(pu, pv) == 0 and sp.expand(pb(pu, H) + C * pv) == 0 and sp.expand(pb(pv, H) - C * pu) == 0
    ok &= sp.expand(pb(ell, H)) == 0
    # A = 0: Option II with the ambient shell term; exact invariance of the Lagrangian under du = eps(tau), dlambda = eps'
    tau = sp.Symbol("tau")
    N = sp.Function("N")(tau)
    eps = [sp.Function(f"eps{a}")(tau) for a in range(2)]
    U = [sp.Function(f"u{a}")(tau) for a in range(2)]
    Q = [sp.Function(f"q{a}")(tau) for a in range(2)]
    lam = [sp.Function(f"lam{a}")(tau) for a in range(2)]
    X = [sp.Function(f"x{m}")(tau) for m in range(4)]
    Pt = [sp.Function(f"P{m}")(tau) for m in range(4)]
    PP = sum(h[m, m] * Pt[m] ** 2 for m in range(4))

    def Lag(Uv, lv):
        return (sum(Pt[m] * sp.diff(X[m], tau) for m in range(4)) + sum(Q[a] * sp.diff(Uv[a], tau) for a in range(2))
                - N / 2 * (PP - M6 ** 2 * cc ** 2 + Q[0] ** 2 + Q[1] ** 2) - sum(lv[a] * Q[a] for a in range(2)))
    dL = Lag([U[a] + eps[a] for a in range(2)], [lam[a] + sp.diff(eps[a], tau) for a in range(2)]) - Lag(U, lam)
    ok &= sp.simplify(dL) == 0
    # Option II removes every q != 0, a strict superset of the unstable band q^2 > |k|^2 + m^2 of ES17 Prop 27.1
    kk, mm, qq = 1, 1, sp.Rational(1, 2)
    ok &= kk ** 2 + mm ** 2 - qq ** 2 > 0 and qq != 0
    report("8", ok,
           "ES17 Prop 23.4/Thm 30.1 re-derived: with k = P - A(x) l, l = v p_u - u p_v, {p_u,p_v}=0, {p_u,H}=-C p_v, "
           "{p_v,H}=C p_u, {l,H}=0, C = A.h.k; at A=0 the action of FMB eq. 9 is exactly invariant under du=eps(tau), "
           "dlambda=eps'; q~0 removes every nonzero surplus momentum, including stable ones (k=m=1, q=1/2 has omega^2=7/4>0): "
           "the unstable band q^2>|k|^2+m^2 is a proper subset of the removed set")


def chk_option3():
    ok = True
    Ec, p1, p2, p3, q1, q2 = sp.symbols("Ec p1 p2 p3 q1 q2", real=True)
    th = sp.Symbol("theta", real=True)
    for (n, u1, u2), R in FRAMES[:2]:
        P, Q = projectors(n)
        Rn = P - Q
        B = wedge(u1, u2)
        expB = I6 + sp.sin(th) * B + (1 - sp.cos(th)) * B * B
        ok &= is_zero(B ** 3 + B) and is_zero(expB.subs(th, sp.pi) - Rn)
        ok &= Rn.T * ETA * Rn == ETA and Rn.det() == 1
        ok &= inertia(ETA * Rn) == (1, 5, 0)
        Ks = Ec * n + p1 * S_(0) + p2 * S_(1) + p3 * S_(2) + q1 * u1 + q2 * u2
        ok &= sp.expand((Ks.T * ETA * Rn * Ks)[0, 0] - (Ec ** 2 - p1 ** 2 - p2 ** 2 - p3 ** 2 - q1 ** 2 - q2 ** 2)) == 0
    # toy: H = (N/2)(P^2 - c^2 m0^2 - |q|^2) gives du/dtau = -N q; ambient H = (N/2)(P^2 - M^2c^2 + |q|^2) gives +N q
    N, cc, m0 = sp.symbols("N c m0", positive=True)
    Htoy = N / 2 * (Ec ** 2 - p1 ** 2 - cc ** 2 * m0 ** 2 - q1 ** 2 - q2 ** 2)
    Hamb = N / 2 * (Ec ** 2 - p1 ** 2 - cc ** 2 * m0 ** 2 + q1 ** 2 + q2 ** 2)
    ok &= sp.diff(Htoy, q1) == -N * q1 and sp.diff(Hamb, q1) == N * q1
    # arbitrary M^2(u,q): conserved
    u1s, u2s = sp.symbols("u1 u2")
    M2 = sp.Function("M2")(u1s, u2s, q1, q2)
    Hgen = N / 2 * (Ec ** 2 - p1 ** 2 - cc ** 2 * M2)
    pbuq = lambda f, g: sum(sp.diff(f, a) * sp.diff(g, b) - sp.diff(f, b) * sp.diff(g, a) for a, b in ((u1s, q1), (u2s, q2)))
    ok &= sp.simplify(pbuq(M2, Hgen)) == 0
    report("9", ok,
           "two Cayley frames: R_n = P_n - Q_n = exp(pi u1^u2) is in SO(3,3), eta(R_n.,.) has signature (1,5) and equals "
           "P^2-|q|^2 on K, so the clock supplies a quadratic form of the sign of Option III's toy; du/dtau = -N q for the "
           "toy against +N q on the ambient shell; any M^2(u,q) is conserved")


def chk_rotate():
    ok = True
    E1, E2, E3, cc = sp.symbols("E1 E2 E3 c", real=True)
    p1, p2, p3 = sp.symbols("p1 p2 p3", real=True)
    Ev = sp.Matrix([E1, E2, E3])
    for (n, u1, u2), R in FRAMES:
        nn = sp.Matrix(n[:3])
        En = nn.dot(Ev)
        m4sq = En ** 2 - (p1 ** 2 + p2 ** 2 + p3 ** 2) * cc ** 2
        mu2 = Ev.dot(Ev) - (p1 ** 2 + p2 ** 2 + p3 ** 2) * cc ** 2
        ok &= sp.simplify(m4sq - (mu2 - nn.cross(Ev).dot(nn.cross(Ev)))) == 0
    # exact examples
    Ev = sp.Matrix([2, 3, 6])
    mu2 = Ev.dot(Ev) - 9
    W = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    ok &= mu2 == 40 and (Ev / 7).dot(Ev) ** 2 - 9 == 40 and Ev[0] ** 2 - 9 == -5
    ok &= sp.simplify(W.dot(Ev) ** 2 - 9 - sp.Rational(94, 3)) == 0 and sp.simplify(Ev.dot(Ev) - W.dot(Ev) ** 2 - sp.Rational(26, 3)) == 0
    ok &= [x ** 2 for x in Ev / 7] == [sp.Rational(4, 49), sp.Rational(9, 49), sp.Rational(36, 49)]
    # equality m_4^2 = mu^2 also at the anti-aligned clock -E/|E| (which reads -|E|)
    ok &= (-Ev / 7).dot(Ev) == -7 and ((-Ev / 7).dot(Ev)) ** 2 - 9 == 40
    report("10a", ok,
           "three Cayley clocks (symbolic E, p): m_4(n)^2c^4 = mu^2c^4-|n x E|^2, so m_4(n)^2 <= mu^2 with equality iff n is "
           "parallel or antiparallel to E; E=(2,3,6), p=(0,0,3), c=1: mu^2=40, the clock (2,3,6)/7 (weights (4,9,36)/49) and "
           "the clock -(2,3,6)/7 both read m_4^2=40 (energies +7, -7), e1 reads -5, W reads 94/3 = 40-26/3")
    # octant maximum, positive part nonzero: random float test with a different method (projected-gradient ascent)
    rng = np.random.default_rng(20261025)
    ok = True
    worst = 0.0
    for _ in range(300):
        Evf = rng.normal(size=3)
        if np.all(Evf <= 0):
            Evf[rng.integers(3)] = abs(Evf[0]) + 0.1
        Ep = np.maximum(Evf, 0.0)
        best = np.linalg.norm(Ep)
        nstar = Ep / best
        ok &= abs(nstar @ Evf - best) < 1e-12
        # dense sampling of the octant through the simplex, w_i = n_i^2
        w = rng.dirichlet([1, 1, 1], size=3000)
        vals = np.sqrt(w) @ Evf
        ok &= vals.max() <= best + 1e-12
        worst = max(worst, vals.max() - best)
    # positive part zero: the maximum is max_i E_i <= 0, at a vertex, and |E^+| = 0 is NOT attained when all E_i < 0
    Ev = sp.Matrix([-1, -2, -3])
    vertices = [Ev[i] for i in range(3)]
    ok &= max(vertices) == -1
    w = rng.dirichlet([1, 1, 1], size=20000)
    vals = np.sqrt(w) @ np.array([-1.0, -2.0, -3.0])
    ok &= vals.max() <= -1 + 1e-12
    # exact argument: for n_i >= 0, |n| = 1, sum n_i >= 1, so n.E <= max_i E_i * sum n_i <= max_i E_i when max_i E_i <= 0
    for _ in range(50):
        Evf = -np.abs(rng.normal(size=3))
        w = rng.dirichlet([1, 1, 1], size=2000)
        ok &= (np.sqrt(w) @ Evf).max() <= Evf.max() + 1e-12
    # largest future-reading mass with a negative component: mu^2 - |E - E^+|^2, exact example
    Ev = sp.Matrix([2, -1, 2])
    Ep = sp.Matrix([2, 0, 2])
    pp2 = 1
    mu2 = Ev.dot(Ev) - pp2
    nstar = Ep / sp.sqrt(Ep.dot(Ep))
    ok &= sp.simplify(nstar.dot(Ev) ** 2 - pp2 - (mu2 - (Ev - Ep).dot(Ev - Ep))) == 0 and mu2 - (Ev - Ep).dot(Ev - Ep) == 7
    report("10b", ok,
           "Fisher-Rao octant (float, 300 random E with E^+ != 0 against 3000 Dirichlet-sampled clocks, tol 1e-12): "
           "max n.E = |E^+| at n = E^+/|E^+|; when E^+ = 0 the claim fails: E=(-1,-2,-3) has octant maximum -1 = max_i E_i "
           "(a vertex), not |E^+|=0 (20000 samples and 50 random all-negative E, tol 1e-12); E=(2,-1,2), p=(1,0,0): the "
           "largest future-reading mass is m_4^2 = 7 = mu^2-|E-E^+|^2 < mu^2 = 8")


# ----------------------------------------------------------------------------- field module
def F_of(Bv, Ev):
    F = sp.zeros(4, 4)
    for i in range(3):
        F[0, i + 1] = Ev[i]
        F[i + 1, 0] = -Ev[i]
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        F[j + 1, k + 1] = Bv[i]
        F[k + 1, j + 1] = -Bv[i]
    return F


H4 = sp.diag(1, -1, -1, -1)


def X_to_F(X):
    return F_of(X[:3, 0], X[3:, 0])


def rho2(A, F):
    return -(A.T * F + F * A)


def adapted_matrix(Bop, n):
    """4x4 matrix of an operator on E_n in the basis (n, s1, s2, s3), components by eta-orthonormality"""
    basis = [n] + [S_(a) for a in range(3)]
    M = sp.zeros(4, 4)
    for c, f in enumerate(basis):
        w = Bop * f
        for r, g in enumerate(basis):
            M[r, c] = eta(g, w) / eta(g, g)
    return M


def phi_images(n):
    out = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        out.append(adapted_matrix(wedge(S_(j), S_(k)), n))
    for i in range(3):
        out.append(adapted_matrix(wedge(S_(i), n), n))
    return out


def levi4():
    eps = sp.MutableDenseNDimArray.zeros(4, 4, 4, 4)
    import itertools
    from sympy.combinatorics import Permutation
    for perm in itertools.permutations(range(4)):
        eps[perm] = Permutation(list(perm)).signature()
    return eps


EPS4 = levi4()


def star(F):
    Fu = H4 * F * H4
    S = sp.zeros(4, 4)
    for m in range(4):
        for nu in range(4):
            S[m, nu] = sp.Rational(1, 2) * sum(EPS4[m, nu, r, s] * Fu[r, s] for r in range(4) for s in range(4))
    return S


def pairing(F1, F2):
    return sp.Rational(1, 2) * sum((F1[a, b] * (H4 * F2 * H4)[a, b]) for a in range(4) for b in range(4))


def chk_dictionary():
    Xs = sp.Matrix(sp.symbols("B1 B2 B3 E1 E2 E3", real=True))
    F = X_to_F(Xs)
    ok = True
    for (n, u1, u2), R in FRAMES[:2]:
        ph = phi_images(n)
        for g, A in zip(LG, ph):
            ok &= is_zero(X_to_F(g * Xs) - rho2(A, F))
        # uniqueness: the Lie-algebra element of so(1,3) intertwining each generator is unique
        cs = sp.symbols("y0:6")
        gens4 = [adapted_matrix(b, n) for b in lint_gens(n)]
        for g in LG:
            A = sum((cs[t] * gens4[t] for t in range(6)), sp.zeros(4, 4))
            eqs = list(sp.Matrix(list(X_to_F(g * Xs) - rho2(A, F))).jacobian(list(Xs)))
            ok &= len(sp.linsolve(eqs, cs).args[0].free_symbols) == 0
    report("12a", ok,
           "the dictionary Phi(X)=F_X intertwines ad of lG on V with -(A^T F + F A) on two-forms through the unique map "
           "L_i -> s_j^s_k, K_i -> s_i^n (component matrices in the adapted basis of two Cayley clocks)")
    ok = is_zero(X_to_F(G6 * Xs) - star(F))
    S2 = star(star(F))
    ok &= is_zero(S2 + F)
    gens4 = [adapted_matrix(b, unitv(0)) for b in lint_gens(unitv(0))]
    ok &= all(is_zero(star(rho2(A, F)) - rho2(A, star(F))) for A in gens4)
    ok &= sp.expand(pairing(F, F) - eta(Xs, Xs)) == 0
    Ys = sp.Matrix(sp.symbols("b1 b2 b3 f1 f2 f3", real=True))
    Fy = X_to_F(Ys)
    ok &= sp.expand(pairing(star(F), Fy) - pairing(F, star(Fy))) == 0
    ok &= sp.expand(pairing(star(F), star(Fy)) + pairing(F, Fy)) == 0
    basisF = [X_to_F(unitv(k)) for k in range(6)]
    gram = sp.Matrix(6, 6, lambda r, s: pairing(basisF[r], basisF[s]))
    ok &= inertia(gram) == (3, 3, 0)
    report("12b", ok,
           "F_{GX} = star F_X with eps_0123=+1; star^2=-1; star commutes with lint; <F,F> = F_ab F^ab/2 = |B|^2-|E|^2 = "
           "eta(X,X); star is self-adjoint and anti-isometric; signature (3,3) (exact, symbolic X)")


def hom_dim(repA, repB, dimA, dimB):
    """dim of {T: rhoB(x) T = T rhoA(x)} for paired lists of generator matrices"""
    rows = []
    for A, B in zip(repA, repB):
        rows.append(sp.kronecker_product(sp.eye(dimA), B) - sp.kronecker_product(A.T, sp.eye(dimB)))
    return null_dim(rows, dimA * dimB)


def sym2_rep(M, d):
    idx = [(i, j) for i in range(d) for j in range(i, d)]
    out = sp.zeros(len(idx), len(idx))
    for c, (i, j) in enumerate(idx):
        Bm = sp.zeros(d, d)
        Bm[i, j] = 1
        Bm[j, i] = 1
        img = M * Bm + Bm * M.T
        for r, (a, b) in enumerate(idx):
            out[r, c] = img[a, b]
    return out, idx


def chk_reps():
    ok = True
    for (n, u1, u2), R in [FRAMES[0], ((unitv(0), unitv(1), unitv(2)), None)]:
        gens4 = [adapted_matrix(b, n) for b in lint_gens(n)]
        # Lambda^2 E^*: basis of antisymmetric 4x4 matrices
        pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]

        def l2rep(A):
            M = sp.zeros(6, 6)
            for c, (a, b) in enumerate(pairs):
                Bm = sp.zeros(4, 4)
                Bm[a, b], Bm[b, a] = 1, -1
                img = rho2(A, Bm)
                for r, (x, y) in enumerate(pairs):
                    M[r, c] = img[x, y]
            return M
        L2 = [l2rep(A) for A in gens4]
        ok &= hom_dim(L2, gens4, 6, 4) == 0
        ok &= hom_dim(L2, L2, 6, 6) == 2
        V6 = [adapted_full for adapted_full in [b for b in lint_gens(n)]]
        # lint(n) on V: the 6x6 matrices; Hom(V, E_n): use E_n as subspace via P_n restricted
        ok &= hom_dim(V6, gens4, 6, 4) == 1
    report("11a", ok,
           "with lint(n) at a Cayley clock and at e1: dim Hom(Lambda^2 E^*, E) = 0; controls dim End(Lambda^2 E^*) = 2 "
           "(identity, star) and dim Hom(V, E) = 1")
    ph = phi_images(FRAMES[0][0][0])
    S2V = [sym2_rep(g, 6)[0] for g in LG]
    ok = hom_dim(LG, ph, 6, 4) == 0
    ok &= hom_dim(S2V, ph, 21, 4) == 0
    ok &= hom_dim(S2V, [sp.zeros(1, 1)] * 6, 21, 1) == 2
    ok &= hom_dim(S2V, [sp.zeros(2, 2)] * 6, 21, 2) == 4
    S2E = [sym2_rep(A, 4)[0] for A in ph]
    ok &= hom_dim(S2V, S2E, 21, 10) == 3
    # traceless part: symmetric S^{mu nu} with eta_{mu nu} S^{mu nu} = 0, basis of 9
    _, idx4 = sym2_rep(ph[0], 4)
    trace_row = sp.Matrix([[H4[i, j] if i == j else 0 for (i, j) in idx4]])
    Bt = trace_row.nullspace()
    Bt = sp.Matrix.hstack(*Bt)
    pinv = (Bt.T * Bt).inv() * Bt.T
    S2E0 = [pinv * M * Bt for M in S2E]
    ok &= all(is_zero(M * Bt - Bt * (pinv * M * Bt)) for M in S2E)      # invariant subspace
    ok &= hom_dim(S2V, S2E0, 21, 9) == 1
    report("11b", ok,
           "with lG acting on E_n through phi: dim Hom(V_EM,E)=0, dim Hom(Sym^2 V_EM,E)=0, dim Hom(Sym^2 V_EM,R)=2, "
           "dim Hom(Sym^2 V_EM,Sym^2 E)=3 with traceless part 1; but dim Hom(Sym^2 V_EM,U_n)=4 (U_n is a trivial module): "
           "covariance alone does not forbid quadratic maps from the field to the surplus plane")


def theta(F):
    Fu = H4 * F * H4
    inv = sum(F[a, b] * Fu[a, b] for a in range(4) for b in range(4))
    return H4 * F * Fu + sp.Rational(1, 4) * H4 * inv


def chk_stress():
    Bv = sp.Matrix(sp.symbols("B1 B2 B3", real=True))
    Ev = sp.Matrix(sp.symbols("E1 E2 E3", real=True))
    F = F_of(Bv, Ev)
    Th = sp.expand(theta(F))
    E6 = (Bv.dot(Bv) + Ev.dot(Ev)) / 2
    S = Ev.cross(Bv)
    E0sq = ((Bv.dot(Bv) - Ev.dot(Ev)) ** 2 + 4 * Ev.dot(Bv) ** 2) / 4
    ok = sp.expand(Th[0, 0] - E6) == 0 and all(sp.expand(Th[0, a + 1] + S[a]) == 0 for a in range(3))
    Bt = -Bv
    St = Ev.cross(Bt)
    ok &= all(sp.expand(Th[0, a + 1] - St[a]) == 0 for a in range(3))
    ok &= Th == Th.T and sp.expand((H4 * Th).trace()) == 0
    mixed = Th * H4
    ok &= is_zero(sp.expand(mixed * mixed - E0sq * I4))
    ok &= sp.expand(E6 ** 2 - E0sq - S.dot(S)) == 0
    report("13", ok,
           "symbolic (B,E), paper's B: Theta^00 = E6, Theta^0a = -(E x B)_a = (E x B_t)_a, so the momentum density is "
           "minus the frame Poynting vector; Theta symmetric, traceless, Theta^mu_l Theta^l_nu = E0^2 delta, E6^2 = E0^2+|S|^2")
    chi = sp.Symbol("chi", real=True)
    M = LG[5]
    ok = M ** 3 == M
    expM = I6 + sp.sinh(chi) * M + (sp.cosh(chi) - 1) * M * M
    X1 = expM * unitv(0)
    ok &= is_zero(X1 - sp.Matrix([sp.cosh(chi), 0, 0, 0, sp.sinh(chi), 0]))
    F1 = X_to_F(X1)
    Th1 = sp.simplify(theta(F1))
    ok &= is_zero(Th1[0, 0] - sp.cosh(2 * chi) / 2) and is_zero(Th1[0, 3] - sp.sinh(2 * chi) / 2)
    report("14a", ok,
           "ad_{K3}^3 = ad_{K3}; the rest field e1 boosted by exp(chi ad K3) is B=(cosh chi,0,0), E=(0,sinh chi,0), with "
           "E6 = cosh(2chi)/2 and S = E x B = -(sinh(2chi)/2)e3 (Theta^03 = +sinh(2chi)/2), not (cosh chi, sinh chi)/2")
    Xs = sp.Matrix(sp.symbols("x1:7", real=True))
    L = sp.eye(4)
    L[0, 0] = L[3, 3] = sp.cosh(chi)
    L[0, 3] = L[3, 0] = sp.sinh(chi)
    lhs = theta(X_to_F(expM * Xs))
    rhs = L * theta(X_to_F(Xs)) * L.T
    ok = is_zero(sp.expand(sp.simplify(lhs - rhs).rewrite(sp.exp)))
    # floating-point cross-check with scipy's matrix exponential
    rng = np.random.default_rng(7)
    Mf = np.array(M.tolist(), dtype=float)
    for _ in range(20):
        x = rng.normal(size=6)
        ch = rng.normal()
        xb = expm(ch * Mf) @ x
        Tf = np.array(theta(X_to_F(sp.Matrix(xb))).tolist(), dtype=float)
        Lf = np.array(L.subs(chi, ch).tolist(), dtype=float)
        T0 = np.array(theta(X_to_F(sp.Matrix(x))).tolist(), dtype=float)
        ok &= np.allclose(Tf, Lf @ T0 @ Lf.T, atol=1e-10)
    report("14b", ok,
           "for every X: Theta[exp(chi ad K3)X] = L Theta[X] L^T with L^0_3 = L^3_0 = +sinh chi (symbolic, and 20 random "
           "float tests with scipy expm, tol 1e-10): the passage to the frame moving with -tanh(chi) e3")


def chk_packets():
    f = lambda s: (1 - s * s) ** 2 if abs(s) <= 1 else 0.0
    s = sp.Symbol("s")
    I = sp.integrate((1 - s ** 2) ** 4, (s, -1, 1))
    ok = I == sp.Rational(256, 315)
    tol = 1e-10
    c = 256 / 315

    def UP(a, b, t):
        Ex = lambda z: a * f(z - t) + b * f(z + t)
        By = lambda z: a * f(z - t) - b * f(z + t)
        pts = sorted({t - 1, t + 1, -t - 1, -t + 1})
        lo, hi = min(pts), max(pts)
        U = integrate.quad(lambda z: 0.5 * (Ex(z) ** 2 + By(z) ** 2), lo, hi, points=pts, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
        P = integrate.quad(lambda z: Ex(z) * By(z), lo, hi, points=pts, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
        return U, P
    for (a, b) in [(1.0, 1.0), (1.3, 0.4), (0.7, -1.1), (-2.0, 0.5)]:
        for t in [0.0, 0.3, 0.9, 1.0, 2.5]:
            U, P = UP(a, b, t)
            ok &= abs(U - c * (a * a + b * b)) < tol and abs(P - c * (a * a - b * b)) < tol
        U, P = UP(a, b, 0.0)
        M = np.sqrt(U * U - P * P)
        ok &= abs(M - 2 * c * abs(a * b)) < tol
        ok &= (a * b >= 0) or abs(M - 2 * c * a * b) > 0.1          # the formula needs |ab|
    # boost: frame moving with +beta along z, explicit Lorentz transformation of fields and coordinates, at t' = 0
    for (a, b) in [(1.0, 1.0), (1.3, 0.4)]:
        for chi in [0.4, -0.9]:
            beta, gam = np.tanh(chi), np.cosh(chi)
            U, P = UP(a, b, 0.0)

            def fields_primed(zp):
                t, z = gam * beta * zp, gam * zp
                Ex = a * f(z - t) + b * f(z + t)
                By = a * f(z - t) - b * f(z + t)
                return gam * (Ex - beta * By), gam * (By - beta * Ex)
            ends = sorted({x / (gam * (1 - beta)) for x in (-1, 1)} | {x / (gam * (1 + beta)) for x in (-1, 1)})
            lo, hi = min(ends), max(ends)
            Up = integrate.quad(lambda zp: 0.5 * sum(v * v for v in fields_primed(zp)), lo, hi, points=ends, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
            Pp = integrate.quad(lambda zp: np.prod(fields_primed(zp)), lo, hi, points=ends, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
            ok &= abs(Up - (U * np.cosh(chi) - P * np.sinh(chi))) < 1e-9 and abs(Pp - (P * np.cosh(chi) - U * np.sinh(chi))) < 1e-9
    # Maxwell equations for the two pulses (textbook, HL, c=1) and disjointness
    zz, tt = sp.symbols("z t", real=True)
    g = sp.Function("g")
    Ex = g(zz - tt) + g(zz + tt)
    By = g(zz - tt) - g(zz + tt)
    ok &= sp.simplify(sp.diff(By, tt) + sp.diff(Ex, zz)) == 0 and sp.simplify(sp.diff(Ex, tt) + sp.diff(By, zz)) == 0
    report("15", ok,
           "packets E_x = a f(z-t)+b f(z+t), B_ty = a f(z-t)-b f(z+t) (HL, c=1; quadrature, tol 1e-10): U = (256/315)(a^2+b^2), "
           "P_z = (256/315)(a^2-b^2) at t = 0,...,2.5, so M = (512/315)|ab| (the formula (512/315)ab is negative for ab<0); "
           "transforming fields and coordinates to the frame moving with tanh(chi) e_z gives (U cosh-P sinh, P cosh-U sinh) "
           "(tol 1e-9); the pulses solve Maxwell's equations")


def chk_two_waves():
    t, x, y, z = sp.symbols("t x y z", real=True)
    ph1, ph2 = 3 * z - 3 * t, 4 * x - 4 * t
    E1 = sp.Matrix([sp.cos(ph1), 0, 0])
    B1 = sp.Matrix([0, sp.cos(ph1), 0])
    E2 = sp.Matrix([0, sp.cos(ph2), 0])
    B2 = sp.Matrix([0, 0, sp.cos(ph2)])
    X = [x, y, z]
    curl = lambda V: sp.Matrix([sp.diff(V[2], X[1]) - sp.diff(V[1], X[2]), sp.diff(V[0], X[2]) - sp.diff(V[2], X[0]),
                                sp.diff(V[1], X[0]) - sp.diff(V[0], X[1])])
    div = lambda V: sum(sp.diff(V[i], X[i]) for i in range(3))
    ok = True
    for (Ev, Bv) in [(E1, B1), (E2, B2)]:
        ok &= is_zero(sp.diff(Bv, t) + curl(Ev)) and is_zero(sp.diff(Ev, t) - curl(Bv)) and div(Ev) == 0 and div(Bv) == 0

    def T(Ev, Bv):
        u = (Ev.dot(Ev) + Bv.dot(Bv)) / 2
        Sv = Ev.cross(Bv)
        Tm = sp.zeros(4, 4)
        Tm[0, 0] = u
        for i in range(3):
            Tm[0, i + 1] = Tm[i + 1, 0] = Sv[i]
            for j in range(3):
                Tm[i + 1, j + 1] = -(Ev[i] * Ev[j] + Bv[i] * Bv[j]) + (u if i == j else 0)
        return Tm
    Ti = T(E1 + E2, B1 + B2) - T(E1, B1) - T(E2, B2)
    co = [t, x, y, z]
    ok &= all(sp.simplify(sum(sp.diff(Ti[m, nu], co[m]) for m in range(4))) == 0 for nu in range(4))
    ok &= not is_zero(Ti)
    # null sums: parallel, orthogonal and antiparallel momenta
    def m2(P1, P2):
        P = [a + b for a, b in zip(P1, P2)]
        return P[0] ** 2 - P[1] ** 2 - P[2] ** 2 - P[3] ** 2
    ok &= m2((3, 0, 0, 3), (4, 4, 0, 0)) == 24 == 2 * 3 * 4 * (1 - 0)
    ok &= m2((3, 0, 0, 3), (4, 0, 0, 4)) == 0
    ok &= m2((3, 0, 0, 3), (4, 0, 0, -4)) == 48 == 2 * 3 * 4 * (1 - (-1))
    report("16", ok,
           "two Maxwell plane waves along (0,0,3) and (4,0,0): the interference part of Theta is nonzero and divergence free; "
           "null sums: (3;0,0,3)+(4;4,0,0) has M^2=24, parallel momenta give 0, and ANTIPARALLEL (collinear) momenta "
           "(3;0,0,3)+(4;0,0,-4) give M^2=48=2U1U2(1-cos pi) > 0: the sum is timelike unless theta = 0")


def chk_photon():
    zz, tt, k = sp.symbols("z t k", real=True)
    ph = k * (zz - tt)
    Bv = sp.Matrix([sp.cos(ph), -sp.sin(ph), 0])        # Re (1,i,0)e^{i ph}
    Ev = sp.Matrix([sp.sin(ph), sp.cos(ph), 0])         # Im (1,i,0)e^{i ph}
    ok = is_zero(Bv.dot(Bv) - Ev.dot(Ev)) and is_zero(Ev.dot(Bv))
    E6 = sp.simplify((Bv.dot(Bv) + Ev.dot(Ev)) / 2)
    S = sp.simplify(Ev.cross(Bv))
    ok &= E6 == 1 and is_zero(S - sp.Matrix([0, 0, -1]))
    # paper's Riemann-Silberstein equation d_t z = -i curl z for z = B + iE
    zc = Bv + sp.I * Ev
    curlz = sp.Matrix([-sp.diff(zc[1], zz), sp.diff(zc[0], zz), 0])
    ok &= is_zero(sp.diff(zc, tt) + sp.I * curlz)
    A = sp.Matrix(3, 3, sp.symbols("a0:9"))
    ok &= sp.expand((A * A.T - A.T * A).trace()) == 0
    report("17", ok,
           "ES17 Ex. 12.11 wave z=(1,i,0)e^{ik(x3-t)} solves d_t z = -i curl z, is null, E6 = |S| = 1, S = -e3, momentum "
           "density -S = +e3 along propagation; tr[A,A^T] = 0 for every finite matrix, so no finite CCR")


def chk_nogo():
    v = sp.Matrix(sp.symbols("v1:7"))
    w = sp.Matrix(sp.symbols("w1:7"))
    ok = sp.expand(eta(G6 * v, G6 * w) + eta(v, w)) == 0
    a_ = eta(v, v)
    b_ = eta(v, G6 * v)
    ok &= sp.expand(eta(G6 * v, G6 * v) + a_) == 0 and sp.expand(eta(G6 * v, v) - b_) == 0
    random.seed(11)
    cnt = 0
    for _ in range(200):
        v1 = sp.Matrix([random.randint(-5, 5) for _ in range(6)])
        w1 = sp.Matrix([random.randint(-5, 5) for _ in range(6)])
        basis = [v1, G6 * v1, w1, G6 * w1]
        Bm = sp.Matrix.hstack(*basis)
        if rank_q(Bm) < 4:
            continue
        gram = Bm.T * ETA * Bm
        if gram.det() == 0:
            continue
        cnt += 1
        ok &= inertia(gram) == (2, 2, 0)
    ok &= cnt > 150
    # other G: a conjugate b G b^-1 with b in O(3,3) is again eta-self-adjoint with square -1
    b = projectors(FRAMES[0][0][0])[0] - projectors(FRAMES[0][0][0])[1]
    G2 = b * G6 * b.inv()
    ok &= G2 * G2 == -I6 and G2.T * ETA == ETA * G2
    report("18", ok,
           f"eta(Gv,Gw)=-eta(v,w) (symbolic); {cnt} random integer G-invariant nondegenerate four-planes have inertia (2,2,0) "
           "(exact, Descartes' rule): no (1,3) plane is G-invariant")


def chk_leak():
    lam, kap, m = sp.symbols("lambda kappa m", positive=True)
    X = sp.Matrix([sp.Function(f"X{A}")(lam) for A in range(6)])
    Xd = X.diff(lam)
    Lg = (m / 2 * (Xd + kap * G6 * X).T * ETA * (Xd + kap * G6 * X))[0, 0]
    EL = sp.Matrix([sp.diff(sp.diff(Lg, Xd[A]), lam) - sp.diff(Lg, X[A]) for A in range(6)])
    p = m * ETA * (Xd + kap * G6 * X)
    ok = is_zero(sp.expand(EL - (p.diff(lam) - kap * G6.T * p)))
    ok &= ETA.inv() * G6.T * ETA == G6
    pi0 = unitv(4)
    pil = (sp.cos(kap * lam) * I6 + sp.sin(kap * lam) * G6) * pi0
    ok &= is_zero(pil.diff(lam) - kap * G6 * pil)
    ok &= is_zero(pil - (sp.cos(kap * lam) * unitv(4) - sp.sin(kap * lam) * unitv(1)))
    P1, Q1 = projectors(unitv(0))
    ok &= is_zero(P1 * pil - sp.cos(kap * lam) * unitv(4))
    v0 = sp.Matrix(sp.symbols("y1:7"))
    vl = (sp.cos(kap * lam) * I6 + sp.sin(kap * lam) * G6) * v0
    ok &= is_zero(sp.expand(sp.expand_trig(eta(vl, vl) - (sp.cos(2 * kap * lam) * eta(v0, v0) + sp.sin(2 * kap * lam) * eta(v0, G6 * v0)))))
    report("19a", ok,
           "sympy Euler-Lagrange of L_int = (m/2)eta(X'+kappa GX, X'+kappa GX) is p' = kappa G^T p, p = m eta(X'+kappa GX); "
           "pi' = kappa G pi with pi(lambda) = (cos + sin G)pi(0); pi(0)=e5 -> cos e5 - sin e2, P_e1 pi = cos e5; eta(pi,pi) "
           "-> cos(2 kappa lambda) eta + sin(2 kappa lambda) eta(pi, G pi)")
    ok = True
    for (n, u1, u2), R in FRAMES:
        P, Q = projectors(n)
        QGP = Q * G6 * P
        ok &= Q * G6 * n == sp.zeros(6, 1) and rank_q(QGP) == 2
        nn = sp.Matrix(n[:3])
        sigma = sp.zeros(6, 6)
        for a in range(3):
            sigma[a, a + 3] = 1
        target = -(PIP - n * n.T) * sigma
        ok &= QGP == target
        ok &= QGP * (G6 * n) == sp.zeros(6, 1) and G6 * n == svec(nn)
        # v in E_n stays for all theta iff Q G v = 0; kernel on E_n is span(n, Gn), signature (1,1)
        basisE = sp.Matrix.hstack(n, S_(0), S_(1), S_(2))
        kerE = (QGP * basisE).nullspace()
        kerv = [basisE * k for k in kerE]
        ok &= len(kerv) == 2 and rank_q(sp.Matrix.hstack(*kerv, n, G6 * n)) == 2
        ok &= inertia(sp.Matrix([[eta(x, y) for y in (n, G6 * n)] for x in (n, G6 * n)])) == (1, 1, 0)
    report("19b", ok,
           "three Cayley clocks: Q_n G n = 0, Q_n G P_n = -(pi_+ - n n^T) sigma pi_- with sigma: s_a -> e_a, rank 2, kernel on "
           "E_n = span(n, Gn) of signature (1,1); at n = e1 this is <e1, e4>")


def chk_twoalg():
    ok = True
    LGs = LG
    fixG = null_dim([sp.Matrix.vstack(*LGs)], 6)
    ok &= fixG == 0
    ok &= all(G6 * g == g * G6 for g in LGs)
    for (n, u1, u2), R in FRAMES[:2]:
        P, Q = projectors(n)
        li = lint_gens(n)
        ok &= null_dim([sp.Matrix.vstack(*li)], 6) == 2 and all(g * u == sp.zeros(6, 1) for g in li for u in (u1, u2))
        both = sp.Matrix.hstack(*[g.reshape(36, 1) for g in li + LGs])
        ok &= rank_q(both) == 12
        cs = sp.symbols("c0:6")
        Xg = sum((cs[t] * LGs[t] for t in range(6)), sp.zeros(6, 6))
        eqs = list(Q * Xg * P) + list(P * Xg * Q)
        (sol,) = sp.linsolve(eqs, cs)
        free = set().union(*[sp.sympify(s).free_symbols for s in sol])
        ok &= len(free) == 1
        Ln = sum((n[i] * LGs[i] for i in range(3)), sp.zeros(6, 6))
        ok &= is_zero(Q * Ln * P) and is_zero(P * Ln * Q)
        ok &= Ln * n == sp.zeros(6, 1) and Ln * (G6 * n) == sp.zeros(6, 1)
        ok &= any(G6 * g != g * G6 for g in li)
    report("20", ok,
           "two Cayley clocks: lint(n) fixes U_n, lG fixes no nonzero vector, lG + lint(n) has rank 12 (intersection 0); "
           "solving Q x P = P x Q = 0 over lG gives exactly the line of sum n_i L_i, whose fixed plane is span(n, Gn); G "
           "commutes with lG and not with lint(n)")


def main():
    import time
    t0 = time.time()
    for chk in (chk_conventions, chk_planes, chk_covariance, chk_J, chk_dirac, chk_poisson, chk_shells, chk_option2,
                chk_option3, chk_rotate, chk_dictionary, chk_reps, chk_stress, chk_packets, chk_two_waves, chk_photon,
                chk_nogo, chk_leak, chk_twoalg):
        chk()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed: {', '.join(FAILS)}")
        return 1
    print(f"all s2 checks passed ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
