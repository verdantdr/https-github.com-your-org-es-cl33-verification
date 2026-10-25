#!/usr/bin/env python3
"""Track r2 -- the four-momentum with X on the spatial triple.

Verification script for investigations/two_readings/sections/r2_fourmomentum.tex.
Conventions (tex/s02_arena.tex, tex/sA_realisation.tex, tex/s12_emfield.tex of the 17 October 2026 paper):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1), V+ = <e1,e2,e3> (times), V- = <e4,e5,e6> (lengths);
  e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1, e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep;
  binding root G e_i = e_{i+3}, G e_{i+3} = -e_i; para-complex K e_i = e_{i+3}, K e_{i+3} = e_i;
  a clock is a unit n in V+, E_n = Rn + V-, U_n = n^perp in V+, (n,u1,u2) an oriented orthonormal frame of V+;
  field reading X = sum_i (B_i e_i + E_i e_{i+3}), z = B + iE, paper B = minus the textbook field (B_t = -B),
  E6 = |z|^2/2, E0 = |q|/2, q = z.z, frame Poynting vector S = E x B = -S_t, S_t = E x B_t.
A bivector a^b acts on V by M(a,b) v = a eta(b,v) - b eta(a,v) = (1/2)[ab, v] (Clifford commutator).
Claims r2-0 ... r2-20 (ids as in the section; sub-ids a,b,c,...).  Exact sympy arithmetic for algebraic and
symbolic statements; floating point only where stated (random tests, tolerance given in the line).
Run from any directory:  python3 r2_fourmomentum.py   (exit status 0 iff every check passes)
"""
import sys
import itertools
import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

FAILED = []


def report(cid, ok, text):
    print(("PASS" if ok else "FAIL") + f" {cid}: {text}")
    if not ok:
        FAILED.append(cid)


def zero(M):
    if not isinstance(M, sp.MatrixBase):
        return sp.simplify(M) == 0
    return all(sp.simplify(x) == 0 for x in M)


def qrank(M):
    """Exact rank of a rational sympy matrix."""
    return DomainMatrix.from_Matrix(sp.Matrix(M)).convert_to(sp.QQ).rank()


def nullity(M):
    M = sp.Matrix(M)
    return M.shape[1] - qrank(M)


# ----------------------------------------------------------------------------- realisation
sx = sp.Matrix([[0, 1], [1, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
ep = sp.Matrix([[0, -1], [1, 0]])
I2 = sp.eye(2)


def k3(a, b, c):
    return sp.kronecker_product(sp.kronecker_product(a, b), c)


e = [k3(sx, I2, I2), k3(sz, sx, I2), k3(sz, sz, sx),
     k3(ep, I2, I2), k3(sz, ep, I2), k3(sz, sz, ep)]
eta_d = [1, 1, 1, -1, -1, -1]
ETA = sp.diag(*eta_d)
I8 = sp.eye(8)
I6 = sp.eye(6)
Gm = sp.zeros(6, 6)          # binding root on V
Km = sp.zeros(6, 6)          # para-complex K on V
for i in range(3):
    Gm[i + 3, i] = 1
    Gm[i, i + 3] = -1
    Km[i + 3, i] = 1
    Km[i, i + 3] = 1
PIm = sp.diag(0, 0, 0, 1, 1, 1)   # projector onto V-
PIp = sp.diag(1, 1, 1, 0, 0, 0)   # projector onto V+
EPS = sp.LeviCivita


def cl(v):
    """Clifford image sum_a v_a e_a of a 6-vector."""
    return sum((v[a] * e[a] for a in range(6)), sp.zeros(8, 8))


def unit(a):
    v = sp.zeros(6, 1)
    v[a] = 1
    return v


def tvec(n3):
    """Embed a 3-vector into V+."""
    return sp.Matrix([n3[0], n3[1], n3[2], 0, 0, 0])


def svec(p3):
    """Embed a 3-vector into V-."""
    return sp.Matrix([0, 0, 0, p3[0], p3[1], p3[2]])


def eta(v, w):
    return (v.T * ETA * w)[0, 0]


def Mb(a, b):
    """so(3,3) matrix of the bivector a^b: v -> a eta(b,v) - b eta(a,v)."""
    return a * (b.T * ETA) - b * (a.T * ETA)


def comm(A, B):
    return A * B - B * A


def exp_boost(B, chi):
    """exp(chi B) for a generator with B^3 = B (checked)."""
    assert B**3 == B
    return sp.eye(B.shape[0]) + sp.sinh(chi) * B + (sp.cosh(chi) - 1) * B**2


def exp_rot(M, th):
    """exp(th M) for a generator with M^3 = -M (checked)."""
    assert M**3 == -M
    return sp.eye(M.shape[0]) + sp.sin(th) * M + (1 - sp.cos(th)) * M**2


def acomm(A, B):
    return A * B + B * A


def frame_from_quaternion(w, x, y, z):
    """Oriented orthonormal frame (n,u1,u2) of R^3 as the columns of the rotation matrix of a quaternion."""
    s = w**2 + x**2 + y**2 + z**2
    R = sp.Matrix([[w**2 + x**2 - y**2 - z**2, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), w**2 - x**2 + y**2 - z**2, 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), w**2 - x**2 - y**2 + z**2]]) / s
    return [R[:, 0], R[:, 1], R[:, 2]]


# exact clocks with rational (or quadratic-irrational) oriented orthonormal frames (n, u1, u2), n x u1 = u2
R3 = sp.Rational
FRAMES = {
    "e1": ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
    "e2": ([0, 1, 0], [0, 0, 1], [1, 0, 0]),
    "e3": ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
    "W": ([1 / sp.sqrt(3)] * 3, [1 / sp.sqrt(2), -1 / sp.sqrt(2), 0], [1 / sp.sqrt(6), 1 / sp.sqrt(6), -2 / sp.sqrt(6)]),
    "(1,2,2)/3": ([R3(1, 3), R3(2, 3), R3(2, 3)], [R3(2, 3), R3(1, 3), R3(-2, 3)], [R3(-2, 3), R3(2, 3), R3(-1, 3)]),
    "(2,3,6)/7": ([R3(2, 7), R3(3, 7), R3(6, 7)], [R3(3, 7), R3(-6, 7), R3(2, 7)], [R3(6, 7), R3(2, 7), R3(-3, 7)]),
    "(-1,4,8)/9": ([R3(-1, 9), R3(4, 9), R3(8, 9)], [R3(4, 9), R3(-7, 9), R3(4, 9)], [R3(8, 9), R3(4, 9), R3(-1, 9)]),
}
RATIONAL = ["e1", "e2", "e3", "(1,2,2)/3", "(2,3,6)/7", "(-1,4,8)/9"]


def frame6(name):
    n3, u13, u23 = FRAMES[name]
    return tvec(n3), tvec(u13), tvec(u23)


def projectors(n):
    """P_n onto E_n = Rn + V-, Q_n onto U_n = n^perp in V+."""
    Pn = n * (n.T * ETA) + PIm
    Qn = PIp - n * (n.T * ETA)
    return Pn, Qn


def lint_gens(n):
    s = [unit(3), unit(4), unit(5)]
    gens = [Mb(n, s[a]) for a in range(3)] + [Mb(s[1], s[2]), Mb(s[2], s[0]), Mb(s[0], s[1])]
    return gens


def lint_cl(n):
    """Interior Lorentz algebra of E_n as Clifford bivectors."""
    N = cl(n)
    return [N * e[3], N * e[4], N * e[5], e[4] * e[5], e[5] * e[3], e[3] * e[4]]


# ============================================================================= r2-0 conventions
def check_conventions():
    ok = all(acomm(e[a], e[b]) == 2 * eta_d[a] * (1 if a == b else 0) * I8 for a in range(6) for b in range(6))
    om = [e[i] * e[i + 3] for i in range(3)]
    ok_om = (om[0] == k3(sz, I2, I2) and om[1] == k3(I2, sz, I2) and om[2] == k3(I2, I2, sz))
    ok_G = (Gm * Gm == -I6 and ETA.inv() * Gm.T * ETA == Gm and Gm.T * ETA * Gm == -ETA)
    ok_K = (Km * Km == I6 and ETA.inv() * Km.T * ETA == -Km)
    # G e_i = e_{i+3} is Clifford conjugation-free: check G = -cK o Ad_T on vectors (ES17 Lemma 3.4)
    T = e[0] * e[1] * e[2]
    ok_T = all(T * e[a] * T.inv() == eta_d[a] * e[a] for a in range(6))
    report("r2-0a", ok and ok_om and ok_G and ok_K and ok_T,
           "Clifford relations of the realisation; omega_i = e_i e_{i+3} are sz(x)1(x)1, 1(x)sz(x)1, 1(x)1(x)sz; "
           "G^2=-1, G eta-self-adjoint, eta(GX,GY)=-eta(X,Y); K^2=1, K eta-skew; Ad_{e1e2e3}|_V = eta")
    # Fock modes
    ap = [(e[i] + e[i + 3]) / 2 for i in range(3)]
    am = [(e[i] - e[i + 3]) / 2 for i in range(3)]
    car = all(acomm(ap[i], am[j]) == (I8 if i == j else sp.zeros(8, 8)) and acomm(ap[i], ap[j]) == sp.zeros(8, 8)
              and acomm(am[i], am[j]) == sp.zeros(8, 8) for i in range(3) for j in range(3))
    transp = all(ap[i].T == am[i] for i in range(3))
    Nop = sum((ap[i] * am[i] for i in range(3)), sp.zeros(8, 8))
    diagN = all(Nop[r, s] == 0 for r in range(8) for s in range(8) if r != s)
    countN = all(Nop[r, r] == sum(1 for i in range(3) if om[i][r, r] == -1) for r in range(8))
    vac = sp.zeros(8, 1)
    vac[0] = 1
    vac_ok = all(om[i] * vac == vac for i in range(3)) and all(am[i] * vac == sp.zeros(8, 1) for i in range(3))
    report("r2-0b", car and transp and diagN and countN and vac_ok,
           "a+-_i=(e_i+-e_{i+3})/2 obey the CAR, (a+_i)^T=a-_i, N=sum a+_i a-_i is diagonal and counts the minus signs "
           "of the omega-eigenvalues; the (+,+,+) line is the Fock vacuum")
    # lG generators: L_i, K_i act on (B,E) as in RE / ES17 Thm 12.3
    okL = True
    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        Lm = Mb(unit(j + 3), unit(k + 3)) - Mb(unit(j), unit(k))
        Kmat = Mb(unit(k), unit(j + 3)) - Mb(unit(j), unit(k + 3))
        Bs = sp.symbols("B1:4")
        Es = sp.symbols("E1:4")
        X = sp.Matrix(list(Bs) + list(Es))
        ei = sp.Matrix([1 if t == i else 0 for t in range(3)])
        Bv, Ev = sp.Matrix(Bs), sp.Matrix(Es)
        dL = Lm * X
        dK = Kmat * X
        okL &= zero(dL[:3, 0] - ei.cross(Bv)) and zero(dL[3:, 0] - ei.cross(Ev))
        okL &= zero(dK[3:, 0] - ei.cross(Bv)) and zero(dK[:3, 0] + ei.cross(Ev))
        # Clifford form of the generators: [L_i, v] with L_i = (1/2)(e_{j+3}e_{k+3} - e_j e_k)
        Lc = (e[j + 3] * e[k + 3] - e[j] * e[k]) / 2
        okL &= all(comm(Lc, cl(unit(a))) == cl(Lm * unit(a)) for a in range(6))
        okL &= all(Gm * Lm == Lm * Gm for _ in [0]) and Gm * Kmat == Kmat * Gm
    report("r2-0c", okL,
           "L_i=(1/2)(e_{j+3}e_{k+3}-e_je_k) and K_i=(1/2)(e_ke_{j+3}-e_je_{k+3}) act on X by dB=e_i x B, dE=e_i x E "
           "and dE=e_i x B, dB=-e_i x E, and commute with G (generators of lG)")


# ============================================================================= r2-1 clock, E_n, U_n
def check_decomposition():
    w, x, y, z = sp.symbols("w x y z", real=True)
    n3, u13, u23 = frame_from_quaternion(w, x, y, z)
    n, u1, u2 = tvec(n3), tvec(u13), tvec(u23)
    Pn, Qn = projectors(n)
    ok = zero(sp.simplify(eta(n, n) - 1)) and zero(Pn * Pn - Pn) and zero(Qn * Qn - Qn) and zero(Pn + Qn - I6)
    ok &= zero(ETA * Pn - (ETA * Pn).T) and zero(Pn.T * ETA * Qn)          # eta-self-adjoint, eta-orthogonal
    ok &= zero(Pn * n - n) and all(zero(Pn * unit(a) - unit(a)) for a in (3, 4, 5)) and zero(Qn * u1 - u1) and zero(Qn * u2 - u2)
    gram_E = sp.Matrix([[eta(a, b) for b in [n, unit(3), unit(4), unit(5)]] for a in [n, unit(3), unit(4), unit(5)]])
    gram_U = sp.Matrix([[eta(a, b) for b in [u1, u2]] for a in [u1, u2]])
    ok &= zero(gram_E - sp.diag(1, -1, -1, -1)) and zero(gram_U - sp.eye(2))
    ok &= zero(sp.simplify(sp.Matrix.hstack(n3, u13, u23).det() - 1))
    report("r2-1a", ok,
           "for a general oriented frame (n,u1,u2) of V+ (quaternion parametrisation, symbolic): P_n, Q_n are complementary "
           "eta-orthogonal eta-self-adjoint projectors onto E_n = Rn+V- (Gram diag(1,-1,-1,-1)) and U_n = <u1,u2> (Gram 1)")
    # every Lorentzian four-plane containing V- is an E_n: E = V- + (E cap V+), the latter a positive line
    # check: the four-planes V- + R v, v in V generic, are of signature (1,3) iff v has a nonzero V+ part, and equal E_{n}
    v = sp.Matrix(sp.symbols("v1:7", real=True))
    vp = PIp * v
    nrm = sp.sqrt(vp.dot(vp))
    nn = vp / nrm
    span_eq = qrank(sp.Matrix.hstack(unit(3), unit(4), unit(5), v.subs({v[0]: 1, v[1]: 2, v[2]: 2, v[3]: 7, v[4]: -3, v[5]: 5}),
                                    tvec([R3(1, 3), R3(2, 3), R3(2, 3)]))) == 4
    report("r2-1b", span_eq and zero(sp.simplify(eta(nn, nn) - 1)),
           "a four-plane V- + Rv contains the clock n = (V+ part of v)/|.|: the Lorentzian four-planes containing the three "
           "lengths are exactly the E_n, n in S^2 (up to sign); exact example v=(1,2,2;7,-3,5) gives E_{(1,2,2)/3}")


# ============================================================================= r2-2 covariance (FMB Prop 3.2)
def check_covariance():
    ok_all = True
    dims = {}
    for name in RATIONAL + ["W"]:
        n, u1, u2 = frame6(name)
        Pn, Qn = projectors(n)
        gens = lint_gens(n) + [Mb(u1, u2)]
        ok_all &= all(zero(g * Pn - Pn * g) for g in gens)
        # q_a are Lorentz scalars: Q_n g = 0 for g in lint(n) (g maps into E_n and kills U_n)
        ok_all &= all(zero(Qn * g) and zero(g * Qn) for g in lint_gens(n))
        if name in RATIONAL:
            # full commutant of P_n in so(3,3): solve [X,P_n]=0 for X = sum c_ab M(e_a,e_b)
            basis = [Mb(unit(a), unit(b)) for a in range(6) for b in range(a + 1, 6)]
            rows = []
            for r in range(6):
                for s in range(6):
                    rows.append([comm(B, Pn)[r, s] for B in basis])
            dims[name] = 15 - qrank(sp.Matrix(rows))
    ok_all &= all(d == 7 for d in dims.values())
    report("r2-2a", ok_all,
           "for n in {e1,e2,e3,W,(1,2,2)/3,(2,3,6)/7,(-1,4,8)/9}: P_n commutes with lint(n)+so(U_n); lint(n) kills U_n "
           "(q_a are Lorentz scalars); the commutant of P_n in so(3,3) has dimension 7 = 6+1 (rational n): so(1,3)+so(2)")
    # cotangent lift of the SO(2) of U_n is canonical and rotates (u^a, q_a) together; finite group check
    th = sp.symbols("theta", real=True)
    n, u1, u2 = frame6("(2,3,6)/7")
    Pn, Qn = projectors(n)
    Rth = Pn + sp.cos(th) * Qn + sp.sin(th) * Mb(u1, u2)
    ok = zero(sp.simplify(Rth.T * ETA * Rth - ETA)) and zero(sp.simplify(Rth * Pn - Pn * Rth))
    ok &= zero(sp.simplify(Rth * u1 - (sp.cos(th) * u1 - sp.sin(th) * u2)))
    ok &= zero(sp.simplify(exp_rot(Mb(u1, u2), th) - Rth))
    # the lifted map (Y,K_low) -> (R Y, R^{-T} K_low) preserves K_A dY^A, and R^{-T} = eta R eta^{-1}: K^sharp moves like Y
    ok &= zero(sp.simplify(Rth.T * (ETA * Rth * ETA.inv()) - I6))
    # a finite boost of E_n extended by the identity: P_n(L K) = L P_n K, q(L K) = q(K)
    chi = sp.symbols("chi", real=True)
    Lb = exp_boost(Mb(n, unit(3)), chi)
    Kv = sp.Matrix(sp.symbols("k1:7", real=True))
    ok &= zero(sp.simplify(Pn * Lb * Kv - Lb * Pn * Kv)) and zero(sp.simplify(Qn * Lb * Kv - Qn * Kv))
    ok &= zero(sp.simplify(Lb.T * ETA * Lb - ETA))
    report("r2-2b", ok,
           "n=(2,3,6)/7: the SO(2) of U_n is an eta-isometry commuting with P_n and rotating (u1,u2); its cotangent lift "
           "preserves K_A dY^A; a finite boost exp(chi n^e4) of E_n, identity on U_n, satisfies P_n L = L P_n, Q_n L = Q_n")
    # limitation: the eight mixed generators of E_n ^ U_n do not commute with P_n, e.g. a rotation in (n,u1)
    ok = True
    for name in ["e1", "(1,2,2)/3"]:
        n, u1, u2 = frame6(name)
        Pn, Qn = projectors(n)
        mixed = [Mb(a, b) for a in [n, unit(3), unit(4), unit(5)] for b in [u1, u2]]
        ok &= all(not zero(comm(g, Pn)) for g in mixed)
        ok &= qrank(sp.Matrix.hstack(*[comm(g, Pn).reshape(36, 1) for g in mixed])) == 8
    phi = sp.symbols("phi", real=True)
    n, u1, u2 = frame6("e1")
    Pn, Qn = projectors(n)
    Rr = exp_rot(Mb(n, u1), phi)
    Kv = sp.Matrix([1, 0, 0, 0, 0, 0])   # K = (E/c) n with E/c = 1, at rest
    ok &= zero(sp.simplify(Qn * Rr * Kv - sp.Matrix([0, -sp.sin(phi), 0, 0, 0, 0]))) or \
        zero(sp.simplify(Qn * Rr * Kv - sp.Matrix([0, sp.sin(phi), 0, 0, 0, 0])))
    report("r2-2c", ok,
           "no SO(3,3) covariance: each of the 8 generators of E_n^U_n fails to commute with P_n and the 8 commutators are "
           "independent (n=e1,(1,2,2)/3); a rotation by phi in (e1,e2) moves the rest momentum K=(1;0) to q_1 = +-sin(phi)")


# ============================================================================= r2-3, r2-4  J(n) and the Dirac structure
def check_J():
    IT = e[0] * e[1] * e[2]
    ok = IT * IT == -I8 and all(comm(IT, e[i]) == sp.zeros(8, 8) for i in range(3)) \
        and all(acomm(IT, e[a]) == sp.zeros(8, 8) for a in range(3, 6))
    nn = sp.symbols("n1:4", real=True)
    N = nn[0] * e[0] + nn[1] * e[1] + nn[2] * e[2]
    Jn = IT * N
    ok &= Jn == nn[0] * e[1] * e[2] + nn[1] * e[2] * e[0] + nn[2] * e[0] * e[1]
    ok &= zero(Jn * Jn + (nn[0]**2 + nn[1]**2 + nn[2]**2) * I8)
    ok &= zero(comm(Jn, N)) and all(zero(comm(Jn, e[a])) for a in range(3, 6))
    ok &= (IT * e[0] == e[1] * e[2])
    ok &= zero(IT * (-N) + Jn)
    report("r2-3a", ok,
           "I_T=e1e2e3: I_T^2=-1, commutes with e1,e2,e3, anticommutes with e4,e5,e6; J(n)=I_T n = n1 e2e3+n2 e3e1+n3 e1e2 "
           "(symbolic n), J(n)^2=-|n|^2, [J(n),n]=[J(n),e_{a+3}]=0, J(e1)=e2e3, J(-n)=-J(n)")
    # J(n) = u1 u2 for every oriented orthonormal frame (n,u1,u2): n u1 u2 = I_T
    w, x, y, z = sp.symbols("w x y z", real=True)
    n3, u13, u23 = frame_from_quaternion(w, x, y, z)
    Nn, U1, U2 = cl(tvec(n3)), cl(tvec(u13)), cl(tvec(u23))
    ok2 = zero(sp.simplify(Nn * U1 * U2 - IT)) and zero(sp.simplify(U1 * U2 - IT * Nn))
    report("r2-3b", ok2,
           "for a general oriented orthonormal frame (n,u1,u2) of V+ (symbolic): n u1 u2 = I_T, hence J(n) = u1 u2 is the "
           "oriented unit bivector of the surplus plane U_n")
    # J(n) commutes with every bivector of E_n, for symbolic n
    s = [e[3], e[4], e[5]]
    Jn_s = IT * cl(tvec(nn))
    ok3 = all(zero(comm(Jn_s, B)) for B in [cl(tvec(nn)) * s[a] for a in range(3)] + [s[0] * s[1], s[1] * s[2], s[2] * s[0]])
    report("r2-3c", ok3, "J(n) commutes with the whole interior Lorentz algebra of E_n (symbolic n)")
    # time reversal n -> -n: same four-plane and projector, opposite energy, opposite J and Omega, same Gamma
    nv = tvec(nn)
    Pp, Qp = projectors(nv)
    Pm_, Qm_ = projectors(-nv)
    Kv = sp.Matrix(sp.symbols("k1:7", real=True))
    Om_p = cl(nv) * e[3] * e[4] * e[5]
    Om_m = cl(-nv) * e[3] * e[4] * e[5]
    J_p, J_m = IT * cl(nv), IT * cl(-nv)
    ok4 = zero(Pp - Pm_) and zero(Qp - Qm_) and zero(eta(-nv, Kv) + eta(nv, Kv))
    ok4 &= zero(J_m + J_p) and zero(Om_m + Om_p) and zero(J_m * Om_m - J_p * Om_p)
    report("r2-3d", ok4,
           "time reversal n -> -n (symbolic n): P_{-n} = P_n, Q_{-n} = Q_n (same E_n, U_n), the energy read by the clock "
           "E(-n) = c eta(-n,K) = -E(n), J(-n) = -J(n), "
           "Omega_{-n} = -Omega_n, so Gamma_n = J(n) Omega_n is unchanged")


def commutant_dim(mats):
    """Dimension of the joint commutant of a list of 8x8 matrices in M_8(R), and a basis."""
    X = sp.Matrix(8, 8, sp.symbols("x0:64"))
    eqs = []
    for A in mats:
        C = A * X - X * A
        eqs += list(C)
    xs = list(X)
    Mrows = sp.Matrix([[sp.diff(eq, v) for v in xs] for eq in eqs])
    ns = Mrows.nullspace()
    return len(ns), [sp.Matrix(8, 8, list(v)) for v in ns]


def in_span(target, spanning):
    A = sp.Matrix.hstack(*[S.reshape(64, 1) for S in spanning])
    return qrank(A) == qrank(sp.Matrix.hstack(A, target.reshape(64, 1)))


def check_dirac_clock():
    IT = e[0] * e[1] * e[2]
    ok_all = True
    ok_sq = True
    ok_q = True
    for name in ["e1", "e2", "(1,2,2)/3", "(2,3,6)/7", "(-1,4,8)/9"]:
        n, u1, u2 = frame6(name)
        N = cl(n)
        Jn = IT * N
        Om = N * e[3] * e[4] * e[5]
        gens = lint_cl(n) + [Jn]
        d, basis = commutant_dim(gens)
        expected = [I8, Om, Jn, Om * Jn]
        ok = d == 4 and all(in_span(Y, basis) for Y in expected)
        ok &= qrank(sp.Matrix.hstack(*[Y.reshape(64, 1) for Y in expected])) == 4
        ok_all &= ok
        # square roots of -1 in span{1,Om,J,OmJ}
        a, b, c, dd = sp.symbols("a b c d", real=True)
        Xg = a * I8 + b * Om + c * Jn + dd * Om * Jn
        eqs = list(Xg * Xg + I8)
        sols = sp.solve(eqs, [a, b, c, dd], dict=True)
        solset = {tuple(sp.nsimplify(s[v]) for v in (a, b, c, dd)) for s in sols}
        ok_sq &= solset == {(0, 1, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, -1, 0)}
        gam = [N, e[3], e[4], e[5]]
        ok_q &= all(zero(comm(Jn, g)) for g in gam) and all(not zero(comm(Om, g)) for g in gam)
        # commutant of the four gamma matrices alone: quaternionic, contains u_a Omega_n; with I_T: span{1, J(n)}
        d4, basis4 = commutant_dim(gam)
        ok_q &= d4 == 4 and all(in_span(Y, basis4) for Y in [I8, Jn, cl(u1) * Om, cl(u2) * Om])
        d5, _ = commutant_dim(gam + [IT])
        ok_q &= d5 == 2
    report("r2-4a", ok_all,
           "ES17 Thm 4.1 for the clock n (n=e1,e2,(1,2,2)/3,(2,3,6)/7,(-1,4,8)/9): the joint commutant of lint(n) and of the "
           "surplus bivector J(n) in M_8(R) is span{1,Omega_n,J(n),Omega_n J(n)}, Omega_n = n e4 e5 e6, of dimension 4")
    report("r2-4b", ok_sq and ok_q,
           "its solutions of x^2=-1 are exactly +-Omega_n, +-J(n), and only +-J(n) commute with gamma(n)=(n,e4,e5,e6); the "
           "commutant of gamma(n) alone is 4-dimensional (quaternionic, containing u_a Omega_n), and adding I_T leaves span{1,J(n)}")
    # mass form and kinetic kernels for the clock n; Dirac Hamiltonian for clock n
    w, x, y, z = sp.symbols("w x y z", real=True)
    n3, u13, u23 = frame_from_quaternion(w, x, y, z)
    N = cl(tvec(n3))
    Jn = IT * N
    ok = zero(sp.simplify(N * Jn - IT))                       # gamma^0(n) J(n) = I_T for every clock
    gam = [N, e[3], e[4], e[5]]
    ok &= all(zero(sp.simplify(IT * Jn * g + N * g)) for g in gam)   # K^mu = T J gamma^mu = -gamma^0 gamma^mu
    ok &= zero(sp.simplify(IT.T + IT))
    p1, p2, p3, m, c = sp.symbols("p1 p2 p3 m c", positive=True)
    H = c * (N * e[3] * p1 + N * e[4] * p2 + N * e[5] * p3) + m * c**2 * N
    ok &= zero(sp.simplify(H * H - (c**2 * (p1**2 + p2**2 + p3**2) + m**2 * c**4) * I8))
    ok &= zero(sp.simplify(comm(H, Jn))) and zero(sp.simplify(H - H.T))
    report("r2-4c", ok,
           "for a general clock n (symbolic frame): the mass form gamma^0 J = n J(n) = I_T does not depend on n; the kinetic "
           "kernels T J(n) gamma^mu(n) = -n gamma^mu(n); H_n(p) = c n e_{a+3} p_a + m c^2 n is symmetric, commutes with J(n), "
           "H_n^2 = (c^2|p|^2 + m^2 c^4) 1")
    # rotor transport: R in Spin(V+) with R e1 R^-1 = n carries the n=e1 structure to the clock n
    th = sp.symbols("theta", real=True)
    Rr = sp.cos(th / 2) * I8 - sp.sin(th / 2) * e[0] * e[1]     # rotor in the (e1,e2) plane
    Rinv = sp.cos(th / 2) * I8 + sp.sin(th / 2) * e[0] * e[1]
    nth = cl(tvec([sp.cos(th), sp.sin(th), 0]))
    ok = zero(sp.simplify(Rr * Rinv - I8)) and zero(sp.simplify(Rr * e[0] * Rinv - nth))
    ok &= all(zero(sp.simplify(Rr * e[a] * Rinv - e[a])) for a in (3, 4, 5))
    ok &= zero(sp.simplify(Rr * (e[1] * e[2]) * Rinv - IT * nth)) and zero(sp.simplify(Rr * IT * Rinv - IT))
    report("r2-4d", ok,
           "the rotor R=cos(th/2)-sin(th/2)e1e2 of V+ fixes e4,e5,e6 and I_T, carries e1 to n(th)=(cos th, sin th, 0) and "
           "e2e3 to J(n(th)): the clock-n Dirac structure is the transport of the n=e1 one")


# ============================================================================= r2-5 Poisson brackets, four-vector, conservation
Ysym = sp.symbols("Y1:7", real=True)
Ksym = sp.symbols("K1:7", real=True)


def PB(f, g, Y=Ysym, K=Ksym):
    return sp.expand(sum(sp.diff(f, Y[A]) * sp.diff(g, K[A]) - sp.diff(f, K[A]) * sp.diff(g, Y[A]) for A in range(len(Y))))


def adapted(name=None, frame=None):
    if frame is None:
        n, u1, u2 = frame6(name)
    else:
        n, u1, u2 = frame
    Y = sp.Matrix(Ysym)
    Klow = sp.Matrix(Ksym)
    Ksharp = ETA * Klow
    x0 = eta(n, Y)                    # c t_n
    xs = [Y[3], Y[4], Y[5]]           # lengths
    us = [eta(u1, Y), eta(u2, Y)]     # surplus coordinates
    Ec = eta(n, Ksharp)               # E/c
    ps = [Ksharp[3], Ksharp[4], Ksharp[5]]     # p_a = K^{a+3}
    qs = [eta(u1, Ksharp), eta(u2, Ksharp)]
    return n, u1, u2, x0, xs, us, Ec, ps, qs, Ksharp


def check_poisson():
    ok = True
    for name in ["e1", "(2,3,6)/7", "(-1,4,8)/9"]:
        n, u1, u2, x0, xs, us, Ec, ps, qs, Ks = adapted(name)
        coords = [x0] + xs + us
        moms = [Ec] + [-p for p in ps] + qs           # covariant P_mu = (E/c, -p) and q_a
        M = sp.Matrix([[sp.simplify(PB(a, b)) for b in moms] for a in coords])
        ok &= M == sp.eye(6)
        ok &= all(sp.simplify(PB(a, b)) == 0 for a in coords for b in coords)
        ok &= all(sp.simplify(PB(a, b)) == 0 for a in moms for b in moms)
    report("r2-5a", ok,
           "{Y^A,K_B}=delta in the clock-adapted coordinates (c t_n = eta(n,Y), x^a, u^a = eta(u_a,Y)): {c t_n, E/c} = 1, "
           "{x^a, -p_b} = delta_ab, {u^a, q_b} = delta_ab, all others 0 (n = e1, (2,3,6)/7, (-1,4,8)/9)")
    # conservation for any Hamiltonian H(K, u) (general function), symbolic frame
    w, x, y, z = sp.symbols("w x y z", real=True)
    fr = [tvec(v) for v in frame_from_quaternion(w, x, y, z)]
    n, u1, u2, x0, xs, us, Ec, ps, qs, Ks = adapted(frame=fr)
    Hf = sp.Function("H")
    Hexpr = Hf(*Ksym, us[0], us[1])
    okc = all(sp.simplify(PB(P, Hexpr)) == 0 for P in [Ec] + ps)
    Vf = sp.Function("V")
    Hx = Hexpr + Vf(x0)
    okx = sp.simplify(PB(Ec, Hx)) != 0 and all(sp.simplify(PB(p, Hx)) == 0 for p in ps)
    okq = sp.simplify(PB(qs[0], Hexpr)) != 0
    H2only = Hf(*Ksym, us[1])                      # depends on u^2 but not on u^1
    okq &= sp.simplify(PB(qs[0], H2only)) == 0 and sp.simplify(PB(qs[1], H2only)) != 0
    report("r2-5b", okc and okx and okq,
           "for a general frame (symbolic) and an arbitrary function H(K_A,u^1,u^2): {E/c,H}={p_a,H}=0, so P_n=(E/c,p) is "
           "conserved by every dynamics independent of (c t_n, x^a); adding V(c t_n) breaks only {E,H}=0; q_a is conserved "
           "iff H does not depend on u^a (an H(K,u^2) conserves q_1 and not q_2)")
    # P_n is a four-vector: under L = exp(chi n^s_1) + id on U_n, components transform by the 4x4 boost
    n, u1, u2 = frame6("(1,2,2)/3")
    chi = sp.symbols("chi", real=True)
    L6 = exp_boost(Mb(unit(3), n), chi)
    basisE = [n, unit(3), unit(4), unit(5)]
    L4 = sp.Matrix([[sp.simplify(eta(basisE[r], L6 * basisE[s]) * [1, -1, -1, -1][r]) for s in range(4)] for r in range(4)])
    Kv = sp.Matrix(sp.symbols("k1:7", real=True))
    comps = lambda V: sp.Matrix([eta(n, V), -eta(unit(3), V), -eta(unit(4), V), -eta(unit(5), V)])
    ok4 = zero(sp.simplify(comps(L6 * Kv) - L4 * comps(Kv)))
    ok4 &= zero(sp.simplify(L4.T * sp.diag(1, -1, -1, -1) * L4 - sp.diag(1, -1, -1, -1)))
    ok4 &= zero(sp.simplify(L4 - sp.Matrix([[sp.cosh(chi), sp.sinh(chi), 0, 0], [sp.sinh(chi), sp.cosh(chi), 0, 0],
                                            [0, 0, 1, 0], [0, 0, 0, 1]])))
    report("r2-5c", ok4,
           "n=(1,2,2)/3: under exp(chi e4^n) (identity on U_n) the components (E/c,p) of P_n transform by the 4x4 boost "
           "of rapidity chi along p_1, a Lorentz transformation of signature (1,3)")


# ============================================================================= r2-6 .. r2-9 mass shells
def check_shells():
    w, x, y, z = sp.symbols("w x y z", real=True)
    n3, u13, u23 = frame_from_quaternion(w, x, y, z)
    n, u1, u2 = tvec(n3), tvec(u13), tvec(u23)
    Ec, p1, p2, p3, q1, q2, c, M6 = sp.symbols("Ec p1 p2 p3 q1 q2 c M6", real=True)
    Ksharp = Ec * n + svec([p1, p2, p3]) + q1 * u1 + q2 * u2
    ok = zero(sp.simplify(eta(Ksharp, Ksharp) - (Ec**2 - p1**2 - p2**2 - p3**2 + q1**2 + q2**2)))
    E = sp.symbols("E", real=True)
    shell = (E / c)**2 - (p1**2 + p2**2 + p3**2) + q1**2 + q2**2 - M6**2 * c**2
    ok &= sp.expand(shell * c**2 - (E**2 - (c**2 * (p1**2 + p2**2 + p3**2) + M6**2 * c**4 - c**2 * (q1**2 + q2**2)))) == 0
    meff2 = M6**2 - (q1**2 + q2**2) / c**2
    ok &= sp.expand(E**2 - c**2 * (p1**2 + p2**2 + p3**2) - meff2 * c**4 - shell * c**2) == 0
    # tachyonic iff |q| > M6 c (example)
    ok &= (meff2.subs({M6: 1, c: 1, q1: R3(3, 5), q2: R3(4, 5)}) == 0) and (meff2.subs({M6: 1, c: 1, q1: 1, q2: 1}) < 0)
    report("r2-6", ok,
           "general frame (symbolic): eta(K,K) = (E/c)^2 - |p|^2 + |q|^2; on the ambient shell eta(K,K)=M6^2c^2, "
           "E^2 = c^2|p|^2 + M6^2c^4 - c^2|q|^2, m_eff^2 = M6^2 - |q|^2/c^2, zero at |q| = M6 c and negative beyond")
    # relation to RE Prop 2.10: K = sum (E_i/c) e_i + sum p_a e_{a+3}
    E1, E2, E3 = sp.symbols("E1 E2 E3", real=True)
    Ev = sp.Matrix([E1, E2, E3])
    P = tvec(Ev / c) + svec([p1, p2, p3])
    En = c * eta(n, P)
    qn = [eta(u1, P), eta(u2, P)]
    ok = zero(sp.simplify(En - n3.dot(Ev)))
    ok &= zero(sp.simplify(c**2 * (qn[0]**2 + qn[1]**2) - (Ev.dot(Ev) - n3.dot(Ev)**2)))
    ok &= zero(sp.simplify(c**2 * (qn[0]**2 + qn[1]**2) - n3.cross(Ev).dot(n3.cross(Ev))))
    mu2c4 = c**2 * eta(P, P)
    m42c4 = En**2 - (p1**2 + p2**2 + p3**2) * c**2
    ok &= zero(sp.simplify(m42c4 - (mu2c4 - c**2 * (qn[0]**2 + qn[1]**2))))
    # at the symmetric clock: c^2|q|^2 = |E_perp|^2
    Wn = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    EW = Wn.dot(Ev)
    Eperp = Ev - EW * Wn
    ok &= zero(sp.simplify((Ev.dot(Ev) - EW**2) - Eperp.dot(Eperp)))
    # RE's example E=(3,4,0), p=(5,0,0), c=1
    sub = {E1: 3, E2: 4, E3: 0, p1: 5, p2: 0, p3: 0, c: 1}
    mu2 = (E1**2 + E2**2 + E3**2 - p1**2 - p2**2 - p3**2).subs(sub)
    m4e1 = (E1**2 - p1**2).subs(sub)
    m4W = sp.nsimplify((EW**2 - p1**2).subs(sub))
    ok &= mu2 == 0 and m4e1 == -16 and m4W == R3(-26, 3)
    report("r2-7", ok,
           "with K = RE's six-momentum sum(E_i/c)e_i + sum p_a e_{a+3} and any clock n: E = n.E, c^2|q|^2 = |E|^2-(n.E)^2 "
           "= |n x E|^2, so Option I with M6 = mu is RE Prop 2.10 (m_4^2c^4 = mu^2c^4 - c^2|q|^2); at W: c^2|q|^2 = |E_perp|^2; "
           "E=(3,4,0), p=(5,0,0), c=1: mu=0, m_4^2 = -16 at e1 and -26/3 at W")


def PB_c(f, g, X, P):
    return sp.expand(sum(sp.diff(f, X[i]) * sp.diff(g, P[i]) - sp.diff(f, P[i]) * sp.diff(g, X[i]) for i in range(len(X))))


def check_option2():
    # adapted canonical coordinates: x^mu, P_mu (covariant), u^a, q_a
    xs = sp.symbols("x0:4", real=True)
    Ps = sp.symbols("P0:4", real=True)
    us = sp.symbols("u1:3", real=True)
    qs = sp.symbols("q1:3", real=True)
    X = list(xs) + list(us)
    Pm = list(Ps) + list(qs)
    N, M6, c = sp.symbols("N M6 c", positive=True)
    P2 = Ps[0]**2 - Ps[1]**2 - Ps[2]**2 - Ps[3]**2
    H = N / 2 * (P2 - M6**2 * c**2 + qs[0]**2 + qs[1]**2)
    ok = all(PB_c(qa, qb, X, Pm) == 0 for qa in qs for qb in qs) and all(PB_c(qa, H, X, Pm) == 0 for qa in qs)
    # gauge flow of eps^a q_a shifts only u^a
    e1s, e2s = sp.symbols("eps1 eps2", real=True)
    gen = e1s * qs[0] + e2s * qs[1]
    flows = [PB_c(v, gen, X, Pm) for v in X + Pm]
    ok &= flows == [0, 0, 0, 0, e1s, e2s, 0, 0, 0, 0, 0, 0]
    # invariance of the extended action under du^a = eps^a(tau), dlambda^a = d eps^a/d tau
    tau = sp.symbols("tau", real=True)
    uf = [sp.Function(f"u{a}")(tau) for a in (1, 2)]
    qf = [sp.Function(f"q{a}")(tau) for a in (1, 2)]
    lf = [sp.Function(f"lam{a}")(tau) for a in (1, 2)]
    ef = [sp.Function(f"eps{a}")(tau) for a in (1, 2)]
    s = sp.symbols("s", real=True)
    Lsur = lambda uu, ll: sum(qf[a] * sp.diff(uu[a], tau) - ll[a] * qf[a] for a in range(2))
    dL = sp.diff(Lsur([uf[a] + s * ef[a] for a in range(2)], [lf[a] + s * sp.diff(ef[a], tau) for a in range(2)]), s)
    ok &= sp.simplify(dL) == 0
    # reduced shell and dimension count
    ok &= sp.simplify(H.subs({qs[0]: 0, qs[1]: 0}) - N / 2 * (P2 - M6**2 * c**2)) == 0
    # q ~ 0 removes every q != 0; the unstable band q^2 > |k|^2 + m^2 of ES17 Prop 27.1 is a proper subset:
    # k = m = 1, q = 1/2 is a stable mode (omega^2 = 7/4 > 0) that the constraints also remove
    kq, mq, qq = 1, 1, R3(1, 2)
    ok &= kq**2 + mq**2 - qq**2 == R3(7, 4) and qq != 0
    report("r2-8a", ok,
           "Option II (FMB eq. 9): {q_a,q_b}=0, {q_a,H}=0 for H=(N/2)(P^2-M6^2c^2+|q|^2), so q_a~0 are first class; "
           "eps^a q_a generates du^a = eps^a only; the action is exactly invariant under du=eps, dlambda=d eps/d tau; "
           "on q=0 the shell is P^2 = M6^2 c^2; 12 - 2x2 = 8 = dim T*E_n; q~0 removes every q != 0, a proper superset of the "
           "unstable band of ES17 Prop 27.1 (k=m=1, q=1/2 is stable, omega^2 = 7/4)")
    # ES17 Prop 23.4 / Thm 30.1 with a normal connection A_mu(x): structure functions
    xs4 = list(xs)
    u, v, pu, pv = sp.symbols("u v p_u p_v", real=True)
    A = [sp.Function(f"A{m}")(*xs4) for m in range(4)]
    ell = v * pu - u * pv
    k = [Ps[m] - A[m] * ell for m in range(4)]
    h = [1, -1, -1, -1]
    Hk = sp.Rational(1, 2) * (sum(h[m] * k[m]**2 for m in range(4)) + pu**2 + pv**2)
    C = sum(A[m] * h[m] * k[m] for m in range(4))
    Xk = xs4 + [u, v]
    Pk = list(Ps) + [pu, pv]
    ok = sp.simplify(PB_c(pu, Hk, Xk, Pk) + C * pv) == 0 and sp.simplify(PB_c(pv, Hk, Xk, Pk) - C * pu) == 0
    ok &= PB_c(pu, pv, Xk, Pk) == 0 and sp.simplify(PB_c(ell, Hk, Xk, Pk)) == 0
    report("r2-8b", ok,
           "ES17 Prop 23.4 re-verified: with k_mu = p_mu - A_mu(x) l, l = v p_u - u p_v, H = (h k k + p_u^2 + p_v^2)/2, "
           "{p_u,H} = -C p_v, {p_v,H} = C p_u, {p_u,p_v} = 0, {l,H} = 0, C = A.h.k; at A = 0 this is Option II at n = e1")


def check_option3():
    xs = sp.symbols("x0:4", real=True)
    Ps = sp.symbols("P0:4", real=True)
    us = sp.symbols("u1:3", real=True)
    qs = sp.symbols("q1:3", real=True)
    X = list(xs) + list(us)
    Pm = list(Ps) + list(qs)
    N, c, m0 = sp.symbols("N c m0", positive=True)
    P2 = Ps[0]**2 - Ps[1]**2 - Ps[2]**2 - Ps[3]**2
    M2 = sp.Function("M2")(us[0], us[1], qs[0], qs[1])
    H = N / 2 * (P2 - c**2 * M2)
    ok = all(PB_c(P, H, X, Pm) == 0 for P in Ps) and sp.simplify(PB_c(M2, H, X, Pm)) == 0
    E, p1, p2, p3 = sp.symbols("E p1 p2 p3", real=True)
    cons = (P2 - c**2 * M2).subs({Ps[0]: E / c, Ps[1]: -p1, Ps[2]: -p2, Ps[3]: -p3})
    ok &= sp.expand(cons * c**2 - (E**2 - c**2 * (p1**2 + p2**2 + p3**2) - c**4 * M2)) == 0
    report("r2-9a", ok,
           "Option III (FMB eq. 10): for an arbitrary M^2(u,q), {P_mu,H}=0 and {M^2,H}=0 along the flow of "
           "H=(N/2)(P^2-c^2M^2); the constraint is E^2 = c^2|p|^2 + c^4 M^2 (FMB Prop 4.1 confirmed)")
    # toy model: M^2 = m0^2 + |q|^2/c^2; its quadratic form is eta(R_n K, K), R_n = P_n - Q_n, signature (1,5)
    Mtoy = m0**2 + (qs[0]**2 + qs[1]**2) / c**2
    Htoy = N / 2 * (P2 - c**2 * Mtoy)
    Hamb = N / 2 * (P2 + qs[0]**2 + qs[1]**2 - m0**2 * c**2)
    udot_toy = [PB_c(ua, Htoy, X, Pm) for ua in us]
    udot_amb = [PB_c(ua, Hamb, X, Pm) for ua in us]
    ok = udot_toy == [-N * qs[0], -N * qs[1]] and udot_amb == [N * qs[0], N * qs[1]]
    okR = True
    for name in ["e1", "(2,3,6)/7"]:
        n, u1, u2 = frame6(name)
        Pn, Qn = projectors(n)
        Rn = Pn - Qn
        okR &= Rn.T * ETA * Rn == ETA and Rn.det() == 1 and zero(sp.simplify(exp_rot(Mb(u1, u2), sp.pi) - Rn))
        form = ETA * Rn
        okR &= form == form.T
        ev = form.eigenvals()
        pos = sum(mult for val, mult in ev.items() if val > 0)
        neg = sum(mult for val, mult in ev.items() if val < 0)
        okR &= (pos, neg) == (1, 5)
        Ec, a1, a2, a3, b1, b2 = sp.symbols("Ec a1 a2 a3 b1 b2", real=True)
        Ksh = Ec * n + svec([a1, a2, a3]) + b1 * u1 + b2 * u2
        okR &= sp.expand((Ksh.T * ETA * Rn * Ksh)[0, 0] - (Ec**2 - a1**2 - a2**2 - a3**2 - b1**2 - b2**2)) == 0
    report("r2-9b", ok and okR,
           "toy M^2 = m0^2+|q|^2/c^2: the surplus velocity is du/dtau = -N q (ambient shell: +N q); its quadratic form "
           "P^2-|q|^2 = eta(R_n K,K), R_n = P_n - Q_n = exp(pi u1^u2) in SO(3,3) the half-turn of U_n, signature (1,5) "
           "(n = e1, (2,3,6)/7): the toy sign is the ambient one with the surplus times made spacelike")


# ============================================================================= r2-10 rotating the clock
def check_clock_rotation():
    w, x, y, z = sp.symbols("w x y z", real=True)
    n3, u13, u23 = frame_from_quaternion(w, x, y, z)
    E1, E2, E3, c = sp.symbols("E1 E2 E3 c", real=True)
    p1, p2, p3 = sp.symbols("p1 p2 p3", real=True)
    Ev = sp.Matrix([E1, E2, E3])
    p2s = p1**2 + p2**2 + p3**2
    En = n3.dot(Ev)
    mu2c4 = Ev.dot(Ev) - p2s * c**2
    m42c4 = En**2 - p2s * c**2
    ok = zero(sp.simplify(m42c4 - (mu2c4 - n3.cross(Ev).dot(n3.cross(Ev)))))
    # E_n <= |E| with equality iff n parallel to E (Cauchy-Schwarz via Lagrange identity); explicit maximiser
    ok &= zero(sp.simplify(Ev.dot(Ev) - En**2 - n3.cross(Ev).dot(n3.cross(Ev))))
    report("r2-10a", ok,
           "general clock (symbolic): E(n) = c eta(K,n) = n.E and m_4(n)^2 c^4 = mu^2 c^4 - |n x E|^2 with mu^2 c^4 = |E|^2 - |p|^2 c^2; "
           "hence m_4(n)^2 <= mu^2, equality iff n is parallel or antiparallel to E (the temporal part of K); the energy read "
           "by the clock, E(n) = n.E, is maximal (= |E|) at n = E/|E|")
    # exact example with mu > 0: E = (2,3,6), p = (0,0,3), c = 1
    Eex = sp.Matrix([2, 3, 6])
    pex2 = 9
    mu2 = Eex.dot(Eex) - pex2
    nstar = Eex / 7
    m4 = lambda nn: (nn.dot(Eex))**2 - pex2
    W = sp.Matrix([1, 1, 1]) / sp.sqrt(3)
    EW = W.dot(Eex)
    Eperp = Eex - EW * W
    ok = mu2 == 40 and m4(nstar) == 40 and m4(sp.Matrix([1, 0, 0])) == -5 and sp.nsimplify(m4(W)) == R3(94, 3)
    ok &= sp.nsimplify(mu2 - Eperp.dot(Eperp)) == R3(94, 3) and all(v >= 0 for v in nstar)
    wstar = [v**2 for v in nstar]
    ok &= wstar == [R3(4, 49), R3(9, 49), R3(36, 49)]
    ok &= m4(-nstar) == 40 and (-nstar).dot(Eex) == -7
    report("r2-10b", ok,
           "E=(2,3,6), p=(0,0,3), c=1: mu^2=40; the maximising clock (2,3,6)/7 lies in the Fisher-Rao octant with weights "
           "(4,9,36)/49 and gives m_4^2=40, as does the opposite clock -(2,3,6)/7 (energy -7); at e1 m_4^2=-5; at W "
           "m_4^2 = 94/3 = mu^2 - |E_perp|^2 (RE's E_W)")
    # Fisher-Rao octant: max of n.E over n>=0, |n|=1 is |E^+| at n = E^+/|E^+|  (random test, floats)
    rng = np.random.default_rng(20261025)
    worst = 0.0
    attained = True
    for _ in range(200):
        Er = rng.normal(size=3)
        Ep = np.maximum(Er, 0.0)
        if np.linalg.norm(Ep) < 1e-9:
            continue
        pts = np.abs(rng.normal(size=(4000, 3)))
        pts /= np.linalg.norm(pts, axis=1)[:, None]
        worst = max(worst, float(np.max(pts @ Er) - np.linalg.norm(Ep)))
        nst = Ep / np.linalg.norm(Ep)
        attained &= abs(nst @ Er - np.linalg.norm(Ep)) < 1e-12
    report("r2-10c", worst <= 1e-12 and attained,
           "Fisher-Rao octant (200 random E, 4000 random octant clocks each, float, tol 1e-12): n.E <= |E^+| with E^+ the "
           "positive part, attained at n = E^+/|E^+|, i.e. weights w_i = (E_i^+)^2/|E^+|^2")
    # the case E^+ = 0: the octant maximum is max_i E_i <= 0, at a vertex (not |E^+| = 0)
    n1s, n2s, n3s = sp.symbols("n1 n2 n3", nonnegative=True)
    # (sum n_i)^2 = |n|^2 + 2 sum_{i<j} n_i n_j >= |n|^2 = 1 for n_i >= 0
    oks = sp.expand((n1s + n2s + n3s)**2 - (n1s**2 + n2s**2 + n3s**2) - 2 * (n1s * n2s + n1s * n3s + n2s * n3s)) == 0
    Eneg = sp.Matrix([-1, -2, -3])
    oks &= max(Eneg) == -1 and sp.Matrix([1, 0, 0]).dot(Eneg) == -1
    worst0 = -np.inf
    for _ in range(100):
        Er = -np.abs(rng.normal(size=3))
        pts = np.abs(rng.normal(size=(4000, 3)))
        pts /= np.linalg.norm(pts, axis=1)[:, None]
        worst0 = max(worst0, float(np.max(pts @ Er) - Er.max()))
    pts = np.abs(rng.normal(size=(20000, 3)))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    oks &= float(np.max(pts @ np.array([-1.0, -2.0, -3.0]))) <= -1 + 1e-12
    report("r2-10d", oks and worst0 <= 1e-12,
           "if E^+ = 0 the octant maximum of n.E is max_i E_i <= 0, attained at a vertex, since sum n_i >= |n| = 1 for n_i >= 0: "
           "E=(-1,-2,-3) gives -1 (not |E^+| = 0), and no octant clock reads a positive energy (100 random all-negative E "
           "against 4000 octant clocks and 20000 clocks at (-1,-2,-3), float, tol 1e-12)")


# ============================================================================= r2-11 .. r2-16 field four-momentum
def lint_on_E4():
    """lint(e1) as 4x4 matrices on E = <e1,e4,e5,e6> in the basis (e1,e4,e5,e6)."""
    idx = [0, 3, 4, 5]
    mats = []
    for g in lint_gens(unit(0)):
        mats.append(sp.Matrix(4, 4, lambda r, s: g[idx[r], idx[s]]))
    return mats


ETA4 = sp.diag(1, -1, -1, -1)
PAIRS = [(0, 1), (0, 2), (0, 3), (2, 3), (3, 1), (1, 2)]   # F01,F02,F03,F23,F31,F12


def F_from_X(X):
    """ES17 Thm 12.3: F_{0i} = E_i, F_{jk} = eps_ijk B_i; X = (B1,B2,B3,E1,E2,E3)."""
    F = sp.zeros(4, 4)
    for i in range(3):
        F[0, i + 1] = X[i + 3]
        F[i + 1, 0] = -X[i + 3]
    for i, (j, k) in enumerate([(2, 3), (3, 1), (1, 2)]):
        F[j, k] = X[i]
        F[k, j] = -X[i]
    return F


def F_vec(F):
    return sp.Matrix([F[a, b] for (a, b) in PAIRS])


def rho_L2(X4):
    """Action of a Lorentz matrix X4 (on vectors) on covariant two-forms F -> -(X^T F + F X), in the PAIRS basis."""
    cols = []
    for (a, b) in PAIRS:
        F = sp.zeros(4, 4)
        F[a, b] = 1
        F[b, a] = -1
        cols.append(F_vec(-(X4.T * F + F * X4)))
    return sp.Matrix.hstack(*cols)


def sym2_rep(R, d):
    """Representation on Sym^2 of a d-dim rep matrix R, basis v_i v_j (i<=j)."""
    basis = [(i, j) for i in range(d) for j in range(i, d)]
    pos = {b: t for t, b in enumerate(basis)}
    M = sp.zeros(len(basis), len(basis))
    for t, (i, j) in enumerate(basis):
        # (R v_i) v_j + v_i (R v_j)
        for k in range(d):
            for (a, b, coef) in [(k, j, R[k, i]), (i, k, R[k, j])]:
                if coef == 0:
                    continue
                key = (min(a, b), max(a, b))
                M[pos[key], t] += coef
    return M


def hom_dim(repA, repB, extra=None):
    """dim Hom_g(A,B) = dim{T : T rhoA(x) = rhoB(x) T for all generators}."""
    dA = repA[0].shape[0]
    dB = repB[0].shape[0]
    T = sp.Matrix(dB, dA, sp.symbols(f"t0:{dA * dB}"))
    eqs = []
    for RA, RB in zip(repA, repB):
        eqs += list(T * RA - RB * T)
    if extra is not None:
        eqs += extra(T)
    ts = list(T)
    Mrows = sp.Matrix([[sp.diff(q, t) for t in ts] for q in eqs])
    return len(ts) - qrank(Mrows)


def lG_on_V():
    gens = []
    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        gens.append(Mb(unit(j + 3), unit(k + 3)) - Mb(unit(j), unit(k)))
    for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        gens.append(Mb(unit(k), unit(j + 3)) - Mb(unit(j), unit(k + 3)))
    return gens


def check_field_reps():
    L4 = lint_on_E4()
    L2 = [rho_L2(X4) for X4 in L4]
    # (a) lint: Hom(Lambda^2, E) = 0, also Hom(Lambda^2, Lambda^2) = 2 (id and star) as a sanity check
    d_lin = hom_dim(L2, L4)
    d_end = hom_dim(L2, L2)
    # lint on the six directions of V (4+1+1): Hom(V restricted, E) = 1
    d_V = hom_dim(lint_gens(unit(0)), L4)
    # general clock n: lint(n) on V; Hom(V, V) contains Hom(Lambda^2-module...) -- check Hom(Lambda^2 E_n, E_n)=0 for n=(2,3,6)/7
    n, u1, u2 = frame6("(2,3,6)/7")
    basisE = [n, unit(3), unit(4), unit(5)]
    sig = [1, -1, -1, -1]
    L4n = [sp.Matrix(4, 4, lambda r, s: sig[r] * eta(basisE[r], g * basisE[s])) for g in lint_gens(n)]
    d_lin_n = hom_dim([rho_L2(X4) for X4 in L4n], L4n)
    report("r2-11a", d_lin == 0 and d_end == 2 and d_V == 1 and d_lin_n == 0,
           "dim Hom_lint(Lambda^2 E^*, E) = 0 (n = e1 and n = (2,3,6)/7); controls: dim End_lint(Lambda^2) = 2 "
           "(identity and Hodge star), dim Hom_lint(V, E) = 1")
    # lG: find phi: lG -> lint with Phi rho_V(Y) = rho_L2(phi(Y)) Phi, Phi = F_from_X
    Phi = sp.Matrix.hstack(*[F_vec(F_from_X(unit(a))) for a in range(6)])
    gensG = lG_on_V()
    phis = []
    okphi = True
    cs = sp.symbols("c0:6")
    for Y in gensG:
        target = Phi * Y * Phi.inv()
        cand = sum((cs[t] * L2[t] for t in range(6)), sp.zeros(6, 6))
        sol = sp.solve(list(cand - target), cs, dict=True)
        if len(sol) != 1:
            okphi = False
            phis.append(None)
            continue
        phis.append([sol[0][cc] for cc in cs])
    phi4 = [sum((ph[t] * L4[t] for t in range(6)), sp.zeros(4, 4)) for ph in phis] if okphi else []
    # phi is a Lie algebra homomorphism
    if okphi:
        for a in range(6):
            for b in range(6):
                br = comm(gensG[a], gensG[b])
                # express br in gensG
                ds = sp.symbols("d0:6")
                sol = sp.solve(list(sum((ds[t] * gensG[t] for t in range(6)), sp.zeros(6, 6)) - br), ds, dict=True)[0]
                lhs = sum((sol[ds[t]] * phi4[t] for t in range(6)), sp.zeros(4, 4))
                okphi &= lhs == comm(phi4[a], phi4[b])
    report("r2-12a", okphi,
           "the dictionary Phi: X -> F_X (F_0i = E_i, F_jk = eps_ijk B_i) intertwines lG on V with lint on Lambda^2 E^* "
           "through a unique Lie-algebra isomorphism phi: lG -> lint (ES17 Thm 12.3 restated)")
    if okphi:
        d_G = hom_dim(gensG, phi4)
        Sym2G = [sym2_rep(Y, 6) for Y in gensG]
        triv = [sp.zeros(1, 1) for _ in range(6)]
        d_s_vec = hom_dim(Sym2G, phi4)
        d_s_triv = hom_dim(Sym2G, triv)
        Sym2E = [sym2_rep(X4, 4) for X4 in phi4]
        d_s_sym = hom_dim(Sym2G, Sym2E)
        # traceless condition: eta_{mu nu} T^{mu nu} = 0 on the image; basis (i<=j) of Sym^2 E: trace = sum_i eta_ii coef(i,i)
        basis = [(i, j) for i in range(4) for j in range(i, 4)]
        trrow = [ETA4[i, i] if i == j else 0 for (i, j) in basis]

        def traceless(T):
            return [sum(trrow[r] * T[r, col] for r in range(10)) for col in range(T.shape[1])]
        d_s_sym0 = hom_dim(Sym2G, Sym2E, extra=traceless)
        d_s_U = hom_dim(Sym2G, [sp.zeros(2, 2) for _ in range(6)])     # U_n: a trivial two-dimensional module
        report("r2-11b", d_G == 0 and d_s_vec == 0 and d_s_triv == 2 and d_s_sym == 3 and d_s_sym0 == 1 and d_s_U == 4,
               "with lG as Lorentz algebra: dim Hom(V_EM, E) = 0; quadratic maps: dim Hom(Sym^2 V_EM, E) = 0, "
               "dim Hom(Sym^2 V_EM, R) = 2 (|B|^2-|E|^2 and E.B), dim Hom(Sym^2 V_EM, Sym^2 E) = 3, traceless part 1 "
               "(the stress tensor): no covariant map, linear or quadratic, from the field to a four-vector; but "
               "dim Hom(Sym^2 V_EM, U_n) = 4 (U_n trivial): covariance alone allows quadratic maps to the surplus plane")
    else:
        report("r2-11b", False, "lG representation-theoretic counts (phi not found)")
    # Hodge star: Phi(G X) = star Phi(X), star^2 = -1, star commutes with lint on Lambda^2, anti-isometry, neutral
    epsl = lambda a, b, c_, d: sp.LeviCivita(a, b, c_, d)
    Xs = sp.Matrix(sp.symbols("b1 b2 b3 f1 f2 f3", real=True))
    F = F_from_X(Xs)
    Fup = ETA4 * F * ETA4
    star = sp.zeros(4, 4)
    for a in range(4):
        for b in range(4):
            star[a, b] = sp.Rational(1, 2) * sum(epsl(a, b, c_, d) * Fup[c_, d] for c_ in range(4) for d in range(4))
    ok = zero(star - F_from_X(Gm * Xs))
    S6 = sp.Matrix.hstack(*[F_vec(F_from_X(Gm * unit(a))) for a in range(6)]) * Phi.inv()
    ok &= S6 * S6 == -sp.eye(6) and all(S6 * R == R * S6 for R in L2)
    gram = sp.Matrix(6, 6, lambda r, s: sp.Rational(1, 2) * sum(
        (F_from_X(Phi.inv() * unit(r)))[a, b] * (ETA4 * F_from_X(Phi.inv() * unit(s)) * ETA4)[a, b] for a in range(4) for b in range(4)))
    ok &= S6.T * gram * S6 == -gram and gram * S6 == (gram * S6).T
    ok &= sorted([sp.sign(v) for v in gram.eigenvals(multiple=True)]) == [-1, -1, -1, 1, 1, 1]
    ok &= sp.expand(sp.Rational(1, 2) * sum(F[a, b] * Fup[a, b] for a in range(4) for b in range(4))
                    - (Xs[0]**2 + Xs[1]**2 + Xs[2]**2 - Xs[3]**2 - Xs[4]**2 - Xs[5]**2)) == 0
    report("r2-12b", ok,
           "F_{GX} = star F_X (eps_0123 = +1); on Lambda^2 E^* star^2 = -1, star commutes with lint, is self-adjoint and "
           "anti-isometric for <F,F> = F_ab F^ab/2 = |B|^2-|E|^2 = eta(X,X), whose signature is (3,3): the Hodge star "
           "acts on the whole (neutral) field module")
    return phis


def stress(X):
    """Stress tensor Theta^{mu nu} (Heaviside-Lorentz, c=1) of the paper field X = (B;E), Jackson's formula."""
    F = F_from_X(X)
    Fup = ETA4 * F * ETA4
    FF = sum(F[a, b] * Fup[a, b] for a in range(4) for b in range(4))
    Th = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            Th[mu, nu] = sum(ETA4[mu, mu] * F[mu, lam] * Fup[lam, nu] for lam in range(4)) + sp.Rational(1, 4) * ETA4[mu, nu] * FF
    return Th


def check_stress():
    Bs = sp.symbols("B1:4", real=True)
    Es = sp.symbols("E1:4", real=True)
    X = sp.Matrix(list(Bs) + list(Es))
    Th = stress(X)
    Bv, Ev = sp.Matrix(Bs), sp.Matrix(Es)
    Bt = -Bv
    E6 = (Bv.dot(Bv) + Ev.dot(Ev)) / 2
    S = Ev.cross(Bv)
    St = Ev.cross(Bt)
    ok = sp.expand(Th[0, 0] - E6) == 0
    ok &= all(sp.expand(Th[0, a + 1] - St[a]) == 0 and sp.expand(Th[0, a + 1] + S[a]) == 0 for a in range(3))
    ok &= zero(Th - Th.T) and sp.expand(sum(ETA4[m, m] * Th[m, m] for m in range(4))) == 0
    # Rainich: Theta^mu_lam Theta^lam_nu = E0^2 delta, E0^2 = |q|^2/4
    qre = Bv.dot(Bv) - Ev.dot(Ev)
    qim = 2 * Ev.dot(Bv)
    E0sq = (qre**2 + qim**2) / 4
    mixed = Th * ETA4
    ok &= zero(sp.expand(mixed * mixed - E0sq * sp.eye(4)))
    ok &= sp.expand(E6**2 - E0sq - S.dot(S)) == 0
    report("r2-13", ok,
           "Theta^00 = E6, Theta^0a = (S_t)_a = -(E x B)_a = -S_a (paper B), Theta symmetric and traceless, "
           "Theta^mu_l Theta^l_nu = E0^2 delta (RE Prop 2.5), E6^2 = E0^2 + |S|^2 (RE Prop 2.4), symbolic in (B,E)")
    # explicit boost: X' = exp(chi ad K3) X with ad K3^3 = ad K3
    K3 = Mb(unit(1), unit(3)) - Mb(unit(0), unit(4))
    chi = sp.symbols("chi", real=True)
    ok = K3**3 == K3
    expK = I6 + sp.sinh(chi) * K3 + (sp.cosh(chi) - 1) * K3**2
    Xr = unit(0)
    Xb = expK * Xr
    ok &= zero(sp.simplify(Xb - sp.Matrix([sp.cosh(chi), 0, 0, 0, sp.sinh(chi), 0])))
    Bb, Eb = Xb[:3, 0], Xb[3:, 0]
    E6b = sp.simplify((Bb.dot(Bb) + Eb.dot(Eb)) / 2)
    Sb = sp.simplify(Eb.cross(Bb))
    ok &= sp.simplify(E6b - sp.cosh(2 * chi) / 2) == 0 and zero(sp.simplify(Sb - sp.Matrix([0, 0, -sp.sinh(2 * chi) / 2])))
    ok &= sp.simplify(E6b - sp.cosh(chi) / 2) != 0
    report("r2-14a", ok,
           "pointwise (E6,S) is not a four-vector: the rest field X=e1 (E6=E0=1/2, S=0) boosted by exp(chi ad K3) has "
           "B=(cosh chi,0,0), E=(0,sinh chi,0), E6 = cosh(2chi)/2, S = (0,0,-sinh(2chi)/2), not (cosh chi, sinh chi)/2")
    # tensor law: Theta[exp(chi K3) X] = L Theta[X] L^T, L the passage to the frame moving with -tanh(chi) e3
    Lp = sp.Matrix([[sp.cosh(chi), 0, 0, sp.sinh(chi)], [0, 1, 0, 0], [0, 0, 1, 0], [sp.sinh(chi), 0, 0, sp.cosh(chi)]])
    Lm = Lp.subs(chi, -chi)
    Thb = stress(expK * X)
    okp = zero(sp.simplify(Thb - Lp * Th * Lp.T))
    okm = zero(sp.simplify(Thb - Lm * Th * Lm.T))
    report("r2-14b", okp and not okm,
           "for general (B,E): Theta[exp(chi ad K3)X] = L Theta[X] L^T with L = [[cosh,sinh],[sinh,cosh]] on (0,3), the "
           "passage to the frame of velocity -tanh(chi) e3 (ES17 Sec. 12, RE Thm 2.6): the first row of Theta "
           "transforms quadratically in L")


def check_packets():
    s, zc = sp.symbols("s z", real=True)
    t = sp.symbols("t", positive=True)
    a, b = sp.symbols("a b", positive=True)
    chi = sp.symbols("chi", real=True)
    f = lambda arg: (1 - arg**2)**2      # on |arg|<=1, zero outside
    Nint = sp.integrate(f(s)**2, (s, -1, 1))
    ok = Nint == R3(256, 315)
    # pointwise densities of E_x = a f(z-t) + b f(z+t), B_ty = a f(z-t) - b f(z+t) (textbook), B_paper = -B_t
    f1, f2 = sp.symbols("f1 f2", real=True)
    Ex = a * f1 + b * f2
    Bty = a * f1 - b * f2
    u = (Ex**2 + Bty**2) / 2
    Stz = Ex * Bty
    ok &= sp.expand(u - (a**2 * f1**2 + b**2 * f2**2)) == 0 and sp.expand(Stz - (a**2 * f1**2 - b**2 * f2**2)) == 0
    # paper variables: B = (0,-B_ty,0), E = (E_x,0,0): S = E x B has z-component -Ex*Bty = -S_tz
    Bp = sp.Matrix([0, -Bty, 0])
    Ep = sp.Matrix([Ex, 0, 0])
    ok &= sp.expand(Ep.cross(Bp)[2] + Stz) == 0
    qpaper = Bp.dot(Bp) - Ep.dot(Ep)
    ok &= sp.expand(qpaper + 4 * a * b * f1 * f2) == 0 and Ep.dot(Bp) == 0
    # integrals per unit transverse area at time t: support of f(z-t) is [t-1,t+1]; cross terms are absent pointwise
    U_t = sp.integrate(a**2 * f(zc - t)**2, (zc, t - 1, t + 1)) + sp.integrate(b**2 * f(zc + t)**2, (zc, -t - 1, -t + 1))
    P_t = sp.integrate(a**2 * f(zc - t)**2, (zc, t - 1, t + 1)) - sp.integrate(b**2 * f(zc + t)**2, (zc, -t - 1, -t + 1))
    U = (a**2 + b**2) * Nint
    P = (a**2 - b**2) * Nint
    ok &= sp.simplify(U_t - U) == 0 and sp.simplify(P_t - P) == 0
    M = sp.sqrt(sp.expand(U**2 - P**2))
    ok &= sp.simplify(M - 2 * a * b * Nint) == 0
    ar, br = sp.symbols("a_r b_r", real=True)          # for real amplitudes of either sign M = (512/315)|ab|
    Ur, Pr = (ar**2 + br**2) * Nint, (ar**2 - br**2) * Nint
    ok &= sp.simplify(sp.sqrt(sp.factor(Ur**2 - Pr**2)) - 2 * Nint * sp.Abs(ar * br)) == 0
    # genuine boost of the fields along z with velocity beta = tanh(chi): E'_x = gamma(E_x - beta B_ty),
    # B'_ty = gamma(B_ty - beta E_x), at z = gamma(z' + beta t'), t = gamma(t' + beta z')
    zp, tp = sp.symbols("zp tp", real=True)
    gam = sp.cosh(chi)
    beta = sp.tanh(chi)
    zz = gam * (zp + beta * tp)
    tt = gam * (tp + beta * zp)
    ok &= sp.simplify((zz - tt - sp.exp(-chi) * (zp - tp)).rewrite(sp.exp)) == 0
    ok &= sp.simplify((zz + tt - sp.exp(chi) * (zp + tp)).rewrite(sp.exp)) == 0
    F1, F2 = sp.symbols("F1 F2", real=True)     # values f(z-t), f(z+t)
    Exp_ = gam * ((a * F1 + b * F2) - beta * (a * F1 - b * F2))
    Byp_ = gam * ((a * F1 - b * F2) - beta * (a * F1 + b * F2))
    ok &= sp.simplify((Exp_ - (a * sp.exp(-chi) * F1 + b * sp.exp(chi) * F2)).rewrite(sp.exp)) == 0
    ok &= sp.simplify((Byp_ - (a * sp.exp(-chi) * F1 - b * sp.exp(chi) * F2)).rewrite(sp.exp)) == 0
    # so u' = a^2 e^{-2chi} f(e^{-chi}(z'-t'))^2 + b^2 e^{2chi} f(e^{chi}(z'+t'))^2 (no cross terms), integrate over z'
    X1 = sp.exp(-chi) * (zp - tp)
    X2 = sp.exp(chi) * (zp + tp)
    I1 = sp.integrate(sp.expand(f(X1)**2), (zp, tp - sp.exp(chi), tp + sp.exp(chi)))
    I2 = sp.integrate(sp.expand(f(X2)**2), (zp, -tp - sp.exp(-chi), -tp + sp.exp(-chi)))
    Ub = a**2 * sp.exp(-2 * chi) * I1 + b**2 * sp.exp(2 * chi) * I2
    Pb = a**2 * sp.exp(-2 * chi) * I1 - b**2 * sp.exp(2 * chi) * I2
    ok &= sp.simplify((Ub - (U * sp.cosh(chi) - P * sp.sinh(chi))).rewrite(sp.exp)) == 0
    ok &= sp.simplify((Pb - (P * sp.cosh(chi) - U * sp.sinh(chi))).rewrite(sp.exp)) == 0
    ok &= sp.simplify(sp.expand(Ub**2 - Pb**2) - (U**2 - P**2)) == 0
    # for t >= 1 the supports [t-1,t+1] and [-t-1,-t+1] meet at most in a point, so f(z-t) f(z+t) = 0 a.e.
    ok &= sp.solve_univariate_inequality((t - 1) - (-t + 1) >= 0, t, relational=False) == sp.Interval(1, sp.oo)
    report("r2-15", ok,
           "counter-propagating packets E_x = a f(z-t)+b f(z+t), B_ty = a f(z-t)-b f(z+t), f=(1-s^2)^2 (HL, c=1): u and "
           "S_tz have no cross terms, U=(a^2+b^2)256/315, P_z=(a^2-b^2)256/315 at all t, M = 2|ab| 256/315; under the boost of "
           "rapidity chi along z (U,P_z) -> (U cosh - P sinh, P cosh - U sinh) exactly; paper sign S_z = -S_tz; "
           "q = -4ab f(z-t)f(z+t) vanishes for |t|>=1 while M > 0")
    # the interference term is conserved: Theta[z1+z2]-Theta[z1]-Theta[z2] is divergence free for two solutions
    # (checked on two plane waves with non-collinear wave vectors; symbolic in the coordinates)
    x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3", real=True)
    coords = [x0, x1, x2, x3]
    def wave(kvec, pol, phase0):
        k = sp.Matrix(kvec)
        kn = sp.sqrt(k.dot(k))
        ph = k.dot(sp.Matrix([x1, x2, x3])) - kn * x0 + phase0
        Evec = sp.Matrix(pol) * sp.cos(ph)
        Btv = (k / kn).cross(sp.Matrix(pol)) * sp.cos(ph)     # textbook B for a wave along k
        return Evec, Btv
    Ea, Ba = wave([0, 0, 3], [1, 0, 0], 0)
    Eb, Bb = wave([4, 0, 0], [0, 1, 1], sp.Rational(1, 3))
    def Xpaper(Ev, Btv):
        return sp.Matrix(list(-Btv) + list(Ev))
    # Maxwell check (textbook vacuum): div E = 0, dB_t/dt = -curl E, dE/dt = curl B_t
    def curl(V):
        return sp.Matrix([sp.diff(V[2], x2) - sp.diff(V[1], x3), sp.diff(V[0], x3) - sp.diff(V[2], x1),
                          sp.diff(V[1], x1) - sp.diff(V[0], x2)])
    okm = True
    for (Ev, Btv) in [(Ea, Ba), (Eb, Bb)]:
        okm &= zero(sp.simplify(sp.diff(Btv, x0) + curl(Ev))) and zero(sp.simplify(sp.diff(Ev, x0) - curl(Btv)))
        okm &= sp.simplify(sum(sp.diff(Ev[i], [x1, x2, x3][i]) for i in range(3))) == 0
    Tsum = stress(Xpaper(Ea + Eb, Ba + Bb))
    Tcross = sp.expand(Tsum - stress(Xpaper(Ea, Ba)) - stress(Xpaper(Eb, Bb)))
    div = [sp.simplify(sum(sp.diff(Tcross[mu, nu], coords[mu]) for mu in range(4))) for nu in range(4)]
    # total four-momenta of the two waves per unit volume (averaged): null vectors k1, k2; sum timelike
    k1 = sp.Matrix([3, 0, 0, 3])
    k2 = sp.Matrix([4, 4, 0, 0])
    m2 = lambda v: (v.T * ETA4 * v)[0, 0]
    th = sp.symbols("theta", real=True)
    ka = sp.Matrix([1, 0, 0, 1])
    kb = sp.Matrix([1, sp.sin(th), 0, sp.cos(th)])
    okk = m2(k1) == 0 and m2(k2) == 0 and m2(k1 + k2) == 24 and sp.simplify(m2(ka + kb) - 2 * (1 - sp.cos(th))) == 0
    k3 = sp.Matrix([4, 0, 0, -4])                    # antiparallel (collinear) null momentum: still timelike sum
    okk &= m2(k3) == 0 and m2(k1 + k3) == 48 and sp.simplify(m2(ka + kb).subs(th, sp.pi)) == 4
    report("r2-16", okm and all(d == 0 for d in div) and okk,
           "two Maxwell plane waves with non-collinear wave vectors (0,0,3), (4,0,0): the interference part of Theta is "
           "divergence free (so it is conserved and drops out once the parts are disjoint); two future null momenta at angle "
           "theta add to a vector of mass^2 2E1E2(1-cos theta), timelike unless theta = 0: (3;0,0,3)+(4;4,0,0) has M^2 = 24 and "
           "the antiparallel (3;0,0,3)+(4;0,0,-4) has M^2 = 48")


def check_photon_and_null():
    # ES17 Ex 12.11: z = (1,i,0) e^{ik(x3-t)} is null, E6 = |S|, and S_t = -S points along +x3
    x3, t, k = sp.symbols("x3 t k", real=True)
    ph = k * (x3 - t)
    z = sp.Matrix([1, sp.I, 0]) * sp.exp(sp.I * ph)
    B = z.applyfunc(sp.re)
    E = z.applyfunc(sp.im)
    B = B.applyfunc(lambda v: sp.simplify(sp.expand_complex(v)))
    E = E.applyfunc(lambda v: sp.simplify(sp.expand_complex(v)))
    E6 = sp.simplify((B.dot(B) + E.dot(E)) / 2)
    S = (E.cross(B)).applyfunc(sp.simplify)
    ok = sp.simplify(B.dot(B) - E.dot(E)) == 0 and sp.simplify(E.dot(B)) == 0
    ok &= sp.simplify(E6 - 1) == 0 and zero(S - sp.Matrix([0, 0, -1]))
    # finite-dimensional matrices cannot obey [a, a^T] = 1 (trace), unlike the CAR of r2-0b
    A = sp.Matrix(8, 8, lambda r, s: (3 * r + 5 * s) % 7 - 3)
    ok &= (A * A.T - A.T * A).trace() == 0
    report("r2-17", ok,
           "ES17 Ex 12.11 wave: |B|=|E|, E.B=0, E6 = |S| = 1, textbook S_t = -S = (0,0,1) along propagation; a finite "
           "matrix has tr[A,A^T] = 0 != tr 1, so the arena's eight-dimensional CAR modes cannot be photon (CCR) modes")


# ============================================================================= r2-18 .. r2-22 the no-go in its scope
def check_nogo():
    # anti-isometry and G-invariant planes
    v = sp.Matrix(sp.symbols("v1:7", real=True))
    wv = sp.Matrix(sp.symbols("w1:7", real=True))
    ok = sp.expand(eta(Gm * v, Gm * wv) + eta(v, wv)) == 0
    g11, g12, g22 = eta(v, v), eta(v, Gm * v), eta(Gm * v, Gm * v)
    ok &= sp.expand(g22 + g11) == 0 and sp.expand((g11 * g22 - g12**2) + g11**2 + g12**2) == 0
    # random G-invariant 4-planes span(v,Gv,w,Gw): signature (2,2) (float, 300 samples)
    rng = np.random.default_rng(7)
    Gn = np.array(Gm.tolist(), dtype=float)
    En = np.diag(eta_d).astype(float)
    sig_ok = True
    for _ in range(300):
        a1, a2 = rng.normal(size=6), rng.normal(size=6)
        B = np.column_stack([a1, Gn @ a1, a2, Gn @ a2])
        gram = B.T @ En @ B
        evs = np.linalg.eigvalsh(gram)
        sig_ok &= (np.sum(evs > 1e-9) == 2 and np.sum(evs < -1e-9) == 2)
    report("r2-18", ok and sig_ok,
           "eta(Gv,Gw) = -eta(v,w) (symbolic); the Gram matrix of a G-invariant plane span(v,Gv) is [[a,b],[b,-a]], "
           "det = -(a^2+b^2) <= 0: signature (1,1) or degenerate; 300 random G-invariant four-planes have signature (2,2) "
           "(float, tol 1e-9): every nondegenerate G-invariant subspace is neutral, so no (1,3) plane is G-invariant")
    # leakage: Euler-Lagrange of L_int and the flow
    lam_ = sp.symbols("lambda", real=True)
    kap, m = sp.symbols("kappa m", positive=True)
    Xf = sp.Matrix([sp.Function(f"X{a}")(lam_) for a in range(1, 7)])
    Xd = Xf.diff(lam_)
    L = m / 2 * ((Xd + kap * Gm * Xf).T * ETA * (Xd + kap * Gm * Xf))[0, 0]
    p = sp.Matrix([sp.diff(L, Xd[a]) for a in range(6)])
    EL = sp.Matrix([sp.diff(p[a], lam_) - sp.diff(L, Xf[a]) for a in range(6)])
    ok = zero(sp.simplify(p - m * ETA * (Xd + kap * Gm * Xf)))
    ok &= zero(sp.simplify(EL - (p.diff(lam_) - kap * Gm.T * p)))
    pi = ETA.inv() * p
    ok &= zero(sp.simplify(ETA.inv() * Gm.T * ETA - Gm))      # so pi' = kappa G pi on shell
    th = sp.symbols("theta", real=True)
    expG = sp.cos(th) * I6 + sp.sin(th) * Gm
    ok &= zero(sp.simplify(exp_rot(Gm, th) - expG))
    pil = expG.subs(th, kap * lam_) * unit(4)
    ok &= zero(sp.simplify(pil - (sp.cos(kap * lam_) * unit(4) - sp.sin(kap * lam_) * unit(1))))
    ok &= zero(sp.simplify(pil.diff(lam_) - kap * Gm * pil))
    # neither P_n pi nor eta(pi,pi) is conserved: eta(e^{th G}pi, e^{th G}pi) = cos(2th) eta(pi,pi) + sin(2th) eta(pi,G pi)
    pv = sp.Matrix(sp.symbols("r1:7", real=True))
    ok &= sp.simplify(eta(expG * pv, expG * pv) - (sp.cos(2 * th) * eta(pv, pv) + sp.sin(2 * th) * eta(pv, Gm * pv))) == 0
    Pe1, Qe1 = projectors(unit(0))
    ok &= zero(sp.simplify(Pe1 * pil - sp.cos(kap * lam_) * unit(4)))
    report("r2-19a", ok,
           "L_int = (m/2) eta(X'+kappa G X, X'+kappa G X): p = m eta(X'+kappa GX), Euler-Lagrange p' = kappa G^T p, "
           "pi = eta^-1 p obeys pi' = kappa G pi; exp(theta G) = cos + sin G; pi(0)=e5 gives "
           "pi(lambda) = cos(kappa lambda) e5 - sin(kappa lambda) e2 (FMB eq. 17 confirmed), P_e1 pi = cos(kappa lambda) e5; "
           "eta(pi,pi) -> cos(2th) eta(pi,pi) + sin(2th) eta(pi,G pi) along the flow (symbolic)")
    # rank Q_n G P_n = 2 for every clock (symbolic frame), kernel on E_n = span(n, Gn)
    w, x, y, z = sp.symbols("w x y z", real=True)
    n3, u13, u23 = frame_from_quaternion(w, x, y, z)
    n = tvec(n3)
    Pn, Qn = projectors(n)
    QGP = sp.simplify(Qn * Gm * Pn)
    ok = zero(sp.simplify(Qn * Gm * n))
    block = QGP[:3, 3:]
    proj3 = sp.eye(3) - n3 * n3.T
    ok &= zero(sp.simplify(block + proj3)) and zero(sp.simplify(QGP[:, :3])) and zero(sp.simplify(QGP[3:, :]))
    ok &= zero(sp.simplify(proj3 * proj3 - proj3)) and sp.simplify(proj3.trace() - 2) == 0
    ok &= zero(sp.simplify(QGP * (Gm * n))) and zero(sp.simplify(QGP * n))
    # at n = e1 the rank is 2 exactly, and for the rational clocks
    ranks = [qrank(projectors(frame6(nm)[0])[1] * Gm * projectors(frame6(nm)[0])[0]) for nm in RATIONAL]
    ok &= all(r == 2 for r in ranks)
    report("r2-19b", ok,
           "for a general clock n (symbolic): Q_n G n = 0 and Q_n G P_n = -(1 - n n^T) o (e_{a+3} -> e_a), an idempotent of "
           "trace 2 composed with an isomorphism, so rank Q_n G P_n = 2 with kernel span(n, Gn) on E_n; exact rank 2 at "
           "e1,e2,e3,(1,2,2)/3,(2,3,6)/7,(-1,4,8)/9")
    # only span(n, Gn) stays: Q_n G v = 0 for v = alpha n + s iff s || G n
    al = sp.symbols("alpha", real=True)
    bs = sp.symbols("beta1:4", real=True)
    vv = al * n + svec(bs)
    cond = sp.simplify(Qn * Gm * vv)
    bvec = sp.Matrix(bs)
    ok = zero(sp.simplify(cond[:3, 0] + (bvec - n3 * n3.dot(bvec))))
    ok &= zero(sp.simplify(Gm * n - svec(n3)))
    n0, u10, u20 = frame6("e1")
    ok &= zero(Gm * n0 - unit(3))
    report("r2-19c", ok,
           "for v = alpha n + s in E_n, Q_n G v = -(s_+ - (n.s_+) n) with s_+ the temporal copy of s: Q_n G v = 0 iff s || Gn; "
           "hence exp(theta G) keeps v in E_n for all theta iff v in span(n, Gn), which is <e1,e4> at n = e1")
    # ES17 Prop 12.2: common fixed vectors and intersection of the two Lorentz algebras
    okc = True
    gG = lG_on_V()
    fixG = 6 - qrank(sp.Matrix.vstack(*gG))
    okc &= fixG == 0
    for name in ["e1", "(1,2,2)/3", "W"]:
        nn, u1, u2 = frame6(name)
        gi = lint_gens(nn)
        Mstack = sp.Matrix.vstack(*gi)
        fixI = 6 - qrank(Mstack) if name != "W" else 6 - sp.Matrix.vstack(*gi).rank(simplify=True)
        okc &= fixI == 2
        if name != "W":
            both = sp.Matrix.hstack(*[g.reshape(36, 1) for g in gi + gG])
            okc &= qrank(both) == 12
    # lG meets the stabiliser so(E_n)+so(U_n) of the clock in the line of the diagonal rotation sum n_i L_i
    for name in ["e1", "(1,2,2)/3"]:
        nn, u1, u2 = frame6(name)
        stab = lint_gens(nn) + [Mb(u1, u2)]
        A = sp.Matrix.hstack(*[g.reshape(36, 1) for g in stab])
        B = sp.Matrix.hstack(*[g.reshape(36, 1) for g in gG])
        inter = qrank(A) + qrank(B) - qrank(sp.Matrix.hstack(A, B))
        Ln = sum((nn[i] * gG[i] for i in range(3)), sp.zeros(6, 6))
        okc &= inter == 1 and qrank(sp.Matrix.hstack(A, Ln.reshape(36, 1))) == qrank(A)
    # G does not commute with lint(n), and commutes with lG
    okc &= all(Gm * g == g * Gm for g in gG) and any(Gm * g != g * Gm for g in lint_gens(unit(0)))
    report("r2-20", okc,
           "ES17 Prop 12.2 re-verified: lint(n) fixes the plane U_n (dim 2), lG fixes no nonzero vector; lG and lint(n) "
           "intersect in 0 (rank 12 of the 12 generators, n = e1, (1,2,2)/3); lG meets so(E_n)+so(U_n) only in the line of the "
           "diagonal rotation sum n_i L_i; G commutes with lG but not with lint")


CHECKS = [check_conventions, check_decomposition, check_covariance, check_J, check_dirac_clock, check_poisson,
          check_shells, check_option2, check_option3, check_clock_rotation, check_field_reps, check_stress,
          check_packets, check_photon_and_null, check_nogo]


def main():
    import time
    only = set(sys.argv[1:])
    for chk in CHECKS:
        if only and chk.__name__ not in only:
            continue
        t0 = time.time()
        chk()
        if only:
            print(f"    [{chk.__name__}: {time.time() - t0:.1f}s]")
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("all r2 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
