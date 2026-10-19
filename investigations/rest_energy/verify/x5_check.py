#!/usr/bin/env python3
"""Independent adversarial check of track d5 (duplication: Cl(3,3) -> Cl(6,6)).

Rebuilt from the paper sources only (tex/s02_arena.tex eq. realisation and eq. tower, s03 prop:twoK,
s10 sec:diagonal / thm:algebra / prop:orthospin / thm:casid, s12 thm:emfield, s13 prop:commutantsn /
sec:mixed, s17 eq:wdef / thm:drift / prop:several, s26 prop:rigid / def:towervac / thm:towervac / thm:S4,
s04 thm:i) and from the d4 conventions (L_i, K_i, admissible b).  Nothing is imported from
d5_duplication_math.py.

Methods (chosen to differ from a blade-parity derivation where possible):
  * every commutant / centraliser is computed as a numerical null space of the linear commutation
    conditions on matrices (SVD), or by direct matrix commutation of the 4096 blades of Cl(6,6);
  * vector actions of spinor-level generators are extracted by traces Tr(E_d^T [X, E_c]) / 64;
  * the Lorentz content of R^{6,6} is read from Casimir spectra, spin from exp(2 pi R_3);
  * the inter-copy b-field: the 24 x 24 generator is reduced to three identical 8 x 8 blocks, the
    characteristic polynomial and the identity of part (iv) are computed symbolically, exact
    boundedness is decided by the square-free part of the characteristic polynomial over QQ;
  * orbit sizes are sup_t ||exp(Mt)||_2 from an eigen-decomposition on a fine grid plus refinement.

A claim of the section found FALSE is checked in its CORRECTED form (so it passes); see the findings file.
Prints one PASS/FAIL line per check; exit status 0 iff every check passes.
"""
import sys
import time
import itertools
import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

T0 = time.time()
FAILS = []
RNG = np.random.default_rng(20261018)


def null_space(A, rcond=1e-9):
    """orthonormal basis (columns) of ker A; economy SVD so tall systems stay cheap."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if m >= n:
        _, s, vt = np.linalg.svd(A, full_matrices=False)
    else:
        _, s, vt = np.linalg.svd(A, full_matrices=True)
        s = np.concatenate([s, np.zeros(n - m)])
    smax = s.max() if s.size and s.max() > 0 else 1.0
    return vt[s <= rcond * smax].T


def report(cid, ok, text):
    ok = bool(ok)
    print(("PASS" if ok else "FAIL") + f" x5-{cid}: {text}")
    if not ok:
        FAILS.append(cid)


def kron(*ms):
    out = np.array([[1]])
    for m in ms:
        out = np.kron(out, m)
    return out


def close(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol, rtol=0)


def comm(a, b):
    return a @ b - b @ a


def acomm(a, b):
    return a @ b + b @ a


# ---------------------------------------------------------------------------------------------
# realisation (tex/s02_arena.tex eq. realisation) and tower (eq. tower)
# ---------------------------------------------------------------------------------------------
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
ep = np.array([[0, -1], [1, 0]])
I2 = np.eye(2, dtype=int)
e = [kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx),
     kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
eta = np.diag([1, 1, 1, -1, -1, -1])
I8 = np.eye(8, dtype=int)
I64 = np.eye(64, dtype=int)
w8 = e[0] @ e[1] @ e[2] @ e[3] @ e[4] @ e[5]
E = [np.kron(x, I8) for x in e] + [np.kron(w8, x) for x in e]       # E[6k+a] = e^{(k+1)}_{a+1}
eta12 = np.diag([1, 1, 1, -1, -1, -1] * 2)
G = np.zeros((6, 6), dtype=int)
for i in range(3):
    G[i + 3, i] = 1      # G e_i = e_{i+3}
    G[i, i + 3] = -1     # G e_{i+3} = -e_i
I6 = np.eye(6, dtype=int)
G2 = np.kron(I2, G)
Cv = np.kron(ep, I6)
eta2 = np.kron(I2, eta)
assert (eta2 == eta12).all()


def prodm(ms, n):
    out = np.eye(n, dtype=int)
    for m in ms:
        out = out @ m
    return out


def vecrep(X, gens):
    """matrix of v -> [X, v] on span(gens), coefficients by trace (gens orthogonal)."""
    n = len(gens)
    R = np.zeros((n, n))
    nn = gens[0].shape[0]
    for c in range(n):
        Y = X @ gens[c] - gens[c] @ X
        coeffs = np.array([np.trace(gens[d].T @ Y) / nn for d in range(n)])
        rec = sum(coeffs[d] * gens[d] for d in range(n))
        if not np.allclose(rec, Y, atol=1e-9):
            raise ValueError("commutator not in span of generators")
        R[:, c] = coeffs
    return R


def solve_space(n, funcs, tol=1e-9):
    """basis of {X in M_n(R): f(X) = 0 for all f} via SVD null space."""
    cols = []
    for k in range(n * n):
        Xk = np.zeros((n, n))
        Xk.flat[k] = 1.0
        cols.append(np.concatenate([np.asarray(f(Xk), dtype=float).ravel() for f in funcs]))
    A = np.array(cols).T
    ns = null_space(A, rcond=tol)
    return [ns[:, i].reshape(n, n) for i in range(ns.shape[1])]


def span_dim(mats):
    if len(mats) == 0:
        return 0
    return np.linalg.matrix_rank(np.array([m.ravel() for m in mats]), tol=1e-8)


def same_span(A, B):
    return span_dim(A) == span_dim(B) == span_dim(list(A) + list(B))


def in_span(X, mats):
    return span_dim(list(mats) + [X]) == span_dim(mats)


# linear conditions on n x n matrices
def c_comm(A):
    return lambda X: X @ A - A @ X


def c_antisym_eta(et):
    return lambda X: X.T @ et + et @ X


def c_antisym():
    return lambda X: X + X.T


# ---------------------------------------------------------------------------------------------
# 1  the duplicated algebra (prop:graded)
# ---------------------------------------------------------------------------------------------
ok = all((acomm(e[a], e[b]) == 2 * eta[a, b] * I8).all() for a in range(6) for b in range(6))
ok &= all(((np.abs(x).sum(axis=1) == 1).all() and (np.abs(x).sum(axis=0) == 1).all()) for x in e)
ok &= (w8 @ w8 == I8).all() and (w8.T == w8).all() and np.trace(w8) == 0
ok &= all((w8 @ x == -x @ w8).all() for x in e)
ok &= (w8 == -kron(sz, sz, sz)).all()
report("1a", ok, "realisation: e_a satisfy Cl(3,3), signed permutations; omega^2=1, omega^T=omega, Tr omega=0, "
       "omega anticommutes with e_a; omega=-sz(x)sz(x)sz is diagonal")

ok = all((acomm(E[A], E[B]) == 2 * eta12[A, B] * I64).all() for A in range(12) for B in range(12))
report("1b", ok, "tower: the twelve e^(k)_a satisfy the Clifford relations of R^{6,6}")

# all 4096 blades, built by products in increasing index order
blades = np.zeros((4096, 64, 64), dtype=np.int8)
for mask in range(4096):
    if mask == 0:
        blades[0] = I64
        continue
    hi = mask.bit_length() - 1
    blades[mask] = np.rint(blades[mask ^ (1 << hi)].astype(float) @ E[hi]).astype(np.int8)
absb = np.abs(blades.astype(np.int16))
signed_perm = bool((absb.sum(axis=2) == 1).all() and (absb.sum(axis=1) == 1).all())
tr = np.einsum('kii->k', blades.astype(np.int32))
traceless = bool(tr[0] == 64 and (tr[1:] == 0).all())
F = blades.reshape(4096, 4096).astype(np.float32)
gram = F @ F.T
gram_ok = bool(np.allclose(gram, 64 * np.eye(4096, dtype=np.float32)))
del F, gram
report("1c", signed_perm and traceless and gram_ok,
       "the 4096 blades of Cl(6,6) are signed permutation matrices, all but id traceless, and trace-orthogonal "
       "(Gram = 64 id), so they span M_64(R)")

# (ii) e^(1)_I e^(2)_J = (e_I omega^{|J|}) (x) e_J and the graded swap sign, for all 64 x 64 pairs
eb = {}
for mask in range(64):
    idx = [a for a in range(6) if mask >> a & 1]
    eb[mask] = (prodm([e[a] for a in idx], 8), len(idx))
ok = True
for mI in range(64):
    eI, gI = eb[mI]
    B1 = blades[mI].astype(float)
    for mJ in range(64):
        eJ, gJ = eb[mJ]
        B2 = blades[mJ << 6].astype(float)
        lhs = B1 @ B2
        ok &= (lhs == np.kron(eI @ np.linalg.matrix_power(w8, gJ), eJ)).all()
        ok &= (B2 @ B1 == (-1) ** (gI * gJ) * lhs).all()
        if not ok:
            break
# graded tensor product: phi(a^b) = a omega^{|b|} (x) b is multiplicative with the Koszul sign
def rand_hom(parity):
    masks = [m for m in range(64) if bin(m).count('1') % 2 == parity]
    sel = RNG.choice(masks, size=5, replace=False)
    return sum(RNG.integers(-3, 4) * eb[m][0] for m in sel)


def phi(a, b, pb):
    return np.kron(a @ np.linalg.matrix_power(w8, pb), b)


for _ in range(60):
    pa, pb, pc, pd = RNG.integers(0, 2, size=4)
    a, b, c, d = rand_hom(pa), rand_hom(pb), rand_hom(pc), rand_hom(pd)
    ok &= (phi(a, b, pb) @ phi(c, d, pd) == (-1) ** (pb * pc) * phi(a @ c, b @ d, (pb + pd) % 2)).all()
report("1d", ok, "e^(1)_I e^(2)_J=(e_I w^{|J|})(x)e_J, e^(2)_J e^(1)_I=(-1)^{|I||J|}e^(1)_I e^(2)_J (all 4096 pairs); "
       "a^b -> a w^{|b|}(x)b multiplicative for the graded product (60 random homogeneous quadruples)")

# (iii) commutant of {e_a (x) 1} via direct commutation of all blades (blades form a basis and each
# generator is a blade, so the commutant is spanned by the blades that commute)
Bf = blades.astype(np.float32)
def blade_commutes(gen, sign=+1):
    g = gen.astype(np.float32)
    left = (Bf.reshape(-1, 64) @ g).reshape(4096, 64, 64)
    right = (g @ Bf.transpose(1, 0, 2).reshape(64, -1)).reshape(64, 4096, 64).transpose(1, 0, 2)
    return np.all(np.abs(left - sign * right) < 1e-6, axis=(1, 2))


cm = np.ones(4096, bool)
am = np.ones(4096, bool)
for a in range(6):
    cm &= blade_commutes(E[a], +1)
    am &= blade_commutes(E[a], -1)
ok = cm.sum() == 64 and am.sum() == 64
for k in np.nonzero(cm)[0]:
    B = blades[k].astype(int)
    ok &= (B == np.kron(I8, B[:8, :8])).all()
W8I = np.kron(w8, I8)
for k in np.nonzero(am)[0]:
    C = W8I @ blades[k].astype(int)
    ok &= (C == np.kron(I8, C[:8, :8])).all()
report("1e", ok, "commutant of {e_a(x)1} in M_64 is 1(x)M_8 (64 blades), anticommutant is omega(x)M_8 (64 blades)")

# (iii) forced Klein factor: the naive copy 1 (x) e_a fails; any alternative omega(x)f_a is conjugate by 1(x)u
naive_fails = (np.kron(I8, e[0]) @ E[0] == E[0] @ np.kron(I8, e[0])).all()
u = RNG.integers(-2, 3, size=(8, 8)) + 5 * np.eye(8)
ui = np.linalg.inv(u)
f = [u @ x @ ui for x in e]
alt = [np.kron(w8, x) for x in f]
ok = all(np.allclose(acomm(alt[a], alt[b]), 2 * eta[a, b] * np.eye(64)) for a in range(6) for b in range(6))
ok &= all(np.allclose(acomm(alt[a], E[b]), 0) for a in range(6) for b in range(6))
U = np.kron(np.eye(8), u)
ok &= all(np.allclose(U @ E[6 + a] @ np.linalg.inv(U), alt[a]) for a in range(6))
ok &= all(np.allclose(U @ E[a] @ np.linalg.inv(U), E[a]) for a in range(6))
report("1f", ok and naive_fails, "Klein factor forced: 1(x)e_a commutes with e_a(x)1; a random Skolem-Noether "
       "alternative omega(x)(u e_a u^-1) is the tower conjugated by 1(x)u")

# ---------------------------------------------------------------------------------------------
# 2  Fock space (prop:fock)
# ---------------------------------------------------------------------------------------------
npl = [(e[i] + e[i + 3]) / 2 for i in range(3)]
nmi = [(e[i] - e[i + 3]) / 2 for i in range(3)]
ok = all(close(acomm(npl[i], npl[j]), 0) and close(acomm(nmi[i], nmi[j]), 0)
         and close(acomm(npl[i], nmi[j]), (i == j) * np.eye(8)) for i in range(3) for j in range(3))
ok &= all(close(npl[i].T, nmi[i]) for i in range(3))
pole = [e[i] @ e[i + 3] for i in range(3)]
Nn = [npl[i] @ nmi[i] for i in range(3)]
ok &= all(close(Nn[i], (np.eye(8) - pole[i]) / 2) for i in range(3))
par = prodm([(I8 - 2 * Nn[i]).round().astype(int) for i in range(3)], 8)
ok &= (par == -w8).all()
diagN = all(close(Nn[i], np.diag(np.diag(Nn[i]))) for i in range(3))
patterns = {tuple(int(round(Nn[i][r, r])) for i in range(3)) for r in range(8)}
ok &= diagN and len(patterns) == 8
report("2a", ok, "n^pm_i=(e_i pm e_{i+3})/2 satisfy the CAR with (n^+)^T=n^-; N_i=(1-omega_i)/2; (-1)^{N1+N2+N3}=-omega; "
       "the eight coordinate (pole) eigenlines are the eight Fock states")

modes = [np.kron(nmi[i], I8) for i in range(3)] + [np.kron(w8, nmi[i]) for i in range(3)]
ok = all(close(acomm(modes[i], modes[j]), 0) and close(acomm(modes[i], modes[j].T), (i == j) * np.eye(64))
         for i in range(6) for j in range(6))
Ntot = sum(m.T @ m for m in modes)
ev = np.round(np.linalg.eigvalsh(Ntot)).astype(int)
mult = [int((ev == k).sum()) for k in range(7)]
ok &= mult == [1, 6, 15, 20, 15, 6, 1]
parity = prodm([np.round(np.eye(64) - 2 * m.T @ m).astype(int) for m in modes], 64)
vol12 = prodm(E, 64)
ok &= (parity == vol12).all()
# the copy-2 string omega = -(-1)^{N^(1)}
ok &= (w8 == -prodm([np.round(I8 - 2 * Nn[i]).astype(int) for i in range(3)], 8)).all()
report("2b", ok, "n=2: six modes (copy-2 modes omega(x)n^pm_i) satisfy the CAR; total number spectrum 0..6 with "
       "multiplicities 1,6,15,20,15,6,1; (-1)^N = E_1...E_12, the volume element")

# ---------------------------------------------------------------------------------------------
# 3  halves and exchange (prop:cmrel)
# ---------------------------------------------------------------------------------------------
s2 = np.sqrt(2)
fh = [(E[a] + E[6 + a]) / s2 for a in range(6)]
gh = [(E[a] - E[6 + a]) / s2 for a in range(6)]
ok = all(close(acomm(fh[a], fh[b]), 2 * eta[a, b] * np.eye(64)) and close(acomm(gh[a], gh[b]), 2 * eta[a, b] * np.eye(64))
         and close(acomm(fh[a], gh[b]), 0) for a in range(6) for b in range(6))
report("3a", ok, "{f_a,f_b}={g_a,g_b}=2 eta_ab, {f_a,g_b}=0")

terms = [0.5 * eta[a, a] * E[a] @ E[6 + a] for a in range(6)]
Cb = sum(terms)
ok = all(close(comm(Cb, E[a]), -E[6 + a]) and close(comm(Cb, E[6 + a]), E[a]) for a in range(6))
ok &= all(close(comm(terms[a], terms[b]), 0) for a in range(6) for b in range(6))
ok &= all(close(t @ t, -0.25 * np.eye(64)) for t in terms)
ok &= close(expm(2 * np.pi * Cb), np.eye(64), 1e-8)
Uq = expm(np.pi / 4 * Cb)
Uqi = expm(-np.pi / 4 * Cb)
ok &= all(close(Uq @ E[a] @ Uqi, gh[a], 1e-9) and close(Uq @ E[6 + a] @ Uqi, fh[a], 1e-9) for a in range(6))
ok &= close(Cb, -0.5 * sum(eta[a, a] * fh[a] @ gh[a] for a in range(6)))
# sign remark: the vector action of C_b is -C_v
adCb = vecrep(Cb, E)
ok &= close(adCb, -Cv)
report("3b", ok, "C_b: e1->-e2, e2->e1 (ad C_b = -C_v on vectors); six commuting terms of square -1/4; exp(2pi C_b)=1; "
       "exp(pi/4 C_b) conjugates e^(1)->g, e^(2)->f; C_b=-1/2 sum eta f g")

pbits = ((1 - np.diag(w8)) // 2).astype(int)
Dk = np.diag([(-1) ** (pbits[i] * pbits[j]) for i in range(8) for j in range(8)])
tau = np.zeros((64, 64), dtype=int)
for i in range(8):
    for j in range(8):
        tau[j * 8 + i, i * 8 + j] = 1
tauK = Dk @ tau
gp = np.eye(64)
for x in gh:
    gp = gp @ x
ok = all((tauK @ E[a] @ tauK == E[6 + a]).all() for a in range(6))
ok &= (tauK @ tauK == I64).all() and np.trace(tauK) == 0
ok &= close(tauK, gp)
ok &= all(close(comm(tauK, fh[a]), 0) and close(acomm(tauK, gh[a]), 0) for a in range(6))
naive = not all((tau @ E[a] @ tau == E[6 + a]).all() for a in range(6))
report("3c", ok and naive, "Koszul swap tau_K=D tau: tau_K e^(1) tau_K=e^(2), tau_K^2=1, Tr=0, tau_K=+g1...g6; commutes with f, "
       "anticommutes with g; the naive swap tau fails")

ok = all(close(E[a] @ E[b] + E[6 + a] @ E[6 + b], fh[a] @ fh[b] + gh[a] @ gh[b]) for a in range(6) for b in range(6) if a != b)
ok &= close(E[1] @ E[2] + E[7] @ E[8], fh[1] @ fh[2] + gh[1] @ gh[2])
fvecs = np.hstack([np.eye(6), np.eye(6)]).T / s2
gvecs = np.hstack([np.eye(6), -np.eye(6)]).T / s2
ok &= np.linalg.matrix_rank(np.hstack([fvecs, G2 @ fvecs])) == 6 and np.linalg.matrix_rank(np.hstack([gvecs, G2 @ gvecs])) == 6
M12 = sum(0.5 * eta[m, m] * E[m] @ E[6 + m] for m in (0, 3, 4, 5))
ok &= close(acomm(Cb, tauK), 0) and close(acomm(M12, tauK), 0)
report("3d", ok, "diagonal bivectors = f_a f_b + g_a g_b, J^(1)+J^(2)=f2f3+g2g3, G_2 preserves both halves; C_b and M_12 "
       "anticommute with tau_K")

# ---------------------------------------------------------------------------------------------
# 4  symmetry of the duplicated energy (prop:energysym)
# ---------------------------------------------------------------------------------------------
# one-copy lG generators from the Clifford algebra (d4 conventions)
cyc = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
Lsp1 = [0.5 * (e[j + 3] @ e[k + 3] - e[j] @ e[k]) for (i, j, k) in cyc]
Ksp1 = [0.5 * (e[k] @ e[j + 3] - e[j] @ e[k + 3]) for (i, j, k) in cyc]
Lv = [vecrep(x, e) for x in Lsp1]
Kv = [vecrep(x, e) for x in Ksp1]
lG_ok = all(close(comm(x, G), 0) for x in Lv + Kv) and close(comm(Lv[0], Lv[1]), Lv[2]) \
    and all(close(Kv[i], G @ Lv[i]) for i in range(3))
# lG equals the centraliser of G in so(3,3)
cent1 = solve_space(6, [c_comm(G), c_antisym_eta(eta)])
lG_ok &= len(cent1) == 6 and same_span(cent1, Lv + Kv)
Ldiag = [np.kron(I2, x) for x in Lv]
Kdiag = [np.kron(I2, x) for x in Kv]
sxG, szG, epG, IG = np.kron(sx, G), np.kron(sz, G), np.kron(ep, G), np.kron(I2, G)
base = [c_comm(x) for x in Ldiag] + [c_comm(G2)]
sp_all = solve_space(12, base)
sp_u2 = solve_space(12, base + [c_antisym()])
sp_c2 = solve_space(12, base + [c_antisym_eta(eta2)])
sp_so2 = solve_space(12, base + [c_antisym(), c_antisym_eta(eta2)])
ok = lG_ok and len(sp_all) == 8 and len(sp_u2) == 4 and len(sp_c2) == 2 and len(sp_so2) == 1
ok &= same_span(sp_u2, [Cv, IG, szG, sxG]) and same_span(sp_c2, [Cv, epG]) and same_span(sp_so2, [Cv])
dag = lambda X: eta2 @ X.T @ eta2
ok &= all(close(dag(X), X) for X in (IG, szG, sxG))
report("4a", ok, "commutant of diagonal rotations and G_2 in gl(12): dim 8 = gl(2,C); in so(12): u(2)=span{C_v,1(x)G,"
       "sz(x)G,sx(x)G}; in so(6,6): so(2,C)=span{C_v,eps(x)G}; in both: R C_v; 1(x)G, sz(x)G, sx(x)G eta_2-self-adjoint")

u6 = solve_space(12, [c_comm(G2), c_antisym()])
u3_1 = solve_space(6, [c_comm(G), c_antisym()])
diag_u3 = [np.kron(I2, Y) for Y in u3_1]
com_u3 = solve_space(12, [c_comm(G2), c_antisym()] + [c_comm(D) for D in diag_u3])
com_u2 = solve_space(12, [c_comm(G2), c_antisym()] + [c_comm(D) for D in [Cv, IG, szG, sxG]])
ok = len(u6) == 36 and len(u3_1) == 9 and len(com_u3) == 4 and same_span(com_u3, [Cv, IG, szG, sxG])
ok &= len(com_u2) == 9 and same_span(com_u2, diag_u3)
report("4b", ok, "u(6)=so(12) cap Cent(G_2) has dim 36; diagonal u(3) and u(2)(x)1 are mutual commutants in it (dims 9, 4)")

# ---------------------------------------------------------------------------------------------
# 5  duplication maps (thm:dup) and the two rest energies (rem:twoE0)
# ---------------------------------------------------------------------------------------------
def solve_rect(nr, nc, funcs):
    cols = []
    for k in range(nr * nc):
        Xk = np.zeros((nr, nc))
        Xk.flat[k] = 1.0
        cols.append(np.concatenate([np.asarray(fn(Xk), float).ravel() for fn in funcs]))
    ns = null_space(np.array(cols).T)
    return [ns[:, i].reshape(nr, nc) for i in range(ns.shape[1])]


maps = solve_rect(12, 6, [lambda X, i=i: X @ Lv[i] - Ldiag[i] @ X for i in range(3)] + [lambda X: X @ G - G2 @ X])
ref = [np.vstack([I6, 0 * I6]), np.vstack([G, 0 * G]), np.vstack([0 * I6, I6]), np.vstack([0 * G, G])]
ok = len(maps) == 4 and same_span(maps, ref)
a1, b1, a2, b2 = sp.symbols('a1 b1 a2 b2', real=True)
Gs, es = sp.Matrix(G), sp.Matrix(eta)
Ls = sp.Matrix.vstack(a1 * sp.eye(6) + b1 * Gs, a2 * sp.eye(6) + b2 * Gs)
al, be = a1 + sp.I * b1, a2 + sp.I * b2
LtL = sp.simplify(Ls.T * Ls - (a1**2 + b1**2 + a2**2 + b2**2) * sp.eye(6))
s_ = sp.expand(al**2 + be**2)
Lte = sp.simplify(Ls.T * sp.Matrix(eta2) * Ls - (sp.re(s_) * es + sp.im(s_) * es * Gs))
ok &= LtL == sp.zeros(6) and Lte == sp.zeros(6)
report("5a", ok, "G-linear rotation-equivariant maps V->V_2 are iota_{alpha,beta} (dim 4); L^T L=(|a|^2+|b|^2)1 and "
       "L^T eta_2 L = Re(a^2+b^2) eta + Im(a^2+b^2) eta G (symbolic)")


def iota(alpha, beta):
    blk = lambda c: c.real * I6 + c.imag * G
    return np.vstack([blk(alpha), blk(beta)])


t_ = 0.8
Lh = iota(np.cosh(t_), 1j * np.sinh(t_))
ok = close(Lh.T @ eta2 @ Lh, eta) and not close(Lh.T @ Lh, np.eye(6))
ok &= sp.simplify((sp.Abs(al)**2 + sp.Abs(be)**2 - sp.re(al**2 + be**2)) - 2 * (b1**2 + b2**2)) == 0
report("5b", ok, "CORRECTED thm:dup(i): eta-isometry alone means alpha^2+beta^2=1 over C (non-compact conic; e.g. "
       "(cosh t, i sinh t) is an isometry, not real, not energy-preserving); only with |a|^2+|b|^2=1 does it force the real circle")


def zof(X):   # X in R^6 (B; E) -> z = B + iE
    return X[:3] + 1j * X[3:]


def q(z):
    return np.sum(z * z)


def Tmunu(z):
    B, Ef = z.real, z.imag
    Bt = -B
    T = np.zeros((4, 4))
    T[0, 0] = 0.5 * (Ef @ Ef + Bt @ Bt)
    T[0, 1:] = T[1:, 0] = np.cross(Ef, Bt)
    T[1:, 1:] = -(np.outer(Ef, Ef) + np.outer(Bt, Bt)) + 0.5 * np.eye(3) * (Ef @ Ef + Bt @ Bt)
    return T


eta4 = np.diag([1, -1, -1, -1])
ok = True
for _ in range(20):
    z = RNG.normal(size=3) + 1j * RNG.normal(size=3)
    alpha, beta = RNG.normal() + 1j * RNG.normal(), RNG.normal() + 1j * RNG.normal()
    nrm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    alpha, beta = alpha / nrm, beta / nrm
    X = np.concatenate([z.real, z.imag])
    Y = iota(alpha, beta) @ X
    z1, z2 = zof(Y[:6]), zof(Y[6:])
    E0 = abs(q(z)) / 2
    ok &= np.isclose(q(z1) + q(z2), (alpha**2 + beta**2) * q(z))
    ok &= np.isclose(abs(q(z1)) / 2 + abs(q(z2)) / 2, E0)
    ok &= close(Tmunu(alpha * z), abs(alpha) ** 2 * Tmunu(z)) and close(Tmunu(z1) + Tmunu(z2), Tmunu(z))
    TE = Tmunu(z) @ eta4
    ok &= close(TE @ TE, E0 ** 2 * np.eye(4))
    Tt = (Tmunu(z1) + Tmunu(z2)) @ eta4
    ok &= np.isclose(0.25 * np.trace(Tt @ Tt), E0 ** 2)
    ok &= np.isclose(0.5 * np.vdot(z1, z1).real + 0.5 * np.vdot(z2, z2).real, 0.5 * np.vdot(z, z).real)
report("5c", ok, "on S^3: q_2=(a^2+b^2)q, E0^(1)+E0^(2)=E0, T(az)=|a|^2T(z), T_tot=T, Rainich (T eta)^2=E0^2 1, E_pair=E0, "
       "E6+E6=E6 (20 random points)")

ab_, bb_ = sp.symbols('alpha beta')
x1, y1, x2, y2 = sp.symbols('x1 y1 x2 y2', real=True)
A_, B_ = x1 + sp.I * y1, x2 + sp.I * y2
wexp = sp.expand(A_**2 + B_**2)
imab = sp.expand((A_ * sp.conjugate(B_) - sp.conjugate(A_) * B_) / (2 * sp.I))
iden = sp.expand(wexp * sp.conjugate(wexp) - ((x1**2 + y1**2 + x2**2 + y2**2)**2 - 4 * imab**2))
ok = sp.simplify(iden) == 0
zt = np.array([1.0 + 0.3j, -0.4 + 1.1j, 0.7 - 0.2j])
vals = []
for _ in range(4000):
    v = RNG.normal(size=4)
    v /= np.linalg.norm(v)
    al_, be_ = v[0] + 1j * v[1], v[2] + 1j * v[3]
    vals.append(abs(al_**2 + be_**2))
ok &= min(vals) < 0.05 and max(vals) <= 1 + 1e-12 and abs((1 / s2) ** 2 + (1j / s2) ** 2) < 1e-15
torus = [(1j, 0.0), (np.exp(0.4j) * np.cos(0.7), np.exp(0.4j) * np.sin(0.7)), (np.exp(1.1j) / s2, -np.exp(1.1j) / s2)]
for al_, be_ in torus:
    q2 = (al_**2 + be_**2) * q(zt)
    ok &= np.isclose(abs(q2), abs(q(zt))) and not np.isclose(q2, q(zt))
    ok &= not (abs(np.imag(al_)) < 1e-12 and abs(np.imag(be_)) < 1e-12)
for th in (0.3, 1.0, np.pi / 2):
    ok &= np.isclose(abs(0.5 + 0.5 * np.exp(2j * th)), abs(np.cos(th)))
report("5d", ok, "CORRECTED thm:dup(ii)/assessment: |q_2|^2=|q|^2[(|a|^2+|b|^2)^2-4 Im(a conj b)^2]; on S^3 |q_2| sweeps [0,|q|] "
       "and |q_2|=|q| exactly on the 2-dim set e^{i phi}(cos th, sin th) (e.g. (i,0)), not only on the circle; q_2=q only on "
       "the circle; equal split |q_2|=|cos th||q|")

ok = True
for _ in range(10):
    z = RNG.normal(size=3) + 1j * RNG.normal(size=3)
    th = RNG.normal()
    ok &= close(Tmunu(np.exp(1j * th) * z), Tmunu(z))
    Oq, _r = np.linalg.qr(RNG.normal(size=(3, 3)))
    ok &= np.isclose(q(Oq @ z), q(z))
    Hm = RNG.normal(size=(3, 3)) + 1j * RNG.normal(size=(3, 3))
    Um = expm(1j * (Hm + Hm.conj().T))
    ok &= not close(Tmunu(Um @ z), Tmunu(z), 1e-6) and not np.isclose(q(Um @ z), q(z))
    phi_ = 0.6
    for R in (np.eye(3), -np.eye(3)):
        T2 = np.cos(phi_) ** 2 * Tmunu(z) + np.sin(phi_) ** 2 * Tmunu(R @ z)
        ok &= close(T2, Tmunu(z)) and np.isclose(np.cos(phi_) ** 2 * q(z) + np.sin(phi_) ** 2 * q(R @ z), q(z))
    Um_ = RNG.normal(size=(6, 3)) + 1j * RNG.normal(size=(6, 3))
    Ur, Vr = Um_.real, Um_.imag
    ok &= close(Um_.conj().T @ Um_ - Um_.T @ Um_, 2 * Vr.T @ Vr - 2j * Vr.T @ Ur)
report("5e", ok, "thm:dup(iii): duality phases preserve T, O(3) preserves q, generic U(3) neither, +-1 both; "
       "W*W - W^T W = 2V^TV - 2iV^TU")


def to_z12(Y):
    return zof(Y[:6]), zof(Y[6:])


ok = True
for _ in range(10):
    Y = RNG.normal(size=12)
    aa, bb = RNG.normal(size=2)
    Y2 = expm(aa * Cv + bb * epG) @ Y
    za, zb = to_z12(Y)
    zc, zd = to_z12(Y2)
    Q1 = np.array([[q(za), np.sum(za * zb)], [np.sum(zb * za), q(zb)]])
    Q2 = np.array([[q(zc), np.sum(zc * zd)], [np.sum(zd * zc), q(zd)]])
    ok &= np.isclose(np.trace(Q1), np.trace(Q2)) and np.isclose(np.linalg.det(Q1), np.linalg.det(Q2))
    ok &= not np.isclose(Q1[0, 0], Q2[0, 0])
zA, zB = np.array([1, 1j, 0]), np.array([1, -1j, 0])
Tt = (Tmunu(zA) + Tmunu(zB)) @ eta4
ok &= abs(q(zA)) < 1e-12 and abs(q(zB)) < 1e-12 and abs(q(zA) + q(zB)) < 1e-12
ok &= np.isclose(0.25 * np.trace(Tt @ Tt), 2.0)
ok &= np.isclose(Tmunu(zA)[0, 3], -Tmunu(zB)[0, 3]) and abs(Tmunu(zA)[0, 3]) > 0.5


def rand_null():
    n1 = RNG.normal(size=3)
    n1 /= np.linalg.norm(n1)
    u1 = np.cross(n1, RNG.normal(size=3))
    u1 /= np.linalg.norm(u1)
    v1 = np.cross(n1, u1)
    return (u1 + 1j * v1) * RNG.uniform(0.5, 2)


for _ in range(10):
    za, zb = rand_null(), rand_null()
    Tt = (Tmunu(za) + Tmunu(zb)) @ eta4
    ok &= abs(q(za)) < 1e-10 and 0.25 * np.trace(Tt @ Tt) > 1e-6
    Tt = (Tmunu(za) + Tmunu(2 * za)) @ eta4
    ok &= abs(0.25 * np.trace(Tt @ Tt)) < 1e-9
    al_ = RNG.normal() + 1j * RNG.normal()
    ok &= abs(q(al_ * za)) < 1e-9
report("5f", ok, "rem:twoE0: so(2,C) preserves Tr Q and det Q, not Q_11; z1=(1,i,0), z2=(1,-i,0): E0^(k)=0, q_2=0, "
       "counter-propagating, E_pair^2=2; non-collinear null pairs have E_pair>0, collinear ones 0; null stays null")

# ---------------------------------------------------------------------------------------------
# 6  isometric complex structure only on the duplicate (prop:cx)
# ---------------------------------------------------------------------------------------------
ok = True
J0 = np.kron(ep, np.eye(3))
for _ in range(300):
    A = RNG.normal(size=(6, 6))
    J = A @ J0 @ np.linalg.inv(A)
    Ssym = RNG.normal(size=(6, 6))
    Ssym = Ssym + Ssym.T
    Bf_ = Ssym + J.T @ Ssym @ J
    w_ = np.linalg.eigvalsh(Bf_)
    ok &= int((w_ > 1e-9).sum()) % 2 == 0
# no J in O(3,3) with J^2=-1: the eta-invariant complex structures would give signature (3,3) = odd
ok &= close(Cv @ Cv, -np.eye(12)) and close(Cv.T @ Cv, np.eye(12)) and close(Cv.T @ eta2 @ Cv, eta2)
ok &= close(comm(Cv, G2), 0) and all(close(comm(Cv, X), 0) for X in Ldiag + Kdiag)
report("6a", ok, "prop:cx(i) via the lemma 'J-invariant symmetric forms have even signature' (300 random J, forms), so no "
       "J in O(3,3) has J^2=-1; (ii) C_v^2=-1, orthogonal, eta_2-isometric, commutes with G_2 and diagonal lG")

so6 = solve_space(12, [c_comm(G2), c_antisym(), c_antisym_eta(eta2)])
so3_1 = solve_space(6, [c_comm(G), c_antisym(), c_antisym_eta(eta)])
cen_cv = solve_space(12, [c_comm(G2), c_antisym(), c_antisym_eta(eta2), c_comm(Cv)])
u3p = []
for (i, j) in itertools.combinations(range(3), 2):
    A = np.zeros((3, 3))
    A[i, j], A[j, i] = 1, -1
    u3p.append(np.kron(I2, np.kron(I2, A)))
for (i, j) in itertools.combinations_with_replacement(range(3), 2):
    Bm = np.zeros((3, 3))
    Bm[i, j] = Bm[j, i] = 1
    u3p.append(np.kron(ep, np.kron(I2, Bm)))
cen_boost = solve_space(12, [c_comm(G2), c_antisym(), c_antisym_eta(eta2), c_comm(Cv)] + [c_comm(K) for K in Kdiag])
tp, spp = [0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]
paired = all(close(X[np.ix_(tp, spp)], 0) and close(X[np.ix_(tp, tp)], X[np.ix_(spp, spp)]) for X in so6)
ok = len(so6) == 15 and len(so3_1) == 3 and paired and len(cen_cv) == 9 and same_span(cen_cv, u3p)
ok &= len(cen_boost) == 1 and same_span(cen_boost, [Cv])
report("6b", ok, "u(6) cap so(6,6)=so(6), dim 15, paired rotations; n=1: u(3) cap so(3,3) dim 3; centraliser of C_v in so(6) "
       "= {1(x)diag(A,A)+eps(x)diag(B,B)} dim 9; its part commuting with diagonal boosts = R C_v")

# ---------------------------------------------------------------------------------------------
# 7  Cent(G_2) = so(6,C) and the copy algebra (prop:so6C)
# ---------------------------------------------------------------------------------------------
dims, cdims = [], []
for n in (1, 2, 3):
    Gn, en = np.kron(np.eye(n), G), np.kron(np.eye(n), eta)
    dims.append(len(solve_space(6 * n, [c_comm(Gn), c_antisym_eta(en)])))
    cdims.append(len(solve_space(6 * n, [c_comm(Gn), c_antisym_eta(en), c_antisym()])))
cent2 = solve_space(12, [c_comm(G2), c_antisym_eta(eta2)])
perm = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]
blk = True
for X in cent2:
    Y = X[np.ix_(perm, perm)]
    R_, S_ = Y[:6, :6], Y[:6, 6:]
    blk &= close(Y[6:, 6:], R_) and close(Y[6:, :6], -S_) and close(R_, -R_.T) and close(S_, -S_.T)
ok = dims == [6, 30, 72] and cdims == [3, 15, 36] and blk
report("7a", ok, f"centraliser of G_n in so(3n,3n): dims {dims}, compact parts {cdims}; at n=2 every element is "
       "[[R,S],[-S,R]] (temporal, partners) with R,S antisymmetric")

zero_diag = lambda X: np.concatenate([X[:6, :6].ravel(), X[6:, 6:].ravel()])
cross = solve_space(12, [c_comm(G2), c_antisym_eta(eta2), zero_diag])
diagb = solve_space(12, [c_comm(G2), c_antisym_eta(eta2), lambda X: X[:6, 6:], lambda X: X[6:, :6]])
cb_ = np.array([X.ravel() for X in cross]).T
cas = np.zeros((len(cross), len(cross)))
for c, X in enumerate(cross):
    Y = -sum(comm(D, comm(D, X)) for D in Ldiag)
    cas[:, c] = np.linalg.lstsq(cb_, Y.ravel(), rcond=None)[0]
cev = np.round(np.linalg.eigvals(cas).real, 6)
ok = len(cross) == 18 and len(diagb) == 12 and sorted(cev.tolist()) == [0.0] * 2 + [2.0] * 6 + [6.0] * 10
report("7b", ok, "so(6,C)=lG^(1)+lG^(2)+(C^3(x)C^3): dims 6+6+18; diagonal-rotation Casimir on the cross part 0,2,6 on 2,6,10")

com_lG = solve_space(12, [c_antisym_eta(eta2)] + [c_comm(X) for X in Ldiag + Kdiag])
com_lG1 = solve_space(6, [c_antisym_eta(eta)] + [c_comm(X) for X in Lv + Kv])
ok = len(com_lG) == 2 and same_span(com_lG, [Cv, epG]) and len(com_lG1) == 0
ok &= close(Cv @ Cv, -np.eye(12)) and close(epG @ epG, np.eye(12))
Y = RNG.normal(size=12)
z1, z2 = to_z12(Y)
w1, w2 = to_z12(epG @ Y)
ok &= np.allclose(w1, -1j * z2) and np.allclose(w2, 1j * z1)
Yt = expm(0.7 * epG) @ Y
za, zb = to_z12(Yt)
ok &= np.isclose(q(za) + q(zb), q(z1) + q(z2)) and not np.isclose(np.linalg.norm(Yt), np.linalg.norm(Y))
report("7c", ok, "commutant of diagonal lG in so(6,6) = span{C_v, eps(x)G} (0 at n=1); C_v^2=-1, (eps(x)G)^2=+1; eps(x)G acts "
       "on (z1,z2) as [[0,-i],[i,0]], preserves q_2, not E6+E6")

lint_sp1 = [0.5 * e[a] @ e[b] for a, b in itertools.combinations((0, 3, 4, 5), 2)]
lint1 = [np.kron(np.diag([1, 0]), vecrep(x, e)) for x in lint_sp1]
coef = null_space(np.array([comm(X, G2).ravel() for X in lint1]).T)
lG2 = [np.kron(np.diag([0, 1]), x) for x in Lv + Kv]
both = solve_space(12, [c_antisym_eta(eta2), c_comm(G2)] + [c_comm(X) for X in lint1])
ok = coef.shape[1] == 0 and len(both) == 6 and same_span(both, lG2)
report("7d", ok, "Cent(G_2) meets lint^(1) in 0; so(6,6) cap Cent(G_2) cap Cent(lint^(1)) = lG^(2) (dim 6)")

# ---------------------------------------------------------------------------------------------
# 8  duplicated momentum fields in the closure (prop:balance, ES17 prop:several)
# ---------------------------------------------------------------------------------------------
def wofp(p):
    return q(zof(eta @ p))


ok = True
nu1, nu2, mm = 0.7, -1.3, 1.9
Nl = (nu1 * eta + nu2 * G @ eta) / mm
for _ in range(10):
    pi0 = RNG.normal(size=6)
    alpha, beta = RNG.normal() + 1j * RNG.normal(), RNG.normal() + 1j * RNG.normal()
    act = lambda c, v: c.real * v + c.imag * (G @ v)
    p1, p2 = eta @ act(alpha, pi0), eta @ act(beta, pi0)
    ok &= np.isclose(wofp(p1) + wofp(p2), (alpha**2 + beta**2) * q(zof(pi0)))
    pa = RNG.normal(size=6)
    pb = -G @ pa
    ok &= close(eta @ pb, G @ (eta @ pa)) and np.isclose(wofp(pb), -wofp(pa))
    # torque of one rotor on both constituents along p_k(theta) = exp(-theta G) p_k0
    for th in np.linspace(0, 3, 7):
        Rt = expm(-th * G)
        torque = sum((Rt @ pk) @ G @ Nl @ (Rt @ pk) for pk in (pa, pb))
        ok &= abs(torque) < 1e-9
    torque1 = [(expm(-th * G) @ pa) @ G @ Nl @ (expm(-th * G) @ pa) for th in np.linspace(0, 3, 7)]
    ok &= max(abs(np.array(torque1))) > 1e-3
    half = [act(1 / s2 + 0j, pi0), act(1j / s2, pi0)]
    ok &= all(np.isclose(abs(q(zof(h))) / 2, 0.5 * abs(q(zof(pi0))) / 2) for h in half)
report("8a", ok, "w1+w2=(a^2+b^2)w(pi); p^(2)=-Gp^(1) <=> pi^(2)=G pi^(1), w2=-w1; the one-rotor torque vanishes for all "
       "theta on the balance (not for one non-null constituent); equal split: E0(pi^(k))=E0(pi)/2")

# ---------------------------------------------------------------------------------------------
# 9  the five Lorentz actions at n=2 (tab:actions), mixed rung, Theta_F
# ---------------------------------------------------------------------------------------------
def gens_copy(k, kind):
    o = 6 * k
    P = lambda a, b: E[o + a] @ E[o + b]
    if kind == 'I':
        R = [0.5 * P(4, 5), 0.5 * P(5, 3), 0.5 * P(3, 4)]
        B = [0.5 * P(0, 3), 0.5 * P(0, 4), 0.5 * P(0, 5)]
    elif kind == 'B':
        R = [0.5 * (P(4, 5) - P(1, 2)), 0.5 * (P(5, 3) - P(2, 0)), 0.5 * (P(3, 4) - P(0, 1))]
        B = [0.5 * (P(2, 4) - P(1, 5)), 0.5 * (P(0, 5) - P(2, 3)), 0.5 * (P(1, 3) - P(0, 4))]
    else:
        R = [np.zeros((64, 64))] * 3
        B = [np.zeros((64, 64))] * 3
    return R, B


def struct(R, B):
    g = R + B
    basis = np.array([x.ravel() for x in g]).T
    out = []
    for x in g:
        for y in g:
            out.append(np.linalg.lstsq(basis, comm(x, y).ravel(), rcond=None)[0])
    return np.array(out)


RI, BI = gens_copy(0, 'I')
RB, BB = gens_copy(0, 'B')
same_struct = close(struct(RI, BI), struct(RB, BB))
actions = {'spectator (1,0,1)': ('I', 'S'), 'binding+spectator (0,1,1)': ('B', 'S'), 'diagonal interior (2,0,0)': ('I', 'I'),
           'mixed (1,1,0)': ('I', 'B'), 'binding on both (0,2,0)': ('B', 'B')}
expect = {'spectator (1,0,1)': (28, 13, {3: 4, 0: 8}, -1),
          'binding+spectator (0,1,1)': (15, 6, {4: 6, 0: 6}, +1),
          'diagonal interior (2,0,0)': (7, 7, {3: 8, 0: 4}, +1),
          'mixed (1,1,0)': (1, 1, {3: 4, 4: 6, 0: 2}, -1),
          'binding on both (0,2,0)': (2, 1, {4: 12}, +1)}
res_dims, res_cont, res_spin, closes = {}, {}, {}, True
ACT = {}
for name, (k1, k2) in actions.items():
    R1, B1 = gens_copy(0, k1)
    R2, B2 = gens_copy(1, k2)
    R = [R1[i] + R2[i] for i in range(3)]
    B = [B1[i] + B2[i] for i in range(3)]
    ACT[name] = (R, B)
    gsp = R + B
    basis = np.array([x.ravel() for x in gsp]).T
    for x in gsp:
        for y in gsp:
            cc = comm(x, y).ravel()
            closes &= np.allclose(basis @ np.linalg.lstsq(basis, cc, rcond=None)[0], cc, atol=1e-9)
    vr = [vecrep(x, E) for x in gsp]
    cen = solve_space(12, [c_antisym_eta(eta2)] + [c_comm(X) for X in vr])
    cenc = solve_space(12, [c_antisym_eta(eta2), c_antisym()] + [c_comm(X) for X in vr])
    res_dims[name] = (len(cen), len(cenc))
    Cv12 = -sum(x @ x for x in vr[:3]) + sum(x @ x for x in vr[3:])
    evs = np.round(np.linalg.eigvals(Cv12).real, 6)
    res_cont[name] = {int(round(v)): int((np.abs(evs - v) < 1e-6).sum()) for v in set(evs.tolist())}
    U2pi = expm(2 * np.pi * R[2])
    res_spin[name] = +1 if close(U2pi, np.eye(64), 1e-8) else (-1 if close(U2pi, -np.eye(64), 1e-8) else 0)
ok = same_struct and closes and all(res_dims[nm] == expect[nm][:2] for nm in expect)
report("9a", ok, "tab:actions: lint and lG spinor bases have equal structure constants; centralisers in so(6,6) of "
       "spectator/binding+spectator/diagonal/mixed/both: dims " + str([res_dims[nm][0] for nm in expect]) +
       ", compact " + str([res_dims[nm][1] for nm in expect]))
ok = all(res_cont[nm] == expect[nm][2] for nm in expect)
report("9b", ok, "tab:actions content of R^{6,6} by Lorentz Casimir (3=V, 4=Lambda^2V, 0=trivial): " +
       "; ".join(f"{nm.split()[0]} {res_cont[nm]}" for nm in expect))
ok = all(res_spin[nm] == expect[nm][3] for nm in expect)
report("9c", ok, "tab:actions/prop:spin(ii): exp(2pi R_3) on S_2 = " + str([res_spin[nm] for nm in expect]) +
       " (half-integer, integer, integer, half-integer, integer)")

Rm, Bm_ = ACT['mixed (1,1,0)']
Cmix = -sum(x @ x for x in Rm) + sum(x @ x for x in Bm_)
ok = close((Cmix - 1.5 * np.eye(64)) @ (Cmix - 5.5 * np.eye(64)), 0)
ok &= np.linalg.matrix_rank(Cmix - 5.5 * np.eye(64), tol=1e-8) == 16 and np.linalg.matrix_rank(Cmix - 1.5 * np.eye(64), tol=1e-8) == 48
report("9d", ok, "mixed rung: (C_mix-3/2)(C_mix-11/2)=0 with the Dirac value on 16 and 11/2 on 48 dimensions")

tri = list(itertools.combinations(range(12), 3))
tri_mats = [E[i] @ E[j] @ E[k] for (i, j, k) in tri]


def invariant_dim(idxs):
    cols = []
    for t in idxs:
        Bt = tri_mats[t]
        cols.append(np.concatenate([comm(g, Bt).ravel() for g in Rm + Bm_]))
    return null_space(np.array(cols).T, rcond=1e-9).shape[1]


Eone = {0, 3, 4, 5}
sel_mix = [n for n, t in enumerate(tri) if sum(1 for x in t if x in Eone) == 2 and sum(1 for x in t if x >= 6) == 1]
sel_c2 = [n for n, t in enumerate(tri) if all(x >= 6 for x in t)]
d_all, d_mix, d_c2 = invariant_dim(range(len(tri))), invariant_dim(sel_mix), invariant_dim(sel_c2)
g1 = {0: E[0], 1: E[3], 2: E[4], 3: E[5]}
ep2 = [E[6 + a] for a in range(6)]
ThF = (g1[2] @ g1[3] @ ep2[0] + g1[3] @ g1[1] @ ep2[1] + g1[1] @ g1[2] @ ep2[2]
       - g1[0] @ g1[1] @ ep2[3] - g1[0] @ g1[2] @ ep2[4] - g1[0] @ g1[3] @ ep2[5])
Gep2 = [sum(G[b, a] * ep2[b] for b in range(6)) for a in range(6)]       # e'_a -> G e'_a
ThG = (g1[2] @ g1[3] @ Gep2[0] + g1[3] @ g1[1] @ Gep2[1] + g1[1] @ g1[2] @ Gep2[2]
       - g1[0] @ g1[1] @ Gep2[3] - g1[0] @ g1[2] @ Gep2[4] - g1[0] @ g1[3] @ Gep2[5])
ok = d_all == 4 and d_mix == 2 and d_c2 == 2
ok &= all(close(comm(g, ThF), 0) and close(comm(g, ThG), 0) for g in Rm + Bm_)
ok &= np.linalg.matrix_rank(np.array([ThF.ravel(), ThG.ravel()])) == 2
report("9e", ok, f"grade-three invariants of the mixed action: {d_all} in all, {d_mix} in Lambda^2(E_1)(x)V^(2) "
       f"(Theta_F and Theta_F followed by G, both invariant), {d_c2} in Lambda^3 V^(2)")

# ---------------------------------------------------------------------------------------------
# 10  spin and statistics (prop:spin)
# ---------------------------------------------------------------------------------------------
evM = np.linalg.eigvals(M12)
evC = np.linalg.eigvals(Cb)
ok = np.allclose(evM.real, 0, atol=1e-9) and np.allclose(evC.real, 0, atol=1e-9)
setM = sorted(set(np.round(evM.imag, 6).tolist()))
setC = sorted(set(np.round(evC.imag, 6).tolist()))
ok &= setM == [-2.0, -1.0, 0.0, 1.0, 2.0] and setC == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
mtermsM = [0.5 * eta[m, m] * E[m] @ E[6 + m] for m in (0, 3, 4, 5)]
ok &= all(close(t @ t, -0.25 * np.eye(64)) for t in mtermsM) and all(close(comm(s, t), 0) for s in mtermsM for t in mtermsM)
ok &= close(expm(2 * np.pi * M12), np.eye(64), 1e-8) and close(expm(2 * np.pi * Cb), np.eye(64), 1e-8)
ok &= close(expm(np.pi * E[1] @ E[2]), -np.eye(64), 1e-8) and close(expm(np.pi * E[6] @ E[7]), -np.eye(64), 1e-8)
report("10a", ok, "M_12 and C_b have eigenvalues i{-2..2}, i{-3..3}; exp(2pi M_12)=exp(2pi C_b)=1; surplus exp(pi e_s e_t)=-1")

Rd, Bd = ACT['diagonal interior (2,0,0)']
Rs, Bs = ACT['spectator (1,0,1)']
spinC = -sum(x @ x for x in Rd)
ev = np.round(np.linalg.eigvalsh(spinC), 6)
ok = int((np.abs(ev) < 1e-6).sum()) == 16 and int((np.abs(ev - 2) < 1e-6).sum()) == 48
ok &= close(-sum(x @ x for x in Rs), 0.75 * np.eye(64))
report("10b", ok, "spin Casimir: 0 on 16 and 2 on 48 (diagonal interior), 3/4 on all of S_2 (spectator)")

CD = -sum(x @ x for x in Rd) + sum(x @ x for x in Bd)
Ccop = -M12 @ M12
evD = np.round(np.linalg.eigvalsh(CD), 6)
evc = np.round(np.linalg.eigvalsh(Ccop), 6)
cntD = {v: int((np.abs(evD - v) < 1e-6).sum()) for v in (0, 3, 4)}
cntc = {v: int((np.abs(evc - v) < 1e-6).sum()) for v in (0, 1, 4)}
ok = cntD == {0: 8, 3: 32, 4: 24} and cntc == {0: 24, 1: 32, 4: 8} and close(CD + Ccop, 4 * np.eye(64))
ok &= all(close(comm(tauK, x), 0) for x in Rd + Bd) and close(acomm(tauK, M12), 0)
wt, vt = np.linalg.eigh(tauK.astype(float))
for sgn in (+1, -1):
    Qs = vt[:, np.abs(wt - sgn) < 1e-9]
    evs = np.round(np.linalg.eigvalsh(Qs.T @ CD @ Qs), 6)
    ok &= {v: int((np.abs(evs - v) < 1e-6).sum()) for v in (0, 3, 4)} == {0: 4, 3: 16, 4: 12}
report("10c", ok, "diagonal action: C_Delta 0,3,4 on 8,32,24; C_cop 0,1,4 on 24,32,8; C_cop+C_Delta=4; tau_K commutes with "
       "the diagonal Lorentz action, anticommutes with M_12; each tau_K sector carries 0,3,4 on 4,16,12")

# ---------------------------------------------------------------------------------------------
# 11  vacuum and imaginary unit (prop:vacuum)
# ---------------------------------------------------------------------------------------------
lint8 = [0.5 * e[a] @ e[b] for a, b in itertools.combinations((0, 3, 4, 5), 2)]
Q1 = 0.5 * e[1] @ e[2]
com8 = solve_space(8, [c_comm(X) for X in lint8 + [Q1]])
sym8 = solve_space(8, [c_comm(X) for X in lint8 + [Q1]] + [lambda X: X - X.T])
ok = len(com8) == 4 and len(sym8) == 2 and same_span(sym8, [np.eye(8), w8])
xs, ys = sp.symbols('x y', real=True)
Wsym = sp.Matrix(w8)
Pm = xs * sp.eye(8) + ys * Wsym
sols = sp.solve(list(set((Pm * Pm - Pm).values())), [xs, ys], dict=True)
rank4 = [s_ for s_ in sols if sp.simplify((Pm.subs(s_)).trace()) == 4]
ok &= sorted([(s_[xs], s_[ys]) for s_ in rank4]) == [(sp.Rational(1, 2), -sp.Rational(1, 2)), (sp.Rational(1, 2), sp.Rational(1, 2))]
ok &= all((comm(e[a] @ e[b], w8) == 0).all() for a in range(6) for b in range(6))
report("11a", ok, "n=1: commutant of lint and Q=e2e3/2 in M_8 has dim 4, symmetric part span{1, V}; rank-4 projectors "
       "(1 pm V)/2; every bivector commutes with V (rigid)")

a_, b_, c_, d_, e_ = E[1], E[2], E[6], E[7], E[8]
gam = [E[0], E[3], E[4], E[5]]
internal = [1, 2, 6, 7, 8, 9, 10, 11]
Jint = prodm([E[i] for i in internal], 64)
H = a_ @ b_ @ c_ @ d_
Pi0 = (I64 + H) / 2
Qv = 0.5 * a_ @ b_
pi_ = comm(0.5 * b_ @ e_, Pi0)
ok = (H @ H == I64).all() and (H.T == H).all() and np.trace(H) == 0
ok &= close(comm(Qv, Pi0), 0) and all(close(comm(g, Pi0), 0) for g in gam) and close(comm(Jint, Pi0), 0)
ok &= close(pi_, -0.5 * a_ @ c_ @ d_ @ e_) and close(pi_, pi_.T) and np.isclose(np.trace(pi_ @ pi_), 16)
ok &= close(comm(Qv, pi_), 0.5 * b_ @ c_ @ d_ @ e_)
report("11b", ok, "H=abcd: H^2=1, H^T=H, Tr H=0; Pi_0 commutes with Q, the four gammas and the internal volume; "
       "[be/2,Pi_0]=-acde/2 symmetric, Tr pi^2=16, [Q,pi]=bcde/2")

I5 = a_ @ b_ @ c_ @ d_ @ e_
vs = [a_, b_, c_, d_, e_]
Z = [I5 @ v for v in vs]
ok = all((Zi.T == Zi).all() for Zi in Z)
ok &= all((acomm(Z[i], Z[j]) == 2 * (i == j) * I64).all() for i in range(5) for j in range(5))
for nvec in [np.array([2, 3, 6, 0, 0]) / 7] + [x / np.linalg.norm(x) for x in RNG.normal(size=(3, 5))]:
    Pn = (np.eye(64) + sum(nvec[i] * Z[i] for i in range(5))) / 2
    ok &= close(Pn @ Pn, Pn) and close(Pn, Pn.T) and np.isclose(np.trace(Pn), 32) and all(close(comm(g, Pn), 0) for g in gam)
Tt_ = [comm(0.5 * v @ e_, Pi0) for v in vs[:4]]
ok &= all(close(Tt_[i], 0.5 * Z[i]) for i in range(4))
ok &= all(np.isclose(np.trace(Tt_[i] @ Tt_[j]), 16 * (i == j)) for i in range(4) for j in range(4))
ok &= close(comm(Qv, Tt_[0]), -Tt_[1]) and close(comm(Qv, Tt_[1]), Tt_[0]) and close(comm(Qv, Tt_[2]), 0) and close(comm(Qv, Tt_[3]), 0)
report("11c", ok, "Z_i=I5 v_i symmetric, {Z_i,Z_j}=2 delta; Pi(n) rank-32 projector commuting with the gammas (n=(2,3,6,0,0)/7 "
       "and random); T_i=[v_i e/2,Pi_0]=I5 v_i/2, Tr T_iT_j=16 delta, [Q,T_a]=-T_b, [Q,T_b]=T_a, [Q,T_c]=[Q,T_d]=0")

four = list(itertools.combinations(range(5), 4))
commuting4 = [t for t in four if close(comm(Qv, prodm([vs[i] for i in t], 64)), 0)]
ok = commuting4 == [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4)]
report("11d", ok, "grade-four blades in a..e commuting with Q are exactly abcd, abce, abde (all straddle the copies)")

Om = E[0] @ E[3] @ E[4] @ E[5]
ok = (Jint == np.kron(e[1] @ e[2], w8)).all() and (Jint @ Jint == -I64).all()
gens_iv = [0.5 * E[x] @ E[y] for x, y in itertools.combinations((0, 3, 4, 5), 2)] + \
          [0.5 * E[x] @ E[y] for x, y in itertools.combinations(internal, 2)]
cmask = np.ones(4096, bool)
for g in gens_iv:
    cmask &= blade_commutes(2 * g, +1)
cset = [blades[k].astype(int) for k in np.nonzero(cmask)[0]]
ok &= len(gens_iv) == 34 and len(cset) == 4 and same_span(cset, [I64, Om, Jint, Om @ Jint])
cs = sp.symbols('c0:4', real=True)
basis4 = [I64, Om, Jint, Om @ Jint]
bm = np.array([x.ravel() for x in basis4]).T
table = [[np.linalg.lstsq(bm, (x @ y).ravel(), rcond=None)[0] for y in basis4] for x in basis4]
sq = [0] * 4
for i in range(4):
    for j in range(4):
        for k in range(4):
            sq[k] += cs[i] * cs[j] * sp.nsimplify(round(table[i][j][k]))
eqs = [sq[0] + 1] + sq[1:]
sol = sp.solve(eqs, cs, dict=True)
sol_m = [sum(float(s_[cs[k]]) * basis4[k] for k in range(4)) for s_ in sol]
ok &= len(sol) == 4 and all(any(close(m, t) for m in sol_m) for t in (Om, -Om, Jint, -Jint))
ok &= [all(close(comm(m, g), 0) for g in gam) for m in (Om, Jint)] == [False, True]
report("11e", ok, "product of the 8 internal generators J=(e2e3)(x)omega, J^2=-1; joint commutant of lint^(1) and the 28 "
       "internal bivectors = span{1,Omega,J,Omega J}; x^2=-1 there exactly at +-Omega, +-J; only +-J commute with the gammas")

# ---------------------------------------------------------------------------------------------
# 12  the inter-copy b-field (thm:bfield2)
# ---------------------------------------------------------------------------------------------
def Smat(k1, k2, bt):
    return np.kron(np.diag([k1, k2]), G) + bt * Cv


def Mmat(k1, k2, bt, m=1.0):
    S = Smat(k1, k2, bt)
    Sd = eta2 @ S.T @ eta2
    return np.block([[-S, np.eye(12) / m], [np.zeros((12, 12)), Sd]])


k1s, k2s = sp.symbols('kappa1 kappa2', nonnegative=True)
bs = sp.symbols('beta', real=True)
ms, lam = sp.symbols('m lambda')
k2p = sp.symbols('k2p', positive=True)
Kz = sp.diag(k1s, k2s)
Ssym = sp.kronecker_product(Kz, sp.Matrix(G)) + bs * sp.kronecker_product(sp.Matrix(ep), sp.eye(6))
Sdag = sp.Matrix(eta2) * Ssym.T * sp.Matrix(eta2)
ok = sp.simplify(Sdag - (sp.kronecker_product(Kz, sp.Matrix(G)) - bs * sp.kronecker_product(sp.Matrix(ep), sp.eye(6)))) == sp.zeros(12)
report("12a", ok, "thm:bfield2(i): S^dag = K(x)G - b (symbolic); uniqueness of b = R C_v is x5-4a")

blocks = [[j, j + 3, 6 + j, 9 + j] for j in range(3)]
pS = sum(blocks, [])
Sp_ = Ssym.extract(pS, pS)
blockdiag = all(Sp_[4 * r:4 * r + 4, 4 * c:4 * c + 4] == sp.zeros(4) for r in range(3) for c in range(3) if r != c)
S4 = Ssym.extract(blocks[0], blocks[0])
blocks_equal = all(Ssym.extract(blocks[j], blocks[j]) == S4 for j in range(3))
p4 = sp.expand((lam * sp.eye(4) - S4).det())
pform = sp.expand(lam**4 + (k1s**2 + k2s**2 + 2 * bs**2) * lam**2 + (k1s * k2s - bs**2)**2)
ok = blockdiag and blocks_equal and sp.simplify(p4 - pform) == 0
Hm2 = sp.Matrix([[k1s, sp.I * bs], [-sp.I * bs, k2s]])
wp = (k1s + k2s) / 2 + sp.sqrt((k1s - k2s)**2 / 4 + bs**2)
wm = (k1s + k2s) / 2 - sp.sqrt((k1s - k2s)**2 / 4 + bs**2)
ok &= sp.simplify(sp.expand((lam - wp) * (lam - wm)) - sp.expand(Hm2.charpoly(lam).as_expr())) == 0
ok &= sp.simplify(sp.expand(pform.subs(lam, sp.I * wp))) == 0 and sp.simplify(sp.expand(pform.subs(lam, sp.I * wm))) == 0
for (x, y, z) in [(1, 3, 0.4), (0, 5, 0.5), (0.3, 0.3, 1.2)]:
    evS = np.sort(np.linalg.eigvals(Smat(x, y, z)).imag)
    wpn = (x + y) / 2 + np.sqrt((x - y)**2 / 4 + z**2)
    wmn = (x + y) / 2 - np.sqrt((x - y)**2 / 4 + z**2)
    ok &= np.allclose(evS, np.sort([wpn] * 3 + [-wpn] * 3 + [wmn] * 3 + [-wmn] * 3), atol=1e-9)
report("12b", ok, "thm:bfield2(ii): S is three identical 4x4 blocks; det(lambda-S)=(lambda^4+(k1^2+k2^2+2b^2)lambda^2+"
       "(k1k2-b^2)^2)^3; spec S={+-i w_pm}, w_pm = eigenvalues of [[k1,ib],[-ib,k2]]")

Kg = sp.kronecker_product(Kz, sp.Matrix(G))
bC = bs * sp.kronecker_product(sp.Matrix(ep), sp.eye(6))
ok = sp.simplify(Kg * bC - bC * Kg - bs * (k2s - k1s) * sp.kronecker_product(sp.Matrix(sx), sp.Matrix(G))) == sp.zeros(12)
cd = []
for (x, y) in [(1, 1), (1, 2), (0, 2), (2, 0)]:
    KG = np.kron(np.diag([x, y]), G)
    cd.append(len(solve_space(12, [c_antisym_eta(eta2), c_comm(KG)])))
ok &= cd == [30, 12, 21, 21]
# removability in the sense of d4:prop:remove: [b, K(x)G]=0  <=>  kappa1=kappa2 or beta=0
for (x, y, z) in [(1, 1, 0.3), (1, 2, 0.0), (1, 2, 0.3), (0, 2, 0.3), (2, 2, 0.0)]:
    ok &= close(comm(np.kron(np.diag([x, y]), G), z * Cv), 0) == (x == y or z == 0)
report("12c", ok, f"thm:bfield2(iii): [K(x)G,b]=b(k2-k1) sx(x)G; centraliser of K(x)G in so(6,6): {cd} for "
       "(k1,k2)=(1,1),(1,2),(0,2),(2,0); non-removable iff k1!=k2 and beta!=0")

M8 = sp.zeros(8)
S4d = sp.diag(1, -1, 1, -1) * S4.T * sp.diag(1, -1, 1, -1)
M8[:4, :4] = -S4
M8[:4, 4:] = sp.eye(4) / ms
M8[4:, 4:] = S4d
pM = sp.expand(M8**4 + (k1s**2 + k2s**2 + 2 * bs**2) * M8**2 + (k1s * k2s - bs**2)**2 * sp.eye(8))
ok = pM.applyfunc(sp.simplify) == sp.zeros(8)
Mn = Mmat(0.7, 1.9, 0.45, 1.3)
wpn, wmn = (0.7 + 1.9) / 2 + np.sqrt(0.36 + 0.45**2), (0.7 + 1.9) / 2 - np.sqrt(0.36 + 0.45**2)
ok &= close((Mn @ Mn + wpn**2 * np.eye(24)) @ (Mn @ Mn + wmn**2 * np.eye(24)), 0, 1e-9)
report("12d", ok, "thm:bfield2(iv): (M^2+w_+^2)(M^2+w_-^2)=0 identically (symbolic on the 8x8 block, numeric on 24x24)")


def exact_bounded(k1, k2, bt, full=False):
    k1, k2, bt = sp.nsimplify(k1), sp.nsimplify(k2), sp.nsimplify(bt)
    Ssub = (Ssym if full else S4).subs({k1s: k1, k2s: k2, bs: bt})
    n = Ssub.shape[0]
    etan = sp.Matrix(eta2) if full else sp.diag(1, -1, 1, -1)
    Mx = sp.zeros(2 * n)
    Mx[:n, :n] = -Ssub
    Mx[:n, n:] = sp.eye(n)
    Mx[n:, n:] = etan * Ssub.T * etan
    pol = sp.Poly(pform.subs({k1s: k1, k2s: k2, bs: bt}), lam)
    sqf = sp.Poly(sp.sqf_part(pol.as_expr()), lam)
    acc = sp.zeros(2 * n)
    for cf in sqf.all_coeffs():
        acc = acc * Mx + cf * sp.eye(2 * n)
    return acc == sp.zeros(2 * n)


def criterion(k1, k2, bt):
    return (k1 == k2 and k1 > 0) or (k1 != k2 and k1 * k2 != bt**2)


R = sp.Rational
bounded_pts = [(1, 3, R(2, 5)), (0, 5, R(1, 2)), (2, 2, R(1, 3)), (1, 1, 1), (2, 2, 0), (0, R(1, 100), 1)]
unbounded_pts = [(0, 0, R(1, 2)), (1, 4, 2), (1, 9, 3), (0, 3, 0)]
ok = all(exact_bounded(*p) for p in bounded_pts) and not any(exact_bounded(*p) for p in unbounded_pts)
ok &= exact_bounded(1, 3, R(2, 5), full=True) and not exact_bounded(1, 4, 2, full=True)
agree = True
pool = [0, R(1, 2), 1, 2, 3, 4]
rpts = [(x, y, z) for x in pool for y in pool for z in [0, R(1, 2), 1, 2]]
sel = RNG.choice(len(rpts), size=40, replace=False)
for s_ in sel:
    x, y, z = rpts[s_]
    agree &= exact_bounded(x, y, z) == criterion(x, y, z)
for (x, y, z) in [(2, 2, 2), (3, 3, 3), (0, 0, 0), (R(1, 2), 2, 1), (0, 2, 0)]:
    agree &= exact_bounded(x, y, z) == criterion(x, y, z)
# the Jordan mechanism: at k1!=k2, k1k2=b^2 ker S is not eta_2-isotropic; at k1=k2=b it is
def kerS_isotropic(k1, k2, bt):
    ns = null_space(Smat(k1, k2, bt))
    return ns.shape[1] > 0 and close(ns.T @ eta2 @ ns, 0, 1e-9)


agree &= (not kerS_isotropic(1, 4, 2)) and (not kerS_isotropic(0, 3, 0)) and kerS_isotropic(2, 2, 2)
report("12e", ok and agree, "thm:bfield2(v): exact semisimplicity (square-free char. poly over QQ) bounded at the six listed "
       "points, unbounded at the four; criterion agrees at 45 further rational points; Jordan block at k1!=k2, k1k2=b^2 "
       "because ker S is not eta_2-isotropic")

wm0 = wm.subs(k1s, 0)
wp0 = wp.subs(k1s, 0)
ok = sp.simplify(wp0 * wm0 + bs**2) == 0 and sp.simplify(wp0 + wm0 - k2s) == 0
ser = sp.series(-wm0.subs(k2s, k2p), bs, 0, 6).removeO()
ok &= sp.simplify(sp.expand(ser) - (bs**2 / k2p - bs**4 / k2p**3)) == 0
lim = [sp.limit(wp0 / bs, bs, sp.oo), sp.limit(wm0 / bs, bs, sp.oo)]
ok &= lim == [1, -1]
report("12f", ok, "thm:bfield2(vi): k1=0: w+w-=-b^2, w++w-=k2; |w-|=b^2/k2-b^4/k2^3+...; w_pm/b -> +-1 as b>>k2")


def block8(k1, k2, bt, m=1.0):
    S = np.array(S4.subs({k1s: k1, k2s: k2, bs: bt}).tolist(), dtype=float)
    et4 = np.diag([1, -1, 1, -1])
    return np.block([[-S, np.eye(4) / m], [np.zeros((4, 4)), et4 @ S.T @ et4]])


def sup_norm(k1, k2, bt, m, Tmax, n=40000):
    Mb = block8(k1, k2, bt, m)
    lamb, V = np.linalg.eig(Mb)
    Vi = np.linalg.inv(V)
    def nrm(t):
        return np.linalg.norm((V * np.exp(lamb * t)) @ Vi, 2)
    ts = np.linspace(0, Tmax, n)
    Et = (V[None, :, :] * np.exp(np.outer(ts, lamb))[:, None, :]) @ Vi[None, :, :]
    vals = np.linalg.norm(Et, ord=2, axis=(1, 2))
    i = int(np.argmax(vals))
    lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, n - 1)]
    r = minimize_scalar(lambda t: -nrm(t), bounds=(lo, hi), method='bounded', options={'xatol': 1e-10})
    return max(vals[i], -r.fun)


ok = close(np.linalg.norm(expm(Mmat(0, 5, 0.5) * 37.3), 2), np.linalg.norm(expm(block8(0, 5, 0.5) * 37.3), 2), 1e-8)
wmn = 2.5 - np.sqrt(6.5)
s1 = sup_norm(0, 5, 0.5, 1.0, 600, 60001)
ok &= round(wmn, 5) == -0.04951 and round(1 / abs(wmn), 1) == 20.2 and round(s1, 2) == 19.86
wp_, wm_ = 0.005 + np.sqrt(0.005**2 + 1), 0.005 - np.sqrt(0.005**2 + 1)
s2_ = sup_norm(0, 0.01, 1.0, 1.0, 2 * np.pi / 0.01, 80001)
ok &= round(wp_, 5) == 1.00501 and round(wm_, 5) == -0.99501 and round(s2_, 1) == 200.0
scal = [sup_norm(0, k2, 1.0, m, 2 * np.pi / k2, 40001) * m * k2 / 2 for (k2, m) in [(0.02, 1.0), (0.05, 1.0), (0.01, 2.0)]]
ok &= all(abs(x - 1) < 0.03 for x in scal)
report("12g", ok, f"numbers: w_-(0,5,1/2)={wmn:.5f}, sup_(t<=600)||e^Mt||={s1:.2f} vs 1/|w_-|={1/abs(wmn):.1f}; "
       f"(0,1/100,1): lines {wp_:.5f}, {wm_:.5f}, sup over a beat period {s2_:.1f}=2/(m k2); "
       f"sup*m*k2/2 = {[round(float(x), 3) for x in scal]} at (k2,m)=(0.02,1),(0.05,1),(0.01,2)")

# ---------------------------------------------------------------------------------------------
# 13  context for rem:dealbreaker: what one manifold already does (d4 conventions)
# ---------------------------------------------------------------------------------------------
def cross_mat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


av, cvv, kap = np.array([0, 0, 1]), np.array([1, 0, 0]), 1
b1 = np.block([[cross_mat(av + cvv), np.zeros((3, 3))], [np.zeros((3, 3)), cross_mat(av - cvv)]])
S1 = sp.Matrix(b1 + kap * G)
M1 = sp.zeros(12)
M1[:6, :6] = -S1
M1[:6, 6:] = sp.eye(6)
M1[6:, 6:] = sp.Matrix(eta) * S1.T * sp.Matrix(eta)
chi = sp.Poly(M1.charpoly(lam).as_expr(), lam)
sqf = sp.Poly(sp.sqf_part(chi.as_expr()), lam)
acc = sp.zeros(12)
for cf in sqf.all_coeffs():
    acc = acc * M1 + cf * sp.eye(12)
semis = acc == sp.zeros(12)
roots = [complex(r) for r in sp.Poly(chi.as_expr(), lam).nroots()]
imag_only = all(abs(r.real) < 1e-9 for r in roots)
spec_changed = any(abs(abs(r.imag) - kap) > 1e-3 for r in roots)
nonremovable = not close(comm(b1, G), 0)
free_n2 = not exact_bounded(0, 0, R(3, 7)) and not exact_bounded(0, 0, 2)
report("13a", semis and imag_only and spec_changed and nonremovable and free_n2,
       "context for rem:dealbreaker: already at n=1 the admissible b=diag(A+C,A-C), a=e3, c=e1, kappa=1 is bounded "
       "(exact), changes the spectrum and has [b,G]!=0 (not removable, d4:thm:screen); at n=2 b alone still does not "
       "bind (k1=k2=0 unbounded)")

print(f"# x5_check: {'all checks passed' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}; {time.time() - T0:.1f} s")
sys.exit(0 if not FAILS else 1)
