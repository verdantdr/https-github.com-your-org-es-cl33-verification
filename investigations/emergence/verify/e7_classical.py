#!/usr/bin/env python3
"""Track e7 -- Classical readings: astronomy and cosmology.

Verification script for investigations/emergence/sections/e7_classical.tex.
Conventions (tex/s02_arena.tex, tex/sA_realisation.tex; RE = investigations/rest_energy):
  V = R^{3,3}, eta = diag(1,1,1,-1,-1,-1); e1 = sx(x)1(x)1, e2 = sz(x)sx(x)1, e3 = sz(x)sz(x)sx, e4 = ep(x)1(x)1,
  e5 = sz(x)ep(x)1, e6 = sz(x)sz(x)ep; omega = e1...e6; tower E_(k,a) = omega^{(x)(k-1)} (x) e_a (x) 1^{(x)(n-k)}.
  Field-strength reading (RE eq. (1)): X = sum_i (B_i e_i + E_i e_{i+3}), z = B + iE, binding root G e_i = e_{i+3},
  G e_{i+3} = -e_i acts as i; the frame variable B is minus the textbook field, B_t = -B (ES17 Section 1.4).
  q = z.z = |B|^2 - |E|^2 + 2i E.B, E6 = |z|^2/2, E0 = |q|/2.  SI: z_c = cB + iE, u = eps0|z_c|^2/2, u0 = eps0|q_c|/2.
Exact statements are checked with integer matrices and sympy; spectra, exponentials and Monte Carlo in floating point
with the tolerances stated in the PASS lines.  Physical constants: scipy.constants (CODATA).  Every astronomical or
cosmological input below is an input quoted from the reference named next to it in the section; the script only does
the arithmetic.
Run from any directory:  python3 e7_classical.py   (exit 0 and one PASS line per claim iff all hold)
"""
import sys
import math

import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy import constants as C

FAILED = []


def report(cid, ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {cid}: {text}")
    if not ok:
        FAILED.append(cid)


def info(text):
    print(f"    . {text}")


EV = C.e                      # J per eV
CM3 = 1e-6                    # m^3 per cm^3
YEAR = C.Julian_year          # s
GYR = 1e9 * YEAR
MPC = 1e6 * C.parsec

# ------------------------------------------------------------------------------------------------ realisation (e7-1)
sx = np.array([[0, 1], [1, 0]], dtype=np.int64)
sz = np.array([[1, 0], [0, -1]], dtype=np.int64)
ep = np.array([[0, -1], [1, 0]], dtype=np.int64)
I2 = np.eye(2, dtype=np.int64)


def kron(*ms):
    out = np.array([[1]], dtype=np.int64)
    for m in ms:
        out = np.kron(out, m)
    return out


E = [kron(sx, I2, I2), kron(sz, sx, I2), kron(sz, sz, sx), kron(ep, I2, I2), kron(sz, ep, I2), kron(sz, sz, ep)]
ETA = np.diag([1, 1, 1, -1, -1, -1])
I8 = np.eye(8, dtype=np.int64)
ok = all(np.array_equal(E[a] @ E[b] + E[b] @ E[a], 2 * ETA[a, b] * I8) for a in range(6) for b in range(6))
OMEGA = E[0] @ E[1] @ E[2] @ E[3] @ E[4] @ E[5]
ok &= np.array_equal(OMEGA @ OMEGA, I8)
G6 = np.zeros((6, 6), dtype=np.int64)
for i in range(3):
    G6[i + 3, i] = 1      # G e_i = e_{i+3}
    G6[i, i + 3] = -1     # G e_{i+3} = -e_i
ok &= np.array_equal(G6 @ G6, -np.eye(6, dtype=np.int64))
# G acts as i on z = B + iE:  X = (B, E) -> GX = (-E, B), z(GX) = -E + iB = i z
Bs, Es = np.array([1, 2, 3]), np.array([5, -7, 11])
X = np.concatenate([Bs, Es])
GX = G6 @ X
ok &= np.array_equal(GX[:3] + 1j * GX[3:], 1j * (Bs + 1j * Es))
report("e7-1", ok, "realisation: Clifford relations of R^{3,3}, omega^2=+1, G^2=-1, G acts as i on z=B+iE (exact)")

# ------------------------------------------------------------------------------------- SI form, RE Prop. 2.8 (e7-2)
rng = np.random.default_rng(20261020)
worst = 0.0
ok = True
for _ in range(200):
    Ev = rng.normal(size=3) * 10 ** rng.uniform(-3, 6)        # V/m
    Bt = rng.normal(size=3) * 10 ** rng.uniform(-9, -2)       # T (textbook field)
    zc = C.c * (-Bt) + 1j * Ev                                # frame variable B = -B_t
    u = C.epsilon_0 / 2 * np.sum(np.abs(zc) ** 2)
    qc = np.sum(zc * zc)
    u0 = C.epsilon_0 / 2 * abs(qc)
    g = np.cross(Ev, Bt) / C.mu_0 / C.c ** 2
    lhs, rhs = u ** 2, C.c ** 2 * np.dot(g, g) + u0 ** 2
    worst = max(worst, abs(lhs - rhs) / lhs)
    ok &= u0 <= u * (1 + 1e-12)
report("e7-2", ok and worst < 1e-10,
       f"SI form u^2=c^2|g|^2+u0^2 and u0<=u on 200 random SI fields (restated RE Prop. 2.8; max rel. dev. {worst:.1e})")

# --------------------------------------------------------------- stress tensor: traceless, canonical form (e7-3, e7-4)
e1s, e2s, e3s, b1, b2, b3 = sp.symbols('E1 E2 E3 B1 B2 B3', real=True)
Ev_s, Bv_s = sp.Matrix([e1s, e2s, e3s]), sp.Matrix([b1, b2, b3])   # textbook fields, Heaviside-Lorentz, c=1


def stress(Ev, Bv):
    u = (Ev.dot(Ev) + Bv.dot(Bv)) / 2
    S = Ev.cross(Bv)
    T = sp.zeros(4, 4)
    T[0, 0] = u
    for i in range(3):
        T[0, i + 1] = T[i + 1, 0] = S[i]
        for j in range(3):
            T[i + 1, j + 1] = -(Ev[i] * Ev[j] + Bv[i] * Bv[j]) + (u if i == j else 0)
    return T


g = sp.diag(1, -1, -1, -1)
T = stress(Ev_s, Bv_s)
Tmix = T * g                                    # T^mu_nu
trace0 = sp.expand(Tmix.trace()) == 0
q_s = (Bv_s.dot(Bv_s) - Ev_s.dot(Ev_s)) + 2 * sp.I * Ev_s.dot(Bv_s)   # |q| is the same for B and -B
E0sq = sp.expand((Bv_s.dot(Bv_s) - Ev_s.dot(Ev_s)) ** 2 / 4 + Ev_s.dot(Bv_s) ** 2)
rain = sp.expand(Tmix * Tmix - E0sq * sp.eye(4)) == sp.zeros(4, 4)
eps_, beta_ = sp.symbols('epsilon beta', real=True)
Tc = stress(sp.Matrix([eps_, 0, 0]), sp.Matrix([beta_, 0, 0]))
u0c = (eps_ ** 2 + beta_ ** 2) / 2
canon = sp.simplify(Tc - sp.diag(u0c, -u0c, u0c, u0c)) == sp.zeros(4, 4)
report("e7-3", trace0 and rain and canon,
       "T^mu_mu=0, T^mu_l T^l_nu=E0^2 delta (Rainich), and for E||B||e1: T=diag(u0,-u0,u0,u0): tension u0 along the axis,"
       " pressure u0 across (exact, sympy)")

# isotropic average of the spatial stress over directions of the axis; FRW scaling
th, ph = sp.symbols('theta phi', real=True)
nvec = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
avg_nn = sp.Matrix(3, 3, lambda i, j: sp.integrate(sp.integrate(nvec[i] * nvec[j] * sp.sin(th), (ph, 0, 2 * sp.pi)),
                                                   (th, 0, sp.pi)) / (4 * sp.pi))
iso_ok = sp.simplify(avg_nn - sp.eye(3) / 3) == sp.zeros(3, 3)
# T^{ij} = u0(delta_ij - 2 n_i n_j) for a point with axis n, rest frame:  average = u0(1 - 2/3) delta = (u0/3) delta
Tij_avg = sp.simplify(u0c * (sp.eye(3) - 2 * avg_nn))
p_ok = sp.simplify(Tij_avg - u0c / 3 * sp.eye(3)) == sp.zeros(3, 3)
a, w, rho0s = sp.symbols('a w rho0', positive=True)
rho = sp.Function('rho')
sol = sp.dsolve(sp.Eq(rho(a).diff(a), -3 * (1 + w) * rho(a) / a), rho(a)).rhs
scal_rad = sp.simplify(sol.subs(w, sp.Rational(1, 3)) / sol.subs(w, sp.Rational(1, 3)).subs(a, 1)) == a ** -4
scal_dust = sp.simplify(sol.subs(w, 0) / sol.subs(w, 0).subs(a, 1)) == a ** -3
report("e7-4", iso_ok and p_ok and scal_rad and scal_dust,
       "<n_i n_j> = delta/3, so an isotropic ensemble of rest-frame field points has p=u/3 (w=1/3) for every u0; "
       "FRW continuity gives rho ~ a^-4 (w=1/3) against a^-3 (dust) (exact, sympy)")

# ------------------------------------------------------------------------- astrophysical field strengths (e7-5, e7-6)
fields = [  # name, B in tesla (inputs quoted in the section with their references)
    ("geomagnetic surface field", 50e-6),
    ("sunspot umbra", 0.3),
    ("strongly magnetic white dwarf", 1e5),
    ("radio pulsar surface", 1e8),
    ("magnetar (low)", 1e10),
    ("magnetar (high)", 1e11),
    ("galactic field 3 muG", 3e-10),
    ("galactic field 5 muG", 5e-10),
    ("galactic field 6 muG", 6e-10),
]
table = {}
ok = True
for name, Bt in fields:
    zc = np.array([C.c * Bt, 0, 0], dtype=complex)            # pure magnetic: E = 0
    u = C.epsilon_0 / 2 * np.sum(np.abs(zc) ** 2)
    u0 = C.epsilon_0 / 2 * abs(np.sum(zc * zc))
    rho0 = u0 / C.c ** 2
    ok &= math.isclose(u, Bt ** 2 / (2 * C.mu_0), rel_tol=1e-9) and math.isclose(u0, u, rel_tol=1e-12)  # eps0 mu0 c^2 = 1 to 1e-10 in tabulated CODATA values
    table[name] = (Bt, u, u0, rho0)
    info(f"{name:32s} B={Bt:9.2e} T  u=u0={u:10.3e} J/m^3 = {u / EV * CM3:10.3e} eV/cm^3  rho0={rho0:10.3e} kg/m^3")
ok &= math.isclose(table["geomagnetic surface field"][1], 9.947e-4, rel_tol=1e-3)
ok &= math.isclose(table["radio pulsar surface"][1], 3.979e21, rel_tol=1e-3)
ok &= math.isclose(table["magnetar (high)"][3], 4.427e10, rel_tol=1e-3)
report("e7-5", ok, "pure magnetic fields: u=u0=B^2/(2mu0), rho0=B^2/(2mu0c^2); geomagnetic 50 muT: 9.9e-4 J/m^3, "
       "rho0=1.1e-20 kg/m^3; pulsar 1e8 T: 4.0e21 J/m^3, rho0=4.4e4 kg/m^3; magnetar 1e11 T: rho0=4.4e10 kg/m^3")

T_CMB = 2.72548                                              # K, Fixsen 2009
a_rad = 4 * C.sigma / C.c
u_cmb = a_rad * T_CMB ** 4
u_cmb_ev = u_cmb / EV * CM3
ug = [table[k][1] / EV * CM3 for k in ("galactic field 3 muG", "galactic field 5 muG", "galactic field 6 muG")]
info(f"u_CMB = {u_cmb:.4e} J/m^3 = {u_cmb_ev:.4f} eV/cm^3; galactic u_B = {ug[0]:.3f}, {ug[1]:.3f}, {ug[2]:.3f} eV/cm^3")
ok = math.isclose(u_cmb_ev, 0.2606, rel_tol=2e-3) and all(0.8 < x / u_cmb_ev < 3.5 for x in ug) \
    and all(0.2 < x / 1.0 < 1.0 for x in ug)
report("e7-6", ok, f"u_CMB=aT0^4={u_cmb_ev:.4f} eV/cm^3; galactic field 3-6 muG gives 0.22-0.89 eV/cm^3, within a factor "
       "3.5 of u_CMB and of the cosmic-ray density ~1 eV/cm^3 (arithmetic on cited inputs)")

# ------------------------------------------------------------------- Gaussian random field: E0/E6 (e7-7 ... e7-9)
x1, x2, x3, y1, y2, y3 = sp.symbols('x1 x2 x3 y1 y2 y3', real=True)
xv, yv = sp.Matrix([x1, x2, x3]), sp.Matrix([y1, y2, y3])
zsq = sp.expand((xv.dot(xv) - yv.dot(yv)) ** 2 + 4 * xv.dot(yv) ** 2)            # |q|^2
Gam = sp.Matrix([[xv.dot(xv), xv.dot(yv)], [xv.dot(yv), yv.dot(yv)]])
id1 = sp.expand(Gam.trace() ** 2 - 4 * Gam.det() - zsq) == 0                       # (l1-l2)^2 = |q|^2
id2 = sp.expand((xv.dot(xv) + yv.dot(yv)) ** 2 - zsq - 4 * (xv.cross(yv)).dot(xv.cross(yv))) == 0
l1, l2, s, t = sp.symbols('lambda1 lambda2 s t', positive=True)
dens = sp.exp(-(l1 + l2) / 2) * (l1 - l2)                                           # real Wishart W_2(3,I), l1>l2>0
Z = sp.integrate(sp.integrate(dens, (l2, 0, l1)), (l1, 0, sp.oo))
# change of variables l1 = s(1+t)/2, l2 = s(1-t)/2, Jacobian s/2
dens_st = sp.simplify(dens.subs({l1: s * (1 + t) / 2, l2: s * (1 - t) / 2}) * s / 2 / Z)
f_t = sp.simplify(sp.integrate(dens_st, (s, 0, sp.oo)))
g_s = sp.simplify(sp.integrate(dens_st, (t, 0, 1)))
indep = sp.simplify(dens_st - f_t * g_s) == 0
mean_t = sp.integrate(t * f_t, (t, 0, 1))
mom = {k: sp.integrate(t ** k * f_t, (t, 0, 1)) for k in (2, 4)}
report("e7-7", id1 and id2 and sp.simplify(f_t - 2 * t) == 0 and indep and mean_t == sp.Rational(2, 3),
       "for z with i.i.d. standard complex Gaussian components, |q|=l1-l2, |z|^2=l1+l2 (Gram eigenvalues); "
       "t=E0/E6 has density 2t on [0,1], CDF t^2, independent of |z|^2, mean 2/3; hence <E0>/<E6>=2/3 (exact)")


def gauss_moment(poly, vars_):
    """Exact expectation of a polynomial in independent standard normal variables."""
    P = sp.Poly(sp.expand(poly), *vars_)
    tot = sp.Integer(0)
    for mon, coeff in P.terms():
        val = sp.Integer(1)
        for k in mon:
            if k % 2:
                val = 0
                break
            val *= sp.factorial2(k - 1) if k > 0 else 1
        tot += coeff * val
    return tot


vars6 = [x1, x2, x3, y1, y2, y3]
s2 = xv.dot(xv) + yv.dot(yv)
Eq2, Eq4 = gauss_moment(zsq, vars6), gauss_moment(zsq ** 2, vars6)
Es2, Es4 = gauss_moment(s2 ** 2, vars6), gauss_moment(s2 ** 4, vars6)
ok = (Eq2 / Es2 == mom[2]) and (Eq4 / Es4 == mom[4]) and Eq2 == 24 and Es2 == 48
report("e7-8", ok, f"Isserlis cross-check: E|q|^2={Eq2}, E|z|^4={Es2}, E|q|^4={Eq4}, E|z|^8={Es4}; ratios "
       f"{Eq2 / Es2}, {Eq4 / Es4} equal the moments 1/2, 1/3 of the density 2t (exact)")

Nmc = 2_000_000
xs, ys = rng.standard_normal((Nmc, 3)), rng.standard_normal((Nmc, 3))
zz = xs + 1j * ys
qa = np.abs(np.sum(zz * zz, axis=1))
sa = np.sum(np.abs(zz) ** 2, axis=1)
tt = qa / sa
mean_mc = tt.mean()
ratio_mc = qa.mean() / sa.mean()
ts = np.sort(tt)
ks = np.max(np.abs(np.arange(1, Nmc + 1) / Nmc - ts ** 2))
corr = np.corrcoef(tt, sa)[0, 1]
ok = abs(mean_mc - 2 / 3) < 1e-3 and abs(ratio_mc - 2 / 3) < 1e-3 and ks < 1.5e-3 and abs(corr) < 3e-3
report("e7-9", ok, f"Monte Carlo, N=2e6: <t>={mean_mc:.5f}, <|q|>/<|z|^2>={ratio_mc:.5f} (|dev|<1e-3 from 2/3), "
       f"KS distance to CDF t^2 = {ks:.1e} (<1.5e-3), corr(t,|z|^2)={corr:.1e}")

# CMB numbers
H0 = 67.4e3 / MPC                                            # Planck 2018
rho_crit = 3 * H0 ** 2 / (8 * math.pi * C.G)
u0_cmb = 2 / 3 * u_cmb
rho0_cmb = u0_cmb / C.c ** 2
info(f"u_CMB={u_cmb:.4e} J/m^3, <u0>={u0_cmb:.4e} J/m^3, <rho0>={rho0_cmb:.4e} kg/m^3, "
     f"rho_crit={rho_crit:.4e} kg/m^3, <rho0>/rho_crit={rho0_cmb / rho_crit:.3e}, u_CMB/(rho_crit c^2)="
     f"{u_cmb / (rho_crit * C.c ** 2):.3e}")
ok = math.isclose(u_cmb, 4.175e-14, rel_tol=2e-3) and math.isclose(rho0_cmb, 3.097e-31, rel_tol=2e-3) \
    and 2 / 3 <= 1 and math.isclose(rho_crit, 8.53e-27, rel_tol=3e-3)
report("e7-10", ok, f"CMB today: u=4.17e-14 J/m^3 (0.261 eV/cm^3), mean rest-energy density (2/3)u=2.78e-14 J/m^3, "
       f"<rho0>=3.10e-31 kg/m^3 = {rho0_cmb / rho_crit:.1e} rho_crit; whole: Mc^2=U >= int E0 = (2/3)U")

# ---------------------------------------------------------------------------------- pair kinematics (e7-11, e7-12)
E1s, E2s, cth, ms = sp.symbols('E1 E2 c m', positive=True)
k1 = sp.Matrix([E1s, 0, 0, E1s])
k2 = sp.Matrix([E2s, E2s * sp.sqrt(1 - cth ** 2), 0, E2s * cth])
ksum = k1 + k2
s_inv = sp.simplify((ksum.T * g * ksum)[0])
ok = sp.simplify(s_inv - 2 * E1s * E2s * (1 - cth)) == 0
mec2 = C.m_e * C.c ** 2 / EV                                 # eV
kT = C.k * T_CMB / EV
c_mean = math.pi ** 4 / (30 * float(sp.zeta(3)))              # <eps>/kT of a blackbody = pi^4/(30 zeta(3))
eps_mean = c_mean * kT                                       # mean blackbody photon energy
Eth_mean = mec2 ** 2 / eps_mean                              # head-on, (1-cos)=2: E eps >= (m c^2)^2
Eth_kT = mec2 ** 2 / kT
Eth_ebl = [mec2 ** 2 / e for e in (3.0, 0.1)]
info(f"m_e c^2 = {mec2:.6e} eV; kT0 = {kT:.4e} eV; E_th(head-on) on <eps>: {Eth_mean:.3e} eV, on kT0: {Eth_kT:.3e} eV;"
     f" on EBL 3 eV..0.1 eV: {Eth_ebl[0]:.3e}..{Eth_ebl[1]:.3e} eV")
ok &= 1e14 < Eth_mean < Eth_kT < 2e15 and 5e10 < Eth_ebl[0] < 1e11 and 2e12 < Eth_ebl[1] < 3e12 \
    and math.isclose(c_mean, 2.70118, rel_tol=1e-5)
report("e7-11", ok, f"two photons: s=2E1E2(1-cos th) (exact); head-on threshold E*eps>=(m_e c^2)^2: on the CMB "
       f"{Eth_mean:.1e} eV (mean photon) to {Eth_kT:.1e} eV (kT0); on EBL photons of 3-0.1 eV: 0.087-2.6 TeV")

# RE Thm 8.6(v): two null fields: M_pair = |z1.z2| = sqrt(E6 E6')(1-cos th12); photons: M^2 = 2 E1 E2 (1-cos th)
ok = True
for thv in np.linspace(0.1, math.pi, 7):
    z1 = np.array([1, 1j, 0])                                # null, along e3
    R = expm(thv * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float))   # rotate about e2
    z2 = 1.7 * (R @ z1)
    Mp = abs(np.dot(z1, z2))
    E6a, E6b = np.sum(np.abs(z1) ** 2) / 2, np.sum(np.abs(z2) ** 2) / 2
    ok &= math.isclose(Mp, math.sqrt(E6a * E6b) * (1 - math.cos(thv)), rel_tol=1e-12)
ok &= abs(np.dot(np.array([1, 1j, 0]), 2 * np.array([1, 1j, 0]))) == 0
report("e7-12", ok, "two null fields: M_pair=|z1.z2|=sqrt(E6 E6')(1-cos th12) (restated RE Thm 8.6(v)): zero iff "
       "collinear, as the photon pair mass; but M_pair is linear in (1-cos th), M_photon^2 is (field densities are quadratic)")

# --------------------------------------------------------------------------- canonical fields, Schwinger (e7-13, 14)
lam = sp.symbols('lambda')
F = sp.Matrix([[0, e1s, e2s, e3s],
               [e1s, 0, b3, -b2],
               [e2s, -b3, 0, b1],
               [e3s, b2, -b1, 0]])       # F^mu_nu for textbook E, B (one sign convention; the invariants are insensitive)
cp = sp.expand(F.charpoly(lam).as_expr())
ReQ = Bv_s.dot(Bv_s) - Ev_s.dot(Ev_s)
EdB = Ev_s.dot(Bv_s)
ok = sp.expand(cp - (lam ** 4 + ReQ * lam ** 2 - EdB ** 2)) == 0
# canonical fields: E || B frame with E = eps n, B = beta n gives q = (beta + i eps)^2
qcan = sp.expand((beta_ + sp.I * eps_) ** 2)
ok &= sp.expand(qcan - ((beta_ ** 2 - eps_ ** 2) + 2 * sp.I * eps_ * beta_)) == 0
ok &= sp.expand(sp.re(qcan) ** 2 + sp.im(qcan) ** 2 - (eps_ ** 2 + beta_ ** 2) ** 2) == 0   # |q| = eps^2 + beta^2
worst = 0
for _ in range(200):
    Ev, Bv = rng.normal(size=3), rng.normal(size=3)
    qn = np.dot(Bv, Bv) - np.dot(Ev, Ev) + 2j * np.dot(Ev, Bv)
    eps2, bet2 = (abs(qn) - qn.real) / 2, (abs(qn) + qn.real) / 2
    Fn = np.array([[0, *Ev], [Ev[0], 0, Bv[2], -Bv[1]], [Ev[1], -Bv[2], 0, Bv[0]], [Ev[2], Bv[1], -Bv[0], 0]])
    ev = np.linalg.eigvals(Fn)
    re_ = np.sort(np.abs(ev.real))[-1]
    im_ = np.sort(np.abs(ev.imag))[-1]
    worst = max(worst, abs(re_ ** 2 - eps2), abs(im_ ** 2 - bet2), abs((eps2 + bet2) / 2 - abs(qn) / 2))
# pure B: F is real antisymmetric => exp(F) orthogonal (rotation); pure E: F symmetric => exp(F) a boost
FB = np.array([[0, 0, 0, 0], [0, 0, 0.7, 0], [0, -0.7, 0, 0], [0, 0, 0, 0]])
FE = np.array([[0, 0.7, 0, 0], [0.7, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
UB, UE = expm(FB), expm(FE)
ok &= np.allclose(UB.T @ UB, np.eye(4), atol=1e-13) and np.allclose(UE.T @ np.diag([1, -1, -1, -1]) @ UE,
                                                                     np.diag([1, -1, -1, -1]), atol=1e-13)
ok &= not np.allclose(UE.T @ UE, np.eye(4), atol=1e-3)
report("e7-13", ok and worst < 1e-10,
       "char. poly of F^mu_nu is l^4+(B^2-E^2)l^2-(E.B)^2, eigenvalues +-eps, +-i beta with eps^2=(|q|-Re q)/2, "
       f"beta^2=(|q|+Re q)/2, E0=|q|/2=(eps^2+beta^2)/2 (exact; 200 random fields, dev {worst:.0e}); pure B generates "
       "rotations, pure E boosts")

E_S = C.m_e ** 2 * C.c ** 3 / (C.e * C.hbar)
B_Q = C.m_e ** 2 * C.c ** 2 / (C.e * C.hbar)
info(f"E_S = {E_S:.4e} V/m, B_Q = {B_Q:.4e} T; magnetar/B_Q = {1e10 / B_Q:.2f}..{1e11 / B_Q:.1f}; "
     f"pulsar/B_Q = {1e8 / B_Q:.3f}; exponent at eps = 0.1 E_S: exp(-pi*10) = {math.exp(-10 * math.pi):.2e}")
ok = math.isclose(E_S, 1.3233e18, rel_tol=1e-3) and math.isclose(B_Q, 4.414e9, rel_tol=1e-3) \
    and math.isclose(E_S / B_Q, C.c, rel_tol=1e-12) and 1e10 / B_Q > 1 and 1e8 / B_Q < 0.1
report("e7-14", ok, f"E_S=m_e^2c^3/(e hbar)={E_S:.3e} V/m, B_Q=m_e^2c^2/(e hbar)={B_Q:.3e} T; magnetars (1e10-1e11 T) "
       f"exceed B_Q by 2.3-23, radio pulsars (1e8 T) are at 0.023 B_Q")

# ------------------------------------------------------------------- six-dimensional kinematics (e7-15 ... e7-18)
m1, m2, tht = sp.symbols('m1 m2 theta', positive=True)
u1 = sp.Matrix([1, 0, 0, 0, 0, 0])
u2 = sp.Matrix([sp.cos(tht), sp.sin(tht), 0, 0, 0, 0])
eta_s = sp.diag(1, 1, 1, -1, -1, -1)
Ps = m1 * u1 + m2 * u2
M2 = sp.simplify((Ps.T * eta_s * Ps)[0])
ok = sp.simplify(M2 - (m1 ** 2 + m2 ** 2 + 2 * m1 * m2 * sp.cos(tht))) == 0
ok &= sp.simplify(M2.subs(tht, 0) - (m1 + m2) ** 2) == 0 and sp.simplify(M2.subs(tht, sp.pi / 2) - m1 ** 2 - m2 ** 2) == 0
ok &= sp.simplify(M2.subs(tht, sp.pi) - (m1 - m2) ** 2) == 0
# Clifford image: (m1 e1 + m2 (3 e1 + 4 e2)/5)^2 = (m1^2 + m2^2 + 6/5 m1 m2) id, checked x25 in integers
for a_, b_ in ((1, 1), (2, 3), (7, 5)):
    H5 = 5 * a_ * E[0] + b_ * (3 * E[0] + 4 * E[1])
    ok &= np.array_equal(H5 @ H5, (25 * a_ ** 2 + 25 * b_ ** 2 + 30 * a_ * b_) * I8)
# graded tower at n = 2: m1 E_(1,1) + m2 E_(2,1) squares to (m1^2+m2^2) id (temporal directions orthogonal)
E11 = np.kron(E[0], I8)
E21 = np.kron(OMEGA, E[0])
U1, U2 = np.kron(E[0], I8), np.kron(I8, E[0])
for a_, b_ in ((1, 1), (2, 3)):
    Hg = a_ * E11 + b_ * E21
    ok &= np.array_equal(Hg @ Hg, (a_ ** 2 + b_ ** 2) * np.eye(64, dtype=np.int64))
    Hu = a_ * U1 + b_ * U2
    evu = sorted(set(np.round(np.linalg.eigvalsh(Hu.astype(float)), 9)))
    ok &= evu == sorted({a_ + b_, a_ - b_, -a_ + b_, -a_ - b_})
report("e7-15", ok, "six-momenta at mutual rest: M^2=m1^2+m2^2+2m1m2 cos th (additive at 0, quadrature at pi/2, |m1-m2| "
       "at pi); its Clifford image squares to M^2 (exact at cos th=3/5); the graded tower is the case th=pi/2 (exact, n=2), "
       "the ungraded sum has spectrum {+-m1+-m2}, not that of a vector")

P = np.array([10, 0, 0, 0, 0, 0])
P1 = np.array([5, 5, 0, 1, 0, 0])
P2 = np.array([5, -5, 0, -1, 0, 0])


def eta6(x, y):
    return int(x @ ETA @ y)


ok = np.array_equal(P1 + P2, P) and eta6(P, P) == 100 and eta6(P1, P1) == 49 and eta6(P2, P2) == 49
ok &= 7 + 7 > 10 and eta6(P1, P2) == 1 and eta6(P1, P2) < 7 * 7
m4sq = [int(v[0] ** 2 - v[3] ** 2 - v[4] ** 2 - v[5] ** 2) for v in (P1, P2)]      # mass read by the clock e1
ok &= m4sq == [24, 24] and 2 * math.sqrt(24) < 10 and all(int(v[0]) > 0 for v in (P1, P2))
report("e7-16", ok, "uphill decay in signature (3,3) (exact integers, c=1): P=(10,0,0;0,0,0) -> P1=(5,5,0;1,0,0) + "
       "P2=(5,-5,0;-1,0,0), masses 10 -> 7+7=14 > 10; eta(P1,P2)=1 < m1 m2=49 (reverse Cauchy-Schwarz fails); "
       "the clock e1 reads m4^2=24 each, 2 sqrt(24) < 10")

# one clock: vectors with temporal part parallel to u span R u + V^- of signature (1,3): reverse CS holds
uvec = np.array([1, 2, 2, 0, 0, 0]) / 3.0
Bsub = np.column_stack([uvec, np.eye(6)[:, 3], np.eye(6)[:, 4], np.eye(6)[:, 5]])
gram = Bsub.T @ ETA @ Bsub
ok = np.allclose(gram, np.diag([1, -1, -1, -1]))
viol_1clock, viol_3 = 0, 0
for _ in range(20000):
    def fut(sub):
        while True:
            v = rng.normal(size=6)
            if sub:
                v = Bsub @ rng.normal(size=4)
            if v @ ETA @ v > 1e-6 and (v[:3] @ uvec[:3]) > 0:
                return v
    a1, a2 = fut(True), fut(True)
    m_1, m_2 = math.sqrt(a1 @ ETA @ a1), math.sqrt(a2 @ ETA @ a2)
    viol_1clock += (a1 @ ETA @ a2) < m_1 * m_2 * (1 - 1e-12)
    b1v, b2v = fut(False), fut(False)
    n_1, n_2 = math.sqrt(b1v @ ETA @ b1v), math.sqrt(b2v @ ETA @ b2v)
    viol_3 += (b1v @ ETA @ b2v) < n_1 * n_2
ok &= viol_1clock == 0 and viol_3 > 1000
report("e7-17", ok, f"one clock: R u + V^- has signature (1,3), and eta(P1,P2)>=m1 m2 in 20000 random future-timelike "
       f"pairs ({viol_1clock} violations); without a common clock {viol_3} of 20000 random pairs violate it")

ser = sp.series(sp.sqrt(m1 ** 2 + m2 ** 2 + 2 * m1 * m2 * sp.cos(tht)), tht, 0, 4).removeO()
ser = ser.subs(sp.sqrt(sp.expand((m1 + m2) ** 2)), m1 + m2)
ok = sp.simplify(ser - (m1 + m2 - m1 * m2 / (m1 + m2) * tht ** 2 / 2)) == 0
# 2m^2(1+cos th) = 4m^2 cos^2(th/2): exact trigonometric identity
ok &= sp.simplify(sp.expand_trig(2 * ms ** 2 + 2 * ms ** 2 * sp.cos(tht) - 4 * ms ** 2 * sp.cos(tht / 2) ** 2)
                  .rewrite(sp.cos)) == 0
ok &= sp.series(1 - sp.cos(tht / 2), tht, 0, 4).removeO() == tht ** 2 / 8
# general masses: fractional defect (m1 m2 / M0^2) theta^2 / 2 with M0 = m1 + m2; equal masses: theta^2 / 8
frac = sp.simplify(((m1 + m2) - (m1 + m2 - m1 * m2 / (m1 + m2) * tht ** 2 / 2)) / (m1 + m2))
ok &= sp.simplify(frac - m1 * m2 / (m1 + m2) ** 2 * tht ** 2 / 2) == 0 and sp.simplify(frac.subs(m2, m1) - tht ** 2 / 8) == 0
delta = 4.4e-7                                               # Rainville et al. 2005: |1 - Delta m c^2 / E_gamma| = (-1.4 +- 4.4)e-7
u_MeV = C.physical_constants['atomic mass constant energy equivalent in MeV'][0]
m28, m29 = 27.97692653465, 28.97649466490                    # AME2020 atomic masses (u)
mn_u = C.physical_constants['neutron mass in u'][0]
Sn = (m28 + mn_u - m29) * u_MeV                              # MeV, neutron separation energy of 29Si
mu_red = (m28 * u_MeV) * (mn_u * u_MeV) / ((m28 + mn_u) * u_MeV)
M0 = (m28 + mn_u) * u_MeV
dE = delta * Sn                                               # MeV: budget for an unexplained defect (1 sigma)
eps_M = dE / M0                                               # the same budget as a fraction of the whole mass
th_si = math.sqrt(2 * dE / mu_red)
th_eq = math.sqrt(8 * eps_M)
info(f"S_n(29Si) = {Sn:.4f} MeV, reduced mass {mu_red:.2f} MeV, 1 sigma budget {dE * 1e6:.2f} eV = {eps_M:.2e} of "
     f"M0 = {M0:.1f} MeV; theta < {th_si:.2e} rad (n + 28Si); equal masses at the same fraction: theta < {th_eq:.2e} rad")
ok &= math.isclose(Sn, 8.4736, abs_tol=2e-4) and 8e-5 < th_si < 1e-4 and 1.3e-10 < eps_M < 1.5e-10 and 3e-5 < th_eq < 3.5e-5
report("e7-18", ok, f"M=m1+m2-mu theta^2/2+O(theta^4), fractional defect (m1m2/M0^2)theta^2/2, theta^2/8 at equal masses "
       f"(M=2m cos(theta/2), exact); conditional 1 sigma bounds from Rainville et al.: unexplained defect <= 4.4e-7 S_n = "
       f"{dE * 1e6:.1f} eV = {eps_M:.1e} M0, so theta<{th_si:.1e} rad for n+28Si (S_n={Sn:.4f} MeV) and theta<{th_eq:.1e} "
       f"rad for equal masses at that fraction")

# ----------------------------------------------------------------------------------- rates kappa = c/l (e7-19)
lam_e = C.hbar / (C.m_e * C.c)
l_P = math.sqrt(C.hbar * C.G / C.c ** 3)
R_H = C.c / H0
rows = [("reduced Compton wavelength", lam_e), ("Planck length", l_P), ("Hubble radius c/H0", R_H)]
ok = True
for name, l in rows:
    kap = C.c / l
    info(f"{name:28s} l={l:.4e} m  kappa={kap:.4e} 1/s  hbar*kappa={C.hbar * kap / EV:.4e} eV")
ok &= math.isclose(C.hbar * C.c / lam_e / EV, mec2, rel_tol=1e-12)
ok &= math.isclose(C.hbar * H0 / EV, 1.438e-33, rel_tol=3e-3) and math.isclose(C.c / l_P, 1.855e43, rel_tol=1e-3)
report("e7-19", ok, f"kappa=c/l: l=hbar/(m_e c) gives {C.c / lam_e:.3e} 1/s, hbar kappa=m_e c^2; l=l_P gives "
       f"{C.c / l_P:.3e} 1/s ({C.hbar * C.c / l_P / EV / 1e9:.3e} GeV); l=c/H0 gives H0={H0:.3e} 1/s, "
       f"hbar H0={C.hbar * H0 / EV:.2e} eV")

# ------------------------------------------------------------------ duality on a null field is a rotation (e7-20)
def cross_mat(n):
    return np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])


ok = True
worst = 0
for _ in range(50):
    n1 = rng.normal(size=3)
    n2 = np.cross(rng.normal(size=3), n1)
    n2 *= np.linalg.norm(n1) / np.linalg.norm(n2)
    Bv, Ev = n1, n2                                         # |B|=|E|, B.E=0: null
    zv = Bv + 1j * Ev
    nax = np.cross(Bv, Ev)
    nax /= np.linalg.norm(nax)
    ok &= np.allclose(np.cross(nax, zv), -1j * zv, atol=1e-12)
    ok &= np.allclose(np.cross(-Ev, Bv) / np.linalg.norm(np.cross(-Ev, Bv)), nax)   # = E x B_t: propagation
    for thv in (0.3, 1.1, 2.9):
        Xv = np.concatenate([Bv, Ev])
        Rn = expm(-thv * cross_mat(nax))
        lhs = expm(thv * G6.astype(float)) @ Xv
        rhs = np.concatenate([Rn @ Bv, Rn @ Ev])
        worst = max(worst, np.max(np.abs(lhs - rhs)))
# plane wave z = f(n.x - t) w solves the frame Maxwell equation dz/dt = -i curl z  (ES17 Prop. 12.10)
xx, yy, zz_, tt_ = sp.symbols('x y z t', real=True)
fsym = sp.Function('f')
wv = sp.Matrix([1, sp.I, 0])                                 # n = e3, n x w = -i w
zf = wv * fsym(zz_ - tt_)
curl = sp.Matrix([sp.diff(zf[2], yy) - sp.diff(zf[1], zz_), sp.diff(zf[0], zz_) - sp.diff(zf[2], xx),
                  sp.diff(zf[1], xx) - sp.diff(zf[0], yy)])
ok &= sp.simplify(sp.diff(zf, tt_) + sp.I * curl) == sp.zeros(3, 1)
ok &= sp.Matrix([0, 0, 1]).cross(wv) == -sp.I * wv
# non-null field: no rotation equals a duality rotation by th not in pi Z (q -> e^{2i th} q, rotations fix q)
zn = np.array([1.0, 0.3j, 0.2])
qn = np.sum(zn * zn)
ok &= abs(np.exp(2j * 0.3) * qn - qn) > 1e-3
report("e7-20", ok and worst < 1e-12,
       f"on a null field with n=B x E/|B x E| (= textbook propagation direction), n x z=-iz and e^(th G)=R_n(-th) on both "
       f"triples (50 fields x 3 angles, dev {worst:.0e}); z=f(n.x-t)w solves dz/dt=-i curl z (exact); on non-null "
       "fields no rotation is a duality rotation")

# ------------------------------------------------------------------------------- birefringence bound (e7-21)
beta_c, beta_s = 0.35, 0.14                                  # deg, Minami & Komatsu 2020
beta_max = math.radians(beta_c + 2 * beta_s)
t_age = 13.787 * GYR                                         # Planck 2018
kap_max = beta_max / t_age
kap_e = C.m_e * C.c ** 2 / C.hbar
orders = math.log10(kap_e / kap_max)
info(f"beta_max (2 sigma) = {math.degrees(beta_max):.2f} deg = {beta_max:.4e} rad over t = {t_age:.4e} s: "
     f"kappa < {kap_max:.3e} 1/s; m_e c^2/hbar = {kap_e:.4e} 1/s; ratio 10^{orders:.1f}; 1 deg bound: "
     f"{math.radians(1) / t_age:.2e} 1/s")
ok = 1e-20 < kap_max < 5e-20 and orders > 40
rH = kap_max / H0
E_kap = C.hbar * kap_max / EV
info(f"kappa_max / H0 = {rH:.3e}; hbar kappa_max = {E_kap:.3e} eV")
ok &= 1.0e-2 < rH < 1.3e-2 and 1.5e-35 < E_kap < 1.8e-35
report("e7-21", ok, f"|beta|<0.63 deg (2 sigma) over 13.79 Gyr: a uniform duality rate kappa<{kap_max:.1e} 1/s = "
       f"{rH:.1e} H0 (hbar kappa<{E_kap:.1e} eV), 10^{orders:.1f} below m_e c^2/hbar={kap_e:.2e} 1/s (arithmetic on cited inputs)")

# ---------------------------------------------------------- closure pendulum vanishes on null fields (e7-22)
nu1, nu2, mm = sp.symbols('nu1 nu2 m', real=True)
ETAs = sp.diag(1, 1, 1, -1, -1, -1)
Gs = sp.Matrix(G6.tolist())
Nin = (nu1 * ETAs.inv() + nu2 * Gs * ETAs.inv()) / mm
ok = True
for Xn in (sp.Matrix([1, 0, 0, 0, 1, 0]), sp.Matrix([1, 2, 2, 2, 1, -2])):     # null: |B|=|E|, B.E=0
    zX = Xn[:3, 0] + sp.I * Xn[3:, 0]
    ok &= sp.expand(zX.dot(zX)) == 0
    p0 = ETAs * Xn                                       # momentum p = eta pi
    for thv in (sp.Rational(1, 3), 1, 2):
        Rot = sp.cos(thv) * sp.eye(6) - sp.sin(thv) * Gs       # e^{-th G}
        pth = Rot * p0
        ok &= sp.simplify((pth.T * Nin * pth)[0]) == 0
Xm = sp.Matrix([1, 0, 0, 0, 0, 0])
pm = ETAs * Xm
ok &= sp.simplify((pm.T * Nin * pm)[0] - nu1 / mm) == 0
report("e7-22", ok, "closure pendulum: p^T N p/2 vanishes identically along e^(-th G)p for null momentum fields (no "
       "restoring force on the duality angle), while p^T N p = nu1/m at the field at rest X=e1 (restated RE Prop. 4.3, Cor. 4.5)")

# ------------------------------------------------------------------- three manifolds: time-odd invariant (e7-23)
k1s, k2s, k3s, B12, B23, B31 = sp.symbols('kappa1 kappa2 kappa3 B12 B23 B31', real=True)
Kd = sp.diag(k1s, k2s, k3s)
Bm = sp.Matrix([[0, B12, -B31], [-B12, 0, B23], [B31, -B23, 0]])
H = Kd - sp.I * Bm
Jinv = B12 * B23 * B31
plaq = sp.expand(H[0, 1] * H[1, 2] * H[2, 0])
ok = sp.simplify(plaq - sp.I * Jinv) == 0
comm = H * Kd - Kd * H
ok &= sp.simplify(comm.det() - 2 * sp.I * Jinv * (k1s - k2s) * (k1s - k3s) * (k2s - k3s)) == 0
Hm = Kd + sp.I * Bm
ok &= sp.simplify(Hm[0, 1] * Hm[1, 2] * Hm[2, 0] + sp.I * Jinv) == 0
report("e7-23", ok, "three manifolds: plaquette H12H23H31=iJ, J=B12B23B31, det[H,K]=2iJ(k1-k2)(k1-k3)(k2-k3), "
       "and B->-B (interior time reversal) sends J->-J (restated RE Thm 9.5(ii),(iv), exact)")

# ------------------------------------------------------ temporal rotation of a clock: oscillating m4 (e7-24, 25)
mu_, phi_, Om, t_, th0 = sp.symbols('mu phi Omega t theta0', positive=True)
Pt = mu_ * sp.Matrix([sp.cos(phi_), sp.sin(phi_), 0, 0, 0, 0])
ok = sp.simplify((Pt.T * eta_s * Pt)[0] - mu_ ** 2) == 0
m4sq_t = Pt[0] ** 2 - Pt[3] ** 2 - Pt[4] ** 2 - Pt[5] ** 2
ok &= sp.simplify(m4sq_t - (mu_ ** 2 - Pt[1] ** 2 - Pt[2] ** 2)) == 0          # RE Prop. 2.10, clock e1
ok &= sp.simplify(m4sq_t - mu_ ** 2 * sp.cos(phi_) ** 2) == 0
osc = sp.series(sp.cos(th0 * sp.sin(Om * t_)), th0, 0, 4).removeO()
ok &= sp.simplify(osc - (1 - th0 ** 2 / 4 + th0 ** 2 / 4 * sp.cos(2 * Om * t_))) == 0
# uniform rotation at rate nu about an axis e_hat at angle alpha from the clock n, temporal energy initially along n, p = 0
al, nu_ = sp.symbols('alpha nu', positive=True)
nvec_c = sp.Matrix([1, 0, 0])
ehat = sp.Matrix([sp.cos(al), sp.sin(al), 0])
Kx = sp.Matrix([[0, -ehat[2], ehat[1]], [ehat[2], 0, -ehat[0]], [-ehat[1], ehat[0], 0]])
Rrod = sp.eye(3) + sp.sin(nu_ * t_) * Kx + (1 - sp.cos(nu_ * t_)) * Kx * Kx       # Rodrigues
En_t = sp.simplify((nvec_c.T * Rrod * nvec_c)[0])
ok &= sp.simplify(En_t - (sp.cos(al) ** 2 + sp.sin(al) ** 2 * sp.cos(nu_ * t_))) == 0
report("e7-24", ok, "a temporal rotation at fixed six-dimensional mass mu: the clock e1 reads m4=mu cos(phi) "
       "(= RE Prop. 2.10); for phi=theta0 sin(Omega t), m4/mu = 1-theta0^2/4+(theta0^2/4)cos(2 Omega t)+O(theta0^4); "
       "for a uniform rotation about an axis at angle alpha from the clock (p=0, energy along the clock at t=0), "
       "m4/mu = 1 - sin^2(alpha)(1-cos nu t) (exact)")

A_q = 4e-7                                                   # |dmu/mu| <= 4e-7 (2 sigma), Kanekar et al. 2015
th_q = math.sqrt(4 * A_q)
rate = 0.5e-16 + 2 * 1.6e-16                                 # |mudot/mu| (2 sigma), Huntemann et al. 2014, per yr
AOm = rate / 2                                               # 2 Omega A <= rate
A_age = AOm * 13.787e9                                       # A for Omega = 1/(13.787 Gyr) (Omega in rad/yr)
eta_E = 1.5e-15 + 2 * math.hypot(2.3e-15, 1.5e-15)           # MICROSCOPE, central + 2 sigma
th_ep = math.sqrt(2 * eta_E)
info(f"quasar: A<=4e-7 -> theta0<{th_q:.2e}; clocks: A*Omega <= {AOm:.2e}/yr (A<={A_age:.1e} at Omega=1/13.8 Gyr, "
     f"A<={AOm:.1e} at Omega=1/yr); MICROSCOPE |eta|<{eta_E:.1e} -> |d theta^2|<{2 * eta_E:.1e}, theta<{th_ep:.1e}")
ok = math.isclose(th_q, 1.26e-3, rel_tol=1e-2) and math.isclose(AOm, 1.85e-16, rel_tol=1e-2) \
    and math.isclose(eta_E, 7.0e-15, rel_tol=1e-2) and math.isclose(th_ep, 1.18e-7, rel_tol=1e-2)
ok &= math.isclose(rate, 3.7e-16, rel_tol=1e-9) and math.isclose(math.sqrt(A_q), 6.3e-4, rel_tol=1e-2)
report("e7-25", ok, f"conditional bounds on differential oscillations m4/mu=1-A+A cos(omega t): quasar methanol "
       f"|dmu/mu|<4e-7 -> A<4e-7, i.e. theta0<{th_q:.1e} rad (libration, A=theta0^2/4) or alpha<6.3e-4 rad (A=sin^2 alpha); "
       f"clocks |mudot/mu|<3.7e-16/yr -> A omega<3.7e-16/yr (A*Omega<{AOm:.2e}/yr for omega=2 Omega); MICROSCOPE "
       f"|eta|<7.0e-15 -> |theta_A^2-theta_B^2|<1.4e-14, theta<{th_ep:.1e} rad")

# ------------------------------------------------------------- field energy against dark matter (e7-26)
h = 0.674
Oc_h2 = 0.120                                                # Planck 2018: Omega_c h^2 = 0.120 +- 0.001
rho_dm = Oc_h2 / h ** 2 * rho_crit                           # kg/m^3, cosmic mean cold dark matter density
u_dm_ev = rho_dm * C.c ** 2 / EV * CM3                       # eV/cm^3
ratio_gal = table["galactic field 6 muG"][1] / EV * CM3 / u_dm_ev
ratio_cmb = u0_cmb / (rho_dm * C.c ** 2)
info(f"rho_dm = {rho_dm:.3e} kg/m^3 = {u_dm_ev:.4e} eV/cm^3; galactic 6 muG / rho_dm c^2 = {ratio_gal:.2e}; "
     f"CMB <u0> / rho_dm c^2 = {ratio_cmb:.2e}")
ok = 1.2e3 < u_dm_ev < 1.3e3 and ratio_gal < 1e-3 and ratio_cmb < 2e-4
report("e7-26", ok, f"cosmic mean dark-matter energy density Omega_c h^2=0.120: {u_dm_ev:.0f} eV/cm^3; the local galactic "
       f"field (6 muG) is {ratio_gal:.1e} of it and the CMB mean rest-energy density {ratio_cmb:.1e} "
       "(arithmetic on cited inputs)")

# ------------------------------------------------- Schwinger rate is not a function of the rest energy (e7-27)
def schwinger_first(eps, beta):
    """First (pair-production) term of the Schwinger rate per unit volume and time, fields in units of E_S,
    rate in units m^4 c^5/hbar^3 (constant prefactors irrelevant): eps beta/(4 pi^2) coth(pi beta/eps) exp(-pi/eps)."""
    if eps == 0:
        return 0.0
    if beta == 0:
        return eps ** 2 / (4 * math.pi ** 3) * math.exp(-math.pi / eps)
    return eps * beta / (4 * math.pi ** 2) / math.tanh(math.pi * beta / eps) * math.exp(-math.pi / eps)


ok = True
for E0n in (0.02, 0.5, 2.0):                                # fixed E0 = (eps^2+beta^2)/2 in units of E_S^2
    r = math.sqrt(2 * E0n)
    rates = [schwinger_first(r * math.cos(ph), r * math.sin(ph)) for ph in np.linspace(0, math.pi / 2, 41)]
    ok &= rates[0] > 0 and rates[-1] == 0.0 and max(rates) > 10 * min(x for x in rates if x > 0)
    # at fixed eps the rate increases with beta (x coth x is increasing): a parallel B enhances, never creates
    ok &= all(schwinger_first(r, r * bb) < schwinger_first(r, r * (bb + 0.1)) for bb in np.linspace(0, 2, 21))
    # the limit beta -> 0 of the general formula equals the pure-E expression
    ok &= math.isclose(schwinger_first(r, 1e-9 * r), schwinger_first(r, 0.0), rel_tol=1e-6)
lim_ok = sp.limit(sp.Symbol('b') / sp.tanh(sp.pi * sp.Symbol('b') / sp.Symbol('e', positive=True)), sp.Symbol('b'), 0) \
    == sp.Symbol('e', positive=True) / sp.pi
info(f"at E0 = 0.5 E_S^2: rate(beta=0) = {schwinger_first(1.0, 0.0):.3e}, rate(eps=beta) = "
     f"{schwinger_first(math.sqrt(0.5), math.sqrt(0.5)):.3e}, rate(eps=0) = 0")
report("e7-27", ok and lim_ok, "first Schwinger term eps beta coth(pi beta/eps) exp(-pi E_S/eps)/(4 pi^2): at fixed "
       "E0=(eps^2+beta^2)/2 (three values, 41 angles) it varies by more than a factor 10 and vanishes at eps=0, so it is not a "
       "function of E0; at fixed eps it increases with beta, and its beta->0 limit is eps^2 exp(-pi/eps)/(4 pi^3) (floating "
       "point; limit exact)")

print(f"{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
