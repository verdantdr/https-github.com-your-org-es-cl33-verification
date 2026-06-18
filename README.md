# es-cl33-verification

Machine-verification suite **and companion documents** for **Part I** of the
Electromagnetic Sphere (ES) monograph. This is the repository referenced in
**Appendix F** ("Reproducibility: the representation and the verified identities"),
*Code availability*:

> The complete verification suite — the Wolfram Language scripts that produce every
> symbolic output quoted in *Worked derivations*, together with an independent
> `NumPy` reimplementation of the 8×8 representation and all algebraic checks — is
> openly archived here. Each script is named for the identity it establishes; running
> them reproduces the boxed results and every row of the table of verified identities.

Everything is checked in a concrete **real 8×8 representation of `Cl(3,3)`**
(Jordan–Wigner), with signature `eta = diag(+,+,+,-,-,-)`. The gravitational and
constraint identities are checked symbolically.

---

## Repository layout

```
es-cl33-verification/
├── ES_monograph_partI.pdf             original paper — Electromagnetic Sphere, Part I
├── ES_P1.pdf                          Part I derivations: Ch. 5, Ch. 8, Ch. 10, Appendix F
├── ES_wolfram_interpretation.pdf      typeset PDF rendering of this verification suite
├── thm6.23_prop7.6.pdf                written proof of Theorem 6.23 and Proposition 7.6
├── wolfram/
│   ├── representation.wl              shared 8×8 Cl(3,3) rep + helpers (loaded by all)
│   ├── 01_photon_centralizer.wl       rows 2–6   J forced uniquely; Solve -> {b->a, c->a}
│   ├── 02_charge_spectrum.wl          row 14     spec(J)={(+i)^4,(-i)^4} -> q=±1/2
│   ├── 03_dispersion.wl               rows 7,9–11 H^2=(p^2+m^2)·1
│   ├── 04_dirac_signature.wl          rows 12,13 Minkowski signature read off
│   ├── 05_spin_curvature.wl    ★ NEW  row 8      [G_X,G_Y]=-2 J3  (the not-shown derivation)
│   ├── 06_gravity_double_contraction.wl  row 15  -4 G^a_d in n=4 AND n=6
│   ├── 07_constraint_killing.wl       rows 16,17 {phi,C}=-½(Lie_u g)PP; closure ⇔ Killing
│   ├── thm4.3.wl                      code proof of Theorem 4.3
│   ├── prop9.4_cl22.wl                Cl(2,2) minimal-model code (Proposition 9.4)
│   └── run_all.wl                     master PASS/FAIL report over all 17 rows
└── numpy/
    └── verify_all.py                  independent reimplementation; all rows, separate code path
```

### Companion documents

- **`ES_monograph_partI.pdf`** — the original paper (Electromagnetic Sphere, Part I).
- **`ES_P1.pdf`** — the Part I derivations that this suite mechanizes: Chapter 5,
  Chapter 8, Chapter 10, and Appendix F. The scripts under `wolfram/` and `numpy/`
  reproduce the boxed results stated in these chapters.
- **`ES_wolfram_interpretation.pdf`** — the PDF rendering of this verification suite:
  the prose-and-output companion to the scripts (the same checks, typeset).
- **`thm6.23_prop7.6.pdf`** — the written proof of **Theorem 6.23** and **Proposition 7.6**.

### Additional proofs (other chapters)

Beyond the Appendix F table (rows 1–17, below), three artifacts establish results from
other chapters and sit **alongside** that table rather than inside it:

- **`thm4.3.wl`** — code proof of **Theorem 4.3** (§4).
- **`prop9.4_cl22.wl`** — the **Cl(2,2) minimal-model** code for **Proposition 9.4** (§9),
  run in the separate real 4×4 representation of `Cl(2,2)`.
- **`thm6.23_prop7.6.pdf`** — the written proof for **Theorem 6.23** (§6) and
  **Proposition 7.6** (§7).

---

## What is new here: the previously *not-shown* derivation

The appendix works **six** derivations in full prose but the summary table contains
one substantive identity that is tabulated yet **never derived in the text** — **row 8**,
the boost / spin-curvature commutator

```
[G_X, G_Y] = -2 e4 e5 = -2 J3     (and cyclic)
```

[`wolfram/05_spin_curvature.wl`](wolfram/05_spin_curvature.wl) supplies that missing
derivation, written in the same step-by-step style as the six that *are* shown, and
closes with the explicit 8×8 check. The analytic core is a two-line collapse:

```
W e4 W = W(e4 W) = W(-W e4) = -W^2 e4 = -e4      [uses {W,e4}=0 and W^2=+1]
  =>  G_X G_Y = (W e4 W) e5 = -e4 e5
      G_Y G_X = (W e5 W) e4 = -e5 e4 = +e4 e5
  =>  [G_X,G_Y] = -2 e4 e5 = -2 J3.
```

so the commutator of two **boosts** is `-2 ×` a spatial **rotation**: the boost
connection has rotational curvature, the quadratic part `A∧A` of a boost field
strength is `su(2)_S`-valued, and the spinor holonomy lies in `Spin(3)=SU(2)` — spin,
quantized in half-integers. **Status: EXACT** (the identity); **DERIVED** (the spin reading).

---

## The representation

With real Pauli matrices `sx`, `sz` and `ep = [[0,-1],[1,0]]` (the real "i", `ep^2=-1`):

```
e1 = sx⊗1⊗1     e2 = sz⊗sx⊗1     e3 = sz⊗sz⊗sx
e4 = ep⊗1⊗1     e5 = sz⊗ep⊗1     e6 = sz⊗sz⊗ep
```

These satisfy `{e_i,e_j} = 2 eta_ij` with `eta = diag(+,+,+,-,-,-)`. Derived objects:
the clock `W=(e1+e2+e3)/√3`; the photon `J=(e2e3+e3e1+e1e2)/√3`; the boosts
`G_a = W e_{3+a}`; the spatial rotations `J1=e5e6, J2=e6e4, J3=e4e5`; the mass
trivector `V=e4e5e6`; the Dirac matrices `gamma^0=W, gamma^k=e_{3+k}`.

---

## Running

**Wolfram** (WolframScript / a Wolfram kernel):

```bash
cd wolfram
wolframscript -f run_all.wl             # full PASS/FAIL table, exits 0 iff all pass
wolframscript -f 05_spin_curvature.wl   # just the not-shown derivation, step by step
wolframscript -f thm4.3.wl              # code proof of Theorem 4.3
wolframscript -f prop9.4_cl22.wl        # Cl(2,2) minimal-model check (Proposition 9.4)
```

Each individual script loads `representation.wl` by relative path, so run them from
inside `wolfram/` (or evaluate from a notebook with that directory on the path).

**NumPy** (independent cross-check):

```bash
cd numpy
python3 verify_all.py                   # all rows, separate code path, exits 0 iff all pass
```

---

## Table of verified identities (Appendix F)

|#|Identity|Script|
|-|-|-|
|1|`{e_i,e_j}=2 eta_ij`|representation.wl / verify_all.py|
|2|`W^2=+1`|01|
|3|`J^2=-1`|01|
|4|`Tr[J^2]=-8`|01|
|5|`[J,W]=0`|01|
|6|centralizer of `W` in `su(2)_T` is 1-dim `=J`|01|
|7|`G_a^2=+1`, `{G_a,G_b}=0` (a≠b)|03|
|**8**|**`[G_X,G_Y]=-2 e4e5` (and cyclic)**|**05 ★**|
|9|`V^2=+1`, `{G_a,V}=0`, `V^T=V`|03|
|10|`H=Σ G_a p_a + m V` real-symmetric|03|
|11|`H^2=(p^2+m^2)·1`, `spec(H)=±√(p^2+m^2)`|03|
|12|`{gamma^mu,gamma^nu}=2 diag(+,-,-,-)`|04|
|13|`Tr[g0]=0`, `spec(g0)={(+1)^4,(-1)^4}`|04|
|14|`spec(J)={(+i)^4,(-i)^4} ⇒ q=±1/2`|02|
|15|`delta^{apq}_{drs} R^{rs}_{pq} = -4 G^a_d` (n=4,6)|06|
|16|`{phi_1,phi_2}=0` (flat surplus plane)|07|
|17|`{phi_a,C}=-(Lie_{u_a}g)PP`; `=0 ⇔ Lie_{u_a}g=0`|07|

Rows 1–14 are exact statements in the real 8×8 representation; row 15 is the
generalized-Kronecker contraction on a generic Riemann tensor in dimensions 4 and 6,
returning the Einstein tensor with coefficient −4 in each; rows 16–17 are symbolic
phase-space identities on `T*M^6`.

---

## Status discipline

Every result carries its epistemic status, matching the monograph:
`[EXACT]`, `[DERIVED]`, `[AXIOM]`, `[CONJECTURE]`, `[OPEN]`. The scripts establish
the `[EXACT]` algebraic facts; the physical *readings* attached to them (charge
assignment, gravitational reading, the spin interpretation of row 8) are `[DERIVED]`
and are flagged as such in each file.
