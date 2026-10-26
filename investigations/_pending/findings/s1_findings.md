# Referee findings: investigations/two_readings/sections/r1_lagrangians.tex

Independent check: `investigations/two_readings/verify/s1_check.py` (ids s1-n; written without importing or copying
`verify/r1_lagrangians.py`; polarised commuting jets for the Grassmann field, sympy `euler_equations`, sympy SI units,
exact inertia by Descartes' rule, scipy integration, rational finite boosts and Cayley transforms).

Verified and NOT findings (refutation attempted, statement survives): the conventions note's six items
(E.B = +Im q/2, i.e. -E.B_text, s1-23; T^{0i} = -(E x B)_i = (E x B_text)_i, with a textbook wave along +z giving T^{0z} = +1,
s1-23; gamma^mu d_mu psi = -m J psi from L_D, s1-17; the Chern-Simons current needs the factor 1/2, factor 1 fails, s1-25;
L0 = E0 fails Gaillard-Zumino with defect -2P computed directly, s1-27/28; ES17 eq. (35) has the opposite overall sign
to RE Prop. 3.6(iii), s1-21), every other formula of Theorems/Propositions r1:thm:type, r1:prop:free, r1:prop:shift,
r1:prop:mix, r1:prop:dirac, r1:prop:maxwell, r1:thm:L0, r1:prop:ham, r1:prop:E6, r1:prop:total (s1-1 ... s1-37),
the ModMax Hamiltonian as the Legendre transform of L_gamma for gamma >= 0 (s1-31), all ES17/RE/FMB numbers
(checked against paper_labels.txt, re_labels.txt and a compilation of the FMB source: Defs. 2.1-2.2, Prop. 6.2,
eqs. (8), (15), (16)-(17), (18), (19)), the cross-references r2:prop:leak and r2:sec:field (present, and saying what
r1 uses), and every bibliographic datum (11 entries verbatim from tex/refs.tex; Belinfante40, BLST20 (published title
"Nonlinear duality-invariant conformal extension of Maxwell's equations", PRD 102, 121703), BabaeiAghbolagh22
(PLB 829, 137079, arXiv:2202.11156) and BialynickiBirula83 (pp. 31-48, the most widely cited range) verified by web search).

1. **OVERCLAIM** (kept, applied). Location: r1:thm:type(i) and the class sentence "(i) is obstructed in the sense that the
   arena's constants cannot supply the conversion"; Answer item (1) "(... fixed and obstructed)". What is wrong: the
   dimensional obstruction is a property of SI, where current is an independent base dimension. Evidence (s1-3):
   (eps0/(hbar c))^{1/2} E has dimension m^-2, which is the Heaviside-Lorentz field with hbar = c = 1, the paper's default
   units (Section 1.4 of ES17); in those units ell^3 alone converts a field strength into a length. So (i) cannot carry the
   type distinction; (iii) does. Corrected wording: "(i) is fixed and obstructs only conversions built from c, ell and
   hbar in SI units: since (eps0/(hbar c))^{1/2}E has dimension m^-2, the field of Heaviside-Lorentz units with hbar = c = 1,
   in those units ell^3 supplies the dimension, so what obstructs a dictionary between the readings is (iii), not (i)."
   Second look: the theorem does state the eps0 monomial honestly, but the class sentence and the Answer present (i) as an
   obstruction to the dictionary; in the paper's own units it is none. Survives.

2. **CITATION** (kept, applied). Location: r1:rem:fmb1, "and $\lG$ preserves no Lorentzian four-plane of $V$
   (Proposition~6.2 of \cite{FMB})". What is wrong: FMB Prop. 6.2 is a statement about an eta-self-adjoint G with G^2 = -1,
   and G is not in lG (G is eta-self-adjoint, elements of so(3,3) are eta-antiself-adjoint). That lG preserves no
   four-plane follows from irreducibility (Corollary 12.5 of ES17). Evidence: s1-7 (commutant C), s1-8 (G-invariant
   four-planes have signature (2,2)). Corrected wording: "and $\lG$, under which $V$ is irreducible (Corollary~12.5 of
   \cite{ES17}), preserves no proper subspace of $V$, while $G$ itself preserves no Lorentzian four-plane
   (Proposition~6.2 of \cite{FMB})". Second look: no reading of the sentence attaches FMB 6.2 to G. Survives.

3. **WRONG-DETAIL** (kept, applied). Location: Table r1:tab:objects, row $X_A=(\mathbf A;\mathbf C)$, SI unit
   "$\mathrm{V\,s\,m^{-1}}$". What is wrong: with $\bE=\curl\mathbf C$, $[\mathbf C]=\mathrm V$; only $\mathbf A$ is in
   V s/m, and the pair is homogeneous as $(c\mathbf A;\mathbf C)$, in V. Evidence s1-4. Corrected entry:
   "$\mathbf A$: $\mathrm{V\,s\,m^{-1}}$, $\mathbf C$: V". Also row $A_\mu$: the unit V s/m holds for $A_0=\phi/c$
   (s1-4); add "($A_0=\phi/c$)". Second look: no convention in the section makes C carry V s/m (E = curl C in Prop. E6(iii)).
   Survives.

4. **UNCLEAR** (kept, applied). Location: end of Section r1:sec:dirac, "positivity is a property of the quantised theory,
   whose state space Theorems~5.6--5.7 of \cite{ES17} construct from outside the real module". What is wrong: on that state
   space the one-momentum Hamiltonian has the spectrum {0, +-E, +-2E} (Theorem 5.7 of ES17), symmetric about zero; positivity
   needs the choice of the lowest state as vacuum and is not shown there. Evidence: ES17 Thm 5.7 (tex/s05_discrete.tex);
   s1-38 (the multiplicities 6,4,4,1,1 are those of sum_k eps_k (n_k - 1/2) with eps = (E,E,-E,-E)). Corrected wording:
   "the sign of the energy is decided only after quantisation: on the state space that Theorems 5.6-5.7 of ES17 construct
   from outside the real module the one-momentum Hamiltonian has the spectrum {0, +-E, +-2E}, symmetric about zero and
   bounded below, and a positive energy requires measuring it from the lowest state, the usual choice of vacuum."
   Second look: a reader of the original sentence infers that ES17's quantised theory has positive energy; it does not
   without that choice. Survives.

5. **WRONG-DETAIL** (kept, applied). Location: Section r1:sec:dirac, "Equation~(35) of \cite{ES17} uses kinetic kernels
   $K^\mu=P_*\gamma^0\gamma^\mu P_*$, with $K^0=+\id$ on its block, and a mass term of the opposite sign: its Lagrangian is
   minus \eqref{r1:eq:LD} on the corresponding block, with the same field equation." What is wrong: (35) lives on the
   rank-64 image of P_* in the three-constituent module R^512 and has no mass term mT; the fermion mass enters through the
   Yukawa term +(y/2) psi^T Phi psi. The sign statement holds with y Phi in the place of m T. Evidence: tex/s19_kernels.tex
   eq. (35) and Prop. "coupled field equations" (K^mu D_mu psi + y Phi psi = 0); s1-21 (one-constituent analogue).
   Corrected wording: "Equation (35) of ES17, on the rank-64 block of P_* in the three-constituent module, uses the kinetic
   kernels P_* gamma^0 gamma^mu P_* = -P_* K^mu P_*, with K^0 = +1 on its block, and the Yukawa term +(y/2) psi^T Phi psi in place
   of the mass term: with y Phi in the place of m T its fermion Lagrangian is minus (r1:eq:LD), and its field equation
   K^mu D_mu psi + y Phi psi = 0 has the same form." Second look: "on the corresponding block" does not supply the
   identification m T <-> y Phi; the statement needs it. Survives.

6. **UNCLEAR** (kept, applied). Location: proof of r1:thm:type(iii), "Since an intertwiner $\Lambda^2E\to V_\phi$ exists
   (r1-5)" and "The exact solution of the intertwining equations confirms all six dimensions for $\phi$ and for its parity
   twist". What is wrong: the theorem is stated for every isomorphism phi, but r1-5 computes only two isomorphisms, and for
   the parity twist only two of the Hom spaces. Evidence s1-6: the adjoint module of lG is isomorphic to V (Hom dimension 2),
   and phi intertwines ad_lint, which is Lambda^2E (the bivectors of E span lint), with ad_lG o phi; hence an intertwiner
   exists for every phi. All Hom dimensions were computed exactly for phi, its parity twist, a generic inner twist
   (rational Cayley transform) and their composite. Corrected wording: "An intertwiner Lambda^2E -> V_phi exists for every
   phi: the bivectors of E span lint, so Lambda^2E is the adjoint module of lint (Proposition 12.7 of ES17), phi intertwines
   it with the adjoint module of lG pulled back by phi, and the latter is isomorphic to V (r1-5)." The r1-5 line is
   extended to all Hom spaces for the twist and to the isomorphism ad_lG = V. Second look: the abstract argument above was
   absent from the proof; survives.

7. **CLASS** (kept, applied). Location: "Part (ii) is restated from \cite{ES17} (Propositions~12.2 and~12.6)". What is
   wrong: the last sentence of (ii), "The two subalgebras of so(3,3) meet in 0", is not in ES17 (Prop. 12.2 says only that
   they are not conjugate); it is fixed (r1-4; s1-5). Corrected wording: "Part (ii) is restated from ES17 (Propositions 12.2
   and 12.6) except its last sentence, which is fixed (r1-4)". Survives.

8. **CLASS** (kept, applied). Location: after r1:thm:L0, "Every computation in the theorem is fixed." What is wrong: the
   general identity D.H - E.B_t = P(L_S^2 - L_P^2 - 1) - 2 S L_S L_P of (iv) is displayed in Section 14 of ES17 (restated);
   the defects for L0, lambda L0, ModMax and S + E0 are fixed (s1-28 computes them directly without the formula). Corrected
   wording: "The general defect formula in (iv) is restated from Section 14 of ES17; every other computation in the theorem
   is fixed." Survives.

9. **UNCLEAR** (kept, applied). Location: r1:prop:free(iii) and its proof, "The timelike set $\{\eta>0\}$ and the shell
   ... are connected: the curve ... joins $e_1$ to $-e_1$". What is wrong: the curve shows only that e1 and -e1 lie in one
   component; ES17 Prop. 27.2 covers the timelike set, not the shell, and the proof gives no argument for connectedness.
   Evidence s1-11: the shell is the image of S^2 x R^3 under (w,v) -> (sqrt(M^2c^2+|v|^2) w, v), with inverse
   (u,v) -> (u/|u|, v); {eta > 0} retracts onto the unit sphere of V^+ along (u,(1-s)v). Corrected proof sentence added.
   Survives.

10. **WRONG-DETAIL** (withdrawn). Location: after r1:prop:mix, "carried by the gradient $m\kappa\eta GY$ of a boundary term"
    for the rotation of the raised pi. Proposed: pi - mY' = eta^{-1} grad Lambda = m kappa G Y. Second look: the gradient is
    a covector, and the section uses pi = eta^{-1}p throughout; identifying a covector with its raised vector by eta is the
    convention of r1:prop:shift(i), and (ii) of that proposition is stated for p. Nothing is wrong; withdrawn (s1-13 confirms
    both forms).

11. **WRONG-DETAIL** (kept, applied). Location: r1:prop:mix(ii), "$\pi(0)^+=-m\kappa\sum_ax^a_0e_a\neq0$". What is wrong:
    the temporal part vanishes for Y0 = 0, which is exactly the case of the example pi(0) = e5 given in the same item.
    Evidence s1-15. Corrected: "$\neq0$ for $Y_0\neq0$". Survives.

12. **WRONG-DETAIL** (kept, applied). Location: section opening, "which checks every finite-dimensional and symbolic claim
    in exact arithmetic", against the proof of r1:prop:shift(iii), "computed in floating point with tolerance $10^{-9}$
    (r1-12)". What is wrong: the two sentences contradict each other. Evidence: s1-14 computes both inertias exactly
    ((6,6,0) at two parameter points). Correction: r1-12 now computes the inertia exactly (exact eigenvalues), and the proof
    sentence says so. Survives.

13. **CLASS** (kept, applied). Location: r1:rem:fmb1, "Definition~2.1 of \cite{FMB} is confirmed with one correction: its
    surplus coordinates $u^a$, there said to be ``also length-valued'', are the converted times". What is wrong: FMB's
    statement is true (the converted times are lengths); the remark adds a precision, not a correction. Corrected wording:
    "is confirmed, with one precision". Second look: nothing in FMB Def. 2.1 claims the u^a are intrinsic lengths. Survives.

14. **UNCLEAR** (kept, applied). Location: Answer item (4), "$E_0$ is a Lorentz-invariant conformal Lagrangian with
    indefinite energy and no duality invariance (fixed)". What is wrong: r1:thm:L0(i) shows E0 pointwise duality invariant;
    what fails is duality invariance in the sense of Gaillard and Zumino. Corrected wording: "not duality invariant in the
    sense of Gaillard and Zumino, although pointwise invariant". Survives.

15. **UNCLEAR** (kept, applied; author's instruction of 25 October). Location: section opening, "the second the pair
    $(\bE,\bB)$"; r1:def:em. The field reading is X_EM = (B,E), a pair of three-vectors labelling the six components
    F_{0i} = E_i, F_{jk} = eps_{ijk} B_i of the field two-form, not a sum E + B. r1 nowhere reads it as a sum (checked: no
    occurrence of E+B or B+E as a sum), but the opening lists the pair in the opposite order to r1:def:em, (cB;E). Correction:
    opening "the pair $(\bB,\bE)$ of three-vectors", and in r1:def:em the explicit statement that X_EM is the pair
    labelling the six components of F, not a sum. s1-1 (G is the Hodge star on F_X) and s1-22 (orientation) cover it.

## Step 3: applied

Section edits: findings 1-9 and 11-15 applied as worded above (finding 10 withdrawn). Derivation script
`verify/r1_lagrangians.py` updated: r1-3 (Heaviside-Lorentz field has dimension m^-2, ell^3 converts it), r1-5 (all Hom
spaces for the parity twist, Hom(Lambda^2E,E) = 0, Hom_lG(ad_lG,V) = 2: existence for every phi), r1-9 (connectedness of
{eta > 0} and of the shell), r1-12 (exact inertia from exact eigenvalues; docstring no longer mentions floating point),
r1-13 (claim text: temporal part nonzero exactly when Y0 != 0), r1-16 (sign pattern of ES17 eq. (35) on one constituent),
r1-36 ([C] = V, (cA;C) in V, phi/c in V s/m), new r1-37 (ES17 Thm 5.7 spectrum, exact characteristic polynomials).
Runs: r1_lagrangians.py 37/37 PASS (exit 0, 12 s); s1_check.py 38/38 PASS (exit 0, 52 s).
Compile (scratchpad/compile_r1_v, bib_r1/r2/r3 merged with each key once, 27 entries appended, pdflatex x2): no errors,
no overfull boxes, no undefined references from r1 (the only undefined reference, r4:sec:main, comes from r3_octants.tex);
r1 occupies pages 1-10; two pre-existing underfull boxes in r1 (Prop. r1:prop:E6(i), Prop. r1:prop:total(i)).
