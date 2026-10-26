# Referee findings: sections/r2_fourmomentum.tex (four-momentum with X on the spatial triple)

Independent script: investigations/two_readings/verify/s2_check.py (27 PASS lines, s2-0a ... s2-20; it imports nothing from
r2_fourmomentum.py). It uses different methods from the derivation script: Clifford generators from the bit-string rule of
ES17 App. A, Cayley-transform frames instead of quaternion frames, intertwiner spaces as null spaces of Kronecker systems,
exact inertia by Descartes' rule, field integrals by quadrature, and field boosts by an explicit Lorentz transformation of
(E, B) and of the coordinates.

What was checked and survives without a finding: every formula, sign and unit of Def. config, Props planes, cov, J, Thm i,
the bracket table, Props cons, shellI, RE, shellII, shellIII, rotate(i),(iii), dict, nolinear, stress, whole(i), Lemma anti,
Props nogo, leak, twoalg; the momentum-density sign Theta^{0a} = -(E x B)_a = (E x B_t)_a with the paper's B (s2-13); the boost
law Theta -> L Theta L^T with L^0_3 = +sinh chi, i.e. the frame moving with -tanh chi e3 (s2-14b); phi(L_i) = s_j^s_k,
phi(K_i) = s_i^n, unique (s2-12a); F_{GX} = star F_X with eps_0123 = +1 (s2-12b); every FMB number (Def. 3.1, Prop. 3.2,
eqs. (6)-(10), Prop. 4.1, Prop. 5.1, Lemma 6.1, Prop. 6.2, eqs. (16), (17), (20), Sections 5-7) against the FMB source; every
ES17 number against paper_labels.txt and the cited text, and every RE number against re_labels.txt and the cited text; the
nine literature entries (verbatim from tex/refs.tex; Laue 1911, Rainich 1925 (Trans. AMS 27, 106-136) and Wigner 1939
(Ann. Math. 40, 149-204) confirmed by web search, the books are standard) and what they are cited for.

## Findings

1. **WRONG-DETAIL** Prop. r2:prop:rotate(ii): "With $\boldsymbol{\mathsf E}^+$ the componentwise positive part,
   $\max E_n=|\boldsymbol{\mathsf E}^+|$, attained at $w_i=(\mathsf E_i^+)^2/|\boldsymbol{\mathsf E}^+|^2$." The statement needs
   the hypothesis E^+ != 0 (the weights are 0/0 otherwise), and it is false when every component of E is negative. Evidence
   (s2-10b): E = (-1,-2,-3) has octant maximum -1 = max_i E_i, at the vertex e1, not |E^+| = 0. Exactly: for n_i >= 0,
   |n| = 1 one has sum_i n_i >= 1, so n.E <= max_i E_i when max_i E_i <= 0. Proposed: "If $\boldsymbol{\mathsf E}^+\neq0$,
   $\max E(n)=|\boldsymbol{\mathsf E}^+|$, attained at ...; if $\boldsymbol{\mathsf E}^+=0$ the maximum is $\max_i\mathsf E_i\le0$,
   at a vertex, and no clock of the octant reads a positive energy."
   Second look: kept. A second computation (20000 Dirichlet-sampled octant clocks, and 50 random all-negative E) confirms it.
   Applied (statement, proof and the new check r2-10d).

2. **WRONG-DETAIL** Prop. r2:prop:whole(ii): "two null wholes with momenta at angle $\vartheta$ have
   $M^2c^4=2U_1U_2(1-\cos\vartheta)$, timelike unless the momenta are collinear". Antiparallel momenta are collinear and give
   M^2 c^4 = 4 U1 U2 > 0. Evidence (s2-16): (3;0,0,3) + (4;0,0,-4) has M^2 = 48. The packets of (iii) are such a case:
   their three-momenta point along +z and -z, and M > 0. The same word recurs in "The mass of the whole comes from the
   non-collinearity of the two parts" and in the Answer ("with the mass of a whole from non-collinear parts"); this is
   right only for the four-momenta (RE's wording is "the non-collinearity of its constituents' energy-momenta").
   Proposed: "timelike unless $\vartheta=0$, that is unless the two four-momenta are proportional", and "from the
   non-collinearity of the two parts' four-momenta" (Answer: "from parts whose four-momenta are not collinear").
   Second look: kept. 1 - cos(pi) = 2. Applied (text, and r2-16 extended).

3. **WRONG-DETAIL** Prop. r2:prop:whole(iii): "so $M=\tfrac{512}{315}ab$". M is non-negative. For ab < 0 the formula gives a
   negative number. Evidence (s2-15): quadrature gives M = (512/315)|ab| for (a,b) = (0.7,-1.1) and (-2,0.5). Proposed:
   "$M=\tfrac{512}{315}|ab|$". Second look: kept (M^2 = U^2 - P_z^2 = (512/315)^2 a^2 b^2). Applied (text, and r2-15).

4. **WRONG-DETAIL** Rem. r2:rem:prefer: "whose unstable band is exactly the set removed by Option II". Option II imposes
   q = 0, so it removes every nonzero surplus momentum, including the stable ones with q^2 <= |k|^2 + m^2. The band
   q^2 > |k|^2 + m^2 is a proper subset of what is removed. Evidence (s2-8): k = m = 1, q = 1/2 has omega^2 = 7/4 > 0 and is
   removed. ES17's own text says "remove exactly the unstable band" (Sec. 27), but its Fig. 19 caption is precise ("keep
   exactly the point q = 0"). Proposed: "whose unstable band Option II removes, together with every other nonzero surplus
   momentum (Figure 19 and Theorem 30.1 there)". Second look: kept. Applied.

5. **OVERCLAIM** Rem. r2:rem:prefer: "Option~III needs a functional of the sign opposite to the metric, which no structure of
   the arena supplies." The section's own Prop. r2:prop:shellIII(ii) refutes this. Once the clock is chosen, as it already is
   for Options I and II, the arena supplies the twisted form eta(R_n.,.), with R_n = P_n - Q_n = exp(pi u1^u2) in SO(3,3), of
   signature (1,5), which is exactly the toy's form. Evidence: s2-9 (two Cayley clocks) and r2-9b. What the arena does not
   do is select it over eta. Proposed: "Option~III needs a functional whose surplus sign is opposite to that of the metric;
   the clock supplies one, the twisted form $\eta(R_n\cdot,\cdot)$ of \cref{r2:prop:shellIII}(ii), but nothing in the arena
   selects it in place of $\eta$." Second look: kept. The remark itself contrasts "the arena's metric" with the alternatives,
   so the clock-built form counts as arena structure. Applied.

6. **CLASS** (status of an FMB statement) after Prop. r2:prop:cons: "Part (iii) completes Proposition~3.2 of \cite{FMB} and
   corrects its limitation that nothing proves $E/c$ to be a Noether energy", and the table entry "limitation corrected".
   FMB's limitation ("Neither this argument nor the axiom selects a dynamics or establishes conservation ... Nor does it
   prove that E/c is a Noether energy") is accurate: the axiom fixes no dynamics. Part (iii) supplies the conditional
   statement (Noether charge of every dynamics invariant under the translations of E_n). The FMB statement is therefore
   confirmed and completed, not corrected. Proposed: "Part (iii) completes Proposition~3.2 of \cite{FMB}. Its limitation, that
   the axiom alone does not make $E/c$ a Noether energy, is confirmed, since the axiom fixes no dynamics. (iii) supplies
   what is missing: ...". Table: "limitation confirmed; completed". Second look: kept (FMB's text re-read). Applied.

7. **CLASS** (status of an FMB statement, and claim class) after Prop. r2:prop:nolinear: "Part (ii) is fixed by computation
   (r2-11b) and corrects that section in one respect: \cite{FMB} says that field momentum ``can be quadratic in the field
   and integrated over space'', but quadratic is not enough." FMB says "quadratic in the field AND integrated over space",
   which is exactly what the section establishes, so FMB is confirmed and sharpened, not corrected. Part (ii) is also as
   standard as part (i): Sym^2((1,0)+(0,1)) = (2,0)+(0,2)+(1,1)+2(0,0), contracted against (1/2,1/2). Its class is
   inherited (computed), not fixed. Table: "no quadratic map to a four-vector: fixed / corrected". Proposed: "Part (ii) is
   inherited as well, computed (r2-11b), and sharpens that section: \cite{FMB} says that field momentum ``can be quadratic
   in the field and integrated over space'', and both halves are needed. Quadratic alone is not enough: ...". Table:
   "inherited / confirmed, sharpened". Second look: kept. Applied.

8. **CLASS** (and OVERCLAIM) Rem. r2:rem:base: "supplies no extra time components of momentum, as Section~7 of \cite{FMB}
   states (confirmed; fixed, since the four-momentum of \cref{r2:prop:whole} has its index in $E_n$)", and the table row
   "$X_{\rm EM}$ supplies no time components / fixed / --". The reason given is the construction itself, because the field
   is defined on the base E_n, so its translation charges carry E_n indices. That is not a theorem that could have failed.
   Covariance alone does not forbid such components: dim Hom_{lG}(Sym^2 V_EM, U_n) = 4, since U_n is a trivial module and
   the two invariants |B|^2 - |E|^2 and E.B can be placed on it (s2-11b). Proposed: "(confirmed; constructed: the field is
   a function on the base $E_n$, so its translation charges carry indices of $E_n$. Covariance alone would allow more, since
   the invariants $|\bB|^2-|\bE|^2$ and $\bE\cdot\bB$ map covariantly to the trivial module $U_n$, r2-11b.)". Table class:
   "constructed". Second look: kept. A covariant map into U_n is not a translation charge, so FMB's statement stands as a
   statement about momentum. Only the class changes. Applied (text, and r2-11b extended).

9. **CLASS** end of Sec. r2:sec:nogo: "This is fixed (r2-12b with \cref{r2:prop:nogo})", and the table row "$G=\star$ on
   $\Lambda^2E_n^*$ not obstructed / fixed". That G acts as the Hodge star of a Lorentzian 2-form module, of neutral
   signature, is Theorem 12.3 of ES17 (and inherited: star^2 = -1 on Lorentzian 2-forms). Compatibility with the no-go is the
   neutral case P = Q of ES17 Thm 16.2. Nothing new could fail here. Proposed: "This is restated (Theorem~12.3 and
   Theorem~16.2 of \cite{ES17}; r2-12b with \cref{r2:prop:nogo})"; table "restated". Second look: kept (minor). Applied.

10. **UNCLEAR** The symbol E_n denotes both the four-plane E_n = Rn + V^- (Def. r2:def:config and throughout) and the energy
    read by the clock, E_n := c eta(K,n) (Sec. r2:sec:rotate, Prop. r2:prop:rotate, Rem. r2:rem:projection, and Prop.
    r2:prop:J(iv): "$E_{-n}=-E_n$ (the energy read)", where as four-planes E_{-n} = E_n). The reader cannot tell
    "$E_n\le|\boldsymbol{\mathsf E}|$" from a statement about planes. Proposed: write the energy read by the clock as E(n)
    (and E(W) at W). Second look: kept. Applied, together with the PASS texts of r2-3d and r2-10a.

11. **UNCLEAR** Intro item (3): "a four-dimensional mass $m_4(n)\le\mu$, with equality exactly at the clock aligned with the
    temporal momentum", and the table "$m_4(n)\le\mu$, equality at the aligned clock". m_4(n) may be imaginary
    (m_4^2 = -5 at e1 in the example), so only the squares compare. Equality holds iff n is parallel or antiparallel to E:
    the clock -E/|E| also reads m_4^2 = mu^2, with energy -|E| (s2-10a). The proposition itself is stated correctly
    ("$m_4(n)^2\le\mu^2$, with equality iff $n\parallel\boldsymbol{\mathsf E}$"). Proposed: "$m_4(n)^2\le\mu^2$, with equality
    exactly at the clocks parallel to the temporal momentum (the aligned one reads the largest energy)"; table likewise;
    Rem. r2:rem:projection "equal to the six-dimensional $\mu$ only for the clocks parallel to it". Second look: kept.
    Applied (text, and r2-10a PASS text and check).

12. **WRONG-DETAIL** after Prop. r2:prop:shellI: "the normal plane of a Lorentzian four-plane in $\R^{3,3}$ has the sign of
    time (Theorem~28.1(ii) of \cite{ES17})". Theorem 28.1(ii) says this for complements of signature (1,3) only. A four-plane
    of signature (3,1), also Lorentzian, has a negative normal plane: <e1,e2,e3,e4> has normal <e5,e6> (s2-1). Proposed:
    "the normal plane of a four-plane of signature $(1,3)$ in $\R^{3,3}$ has the sign of time". Second look: kept (minor;
    ES17 Thm 30.4 uses the same loose wording). Applied.

13. **CITATION** proof of Thm. r2:thm:i: "the square of $H_n$ is that of Example~4.7 of \cite{ES17} with $\gamma^0=n$". The
    general identity H(p)^2 = E^2 is equation (8) of ES17 (eq:Hp). Example 4.7 is its numerical instance at p = (0,0,3),
    m = 4. Proposed: "is that of equation~(8) of \cite{ES17}". Second look: kept. Applied.

14. **CITATION** after Prop. r2:prop:shellII: "in Dirac quantisation the physical states are functions on $E_n$ obeying the
    Klein--Gordon equation (Corollary~30.3)". The Dirac-quantisation statement (physical states are functions on spacetime,
    with H Psi = -(1/2) Box Psi) is in Theorem 30.1. Corollary 30.3 is the reduction of the scalar sector to the
    Klein-Gordon field. Proposed: "(Theorem~30.1 and Corollary~30.3)". Second look: kept. Applied.

15. **WRONG-DETAIL** Conventions: "As in Sections~2, 3 and 12 of \cite{ES17}: ... the para-complex $\Kc e_i=e_{i+3}$,
    $\Kc e_{i+3}=e_i$". ES17's grading root (Prop. 3.3, Lemma 3.4, App. A) is $\cK e_i=-e_{i+3}$, $\cK e_{i+3}=-e_i$, so
    the section's K is minus ES17's. Evidence (s2-0a): (1/2)ad_P with P = sum omega_i equals -K. K is not used elsewhere in the
    section. Proposed: add "(this is $-\cK$ for the grading root $\cK$ of Proposition~3.3 of \cite{ES17})". Second look: kept
    (minor). Applied.

16. **UNCLEAR** (house style) Rem. r2:rem:minus(b): "which the author's older monograph proposed as the carrier of the
    Lorentzian interior". The note must be self-contained, and the monograph is not a citable authority. The construction
    should be reported as proposed, as r3 does ("was proposed in an earlier, superseded version of the programme"). The
    claim itself is accurate: monograph Remark 2.10 and Fig. 2.2 call the all-minus sign sector of the temporal R^3 the
    carrier of the Lorentzian (1,3) interior. Proposed: "which was proposed, in an earlier and superseded version of the
    programme, as the carrier of the Lorentzian interior". Second look: kept. Applied.

17. **UNCLEAR** Rem. r2:rem:projection: "The four-dimensional mass is the projection of the undivided temporal energy onto
    the clock". The projection is the energy E(n). The mass follows from it through m_4^2 c^4 = E(n)^2 - |p|^2 c^2. Proposed:
    "The energy read by the clock is the projection of the undivided temporal energy onto it; the four-dimensional mass
    computed from it is maximal, and equal to the six-dimensional $\mu$, only for the clocks parallel to
    $\boldsymbol{\mathsf E}$, and lowered by every misalignment". Second look: kept. Applied.

18. **WRONG-DETAIL** Prop. r2:prop:cons(iii): "$q_a$ is conserved only if $H$ does not depend on $u$". The bracket table gives
    {q_a, H} = -dH/du^a. So q_1 is conserved by every H that depends on u^2 but not on u^1, and the "only if" fails for each
    single q_a. Evidence (s2-5). Proposed: "$q_a$ is conserved iff $H$ does not depend on $u^a$". Second look: kept. Applied
    (statement, and r2-5b check and text).

## Withdrawn on second look

19. (withdrawn) Candidate sign slip in "$\Theta^{0a}=(\mathbf S_{\rm t})_a=-S_a$" with the paper's B. Refuted by s2-13, which
    computes Theta directly from F_X: Theta^{0a} = -(E x B)_a = (E x B_t)_a. The section and the derivation agree, and so
    does RE Prop. 2.5.
20. (withdrawn) Candidate CITATION in Prop. r2:prop:twoalg: "The first two statements restate Proposition~12.2". Read as
    "lint(n) fixes U_n pointwise" and "lG fixes no nonzero vector", both of which are in ES17 Prop. 12.2. The intersection
    lG cap lint(n) = 0 falls under "the rest is fixed". No error.
21. (withdrawn) Candidate UNCLEAR in Prop. r2:prop:shellII: "the reduced phase space is $T^*E_n$, of dimension $12-2\times2=8$,
    with the shell". The mass-shell constraint is first class too, but the sentence describes the reduction by q only, in
    the same words as ES17 Prop. 23.4. No error.
22. (withdrawn) Candidate CLASS: Thm. r2:thm:i labelled "fixed" although (i), (ii) are transports of ES17 Thm 4.1. Part (iii),
    the clock-independence of the mass form, could have failed, and each part is checked directly. Acceptable.
23. (withdrawn) Candidate UNCLEAR: "The surplus momenta of \cite{FMB} are the relative-time energies of \cite{RE} divided by
    $c$". RE defines relative-time energies at the clock W. The general relation c q_a = u_a.E is displayed in Prop. r2:prop:RE
    and the clock-W case is stated next to it, so the sentence is not misleading.
