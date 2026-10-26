# Referee findings for `investigations/two_readings/sections/r4_bridge.tex` (track r4, the bridge T = nJ(n))

Independent script: `investigations/two_readings/verify/s4_check.py` (24 checks, all PASS, about 3 s). It does not import or
copy `r4_bridge.py` and uses other methods: an abstract blade algebra with sympy coefficients for identities in a general
clock; integer matrices with integer-scaled rational clocks; dimensions bounded above by the rank modulo p = 2^31 - 1 and
below by exact solutions that are exhibited and shown independent; linear identities checked on bases.

## Claims the derivation flagged as surprising: all confirmed independently

- The Pauli term keeps its sign under an odd temporal permutation. Confirmed (s4-15, all six permutations at four clocks).
  Precisely: under the temporal permutation alone neither M_n nor M*_n changes sign while T does. Under the paired
  permutation M_n keeps its sign and M*_n changes it. The reason is that M_n = -n S_n does not contain T at all.
  The section states (iii) correctly, but its Answer (3) does not (finding 3).
- Two clocks do not single out T and three do. Confirmed (s4-19): the nullity mod p is 1 for three clocks and 2 for two,
  with the solutions exhibited. This is strengthened: for two clocks a whole circle (3T + 4 w e4e5e6)/5 of products
  "clock times a square root of -1 commuting with the gamma's" exists, so Theorem (iii) also needs three clocks (finding 9).
- rho_oct(L_i) = rho_n(L_i) - J(e_i)/2. Confirmed (s4-9), as an exact matrix identity for every clock, since rho_n(L_i) = s_js_k/2
  does not depend on the clock.
- C = (clock half-turn) x T. Confirmed (s4-6): R = n u2 is the half-turn about u1, it reverses n, fixes T, reverses J(n), and
  R T = u1. The antilinear Dirac intertwiners at each of four clocks are exactly U_n (nullity 2). Every unit C in U_n is
  (the half-turn about C) times T (finding 15, wording).
- Dimensions 2 (Grassmann) and 6 (commuting) of the trilinear couplings. Confirmed exactly (s4-13). The 2304 x 384 system has
  nullity 8 mod p. The eight kernels n C S_n(X) are exact independent solutions, so the dimension is 8 over Q. Two of them are
  antisymmetric (C = 1, Omega_n) and six symmetric. This holds at e1, (2,3,6)/7 and (-1,4,8)/9. It also agrees with the
  invariant forms on S (2 symmetric, 6 antisymmetric), consistent with Sym^2 S containing 2 singlets and Lambda^2 S
  containing 6.
- Commutant dimensions 8, 6 and 1. Confirmed (s4-10), with exact bases exhibited at three clocks.

## Findings

1. **WRONG-DETAIL**: before Theorem "Covariant couplings", the phrase "under the dictionary of \cref{r2:prop:dict} it is the
   Clifford image of $F_X^\sharp=\sum_{\mu<\nu}F_X^{\mu\nu}\gamma_\mu\wedge\gamma_\nu$".
   - What is wrong: the note only has the frame vectors gamma^mu_n = (n, s_1, s_2, s_3). With the standard lowering
     gamma_mu = eta_{mu nu} gamma^nu_n, the stated sum equals S_n(eta X), so its electric part has the wrong sign. The correct
     bivector is sum_{mu<nu} (F_X)_{mu nu} gamma_mu ^ gamma_nu = sum_i (E_i s_i^n + B_i s_j^s_k), which equals
     F^{mu nu} gamma^mu_n ^ gamma^nu_n with mixed index placement. The stated formula is right only if gamma_mu is read as the
     frame vector gamma^mu_n, a reading the note never introduces.
   - Evidence: s4-14 checks both sums against S_n(X) and S_n(eta X) at four clocks; a hand computation gives F^{0i} = -E_i.
   - Proposed wording: "it is the Clifford image of the bivector $\sum_{\mu<\nu}(F_X)_{\mu\nu}\gamma_\mu\wedge\gamma_\nu=
     \sum_i(E_i\,s_i\wedge n+B_i\,s_j\wedge s_k)$, $\gamma_\mu:=\eta_{\mu\nu}\gamma^\nu_n$; the same sum with $F_X^{\mu\nu}$ is
     $S_n(\eta X)$".
   - Second look: kept, since the sign is wrong under the only index convention the note uses. **Applied.** The derivation
     script's r4-16 now checks the corrected statement.

2. **OVERCLAIM**: Answer (3), the phrase "the only covariant local couplings without derivatives are trilinear".
   - What is wrong: covariant, derivative-free couplings of higher order in X exist, for example
     $(\psi^{\!\top}T\psi)\,\eta(X,X)$. T is a rho_n-invariant antisymmetric form, so psi^T T psi is nonzero for a Grassmann psi,
     and eta(X,X) is l_G-invariant. The theorem only classifies kernels that are linear in X and quadratic in psi.
   - Evidence: s4-23.
   - Proposed wording: "the covariant local couplings without derivatives that are linear in the field strength are
     trilinear, quadratic in psi: ...; couplings of higher order in X exist, such as $(\psi^{\!\top}T\psi)\eta(X,X)$".
   - Second look: kept; the counterexample is checked directly. **Applied**, with new derivation check r4-25.
   - The same overclaim appears in `sections/abstract.tex`: "couples to the field octants covariantly only through the Pauli
     term and its dual". That file was not edited (outside the files this referee may change) and is flagged for the
     coordinator. `r5_synthesis.tex` says "the trilinear couplings are exactly ...", which is correct.

3. **WRONG-DETAIL**: Answer (3), the phrase "an odd permutation reverses the electric-dipole term with $T$ but not the Pauli term".
   - What is wrong: under the temporal permutation alone the electric-dipole term does not change sign either. Only the
     paired odd permutation reverses it, as Proposition coupclock(iii) states.
   - Evidence: s4-15.
   - Proposed wording: "the paired odd permutation reverses the electric-dipole term with $T$ but not the Pauli term, the
     temporal one alone neither".
   - Second look: kept. **Applied.**

4. **WRONG-DETAIL**: introduction item (1) and Answer (1).
   - Location: item (1) says "an odd permutation of the temporal axes reverses $T$ and $J(n)$ together, and with them the mass
     form and the duality operator, and turns the octant map into its textbook-orientation form". Answer (1) says "reversed
     together with $J(n)$, and with the duality operator, by odd permutations".
   - What is wrong: the statements about the duality operator and the octant map hold only for the paired permutation. The
     temporal permutation alone breaks G and the octant structure (Prop. perm (ii), (iii)), so after it there is no duality
     operator of the octant map to reverse. Proposition perm and the discussion after it state this correctly.
   - Proposed wording: "...hence the mass form; paired with the same permutation of the spatial axes, the only form that
     respects $G$, it also reverses the duality operator and turns the octant map into its textbook-orientation form".
   - Second look: kept, minor (A_sigma(-T) = T holds formally, but A_sigma does not preserve Phi or G). **Applied** in both
     places.

5. **UNCLEAR** (house rule: the note must be self-contained): the passage refers to expectations and requests that the note
   never states.
   - Locations: "The statement asked for is confirmed in the form (i)"; "written through $T$ and $J(n)$ as asked"; "The
     expectation that the couplings change sign with $T$ ..."; "a correction to the expectation that two clocks suffice";
     Answer (3) "the sign expectation refuted"; and the table row r4-19 "expectation refuted".
   - What is wrong: these come from the track prompt, not from the note. The "two clocks suffice" expectation appears
     nowhere, not even in the prompt.
   - Fix: state the three expectations once in the section introduction (an odd permutation reverses the mass form and the
     duality together; the couplings, which contain T, change sign with it; the Dirac-field coupling can be written through T
     and J(n)), refer to them by number, drop "as asked", and say plainly that two clocks do not suffice.
   - Second look: kept. **Applied.**

6. **WRONG-DETAIL**: Proposition Jdec(ii), the phrase "$\rho_n(L(n))=\tfrac12\sum_in_is_js_k$ is the Dirac spin rotation about
   $n$"; the comment after it ("about the clock axis the field rotation is ..."); Answer (4) ("the field rotation about the
   clock axis").
   - What is wrong: rho_n(L(n)) rotates V^- about the spatial axis sum_i n_i s_i, the axis paired with n, and acts trivially on
     V^+, the clock included. A Dirac rotation "about the timelike clock n" is not meant and does not exist. Only the field
     rotation L(n) turns B, on V^+, about n.
   - Evidence: s4-9 checks the action of rho_n(L(n)) on V.
   - Proposed wording: "the Dirac spin rotation of $V^-$ about the spatial axis $\sum_in_is_i$ paired with $n$".
   - Second look: kept. **Applied.** The derivation script's r4-10 now checks the action on V.

7. **WRONG-DETAIL**: remark after Theorem "The three roles of T", the phrase "Of the three roles only (c) belongs to the field
   reading".
   - What is wrong: role (b) begins with Ad_T|_V = eta, "the frame parity fixing B and reversing E". That is the field
     reading's parity (Proposition 2.11 of RE), and on the lengths it is also the classical parity. So (b) belongs to both
     readings, and it too uses T alone.
   - Proposed wording: "Role (a) belongs to the particle reading and uses $T$ as the product $nJ(n)$; (c) belongs to the field
     reading and uses $T$ alone; (b) belongs to both".
   - Second look: kept. **Applied.**

8. **UNCLEAR** (domain mismatch).
   - Locations: "the joint action of $Y\in\lint(n)$ is $\rho_n$ on $\psi$ and $\rho_{\rm oct}\circ\phi_n^{-1}$ on the octants",
     and Theorem couplings, "Let $\psi$ carry $\rho_n$ and $\Phi(X)$ carry $\rho_{\rm oct}\circ\phi_n^{-1}$".
   - What is wrong: rho_n = spin o phi_n is a representation of l_G, while rho_oct o phi_n^{-1} is a representation of
     lint(n). The invariance condition actually used is that of l_G acting by rho_n(Y) on psi and by Y.X on X.
   - Evidence: s4-12 and s4-13 compute with that condition.
   - Proposed wording: "the spin lift $\rho_n(\phi_n^{-1}Y)$ on $\psi$ and $\rho_{\rm oct}(\phi_n^{-1}Y)$ on the octants", and in
     the theorem "Let $Y\in\lG$ act by $\rho_n(Y)$ on $\psi$ and by $\rho_{\rm oct}(Y)$ on $\Phi(X)$, that is by $Y\cdot X$ on $X$".
   - Second look: kept; the computations are right and only the notation is not. **Applied.**

9. **UNCLEAR** (proof gap): Theorem unique, proof of (iii), "By Theorem~4.1 of \cite{ES17} at every clock the Dirac complex
   structures are $\pm J(n)$".
   - What is wrong: item (iii) defines a Dirac complex structure as "a square root of $-\id$ in the commutant of the
     $\gamma^\mu_n$". That set is a two-sphere in H (Remark 4.2 of ES17), not ±J(n). Theorem 4.1 needs the extra condition of
     commuting with the internal bivectors. The statement is still true: by (i), Y lies in R T, so n^{-1}Y = aJ(n), and
     squaring to -1 gives a = ±1. But with two clocks there is a circle of solutions, so (iii) needs three clocks just as
     (i) does.
   - Evidence: s4-19 (clocks e1, (0,3,4)/5).
   - Proposed proof: as above, with the two-clock circle stated.
   - Second look: kept. **Applied**, and the derivation script's r4-23 now checks the circle.

10. **UNCLEAR** (proof covers one clock only): Proposition triples(iii), proof "$e_a\alpha_a(e_1)$ takes three different values".
    - What is wrong: the claim is made for every clock, but the proof (and r4-20) treats n = e1 only.
    - General argument: Z = n s_a e_a for all a would need s1e1 = s2e2. Z = e_a n s_a would need e1ns1 = e2ns2. The grade-one
      part of that equation, n1 s1 - n2 s2, forces n = ±e3, and then e1e3s1 ≠ e2e3s2.
    - Evidence: s4-16, symbolic in n.
    - Second look: kept, minor. **Applied** in the proof, and the derivation script's r4-20 now checks it symbolically.

11. **UNCLEAR** (proof gap): Theorem conj, proof "If $U$ had definite parity, $U\alpha_aU^{-1}$ would be even".
    - What is wrong: the argument covers invertible U only, while the claim is about every nonzero solution.
    - Fix: "$U\alpha_a$ would have the parity of $U$ and $e_aU$ the opposite one, so both vanish, and $U=0$".
    - Evidence: s4-17 gives nullity 0 on each parity.
    - Second look: kept, minor. **Applied.**

12. **UNCLEAR** (wording): Theorem conj, the phrase "form an eight-dimensional space, a coset of the commutant".
    - What is wrong: the solution set is the linear space C U_0 (U_0 any invertible solution), not an additive coset of C.
    - Proposed wording: "the space $\mathcal C\,U_0$ for any invertible solution $U_0$, where $\mathcal C$ ... is the commutant".
    - Second look: kept, minor. **Applied.**

13. **OVERCLAIM**: paragraph "On the working note", the phrase "its item~5 of Section~8 is answered by \cref{r4:prop:Jdec} and
    \cref{r4:sec:reps}".
    - What is wrong: item 5 of FMB Section 8 also asks for "a common, fully compatible Maxwell--Dirac action and projector
      vacuum ..., especially across the mixed/spectator embedding mismatch". The section does not treat the projector vacuum
      or the mismatch.
    - Fix: "answered in part ...; the projector vacuum and the mixed/spectator mismatch named there are not treated here
      (open)".
    - Second look: kept (checked against the FMB source). **Applied.**

14. **CITATION / cross-section** (r3 versus r4):
    - Agreement: r3's paragraph "The main obstruction" and its remark after its Maxwell theorem agree with r4. The Dirac boosts
      are even and the octant boosts odd; the Dirac operator couples ∂_a to s_a through gamma^0 gamma^a, and the octant
      operator couples it to e_a (s4-24).
    - Disagreement: r3's paragraph "Open" still lists "whether rho_oct ... can enter a coupling to the Dirac sector" as open,
      while r4's Theorem couplings answers it (yes, by the Pauli and dual couplings in which Phi(X) carries rho_oct).
    - Fix in r4: the introduction now says that this question is answered here too. r3 was not edited (outside the files this
      referee may change), so the coordinator should replace that item of r3's "Open" paragraph by a pointer to
      \cref{r4:thm:couplings}.
    - Second look: kept. **Applied in r4 only.**

15. **UNCLEAR** (wording): Proposition perm(iv), the phrase "every unit vector $C$ of the surplus plane ... is the half-turn
    reversing the clock composed with $T$".
    - What is wrong: there is a circle of half-turns that reverse the clock. The one that gives C is the half-turn about C
      itself.
    - Proposed wording: "is the half-turn about $C$, which reverses the clock, composed with $T$".
    - Evidence: s4-6, with C = (3u1+4u2)/5. The check of the antilinear intertwiners at a general clock is s4-6 as well.
    - Second look: kept, minor. **Applied.**

## Withdrawn after the second look

16. *withdrawn*. CITATION: "that a Dirac mass sign is not observable alone is inherited \cite{ItzyksonZuber}" gives no locus.
    - Reason for withdrawal: it is the standard chiral rotation psi -> gamma_5 psi, an inherited textbook fact. The arena
      version is proved in the corollary itself (Omega_n reverses T and keeps every K^mu, s4-7). No error.

17. *withdrawn*. WRONG-DETAIL suspicion: with the textbook gamma^mu identified with gamma^mu_n, sigma^{mu nu}F_{mu nu} maps to
    a multiple of J(n) S_n(eta X), not of J(n) S_n(X) (s4-14).
    - Reason for withdrawal: the text says "up to normalisation and the sign convention for $\gamma^a$", and gamma^a -> -s_a is
      exactly that convention. The identification of M_n with the Pauli term and of M*_n with the electric-dipole term stands,
      and their parities agree (s4-14).

18. *withdrawn*. UNCLEAR suspicion: "the diagonal rotations move the clock as well and are symmetries of both".
    - Reason for withdrawal: the meaning is covariance, with (n, psi, X) going to (Rn, R psi, RX) (Prop. coupclock(ii) and
      clockdep). This is verified in s4-15.

19. *withdrawn*. Suspicion that psi -> T psi is not a Dirac parity at every clock.
    - Reason for withdrawal: the admissible parities at each of four clocks are exactly span{n, T} (s4-8).

## Also verified, no finding

- Citations against paper_labels.txt / re_labels.txt and the sources:
  - ES17: Lemma 2.1, Remark 4.2, Propositions 4.5 and 4.6 (Omega is the real form of i gamma_5), Theorems 4.1, 5.1 and 5.3,
    Theorem 12.3 with eq. (14) inside it, Proposition 13.12, Section 13.6, eq. (1).
  - RE: Propositions 2.11, 3.6(iv), 3.7 and 3.8.
  - FMB: Proposition 5.1, Section 1 (permutations deferred), Section 5 (reversing n reverses J(n)), Section 8 items 5 and 6,
    and the Section 7 requirement on cross-sector couplings.
  - The monograph's Theorem 7.1 (two trivectors, V = e4e5e6 as covariant mass, IV = e1e2e3 as pseudoscalar) is reported
    correctly in Remark V. Its anticommutation statement is confirmed at the monograph's clock W, too (s4-20).
- Literature: Pauli, Rev. Mod. Phys. 13 (1941) 203 (web-verified); Hestenes, J. Math. Phys. 8 (1967) 798 (web-verified);
  Itzykson-Zuber (McGraw-Hill 1980); Peskin-Schroeder (Addison-Wesley 1995); Goodman-Wallach, Chevalley and Hestenes 1966
  verbatim from tex/refs.tex and the RE references. No key clashes.
- Units: the SI form of eq. (r4:eq:LP) is J m^-3 for [mu] = J/T and [d] = C m; mass dimension -1 (s4-22).
- Paper's B = -B_t: the axial transformation (sgn P B; P E), Phi_t = -(-1)^N Phi and the duality +T for the textbook
  orientation are all consistent (s4-5).
