# Referee findings: investigations/two_readings/sections/r3_octants.tex

Independent check script: `investigations/two_readings/verify/s3_check.py` (31 PASS lines, exit 0). It builds the generators from the
bit-flip rules of ES17 Appendix A, does every dimension count as an exact rank of an integer system, and checks the field
equations as exact residual identities on general symbolic fields. Where the section was wrong, the script checks the corrected
statement (s3-7, s3-9, s3-19).

Each finding was given a second look in Step 2 (a second computation or a re-reading of the source). The result is under
"Second look". "Kept" means the finding survived and was applied in Step 3.

1. **REFUTED** Prop. `r3:prop:octgroup`, statement: "generate a group of order $16$ with centre $\{\pm\id\}$".
   - **What is wrong.** The centre has order 4: it is {±1, ±g_1g_2g_3}. Moving g_i past the other two lifts costs two signs, so
     g_1g_2g_3 commutes with every g_i. This element is g_1g_2g_3 = -Tω = e_4e_5e_6. It squares to +1 and maps each octant to
     its antipode. The group is Q_8 × Z_2. What equals {±1} is the kernel of the action on the octant lines.
   - **Evidence.** s3-7: a brute-force closure gives 16 elements and a centre of 4 elements. The derivation script printed
     "centre +-1" (r3-5) but never computed the centre.
   - **Corrected wording.** "...generate a group of order 16 whose centre {±1, ±g_1g_2g_3} contains g_1g_2g_3 = e_4e_5e_6, the
     element mapping each octant to its antipode; the kernel of the action on the octants is {±1}, and the group acts through
     Z_2^3, simply transitively." The proof gains one sentence.
   - **Second look.** Kept. A separate numpy integer computation (scratch explore1.py) also found a centre of size 4 and
     g_1g_2g_3 = e_4e_5e_6, and the hand argument confirms it.

2. **REFUTED** Prop. `r3:prop:energies`(iv): "$(-1)^N$ and the Euclidean norm are invariant under the six compact bivectors and
   under $\rho_{\rm oct}$".
   - **What is wrong.** The Euclidean norm is not invariant under ρ_oct(K_k) = Q_k. The Q_k are symmetric, because e_k and
     e_{k+3}ω are both symmetric, so Q_k^T + Q_k = 2Q_k ≠ 0. Physically, |ψ|^2 = 2E_6 is the energy density, and boosts change
     it: |Φ(e^{sK_3}e_1)|^2 = cosh 2s.
   - **Evidence.** s3-19 (corrected statement). The derivation script's r3-18 checked only (-1)^N.
   - **Corrected wording.** "(-1)^N is invariant under the six compact bivectors and under ρ_oct, and under none of the nine
     boosts; the Euclidean norm is invariant under the compact bivectors and the ρ_oct(L_k), not under the symmetric ρ_oct(K_k) = Q_k."
   - **Second look.** Kept. numpy (explore1.py) gives Q_k symmetric and Q^T + Q ≠ 0. The energy-density argument is a second,
     realisation-free reason.

3. **UNCLEAR** (a reader is misled) Opening paragraph. The phrase "those of weight one carrying the electric components (as
   boosts) and those of weight two the magnetic components" is followed two lines later by "a field-octant map $\Phi$ placing
   $\bE$ on weight one and $\bB$ on weight two".
   - **What is wrong.** "Weight" means two different things. In the first sentence it is the earlier version's temporal weight,
     which counts plus signs (w = 3 - N, monograph Prop. 2.9 and Def. 5.5). In the second it is the Fock weight N. The paragraph
     therefore reads as if Φ confirms the earlier labelling. Remark `r3:rem:bijection` shows the opposite: the earlier labelling
     is Φ∘G, with E on N = 2.
   - **Evidence.** Monograph lines 726-731 and 1117-1121; s3-28.
   - **Corrected wording.** "...those with one plus sign (temporal weight one there, Fock weight N=2 here) carrying the electric
     components..." and "...placing E on the Fock weight N=1 and B on N=2, the reverse of the earlier labelling, which is its
     quarter turn of duality...".
   - **Second look.** Kept. Re-reading the monograph confirms that "temporal weight w is the number of +1 eigenvalues".

4. **UNCLEAR** (house rule: self-contained note) The section refers to expectations and proposals the note never states:
   - "It corrects the expectation that $\mathfrak{gl}(3)$ acts on $\Lambda^w$ as the bare exterior power"
   - "The paired combination proposed as a candidate is the wrong one"
   - "the precise form of the expectation that ``under $\mathfrak{gl}(3,\R)$ ...''"
   - "it contradicts the expectation that ``parity acts as $-1$ on $\bE$ and $+1$ on $\bB$''"
   - "This confirms the expected identification and corrects its sign statement: ... carry the same sign". Here the
     expectation had left the sign open (±), so nothing was "corrected".
   - "The signs proposed in the question, $\partial_t\bE=\curl\bB-\bj$ ...". The question asked in the introduction (the four
     requests) proposes no signs.
   - **What is wrong.** None of these expectations appears in the note, its introduction, ES17, RE, FMB or the monograph. They
     came from outside the note.
   - **Corrected wording.** Each sentence is restated as a self-contained claim, for example "It shows that gl(3) does not act
     on Λ^w as the bare exterior power" and "The textbook signs ∂_tE = curl B_t - j, ∂_tB_t = -curl E become those of (i) in the
     frame variable B = -B_t". The mass-form sentence becomes "...and fixes its sign: ψ_E^T T ψ_B = -E·B = -½ Im q".
   - **Second look.** Kept. grep finds no such statements in the monograph extract or FMB, and sections/r0_intro.tex does not
     exist.

5. **CITATION** "the note's statement that $G$ can represent the Hodge star on the field (its Definition~2.2 and its verdict table)".
   - **What is wrong.** FMB Definition 2.2 ("Electromagnetic field-strength reading") introduces Φ_n and X_EM ≅ Λ^2E_n^*. It
     does not mention G. The statement Φ_n(G_EM X_EM) = ⋆Φ_n(X_EM) is FMB eq. (3), in the subsection "Which operator acts on
     which reading?" of Section 2. The FMB aux file gives eq:Gtypes = (3).
   - **Corrected wording.** "(its eq. (3) in Section 2 and its verdict table)".
   - **Second look.** Kept. Re-reading lines 72-97 of FMB confirms it.

6. **CLASS** Claims table, row r3-8: "fixed; standard".
   - **What is wrong.** "Standard" is not one of the note's classes. The text calls the statement standard (Gualtieri Ex. 2.12)
     and fixed in the realisation.
   - **Corrected wording.** "inherited (Gualtieri), fixed here".
   - **Second look.** Kept.

7. **CLASS** "Constructed (i)--(iii), restated and fixed (iv)." (after Prop. `r3:prop:odd`), and table row r3-15,16.
   - **What is wrong.** (i) and (iii) also contain statements that could have failed and are therefore fixed: the affine
     dimensions 16 and 8, "no Clifford vector", the Casimir values 4 and 0, and the intertwiner space span{Φ, ΦG}. Only the
     operators Q_k and ρ_oct are constructed.
   - **Corrected wording.** "Constructed: Q_k and ρ_oct; fixed: the solution counts of (i) and the Casimir values and
     intertwiners of (iii); restated and fixed: (iv)."
   - **Second look.** Kept. s3-17 and s3-18 show these are nontrivial counts.

8. **WRONG-DETAIL** (B sign convention) Answer subsection: "$\Phi(X)=\sum_k(E_k+B_kT)e_k|\mathrm{vac}\rangle$ ... in textbook
   variables it is Hestenes' $\bE+I\Btext$".
   - **What is wrong.** In textbook variables Φ(X) = (E - T B_t)|vac>. Hestenes' E + I B_t is Φ_t = -(-1)^N Φ (Thm.
     `r3:thm:maxwell`(iv)), which differs from Φ by the sign of the magnetic octants.
   - **Corrected wording.** "...in textbook variables the same octants carry, through Φ_t = -(-1)^N Φ, Hestenes' E + I B_t...".
   - **Second look.** Kept. s3-24 confirms Φ_t = (Σ E_k e_k + B_tk T e_k)|vac> = -(-1)^N Φ ≠ Φ.

9. **WRONG-DETAIL** "$L_M=-\tfrac12\Re q$ ... is $-\tfrac12\psi^{\!\top}(-1)^N\psi$ on the octants, the difference of the
   magnetic and electric octant weights".
   - **What is wrong.** The value is ½(|ψ_E|^2 - |ψ_B|^2): half of electric minus magnetic. The text gives the opposite order
     and omits the ½. "Weights" also clashes with the Fock weight N.
   - **Corrected wording.** "...= ½(|ψ_E|^2 - |ψ_B|^2), half the squared electric minus the squared magnetic octant amplitudes."
   - **Second look.** Kept (s3-20).

10. **CITATION** Proof of Prop. `r3:prop:energies`: "...and $\Gamma$ commutes with all of them (Proposition~2.11 of \cite{RE})".
    - **What is wrong.** RE Prop. 2.11(3) states only the T part. That Γ commutes with bivectors is not in RE 2.11; it holds
      because Γ anticommutes with every vector.
    - **Corrected wording.** Move the citation to the T clause and give the one-line reason for Γ.
    - **Second look.** Kept. Re-reading RE d1:prop:Tparity confirms it.

11. **UNCLEAR** (grammar) Prop. `r3:prop:clock`: "and for the symmetric clock $B_{\widehat W}=\mathsf P/\sqrt3$ with top
    eigenvalue $\sqrt3$ on $|\mathrm{vac}\rangle$". The clause has no verb.
    - **Corrected wording.** "...and for the symmetric clock $B_{\widehat W}=\mathsf P/\sqrt3$ has top eigenvalue $\sqrt3$ on
      $|\mathrm{vac}\rangle$".
    - **Second look.** Kept (s3-8).

12. **CITATION** "Gualtieri identifies the form with the Mukai pairing (Section~2.4 and Proposition~2.21 of \cite{Gualtieri03})".
    - **What is wrong.** The identification is made in Section 2.4 ("For V⊕V* this bilinear form coincides with the Mukai
      pairing"). Proposition 2.21 is Chevalley's transversality criterion (Chevalley III.2.4), which Gualtieri restates.
    - **Evidence.** Read from the thesis via alphaXiv (math/0401221v1, pp. 11-13). Example 2.12 ("S = ∧•V* ⊗ (det V)^{1/2}")
      and Proposition 2.15 (symmetric iff m ≡ 0, 1 mod 4) were also confirmed.
    - **Corrected wording.** "...with the Mukai pairing (Section 2.4 of \cite{Gualtieri03}), where Proposition 2.21 restates the
      transversality criterion".
    - **Second look.** Kept as minor. The numbers exist and are relevant; only the attribution is imprecise.

13. **WRONG-DETAIL** (factor 2) Prop. `r3:prop:fock`: "A vector $X\in V$ acts as exterior multiplication by its $L$-part plus
    contraction with its $L'$-part".
    - **What is wrong.** The normalisation in the section is ℓ_1∧…∧ℓ_r ↦ 2^{-r}ℓ_1⋯ℓ_r|vac>, with ℓ_i = e_i + e_{i+3} = 2a^+_i.
      Under it, Clifford multiplication by ℓ ∈ L is exterior multiplication by 2ℓ. A vector ℓ' ∈ L' acts as contraction with
      η(ℓ',·). Equation (r3:eq:Xfock) itself is right.
    - **Evidence.** s3-9 implements the wedge and the contraction on Λ^*(L) explicitly.
    - **Corrected wording.** "Writing X = X_L + X_{L'}, a vector acts as exterior multiplication by 2X_L plus contraction with
      η(X_{L'},·)".
    - **Second look.** Kept. By hand, a^+_i ↔ ℓ_i∧, so ℓ_i = 2a^+_i ↔ 2ℓ_i∧.

14. **UNCLEAR** (notation) "$\psi=\Phi(\bE,c\bB)$" (Prop. `r3:prop:energies`(v)) and "$\Phi(\bE,c\bB)$" (Thm.
    `r3:thm:maxwell`(iii)).
    - **What is wrong.** The Conventions fix the coordinates as (B;E) on V, so the argument order is reversed.
    - **Corrected wording.** "Φ(cB;E)".
    - **Second look.** Kept as minor (s3-20, s3-24 use Φ(cB;E)).

## Withdrawn after the second look

W1. CITATION? "the chirality $\gamma_5$ of Section~5 of \cite{ES17}". Withdrawn: γ5 = -Γ is introduced in Thm 4.1 (with a sign
choice) and restated with that sign in Section 5 (s05_discrete.tex line 102).

W2. OVERCLAIM? "fixed by Faraday's law" for the ratio c of the two scales. Withdrawn: with the operator (1/c)∂_t - D either
evolution law fixes β = cα, and the paragraph after Prop. `r3:prop:maxunique` names both laws.

W3. OVERCLAIM? Corollary `r3:cor:carries`: boosts and duality "act only through the odd operators $Q_k$ and $-T$". Withdrawn: the
finite group elements e^{sQ_k}, e^{-θT} have mixed parity, but the sentence concerns generators, and the group level is stated
correctly in Thm. `r3:thm:noeven`. Every solution of QΦ = ΦK_k agrees with Q_k on the field octants (s3-17).

W4. CITATION? "(i) is in Proposition~2.5 and Appendix~A". Withdrawn: Prop. 2.5(i) gives the joint spectrum and Appendix A the bit
formulas and the diagonal ω_i.

W5. CITATION? Hehl and Obukhov "$F=\bE\wedge d\sigma+\bB$". Withdrawn. The bibliographic data were verified (Progress in
Mathematical Physics 33, Birkhäuser, Boston, 2003), and the decomposition F = E∧dσ + B, with E a 1-form and B a 2-form, matches
the book's premetric split as far as the referee knows it. The page could not be fetched.

W6. Literature data. Checked and correct:
- BecherJoos82: Z. Phys. C 15 (1982) 343-365.
- Kahler62: Rend. Mat. Appl. (5) 21 (1962) 425-523.
- Graf78: Ann. Inst. H. Poincaré A 29 (1978) 85-109.
- Gualtieri03: Ex. 2.12, Sec. 2.4, Prop. 2.21.
- Chevalley54, GoodmanWallach, Hestenes66, Jackson and LawsonMichelsohn: verbatim from tex/refs.tex or RE.

Checked and confirmed, no finding:
- Every ES17 number: eq. (1), Cor. 18.4, Thm 12.3, Prop. 3.3, Lem. 3.4, Prop. 2.5, App. A, Ex. 2.6, Thm 3.6, Sec. 3.3, Prop. 13.6,
  Thm 12.1, Prop. 13.12, Thm 5.3, Thm 13.13, Ex. 12.4, eq. (16), Thm 13.3, Ex. 12.11, Sec. 13 (all charges electric,
  s13_spin.tex line 454).
- Every RE number: eq. (1), Prop. 2.7, 2.8, 2.11.
- FMB Sec. 5 (I_T) and Sec. 7 (no linear equivariant map).
- Cross-references r1:prop:maxwell and r2:prop:nolinear (both exist with the cited content).
- The three corrected conventions named by the caller, verified independently:
  - Maxwell's equations in paper variables are ∂_tE = -curl B - j, ∂_tB = curl E (s3-23, from ES17 eq. (16)).
  - T(Φ(E),Φ(B)) = -E·B (s3-20).
  - The l_int stabiliser of an octant line is 3-dimensional (s3-29).
