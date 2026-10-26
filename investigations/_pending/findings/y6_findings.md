# y6 referee findings: investigations/emergence/sections/e6_bfield_triples.tex

Independent check script: investigations/emergence/verify/y6_check.py (33 PASS lines, exit 0, ~16 s).
Fresh build of investigations/two_readings (scratch copy with bib_r* merged once each) used for the TR numbers.

## Findings

1. **REFUTED** Remark e6:rem:transfer, "In the ungraded reading, ... the transfer is obstructed: no conjugation turns one clock into the
   other, and the additive spectrum fixes the pair of masses". The flip P(x(x)y) = y(x)x is an orthogonal matrix with P u_1 P^-1 = u_2
   (u_1 = e1(x)1, u_2 = 1(x)e1) and P H_u(m1,m2) P^-1 = H_u(m2,m1): the complete exchange (M,0) -> (0,M) is a conjugation at fixed spectrum
   {+-M}. The spectrum fixes only the unordered pair {|m1|,|m2|}. What is true (e6:prop:transfer(ii) itself is careful) is that no
   conjugation carries u_1 to cos t u_1 + sin t u_2 with sin 2t != 0, so no continuous transfer at fixed spectrum exists. Evidence: y6-6f
   (exact). Proposed wording: "the continuous transfer is obstructed: no conjugation turns the clock u_1 into a combination
   cos t u_1 + sin t u_2 with sin 2t != 0, and the additive spectrum fixes the unordered pair of masses; the only spectrum-preserving change
   is the exchange of the two masses, realised by the flip of the tensor factors, a relabelling of the constituents rather than a transfer."
   KEPT (second look: P is a genuine invertible, even orthogonal, matrix of M_64(R); the sentence has no qualifier). APPLIED.

2. **WRONG-DETAIL** e6:prop:transfer(ii), "Rotating (m_1,m_2) inside the ungraded family changes the spectrum, from {+-M} at (M,0) to
   {+-M(cos t + sin t), +-M(cos t - sin t)}". At t = pi/2 the displayed set is {+-M} again: the spectrum changes iff sin 2t != 0.
   Evidence: y6-6f (spectrum {+-1} at t = pi/2, 0, pi). Proposed: "changes the spectrum unless sin 2t = 0" and add the flip.
   KEPT. APPLIED.

3. **OVERCLAIM** Introduction item (6), "in the ungraded reading no transfer at fixed spectrum exists"; Remark e6:rem:compare(d),
   "Obstructed: ... mass transfer at fixed spectrum in the ungraded reading"; Remark e6:rem:compare(e), "In the reading that has two particles
   it is obstructed". By finding 1 only the continuous transfer is obstructed; the discrete exchange exists. Proposed: insert "continuous"
   and, in (e), "only the exchange of the two masses, a relabelling, keeps the spectrum". KEPT. APPLIED.

4. **WRONG-DETAIL** Discussion after e6:prop:clock, "Equivalently, the clock turns against a fixed momentum along a great circle of the clock
   sphere". The passive clock n(t) = R(-Omega t) n keeps n.e^ fixed: it runs on the circle of latitude about e^ of radius |n_perp|, a great
   circle only if n is perpendicular to e^ (for the example W^ about e3 the radius is sqrt(2/3)). Evidence: y6-5d. Proposed: "along the
   circle of the clock sphere about the axis e^ through n (a great circle only when n is perpendicular to e^)". KEPT. APPLIED.

5. **UNCLEAR** e6:prop:clock(iv), "read by $\widehat W$": W^ is never defined in the section. It is the symmetric clock
   (e1+e2+e3)/sqrt3 (Proposition 3.5 of ES17; Proposition 2.10 of RE; Proposition 3.13(iii) of TR). Evidence: grep; the numbers
   35/3 +- 4 sqrt26 and 94/3 are reproduced with that W^ (y6-5b). Proposed: define it at first use. KEPT. APPLIED.

6. **UNCLEAR** Notation clashes. (a) Theorem e6:thm:ladder uses S_n for the spatial volume and, in the same item, "the forms on S_n" and
   "Sym^2 S_n, Lambda^2 S_n" for the spinor module, which ES17 eq. (2) calls S_n = (R^8)^(x)n. (b) In Section e6:sec:clock the rate of the
   temporal b-field is called Omega, while Omega = e1e4e5e6 in the Notation paragraph and in Section e6:sec:dirac. Proposed: write the
   module as (R^8)^(x)n where it is meant, with one sentence warning that S_n is the spatial volume; rename the rate to nu. (Other reuses,
   S for spinors and for the shift, M for the phase-space generator and for (m1^2+m2^2)^{1/2}, are local and left.) KEPT. APPLIED.

7. **CITATION** Notation paragraph, "The temporal generators are symmetric and the spatial ones antisymmetric (Theorem~2.3 of ES17)":
   Theorem 2.3 of ES17 is about bivectors (squares, compact versus boost); the generator transposes e_a^T = eta_aa e_a are Lemma 2.1 of ES17,
   equation (4) with r = 1. Also "$J=e_2e_3$, $\Omega=e_1e_4e_5e_6$ (Theorem~18.1 of ES17)": Omega is not in Theorem 18.1; it is defined in
   Section 4 of ES17 (before Theorem 4.1). Proposed: "(Lemma~2.1 of ES17, equation~(4))", Theorem 2.3 moved to the compact/boost sentence;
   "(Section~4 and Theorem~18.1 of ES17)". KEPT. APPLIED.

8. **CITATION** "Part (iv) restates Proposition~2.11(i) of RE" and "Part (v) is the passive form of Proposition~2.14(ii) of RE": the parts of
   RE Propositions 2.11 and 2.14 are numbered (1), (2), (3) in RE (d1_unitary_energy.tex), so these are 2.11(1) and 2.14(2). Content
   checked: 2.11(1) is "P = Ad_T|_V = eta ... parity on the frame"; 2.14(2) is the anti-diagonal transport of a null field to one at rest.
   KEPT. APPLIED.

9. **CLASS** Remark e6:rem:csym(a), labelled "Fixed", contains "In this precise sense c^2 sits on the spatial triple and the temporal
   b-field cannot reach it", which is the reading declared interpretive at the end of (c). The item omits the counterpoint that the energy
   E_C = (1/2)X^T N_C X sees the other symmetric factor C^T C, a form on the temporal triple (it weighs |B|^2), which the temporal rotations
   conjugate; the two factors coincide exactly in the isotropic case. Evidence: y6-4d (C^T C conjugated, changes), y6-4f. Proposed:
   "In this sense (a reading, (c)) ... The energy E_C sees the other factor C^T C, a form on the temporal triple that the temporal rotations
   conjugate; it equals C C^T exactly in the isotropic case." KEPT. APPLIED.

10. **OVERCLAIM** Remark e6:rem:csym(a), "A $b$-field on the temporal triple can change anything about the conversion except this": a
    temporal b-field generates SO(3)_+, so it also keeps sgn det C; C and C diag(1,1,-1) have the same C C^T and lie in one O(3)_+ orbit,
    but in different SO(3)_+ orbits. Evidence: y6-4f. Proposed: "except this and the sign of det C". KEPT. APPLIED.

11. **CLASS** Table e6:tab:claims, row e6:rem:csym, "(a), (b) fixed ... checks ---": the finite-dimensional claims of (a) and (b), in particular
    the converse "C and C' lie in one orbit iff C'C'^T = CC^T", have no PASS line (e6-4d checks only the forward direction). Proposed: extend
    e6-4d with the converse (and the determinant sign) and cite it. KEPT. APPLIED (new PASS line e6-4f in the derivation script).

12. **CLASS** Theorem e6:thm:ladder(iv) at n = 1, classed "fixed": the antisymmetric count 10 is dim K_W = dim g_W = 10 for W = T in
    Theorem 18.3 of ES17 (the antisymmetric operators commuting with so T). Proposed: "at n = 1 the antisymmetric count restates
    Theorem~18.3 of ES17". KEPT. APPLIED.

13. **UNCLEAR** Answer, "The mass is unique in $T$ on the first rung only": literally true for T = e1e2e3, but next to item (7) ("the
    temporal volume only for n = 1 mod 4") it reads as a contradiction with the theorem (T_5 is the unique kernel at n = 5). Proposed:
    "The mass is unique in T = e1e2e3 on the first rung only; up the ladder the unique invariant Grassmann kernel is the temporal volume
    T_n for n = 1 and the spatial volume S_n for n = 3 (mod 4), and at even rungs there is a plane of them or none." Evidence: y6-7a
    (n = 1..6). KEPT. APPLIED.

14. **WRONG-DETAIL** Discussion after e6:prop:binding, "Its counter half lies in $\mathfrak p_G$. The latter is the gauge-covariant remainder
    $D_\tau G=[b,G]$": the counter half b_p is not D_tau G; it determines it, D_tau G = [b,G] = 2 b_p G (e6:prop:binding(ii); RE Prop. 5.9).
    Proposed: "The latter gives the gauge-covariant remainder D_tau G = [b,G] = 2 b_p G". KEPT. APPLIED.

15. **CITATION** (withdrawn) Chevalley--Lawson--Michelsohn attribution of the symmetries (-1)^{m(m-1)/2}, (-1)^{m(m+1)/2}. The pattern is
    Chevalley's and correct (y6-7a, n = 1..6). Whether Lawson--Michelsohn tabulates it could not be verified (no access to the book; a web
    search found Chevalley, Alekseevsky--Cortes and Gualtieri for it). WITHDRAWN: no evidence that the citation is wrong. NOT APPLIED.

16. **UNCLEAR** (withdrawn) e6:prop:binding(iv) states the Jordan structure only "at (kappa,omega) = (2,3) and (6,5)". The second look shows it
    holds for all kappa, omega > 0 (y6-3d: exact at three rational points with generic directions; the RE Thm 5.5 pairing criterion at 30
    random real points); the proof sketch already gives the general argument (b_l commutes with M_p, Jordan--Chevalley). An understatement,
    not an error. WITHDRAWN. NOT APPLIED.

## Verified as correct (independent method in brackets)

- The flagged period-four pattern (e6:thm:ladder(ii)): T_n^T = (-1)^{3n(3n-1)/2} T_n, S_n^T = (-1)^{3n(3n+1)/2} S_n, (anti,sym), (anti,anti),
  (sym,anti), (sym,sym) for n = 1,2,3,4 mod 4, confirmed for n = 1..6 (signed permutations up to 8^6); the invariant forms are span{T_n,S_n}
  at n = 1..4 (Cartan reduction + permutation-graph components = 2); hence the Grassmann kernel is unique exactly at odd n, T_n at n = 1 mod 4,
  S_n at n = 3 mod 4. "The mass is unique in T only at n = 1 mod 4" is confirmed in the sense of the temporal volume T_n. (y6-7a, y6-7b)
- The flagged CC^T statement (e6:prop:polar(iv)): a temporal rotation (Q,1) sends C to CQ^-1, fixes CC^T (a form on V^-) and conjugates
  C^TC; CC^T is a complete invariant of the O(3)_+ action (polar decomposition, random test); a spatial rotation fixes C^TC. (y6-4d)
- The flagged Jordan blocks (e6:prop:binding(iii),(iv)): det(lambda - S) symbolic in r; lines kappa, s +- omega/2; exact Jordan profiles
  (geo 1, alg 2 at the split lines; (2,2) at kappa) at (2,3), (6,5), (4,6) in generic directions and two masses; the RE Thm 5.5 eta-pairing
  criterion at 30 random points; linear growth of ||e^{tau M}||. (y6-3c, y6-3d) Same for the copy field (y6-6e, also at (4,6)).
- The flagged Georgi--Glashow form: sum_a ||Y^T K^a + K^a Y||^2 = 96(beta^2+gamma^2) symbolically, and = 96|y x n|^2 for a general clock,
  kernel R J(n) with J(n) = Tn; the identification with g^2|[b,<Phi>]|^2 of the SO(3) model is correctly classed as a reading with the
  mechanism open. Reference checked (Phys. Rev. Lett. 28 (1972) 1494--1497). (y6-2a)
- e6:prop:Tform (i)-(iv) [expm of random so(3,3) elements; exact solve of the invariant forms; commutant dimension 32 = dim gl(4,C), centre
  span{1,T}, quaternion relations] (y6-1a..c).
- e6:prop:dirac (i)-(iv) [symbolic; covariance system dim 7 meeting so(3)_+ in RJ; rotor (1 + n e1); Fock-built field-octant map]
  (y6-2a..c); e6:prop:twosu2 (i)-(v) [bracket closure, expm, blade expansion] (y6-2d).
- e6:prop:binding (i),(ii),(v) (y6-3a, y6-3b, y6-3e); e6:prop:polar (i)-(v) (y6-4a..e); e6:prop:clock (i)-(iv) [300 random draws; exact
  example] (y6-5a, y6-5b); e6:prop:copyrot (i)-(iv) [numerical spectra and expm, spin action on the generators, Fock weights at two angles]
  (y6-6a..c, y6-6e); e6:prop:transfer (i),(iii) (y6-6d).
- e6:thm:ladder (iii) [Cartan-support exact solve, 4-dim], (iv) [character sums over the finite group of even temporal blades: 16, 128, 1024;
  (6,10), (64,64), (528,496)], (v) [SU(4) Weyl integration of Sym^2/Lambda^2 of (4+4*)^(x)n: (1,1), (10,4), (84,84); exact solve at n = 2]
  (y6-7c..e).
- Citations: every ES17 number (Lemma 2.1, Thm 2.3, Cor 2.4, Thm 3.2, Sec 4.4, Prop 4.5, Thm 5.1, Prop 5.5, Thm 12.3, Thm 18.1, Cor 18.4,
  Sec 28, Thm 30.1) against paper_labels.txt and tex/; every RE number (2.7, 2.8, 2.10, 2.11, 2.14, Sec 2.5, 3.3, 3.5, 3.6, 3.7, 3.8, 4.9,
  eq. (9), (11), 5.3, 5.4, 5.7, 5.8, 5.9, 5.10, 8.11(iv), 9.3, 9.6(i), 9.8(e)) against re_labels.txt/main.aux and the text; every TR number
  (Prop 3.4, Thm 3.5, Prop 3.10, Prop 3.13, Rem 3.14, Sec 3, Thm 4.7, Prop 4.12) against a fresh build of investigations/two_readings;
  e1/e2 cross-references (e1:prop:copies, e1:prop:rest, e1:prop:Nbiv, e1:prop:Nlorentz, e2:lem:N, e2:sec:quanta, e2:prop:copyboost).
- Literature: Pauli 1957 (Nuovo Cim. 6, 204--215), Guersey 1958 (Nuovo Cim. 7, 411--415), Georgi--Glashow 1972 (PRL 28, 1494--1497),
  Englert--Brout, Higgs, Kibble, Larmor 1897, Humphreys GTM 9: entries correct; GeorgiGlashowPRL72 correctly renamed (ES17 uses
  GeorgiGlashow72 for the anomaly paper).
