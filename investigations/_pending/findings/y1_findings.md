# Referee report y1 on sections/e1_two_from_one.tex

Independent script: investigations/emergence/verify/y1_check.py (45 PASS, 0 FAIL, exit 0, about 30 s). It does not import or copy
verify/e1_two_from_one.py. Its methods differ from the derivation script: operators are encoded as signed real Pauli words with exact
GF(2) sign arithmetic, commutants are counted with the character formula (dim = 2^(2m - r), where r is the GF(2) rank of the
generators), algebra types are fixed by the signature of the trace form, the Fock content of bivectors is read entry by entry in the
occupation basis, and dense integer matrices are used for the n=2 cross-checks and the statistics table. A statement of the section
that I found false appears in the script as a check of the corrected statement ("corrected: ...").

## Findings

1. **REFUTED** - Section 4 (e1:sec:fock), paragraph after Proposition e1:prop:Nlorentz: "They are invariant under the compact part of
   $\lG$, the rotations of the field-strength reading, and under nothing that contains a boost or the electric charge."
   *What is wrong.* The section's own Proposition e1:prop:Nlorentz(ii) says the boost $e_1e_4\in\lint$ commutes with $N$. More
   generally, the bivectors that commute with $N$ form $\mathfrak{gl}(3n,\R)$ (Proposition e1:prop:Nbiv(ii)). Its non-compact
   (symmetric) part has dimension $3n(3n+1)/2$ and consists of boosts, for example the pole operators $e_ie_{i+3}$ and
   $e_ie_{j+3}+e_je_{i+3}$.
   *Evidence.* y1-6e: at $n=1$ the commutant of $N$ in $\so{3,3}$ has dimension 9, with 6 symmetric (boost) and 3 antisymmetric
   dimensions. y1-6c: $e_1e_4=\id-2N_1$ is symmetric.
   *Corrected wording.* "Within $\lint$ they are invariant only under the one boost $e_1e_4$, the pole operator of the first mode.
   Within $\lG$ they are invariant only under its compact $\so3$. They are not invariant under the charge $J$ or under any boost of
   $\lG$. The bivectors that preserve $N$ form $\mathfrak{gl}(3n,\R)$, which contains boosts, so $N$ is tied to the split structure
   $K$ and not to a Lorentz algebra."

2. **OVERCLAIM** - Same paragraph: "These are excluded as gauge fields, so pair creation of quanta would need a non-compact, non-gauge
   interaction (fixed, with the identification of quanta interpretive)."
   *What is wrong.* The statement holds only inside the binding-compatible algebra $\Cent(G_n)$, the hypothesis of Theorem
   e1:thm:Nrule. Compact, gauge-admissible bivectors that do not commute with $G_n$ do create and annihilate pairs of quanta. Examples
   are the electric charge $J=e_2e_3$, which Proposition e1:prop:Nlorentz(ii) itself notes involves $a^+_2a^+_3$, and the time-time
   rotation $e_1e_2$. Both have $\Delta N=\pm2$ components.
   *Evidence.* y1-6f: $J$ and $e_1e_2$ are antisymmetric (compact), do not commute with $G$ on vectors, and have entries with
   $\Delta N=+2$, $0$ and $-2$.
   *Corrected wording.* "...so within the binding-compatible algebra, pair creation of quanta would need a non-compact, non-gauge
   interaction. Outside it, compact gauge-admissible fields such as the charge $J$ create and annihilate quanta in pairs." The
   intro, Theorem e1:thm:Nrule(iii), answer (4) and the table are correctly qualified with "binding-compatible" and need no change.

3. **OVERCLAIM** - Section 5 (e1:sec:exchange), after Table e1:tab:stat: "The Clifford exchange $\tau_K$ does not sort by Lorentz
   content at all. Its eigenvalue is a chirality-dependent sign, not a statistics label."
   *What is wrong.* The same paragraph and Proposition e1:prop:stat(ii) say the two $\tau_K$ sectors on $W$ are mirror images once
   $J$ is the imaginary unit. The spin-one part of $\tau_K=+1$ is $(1,0)$ and that of $\tau_K=-1$ is $(0,1)$. So $\tau_K$ does sort
   by complex Lorentz content (self-dual against anti-self-dual). It does not sort by real Lorentz content or by symmetry type.
   *Evidence.* y1-7g: on the $C=4$ part of each $\tau_K$ sector of $W$, $\Delta_{14}=s\,J\,\Delta_{56}$, with opposite signs $s$ in
   the two sectors. y1-7d and y1-7f give equal real content and the chirality location of spin one.
   *Corrected wording.* "The Clifford exchange $\tau_K$ does not separate symmetric from antisymmetric content. Its two sectors have
   the same real Lorentz content and differ only in the chirality of their spin-one part, so its eigenvalue is a chirality label, not
   a statistics label."

4. **WRONG-DETAIL** - Proposition e1:prop:rest(iii): "$\spec H_u=\{\pm m_1\pm m_2\}$, each of multiplicity $16$", and (vi): "the
   ungraded sum has spectrum $\{\sum_k\pm m_k\}$, each value of multiplicity $4^n$".
   *What is wrong.* The multiplicity statements hold per sign pattern, that is for generic masses. At $m_1=m_2$ the value $0$ has
   multiplicity 32. The paragraph after the proof cites exactly this case as a restatement of RE Prop. 8.11(iv), which lists
   multiplicities 16, 32, 16. At $n=3$ with $m=(1,2,2)$ the value $\pm1$ has multiplicity 128.
   Separately, the restriction "For $m_1\neq m_2$, both nonzero, ... not conjugate to any vector" is narrower than necessary. At
   $m_1=m_2\neq0$, $H_u$ is diagonalisable with three distinct eigenvalues, while a vector $v$ has $v^2=\eta(v,v)$ and is either
   diagonalisable with at most two eigenvalues or nilpotent. So $H_u$ is not conjugate to a vector for any nonzero masses.
   *Evidence.* y1-2c (generic $(1,2)$: four values, 16 each), y1-2d ($m_1=m_2$: 16, 32, 16), y1-2g ($n=3$, $(1,2,4)$: 64 each;
   $(1,2,2)$: 128 at $\pm1$).
   *Corrected wording.* "$H_u$ has eigenvalue $s_1m_1+s_2m_2$ on the $16$-dimensional joint eigenspace of each sign pair
   $(s_1,s_2)$." Then "For $m_1,m_2\neq0$ it has at least three distinct eigenvalues and is diagonalisable, so it is not conjugate to
   any vector." Write (vi) as "...each sign pattern on $4^n$ dimensions".

5. **WRONG-DETAIL** - Proof of Proposition e1:prop:W(iii): "The blade rule again: none of these contains a surplus index, except
   $J^{(k)}$ itself and $M_{12}$, whose terms share no index with $J^{(1)}$ or $J^{(2)}$."
   *What is wrong.* The ungraded gammas $u_\mu=\Gamma^{(1)}E_{(2,\mu)}=E_{(1,1)}\cdots E_{(1,6)}E_{(2,\mu)}$, and therefore $H_u$,
   contain both surplus indices $E_{(1,2)},E_{(1,3)}$ of copy 1. The claim of (iii) is still true, because they meet $J^{(1)}$ in two
   indices, an even number, and meet $J^{(2)}$ in none. The terms of $M_{12}$ contain no surplus index, so the "except ... $M_{12}$"
   clause is also misplaced.
   *Evidence.* y1-3c2: the unique $E$-blade equal to $u_\mu$ has index set $\{(1,1),\dots,(1,6),(2,\mu)\}$. y1-3c: the
   commutations themselves hold.
   *Corrected wording.* "By the blade rule: the graded gammas, the bivectors of $\lint^{(k)}$ and the terms of $M_{12}$ contain no
   surplus index; the ungraded gammas $u_\mu=\Gamma^{(1)}E_{(2,\mu)}$ contain both surplus indices of copy 1, hence meet $J^{(1)}$
   evenly and $J^{(2)}$ not at all; or directly, $\id\otimes\gamma$ commutes with $J\otimes\id$ and $\id\otimes J$."

6. **CLASS** - Intro: "The graded tower describes one particle: its two mass terms anticommute, they add in quadrature, and no state has
   a sharp rest energy in each copy (fixed)." Answer (1): "...so the rung is one particle with a larger internal space (fixed;
   ...)".
   *What is wrong.* The algebraic facts are fixed: anticommuting mass terms, quadrature, no joint mass shell, and copy 2 carrying no
   $J_{\rm sel}$-linear Dirac operator. The conclusion "describes one particle" is a reading of them. The paragraph after Proposition
   e1:prop:rest, which explains the "second kind", is itself a reading.
   *Corrected wording.* "...anticommute, add in quadrature, and no state has a sharp rest energy in each copy (fixed); read as
   particle content, the graded rung is one particle (interpretive)." The same applies in answer (1).

7. **CLASS / CITATION** - After the proof of e1:prop:copies: "Parts (i)--(iii) restate Proposition~7.1 of \cite{RE} and Appendix~A of
   \cite{ES17} (fixed there), with the ungraded copy named. ... The identification of the Klein factor with the chirality of the first
   copy, $u_a=\Gamma^{(1)}E_{(2,a)}$, is fixed (e1-1a to e1-1d)."
   *What is wrong.* (a) RE classes Proposition 7.1 as inherited (the exponential law), not "fixed there". (b) Part (iii),
   $\Gamma^{(1)}=-(-1)^{N^{(1)}}$, is RE Proposition 7.2, not 7.1. (c) The "identification" is already in the cited sources. RE Prop.
   7.1(ii)-(iii) gives $e^{(1)}_Ie^{(2)}_J=(e_I\omega^{|J|})\otimes e_J$, hence $\id\otimes e_a=(\omega\otimes\id)e^{(2)}_a$. RE Prop. 7.2
   calls the string $-(-1)^{N^{(1)}}$. ES17 Appendix A gives $\omega=\Gamma=J\Omega$. So it is restated, not new.
   *Corrected wording.* "Parts (i)--(iii) restate Propositions~7.1 and~7.2 of \cite{RE} (inherited there) and Appendix~A of
   \cite{ES17}; naming $u_a=\Gamma^{(1)}E_{(2,a)}$ as the commuting copy is a restatement in this notation (e1-1a to e1-1d)."

8. **CITATION** - Theorem e1:thm:Nrule: "the algebra $\so{3n,\C}$ of binding-compatible bivectors (Theorem~17.19 of \cite{ES17};
   Proposition~7.8 of \cite{RE})".
   *What is wrong.* ES17 Theorem 17.19 is the $n=3$ statement ($\so{9,\C}$). RE Proposition 7.8(i) covers $n=1,2,3$. The general-$n$
   statement, with $\dim=3n(3n-1)$, is proved by the block form in the section's own proof and was checked at $n=1..4$.
   *Corrected wording.* "(the case $n=3$ is Theorem~17.19 of \cite{ES17}, $n\le3$ Proposition~7.8(i) of \cite{RE}; the block form gives
   all $n$)". y1-5c re-verifies the dimensions 6, 30, 72, 132 and 9, 36, 81, 144 by exact rational ranks.

9. **CITATION** - After Table e1:tab:stat: "Identical fermions, antisymmetric under $P$, have only scalar and vector content at a single
   momentum mode (inherited \cite{Pauli40})."
   *What is wrong.* Pauli (1940) proves the spin-statistics connection. It does not contain the decomposition
   $\Lambda^2[(\tfrac12,0)\oplus(0,\tfrac12)]=2(0,0)\oplus(\tfrac12,\tfrac12)$, which is elementary representation theory. The same
   decomposition is already credited to \cite{WeinbergQFT} two sentences earlier.
   *Corrected wording.* Cite \cite{WeinbergQFT} (or \cite{PenroseRindler84}) for the decomposition, and \cite{Pauli40} only for the
   requirement that identical fermions be antisymmetric.

10. **UNCLEAR** - Notation and symbols used before they are defined. (a) "$k_I=1$", "$k_I=2$" in Proposition e1:prop:select
    (ii)-(iii) are not defined in the section. They are the numbers of constituents carrying $\lint$ (ES17 Thm 10.8; RE Table 9). (b)
    $\Delta_{\mu\nu}$ in Proposition e1:prop:swap is used before its normalisation, which appears only implicitly through the Casimir.
    The checked normalisation is $\Delta_{\mu\nu}=\tfrac12(E_{(1,\mu)}E_{(1,\nu)}+E_{(2,\mu)}E_{(2,\nu)})$, giving $C=\tfrac32$ on one
    constituent (y1-7c). (c) $M_{12}$ is defined only inside Proposition e1:prop:W(iii). (d) "two-excitation states of one Dirac mode"
    (after Corollary e1:cor:nocx) should read "of one Dirac momentum (four fermionic modes)", which is the wording of ES17 Thm 5.7 and
    RE Prop. 3.9. (e) Corollary e1:cor:nocx: "commutes with every gamma matrix" should say "every interior gamma matrix", because
    $J^{(k)}$ anticommutes with the surplus generators.
    *Corrected wording.* Define $k_I$, $\Delta_{\mu\nu}$ and $M_{12}$ in the Notation paragraph, and make the two wording changes.

## Verified as correct (independent method in brackets)

- e1:prop:copies (i)-(v). Clifford relations of the 12 $E$'s and the 6 $u_a$. $u_a=\Gamma^{(1)}E_{(2,a)}=E_{(1,1)}\cdots E_{(1,6)}E_{(2,a)}$.
  The commutant of $\Cl(3,3)\otimes\id$ has dimension 64 [character count] and is spanned by $\id\otimes e_J$.
  $\Gamma^{(1)}=(J\Omega)\otimes\id=-(-1)^{N^{(1)}}$ [bit count]. The graded and ungraded blade sets coincide up to sign.
  $(e_1\otimes\id+u_1)^2=2+2e_1\otimes e_1$. Rung 3 and 4 dressing, grade $6(k-1)+1$, and commute/anticommute [Pauli words]. (y1-1a..1f)
- e1:prop:rest (i), (ii), (iv), (v), and (vi) for generic masses. $H_g^2=M^2$. Exact rotation $U_\phi$ at $(3,4)$ with $B$
  antisymmetric. No joint mass shell [rank]. Joint signs 16 each. $(D_1+D_2)^2$ [random momenta]. Rest flows, including $\exp$
  [tol 1e-10]. Rung 3 spectra and rung 4 subset traces. (y1-2a..2g)
- e1:prop:W (i)-(iv). $\omega_S=J^{(1)}J^{(2)}$ is a symmetric traceless involution. $\dim W=32$.
  $\overline W=\ker(J^{(1)}+J^{(2)})=\operatorname{im}(J^{(1)}-J^{(2)})$. Surplus-parity rule on all 4096 blades. Commutations of
  (iii). $C_b|_W=M_{12}|_W$. Cross surplus bivectors antilinear. (y1-3a..3d)
- e1:prop:select (i)-(iv). Commutant of the ungraded gammas: dim 16, $M_4(\R)$ [trace-form signature (10,6)]. With $T^{(1)},T^{(2)}$:
  $\operatorname{span}\{\id,J^{(1)},J^{(2)},\omega_S\}$, with exactly four roots of $-\id$ [sympy, structure constants from matrices].
  $J_{\rm sel}=J\otimes\omega=J^{(1)}\Gamma^{(2)}$ with all stated eigen-splittings. Commutant of the graded interior gammas: dim 16,
  $M_2(\HH)$ [signature (6,10)]. With the surplus bivectors: $\operatorname{span}\{\id,\omega_S\}$. On $W$ and on $\overline W$:
  $\R$ [character average with $\Tr(gP_W)$]. Rungs 2, 3, 4: dim 2 and $\omega_S^2=(-1)^n$. Common subspace of dimension $2\cdot4^n$
  ($n=2,3$). $\omega_S=-J$ there at $n=3$. (y1-4a..4f)
- e1:cor:nocx as a statement about the stated selection principles.
- e1:prop:Nbiv (i)-(iii). $N$ equals the bit count with multiplicities 1, 6, 15, 20, 15, 6, 1. $(-1)^N$ is the volume element.
  $[N,v]=Kv$. $\ad N$ on bivectors: 0, +2, -2 on 36, 15, 15. (y1-5a, 5b)
- e1:thm:Nrule. Dimensions 9, 36, 81, 144 and 6, 30, 72, 132 [exact rank over $\Q$]. $KXK=-X^{\!\top}$. Every paired rotation
  conserves $N$ and every paired boost has only $\Delta N=\pm2$ entries, both signs, at $n=1..4$ [occupation-basis entries]. The
  vector actions are of the stated type. (y1-5c..5e)
- e1:prop:Nlorentz (i)-(iii). $U=\tfrac18\prod(\id+\eta_{aa}E_{(1,a)}E_{(2,a)})=e^{\pi C_b/2}$, with $UE_{(1,a)}U^{-1}=-E_{(2,a)}$ and
  $UN^{(1)}U^{-1}=N^{(2)}$. The lifted copy boost changes $N$ by $\pm2$ only [least-squares lift, then exact check]. $\lint$ and $\lG$
  statements. (y1-6a..6d)
- e1:prop:swap (i)-(iii), the Casimir normalisation and $C(C-3)(C-4)=0$, $e_1e_4=J\Gamma e_5e_6$ cyclically, the entire Table
  e1:tab:stat [exact traces of scaled projectors], and e1:prop:stat (i)-(iii). (y1-7a..7f)
- Citations checked against paper_labels.txt, re_labels.txt and the cited text, all correct in number and content. ES17: Sec. 1.4;
  eqs. (1)-(4); Cor. 2.4; Sec. 2.2; Thm 4.1; Rem. 4.2; Sec. 4.4; Thm 5.1; Thm 5.7 (level of dimension 6, $q=\pm2$ once, $q=0$ four
  times); Sec. 10.2; Prop. 10.10(ii); Thm 12.1; App. A ($\cK=-K$). RE: Props. 3.9, 4.9, 7.1(iii), 7.2, 7.3(ii)-(iii) (sign of $C_b$
  as in RE), 7.8(i)/(iii) (the unlabeled Prop. 7.8 "$\Cent(G_2)=\so{6,\C}$ and the copy algebra"), 7.10(iii), 7.11(iv), 8.11(i)/(iv),
  Sec. 8.5. Exceptions are findings 7-9.
