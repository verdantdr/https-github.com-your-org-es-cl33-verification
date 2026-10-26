# y5 findings: adversarial review of `investigations/emergence/sections/e5_closure_two.tex`

Independent check script: `investigations/emergence/verify/y5_check.py`. It runs from any directory, prints 31 PASS lines, exits 0 and takes about 25 s. Nothing is imported or copied from `e5_closure_two.py`.

Methods used:
- **Exact linear algebra.** Elimination mod p = 2^31 - 1, rational reconstruction, and exact integer verification of every null vector. The dimensions are therefore exact over Q.
- **Lie algebras** are built from Clifford bivectors and their adjoint action. The spin-lift formula S(M) is checked against them.
- **Stabiliser of M_pair.** Exact rational rank at Gaussian-integer points, not a numerical rank.
- **M_pair** is also checked against its definition as an infimum over observers.
- **Drift law.** Exact sympy series and mpmath quadrature. The rotor law is checked along the leaf with no integration.
- **Dynamics.** Radau integrator and fresh random data.

Where a statement is false, the script checks the corrected statement:
- y5-5e: libration drift.
- y5-5g: exchange hypothesis.
- y5-3e: reason for the vanishing odd traces.

The section's quoted numbers were also recomputed from the derivation script's seeds (y5-5f):
- E_0 = 1.2156
- diameter 0.942
- R_eff = 0.391
- eps = 0.098

## Findings

1. **REFUTED** (minor scope). Location: Thm `e5:thm:stab`(iv), "Off the surface the rotor librates about $\psi=\pi$ with $\Omega^2=2R_{\rm eff}/I_0$ and period $4\mathrm K(\sin^2(\xi_0/2))/\Omega$, which diverges as $R_{\rm eff}\to0$, and the frames drift."
   - **What is wrong:**
     - The frames do not drift at one libration amplitude.
     - The rotor does not librate at the unstable equilibrium $\psi_0\in2\pi\Z$.
   - **Evidence (y5-5e):**
     - Write $\psi=\pi+\xi$. Then $\dot x_k=-e^{-\varphi_{\rm eff}G}(\cos\xi+G\sin\xi)N_kp^{(k)}(0)$.
     - Over one libration period the sine part integrates to zero. With $\sin(\xi/2)=\sin(\xi_0/2)\sin t$, which is the substitution of RE Prop. 4.4, $\int_0^T\cos\xi\,d\tau=(4/\Omega)(2E(m)-K(m))$, where $m=\sin^2(\xi_0/2)$.
     - So the mean velocity is $(1-2E(m)/K(m))\,e^{-\varphi_{\rm eff}G}N_kp^{(k)}(0)$. This matches a full integration of the closure to $10^{-13}$ (relative) at $\xi_0=1.0$ and $2.5$.
     - It vanishes at $2E(m)=K(m)$: $m^*=0.826115$, $\xi_0^*=130.71^\circ$. For a datum with $h(0)=0$ and $\varphi_{\rm eff}=\pi-\xi_0^*$, each frame returns to its start after every period: $|x(5T)|<10^{-9}|Np|T$, while $\max|x|=O(|Np|T)$. Those frames are bounded.
     - At $\psi_0=0$ with $h(0)=0$ the vector field of $(\theta,h,p)$ vanishes. The datum is an unstable equilibrium, so the rotor does not librate.
     - $\xi_0$ is not defined in the section. The script e5-5e checks one datum only, which has $\xi_0=1.48$.
   - **Corrected wording:** "Off the surface, unless $\psi_0\in\pi\Z$ (an equilibrium), the rotor librates about $\psi=\pi$ with $\Omega^2=2R_{\rm eff}/I_0$, amplitude $\xi_0\in(0,\pi)$, $\cos\xi_0=-\cos\psi_0$, and period $4\mathrm K(m)/\Omega$, $m=\sin^2(\xi_0/2)$, with $\mathrm K,\mathrm E$ the complete elliptic integrals in the parameter $m$ \cite{DLMF}. The period diverges as $R_{\rm eff}\to0$. The frames move with mean velocity $(1-2\mathrm E(m)/\mathrm K(m))\,e^{-\varphi_{\rm eff}G}N_kp^{(k)}(0)$. They drift except at the one amplitude with $2\mathrm E(m)=\mathrm K(m)$ ($m=0.8261$, $\xi_0=130.7^\circ$), where every frame returns to its initial position after each period." Add the two-line proof given above.

2. **WRONG-DETAIL** (missing hypothesis). Locations:
   - Prop `e5:prop:exchange`(iii): "The exchange eigenstates of the momenta, $p^{(2)}=\pm p^{(1)}$, both have $w_2=w_1$ and are balanced only when null. A massive balanced pair is never an exchange eigenstate of its momenta."
   - Introduction item (5): "A massive balanced pair is never an exchange eigenstate of its momenta (fixed".
   - **What is wrong:** item (iii) carries no equal-coefficient hypothesis, unlike (ii). With $c_2=-c_1$, which is a legitimate pair of Lorentz-class inertias, $F=(c_1+c_2)w_1=0$ for every exchange eigenstate, massive ones included.
   - **Evidence:** y5-5g, 20 random massive data with $c_2=-c_1$: $|F|<10^{-12}$ and $|w_1|>10^{-6}$. Symbolically, $F(p,\pm p)=(c_1+c_2)w(p)$.
   - **Corrected wording:**
     - In (iii): "both have $w_2=w_1$, so $F=(c_1+c_2)w_1$. Unless $c_2=-c_1$ they are balanced only when null. With equal inertias, a massive balanced pair is never an exchange eigenstate of its momenta."
     - In the introduction: "with equal inertias a massive balanced pair is never ...".

3. **WRONG-DETAIL** (reason). Location: proof of Prop `e5:prop:spectator`(iii), "Integer isospin makes every odd trace of $h$ vanish on each half separately."
   - **What is wrong:** odd traces vanish for every representation of $\su2$, because the weights are symmetric under $m\mapsto-m$. Integer isospin is not the reason. What integer isospin adds is the absence of doublets, hence of the global anomaly.
   - **Evidence:** y5-3e. The odd traces vanish exactly also for $\so N\cong\su2$ of the spectator algebra, which acts on the Weyl block by doublets ($\so N$ Casimir $3/4$ on all of $S_2$).
   - **Corrected wording:** "Every representation of $\su2$ has weights symmetric under $m\mapsto-m$, so every odd trace of $h$ vanishes on each half separately. Integer isospin excludes doublets, and with them the global anomaly."

4. **CLASS.** Location: Answer paragraph, "Yes, in a precise sense (fixed)".
   - **What is wrong:** the claims table classifies `e5:sec:ladder` as "interpretive (drawn from fixed statements)". Whether $n=2$ is "a closure rung in its own right" is a reading of fixed facts.
   - **Corrected wording:** "Yes, in a precise sense (interpretive, drawn from the fixed statements cited below)".

5. **OVERCLAIM.** Location: Answer, "every structure of the three-constituent analysis of \cite{ES17} has a computed $n=2$ analogue". Also the introduction: "This section supplies the $n=2$ analogue of every item".
   - **What is wrong:**
     - The projectors $\Pfam$ and $P_*$ are defined only at three constituents (ES17 Sections 7.2 and 8). The ladder table shows "---" at $n=2$ for "under the projectors".
     - At $n=2$ the section replaces "what survives each projector" by what survives each Lorentz action's gauge algebra.
   - **Corrected wording:**
     - Answer: "every structure of the three-constituent analysis of \cite{ES17} except the two projectors, which exist only at $n=3$, has a computed $n=2$ analogue".
     - Introduction: "the $n=2$ analogue of every item, with the five Lorentz actions in place of the two projectors, which exist only at $n=3$".

6. **CITATION.** Location: end of `e5:sec:symmetry`, "the frame energy is $\half\Re(c\,q_{\rm tot}(\pi))$ (Proposition~7.9 of \cite{RE})".
   - **What is wrong:** RE Prop. 7.9 (`d5:prop:balance`) is about duplicated momentum fields, $\pi^{(1)}=\alpha\pi$, $\pi^{(2)}=\beta\pi$. The equal-inertia formula for general data is Prop. 17.18 of ES17 with $c_1=c_2$, which is `e5:prop:rotorlaw`(i) here.
   - **Corrected wording:** "(Proposition~17.18 of \cite{ES17}; \cref{e5:prop:rotorlaw}(i) with $c_1=c_2=c$)".

7. **CITATION** (minor). Location: caption of `e5:tab:ladder`, "The $n=3$ column restates Theorems~17.19--17.24 of \cite{ES17}, except the spectator and diagonal rows".
   - **What is wrong:** two $n=3$ rows come from elsewhere:
     - The Pati–Salam row ("in $\so8\oplus\so6$") is Section 10.7 of ES17.
     - The balance-surface row ("codim 2, dim 16") is Theorem 8.1(iii) of RE.
   - **Corrected wording:** "The $n=3$ column restates Theorems~17.19--17.24 and Section~10.7 of \cite{ES17} and Theorem~8.1(iii) of \cite{RE}, except the spectator and diagonal rows, which are computed here (e5-6a)."

8. **UNCLEAR** (undefined symbols).
   - **What is undefined:**
     - (a) $\Omega^{(1)}$ in Prop. `e5:prop:spectator`(iii): the interior volume element $E_{(1,1)}E_{(1,4)}E_{(1,5)}E_{(1,6)}$ (ES17 Section 5.1).
     - (b) $\xi_0$ and the convention for $\mathrm K$ in Thm `e5:thm:stab`(iv). The entry `DLMF` of `bib_e5.tex` is never cited.
     - (c) $K$ in Prop. `e5:prop:carry`(iv), "the rotor acting by $K\otimes G$". It is undefined, and it clashes with the kernels $K^\mu$ and with the para-complex structure of the paper. It is $K=\operatorname{diag}(\kappa_1,\kappa_2)$ of RE Section 9.
   - **Corrected wording:**
     - (a) "and $\Gamma=w\,\Omega^{(1)}$, with $\Omega^{(1)}=E_{(1,1)}E_{(1,4)}E_{(1,5)}E_{(1,6)}$ the interior volume element of constituent~1, is ...".
     - (b) As in finding 1.
     - (c) "with the rotor acting by $\operatorname{diag}(\kappa_1,\kappa_2)\otimes G$ (Proposition~9.2 of \cite{RE})".

9. **UNCLEAR** (notational slips).
   - **The slips:**
     - (a) Proof of Prop. `e5:prop:diagonal`(ii): "cross terms $E_{(1,0)}E_{(2,\mu)}$". Index 0 is not a generator index; it should read $E_{(1,1)}E_{(2,\mu)}$ ($\gamma^0=e_1$).
     - (b) Prop. `e5:prop:carry`(i): "$\bar z_jz_k$, with $z\in\C^6$". Here $z_k$ elsewhere denotes the 3-vector of constituent $k$; use $\bar z_az_b$, $a,b=1,\dots,6$.
     - (c) Thm `e5:thm:pair`(iii): "$R^{(1)}_\varphi\in\so6$". This is a group element, so write $\in SO(6)$.
     - (d) Thm `e5:thm:stab`(iv): "$x_k=\tau N_kp^{(k)}(0)$" assumes $x_k(0)=0$; write $x_k=x_k(0)+\tau N_kp^{(k)}(0)$.
   - **Corrected wording:** as given in each item.

10. **OVERCLAIM** (minor). Location: after Prop. `e5:prop:carry`, "The copy rotor of Remark~8.10 of \cite{RE}, ... is the one structure of the rest closure that exists only from $n=2$ on."
    - **What is wrong:** item (iv) of the same proposition has the two-rate closure, a single rotor acting by $\operatorname{diag}(\kappa_1,\kappa_2)\otimes G$ with $\kappa_1\neq\kappa_2$ (RE Prop. 9.2). It also exists only from $n=2$ on.
    - **Corrected wording:** "The copy rotor of Remark~8.10 of \cite{RE}, ..., and the two-rate rotor of (iv) are the structures of the rest closure that exist only from $n=2$ on."

11. **UNCLEAR / CLASS** (minor). Location: closing paragraph of `e5:sec:ladder`, "In general the binding-compatible parts are $\so{3n-3}$ and $\so n$."
    - **What is wrong:** the statement carries no proof or class, and the checks stop at $n=3$.
    - **Evidence:** the statement is correct.
      - The proofs of Prop. `e5:prop:spectator`(i) and `e5:prop:diagonal`(i) apply verbatim at every rung.
      - y5-6a confirms dimensions (36, 6) at $n=4$ exactly.
    - **Corrected wording:** "At every rung the binding-compatible parts are $\so{3n-3}$, the paired rotations of constituents $2,\dots,n$, and $\so n$, the copy rotations: the proofs of \cref{e5:prop:spectator}(i) and \cref{e5:prop:diagonal}(i) apply verbatim (fixed; checked at $n\le4$, e5-6a)."

## Verified as correct

- **Prop. so6C (y5-1a).**
  - $G_2$ identities.
  - $\dim\Cent(G_2)\cap\so{6,6}=30$, spanned by 15 paired rotations and 15 paired boosts.
  - Block form $[[R,S],[-S,R]]$, and $R-iS$ is a Lie isomorphism onto $\so{6,\C}$.
  - The compact part is the 15 paired rotations.
- **Prop. quad (y5-1b).** Real dimensions 2, 4, 6, with the stated spans. On the complex side the holomorphic invariants have dimensions 1, 2, 3.
- **Thm pair.**
  - (i) The identities, and the $M_{\rm pair}^2$ formula against the singular values. Also against the observer infimum of RE Thm 8.6 by direct minimisation (y5-1c). Non-invariance under one-constituent boosts and rotations; invariance under the diagonal action.
  - (ii) Stabiliser exactly $\lG^\Delta\oplus\R C_v$, dimension 7, by exact rank 23 at 30 Gaussian-integer points (y5-1d). The section's numerical claim is right, and it can be upgraded to exact.
  - (iii) Copy-boost eigenvector, $M_{\rm pair}=e^{2t}$, and $R^{(1)}_\varphi$ giving $\sin\varphi$ (y5-1e).
  - (iv) Orbit dimension 10, the bound, and the $\cosh 2t$ family (y5-1f).
- **Prop. nocommon.**
  - (i) $\Cent\cap\lint^{(1)}=0$. The commuting algebra is 0, 6, 30, 72 at $n=1..4$ ($=3(n-1)(3n-4)$): $\lG^{(2)}$ at $n=2$, and the centraliser on the tail for $n\ge3$ (y5-2a).
  - (ii), (iii) Bivector structure; $2(\half,\half)$ and $2(\half,0)+2(0,\half)$ at one constituent; $16(\half,\half)$, $16(\half,0)+16(0,\half)$, $16(\half,\half)$, and $4[(0,0)+(1,0)+(0,1)+(1,1)]$ (y5-2b, 2c).
- **Prop. noPS (y5-3a).** Dimensions 28, 15, 7, 1, 2, with compact parts 13, 6, 7, 1, 1. The algebras are transpose-closed, and the dimension count holds.
- **Prop. spectator.**
  - (i) $\so{5,3}$, $\so P\oplus\so N$, the intersections, the graph form, and su(2) brackets (y5-3b).
  - (ii) Kinetic null space is exactly the 3 paired rotations of constituent 2; the boosts fail (y5-3c).
  - (iii) $w^2=-1$, commutation, $\Gamma$; Weyl blocks $(\half,0)\otimes(4,2)$ and $(0,\half)\otimes(4,2)$; su(2) content $2\cdot1+2\cdot3$ on both halves; anomaly traces zero (y5-3d, 3e).
- **Prop. diagonal.**
  - (i) Dimension 7, compact, $\cap\Cent(G_2)=\R C_v$, $C_v=M_{12}+C_{\rm surp}$, $S(C_v)=-C_b$ (y5-3f).
  - (ii) Summed kernels hold, and $f_0f_\mu$ fails for $\mu=1,2,3$ (y5-3g).
  - (iii) The content table is exactly right, including which $\Gamma_{12}$ sign carries the odd charges; odd traces vanish (y5-3h).
- **Prop. mixed and the binding actions (y5-3i, 3j).**
  - Brackets; $V_2=V+\Lambda^2V+2\cdot\mathbf1$; centraliser $\R J_1$; $\cap\Cent=0$.
  - 7-dimensional equivariant endomorphisms, with antisymmetric part $\{G^{(2)},J_1\}$.
  - $\so{3,3}^{(2)}$ and $\so{2,\C}$ statements.
- **Thm inertia (y5-4a, 4b).** All eight rows; $(2,0)$ at $n=1,2,3$; $(12,9)$, $(42,36)$, $(90,81)$. The stabiliser forms equal the closure-symmetry forms; the $\lG^\Delta$ Lorentz forms are Re/Im $z_k\cdot z_l$.
- **Prop. rotorlaw (y5-5a).** Symbolic identity, and the cross-term rotor law along the leaf, with no integration.
- **Prop. Bgeom (y5-5b).**
  - The null-cone parametrisation for unequal coefficients.
  - rank dF = 2; all-null part of dimension 8; massive part of rank 4, with $|c_1|E_0^{(1)}=|c_2|E_0^{(2)}$.
- **Thm stab.**
  - (i), (ii) (y5-5c).
  - (iii): exact series $\lambda=\frac{\pi\varepsilon}{4|\kappa_0|}(1-\frac32\varepsilon\cos\psi_0)$, period $\frac\pi{|\kappa_0|}(1-\frac12\varepsilon\cos\psi_0)$, and $\bar v$ to second order; displacement per period by Radau (y5-5d).
  - (iv), the balanced $h(0)=0$ part, and the libration period (y5-5e).
  - Quoted numbers (y5-5f).
- **Prop. exchange (y5-5g).** (i) and (ii) as stated: isometry, commutation with $G_2$, $H$, $\det=-1$, $\sigma V_+=V_-$.
- **Prop. carry (y5-5h).**
  - Poisson rank 2; 11 and 5 Casimirs, with their invariance.
  - $\Gamma_{12}$; the isometry formula.
  - `e1:prop:rest`(v) frequencies; $U\equiv0$ on leaves through $\mathcal B$.
- **Ladder (y5-6a, 6b).**
  - $n=1$: $\R J$.
  - Spectator/diagonal compact algebras: 1, 13, 43 and 1, 7, 18.
  - Binding-compatible parts (0,0), (3,1), (15,3), (36,6); $\so4_\Delta$ inside; $\mathfrak{ps}_U\cap\Cent(G_3)=\so4_\Delta$.
  - Codimension 2 at $n=1,2,3$.
- **Citations.** All ES17 and RE numbers were checked against `paper_labels.txt`, `re_labels.txt`, RE `main.aux` (for `d5:prop:so6C`=7.8, `d5:tab:actions`=Table 8) and the cited text, except findings 6 and 7. Examples: Prop. 17.18; Thms 17.19–17.24; Section 17.6 = three constituents; eq. (30); Prop. 5.5; Prop. 4.5; Thm 12.1; Thm 10.8; Section 10.7; Prop. 10.12; Section 10.8; Props 10.9, 13.15, 10.2, 13.16, 13.17, 17.4; Def. 17.10; Thm 17.11; Cor. 17.15; Thm 17.16; Thm 17.7; RE 7.3, 7.8, 8.1, 8.3, 8.4, 8.6, 8.8, 8.9, 8.10, 9.1, 9.2, 9.6, 4.1, 4.3–4.10, 4.14–4.16, Table 8; e1:prop:rest(v).
- **Literature.**
  - Pati–Salam, Phys. Rev. D 10 (1974) 275–289.
  - Witt, J. Reine Angew. Math. 176 (1937) 31–44.
  - Witten, Phys. Lett. B 117 (1982) 324–328.
  - Weyl 1939 (first fundamental theorem).
  - All checked by web search.
- **Claim classes.** Honest throughout, except findings 4 and 11.

## Second look (step 2)

Status after the refute-first second look:
- **Finding 1: kept.**
  - Second computation, mpmath at 20 digits, independent of the closure ODE:
    - Quadrature of $\int\cos\xi\,d\tau$ over a libration with the substitution $\sin(\xi/2)=k\sin t$. It equals $4(2E-K)$ at $m=0.2$ and $0.95$, and $-1.3\times10^{-21}$ at $m^*=0.826114765984970336$.
    - A Taylor-series ODE integration of $\ddot\xi=-\sin\xi$ over one period at $m^*$ gives $-3.1\times10^{-21}$.
  - An independent DOP853 run gives $|x(T)|\sim3\times10^{-13}$ at $\xi_0^*$, against $|x(T)|\approx3.7$ at $\xi_0=1$.
  - "Drift" in this note and in ES17 Thm 17.16 means unbounded motion. No hypothesis of (iv) excludes the amplitude.
- **Finding 2: kept.** Symbolically $F(p,\pm p)=(c_1+c_2)w(p)$. The Lorentz class $(a,b)\in\R^2$ allows $c_2=-c_1$, and (iii) carries no equal-coefficient hypothesis. Severity stays WRONG-DETAIL, because the intended reading is plausibly the equal-coefficient setting of (ii).
- **Finding 3: kept.** For the doublet-carrying $\so N$, the odd traces are exactly zero (y5-3e). So "integer isospin" is not what makes them vanish.
- **Findings 4–7: kept.** Re-read against the claims table, `s07_projectors.tex` (projectors only "at three constituents"), RE `d5_duplication.tex` Prop. 7.9 and ES17 Section 10.7.
- **Findings 8–11: kept.** These are presentation items.
- **Withdrawn: none.**

## Applied (step 3)

All 11 findings were applied to `sections/e5_closure_two.tex` as minimal edits in the wording proposed above.

New PASS lines in `verify/e5_closure_two.py` for the changed claims:
- **e5-3j:** odd traces also vanish for the doublet-carrying so(3) on N.
- **e5-5i:** the libration mean-velocity law $(1-2E/K)$ at four amplitudes, zero at $2E=K$; equilibrium at $\psi_0=0$.
- **e5-5j:** $F(p,\pm p)=(c_1+c_2)w(p)$; $c_2=-c_1$ balances massive exchange eigenstates.
- **e5-6b:** binding-compatible parts (36, 6) at $n=4$.

The e5-5e description now says "at this amplitude".

Updated cross-references:
- The section cites e5-3j, 5i, 5j and 6b.
- The claims table lists the new ids.
- The header comment reads "e5-1a ... e5-6b".
- `\cite{DLMF}` is now used.

Results:
- `e5_closure_two.py`: 32 PASS, exit 0, 17 s.
- `y5_check.py`: 31 PASS, exit 0, about 37 s.
- Compile check in `scratchpad/compile_e5_v` (bib_* merged into references.tex, pdflatex three times): 33 pages, no errors, no undefined references, no overfull boxes, no warnings.
