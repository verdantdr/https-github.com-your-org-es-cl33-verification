# Errata to *The electromagnetic sphere* (17 October 2026)

An external review of the 119-page paper raised five mathematical defects and two points about hidden assumptions and
normalisations. Each point was re-derived independently before anything was changed. All seven were confirmed, and the
sources in `tex/` were corrected.

Every check is in `verify_errata.py` in this folder. It prints one `PASS` line per check and exits 0 only if every check
passes (19 checks, about 5 s):

```
python3 investigations/errata_2026-10-17/verify_errata.py
```

**No theorem, proposition, section or equation was renumbered.** The `.aux` label tables before and after the corrections
are identical, so every cross-reference that cites a number (inside the paper, in `investigations/rest_energy/`, or
elsewhere) stays valid. The corrected build has 120 pages instead of 119.

| # | Where | Defect | Status | Correction | Check |
|---|---|---|---|---|---|
| 1 | Thm 18.7 (*A universal Fock vacuum*), `s18_subfields.tex` | The stated hypotheses (A, Q antisymmetric and commuting, A² = −E², Q² = −1) do not imply a neutral ground state. Counterexample: A = E e₄, Q = e₄ gives a ground state of charge ±2. | confirmed | New hypothesis **Tr(AQ) = 0**. Over all 256 commuting pairs, the joint-spectrum table holds exactly when this is satisfied. A proof was added, the counterexample is recorded, and the charges the theorem names (ω_T on T, J on S, ω_W on (1,2) planes) are shown to satisfy it. | E1a–E1d |
| 2 | Thm 18.1 (*Complex structure or projector*), `s18_subfields.tex` | The transpose rule ω_Wᵀ = −(−1)^q ω_W was stated for *every* three-plane. It fails for tilted planes: ω for ⟨cosh t e₄ + sinh t e₁, e₅, e₆⟩ is not symmetric, although ω² = 1. | confirmed | Statement restricted to coordinate planes. A paragraph records that the square rule and the (anti)commutation with W and W^⊥ hold for every nondegenerate plane, and the transpose rule for every plane spanned by vectors of V⁺ and V⁻. | E2a–E2d |
| 3 | after Thm 20.5, `s20_selection.tex` | Benchmark sentence reversed. With f(𝟙⊗𝟙) = (1/3, 1/6, 1/3, 1/6, 0) and d = 1/16, Λ_κ(𝟙⊗𝟙) − Λ(S₀) = 0.9 − \|κ\|/8. The 𝟙⊗𝟙 orbit is therefore the **higher** minimum for \|κ\| ≤ 7 and the **lower** for \|κ\| ≥ 7.5, crossing at 7.2. | confirmed (the threshold 7.2 itself was right) | Sentence reversed, and the explicit difference added. | E3a–E3e (E3e also reproduces the paper's 48 − 7x/4 of Thm 20.9) |
| 4 | abstract (`main.tex`), §1 summary (`s01_intro.tex`), §20 opening, §31 (`s31_assessment.tex`) | Claimed "necessary and sufficient conditions" for the Standard-Model orbit to be the global minimum. Thm 20.5 gives sufficient conditions (b) and separate necessary ones (c), with a gap between them (at the benchmark: 2.8 vs 7.2). | confirmed | All four places now say "sufficient and, separately, necessary conditions", with the gap resolved numerically, not proved. x* ≈ 24.884 was already stated as a conjecture. | — (wording) |
| 5 | §14, *Topological terms*, `s14_orth.tex` | "a constant gradient gives birefringence without a gap": false for a spacelike gradient. At zero wave vector one branch has ω = \|p\|. | confirmed | Narrowed: a timelike gradient leaves both branches gapless. A spacelike gradient gaps one branch, but this is a Lorentz-violating dispersion fixed by an external pseudoscalar, not a Proca mass. The section's conclusion (E·B alone generates no mass) is unchanged. | E4a–E4c |
| 6 | Assumption 19.4 and after Prop 19.5 (`s19_kernels.tex`); §20.5 restrictions (`s20_selection.tex`); §21 (`s21_matter.tex`) | "the single Yukawa coupling y, since Φ is one field": Φ is reducible under Pati–Salam, so symmetry allows one Yukawa coupling and one kinetic normalisation per irreducible component (plus mixing of the two (6,1,1)). The same applies to the common gauge coupling of su(4) and su(2)_R. | confirmed | A single y, one trace form and a common g are now listed as parts of Assumption 19.4 and among the restrictions of the radiative bounds. The §21 sentence "matter opposes the SM vacuum" is qualified accordingly. | — (logic) |
| 7 | §20.4–20.5 (`s20_selection.tex`) | Unstated change of normalisation. §19 uses M_N = \|y\|r, §20.4 uses M_N = yv with v = \|S\|, and §20.5 and §21 use v = \|Φ\|/√2 = √2 r. The two y's differ by √2, which made the translation of x into mass ratios ambiguous. | confirmed | §20.4 now states that S = √2 ι(Φ) is the canonically normalised field (the kinetic term of Assumption 19.4 is Tr(DS)†(DS)), so v = \|S\| = \|Φ\|/√2 throughout, g is the coupling of Thm 21.1, and y(§20.4) = y(eq. (35))/√2. Then x = Σ y⁴/(4g⁴) in §19/§21 couplings, which equals Σ(M_N/M_W_R)⁴. **The physical bounds (M_N < 2.280 M_W_R, etc.) are unchanged.** | E5a, E5c, E5d |
| 8 | §20.5, after Table 5 (`s20_selection.tex`) | "Radiatively generated masses" are Hessian curvatures of the leading-log potential, not pole masses. | added qualification | One sentence: the masses are curvatures divided by the kinetic norm; pole masses would need wave-function renormalisation and a gauge-consistent expansion. | — |

## What was *not* changed

- **The central constructions and numbers:** rank 64 of P_*, 23 internal bivectors, 272 = 144 + 128 kernels, Φ₀ᵀΦ₀ = P_ν, the 12 + 9 split of the
  vector masses, and 108 − 4x with the threshold 27. The review reproduced all of them independently. The checks here also reproduce the
  (W_R : X : Z′)² = 1 : 1 : 5 pattern and c₁ = 33 (E5a).
- **x* ≈ 24.884:** it stays a numerical conjecture, as the paper already said.

## Files changed

- `tex/main.tex`, `tex/s01_intro.tex`, `tex/s14_orth.tex`, `tex/s18_subfields.tex`, `tex/s19_kernels.tex`, `tex/s20_selection.tex`,
  `tex/s21_matter.tex`, `tex/s31_assessment.tex`.
- `ES_electromagnetic_sphere_2026-10-17.pdf` rebuilt from the corrected sources (TeX Live 2023, three pdflatex passes; the original
  MiKTeX build is in git history and on branch `17-10-unified`).
- `ES_electromagnetic_sphere_sources_2026-10-17.zip` regenerated from the corrected `tex/` (same `ES_unified_paper_sources/tex/` layout,
  with forward-slash paths).

## Still recommended (not done here)

- A reproducibility package for the paper's own central computations: the P_* projector, the 272-kernel basis, the 271×271 Hessian and
  the random-start searches (Appendix B describes them, but the scripts are not in the repository).
- A robustness study of vacuum selection once the common couplings and the degenerate tree potential are relaxed. This is the review's step 3.
