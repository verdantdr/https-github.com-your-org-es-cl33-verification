# The electromagnetic sphere — unified paper (26 October 2026)

A single, self-contained academic paper collecting the findings on the split-signature Clifford algebra Cl(3,3) and its tower,
written bottom up and without references to earlier ES papers, reports or versions.

- `ES_electromagnetic_sphere_2026-10-26.pdf` — the current paper, 154 pages, with four new sections (15, 16, 20, 21) on the undivided
  energy and its differentiation, what a rest energy breaks, rest energy in the closure, and the connection part of the shift.
- `ES_electromagnetic_sphere_sources_2026-10-26.zip` — its LaTeX sources (same as `tex/`).
- `ES_electromagnetic_sphere_2026-10-17.pdf` — the previous version, 119 pages, 21 figures, 7 tables, 151 references (cited by the
  investigation notes as ES17).
- `ES_electromagnetic_sphere_sources_2026-10-17.zip` — its LaTeX sources.
- `investigations/` — verified investigation notes (rest energy; two readings of X; emergence; photon selection) and
  `investigations/_pending/PENDING.md`, the list of unfinished work.
- `tex/` — sources: `main.tex` plus one file per section (`s01`–`s31`), appendices `sA_realisation`, `sB_methods`, and `refs.tex`.

## Structure

| Part | Sections | Content |
|---|---|---|
| — | 1 | Introduction, deflationary principle, four claim classes, summary, conventions |
| I | 2–5 | Arena at one constituent: realisation, signature rule, pole grading, clock, two roots of the pairing, imaginary unit, discrete symmetries, Fock space |
| II | 6–10 | Tower: centralisers, projector family, P\*, charges and anomalies, diagonal sector, every Lorentz action |
| III | 11–14 | Projector as a field, field-strength reading and Maxwell, spin one in both readings, E·B orthogonality |
| IV | 15–17 | Interior dynamics: positive form, neutral-signature binding, closure of the force law |
| V | 18–22 | Vacuum: subfields, 272 mass kernels, selection of the Standard-Model orbit, matter, Goldstone/CP/baryon number/topology |
| VI | 23–26 | Curved arena, curvature operator and phase family, projector transport, massless photon |
| VII | 27–30 | Causal structure: baseline, kinematic selector, type selection, surplus constraints |
| VIII | 31 | What the algebra decides (fixed / constructed / obstructed / open) |
| App. | A, B | Explicit matrices; how the statements were verified |

## Build

```
pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex
```

run three times. Packages: amsmath/amsthm/mathtools, booktabs/longtable/array/enumitem, tikz (+ libraries), pgfplots 1.16, hyperref, cleveref.
No `caption`, `subcaption` or `microtype` (not installed in this MiKTeX).

## Notes

- **Errata (17 October 2026, after an external review).** Ten statements (Theorems 18.1 and 18.7, a benchmark sentence in
  Section 20, the global-minimum claim, the Carroll–Field–Jackiw remark in Section 14, the τᵢ convention of Section 19, P_ν, a sign in Section 5, the rescaling remark of Section 17 and the wording of Example 17.17) and two hidden assumptions (a single Yukawa coupling,
  and the normalisation of $y$, $g$, $v$ between Sections 19–21) were corrected in `tex/`. The PDF and source zip were rebuilt from the corrected
  sources. No statement was renumbered. Details and an executable check: `investigations/errata_2026-10-17/`.
- Status labels and program listings are deliberately absent; qualifications are written into the text, and Appendix B describes the
  verification (exact computation with a second independent instrument for the central statements; numerical items named explicitly).
- The appendix matrices were regenerated and checked (Clifford relations entry by entry) by a short exact script during writing.
