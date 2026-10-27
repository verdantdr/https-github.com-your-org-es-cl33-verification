# Pending work (paused 26 October 2026, to resume next week)

Everything below is committed in its current state. Prompts used for the agents are in `prompts/`, and the referee findings already
applied are in `findings/`.

## Done and verified

| Item | Location | Checks |
|---|---|---|
| Paper with the four rest-energy sections (Sections 15, 16, 20, 21 of the new numbering) | `tex/`, `ES_electromagnetic_sphere_2026-10-26.pdf` | builds, 154 pp. |
| Rest-energy note | `investigations/rest_energy/` | 14 scripts, 464 PASS |
| Two readings of X (Lagrangians; four-momentum on the spatial triple; X_EM=(B;E) on the octants; bridge T=nJ(n)) | `investigations/two_readings/` (50 pp.) | 8 scripts, 252 PASS; r1–r4 refereed |
| Emergence: two from one (e1), processes (e2), ladder (e3), b-field up the ladder (e4), closure at n=2 (e5, refereed), b-field on +++ / c on −−− (e6, refereed), classical readings (e7) | `investigations/emergence/` (draft) | e1–e7 scripts pass |
| Photon selection: compact families at n=3 (p1) | `investigations/photon_selection/` (22 pp., draft) | p1 47 PASS |

## Unfinished

1. ~~**Two readings, track r4 (bridge T=nJ(n))**~~ — done 26 October: referee applied 15 corrections (`findings/s4_findings.md`); `r4_bridge.py` 25 PASS, `s4_check.py` 24 PASS; note rebuilt (50 pp.).
2. ~~**Emergence, track e6**~~ — done 26 October: referee applied 14 corrections (`findings/y6_findings.md`); `e6_bfield_triples.py` 33 PASS, `y6_check.py` 33 PASS.
3. ~~**Emergence, tracks e3 (ladder) and e4 (b-field up the ladder)**~~ — sections written 26 October (`e3_ladder.py` 22 PASS, `e4_bfield_ladder.py`
   24 PASS); not yet refereed: `prompts/check_e3.txt`, `prompts/check_e4.txt` (add STEP 2/3 as in `referee_e6.txt`). Small fix for e1:
   in `e1:prop:Nlorentz`(i) the product (1/8)prod(1 + eta E E) equals exp(-pi C_b/2), not exp(+pi C_b/2) (harmless).
4. ~~**Emergence, track e7 (classical physics: astronomy, cosmology)**~~ — section written 27 October (`e7_classical.py` 27 PASS); not yet
   refereed (`prompts/check_e7.txt`). Partly confirmed references are listed in the e7 agent report: Gould–Schreder title, Nikishov title,
   Kaspi–Beloborodov numbers, Lorimer–Kramer typical field, IGRF range, Goodman sections, Planck age, CFJ radio-galaxy bound.
5. **Emergence, referees for e1 and e2**: partial independent scripts `y1_check.py`, `y2_check.py`, findings `findings/y1_findings.md`.
   Resume: `prompts/check_e1.txt`, `prompts/check_e2.txt` (combine with a STEP 2/3 as in `referee_e6.txt`).
6. **Emergence, intro, synthesis, methods, abstract** (`e0_intro`, `e8_synthesis`, `e9_methods`): not written.
7. **Photon selection, tracks p2 (alignment by a gauged subalgebra) and p3 (which charge is the photon)**: p3 section drafted
   (`p3_identification.tex`, script `p3_identification.py`, unchecked, not refereed); p2 has a partial script only. Resume:
   `prompts/derive_p2_v2.txt`, `prompts/derive_p3_v2.txt` (both end with the p1 results to build on).
8. **Photon selection, track p4 (the gluon, if the obstruction persists at n≥3)**: not started. Resume: `prompts/derive_p4.txt`.
9. **Photon selection, intro and synthesis**: not written.
10. **Paper integration** of the emergence, two-readings and photon findings as a new Part in house style, with the introduction,
    the assessment (Section 35), Appendix B and the references updated; then a new dated build.

## Notes for resuming

- The prompts refer to a scratch directory of the previous session; the files they name under `scratchpad/prompts/` are the copies
  in `prompts/` here, and the findings under `scratchpad/wf/` are in `findings/`.
- The two-readings `references.tex` now contains the merged entries of `bib_r1`–`bib_r4`; a new track must add only new keys.
- The emergence and photon notes still keep their references in `bib_*.tex` files; merge them (each key once) before a final build.
- Sign convention to harmonise in the emergence note: e1, e3 and RE use C_b = C^{12} = +(1/2) sum_a eta_aa E_(1,a)E_(2,a); e2 and e6 use the
  opposite sign, and e6 also calls it C_b. Rename or align before the final build.
- The author's instruction for the field reading: `X_EM = (B,E)` is the pair labelling the six components of the field two-form,
  never a sum `E + B`.
