# The undivided energy and its differentiation — $E=mc^2$ in the arena

Investigation note to *The electromagnetic sphere* (17 October 2026, `tex/`), testing the hypothesis

> *$E=mc^2$: $c^2$ is the symmetrical part of the equation and $m$ is what breaks this symmetry; $m$ is generated from a $b$ particle,
> $b$ standing for the $b$-particle of Yang–Mills* — with the energy one undivided, six-dimensional entity whose internal differentiation
> produces the mass, before any external interaction.

- `main.pdf` — the note (built from `main.tex`, `preamble.tex` and `sections/`).
- `sections/` — `s0_intro` (the question, the dictionary, how claims are classified); `d1_unitary_energy` (the undivided energy and its
  differentiation); `d2_symmetry_breaking` (what a mass breaks; particles, antiparticles, vacua, Goldstone bosons); `d3_closure` (the
  differentiation in motion: rotor, rate and duality angle); `d4_bfield` (the $b$-field as the antisymmetric part of the shift);
  `s5_synthesis` (the hypothesis clause by clause); `s9_methods`.
- `verify/` — `d1`–`d4`: derivation checks; `x1`–`x4`: independent adversarial re-checks; `run_all.py` runs all eight and writes
  `verify/RESULTS.md`.
- `paper_labels.txt`, `paper_index.txt` — label → number tables of the paper, used for every citation.

## Verdict in one table

| clause | what the arena gives | verdict |
|---|---|---|
| energy undivided, six-dimensional | $E_6=\tfrac12(\|B\|^2+\|E\|^2)$: the unique rotation- and duality-invariant quadratic energy, positive, pointwise symmetry $U(3)$; a Casimir of the momentum field in the closure | holds (inherited / restates the paper) |
| mass from internal differentiation | $E_6^2=E_0^2+\|S\|^2$ with $E_0=\tfrac12\|q\|$ the Lorentz×duality invariant and $E_0=\min_{\text{boosts}}E_6$; the field-point form of $E^2=(pc)^2+(mc^2)^2$ | holds as an identity, not as a dynamical process |
| $c^2$ the symmetric part | $c$ is a Schur-forced conversion, not a symmetry; $\kappa G$ is the $\eta$-self-adjoint summand of the shift, $(\kappa G)^2=-c^2/\ell^2$, the only summand that binds | holds in the shift decomposition (interpretive reading of a fixed result) |
| $m$ breaks the symmetry | breaks duality ($U(1)\to\mathbb Z_2$) and, for matter, $\mathfrak u(1)_\Omega\oplus\mathfrak{su}(2)\to\mathfrak u(1)_J$; the Dirac mass form $e_1e_2e_3$ is invariant under all of $\mathrm{Spin}(3,3)$ | true for internal symmetries, false for spacetime ones |
| $m$ generated from the $b$-field | $b$ is the connection summand: it splits (commuting part, Zeeman-like) or destabilises (anticommuting part) the binding, never creates it; Pati–Salam vector bosons are massive by eating Goldstones | fails classically; the scale $\ell=\hbar c/\Lambda_b$ by dimensional transmutation is an open conjecture |

## Running the checks

```
python3 investigations/rest_energy/verify/run_all.py
```

Building the note: `cd investigations/rest_energy && pdflatex main.tex` (twice).
