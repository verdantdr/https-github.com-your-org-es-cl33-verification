# The undivided energy and its differentiation — $E=mc^2$ in the arena

Investigation note to *The electromagnetic sphere* (17 October 2026, `tex/`), testing the hypothesis

> *$E=mc^2$: $c^2$ is the symmetrical part of the equation and $m$ is what breaks this symmetry; $m$ is generated from a $b$ particle,
> $b$ standing for the $b$-particle of Yang–Mills* — with the energy one undivided, six-dimensional entity whose internal differentiation
> produces the mass, before any external interaction.

- `main.pdf` — the note (built from `main.tex`, `preamble.tex` and `sections/`).
- `sections/` — `s0_intro` (the question, the dictionary, how claims are classified); `d1_unitary_energy` (the undivided energy and its
  differentiation); `d2_symmetry_breaking` (what a mass breaks; particles, antiparticles, vacua, Goldstone bosons); `d3_closure` (the
  differentiation in motion: rotor, rate and duality angle); `d4_bfield` (the $b$-field as the antisymmetric part of the shift);
  `s5_synthesis` (the hypothesis clause by clause); then the second question — `d5_duplication` (from $\mathrm{Cl}(3,3)$ to
  $\mathrm{Cl}(6,6)$: what is forced, what is chosen, what the duplicate adds), `d6_pair` (the duplicate in motion: balance surface, pairs,
  gauge status of a copy's mass), `d7_between` (the $b$-field between $n$ manifolds), `s6_duplication_synthesis`; `s9_methods`.
- `verify/` — `d1`–`d7`: derivation checks; `x1`–`x7`: independent adversarial re-checks; `run_all.py` runs all fourteen and writes
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

## Second question: duplication, and the $b$-field between manifolds

| statement | what the arena gives | verdict |
|---|---|---|
| the algebra is duplicated | $\mathrm{Cl}(6,6)=\mathrm{Cl}(3,3)\hat\otimes\mathrm{Cl}(3,3)$, Klein factor = Jordan–Wigner sign, spinors = Fock space of six modes | forced (inherited) |
| the energy is duplicated | isometric embeddings of $(\mathbb C^3,E_6)$: a beam-splitter $S^3$, cut to one circle (the copy rotation) by $\eta$; literal copying excluded | chosen; cloning obstructed |
| the duplicate adds structure | $\eta$-isometric complex structure, copy charge, $\mathfrak{so}(2,\mathbb C)$, non-rigid $S^4$ vacuum, balance surface (constant rate with massive constituents) | fixed (partly restates the paper) |
| a massive pair from one null field | impossible for any Lorentz-covariant duplication; needs a second null field ($\gamma\gamma\to e^+e^-$) or a non-covariant third body | obstructed; kinematics inherited |
| the mass of one copy | gauge dependent unless the copies are distinguished | fixed; the distinguishing structure is open |
| a $b$-field between manifolds | $\mathfrak{so}(n,\mathbb C)$ under a common Lorentz action, compact part $\mathfrak{so}(n)$; a spectator Lorentz manifold is decoupled | fixed |
| it is a deal breaker | at unequal rates: not removable, level repulsion, seesaw confinement of a rate-less frame, resonances; from three manifolds a time-odd invariant $B_{12}B_{23}B_{31}$ (Jarlskog analogue) | holds (fixed) |
| it generates a rate (a mass) | rates are the spectrum of the self-adjoint part, a gauge tensor: never changed; equal rates ⇒ removable | fails (obstructed) |
| where unequal rates come from | not the paper's one-rotor closure; a rotor acting by a non-central element of $\mathfrak u(n)$ | constructed; origin open |
| isospin ½ between manifolds | copy isospin is integral; the Yang–Mills doublet is one connection plus two rate terms | fails under a common Lorentz action |
| the arrow of the hypothesis | the non-removable part $[K,B]_{kl}=(\kappa_k-\kappa_l)B_{kl}$: rate differences make $b$ physical (Higgs–Kibble-like) | reversed (interpretive) |

## Running the checks

```
python3 investigations/rest_energy/verify/run_all.py
```

Building the note: `cd investigations/rest_energy && pdflatex main.tex` (twice).
