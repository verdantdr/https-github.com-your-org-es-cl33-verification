(* ::Package:: *)

(* =====================================================================
   04_dirac_signature.wl   --   Appendix F, "The Dirac pairing and the
   Lorentzian signature" (rows 12, 13).

   Establishes: {gamma^mu,gamma^nu} = 2 diag(+,-,-,-); Tr(gamma^0)=0;
   spec(gamma^0) = {(+1)^4,(-1)^4} -> the 4+4 particle/antiparticle pair.
   The Minkowski signature is read off, not imposed.
   Status: EXACT.
   ===================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "representation.wl"}]];

Module[{g, eta, Id8, z, alg},
  g = ESCl33`es$gamma; eta = ESCl33`es$eta4; Id8 = ESCl33`es$Id8; z = ESCl33`zero8;

  alg = Table[
    Simplify[ESCl33`acomm[g[[mu]], g[[nu]]] - 2 eta[[mu, nu]] Id8] === z,
    {mu, 4}, {nu, 4}];
  Print["{gamma^mu,gamma^nu} = 2 diag(+,-,-,-) (all 16) ?  ",
        AllTrue[Flatten[alg], TrueQ]];
  Print["   anticommutator boolean grid (mu,nu = 0..3):"];
  Do[Print["     ", alg[[mu]]], {mu, 4}];
  Print["(gamma^0)^2 = +Id8 ?  ", Simplify[g[[1]].g[[1]]] === Id8];
  Print["(gamma^k)^2 = -Id8 ?  ", Simplify[#.#] === -Id8 & /@ g[[2 ;; 4]]];
  Print["Tr(gamma^0) = ", Tr[g[[1]]]];
  Print["spec(gamma^0) = ", Sort[Eigenvalues[g[[1]]]], "   = {(+1)^4,(-1)^4}"];
  Print["=> 4 positive-energy + 4 negative-energy: one Dirac fermion + conjugate"];
];
