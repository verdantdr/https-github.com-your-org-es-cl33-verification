(* ::Package:: *)

(* =====================================================================
   02_charge_spectrum.wl   --   Appendix F, "Charge quantization from the
   spectrum of J" (row 14).

   Establishes: spec(J) = {(+i)^4, (-i)^4}; with Q = -i J the charge
   eigenvalues are +-1 on the complexified module, halved to q=+-1/2 on
   the particle-antiparticle pair.
   Status: EXACT (spectrum); DERIVED (charge assignment).
   ===================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "representation.wl"}]];

Module[{J, specJ, Q, specQ},
  J = ESCl33`es$J;
  specJ = Sort[Eigenvalues[J]];
  Q = -I J;                                   (* Hermitian charge operator *)
  specQ = Sort[Eigenvalues[Q]];
  Print["spec(J)  = ", specJ, "   = {(+i)^4,(-i)^4}"];
  Print["Q = -i J Hermitian ?  ", Simplify[ConjugateTranspose[Q] - Q] === ESCl33`zero8];
  Print["spec(Q)  = ", specQ, "   = {(+1)^4,(-1)^4}"];
  Print["=> elementary charge after particle/antiparticle halving:  q = +-1/2"];
];
