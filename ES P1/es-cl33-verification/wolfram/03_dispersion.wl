(* ::Package:: *)

(* =====================================================================
   03_dispersion.wl   --   Appendix F, "The relativistic dispersion
   relation" (rows 7, 9, 10, 11).

   Establishes: H = sum_a G_a p_a + m V is real-symmetric and
   H^2 = (p1^2+p2^2+p3^2+m^2) Id8, so spec(H) = +- sqrt(p^2+m^2).
   Every would-be cross term cancels by anticommutation.
   Status: EXACT.
   ===================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "representation.wl"}]];

Module[{G, V, Id8, z, p1, p2, p3, m, H, Hsq, crossGG, crossGV},
  G = ESCl33`es$G; V = ESCl33`es$V; Id8 = ESCl33`es$Id8; z = ESCl33`zero8;

  (* Step 1: building blocks *)
  Print["G_a^2 = +Id8 ?  ", Simplify[#.#] === Id8 & /@ G];
  crossGG = Flatten@Table[Simplify[ESCl33`acomm[G[[a]], G[[b]]]] === z, {a, 3}, {b, a + 1, 3}];
  Print["{G_a,G_b}=0 (a<b) ?  ", crossGG];
  Print["V^2 = +Id8 ?  ", Simplify[V.V] === Id8];
  crossGV = Simplify[ESCl33`acomm[#, V]] === z & /@ G;
  Print["{G_a,V}=0 ?  ", crossGV];
  Print["G_a, V symmetric ?  ", {AllTrue[G, Transpose[#] === # &], Transpose[V] === V}];

  (* Step 2-3: the Hamiltonian and its square *)
  H = G[[1]] p1 + G[[2]] p2 + G[[3]] p3 + m V;
  Print["H symmetric ?  ", Transpose[H] === H];
  Hsq = Expand[H.H];
  Print["H^2 = (p1^2+p2^2+p3^2+m^2) Id8 ?  ",
    Simplify[Hsq - (p1^2 + p2^2 + p3^2 + m^2) Id8] === z];
  Print["=> spec(H) = +- Sqrt[p^2 + m^2]  (exact relativistic mass shell)"];
];
