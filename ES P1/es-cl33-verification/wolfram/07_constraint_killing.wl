(* ::Package:: *)

(* =====================================================================
   07_constraint_killing.wl   --   Appendix F, "First-class closure is
   equivalent to the Killing condition" (rows 16, 17).

   Establishes, symbolically for a GENERAL inverse metric g^{NK}(x) and a
   GENERAL vector field u^M(x), the bracket of the flow generator
   phi = u^M P_M with the mass-shell constraint C = 1/2 g^{NK} P_N P_K:

       {phi, C} = -1/2 (Lie_u g^{NK}) P_N P_K,

   with the Lie derivative of the inverse metric

       (Lie_u g)^{NK} = u^L d_L g^{NK} - g^{LK} d_L u^N - g^{NL} d_L u^K.

   Hence {phi,C}=0  <=>  Lie_u g = 0  (u Killing): first-class closure of
   the surplus constraints is EQUIVALENT to the Killing condition. Two
   constant surplus directions Poisson-commute (flat surplus plane, row 16).

   Status: EXACT (identity); DERIVED (application).
   ===================================================================== *)

Module[{dim, x, P, gi, u, PB, C, phi, Lie, target, resid, phi1, phi2},
  dim = 3;
  x = Array[Symbol["x" <> ToString[#]] &, dim];
  P = Array[Symbol["P" <> ToString[#]] &, dim];
  (* general symmetric inverse metric and general vector field, x-dependent *)
  gi = Table[Symbol["g" <> ToString[Min[i, j]] <> ToString[Max[i, j]]][Sequence @@ x],
             {i, dim}, {j, dim}];
  u  = Table[Symbol["u" <> ToString[i]][Sequence @@ x], {i, dim}];
  (* canonical bracket {x^M,P_N}=delta:  {F,G}=sum dF/dx dG/dP - dF/dP dG/dx *)
  PB[F_, G_] := Sum[D[F, x[[k]]] D[G, P[[k]]] - D[F, P[[k]]] D[G, x[[k]]], {k, dim}];

  C   = (1/2) Sum[gi[[i, j]] P[[i]] P[[j]], {i, dim}, {j, dim}];
  phi = Sum[u[[i]] P[[i]], {i, dim}];

  Lie = Table[
     Sum[u[[L]] D[gi[[i, j]], x[[L]]], {L, dim}]
   - Sum[gi[[L, j]] D[u[[i]], x[[L]]], {L, dim}]
   - Sum[gi[[i, L]] D[u[[j]], x[[L]]], {L, dim}], {i, dim}, {j, dim}];
  target = -(1/2) Sum[Lie[[i, j]] P[[i]] P[[j]], {i, dim}, {j, dim}];

  resid = Simplify[PB[phi, C] - target];
  Print["{phi,C} = -1/2 (Lie_u g) P P  identically ?   residual = ", resid];

  (* row 16: two constant surplus directions commute *)
  phi1 = {1, 0, 0}.P; phi2 = {0, 1, 0}.P;
  Print["{phi1,phi2} on flat surplus plane = ", PB[phi1, phi2]];
  Print["=> {phi,C}=0  <=>  Lie_u g = 0  (u Killing): first-class closure"];
  Print["   is equivalent to the Killing condition."];
];
