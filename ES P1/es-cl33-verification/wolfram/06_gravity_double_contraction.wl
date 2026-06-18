(* ::Package:: *)

(* =====================================================================
   06_gravity_double_contraction.wl   --   Appendix F, "The gravitational
   double-contraction" (row 15).

   Establishes the split-epsilon (Lovelock) identity

       delta^{apq}_{drs} R^{rs}_{pq} = -4 R^a_d + 2 delta^a_d R
                                     = -4 ( R^a_d - 1/2 delta^a_d R )
                                     = -4 G^a_d

   on a GENERIC Riemann tensor carrying the algebraic symmetries, in
   BOTH dimension 4 and dimension 6 -- so the coefficient -4 is a
   combinatorial property of the contraction, not an artefact of a
   particular dimension.

   The generalized rank-3 Kronecker delta is the determinant of a 3x3
   array of ordinary deltas. With a Euclidean reference metric the index
   raising is trivial, so the linear fit C = alpha Ric + beta delta R
   pins (alpha,beta) unambiguously; it returns (-4, 2) and the residual
   over all components is exactly 0.

   Status: EXACT (identity); DERIVED (gravitational reading).
   ===================================================================== *)

ClearAll[fitContraction];
fitContraction[n_Integer, seed_Integer] := Module[
  {A, B, C1, R, d, GKD, Cad, Ric, Rscal, a1, b1, sol, resid, sym},
  SeedRandom[seed];
  d = KroneckerDelta;
  A = Table[RandomInteger[{-5, 5}], {n}, {n}, {n}, {n}];
  (* project A onto the space of algebraic curvature tensors *)
  B  = Table[(A[[a,b,c,e]] - A[[b,a,c,e]] - A[[a,b,e,c]] + A[[b,a,e,c]])/4,
             {a,n},{b,n},{c,n},{e,n}];
  C1 = Table[(B[[a,b,c,e]] + B[[c,e,a,b]])/2, {a,n},{b,n},{c,n},{e,n}];
  R  = Table[C1[[a,b,c,e]] - (C1[[a,b,c,e]] + C1[[a,c,e,b]] + C1[[a,e,b,c]])/3,
             {a,n},{b,n},{c,n},{e,n}];
  (* verify all four Riemann symmetries on the projected tensor *)
  sym = Max@Abs@Flatten@{
    Table[R[[a,b,c,e]] + R[[b,a,c,e]],            {a,n},{b,n},{c,n},{e,n}],
    Table[R[[a,b,c,e]] + R[[a,b,e,c]],            {a,n},{b,n},{c,n},{e,n}],
    Table[R[[a,b,c,e]] - R[[c,e,a,b]],            {a,n},{b,n},{c,n},{e,n}],
    Table[R[[a,b,c,e]] + R[[a,c,e,b]] + R[[a,e,b,c]], {a,n},{b,n},{c,n},{e,n}]};
  (* generalized rank-3 Kronecker delta and the double-dual contraction *)
  GKD[a_,p_,q_,dd_,r_,s_] := Det[{{d[a,dd], d[a,r], d[a,s]},
                                  {d[p,dd], d[p,r], d[p,s]},
                                  {d[q,dd], d[q,r], d[q,s]}}];
  Cad = Table[Sum[GKD[a,p,q,dd,r,s] R[[r,s,p,q]], {p,n},{q,n},{r,n},{s,n}],
              {a,n},{dd,n}];
  (* Ricci and scalar in one fixed convention (contract 2nd & 4th) *)
  Ric = Table[Sum[R[[a,mm,dd,mm]], {mm,n}], {a,n},{dd,n}];
  Rscal = Sum[Ric[[a,a]], {a,n}];
  sol = Solve[{Cad[[1,1]] == a1 Ric[[1,1]] + b1 Rscal,
               Cad[[1,2]] == a1 Ric[[1,2]]}, {a1, b1}][[1]];
  resid = Max@Abs@Flatten@Table[
    Cad[[a,dd]] - ((a1 Ric[[a,dd]] + b1 d[a,dd] Rscal) /. sol), {a,n},{dd,n}];
  <|"dimension" -> n, "Riemann-symmetry residual" -> sym,
    "(alpha,beta)" -> ({a1,b1} /. sol),
    "max residual of  C - (alpha Ric + beta delta R)" -> resid|>
];

Print["n = 4 :  ", fitContraction[4, 171]];
Print["n = 6 :  ", fitContraction[6, 66]];
Print["=> (alpha,beta) = (-4,2) in both dimensions: ",
      "C = -4 (R^a_d - 1/2 delta^a_d R) = -4 G^a_d."];
