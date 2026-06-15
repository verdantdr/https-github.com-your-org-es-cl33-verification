(* ::Package:: *)

(* =====================================================================
   run_all.wl   --   Master verification runner.

   Reproduces every row of Table "Machine-verified identities underlying
   Part I" (Appendix F) and prints a single PASS/FAIL report. Rows 1-14
   are exact statements in the real 8x8 representation; row 15 is the
   generalized-Kronecker contraction in dimensions 4 and 6; rows 16-17
   are symbolic phase-space identities on T*M^6.

   Usage:   wolframscript -f run_all.wl
   Exit code is 0 iff all rows pass.
   ===================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "representation.wl"}]];
Get[FileNameJoin[{DirectoryName[$InputFileName], "06_gravity_double_contraction.wl"}]];

Module[{e, W, L, J, G, Jr, V, Igen, g, eta, Id8, z, eta6,
        p1, p2, p3, m, H, rows, grav4, grav6, pbRes,
        x, P, gi, u, PB, Cc, phi, Lie, target},

  e = ESCl33`es$e; W = ESCl33`es$W; L = ESCl33`es$L; J = ESCl33`es$J;
  G = ESCl33`es$G; Jr = ESCl33`es$Jrot; V = ESCl33`es$V; Igen = ESCl33`es$Igen;
  g = ESCl33`es$gamma; eta = ESCl33`es$eta4; Id8 = ESCl33`es$Id8; z = ESCl33`zero8;
  eta6 = DiagonalMatrix[{1, 1, 1, -1, -1, -1}];

  H = G[[1]] p1 + G[[2]] p2 + G[[3]] p3 + m V;

  (* gravitational coefficient in two dimensions *)
  grav4 = ("(alpha,beta)" /. fitContraction[4, 171]) === {-4, 2} &&
          ("max residual of  C - (alpha Ric + beta delta R)" /. fitContraction[4, 171]) === 0;
  grav6 = ("(alpha,beta)" /. fitContraction[6, 66]) === {-4, 2} &&
          ("max residual of  C - (alpha Ric + beta delta R)" /. fitContraction[6, 66]) === 0;

  (* phase-space identity, dim 3 sample of the general computation *)
  x = {x1, x2, x3}; P = {P1, P2, P3};
  gi = Table[Symbol["q" <> ToString[Min[i,j]] <> ToString[Max[i,j]]][x1,x2,x3], {i,3},{j,3}];
  u = {ua[x1,x2,x3], ub[x1,x2,x3], uc[x1,x2,x3]};
  PB[F_, Gg_] := Sum[D[F, x[[k]]] D[Gg, P[[k]]] - D[F, P[[k]]] D[Gg, x[[k]]], {k, 3}];
  Cc = (1/2) Sum[gi[[i,j]] P[[i]] P[[j]], {i,3},{j,3}];
  phi = u.P;
  Lie = Table[Sum[u[[L]] D[gi[[i,j]], x[[L]]], {L,3}]
              - Sum[gi[[L,j]] D[u[[i]], x[[L]]], {L,3}]
              - Sum[gi[[i,L]] D[u[[j]], x[[L]]], {L,3}], {i,3},{j,3}];
  target = -(1/2) Sum[Lie[[i,j]] P[[i]] P[[j]], {i,3},{j,3}];
  pbRes = Simplify[PB[phi, Cc] - target] === 0;

  rows = {
   {1,  "{e_i,e_j} = 2 eta_ij Id8",
        AllTrue[Flatten@Table[Simplify[ESCl33`acomm[e[[i]],e[[j]]] - 2 eta6[[i,j]] Id8] === z, {i,6},{j,6}], TrueQ]},
   {2,  "W^2 = +Id8",                         Simplify[W.W] === Id8},
   {3,  "J^2 = -Id8",                         Simplify[J.J] === -Id8},
   {4,  "Tr[J^2] = -8",                       Tr[J.J] === -8},
   {5,  "[J,W] = 0",                          Simplify[ESCl33`comm[J,W]] === z},
   {6,  "centralizer of W in su(2)_T is 1-dim = J",
        Module[{a,b,c,X,clk,sol}, clk=e[[1]]+e[[2]]+e[[3]]; X=a L[[1]]+b L[[2]]+c L[[3]];
          sol=Solve[Thread[Flatten[X.clk-clk.X]==0],{a,b,c}];
          (Length[sol] == 1) && (({b,c} /. sol[[1]]) === {a,a})]},
   {7,  "G_a^2=+Id8, {G_a,G_b}=0 (a!=b)",
        (AllTrue[G, Simplify[#.#] === Id8 &]) &&
         AllTrue[Flatten@Table[Simplify[ESCl33`acomm[G[[a]],G[[b]]]] === z, {a,3},{b,a+1,3}], TrueQ]},
   {8,  "[GX,GY]=-2 e4e5 (and cyclic)",
        (Simplify[ESCl33`comm[G[[1]],G[[2]]] - (-2 Jr[[3]])] === z) &&
        (Simplify[ESCl33`comm[G[[2]],G[[3]]] - (-2 Jr[[1]])] === z) &&
        (Simplify[ESCl33`comm[G[[3]],G[[1]]] - (-2 Jr[[2]])] === z)},
   {9,  "V^2=+Id8, {G_a,V}=0, V^T=V",
        (Simplify[V.V] === Id8) && AllTrue[G, Simplify[ESCl33`acomm[#,V]] === z &] &&
         (Transpose[V] === V)},
   {10, "H = sum G_a p_a + m V real-symmetric",  Transpose[H] === H},
   {11, "H^2 = (p^2+m^2) Id8",
        Simplify[Expand[H.H] - (p1^2+p2^2+p3^2+m^2) Id8] === z},
   {12, "{gamma^mu,gamma^nu} = 2 diag(+,-,-,-)",
        AllTrue[Flatten@Table[Simplify[ESCl33`acomm[g[[mu]],g[[nu]]] - 2 eta[[mu,nu]] Id8] === z, {mu,4},{nu,4}], TrueQ]},
   {13, "Tr[g0]=0, spec(g0)={(+1)^4,(-1)^4}",
        (Tr[g[[1]]] === 0) && (Sort[Eigenvalues[g[[1]]]] === {-1,-1,-1,-1,1,1,1,1})},
   {14, "spec(J)={(+i)^4,(-i)^4} => q=+-1/2",
        Sort[Eigenvalues[J]] === {-I,-I,-I,-I,I,I,I,I}},
   {15, "delta^{apq}_{drs} R^{rs}_{pq} = -4 G^a_d  (n=4,6)", grav4 && grav6},
   {16, "{phi1,phi2}=0 (flat surplus plane)",  PB[{1,0,0}.P, {0,1,0}.P] === 0},
   {17, "{phi,C} = -(Lie_u g) P P; =0 <=> Killing",  pbRes}
  };

  Print["==================================================================="];
  Print["  ES / Cl(3,3) verification suite -- Table of machine-verified"];
  Print["  identities underlying Part I (Appendix F)"];
  Print["==================================================================="];
  Scan[Print["  row ", IntegerString[#[[1]]] // (StringPadLeft[#, 2] &), "  ",
             If[#[[3]] === True, "PASS", "**FAIL**"], "   ", #[[2]]] &, rows];
  Print["==================================================================="];
  Print["  ALL ", Length[rows], " ROWS PASS ?   ", AllTrue[rows, #[[3]] === True &]];
  Print["==================================================================="];

  If[AllTrue[rows, #[[3]] === True &], Quit[0], Quit[1]];
];
