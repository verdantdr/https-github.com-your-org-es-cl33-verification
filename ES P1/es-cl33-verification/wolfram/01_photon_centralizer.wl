(* ::Package:: *)

(* =====================================================================
   01_photon_centralizer.wl   --   Appendix F, Section "The electromagnetic
   generator is forced uniquely" (rows 2,3,4,5,6).

   Establishes: the photon direction J is the UNIQUE element of su(2)_T
   commuting with the clock W; the centralizer is one-dimensional.
   Reproduces the literal  Solve  output  {b -> a, c -> a}.
   Status: EXACT.
   ===================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "representation.wl"}]];

Module[{e, W, L, J, Id8, z, a, b, c, X, clock, eqs, sol, dem},
  e = ESCl33`es$e; W = ESCl33`es$W; L = ESCl33`es$L; J = ESCl33`es$J;
  Id8 = ESCl33`es$Id8; z = ESCl33`zero8;

  (* Step 1: su(2)_T closes with structure constant -2 *)
  Print["L1^2=L2^2=L3^2=-Id8 ?  ",
    {Simplify[L[[1]].L[[1]]], Simplify[L[[2]].L[[2]]], Simplify[L[[3]].L[[3]]]} ===
      {-Id8, -Id8, -Id8}];
  Print["[L1,L2]=-2L3, cyclic ?  ",
    {Simplify[ESCl33`comm[L[[1]],L[[2]]] - (-2 L[[3]])] === z,
     Simplify[ESCl33`comm[L[[2]],L[[3]]] - (-2 L[[1]])] === z,
     Simplify[ESCl33`comm[L[[3]],L[[1]]] - (-2 L[[2]])] === z}];

  (* Step 2: no single generator commutes with the clock *)
  clock = e[[1]] + e[[2]] + e[[3]];
  Print["each [Li,clock] != 0 ?  ",
    {ESCl33`comm[L[[1]],clock] =!= z, ESCl33`comm[L[[2]],clock] =!= z,
     ESCl33`comm[L[[3]],clock] =!= z}];

  (* Step 3: centralizer condition -> the literal Solve output *)
  X = a L[[1]] + b L[[2]] + c L[[3]];
  eqs = Thread[Flatten[X.clock - clock.X] == 0];
  sol = Solve[eqs, {a, b, c}];
  Print["centralizer Solve output      :  ", sol /. {a -> a, b -> b, c -> c}];
  Print["  (i.e. a=b=c: one-dimensional, spanned by L1+L2+L3)"];

  (* Step 4: normalization *)
  dem = L[[1]] + L[[2]] + L[[3]];
  Print["(L1+L2+L3)^2 = -3 Id8 ?  ", dem.dem === -3 Id8];
  Print["J^2=-Id8 ?  ", Simplify[J.J] === -Id8,
        "   Tr[J^2]= ", Tr[J.J], "   [J,W]=0 ? ", Simplify[ESCl33`comm[J,W]] === z];
];
