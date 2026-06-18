(* ::Package:: *)

(* =====================================================================
   representation.wl
   ---------------------------------------------------------------------
   The real 8x8 Jordan-Wigner representation of Cl(3,3) used throughout
   Part I of the Electromagnetic Sphere monograph (Appendix F,
   "The representation").

   Loaded by every verification script in this repository via
       Get[FileNameJoin[{DirectoryName[$InputFileName],"representation.wl"}]]

   Signature  eta = diag(+,+,+,-,-,-).
   ===================================================================== *)

BeginPackage["ESCl33`"];

es$e::usage      = "es$e[[i]] is the i-th Clifford generator e_i (8x8 real), i=1..6.";
es$Id8::usage    = "8x8 identity matrix.";
es$W::usage      = "Clock / frame vector  W = (e1+e2+e3)/Sqrt[3].";
es$L::usage      = "es$L[[i]] are the temporal bivectors L1=e2e3, L2=e3e1, L3=e1e2.";
es$J::usage      = "Photon generator J = (L1+L2+L3)/Sqrt[3] (democratic su(2)_T diagonal).";
es$G::usage      = "es$G[[a]] are the Dirac boosts G_a = W.e_{3+a}, a=1..3 (so GX,GY,GZ).";
es$Jrot::usage   = "es$Jrot[[i]] are the spatial rotations J1=e5e6, J2=e6e4, J3=e4e5.";
es$V::usage      = "Covariant mass trivector V = e4 e5 e6.";
es$Igen::usage   = "Pseudoscalar I = e1 e2 e3 e4 e5 e6.";
es$gamma::usage  = "es$gamma[[mu]] are the Dirac matrices gamma^0=W, gamma^k=e_{3+k}.";
es$eta4::usage   = "Mostly-minus Minkowski metric diag(+1,-1,-1,-1).";

comm::usage  = "comm[a,b] = a.b - b.a (matrix commutator).";
acomm::usage = "acomm[a,b] = a.b + b.a (matrix anticommutator).";
zero8::usage = "8x8 zero matrix, for === comparisons.";

Begin["`Private`"];

id = IdentityMatrix[2];
sx = {{0, 1}, {1, 0}};
sz = {{1, 0}, {0, -1}};
ep = {{0, -1}, {1, 0}};                 (* the real "i": epsilon, eps^2 = -1 *)
KP[a_, b_, c_] := KroneckerProduct[a, KroneckerProduct[b, c]];

es$e = {
  KP[sx, id, id],   (* e1 *)
  KP[sz, sx, id],   (* e2 *)
  KP[sz, sz, sx],   (* e3 *)
  KP[ep, id, id],   (* e4 *)
  KP[sz, ep, id],   (* e5 *)
  KP[sz, sz, ep]    (* e6 *)
};

es$Id8 = IdentityMatrix[8];
zero8  = ConstantArray[0, {8, 8}];

comm[a_, b_]  := a.b - b.a;
acomm[a_, b_] := a.b + b.a;

With[{e = es$e},
  es$W     = (e[[1]] + e[[2]] + e[[3]])/Sqrt[3];
  es$L     = {e[[2]].e[[3]], e[[3]].e[[1]], e[[1]].e[[2]]};
  es$J     = (es$L[[1]] + es$L[[2]] + es$L[[3]])/Sqrt[3];
  es$G     = {es$W.e[[4]], es$W.e[[5]], es$W.e[[6]]};
  es$Jrot  = {e[[5]].e[[6]], e[[6]].e[[4]], e[[4]].e[[5]]};
  es$V     = e[[4]].e[[5]].e[[6]];
  es$Igen  = e[[1]].e[[2]].e[[3]].e[[4]].e[[5]].e[[6]];
  es$gamma = {es$W, e[[4]], e[[5]], e[[6]]};
];

es$eta4 = DiagonalMatrix[{1, -1, -1, -1}];

End[];
EndPackage[];

(* Sanity self-check on load: the defining Clifford relations. *)
Module[{e = ESCl33`es$e, eta, ok},
  eta = DiagonalMatrix[{1, 1, 1, -1, -1, -1}];
  ok = AllTrue[
    Flatten@Table[
      Simplify[(e[[i]].e[[j]] + e[[j]].e[[i]]) - 2 eta[[i, j]] IdentityMatrix[8]],
      {i, 6}, {j, 6}], # === 0 &];
  If[ok,
    Print["[representation.wl] loaded: {e_i,e_j} = 2 diag(+,+,+,-,-,-) verified."],
    Print["[representation.wl] WARNING: Clifford relations FAILED to verify."]]
];
