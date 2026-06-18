(* ::Package:: *)

(* =====================================================================
   05_spin_curvature.wl
   ---------------------------------------------------------------------
   THE NOT-SHOWN DERIVATION OF APPENDIX F.

   Row 8 of Table "Machine-verified identities underlying Part I" records

        [G_X,G_Y] = -2 e4 e5 = -2 J3   (and cyclic)        (Prop. spin-curvature)

   but, unlike the other substantive identities, it is the one row that is
   *not* worked out in Section "Worked derivations". This script supplies
   that missing derivation in the same step-by-step style as the six
   derivations that ARE shown, and closes with the explicit 8x8 check.

   Result it establishes (Proposition "Spin from the boost curvature"):
   the curvature of the boost connection is rotational -- the commutator
   of two Dirac boosts is (-2x) a spatial su(2)_S rotation, so the
   quadratic part A^A of a boost field strength is rotation-valued and
   acts on the spinor as spin, with holonomy in Spin(3)=SU(2).

   Status: EXACT (the commutator identity); DERIVED (the spin reading).
   ===================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "representation.wl"}]];

Module[{e, W, GX, GY, GZ, J1, J2, J3, Id8, z,
        antiW, GXsq, J3sq, cXY, cYZ, cZX, an, analytic},

  e   = ESCl33`es$e;
  W   = ESCl33`es$W;
  Id8 = ESCl33`es$Id8;
  z   = ESCl33`zero8;

  (* ---- Step 1: the building blocks ---------------------------------- *)
  (* boosts  G_a = W e_{3+a}                                              *)
  GX = W.e[[4]];  GY = W.e[[5]];  GZ = W.e[[6]];
  (* spatial rotations  J1=e5e6, J2=e6e4, J3=e4e5  (su(2)_S)             *)
  J1 = e[[5]].e[[6]];  J2 = e[[6]].e[[4]];  J3 = e[[4]].e[[5]];

  Print["================================================================"];
  Print["Step 1.  W^2 = +Id8 ?                       ", Simplify[W.W] === Id8];
  Print["         J3^2 = -Id8  (rotation) ?          ", Simplify[J3.J3] === -Id8];
  Print["         GX^2 = +Id8  (boost) ?             ", Simplify[GX.GX] === Id8];

  (* ---- Step 2: the clock anticommutes with every spatial generator -- *)
  (* each of e1,e2,e3 anticommutes with each of e4,e5,e6 (distinct       *)
  (* Clifford generators), hence {W, e_{3+a}} = 0.                        *)
  antiW = {Simplify[ESCl33`acomm[W, e[[4]]]] === z,
           Simplify[ESCl33`acomm[W, e[[5]]]] === z,
           Simplify[ESCl33`acomm[W, e[[6]]]] === z};
  Print["----------------------------------------------------------------"];
  Print["Step 2.  {W,e4}={W,e5}={W,e6}=0 ?           ", antiW];

  (* ---- Step 3: the commutator, collapsed by anticommutation ---------- *)
  (* Analytic core (no matrices needed):                                  *)
  (*   W e4 W = W(e4 W) = W(-W e4) = -W^2 e4 = -e4    [uses {W,e4}=0, W^2=1]*)
  (*   => GX GY = (W e4 W) e5 = -e4 e5                                     *)
  (*      GY GX = (W e5 W) e4 = -e5 e4 = +e4 e5                            *)
  (*   => [GX,GY] = -e4 e5 - e4 e5 = -2 e4 e5 = -2 J3.                     *)
  analytic = Simplify[(W.e[[4]].W) - (-e[[4]])];          (* must be z *)
  Print["----------------------------------------------------------------"];
  Print["Step 3.  identity  W e4 W = -e4  (the collapse) ?   ",
        analytic === z];

  (* ---- Step 4: explicit 8x8 verification, all three pairs ----------- *)
  cXY = ESCl33`comm[GX, GY];
  cYZ = ESCl33`comm[GY, GZ];
  cZX = ESCl33`comm[GZ, GX];
  Print["----------------------------------------------------------------"];
  Print["Step 4.  [GX,GY] = -2 J3 = -2 e4e5 ?        ",
        Simplify[cXY - (-2 J3)] === z];
  Print["         [GY,GZ] = -2 J1 ?                  ",
        Simplify[cYZ - (-2 J1)] === z];
  Print["         [GZ,GX] = -2 J2 ?                  ",
        Simplify[cZX - (-2 J2)] === z];

  Print["================================================================"];
  Print["CONCLUSION (row 8): the boost connection has rotational"];
  Print["curvature -- [boost,boost] = -2 (rotation).  The quadratic part"];
  Print["A^A of a boost field strength is su(2)_S-valued and acts on the"];
  Print["spinor as spin; holonomy lies in Spin(3)=SU(2), quantizing spin"];
  Print["in half-integers.   [EXACT identity; DERIVED spin reading]"];
];
