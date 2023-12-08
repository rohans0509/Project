import numpy as np
from .GEPP import GEPP
from .GE import GE

def GEM(A,compute_P=False,end_gf=False):
   P,L,U,gf=GEPP(A,compute_P=True,end_gf=end_gf)
   L,U,gf=GE(P@A,end_gf=end_gf)
   
   if compute_P:
        if end_gf:
            return P,L, U, gf
        return P,L, U, gf
   else:
        if end_gf:
            return L, U, gf
        return L, U, gf
