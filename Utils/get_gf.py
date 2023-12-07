import numpy as np
from Algorithms import GEPP,GE,GEPivot

def get_gf(A,method='GE',end_gf=False):
    if method=='GE':
        return GE(A,end_gf=end_gf)[-1]
    elif method=='GEPP':
        return GEPP(A,end_gf=end_gf)[-1]
    elif method=='GEPivot':
        return GEPivot(A,end_gf=end_gf)[-1]
    else:
        raise ValueError('Method not found')