import numpy as np
from Algorithms import GEM, GEPP,GE

def get_gf(A,method='GE',end_gf=False):
    if method=='GE':
        return GE(A,end_gf=end_gf)[-1]
    elif method=='GEPP':
        return GEPP(A,end_gf=end_gf)[-1]
    elif method=='GEM':
        return GEM(A,end_gf=end_gf)[-1]
    else:
        raise ValueError('Method not found')