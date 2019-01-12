import numpy as np
from copy import deepcopy
from omperr import omperr
from KSVD import KSVD

def KSVDDL(Y,param):
    D = deepcopy(param.initialDictionary)
    iterations = param.itN
    errglobal = param.errorGoal

    for j in range(iterations):
        X = omperr(D,Y,errglobal)
        D,X = KSVD(Y,D,X)

    return D