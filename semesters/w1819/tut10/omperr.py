import numpy as np
from copy import deepcopy
from numpy.linalg import pinv

def omperr(D,X,errorGoal):
    n,P = X.shape
    n,K = D.shape
    E2 = (errorGoal**2)*n
    maxNumCoef = n/2
    A = np.zeros((D.shape[1],X.shape[1]))
    for k in range(P):
        a = 0
        x = deepcopy(X[:,k])
        residual = deepcopy(x)
        indx = np.array([],dtype=int)
        currResNorm2 = np.sum(residual**2)
        j = 0
        while currResNorm2 > E2 and j < maxNumCoef:
            j = j+1
            proj = np.dot(D.T,residual)
            pos = np.argmax(np.abs(proj))
            indx = np.append(indx,int(pos))
            a = np.dot(pinv(D[:,indx[0:j]]),x)
            residual = x-np.dot(D[:,indx[0:j]],a)
            currResNorm2 = np.sum(residual**2)
        if (len(indx) > 0):
           A[indx,k] = deepcopy(a)
    return A
