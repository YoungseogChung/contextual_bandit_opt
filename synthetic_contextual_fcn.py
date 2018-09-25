import numpy as np


def hartmann6(X, alpha, A, P):
    """
    X: 6x1
    alpha: 4x1
    A: 4x6
    P: 4x6
    """

    out = (-1)*np.sum(alpha*np.exp((-1)*(A*((X-P)**2))))

    return out

def main(X, alpha, A, P):
    if alpha is None:
      np.array([[1.0,1.2,3.0,3.2]]).T

    if A is None:
      A = np.array([ \
        [10,3,17,3.50,1.7,8],\
        [0.05, 10, 17, 0.1, 8, 14], \
        [3,3.5,1.7,10,17,8],\
        [17,8,0.05,10,0.1,14] \
                 ])
    if P is None:
       P = 10e-4*np.array([\
        [1312,1696,5569,124,8283,5886],\
        [2329,4135,8307,3736,1004,9991], \
        [2348,1451,3522,2883,3047,6650],\
        [4047,8828,8732,5743,1091,381]\
                         ])



    return hartmann6(X, alpha, A, P)
