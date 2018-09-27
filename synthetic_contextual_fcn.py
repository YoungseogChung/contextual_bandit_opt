import numpy as np


def hartmann6(X, alpha, A, P):
    """
    X: 6x1
    alpha: 4x1
    A: 4x6
    P: 4x6
    """
    out =(-1)*np.sum(alpha*np.exp(np.sum((-1)*A*(X-P)**2, axis=1)),axis=0)

    return out

def main(X, alpha, A, P):
    """
    minimum achieved for default alpha, A, P when
    X = np.array([0.20169, 0.150011, 0.476874,0.275332, 0.311652, 0.6573])
    """
    
    if alpha is None:
        alpha = np.array([1.0,1.2,3.0,3.2])

    if A is None:
        A = np.array([[10,   3,   17,  3.50, 1.7, 8],\
                    [  0.05, 10,  17,  0.1,  8,   14], \
                    [  3,    3.5, 1.7, 10,   17,  8],\
                    [  17,   8,   0.05,10,   0.1, 14]])

    if P is None:
        P = 10e-5*np.array([[1312, 1696, 5569, 124,  8283, 5886],\
                           [ 2329, 4135, 8307, 3736, 1004, 9991], \
                           [ 2348, 1451, 3522, 2883, 3047, 6650],\
                           [ 4047, 8828, 8732, 5743, 1091, 381]])


    print(hartmann6(X, alpha, A, P))
    return hartmann6(X, alpha, A, P)
