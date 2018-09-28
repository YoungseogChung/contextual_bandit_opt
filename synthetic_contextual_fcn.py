import numpy as np

# The values of the hartmann6 function as given by
# https://www.sfu.ca/~ssurjano/hart6.html
STANDARD_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
STANDARD_A = np.array([[10, 3, 17, 3.50, 1.7, 8],\
                       [0.05, 10, 17, 0.1, 8, 14],\
                       [3, 3.5, 1.7, 10, 17, 8],\
                       [17, 8, 0.05, 10, 0.1, 14]])
STANDARD_P = 10e-5 * np.array([[1312, 1696, 5569, 124, 8283, 5886],\
                               [2329, 4135, 8307, 3736, 1004, 9991],\
                               [2348, 1451, 3522, 2883, 3047, 6650],\
                               [4047, 8828, 8732, 5743, 1091, 381]])

def hartmann6(X, alpha, A, P):
    """
    X: 6x1
    alpha: 4x1
    A: 4x6
    P: 4x6
    """
    return (-1)*np.sum(alpha*np.exp(np.sum((-1)*A*(X-P)**2, axis=1)),axis=0)

def main(X, alpha=None, A=None, P=None):
    """
    minimum achieved for default alpha, A, P when
    X = np.array([0.20169, 0.150011, 0.476874,0.275332, 0.311652, 0.6573])
    """
    alpha = STANDARD_ALPHA if alpha is None else alpha
    A = STANDARD_A if A is None else A
    P = STANDARD_P if P is None else P
    return hartmann6(X, alpha, A, P)

if __name__ == '__main__':
    print main(np.random.rand(1, 6))
