import tensorly as tl
import numpy as np
from functools import reduce

def CP_ALS(X, R):
    """
    the function implements Algorithm CP-ALS given on Figure 3.3
    in http://www.kolda.net/publication/koba09/
    the notation is taken from there
    """

    # TODO: initialize normally
    As = [ np.ones((In, R), dtype=float) for In in X.shape ]
    lambdas = np.ones(R, dtype=float)


    Xs = [ tl.unfold(X, mode=i) for i in range(len(X.shape)) ]

    it = 0
    # TODO: write good stop criteria, e.g 
    #       frobenius norm distance or distance between subspaces of As
    while it < 10:
        lambdas = np.ones(R, dtype=float)
        for i in range(len(X.shape)):
            V = reduce(np.dot, [ A.T @ A for j, A in enumerate(As) if j != i ])

            tmp = tl.tenalg.khatri_rao([
                A for j, A in reversed(list(enumerate(As))) 
                if j != i 
            ])
            As[i] = Xs[i] @ tmp @ np.linalg.pinv(V)

            for j in range(R):
                # something strange happens

                # seems that imposing such orthonormality decreases
                # accuracy of decomposition
                length = np.linalg.norm(As[i][:, j])
                lambdas[j] *= length
                As[i][:, j] /= length
            
        it += 1

    return lambdas, As



filler = lambda i, j, k: i + j + k
A = np.fromfunction(filler, (2, 2, 3))
print(A)
lambdas, As = CP_ALS(A, 8)

print(np.einsum("r, ir, jr, kr -> ijk", lambdas, *As))










