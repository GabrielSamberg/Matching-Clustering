import numpy as np
import ot

def lot_quad(C, K, L, a, b, r, l, l_x, l_y, alpha, gamma, tol=1e-9, max_iter=1000, dykstra_iter=1000, initialize_U=None, initialize_V=None):
    # C = cost matrix
    # K = row quadratic term
    # L = column quadratic term
    # a, b = marginals
    # r = rank
    # l = regularization parameter
    # alpha = for stability
    # gamma = fixed step size
    n, m = C.shape

    if initialize_U is None and initialize_V is None:



        # random initialization
        U = np.random.rand(n, r)
        V = np.random.rand(m, r)
        g = np.random.rand(r)

        # Try different initialization
       # U = a[:, None] / r
       # U = np.repeat(U, r, axis=1)

       # V = b[:, None] / r
       # V = np.repeat(V, r, axis=1)

       # g = np.ones(r)/r

    else:
      U,V = initialize_U, initialize_V
      g = np.ones(r)/r

    for k in range(max_iter):
        P = U @ np.diag(1/g) @ V.T
        M = C + l_x*(K + K.T) @ P + l_y*P @ (L + L.T)
        grad_dual_norm_U = np.max(M @ V @ np.diag(1/g)+ l*np.log(U))
        grad_dual_norm_V = np.max(M.T @ U @ np.diag(1/g)+ l*np.log(V))
        grad_dual_norm_g = np.max(-np.diag(U.T @ M @ V)/(g**2) + l*np.log(g))
        adapt_gamma = gamma/np.max([grad_dual_norm_U, grad_dual_norm_V, grad_dual_norm_g])**2
        xi1 = U * np.exp(-adapt_gamma*(M @ V @ np.diag(1/g) + l*np.log(U)))
        xi2 = V * np.exp(-adapt_gamma*(M.T @ U @ np.diag(1/g) + l*np.log(V)))
        xi3 = g * np.exp(-adapt_gamma*(-np.diag(U.T @ M @ V)/(g**2) + l*np.log(g)))
        U, V, g = ot.lowrank._LR_Dysktra(xi1, xi2, xi3, a, b, alpha, stopThr=tol, numItermax=dykstra_iter, warn=True)
        P = U @ np.diag(1 / g) @ V.T
        if k % 10 == 0:
            row_err = np.sum(np.abs(np.sum(P, axis=1) - a))
            col_err = np.sum(np.abs(np.sum(P, axis=0) - b))
            err = row_err + col_err
            print(f'Iteration {k}, error: {err}')
            print(f'Objective {np.sum(P * C) + l * np.sum(U * np.log(U)) + l * np.sum(V * np.log(V)) + l * np.sum(g * np.log(g))}')

    P = U @ np.diag(1 / g) @ V.T

    return P
