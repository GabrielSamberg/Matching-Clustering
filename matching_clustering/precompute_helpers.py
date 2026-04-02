import numpy as np
from itertools import product
from joblib import Parallel, delayed
import ot
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

# below is essentially the RBF kernel (or Gaussian kernel) and the resulting graph Laplacian with the marginals a and b. 
# Google: python most efficient ways to construct RBF kernel matrix - DONE

def graph_laplacians(x, y, sigma_x=0.1, sigma_y=0.1, affinity=None):
    if affinity is None:
        w_x = rbf_kernel(x, gamma=1/(2*sigma_x**2)) # (N, N) dense, exact
        w_y = rbf_kernel(y, gamma=1/(2*sigma_y**2))  # (N, N) dense, exact

    else:
        w_x, w_y = affinity


    np.fill_diagonal(w_x, 0)                # remove self-loops
    d_x = np.diag(w_x.sum(axis=1))
    l_x = d_x - w_x
    a = w_x.sum(axis=1) / w_x.sum()         # Non uniform distribution a based on graph degrees in X

    
    np.fill_diagonal(w_y, 0)                # remove self-loops
    d_y = np.diag(w_y.sum(axis=1))
    l_y = d_y - w_y                             # or normalized variant
    b = w_y.sum(axis=1) / w_y.sum()             # Non uniform distribution a based on graph degrees in X
    return l_x, l_y, w_x, w_y, a, b


# make this applicable to any two input matrices. - DONE 
# there are some python functions to compute pairwise distance matrices. 
# Google: python most efficient ways to construct pairwise distance matrices.

def cost_matrix_c(dx, dy, a=None, b=None):
    assert dx.ndim == 2 and dx.shape[0] == dx.shape[1], "dx must be a square matrix"
    assert dy.ndim == 2 and dy.shape[0] == dy.shape[1], "dy must be a square matrix"
    
    n = dx.shape[0]
    m = dy.shape[0]

    a_uniform, b_uniform = np.ones(n)/n, np.ones(m)/m

    pairs = list(product(range(n), range(m)))

    if a is not None and b is not None:
        vals = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(ot.wasserstein_1d)(dx[i], dy[j], a, b) for i, j in pairs
        )
        w1_matrix = np.asarray(vals, dtype=float).reshape(n, m)
        return w1_matrix

    else:
        vals_uniform = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(ot.wasserstein_1d)(dx[i], dy[j], a_uniform, b_uniform) for i, j in pairs
        )
        w1_matrix_uniform = np.asarray(vals_uniform, dtype=float).reshape(n, m)
        return w1_matrix_uniform


def build_cluster_matrix(A: np.ndarray, k: int) -> np.ndarray:
    """
    Clusters the rows of matrix A into k clusters and returns a matrix X such that:
    X[i, j] = 1 if row i and row j are in the same cluster, else 0.
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "Matrix A must be square"

    # Convert to NumPy for sklearn

    # Perform KMeans clustering on the rows
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
    labels = kmeans.fit_predict(A)

    # Construct the nxn cluster indicator matrix
    x = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                x[i, j] = 1

    return x