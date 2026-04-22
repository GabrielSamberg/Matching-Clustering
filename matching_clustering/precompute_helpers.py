import numpy as np
from itertools import product
from joblib import Parallel, delayed
import ot
from sklearn.cluster import KMeans


# compute the cost matrix C where C[i, j] = W1(dx[i], dy[j]) for all i, j.
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
    # Perform KMeans clustering on the rows
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
    labels = kmeans.fit_predict(A)
    cluster_matrix = (labels[:, None] == labels[None, :]).astype(int)
    return cluster_matrix