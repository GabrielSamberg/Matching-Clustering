import numpy as np
from itertools import product
from joblib import Parallel, delayed
import ot

from matching_clustering.create_clouds import plot_point_clouds_side_by_side  # POT: pip install POT
# we don't need to import create_clouds since we are not using it here, but we do need the plotting function for visualizations in the notebook.



# below is essentially the RBF kernel (or Gaussian kernel) and the resulting graph Laplacian with the marginals a and b. 
# Google: python most efficient ways to construct RBF kernel matrix
def graph_laplacians(x, y, sigma_x=0.1, sigma_y=0.1):
    n_x, k_x = x.shape[0], x.shape[1]
    n_y, k_y = y.shape[0], y.shape[1]

    x_np = x.reshape(n_x, k_x)  # Renamed X_ to X_np to avoid conflict
    y_np = y.reshape(n_y, k_y)  # Renamed Y_ to Y_np to avoid conflict

    # Step 1: Compute pairwise distance vectors for each point
    dx = np.linalg.norm(x_np[:, None, :] - x_np[None, :, :], axis=2)  # shape (n, n)
    dy = np.linalg.norm(y_np[:, None, :] - y_np[None, :, :], axis=2)  # shape (n, n)

    # Step 2: Compute the similarity matrices W_X, W_Y based on the gaussian similarity function.
    w_x = np.exp(-dx**2/(2*sigma_x**2))
    w_y = np.exp(-dy**2/(2*sigma_y**2))

    # Step 3: Compute the degrees of points in both X and Y
    row_sums_x = np.sum(w_x, axis=1)
    row_sums_y = np.sum(w_y, axis=1)

    # Create diagonal matrices D_X, D_Y from row sums
    d_x = np.diag(row_sums_x)
    d_y = np.diag(row_sums_y)

    # Now the laplacians L_X, L_Y:
    l_x = d_x - w_x
    l_y = d_y - w_y

    # Also compute the non-uniform distributions instead on the uniformly weighted distance profiles
    sum_x = np.sum(row_sums_x)
    sum_y = np.sum(row_sums_y)
    a = row_sums_x / sum_x
    b = row_sums_y / sum_y

    return l_x, l_y, w_x, w_y, a, b

# make this applicable to any two input matrices.
# there are some python functions to compute pairwise distance matrices. 
# Google: python most efficient ways to construct pairwise distance matrices.
def cost_matrix_c(x, y, a=None, b=None):
    n_x, k_x = x.shape[0], x.shape[1]
    n_y, k_y = y.shape[0], y.shape[1]

    x_np = x.reshape(n_x, k_x)  # Renamed X_ to X_np to avoid conflict
    y_np = y.reshape(n_y, k_y)  # Renamed Y_ to Y_np to avoid conflict

    # Step 1: Compute pairwise distance vectors for each point
    dx = np.linalg.norm(x_np[:, None, :] - x_np[None, :, :], axis=2)  # shape (n, n)
    dy = np.linalg.norm(y_np[:, None, :] - y_np[None, :, :], axis=2)  # shape (n, n)

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


