import numpy as np
from itertools import product
from joblib import Parallel, delayed
import ot  # POT: pip install POT


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


if __name__ == "__main__":
    from create_clouds import create_clouds
    import matplotlib
    matplotlib.use("TkAgg")  # or "TkAgg" if you want an interactive window
    import matplotlib.pyplot as plt
    path_X = "Data/CAPOD/class3/m25.obj"
    path_Y = "Data/CAPOD/class3/m26.obj"
    X, Y = create_clouds(path_X, path_Y)
    X = X[:500, :]
    Y = Y[:500, :]
    a, b = graph_laplacians(X, Y)[4], graph_laplacians(X, Y)[5]

    W1_matrix = cost_matrix_c(X, Y, a, b)
    W1_matrix_uniform = cost_matrix_c(X, Y)

    W = W1_matrix - W1_matrix_uniform
    matrices = [W1_matrix, W1_matrix_uniform, W]
    titles = ['C non-uniform', 'C uniform', 'C_nonuni - C_uni']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        im = ax.imshow(matrices[i], cmap='viridis', interpolation='nearest')
        ax.set_title(titles[i])
        cbar = fig.colorbar(im, ax=ax)  # <-- IMPORTANT
        cbar.set_label('Pixel Value')

    plt.tight_layout()
    plt.show()
