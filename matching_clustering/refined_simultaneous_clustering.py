import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


# -----------------------------
# Graph Laplacian + Spectral clustering
# -----------------------------

def compute_unnormalized_laplacian(W: np.ndarray) -> np.ndarray:
    """
    L = D - W for a (symmetric) similarity matrix W.
    """
    W_sym = 0.5 * (W + W.T)           # symmetrize (robust to tiny asymmetries)
    W_sym = np.maximum(W_sym, 0.0)    # clip numerical negatives
    d = W_sym.sum(axis=1)
    L = np.diag(d) - W_sym
    return L

def choose_k_via_eigengap(L: np.ndarray, max_k: int = 8) -> int:
    """
    Eigengap heuristic on the smallest (max_k+1) eigenvalues of L.
    Returns k = argmax gap + 1 (min 2).
    """
    L_sym = 0.5 * (L + L.T)
    k_eval = min(max_k + 1, max(2, L_sym.shape[0] - 1))
    vals, _ = eigsh(A=L_sym, k=k_eval, which='SM')  # smallest eigenvalues
    vals = np.sort(np.real(vals))
    gaps = np.diff(vals[:max_k+1])
    k = int(np.argmax(gaps) + 1)
    return max(2, k)

def spectral_clustering_from_W(W: np.ndarray, n_clusters: int | None = None, random_state: int = 0):
    """
    Spectral clustering on a precomputed similarity matrix W using the unnormalized Laplacian.
    Returns labels, L, U (spectral embedding), and chosen k.
    """
    L = compute_unnormalized_laplacian(W)
    if n_clusters is None:
        n_clusters = choose_k_via_eigengap(L, max_k=min(10, max(2, W.shape[0] - 2)))

    # Get k smallest eigenvectors of L
    L_sym = 0.5 * (L + L.T)
    k = min(n_clusters, max(2, W.shape[0] - 2))
    _, vecs = eigsh(A=L_sym, k=k, which='SM')

    # Normalize rows (often helps with unnormalized Laplacian)
    U = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    # k-means on rows of U
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(U)
    return labels, L, U, k

# -----------------------------
# Plotly side-by-side 3D figure
# -----------------------------

def plot_clouds_side_by_side_3d(X: np.ndarray, labels_X: np.ndarray,
                                Y: np.ndarray, labels_Y: np.ndarray,
                                title_left: str = "Cloud X", title_right: str = "Cloud Y"):
    unique_X = np.unique(labels_X)
    unique_Y = np.unique(labels_Y)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(title_left, title_right)
    )

    # Left: X
    for c in unique_X:
        mask = labels_X == c
        fig.add_trace(
            go.Scatter3d(
                x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
                mode="markers",
                marker=dict(size=4),
                name=f"X cluster {int(c)}",
                hovertemplate="X(%{x:.2f}, %{y:.2f}, %{z:.2f})<br>cluster=%{text}",
                text=[int(c)] * mask.sum(),
                showlegend=True
            ),
            row=1, col=1
        )
    fig.update_scenes(xaxis_title="x", yaxis_title="y", zaxis_title="z", row=1, col=1)

    # Right: Y
    for c in unique_Y:
        mask = labels_Y == c
        fig.add_trace(
            go.Scatter3d(
                x=Y[mask, 0], y=Y[mask, 1], z=Y[mask, 2],
                mode="markers",
                marker=dict(size=4),
                name=f"Y cluster {int(c)}",
                hovertemplate="Y(%{x:.2f}, %{y:.2f}, %{z:.2f})<br>cluster=%{text}",
                text=[int(c)] * mask.sum(),
                showlegend=True
            ),
            row=1, col=2
        )
    fig.update_scenes(xaxis_title="x", yaxis_title="y", zaxis_title="z", row=1, col=2)

    fig.update_layout(
        title_text="Spectral clustering on unnormalized Laplacians",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=60, b=0),
        height=600
    )
    fig.update_scenes(aspectmode="data", row=1, col=1)
    fig.update_scenes(aspectmode="data", row=1, col=2)
    return fig

if __name__ == "__main__":
    from matching_clustering.create_clouds import create_clouds
    from matching_clustering.graph_laplacians_and_c import graph_laplacians,cost_matrix_c
    from matching_clustering.utils import LapOT



    path_X = "Data/CAPOD/class1/m1.obj"
    path_Y = "Data/CAPOD/class1/m7.obj"
    X, Y = create_clouds(path_X, path_Y)
    X = X[:1000]
    Y = Y[:1000]
    L_X, L_Y, W_x, W_y, a, b = graph_laplacians(X, Y)
    W1_matrix = cost_matrix_c(X, Y, a, b)

    # Cost matrix
    C = W1_matrix

    # Laplacian matrices
    K = L_X
    L = L_Y
    l = 0.7  # lambda
    l_x = 1
    l_y = 1

    # optimize our objective
    prob = KSC(K, L, C, l, l_x, l_y, w=a, v=b)
    pi_fp = prob.solve(method='fp', verbose=True)

    # derive the switch matrices
    pi_switch_x = build_cluster_matrix(pi_fp, k=2)
    pi_switch_y = build_cluster_matrix(pi_fp.T, k=2)

    # refine the similarity matrices
    new_similarity_x, new_similarity_y = pi_switch_x*W_x, pi_switch_y*W_y

    # Run spectral clustering based on the refined similarity matrices
    labels_X, L_X, U_X, k_X = spectral_clustering_from_W(new_similarity_x, n_clusters=5, random_state=1)
    labels_Y, L_Y, U_Y, k_Y = spectral_clustering_from_W(new_similarity_y, n_clusters=5, random_state=1)
    print(f"[Info] Chosen k for cloud X: {k_X}")
    print(f"[Info] Chosen k for cloud Y: {k_Y}")

    fig = plot_clouds_side_by_side_3d(
        X, labels_X, Y, labels_Y,
        title_left=f"Cloud X (k={k_X})", title_right=f"Cloud Y (k={k_Y})"
    )
    fig.show()
