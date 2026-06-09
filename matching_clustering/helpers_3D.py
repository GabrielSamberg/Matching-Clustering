import pywavefront
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import SpectralClustering
from plotly.colors import qualitative
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional


# This is 3D specific, so maybe name the file 3D_helpers.py instead of create_clouds.py?
# Also, rename the functions below to be something like create_3D_clouds and plot_3D_clouds_side_by_side for clarity.


def create_3D_clouds(path_x, path_y):
    scene1 = pywavefront.Wavefront(path_x)
    scene2 = pywavefront.Wavefront(path_y)
    x = np.array(scene1.vertices)
    y = np.array(scene2.vertices)
    return x, y


def plot_3D_point_clouds_side_by_side(X: np.ndarray, Y: np.ndarray, title=None):
    """
    Plot two 3D point clouds side by side in a single Plotly figure.

    Args:
        X, Y: (N, 3) numpy arrays representing the point clouds.
    """
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must have shape (N, 3)")
    if Y.ndim != 2 or Y.shape[1] != 3:
        raise ValueError("Y must have shape (N, 3)")

    # Create side-by-side 3D subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("Cloud X", "Cloud Y")
    )

    # Plot X
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.8, color="royalblue"),
        ),
        row=1, col=1
    )

    # Plot Y
    fig.add_trace(
        go.Scatter3d(
            x=Y[:, 0],
            y=Y[:, 1],
            z=Y[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.8, color="crimson"),
        ),
        row=1, col=2
    )

    # Layout adjustments
    fig.update_layout(
        title="3D Point Clouds: X vs Y",
        height=600,
        width=1200,
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=False,
    )

    # Keep axes aspect ratio consistent
    fig.update_scenes(aspectmode="data")

    # ---- REMOVE AXES & GRID COMPLETELY ----
    axis_off = dict(
        visible=False,
        showgrid=False,
        showline=False,
        zeroline=False,
        showticklabels=False,
    )
    if title is None:
        title = "title"
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis=axis_off,
            yaxis=axis_off,
            zaxis=axis_off,
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
        ),
        scene2=dict(
            xaxis=axis_off,
            yaxis=axis_off,
            zaxis=axis_off,
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig



def _validate_cloud(arr: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr.astype(np.float64, copy=False)

def spectral_cluster_labels(
    points: np.ndarray,
    k: int,
    gamma: Optional[float] = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run spectral clustering with an RBF affinity on a single point cloud.

    Args:
        points: (N, 3) array.
        k: number of clusters.
        gamma: RBF gamma (1/(2*sigma^2)). If None, sklearn default (1.0) is used.
        random_state: for reproducibility.

    Returns:
        labels: (N,) cluster labels in [0, k-1].
    """
    pts = _validate_cloud(points, "points")
    n = pts.shape[0]
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, N], got k={k}, N={n}")

    # SpectralClustering parameters tuned for stability on small-to-medium point clouds.
    sc = SpectralClustering(
        n_clusters=k,
        affinity="rbf",
        gamma=gamma,
        assign_labels="kmeans",
        random_state=random_state,
        n_init=10,
        eigen_solver=None,  # let sklearn choose
    )
    labels = sc.fit_predict(pts)
    return labels


def make_plotly_palette(k):
    """
    Build a matplotlib colormap using Plotly qualitative palettes.
    """
    palette = (
        qualitative.Plotly
        + qualitative.D3
        + qualitative.Set3
        + qualitative.Dark24
    )

    colors = [palette[i % len(palette)] for i in range(k)]

    return ListedColormap(colors)


def prepare_label_coloring(labels_x, labels_y):
    all_labels = np.concatenate([labels_x, labels_y])
    unique_labels = np.unique(all_labels)

    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    labels_x_idx = np.array([label_to_idx[l] for l in labels_x])
    labels_y_idx = np.array([label_to_idx[l] for l in labels_y])

    cmap = make_plotly_palette(len(unique_labels))

    return labels_x_idx, labels_y_idx, cmap


def set_axes_equal(ax, pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2

    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def remove_axes_visuals(ax):
    ax.set_axis_off()
    ax.grid(False)

    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)


# def plot_point_clouds_only(
#     X,
#     Y,
#     labels_X,
#     labels_Y,
#     elev=20,
#     azim=45,
#     point_size=4,
#     alpha=1.0,
#     figsize=(10, 5),
#     save_path=None,
#     plot_clouds_only=False
# ):
    



#     labels_X_idx, labels_Y_idx, cmap = prepare_label_coloring(labels_X, labels_Y)

#     fig = plt.figure(figsize=figsize)

#     ax1 = fig.add_subplot(1, 2, 1, projection="3d")
#     ax2 = fig.add_subplot(1, 2, 2, projection="3d")


#     if plot_clouds_only:
#         colors_x = ['royalblue' for label in labels_X_idx]

#         ax1.scatter(
#             X[:, 0], X[:, 1], X[:, 2],
#             c=colors_x,
#             s=point_size,
#             alpha=alpha
#         ) 

#         colors_y = ['crimson' for label in labels_Y_idx]  
#         ax2.scatter(
#             Y[:, 0], Y[:, 1], Y[:, 2],
#             c=colors_y,
#             s=point_size,
#             alpha=alpha
#         )
#     else:
#         ax1.scatter(
#             X[:, 0], X[:, 1], X[:, 2],
#             c=labels_X_idx,
#             cmap=cmap,
#             s=point_size,
#             alpha=alpha
#         )

#         ax2.scatter(
#             Y[:, 0], Y[:, 1], Y[:, 2],
#             c=labels_Y_idx,
#             cmap=cmap,
#             s=point_size,
#             alpha=alpha
#         )

#     for ax, pts in [(ax1, X), (ax2, Y)]:
#         ax.view_init(elev=elev, azim=azim)
#         set_axes_equal(ax, pts)
#         remove_axes_visuals(ax)

   
#     plt.subplots_adjust(wspace=0, hspace=0)
#     if save_path is not None:
#         plt.savefig(save_path, bbox_inches="tight")
#     plt.show()


def zoom_3d_axis(ax, pts, zoom=1.0):
    """
    Zooms a 3D axis by changing x/y/z limits around the cloud center.

    zoom > 1  : zoom in
    zoom < 1  : zoom out
    zoom = 1  : no zoom
    """
    pts = np.asarray(pts)

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    )

    half_range = 0.5 * max_range / zoom

    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)

def plot_point_clouds_only(
    X,
    Y,
    labels_X,
    labels_Y,
    elev_X=20,
    azim_X=45,
    roll_X=0,
    elev_Y=20,
    azim_Y=45,
    roll_Y=0,
    zoom_X=1.0,
    zoom_Y=1.0,
    point_size=4,
    alpha=1.0,
    figsize=(10, 5),
    save_path=None,
    plot_clouds_only=False,
    wspace=0
):

    labels_X_idx, labels_Y_idx, cmap = prepare_label_coloring(labels_X, labels_Y)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    if plot_clouds_only:
        colors_x = ['royalblue' for label in labels_X_idx]

        ax1.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=colors_x,
            s=point_size,
            alpha=alpha
        )

        colors_y = ['crimson' for label in labels_Y_idx]

        ax2.scatter(
            Y[:, 0], Y[:, 1], Y[:, 2],
            c=colors_y,
            s=point_size,
            alpha=alpha
        )

    else:
        ax1.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=labels_X_idx,
            cmap=cmap,
            s=point_size,
            alpha=alpha
        )

        ax2.scatter(
            Y[:, 0], Y[:, 1], Y[:, 2],
            c=labels_Y_idx,
            cmap=cmap,
            s=point_size,
            alpha=alpha
        )

    # Left plot: X
    ax1.view_init(elev=elev_X, azim=azim_X, roll=roll_X)
    zoom_3d_axis(ax1, X, zoom=zoom_X)
    remove_axes_visuals(ax1)

    # Right plot: Y
    ax2.view_init(elev=elev_Y, azim=azim_Y, roll=roll_Y)
    zoom_3d_axis(ax2, Y, zoom=zoom_Y)
    remove_axes_visuals(ax2)

    plt.subplots_adjust(wspace=wspace, hspace=0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()