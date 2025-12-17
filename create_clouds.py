import pywavefront
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_clouds(path_x, path_y):
    scene1 = pywavefront.Wavefront(path_x)
    scene2 = pywavefront.Wavefront(path_y)
    x = np.array(scene1.vertices)
    y = np.array(scene2.vertices)
    return x, y


def plot_point_clouds_side_by_side(X: np.ndarray, Y: np.ndarray):
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

    fig.show()
    return fig


if __name__ == "__main__":
    path_X = "Data/CAPOD/class1/m1.obj"
    path_Y = "Data/CAPOD/class1/m7.obj"
    X, Y = create_clouds(path_X, path_Y)
    plot_point_clouds_side_by_side(X, Y)
