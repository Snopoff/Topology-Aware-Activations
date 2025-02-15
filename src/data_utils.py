import numpy as np
import pandas as pd

from typing import Tuple

import matplotlib.pyplot as plt
import plotly.graph_objects as go


def create_ring(center, inner_radius, outer_radius, num_points) -> np.ndarray:
    """
    Create a ring with a given center, inner radius, and outer radius containing num_points.
    """
    angles = np.linspace(0, 2 * np.pi, num_points)
    radii = np.sqrt(
        np.random.rand(num_points) * (outer_radius**2 - inner_radius**2)
        + inner_radius**2
    )
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.vstack((x, y)).T


def create_torus(center, major_radius, minor_radius, num_points) -> np.ndarray:
    """
    Create a torus with a given center, major radius, and minor radius containing num_points.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, 2 * np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()

    x = center[0] + (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
    y = center[1] + (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
    z = center[2] + minor_radius * np.sin(theta)

    return np.vstack((x, y, z)).T


def create_sphere(center, radius, num_points) -> np.ndarray:
    """
    Create a sphere with a given center and radius containing num_points.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()

    x = center[0] + radius * np.cos(theta) * np.sin(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(phi)

    return np.vstack((x, y, z)).T


def prepare_data_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    print(len(df.dropna()))


def scatterplot(
    x_coords,
    y_coords,
    color,
    z_coords=None,
    dim=2,
    engine="plotly",
    save=False,
    name="images/test.png",
    plotly_marker=dict(
        size=10, colorscale="RdBu", colorbar=dict(title="Colorbar"), opacity=1
    ),
    **kwargs,
):
    if engine == "matplotlib":
        fig = plt.figure()

        if dim == 2:
            ax = fig.add_subplot()
            ax.scatter(x_coords, y_coords, c=color)
        elif dim == 3:
            ax = fig.add_subplot(projection="3d")
            ax.scatter(x_coords, y_coords, z_coords, c=color)
        else:
            raise ValueError(f"Dimension {dim} is not supported.")

        if save:
            fig.savefig(name)
        return fig

    if engine == "plotly":
        plotly_marker["color"] = color
        fig = go.Figure()
        if dim == 2:
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker=plotly_marker,
                    **kwargs,
                )
            )

            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            if save:
                fig.write_image(name)
            return fig
        if dim == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="markers",
                    marker=plotly_marker,
                    **kwargs,
                )
            )

            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            if save:
                fig.write_image(name)
            return fig
