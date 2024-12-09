from dataclasses import dataclass
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Plot:
    title: str
    plotgrid: list[list["SubPlot"]]


@dataclass
class SubPlot:
    data_ND: np.ndarray
    title: str
    show_reference_sphere: bool


def setup_3d_axis(ax: Any, title: str):
    """Configure consistent 3D axis styling"""
    ax.set_title(title)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel("Neuron 1")
    ax.set_ylabel("Neuron 2")
    ax.set_zlabel("Neuron 3")
    ax.view_init(elev=20, azim=-40)
    ax.set_box_aspect([1, 1, 1])


def add_reference_sphere(ax: Any, radius: float):
    """Add reference sphere of specified radius to the plot."""
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    ax.plot_surface(x, y, z, alpha=0.1, color="gray")

def show_subplot(
    subplot: SubPlot,
    rows: int,
    cols: int,
    index: int,
    fig: plt.Figure,
    colors: np.ndarray,
):
    ax = fig.add_subplot(
        rows,
        cols,
        index,
        projection="3d"
    )
    ax.scatter(
        subplot.data_ND[:, 0],
        subplot.data_ND[:, 1],
        subplot.data_ND[:, 2],
        c=colors,
    )
    setup_3d_axis(ax, subplot.title)
    if subplot.show_reference_sphere:
        add_reference_sphere(ax, radius=np.sqrt(3))

def show_plot(plot: Plot):
    """Create visualization from prepared data"""
    fig = plt.figure(figsize=(15, 15))

    # Generate colors based on initial position
    colors = plot.plotgrid[0][0].data_ND @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colors = (colors - colors.min()) / (colors.max() - colors.min())

    # set title
    fig.suptitle(plot.title)
    
    # make square roughly
    rows = len(plot.plotgrid)
    cols = max(len(row) for row in plot.plotgrid)
    for row_index, row in enumerate(plot.plotgrid):
        for col_index, subplot in enumerate(row):
            # Plot original data
            show_subplot(
                subplot,
                rows=rows,
                cols=cols,
                index=row_index * cols + col_index + 1,
                fig=fig,
                colors=colors,

            )

    plt.tight_layout()
    plt.show() 

