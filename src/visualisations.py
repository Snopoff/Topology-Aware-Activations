import matplotlib.pyplot as plt

import plotly.graph_objects as go

from gudhi import plot_persistence_diagram


def scatterplot(
    x_coords,
    y_coords,
    color,
    z_coords=None,
    dim=2,
    engine="plotly",
    save=False,
    name="images/test.png",
    plotly_marker=dict(size=10, colorscale="RdBu", colorbar=dict(title="Colorbar"), opacity=1),
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
            return fig


def plot_persistence(data, is_from="torch_topological"):
    if is_from == "torch_topological":
        diagrams_dimensions = [element[1:] for element in data]
        diagrams_in_gudhi_format = []
        for diagram, dimension in diagrams_dimensions:
            diagrams_in_gudhi_format.extend([[dimension] + [tuple(point)] for point in diagram.numpy().tolist()])

        return plot_persistence_diagram(diagrams_in_gudhi_format)


def plot_line(x_range, y_range, label=None, title=None, save=False, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x_range, y_range, label=label)
    fig.suptitle(title)
    if save:
        fig.savefig(filename)


def plot_lines(
    x_ranges,
    y_ranges,
    stds=None,
    labels=None,
    titles=None,
    fig_title=None,
    xlabels=None,
    share_x_range=True,
    save=True,
    filename="",
    **kwargs,
):
    fig, axs = plt.subplots(**kwargs)
    axs = axs.flatten()
    n_axes = len(y_ranges)
    if share_x_range:
        x_ranges = [x_ranges] * n_axes
    for i, x_range in enumerate(x_ranges):
        for j, y_range in enumerate(y_ranges[i]):
            axs[i].plot(x_range, y_range, label=labels[i][j])
            axs[i].fill_between(x_range, y_range - stds[i][j], y_range + stds[i][j], alpha=0.1)
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_title(titles[i])
        axs[i].legend(loc="best")
    fig.suptitle(fig_title)
    if save:
        fig.savefig(filename)


def lineplot(
    x_range,
    y_ranges,
    stds=None,
    title=None,
    labels=None,
    xlabel=None,
    ylabel=None,
    save=True,
    filename=True,
):
    plt.cla()
    for i, y_range in enumerate(y_ranges):
        plt.plot(x_range, y_range, "D-", label=labels[i])
        plt.fill_between(x_range, y_range - stds[i], y_range + stds[i], alpha=0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    if save:
        plt.savefig(filename)
