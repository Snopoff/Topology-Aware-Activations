import matplotlib.pyplot as plt
# from persim import plot_diagrams


def scatterplot(
    x_coords,
    y_coords,
    color,
    z_coords=None,
    dim=2,
    engine="matplotlib",
    save=False,
    name="images/test.png",
):
    if engine == "matplotlib":
        if dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(x_coords, y_coords, c=color)
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(x_coords, y_coords, z_coords, c=color)
    if save:
        fig.savefig(name)
    return fig


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


def plot_dgm(dgm, **kwargs):
    raise NotImplementedError("Uncomment the line after installing `persim`.")
    # return plot_diagrams(dgm, show=True, **kwargs)


def main():
    pass


if __name__ == "__main__":
    main()
