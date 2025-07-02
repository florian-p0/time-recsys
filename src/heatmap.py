import matplotlib.pyplot as plt
import numpy as np
import evaluate_result
import matplotlib
import matplotlib as mpl

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

algo_encode = {"Pop":0, "SVD":1, "UU":2, "Bias":3, "BiasedMF":4, "HPF":5, "II":6}


np.random.seed(19680801)

fig = plt.plot()

prefix = r"C:\Users\flerp\repos\time-recsys\Results"
folders = [     r"Amazon Electronics",
                r"Amazon Instant Video",
                r"Amazon Software",
                r"Amazon Video Games",
                r"Beer Advocate",
                r"Food Com",
                r"ML-1M",
                r"ML-100k"          ]

results = evaluate_result.evaluate_folders(prefix, folders)

n_datasets = 8
n_epochs = 14
data = np.random.randn(n_datasets, n_epochs)
for i in range(n_datasets):
    for j in range(n_epochs):
        if j < len(results[i][4]):
            print(results[i][4][j])
            data[i,j] = algo_encode[results[i][4][j]]
            print(data[i,j])
        else:
            data[i,j] = None

print(data)

y = [     r"Amazon Electronics",
                r"Amazon Instant Video",
                r"Amazon Software",
                r"Amazon Video Games",
                r"Beer Advocate",
                r"Food Com",
                r"ML-1M",
                r"ML-100k"          ]
x = [f"Epoch {i}" for i in range(1, 15)]

qrates = list(algo_encode.keys())
norm = matplotlib.colors.BoundaryNorm(np.linspace(-0.5, 6.5, 8), 7)
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[x])

im, _ = heatmap(data, y, x,
                cmap=mpl.colormaps['tab10'].resampled(7), norm=norm,
                cbar_kw=dict(ticks=np.arange(0, 7), format=fmt),
                cbarlabel="best algorithm")



def func(x, pos):
    return f"{x:.2f}".replace("0.", ".").replace("1.00", "")

plt.tight_layout()
plt.show()