import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List

from polaris.splitter.utils import get_iqr_outlier_bounds


def plot_distance_distributions(
    distances,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    styles: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    outlier_factor: Optional[float] = 3.0,
):
    """Visualizes the distribution of distances between the clusters in the style of the MOOD paper."""
    n = len(distances)
    show_legend = True

    # Set defaults
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if colors is None:
        cmap = sns.color_palette("rocket", n)
        colors = [cmap[i] for i in range(n)]
    if labels is None:
        show_legend = False
        labels = [""] * n
    if styles is None:
        styles = ["-"] * n

    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])

    if outlier_factor is not None:
        all_distances = np.concatenate(distances)
        lower, upper = get_iqr_outlier_bounds(all_distances, factor=outlier_factor)
        distances = [X[(X >= lower) & (X <= upper)] for X in distances]

    # Visualize all splitting methods
    for idx, dist in enumerate(distances):
        sns.kdeplot(dist, color=colors[idx], linestyle=styles[idx], ax=ax, label=labels[idx])

    ax.set_xlabel(f"Distance")

    if show_legend:
        ax.legend()
    return ax
