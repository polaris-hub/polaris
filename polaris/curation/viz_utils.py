from loguru import logger
from typing import List, Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import datamol as dm
import pandas as pd
import umap

from .utils import PandasDataFrame
from ._chemistry_curator import (
    NUM_UNDEF_STEREO_CENTER,
    NUM_DEF_STEREO_CENTER,
    NUM_STEREOISOMERS,
    NO_STEREO_UNIQUE_ID,
)


def visualize_distribution(data_cols: List[str], dataset: PandasDataFrame):
    """Visualize the distribution of the data and highlight the potential outliers.

    Args:
        data_cols: Column names which the `OUTLIER` was previously detected.
        dataset: Dataset Dataframe which inludes columns of `OUTLIER` flags.

    Returns:
        figs: List of distribution plots.
    """
    figs = []
    for col in data_cols:
        outlier_col = f"OUTLIER_{col}"
        if outlier_col not in dataset.columns:
            logger.warning(
                f"{outlier_col} is not found in the given dataset. Please run <polaris.curation.utils.outlier_detection> first."
            )
        else:
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 4))
            to_plot = dataset.sort_values(by=col).reset_index()
            to_plot["order"] = range(to_plot.shape[0])
            sns.scatterplot(
                to_plot,
                x="order",
                y=col,
                hue=outlier_col,
                palette={True: "red", False: "navy"},
                ax=axes[0],
            )
            stats.probplot(to_plot[col].values, dist="norm", plot=axes[1])
            figs.append(fig)
    return figs


def verify_stereoisomers(
    data_cols: List[str],
    dataset: PandasDataFrame,
    legend_cols: List[str] = None,
    mol_col: str = "smiles",
):
    """Visual verification the stereoisomers from the original dataset

    Args:
        data_cols: Column names which the activity cliff was previously detected.
        dataset: Dataframe which includes stereo chemistry information of the molcules.

    """
    figs = []
    if legend_cols is None:
        legend_cols = []
    for col in data_cols:
        logger.info(f"Verify the stereo ismomers for readout `{col}`")
        cliff_col = f"CLASS_{col}_stereo_cliff"
        if cliff_col not in dataset:
            cliff_col = f"{col}_stereo_cliff"
            if cliff_col not in dataset:
                raise ValueError(
                    "The stereo chemistry information is unavailable. \
                                Please run the <polaris.curation.run_chemistry_curation> first."
                )
        cliff = dataset[cliff_col].notna()
        if cliff.sum() > 0:
            to_plot = dataset.loc[
                cliff,
                [
                    mol_col,
                    NUM_UNDEF_STEREO_CENTER,
                    NUM_DEF_STEREO_CENTER,
                    NO_STEREO_UNIQUE_ID,
                    col,
                    cliff_col,
                ]
                + legend_cols,
            ].sort_values(by=NO_STEREO_UNIQUE_ID)
            fig_cols = [col] + legend_cols
            legends = (
                to_plot[fig_cols]
                .apply(
                    lambda x: "\n".join(
                        [f"{fig_col}: {x[i]}" for i, fig_col in enumerate(fig_cols)]
                    ),
                    axis=1,
                )
                .tolist()
            )
            fig = dm.to_image([dm.to_mol(s) for s in to_plot[mol_col]], legends=legends)
            figs.append(fig)
        else:
            logger.info("No activity cliffs found in stereosimoers.")
    return figs


def check_undefined_stereocenters(dataset: PandasDataFrame):
    """Visual check the molecules with undefined stereo centers

    Args:
        dataset: Dataframe which includes stereo chemistry information of the molcules.

    Returns:
        fig: Image figure of the molecules with stereo center annotations.

    See Also:
        <polaris.curation.run_chemistry_curation>

    """

    if NUM_UNDEF_STEREO_CENTER not in dataset or NUM_DEF_STEREO_CENTER not in dataset:
        raise ValueError(
            "The stereo information of the molecules are unavailable.\
                         Please run the <polaris.curation.run_chemistry_curation> first."
        )
    dataset = dataset.query(f"{NUM_STEREOISOMERS} > 0 & {NUM_UNDEF_STEREO_CENTER} >0")
    to_plot = dataset[
        dataset[[NUM_UNDEF_STEREO_CENTER, NUM_DEF_STEREO_CENTER]].notna().any(axis=1)
    ]
    legends = (
        to_plot[[NUM_UNDEF_STEREO_CENTER, NUM_DEF_STEREO_CENTER]]
        .apply(
            lambda x: f"num_undefined_stereo_center:{x[NUM_UNDEF_STEREO_CENTER]}\n num_defined_stereo_center{x[NUM_DEF_STEREO_CENTER]}",
            axis=1,
        )
        .tolist()
    )
    fig = dm.to_image(to_plot.mol.tolist(), legends=legends)
    return fig


def visualize_chemspace(
    data: pd.DataFrame,
    split_names: List[str],
    mol_col: str = "smiles",
    size_col=None,
    seed=1428,
    umap_metric="jaccard",
):
    """
    Visualize the chemical space by doing a dimensionality reduction with UMAP.
    -- Inputs --
        data: a dataframe
        split_names: names of the types of splits, should be found in columns of the "data" dataframe
        mol_col: string column where we will look for a molecule to featurize
        size_col: used for setting the style of the resulting scatterplot
        seed: int, random seed to use for reproducibility
        umap_metric: similarity metric to use for constructing the UMAP representation
    -- Outputs --
        figs, figures for each kind of split specified in split_names
    """
    figs = plt.figure(num=3)
    features = [
        dm.to_fp(mol) for mol in data[mol_col]
    ]  # Convert molecules to fingerprint
    embedding = umap.UMAP(metric=umap_metric, random_state=seed).fit_transform(
        features
    )  # Embed the features with UMAP using a similarity metric
    data["UMAP_0"], data["UMAP_1"] = embedding[:, 0], embedding[:, 1]
    for split_name in split_names:
        plt.figure()
        fig = sns.scatterplot(
            data=data,
            x="UMAP_0",
            y="UMAP_1",
            style=size_col,
            hue=split_name,
            alpha=0.7,
            palette="colorblind",
        )
        fig.set_title(f"UMAP embedding of compounds for {split_name}")
    return figs


def detailed_distributions_plots(
    df: pd.DataFrame,
    thresholds: Dict[str, Tuple[int, Callable]] = None,
    label_names: List[str] = None,
    log_scale_mapping: Dict[str, bool] = None,
    positive_color: str = "#3db371",
    negative_color: str = "#a9a9a9",
    n_cols: int = 3,
    fig_base_size: float = 8,
    w_h_ratio: float = 0.5,
    legend_fontsize: int = 18,
    ticks_fontsize: int = 18,
    title_fontsize: int = 18,
    gridsize: int = 1000,
    dpi: int = 150,
    seaborn_theme: Optional[str] = "whitegrid",
):
    """Plot the detailed distribution of the columns in `df`. Also, color the part of the
    "positive" distribution using `thresholds`.

    Args:
        df: A dataframe with binarized readouts only. NaN are allowed.
        thresholds: A dict mapping of the `df` column. Value is a tuple where the first
            element is the threshold value and the second element is a callable deciding wether
            a datapoint meets the criterai or not (something like `np.less` or np.greater`).
        label_names: Name of the labels (same order as the columns in `df`). If not set
            the name of the columns are used.
        log_scale_mapping: A dict mapping of the `df` column. If True,
            the plot for this readout will be log scaled.
        positive_color: Color for `True` or `1`.
        negative_color: Color for `False` or `0`.
        n_cols: Number of columns in the subplots.
        fig_base_size: Base size of the plots.
        w_h_ratio: Width/height ratio.
        legend_fontsize: Font size of the legend.
        ticks_fontsize: Font size of the x ticks and x label.
        title_fontsize: Font size of the title.
        gridsize: Gridsize for the kernel density estimate (KDE).
        dpi: DPI value of the figure.
        seaborn_theme: Seaborn theme.
    """

    # NOTE: the `thresholds` API is not super nice, consider an alternative.
    # NOTE: we could eventually add support for multiclass here if we need it.
    if thresholds is None:
        thresholds = {}

    if log_scale_mapping is None:
        log_scale_mapping = {}

    if label_names is None:
        label_names = df.columns.tolist()

    # Check all columns are numeric
    numerics = df.apply(lambda x: x.dtype.kind in "biufc")
    if not numerics.all():
        raise ValueError(
            f"Not all columns are numeric: {numerics[~numerics].to_dict()}"
        )

    if seaborn_theme is not None:
        sns.set_theme(style=seaborn_theme)

    n_plots = len(df.columns)

    # Compute the size of the figure
    if n_cols > n_plots:
        n_cols = n_plots

    n_rows = n_plots // n_cols
    if n_plots % n_cols > 0:
        n_rows += 1

    fig_w = fig_base_size * n_cols
    fig_h = fig_base_size * w_h_ratio * n_rows

    # Create the figure
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        dpi=dpi,
    )
    if isinstance(axes, list):
        axes = list(axes.flatten())
    else:
        axes = [axes]

    for ax, readout, label_name in zip(axes, df.columns, label_names):
        values = df[readout].dropna()

        # Get threshold value and function
        threshold_value, threshold_fn = None, None
        threshold = thresholds.get(readout, None)
        if threshold is not None:
            threshold_value, threshold_fn = threshold

        # Whether to log scale
        log_scale = log_scale_mapping.get(readout, False)

        # Draw distribution and kde plot
        kde_kws = {}
        kde_kws["clip"] = values.min(), values.max()
        kde_kws["gridsize"] = gridsize
        kplot = sns.histplot(
            values,
            kde=True,
            ax=ax,
            color=negative_color,
            kde_kws=kde_kws,
            log_scale=log_scale,
        )

        # Label
        ax.set_title(label_name, fontsize=title_fontsize)
        ax.set_xlabel(None)
        ax.set_ylabel("Count", fontsize=ticks_fontsize)

        ax.xaxis.set_tick_params(labelsize=ticks_fontsize)
        ax.yaxis.set_tick_params(labelsize=ticks_fontsize)

        if threshold_value is not None and threshold_fn is not None:
            # Fill between on active values
            x, y = kplot.get_lines()[0].get_data()
            ax.fill_between(
                x,
                y,
                where=threshold_fn(x, threshold_value),
                facecolor=positive_color,
                alpha=0.8,
            )

            # Active ratio text box
            positive_ratio = (
                threshold_fn(values, threshold_value).sum() / len(values) * 100
            )
            ax.text(
                0.85,
                0.95,
                f"{positive_ratio:.1f} %",
                transform=ax.transAxes,
                fontsize=legend_fontsize,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            logger.warning(f"Threshold not available for readout '{readout}'")

    # Remove unused axes
    _ = [fig.delaxes(a) for a in axes[n_plots:]]

    return fig


def display_chemspace(
    data: pd.DataFrame,
    mol_col: str,
    split: tuple = None,
    split_name: str = None,
    data_cols: list = None,
    method: Literal["tsne", "umap"] = "tsne",
    nrows: int = 2,
):
    """Show chemical space of molecule, optionally between traint/test split"""
    mols = data[mol_col].apply(dm.to_mol)
    features = np.array([dm.to_fp(mol) for mol in mols])
    if method == "umap":
        embedding = umap.UMAP().fit_transform(features)
    elif method == "tsne":
        embedding = TSNE(n_components=2).fit_transform(features)
    else:
        raise ValueError("Specify the embedding method")
    data[f"{method}_0"], data[f"{method}_1"] = embedding[:, 0], embedding[:, 1]
    if split is not None and split_name is not None:
        data.loc[split[0], split_name] = "train"
        data.loc[split[1], split_name] = "test"

    ncols = 1
    nrows = 1 if data_cols is None else nrows

    if data_cols is not None:
        if split_name is not None:
            ncols += len(data_cols)
        else:
            ncols = len(data_cols)
        ncols = np.ceil(ncols / nrows).astype(int)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 6, 5 * nrows))
    if data_cols is not None:
        axes = axes.flatten()
        for i, col in enumerate(data_cols):
            sns.scatterplot(
                data=data,
                x=f"{method}_0",
                y=f"{method}_1",
                hue=data[data_cols[i]].values,
                ax=axes[i],
                s=20,
            )
            axes[i].set_title(f"{method} embedding\n{col}")
        ax = axes[-1]
    else:
        ax = axes
    if split_name is not None:
        sns.scatterplot(
            data=data,
            x=f"{method}_0",
            y=f"{method}_1",
            hue=data[split_name].values,
            ax=ax,
            s=20,
        )
        ax.set_title(f"{method} embedding of compounds for {split_name}")
    fig.tight_layout()
    return fig
