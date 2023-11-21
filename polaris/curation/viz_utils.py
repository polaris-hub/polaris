from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from scipy import stats
import datamol as dm

from .utils import PandasDataFrame
from ._chemistry_curator import NUM_UNDEF_STEREO_CENTER, NUM_DEF_STEREO_CENTER


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
            raise ValueError(
                f"{outlier_col} is not found in the given dataset.\
                             Please run <polaris.curation.utils.outlier_detection> first."
            )
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


def verify_stereoisomers(data_cols: List[str], dataset: PandasDataFrame):
    """Visual verification the stereoisomers from the original dataset

    Args:
        data_cols: Column names which the activity cliff was previously detected.
        dataset: Dataframe which includes stereo chemistry information of the molcules.

    """
    figs = []
    for col in data_cols:
        logger.info(f"Verify the stereo ismomers for readout `{col}`")
        cliff_col = f"CLASS_{col}_stereo_cliff"
        if cliff_col not in dataset:
            raise ValueError(
                "The stereo chemistry information is unavailable. \
                             Please run the <polaris.curation.run_chemistry_curation> first."
            )
        cliff = dataset[cliff_col].notna()
        if cliff.sum() > 0:
            to_plot = dataset.loc[
                cliff, ["smiles", NUM_UNDEF_STEREO_CENTER, NUM_DEF_STEREO_CENTER, col, cliff_col]
            ]
            fig = dm.to_image([dm.to_mol(s) for s in to_plot.smiles])
            figs.append(fig)
        else:
            logger.info("No activity cliffs found in stereosimoers.")


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

    to_plot = dataset[dataset[[NUM_UNDEF_STEREO_CENTER, NUM_DEF_STEREO_CENTER]].notna().any(axis=1)]
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
