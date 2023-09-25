import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import datamol as dm
import warnings

warnings.filterwarnings("ignore")


def visulize_distribution(data_cols, dataset):
    for col in data_cols:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 4))
        to_plot = dataset.sort_values(by=col).reset_index()
        to_plot["order"] = range(to_plot.shape[0])
        scatter = sns.scatterplot(
            to_plot,
            x="order",
            y=col,
            hue=f"OUTLIER_{col}",
            palette={True: "red", False: "navy"},
            ax=axes[0],
        )
        stats.probplot(to_plot[col].values, dist="norm", plot=axes[1])
    plt.close()
    return fig


def verify_stereoisomers(data_cols, dataset):
    cliff_cols = [f"CLASS_{col}_stereo_cliff" for col in data_cols]
    for col in data_cols:
        print(f"Verify the stereo ismomers for readout `{col}`")
        cliff_col = f"CLASS_{col}_stereo_cliff"
        cliff = dataset[cliff_col].notna()
        if cliff.sum() > 0:
            to_plot = df_full.loc[
                cliff, ["smiles", "num_undefined_stereo_center", "num_defined_stereo_center", col, cliff_col]
            ]
            display(to_plot)
            display(dm.to_image([dm.to_mol(s) for s in to_plot.smiles]))
        else:
            print("-- No activity cliffs found in stereosimoers.")


def check_undefined_stereocenters(data_cols, dataset):
    to_plot = dataset[
        dataset[["num_undefined_stereo_center", "num_defined_stereo_center"]].notna().any(axis=1)
    ]
    legends = (
        to_plot[["num_undefined_stereo_center", "num_defined_stereo_center"]]
        .apply(
            lambda x: f"num_undefined_stereo_center:{x['num_undefined_stereo_center']}\n num_defined_stereo_center{x['num_defined_stereo_center']}",
            axis=1,
        )
        .tolist()
    )
    return dm.to_image(to_plot.mol.tolist(), legends=legends)
