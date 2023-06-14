import tqdm

import numpy as np
import pandas as pd
import seaborn as sns

from loguru import logger
from dataclasses import dataclass
from typing import Dict, Union, Callable, Optional, List

from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from sklearn.model_selection import BaseShuffleSplit
from sklearn.neighbors import NearestNeighbors

from polaris.utils.plot import plot_distance_distributions
from polaris.utils.misc import get_outlier_bounds
from ._base import convert_to_default_feats_if_smiles


@dataclass
class _SplitCharacterization:
    """
    Within the context of MOOD, a split is characterized by
    a distribution of distances and an associated representativeness score.
    This class groups some functionality in a single place.
    """

    distances: np.ndarray
    representativeness: float
    label: str

    @classmethod
    def merge(cls, splits):
        names = set([obj.label for obj in splits])
        if len(names) != 1:
            raise RuntimeError("Can only concatenate equally labeled split characterizations")
        dist = np.concatenate([obj.distances for obj in splits])
        score = np.mean([obj.representativeness for obj in splits]).item()
        return cls(dist, score, names.pop())

    @staticmethod
    def sort(splits):
        return sorted(splits, key=lambda spl: spl.representativeness)

    @staticmethod
    def best(splits):
        return _SplitCharacterization.sort(splits)[-1]

    @staticmethod
    def as_dataframe(splits):
        df = pd.DataFrame()
        best = _SplitCharacterization.best(splits)
        for split in splits:
            df_ = pd.DataFrame(
                {
                    "split": split.label,
                    "representativeness": split.representativeness,
                    "best": split == best,
                },
                index=[0],
            )
            df = pd.concat((df, df_), ignore_index=True)
        df["rank"] = df["representativeness"].rank(ascending=False)
        return df

    def __eq__(self, other):
        return self.label == other.label and self.representativeness == other.representativeness

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}[{self.label}]"


class MOODSplitter(BaseShuffleSplit):
    """
    The MOOD splitter takes in multiple candidate splitters and a set of
    deployment molecules you plan to use a model on and prescribes one splitting method
    that creates the test set that is most representative of the deployment set.
    """

    def __init__(
        self,
        candidate_splitters: Dict[str, BaseShuffleSplit],
        metric: Union[str, Callable] = "minkowski",
        p: int = 2,
        k: int = 5,
        n_jobs: Optional[int] = None,
    ):
        """
        Creates the splitter object.

        Args:
            candidate_splitters: A list of splitter methods you are considering
            metric: The distance metric to use. Needs to be supported by `sklearn.neighbors.NearestNeighbors`
            p: If the metric is the minkowski distance, this is the p in that distance.
            k: The number of nearest neighbors to use to compute the distance.
        """
        super().__init__()
        if not all(isinstance(obj, BaseShuffleSplit) for obj in candidate_splitters.values()):
            raise TypeError("All splitters should be BaseShuffleSplit objects")

        n_splits_per_splitter = [obj.get_n_splits() for obj in candidate_splitters.values()]
        if not len(set(n_splits_per_splitter)) == 1:
            n_splits = np.max(n_splits_per_splitter)
            logger.warning(
                f"n_splits is inconsistent across the different splitters. "
                f"Setting the value for each splitter to the maximum of {n_splits} splits."
            )
            for splitter in candidate_splitters:
                splitter.n_splits = n_splits
            n_splits_per_splitter = [n_splits] * len(candidate_splitters)

        self.n_splits = n_splits_per_splitter[0]

        self._p = p
        self._k = k
        self._metric = metric
        self._splitters = candidate_splitters
        self._n_jobs = n_jobs

        self._split_chars = None
        self._prescribed_splitter_label = None
        self._split_axes = None

    @staticmethod
    def visualize(
        downstream_distances: np.ndarray, splits: List[_SplitCharacterization], ax: Optional = None
    ):
        """
        Visualizes the results of the splitting protocol by visualizing
        the test-to-train distance distributions resulting from each of the candidate splitters
        and coloring them based on their representativeness.
        """
        splits = sorted(splits, key=lambda spl: spl.representativeness)
        cmap = sns.color_palette("rocket", len(splits) + 1)

        distances = [spl.distances for spl in splits]
        colors = [cmap[rank + 1] for rank, spl in enumerate(splits)]
        labels = [spl.label for spl in splits]

        ax = plot_distance_distributions(distances, labels, colors, ax=ax)

        lower, upper = get_outlier_bounds(downstream_distances, factor=3.0)
        mask = (downstream_distances >= lower) & (downstream_distances <= upper)
        downstream_distances = downstream_distances[mask]

        sns.kdeplot(downstream_distances, color=cmap[0], linestyle="--", alpha=0.3, ax=ax)
        return ax

    @staticmethod
    def score_representativeness(downstream_distances, distances, num_samples: int = 100):
        """
        Scores a candidate split by comparing the test-to-train and deployment-to-dataset distributions.
        A higher score should be interpreted as _more_ representative
        """
        pdf_split = gaussian_kde(distances)
        pdf_downstream = gaussian_kde(downstream_distances)

        vmin = np.min(np.concatenate((downstream_distances, distances)))
        vmax = np.max(np.concatenate((downstream_distances, distances)))
        positions = np.linspace(vmin, vmax, num=num_samples)

        samples_split = pdf_split(positions)
        samples_downstream = pdf_downstream(positions)

        return 1.0 - jensenshannon(samples_downstream, samples_split, base=2)

    @property
    def prescribed_splitter_label(self):
        """Textual identifier of the splitting method that was deemed most representative."""
        if not self._fitted:
            raise RuntimeError("The splitter has not be fitted yet")
        return self._prescribed_splitter_label

    @property
    def _fitted(self):
        return self._prescribed_splitter_label is not None

    def _compute_distance(self, X_from, X_to):
        """
        Computes the k-NN distance from one set to another

        Args:
            X_from: The set to compute the distance for
            X_to: The set to compute the distance to (i.e. the neighbor candidates)
        """
        knn = NearestNeighbors(n_neighbors=self._k, metric=self._metric, p=self._p).fit(X_to)
        distances, ind = knn.kneighbors(X_from)
        distances = np.mean(distances, axis=1)
        return distances

    def get_prescribed_splitter(self) -> BaseShuffleSplit:
        """Returns the prescribed scikit-learn Splitter object that is most representative"""
        return self._splitters[self.prescribed_splitter_label]

    def get_protocol_visualization(self) -> plt.Axes:
        """Visualizes the results of the splitting protocol"""
        if self._split_axes is None:
            raise RuntimeError("The splitter has not been fitted yet")
        return self._split_axes

    def get_protocol_results(self) -> pd.DataFrame:
        """Returns the results of the splitting protocol in tabular form"""
        if self._split_chars is None:
            raise RuntimeError("The splitter has not been fitted yet")
        return _SplitCharacterization.as_dataframe(self._split_chars)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        X_deployment: Optional[np.ndarray] = None,
        deployment_distances: Optional[np.ndarray] = None,
        plot: bool = False,
        progress: bool = False,
    ):
        """
        Follows the MOOD specification protocol to prescribe a train-test split that
        is most representative of the deployment setting and as such closes the
        testing-deployment gap.

        The k-NN distance in the representation space is used as a proxy of difficulty.
        The further a datapoint is from the training set, the lower the expected model's performance.
        Using that observation, we select the train-test split that best replicates the distance
        distribution (i.e. "the difficulty") of the deployment set.

        Args:
            X: An array of (n_samples, n_features)
            y: An array of (n_samples, 1) targets, passed to candidate splitter's split() method
            groups: An array of (n_samples,) groups, passed to candidate splitter's split() method
            X_deployment: An array of (n_deployment_samples, n_features)
            deployment_distances:  An array of (n_deployment_samples, 1) precomputed distances.
            plot: Whether to visualize the evaluation of the different candidate splitters
            progress: Whether to show a progress bar
        """

        if X_deployment is None and deployment_distances is None:
            raise ValueError(f"You need to specify either `X_deployment` or `deployment_distances`")

        deployment_is_smiles = all(isinstance(x, str) for x in X_deployment)
        train_is_smiles = all(isinstance(x, str) for x in X)

        if deployment_is_smiles != train_is_smiles:
            raise ValueError("Deployment and training set must both be SMILES or both be feature vectors")

        # Convert to default FP if SMILES
        X, self._metric = convert_to_default_feats_if_smiles(X, self._metric, n_jobs=self._n_jobs)
        X_deployment, _ = convert_to_default_feats_if_smiles(X_deployment, self._metric, n_jobs=self._n_jobs)

        if deployment_distances is None:
            deployment_distances = self._compute_distance(X_deployment, X)

        # Precompute all splits. Since splitters are implemented as generators,
        # we store the resulting splits so we can replicate them later on.
        split_chars = list()

        it = self._splitters.items()
        if progress:
            it = tqdm.tqdm(it, desc="Splitter")

        for name, splitter in it:
            # We possibly repeat the split multiple times to
            # get a more reliable  estimate
            chars = []

            it_ = splitter.split(X, y, groups)
            if progress:
                it_ = tqdm.tqdm(it_, leave=False, desc="Split", total=self.n_splits)

            for train, test in it_:
                distances = self._compute_distance(X[test], X[train])
                distances = distances[np.isfinite(distances)]
                distances = distances[~np.isnan(distances)]

                score = self.score_representativeness(deployment_distances, distances)
                chars.append(_SplitCharacterization(distances, score, name))

            split_chars.append(_SplitCharacterization.merge(chars))

        # Rank different splitting methods by their ability to
        # replicate the downstream distance distribution.
        chosen = _SplitCharacterization.best(split_chars)

        self._split_chars = split_chars
        self._prescribed_splitter_label = chosen.label

        logger.info(
            f"Ranked all different splitting methods:\n{_SplitCharacterization.as_dataframe(split_chars)}"
        )
        logger.info(f"Selected {chosen.label} as the most representative splitting method")

        if not plot:
            return

        # Visualize the results
        ax = self.visualize(deployment_distances, split_chars)
        self._split_axes = ax
        return ax

    def _iter_indices(self, X=None, y=None, groups=None):
        """Generate (train, test) indices"""
        if not self._fitted:
            raise RuntimeError("The splitter has not be fitted yet")
        yield from self.get_prescribed_splitter()._iter_indices(X, y, groups)
