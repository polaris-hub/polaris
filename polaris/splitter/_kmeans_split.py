import numpy as np
from numpy.random import RandomState
from typing import Callable, Union, Optional
from sklearn.model_selection import GroupShuffleSplit

from polaris.splitter._distance_split_base import convert_to_default_feats_if_smiles
from polaris.splitter.utils import get_kmeans_clusters


class KMeansSplit(GroupShuffleSplit):
    """Group-based split that uses the k-Mean clustering in the input space for splitting."""

    def __init__(
        self,
        n_clusters: int = 10,
        n_splits: int = 5,
        metric: Union[str, Callable] = "euclidean",
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._n_clusters = n_clusters
        self._cluster_metric = metric

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices"""
        if X is None:
            raise ValueError(f"{self.__class__.__name__} requires X to be provided.")

        X, self._cluster_metric = convert_to_default_feats_if_smiles(X, self._cluster_metric)
        groups = get_kmeans_clusters(
            X=X,
            n_clusters=self._n_clusters,
            random_state=self.random_state,
            base_metric=self._cluster_metric,
        )
        yield from super()._iter_indices(X, y, groups)
