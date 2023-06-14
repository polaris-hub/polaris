from typing import Callable, Union

from loguru import logger
from sklearn.model_selection import GroupShuffleSplit

from polaris.utils.misc import get_kmeans_clusters


class KMeansSplit(GroupShuffleSplit):
    """Group-based split that uses the k-Mean clustering in the input space for splitting."""

    def __init__(
        self,
        n_clusters: int = 10,
        n_splits: int = 5,
        metric: Union[str, Callable] = "euclidean",
        *,
        test_size=None,
        train_size=None,
        random_state=None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._n_clusters = n_clusters
        self._cluster_metric = metric

    def _iter_indices(self, X, y=None, groups=None):
        """Generate (train, test) indices"""
        if groups is not None:
            logger.warning("Ignoring the groups parameter in favor of the predefined groups")
        groups = get_kmeans_clusters(
            X=X, n_clusters=self._n_clusters, random_state=self.random_state, base_metric=self._cluster_metric
        )
        yield from super()._iter_indices(X, y, groups)
