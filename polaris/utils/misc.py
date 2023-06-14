import numpy as np
from typing import Optional

from loguru import logger
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans


class EmpiricalKernelMapTransformer:
    def __init__(self, n_samples: int, metric: str, random_state: Optional[int] = None):
        self._n_samples = n_samples
        self._random_state = random_state
        self._samples = None
        self._metric = metric

    def __call__(self, X):
        """Transforms a list of datapoints"""
        return self.transform(X)

    def transform(self, X):
        """Transforms a single datapoint"""
        if self._samples is None:
            rng = np.random.default_rng(self._random_state)
            self._samples = X[rng.choice(np.arange(len(X)), self._n_samples)]
        X = cdist(X, self._samples, metric=self._metric)
        return X


def get_outlier_bounds(X, factor: float = 1.5):
    q1 = np.quantile(X, 0.25)
    q3 = np.quantile(X, 0.75)
    iqr = q3 - q1

    lower = max(np.min(X), q1 - factor * iqr)
    upper = min(np.max(X), q3 + factor * iqr)

    return lower, upper


def get_kmeans_clusters(
    X,
    n_clusters: int,
    random_state: Optional[int] = None,
    return_centers: bool = False,
    base_metric: str = "euclidean",
):
    if base_metric != "euclidean":
        logger.debug(f"To use KMeans with the {base_metric} metric, we use the Empirical Kernel Map")
        transformer = EmpiricalKernelMapTransformer(
            n_samples=min(512, len(X)),
            metric=base_metric,
            random_state=random_state,
        )
        X = transformer(X)

    model = MiniBatchKMeans(n_clusters, random_state=random_state, compute_labels=True, n_init=3)
    model.fit(X)

    indices = model.labels_
    if not return_centers:
        return indices

    centers = model.cluster_centers_[indices]
    return indices, centers
