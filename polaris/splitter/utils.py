import numpy as np
from typing import Optional

from loguru import logger
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans


class EmpiricalKernelMapTransformer:
    """
    Transforms a dataset using the Empirical Kernel Map method.
    In this, a point is defined by its distance to a set of reference points.
    After this transformation, one can use the euclidean metric even if the original space was not euclidean compatible.
    """

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
            # Select the reference set
            rng = np.random.default_rng(self._random_state)
            self._samples = X[rng.choice(np.arange(len(X)), self._n_samples)]
        # Compute the distance to the reference set
        X = cdist(X, self._samples, metric=self._metric)
        return X


def get_iqr_outlier_bounds(X, factor: float = 1.5):
    """
    Return the bounds for outliers using the Inter-Quartile Range (IQR) method.
    Returns a lower and upper bound. Any value exceeding these bounds is considered an outlier.
    """
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
    """
    Get the k-means clusters for a set of datapoints

    If the base metric is not euclidean, we use the Empirical Kernel Map
    to transform the data into a euclidean compatible space.
    """

    # Use the Empirical Kernel Map if the base metric is not euclidean
    if base_metric != "euclidean":
        logger.debug(f"To use KMeans with the {base_metric} metric, we use the Empirical Kernel Map")
        transformer = EmpiricalKernelMapTransformer(
            n_samples=min(512, len(X)),
            metric=base_metric,
            random_state=random_state,
        )
        X = transformer(np.array(X))

    # Perform the clustering
    model = MiniBatchKMeans(n_clusters, random_state=random_state, compute_labels=True, n_init=3)
    model.fit(X)

    indices = model.labels_
    if not return_centers:
        return indices

    centers = model.cluster_centers_[indices]
    return indices, centers
