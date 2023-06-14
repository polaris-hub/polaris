import enum
import numpy as np
import scipy.stats as ss
from typing import Union, Optional
from jenkspy import jenks_breaks
from scipy.signal import argrelextrema
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples


def _classify(value, breaks):
    """Assigns a value to a class given a set of class boundaries (i.e. breaks)"""
    for i in range(1, len(breaks)):
        if value <= breaks[i]:
            return i
    return len(breaks) - 1


def _goodness_of_variance_fit(array, classes, fn=jenks_breaks):
    """
    Computes the goodness of variance fit, a score in between 0 and 1.

    Adopted from:
    https://stats.stackexchange.com/questions/143974/jenks-natural-breaks-in-python-how-to-find-the-optimum-number-of-breaks
    """

    classes = fn(array, classes)
    classified = np.array([_classify(i, classes) for i in array])
    maxz = max(classified)
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    sdam = np.sum((array - array.mean()) ** 2)
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])
    gvf = (sdam - sdcm) / sdam
    return gvf


def _find_cluster_size(data, min_fitness=0.75, max_cluster=10):
    gvf = 0.0
    nclasses = 2
    while gvf < min_fitness and nclasses < max_cluster:
        gvf = _goodness_of_variance_fit(data, nclasses)
        nclasses += 1
    return nclasses


def _jenks(y, nclasses):
    intervals = jenks_breaks(y, n_classes=nclasses)
    idx_cluster = np.array([_classify(val, intervals) for val in y])
    return idx_cluster


def _kde_clustering(y, bw_method="silverman", margin=None):
    """Create clusters based on kde density estimation of given population"""
    kde = ss.gaussian_kde(y, bw_method=bw_method)
    if not margin:
        margin = y.min()
    s = np.linspace(y.min() - margin, y.max() + margin)
    score = [kde(x) for x in s]
    # list of local minimal and maximal
    mi = argrelextrema(score, np.less_equal)[0]
    intervals = [y.min()]
    max_val = y.max()
    # values between two local minimum is defined as one cluster
    for val in mi:
        if max_val > val > intervals[0]:
            intervals.append(val)
    intervals.append(max_val)
    # make sure intervals are sorted ascendant order
    idx_cluster = np.array([_classify(val, intervals) for val in y])
    return idx_cluster


class OneDimensionalClustering(enum.Enum):
    JENKS = "jenks"
    KDE = "kde"


class StratifiedDistributionSplit(GroupShuffleSplit):
    """
    Split a dataset using the values of a readout, so both train, test and valid have the same
    distribution of values. Instead of bining using some kind of interval (rolling_windows),
    We will use a 1D clustering of the readout instead. Support for nD clustering should be added
    at some point.
    """

    ALLOWED_TARGET_TYPES = ("binary", "continuous")

    def __init__(
        self,
        n_splits: int = 5,
        n_clusters: Optional[int] = None,
        algorithm: Union[OneDimensionalClustering, str] = OneDimensionalClustering.JENKS,
        algorithm_kwargs: Optional[dict] = None,
        *,
        test_size=None,
        train_size=None,
        random_state=None,
    ):
        super().__init__(
            n_splits=n_splits, random_state=random_state, train_size=train_size, test_size=test_size
        )
        self._n_clusters = n_clusters
        self.algorithm = (
            OneDimensionalClustering[algorithm.upper()] if isinstance(algorithm, str) else algorithm
        )
        self.algorithm_kwargs = algorithm_kwargs if algorithm_kwargs is not None else {}

    def _iter_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)

        y = np.asarray(y)
        type_of_target_y = type_of_target(y)

        if type_of_target_y not in self.ALLOWED_TARGET_TYPES:
            raise ValueError(
                f"Supported target types are: {self.ALLOWED_TARGET_TYPES}. Got {type_of_target_y} instead."
            )

        y = column_or_1d(y)
        sorted_idx = np.argsort(y)
        y_sorted = y[sorted_idx]

        if self._n_clusters is None:
            n_clusters = _find_cluster_size(y_sorted, max_cluster=int(np.sqrt(n_samples)))
        else:
            n_clusters = min(self._n_clusters, n_samples)

        if self.algorithm == OneDimensionalClustering.JENKS:
            if np.any(np.isnan(y_sorted)):
                raise ValueError("NaN values are not supported when using the Jenks algorithm.")
            clusters = _jenks(y_sorted, n_clusters)

        elif self.algorithm == OneDimensionalClustering.CKMEANS:
            clusters = _ckmeans(y_sorted, n_clusters)

        else:
            clusters = _kde_clustering(y_sorted, **self.algorithm_kwargs)

        yield from super()._iter_indices(X, y, groups=clusters)
