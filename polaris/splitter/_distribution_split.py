import enum
import numpy as np
import scipy.stats as ss
from typing import Union, Optional, Sequence, Callable
from jenkspy import jenks_breaks
from numpy.random import RandomState
from scipy.signal import argrelextrema
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples  # noqa W0212


class Clustering1D(enum.Enum):
    """Enum with the available clustering algorithms"""

    JENKS = "jenks"
    KDE = "kde"


def _classify(value: float, breaks: Sequence[float]) -> int:
    """Assigns a value to a class given a set of class boundaries (i.e. breaks)"""
    for i in range(1, len(breaks)):
        if value <= breaks[i]:
            return i
    return len(breaks) - 1


def _goodness_of_variance_fit(
    readouts: np.ndarray, n_classes: int, get_class_boundaries_fn: Callable = jenks_breaks
):
    """
    Computes the goodness of variance fit, a score in between 0 and 1.

    Adopted from:
    https://stats.stackexchange.com/questions/143974/jenks-natural-breaks-in-python-how-to-find-the-optimum-number-of-breaks
    """
    # Automatically determine the class boundaries (or: breaks) and classify the datapoints
    classes = get_class_boundaries_fn(readouts, n_classes)
    class_indices = np.array([_classify(i, classes) for i in readouts])

    grouped_indices = [
        [idx for idx, val in enumerate(class_indices) if cls + 1 == val] for cls in range(max(class_indices))
    ]

    grouped_readouts = [readouts[indices] for indices in grouped_indices]

    def _sum_of_squared_deviations_from_mean(array: np.ndarray) -> float:
        return np.sum((array - array.mean()) ** 2).item()

    # sum of squared deviations from array mean
    sdam = _sum_of_squared_deviations_from_mean(readouts)
    sdcm = sum([_sum_of_squared_deviations_from_mean(readouts_group) for readouts_group in grouped_readouts])
    gvf = (sdam - sdcm) / sdam
    return gvf


def _find_no_classes(readouts: np.ndarray, min_fitness: float = 0.75, max_classes: int = 10):
    """
    Incrementally increments the number of classes until we either surpassed a goodness of variance fit threshold
    or until we reached the maximum number of classes.
    """
    gvf = 0.0
    nclasses = 2
    while gvf < min_fitness and nclasses < max_classes:
        gvf = _goodness_of_variance_fit(readouts, nclasses)
        nclasses += 1
    return nclasses


def _jenks(readouts: np.ndarray, n_classes: int):
    """
    Uses the Jenks Natural Breaks algorithm to classify the given readouts
    into an automatically determined number of classes.
    """
    breaks = jenks_breaks(readouts, n_classes=n_classes)
    class_indices = np.array([_classify(val, breaks) for val in readouts])
    return class_indices


def _kde_clustering(
    readouts: np.ndarray, bw_method: str = "silverman", margin: Optional[float] = None, num_samples: int = 50
):
    """
    Create clusters based on kde density estimation of given population. Computes the density for the readouts and
    assigns all readouts between local minima of the density to the same class.
    """

    margin = readouts.min() if margin is None else margin
    kde = ss.gaussian_kde(readouts, bw_method=bw_method)
    samples = np.linspace(readouts.min() - margin, readouts.max() + margin, num=num_samples)

    # Compute the density for 50 evenly spaced points
    density = np.array([kde(x) for x in samples])

    # Find the local minima in the density
    minima_indices = argrelextrema(density, np.less_equal)[0]

    # Values between two local minimum are assigned to the same class
    breaks = [samples[idx] for idx in minima_indices if readouts.min() < samples[idx] < readouts.max()]
    breaks.insert(0, readouts.min())
    breaks.append(readouts.max())

    # Classify the readouts
    class_indices = np.array([_classify(y, breaks) for y in readouts])
    return class_indices


class StratifiedDistributionSplit(GroupShuffleSplit):
    """
    Split a dataset using the values of a readout, so both train, test and valid have the same
    distribution of values. Instead of bining using some kind of interval (rolling_windows),
    we will instead use a 1D clustering of the readout.
    """

    ALLOWED_TARGET_TYPES = ("binary", "continuous")

    def __init__(
        self,
        n_splits: int = 5,
        n_clusters: Optional[int] = None,
        algorithm: Union[Clustering1D, str] = Clustering1D.JENKS,
        algorithm_kwargs: Optional[dict] = None,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            random_state=random_state,
            train_size=train_size,
            test_size=test_size,
        )
        self._n_clusters = n_clusters
        self.algorithm = Clustering1D[algorithm.upper()] if isinstance(algorithm, str) else algorithm
        self.algorithm_kwargs = algorithm_kwargs if algorithm_kwargs is not None else {}

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        if y is None:
            raise ValueError(f"{self.__class__.__name__} requires y to be defined.")

        y = np.asarray(y)
        type_of_target_y = type_of_target(y)

        if type_of_target_y not in self.ALLOWED_TARGET_TYPES:
            raise ValueError(
                f"Supported target types are: {self.ALLOWED_TARGET_TYPES}. Got {type_of_target_y} instead."
            )

        y = column_or_1d(y)
        sorted_idx = np.argsort(y)
        y_sorted = y[sorted_idx]

        n_samples = _num_samples(y)
        if self._n_clusters is None:
            n_clusters = _find_no_classes(y_sorted, max_classes=int(np.sqrt(n_samples)))
        else:
            n_clusters = min(self._n_clusters, n_samples)

        if self.algorithm == Clustering1D.JENKS:
            if np.any(np.isnan(y_sorted)):
                raise ValueError("NaN values are not supported when using the Jenks algorithm.")
            clusters = _jenks(y_sorted, n_clusters)

        else:
            clusters = _kde_clustering(y_sorted, **self.algorithm_kwargs)

        yield from super()._iter_indices(X, y, groups=clusters)
