from enum import Enum
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from polaris.utils.types import DirectionType


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate a pearson r correlation"""
    return stats.pearsonr(y_true, y_pred).statistic


def spearman(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate a Spearman correlation"""
    return stats.spearmanr(y_true, y_pred).statistic


class MetricInfo(BaseModel):
    """
    Metric metadata

    Attributes:
        fn: The callable that actually computes the metric.
        is_multitask: Whether the metric expects a single set of predictions or a dict of predictions.
        kwargs: Additional parameters required for the metric.
        direction:  The direction for ranking of the metric,  "max" for maximization and "min" for minimization.
    """

    fn: Callable
    is_multitask: bool = False
    kwargs: dict = Field(default_factory=dict)
    direction: DirectionType


class Metric(Enum):
    """
    A metric within the Polaris ecosystem is uniquely identified by its name
    and is associated with additional metadata in a `MetricInfo` instance.

    Implemented as an enum.
    """

    # TODO (cwognum):
    #  - Any preprocessing needed? For example changing the shape / dtype? Converting from torch tensors or lists?

    # regression
    mean_absolute_error = MetricInfo(fn=mean_absolute_error, direction="min")
    mean_squared_error = MetricInfo(fn=mean_squared_error, direction="min")
    r2 = MetricInfo(fn=r2_score, direction="max")
    pearsonr = MetricInfo(fn=pearsonr, direction="max")
    spearmanr = MetricInfo(fn=spearman, direction="max")
    explained_var = MetricInfo(fn=explained_variance_score, direction="max")

    # classification
    accuracy = MetricInfo(fn=accuracy_score, direction="max")
    f1 = MetricInfo(fn=f1_score, kwargs={"average": "binary"}, direction="max")
    f1_macro = MetricInfo(fn=f1_score, kwargs={"average": "macro"}, direction="max")
    f1_micro = MetricInfo(fn=f1_score, kwargs={"average": "micro"}, direction="max")
    roc_auc = MetricInfo(fn=roc_auc_score, direction="max")
    pr_auc = MetricInfo(fn=average_precision_score, direction="max")
    mcc = MetricInfo(fn=matthews_corrcoef, direction="max")
    cohen_kappa = MetricInfo(fn=cohen_kappa_score, direction="max")
    # TODO: adding metrics for multiclass tasks

    @property
    def fn(self) -> Callable:
        """The callable that actually computes the metric"""
        return self.value.fn

    @property
    def is_multitask(self) -> bool:
        """Whether the metric expects a single set of predictions or a dict of predictions."""
        return self.value.is_multitask

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Endpoint for computing the metric.

        For convenience, calling a `Metric` will result in this method being called.

        ```python
        metric = Metric.mean_absolute_error
        assert metric.score(y_true=first, y_pred=second) == metric(y_true=first, y_pred=second)
        ```
        """
        return self.fn(y_true, y_pred, **self.value.kwargs)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """For convenience, make metrics callable"""
        return self.score(y_true, y_pred)
