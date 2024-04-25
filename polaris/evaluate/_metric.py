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
    balanced_accuracy_score,
)

from polaris.utils.types import DirectionType


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate a pearson r correlation"""
    return stats.pearsonr(y_true, y_pred).statistic


def spearman(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate a Spearman correlation"""
    return stats.spearmanr(y_true, y_pred).statistic


def absolute_average_fold_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Absolute Average Fold Error (AAFE) metric.
    It measures the fold change between predicted values and observed values.
    The implementation is based on [this paper](https://pubs.acs.org/doi/10.1021/acs.chemrestox.3c00305).

    Args:
        y_true: The true target values of shape (n_samples,)
        y_pred: The predicted target values of shape (n_samples,).

    Returns:
        aafe: The Absolute Average Fold Error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same.")

    if np.any(y_true == 0):
        raise ValueError("`y_true` contains zero which will result `Inf` value.")

    aafe = np.mean(np.abs(y_pred) / np.abs(y_true))

    return aafe


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
    needs_probs: bool = False


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
    absolute_average_fold_error = MetricInfo(fn=absolute_average_fold_error, direction=1)

    # binary and multiclass classification
    accuracy = MetricInfo(fn=accuracy_score, direction="max")
    balanced_accuracy = MetricInfo(fn=balanced_accuracy_score, direction="max")
    mcc = MetricInfo(fn=matthews_corrcoef, direction="max")
    cohen_kappa = MetricInfo(fn=cohen_kappa_score, direction="max")
    pr_auc = MetricInfo(fn=average_precision_score, direction="max", needs_probs=True)

    # binary only
    f1 = MetricInfo(fn=f1_score, kwargs={"average": "binary"}, direction="max")
    # note: At the moment, multi-dimension inputs for classification are not supported
    roc_auc = MetricInfo(fn=roc_auc_score, direction="max", needs_probs=True)

    # multiclass tasks only
    f1_macro = MetricInfo(fn=f1_score, kwargs={"average": "macro"}, direction="max")
    f1_micro = MetricInfo(fn=f1_score, kwargs={"average": "micro"}, direction="max")
    roc_auc_ovr = MetricInfo(
        fn=roc_auc_score, kwargs={"multi_class": "ovr"}, direction="max", needs_probs=True
    )
    roc_auc_ovo = MetricInfo(
        fn=roc_auc_score, kwargs={"multi_class": "ovo"}, direction="max", needs_probs=True
    )

    @property
    def fn(self) -> Callable:
        """The callable that actually computes the metric"""
        return self.value.fn

    @property
    def is_multitask(self) -> bool:
        """Whether the metric expects a single set of predictions or a dict of predictions."""
        return self.value.is_multitask

    @property
    def needs_probs(self) -> bool:
        """Whether the metric expects preditive probablities."""
        return self.value.needs_probs

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
