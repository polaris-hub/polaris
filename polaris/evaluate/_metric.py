from enum import Enum
from typing import Callable, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score as sk_cohen_kappa_score,
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


def cohen_kappa_score(y_true, y_pred, **kwargs):
    """Scikit learn cohen_kappa_score wraper with renamed arguments"""
    return sk_cohen_kappa_score(y1=y_true, y2=y_pred, **kwargs)


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
    y_type: Literal["y_pred", "y_prob", "y_score"] = "y_pred"


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
    pr_auc = MetricInfo(fn=average_precision_score, direction="max", y_type="y_score")

    # binary only
    f1 = MetricInfo(fn=f1_score, kwargs={"average": "binary"}, direction="max")
    roc_auc = MetricInfo(fn=roc_auc_score, direction="max", y_type="y_score")

    # multiclass tasks only
    f1_macro = MetricInfo(fn=f1_score, kwargs={"average": "macro"}, direction="max")
    f1_micro = MetricInfo(fn=f1_score, kwargs={"average": "micro"}, direction="max")
    roc_auc_ovr = MetricInfo(
        fn=roc_auc_score, kwargs={"multi_class": "ovr"}, direction="max", y_type="y_score"
    )
    roc_auc_ovo = MetricInfo(
        fn=roc_auc_score, kwargs={"multi_class": "ovo"}, direction="max", y_type="y_score"
    )
    # TODO: add metrics to handle multitask multiclass predictions.

    @property
    def fn(self) -> Callable:
        """The callable that actually computes the metric"""
        return self.value.fn

    @property
    def is_multitask(self) -> bool:
        """Whether the metric expects a single set of predictions or a dict of predictions."""
        return self.value.is_multitask

    @property
    def y_type(self) -> bool:
        """Whether the metric expects preditive probablities."""
        return self.value.y_type

    def score(
        self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None, y_prob: Optional[np.ndarray] = None
    ) -> float:
        """Endpoint for computing the metric.

        For convenience, calling a `Metric` will result in this method being called.

        ```python
        metric = Metric.mean_absolute_error
        assert metric.score(y_true=first, y_pred=second) == metric(y_true=first, y_pred=second)
        ```
        """
        if y_pred is None and y_prob is None:
            raise ValueError("Neither `y_pred` nor `y_prob` is specified.")

        if self.y_type == "y_pred":
            if y_pred is None:
                raise ValueError(f"{self} requires `y_pred` input. ")
            pred = y_pred
        else:
            if y_prob is None:
                raise ValueError(f"{self} requires `y_prob` input. ")
            pred = y_prob

        kwargs = {"y_true": y_true, self.y_type: pred}
        return self.fn(**kwargs, **self.value.kwargs)

    def __call__(
        self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None, y_prob: Optional[np.ndarray] = None
    ) -> float:
        """For convenience, make metrics callable"""
        return self.score(y_true, y_pred, y_prob)
