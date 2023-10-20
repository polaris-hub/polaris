from enum import Enum
from typing import Callable

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, cohen_kappa_score
from scipy import stats

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
        fn: The callable that actually computes the metric
        is_multitask: Whether the metric expects a single set of predictions or a dict of predictions.
    """

    fn: Callable
    is_multitask: bool = False
    average: str = None


class Metric(Enum):
    """
    A metric within the Polaris ecosystem is uniquely identified by its name
    and is associated with additional metadata in a `MetricInfo` instance.

    Implemented as an enum.
    """

    # TODO (cwognum):
    #  - Add support for more metrics
    #  - Any preprocessing needed? For example changing the shape / dtype? Converting from torch tensors or lists?

    mean_absolute_error = MetricInfo(fn=mean_absolute_error)
    mean_squared_error = MetricInfo(fn=mean_squared_error)
    r2_score = MetricInfo(fn=r2_score)
    pearsonr = MetricInfo(fn=pearsonr)
    spearman = MetricInfo(fn=spearman)
    explained_var = MetricInfo(fn=explained_variance_score)

    accuracy = MetricInfo(fn=accuracy_score)
    f1_score_macro = MetricInfo(fn=f1_score, average="marco")
    f1_score_binary = MetricInfo(fn=f1_score, average="binary")
    f1_score_micro = MetricInfo(fn=f1_score, average="micro")
    roc_auc = MetricInfo(fn=roc_auc_score)
    pr_auc = MetricInfo(fn=average_precision_score)
    mcc = MetricInfo(fn=matthews_corrcoef)
    cohen_kappa = MetricInfo(fn=cohen_kappa_score)


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
        return self.fn(y_true, y_pred)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """For convenience, make metrics callable"""
        return self.score(y_true, y_pred)
