import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score as sk_average_precision_score
from sklearn.metrics import cohen_kappa_score as sk_cohen_kappa_score


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


def average_precision_score(y_true, y_score, **kwargs):
    """Scikit learn average_precision_score wrapper that throws an error if y_true has no positive class"""
    if len(y_true) == 0 or not np.any(y_true):
        raise ValueError("Average precision requires at least a single positive class")
    return sk_average_precision_score(y_true=y_true, y_score=y_score, **kwargs)
