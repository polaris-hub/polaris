from typing import Any, Literal, Union

import numpy as np
from sklearn.base import check_array
from polaris.utils.types import SlugCompatibleStringType


def listit(t: Any):
    """
    Converts all tuples in a possibly nested object to lists
    https://stackoverflow.com/questions/1014352/how-do-i-convert-a-nested-tuple-of-tuples-and-lists-to-lists-of-lists-in-python
    """
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def sluggify(sluggable: SlugCompatibleStringType):
    """
    Converts a string to a slug-compatible string.
    """
    return sluggable.lower().replace("_", "-")


def discretize(
    X: np.ndarray,
    thresholds: Union[np.ndarray, list],
    inplace: bool = False,
    allow_nan: bool = True,
    label_order: Literal["ascending", "descending"] = "ascending",
) -> np.ndarray:
    """
    Thresholding of array-like or scipy.sparse matrix into binary or multiclass labels.

    Args:
        X : The data to discretize, element by element.
            scipy.sparse matrices should be in CSR or CSC format to avoid an
            un-necessary copy.

        thresholds: Interval boundaries that include the right bin edge.

        inplace: Set to True to perform inplace discretization and avoid a copy
            (if the input is already a numpy array or a scipy.sparse CSR / CSC
            matrix and if axis is 1).

        allow_nan: Set to True to allow nans in the array for discretization. Otherwise,
            an error will be raised instead.

        label_order: The continuous values are discretized to labels 0, 1, 2, .., N with respect to given
            threshold bins [threshold_1, threshold_2,.., threshould_n].
            When set to 'ascending', the class label is in ascending order with the threshold
            bins that `0` represents negative class or lower class, while 1, 2, 3 are for higher classes.
            When set to 'descending' the class label is in ascending order with the threshold bins.
            Sometimes the positive labels are on the left side of provided threshold.
            E.g. For binarization with threshold [0.5],  the positive label is defined
            by`X < 0.5`. In this case, `label_order` should be `descending`.

    Returns:
        X_tr: The transformed data.
    """
    if label_order not in ["ascending", "descending"]:
        raise ValueError(
            f"{label_order} is not a valid label_order. Choose from 'ascending' or 'descending'."
        )

    X = check_array(
        X,
        accept_sparse=["csr", "csc"],
        copy=not inplace,
        force_all_finite="allow-nan" if allow_nan else True,
        ensure_2d=False,
    )

    nan_idx = np.isnan(X)

    thresholds = thresholds
    binarize = True if len(thresholds) == 1 else False

    if label_order == "descending":
        thresholds = np.flip(thresholds)
    X = np.digitize(X, thresholds)

    if allow_nan:
        X = X.astype(np.float64)
        X[nan_idx] = np.nan
    if binarize and label_order == "descending":
        X = 1 - X
    return X
