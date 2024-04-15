from enum import Enum
from typing import TypeVar, Union
import numpy as np
from pydantic import BaseModel
from scipy import stats

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")
NumpyNDArray = TypeVar("numpy.ndarray")


class LabelOrder(Enum):
    acs = "ascending"
    desc = "descending"


def discretizer(
    X: np.ndarray,
    thresholds: Union[np.ndarray, list],
    copy: bool = True,
    allow_nan: bool = True,
    label_order: LabelOrder = LabelOrder.acs.value,
):
    """Thresholding of array-like or scipy.sparse matrix into binary or multiclass labels.

    Args:
        X : The data to discretize, element by element.
            scipy.sparse matrices should be in CSR or CSC format to avoid an
            un-necessary copy.

        thresholds: Feature values below or equal to this are replaced by 0, above it by 1.
            Threshold may not be less than 0 for operations on sparse matrices.

        copy: Set to False to perform inplace discretization and avoid a copy
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

    See Also:
        Discretizer : Performs multiclass discretizer using the Transformer API
    """
    if label_order not in [LabelOrder.acs.value, LabelOrder.desc.value]:
        raise ValueError(
            f"Please specify `label_order` using the {LabelOrder.desc.value} or {LabelOrder.acs.value}"
        )

    X = check_array(
        X,
        accept_sparse=["csr", "csc"],
        copy=copy,
        force_all_finite="allow-nan" if allow_nan else True,
        ensure_2d=False,
    )

    nan_idx = np.isnan(X)

    binarize = True if len(thresholds) == 1 else False
    if label_order == LabelOrder.desc.value:
        thresholds = np.flip(thresholds)
    X = np.digitize(X, thresholds)
    if allow_nan:
        X = X.astype(np.float64)
        X[nan_idx] = np.nan
    if binarize and label_order == LabelOrder.desc.value:
        X = 1 - X
    return X


def modified_zscore(data: np.ndarray, consistency_correction: float = 1.4826):
    """
    The modified z score is calculated from the median absolute deviation (MAD).
    These values must be multiplied by a constant to approximate the standard deviation.

    The modified z score might be more robust than the standard z score because it relies
    on the median (MED) for calculating the z score.

            modified Z score = (X-MED)/(consistency_correction*MAD)

    """
    median = np.nanmedian(data)

    deviation_from_med = np.array(data) - median

    mad = np.nanmedian(np.abs(deviation_from_med))
    mod_zscore = deviation_from_med / (consistency_correction * mad)
    return mod_zscore, mad


class ZscoreOutlier(BaseModel):
    """Detect outliers by the absolute value of modified Zscore.
    Args:
        threshold: zscore threshold to define the outliers. By default, the outlier is defined if the absolute value
        of modified zscore greater than 3.
        modified: When set to 'True', modified Zscore is used.
    """

    threshold: float = 3
    modified: bool = False
    zscore_: NumpyNDArray = None
    mad: float = None

    def fit(self, X):
        if self.modified:
            self.zscore_, self.mad = modified_zscore(X)
        else:
            self.zscore_ = stats.zscore(X)

    def predict(self, X, y=None):
        check_is_fitted(self)
        result = np.ones_like(X)
        ind = np.argwhere(np.absolute(self.zscore_) > self.threshold)
        result[ind] = -1
        return result

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


OUTLIER_METHOD = {
    "iso": IsolationForest,
    "lof": LocalOutlierFactor,
    "svm": OneClassSVM,
    "ee": EllipticEnvelope,
    "zscore": ZscoreOutlier,
}


def outlier_detection(X: np.ndarray, method="zscore", **kwargs) -> np.ndarray:
    """Automatic detection of data outliers.

    Args:
        X: Array of values to be analyzed.
        method: Name of the algorithm to be used for outlier detection.
                The supported methods are
                    'iso' for IsolationForest,
                    'lof' for LocalOutlierFactor,
                    'svm' for OneClassSVM,
                    'ee' for EllipticEnvelope, and
                    'zscore' for ZscoreOutlier.
        kwargs: Additional parameters for the outlier detection algorithm.

    Returns:
        outlier_index: Index of detected outliers in the input data X.
    """
    if method not in OUTLIER_METHOD:
        raise ValueError("The detection name must be in 'iso', 'lof', 'svm', 'ee' and 'zscore'.")
    detector = OUTLIER_METHOD.get(method)(**kwargs)
    pred = detector.fit_predict(X)
    outlier_index = np.argwhere(pred == -1)
    return outlier_index
