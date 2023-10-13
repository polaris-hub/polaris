from enum import Enum
from typing import List, TypeVar, Union
import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
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


class Discretizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator, BaseModel):
    """Discretizer continuous data according to a list thresholds e.g. [threshold_1, threshold_2].

       In the above example, values falls in left of threshold_1 are mapped to 0, values fall in
       the interval between threshold_1 and threshold_2 are mapped to 1, while values fall on the right
       threshold_2 are mapped to 2. The threshold values must be in ascending or descending order.
       To reverse the label order, set `label_order` to `descending`.

    Args:
        thresholds : np.ndarray, default=0.0
            Feature values below or equal to this are replaced by 0, above it by 1.
            Threshold may not be less than 0 for operations on sparse matrices.

        copy : bool, default=True
            Set to False to perform inplace discretization and avoid a copy (if
            the input is already a numpy array or a scipy.sparse CSR matrix).

    See Also:
        discretizer : Equivalent function without the estimator API.
    """

    thresholds: List = Field(default=[0])
    copy_object: bool = Field(default=True, alias="copy")
    label_order: LabelOrder = Field(default=LabelOrder.acs)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data.

            y : None
                Ignored.

        Returns:
            self : object
                Fitted transformer.
        """
        self._validate_data(X, accept_sparse="csr")
        return self

    def transform(self, X, copy=None):
        """Convert each element of X to multiclass label.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data to binarize, element by element.
                scipy.sparse matrices should be in CSR format to avoid an
                un-necessary copy.

            copy : bool
                Copy the input X or not.

        Returns
            X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
                Transformed array.
        """
        copy = copy if copy is not None else self.copy
        # check_array
        X = self._validate_data(X, accept_sparse=["csr", "csc"], copy=copy, reset=False)
        return discretizer(X, thresholds=self.thresholds, copy=False, label_order=self.label_order)

    def _more_tags(self):
        return {"stateless": True}


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
