from typing import Union, List, Any, Optional

import numpy as np
import pandas as pd


class DatasetClassConverter:
    """Apply a list of thresholds to a column of continuous values and return a column
    with associated labels.

    Args:
        data_columns: A list of data columns to convert
        converter_params: A list of dictionaries for class conversion for the above data columns.
            The parameters see <MultiClassConverter>
    """

    def __init__(self, data_columns: List[str], converter_params: List[dict]):
        self.converters = {
            data_col: MultiClassConverter(**params)
            for data_col, params in zip(data_columns, converter_params)
        }

    def transform(
        self, dataset: Union[pd.DataFrame, np.ndarray], to_bool: bool = False, prefix: str = "CLASS_"
    ):
        """
        Args:
            dataset: Data to apply the thresholds to.
            to_bool: Whether to convert to boolean. Only possible when the
                number of label is exactly 2.
            prefix: Prefix for the new column name
        Returns:
            data: Converted dataset.
        """
        for data_col, converter in self.converters.items():
            dataset[f"{prefix}{data_col}"] = converter.transform(data=dataset[data_col], to_bool=to_bool)
        return dataset


class MultiClassConverter:
    """Apply a list of thresholds to a column of continuous values and return a column
    with associated labels.

    Args:
        threshold_values: A list of float values. inf and -inf will be added
            during binning.
        threshold_labels: A list of ordered labels.
        nan_value: Value to replace the nans
        positive_label: Label for positive class.
    """

    def __init__(
        self,
        threshold_values: List[float],
        threshold_labels: List[str],
        nan_value: Any = np.nan,
        positive_label: Optional[str] = None,
    ):
        self.threshold_labels = threshold_labels
        self.threshold_values = list(threshold_values)
        # Augment the bins to include inf/-inf
        self.bins = [-np.inf] + self.threshold_values + [np.inf]

        # Threshold values should be sorted
        if threshold_values != sorted(self.threshold_values):
            raise ValueError("The threshold values must monotonically increase.")

        self.nan_value = nan_value
        self.positive_label = positive_label
        self.negative_label = (
            list(filter(lambda x: x != self.positive_label, self.threshold_labels))[0]
            if self.positive_label is not None
            else None
        )

    def transform(
        self,
        data: Union[pd.Series, np.ndarray],
        to_bool: bool = False,
    ):
        """
        Args:
            data: Data to apply the threshold to.
            to_bool: Whether to convert to boolean. Only possible when the
                number of label is exactly 2.
        Return:
            class_column: Converted data.
        """
        # Apply the thresholds
        classes_column = pd.cut(data, bins=self.bins, right=False, labels=self.threshold_labels)

        # Convert category type to string and replace
        # nan as needed.
        classes_column = pd.Series(classes_column).astype(str)
        classes_column = classes_column.replace(str(np.nan), self.nan_value)

        if to_bool:
            return self.convert_to_bool(classes_column)
        return classes_column

    def convert_to_bool(self, data: pd.Series):
        """
        Args:
            data: Data tp convert to boolean
        """
        if len(self.threshold_labels) != 2:
            raise ValueError(
                "When using `to_bool=True`, the number of labels must be exactly 2 instead of"
                f" {len(self.threshold_labels)}"
            )

        if self.positive_label is None:
            raise ValueError(
                "When using `to_bool=True`, the `positive_label` arg must be provided and not None."
            )

        data = data.replace(self.positive_label, True)
        data = data.replace(self.negative_label, False)

        return data
