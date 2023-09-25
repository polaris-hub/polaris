from typing import List, Literal, Optional, Sequence, Union

import numpy as np

from polaris.dataset import Dataset
from polaris.utils.context import tmp_attribute_change
from polaris.utils.errors import TestAccessError
from polaris.utils.types import DataFormat, DatapointType


class Subset:
    """The `Subset` class provides easy access to a single partition of a split dataset.

    This should be the starting point for any framework-specific (e.g. PyTorch, Tensorflow) data-loader implementation.
    How the data is loaded in Polaris can be non-trivial, so this class is provided to abstract away the details.
    To easily build framework-specific data-loaders, a `Subset` supports various styles of accessing the data:

    1. **In memory**: Loads the entire dataset in memory and returns a single array with all datapoints,
        this style is accessible through the `subset.targets` and `subset.inputs` properties.
    2. **List**: Index the subset like a list, this style is accessible through the `subset[idx]` syntax.
    3. **Iterator**: Iterate over the subset, this style is accessible through the `iter(subset)` syntax.

    Examples:
        The different styles of accessing the data:

        ```python
        import polaris as po

        benchmark = po.load_benchmark(...)
        train, test = benchmark.get_train_test_split()

        # Load the entire dataset in memory, useful for e.g. scikit-learn.
        X = train.inputs
        y = train.targets

        # Access a single datapoint as with a list, useful for e.g. PyTorch.
        x, y = train[0]

        # Iterate over the dataset, useful for very large datasets.
        for x, y in train:
            ...
        ```

    Raises:
        TestAccessError: When trying to access the targets of the test set (specified by the `hide_targets` attribute).
    """

    _SUPPORTED_FORMATS = ["dict", "tuple"]

    def __init__(
        self,
        dataset: Dataset,
        indices: List[Union[int, Sequence[int]]],
        input_cols: Union[List[str], str],
        target_cols: Union[List[str], str],
        input_format: DataFormat = "dict",
        target_format: DataFormat = "tuple",
        hide_targets: bool = False,
    ):
        self.dataset = dataset
        self.indices = indices
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
        self.input_cols = input_cols if isinstance(input_cols, list) else [input_cols]

        # Validate the output format
        if input_format not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported output format {input_format}. Choose from {self._SUPPORTED_FORMATS}"
            )
        if target_format not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported output format {target_format}. Choose from {self._SUPPORTED_FORMATS}"
            )
        self._input_format = input_format
        self._target_format = target_format

        # For the iterator implementation
        self._pointer = 0

        # This is a protected attribute to make explicit it should not be changed once set.
        self._hide_targets = hide_targets

    @property
    def is_multi_task(self):
        return len(self.target_cols) > 1

    @property
    def is_multi_input(self):
        return len(self.input_cols) > 1

    @property
    def inputs(self):
        """
        Scikit-learn style access to the inputs.
        If the dataset is multi-input, this will return a dict of arrays.
        """
        return self.as_array("x")

    @property
    def targets(self):
        """
        Scikit-learn style access to the targets.
        If the dataset is multi-target, this will return a dict of arrays.
        """
        return self.as_array("y")

    @staticmethod
    def _convert(data: dict, order: List[str], fmt: str):
        """Converts from the default dict format to the specified format"""
        if len(data) == 1:
            data = list(data.values())[0]
        elif fmt == "tuple":
            data = tuple(data[k] for k in order)
        return data

    def _extract(
        self,
        data: DatapointType,
        data_type: Union[Literal["x"], Literal["y"], Literal["xy"]],
        key: Optional[str] = None,
    ):
        """Helper function to extract data from the return format of this class"""
        if self._hide_targets:
            return data
        x, y = data
        ret = x if data_type == "x" else y
        if not isinstance(ret, dict) or key is None:
            return ret
        return ret[key]

    def as_array(self, data_type: Union[Literal["x"], Literal["y"], Literal["xy"]]):
        """
        Scikit-learn style access to the targets and inputs.
        If the dataset is multi-target, this will return a dict of arrays.
        """

        if data_type == "xy":
            return self.as_array("x"), self.as_array("y")

        if data_type == "y" and self._hide_targets:
            raise TestAccessError("Within Polaris, you should not need to access the targets of the test set")

        if not self.is_multi_task:
            return np.array([self._extract(ret, data_type) for ret in self])

        out = {}
        columns = self.input_cols if data_type == "x" else self.target_cols

        # Temporarily change the target format for easier conversion
        with tmp_attribute_change(self, "_target_format", "dict"):
            with tmp_attribute_change(self, "_input_format", "dict"):
                for k in columns:
                    out[k] = np.array([self._extract(ret, data_type, k) for ret in self])

        return self._convert(out, self.target_cols, self._target_format)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item) -> DatapointType:
        """
        This method always returns an (X, y) tuple

        Some special cases:

        1. It supports multi-input datasets, in which case X is a dict with multiple values.
        2. It supports multi-task datasets, in which case y is a dict with multiple values.
        3. In case a dataset has a pointer column (i.e. a path to an external file with the actual data),
            this method also loads that data to memory.
        """

        idx = self.indices[item]

        # Get the row from the base table
        row = self.dataset.table.iloc[idx]

        # Load the input modalities
        ins = {col: self.dataset.get_data(row.name, col) for col in self.input_cols}
        ins = self._convert(ins, self.input_cols, self._input_format)

        if self._hide_targets:
            # If we are not allowed to access the targets, we return the inputs only.
            # This is useful to make accidental access to the test set less likely.
            return ins

        # Retrieve the targets
        outs = {col: self.dataset.get_data(row.name, col) for col in self.target_cols}
        outs = self._convert(outs, self.target_cols, self._target_format)

        return ins, outs

    def __iter__(self):
        """Iterator implementation"""
        self._pointer = 0
        return self

    def __next__(self):
        """
        Allows for iterator-style access to the dataset, e.g. useful when datasets get large.
        Remember that pointer columns are not loaded into memory yet.
        """
        if self._pointer >= len(self):
            raise StopIteration

        item = self[self._pointer]
        self._pointer += 1
        return item
