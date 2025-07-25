from copy import copy
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Self

from polaris.dataset import DatasetV1
from polaris.dataset._adapters import Adapter
from polaris.utils.errors import TestAccessError
from polaris.utils.types import DatapointType


class Subset:
    """The `Subset` class provides easy access to a single partition of a split dataset.

    Info: No need to create this class manually
        You should not have to create this class manually. In most use-cases, you can create a `Subset` through the
        `get_train_test_split` method of a benchmark object.

    Tip: Featurize your inputs
        Not all datasets are already featurized. For example, a small-molecule task might simply provide the SMILES string.
        To easily featurize the inputs, you can pass or set a transformation function. For example:

        ```python
        import datamol as dm

        benchmark.get_train_test_split(..., featurization_fn=dm.to_fp)
        ```

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

    def __init__(
        self,
        dataset: DatasetV1,
        indices: list[int | Sequence[int]],
        input_cols: Iterable[str] | str,
        target_cols: Iterable[str] | str,
        adapters: dict[str, Adapter] | None = None,
        featurization_fn: Callable | None = None,
        hide_targets: bool = False,
    ):
        self.dataset = dataset
        self.indices = indices
        self.target_cols = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        self.input_cols = [input_cols] if isinstance(input_cols, str) else list(input_cols)

        self._adapters = adapters
        self._featurization_fn = featurization_fn

        # Storing all indices in memory can be memory consuming for XL datasets.
        # This is why we constrain the iloc to loc mapping to be the identity function for Dataset V2.
        match self.dataset:
            case DatasetV1():
                self._iloc_to_loc = lambda idx: self.dataset.rows[idx]
            case _:
                self._iloc_to_loc = lambda idx: idx

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
        """Alias for `self.as_array("x")`"""
        return self.as_array("x")

    @property
    def X(self):
        """Alias for `self.as_array("x")`"""
        return self.as_array("x")

    @property
    def targets(self):
        """Alias for `self.as_array("y")`"""
        return self.as_array("y")

    @property
    def y(self):
        """Alias for `self.as_array("y")`"""
        return self.as_array("y")

    def _get_single(
        self,
        row: str | int,
        cols: list[str],
        featurization_fn: Callable | None,
    ):
        """
        Loads a subset of the variables for a single data-point from the datasets.
        The dataset stores datapoint in a row-wise manner, so this method is used to access a single row.

        Args:
            row: The row index of the datapoint.
            cols: The columns (i.e. variables) to load for that data point.
            featurization_fn: The transformation function to apply to the data-point.
        """
        # Load the data-point
        # Also handles loading data stored in external files for pointer columns
        if len(cols) > 1:
            ret = {col: self.dataset.get_data(row, col, adapters=self._adapters) for col in cols}
        else:
            ret = self.dataset.get_data(row, cols[0], adapters=self._adapters)

        # Featurize
        if featurization_fn is not None:
            ret = featurization_fn(ret)

        return ret

    def _get_single_input(self, row: str | int):
        """Get a single input for a specific data-point and given the benchmark specification."""
        return self._get_single(row, self.input_cols, self._featurization_fn)

    def _get_single_output(self, row: str | int):
        """Get a single output for a specific data-point and given the benchmark specification."""
        return self._get_single(row, self.target_cols, None)

    def as_array(self, data_type: Literal["x", "y", "xy"]):
        """
        Scikit-learn style access to the targets and inputs.
        If the dataset is multi-target, this will return a dict of arrays.
        """

        if data_type == "xy":
            return self.as_array("x"), self.as_array("y")

        if data_type == "y" and self._hide_targets:
            raise TestAccessError("Within Polaris you should not need to access the targets of the test set")

        # We reset the index of the Pandas Table during Dataset class validation.
        # We can thus always assume that .iloc[idx] is the same as .loc[idx].
        if data_type == "x":
            ret = [self._get_single_input(self._iloc_to_loc(idx)) for idx in self.indices]
        else:
            ret = [self._get_single_output(self._iloc_to_loc(idx)) for idx in self.indices]

        if not ((self.is_multi_input and data_type == "x") or (self.is_multi_task and data_type == "y")):
            # If the target format is not a dict, we can just create the array directly.
            # With a single-task or single-input data point, this will be a 1D array.
            # With a multi-task or multi-input data point, this will be a 2D array.
            return np.array(ret)

        # If the return format is a dict, we want to convert
        # from an array of dicts to a dict of arrays.
        if data_type == "y":
            ret = {k: np.array([v[k] for v in ret]) for k in self.target_cols}
        elif data_type == "x":
            ret = {k: np.array([v[k] for v in ret]) for k in self.input_cols}

        return ret

    def as_dataframe(self) -> pd.DataFrame:
        """
        Returns the subset as a Pandas DataFrame.

        Warning: Memory usage
            This method loads the entire dataset in memory.
        """
        # Create an empty dataframe
        cols = self.input_cols
        if not self._hide_targets:
            cols += self.target_cols
        df = pd.DataFrame(columns=cols)

        # Fill the dataframe
        if not self._hide_targets:
            targets = self.targets
            if not self.is_multi_task:
                targets = {self.target_cols[0]: targets}

            for k in targets:
                df[k] = targets[k]

        inputs = self.inputs
        if not self.is_multi_input:
            inputs = {self.input_cols[0]: inputs}

        for k in inputs:
            df[k] = inputs[k]

        return df

    def copy(self) -> Self:
        """Returns a copy of the subset."""
        return copy(self)

    def extend_inputs(self, input_cols: Iterable[str] | str) -> Self:
        """
        Extend the subset to include additional input columns.

        Args:
            input_cols: The input columns to add.
        """
        input_cols = [input_cols] if isinstance(input_cols, str) else list(input_cols)
        copy = self.copy()
        copy.input_cols = list(set(self.input_cols + input_cols))
        return copy

    def filter_targets(self, target_cols: list[str] | str) -> Self:
        """
        Filter the subset to only include the specified target columns.

        Args:
            target_cols: The target columns to keep.
        """
        target_cols_subset = target_cols if isinstance(target_cols, list) else [target_cols]

        # Verify all target columns are in the original subset
        if not all(col in self.target_cols for col in target_cols_subset):
            raise ValueError("All new target columns need to be in the original subset.")
        target_cols_subset = [c for c in self.target_cols if c in target_cols_subset]

        copy = self.copy()
        copy.target_cols = target_cols_subset

        return copy

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
        idx = self._iloc_to_loc(idx)

        # Load the input modalities
        ins = self._get_single_input(idx)

        if self._hide_targets:
            # If we are not allowed to access the targets, we return the inputs only.
            # This is useful to make accidental access to the test set less likely.
            return ins

        # Retrieve the targets
        outs = self._get_single_output(idx)
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
