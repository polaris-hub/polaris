import numpy as np
from typing import Union, List, Sequence, Tuple, Any, Dict
from polaris.data import Dataset


class Subset:
    """
    A subset is an ML-ready dataset, corresponding to a single partition of a split dataset.
    This is the starting point for any framework-specific Dataset implementation and therefore
    implements various ways to access the data. Since the logic can be complex, this class ensures
    everyone using our benchmarks uses the exact same datasets.
    """

    _SUPPORTED_FORMATS = ["dict", "tuple"]

    def __init__(
        self,
        dataset: Dataset,
        indices: List[Union[int, Sequence[int]]],
        input_cols: Union[List[str], str],
        target_cols: Union[List[str], str],
        input_format: str = "dict",
        target_format: str = "tuple",
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
        if not self.is_multi_input:
            return np.array([x for x, y in self])

        # Temporarily set it to dict
        original = self._input_format
        self._input_format = "dict"

        out = {}
        for k in self.input_cols:
            out[k] = np.array([x[k] for x, y in self])

        # Revert format
        self._input_format = original
        return self._convert(out, self.input_cols, self._input_format)

    @property
    def targets(self):
        """
        Scikit-learn style access to the targets.
        If the dataset is multi-target, this will return a dict of arrays.
        """
        if not self.is_multi_task:
            return np.array([y for x, y in self])

        # Temporarily set it to dict
        original = self._target_format
        self._target_format = "dict"

        out = {}
        for k in self.input_cols:
            out[k] = np.array([y[k] for x, y in self])

        # Revert format
        self._target_format = original
        return self._convert(out, self.target_cols, self._target_format)

    @staticmethod
    def _convert(data: dict, order: List[str], fmt: str):
        """Converts from the default dict format to the specified format"""
        if len(data) == 1:
            data = list(data.values())[0]
        elif fmt == "tuple":
            data = tuple(data[k] for k in order)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(
        self, item
    ) -> Tuple[Union[Any, Tuple, Dict[str, Any]], Union[Any, Tuple, Dict[str, Any]]]:
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
        row_idx = idx[0] if self.is_multi_task else idx
        row = self.dataset.table.iloc[row_idx]

        # Retrieve the targets
        target_idx = self.target_cols
        if self.is_multi_task:
            target_idx = [target_idx[i] for i in idx[1]]

        # If in a multi-task setting a target is missing due to indexing, we return the np NaN.
        outs = {c: row[c] if c in target_idx else np.nan for c in self.target_cols}

        # Load the input modalities
        # NOTE (cwognum): We currently do not support splits across inputs, so we do not need to do any indexing.
        ins = {col: self.dataset.get_data(row.name, col) for col in self.input_cols}

        # Convert to the right format
        ins = self._convert(ins, self.input_cols, self._input_format)
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
