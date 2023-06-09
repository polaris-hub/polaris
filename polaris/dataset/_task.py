import numpy as np
from typing import Union, List, Sequence, Tuple, Any
from polaris.dataset import Dataset


class Task:
    """
    A task is an ML-ready dataset.
    This is the starting point for any framework-specific Dataset implementation.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: List[Union[int, Sequence[int]]],
        input_cols: Union[List[str], str],
        target_cols: Union[List[str], str],
    ):
        self.dataset = dataset
        self.indices = indices
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
        self.input_cols = input_cols if isinstance(input_cols, list) else [input_cols]

        # For the iterator implementation
        self._pointer = 0

    @property
    def is_multi_task(self):
        return len(self.target_cols) > 1

    @property
    def is_multi_modal(self):
        return len(self.input_cols) > 1

    @property
    def inputs(self):
        return np.array([x for x, y in self])

    @property
    def targets(self):
        return np.array([y for x, y in self])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item) -> Tuple[Union[Any, Tuple[Any]], Union[Any, Tuple[Any]]]:
        """
        This method always returns a (X, y) tuple

        Some special cases:
            1. It supports multi-modal datasets, in which case X is a tuple of values.
               NOTE (cwognum): We currently do not support splits across modalities
            2. It supports multi-task datasets, in which case y is a tuple of values.
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
        outs = tuple([row[c] if c in target_idx else np.nan for c in self.target_cols])

        # Load the input modalities
        ins = tuple([self.dataset.get_data(row.name, col) for col in self.input_cols])

        if len(ins) == 1:
            ins = ins[0]
        if len(outs) == 1:
            outs = outs[0]
        return ins, outs

    def __iter__(self):
        self._pointer = 0
        return self

    def __next__(self):
        if self._pointer >= len(self):
            raise StopIteration

        item = self[self._pointer]
        self._pointer += 1
        return item
