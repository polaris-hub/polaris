import numpy as np
from typing import Union, List, Tuple
from polaris.dataset import Dataset


class Task:
    """
    A task is an ML-ready dataset.
    This is the starting point for any framework-specific Dataset implementation.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: List[Union[int, Tuple[int, int]]],
        input_cols: Union[List[str], str],
        target_cols: Union[List[str], str],
    ):
        self.dataset = dataset
        self.indices = np.array(indices)
        self.target_cols = target_cols
        self.input_cols = input_cols

        # For the iterator implementation
        self._pointer = 0

    @property
    def is_multi_task(self):
        return len(self.target_cols) > 1

    @property
    def is_multi_modal(self):
        return len(self.input_cols) > 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        """
        This method always returns a (X, y) tuple

        Some special cases:
            1. It supports multi-modal datasets, in which case X is a sequence of values.
               NOTE (cwognum): We currently do not allow splits across modalities.
            2. It supports multi-task datasets, in which case y is a sequence of values.
        """

        idx = self.indices[item]

        row_idx = idx[0] if self.is_multi_task else idx
        row = self.dataset.table.iloc[row_idx]

        target_idx = self.target_cols
        if self.is_multi_task:
            target_idx = target_idx[idx[1]]

        ins = row[self.input_cols].values
        outs = row[target_idx].values

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
