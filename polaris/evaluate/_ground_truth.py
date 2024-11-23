import numpy as np
import pandas as pd

from polaris.dataset._subset import Subset


class GroundTruth:
    def __init__(self, test_set: Subset):
        self._test_set = test_set
        self._mask = None

    def _mask_array(self, arr: np.ndarray) -> np.ndarray:
        if self._mask is not None:
            arr = arr[self._mask]
        return arr

    def as_array(self, target_label: str | None = None) -> tuple:
        targets = self._test_set.targets

        if target_label is None:
            return self._mask_array(targets)

        if not self._test_set.is_multi_task:
            targets = {self._test_set.target_cols[0]: targets}
        return self._mask_array(targets[target_label])

    def as_dataframe(self) -> pd.DataFrame:
        # Create an empty dataframe
        cols = self._test_set.input_cols + self._test_set.target_cols
        df = pd.DataFrame(columns=cols)

        # Fill the dataframe
        targets = self._test_set.targets
        if not self._test_set.is_multi_task:
            targets = {self._test_set.target_cols[0]: targets}

        for k in targets:
            df[k] = self._mask_array(targets[k])

        inputs = self._test_set.inputs
        if not self._test_set.is_multi_input:
            inputs = {self._test_set.input_cols: inputs}

        for k in inputs:
            df[k] = self._mask_array(inputs[k])

        return df

    def set_mask(self, mask: pd.Series) -> "GroundTruth":
        self._mask = mask
