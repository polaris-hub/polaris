from typing import Union, List, Dict, Tuple, Optional

import numpy as np

from polaris.dataset import Dataset, Modality, Task
from polaris.evaluate import Metric
from polaris.utils.errors import InvalidBenchmarkError


# A split is defined by a sequence of integers (single-task) or by a sequence of integer pairs (multi-task)
_SPLIT_PARTITION_TYPE = List[Union[int, Tuple[int, Union[List[int], int]]]]

# A split is a pair of which the first item is always assumed to be the train set.
# The second item can either be a single test set or a dictionary with multiple, named test sets.
_SPLIT_TYPE_HINT = Tuple[
    _SPLIT_PARTITION_TYPE, Union[_SPLIT_PARTITION_TYPE, Dict[str, _SPLIT_PARTITION_TYPE]]
]


class Benchmark:
    """
    A benchmark wraps a dataset with additional information to form a machine learning task.

    Specifically, a Task:
      1) Specifies which columns are used as input and which columns are used as target;
      2) Which metrics should be used to evaluate performance on this task;
      3) A predefined, static train-test split to use during evaluation.
    """

    def __init__(
        self,
        dataset: Dataset,
        target_cols: Union[List[str], str],
        input_cols: Union[List[str], str],
        split: _SPLIT_TYPE_HINT,
        metrics: Union[Union[Metric, str], List[Union[Metric, str]]],
        version: Optional[str] = None,
    ):
        self.dataset = dataset
        self.target_cols = self._verify_cols(target_cols)
        self.input_cols = self._verify_cols(input_cols)
        self._split = self._verify_split(split)
        self._metrics = self._verify_metrics(metrics)
        self._version = version

    def get_no_tasks(self):
        """The number of tasks."""
        return len(self.target_cols)

    def is_multi_task(self):
        """Whether a dataset has multiple targets."""
        return self.get_no_tasks() > 1

    def get_no_modalities(self):
        """The number of modalities."""
        return len(self.input_cols)

    def is_multi_modal(self):
        """Whether a dataset has multiple modalities."""
        return self.get_no_modalities() > 1

    def get_no_test_sets(self):
        """The number of test sets."""
        return len(self._split) if isinstance(self._split[1], dict) else 1

    def prepare(self):
        pass

    def get_train_test_split(self) -> Tuple[Task, Union[Task, Dict[str, Task]]]:
        """
        NOTE (cwognum):
            - For some modalities (e.g. images, 3D structures), we should cache the additionally needed files
            - Should we filter out NaN values? E.g. needed when a task wraps a sparse multi-task dataset.
            - What about invalid inputs (e.g. invalid SMILES)? Maybe possible with sparse, multi-modal tasks?
        """
        train = Task(self.dataset, self._split[0], self.input_cols, self.target_cols)
        if isinstance(self._split[1], dict):
            test = {
                k: Task(self.dataset, v, self.input_cols, self.target_cols) for k, v in self._split[1].items()
            }
        else:
            test = Task(self.dataset, self._split[1], self.input_cols, self.target_cols)
        return train, test

    def evaluate(
        self,
        y_pred: Union[np.ndarray, Dict[str, np.ndarray]],
        y_true: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ):
        """Evaluate the performance given a set of predictions"""
        ...

    def _verify_cols(self, cols: List[str]):
        """Verifies all columns are present in the dataset."""
        if not isinstance(cols, List):
            cols = [cols]
        if not all(c in self.dataset.table.columns for c in cols):
            raise InvalidBenchmarkError(f"Not all specified target columns were found in the dataset.")
        return cols

    def _verify_metrics(self, metrics: List[Union[Metric, str]]):  # noqa
        """
        Verifies all specified metrics are either a Metric object or a valid metric name.
        Also verifies there are no duplicate metrics.

        If there are multiple test sets, it is assumed the same metrics are used across test sets
        """
        if not isinstance(metrics, List):
            metrics = [metrics]
        metrics = [m if isinstance(m, Metric) else Metric.get_by_name(m) for m in metrics]
        if len(set(m.name for m in metrics)) != len(metrics):
            raise InvalidBenchmarkError("The task specifies duplicate metrics")
        return metrics

    def _verify_split(self, split: _SPLIT_TYPE_HINT):
        """
        Verifies there is at least two, non-empty partitions and
        that there is no overlap between the various partitions.

        NOTE (cwognum): This assumes that for multi-modal dataset, the split is not _across_ modalities.
          In other words, if one modality is in a subset, all of them are. The split is the same per modality.
        """
        if len(split) != 2 or isinstance(split[1], dict) and len(split[1]) == 0:
            raise InvalidBenchmarkError("The predefined split should specify at least 2 partitions")
        if (
            len(split[0]) == 0
            or (isinstance(split[1], dict) and any(len(v) == 0 for v in split[1].values()))
            or (not isinstance(split[1], dict) and len(split[1]) == 0)
        ):
            raise InvalidBenchmarkError("The predefined split contains empty partitions")

        if self.is_multi_task():
            split = self._verify_multi_task_split(split)
        else:
            split = self._verify_single_task_split(split)
        return split

    def _verify_single_task_split(self, split: _SPLIT_TYPE_HINT):
        """
        A valid single task split assigns inputs exclusively to a single partition.
        It is not required that the split covers the entirety of a dataset.
        """
        train_indices = split[0]
        test_indices = (
            [i for part in split[1].values() for i in part] if isinstance(split[1], dict) else split[1]
        )
        if any(i < 0 or i >= len(self.dataset) for i in train_indices + test_indices):
            raise InvalidBenchmarkError("The predefined split contains invalid indices")
        if any(i in train_indices for i in test_indices):
            raise InvalidBenchmarkError("The predefined split specifies overlapping train and test sets")
        return split

    def _verify_multi_task_split(self, split: _SPLIT_TYPE_HINT):
        """
        A valid multitask split assigns each input, target pair exclusively to a single partition
        """
        train_pairs = split[0]
        test_pairs = (
            [i for part in split[1].values() for i in part] if isinstance(split[1], dict) else split[1]
        )
        for tup in train_pairs + test_pairs:
            if (
                len(tup) != 2
                or tup[0] < 0
                or tup[0] >= len(self.dataset)
                or any(i < 0 for i in tup[1])
                or any(i >= self.get_no_tasks() for i in tup[1])
            ):
                raise InvalidBenchmarkError("The predefined split contains invalid indices")
        if any(tup in train_pairs for tup in test_pairs):
            raise InvalidBenchmarkError("The predefined split specifies overlapping train and test sets")
        return split
