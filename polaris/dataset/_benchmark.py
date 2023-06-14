from hashlib import md5

import yaml
import fsspec

import numpy as np

from pydantic import BaseModel, validator, root_validator
from typing import Union, List, Dict, Tuple, Optional, Sequence, Any

from polaris.dataset import Dataset, Subset
from polaris.evaluate import Metric
from polaris.utils import fs
from polaris.utils.errors import PolarisChecksumError

# A split is defined by a sequence of integers (single-task) or by a sequence of integer pairs (multi-task)
_SPLIT_PARTITION_TYPE = List[Union[int, Tuple[int, Union[List[int], int]]]]

# A split is a pair of which the first item is always assumed to be the train set.
# The second item can either be a single test set or a dictionary with multiple, named test sets.
_SPLIT_TYPE_HINT = Tuple[
    _SPLIT_PARTITION_TYPE, Union[_SPLIT_PARTITION_TYPE, Dict[str, _SPLIT_PARTITION_TYPE]]
]


class BenchmarkSpecification(BaseModel):
    """
    This class contains all information needed to produce ML-ready datasets.

    Specifically, it:
      1) What dataset(s) to use;
      2) Specifies which columns are used as input and which columns are used as target;
      3) Which metrics should be used to evaluate performance on this task;
      4) A predefined, static train-test split to use during evaluation.
    """

    dataset: Union[Dataset, str, Dict[str, Any]]
    target_cols: Union[List[str], str]
    input_cols: Union[List[str], str]
    split: _SPLIT_TYPE_HINT
    metrics: Union[Union[Metric, str], List[Union[Metric, str]]]
    checksum: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("dataset")
    def validate_dataset(cls, v):
        """
        Allows either passing a Dataset object or the kwargs to create one
        TODO (cwognum): Allow multiple datasets to be used as part of a benchmark
        """
        if isinstance(v, dict):
            v = Dataset(**v)
        elif isinstance(v, str):
            v = Dataset.from_yaml(v)
        return v

    @validator("target_cols", "input_cols")
    def validate_cols(cls, v, values):
        """Verifies all columns are present in the dataset."""
        # Exit early as a prior validation step failed
        if "dataset" not in values:
            return v
        if not isinstance(v, List):
            v = [v]
        if len(v) == 0:
            raise ValueError("Specify at least a single column")
        if not all(c in values["dataset"].table.columns for c in v):
            raise ValueError(f"Not all specified target columns were found in the dataset.")
        return v

    @validator("metrics")
    def validate_metrics(cls, v):
        """
        Verifies all specified metrics are either a Metric object or a valid metric name.
        Also verifies there are no duplicate metrics.

        If there are multiple test sets, it is assumed the same metrics are used across test sets.
        """
        if not isinstance(v, List):
            v = [v]
        v = [m if isinstance(m, Metric) else Metric.get_by_name(m) for m in v]
        if len(set(m.name for m in v)) != len(v):
            raise ValueError("The task specifies duplicate metrics")
        return v

    @validator("split")
    def verify_split(cls, v):
        """
        Verifies there is at least two, non-empty partitions
        """
        if (
            len(v[0]) == 0
            or (isinstance(v[1], dict) and any(len(v) == 0 for v in v[1].values()))
            or (not isinstance(v[1], dict) and len(v[1]) == 0)
        ):
            raise ValueError("The predefined split contains empty partitions")
        return v

    @root_validator
    def validate_checksum(cls, values):
        """
        If a checksum is provided, verify it matches what the checksum should be.
        If no checksum is provided, make sure it is set.
        """

        # Skip validation as an earlier step has failed
        if not all(k in values for k in ["dataset", "target_cols", "input_cols", "split", "metrics"]):
            return values

        checksum = values["checksum"]

        expected = cls._compute_checksum(
            dataset=values["dataset"],
            target_cols=values["target_cols"],
            input_cols=values["input_cols"],
            split=values["split"],
            metrics=values["metrics"],
        )

        if checksum is None:
            values["checksum"] = expected
        elif checksum != expected:
            raise PolarisChecksumError(
                "The dataset checksum does not match what was specified in the meta-data. "
                f"{checksum} != {expected}"
            )
        return values

    @classmethod
    def from_yaml(cls, path):
        """Loads a benchmark from a yaml file."""
        with fsspec.open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.parse_obj(data)

    @staticmethod
    def _compute_checksum(dataset, target_cols, input_cols, split, metrics):
        """
        Computes a hash of the benchmark.

        This is meant to uniquely identify the benchmark and can be used to verify the version.

        (1) Is not sensitive to the ordering of the columns or metrics
        (2) IS sensitive to the ordering of the splits
        """

        hash_fn = md5()
        hash_fn.update(dataset.checksum.encode("utf-8"))
        for c in sorted(target_cols):
            hash_fn.update(c.encode("utf-8"))
        for c in sorted(input_cols):
            hash_fn.update(c.encode("utf-8"))
        for m in sorted(metrics, key=lambda k: k.name):
            hash_fn.update(m.name.encode("utf-8"))

        # TODO (cwognum): Just encoding this as a string is a MAJOR limitation.
        #  This means that the same split, but represented in a different format or order is considered different.
        #  As the API is still unstable, I currently left it at is.
        hash_fn.update(str(split).encode("utf-8"))

        checksum = hash_fn.hexdigest()
        return checksum

    def __eq__(self, other):
        if not isinstance(other, BenchmarkSpecification):
            return False
        return self.checksum == other.checksum

    def save(self, destination: str):
        """Saves the benchmark to a yaml file."""
        data = self.dict()
        data["dataset"] = self.dataset.save(destination=destination)
        data["metrics"] = [m.name for m in self.metrics]
        data["split"] = [list(v) if isinstance(v, tuple) else v for v in self.split]
        path = fs.join(destination, "benchmark.yaml")
        with fsspec.open(path, "w") as f:
            yaml.dump(data, f)
        return path

    def get_no_tasks(self):
        """The number of tasks."""
        return len(self.target_cols)

    def is_multi_task(self):
        """Whether a dataset has multiple targets."""
        return self.get_no_tasks() > 1

    def get_no_inputs(self):
        """The number of inputs."""
        return len(self.input_cols)

    def is_multi_input(self):
        """Whether a dataset has multiple inputs."""
        return self.get_no_inputs() > 1

    def get_no_test_sets(self):
        """The number of test sets."""
        return len(self.split[1]) if isinstance(self.split[1], dict) else 1

    def get_train_test_split(self) -> Tuple[Subset, Union[Subset, Dict[str, Subset]]]:
        """Return the train and test sets."""
        train = Subset(self.dataset, self.split[0], self.input_cols, self.target_cols)
        if isinstance(self.split[1], dict):
            test = {
                k: Subset(self.dataset, v, self.input_cols, self.target_cols)
                for k, v in self.split[1].items()
            }
        else:
            test = Subset(self.dataset, self.split[1], self.input_cols, self.target_cols)
        return train, test

    def evaluate(
        self,
        y_pred: Union[np.ndarray, Dict[str, np.ndarray]],
        y_true: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ):
        pass


class SingleTaskBenchmarkSpecification(BenchmarkSpecification):
    @validator("split")
    def verify_split_single_task(cls, v, values):
        """
        A valid single task split assigns inputs exclusively to a single partition.
        It is not required that the split covers the entirety of a dataset.
        """

        if values.get("dataset") is None:
            # This feels hacky to me, but if validation for prior fields fail, this method is still called.
            # As the failed fields are not set, this leads to confusing errors.
            return v

        train_indices = v[0]
        test_indices = [i for part in v[1].values() for i in part] if isinstance(v[1], dict) else v[1]
        if any(i < 0 or i >= len(values["dataset"]) for i in train_indices + test_indices):
            raise ValueError("The predefined split contains invalid indices")
        if any(i in train_indices for i in test_indices):
            raise ValueError("The predefined split specifies overlapping train and test sets")
        return v


class MultiTaskBenchmarkSpecification(BenchmarkSpecification):
    @validator("split")
    def verify_split_multi_task(cls, v, values):
        """
        A valid multitask split assigns each input, target pair exclusively to a single partition.
        It is not required that the split covers the entirety of a dataset.
        """

        if values.get("dataset") is None or values.get("target_cols") is None:
            # This feels hacky to me, but if validation for prior fields fail, this method is still called.
            # As the failed fields are not set, this leads to confusing errors.
            return v

        def _check(indices):
            """Helper method to easily check indices for a single partition."""
            checked_indices = []
            for tup in indices:
                if isinstance(tup, Sequence):
                    invalid = (
                        len(tup) != 2
                        or tup[0] < 0
                        or tup[0] >= len(values["dataset"])
                        or any(i < 0 for i in tup[1])
                        or any(i >= len(values["target_cols"]) for i in tup[1])
                    )
                elif isinstance(tup, int):
                    # With single index, we assume all targets are indexed.
                    # This simplifies split definitions, because you can specify X instead of (X, 1), (X, 2), ...
                    # Changing the index here to a consistent format for downstream processing.
                    invalid = tup < 0 or tup >= len(values["dataset"])
                    tup = (tup, list(range(len(values["target_cols"]))))

                else:
                    invalid = True

                if invalid:
                    raise ValueError("The predefined split contains invalid indices")

                checked_indices.append(tup)

            return checked_indices

        # Check the train set
        train_pairs = _check(v[0])

        # Check the test sets and whether any of them overlap with train
        if isinstance(v[1], dict):
            test_pairs = {k: _check(v) for k, v in v[1].items()}
            overlapping = any(tup in train_pairs for subset in test_pairs.values() for tup in subset)
        else:
            test_pairs = _check(v[1])
            overlapping = any(tup in train_pairs for tup in test_pairs)

        if overlapping:
            raise ValueError("The predefined split specifies overlapping train and test sets")

        return train_pairs, test_pairs
