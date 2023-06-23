import numpy as np
import yaml
import fsspec

from hashlib import md5
from pydantic import BaseModel, validator, root_validator
from typing import Union, List, Dict, Tuple, Optional, Any

from polaris.data import Dataset, Subset
from polaris.evaluate import Metric, BenchmarkResults
from polaris.utils import fs
from polaris.utils.context import tmp_attribute_change
from polaris.utils.errors import PolarisChecksumError
from polaris.utils.types import Predictions, Split


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
    split: Split
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
        if not isinstance(v, List):
            v = [v]
        if len(v) == 0:
            raise ValueError("Specify at least a single column")
        if "dataset" in values and not all(c in values["dataset"].table.columns for c in v):
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
    def validate_split(cls, v):
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

    @root_validator(skip_on_failure=True)
    def validate_checksum(cls, values):
        """
        If a checksum is provided, verify it matches what the checksum should be.
        If no checksum is provided, make sure it is set.
        """

        # Skip validation as some attributes are missing
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

        fs.mkdir(destination, exist_ok=True)

        data = self.dict()
        data["dataset"] = self.dataset.save(destination=destination)
        data["metrics"] = [m.name for m in self.metrics]
        data["split"] = [list(v) if isinstance(v, tuple) else v for v in self.split]

        path = fs.join(destination, "benchmark.yaml")
        with fsspec.open(path, "w") as f:
            yaml.dump(data, f)

        return path

    def get_no_test_sets(self):
        """The number of test sets."""
        return len(self.split[1]) if isinstance(self.split[1], dict) else 1

    def get_train_test_split(
        self, input_format: str = "dict", target_format: str = "dict"
    ) -> Tuple[Subset, Union[Subset, Dict[str, Subset]]]:
        """Endpoint for returning the train and test sets."""

        def _get_subset(indices, hide_targets):
            return Subset(
                dataset=self.dataset,
                indices=indices,
                input_cols=self.input_cols,
                input_format=input_format,
                target_cols=self.target_cols,
                target_format=target_format,
                hide_targets=hide_targets,
            )

        train = _get_subset(self.split[0], hide_targets=False)
        if isinstance(self.split[1], dict):
            test = {k: _get_subset(v, hide_targets=True) for k, v in self.split[1].items()}
        else:
            test = _get_subset(self.split[1], hide_targets=True)
        return train, test

    def evaluate(self, y_pred: Predictions) -> BenchmarkResults:
        """
        Execute the evaluation protocol for the benchmark.

        We make the following assumptions:
            (1) There can be one or multiple test sets
            (2) There can be one or multiple targets
            (3) The metrics are constant across test sets
            (4) The metrics are constant across targets
            (5) There can be metrics which measure across tasks.
        """

        # Instead of having the user pass the ground truth, we extract it from the benchmark spec ourselves.
        # This simplifies the API, but also was added to make accidental access to the test set targets less likely.
        # See also the `hide_targets` parameter in the `Subset` class.
        test = self.get_train_test_split()[1]

        with tmp_attribute_change(test, "hide_targets", False):
            if isinstance(test, dict):
                y_true = {k: v.targets for k, v in test.items()}
            else:
                y_true = {"test": test.targets}

        if not isinstance(y_pred, dict) or all(isinstance(v, np.ndarray) for v in y_pred.values()):
            y_pred = {"test": y_pred}

        scores = {}

        # For every test set...
        for test_label, y_true_subset in y_true.items():
            scores[test_label] = {}

            # For every metric...
            for metric in self.metrics:
                if metric.is_multitask or not isinstance(y_true_subset, dict):
                    # Either single-task or multi-task but with a metric across targets
                    scores[test_label][metric.name] = metric(y_true=y_true_subset, y_pred=y_pred[test_label])
                    continue

                # Otherwise, for every target...
                for target_label, y_true_target in y_true_subset.items():
                    if target_label not in scores[test_label]:
                        scores[test_label][target_label] = {}

                    # Single-task metrics for a multi-task benchmark
                    # In such a setting, there can be NaN values, which we thus have to filter out.
                    mask = ~np.isnan(y_true_target)
                    score = metric(y_true=y_true_target[mask], y_pred=y_pred[test_label][target_label][mask])
                    scores[test_label][target_label][metric.name] = score

        if len(scores) == 1:
            scores = scores["test"]

        return BenchmarkResults(results=scores, benchmark_id=self.checksum)
