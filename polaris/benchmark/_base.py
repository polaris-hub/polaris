import json

import numpy as np
import yaml
import fsspec

from hashlib import md5
from pydantic import (
    BaseModel,
    FieldValidationInfo,
    field_validator,
    model_validator,
    computed_field,
    field_serializer,
)
from typing import Union, List, Dict, Tuple, Optional, Any

from polaris.dataset import Dataset, Subset
from polaris.evaluate import Metric, BenchmarkResults
from polaris.utils import fs
from polaris.utils.context import tmp_attribute_change
from polaris.utils.errors import PolarisChecksumError
from polaris.utils.misc import listit
from polaris.utils.types import PredictionsType, SplitType
from polaris.utils.dict2html import dict2html


class BenchmarkSpecification(BaseModel):
    """
    This class contains all information needed to produce ML-ready datasets.

    Specifically, it:
      1) What dataset(s) to use;
      2) Specifies which columns are used as input and which columns are used as target;
      3) Which metrics should be used to evaluate performance on this task;
      4) A predefined, static train-test split to use during evaluation.
    """

    """
    A benchmark specification is based on a dataset
    """
    dataset: Union[Dataset, str, Dict[str, Any]]

    """
    The columns of the original dataset that should be used as targets.
    """
    target_cols: Union[List[str], str]

    """
    The columns of the original dataset that should be used as inputs.
    """
    input_cols: Union[List[str], str]

    """
    The predefined train-test split to use for evaluation.
    """
    split: SplitType

    """
    The metrics to use for evaluating performance
    """
    metrics: Union[Union[Metric, str], List[Union[Metric, str]]]

    """
    The checksum is used to verify the version of the benchmark specification.
    """
    md5sum: Optional[str] = None

    """
    The main metric is the first on the `metrics` field.
    """
    main_metric: Optional[Union[str, Metric]] = None

    @computed_field
    @property
    def polaris_hub_url(self) -> Optional[str]:
        """
        The benchmark URL on the Polaris Hub.
        """
        # NOTE(hadim): putting as default here but we could make it optional
        return "https://polaris.io/benchmark/ORG_OR_USER/BENCHMARK_NAME?"

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @field_validator("dataset")
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

    @field_validator("target_cols", "input_cols")
    def validate_cols(cls, v, info: FieldValidationInfo):
        """Verifies all columns are present in the dataset."""
        if not isinstance(v, List):
            v = [v]
        if len(v) == 0:
            raise ValueError("Specify at least a single column")
        if info.data.get("dataset") is not None and not all(
            c in info.data["dataset"].table.columns for c in v
        ):
            raise ValueError("Not all specified target columns were found in the dataset.")
        return v

    @field_validator("metrics")
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

        if len(v) == 0:
            raise ValueError("Specify at least one metric")

        return v

    @field_validator("split")
    def validate_split(cls, v, info: FieldValidationInfo):
        """
        Verifies that:
          1) There is at least two, non-empty partitions
          2) All indices are valid given the dataset
          3) There is no duplicate indices in any of the sets
          3) There is no overlap between the train and test set
        """

        # There is at least two, non-empty partitions
        if (
            len(v[0]) == 0
            or (isinstance(v[1], dict) and any(len(v) == 0 for v in v[1].values()))
            or (not isinstance(v[1], dict) and len(v[1]) == 0)
        ):
            raise ValueError("The predefined split contains empty partitions")

        train_indices = v[0]
        test_indices = [i for part in v[1].values() for i in part] if isinstance(v[1], dict) else v[1]

        # The train and test indices do not overlap
        if any(i in train_indices for i in test_indices):
            raise ValueError("The predefined split specifies overlapping train and test sets")

        # Duplicate indices
        if len(set(train_indices)) != len(train_indices):
            raise ValueError("The training set contains duplicate indices")
        if len(set(test_indices)) != len(test_indices):
            raise ValueError("The test set contains duplicate indices")

        # All indices are valid given the dataset
        if info.data["dataset"] is not None:
            if any(i < 0 or i >= len(info.data["dataset"]) for i in train_indices + test_indices):
                raise ValueError("The predefined split contains invalid indices")
        return v

    @model_validator(mode="after")
    @classmethod
    def validate_checksum(cls, m: "BenchmarkSpecification"):
        """
        If a checksum is provided, verify it matches what the checksum should be.
        If no checksum is provided, make sure it is set.
        Also sets a default metric if missing.
        """

        # Validate checksum
        checksum = m.md5sum

        expected = cls._compute_checksum(
            dataset=m.dataset,
            target_cols=m.target_cols,
            input_cols=m.input_cols,
            split=m.split,
            metrics=m.metrics,
        )

        if checksum is None:
            m.md5sum = expected
        elif checksum != expected:
            raise PolarisChecksumError(
                "The dataset checksum does not match what was specified in the meta-data. "
                f"{checksum} != {expected}"
            )

        # Set a default main metric if not set yet
        if m.main_metric is None:
            m.main_metric = m.metrics[0]

        return m

    @field_serializer("metrics", "main_metric")
    def serialize_metrics(self, v):
        """Return the string identifier so we can serialize the object"""
        if isinstance(v, Metric):
            return v.name
        return [m.name for m in v]

    @field_serializer("split")
    def serialize_split(self, v):
        """Convert any tuple to list to make sure it's serializable"""
        return listit(v)

    @classmethod
    def from_yaml(cls, path):
        """Loads a benchmark from a yaml file."""
        with fsspec.open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @staticmethod
    def _compute_checksum(dataset, target_cols, input_cols, split, metrics):
        """
        Computes a hash of the benchmark.

        This is meant to uniquely identify the benchmark and can be used to verify the version.
        """

        hash_fn = md5()
        hash_fn.update(dataset.md5sum.encode("utf-8"))
        for c in sorted(target_cols):
            hash_fn.update(c.encode("utf-8"))
        for c in sorted(input_cols):
            hash_fn.update(c.encode("utf-8"))
        for m in sorted(metrics, key=lambda k: k.name):
            hash_fn.update(m.name.encode("utf-8"))

        if not isinstance(split[1], dict):
            split = split[0], {"test": split[1]}

        # Train set
        s = json.dumps(sorted(split[0]))
        hash_fn.update(s.encode("utf-8"))

        # Test sets
        for k in sorted(split[1].keys()):
            s = json.dumps(sorted(split[1][k]))
            hash_fn.update(k.encode("utf-8"))
            hash_fn.update(s.encode("utf-8"))

        checksum = hash_fn.hexdigest()
        return checksum

    def __eq__(self, other):
        if not isinstance(other, BenchmarkSpecification):
            return False
        return self.md5sum == other.md5sum

    def to_yaml(self, destination: str):
        """Saves the benchmark to a yaml file."""

        fs.mkdir(destination, exist_ok=True)

        data = self.model_dump()
        data["dataset"] = self.dataset.to_yaml(destination=destination)

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

    def evaluate(self, y_pred: PredictionsType) -> BenchmarkResults:
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

        if not isinstance(test, dict):
            test = {"test": test}

        y_true = {}
        for k, test_subset in test.items():
            with tmp_attribute_change(test_subset, "_hide_targets", False):
                y_true[k] = test_subset.targets

        if not isinstance(y_pred, dict) or all(k in self.target_cols for k in y_pred):
            y_pred = {"test": y_pred}
        if any(k not in y_pred for k in test.keys()):
            raise KeyError(
                f"Missing keys for at least one of the test sets. Expecting: {sorted(test.keys())}"
            )

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

        return BenchmarkResults(results=scores, benchmark_id=self.md5sum)

    def _repr_dict_(self) -> dict:
        repr_dict = self.model_dump()

        repr_dict.pop("dataset")
        repr_dict.pop("split")

        repr_dict["dataset_name"] = self.dataset.name

        # Make them properties?
        repr_dict["n_input_cols"] = len(self.input_cols)
        repr_dict["n_target_cols"] = len(self.target_cols)

        # TODO(hadim): remove once @compute_field is available
        repr_dict["polaris_hub_url"] = self.polaris_hub_url

        return repr_dict

    def __repr__(self):
        return json.dumps(self._repr_dict_(), indent=2)

    def _repr_html_(self):
        return dict2html(self._repr_dict_())

    def __str__(self):
        return self.__repr__()
