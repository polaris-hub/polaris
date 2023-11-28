import json
import os
from hashlib import md5
from typing import Any, Optional, Union

import fsspec
import numpy as np
import pandas as pd
from pydantic import (
    FieldValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from polaris._artifact import BaseArtifactModel
from polaris.dataset import Dataset, Subset
from polaris.evaluate import BenchmarkResults, Metric, ResultsType
from polaris.hub.settings import PolarisHubSettings
from polaris.utils import fs
from polaris.utils.context import tmp_attribute_change
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidBenchmarkError, PolarisChecksumError
from polaris.utils.misc import listit
from polaris.utils.types import AccessType, DataFormat, HubOwner, PredictionsType, SplitType

ColumnsType = Union[str, list[str]]


class BenchmarkSpecification(BaseArtifactModel):
    """This class wraps a [`Dataset`][polaris.dataset.Dataset] with additional data
     to specify the evaluation logic.

    Specifically, it specifies:

    1. Which dataset to use (see [`Dataset`][polaris.dataset.Dataset]);
    2. Which columns are used as input and which columns are used as target;
    3. Which metrics should be used to evaluate performance on this task;
    4. A predefined, static train-test split to use during evaluation.

    info: Subclasses
        Polaris includes various subclasses of the `BenchmarkSpecification` that provide a more precise data-model or
         additional logic, e.g. [`SingleTaskBenchmarkSpecification`][polaris.benchmark.SingleTaskBenchmarkSpecification].

    Examples:
        Basic API usage:
        ```python
        import polaris as po

        benchmark = po.load_benchmark("/path/to/benchmark")
        train, test = benchmark.get_train_test_split()

        # Work your magic
        predictions = ...

        benchmark.evaluate(predictions)
        ```

    Attributes:
        dataset: The dataset the benchmark specification is based on.
        target_cols: The column(s) of the original dataset that should be used as target.
        input_cols: The column(s) of the original dataset that should be used as input.
        split: The predefined train-test split to use for evaluation.
        metrics: The metrics to use for evaluating performance
        main_metric: The main metric used to rank methods. If `None`, the first of the `metrics` field.
        md5sum: The checksum is used to verify the version of the dataset specification. If specified, it will
            raise an error if the specified checksum doesn't match the computed checksum.
        readme: Markdown text that can be used to provide a formatted description of the benchmark.
            If using the Polaris Hub, it is worth noting that this field is more easily edited through the Hub UI
            as it provides a rich text editor for writing markdown.
    For additional meta-data attributes, see the [`BaseArtifactModel`][polaris._artifact.BaseArtifactModel] class.
    """

    # Public attributes
    # Data
    dataset: Union[Dataset, str, dict[str, Any]]
    target_cols: ColumnsType
    input_cols: ColumnsType
    split: SplitType
    metrics: Union[str, Metric, list[Union[str, Metric]]]
    main_metric: Optional[Union[str, Metric]] = None
    md5sum: Optional[str] = None

    # Additional meta-data
    readme: str = ""

    @field_validator("dataset")
    def _validate_dataset(cls, v):
        """
        Allows either passing a Dataset object or the kwargs to create one
        TODO (cwognum): Allow multiple datasets to be used as part of a benchmark
        """
        if isinstance(v, dict):
            v = Dataset(**v)
        elif isinstance(v, str):
            v = Dataset.from_json(v)
        return v

    @field_validator("target_cols", "input_cols")
    def _validate_cols(cls, v, info: FieldValidationInfo):
        """Verifies all columns are present in the dataset."""
        if not isinstance(v, list):
            v = [v]
        if len(v) == 0:
            raise InvalidBenchmarkError("Specify at least a single column")
        if info.data.get("dataset") is not None and not all(
            c in info.data["dataset"].table.columns for c in v
        ):
            raise InvalidBenchmarkError("Not all specified target columns were found in the dataset.")
        return v

    @field_validator("metrics")
    def _validate_metrics(cls, v):
        """
        Verifies all specified metrics are either a Metric object or a valid metric name.
        Also verifies there are no duplicate metrics.

        If there are multiple test sets, it is assumed the same metrics are used across test sets.
        """
        if not isinstance(v, list):
            v = [v]

        v = [m if isinstance(m, Metric) else Metric[m] for m in v]

        if len(set(v)) != len(v):
            raise InvalidBenchmarkError("The task specifies duplicate metrics")

        if len(v) == 0:
            raise InvalidBenchmarkError("Specify at least one metric")

        return v

    @field_validator("main_metric")
    def _validate_main_metric(cls, v):
        """Converts the main metric to a Metric object if it is a string."""
        # lu: v can be None.
        if v and not isinstance(v, Metric):
            v = Metric[v]
        return v

    @field_validator("split")
    def _validate_split(cls, v, info: FieldValidationInfo):
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
            raise InvalidBenchmarkError("The predefined split contains empty partitions")

        train_indices = v[0]
        test_indices = [i for part in v[1].values() for i in part] if isinstance(v[1], dict) else v[1]

        # The train and test indices do not overlap
        if any(i in train_indices for i in test_indices):
            raise InvalidBenchmarkError("The predefined split specifies overlapping train and test sets")

        # Duplicate indices
        if len(set(train_indices)) != len(train_indices):
            raise InvalidBenchmarkError("The training set contains duplicate indices")
        if len(set(test_indices)) != len(test_indices):
            raise InvalidBenchmarkError("The test set contains duplicate indices")

        # All indices are valid given the dataset
        if info.data["dataset"] is not None:
            if any(i < 0 or i >= len(info.data["dataset"]) for i in train_indices + test_indices):
                raise InvalidBenchmarkError("The predefined split contains invalid indices")
        return v

    @model_validator(mode="after")
    @classmethod
    def _validate_checksum(cls, m: "BenchmarkSpecification"):
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
    def _serialize_metrics(self, v):
        """Return the string identifier so we can serialize the object"""
        if isinstance(v, Metric):
            return v.name
        return [m.name for m in v]

    @field_serializer("split")
    def _serialize_split(self, v):
        """Convert any tuple to list to make sure it's serializable"""
        return listit(v)

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

    def get_train_test_split(
        self, input_format: DataFormat = "dict", target_format: DataFormat = "dict"
    ) -> tuple[Subset, Union["Subset", dict[str, Subset]]]:
        """Construct the train and test sets, given the split in the benchmark specification.

        Returns [`Subset`][polaris.dataset.Subset] objects, which offer several ways of accessing the data
        and can thus easily serve as a basis to build framework-specific (e.g. PyTorch, Tensorflow)
        data-loaders on top of.

        Args:
            input_format: How the input data is returned from the `Subset` object.
            target_format: How the target data is returned from the `Subset` object.
                This will only affect the train set.

        Returns:
            A tuple with the train `Subset` and test `Subset` objects.
                If there are multiple test sets, these are returned in a dictionary and each test set has
                an associated name. The targets of the test set can not be accessed.
        """

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
        """Execute the evaluation protocol for the benchmark, given a set of predictions.

        info: What about `y_true`?
            Contrary to other frameworks that you might be familiar with, we opted for a signature that includes just
            the predictions. This reduces the chance of accidentally using the test targets during training.

        For this method, we make the following assumptions:

        1. There can be one or multiple test set(s);
        2. There can be one or multiple target(s);
        3. The metrics are _constant_ across test sets;
        4. The metrics are _constant_ across targets;
        5. There can be metrics which measure across tasks.

        Args:
            y_pred: The predictions for the test set, as NumPy arrays. If there are multiple test sets,
                this should be a dictionary with the test set names as keys.

        Returns:
            A `BenchmarkResults` object. This object can be directly submitted to the Polaris Hub.
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

        # Results are saved in a tabular format. For more info, see the BenchmarkResults docs.
        scores: ResultsType = pd.DataFrame(columns=BenchmarkResults.RESULTS_COLUMNS)

        # For every test set...
        for test_label, y_true_subset in y_true.items():
            # For every metric...
            for metric in self.metrics:
                if metric.is_multitask:
                    # Multi-task but with a metric across targets
                    score = metric(y_true=y_true_subset, y_pred=y_pred[test_label])
                    scores.loc[len(scores)] = (test_label, "all", metric, score)
                    continue

                if not isinstance(y_true_subset, dict):
                    # Single task
                    score = metric(y_true=y_true_subset, y_pred=y_pred[test_label])
                    scores.loc[len(scores)] = (test_label, self.target_cols[0], metric, score)
                    continue

                # Otherwise, for every target...
                for target_label, y_true_target in y_true_subset.items():
                    # Single-task metrics for a multi-task benchmark
                    # In such a setting, there can be NaN values, which we thus have to filter out.
                    mask = ~np.isnan(y_true_target)
                    score = metric(y_true=y_true_target[mask], y_pred=y_pred[test_label][target_label][mask])
                    scores.loc[len(scores)] = (test_label, target_label, metric, score)

        return BenchmarkResults(results=scores, benchmark_name=self.name, benchmark_owner=self.owner)

    def upload_to_hub(
        self,
        env_file: Optional[Union[str, os.PathLike]] = None,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        access: Optional[AccessType] = "private",
        owner: Optional[Union[HubOwner, str]] = None,
        **kwargs: dict,
    ):
        """
        Very light, convenient wrapper around the
        [`PolarisHubClient.upload_benchmark`][polaris.hub.client.PolarisHubClient.upload_benchmark] method.
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(
            env_file=env_file, settings=settings, cache_auth_token=cache_auth_token, **kwargs
        ) as client:
            return client.upload_benchmark(self, access=access, owner=owner)

    def to_json(self, destination: str) -> str:
        """Save the benchmark to a destination directory as a JSON file.

        Warning: Multiple files
            Perhaps unintuitive, this method creates multiple files in the destination directory as it also saves
            the dataset it is based on to the specified destination.
            See the docstring of [`Dataset.to_json`][polaris.dataset.Dataset.to_json] for more information.

        Args:
            destination: The _directory_ to save the associated data to.

        Returns:
            The path to the JSON file.
        """

        fs.mkdir(destination, exist_ok=True)

        data = self.model_dump()
        data["dataset"] = self.dataset.to_json(destination=destination)

        path = fs.join(destination, "benchmark.json")
        with fsspec.open(path, "w") as f:
            json.dump(data, f)

        return path

    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        repr_dict = self.model_dump()
        repr_dict.pop("dataset")
        repr_dict.pop("split")
        repr_dict["dataset_name"] = self.dataset.name
        repr_dict["n_input_cols"] = len(self.input_cols)
        repr_dict["n_target_cols"] = len(self.target_cols)
        return repr_dict

    def _repr_html_(self):
        """For pretty printing in Jupyter."""
        return dict2html(self._repr_dict_())

    def __repr__(self):
        return json.dumps(self._repr_dict_(), indent=2)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, BenchmarkSpecification):
            return False
        return self.md5sum == other.md5sum
