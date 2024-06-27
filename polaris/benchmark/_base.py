import json
from hashlib import md5
from typing import Any, Callable, Optional, Union

import fsspec
import numpy as np
import pandas as pd
from datamol.utils import fs
from pydantic import (
    Field,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from sklearn.utils.multiclass import type_of_target

from polaris._artifact import BaseArtifactModel
from polaris.dataset import Dataset, Subset
from polaris.evaluate import BenchmarkResults, Metric, ResultsType
from polaris.hub.settings import PolarisHubSettings
from polaris.utils.context import tmp_attribute_change
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidBenchmarkError, PolarisChecksumError
from polaris.utils.misc import listit
from polaris.utils.types import (
    AccessType,
    HubOwner,
    PredictionsType,
    SplitType,
    TargetType,
    TaskType,
)

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

        # Load the benchmark from the Hub
        benchmark = po.load_benchmark("polaris/hello-world-benchmark")

        # Get the train and test data-loaders
        train, test = benchmark.get_train_test_split()

        # Use the training data to train your model
        # Get the input as an array with 'train.inputs' and 'train.targets'
        # Or simply iterate over the train object.
        for x, y in train:
            ...

        # Work your magic to accurately predict the test set
        predictions = [0.0 for x in test]

        # Evaluate your predictions
        results = benchmark.evaluate(predictions)

        # Submit your results
        results.upload_to_hub(owner="dummy-user")
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
        target_types: A dictionary that maps target columns to their type. If not specified, this is automatically inferred.
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
    target_types: dict[str, Optional[Union[TargetType, str]]] = Field(
        default_factory=dict, validate_default=True
    )

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
    def _validate_cols(cls, v, info: ValidationInfo):
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
    def _validate_split(cls, v, info: ValidationInfo):
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

    @field_validator("target_types")
    def _validate_target_types(cls, v, info: ValidationInfo):
        """Try to automatically infer the target types if not already set"""

        dataset = info.data.get("dataset")
        target_cols = info.data.get("target_cols")

        if dataset is None or target_cols is None:
            return v

        for target in target_cols:
            if target not in v:
                val = dataset[:, target]
                # remove the nans for mutiple task dataset when the table is sparse
                target_type = type_of_target(val[~np.isnan(val)])
                if target_type == "continuous":
                    v[target] = TargetType.REGRESSION
                elif target_type in ["binary", "multiclass"]:
                    v[target] = TargetType.CLASSIFICATION
                else:
                    v[target] = None
            elif not isinstance(v, TargetType):
                v[target] = TargetType(v[target])
        return v

    @model_validator(mode="after")
    @classmethod
    def _validate_model(cls, m: "BenchmarkSpecification"):
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

    @field_serializer("target_types")
    def _serialize_target_types(self, v):
        """Convert from enum to string to make sure it's serializable"""
        return {k: v.value for k, v in self.target_types.items()}

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

    @computed_field
    @property
    def n_train_datapoints(self) -> int:
        """The size of the train set."""
        return len(self.split[0])

    @computed_field
    @property
    def n_test_sets(self) -> int:
        """The number of test sets"""
        return len(self.split[1]) if isinstance(self.split[1], dict) else 1

    @computed_field
    @property
    def n_test_datapoints(self) -> dict[str, int]:
        """The size of (each of) the test set(s)."""
        if self.n_test_sets == 1:
            return {"test": len(self.split[1])}
        else:
            return {k: len(v) for k, v in self.split[1].items()}

    @computed_field
    @property
    def n_classes(self) -> dict[str, int]:
        """The number of classes for each of the target columns."""
        n_classes = {}
        for target in self.target_cols:
            target_type = self.target_types[target]
            if target_type is None or target_type == TargetType.REGRESSION:
                continue
            n_classes[target] = self.dataset.table.loc[:, target].nunique()
        return n_classes

    @computed_field
    @property
    def task_type(self) -> str:
        """The high-level task type of the benchmark."""
        v = TaskType.MULTI_TASK if len(self.target_cols) > 1 else TaskType.SINGLE_TASK
        return v.value

    def get_train_test_split(
        self, featurization_fn: Optional[Callable] = None
    ) -> tuple[Subset, Union["Subset", dict[str, Subset]]]:
        """Construct the train and test sets, given the split in the benchmark specification.

        Returns [`Subset`][polaris.dataset.Subset] objects, which offer several ways of accessing the data
        and can thus easily serve as a basis to build framework-specific (e.g. PyTorch, Tensorflow)
        data-loaders on top of.

        Args:
            featurization_fn: A function to apply to the input data. If a multi-input benchmark, this function
                expects an input in the format specified by the `input_format` parameter.

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
                target_cols=self.target_cols,
                hide_targets=hide_targets,
                featurization_fn=featurization_fn,
            )

        train = _get_subset(self.split[0], hide_targets=False)
        if isinstance(self.split[1], dict):
            test = {k: _get_subset(v, hide_targets=True) for k, v in self.split[1].items()}
        else:
            test = _get_subset(self.split[1], hide_targets=True)

        return train, test

    def evaluate(
        self, y_pred: Optional[PredictionsType] = None, y_prob: Optional[PredictionsType] = None
    ) -> BenchmarkResults:
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
            y_pred: The predictions for the test set, as NumPy arrays.
                If there are multiple targets, the predictions should be wrapped in a dictionary with the target labels as keys.
                If there are multiple test sets, the predictions should be further wrapped in a dictionary
                    with the test subset labels as keys.
            y_prob: The predicted probabilities for the test set, as NumPy arrays.

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

        if not isinstance(y_prob, dict) or all(k in self.target_cols for k in y_prob):
            y_prob = {"test": y_prob}

        if any(k not in y_pred for k in test.keys()) and any(k not in y_prob for k in test.keys()):
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
                    score = metric(
                        y_true=y_true_subset, y_pred=y_pred.get(test_label), y_prob=y_prob.get(test_label)
                    )
                    scores.loc[len(scores)] = (test_label, "aggregated", metric, score)
                    continue

                if not isinstance(y_true_subset, dict):
                    # Single task
                    score = metric(
                        y_true=y_true_subset, y_pred=y_pred.get(test_label), y_prob=y_prob.get(test_label)
                    )
                    scores.loc[len(scores)] = (
                        test_label,
                        self.target_cols[0],
                        metric,
                        score,
                    )
                    continue

                # Otherwise, for every target...
                for target_label, y_true_target in y_true_subset.items():
                    # Single-task metrics for a multi-task benchmark
                    # In such a setting, there can be NaN values, which we thus have to filter out.
                    mask = ~np.isnan(y_true_target)
                    score = metric(
                        y_true=y_true_target[mask],
                        y_pred=y_pred[test_label][target_label][mask]
                        if y_pred[test_label] is not None
                        else None,
                        y_prob=y_prob[test_label][target_label][mask]
                        if y_prob[test_label] is not None
                        else None,
                    )
                    scores.loc[len(scores)] = (test_label, target_label, metric, score)

        return BenchmarkResults(results=scores, benchmark_name=self.name, benchmark_owner=self.owner)

    def upload_to_hub(
        self,
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
            settings=settings,
            cache_auth_token=cache_auth_token,
            **kwargs,
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
