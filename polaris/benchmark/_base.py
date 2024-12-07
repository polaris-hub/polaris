import abc
import json
from hashlib import md5
from itertools import chain
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Sequence, TypeAlias

import fsspec
import numpy as np
from loguru import logger
from pydantic import (
    Field,
    FieldSerializationInfo,
    SerializerFunctionWrapHandler,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo
from sklearn.utils.multiclass import type_of_target
from typing_extensions import Self

from polaris._artifact import BaseArtifactModel
from polaris.dataset import DatasetV1, Subset
from polaris.evaluate import BenchmarkResults, Metric
from polaris.evaluate.utils import evaluate_benchmark
from polaris.hub.settings import PolarisHubSettings
from polaris.mixins import ChecksumMixin
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.misc import listit
from polaris.utils.types import (
    AccessType,
    HubOwner,
    IncomingPredictionsType,
    SplitType,
    TargetType,
    TaskType,
)

ColumnName: TypeAlias = str


class BenchmarkSpecification(BaseArtifactModel, abc.ABC):
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
        metrics: The metrics to use for evaluating performance
        main_metric: The main metric used to rank methods. If `None`, the first of the `metrics` field.
        readme: Markdown text that can be used to provide a formatted description of the benchmark.
            If using the Polaris Hub, it is worth noting that this field is more easily edited through the Hub UI
            as it provides a rich text editor for writing markdown.
        target_types: A dictionary that maps target columns to their type. If not specified, this is automatically inferred.
    For additional meta-data attributes, see the [`BaseArtifactModel`][polaris._artifact.BaseArtifactModel] class.
    """

    _artifact_type = "benchmark"

    # Public attributes
    dataset: BaseArtifactModel
    target_cols: set[ColumnName] = Field(min_length=1)
    input_cols: set[ColumnName] = Field(min_length=1)
    metrics: set[Metric] = Field(min_length=1)
    main_metric: Metric | str
    readme: str = ""
    target_types: dict[ColumnName, TargetType] = Field(default_factory=dict, validate_default=True)

    @field_validator("target_cols", "input_cols", mode="before")
    @classmethod
    def _parse_cols(cls, v: str | Sequence[str], info: ValidationInfo) -> set[str]:
        """Verifies all columns are present in the dataset."""
        if isinstance(v, str):
            v = {v}
        else:
            v = set(v)
        return v

    @field_validator("target_types", mode="before")
    @classmethod
    def _parse_target_types(
        cls, v: dict[ColumnName, TargetType | str | None]
    ) -> dict[ColumnName, TargetType]:
        """
        Converts the target types to TargetType enums if they are strings.
        """
        return {
            target: TargetType(val) if isinstance(val, str) else val
            for target, val in v.items()
            if val is not None
        }

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, v: str | Metric | list[str | Metric]) -> set[Metric]:
        """
        Verifies all specified metrics are either a Metric object or a valid metric name.
        Also verifies there are no duplicate metrics.

        If there are multiple test sets, it is assumed the same metrics are used across test sets.
        """
        if isinstance(v, str):
            v = {"label": v}
        if isinstance(v, set):
            v = list(v)
        if not isinstance(v, list):
            v = [v]

        def _convert(m: str | dict | Metric) -> Metric:
            if isinstance(m, str):
                return Metric(label=m)
            if isinstance(m, dict):
                return Metric(**m)
            return m

        v = [_convert(m) for m in v]

        unique_metrics = set(v)

        if len(unique_metrics) != len(v):
            raise InvalidBenchmarkError("The benchmark specifies duplicate metrics.")

        unique_names = {m.name for m in unique_metrics}
        if len(unique_names) != len(unique_metrics):
            raise InvalidBenchmarkError(
                "The metrics of a benchmark need to have unique names. Specify a custom name with Metric(custom_name=...)"
            )

        return unique_metrics

    @model_validator(mode="after")
    def _validate_main_metric_is_in_metrics(self) -> Self:
        if isinstance(self.main_metric, str):
            for m in self.metrics:
                if m.name == self.main_metric:
                    self.main_metric = m
                    break
        if self.main_metric not in self.metrics:
            raise InvalidBenchmarkError("The main metric should be one of the specified metrics")
        return self

    @model_validator(mode="after")
    def _validate_cols(self) -> Self:
        """
        Verifies that all specified columns are present in the dataset.
        """
        columns = self.target_cols | self.input_cols
        dataset_columns = set(self.dataset.columns)
        if not columns.issubset(dataset_columns):
            raise InvalidBenchmarkError("Not all specified columns were found in the dataset.")

        return self

    @field_serializer("metrics", mode="wrap")
    @staticmethod
    def _serialize_metrics(
        value: set[Metric], handler: SerializerFunctionWrapHandler, info: FieldSerializationInfo
    ) -> list[dict]:
        """Convert the set to a list"""
        return handler(list(value))

    @model_validator(mode="after")
    def _validate_target_types(self) -> Self:
        """
        Verifies that all target types are for benchmark targets.
        """
        columns = set(self.target_types.keys())
        if not columns.issubset(self.target_cols):
            raise InvalidBenchmarkError(
                f"Not all specified target types were found in the target columns. {columns} - {self.target_cols}"
            )
        return self

    @field_serializer("main_metric")
    def _serialize_main_metric(value: Metric) -> str:
        """Convert the set to a list"""
        return value.name

    @field_serializer("target_types")
    def _serialize_target_types(self, target_types):
        """Convert from enum to string to make sure it's serializable"""
        return {k: v.value for k, v in target_types.items()}

    @field_serializer("target_cols", "input_cols")
    def _serialize_columns(self, v: set[str]) -> list[str]:
        return list(v)

    @computed_field
    @property
    def dataset_artifact_id(self) -> str:
        return self.dataset.artifact_id

    @computed_field
    @property
    def task_type(self) -> str:
        """The high-level task type of the benchmark."""
        v = TaskType.MULTI_TASK if len(self.target_cols) > 1 else TaskType.SINGLE_TASK
        return v.value

    @abc.abstractmethod
    def _get_test_sets(
        self, hide_targets=True, featurization_fn: Callable | None = None
    ) -> dict[str, Subset]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_test_datapoints(self) -> dict[str, int]:
        """
        The size of (each of) the test set(s).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def test_set_labels(self) -> list[str]:
        """
        The labels of the test sets.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_train_datapoints(self) -> int:
        """
        The size of the train set.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_test_sets(self) -> int:
        """
        The number of test sets
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def test_set_sizes(self) -> dict[str, int]:
        """
        The sizes of the test sets.
        """
        raise NotImplementedError

    def _get_subset(self, indices, hide_targets=True, featurization_fn=None) -> Subset:
        """Returns a [`Subset`][polaris.dataset.Subset] using the given indices. Used
        internally to construct the train and test sets."""
        return Subset(
            dataset=self.dataset,
            indices=indices,
            input_cols=self.input_cols,
            target_cols=self.target_cols,
            hide_targets=hide_targets,
            featurization_fn=featurization_fn,
        )

    def evaluate(
        self,
        y_pred: IncomingPredictionsType | None = None,
        y_prob: IncomingPredictionsType | None = None,
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
            y_prob: The predicted probabilities for the test set, formatted similarly to predictions, based on the
                number of tasks and test sets.

        Returns:
            A `BenchmarkResults` object. This object can be directly submitted to the Polaris Hub.

        Examples:
            1. For regression benchmarks:
                pred_scores = your_model.predict_score(molecules) # predict continuous score values
                benchmark.evaluate(y_pred=pred_scores)
            2. For classification benchmarks:
                - If `roc_auc` and `pr_auc` are in the metric list, both class probabilities and label predictions are required:
                    pred_probs = your_model.predict_proba(molecules) # predict probablities
                    pred_labels = your_model.predict_labels(molecules) # predict class labels
                    benchmark.evaluate(y_pred=pred_labels, y_prob=pred_probs)
                - Otherwise:
                    benchmark.evaluate(y_pred=pred_labels)
        """

        # Instead of having the user pass the ground truth, we extract it from the benchmark spec ourselves.
        y_true = self._get_test_set(hide_targets=False)

        scores = evaluate_benchmark(
            target_cols=self.target_cols,
            test_set_labels=self.test_set_labels,
            test_set_sizes=self.test_set_sizes,
            metrics=self.metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return BenchmarkResults(results=scores, benchmark_artifact_id=self.artifact_id)

    def upload_to_hub(
        self,
        settings: PolarisHubSettings | None = None,
        cache_auth_token: bool = True,
        access: AccessType = "private",
        owner: HubOwner | str | None = None,
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
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()
        data["dataset"] = self.dataset.to_json(destination)

        path = dest / "benchmark.json"
        with fsspec.open(str(path), "w") as f:
            json.dump(data, f)

        return str(path)

    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        repr_dict = self.model_dump(exclude={"dataset"})
        repr_dict["dataset_name"] = self.dataset.name
        return repr_dict

    def _repr_html_(self):
        """For pretty printing in Jupyter."""
        return dict2html(self.model_dump(exclude={"dataset"}))

    def __repr__(self):
        return self.model_dump_json(exclude={"dataset"}, indent=2)

    def __str__(self):
        return self.__repr__()


class BenchmarkV1Specification(BenchmarkSpecification, ChecksumMixin):
    _version: ClassVar[Literal[1]] = 1

    dataset: DatasetV1
    split: SplitType

    @field_validator("dataset", mode="before")
    @classmethod
    def _parse_dataset(cls, v: DatasetV1 | str | dict[str, Any]) -> DatasetV1:
        """
        Allows either passing a Dataset object or the kwargs to create one
        """
        match v:
            case dict():
                return DatasetV1(**v)
            case str():
                return DatasetV1.from_json(v)
            case DatasetV1():
                return v

    @model_validator(mode="after")
    def _infer_target_types(self) -> Self:
        """Try to automatically infer the target types if not already set"""

        for target in filter(lambda target: target not in self.target_types, self.target_cols):
            # Skip inferring the target type for pointer columns.
            # This would be complex to implement properly.
            # For these columns, dataset creators can still manually specify the target type.
            column_annotation = self.dataset.annotations.get(target)
            if column_annotation is not None and column_annotation.is_pointer:
                continue

            val = self.dataset.table.loc[:, target]

            # Non-numeric columns can be targets (e.g. prediction molecular reactions),
            # but in that case we currently don't infer the target type.
            if not np.issubdtype(val.dtype, np.number):
                continue

            # Remove the nans for multiple task dataset when the table is sparse
            target_type = type_of_target(val[~np.isnan(val)])
            match target_type:
                case "continuous":
                    self.target_types[target] = TargetType.REGRESSION
                case "binary" | "multiclass":
                    self.target_types[target] = TargetType.CLASSIFICATION

        return self

    @model_validator(mode="after")
    def _validate_split(self) -> Self:
        """
        Verifies that:
          1) There are no empty test partitions
          2) All indices are valid given the dataset
          3) There is no duplicate indices in any of the sets
          4) There is no overlap between the train and test set
          5) No row exists in the test set where all labels are missing/empty
        """

        if not isinstance(self.split[1], dict):
            self.split = self.split[0], {"test": self.split[1]}
        split = self.split

        # Train partition can be empty (zero-shot)
        # Test partitions cannot be empty
        if any(len(v) == 0 for v in split[1].values()):
            raise InvalidBenchmarkError("The predefined split contains empty test partitions")

        train_idx_list = split[0]
        full_test_idx_list = list(chain.from_iterable(split[1].values()))

        if len(train_idx_list) == 0:
            logger.info(
                "This benchmark only specifies a test set. It will return an empty train set in `get_train_test_split()`"
            )

        train_idx_set = set(train_idx_list)
        full_test_idx_set = set(full_test_idx_list)

        # The train and test indices do not overlap
        if len(train_idx_set & full_test_idx_set) > 0:
            raise InvalidBenchmarkError("The predefined split specifies overlapping train and test sets")

        # Check for duplicate indices within the train set
        if len(train_idx_set) != len(train_idx_list):
            raise InvalidBenchmarkError("The training set contains duplicate indices")

        # Check for duplicate indices within a given test set. Because a user can specify
        # multiple test sets for a given benchmark and it is acceptable for indices to be shared
        # across test sets, we check for duplicates in each test set independently.
        for test_set_name, test_set_idx_list in split[1].items():
            if len(test_set_idx_list) != len(set(test_set_idx_list)):
                raise InvalidBenchmarkError(
                    f'Test set with name "{test_set_name}" contains duplicate indices'
                )

        # All indices are valid given the dataset
        dataset = self.dataset
        if dataset is not None:
            max_i = len(dataset)
            if any(i < 0 or i >= max_i for i in chain(train_idx_list, full_test_idx_set)):
                raise InvalidBenchmarkError("The predefined split contains invalid indices")

        return self

    @field_serializer("split")
    def _serialize_split(self, v: SplitType):
        """Convert any tuple to list to make sure it's serializable"""
        return listit(v)

    def _compute_checksum(self) -> str:
        """
        Computes a hash of the benchmark.

        This is meant to uniquely identify the benchmark and can be used to verify the version.
        """

        hash_fn = md5()
        hash_fn.update(self.dataset.md5sum.encode("utf-8"))
        for c in sorted(self.target_cols):
            hash_fn.update(c.encode("utf-8"))
        for c in sorted(self.input_cols):
            hash_fn.update(c.encode("utf-8"))
        for m in sorted(str(m) for m in self.metrics):
            hash_fn.update(m.encode("utf-8"))

        # Train set
        s = json.dumps(sorted(self.split[0]))
        hash_fn.update(s.encode("utf-8"))

        # Test sets
        for k in sorted(self.split[1].keys()):
            s = json.dumps(sorted(self.split[1][k]))
            hash_fn.update(k.encode("utf-8"))
            hash_fn.update(s.encode("utf-8"))

        checksum = hash_fn.hexdigest()
        return checksum

    def _get_test_sets(
        self, hide_targets=True, featurization_fn: Callable | None = None
    ) -> dict[str, Subset]:
        """Construct the test set(s), given the split in the benchmark specification. Used
        internally to construct the test set for client use and evaluation.
        """
        test_split = self.split[1]
        return {
            k: self._get_subset(v, hide_targets=hide_targets, featurization_fn=featurization_fn)
            for k, v in test_split.items()
        }

    def get_train_test_split(
        self, featurization_fn: Callable | None = None
    ) -> tuple[Subset, Subset | dict[str, Subset]]:
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
        train = self._get_subset(self.split[0], hide_targets=False, featurization_fn=featurization_fn)
        test = self._get_test_sets(hide_targets=True, featurization_fn=featurization_fn)

        # For improved UX, we return the object instead of the dictionary if there is only one test set.
        # Internally, however, assume that the test set is always a dictionary simplifies the code.
        if len(test) == 1:
            test = test["test"]
        return train, test

    @computed_field
    @property
    def test_set_sizes(self) -> dict[str, int]:
        """The sizes of the test sets."""
        return {k: len(v) for k, v in self.split[1].items()}

    @computed_field
    @property
    def n_test_sets(self) -> int:
        """The number of test sets"""
        return len(self.split[1])

    @computed_field
    @property
    def n_train_datapoints(self) -> int:
        """The size of the train set."""
        return len(self.split[0])

    @computed_field
    @property
    def test_set_labels(self) -> list[str]:
        """The labels of the test sets."""
        return sorted(list(self.split[1].keys()))

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
        """
        The number of classes for each of the target columns.
        """
        return {
            target: self.dataset.table.loc[:, target].nunique()
            for target in self.target_cols
            if self.target_types.get(target) == TargetType.CLASSIFICATION
        }

    def __eq__(self, other) -> bool:
        if not isinstance(other, BenchmarkSpecification):
            return False
        return self.md5sum == other.md5sum
