import json
from hashlib import md5
from itertools import chain
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Collection
from typing_extensions import deprecated, Self

import fsspec
import numpy as np
from pydantic import (
    Field,
    computed_field,
    field_validator,
    model_validator,
    field_serializer,
)
from sklearn.utils.multiclass import type_of_target

from polaris._artifact import BaseArtifactModel
from polaris.benchmark._split import SplitSpecificationV1Mixin
from polaris.benchmark._task import PredictiveTaskSpecificationMixin
from polaris.dataset import DatasetV1, Subset
from polaris.evaluate import BenchmarkResultsV1, Metric
from polaris.evaluate.utils import evaluate_benchmark
from polaris.hub.settings import PolarisHubSettings
from polaris.mixins import ChecksumMixin
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.types import (
    HubOwner,
    IncomingPredictionsType,
    TargetType,
)


@deprecated(
    "Use BenchmarkV2Specification instead. If you're loading this dataset from the Polaris Hub, you can ignore this warning."
)
class BenchmarkV1Specification(
    PredictiveTaskSpecificationMixin,
    BaseArtifactModel,
    SplitSpecificationV1Mixin,
    ChecksumMixin,
):
    """This class wraps a dataset with additional data to specify the evaluation logic.

    Specifically, it specifies:

    1. Which dataset to use;
    2. A task definition (we currently only support predictive tasks);
    3. A predefined, static train-test split to use during evaluation.

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
        readme: Markdown text that can be used to provide a formatted description of the benchmark.
            If using the Polaris Hub, it is worth noting that this field is more easily edited through the Hub UI
            as it provides a rich text editor for writing markdown.
        artifact_version: The version of the benchmark.
        artifact_changelog: A description of the changes made in this benchmark version.
        metrics: The metrics to use for evaluation.
        main_metric: The main metric to use for evaluation.
    For additional metadata attributes, see the base classes.
    """

    _artifact_type = "benchmark"
    _version: ClassVar[Literal[1]] = 1

    dataset: DatasetV1 = Field(exclude=True)
    readme: str = ""

    # Version-related fields
    artifact_version: int = Field(default=1, frozen=True)
    artifact_changelog: str | None = None

    # Metric-related attributes
    metrics: set[Metric] = Field(min_length=1)
    main_metric: Metric | str

    @computed_field
    @property
    def dataset_artifact_id(self) -> str:
        return self.dataset.artifact_id

    @field_validator("metrics", mode="before")
    @classmethod
    def _validate_metrics(cls, v: str | Metric | Collection[str | Metric]) -> set[Metric]:
        """
        Verifies all specified metrics are either a Metric object or a valid metric name.
        Also verifies there are no duplicate metrics.

        If there are multiple test sets, it is assumed the same metrics are used across test sets.
        """
        if isinstance(v, str):
            v = {"label": v}
        if not isinstance(v, Collection):
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

    @field_serializer("metrics")
    def _serialize_metrics(self, value: set[Metric]) -> list[Metric]:
        """
        Convert the set to a list. Since metrics are models and will be converted to dict,
        they will not be hashable members of a set.
        """
        return list(value)

    @field_serializer("main_metric")
    def _serialize_main_metric(self, value: Metric) -> str:
        """
        Convert the Metric to it's name
        """
        return value.name

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
    def _validate_split_in_dataset(self) -> Self:
        # All indices are valid given the dataset. We check the len of `self` here because a
        # competition entity includes both the dataset and benchmark in one artifact.
        max_i = len(self.dataset)
        if any(i < 0 or i >= max_i for i in chain(self.split[0], *self.split[1].values())):
            raise InvalidBenchmarkError("The predefined split contains invalid indices")

        return self

    @model_validator(mode="after")
    def _validate_cols_in_dataset(self) -> Self:
        """
        Verifies that all specified columns are present in the dataset.
        """
        columns = self.target_cols | self.input_cols
        dataset_columns = set(self.dataset.columns)
        if not columns.issubset(dataset_columns):
            raise InvalidBenchmarkError("Not all target or input columns were found in the dataset.")

        return self

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

    def _get_test_sets(
        self, hide_targets=True, featurization_fn: Callable | None = None
    ) -> dict[str, Subset]:
        """
        Construct the test set(s), given the split in the benchmark specification. Used
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
    def n_classes(self) -> dict[str, int]:
        """
        The number of classes for each of the target columns.
        """
        return {
            target: self.dataset.table.loc[:, target].nunique()
            for target in self.target_cols
            if self.target_types.get(target) == TargetType.CLASSIFICATION
        }

    def evaluate(
        self,
        y_pred: IncomingPredictionsType | None = None,
        y_prob: IncomingPredictionsType | None = None,
    ) -> BenchmarkResultsV1:
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
            A `BenchmarkResultsV1` object. This object can be directly submitted to the Polaris Hub.

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
        y_true = self._get_test_sets(hide_targets=False)

        scores = evaluate_benchmark(
            target_cols=list(self.target_cols),
            test_set_labels=self.test_set_labels,
            test_set_sizes=self.test_set_sizes,
            metrics=self.metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return BenchmarkResultsV1(results=scores, benchmark_artifact_id=self.artifact_id)

    def upload_to_hub(
        self,
        settings: PolarisHubSettings | None = None,
        cache_auth_token: bool = True,
        owner: HubOwner | str | None = None,
        parent_artifact_id: str | None = None,
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
            return client.upload_benchmark(self, owner=owner, parent_artifact_id=parent_artifact_id)

    def to_json(self, destination: str) -> str:
        """Save the benchmark to a destination directory as a JSON file.

        Warning: Multiple files
            Perhaps unintuitive, this method creates multiple files in the destination directory as it also saves
            the dataset it is based on to the specified destination.

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
        repr_dict = self.model_dump()
        repr_dict["dataset_name"] = self.dataset.name
        return repr_dict

    def _repr_html_(self):
        """For pretty printing in Jupyter."""
        return dict2html(self.model_dump())

    def __repr__(self):
        return self.model_dump_json(indent=2)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, BenchmarkV1Specification):
            return False
        return self.md5sum == other.md5sum
