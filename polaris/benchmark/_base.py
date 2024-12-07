import json
from hashlib import md5
from itertools import chain
from typing import Any, Callable, Optional, TypeAlias, Union

import fsspec
import numpy as np
from datamol.utils import fs
from loguru import logger
from pydantic import (
    Field,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from sklearn.utils.multiclass import type_of_target
from typing_extensions import Self

from polaris._artifact import BaseArtifactModel
from polaris.dataset import CompetitionDataset, DatasetV1, Subset
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

ColumnsType: TypeAlias = str | list[str]


class BenchmarkSpecification(BaseArtifactModel, ChecksumMixin):
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
        readme: Markdown text that can be used to provide a formatted description of the benchmark.
            If using the Polaris Hub, it is worth noting that this field is more easily edited through the Hub UI
            as it provides a rich text editor for writing markdown.
        target_types: A dictionary that maps target columns to their type. If not specified, this is automatically inferred.
    For additional meta-data attributes, see the [`BaseArtifactModel`][polaris._artifact.BaseArtifactModel] class.
    """

    _artifact_type = "benchmark"

    # Public attributes
    # Data
    dataset: Union[DatasetV1, CompetitionDataset, str, dict[str, Any]]
    target_cols: ColumnsType
    input_cols: ColumnsType
    split: SplitType
    metrics: Union[str, Metric, list[str | Metric]]
    main_metric: str | Metric | None = None

    # Additional meta-data
    readme: str = ""
    target_types: dict[str, Union[TargetType, str, None]] = Field(default_factory=dict, validate_default=True)

    @field_validator("dataset")
    def _validate_dataset(cls, v):
        """
        Allows either passing a Dataset object or the kwargs to create one
        """
        if isinstance(v, dict):
            v = DatasetV1(**v)
        elif isinstance(v, str):
            v = DatasetV1.from_json(v)
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
            raise InvalidBenchmarkError("Not all specified columns were found in the dataset.")
        if len(set(v)) != len(v):
            raise InvalidBenchmarkError("The task specifies duplicate columns")
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

    @field_validator("target_types")
    def _validate_target_types(cls, v, info: ValidationInfo):
        """Try to automatically infer the target types if not already set"""

        dataset = info.data.get("dataset")
        target_cols = info.data.get("target_cols")

        if dataset is None or target_cols is None:
            return v

        for target in target_cols:
            if target not in v:
                # Skip inferring the target type for pointer columns.
                # This would be complex to implement properly.
                # For these columns, dataset creators can still manually specify the target type.
                anno = dataset.annotations.get(target)
                if anno is not None and anno.is_pointer:
                    v[target] = None
                    continue

                val = dataset.table.loc[:, target]

                # Non numeric columns can be targets (e.g. prediction molecular reactions),
                # but in that case we currently don't infer the target type.
                if not np.issubdtype(val.dtype, np.number):
                    v[target] = None
                    continue

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
    def _validate_model(self) -> Self:
        """
        Sets a default metric if missing.
        """
        # Set a default main metric if not set yet
        if self.main_metric is None:
            self.main_metric = self.metrics[0]
        return self

    @field_serializer("metrics", "main_metric")
    def _serialize_metrics(self, v):
        """Return the string identifier so we can serialize the object"""
        if isinstance(v, Metric):
            return v.name
        return [m.name for m in v]

    @field_serializer("split")
    def _serialize_split(self, v: SplitType):
        """Convert any tuple to list to make sure it's serializable"""
        return listit(v)

    @field_serializer("target_types")
    def _serialize_target_types(self, v):
        """Convert from enum to string to make sure it's serializable"""
        return {k: v.value for k, v in self.target_types.items()}

    def _compute_checksum(self):
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
        for m in sorted(self.metrics, key=lambda k: k.name):
            hash_fn.update(m.name.encode("utf-8"))

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

    @computed_field
    @property
    def dataset_artifact_id(self) -> str:
        return self.dataset.artifact_id

    @computed_field
    @property
    def n_train_datapoints(self) -> int:
        """The size of the train set."""
        return len(self.split[0])

    @computed_field
    @property
    def n_test_sets(self) -> int:
        """The number of test sets"""
        return len(self.split[1])

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
            target_type = self.target_types.get(target)
            if (
                target_type is None
                or target_type == TargetType.REGRESSION
                or target_type == TargetType.DOCKING
            ):
                continue
            # TODO: Don't use table attribute
            n_classes[target] = self.dataset.table.loc[:, target].nunique()
        return n_classes

    @computed_field
    @property
    def task_type(self) -> str:
        """The high-level task type of the benchmark."""
        v = TaskType.MULTI_TASK if len(self.target_cols) > 1 else TaskType.SINGLE_TASK
        return v.value

    @computed_field
    @property
    def test_set_labels(self) -> list[str]:
        """The labels of the test sets."""
        return sorted(list(self.split[1].keys()))

    @computed_field
    @property
    def test_set_sizes(self) -> dict[str, int]:
        """The sizes of the test sets."""
        return {k: len(v) for k, v in self.split[1].items()}

    def _get_subset(self, indices, hide_targets=True, featurization_fn=None):
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

    def _get_test_set(
        self, hide_targets=True, featurization_fn: Optional[Callable] = None
    ) -> Union["Subset", dict[str, Subset]]:
        """Construct the test set(s), given the split in the benchmark specification. Used
        internally to construct the test set for client use and evaluation.
        """

        def make_test_subset(vals):
            return self._get_subset(vals, hide_targets=hide_targets, featurization_fn=featurization_fn)

        test_split = self.split[1]
        test = {k: make_test_subset(v) for k, v in test_split.items()}

        return test

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

        train = self._get_subset(self.split[0], hide_targets=False, featurization_fn=featurization_fn)
        test = self._get_test_set(hide_targets=True, featurization_fn=featurization_fn)

        # For improved UX, we return the object instead of the dictionary if there is only one test set.
        # Internally, however, assume that the test set is always a dictionary simplifies the code.
        if len(test) == 1:
            test = test["test"]
        return train, test

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
        y_true_subset = self._get_test_set(hide_targets=False)
        y_true_values = {k: v.targets for k, v in y_true_subset.items()}

        # Simplify the case where there is only one test set
        if len(y_true_values) == 1:
            y_true_values = y_true_values["test"]

        scores = evaluate_benchmark(
            target_cols=self.target_cols,
            test_set_labels=self.test_set_labels,
            test_set_sizes=self.test_set_sizes,
            metrics=self.metrics,
            y_true=y_true_values,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        return BenchmarkResults(results=scores, benchmark_artifact_id=self.artifact_id)

    def upload_to_hub(
        self,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        access: Optional[AccessType] = "private",
        owner: Union[HubOwner, str, None] = None,
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

    def _repr_html_(self) -> str:
        """For pretty printing in Jupyter."""
        return dict2html(self.model_dump(exclude={"dataset", "split"}))

    def __repr__(self):
        return self.model_dump_json(exclude={"dataset", "split"}, indent=2)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, BenchmarkSpecification):
            return False
        return self.md5sum == other.md5sum
