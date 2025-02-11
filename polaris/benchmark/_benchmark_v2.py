from typing import Any, Callable, ClassVar, Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from polaris.benchmark import BenchmarkSpecification
from polaris.evaluate.utils import evaluate_benchmark
from polaris.utils.types import IncomingPredictionsType

from polaris.evaluate import BenchmarkResultsV2
from polaris.benchmark._split_v2 import SplitSpecificationV2Mixin
from polaris.dataset import DatasetV2, Subset
from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.types import ColumnName


class BenchmarkV2Specification(SplitSpecificationV2Mixin, BenchmarkSpecification[BenchmarkResultsV2]):
    _version: ClassVar[Literal[2]] = 2

    dataset: DatasetV2 = Field(exclude=True)
    n_classes: dict[ColumnName, int] = Field(default_factory=dict)

    @field_validator("dataset", mode="before")
    @classmethod
    def _parse_dataset(
        cls,
        v: DatasetV2 | str | dict[str, Any],
    ) -> DatasetV2:
        """
        Allows either passing a Dataset object or the kwargs to create one
        """
        match v:
            case dict():
                return DatasetV2(**v)
            case str():
                return DatasetV2.from_json(v)
            case DatasetV2():
                return v

    @model_validator(mode="after")
    def _validate_n_classes(self) -> Self:
        """
        The number of classes for each of the target columns.
        """
        columns = set(self.n_classes.keys())
        if not columns.issubset(self.target_cols):
            raise InvalidBenchmarkError("Not all specified class numbers were found in the target columns.")

        return self

    @model_validator(mode="after")
    def _validate_split_in_dataset(self) -> Self:
        """
        Verifies that:
          - All indices are valid given the dataset
        """
        dataset_length = len(self.dataset)
        if self.split.max_index >= dataset_length:
            raise InvalidBenchmarkError("The predefined split contains invalid indices")

        return self

    def _get_test_sets(
        self, hide_targets=True, featurization_fn: Callable | None = None
    ) -> dict[str, Subset]:
        """
        Construct the test set(s), given the split in the benchmark specification. Used
        internally to construct the test set for client use and evaluation.
        """
        # TODO: We need a subset class that can handle very large index sets without copying or materializing all of them
        return {
            label: self._get_subset(index_set.indices, hide_targets, featurization_fn)
            for label, index_set in self.split.test_items()
        }

    def get_train_test_split(
        self, featurization_fn: Callable | None = None
    ) -> tuple[Subset, dict[str, Subset]]:
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
        train = self._get_subset(
            self.split.training.indices, hide_targets=False, featurization_fn=featurization_fn
        )
        test = self._get_test_sets(hide_targets=True, featurization_fn=featurization_fn)
        return train, test

    def evaluate(
        self,
        y_pred: IncomingPredictionsType | None = None,
        y_prob: IncomingPredictionsType | None = None,
    ) -> BenchmarkResultsV2:
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
            A `BenchmarkResultsV2` object. This object can be directly submitted to the Polaris Hub.

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

        return BenchmarkResultsV2(results=scores, benchmark_artifact_id=self.artifact_id)
