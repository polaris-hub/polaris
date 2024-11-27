import json
from hashlib import md5
from typing import Any, Callable, ClassVar, Literal, Optional, Union

from pyroaring import BitMap
from loguru import logger
from typing_extensions import Self

from pydantic import computed_field, field_serializer, field_validator, model_validator
from polaris.dataset import Subset
from polaris.experimental._dataset_v2 import DatasetV2
from polaris.benchmark._base import BenchmarkSpecification
from polaris.evaluate import BenchmarkResults
from polaris.evaluate.utils import evaluate_benchmark
from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.misc import listit
from polaris.utils.types import (
    PredictionsType,
    TargetType,
)


class BenchmarkV2Specification(BenchmarkSpecification):
    version: ClassVar[Literal[2]] = 2

    dataset: Union[DatasetV2, str, dict[str, Any]]

    @field_validator("dataset")
    def _validate_dataset(cls, v):
        """
        Allows either passing a Dataset object or the kwargs to create one
        """
        if isinstance(v, dict):
            v = DatasetV2(**v)
        elif isinstance(v, str):
            v = DatasetV2.from_json(v)
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
        split = self.split

        # Convert train and test indices to BitMap
        train_idx = BitMap(split[0])
        if isinstance(split[1], dict):
            test_idx = {k: BitMap(v) for k, v in split[1].items()}
        else:
            test_idx = BitMap(split[1])

        self.split = (train_idx, test_idx)

        # Train partition can be empty (zero-shot)
        # Test partitions cannot be empty
        if (isinstance(test_idx, dict) and any(len(v) == 0 for v in test_idx.values())) or (
            not isinstance(test_idx, dict) and len(test_idx) == 0
        ):
            raise InvalidBenchmarkError("The predefined split contains empty test partitions")

        if len(train_idx) == 0:
            logger.info(
                "This benchmark only specifies a test set. It will return an empty train set in `get_train_test_split()`"
            )

        # Ensure no overlap between train and test sets
        if isinstance(test_idx, dict):
            for test_set in test_idx.values():
                if not train_idx.isdisjoint(test_set):
                    raise InvalidBenchmarkError(
                        "The predefined split specifies overlapping train and test sets"
                    )
        elif not train_idx.isdisjoint(test_idx):
            raise InvalidBenchmarkError("The predefined split specifies overlapping train and test sets")

        # Check for duplicate indices and validity
        dataset_length = len(self.dataset) if self.dataset else 0
        all_indices = train_idx | (
            BitMap.union(*test_idx.values()) if isinstance(test_idx, dict) else test_idx
        )
        if any(i < 0 or i >= dataset_length for i in all_indices):
            raise InvalidBenchmarkError("The predefined split contains invalid indices")

        return self

    @field_serializer("split")
    def _serialize_split(self, v):
        """Convert any tuple to list to make sure it's serializable"""
        return listit(v)

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

        if not isinstance(self.split[1], dict):
            split = self.split[0], {"test": self.split[1]}
        else:
            split = self.split

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
        if isinstance(test_split, dict):
            test = {k: make_test_subset(v) for k, v in test_split.items()}
        else:
            test = make_test_subset(test_split)

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

        return train, test

    def evaluate(
        self, y_pred: Optional[PredictionsType] = None, y_prob: Optional[PredictionsType] = None
    ) -> BenchmarkResults:
        """Execute the evaluation protocol for the benchmark, given a set of predictions.

        info: What about `y_true`?
            Contrary to other frameworks that you might be familiar with, we opted for a signature that includes just
            the predictions. This reduces the chance of accidentally using the test targets during training.

        info: Expected structure for `y_pred` and `y_prob` arguments
            The supplied `y_pred` and `y_prob` arguments must adhere to a certain structure depending on the number of
            tasks and test sets included in the benchmark. Refer to the following for guidance on the correct structure when
            creating your `y_pred` and `y_prod` objects:

            - Single task, single set: `[values...]`
            - Multi-task, single set: `{task_name_1: [values...], task_name_2: [values...]}`
            - Single task, multi-set: `{test_set_1: {task_name: [values...]}, test_set_2: {task_name: [values...]}}`
            - Multi-task, multi-set: `{test_set_1: {task_name_1: [values...], task_name_2: [values...]}, test_set_2: {task_name_1: [values...], task_name_2: [values...]}}`

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
        # The `evaluate_benchmark` function expects the benchmark labels to be of a certain structure which
        # depends on the number of tasks and test sets defined for the benchmark. Below, we build the structure
        # of the benchmark labels based on the aforementioned factors.
        test = self._get_test_set(hide_targets=False)
        if isinstance(test, dict):
            #
            # For multi-set benchmarks
            y_true = {}
            for test_set_name, values in test.items():
                y_true[test_set_name] = {}
                if isinstance(values.targets, dict):
                    #
                    # For multi-task, multi-set benchmarks
                    for task_name, values in values.targets.items():
                        y_true[test_set_name][task_name] = values
                else:
                    #
                    # For single task, multi-set benchmarks
                    y_true[test_set_name][self.target_cols[0]] = values.targets
        else:
            #
            # For single set benchmarks (single and multiple task)
            y_true = test.targets

        scores = evaluate_benchmark(self.target_cols, self.metrics, y_true, y_pred=y_pred, y_prob=y_prob)

        return BenchmarkResults(results=scores, benchmark_name=self.name, benchmark_owner=self.owner)
