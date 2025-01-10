from typing import Any, Callable, ClassVar, Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from polaris.benchmark import BenchmarkSpecification
from polaris.dataset import DatasetV2, Subset
from polaris.experimental._split_v2 import SplitSpecificationV2Mixin
from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.types import ColumnName


class BenchmarkV2Specification(SplitSpecificationV2Mixin, BenchmarkSpecification):
    _version: ClassVar[Literal[2]] = 2

    dataset: DatasetV2 = Field(exclude=True)
    n_classes: dict[ColumnName, int]

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
