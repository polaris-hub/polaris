from typing import Any, Callable, ClassVar, Generator, Literal, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic.alias_generators import to_camel
from pyroaring import BitMap
from typing_extensions import Self

from polaris.benchmark._base import BenchmarkSpecification
from polaris.benchmark._definitions import MultiTaskMixin, SingleTaskMixin
from polaris.dataset import Subset
from polaris.experimental._dataset_v2 import DatasetV2
from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.types import TargetType


class IndexSet(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    indices: BitMap = Field(default_factory=BitMap, exclude=True)

    @field_validator("indices", mode="before")
    @classmethod
    def _validate_indices(cls, v: BitMap | Sequence[int]) -> BitMap:
        """
        Accept an initial sequence iof ints, and turn it into a BitMap
        """
        if isinstance(v, BitMap):
            return v
        return BitMap(v)

    @computed_field
    @property
    def datapoints(self):
        return len(self.indices)


class SplitV2(BaseModel):
    training: IndexSet
    test: IndexSet

    @field_validator("training")
    @classmethod
    def _validate_training_set(cls, v: IndexSet):
        """
        Train index set can be empty (zero-shot)
        """
        if v.datapoints == 0:
            logger.info(
                "This benchmark only specifies a test set. It will return an empty train set in `get_train_test_split()`"
            )

    @field_validator("test")
    @classmethod
    def _validate_test_set(cls, v: IndexSet):
        """
        Test index set cannot be empty
        """
        if v.datapoints == 0:
            raise InvalidBenchmarkError("The predefined split contains empty test partitions")

    @model_validator(mode="after")
    def validate_set_overlap(self) -> Self:
        """
        The train and test indices do not overlap
        """
        if self.training.intersect(self.test):
            raise InvalidBenchmarkError("The predefined split specifies overlapping train and test sets")
        return self

    @computed_field
    @property
    def n_train_datapoints(self) -> int:
        """
        The size of the train set.
        """
        return self.training.datapoints

    @computed_field
    @property
    def n_test_sets(self) -> int:
        """
        The number of test sets
        """
        # TODO: Until we support multi-test benchmarks
        return 1

    @computed_field
    @property
    def n_test_datapoints(self) -> dict[str, int]:
        """
        The size of (each of) the test set(s).
        """
        return {"test": self.test.datapoints}

    @computed_field
    @property
    def max_index(self):
        return max(self.training.indices.max(), self.test.indices.max())

    def test_items(self) -> Generator[tuple[str, IndexSet], None, None]:
        # TODO: Until we support multi-test benchmarks
        yield "test", self.test


class BenchmarkV2Specification(BenchmarkSpecification):
    version: ClassVar[Literal[2]] = 2

    dataset: DatasetV2
    split: SplitV2 = Field(..., exclude=True)

    @field_validator("dataset", mode="before")
    @classmethod
    def _validate_dataset(
        cls,
        v: DatasetV2 | str | dict[str, Any],
    ):
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
    def _validate_split_in_dataset(self) -> Self:
        """
        Verifies that:
          - All indices are valid given the dataset
          3) There is no duplicate indices in any of the sets
          5) No row exists in the test set where all labels are missing/empty
        """
        dataset_length = len(self.dataset) if self.dataset else 0
        if self.split.max_index >= dataset_length:
            raise InvalidBenchmarkError("The predefined split contains invalid indices")

        return self

    @computed_field
    @property
    def n_train_datapoints(self) -> int:
        """The size of the train set."""
        return self.split.n_train_datapoints

    @computed_field
    @property
    def n_test_sets(self) -> int:
        """The number of test sets"""
        return self.split.n_test_sets

    @computed_field
    @property
    def n_test_datapoints(self) -> dict[str, int]:
        """The size of (each of) the test set(s)."""
        return self.split.n_test_datapoints

    @computed_field
    @property
    def n_classes(self) -> dict[str, int]:
        """
        The number of classes for each of the target columns.
        """
        n_classes = {}
        for target in (
            target
            for target in self.target_cols
            if self.target_types.get(target) == TargetType.CLASSIFICATION
        ):
            # TODO: Don't use table attribute
            n_classes[target] = self.dataset.table.loc[:, target].nunique()
        return n_classes

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


class SingleTaskBenchmarkV2Specification(SingleTaskMixin, BenchmarkV2Specification):
    """Single-task benchmark for version 2."""

    pass


class MultiTaskBenchmarkV2Specification(MultiTaskMixin, BenchmarkV2Specification):
    """Multi-task benchmark for version 2."""

    pass
