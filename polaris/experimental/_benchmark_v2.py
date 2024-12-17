from functools import cached_property
from hashlib import md5
from typing import Any, Callable, ClassVar, Generator, Literal, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic.alias_generators import to_camel
from pyroaring import BitMap
from typing_extensions import Self

from polaris.benchmark import BenchmarkSpecification
from polaris.benchmark._base import ColumnName
from polaris.dataset import Subset
from polaris.experimental._dataset_v2 import DatasetV2
from polaris.utils.errors import InvalidBenchmarkError


class IndexSet(BaseModel):
    """
    A set of indices for a split, either training or test.

    It wraps a Roaring Bitmap object to store the indices, and provides
    useful properties when serializing for upload to the Hub.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, alias_generator=to_camel)

    indices: BitMap = Field(default_factory=BitMap, frozen=True, exclude=True)

    @field_validator("indices", mode="before")
    @classmethod
    def _validate_indices(cls, v: BitMap | Sequence[int]) -> BitMap:
        """
        Accepts an initial sequence of ints, and turn it into a BitMap
        """
        if isinstance(v, BitMap):
            return v
        return BitMap(v)

    @computed_field
    @cached_property
    def datapoints(self) -> int:
        return len(self.indices)

    @computed_field
    @cached_property
    def md5_checksum(self) -> str:
        return md5(self.serialize()).hexdigest()

    def intersect(self, other: Self) -> bool:
        return self.indices.intersect(other.indices)

    def serialize(self) -> bytes:
        return self.indices.serialize()

    @staticmethod
    def deserialize(index_set: bytes) -> "IndexSet":
        return IndexSet(indices=BitMap.deserialize(index_set))


class SplitV2(BaseModel):
    training: IndexSet
    test: IndexSet

    @field_validator("training", "test", mode="before")
    @classmethod
    def _parse_index_sets(cls, v: bytes | IndexSet) -> bytes | IndexSet:
        """
        Accepted a binary serialized IndexSet
        """
        if isinstance(v, bytes):
            return IndexSet.deserialize(v)
        return v

    @field_validator("training")
    @classmethod
    def _validate_training_set(cls, v: IndexSet) -> IndexSet:
        """
        Training index set can be empty (zero-shot)
        """
        if v.datapoints == 0:
            logger.info(
                "This benchmark only specifies a test set. It will return an empty train set in `get_train_test_split()`"
            )
        return v

    @field_validator("test")
    @classmethod
    def _validate_test_set(cls, v: IndexSet) -> IndexSet:
        """
        Test index set cannot be empty
        """
        if v.datapoints == 0:
            raise InvalidBenchmarkError("The predefined split contains empty test partitions")
        return v

    @model_validator(mode="after")
    def validate_set_overlap(self) -> Self:
        """
        The training and test index sets do not overlap
        """
        if self.training.intersect(self.test):
            raise InvalidBenchmarkError("The predefined split specifies overlapping train and test sets")
        return self

    @property
    def n_train_datapoints(self) -> int:
        """
        The size of the train set.
        """
        return self.training.datapoints

    @property
    def n_test_sets(self) -> int:
        """
        The number of test sets
        """
        # TODO: Until we support multi-test benchmarks
        return 1

    @property
    def n_test_datapoints(self) -> dict[str, int]:
        """
        The size of (each of) the test set(s).
        """
        return {"test": self.test.datapoints}

    @property
    def max_index(self) -> int:
        return max(self.training.indices.max(), self.test.indices.max())

    def test_items(self) -> Generator[tuple[str, IndexSet], None, None]:
        # TODO: Until we support multi-test benchmarks
        yield "test", self.test


class BenchmarkV2Specification(BenchmarkSpecification):
    _version: ClassVar[Literal[2]] = 2

    dataset: DatasetV2 = Field(exclude=True)
    split: SplitV2
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
    def test_set_sizes(self) -> dict[str, int]:
        return {label: index_set.datapoints for label, index_set in self.split.test_items()}

    @computed_field
    @property
    def test_set_labels(self) -> list[str]:
        return list(label for label, _ in self.split.test_items())

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
