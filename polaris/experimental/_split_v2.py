from functools import cached_property
from hashlib import md5
from typing import Generator, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic.alias_generators import to_camel
from pyroaring import BitMap
from typing_extensions import Self

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
        # TODO: Until we support multi-test benchmarks
        return {"test": self.test.datapoints}

    @property
    def max_index(self) -> int:
        # TODO: Until we support multi-test benchmarks (need)
        return max(self.training.indices.max(), self.test.indices.max())

    def test_items(self) -> Generator[tuple[str, IndexSet], None, None]:
        # TODO: Until we support multi-test benchmarks
        yield "test", self.test


class SplitSpecificationV2Mixin(BaseModel):
    """
    Mixin class to add a split field to a benchmark. This is the V2 implementation.

    The internal representation for the split is a roaring bitmap,
    which drastically improves scalability over the V1 implementation.

    Attributes:
        split: The predefined train-test split to use for evaluation.
    """

    split: SplitV2

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
