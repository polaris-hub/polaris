import logging
from functools import cached_property
from hashlib import md5
from typing import Generator, Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic.alias_generators import to_camel
from pyroaring import BitMap
from typing_extensions import Self

from polaris.utils.errors import InvalidBenchmarkError

logger = logging.getLogger(__name__)


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


class TrainTestIndices(BaseModel):
    """
    A single train-test split pair containing training and test index sets.

    This intermediate class represents one train-test split, which allows
    SplitV2 to support multiple such pairs for cross-validation scenarios.
    """

    training: IndexSet
    test: IndexSet

    @field_validator("training", "test", mode="before")
    @classmethod
    def _parse_index_set(cls, v: bytes | IndexSet) -> IndexSet:
        """Accept a binary serialized IndexSet"""
        if isinstance(v, bytes):
            return IndexSet.deserialize(v)
        return v

    @field_validator("training")
    @classmethod
    def _validate_training_set(cls, v: IndexSet) -> IndexSet:
        """Training index set can be empty (zero-shot)"""
        if v.datapoints == 0:
            logger.info(
                "This train-test split only specifies a test set. It will return an empty train set in `get_train_test_split()`"
            )
        return v

    @field_validator("test")
    @classmethod
    def _validate_test_set(cls, v: IndexSet) -> IndexSet:
        """Test index set cannot be empty"""
        if v.datapoints == 0:
            raise InvalidBenchmarkError("Test set cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_set_overlap(self) -> Self:
        """The training and test index sets do not overlap"""
        if self.training.intersect(self.test):
            raise InvalidBenchmarkError("The predefined split specifies overlapping train and test sets")
        return self

    @property
    def n_train_datapoints(self) -> int:
        """The size of the train set."""
        return self.training.datapoints

    @property
    def n_test_datapoints(self) -> int:
        """The size of the test set."""
        return self.test.datapoints

    @property
    def max_index(self) -> int:
        """Maximum index across train and test sets"""
        max_indices = []

        # Only add max if the bitmap is not empty
        if len(self.training.indices) > 0:
            max_indices.append(self.training.indices.max())
        max_indices.append(self.test.indices.max())

        return max(max_indices)


class SplitV2(BaseModel):
    """
    A collection of train-test splits for a benchmark.

    This supports multiple train-test pairs, enabling cross-validation and other
    multi-split evaluation scenarios. Each split is labeled and contains its own
    training and test sets.
    """

    splits: dict[str, TrainTestIndices]

    @model_validator(mode="after")
    def validate_splits_not_empty(self) -> Self:
        """Ensure at least one split is provided"""
        if not self.splits:
            raise InvalidBenchmarkError("At least one split must be specified")
        return self

    @property
    def n_splits(self) -> int:
        """The number of splits"""
        return len(self.splits)

    @property
    def split_labels(self) -> list[str]:
        """Labels of all splits"""
        return list(self.splits.keys())

    @property
    def n_train_datapoints(self) -> dict[str, int]:
        """The size of the train set for each split."""
        return {label: split.n_train_datapoints for label, split in self.splits.items()}

    @property
    def n_test_datapoints(self) -> dict[str, int]:
        """The size of the test set for each split."""
        return {label: split.n_test_datapoints for label, split in self.splits.items()}

    @property
    def max_index(self) -> int:
        """Maximum index across all splits"""
        return max(split.max_index for split in self.splits.values())

    def split_items(self) -> Generator[tuple[str, TrainTestIndices], None, None]:
        """Yield all splits with their labels"""
        for label, split in self.splits.items():
            yield label, split
