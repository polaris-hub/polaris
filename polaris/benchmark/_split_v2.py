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


class SplitV2(BaseModel):
    training: IndexSet
    test_sets: dict[str, IndexSet] = Field(default_factory=dict)

    @field_validator("training", mode="before")
    @classmethod
    def _parse_training_set(cls, v: bytes | IndexSet) -> IndexSet:
        """Accepted a binary serialized IndexSet"""
        if isinstance(v, bytes):
            return IndexSet.deserialize(v)
        return v

    @field_validator("test_sets", mode="before")
    @classmethod
    def _parse_test_index_sets(cls, v: dict[str, bytes | IndexSet]) -> dict[str, IndexSet]:
        """Parse test sets from bytes or IndexSet objects"""
        if not isinstance(v, dict):
            return {}

        parsed_sets = {}
        for label, index_set in v.items():
            if isinstance(index_set, bytes):
                parsed_sets[label] = IndexSet.deserialize(index_set)
            else:
                parsed_sets[label] = index_set
        return parsed_sets

    @model_validator(mode="before")
    @classmethod
    def _handle_backward_compatibility(cls, data):
        """Handle backward compatibility with single 'test' field"""
        if isinstance(data, dict) and "test" in data and "test_sets" not in data:
            # Convert single test field to test_sets format
            test_value = data.pop("test")
            data["test_sets"] = {"test": test_value}
        return data

    @field_validator("training")
    @classmethod
    def _validate_training_set(cls, v: IndexSet) -> IndexSet:
        """Training index set can be empty (zero-shot)"""
        if v.datapoints == 0:
            logger.info(
                "This benchmark only specifies a test set. It will return an empty train set in `get_train_test_split()`"
            )
        return v

    @field_validator("test_sets")
    @classmethod
    def _validate_test_sets(cls, v: dict[str, IndexSet]) -> dict[str, IndexSet]:
        """Test index sets cannot be empty"""
        if not v:
            raise InvalidBenchmarkError("At least one test set must be specified")

        for label, index_set in v.items():
            if index_set.datapoints == 0:
                raise InvalidBenchmarkError(f"Test set '{label}' contains empty test partitions")
        return v

    @model_validator(mode="after")
    def validate_set_overlap(self) -> Self:
        """The training and test index sets do not overlap"""
        for label, test_set in self.test_sets.items():
            if self.training.intersect(test_set):
                raise InvalidBenchmarkError(
                    f"The predefined split specifies overlapping train and test sets for test set '{label}'"
                )
        return self

    @property
    def n_train_datapoints(self) -> int:
        """The size of the train set."""
        return self.training.datapoints

    @property
    def n_test_sets(self) -> int:
        """The number of test sets"""
        return len(self.test_sets)

    @property
    def n_test_datapoints(self) -> dict[str, int]:
        """The size of (each of) the test set(s)."""
        return {label: index_set.datapoints for label, index_set in self.test_sets.items()}

    @property
    def max_index(self) -> int:
        """Maximum index across all sets"""
        all_indices = [self.training.indices.max()]
        all_indices.extend(test_set.indices.max() for test_set in self.test_sets.values())
        return max(all_indices)

    def test_items(self) -> Generator[tuple[str, IndexSet], None, None]:
        """Yield all test sets with their labels"""
        for label, index_set in self.test_sets.items():
            yield label, index_set

    # Backward compatibility property
    @property
    def test(self) -> IndexSet:
        """Backward compatibility: return the 'test' set if it exists, otherwise the first test set"""
        if "test" in self.test_sets:
            return self.test_sets["test"]
        elif self.test_sets:
            return next(iter(self.test_sets.values()))
        else:
            raise InvalidBenchmarkError("No test sets available")


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
