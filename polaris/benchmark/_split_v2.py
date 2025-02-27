import logging
from functools import cached_property
from hashlib import md5
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from pydantic.alias_generators import to_camel
from pyroaring import BitMap
from typing_extensions import Self

from polaris.utils.errors import InvalidSplitError

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
    label: str = Field(description="A label for this split")
    method: str = Field(description="The method used to create this split")
    train: IndexSet = Field(description="The training index set")
    test: IndexSet = Field(description="The test index set")

    @field_validator("train", "test", mode="before")
    @classmethod
    def _parse_index_sets(cls, v: bytes | IndexSet) -> bytes | IndexSet:
        """
        Accepted a binary serialized IndexSet
        """
        if isinstance(v, bytes):
            return IndexSet.deserialize(v)
        return v

    @field_validator("train")
    @classmethod
    def _validate_train_set(cls, v: IndexSet) -> IndexSet:
        """
        Training index set can be empty (zero-shot)
        """
        if v.datapoints == 0:
            logger.info(
                "This split only specifies a test set. It will return an empty train set in `get_train_test_split()`"
            )
        return v

    @field_validator("test")
    @classmethod
    def _validate_test_set(cls, v: IndexSet) -> IndexSet:
        """
        Test index set cannot be empty
        """
        if v.datapoints == 0:
            raise InvalidSplitError("This split contains an empty test set")
        return v

    @model_validator(mode="after")
    def validate_set_overlap(self) -> Self:
        """
        The training and test index sets do not overlap
        """
        if self.train.intersect(self.test):
            raise InvalidSplitError("This split specifies overlapping training and test sets")
        return self

    @property
    def n_train_datapoints(self) -> int:
        """
        The size of the train set.
        """
        return self.train.datapoints

    @property
    def n_test_datapoints(self) -> int:
        """
        The size of the test set.
        """
        return self.test.datapoints

    @property
    def max_index(self) -> int:
        return max(self.training.indices.max(), self.test.indices.max())
