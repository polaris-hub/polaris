from itertools import chain

from loguru import logger
from pydantic import BaseModel, computed_field, field_serializer, model_validator
from typing_extensions import Self

from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.misc import listit
from polaris.utils.types import SplitType


class SplitSpecificationV1Mixin(BaseModel):
    """
    Mixin class to add a split field to a benchmark. This is the V1 implementation.

    The split is defined as a (train, test) tuple, where train is a list of indices and
    test is a dictionary that maps test set names to lists of indices.

    Warning: Scalability
        The simple list-based representation we use for the split in this first implementation doesn't scale well.
        We therefore worked on a V2 implementation that uses roaring bitmaps.
        See [`SplitSpecificationV2Mixin`][`polaris.experimental._split_v2.SplitSpecificationV2Mixin`] for more details.

    Attributes:
        split: The predefined train-test split to use for evaluation.
    """

    split: SplitType

    @model_validator(mode="after")
    def _validate_split(self) -> Self:
        """
        Verifies that:
          1) There are no empty test partitions
          2) There is no overlap between the train and test set
          3) There is no duplicate indices in any of the sets
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

        return self

    @field_serializer("split")
    def _serialize_split(self, v: SplitType):
        """Convert any tuple to list to make sure it's serializable"""
        return listit(v)

    @computed_field
    @property
    def test_set_sizes(self) -> dict[str, int]:
        """The sizes of the test sets."""
        return {k: len(v) for k, v in self.split[1].items()}

    @computed_field
    @property
    def n_test_sets(self) -> int:
        """The number of test sets"""
        return len(self.split[1])

    @computed_field
    @property
    def n_train_datapoints(self) -> int:
        """The size of the train set."""
        return len(self.split[0])

    @computed_field
    @property
    def test_set_labels(self) -> list[str]:
        """The labels of the test sets."""
        return sorted(list(self.split[1].keys()))

    @computed_field
    @property
    def n_test_datapoints(self) -> dict[str, int]:
        """The size of (each of) the test set(s)."""
        if self.n_test_sets == 1:
            return {"test": len(self.split[1]["test"])}
        else:
            return {k: len(v) for k, v in self.split[1].items()}
