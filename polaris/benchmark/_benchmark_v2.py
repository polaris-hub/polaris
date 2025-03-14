from typing import Any, Callable, ClassVar, Generator, Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from polaris.benchmark._base import BenchmarkSpecification
from polaris.benchmark._split_v2 import SplitV2
from polaris.dataset import DatasetV2, Subset
from polaris.dataset._subset import SubsetV2
from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.types import ColumnName


class BenchmarkV2Specification(BenchmarkSpecification):
    _version: ClassVar[Literal[2]] = 2

    dataset: DatasetV2 = Field(exclude=True)
    n_classes: dict[ColumnName, int] = Field(default_factory=dict)

    splits: list[SplitV2] = Field(
        default_factory=list, description="The predefined splits for this benchmark."
    )

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
    def _validate_split_indices_in_dataset(self) -> Self:
        """
        Verifies that:
          - All indices are valid given the dataset
        """
        dataset_length = len(self.dataset)

        if max(split.max_index for split in self.splits) >= dataset_length:
            raise InvalidBenchmarkError("The predefined splits contain invalid indices")

        return self

    def get_train_test_splits(
        self, featurization_fn: Callable | None = None
    ) -> Generator[tuple[Subset, Subset], None, None]:
        """Construct the train and test sets, for each split in the benchmark.

        Returns [`Subset`][polaris.dataset.Subset] objects, which offer several ways of accessing the data
        and can thus easily serve as a basis to build framework-specific (e.g. PyTorch, Tensorflow)
        data-loaders on top of.

        Args:
            featurization_fn: A function to apply to the input data. If a multi-input benchmark, this function
                expects an input in the format specified by the `input_format` parameter.

        Returns:
            A generator containing tuples with the train `Subset` and test `Subset` objects.
        """
        for split in self.splits:
            train_split = SubsetV2(
                dataset=self.dataset,
                indices=split.train.indices,
                input_cols=self.input_cols,
                target_cols=self.target_cols,
                hide_targets=False,
                featurization_fn=featurization_fn,
            )
            test_split = SubsetV2(
                dataset=self.dataset,
                indices=split.test.indices,
                input_cols=self.input_cols,
                target_cols=self.target_cols,
                hide_targets=False,
                featurization_fn=featurization_fn,
            )
            yield train_split, test_split
