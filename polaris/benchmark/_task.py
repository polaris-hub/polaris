from typing import Sequence

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.types import ColumnName, TargetType, TaskType


class PredictiveTaskSpecificationMixin(BaseModel):
    """A mixin for task benchmarks without metrics.

    Attributes:
        target_cols: The column(s) of the original dataset that should be used as the target.
        input_cols: The column(s) of the original dataset that should be used as input.
        target_types: A dictionary that maps target columns to their type. If not specified, this is automatically inferred.
    """

    target_cols: set[ColumnName] = Field(min_length=1)
    input_cols: set[ColumnName] = Field(min_length=1)
    target_types: dict[ColumnName, TargetType] = Field(default_factory=dict, validate_default=True)

    @field_validator("target_cols", "input_cols", mode="before")
    @classmethod
    def _parse_cols(cls, v: str | Sequence[str], info: ValidationInfo) -> set[str]:
        """
        Normalize columns input values to a set.
        """
        if isinstance(v, str):
            v = {v}
        else:
            v = set(v)
        return v

    @field_validator("target_types", mode="before")
    @classmethod
    def _parse_target_types(
        cls, v: dict[ColumnName, TargetType | str | None]
    ) -> dict[ColumnName, TargetType]:
        """
        Converts the target types to TargetType enums if they are strings.
        """
        return {
            target: TargetType(val) if isinstance(val, str) else val
            for target, val in v.items()
            if val is not None
        }

    @model_validator(mode="after")
    def _validate_target_types(self) -> Self:
        """
        Verifies that all target types are for benchmark targets.
        """
        columns = set(self.target_types.keys())
        if not columns.issubset(self.target_cols):
            raise InvalidBenchmarkError(
                f"Not all specified target types were found in the target columns. {columns} - {self.target_cols}"
            )
        return self

    @field_serializer("target_types")
    def _serialize_target_types(self, target_types):
        """
        Convert from enum to string to make sure it's serializable
        """
        return {k: v.value for k, v in target_types.items()}

    @field_serializer("target_cols", "input_cols")
    def _serialize_columns(self, v: set[str]) -> list[str]:
        return list(v)

    @computed_field
    @property
    def task_type(self) -> str:
        """The high-level task type of the benchmark."""
        v = TaskType.MULTI_TASK if len(self.target_cols) > 1 else TaskType.SINGLE_TASK
        return v.value
