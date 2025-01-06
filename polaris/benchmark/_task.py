from typing import Collection, Sequence

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

from polaris.evaluate import Metric
from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.types import ColumnName, TargetType, TaskType


class PredictiveTaskSpecificationMixin(BaseModel):
    """A mixin for predictive task benchmarks.

    Attributes:
        target_cols: The column(s) of the original dataset that should be used as the target.
        input_cols: The column(s) of the original dataset that should be used as input.
        metrics: The metrics to use for evaluating performance.
        main_metric: The main metric used to rank methods. If `None`, this defaults to the first of the `metrics` field.
        target_types: A dictionary that maps target columns to their type. If not specified, this is automatically inferred.
    """

    target_cols: set[ColumnName] = Field(min_length=1)
    input_cols: set[ColumnName] = Field(min_length=1)
    metrics: set[Metric] = Field(min_length=1)
    main_metric: Metric | str
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

    @field_validator("metrics", mode="before")
    @classmethod
    def _validate_metrics(cls, v: str | Metric | Collection[str | Metric]) -> set[Metric]:
        """
        Verifies all specified metrics are either a Metric object or a valid metric name.
        Also verifies there are no duplicate metrics.

        If there are multiple test sets, it is assumed the same metrics are used across test sets.
        """
        if isinstance(v, str):
            v = {"label": v}
        if not isinstance(v, Collection):
            v = [v]

        def _convert(m: str | dict | Metric) -> Metric:
            if isinstance(m, str):
                return Metric(label=m)
            if isinstance(m, dict):
                return Metric(**m)
            return m

        v = [_convert(m) for m in v]

        unique_metrics = set(v)

        if len(unique_metrics) != len(v):
            raise InvalidBenchmarkError("The benchmark specifies duplicate metrics.")

        unique_names = {m.name for m in unique_metrics}
        if len(unique_names) != len(unique_metrics):
            raise InvalidBenchmarkError(
                "The metrics of a benchmark need to have unique names. Specify a custom name with Metric(custom_name=...)"
            )

        return unique_metrics

    @model_validator(mode="after")
    def _validate_main_metric_is_in_metrics(self) -> Self:
        if isinstance(self.main_metric, str):
            for m in self.metrics:
                if m.name == self.main_metric:
                    self.main_metric = m
                    break
        if self.main_metric not in self.metrics:
            raise InvalidBenchmarkError("The main metric should be one of the specified metrics")
        return self

    @field_serializer("metrics")
    def _serialize_metrics(self, value: set[Metric]) -> list[Metric]:
        """
        Convert the set to a list. Since metrics are models and will be converted to dict,
        they will not be hashable members of a set.
        """
        return list(value)

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

    @field_serializer("main_metric")
    def _serialize_main_metric(value: Metric) -> str:
        """
        Convert the Metric to it's name
        """
        return value.name

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
