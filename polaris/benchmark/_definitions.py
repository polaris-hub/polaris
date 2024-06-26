from pydantic import computed_field, field_validator

from polaris.benchmark._base import BenchmarkSpecification
from polaris.utils.types import TaskType


class SingleTaskBenchmarkSpecification(BenchmarkSpecification):
    """Subclass for any single-task benchmark specification

    In addition to the data-model and logic of the base-class,
    this class verifies that there is just a single target-column.
    """

    @field_validator("target_cols", check_fields=False)
    def validate_target_cols(cls, v):
        if not len(v) == 1:
            raise ValueError("A single-task benchmark should specify a single target column")
        return v

    @computed_field
    @property
    def task_type(self) -> str:
        """The high-level task type of the benchmark."""
        return TaskType.SINGLE_TASK.value


class MultiTaskBenchmarkSpecification(BenchmarkSpecification):
    """Subclass for any multi-task benchmark specification

    In addition to the data-model and logic of the base-class,
    this class verifies that there are multiple target-columns.
    """

    @field_validator("target_cols", check_fields=False)
    def validate_target_cols(cls, v):
        if not len(v) > 1:
            raise ValueError("A multi-task benchmark should specify at least two target columns")
        return v

    @computed_field
    @property
    def task_type(self) -> str:
        """The high-level task type of the benchmark."""
        return TaskType.MULTI_TASK.value
