from typing import Collection

from pydantic import computed_field, field_validator

from polaris.benchmark._base import BenchmarkV1Specification
from polaris.utils.types import TaskType


class SingleTaskMixin:
    """
    Mixin for single-task benchmarks.
    """

    @field_validator("target_cols", check_fields=False)
    @classmethod
    def validate_target_cols(cls, v: Collection[str]) -> Collection[str]:
        if len(v) != 1:
            raise ValueError("A single-task benchmark should specify exactly one target column.")
        return v

    @computed_field
    @property
    def task_type(self) -> str:
        """Return SINGLE_TASK for single-task benchmarks."""
        return TaskType.SINGLE_TASK.value


class MultiTaskMixin:
    """
    Mixin for multi-task benchmarks.
    """

    @field_validator("target_cols", check_fields=False)
    @classmethod
    def validate_target_cols(cls, v: Collection[str]) -> Collection[str]:
        if len(v) <= 1:
            raise ValueError("A multi-task benchmark should specify at least two target columns.")
        return v

    @computed_field
    @property
    def task_type(self) -> str:
        """
        Return MULTI_TASK for multi-task benchmarks.
        """
        return TaskType.MULTI_TASK.value


class SingleTaskBenchmarkSpecification(SingleTaskMixin, BenchmarkV1Specification):
    """
    Single-task benchmark for the base specification.
    """

    pass


class MultiTaskBenchmarkSpecification(MultiTaskMixin, BenchmarkV1Specification):
    """
    Multitask benchmark for the base specification.
    """

    pass
