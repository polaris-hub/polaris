from pydantic import computed_field, field_validator

from polaris.benchmark._base import BenchmarkSpecification
from polaris.experimental._benchmark_v2 import BenchmarkV2Specification
from polaris.utils.types import TaskType


class SingleTaskMixin:
    """Mixin for single-task benchmarks."""

    @field_validator("target_cols", check_fields=False)
    def validate_target_cols(cls, v):
        if len(v) != 1:
            raise ValueError("A single-task benchmark should specify exactly one target column.")
        return v

    @computed_field
    @property
    def task_type(self) -> str:
        """Return SINGLE_TASK for single-task benchmarks."""
        return TaskType.SINGLE_TASK.value


class MultiTaskMixin:
    """Mixin for multi-task benchmarks."""

    @field_validator("target_cols", check_fields=False)
    def validate_target_cols(cls, v):
        if len(v) <= 1:
            raise ValueError("A multi-task benchmark should specify at least two target columns.")
        return v

    @computed_field
    @property
    def task_type(self) -> str:
        """Return MULTI_TASK for multi-task benchmarks."""
        return TaskType.MULTI_TASK.value


class SingleTaskBenchmarkSpecification(SingleTaskMixin, BenchmarkSpecification):
    """Single-task benchmark for the base specification."""

    pass


class MultiTaskBenchmarkSpecification(MultiTaskMixin, BenchmarkSpecification):
    """Multi-task benchmark for the base specification."""

    pass


class SingleTaskBenchmarkV2Specification(SingleTaskMixin, BenchmarkV2Specification):
    """Single-task benchmark for version 2."""

    pass


class MultiTaskBenchmarkV2Specification(MultiTaskMixin, BenchmarkV2Specification):
    """Multi-task benchmark for version 2."""

    pass


class BenchmarkFactory:
    @staticmethod
    def create_benchmark(version: int, task_type: str, **kwargs) -> BenchmarkSpecification:
        if version == 1:
            if task_type == "single":
                return SingleTaskBenchmarkSpecification(**kwargs)
            elif task_type == "multi":
                return MultiTaskBenchmarkSpecification(**kwargs)
        elif version == 2:
            if task_type == "single":
                return SingleTaskBenchmarkV2Specification(**kwargs)
            elif task_type == "multi":
                return MultiTaskBenchmarkV2Specification(**kwargs)
        raise ValueError(f"Unsupported version ({version}) or task_type ({task_type}).")
