from pydantic import field_validator

from polaris.benchmark._base import BenchmarkSpecification


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
