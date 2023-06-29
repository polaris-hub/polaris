from typing import Sequence

from pydantic import validator, root_validator
from polaris.utils.types import SplitIndicesType
from polaris.benchmark import BenchmarkSpecification


class SingleTaskBenchmarkSpecification(BenchmarkSpecification):
    """Subclass for any single-task benchmark specification"""

    @validator("target_cols", check_fields=False)
    def validate_target_cols(cls, v):
        if not len(v) == 1:
            raise ValueError("A single-task benchmark should specify a single target column")
        return v


class MultiTaskBenchmarkSpecification(BenchmarkSpecification):
    """Subclass for any multi-task benchmark specification"""

    @validator("target_cols", check_fields=False)
    def validate_target_cols(cls, v):
        if not len(v) > 1:
            raise ValueError("A multi-task benchmark should specify at least two target columns")
        return v
