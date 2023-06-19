from polaris.data.benchmark._split_mixins import SingleTaskSplitMixin, MultiTaskSplitMixin
from polaris.data.benchmark._base import BenchmarkSpecification


class SingleTaskBenchmarkSpecification(SingleTaskSplitMixin, BenchmarkSpecification):
    pass


class MultiTaskBenchmarkSpecification(MultiTaskSplitMixin, BenchmarkSpecification):
    pass
