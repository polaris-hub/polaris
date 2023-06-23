from polaris.data.benchmark._mixins import SingleTaskMixin, MultiTaskMixin
from polaris.data.benchmark._base import BenchmarkSpecification


class SingleTaskBenchmarkSpecification(SingleTaskMixin, BenchmarkSpecification):
    pass


class MultiTaskBenchmarkSpecification(MultiTaskMixin, BenchmarkSpecification):
    pass
