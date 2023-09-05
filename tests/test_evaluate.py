import os

from polaris.evaluate._metric import Metric
from polaris.evaluate._results import BenchmarkResults
from polaris.utils.types import HubOwner


def test_result_to_json(tmpdir: str):
    result = BenchmarkResults(
        name="test",
        description="Lorem ipsum!",
        tags=["test"],
        user_attributes={"key": "value"},
        owner=HubOwner(userId="my-user"),
        results={"test": {Metric.mean_absolute_error: 1.0}},
        benchmark_name="my-benchmark",
        benchmark_owner=HubOwner(userId="test-user"),
        github_url="https://github.com/",
        paper_url="https://chemrxiv.org/",
        contributors=["my-user", "other-user"],
    )

    path = os.path.join(tmpdir, "result.json")
    result.to_json(path)
    BenchmarkResults.from_json(path)
