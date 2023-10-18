import os

import pandas as pd

from polaris.evaluate._results import BenchmarkResults
from polaris.utils.types import HubOwner


def test_result_to_json(tmpdir: str, test_user_owner: HubOwner):
    scores = pd.DataFrame({"Test set": ["A"], "Target label": "B", "Metric": "C", "Score": 0.1})

    result = BenchmarkResults(
        name="test",
        description="Lorem ipsum!",
        tags=["test"],
        user_attributes={"key": "value"},
        owner=test_user_owner,
        results=scores,
        benchmark_name="my-benchmark",
        benchmark_owner=test_user_owner,
        github_url="https://github.com/",
        paper_url="https://chemrxiv.org/",
        contributors=["my-user", "other-user"],
    )

    path = os.path.join(tmpdir, "result.json")
    result.to_json(path)
    BenchmarkResults.from_json(path)
