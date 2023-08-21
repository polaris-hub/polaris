from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, PrivateAttr

from polaris.evaluate._metric import Metric
from polaris.hub import PolarisClient

# Define some helpful type aliases
TestLabel = str
TargetLabel = str
MetricScores = dict[Metric, float]
Results = MetricScores | dict[TestLabel, MetricScores | dict[TargetLabel, MetricScores]]


class BenchmarkResults(BaseModel):
    """Class for saving benchmarking results

    This object is returned by [`BenchmarkSpecification.evaluate`][polaris.benchmark.BenchmarkSpecification.evaluate].
    In addition to the metrics on the test set, it contains additional meta-data and logic to integrate
    the results with the Polaris Hub.

    question: Categorizing methods
        An open question is how to best categorize a methodology (e.g. a model).
        This is needed since we would like to be able to aggregate results across benchmarks too,
        to say something about which (type of) methods performs best _in general_.

    Attributes:
        results: Benchmark results are stored as a dictionary
        benchmark_id: The benchmark these results were generated for
        name: The name to identify the results by.
        tags: Tags to categorize the results by.
        user_attributes: User attributes allow for additional meta-data to be stored
        _user_name: The user associated with the results. Automatically set.
        _created_at: The time-stamp at which the results were created. Automatically set.
    """

    # Public attributes
    results: Results
    benchmark_id: str
    name: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    user_attributes: dict[str, str] = Field(default_factory=dict)

    # Private attributes
    _user_name: Optional[str] = PrivateAttr(default_factory=PolarisClient.get_client().get_active_user)
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.name is None:
            if self._user_name is None:
                self.name = str(self._created_at)
            else:
                self.name = f"{self._user_name}_{str(self._created_at)}"

    def upload_to_hub(self):
        """Upload the results to the hub

        This will upload the results to your account in the Polaris Hub. By default, these results are private.
        If you want to share them, you can do so in your account. This might trigger a review process.
        """
        return PolarisClient.get_client().upload_results_to_hub(self)
