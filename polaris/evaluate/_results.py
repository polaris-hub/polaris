from datetime import datetime
from typing import Any, Optional, Union

from pydantic import ConfigDict, Field, HttpUrl, PrivateAttr

from polaris._artifact import BaseArtifactModel
from polaris.evaluate._metric import Metric
from polaris.utils.misc import to_lower_camel
from polaris.utils.types import HubOwner

# Define some helpful type aliases
TestLabelType = str
TargetLabelType = str
MetricScoresType = dict[Metric, float]
ResultsType = Union[
    MetricScoresType, dict[TestLabelType, Union[MetricScoresType, dict[TargetLabelType, MetricScoresType]]]
]


class BenchmarkResults(BaseArtifactModel):
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
        benchmark_name: The name of the benchmark for which these results were generated.
            Together with the benchmark owner, this uniquely identifies the benchmark on the Hub.
        benchmark_owner: The owner of the benchmark for which these results were generated.
            Together with the benchmark name, this uniquely identifies the benchmark on the Hub.
        name: A slug-compatible name for the result.
            Together with the owner, this is used by the Hub to uniquely identify the result.
        description: A beginner-friendly, short description of the result.
        tags: A list of tags to categorize the result by. This is used by the hub to search over results.
        user_attributes: A dict with additional, textual user attributes.
        owner: A slug-compatible name for the owner of the result.
            If the result comes from the Polaris Hub, this is the associated owner (organization or user).
            Together with the name, this is used by the Hub to uniquely identify the result.
        github_url: The URL to the GitHub repository of the code used to generate these results.
        paper_url: The URL to the paper describing the methodology used to generate these results.
        _user_name: The user associated with the results. Automatically set.
        _created_at: The time-stamp at which the results were created. Automatically set.
    """

    # Data
    results: ResultsType
    benchmark_name: str = Field(..., frozen=True)
    benchmark_owner: Optional[HubOwner] = Field(None, frozen=True)

    # Additional meta-data
    github_url: Optional[HttpUrl] = None
    paper_url: Optional[HttpUrl] = None

    # Private attributes
    _user_name: Optional[str] = PrivateAttr(default=None)
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    model_config = ConfigDict(alias_generator=to_lower_camel, populate_by_name=True)

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.name is None:
            if self._user_name is None:
                self.name = str(self._created_at)
            else:
                self.name = f"{self._user_name}_{str(self._created_at)}"
