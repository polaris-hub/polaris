from datetime import datetime
from typing import Optional, Union

from pydantic import ConfigDict, Field, HttpUrl, PrivateAttr, field_serializer

from polaris._artifact import BaseArtifactModel
from polaris.evaluate._metric import Metric
from polaris.utils.misc import to_lower_camel
from polaris.utils.types import HubOwner, HubUser

# Define some helpful type aliases
TestLabelType = str
TargetLabelType = str
MetricScoresType = dict[Union[str, Metric], float]
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
        github_url: The URL to the GitHub repository of the code used to generate these results.
        paper_url: The URL to the paper describing the methodology used to generate these results.
        contributors: The users that are credited for these results.
        _created_at: The time-stamp at which the results were created. Automatically set.
    For additional meta-data attributes, see the [`BaseArtifactModel`][polaris._artifact.BaseArtifactModel] class.
    """

    # Data
    results: ResultsType
    benchmark_name: str = Field(..., frozen=True)
    benchmark_owner: Optional[HubOwner] = Field(None, frozen=True)

    # Additional meta-data
    github_url: Optional[HttpUrl] = None
    paper_url: Optional[HttpUrl] = None
    contributors: Optional[list[HubUser]] = None

    # Private attributes
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    model_config = ConfigDict(alias_generator=to_lower_camel, populate_by_name=True)

    @field_serializer("results")
    def serialize_results(self, value: ResultsType):
        """Change from the Metric enum to a string representation"""

        def _recursive_enum_to_str(d: dict):
            """Utility function to easily traverse the nested dictionary"""
            if not isinstance(d, dict):
                return d
            return {k.name if isinstance(k, Metric) else k: _recursive_enum_to_str(v) for k, v in d.items()}

        return _recursive_enum_to_str(value)

    @field_serializer("github_url", "paper_url")
    def serialize_urls(self, value: HttpUrl):
        if value is not None:
            value = str(value)
        return value
