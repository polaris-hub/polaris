from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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
        owner: If the dataset comes from the Polaris Hub, this is the associated owner (organization or user).
        _user_name: The user associated with the results. Automatically set.
        _created_at: The time-stamp at which the results were created. Automatically set.
    """

    # Public attributes
    results: ResultsType
    benchmark_name: str = Field(..., frozen=True)
    benchmark_owner: Optional[HubOwner] = Field(None, frozen=True)
    name: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    user_attributes: dict[str, str] = Field(default_factory=dict)
    owner: Optional[HubOwner] = None

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
