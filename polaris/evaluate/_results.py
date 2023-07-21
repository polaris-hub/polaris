from typing import Any, Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from polaris.evaluate._metric import METRICS_REGISTRY
from polaris.hub import PolarisClient
from polaris.utils.errors import InvalidResultError


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
    results: dict
    benchmark_id: str
    name: Optional[str] = None
    tags: dict = Field(default_factory=list)
    user_attributes: dict = Field(default_factory=dict)

    # Private attributes
    _user_name: Optional[str] = PrivateAttr(default_factory=PolarisClient.get_client().get_active_user)
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    @field_validator("results")
    def _validate_results(cls, v):
        """Checks if all metrics are valid and if all scores are floats"""

        def _find_lowest_level_dicts(d: Dict) -> List[Dict]:
            """Helper function to find lowest-level dictionaries in a hierarchy of dicts"""
            ret = []
            if isinstance(d, dict) and not any(isinstance(vv, dict) for vv in d.values()):
                return [d]
            if isinstance(d, dict):
                for kk, vv in d.items():
                    ret.extend(_find_lowest_level_dicts(vv))
                return ret
            return ret

        low_level_dicts = _find_lowest_level_dicts(v)
        for low_level_dict in low_level_dicts:
            for metric, score in low_level_dict.items():
                if metric not in METRICS_REGISTRY:
                    raise InvalidResultError(f"{metric} is not a supported metric in the Polaris framework.")
                if not isinstance(score, float):
                    raise InvalidResultError(f"Scores must be floats. Got {type(score)} for metric {metric}")
            return v

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
