from typing import Any, Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field, PrivateAttr, validator

from polaris.evaluate._metric import METRICS_REGISTRY
from polaris.hub import PolarisClient
from polaris.utils.errors import InvalidResultError


class BenchmarkResults(BaseModel):
    """
    Base class for saving benchmarking results

    TODO (cwognum): An open question is how to best categorize a methodology (e.g. a model).
      This is needed since we want to be able to aggregate results across benchmarks for a model too.
    """

    """
    Benchmark results are stored as a dictionary
    """
    results: dict

    """
    The benchmark these results were generated for
    """
    benchmark_id: str

    """
    The name to identify the results by.
    If not specified, this is given a default value which can be edited later through the Hub.
    """
    name: Optional[str] = None

    """
    Tags to categorize the results by.
    """
    tags: dict = Field(default_factory=list)

    """
    User attributes allow for additional meta-data to be stored
    If users repeatedly specify the same attribute, we can extract it into an additional field
    """
    user_attributes: dict = Field(default_factory=dict)

    """
    The user associated with the results. Automatically set.
    """
    _user_name: Optional[str] = PrivateAttr(default_factory=PolarisClient.get_client().get_active_user)

    """
    The time-stamp at which the results were created. Automatically set.
    """
    _created_at: datetime = Field(default_factory=datetime.now)

    @validator("results")
    def validate_results(cls, v):
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
        """Upload to the hub"""
        return PolarisClient.get_client().upload_results_to_hub(self)
