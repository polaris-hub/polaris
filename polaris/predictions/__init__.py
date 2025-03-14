from datetime import datetime

from pandas import DataFrame
from pydantic import ConfigDict, Field, PrivateAttr
from pydantic.alias_generators import to_camel

from polaris._artifact import BaseArtifactModel
from polaris.benchmark import BenchmarkV2Specification
from polaris.model import Model
from polaris.utils.types import HubUser


class BenchmarkV2Prediction(BaseArtifactModel):
    """
    Uploadable predictions for benchmark evaluations.
    """
    model_config = ConfigDict(frozen=True, alias_generator=to_camel, arbitrary_types_allowed=True)

    _artifact_type = "benchmark-prediction"
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    contributors: list[HubUser] = Field(default_factory=list)
    model: Model | None = Field(None, exclude=True)
    benchmark: BenchmarkV2Specification = Field(
        description="The benchmark specification that these predictions were generated for."
    )
    split_label: str = Field(
        description="The label of the split specified in the benchmark that these predictions were generated with."
    )
    predictions: DataFrame = Field(description="A DataFrame with the predictions for each target column.")

    # TODO: Add upload to Hub function here
