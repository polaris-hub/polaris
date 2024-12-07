from datetime import datetime
from typing import ClassVar

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel

from polaris._artifact import BaseArtifactModel
from polaris.evaluate import BenchmarkPredictions
from polaris.evaluate._metric import MetricType, instantiate_metric
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidResultError
from polaris.utils.misc import slugify
from polaris.utils.types import (
    AccessType,
    HttpUrlString,
    HubOwner,
    HubUser,
    SlugCompatibleStringType,
)

# Define some helpful type aliases
TestLabelType = str
TargetLabelType = str


class ResultRecords(BaseModel):
    """
    A grouped, tabular data-structure to save the actual data of the benchmark results.
    The grouping by test set and target label is done to make it easier to build leaderboards.
    """

    test_set: TestLabelType
    target_label: TargetLabelType
    scores: list[tuple[MetricType, float]]

    # Model config
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @field_validator("scores", mode="before")
    @classmethod
    def validate_scores(cls, v):
        validated = []
        for metric, score in v:
            metric = instantiate_metric(metric)
            validated.append((metric, score))
        return validated

    @field_serializer("scores")
    def serialize_scores(self, value: list[tuple[MetricType, float]]) -> dict[dict, float]:
        """Change from the Metric enum to a string representation"""
        return [(metric.model_dump(), score) for metric, score in value]


class ResultsMetadata(BaseArtifactModel):
    """Base class for evaluation results

    Attributes:
        github_url: The URL to the GitHub repository of the code used to generate these results.
        paper_url: The URL to the paper describing the methodology used to generate these results.
        contributors: The users that are credited for these results.
        _created_at: The time-stamp at which the results were created. Automatically set.
    For additional meta-data attributes, see the [`BaseArtifactModel`][polaris._artifact.BaseArtifactModel] class.
    """

    # Additional meta-data
    github_url: HttpUrlString | None = None
    paper_url: HttpUrlString | None = None
    contributors: list[HubUser] = Field(default_factory=list)

    # Private attributes
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    def _repr_html_(self) -> str:
        """For pretty-printing in Jupyter Notebooks"""
        return dict2html(self.model_dump())

    def __repr__(self):
        return self.model_dump_json(indent=2)


class EvaluationResult(ResultsMetadata):
    """Class for saving evaluation results

    The actual results are saved in the `results` field using the following tabular format:

    | Test set | Target label | Metric | Score |
    | -------- | ------------ | ------ | ----- |
    | test_iid | EGFR_WT      | AUC    | 0.9   |
    | test_ood | EGFR_WT      | AUC    | 0.75  |
    |    ...   |      ...     |  ...   |  ...  |
    | test_ood | EGFR_L858R   | AUC    | 0.79  |

    question: Categorizing methods
        An open question is how to best categorize a methodology (e.g. a model).
        This is needed since we would like to be able to aggregate results across benchmarks/competitions too,
        to say something about which (type of) methods performs best _in general_.

    Attributes:
        results: Evaluation results are stored directly in a dataframe or in a serialized, JSON compatible dict
            that can be decoded into the associated tabular format.
    For additional meta-data attributes, see the [`ResultsMetadata`][polaris.evaluate._results.ResultsMetadata] class.
    """

    # Define the columns of the results table
    RESULTS_COLUMNS: ClassVar[list[str]] = ["Test set", "Target label", "Metric", "Score"]

    # Results attribute
    results: pd.DataFrame

    @field_validator("results", mode="before")
    @classmethod
    def _convert_results(cls, v: pd.DataFrame | list[ResultRecords | dict]) -> pd.DataFrame:
        """
        Convert the results to a dataframe if they are not already in that format
        """
        if isinstance(v, pd.DataFrame):
            return v

        try:
            df = pd.DataFrame(columns=cls.RESULTS_COLUMNS)
            for record in v:
                if isinstance(record, dict):
                    record = ResultRecords(**record)

                for metric, score in record.scores:
                    df.loc[len(df)] = {
                        "Test set": record.test_set,
                        "Target label": record.target_label,
                        "Metric": metric,
                        "Score": score,
                    }
            return df

        except (ValueError, UnicodeDecodeError) as error:
            raise InvalidResultError(
                f"The provided dictionary cannot be parsed into a {cls.__name__} instance."
            ) from error

    @field_validator("results")
    @classmethod
    def _validate_results(cls, v: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the results are a valid dataframe and have the expected columns
        """
        # Check if the dataframe contains _only_ the expected columns
        if set(v.columns) != set(cls.RESULTS_COLUMNS):
            raise InvalidResultError(
                f"The results dataframe should have the following columns: {cls.RESULTS_COLUMNS}"
            )

        # Check if the results are not empty
        if v.empty:
            raise InvalidResultError("The results dataframe is empty")

        # NOTE (cwognum): Since we have a reference to the benchmark, I considered validating the values in the
        #  columns as well (e.g. are all metrics, targets and test sets actually part of the benchmark).
        #  However, to keep this class light-weight, I did not want to add a strict dependency on the full benchmark class.
        #  Especially because validation will happen on the Hub as well before it is shown there.
        return v

    @field_serializer("results")
    def _serialize_results(self, value: pd.DataFrame) -> list[ResultRecords]:
        """Change from the Metric enum to a string representation"""
        serialized = []
        grouped = value.groupby(["Test set", "Target label"])
        for (test_set, target_label), group in grouped:
            metrics = [(row["Metric"], row["Score"]) for _, row in group.iterrows()]
            record = ResultRecords(test_set=test_set, target_label=target_label, scores=metrics)
            serialized.append(record)

        return serialized


class BenchmarkResults(EvaluationResult):
    """Class specific to results for standard benchmarks.

    This object is returned by [`BenchmarkSpecification.evaluate`][polaris.benchmark.BenchmarkSpecification.evaluate].
    In addition to the metrics on the test set, it contains additional meta-data and logic to integrate
    the results with the Polaris Hub.

    benchmark_name: The name of the benchmark for which these results were generated.
        Together with the benchmark owner, this uniquely identifies the benchmark on the Hub.
    benchmark_owner: The owner of the benchmark for which these results were generated.
        Together with the benchmark name, this uniquely identifies the benchmark on the Hub.
    """

    _artifact_type = "result"

    benchmark_artifact_id: str | None = Field(None)
    benchmark_name: SlugCompatibleStringType | None = Field(None, deprecated=True)
    benchmark_owner: HubOwner | None = Field(None, deprecated=True)

    @model_validator(mode="after")
    def set_benchmark_artifact_id(self):
        if self.benchmark_artifact_id is None and self.benchmark_name and self.benchmark_owner:
            self.benchmark_artifact_id = f"{self.benchmark_owner}/{slugify(self.benchmark_name)}"
        return self

    def upload_to_hub(
        self,
        access: AccessType = "private",
        owner: HubOwner | str | None = None,
        **kwargs: dict,
    ) -> "BenchmarkResults":
        """
        Very light, convenient wrapper around the
        [`PolarisHubClient.upload_results`][polaris.hub.client.PolarisHubClient.upload_results] method.
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(**kwargs) as client:
            return client.upload_results(self, access=access, owner=owner)


class CompetitionResults(EvaluationResult):
    """Class specific to results for competition benchmarks.

    This object is returned by [`CompetitionSpecification.evaluate`][polaris.competition.CompetitionSpecification.evaluate].
    In addition to the metrics on the test set, it contains additional meta-data and logic to integrate
    the results with the Polaris Hub.

    Attributes:
        competition_name: The name of the competition for which these results were generated.
            Together with the competition owner, this uniquely identifies the competition on the Hub.
        competition_owner: The owner of the competition for which these results were generated.
            Together with the competition name, this uniquely identifies the competition on the Hub.
    """

    _artifact_type = "competitionResult"

    competition_name: SlugCompatibleStringType = Field(..., frozen=True)
    competition_owner: HubOwner | None = Field(None, frozen=True)

    @computed_field
    @property
    def competition_artifact_id(self) -> str:
        return f"{self.competition_owner}/{slugify(self.competition_name)}"


class CompetitionPredictions(ResultsMetadata, BenchmarkPredictions):
    """
    Predictions for competition benchmarks.

    This object is to be used as input to [`CompetitionSpecification.evaluate`][polaris.competition.CompetitionSpecification.evaluate].
    It is used to ensure that the structure of the predictions are compatible with evaluation methods on the Polaris Hub.
    In addition to the predictions, it contains additional meta-data to create a results object.

    Attributes:
        access: The access the returned results should have
    """

    access: AccessType = "private"
