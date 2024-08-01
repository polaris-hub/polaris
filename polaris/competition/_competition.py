from datetime import datetime
import os
from typing import Optional

from pydantic import field_serializer, field_validator, ValidationInfo
from polaris.benchmark import BenchmarkSpecification
from polaris.hub.settings import PolarisHubSettings
from polaris.utils.types import AccessType, HubOwner, PredictionsType, TimeoutTypes, ZarrConflictResolution
from polaris.utils.errors import InvalidCompetitionError


class CompetitionSpecification(BenchmarkSpecification):
    """This class extends the [`BenchmarkSpecification`][polaris.benchmark.BenchmarkSpecification] to
    facilitate interactions with Polaris Competitions.

    Much of the underlying data model and logic is shared across Benchmarks and Competitions, and
    anything within this class serves as a point of differentiation between the two.

    facilitate interactions with Polaris Competitions.

    Much of the underlying data model and logic is shared across Benchmarks and Competitions, and
    anything within this class serves as a point of differentiation between the two.

    Currently, these entities will primarily differ at how user predictions are evaluated.
    """

    # Additional properties specific to Competitions
    owner: HubOwner
    start_time: datetime | None = None
    scheduled_end_time: datetime | None = None
    actual_end_time: datetime | None = None

    @field_validator("split")
    def _validate_test_set(cls, split, info: ValidationInfo):
        """Verifies that the test does not have too many missing values. There must be at
        least one value per row in the test set across the target columns."""
        dataset = info.data.get("dataset")
        target_cols = info.data.get("target_cols")
        test_indices = split[1]

        if dataset.table.loc[test_indices, target_cols].notna().any(axis=1).all():
            return split
        else:
            raise InvalidCompetitionError("All rows of the test set must have at least one value.")

    def evaluate(
        self,
        y_pred: PredictionsType,
        result_name: str,
        env_file: str | os.PathLike | None = None,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        **kwargs: dict,
    ):
        """Light convenience wrapper around
        [`PolarisHubClient.evaluate_competition`][polaris.hub.client.PolarisHubClient.evaluate_competition].
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(
            env_file=env_file,
            settings=settings,
            cache_auth_token=cache_auth_token,
            **kwargs,
        ) as client:
            return client.evaluate_competition(self, y_pred=y_pred, result_name=result_name)

    def upload_to_hub(
        self,
        env_file: str | os.PathLike | None = None,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        access: AccessType = "private",
        timeout: TimeoutTypes = (10, 200),
        owner: HubOwner | str | None = None,
        if_exists: ZarrConflictResolution = "replace",
    ):
        """Very light, convenient wrapper around the
        [`PolarisHubClient.upload_competition`][polaris.hub.client.PolarisHubClient.upload_competition] method."""

        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(
            env_file=env_file,
            settings=settings,
            cache_auth_token=cache_auth_token,
        ) as client:
            return client.upload_competition(self.dataset, self, access, timeout, owner, if_exists)

    @field_serializer("start_time", "scheduled_end_time")
    def _serialize_start_date(self, v):
        """Convert from datetime to string to make sure it's serializable"""
        if v:
            return v.isoformat()
