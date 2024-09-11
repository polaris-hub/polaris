from datetime import datetime
from typing import Optional

from polaris.benchmark import BenchmarkSpecification
from polaris.evaluate import CompetitionPredictions
from polaris.hub.settings import PolarisHubSettings
from polaris.utils.types import HubOwner


class CompetitionSpecification(BenchmarkSpecification):
    """Much of the underlying data model and logic is shared across Benchmarks and Competitions, and
    anything within this class serves as a point of differentiation between the two.

    Attributes:
        owner: A slug-compatible name for the owner of the competition. This is redefined such
            that it is required.
        start_time: The time at which the competition becomes active and interactable.
        end_time: The time at which the competition ends and is no longer interactable.
    """

    _artifact_type = "competition"

    # Additional properties specific to Competitions
    owner: HubOwner
    start_time: datetime | None = None
    end_time: datetime | None = None

    def evaluate(
        self,
        predictions: CompetitionPredictions,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        **kwargs: dict,
    ):
        """Light convenience wrapper around
        [`PolarisHubClient.evaluate_competition`][polaris.hub.client.PolarisHubClient.evaluate_competition].
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(
            settings=settings,
            cache_auth_token=cache_auth_token,
            **kwargs,
        ) as client:
            return client.evaluate_competition(self, predictions)
