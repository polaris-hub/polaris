from polaris.competition import CompetitionSpecification
from polaris.model import Model
from polaris.utils.types import HubOwner, SlugStringType


class ModelBasedCompetition(CompetitionSpecification):
    def submit_entry(
        self,
        model: Model | SlugStringType,
        owner: HubOwner | str
    ) -> None:
        """Implementation for the submit_entry abstract method"""
        self.submit_model(model, owner)

    def submit_model(
        self,
        model: Model | SlugStringType,
        owner: HubOwner | str
    ) -> None:
        """
        Convenient wrapper around the
        [`PolarisHubClient.submit_competition_model`][polaris.hub.client.PolarisHubClient.submit_competition_model] method.

        Args:
            model: The model to submit. Can either be the artifact id of a model, or a model object.
            owner: Which Hub user or organization owns the submission.
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient() as client:
            client.submit_competition_model(competition=self, competition_model=model, owner=owner)