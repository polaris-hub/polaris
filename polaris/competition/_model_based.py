from polaris.competition import CompetitionSpecification
from polaris.utils.types import HubOwner, SlugStringType


class ModelBasedCompetition(CompetitionSpecification):
    def submit_entry(self, model: SlugStringType, owner: HubOwner | str) -> None:
        """
        Convenient wrapper around the
        [`PolarisHubClient.submit_competition_model`][polaris.hub.client.PolarisHubClient.submit_competition_model] method.

        Args:
            model: The artifact id of the model to submit. The model must already exist in the Hub.
            owner: Which Hub user or organization owns the submission.
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient() as client:
            client.submit_competition_model(competition=self, competition_model=model, owner=owner)
