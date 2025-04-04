from polaris.competition import CompetitionSpecification
from polaris.model import Model
from polaris.utils.types import SlugStringType
from polaris.utils.errors import (
    PolarisRetrieveArtifactError,
)


class ModelBasedCompetition(CompetitionSpecification):
    def submit_model(
        self,
        model: Model | SlugStringType,
        owner: str,
    ) -> None:
        """
        Convenient wrapper around the
        [`PolarisHubClient.submit_competition_model`][polaris.hub.client.PolarisHubClient.submit_competition_model] method.

        Args:
            model: The model to submit.
            owner: The user submitting the model.
        """
        from polaris.hub.client import PolarisHubClient

        # do we need to wrap in a progress indicator?

        with PolarisHubClient() as client:

            # if artifact id, load model
            if isinstance(model, str):
                print("this is an artifact id", model)
                # try:
                #     model = client.get_model(model)
                #     model_id = model.artifact_id

                # except PolarisRetrieveArtifactError: # check if this is needed, get_model might throw an error already
                #     raise PolarisRetrieveArtifactError(f"Model {model} not found")

            # if not artifact id, upload model
            else:
                print("this is not an artifact id", model)
                #     model = client.upload_model(model, owner=owner) # need to somehow make this return the artifact id
                #     model_id = model.artifact_id

            # client.submit_competition_model(competition=self, competition_model=model_id)