from polaris.competition import CompetitionSpecification
from polaris.evaluate import CompetitionPredictions
from polaris.utils.types import (
    HttpUrlString,
    HubOwner,
    HubUser,
    IncomingPredictionsType,
    SlugCompatibleStringType,
)


class PredictionBasedCompetition(CompetitionSpecification):
    def submit_predictions(
        self,
        predictions: IncomingPredictionsType,
        prediction_name: SlugCompatibleStringType,
        prediction_owner: str,
        report_url: HttpUrlString,
        contributors: list[HubUser] | None = None,
        github_url: HttpUrlString | None = None,
        description: str = "",
        tags: list[str] | None = None,
        user_attributes: dict[str, str] | None = None,
    ) -> None:
        """
        Convenient wrapper around the
        [`PolarisHubClient.submit_competition_predictions`][polaris.hub.client.PolarisHubClient.submit_competition_predictions] method.
        It handles the creation of a standardized predictions object, which is expected by the Hub, automatically.

        Args:
            prediction_name: The name of the prediction.
            prediction_owner: The slug of the user/organization which owns the prediction.
            predictions: The predictions for each test set defined in the competition.
            report_url: A URL to a report/paper/write-up which describes the methods used to generate the predictions.
            contributors: The users credited with generating these predictions.
            github_url: An optional URL to a code repository containing the code used to generated these predictions.
            description: An optional and short description of the predictions.
            tags: An optional list of tags to categorize the prediction by.
            user_attributes: An optional dict with additional, textual user attributes.
        """
        from polaris.hub.client import PolarisHubClient

        standardized_predictions = CompetitionPredictions(
            name=prediction_name,
            owner=HubOwner(slug=prediction_owner),
            predictions=predictions,
            report_url=report_url,
            contributors=contributors or [],
            github_url=github_url,
            description=description,
            tags=tags or [],
            user_attributes=user_attributes or {},
            target_labels=self.target_cols,
            test_set_labels=self.test_set_labels,
            test_set_sizes=self.test_set_sizes,
        )

        with PolarisHubClient() as client:
            client.submit_competition_predictions(
                competition=self, competition_predictions=standardized_predictions
            )
