from collections import defaultdict
from datetime import datetime
from itertools import chain
from typing import Callable

from pydantic import Field, computed_field, model_validator
from typing_extensions import Self

from polaris.benchmark._split import SplitSpecificationV1Mixin
from polaris.benchmark._task import PredictiveTaskSpecificationMixin
from polaris.dataset import DatasetV2, Subset
from polaris.evaluate import CompetitionPredictions
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidCompetitionError
from polaris.utils.types import (
    ColumnName,
    HttpUrlString,
    HubOwner,
    HubUser,
    IncomingPredictionsType,
    SlugCompatibleStringType,
)


class CompetitionSpecification(DatasetV2, PredictiveTaskSpecificationMixin, SplitSpecificationV1Mixin):
    """An instance of this class represents a Polaris competition.

    Examples:
        Basic API usage:
        ```python
        import polaris as po

        # Load the benchmark from the Hub
        competition = po.load_competition("dummy-user/dummy-name")

        # Get the train and test data-loaders
        train, test = competition.get_train_test_split()

        # Use the training data to train your model
        # Get the input as an array with 'train.inputs' and 'train.targets'
        # Or simply iterate over the train object.
        for x, y in train:
            ...

        # Work your magic to accurately predict the test set
        prediction_values = np.array([0.0 for x in test])

        # Submit your predictions
        competition.submit_predictions(
            prediction_name="first-prediction",
            prediction_owner="dummy-user",
            report_url="REPORT_URL",
            predictions=prediction_values,
        )
        ```

    Attributes:
        start_time: The time at which the competition starts accepting prediction submissions.
        end_time: The time at which the competition stops accepting prediction submissions.
        n_classes: The number of classes within each target column that defines a classification task.

    For additional metadata attributes, see the base classes.
    """

    _artifact_type = "competition"

    dataset: None = None

    start_time: datetime
    end_time: datetime
    n_classes: dict[ColumnName, int | None] = Field(..., default_factory=lambda: defaultdict(None))

    @model_validator(mode="after")
    def _validate_split_in_dataset(self) -> Self:
        """
        All indices are valid given the dataset. We check the len of `self` here because a
        competition entity includes both the dataset and benchmark in one artifact.
        """
        max_i = len(self)
        if any(i < 0 or i >= max_i for i in chain(self.split[0], *self.split[1].values())):
            raise InvalidCompetitionError("The predefined split contains invalid indices")

        return self

    @model_validator(mode="after")
    def _validate_cols_in_dataset(self) -> Self:
        """
        Verifies that all specified columns are present in the dataset.
        """
        columns = self.target_cols | self.input_cols
        dataset_columns = set(self.columns)
        if not columns.issubset(dataset_columns):
            raise InvalidCompetitionError("Not all target or input columns were found in the dataset.")

        return self

    @model_validator(mode="after")
    def _validate_n_classes(self) -> Self:
        """
        The number of classes for each of the target columns.
        """
        columns = set(self.n_classes.keys())
        if not columns.issubset(self.target_cols):
            raise InvalidCompetitionError("Not all target class members were found in the target columns.")

        return self

    @computed_field
    @property
    def dataset_artifact_id(self) -> str:
        return self.artifact_id

    def _get_subset(self, indices, hide_targets=True, featurization_fn=None) -> Subset:
        """Returns a [`Subset`][polaris.dataset.Subset] using the given indices. Used
        internally to construct the train and test sets."""
        return Subset(
            dataset=self,
            indices=indices,
            input_cols=self.input_cols,
            target_cols=self.target_cols,
            hide_targets=hide_targets,
            featurization_fn=featurization_fn,
        )

    def _get_test_sets(
        self, hide_targets=True, featurization_fn: Callable | None = None
    ) -> dict[str, Subset]:
        """
        Construct the test set(s), given the split in the competition specification. Used
        internally to construct the test set for client use and evaluation.
        """
        test_split = self.split[1]
        return {
            k: self._get_subset(v, hide_targets=hide_targets, featurization_fn=featurization_fn)
            for k, v in test_split.items()
        }

    def get_train_test_split(
        self, featurization_fn: Callable | None = None
    ) -> tuple[Subset, Subset | dict[str, Subset]]:
        """Construct the train and test sets, given the split in the competition specification.

        Returns [`Subset`][polaris.dataset.Subset] objects, which offer several ways of accessing the data
        and can thus easily serve as a basis to build framework-specific (e.g. PyTorch, Tensorflow)
        data-loaders on top of.

        Args:
            featurization_fn: A function to apply to the input data. If a multi-input benchmark, this function
                expects an input in the format specified by the `input_format` parameter.

        Returns:
            A tuple with the train `Subset` and test `Subset` objects.
                If there are multiple test sets, these are returned in a dictionary and each test set has
                an associated name. The targets of the test set can not be accessed.
        """

        train = self._get_subset(self.split[0], hide_targets=False, featurization_fn=featurization_fn)
        test = self._get_test_sets(hide_targets=True, featurization_fn=featurization_fn)

        # For improved UX, we return the object instead of the dictionary if there is only one test set.
        # Internally, however, assume that the test set is always a dictionary simplifies the code.
        if len(test) == 1:
            test = test["test"]
        return train, test

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

    def _repr_html_(self):
        """For pretty printing in Jupyter."""
        return dict2html(self.model_dump(exclude={"zarr_manifest_path", "zarr_manifest_md5sum", "split"}))

    def __repr__(self):
        return self.model_dump_json(exclude={"zarr_manifest_path", "zarr_manifest_md5sum", "split"}, indent=2)

    def __str__(self):
        return self.__repr__()
