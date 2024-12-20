from datetime import datetime
from itertools import chain
from typing import Callable

from loguru import logger
from pydantic import computed_field, field_serializer, model_validator
from typing_extensions import Self

from polaris.benchmark import BenchmarkSpecification
from polaris.benchmark._base import ColumnName
from polaris.dataset import Subset
from polaris.evaluate import CompetitionPredictions
from polaris.experimental._dataset_v2 import DatasetV2
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidCompetitionError
from polaris.utils.misc import listit
from polaris.utils.types import (
    HttpUrlString,
    HubOwner,
    HubUser,
    PredictionsType,
    SlugCompatibleStringType,
    SplitType,
)


class CompetitionSpecification(DatasetV2, BenchmarkSpecification):
    """An instance of this class represents a Polaris competition. It defines fields and functionality
    that in combination with the `polaris.experimental._dataset_v2.DatasetV2` class, allow
    users to participate in competitions hosted on Polaris Hub.

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
        target_cols: The column(s) of the original dataset that should be used as target.
        input_cols: The column(s) of the original dataset that should be used as input.
        split: The predefined train-test split to use for evaluation.
        metrics: The metrics to use for evaluating performance
        main_metric: The main metric used to rank methods.
        readme: Markdown text that contains a formatted description of the competition.
        target_types: A dictionary that maps target columns to their type.
        start_time: The time at which the competition starts accepting prediction submissions.
        end_time: The time at which the competition stops accepting prediction submissions.
        n_classes: The number of classes within target columns that define a classification task.

    For additional meta-data attributes, see the `polaris.experimental._dataset_v2.DatasetV2` class.
    """

    _artifact_type = "competition"

    dataset: None = None
    split: SplitType

    start_time: datetime
    end_time: datetime
    n_classes: dict[ColumnName, int]

    @model_validator(mode="after")
    def _validate_split(self) -> Self:
        """
        Verifies that:
          1) There are no empty test partitions
          2) All indices are valid given the dataset
          3) There is no duplicate indices in any of the sets
          4) There is no overlap between the train and test set
          5) No row exists in the test set where all labels are missing/empty
        """

        if not isinstance(self.split[1], dict):
            self.split = self.split[0], {"test": self.split[1]}
        split = self.split

        # Train partition can be empty (zero-shot)
        # Test partitions cannot be empty
        if any(len(v) == 0 for v in split[1].values()):
            raise InvalidCompetitionError("The predefined split contains empty test partitions")

        train_idx_list = split[0]
        full_test_idx_list = list(chain.from_iterable(split[1].values()))

        if len(train_idx_list) == 0:
            logger.info(
                "This competition only specifies a test set. It will return an empty train set in `get_train_test_split()`"
            )

        train_idx_set = set(train_idx_list)
        full_test_idx_set = set(full_test_idx_list)

        # The train and test indices do not overlap
        if len(train_idx_set & full_test_idx_set) > 0:
            raise InvalidCompetitionError("The predefined split specifies overlapping train and test sets")

        # Check for duplicate indices within the train set
        if len(train_idx_set) != len(train_idx_list):
            raise InvalidCompetitionError("The training set contains duplicate indices")

        # Check for duplicate indices within a given test set. Because a user can specify
        # multiple test sets for a given benchmark and it is acceptable for indices to be shared
        # across test sets, we check for duplicates in each test set independently.
        for test_set_name, test_set_idx_list in split[1].items():
            if len(test_set_idx_list) != len(set(test_set_idx_list)):
                raise InvalidCompetitionError(
                    f'Test set with name "{test_set_name}" contains duplicate indices'
                )

        # All indices are valid given the dataset. We check the len of `self` here because a
        # competition entity includes both the dataset and benchmark in one artifact.
        max_i = len(self)
        if any(i < 0 or i >= max_i for i in chain(train_idx_list, full_test_idx_set)):
            raise InvalidCompetitionError("The predefined split contains invalid indices")

        return self

    @model_validator(mode="after")
    def _validate_n_classes(self) -> Self:
        """
        The number of classes for each of the target columns.
        """
        columns = set(self.n_classes.keys())
        if not columns.issubset(self.target_cols):
            raise InvalidCompetitionError("Not all target class numbers were found in the target columns.")

        return self

    @model_validator(mode="after")
    def _validate_cols(self) -> Self:
        """
        Verifies that all specified columns are present in the dataset.
        """
        columns = self.target_cols | self.input_cols
        dataset_columns = set(self.columns)
        if not columns.issubset(dataset_columns):
            raise InvalidCompetitionError("Not all target or input columns were found in the dataset.")

        return self

    @field_serializer("split")
    def _serialize_split(self, v: SplitType):
        """Convert any tuple to list to make sure it's serializable"""
        return listit(v)


    @computed_field
    @property
    def dataset_artifact_id(self) -> str:
        return self.artifact_id

    @computed_field
    @property
    def test_set_sizes(self) -> dict[str, int]:
        """The sizes of the test sets."""
        return {k: len(v) for k, v in self.split[1].items()}

    @computed_field
    @property
    def n_test_sets(self) -> int:
        """The number of test sets"""
        return len(self.split[1])

    @computed_field
    @property
    def n_train_datapoints(self) -> int:
        """The size of the train set."""
        return len(self.split[0])

    @computed_field
    @property
    def test_set_labels(self) -> list[str]:
        """The labels of the test sets."""
        return sorted(list(self.split[1].keys()))

    @computed_field
    @property
    def n_test_datapoints(self) -> dict[str, int]:
        """The size of (each of) the test set(s)."""
        if self.n_test_sets == 1:
            return {
                "test": len(self.split[1])
            }
        else:
            return {k: len(v) for k, v in self.split[1].items()}

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
        Construct the test set(s), given the split in the benchmark specification. Used
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
        prediction_name: SlugCompatibleStringType,
        prediction_owner: str,
        predictions: PredictionsType,
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
            contributors=contributors,
            github_url=github_url,
            description=description,
            tags=tags,
            user_attributes=user_attributes,
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
