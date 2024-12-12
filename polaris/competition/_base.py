from datetime import datetime
from itertools import chain
from typing import Callable, TypeAlias, Union
from typing_extensions import Self
from loguru import logger
from pydantic import (
    Field,
    FieldSerializationInfo,
    SerializerFunctionWrapHandler,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from polaris.dataset._subset import Subset
from polaris.evaluate import Metric
from polaris.evaluate._predictions import CompetitionPredictions
from polaris.experimental._dataset_v2 import DatasetV2
from polaris.utils.dict2html import dict2html
from polaris.utils.errors import InvalidCompetitionError
from polaris.utils.misc import listit
from polaris.utils.types import (
    HubOwner,
    TargetType,
    TaskType,
    SplitType,
    HttpUrlString,
    SlugCompatibleStringType,
    PredictionsType,
    HubUser,
)

ColumnsType: TypeAlias = str | list[str]


class CompetitionSpecification(DatasetV2):
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
        n_test_sets: The number of test sets defined in the test split.
        n_test_datapoints: The number of datapoints in each test set.
        n_classes: The number of classes within target columns that define a classification task.

    For additional meta-data attributes, see the `polaris.experimental._dataset_v2.DatasetV2` class.
    """

    _artifact_type = "competition"

    target_cols: ColumnsType
    input_cols: ColumnsType
    split: SplitType
    metrics: set[Metric] = Field(min_length=1)
    main_metric: Metric | str
    readme: str
    target_types: dict[str, TargetType] = Field(default_factory=dict, validate_default=True)
    start_time: datetime
    end_time: datetime
    n_test_sets: int
    n_test_datapoints: dict[str, int]
    n_classes: dict[str, int]

    @field_validator("metrics", mode="before")
    @classmethod
    def _validate_metrics(cls, v) -> set[Metric]:
        """
        Verifies all specified metrics are either a Metric object or a valid metric name.
        Also verifies there are no duplicate metrics.

        If there are multiple test sets, it is assumed the same metrics are used across test sets.
        """
        if isinstance(v, str):
            v = {"label": v}
        if isinstance(v, set):
            v = list(v)
        if not isinstance(v, list):
            v = [v]

        def _convert(m: str | dict | Metric) -> Metric:
            if isinstance(m, str):
                return Metric(label=m)
            if isinstance(m, dict):
                return Metric(**m)
            return m

        v = [_convert(m) for m in v]

        unique_metrics = set(v)

        if len(unique_metrics) != len(v):
            raise InvalidCompetitionError("The competition specifies duplicate metrics.")

        unique_names = {m.name for m in unique_metrics}
        if len(unique_names) != len(unique_metrics):
            raise InvalidCompetitionError(
                "The metrics of a competition need to have unique names. Specify a custom name with Metric(custom_name=...)"
            )

        return unique_metrics

    @model_validator(mode="after")
    def _validate_main_metric_is_in_metrics(self) -> Self:
        if isinstance(self.main_metric, str):
            for m in self.metrics:
                if m.name == self.main_metric:
                    self.main_metric = m
                    break
        if self.main_metric not in self.metrics:
            raise InvalidCompetitionError("The main metric should be one of the specified metrics")
        return self

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
                "This competitioon only specifies a test set. It will return an empty train set in `get_train_test_split()`"
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

    @field_serializer("metrics", mode="wrap")
    @staticmethod
    def _serialize_metrics(
        value: set[Metric], handler: SerializerFunctionWrapHandler, info: FieldSerializationInfo
    ) -> list[dict]:
        """Convert the set to a list"""
        return handler(list(value))

    @field_serializer("main_metric")
    def _serialize_main_metric(value: Metric) -> str:
        """Convert the set to a list"""
        return value.name

    @field_serializer("target_types")
    def _serialize_target_types(self, v) -> dict[str, str]:
        """Convert from enum to string to make sure it's serializable"""
        return {k: v.value for k, v in self.target_types.items()}

    @field_serializer("split")
    def _serialize_split(self, v: SplitType):
        """Convert any tuple to list to make sure it's serializable"""
        return listit(v)

    @computed_field
    @property
    def task_type(self) -> str:
        """The high-level task type of the benchmark."""
        v = TaskType.MULTI_TASK if len(self.target_cols) > 1 else TaskType.SINGLE_TASK
        return v.value

    @computed_field
    @property
    def test_set_labels(self) -> list[str]:
        """The labels of the test sets."""
        return sorted(list(self.split[1].keys()))

    @computed_field
    @property
    def test_set_sizes(self) -> dict[str, int]:
        """The sizes of the test sets."""
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

    def _get_test_set(self, featurization_fn: Callable | None = None) -> Union["Subset", dict[str, Subset]]:
        """Construct the test set(s), given the split in the competition specification. Used
        internally to construct the test set for client use and evaluation.
        """

        def make_test_subset(vals):
            return self._get_subset(vals, hide_targets=False, featurization_fn=featurization_fn)

        test_split = self.split[1]
        test = {k: make_test_subset(v) for k, v in test_split.items()}

        return test

    def get_train_test_split(
        self, featurization_fn: Callable | None = None
    ) -> tuple[Subset, Union["Subset", dict[str, Subset]]]:
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
        test = self._get_test_set(featurization_fn=featurization_fn)

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
        contributors: list[HubUser] = [],
        github_url: HttpUrlString | None = None,
        description: str = "",
        tags: list[str] = [],
        user_attributes: dict[str, str] = {},
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
        return dict2html(self.model_dump(exclude=["zarr_manifest_md5sum", "split"]))

    def __repr__(self):
        return self.model_dump_json(exclude=["zarr_manifest_md5sum", "split"], indent=2)

    def __str__(self):
        return self.__repr__()
