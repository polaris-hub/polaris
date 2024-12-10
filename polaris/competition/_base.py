from datetime import datetime
import json
from typing import Callable, TypeAlias, Union

from pydantic import (
    Field,
    computed_field,
    field_serializer,
    field_validator,
)

from polaris.dataset._subset import Subset
from polaris.evaluate import Metric
from polaris.evaluate._predictions import CompetitionPredictions
from polaris.experimental._dataset_v2 import DatasetV2
from polaris.utils.dict2html import dict2html
from polaris.utils.types import TargetType, TaskType, SplitType

ColumnsType: TypeAlias = str | list[str]


class CompetitionSpecification(DatasetV2):
    """An instance of this class represents a Polaris competition. It defines fields and functionality
    that in combination with the [`DatasetV2`][polaris.experimental._dataset_v2.DatasetV2] class, allow
    users to participate in competitions hosted on Polaris Hub.

    Examples:
        Basic API usage:
        ```python
        import polaris as po

        # Load the benchmark from the Hub
        competition = po.load_competition("polaris/hello-world-competition")

        # Get the train and test data-loaders
        train, test = competition.get_train_test_split()

        # Use the training data to train your model
        # Get the input as an array with 'train.inputs' and 'train.targets'
        # Or simply iterate over the train object.
        for x, y in train:
            ...

        # Work your magic to accurately predict the test set
        prediction_values = np.array([0.0 for x in test])
        predictions = CompetitionPredictions(
            name="first-prediction",
            owner=HubOwner(slug="dummy-user"),
            report_url="REPORT_URL",
            target_labels=competition.target_cols,
            test_set_labels=competition.test_set_labels,
            test_set_sizes=competition.test_set_sizes,
            predictions=prediction_values,
        )

        # Submit your predictions
        competition.submit_predictions(predictions)
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
    For additional meta-data attributes, see the [`DatasetV2`][polaris.experimental._dataset_v2.DatasetV2] class.
    """

    _artifact_type = "competition"

    target_cols: ColumnsType
    input_cols: ColumnsType
    split: SplitType
    metrics: Union[str, Metric, list[str | Metric]]
    main_metric: str | Metric
    readme: str
    target_types: dict[str, Union[TargetType, str]] = Field(default_factory=dict, validate_default=True)
    start_time: datetime
    end_time: datetime
    n_test_sets: int
    n_test_datapoints: dict[str, int]
    n_classes: dict[str, int]

    @field_validator("metrics")
    def _validate_metrics(cls, v) -> list[Metric]:
        """
        Verifies all specified metrics are either a Metric object or a valid metric name.

        If there are multiple test sets, it is assumed the same metrics are used across test sets.
        """
        if not isinstance(v, list):
            v = [v]

        v = [m if isinstance(m, Metric) else Metric[m] for m in v]

        return v

    @field_validator("main_metric")
    def _validate_main_metric(cls, v) -> Metric:
        """Converts the main metric to a Metric object if it is a string."""
        # lu: v can be None.
        if v and not isinstance(v, Metric):
            v = Metric[v]
        return v

    @field_serializer("metrics", "main_metric")
    def _serialize_metrics(self, v) -> list[str] | str:
        """Return the string identifier so we can serialize the object"""
        if isinstance(v, Metric):
            return v.name
        return [m.name for m in v]

    @field_serializer("target_types")
    def _serialize_target_types(self, v) -> dict[str, str]:
        """Convert from enum to string to make sure it's serializable"""
        return {k: v for k, v in self.target_types.items()}

    @field_serializer("start_time", "end_time")
    def _serialize_times(self, v: datetime) -> str:
        """Convert from datetime to string to make sure it's serializable"""
        return v.isoformat()

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
        return sorted(list(self.n_test_datapoints.keys()))

    @computed_field
    @property
    def test_set_sizes(self) -> dict[str, int]:
        """The sizes of the test sets."""
        return {k: v for k, v in self.n_test_datapoints.items()}

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
        predictions: CompetitionPredictions,
    ) -> None:
        """
        Very light, convenient wrapper around the
        [`PolarisHubClient.submit_competition_predictions`][polaris.hub.client.PolarisHubClient.submit_competition_predictions] method.
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient() as client:
            client.submit_competition_predictions(competition=self, competition_predictions=predictions)

    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        repr_dict = self.model_dump(exclude=["zarr_manifest_md5sum"])
        repr_dict.pop("split")
        return repr_dict

    def _repr_html_(self):
        """For pretty printing in Jupyter."""
        return dict2html(self._repr_dict_())

    def __repr__(self):
        return json.dumps(self._repr_dict_(), indent=2)

    def __str__(self):
        return self.__repr__()
