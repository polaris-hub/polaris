from typing import Any, Callable, ClassVar, Literal

from pydantic import Field, field_validator, model_validator, computed_field
from typing_extensions import Self

from polaris.benchmark._task import PredictiveTaskSpecificationMixin
from polaris._artifact import BaseArtifactModel
from polaris.utils.types import (
    IncomingPredictionsType,
    HubOwner,
    SlugCompatibleStringType,
    HubUser,
)

from polaris.benchmark._split_v2 import SplitSpecificationV2Mixin
from polaris.dataset import DatasetV2, Subset
from polaris.utils.errors import InvalidBenchmarkError
from polaris.utils.types import ColumnName
from polaris.model import Model
from polaris.hub.settings import PolarisHubSettings


class BenchmarkV2Specification(
    PredictiveTaskSpecificationMixin, BaseArtifactModel, SplitSpecificationV2Mixin
):
    """This class wraps a dataset with additional data to specify the evaluation logic for V2 benchmarks.

    Unlike V1 benchmarks, V2 benchmarks do not require metrics to be specified client-side
    as evaluation happens server-side.

    Attributes:
        dataset: The dataset the benchmark specification is based on.
        splits: The predefined train-test splits to use for evaluation.
        n_classes: The number of classes for each of the target columns.
        readme: Markdown text that can be used to provide a formatted description of the benchmark.
        artifact_version: The version of the benchmark.
        artifact_changelog: A description of the changes made in this benchmark version.
    """

    _artifact_type = "benchmark"
    _version: ClassVar[Literal[2]] = 2

    dataset: DatasetV2 = Field(exclude=True)
    n_classes: dict[ColumnName, int] = Field(default_factory=dict)
    readme: str = ""
    artifact_version: int = Field(default=1, frozen=True)
    artifact_changelog: str | None = None

    @computed_field
    @property
    def dataset_artifact_id(self) -> str:
        return self.dataset.artifact_id

    @field_validator("dataset", mode="before")
    @classmethod
    def _parse_dataset(
        cls,
        v: DatasetV2 | str | dict[str, Any],
    ) -> DatasetV2:
        """
        Allows either passing a Dataset object or the kwargs to create one
        """
        match v:
            case dict():
                return DatasetV2(**v)
            case str():
                return DatasetV2.from_json(v)
            case DatasetV2():
                return v

    @model_validator(mode="after")
    def _validate_n_classes(self) -> Self:
        """
        The number of classes for each of the target columns.
        """
        columns = set(self.n_classes.keys())
        if not columns.issubset(self.target_cols):
            raise InvalidBenchmarkError("Not all specified class numbers were found in the target columns.")

        return self

    @model_validator(mode="after")
    def _validate_split_in_dataset(self) -> Self:
        """
        Verifies that:
          - All indices are valid given the dataset
        """
        dataset_length = len(self.dataset)
        if self.max_index >= dataset_length:
            raise InvalidBenchmarkError("The predefined split contains invalid indices")

        return self

    @model_validator(mode="after")
    def _validate_cols_in_dataset(self) -> Self:
        """
        Verifies that all specified columns are present in the dataset.
        """
        columns = self.target_cols | self.input_cols
        dataset_columns = set(self.dataset.columns)
        if not columns.issubset(dataset_columns):
            raise InvalidBenchmarkError("Not all target or input columns were found in the dataset.")

        return self

    def _get_splits(
        self, hide_targets=True, featurization_fn: Callable | None = None
    ) -> dict[str, tuple[Subset, Subset]]:
        """
        Construct all train-test split pairs, given the splits in the benchmark specification.
        Used internally to construct the splits for client use and evaluation.
        """
        # TODO: We need a subset class that can handle very large index sets without copying or materializing all of them
        return {
            label: (
                self._get_subset(
                    split.training.indices, hide_targets=False, featurization_fn=featurization_fn
                ),
                self._get_subset(
                    split.test.indices, hide_targets=hide_targets, featurization_fn=featurization_fn
                ),
            )
            for label, split in self.split_items()
        }

    def _get_subset(self, indices, hide_targets=True, featurization_fn=None) -> Subset:
        """Returns a [`Subset`][polaris.dataset.Subset] using the given indices. Used
        internally to construct the train and test sets."""
        return Subset(
            dataset=self.dataset,
            indices=indices,
            input_cols=self.input_cols,
            target_cols=self.target_cols,
            hide_targets=hide_targets,
            featurization_fn=featurization_fn,
        )

    def get_train_test_split(
        self, featurization_fn: Callable | None = None
    ) -> dict[str, tuple[Subset, Subset]]:
        """Construct the train and test sets for all splits, given the splits in the benchmark specification.

        Returns [`Subset`][polaris.dataset.Subset] objects, which offer several ways of accessing the data
        and can thus easily serve as a basis to build framework-specific (e.g. PyTorch, Tensorflow)
        data-loaders on top of.

        Args:
            featurization_fn: A function to apply to the input data. If a multi-input benchmark, this function
                expects an input in the format specified by the `input_format` parameter.

        Returns:
            A dictionary mapping split labels to (train, test) tuples of `Subset` objects.
            The targets of the test sets cannot be accessed.
        """
        return self._get_splits(hide_targets=True, featurization_fn=featurization_fn)

    def upload_to_hub(
        self,
        settings: PolarisHubSettings | None = None,
        cache_auth_token: bool = True,
        owner: HubOwner | str | None = None,
        parent_artifact_id: str | None = None,
        **kwargs: dict,
    ):
        """
        Very light, convenient wrapper around the
        [`PolarisHubClient.upload_benchmark`][polaris.hub.client.PolarisHubClient.upload_benchmark] method.
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(
            settings=settings,
            cache_auth_token=cache_auth_token,
            **kwargs,
        ) as client:
            return client.upload_benchmark(self, owner=owner, parent_artifact_id=parent_artifact_id)

    def submit_predictions(
        self,
        predictions: IncomingPredictionsType,
        prediction_name: SlugCompatibleStringType,
        prediction_owner: str,
        contributors: list[HubUser] | None = None,
        model: Model | None = None,
        description: str = "",
        tags: list[str] | None = None,
        user_attributes: dict[str, str] | None = None,
    ) -> None:
        """
        Convenient wrapper around the
        [`PolarisHubClient.submit_benchmark_predictions`][polaris.hub.client.PolarisHubClient.submit_benchmark_predictions] method.
        It handles the creation of a standardized Predictions object, which is expected by the Hub, automatically.

        Args:
            predictions: The predictions for each test set defined in the benchmark.
            prediction_name: The name of the prediction.
            prediction_owner: The slug of the user/organization which owns the prediction.
            contributors: The users credited with generating these predictions.
            model: (Optional) The Model artifact used to generate these predictions.
            description: An optional and short description of the predictions.
            tags: An optional list of tags to categorize the prediction by.
            user_attributes: An optional dict with additional, textual user attributes.
        """
        from polaris.hub.client import PolarisHubClient
        from polaris.prediction import BenchmarkPredictionsV2

        standardized_predictions = BenchmarkPredictionsV2(
            name=prediction_name,
            owner=HubOwner(slug=prediction_owner),
            dataset_zarr_root=self.dataset.zarr_root,
            benchmark_artifact_id=self.artifact_id,
            predictions=predictions,
            target_labels=list(self.target_cols),
            test_set_labels=self.split_labels,
            test_set_sizes=self.n_test_datapoints,
            contributors=contributors or [],
            model=model,
            description=description,
            tags=tags or [],
            user_attributes=user_attributes or {},
        )

        with PolarisHubClient() as client:
            client.submit_benchmark_predictions(prediction=standardized_predictions, owner=prediction_owner)

    def to_json(self, destination: str) -> str:
        """Save the benchmark to a destination directory as a JSON file.

        Warning: Multiple files
            Perhaps unintuitive, this method creates multiple files in the destination directory as it also saves
            the dataset it is based on to the specified destination.

        Args:
            destination: The _directory_ to save the associated data to.

        Returns:
            The path to the JSON file.
        """
        from pathlib import Path
        import json
        import fsspec

        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()
        data["dataset"] = self.dataset.to_json(destination)

        path = dest / "benchmark.json"
        with fsspec.open(str(path), "w") as f:
            json.dump(data, f)

        return str(path)

    def _repr_dict_(self) -> dict:
        """Utility function for pretty-printing to the command line and jupyter notebooks"""
        repr_dict = self.model_dump()
        repr_dict["dataset_name"] = self.dataset.name
        return repr_dict

    def _repr_html_(self):
        """For pretty printing in Jupyter."""
        from polaris.utils.dict2html import dict2html

        return dict2html(self.model_dump())

    def __repr__(self):
        return self.model_dump_json(indent=2)

    def __str__(self):
        return self.__repr__()
