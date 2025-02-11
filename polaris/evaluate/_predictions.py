from collections import defaultdict

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from polaris.evaluate import ResultsMetadataV1
from polaris.utils.misc import convert_lists_to_arrays
from polaris.utils.types import (
    HttpUrlString,
    HubOwner,
    IncomingPredictionsType,
    PredictionsType,
    SlugCompatibleStringType,
)


class BenchmarkPredictions(BaseModel):
    """
    Base model to represent predictions in the Polaris code base.

    Guided by [Postel's Law](https://en.wikipedia.org/wiki/Robustness_principle),
    this class normalizes different formats to a single, internal representation.

    Attributes:
        predictions: The predictions for the benchmark.
        target_labels: The target columns for the associated benchmark.
        test_set_labels: The names of the test sets for the associated benchmark.
        test_set_sizes: The number of rows in each test set for the associated benchmark.
    """

    predictions: PredictionsType
    target_labels: list[str]
    test_set_labels: list[str]
    test_set_sizes: dict[str, int]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("predictions")
    def _serialize_predictions(self, predictions: PredictionsType):
        """
        Recursively converts all numpy values in the predictions dictionary to lists
        so they can be serialized.
        """

        def convert_to_list(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, dict):
                return {k: convert_to_list(v) for k, v in v.items()}

        return convert_to_list(predictions)

    @field_validator("target_labels", "test_set_labels")
    @classmethod
    def _validate_labels(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("The predictions contain duplicate columns")
        return v

    @model_validator(mode="before")
    @classmethod
    def _validate_predictions(cls, data: dict) -> dict:
        """Normalizes the predictions format to a standard representation we use internally"""

        # This model validator runs before any Pydantic internal validation.
        # This way we can normalize the incoming data to a standard representation.
        # However, this implies that the fields can theoretically be any type.

        # Ensure the type of the incoming predictions is correct
        validator = TypeAdapter(IncomingPredictionsType, config={"arbitrary_types_allowed": True})
        predictions = validator.validate_python(data.get("predictions"))

        # Ensure the type of the target_labels and test_set_labels is correct
        validator = TypeAdapter(list[str])
        target_labels = validator.validate_python(data.get("target_labels"))
        test_set_labels = validator.validate_python(data.get("test_set_labels"))

        validator = TypeAdapter(dict[str, int])
        test_set_sizes = validator.validate_python(data.get("test_set_sizes"))

        # Normalize the predictions to a standard representation
        predictions = convert_lists_to_arrays(predictions)
        predictions = cls._normalize_predictions(predictions, target_labels, test_set_labels)

        # Update class data with the normalized fields. Use of the `update()` method
        # is required to prevent overwriting class data when this class is inherited.
        data.update(
            {
                "predictions": predictions,
                "target_labels": target_labels,
                "test_set_labels": test_set_labels,
                "test_set_sizes": test_set_sizes,
            }
        )

        return data

    @model_validator(mode="after")
    def check_test_set_size(self) -> Self:
        """Verify that the size of all predictions"""
        for test_set_label, test_set in self.predictions.items():
            for target in test_set.values():
                if test_set_label not in self.test_set_sizes:
                    raise ValueError(f"Expected size for test set '{test_set_label}' is not defined")

                if len(target) != self.test_set_sizes[test_set_label]:
                    raise ValueError(
                        f"Predictions size mismatch: The predictions for test set '{test_set_label}' "
                        f"should have a size of {self.test_set_sizes[test_set_label]}, but have a size of {len(target)}."
                    )
        return self

    @classmethod
    def _normalize_predictions(
        cls, predictions: IncomingPredictionsType, target_labels: list[str], test_set_labels: list[str]
    ) -> PredictionsType:
        """
        Normalizes  the predictions to a standard representation we use internally.
        This standard representation is a nested, two-level dictionary:
        `{test_set_name: {target_column: np.ndarray}}`
        """
        # (1) If the predictions are already fully specified, no need to do anything
        if cls._is_fully_specified(predictions, target_labels, test_set_labels):
            return predictions

        # If not fully specified, we distinguish 4 cases based on the type of benchmark.
        is_single_task = len(target_labels) == 1
        is_single_test = len(test_set_labels) == 1

        # (2) Single-task, single test set: We expect a numpy array as input.
        if is_single_task and is_single_test:
            if isinstance(predictions, dict):
                raise ValueError(
                    "The predictions for single-task, single test set benchmarks should be a numpy array."
                )

            predictions = {test_set_labels[0]: {target_labels[0]: predictions}}

        # (3) Single-task, multiple test sets: We expect a dictionary with the test set labels as keys.
        elif is_single_task and not is_single_test:
            if not isinstance(predictions, dict) or set(predictions.keys()) != set(test_set_labels):
                raise ValueError(
                    "The predictions for single-task, multiple test sets benchmarks "
                    "should be a dictionary with the test set labels as keys."
                )
            predictions = {k: {target_labels[0]: v} for k, v in predictions.items()}

        # (4) Multi-task, single test set: We expect a dictionary with the target labels as keys.
        elif not is_single_task and is_single_test:
            if not isinstance(predictions, dict) or set(predictions.keys()) != set(target_labels):
                raise ValueError(
                    "The predictions for multi-task, single test set benchmarks "
                    "should be a dictionary with the target labels as keys."
                )
            predictions = {test_set_labels[0]: predictions}

        # (5) Multi-task, multi-test sets: The predictions should be fully-specified
        else:
            raise ValueError(
                "The predictions for multi-task, multi-test sets benchmarks should be fully-specified "
                "as a nested, two-level dictionary: { test_set_name: { target_column: np.ndarray } }"
            )

        return predictions

    @classmethod
    def _is_fully_specified(
        cls, predictions: IncomingPredictionsType, target_labels: list[str], test_set_labels: list[str]
    ) -> bool:
        """
        Check if the predictions are fully specified for the target columns and test set names.
        """
        # Not a dictionary
        if not isinstance(predictions, dict):
            return False

        # Outer-level of the dictionary should correspond to the test set names
        if set(predictions.keys()) != set(test_set_labels):
            return False

        # Inner-level of the dictionary should correspond to the target columns
        for test_set_predictions in predictions.values():
            if not isinstance(test_set_predictions, dict):
                return False
            if set(test_set_predictions.keys()) != set(target_labels):
                return False

        return True

    def get_subset(
        self, test_set_subset: list[str] | None = None, target_subset: list[str] | None = None
    ) -> "BenchmarkPredictions":
        """Return a subset of the original predictions"""

        if test_set_subset is None and target_subset is None:
            return self

        if test_set_subset is None:
            test_set_subset = self.test_set_labels
        if target_subset is None:
            target_subset = self.target_labels

        predictions = defaultdict(dict)
        for test_set_name in self.predictions.keys():
            if test_set_name not in test_set_subset:
                continue

            for target_name, target in self.predictions[test_set_name].items():
                if target_name not in target_subset:
                    continue

                predictions[test_set_name][target_name] = target

        test_set_sizes = {k: v for k, v in self.test_set_sizes.items() if k in predictions.keys()}
        return BenchmarkPredictions(
            predictions=predictions,
            target_labels=target_subset,
            test_set_labels=test_set_subset,
            test_set_sizes=test_set_sizes,
        )

    def get_size(
        self, test_set_subset: list[str] | None = None, target_subset: list[str] | None = None
    ) -> int:
        """Return the total number of predictions, allowing for filtering by test set and target"""
        if test_set_subset is None and target_subset is None:
            return sum(self.test_set_sizes.values()) * len(self.target_labels)
        return len(self.get_subset(test_set_subset, target_subset))

    def flatten(self) -> np.ndarray:
        """Return the predictions as a single, flat numpy array"""
        if len(self.test_set_labels) != 1 and len(self.target_labels) != 1:
            raise ValueError(
                "Can only flatten predictions for benchmarks with a single test set and a single target"
            )
        return self.predictions[self.test_set_labels[0]][self.target_labels[0]]

    def __len__(self) -> int:
        """Return the total number of predictions"""
        return self.get_size()


class CompetitionPredictions(BenchmarkPredictions, ResultsMetadataV1):
    """
    Predictions for competition benchmarks.

    This object is to be used as input to
    [`PolarisHubClient.submit_competition_predictions`][polaris.hub.client.PolarisHubClient.submit_competition_predictions].
    It is used to ensure that the structure of the predictions are compatible with evaluation methods on the Polaris Hub.
    In addition to the predictions, it contains metadata that describes a predictions object.

    Attributes:
        name: A slug-compatible name for the artifact. It is redeclared here to be required.
        owner: A slug-compatible name for the owner of the artifact. It is redeclared here to be required.
        report_url: A URL to a report/paper/write-up which describes the methods used to generate the predictions.
    """

    _artifact_type = "competition-prediction"

    name: SlugCompatibleStringType
    owner: HubOwner
    paper_url: HttpUrlString = Field(alias="report_url", serialization_alias="reportUrl")

    def __repr__(self):
        return self.model_dump_json(by_alias=True, indent=2)

    def __str__(self):
        return self.__repr__()
