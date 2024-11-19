import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from polaris.utils.misc import convert_lists_to_arrays
from polaris.utils.types import IncomingPredictionsType, PredictionsType


class BenchmarkPredictions(BaseModel):
    """
    Base model to represent predictions in the Polaris code base.

    Guided by [Postel's Law](https://en.wikipedia.org/wiki/Robustness_principle),
    this class normalizes different formats to a single, internal representation.

    Attributes:
        predictions: The predictions for the benchmark.
        target_labels: The target columns for the associated benchmark.
        test_set_labels: The names of the test sets for the associated benchmark.
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

        return {
            "predictions": predictions,
            "target_labels": target_labels,
            "test_set_labels": test_set_labels,
            "test_set_sizes": test_set_sizes,
        }

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
