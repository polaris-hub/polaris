import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)
from typing import Any, Union

IncomingPredictionsType = Union[np.ndarray, dict[str, Union[np.ndarray, dict[str, np.ndarray]]]]


class BenchmarkPredictions(BaseModel):
    # predictions: Union[np.ndarray, dict[str, Union[np.ndarray, dict[str, np.ndarray]]]]
    predictions: Any
    target_cols: list[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # # benchmark: BenchmarkSpecification = Field(..., exclude=True)

    # @field_serializer("predictions")
    # def _serialize_predictions(self, predictions: PredictionsType):
    #     return {
    #         k1: {k2: v2.tolist() for k2, v2 in v1.items()}
    #         for k1, v1 in predictions.items()
    #     }

    @model_validator(mode="before")
    @classmethod
    def _validate_predictions(cls, data):
        """Normalizes the predictions so they are always in the standard format,
        {"test-set-name": {"target-name": np.ndarray}} regardless of how they are provided."""

        vals = data.get("predictions")
        target_cols = data.get("target_cols")

        if vals is None:
            raise ValueError("Predictions cannot be None")

        if not target_cols:
            raise ValueError("target_cols must be provided and cannot be empty")

        if cls.is_multi_task_single_test_set(vals, target_cols):
            data["predictions"] = {"test": vals}
        elif not isinstance(vals, dict):
            data["predictions"] = {"test": {target_cols[0]: vals}}
        else:
            # Check if the structure is valid
            for test_set, targets in vals.items():
                if not isinstance(targets, dict):
                    raise ValueError(f"Invalid structure for test set '{test_set}'. Expected a dictionary of targets.")
                for target, predictions in targets.items():
                    if not isinstance(predictions, np.ndarray):
                        raise ValueError(f"Invalid predictions for test set '{test_set}', target '{target}'. Expected a numpy array.")

        return data

    @property
    def is_multi_task_single_test_set(self) -> bool:
        """Check if the given values are for a multiple-task benchmark with a single
        test set. This is inferred by comparing the target names with the keys of the
        given data. If all keys in the given data match the target column names, we
        assume they are target names (as opposed to test set names for a single-task,
        multiple test set benchmark)."""
        return isinstance(self.predictions, dict) \
            and all(k in self.target_cols for k in self.predictions)