from datetime import datetime
from hashlib import md5
from typing import Optional, Union

import numpy as np
from pydantic import ValidationInfo, computed_field, field_validator

from polaris.benchmark._base import BenchmarkV1Specification
from polaris.evaluate import CompetitionPredictions
from polaris.evaluate._metric import Metric
from polaris.experimental._dataset_v2 import DatasetV2
from polaris.hub.settings import PolarisHubSettings
from polaris.utils.errors import InvalidBenchmarkError
from sklearn.utils.multiclass import type_of_target

from polaris.utils.types import TargetType


class CompetitionSpecification(BenchmarkV1Specification):
    """Much of the underlying data model and logic is shared across Benchmarks and Competitions, and
    anything within this class serves as a point of differentiation between the two.

    Attributes:
        dataset: The V2 dataset the competition specification is based on.
        start_time: The time at which the competition becomes active and interactable.
        end_time: The time at which the competition ends and is no longer interactable.
        main_metric: The main metric used to rank methods.
    """

    _artifact_type = "competition"

    dataset: DatasetV2 # Can only load it from the Hub
    start_time: datetime
    end_time: datetime
    main_metric: str | Metric
    
    @field_validator("dataset")
    def _validate_dataset(cls, v):
        """
        Allows either passing a Dataset object or the kwargs to create one
        """
        if isinstance(v, dict):
            v = DatasetV2(**v)
        return v
    
    @field_validator("target_cols", "input_cols")
    def _validate_cols(cls, v, info: ValidationInfo):
        """Verifies all columns are present in the dataset."""
        if not isinstance(v, list):
            v = [v]
        if len(v) == 0:
            raise InvalidBenchmarkError("Specify at least a single column")
        if info.data.get("dataset") is not None and not all(
            c in info.data["dataset"].columns for c in v
        ):
            raise InvalidBenchmarkError("Not all specified columns were found in the dataset.")
        if len(set(v)) != len(v):
            raise InvalidBenchmarkError("The task specifies duplicate columns")
        return v
    
    @field_validator("target_types")
    def _validate_target_types(cls, v, info: ValidationInfo):
        """ All competitions on the Hub will have valid target types """
        pass
    
    @computed_field
    @property
    def n_classes(self) -> dict[str, int]:
        """The number of classes for each of the target columns."""
        n_classes = {}
        for target in self.target_cols:
            target_type = self.target_types.get(target)
            if (
                target_type is None
                or target_type == TargetType.REGRESSION
                or target_type == TargetType.DOCKING
            ):
                continue
            n_classes[target] = self.dataset.self.zarr_root[target].nunique()
        return n_classes
    
    def _compute_checksum(self):
        """ Competition checksums cannot be deterministically computed post-creation """
        pass

    
    def evaluate(
        self,
        predictions: CompetitionPredictions,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        **kwargs: dict,
    ):
        """Light convenience wrapper around
        [`PolarisHubClient.evaluate_competition`][polaris.hub.client.PolarisHubClient.submit_prediction].
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(
            settings=settings,
            cache_auth_token=cache_auth_token,
            **kwargs,
        ) as client:
            return client.submit_prediction(self, predictions)

    def __eq__(self, other):
        pass