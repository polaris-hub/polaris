from datetime import datetime
import os
import numpy as np
from typing import Optional, Union

from pydantic import field_serializer
import numpy as np
import pandas as pd
from polaris.benchmark import BenchmarkSpecification
from polaris.evaluate import BenchmarkResults
from polaris.evaluate.utils import evaluate_benchmark
from polaris.hub.settings import PolarisHubSettings
from polaris.utils.types import AccessType, HubOwner, PredictionsType, TimeoutTypes, ZarrConflictResolution

class CompetitionSpecification(BenchmarkSpecification):
    """This class extends the [`BenchmarkSpecification`][polaris.benchmark.BenchmarkSpecification] to
    facilitate interactions with Polaris Competitions.

    Much of the underlying data model and logic is shared across Benchmarks and Competitions, and
    anything within this class serves as a point of differentiation between the two.

    facilitate interactions with Polaris Competitions.

    Much of the underlying data model and logic is shared across Benchmarks and Competitions, and
    anything within this class serves as a point of differentiation between the two.

    Currently, these entities will primarily differ at how user predictions are evaluated.
    """

    # Additional properties specific to Competitions
    start_time: datetime | None = None
    scheduled_end_time: datetime | None = None
    actual_end_time: datetime | None = None

    def evaluate(
        self,
        y_pred: PredictionsType,
        env_file: Optional[Union[str, os.PathLike]] = None,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        **kwargs: dict
    ):
        """Light convenience wrapper around
        [`PolarisHubClient.evaluate_competition`][polaris.hub.client.PolarisHubClient.evaluate_competition].
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(
            env_file=env_file,
            settings=settings,
            cache_auth_token=cache_auth_token,
            **kwargs,
        ) as client:
            client.evaluate_competition(self, y_pred=y_pred)

    def _hub_evaluate(self, y_pred: PredictionsType, y_true: PredictionsType):
        """Executes the evaluation logic for a competition, given a set of predictions.
        Called only by Polaris Hub to evaluate competitions after labels are
        downloaded from R2 on the hub. Evalutaion logic is the same as for regular benchmarks.

        Args:
            y_pred: The predictions for the test set, as NumPy arrays.
                If there are multiple targets, the predictions should be wrapped in a
                dictionary with the target labels as keys.

            test: The test set. If there are multiple targets, the target columns should
                be wrapped in a dictionary with the target labels as keys.

        Returns:
            A `BenchmarkResults` object containing the evaluation results.
        """
        scores = evaluate_benchmark(y_pred, y_true, self.target_cols, self.metrics)
        return BenchmarkResults(results=scores,
                                benchmark_name=self.name,
                                benchmark_owner=self.owner)

    def upload_to_hub(
        self,
        env_file: Optional[Union[str, os.PathLike]] = None,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        access: AccessType = "private",
        timeout: TimeoutTypes = (10, 200),
        owner: Optional[Union[HubOwner, str]] = None,
        if_exists: ZarrConflictResolution = "replace",
    ):
        """Very light, convenient wrapper around the
        [`PolarisHubClient.upload_competition`][polaris.hub.client.PolarisHubClient.upload_competition] method."""

        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient(
            env_file=env_file,
            settings=settings,
            cache_auth_token=cache_auth_token,
        ) as client:
            return client.upload_competition(self.dataset, self, access, timeout, owner, if_exists)

    @field_serializer("start_time", "scheduled_end_time")
    def _serialize_start_date(self, v):
        """Convert from datetime to string to make sure it's serializable"""
        if v:
            return v.isoformat()
