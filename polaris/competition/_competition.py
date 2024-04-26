from datetime import datetime
import os
from typing import Optional, Union
from polaris.benchmark import BenchmarkSpecification
from polaris.hub.settings import PolarisHubSettings
from polaris.utils.types import AccessType, HubOwner, TimeoutTypes, ZarrConflictResolution

class CompetitionSpecification(BenchmarkSpecification):
    """This class extends the [`BenchmarkSpecification`][polaris.benchmark.BenchmarkSpecification] to
    facilitate interactions with Polaris Competitions. 
    
    Much of the underlying data model and logic is shared across Benchmarks and Competitions, and 
    anything within this class serves as a point of differentiation between the two. 
    
    Currently, these entities will primarily differ at how user predictions are evaluated.
    """
    
    # Additional properties specific to Competitions
    start_time: datetime | None = None
    scheduled_end_time: datetime | None = None
    actual_end_time: datetime | None = None
    
    def evaluate(self, predictions):
        """Wrapper method which ultimately triggers an evaluation service to assess and score user predictions
        for a given competition
        """
        
        # TODO validate that the number of predictions supplied matches the number of test set rows
        pass
    
    def upload_to_hub(
        self,
        env_file: Optional[Union[str, os.PathLike]] = None,
        settings: Optional[PolarisHubSettings] = None,
        cache_auth_token: bool = True,
        access: AccessType = "private",
        timeout: TimeoutTypes = (10, 200),
        owner: Optional[Union[HubOwner, str]] = None,
        if_exists: ZarrConflictResolution = "replace"
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