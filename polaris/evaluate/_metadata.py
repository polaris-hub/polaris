from datetime import datetime

from pydantic import Field, PrivateAttr

from polaris._artifact import BaseArtifactModel
from polaris.utils.types import HttpUrlString, HubUser


class ResultsMetadataV1(BaseArtifactModel):
    """V1 implementation of evaluation results without model field support

    Attributes:
        github_url: The URL to the code repository that was used to generate these results.
        paper_url: The URL to the paper describing the methodology used to generate these results.
        contributors: The users that are credited for these results.

    For additional metadata attributes, see the base classes.
    """

    # Additional metadata
    github_url: HttpUrlString | None = None
    paper_url: HttpUrlString | None = None
    contributors: list[HubUser] = Field(default_factory=list)

    # Private attributes
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)
