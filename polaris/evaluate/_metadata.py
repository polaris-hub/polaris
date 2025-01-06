from datetime import datetime

from pydantic import Field, PrivateAttr

from polaris._artifact import BaseArtifactModel
from polaris.utils.dict2html import dict2html
from polaris.utils.types import HttpUrlString, HubUser


class ResultsMetadata(BaseArtifactModel):
    """Base class for evaluation results

    Attributes:
        github_url: The URL to the code repository that was used to generate these results.
        paper_url: The URL to the paper describing the methodology used to generate these results.
        contributors: The users that are credited for these results.

    For additional meta-data attributes, see the base classes.
    """

    # Additional meta-data
    github_url: HttpUrlString | None = None
    paper_url: HttpUrlString | None = None
    contributors: list[HubUser] = Field(default_factory=list)

    # Private attributes
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    def _repr_html_(self) -> str:
        """For pretty-printing in Jupyter Notebooks"""
        return dict2html(self.model_dump())

    def __repr__(self):
        return self.model_dump_json(indent=2)
