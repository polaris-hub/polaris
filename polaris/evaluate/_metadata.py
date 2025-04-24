from datetime import datetime

from pydantic import Field, PrivateAttr, computed_field

from polaris._artifact import BaseArtifactModel
from polaris.utils.dict2html import dict2html
from polaris.utils.types import HttpUrlString, HubUser
from polaris.model import Model


class ResultsMetadataV1(BaseArtifactModel):
    """V1 implementation of evaluation results without model field support

    Attributes:
        github_url: The URL to the code repository that was used to generate these results.
        paper_url: The URL to the paper describing the methodology used to generate these results.
        contributors: The users that are credited for these results.

    For additional metadata attributes, see the base classes.
    """

    # Additional metadata
    github_url: HttpUrlString | None = Field(None, alias="code_url")
    paper_url: HttpUrlString | None = Field(None, alias="report_url")
    contributors: list[HubUser] = Field(default_factory=list)

    # Private attributes
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    def _repr_html_(self) -> str:
        """For pretty-printing in Jupyter Notebooks"""
        return dict2html(self.model_dump())

    def __repr__(self):
        return self.model_dump_json(indent=2)


class ResultsMetadataV2(BaseArtifactModel):
    """V2 implementation of evaluation results with model field replacing URLs

    Attributes:
        model: The model that was used to generate these results.
        contributors: The users that are credited for these results.

    For additional metadata attributes, see the base classes.
    """

    # Additional metadata
    model: Model | None = Field(None, exclude=True)
    contributors: list[HubUser] = Field(default_factory=list)

    # Private attributes
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    @computed_field
    @property
    def model_artifact_id(self) -> str:
        return self.model.artifact_id if self.model else None

    def _repr_html_(self) -> str:
        return dict2html(self.model_dump())

    def __repr__(self):
        return self.model_dump_json(indent=2)
