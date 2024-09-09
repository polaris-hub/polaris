import json
from typing import ClassVar, Dict, Self

import fsspec
from loguru import logger
from packaging.version import Version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)
from pydantic.alias_generators import to_camel

import polaris as po
from polaris.utils.misc import slugify
from polaris.utils.types import HubOwner, SlugCompatibleStringType, SlugStringType


class BaseArtifactModel(BaseModel):
    """
    Base class for all artifacts on the Hub. Specifies meta-data that is used by the Hub.

    Info: Optional
        Despite all artifacts basing this class, note that all attributes are optional.
        This ensures the library can be used without the Polaris Hub.
        Only when uploading to the Hub, some of the attributes are required.

    Attributes:
        name: A slug-compatible name for the dataset.
            Together with the owner, this is used by the Hub to uniquely identify the benchmark.
        description: A beginner-friendly, short description of the dataset.
        tags: A list of tags to categorize the benchmark by. This is used by the hub to search over benchmarks.
        user_attributes: A dict with additional, textual user attributes.
        owner: A slug-compatible name for the owner of the dataset.
            If the dataset comes from the Polaris Hub, this is the associated owner (organization or user).
            Together with the name, this is used by the Hub to uniquely identify the benchmark.
        polaris_version: The version of the Polaris library that was used to create the artifact.
    """

    artifact_type: ClassVar[str] = "baseArtifact"

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, arbitrary_types_allowed=True)

    # Model attributes
    name: SlugCompatibleStringType | None = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    user_attributes: Dict[str, str] = Field(default_factory=dict)
    owner: HubOwner | None = None
    polaris_version: str = po.__version__

    @computed_field
    @property
    def slug(self) -> SlugStringType | None:
        return slugify(self.name)

    @computed_field
    @property
    def artifact_id(self) -> str | None:
        if self.owner and self.slug:
            return f"{self.owner}/{self.slug}"
        return None

    @computed_field
    @property
    def urn(self) -> str | None:
        if self.owner and self.slug:
            return f"urn:polaris:{self.artifact_type}:{self.owner}:{self.slug}"
        return None

    @field_validator("polaris_version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if value != "dev":
            # Make sure it is a valid semantic version
            Version(value)

        current_version = po.__version__
        if value != current_version:
            logger.info(
                f"The version of Polaris that was used to create the artifact ({value}) is different "
                f"from the currently installed version of Polaris ({current_version})."
            )
        return value

    @field_validator("owner", mode="before")
    @classmethod
    def _validate_owner(cls, value: str | HubOwner | None):
        if isinstance(value, str):
            return HubOwner(slug=value)
        return value

    @field_serializer("owner")
    def _serialize_owner(self, value: HubOwner) -> str | None:
        return value.slug if value else None

    @classmethod
    def from_json(cls, path: str) -> Self:
        """Loads an artifact from a JSON file.

        Args:
            path: Loads a benchmark specification from a JSON file.
        """
        with fsspec.open(path, "r") as f:
            return cls.model_validate_json(f.read())

    def to_json(self, path: str) -> None:
        """Saves an artifact to a JSON file.

        Args:
            path: Saves the benchmark specification to a JSON file.
        """
        with fsspec.open(path, "w") as f:
            f.write(self.model_dump_json())
