import json
import logging
from typing import Annotated, ClassVar, Literal, Optional

import fsspec
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
from typing_extensions import Self

import polaris
from polaris.utils.misc import build_urn, slugify
from polaris.utils.types import ArtifactUrn, HubOwner, SlugCompatibleStringType, SlugStringType

logger = logging.getLogger(__name__)


class BaseArtifactModel(BaseModel):
    """
    Base class for all artifacts on the Hub. Specifies metadata that is used by the Hub.

    Info: Optional
        Despite all artifacts basing this class, note that all attributes are optional.
        This ensures the library can be used without the Polaris Hub.
        Only when uploading to the Hub, some of the attributes are required.

    Attributes:
        name: A slug-compatible name for the artifact.
            Together with the owner, this is used by the Hub to uniquely identify the artifact.
        description: A beginner-friendly, short description of the artifact.
        tags: A list of tags to categorize the artifact by. This is used by the Hub to search over artifacts.
        user_attributes: A dict with additional, textual user attributes.
        owner: A slug-compatible name for the owner of the artifact.
            If the artifact comes from the Polaris Hub, this is the associated owner (organization or user).
            Together with the name, this is used by the Hub to uniquely identify the artifact.
        polaris_version: The version of the Polaris library that was used to create the artifact.
    """

    _version: ClassVar[Literal[1]] = 1
    _artifact_type: ClassVar[str]

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, arbitrary_types_allowed=True)

    # Model attributes
    name: SlugCompatibleStringType | None = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    user_attributes: dict[str, str] = Field(default_factory=dict)
    owner: HubOwner | None = None
    polaris_version: str = polaris.__version__
    slug: Annotated[Optional[SlugStringType], Field(validate_default=True)] = None

    @field_validator("slug")
    def _validate_slug(cls, val: Optional[str], info) -> SlugStringType | None:
        # A slug may be None when an artifact is created locally
        if val is None:
            if info.data.get("name") is not None:
                return slugify(info.data.get("name"))
        return val

    @computed_field
    @property
    def artifact_id(self) -> str | None:
        if self.owner and self.slug:
            return f"{self.owner}/{self.slug}"
        return None

    @computed_field
    @property
    def urn(self) -> ArtifactUrn | None:
        if self.owner and self.slug:
            return self.urn_for(self.owner, self.slug)
        return None

    @computed_field
    @property
    def version(self) -> int:
        return self._version

    @field_validator("polaris_version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if value != "dev":
            # Make sure it is a valid semantic version
            Version(value)

        current_version = polaris.__version__
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
            path: Path to a JSON file containing the artifact definition.
        """
        with fsspec.open(path, "r") as f:
            data = json.load(f)
            return cls.model_validate(data)

    def to_json(self, path: str) -> None:
        """Saves an artifact to a JSON file.

        Args:
            path: Path to save the artifact definition as JSON.
        """
        with fsspec.open(path, "w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def urn_for(cls, owner: str | HubOwner, slug: str) -> ArtifactUrn:
        return build_urn(cls._artifact_type, owner, slug)
