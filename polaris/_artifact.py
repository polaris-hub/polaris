from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator

from polaris.utils.types import HubOwner, SlugCompatibleStringType


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
    """

    name: Optional[SlugCompatibleStringType] = None
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    user_attributes: Dict[str, str] = Field(default_factory=dict)
    owner: Optional[HubOwner] = None

    @field_validator("name")
    def _validate_name(cls, v):
        """
        Since look-around does not work with Pydantic's constr regex,
        this verifies that the name does not end with a dash or underscore.
        """
        if v is None:
            return v
        if v.endswith("-") or v.endswith("_") or v.startswith("-") or v.startswith("_"):
            raise ValueError("Name cannot end with a dash or underscore.")
        return v
