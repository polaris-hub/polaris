import string
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator

from polaris.utils.types import HubOwner


class BaseArtifactModel(BaseModel):
    """
    Base class for all artifacts on the Hub. Specifies meta-data that is used by the Hub.

    Info: Optional
        Despite all artifact basing this class, note that all attributes are optional.
        This ensures the library can be used without the Polaris Hub.
        Only when uploading to the Hub, some of the attributes are required.

    Attributes:
        name: A URL-compatible name for the dataset, can only use alpha-numeric characters, underscores and dashes).
            Together with the owner, this is used by the Hub to uniquely identify the benchmark.
        description: A beginner-friendly, short description of the dataset.
        tags: A list of tags to categorize the benchmark by. This is used by the hub to search over benchmarks.
        user_attributes: A dict with additional, textual user attributes.
        owner: If the dataset comes from the Polaris Hub, this is the associated owner (organization or user).
            Together with the name, this is used by the Hub to uniquely identify the benchmark.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    user_attributes: Dict[str, str] = Field(default_factory=dict)
    owner: Optional[HubOwner] = None

    @field_validator("name")
    def _validate_name(cls, v):
        """
        Verify the name only contains valid characters which can be used in a file path or URL.
        """
        if v is None:
            return v
        valid_characters = string.ascii_letters + string.digits + "_-"
        if not all(c in valid_characters for c in v):
            raise ValueError(f"`name` can only contain alpha-numeric characters, - or _, found {v}")
        return v
