import json
from typing import Dict, Optional

import fsspec
from pydantic import BaseModel, Field

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

    @classmethod
    def from_json(cls, path: str):
        """Loads a benchmark from a JSON file.

        Args:
            path: Loads a benchmark specification from a JSON file.
        """
        with fsspec.open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json(self, path: str):
        """Saves the benchmark to a JSON file.

        Args:
            path: Saves the benchmark specification to a JSON file.
        """
        with fsspec.open(path, "w") as f:
            json.dump(self.model_dump(), f)
