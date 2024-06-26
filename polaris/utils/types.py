from enum import Enum
from typing import Annotated, Any, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    HttpUrl,
    StringConstraints,
    TypeAdapter,
)
from pydantic.alias_generators import to_camel
from typing_extensions import TypeAlias

SplitIndicesType: TypeAlias = list[int]
"""
A split is defined by a sequence of integers.
"""

SplitType: TypeAlias = tuple[SplitIndicesType, Union[SplitIndicesType, dict[str, SplitIndicesType]]]
"""
A split is a pair of which the first item is always assumed to be the train set.
The second item can either be a single test set or a dictionary with multiple, named test sets.
"""

PredictionsType: TypeAlias = Union[np.ndarray, dict[str, Union[np.ndarray, dict[str, np.ndarray]]]]
"""
A prediction is one of three things:

- A single array (single-task, single test set)
- A dictionary of arrays (single-task, multiple test sets) 
- A dictionary of dictionaries of arrays (multi-task, multiple test sets)
"""

DatapointPartType = Union[Any, tuple[Any], dict[str, Any]]
DatapointType: TypeAlias = tuple[DatapointPartType, DatapointPartType]
"""
A datapoint has:

- A single input or multiple inputs (either as dict or tuple)
- No target, a single target or a multiple targets (either as dict or tuple)
"""

SlugStringType: TypeAlias = Annotated[
    str, StringConstraints(pattern="^[a-z0-9-]+$", min_length=4, max_length=64)
]
"""
A URL-compatible string that can serve as slug on the hub.
"""

SlugCompatibleStringType: TypeAlias = Annotated[
    str, StringConstraints(pattern="^[A-Za-z0-9_-]+$", min_length=4, max_length=64)
]
"""
A URL-compatible string that can be turned into a slug by the hub.

Can only use alpha-numeric characters, underscores and dashes. 
The string must be at least 4 and at most 64 characters long.
"""


HubUser: TypeAlias = SlugCompatibleStringType
"""
A user on the Polaris Hub is identified by a username, 
which is a [`SlugCompatibleStringType`][polaris.utils.types.SlugCompatibleStringType].
"""

HttpUrlAdapter = TypeAdapter(HttpUrl)
HttpUrlString: TypeAlias = Annotated[str, BeforeValidator(lambda v: HttpUrlAdapter.validate_python(v) and v)]

"""
A validated URL that will be turned into a string.
This is useful for interactions with httpx and authlib, who have their own URL types.
"""

DirectionType: TypeAlias = float | Literal["min", "max"]
"""
The direction of any variable to be sorted.
This can be used to sort the metric score, indicate the optmization direction of endpoint.
"""

AccessType: TypeAlias = Literal["public", "private"]
"""
Type to specify access to a dataset, benchmark or result in the Hub.
"""

TimeoutTypes = Union[Tuple[int, int], Literal["timeout", "never"]]
"""
Timeout types for specifying maximum wait times.
"""

IOMode: TypeAlias = Literal["r", "r+", "a", "w", "w-"]
"""
Type to specify the mode for input/output operations (I/O) when interacting with a file or resource.
"""

SupportedLicenseType: TypeAlias = Literal[
    "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC0-1.0", "MIT"
]
"""
Supported license types for dataset uploads to Polaris Hub
"""

ZarrConflictResolution: TypeAlias = Literal["raise", "replace", "skip"]
"""
Type to specify which action to take when encountering existing files within a Zarr archive.
"""


class HubOwner(BaseModel):
    """An owner of an artifact on the Polaris Hub

    The slug is most important as it is the user-facing part of this data model.
    The externalId and type are added to be consistent with the model returned by the Polaris Hub .
    """

    slug: SlugStringType
    external_id: Optional[str] = None
    type: Optional[Literal["user", "organization"]] = None

    # Pydantic config
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    def __str__(self):
        return self.slug


class TargetType(Enum):
    """The high-level classification of different targets."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class TaskType(Enum):
    """The high-level classification of different tasks."""

    MULTI_TASK = "multi_task"
    SINGLE_TASK = "single_task"
