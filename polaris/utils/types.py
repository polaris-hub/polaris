from enum import Enum
from typing import Annotated, Any, Literal, Optional

import numpy as np
from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    HttpUrl,
    StringConstraints,
    TypeAdapter,
)
from pydantic.alias_generators import to_camel
from typing_extensions import Self, TypeAlias

SplitIndicesType: TypeAlias = list[int]
"""
A split is defined by a sequence of integers.
"""

SplitType: TypeAlias = tuple[SplitIndicesType, SplitIndicesType | dict[str, SplitIndicesType]]
"""
A split is a pair of which the first item is always assumed to be the train set.
The second item can either be a single test set or a dictionary with multiple, named test sets.
"""

ListOrArrayType: TypeAlias = list | np.ndarray
"""
A list of numbers or a numpy array. Predictions can be provided as either a list or a numpy array.
"""

IncomingPredictionsType: TypeAlias = ListOrArrayType | dict[str, ListOrArrayType | dict[str, ListOrArrayType]]
"""
The type of the predictions that are ingested into the Polaris BenchmarkPredictions object. Can be one
of the following:

- A single array (single-task, single test set)
- A dictionary of arrays (single-task, multiple test sets)
- A dictionary of dictionaries of arrays (multi-task, multiple test sets)
"""

PredictionsType: TypeAlias = dict[str, dict[str, np.ndarray]]
"""
The normalized format for predictions for internal use. Predictions are accepted in a generous
variety of representations and normalized into this standard format, a dictionary of dictionaries
that looks like {"test_set_name": {"target_name": np.ndarray}}.
"""

DatapointPartType = Any | tuple[Any] | dict[str, Any]
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
A URL-compatible string that can serve as slug on the Hub.
"""

SlugCompatibleStringType: TypeAlias = Annotated[
    str, StringConstraints(pattern="^[A-Za-z0-9_-]+$", min_length=4, max_length=64)
]
"""
A URL-compatible string that can be turned into a slug by the Hub.

Can only use alpha-numeric characters, underscores and dashes.
The string must be at least 4 and at most 64 characters long.
"""

Md5StringType: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{32}$")]
"""
A string that represents an MD5 hash.
"""

HubUser: TypeAlias = SlugCompatibleStringType
"""
A user on the Polaris Hub is identified by a username,
which is a [`SlugCompatibleStringType`][polaris.utils.types.SlugCompatibleStringType].
"""

HttpUrlAdapter = TypeAdapter(HttpUrl)
HttpUrlString: TypeAlias = Annotated[str, BeforeValidator(lambda v: HttpUrlAdapter.validate_python(v) and v)]
"""
A validated HTTP URL that will be turned into a string.
This is useful for interactions with httpx and authlib, who have their own URL types.
"""

AnyUrlAdapter = TypeAdapter(AnyUrl)
AnyUrlString: TypeAlias = Annotated[str, BeforeValidator(lambda v: AnyUrlAdapter.validate_python(v) and v)]
"""
A validated generic URL that will be turned into a string.
This is useful for interactions with other libraries that expect a string.
"""

DirectionType: TypeAlias = float | Literal["min", "max"]
"""
The direction of any variable to be sorted.
This can be used to sort the metric score, indicate the optmization direction of endpoint.
"""

TimeoutTypes = tuple[int, int] | Literal["timeout", "never"]
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

ChecksumStrategy: TypeAlias = Literal["verify", "verify_unless_zarr", "ignore"]
"""
Type to specify which action to take to verify the data integrity of an artifact through a checksum.
"""

ArtifactUrn: TypeAlias = Annotated[str, StringConstraints(pattern=r"^urn:polaris:\w+:\w+:\w+$")]
"""
A Uniform Resource Name (URN) for an artifact on the Polaris Hub.
"""

RowIndex: TypeAlias = int | str
ColumnIndex: TypeAlias = str
DatasetIndex: TypeAlias = RowIndex | tuple[RowIndex, ColumnIndex]
"""
To index a dataset using square brackets, we have a few options:

- A single row, e.g. dataset[0]
- Specify a specific value, e.g. dataset[0, "col1"]

There are more exciting options we could implement, such as slicing,
but this gets complex.
"""


PredictionKwargs: TypeAlias = Literal["y_pred", "y_prob", "y_score"]
"""
The type of predictions expected by the metric interface.
"""

ColumnName: TypeAlias = str
"""A column name in a dataset."""


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

    @staticmethod
    def normalize(owner: str | Self) -> Self:
        """
        Normalize a string or `HubOwner` instance to a `HubOwner` instance.
        """
        return owner if isinstance(owner, HubOwner) else HubOwner(slug=owner)


class TargetType(Enum):
    """The high-level classification of different targets."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    DOCKING = "docking"


class TaskType(Enum):
    """The high-level classification of different tasks."""

    MULTI_TASK = "multi_task"
    SINGLE_TASK = "single_task"
