import json
from typing import Annotated, Any, ClassVar, Literal, Optional, Union

import fsspec
import numpy as np
from loguru import logger
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    HttpUrl,
    computed_field,
    constr,
    model_validator,
)
from typing_extensions import TypeAlias

from polaris.utils.misc import to_lower_camel

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

DataFormat: TypeAlias = Literal["dict", "tuple"]
"""
The target formats that are supported by the `Subset` class. 
"""

SlugCompatibleStringType: TypeAlias = constr(pattern="^[A-Za-z0-9_-]+$", min_length=4, max_length=64)
"""
A URL-compatible string that can serve as slug on the hub.

Can only use alpha-numeric characters, underscores and dashes. 
The string must be at least 4 and at most 64 characters long.
"""


HubUser: TypeAlias = SlugCompatibleStringType
"""
A user on the Polaris Hub is identified by a username, 
which is a [`SlugCompatibleStringType`][polaris.utils.types.SlugCompatibleStringType].
"""

HttpUrlString: TypeAlias = Annotated[HttpUrl, AfterValidator(str)]
"""
A validated URL that will be turned into a string.
This is useful for interactions with httpx and authlib, who have their own URL types.
"""

DirectionType: TypeAlias = Literal["min", "max"]
"""
The direction of any variable to be sorted.
This can be used to sort the metric score, indicate the optmization direction of endpoint.
"""

AccessType: TypeAlias = Literal["public", "private"]
"""
Type to specify access to a dataset, benchmark or result in the Hub.
"""


class HubOwner(BaseModel):
    """An owner of an artifact on the Polaris Hub

    The slug is most important as it is the user-facing part of this data model. The organization
    and user id are added to be consistent with the Polaris Hub.

    The username is specified as a [`SlugCompatibleStringType`][polaris.utils.types.SlugCompatibleStringType],
    whereas the organization is specified as a string that can contain only alpha-numeric characters,
    underscores and dashes. Contrary to the username, an organization name can currently be of arbitrary length.
    """

    slug: constr(pattern="^[A-Za-z0-9_-]+$")
    organization_id: Optional[constr(pattern="^[A-Za-z0-9_-]+$")] = None
    user_id: Optional[HubUser] = None

    # Pydantic config
    model_config = ConfigDict(alias_generator=to_lower_camel, populate_by_name=True)

    @model_validator(mode="after")  # type: ignore
    @classmethod
    def _validate_model(cls, m: "HubOwner"):
        if m.organization_id is not None and m.user_id is not None:
            raise ValueError("An owner cannot both have an `organization_id` and a `user_id`")
        return m

    @computed_field
    @property
    def owner(self) -> str:
        return self.organization_id or self.user_id  # type: ignore

    def __str__(self) -> str:
        return self.slug

    def __repr__(self) -> str:
        return self.__str__()


class License(BaseModel):
    """An artifact license.

    Attributes:
        id: The license ID. Either from [SPDX](https://spdx.org/licenses/) or custom.
        reference: A reference to the license text. If the ID is found in SPDX, this is automatically set.
            Else it is required to manually specify this.
    """

    SPDX_LICENSE_DATA_PATH: ClassVar[
        str
    ] = "https://raw.githubusercontent.com/spdx/license-list-data/main/json/licenses.json"

    id: str
    reference: Optional[HttpUrlString] = None

    @model_validator(mode="after")  # type: ignore
    @classmethod
    def _validate_license_id(cls, m: "License"):
        """
        If a license ID exists in the SPDX database, we use the reference from there.
        Otherwise, it is up to the user to specify the license.
        """

        # Load the ground truth references
        with fsspec.open(cls.SPDX_LICENSE_DATA_PATH) as f:
            data = json.load(f)
        data = {license["licenseId"]: license for license in data["licenses"]}

        if m.id in data:
            if m.reference is not None and m.reference != data[m.id]["reference"]:
                logger.warning(f"Found license ID {m.id} in SPDX, using the associated reference.")
            m.reference = data[m.id]["reference"]

        if m.id not in data and m.reference is None:
            raise ValueError(
                f"License with ID {m.id} not found in SPDX. "
                "It is required to then also specify the name and reference."
            )
        return m
