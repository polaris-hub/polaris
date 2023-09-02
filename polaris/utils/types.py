from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, computed_field, constr, field_validator, model_validator
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

DataFormat: TypeAlias = Literal["dict", "tuple"]
"""
The target formats that are supported by the `Subset` class. 
"""

SlugCompatibleStringType: TypeAlias = constr(pattern="^[A-Za-z0-9_-]+$", min_length=3)
"""
A URL-compatible string that can serve as slug on the hub.

Can only use alpha-numeric characters, underscores and dashes. 
Cannot end or start with a dash or underscore.
The string must be at least 3 characters long.
"""


class HubOwner(BaseModel):
    """An owner of an artifact on the Polaris Hub

    Either specifies an organization or a user, but not both.
    Specified as a [`SlugCompatibleStringType`][polaris.utils.types.SlugCompatibleStringType].
    """

    organizationId: Optional[SlugCompatibleStringType] = None
    userId: Optional[SlugCompatibleStringType] = None

    @field_validator("organizationId", "userId")
    def _validate_name(cls, v):
        """
        Since look-around does not work with Pydantic's constr regex,
        this verifies that the string does not end with a dash or underscore.
        """
        if v is None:
            return v
        if v.endswith("-") or v.endswith("_") or v.startswith("-") or v.startswith("_"):
            raise ValueError("String cannot end with a dash or underscore.")
        return v

    @model_validator(mode="after")  # type: ignore
    @classmethod
    def _validate_model(cls, m: "HubOwner"):
        if (m.organizationId is None and m.userId is None) or (
            m.organizationId is not None and m.userId is not None
        ):
            raise ValueError("Either `organization` or `user` must be specified, but not both.")
        return m

    @computed_field
    @property
    def owner(self) -> str:
        return self.organizationId or self.userId  # type: ignore

    def __str__(self) -> str:
        return self.owner

    def __repr__(self) -> str:
        return self.__str__()
