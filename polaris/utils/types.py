from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, computed_field, model_validator
from typing_extensions import TypeAlias

SplitIndicesType: TypeAlias = List[int]
"""
A split is defined by a sequence of integers.
"""


SplitType: TypeAlias = Tuple[SplitIndicesType, Union[SplitIndicesType, Dict[str, SplitIndicesType]]]
"""
A split is a pair of which the first item is always assumed to be the train set.
The second item can either be a single test set or a dictionary with multiple, named test sets.
"""


PredictionsType: TypeAlias = Union[np.ndarray, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]
"""
A prediction is one of three things:

- A single array (single-task, single test set)
- A dictionary of arrays (single-task, multiple test sets) 
- A dictionary of dictionaries of arrays (multi-task, multiple test sets)
"""

DatapointType: TypeAlias = Tuple[Union[Any, Tuple, Dict[str, Any]], Union[Any, Tuple, Dict[str, Any]]]
"""
A datapoint has:

- A single input or multiple inputs (either as dict or tuple)
- No target, a single target or a multiple targets (either as dict or tuple)
"""


DataFormat: TypeAlias = Literal["dict", "tuple"]
"""
The target formats that are supported by the `Subset` class. 
"""


class HubOwner(BaseModel):
    """An owner of an artifact on the Polaris Hub"""

    organizationId: Optional[str] = None
    userId: Optional[str] = None

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
        if isinstance(self.organizationId, str):
            return self.organizationId
        if isinstance(self.userId, str):
            return self.userId
        raise ValueError("Either `organization` or `user` must be specified, but not both.")

    def __str__(self) -> str:
        return self.owner

    def __repr__(self) -> str:
        return self.__str__()
