from typing import Any, Literal

import numpy as np
from typing_extensions import TypeAlias

SplitIndicesType: TypeAlias = list[int]
"""
A split is defined by a sequence of integers.
"""


SplitType: TypeAlias = tuple[SplitIndicesType, SplitIndicesType | dict[str, SplitIndicesType]]
"""
A split is a pair of which the first item is always assumed to be the train set.
The second item can either be a single test set or a dictionary with multiple, named test sets.
"""


PredictionsType: TypeAlias = np.ndarray | dict[str, np.ndarray | dict[str, np.ndarray]]
"""
A prediction is one of three things:

- A single array (single-task, single test set)
- A dictionary of arrays (single-task, multiple test sets) 
- A dictionary of dictionaries of arrays (multi-task, multiple test sets)
"""

DatapointType: TypeAlias = tuple[Any | tuple | dict[str, Any], Any | tuple | dict[str, Any]]
"""
A datapoint has:

- A single input or multiple inputs (either as dict or tuple)
- No target, a single target or a multiple targets (either as dict or tuple)
"""


DataFormat: TypeAlias = Literal["dict", "tuple"]
"""
The target formats that are supported by the `Subset` class. 
"""
