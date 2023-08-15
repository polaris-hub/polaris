from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
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
