import numpy as np
from typing_extensions import TypeAlias
from typing import List, Tuple, Union, Dict, Any

# A split is defined by a sequence of integers (single-task) or by a sequence of integer pairs (multi-task)
SplitIndices: TypeAlias = List[Union[int, Tuple[int, Union[List[int], int]]]]

# A split is a pair of which the first item is always assumed to be the train set.
# The second item can either be a single test set or a dictionary with multiple, named test sets.
Split: TypeAlias = Tuple[SplitIndices, Union[SplitIndices, Dict[str, SplitIndices]]]

# A prediction is one of three things:
# - A single array (single-task, single test set)
# - A dictionary of arrays (single-task, multiple test sets)
# - A dictionary of dictionaries of arrays (multi-task, multiple test sets)
Predictions: TypeAlias = Union[np.ndarray, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]

# A datapoint has:
# - A single input or multiple inputs (either as dict or tuple)
# - No target, a single target or a multiple targets (either as dict or tuple)
Datapoint: TypeAlias = Tuple[Union[Any, Tuple, Dict[str, Any]], Union[Any, Tuple, Dict[str, Any]]]
