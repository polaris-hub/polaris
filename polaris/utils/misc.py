from typing import Any

import numpy as np

from polaris.utils.types import ListOrArrayType, SlugCompatibleStringType, SlugStringType


def listit(t: Any):
    """
    Converts all tuples in a possibly nested object to lists
    https://stackoverflow.com/questions/1014352/how-do-i-convert-a-nested-tuple-of-tuples-and-lists-to-lists-of-lists-in-python
    """
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def slugify(sluggable: SlugCompatibleStringType) -> SlugStringType:
    """
    Converts a slug-compatible string to a slug.
    """
    return sluggable.lower().replace("_", "-").strip("-")


def convert_lists_to_arrays(predictions: ListOrArrayType | dict) -> np.ndarray | dict:
    """
    Recursively converts all plain Python lists in the predictions object to numpy arrays
    """

    def convert_to_array(v):
        if isinstance(v, np.ndarray):
            return v
        elif isinstance(v, list):
            return np.array(v)
        elif isinstance(v, dict):
            return {k: convert_to_array(v) for k, v in v.items()}

    return convert_to_array(predictions)
