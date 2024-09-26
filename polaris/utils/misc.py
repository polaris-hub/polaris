from typing import Any

from polaris.utils.types import SlugCompatibleStringType, SlugStringType


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
