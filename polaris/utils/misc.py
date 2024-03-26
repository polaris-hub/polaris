from typing import Any, Literal

from packaging.version import Version

from polaris.utils.types import SlugCompatibleStringType


def listit(t: Any):
    """
    Converts all tuples in a possibly nested object to lists
    https://stackoverflow.com/questions/1014352/how-do-i-convert-a-nested-tuple-of-tuples-and-lists-to-lists-of-lists-in-python
    """
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def sluggify(sluggable: SlugCompatibleStringType):
    """
    Converts a string to a slug-compatible string.
    """
    return sluggable.lower().replace("_", "-")


def str_to_version(value: str) -> Version | Literal["dev"]:
    """
    Convert a string to a Version object if possible, unless it is 'dev'
    """
    if value == "dev":
        return "dev"
    return Version(value)
