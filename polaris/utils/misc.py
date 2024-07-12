from typing import TYPE_CHECKING, Any

from polaris.utils.types import ChecksumStrategy, SlugCompatibleStringType

if TYPE_CHECKING:
    from polaris.dataset import Dataset


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


def should_verify_checksum(strategy: ChecksumStrategy, dataset: "Dataset") -> bool:
    """
    Determines whether a checksum should be verified.
    """
    if strategy == "ignore":
        return False
    elif strategy == "verify":
        return True
    else:
        return not dataset.uses_zarr
