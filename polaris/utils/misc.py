from typing import Any, TYPE_CHECKING

from polaris.utils.types import ChecksumStrategy, SlugCompatibleStringType, SlugStringType

if TYPE_CHECKING:
    from polaris.dataset import DatasetV1


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


def should_verify_checksum(strategy: ChecksumStrategy, dataset: "DatasetV1") -> bool:
    """
    Determines whether a checksum should be verified.
    """
    if strategy == "ignore":
        return False
    elif strategy == "verify":
        return True
    else:
        return not dataset.uses_zarr
