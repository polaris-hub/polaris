from typing import Any


def listit(t: Any):
    """
    Converts all tuples in a possibly nested object to lists
    https://stackoverflow.com/questions/1014352/how-do-i-convert-a-nested-tuple-of-tuples-and-lists-to-lists-of-lists-in-python
    """
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t
