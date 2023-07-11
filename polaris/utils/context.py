from contextlib import contextmanager


@contextmanager
def tmp_attribute_change(obj, attribute, value):
    """Temporarily set and reset an attribute of an object."""
    original_value = getattr(obj, attribute)
    try:
        setattr(obj, attribute, value)
        yield obj
    finally:
        setattr(obj, attribute, original_value)
