import zarr
import zarr.storage

from polaris.utils.errors import InvalidZarrCodec

try:
    # Register imagecodecs if they are available.
    from imagecodecs.numcodecs import register_codecs

    register_codecs()
except ImportError:
    pass


def load_zarr_group_to_memory(group: zarr.Group) -> dict:
    """Loads an entire Zarr group into memory."""

    if isinstance(group, dict):
        # If a Zarr group is already loaded to memory (e.g. with dataset.load_to_memory()),
        # the adapter would receive a dictionary instead of a Zarr group.
        return group

    data = {}
    for key, item in group.items():
        if isinstance(item, zarr.Array):
            data[key] = item[:]
        elif isinstance(item, zarr.Group):
            data[key] = load_zarr_group_to_memory(item)
    return data


def check_zarr_codecs(group: zarr.Group):
    """Check if all codecs in the Zarr group are registered."""
    try:
        for key, item in group.items():
            if isinstance(item, zarr.Group):
                check_zarr_codecs(item)

    except ValueError as error:
        # Zarr raises a generic ValueError if a codec is not registered.
        # See also: https://github.com/zarr-developers/zarr-python/issues/2508
        prefix = "codec not available: "
        error_message = str(error)

        if not error_message.startswith(prefix):
            raise error

        # Remove prefix and apostrophes
        codec_id = error_message.removeprefix(prefix).strip("'")
        raise InvalidZarrCodec(codec_id)
