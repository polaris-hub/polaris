import zarr
import zarr.storage


def load_zarr_group_to_memory(group: zarr.Group) -> dict:
    """Loads an entire Zarr group into memory."""
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
        message = str(error)
        codec_id = message.removeprefix(prefix)

        if message.startswith(prefix):
            raise RuntimeError(
                f"This Zarr archive requires the {codec_id} codec. Install all optional codecs with 'pip install polaris-lib[codecs]'."
            ) from error
