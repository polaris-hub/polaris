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
