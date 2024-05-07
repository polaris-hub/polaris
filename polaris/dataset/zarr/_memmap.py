import mmap

import zarr


class MemoryMappedDirectoryStore(zarr.DirectoryStore):
    """
    A Zarr Store to open chunks as memory-mapped files.
    See https://github.com/zarr-developers/zarr-python/issues/1245

    Memory mapping leverages low-level OS functionality to reduce the time it takes
    to read the content of a file by directly mapping to memory.
    """

    def _fromfile(self, fn):
        with open(fn, "rb") as fh:
            return memoryview(mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ))
