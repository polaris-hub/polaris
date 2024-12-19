import numpy as np
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from polaris.experimental._dataset_v2 import DatasetV2


class CachedChunksIterator:
    """Iterate over a dataset in chunks, caching the rows of each chunk."""

    def __init__(
        self,
        dataset: "DatasetV2",
        columns: list[str],
        chunk_size: int,
        shuffle_within_chunk: bool = False,
    ):
        # Dataset
        self.dataset = dataset
        self.columns = columns

        # Config
        self.shuffle_within_chunk = shuffle_within_chunk
        self.chunk_size = chunk_size

        # State
        self.idx = 0
        self._chunk_idx = 0
        self._chunk_cache = {}
        self._chunk_ordering = None

        # Initialize state
        self._prepare_chunk(chunk_idx=self._chunk_idx)

    def _get_chunk_idx(self, idx: int):
        """Get the chunk index from the row index."""
        return idx // self.chunk_size

    def _load_chunk(self, chunk_idx: int, column: str):
        """Load a chunk of data from Zarr to a NumPy array"""
        start_idx = chunk_idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        return self.dataset.zarr_data[column][start_idx:end_idx]

    def _prepare_chunk(self, chunk_idx: int):
        """Cache the rows of a chunk."""
        print(f"Preparing chunk {chunk_idx}")
        # Cache
        for column in self.columns:
            self._chunk_cache[column] = self._load_chunk(chunk_idx, column)

        # Indices within chunk
        chunk_size = min(len(self._chunk_cache[c]) for c in self.columns)
        if self.shuffle_within_chunk:
            self._chunk_ordering = np.random.permutation(chunk_size)
        else:
            self._chunk_ordering = np.arange(chunk_size)

        # Set chunk index
        self._chunk_idx = chunk_idx

    def _load_row(self, idx: int):
        """Load a row of data from Zarr to a dictionary."""

        chunk_idx = self._get_chunk_idx(idx)
        if chunk_idx != self._chunk_idx:
            self._prepare_chunk(chunk_idx=chunk_idx)

        row_idx = idx % self.chunk_size
        data = {c: self._chunk_cache[c][row_idx] for c in self.columns}

        if len(data) == 1:
            data = data[self.columns[0]]
        return data

    def __iter__(self):
        """Initiate the iterator."""
        self.idx = 0
        self._chunk_idx = 0
        return self

    def __next__(self):
        """Get the next row of the dataset"""
        if self.idx >= len(self.dataset):
            raise StopIteration

        row = self._load_row(self.idx)
        self.idx += 1

        return row
