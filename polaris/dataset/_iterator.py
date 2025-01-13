from math import ceil

import numpy as np
from pyroaring import BitMap
from typing_extensions import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from polaris.dataset import DatasetV2


class ChunkedIterator:
    """Iterate over a dataset in chunks, caching the rows of each chunk."""

    def __init__(
        self,
        total_size: int,
        chunk_size: int,
        shuffle_chunks: bool = False,
        shuffle_within_chunk: bool = False,
        random_state: int | None = None,
        mask: BitMap | None = None,
    ):
        # Config
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.total_size = total_size
        self.chunk_size = chunk_size
        self.random_state = random_state

        if mask is not None and mask.max() > total_size:
            raise ValueError("Mask contains indices greater than the total_size of the iterator.")
        self.mask = mask

        # State
        self.idx = 0
        self.chunk_idx = 0
        self.chunk_start = 0

        self._offset = 0
        self._rng = None
        self._chunk_ordering = None
        self._within_chunk_ordering = None

    @property
    def is_sequential(self):
        return not (self.shuffle_chunks or self.shuffle_within_chunk)

    @property
    def no_chunks(self):
        return ceil(self.total_size / self.chunk_size)

    def reset(self) -> Self:
        """
        Resets the state to the initial state of the iterator
        """
        self.idx = 0
        self.chunk_idx = 0
        self._offset = 0

        self._rng = np.random.default_rng(self.random_state)

        # Shuffle the chunks
        self._chunk_ordering = self._get_ordering(self.no_chunks, self.shuffle_chunks)
        self.chunk_start = self._chunk_ordering[self.chunk_idx] * self.chunk_size

        # Shuffle within a chunk
        # This gets reshuffled whenever we cross a chunk boundary
        size = self._get_chunk_size(self._chunk_ordering[self.chunk_idx])
        self._within_chunk_ordering = self._get_ordering(size, self.shuffle_within_chunk)

        return self

    def update_chunk(self):
        """
        Update the chunk index to the given chunk index.
        """

        # Update the current chunk
        self.chunk_idx += 1
        self.chunk_start = self._chunk_ordering[self.chunk_idx] * self.chunk_size
        self._offset += len(self._within_chunk_ordering)

        # Reshuffle the within chunk ordering
        self._within_chunk_ordering = self._get_ordering(
            self._get_chunk_size(self._chunk_ordering[self.chunk_idx]),
            self.shuffle_within_chunk,
        )

    def _get_chunk_size(self, chunk_idx: int):
        """
        Get the size of the chunk.

        When the chunk is the last one and the total size is not a multiple of the chunk size,
        the size of the last chunk will differ from all others.
        """
        mod = self.total_size % self.chunk_size
        if chunk_idx == self.no_chunks - 1 and mod != 0:
            return mod
        return self.chunk_size

    def _get_ordering(self, n: int, shuffle: bool):
        """
        Return either a random or sequential ordering.

        We assume the ordering fits in memory, because we assume its size to be small.
        This is a decent assumption because the chunk_size and no_chunks tend to be small.
        """
        if self._rng is None:
            raise RuntimeError("Iterator not initialized.")

        # Convert from NumPy's buffers to a list,
        # Copying everything over at once is faster than copying one at a time during iteration
        return self._rng.permutation(n).tolist() if shuffle else list(range(n))

    def _get_chunk_idx(self, idx: int):
        """Get the chunk index from the row index."""
        return idx // self.chunk_size

    def __iter__(self):
        """Initiate the iterator."""
        return self.reset()

    def __next__(self):
        """Get the next row of the dataset"""

        if self.idx >= self.total_size:
            raise StopIteration

        # Check if we cross a chunk boundary
        current_chunk_size = len(self._within_chunk_ordering)
        if self.idx - self._offset >= current_chunk_size:
            self.update_chunk()

        if self.mask is not None and self.idx not in self.mask:
            self.idx += 1
            return self.__next__()

        i = (
            self.idx
            if self.is_sequential
            else self.chunk_start + self._within_chunk_ordering[self.idx - self._offset]
        )

        # Update the state
        self.idx += 1
        return i


class CachedChunkIterator(ChunkedIterator):
    def __init__(
        self,
        dataset: "DatasetV2",
        column: str,
        cache_multiplier: int = 1,
        shuffle_chunks: bool = False,
        shuffle_within_chunk: bool = False,
        random_state: int | None = None,
        mask: BitMap | None = None,
    ):
        super().__init__(
            total_size=len(dataset),
            chunk_size=dataset.zarr_root[column].chunks[0] * cache_multiplier,
            shuffle_chunks=shuffle_chunks,
            shuffle_within_chunk=shuffle_within_chunk,
            random_state=random_state,
            mask=mask,
        )

        self.dataset = dataset
        self.column = column
        self.chunk_cache = None

    def load_current_chunk(self) -> np.ndarray:
        start_idx = self.chunk_start
        end_idx = self.chunk_start + len(self._within_chunk_ordering)
        return self.dataset.zarr_data[self.column][start_idx:end_idx]

    def reset(self) -> Self:
        super().reset()
        self.chunk_cache = self.load_current_chunk()
        return self

    def update_chunk(self):
        super().update_chunk()
        self.chunk_cache = self.load_current_chunk()

    def __next__(self):
        """Get the next row of the dataset"""
        idx = super().__next__()
        return self.chunk_cache[idx - self._offset]
