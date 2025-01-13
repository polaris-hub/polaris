import pytest

from polaris.dataset._iterator import ChunkedIterator, chunk_iterator


@pytest.mark.parametrize("chunk_size", [1, 3, 5, 10, 100, 500])
def test_chunked_iterator_sequential_access(chunk_size):
    """Check if the ordering with sequential access is as expected"""
    assert list(ChunkedIterator(100, chunk_size)) == list(range(100))


def test_chunked_iterator_random_chunks_access():
    """
    Shuffle the chunks!

    For example: [10, 11, ..., 19] -> [40, 41, ..., 49] -> ... -> [70, 71, ..., 79]
    """

    chunk_size = 10
    indices = list(ChunkedIterator(100, chunk_size, shuffle_chunks=True))

    for i in range(10):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        start = indices[start_idx]
        end = start + chunk_size

        found = indices[start_idx:end_idx]
        expected = list(range(start, end))
        assert found == expected


def test_chunked_iterator_fully_random_access():
    """
    Shuffle both the chunks as well as the items within a chunk

    For example: [19, 13, ..., 16] -> [40, 45, ..., 47] -> ... -> [71, 73, ..., 76]
    """

    chunk_size = 10
    indices = list(ChunkedIterator(100, chunk_size, shuffle_chunks=True, shuffle_within_chunk=True))

    for i in range(10):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        found = indices[start_idx:end_idx]

        start = min(found)
        end = start + chunk_size

        expected = list(range(start, end))

        assert len(found) == len(expected)
        assert set(found) == set(expected)


def test_chunked_iterator_random_state():
    """
    Using the same random state twice should yield the same result
    """

    iterator = ChunkedIterator(
        100,
        10,
        shuffle_chunks=True,
        shuffle_within_chunk=True,
        random_state=42,
    )

    indices_1 = list(iterator)
    indices_2 = list(iterator)
    assert indices_1 == indices_2

    iterator = ChunkedIterator(
        100,
        10,
        shuffle_chunks=True,
        shuffle_within_chunk=True,
        random_state=42,
    )
    indices_3 = list(iterator)
    assert indices_1 == indices_3

    iterator = ChunkedIterator(
        100,
        10,
        shuffle_chunks=True,
        shuffle_within_chunk=True,
        random_state=0,
    )
    indices_4 = list(iterator)
    assert indices_1 != indices_4


@pytest.mark.parametrize("chunk_size", [1, 3, 5, 10, 100, 500])
def test_chunk_iterator_sequential_access(chunk_size):
    """
    Check if the ordering with sequential access is as expected
    """
    content = list(range(100))
    result = list(chunk_iterator(content, 100, chunk_size))

    assert result == content


def test_chunk_iterator_random_chunks_access():
    """
    Shuffle the chunks!

    For example: [10, 11, ..., 19] -> [40, 41, ..., 49] -> ... -> [70, 71, ..., 79]
    """
    content = list(range(100))
    chunk_size = 10

    result = list(chunk_iterator(content, 100, chunk_size, shuffle_chunks=True))

    assert set(content) == set(result)
    assert content != result

    for i in range(0, len(result), chunk_size):
        chunk = result[i : i + chunk_size]
        assert chunk == list(sorted(chunk))


def test_chunk_iterator_fully_random_access():
    """
    Shuffle both the chunks and the items within a chunk

    For example: [19, 13, ..., 16] -> [40, 45, ..., 47] -> ... -> [71, 73, ..., 76]
    """
    content = list(range(100))
    chunk_size = 10

    result = list(chunk_iterator(content, 100, chunk_size, shuffle_chunks=True, shuffle_within_chunk=True))

    assert set(content) == set(result)
    assert content != result

    for i in range(0, len(result), chunk_size):
        chunk = result[i : i + chunk_size]
        assert chunk != list(sorted(chunk))


def test_chunk_iterator_random_state():
    """
    Using the same random state twice should yield the same result
    """
    content = list(range(100))

    iterator_1 = chunk_iterator(
        content,
        100,
        10,
        shuffle_chunks=True,
        shuffle_within_chunk=True,
        random_state=42,
    )

    iterator_2 = chunk_iterator(
        content,
        100,
        10,
        shuffle_chunks=True,
        shuffle_within_chunk=True,
        random_state=42,
    )

    indices_1 = list(iterator_1)
    indices_2 = list(iterator_2)
    assert indices_1 == indices_2

    iterator_3 = chunk_iterator(
        content,
        100,
        10,
        shuffle_chunks=True,
        shuffle_within_chunk=True,
        random_state=0,
    )
    indices_3 = list(iterator_3)
    assert indices_1 != indices_3
