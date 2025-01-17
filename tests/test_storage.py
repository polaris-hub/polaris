import obstore as obs
import pytest
from obstore.store import MemoryStore, PrefixStore

from polaris.hub.storage import S3Store


@pytest.fixture(scope="function")
def s3_store():
    # Create an instance of your S3Store
    store = S3Store(
        path="test_bucket/prefix",
        access_key="fake-access-key",
        secret_key="fake-secret-key",
        token="fake-token",
        endpoint_url="https://s3.amazonaws.com"
    )

    # Mock the backing Obstore
    store.backend_store = PrefixStore(MemoryStore(), "/prefix")

    yield store


def test_set_and_get_item(s3_store):
    key = "test-key"
    value = b"test-value"
    s3_store[key] = value

    retrieved_value = s3_store[key]
    assert retrieved_value == value


def test_get_nonexistent_item(s3_store):
    with pytest.raises(KeyError):
        _ = s3_store["nonexistent-key"]


def test_contains_item(s3_store):
    key = "test-key"
    value = b"test-value"
    s3_store[key] = value

    assert key in s3_store
    assert "nonexistent-key" not in s3_store


def test_store_iterator_empty(s3_store):
    stored_keys = list(s3_store)
    assert stored_keys == []


def test_store_iterator(s3_store):
    keys = ["dir1/subdir1", "dir1/subdir2", "dir1/file1.ext", "dir2/file2.ext"]
    for key in keys:
        s3_store[key] = b"test"

    stored_keys = list(s3_store)
    assert sorted(stored_keys) == sorted(keys)


def test_store_length(s3_store):
    keys = ["dir1/subdir1", "dir1/subdir2", "dir1/file1.ext", "dir2/file2.ext"]
    for key in keys:
        s3_store[key] = b"test"

    assert len(s3_store) == len(keys)


def test_listdir(s3_store):
    keys = ["dir1/subdir1/subfile1.ext", "dir1/subdir2/subfile1.ext", "dir1/file1.ext", "dir2/file2.ext"]
    for key in keys:
        s3_store[key] = b"test"

    dir1_contents = list(s3_store.listdir("dir1/"))
    print(obs.list_with_delimiter(s3_store.backend_store, "dir1"))
    assert set(dir1_contents) == {"file1.ext", "subdir1", "subdir2"}

    dir1_contents = list(s3_store.listdir())
    print(obs.list_with_delimiter(s3_store.backend_store))
    assert set(dir1_contents) == {"dir1", "dir2"}


def test_getsize(s3_store):
    key = "test-key"
    value = b"test-value"
    s3_store[key] = value

    size = s3_store.getsize(key)
    assert size == len(value)


def test_getitems(s3_store):
    keys = ["dir1/subdir1", "dir1/subdir2", "dir1/file1.ext", "dir2/file2.ext"]
    for key in keys:
        s3_store[key] = b"test"

    items = s3_store.getitems(keys, contexts={})
    assert len(items) == len(keys)
    assert all(key in items for key in keys)


def test_delete_item_not_supported(s3_store):
    with pytest.raises(NotImplementedError):
        del s3_store["some-key"]
