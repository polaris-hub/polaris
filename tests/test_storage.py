import os

import boto3
import pytest
from moto import mock_aws

from polaris.hub.storage import S3Store


@pytest.fixture(scope="function")
def aws_credentials():
    """
    Mocked AWS Credentials for moto.
    """
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"


@pytest.fixture(scope="function")
def mocked_aws(aws_credentials):
    """
    Mock all AWS interactions
    Requires you to create your own boto3 clients
    """
    with mock_aws():
        yield


@pytest.fixture
def s3_store(mocked_aws):
    # Setup mock S3 environment
    s3 = boto3.client("s3", region_name="us-east-1")
    bucket_name = "test-bucket"
    s3.create_bucket(Bucket=bucket_name)

    # Create an instance of your S3Store
    store = S3Store(
        path=f"{bucket_name}/prefix",
        access_key="fake-access-key",
        secret_key="fake-secret-key",
        token="fake-token",
        endpoint_url="https://s3.amazonaws.com",
    )

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
    keys = ["dir1/subdir1", "dir1/subdir2", "dir1/file1.ext", "dir2/file2.ext"]
    for key in keys:
        s3_store[key] = b"test"

    dir1_contents = list(s3_store.listdir("dir1"))
    assert set(dir1_contents) == {"file1.ext", "subdir1", "subdir2"}

    dir1_contents = list(s3_store.listdir())
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
