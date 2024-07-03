"""
The code in this file is based on the zarr-checksum package

Mainted by Jacob Nesbitt, released under the DANDI org on Github
and with Kitware, Inc. credited as the author. This code is released
with the Apache 2.0 license.

See also: https://github.com/dandi/zarr_checksum

Instead of adding the package as a dependency, we opted to copy over the code
because it is a small and self-contained module that we will want to alter to
support our Polaris code base.

NOTE: We have made some modifications to the original code.

----

Copyright 2023 Kitware, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import uuid
from pathlib import Path
from shutil import copytree, rmtree

import pytest
import zarr

from polaris.dataset.zarr._checksum import (
    EMPTY_CHECKSUM,
    InvalidZarrChecksum,
    _ZarrChecksum,
    _ZarrChecksumManifest,
    _ZarrChecksumTree,
    _ZarrDirectoryDigest,
    compute_zarr_checksum,
)


def test_generate_digest() -> None:
    manifest = _ZarrChecksumManifest(
        directories=[_ZarrChecksum(digest="a7e86136543b019d72468ceebf71fb8e-1-1", name="a/b", size=1)],
        files=[_ZarrChecksum(digest="92eb5ffee6ae2fec3ad71c777531578f-0-1", name="b", size=1)],
    )
    assert manifest.generate_digest().digest == "9c5294e46908cf397cb7ef53ffc12efc-1-2"


def test_zarr_checksum_sort_order() -> None:
    # The a < b in the name should take precedence over z > y in the md5
    a = _ZarrChecksum(name="a", digest="z", size=3)
    b = _ZarrChecksum(name="b", digest="y", size=4)
    assert sorted([b, a]) == [a, b]


def test_parse_zarr_directory_digest() -> None:
    # Parse valid
    _ZarrDirectoryDigest.parse("c228464f432c4376f0de6ddaea32650c-37481-38757151179")
    _ZarrDirectoryDigest.parse(None)

    # Ensure exception is raised
    with pytest.raises(InvalidZarrChecksum):
        _ZarrDirectoryDigest.parse("asd")
    with pytest.raises(InvalidZarrChecksum):
        _ZarrDirectoryDigest.parse("asd-0--0")


def test_pop_deepest() -> None:
    tree = _ZarrChecksumTree()
    tree.add_leaf(Path("a/b"), size=1, digest="asd")
    tree.add_leaf(Path("a/b/c"), size=1, digest="asd")
    node = tree.pop_deepest()

    # Assert popped node is a/b/c, not a/b
    assert str(node.path) == "a/b"
    assert len(node.checksums.files) == 1
    assert len(node.checksums.directories) == 0
    assert node.checksums.files[0].name == "c"


def test_process_empty_tree() -> None:
    tree = _ZarrChecksumTree()
    assert tree.process().digest == EMPTY_CHECKSUM


def test_process_tree() -> None:
    tree = _ZarrChecksumTree()
    tree.add_leaf(Path("a/b"), size=1, digest="9dd4e461268c8034f5c8564e155c67a6")
    tree.add_leaf(Path("c"), size=1, digest="415290769594460e2e485922904f345d")
    checksum = tree.process()

    # This zarr checksum was computed against the same file structure using the previous
    # zarr checksum implementation
    # Assert the current implementation produces a matching checksum
    assert checksum.digest == "e53fcb7b5c36b2f4647fbf826a44bdc9-2-2"


def test_checksum_for_zarr_archive(zarr_archive, tmpdir):
    # NOTE: This test was not in the original code base of the zarr-checksum package.
    checksum, _ = compute_zarr_checksum(zarr_archive)

    path = tmpdir.join("copy")
    copytree(zarr_archive, path)
    assert checksum == compute_zarr_checksum(str(path))[0]

    root = zarr.open(path)
    root["A"][0:10] = 0
    assert checksum != compute_zarr_checksum(str(path))[0]


def test_zarr_leaf_to_checksum(zarr_archive):
    # NOTE: This test was not in the original code base of the zarr-checksum package.
    _, leaf_to_checksum = compute_zarr_checksum(zarr_archive)
    root = zarr.open(zarr_archive)

    # Check the basic structure - Each key corresponds to a file in the zarr archive
    assert len(leaf_to_checksum) == len(root.store)
    assert all(k.path in root.store for k in leaf_to_checksum)


def test_zarr_checksum_fails_for_remote_storage(zarr_archive):
    # NOTE: This test was not in the original code base of the zarr-checksum package.
    with pytest.raises(RuntimeError):
        compute_zarr_checksum("s3://bucket/data.zarr")
    with pytest.raises(RuntimeError):
        compute_zarr_checksum("gs://bucket/data.zarr")


def test_zarr_checksum_with_path_normalization(zarr_archive):
    # NOTE: This test was not in the original code base of the zarr-checksum package.

    baseline = compute_zarr_checksum(zarr_archive)[0]
    rootdir = os.path.dirname(zarr_archive)

    # Test a relative path
    copytree(zarr_archive, os.path.join(rootdir, "relative", "data.zarr"))
    compute_zarr_checksum(f"{zarr_archive}/../relative/data.zarr")[0] == baseline

    # Test with variables
    rng_id = str(uuid.uuid4())
    os.environ["TMP_TEST_DIR"] = rng_id
    copytree(zarr_archive, os.path.join(rootdir, "vars", rng_id))
    compute_zarr_checksum(f"{rootdir}/vars/${{TMP_TEST_DIR}}")[0] == baseline  # Format ${...}
    compute_zarr_checksum(f"{rootdir}/vars/$TMP_TEST_DIR")[0] == baseline  # Format $...

    # And with the user abbreviation
    try:
        path = os.path.expanduser("~/data.zarr")
        copytree(zarr_archive, path)
        compute_zarr_checksum("~/data.zarr")[0] == baseline
    finally:
        rmtree(path)
