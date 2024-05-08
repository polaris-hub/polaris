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

from pathlib import Path
from shutil import copytree

import pytest
import zarr

from polaris.dataset.zarr._checksum import (
    EMPTY_CHECKSUM,
    InvalidZarrChecksum,
    ZarrChecksum,
    ZarrChecksumManifest,
    ZarrChecksumTree,
    ZarrDirectoryDigest,
    compute_zarr_checksum,
)


def test_generate_digest() -> None:
    manifest = ZarrChecksumManifest(
        directories=[ZarrChecksum(digest="a7e86136543b019d72468ceebf71fb8e-1--1", name="a/b", size=1)],
        files=[ZarrChecksum(digest="92eb5ffee6ae2fec3ad71c777531578f-1--1", name="b", size=1)],
    )
    assert manifest.generate_digest().digest == "2ed39fd5ae56fd4177c4eb503d163528-2--2"


def test_zarr_checksum_sort_order() -> None:
    # The a < b in the name should take precedence over z > y in the md5
    a = ZarrChecksum(name="a", digest="z", size=3)
    b = ZarrChecksum(name="b", digest="y", size=4)
    assert sorted([b, a]) == [a, b]


def test_parse_zarr_directory_digest() -> None:
    # Parse valid
    ZarrDirectoryDigest.parse("c228464f432c4376f0de6ddaea32650c-37481--38757151179")
    ZarrDirectoryDigest.parse(None)

    # Ensure exception is raised
    with pytest.raises(InvalidZarrChecksum):
        ZarrDirectoryDigest.parse("asd")
    with pytest.raises(InvalidZarrChecksum):
        ZarrDirectoryDigest.parse("asd-0--0")


def test_pop_deepest() -> None:
    tree = ZarrChecksumTree()
    tree.add_leaf(Path("a/b"), size=1, digest="asd")
    tree.add_leaf(Path("a/b/c"), size=1, digest="asd")
    node = tree.pop_deepest()

    # Assert popped node is a/b/c, not a/b
    assert str(node.path) == "a/b"
    assert len(node.checksums.files) == 1
    assert len(node.checksums.directories) == 0
    assert node.checksums.files[0].name == "c"


def test_process_empty_tree() -> None:
    tree = ZarrChecksumTree()
    assert tree.process().digest == EMPTY_CHECKSUM


def test_process_tree() -> None:
    tree = ZarrChecksumTree()
    tree.add_leaf(Path("a/b"), size=1, digest="9dd4e461268c8034f5c8564e155c67a6")
    tree.add_leaf(Path("c"), size=1, digest="415290769594460e2e485922904f345d")
    checksum = tree.process()

    # This zarr checksum was computed against the same file structure using the previous
    # zarr checksum implementation
    # Assert the current implementation produces a matching checksum
    assert checksum.digest == "26054e501f570a8bfa69a2bc75e7c82d-2--2"


def test_checksum_for_zarr_archive(zarr_archive, tmpdir):
    # NOTE: This test was not in the original code base of the zarr-checksum package.
    checksum = compute_zarr_checksum(zarr_archive)

    path = tmpdir.join("copy")
    copytree(zarr_archive, path)
    assert checksum == compute_zarr_checksum(path)

    root = zarr.open(path)
    root["A"][0:10] = 0
    assert checksum != compute_zarr_checksum(path)
