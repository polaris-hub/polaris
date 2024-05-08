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

import hashlib
import heapq
import os
import re
from dataclasses import asdict, dataclass, field
from functools import total_ordering
from json import dumps
from pathlib import Path
from typing import Optional

import datamol as dm
import zarr
import zarr.errors
from fsspec import AbstractFileSystem
from tqdm import tqdm

from polaris.utils.errors import InvalidZarrChecksum

ZARR_DIGEST_PATTERN = "([0-9a-f]{32})-([0-9]+)--([0-9]+)"


def compute_zarr_checksum(zarr_root_path: str, fs: Optional[AbstractFileSystem] = None) -> str:
    r"""
    Implements an algorithm to compute the Zarr checksum.

    Warning: This checksum is sensitive to Zarr configuration. 
        This checksum is not insensitive to change in the Zarr structure. For example, if you change the chunk size, 
        the checksum will also change.

    To understand how this works, consider the following directory structure:

               . (root)
              / \
             a   c
            /
           b
    
    Within zarr, this would for example be:

    - `root`: A Zarr Group with a single Array.
    - `a`: A Zarr Array
    - `b`: A single chunk of the Zarr Array
    - `c`: A metadata file (i.e. .zarray, .zattrs or .zgroup) 

    To compute the checksum, we first find all the trees in the node, in this case b and c. 
    We compute the hash of the content (the raw bytes) for each of these files.
    
    We then work our way up the tree. For any node (directory), we find all children of that node.
    In an sorted order, we then serialize a list with - for each of the children - the checksum, size, and number of children.
    The hash of the directory is then equal to the hash of the serialized JSON.

    The Polaris implementation is heavily based on the [`zarr-checksum` package](https://github.com/dandi/zarr_checksum).
    This method is the biggest deviation of the original code.
    """

    if fs is None:
        # Try guess the filesystem if it's not specified
        fs = dm.utils.fs.get_mapper(zarr_root_path).fs

    # Get the protocol of the path
    protocol = dm.utils.fs.get_protocol(zarr_root_path, fs)

    # For a local path, we extend the path to an absolute path
    # Otherwise, we assume the path is already absolute
    if protocol == "file":
        zarr_root_path = os.path.expandvars(zarr_root_path)
        zarr_root_path = os.path.expanduser(zarr_root_path)
        zarr_root_path = os.path.abspath(zarr_root_path)

    # Make sure the path exists and is a Zarr archive
    zarr.open_group(zarr_root_path, mode="r")

    # Generate the checksum
    tree = ZarrChecksumTree()

    # Find all files in the root
    leaves = fs.find(zarr_root_path, detail=True)

    for file in tqdm(leaves.values(), desc="Finding all files in the Zarr archive"):
        path = file["name"]

        relpath = path.removeprefix(zarr_root_path)
        relpath = relpath.lstrip("/")
        relpath = Path(relpath)

        size = file["size"]

        # Compute md5sum of file
        md5sum = hashlib.md5()
        with fs.open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5sum.update(chunk)
        digest = md5sum.hexdigest()

        # Yield file
        tree.add_leaf(
            path=relpath,
            size=size,
            digest=digest,
        )

    # Compute digest
    return tree.process().digest


# Pydantic models aren't used for performance reasons
@dataclass
class ZarrChecksumNode:
    """Represents the aggregation of zarr files at a specific path in the tree."""

    path: Path
    checksums: "ZarrChecksumManifest"

    def __lt__(self, other: "ZarrChecksumNode") -> bool:
        return str(self.path) < str(other.path)


class ZarrChecksumTree:
    """A tree that represents the checksummed files in a zarr."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, ZarrChecksumNode]] = []
        self._path_map: dict[Path, ZarrChecksumNode] = {}

    @property
    def empty(self) -> bool:
        return len(self._heap) == 0

    def _add_path(self, key: Path) -> None:
        node = ZarrChecksumNode(path=key, checksums=ZarrChecksumManifest())

        # Add link to node
        self._path_map[key] = node

        # Add node to heap with length (negated to representa max heap)
        length = len(key.parents)
        heapq.heappush(self._heap, (-1 * length, node))

    def _get_path(self, key: Path) -> ZarrChecksumNode:
        if key not in self._path_map:
            self._add_path(key)

        return self._path_map[key]

    def add_leaf(self, path: Path, size: int, digest: str) -> None:
        """Add a leaf file to the tree."""
        parent_node = self._get_path(path.parent)
        parent_node.checksums.files.append(ZarrChecksum(name=path.name, size=size, digest=digest))

    def add_node(self, path: Path, size: int, digest: str) -> None:
        """Add an internal node to the tree."""
        parent_node = self._get_path(path.parent)
        parent_node.checksums.directories.append(
            ZarrChecksum(
                name=path.name,
                size=size,
                digest=digest,
            )
        )

    def pop_deepest(self) -> ZarrChecksumNode:
        """Find the deepest node in the tree, and return it."""
        _, node = heapq.heappop(self._heap)
        del self._path_map[node.path]

        return node

    def process(self) -> "ZarrDirectoryDigest":
        """Process the tree, returning the resulting top level digest."""
        # Begin with empty root node, so if no files are present, the empty checksum is returned
        node = ZarrChecksumNode(path=Path("."), checksums=ZarrChecksumManifest())
        while not self.empty:
            # Pop the deepest directory available
            node = self.pop_deepest()

            # If we have reached the root node, then we're done.
            if node.path == Path(".") or node.path == Path("/"):
                break

            # Add the parent of this node to the tree
            directory_digest = node.checksums.generate_digest()
            self.add_node(
                path=node.path,
                size=directory_digest.size,
                digest=directory_digest.digest,
            )

        # Return digest
        return node.checksums.generate_digest()


@dataclass
class ZarrDirectoryDigest:
    """The data that can be serialized to / deserialized from a checksum string."""

    md5: str
    count: int
    size: int

    @classmethod
    def parse(cls, checksum: str | None) -> "ZarrDirectoryDigest":
        if checksum is None:
            return cls.parse(EMPTY_CHECKSUM)

        match = re.match(ZARR_DIGEST_PATTERN, checksum)
        if match is None:
            raise InvalidZarrChecksum()

        md5, count, size = match.groups()
        return cls(md5=md5, count=int(count), size=int(size))

    def __str__(self) -> str:
        return self.digest

    @property
    def digest(self) -> str:
        return f"{self.md5}-{self.count}--{self.size}"


@total_ordering
@dataclass
class ZarrChecksum:
    """
    A checksum for a single file/directory in a zarr file.

    Every file and directory in a zarr archive has a name, digest, and size.
    Leaf nodes are created by providing an md5 digest.
    Internal nodes (directories) have a digest field that is a zarr directory digest

    This class is serialized to JSON, and as such, key order should not be modified.
    """

    digest: str
    name: str
    size: int

    # To make this class sortable
    def __lt__(self, other: "ZarrChecksum") -> bool:
        return self.name < other.name


@dataclass
class ZarrChecksumManifest:
    """
    A set of file and directory checksums.

    This is the data hashed to calculate the checksum of a directory.
    """

    directories: list[ZarrChecksum] = field(default_factory=list)
    files: list[ZarrChecksum] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.files or self.directories)

    def generate_digest(self) -> ZarrDirectoryDigest:
        """Generate an aggregated digest for the provided files/directories."""
        # Ensure sorted first
        self.files.sort()
        self.directories.sort()

        # Aggregate total file count
        count = len(self.files) + sum(
            ZarrDirectoryDigest.parse(checksum.digest).count for checksum in self.directories
        )

        # Aggregate total size
        size = sum(file.size for file in self.files) + sum(directory.size for directory in self.directories)

        # Serialize json without any spacing
        json = dumps(asdict(self), separators=(",", ":"))

        # Generate digest
        md5 = hashlib.md5(json.encode("utf-8")).hexdigest()

        # Construct and return
        return ZarrDirectoryDigest(md5=md5, count=count, size=size)


# The "null" zarr checksum
EMPTY_CHECKSUM = ZarrChecksumManifest().generate_digest().digest
