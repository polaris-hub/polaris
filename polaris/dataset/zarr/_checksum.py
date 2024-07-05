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
from typing import List, Tuple

import fsspec
import zarr
import zarr.errors
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from tqdm import tqdm

from polaris.utils.errors import InvalidZarrChecksum

ZARR_DIGEST_PATTERN = "([0-9a-f]{32})-([0-9]+)-([0-9]+)"


def compute_zarr_checksum(zarr_root_path: str) -> Tuple[str, List["ZarrFileChecksum"]]:
    r"""
    Implements an algorithm to compute the Zarr checksum.

    Warning: This checksum is sensitive to Zarr configuration. 
        This checksum is sensitive to change in the Zarr structure. For example, if you change the chunk size, 
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

    # Get the protocol of the path
    protocol = fsspec.utils.get_protocol(zarr_root_path)

    # We only support computing checksum for local datasets.
    # NOTE (cwognum): We don't have to verify the checksum for datasets stored on the Hub,
    # as the Hub will do this on upload. And if you're streaming the data from the Hub,
    # we will check the checksum of each chunk on download.
    if protocol != "file":
        raise RuntimeError(
            "You can only compute the checksum for a local Zarr archive. "
            "You can cache a dataset to your local machine with `dataset.cache()`."
        )

    # Normalize the path
    zarr_root_path = os.path.expandvars(zarr_root_path)
    zarr_root_path = os.path.expanduser(zarr_root_path)
    zarr_root_path = os.path.abspath(zarr_root_path)

    fs, zarr_root_path = fsspec.url_to_fs(zarr_root_path)

    # Make sure the path exists and is a Zarr archive
    zarr.open_group(zarr_root_path, mode="r")

    # Generate the checksum
    tree = _ZarrChecksumTree()

    # Find all files below the root
    leaves = fs.find(zarr_root_path, detail=True)
    zarr_md5sum_manifest = []

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

        # Add a leaf to the tree
        # (This actually adds the file's checksum to the parent directory's manifest)
        tree.add_leaf(
            path=relpath,
            size=size,
            digest=digest,
        )

        # We persist the checksums for leaf nodes separately,
        # because this is what the Hub needs to verify data integrity.
        zarr_md5sum_manifest.append(ZarrFileChecksum(path=str(relpath), md5sum=digest, size=size))

    # Compute digest
    return tree.process().digest, zarr_md5sum_manifest


class ZarrFileChecksum(BaseModel):
    """
    This data is sent to the Hub to verify the integrity of the Zarr archive on upload.

    Attributes:
        path: The path of the file relative to the Zarr root.
        md5sum: The md5sum of the file.
        size: The size of the file in bytes.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, arbitrary_types_allowed=True)

    path: str
    md5sum: str
    size: int


# ================================
# Overview of the data structures
# ================================

# NOTE (cwognum): I kept forgetting how this works, so I'm writing it down
# - The ZarrChecksumTree is a binary tree (heap queue). It determines the order in which to process the nodes.
# - The ZarrChecksumNode is a node in the ZarrChecksumTree queue. It represents a directory in the Zarr archive and
#   stores a manifest with all the data needed to compute the checksum for that node.
# - The ZarrChecksumManifest is a collection of checksums for all direct (non-recursive) children of a directory.
# - The ZarrChecksum is the data used to compute the checksum for a file or directory in a Zarr Archive.
#   This is the object that the ZarrChecksumManifest stores a collection of.
# - A ZarrDirectoryDigest is the result of processing a directory. Once completed,
#   it is added to the ZarrChecksumManifest of its parent as part of a ZarrChecksum.

# NOTE (cwognum): As a first impression, it seems there is some redundancy in the data structures.
#   My feeling is that we could reduce the redundancy to simplify things and improve maintainability.
#   However, for the time being, let's stick close to the original code.

# ================================


# Pydantic models aren't used for performance reasons
class _ZarrChecksumTree:
    """
    The ZarrChecksumTree is a tree structure that maintains the state of the checksum algorithm.

    Initialized with a set of leafs (i.e. files), the nodes in this tree correspond to all directories
    that are above those leafs and below the Zarr Root.

    The tree then implements the logic for retrieving the next node (i.e. directory) to process,
    and for computing the checksum for that node based on its children.
    Once it reaches the root, it has computed the checksum for the entire Zarr archive.
    """

    def __init__(self) -> None:
        # Queue to prioritize the next node to process
        self._heap: list[tuple[int, _ZarrChecksumNode]] = []

        # Map of (relative) paths to nodes.
        self._path_map: dict[Path, _ZarrChecksumNode] = {}

    @property
    def empty(self) -> bool:
        """Check if the tree is empty."""
        # This is used as an exit condition in the process() method
        return len(self._heap) == 0

    def _add_path(self, key: Path) -> None:
        """Adds a new entry to the heap queue for which we need to compute the checksum."""

        # Create a new node
        # A node represents a file or directory.
        # A node refers to a node in the heap queue (i.e. binary tree)
        # The structure of the heap is thus _not_ the same as the structure of the file system!
        node = _ZarrChecksumNode(path=key, checksums=_ZarrChecksumManifest())
        self._path_map[key] = node

        # Add node to heap with length (negated to represent a max heap)
        # We use the length of the parents (relative to the Zarr root) to structure the heap.
        # The node with the longest path is the deepest node in the tree.
        # This node will be prioritized for processing next.
        length = len(key.parents)
        heapq.heappush(self._heap, (-1 * length, node))

    def _get_path(self, key: Path) -> "_ZarrChecksumNode":
        """
        If an entry for this path already exists, return it.
        Otherwise create a new one and return that.
        """
        if key not in self._path_map:
            self._add_path(key)
        return self._path_map[key]

    def add_leaf(self, path: Path, size: int, digest: str) -> None:
        """Add a leaf file to the tree."""
        parent_node = self._get_path(path.parent)
        parent_node.checksums.files.append(_ZarrChecksum(name=path.name, size=size, digest=digest))

    def add_node(self, path: Path, size: int, digest: str, count: int) -> None:
        """Add an internal node to the tree."""
        parent_node = self._get_path(path.parent)
        parent_node.checksums.directories.append(
            _ZarrChecksum(
                name=path.name,
                size=size,
                digest=digest,
                count=count,
            )
        )

    def pop_deepest(self) -> "_ZarrChecksumNode":
        """
        Returns the node with the highest priority for processing next.

        Returns (one of the) node(s) with the most parent directories
        (i.e. the deepest directory in the file system)
        """
        _, node = heapq.heappop(self._heap)
        del self._path_map[node.path]
        return node

    def process(self) -> "_ZarrDirectoryDigest":
        """Process the tree, returning the resulting top level digest."""

        # Begin with empty root node, so that if no files are present, the empty checksum is returned
        node = _ZarrChecksumNode(path=Path("."), checksums=_ZarrChecksumManifest())

        while not self.empty:
            # Get the next directory to process
            # Priority is based on the number of parents a directory has
            # In other word, the depth of the directory in the file system.
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
                count=directory_digest.count,
            )

        # Return digest
        return node.checksums.generate_digest()


@dataclass
class _ZarrChecksumNode:
    """
    A node in the ZarrChecksumTree.

    This node represents a file or directory in the Zarr archive,
    but "node" here refers to a node in the heap queue (i.e. binary tree).
    The structure of the heap is thus _not_ the same as the structure of the file system!

    The node stores a manifest of checksums for all files and directories below it.
    """

    path: Path
    checksums: "_ZarrChecksumManifest"

    def __lt__(self, other: "_ZarrChecksumNode") -> bool:
        return str(self.path) < str(other.path)


@dataclass
class _ZarrChecksumManifest:
    """
    For a directory in the Zarr archive (i.e. a node in the heap queue),
    we maintain a manifest of the checksums for all files and directories
    below that directory.

    This data is then used to calculate the checksum of a directory.
    """

    directories: list["_ZarrChecksum"] = field(default_factory=list)
    files: list["_ZarrChecksum"] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.files or self.directories)

    def generate_digest(self) -> "_ZarrDirectoryDigest":
        """Generate an aggregated digest for the provided files/directories."""

        # Sort everything to ensure the checksum is deterministic
        self.files.sort()
        self.directories.sort()

        # Aggregate total file count
        count = len(self.files) + sum(checksum.count for checksum in self.directories)

        # Aggregate total size
        size = sum(file.size for file in self.files) + sum(directory.size for directory in self.directories)

        # Serialize json without any spacing
        json = dumps(asdict(self), separators=(",", ":"))

        # Generate digest
        md5 = hashlib.md5(json.encode("utf-8")).hexdigest()

        # Construct and return
        return _ZarrDirectoryDigest(md5=md5, count=count, size=size)


@total_ordering
@dataclass
class _ZarrChecksum:
    """
    The data used to compute the checksum for a file or directory in a Zarr Archive.

    This class is serialized to JSON, and as such, key order should not be modified.
    """

    digest: str
    name: str
    size: int
    count: int = 0

    # To make this class sortable
    def __lt__(self, other: "_ZarrChecksum") -> bool:
        return self.name < other.name


@dataclass
class _ZarrDirectoryDigest:
    """
    The digest for a directory in a Zarr Archive.

    The digest is a string representation that serves as a checksum for the directory.
    This is a utility class to (de)serialize that string.
    """

    md5: str
    count: int
    size: int

    @classmethod
    def parse(cls, checksum: str | None) -> "_ZarrDirectoryDigest":
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
        return f"{self.md5}-{self.count}-{self.size}"


# The "null" zarr checksum
EMPTY_CHECKSUM = _ZarrChecksumManifest().generate_digest().digest
