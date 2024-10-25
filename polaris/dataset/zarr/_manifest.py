import os
from hashlib import md5
from pathlib import Path

from pyarrow import Table, schema, string
from pyarrow.parquet import write_table

# PyArrow table schema for the V2 Zarr manifest file
ZARR_MANIFEST_SCHEMA = schema([("path", string()), ("md5_checksum", string())])

ROW_GROUP_SIZE = 128 * 1024 * 1024  # 128 MB


def generate_zarr_manifest(zarr_root_path: str, output_dir: str) -> str:
    """
    Entry point function which triggers the creation of a Zarr manifest for a V2 dataset.

    Parameters:
        zarr_root_path: The path to the root of a Zarr archive
        output_dir: The path to the directory which will hold the generated manifest
    """
    zarr_manifest_path = f"{output_dir}/zarr_manifest.parquet"

    entries = manifest_entries(zarr_root_path, zarr_root_path)
    manifest = Table.from_pylist(mapping=entries, schema=ZARR_MANIFEST_SCHEMA)
    write_table(manifest, zarr_manifest_path, row_group_size=ROW_GROUP_SIZE)

    return zarr_manifest_path


def manifest_entries(dir_path: str, root_path: str) -> list[dict[str, str]]:
    """
    Recursive function that traverses a directory, returning entries consisting of every file's path and MD5 hash

    Parameters:
        dir_path: The path to the current directory being traversed
        root_path: The root path from which to compute a relative path
    """
    entries = []
    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.is_file():
                entries.append(
                    {
                        "path": str(Path(entry.path).relative_to(root_path)),
                        "md5_checksum": calculate_file_md5(entry.path),
                    }
                )
            elif entry.is_dir():
                entries.extend(manifest_entries(entry.path, root_path))

    return entries


def calculate_file_md5(file_path: str) -> str:
    """Calculates the md5 hash for a file at a given path"""

    md5_hash = md5()
    with open(file_path, "rb") as file:
        #
        # Read the file in chunks to avoid using too much memory for large files
        for chunk in iter(lambda: file.read(4096), b""):
            md5_hash.update(chunk)

    # Return the hex representation of the digest
    return md5_hash.hexdigest()
