from hashlib import md5
import os
import pyarrow as pa
import pyarrow.parquet as pq

# PyArrow table schema for the V2 Zarr manifest file
ZARR_MANIFEST_SCHEMA = pa.schema([("path", pa.string()), ("checksum", pa.string())])


def generate_zarr_manifest(dir_path: str, writer: pq.ParquetWriter, zarr_root_path: str | None = None) -> str:
    """
    Recursive function that traverses a Zarr archive to build a V2 manifest file.

    Parameters:
        dir_path: The path to the current directory being processed in the archive
        writer: Writer object for incrementally adding rows to the manifest Parquet file
        zarr_root_path: The root path which triggered the first recursive call
    """

    # This simplifies the function interface on the initial call. `dir_path` will always be the
    # Zarr root path of the dataset.
    if not zarr_root_path:
        zarr_root_path = dir_path

    # Get iterator of items located in the directory at `dir_path`
    with os.scandir(dir_path) as it:
        #
        # Loop through directory items in iterator
        for entry in it:
            if entry.is_dir():
                #
                # If item is a directory, recurse into that directory
                generate_zarr_manifest(entry.path, writer, zarr_root_path)
            elif entry.is_file():
                #
                # If item is a file, calculate its relative path and chunk checksum. Then, append that
                # to the Zarr manifest parquet.
                table = pa.Table.from_pydict(
                    {
                        "path": [os.path.relpath(entry.path, zarr_root_path)],
                        "checksum": [calculate_file_md5(entry.path)],
                    },
                    schema=ZARR_MANIFEST_SCHEMA,
                )
                writer.write_table(table)


def calculate_file_md5(file_path: str):
    """Calculates the md5 hash for a file at a given path"""

    md5_hash = md5()
    with open(file_path, "rb") as file:
        #
        # Read the file in chunks to avoid using too much memory for large files
        for chunk in iter(lambda: file.read(4096), b""):
            md5_hash.update(chunk)

    # Return the hex representation of the digest
    return md5_hash.hexdigest()
