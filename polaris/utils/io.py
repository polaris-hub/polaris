import os.path
import uuid
from typing import Optional

import filelock
import fsspec
from loguru import logger
from tenacity import Retrying
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_fixed

from polaris.utils import fs
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.errors import PolarisChecksumError


def create_filelock(lock_name: str, cache_dir_path: str = DEFAULT_CACHE_DIR):
    """Create an empty lock file into `cache_dir_path/locks/lock_name`"""
    lock_path = fs.join(cache_dir_path, "_lock_files", lock_name)
    with fsspec.open(lock_path, "w", auto_mkdir=True):
        pass
    return filelock.FileLock(lock_path)


def robust_copy(
    source_path: str,
    destination_path: str,
    md5sum: Optional[str] = None,
    max_retries: int = 5,
    wait_after_try: int = 2,
    progress: bool = True,
    leave_progress: bool = True,
    chunk_size: int = 2048,
):
    if not fs.is_file(source_path) and get_zarr_root(source_path) is None:
        raise ValueError(f"{source_path} is a directory and not part of a .zarr hierarchy!")

    if md5sum is None and fs.is_file(source_path):
        # NOTE (cwognum): This effectively means we will not check the checksum of .zarr files.
        #  The reason being that I'm not sure how to effectively compute a checksum for a .zarr
        md5sum = fs.hash_file(source_path)

    artifact_cache_lock = create_filelock(f"artifact_version_{md5sum or uuid.uuid4()}.lock")

    def log_failure(retry_state):
        logger.warning(
            f"""Downloading the artifact from {source_path} to {destination_path} failed. """
            f"""Retrying attempt {retry_state.attempt_number}/{max_retries} """
            f"""after a sleeping period of {wait_after_try} seconds."""
        )

    # This context manager will lock any process that try to download the same file. Only one process
    # will be able to download the artifact and all the other ones will be waiting at that line.
    # Once the lock is released the other processes will call `download_with_checksum` but the download will
    # not happen since the artifact file will already exist and its checksum will be correct.
    with artifact_cache_lock:
        # This loop will retry downloading the artifact for multiple attempts. Downloading an artifact
        # might fail for multiple reasons such as disk IO failures or network failures. The checksum logic
        # and the retry mechanism together allow to be resilient in case of intermitent failures.
        for attempt in Retrying(
            reraise=True,
            stop=stop_after_attempt(max_retries),
            after=log_failure,
            wait=wait_fixed(wait_after_try),
        ):
            with attempt:
                # The checksum logic will only validate an artifact download if its checksum matches
                # the excepted one. If not then it will be deleted and the download will happen again
                # until it succeeds (or until the number of attemps have been reached).
                download_with_checksum(
                    source_path=source_path,
                    destination_path=destination_path,
                    md5sum=md5sum,
                    progress=progress,
                    leave_progress=leave_progress,
                    chunk_size=chunk_size,
                )

    return destination_path


def download_with_checksum(
    source_path: str,
    destination_path: str,
    md5sum: Optional[str],
    progress: bool = False,
    leave_progress: bool = True,
    chunk_size: int = 2048,
):
    """Download an artifact from the bucket to a cache path while checking for its md5sum given a true md5sum.

    Args:
        source_path: The path to the artifact in the bucket.
        destination_path: The path of the artifact in the local cache.
        md5sum: The true md5sum to check against. If None, no checksum is performed but a warning is logged.
        progress: whether to display a progress bar.
        leave_progress: whether to hide the progress bar once the copy is done.
        chunk_size: the chunk size for the download.
    """

    # Download the artifact if not already in the cache.
    if not fs.exists(destination_path):
        if fs.is_dir(source_path):
            fs.copy_dir(
                source_path,
                destination_path,
                progress=progress,
                leave_progress=leave_progress,
                chunk_size=chunk_size,
            )

        else:
            fs.copy_file(
                source_path,
                destination_path,
                progress=progress,
                leave_progress=leave_progress,
                chunk_size=chunk_size,
            )

    # Check the cached artifact has the correct md5sum
    if md5sum is not None:
        cache_md5sum = fs.hash_file(destination_path)
        if cache_md5sum != md5sum:
            file_system = fs.get_mapper(destination_path).fs
            file_system.delete(destination_path)

            raise PolarisChecksumError(
                f"""The destination artifact at {destination_path} has a different md5sum ({cache_md5sum})"""
                f"""than the expected artifact md5sum ({md5sum}). The destination artifact has been deleted. """
            )


def get_zarr_root(path):
    """
    Recursive function to find the root of a .zarr file.
    Finds the highest level directory that has the .zarr extension.
    """
    if os.path.dirname(path) == path:
        # We reached the root of the filesystem
        return
    root = get_zarr_root(os.path.dirname(path))
    if root is None and fs.get_extension(path) == "zarr":
        root = path
    return root
