from hashlib import md5


def calculate_file_md5(file_path):
    """Calculates the md5 hash for a file at a given path"""

    md5_hash = md5()
    with open(file_path, "rb") as file:
        #
        # Read the file in chunks to avoid using too much memory for large files
        for chunk in iter(lambda: file.read(4096), b""):
            md5_hash.update(chunk)

    # Return the hex representation of the digest
    return md5_hash.hexdigest()
