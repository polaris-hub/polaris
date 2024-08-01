from typing import Optional

from pydantic import model_validator
from polaris.dataset import Dataset
from polaris.utils.constants import DEFAULT_CACHE_DIR
from polaris.utils.errors import InvalidDatasetError, PolarisChecksumError
from datamol.utils import fs
from polaris.dataset._column import ColumnAnnotation

_CACHE_SUBDIR = "datasets"


class CompetitionDataset(Dataset):
    masked_md5sum: Optional[str] = None

    @model_validator(mode="after")
    @classmethod
    def _validate_model(cls, m: "CompetitionDataset"):
        """If a checksum is provided, verify it matches what the checksum should be for the masked dataset.
        If no checksum is provided, make sure it is set.
        If no cache_dir is provided, set it to the default cache dir and make sure it exists
        """

        # Verify that all annotations are for columns that exist
        if any(k not in m.table.columns for k in m.annotations):
            raise InvalidDatasetError("There are annotations for columns that do not exist")

        # Verify that all adapters are for columns that exist
        if any(k not in m.table.columns for k in m.default_adapters.keys()):
            raise InvalidDatasetError("There are default adapters for columns that do not exist")

        # Set a default for missing annotations and convert strings to Modality
        for c in m.table.columns:
            if c not in m.annotations:
                m.annotations[c] = ColumnAnnotation()
            m.annotations[c].dtype = m.table[c].dtype

        # Verify the checksum
        actual = m.masked_md5sum
        expected = cls._compute_checksum(m.table)

        if actual is None:
            m.md5sum = expected
        elif actual != expected:
            raise PolarisChecksumError(
                "The dataset md5sum does not match what was specified in the meta-data. "
                f"{actual} != {expected}"
            )

        # Set the default cache dir if none and make sure it exists
        if m.cache_dir is None:
            m.cache_dir = fs.join(DEFAULT_CACHE_DIR, _CACHE_SUBDIR, m.name, m.md5sum)
        fs.mkdir(m.cache_dir, exist_ok=True)

        return m
