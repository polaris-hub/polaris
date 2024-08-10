from typing import Optional

from pydantic import model_validator
from polaris.dataset import Dataset
from polaris.utils.errors import PolarisChecksumError

_CACHE_SUBDIR = "datasets"


class CompetitionDataset(Dataset):
    """Dataset subclass for Polaris competitions.

    In addition to the data model and logic of the base Dataset class,
    this class adds additional functionality which validates the integrity
    of the training data for a given competition.

    Attributes:
        masked_md5sum: The checksum for the competition's training dataset.
    """

    masked_md5sum: Optional[str] = None

    @model_validator(mode="after")
    def _validate_model(cls, m: "CompetitionDataset"):
        """If a checksum is provided, verify it matches what the checksum should be for the masked dataset.
        If no checksum is provided, make sure it is set.
        If no cache_dir is provided, set it to the default cache dir and make sure it exists
        """

        # Verify the checksum
        actual = m.masked_md5sum
        expected = cls._compute_checksum(m)

        if actual is None:
            m.masked_md5sum = expected
        elif actual != expected:
            raise PolarisChecksumError(
                "The dataset md5sum does not match what was specified in the meta-data. "
                f"{actual} != {expected}"
            )

        return m
