import abc
import re

from loguru import logger
from pydantic import BaseModel, PrivateAttr, computed_field

from polaris.utils.errors import PolarisChecksumError


class ChecksumMixin(BaseModel, abc.ABC):
    """
    Mixin class to add checksum functionality to a class.
    """

    _md5sum: str | None = PrivateAttr(None)

    @abc.abstractmethod
    def _compute_checksum(self) -> str:
        """Compute the checksum of the dataset."""
        raise NotImplementedError

    @computed_field
    @property
    def md5sum(self) -> str:
        """Lazily compute the checksum once needed."""
        if not self.has_md5sum:
            logger.info("Computing the checksum. This can be slow for large datasets.")
            self.md5sum = self._compute_checksum()
        return self._md5sum

    @md5sum.setter
    def md5sum(self, value: str):
        """Set the checksum."""
        if not re.fullmatch(r"^[a-f0-9]{32}$", value):
            raise ValueError("The checksum should be the 32-character hexdigest of a 128 bit MD5 hash.")
        self._md5sum = value

    @property
    def has_md5sum(self) -> bool:
        """Whether the md5sum for this class has been computed and stored."""
        return self._md5sum is not None

    def verify_checksum(self, md5sum: str | None = None):
        """
        Recomputes the checksum and verifies whether it matches the stored checksum.

        Warning: Slow operation
            This operation can be slow for large datasets.

        Info: Only works for locally stored datasets
            The checksum verification only works for datasets that are stored locally in its entirety.
            We don't have to verify the checksum for datasets stored on the Hub, as the Hub will do this on upload.
            And if you're streaming the data from the Hub, we will check the checksum of each chunk on download.
        """
        if md5sum is None:
            md5sum = self._md5sum
        if md5sum is None:
            logger.warning(
                "No checksum to verify against. Specify either the md5sum parameter or "
                "store the checksum in the dataset.md5sum attribute."
            )
            return

        # Recompute the checksum
        logger.info("To verify the checksum, we need to recompute it. This can be slow for large datasets.")
        self.md5sum = self._compute_checksum()

        if self.md5sum != md5sum:
            raise PolarisChecksumError(
                f"The specified checksum {md5sum} does not match the computed checksum {self.md5sum}"
            )
