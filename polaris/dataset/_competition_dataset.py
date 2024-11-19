from pydantic import model_validator
from typing_extensions import Self

from polaris.dataset._dataset import DatasetV1
from polaris.utils.errors import InvalidCompetitionError


class CompetitionDataset(DatasetV1):
    """Dataset subclass for Polaris competitions.

    In addition to the data model and logic of the base Dataset class,
    this class adds additional functionality which validates certain aspects
    of the training data for a given competition.
    """

    _artifact_type = "competitionDataset"

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        """We reject the instantiation of competition datasets which leverage Zarr for the time being"""

        if self.uses_zarr:
            raise InvalidCompetitionError("Pointer columns are not currently supported in competitions.")

        return self
