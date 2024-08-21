from pydantic import model_validator
from polaris.dataset import Dataset
from polaris.utils.errors import InvalidCompetitionError

_CACHE_SUBDIR = "datasets"


class CompetitionDataset(Dataset):
    """Dataset subclass for Polaris competitions.

    In addition to the data model and logic of the base Dataset class,
    this class adds additional functionality which validates certain aspects
    of the training data for a given competition.
    """

    @model_validator(mode="after")
    def _validate_model(cls, m: "CompetitionDataset"):
        """We reject the instantiation of competition datasets which leverage Zarr for the time being"""

        if m.uses_zarr:
            raise InvalidCompetitionError("Pointer columns are not currently supported in competitions.")
