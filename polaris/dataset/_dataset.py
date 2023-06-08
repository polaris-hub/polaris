import dataclasses
import pandas as pd
from typing import Dict
from polaris.dataset import Modality


@dataclasses.dataclass
class DatasetInfo:
    """
    Stores additional information about the dataset.
    """

    # The public-facing name of the dataset
    name: str

    # A beginner-friendly description of the dataset
    description: str

    # The data source, e.g. a DOI, Github repo or URL
    source: str

    # Annotates each column with the modality of that column
    modalities: Dict[str, Modality]

    def serialize(self) -> dict:
        """Convert the object into a YAML-serializable dictionary."""
        data = dataclasses.asdict(self)
        data["modalities"] = {k: v.name for k, v in data["modalities"].items()}
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DatasetInfo":
        """Takes a dictionary and converts it back into a DatasetInfo object."""
        data["modalities"] = {k: Modality[v] for k, v in data["modalities"].items()}
        return cls(**data)


class Dataset:
    """
    A dataset class is a wrapper around a pandas DataFrame.

    The dataset contains all information to construct a ML-dataset, but is not ready yet.
    For example, if the dataset contains the Image modality, the image data is not yet loaded but the column
    rather contains a pointer to the image file.

    A Dataclass can be part of one or multiple Tasks.
    """

    def __init__(self, table: pd.DataFrame, info: DatasetInfo):
        self.table = table
        self.info = info

        for c in table.columns:
            if c not in info.modalities:
                info.modalities[c] = Modality.UNKNOWN

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        s = (
            f"Name: {self.info.name}\n"
            f"Description: {self.info.description}\n"
            f"Source: {self.info.source}\n"
        )
        self.table.loc[""] = [f"[{self.info.modalities[c]}]" for c in self.table.columns]
        s += repr(self.table)
        self.table.drop(self.table.tail(1).index, inplace=True)
        return s

    def size(self):
        return len(self), len(self.table.columns)
