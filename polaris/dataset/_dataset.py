import dataclasses
import pandas as pd
from typing import Dict
from polaris.dataset import Modality


@dataclasses.dataclass
class DatasetInfo:
    """
    Stores additional information about the dataset.
    """

    name: str
    description: str
    source: str
    modalities: Dict[str, Modality]

    def to_yaml(self):
        data = dataclasses.asdict(self)
        data["modalities"] = {k: v.name for k, v in data["modalities"].items()}
        return data

    @classmethod
    def from_yaml(cls, data: dict):
        data["modalities"] = {k: Modality[v] for k, v in data["modalities"].items()}
        return cls(**data)


class Dataset:
    """
    A dataset class is a light wrapper around a pandas DataFrame.

    TODO:
        - How to support non-tabular datasets? Different class?
    """

    def __init__(self, table: pd.DataFrame, info: DatasetInfo):
        self.table = table
        self.info = info

        for c in table.columns:
            if c not in info.modalities:
                info.modalities[c] = Modality.UNKNOWN

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
