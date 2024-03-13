import abc
from typing import Optional

import datamol as dm
from pydantic import BaseModel


class Adapter(BaseModel, abc.ABC):
    column: Optional[str] = None

    def __call__(self, data: dict) -> dict:
        v = data[self.column]
        if isinstance(v, tuple):
            data[self.column] = [self.adapt(x) for x in v]
        else:
            data[self.column] = self.adapt(v)

        return data

    @abc.abstractmethod
    def adapt(self, data: dict):
        raise NotImplementedError


class SmilesAdapter(Adapter):
    def adapt(self, data: str) -> dm.Mol:
        return dm.to_mol(data)


class MolBytestringAdapter(Adapter):
    def adapt(self, data: bytes) -> dm.Mol:
        return dm.Mol(data)
