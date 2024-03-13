import abc
from typing import Any

import datamol as dm
from pydantic import BaseModel


class Adapter(BaseModel, abc.ABC):
    """
    Adapters are callable, serializable objects that can be used to _adapt_ the
    datapoint in a dataset. This is for example
    """

    column: str

    def __call__(self, data: dict) -> dict:
        """Adapts the entire datapoint

        Used like:
        ```python
        adapter = Adapter(column="my_column")
        adapter({"my_column": datapoint})
        ```

        Args:
            data: The entire datapoint with column -> value pairs.
        """
        if self.column not in data:
            return data
        v = data[self.column]
        if isinstance(v, tuple):
            data[self.column] = [self.adapt(x) for x in v]
        else:
            data[self.column] = self.adapt(v)
        return data

    @abc.abstractmethod
    def adapt(self, data: Any) -> Any:
        """
        Adapt the value for a specific column.
        This method has to be overwritten by subclasses.

        Used like:
        ```python
        adapter = Adapter(column="my_column")
        adapter().adapt(datapoint["my_column"])
        ```

        Args:
            data: The value to adapt
        """
        raise NotImplementedError


class SmilesAdapter(Adapter):
    """
    Creates a RDKit `Mol` object from a SMILES string
    """

    def adapt(self, data: str) -> dm.Mol:
        return dm.to_mol(data)


class MolBytestringAdapter(Adapter):
    """
    Creates a RDKit `Mol` object from the RDKit-specific bytestring serialization
    """

    def adapt(self, data: bytes) -> dm.Mol:
        return dm.Mol(data)
