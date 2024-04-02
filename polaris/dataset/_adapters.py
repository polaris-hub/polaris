from enum import Enum, auto, unique
import datamol as dm

# Map of conversion operations which can be applied to dataset columns
conversion_map = {"SMILES_TO_MOL": dm.to_mol, "BYTES_TO_MOL": dm.Mol}


@unique
class Adapter(Enum):
    """
    Adapters are predefined callables that change the format of the data.
    Adapters are serializable and can thus be saved alongside datasets.

    Attributes:
        SMILES_TO_MOL: Convert a SMILES string to a RDKit molecule.
        BYTES_TO_MOL: Convert a RDKit binary string to a RDKit molecule.
    """

    SMILES_TO_MOL = auto()
    BYTES_TO_MOL = auto()

    def __call__(self, data):
        if isinstance(data, tuple):
            return tuple(conversion_map[self.name](d) for d in data)
        return conversion_map[self.name](data)
