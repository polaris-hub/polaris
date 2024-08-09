from enum import Enum, auto, unique

import datamol as dm
import biotite.structure as struc
from biotite.structure import Atom


def zarr_to_pdb(zarr_file, pdb_column:str="pdb", pdb_pointer="0"):
    """ Load zarr from """
    store = zarr.DirectoryStore(zarr_file)
    root = zarr.open_group(store=store, mode="r")  # 'r' mode for read-only
    group = root[pdb_column][pdb_pointer]
    atom_array = []
    atom_dict = load_group_as_numpy_arrays(group)

    # convert dictionary to array list of Atom object
    array_length = atom_dict["X"].shape[0]
    for ind in range(array_length):
        atom = Atom(
            coord=(atom_dict["X"][ind], atom_dict["Y"][ind], atom_dict["Z"][ind]),
            chain_id=atom_dict["chain_id"][ind],
            res_id=atom_dict["res_id"][ind],
            ins_code=atom_dict["ins_code"][ind],
            res_name=atom_dict["res_name"][ind],
            hetero=atom_dict["hetero"][ind],
            atom_name=atom_dict["atom_name"][ind],
            element=atom_dict["element"][ind],
            b_factor=atom_dict["b_factor"][ind],
            occupancy=atom_dict["occupancy"][ind],
            charge=atom_dict["charge"][ind],
        )
        atom_array.append(atom)
    return struc.array(atom_array)


# Map of conversion operations which can be applied to dataset columns
conversion_map = {"SMILES_TO_MOL": dm.to_mol, "BYTES_TO_MOL": dm.Mol, "PDB_TO_ARRAY":zarr_to_pdb}


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
    PDB_TO_ARRAY = auto()

    def __call__(self, data):
        if isinstance(data, tuple):
            return tuple(conversion_map[self.name](d) for d in data)
        return conversion_map[self.name](data)
