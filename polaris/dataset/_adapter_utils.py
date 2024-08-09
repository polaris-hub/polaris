import zarr
import biotite.structure as struc
from biotite.structure import Atom


# load nested zarr file to dictionary
def load_group_as_numpy_arrays(group):
    """Load zarr group as dictionary of numpy arrays"""
    arrays = {}
    for key, item in group.items():
        if isinstance(item, zarr.core.Array):
            arrays[key] = item[:]
        elif isinstance(item, zarr.hierarchy.Group):
            arrays[key] = load_group_as_numpy_arrays(item)
    return arrays


def zarr_to_pdb(group: zarr.Group):
    """Load pdb in zarr as fastpdb AtomArray"""

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
