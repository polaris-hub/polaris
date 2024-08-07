import zarr
import numpy as np
import pandas as pd
from pathlib import Path

## load from zarr  and export to PDB
import biotite.structure as struc
import fastpdb

import zarr
import numpy as np

DTYPE_DICT = {
    "chain_id": "U4",
    "res_id": int,
    "ins_code": "U1",
    "res_name": "U5",
    "hetero": bool,
    "atom_name": "U6",
    "element": "U2",
}

# convert atom arrays to dictionary
KEYS = [
    "X",
    "Y",
    "Z",
    "chain_id",
    "res_id",
    "ins_code",
    "res_name",
    "hetero",
    "atom_name",
    "element",
    "atom_id",
    "b_factor",
    "occupancy",
    "charge",
]


def pdb_to_dict(atom_array):
    pdb_dict = {k: [] for k in KEYS}
    for row in atom_array:
        pdb_dict["X"].append(row.coord[0])
        pdb_dict["Y"].append(row.coord[1])
        pdb_dict["Z"].append(row.coord[2])
        pdb_dict["chain_id"].append(row.chain_id)
        pdb_dict["res_id"].append(row.res_id)
        pdb_dict["ins_code"].append(row.ins_code)
        pdb_dict["res_name"].append(row.res_name)
        pdb_dict["hetero"].append(row.hetero)
        pdb_dict["atom_name"].append(row.atom_name)
        pdb_dict["element"].append(row.element)
        pdb_dict["atom_id"].append(row.atom_id)
        pdb_dict["b_factor"].append(row.b_factor)
        pdb_dict["occupancy"].append(row.occupancy)
        pdb_dict["charge"].append(row.charge)
    return pdb_dict


def create_zarr_from_pdb(pdb_file, zarr_file):
    # load structure
    in_file = fastpdb.PDBFile.read(pdb_file)
    pdb_name = Path(pdb_file).stem
    atom_array = in_file.get_structure(
        model=1, include_bonds=True, extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )

    # fastpbd to dict
    pdb_dict = pdb_to_dict(atom_array)

    # Create a Zarr store
    store = zarr.DirectoryStore(zarr_file)

    # Create a root group
    root = zarr.group(store=store)

    # Create group and add datasets
    group = root.create_group(pdb_name)
    for col_name, col_val in pdb_dict.items():
        col_val = np.array(col_val)
        # get the accepted dtype
        dtype = DTYPE_DICT.get(col_name, col_val.dtype)
        group.create_dataset(col_name, data=col_val, dtype=dtype)


# load nested zarr file to dictionary
def load_group_as_numpy_arrays(group):
    """Load group datasets to numpy arrays exluding `column_names`"""
    arrays = {}
    for key, item in group.items():
        if isinstance(item, zarr.core.Array):
            arrays[key] = item[:]
        elif isinstance(item, zarr.hierarchy.Group):
            arrays[key] = load_group_as_numpy_arrays(item)
    return arrays


def zarr_to_pdb(zarr_file, pdb_file, pdb_pointer):
    store = zarr.DirectoryStore(zarr_file)
    root = zarr.open_group(store=store, mode="r")  # 'r' mode for read-only
    group = root[pdb_pointer]
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

    # write the arrays to fastpdb file
    out_file = fastpdb.PDBFile()
    out_file.set_structure(struc.array(atom_array))
    out_file.write(pdb_file)


# Function to load a PDB file
def load_structure(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", file_path)
    return structure


# Function to compare two structures
def compare_structures(structure1, structure2):
    parser = PDBParser(QUIET=True)

    # Load structure from pdb files
    structure1 = parser.get_structure("", structure1)
    structure2 = parser.get_structure("", structure2)

    # Check if the number of atoms is the same
    atoms1 = list(structure1.get_atoms())
    atoms2 = list(structure2.get_atoms())

    if len(atoms1) != len(atoms2):
        return False

    # Check if atomic coordinates are the same
    for atom1, atom2 in zip(atoms1, atoms2):
        if not atom1 - atom2 < 1e-3:  # Use a small tolerance for numerical precision
            return False

    # Superimpose the structures and check the RMSD
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(structure2.get_atoms())

    # Print the RMSD
    print(f"RMSD: {sup.rms:.4f} Å")
    if sup.rms > 1e-3:  # Again, use a small tolerance
        return False

    return True
