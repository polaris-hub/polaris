import zarr
import numpy as np
import pandas as pd
from pathlib import Path

def create_zarr_from_pdb(pdb_file, zarr_file, mode="w", index=0):
    from biopandas.pdb import PandasPdb
    # load structure
    ppdb_df = PandasPdb().read_pdb(pdb_file)

    # Create a Zarr store
    store = zarr.DirectoryStore(zarr_file)

    if mode =='w':
        root = zarr.open_group(store=store, mode='a') 
    else:
        # Create a root group
        root = zarr.group(store=store)

    protein = root.create_group(index)

    dtype_dict = {'object': 'str'}

    for key in ['ATOM', 'HETATM', 'ANISOU', 'OTHERS']:

        # Create group and add datasets
        group = protein.create_group(key)
        
        # create a dataset  
        group.create_dataset("column_names", data=ppdb_df.df[key].columns.tolist(), dtype="str")

        for col_name, col_val in ppdb_df.df[key].items():
            dtype = col_val.values.dtype
            dtype = dtype_dict.get(str(dtype) ,dtype)
            group.create_dataset(col_name, data=col_val.values, dtype=dtype)



# Use Bio.PDB to check if two pdb are identifcal

from Bio.PDB import PDBParser, Superimposer

# Function to load a PDB file
def load_structure(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', file_path)
    return structure

# Function to compare two structures
def compare_structures(structure1, structure2):
    parser = PDBParser(QUIET=True)

    # Load structure from pdb files
    structure1 = parser.get_structure('', structure1)
    structure2 = parser.get_structure('', structure2)

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


