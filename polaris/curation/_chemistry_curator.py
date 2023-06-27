# script for curating chemical molecules from either sdf or smiles

from typing import Optional, Union, List, Tuple, Iterable
import pandas as pd
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

import datamol as dm
from datamol.mol import Mol

UNIQUE_ID = "molhash_id"
NO_STEREO_UNIQUE_ID = "molhash_id_no_stereo"
SMILES_COL = "smiles"


def clean_mol(mol: Union[Mol, str], remove_stereo: bool = False) -> Mol:
    """Perform chemistry curation on one molecule

    Args:
        mol: Molecule
        remove_stereo: Whether remove stereochemistry information
    """
    with dm.without_rdkit_log():
        mol = dm.to_mol(mol)
        # fix mol
        mol = dm.fix_mol(mol)
        # sanitize molecule
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
        # standardize
        mol = dm.standardize_mol(
            mol=mol,
            disconnect_metals=False,
            reionize=True,
            normalize=True,
            uncharge=False,
            stereo=not remove_stereo,
        )
        # remove salts
        mol = dm.remove_salts_solvents(mol)

        # remove stereo
        if remove_stereo:
            mol = dm.remove_stereochemistry(mol)

        # standardize again
        mol = dm.standardize_mol(
            mol=mol,
            disconnect_metals=False,
            reionize=True,
            normalize=True,
            uncharge=False,
            stereo=not remove_stereo,
        )
        return mol


def _num_stereo_isomers(mol: Mol) -> int:
    """Get the number of stereo isomers. 1 means no stereo center exist in the molecule."""
    try:
        isomers = dm.enumerate_stereoisomers(mol)
        return len(isomers)
    except Exception:
        return None


def run_chemistry_curation(mols: List[Union[Mol, str]], ignore_stereo: bool = False) -> pd.DataFrame:
    """Perform curation on the input molecules.

    Args:
        mols: List of molecules.
        ignore_stereo: If set to `True`, the stereochemisty information will be ignored and removed from molecule
            during the curation procedure.

    Returns:
        A dataframe which contains the smiles, unique_id with and without stereo chemistry information of the molecules.

    """
    clean_mols = dm.parallelized(fn=lambda mol: clean_mol(mol, remove_stereo=ignore_stereo), inputs_list=mols)
    clean_smiles = [dm.to_smiles(mol, canonical=True) for mol in clean_mols]

    # compute mol hash with all molecular info
    mol_ids = dm.parallelized(fn=dm.hash_mol, inputs_list=clean_mols)
    # compute mol ignoring stereo info layer
    mol_ids_no_stereo = dm.parallelized(
        fn=lambda mol: dm.hash_mol(mol, hash_scheme="no_stereo"), inputs_list=clean_mols
    )

    return pd.DataFrame(
        {
            SMILES_COL: clean_smiles,
            UNIQUE_ID: mol_ids,
            NO_STEREO_UNIQUE_ID: mol_ids_no_stereo,
        }
    )
