# script for curating chemical molecules from either sdf or smiles

from typing import Optional, Union, List, Tuple, Iterable
import pandas as pd
from rdkit.Chem import FindMolChiralCenters

import datamol as dm
from datamol.mol import Mol

UNIQUE_ID = "molhash_id"
NO_STEREO_UNIQUE_ID = "molhash_id_no_stereo"
SMILES_COL = "smiles"
STEREO_DEF = "stereo_defined"
NUM_STEREO_CENTER = "num_stereo_center"
NUM_DEF_STEREO_CENTER = "num_defined_stereo_center"


def _num_stereo_centers(mol: Mol, only_defined=False):
    stereo_centers = FindMolChiralCenters(mol, force=True, includeUnassigned=not only_defined)
    return len(stereo_centers)


def _curate_mol(
    mol: Union[Mol, str],
    remove_salt_solvent: bool = True,
    remove_stereo: bool = False,
) -> dict:
    """Clean and standardize molecule to ensure the quality molecular structures from various resources.
       It comes with the option of remove salts/solvents and stereochemistry information from the molecule.

    Args:
        mol: Molecule
        remove_salt_solvent: When set to 'True', all disconnected salts and solvents
                             will be removed from molecule. In most of the cases,
                             the salts/solvents are recommended to be removed.
        remove_stereo: Whether remove stereochemistry information from molecule.
                       If it's known that the stereochemistry do not contribute to the bioactivity of interest,
                       the stereochemistry information can be removed.

    Returns:
        mol_dict: Dictionary of curated molecule with unique ids.
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
        if remove_salt_solvent:
            mol = dm.remove_salts_solvents(mol)

        # remove stereo
        if remove_stereo:
            mol = dm.remove_stereochemistry(mol)

        # standardize again
        if remove_salt_solvent or remove_stereo:
            mol = dm.standardize_mol(
                mol=mol,
                disconnect_metals=False,
                reionize=True,
                normalize=True,
                uncharge=False,
                stereo=not remove_stereo,
            )

        mol_dict = {
            SMILES_COL: dm.to_smiles(mol, canonical=True),
            UNIQUE_ID: dm.hash_mol(mol),
            NO_STEREO_UNIQUE_ID: dm.hash_mol(mol, hash_scheme="no_stereo"),
            NUM_STEREO_CENTER: _num_stereo_centers(mol),
            NUM_DEF_STEREO_CENTER: _num_stereo_centers(mol, only_defined=True),
        }
        return mol_dict


def run_chemistry_curation(
    mols: List[Union[Mol, str]],
    ignore_stereo: bool = False,
    remove_salt_solvent: bool = True,
    **parallelized_args,
) -> pd.DataFrame:
    """Perform curation on the input molecules.

    Args:
        mols: List of molecules.
        ignore_stereo: If set to `True`, the stereochemisty information will be ignored and removed from molecule
                       during the curation procedure. If it's known that the stereochemistry do not contribute to
                       the bioactivity of interest, the stereochemistry information can be ignored.
                       It's recommended to always keep the stererochemistry information.
        remove_salt_solvent: When set to 'True', all disconnected salts and solvents
                             will be removed from molecule. In most of the cases,
                             the salts/solvents are recommended to be removed.
        parallelized_args: Additional parameters for <datamol.utils.parallelized> to control the parallelization process.

    Returns:
        A dataframe which contains the smiles, unique_id with and without stereo chemistry information of the molecules.

    See Also:
        <datamol.utils.parallelized>
    """
    mol_list = dm.parallelized(
        fn=lambda mol: _curate_mol(mol, remove_stereo=ignore_stereo, remove_salt_solvent=remove_salt_solvent),
        inputs_list=mols,
        **parallelized_args,
    )

    mol_dataframe = pd.DataFrame(mol_list)
    return mol_dataframe
