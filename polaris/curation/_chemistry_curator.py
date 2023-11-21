import functools
from typing import Any, List, Union, Tuple

import pandas as pd

from rdkit.Chem import FindMolChiralCenters

import datamol as dm
from datamol.mol import Mol
from datamol.isomers._enumerate import count_stereoisomers

UNIQUE_ID = "molhash_id"
NO_STEREO_UNIQUE_ID = "molhash_id_no_stereo"
SMILES_COL = "smiles"
STEREO_DEF = "stereo_defined"
NUM_UNDEF_STEREO_CENTER = "num_undefined_stereo_center"
NUM_DEF_STEREO_CENTER = "num_defined_stereo_center"
NUM_STEREO_CENTER = "num_stereo_center"
NUM_STEREOISOMERS = "num_stereoisomers"
NUM_UNDEF_STEREOISOMERS = "num_undefined_stereoisomers"
UNDEF_EZ = "undefined_E/Z"  # E/Z diastereomer
UNDEF_ED = "undefined_E_D"  # enantiomers and diastereomer


def _num_stereo_centers(mol: Mol) -> Tuple[int]:
    """Get the number of defined and undefined stereo centers of a given molecule
        by accessing the all and only defined stereo centers.
        It's to facilitate the analysis of the stereo isomers.
        None will be return if there is no stereo centers in the molecule.

     Args:
         mol: Molecule

    Returns:
        nun_defined_centers: Number of defined stereo centers.
        nun_undefined_centers: Number of undefined stereo centers.

    See Also:
        <rdkit.Chem.FindMolChiralCenters>

    """
    num_all_centers = len(FindMolChiralCenters(mol, force=True, includeUnassigned=True))
    num_defined_centers = len(FindMolChiralCenters(mol, force=True, includeUnassigned=False))
    if num_all_centers == 0:
        return 0, 0, 0
    nun_undefined_centers = num_all_centers - num_defined_centers
    return num_all_centers, num_defined_centers, nun_undefined_centers


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

        if remove_salt_solvent:
            # standardize here to ensure the success the substructure matching for
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

        # remove stereochemistry information
        if remove_stereo:
            mol = dm.remove_stereochemistry(mol)

        # standardize
        mol = dm.standardize_mol(
            mol=mol,
            disconnect_metals=False,
            reionize=True,
            normalize=True,
            uncharge=False,
            stereo=not remove_stereo,
        )

        # number of possible stereoisomers
        num_stereoisomers = count_stereoisomers(
            mol=mol, undefined_only=False, rationalise=True, clean_it=True
        )

        # number of undefined stereoisomers
        num_undefined_stereoisomers = count_stereoisomers(
            mol=mol, undefined_only=True, rationalise=True, clean_it=True
        )

        # number of stereocenters
        num_all_centers, num_defined_centers, num_undefined_centers = _num_stereo_centers(mol)

        mol_dict = {
            SMILES_COL: dm.to_smiles(mol, canonical=True),
            UNIQUE_ID: dm.hash_mol(mol),
            NO_STEREO_UNIQUE_ID: dm.hash_mol(mol, hash_scheme="no_stereo"),
            NUM_STEREO_CENTER: num_all_centers,
            NUM_UNDEF_STEREO_CENTER: num_undefined_centers,
            NUM_DEF_STEREO_CENTER: num_defined_centers,
            NUM_STEREOISOMERS: num_stereoisomers,
            NUM_UNDEF_STEREOISOMERS: num_undefined_stereoisomers,
            # None of the stereochemistry is defined in the molecule
            UNDEF_ED: num_defined_centers == 0 and num_all_centers > 0,
            # Undefined EZ stereochemistry which has no stereocenter.
            UNDEF_EZ: num_all_centers == 0 and num_undefined_stereoisomers > 0,
        }
        return mol_dict


def run_chemistry_curation(
    mols: List[Union[Mol, str]],
    ignore_stereo: bool = False,
    remove_salt_solvent: bool = True,
    progress: bool = False,
    **parallelized_args: Any,
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
        progress: Whether show progress of the parallelization process.
        parallelized_args: Additional parameters for <datamol.utils.parallelized> to control the parallelization process.

    Returns:
        A dataframe which contains the smiles, unique_id with and without stereo chemistry information of the molecules.

    See Also:
        <datamol.utils.parallelized>
    """
    _curate_fn = functools.partial(
        _curate_mol, remove_stereo=ignore_stereo, remove_salt_solvent=remove_salt_solvent
    )
    mol_list = dm.parallelized(
        fn=_curate_fn,
        inputs_list=mols,
        progress=progress,
        **parallelized_args,
    )

    mol_dataframe = pd.DataFrame(mol_list)
    return mol_dataframe
