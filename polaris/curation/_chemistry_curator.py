# script for curating chemical molecules from either sdf or smiles

from typing import Optional, Union, List, Tuple, Iterable
import pandas as pd
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

import datamol as dm
from datamol.mol import Mol

UNIQUE_ID = "molhash_id"
NO_STEREO_UNIQUE_ID = "molhash_id_no_stereo"
SMILES_COL = "smiles"


def clean_mol(
    mol: Union[Mol, str],
    remove_salt_solvent: bool = True,
    remove_stereo: bool = False,
) -> Mol:
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
        return mol


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
    clean_mols = dm.parallelized(
        fn=lambda mol: clean_mol(mol, remove_stereo=ignore_stereo, remove_salt_solvent=remove_salt_solvent),
        inputs_list=mols,
        **parallelized_args,
    )
    clean_smiles = [dm.to_smiles(mol, canonical=True) for mol in clean_mols]

    # compute mol hash with all molecular info
    mol_ids = dm.parallelized(fn=dm.hash_mol, inputs_list=clean_mols, **parallelized_args)
    # compute mol ignoring stereo info layer
    mol_ids_no_stereo = dm.parallelized(
        fn=lambda mol: dm.hash_mol(mol, hash_scheme="no_stereo"), inputs_list=clean_mols, **parallelized_args
    )

    return pd.DataFrame(
        {
            SMILES_COL: clean_smiles,
            UNIQUE_ID: mol_ids,
            NO_STEREO_UNIQUE_ID: mol_ids_no_stereo,
        }
    )
