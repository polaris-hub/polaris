# This script includes docking related evaluation metrics.

from typing import Union, List

import numpy as np
from rdkit.Chem.rdMolAlign import CalcRMS
import datamol as dm


## The following rmsd implementation are modified based on ppsebusters https://github.com/maabuu/posebusters/blob/main/posebusters/modules/rmsd.py


def _rmsd(mol_probe: dm.Mol, mol_ref: dm.Mol) -> float:
    """Calculate RMSD between predicted molecule and closest ground truth molecule.
       The RMSD is calculated with first conformer of predicted molecule and only consider heavy atoms for RMSD calculation

    Args:
        mol_probe: Predicted molecule (docked ligand) with exactly one conformer.
        mol_ref: Ground truth molecule (crystal ligand) with at least one conformer. If multiple conformers are
            present, the lowest RMSD will be reported.

    Returns:
        PoseBusters results dictionary.
    """

    # copy the molecule for modification.
    mol_probe = dm.copy_mol(mol_probe)
    mol_ref = dm.copy_mol(mol_ref)

    # remove hydrogen from molecule
    mol_probe = dm.remove_hs(mol_probe)
    mol_ref = dm.remove_hs(mol_ref)

    # calculate RMSD
    try:
        return CalcRMS(
            prbMol=mol_probe, refMol=mol_ref, symmetrizeConjugatedTerminalGroups=True, prbId=-1, refId=-1
        )

    # This can happen if ...
    except RuntimeError:
        pass
    # This can happen if ...
    except ValueError:
        pass

    return np.nan


def rmsd_coverage(y_pred: Union[str, List[dm.Mol]], y_true: Union[str, list[dm.Mol]], max_rsmd: float = 2):
    """
    Calculate the coverage of molecules with an RMSD less than 2 Å compared to the reference molecule conformer.

    Attributes:
        mols_probe: List of predicted binding conformers.
        mols_ref: List of ground truth binding confoermers.
        max_rsmd: The threshold for determining acceptable rsmd.
    """

    if len(y_pred) != len(y_true):
        assert ValueError(
            f"The list of probing molecules and the list of reference molecules are different sizes. {len(y_pred)} != {len(y_true)} "
        )

    rmsds = np.array(
        [_rmsd(mol_probe=mol_probe, mol_ref=mol_ref) for mol_probe, mol_ref in zip(y_pred, y_true)]
    )

    return np.nansum(rmsds <= max_rsmd) / len(rmsds)
