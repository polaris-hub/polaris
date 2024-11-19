# This script includes docking related evaluation metrics.

from typing import Union, List

import numpy as np
from rdkit.Chem.rdMolAlign import CalcRMS

import datamol as dm


def _rmsd(mol_probe: dm.Mol, mol_ref: dm.Mol) -> float:
    """Calculate RMSD between predicted molecule and closest ground truth molecule.
       The RMSD is calculated with first conformer of predicted molecule and only consider heavy atoms for RMSD calculation.
       It is assumed that the predicted binding conformers are extracted from the docking output, where the receptor (protein) coordinates have been aligned with the original crystal structure.

    Args:
        mol_probe: Predicted molecule (docked ligand) with exactly one conformer.
        mol_ref: Ground truth molecule (crystal ligand) with at least one conformer. If multiple conformers are
            present, the lowest RMSD will be reported.

    Returns:
        Returns the RMS between two molecules, taking symmetry into account.
    """

    # copy the molecule for modification.
    mol_probe = dm.copy_mol(mol_probe)
    mol_ref = dm.copy_mol(mol_ref)

    # remove hydrogen from molecule
    mol_probe = dm.remove_hs(mol_probe)
    mol_ref = dm.remove_hs(mol_ref)

    # calculate RMSD
    return CalcRMS(
        prbMol=mol_probe, refMol=mol_ref, symmetrizeConjugatedTerminalGroups=True, prbId=-1, refId=-1
    )


def rmsd_coverage(y_pred: Union[str, List[dm.Mol]], y_true: Union[str, list[dm.Mol]], max_rsmd: float = 2):
    """
    Calculate the coverage of molecules with an RMSD less than a threshold (2 Å by default) compared to the reference molecule conformer.

    It is assumed that the predicted binding conformers are extracted from the docking output, where the receptor (protein) coordinates have been aligned with the original crystal structure.

    Attributes:
        y_pred: List of predicted binding conformers.
        y_true: List of ground truth binding confoermers.
        max_rsmd: The threshold for determining acceptable rsmd.
    """

    if len(y_pred) != len(y_true):
        assert ValueError(
            f"The list of probing molecules and the list of reference molecules are different sizes. {len(y_pred)} != {len(y_true)} "
        )

    rmsds = np.array(
        [_rmsd(mol_probe=mol_probe, mol_ref=mol_ref) for mol_probe, mol_ref in zip(y_pred, y_true)]
    )

    return np.sum(rmsds <= max_rsmd) / len(rmsds)
