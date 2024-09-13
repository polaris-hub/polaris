# This script includes docking related evaluation metrics.

from typing import Union, List
from copy import deepcopy

import numpy as np
from rdkit.Chem.rdMolAlign import CalcRMS, GetBestRMS

import datamol as dm


## The following rmsd implementation are modified based on ppsebusters https://github.com/maabuu/posebusters/blob/main/posebusters/modules/rmsd.py


def _call_rdkit_rmsd(mol_probe: dm.Mol, mol_ref: dm.Mol, conf_id_probe: int, conf_id_ref: int, **params):
    try:
        return _rmsd(mol_probe, mol_ref, conf_id_probe, conf_id_ref, **params)
    except RuntimeError:
        pass
    except ValueError:
        pass

    return np.nan


def _rmsd(
    mol_probe: dm.Mol, mol_ref: dm.Mol, conf_id_probe: int, conf_id_ref: int, kabsch: bool = False, **params
):
    if kabsch is True:
        return GetBestRMS(prbMol=mol_probe, refMol=mol_ref, prbId=conf_id_probe, refId=conf_id_ref, **params)
    return CalcRMS(prbMol=mol_probe, refMol=mol_ref, prbId=conf_id_probe, refId=conf_id_ref, **params)


def robust_rmsd(
    mol_probe: dm.Mol,
    mol_ref: dm.Mol,
    conf_id_probe: int = -1,
    conf_id_ref: int = -1,
    drop_stereo: bool = False,
    heavy_only: bool = True,
    kabsch: bool = False,
    symmetrizeConjugatedTerminalGroups=True,
    **params,
) -> float:
    """RMSD calculation for isomers."""
    mol_probe = deepcopy(mol_probe)
    mol_ref = deepcopy(mol_ref)  # copy mols because rdkit RMSD calculation aligns mols

    if drop_stereo:
        mol_probe = dm.remove_stereochemistry(mol_probe)
        mol_ref = dm.remove_stereochemistry(mol_ref)

    if heavy_only:
        mol_probe = dm.remove_hs(mol_probe)
        mol_ref = dm.remove_hs(mol_ref)

    # combine parameters
    params = dict(
        symmetrizeConjugatedTerminalGroups=symmetrizeConjugatedTerminalGroups, kabsch=kabsch, **params
    )

    # calculate RMSD
    rmsd = _call_rdkit_rmsd(mol_probe, mol_ref, conf_id_probe, conf_id_ref, **params)
    if not np.isnan(rmsd):
        return rmsd

    # try again but remove charges and hydrogens
    mol_ref_uncharged = dm.to_neutral(dm.remove_hs(mol_ref))
    mol_probe_uncharged = dm.to_neutral(dm.remove_hs(mol_probe))
    rmsd = _call_rdkit_rmsd(mol_probe_uncharged, mol_ref_uncharged, conf_id_probe, conf_id_ref, **params)
    if not np.isnan(rmsd):
        return rmsd

    # try again but neutralize atoms
    mol_ref_neutralized = dm.to_neutral(mol_ref)
    mol_probe_neutralized = dm.to_neutral(mol_probe)
    rmsd = _call_rdkit_rmsd(mol_probe_neutralized, mol_ref_neutralized, conf_id_probe, conf_id_ref, **params)
    if not np.isnan(rmsd):
        return rmsd

    # try again but on canonical tautomers
    mol_ref_canonical = dm.canonical_tautomer(mol_ref)
    mol_probe_canonical = dm.canonical_tautomer(mol_probe)
    rmsd = _call_rdkit_rmsd(mol_probe_canonical, mol_ref_canonical, conf_id_probe, conf_id_ref, **params)
    if not np.isnan(rmsd):
        return rmsd

    # try again but after neutralizing atoms
    mol_ref_neutral_canonical = dm.canonical_tautomer(dm.to_neutral(mol_ref))
    mol_probe_neutral_canonical = dm.canonical_tautomer(dm.to_neutral(mol_probe))
    rmsd = _call_rdkit_rmsd(
        mol_probe_neutral_canonical, mol_ref_neutral_canonical, conf_id_probe, conf_id_ref, **params
    )
    if not np.isnan(rmsd):
        return rmsd

    return np.nan


def conformer_to_mol(mol, conf):
    """conv"""
    mol = dm.copy_mol(mol)
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def rmsd_coverage(
    y_pred: Union[str, List[dm.Mol]], y_true: Union[str, list[dm.Mol]], max_rsmd: float = 2, ignore_nan=False
):
    """
    Calculate the coverage of molecules with an RMSD less than 2 Å compared to the reference molecule conformer.

    Attributes:
        mols_probe: List of predicted binding conformers.
        mols_ref: List of ground truth binding confoermers.
        max_rsmd: The threshold for determining acceptable rsmd.
        ignore_nan: Ignore conformers that failed to compute RMSD.

    """

    if len(y_pred) != len(y_true):
        assert ValueError(
            f"The list of probing molecules and the list of reference molecules are different sizes. {len(y_pred)} != {len(y_true)} "
        )

    rmsds = np.array(
        [robust_rmsd(mol_probe=mol_probe, mol_ref=mol_ref) for mol_probe, mol_ref in zip(y_pred, y_true)]
    )

    if ignore_nan:
        rmsds = rmsds[~np.isnan(rmsds)]

    return np.nansum(rmsds <= 2) * 100 / len(rmsds)
