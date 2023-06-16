import numpy as np
import datamol as dm
from functools import partial
from typing import Optional, Union

from numpy.random import RandomState
from sklearn.model_selection import GroupShuffleSplit


def get_scaffold(mol, make_generic: bool = False):
    """
    Computes the Bemis-Murcko scaffold of a compound.
    If make_generic is True, the scaffold is made generic by replacing all side chains with R groups.
    """
    mol = dm.to_mol(mol)
    scaffold = dm.to_scaffold_murcko(mol, make_generic=make_generic)
    scaffold = dm.to_smiles(scaffold)
    return scaffold


class ScaffoldSplit(GroupShuffleSplit):
    """The default scaffold split popular in molecular modeling literature"""

    def __init__(
        self,
        smiles,
        n_jobs: Optional[int] = None,
        n_splits=10,
        make_generic: bool = False,
        progress: bool = False,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )

        fn = partial(get_scaffold, make_generic=make_generic)
        self.scaffolds = dm.utils.parallelized(fn, smiles, n_jobs=n_jobs, progress=progress)

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices"""
        yield from super()._iter_indices(X, y, self.scaffolds)
