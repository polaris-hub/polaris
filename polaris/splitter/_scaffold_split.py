from functools import partial

import datamol as dm
from typing import Optional
from ._predefined_group_split import PredefinedGroupShuffleSplit


def compute_murcko_scaffold(mol, make_generic: bool = False):
    """Computes the Bemis-Murcko scaffold of a compounds."""
    mol = dm.to_mol(mol)
    scaffold = dm.to_scaffold_murcko(mol, make_generic=make_generic)
    scaffold = dm.to_smiles(scaffold)
    return scaffold


class ScaffoldSplit(PredefinedGroupShuffleSplit):
    """The default scaffold split popular in molecular modeling literature"""

    def __init__(
        self,
        smiles,
        n_jobs: Optional[int] = None,
        n_splits=10,
        make_generic: bool = False,
        progress: bool = False,
        *,
        test_size=None,
        train_size=None,
        random_state=None,
    ):
        fn = partial(compute_murcko_scaffold, make_generic=make_generic)
        scaffolds = dm.utils.parallelized(fn, smiles, n_jobs=n_jobs, progress=progress)
        super().__init__(
            groups=scaffolds,
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
