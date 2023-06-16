import numpy as np
import datamol as dm
from numpy.random import RandomState
from typing import Union, Optional, List, Sequence
from sklearn.model_selection import BaseShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split  # noqa W0212
from sklearn.utils.validation import _num_samples  # noqa W0212


class MolecularMinMaxSplit(BaseShuffleSplit):
    """
    Uses the Min-Max Diversity picker from RDKit and Datamol to have a diverse set of molecules in the train set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        smiles: Optional[Sequence[str]] = None,
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
        self._smiles = smiles

    def _iter_indices(
        self,
        X: Union[Sequence[str], np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices"""

        requires_smiles = X is None or not all(isinstance(x, str) for x in X)
        if self._smiles is None and requires_smiles:
            raise ValueError(
                "If the input is not a list of SMILES, you need to provide the SMILES to the constructor."
            )

        smiles = self._smiles if requires_smiles else X

        n_samples = _num_samples(smiles)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        base_seed = self.random_state
        if base_seed is None:
            base_seed = 0

        mols: List[dm.Mol] = list(dm.utils.parallelized(dm.to_mol, smiles, n_jobs=1, progress=False))

        for i in range(self.n_splits):
            picked_samples, _ = dm.pick_diverse(mols=mols, npick=n_train, seed=base_seed + i)
            yield picked_samples, np.setdiff1d(np.arange(n_samples), picked_samples)
