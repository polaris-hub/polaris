import datamol as dm
from typing import Optional, Union, Sequence

import numpy as np
from loguru import logger
from numpy.random import RandomState
from sklearn.model_selection import BaseShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split  # noqa W0212
from sklearn.utils.validation import _num_samples  # noqa W0212


class MolecularWeightSplit(BaseShuffleSplit):
    """
    Splits the dataset by sorting the molecules by their molecular weight
    and then finding an appropriate cutoff to split the molecules in two sets.
    """

    def __init__(
        self,
        generalize_to_larger: bool = False,
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
        self._generalize_to_larger = generalize_to_larger

    def _iter_indices(
        self,
        X: Union[Sequence[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[Union[int, np.ndarray]] = None,
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

        mols = dm.utils.parallelized(dm.to_mol, smiles, n_jobs=1, progress=False)
        mws = dm.utils.parallelized(dm.descriptors.mw, mols, n_jobs=1, progress=False)

        sorted_idx = np.argsort(mws)

        if self.n_splits > 1:
            logger.warning(
                f"n_splits={self.n_splits} > 1, but {self.__class__.__name__} is deterministic "
                f"and will always return the same split!"
            )

        for i in range(self.n_splits):
            if self._generalize_to_larger:
                yield sorted_idx[:n_train], sorted_idx[n_train:]
            else:
                yield sorted_idx[n_test:], sorted_idx[:n_test]
