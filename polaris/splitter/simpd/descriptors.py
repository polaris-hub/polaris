import pandas as pd

from rdkit.Chem.Descriptors import fr_benzene  # type: ignore


def fr_benzene_1000_heavy_atoms_count(mol):
    return 1000 * fr_benzene(mol) / mol.GetNumHeavyAtoms()


# From https://github.com/rinikerlab/molecular_time_series/blob/55eb420ab0319fbb18cc00fe62a872ac568ad7f5/ga_lib_3.py#L323
DEFAULT_SIMPD_DESCRIPTORS = pd.DataFrame(
    [
        {
            "name": "SA_Score",
            "function": "datamol.descriptors.sas",
            "target_delta_value": 0.10 * 2.8,
        },
        {
            "name": "HeavyAtomCount",
            "function": "datamol.descriptors.n_heavy_atoms",
            "target_delta_value": 0.1 * 31,
        },
        {
            "name": "TPSA",
            "function": "datamol.descriptors.tpsa",
            "target_delta_value": 0.15 * 88.0,
        },
        {
            "name": "fr_benzene/1000 HeavyAtoms",
            "function": "polaris.splitter.simpd.descriptors.fr_benzene_1000_heavy_atoms_count",
            "target_delta_value": -0.2 * 0.44,
        },
    ]
)
