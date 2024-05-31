import os
import warnings
import tempfile
import datamol as dm
import numpy as np
import polaris as po
from polaris.hub.client import PolarisHubClient
import pandas as pd
from polaris.dataset import ColumnAnnotation
from polaris.dataset import Dataset
from polaris.utils.types import HubOwner
from polaris.competition import CompetitionSpecification

# from polaris.competition import CompetitionSpecification

os.environ["POLARIS_HUB_URL"] = (
    "http://localhost:3000/"  # "https://polaris-hub-git-feat-competitions-invivoai-platform.vercel.app/"
)
os.environ["POLARIS_CALLBACK_URL"] = "http://localhost:3000/oauth2/callback"
os.environ["POLARIS_CLIENT_ID"] = "pci6tYe8jnvJ53ZB"

warnings.filterwarnings("ignore")

PATH = (
    "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv"
)
table = pd.read_csv(PATH)

# Additional meta-data on the column level
# Of course, for a real dataset we should annotate all columns.
annotations = {
    "LOG HLM_CLint (mL/min/kg)": ColumnAnnotation(
        description="Microsomal stability",
        user_attributes={"unit": "mL/min/kg"},
    ),
    "SMILES": ColumnAnnotation(description="Molecule SMILES string", modality="molecule"),
}

default_adapters = {"SMILES": "SMILES_TO_MOL"}

NAME = "competition_test_242"

dataset = Dataset(
    # The table is the core data-structure required to construct a dataset
    table=table,
    # Additional meta-data on the dataset level.
    name=NAME,
    description="120 prospective data sets, collected over 20 months across six ADME in vitro endpoints",
    source="https://doi.org/10.1021/acs.jcim.3c00160",
    annotations=annotations,
    tags=["DMPK", "ADME"],
    owner=HubOwner(slug="andrew"),
    license="CC-BY-4.0",
    user_attributes={"year": "2023"},
    default_adapters=default_adapters,
)
temp_dir = tempfile.TemporaryDirectory().name

save_dir = dm.fs.join(temp_dir, "dataset")
path = dataset.to_json(save_dir)
fs = dm.fs.get_mapper(save_dir).fs

dataset = po.load_dataset(path)

# For the sake of simplicity, we use a very simple, ordered split
split = (np.arange(3000).tolist(), (np.arange(521) + 3000).tolist())  # train  # test

competition = CompetitionSpecification(
    name=NAME,
    dataset=dataset,
    target_cols="LOG SOLUBILITY PH 6.8 (ug/mL)",
    input_cols="SMILES",
    split=split,
    metrics="mean_absolute_error",
)

# benchmark = BenchmarkSpecification(
#     name=NAME,
#     dataset=dataset,
#     target_cols="LOG SOLUBILITY PH 6.8 (ug/mL)",
#     input_cols="SMILES",
#     split=split,
#     metrics="mean_absolute_error"
# )

with PolarisHubClient() as client:
    res = client.upload_competition(competition=competition, owner=HubOwner(slug="andrew"))
    # res = client.upload_dataset(dataset=dataset)

    # res = client.get_competition("andrew", NAME)
    # res = client.list_competitions()

    # res = po.load_competition(f'andrew/{NAME}')

    # res = client.upload_dataset(dataset=dataset, owner=HubOwner(user_id="andrew", slug="andrew"))
    # res = client.upload_benchmark(benchmark=benchmark, owner=HubOwner(user_id="andrew", slug="andrew"))

    # res = po.load_benchmark('andrew/competition_test_63')

    # res = client.list_datasets(10)
    # res = client.list_benchmarks(10)
    # res = client.u
    #
    # res = po.load_competition(f'andrew/{NAME}')
    # res = client._initiate_split('andrew', 'competition_test_70')
    print(res)
