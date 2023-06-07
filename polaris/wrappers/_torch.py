from polaris.dataset import Dataset
from polaris.wrappers import FrameworkWrapper


class PyTorchWrapper(FrameworkWrapper):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
