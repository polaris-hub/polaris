from polaris.dataset import Dataset
from polaris.wrappers import PyTorchWrapper


class PyGWrapper(PyTorchWrapper):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
