import abc
from polaris.dataset import Dataset


class FrameworkWrapper(abc.ABC):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
