from ._version import __version__
from .loader import load_benchmark, load_dataset, load_competition

__all__ = ["load_dataset", "load_benchmark", "__version__", "load_competition"]
