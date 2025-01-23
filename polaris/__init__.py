import os
import sys

from loguru import logger

from ._version import __version__
from .loader import load_benchmark, load_competition, load_dataset, load_model

__all__ = ["load_dataset", "load_benchmark", "load_competition", "load_model", "__version__"]

# Configure the default logging level
os.environ["LOGURU_LEVEL"] = os.environ.get("LOGURU_LEVEL", "INFO")
logger.remove()
logger.add(sys.stderr, level=os.environ["LOGURU_LEVEL"])
