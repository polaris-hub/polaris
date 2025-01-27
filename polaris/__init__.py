import logging

from rich.logging import RichHandler

from ._version import __version__
from .loader import load_benchmark, load_competition, load_dataset, load_model

__all__ = ["load_dataset", "load_benchmark", "load_competition", "load_model", "__version__"]

# Polaris specific logger
logger = logging.getLogger(__name__)

# Only add handler if the logger has not already been configured externally
if not logger.handlers:
    handler = RichHandler(rich_tracebacks=True)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%Y-%m-%d %X]"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
