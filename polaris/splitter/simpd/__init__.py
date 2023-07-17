# Check pymoo is available
try:
    import pymoo
except ImportError:
    raise ImportError(
        "To use the SIMPD splitter you need to install `pymoo`. Please install it with `micromamba install pymoo`."
    )


# Disable pymoo compile hints
from pymoo.config import Config

Config.show_compile_hint = False


from .simpd import run_SIMPD
from .splitter import SIMPDSplitter
from .descriptors import DEFAULT_SIMPD_DESCRIPTORS
