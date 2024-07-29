from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("polaris-lib")
except PackageNotFoundError:
    # package is not installed
    __version__ = "dev"
