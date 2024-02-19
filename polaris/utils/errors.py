class InvalidDatasetError(ValueError):
    pass


class InvalidBenchmarkError(ValueError):
    pass


class InvalidResultError(ValueError):
    pass


class PolarisChecksumError(ValueError):
    pass


class PolarisHubError(Exception):
    pass


class PolarisUnauthorizedError(PolarisHubError):
    DEFAULT_ERROR_MSG = (
        "You are not logged in to Polaris or your login has expired. "
        "You can use the Polaris CLI to easily authenticate yourself again, see `polaris login --help`."
    )

    def __init__(self, message: str = DEFAULT_ERROR_MSG):
        super().__init__(message)


class TestAccessError(Exception):
    # Prevent pytest to collect this as a test
    __test__ = False

    pass
