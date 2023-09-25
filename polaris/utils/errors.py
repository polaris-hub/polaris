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
        "You are not logged in to the Polaris Hub. Please use the Polaris CLI to login. "
        "Use `polaris --help` for more information."
    )

    def __init__(self, message: str = DEFAULT_ERROR_MSG):
        super().__init__(message)


class TestAccessError(Exception):
    # Prevent pytest to collect this as a test
    __test__ = False

    pass
