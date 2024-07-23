from httpx import Response

from polaris._mixins import FormattingMixin


class InvalidDatasetError(ValueError):
    pass


class InvalidBenchmarkError(ValueError):
    pass


class InvalidResultError(ValueError):
    pass


class TestAccessError(Exception):
    # Prevent pytest to collect this as a test
    __test__ = False

    pass


class PolarisChecksumError(ValueError):
    pass


class InvalidZarrChecksum(Exception):
    pass


class PolarisHubError(Exception, FormattingMixin):
    def __init__(self, message: str, response: Response):
        prefix = f"The request to the Polaris Hub failed with status {response.status_code}."
        suffix = "If the issue persists, please reach out to the Polaris team for support."
        super().__init__("\n".join([prefix, message, suffix]))


class PolarisUnauthorizedError(PolarisHubError):
    DEFAULT_ERROR_MSG = (
        "You are not logged in to Polaris or your login has expired. "
        "You can use the Polaris CLI to easily authenticate yourself again, see `polaris login --help`."
    )

    def __init__(self, message: str = DEFAULT_ERROR_MSG):
        super().__init__(message)


class PolarisCreateArtifactError(PolarisHubError):
    def __init__(self, response: Response):
        message = (
            "Note: If you can confirm that you are authorized to perform this action, "
            "please call 'polaris login --overwrite' and try again. "
        )
        message = self.format(message, [self.BOLD, self.YELLOW])
        super().__init__(message, response)


class PolarisRetrieveArtifactError(PolarisHubError):
    def __init__(self, response: Response):
        message = (
            "Note: If this artifact exists and you can confirm that you are authorized to retrieve it, "
            "please call 'polaris login --overwrite' and try again."
        )
        message = self.format(message, [self.BOLD, self.YELLOW])
        super().__init__(message, response)
