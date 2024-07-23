from httpx import Response

from polaris.mixins._format_text import FormattingMixin  # Imported with full path to avoid circular import


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
    def __init__(self, message: str = "", response: Response | None = None):
        prefix = "The request to the Polaris Hub failed."

        if response is not None:
            prefix += f" The Hub responded with:\n{response}"

        super().__init__("\n".join([prefix, message]))


class PolarisUnauthorizedError(PolarisHubError):
    def __init__(self, response: Response | None = None):
        message = (
            "You are not logged in to Polaris or your login has expired. "
            "You can use the Polaris CLI to easily authenticate yourself again with `polaris login --overwrite`."
        )
        message = self.format(message, [self.BOLD, self.YELLOW])
        super().__init__(message, response)


class PolarisCreateArtifactError(PolarisHubError):
    def __init__(self, response: Response | None = None):
        message = (
            "Note: If you can confirm that you are authorized to perform this action, "
            "please call 'polaris login --overwrite' and try again. If the issue persists, please reach out to the Polaris team for support."
        )
        message = self.format(message, [self.BOLD, self.YELLOW])
        super().__init__(message, response)


class PolarisRetrieveArtifactError(PolarisHubError):
    def __init__(self, response: Response | None = None):
        message = (
            "Note: If this artifact exists and you can confirm that you are authorized to retrieve it, "
            "please call 'polaris login --overwrite' and try again. If the issue persists, please reach out to the Polaris team for support."
        )
        message = self.format(message, [self.BOLD, self.YELLOW])
        super().__init__(message, response)
