from httpx import Response

from polaris.utils.misc import format_text


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


class InvalidZarrChecksum(Exception):
    pass


class InformativePolarisHubError(PolarisHubError):
    # Suggestion messages
    SUGGEST_LOGIN_ON_CREATE = "Note: If you can confirm that you are authorized to perform this action, please call 'polaris login --overwrite' and try again. If this issue persists, please reach out to the Polaris team for support"
    SUGGEST_LOGIN_ON_RETRIEVE = "Note: If this artifact exists and you can confirm that you are authorized to retrieve it, please call 'polaris login --overwrite' and try again. If this issue persists, please reach out to the Polaris team for support"

    # Text styling codes
    BOLD_CODE = "\033[1m"
    YELLOW_CODE = "\033[93m"

    def __init__(self, response: Response, method: str):
        suggestion_message = (
            self.SUGGEST_LOGIN_ON_CREATE
            if method == "PUT" or method == "POST"
            else self.SUGGEST_LOGIN_ON_RETRIEVE
        )
        suggestion_message = format_text(suggestion_message, [self.BOLD_CODE, self.YELLOW_CODE])

        error_message = f"The request to the Polaris Hub failed. See the error message below for more details:\n{response}\n\n{suggestion_message}."
        super().__init__(error_message)
