from polaris._mixins import FormattingMixin


class InvalidDatasetError(ValueError):
    pass


class InvalidBenchmarkError(ValueError):
    pass


class InvalidResultError(ValueError):
    pass


class PolarisHubError(Exception, FormattingMixin):
    def __str__(self):
        response_message = self.__cause__.response.text
        error_message = f"The request to the Polaris Hub failed. See the error message below for more details:\n\n\n{response_message}\n\n"

        return error_message


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


class PolarisCreateArtifactError(PolarisHubError):
    SUGGEST_LOGIN_ON_CREATE = "Note: If you can confirm that you are authorized to perform this action, please call 'polaris login --overwrite' and try again. If this issue persists, please reach out to the Polaris team for support"

    def __init__(self, method: str):
        self.method = method

    def __str__(self):
        suggestion_message = self.format(self.SUGGEST_LOGIN_ON_CREATE, [self.BOLD, self.YELLOW])
        return f"{super().__str__()}\n{suggestion_message}."


class PolarisRetrieveArtifactError(PolarisHubError):
    SUGGEST_LOGIN_ON_RETRIEVE = "Note: If this artifact exists and you can confirm that you are authorized to retrieve it, please call 'polaris login --overwrite' and try again. If this issue persists, please reach out to the Polaris team for support"

    def __init__(self, method: str):
        self.method = method

    def __str__(self):
        suggestion_message = self.format(self.SUGGEST_LOGIN_ON_RETRIEVE, [self.BOLD, self.YELLOW])
        return f"{super().__str__()}\n{suggestion_message}."
