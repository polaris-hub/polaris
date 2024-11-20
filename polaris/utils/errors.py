from httpx import Response


class InvalidDatasetError(ValueError):
    pass


class InvalidBenchmarkError(ValueError):
    pass


class InvalidCompetitionError(ValueError):
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


class InvalidZarrCodec(Exception):
    """Raised when an expected codec is not registered."""

    def __init__(self, codec_id: str):
        self.codec_id = codec_id
        super().__init__(
            f"This Zarr archive requires the {self.codec_id} codec. "
            "Install all optional codecs with 'pip install polaris-lib[codecs]'."
        )


class PolarisHubError(Exception):
    BOLD = "\033[1m"
    YELLOW = "\033[93m"
    _END_CODE = "\033[0m"

    def __init__(self, message: str = "", response: Response | None = None):
        prefix = "The request to the Polaris Hub failed."

        if response is not None:
            prefix += f" The Hub responded with:\n{response}"

        super().__init__("\n".join([prefix, message]))

    def format(self, text: str, codes: str | list[str]):
        if not isinstance(codes, list):
            codes = [codes]

        return "".join(codes) + text + self._END_CODE


class PolarisUnauthorizedError(PolarisHubError):
    def __init__(self, response: Response | None = None):
        message = (
            "You are not logged in to Polaris or your login has expired. "
            "You can use the Polaris CLI to easily authenticate yourself again with `polaris login --overwrite`."
        )
        super().__init__(message, response)


class PolarisCreateArtifactError(PolarisHubError):
    def __init__(self, response: Response | None = None):
        message = (
            "Note: If you can confirm that you are authorized to perform this action, "
            "please call 'polaris login --overwrite' and try again. If the issue persists, please reach out to the Polaris team for support."
        )
        super().__init__(message, response)


class PolarisRetrieveArtifactError(PolarisHubError):
    def __init__(self, response: Response | None = None):
        message = (
            "Note: If this artifact exists and you can confirm that you are authorized to retrieve it, "
            "please call 'polaris login --overwrite' and try again. If the issue persists, please reach out to the Polaris team for support."
        )
        super().__init__(message, response)
