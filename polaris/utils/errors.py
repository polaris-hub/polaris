import certifi


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

    def __init__(self, message: str = "", response_text: str = ""):
        parts = filter(
            bool,
            [
                f"{self.BOLD}The request to the Polaris Hub has failed.{self._END_CODE}",
                f"{self.YELLOW}{message}{self._END_CODE}" if message else "",
                f"----------------------\nError reported was:\n{response_text}" if response_text else "",
            ],
        )

        super().__init__("\n".join(parts))


class PolarisUnauthorizedError(PolarisHubError):
    def __init__(self, response_text: str = ""):
        message = (
            "You are not logged in to Polaris or your login has expired. "
            "You can use the Polaris CLI to easily authenticate yourself again with `polaris login --overwrite`."
        )
        super().__init__(message, response_text)


class PolarisCreateArtifactError(PolarisHubError):
    def __init__(self, response_text: str = ""):
        message = (
            "Note: If you can confirm that you are authorized to perform this action, "
            "please call 'polaris login --overwrite' and try again. If the issue persists, please reach out to the Polaris team for support."
        )
        super().__init__(message, response_text)


class PolarisRetrieveArtifactError(PolarisHubError):
    def __init__(self, response_text: str = ""):
        message = (
            "Note: If this artifact exists and you can confirm that you are authorized to retrieve it, "
            "please call 'polaris login --overwrite' and try again. If the issue persists, please reach out to the Polaris team for support."
        )
        super().__init__(message, response_text)


class PolarisSSLError(PolarisHubError):
    def __init__(self, response_text: str = ""):
        message = (
            "We could not verify the SSL certificate. "
            f"Please ensure the installed version ({certifi.__version__}) of the `certifi` package is the latest. "
            "If you require the usage of a custom CA bundle, you can set the POLARIS_CA_BUNDLE "
            "environment variable to the path of your CA bundle. For debugging, you can temporarily disable "
            "SSL verification by setting the POLARIS_CA_BUNDLE environment variable to `false`."
        )
        super().__init__(message, response_text)


class PolarisDeprecatedError(PolarisHubError):
    def __init__(self, feature: str, response_text: str = ""):
        message = (
            f"The '{feature}' feature has been deprecated and is no longer supported. "
            "Please contact the Polaris team for more information about alternative approaches."
        )
        super().__init__(message, response_text)
