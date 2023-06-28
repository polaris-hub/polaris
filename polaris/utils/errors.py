class InvalidDatasetError(ValueError):
    pass


class InvalidBenchmarkError(ValueError):
    pass


class InvalidResultError(ValueError):
    pass


class PolarisChecksumError(ValueError):
    pass


class PolarisUnauthorizedError(Exception):
    pass


class TestAccessError(Exception):
    pass
