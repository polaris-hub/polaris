import pytest
from pydantic import ValidationError

import polaris as po
from polaris._artifact import BaseArtifactModel
from polaris.utils.types import HubOwner


def test_slug_string_type():
    """
    Verifies that the slug is validated correctly.
    Fails if:
    - Is too short (<4 characters)
    - Is too long (>64 characters)
    - Contains something other than lowercase letters, numbers, and hyphens.
    """
    for name in [
        "",
        "x",
        "xx",
        "xxx",
        "x" * 65,
        "invalid@",
        "invalid!",
        "InvalidName1",
        "invalid_name",
    ]:
        with pytest.raises(ValidationError):
            HubOwner(slug=name)

    for name in ["valid", "valid-name-1", "x" * 64, "x" * 4]:
        HubOwner(slug=name)


def test_slug_compatible_string_type():
    """Verifies that the artifact name is validated correctly."""

    # Fails if:
    # - Is too short (<4 characters)
    # - Is too long (>64 characters)
    # - Contains non-alphanumeric characters
    for name in ["", "x", "xx", "xxx", "x" * 65, "invalid@", "invalid!"]:
        with pytest.raises(ValidationError):
            BaseArtifactModel(name=name)

    # Does not fail
    for name in [
        "valid",
        "valid-name",
        "valid_name",
        "ValidName1",
        "Valid_",
        "Valid-",
        "x" * 64,
        "x" * 4,
    ]:
        BaseArtifactModel(name=name)


def test_version():
    with pytest.raises(ValidationError):
        BaseArtifactModel(polaris_version="invalid")
    assert BaseArtifactModel().polaris_version == po.__version__
    assert BaseArtifactModel(polaris_version="0.1.2")
