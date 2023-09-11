import pytest

from polaris._artifact import BaseArtifactModel
from polaris.utils.types import HubOwner, License


def test_slug_compatible_string_type():
    """Verifies that the artifact name is validated correctly."""

    # Fails if:
    # - Is too short (<4 characters)
    # - Is too long (>64 characters)
    # - Contains non alpha-numeric characters
    for name in ["", "x", "xx", "xxx", "x" * 65, "invalid@", "invalid!"]:
        with pytest.raises(ValueError):
            BaseArtifactModel(name=name)
        with pytest.raises(ValueError):
            HubOwner(userId=name, slug=name)

    # Does not fail
    for name in ["valid", "valid-name", "valid_name", "ValidName1", "Valid_", "Valid-", "x" * 64, "x" * 4]:
        BaseArtifactModel(name=name)
        HubOwner(userId=name, slug=name)


def test_artifact_owner():
    with pytest.raises(ValueError):
        # No owner specified
        HubOwner()
    with pytest.raises(ValueError):
        # Conflicting owner specified
        HubOwner(organizationId="org", userId="user", slug="test")

    # Valid - Only specifies one!
    assert HubOwner(organizationId="org", slug="org").owner == "org"
    assert HubOwner(userId="user", slug="user").owner == "user"


def test_license():
    for license in ["0BSD", "CC-BY-NC-4.0", "MIT"]:
        # If a valid SPDX license, the reference is automatically set
        assert License(id=license).reference is not None

    # If not a valid SPDX license, you must specify a valid reference
    with pytest.raises(ValueError):
        License(id="invalid")
    with pytest.raises(ValueError):
        License(id="invalid", reference="invalid")

    # If you specify a URL, we trust the user that this is a valid license
    License(id="invalid", reference="https://example.com")
