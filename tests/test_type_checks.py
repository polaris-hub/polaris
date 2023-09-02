import pytest

from polaris._artifact import BaseArtifactModel
from polaris.utils.types import HubOwner


def test_slug_compatible_string_type():
    """Verifies that the artifact name is validated correctly."""

    # Fails if:
    # - Ends in a dash or underscore
    # - Is too short (<3 characters)
    # - Contains non alpha-numeric characters
    for name in ["invalid-", "invalid_", "", "x", "xx", "invalid@", "invalid!"]:
        with pytest.raises(ValueError):
            BaseArtifactModel(name=name)
        with pytest.raises(ValueError):
            HubOwner(userId=name)

    # Does not fail
    for name in ["valid", "valid-name", "valid_name", "ValidName1"]:
        BaseArtifactModel(name=name)
        HubOwner(userId=name)


def test_artifact_owner():
    with pytest.raises(ValueError):
        # No owner specified
        HubOwner()
    with pytest.raises(ValueError):
        # Conflicting owner specified
        HubOwner(organizationId="org", userId="user")

    # Valid - Only specifies one!
    assert HubOwner(organizationId="org").owner == "org"
    assert HubOwner(userId="user").owner == "user"
