from polaris._artifact import BaseArtifactModel
from polaris.utils.types import HttpUrlString
from polaris.utils.types import AccessType, HubOwner


class Model(BaseArtifactModel):
    _artifact_type = "model"

    readme: str = ""
    code_url: HttpUrlString | None = None
    report_url: HttpUrlString | None = None

    def upload_to_hub(self, access: AccessType = "private", owner: HubOwner | str | None = None):
        """
        Uploads the model to the Polaris Hub.
        """
        from polaris.hub.client import PolarisHubClient

        with PolarisHubClient() as client:
            client.upload_model(self, owner=owner, access=access)
