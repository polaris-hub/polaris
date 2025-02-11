from polaris._artifact import BaseArtifactModel
from polaris.utils.types import HttpUrlString
from polaris.utils.types import AccessType, HubOwner


class Model(BaseArtifactModel):
    """
    Represents a Model artifact in the Polaris ecosystem.

    A Model artifact serves as a centralized representation of a method or model, encapsulating its metadata.
    It can be associated with multiple result artifacts but is immutable after upload, except for the README field.

    Examples:
        Basic API usage:
        ```python
        from polaris.model import Model

        # Create a new Model Card
        model = Model(
            name="MolGPS",
            description="Graph transformer foundation model for molecular modeling",
            code_url="https://github.com/datamol-io/graphium"
        )

        # Upload the model card to the Hub
        model.upload_to_hub(owner="recursion")
        ```

    Attributes:
        readme (str): A detailed README describing the model.
        code_url (HttpUrlString | None): Optional URL pointing to the model's code repository.
        report_url (HttpUrlString | None): Optional URL linking to a report or publication related to the model.

    Methods:
        upload_to_hub(access: AccessType = "private", owner: HubOwner | str | None = None):
            Uploads the model artifact to the Polaris Hub, associating it with a specified owner and access level.

    For additional metadata attributes, see the base class.
    """

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
