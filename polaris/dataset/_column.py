import string
from typing import Optional
from pydantic import BaseModel, field_validator


class ColumnAnnotation(BaseModel):
    """
    The `ColumnAnnotation` class is used to annotate the columns of the [`Dataset`][polaris.dataset.Dataset] object.
    This mostly just stores meta-data and does not affect the logic. The exception is the `is_pointer` attribute.

    Attributes:
        is_pointer: Annotates whether a column is a pointer column. If so, it does not contain data,
            but rather contains references to blobs of data from which the data is loaded.
        modality: The data modality describes the data type and is used to categorize datasets on the hub.
            This is a string that can only contain alpha-numeric characters, - or _.
        protocol: The protocol describes how the data was generated.
        user_attributes: Any additional meta-data can be stored in the user attributes.
    """

    is_pointer: bool = False
    modality: Optional[str] = None
    protocol: Optional[str] = None
    user_attributes: dict = {}

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @field_validator("modality")
    def _validate_modality(cls, v):
        """
        Since this might be used on the hub as a URL, we want to avoid any special characters.
        """
        if v is not None:
            valid_characters = string.ascii_letters + string.digits + "_-"
            if not all(c in valid_characters for c in v):
                raise ValueError(f"`name` can only contain alpha-numeric characters, - or _, found {v}")
            v = v.lower()
        return v
