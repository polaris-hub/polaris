import enum
from typing import Optional, Union

from pydantic import BaseModel, validator


class Modality(enum.Enum):
    """
    Used to Annotate columns in a dataset.
    """

    UNKNOWN = "unknown"
    MOLECULE = "molecule"
    MOLECULE_3D = "molecule_3D"
    PROTEIN = "protein"
    PROTEIN_3D = "protein_3D"
    IMAGE = "image"

    def is_pointer(self):
        """
        For complex modalities, we store the data in external files. The columns then
        just contain pointers to these files, instead of the actual data itself.
        """
        return self in [Modality.MOLECULE_3D, Modality.PROTEIN_3D, Modality.IMAGE]


class ColumnAnnotation(BaseModel):
    """
    Annotates columns in the dataset with additional information
    """

    """
    The data modality describes the data type and affects how the data is processed
    """
    modality: Union[str, Modality] = Modality.UNKNOWN

    """
    The protocol describes how the data was generated
    """
    protocol: Optional[str] = None

    """
    User attributes allow for additional meta-data to be stored
    """
    user_attributes: dict = {}

    class Config:
        arbitrary_types_allowed = True

    @validator("modality")
    def validate_modality(cls, v):
        """Tries to converts a string to the Enum"""
        if isinstance(v, str):
            v = Modality[v.upper()]
        return v

    def model_dump(self, *args, **kwargs):
        """Light wrapper to always return the modality as a string, keeping it serializable"""
        data = super().model_dump(*args, **kwargs)

        # The Pydantic dict() API allows a user to exclude keys, so we have to check if `modality` exists.
        if "modality" in data:
            data["modality"] = data["modality"].name
        return data

    def is_pointer(self):
        """
        If a column is a pointer, it means that it's data is stored in an external file
        and that the associated column contains paths to those files.
        """
        return self.modality.is_pointer()
