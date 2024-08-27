import enum
from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import DTypeLike
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic.alias_generators import to_camel


class Modality(enum.Enum):
    """Used to Annotate columns in a dataset."""

    UNKNOWN = "unknown"
    MOLECULE = "molecule"
    MOLECULE_3D = "molecule_3D"
    PROTEIN = "protein"
    PROTEIN_3D = "protein_3D"
    IMAGE = "image"


class KnownContentType(enum.Enum):
    """Used to specify column's IANA content type in a dataset."""

    SMILES = "chemical/x-smiles"
    PDB = "chemical/x-pdb"


class ColumnAnnotation(BaseModel):
    """
    The `ColumnAnnotation` class is used to annotate the columns of the [`Dataset`][polaris.dataset.Dataset] object.
    This mostly just stores meta-data and does not affect the logic. The exception is the `is_pointer` attribute.

    Attributes:
        is_pointer: Annotates whether a column is a pointer column. If so, it does not contain data,
            but rather contains references to blobs of data from which the data is loaded.
        modality: The data modality describes the data type and is used to categorize datasets on the hub
            and while it does not affect logic in this library, it does affect the logic of the hub.
        description: Describes how the data was generated.
        user_attributes: Any additional meta-data can be stored in the user attributes.
        content_type: Specify column's IANA content type. If the the content type matches with a known type for
            molecules (e.g. "chemical/x-smiles"), visualization for its content will be activated on the Hub side
    """

    is_pointer: bool = False
    modality: Union[str, Modality] = Modality.UNKNOWN
    description: Optional[str] = None
    user_attributes: Dict[str, str] = Field(default_factory=dict)
    dtype: Union[np.dtype, str, None] = None
    content_type: Union[KnownContentType, str, None] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, alias_generator=to_camel, populate_by_name=True)

    @field_validator("modality")
    def _validate_modality(cls, v, values):
        """Tries to convert a string to the Enum"""
        if isinstance(v, str):
            v = Modality[v.upper()]
        return v

    @field_validator("content_type")
    def _validate_content_type(cls, v, values):
        """Tries to convert a string to the Enum"""
        if isinstance(v, str):
            v = KnownContentType[v.upper()]
        return v

    @field_validator("dtype")
    def _validate_dtype(cls, v):
        """Tries to convert a string to the Enum"""
        if isinstance(v, str):
            v = np.dtype(v)
        return v

    @field_serializer("modality")
    def _serialize_modality(self, v: Modality):
        """Return the modality as a string, keeping it serializable"""
        return v.name

    @field_serializer("content_type")
    def _serialize_content_type(self, v: KnownContentType):
        """Return the content_type as a string, keeping it serializable"""
        if v is not None:
            v = v.name
        return v

    @field_serializer("dtype")
    def _serialize_dtype(self, v: Optional[DTypeLike]):
        """Return the dtype as a string, keeping it serializable"""
        if v is not None:
            v = v.name
        return v
