import enum


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
        return self in [Modality.MOLECULE_3D, Modality.PROTEIN_3D, Modality.IMAGE]
