import enum


class Modality(enum.Enum):
    """
    Used to Annotate columns in a dataset.
    """

    UNKNOWN = "unknown"
    TARGET = "target"
    MOLECULE = "molecule"
    MOLECULE_3D = "molecule_3D"
    PROTEIN = "protein"
    PROTEIN_3D = "protein_3D"
    IMAGE = "image"

    def is_target(self):
        return self == Modality.TARGET
