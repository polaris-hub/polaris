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
        """
        If a modality is a pointer, it means that it's data is stored in an external file
        and that a column with the pointer modality contains paths to those files.
        """
        return self in [Modality.MOLECULE_3D, Modality.PROTEIN_3D, Modality.IMAGE]
