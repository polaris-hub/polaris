import enum


class Modality(enum.Enum):
    """
    TODO:
      - Add support for 3D protein or molecular structures.
      - Add support for images.
    """

    UNKNOWN = "unknown"
    TARGET = "target"
    MOLECULE = "molecule"
    PROTEIN = "protein"

    def is_target(self):
        return self == Modality.TARGET
