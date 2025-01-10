import numpy as np
from fastpdb import struc
from numcodecs import MsgPack, register_codec
from numcodecs.vlen import VLenBytes
from rdkit import Chem


class RDKitMolCodec(VLenBytes):
    """
    Codec for RDKit's Molecules.

    Info: Binary strings for serialization
        This class converts the molecules to binary strings (for ML purposes, this should be lossless).
        This might not be the most storage efficient, but is fastest and easiest to maintain.
        See this [Github Discussion](https://github.com/rdkit/rdkit/discussions/7235) for more info.

    """

    codec_id = "rdkit_mol"

    def encode(self, buf: np.ndarray):
        """
        Encode a chunk of RDKit Mols to byte strings
        """
        to_encode = np.empty(shape=len(buf), dtype=object)
        for idx, mol in enumerate(buf):
            if mol is None or (isinstance(mol, bytes) and len(mol) == 0):
                continue
            if not isinstance(mol, Chem.Mol):
                raise ValueError(f"Expected an RDKitMol, but got {type(buf)} instead.")
            props = Chem.PropertyPickleOptions.AllProps
            to_encode[idx] = mol.ToBinary(props)

        to_encode = np.array(to_encode, dtype=object)
        return super().encode(to_encode)

    def decode(self, buf, out=None):
        """Decode the variable length bytes encoded data into a RDKit Mol."""
        dec = super().decode(buf, out)
        for idx, mol in enumerate(dec):
            if len(mol) == 0:
                continue
            dec[idx] = Chem.Mol(mol)

        if out is not None:
            np.copyto(out, dec)
            return out
        else:
            return dec


class AtomArrayCodec(MsgPack):
    """
    Codec for FastPDB (i.e. Biotite) Atom Arrays.

    Info: Only the most essential structural information of a protein is retained
        This conversion saves the 3D coordinates, chain ID, residue ID, insertion code, residue name, heteroatom indicator, atom name, element, atom ID, B-factor, occupancy, and charge.
        Records such as CONECT (connectivity information), ANISOU (anisotropic Temperature Factors), HETATM (heteroatoms and ligands) are handled by `fastpdb`.
        We believe this makes for a good _ML-ready_ format, but let us know if you require any other information to be saved.


    Info: PDBs as ND-arrays using `biotite`
        To save PDBs in a Polaris-compatible format, we convert them to ND-arrays using `fastpdb` and `biotite`.
        We then save these ND-arrays to Zarr archives.
        For more info, see [fastpdb](https://github.com/biotite-dev/fastpdb)
        and [biotite](https://github.com/biotite-dev/biotite/blob/main/src/biotite/structure/atoms.py)

    This codec is a subclass of the `MsgPack` codec from the `numcodecs`
    """

    codec_id = "atom_array"

    def encode(self, buf: np.ndarray):
        """
        Encode a chunk of AtomArrays to a plain Python structure that MsgPack can encode
        """

        to_pack = np.empty_like(buf)

        for idx, atom_array in enumerate(buf):
            # A chunk can have missing values
            if atom_array is None:
                continue

            if not isinstance(atom_array, struc.AtomArray):
                raise ValueError(f"Expected an AtomArray, but got {type(atom_array)} instead")

            data = {
                "coord": atom_array.coord,
                "chain_id": atom_array.chain_id,
                "res_id": atom_array.res_id,
                "ins_code": atom_array.ins_code,
                "res_name": atom_array.res_name,
                "hetero": atom_array.hetero,
                "atom_name": atom_array.atom_name,
                "element": atom_array.element,
                "atom_id": atom_array.atom_id,
                "b_factor": atom_array.b_factor,
                "occupancy": atom_array.occupancy,
                "charge": atom_array.charge,
            }
            data = {k: v.tolist() for k, v in data.items()}
            to_pack[idx] = data

        return super().encode(to_pack)

    def decode(self, buf, out=None):
        """Decode the MsgPack decoded data into a `fastpdb` AtomArray."""

        dec = super().decode(buf, out)

        structs = np.empty(shape=len(dec), dtype=object)

        for idx, data in enumerate(dec):
            if data is None:
                continue

            atom_array = []
            array_length = len(data["coord"])

            for ind in range(array_length):
                atom = struc.Atom(
                    coord=data["coord"][ind],
                    chain_id=data["chain_id"][ind],
                    res_id=data["res_id"][ind],
                    ins_code=data["ins_code"][ind],
                    res_name=data["res_name"][ind],
                    hetero=data["hetero"][ind],
                    atom_name=data["atom_name"][ind],
                    element=data["element"][ind],
                    b_factor=data["b_factor"][ind],
                    occupancy=data["occupancy"][ind],
                    charge=data["charge"][ind],
                    atom_id=data["atom_id"][ind],
                )
                atom_array.append(atom)

            # Note that this is a `fastpdb` AtomArray, not a NumPy array.
            structs[idx] = struc.array(atom_array)

        if out is not None:
            np.copyto(out, structs)
            return out
        else:
            return structs


register_codec(RDKitMolCodec)
register_codec(AtomArrayCodec)
